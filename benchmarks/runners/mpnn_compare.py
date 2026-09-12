"""Compare MPNN kernel families using the repository's measured execution path.

Run on a GPU node: python -m benchmarks.runners.mpnn_compare --out results.json
Inference uses manual CUDA graphs; training measures forward/backward without graphs.
Parameters are FP32, activations BF16, CUDA BF16 autocast, width 128, 48 neighbors.
Dropout-bearing training operations use p=0.25. Accuracy probes use p=0 so independent
random masks are not mistaken for arithmetic error. No optimizer step is measured.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from benchmarks.runners.bench import BenchConfig, measured_result
from benchmarks.runners.measurement import (
    benchmark_source_hash,
    compile_for_benchmark,
    require_source_identity,
)

from miniworld_engine.kernels.mpnn_edge_dropout import EdgeDropoutBackend, edge_dropout
from miniworld_engine.kernels.mpnn_edge_layernorm import (
    EdgeNormBackend,
    edge_layer_norm,
)
from miniworld_engine.kernels.mpnn_edge_mlp import EdgeMLPBackend, edge_mlp_update
from miniworld_engine.kernels.mpnn_edge_tail import EdgeTailBackend, edge_tail_update
from miniworld_engine.kernels.mpnn_message import MessageBackend, message_hidden_reduce
from miniworld_engine.kernels.mpnn_node_message import (
    NodeMessageBackend,
    node_message_reduce,
)
from miniworld_engine.kernels.mpnn_node_message.reference import (
    node_message_reduce_pytorch,
)
from miniworld_engine.kernels.mpnn_relative_position import (
    RelativePositionBackend,
    relative_position_embed,
)

BACKENDS = {
    "message": ("pytorch", "triton_compute", "triton_memory"),
    "edge_mlp": ("pytorch", "triton_compute", "triton_memory"),
    "edge_tail": ("pytorch", "triton_compute", "triton"),
    "edge_layernorm": ("pytorch", "memory"),
    "node_message": ("pytorch", "triton", "triton_compute"),
    "relative_position": ("pytorch", "triton", "index_add"),
    "edge_dropout": ("pytorch", "bitpack"),
}


def make_case(family: str, nodes: int, backend: str, training: bool):
    torch.manual_seed(20260912)

    def tensor(*shape, parameter=False, scale=1.0):
        dtype = torch.float32 if parameter else torch.bfloat16
        return (torch.randn(*shape, device="cuda", dtype=dtype) * scale).requires_grad_(training)

    x = tensor(1, nodes, 48, 128, parameter=family == "edge_layernorm")
    q, neighbor = tensor(1, nodes, 128), tensor(1, nodes, 128)
    weights = [tensor(128, 128, parameter=True, scale=128**-0.5) for _ in range(3)]
    biases = [tensor(128, parameter=True, scale=0.1) for _ in range(2)]
    gamma = torch.ones(128, device="cuda", requires_grad=training)
    beta = torch.zeros(128, device="cuda", requires_grad=training)
    index = torch.randint(nodes, (1, nodes, 48), device="cuda")
    query_index = torch.arange(nodes, device="cuda")[None, :, None]
    index[..., :24] = (query_index + torch.arange(-12, 12, device="cuda")).clamp(0, nodes - 1)
    mask = (torch.rand(1, nodes, 48, device="cuda") > 0.2).float()
    bucket = (query_index - index).clamp(-32, 32) + 32
    table, position_bias = tensor(66, 16, parameter=True), tensor(16, parameter=True)
    seed = torch.tensor([117], dtype=torch.int64, device="cuda")
    probability = 0.25 if training and family in {"edge_tail", "edge_dropout"} else 0.0
    leaves_by_family = {
        "message": [x, weights[1], biases[0]],
        "edge_mlp": [x, weights[1], biases[0], weights[2], biases[1]],
        "edge_tail": [x, q, neighbor, *weights, *biases, gamma, beta],
        "edge_layernorm": [x, gamma, beta],
        "node_message": [x, q, neighbor, weights[0], weights[1], biases[0]],
        "relative_position": [table, position_bias],
        "edge_dropout": [x],
    }
    leaves = leaves_by_family[family] if training else []

    def forward(p=probability, impl=backend):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if family == "message":
                return message_hidden_reduce(x, weights[1], biases[0], mask, 48,
                                             backend=cast(MessageBackend, impl))
            if family == "edge_mlp":
                return edge_mlp_update(x, weights[1], biases[0], weights[2], biases[1],
                                       backend=cast(EdgeMLPBackend, impl))
            if family == "edge_layernorm":
                return edge_layer_norm(x, gamma, beta, 1e-5, backend=cast(EdgeNormBackend, impl))
            if family == "edge_dropout":
                return edge_dropout(x, p, training=training, backend=cast(EdgeDropoutBackend, impl))
            if family == "relative_position":
                return relative_position_embed(bucket, table, position_bias,
                                               cast(RelativePositionBackend, "off" if impl == "pytorch" else impl))
            if family == "node_message":
                arguments = (x, q, neighbor, index, weights[0], weights[1], biases[0], mask, 48)
                if impl == "pytorch":
                    return node_message_reduce_pytorch(*arguments)
                return node_message_reduce(*arguments, backend=cast(NodeMessageBackend, impl))
            if family == "edge_tail" and impl != "pytorch":
                return edge_tail_update(x, q, neighbor, index, weights[0], weights[1], biases[0],
                                        weights[2], biases[1], gamma, beta, seed, 1e-5, p,
                                        backend=cast(EdgeTailBackend, impl))
            # Pure PyTorch, including gather, both GELUs, residual and dropout.
            hidden = F.linear(x, weights[0]) + q.unsqueeze(-2)
            hidden = hidden + F.embedding(index, neighbor.reshape(-1, 128))
            hidden = F.gelu(F.linear(F.gelu(hidden), weights[1], biases[0]))
            update = F.linear(hidden, weights[2], biases[1])
            update = F.dropout(update, p=p, training=training)
            return F.layer_norm((x + update).float(), (128,), gamma, beta, 1e-5).to(x.dtype)

    return forward, leaves, probability


def relative_error(actual, reference):
    return float((actual.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-30))


def evaluate(family, nodes, backend, training, repeats, compiled, source, metric):
    if family == "edge_dropout" and not training:
        return {"status": "not_applicable", "reason": "evaluation dropout is the identity", "samples": []}
    if family == "edge_layernorm" and not training and backend != "pytorch":
        return {"status": "not_applicable", "reason": "compressed saves are training-only; inference uses PyTorch", "samples": []}
    forward, leaves, probability = make_case(family, nodes, backend, training)
    guard = contextlib.nullcontext if training else torch.no_grad
    with guard():
        reference = forward(0.0, "pytorch")
        actual = forward(0.0)
        upstream = torch.randn_like(actual) * 0.01
        accuracy = {"forward_relative_l2": relative_error(actual, reference)}
        if training:
            reference_grads = torch.autograd.grad(reference, leaves, upstream)
            actual_grads = torch.autograd.grad(actual, leaves, upstream)
            accuracy["gradient_relative_l2_max"] = max(map(relative_error, actual_grads, reference_grads))
            del reference_grads, actual_grads
        if not torch.isfinite(actual).all():
            raise ValueError("nonfinite eager kernel output")
        del reference, actual
        measured_forward = compile_for_benchmark(forward, fullgraph=True) if compiled else forward

        def step():
            result = measured_forward()
            if training:
                result.backward(upstream)
            return result

        # The shared helper controls timing/compile evidence/capture. Workload
        # dropout belongs to the forward above and is recorded explicitly below.
        conf = BenchConfig(target=f"mpnn_{family}", level="module", compile=compiled,
                           mode="training" if training else "inference", metric=metric,
                           precision="bf16-mixed", cudagraph="disabled" if training or metric == "memory" else "manual")
        samples = []
        for _ in range(repeats):
            require_source_identity(source)
            result = measured_result(conf=conf, func=step, grad_to_none=leaves, params=[],
                                     is_train=training,
                                     input_dtype=("int64" if family == "relative_position" else
                                                  "float32" if family == "edge_layernorm" else "bfloat16"),
                                     parameter_dtype="" if family == "edge_dropout" else "float32",
                                     execution_path=backend, reference="pytorch")
            require_source_identity(source)
            samples.append(result._asdict())
        if training and any(t.grad is None or not torch.isfinite(t.grad).all() for t in leaves):
            raise ValueError("missing or nonfinite gradient")
    limit = 1e-4 if family == "relative_position" else 0.05
    return {"status": "ok", "dropout": probability, "accuracy_dropout": 0.0,
            "accuracy": accuracy, "accuracy_limit": limit,
            "accuracy_pass": all(value <= limit for value in accuracy.values()), "samples": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nodes", type=int, nargs="+", default=[2048])
    parser.add_argument("--families", nargs="+", choices=tuple(BACKENDS), default=list(BACKENDS))
    parser.add_argument("--backends", nargs="+")
    parser.add_argument("--modes", nargs="+", choices=["training", "inference"], default=["inference", "training"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--metric", choices=["time", "memory"], default="time")
    parser.add_argument("--eager", action="store_true", help="explicit uncompiled diagnostic")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists; choose a new result path")
    source = benchmark_source_hash()
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    torch.backends.cuda.matmul.allow_tf32 = True
    output: dict[str, Any] = {
        "runner_hash": runner_hash, "source_hash": source, "device": torch.cuda.get_device_name(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "compile_requested": not args.eager,
        "width": 128, "neighbors": 48, "mask_probability": 0.2,
        "configuration": {**vars(args), "out": str(args.out)}, "rows": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for nodes in args.nodes:
        for family in args.families:
            for mode in args.modes:
                for backend in BACKENDS[family]:
                    if args.backends and backend not in args.backends:
                        continue
                    print(f"START {family} N={nodes} {mode} {backend}", flush=True)
                    # A sweep uses the same Python code object at different shapes
                    # and policies. Clear Dynamo between cases rather than hitting
                    # its per-frame recompile limit and reporting fake regressions.
                    torch.compiler.reset()
                    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != runner_hash:
                        raise RuntimeError("comparison runner changed during the run")
                    started = time.monotonic()
                    row: dict[str, Any] = {"family": family, "nodes": nodes, "mode": mode, "backend": backend}
                    try:
                        row.update(evaluate(family, nodes, backend, mode == "training",
                                            args.repeats, not args.eager, source, args.metric))
                    except Exception as exc:
                        row.update(status="error", error=f"{type(exc).__name__}: {exc}"[:1800])
                    row["elapsed_seconds"] = time.monotonic() - started
                    output["rows"].append(row)
                    args.out.write_text(json.dumps(output, indent=2))
                    print(json.dumps({key: value for key, value in row.items() if key != "samples"}
                                     | {"milliseconds" if args.metric == "time" else "peak_delta_mb": [sample["value"] for sample in row.get("samples", [])]}), flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
    return int(any(row["status"] == "error" or row.get("accuracy_pass") is False for row in output["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
