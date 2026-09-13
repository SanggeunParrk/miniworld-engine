"""Compare actual MPNN feature/encoder/decoder modules, native BF16 training.

Allocated GPU only. Synthetic B8/L8192 inputs; seven shared-harness samples.
Encoder uses the general nonzero-node layer, not the first zero-node shortcut.
"""
import argparse
import gc
import hashlib
import json
import statistics
from pathlib import Path

import torch
from benchmarks.runners.bench import BenchConfig, measured_result
from benchmarks.runners.measurement import (
    benchmark_source_hash,
    compile_for_benchmark,
    observe_execution,
)
from benchmarks.runners.mpnn_attribution import model_for
from benchmarks.runners.mpnn_training import make_inputs, param_audit


def make_block(block, backend, batch, length, dropout):
    model, _ = model_for("pytorch" if backend == "pytorch" else "all_compute", dropout)
    param_audit(model)
    torch.manual_seed(20260915)
    if block == "features":
        if backend == "memory":
            model.backbone_features.feature_backend = "memory"
        inputs = make_inputs(batch, length)
        params = list(model.backbone_features.parameters()) + list(model.edge_input_projection.parameters())
        def forward():
            graph = model.backbone_features.build_graph(inputs[0], inputs[2], inputs[3], inputs[4])
            edge = model.edge_input_projection(graph.edge_features)
            return torch.where(graph.edge_mask.bool().unsqueeze(-1), edge, torch.zeros_like(edge))
        return forward, params, params
    def state(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16).requires_grad_()
    nodes, edges = state(batch, length, 128), state(batch, length, 48, 128)
    index = torch.randint(length, (batch, length, 48), device="cuda")
    index[..., :24] = (torch.arange(length, device="cuda")[None, :, None] +
                       torch.arange(-12, 12, device="cuda")).clamp(0, length-1)
    mask = torch.ones(batch, length, device="cuda")
    edge_mask = torch.ones(batch, length, 48, device="cuda")
    if block == "encoder":
        layer = model.encoder.layers[1]
        if backend == "memory":
            layer.node_message_backend = "triton"
        params = list(layer.parameters())
        def forward():
            return layer(nodes, edges, index, mask, edge_mask, allow_transition_recompute=False)
        return forward, [nodes, edges, *params], params
    layer = model.decoder.layers[0]
    params = list(layer.parameters())
    sequence, encoder = state(batch, length, 128), state(batch, length, 128)
    past = (torch.rand(batch, length, 48, 1, device="cuda") > .5).float()
    future = 1-past
    def forward():
        return layer(nodes, edges, sequence, encoder, index, future, past, mask,
                     edge_mask, allow_transition_recompute=False)
    return forward, [nodes, edges, sequence, encoder, *params], params


def tensors(output):
    return output if isinstance(output, tuple) else (output,)


def accuracy(block, backend):
    ref, ref_leaves, _ = make_block(block, "pytorch", 2, 128, 0)
    expected = tensors(ref())
    torch.manual_seed(42)
    upstream = tuple(torch.randn_like(x) * .01 for x in expected)
    expected_grad = torch.autograd.grad(expected, ref_leaves, upstream)
    fn, leaves, _ = make_block(block, backend, 2, 128, 0)
    compiled = compile_for_benchmark(fn, fullgraph=True)
    with observe_execution() as evidence:
        actual = tensors(compiled())
        actual_grad = torch.autograd.grad(actual, leaves, upstream)
    assert evidence.compiled
    def relative(a, b):
        a = torch.cat([x.detach().flatten().double() for x in a])
        b = torch.cat([x.detach().flatten().double() for x in b])
        return float((a-b).norm()/b.norm())
    result = {"output_relative_l2": relative(actual, expected),
              "gradient_relative_l2": relative(actual_grad, expected_grad),
              "batch": 2, "length": 128, "dropout": 0, "compiled": evidence.compiled}
    assert result["output_relative_l2"] < .02 and result["gradient_relative_l2"] < .02, result
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--block", choices=["features", "encoder", "decoder"], required=True)
    p.add_argument("--backend", choices=["pytorch", "compute", "memory"], required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    assert not args.out.exists() and "A6000" in torch.cuda.get_device_name()
    assert not torch.is_autocast_enabled("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    source = benchmark_source_hash()
    record = {"status": "running", "block": args.block, "backend": args.backend,
              "source_hash": source, "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "device": torch.cuda.get_device_name(), "device_uuid": str(torch.cuda.get_device_properties(0).uuid),
              "batch": 8, "length": 8192, "neighbors": 48, "dropout": 0 if args.block == "features" else .25,
              "precision": "native_bf16_fp32_norm", "autocast": False, "compile": True,
              "cudagraph": "disabled", "scope": "module forward+all input/parameter backward, no optimizer"}
    def save():
        args.out.write_text(json.dumps(record, indent=2)+"\n")
        print(json.dumps({k: record[k] for k in ["status", "block", "backend", "phase"] if k in record}), flush=True)
    try:
        record.update(phase="accuracy", accuracy=accuracy(args.block, args.backend))
        save()
        torch.compiler.reset()
        gc.collect()
        torch.cuda.empty_cache()
        fn, leaves, params = make_block(args.block, args.backend, 8, 8192, .25)
        compiled = compile_for_benchmark(fn, fullgraph=True)
        output = tensors(compiled())
        torch.manual_seed(42)
        upstream = tuple(torch.randn_like(x) * .01 for x in output)
        del output
        def step():
            output = tensors(compiled())
            torch.autograd.backward(output, upstream)
            return output
        conf = BenchConfig(target="mpnn_actual_module", level="module", mode="training",
                           metric="time", compile=True, precision="bf16", cudagraph="disabled")
        record["phase"] = "timing"
        save()
        samples = []
        for _ in range(7):
            row = measured_result(conf=conf, func=step, grad_to_none=leaves, params=params, is_train=True,
                                  input_dtype="float32" if args.block == "features" else "bfloat16",
                                  parameter_dtype="bfloat16+float32", execution_path=args.backend, reference="pytorch")
            assert row.compiled and row.cudagraph == "disabled"
            samples.append(row._asdict())
        assert all(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.dtype == x.dtype for x in leaves)
        assert benchmark_source_hash() == source
        record.update(status="ok", phase="done", samples=samples,
                      median_ms=statistics.median(x["value"] for x in samples))
        save()
    except Exception as exc:
        record.update(status="error", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    main()
