"""Like-for-like frozen-weight Transition forward + input-gradient comparison.

Both sides use C256/H1024, identical BF16 inputs/weights and FP32 LN affine.
No parameter gradients are requested, matching the original ESM design rows.
Run in the isolated pinned ESM environment on an allocated A100.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
from benchmarks.runners.compare_anthropic_a100 import (
    backend_environment,
    elapsed,
    metadata,
)

from miniworld_engine.integrations import anthropic as A
from miniworld_engine.modules import ImplementationType, Transition


def run(length):
    assert torch.cuda.get_device_capability() == (8, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(513)
    provider = A.provider("transition")
    eps = float(provider.carried_module("esm_kd3").C._EPS)
    model = Transition(256, n=4, implementation=ImplementationType.MINIWORLD).cuda().bfloat16().eval()
    model.ln_in.eps = eps
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim > 1:
                p.normal_(std=p.shape[-1] ** -.5)
            elif name.endswith("weight"):
                p.fill_(1)
            else:
                p.zero_()
    model.requires_grad_(False)
    weights = provider.pack(w_a=model.expand_a.weight, w_b=model.expand_b.weight,
                            w_o=model.squeeze.weight, ln_w=model.ln_in.weight, ln_b=model.ln_in.bias, eps=eps)
    x = torch.randn(1, length, length, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    ref_x = x.detach().float().requires_grad_()
    norm = torch.nn.functional.layer_norm(ref_x, (256,), weights.ln_w, weights.ln_b, eps)
    a = torch.nn.functional.linear(norm, weights.w_a.float())
    b = torch.nn.functional.linear(norm, weights.w_b.float())
    reference = ref_x + torch.nn.functional.linear(torch.nn.functional.silu(a)*b, weights.w_o.float())
    reference_dx, = torch.autograd.grad(reference, ref_x, dy.float())
    reference, reference_dx = reference.detach(), reference_dx.detach()
    del ref_x, norm, a, b
    result = {"family": "transition_frozen_dx", "length": length,
              "mode": "forward+input_gradient", "dtype": "bf16", "c": 256, "hidden": 1024, "eps": eps,
              "parameter_gradients": False, "cudagraph": True, "tf32": False,
              "hardware": metadata(), "accuracy": {}, "times_ms": {}, "rounds": 9}
    graphs = {}
    for label in ("miniworld", "esm_kd3", "esm_kd3:lean", "esm_t15_kd3"):
        def step(label=label):
            if label == "miniworld":
                y = model(x)
            else:
                y, _ = A.transition_autograd(x, weights, row=label, n_tokens=length)
            dx, = torch.autograd.grad(y, x, dy)
            return y, dx
        with backend_environment("miniworld" if label == "miniworld" else "anthropic"):
            got, dx = step()
            errors = {"forward_relative_frobenius": ((got.float()-reference).norm()/reference.norm()).item(),
                      "dx_relative_frobenius": ((dx.float()-reference_dx).norm()/reference_dx.norm()).item()}
            result["accuracy"][label] = errors
            assert errors["forward_relative_frobenius"] < .02 and errors["dx_relative_frobenius"] < .03, (label, errors)
            # Release the eager autograd graph's default-stream AccumulateGrad
            # before warming/capturing the same leaf on the capture stream.
            got, dx = got.detach(), dx.detach()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(4):
                    step()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output, output_dx = step()
            graph.replay()
            torch.testing.assert_close(output, got, rtol=0, atol=0)
            torch.testing.assert_close(output_dx, dx, rtol=0, atol=0)
            graphs[label] = graph
            del output, output_dx, got, dx
    samples = {label: [] for label in graphs}
    pilot = {label: elapsed(graph, 3, 1) for label, graph in graphs.items()}
    repeats = {label: max(3, min(100, int(30 / ms))) for label, ms in pilot.items()}
    labels = list(graphs)
    for i in range(9):
        order = labels[i % len(labels):] + labels[:i % len(labels)]
        for label in order:
            samples[label].append(elapsed(graphs[label], repeats[label], 1))
    result["times_ms"] = {label: {"median": statistics.median(v), "min": min(v), "max": max(v), "samples": v} for label, v in samples.items()}
    result["speedup_miniworld"] = {label: statistics.median(b/a for a, b in zip(samples["miniworld"], samples[label], strict=True)) for label in labels[1:]}
    result["status"] = "ok"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.length)
    except Exception as exc:
        result = {"family": "transition_frozen_dx", "length": args.length, "status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: result[k] for k in ("length", "status", "speedup_miniworld", "error") if k in result}), flush=True)
    return result["status"] != "ok"


if __name__ == "__main__":
    raise SystemExit(main())
