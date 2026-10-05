"""Paired public-module A100 inference comparison, identical weights and inputs.

Run one case per process on an allocated GPU. Source the Anthropic environment
first. MiniWorld's optional Anthropic TriMul override is disabled for its calls.
Timing alternates order across rounds; each graph contains eight module calls.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import statistics
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch

from miniworld_engine import settings
from miniworld_engine.modules import (
    AdaptiveLayerNorm,
    AttentionPairBias,
    AugmentedAttentionPairBias,
    BidirectionalTriangleMultiplication,
    ConditionedTransition,
    MSAPairWeightedAveraging,
    OuterProductMean,
    Transition,
    TriangleAttention,
    TriangleMultiplication,
)
from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.local_dit import LocalDiTBlock, windows
from miniworld_engine.modules.pairformer.module import PairformerBlock, PairformerConfig
from miniworld_engine.modules.primitives import LayerNorm, RMSNorm
from miniworld_engine.modules.swa_atom_attention import (
    SWA3DRoPEAttention,
    build_attention_params,
)
from miniworld_engine.modules.swa_dit.module import SWADiTBlock, SwiGLUFFN
from miniworld_engine.modules.triangle_attention.bidirectional import (
    BidirectionalTriangleAttention,
)

FAMILIES = (
    "layernorm", "rmsnorm", "transition", "trimul", "trimul_incoming", "trimul_bidir",
    "triattn", "triattn_ending", "triattn_bias_start", "triattn_bias_end",
    "triattn_bidir_bias", "triattn_bidir_sa", "adaln", "conditioned_transition",
    "attention_pair_bias", "augmented_attention", "dit_token", "dit_atom", "dit_atom_local", "bias_only_dit",
    "opm", "opm_post", "pwa", "swiglu_ffn", "swa_attention", "swa_dit",
    "pairformer", "pairformer_bias", "pairformer_bidir", "pairformer_bidir_bias",
)


def fixture(family, length, impl, samples=2):
    kw = {"implementation": impl}
    rand = lambda *s: torch.randn(*s, device="cuda", dtype=torch.bfloat16)
    mask = torch.arange(length, device="cuda")[None] % 7 != 0
    pair = lambda: (rand(1, length, length, 128), mask)
    if family.startswith("pairformer"):
        cfg = PairformerConfig(n_block=1, p_drop=0, use_self_attention="bias" not in family,
                               bidirectional_trimul="bidir" in family)
        block = PairformerBlock(cfg, **kw)
        if impl == "anthropic":
            # Use the release fast tier, matching the standalone comparison.
            for module in block.modules():
                if isinstance(module, TriangleAttention) and module.use_self_attention:
                    module.anthropic_row = "block:fast"
        return block, pair()
    if family.startswith("trimul"):
        if family == "trimul_bidir":
            return BidirectionalTriangleMultiplication(128, p_drop=0, **kw), pair()
        return TriangleMultiplication(128, outgoing=family != "trimul_incoming", p_drop=0, **kw), pair()
    if family in {"layernorm", "rmsnorm"}:
        return (LayerNorm if family == "layernorm" else RMSNorm)(128, **kw), (rand(1, length, 128),)
    if family == "transition":
        return Transition(128, **kw), (rand(1, length, length, 128),)
    if family.startswith("triattn_bidir"):
        return BidirectionalTriangleAttention(128, 4, use_self_attention=family.endswith("sa"), **kw), pair()
    if family.startswith("triattn"):
        return TriangleAttention(128, 4, use_self_attention="bias" not in family,
                                 starting=family not in {"triattn_ending", "triattn_bias_end"}, p_drop=0,
                                 anthropic_row="block:fast", **kw), pair()
    if family in {"adaln", "conditioned_transition"}:
        cls = AdaptiveLayerNorm if family == "adaln" else ConditionedTransition
        return cls(128, 128, **kw), (rand(samples, 1, length, 128), rand(1, 1, length, 128))
    if family == "attention_pair_bias":
        return AttentionPairBias(384, 128, 8, **kw), (rand(1, length, 384), *pair())
    if family == "dit_atom_local":
        return LocalDiTBlock(cross_attention=True, **kw), (
            rand(samples, 1, length, 128), rand(1, 1, length, 128),
            rand(1, windows(length), 32, 128, 16), mask)
    if family in {"augmented_attention", "dit_token", "dit_atom", "bias_only_dit"}:
        d, c, p, h = (128, 128, 16, 4) if family == "dit_atom" else (768, 384, 128, 16)
        cls = (BiasOnlyDiTBlock if family == "bias_only_dit" else
               AugmentedAttentionPairBias if family == "augmented_attention" else DiTBlock)
        return cls(d, c, p, h, **kw), (rand(samples, 1, length, d), rand(1, 1, length, c), rand(1, length, length, p), mask)
    if family.startswith("opm"):
        post = family == "opm_post"
        return OuterProductMean(64, 256 if post else 128, normalize_before_proj=not post, **kw), (
            rand(1, 64, length, 64), mask[:, None].expand(1, 64, length))
    if family == "pwa":
        return MSAPairWeightedAveraging(64, 128, 8, 32, **kw), (rand(1, 64, length, 64), *pair())
    if family == "swiglu_ffn":
        return SwiGLUFFN(128, **kw), (rand(2, length, 128),)
    if family.startswith("swa_"):
        angles = torch.randn(1, length, 16, device="cuda")
        valid = torch.ones(2, length, device="cuda", dtype=torch.bool)
        ap = build_attention_params(angles.cos(), angles.sin(), valid, 2)
        if family == "swa_attention":
            return SWA3DRoPEAttention(128, 4, **kw), (rand(2, length, 128), ap)
        return SWADiTBlock(**kw), (rand(2, length, 128), rand(2, length, 128), ap)
    raise ValueError(family)


@contextlib.contextmanager
def backend_environment(implementation):
    from miniworld_engine.integrations.anthropic import execution_policy
    saved = os.environ.get("TRIMUL_NATIVE_BUILD_DIR")
    if implementation != "anthropic":
        os.environ.pop("TRIMUL_NATIVE_BUILD_DIR", None)
    try:
        with execution_policy(timing="graph"):
            yield
    finally:
        if saved is not None:
            os.environ["TRIMUL_NATIVE_BUILD_DIR"] = saved


def capture(model, inputs, implementation, batch):
    with backend_environment(implementation):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(8):
                model(*inputs)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(batch):
                output = model(*inputs)
        graph.replay()
        torch.cuda.synchronize()
        return graph, output


def elapsed(graph, repeats, batch):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / (repeats * batch)


def metadata():
    return {"gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "node": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "settings": dataclasses.asdict(settings.current()),
            "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,clocks.sm,power.limit,temperature.gpu", "--format=csv,noheader"], text=True).strip()}


@torch.no_grad()
def run(args):
    assert torch.cuda.get_device_capability() == (8, 0), "A100 only"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(191)
    samples = getattr(args, "samples", 2)
    model, inputs = fixture(args.family, args.length, "miniworld", samples)
    model = model.cuda().bfloat16().eval()
    for name, p in model.named_parameters():
        if p.ndim > 1:
            p.normal_(std=p.shape[-1] ** -.5)
        elif name.endswith("weight"):
            p.normal_(mean=1, std=.1)
        else:
            p.normal_(std=.1)
    models = {"miniworld": model}
    for impl in ("anthropic", "pytorch"):
        other, unused = fixture(args.family, args.length, impl, samples)
        del unused
        other = other.cuda().eval()
        other = other.float() if impl == "pytorch" else other.bfloat16()
        other.load_state_dict(model.state_dict())
        models[impl] = other
    reference_inputs = tuple(t.float() if isinstance(t, torch.Tensor) and t.is_floating_point() else t for t in inputs)
    with backend_environment("pytorch"), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        reference = models.pop("pytorch")(*reference_inputs).float()
    del reference_inputs
    result = {"family": args.family, "length": args.length, "mode": "inference", "dtype": "bf16",
              "tf32": False, "compiled": False, "cudagraph": True, "seed": 191,
              "input_shapes": [list(t.shape) if isinstance(t, torch.Tensor) else str(type(t)) for t in inputs],
              "mask": "every seventh key masked; SWA uses two fully valid sequences", "graph_batch": args.graph_batch,
              "rounds": args.rounds, "miniworld_anthropic_override": False, "hardware": metadata(),
              "anthropic_policy": "release fast tier for registered APB geometries and TriangleAttention; graph selection column; explicit compositions elsewhere",
              "accuracy": {}, "selection": {}, "gpu_kernel_names": {}, "times_ms": {}}
    graphs = {}
    for impl, current in models.items():
        with backend_environment(impl):
            got = current(*inputs)
        rel = ((got.float()-reference).norm()/reference.norm().clamp_min(1e-8)).item()
        result["accuracy"][impl] = {"relative_frobenius": rel, "finite": bool(torch.isfinite(got).all())}
        residual = (args.family.startswith(("trimul", "triattn", "pairformer", "dit_")) or
                    args.family in {"transition", "conditioned_transition", "attention_pair_bias", "augmented_attention", "bias_only_dit", "pwa"})
        if residual:
            delta_rel = ((got.float()-reference).norm()/(reference-inputs[0].float()).norm().clamp_min(1e-8)).item()
            result["accuracy"][impl]["update_relative_frobenius"] = delta_rel
            if delta_rel >= .08:
                raise RuntimeError(f"{impl} fails update accuracy: {result['accuracy'][impl]}")
        if not result["accuracy"][impl]["finite"] or rel >= .04:
            raise RuntimeError(f"{impl} fails accuracy: {result['accuracy'][impl]}")
        selected = {f"{name}.{attribute}": str(getattr(m, attribute))
                    for name, m in current.named_modules()
                    for attribute in ("anthropic_selection", "native_selection") if hasattr(m, attribute)}
        if impl == "miniworld":
            assert not selected, f"MiniWorld unexpectedly executed Anthropic: {selected}"
        else:
            assert selected, "No executed Anthropic selection"
        result["selection"][impl] = selected
        graph, output = capture(current, inputs, impl, args.graph_batch)
        torch.testing.assert_close(output, got, rtol=0, atol=0)
        graphs[impl] = graph
        # Record actually executed GPU kernels outside the timing window.
        with backend_environment(impl), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
            current(*inputs)
            torch.cuda.synchronize()
        result["gpu_kernel_names"][impl] = sorted({event.name for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA})
    pilot = {impl: elapsed(graph, 10, args.graph_batch) for impl, graph in graphs.items()}
    repeats = {impl: max(10, min(2000, int(30 / (ms * args.graph_batch)))) for impl, ms in pilot.items()}
    samples = {impl: [] for impl in graphs}
    for round_number in range(args.rounds):
        order = list(graphs) if round_number % 2 == 0 else list(reversed(graphs))
        for impl in order:
            samples[impl].append(elapsed(graphs[impl], repeats[impl], args.graph_batch))
    for impl, values in samples.items():
        result["times_ms"][impl] = {"median": statistics.median(values), "min": min(values), "max": max(values), "samples": values}
    ratios = [b/a for a, b in zip(samples["miniworld"], samples["anthropic"], strict=True)]
    result["speedup_miniworld"] = statistics.median(ratios)
    result["paired_ratio_min_max"] = [min(ratios), max(ratios)]
    result["status"] = "ok"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--graph-batch", type=int, default=8)
    parser.add_argument("--samples", type=int, default=2, help="Augmentation count for conditioned/token modules")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        result = {"family": args.family, "length": args.length, "status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: result[k] for k in ("family", "length", "status", "speedup_miniworld", "error") if k in result}), flush=True)
    return result["status"] != "ok"


if __name__ == "__main__":
    raise SystemExit(main())
