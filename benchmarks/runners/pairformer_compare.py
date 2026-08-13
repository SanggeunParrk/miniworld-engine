#!/usr/bin/env python3
"""Pairformer (pair-track) timing comparison: pytorch vs cuequivariance vs ours.

Measures **inference** and **training** as two separate cases (never a single
"forward" number):
  * inference = eval + no_grad (the forward-only kernel paths)
  * training  = fwd + bwd (autograd; ours-trimul dispatches to the v6 merged
                training kernel inside the module)

The Pairformer is a pure shell (miniworld_engine.modules.Pairformer): it only
forwards ``implementation`` to its sub-modules, which each dispatch to the
concrete kernel for the running GPU. So this measures the composed effect of
every per-op backend choice under a realistic AF3 block stack.

Two variants:
  * full        = the AF3 block: trimul(out)+trimul(in)+triattn(start)+triattn(end)+transition
  * trimul_only = bidir trimul only: trimul(out)+trimul(in)+transition  (no triangle attention)

Usage (on a CUDA box):
    python benchmarks/runners/pairformer_compare.py --L 256 384 512
    python benchmarks/runners/pairformer_compare.py --variant trimul_only --modes training
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
import triton

from miniworld_engine.modules import ImplementationType, Pairformer, PairformerConfig
from miniworld_engine import settings

DEVICE = torch.device("cuda")
IMPLS = {
    "pytorch": ImplementationType.PYTORCH,
    "cuequiv": ImplementationType.CUEQUIVARIANCE,
    "ours": ImplementationType.MINIWORLD,
}


def apply_stability_workarounds() -> list[str]:
    """Make every backend cudagraph-capturable so all three are timed identically.

    cudagraph is required: the cute (trimul) kernels have large per-launch host
    overhead in eager mode (~10-30 ms/call), so only graph replay reflects GPU
    time. Two capture blockers are worked around here (both orthogonal to the
    Pairformer wiring):
      1. sm_100 only: the transition hand-CUDA b2b kernel hardcodes Hopper
         (sm_90a) cutlass include paths and does not build on Blackwell -> route
         transition through the triton-family path.
      2. at L >= the use_kernels threshold, triangle-attention's LayerNorm routes
         to the autotuned triton kernel; that autotuner runs a synchronizing
         do_bench during capture (only when combined with the fused gate-out
         path), invalidating the stream. Forcing the non-fused gate-out path
         (sigmoid-gate + cuBLAS to_out) avoids it and is faster at these shapes.
    """
    notes: list[str] = []
    cap = torch.cuda.get_device_capability(0)
    if cap[0] >= 10 and settings.current().transition_cuda_b2b:
        settings.configure(transition_cuda_b2b=False)
        notes.append("transition CUDA b2b disabled on sm_100 (build unsupported) -> triton path")
    from miniworld_engine.kernels.bias_only_attention import dispatch as _bod

    _bod.gate_use_fused = lambda *a, **k: False  # noqa: ARG005
    notes.append("triangle-attention gate forced non-fused (cudagraph-capturable, faster here)")
    return notes


def build(impl: ImplementationType, cfg: PairformerConfig) -> Pairformer:
    torch.manual_seed(0)  # identical init across impls (same seed each build)
    # bf16 params: our cute/triton kernels require the weight dtype to match the
    # bf16 activations; also the deployment regime. All three impls use bf16 for
    # a fair comparison.
    model = Pairformer(cfg, implementation=impl).to(DEVICE, dtype=torch.bfloat16)
    return model


def make_inputs(B: int, L: int, d_pair: int):
    # Dense (no padding): mask=None. An all-ones mask is semantically a no-op but
    # routes the trimul cute path away from the sm100 cuequiv-free kernel (which is
    # gated on `mask is None`), so None is required to benchmark the developed
    # sm100/b200 kernels. Also avoids the b200 bidir backward's masked path.
    torch.manual_seed(1)
    pair = torch.randn(B, L, L, d_pair, device=DEVICE, dtype=torch.bfloat16)
    return pair, None


def _capture(step, warmup: int = 10):
    """Warm up (builds lazy submodules + autotunes) on a side stream, then capture."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    return graph


def inference_time(model: Pairformer, pair, mask, cudagraph: bool) -> float:
    model.eval()

    def step():
        with torch.no_grad():
            model(pair, mask)

    if cudagraph:
        graph = _capture(step)
        fn = graph.replay
    else:
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        fn = step
    return float(triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5]))


def training_time(model: Pairformer, pair, mask, cudagraph: bool) -> float:
    model.train()

    def step():
        out = model(pair, mask)
        out.float().sum().backward()

    if cudagraph:
        # Warm up FIRST (builds the lazy v6 training submodule + autotunes), THEN
        # collect params so the lazily-created ones are included, zero their grad
        # buffers (static addresses for capture), and capture fwd+bwd.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(10):
                step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.zeros_like(p)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        fn = graph.replay
    else:
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        fn = step
    return float(triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5]))


TIMERS = {"inference": inference_time, "training": training_time}


@torch.no_grad()
def infer_out(model: Pairformer, pair, mask) -> torch.Tensor:
    model.eval()
    return model(pair, mask).float()


def run_table(mode: str, args, cfg: PairformerConfig) -> None:
    timer = TIMERS[mode]
    print(f"\n### {mode.upper()}  (variant={args.variant})")
    speedup_bases = [k for k in args.impls if k != "ours"] if "ours" in args.impls else []
    hdr = f"{'L':>6} | " + " | ".join(f"{k+' (ms)':>12}" for k in args.impls)
    for b in speedup_bases:
        hdr += f" | {'ours vs ' + b:>14}"
    print(hdr)
    print("-" * len(hdr))

    for L in args.Ls:
        pair, mask = make_inputs(args.B, L, args.d_pair)
        times: dict[str, float] = {}
        ref = None
        for label in args.impls:
            model = build(IMPLS[label], cfg)
            if args.compile:
                # Raise dynamo's recompile ceiling: the manual-autograd + warmup/capture flow
                # churns guards (requires_grad / shape) and hits the default limit (8) -> eager
                # fallback at later shapes. A high limit lets every shape compile cleanly.
                import torch._dynamo as _dyn
                _dyn.config.recompile_limit = 128
                _dyn.config.cache_size_limit = 128
                # default mode (inductor fusion, NO internal cudagraph tree) so it composes
                # with the manual capture below; warmup triggers+finishes compilation.
                model = torch.compile(model)
            if args.check and mode == "inference":
                out = infer_out(model, pair, mask)
                if label == "pytorch":
                    ref = out
                elif ref is not None:
                    cos = torch.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
                    print(f"    [check] L={L} {label} cos_vs_pytorch={cos:.6f}")
            try:
                times[label] = timer(model, pair, mask, args.cudagraph)
            except Exception as e:  # noqa: BLE001
                times[label] = float("nan")
                print(f"    [warn] L={L} {mode} {label} failed: {type(e).__name__}: {str(e)[:100]}")
            del model
            torch.cuda.empty_cache()

        row = f"{L:>6} | " + " | ".join(f"{times[k]:>12.3f}" for k in args.impls)
        ours = times.get("ours", float("nan"))
        for b in speedup_bases:
            spd = times[b] / ours if ours == ours and ours else float("nan")
            row += f" | {spd:>13.2f}x"
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--L", "--seq-len", dest="Ls", type=int, nargs="+", default=[256, 384, 512])
    ap.add_argument("--B", "--batch", dest="B", type=int, default=1)
    ap.add_argument("--d-pair", type=int, default=128)
    ap.add_argument("--n-block", type=int, default=4)
    ap.add_argument("--p-drop", type=float, default=0.25,
                    help="dropout prob in the residual updates (0 = no dropout)")
    ap.add_argument("--variant", choices=["full", "trimul_only", "trimul_dir"], default="full")
    ap.add_argument("--modes", nargs="+", choices=list(TIMERS), default=list(TIMERS))
    ap.add_argument("--impls", nargs="+", default=list(IMPLS), choices=list(IMPLS))
    ap.add_argument("--no-cudagraph", dest="cudagraph", action="store_false")
    ap.add_argument("--compile", action="store_true",
                    help="wrap each model in torch.compile (default inductor mode; our cute/"
                         "triton kernels stay opaque via @torch.compiler.disable, so inductor "
                         "only fuses the shell residual+dropout elementwise glue)")
    ap.add_argument("--check", action="store_true", help="report inference cosine vs pytorch")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    print(f"GPU: {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability(0)}")
    print(f"B={args.B} d_pair={args.d_pair} n_block={args.n_block} "
          f"variant={args.variant} cudagraph={args.cudagraph}")
    if args.cudagraph:
        for note in apply_stability_workarounds():
            print(f"  note: {note}")

    cfg = PairformerConfig(
        d_pair=args.d_pair,
        n_block=args.n_block,
        p_drop=args.p_drop,
        # full = team-gm faithful (two directional trimuls + attention + transition).
        # trimul_only = the developed fused bidirectional trimul + transition, no attention.
        use_triangle_attention=(args.variant == "full"),
        bidirectional_trimul=(args.variant == "trimul_only"),
    )
    for mode in args.modes:
        run_table(mode, args, cfg)
    print("\n(times = median ms over the n_block stack; lower is better)")


if __name__ == "__main__":
    main()
