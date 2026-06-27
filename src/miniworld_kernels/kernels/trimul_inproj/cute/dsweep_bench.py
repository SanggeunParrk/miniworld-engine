"""trimul forward D-SWEEP: OURS (v4) vs pytorch vs cuequivariance vs nvidia(dtv1).
B=1, bf16, NO mask, d_pair == d_hidden == D (square). Sweeps the channel dim D at a
couple of representative L; the L-axis sweep lives in compare_bench (D=128 there).

HARD RULE (benchmark/BENCHMARKING.md): ALL benchmarks run compiled, NEVER eager.
  - pytorch baseline = torch.compile(module) (mandated; eager pytorch is launch-
    bound and an unfair baseline) — warmed, steady-state.
  - cuequivariance / nvidia(dtv1) / ours_v4 = manual CUDA-graph (the launch-
    overhead-free regime; torch.compile graph-breaks the cute/triton path and dtv1's
    autograd Fns are @torch.compiler.disable'd, so cudagraph is the fair method for
    them — see memory `compile-vs-cudagraph-for-cute`).
  - NO eager numbers anywhere in the table/graph.

OURS = ours_v4 = triton LN_in -> trimul_inproj(left+right, bdll) -> bmm -> triton
fused back (LN_out + @Wp + gate + mul, one kernel). v5 (cute layernorm_linear) is
OMITTED: its tile configs are N=128-specialized and don't generalize across D.

Each D runs in its OWN process: an illegal-memory-access poisons the whole CUDA
context (every later launch then errors, try/except can't recover). Pass the D
values as argv; the launcher loops `python dsweep_bench.py <D>` once per D.

COMPUTE NODE only. Run with a fresh QUACK_CACHE_DIR.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_src_root = _Path(__file__).resolve()
while _src_root.name != "src" and _src_root.parent != _src_root:
    _src_root = _src_root.parent
if str(_src_root) not in _sys.path:
    _sys.path.insert(0, str(_src_root))

import torch
import torch.nn as nn
import triton

from miniworld_kernels.kernels.layernorm.triton.main import triton_layernorm
from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.back_split import trimul_back_split
from miniworld_kernels.kernels.trimul_inproj.cute.launch import (
    prepack_lr_operand, trimul_inproj_cute_forward,
)
from miniworld_kernels.kernels.trimul_inproj.triton.back import trimul_back_triton
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.baseline_dtv1 import (
    fused_triangle_multiplicative_update_dtv1,
)
from miniworld_kernels.modules.triangle_multiplication.module import TriangleMultiplication

import os as _os

DS = [64, 128, 256, 512]
# L grid: override with DSWEEP_LS=128,256,... for a fine L-sweep at fixed D.
LS = [int(x) for x in _os.environ.get("DSWEEP_LS", "512,1024").split(",")]
# pytorch is COMPILED (HARD RULE); the rest are CUDA-graph'd.
# ours_v4 = single fused back (LN+@Wp+gate+mul, one kernel).
# ours_v6 = SPLIT back (① cute LayerNormLinear + ② triton GateElem).
COLS = ["pytorch", "nvidia(dtv1)", "cuequivariance", "ours_v4", "ours_v6"]


def _bench(fn, *, warmup=25, rep=100):
    try:
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
    except Exception as e:  # noqa: BLE001
        print(f"   bench fail: {type(e).__name__}: {str(e)[:80]}", flush=True)
        return float("nan")


def bench_compiled(fn, pair):
    """torch.compile'd baseline: warm (compile + steady-state) then time. No eager."""
    try:
        for _ in range(10):
            fn(pair)
        torch.cuda.synchronize()
        return _bench(lambda: fn(pair))
    except Exception as e:  # noqa: BLE001
        print(f"   compiled fail: {type(e).__name__}: {str(e)[:80]}", flush=True)
        return float("nan")


def bench_cudagraph(fn, pair):
    """Manual CUDA-graph capture of one layer (launch-overhead-free regime)."""
    try:
        with torch.no_grad():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(5):
                    fn(pair)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn(pair)
        return _bench(g.replay)
    except Exception as e:  # noqa: BLE001
        print(f"   cudagraph fail: {type(e).__name__}: {str(e)[:80]}", flush=True)
        return float("nan")


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def build_for_D(D, dtype):
    """Build method callables for channel dim D, shared weights. pytorch is compiled."""
    pyt = TriangleMultiplication(d_pair=D, implementation=ImplementationType.PYTORCH).cuda().to(dtype)
    cq = TriangleMultiplication(d_pair=D, implementation=ImplementationType.CUEQUIVARIANCE).cuda().to(dtype)
    mod = TriangleMultiplication(d_pair=D, implementation=ImplementationType.CUTE).cuda()
    torch.manual_seed(0)
    for lin in (mod.to_left, mod.to_left_gate, mod.to_right, mod.to_right_gate,
                mod.to_gate, mod.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)
    mod = mod.to(dtype)
    pyt.load_state_dict(mod.state_dict())
    cq.load_state_dict(mod.state_dict())

    WL, WLg = mod.to_left.weight.T, mod.to_left_gate.weight.T
    WR, WRg = mod.to_right.weight.T, mod.to_right_gate.weight.T
    Wp_t, Wg_t = mod.to_out.weight.T, mod.to_gate.weight.T
    gln_w, gln_b, eps = mod.ln_out.weight, mod.ln_out.bias, mod.ln_out.eps
    b_lr = prepack_lr_operand(WL, WLg, WR, WRg)

    dtv1_kw = dict(
        norm_in_weight=mod.ln_pair.weight, norm_in_bias=mod.ln_pair.bias,
        p_in_weight=torch.cat([mod.to_left.weight, mod.to_right.weight], dim=0),
        g_in_weight=torch.cat([mod.to_left_gate.weight, mod.to_right_gate.weight], dim=0),
        norm_out_weight=mod.ln_out.weight, norm_out_bias=mod.ln_out.bias,
        p_out_weight=mod.to_out.weight, g_out_weight=mod.to_gate.weight,
    )

    # HARD RULE: pytorch baseline is COMPILED, never eager.
    pyt_c = torch.compile(pyt, mode="reduce-overhead")

    Wp_nn = mod.to_out.weight  # (N,K) nn.Linear form — cute LayerNormLinear wants this

    def _front(pair):
        b, l1, l2, d = pair.shape
        xf = pair.reshape(b * l1 * l2, d)
        xn = triton_layernorm(xf, mod.ln_pair.weight, mod.ln_pair.bias, eps).view(b, l1, l2, d)
        left, right, _ = trimul_inproj_cute_forward(
            xn, WL, WLg, WR, WRg, None, bdll_direct=True, compute_gate=False, b_lr=b_lr)
        tri = torch.einsum("bdik,bdjk->bdij", left, right)
        return tri, xn

    def ours_v4(pair):  # single fused back
        tri, xn = _front(pair)
        return trimul_back_triton(tri, xn, Wp_t, Wg_t, gln_w, gln_b, eps)

    def ours_v6(pair):  # split back: cute LayerNormLinear + triton GateElem
        tri, xn = _front(pair)
        return trimul_back_split(tri, xn, Wp_nn, Wg_t, gln_w, gln_b, eps)

    variants = {
        "pytorch": lambda p: pyt_c(p),
        "nvidia(dtv1)": lambda p: fused_triangle_multiplicative_update_dtv1(
            p, "outgoing", None, eps=eps, **dtv1_kw),
        "cuequivariance": lambda p: cq(p),
        "ours_v4": ours_v4,
        "ours_v6": ours_v6,
    }
    # ref for cos is the EAGER pytorch module (a correctness oracle, not a timed result).
    ref_fn = lambda p: pyt(p)  # noqa: E731
    return variants, ref_fn


def main():
    assert torch.cuda.is_available()
    ds = [int(a) for a in _sys.argv[1:]] or DS
    print(f"trimul D-sweep on {torch.cuda.get_device_name(0)} | D={ds}", flush=True)
    print("regime: pytorch=torch.compile(reduce-overhead); others=CUDA-graph (no eager)",
          flush=True)
    _bdll_patch.apply()
    dtype = torch.bfloat16

    rows = {}  # keyed by (D, L) -> {col: ms}
    for D in ds:
        variants, ref_fn = build_for_D(D, dtype)
        for L in LS:
            key = (D, L)
            pair = torch.randn(1, L, L, D, device="cuda", dtype=dtype)
            # correctness vs eager pytorch (oracle, not a timed result); per-variant
            # so one variant's compile failure doesn't hide the others.
            with torch.no_grad():
                ref = ref_fn(pair)
            cstr = []
            for name in ("ours_v4", "ours_v6", "nvidia(dtv1)"):
                try:
                    with torch.no_grad():
                        cstr.append(f"{name}={cos(variants[name](pair), ref):.5f}")
                except Exception as e:  # noqa: BLE001
                    cstr.append(f"{name}=FAIL({type(e).__name__})")
            print(f"   [D={D} L={L}] cos vs pytorch: " + " ".join(cstr), flush=True)
            t = {}
            for name in COLS:
                if name == "pytorch":
                    t[name] = bench_compiled(variants[name], pair)
                else:
                    with torch.no_grad():
                        t[name] = bench_cudagraph(variants[name], pair)
            rows[key] = t
            print(f"[D={D} L={L}] " + " ".join(f"{c}={t[c]:.3f}" for c in COLS), flush=True)
            del pair
            torch.cuda.empty_cache()

    print("\n=== ms/layer (pytorch=compiled, others=CUDA-graph) ===")
    print(f"{'D':>5} {'L':>5} | " + " | ".join(f"{c:>14}" for c in COLS))
    print("-" * 98)
    for D in ds:
        for L in LS:
            r = rows[(D, L)]
            print(f"{D:>5} {L:>5} | " + " | ".join(f"{r[c]:>14.3f}" for c in COLS))
    print("DATA " + ";".join(
        f"{D}:{L}=" + ",".join(f"{rows[(D, L)][c]:.4f}" for c in COLS)
        for D in ds for L in LS), flush=True)


if __name__ == "__main__":
    main()
