"""Bidirectional trimul forward: OURS (our fusion) vs PyTorch — same harness.

HARD RULE (BENCHMARKING.md): pytorch baseline = torch.compile (NO eager); ours =
manual CUDA-graph (launch-overhead-free; our cute/triton path graph-breaks under
torch.compile). B=1, bf16, no mask. d_pair=128, d_hidden=128 -> the back runs over
2*d_hidden = 256 channels (the regime the SINGLE fused back can't compile, which is
exactly why bidirectional uses the SPLIT back: cute LayerNormLinear + triton GateElem).

COMPUTE NODE only. Fresh QUACK_CACHE_DIR.
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

from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.bidirectional import bidirectional_trimul_ours
from miniworld_kernels.kernels.trimul_inproj.cute.launch import prepack_lr_operand
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)

LS = [256, 512, 1024]
COLS = ["pytorch", "ours"]
D_PAIR = 128


def _bench(fn, *, warmup=25, rep=100):
    try:
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
    except Exception as e:  # noqa: BLE001
        print(f"   bench fail: {type(e).__name__}: {str(e)[:80]}", flush=True)
        return float("nan")


def bench_compiled(fn, pair):
    try:
        for _ in range(10):
            fn(pair)
        torch.cuda.synchronize()
        return _bench(lambda: fn(pair))
    except Exception as e:  # noqa: BLE001
        print(f"   compiled fail: {type(e).__name__}: {str(e)[:80]}", flush=True)
        return float("nan")


def bench_cudagraph(fn, pair):
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


def main():
    assert torch.cuda.is_available()
    print(f"bidirectional trimul on {torch.cuda.get_device_name(0)} | d_pair={D_PAIR}", flush=True)
    print("regime: pytorch=torch.compile(reduce-overhead); ours=CUDA-graph (no eager)", flush=True)
    _bdll_patch.apply()
    dtype = torch.bfloat16
    h = D_PAIR  # d_hidden = d_pair -> 2h = 256

    mod = BidirectionalTriangleMultiplication(
        d_pair=D_PAIR, implementation=ImplementationType.PYTORCH).cuda()
    torch.manual_seed(0)
    for lin in (mod.to_left, mod.to_left_gate, mod.to_right, mod.to_right_gate,
                mod.to_gate, mod.to_out):
        nn.init.normal_(lin.weight, std=D_PAIR**-0.5)
    mod = mod.to(dtype)

    WL = mod.to_left.weight.T.contiguous()
    WLg = mod.to_left_gate.weight.T.contiguous()
    WR = mod.to_right.weight.T.contiguous()
    WRg = mod.to_right_gate.weight.T.contiguous()
    Wg = mod.to_gate.weight.T.contiguous()
    Wp_nn = mod.to_out.weight.contiguous()                     # (d_pair, 2h) nn.Linear (N,K)
    ln_in_w, ln_in_b = mod.ln_pair.weight, mod.ln_pair.bias
    ln_out_w, ln_out_b = mod.ln_out.weight, mod.ln_out.bias    # 2h-wide
    eps = mod.ln_pair.eps
    b_lr = prepack_lr_operand(WL, WLg, WR, WRg)                # (d_pair, 4*2h)

    pyt_c = torch.compile(mod, mode="reduce-overhead")         # HARD RULE: compiled baseline

    def ours(pair):
        return bidirectional_trimul_ours(pair, WL, WLg, WR, WRg, Wg, Wp_nn,
                                         ln_in_w, ln_in_b, ln_out_w, ln_out_b, eps, b_lr, h)

    rows = {}
    for L in LS:
        pair = torch.randn(1, L, L, D_PAIR, device="cuda", dtype=dtype)
        with torch.no_grad():
            ref = mod(pair)
        try:
            c = cos(ours(pair), ref)
            print(f"   [L={L}] cos(ours, pytorch) = {c:.5f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"   [L={L}] ours FAIL: {type(e).__name__}: {str(e)[:80]}", flush=True)
        t = {"pytorch": bench_compiled(pyt_c, pair)}
        with torch.no_grad():
            t["ours"] = bench_cudagraph(ours, pair)
        rows[L] = t
        print(f"[L={L}] " + " ".join(f"{c}={t[c]:.3f}" for c in COLS), flush=True)
        del pair
        torch.cuda.empty_cache()

    print("\n=== bidirectional fwd, ms/layer (pytorch=compiled, ours=CUDA-graph) ===")
    print(f"{'L':>5} | " + " | ".join(f"{c:>10}" for c in COLS) + " | speedup")
    print("-" * 50)
    for L in LS:
        r = rows[L]
        sp = r["pytorch"] / r["ours"] if r["ours"] == r["ours"] else float("nan")
        print(f"{L:>5} | " + " | ".join(f"{r[c]:>10.3f}" for c in COLS) + f" | {sp:.2f}x")
    print("DATA " + ";".join(
        f"{L}=" + ",".join(f"{rows[L][c]:.4f}" for c in COLS) for L in LS), flush=True)


if __name__ == "__main__":
    main()
