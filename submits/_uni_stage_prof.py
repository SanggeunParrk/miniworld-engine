"""Per-stage batched-vs-loop timing for the unidirectional triton trimul forward, to
localise WHICH kernel makes the single big (batched) launch lose to B small (loop) launches.

Each stage is timed in isolation: batched = one call on the (B,...) tensor; loop = B calls
on (1,...) slices. ratio = batched / loop_total. ratio > 1 => batched is the slower one there.
d=128, H=d_hidden=128. A100.
"""

from __future__ import annotations

import statistics
import time

import torch

from miniworld_kernels.kernels.layernorm.triton.main import triton_layernorm
from miniworld_kernels.kernels.layernorm_linear.te_style import _te_forward
from miniworld_kernels.kernels.trimul_inproj.triton.bidirectional import bidir_front_triton
from miniworld_kernels.kernels.trimul_inproj.triton.gate_elem import gate_elem_triton

DEVICE, BF16 = "cuda", torch.bfloat16


def _t(fn, n=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def run(B=8, L=384, d=128):
    H = d
    g = lambda *s: torch.randn(*s, device=DEVICE, dtype=BF16)  # noqa: E731
    x = g(B, L, L, d)
    lw, lb = g(d), g(d)
    WL, WLg, WR, WRg = (g(d, H) for _ in range(4))
    lo_w, lo_b = g(H), g(H)
    Wp = g(d, H)
    Wg = g(d, d)
    M = B * L * L

    # pre-make per-stage inputs (batched + per-batch slices)
    xb = [x[b:b + 1].contiguous() for b in range(B)]
    left = g(B, H, L, L)
    right = g(B, H, L, L)
    view = g(M, H)
    viewb = [g(L * L, H) for _ in range(B)]
    xn2 = g(M, d)
    proj = g(M, d)
    xn2b = [g(L * L, d) for _ in range(B)]
    projb = [g(L * L, d) for _ in range(B)]

    def stage(name, bat, loop):
        tb, tl = _t(bat), _t(loop)
        print(f"  {name:14s} batched={tb:8.3f}  loop={tl:8.3f}  ratio={tb / tl:5.2f}"
              f"  {'<-- batched slower' if tb > tl * 1.05 else ''}")

    print(f"== per-stage (B={B}, L={L}, d={d}); ms, batched(1x) vs loop({B}x) ==")
    stage("LN_in",
          lambda: triton_layernorm(x, lw, lb, 1e-5),
          lambda: [triton_layernorm(t, lw, lb, 1e-5) for t in xb])
    stage("front",
          lambda: bidir_front_triton(x, WL, WLg, WR, WRg, save_preact=True),
          lambda: [bidir_front_triton(t, WL, WLg, WR, WRg, save_preact=True) for t in xb])
    stage("contraction",
          lambda: torch.einsum("bhik,bhjk->bijh", left, right),
          lambda: [torch.einsum("bhik,bhjk->bijh", left[b:b + 1], right[b:b + 1]) for b in range(B)])
    stage("te_forward",
          lambda: _te_forward(view, lo_w, lo_b, Wp, None, 1e-5),
          lambda: [_te_forward(v, lo_w, lo_b, Wp, None, 1e-5) for v in viewb])
    stage("gate_elem",
          lambda: gate_elem_triton(xn2, proj, Wg),
          lambda: [gate_elem_triton(xn2b[b], projb[b], Wg) for b in range(B)])


def main():
    assert torch.cuda.is_available()
    print(f"device={torch.cuda.get_device_name(0)}")
    for L in (256, 384):
        run(B=8, L=L)
    print("STAGE PROF DONE")


if __name__ == "__main__":
    main()
