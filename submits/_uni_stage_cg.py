"""Per-stage batched-vs-loop timing UNDER CUDA GRAPHS — the regime-consistent version of
_uni_stage_prof.py (which was eager and thus penalised the loop with per-launch overhead).

Each stage captured in its own cuda graph (batched = 1 call on B; loop = B calls on B=1),
timed by replay. Now the per-stage sum should reconcile with the end-to-end cudagraph number,
and the stage with ratio>1 (batched slower) is the real culprit. uni, d=H=128. A100.
"""

from __future__ import annotations

import statistics
import time

import torch

from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm
from miniworld_engine.kernels.layernorm_linear.te_style import _te_forward
from miniworld_engine.kernels.trimul_inproj.triton.bidirectional import bidir_front_triton
from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import gate_elem_triton

DEVICE, BF16 = "cuda", torch.bfloat16


def _capture(step, warmup=8):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            with torch.no_grad():
                step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad():
        step()
    return g


def _time(g, n=50):
    for _ in range(8):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.replay()
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
    Wp, Wg = g(d, H), g(d, d)
    M = B * L * L
    xb = [x[b:b + 1].contiguous() for b in range(B)]
    left, right = g(B, H, L, L), g(B, H, L, L)
    view = g(M, H)
    viewb = [g(L * L, H) for _ in range(B)]
    xn2, proj = g(M, d), g(M, d)
    xn2b = [g(L * L, d) for _ in range(B)]
    projb = [g(L * L, d) for _ in range(B)]

    sb = st = 0.0

    def stage(name, bat, loop):
        nonlocal sb, st
        tb, tl = _time(_capture(bat)), _time(_capture(loop))
        sb += tb
        st += tl
        print(f"  {name:14s} batched={tb:8.3f}  loop={tl:8.3f}  ratio={tb / tl:5.2f}"
              f"  {'<-- batched SLOWER' if tb > tl * 1.03 else ''}")

    print(f"== per-stage CUDA-GRAPH (B={B}, L={L}); batched(1x) vs loop({B}x) ==")
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
    print(f"  {'SUM':14s} batched={sb:8.3f}  loop={st:8.3f}  ratio={sb / st:5.2f}")


def main():
    assert torch.cuda.is_available()
    print(f"device={torch.cuda.get_device_name(0)}")
    for L in (256, 384):
        run(B=8, L=L)
    print("STAGE CG DONE")


if __name__ == "__main__":
    main()
