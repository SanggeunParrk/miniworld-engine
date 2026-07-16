"""Reconcile per-stage vs end-to-end (both CUDA-GRAPH) for the uni triton forward.

9677 showed the 5 COMPUTE stages sum to batched-faster, yet the real forward is batched-
slower — so the gap is in ops my decomposition omitted. Here every op of the real inference
forward is a stage (incl. the MASK MULTIPLY and weight transposes, front with save_preact=False),
so SUM must reconcile with the captured FULL forward. The stage with ratio>1 is the culprit.
uni, d=H=128, B=8. A100.
"""

from __future__ import annotations

import statistics
import time

import torch

from miniworld_kernels.kernels.layernorm.triton.main import triton_layernorm
from miniworld_kernels.kernels.layernorm_linear.te_style import _te_forward
from miniworld_kernels.kernels.trimul_inproj.triton.bidirectional import bidir_front_triton
from miniworld_kernels.kernels.trimul_inproj.triton.gate_elem import gate_elem_triton
from miniworld_kernels.kernels.trimul_inproj.triton.unidirectional import trimul_triton

DEVICE, BF16 = "cuda", torch.bfloat16


def _cap(step, warmup=8):
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


def _t(g, n=50):
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
    r = lambda *s: torch.randn(*s, device=DEVICE, dtype=BF16)  # noqa: E731
    pair = r(B, L, L, d)
    pairb = [pair[b:b + 1].contiguous() for b in range(B)]
    m2d = torch.ones(B, L, L, device=DEVICE, dtype=BF16)
    m2db = [m2d[b:b + 1].contiguous() for b in range(B)]
    lnw, lnb = r(d), r(d)
    WL, WLg, WR, WRg = (r(H, d) for _ in range(4))          # module form (d_hidden, d_pair)
    Wg, Wout = r(d, d), r(d, H)
    lo_w, lo_b = r(H), r(H)

    # pre-transposed weights (as trimul_triton builds internally)
    WLt = WL.t().contiguous()
    WLgt, WRt, WRgt, Wgt = WLg.t().contiguous(), WR.t().contiguous(), WRg.t().contiguous(), Wg.t().contiguous()
    x_n = r(B, L, L, d)
    left, right = r(B, H, L, L), r(B, H, L, L)
    mm = m2d.reshape(B, 1, L, L)
    mmb = [m.reshape(1, 1, L, L) for m in m2db]
    M = B * L * L
    view = r(M, H)
    viewb = [r(L * L, H) for _ in range(B)]
    xn2, proj = r(M, d), r(M, d)
    xn2b, projb = [r(L * L, d) for _ in range(B)], [r(L * L, d) for _ in range(B)]

    sb = st = 0.0

    def stage(name, bat, loop, in_sum=True):
        nonlocal sb, st
        tb, tl = _t(_cap(bat)), _t(_cap(loop))
        if in_sum:
            sb += tb
            st += tl
        flag = "<-- batched SLOWER" if tb > tl * 1.03 else ""
        print(f"  {name:14s} batched={tb:8.3f}  loop={tl:8.3f}  ratio={tb / tl:5.2f}  {flag}")

    print(f"\n== FAITHFUL per-stage CUDA-GRAPH (B={B}, L={L}) ==")
    stage("LN_in",
          lambda: triton_layernorm(pair, lnw, lnb, 1e-5),
          lambda: [triton_layernorm(p, lnw, lnb, 1e-5) for p in pairb])
    stage("wt_transpose",
          lambda: (WL.t().contiguous(), WLg.t().contiguous(), WR.t().contiguous(),
                   WRg.t().contiguous(), Wg.t().contiguous()),
          lambda: [(WL.t().contiguous(), WLg.t().contiguous(), WR.t().contiguous(),
                    WRg.t().contiguous(), Wg.t().contiguous()) for _ in range(B)])
    stage("front(infer)",
          lambda: bidir_front_triton(x_n, WLt, WLgt, WRt, WRgt, save_preact=False),
          lambda: [bidir_front_triton(x_n[b:b + 1].contiguous(), WLt, WLgt, WRt, WRgt,
                                      save_preact=False) for b in range(B)])
    stage("mask_mul",
          lambda: (left * mm, right * mm),
          lambda: [(left[b:b + 1] * mmb[b], right[b:b + 1] * mmb[b]) for b in range(B)])
    stage("contraction",
          lambda: torch.einsum("bhik,bhjk->bijh", left, right),
          lambda: [torch.einsum("bhik,bhjk->bijh", left[b:b + 1], right[b:b + 1]) for b in range(B)])
    stage("te_forward",
          lambda: _te_forward(view, lo_w, lo_b, Wout, None, 1e-5),
          lambda: [_te_forward(v, lo_w, lo_b, Wout, None, 1e-5) for v in viewb])
    stage("gate_elem",
          lambda: gate_elem_triton(xn2, proj, Wgt),
          lambda: [gate_elem_triton(xn2b[b], projb[b], Wgt) for b in range(B)])
    print(f"  {'SUM':14s} batched={sb:8.3f}  loop={st:8.3f}  ratio={sb / st:5.2f}")

    # cross-check: the REAL full forward captured end-to-end (should ~= SUM)
    stage("FULL forward",
          lambda: trimul_triton(pair, WL, WLg, WR, WRg, Wg, Wout, lnw, lnb, lo_w, lo_b,
                                1e-5, 1e-5, H, True, mask=m2d),
          lambda: [trimul_triton(pairb[b], WL, WLg, WR, WRg, Wg, Wout, lnw, lnb, lo_w, lo_b,
                                 1e-5, 1e-5, H, True, mask=m2db[b]) for b in range(B)],
          in_sum=False)


def main():
    assert torch.cuda.is_available()
    print(f"device={torch.cuda.get_device_name(0)}")
    for L in (256, 384):
        run(B=8, L=L)
    print("UNI FULL CG DONE")


if __name__ == "__main__":
    main()
