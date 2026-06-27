"""Roofline: how close is ours-bidir to the H100 ceiling, and how much is irreducible?

Reports, per (d_pair, L), for the bidirectional trimul TRAINING block (fwd+bwd):
  - achieved TFLOPS = total_FLOPs / time, and % of H100 bf16 dense peak (989 TFLOPS)
  - the CONTRACTION FLOOR: the 6 batched bmms (2 fwd + 4 bwd) that ANY implementation must
    run — measured in isolation. Its time is a hard lower bound; contraction_time / ours_time
    = the fraction of ours that is irreducible vs glue (front/back GEMMs, LN, gate, launches).

FLOPs (B=1, h=d, M=L²), fwd+bwd = 3× fwd:
  fwd = 22·M·d² (front 16 + output 6 GEMM) + 4·d·L³ (2 contractions)
  → total = 66·L²·d² + 12·d·L³ ;  contraction part = 12·d·L³.
COMPUTE NODE only, fresh QUACK_CACHE_DIR.
"""

from __future__ import annotations

import os
import sys as _sys
from pathlib import Path as _Path

_here = _Path(__file__).resolve().parent
_sys.path[:] = [p for p in _sys.path if _Path(p).resolve() != _here]
_src_root = _here
while _src_root.name != "src" and _src_root.parent != _src_root:
    _src_root = _src_root.parent
if str(_src_root) not in _sys.path:
    _sys.path.insert(0, str(_src_root))

import torch
import torch.nn as nn

from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.bidir_training import BidirV6TriMul
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)

PEAK_TFLOPS = 989.4   # H100 SXM5 bf16 dense (no sparsity)
PEAK_BW_TBs = 3.35    # H100 80GB HBM3
CASES = [(128, 512), (128, 768), (128, 1024), (256, 512), (256, 768), (256, 1024),
         (512, 384), (512, 512)]


def _evtime(step, iters=50, warm=12):
    for _ in range(warm):
        step()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        step()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def contraction_floor_ms(d, L, dt):
    """6 batched bmm (2 fwd + 4 bwd) at (d, L, L) — the irreducible contraction cost."""
    lf = torch.randn(d, L, L, device="cuda", dtype=dt)
    rf = torch.randn(d, L, L, device="cuda", dtype=dt)
    li = torch.randn(d, L, L, device="cuda", dtype=dt)
    ri = torch.randn(d, L, L, device="cuda", dtype=dt)
    g = torch.randn(d, L, L, device="cuda", dtype=dt)

    def step():
        torch.bmm(lf, rf.transpose(1, 2))              # fwd outgoing
        torch.bmm(li.transpose(1, 2), ri)              # fwd incoming
        torch.bmm(g, rf); torch.bmm(g.transpose(1, 2), lf)   # bwd outgoing (2)
        torch.bmm(ri, g.transpose(1, 2)); torch.bmm(li, g)   # bwd incoming (2)
    return _evtime(step)


def ours_ms(d, L, dt):
    torch.manual_seed(0)
    base = BidirectionalTriangleMultiplication(
        d_pair=d, d_hidden=d, implementation=ImplementationType.PYTORCH).cuda()
    for lin in (base.to_left, base.to_left_gate, base.to_right, base.to_right_gate,
                base.to_gate, base.to_out):
        nn.init.normal_(lin.weight, std=d**-0.5)
    mod = torch.compile(BidirV6TriMul(base.to(dt)))
    p = torch.randn(1, L, L, d, device="cuda", dtype=dt, requires_grad=True)
    gy = torch.randn_like(p)
    params = [pr for pr in mod.parameters() if pr.requires_grad]

    def step():
        p.grad = None
        for pr in params:
            pr.grad = None
        mod(p).backward(gy)
    try:
        return _evtime(step)
    except Exception as e:  # noqa: BLE001
        print(f"   ours fail d={d} L={L}: {type(e).__name__}: {str(e)[:80]}", flush=True)
        return float("nan")


def main():
    assert torch.cuda.is_available()
    print(f"roofline on {torch.cuda.get_device_name(0)}  "
          f"(peak {PEAK_TFLOPS} TFLOPS bf16 dense, {PEAK_BW_TBs} TB/s)", flush=True)
    _bdll_patch.apply()
    dt = torch.bfloat16
    rows = []
    for d, L in CASES:
        flops_total = 66 * L * L * d * d + 12 * d * L**3
        flops_contr = 12 * d * L**3
        o = ours_ms(d, L, dt)
        c = contraction_floor_ms(d, L, dt)
        tflops = flops_total / (o * 1e-3) / 1e12
        pct = 100 * tflops / PEAK_TFLOPS
        contr_tflops = flops_contr / (c * 1e-3) / 1e12
        contr_pct = 100 * contr_tflops / PEAK_TFLOPS
        frac = 100 * c / o
        rows.append((d, L, o, tflops, pct, c, contr_tflops, contr_pct, frac))
        print(f"[d={d} L={L}] ours={o:.3f}ms {tflops:.0f}TFLOPS ({pct:.1f}% peak) | "
              f"contr-floor={c:.3f}ms ({contr_tflops:.0f}TFLOPS {contr_pct:.1f}% peak) | "
              f"contr={frac:.0f}% of ours", flush=True)
        torch.cuda.empty_cache()

    print("\n=== ROOFLINE (bidir trimul fwd+bwd) ===")
    print(f"{'d':>4} {'L':>5} | {'ours ms':>8} {'TFLOPS':>7} {'%peak':>6} | "
          f"{'contr ms':>8} {'cTFLOPS':>8} {'c%peak':>7} {'contr/ours':>10}")
    print("-" * 78)
    for d, L, o, tf, pct, c, ctf, cpct, frac in rows:
        print(f"{d:>4} {L:>5} | {o:>8.3f} {tf:>7.0f} {pct:>5.1f}% | "
              f"{c:>8.3f} {ctf:>8.0f} {cpct:>6.1f}% {frac:>9.0f}%")


if __name__ == "__main__":
    main()
