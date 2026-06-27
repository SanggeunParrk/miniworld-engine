"""Diagnose: (1) does ONE bidir layer really OOM at d=512 L≥768? (peak mem alone)
(2) WHERE is ours_bidir slower than dtv1_bidir at small L? (eager fwd+bwd kernel profile)."""

from __future__ import annotations

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
from miniworld_kernels.modules.triangle_multiplication.baseline_dtv1_bidir import (
    fused_bidirectional_dtv1,
)
from miniworld_kernels.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)


def make_base(D):
    torch.manual_seed(0)
    base = BidirectionalTriangleMultiplication(
        d_pair=D, d_hidden=D, implementation=ImplementationType.PYTORCH).cuda()
    for lin in (base.to_left, base.to_left_gate, base.to_right, base.to_right_gate,
                base.to_gate, base.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)
    return base


class DtV1Bidir(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.b, self.h = base, base.d_hidden

    def forward(self, p):
        b = self.b
        return fused_bidirectional_dtv1(
            p, None, norm_in_weight=b.ln_pair.weight, norm_in_bias=b.ln_pair.bias,
            p_in_weight=torch.cat([b.to_left.weight, b.to_right.weight], dim=0),
            g_in_weight=torch.cat([b.to_left_gate.weight, b.to_right_gate.weight], dim=0),
            norm_out_weight=b.ln_out.weight, norm_out_bias=b.ln_out.bias,
            p_out_weight=b.to_out.weight, g_out_weight=b.to_gate.weight, h=self.h, eps=1e-5)


def mem_probe(D, L, dt):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = make_base(D)
    mod = BidirV6TriMul(base.to(dt))
    p = torch.randn(1, L, L, D, device="cuda", dtype=dt, requires_grad=True)
    g = torch.randn_like(p)
    try:
        y = mod(p)
        y.backward(g)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        resv = torch.cuda.max_memory_reserved() / 1e9
        print(f"  [mem] ours_bidir TRAIN d={D} L={L}: peak alloc {peak:.1f} GB, "
              f"reserved {resv:.1f} GB  (1 layer, fwd+bwd)", flush=True)
    except torch.cuda.OutOfMemoryError as e:  # noqa
        print(f"  [mem] ours_bidir TRAIN d={D} L={L}: OOM ({str(e)[:60]})", flush=True)
    del base, mod, p, g
    torch.cuda.empty_cache()


def profile(mod, p, g, label):
    for _ in range(5):
        p.grad = None
        for pr in mod.parameters():
            pr.grad = None
        mod(p).backward(g)
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile as tprofile
    with tprofile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(10):
            p.grad = None
            for pr in mod.parameters():
                pr.grad = None
            mod(p).backward(g)
        torch.cuda.synchronize()
    ka = prof.key_averages()
    rows = sorted(ka, key=lambda e: e.self_device_time_total, reverse=True)
    tot = sum(e.self_device_time_total for e in ka) / 1e3 / 10
    print(f"\n=== {label}: total self-CUDA {tot:.3f} ms/iter; top kernels (ms/iter) ===", flush=True)
    for e in rows[:16]:
        t = e.self_device_time_total / 1e3 / 10
        if t < 0.005:
            break
        print(f"  {t:7.3f}  {e.key[:62]}", flush=True)


def main():
    assert torch.cuda.is_available()
    print(f"diag on {torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    dt = torch.bfloat16

    print("\n##### (2) eager fwd+bwd kernel profile: ours_bidir vs dtv1_bidir #####", flush=True)
    for D, L in [(256, 256), (256, 1024)]:
        base = make_base(D)
        ours = BidirV6TriMul(base.to(dt))
        dtv1 = DtV1Bidir(base.to(dt))
        p = torch.randn(1, L, L, D, device="cuda", dtype=dt, requires_grad=True)
        g = torch.randn_like(p)
        profile(ours, p, g, f"ours_bidir d{D} L{L}")
        profile(dtv1, p, g, f"dtv1_bidir d{D} L{L}")
        del base, ours, dtv1, p, g
        torch.cuda.empty_cache()

    print("\n##### (1) single-layer peak memory (does one layer OOM? + d512 L1024 int64 fix) #####",
          flush=True)
    for D, L in [(512, 512), (512, 768), (512, 1024)]:
        mem_probe(D, L, dt)


if __name__ == "__main__":
    main()
