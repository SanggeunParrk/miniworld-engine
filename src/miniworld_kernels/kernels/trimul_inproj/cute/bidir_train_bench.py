"""Bidirectional trimul TRAINING (fwd+bwd) L-sweep: ours (BidirV6TriMul) vs compiled pytorch.

Correctness: ours (bf16) fwd + grad_x vs fp32 BidirectionalTriangleMultiplication ref (cos>0.99)
at every L. Speed (fwd+bwd ms/layer): ours vs pytorch, BOTH torch.compile (default, event-timed
— HARD RULE: pytorch baseline is COMPILED, no eager). B=1, bf16, d_pair=d_hidden (back K=2·d).
One d_pair per process via argv. COMPUTE NODE only.
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

import copy

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
from miniworld_kernels.modules.triangle_multiplication.module import TriangleMultiplication

EPS = 1e-5
LS = [int(x) for x in os.environ.get("BIDIR_LS", "256,384,512,768,1024").split(",")]
# baselines: dtv1_bidir = a FUSED bidirectional dt-v1 (dt-v1's own kernels, same architecture
# as ours — apples-to-apples). cuequiv_x2 = cuequiv vendor op run for both dirs (can't fuse a
# black-box vendor op). pytorch_bmm = efficient-contraction reference (slow at large L).
COLS = ["pytorch_bmm", "dtv1_bidir", "cuequiv_x2", "ours_bidir"]


class BidirDtV1Fused(nn.Module):
    """Fused bidirectional dt-v1 from a BidirectionalTriangleMultiplication's weights."""

    def __init__(self, base):
        super().__init__()
        self.b = base
        self.h = base.d_hidden

    def forward(self, p):
        b = self.b
        return fused_bidirectional_dtv1(
            p, None,
            norm_in_weight=b.ln_pair.weight, norm_in_bias=b.ln_pair.bias,
            p_in_weight=torch.cat([b.to_left.weight, b.to_right.weight], dim=0),
            g_in_weight=torch.cat([b.to_left_gate.weight, b.to_right_gate.weight], dim=0),
            norm_out_weight=b.ln_out.weight, norm_out_bias=b.ln_out.bias,
            p_out_weight=b.to_out.weight, g_out_weight=b.to_gate.weight, h=self.h, eps=EPS)


class BidirTwo(nn.Module):
    """Bidirectional = outgoing + incoming with an EXISTING single-dir kernel (dtv1/cuequiv),
    both reading the same input (matches our bidirectional semantics: not sequential). Two
    independent single-dir blocks (d_hidden=d) → work-matched to ours-bidir (h=d)."""

    def __init__(self, out_mod, in_mod):
        super().__init__()
        self.out_mod, self.in_mod = out_mod, in_mod

    def forward(self, p):
        return self.out_mod(p) + self.in_mod(p)


class GoodBidirPytorch(nn.Module):
    """Bidirectional trimul reference with the SAME math but an EFFICIENT contraction:
    permute-to-(B,h,L,L) + matmul instead of `torch.einsum`. einsum's autograd materializes
    a huge (B,L,L,L,h) intermediate (the L³ bwd blowup) → the stock reference's backward is
    pathologically slow; this is the fair compiled-pytorch baseline."""

    def __init__(self, base):
        super().__init__()
        self.b = base
        self.h = base.d_hidden

    def forward(self, pair):
        b, h = self.b, self.h
        p = b.ln_pair(pair)
        left = torch.sigmoid(b.to_left_gate(p)) * b.to_left(p)     # (B,L,L,2h)
        right = torch.sigmoid(b.to_right_gate(p)) * b.to_right(p)
        lo, li = left[..., :h].permute(0, 3, 1, 2), left[..., h:].permute(0, 3, 1, 2)
        ro, ri = right[..., :h].permute(0, 3, 1, 2), right[..., h:].permute(0, 3, 1, 2)
        oo = (lo @ ro.transpose(-1, -2)).permute(0, 2, 3, 1)        # outgoing  (B,L,L,h)
        oi = (li.transpose(-1, -2) @ ri).permute(0, 2, 3, 1)        # incoming
        out = b.ln_out(torch.cat([oo, oi], dim=-1))
        return torch.sigmoid(b.to_gate(p)) * b.to_out(out)


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def bench_compiled_train(mod, p, gy):
    comp = torch.compile(mod)
    params = [pr for pr in mod.parameters() if pr.requires_grad]

    def step():
        p.grad = None
        for pr in params:
            pr.grad = None
        comp(p).backward(gy)
    try:
        for _ in range(12):
            step()
        torch.cuda.synchronize()
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(50):
            step()
        ev1.record(); torch.cuda.synchronize()
        return ev0.elapsed_time(ev1) / 50
    except Exception as e:  # noqa: BLE001
        print(f"   compiled-train fail: {type(e).__name__}: {str(e)[:90]}", flush=True)
        return float("nan")


def main():
    assert torch.cuda.is_available()
    D = int(os.environ.get("BIDIR_D", "128"))
    print(f"bidir trimul fwd+bwd d_pair={D} (back K={2*D}) on {torch.cuda.get_device_name(0)}",
          flush=True)
    print("regime: both methods torch.compile (default) fwd+bwd, params require grad, event-timed",
          flush=True)
    _bdll_patch.apply()
    dt = torch.bfloat16

    torch.manual_seed(0)
    base = BidirectionalTriangleMultiplication(
        d_pair=D, d_hidden=D, implementation=ImplementationType.PYTORCH).cuda()
    for lin in (base.to_left, base.to_left_gate, base.to_right, base.to_right_gate,
                base.to_gate, base.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)

    base_fp = copy.deepcopy(base).float()
    ours = BidirV6TriMul(base.to(dt))

    all_ok = True
    for Lc in LS:
        pair = torch.randn(1, Lc, Lc, D, device="cuda")
        dy = torch.randn_like(pair)
        xr = pair.float().clone().requires_grad_(True)
        gxr = torch.autograd.grad(base_fp(xr), xr, dy)[0]
        xo = pair.to(dt).clone().requires_grad_(True)
        yo = ours(xo)
        yo.backward(dy.to(dt))
        fwd_cos, dx_cos = cos(base_fp(xr), yo), cos(gxr, xo.grad)
        ok = min(fwd_cos, dx_cos) > 0.99
        all_ok &= ok
        print(f"  correctness L={Lc}: fwd cos={fwd_cos:.5f}  grad_x cos={dx_cos:.5f} "
              f"-> {'PASS' if ok else 'FAIL'}", flush=True)
        del pair, dy, xr, xo, yo
    print(f"  -> correctness ALL L: {'PASS' if all_ok else 'FAIL'}", flush=True)
    torch.cuda.empty_cache()

    # --- baselines: fused bidir dtv1 (apples-to-apples) + cuequiv ×2 (vendor op, can't fuse) ---
    dtv1_bidir = BidirDtV1Fused(base.to(dt))
    cq_out = TriangleMultiplication(d_pair=D, d_hidden=D, outgoing=True,
                                    implementation=ImplementationType.CUEQUIVARIANCE).cuda().to(dt)
    cq_in = TriangleMultiplication(d_pair=D, d_hidden=D, outgoing=False,
                                   implementation=ImplementationType.CUEQUIVARIANCE).cuda().to(dt)
    cuequiv_x2 = BidirTwo(cq_out, cq_in)

    # correctness of the fused bidir-dtv1 baseline vs fp32 ref (new kernel — verify it)
    pc = torch.randn(1, 512, 512, D, device="cuda")
    dyc = torch.randn_like(pc)
    xrc = pc.float().clone().requires_grad_(True)
    gxc = torch.autograd.grad(base_fp(xrc), xrc, dyc)[0]
    xdc = pc.to(dt).clone().requires_grad_(True)
    ydc = dtv1_bidir(xdc); ydc.backward(dyc.to(dt))
    print(f"  dtv1_bidir L=512 correctness: fwd cos={cos(base_fp(xrc), ydc):.5f} "
          f"grad_x cos={cos(gxc, xdc.grad):.5f}", flush=True)
    del pc, dyc, xrc, xdc, ydc
    torch.cuda.empty_cache()

    mods = {"pytorch_bmm": GoodBidirPytorch(base.to(dt)), "dtv1_bidir": dtv1_bidir,
            "cuequiv_x2": cuequiv_x2, "ours_bidir": ours}
    only = _sys.argv[1] if len(_sys.argv) > 1 else None
    cols = [only] if only else COLS

    rows = {}
    for Lb in LS:
        p = torch.randn(1, Lb, Lb, D, device="cuda", dtype=dt, requires_grad=True)
        g = torch.randn_like(p)
        t = {name: bench_compiled_train(mods[name], p, g) for name in cols}
        rows[Lb] = t
        print(f"[L={Lb}] " + " ".join(f"{c}={t[c]:.3f}" for c in cols), flush=True)
        del p, g
        torch.cuda.empty_cache()

    print(f"\n=== bidir trimul fwd+bwd d_pair={D}, ms/layer (both torch.compile, event-timed) ===")
    full = set(cols) == set(COLS)
    print(f"{'L':>5} | " + " | ".join(f"{c:>14}" for c in cols)
          + (" | vs dtv1 | vs cueq" if full else ""))
    print("-" * 100)
    for Lb in LS:
        r = rows[Lb]
        line = f"{Lb:>5} | " + " | ".join(f"{r[c]:>14.3f}" for c in cols)
        if full:
            line += f" | {r['dtv1_bidir']/r['ours_bidir']:.2f}x | {r['cuequiv_x2']/r['ours_bidir']:.2f}x"
        print(line)
    for c in cols:
        print(f"DATA d{D} {c} " + ",".join(f"{Lb}:{rows[Lb][c]:.4f}" for Lb in LS), flush=True)


if __name__ == "__main__":
    main()
