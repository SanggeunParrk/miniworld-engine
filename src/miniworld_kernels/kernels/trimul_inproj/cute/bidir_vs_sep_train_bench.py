"""Pairformer SEPARATE (dtv1 outgoing+incoming, sequential residual + rowwise dropout) vs
BIDIRECTIONAL (one residual block), TRAINING (fwd+bwd).

team-gm pairformer applies trimul as TWO sequential residual blocks with rowwise dropout:

    pair = pair + drop_row(tri_outgoing(pair))
    pair = pair + drop_row(tri_incoming(pair))     # incoming sees the OUTGOING-updated pair

This is a DIFFERENT model from the bidirectional block (both directions from one shared
input, ONE residual) — so this answers the SPEED question only. Compared (all fwd+bwd, train
mode = dropout active, torch.compile, event-timed):

  dtv1_sep    : the faithful pairformer block, both directions = dt-v1 single-dir kernels
  dtv1_bidir  : a fused bidirectional dt-v1 in one residual (baseline_dtv1_bidir)
  ours_bidir  : our fused bidirectional (BidirV6TriMul) in one residual

`fuse↑` = dtv1_sep / bidir = how much one fused bidirectional block beats two separate
dt-v1 residual blocks. B=1, bf16, h = d_pair. COMPUTE NODE only, fresh QUACK_CACHE_DIR.
"""

from __future__ import annotations

import copy
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
from miniworld_kernels.modules.triangle_multiplication.baseline_dtv1 import (
    fused_triangle_multiplicative_update_dtv1,
)
from miniworld_kernels.modules.triangle_multiplication.baseline_dtv1_bidir import (
    fused_bidirectional_dtv1,
)
from miniworld_kernels.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)
from miniworld_kernels.modules.triangle_multiplication.module import TriangleMultiplication

EPS = 1e-5
P_DROP = 0.25
LS = [int(x) for x in os.environ.get("BIDIR_LS", "256,384,512,768,1024").split(",")]
COLS = ["dtv1_sep", "dtv1_bidir", "ours_bidir"]


def drop_row(x, training):
    if not training or P_DROP == 0:
        return x
    B, I, _, _ = x.shape
    keep = (torch.rand(B, I, 1, 1, device=x.device) > P_DROP).to(x.dtype) / (1 - P_DROP)
    return x * keep


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


class _DtV1Dir(nn.Module):
    """One single-direction dt-v1 call from a single-dir module's weights."""

    def __init__(self, base, direction):
        super().__init__()
        self.b, self.dir = base, direction

    def forward(self, p):
        b = self.b
        return fused_triangle_multiplicative_update_dtv1(
            p, self.dir, None, eps=EPS,
            norm_in_weight=b.ln_pair.weight, norm_in_bias=b.ln_pair.bias,
            p_in_weight=torch.cat([b.to_left.weight, b.to_right.weight], dim=0),
            g_in_weight=torch.cat([b.to_left_gate.weight, b.to_right_gate.weight], dim=0),
            norm_out_weight=b.ln_out.weight, norm_out_bias=b.ln_out.bias,
            p_out_weight=b.to_out.weight, g_out_weight=b.to_gate.weight)


class BidirDtV1Fused(nn.Module):
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
            p_out_weight=b.to_out.weight, g_out_weight=b.to_gate.weight, h=self.h, eps=EPS)


class SepResidual(nn.Module):
    """Faithful pairformer: sequential residuals, incoming sees the outgoing-updated pair."""

    def __init__(self, out_mod, in_mod):
        super().__init__()
        self.out_mod, self.in_mod = out_mod, in_mod

    def forward(self, pair):
        pair = pair + drop_row(self.out_mod(pair), self.training)
        pair = pair + drop_row(self.in_mod(pair), self.training)
        return pair


class BidirResidual(nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, pair):
        return pair + drop_row(self.mod(pair), self.training)


def bench_compiled_train(mod, p, gy):
    mod.train()
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
        print(f"   fail: {type(e).__name__}: {str(e)[:90]}", flush=True)
        return float("nan")


def _seed_single(d, outgoing, seed):
    torch.manual_seed(seed)
    m = TriangleMultiplication(d_pair=d, d_hidden=d, outgoing=outgoing,
                               implementation=ImplementationType.PYTORCH).cuda()
    for lin in (m.to_left, m.to_left_gate, m.to_right, m.to_right_gate, m.to_gate, m.to_out):
        nn.init.normal_(lin.weight, std=d**-0.5)
    return m


def main():
    assert torch.cuda.is_available()
    D = int(os.environ.get("BIDIR_D", "128"))
    print(f"bidir-vs-separate TRAINING d_pair={D} h={D} dropout={P_DROP} on "
          f"{torch.cuda.get_device_name(0)}", flush=True)
    print("separate = sequential dt-v1 residuals (incoming sees outgoing-updated pair) + drop_row",
          flush=True)
    print("regime: all torch.compile fwd+bwd, train mode (dropout on), event-timed", flush=True)
    _bdll_patch.apply()
    dt = torch.bfloat16

    # separate path: two single-dir bases (own d→d weights)
    out_base = _seed_single(D, True, 0)
    in_base = _seed_single(D, False, 1)
    # bidir path: one bidirectional base (2h-wide weights)
    torch.manual_seed(2)
    bidir_base = BidirectionalTriangleMultiplication(
        d_pair=D, d_hidden=D, implementation=ImplementationType.PYTORCH).cuda()
    for lin in (bidir_base.to_left, bidir_base.to_left_gate, bidir_base.to_right,
                bidir_base.to_right_gate, bidir_base.to_gate, bidir_base.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)

    # fp32 refs for a light correctness check (eval, dropout off) — WRAP in the SAME residual
    # as the kernels (BidirResidual adds pair), else cos compares (pair+upd) vs upd → ~0.6.
    sep_fp = SepResidual(copy.deepcopy(out_base).float(), copy.deepcopy(in_base).float()).eval()
    bidir_fp = BidirResidual(copy.deepcopy(bidir_base).float()).eval()

    dtv1_sep = SepResidual(_DtV1Dir(out_base.to(dt), "outgoing"),
                           _DtV1Dir(in_base.to(dt), "incoming"))
    dtv1_bidir = BidirResidual(BidirDtV1Fused(bidir_base.to(dt)))
    ours_bidir = BidirResidual(BidirV6TriMul(bidir_base.to(dt)))
    mods = {"dtv1_sep": dtv1_sep, "dtv1_bidir": dtv1_bidir, "ours_bidir": ours_bidir}

    # correctness (eval, dropout off): each path's fwd vs its fp32 reference
    pc = torch.randn(1, 512, 512, D, device="cuda")
    for m in mods.values():
        m.eval()
    with torch.no_grad():
        c_sep = cos(sep_fp(pc.float()), dtv1_sep(pc.to(dt)))
        c_db = cos(bidir_fp(pc.float()), dtv1_bidir(pc.to(dt)))
        c_ob = cos(bidir_fp(pc.float()), ours_bidir(pc.to(dt)))
    print(f"  cos(eval) vs fp32 ref: dtv1_sep={c_sep:.5f} dtv1_bidir={c_db:.5f} "
          f"ours_bidir={c_ob:.5f}", flush=True)
    del pc
    torch.cuda.empty_cache()

    only = _sys.argv[1] if len(_sys.argv) > 1 else None
    cols = [only] if only else COLS
    rows = {}
    for L in LS:
        p = torch.randn(1, L, L, D, device="cuda", dtype=dt, requires_grad=True)
        g = torch.randn_like(p)
        t = {c: bench_compiled_train(mods[c], p, g) for c in cols}
        rows[L] = t
        print(f"[L={L}] " + " ".join(f"{c}={t[c]:.3f}" for c in cols), flush=True)
        del p, g
        torch.cuda.empty_cache()

    full = set(cols) == set(COLS)
    print(f"\n=== separate vs bidir, fwd+bwd ms/layer (train mode); fuse↑ = dtv1_sep/bidir ===")
    print(f"{'L':>5} | " + " | ".join(f"{c:>12}" for c in cols)
          + (" | dtv1_fuse↑ | ours_fuse↑" if full else ""))
    print("-" * 92)
    for L in LS:
        r = rows[L]
        line = f"{L:>5} | " + " | ".join(f"{r[c]:>12.3f}" for c in cols)
        if full:
            line += f" | {r['dtv1_sep']/r['dtv1_bidir']:>9.2f}x | {r['dtv1_sep']/r['ours_bidir']:>9.2f}x"
        print(line)
    for c in cols:
        print(f"DATA d{D} {c} " + ",".join(f"{L}:{rows[L][c]:.4f}" for L in LS), flush=True)


if __name__ == "__main__":
    main()
