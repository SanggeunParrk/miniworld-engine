"""Unified bidirectional-vs-separate matrix: 5 methods × {inference, fwd+bwd} × D × L.

Methods (all in the pairformer residual framing so sep vs bidir is apples-to-apples):
  ours_bidir / dtv1_bidir : ONE fused bidirectional update in ONE residual block.
  ours_sep / dtv1_sep / cuequiv_sep : the faithful pairformer — TWO sequential single-dir
    residual blocks (incoming sees the outgoing-updated pair) with rowwise dropout.

Regimes:
  infer (BIDIR_MODE=infer): forward only, no_grad, dropout OFF (eval). ours uses its DEDICATED
    inference paths (no backward save). pytorch=N/A; CUDA-graph timed (deployment regime).
  train (BIDIR_MODE=train): fwd+bwd, params require grad, dropout ON (p=0.25), torch.compile,
    event-timed. ours uses the autograd training modules.

B=1, bf16, h=d_pair. cuequiv has no fused-bidir (vendor op) → cuequiv_sep only.
COMPUTE NODE only, fresh QUACK_CACHE_DIR.
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
import triton

from miniworld_kernels.kernels.layernorm.triton.main import triton_layernorm
from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.back_split import trimul_back_split
from miniworld_kernels.kernels.trimul_inproj.cute.bidir_training import BidirV6TriMul
from miniworld_kernels.kernels.trimul_inproj.cute.bidirectional import bidirectional_trimul_ours
from miniworld_kernels.kernels.trimul_inproj.cute.launch import (
    prepack_lr_operand, trimul_inproj_cute_forward,
)
from miniworld_kernels.kernels.trimul_inproj.cute.v6_training import V6TriMul
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
COLS = ["ours_bidir", "dtv1_bidir", "ours_sep", "dtv1_sep", "cuequiv_sep"]


def drop_row(x, training):
    if not training or P_DROP == 0:
        return x
    B, I, _, _ = x.shape
    keep = (torch.rand(B, I, 1, 1, device=x.device) > P_DROP).to(x.dtype) / (1 - P_DROP)
    return x * keep


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


# ---- shared module pieces -------------------------------------------------
class _DtV1Dir(nn.Module):
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


class _DtV1BidirMod(nn.Module):
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


def _pack(mod):
    WL = mod.to_left.weight.t().contiguous()
    WLg = mod.to_left_gate.weight.t().contiguous()
    WR = mod.to_right.weight.t().contiguous()
    WRg = mod.to_right_gate.weight.t().contiguous()
    return dict(WL=WL, WLg=WLg, WR=WR, WRg=WRg, Wg=mod.to_gate.weight.t().contiguous(),
                Wp=mod.to_out.weight.contiguous(), ln_in_w=mod.ln_pair.weight,
                ln_in_b=mod.ln_pair.bias, ln_out_w=mod.ln_out.weight, ln_out_b=mod.ln_out.bias,
                b_lr=prepack_lr_operand(WL, WLg, WR, WRg))


def _single_dir_ours_infer(pair, p, direction, h):
    B, L, _, d = pair.shape
    xn = triton_layernorm(pair.reshape(B * L * L, d), p["ln_in_w"], p["ln_in_b"],
                          EPS).view(B, L, L, d)
    left, right, _ = trimul_inproj_cute_forward(
        xn, p["WL"], p["WLg"], p["WR"], p["WRg"], None, bdll_direct=True, compute_gate=False,
        b_lr=p["b_lr"])
    tri = (torch.einsum("bdik,bdjk->bdij", left, right) if direction == "out"
           else torch.einsum("bdki,bdkj->bdij", left, right))
    return trimul_back_split(tri, xn, p["Wp"], p["Wg"], p["ln_out_w"], p["ln_out_b"], EPS)


# ---- timing ---------------------------------------------------------------
def _bench(fn, warmup=20, rep=80):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def bench_train(mod, p, gy):
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
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(50):
            step()
        e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1) / 50
    except Exception as e:  # noqa: BLE001
        print(f"   train fail: {type(e).__name__}: {str(e)[:80]}", flush=True)
        return float("nan")


def bench_infer(fn, pair):
    try:
        with torch.no_grad():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(10):
                    fn(pair)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn(pair)
        return _bench(g.replay)
    except Exception as e:  # noqa: BLE001
        print(f"   cudagraph fail ({type(e).__name__}: {str(e)[:50]}); eager", flush=True)
        try:
            with torch.no_grad():
                for _ in range(10):
                    fn(pair)
                return _bench(lambda: fn(pair))
        except Exception as e2:  # noqa: BLE001
            print(f"   infer fail: {type(e2).__name__}: {str(e2)[:60]}", flush=True)
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
    MODE = os.environ.get("BIDIR_MODE", "infer")
    print(f"bidir FULL [{MODE}] d_pair={D} h={D} on {torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    dt = torch.bfloat16

    out_base = _seed_single(D, True, 0)
    in_base = _seed_single(D, False, 1)
    torch.manual_seed(2)
    bidir_base = BidirectionalTriangleMultiplication(
        d_pair=D, d_hidden=D, implementation=ImplementationType.PYTORCH).cuda()
    for lin in (bidir_base.to_left, bidir_base.to_left_gate, bidir_base.to_right,
                bidir_base.to_right_gate, bidir_base.to_gate, bidir_base.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)

    cq_out = TriangleMultiplication(d_pair=D, d_hidden=D, outgoing=True,
                                    implementation=ImplementationType.CUEQUIVARIANCE).cuda().to(dt)
    cq_in = TriangleMultiplication(d_pair=D, d_hidden=D, outgoing=False,
                                   implementation=ImplementationType.CUEQUIVARIANCE).cuda().to(dt)

    if MODE == "train":
        mods = {
            "ours_bidir": BidirResidual(BidirV6TriMul(bidir_base.to(dt))),
            "dtv1_bidir": BidirResidual(_DtV1BidirMod(bidir_base.to(dt))),
            "ours_sep": SepResidual(V6TriMul(out_base.to(dt), "out"),
                                    V6TriMul(in_base.to(dt), "in")),
            "dtv1_sep": SepResidual(_DtV1Dir(out_base.to(dt), "outgoing"),
                                    _DtV1Dir(in_base.to(dt), "incoming")),
            "cuequiv_sep": SepResidual(cq_out, cq_in),
        }
        run = lambda name, p, g: bench_train(mods[name], p, g)
    else:
        pout, pin = _pack(out_base.to(dt)), _pack(in_base.to(dt))
        bpk = _pack(bidir_base.to(dt))
        dtv1_b = _DtV1BidirMod(bidir_base.to(dt)).eval()
        dtv1_o = _DtV1Dir(out_base.to(dt), "outgoing").eval()
        dtv1_i = _DtV1Dir(in_base.to(dt), "incoming").eval()
        cq_out.eval(); cq_in.eval()
        h = D

        def ours_bidir_fn(p):
            return p + bidirectional_trimul_ours(
                p, bpk["WL"], bpk["WLg"], bpk["WR"], bpk["WRg"], bpk["Wg"], bpk["Wp"],
                bpk["ln_in_w"], bpk["ln_in_b"], bpk["ln_out_w"], bpk["ln_out_b"], EPS,
                bpk["b_lr"], h)

        def dtv1_bidir_fn(p):
            return p + dtv1_b(p)

        def ours_sep_fn(p):
            p = p + _single_dir_ours_infer(p, pout, "out", h)
            p = p + _single_dir_ours_infer(p, pin, "in", h)
            return p

        def dtv1_sep_fn(p):
            p = p + dtv1_o(p)
            p = p + dtv1_i(p)
            return p

        def cuequiv_sep_fn(p):
            p = p + cq_out(p)
            p = p + cq_in(p)
            return p

        fns = {"ours_bidir": ours_bidir_fn, "dtv1_bidir": dtv1_bidir_fn,
               "ours_sep": ours_sep_fn, "dtv1_sep": dtv1_sep_fn, "cuequiv_sep": cuequiv_sep_fn}
        run = lambda name, p, g: bench_infer(fns[name], p)

    rows = {}
    for L in LS:
        p = torch.randn(1, L, L, D, device="cuda", dtype=dt,
                        requires_grad=(MODE == "train"))
        g = torch.randn_like(p) if MODE == "train" else None
        t = {c: run(c, p, g) for c in COLS}
        rows[L] = t
        print(f"[{MODE} D={D} L={L}] " + " ".join(f"{c}={t[c]:.3f}" for c in COLS), flush=True)
        del p, g
        torch.cuda.empty_cache()

    print(f"\n=== bidir FULL [{MODE}] d_pair={D}, ms/layer ===")
    print(f"{'L':>5} | " + " | ".join(f"{c:>12}" for c in COLS))
    for L in LS:
        r = rows[L]
        print(f"{L:>5} | " + " | ".join(f"{r[c]:>12.3f}" for c in COLS))
    for c in COLS:
        print(f"DATA {MODE} d{D} {c} " + ",".join(f"{L}:{rows[L][c]:.4f}" for L in LS), flush=True)


if __name__ == "__main__":
    main()
