"""Bidirectional trimul INFERENCE (forward only): ours vs fused dtv1-bidir / cuequiv / pytorch.

HARD RULE: pytorch = torch.compile (no eager); our cute/triton path graph-breaks under compile
→ timed with a manual CUDA graph (the launch-overhead-free deployment regime). dtv1-bidir and
cuequiv also via CUDA graph. ms/layer forward, no_grad. ours uses the DEDICATED inference path
(`bidirectional_trimul_ours` — no backward bookkeeping). B=1, bf16, h=d_pair.
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
import triton

from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.bidirectional import bidirectional_trimul_ours
from miniworld_kernels.kernels.trimul_inproj.cute.launch import prepack_lr_operand
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
COLS = ["pytorch", "dtv1_bidir", "cuequiv_x2", "ours"]


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def _bench(fn, warmup=25, rep=100):
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
                for _ in range(10):
                    fn(pair)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn(pair)
        return _bench(g.replay)
    except Exception as e:  # noqa: BLE001
        print(f"   cudagraph fail ({type(e).__name__}: {str(e)[:60]}); event-timed eager",
              flush=True)
        with torch.no_grad():
            for _ in range(10):
                fn(pair)
            return _bench(lambda: fn(pair))


class GoodBidirPytorch(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.b, self.h = base, base.d_hidden

    def forward(self, pair):
        b, h = self.b, self.h
        p = b.ln_pair(pair)
        left = torch.sigmoid(b.to_left_gate(p)) * b.to_left(p)
        right = torch.sigmoid(b.to_right_gate(p)) * b.to_right(p)
        lo, li = left[..., :h].permute(0, 3, 1, 2), left[..., h:].permute(0, 3, 1, 2)
        ro, ri = right[..., :h].permute(0, 3, 1, 2), right[..., h:].permute(0, 3, 1, 2)
        oo = (lo @ ro.transpose(-1, -2)).permute(0, 2, 3, 1)
        oi = (li.transpose(-1, -2) @ ri).permute(0, 2, 3, 1)
        out = b.ln_out(torch.cat([oo, oi], dim=-1))
        return torch.sigmoid(b.to_gate(p)) * b.to_out(out)


def main():
    assert torch.cuda.is_available()
    D = int(os.environ.get("BIDIR_D", "128"))
    print(f"bidir trimul INFERENCE (fwd) d_pair={D} (back K={2*D}) on "
          f"{torch.cuda.get_device_name(0)}", flush=True)
    print("regime: pytorch=torch.compile; dtv1/cuequiv/ours=CUDA-graph; forward, no_grad",
          flush=True)
    _bdll_patch.apply()
    dt = torch.bfloat16

    torch.manual_seed(0)
    base = BidirectionalTriangleMultiplication(
        d_pair=D, d_hidden=D, implementation=ImplementationType.PYTORCH).cuda()
    for lin in (base.to_left, base.to_left_gate, base.to_right, base.to_right_gate,
                base.to_gate, base.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)
    import copy
    base_fp = copy.deepcopy(base).float().eval()
    bf = base.to(dt)

    # ours inference: pack weights (x@W form) + prepacked b_lr
    WL = bf.to_left.weight.t().contiguous()
    WLg = bf.to_left_gate.weight.t().contiguous()
    WR = bf.to_right.weight.t().contiguous()
    WRg = bf.to_right_gate.weight.t().contiguous()
    Wg = bf.to_gate.weight.t().contiguous()
    Wp = bf.to_out.weight.contiguous()
    b_lr = prepack_lr_operand(WL, WLg, WR, WRg)
    h = D

    def ours_fn(p):
        return bidirectional_trimul_ours(
            p, WL, WLg, WR, WRg, Wg, Wp, bf.ln_pair.weight, bf.ln_pair.bias,
            bf.ln_out.weight, bf.ln_out.bias, EPS, b_lr, h)

    def dtv1_fn(p):
        return fused_bidirectional_dtv1(
            p, None, norm_in_weight=bf.ln_pair.weight, norm_in_bias=bf.ln_pair.bias,
            p_in_weight=torch.cat([bf.to_left.weight, bf.to_right.weight], dim=0),
            g_in_weight=torch.cat([bf.to_left_gate.weight, bf.to_right_gate.weight], dim=0),
            norm_out_weight=bf.ln_out.weight, norm_out_bias=bf.ln_out.bias,
            p_out_weight=bf.to_out.weight, g_out_weight=bf.to_gate.weight, h=h, eps=EPS)

    cq_out = TriangleMultiplication(d_pair=D, d_hidden=D, outgoing=True,
                                    implementation=ImplementationType.CUEQUIVARIANCE).cuda().to(dt).eval()
    cq_in = TriangleMultiplication(d_pair=D, d_hidden=D, outgoing=False,
                                   implementation=ImplementationType.CUEQUIVARIANCE).cuda().to(dt).eval()

    def cuequiv_fn(p):
        return cq_out(p) + cq_in(p)

    pyt = torch.compile(GoodBidirPytorch(bf).eval(), mode="reduce-overhead")

    rows = {}
    for L in LS:
        pair = torch.randn(1, L, L, D, device="cuda", dtype=dt)
        with torch.no_grad():
            ref = base_fp(pair.float())
            c_o = cos(ref, ours_fn(pair))
            c_d = cos(ref, dtv1_fn(pair))
        print(f"  [L={L}] cos vs fp32 ref: ours={c_o:.5f}  dtv1_bidir={c_d:.5f}", flush=True)
        t = {}
        t["pytorch"] = bench_compiled(pyt, pair)
        t["dtv1_bidir"] = bench_cudagraph(dtv1_fn, pair)
        t["cuequiv_x2"] = bench_cudagraph(cuequiv_fn, pair)
        t["ours"] = bench_cudagraph(ours_fn, pair)
        rows[L] = t
        print(f"[L={L}] " + " ".join(f"{c}={t[c]:.3f}" for c in COLS), flush=True)
        del pair
        torch.cuda.empty_cache()

    print(f"\n=== bidir trimul INFERENCE (fwd) d_pair={D}, ms/layer ===")
    print(f"{'L':>5} | " + " | ".join(f"{c:>12}" for c in COLS) + " | vs dtv1 | vs cueq")
    print("-" * 84)
    for L in LS:
        r = rows[L]
        print(f"{L:>5} | " + " | ".join(f"{r[c]:>12.3f}" for c in COLS)
              + f" | {r['dtv1_bidir']/r['ours']:.2f}x | {r['cuequiv_x2']/r['ours']:.2f}x")
    for c in COLS:
        print(f"DATA d{D} {c} " + ",".join(f"{L}:{rows[L][c]:.4f}" for L in LS), flush=True)


if __name__ == "__main__":
    main()
