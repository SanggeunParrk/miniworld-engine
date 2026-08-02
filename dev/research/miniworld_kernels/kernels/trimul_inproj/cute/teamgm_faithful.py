"""team-gm faithful harness + ours, measuring EVERY kernel in EVERY regime so the
comparison is apples-to-apples. forward, bf16, H100, single layer + mask, do_bench.

Regimes per kernel:
  eager      : raw, no compile (dt-v1's native best; @torch.compiler.disable'd Fns)
  compile    : model.compile() default mode (cuequiv's best — cudagraph-free Inductor)
  cudagraph  : manual torch.cuda.graph capture (kills launch overhead uniformly)

ours' cute kernels are wrapped in @torch.compiler.disable so torch.compile can
fuse/capture the surrounding glue (einsum/bmm/mask/cast) instead of erroring on
the opaque kernels — exactly the pattern dt-v1 uses for its triton Fns.
COMPUTE NODE only.
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
from lightning import Fabric

from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.launch import (
    prepack_lr_operand, trimul_inproj_cute_forward,
)
from miniworld_kernels.kernels.trimul_inproj.triton.back import trimul_back_triton
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.baseline_dtv1 import (
    fused_triangle_multiplicative_update_dtv1,
)
from miniworld_kernels.modules.triangle_multiplication.module import (
    TriangleMultiplication, _load_cute_fns,
)

DEVICE = torch.device("cuda")
D = 128
EPS = 1e-5
LS = [128, 256, 384, 512, 768, 1024]

# Opaque wrappers: torch.compile graph-breaks here (kernels run eager) but fuses
# the surrounding torch glue. Same idea as dt-v1's @torch.compiler.disable Fns.
_lnt = None


@torch.compiler.disable()
def _d_lnt(x, w, b, eps, layout):
    o = _lnt(x, w, b, eps=eps, layout=layout)
    return o[0] if isinstance(o, tuple) else o


@torch.compiler.disable()
def _d_front(xn, WL, WLg, WR, WRg, b_lr):
    return trimul_inproj_cute_forward(xn, WL, WLg, WR, WRg, None,
                                      bdll_direct=True, compute_gate=False, b_lr=b_lr)


@torch.compiler.disable()
def _d_back(tri, xn, Wp_t, Wg_t, gln_w, gln_b):
    return trimul_back_triton(tri, xn, Wp_t, Wg_t, gln_w, gln_b, EPS)


def bench(func):
    try:
        for _ in range(3):
            func()
        return triton.testing.do_bench(func, warmup=10, rep=100, return_mode="median")
    except Exception as e:  # noqa: BLE001
        print(f"      fail: {type(e).__name__}: {str(e)[:80]}", flush=True)
        return float("nan")


class DTV1Mod(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.ln_in_w, self.ln_in_b = base.ln_pair.weight, base.ln_pair.bias
        self.p_in = nn.Parameter(torch.cat([base.to_left.weight, base.to_right.weight], 0).detach())
        self.g_in = nn.Parameter(torch.cat([base.to_left_gate.weight, base.to_right_gate.weight], 0).detach())
        self.ln_out_w, self.ln_out_b = base.ln_out.weight, base.ln_out.bias
        self.p_out, self.g_out = base.to_out.weight, base.to_gate.weight

    def forward(self, pair, mask):
        m2 = (mask.unsqueeze(-1) & mask.unsqueeze(-2)) if mask is not None else None
        return fused_triangle_multiplicative_update_dtv1(
            pair, "outgoing", m2, self.ln_in_w, self.ln_in_b, self.p_in, self.g_in,
            self.ln_out_w, self.ln_out_b, self.p_out, self.g_out, eps=EPS)


class OursMod(nn.Module):
    def __init__(self, base):
        super().__init__()
        b = base.to(torch.bfloat16)
        self.ln_pair = b.ln_pair
        self.WL, self.WLg = b.to_left.weight.T, b.to_left_gate.weight.T
        self.WR, self.WRg = b.to_right.weight.T, b.to_right_gate.weight.T
        self.Wp_t, self.Wg_t = b.to_out.weight.T, b.to_gate.weight.T
        self.gln_w, self.gln_b = b.ln_out.weight, b.ln_out.bias
        self.b_lr = prepack_lr_operand(self.WL, self.WLg, self.WR, self.WRg)

    def forward(self, pair, mask):
        pair = pair.to(torch.bfloat16)
        b, l1, l2, d = pair.shape
        xn = _d_lnt(pair.reshape(b * l1 * l2, d), self.ln_pair.weight, self.ln_pair.bias,
                    self.ln_pair.eps, "nd->nd").view(b, l1, l2, d)
        if mask is not None:
            xn = xn * (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(-1).to(xn.dtype)
        left, right, _ = _d_front(xn, self.WL, self.WLg, self.WR, self.WRg, self.b_lr)
        tri = torch.einsum("bdik,bdjk->bdij", left, right)
        return _d_back(tri, xn, self.Wp_t, self.Wg_t, self.gln_w, self.gln_b)


def build_base():
    base = TriangleMultiplication(d_pair=D, implementation=ImplementationType.PYTORCH).cuda()
    torch.manual_seed(0)
    for lin in (base.to_left, base.to_left_gate, base.to_right, base.to_right_gate,
                base.to_gate, base.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)
    return base


def make_model(name, base):
    if name == "pytorch":
        m = TriangleMultiplication(d_pair=D, implementation=ImplementationType.PYTORCH).cuda()
        m.load_state_dict(base.state_dict()); return m
    if name == "cuequiv":
        m = TriangleMultiplication(d_pair=D, implementation=ImplementationType.CUEQUIVARIANCE).cuda()
        m.load_state_dict(base.state_dict()); return m
    if name == "dtv1":
        return DTV1Mod(base).cuda()
    if name == "ours":
        return OursMod(base).cuda()
    raise ValueError(name)


def cudagraph_runner(model, pair, mask, autocast):
    """Manual CUDA-graph capture of forward. Returns a replay thunk or raises."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.no_grad():
            for _ in range(3):
                with (torch.autocast("cuda", dtype=torch.bfloat16) if autocast else _null()):
                    model(pair, mask)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        with (torch.autocast("cuda", dtype=torch.bfloat16) if autocast else _null()):
            model(pair, mask)
    return g.replay


import contextlib


def _null():
    return contextlib.nullcontext()


def main():
    global _lnt
    assert torch.cuda.is_available()
    print(f"ours vs NVIDIA — all regimes — on {torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    torch.backends.cuda.matmul.allow_tf32 = True
    _t1, _t2, _f, _lnt = _load_cute_fns()
    fabric = Fabric(precision="bf16-mixed", accelerator="cuda", devices=1)
    fabric.launch()

    # which regimes to run per impl
    plan = {
        "pytorch": ["eager"],
        "cuequiv": ["eager", "compile", "cudagraph"],
        "dtv1":    ["eager", "compile", "cudagraph"],
        "ours":    ["eager", "compile", "cudagraph"],
    }
    cols = []
    for name in ["pytorch", "cuequiv", "dtv1", "ours"]:
        for reg in plan[name]:
            cols.append(f"{name}.{reg}")

    rows = {}
    print(f"\n{'L':>5} | " + " | ".join(f"{c:>14}" for c in cols), flush=True)
    print("-" * (8 + 17 * len(cols)))
    for L in LS:
        rec = {}
        for name in ["pytorch", "cuequiv", "dtv1", "ours"]:
            autocast = (name != "ours")  # ours is bf16-native; others bf16-mixed
            for reg in plan[name]:
                torch._dynamo.reset()
                torch.cuda.empty_cache()
                model = make_model(name, base=build_base())
                pair = torch.randn(1, L, L, D, device=DEVICE)
                mask = torch.rand(1, L, device=DEVICE) > 0.2

                if reg == "compile":
                    model.compile()
                    if name != "ours":
                        model = fabric.setup_module(model)
                elif name != "ours" and reg == "eager":
                    model = fabric.setup_module(model)

                key = f"{name}.{reg}"
                if reg == "cudagraph":
                    try:
                        replay = cudagraph_runner(model, pair, mask, autocast)
                        rec[key] = bench(replay)
                    except Exception as e:  # noqa: BLE001
                        print(f"   {key}@{L} cudagraph fail: {type(e).__name__}: {str(e)[:70]}", flush=True)
                        rec[key] = float("nan")
                else:
                    def fwd():
                        with torch.no_grad():
                            with (torch.autocast("cuda", dtype=torch.bfloat16) if autocast else _null()):
                                model(pair, mask)
                    rec[key] = bench(fwd)
        rows[L] = rec
        print(f"{L:>5} | " + " | ".join(f"{rec.get(c, float('nan')):>14.3f}" for c in cols), flush=True)

    print("\nDATA " + ";".join(f"{L}:" + ",".join(f"{rows[L].get(c, float('nan')):.4f}" for c in cols)
                                for L in LS), flush=True)


if __name__ == "__main__":
    main()
