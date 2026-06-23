"""Reconcile with team-gm: replicate team-gm's bench harness EXACTLY, then add ours.

team-gm scripts/bench.py::bench_triangle_multiplication does, per seq_len:
  - ONE TriangleMultiplication layer (n_layers=1)
  - pair=(1,L,L,D) requires_grad, mask=rand(1,L)>0.2, dy=randn_like(pair)
  - model.compile() (default mode — NOT reduce-overhead, NOT a K-stack)
  - precision bf16-mixed (autocast) OR fp32; allow_tf32
  - forward()  = model(pair, mask)
  - full()     = y=forward(); backward(y, dy)
  - triton.do_bench(func, warmup=10, rep=100, grad_to_none=[pair]) median

This script runs that harness for pytorch / cuequivariance / dt-v1 in {fp32, bf16}
× {forward, full} to REPRODUCE team-gm's performance.md, and adds ours (bf16
forward; full pending backward). COMPUTE NODE only.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_src_root = _Path(__file__).resolve()
while _src_root.name != "src" and _src_root.parent != _src_root:
    _src_root = _src_root.parent
if str(_src_root) not in _sys.path:
    _sys.path.insert(0, str(_src_root))

import contextlib

import torch
import torch.nn as nn
import triton

from miniworld_kernels.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
    layernorm_linear_cute_fused,
)
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

D = 128
EPS = 1e-5
LS = [128, 256, 384, 512, 768, 1024]


def _bench(func, grad_to_none):
    try:
        return triton.testing.do_bench(func, warmup=10, rep=100, return_mode="median",
                                       grad_to_none=grad_to_none)
    except Exception as e:  # noqa: BLE001
        print(f"      fail: {type(e).__name__}: {str(e)[:90]}", flush=True)
        return float("nan")


class DTV1Mod(nn.Module):
    def __init__(self, base: TriangleMultiplication):
        super().__init__()
        self.ln_in_w = base.ln_pair.weight
        self.ln_in_b = base.ln_pair.bias
        self.p_in = nn.Parameter(torch.cat([base.to_left.weight, base.to_right.weight], 0).detach())
        self.g_in = nn.Parameter(torch.cat([base.to_left_gate.weight, base.to_right_gate.weight], 0).detach())
        self.ln_out_w = base.ln_out.weight
        self.ln_out_b = base.ln_out.bias
        self.p_out = base.to_out.weight
        self.g_out = base.to_gate.weight

    def forward(self, pair, mask):
        mask_2d = (mask.unsqueeze(-1) & mask.unsqueeze(-2)) if mask is not None else None
        return fused_triangle_multiplicative_update_dtv1(
            pair, "outgoing", mask_2d, self.ln_in_w, self.ln_in_b, self.p_in, self.g_in,
            self.ln_out_w, self.ln_out_b, self.p_out, self.g_out, eps=EPS)


class OursMod(nn.Module):
    """Forward-only ours (v4 = triton fused back). bf16 kernels. Mask applied to x_n."""
    def __init__(self, base: TriangleMultiplication, layer_norm_transpose):
        super().__init__()
        self.b = base
        self.lnt = layer_norm_transpose
        self.WL, self.WLg = base.to_left.weight.T, base.to_left_gate.weight.T
        self.WR, self.WRg = base.to_right.weight.T, base.to_right_gate.weight.T
        self.Wg = base.to_gate.weight.T
        self.Wp_t, self.Wg_t = base.to_out.weight.T, base.to_gate.weight.T
        self.gln_w, self.gln_b = base.ln_out.weight, base.ln_out.bias
        self.b_lr = prepack_lr_operand(self.WL, self.WLg, self.WR, self.WRg)

    def forward(self, pair, mask):
        b, l1, l2, d = pair.shape
        o = self.lnt(pair.reshape(b * l1 * l2, d), self.b.ln_pair.weight,
                     self.b.ln_pair.bias, eps=self.b.ln_pair.eps, layout="nd->nd")
        xn = (o[0] if isinstance(o, tuple) else o).view(b, l1, l2, d)
        if mask is not None:
            m2 = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(-1).to(xn.dtype)
            xn = xn * m2
        left, right, _ = trimul_inproj_cute_forward(
            xn, self.WL, self.WLg, self.WR, self.WRg, None,
            bdll_direct=True, compute_gate=False, b_lr=self.b_lr)
        tri = torch.einsum("bdik,bdjk->bdij", left, right)
        return trimul_back_triton(tri, xn, self.Wp_t, self.Wg_t, self.gln_w, self.gln_b, EPS)


def build(name, dtype_mode, layer_norm_transpose):
    """dtype_mode in {'fp32','bf16'}. Returns (module-or-callable, params)."""
    base = TriangleMultiplication(d_pair=D, implementation=ImplementationType.PYTORCH).cuda()
    torch.manual_seed(0)
    for lin in (base.to_left, base.to_left_gate, base.to_right, base.to_right_gate,
                base.to_gate, base.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)

    if name == "pytorch":
        m = TriangleMultiplication(d_pair=D, implementation=ImplementationType.PYTORCH).cuda()
        m.load_state_dict(base.state_dict())
    elif name == "cuequivariance":
        m = TriangleMultiplication(d_pair=D, implementation=ImplementationType.CUEQUIVARIANCE).cuda()
        m.load_state_dict(base.state_dict())
    elif name == "nvidia(dtv1)":
        m = DTV1Mod(base).cuda()
    elif name == "ours_v4":
        m = OursMod(base.cuda(), layer_norm_transpose).cuda()
    else:
        raise ValueError(name)

    if dtype_mode == "bf16" and name == "ours_v4":
        m = m.to(torch.bfloat16)
    elif dtype_mode == "bf16":
        pass  # bf16-mixed: keep fp32 params, autocast at call
    return m


def main():
    assert torch.cuda.is_available()
    print(f"team-gm-faithful trimul bench on {torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    _t1, _t2, _f, layer_norm_transpose = _load_cute_fns()

    configs = [
        ("fp32", "forward", ["pytorch", "cuequivariance", "nvidia(dtv1)"]),
        ("fp32", "full",    ["pytorch", "cuequivariance", "nvidia(dtv1)"]),
        ("bf16", "forward", ["pytorch", "cuequivariance", "nvidia(dtv1)", "ours_v4"]),
        ("bf16", "full",    ["pytorch", "cuequivariance", "nvidia(dtv1)"]),
    ]

    for dtm, mode, names in configs:
        print(f"\n===== {dtm} / {mode} (single layer, mask, model.compile default) =====")
        print(f"{'L':>5} | " + " | ".join(f"{n:>16}" for n in names))
        print("-" * (8 + 19 * len(names)))
        for L in LS:
            row = {}
            for name in names:
                pair_dtype = torch.bfloat16 if (dtm == "bf16" and name == "ours_v4") else torch.float32
                pair = torch.randn(1, L, L, D, device="cuda", dtype=pair_dtype, requires_grad=(mode == "full"))
                dy = torch.randn_like(pair)
                mask = torch.rand(1, L, device="cuda") > 0.2
                m = build(name, dtm, layer_norm_transpose)

                use_autocast = (dtm == "bf16" and name != "ours_v4")
                ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_autocast \
                    else contextlib.nullcontext
                try:
                    cm = torch.compile(m)
                except Exception:  # noqa: BLE001
                    cm = m

                def fwd():
                    with ctx():
                        return cm(pair, mask)

                if mode == "full":
                    params = list(m.parameters()) if hasattr(m, "parameters") else []
                    gtn = [pair] + params

                    def step():
                        pair.grad = None
                        y = fwd()
                        y.backward(dy)
                else:
                    gtn = []

                    def step():
                        with torch.no_grad():
                            fwd()

                # warmup compile
                try:
                    for _ in range(3):
                        step()
                except Exception as e:  # noqa: BLE001
                    print(f"   {name}@{L} warmup fail: {type(e).__name__}: {str(e)[:70]}", flush=True)
                    row[name] = float("nan")
                    continue

                row[name] = _bench(step, gtn)
            print(f"{L:>5} | " + " | ".join(f"{row.get(n, float('nan')):>16.3f}" for n in names), flush=True)


if __name__ == "__main__":
    main()
