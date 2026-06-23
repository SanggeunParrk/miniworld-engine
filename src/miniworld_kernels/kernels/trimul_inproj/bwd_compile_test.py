"""B2 — compile-native ours: correctness + full-mode (fwd+bwd) timing vs dt-v1
and cuequiv, eager and torch.compile. bf16, H100, single layer + mask. COMPUTE NODE.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_here = _Path(__file__).resolve().parent  # strip script dir (has a triton/ subpkg)
_sys.path[:] = [p for p in _sys.path if _Path(p).resolve() != _here]
_src_root = _here
while _src_root.name != "src" and _src_root.parent != _src_root:
    _src_root = _src_root.parent
if str(_src_root) not in _sys.path:
    _sys.path.insert(0, str(_src_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.compile_native import TriMulCompile
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.baseline_dtv1 import (
    fused_triangle_multiplicative_update_dtv1,
)
from miniworld_kernels.modules.triangle_multiplication.module import TriangleMultiplication

D = 128
EPS = 1e-5


class DTV1Mod(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.ln_in_w, self.ln_in_b = base.ln_pair.weight, base.ln_pair.bias
        self.p_in = nn.Parameter(torch.cat([base.to_left.weight, base.to_right.weight], 0).detach())
        self.g_in = nn.Parameter(torch.cat([base.to_left_gate.weight, base.to_right_gate.weight], 0).detach())
        self.ln_out_w, self.ln_out_b = base.ln_out.weight, base.ln_out.bias
        self.p_out, self.g_out = nn.Parameter(base.to_out.weight.detach()), nn.Parameter(base.to_gate.weight.detach())

    def forward(self, pair, mask):
        m2 = (mask.unsqueeze(-1) & mask.unsqueeze(-2)) if mask is not None else None
        return fused_triangle_multiplicative_update_dtv1(
            pair, "outgoing", m2, self.ln_in_w, self.ln_in_b, self.p_in, self.g_in,
            self.ln_out_w, self.ln_out_b, self.p_out, self.g_out, eps=EPS)


def ref_forward(x, m, mask2d):
    x_n = F.layer_norm(x, (D,), m.ln_pair.weight, m.ln_pair.bias, EPS)
    if mask2d is not None:
        x_n = x_n * mask2d
    WL, WLg = m.to_left.weight.t(), m.to_left_gate.weight.t()
    WR, WRg = m.to_right.weight.t(), m.to_right_gate.weight.t()
    Wg, Wp = m.to_gate.weight.t(), m.to_out.weight.t()
    left = (x_n @ WL) * torch.sigmoid(x_n @ WLg)
    right = (x_n @ WR) * torch.sigmoid(x_n @ WRg)
    lb, rb = left.permute(0, 3, 1, 2), right.permute(0, 3, 1, 2)
    tri = torch.einsum("bdik,bdjk->bdij", lb, rb).permute(0, 2, 3, 1)
    out_n = F.layer_norm(tri, (D,), m.ln_out.weight, m.ln_out.bias, EPS)
    return (out_n @ Wp) * torch.sigmoid(x_n @ Wg)


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def base_module():
    base = TriangleMultiplication(d_pair=D, implementation=ImplementationType.PYTORCH).cuda()
    torch.manual_seed(0)
    for lin in (base.to_left, base.to_left_gate, base.to_right, base.to_right_gate,
                base.to_gate, base.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)
    return base


def main():
    assert torch.cuda.is_available()
    _bdll_patch.apply()
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"B2 compile-native on {torch.cuda.get_device_name(0)}", flush=True)

    # ---- correctness (bf16 vs fp32 ref) ----
    L = 256
    base = base_module()
    pair = torch.randn(1, L, L, D, device="cuda")
    mask = torch.rand(1, L, device="cuda") > 0.2
    mask2d = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(-1)
    dy = torch.randn_like(pair)

    xr = pair.float().clone().requires_grad_(True)
    yr = ref_forward(xr, base.float(), mask2d.float())
    gxr = torch.autograd.grad(yr, xr, dy.float())[0]

    mod = TriMulCompile(base.to(torch.bfloat16)).cuda()
    xo = pair.to(torch.bfloat16).clone().requires_grad_(True)
    yo = mod(xo, mask)
    yo.backward(dy.to(torch.bfloat16))
    print(f"forward cos={cos(yr, yo):.5f}  grad_x cos={cos(gxr, xo.grad):.5f}", flush=True)

    # ---- full-mode timing ----
    def make(name, b):
        if name == "ours":
            return TriMulCompile(b.to(torch.bfloat16)).cuda()
        if name == "dtv1":
            return DTV1Mod(b).cuda().to(torch.bfloat16)
        if name == "cuequiv":
            m = TriangleMultiplication(d_pair=D, implementation=ImplementationType.CUEQUIVARIANCE).cuda()
            m.load_state_dict(b.state_dict()); return m.to(torch.bfloat16)

    names = ["ours", "dtv1", "cuequiv"]
    print(f"\nfull-mode (fwd+bwd) ms/layer — bf16, single layer + mask", flush=True)
    print(f"{'L':>5} | " + " | ".join(f"{n+' '+r:>14}" for n in names for r in ("eag", "comp")), flush=True)
    for Lb in (256, 512, 1024):
        cells = []
        for name in names:
            for reg in ("eager", "compile"):
                torch._dynamo.reset(); torch.cuda.empty_cache()
                m = make(name, base_module())
                p = torch.randn(1, Lb, Lb, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                msk = torch.rand(1, Lb, device="cuda") > 0.2
                g = torch.randn_like(p)
                fn = torch.compile(m) if reg == "compile" else m

                def step():
                    p.grad = None
                    y = fn(p, msk)
                    y.backward(g)
                try:
                    for _ in range(5):
                        step()
                    t = triton.testing.do_bench(step, warmup=10, rep=50, return_mode="median",
                                                grad_to_none=[p])
                except Exception as e:  # noqa: BLE001
                    print(f"   {name}.{reg}@{Lb} fail: {type(e).__name__}: {str(e)[:60]}", flush=True)
                    t = float("nan")
                cells.append(t)
        print(f"{Lb:>5} | " + " | ".join(f"{c:>14.3f}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
