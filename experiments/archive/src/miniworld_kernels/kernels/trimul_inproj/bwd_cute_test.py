"""B1 check: TriMulCute (cute fwd + manual bwd) grads vs fp32 torch-autograd
reference (bf16 tolerance), + full-mode (fwd+bwd) timing. COMPUTE NODE only.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Running by path puts this file's dir on sys.path[0]; it contains a `triton/`
# subpackage that shadows the real triton (-> "No module named triton.language").
# Drop the script dir, add the src root instead.
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
from miniworld_kernels.kernels.trimul_inproj.autograd import _ln_fwd
from miniworld_kernels.kernels.trimul_inproj.autograd_cute import TriMulCute
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.module import TriangleMultiplication

D = 128
EPS = 1e-5


def ref_forward(x, m, mask2d):
    """fp32 torch reference matching TriMulCute math."""
    x_n, *_ = _ln_fwd(x, m.ln_pair.weight, m.ln_pair.bias, EPS)
    if mask2d is not None:
        x_n = x_n * mask2d
    WL, WLg = m.to_left.weight.t(), m.to_left_gate.weight.t()
    WR, WRg = m.to_right.weight.t(), m.to_right_gate.weight.t()
    Wg, Wp = m.to_gate.weight.t(), m.to_out.weight.t()
    left = (x_n @ WL) * torch.sigmoid(x_n @ WLg)
    right = (x_n @ WR) * torch.sigmoid(x_n @ WRg)
    lb, rb = left.permute(0, 3, 1, 2), right.permute(0, 3, 1, 2)
    tri = torch.einsum("bdik,bdjk->bdij", lb, rb).permute(0, 2, 3, 1)
    out_n, *_ = _ln_fwd(tri, m.ln_out.weight, m.ln_out.bias, EPS)
    return (out_n @ Wp) * torch.sigmoid(x_n @ Wg)


def main():
    assert torch.cuda.is_available()
    _bdll_patch.apply()
    print(f"B1 on {torch.cuda.get_device_name(0)}", flush=True)
    torch.manual_seed(0)
    L = 256

    base = TriangleMultiplication(d_pair=D, implementation=ImplementationType.PYTORCH).cuda()
    for lin in (base.to_left, base.to_left_gate, base.to_right, base.to_right_gate,
                base.to_gate, base.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)

    pair = torch.randn(1, L, L, D, device="cuda")
    mask = torch.rand(1, L, device="cuda") > 0.2
    mask2d = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(-1)
    dy = torch.randn_like(pair)

    # --- reference grads (fp32) ---
    base_fp = base.float()
    xr = pair.float().clone().requires_grad_(True)
    yr = ref_forward(xr, base_fp, mask2d.float())
    gxr = torch.autograd.grad(yr, xr, dy.float(), retain_graph=False)[0]

    # --- ours (bf16) ---
    mod = TriMulCute(base.to(torch.bfloat16))
    xo = pair.to(torch.bfloat16).clone().requires_grad_(True)
    yo = mod(xo, mask)
    yo.backward(dy.to(torch.bfloat16))
    gxo = xo.grad

    def cos(a, b):
        a, b = a.float().flatten(), b.float().flatten()
        return (a @ b / (a.norm() * b.norm() + 1e-20)).item()

    fwd_cos = cos(yr, yo)
    dx_cos = cos(gxr, gxo)
    print(f"forward  cos(ref, ours) = {fwd_cos:.5f}", flush=True)
    print(f"grad_x   cos(ref, ours) = {dx_cos:.5f}", flush=True)
    ok = fwd_cos > 0.99 and dx_cos > 0.99
    print(f"-> {'PASS' if ok else 'FAIL'} (bf16, cos>0.99)", flush=True)

    # --- full-mode timing (fwd+bwd), ours, single layer + mask ---
    print("\nfull-mode (fwd+bwd) ms/layer, ours (B1: torch backward):", flush=True)
    for Lb in (256, 512, 1024):
        p = torch.randn(1, Lb, Lb, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        msk = torch.rand(1, Lb, device="cuda") > 0.2
        g = torch.randn_like(p)

        def step():
            p.grad = None
            y = mod(p, msk)
            y.backward(g)
        try:
            for _ in range(3):
                step()
            t = triton.testing.do_bench(step, warmup=10, rep=50, return_mode="median",
                                        grad_to_none=[p])
            print(f"  L={Lb:>4}: {t:.3f} ms", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  L={Lb:>4}: FAIL {type(e).__name__}: {str(e)[:70]}", flush=True)


if __name__ == "__main__":
    main()
