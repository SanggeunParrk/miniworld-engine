"""Correctness check for the CuTeDSL TM1 wrapper.

Compares ``tm1_cute_forward`` (output layout ``[B,d,L,L]``) against a plain
PyTorch reference run in fp32, applying the same ``[B,L,L,d] → [B,d,L,L]``
permute to the reference before comparing.
"""

from __future__ import annotations


# --- miniworld-kernels: make sibling cute dirs importable (vendored script layout) ---
import sys as _sys
from pathlib import Path as _Path
_src_root = _Path(__file__).resolve()
while _src_root.name != "src" and _src_root.parent != _src_root:
    _src_root = _src_root.parent
for _p in (
    _src_root,
    _src_root / "miniworld_kernels" / "kernels" / "tm1" / "cute",
    _src_root / "miniworld_kernels" / "kernels" / "tm2" / "cute",
    _src_root / "miniworld_kernels" / "kernels" / "fused_ln_mask" / "cute",
    _src_root / "miniworld_kernels" / "modules" / "triangle_multiplication",
):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# -------------------------------------------------------------------------------------

import sys

import torch
from launch import tm1_cute_forward


def tm1_reference(x, WL, WLg, WR, WRg):
    """PyTorch reference (fp32 math, cast at the end)."""
    out_dtype = x.dtype
    xf, WLf, WLgf, WRf, WRgf = (t.float() for t in (x, WL, WLg, WR, WRg))
    left = torch.sigmoid(xf @ WLgf) * (xf @ WLf)
    right = torch.sigmoid(xf @ WRgf) * (xf @ WRf)
    left_bdll = left.permute(0, 3, 1, 2).contiguous().to(out_dtype)
    right_bdll = right.permute(0, 3, 1, 2).contiguous().to(out_dtype)
    return left_bdll, right_bdll


def check(B, L, D, dtype, atol, rtol):
    torch.manual_seed(0)
    x = torch.randn(B, L, L, D, device="cuda", dtype=dtype)
    scale = D**-0.5
    WL = (torch.randn(D, D, device="cuda", dtype=dtype) * scale).contiguous()
    WLg = (torch.randn(D, D, device="cuda", dtype=dtype) * scale).contiguous()
    WR = (torch.randn(D, D, device="cuda", dtype=dtype) * scale).contiguous()
    WRg = (torch.randn(D, D, device="cuda", dtype=dtype) * scale).contiguous()

    left_ref, right_ref = tm1_reference(x, WL, WLg, WR, WRg)
    left_cu, right_cu = tm1_cute_forward(x, WL, WLg, WR, WRg)

    for name, ref, cu in (("left", left_ref, left_cu), ("right", right_ref, right_cu)):
        assert cu.shape == ref.shape, f"{name}: shape mismatch {cu.shape} vs {ref.shape}"
        assert cu.dtype == ref.dtype, f"{name}: dtype mismatch {cu.dtype} vs {ref.dtype}"
        assert cu.is_contiguous(), f"{name}: cute output not contiguous"
        diff = (cu.float() - ref.float()).abs()
        max_abs = diff.max().item()
        ref_abs = ref.float().abs().max().item() + 1e-12
        rel = max_abs / ref_abs
        ok = torch.allclose(cu, ref, atol=atol, rtol=rtol)
        flag = "OK " if ok else "FAIL"
        print(
            f"  {flag} {name:5s}  B={B} L={L} D={D} {dtype!s:>14s}  "
            f"max_abs={max_abs:.3e}  rel={rel:.3e}"
        )
        if not ok:
            return False
    return True


def main():
    assert torch.cuda.is_available()
    print(f"CuTeDSL TM1 correctness on {torch.cuda.get_device_name(0)}")
    configs = [
        # (B, L, D, dtype, atol, rtol)
        (1, 64, 128, torch.bfloat16, 5e-3, 1e-2),
        (1, 128, 128, torch.bfloat16, 5e-3, 1e-2),
        (1, 384, 128, torch.bfloat16, 8e-3, 1e-2),
        (1, 64, 128, torch.float16, 2e-3, 1e-2),
        (1, 384, 128, torch.float16, 4e-3, 1e-2),
        (2, 128, 128, torch.bfloat16, 5e-3, 1e-2),
    ]
    all_ok = True
    for c in configs:
        all_ok &= check(*c)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
