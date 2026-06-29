"""Correctness check for the CuTeDSL trimul_inproj kernel.

Compares ``trimul_inproj_cute_forward`` against a plain fp32 PyTorch reference,
applying the same ``[B,L,L,D] -> [B,D,L,L]`` permute to the left/right reference
(gate is compared in ``[B,L,L,D]``).

Run on a COMPUTE NODE only:

    srun --partition=h100 --account=cssb --qos=cssb_h100 --gres=gpu:h100:1 \
         --cpus-per-task=8 --mem=64G --time=00:20:00 \
         bash -c 'cd /home/psk6950/miniworld-kernels && PYTHONPATH=src \
         pixi run --frozen python \
         src/miniworld_kernels/kernels/trimul_inproj/cute/verify.py'
"""

from __future__ import annotations


# --- make sibling cute dirs importable (vendored script layout, as in tm1) ---
import sys as _sys
from pathlib import Path as _Path

_src_root = _Path(__file__).resolve()
while _src_root.name != "src" and _src_root.parent != _src_root:
    _src_root = _src_root.parent
for _p in (
    _src_root,
    _src_root / "miniworld_kernels" / "kernels" / "trimul_inproj" / "cute",
):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
# -----------------------------------------------------------------------------

import sys

import torch
from launch import trimul_inproj_cute_forward


def reference(x, WL, WLg, WR, WRg, Wg):
    """fp32 reference, cast back to x.dtype."""
    out_dtype = x.dtype
    xf, *Wf = (t.float() for t in (x, WL, WLg, WR, WRg, Wg))
    WLf, WLgf, WRf, WRgf, Wgf = Wf
    left = torch.sigmoid(xf @ WLgf) * (xf @ WLf)
    right = torch.sigmoid(xf @ WRgf) * (xf @ WRf)
    gate = torch.sigmoid(xf @ Wgf)
    left_bdll = left.permute(0, 3, 1, 2).contiguous().to(out_dtype)
    right_bdll = right.permute(0, 3, 1, 2).contiguous().to(out_dtype)
    return left_bdll, right_bdll, gate.to(out_dtype)


def check(B, L, D, dtype, atol, rtol, bdll_direct=False):
    torch.manual_seed(0)
    x = torch.randn(B, L, L, D, device="cuda", dtype=dtype)
    scale = D**-0.5

    def w():
        return (torch.randn(D, D, device="cuda", dtype=dtype) * scale).contiguous()

    WL, WLg, WR, WRg, Wg = w(), w(), w(), w(), w()

    ref = reference(x, WL, WLg, WR, WRg, Wg)
    cu = trimul_inproj_cute_forward(x, WL, WLg, WR, WRg, Wg, bdll_direct=bdll_direct)

    ok_all = True
    for name, r, c in zip(("left", "right", "gate"), ref, cu):
        assert c.shape == r.shape, f"{name}: shape {tuple(c.shape)} vs {tuple(r.shape)}"
        assert c.dtype == r.dtype, f"{name}: dtype {c.dtype} vs {r.dtype}"
        assert c.is_contiguous(), f"{name}: cute output not contiguous"
        diff = (c.float() - r.float()).abs()
        max_abs = diff.max().item()
        rel = max_abs / (r.float().abs().max().item() + 1e-12)
        ok = torch.allclose(c, r, atol=atol, rtol=rtol)
        ok_all &= ok
        print(
            f"  {'OK ' if ok else 'FAIL'} {name:5s}  B={B} L={L} D={D} "
            f"{dtype!s:>14s}  bdll_direct={bdll_direct!s:5s}  "
            f"max_abs={max_abs:.3e}  rel={rel:.3e}"
        )
    return ok_all


def main():
    assert torch.cuda.is_available()
    print(f"CuTeDSL trimul_inproj correctness on {torch.cuda.get_device_name(0)}")
    configs = [
        # (B, L, D, dtype, atol, rtol)
        (1, 64, 128, torch.bfloat16, 5e-3, 1e-2),
        (1, 128, 128, torch.bfloat16, 5e-3, 1e-2),
        (1, 384, 128, torch.bfloat16, 8e-3, 1e-2),
        (1, 512, 128, torch.bfloat16, 8e-3, 1e-2),
        (1, 64, 128, torch.float16, 2e-3, 1e-2),
        (1, 384, 128, torch.float16, 4e-3, 1e-2),
    ]
    all_ok = True
    for direct in (False, True):  # fallback (permute) and the bdll-direct patch
        print(f"--- bdll_direct={direct} ---")
        for c in configs:
            all_ok &= check(*c, bdll_direct=direct)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
