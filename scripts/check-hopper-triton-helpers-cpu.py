"""Execute real Triton helper bodies in its CPU interpreter (FP32 arithmetic)."""
import os

os.environ["TRITON_INTERPRET"] = "1"

from itertools import product
from unittest.mock import patch

import torch
import triton
from triton.runtime import interpreter

from miniworld_engine.kernels.transition.cute.fused import _xn_recompute_kernel
from miniworld_engine.kernels.trimul_inproj.triton.back_fused import _dconcat_kernel
from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import (
    _gate_elem_bwd_ew_kernel,
)


def main():
    torch.manual_seed(71)
    count = 0
    for m, d, block, mask_kind in product((9, 25), (7, 13), (32, 128), ("none", "holes", "zero", "fractional")):
        dl, dr, pre = torch.randn(d, m), torch.randn(d, m), torch.randn(4 * d, m)
        mask = torch.rand(m)
        if mask_kind == "none":
            mask = None
        elif mask_kind == "holes":
            mask = (mask > .5).float()
        elif mask_kind == "zero":
            mask.zero_()
        storage = torch.full((4 * d * m + 16,), 12345.)
        out = storage[:4 * d * m].reshape(4 * d, m)
        _dconcat_kernel.fn[(triton.cdiv(d * m, block),)](
            dl, dr, pre, out, mask, m, d * m, D=d, BLOCK_E=block, shape_key=0)
        if mask is not None:
            dl, dr = dl * mask, dr * mask
        gl, gr = pre[:2 * d:2].sigmoid(), pre[2 * d::2].sigmoid()
        reference = torch.cat((dl * pre[1:2 * d:2] * gl * (1 - gl), dl * gl,
                               dr * pre[2 * d + 1::2] * gr * (1 - gr), dr * gr))
        torch.testing.assert_close(out, reference, atol=2e-6, rtol=2e-6)
        assert torch.all(storage[-16:] == 12345)
        count += 1
    for length, n, bm, bk, preact in product((3, 5), (7, 13), (4, 16), (8, 32), (False, True)):
        m = length * length
        dy, proj, gate = (torch.randn(m, n) for _ in range(3))
        drop = (torch.rand(length, n) > .3).float() / .7
        dp, dg = torch.empty_like(dy), torch.empty_like(dy)
        _gate_elem_bwd_ew_kernel.fn[(triton.cdiv(m, bm),)](
            dy, proj, gate, dp, dg, drop, length, m, N=n, BLOCK_M1=bm, BLOCK_K=bk,
            shape_key=0, FROM_PREACT=preact)
        g = gate.sigmoid() if preact else gate
        scaled = dy * drop[torch.arange(m) % length]
        torch.testing.assert_close(dp, scaled * g, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(dg, scaled * proj * g * (1 - g), atol=2e-6, rtol=2e-6)
        count += 1
    for major, bm, bk in product(("k", "m"), (4, 16), (8, 32)):
        m, k = 7, 13
        x = torch.randn(m, k)
        if major == "m":
            x = x.t().contiguous().t()
        mean = x.mean(-1)
        rstd = (x.var(-1, unbiased=False) + .03).rsqrt()
        c1, g, b = mean * rstd, torch.randn(k), torch.randn(k)
        out = torch.empty_like(x)
        _xn_recompute_kernel.fn[(triton.cdiv(m, bm),)](
            x, rstd, c1, g, b, out, m, k, *x.stride(), BLOCK_M1=bm, BLOCK_K=bk, shape_key=0)
        torch.testing.assert_close(out, (x * rstd[:, None] - c1[:, None]) * g + b,
                                   atol=2e-6, rtol=2e-6)
        count += 1
    print(f"PASS {count} FP32 Triton interpreter cases; no GPU instructions executed", flush=True)


if __name__ == "__main__":
    # Triton's interpreter uses int(ndarray([scalar])), rejected by NumPy 2.4.
    # Repair scalar conversion only inside this CPU harness; kernel bodies and
    # the installed Triton package are unchanged.
    original_patch = interpreter._patch_lang_tensor

    def patch_scalar_index(tensor, scope):
        original_patch(tensor, scope)
        scope.set_attr(tensor, "__index__", lambda self: int(self.handle.data.item()))

    with patch.object(interpreter, "_patch_lang_tensor", patch_scalar_index):
        main()
