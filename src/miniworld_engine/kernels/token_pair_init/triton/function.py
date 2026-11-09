"""The portable (Triton) backend of ``TokenPairInit``: the forward / backward kernels of ``main.py`` as an autograd function. The inputs are the prepared
tensors of ``cuda/function.py`` (the same module-side preparation); z comes out fp32, every gradient goes back in its input's dtype (the table gradient is
fp32-accurate and bitwise deterministic)."""
from __future__ import annotations

import torch

from miniworld_engine.kernels.token_pair_init.triton.main import portable_backward, portable_forward


class PortableTokenPairInitFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right, tbl, ids, bond, r_max, s_max):
        ctx.dtypes = (left.dtype, right.dtype, tbl.dtype)
        ctx.save_for_backward(ids, bond)
        ctx.meta = (r_max, s_max)
        f32 = torch.float32
        return portable_forward(left.to(f32).contiguous(), right.to(f32).contiguous(), tbl.to(f32).contiguous(), ids, bond, r_max, s_max)

    @staticmethod
    def backward(ctx, g):
        ids, bond = ctx.saved_tensors
        grads = portable_backward(g.float().contiguous(), ids, bond, *ctx.meta)
        dleft, dright, dtbl = (d.to(dt) for d, dt in zip(grads, ctx.dtypes, strict=True))
        return dleft, dright, dtbl, None, None, None, None


def token_pair_init_triton(left, right, tbl, ids, bond, r_max: int, s_max: int) -> torch.Tensor:
    """z [B, L, L, 128] fp32, differentiable in ``left``, ``right`` and ``tbl``."""
    return PortableTokenPairInitFunction.apply(left, right, tbl, ids, bond, r_max, s_max)
