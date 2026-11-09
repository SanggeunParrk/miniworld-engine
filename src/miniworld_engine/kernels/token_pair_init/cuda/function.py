"""The CUDA backend of ``TokenPairInit``: the prep / forward / backward launches of ``sm90`` (``token_pair_init.cu``, H100 sm_90a) as opaque ops and an
autograd function. Inputs are the module's prepared tensors (``modules/token_pair_init.py``): ``left`` / ``right`` [B, L, 128], the table ``tbl``
[n_rel + 2, 128] (= [W_rel^T ; W_bond^T]), ``ids`` [5, B * L] int32 (asym, residue, token, entity, sym) and ``bond`` [B * L * L] uint8. The kernels compute in fp32
and return z fp32; every gradient goes back in its input's dtype."""
from __future__ import annotations

import torch

from miniworld_engine.kernels._compile import opaque

D_PAIR = 128


def _token_pair_init_fwd_fake(left, right, tbl, ids, bond, r_max, s_max):
    """z [B, L, L, P] fp32 (the sum of the two streams and the two table lookups; P comes from ``left``)."""
    B, L, P = left.shape
    return torch.empty((B, L, L, P), dtype=torch.float32, device=left.device)


@opaque(fake=_token_pair_init_fwd_fake, name="token_pair_init_fwd_sm90")
def token_pair_init_fwd(left: torch.Tensor, right: torch.Tensor, tbl: torch.Tensor, ids: torch.Tensor, bond: torch.Tensor, r_max: int,
                        s_max: int) -> torch.Tensor:
    """z [B, L, L, 128] from left / right [B, L, 128], tbl [n_rel + 2, 128], ids [5, B*L] int32, bond [B*L*L] uint8."""
    from miniworld_engine.kernels.token_pair_init.cuda import sm90

    return sm90.forward(left, right, tbl, ids, bond, r_max, s_max)


def _token_pair_init_bwd_fake(g, ids, bond, nbin, r_max, s_max):
    """[d_left [B, L, P], d_right [B, L, P], d_table [nbin, P]] fp32."""
    B, L = g.shape[:2]
    return [torch.empty((B, L, D_PAIR), dtype=torch.float32, device=g.device), torch.empty((B, L, D_PAIR), dtype=torch.float32, device=g.device),
            torch.empty((nbin, D_PAIR), dtype=torch.float32, device=g.device)]


@opaque(fake=_token_pair_init_bwd_fake, name="token_pair_init_bwd_sm90")
def token_pair_init_bwd(g: torch.Tensor, ids: torch.Tensor, bond: torch.Tensor, nbin: int, r_max: int, s_max: int) -> list[torch.Tensor]:
    """[dleft, dright, dtbl] from dz = g [B, L, L, 128] fp32 (``nbin`` = table rows)."""
    from miniworld_engine.kernels.token_pair_init.cuda import sm90

    return list(sm90.backward(g, ids, bond, nbin, r_max, s_max))


def _token_pair_init_prep_fake(asym, residue, token, entity, sym, w_rel, w_bond):
    """ids [5, B*L] int32 and the table [n_rel + 2, 128] fp32."""
    return [torch.empty((5, asym.numel()), dtype=torch.int32, device=w_rel.device),
            torch.empty((w_rel.shape[1] + 2, D_PAIR), dtype=torch.float32, device=w_rel.device)]


@opaque(fake=_token_pair_init_prep_fake, name="token_pair_init_prep_sm90")
def token_pair_init_prep(asym: torch.Tensor, residue: torch.Tensor, token: torch.Tensor, entity: torch.Tensor, sym: torch.Tensor,
                         w_rel: torch.Tensor, w_bond: torch.Tensor) -> list[torch.Tensor]:
    """[ids [5, B*L] int32, tbl [n_rel + 2, 128] fp32] in one launch (forward-only calls: no gradient flows through it)."""
    from miniworld_engine.kernels.token_pair_init.cuda import sm90

    return list(sm90.prep([asym, residue, token, entity, sym], w_rel, w_bond))


class TokenPairInitFunction(torch.autograd.Function):
    """The kernels read fp32: bf16 ``left`` / ``right`` (autocast) and a bf16 table are cast here, outside autograd, and the gradients go back in each
    input's dtype (an fp32 table -- fp32 master weights -- gets its fp32 gradient unrounded)."""

    @staticmethod
    def forward(ctx, left, right, tbl, ids, bond, r_max, s_max):
        ctx.dtypes = (left.dtype, right.dtype, tbl.dtype)
        ctx.save_for_backward(ids, bond)
        ctx.meta = (tbl.shape[0], r_max, s_max)
        f32 = torch.float32
        return token_pair_init_fwd(left.to(f32).contiguous(), right.to(f32).contiguous(), tbl.to(f32).contiguous(), ids, bond, r_max, s_max)

    @staticmethod
    def backward(ctx, g):
        ids, bond = ctx.saved_tensors
        grads = token_pair_init_bwd(g.float().contiguous(), ids, bond, *ctx.meta)
        dleft, dright, dtbl = (d.to(dt) for d, dt in zip(grads, ctx.dtypes, strict=True))
        return dleft, dright, dtbl, None, None, None, None


def token_pair_init_cuda(left, right, tbl, ids, bond, r_max: int, s_max: int) -> torch.Tensor:
    """z [B, L, L, 128] fp32, differentiable in ``left``, ``right`` and ``tbl``."""
    return TokenPairInitFunction.apply(left, right, tbl, ids, bond, r_max, s_max)
