"""Fused token-pair initialisation (kernel family ``token_pair_init``): the input feature embedder's

    z[b,i,j] = left[b,i] + right[b,j] + Linear_rel(relative_position_one_hot(i, j)) + Linear_bond(one_hot(bond[b,i,j]))

as one kernel pair instead of a 139-wide fp32 one-hot (84 MB at L = 384), a Linear over it, the outer sum, the bond one-hot
Linear and their adds -- and, in the backward, one pass over dz for the gradients of ``left``, ``right`` and both weights
instead of a one-hot re-materialisation and a long-K weight-gradient GEMM. Exact fp32 arithmetic (the reference's TF32
Linears round the weights; this does not). The class-bin and dright sums use global atomics: run-to-run the last bits of
dright / dW can differ.

Served: B200 (sm_100a), d_pair 128. The kernels compute in fp32; ``left`` / ``right`` may be fp32 or bf16 (a bf16-autocast
caller) and the weights fp32 or bf16 (an fp32 master or bf16 parameters): the autograd function casts them to fp32 outside
autograd and hands every gradient back in its input's dtype. :func:`refusal` says why anything else is not served; it never
raises.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels._compile import opaque

D_PAIR = 128


def refusal(left: torch.Tensor, right: torch.Tensor, w_rel: torch.Tensor, w_bond: torch.Tensor, bond: torch.Tensor | None,
            *, r_max: int = 32, s_max: int = 2) -> str | None:
    """None if the fused op can run this call, else why it cannot. Only metadata is read (traces under ``torch.compile``)."""
    from miniworld_engine.kernels.token_pair_init.cuda.sm100 import n_rel

    if bond is None:
        return "no dense bond adjacency (token_bond_feat) to gather from"
    if not left.is_cuda:
        return "the input is not on a CUDA device"
    if any(t.dtype not in (torch.float32, torch.bfloat16) for t in (left, right, w_rel, w_bond)):
        return "left, right and both weights must be fp32 or bf16 (the kernels compute in fp32)"
    if left.dim() != 3 or left.shape != right.shape or left.shape[-1] != D_PAIR:
        return f"left / right must be [B, L, {D_PAIR}], got {tuple(left.shape)} / {tuple(right.shape)}"
    if n_rel(r_max, s_max) - 2 > 255:
        return f"the kernel packs each relative-position bin into 8 bits; r_max={r_max}, s_max={s_max} needs more"
    if tuple(w_rel.shape) != (D_PAIR, n_rel(r_max, s_max)) or tuple(w_bond.shape) != (D_PAIR, 2):
        return f"weights must be [{D_PAIR}, {n_rel(r_max, s_max)}] and [{D_PAIR}, 2], got {tuple(w_rel.shape)} / {tuple(w_bond.shape)}"
    if tuple(bond.shape) != (left.shape[0], left.shape[1], left.shape[1]):
        return f"bond must be [B, L, L], got {tuple(bond.shape)}"
    if torch.cuda.get_device_capability(left.device) != (10, 0):
        return "served on B200 (sm_100a) only"
    return None


def _token_pair_init_fwd_fake(left, right, tbl, ids, bond, r_max, s_max):
    """z [B, L, L, P] fp32 (the sum of the two streams and the two table lookups; P comes from ``left``)."""
    B, L, P = left.shape
    return torch.empty((B, L, L, P), dtype=torch.float32, device=left.device)


@opaque(fake=_token_pair_init_fwd_fake, name="token_pair_init_fwd")
def token_pair_init_fwd(left: torch.Tensor, right: torch.Tensor, tbl: torch.Tensor, ids: torch.Tensor, bond: torch.Tensor,
                        r_max: int, s_max: int) -> torch.Tensor:
    """z [B, L, L, 128] from left / right [B, L, 128], tbl [n_rel + 2, 128] = [w_rel^T ; w_bond^T], ids [5, B*L] int32,
    bond [B*L*L] uint8."""
    from miniworld_engine.kernels.token_pair_init.cuda import sm100

    return sm100.forward(left, right, tbl, ids, bond, r_max, s_max)


def _token_pair_init_bwd_fake(g, ids, bond, r_max, s_max):
    """[d_left [B, L, P], d_right [B, L, P], d_table [n_rel + 2, P]] fp32; the table rows follow from ``r_max`` / ``s_max``."""
    from miniworld_engine.kernels.token_pair_init.cuda.sm100 import n_rel

    B, L = g.shape[:2]
    return [torch.empty((B, L, D_PAIR), dtype=torch.float32, device=g.device),
            torch.empty((B, L, D_PAIR), dtype=torch.float32, device=g.device),
            torch.empty((n_rel(r_max, s_max) + 2, D_PAIR), dtype=torch.float32, device=g.device)]


@opaque(fake=_token_pair_init_bwd_fake, name="token_pair_init_bwd")
def token_pair_init_bwd(g: torch.Tensor, ids: torch.Tensor, bond: torch.Tensor, r_max: int, s_max: int) -> list[torch.Tensor]:
    """[dleft, dright, dtbl] from dz = g [B, L, L, 128] fp32."""
    from miniworld_engine.kernels.token_pair_init.cuda import sm100

    return list(sm100.backward(g, ids, bond, r_max, s_max))


class TokenPairInitFunction(torch.autograd.Function):
    """The kernels read fp32: bf16 ``left`` / ``right`` (autocast) and a bf16 table are cast here, outside autograd, and the
    gradients go back in each input's dtype (an fp32 table -- fp32 master weights -- gets its fp32 gradient unrounded)."""

    @staticmethod
    def forward(ctx, left, right, tbl, ids, bond, r_max, s_max):
        ctx.dtypes = (left.dtype, right.dtype, tbl.dtype)
        ctx.save_for_backward(ids, bond)
        ctx.meta = (r_max, s_max)
        f32 = torch.float32
        return token_pair_init_fwd(left.to(f32).contiguous(), right.to(f32).contiguous(), tbl.to(f32).contiguous(), ids, bond,
                                   r_max, s_max)

    @staticmethod
    def backward(ctx, g):
        ids, bond = ctx.saved_tensors
        dleft, dright, dtbl = token_pair_init_bwd(g.float().contiguous(), ids, bond, *ctx.meta)
        dleft, dright, dtbl = (d.to(dt) for d, dt in zip((dleft, dright, dtbl), ctx.dtypes, strict=True))
        return dleft, dright, dtbl, None, None, None, None


def token_pair_init(left: torch.Tensor, right: torch.Tensor, w_rel: torch.Tensor, w_bond: torch.Tensor, asym_id: torch.Tensor,
                    residue_idx: torch.Tensor, token_idx: torch.Tensor, entity_id: torch.Tensor, sym_id: torch.Tensor,
                    bond: torch.Tensor, *, r_max: int = 32, s_max: int = 2) -> torch.Tensor:
    """z [B, L, L, 128] fp32, differentiable in ``left``, ``right``, ``w_rel`` [128, n_rel] and ``w_bond`` [128, 2].

    ``left`` / ``right`` [B, L, 128] fp32 or bf16 (the weights fp32 or bf16; computed in fp32 either way); the five id tensors [B, L] (any integer dtype); ``bond`` [B, L, L] 0/1 (bool, integer or
    float). Check :func:`refusal` first."""
    B, L, _ = left.shape
    ids = torch.stack([asym_id, residue_idx, token_idx, entity_id, sym_id]).to(torch.int32).reshape(5, B * L)
    bond = (bond != 0).to(torch.uint8).reshape(-1)                     # any nonzero is a bond: the kernel's table has two bond rows
    tbl = torch.cat([w_rel.t(), w_bond.t()], 0).contiguous()
    return TokenPairInitFunction.apply(left.contiguous(), right.contiguous(), tbl, ids, bond, r_max, s_max)


__all__ = ["D_PAIR", "refusal", "token_pair_init"]
