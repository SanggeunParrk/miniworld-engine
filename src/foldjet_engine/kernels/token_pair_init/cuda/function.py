"""The CUDA backend of ``TokenPairInit``: the prep / forward / backward launches of ``sm90`` (``token_pair_init.cu``, H100 sm_90a; on an A100 the sm_80 build of the same source, ``sm80``; on a B200 the sm_100a kernels of ``sm100``) as opaque ops and an
autograd function. Inputs are the module's prepared tensors (``modules/token_pair_init.py``): ``left`` / ``right`` [B, L, 128], the table ``tbl``
[n_rel + 2, 128] (= [W_rel^T ; W_bond^T]), ``ids`` [5, B * L] int32 (asym, residue, token, entity, sym) and ``bond`` [B * L * L] uint8. The kernels compute in fp32
and return z fp32; every gradient goes back in its input's dtype."""

from __future__ import annotations

import torch

from foldjet_engine.kernels._compile import opaque

D_PAIR = 128


def _build():
    """The launchers of the current device: the B200 sm_100a build (``sm100``), the A100 / RTX 5090 sm_80 build, else the H100 (sm_90a) one."""
    if torch.cuda.get_device_capability() == (10, 0):
        from foldjet_engine.kernels.token_pair_init.cuda import sm100

        return sm100
    from foldjet_engine.kernels.token_pair_init.cuda.sm80 import device_build

    return device_build()


def _token_pair_init_fwd_fake(left, right, tbl, ids, bond, r_max, s_max):
    """z [B, L, L, P] fp32 (the sum of the two streams and the two table lookups; P comes from ``left``)."""
    B, L, P = left.shape
    return torch.empty((B, L, L, P), dtype=torch.float32, device=left.device)


@opaque(fake=_token_pair_init_fwd_fake, name="token_pair_init_fwd_sm90")
def token_pair_init_fwd(
    left: torch.Tensor, right: torch.Tensor, tbl: torch.Tensor, ids: torch.Tensor, bond: torch.Tensor, r_max: int, s_max: int
) -> torch.Tensor:
    """z [B, L, L, 128] from left / right [B, L, 128], tbl [n_rel + 2, 128], ids [5, B*L] int32, bond [B*L*L] uint8."""
    return _build().forward(left, right, tbl, ids, bond, r_max, s_max)


def _token_pair_init_bwd_fake(g, ids, bond, nbin, r_max, s_max):
    """[d_left [B, L, P], d_right [B, L, P], d_table [nbin, P]] fp32."""
    B, L = g.shape[:2]
    return [
        torch.empty((B, L, D_PAIR), dtype=torch.float32, device=g.device),
        torch.empty((B, L, D_PAIR), dtype=torch.float32, device=g.device),
        torch.empty((nbin, D_PAIR), dtype=torch.float32, device=g.device),
    ]


@opaque(fake=_token_pair_init_bwd_fake, name="token_pair_init_bwd_sm90")
def token_pair_init_bwd(g: torch.Tensor, ids: torch.Tensor, bond: torch.Tensor, nbin: int, r_max: int, s_max: int) -> list[torch.Tensor]:
    """[dleft, dright, dtbl] from dz = g [B, L, L, 128] fp32 (``nbin`` = table rows)."""
    return list(_build().backward(g, ids, bond, nbin, r_max, s_max))


def _token_pair_init_prep_fake(asym, residue, token, entity, sym, w_rel, w_bond):
    """ids [5, B*L] int32 and the table [n_rel + 2, 128] fp32."""
    return [
        torch.empty((5, asym.numel()), dtype=torch.int32, device=w_rel.device),
        torch.empty((w_rel.shape[1] + 2, D_PAIR), dtype=torch.float32, device=w_rel.device),
    ]


@opaque(fake=_token_pair_init_prep_fake, name="token_pair_init_prep_sm90")
def token_pair_init_prep(
    asym: torch.Tensor,
    residue: torch.Tensor,
    token: torch.Tensor,
    entity: torch.Tensor,
    sym: torch.Tensor,
    w_rel: torch.Tensor,
    w_bond: torch.Tensor,
) -> list[torch.Tensor]:
    """[ids [5, B*L] int32, tbl [n_rel + 2, 128] fp32] in one launch (forward-only calls: no gradient flows through it)."""
    return list(_build().prep([asym, residue, token, entity, sym], w_rel, w_bond))


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


def _token_pair_init_fwd_w_fake(left, right, w_rel, w_bond, asym, residue, token, entity, sym, bond, r_max, s_max):
    """z [B, L, L, P] fp32 (as ``token_pair_init_fwd``)."""
    B, L, P = left.shape
    return torch.empty((B, L, L, P), dtype=torch.float32, device=left.device)


@opaque(fake=_token_pair_init_fwd_w_fake, name="token_pair_init_fwd_w_sm100")
def token_pair_init_fwd_w(
    left: torch.Tensor,
    right: torch.Tensor,
    w_rel: torch.Tensor,
    w_bond: torch.Tensor,
    asym: torch.Tensor,
    residue: torch.Tensor,
    token: torch.Tensor,
    entity: torch.Tensor,
    sym: torch.Tensor,
    bond: torch.Tensor,
    r_max: int,
    s_max: int,
) -> torch.Tensor:
    """B200 forward-only call in ONE launch (``sm100.forward_w``, kernel ``tpi_fwd_w``): z [B, L, L, 128] fp32 from left / right [B, L, 128] fp32, the
    fp32 weights as they are (the table staged in the kernel), the five id tensors [B * L] int32 / int64 and bond [B*L*L] uint8. The same z as the
    prep + forward launches."""
    from foldjet_engine.kernels.token_pair_init.cuda import sm100

    return sm100.forward_w(left, right, w_rel, w_bond, [asym, residue, token, entity, sym], bond, r_max, s_max)


def _token_pair_init_bwd_w_fake(g, asym, residue, token, entity, sym, bond, r_max, s_max, bf16_left, bf16_right):
    """[d_left, d_right [B, L, P] (fp32, or bf16 with ``bf16_left`` / ``bf16_right``), d_table [n_rel + 2, P] fp32]."""
    B, L = g.shape[:2]
    nbin = 2 * (2 * r_max + 2) + (2 * s_max + 2) + 1 + 2
    return [
        torch.empty((B, L, D_PAIR), dtype=torch.bfloat16 if bf16_left else torch.float32, device=g.device),
        torch.empty((B, L, D_PAIR), dtype=torch.bfloat16 if bf16_right else torch.float32, device=g.device),
        torch.empty((nbin, D_PAIR), dtype=torch.float32, device=g.device),
    ]


@opaque(fake=_token_pair_init_bwd_w_fake, name="token_pair_init_bwd_w_sm100")
def token_pair_init_bwd_w(
    g: torch.Tensor,
    asym: torch.Tensor,
    residue: torch.Tensor,
    token: torch.Tensor,
    entity: torch.Tensor,
    sym: torch.Tensor,
    bond: torch.Tensor,
    r_max: int,
    s_max: int,
    bf16_left: bool,
    bf16_right: bool,
) -> list[torch.Tensor]:
    """B200: the backward of ``token_pair_init_fwd_w`` on the five id tensors as they are (``sm100.backward_w``: ``tpi_bwd_w`` + ``tpi_bwd_red``)."""
    from foldjet_engine.kernels.token_pair_init.cuda import sm100

    f32, bf = torch.float32, torch.bfloat16
    return list(sm100.backward_w(g, [asym, residue, token, entity, sym], bond, r_max, s_max, bf if bf16_left else f32, bf if bf16_right else f32))


class TokenPairInitFunctionW(torch.autograd.Function):
    """B200, fp32 weights and int32 / int64 ids as the caller holds them: the forward is the ONE launch of the forward-only call
    (``tpi_fwd_w``: no ids / table prep, bf16 ``left`` / ``right`` read as they are) and the backward two launches on the same tensors
    (``tpi_bwd_w`` and its fixed-order reduce ``tpi_bwd_red``: bitwise repeatable, nothing zero-filled, ``dleft`` / ``dright`` written in the
    inputs' dtypes) -- 2026-11-17: the prep by torch ops (stack, cast, cat), the bf16 casts and the fills were 3-5 of the step's 7-11
    launches. The weight gradients are transposed views of the kernel's table gradient (as ``TokenPairInitFunction``'s are)."""

    @staticmethod
    def forward(ctx, left, right, w_rel, w_bond, asym, residue, token, entity, sym, bond, r_max, s_max):
        from foldjet_engine.kernels.token_pair_init.cuda import sm100

        sm100.backward_w_ready(left, r_max, s_max)  # the backward's kernels are loaded here, not on autograd's device thread
        ctx.save_for_backward(asym, residue, token, entity, sym, bond)
        ctx.meta = (r_max, s_max, left.dtype, right.dtype)
        return token_pair_init_fwd_w(left, right, w_rel, w_bond, asym, residue, token, entity, sym, bond, r_max, s_max)  # pyright: ignore[reportCallIssue]

    @staticmethod
    def backward(ctx, g):
        asym, residue, token, entity, sym, bond = ctx.saved_tensors
        r_max, s_max, ldt, rdt = ctx.meta
        dleft, dright, dtbl = token_pair_init_bwd_w(  # pyright: ignore[reportCallIssue]
            g.float().contiguous(), asym, residue, token, entity, sym, bond, r_max, s_max, ldt == torch.bfloat16, rdt == torch.bfloat16
        )
        nrel = dtbl.shape[0] - 2
        return dleft, dright, dtbl[:nrel].t(), dtbl[nrel:].t(), None, None, None, None, None, None, None, None
