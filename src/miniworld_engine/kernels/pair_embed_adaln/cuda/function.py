"""The CUDA backend of ``PairEmbedAdaLN``: the forward / backward launches of ``sm90`` (``pair_embed_adaln.cu``, H100 sm_90a; on an A100 the sm_80 build of the same source, ``sm80``) as opaque ops and an
autograd function. Inputs are the module's prepared tensors (``modules/pair_embed.py``): the table ``tbl`` [NT, 256] (= W^T), ``lnw`` / ``lnb`` [256],
``gamma`` / ``beta`` [B, 256], fp32 coordinates ``xt`` / ``sc`` [B, n, 3] (``sc`` = ``xt`` when there is no self-conditioning, ``has_sc`` 0), int32
``idx`` [B, n], uint8 ``mask`` [B, n] and the fp32 bin limits ``xlim`` / ``clim``. The pair comes out bf16."""

from __future__ import annotations

import torch

from miniworld_engine.kernels._compile import opaque

D_PAIR = 256


def _pair_embed_adaln_fwd_fake(tbl, lnw, lnb, gamma, beta, xt, sc, idx, mask, xlim, clim, has_sc, nS, nX, eps1, eps2):
    """[pair bf16 [B, n, n, 256], bins int32 [B, n, n]]: the shapes of the forward (the packed bin codes the backward replays)."""
    B, n = idx.shape
    return [
        torch.empty((B, n, n, D_PAIR), dtype=torch.bfloat16, device=tbl.device),
        torch.empty((B, n, n), dtype=torch.int32, device=tbl.device),
    ]


@opaque(fake=_pair_embed_adaln_fwd_fake, name="pair_embed_adaln_fwd_sm90")
def pair_embed_adaln_fwd(
    tbl: torch.Tensor,
    lnw: torch.Tensor,
    lnb: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    xt: torch.Tensor,
    sc: torch.Tensor,
    idx: torch.Tensor,
    mask: torch.Tensor,
    xlim: torch.Tensor,
    clim: torch.Tensor,
    has_sc: int,
    nS: int,
    nX: int,
    eps1: float,
    eps2: float,
) -> list[torch.Tensor]:
    """[pair [B, n, n, 256] bf16, bins [B, n, n] int32] from the table [NT, 256] fp32, the LN affine and gamma / beta [B, 256]."""
    from miniworld_engine.kernels.pair_embed_adaln.cuda.sm80 import device_build

    return list(device_build().forward(tbl, lnw, lnb, gamma, beta, xt, sc, has_sc, idx, mask, xlim, clim, nS, nX, eps1, eps2))


def _pair_embed_adaln_bwd_fake(tbl, lnw, lnb, gamma, bins, dout, nS, nX, eps1, eps2):
    """[dtbl [NT, 256], dln_w, dln_b [256], dgamma, dbeta [B, 256]], all fp32."""
    B = bins.shape[0]

    def f(*s):
        return torch.empty(s, dtype=torch.float32, device=tbl.device)

    return [f(*tbl.shape), f(D_PAIR), f(D_PAIR), f(B, D_PAIR), f(B, D_PAIR)]


@opaque(fake=_pair_embed_adaln_bwd_fake, name="pair_embed_adaln_bwd_sm90")
def pair_embed_adaln_bwd(
    tbl: torch.Tensor,
    lnw: torch.Tensor,
    lnb: torch.Tensor,
    gamma: torch.Tensor,
    bins: torch.Tensor,
    dout: torch.Tensor,
    nS: int,
    nX: int,
    eps1: float,
    eps2: float,
) -> list[torch.Tensor]:
    """[dtbl [NT, 256], dln_w, dln_b [256], dgamma, dbeta [B, 256]] fp32 from d pair = dout [B, n, n, 256] bf16."""
    from miniworld_engine.kernels.pair_embed_adaln.cuda.sm80 import device_build

    return list(device_build().backward(tbl, lnw, lnb, gamma, bins, dout, nS, nX, eps1, eps2))


class PairEmbedAdaLNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tbl, lnw, lnb, gamma, beta, xt, sc, idx, mask, xlim, clim, has_sc, nS, nX, eps1, eps2):
        ctx.dtypes = (tbl.dtype, lnw.dtype, lnb.dtype, gamma.dtype, beta.dtype)
        f32 = torch.float32
        tbl, lnw, lnb, gamma, beta = (t.to(f32).contiguous() for t in (tbl, lnw, lnb, gamma, beta))
        out, bins = pair_embed_adaln_fwd(tbl, lnw, lnb, gamma, beta, xt, sc, idx, mask, xlim, clim, has_sc, nS, nX, eps1, eps2)
        ctx.save_for_backward(tbl, lnw, lnb, gamma, bins)
        ctx.meta = (nS, nX, eps1, eps2)
        return out

    @staticmethod
    def backward(ctx, g):
        tbl, lnw, lnb, gamma, bins = ctx.saved_tensors
        grads = pair_embed_adaln_bwd(tbl, lnw, lnb, gamma, bins, g.to(torch.bfloat16).contiguous(), *ctx.meta)
        outs = [d.to(dt) for d, dt in zip(grads, ctx.dtypes, strict=True)]
        return (*outs, *([None] * 11))


def pair_embed_adaln_cuda(tbl, lnw, lnb, gamma, beta, xt, sc, idx, mask, xlim, clim, has_sc, nS, nX, eps1, eps2) -> torch.Tensor:
    """pair [B, n, n, 256] bf16, differentiable in ``tbl``, ``lnw``, ``lnb``, ``gamma`` and ``beta``."""
    return PairEmbedAdaLNFunction.apply(tbl, lnw, lnb, gamma, beta, xt, sc, idx, mask, xlim, clim, has_sc, nS, nX, eps1, eps2)


# ---- B200 (sm_100a): kernels/pair_embed_adaln/cuda/sm100 ``forward_b200`` / ``backward_b200`` (2026-11-17) -------------------------------
def _fwd_b200_fake(tbl, lnw, lnb, gamma, beta, xt, sc, idx, mask, xlim, clim, has_sc, nS, nX, eps1, eps2, train):
    B, n = idx.shape
    return [
        torch.empty((B, n, n, D_PAIR), dtype=torch.bfloat16, device=tbl.device),
        torch.empty((B, n, n) if train else (0,), dtype=torch.int32, device=tbl.device),
    ]


@opaque(fake=_fwd_b200_fake, name="pair_embed_adaln_fwd_sm100")
def pair_embed_adaln_fwd_b200(
    tbl: torch.Tensor,
    lnw: torch.Tensor,
    lnb: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    xt: torch.Tensor,
    sc: torch.Tensor,
    idx: torch.Tensor,
    mask: torch.Tensor,
    xlim: torch.Tensor,
    clim: torch.Tensor,
    has_sc: int,
    nS: int,
    nX: int,
    eps1: float,
    eps2: float,
    train: bool,
) -> list[torch.Tensor]:
    """[pair [B, n, n, 256] bf16, bins [B, n, n] int32 (empty unless ``train``)]: ``gamma`` / ``beta`` fp32 or bf16, ``idx`` int32 or int64."""
    from miniworld_engine.kernels.pair_embed_adaln.cuda import sm100

    return list(sm100.forward_b200(tbl, lnw, lnb, gamma, beta, xt, sc, has_sc, idx, mask, xlim, clim, nS, nX, eps1, eps2, train))


def _bwd_b200_fake(tbl, lnw, lnb, gamma, bins, dout, nS, nX, eps1, eps2):
    B = bins.shape[0]
    dev = tbl.device
    f32 = torch.float32
    return [
        torch.empty((D_PAIR, tbl.shape[0]), dtype=f32, device=dev),
        torch.empty((D_PAIR,), dtype=f32, device=dev),
        torch.empty((D_PAIR,), dtype=f32, device=dev),
        torch.empty((B, D_PAIR), dtype=gamma.dtype, device=dev),
        torch.empty((B, D_PAIR), dtype=gamma.dtype, device=dev),
    ]


@opaque(fake=_bwd_b200_fake, name="pair_embed_adaln_bwd_sm100")
def pair_embed_adaln_bwd_b200(
    tbl: torch.Tensor,
    lnw: torch.Tensor,
    lnb: torch.Tensor,
    gamma: torch.Tensor,
    bins: torch.Tensor,
    dout: torch.Tensor,
    nS: int,
    nX: int,
    eps1: float,
    eps2: float,
) -> list[torch.Tensor]:
    """[dW [256, NT] fp32 (the weight's layout), dln_w, dln_b [256] fp32, dgamma, dbeta [B, 256] in ``gamma``'s dtype]."""
    from miniworld_engine.kernels.pair_embed_adaln.cuda import sm100

    return list(sm100.backward_b200(tbl, lnw, lnb, gamma, bins, dout, nS, nX, eps1, eps2))


def _fwd_s_b200_fake(u, gg, v, bt, bb, gamma, beta, xt, sc, idx, mask, xlim, clim, has_sc, nS, nX, eps1, eps2):
    B, n = idx.shape
    return torch.empty((B, n, n, D_PAIR), dtype=torch.bfloat16, device=xt.device)


@opaque(fake=_fwd_s_b200_fake, name="pair_embed_adaln_fwd_s_sm100")
def pair_embed_adaln_fwd_s_b200(
    u: torch.Tensor,
    gg: torch.Tensor,
    v: torch.Tensor,
    bt: torch.Tensor,
    bb: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    xt: torch.Tensor,
    sc: torch.Tensor,
    idx: torch.Tensor,
    mask: torch.Tensor,
    xlim: torch.Tensor,
    clim: torch.Tensor,
    has_sc: int,
    nS: int,
    nX: int,
    eps1: float,
    eps2: float,
) -> torch.Tensor:
    """Inference: pair [B, n, n, 256] bf16 from the LayerNorm statistics tables of the weights (``sm100.stats_tables``)."""
    from miniworld_engine.kernels.pair_embed_adaln.cuda import sm100

    return sm100.forward_s_b200((u, gg, v, bt, bb), gamma, beta, xt, sc, has_sc, idx, mask, xlim, clim, nS, nX, eps1, eps2)


def _table_b200(weight: torch.Tensor, train: bool) -> torch.Tensor:
    """``W^T`` [NT, 256] fp32, contiguous: one kernel in a training step; inference: made once per weight version (a capture reads the entry
    its warm-up made)."""

    def build():
        tbl = torch.empty((weight.shape[1], weight.shape[0]), dtype=torch.float32, device=weight.device)
        tbl.copy_(weight.detach().t())
        return tbl

    if train:
        return build()
    from miniworld_engine.kernels import _capture

    return _capture.param_forms("pair_embed_adaln_tbl_b200", (weight,), build, capture_reads_eager=True)


class PairEmbedAdaLNB200Function(torch.autograd.Function):
    """The B200 route: takes the module's ``weight`` [256, NT] itself (the table is made here, its gradient comes back in the weight's layout)
    and reads ``gamma`` / ``beta`` (fp32 or bf16) and ``idx`` (int32 or int64) as they are: no cast kernels around the launches."""

    @staticmethod
    def forward(ctx, weight, lnw, lnb, gamma, beta, xt, sc, idx, mask, xlim, clim, has_sc, nS, nX, eps1, eps2, train):
        ctx.dtypes = (weight.dtype, lnw.dtype, lnb.dtype, gamma.dtype, beta.dtype)
        f32 = torch.float32
        lnw0, lnb0 = lnw, lnb  # the parameters themselves: what the inference tables are cached on
        lnw, lnb = lnw.detach().to(f32).contiguous(), lnb.detach().to(f32).contiguous()
        if gamma.dtype is beta.dtype and gamma.dtype in (f32, torch.bfloat16):
            gamma, beta = gamma.detach().contiguous(), beta.detach().contiguous()
        else:
            gamma, beta = gamma.detach().to(f32).contiguous(), beta.detach().to(f32).contiguous()
        if idx.dtype not in (torch.int32, torch.int64) or not idx.is_contiguous():
            idx = idx.to(torch.int32).contiguous()
        if not train:
            from miniworld_engine.kernels.pair_embed_adaln.cuda import sm100

            if sm100.STATS and all(t.dtype is f32 and t.is_contiguous() for t in (weight, lnw0, lnb0)):
                # inference: both LayerNorms' statistics from tables of the weights (made once per weight version): no reduction per pair
                tables = sm100.stats_tables(weight, lnw0, lnb0)
                return pair_embed_adaln_fwd_s_b200(*tables, gamma, beta, xt, sc, idx, mask, xlim, clim, has_sc, nS, nX, eps1, eps2)
        tbl = _table_b200(weight, train)
        out, bins = pair_embed_adaln_fwd_b200(tbl, lnw, lnb, gamma, beta, xt, sc, idx, mask, xlim, clim, has_sc, nS, nX, eps1, eps2, train)
        if train:
            ctx.save_for_backward(tbl, lnw, lnb, gamma, bins)
            ctx.meta = (nS, nX, eps1, eps2)
        return out

    @staticmethod
    def backward(ctx, g):
        tbl, lnw, lnb, gamma, bins = ctx.saved_tensors
        grads = pair_embed_adaln_bwd_b200(tbl, lnw, lnb, gamma, bins, g.to(torch.bfloat16).contiguous(), *ctx.meta)
        outs = [d.to(dt) for d, dt in zip(grads, ctx.dtypes, strict=True)]
        return (*outs, *([None] * 12))


def pair_embed_adaln_b200(weight, lnw, lnb, gamma, beta, xt, sc, idx, mask, xlim, clim, has_sc, nS, nX, eps1, eps2) -> torch.Tensor:
    """pair [B, n, n, 256] bf16 on a B200, differentiable in ``weight`` [256, NT], ``lnw``, ``lnb``, ``gamma`` and ``beta``."""
    train = torch.is_grad_enabled() and any(t.requires_grad for t in (weight, lnw, lnb, gamma, beta))
    return PairEmbedAdaLNB200Function.apply(weight, lnw, lnb, gamma, beta, xt, sc, idx, mask, xlim, clim, has_sc, nS, nX, eps1, eps2, train)
