"""The CUDA backend of ``PairEmbedGeom`` / ``PairEmbedCA``: the launches of ``sm90`` (``pair_embed_geom.cu``, H100 sm_90a; on an A100 the sm_80 build of the same source, ``sm80``) as opaque ops and an autograd
function. Inputs are the modules' prepared tensors (``modules/pair_embed.py``): the bf16 table ``tbl`` ([317, 256] encoder, [157, 256] decoder, = W^T), fp32
coordinates, uint8 masks, int32 ``idx`` / ``chains`` and the fp32 bin limits. ``eps`` >= 0 folds the attention-bias LayerNorm (no affine, that eps) into
the kernel, ``eps`` < 0 returns the pair itself. The pair comes out bf16; the table gradient bf16, accumulated in fp32 in a fixed order (no atomics)."""

from __future__ import annotations

import torch

from foldjet_engine.kernels._compile import opaque
from foldjet_engine.kernels._sm120 import is_sm120

D_PAIR = 256
N_ROWS = {0: 317, 1: 157}
#: RTX 5090: the fused LayerNorm + table gradient above this many pairs (measured, L2 flushed: L 384 encoder 264 vs 274 us, decoder 172 vs
#: 192; L 256 equal; L 128 the two-kernel path faster, 45 vs 49 / 35 vs 41 us)
LN_FUSED_MIN_PAIRS = 256 * 256


def _ln_eps(eps: float) -> float | None:
    return eps if eps >= 0 else None


def _pair_fake(tbl, idx):
    B, n = idx.shape
    return torch.empty((B, n, n, D_PAIR), dtype=torch.bfloat16, device=tbl.device)


def _pair_codes_fake(tbl, idx):
    """[pair [B, n, n, 256] bf16, codes [B, n, n, 4] int32]."""
    B, n = idx.shape
    return [
        torch.empty((B, n, n, D_PAIR), dtype=torch.bfloat16, device=tbl.device),
        torch.empty((B, n, n, 4), dtype=torch.int32, device=tbl.device),
    ]


def _geom_fwd_fake(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, eps):
    return _pair_fake(tbl, idx)


@opaque(fake=_geom_fwd_fake, name="pair_embed_geom_fwd_sm90")
def pair_embed_geom_fwd(
    tbl: torch.Tensor,
    bb: torch.Tensor,
    ca: torch.Tensor,
    cb: torch.Tensor,
    mask: torch.Tensor,
    idx: torch.Tensor,
    chains: torch.Tensor,
    dlim: torch.Tensor,
    alim: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """The encoder pair [B, n, n, 256] bf16 (forward only: no codes written), or its LayerNorm when ``eps`` >= 0."""
    from foldjet_engine.kernels.pair_embed_geom.cuda.sm80 import device_build

    return device_build().forward(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, ln_eps=_ln_eps(eps))


def _geom_fwd_train_fake(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, eps):
    return _pair_codes_fake(tbl, idx)


@opaque(fake=_geom_fwd_train_fake, name="pair_embed_geom_fwd_train_sm90")
def pair_embed_geom_fwd_train(
    tbl: torch.Tensor,
    bb: torch.Tensor,
    ca: torch.Tensor,
    cb: torch.Tensor,
    mask: torch.Tensor,
    idx: torch.Tensor,
    chains: torch.Tensor,
    dlim: torch.Tensor,
    alim: torch.Tensor,
    eps: float,
) -> list[torch.Tensor]:
    """[pair (or its LayerNorm, ``eps`` >= 0) bf16, codes [B, n, n, 4] int32]: 16 bytes per pair (the packed bins with a live bit, the angle bins, the
    chain product), what the backward replays."""
    from foldjet_engine.kernels.pair_embed_geom.cuda.sm80 import device_build

    return list(device_build().forward(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, with_codes=True, ln_eps=_ln_eps(eps)))


def _ca_fwd_fake(tbl, ca, mask, idx, dlim, eps):
    return _pair_fake(tbl, idx)


@opaque(fake=_ca_fwd_fake, name="pair_embed_ca_fwd_sm90")
def pair_embed_ca_fwd(
    tbl: torch.Tensor, ca: torch.Tensor, mask: torch.Tensor, idx: torch.Tensor, dlim: torch.Tensor, eps: float
) -> torch.Tensor:
    """The decoder pair [B, n, n, 256] bf16 (forward only), or its LayerNorm when ``eps`` >= 0."""
    from foldjet_engine.kernels.pair_embed_geom.cuda.sm80 import device_build

    return device_build().forward_ca(tbl, ca, mask, idx, dlim, ln_eps=_ln_eps(eps))


def _ca_fwd_train_fake(tbl, ca, mask, idx, dlim, eps):
    return _pair_codes_fake(tbl, idx)


@opaque(fake=_ca_fwd_train_fake, name="pair_embed_ca_fwd_train_sm90")
def pair_embed_ca_fwd_train(
    tbl: torch.Tensor, ca: torch.Tensor, mask: torch.Tensor, idx: torch.Tensor, dlim: torch.Tensor, eps: float
) -> list[torch.Tensor]:
    """[decoder pair (or its LayerNorm) bf16, codes [B, n, n, 4] int32] (two live code words)."""
    from foldjet_engine.kernels.pair_embed_geom.cuda.sm80 import device_build

    return list(device_build().forward_ca(tbl, ca, mask, idx, dlim, with_codes=True, ln_eps=_ln_eps(eps)))


def _bwd_fake(g, codes, mode):
    return torch.empty((N_ROWS[mode], D_PAIR), dtype=torch.bfloat16, device=g.device)


@opaque(fake=_bwd_fake, name="pair_embed_geom_bwd_sm90")
def pair_embed_geom_bwd(g: torch.Tensor, codes: torch.Tensor, mode: int) -> torch.Tensor:
    """Table gradient [317 (mode 0) or 157 (mode 1), 256] bf16 from d pair [B, n, n, 256] bf16 (bitwise deterministic)."""
    from foldjet_engine.kernels.pair_embed_geom.cuda.sm80 import device_build

    return device_build().backward(g, codes, mode)


def _ln_bwd_fake(tbl, codes, dh, mode, eps):
    return torch.empty_like(dh)


@opaque(fake=_ln_bwd_fake, name="pair_embed_geom_ln_bwd_sm90")
def pair_embed_geom_ln_bwd(tbl: torch.Tensor, codes: torch.Tensor, dh: torch.Tensor, mode: int, eps: float) -> torch.Tensor:
    """d pair [B, n, n, 256] bf16 from d LayerNorm(pair) (recomputes the pair from the codes)."""
    from foldjet_engine.kernels.pair_embed_geom.cuda.sm80 import device_build

    return device_build().ln_backward(tbl, codes, dh, mode, eps)


def _eval_build():
    """The build of the one-launch inference forwards: this card's device build; on an H100 the sm_80 source built for sm_90a (the training launchers stay the sm90 build)."""
    from foldjet_engine.kernels.pair_embed_geom.cuda import sm80

    return sm80 if torch.cuda.get_device_capability() == (9, 0) else sm80.device_build()


def _eval_fake(weight, a, *rest):
    B, n = a.shape[:2]
    return torch.empty((B, n, n, D_PAIR), dtype=torch.bfloat16, device=a.device)


@opaque(fake=_eval_fake, name="pair_embed_geom_eval_sm120")
def pair_embed_geom_eval_sm120(
    weight: torch.Tensor,
    bb: torch.Tensor,
    ca_mask: torch.Tensor,
    has_cb: torch.Tensor,
    mask: torch.Tensor,
    residue_idx: torch.Tensor,
    chains: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """RTX 5090 inference: the encoder pair (its LayerNorm when ``eps`` >= 0) straight from the module's inputs and weight (sm80/__init__.py forward_eval)."""
    return _eval_build().forward_eval(weight, bb, ca_mask, has_cb, mask, residue_idx, chains, eps)


@opaque(fake=_eval_fake, name="pair_embed_ca_eval_sm120")
def pair_embed_ca_eval_sm120(weight: torch.Tensor, ca: torch.Tensor, mask: torch.Tensor, residue_idx: torch.Tensor, eps: float) -> torch.Tensor:
    """RTX 5090 inference: the decoder pair (its LayerNorm when ``eps`` >= 0) straight from the module's inputs and weight (sm80/__init__.py forward_ca_eval)."""
    return _eval_build().forward_ca_eval(weight, ca, mask, residue_idx, eps)


@opaque(fake=_bwd_fake, name="pair_embed_geom_ln_tbl_bwd_sm120")
def pair_embed_geom_ln_tbl_bwd(g: torch.Tensor, codes: torch.Tensor, tbl: torch.Tensor, mode: int, eps: float) -> torch.Tensor:
    """RTX 5090: the table gradient straight from d LayerNorm(pair) = ``g`` (the LayerNorm backward fused in; sm80/__init__.py ln_table_grad)."""
    from foldjet_engine.kernels.pair_embed_geom.cuda.sm80 import ln_table_grad

    return ln_table_grad(tbl, codes, g, mode, eps)


class PairEmbedFunction(torch.autograd.Function):
    """``mode`` 0: the encoder pair (``a`` = bb [B, n, 4, 3]); 1: the decoder pair (``a`` = CA [B, n, 3]; ``cb`` / ``chains`` / ``alim`` unused)."""

    @staticmethod
    def forward(ctx, tbl, mode, a, ca, cb, mask, idx, chains, dlim, alim, eps):
        if mode == 0:
            out, codes = pair_embed_geom_fwd_train(tbl, a, ca, cb, mask, idx, chains, dlim, alim, eps)
        else:
            out, codes = pair_embed_ca_fwd_train(tbl, a, mask, idx, dlim, eps)
        ctx.save_for_backward(codes, tbl)
        ctx.mode, ctx.eps = mode, eps
        return out

    @staticmethod
    def backward(ctx, g):
        codes, tbl = ctx.saved_tensors
        g = g.to(torch.bfloat16).contiguous()
        if ctx.eps >= 0 and is_sm120(g.device) and g.shape[0] * g.shape[1] * g.shape[2] > LN_FUSED_MIN_PAIRS:
            # RTX 5090: the LayerNorm backward fused into the table gradient
            return (pair_embed_geom_ln_tbl_bwd(g, codes, tbl, ctx.mode, ctx.eps), *([None] * 10))
        if ctx.eps >= 0:
            g = pair_embed_geom_ln_bwd(tbl, codes, g, ctx.mode, ctx.eps)
        return (pair_embed_geom_bwd(g, codes, ctx.mode), *([None] * 10))


def pair_embed_geom_cuda(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, eps, train: bool) -> torch.Tensor:
    """The encoder pair [B, n, n, 256] bf16 (LayerNorm of it when ``eps`` >= 0); ``train``: differentiable in ``tbl``."""
    if train:
        return PairEmbedFunction.apply(tbl, 0, bb, ca, cb, mask, idx, chains, dlim, alim, eps)
    return pair_embed_geom_fwd(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, eps)


def pair_embed_ca_cuda(tbl, ca_xyz, mask, idx, dlim, eps, train: bool) -> torch.Tensor:
    """The decoder pair [B, n, n, 256] bf16 (LayerNorm of it when ``eps`` >= 0); ``train``: differentiable in ``tbl``."""
    if train:
        return PairEmbedFunction.apply(tbl, 1, ca_xyz, None, None, mask, idx, None, dlim, None, eps)
    return pair_embed_ca_fwd(tbl, ca_xyz, mask, idx, dlim, eps)


# ---- B200 (sm_100a): kernels/pair_embed_geom/cuda/sm100 round-3 launches (2026-11-17) from the module's weight and inputs ------------------
def _b200_fwd_fake(tbl, a, ca, cb, mask, idx, chains, dlim, alim, eps, mode, train):
    B, n = idx.shape
    return [
        torch.empty((B, n, n, D_PAIR), dtype=torch.bfloat16, device=a.device),
        torch.empty((B, n, n, 4) if train else (0,), dtype=torch.int32, device=a.device),
        torch.empty((N_ROWS[mode], D_PAIR) if tbl.shape[0] == D_PAIR else (0,), dtype=torch.bfloat16, device=a.device),
    ]


@opaque(fake=_b200_fwd_fake, name="pair_embed_fwd_sm100")
def pair_embed_fwd_b200(
    tbl: torch.Tensor,
    a: torch.Tensor,
    ca: torch.Tensor | None,
    cb: torch.Tensor | None,
    mask: torch.Tensor,
    idx: torch.Tensor,
    chains: torch.Tensor | None,
    dlim: torch.Tensor,
    alim: torch.Tensor | None,
    eps: float,
    mode: int,
    train: bool,
) -> list[torch.Tensor]:
    """[pair (its LayerNorm when ``eps`` >= 0) [B, n, n, 256] bf16, codes [B, n, n, 4] int32 (empty unless ``train``), the bf16 table when the
    launch made it (else empty)]; mode 0: the encoder pair from its bf16 table and ``a`` = bb, mode 1: the decoder pair from its bf16 table --
    or from the weight [256, 157] itself: the launch makes the table -- and ``a`` = CA."""
    from foldjet_engine.kernels.pair_embed_geom.cuda import sm100

    if mode == 0:
        out, codes, made = sm100.forward_b200(tbl, a, ca, cb, mask, idx, chains, dlim, alim, eps, train)
        return [out, codes, made if tbl.shape[0] == D_PAIR else torch.empty(0, device=a.device, dtype=torch.bfloat16)]
    return list(sm100.forward_ca_b200(tbl, a, mask, idx, dlim, eps, train))


def _b200_ln_bwd_fake(tbl, codes, dh, mode, eps):
    return torch.empty_like(dh)


@opaque(fake=_b200_ln_bwd_fake, name="pair_embed_ln_bwd_sm100")
def pair_embed_ln_bwd_b200(tbl: torch.Tensor, codes: torch.Tensor, dh: torch.Tensor, mode: int, eps: float) -> torch.Tensor:
    """d pair [B, n, n, 256] bf16 from d LayerNorm(pair)."""
    from foldjet_engine.kernels.pair_embed_geom.cuda import sm100

    return sm100.ln_backward_b200(tbl, codes, dh, mode, eps)


def _b200_bwd_fake(g, codes, mode):
    return torch.empty((D_PAIR, N_ROWS[mode]), dtype=torch.float32, device=g.device)


@opaque(fake=_b200_bwd_fake, name="pair_embed_bwd_sm100")
def pair_embed_bwd_b200(g: torch.Tensor, codes: torch.Tensor, mode: int) -> torch.Tensor:
    """The weight gradient [256, rows] fp32 (bf16-rounded entries) from d pair [B, n, n, 256] bf16 (bitwise deterministic)."""
    from foldjet_engine.kernels.pair_embed_geom.cuda import sm100

    return sm100.backward_w(g, codes, mode)


class PairEmbedB200Function(torch.autograd.Function):
    """The B200 route of both builders: takes the module's ``weight`` [256, rows] itself (the table is made here, its gradient comes back in
    the weight's layout: no transposed copies in the autograd graph). ``mode`` 0: encoder (``a`` = bb), 1: decoder (``a`` = CA)."""

    @staticmethod
    def forward(ctx, weight, mode, a, ca, cb, mask, idx, chains, dlim, alim, eps, train):
        from foldjet_engine.kernels.pair_embed_geom.cuda import sm100

        if train and sm100.weight_launch(weight):
            tbl = weight.detach()  # the launch makes the table itself
        else:
            tbl = sm100.table(weight, train)
        out, codes, made = pair_embed_fwd_b200(tbl, a, ca, cb, mask, idx, chains, dlim, alim, eps, mode, train)
        if made.numel():
            tbl = made
        if train:
            ctx.save_for_backward(codes, tbl)
            ctx.mode, ctx.eps, ctx.wdtype = mode, eps, weight.dtype
        return out

    @staticmethod
    def backward(ctx, g):
        codes, tbl = ctx.saved_tensors
        g = g.to(torch.bfloat16).contiguous()
        if ctx.eps >= 0:
            g = pair_embed_ln_bwd_b200(tbl, codes, g, ctx.mode, ctx.eps)
        return (pair_embed_bwd_b200(g, codes, ctx.mode).to(ctx.wdtype), *([None] * 11))


def pair_embed_geom_b200(weight, bb, ca_mask, has_cb, mask, residue_idx, chains, eps) -> torch.Tensor:
    """The encoder pair [B, n, n, 256] bf16 on a B200 (LayerNorm of it when ``eps`` >= 0), differentiable in ``weight`` [256, 317]."""
    from foldjet_engine.kernels.pair_embed_geom.cuda import sm80

    dev = bb.device
    idx, ch = sm80._indices(residue_idx, chains)
    return PairEmbedB200Function.apply(
        weight, 0, bb.float().contiguous(), sm80._bytes(ca_mask), sm80._bytes(has_cb), sm80._bytes(mask), idx, ch, sm80._limits(dev, "dist"),
        sm80._limits(dev, "angle"), eps, torch.is_grad_enabled() and weight.requires_grad,
    )  # fmt: skip


def pair_embed_ca_b200(weight, ca_xyz, mask, residue_idx, eps) -> torch.Tensor:
    """The decoder pair [B, n, n, 256] bf16 on a B200 (LayerNorm of it when ``eps`` >= 0), differentiable in ``weight`` [256, 157]."""
    from foldjet_engine.kernels.pair_embed_geom.cuda import sm80

    dev = ca_xyz.device
    if residue_idx is None:
        b, n = mask.shape
        residue_idx = torch.arange(1, n + 1, device=dev, dtype=torch.int32).expand(b, n).contiguous()
    (idx,) = sm80._indices(residue_idx)
    return PairEmbedB200Function.apply(
        weight, 1, ca_xyz.float().contiguous(), None, None, sm80._bytes(mask), idx, None, sm80._limits(dev, "ca"), None, eps,
        torch.is_grad_enabled() and weight.requires_grad,
    )
