"""Fused backbone pair embedding (kernel family ``pair_embed_geom``): La-Proteina's autoencoder-encoder pair representation

    pair[b,i,j] = M_ij * W [onehot_sep | onehot(CA_i - N_j, CA_j, C_j, CB_j) | onehot(5 inter-residue angles) | chain_i chain_j]

as a row gather of up to ten table rows in shared memory instead of 316 fp32 one-hot channels per pair, a concatenation and a Linear.
:func:`pair_embed_geom` is the forward-only form (the encoder is frozen in the flow stage; the output is not differentiable); :func:`pair_embed_geom_train` is differentiable in the
weight (autoencoder training) and :func:`pair_embed_ca` is the autoencoder decoder's pair embedding (sequence separation + CA distances, 157 rows); both can fold the attention-bias
LayerNorm into the kernel (``ln_eps``). bf16 pair out.  256 channels.  Served: H100 by the CUDA kernels, any other card by the portable kernels (``triton/``, a gather per pair too, a deterministic
table-gradient reduction) when the engine holds a record for the card, else by the PyTorch reference with the standard warning (``kernels/_pair_route``);
:func:`refusal` says why a call is not served at all; it never raises.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._pair_route import CUDA, REFERENCE, restricted_reason, route
from miniworld_engine.kernels.pair_embed_geom.reference import pair_embed_ca_reference, pair_embed_geom_reference

D_PAIR = 256
N_ROWS = 317
_CUDA_CAPS = ((9, 0),)

_CONSTANTS: dict[tuple, torch.Tensor] = {}


def _constant(key: tuple, make) -> torch.Tensor:
    """A small device constant (bin limits, the default residue index) built once per (key, device) and reused: every call of the builder used to launch its own
    ``linspace`` / ``arange`` / cast kernels, which at B16 x L128 (a 60 us step) is a sixth of it. Read-only (the kernels never write them). Built only outside a CUDA
    graph capture (a tensor allocated inside one belongs to the graph pool); a capture that finds none builds it as before."""
    if torch.compiler.is_compiling():                 # traced: the constant folds into the compiled graph
        return make()
    t = _CONSTANTS.get(key)
    if t is None:
        t = make()
        if not torch.cuda.is_current_stream_capturing():
            _CONSTANTS[key] = t
    return t


def refusal(w, bb, ca_mask, has_cb, mask, residue_idx, chains, *, train=False) -> str | None:
    if not bb.is_cuda:
        return "the input is not on a CUDA device"
    if (why := restricted_reason(bb.device, _CUDA_CAPS)) is not None:
        return why
    if w.shape[0] != D_PAIR or w.shape[1] < N_ROWS:
        return f"w must be [{D_PAIR}, >= {N_ROWS}], got {tuple(w.shape)}"
    if bb.dim() != 4 or tuple(bb.shape[2:]) != (4, 3):
        return "bb must be [B, n, 4, 3] (N, CA, C, CB)"
    if torch.is_grad_enabled() and w.requires_grad and not train:
        return "forward only: the weight needs a gradient (use the training entry point)"
    return None


def _pair_embed_geom_fwd_fake(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, portable):
    """pair [B, n, n, 256] bf16."""
    B, n = idx.shape
    return torch.empty((B, n, n, D_PAIR), dtype=torch.bfloat16, device=tbl.device)


@opaque(fake=_pair_embed_geom_fwd_fake, name="pair_embed_geom_fwd")
def pair_embed_geom_fwd(tbl: torch.Tensor, bb: torch.Tensor, ca: torch.Tensor, cb: torch.Tensor, mask: torch.Tensor, idx: torch.Tensor,
                        chains: torch.Tensor, dlim: torch.Tensor, alim: torch.Tensor, portable: int) -> torch.Tensor:
    """pair [B, n, n, 256] bf16 from the bf16 table [317, 256] and the per-residue inputs."""
    if portable:
        from miniworld_engine.kernels.pair_embed_geom.triton import portable_forward

        return portable_forward(tbl, bb, ca, cb, mask, idx, chains, dlim, alim)
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm90

    return sm90.forward(tbl, bb, ca, cb, mask, idx, chains, dlim, alim)


def pair_embed_geom(w, bb, ca_mask, has_cb, mask, residue_idx, chains) -> torch.Tensor:
    """pair [B, n, n, 256] bf16 (not differentiable). ``w`` [256, >= 317] Linear weight (columns: sep 127 | 4 x 21 distances | 5 x 21 angles |
    chain), ``bb`` [B, n, 4, 3] fp32 nm, the masks [B, n] bool, ``residue_idx`` / ``chains`` [B, n] integers. Check :func:`refusal` first."""
    dev = bb.device
    where = route("pair_embed_geom", dev)
    if where == REFERENCE:
        with torch.no_grad():
            return pair_embed_geom_reference(w.detach(), bb, ca_mask.bool(), has_cb.bool(), mask.bool(), residue_idx, chains).to(torch.bfloat16)
    tbl = w[:, :N_ROWS].detach().t().to(torch.bfloat16).contiguous()
    dlim = torch.linspace(0.1, 2, 20, device=dev)
    alim = torch.linspace(-torch.pi, torch.pi, 20, device=dev)
    return pair_embed_geom_fwd(tbl, bb.float().contiguous(), ca_mask.to(torch.uint8).contiguous(), has_cb.to(torch.uint8).contiguous(),
                               mask.to(torch.uint8).contiguous(), residue_idx.to(torch.int32).contiguous(), chains.to(torch.int32).contiguous(),
                               dlim, alim, int(where != CUDA))


# ---------------------------------------------------------------------------------------------------------------------------------------
# Training entry points: the same pair, differentiable in the weight (the binned features are inputs, not functions of any parameter).
# ---------------------------------------------------------------------------------------------------------------------------------------
N_ROWS_CA = 157


def _pair_embed_geom_fwd_train_fake(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, portable):
    """[pair [B, n, n, 256] bf16, codes [B, n, n, 4] int32]."""
    B, n = idx.shape
    return [torch.empty((B, n, n, D_PAIR), dtype=torch.bfloat16, device=tbl.device), torch.empty((B, n, n, 4), dtype=torch.int32, device=tbl.device)]


@opaque(fake=_pair_embed_geom_fwd_train_fake, name="pair_embed_geom_fwd_train")
def pair_embed_geom_fwd_train(tbl: torch.Tensor, bb: torch.Tensor, ca: torch.Tensor, cb: torch.Tensor, mask: torch.Tensor, idx: torch.Tensor,
                              chains: torch.Tensor, dlim: torch.Tensor, alim: torch.Tensor, portable: int) -> list[torch.Tensor]:
    """[pair [B, n, n, 256] bf16, codes [B, n, n, 4] int32] (16 bytes per pair: the packed bins with a live bit, the angle bins, the chain product; what the backward replays)."""
    if portable:
        from miniworld_engine.kernels.pair_embed_geom.triton import portable_forward

        return list(portable_forward(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, with_codes=True))
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm90

    return list(sm90.forward(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, with_codes=True))


def _pair_embed_geom_ca_fwd_fake(tbl, ca, mask, idx, dlim, portable):
    """[pair [B, n, n, 256] bf16, codes [B, n, n, 4] int32] (the decoder pair keeps two live code words)."""
    B, n = idx.shape
    return [torch.empty((B, n, n, D_PAIR), dtype=torch.bfloat16, device=tbl.device), torch.empty((B, n, n, 4), dtype=torch.int32, device=tbl.device)]


@opaque(fake=_pair_embed_geom_ca_fwd_fake, name="pair_embed_geom_ca_fwd")
def pair_embed_geom_ca_fwd(tbl: torch.Tensor, ca: torch.Tensor, mask: torch.Tensor, idx: torch.Tensor, dlim: torch.Tensor,
                           portable: int) -> list[torch.Tensor]:
    """[pair [B, n, n, 256] bf16, codes [B, n, n, 2] int32] from the decoder table [157, 256] bf16."""
    if portable:
        from miniworld_engine.kernels.pair_embed_geom.triton import portable_forward_ca

        return list(portable_forward_ca(tbl, ca, mask, idx, dlim, with_codes=True))
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm90

    return list(sm90.forward_ca(tbl, ca, mask, idx, dlim, with_codes=True))


def _pair_embed_geom_bwd_fake(g, codes, mode, portable):
    """Table gradient [317 or 157, 256] bf16, by mode."""
    return torch.empty((N_ROWS if mode == 0 else N_ROWS_CA, D_PAIR), dtype=torch.bfloat16, device=g.device)


@opaque(fake=_pair_embed_geom_bwd_fake, name="pair_embed_geom_bwd")
def pair_embed_geom_bwd(g: torch.Tensor, codes: torch.Tensor, mode: int, portable: int) -> torch.Tensor:
    """Table gradient [317 or 157, 256] bf16 from d pair [B, n, n, 256] bf16 (bitwise deterministic: fixed-order partial sums, no atomics)."""
    if portable:
        from miniworld_engine.kernels.pair_embed_geom.triton import portable_backward

        return portable_backward(g, codes, mode)
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm90

    return sm90.backward(g, codes, mode)


def _pair_embed_geom_fwd_ln_fake(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, eps, portable):
    """As the training forward: [LayerNorm(pair) bf16, codes int32]."""
    return _pair_embed_geom_fwd_train_fake(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, portable)


@opaque(fake=_pair_embed_geom_fwd_ln_fake, name="pair_embed_geom_fwd_ln")
def pair_embed_geom_fwd_ln(tbl: torch.Tensor, bb: torch.Tensor, ca: torch.Tensor, cb: torch.Tensor, mask: torch.Tensor, idx: torch.Tensor,
                           chains: torch.Tensor, dlim: torch.Tensor, alim: torch.Tensor, eps: float, portable: int) -> list[torch.Tensor]:
    """[LayerNorm(pair) [B, n, n, 256] bf16, codes]: the pair is normalised over its 256 channels (no affine) inside the kernel and never stored."""
    if portable:
        from miniworld_engine.kernels.pair_embed_geom.triton import portable_forward

        return list(portable_forward(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, with_codes=True, ln_eps=eps))
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm90

    return list(sm90.forward(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, with_codes=True, ln_eps=eps))


def _pair_embed_geom_ca_fwd_ln_fake(tbl, ca, mask, idx, dlim, eps, portable):
    """As the decoder forward: [LayerNorm(pair) bf16, codes int32]."""
    return _pair_embed_geom_ca_fwd_fake(tbl, ca, mask, idx, dlim, portable)


@opaque(fake=_pair_embed_geom_ca_fwd_ln_fake, name="pair_embed_geom_ca_fwd_ln")
def pair_embed_geom_ca_fwd_ln(tbl: torch.Tensor, ca: torch.Tensor, mask: torch.Tensor, idx: torch.Tensor, dlim: torch.Tensor, eps: float,
                              portable: int) -> list[torch.Tensor]:
    """[LayerNorm(pair) [B, n, n, 256] bf16 (no affine, ``eps``), codes]: the decoder pair, normalised inside the kernel and never stored."""
    if portable:
        from miniworld_engine.kernels.pair_embed_geom.triton import portable_forward_ca

        return list(portable_forward_ca(tbl, ca, mask, idx, dlim, with_codes=True, ln_eps=eps))
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm90

    return list(sm90.forward_ca(tbl, ca, mask, idx, dlim, with_codes=True, ln_eps=eps))


def _pair_embed_geom_ln_bwd_fake(tbl, codes, dh, mode, eps, portable):
    """d pair, shaped like d LayerNorm(pair)."""
    return torch.empty_like(dh)


@opaque(fake=_pair_embed_geom_ln_bwd_fake, name="pair_embed_geom_ln_bwd")
def pair_embed_geom_ln_bwd(tbl: torch.Tensor, codes: torch.Tensor, dh: torch.Tensor, mode: int, eps: float, portable: int) -> torch.Tensor:
    """d pair [B, n, n, 256] bf16 from d LayerNorm(pair) (recomputes the pair from the codes)."""
    if portable:
        from miniworld_engine.kernels.pair_embed_geom.triton import portable_ln_backward

        return portable_ln_backward(tbl, codes, dh, mode, eps)
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm90

    return sm90.ln_backward(tbl, codes, dh, mode, eps)


class _PairEmbedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tbl, mode, bb, ca, cb, mask, idx, chains, dlim, alim, ln_eps, portable):
        if mode == 0:
            out, codes = (pair_embed_geom_fwd_train(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, portable) if ln_eps is None
                          else pair_embed_geom_fwd_ln(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, ln_eps, portable))
        else:
            out, codes = (pair_embed_geom_ca_fwd(tbl, bb, mask, idx, dlim, portable) if ln_eps is None
                          else pair_embed_geom_ca_fwd_ln(tbl, bb, mask, idx, dlim, ln_eps, portable))   # `bb` is the CA coordinates [B, n, 3] here
        ctx.save_for_backward(codes, tbl)
        ctx.mode, ctx.ln_eps, ctx.portable = mode, ln_eps, portable
        return out

    @staticmethod
    def backward(ctx, g):
        codes, tbl = ctx.saved_tensors
        g = g.to(torch.bfloat16).contiguous()
        if ctx.ln_eps is not None:
            g = pair_embed_geom_ln_bwd(tbl, codes, g, ctx.mode, ctx.ln_eps, ctx.portable)
        dtbl = pair_embed_geom_bwd(g, codes, ctx.mode, ctx.portable)
        return (dtbl, None, None, None, None, None, None, None, None, None, None, None)


def pair_embed_geom_train(w, bb, ca_mask, has_cb, mask, residue_idx, chains, *, ln_eps=None) -> torch.Tensor:
    """As :func:`pair_embed_geom`, differentiable in ``w`` ([256, >= 317]): the gradient is that of a bf16-autocast Linear over the binned
    features (the weight is rounded to bf16 by an ordinary autograd op, the table gradient is accumulated in fp32 in a fixed order and
    rounded to bf16 once). Use under ``torch.autocast(bfloat16)``: the output is bf16. ``ln_eps``: return LayerNorm(pair) over the 256 channels
    (no affine, that eps; the pair is never stored and the backward recomputes it) -- what the engine's attention-bias projection reads."""
    dev = bb.device
    where = route("pair_embed_geom", dev, w)
    if where == REFERENCE:
        out = pair_embed_geom_reference(w, bb, ca_mask.bool(), has_cb.bool(), mask.bool(), residue_idx, chains).to(torch.bfloat16)
        return out if ln_eps is None else F.layer_norm(out, (D_PAIR,), None, None, ln_eps)
    tbl = w[:, :N_ROWS].t().to(torch.bfloat16).contiguous()
    dlim = torch.linspace(0.1, 2, 20, device=dev)
    alim = torch.linspace(-torch.pi, torch.pi, 20, device=dev)
    return _PairEmbedFn.apply(tbl, 0, bb.float().contiguous(), ca_mask.to(torch.uint8).contiguous(), has_cb.to(torch.uint8).contiguous(),
                              mask.to(torch.uint8).contiguous(), residue_idx.to(torch.int32).contiguous(), chains.to(torch.int32).contiguous(), dlim, alim, ln_eps, int(where != CUDA))


def refusal_ca(w, ca, mask, residue_idx) -> str | None:
    if not ca.is_cuda:
        return "the input is not on a CUDA device"
    if (why := restricted_reason(ca.device, _CUDA_CAPS)) is not None:
        return why
    if w.shape[0] != D_PAIR or w.shape[1] != N_ROWS_CA:
        return f"w must be [{D_PAIR}, {N_ROWS_CA}] (127 sequence separations | 30 CA distance bins), got {tuple(w.shape)}"
    if ca.dim() != 3 or ca.shape[-1] != 3 or mask.shape != ca.shape[:2]:
        return "ca must be [B, n, 3] and mask [B, n]"
    return None


def pair_embed_ca(w, ca, mask, residue_idx=None, *, ln_eps=None) -> torch.Tensor:
    """Decoder pair [B, n, n, 256] bf16 = M_ij * W [onehot_sep | onehot_dist30(|ca_i - ca_j|)], differentiable in ``w`` [256, 157].
    ``ca`` [B, n, 3] fp32 nm (the caller zeroes padded residues, as the decoder does), ``residue_idx`` defaults to 1 .. n."""
    dev = ca.device
    B, n = mask.shape
    where = route("pair_embed_ca", dev, w)
    if where == REFERENCE:
        out = pair_embed_ca_reference(w, ca, mask.bool(), residue_idx).to(torch.bfloat16)
        return out if ln_eps is None else F.layer_norm(out, (D_PAIR,), None, None, ln_eps)
    idx = residue_idx.to(torch.int32).contiguous() if residue_idx is not None else _constant(("ca_idx", B, n, str(dev)), lambda: torch.arange(1, n + 1, device=dev, dtype=torch.int32).expand(B, n).contiguous())
    tbl = w.t().to(torch.bfloat16).contiguous()
    dlim = _constant(("ca_dlim", str(dev)), lambda: torch.linspace(0.1, 3.0, 29, device=dev))
    # a bool mask is already 0 / 1 bytes: no cast kernel
    m8 = mask.contiguous().view(torch.uint8) if mask.dtype is torch.bool else mask.to(torch.uint8).contiguous()
    return _PairEmbedFn.apply(tbl, 1, ca.float().contiguous(), None, None, m8, idx, None, dlim, None, ln_eps,
                              int(where != CUDA))


__all__ = ["D_PAIR", "refusal", "pair_embed_geom", "pair_embed_geom_train", "pair_embed_ca", "refusal_ca"]
