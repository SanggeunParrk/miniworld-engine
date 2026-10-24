"""Fused backbone pair embedding (kernel family ``pair_embed_geom``): La-Proteina's autoencoder-encoder pair representation

    pair[b,i,j] = M_ij * W [onehot_sep | onehot(CA_i - N_j, CA_j, C_j, CB_j) | onehot(5 inter-residue angles) | chain_i chain_j]

as a row gather of up to ten table rows in shared memory instead of 316 fp32 one-hot channels per pair, a concatenation and a Linear.
:func:`pair_embed_geom` is the forward-only form (the encoder is frozen in the flow stage; the output is not differentiable); :func:`pair_embed_geom_train` is differentiable in the
weight (autoencoder training) and :func:`pair_embed_ca` is the autoencoder decoder's pair embedding (sequence separation + CA distances, 157 rows); both can fold the attention-bias
LayerNorm into the kernel (``ln_eps``). bf16 pair out.  Served: H100, 256 channels. :func:`refusal` says why anything else is not served; it never raises.
"""
from __future__ import annotations

import torch

from miniworld_engine.kernels._compile import opaque

D_PAIR = 256
N_ROWS = 317


def refusal(w, bb, ca_mask, has_cb, mask, residue_idx, chains, *, train=False) -> str | None:
    if not bb.is_cuda:
        return "the input is not on a CUDA device"
    if torch.cuda.get_device_capability(bb.device) != (9, 0):
        return "served on H100 (sm_90a) only"
    if w.shape[0] != D_PAIR or w.shape[1] < N_ROWS:
        return f"w must be [{D_PAIR}, >= {N_ROWS}], got {tuple(w.shape)}"
    if bb.dim() != 4 or tuple(bb.shape[2:]) != (4, 3):
        return "bb must be [B, n, 4, 3] (N, CA, C, CB)"
    if torch.is_grad_enabled() and w.requires_grad and not train:
        return "forward only: the weight needs a gradient (use the training entry point)"
    return None


def _pair_embed_geom_fwd_fake(tbl, bb, ca, cb, mask, idx, chains, dlim, alim):
    """pair [B, n, n, 256] bf16."""
    B, n = idx.shape
    return torch.empty((B, n, n, D_PAIR), dtype=torch.bfloat16, device=tbl.device)


@opaque(fake=_pair_embed_geom_fwd_fake, name="pair_embed_geom_fwd")
def pair_embed_geom_fwd(tbl: torch.Tensor, bb: torch.Tensor, ca: torch.Tensor, cb: torch.Tensor, mask: torch.Tensor, idx: torch.Tensor,
                        chains: torch.Tensor, dlim: torch.Tensor, alim: torch.Tensor) -> torch.Tensor:
    """pair [B, n, n, 256] bf16 from the bf16 table [317, 256] and the per-residue inputs."""
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm90

    return sm90.forward(tbl, bb, ca, cb, mask, idx, chains, dlim, alim)


def pair_embed_geom(w, bb, ca_mask, has_cb, mask, residue_idx, chains) -> torch.Tensor:
    """pair [B, n, n, 256] bf16 (not differentiable). ``w`` [256, >= 317] Linear weight (columns: sep 127 | 4 x 21 distances | 5 x 21 angles |
    chain), ``bb`` [B, n, 4, 3] fp32 nm, the masks [B, n] bool, ``residue_idx`` / ``chains`` [B, n] integers. Check :func:`refusal` first."""
    dev = bb.device
    tbl = w[:, :N_ROWS].detach().t().to(torch.bfloat16).contiguous()
    dlim = torch.linspace(0.1, 2, 20, device=dev)
    alim = torch.linspace(-torch.pi, torch.pi, 20, device=dev)
    return pair_embed_geom_fwd(tbl, bb.float().contiguous(), ca_mask.to(torch.uint8).contiguous(), has_cb.to(torch.uint8).contiguous(),
                               mask.to(torch.uint8).contiguous(), residue_idx.to(torch.int32).contiguous(), chains.to(torch.int32).contiguous(),
                               dlim, alim)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Training entry points: the same pair, differentiable in the weight (the binned features are inputs, not functions of any parameter).
# ---------------------------------------------------------------------------------------------------------------------------------------
N_ROWS_CA = 157


def _pair_embed_geom_fwd_train_fake(tbl, bb, ca, cb, mask, idx, chains, dlim, alim):
    """[pair [B, n, n, 256] bf16, codes [B, n, n, 4] int32]."""
    B, n = idx.shape
    return [torch.empty((B, n, n, D_PAIR), dtype=torch.bfloat16, device=tbl.device), torch.empty((B, n, n, 4), dtype=torch.int32, device=tbl.device)]


@opaque(fake=_pair_embed_geom_fwd_train_fake, name="pair_embed_geom_fwd_train")
def pair_embed_geom_fwd_train(tbl: torch.Tensor, bb: torch.Tensor, ca: torch.Tensor, cb: torch.Tensor, mask: torch.Tensor, idx: torch.Tensor,
                              chains: torch.Tensor, dlim: torch.Tensor, alim: torch.Tensor) -> list[torch.Tensor]:
    """[pair [B, n, n, 256] bf16, codes [B, n, n, 4] int32] (16 bytes per pair: the packed bins with a live bit, the angle bins, the chain product; what the backward replays)."""
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm90

    return list(sm90.forward(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, with_codes=True))


def _pair_embed_geom_ca_fwd_fake(tbl, ca, mask, idx, dlim):
    """[pair [B, n, n, 256] bf16, codes [B, n, n, 4] int32] (the decoder pair keeps two live code words)."""
    B, n = idx.shape
    return [torch.empty((B, n, n, D_PAIR), dtype=torch.bfloat16, device=tbl.device), torch.empty((B, n, n, 4), dtype=torch.int32, device=tbl.device)]


@opaque(fake=_pair_embed_geom_ca_fwd_fake, name="pair_embed_geom_ca_fwd")
def pair_embed_geom_ca_fwd(tbl: torch.Tensor, ca: torch.Tensor, mask: torch.Tensor, idx: torch.Tensor, dlim: torch.Tensor) -> list[torch.Tensor]:
    """[pair [B, n, n, 256] bf16, codes [B, n, n, 2] int32] from the decoder table [157, 256] bf16."""
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm90

    return list(sm90.forward_ca(tbl, ca, mask, idx, dlim, with_codes=True))


def _pair_embed_geom_bwd_fake(g, codes, mode):
    """Table gradient [317 or 157, 256] bf16, by mode."""
    return torch.empty((N_ROWS if mode == 0 else N_ROWS_CA, D_PAIR), dtype=torch.bfloat16, device=g.device)


@opaque(fake=_pair_embed_geom_bwd_fake, name="pair_embed_geom_bwd")
def pair_embed_geom_bwd(g: torch.Tensor, codes: torch.Tensor, mode: int) -> torch.Tensor:
    """Table gradient [317 or 157, 256] bf16 from d pair [B, n, n, 256] bf16 (bitwise deterministic: fixed-order partial sums, no atomics)."""
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm90

    return sm90.backward(g, codes, mode)


def _pair_embed_geom_fwd_ln_fake(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, eps):
    """As the training forward: [LayerNorm(pair) bf16, codes int32]."""
    return _pair_embed_geom_fwd_train_fake(tbl, bb, ca, cb, mask, idx, chains, dlim, alim)


@opaque(fake=_pair_embed_geom_fwd_ln_fake, name="pair_embed_geom_fwd_ln")
def pair_embed_geom_fwd_ln(tbl: torch.Tensor, bb: torch.Tensor, ca: torch.Tensor, cb: torch.Tensor, mask: torch.Tensor, idx: torch.Tensor,
                           chains: torch.Tensor, dlim: torch.Tensor, alim: torch.Tensor, eps: float) -> list[torch.Tensor]:
    """[LayerNorm(pair) [B, n, n, 256] bf16, codes]: the pair is normalised over its 256 channels (no affine) inside the kernel and never stored."""
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm90

    return list(sm90.forward(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, with_codes=True, ln_eps=eps))


def _pair_embed_geom_ca_fwd_ln_fake(tbl, ca, mask, idx, dlim, eps):
    """As the decoder forward: [LayerNorm(pair) bf16, codes int32]."""
    return _pair_embed_geom_ca_fwd_fake(tbl, ca, mask, idx, dlim)


@opaque(fake=_pair_embed_geom_ca_fwd_ln_fake, name="pair_embed_geom_ca_fwd_ln")
def pair_embed_geom_ca_fwd_ln(tbl: torch.Tensor, ca: torch.Tensor, mask: torch.Tensor, idx: torch.Tensor, dlim: torch.Tensor, eps: float) -> list[torch.Tensor]:
    """[LayerNorm(pair) [B, n, n, 256] bf16 (no affine, ``eps``), codes]: the decoder pair, normalised inside the kernel and never stored."""
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm90

    return list(sm90.forward_ca(tbl, ca, mask, idx, dlim, with_codes=True, ln_eps=eps))


def _pair_embed_geom_ln_bwd_fake(tbl, codes, dh, mode, eps):
    """d pair, shaped like d LayerNorm(pair)."""
    return torch.empty_like(dh)


@opaque(fake=_pair_embed_geom_ln_bwd_fake, name="pair_embed_geom_ln_bwd")
def pair_embed_geom_ln_bwd(tbl: torch.Tensor, codes: torch.Tensor, dh: torch.Tensor, mode: int, eps: float) -> torch.Tensor:
    """d pair [B, n, n, 256] bf16 from d LayerNorm(pair) (recomputes the pair from the codes)."""
    from miniworld_engine.kernels.pair_embed_geom.cuda import sm90

    return sm90.ln_backward(tbl, codes, dh, mode, eps)


class _PairEmbedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tbl, mode, bb, ca, cb, mask, idx, chains, dlim, alim, ln_eps):
        if mode == 0:
            out, codes = (pair_embed_geom_fwd_train(tbl, bb, ca, cb, mask, idx, chains, dlim, alim) if ln_eps is None
                          else pair_embed_geom_fwd_ln(tbl, bb, ca, cb, mask, idx, chains, dlim, alim, ln_eps))
        else:
            out, codes = (pair_embed_geom_ca_fwd(tbl, bb, mask, idx, dlim) if ln_eps is None
                          else pair_embed_geom_ca_fwd_ln(tbl, bb, mask, idx, dlim, ln_eps))   # `bb` is the CA coordinates [B, n, 3] here
        ctx.save_for_backward(codes, tbl)
        ctx.mode, ctx.ln_eps = mode, ln_eps
        return out

    @staticmethod
    def backward(ctx, g):
        codes, tbl = ctx.saved_tensors
        g = g.to(torch.bfloat16).contiguous()
        if ctx.ln_eps is not None:
            g = pair_embed_geom_ln_bwd(tbl, codes, g, ctx.mode, ctx.ln_eps)
        dtbl = pair_embed_geom_bwd(g, codes, ctx.mode)
        return (dtbl, None, None, None, None, None, None, None, None, None, None)


def pair_embed_geom_train(w, bb, ca_mask, has_cb, mask, residue_idx, chains, *, ln_eps=None) -> torch.Tensor:
    """As :func:`pair_embed_geom`, differentiable in ``w`` ([256, >= 317]): the gradient is that of a bf16-autocast Linear over the binned
    features (the weight is rounded to bf16 by an ordinary autograd op, the table gradient is accumulated in fp32 in a fixed order and
    rounded to bf16 once). Use under ``torch.autocast(bfloat16)``: the output is bf16. ``ln_eps``: return LayerNorm(pair) over the 256 channels
    (no affine, that eps; the pair is never stored and the backward recomputes it) -- what the engine's attention-bias projection reads."""
    dev = bb.device
    tbl = w[:, :N_ROWS].t().to(torch.bfloat16).contiguous()
    dlim = torch.linspace(0.1, 2, 20, device=dev)
    alim = torch.linspace(-torch.pi, torch.pi, 20, device=dev)
    return _PairEmbedFn.apply(tbl, 0, bb.float().contiguous(), ca_mask.to(torch.uint8).contiguous(), has_cb.to(torch.uint8).contiguous(),
                              mask.to(torch.uint8).contiguous(), residue_idx.to(torch.int32).contiguous(), chains.to(torch.int32).contiguous(), dlim, alim, ln_eps)


def refusal_ca(w, ca, mask, residue_idx) -> str | None:
    if not ca.is_cuda:
        return "the input is not on a CUDA device"
    if torch.cuda.get_device_capability(ca.device) != (9, 0):
        return "served on H100 (sm_90a) only"
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
    idx = residue_idx if residue_idx is not None else torch.arange(1, n + 1, device=dev).expand(B, n)
    tbl = w.t().to(torch.bfloat16).contiguous()
    dlim = torch.linspace(0.1, 3.0, 29, device=dev)
    return _PairEmbedFn.apply(tbl, 1, ca.float().contiguous(), None, None, mask.to(torch.uint8).contiguous(), idx.to(torch.int32).contiguous(), None, dlim, None, ln_eps)


__all__ = ["D_PAIR", "refusal", "pair_embed_geom", "pair_embed_geom_train", "pair_embed_ca", "refusal_ca"]
