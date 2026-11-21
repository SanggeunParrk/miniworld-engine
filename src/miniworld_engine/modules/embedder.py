"""Input feature embedder, ESMFold2 style: the atom encoder that turns reference-conformer atom features into one vector per token.

Per atom: the six reference features (position, mask, element, arcsinh(charge)) -> a ``d_atom`` single -> an ESMFold2
``SWAAtomTransformer`` (adaLN-Zero blocks, sliding-window 3D-RoPE self-attention, SwiGLU FFN; the single conditions itself) -> an
affine-free terminal RMSNorm -> ``Linear + ReLU`` to the token width -> the mean over each token's atoms.

This module is a composition: the reference features, a projection, the ``SWAAtomTransformer``, a terminal norm, the token read-out and
one scatter-add for the token mean.

Parameter names match MiniWorld's ``ESMFold2InputAtomAttentionEncoder`` (``to_atom_single_cond``, ``atom_transformer``,
``atom_out_norm``, ``atom_single_rep_to_token_single``), so its checkpoints load unchanged. Unlike that encoder this one never builds the
O(L_atom^2) atom-pair features the ESMFold2 path discards.

``features="esmfold2"`` is ESMFold2's own encoder (biohub/esm ``EsmFold2AtomEncoder``) without the atom-name characters: per atom
position, charge, atom mask and the element one-hot (atomic number, 128 classes) -> ``Linear`` -> an affine ``LayerNorm`` (``atom_norm``)
that both starts the atom stream and conditions every block -> the transformer -> ``Linear + ReLU`` with no terminal norm -> the token
mean. The default ``"legacy"`` layout above is the one MiniWorld trained through v2.3.0: its scalar element and un-normalised
condition let the residual stream grow without bound and collapse onto one direction, after which the bias-free ReLU dies (2026-10-22).

``token_pool="linear"`` replaces the terminal ``Linear + ReLU`` with the LLM / ViT read-out: an affine ``LayerNorm`` (``atom_out_norm``)
-> a bias-free ``Linear`` -> the token mean, no activation. The bias-free ReLU sat on a stream whose atoms share one direction, so
most units went silent for every atom at once and never recovered (ESMFold2 layout, v2.3.1: 30 of 384 alive, 0.2% active).

A :class:`~miniworld_engine.modules.primitives.BackendModule`; ``_forward_pytorch`` is the reference.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal, cast

import torch
import torch.nn.functional as F
from torch import nn

from miniworld_engine.modules.atom_dit import SWAAtomTransformer
from miniworld_engine.modules.functional import build_attention_params
from miniworld_engine.modules.norm import LayerNorm, RMSNorm
from miniworld_engine.modules.primitives import BackendModule, Linear, key_mask, random_input, wide

#: Reference features per atom: position (3), mask, element, arcsinh(charge).
N_ATOM_FEATURES = 6
#: Element one-hot width of the ``"esmfold2"`` layout (atomic numbers 0..127, ESMFold2's ``MAX_ATOMIC_NUMBER``).
MAX_ATOMIC_NUMBER = 128
#: ``"esmfold2"`` features per atom: position (3), charge, atom mask, element one-hot.
N_ATOM_FEATURES_ESMFOLD2 = 3 + 1 + 1 + MAX_ATOMIC_NUMBER

AtomFeatures = Literal["legacy", "esmfold2"]
TokenPool = Literal["relu", "linear"]


def atom_single_features(
    pos: torch.Tensor,
    mask: torch.Tensor,
    element: torch.Tensor,
    charge: torch.Tensor,
) -> torch.Tensor:
    """The six per-atom reference features (AF3 Alg. 5 inputs without the atom-pair part), zero on masked atoms."""
    mask = mask.to(pos.dtype)
    features = torch.cat([pos, mask[..., None], element.to(pos.dtype)[..., None], torch.arcsinh(charge.to(pos.dtype))[..., None]], dim=-1)
    return features * mask[..., None]


def atom_single_features_esmfold2(
    pos: torch.Tensor,
    charge: torch.Tensor,
    atom_mask: torch.Tensor,
    element: torch.Tensor,
) -> torch.Tensor:
    """ESMFold2's per-atom features without the atom-name characters: position, charge, atom mask, element one-hot (atomic number),
    zero on masked atoms (ESMFold2 zeroes the one-hots there; positions and charges of padding are zero already)."""
    mask = atom_mask.to(pos.dtype)
    element = F.one_hot(element.long().clamp(0, MAX_ATOMIC_NUMBER - 1), MAX_ATOMIC_NUMBER).to(pos.dtype)
    features = torch.cat([pos, charge.to(pos.dtype)[..., None], mask[..., None], element], dim=-1)
    return features * mask[..., None]


def _colsum(K, gb: torch.Tensor) -> torch.Tensor:
    """B200 (round 3 part 13): the column sums [R, 128] of the per-row LayerNorm weight / bias gradient terms ``gb`` [R, M, 128] fp32 in
    one cluster launch (embed_glue.cu ``swa_embed_colsum_sm100``, fixed order) -- was ``gb.sum(1)``: a memset + an 11-us reduce."""
    out = torch.empty(gb.shape[0], 128, device=gb.device, dtype=torch.float32)
    K.colsum((gb.shape[0], 8, 1), (1024, 1, 1), gb, out, int(gb.shape[1]))
    return out


class _TokenMeanB200(torch.autograd.Function):
    """B200: the mean over each token's atoms of ``x`` [B, N, T] (bf16 / fp32; the read-out after its mask and activation), as
    ``InputFeatureEmbedder.to_tokens`` states it (fp32 sums, the count of valid atoms clamped at 1) on embed_glue.cu: a zero fill, the
    scatter launch (one add per cell when a token's atoms are contiguous) and the normalising launch; the backward is the gather."""

    @staticmethod
    def forward(ctx, x, atom_mask, idx, n_token, relu=False):
        from miniworld_engine.kernels._device import num_sms
        from miniworld_engine.kernels.swa_dit.cuda import sm100

        b, n, t = x.shape
        dev = x.device
        bf = x.dtype == torch.bfloat16
        with torch.cuda.device(dev), torch.autocast("cuda", enabled=False):
            K = sm100.embed_glue(dev.index if dev.index is not None else torch.cuda.current_device())
            acc = torch.zeros(b, n_token, t + 1, device=dev, dtype=torch.float32)
            out = torch.empty(b, n_token, t, device=dev, dtype=x.dtype)
            cap = 16 * num_sms()
            (K.scatter8 if (_SCATTER8 and n % 8 == 0) else K.scatter)(
                (min(((b * (n // 8) if (_SCATTER8 and n % 8 == 0) else b * n) * (t // 8) + 255) // 256, cap), 1, 1),
                (256, 1, 1),
                x,
                atom_mask,
                idx,
                acc,
                int(b),
                int(n),
                int(n_token),
                int(t),
                int(relu),
                int(bf),
            )
            K.tokmean((min((b * n_token * (t // 8) + 255) // 256, cap), 1, 1), (256, 1, 1), acc, out, int(b * n_token), int(t), int(bf))
        ctx.save_for_backward(idx, acc, x if relu else acc)
        ctx.x_dtype, ctx.relu = x.dtype, relu
        return out

    @staticmethod
    def backward(ctx, g):
        idx, acc, x = ctx.saved_tensors
        t = acc.shape[-1] - 1
        if not _READOUT_B200 or g.dtype != ctx.x_dtype:
            with torch.autocast("cuda", enabled=False):
                gw = g.float() / acc[..., t:].clamp(min=1.0)
                dx = gw.gather(1, idx[..., None].expand(-1, -1, t)).to(ctx.x_dtype)
                if ctx.relu:
                    dx = dx * (x > 0)
            return dx, None, None, None, None
        from miniworld_engine.kernels._device import num_sms
        from miniworld_engine.kernels.swa_dit.cuda import sm100

        b, n = idx.shape
        dev = g.device
        g = g.contiguous()
        dx = torch.empty(b, n, t, device=dev, dtype=ctx.x_dtype)
        with torch.cuda.device(dev):  # round 3 part 11: one launch (gather, / count, the ReLU's mask)
            sm100._ctx_current(dev.index if dev.index is not None else torch.cuda.current_device())  # the autograd worker thread
            K = sm100.embed_glue(dev.index if dev.index is not None else torch.cuda.current_device())
            K.tokgrad((min((b * n * (t // 8) + 255) // 256, 16 * num_sms()), 1, 1), (256, 1, 1), g, acc, idx, x if ctx.relu else g, dx, int(b), int(n),
                      int(acc.shape[1]), int(t), int(ctx.relu), int(g.dtype == torch.bfloat16))
        return dx, None, None, None, None


#: B200 training (round 3 part 11, development switch): the read-out's terminal norm + mask as :class:`_OutNormB200`, the ReLU inside
#: :class:`_TokenMeanB200`, its backward one launch
_READOUT_B200 = True


#: N1 round 4 (development switch): the rope launch's seqused count on 16-byte loads (embed_glue.cu swa_embed_rope2), inference only (the step
#: of a training forward measured +8 us at window 128 L 4096 with it, -4 us with global attention: the old launch stays there)
_ROPE2 = True


class _OutNormB200(torch.autograd.Function):
    """B200 training (round 3 part 11): ``atom_out_norm(x).to(x.dtype) * mask`` in one launch (mode 0 identity, 1 affine LayerNorm, 2
    affine-free RMSNorm; fp32 statistics, the norm rounded to x's dtype before the mask, as the statements); backward one launch (+ one
    column sum for the LayerNorm's weight / bias gradients, fixed order)."""

    @staticmethod
    def forward(ctx, x, w, b, mask, mode, eps):
        from miniworld_engine.kernels._device import num_sms
        from miniworld_engine.kernels.swa_dit.cuda import sm100

        x = x.contiguous()
        m, dev = x.numel() // 128, x.device
        y = torch.empty_like(x)
        grid = (min((m + 7) // 8, 16 * num_sms()), 1, 1)
        with torch.cuda.device(dev), torch.autocast("cuda", enabled=False):
            K = sm100.embed_glue(dev.index if dev.index is not None else torch.cuda.current_device())
            lw = w.detach().float().contiguous() if mode == 1 else x
            lb = b.detach().float().contiguous() if mode == 1 else x
            K.outnorm(grid, (256, 1, 1), x, mask, lw, lb, y, int(m), int(mode), int(x.dtype == torch.bfloat16), float(eps), None, 0)
        ctx.save_for_backward(x, mask, lw)
        ctx.meta = (mode, float(eps), grid)
        return y

    @staticmethod
    def backward(ctx, g):
        from miniworld_engine.kernels.swa_dit.cuda import sm100

        x, mask, lw = ctx.saved_tensors
        mode, eps, grid = ctx.meta
        m, dev = x.numel() // 128, x.device
        g = g.contiguous()
        dx = torch.empty_like(x)
        gb = torch.empty(2, m, 128, device=dev, dtype=torch.float32) if mode == 1 else None
        with torch.cuda.device(dev), torch.autocast("cuda", enabled=False):
            sm100._ctx_current(dev.index if dev.index is not None else torch.cuda.current_device())  # the autograd worker thread
            K = sm100.embed_glue(dev.index if dev.index is not None else torch.cuda.current_device())
            K.outnormbwd(grid, (256, 1, 1), g, x, mask, lw, dx, gb if mode == 1 else dx, int(m), int(mode), int(g.dtype == torch.bfloat16),
                         int(x.dtype == torch.bfloat16), int(dx.dtype == torch.bfloat16), float(eps))
            dw = db = None
            if mode == 1:
                dw, db = _colsum(K, cast(torch.Tensor, gb)).unbind(0)
        return dx, dw, db, None, None, None


class _ReadoutB200(torch.autograd.Function):
    """B200 training (round 3 part 14): :class:`_OutNormB200` and the read-out's bias-free GEMM ``y Wr^T`` in one Function, the GEMM on
    ``wr`` -- Wr in the activation dtype, made ahead on a side stream once per weight version (``_b200_side_forms``; was a cast in front
    of the GEMM and the gradient's cast back to fp32 every step). Backward: dy = g Wr and the norm's backward on the current stream, the
    weight gradient g^T y with fp32 output (no bf16 rounding of it, no cast) on a side stream beside them."""

    @staticmethod
    def forward(ctx, x, w, b, mask, mode, eps, lin_w, wr):
        from miniworld_engine.kernels._device import num_sms
        from miniworld_engine.kernels.swa_dit.cuda import sm100

        x = x.contiguous()
        m, dev = x.numel() // 128, x.device
        y = torch.empty_like(x)
        grid = (min((m + 7) // 8, 16 * num_sms()), 1, 1)
        t = wr.shape[0]
        with torch.cuda.device(dev), torch.autocast("cuda", enabled=False):
            K = sm100.embed_glue(dev.index if dev.index is not None else torch.cuda.current_device())
            lw = w.detach().float().contiguous() if mode == 1 else x
            lb = b.detach().float().contiguous() if mode == 1 else x
            K.outnorm(grid, (256, 1, 1), x, mask, lw, lb, y, int(m), int(mode), int(x.dtype == torch.bfloat16), float(eps), None, 0)
            out = torch.mm(y.view(m, 128), wr.t()).view(*y.shape[:-1], t)
        ctx.save_for_backward(x, mask, lw, y, wr)
        ctx.meta = (mode, float(eps), grid)
        ctx.lin_dtype = lin_w.dtype
        return out

    @staticmethod
    def backward(ctx, g):
        from miniworld_engine.kernels.swa_dit.cuda import sm100

        x, mask, lw, y, wr = ctx.saved_tensors
        mode, eps, grid = ctx.meta
        m, dev = x.numel() // 128, x.device
        t = wr.shape[0]
        g2 = g.reshape(m, t).to(y.dtype).contiguous()
        y2 = y.view(m, 128)
        dwl = torch.empty(t, 128, device=dev, dtype=torch.float32)  # (allocated on the current stream)
        res = {}

        def wgrad():
            with torch.autocast("cuda", enabled=False):
                if y2.dtype == torch.float32:
                    torch.mm(g2.t(), y2, out=dwl)
                else:
                    torch.mm(g2.t(), y2, out_dtype=torch.float32, out=dwl)

        def chain():
            with torch.cuda.device(dev), torch.autocast("cuda", enabled=False):
                sm100._ctx_current(dev.index if dev.index is not None else torch.cuda.current_device())  # the autograd worker thread
                K = sm100.embed_glue(dev.index if dev.index is not None else torch.cuda.current_device())
                dyn = torch.mm(g2, wr).view(x.shape)
                dx = torch.empty_like(x)
                gb = torch.empty(2, m, 128, device=dev, dtype=torch.float32) if mode == 1 else None
                K.outnormbwd(grid, (256, 1, 1), dyn, x, mask, lw, dx, gb if mode == 1 else dx, int(m), int(mode), int(dyn.dtype == torch.bfloat16),
                             int(x.dtype == torch.bfloat16), int(dx.dtype == torch.bfloat16), float(eps))
                res["dx"] = dx
                res["dl"] = _colsum(K, cast(torch.Tensor, gb)).unbind(0) if mode == 1 else (None, None)

        sm100._fork_join([wgrad], chain)
        dlw, dlb = res["dl"]
        return res["dx"], dlw, dlb, None, None, None, dwl.to(ctx.lin_dtype), None


def _b200_side_forms(stream, forms):
    """B200 round 3 part 14: weight forms made on ``stream`` (forked by the caller), cached per weight version
    (``kernels._capture.param_forms``: eager once per optimizer update, a capture once per capture). ``forms``: (tag, w, dtype, transpose);
    returns the forms (allocated on the current stream)."""
    from miniworld_engine.kernels import _capture

    out = []
    for tag, w, dtype, tr in forms:

        def build(w=w, dtype=dtype, tr=tr):
            src = w.detach().t() if tr else w.detach()
            o = torch.empty(src.shape, device=w.device, dtype=dtype)
            with torch.cuda.stream(stream):
                o.copy_(src)
            return o

        out.append(_capture.param_forms(("ife_b200_side", tag, dtype, tr), (w,), build))
    return out


#: B200 training (round 3 part 14, development switch): the step's weight forms on a side stream ahead (_b200_side_forms, the blocks'
#: packs), the positions cast once, the read-out norm + GEMM as one Function (_ReadoutB200)
_GLUE_B200 = True


#: B200 training (round 3 part 11, development switch): the features, projection and LayerNorm as :class:`_FeatB200`
_FEAT_B200 = True


_FEAT_DWK = __import__("os").environ.get("SWA_FEATDWK", "1") == "1"  # N2: swa_embed_dwf_sm100 for the features' weight gradient
#: N2 (development switches, 2026-11-18; all on): the inference read-out's terminal norm on swa_embed_outnorm_sm100 (it also zeroes the scatter's
#: accumulator), the master read-out weight's bf16 form cached, the read-out scatter on one thread a group of 8 atoms (swa_embed_scatter8_sm100), the RoPE
#: launch beside the features on a side stream (training)
_EVAL_OUTNORM = __import__("os").environ.get("SWA_EVAL_OUTNORM", "1") == "1"
_EVAL_RW = __import__("os").environ.get("SWA_EVAL_RW", "1") == "1"
_SCATTER8 = __import__("os").environ.get("SWA_SCATTER8", "1") == "1"
_ROPE_SIDE = __import__("os").environ.get("SWA_ROPE_SIDE", "1") == "1"


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class _FeatB200(torch.autograd.Function):
    """B200 training (round 3 part 11): the reference features, their bias-free projection and (``"esmfold2"``) the affine LayerNorm as
    one launch (embed_glue.cu ``swa_embed_feat_sm100``, fp32 arithmetic on the fp32 weights, as inference; it also keeps the masked
    projection for the backward), the output in ``act``. Backward: one launch (``swa_embed_featbwd_sm100``: the LayerNorm backward, the
    positions' gradient, the features), the weight gradient as one GEMM DP^T F and the LayerNorm's weight / bias gradients as one column
    sum (fixed order: run-to-run the same bits). Was ~10 torch launches forward and ~10 backward."""

    @staticmethod
    def forward(ctx, pos, w, lnw, lnb, charge, element, fmask, esm, eps, act):
        from miniworld_engine.kernels._device import num_sms
        from miniworld_engine.kernels.swa_dit.cuda import sm100

        b, n = pos.shape[:2]
        m, dev = b * n, pos.device
        with torch.cuda.device(dev), torch.autocast("cuda", enabled=False):
            K = sm100.embed_glue(dev.index if dev.index is not None else torch.cuda.current_device())
            # round 3 part 14: the positions read in place (bf16 widened in the kernel: no cast launch) and W [128, nf] read as it is
            # (no transposed copy per step): flags bit 0 / bit 1 (embed_glue.cu ldpos / ldw)
            posk = pos.detach().contiguous() if pos.dtype in (torch.float32, torch.bfloat16) else pos.detach().float().contiguous()
            wk = w.detach().contiguous()
            fl = int(posk.dtype == torch.bfloat16) | 2
            ln = lnw is not None
            lw = lnw.detach().float().contiguous() if ln else wk
            lb = lnb.detach().float().contiguous() if ln else wk
            single = torch.empty(b, n, 128, device=dev, dtype=act)
            proj = torch.empty(b, n, 128, device=dev, dtype=torch.float32) if ln else None
            grid = (min((m + 7) // 8, 16 * num_sms()), 1, 1)
            K.feat(grid, (256, 1, 1), posk, charge, element, fmask, wk, lw, lb, single, int(m), int(esm), int(act == torch.bfloat16), int(ln),
                   float(eps), proj, fl)
        ctx.save_for_backward(posk, charge, element, fmask, wk, lw, proj)
        ctx.meta = (esm, float(eps), ln, pos.dtype, w.shape[1], grid, fl)
        return single

    @staticmethod
    def backward(ctx, g):
        from miniworld_engine.kernels.swa_dit.cuda import sm100

        posk, charge, element, fmask, wk, lw, proj = ctx.saved_tensors
        esm, eps, ln, pdt, nf, grid, fl = ctx.meta
        b, n = posk.shape[:2]
        m, dev = b * n, posk.device
        # DPOS written in the positions' dtype when that is bf16 (flag bit 2: no cast launch), else fp32 (cast to pdt below if needed)
        dfl = fl | (4 if pdt == torch.bfloat16 else 0)
        g = g.contiguous()
        with torch.cuda.device(dev), torch.autocast("cuda", enabled=False):
            sm100._ctx_current(dev.index if dev.index is not None else torch.cuda.current_device())  # the autograd worker thread
            K = sm100.embed_glue(dev.index if dev.index is not None else torch.cuda.current_device())
            dp = torch.empty(m, 128, device=dev, dtype=torch.float32)
            gb = torch.empty(2, m, 128, device=dev, dtype=torch.float32) if ln else None
            nfs = (nf + 7) // 8 * 8  # N2: F's rows padded to 16-B multiples (featbwd writes them so; the GEMM fallback reads f[:, :nf])
            dwk = _FEAT_DWK and m <= 8192 and nf in (6, 133) and nf == (133 if esm else 6)  # N2: the weight gradient by swa_embed_dwf_sm100 (no F)
            f = None if dwk else torch.empty(m, nfs, device=dev, dtype=torch.float32)
            dpos = torch.empty(b, n, 3, device=dev, dtype=torch.bfloat16 if dfl & 4 else torch.float32)
            K.featbwd(grid, (256, 1, 1), g, proj if ln else dp, posk, charge, element, fmask, wk, lw, dp, gb if ln else dp, f, dpos, int(m),
                      int(esm), int(g.dtype == torch.bfloat16), int(ln), float(eps), dfl)
            res: dict[str, Any] = {"dw": torch.empty(128, nf, device=dev, dtype=torch.float32)}  # (allocated on the current stream)
            def gemm():  # round 3 part 14: the weight gradient's GEMM on a side stream beside the column sum and the cast
                if dwk:
                    K.dwf(((6 if not esm else 5 + 16) * 8, 1, 1), (1024, 1, 1), dp, posk, charge, element, fmask, res["dw"], int(m), int(esm), int(fl))
                else:
                    torch.mm(dp.t(), f[:, :nf], out=res["dw"])

            def rest():
                res["dl"] = _colsum(K, cast(torch.Tensor, gb)).unbind(0) if ln else (None, None)
                res["dpos"] = dpos.to(pdt)

            sm100._fork_join([gemm], rest)
            dlw, dlb = res["dl"]
        return res["dpos"], res["dw"], dlw, dlb, None, None, None, None, None, None


class InputFeatureEmbedder(BackendModule):
    """Atom features -> per-token single (ESMFold2-style input atom encoder).

    ``forward(pos, ref_mask, element, charge, space_uid, atom_mask, atom_to_token, n_token)`` with ``pos`` [B, N, 3], the other atom
    tensors [B, N] and ``atom_to_token`` [B, N] the token index of each atom; returns [B, n_token, d_token].

    ``backend="anthropic"`` sets its ``atom_transformer``'s (the SWA attention on ESMFold2's release band core); the rest stays
    ``"pytorch"``'s (no release op has the features, the read-out or the token mean).

    ``backend="cuda"`` / ``"triton"`` (the engine's kernels) sets the same backend on its ``atom_transformer`` and on its norms
    (``atom_norm``, ``atom_out_norm``: the LayerNorm / RMSNorm kernels); the reference features, the two bias-free projections, the ReLU
    and the token mean (one scatter-add) are the composition's own statements, as in ``"pytorch"``. Each child states its own range and
    raises :class:`~miniworld_engine.modules.primitives.BackendUnsupported` outside it; this class refuses a backend its atom transformer
    does not have.
    """

    version = 1  # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "anthropic", "cuda", "triton")
    # 2026-11-10: kits; sm_80 paths; H100 inference (cached weight forms, slim modulation, fp32 global on the fp32 stages, PDL)
    # cuda:a100_sm80 6 (2026-11-12): bf16 training: the weight forms in one launch (train_forms), row-count launch shapes and K splits, the
    # few-row modulation backward on cuBLAS (sm80.train_cfg)
    # cuda:h100_sm90 3 (2026-11-12): bf16 / master global attention on the composed bf16 stages around FlashAttention-4 (was the native stages)
    # triton 2 (2026-11-12): served (its SWAAtomTransformer gained "triton", window and global)
    # cuda:5090_sm120 1 (2026-11-12): routed on the RTX 5090 (its children's sm_120a builds)
    # cuda:b200_sm100 21 (2026-11-18, round 3 part 15): its stack as SWAAtomTransformer 20 (cluster dW sum, TF32-route prefetch)
    # cuda:b200_sm100 20 (2026-11-18, round 3 part 14): the step's glue -- positions read in place (bf16 widened in the RoPE / feature
    # kernels, dpos written bf16), W read in place by the feature kernels, the read-out weight form and the blocks' packs on a side stream,
    # the terminal norm + read-out GEMM as one Function (weight gradient fp32 out on a side stream), the features' weight GEMM beside the
    # column sum; the stack's x = c gradient summed inside the first block's dsilu launch (SWAAtomTransformer 19)
    # cuda:b200_sm100 19 (2026-11-18, round 3 part 13): the LayerNorm weight / bias gradients of the features and the terminal norm as one
    # cluster column-sum launch (embed_glue.cu swa_embed_colsum_sm100, fixed order) in place of gb.sum(1) (memset + 11-us reduce, twice a step)
    backend_versions: ClassVar[dict[str, int]] = {
        "anthropic": 2,
        "cuda:a100_sm80": 8,
        "cuda:h100_sm90": 4,
        "triton": 2,
        "cuda:5090_sm120": 5,
        "cuda:b200_sm100": 34,
    }  # cuda:a100_sm80 8 / cuda:5090_sm120 2 (2026-11-15): its atom_norm LayerNorm runs the changed forward row kernels (sm_80 / sm_120a; LayerNorm cuda 3 on both cards); no kernel row bump: one would also owe every triton case that lists it
    # cuda:b200_sm100 34 (2026-11-19, N7, inference) / 33 (training): the global attention forward's split build (gattn_fwd.cu CMBS=2) combines the pieces' partials through shared memory, CTA order kept (same bits; N 1 L 5120 - 7168 -4 .. -7 % of the kernel)
    # cuda:b200_sm100 32 (2026-11-18, N2 round, inference; inference versions are even): the terminal norm of the read-out on the one-launch norm of the training
    # read-out (embed_glue.cu swa_embed_outnorm_sm100: fp32 statistics, rounded to the activation dtype, times the mask; it also zeroes the scatter's accumulator),
    # the scatter on one thread a group of 8 atoms (swa_embed_scatter8_sm100: the same bits, 11 -> 8 us at 6144 atoms), the master read-out weight's bf16 form
    # cached (no cast launch); L 6144 eval master legacy 305 -> 293 us, float32 360 -> 344 us
    # cuda:b200_sm100 31 (2026-11-18, N2 round, training; training versions are odd): the features' weight gradient on one structured kernel
    # (embed_glue.cu swa_embed_dwf_sm100: dense columns by row chunks over a cluster, the element one-hot columns by a block-ordered match list; fixed order;
    # featbwd writes no F), the RoPE launch on a side stream beside the features, the TF32 route's weight forms made at the start of the step, the scatter8 launch
    # in the training read-out, dWmod on the cluster dW kernel (its stack as SWAAtomTransformer 29); L 4096 master global training 669.9 -> 647 us, L 8192 1418 -> 1386 us
    # cuda:b200_sm100 30 (2026-11-18, N1 round 4, inference; inference versions are even): the rope launch's seqused count (embed_glue.cu swa_embed_rope2: 16-byte loads in the one-block-a-row count that was the launch's tail, 7.4 - 10.8 -> 3.2 - 3.4 us) in inference (same integers, same bits); eval step -2 - -12 us (L 3072 - 8192)
    # cuda:b200_sm100 29 (2026-11-18, N1 round 4, training; training versions are odd): its atom transformer's global blocks as SWAAtomTransformer 27 (fused cast + delta launch, early attention trigger; the same bits)
    # cuda:b200_sm100 27 (2026-11-18, r3-swa part 21): its atom transformer's training as SWAAtomTransformer 26
    # cuda:b200_sm100 26 (2026-11-18, r3-swa part 20): its atom transformer's training as SWAAtomTransformer 25 (the same bits)
    # cuda:b200_sm100 25 (2026-11-18, r3-swa part 19): its atom transformer's master training as SWAAtomTransformer 24
    # cuda:b200_sm100 24 (2026-11-18, r3-swa part 18): the global blocks' attention backward of one sample on the cluster split (SWAAtomTransformer 23)
    # cuda:b200_sm100 23 (2026-11-18, r3-swa part 17): the global blocks' attention forward of one sample on the cluster split (SWAAtomTransformer 22)
    # cuda:b200_sm100 22 (2026-11-18, round 3 part 16): its global blocks as SWAAtomTransformer 21 (F32IO training forward, eight-thread glue)
    # cuda:b200_sm100 18 (2026-11-18, round 3 part 12): its SWAAtomTransformer child (cuda:b200_sm100 18)
    # cuda:b200_sm100 17 (2026-11-17, round 3 part 11): its SWAAtomTransformer child (cuda:b200_sm100 17: the few-tile fp32 FFN stage on CTA pairs);
    # the RoPE tables in one launch also under a gradient into the positions (no B200 block Function differentiates cos / sin: the same
    # bits; master training L 1024 window 436 -> 393 us, float32 training L 2048 global 725 -> 684)
    # and its training statements as Functions on embed_glue.cu: features + projection + LayerNorm (_FeatB200: 1 launch forward, 1 + a GEMM +
    # a column sum backward), the terminal norm + mask (_OutNormB200), the ReLU inside the token mean with a one-launch backward: master training
    # L 1024 window 393 -> 358 us, legacy L 2048 window 479 -> 389 (lights within YELLOW, none above the B200 tokens)
    # cuda:b200_sm100 16 (2026-11-17, round 3 part 10): its SWAAtomTransformer child (cuda:b200_sm100 16: multi-lane TMA issue in the fp32 forward stages)
    # cuda:b200_sm100 15 (2026-11-17, round 3 part 4, swa-global): its SWAAtomTransformer child (cuda:b200_sm100 15: fp32 global inference on gattn_fwd32)
    # cuda:b200_sm100 14 (2026-11-17, round 3 part 8): its SWAAtomTransformer child (cuda:b200_sm100 14: master training on the bf16 forward + TF32 backward)
    # cuda:b200_sm100 13 (2026-11-17, round 3 part 3, swa-global): its SWAAtomTransformer child (cuda:b200_sm100 13)
    # cuda:b200_sm100 12 (2026-11-17, round 3 part 7): its SWAAtomTransformer child (cuda:b200_sm100 12)
    # cuda:b200_sm100 1 (2026-11-16): routed on the B200 (its atom transformer and norms run their sm_100a builds)
    # cuda:b200_sm100 11 (2026-11-17, merge of round 3 part 6 + swa-global part 2): its SWAAtomTransformer child (cuda:b200_sm100 11)
    # cuda:b200_sm100 10 (2026-11-17, round 3 part 2, swa-global): its SWAAtomTransformer child (cuda:b200_sm100 10: global attention training backward on gattn_bwd.cu)
    # cuda:b200_sm100 9 (2026-11-17, round 3 part 6): its SWAAtomTransformer child (cuda:b200_sm100 9: master training without fp32 copies,
    # dy inside the FFN gate kernel)
    # cuda:b200_sm100 7 (2026-11-17, round 3 part 4): its SWAAtomTransformer child (cuda:b200_sm100 7: float32 training of few tiles)
    # cuda:b200_sm100 8 (2026-11-17, round 3 part 5): its SWAAtomTransformer child (cuda:b200_sm100 8: master training of one sample on the TF32 kernels)
    # cuda:b200_sm100 6 (2026-11-17, round 3, swa-global): its SWAAtomTransformer child (cuda:b200_sm100 6: global attention forward on gattn_fwd.cu)
    # cuda:b200_sm100 2 (2026-11-17): its SWA atom transformer (SWAAtomTransformer cuda:b200_sm100 2: PDL, streams, FA4 global) and LayerNorm (cuda:b200_sm100 2: sub-warp row kernels) children changed
    # cuda:b200_sm100 5 (2026-11-17, round 3): its SWAAtomTransformer child (cuda:b200_sm100 5: many-tile inference blocks, modulation inside)
    # cuda:b200_sm100 4 (2026-11-17, round 3): its SWAAtomTransformer child (cuda:b200_sm100 4: few-tile inference blocks)
    # cuda:b200_sm100 3 (2026-11-17, round 3): its own statements on embed_glue.cu (RoPE tables + seqused, inference features + projection + LayerNorm,
    # the token mean; training: the token mean as one Function) and its SWAAtomTransformer child (cuda:b200_sm100 3)
    # cuda:a100_sm80 7 (2026-11-13): its SWAAtomTransformer's
    #: The engine backends this composition hands to its children.
    _ENGINE_BACKENDS = ("cuda", "triton")

    def __init__(
        self,
        d_atom: int,
        d_token: int,
        n_head: int,
        n_block: int,
        *,
        swa_window_size: int = 128,
        expansion_ratio: int = 2,
        n_spatial_rope_pairs_per_axis: int = 2,
        n_uid_rope_pairs: int = 10,
        spatial_rope_base_frequency: float = 20.0,
        uid_rope_base_frequency: float = 10000.0,
        features: AtomFeatures = "legacy",
        token_pool: TokenPool = "relu",
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        if features not in ("legacy", "esmfold2"):
            message = f"features must be 'legacy' or 'esmfold2', got {features!r}"
            raise ValueError(message)
        if token_pool not in ("relu", "linear"):
            message = f"token_pool must be 'relu' or 'linear', got {token_pool!r}"
            raise ValueError(message)
        self.features = features
        self.token_pool = token_pool
        self.d_token = d_token
        n_in = N_ATOM_FEATURES_ESMFOLD2 if features == "esmfold2" else N_ATOM_FEATURES
        self.to_atom_single_cond = Linear(n_in, d_atom, init="default", bias=False)
        if features == "esmfold2":
            # ESMFold2's ``atom_norm``: the projected features are layer-normed once; the result starts the atom stream and is the
            # condition of every block, so the adaLN modulation sees a bounded input
            self.atom_norm = LayerNorm(d_atom)
        self.atom_transformer = SWAAtomTransformer(
            SWAAtomTransformer.Config(
                d_atom=d_atom,
                d_cond=d_atom,
                n_block=n_block,
                n_head=n_head,
                swa_window_size=swa_window_size,
                expansion_ratio=expansion_ratio,
                n_spatial_rope_pairs_per_axis=n_spatial_rope_pairs_per_axis,
                n_uid_rope_pairs=n_uid_rope_pairs,
                spatial_rope_base_frequency=spatial_rope_base_frequency,
                uid_rope_base_frequency=uid_rope_base_frequency,
                block_style="esmfold2",
            )
        )
        # Terminal norm of the pre-norm stack (GPT-2 / LLaMA ``ln_f``): affine-free, no parameters. ESMFold2 has none; the linear
        # read-out takes an affine LayerNorm, as an LLM's final norm before its head.
        if token_pool == "linear":
            self.atom_out_norm = LayerNorm(d_atom)
        elif features == "esmfold2":
            self.atom_out_norm = nn.Identity()
        else:
            self.atom_out_norm = RMSNorm(d_atom, elementwise_affine=False)
        self.atom_single_rep_to_token_single = nn.Sequential(
            Linear(d_atom, d_token, init="default", bias=False),
            *([nn.ReLU()] if token_pool == "relu" else []),
        )
        self.set_backend(backend)

    def _engine_children(self) -> list[BackendModule]:
        return [m for m in (getattr(self, "atom_norm", None), self.atom_out_norm) if isinstance(m, BackendModule)]

    def set_backend(self, backend: str = "pytorch"):
        super().set_backend(backend)
        if self.backend in self._ENGINE_BACKENDS:  # engine-fp8 resolved to cuda by super().set_backend
            # a child without this backend stays as it is; _supports_<backend> refuses the call when the atom transformer is one of them
            if self.atom_transformer.has_backend(backend):
                self.atom_transformer.set_backend(backend)
            else:
                self.atom_transformer.set_backend("pytorch")
            for child in self._engine_children():
                child.set_backend(backend if child.has_backend(backend) else "pytorch")
        else:
            self.atom_transformer.set_backend(backend)
            for child in self._engine_children():
                child.set_backend("pytorch")
        return self

    def _forward_anthropic(self, *args, **kwargs) -> torch.Tensor:
        return self._forward_pytorch(*args, **kwargs)  # the atom transformer runs its own backend

    def _supports_engine(self, backend: str, pos: torch.Tensor) -> str | None:
        from miniworld_engine.modules.norm import a100, b200, h100_unsupported, sm120, triton_unsupported

        if backend not in self.atom_transformer.backends:
            return f"its atom transformer ({type(self.atom_transformer).__name__}) has no {backend!r} backend"
        missing = [type(c).__name__ for c in self._engine_children() if backend not in c.backends]
        if missing:
            return f"its {', '.join(missing)} has no {backend!r} backend"
        # cuda: the H100, the A100 and the RTX 5090 (the children's sm_80 builds), or the B200; each child states its own range when it runs
        if backend == "cuda":
            return None if a100(pos) or sm120(pos) or b200(pos) else h100_unsupported(pos)
        return triton_unsupported(pos)

    def _supports_cuda(self, pos: torch.Tensor, *args, **kwargs) -> str | None:
        return self._supports_engine("cuda", pos)

    def _supports_triton(self, pos: torch.Tensor, *args, **kwargs) -> str | None:
        return self._supports_engine("triton", pos)

    def _forward_cuda(self, pos, ref_mask, element, charge, space_uid, atom_mask, atom_to_token, n_token) -> torch.Tensor:
        from miniworld_engine.modules.norm import b200

        args = (pos, ref_mask, element, charge, space_uid, atom_mask, atom_to_token)
        if b200(pos) and self._b200_glue(*args):
            return self._forward_b200(*args, n_token)
        return self._forward_pytorch(*args, n_token)  # the composition; the transformer and the norms run their "cuda" backends

    # ------------------------------------------------------------------------------------------------ B200 (sm_100a) glue, round 3
    # One system of a few thousand atoms is launch-bound: the composition's own statements were ~50 small torch launches around the
    # transformer's ~12. On a B200 the "cuda" forward runs them on kernels/swa_dit/cuda/sm100/embed_glue.cu:
    #   every call      the 3D RoPE tables and seqused in one launch (no gradient flows through them: pos / uid / mask only)
    #   inference       the reference features + projection (+ the "esmfold2" LayerNorm) in one launch (fp32 arithmetic on the fp32
    #                   weights, the result in the activation dtype), and the token mean as a zero fill, one scatter launch and one
    #                   normalising launch around the read-out's GEMM (its mask folded into the scatter; fp32 sums, as the statements)
    # A gradient keeps the features, the projection and the read-out GEMM on their statements (autograd) and takes the token mean as one
    # Function (_TokenMeanB200: fixed-order sums); a gradient into the positions also keeps the RoPE tables on theirs. The transformer and the norms run their
    # own "cuda" backends as before. Range: bool masks, int64 uid / element / charge / token index, fp32 or bf16 positions,
    # d_atom 128, d_token a multiple of 8; anything else runs the statements.
    _B200_CONST: ClassVar[dict] = {}

    def _b200_glue(self, pos, ref_mask, element, charge, space_uid, atom_mask, atom_to_token) -> bool:
        i64 = torch.int64
        return (
            pos.ndim == 3
            and pos.shape[-1] == 3
            and pos.dtype in (torch.float32, torch.bfloat16)
            and ref_mask.dtype == torch.bool
            and atom_mask.dtype == torch.bool
            and all(t.dtype == i64 and tuple(t.shape) == tuple(pos.shape[:2]) for t in (element, charge, space_uid, atom_to_token))
            and tuple(ref_mask.shape) == tuple(pos.shape[:2]) == tuple(atom_mask.shape)
            and self.to_atom_single_cond.weight.shape[0] == 128
            and self.to_atom_single_cond.weight.dtype in (torch.float32, torch.bfloat16)
            and self.d_token % 8 == 0
            and (not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") == torch.bfloat16)
        )

    def _b200_const(self, key: tuple, build):
        """A constant device table (the RoPE frequencies, cu_seqlens), made once per ``key`` -- inside a CUDA-graph capture made each time
        (a tensor first filled by a captured launch holds nothing until a replay)."""
        hit = self._B200_CONST.get(key)
        if hit is None:
            hit = build()
            if not torch.cuda.is_current_stream_capturing():
                self._B200_CONST[key] = hit
        return hit

    def _b200_params(self, pos32: torch.Tensor, space_uid: torch.Tensor, atom_mask: torch.Tensor) -> tuple:
        """``build_attention_params(*build_rope(pos, uid), atom_mask, 1)`` in one launch (cu_seqlens: a cached constant)."""
        from miniworld_engine.kernels.swa_dit.cuda import sm100

        cfg, dev = self.atom_transformer.config, pos32.device
        b, n = pos32.shape[:2]
        half = self.atom_transformer.head_dim // 2
        nsp, nuid = cfg.n_spatial_rope_pairs_per_axis, cfg.n_uid_rope_pairs
        if 3 * nsp + nuid > half:
            return build_attention_params(*self.atom_transformer.build_rope(pos32.float(), space_uid), atom_mask, num_aug=1)  # raises there

        def tables():  # functional.build_3d_rope's frequencies, with the same torch statements (the same fp32 values)
            f32 = torch.float32
            sp = 1.0 / (cfg.spatial_rope_base_frequency ** (torch.arange(0, nsp, dtype=f32, device=dev) / nsp))
            ui = 1.0 / (cfg.uid_rope_base_frequency ** (torch.arange(0, nuid, dtype=f32, device=dev) / nuid))
            inv = torch.cat([sp.repeat(3), ui, torch.zeros(half - 3 * nsp - nuid, dtype=f32, device=dev)])
            src = [a for a in range(3) for _ in range(nsp)] + [3] * nuid + [4] * (half - 3 * nsp - nuid)
            return inv.contiguous(), torch.tensor(src, dtype=torch.int32, device=dev)

        inv, src = self._b200_const(
            ("rope", dev, half, nsp, nuid, float(cfg.spatial_rope_base_frequency), float(cfg.uid_rope_base_frequency)), tables
        )
        cu = self._b200_const(("cu", dev, b, n), lambda: torch.arange(0, (b + 1) * n, n, dtype=torch.int32, device=dev))
        cos, sin = torch.empty(b, n, half, device=dev), torch.empty(b, n, half, device=dev)
        seq = torch.empty(b, dtype=torch.int32, device=dev)
        mask = atom_mask.contiguous()
        nrope = (b * n * half + 255) // 256
        with torch.cuda.device(dev):
            K = sm100.embed_glue(dev.index if dev.index is not None else torch.cuda.current_device())
            (K.rope2 if _ROPE2 and not torch.is_grad_enabled() else K.rope)(
                (nrope + b, 1, 1),
                (256, 1, 1),
                pos32,
                space_uid.contiguous(),
                mask,
                inv,
                src,
                cos,
                sin,
                seq,
                int(b),
                int(n),
                int(half),
                int(nrope),
                int(pos32.dtype == torch.bfloat16),  # round 3 part 14: bf16 positions read in place (flag bit 0)
            )
        return cos, sin, seq, cu, n, atom_mask

    def _forward_b200(self, pos, ref_mask, element, charge, space_uid, atom_mask, atom_to_token, n_token) -> torch.Tensor:
        from miniworld_engine.kernels._device import num_sms
        from miniworld_engine.kernels.swa_dit.cuda import sm100
        from miniworld_engine.kernels.swa_dit.dispatch import eval_form

        dev = pos.device
        b, n = pos.shape[:2]
        pos_grad = pos.requires_grad and torch.is_grad_enabled()
        # the RoPE tables in one launch also under a gradient into the positions (round 3 part 11): no B200 block Function differentiates
        # cos / sin (_SWATrainB200, SWADiTBlockFunction return None for them), so the statements' ~25 launches (~65 us of a one-sample
        # step) built tables whose gradient was dropped -- the positions' gradient is the features' alone either way
        train = pos_grad or (torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters()))
        esm = self.features == "esmfold2"
        ln = self.atom_norm if esm else None
        fused = not train and (ln is None or (ln.weight is not None and ln.bias is not None and ln.weight.dtype == torch.float32))
        w = self.to_atom_single_cond.weight
        lnok = ln is None or (ln.weight is not None and ln.bias is not None and ln.weight.dtype == torch.float32)
        act = torch.bfloat16 if pos.dtype == torch.bfloat16 or torch.is_autocast_enabled("cuda") else torch.float32
        glue = None
        if _GLUE_B200 and not fused and _FEAT_B200 and w.dtype == torch.float32 and lnok:
            # round 3 part 14: the weight forms this step needs, made ahead on a side stream while the RoPE launch and the features run (per
            # weight version: an eager step after the first makes none): the read-out weight in the activation dtype, the atom transformer's
            # block packs (the features read the positions and W in place: no forms)
            from miniworld_engine.modules.atom_dit import _b200_prefetch_hybrid

            cur = torch.cuda.current_stream()
            side = sm100._side_streams(cur.device.index if cur.device.index is not None else torch.cuda.current_device())[0]
            side.wait_stream(cur)
            lw_ = self.atom_single_rep_to_token_single[0].weight
            wr = lw_
            if act == torch.bfloat16 and lw_.dtype == torch.float32:
                wr = _b200_side_forms(side, [("wr", lw_, torch.bfloat16, False)])[0]
            xs = torch.empty((1, 1, 1), device=dev, dtype=act).expand(b, n, 128)
            evs = _b200_prefetch_hybrid(list(self.atom_transformer.blocks), xs)  # (on the same side stream)
            ev_all = torch.cuda.Event()
            ev_all.record(side)
            if evs is None:
                from miniworld_engine.modules.atom_dit import _b200_prefetch_tf32_forms

                _b200_prefetch_tf32_forms(list(self.atom_transformer.blocks), xs)
            glue = (None, wr, None, ev_all, evs)
        # N2: the RoPE launch on a side stream beside the features (it needs nothing of them; both ran back to back on the chain: 7 + 7 us),
        # joined before the transformer
        rope_side = _ROPE_SIDE and train  # (inference: the RoPE launch is 3 us and measured level)
        cur0 = torch.cuda.current_stream()
        s2r = None
        if rope_side:
            s2r = sm100._side_streams(dev.index if dev.index is not None else torch.cuda.current_device())[1]
            s2r.wait_stream(cur0)
        with torch.autocast("cuda", enabled=False), (torch.cuda.stream(s2r) if rope_side else _nullctx()):
            if pos.dtype in (torch.float32, torch.bfloat16):
                params = self._b200_params(pos.detach().contiguous(), space_uid, atom_mask)  # (round 3 part 14: read in place)
            else:
                params = self._b200_params(pos.detach().float().contiguous(), space_uid, atom_mask)
        if rope_side:
            for t_ in params:
                if isinstance(t_, torch.Tensor):
                    t_.record_stream(cur0)
        if not fused:  # a gradient: the features + projection (+ LayerNorm) as one Function (round 3 part 11), the read-out on its statements
            if _FEAT_B200 and w.dtype == torch.float32 and lnok:
                single = _FeatB200.apply(
                    pos, w, cast(Any, ln).weight if esm else None, cast(Any, ln).bias if esm else None, charge.contiguous(), element.contiguous(),
                    (atom_mask if esm else ref_mask).contiguous(), esm, float(cast(Any, ln).eps) if esm else 0.0, act,
                )
            elif esm:
                projected = self.to_atom_single_cond(atom_single_features_esmfold2(pos, charge, atom_mask, element))
                single = self.atom_norm(projected).to(projected.dtype)
            else:
                single = self.to_atom_single_cond(atom_single_features(pos, ref_mask, element, charge))
            if glue is not None:  # (the side stream joined: every form made above is ready)
                cur = torch.cuda.current_stream()
                if glue[3] is not None:
                    cur.wait_event(glue[3])
                if glue[4] is not None:
                    cur.wait_event(glue[4][-1])
            if rope_side:
                cur0.wait_stream(s2r)
            single = self.atom_transformer(single, single, params)
            on = self.atom_out_norm
            mode = 1 if isinstance(on, LayerNorm) else 2 if isinstance(on, RMSNorm) and on.weight is None else 0 if isinstance(on, nn.Identity) else -1
            lnok = mode != 1 or (isinstance(on, LayerNorm) and on.weight is not None and on.bias is not None and on.weight.dtype == torch.float32)
            if _READOUT_B200 and mode >= 0 and lnok and single.dtype in (torch.bfloat16, torch.float32) and single.shape[-1] == 128:
                # round 3 part 11: the terminal norm + mask in one launch, the ReLU inside the token mean (its backward one launch)
                eps = float(on.eps) if isinstance(on, LayerNorm) else float(on.effective_eps(single.dtype)) if isinstance(on, RMSNorm) and mode == 2 else 0.0
                lin = self.atom_single_rep_to_token_single[0]
                if glue is not None and lin.bias is None and glue[1].dtype == single.dtype and lin.weight.dtype == torch.float32:
                    # round 3 part 14: the norm and the read-out GEMM as one Function on the read-out weight's form made ahead
                    per_atom = _ReadoutB200.apply(
                        single, on.weight if mode == 1 else None, on.bias if mode == 1 else None, atom_mask.contiguous(), mode, eps, lin.weight,
                        glue[1],
                    )
                else:
                    y = _OutNormB200.apply(single, on.weight if mode == 1 else None, on.bias if mode == 1 else None, atom_mask.contiguous(), mode, eps)
                    per_atom = lin(y)  # (d_token % 8 == 0: _b200_glue)
                return _TokenMeanB200.apply(
                    per_atom.contiguous(), atom_mask.contiguous(), atom_to_token.contiguous(), n_token, self.token_pool == "relu"
                )
            single = self.atom_out_norm(single).to(single.dtype)
            per_atom = self.atom_single_rep_to_token_single(single * atom_mask[..., None])
            if per_atom.dtype not in (torch.bfloat16, torch.float32):
                return self.to_tokens(single, atom_mask, atom_to_token, n_token)
            # the token mean as one Function (fixed-order sums: the same bits run to run, which torch's scatter_add does not give)
            return _TokenMeanB200.apply(per_atom.contiguous(), atom_mask.contiguous(), atom_to_token.contiguous(), n_token)
        bf = torch.bfloat16
        act = bf if pos.dtype == bf or torch.is_autocast_enabled("cuda") else torch.float32
        w = self.to_atom_single_cond.weight
        wt = eval_form("ife_b200_wt", [w], lambda: w.detach().float().t().contiguous())
        pos32 = pos.detach().contiguous() if pos.dtype in (torch.float32, bf) else pos.detach().float().contiguous()
        m = b * n
        kidx = dev.index if dev.index is not None else torch.cuda.current_device()
        with torch.cuda.device(dev), torch.autocast("cuda", enabled=False):
            K = sm100.embed_glue(kidx)
            single = torch.empty(b, n, 128, device=dev, dtype=act)
            fmask = (atom_mask if esm else ref_mask).contiguous()
            lnw, lnb, eps = (ln.weight, ln.bias, float(ln.eps)) if ln is not None else (wt, wt, 0.0)
            grid = (min((m + 7) // 8, 16 * num_sms()), 1, 1)
            K.feat(
                grid,
                (256, 1, 1),
                pos32,
                charge.contiguous(),
                element.contiguous(),
                fmask,
                wt,
                lnw,
                lnb,
                single,
                int(m),
                int(esm),
                int(act == bf),
                int(ln is not None),
                eps,
                None,
                int(pos32.dtype == bf),  # round 3 part 14: bf16 positions read in place (flag bit 0); W^T the cached form
            )
        if rope_side:
            cur0.wait_stream(s2r)
        single = self.atom_transformer(single, single, params)
        on = self.atom_out_norm
        nmode = 1 if isinstance(on, LayerNorm) else 2 if isinstance(on, RMSNorm) and on.weight is None else 0 if isinstance(on, nn.Identity) else -1
        nok = nmode != 1 or (isinstance(on, LayerNorm) and on.weight is not None and on.bias is not None and on.weight.dtype == torch.float32)
        acc_pre = None
        if _EVAL_OUTNORM and nmode >= 0 and nok and single.dtype in (bf, torch.float32) and single.shape[-1] == 128 and self.d_token % 8 == 0:
            # N2: the terminal norm (mwpipe's fp32 RMSNorm is a 22-us launch at 6144 atoms; bf16 2 us + a cast) as the one-launch norm of the
            # training read-out (embed_glue.cu swa_embed_outnorm_sm100: fp32 statistics, rounded to the activation dtype, times the mask --
            # masked rows are dropped by the scatter anyway), which also zeroes the scatter's accumulator
            eps_n = float(on.eps) if isinstance(on, LayerNorm) else float(on.effective_eps(single.dtype)) if isinstance(on, RMSNorm) else 0.0
            single = single.contiguous()
            mm_ = single.numel() // 128
            acc_pre = torch.empty(b, n_token, self.d_token + 1, device=dev, dtype=torch.float32)
            yn = torch.empty_like(single)
            with torch.cuda.device(dev), torch.autocast("cuda", enabled=False):
                K.outnorm(
                    (min((mm_ + 7) // 8, 16 * num_sms()), 1, 1), (256, 1, 1), single, atom_mask.contiguous(),
                    on.weight.detach().float().contiguous() if nmode == 1 else single, on.bias.detach().float().contiguous() if nmode == 1 else single,
                    yn, int(mm_), int(nmode), int(single.dtype == bf), eps_n, acc_pre, int(acc_pre.numel() // 4) if acc_pre.numel() % 4 == 0 else 0,
                )
            if acc_pre.numel() % 4:
                acc_pre.zero_()
            single = yn
        else:
            single = self.atom_out_norm(single).to(single.dtype)
        lin0 = self.atom_single_rep_to_token_single[0]
        if _EVAL_RW and single.dtype == bf and lin0.bias is None and lin0.weight.dtype == torch.float32:
            # N2: the master read-out weight's bf16 form made once per weight version (inference form: a graph replay runs no cast launch)
            wrb = eval_form("ife_b200_rw_bf16", [lin0.weight], lambda: lin0.weight.detach().to(bf).contiguous())
            per_atom = F.linear(single, wrb)
        else:
            per_atom = lin0(single)  # the read-out GEMM; its mask and ReLU run in the scatter launch
        t = self.d_token
        if per_atom.dtype not in (bf, torch.float32) or per_atom.shape[-1] != t:
            return self.to_tokens(single, atom_mask, atom_to_token, n_token)
        per_atom = per_atom.contiguous()
        with torch.cuda.device(dev), torch.autocast("cuda", enabled=False):
            acc = acc_pre if acc_pre is not None else torch.zeros(b, n_token, t + 1, device=dev, dtype=torch.float32)
            out = torch.empty(b, n_token, t, device=dev, dtype=per_atom.dtype)
            cap = 16 * num_sms()
            (K.scatter8 if (_SCATTER8 and n % 8 == 0) else K.scatter)(
                (min(((b * (n // 8) if (_SCATTER8 and n % 8 == 0) else b * n) * (t // 8) + 255) // 256, cap), 1, 1),
                (256, 1, 1),
                per_atom,
                atom_mask.contiguous(),
                atom_to_token.contiguous(),
                acc,
                int(b),
                int(n),
                int(n_token),
                int(t),
                int(self.token_pool == "relu"),
                int(per_atom.dtype == bf),
            )
            K.tokmean(
                (min((b * n_token * (t // 8) + 255) // 256, cap), 1, 1),
                (256, 1, 1),
                acc,
                out,
                int(b * n_token),
                int(t),
                int(per_atom.dtype == bf),
            )
        return out

    def _forward_triton(self, *args, **kwargs) -> torch.Tensor:
        return self._forward_pytorch(*args, **kwargs)  # the composition; the transformer and the norms run their "triton" backends

    def atom_single(
        self,
        pos: torch.Tensor,
        ref_mask: torch.Tensor,
        element: torch.Tensor,
        charge: torch.Tensor,
        space_uid: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """The atom stream after the transformer and the terminal norm (none for ``"esmfold2"``)."""
        if self.features == "esmfold2":
            projected = self.to_atom_single_cond(atom_single_features_esmfold2(pos, charge, atom_mask, element))
            # the LayerNorm runs in fp32 under autocast; hand the transformer the projection's dtype, as the legacy layout does
            single = self.atom_norm(projected).to(projected.dtype)
        else:
            single = self.to_atom_single_cond(atom_single_features(pos, ref_mask, element, charge))
        cos, sin = self.atom_transformer.build_rope(pos, space_uid)
        params = build_attention_params(cos, sin, atom_mask, num_aug=1)
        single = self.atom_transformer(single, single, params)  # the single conditions itself
        return self.atom_out_norm(single).to(single.dtype)  # the LayerNorm of token_pool="linear" runs in fp32 under autocast

    def to_tokens(
        self,
        atom_single: torch.Tensor,
        atom_mask: torch.Tensor,
        atom_to_token: torch.Tensor,
        n_token: int,
    ) -> torch.Tensor:
        """``Linear + ReLU`` (``token_pool="linear"``: ``Linear``) per atom, then the mean over each token's atoms: one scatter-add (its
        backward a gather), summed in fp32."""
        per_atom = self.atom_single_rep_to_token_single(atom_single * atom_mask[..., None])
        idx = atom_to_token.long()
        src = per_atom.to(wide(per_atom))
        total = src.new_zeros(idx.shape[0], n_token, src.shape[-1]).scatter_add(1, idx[..., None].expand_as(src), src)
        count = src.new_zeros(idx.shape[0], n_token).scatter_add(1, idx, atom_mask.to(src.dtype))
        return (total / count[..., None].clamp(min=1.0)).to(per_atom.dtype)

    def _forward_pytorch(
        self,
        pos: torch.Tensor,
        ref_mask: torch.Tensor,
        element: torch.Tensor,
        charge: torch.Tensor,
        space_uid: torch.Tensor,
        atom_mask: torch.Tensor,
        atom_to_token: torch.Tensor,
        n_token: int,
    ) -> torch.Tensor:
        single = self.atom_single(pos, ref_mask, element, charge, space_uid, atom_mask)
        return self.to_tokens(single, atom_mask, atom_to_token, n_token)

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        """``samples`` systems of ``length`` atoms (real atoms first), 8 atoms per token and per conformer, elements 1..19, charges -1..1."""
        mask = key_mask(samples, length, device=device)
        index = (torch.arange(length, device=device) // 8).expand(samples, length).contiguous()
        return {
            "pos": random_input((samples, length, 3), dtype=dtype, device=device, generator=generator, scale=5.0),
            "ref_mask": mask,
            "atom_mask": mask,
            "space_uid": index,
            "atom_to_token": index,
            "n_token": (length + 7) // 8,
            "element": torch.randint(1, 20, (samples, length), device=device, generator=generator),
            "charge": torch.randint(-1, 2, (samples, length), device=device, generator=generator),
        }
