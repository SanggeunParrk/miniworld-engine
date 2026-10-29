"""B200 (sm_100a) attention glue of the DiT variant stacks (Boltz-2, RoseTTAFold3, Chai-1; ``modules.dit_variant``).

The variants' attention is the AF3 math the engine's B200 cores already compute, so B200 runs those cores (tcgen05 / TMEM / TMA, built by
``augmented_attention/cuda/sm100`` and ``sm100_atom_local``) around the variants' CUDA row kernels (``rows.cu`` / ``train_rows.cu``, built for
sm_100a) and cuBLAS:

  token stack (768 wide, 16 heads x 48; ``augmented_attention/cuda/sm100``)
    inference   ``GatedInferenceCore``: ``attn_inf.cu`` (bf16, q | k | v | g, logits pre-scaled to exp2 units, sigmoid(g) o over q) and
                ``attn_inf_tf32.cu`` (fp32 on TF32 tensor cores, q | k | g + v^T). The variants' bias of every block is one head-major
                [nb 16, L, L] tensor in log2 units, the masked keys -inf; a block without a gating query gets a gate column of +1e4.
    training    ``attn_fwd2`` / ``attn_dkv`` + ``attn_dqb`` (bf16) and ``attn_fwd_tf32`` / ``attn_dkv_tf32`` + ``attn_dqb_tf32`` (fp32): q, k, v
                unscaled (the cores apply 1 / sqrt 48), the bias [16, L, L] in natural units with the masked keys -inf, the fp32 O gated by
                ``ogate_fwd`` (train_rows.cu); dbias comes back in natural units; the backward is the deterministic one of the
                AugmentedAttention core (``-DDET=1 -DDQPART=1`` builds, dQ partials summed in chunk order). The forward pairs the samples: an odd A is padded by one
                zero sample (its rows are dropped; its gradients are exactly zero).
  atom stack (128 wide, 4 heads x 32, 32 x 128 windows; ``sm100_atom_local``)
    the local DiT's window kernels: ``lattn_fwd`` / ``lattn_dq`` / ``lattn_dkv`` (bf16) and their ``*_tf32`` twins, the pair bias ``lbias``
    (natural units, fp32, [4, nwin, 32, 128]), the slid key window inside the attention kernels (``slide.cuh`` edge CTAs, ``lslide`` for
    their dK / dV partials). N is padded to a multiple of 128 (the local DiT's rule), the padded atoms masked. Chai-1's same-conformer mask
    is folded into the bias as -inf at the keys each window really reads (the slid window's keys for "slide": the edge CTAs read the same
    bias slots over their slid keys).
"""

from __future__ import annotations

import functools

import torch

from miniworld_engine.kernels.dit_variants.reference import ATOM_KEYS, ATOM_QUERIES, atom_window_keys

LOG2E = 1.4426950408889634
HEADS, HEAD_DIM = 16, 48
#: the atom stacks' padded length: a multiple of the local DiT kernels' 128-row tile
ATOM_TILE = 128


def is_b200(t: torch.Tensor) -> bool:
    return t.is_cuda and torch.cuda.get_device_capability(t.device) == (10, 0)


def _index(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


# ------------------------------------------------------------------------------------------------------------------- token
def token_inference_supported(dtype: torch.dtype, length: int, d: int, heads: int, device_index: int) -> bool:
    """The B200 gated inference core takes this call: bf16 / fp32, a token DiT head layout, L a multiple of 8."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    return sm100.inference_core_supported(dtype, length, d, heads, device_index)


@functools.lru_cache(maxsize=8)
def _gated_core(device_index: int, fp32: bool, heads: int, dh: int):
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    return sm100.GatedInferenceCore(device_index, torch.float32 if fp32 else torch.bfloat16, heads, dh)


_RUNS: dict = {}


@torch.compiler.disable
def gated_inference(qkvg: torch.Tensor, bias: torch.Tensor, block: int, samples: int, heads: int, dh: int, vt: torch.Tensor | None = None):
    """sigmoid(g) softmax(q k^T + bias[block]) v written over q (``GatedInferenceCore``). bf16: ``qkvg`` [S L, 4 D] = q | k | v | g;
    fp32: ``qkvg`` [S L, 3 D] = q | k | g and ``vt`` = v^T [D, S L]. Logits in exp2 units (q and the bias pre-scaled).

    The variants allocate their step buffers per call, so the launch bindings (TMA descriptors, ~5-10 us of host time each) are cached by
    everything they encode -- addresses and geometry -- WITHOUT holding the tensors (the core's own cache keeps them alive, which would pin
    every call's buffers): a reused address with the same geometry encodes the same descriptors."""
    core = _gated_core(_index(qkvg), qkvg.dtype is torch.float32, heads, dh)
    length = qkvg.shape[0] // samples
    key = (_index(qkvg), qkvg.dtype, qkvg.data_ptr(), tuple(qkvg.shape), qkvg.stride(0), None if vt is None else vt.data_ptr(),
           bias.data_ptr(), tuple(bias.shape), bias.dtype, block, samples, heads, dh)
    run = _RUNS.get(key)
    if run is None:
        if len(_RUNS) >= 512:
            _RUNS.clear()
        run = core._bind(qkvg, vt, bias, block, samples, length)
        for m in run.keep[0]:
            m.keep = None
        run.keep = None
        _RUNS[key] = run
    run()
    return qkvg


def token_attn_fwd(q, k, v, bias, samples: int, length: int):
    """O [A L, 768] fp32 and LSE [A, 16, L] (log2) of softmax(q k^T / sqrt 48 + bias) v: q, k, v [A L, 768] rows (bf16 views with 16-byte
    rows, or fp32), bias [16, L, L] natural units in their dtype (masked keys -inf). A even, L a multiple of 128."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm100

    if q.dtype is torch.float32:
        return sm100.forward_tf32(q, k, v, bias, samples, length)
    return sm100.forward(q, k, v, bias, samples, length)


def token_attn_bwd(q, k, v, do, bias, lse, dd, samples: int, length: int):
    """dQ, dK, dV [A L, 768] fp32 and dbias [16, L, L] fp32 (natural units) of ``token_attn_fwd``; ``do`` in the operands' dtype, dd =
    rowsum(dO O) [A, 16, L] fp32. Deterministic: the AugmentedAttention core's backward (``sm100/fused_pair.py`` ``_bwd_bf16`` /
    ``_bwd_tf32``: attn_dkv / attn_dqb and the TF32 twins built ``-DDET=1 -DDQPART=1``, the two MMA issuers in a fixed order, one dQ
    partial per key chunk) and the partials summed here in chunk order (the default builds' reduce-adds and unordered issuers changed
    dQ -- and in fp32 dK / dV -- run to run)."""
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import fused_pair

    bwd = fused_pair._bwd_tf32 if q.dtype is torch.float32 else fused_pair._bwd_bf16
    dq_parts, dk, dv, db = bwd(q, k, v, do, bias, lse, dd, samples, length, HEADS, HEAD_DIM)
    return dq_parts.sum(0), dk, dv, db


@functools.lru_cache(maxsize=8)
def _swiglu_op(device_index: int, k: int, h: int, save_ab: bool):
    from miniworld_engine.kernels.conditioned_transition.cuda import gemm_swiglu

    return gemm_swiglu.GemmSwiglu(device_index, K=k, H=h, save_ab=save_ab)


_SWIGLU_RUNS: dict = {}


@torch.compiler.disable
def swiglu_gemm(x: torch.Tensor, w_ab: torch.Tensor, out: torch.Tensor, ab: torch.Tensor | None = None) -> bool:
    """The token DiT's expand GEMM with the SwiGLU in its epilogue (``gemm_swiglu2_sm100``, 2-CTA persistent): out = silu(a) b for
    [a | b] = x [Wa; Wb]^T -- the variants' ``expand`` weight is exactly [Wa; Wb] (gate rows first) -- and, with ``ab`` (training), the
    bf16 [a | b] for the backward. False (nothing launched) where it does not serve: bf16, K 768 (the token stacks), M >= its row
    threshold (``gemm_swiglu.MIN_ROWS``: below it cuBLAS + the row pass is as fast). Bindings cached by address without pinning buffers."""
    from miniworld_engine.kernels.conditioned_transition.cuda import gemm_swiglu

    if x.shape[1] != 768 or not gemm_swiglu.supported(x, w_ab):
        return False
    idx = _index(x)
    op = _swiglu_op(idx, x.shape[1], w_ab.shape[0] // 2, ab is not None)
    key = (idx, x.data_ptr(), tuple(x.shape), w_ab.data_ptr(), tuple(w_ab.shape), out.data_ptr(), None if ab is None else ab.data_ptr())
    run = _SWIGLU_RUNS.get(key)
    if run is None:
        if len(_SWIGLU_RUNS) >= 256:
            _SWIGLU_RUNS.clear()
        run = op._bind(x, w_ab, out, ab)
        for m in run.keep[0]:
            if hasattr(m, "keep"):
                m.keep = None
        run.keep = None
        _SWIGLU_RUNS[key] = run
    run()
    return True


# -------------------------------------------------------------------------------------------------------------------- atom
def atom_padded(n: int) -> int:
    return (n + ATOM_TILE - 1) // ATOM_TILE * ATOM_TILE


def atom_same_conformer(uid: torch.Tensor, kmask: torch.Tensor, n: int, key_window: str) -> torch.Tensor:
    """[nwin, 32, 128] bool: key slot j of window w is kept for query i (Chai-1: the same conformer, or a padding query) -- at the keys the
    window really reads (the reference's ``atom_window_keys``: slid for "slide", clipped for "pad")."""
    nwin = n // ATOM_QUERIES
    idx, _ = atom_window_keys(n, kmask, key_window, uid.device)
    return (uid.view(nwin, ATOM_QUERIES, 1) == uid[idx][:, None, :]) | ~kmask.view(nwin, ATOM_QUERIES, 1)


def _local():
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom_local as local

    return local


class AtomPairBias(torch.autograd.Function):
    """bias [4, nwin, 32, 128] fp32, natural units: LN(z) gamma . Wb over the trunked pair z (``lbias.cu``; bf16 z, or fp32 z on the fp32
    twin), and its backward (dz in z's dtype, dgamma, dWb)."""

    @staticmethod
    def forward(ctx, z, gamma, wb, eps):  # noqa: D102
        local = _local()
        ctx.save_for_backward(z, gamma, wb)
        ctx.eps = eps
        if z.dtype == torch.float32:
            return local.pair_bias_fwd32(z, gamma, wb, eps)
        return local.pair_bias_fwd(z.to(torch.bfloat16).contiguous(), gamma, wb, eps)

    @staticmethod
    def backward(ctx, dbias):  # noqa: D102
        local = _local()
        z, gamma, wb = ctx.saved_tensors
        if z.dtype == torch.float32:
            dz, dgamma, dwb = local.pair_bias_bwd32(z, gamma, wb, dbias.contiguous(), ctx.eps)
        else:
            dz, dgamma, dwb = local.pair_bias_bwd(z.to(torch.bfloat16).contiguous(), gamma, wb, dbias.contiguous(), ctx.eps)
        return dz.to(z.dtype), dgamma.to(gamma.dtype), dwb.to(wb.dtype), None


def atom_bias(z, gamma, wb, eps, same=None):
    """One block's bias (``AtomPairBias``, differentiable), the same-conformer mask folded in as -inf."""
    bias = AtomPairBias.apply(z, gamma, wb, eps)
    return bias if same is None else bias.masked_fill(~same[None], float("-inf"))


@torch.compiler.disable
def atom_attn_fwd(q, k, v, bias, kmask, slide: bool):
    """O [A, N, 128] (q's dtype) and LSE [A, 4, N] of the window attention; ``slide``: AlphaFold 3's slid key window, the windows it moves
    run as edge CTAs of the same launch (``sm100_atom_local/slide.cuh``; the real atom count is ``kmask``'s population count, N without
    one). q, k, v [A, N, 128] (bf16 contiguous, or fp32 views); kmask [N] bool or None."""
    local = _local()
    if q.dtype == torch.float32:
        return local.attn_fwd32(q, k, v, bias, kmask, slide=slide)
    return local.attn_fwd(q, k, v, bias, kmask, slide=slide)


def atom_attn_bwd(q, k, v, do, bias, lse, dd, dp, kmask, slide: bool):
    """dQ, dK, dV into the column blocks 1, 2, 3 (of 128) of dp [A N, 512] (the operands' dtype) and dbias [4, nwin, 32, 128] fp32, natural
    units, summed over the samples (the slid window's moved windows in edge CTAs, their dK / dV partials added by ``lslide.cu``)."""
    local = _local()
    a, n, d = q.shape
    if q.dtype == torch.float32:
        return local.attn_bwd32(q, k, v, do.view(a, n, d), bias, lse, dd, dp, (d, 2 * d, 3 * d), kmask, slide=slide)
    return local.attn_bwd(q, k, v, do.view(a, n, d), bias, lse, dd, dp[:, d:], kmask, slide=slide)


__all__ = [
    "ATOM_KEYS",
    "AtomPairBias",
    "atom_attn_bwd",
    "atom_attn_fwd",
    "atom_bias",
    "atom_padded",
    "atom_same_conformer",
    "gated_inference",
    "is_b200",
    "swiglu_gemm",
    "token_attn_bwd",
    "token_attn_fwd",
    "token_inference_supported",
]
