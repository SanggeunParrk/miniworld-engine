"""Fused bias-only token DiT inference on B200 (sm_100), wired to ``BiasOnlyDiTBlock``'s parameter contract.

CUDA and cuBLAS only (``kernels/bias_only_dit/cuda``: the runner, the ``pv_gate_inf`` core, the row kernels; the token DiT's
CUDA rows and SwiGLU GEMM). Inference only: the weights are packed once and the pack reused while every weight's
(pointer, version) is unchanged, CUDA-graph replays included; the attention weights P = softmax(pair bias) are made once per
pair and mask (keyed on the tensors' pointer and version) -- they depend on nothing else.

fp32 (single, cond and pair fp32, the block's weights fp32, no CUDA autocast) runs the same schedule in fp32 on TF32 tensor
cores (``kernels/bias_only_dit/cuda/tf32.py``: the ``pv_gate_tf32`` core, the fp32 rows and pair bias, cuBLAS forced to TF32);
the fp32 kernels build on first use and a failed build warns once and keeps the module path. The fp32 step runs as three kernels
per block -- ``bo_front_tf32`` (LN + AdaLN + v|g GEMM), ``pv_gate_tf32``, ``bo_tail_tf32`` (everything after the core) -- behind
conditioning tables hoisted once per conditioning tensor (``runner._tables3``; ``kernels._capture.lookup_inputs`` rules);
settings.bias_only_dit_inf3=0 (read per call) keeps the 12-launch cuBLAS + rows step. bf16 likewise runs three kernels per block
(``kernels/bias_only_dit/cuda/inf3_bf16.py``: ``bo_front_bf16``, ``pv_gate_inf`` with PDL, ``bo_tail_bf16`` or the pair tail
``bo_tail2_bf16``) behind bf16 tables (``runner._tables3b``); settings.bias_only_dit_inf3_bf16=0 (read per call) keeps the 12-launch
bf16 step, which is also what a failed build falls back to.

``serves()`` is the whole gate: no autograd, the engine's kernels (implementation MINIWORLD), B200 or H100, bf16 or fp32, the
token widths (768; the attention as 16 heads x 48, 24 x 32, 12 x 64 or 16 x 64 / cond 384 / pair 128 / transition n = 2), B == 1, L a multiple of 128 (bf16 up to 3072 on B200 -- past 768 the
attention core streams P, ``pv_stream_sm100.cu`` -- fp32 up to 768; 3072 on H100: ``core_supported``), a key mask [1, L] or
none, LayerNorm eps 1e-5. B200 serves any other L up to those bounds by padding on the host (``update``): single / cond with
zero rows, the pair (or the hoisted bias) with zero rows and columns, the key mask with False -- so the padded keys get
P = 0 and no real row sees them -- and returns the first L rows. The conditioning may be one per sample or one shared by the samples (sample axis 1 or stride 0).
settings.bias_only_dit=0 turns it off. Anything else keeps the module's PyTorch composition.

A block given its pair bias precomputed (``bias`` [1, H, L, L] in the step's dtype, from ``pair_bias_all``) runs the same step on
P = softmax(bias) made per call (the softmax rows, the key mask folded in) instead of the per-pair cache: the caller owns the bias.
A sampler is better served by the pair itself (the cache makes P once per pair for every solver step), which is why MiniWorld's
``hoist_pair_bias`` hoists only under autograd.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from miniworld_engine import settings
from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import opaque

D, DC, DP = 768, 384, 128
LAYOUTS = (
    (16, 48),
    (24, 32),
    (12, 64),
    (16, 64),
)  # (n_head, head width) served (768 or 1024 attention channels)

WEIGHTS = (
    "attention.to_value.weight",
    "attention.to_gate.weight",
    "attention.to_out.weight",
    "attention.ada_ln_in.ln_cond.weight",
    "attention.ada_ln_in.to_scale.weight",
    "attention.ada_ln_in.to_scale.bias",
    "attention.ada_ln_in.to_bias.weight",
    "attention.to_scale.weight",
    "attention.to_scale.bias",
    "attention.ln_pair.weight",
    "attention.to_bias.weight",
    "transition.ada_ln_in.ln_cond.weight",
    "transition.ada_ln_in.to_scale.weight",
    "transition.ada_ln_in.to_scale.bias",
    "transition.ada_ln_in.to_bias.weight",
    "transition.to_scale.weight",
    "transition.to_scale.bias",
    "transition.expand_a.weight",
    "transition.expand_b.weight",
    "transition.squeeze.weight",
)
#: Packed runners keyed by the weights' (pointer, version): see ``_infer``.
_RUNNERS: dict = {}


def serves(module, single, cond, pair, mask=None, bias=None) -> bool:
    from miniworld_engine.modules.exceptions import ImplementationType

    if (not settings.current().bias_only_dit) or torch.is_grad_enabled():
        return False
    if module.implementation != ImplementationType.MINIWORLD:
        return False
    if settings.current().force_portable_kernels:
        return False
    dt = single.dtype
    src = (
        pair if bias is None else bias
    )  # the pair, or the block's hoisted bias [1, H, L, L]
    if src is None or not (
        single.is_cuda
        and dt in (torch.bfloat16, torch.float32)
        and cond.dtype is dt
        and src.dtype is dt
    ):
        return False
    if dt is torch.float32 and (
        module.attention.to_value.weight.dtype is not torch.float32
        or torch.is_autocast_enabled("cuda")
    ):
        return False  # fp32 = an fp32 block; under autocast the module path keeps its casts
    if single.ndim != 4 or single.shape[1] != 1 or single.shape[-1] != D or single.shape[2] < 1:
        return False
    if single.shape[2] % 128 and not _pads(single):
        return False
    A, _, L, _ = single.shape
    if (
        cond.ndim != 4
        or tuple(cond.shape[1:]) != (1, L, DC)
        or cond.shape[0] not in (1, A)
    ):
        return False
    if bias is None and tuple(pair.shape) != (1, L, L, DP):
        return False
    if bias is not None and (
        tuple(bias.shape) != (1, module.attention.n_head, L, L)
        or bias.device != single.device
    ):
        return False
    if mask is not None and not (mask.ndim == 2 and tuple(mask.shape) == (1, L)):
        return False
    at = module.attention
    da = at.to_value.weight.shape[0]
    if (
        at.n_head,
        da // at.n_head,
    ) not in LAYOUTS or module.transition.expand_a.weight.shape[0] != 2 * D:
        return False
    norms = (
        at.ada_ln_in.ln_in,
        at.ada_ln_in.ln_cond,
        at.ln_pair,
        module.transition.ada_ln_in.ln_in,
        module.transition.ada_ln_in.ln_cond,
    )
    if any(norm.eps != 1e-5 for norm in norms):
        return False
    from miniworld_engine.kernels.bias_only_dit.interface import core_supported

    idx = (
        single.device.index
        if single.device.index is not None
        else torch.cuda.current_device()
    )
    return core_supported(single.dtype, _lp(L), da, at.n_head, idx)


def _pads(single) -> bool:
    """B200 pads an L off the cores' 128-row tile on the host (``update``); H100 keeps its multiples of 128."""
    return torch.cuda.get_device_capability(single.device) == (10, 0)


def _lp(L: int) -> int:
    return -(-L // 128) * 128


def _fake(single, cond, pair, mask, weights):
    return torch.empty_like(single)


def _runner(single, weights):
    """The block's packed runner, reused while every weight's (pointer, version) is unchanged (see ``_infer``)."""
    from miniworld_engine.kernels.bias_only_dit.interface import FusedBiasOnlyDiT

    def build():
        block = SimpleNamespace()
        for path, weight in zip(WEIGHTS, weights, strict=True):
            node = block
            names = path.split(".")
            for name in names[:-1]:
                if not hasattr(node, name):
                    setattr(node, name, SimpleNamespace())
                node = getattr(node, name)
            setattr(node, names[-1], weight)
        block.attention.n_head = block.attention.to_bias.weight.shape[0]
        return FusedBiasOnlyDiT([block], dtype=single.dtype)

    # The weights do not change between the calls of a sampling run; packing is tens of small kernels. Reuse the pack
    # while every weight's (pointer, version) is the same -- an in-place update bumps ``_version`` and misses. Scoped
    # to the CUDA-graph capture (``kernels._capture``): a capture packs once, recorded, so a replay packs the weights as
    # they are then.
    wkey = (single.dtype, *((w.data_ptr(), w._version) for w in weights))
    return _capture.lookup(_RUNNERS, wkey, build)


@opaque(fake=_fake, name="bias_only_dit_infer")
def _infer(
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor,
    weights: list[torch.Tensor],
) -> torch.Tensor:
    with torch.cuda.device(single.device):
        runner = _runner(single, weights)
        # The attention weights depend on the pair and the mask only: the same at every diffusion step of a sample and for
        # every augmented sample. Keyed by the caller's pair / mask tensors (pointer, version, layout).
        pkey = _capture.scoped(
            (
                pair.data_ptr(),
                pair._version,
                tuple(pair.shape),
                pair.stride(),
                mask.data_ptr(),
                mask._version,
            )
        )
        hit = runner.__dict__.get("_p_cache")
        if pkey is not None and hit is not None and hit[0] == pkey:
            P = hit[1]
        else:
            L, Lp = pair.shape[1], single.shape[2]
            if Lp != L:  # B200 host padding (``update``): zero pair rows / columns, the padded keys masked (P = 0)
                F = torch.nn.functional
                P = runner.hoist(F.pad(pair, (0, 0, 0, Lp - L, 0, Lp - L)).contiguous(), F.pad(mask.to(torch.bool), (0, Lp - L), value=False))
            else:
                P = runner.hoist(pair.contiguous(), mask)
            if pkey is not None:
                # the entry keeps its key tensors alive: a freed address cannot alias another tensor
                runner._p_cache = (pkey, P, (pair, mask))
        return runner.step(single, cond, P)


def _fake_bias(single, cond, bias, mask, weights):
    return torch.empty_like(single)


@opaque(fake=_fake_bias, name="bias_only_dit_infer_bias")
def _infer_bias(
    single: torch.Tensor,
    cond: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    weights: list[torch.Tensor],
) -> torch.Tensor:
    """``_infer`` on a hoisted bias [1, H, L, L]: P = softmax(bias) over the keys (masked keys 0), made per call. L is single's
    (``update`` pads the bias and the mask with it)."""
    with torch.cuda.device(single.device):
        runner = _runner(single, weights)
        H, L = bias.shape[1], bias.shape[2]
        m = mask.reshape(L).to(torch.bool).contiguous()
        P = torch.empty(H, L, L, device=single.device, dtype=single.dtype)
        if single.dtype is torch.float32:
            from miniworld_engine.kernels.bias_only_dit.cuda import tf32

            tf32.rows32().softmax_rows_cuda(bias.reshape(H * L, L), P.view(H * L, L), m)
        else:
            from miniworld_engine.kernels.bias_only_dit import cuda as C

            C.softmax_rows(bias.reshape(H * L, L), P.view(H * L, L), m)
        return runner.step(single, cond, P)


def _fake_psb(single, cond, bias, mask, weights):
    return torch.empty_like(single)


@opaque(fake=_fake_psb, name="bias_only_dit_infer_psb")
def _infer_psb(
    single: torch.Tensor,
    cond: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor | None,
    weights: list[torch.Tensor],
) -> torch.Tensor:
    """The inference step for a bias of its own per sample: single / cond [S, 1, L, d], ``bias`` [H, S, L, L] head-major (natural
    logits, contiguous), ``mask`` bool [1, L] / [S, L] or None. ``FusedBiasOnlyDiT.step_psb``: nothing is shared across the samples
    (no P cache), the attention core softmaxes each (head, sample) bias itself."""
    with torch.cuda.device(single.device):
        runner = _runner(single, weights)
        H, S, L = bias.shape[0], bias.shape[1], bias.shape[2]
        m = None if mask is None else mask.to(torch.bool).contiguous().reshape(-1)
        return runner.step_psb(single, cond, bias.reshape(H * S * L, L), m)


#: Most bias bytes of one pair-fed chunk (``update_psb`` with a pair): the samples of a larger call run as consecutive chunks of
#: ``chunk_samples`` samples, a function of the shapes only.
PSB_CHUNK_BYTES = 2 << 30


def chunk_samples(n_samples: int, n_head: int, length: int, itemsize: int) -> int:
    """Samples per chunk so that one chunk's bias [H, S, L, L] stays within ``PSB_CHUNK_BYTES`` (at least one sample)."""
    return max(1, min(n_samples, PSB_CHUNK_BYTES // (n_head * length * length * itemsize)))


def serves_psb(module, single, cond, pair, mask=None, bias=None) -> bool:
    """Inference with a pair (``pair`` [B, L, L, 128]) or a hoisted bias (``bias`` [B, H, L, L]) of its own per sample: ``single`` /
    ``cond`` [1, B, L, d], B > 1, no autograd, H100 (bf16 or fp32 (TF32); fp32 beyond 512 takes the cuBLAS core) or B200 (bf16 or
    fp32: the per-sample softmax and the ``-DPSB`` build of ``pv_gate_inf`` / ``pv_gate_tf32``), L a multiple of 128 up to 768, a key
    mask [1, L] / [B, L] or none."""
    from miniworld_engine.modules.exceptions import ImplementationType

    if (not settings.current().bias_only_dit) or torch.is_grad_enabled():
        return False
    if module.implementation != ImplementationType.MINIWORLD or settings.current().force_portable_kernels:
        return False
    from miniworld_engine.kernels.bias_only_dit.cuda import h100_enabled

    dt = single.dtype
    src = pair if bias is None else bias
    if src is None or not single.is_cuda:
        return False
    b200 = torch.cuda.get_device_capability(single.device) == (10, 0)
    if not ((h100_enabled(single.device) or b200) and dt in (torch.bfloat16, torch.float32)):
        return False
    if not (cond.dtype is dt and src.dtype is dt):
        return False
    if dt is torch.float32 and (module.attention.to_value.weight.dtype is not torch.float32 or torch.is_autocast_enabled("cuda")):
        return False
    if single.ndim != 4 or single.shape[0] != 1 or single.shape[1] < 2 or single.shape[-1] != D or single.shape[2] % 128:
        return False
    _, B, L, _ = single.shape
    if L > 768 or tuple(cond.shape) != (1, B, L, DC):
        return False
    at = module.attention
    H = at.n_head
    if bias is None and tuple(pair.shape) != (B, L, L, DP):
        return False
    if bias is not None and (tuple(bias.shape) != (B, H, L, L) or bias.device != single.device):
        return False
    if mask is not None and not (mask.ndim == 2 and tuple(mask.shape) in ((1, L), (B, L))):
        return False
    da = at.to_value.weight.shape[0]
    if (H, da // H) not in LAYOUTS or module.transition.expand_a.weight.shape[0] != 2 * D:
        return False
    norms = (at.ada_ln_in.ln_in, at.ada_ln_in.ln_cond, at.ln_pair, module.transition.ada_ln_in.ln_in, module.transition.ada_ln_in.ln_cond)
    return all(n.eps == 1e-5 for n in norms)


def update_psb(module, single, cond, pair, mask, bias=None):
    """One block on a bias of its own per sample, inference. ``bias`` [B, H, L, L] (any layout; the hoist's head-major buffer is read
    without a copy) or the pair [B, L, L, 128], whose bias is then made per chunk of samples (one pass over the pair, LayerNorm on chip) so
    that a large B never holds more than ``PSB_CHUNK_BYTES`` of bias. Call ``serves_psb`` first."""
    weights = [module.get_parameter(name) for name in WEIGHTS]
    _, B, L, _ = single.shape
    s, c = single.reshape(B, 1, L, -1), cond.reshape(B, 1, L, -1)
    s = s if s.is_contiguous() else s.contiguous()
    if mask is not None and mask.shape[0] == 1:
        mask = mask.reshape(1, L)
    if bias is not None:
        hb = bias.permute(1, 0, 2, 3)
        hb = hb if hb.is_contiguous() else hb.contiguous()
        return _infer_psb(s, c, hb, mask, weights).reshape(single.shape)
    H = module.attention.n_head
    k = chunk_samples(B, H, L, single.element_size())
    from miniworld_engine.integrations import bias_only_dit_hoist as hoist

    outs = []
    for b0 in range(0, B, k):
        b1 = min(B, b0 + k)
        bc = hoist.pair_bias_all_infer([module.attention], pair[b0:b1])[0].permute(1, 0, 2, 3)  # [H, S, L, L]
        mc = mask if (mask is None or mask.shape[0] == 1) else mask[b0:b1]
        outs.append(_infer_psb(s[b0:b1], c[b0:b1], bc, mc, weights))
    return (outs[0] if len(outs) == 1 else torch.cat(outs, 0)).reshape(single.shape)


_ALL_TRUE: dict = {}
#: padded conditioning per caller tensor (``_padded_cond``)
_PAD_COND: dict = {}


def _padded_cond(cond, Lp):
    """cond with zero rows to Lp (a shared cond stays one table), one tensor per caller tensor (address, version, layout; a weak reference
    guards a reused address): the runner's conditioning tables are keyed on the tensor they read, so a sampler that hands every block the
    same cond gets the same padded tensor, and the tables made from it, as an unpadded call would. Scoped like those tables
    (``kernels._capture.lookup_inputs``)."""
    import weakref

    shared = cond.shape[0] == 1 or cond.stride(0) == 0
    src = cond[:1] if shared else cond
    key = ("padc", cond.data_ptr(), cond._version, tuple(cond.shape), cond.stride(), cond.dtype, cond.device, Lp)
    return _capture.lookup_inputs(
        _PAD_COND,
        key,
        lambda: (weakref.ref(cond), torch.nn.functional.pad(src, (0, 0, 0, Lp - src.shape[2]))),
        limit=16,
        valid=lambda e: e[0]() is cond,
        alive=lambda e: e[0]() is not None,
    )[1]


def update(module, single, cond, pair, mask, bias=None):
    if (
        mask is None
    ):  # one all-true mask per (L, device), so the P cache key stays stable
        k = (single.shape[2], single.device)
        mask = _ALL_TRUE.get(k)
        if mask is None:
            mask = _ALL_TRUE[k] = torch.ones(
                (1, single.shape[2]), device=single.device, dtype=torch.bool
            )
    weights = [module.get_parameter(name) for name in WEIGHTS]
    L = single.shape[2]
    Lp = _lp(L)
    if Lp != L:
        # B200 at L % 128 != 0: zero single / cond rows (a shared cond stays one table), the pair / bias padded with zero rows and
        # columns and the padded keys masked out (P = 0 there); the padded query rows are computed and dropped. The pair is padded
        # inside ``_infer`` on a P-cache miss only, so the cache stays keyed on the caller's pair.
        F = torch.nn.functional
        s, c = F.pad(single, (0, 0, 0, Lp - L)), _padded_cond(cond, Lp)
        if bias is not None:
            mp = F.pad(mask.to(torch.bool).reshape(1, L), (0, Lp - L), value=False)
            out = _infer_bias(s, c, F.pad(bias, (0, Lp - L, 0, Lp - L)).contiguous(), mp, weights)
        else:
            out = _infer(s, c, pair, mask, weights)
        return out[:, :, :L].contiguous()
    if bias is not None:
        return _infer_bias(single.contiguous(), cond, bias.contiguous(), mask, weights)
    return _infer(single.contiguous(), cond, pair, mask, weights)


__all__ = ["WEIGHTS", "chunk_samples", "serves", "serves_psb", "update", "update_psb"]
