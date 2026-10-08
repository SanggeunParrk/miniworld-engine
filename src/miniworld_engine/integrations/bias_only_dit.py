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
MINIWORLD_BIAS_ONLY_DIT_INF3=0 (read per call) keeps the 12-launch cuBLAS + rows step. bf16 likewise runs three kernels per block
(``kernels/bias_only_dit/cuda/inf3_bf16.py``: ``bo_front_bf16``, ``pv_gate_inf`` with PDL, ``bo_tail_bf16`` or the pair tail
``bo_tail2_bf16``) behind bf16 tables (``runner._tables3b``); MINIWORLD_BIAS_ONLY_DIT_INF3_BF16=0 (read per call) keeps the 12-launch
bf16 step, which is also what a failed build falls back to.

``serves()`` is the whole gate: no autograd, the engine's kernels (implementation MINIWORLD or TRITON), B200, bf16 or fp32, the
token widths (768; the attention as 16 heads x 48, 24 x 32, 12 x 64 or 16 x 64 / cond 384 / pair 128 / transition n = 2), B == 1, L a multiple of 128, a key mask [1, L] or
none, LayerNorm eps 1e-5. The conditioning may be one per sample or one shared by the samples (sample axis 1 or stride 0).
MINIWORLD_BIAS_ONLY_DIT=0 turns it off. Anything else keeps the module's PyTorch composition.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch

from miniworld_engine import settings
from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import opaque

D, DC, DP = 768, 384, 128
LAYOUTS = ((16, 48), (24, 32), (12, 64), (16, 64))   # (n_head, head width) served (768 or 1024 attention channels)

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


def serves(module, single, cond, pair, mask=None) -> bool:
    from miniworld_engine.modules.exceptions import ImplementationType

    if os.environ.get("MINIWORLD_BIAS_ONLY_DIT", "1") == "0" or torch.is_grad_enabled():
        return False
    if module.implementation not in (ImplementationType.MINIWORLD, ImplementationType.TRITON):
        return False
    if settings.current().engine_backend == "triton":
        return False
    dt = single.dtype
    if not (single.is_cuda and dt in (torch.bfloat16, torch.float32) and cond.dtype is dt and pair.dtype is dt):
        return False
    if dt is torch.float32 and (module.attention.to_value.weight.dtype is not torch.float32 or torch.is_autocast_enabled("cuda")):
        return False                     # fp32 = an fp32 block; under autocast the module path keeps its casts
    if single.ndim != 4 or single.shape[1] != 1 or single.shape[-1] != D or single.shape[2] % 128:
        return False
    A, _, L, _ = single.shape
    if cond.ndim != 4 or tuple(cond.shape[1:]) != (1, L, DC) or cond.shape[0] not in (1, A):
        return False
    if tuple(pair.shape) != (1, L, L, DP):
        return False
    if mask is not None and not (mask.ndim == 2 and tuple(mask.shape) == (1, L)):
        return False
    at = module.attention
    da = at.to_value.weight.shape[0]
    if (at.n_head, da // at.n_head) not in LAYOUTS or module.transition.expand_a.weight.shape[0] != 2 * D:
        return False
    norms = (at.ada_ln_in.ln_in, at.ada_ln_in.ln_cond, at.ln_pair, module.transition.ada_ln_in.ln_in,
             module.transition.ada_ln_in.ln_cond)
    if any(norm.eps != 1e-5 for norm in norms):
        return False
    from miniworld_engine.kernels.bias_only_dit.interface import core_supported
    idx = single.device.index if single.device.index is not None else torch.cuda.current_device()
    return core_supported(single.dtype, L, da, at.n_head, idx)


def _fake(single, cond, pair, mask, weights):
    return torch.empty_like(single)


@opaque(fake=_fake, name="bias_only_dit_infer")
def _infer(
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor,
    weights: list[torch.Tensor],
) -> torch.Tensor:
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

    with torch.cuda.device(single.device):
        # The weights do not change between the calls of a sampling run; packing is tens of small kernels. Reuse the pack
        # while every weight's (pointer, version) is the same -- an in-place update bumps ``_version`` and misses. Scoped
        # to the CUDA-graph capture (``kernels._capture``): a capture packs once, recorded, so a replay packs the weights as
        # they are then.
        wkey = (single.dtype, *((w.data_ptr(), w._version) for w in weights))
        runner = _capture.lookup(_RUNNERS, wkey, build)
        # The attention weights depend on the pair and the mask only: the same at every diffusion step of a sample and for
        # every augmented sample. Keyed by the caller's pair / mask tensors (pointer, version, layout).
        pkey = _capture.scoped((pair.data_ptr(), pair._version, tuple(pair.shape), pair.stride(), mask.data_ptr(), mask._version))
        hit = runner.__dict__.get("_p_cache")
        if pkey is not None and hit is not None and hit[0] == pkey:
            P = hit[1]
        else:
            P = runner.hoist(pair.contiguous(), mask)
            if pkey is not None:
                runner._p_cache = (pkey, P)
        return runner.step(single, cond, P)


_ALL_TRUE: dict = {}


def update(module, single, cond, pair, mask):
    if mask is None:                     # one all-true mask per (L, device), so the P cache key stays stable
        k = (single.shape[2], single.device)
        mask = _ALL_TRUE.get(k)
        if mask is None:
            mask = _ALL_TRUE[k] = torch.ones((1, single.shape[2]), device=single.device, dtype=torch.bool)
    return _infer(single.contiguous(), cond, pair, mask, [module.get_parameter(name) for name in WEIGHTS])


__all__ = ["WEIGHTS", "serves", "update"]
