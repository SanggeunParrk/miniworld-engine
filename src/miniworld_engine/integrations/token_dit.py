"""Fused token DiT inference (H100, B200) wired to DiTBlock's existing parameter contract.

On B200 (sm_100) the bf16 step runs the sm_100a gated attention core (``kernels/augmented_attention/cuda/sm100``);
every other kernel is the same as on H100.

Inference only: the weights are packed once and the pack reused while every weight's (pointer, version) is unchanged,
including by CUDA-graph replays (a replay does not re-pack; the pair and the inputs stay live). Calls with per-sample
conditioning, QK-norm, autograd or different model dimensions keep the general module path.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from miniworld_engine import settings
from miniworld_engine.kernels._compile import opaque

WEIGHTS = (
    "attention.to_query.weight",
    "attention.to_query.bias",
    "attention.to_key.weight",
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
_TUNING: dict = {}
#: Packed runners keyed by the weights' (pointer, version): see ``_infer``.
_RUNNERS: dict = {}


def serves(module, single, cond, pair, compute_dtype=None):
    if torch.is_grad_enabled() or settings.current().engine_backend == "triton":
        return False
    a = module.attention
    # the engine's kernels (implementation TRITON or MINIWORLD resolve to them), as the attention cores and the training
    # path gate on; PYTORCH keeps the reference
    from miniworld_engine.modules.dispatch import KernelBackend
    if a._backend != KernelBackend.TRITON or a.use_qk_norm:
        return False
    if not single.is_cuda or single.dtype not in (torch.bfloat16, torch.float32):
        return False
    if compute_dtype is not None and compute_dtype != single.dtype:
        return False
    if (
        single.ndim != 4
        or single.shape[1] != 1
        or single.shape[-1] != 768
        or single.shape[2] % 128
    ):
        return False
    if (
        a.n_head,
        cond.shape[-1],
        pair.shape[-1],
        module.transition.expand_a.weight.shape[0],
    ) != (16, 384, 128, 1536):
        return False
    # The stack algorithm shares conditioning across samples; never silently take cond[0] otherwise.
    if cond.shape[0] != 1 and cond.stride(0) != 0:
        return False
    norms = (
        a.ada_ln_in.ln_in,
        a.ada_ln_in.ln_cond,
        a.ln_pair,
        module.transition.ada_ln_in.ln_in,
        module.transition.ada_ln_in.ln_cond,
    )
    if any(norm.eps != 1e-5 for norm in norms):
        return False
    return torch.cuda.get_device_capability(single.device) in ((9, 0), (10, 0))


def _fake(single, cond, pair, mask, weights):
    return torch.empty_like(single)


@opaque(fake=_fake, name="token_dit_h100_infer")
def _infer(
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor,
    weights: list[torch.Tensor],
) -> torch.Tensor:
    from miniworld_engine.kernels.conditioned_transition.triton.token_dit_runner import (
        FusedTokenDiT,
    )

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
        block.attention.use_qk_norm = False
        block.attention.n_head = 16
        return FusedTokenDiT([block], dtype=single.dtype)

    with torch.cuda.device(single.device):
        # This path is inference only (the token DiT is not recycled; training takes integrations/token_dit_train), so
        # the weights do not change between the calls of a sampling run. Packing is tens of small kernels (~0.2 ms a
        # call, more than the block's step at L384): reuse the pack while every weight's (pointer, version) is the same
        # -- an in-place update bumps ``_version`` and misses -- also inside a CUDA-graph capture, whose replays then
        # read the pack made before it. A pack built during a capture lives in the graph's pool and is not cached.
        wkey = (single.dtype, *((w.data_ptr(), w._version) for w in weights))
        runner = _RUNNERS.get(wkey)
        if runner is None:
            runner = build()
            if not torch.cuda.is_current_stream_capturing():
                if len(_RUNNERS) >= 64:       # a model's worth of blocks; drop the oldest
                    _RUNNERS.pop(next(iter(_RUNNERS)))
                _RUNNERS[wkey] = runner
        key = (single.device, single.dtype)
        caches = _TUNING.setdefault(key, ({}, {}))
        runner._mm_cfg, runner._gated_cfg = caches
        # The pair is the trunk's output: the same at every diffusion step of a sample, so the block's pair bias (a
        # LayerNorm over L^2 pair rows + a projection, ~40 % of a call at L = 768) is computed once per sample. Keyed by
        # the caller's pair / mask tensors (pointer, version, layout): a new pair, or an in-place change of it, misses.
        # As with the pack, a bias computed during a capture is not cached, and a replay reuses the one it captured.
        bkey = (pair.data_ptr(), pair._version, tuple(pair.shape), pair.stride(), mask.data_ptr(), mask._version)
        hit = runner.__dict__.get("_bias_cache")
        if hit is not None and hit[0] == bkey:
            bias = hit[1]
        else:
            bias = runner.hoist(pair.contiguous(), mask)
            if not torch.cuda.is_current_stream_capturing():
                runner._bias_cache = (bkey, bias)
        return runner.step(single, cond, bias)


_ALL_TRUE: dict = {}


def update(module, single, cond, pair, mask):
    if mask is None:                     # one all-true mask per (L, device), so the pair-bias cache key stays stable
        k = (single.shape[2], single.device)
        mask = _ALL_TRUE.get(k)
        if mask is None:
            mask = _ALL_TRUE[k] = torch.ones((1, single.shape[2]), device=single.device, dtype=torch.bool)
    # the pair as the caller holds it: _infer keys its pair-bias cache on this tensor and makes it contiguous on a miss
    return _infer(single.contiguous(), cond, pair, mask, [module.get_parameter(name) for name in WEIGHTS])
