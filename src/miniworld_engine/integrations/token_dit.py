"""Fused token DiT inference (H100, B200) wired to DiTBlock's existing parameter contract.

On B200 (sm_100) the step runs the sm_100a gated attention core (``kernels/augmented_attention/cuda/sm100``: bf16, or
TF32 tensor cores for fp32) and the CUDA row kernels; the GEMMs are cuBLAS as on H100.

Inference only: the weights are packed once and the pack reused while every weight's (pointer, version) is unchanged,
and the pair bias likewise per pair tensor, CUDA-graph replays included (a replay reads the pack and pair bias it was
captured with; single and cond stay live). The conditioning may be shared by the samples or per sample. QK-norm runs on
B200 (one in-place CUDA row pass after the projection; the logit scale folds into the q norm's weight). Calls with
autograd, different model dimensions, or QK-norm off B200 keep the general module path.

Lengths: on H100 L must be a multiple of 128 (the Triton gated core's tiles). On B200 any L >= 8: the sm_100a core takes a
multiple of 8 (its TMA maps are 3-D per sample, so tile tails load as zeros and stores clip), and ``update`` pads the other
lengths to the next multiple of 8 -- single / cond with zero rows, the pair with zero rows and columns, the key mask with
False -- and returns the first L rows.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from miniworld_engine import settings
from miniworld_engine.kernels import _capture
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
#: (heads, d_single) layouts the fused step serves: 16 x 48 everywhere; 24 x 32, 12 x 64 and 16 x 64 (d 1024) on B200 (bf16)
LAYOUTS = ((16, 768), (24, 768), (12, 768), (16, 1024))
#: the QK-norm weights, appended to WEIGHTS when the block has use_qk_norm
QK_WEIGHTS = ("attention.norm_query.weight", "attention.norm_key.weight")
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
    if a._backend != KernelBackend.TRITON:
        return False
    if not single.is_cuda or single.dtype not in (torch.bfloat16, torch.float32):
        return False
    if compute_dtype is not None and compute_dtype != single.dtype:
        return False
    if single.ndim != 4 or single.shape[1] != 1 or single.shape[2] < 8:
        return False
    d = single.shape[-1]
    if (a.n_head, d) not in LAYOUTS or (cond.shape[-1], pair.shape[-1], module.transition.expand_a.weight.shape[0]) != (384, 128, 2 * d):
        return False
    # One conditioning shared by the samples (a sampling step: L table rows) or one per sample (S L rows); the runner
    # takes either (FusedTokenDiT._cond). Anything else does not describe these samples.
    if cond.shape[0] not in (1, single.shape[0]) or cond.shape[1:3] != single.shape[1:3]:
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
    cap = torch.cuda.get_device_capability(single.device)
    if a.use_qk_norm and not (cap == (10, 0) and _cuda_rows()):
        return False                         # the QK-norm pass is a CUDA row kernel (B200)
    if (a.n_head, d) != (16, 768) and not (cap == (10, 0) and _cuda_rows()):
        return False                         # the other head layouts: B200's CUDA rows and sm_100a core only
    if cap == (10, 0):
        # any length, through the sm_100a core (a length the Triton gated core cannot tile needs it)
        if single.shape[2] % 128 == 0 and (a.n_head, d) == (16, 768):
            return True
        from miniworld_engine.kernels.augmented_attention.cuda import sm100
        idx = single.device.index if single.device.index is not None else torch.cuda.current_device()
        return sm100.inference_core_supported(single.dtype, _pad8(single.shape[2]), d, a.n_head, idx)
    return cap == (9, 0) and single.shape[2] % 128 == 0 and (a.n_head, d) == (16, 768)


def _pad8(L: int) -> int:
    return -(-L // 8) * 8


def _cuda_rows() -> bool:
    """The fused step's CUDA row kernels are on (and build): the runner takes them on B200."""
    import os

    if os.environ.get("MINIWORLD_TOKEN_DIT_ROWS_CUDA", "1") == "0":
        return False
    try:
        from miniworld_engine.kernels.conditioned_transition import cuda as cuda_rows
        return cuda_rows.available()
    except Exception:                    # a failed build keeps the module path for QK-norm blocks
        return False


def _fake(single, cond, pair, mask, weights, qk, eq, ek, heads):
    return torch.empty_like(single)


@opaque(fake=_fake, name="token_dit_h100_infer")
def _infer(
    single: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    mask: torch.Tensor,
    weights: list[torch.Tensor],
    qk: bool,
    eq: float,
    ek: float,
    heads: int,
) -> torch.Tensor:
    from miniworld_engine.kernels.conditioned_transition.triton.token_dit_runner import (
        FusedTokenDiT,
    )

    def build():
        block = SimpleNamespace()
        for path, weight in zip(WEIGHTS + (QK_WEIGHTS if qk else ()), weights, strict=True):
            node = block
            names = path.split(".")
            for name in names[:-1]:
                if not hasattr(node, name):
                    setattr(node, name, SimpleNamespace())
                node = getattr(node, name)
            setattr(node, names[-1], weight)
        block.attention.use_qk_norm = qk
        block.attention.qk_eps = (eq, ek) if qk else None
        block.attention.n_head = heads
        return FusedTokenDiT([block], dtype=single.dtype)

    with torch.cuda.device(single.device):
        # This path is inference only (the token DiT is not recycled; training takes integrations/token_dit_train), so
        # the weights do not change between the calls of a sampling run. Packing is tens of small kernels (~0.2 ms a
        # call, more than the block's step at L384): reuse the pack while every weight's (pointer, version) is the same
        # -- an in-place update bumps ``_version`` and misses. Scoped to the CUDA-graph capture (``kernels._capture``): a
        # capture packs once and records it, so a replay packs the weights as they are then (a training run that samples
        # inside a graph -- a mini-rollout -- changes them between replays), and reuses that pack for the capture's other calls.
        wkey = (single.dtype, qk, eq, ek, heads, *((w.data_ptr(), w._version) for w in weights))
        runner = _capture.lookup(_RUNNERS, wkey, build)
        key = (single.device, single.dtype)
        caches = _TUNING.setdefault(key, ({}, {}))
        runner._mm_cfg, runner._gated_cfg = caches
        # The pair is the trunk's output: the same at every diffusion step of a sample, so the block's pair bias (a
        # LayerNorm over L^2 pair rows + a projection, ~40 % of a call at L = 768) is computed once per sample. Keyed by
        # the caller's pair / mask tensors (pointer, version, layout): a new pair, or an in-place change of it, misses.
        # Scoped as the pack: a capture computes the bias once (recorded), so each replay recomputes it from the pair it has.
        bkey = _capture.scoped((pair.data_ptr(), pair._version, tuple(pair.shape), pair.stride(), mask.data_ptr(), mask._version))
        hit = runner.__dict__.get("_bias_cache")
        L, L8 = single.shape[2], _pad8(single.shape[2])
        if bkey is not None and hit is not None and hit[0] == bkey:
            bias = hit[1]
        else:
            if L8 != L:                      # padded keys: zero pair rows / columns, masked out of every softmax
                pair = torch.nn.functional.pad(pair, (0, 0, 0, L8 - L, 0, L8 - L))
                mask = torch.nn.functional.pad(mask.to(torch.bool), (0, L8 - L), value=False)
            bias = runner.hoist(pair.contiguous(), mask)
            if bkey is not None:
                runner._bias_cache = (bkey, bias)
        if L8 == L:
            return runner.step(single, cond, bias)
        # padded query rows: zero single / cond rows, computed and dropped (a shared cond stays one table)
        single = torch.nn.functional.pad(single, (0, 0, 0, L8 - L))
        shared = cond.shape[0] == 1 or cond.stride(0) == 0
        cond = torch.nn.functional.pad(cond[:1] if shared else cond, (0, 0, 0, L8 - L))
        return runner.step(single, cond, bias)[:, :, :L].contiguous()


_ALL_TRUE: dict = {}


def update(module, single, cond, pair, mask):
    if mask is None:                     # one all-true mask per (L, device), so the pair-bias cache key stays stable
        k = (single.shape[2], single.device)
        mask = _ALL_TRUE.get(k)
        if mask is None:
            mask = _ALL_TRUE[k] = torch.ones((1, single.shape[2]), device=single.device, dtype=torch.bool)
    # the pair as the caller holds it: _infer keys its pair-bias cache on this tensor and makes it contiguous on a miss
    a = module.attention
    qk = bool(a.use_qk_norm)
    names = WEIGHTS + (QK_WEIGHTS if qk else ())
    eq = float(a.norm_query.effective_eps(single.dtype)) if qk else 0.0
    ek = float(a.norm_key.effective_eps(single.dtype)) if qk else 0.0
    return _infer(single.contiguous(), cond, pair, mask, [module.get_parameter(name) for name in names], qk, eq, ek, int(a.n_head))
