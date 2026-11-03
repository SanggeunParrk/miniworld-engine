"""The engine's runtime controls for a model built from its modules: what FoldForge and MiniWorld call around the modules, so they
never import a kernel package.

* :func:`static_weights` / :func:`static_inputs`: a CUDA-graph capture (or a compiled region) whose weights / pair and conditioning
  inputs stay fixed serves the hoisted tables (the token DiT's pair bias, the atom blocks' conditioning) from their eager entries.
* :func:`clear_weight_caches`: drop the eager weight re-layouts (``w.t().contiguous()`` keyed by address and version) before weights are
  replaced; a second model built in the process could otherwise be served the first one's copy.
* :func:`set_adaln_forward_mode`: pin the AdaLN training forward ("cublas" / "fused"; "auto", the default, is the per-width measured best).
* :func:`token_dit_serves_extra_bias`: whether the token DiT's fused path takes an extra [L, L] pair bias for this activation.
* :func:`trimul_batch_limit`: how many samples one TriMul call may take on a card (the H100 module splits any batch itself).
* :func:`parallel_residual`: a parallel block's output ``x + sum_i (y_i - x)`` from the residual outputs of its parts (fp32 sums, one
  rounding; the H100 kernel where it serves, the same in torch elsewhere).
"""

from __future__ import annotations

import torch

#: The most residual outputs the H100 parallel-residual kernel sums in one launch.
_PARALLEL_MAX = 8


def static_weights(on: bool = True):
    """Context: the weights do not change while it is open (inference capture)."""
    from miniworld_engine.kernels import _capture

    return _capture.static_weights(on)


def static_inputs(on: bool = True):
    """Context: the captured callable's pair and conditioning inputs keep their contents while it is open."""
    from miniworld_engine.kernels import _capture

    return _capture.static_inputs(on)


def clear_weight_caches() -> None:
    """Drop every eager weight re-layout the modules keep (the H100 Transition, AdaLN and DiT packs share one store)."""
    from miniworld_engine.kernels import _capture

    _capture._FORMS.clear()


def set_adaln_forward_mode(name: str) -> None:
    """``"auto"`` (default), ``"cublas"`` or ``"fused"``: the AdaLN training forward backend (an A/B switch)."""
    from miniworld_engine.kernels.adaln.triton import training

    training.set_forward_mode(name)


def token_dit_serves_extra_bias(single: torch.Tensor, extra_pair_bias: torch.Tensor) -> bool:
    """Whether the token DiT's fused path folds ``extra_pair_bias`` ([L, L]) into its hoisted bias for ``single`` ([A, B, L, D])."""
    from miniworld_engine.kernels.integrations.token_dit import serves_extra_bias

    return serves_extra_bias(single, extra_pair_bias)


def parallel_residual(x: torch.Tensor, outputs: list[torch.Tensor]) -> torch.Tensor:
    """``x + sum_i (y_i - x)``: the CUDA sum (fp32, one rounding) on an H100 for a bf16 / fp32 ``x``, else the same in torch."""
    if (x.is_cuda and x.dtype in (torch.bfloat16, torch.float32) and torch.cuda.get_device_capability(x.device) == (9, 0)
            and x.numel() % 4 == 0 and len(outputs) <= _PARALLEL_MAX):
        from miniworld_engine.kernels.dit_variants import cuda

        return cuda.parallel_residual(x.contiguous(), [y.contiguous() for y in outputs])
    acc = x.to(torch.promote_types(x.dtype, torch.float32))
    total = acc + sum(y.to(acc.dtype) - acc for y in outputs)
    return total.to(x.dtype)


def trimul_batch_limit(capability: tuple[int, int]) -> int:
    """The samples a caller may fold into ONE TriMul call's batch on a card of ``capability``: any number on an H100 (the module splits
    the call into launches it takes and that fit, deterministically), ``trimul_b200.MAX_BATCH`` on a B200 (its batched path takes up to
    that many), 1 elsewhere."""
    import sys

    if capability == (9, 0):
        return sys.maxsize
    if capability == (10, 0):
        from miniworld_engine.kernels.integrations.trimul_b200 import MAX_BATCH

        return MAX_BATCH
    return 1
