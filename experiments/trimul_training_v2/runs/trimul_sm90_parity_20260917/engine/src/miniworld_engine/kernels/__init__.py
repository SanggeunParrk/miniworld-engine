"""Flat re-export surface for the **primitive** fusion kernels.

This is an INTERNAL surface: the per-GEMM / LN / gate / attention fusion units out
of which the public whole-ops (:mod:`miniworld_engine.ops`) are built. It is also
used by the internal benchmark/reference harness (``miniworld_engine.modules``).
Model code should consume ``ops.*`` (whole model-layer ops), never reach in here.

Each name resolves lazily to the canonical Triton entry point for that op, without
knowing the per-op / per-backend folder layout. Import stays side-effect-free (no
triton/cutlass loaded until a name is first accessed); the name set is pinned by
``tests/compile/test_public_api.py`` for internal stability.
"""

from __future__ import annotations

import warnings
from importlib import import_module

#: Public names on their way out: ``name -> why, and what to use instead``.
#:
#: Removing a name from :data:`__all__` fails ``tests/compile/test_public_api.py`` on purpose, which is
#: the right guard and was also the whole mechanism -- so in practice nothing was ever removed.
#: This is the missing half. A name listed here still resolves and still works exactly as before;
#: it just says, once per process, that it is going away. See CONTRIBUTING.md ("Deprecation"):
#: deprecated in release N, removed no earlier than N+2, listed under `### Deprecated` in the
#: CHANGELOG for both.
#:
#: Keep the message actionable. "Deprecated" alone makes a consumer grep this repo to find out
#: what to do; the replacement is the point.
_DEPRECATED: dict[str, str] = {
    "cuda_transition": (
        "it has never had an implementation -- it deferred to transition/cuda's "
        "`cuda_transition`, which git has no record of, and calling it raises "
        "NotImplementedError. Use `implementation='triton'` on the Transition module, or "
        "`kernels.cuda_transition_b2b` for the hand-CUDA LN-fused path."
    ),
}


def _warn_deprecated(name: str) -> None:
    """Emit the deprecation for `name`, if it has one.

    `stacklevel=3` so the warning points at the CALLER's line -- through `__getattr__` or through
    the wrapper -- rather than at this file, which is not where anyone can act on it.
    """
    why = _DEPRECATED.get(name)
    if why is not None:
        warnings.warn(f"miniworld_engine.kernels.{name} is deprecated: {why}",
                      DeprecationWarning, stacklevel=3)

_LAZY_EXPORTS = {
    # Every entry names a FAMILY INTERFACE, never a backend module. That is the promise in this
    # module's docstring, and it was false for 13 of these 16 until the interfaces existed: the
    # map pointed at `.adaln.triton.main`, `.transition.triton.fused` and so on, so moving a
    # kernel between backends -- the whole point of having backends -- silently broke the public
    # surface. tests/layout/test_kernel_layout.py keeps it true.
    "adaln_inference": (".adaln.interface", "adaln_inference"),
    "adaln_train": (".adaln.interface", "adaln_train"),
    "cond_transition_inference_dispatch": (
        ".conditioned_transition.interface",
        "cond_transition_inference_dispatch",
    ),
    "cond_transition_train": (
        ".conditioned_transition.interface",
        "cond_transition_train",
    ),
    "fused_gate_out": (".bias_only_attention.interface", "fused_gate_out"),
    "layernorm_kernel": (".layernorm.interface", "layernorm_kernel"),
    "sigmoid_gate_fused": (".gated_projection.interface", "sigmoid_gate_fused"),
    "triton_rmsnorm": (".rmsnorm.interface", "triton_rmsnorm"),
    "triton_rope_3d": (".rope.interface", "triton_rope_3d"),
    "triton_swiglu_ffn": (".transition.interface", "triton_swiglu_ffn"),
    "triton_rmsnorm_adamod": (".rmsnorm.interface", "triton_rmsnorm_adamod"),
    "triton_augmented_attention_pair_bias": (
        ".augmented_attention.interface",
        "triton_augmented_attention_pair_bias",
    ),
    "triton_bias_only_attention": (
        ".bias_only_attention.interface",
        "triton_bias_only_attention",
    ),
    "triton_layernorm": (".layernorm.interface", "triton_layernorm"),
    "triton_tm1": (".tm1.interface", "triton_tm1"),
    "triton_tm2": (".tm2.interface", "triton_tm2"),
    "triton_transition": (".transition.interface", "triton_transition"),
    "triton_transition_fused": (".transition.interface", "triton_transition_fused"),
    "triton_triangle_attention_pair_bias": (
        ".triangle_attention.interface",
        "triton_triangle_attention_pair_bias",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    _warn_deprecated(name)
    value = getattr(import_module(module_name, __name__), attribute)
    # A deprecated name is deliberately NOT cached into globals(): the cache is what makes
    # `__getattr__` run once per process, and a warning that fires only on the first access in a
    # long-lived process is a warning most callers never see.
    if name not in _DEPRECATED:
        globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def cuda_transition(*args, **kwargs):  # signature kept for the frozen surface
    """NOT IMPLEMENTED. Kept as a name because the public surface is frozen.

    This wrapper deferred to ``transition.cuda.cuda_transition``, which has never existed in this
    repo -- git has no record of it, and the module binds only ``cuda_transition_b2b`` and
    ``cuda_transition_expand_gate``, both of which fuse the LayerNorm and take ``eps``. The
    ``Transition`` module's ``KernelBackend.CUDA`` branch calls this with an already-normalised
    ``x`` and an expansion factor ``n``, a signature nothing here provides.

    It raised ``ImportError`` from inside a forward. Raising here instead says what is wrong and
    what does exist; ``tests/builder/test_lazy_import_targets.py`` keeps any other lazy wrapper from
    reaching the same state.
    """
    _warn_deprecated("cuda_transition")
    msg = ("kernels.cuda_transition is not implemented: transition/cuda exposes only "
           "cuda_transition_b2b (LN-fused b2b, fixed shapes) and cuda_transition_expand_gate. "
           "Use implementation='triton' for Transition, or call one of those directly.")
    raise NotImplementedError(msg)


def cuda_transition_b2b(*args, **kwargs):
    """Hand-CUDA fused b2b Transition forward. See ``transition.interface``, where it lives."""
    from miniworld_engine.kernels.transition.interface import (
        cuda_transition_b2b as _impl,
    )

    return _impl(*args, **kwargs)


def cute_transition_fused(*args, **kwargs):
    """Cute (quack SM90 WGMMA) Transition fwd+bwd. See ``transition.interface``."""
    from miniworld_engine.kernels.transition.interface import (
        cute_transition_fused as _impl,
    )

    return _impl(*args, **kwargs)


__all__ = [
    "adaln_inference",
    "adaln_train",
    "cond_transition_inference_dispatch",
    "cond_transition_train",
    "cuda_transition",
    "cuda_transition_b2b",
    "cute_transition_fused",
    "fused_gate_out",
    "layernorm_kernel",
    "sigmoid_gate_fused",
    "triton_augmented_attention_pair_bias",
    "triton_bias_only_attention",
    "triton_layernorm",
    "triton_rmsnorm",
    "triton_rmsnorm_adamod",
    "triton_rope_3d",
    "triton_swiglu_ffn",
    "triton_tm1",
    "triton_tm2",
    "triton_transition",
    "triton_transition_fused",
    "triton_triangle_attention_pair_bias",
]
