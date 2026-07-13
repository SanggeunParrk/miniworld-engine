"""Whole-op public surface — complete, autograd-transparent model ops.

Each name here is a **composite** operation (e.g. the full triangle multiplicative
update: ``LN → gated in-proj → contraction → LN → out-proj+gate``) wrapped as a
single autograd-transparent call with **weights as arguments** and backend dispatch
inside. This is the supported contract for model code: a layer holds the weights as
``nn.Parameter`` and calls one op — it never composes primitives itself.

Contrast with :mod:`miniworld_kernels.kernels`, which holds the **primitive** fusion
units (per-GEMM / LN / gate kernels). Those are an implementation detail of these ops
and are not part of the consumed contract.

Import stays cheap and side-effect-free: heavy backends (triton / cutlass) load lazily
on first *call*, not on import (see ``tests/test_public_api.py``).
"""

from __future__ import annotations

from importlib import import_module

# op name -> (absolute module, attribute). The implementation lives next to its
# primitives under kernels/<op>/; this namespace is only the public façade.
_LAZY_OPS = {
    "triangle_multiplicative_update": (
        "miniworld_kernels.kernels.trimul_inproj.whole_op",
        "triangle_multiplicative_update",
    ),
    "triangle_attention": (
        "miniworld_kernels.kernels.triangle_attention.whole_op",
        "triangle_attention",
    ),
    "transition": (
        "miniworld_kernels.kernels.transition.whole_op",
        "transition",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_OPS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_OPS))


__all__ = ["transition", "triangle_attention", "triangle_multiplicative_update"]
