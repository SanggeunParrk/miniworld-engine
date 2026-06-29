"""Standalone replacement for ``team_gm.typecheck``.

Vendored kernels were written against ``from team_gm import typecheck``. This
shim reproduces that decorator without depending on the team-gm package.

As in team-gm, type checking is opt-in via the ``SHOULD_TYPECHECK`` env var
(default off → ``typecheck`` is an identity decorator). When enabled, it wraps
the target with ``jaxtyped(typechecker=beartype)``; ``beartype`` / ``jaxtyping``
are imported lazily so they are only required when checking is actually on.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar, overload

_SHOULD_TYPECHECK = os.environ.get("SHOULD_TYPECHECK", "false").lower() == "true"

T = TypeVar("T")
F = TypeVar("F", bound=Callable)
C = TypeVar("C", bound=type)


@overload
def typecheck(cls_or_func: type[C]) -> type[C]: ...


@overload
def typecheck(cls_or_func: F) -> F: ...


def typecheck(cls_or_func: type[C] | F) -> type[C] | F:
    """Decorate with jaxtyped+beartype when ``SHOULD_TYPECHECK=true``, else no-op."""
    if _SHOULD_TYPECHECK:
        from beartype import beartype  # pyright: ignore[reportMissingImports]
        from jaxtyping import jaxtyped

        return jaxtyped(typechecker=beartype)(cls_or_func)
    return cls_or_func


__all__ = ["typecheck"]
