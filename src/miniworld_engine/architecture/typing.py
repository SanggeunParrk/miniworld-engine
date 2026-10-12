"""``typecheck``: jaxtyping + beartype when ``SHOULD_TYPECHECK=true``, otherwise a no-op."""

import os
from collections.abc import Callable
from typing import TypeVar, overload

_SHOULD_TYPECHECK = os.environ.get("SHOULD_TYPECHECK", "false").lower() == "true"

F = TypeVar("F", bound=Callable)
C = TypeVar("C", bound=type)


@overload
def typecheck(cls_or_func: type[C]) -> type[C]: ...


@overload
def typecheck(cls_or_func: F) -> F: ...


def typecheck(cls_or_func):
    """Decorate the class or function with jaxtyped and beartype."""
    if _SHOULD_TYPECHECK:
        from beartype import beartype
        from jaxtyping import jaxtyped

        return jaxtyped(typechecker=beartype)(cls_or_func)
    return cls_or_func


__all__ = ["typecheck"]
