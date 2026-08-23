"""Compatibility shim for quack 0.5.0 (as forced by FlashAttention-4) on py<3.11.

The cute kernels were written against quack 0.3.11. The consumer env now ships
quack **0.5.0** because ``flash-attn-4`` requires ``quack-kernels>=0.5.0``. Two
0.3.11 -> 0.5.0 breaks are absorbed here so the cute kernels can import their quack
dependencies through one place instead of ``quack.gemm_interface`` / ``quack.cache_utils``
directly:

1. **``quack.gemm_interface`` fails to import on torch 2.10 + py<3.11.** Its GEMM
   ``@torch.library.custom_op`` schemas declare ``rounding_mode: int = RoundingMode.RN``
   (an :class:`~enum.IntEnum`). torch's ``infer_schema`` treats an ``IntEnum`` as an
   ``int`` and serializes the default with ``str()``; on **py<3.11**
   ``str(IntEnum.member)`` is the member *name* (``"RoundingMode.RN"``), not its value
   (``"0"``), so the inferred schema ``SymInt rounding_mode=RoundingMode.RN`` fails
   ``parse_schema``. (py3.11+ ``IntEnum.__str__`` already returns the value, which is why
   quack ships fine there.) We restore that behaviour by forcing
   ``RoundingMode.__str__`` to the int value **before** importing ``gemm_interface``.
   This is cosmetic-only — nothing depends on ``str(RoundingMode)`` for correctness, and
   the functions the cute kernels use (``gemm`` / ``gemm_act`` / ``gemm_act_tuned`` /
   ``default_config``) do not touch the broken ``gemm_out`` custom op at all.

2. **``quack.cache_utils`` was removed in 0.5.0.** ``jit_cache`` moved to
   :mod:`quack.cache`; the module-level ``COMPILE_ONLY`` flag became the
   :func:`quack.cache.is_compile_only` state accessor (so ``if COMPILE_ONLY:`` becomes
   ``if is_compile_only():``).

No ``site-packages`` are patched: this only re-exports quack symbols and adjusts one
``__str__`` at runtime.
"""

from __future__ import annotations

# This shim supports BOTH quack 0.3.11 (the cute kernels' original pin) and quack 0.5.0
# (forced by FA4). Everything below is version-tolerant so a 0.3.11 env is not regressed.

# --- fix 1: make gemm_interface's IntEnum-default custom_op schemas parse on py<3.11.
# (Only quack >= 0.5.0 declares those IntEnum-default custom ops; on 0.3.11 quack.rounding
# may not exist, so this is best-effort.)
try:
    from quack.rounding import RoundingMode as _RoundingMode

    if str(_RoundingMode.RN) != str(int(_RoundingMode.RN)):  # py<3.11 only; no-op on py3.11+
        _RoundingMode.__str__ = lambda self: str(int(self))  # type: ignore[assignment,method-assign]
except Exception:  # noqa: BLE001 - older quack without quack.rounding / no schema issue
    pass

# --- fix 2: jit_cache / compile-only flag location (moved between 0.3.11 and 0.5.0).
try:
    # quack >= 0.5.0
    from quack.cache import is_compile_only, jit_cache  # noqa: E402
except ImportError:
    # quack 0.3.11: jit_cache + module-level COMPILE_ONLY flag in quack.cache_utils.
    from quack.cache_utils import jit_cache  # noqa: E402,F401

    def is_compile_only() -> bool:  # noqa: D401 - read the flag live, not snapshot at import
        import quack.cache_utils as _cu

        return bool(getattr(_cu, "COMPILE_ONLY", False))

# --- fix 3 (CRITICAL): register OUR cute-kernel source in quack's jit disk-cache key.
# quack.cache.jit_cache keys compiled .o by (qualname, *args) + a hash of QUACK's source dir
# (+ EXTRA_SOURCE_DIRS) — but NOT our kernels' source. So editing one of our .cute kernels does
# NOT invalidate its cached .o: a stale binary from a previous (possibly broken) version is
# silently reused. This masked the gated-postact fix (mPostAct->mAuxOut) for an entire debug
# session — the edit compiled correctly but the node-local /tmp cache kept serving the broken .o.
# Adding our package root to EXTRA_SOURCE_DIRS makes any change under it bust the cache key, so
# edits recompile and old .o become orphaned (never hit). Idempotent; 0.5.0-only (0.3.11 lacks it).
try:
    import os as _os

    import quack.cache as _qcache  # noqa: E402

    _pkg_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # miniworld_engine/
    if hasattr(_qcache, "EXTRA_SOURCE_DIRS") and _pkg_root not in _qcache.EXTRA_SOURCE_DIRS:
        _qcache.EXTRA_SOURCE_DIRS.append(_pkg_root)
except Exception:  # noqa: BLE001 - never let cache-key hardening break import
    pass

# gemm_interface is imported lazily (below) so merely needing jit_cache / is_compile_only
# does not eagerly register every quack GEMM custom op. The RoundingMode fix above is
# already applied, so the deferred import succeeds.
_GEMM_INTERFACE_SYMBOLS = frozenset(
    {"gemm", "gemm_act", "gemm_act_tuned", "default_config"}
)

# The four GEMM names are supplied lazily by the PEP 562 ``__getattr__`` below, so they are
# genuinely exported but have no module-level binding for ruff to see -- hence F822 on each.
# Suppressed here rather than in per-file-ignores so the reason travels with the code, and
# narrowly so a name that is REALLY missing still fails.
__all__ = [
    "default_config",  # noqa: F822 -- lazy, see __getattr__
    "gemm",  # noqa: F822 -- lazy, see __getattr__
    "gemm_act",  # noqa: F822 -- lazy, see __getattr__
    "gemm_act_tuned",  # noqa: F822 -- lazy, see __getattr__
    "is_compile_only",
    "jit_cache",
]


def __getattr__(name: str):  # noqa: ANN202  (PEP 562: consulted by `from ... import name` too)
    if name in _GEMM_INTERFACE_SYMBOLS:
        import quack.gemm_interface as _gi  # RoundingMode fix already applied above

        return getattr(_gi, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
