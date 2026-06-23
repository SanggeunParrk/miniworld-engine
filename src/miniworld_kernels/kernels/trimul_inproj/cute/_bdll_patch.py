"""In-repo ownership of quack's gated-postact layout policy (the "bdll patch").

We do NOT edit quack's files, and we do NOT vendor the GEMM body (~5400 lines of
gemm_sm90 + gemm_act + epi_ops + utils + scheduler/pipeline). We only OWN the two
tiny policy points that make stock quack reject an M-major (bdll) gated postact:

  1. ``GemmGatedMixin.epi_to_underlying_arguments`` — drop the n-major-c asserts.
  2. ``_compile_gemm_act`` — ``pa_leading_dim`` follows the DETECTED postact
     layout instead of being forced to n-major (``1``) for the gated path.

Both are applied at import time by re-deriving the patched function/method from
quack's OWN source (via ``inspect.getsource`` + a single string replacement), so
we never hand-transcribe quack internals — if quack changes those lines the
``assert ... in src`` guards fire loudly instead of silently misbehaving.

This lives in our repo, reuses quack's (tested) GEMM body unchanged, and survives
``pixi install`` (nothing on quack's disk is modified). Call :func:`apply` before
issuing a gated GEMM with an M-major postact. Idempotent.
"""

from __future__ import annotations

import inspect
import textwrap

import quack.gemm_act as _ga

_APPLIED = False


def _redefine(src: str, name: str, tag: str):
    """exec ``src`` with quack.gemm_act's globals; return the defined object."""
    # Strip any decorator lines so we exec the bare def, then re-decorate by hand.
    src = "\n".join(
        line for line in textwrap.dedent(src).splitlines()
        if not line.strip().startswith("@")
    )
    ns: dict = {}
    exec(compile(src, f"<bdll_patch:{tag}>", "exec"), _ga.__dict__, ns)
    return ns[name]


def _patch_pa_leading_dim() -> None:
    """``pa_leading_dim`` follows the detected layout (was forced to 1 if gated)."""
    orig = _ga._compile_gemm_act.__wrapped__  # unwrap functools.wraps(jit_cache)
    src = inspect.getsource(orig)
    old = 'pa_leading_dim = 1 if gemm_cls_name == "gated" else pa_leading'
    new = "pa_leading_dim = pa_leading  # bdll patch: follow detected postact layout"
    assert old in src, "quack _compile_gemm_act changed; review the bdll patch"
    fn = _redefine(src.replace(old, new), "_compile_gemm_act", "compile")
    # Distinct qualname so jit_cache's disk key (qualname, *args) does NOT collide
    # with quack's stock _compile_gemm_act — otherwise a stale n-major-compiled
    # kernel from a pre-patch run is reused and our pa_leading_dim change never
    # recompiles.
    fn.__qualname__ = "_compile_gemm_act_bdll"
    fn.__name__ = "_compile_gemm_act_bdll"
    _ga._compile_gemm_act = _ga.jit_cache(fn)


def _patch_postact_assert() -> None:
    """Drop the n-major-c asserts on the gated postact (allow M-major bdll)."""
    src = inspect.getsource(_ga.GemmGatedMixin.epi_to_underlying_arguments)
    drops = (
        "assert self.d_layout is None or self.d_layout.is_n_major_c()",
        "assert cutlass.utils.LayoutEnum.from_tensor(args.mPostAct).is_n_major_c()",
    )
    for d in drops:
        assert d in src, f"quack gated postact assert changed; review: {d!r}"
        src = src.replace(d, "pass  # bdll patch: n-major postact assert dropped")
    method = _redefine(src, "epi_to_underlying_arguments", "epi")
    # Shadow the mixin method on the concrete SM90 class used by _compile_gemm_act.
    _ga.GemmGatedSm90.epi_to_underlying_arguments = method


def ensure_sigmoid_act() -> None:
    """Register ``"sigmoid"`` in quack's non-gated ``act_fn_map`` (idempotent).

    Stock quack has silu/relu/relu_sq/gelu but no plain sigmoid, so
    ``gemm_act(activation="sigmoid")`` (a fused gemm + sigmoid in ONE launch) is
    unavailable. We add it — `quack.activation.sigmoid` is already a unary
    tanh-based fn that the SM90 act epilogue calls as ``act_fn(x)``. Lets the
    trimul gate be a fused kernel instead of a torch matmul + separate sigmoid.
    """
    from quack.activation import act_fn_map, sigmoid

    if "sigmoid" not in act_fn_map:
        act_fn_map["sigmoid"] = sigmoid


def apply() -> None:
    """Apply the bdll patch to quack (idempotent)."""
    global _APPLIED
    if _APPLIED:
        return
    _patch_pa_leading_dim()
    _patch_postact_assert()
    ensure_sigmoid_act()  # register "sigmoid" up front (dict mutation must happen
    #                       OUTSIDE any torch.compile region, else it's not traced)
    _APPLIED = True
