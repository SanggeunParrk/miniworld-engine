"""Public/contract API guarantees.

``miniworld_engine.ops`` is the **consumer contract** (whole model-layer ops);
``miniworld_engine.kernels`` is the **internal primitive** surface out of which the
ops are built (also used by the benchmark harness). Both are pinned here so that:

  1. Each name set is stable — adding/removing one is a conscious change that must
     update the relevant frozen set (``_OPS_CONTRACT`` / ``_CONTRACT``) and CHANGELOG.
  2. ``import miniworld_engine.ops`` / ``.kernels`` stays cheap and side-effect-free:
     importing must NOT pull triton / cutlass / cuequivariance / lightning / hydra
     into ``sys.modules``. Heavy backends load lazily on first *call*, so the package
     imports on a CPU/login node without a GPU stack.

If you change a surface, update its frozen set and the CHANGELOG in the same commit.
"""

from __future__ import annotations

import contextlib
import importlib.util
import subprocess
import sys

import pytest

_HAS_TRITON = importlib.util.find_spec("triton") is not None

# Frozen public surface of miniworld_engine.kernels (== its __all__).
_CONTRACT = frozenset(
    {
        "adaln_inference",
        "adaln_train",
        "cond_transition_inference_dispatch",
        "cond_transition_train",
        "cuda_transition",
        "cuda_transition_b2b",
        "cute_transition_fused",
        "fused_gate_out",
        "sigmoid_gate_fused",
        "layernorm_kernel",
        "triton_augmented_attention_pair_bias",
        "triton_bias_only_attention",
        "triton_layernorm",
        "triton_rmsnorm",
        "triton_rmsnorm_adamod",
        "triton_rmsnorm_modulate",
        "triton_swiglu_ffn",
        "triton_tm1",
        "triton_tm2",
        "triton_transition",
        "triton_transition_fused",
        "triton_triangle_attention_pair_bias",
    }
)

# Backends that must NOT be imported just by importing the package.
_HEAVY = ("triton", "cutlass", "cuequivariance", "lightning", "hydra", "quack", "cuda.tile")


def test_public_kernel_surface_is_frozen() -> None:
    from miniworld_engine import kernels

    surface = set(kernels.__all__)
    added = surface - _CONTRACT
    removed = _CONTRACT - surface
    assert (added, removed) == (set(), set()), (
        f"kernels public surface changed (added={sorted(added)}, "
        f"removed={sorted(removed)}). This is a semver-relevant contract change: "
        f"update _CONTRACT in tests/compile/test_public_api.py and CHANGELOG.md."
    )


@pytest.mark.skipif(
    not _HAS_TRITON, reason="resolving kernel entrypoints needs the triton backend"
)
def test_every_public_name_resolves() -> None:
    """Each advertised name must actually resolve (lazily) to a callable."""
    from miniworld_engine import kernels

    for name in sorted(_CONTRACT):
        obj = getattr(kernels, name)
        assert callable(obj), f"kernels.{name} is not callable"


def test_import_is_side_effect_free() -> None:
    """Importing the kernels package must not pull any heavy backend.

    Run in a fresh subprocess so the check is not polluted by test-suite imports.
    """
    heavy = ", ".join(repr(h) for h in _HEAVY)
    code = (
        "import sys\n"
        "import miniworld_engine.kernels as k\n"
        f"_heavy = ({heavy},)\n"
        "leaked = sorted({m.split('.')[0] for m in sys.modules "
        "for h in _heavy if h in m})\n"
        "assert not leaked, f'kernels import pulled heavy backends: {leaked}'\n"
        "assert 'triton_tm1' in dir(k)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# Frozen public surface of miniworld_engine.ops — the WHOLE-OP (composite,
# weights-as-args, autograd-transparent) contract consumed by model code. Primitives
# live in `kernels` and are NOT re-exported here. Grows as ops are added.
_OPS_CONTRACT = frozenset(
    {
        "augmented_attention_pair_bias",
        "bidirectional_triangle_multiplicative_update",
        "conditioned_transition",
        "layer_norm",
        "layer_norm_linear",
        "transition",
        "triangle_attention",
        "triangle_multiplicative_update",
    }
)


def test_ops_surface_is_frozen() -> None:
    from miniworld_engine import ops

    surface = set(ops.__all__)
    added = surface - _OPS_CONTRACT
    removed = _OPS_CONTRACT - surface
    assert (added, removed) == (set(), set()), (
        f"ops public surface changed (added={sorted(added)}, removed={sorted(removed)}). "
        f"Update _OPS_CONTRACT in tests/compile/test_public_api.py and CHANGELOG.md."
    )


def test_ops_import_is_side_effect_free() -> None:
    """Importing the ops package must not pull any heavy backend (lazy on call)."""
    heavy = ", ".join(repr(h) for h in _HEAVY)
    code = (
        "import sys\n"
        "import miniworld_engine.ops as o\n"
        f"_heavy = ({heavy},)\n"
        "leaked = sorted({m.split('.')[0] for m in sys.modules "
        "for h in _heavy if h in m})\n"
        "assert not leaked, f'ops import pulled heavy backends: {leaked}'\n"
        "assert 'triangle_multiplicative_update' in dir(o)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def _kernels_deprecated() -> list[str]:
    """Names currently deprecated. Empty is legal -- the mechanism tests below do not depend on it."""
    from miniworld_engine import kernels

    return sorted(kernels._DEPRECATED)


# --- deprecation ---------------------------------------------------------------------------- #
# Freezing `__all__` makes a removal fail this file, which is the right guard -- and was the whole
# mechanism, so in practice nothing was ever removed. `kernels._DEPRECATED` is the missing half:
# a name on its way out still works and says so. These tests cover both the live entries and the
# mechanism itself, so they keep meaning something when `_DEPRECATED` is empty again.


def test_every_deprecated_name_is_still_part_of_the_contract() -> None:
    """Deprecated means "going away", not "gone". A name dropped from `__all__` the moment it was
    deprecated would break consumers with no warning period at all -- the opposite of the point."""
    from miniworld_engine import kernels

    stray = sorted(set(kernels._DEPRECATED) - set(kernels.__all__))
    assert not stray, f"{stray} are deprecated but no longer exported; deprecate then remove"


def test_every_deprecated_name_says_what_to_use_instead() -> None:
    """A bare "deprecated" makes the consumer grep this repo. The replacement is the point."""
    from miniworld_engine import kernels

    for name, why in kernels._DEPRECATED.items():
        assert len(why) > 40, f"{name}: message too short to be actionable: {why!r}"
        assert "use" in why.lower() or "instead" in why.lower(), (
            f"{name}: the message does not name a replacement: {why!r}")


@pytest.mark.parametrize("name", sorted(_kernels_deprecated()))
def test_a_deprecated_name_warns_when_it_is_used(name: str) -> None:
    """The rule: a deprecated name warns when it is USED, and use has two shapes here.

    Most of this surface resolves through `__getattr__`, so for those *resolution is the use* and
    the warning fires on attribute access. Three names are plain module-level functions
    (`cuda_transition`, `cuda_transition_b2b`, `cute_transition_fused`); `__getattr__` never runs
    for them, and access alone is not use anyway -- `hasattr`, `dir()` and a re-export would all
    warn for nothing. For those the call is the use.

    So this accepts either, and requires at least one: a name that warns on neither is not
    actually deprecated to anybody.
    """
    import warnings

    from miniworld_engine import kernels

    def kinds(fn) -> list[str]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # A deprecated name is allowed to raise -- `cuda_transition` does, by design. The
            # warning is what is under test and it is emitted before the body runs.
            with contextlib.suppress(Exception):
                fn()
        return [type(c.message).__name__ for c in caught]

    on_access = kinds(lambda: getattr(kernels, name))
    obj = getattr(kernels, name)
    on_call = kinds(obj) if callable(obj) else []
    assert "DeprecationWarning" in on_access + on_call, (
        f"{name} is in _DEPRECATED but warns on neither access ({on_access}) nor call "
        f"({on_call})")


@pytest.mark.parametrize("name", sorted(_kernels_deprecated()))
def test_a_deprecated_name_is_still_reachable(name: str) -> None:
    """Deprecated is not removed: the attribute must still exist."""
    from miniworld_engine import kernels

    assert hasattr(kernels, name)


def test_the_mechanism_works_for_a_lazy_name(monkeypatch) -> None:
    """The live entry is a plain function; most of the surface resolves through `__getattr__`.

    A synthetic entry covers that path, so this file still proves the mechanism when
    `_DEPRECATED` holds nothing (or nothing lazy).
    """
    import warnings

    from miniworld_engine import kernels

    victim = "triton_layernorm"
    monkeypatch.setitem(kernels._DEPRECATED, victim, "use something else instead")
    monkeypatch.delitem(kernels.__dict__, victim, raising=False)   # force __getattr__ to run
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = getattr(kernels, victim)
    assert resolved is not None
    assert [type(c.message).__name__ for c in caught] == ["DeprecationWarning"]
    assert victim not in kernels.__dict__, (
        "a deprecated name must not be cached into globals(), or the warning fires once per "
        "process and most callers never see it")
