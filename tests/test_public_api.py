"""Public API guarantees for ``miniworld_kernels.kernels``.

The **kernels** namespace is the supported surface, so this test pins two
properties:

  1. Its set of public names is stable — adding/removing one is a conscious
     change that must update ``_CONTRACT`` here (and CHANGELOG).
  2. ``import miniworld_kernels.kernels`` must stay cheap and side-effect-free:
     importing it must NOT pull triton / cutlass / cuequivariance / lightning /
     hydra into ``sys.modules``. Heavy backends load lazily on first *access*,
     so the package can be imported on a CPU/login node without a GPU stack.

If you intend to change the surface, update ``_CONTRACT`` and the CHANGELOG in
the same commit — that is the point of this test.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

_HAS_TRITON = importlib.util.find_spec("triton") is not None

# Frozen public surface of miniworld_kernels.kernels (== its __all__).
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
        "triton_adaptive_layer_norm",
        "triton_augmented_attention_pair_bias",
        "triton_bias_only_attention",
        "triton_layernorm",
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
    from miniworld_kernels import kernels

    surface = set(kernels.__all__)
    added = surface - _CONTRACT
    removed = _CONTRACT - surface
    assert not added and not removed, (
        f"kernels public surface changed (added={sorted(added)}, "
        f"removed={sorted(removed)}). This is a semver-relevant contract change: "
        f"update _CONTRACT in tests/test_public_api.py and CHANGELOG.md."
    )


@pytest.mark.skipif(
    not _HAS_TRITON, reason="resolving kernel entrypoints needs the triton backend"
)
def test_every_public_name_resolves() -> None:
    """Each advertised name must actually resolve (lazily) to a callable."""
    from miniworld_kernels import kernels

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
        "import miniworld_kernels.kernels as k\n"
        f"_heavy = ({heavy},)\n"
        "leaked = sorted({m.split('.')[0] for m in sys.modules "
        "for h in _heavy if h in m})\n"
        "assert not leaked, f'kernels import pulled heavy backends: {leaked}'\n"
        "assert 'triton_tm1' in dir(k)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# Frozen public surface of miniworld_kernels.ops — the WHOLE-OP (composite,
# weights-as-args, autograd-transparent) contract consumed by model code. Primitives
# live in `kernels` and are NOT re-exported here. Grows as ops are added.
_OPS_CONTRACT = frozenset(
    {
        "conditioned_transition",
        "transition",
        "triangle_attention",
        "triangle_multiplicative_update",
    }
)


def test_ops_surface_is_frozen() -> None:
    from miniworld_kernels import ops

    surface = set(ops.__all__)
    added = surface - _OPS_CONTRACT
    removed = _OPS_CONTRACT - surface
    assert not added and not removed, (
        f"ops public surface changed (added={sorted(added)}, removed={sorted(removed)}). "
        f"Update _OPS_CONTRACT in tests/test_public_api.py and CHANGELOG.md."
    )


def test_ops_import_is_side_effect_free() -> None:
    """Importing the ops package must not pull any heavy backend (lazy on call)."""
    heavy = ", ".join(repr(h) for h in _HEAVY)
    code = (
        "import sys\n"
        "import miniworld_kernels.ops as o\n"
        f"_heavy = ({heavy},)\n"
        "leaked = sorted({m.split('.')[0] for m in sys.modules "
        "for h in _heavy if h in m})\n"
        "assert not leaked, f'ops import pulled heavy backends: {leaked}'\n"
        "assert 'triangle_multiplicative_update' in dir(o)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
