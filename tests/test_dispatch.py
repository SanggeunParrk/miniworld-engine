"""Dispatch-layer tests: public/internal backend split, MINIWORLD auto-routing,
GPU-arch policy, and the parity of MINIWORLD vs the PYTORCH reference.

Run: ``pixi run python -m pytest tests/test_dispatch.py -q``
"""

from __future__ import annotations

import importlib

import pytest
import torch

from miniworld_engine.modules import dispatch
from miniworld_engine.modules.dispatch import KernelBackend
from miniworld_engine.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)


# --------------------------------------------------------------------------- #
# enum + public/internal mapping
# --------------------------------------------------------------------------- #
def test_kernelbackend_has_only_concrete_backends():
    assert {b.value for b in KernelBackend} == {
        "pytorch",
        "triton",
        "cuda",
        "cute",
        "cuequivariance",
    }
    # MINIWORLD is public-only; it is deliberately NOT a KernelBackend.
    assert "miniworld" not in {b.value for b in KernelBackend}


def test_to_kernel_backend_rejects_miniworld():
    assert dispatch.to_kernel_backend(ImplementationType.CUTE) is KernelBackend.CUTE
    with pytest.raises(InvalidImplementationError):
        dispatch.to_kernel_backend(ImplementationType.MINIWORLD)


# --------------------------------------------------------------------------- #
# import surface: every model-level module must construct with MINIWORLD
# (the single production option) AND with PYTORCH (the reference).
# --------------------------------------------------------------------------- #
def _module_builders():
    from miniworld_engine.modules.adaptive_layernorm.module import AdaptiveLayerNorm
    from miniworld_engine.modules.augmented_attention.module import (
        AugmentedAttentionPairBias,
    )
    from miniworld_engine.modules.conditioned_transition.module import (
        ConditionedTransition,
    )
    from miniworld_engine.modules.primitives import LayerNorm
    from miniworld_engine.modules.transition.module import Transition
    from miniworld_engine.modules.triangle_attention.bidirectional import (
        BidirectionalTriangleAttention,
    )
    from miniworld_engine.modules.triangle_attention.module import (
        TriangleAttention,
        TrianglePairAttention,
    )
    from miniworld_engine.modules.triangle_multiplication.bidirectional import (
        BidirectionalTriangleMultiplication,
    )
    from miniworld_engine.modules.triangle_multiplication.module import (
        TriangleMultiplication,
    )

    return {
        "LayerNorm": lambda impl: LayerNorm(128, implementation=impl),
        "Transition": lambda impl: Transition(128, 4, implementation=impl),
        "AdaptiveLayerNorm": lambda impl: AdaptiveLayerNorm(128, 384, implementation=impl),
        "ConditionedTransition": lambda impl: ConditionedTransition(
            128, 384, 2, implementation=impl
        ),
        "AugmentedAttentionPairBias": lambda impl: AugmentedAttentionPairBias(
            128, 384, 128, 4, implementation=impl
        ),
        "TriangleAttention": lambda impl: TriangleAttention(128, 4, implementation=impl),
        "TrianglePairAttention": lambda impl: TrianglePairAttention(
            128, 4, implementation=impl
        ),
        "BidirectionalTriangleAttention": lambda impl: BidirectionalTriangleAttention(
            128, 4, implementation=impl
        ),
        "TriangleMultiplication": lambda impl: TriangleMultiplication(
            128, implementation=impl
        ),
        "BidirectionalTriangleMultiplication": (
            lambda impl: BidirectionalTriangleMultiplication(128, implementation=impl)
        ),
    }


@pytest.mark.parametrize("name", list(_module_builders().keys()))
@pytest.mark.parametrize("impl", [ImplementationType.MINIWORLD, ImplementationType.PYTORCH])
def test_module_accepts_public_option(name, impl):
    """MINIWORLD and PYTORCH must build every module and resolve to a concrete
    KernelBackend (never MINIWORLD, never left as a raw ImplementationType)."""
    module = _module_builders()[name](impl)
    assert isinstance(module.implementation, ImplementationType)
    if hasattr(module, "_backend"):
        assert isinstance(module._backend, KernelBackend)


@pytest.mark.parametrize("name", list(_module_builders().keys()))
def test_module_accepts_string_option(name):
    """String options are coerced (Pairformer passes strings down)."""
    module = _module_builders()[name]("miniworld")
    assert module.implementation is ImplementationType.MINIWORLD


# --------------------------------------------------------------------------- #
# GPU-arch policy (mock capability so it runs on any host)
# --------------------------------------------------------------------------- #
@pytest.fixture
def arch(monkeypatch):
    def _set(major: int, minor: int = 0):
        monkeypatch.setattr(dispatch, "capability", lambda device=None: (major, minor))

    return _set


def test_trimul_arch_policy(arch, monkeypatch):
    monkeypatch.delenv("MINIWORLD_TRIMUL_IMPL", raising=False)
    monkeypatch.delenv("MINIWORLD_TRIMUL_OUT_LAYOUT", raising=False)
    mw = ImplementationType.MINIWORLD

    arch(10)  # Blackwell / B200
    assert dispatch.resolve_triangle_multiplication(mw) is KernelBackend.CUTE
    assert dispatch.trimul_out_layout() == "bdll_sm100"

    arch(9)  # Hopper / H100
    assert dispatch.resolve_triangle_multiplication(mw) is KernelBackend.CUTE
    assert dispatch.trimul_out_layout() == "bdll_direct_wide"

    arch(8)  # pre-Hopper -> no cute GEMM -> triton
    assert dispatch.resolve_triangle_multiplication(mw) is KernelBackend.TRITON
    assert dispatch.trimul_out_layout() == "bdll_direct_wide"


def test_trimul_env_override_wins(arch, monkeypatch):
    arch(10)
    monkeypatch.setenv("MINIWORLD_TRIMUL_IMPL", "triton")
    assert dispatch.resolve_triangle_multiplication(ImplementationType.MINIWORLD) is (
        KernelBackend.TRITON
    )
    monkeypatch.setenv("MINIWORLD_TRIMUL_OUT_LAYOUT", "blld")
    assert dispatch.trimul_out_layout() == "blld"


def test_arch_independent_resolvers(arch):
    """Transition / triangle-attention / adaln / cond-transition / augmented all
    resolve MINIWORLD to TRITON regardless of arch; LayerNorm to CUDA."""
    mw = ImplementationType.MINIWORLD
    for major in (8, 9, 10):
        arch(major)
        assert dispatch.resolve_transition(mw) is KernelBackend.TRITON
        assert dispatch.resolve_triangle_attention(mw) is KernelBackend.TRITON
        assert dispatch.resolve_adaptive_layernorm(mw) is KernelBackend.TRITON
        assert dispatch.resolve_conditioned_transition(mw) is KernelBackend.TRITON
        assert dispatch.resolve_augmented_attention(mw) is KernelBackend.TRITON
        assert dispatch.resolve_layernorm(mw) is KernelBackend.CUDA


# --------------------------------------------------------------------------- #
# parity: MINIWORLD forward must match the PYTORCH reference (CUDA-only)
# --------------------------------------------------------------------------- #
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="parity kernels require CUDA"
)


@requires_cuda
def test_layernorm_parity():
    from miniworld_engine.modules.primitives import LayerNorm

    torch.manual_seed(0)
    x = torch.randn(2, 384, 384, 128, device="cuda", dtype=torch.bfloat16)
    ref = LayerNorm(128, implementation=ImplementationType.PYTORCH).cuda().to(torch.bfloat16)
    mw = LayerNorm(128, implementation=ImplementationType.MINIWORLD).cuda().to(torch.bfloat16)
    mw.load_state_dict(ref.state_dict())
    with torch.no_grad():
        a = ref(x).float()
        b = mw(x).float()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    assert cos > 0.99, f"LayerNorm MINIWORLD vs PYTORCH cos={cos:.5f}"
