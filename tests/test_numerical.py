"""Numerical correctness of the miniworld kernels vs the PyTorch reference.

For each op we build the same module twice — once with the fused ``MINIWORLD``
backend, once with the ``PYTORCH`` reference — sync their weights, and assert the
fused forward (and input gradient) match the reference within a bf16-appropriate
tolerance. This promotes the ad-hoc cosine checks in the benchmark harness into
an enforced correctness suite (the guarantee team-gm relies on when it bumps the
pinned commit).

GPU-only: the fused kernels are Triton/CUDA, so the whole module skips without
CUDA. We also assert the MINIWORLD instance actually resolved to a non-PyTorch
backend, so a silent dtype-degrade can't make the comparison pass trivially.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused kernels require a CUDA GPU"
)

from miniworld_kernels.modules import (  # noqa: E402
    AdaptiveLayerNorm,
    AugmentedAttentionPairBias,
    ConditionedTransition,
    ImplementationType,
    TriangleAttention,
    TriangleMultiplication,
    Transition,
)
from miniworld_kernels.modules.dispatch import KernelBackend  # noqa: E402

DEVICE = "cuda"
BF16, FP32 = torch.bfloat16, torch.float32


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _sync_weights(mw: torch.nn.Module, ref: torch.nn.Module, seed: int) -> None:
    """Make both modules compute the same function, in a *realistic* regime.

    We keep the module's own initialization (glorot projections, LayerNorm
    weight≈1) and only lift the zero-init output/gate layers slightly off zero,
    so the output is non-degenerate without pushing softmax/normalization into a
    pathological regime that would make the bf16 backward artificially unstable.
    Then mirror the (identical) weights onto the reference.
    """
    g = torch.Generator().manual_seed(seed)
    for p in mw.parameters():
        if p.detach().abs().max().item() < 1e-6:  # zero-init layer → lift gently
            r = torch.randn(p.shape, generator=g, dtype=torch.float32) * 0.02
            p.data.copy_(r.to(dtype=p.dtype, device=p.device))
    ref.load_state_dict(mw.state_dict())


def _build_pair(op_cls, dtype, **kw):
    mw = op_cls(implementation=ImplementationType.MINIWORLD, **kw).to(DEVICE)
    ref = op_cls(implementation=ImplementationType.PYTORCH, **kw).to(DEVICE)
    if dtype is not None:
        mw, ref = mw.to(dtype), ref.to(dtype)
    _sync_weights(mw, ref, seed=0)
    return mw, ref


# (id, builder) -> returns (mw, ref, forward_inputs, grad_input_index)
def _case_transition(L=256, d=128):
    mw, ref = _build_pair(Transition, None, d_hidden=d)  # weights bf16-pinned
    x = torch.randn(1, L, L, d, device=DEVICE, dtype=BF16, requires_grad=True)
    return mw, ref, (x,), x


def _case_trimul(L=256, d=128):
    mw, ref = _build_pair(TriangleMultiplication, BF16, d_pair=d)
    pair = torch.randn(1, L, L, d, device=DEVICE, dtype=BF16, requires_grad=True)
    mask = torch.ones(1, L, dtype=torch.bool, device=DEVICE)
    return mw, ref, (pair, mask), pair


def _case_triangle_attention(L=256, d=128):
    mw, ref = _build_pair(TriangleAttention, BF16, d_pair=d, n_head=4)
    pair = torch.randn(1, L, L, d, device=DEVICE, dtype=BF16, requires_grad=True)
    mask = torch.ones(1, L, dtype=torch.bool, device=DEVICE)
    return mw, ref, (pair, mask), pair


def _case_adaln(L=256, d=128, dc=384):
    mw, ref = _build_pair(AdaptiveLayerNorm, BF16, d_hidden=d, d_cond=dc)
    x = torch.randn(1, L, d, device=DEVICE, dtype=BF16, requires_grad=True)
    cond = torch.randn(1, L, dc, device=DEVICE, dtype=BF16)
    return mw, ref, (x, cond), x


def _case_cond_transition(M=4096, d=128, dc=384):
    mw, ref = _build_pair(ConditionedTransition, None, d_hidden=d, d_cond=dc)  # fp32-pinned
    x = torch.randn(M, d, device=DEVICE, dtype=FP32, requires_grad=True)
    cond = torch.randn(M, dc, device=DEVICE, dtype=FP32)
    return mw, ref, (x, cond), x


def _case_augmented_attention(L=256, d_single=256, d_pair=128, n_head=16, n_aug=2):
    mw, ref = _build_pair(
        AugmentedAttentionPairBias, BF16,
        d_single=d_single, d_cond=d_single, d_pair=d_pair, n_head=n_head,
    )
    single = torch.randn(n_aug, 1, L, d_single, device=DEVICE, dtype=BF16, requires_grad=True)
    cond = torch.randn(n_aug, 1, L, d_single, device=DEVICE, dtype=BF16)
    pair = torch.randn(1, L, L, d_pair, device=DEVICE, dtype=BF16)
    mask = torch.ones(1, L, dtype=torch.bool, device=DEVICE)
    return mw, ref, (single, cond, pair, mask), single


_CASES = {
    "transition": _case_transition,
    "trimul": _case_trimul,
    "triangle_attention": _case_triangle_attention,
    "adaptive_layernorm": _case_adaln,
    "conditioned_transition": _case_cond_transition,
    "augmented_attention": _case_augmented_attention,
}

# bf16 kernels: cosine ~0.99; fp32 (conditioned_transition): tighter.
_FWD_TOL = {"conditioned_transition": 0.999}
_GRAD_TOL = {"conditioned_transition": 0.999}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_forward_matches_reference(name):
    mw, ref, inputs, _ = _CASES[name]()
    assert mw._backend != KernelBackend.PYTORCH, (
        f"{name}: MINIWORLD degraded to PYTORCH — test would be vacuous"
    )
    y_mw = mw(*inputs)
    y_ref = ref(*inputs)
    c = _cos(y_mw, y_ref)
    assert c >= _FWD_TOL.get(name, 0.99), f"{name} forward cosine {c:.5f} too low"


@pytest.mark.parametrize("name", sorted(_CASES))
def test_input_grad_matches_reference(name):
    mw, ref, inputs, gin = _CASES[name]()
    # separate input tensors per module so grads don't alias
    ref_inputs = tuple(
        t.detach().clone().requires_grad_(t.requires_grad) if torch.is_tensor(t) else t
        for t in inputs
    )
    ref_gin = ref_inputs[list(inputs).index(gin)]
    mw(*inputs).float().sum().backward()
    ref(*ref_inputs).float().sum().backward()
    c = _cos(gin.grad, ref_gin.grad)
    assert c >= _GRAD_TOL.get(name, 0.99), f"{name} input-grad cosine {c:.5f} too low"
