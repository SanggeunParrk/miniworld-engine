"""fp32 master parameters over bf16 activations on the A100 hand-CUDA paths (AMP's ``bf16-mixed``), the A100 counterpart of
``test_b200_fp32_master_gpu.py``: an fp32-parameter module over bf16 activations must (1) take the same CUDA path as the bf16-parameter
module, (2) hand every parameter an fp32 gradient that was never rounded to bf16, and (3) match the gradient the bf16-parameter module
gives (same bf16 kernels, the same math up to the parameters' own bf16 rounding)."""

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0), reason="A100 (sm_80) paths")]

BF, F32 = torch.bfloat16, torch.float32


def _randomize(m):
    """Nonzero, well-scaled parameters (zero-initialised projections would zero most gradients)."""
    g = torch.Generator(device="cpu").manual_seed(3)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim >= 2:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
            elif "weight" in n:
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
    return m


@pytest.fixture
def triton_launches(monkeypatch):
    """Names of the Triton kernels launched while active (every ``@triton.jit`` launch goes through ``JITFunction.run``)."""
    from triton.runtime.jit import JITFunction

    calls = []
    run = JITFunction.run

    def counting_run(self, *args, **kwargs):
        calls.append(getattr(self, "__name__", repr(self)))
        return run(self, *args, **kwargs)

    monkeypatch.setattr(JITFunction, "run", counting_run)
    return calls


def _grads(make, inputs, call, pdt, launches):
    torch.manual_seed(0)
    m = _randomize(make()).cuda().to(pdt).train()
    params = [p for p in m.parameters() if p.requires_grad]
    del launches[:]
    y = call(m, [x.clone().requires_grad_() for x in inputs])
    dy = torch.randn(y.shape, generator=torch.Generator(device="cuda").manual_seed(1), device="cuda").to(y.dtype) * 0.1
    g = torch.autograd.grad(y, params, dy, allow_unused=True)
    return [n for n, p in m.named_parameters() if p.requires_grad], g, list(launches)


def _check(make, inputs, call, launches, tol=3e-2, skip=()):
    names, g32, t32 = _grads(make, inputs, call, F32, launches)
    _, g16, t16 = _grads(make, inputs, call, BF, launches)
    assert sorted(set(t32)) == sorted(set(t16)), f"the fp32-master call takes another path than the bf16-parameter call (Triton launches {sorted(set(t32))} vs {sorted(set(t16))})"
    for n, a, b in zip(names, g32, g16, strict=True):
        if a is None:
            assert b is None, n
            continue
        assert a.dtype == F32, f"{n}: {a.dtype}"
        if a.numel() > 64 and a.any():
            assert not torch.equal(a, a.to(BF).float()), f"{n}: the fp32 gradient was rounded to bf16"
        ref = b.float()
        scale = ref.norm()
        if n.endswith(tuple(skip)):
            continue
        if scale > 0:
            assert float((a - ref).norm() / scale) < tol, f"{n}: fp32-master gradient vs bf16-parameter gradient"


def _r(*shape):
    return (torch.randn(*shape, device="cuda") * 0.5).to(BF)


def test_trimul(triton_launches):
    from miniworld_engine.modules.exceptions import ImplementationType as IT
    from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
    from miniworld_engine.modules.triangle_multiplication.bidirectional import (
        BidirectionalTriangleMultiplication,
    )

    x = _r(1, 256, 256, 128)
    _check(lambda: BidirectionalTriangleMultiplication(128, p_drop=0.0, implementation=IT.MINIWORLD), [x], lambda m, v: m(v[0]), triton_launches)
    _check(lambda: TriangleMultiplication(128, d_hidden=128, outgoing=True, p_drop=0.0, implementation=IT.MINIWORLD), [x], lambda m, v: m(v[0]), triton_launches)
    x = _r(1, 256, 256, 256)
    _check(lambda: TriangleMultiplication(256, d_hidden=256, outgoing=False, p_drop=0.0, implementation=IT.MINIWORLD), [x], lambda m, v: m(v[0]), triton_launches)


@pytest.mark.parametrize("d", [64, 128, 256])
def test_transition(d, triton_launches):
    from miniworld_engine.modules.exceptions import ImplementationType as IT
    from miniworld_engine.modules.transition import Transition

    _check(lambda: Transition(d, n=4, implementation=IT.MINIWORLD), [_r(1, 256, 256, d)], lambda m, v: m(v[0]), triton_launches)


def test_triangle_attention(triton_launches):
    from miniworld_engine.modules.exceptions import ImplementationType as IT
    from miniworld_engine.modules.triangle_attention import TriangleAttention

    _check(lambda: TriangleAttention(128, 4, d_hidden=128, starting=True, implementation=IT.MINIWORLD, p_drop=0.0), [_r(1, 256, 256, 128)], lambda m, v: m(v[0], None), triton_launches)


def test_attention_pair_bias(triton_launches):
    from miniworld_engine.modules import AttentionPairBias
    from miniworld_engine.modules.exceptions import ImplementationType as IT

    _check(lambda: AttentionPairBias(384, 128, 8, implementation=IT.MINIWORLD), [_r(1, 256, 384), _r(1, 256, 256, 128)], lambda m, v: m(v[0], v[1], None), triton_launches,
           skip=("ln_pair.bias",))


def test_outer_product_mean(triton_launches):
    from miniworld_engine.modules import OuterProductMean
    from miniworld_engine.modules.exceptions import ImplementationType as IT

    _check(lambda: OuterProductMean(64, 128, 32, implementation=IT.MINIWORLD), [_r(1, 256, 256, 64)], lambda m, v: m(v[0], None), triton_launches)


def test_pair_weighted_averaging(triton_launches):
    from miniworld_engine.modules import MSAPairWeightedAveraging
    from miniworld_engine.modules.exceptions import ImplementationType as IT

    _check(lambda: MSAPairWeightedAveraging(64, 128, n_head=8, d_hidden=32, p_drop=0.0, implementation=IT.MINIWORLD), [_r(1, 256, 256, 64), _r(1, 256, 256, 128)],
           lambda m, v: m(v[0], v[1], None), triton_launches, skip=("ln_pair.bias",))


@pytest.mark.parametrize(("d", "dc"), [(128, 128), (768, 384)])
def test_adaln_and_conditioned_transition(d, dc, triton_launches):
    from miniworld_engine.modules.adaptive_layernorm import AdaptiveLayerNorm
    from miniworld_engine.modules.conditioned_transition import ConditionedTransition
    from miniworld_engine.modules.exceptions import ImplementationType as IT

    x, c = _r(3, 1, 512, d), _r(3, 1, 512, dc)
    _check(lambda: AdaptiveLayerNorm(d, dc, implementation=IT.MINIWORLD), [x, c], lambda m, v: m(v[0], v[1]), triton_launches)
    _check(lambda: ConditionedTransition(d, dc, 2, implementation=IT.MINIWORLD), [x, c], lambda m, v: m(v[0], v[1]), triton_launches)


@pytest.mark.parametrize("shape", [(128, 128, 16, 4), (768, 384, 128, 16)])
def test_augmented_attention(shape, triton_launches):
    from miniworld_engine.modules.augmented_attention import AugmentedAttentionPairBias
    from miniworld_engine.modules.exceptions import ImplementationType as IT

    d, dc, dp, _h = shape
    L = 256
    mask = torch.ones(1, L, dtype=torch.bool, device="cuda")
    _check(lambda: AugmentedAttentionPairBias(*shape, implementation=IT.MINIWORLD), [_r(3, 1, L, d), _r(3, 1, L, dc), _r(1, L, L, dp)],
           lambda m, v: m(v[0], v[1], v[2], mask), triton_launches, skip=("ln_pair.bias",))


def test_layernorm(triton_launches):
    from miniworld_engine.modules.exceptions import ImplementationType as IT
    from miniworld_engine.modules.primitives import LayerNorm

    _check(lambda: LayerNorm(384, implementation=IT.MINIWORLD), [_r(8192, 384)], lambda m, v: m(v[0]), triton_launches)
