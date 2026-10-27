"""The portable ConditionedTransition / AdaptiveLayerNorm paths that carry a residual add and cached weight casts.

* ``ConditionedTransition.forward`` folds the stream add (``x + delta``) into the gate pass of its kernels (inference: the
  squeeze + gate epilogue; training: the gate pass of the forward, the residual's gradient is the incoming one). It must be the
  very tensor ``x + delta(x, cond)`` returns, forward and backward, on both training forward shapes (the fused token forward at
  M <= 512, the cuBLAS one above).
* Inference casts the fp32 master weights once while they are unchanged (``kernels._capture.param_forms``): an in-place weight
  update must be seen by the next call.
"""
from __future__ import annotations

import pytest
import torch
from portable_helpers import portable_on, rel

from miniworld_engine.modules.adaptive_layernorm import AdaptiveLayerNorm
from miniworld_engine.modules.conditioned_transition import ConditionedTransition
from miniworld_engine.modules.exceptions import ImplementationType as I

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]
DEV, BF, F32 = "cuda", torch.bfloat16, torch.float32


@pytest.fixture(autouse=True)
def portable():
    with portable_on():
        yield


def _ct(impl, dtype, seed=0, d=768, dc=384):
    torch.manual_seed(seed)
    m = ConditionedTransition(d_hidden=d, d_cond=dc, n=2, implementation=impl, dtype=dtype).to(DEV)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 0.1 * torch.randn_like(p))
    return m


def _xc(A, L, dtype, d=768, dc=384, seed=1, grad=False):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(A, 1, L, d, device=DEV, dtype=dtype, generator=g).requires_grad_(grad)
    c = torch.randn(A, 1, L, dc, device=DEV, dtype=dtype, generator=g).requires_grad_(grad)
    return x, c


@pytest.mark.parametrize("dtype", [BF, F32], ids=["bf16", "fp32"])
def test_inference_residual_fold_is_the_unfused_add(dtype):
    m = _ct(I.MINIWORLD, dtype)
    assert m._backend.value == "portable"
    x, c = _xc(5, 96, dtype)
    with torch.no_grad():
        fused = m(x, c)
        unfused = x + m.delta(x, c)
    if dtype is BF:
        assert torch.equal(fused, unfused)
    else:   # fp32: the epilogue contracts sigmoid * out + residual into one FMA
        assert rel(fused, unfused) <= 1e-6
    assert rel(fused, x) > 1e-3   # the update is not nothing


@pytest.mark.parametrize("dtype", [BF, F32], ids=["bf16", "fp32"])
@pytest.mark.parametrize(("A", "L"), [(4, 64), (8, 128)], ids=["fused_fwd_M256", "cublas_fwd_M1024"])
def test_training_residual_fold_matches_the_unfused_add(dtype, A, L):
    m = _ct(I.MINIWORLD, dtype)
    dy = torch.randn(A, 1, L, 768, device=DEV, dtype=dtype)
    results = []
    for fold in (True, False):
        x, c = _xc(A, L, dtype, grad=True)
        m.zero_grad(set_to_none=True)
        y = m(x, c) if fold else x + m.delta(x, c)
        y.backward(dy)
        results.append((y.detach(), x.grad, c.grad, *[p.grad.clone() for p in m.parameters()]))
    # the gradient into the conditioning: the fold joins the two paths' gradients in the cond LayerNorm backward kernel (one rounding), the
    # unfused add rounds each path to bf16 first; the bf16 cases are therefore held against the fp32 statements, the fold no further off than the add
    gold = None
    if dtype is BF:
        ref = _ct(I.PYTORCH, F32)
        ref.load_state_dict({k: v.float() for k, v in m.state_dict().items()})
        xg, cg = (t.detach().float().requires_grad_(True) for t in _xc(A, L, dtype))
        ref(xg, cg).backward(dy.float())
        gold = cg.grad
    for i, (a, b) in enumerate(zip(*results, strict=True)):
        assert a.dtype == b.dtype
        if i == 2 and gold is not None:
            assert rel(a, gold) <= max(rel(b, gold), 2e-3), (i, rel(a, gold), rel(b, gold))
            continue
        assert rel(a, b) <= ((0 if dtype is BF else 1e-6) if i == 0 else 2e-3), (i, rel(a, b))
    # the residual's own gradient is dy itself, so x.grad = dy + the update's path
    assert rel(results[0][1], results[1][1]) <= 2e-3


@pytest.mark.parametrize("make", ["adaln", "ct"])
def test_cached_inference_casts_follow_an_in_place_weight_update(make):
    """fp32 master weights under bf16 activations: the bf16 forms are cached per (pointer, version); ``mul_`` bumps the version."""
    if make == "adaln":
        torch.manual_seed(0)
        m = AdaptiveLayerNorm(768, 384, implementation=I.MINIWORLD).to(DEV)
        with torch.no_grad():
            m.to_scale.weight.normal_(0, 0.05)
            m.to_bias.weight.normal_(0, 0.05)
    else:
        m = _ct(I.MINIWORLD, F32)
    x, c = _xc(3, 64, BF)
    with torch.no_grad(), torch.autocast("cuda", dtype=BF):
        before = m(x, c).clone()
        for p in m.parameters():
            p.mul_(2.0)
        after = m(x, c)
    with torch.no_grad():
        # the reference: the same module through the PyTorch statements
        ref = type(m)(768, 384, implementation=I.PYTORCH).to(DEV) if make == "adaln" else _ct(I.PYTORCH, F32)
        ref.load_state_dict(m.state_dict())
        with torch.autocast("cuda", dtype=BF):
            want = ref(x, c)
    assert not torch.equal(before, after), "the weight update was not seen (a stale cached cast)"
    assert rel(after, want) <= 2.5e-2, rel(after, want)


def test_bias_only_attention_weights_are_kept_while_the_pair_is_unchanged():
    """The softmax of the pair bias is kept per (pair, mask, weights) in inference and follows an in-place update of the pair or of a weight."""
    from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock

    torch.manual_seed(0)
    blk = BiasOnlyDiTBlock(256, 128, 64, 8, implementation=I.MINIWORLD).to(DEV)
    with torch.no_grad():
        for p in blk.parameters():
            p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 0.1 * torch.randn_like(p))
    ref = BiasOnlyDiTBlock(256, 128, 64, 8, implementation=I.PYTORCH).to(DEV)
    ref.load_state_dict(blk.state_dict())
    g = torch.Generator(device=DEV).manual_seed(1)
    single = torch.randn(2, 1, 96, 256, device=DEV, generator=g)
    cond = torch.randn(2, 1, 96, 128, device=DEV, generator=g)
    pair = torch.randn(1, 96, 96, 64, device=DEV, generator=g)
    mask = torch.rand(1, 96, device=DEV, generator=g) > 0.1
    with torch.no_grad():
        first, again = blk(single, cond, pair, mask), blk(single, cond, pair, mask)
        assert torch.equal(first, again)
        assert rel(first, ref(single, cond, pair, mask)) <= 6e-3
        pair.mul_(1.7)
        after_pair = blk(single, cond, pair, mask)
        assert not torch.equal(first, after_pair), "an in-place update of the pair was not seen (a stale attention)"
        assert rel(after_pair, ref(single, cond, pair, mask)) <= 6e-3
        blk.attention.to_bias.weight.mul_(1.5)
        ref.attention.to_bias.weight.mul_(1.5)
        assert rel(blk(single, cond, pair, mask), ref(single, cond, pair, mask)) <= 6e-3

