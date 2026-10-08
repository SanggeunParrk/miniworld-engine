"""AttentionPairBias on B200 (integrations/attention_pair_bias_b200.py) against the fp32 PyTorch module: inference and
training, key mask on / off, eager and compiled. Errors are held to the bf16 PyTorch module's own error in the same regime."""

import copy

import pytest
import torch

from miniworld_engine.modules import AttentionPairBias
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def blackwell():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Blackwell (sm_100) required")


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


SHAPES = [(8, 384), (12, 384), (16, 384), (24, 384), (16, 512)]          # (heads, d_single)


def _modules(seed=0, shape=(8, 384)):
    torch.manual_seed(seed)
    ref = AttentionPairBias(shape[1], 128, shape[0], implementation=ImplementationType.PYTORCH)
    with torch.no_grad():                   # non-default parameters: the zero-initialised to_out, unit LayerNorms
        for p in ref.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    ours = copy.deepcopy(ref)
    ours.implementation = ImplementationType.MINIWORLD
    torch_bf = copy.deepcopy(ref)
    return ref.cuda(), ours.cuda().to(torch.bfloat16), torch_bf.cuda().to(torch.bfloat16)


def _inputs(L, masked, seed=1, d=384):
    g = torch.Generator(device="cuda").manual_seed(seed)
    single = torch.randn(1, L, d, device="cuda", generator=g)
    pair = torch.randn(1, L, L, 128, device="cuda", generator=g)
    mask = (torch.rand(1, L, device="cuda", generator=g) > 0.2) if masked else None
    if mask is not None:
        mask[0, 0] = True
    return single, pair, mask


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("L", [128, 208, 384, 768])
def test_apb_b200_inference(L, masked, shape):
    from miniworld_engine.integrations import attention_pair_bias_b200 as b200

    ref, ours, tb = _modules(shape=shape)
    single, pair, mask = _inputs(L, masked, d=shape[1])
    sb, pb = single.bfloat16(), pair.bfloat16()
    with torch.no_grad():
        assert b200.serves_inference(ours, sb, pb, mask)
        want = ref(single, pair, mask)
        got = ours(sb, pb, mask)
        base = tb(sb, pb, mask)
    assert got.dtype == torch.bfloat16
    assert got.shape == single.shape
    e, e0 = _rel(got, want), _rel(base, want)
    assert e < 1.3 * e0 + 1e-3, (e, e0)
    d, d0 = _rel(got - sb, want - single), _rel(base - sb, want - single)   # the attention branch alone
    assert d < 1.3 * d0 + 2e-3, (d, d0)


def _train(mod, single, pair, mask, dy):
    s = single.detach().clone().requires_grad_()
    p = pair.detach().clone().requires_grad_()
    y = mod(s, p, mask)
    y.backward(dy)
    return y, {"single": s.grad, "pair": p.grad, **{n: q.grad for n, q in mod.named_parameters()}}


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("L", [128, 384])
def test_apb_b200_training(L, masked, shape):
    from miniworld_engine.integrations import attention_pair_bias_b200 as b200

    ref, ours, tb = _modules(shape=shape)
    single, pair, mask = _inputs(L, masked, d=shape[1])
    dy = torch.randn_like(single)
    sb, pb = single.bfloat16(), pair.bfloat16()
    assert b200.serves_train(ours, sb, pb, mask)
    yw, gw = _train(ref, single, pair, mask, dy)
    yg, gg = _train(ours, sb, pb, mask, dy.bfloat16())
    yb, gb = _train(tb, sb, pb, mask, dy.bfloat16())
    assert _rel(yg, yw) < 1.3 * _rel(yb, yw) + 1e-3
    assert "ln_pair.bias" not in gw and "ln_pair.bias" not in gg    # no offset: it would shift every logit of a head by one constant
    worst = []
    for n in gw:
        e, e0 = _rel(gg[n], gw[n]), _rel(gb[n], gw[n])
        worst.append((e / max(e0, 1e-3), n, e, e0))
        assert gg[n].dtype == gb[n].dtype
    worst.sort(reverse=True)
    print("worst ratios:", worst[:3])
    for _r, n, e, e0 in worst:
        assert e < 1.3 * e0 + 2e-3, (n, e, e0)


@pytest.mark.parametrize("shape", SHAPES)
def test_apb_b200_compiled(shape):
    """Inference and a training step through torch.compile match eager."""
    torch._dynamo.reset()                   # each shape compiles the module's forward afresh (recompile limit)
    _, ours, _ = _modules(shape=shape)
    single, pair, mask = _inputs(384, True, d=shape[1])
    sb, pb = single.bfloat16(), pair.bfloat16()
    comp = torch.compile(ours, fullgraph=True)
    with torch.no_grad():
        assert _rel(comp(sb, pb, mask), ours(sb, pb, mask)) < 1e-6
    dy = torch.randn_like(sb)
    ye, ge = _train(ours, sb, pb, mask, dy)
    for p in ours.parameters():
        p.grad = None
    yc, gc = _train(comp, sb, pb, mask, dy)
    assert _rel(yc, ye) < 1e-6
    for n in ge:                            # the column sums (dWf, ln weights, dbq) add per-block partials atomically
        assert _rel(gc.get(n, gc.get("_orig_mod." + n)), ge[n]) < 1e-3, n
