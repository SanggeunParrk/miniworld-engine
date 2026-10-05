"""AttentionPairBias on A100 (integrations/attention_pair_bias_sm80.py) against the fp32 PyTorch module: key mask on / off, any L, eager and compiled.
Errors are held to the bf16 PyTorch module's own error in the same regime."""

import copy

import pytest
import torch

from miniworld_engine.modules import AttentionPairBias
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def ampere():
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Ampere (sm_80) required")


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


SHAPES = [(8, 384), (12, 384), (16, 384), (16, 512)]          # (heads, d_single): 8 x 48, 12 x 32, 16 x 24 (padded to 32), 16 x 32


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
@pytest.mark.parametrize("L", [50, 128, 208, 384, 768])
def test_apb_a100_inference(L, masked, shape):
    from miniworld_engine.integrations import attention_pair_bias_sm80 as a100

    ref, ours, tb = _modules(shape=shape)
    single, pair, mask = _inputs(L, masked, d=shape[1])
    sb, pb = single.bfloat16(), pair.bfloat16()
    with torch.no_grad():
        assert a100.serves_inference(ours, sb, pb, mask)
        want = ref(single, pair, mask)
        got = ours(sb, pb, mask)
        base = tb(sb, pb, mask)
    assert got.dtype == torch.bfloat16
    assert got.shape == single.shape
    e, e0 = _rel(got, want), _rel(base, want)
    assert e < 1.3 * e0 + 1e-3, (e, e0)
    d, d0 = _rel(got - sb, want - single), _rel(base - sb, want - single)   # the attention branch alone
    assert d < 1.3 * d0 + 2e-3, (d, d0)


def test_apb_a100_everything_masked_is_the_uniform_softmax():
    """A sample whose keys are all masked: the kernel's finite fill makes every logit equal (no NaN), so the softmax is uniform and the attention output the
    mean of v over the keys. (The PyTorch module's finfo.min fill is not the uniform softmax here -- its SDPA kernels treat the degenerate row their own way --
    so the reference is the limit itself.)"""
    ref, ours, _ = _modules()
    single, pair, _ = _inputs(256, False)
    mask = torch.zeros(1, 256, dtype=torch.bool, device="cuda")
    sb, pb = single.bfloat16(), pair.bfloat16()
    with torch.no_grad():
        got = ours(sb, pb, mask)
        x = ref.ln_single(single)
        o = ref.to_value(x).mean(1, keepdim=True).expand(-1, 256, -1)
        want = single + ref.to_out(torch.sigmoid(ref.to_gate(x)) * o)
    assert torch.isfinite(got).all()
    assert _rel(got, want) < 1e-2


def test_apb_a100_gate_conditions():
    """The path declines what it does not implement: fp32 operands, a grad-enabled call, B > 1, a per-sample mask, QK-norm."""
    from miniworld_engine.integrations import attention_pair_bias_sm80 as a100

    _, ours, _ = _modules()
    single, pair, mask = _inputs(128, True)
    sb, pb = single.bfloat16(), pair.bfloat16()
    with torch.no_grad():
        assert a100.serves_inference(ours, sb, pb, mask)
        assert not a100.serves_inference(ours, single, pair, mask)                  # fp32 operands
        assert not a100.serves_inference(ours, sb.expand(2, -1, -1), pb.expand(2, -1, -1, -1), mask)   # B = 2
        assert not a100.serves_inference(ours, sb, pb, torch.ones(2, 128, dtype=torch.bool, device="cuda"))
    assert not a100.serves_inference(ours, sb, pb, mask)                           # gradients enabled


def test_apb_a100_matches_the_triton_module_path():
    """With the A100 path off the module runs its Triton composition: the two agree to bf16 accuracy."""
    import os

    _, ours, _ = _modules()
    single, pair, mask = _inputs(384, True)
    sb, pb = single.bfloat16(), pair.bfloat16()
    with torch.no_grad():
        got = ours(sb, pb, mask)
        os.environ["MINIWORLD_APB_SM80"] = "0"
        try:
            base = ours(sb, pb, mask)
        finally:
            del os.environ["MINIWORLD_APB_SM80"]
    assert _rel(got, base) < 1.5e-2


@pytest.mark.parametrize("shape", SHAPES)
def test_apb_a100_compiled(shape):
    """Inference through torch.compile (fullgraph) matches eager."""
    torch._dynamo.reset()                   # each shape compiles the module's forward afresh (recompile limit)
    _, ours, _ = _modules(shape=shape)
    single, pair, mask = _inputs(384, True, d=shape[1])
    sb, pb = single.bfloat16(), pair.bfloat16()
    comp = torch.compile(ours, fullgraph=True)
    with torch.no_grad():
        assert _rel(comp(sb, pb, mask), ours(sb, pb, mask)) < 1e-6


def test_apb_a100_cuda_graph_replay():
    """A captured call replays bit-identically (the packs and the extension are resolved before capture)."""
    _, ours, _ = _modules()
    single, pair, mask = _inputs(384, True)
    sb, pb = single.bfloat16(), pair.bfloat16()
    with torch.no_grad():
        want = ours(sb, pb, mask)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            ours(sb, pb, mask)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = ours(sb, pb, mask)
        graph.replay()
        torch.cuda.synchronize()
    assert torch.equal(out, want)


def _train(mod, single, pair, mask, dy):
    s = single.detach().clone().requires_grad_()
    p = pair.detach().clone().requires_grad_()
    y = mod(s, p, mask)
    y.backward(dy)
    return y, {"single": s.grad, "pair": p.grad, **{n: q.grad for n, q in mod.named_parameters()}}


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("L", [50, 128, 208, 384])
def test_apb_a100_training(L, masked, shape):
    from miniworld_engine.integrations import attention_pair_bias_sm80 as a100

    ref, ours, tb = _modules(shape=shape)
    single, pair, mask = _inputs(L, masked, d=shape[1])
    dy = torch.randn_like(single)
    sb, pb = single.bfloat16(), pair.bfloat16()
    assert a100.serves_train(ours, sb, pb, mask)
    yw, gw = _train(ref, single, pair, mask, dy)
    yg, gg = _train(ours, sb, pb, mask, dy.bfloat16())
    yb, gb = _train(tb, sb, pb, mask, dy.bfloat16())
    assert _rel(yg, yw) < 1.3 * _rel(yb, yw) + 1e-3
    # ln_pair.bias shifts every logit of a head by one constant: its true gradient is 0 (the fp32 module returns rounding noise)
    zero = "ln_pair.bias"
    assert (gg[zero] == 0).all()
    assert gw[zero].norm() < 1e-4 * gw["ln_pair.weight"].norm()
    worst = []
    for n in (n for n in gw if n != zero):
        e, e0 = _rel(gg[n], gw[n]), _rel(gb[n], gw[n])
        worst.append((e / max(e0, 1e-3), n, e, e0))
        assert gg[n].dtype == gb[n].dtype
    worst.sort(reverse=True)
    print("worst ratios:", worst[:3])
    for _r, n, e, e0 in worst:
        assert e < 1.3 * e0 + 2e-3, (n, e, e0)


def test_apb_a100_training_is_bit_reproducible_except_the_column_sums():
    """The activations' gradients (dx, dz) are deterministic; the parameter gradients that are column sums add per-block partials atomically (to rounding)."""
    _, ours, _ = _modules()
    single, pair, mask = _inputs(384, True)
    sb, pb = single.bfloat16(), pair.bfloat16()
    dy = torch.randn_like(sb)
    _, g1 = _train(ours, sb, pb, mask, dy)
    for p in ours.parameters():
        p.grad = None
    _, g2 = _train(ours, sb, pb, mask, dy)
    assert torch.equal(g1["single"], g2["single"])
    assert torch.equal(g1["pair"], g2["pair"])
    for n in g1:
        assert _rel(g1[n], g2[n]) < 1e-2, n


@pytest.mark.parametrize("shape", SHAPES)
def test_apb_a100_training_compiled(shape):
    """A training step through torch.compile matches eager."""
    torch._dynamo.reset()
    _, ours, _ = _modules(shape=shape)
    single, pair, mask = _inputs(384, True, d=shape[1])
    sb, pb = single.bfloat16(), pair.bfloat16()
    comp = torch.compile(ours, fullgraph=True)
    dy = torch.randn_like(sb)
    ye, ge = _train(ours, sb, pb, mask, dy)
    for p in ours.parameters():
        p.grad = None
    yc, gc = _train(comp, sb, pb, mask, dy)
    assert _rel(yc, ye) < 1e-6
    for n in ge:                            # the column sums (dWf, ln weights, dbq) add per-block partials atomically
        assert _rel(gc.get(n, gc.get("_orig_mod." + n)), ge[n]) < 1e-3, n


def test_apb_a100_training_everything_masked_is_finite():
    """A sample with no valid key: the finite fill (-1e4 natural) keeps the forward and every gradient finite (the softmax is uniform)."""
    _, ours, _ = _modules()
    single, pair, _ = _inputs(256, False)
    mask = torch.zeros(1, 256, dtype=torch.bool, device="cuda")
    sb, pb = single.bfloat16(), pair.bfloat16()
    y, grads = _train(ours, sb, pb, mask, torch.randn_like(sb))
    assert torch.isfinite(y).all()
    for n, g in grads.items():
        assert torch.isfinite(g).all(), n
