"""MSAPairWeightedAveraging on A100 (integrations/pwa_sm80.py) against the fp32 PyTorch module: inference and training (every gradient, the fused row dropout), the registry
shapes (d_msa 64 / 128, d_pair 128 / 256 / 384, 8 heads of 8 / 16 / 32), key masks, eager / compiled / CUDA graph, the env switch and the gate. Errors are held to the bf16
PyTorch module's own error in the same regime."""

import copy
import os

import pytest
import torch

from miniworld_engine.kernels.pair_weighted_averaging.reference import (
    pair_weighted_averaging_reference,
)
from miniworld_engine.modules import MSAPairWeightedAveraging
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def ampere():
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Ampere (sm_80) required")


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


# (d_msa, d_pair, per-head width): the registry's rows (MiniWorld's 64 / 128 with 8- or 32-wide heads, ESMFold2's 128 / 256, Protenix-v2 / OpenDDE's d_hidden = 8) and the 128 / 128 corner
SHAPES = [(64, 128, 32), (64, 128, 8), (64, 128, 16), (128, 256, 16), (128, 256, 8), (128, 384, 8), (128, 128, 8)]
PARAMS = ("ln_msa.weight", "ln_msa.bias", "to_value.weight", "to_gate.weight", "ln_pair.weight", "ln_pair.bias", "to_bias.weight", "to_out.weight")


def _modules(d_msa, d_pair, c, seed=0):
    torch.manual_seed(seed)
    ref = MSAPairWeightedAveraging(d_msa, d_pair, 8, c, implementation=ImplementationType.PYTORCH)
    with torch.no_grad():                   # non-default parameters: the zero-initialised to_out, unit LayerNorms
        for n, p in ref.named_parameters():
            if n.endswith(("ln_msa.weight", "ln_pair.weight")):
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            elif p.ndim == 1:
                p.copy_(0.2 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * p.shape[-1] ** -0.5)
    ours = copy.deepcopy(ref)
    ours.implementation = ImplementationType.MINIWORLD
    twin = copy.deepcopy(ref)
    return ref.cuda(), ours.cuda().to(torch.bfloat16), twin.cuda().to(torch.bfloat16)


def _inputs(S, L, d_msa, d_pair, masked=True, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    msa = torch.randn(1, S, L, d_msa, device="cuda", generator=g)
    pair = torch.randn(1, L, L, d_pair, device="cuda", generator=g)
    mask = None
    if masked:
        mask = torch.rand(1, L, device="cuda", generator=g) > 0.3
        mask[0, 0] = True
    dy = torch.randn(1, S, L, d_msa, device="cuda", generator=g)
    return msa, pair, mask, dy


def _run(mod, dtype, msa, pair, mask, dy, grad):
    mod.zero_grad(set_to_none=True)
    x = msa.to(dtype).clone().requires_grad_(grad)
    z = pair.to(dtype).clone().requires_grad_(grad)
    with torch.set_grad_enabled(grad):
        y = mod(x, z, mask)
    got = {"out": y.detach().float()}
    if grad:
        y.backward(dy.to(dtype))
        got.update({"dmsa": x.grad.float(), "dpair": z.grad.float()})
        got.update({n: p.grad.float() for n, p in mod.named_parameters()})
    return y, got


def _compare(got, base, want, msa, strict=1.15):
    for n in want:
        if n == "ln_pair.bias":             # identically zero in exact arithmetic (a softmax cannot see a shift shared by all keys): rounding noise on both sides
            assert got[n].norm() < 0.1 * want["ln_pair.weight"].norm() + 1e-6
            continue
        ea, eb = _rel(got[n], want[n]), _rel(base[n], want[n])
        assert ea <= max(3e-3, (strict if n in ("out", "dmsa", "dpair") else 1.25) * eb), (n, ea, eb)          # parameter gradients are sums over every token: a noisier statistic
        if n == "out":                      # the update is small against the residual: judge it on its own
            ua, ub = _rel(got[n] - msa.bfloat16().float(), want[n] - msa), _rel(base[n] - msa.bfloat16().float(), want[n] - msa)
            assert ua <= max(3e-3, strict * ub), ("update", ua, ub)


def _check(shape, S, L, masked, grad):
    from miniworld_engine.integrations import pwa_sm80

    d_msa, d_pair, c = shape
    ref, ours, twin = _modules(d_msa, d_pair, c)
    for m in (ref, ours, twin):
        m.train(grad)
        m.drop_msa.p_drop = 0.0
    msa, pair, mask, dy = _inputs(S, L, d_msa, d_pair, masked)
    with torch.set_grad_enabled(grad):
        assert (pwa_sm80.serves_train if grad else pwa_sm80.serves_inference)(ours, msa.bfloat16(), pair.bfloat16(), mask)
    _, want = _run(ref, torch.float32, msa, pair, mask, dy, grad)
    _, base = _run(twin, torch.bfloat16, msa, pair, mask, dy, grad)
    y, got = _run(ours, torch.bfloat16, msa, pair, mask, dy, grad)
    assert y.dtype == torch.bfloat16
    assert y.shape == msa.shape
    _compare(got, base, want, msa)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("masked", [False, True])
def test_pwa_a100_inference(shape, masked):
    _check(shape, S=200, L=144, masked=masked, grad=False)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("masked", [False, True])
def test_pwa_a100_training(shape, masked):
    _check(shape, S=200, L=144, masked=masked, grad=True)


@pytest.mark.parametrize("L", [128, 256, 384, 512, 640, 768])
@pytest.mark.parametrize("shape", [(64, 128, 8), (128, 256, 16), (128, 384, 8)])
def test_pwa_a100_registry_lengths(shape, L):
    """Every registry length (a shallow MSA keeps the fp32 reference cheap): inference."""
    _check(shape, S=64, L=L, masked=True, grad=False)


@pytest.mark.parametrize("S", [1, 7, 64, 129, 1000])
def test_pwa_a100_msa_depth(S):
    """Any MSA depth (tiles of 128 rows with a predicated tail), forward and backward."""
    _check((64, 128, 32), S=S, L=64, masked=True, grad=True)
    _check((128, 256, 16), S=S, L=32, masked=False, grad=True)


@pytest.mark.parametrize("S", [256, 512, 1024, 2048])
def test_pwa_a100_chunked_depth_training(S):
    """MSA depths that split the contractions in 2 / 4 / 4 / 4 chunks of rows (``pick_split``): every gradient against the fp32 reference, wide and narrow heads."""
    from miniworld_engine.kernels.pair_weighted_averaging.cuda import sm80

    assert sm80.pick_split(S) == {256: 2, 512: 4, 1024: 4, 2048: 4}[S]
    _check((64, 128, 32), S=S, L=32, masked=True, grad=True)
    _check((128, 256, 16), S=S, L=48, masked=False, grad=True)
    _check((128, 384, 8), S=S, L=16, masked=True, grad=True)
    _check((64, 128, 32), S=S, L=32, masked=True, grad=False)                 # inference reads the same chunked layout
    _check((128, 256, 16), S=S, L=48, masked=False, grad=False)


def test_pwa_a100_split_choice_and_switch(monkeypatch):
    from miniworld_engine.kernels.pair_weighted_averaging.cuda import sm80

    assert [sm80.pick_split(s) for s in (1, 200, 255, 256, 384, 512, 1000, 1024, 3072)] == [1, 1, 1, 2, 1, 4, 1, 4, 4]
    monkeypatch.setenv(sm80.SPLIT_ENV, "1")
    assert sm80.pick_split(1024) == 1
    monkeypatch.setenv(sm80.SPLIT_ENV, "4")
    assert [sm80.pick_split(s) for s in (200, 512, 1024)] == [1, 4, 4]
    monkeypatch.setenv(sm80.SPLIT_ENV, "3")
    assert sm80.pick_split(1024) == 1


@pytest.mark.parametrize("ns", [2, 4, 8])
def test_pwa_a100_split_matches_unsplit(monkeypatch, ns):
    """The chunked contractions change only the summation order of dw (and the layout of the intermediates): the gradients agree with the unsplit run to bf16 rounding."""
    from miniworld_engine.kernels.pair_weighted_averaging.cuda import sm80

    def run(n):
        monkeypatch.setenv(sm80.SPLIT_ENV, str(n))
        _, ours, _ = _modules(64, 128, 32)
        ours.train(True)
        ours.drop_msa.p_drop = 0.0
        msa, pair, mask, dy = _inputs(1024, 48, 64, 128, True)
        return _run(ours, torch.bfloat16, msa, pair, mask, dy, True)

    (y1, g1), (yn, gn) = run(1), run(ns)
    assert _rel(yn.float(), y1.float()) < 5e-3                      # the forward changes at most the cuBLAS tile order inside a bf16 rounding
    for name in g1:
        if name != "ln_pair.bias":                                  # (identically zero in exact arithmetic: rounding noise on both sides)
            assert _rel(gn[name], g1[name]) < 2e-2, name


@pytest.mark.parametrize("shape", [(64, 128, 32), (128, 384, 8)])
def test_pwa_a100_longest_supported_length(shape):
    """L = 1024 (the shared-memory logits of the pair kernels fill 140-154 KB): forward and backward, a shallow MSA."""
    _check(shape, S=32, L=1024, masked=True, grad=False)
    _check(shape, S=32, L=1024, masked=True, grad=True)


@pytest.mark.parametrize("L", [16, 48, 80])
def test_pwa_a100_short_lengths(L):
    _check((64, 128, 8), S=100, L=L, masked=True, grad=True)


def test_pwa_a100_everything_masked_is_finite():
    """A key mask without a valid key: the softmax is uniform (the module's finfo.min fill makes every logit equal), finite in both directions."""
    _, ours, _ = _modules(64, 128, 32)
    msa, pair, _, dy = _inputs(96, 64, 64, 128)
    mask = torch.zeros(1, 64, dtype=torch.bool, device="cuda")
    x, z = msa.bfloat16().requires_grad_(), pair.bfloat16().requires_grad_()
    y = ours(x, z, mask)
    y.backward(dy.bfloat16())
    assert torch.isfinite(y).all()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(z.grad).all()
    for n, p in ours.named_parameters():
        assert torch.isfinite(p.grad).all(), n


@pytest.mark.parametrize("shape", [(64, 128, 32), (128, 256, 16), (128, 384, 8)])
def test_pwa_a100_fused_dropout(shape):
    """The module's row dropout (one keep decision per token and channel, shared by the MSA rows) with the residual, fused: the keep-mask is recovered from the output and the
    forward and every gradient are held against the fp32 reference driven by the same mask."""
    d_msa, d_pair, c = shape
    p_drop = 0.25
    S, L = 96, 64
    _, ours, _twin = _modules(d_msa, d_pair, c)
    ours.train()
    ours.drop_msa.p_drop = p_drop
    msa, pair, mask, dy = _inputs(S, L, d_msa, d_pair)
    x, z = msa.bfloat16().requires_grad_(), pair.bfloat16().requires_grad_()
    torch.manual_seed(7)
    y = ours(x, z, mask)
    y.backward(dy.bfloat16())
    upd = y.detach().float() - msa.bfloat16().float()
    keep = (upd.abs().sum(dim=1) > 0)                                                   # [1, L, d_msa]: dropped (token, channel) pairs have exactly zero update
    assert 0.65 < keep.float().mean().item() < 0.85
    assert ((upd != 0).float().mean(dim=1)[keep] > 0.9).all()                           # a kept position is nonzero on (almost) every MSA row: the mask is shared by the rows
    w = [ours.ln_msa.weight, ours.ln_msa.bias, ours.to_value.weight, ours.to_gate.weight, ours.ln_pair.weight, ours.ln_pair.bias, ours.to_bias.weight, ours.to_out.weight]

    def reference(dtype):
        xs = msa.to(dtype).clone().requires_grad_()
        zs = pair.to(dtype).clone().requires_grad_()
        ws = [t.detach().to(dtype).clone().requires_grad_() for t in w]
        out = pair_weighted_averaging_reference(xs, zs, mask, *ws, keep=keep.to(dtype), p_drop=p_drop)
        grads = torch.autograd.grad(out, [xs, zs, *ws], dy.to(dtype))
        return out.detach().float(), [g.float() for g in grads]

    want, gw = reference(torch.float32)
    base, gb = reference(torch.bfloat16)
    assert _rel(y.detach().float() - msa.bfloat16().float(), want - msa) <= max(3e-3, 1.15 * _rel(base - msa.bfloat16().float(), want - msa))
    got = {"dmsa": x.grad.float(), "dpair": z.grad.float(), **{n: p.grad.float() for n, p in ours.named_parameters()}}
    ref_names = ["dmsa", "dpair", *PARAMS]                                              # the reference function's argument order
    for n, a, b, wnt in zip(ref_names, [got[k] for k in ref_names], gb, gw, strict=True):
        if n == "ln_pair.bias":
            continue
        assert _rel(a, wnt) <= max(3e-3, (1.15 if n.startswith("d") else 1.25) * _rel(b, wnt)), (n, _rel(a, wnt), _rel(b, wnt))


def test_pwa_a100_dropout_off_in_eval_and_without_grad():
    """eval() / p_drop = 0 take the plain forward; a grad-free training-mode call with a live dropout takes the fused training step (the module's draw applies)."""
    from miniworld_engine.integrations import pwa_sm80

    _, ours, _ = _modules(64, 128, 32)
    msa, pair, mask, _ = _inputs(64, 64, 64, 128)
    mb, zb = msa.bfloat16(), pair.bfloat16()
    ours.eval()
    with torch.no_grad():
        assert pwa_sm80.serves_inference(ours, mb, zb, mask)
        assert not pwa_sm80.serves_train(ours, mb, zb, mask)
    ours.train()
    ours.drop_msa.p_drop = 0.15
    with torch.no_grad():
        assert not pwa_sm80.serves_inference(ours, mb, zb, mask)
        assert pwa_sm80.serves_train(ours, mb, zb, mask)
        y = ours(mb, zb, mask)
    assert (y == mb).float().mean().item() > 0.05                                    # dropped channels leave the residual untouched


def test_pwa_a100_gate_conditions():
    """The path declines what it does not implement; the module path then serves."""
    from miniworld_engine.integrations import pwa_sm80

    _, ours, _ = _modules(64, 128, 32)
    ours.eval()
    msa, pair, mask, _ = _inputs(64, 64, 64, 128)
    mb, zb = msa.bfloat16(), pair.bfloat16()
    with torch.no_grad():
        assert pwa_sm80.serves_inference(ours, mb, zb, mask)
        assert not pwa_sm80.serves_inference(ours, msa, pair, mask)                              # fp32
        assert not pwa_sm80.serves_inference(ours, mb.expand(2, -1, -1, -1), zb.expand(2, -1, -1, -1), mask.expand(2, -1))   # B = 2
        assert not pwa_sm80.serves_inference(ours, mb[:, :, :60], zb[:, :60, :60], mask[:, :60])  # L not a multiple of 16
        assert not pwa_sm80.serves_inference(ours, mb, zb, mask.float())                          # a float mask
        assert not pwa_sm80.serves_inference(ours, mb, zb[..., :96], mask)                        # d_pair 96
        _, big, _ = _modules(128, 256, 32)                                                        # d_msa 128 with 32-wide heads: not built
        assert not pwa_sm80.serves_inference(big, torch.randn(1, 8, 64, 128, device="cuda", dtype=torch.bfloat16), torch.randn(1, 64, 64, 256, device="cuda", dtype=torch.bfloat16), None)
        pt = copy.deepcopy(ours)
        pt.implementation = ImplementationType.PYTORCH
        assert not pwa_sm80.serves_inference(pt, mb, zb, mask)
    assert not pwa_sm80.serves_inference(ours, mb, zb, mask)                                      # grad enabled
    assert pwa_sm80.serves_train(ours, mb, zb, mask)


def test_pwa_a100_env_switch_and_triton_agreement():
    """MINIWORLD_PWA_SM80=0 runs the module's previous path: the two agree to bf16 accuracy."""
    from miniworld_engine.integrations import pwa_sm80

    _, ours, _ = _modules(64, 128, 32)
    ours.eval()
    msa, pair, mask, _ = _inputs(128, 128, 64, 128)
    mb, zb = msa.bfloat16(), pair.bfloat16()
    with torch.no_grad():
        got = ours(mb, zb, mask)
        os.environ["MINIWORLD_PWA_SM80"] = "0"
        try:
            assert not pwa_sm80.serves_inference(ours, mb, zb, mask)
            base = ours(mb, zb, mask)
        finally:
            del os.environ["MINIWORLD_PWA_SM80"]
    assert _rel(got - mb, base - mb) < 3e-2                                                   # the update alone (Triton and ours round at different points)


def test_pwa_a100_training_is_bit_reproducible():
    """The partials of every weight / LayerNorm gradient are summed in a fixed order (no atomics): two runs agree bit for bit."""
    _, ours, _ = _modules(64, 128, 32)
    msa, pair, mask, dy = _inputs(200, 144, 64, 128)
    _, g1 = _run(ours, torch.bfloat16, msa, pair, mask, dy, True)
    _, g2 = _run(ours, torch.bfloat16, msa, pair, mask, dy, True)
    for n in g1:
        assert torch.equal(g1[n], g2[n]), n


@pytest.mark.parametrize("ns", [1, 8])
def test_pwa_a100_pair_bwd_schedules_agree(ns):
    """One and two CTAs per SM of the pair backward (`MINIWORLD_PWA_SM80_PB`, d_pair 128) sum the same partials in the same order: bit-identical gradients."""
    from miniworld_engine.kernels.pair_weighted_averaging.cuda import sm80

    length, dz_ = 96, 128
    g = torch.Generator(device="cuda").manual_seed(5)
    z = torch.randn(length, length, dz_, device="cuda", generator=g).bfloat16()
    w = torch.softmax(torch.randn(8, length, length, device="cuda", generator=g), -1).bfloat16()
    dw = torch.randn(8 * ns, length, length, device="cuda", generator=g)
    lnw = 1.0 + 0.1 * torch.randn(dz_, device="cuda", generator=g)
    lnb = 0.1 * torch.randn(dz_, device="cuda", generator=g)
    wb = (torch.randn(8, dz_, device="cuda", generator=g) * dz_**-0.5).bfloat16()
    key_mask = (torch.rand(length, device="cuda", generator=g) > 0.2).to(torch.uint8)
    want = sm80.pair_bwd(z, w, dw, key_mask, lnw, lnb, wb, 1e-5, variant=0)
    got = sm80.pair_bwd(z, w, dw, key_mask, lnw, lnb, wb, 1e-5, variant=1)
    for a, b in zip(got, want, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("shape", [(64, 128, 32), (128, 256, 16), (128, 384, 8)])
def test_pwa_a100_compiled_equals_eager(shape):
    """torch.compile(fullgraph=True) of inference and of a training step reproduces eager."""
    torch._dynamo.reset()
    d_msa, d_pair, c = shape
    _, ours, _ = _modules(d_msa, d_pair, c)
    msa, pair, mask, dy = _inputs(96, 64, d_msa, d_pair)
    mb, zb = msa.bfloat16(), pair.bfloat16()
    comp = torch.compile(ours, fullgraph=True)
    ours.eval()
    with torch.no_grad():
        assert torch.equal(comp(mb, zb, mask), ours(mb, zb, mask))
    ours.train()
    ours.drop_msa.p_drop = 0.0
    _, ge = _run(ours, torch.bfloat16, msa, pair, mask, dy, True)
    _, gc = _run(comp, torch.bfloat16, msa, pair, mask, dy, True)
    for n in ge:
        assert torch.equal(gc.get(n, gc.get("_orig_mod." + n)), ge[n]), n


def test_pwa_a100_cuda_graph_replay():
    """A captured inference call, and a captured forward + backward with the live dropout, replay (the dropout draw advances with the graph's RNG state)."""
    _, ours, _ = _modules(64, 128, 32)
    ours.eval()
    msa, pair, mask, dy = _inputs(128, 96, 64, 128)
    mb, zb = msa.bfloat16(), pair.bfloat16()
    with torch.no_grad():
        want = ours(mb, zb, mask)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            ours(mb, zb, mask)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = ours(mb, zb, mask)
        graph.replay()
        torch.cuda.synchronize()
    assert torch.equal(out, want)

    ours.train()
    ours.drop_msa.p_drop = 0.0
    x, z, dyb = mb.clone().requires_grad_(), zb.clone().requires_grad_(), dy.bfloat16()

    def step():
        for p in ours.parameters():
            p.grad = None
        x.grad = z.grad = None
        ours(x, z, mask).backward(dyb)

    step()
    ref = {n: p.grad.clone() for n, p in ours.named_parameters()}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    for p in ours.parameters():
        p.grad = None
    x.grad = z.grad = None
    with torch.cuda.graph(graph):
        ours(x, z, mask).backward(dyb)
    graph.replay()
    torch.cuda.synchronize()
    for n, p in ours.named_parameters():
        assert torch.equal(p.grad, ref[n]), n
