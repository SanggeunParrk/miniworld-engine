"""AttentionPairBias training on H100 (integrations/attention_pair_bias_b200.py, sm_90a branch) against the fp32 PyTorch module.

The bar is the guide's (docs/kernels/gpu-kernel-development-guide.md section 6): the output and every input / parameter gradient
of the native path are no worse than 1.3x the bf16 PyTorch module's error against the fp32 module (the Triton path sits at the
same error), plus fp32 gradients that were never rounded to bf16, bitwise repeatability, the compiled module, and the live
CUDA-graph replay. ``settings.apb_h100_exact_dwb`` (default off) additionally recomputes the pair-weight gradient in fp32; it is
not what the default path is held to.
"""

import copy

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.modules import AttentionPairBias
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="Hopper (sm_90) required",
    ),
]

SHAPES = [(8, 384), (12, 384), (16, 384), (24, 384), (16, 512)]  # (heads, d_single)


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _modules(shape, seed=0):
    torch.manual_seed(seed)
    ref = AttentionPairBias(
        shape[1], 128, shape[0], implementation=ImplementationType.PYTORCH
    )
    with (
        torch.no_grad()
    ):  # non-default parameters: zero-initialised to_out, unit LayerNorms
        for p in ref.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    return ref.cuda()


def _native(ref, master):
    ours = copy.deepcopy(ref)
    ours.implementation = ImplementationType.MINIWORLD
    return ours if master else ours.bfloat16()


def _inputs(L, d, masked, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    single = torch.randn(1, L, d, device="cuda", generator=g)
    pair = torch.randn(1, L, L, 128, device="cuda", generator=g)
    mask = None
    if masked:
        mask = torch.rand(1, L, device="cuda", generator=g) > 0.2
        mask[0, 0] = True
    return single, pair, mask


def _train(mod, single, pair, mask, dy, autocast=False):
    s = single.detach().clone().requires_grad_()
    p = pair.detach().clone().requires_grad_()
    for q in mod.parameters():
        q.grad = None
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        y = mod(s, p, mask)
    y.backward(dy.to(y.dtype))
    return y.detach(), {
        "single": s.grad,
        "pair": p.grad,
        **{n.replace("_orig_mod.", ""): q.grad for n, q in mod.named_parameters()},
    }


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"h{s[0]}d{s[1]}")
@pytest.mark.parametrize("masked", [False, True], ids=["nomask", "mask"])
@pytest.mark.parametrize("L", [128, 384])
@pytest.mark.parametrize("master", [False, True], ids=["bf16params", "fp32master"])
def test_apb_h100_training_accuracy(shape, masked, L, master):
    from miniworld_engine.integrations import attention_pair_bias_b200 as native

    ref = _modules(shape)
    single, pair, mask = _inputs(L, shape[1], masked)
    dy = torch.randn_like(single)
    sb, pb = single.bfloat16(), pair.bfloat16()
    ours = _native(ref, master)
    torch_bf = copy.deepcopy(ref)
    if not master:
        torch_bf = torch_bf.bfloat16()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=master):
        assert native.serves_train(ours, sb, pb, mask)
    yw, gw = _train(ref, single, pair, mask, dy)
    yg, gg = _train(ours, sb, pb, mask, dy, autocast=master)
    yb, gb = _train(torch_bf, sb, pb, mask, dy, autocast=master)
    assert _rel(yg, yw) < 1.3 * _rel(yb, yw) + 1e-3
    assert "ln_pair.bias" not in gg
    for name in gw:
        e, e0 = _rel(gg[name], gw[name]), _rel(gb[name], gw[name])
        assert e < 1.3 * e0 + 2e-3, (name, e, e0)
        assert gg[name].dtype == gb[name].dtype, name
        if master and name not in ("single", "pair"):
            # the master's gradient is the fp32 accumulation, not a bf16 value widened
            assert not torch.equal(gg[name], gg[name].bfloat16().float()), name


@pytest.mark.parametrize(
    "shape", [(8, 384), (16, 384)], ids=lambda s: f"h{s[0]}d{s[1]}"
)
def test_apb_h100_training_repeatable_and_graph(shape):
    ref = _modules(shape, seed=3)
    ours = _native(ref, master=True)
    single, pair, mask = _inputs(256, shape[1], True, seed=4)
    sb, pb = single.bfloat16(), pair.bfloat16()
    dy = torch.randn_like(single).bfloat16()
    y0, g0 = _train(ours, sb, pb, mask, dy, autocast=True)
    y1, g1 = _train(ours, sb, pb, mask, dy, autocast=True)
    assert torch.equal(y0, y1)
    for n in g0:
        assert torch.equal(g0[n], g1[n]), n

    # live CUDA-graph replay: new inputs and weights between replays are seen
    s_ = sb.clone().requires_grad_()
    p_ = pb.clone().requires_grad_()
    for q in ours.parameters():
        q.grad = None

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = ours(s_, p_, mask)
        return torch.autograd.grad(
            y, [s_, p_, *ours.parameters()], dy, allow_unused=False
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    graph.replay()
    first = [t.clone() for t in out]
    eager = step()
    for a, b in zip(first, eager, strict=True):
        assert torch.equal(a, b)
    with torch.no_grad():
        s_.copy_(torch.randn_like(s_))
        ours.to_query.weight.add_(0.01)
    graph.replay()
    changed = step()
    for a, b in zip(out, changed, strict=True):
        assert torch.equal(a, b)
    assert not torch.equal(first[0], out[0])


@pytest.mark.parametrize(
    "shape", [(8, 384), (24, 384)], ids=lambda s: f"h{s[0]}d{s[1]}"
)
def test_apb_h100_training_compiled(shape):
    torch._dynamo.reset()
    ref = _modules(shape, seed=5)
    ours = _native(ref, master=True)
    single, pair, mask = _inputs(256, shape[1], True, seed=6)
    sb, pb = single.bfloat16(), pair.bfloat16()
    dy = torch.randn_like(single).bfloat16()
    comp = torch.compile(ours, fullgraph=True)
    ye, ge = _train(ours, sb, pb, mask, dy, autocast=True)
    yc, gc = _train(comp, sb, pb, mask, dy, autocast=True)
    assert _rel(yc, ye) < 1e-6
    for n, g in ge.items():
        assert _rel(gc[n], g) < 1e-3, n


@pytest.mark.parametrize("shape", [(8, 384)], ids=lambda s: f"h{s[0]}d{s[1]}")
def test_apb_h100_exact_dwb_switch(shape):
    """The opt-in fp32 pair-weight gradient (``apb_h100_exact_dwb``) agrees with the fused gradient to the bf16 noise floor
    and is closer to the fp32 module."""
    pytest.importorskip("torch")
    import os

    if not os.environ.get("CUTLASS_PATH"):
        pytest.skip("the fp32 oracle build needs CUTLASS_PATH (CUTLASS 4.2 headers)")
    ref = _modules(shape, seed=7)
    single, pair, mask = _inputs(384, shape[1], True, seed=8)
    sb, pb = single.bfloat16(), pair.bfloat16()
    dy = torch.randn_like(single)
    _, gw = _train(ref, single, pair, mask, dy)
    ours = _native(ref, master=False)
    prev = settings.configure(apb_h100_exact_dwb=False)
    try:
        _, g_fused = _train(ours, sb, pb, mask, dy.bfloat16())
        settings.configure(apb_h100_exact_dwb=True)
        _, g_exact = _train(ours, sb, pb, mask, dy.bfloat16())
    finally:
        settings.configure(apb_h100_exact_dwb=prev.apb_h100_exact_dwb)
    n = "to_bias.weight"
    assert _rel(g_exact[n], gw[n]) < _rel(g_fused[n], gw[n])
    assert _rel(g_exact[n], g_fused[n]) < 5e-2
