"""AttentionPairBias with a gate bias (Chai-1's pairformer single attention, ``gate_bias=1.0``) and Chai-1's width (16 x 24 heads,
d_pair 256) on B200, inference and training: the native CUDA path (integrations/attention_pair_bias_b200.py: the gate constant on the
g rows of the packed bias, pair_bias_fwd's / pair_bias_bwd's 256-wide rings for 16 heads) against the fp32 PyTorch module, C++ host
path and Python path.

The bar is the H100 test's: the output (and in training every input and parameter gradient) is no worse than 1.3x the bf16 PyTorch
module's error against the fp32 module. Also: the call is served by the native path, lengths off the 16-row grid (host padding) and a
batch of samples with their own pairs, gate bias 0 unchanged, the compiled module / the live CUDA-graph replay agree with the eager
call, and the training step is bitwise run to run, the C++ host path bitwise the Python sequence, a batch bitwise its samples.
"""

import copy

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.modules import AttentionPairBias
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
        reason="Blackwell (sm_100) required",
    ),
]

# (heads, d_single, d_pair, gate_bias)
CASES = [
    (16, 384, 256, 1.0),  # Chai-1
    (16, 384, 256, 0.0),
    (16, 384, 128, 1.0),
    (8, 384, 128, 1.0),
    (12, 384, 128, 0.5),
]


@pytest.fixture(autouse=True)
def restore_settings():
    prev = settings.current()
    yield
    settings.configure(apb_b200_host=prev.apb_b200_host)


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _reference(case, seed=0):
    heads, d, dp, gb = case
    torch.manual_seed(seed)
    ref = AttentionPairBias(d, dp, heads, gate_bias=gb, implementation=ImplementationType.PYTORCH)
    with torch.no_grad():  # non-default parameters: the zero-initialised to_out and unit LayerNorms move
        for p in ref.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    return ref.cuda()


def _native(ref, master):
    ours = copy.deepcopy(ref)
    ours.implementation = ImplementationType.MINIWORLD
    return ours if master else ours.bfloat16()


def _inputs(L, d, dp, masked, seed=1, B=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    single = torch.randn(B, L, d, device="cuda", generator=g)
    pair = torch.randn(B, L, L, dp, device="cuda", generator=g)
    mask = None
    if masked:
        mask = torch.rand(B, L, device="cuda", generator=g) > 0.2
        mask[:, 0] = True
    return single, pair, mask


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"h{c[0]}d{c[1]}p{c[2]}g{c[3]}")
@pytest.mark.parametrize("masked", [False, True], ids=["nomask", "mask"])
@pytest.mark.parametrize("L", [128, 200, 384])
@pytest.mark.parametrize("master", [False, True], ids=["bf16params", "fp32master"])
@pytest.mark.parametrize("host", [True, False], ids=["host", "python"])
def test_b200_apb_gate_bias_inference_accuracy(case, masked, L, master, host):
    from miniworld_engine.integrations import attention_pair_bias_b200 as native

    settings.configure(apb_b200_host=host)
    ref = _reference(case)
    single, pair, mask = _inputs(L, case[1], case[2], masked)
    sb, pb = single.bfloat16(), pair.bfloat16()
    ours = _native(ref, master)
    torch_bf = copy.deepcopy(ref)
    if not master:
        torch_bf = torch_bf.bfloat16()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=master):
        assert native.serves_inference(ours, sb, pb, mask), "the native CUDA path must serve this call"
        y_ref = ref(single, pair, mask)
        y_ours = ours(sb, pb, mask)
        y_bf = torch_bf(sb, pb, mask)
    assert torch.isfinite(y_ours).all()
    assert _rel(y_ours, y_ref) < 1.3 * _rel(y_bf, y_ref) + 1e-3, (_rel(y_ours, y_ref), _rel(y_bf, y_ref))


@pytest.mark.parametrize("host", [True, False], ids=["host", "python"])
def test_b200_apb_chai_batch_is_each_sample_alone(host):
    """Chai-1's width with B samples, each its own pair and mask: the batched call is the per-sample calls, bitwise."""
    settings.configure(apb_b200_host=host)
    case = (16, 384, 256, 1.0)
    ours = _native(_reference(case), False)
    single, pair, mask = _inputs(256, 384, 256, True, seed=5, B=3)
    sb, pb = single.bfloat16(), pair.bfloat16()
    with torch.no_grad():
        y = ours(sb, pb, mask)
        ys = torch.cat([ours(sb[b : b + 1], pb[b : b + 1], mask[b : b + 1]) for b in range(3)])
    assert torch.equal(y, ys)


def test_b200_apb_gate_bias_is_the_constant_on_the_gate_logits():
    """The option is exactly sigmoid(g + c): the native path moves with c as the PyTorch statements do (c = 0 and c = 3 differ by
    far more than the bf16 noise)."""
    case = (16, 384, 256, 3.0)
    ref = _reference(case)
    ref0 = copy.deepcopy(ref)
    ref0.gate_bias = 0.0
    single, pair, mask = _inputs(256, 384, 256, True)
    sb, pb = single.bfloat16(), pair.bfloat16()
    ours, ours0 = _native(ref, False), _native(ref0, False)
    with torch.no_grad():
        y, y0 = ours(sb, pb, mask), ours0(sb, pb, mask)
        r, r0 = ref(single, pair, mask), ref0(single, pair, mask)
    assert _rel(y, r) < 5e-2 and _rel(y0, r0) < 5e-2
    assert _rel(y, y0) > 5 * _rel(y, r)


def _train(mod, single, pair, mask, dy):
    s = single.detach().clone().requires_grad_()
    p = pair.detach().clone().requires_grad_()
    y = mod(s, p, mask)
    y.backward(dy)
    grads = {"single": s.grad, "pair": p.grad, **{n: q.grad for n, q in mod.named_parameters()}}
    for q in mod.parameters():
        q.grad = None
    return y.detach(), grads  # detached: no autograd node (nor its AccumulateGrad streams) outlives the step


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"h{c[0]}d{c[1]}p{c[2]}g{c[3]}")
@pytest.mark.parametrize("masked", [False, True], ids=["nomask", "mask"])
@pytest.mark.parametrize("L", [128, 384])
@pytest.mark.parametrize("master", [False, True], ids=["bf16params", "fp32master"])
@pytest.mark.parametrize("host", [True, False], ids=["host", "python"])
def test_b200_apb_gate_bias_training_accuracy(case, masked, L, master, host):
    """Training with the gate bias and / or the 256-wide pair: output and every gradient within 1.3x the bf16 module's error."""
    from miniworld_engine.integrations import attention_pair_bias_b200 as native

    settings.configure(apb_b200_host=host)
    ref = _reference(case)
    single, pair, mask = _inputs(L, case[1], case[2], masked)
    dy = torch.randn_like(single)
    sb, pb = single.bfloat16(), pair.bfloat16()
    ours = _native(ref, master)
    torch_bf = copy.deepcopy(ref)
    if not master:
        torch_bf = torch_bf.bfloat16()
    assert native.serves_train(ours, sb, pb, mask), "the native CUDA path must serve this call"
    yw, gw = _train(ref, single, pair, mask, dy)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=master):
        yg, gg = _train(ours, sb, pb, mask, dy.bfloat16())
        yb, gb = _train(torch_bf, sb, pb, mask, dy.bfloat16())
    assert torch.isfinite(yg).all()
    assert _rel(yg, yw) < 1.3 * _rel(yb, yw) + 1e-3, (_rel(yg, yw), _rel(yb, yw))
    worst = []
    for n in gw:
        assert gg[n] is not None and torch.isfinite(gg[n]).all(), n
        assert gg[n].dtype == gb[n].dtype, n
        e, e0 = _rel(gg[n], gw[n]), _rel(gb[n], gw[n])
        worst.append((e / max(e0, 1e-3), n, e, e0))
    worst.sort(reverse=True)
    print("worst ratios:", worst[:3])
    for _r, n, e, e0 in worst:
        assert e < 1.3 * e0 + 2e-3, (n, e, e0)


@pytest.mark.parametrize("L", [128, 256])
def test_b200_apb_chai_training_host_is_python_and_bitwise(L):
    """Chai-1's width with the gate bias, B = 3 samples with their own pairs and masks: the C++ host path is the Python sequence bit for
    bit, a second step repeats every output and gradient bit for bit, and each sample's output / dsingle / dpair are those of a
    one-sample call (the parameter gradients are the samples' sum: within fp32 rounding)."""
    case = (16, 384, 256, 1.0)
    ours = _native(_reference(case, seed=7), True)
    single, pair, mask = _inputs(L, 384, 256, True, seed=8, B=3)
    sb, pb = single.bfloat16(), pair.bfloat16()
    dy = torch.randn_like(sb)
    runs = {}
    for host in (True, False, True):
        settings.configure(apb_b200_host=host)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            runs.setdefault(host, []).append(_train(ours, sb, pb, mask, dy))
    (y0, g0), (y1, g1) = runs[True]
    (yp, gp), = runs[False]
    for y, g in ((y1, g1), (yp, gp)):
        assert torch.equal(y0, y)
        for n in g0:
            assert torch.equal(g0[n], g[n]), n
    settings.configure(apb_b200_host=True)
    psum = {n: torch.zeros_like(t) for n, t in g0.items() if n not in ("single", "pair")}
    for b in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yb, gb = _train(ours, sb[b : b + 1], pb[b : b + 1], mask[b : b + 1], dy[b : b + 1])
        assert torch.equal(yb, y0[b : b + 1])
        assert torch.equal(gb["single"], g0["single"][b : b + 1])
        assert torch.equal(gb["pair"], g0["pair"][b : b + 1])
        for n in psum:
            psum[n] += gb[n]
    for n in psum:
        assert _rel(g0[n], psum[n]) < 1e-5, n


def test_b200_apb_gate_bias_training_graph():
    """The whole training step (forward + backward) captured in a CUDA graph replays the eager step's bits."""
    case = (16, 384, 256, 1.0)
    ours = _native(_reference(case, seed=9), True)
    single, pair, mask = _inputs(256, 384, 256, True, seed=10)
    sb, pb = single.bfloat16(), pair.bfloat16()
    dy = torch.randn_like(sb)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y, want = _train(ours, sb, pb, mask, dy)
    sg, pg = sb.clone().requires_grad_(), pb.clone().requires_grad_()

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = ours(sg, pg, mask)
        out.backward(dy)
        return out

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            for t in (sg, pg, *ours.parameters()):
                t.grad = None
            step()
    torch.cuda.current_stream().wait_stream(stream)
    for t in (sg, pg, *ours.parameters()):
        t.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        yg = step()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, yg)
    got = {"single": sg.grad, "pair": pg.grad, **{n: q.grad for n, q in ours.named_parameters()}}
    for n in want:
        assert torch.equal(want[n], got[n]), n


@pytest.mark.parametrize("case", [(16, 384, 256, 1.0), (16, 384, 128, 1.0)], ids=lambda c: f"p{c[2]}")
def test_b200_apb_gate_bias_compiled_and_graph(case):
    ref = _reference(case, seed=3)
    ours = _native(ref, True)
    single, pair, mask = _inputs(256, case[1], case[2], True, seed=4)
    sb, pb = single.bfloat16(), pair.bfloat16()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y = ours(sb, pb, mask)
        yc = torch.compile(ours)(sb, pb, mask)
        assert torch.equal(y, yc)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            ours(sb, pb, mask)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            yg = ours(sb, pb, mask)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(y, yg)
