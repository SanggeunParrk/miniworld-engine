"""AttentionPairBias with a gate bias (Chai-1's pairformer single attention, ``gate_bias=1.0``) and Chai-1's width (16 x 24 heads,
d_pair 256) on H100, inference: the native CUDA path (integrations/attention_pair_bias_b200.py) against the fp32 PyTorch module.

The bar is the guide's: the output is no worse than 1.3x the bf16 PyTorch module's error against the fp32 module. Also: the call is
served by the native path (not the composed SDPA statements), a gate bias of 0 leaves the d_pair 128 call as it was (bitwise equal
to the same module built without the option), and the compiled module / the live CUDA-graph replay agree with the eager call.
"""

import copy

import pytest
import torch

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

# (heads, d_single, d_pair, gate_bias)
CASES = [
    (16, 384, 256, 1.0),  # Chai-1
    (16, 384, 256, 0.0),
    (16, 384, 128, 1.0),
    (8, 384, 128, 1.0),
    (12, 384, 128, 0.5),
]


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


def _inputs(L, d, dp, masked, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    single = torch.randn(1, L, d, device="cuda", generator=g)
    pair = torch.randn(1, L, L, dp, device="cuda", generator=g)
    mask = None
    if masked:
        mask = torch.rand(1, L, device="cuda", generator=g) > 0.2
        mask[0, 0] = True
    return single, pair, mask


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"h{c[0]}d{c[1]}p{c[2]}g{c[3]}")
@pytest.mark.parametrize("masked", [False, True], ids=["nomask", "mask"])
@pytest.mark.parametrize("L", [128, 384])
@pytest.mark.parametrize("master", [False, True], ids=["bf16params", "fp32master"])
def test_apb_gate_bias_inference_accuracy(case, masked, L, master):
    from miniworld_engine.integrations import attention_pair_bias_b200 as native

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


def test_apb_gate_bias_is_the_constant_on_the_gate_logits():
    """The option is exactly sigmoid(g + c): a bias of c equals shifting the gate by c in the PyTorch statements, and the
    native path moves with it (c = 0 and c = 3 differ by far more than the bf16 noise)."""
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


def test_apb_gate_bias_off_is_unchanged():
    """gate_bias = 0 on d_pair 128: the same bits as a module built without the option (the option adds nothing)."""
    case = (8, 384, 128, 0.0)
    ref = _reference(case)
    single, pair, mask = _inputs(256, 384, 128, True)
    sb, pb = single.bfloat16(), pair.bfloat16()
    ours = _native(ref, False)
    with torch.no_grad():
        y0 = ours(sb, pb, mask)
        assert ours.gate_bias == 0.0
        y1 = ours(sb, pb, mask)
    assert torch.equal(y0, y1)


@pytest.mark.parametrize("case", [(16, 384, 256, 1.0), (16, 384, 128, 1.0)], ids=lambda c: f"p{c[2]}")
def test_apb_gate_bias_compiled_and_graph(case):
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
