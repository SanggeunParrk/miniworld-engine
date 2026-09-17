"""The inference module must use the shared packed contraction, including under graphs."""
import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from miniworld_engine.kernels.trimul_inproj.triton import bidirectional as wiring
from miniworld_engine.kernels.trimul_inproj.triton.contract import packed_forward
from miniworld_engine.modules import BidirectionalTriangleMultiplication

pytestmark = pytest.mark.gpu


def split_forward(left, right, h):
    return torch.cat((left[:h] @ right[:h].transpose(1, 2),
                      left[h:].transpose(1, 2) @ right[h:]), dim=0)


def setup(length):
    torch.manual_seed(917)
    m = BidirectionalTriangleMultiplication(128, implementation="triton").cuda().bfloat16().eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "ln_" not in name:
                p.normal_(std=128**-.5)
        m.ln_pair.bias.fill_(.2)
        m.ln_out.bias.fill_(.3)
    x = torch.randn(1, length, length, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, length, device="cuda", dtype=torch.bool)
    mask[:, ::3] = False
    return m, x, mask


@pytest.mark.parametrize("mask_kind", ["holes", "none", "empty"])
@torch.no_grad()
def test_inference_uses_packed_with_identical_output(monkeypatch, mask_kind):
    model, x, mask = setup(33)
    if mask_kind == "none":
        mask = None
    elif mask_kind == "empty":
        mask.zero_()
    monkeypatch.setattr(wiring, "packed_forward", split_forward)
    before = model(x, mask)
    calls = []

    def observed(left, right, h):
        calls.append((left.shape, right.shape, h))
        return packed_forward(left, right, h)

    monkeypatch.setattr(wiring, "packed_forward", observed)
    after = model(x, mask)
    assert len(calls) == 1, "The actual inference module did not call packed_forward"
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert torch.isfinite(after).all()


@torch.no_grad()
def test_static_compile_and_cuda_graph():
    model, x, mask = setup(128)
    reference = model(x, mask)
    compiled = torch.compile(model, dynamic=False, fullgraph=True,
                             options={"triton.cudagraphs": False})
    torch.testing.assert_close(compiled(x, mask), reference, rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            compiled(x, mask)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = compiled(x, mask)
    graph.replay()
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    x.add_(.125)
    mask.zero_()
    graph.replay()
    torch.testing.assert_close(output, model(x, mask), rtol=0, atol=0)


@pytest.mark.parametrize("length", [128, 384, 768])
@torch.no_grad()
def test_packed_has_two_bmm_and_no_cat(length):
    x = torch.randn(256, length, length, device="cuda", dtype=torch.bfloat16)
    z = torch.randn_like(x)
    expected = split_forward(x, z, 128)
    packed_forward(x, z, 128)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        actual = packed_forward(x, z, 128)
    events = {e.key: e.count for e in prof.key_averages()}
    assert events.get("aten::bmm", 0) == 2, events
    assert events.get("aten::cat", 0) == 0, events
    assert actual.is_contiguous()
    assert actual.untyped_storage().data_ptr() not in {
        x.untyped_storage().data_ptr(), z.untyped_storage().data_ptr()}
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
