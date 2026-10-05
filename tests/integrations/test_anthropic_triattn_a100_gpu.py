"""Locally rebuilt, unchanged Anthropic sm80 TriAttn: byte gate + module + graph."""
import copy
import os

import pytest
import torch

from miniworld_engine.integrations import anthropic
from miniworld_engine.modules import TriangleAttention

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not os.environ.get("MINIWORLD_ANTHROPIC_ROOT"), reason="needs upstream payload")]


@pytest.fixture(autouse=True)
def a100():
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("requires A100")
    anthropic.configure()
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def test_upstream_byte_gate():
    from opt_core.kernels.triattn import triattn_native

    report = triattn_native.loadcheck()
    assert len(report) == 3
    assert all(row["bitwise"] and row["route"] == "cuda_80" for row in report.values())


@pytest.mark.parametrize("length", [128, 384, 768])
@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize("mask_kind", ["none", "sparse", "dead"])
def test_native_block_reference_and_graph(length, starting, mask_kind):
    torch.manual_seed(814)
    module = TriangleAttention(128, 4, d_hidden=128, starting=starting,
                               implementation="pytorch", anthropic_row="block:triattn_native").cuda().eval()
    with torch.no_grad():
        for param in module.parameters():
            if param.ndim > 1:
                param.normal_(std=param.shape[-1] ** -0.5)
            else:
                param.normal_(mean=1 if param is module.ln_pair.weight else 0, std=.1)
    module = module.bfloat16()
    reference = copy.deepcopy(module).float()
    pair = torch.randn(1, length, length, 128, device="cuda", dtype=torch.bfloat16)
    mask = None if mask_kind == "none" else torch.ones(1, length, device="cuda", dtype=torch.bool)
    if mask_kind == "sparse":
        mask[:, ::7] = False
    elif mask_kind == "dead":
        mask.zero_()
    with torch.no_grad():
        expected = reference(pair.float(), mask)
        actual = anthropic.module_triangle_attention(module, pair, mask)
        assert module.anthropic_selection["core"]["row"] == "triattn_native"
        update = expected - pair.float()
        relative = (actual.float() - expected).norm() / update.norm().clamp_min(1e-8)
        assert torch.isfinite(actual).all()
        assert relative < .04, float(relative)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                anthropic.module_triangle_attention(module, pair, mask)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = anthropic.module_triangle_attention(module, pair, mask)
        graph.replay()
        torch.testing.assert_close(captured, actual, rtol=0, atol=0)
        # A replay must consume new inputs, not an output cached during capture.
        pair.mul_(.5)
        graph.replay()
        fresh = anthropic.module_triangle_attention(module, pair, mask)
        torch.testing.assert_close(captured, fresh, rtol=0, atol=0)
