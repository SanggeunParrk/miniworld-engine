"""AF3 atom block: exact window geometry, cross AdaLN, batch biases and live masks."""
import os

import pytest
import torch

from miniworld_engine.modules.local_dit import LocalDiTBlock, windows

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("MINIWORLD_ANTHROPIC_ROOT"),
    reason="needs CUDA and the carried Anthropic release")]


def _case(samples, batch, length, cross):
    torch.manual_seed(433)
    model = LocalDiTBlock(cross_attention=cross, implementation="anthropic").cuda().bfloat16().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim > 1:
                p.normal_(std=p.shape[-1] ** -.5)
            elif name.endswith("weight"):
                p.normal_(mean=1, std=.1)
            else:
                p.normal_(std=.1)
    ref = LocalDiTBlock(cross_attention=cross, implementation="pytorch").cuda().float().eval()
    ref.load_state_dict(model.state_dict())
    rand = lambda *s: torch.randn(*s, device="cuda", dtype=torch.bfloat16)
    inputs = (rand(samples, batch, length, 128), rand(1, batch, length, 128),
              rand(batch, windows(length), 32, 128, 16))
    return model, ref, inputs


def _check(model, ref, inputs, mask):
    got = model(*inputs, mask)
    want = ref(*(x.float() for x in inputs), mask)
    assert torch.isfinite(got).all()
    assert ((got.float() - want).norm() / want.norm().clamp_min(1e-8)).item() < .04
    assert ((got.float() - want).norm() / (want - inputs[0].float()).norm().clamp_min(1e-8)).item() < .08
    assert model.anthropic_selection["window"] == "32x128"
    assert all("fpf_atom" in s for s in model.anthropic_selection["attention"])
    return got


@pytest.mark.parametrize(("samples", "batch", "length"), [(1, 1, 31), (2, 2, 65), (5, 1, 129)])
@pytest.mark.parametrize("cross", [False, True])
@pytest.mark.parametrize("mask_kind", ["none", "partial", "empty_window", "all_masked"])
@torch.no_grad()
def test_matches_local_reference(samples, batch, length, cross, mask_kind):
    torch.backends.cuda.matmul.allow_tf32 = False
    model, ref, inputs = _case(samples, batch, length, cross)
    mask = torch.ones(batch, length, device="cuda", dtype=torch.bool)
    if mask_kind == "partial":
        mask[:, ::7] = False
        if batch > 1:
            mask[1, ::3] = False
    elif mask_kind == "empty_window":
        mask[:, :80] = False
    elif mask_kind == "all_masked":
        mask.zero_()
    if mask_kind == "none":
        mask = None
    _check(model, ref, inputs, mask)
    if mask_kind == "all_masked":
        assert torch.count_nonzero(model.attention_delta(*inputs, mask)).item() == 0


@torch.no_grad()
def test_graph_replay_observes_mask_changes():
    model, ref, inputs = _case(2, 1, 129, True)
    mask = torch.ones(1, 129, device="cuda", dtype=torch.bool)
    for _ in range(3):
        model(*inputs, mask)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = model(*inputs, mask)
    for keep in (0, 1, 49, 129):
        mask.zero_()
        mask[:, :keep] = True
        graph.replay()
        expected = _check(model, ref, inputs, mask)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
