"""Expanded H100 B1/B7 buckets: complete gradients and live graph values."""

import runpy
from pathlib import Path

import pytest
import torch

from miniworld_engine.kernels.trimul_inproj.cuda import h100_single, h100_training
from miniworld_engine.kernels.trimul_inproj.cuda._h100_pack import owned_training_mask

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(autouse=True)
def native_candidate(monkeypatch):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper required")
    if torch.cuda.get_device_properties(0).multi_processor_count != 132:
        pytest.skip("full H100 required")
    for key in (
        "NATIVE_BUCKETS",
        "TRAIN_CUDA",
        "TRAIN_FUSED_PREP",
        "CUDA_AUX",
        "PRESERVE_D128",
    ):
        monkeypatch.setenv("MINIWORLD_TRIMUL_H100_" + key, "1")


helpers = runpy.run_path(
    str(Path(__file__).with_name("test_trimul_h100_b200style_gpu.py"))
)
fixture = helpers["fixture"]
assert_reference = helpers["assert_reference"]


def forward(leaves, mask, ds, direction):
    # Match integration._update: a discrete mask is not differentiable.
    with torch.no_grad():
        pm = owned_training_mask(leaves[0], mask)
    if direction == 0:
        return h100_training.bidirectional_trimul(*leaves, pm, ds)
    return h100_single.single_trimul(direction == 1, *leaves, pm, ds)


@pytest.mark.parametrize("direction", [0, 1, 2])
@pytest.mark.parametrize("master", [False, True])
def test_all_gradients(direction, master):
    leaves, mask, ds, dy = fixture(128, 128, direction, master=master)
    y = forward(leaves, mask, ds, direction)
    grads = torch.autograd.grad(y, leaves, dy)
    assert_reference(leaves, mask, ds, dy, direction, y, grads)


@pytest.mark.parametrize("direction", [0, 1, 2])
def test_extended_boundary(direction):
    leaves, mask, ds, dy = fixture(128, 1024, direction)
    y = forward(leaves, mask, ds, direction)
    grads = torch.autograd.grad(y, leaves, dy)
    assert_reference(leaves, mask, ds, dy, direction, y, grads)


@pytest.mark.parametrize("direction", [0, 1, 2])
def test_graph_live_values(direction):
    leaves, mask, ds, dy = fixture(128, 128, direction, master=True)

    def step():
        y = forward(leaves, mask, ds, direction)
        return y, torch.autograd.grad(y, leaves, dy)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        y, grads = step()
    for trial in range(3):
        with torch.no_grad():
            leaves[0].normal_()
            for w in leaves[1:]:
                w.add_(0.003 * torch.randn_like(w))
            dy.normal_(std=0.1)
            mask.copy_(torch.rand_like(mask, dtype=torch.float32) > 0.25)
            ds.copy_((torch.rand_like(ds) > 0.25) / 0.75)
            if trial == 2:
                mask.zero_()
        graph.replay()
        assert_reference(leaves, mask, ds, dy, direction, y, grads)


@pytest.mark.parametrize("direction", [0, 1, 2])
def test_outstanding_forward_ownership(direction):
    leaves, mask, ds, dy = fixture(128, 128, direction, master=True)
    first = forward(leaves, mask, ds, direction)
    other = forward([leaves[0] * 0.91, *leaves[1:]], ~mask, ds, direction)
    assert other.shape == first.shape
    grads = torch.autograd.grad(first, leaves, dy)
    assert_reference(leaves, mask, ds, dy, direction, first, grads)
