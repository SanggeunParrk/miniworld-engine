"""Native BF16 weights must work without autocast, retaining FP32 norm affine."""

import copy
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine import settings
from miniworld_engine.modules.mpnn import ProteinMPNN, ProteinMPNNConfig
from miniworld_engine.modules.mpnn._functional import MPNNLayerNorm


def test_norm_cast_preserves_affine_and_double():
    norm = MPNNLayerNorm(4)
    with torch.no_grad():
        norm.weight.fill_(1.0001)
    expected = norm.weight.detach().clone()
    norm.bfloat16().bfloat16()
    torch.testing.assert_close(norm.weight, expected, rtol=0, atol=0)
    assert norm.weight.dtype == norm.bias.dtype == torch.float32
    x = torch.randn(3, 4, dtype=torch.bfloat16, requires_grad=True)
    y = norm(x)
    assert y.dtype == torch.bfloat16
    y.float().square().sum().backward()
    assert norm.weight.grad.dtype == torch.float32
    norm.double()
    assert norm.weight.dtype == torch.float64
    assert norm(x.double()).dtype == torch.float64


def _config(backend, feature_backend):
    return ProteinMPNNConfig(
        encoder_depth=3,
        decoder_depth=3,
        node_width=128,
        edge_width=128,
        hidden_width=128,
        k_neighbors=48,
        coordinate_noise=0,
        dropout=0,
        block_linear_min_edges=0,
        feature_backend=feature_backend,
        knn_backend="cdist",
        edge_w1_recompute="off",
        encoder_node_w1_recompute="off",
        decoder_node_w1_recompute="off",
        transition_recompute="off",
        message_backend="pytorch" if backend == "off" else "triton_compute",
        edge_mlp_backend="pytorch" if backend == "off" else "triton_compute",
        edge_norm_backend="pytorch",
        relative_position_backend="off",
        edge_dropout_backend="pytorch",
        edge_tail_backend=backend,
        node_message_backend=backend,
    )


def _inputs(device):
    b, length = 2, 64
    mask = torch.ones(b, length, device=device)
    mask[:, -5:] = 0
    return [
        torch.randn(b, length, 4, 3, device=device),
        torch.randint(21, (b, length), device=device),
        mask,
        torch.arange(length, device=device)[None].repeat(b, 1),
        torch.zeros(b, length, dtype=torch.long, device=device),
        torch.stack([torch.randperm(length, device=device) for _ in range(b)]),
        (torch.arange(length, device=device) // 8)[None].repeat(b, 1),
    ]


def _assert_dtypes(model):
    norm_ids = {
        id(p)
        for m in model.modules()
        if isinstance(m, torch.nn.LayerNorm)
        for p in m.parameters(recurse=False)
    }
    for p in model.parameters():
        expected = torch.float32 if id(p) in norm_ids else torch.bfloat16
        assert p.dtype == expected
        if p.grad is not None:
            assert p.grad.dtype == expected
            assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize("feature_backend", ["pytorch", "memory"])
@pytest.mark.parametrize("block_threshold", [0, 10**9])
def test_pytorch_native_bf16_without_autocast(feature_backend, block_threshold):
    torch.manual_seed(17)
    config = replace(
        _config("off", feature_backend), block_linear_min_edges=block_threshold
    )
    model = ProteinMPNN(config).bfloat16().train()
    _assert_dtypes(model)
    inputs = _inputs("cpu")
    assert not torch.is_autocast_enabled("cpu")
    output = model(*inputs)
    assert output.dtype == torch.bfloat16
    F.cross_entropy(output.float(), inputs[1]).backward()
    assert all(p.grad is not None for p in model.parameters())
    _assert_dtypes(model)


@pytest.fixture
def bounded_tuning():
    # Numerical/compile coverage, not a performance sweep. Preserve resource pruning.
    previous = settings.configure(autotune_miss_cap=24)
    try:
        yield
    finally:
        settings.configure(autotune_miss_cap=previous.autotune_miss_cap)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["triton", "triton_compute"])
@pytest.mark.parametrize("feature_backend", ["pytorch", "memory"])
def test_compiled_native_bf16_matches_pytorch(backend, feature_backend, bounded_tuning):
    torch.manual_seed(17)
    model = ProteinMPNN(_config(backend, feature_backend)).cuda().bfloat16().train()
    torch.nn.init.normal_(model.output_projection.weight, std=128**-0.5)
    reference = ProteinMPNN(_config("off", "pytorch")).cuda().bfloat16().train()
    reference.load_state_dict(copy.deepcopy(model.state_dict()))
    inputs = _inputs("cuda")
    assert not torch.is_autocast_enabled("cuda")
    expected = reference(*inputs)
    F.cross_entropy(expected.float(), inputs[1]).backward()
    compiled = torch.compile(
        model, fullgraph=True, options={"triton.cudagraphs": False}
    )
    actual = compiled(*inputs)
    assert actual.dtype == expected.dtype == torch.bfloat16
    F.cross_entropy(actual.float(), inputs[1]).backward()
    _assert_dtypes(model)
    assert all(p.grad is not None for p in model.parameters())
    relative = lambda a, b: (a.double() - b.double()).norm() / b.double().norm()
    assert relative(actual, expected) < 0.02
    actual_grad = torch.cat([p.grad.flatten().float() for p in model.parameters()])
    reference_grad = torch.cat(
        [p.grad.flatten().float() for p in reference.parameters()]
    )
    assert relative(actual_grad, reference_grad) < 0.02
    assert (
        F.cosine_similarity(actual_grad.double(), reference_grad.double(), dim=0)
        > 0.999
    )


@pytest.mark.gpu
def test_native_bf16_memory_norm_dispatch(bounded_tuning):
    from miniworld_engine.kernels.mpnn_edge_layernorm import edge_layer_norm
    from miniworld_engine.kernels.mpnn_edge_layernorm.interface import _memory_supported

    values = torch.randn(
        64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    norm = MPNNLayerNorm(128).cuda().bfloat16()
    assert _memory_supported(values, norm.weight, norm.bias)
    expected = norm(values)
    actual = edge_layer_norm(values, norm.weight, norm.bias, norm.eps, backend="memory")
    torch.testing.assert_close(actual, expected)
    actual.float().square().mean().backward()
    assert values.grad.dtype == torch.bfloat16
    assert norm.weight.grad.dtype == norm.bias.grad.dtype == torch.float32
    assert torch.isfinite(norm.weight.grad).all()
