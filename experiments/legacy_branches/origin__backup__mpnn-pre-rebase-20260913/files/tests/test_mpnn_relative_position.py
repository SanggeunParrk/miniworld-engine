"""The relative-position embedding's three backends agree, and two of them reproduce.

The backends differ only in the order their backward adds in, so the tests that matter
are about summation order rather than about the formula: forward must be bit-identical
across all three, the gradients must agree with an FP64 reduction, and the default must
give the same answer twice.
"""

from __future__ import annotations

import pytest
import torch

from miniworld_kernels.kernels.mpnn_relative_position import (
    relative_position_embed,
    relative_position_embed_pytorch,
    relative_position_supported,
)
from miniworld_kernels.modules.mpnn import ProteinMPNN, ProteinMPNNConfig


_BACKENDS = ("off", "index_add", "triton")


def _inputs(rows: int = 200_000, buckets: int = 66, width: int = 16, skew: bool = True):
    generator = torch.Generator(device="cuda").manual_seed(11)
    bucket = torch.randint(
        0, buckets, (rows,), device="cuda", generator=generator, dtype=torch.long
    )
    if skew:
        # The real graph puts a third of every edge in the two end buckets, because
        # the clamp collapses every long-range contact onto them. Accuracy here is a
        # function of exactly that pile-up, so a uniform index would test the easy case.
        heavy = torch.rand(rows, device="cuda", generator=generator) < 0.33
        pick = torch.rand(rows, device="cuda", generator=generator) < 0.5
        bucket = torch.where(heavy, torch.where(pick, 0, buckets - 2), bucket)
    table = torch.randn(
        buckets, width, device="cuda", generator=generator, requires_grad=True
    )
    bias = torch.randn(width, device="cuda", generator=generator, requires_grad=True)
    grad = torch.randn(rows, width, device="cuda", generator=generator)
    return bucket, table, bias, grad


def _gradients(backend: str, bucket, table, bias, grad):
    table = table.detach().clone().requires_grad_(True)
    bias = bias.detach().clone().requires_grad_(True)
    out = relative_position_embed(bucket, table, bias, backend=backend)
    out.backward(grad)
    return out.detach(), table.grad, bias.grad


def test_relative_position_forward_is_identical_across_backends() -> None:
    """Forward is the same gather everywhere; only backward differs."""
    bucket, table, bias, grad = _inputs()
    reference = relative_position_embed_pytorch(bucket, table, bias)
    for backend in _BACKENDS:
        out, _, _ = _gradients(backend, bucket, table, bias, grad)
        assert torch.equal(out, reference), backend


def test_relative_position_gradients_agree_with_an_fp64_reduction() -> None:
    """Compare every backend to the truth, not to each other.

    Two of these reduce in an order PyTorch's own backward does not, so agreeing with
    ``F.embedding`` to the bit is not the bar and would fail for a correct kernel. The
    bar is distance from an FP64 evaluation of the same sum.
    """
    bucket, table, bias, grad = _inputs()
    buckets, width = table.shape
    exact_table = torch.zeros(
        buckets, width, device="cuda", dtype=torch.float64
    ).index_add_(0, bucket, grad.double())
    exact_bias = grad.double().sum(dim=0)

    for backend in _BACKENDS:
        _, grad_table, grad_bias = _gradients(backend, bucket, table, bias, grad)
        table_error = (
            (grad_table.double() - exact_table).abs().max() / exact_table.abs().max()
        ).item()
        bias_error = (
            (grad_bias.double() - exact_bias).abs().max() / exact_bias.abs().max()
        ).item()
        assert table_error < 1e-4, f"{backend} table {table_error:.3e}"
        assert bias_error < 1e-4, f"{backend} bias {bias_error:.3e}"


def test_relative_position_triton_backward_reproduces_bit_for_bit() -> None:
    """The shipped kernel combines its partials in a fixed order.

    Combining them with an atomic would be marginally faster and would make training
    irreproducible, because the order the programs land in is scheduling order. That
    trade is the reason this backend exists rather than ``index_add``, which is faster
    still and does not reproduce, so it is worth a test rather than a comment.
    """
    bucket, table, bias, grad = _inputs()
    _, first_table, first_bias = _gradients("triton", bucket, table, bias, grad)
    for _ in range(3):
        _, again_table, again_bias = _gradients("triton", bucket, table, bias, grad)
        assert torch.equal(first_table, again_table)
        assert torch.equal(first_bias, again_bias)


def test_relative_position_handles_a_bucket_no_edge_selects() -> None:
    """An unused bucket must get an exactly zero gradient, not a stale partial."""
    generator = torch.Generator(device="cuda").manual_seed(3)
    bucket = torch.randint(0, 40, (5_000,), device="cuda", generator=generator).long()
    table = torch.randn(66, 16, device="cuda", generator=generator, requires_grad=True)
    bias = torch.randn(16, device="cuda", generator=generator, requires_grad=True)
    grad = torch.randn(5_000, 16, device="cuda", generator=generator)
    for backend in _BACKENDS:
        _, grad_table, _ = _gradients(backend, bucket, table, bias, grad)
        assert torch.equal(grad_table[40:], torch.zeros_like(grad_table[40:])), backend


def test_relative_position_rejects_contracts_it_cannot_honor() -> None:
    bucket, table, bias, _ = _inputs(rows=1024)
    assert relative_position_supported(bucket, table, bias)
    assert not relative_position_supported(bucket, table, None)
    assert not relative_position_supported(bucket.int(), table, bias)
    assert not relative_position_supported(bucket, table, bias[:-1])
    assert not relative_position_supported(bucket.cpu(), table.cpu(), bias.cpu())
    with pytest.raises(ValueError, match="relative-position"):
        relative_position_embed(bucket.int(), table, bias, backend="triton")


def test_relative_position_backend_is_off_by_default_and_validated() -> None:
    assert ProteinMPNNConfig().relative_position_backend == "off"
    with pytest.raises(ValueError, match="relative_position_backend"):
        ProteinMPNNConfig(relative_position_backend="sorted")  # type: ignore[arg-type]


def test_relative_position_model_matches_the_default_path() -> None:
    """End to end: the backend changes the reduction, not the model."""
    length, batch = 96, 2
    values = None
    outputs = {}
    for backend in _BACKENDS:
        torch.manual_seed(5)
        config = ProteinMPNNConfig(
            node_width=32,
            edge_width=32,
            hidden_width=32,
            encoder_depth=2,
            decoder_depth=2,
            k_neighbors=12,
            coordinate_noise=0.0,
            dropout=0.0,
            relative_position_backend=backend,
        )
        model = ProteinMPNN(config).cuda().train()
        # The output projection ships zero-initialized, so at init every gradient in
        # the model is exactly zero and this comparison would compare nothing. Seeded
        # identically for each backend, so the three models stay bit-identical.
        torch.manual_seed(7)
        with torch.no_grad():
            model.output_projection.weight.normal_(0.0, 0.05)
        if values is None:
            torch.manual_seed(6)
            values = (
                torch.randn(batch, length, 4, 3, device="cuda") * 20.0,
                torch.randint(0, 21, (batch, length), device="cuda"),
                torch.ones(batch, length, device="cuda"),
                torch.arange(length, device="cuda").expand(batch, -1),
                torch.zeros(batch, length, dtype=torch.long, device="cuda"),
                torch.stack(
                    [torch.randperm(length, device="cuda") for _ in range(batch)]
                ),
                (torch.arange(length, device="cuda") // 8)
                .expand(batch, -1)
                .contiguous(),
            )
        logits = model(*values)
        logits.float().square().mean().backward()
        outputs[backend] = (
            logits.detach(),
            model.backbone_features.relative_position.embedding.weight.grad.clone(),
        )

    reference_logits, reference_grad = outputs["off"]
    scale = reference_grad.abs().max().item()
    # Guard the comparison itself.  An all-zero reference divides to nan and reports as
    # a drift failure, which looks like a broken backend and is not one -- the same
    # trap the edge tail's model test hit when a zero-initialised weight made every
    # gradient exactly zero.
    assert scale > 0.0, "the default path produced no relative-position gradient"
    for backend in ("index_add", "triton"):
        logits, grad = outputs[backend]
        assert torch.equal(logits, reference_logits), f"{backend} forward"
        assert torch.isfinite(grad).all(), f"{backend} gradient is not finite"
        drift = (grad - reference_grad).abs().max().item() / scale
        assert drift < 1e-4, f"{backend} gradient drift {drift:.3e}"
