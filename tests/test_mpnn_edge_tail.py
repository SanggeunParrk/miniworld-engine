from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from miniworld_kernels.kernels.mpnn_edge_tail import (
    edge_tail_supported,
    edge_tail_update,
    edge_tail_update_pytorch,
)
from miniworld_kernels.modules.mpnn import ProteinMPNN, ProteinMPNNConfig


_WIDTH = 128
_GRADIENT_NAMES = (
    "edge_states",
    "node_states",
    "packed_weight",
    "packed_bias",
    "hidden_weight",
    "hidden_bias",
    "output_weight",
    "output_bias",
    "norm_weight",
    "norm_bias",
)


def _reproduce_keep_mask(
    seed: torch.Tensor, shape: torch.Size, probability: float
) -> torch.Tensor:
    """Regenerate the kernel's dropout decision for the PyTorch reference.

    The fused kernel keeps no mask: backward re-draws it from the same Philox
    counter.  This helper redraws it a third time so a reference evaluation can be
    compared element by element, which also pins the exact keep condition the two
    kernels have to agree on.
    """
    import triton
    import triton.language as tl

    from miniworld_kernels.kernels.mpnn_edge_tail.triton.main import _PHILOX_ROUNDS

    @triton.jit
    def _keep_mask_kernel(
        seed_ptr,
        mask_ptr,
        elements,
        keep_probability,
        BLOCK: tl.constexpr,
        ROUNDS: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        keep = tl.rand(tl.load(seed_ptr), offsets, n_rounds=ROUNDS) < keep_probability
        tl.store(mask_ptr + offsets, keep, mask=offsets < elements)

    elements = 1
    for size in shape:
        elements *= size
    flat = torch.empty(elements, dtype=torch.bool, device=seed.device)
    block = 1024
    _keep_mask_kernel[(triton.cdiv(elements, block),)](
        seed,
        flat,
        elements,
        1.0 - probability,
        BLOCK=block,
        ROUNDS=_PHILOX_ROUNDS.value,
        num_warps=4,
    )
    return flat.view(shape)


def _leaves(batch: int, length: int, neighbors: int, *, double: bool):
    torch.manual_seed(31)
    generator = torch.Generator(device="cuda").manual_seed(31)

    def normal(*shape, scale: float):
        return torch.randn(*shape, device="cuda", generator=generator) * scale

    edge = normal(batch, length, neighbors, _WIDTH, scale=0.5)
    # The kernel consumes a BF16 edge tensor, so the exact evaluation starts from
    # the same rounded values instead of charging the kernel for input rounding.
    edge = edge.to(torch.bfloat16).float()
    values = [
        edge,
        normal(batch, length, _WIDTH, scale=0.5),
        normal(_WIDTH, 3 * _WIDTH, scale=0.05),
        normal(_WIDTH, scale=0.05),
        normal(_WIDTH, _WIDTH, scale=0.08),
        normal(_WIDTH, scale=0.05),
        normal(_WIDTH, _WIDTH, scale=0.08),
        normal(_WIDTH, scale=0.05),
        torch.rand(_WIDTH, device="cuda", generator=generator) + 0.5,
        normal(_WIDTH, scale=0.05),
    ]
    made = []
    for value in values:
        tensor = value.double() if double else value.clone()
        made.append(tensor.requires_grad_(True))
    return made


def _evaluate(
    made, indices, seed, probability, keep_mask, *, fused: bool,
    backend: str = "triton",
):
    edge, node, packed, packed_bias = made[0], made[1], made[2], made[3]
    hidden_weight, hidden_bias, output_weight, output_bias = made[4:8]
    norm_weight, norm_bias = made[8], made[9]
    query = F.linear(node, packed[:, :_WIDTH], packed_bias)
    neighbor = F.linear(node, packed[:, 2 * _WIDTH :])
    edge_weight = packed[:, _WIDTH : 2 * _WIDTH]
    if fused:
        return edge_tail_update(
            edge.to(torch.bfloat16),
            query,
            neighbor,
            indices,
            edge_weight,
            hidden_weight,
            hidden_bias,
            output_weight,
            output_bias,
            norm_weight,
            norm_bias,
            seed,
            1e-5,
            probability,
            backend,
        )
    return edge_tail_update_pytorch(
        edge,
        query,
        neighbor,
        indices,
        edge_weight,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
        norm_weight,
        norm_bias,
        keep_mask,
        1e-5,
        probability,
    )


def _relative_error(value: torch.Tensor, exact: torch.Tensor) -> float:
    scale = exact.float().abs().mean().clamp_min(1e-12)
    return ((value.float() - exact.float()).abs().mean() / scale).item()


def _compare_against_exact(batch, length, neighbors, probability, backend="triton"):
    """Return per-quantity (fused, pytorch) errors against an FP64 evaluation."""
    indices = torch.randint(
        0, batch * length, (batch, length, neighbors), device="cuda"
    )
    seed = torch.randint(0, 2**31 - 1, (1,), device="cuda", dtype=torch.int64)
    keep_mask = (
        _reproduce_keep_mask(seed, (batch, length, neighbors, _WIDTH), probability)
        if probability > 0
        else None
    )
    fused_leaves = _leaves(batch, length, neighbors, double=False)
    torch_leaves = _leaves(batch, length, neighbors, double=False)
    exact_leaves = _leaves(batch, length, neighbors, double=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        fused = _evaluate(
            fused_leaves, indices, seed, probability, None, fused=True,
            backend=backend,
        )
        reference = _evaluate(
            torch_leaves, indices, seed, probability, keep_mask, fused=False
        )
    exact = _evaluate(exact_leaves, indices, seed, probability, keep_mask, fused=False)

    upstream = torch.randn(
        fused.shape, device="cuda", dtype=torch.float32, generator=None
    )
    fused.float().mul(upstream).sum().backward()
    reference.float().mul(upstream).sum().backward()
    exact.mul(upstream.double()).sum().backward()

    errors = {
        "output": (
            _relative_error(fused, exact),
            _relative_error(reference, exact),
        )
    }
    for name, made, other, truth in zip(
        _GRADIENT_NAMES, fused_leaves, torch_leaves, exact_leaves
    ):
        errors[name] = (
            _relative_error(made.grad, truth.grad),
            _relative_error(other.grad, truth.grad),
        )
    return errors, fused, reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("backend", ["triton", "triton_compute"])
def test_mpnn_edge_tail_is_no_less_accurate_than_the_pytorch_chain(
    backend: str,
) -> None:
    """The fused path must not lose accuracy against separate BF16 operations.

    Both are compared to FP64, so this asserts a property of the kernel rather than
    agreement between two equally approximate paths.  A generous factor is allowed
    because a single quantity can be near zero and dominated by its own rounding.

    Both fused backends are held to it, and the edge weight here is a SLICE of the packed
    projection -- row stride 384, not 128. That is the shape the model actually passes,
    and reading it at the wrong stride is worth 19-59% relative error while every test on
    a contiguous weight still passes. Neither backend gets to inherit the other's result.
    """
    errors, _fused, _reference = _compare_against_exact(2, 256, 48, 0.0, backend)
    failures = {
        name: pair
        for name, pair in errors.items()
        if pair[0] > max(2.0 * pair[1], 5e-3)
    }
    assert not failures, f"fused path is less accurate: {failures} (all: {errors})"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_tail_dropout_replays_the_same_mask_in_backward() -> None:
    """Dropout is not stored, so forward and backward must redraw the same mask.

    If the two draws disagreed, forward would still look plausible while every
    gradient silently referred to a different network.  Comparing against the
    reference driven by an independently regenerated mask catches that.
    """
    errors, fused, reference = _compare_against_exact(2, 256, 48, 0.1)
    zeros = (reference.float() == reference.float()).all()
    assert zeros  # reference produced finite values
    failures = {
        name: pair
        for name, pair in errors.items()
        if pair[0] > max(2.0 * pair[1], 5e-3)
    }
    assert not failures, f"fused path is less accurate: {failures} (all: {errors})"
    torch.testing.assert_close(fused.float(), reference.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_tail_backward_spans_more_than_one_row_chunk() -> None:
    """Exercise the chunk loop, which every other shape here runs exactly once.

    Backward walks rows in 262144-row chunks. At 8192 x 48 = 393216 rows the second
    chunk is a different size from the first, which is the only condition under which a
    per-chunk autotune key can retune mid-backward. A gradient that accumulates across
    chunks while its pass declares ``reset_to_zero`` is silently wrong exactly there,
    and every shape below 262144 rows misses it.
    """
    errors, _fused, _reference = _compare_against_exact(1, 8192, 48, 0.1)
    failures = {
        name: pair
        for name, pair in errors.items()
        if pair[0] > max(2.0 * pair[1], 5e-3)
    }
    assert not failures, f"fused path is less accurate: {failures} (all: {errors})"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_tail_is_deterministic_for_a_fixed_seed() -> None:
    made = _leaves(1, 256, 48, double=False)
    indices = torch.randint(0, 256, (1, 256, 48), device="cuda")
    seed = torch.randint(0, 2**31 - 1, (1,), device="cuda", dtype=torch.int64)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        first = _evaluate(made, indices, seed, 0.1, None, fused=True)
        second = _evaluate(made, indices, seed, 0.1, None, fused=True)
    torch.testing.assert_close(first, second, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_tail_dropout_zeroes_about_the_requested_fraction() -> None:
    made = _leaves(1, 512, 48, double=False)
    indices = torch.randint(0, 512, (1, 512, 48), device="cuda")
    seed = torch.randint(0, 2**31 - 1, (1,), device="cuda", dtype=torch.int64)
    mask = _reproduce_keep_mask(seed, (1, 512, 48, _WIDTH), 0.1)
    dropped = 1.0 - mask.float().mean().item()
    assert 0.09 < dropped < 0.11
    with torch.autocast("cuda", dtype=torch.bfloat16):
        kept = _evaluate(made, indices, seed, 0.0, None, fused=True)
        thinned = _evaluate(made, indices, seed, 0.1, None, fused=True)
    assert not torch.equal(kept, thinned)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_tail_rejects_contracts_it_cannot_honor() -> None:
    made = _leaves(1, 64, 48, double=False)
    indices = torch.randint(0, 64, (1, 64, 48), device="cuda")
    seed = torch.zeros(1, device="cuda", dtype=torch.int64)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        query = F.linear(made[1], made[2][:, :_WIDTH], made[3])
        neighbor = F.linear(made[1], made[2][:, 2 * _WIDTH :])
    arguments = (
        query,
        neighbor,
        indices,
        made[2][:, _WIDTH : 2 * _WIDTH],
        made[4],
        made[5],
        made[6],
        made[7],
        made[8],
        made[9],
    )
    # FP32 edge states: the fused residual stream is BF16 by contract.
    assert not edge_tail_supported(made[0], *arguments)
    with pytest.raises(ValueError, match="requires contiguous CUDA BF16"):
        edge_tail_update(made[0], *arguments, seed, 1e-5, 0.0)
    # A non-contiguous edge tensor would make the flat row indexing wrong.
    strided = made[0].to(torch.bfloat16).transpose(1, 2)
    assert not edge_tail_supported(strided, *arguments)


def test_mpnn_edge_tail_backend_is_off_by_default_and_validated() -> None:
    assert ProteinMPNNConfig().edge_tail_backend == "off"
    with pytest.raises(ValueError, match="edge_tail_backend must be one of"):
        ProteinMPNNConfig(edge_tail_backend="fused")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="subsumes edge_w1_recompute"):
        ProteinMPNNConfig(
            edge_tail_backend="triton",
            edge_mlp_backend="triton_memory",
            edge_w1_recompute="checkpoint",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mpnn_edge_tail_model_matches_the_separate_operation_encoder() -> None:
    """A whole model with the fused tail must stay close to one without it.

    The two differ by the residual dtype, the LayerNorm reduction order and the
    dropout stream, so dropout is disabled here and the remaining gap has to stay
    inside BF16 activation noise rather than merely being finite.
    """

    def build(backend: str) -> ProteinMPNN:
        torch.manual_seed(5)
        model = (
            ProteinMPNN(
                ProteinMPNNConfig(
                    encoder_depth=2,
                    decoder_depth=2,
                    k_neighbors=48,
                    coordinate_noise=0.0,
                    dropout=0.0,
                    message_backend="triton_memory",
                    edge_mlp_backend="triton_memory",
                    edge_norm_backend="memory",
                    feature_backend="recompute",
                    edge_tail_backend=backend,  # type: ignore[arg-type]
                )
            )
            .cuda()
            .train()
        )
        # The output projection ships zero-initialized, which makes every gradient in
        # the model exactly zero and would leave this comparison vacuous. Perturbing
        # it is what lets a gradient reach the encoder at all.
        with torch.no_grad():
            model.output_projection.weight.normal_(0.0, 0.05)
        return model

    length = 256
    torch.manual_seed(7)
    inputs = (
        torch.randn(1, length, 4, 3, device="cuda") * 20.0,
        torch.randint(0, 21, (1, length), device="cuda"),
        torch.ones(1, length, device="cuda"),
        torch.arange(length, device="cuda").unsqueeze(0),
        torch.zeros(1, length, dtype=torch.long, device="cuda"),
        torch.randperm(length, device="cuda").unsqueeze(0),
        (torch.arange(length, device="cuda") // 8).unsqueeze(0).contiguous(),
    )

    outputs = {}
    gradients = {}
    for backend in ("off", "triton"):
        model = build(backend)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(*inputs)
        logits.float().square().mean().backward()
        outputs[backend] = logits.float()
        gradients[backend] = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }

    reference = outputs["off"]
    fused = outputs["triton"]
    scale = reference.abs().mean().clamp_min(1e-6)
    assert ((fused - reference).abs().mean() / scale).item() < 5e-2
    assert set(gradients["off"]) == set(gradients["triton"])
    comparable = [
        name
        for name in gradients["off"]
        if gradients["off"][name].float().abs().mean().item() > 0
    ]
    # Guard the comparison itself: an all-zero gradient set would make every
    # assertion below pass without comparing anything.
    assert len(comparable) > 10, f"only {len(comparable)} nonzero gradients"
    worst = max(
        (
            (
                (gradients["triton"][name].float() - gradients["off"][name].float())
                .abs()
                .mean()
                / gradients["off"][name].float().abs().mean()
            ).item(),
            name,
        )
        for name in comparable
    )
    assert worst[0] < 0.25, f"gradient drift {worst[0]:.3f} at {worst[1]}"


def test_mpnn_edge_tail_shared_gelu_matches_the_unshared_form_bitwise() -> None:
    """The dX pass evaluates GELU and its derivative at the same point.

    It hoists the one ``tl.erf`` they share and feeds both ``_from_erf`` forms, which is
    only sound while those forms agree with ``_gelu``/``_gelu_grad`` to the last bit.
    They are defined in terms of each other so they cannot drift by accident, but a
    later edit that inlines a different expression into either would silently change
    every stored activation, and the shapes here are far too small for such a change to
    show up as a gradient-magnitude failure.  Compare bit patterns, not tolerances.
    """
    import triton
    import triton.language as tl

    from miniworld_kernels.kernels.mpnn_edge_tail.triton.main import (
        _gelu,
        _gelu_erf,
        _gelu_from_erf,
        _gelu_grad,
        _gelu_grad_from_erf,
    )

    @triton.jit
    def _probe(x_ptr, plain_ptr, shared_ptr, dplain_ptr, dshared_ptr, N: tl.constexpr):
        columns = tl.arange(0, N)
        x = tl.load(x_ptr + columns)
        erf_term = _gelu_erf(x)
        tl.store(plain_ptr + columns, _gelu(x))
        tl.store(shared_ptr + columns, _gelu_from_erf(x, erf_term))
        tl.store(dplain_ptr + columns, _gelu_grad(x))
        tl.store(dshared_ptr + columns, _gelu_grad_from_erf(x, erf_term))

    width = 1024
    generator = torch.Generator(device="cuda").manual_seed(7)
    # Span the ranges GELU behaves differently in: the saturated tails, the curved
    # middle, and exact zero.
    values = torch.randn(width, device="cuda", generator=generator) * 8.0
    values[0] = 0.0
    outputs = [torch.empty_like(values) for _ in range(4)]
    _probe[(1,)](values, *outputs, N=width)

    plain, shared, dplain, dshared = outputs
    assert torch.equal(plain.view(torch.int32), shared.view(torch.int32))
    assert torch.equal(dplain.view(torch.int32), dshared.view(torch.int32))
