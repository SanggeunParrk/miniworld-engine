from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from miniworld_kernels.modules.mpnn import ProteinMPNN, ProteinMPNNConfig
from miniworld_kernels.modules.mpnn.layers import ResidualTransition


def _transition_pair(
    *,
    width: int,
    dropout: float,
    device: torch.device | str,
) -> tuple[ResidualTransition, ResidualTransition]:
    torch.manual_seed(101)
    reference = ResidualTransition(width, dropout, "off").to(device).train()
    # Production initializes Wout to zero. Make every gradient non-degenerate
    # so replay parity exercises the whole block.
    nn.init.normal_(reference.output_projection.weight, std=0.02)
    nn.init.normal_(reference.output_projection.bias, std=0.02)
    candidate = ResidualTransition(width, dropout, "update").to(device).train()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    return reference, candidate


def _run_transition(
    module: ResidualTransition,
    values: torch.Tensor,
    upstream: torch.Tensor,
    *,
    seed: int,
    autocast: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    module.zero_grad(set_to_none=True)
    states = values.detach().clone().requires_grad_(True)
    if values.is_cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)
    with torch.autocast(values.device.type, dtype=torch.bfloat16, enabled=autocast):
        output = module(states)
    forward_rng = (
        torch.cuda.get_rng_state() if values.is_cuda else torch.get_rng_state()
    )
    gradients = torch.autograd.grad(
        output,
        (states, *module.parameters()),
        upstream,
    )
    backward_rng = (
        torch.cuda.get_rng_state() if values.is_cuda else torch.get_rng_state()
    )
    return output.detach(), gradients, forward_rng, backward_rng


def _model_inputs(
    *,
    length: int,
    k_neighbors: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, ...]:
    backbone = torch.randn(1, length, 4, 3, device=device)
    sequence = torch.randint(0, 21, (1, length), device=device)
    residue_mask = torch.ones(1, length, device=device)
    residue_index = torch.arange(length, device=device).unsqueeze(0)
    chain_index = torch.zeros(1, length, dtype=torch.long, device=device)
    decoding_order = torch.randperm(length, device=device).unsqueeze(0)
    patch_index = (torch.arange(length, device=device) // 8).unsqueeze(0)
    assert k_neighbors <= length
    return (
        backbone,
        sequence,
        residue_mask,
        residue_index,
        chain_index,
        decoding_order,
        patch_index,
    )


def test_transition_update_recompute_is_exact_and_rng_transparent_on_cpu() -> None:
    reference, candidate = _transition_pair(width=8, dropout=0.25, device="cpu")
    values = torch.randn(2, 7, 8)
    upstream = torch.randn_like(values)

    expected = _run_transition(
        reference,
        values,
        upstream,
        seed=991,
        autocast=False,
    )
    actual = _run_transition(
        candidate,
        values,
        upstream,
        seed=991,
        autocast=False,
    )

    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    for actual_gradient, expected_gradient in zip(actual[1], expected[1], strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=0, rtol=0)
    torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)
    torch.testing.assert_close(actual[3], expected[3], atol=0, rtol=0)


def test_transition_update_recompute_preserves_schema_hooks_and_bypasses_no_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, candidate = _transition_pair(width=8, dropout=0.0, device="cpu")
    assert tuple(reference.state_dict()) == tuple(candidate.state_dict())
    candidate.load_state_dict(reference.state_dict(), strict=True)

    calls = {"transition": 0, "norm": 0}
    handles = [
        candidate.register_forward_hook(
            lambda _module, _inputs, _output: calls.__setitem__(
                "transition", calls["transition"] + 1
            )
        ),
        candidate.norm.register_forward_hook(
            lambda _module, _inputs, _output: calls.__setitem__(
                "norm", calls["norm"] + 1
            )
        ),
    ]
    values = torch.randn(3, 8, requires_grad=True)
    candidate(values).sum().backward()
    for handle in handles:
        handle.remove()
    assert calls == {"transition": 1, "norm": 1}

    def forbidden_checkpoint(*_args, **_kwargs):
        raise AssertionError("checkpoint must be bypassed")

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", forbidden_checkpoint)
    with torch.no_grad():
        candidate(values.detach())
    candidate(
        values.detach().requires_grad_(True), allow_recompute=False
    ).sum().backward()


def test_whole_layer_checkpoint_bypasses_transition_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ProteinMPNNConfig(
        node_width=8,
        edge_width=8,
        hidden_width=8,
        encoder_depth=1,
        decoder_depth=1,
        k_neighbors=4,
        coordinate_noise=0.0,
        dropout=0.0,
        transition_recompute="update",
    )
    model = ProteinMPNN(config).train()
    values = _model_inputs(length=8, k_neighbors=4, device="cpu")
    original_checkpoint = torch.utils.checkpoint.checkpoint
    functions: list[object] = []

    def tracked(function, *args, **kwargs):
        functions.append(function)
        return original_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", tracked)
    model(*values, checkpoint_layers=True).square().mean().backward()

    # One encoder and one decoder whole-layer boundary, with no nested
    # ResidualTransition._update boundary in either forward or recomputation.
    assert len(functions) == 2
    assert all(getattr(function, "__name__", "") != "_update" for function in functions)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_transition_update_recompute_drops_wide_activations_from_tape() -> None:
    reference, candidate = _transition_pair(width=128, dropout=0.1, device="cuda")
    values = torch.randn(1, 2048, 128, device="cuda", requires_grad=True)

    def saved_wide_count(module: ResidualTransition) -> int:
        saved: list[torch.Tensor] = []
        module.zero_grad(set_to_none=True)
        states = values.detach().clone().requires_grad_(True)
        with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: saved.append(tensor) or tensor,
            lambda tensor: tensor,
        ):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = module(states)
        output.sum().backward()
        wide_elements = 2048 * 512
        return sum(tensor.numel() == wide_elements for tensor in saved)

    assert saved_wide_count(reference) >= 2
    assert saved_wide_count(candidate) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_transition_update_recompute_fullgraph_rng_and_gradient_parity() -> None:
    reference, candidate = _transition_pair(width=128, dropout=0.1, device="cuda")
    values = torch.randn(2, 257, 128, device="cuda")
    upstream = torch.randn_like(values)

    # The recompute boundary changes the graph, and Inductor's functionalized
    # Philox derives its offsets from the generated kernel's own tiling. Two
    # differently shaped graphs are therefore not required to draw the same mask
    # from the same seed, and comparing them bitwise measured codegen rather
    # than this policy -- it passed alone and failed after other tests had
    # compiled in the same process. `fallback_random` routes both graphs through
    # ATen's RNG so that seeded masks are comparable by construction; the
    # eager exactness of the boundary is covered by the CPU test above.
    torch._dynamo.reset()
    with torch._inductor.config.patch(fallback_random=True):
        compiled_reference = torch.compile(reference, fullgraph=True)
        compiled_candidate = torch.compile(candidate, fullgraph=True)

        # Warm compilation separately, then compare stable compiled graphs.
        _run_transition(compiled_reference, values, upstream, seed=1234, autocast=True)
        _run_transition(compiled_candidate, values, upstream, seed=1234, autocast=True)
        expected = _run_transition(
            compiled_reference, values, upstream, seed=4321, autocast=True
        )
        actual = _run_transition(
            compiled_candidate, values, upstream, seed=4321, autocast=True
        )

    # Forward and both RNG states must match exactly: dropout stays outside the
    # boundary, so the policy may not perturb the mask or the generator.
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)
    torch.testing.assert_close(actual[3], expected[3], atol=0, rtol=0)
    # Gradients are exact in eager (see the CPU test above) but not bitwise under
    # compilation: the partitioner replays `_update` inside the backward graph and
    # Inductor may fuse that replay differently than the forward it reproduces.
    # The observed spread is one FP32 ULP -- 4.8e-7 absolute on an A6000 -- while
    # an actually wrong replay would be orders of magnitude away.
    for actual_gradient, expected_gradient in zip(actual[1], expected[1], strict=True):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            atol=1e-6,
            rtol=5e-3,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_transition_update_with_outer_checkpoint_is_fullgraph_compilable() -> None:
    config = ProteinMPNNConfig(
        node_width=128,
        edge_width=128,
        hidden_width=128,
        encoder_depth=1,
        decoder_depth=1,
        k_neighbors=8,
        coordinate_noise=0.0,
        dropout=0.1,
        transition_recompute="update",
    )
    model = ProteinMPNN(config).cuda().train()
    values = _model_inputs(length=16, k_neighbors=8, device="cuda")

    def forward(*args: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(*args, checkpoint_layers=True)

    compiled = torch.compile(forward, fullgraph=True)
    compiled(*values).float().square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
