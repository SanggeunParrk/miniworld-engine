"""Contract tests for the frozen naive ProteinMPNN reference."""

from __future__ import annotations

import torch

from miniworld_kernels.modules.mpnn import NaiveProteinMPNN
from miniworld_kernels.modules.mpnn.naive import gather_edges, gather_nodes


def _inputs(length: int = 8) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(7)
    xyz = torch.randn(1, length, 4, 3, generator=generator)
    seq = torch.randint(0, 21, (1, length), generator=generator)
    mask = torch.ones(1, length)
    residue_idx = torch.arange(length).unsqueeze(0)
    chain_idx = torch.zeros(1, length, dtype=torch.long)
    decoding_order = torch.arange(length).flip(0).unsqueeze(0)
    patch_index = torch.arange(length).unsqueeze(0)
    len_tensor = torch.tensor([length])
    return (
        xyz,
        seq,
        mask,
        residue_idx,
        chain_idx,
        decoding_order,
        patch_index,
        mask.clone(),
        len_tensor,
    )


def test_gather_helpers_preserve_source_layout() -> None:
    nodes = torch.arange(2 * 4 * 3).view(2, 4, 3)
    neighbor_idx = torch.tensor(
        [[[0, 2], [3, 1], [1, 0], [2, 3]], [[3, 1], [0, 2], [2, 3], [1, 0]]]
    )
    gathered = gather_nodes(nodes, neighbor_idx)
    expected = torch.stack(
        [
            torch.stack([nodes[b, neighbor_idx[b, i]] for i in range(4)])
            for b in range(2)
        ]
    )
    torch.testing.assert_close(gathered, expected)

    edges = torch.arange(2 * 4 * 4 * 2).view(2, 4, 4, 2)
    gathered_edges = gather_edges(edges, neighbor_idx)
    expected_edges = torch.stack(
        [
            torch.stack([edges[b, i, neighbor_idx[b, i]] for i in range(4)])
            for b in range(2)
        ]
    )
    torch.testing.assert_close(gathered_edges, expected_edges)


def test_decoding_mask_uses_patch_order_not_residue_index() -> None:
    edge_idx = torch.tensor([[[0, 1, 2], [1, 0, 2], [2, 0, 1]]])
    valid = torch.ones(1, 3)
    # Decode residues in order 2 -> 0 -> 1.  Patch ids are indexed by decode step.
    decoding_order = torch.tensor([[2, 0, 1]])
    patch_index = torch.tensor([[0, 1, 2]])
    mask_fw, mask_bw = NaiveProteinMPNN.get_decoding_masks(
        edge_idx, valid, decoding_order, patch_index
    )

    # Residue 0 is the second decoded residue, so only residue 2 is visible.
    torch.testing.assert_close(mask_bw[0, 0, :, 0], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(mask_fw + mask_bw, torch.ones_like(mask_fw))


def test_forward_shape_log_probs_and_gradients() -> None:
    model = NaiveProteinMPNN(
        node_features=16,
        edge_features=16,
        hidden_dim=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        k_neighbors=4,
        augment_trans=0,
        augment_rot=0,
        dropout=0,
    ).eval()
    # The upstream output projection is zero-initialized. Lift it so this test
    # exercises gradients through the complete model rather than a zero oracle.
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        model.W_out.weight.copy_(
            torch.randn(model.W_out.weight.shape, generator=generator) * 0.02
        )

    inputs = list(_inputs())
    inputs[0].requires_grad_(True)
    logits = model(*inputs)
    assert logits.shape == (1, 21, 8)
    logits.square().mean().backward()
    assert inputs[0].grad is not None
    assert torch.isfinite(inputs[0].grad).all()
    assert model.encoder_layers[0].W1.weight.grad is not None

    log_probs = model(*_inputs(), return_log_prob=True)
    assert log_probs.shape == (1, 8, 21)
    torch.testing.assert_close(
        log_probs.exp().sum(dim=-1), torch.ones(1, 8), atol=1e-6, rtol=1e-6
    )


def test_source_state_dict_names_are_preserved() -> None:
    model = NaiveProteinMPNN(
        node_features=8,
        edge_features=8,
        hidden_dim=8,
        num_encoder_layers=1,
        num_decoder_layers=1,
        k_neighbors=4,
    )
    keys = set(model.state_dict())
    assert {
        "features.embeddings.linear.weight",
        "features.edge_embedding.weight",
        "encoder_layers.0.W1.weight",
        "encoder_layers.0.W13.weight",
        "decoder_layers.0.W1.weight",
        "W_e.weight",
        "W_s.weight",
        "W_out.weight",
    } <= keys


def test_frozen_reference_has_committed_numerical_anchor() -> None:
    model = NaiveProteinMPNN(
        node_features=8,
        edge_features=8,
        hidden_dim=8,
        num_encoder_layers=1,
        num_decoder_layers=1,
        k_neighbors=4,
        augment_trans=0,
        augment_rot=0,
        dropout=0,
    ).eval()
    parameter_generator = torch.Generator().manual_seed(314159)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=parameter_generator) * 0.2
            )

    length = 6
    input_generator = torch.Generator().manual_seed(2718)
    xyz = torch.randn(1, length, 4, 3, generator=input_generator, requires_grad=True)
    seq = torch.randint(0, 21, (1, length), generator=input_generator)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0, 1.0]])
    logits = model(
        xyz,
        seq,
        mask,
        torch.arange(length).unsqueeze(0),
        torch.tensor([[0, 0, 0, 1, 1, 1]]),
        torch.tensor([[4, 1, 5, 0, 2, 3]]),
        torch.tensor([[0, 0, 1, 2, 3, 4]]),
        mask,
        torch.tensor([3, 3]),
    )
    loss_weights = torch.linspace(-1, 1, logits.numel()).reshape_as(logits)
    (logits * loss_weights).sum().backward()

    expected_logits = torch.tensor(
        [
            [-0.0123053342, -0.0113388300, -0.0120705366],
            [0.2443781048, 0.2440996766, 0.2441098392],
            [-0.1009539738, -0.0999285057, -0.1007047445],
            [0.0714179054, 0.0719053149, 0.0714849979],
            [0.1689914763, 0.1688448638, 0.1687808633],
        ]
    )
    expected_ca_grad = torch.tensor(
        [
            [1.3227851596e-3, 1.7769646365e-3, 3.0480441637e-5],
            [-6.3107814640e-4, 3.7325796438e-4, -9.0895709582e-4],
            [1.1314565782e-3, -1.4132412616e-5, -6.6696631256e-4],
        ]
    )
    torch.testing.assert_close(logits[0, :5, :3], expected_logits, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        xyz.grad[0, :3, 1], expected_ca_grad, atol=2e-7, rtol=2e-5
    )
