"""ProteinMPNN as a module row; dimensions live in registry_module.csv.

``protein_mpnn``: ``modules.mpnn.ProteinMPNN`` (backbone features, encoder, decoder, output projection) with the production backends of
``benchmarks/runners/mpnn_training.py``'s ``compute`` policy (the Triton message / edge MLP / edge tail / node message kernels, the
memory-saving features, the Triton relative position). The ``protein_node`` stream counts residues of ONE packed structure (augmentation 1);
an edge launch is ``length * k_neighbors`` rows. Dims: ``node_width``, ``edge_width``, ``hidden_width``, ``k_neighbors``, ``encoder_depth``,
``decoder_depth``; dropout is the ``p_drop`` option.
"""

from __future__ import annotations

import torch

NAMES = ("protein_mpnn",)


def _config(dims: dict, p_drop: float):
    from miniworld_engine.modules.mpnn import ProteinMPNNConfig

    return ProteinMPNNConfig(
        node_width=dims["node_width"], edge_width=dims["edge_width"], hidden_width=dims["hidden_width"],
        k_neighbors=dims["k_neighbors"], encoder_depth=dims["encoder_depth"], decoder_depth=dims["decoder_depth"],
        coordinate_noise=0.0, dropout=p_drop, block_linear_min_edges=0,
        feature_backend="memory", knn_backend="cdist", edge_norm_backend="pytorch", edge_dropout_backend="pytorch",
        message_backend="triton_compute", edge_mlp_backend="triton_compute", relative_position_backend="triton",
        edge_tail_backend="triton_compute", node_message_backend="triton_compute",
    )


class _BF16Autocast(torch.nn.Module):
    """ProteinMPNN as training runs it: under CUDA BF16 autocast (the Triton message / edge kernels take BF16 projection math), the
    structure as data (no input needs a gradient: the kernels take a non-differentiable mask; the parameters are what trains)."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, *args):
        args = tuple(a.detach() if torch.is_tensor(a) else a for a in args)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(*args)


def _mpnn(dims, p, impl, dt):
    from miniworld_engine.modules.mpnn import ProteinMPNN

    # the module has no implementation switch: its backends are the config's (``impl`` is the registry's ``miniworld``)
    return _BF16Autocast(ProteinMPNN(_config(dims, p)).cuda().to(dt))


def _inputs(b, n, dims, dt, s):
    """``b`` structures of ``n`` residues: backbone [b, n, 4, 3] (a chain walk), sequence, residue mask, residue / chain index, a
    decoding order and the patch index, as ``mpnn_training.make_inputs`` builds them."""
    ca = torch.randn(b, n, 1, 3, device="cuda").cumsum(1)
    return (ca + torch.randn(b, n, 4, 3, device="cuda") * 0.3,
            torch.randint(21, (b, n), device="cuda"),
            torch.ones(b, n, device="cuda"),
            torch.arange(n, device="cuda")[None].repeat(b, 1),
            torch.zeros(b, n, device="cuda", dtype=torch.int64),
            torch.stack([torch.randperm(n, device="cuda") for _ in range(b)]),
            (torch.arange(n, device="cuda") // 8)[None].repeat(b, 1))


def cases():
    from miniworld_engine.autotune.builder import Case, _shapes

    return [Case("protein_mpnn", _mpnn, _inputs, **_shapes("protein_mpnn"))]
