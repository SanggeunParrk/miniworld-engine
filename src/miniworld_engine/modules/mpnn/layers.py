"""Message-passing layers of ProteinMPNN: packed projections, the node / edge message MLPs, the transition, encoder and decoder layers.

Each is a :class:`~miniworld_engine.modules.primitives.BackendModule`; ``_forward_pytorch`` is the reference.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from miniworld_engine.modules.mpnn._functional import (
    _INT32_MAX,
    KERNEL_WIDTH,
    EdgeLayerNorm,
    MPNNLayerNorm,
    _flat_neighbor_indices,
    bf16_parameter_dtype_unsupported,
    bf16_projection_unsupported,
    gather_neighbors,
    lecun_normal_,
    mpnn_triton_unsupported,
    project_packed_neighbor_inputs,
    projection_input,
)
from miniworld_engine.modules.primitives import BackendModule


class PackedEncoderProjection(BackendModule, nn.Linear):
    """Single packed ``[query, edge, neighbor]`` message projection, applied by blocks (the neighbour block gathered after its GEMM)."""

    backends = ("pytorch",)

    def __init__(self, node_width: int, edge_width: int, hidden_width: int, *, backend: str = "pytorch") -> None:
        self.node_width = node_width
        self.edge_width = edge_width
        super().__init__(2 * node_width + edge_width, hidden_width, bias=True)
        self.set_backend(backend)

    def _forward_pytorch(
        self,
        query_nodes: torch.Tensor,
        edge_features: torch.Tensor,
        neighbor_nodes: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        return project_packed_neighbor_inputs(
            query_nodes, edge_features, neighbor_nodes, neighbor_indices, self, self.node_width, self.edge_width
        )

    def edge_only(self, edge_features: torch.Tensor) -> torch.Tensor:
        """The projection when the node states are zero (encoder layer 0 in training): the edge block plus the bias."""
        start = self.node_width
        stop = start + self.edge_width
        return F.linear(edge_features, self.weight[:, start:stop], self.bias)


class PackedDecoderProjection(BackendModule, nn.Linear):
    """Single packed ``[query, edge, sequence, neighbor]`` projection, applied by blocks under the decoding masks."""

    backends = ("pytorch",)

    def __init__(self, node_width: int, edge_width: int, hidden_width: int, *, backend: str = "pytorch") -> None:
        self.node_width = node_width
        self.edge_width = edge_width
        super().__init__(3 * node_width + edge_width, hidden_width, bias=True)
        self.set_backend(backend)

    def _forward_pytorch(
        self,
        query_nodes: torch.Tensor,
        edge_features: torch.Tensor,
        sequence_features: torch.Tensor,
        encoder_nodes: torch.Tensor,
        neighbor_indices: torch.Tensor,
        future_mask: torch.Tensor,
        past_mask: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.weight
        node_width = self.node_width
        edge_width = self.edge_width

        sequence_start = node_width + edge_width
        sequence_stop = 2 * node_width + edge_width
        sequence_projection = F.linear(sequence_features, weight[:, sequence_start:sequence_stop])
        sequence_neighbors = gather_neighbors(sequence_projection, neighbor_indices)

        output = F.linear(query_nodes, weight[:, :node_width], self.bias).unsqueeze(2)
        edge_projection = F.linear(edge_features, weight[:, node_width:sequence_start])
        edge_mask = (future_mask + past_mask).to(edge_projection.dtype)
        output = output + edge_mask * edge_projection

        past_compute = past_mask.to(sequence_neighbors.dtype)
        output = output + past_compute * sequence_neighbors
        neighbor_weight = weight[:, sequence_stop:]
        current_neighbors = gather_neighbors(F.linear(query_nodes, neighbor_weight), neighbor_indices)
        encoder_neighbors = gather_neighbors(F.linear(encoder_nodes, neighbor_weight), neighbor_indices)
        future_compute = future_mask.to(encoder_neighbors.dtype)
        return output + past_compute * current_neighbors + future_compute * encoder_neighbors


class EdgeDropout(BackendModule, nn.Dropout):
    """``nn.Dropout`` (state-dict compatible) of the encoder edge messages.

    ``"triton"``: the bit-packed training dropout (``kernels/mpnn_edge_dropout``): ATen's native dropout forward, unchanged, with only
    the boolean mask kept for backward packed to one bit per element and the Triton backward. Range: a contiguous fp32 / bf16 CUDA input
    that requires grad, 0 < p < 1, not in place, not under deterministic algorithms, within 32-bit offsets; first-order gradients only.
    Evaluation, no-grad and p = 0 are the identity in every backend, so there is nothing to pack and the call is the reference statement.
    """

    backends = ("pytorch", "triton")

    def __init__(self, p: float = 0.5, inplace: bool = False, *, backend: str = "pytorch") -> None:
        super().__init__(p=p, inplace=inplace)
        self.set_backend(backend)

    def _identity(self) -> bool:
        return not self.training or self.p == 0.0 or not torch.is_grad_enabled()

    def _forward_pytorch(self, input: torch.Tensor) -> torch.Tensor:
        return F.dropout(input, self.p, self.training, self.inplace)

    def _supports_triton(self, input: torch.Tensor) -> str | None:
        if self._identity():
            return None
        if not 0.0 < self.p < 1.0:
            return f"p = {self.p}: the packed mask needs 0 < p < 1"
        if not input.requires_grad:
            return "the packed-mask dropout needs an input that requires grad"
        if self.inplace:
            return "the packed-mask dropout is not in place"
        if torch.are_deterministic_algorithms_enabled():
            return "the packed-mask backward is not deterministic"
        if not input.is_cuda or input.dtype not in {torch.float32, torch.bfloat16} or not input.is_contiguous():
            return f"input {input.dtype} on {input.device}: the packed mask takes a contiguous fp32 / bf16 CUDA tensor"
        if not 0 < input.numel() <= _INT32_MAX - (8 * 256 - 1):
            return "the input must be non-empty and within the kernels' 32-bit offsets"
        return None

    def _forward_triton(self, input: torch.Tensor) -> torch.Tensor:
        if self._identity():
            return self._forward_pytorch(input)
        from miniworld_engine.kernels.mpnn_edge_dropout.triton import edge_dropout_bitpack

        return edge_dropout_bitpack(input, self.p)


class _MessageUpdate(nn.Module):
    """The parameters of a three-projection message MLP plus its residual normalization (``input_projection`` is the packed projection,
    applied by the layer)."""

    def __init__(
        self,
        input_projection: nn.Linear,
        hidden_width: int,
        output_width: int,
        dropout: float,
        norm_class: type[MPNNLayerNorm] = MPNNLayerNorm,
        dropout_class: type[nn.Dropout] = nn.Dropout,
    ) -> None:
        super().__init__()
        self.input_projection = input_projection
        self.hidden_projection = nn.Linear(hidden_width, hidden_width, bias=True)
        self.output_projection = nn.Linear(hidden_width, output_width, bias=True)
        self.norm = norm_class(output_width)
        self.dropout = dropout_class(dropout)
        self.activation = nn.GELU()
        for projection in (self.input_projection, self.hidden_projection, self.output_projection):
            lecun_normal_(projection)
            nn.init.zeros_(projection.bias)


class NodeMessageUpdate(BackendModule, _MessageUpdate):
    """The node message: per edge GELU -> hidden projection -> GELU, masked and summed over the neighbours and divided by the graph width
    ``neighbor_scale`` (ProteinMPNN's rule, not the number of valid edges), then the output projection (its bias scaled by the valid-edge
    fraction), dropout, residual and normalization::

        h = sum_k mask_k gelu(W_h gelu(pre_k) + b_h) / K
        return norm(nodes + dropout(W_o h + (sum_k mask_k / K) b_o))

    ``"triton"``: the first four lines (GELU, the hidden projection, GELU, the masked neighbour sum / K) on the engine's message kernels
    (``kernels/mpnn_message``): eval the one-kernel no-save fusion (``triton_message_hidden_reduce_inference``), train the two-kernel
    forward that keeps the projection plus its Triton backward (``triton_message_hidden_reduce``), or with ``replay_in_backward`` the
    variant that keeps nothing and recomputes the projection once in backward (``triton_message_hidden_reduce_memory``); the output
    projection, dropout, residual and norm as in ``"pytorch"``. Range: bf16 CUDA preactivation ``[..., 48, 128]`` (exactly 48
    neighbours), hidden projection [128, 128] in bf16 (fp32 masters under bf16 autocast), a 0 / 1 edge mask of the preactivation's
    leading shape (handed to the kernels as fp32).
    """

    backends = ("pytorch", "triton")

    def __init__(
        self,
        input_projection: nn.Linear,
        hidden_width: int,
        output_width: int,
        dropout: float,
        *,
        replay_in_backward: bool = False,
        backend: str = "pytorch",
    ) -> None:
        super().__init__(input_projection, hidden_width, output_width, dropout)
        self.replay_in_backward = replay_in_backward
        self.set_backend(backend)

    def _forward_pytorch(
        self, node_states: torch.Tensor, preactivation: torch.Tensor, edge_mask: torch.Tensor, neighbor_scale: int
    ) -> torch.Tensor:
        """``node_states`` ``[B, L, C]``, ``preactivation`` (the packed projection) ``[B, L, K, H]``, ``edge_mask`` ``[B, L, K]``."""
        reduced_hidden = self.reduced_hidden_features(preactivation, edge_mask, neighbor_scale)
        bias_scale = edge_mask.sum(dim=-1, keepdim=True) / neighbor_scale
        return self.apply_reduced_hidden(node_states, reduced_hidden, bias_scale)

    def reduced_hidden_features(self, preactivation: torch.Tensor, edge_mask: torch.Tensor, scale: int) -> torch.Tensor:
        """GELU -> hidden projection -> GELU per edge, masked and summed over the neighbours, divided by ``scale``: ``[..., K, H]`` ->
        ``[..., H]``."""
        hidden = F.gelu(preactivation)
        hidden = F.gelu(F.linear(hidden, self.hidden_projection.weight, self.hidden_projection.bias))
        hidden = hidden * edge_mask.unsqueeze(-1)
        return hidden.sum(dim=-2) / scale

    #: The neighbour count the message kernels are built for.
    _KERNEL_NEIGHBORS = 48

    def _supports_triton(
        self, node_states: torch.Tensor, preactivation: torch.Tensor, edge_mask: torch.Tensor, neighbor_scale: int
    ) -> str | None:
        why = mpnn_triton_unsupported(preactivation)
        if why is not None:
            return why
        if preactivation.dim() < 3 or preactivation.shape[-2] != self._KERNEL_NEIGHBORS:
            return f"preactivation {tuple(preactivation.shape)}: the message kernels are built for [..., {self._KERNEL_NEIGHBORS}, 128]"
        if tuple(edge_mask.shape) != tuple(preactivation.shape[:-1]) or edge_mask.device != preactivation.device:
            return "edge_mask must be [..., K] on the preactivation's device"
        if edge_mask.requires_grad:
            return "the message kernels take a constant (non-differentiable) edge mask"
        proj = self.hidden_projection
        if tuple(proj.weight.shape) != (KERNEL_WIDTH, KERNEL_WIDTH) or proj.bias is None:
            return "the hidden projection must be [128, 128] with a bias"
        return bf16_projection_unsupported(proj.weight, proj.bias)

    def _forward_triton(
        self, node_states: torch.Tensor, preactivation: torch.Tensor, edge_mask: torch.Tensor, neighbor_scale: int
    ) -> torch.Tensor:
        from miniworld_engine.kernels.mpnn_message.triton import (
            triton_message_hidden_reduce,
            triton_message_hidden_reduce_inference,
            triton_message_hidden_reduce_memory,
        )

        pre = preactivation.contiguous()
        mask = edge_mask.to(torch.float32).contiguous()   # a 0 / 1 mask: exact in fp32, the dtype the kernels read
        w, b = self.hidden_projection.weight, self.hidden_projection.bias
        grad = torch.is_grad_enabled() and any(t.requires_grad for t in (pre, w, b))
        if not grad:
            reduce = triton_message_hidden_reduce_inference
        else:
            reduce = triton_message_hidden_reduce_memory if self.replay_in_backward else triton_message_hidden_reduce
        reduced_hidden = reduce(pre, w, b, mask, neighbor_scale)
        bias_scale = edge_mask.sum(dim=-1, keepdim=True) / neighbor_scale
        return self.apply_reduced_hidden(node_states, reduced_hidden, bias_scale)

    def apply_reduced_hidden(
        self,
        node_states: torch.Tensor,
        reduced_hidden: torch.Tensor,
        bias_scale: torch.Tensor | float,
    ) -> torch.Tensor:
        """The output projection of the reduced hidden features (the bias times ``bias_scale``), dropout, residual and normalization."""
        reduced_hidden = projection_input(reduced_hidden, self.output_projection.weight)
        if isinstance(bias_scale, torch.Tensor):
            bias_scale = projection_input(bias_scale, self.output_projection.weight)
        update = F.linear(reduced_hidden, self.output_projection.weight).to(reduced_hidden.dtype)
        if self.output_projection.bias is not None:
            update = update + bias_scale * self.output_projection.bias
        return self.norm(node_states + self.dropout(update))


class EdgeMessageUpdate(BackendModule, _MessageUpdate):
    """The edge message: per edge the two exact-GELU projections, dropout, residual and normalization::

        return norm(edges + dropout(W_o gelu(W_h gelu(pre) + b_h) + b_o))

    ``"triton"``: the update ``W_o gelu(W_h gelu(pre) + b_h) + b_o`` on the engine's edge-MLP kernels (``kernels/mpnn_edge_mlp``): eval the
    one-kernel back-to-back GEMM that saves nothing (``triton_edge_mlp_update``), train the kernel that saves the projection for its
    backward (``triton_edge_mlp_update_compute``), or with ``replay_in_backward`` the back-to-back kernel again, its projection
    recomputed in backward; dropout (:class:`EdgeDropout`), residual and norm (:class:`EdgeLayerNorm`) are the children's own backend.
    Range: bf16 CUDA preactivation ``[..., 128]`` within 32-bit offsets, both projections [128, 128] + bias in bf16 (fp32 masters under
    bf16 autocast).
    """

    backends = ("pytorch", "triton")

    def __init__(
        self,
        input_projection: nn.Linear,
        hidden_width: int,
        output_width: int,
        dropout: float,
        *,
        replay_in_backward: bool = False,
        backend: str = "pytorch",
    ) -> None:
        super().__init__(input_projection, hidden_width, output_width, dropout, EdgeLayerNorm, EdgeDropout)
        self.replay_in_backward = replay_in_backward
        self.set_backend(backend)

    def _forward_pytorch(self, edge_states: torch.Tensor, preactivation: torch.Tensor) -> torch.Tensor:
        """``edge_states`` ``[B, L, K, C]``, ``preactivation`` (the packed projection) ``[B, L, K, H]``."""
        hidden = F.gelu(preactivation)
        hidden = F.gelu(F.linear(hidden, self.hidden_projection.weight, self.hidden_projection.bias))
        update = F.linear(hidden, self.output_projection.weight, self.output_projection.bias)
        return self.norm(edge_states + self.dropout(update))

    def _projections(self) -> tuple[torch.Tensor, ...]:
        h, o = self.hidden_projection, self.output_projection
        return h.weight, h.bias, o.weight, o.bias

    def _supports_triton(self, edge_states: torch.Tensor, preactivation: torch.Tensor) -> str | None:
        why = mpnn_triton_unsupported(preactivation)
        if why is not None:
            return why
        params = self._projections()
        if any(p is None for p in params):
            return "both projections must have a bias"
        if tuple(params[0].shape) != (KERNEL_WIDTH, KERNEL_WIDTH) or tuple(params[2].shape) != (KERNEL_WIDTH, KERNEL_WIDTH):
            return "both projections must be [128, 128]"
        return bf16_projection_unsupported(*params)

    def _forward_triton(self, edge_states: torch.Tensor, preactivation: torch.Tensor) -> torch.Tensor:
        from miniworld_engine.kernels.mpnn_edge_mlp.triton import triton_edge_mlp_update, triton_edge_mlp_update_compute

        pre = preactivation.contiguous()
        params = self._projections()
        grad = torch.is_grad_enabled() and any(t.requires_grad for t in (pre, *params))
        update = (triton_edge_mlp_update_compute if grad and not self.replay_in_backward else triton_edge_mlp_update)(pre, *params)
        return self.norm(edge_states + self.dropout(update))


class ResidualTransition(BackendModule):
    """Position-wise feed-forward transition with residual normalization: ``norm(x + dropout(W_2 gelu(W_1 x + b_1) + b_2))``."""

    backends = ("pytorch",)

    def __init__(self, width: int, dropout: float, *, backend: str = "pytorch") -> None:
        super().__init__()
        self.expand_projection = nn.Linear(width, width * 4, bias=True)
        self.output_projection = nn.Linear(width * 4, width, bias=True)
        self.norm = MPNNLayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        nn.init.kaiming_normal_(self.expand_projection.weight, nonlinearity="relu")
        nn.init.zeros_(self.expand_projection.bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.set_backend(backend)

    def _forward_pytorch(self, states: torch.Tensor) -> torch.Tensor:
        update = self.output_projection(self.activation(self.expand_projection(states)))
        return self.norm(states + self.dropout(update))


class EncoderLayer(BackendModule):
    """Node and edge updates for one MPNN encoder layer.

    ``"triton"`` with ``fused`` (the default): the two edge-sized updates each in one kernel. The node message on the fused node-message
    kernel (``kernels/mpnn_node_message``: the edge block of the packed projection with the two node-side blocks added and the neighbour
    gather folded in, GELU, hidden projection, GELU, masked neighbour sum / K); the edge update on the fused edge-tail kernel
    (``kernels/mpnn_edge_tail``: the same projection, both MLP projections, dropout, residual, LayerNorm). The node-side blocks of the
    packed projections ([B, L, 128]) stay GEMMs; the output projection, residual norm and transition of the node message are the
    reference statements. Train keeps the GEMM outputs for backward (``edge_tail_compute`` / ``triton_node_message_reduce_compute``), or
    with ``replay_in_backward`` recomputes them (``triton_edge_tail_update`` / ``triton_node_message_reduce``); eval keeps nothing.
    The fused edge tail draws its dropout from its own Philox stream and keeps the residual in bf16, the module's ``dropout`` / ``norm``
    contribute only their probability and parameters. Range (``_supports_triton``): bf16 CUDA edge states [B, L, K, 128] with K <= 128
    within 32-bit offsets, bf16 node states (fp32 under bf16 autocast), bf16 projection parameters (fp32 masters under bf16 autocast),
    contiguous int64 neighbour indices and a float mask of the edge shape. Without ``fused`` the layer is the reference composition whose
    children (:class:`NodeMessageUpdate`, :class:`EdgeMessageUpdate`) run their own ``"triton"`` backends.
    """

    backends = ("pytorch", "triton")

    def __init__(
        self,
        node_width: int,
        edge_width: int,
        hidden_width: int,
        dropout: float,
        neighbor_scale: int,
        *,
        fused: bool = True,
        replay_in_backward: bool = False,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        self.neighbor_scale = neighbor_scale
        self.fused = fused
        self.replay_in_backward = replay_in_backward
        self.node_message = NodeMessageUpdate(
            PackedEncoderProjection(node_width, edge_width, hidden_width),
            hidden_width,
            node_width,
            dropout,
            replay_in_backward=replay_in_backward,
        )
        self.node_transition = ResidualTransition(node_width, dropout)
        self.edge_message = EdgeMessageUpdate(
            PackedEncoderProjection(node_width, edge_width, hidden_width),
            hidden_width,
            edge_width,
            dropout,
            replay_in_backward=replay_in_backward,
        )
        self.set_backend(backend)

    def _supports_triton(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        neighbor_indices: torch.Tensor,
        residue_mask: torch.Tensor,
        neighbor_mask: torch.Tensor,
        *,
        zero_nodes: bool = False,
    ) -> str | None:
        if not self.fused:
            return None
        if not (node_states.is_cuda and edge_states.is_cuda and neighbor_indices.is_cuda and neighbor_mask.is_cuda):
            return "the Triton kernels run on CUDA tensors only"
        if edge_states.dtype is not torch.bfloat16 or edge_states.dim() != 4 or edge_states.shape[-1] != KERNEL_WIDTH:
            return f"edge states {edge_states.dtype} {tuple(edge_states.shape)}: the fused kernels take bf16 [B, L, K, {KERNEL_WIDTH}]"
        if not 0 < edge_states.numel() <= _INT32_MAX:
            return "the edge states must be non-empty and within the kernels' 32-bit offsets"
        if edge_states.shape[-2] > 128:
            return f"{edge_states.shape[-2]} neighbours: the fused node message holds one neighbour group (K <= 128) in registers"
        autocast_bf16 = torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16
        if node_states.dim() != 3 or node_states.shape[:2] != edge_states.shape[:2] or node_states.shape[-1] != KERNEL_WIDTH:
            return f"node states {tuple(node_states.shape)}: the fused kernels take [B, L, {KERNEL_WIDTH}] matching the edge states"
        if node_states.dtype is not torch.bfloat16 and not (autocast_bf16 and node_states.dtype is torch.float32):
            return f"node states {node_states.dtype}: the fused kernels take bf16 node projections (fp32 states only under bf16 autocast)"
        if not edge_states.is_contiguous():
            return "the edge states must be contiguous"
        if neighbor_indices.dtype is not torch.long or tuple(neighbor_indices.shape) != tuple(edge_states.shape[:-1]):
            return "the neighbour indices must be int64 of the edge states' leading shape"
        if not neighbor_indices.is_contiguous():
            return "the neighbour indices must be contiguous"
        if tuple(neighbor_mask.shape) != tuple(edge_states.shape[:-1]) or neighbor_mask.dtype not in {torch.float32, torch.bfloat16}:
            return "the neighbour mask must be a float tensor of the edge states' leading shape"
        if not neighbor_mask.is_contiguous() or neighbor_mask.requires_grad:
            return "the neighbour mask must be a contiguous constant"
        node_projection = self.node_message.input_projection
        edge_projection = self.edge_message.input_projection
        hidden, out = self.node_message.hidden_projection, self.edge_message.output_projection
        edge_hidden = self.edge_message.hidden_projection
        norm = self.edge_message.norm
        for width_ok in (
            node_projection.weight.shape == (KERNEL_WIDTH, 3 * KERNEL_WIDTH),
            edge_projection.weight.shape == (KERNEL_WIDTH, 3 * KERNEL_WIDTH),
            hidden.weight.shape == (KERNEL_WIDTH, KERNEL_WIDTH),
            edge_hidden.weight.shape == (KERNEL_WIDTH, KERNEL_WIDTH),
            out.weight.shape == (KERNEL_WIDTH, KERNEL_WIDTH),
        ):
            if not width_ok:
                return "the fused kernels are built for node / edge / hidden width 128"
        parameters = (hidden.weight, hidden.bias, node_projection.weight, edge_projection.weight)
        parameters += (edge_hidden.weight, edge_hidden.bias, out.weight, out.bias, norm.weight, norm.bias)
        if any(p is None for p in parameters):
            return "every projection needs a bias and the norm its affine"
        why = bf16_parameter_dtype_unsupported(
            hidden.weight, hidden.bias, node_projection.weight, edge_projection.weight, edge_hidden.weight, edge_hidden.bias, out.weight, out.bias
        )
        if why is not None:
            return why
        if norm.weight.dtype not in {torch.float32, torch.bfloat16} or norm.bias.dtype != norm.weight.dtype:
            return f"norm affine {norm.weight.dtype} / {norm.bias.dtype}: the fused tail takes an fp32 or bf16 affine of one dtype"
        if not all(p.is_cuda and p.is_contiguous() for p in parameters):
            return "the projection and norm parameters must be contiguous CUDA tensors"
        return None

    @staticmethod
    def _node_side_projection(
        projection: PackedEncoderProjection, node_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The two node-side blocks of a packed projection ([B, L, 128], the bias with the query block) and its edge block."""
        node_width, edge_width = projection.node_width, projection.edge_width
        weight = projection.weight
        query = F.linear(node_states, weight[:, :node_width], projection.bias)
        neighbor = F.linear(node_states, weight[:, node_width + edge_width :])
        return query, neighbor, weight[:, node_width : node_width + edge_width]

    def _forward_triton(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        neighbor_indices: torch.Tensor,
        residue_mask: torch.Tensor,
        neighbor_mask: torch.Tensor,
        *,
        zero_nodes: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.fused:
            return self._forward_pytorch(
                node_states, edge_states, neighbor_indices, residue_mask, neighbor_mask, zero_nodes=zero_nodes
            )
        from miniworld_engine.kernels.mpnn_edge_tail.triton import triton_edge_tail_update
        from miniworld_engine.kernels.mpnn_edge_tail.triton.compute import edge_tail_compute
        from miniworld_engine.kernels.mpnn_node_message.triton import (
            triton_node_message_reduce,
            triton_node_message_reduce_compute,
        )

        batch, length = node_states.shape[:2]
        flat_indices = neighbor_indices if batch == 1 else _flat_neighbor_indices(neighbor_indices, batch, length)

        # The node message. A zero node state needs no special case here: its query block reduces to the bias, its neighbour block to zero.
        query, neighbor, edge_weight = self._node_side_projection(self.node_message.input_projection, node_states)
        hidden = self.node_message.hidden_projection
        node_reduce = triton_node_message_reduce if self.replay_in_backward else triton_node_message_reduce_compute
        reduced_hidden = node_reduce(
            edge_states, query, neighbor, flat_indices, edge_weight, hidden.weight, hidden.bias, neighbor_mask, self.neighbor_scale
        )
        bias_scale = neighbor_mask.sum(dim=-1, keepdim=True) / self.neighbor_scale
        node_states = self.node_message.apply_reduced_hidden(node_states, reduced_hidden, bias_scale)
        node_states = self.node_transition(node_states)
        node_states = residue_mask.unsqueeze(-1).to(node_states.dtype) * node_states

        # The edge update: projection, both MLP projections, dropout, residual and LayerNorm in one kernel.
        query, neighbor, edge_weight = self._node_side_projection(self.edge_message.input_projection, node_states)
        edge_message = self.edge_message
        probability = edge_message.dropout.p if self.training else 0.0
        seed = torch.randint(0, 2**31 - 1, (1,), device=edge_states.device, dtype=torch.int64) if probability > 0.0 else torch.zeros(
            1, device=edge_states.device, dtype=torch.int64
        )
        tail_arguments = (
            edge_weight,
            edge_message.hidden_projection.weight,
            edge_message.hidden_projection.bias,
            edge_message.output_projection.weight,
            edge_message.output_projection.bias,
            edge_message.norm.weight,
            edge_message.norm.bias,
        )
        if self.replay_in_backward:
            edge_states = triton_edge_tail_update(
                edge_states, query, neighbor, flat_indices, *tail_arguments, seed, edge_message.norm.eps, probability
            )
        else:
            width = edge_states.shape[-1]
            edge_states = edge_tail_compute(
                edge_states.reshape(-1, width),
                query.reshape(-1, width),
                neighbor.reshape(-1, width),
                flat_indices.reshape(-1),
                *tail_arguments,
                edge_message.norm.eps,
                probability,
                seed,
            ).reshape(edge_states.shape)
        return node_states, edge_states

    def _forward_pytorch(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        neighbor_indices: torch.Tensor,
        residue_mask: torch.Tensor,
        neighbor_mask: torch.Tensor,
        *,
        zero_nodes: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``zero_nodes``: the incoming node states are known to be zero (encoder layer 0 in training), so the node message takes only
        the edge block of its packed projection plus the bias."""
        input_projection = cast(PackedEncoderProjection, self.node_message.input_projection)
        if zero_nodes:
            preactivation = input_projection.edge_only(edge_states)
        else:
            preactivation = input_projection(node_states, edge_states, node_states, neighbor_indices)
        node_states = self.node_message(node_states, preactivation, neighbor_mask, self.neighbor_scale)
        node_states = self.node_transition(node_states)
        node_states = residue_mask.unsqueeze(-1).to(node_states.dtype) * node_states

        edge_projection = cast(PackedEncoderProjection, self.edge_message.input_projection)
        edge_preactivation = edge_projection(node_states, edge_states, node_states, neighbor_indices)
        edge_states = self.edge_message(edge_states, edge_preactivation)
        return node_states, edge_states


class DecoderLayer(BackendModule):
    """Teacher-forced node update for one MPNN decoder layer."""

    backends = ("pytorch",)

    def __init__(
        self,
        node_width: int,
        edge_width: int,
        hidden_width: int,
        dropout: float,
        neighbor_scale: int,
        *,
        replay_in_backward: bool = False,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        self.neighbor_scale = neighbor_scale
        self.node_message = NodeMessageUpdate(
            PackedDecoderProjection(node_width, edge_width, hidden_width),
            hidden_width,
            node_width,
            dropout,
            replay_in_backward=replay_in_backward,
        )
        self.node_transition = ResidualTransition(node_width, dropout)
        self.set_backend(backend)

    def _forward_pytorch(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        sequence_features: torch.Tensor,
        encoder_nodes: torch.Tensor,
        neighbor_indices: torch.Tensor,
        future_mask: torch.Tensor,
        past_mask: torch.Tensor,
        residue_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_projection = cast(PackedDecoderProjection, self.node_message.input_projection)
        preactivation = input_projection(
            node_states, edge_states, sequence_features, encoder_nodes, neighbor_indices, future_mask, past_mask
        )
        node_states = self.node_message(node_states, preactivation, edge_mask, self.neighbor_scale)
        node_states = self.node_transition(node_states)
        return residue_mask.unsqueeze(-1).to(node_states.dtype) * node_states
