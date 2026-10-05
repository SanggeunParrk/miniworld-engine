"""Automatic dispatch to the A100 (sm_80) hand-CUDA ProteinMPNN message-side kernels: the hidden-message reduction (``mpnn_message``), the fused encoder node message (``mpnn_node_message``) and the
relative-position embedding backward (``mpnn_relative_position``).

The three kernel families keep their Triton paths and their PyTorch references; their ``interface.py`` asks this module first and, on an A100 (capability exactly 8.0, the engine backend not forced to
Triton, ``MINIWORLD_MPNN_MSG_SM80`` not ``0``) for the shapes / dtypes below, runs the CUDA kernels of ``kernels/<family>/cuda/sm80*`` instead of the Triton ones.  A failed extension build warns once and keeps the
old path.  Every public entry returns what its Triton twin returns and keeps its autograd contract: training is an ``autograd.Function`` whose forward and backward are each one opaque op (``torch.compile``
(fullgraph) and CUDA-graph capture work; no host sync); parameter gradients come back in the parameters' dtype; reductions are deterministic (no atomics anywhere in these paths).

``mpnn_message``  (``message_hidden_reduce``): P [..., 48, 128] bf16, W [128, 128] / b [128] bf16 or fp32 under bf16 autocast, mask [..., 48] fp32 -> reduced [..., 128] fp32.  ONE kernel for the
    forward and the no-grad inference (GELU, projection on tensor cores, bias, GELU, mask, neighbour sum; nothing edge-sized is written).  The backward replays it (it saves only its inputs: both Triton
    policies, ``triton_compute`` and ``triton_memory``, run the same path) and is ONE kernel as well: dP, and dW / db accumulated on the tensor cores in registers (a = bf16(gelu(P)) and dproj never
    reach HBM), in fp32.
``mpnn_node_message`` (``node_message_reduce``, both Triton policies): one forward kernel (two GEMMs, the neighbour gather, both GELUs, the masked sum); the backward replays it in one kernel and writes dedge,
    dquery, dpre, dh and act; dW1e / dW2 are cuBLAS GEMMs over those (fp32), the neighbour gradient a deterministic segmented sum over the edges sorted (stably) by the node they point at.
``mpnn_relative_position`` (``relative_position_embed`` backend ``triton``): the backward's bucket reduction as a one-hot matmul on tensor cores.

Numerics and timings: ``docs/gpus/a100/mpnn/mpnn_msg.md``.
"""

from __future__ import annotations

import os
import warnings

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from miniworld_engine import settings
from miniworld_engine.kernels._compile import opaque

_FAILED: dict[str, bool] = {}
_WIDTH = 128
_NEIGHBORS = 48


def _enabled() -> bool:
    return os.environ.get("MINIWORLD_MPNN_MSG_SM80", "1") != "0" and settings.current().engine_backend != "triton"


def _a100(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (8, 0)


def _load(name: str, module: str) -> bool:
    """Builds (first call) or loads one family's extension; False, with one warning, when the toolchain fails."""
    if _FAILED.get(name):
        return False
    try:
        import importlib

        importlib.import_module(module)._ext()
    except Exception as exc:  # a toolchain problem keeps the old path
        _FAILED[name] = True
        warnings.warn(f"sm_80 MPNN {name} kernels unavailable, keeping the Triton path: {exc!r}", RuntimeWarning, stacklevel=3)
        return False
    return True


@torch.compiler.assume_constant_result
def _message_loads() -> bool:
    """A process-level constant, so ``torch.compile`` evaluates it once at trace time instead of tracing the nvcc lookup / JIT build into the graph."""
    return _load("message", "miniworld_engine.kernels.mpnn_message.cuda.sm80")


@torch.compiler.assume_constant_result
def _node_loads() -> bool:
    return _load("node_message", "miniworld_engine.kernels.mpnn_node_message.cuda.sm80")


@torch.compiler.assume_constant_result
def _relpos_loads() -> bool:
    return _load("relative_position", "miniworld_engine.kernels.mpnn_relative_position.cuda.sm80")


def _node():
    from miniworld_engine.kernels.mpnn_node_message.cuda import sm80

    return sm80


def _msg():
    from miniworld_engine.kernels.mpnn_message.cuda import sm80

    return sm80


def _rp():
    from miniworld_engine.kernels.mpnn_relative_position.cuda import sm80

    return sm80


def _aligned(t: torch.Tensor) -> torch.Tensor:
    """The kernels copy 16-byte chunks: a (rare) view that starts off a 16-byte boundary is copied."""
    return t if t.data_ptr() % 16 == 0 else t.clone()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- hidden message
def serves_message(preactivation: torch.Tensor, backend: str) -> bool:
    """The gate of ``message_hidden_reduce`` after its own contract check passed (bf16 [..., 48, 128] contiguous, [128, 128] / [128] projection, fp32 mask): an A100 and the extension builds."""
    return backend != "pytorch" and _enabled() and _a100(preactivation.device) and _message_loads()


def _message_fwd_fake(preactivation, weight, bias, edge_mask, neighbor_scale):
    """reduced: the leading shape with 128 FP32 channels."""
    return preactivation.new_empty(*preactivation.shape[:-2], _WIDTH, dtype=torch.float32)


@opaque(fake=_message_fwd_fake, name="mpnn_msg_sm80_message_fwd")
def _message_fwd(preactivation: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, edge_mask: torch.Tensor, neighbor_scale: int) -> torch.Tensor:
    """P [..., 48, 128] bf16 -> reduced [..., 128] fp32 (one kernel)."""
    bf = torch.bfloat16
    out = _msg().forward(_aligned(preactivation.reshape(-1, _WIDTH)), _aligned(weight.to(bf).contiguous()), bias.to(bf).contiguous(), _aligned(edge_mask.reshape(-1)), neighbor_scale)
    return out.reshape(*preactivation.shape[:-2], _WIDTH)


def _message_bwd_fake(grad_reduced, preactivation, weight, bias, edge_mask, neighbor_scale):
    """dP like P; dW (128, 128) and db (128,) in FP32 (the caller casts them to the parameters' dtype)."""
    return [torch.empty_like(preactivation), preactivation.new_empty(_WIDTH, _WIDTH, dtype=torch.float32), preactivation.new_empty(_WIDTH, dtype=torch.float32)]


@opaque(fake=_message_bwd_fake, name="mpnn_msg_sm80_message_bwd")
def _message_bwd(grad_reduced: torch.Tensor, preactivation: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, edge_mask: torch.Tensor, neighbor_scale: int) -> list[torch.Tensor]:
    bf = torch.bfloat16
    dp, dw, db = _msg().backward(_aligned(preactivation.reshape(-1, _WIDTH)), _aligned(weight.to(bf).contiguous()), bias.to(bf).contiguous(), _aligned(edge_mask.reshape(-1)),
                                 grad_reduced.reshape(-1, _WIDTH).contiguous(), neighbor_scale)
    return [dp.reshape(preactivation.shape), dw, db]


class _Message(torch.autograd.Function):
    """The autograd boundary: forward and backward are one opaque op each; only the inputs are saved (the backward replays the forward)."""

    @staticmethod
    def forward(ctx, preactivation, weight, bias, edge_mask, neighbor_scale):
        ctx.save_for_backward(preactivation, weight, bias, edge_mask)
        ctx.neighbor_scale = neighbor_scale
        return _message_fwd(preactivation, weight, bias, edge_mask, neighbor_scale)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_reduced):
        preactivation, weight, bias, edge_mask = ctx.saved_tensors
        dp, dw, db = _message_bwd(grad_reduced.contiguous(), preactivation, weight, bias, edge_mask, ctx.neighbor_scale)
        return dp, dw.to(weight.dtype), db.to(bias.dtype), None, None


def message_hidden_reduce(preactivation: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, edge_mask: torch.Tensor, neighbor_scale: int) -> torch.Tensor:
    if not torch.is_grad_enabled():
        return _message_fwd(preactivation, weight, bias, edge_mask, neighbor_scale)
    return _Message.apply(preactivation, weight, bias, edge_mask, neighbor_scale)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- node message
def serves_node(edge_states: torch.Tensor, backend: str) -> bool:
    """The gate of ``node_message_reduce`` after ``node_message_supported`` passed (bf16 [B, T, K, 128] edges with K <= 128, bf16 [B, T, 128] node projections, int64 indices, bf16 or fp32-under-autocast
    parameters, fp32 or bf16 mask): an A100 and the extension builds (``backend`` is one of the Triton policies: both run the same replaying path)."""
    return _enabled() and _a100(edge_states.device) and _node_loads()


def _node_prep(edge_states, query_projection, neighbor_projection, flat_neighbor_indices, edge_weight, hidden_weight, hidden_bias, edge_mask):
    """The kernels' operands: flat rows, contiguous bf16 weights (the edge block of the packed W1 is a strided slice), 16-byte aligned."""
    bf = torch.bfloat16
    k = edge_states.shape[-2]
    return (_aligned(edge_states.reshape(-1, _WIDTH)), _aligned(query_projection.reshape(-1, _WIDTH)), _aligned(neighbor_projection.reshape(-1, _WIDTH)), flat_neighbor_indices.reshape(-1),
            _aligned(edge_weight.to(bf).contiguous()), _aligned(hidden_weight.to(bf).contiguous()), hidden_bias.to(bf).contiguous(), edge_mask.reshape(-1), k)


def _node_fwd_fake(edge_states, query_projection, neighbor_projection, flat_neighbor_indices, edge_weight, hidden_weight, hidden_bias, edge_mask, neighbor_scale):
    """One FP32 row per node: the query projection's leading shape with 128 channels."""
    return query_projection.new_empty(*query_projection.shape[:-1], _WIDTH, dtype=torch.float32)


@opaque(fake=_node_fwd_fake, name="mpnn_msg_sm80_node_fwd")
def _node_fwd(edge_states: torch.Tensor, query_projection: torch.Tensor, neighbor_projection: torch.Tensor, flat_neighbor_indices: torch.Tensor, edge_weight: torch.Tensor,
              hidden_weight: torch.Tensor, hidden_bias: torch.Tensor, edge_mask: torch.Tensor, neighbor_scale: int) -> torch.Tensor:
    """One kernel: the edge projection, both GELUs, the second projection and the masked neighbour sum."""
    e, q, nb, idx, w1, w2, b2, mask, k = _node_prep(edge_states, query_projection, neighbor_projection, flat_neighbor_indices, edge_weight, hidden_weight, hidden_bias, edge_mask)
    out = _node().forward(e, q, nb, idx, w1, w2, b2, mask, k, neighbor_scale)
    return out.reshape(*query_projection.shape[:-1], _WIDTH)


def _node_bwd_fake(grad_reduced, edge_states, query_projection, neighbor_projection, flat_neighbor_indices, edge_weight, hidden_weight, hidden_bias, edge_mask, neighbor_scale):
    """The six gradients in the order the autograd boundary unpacks them: the edge states (their dtype), the query and neighbour projections (theirs), the two (128, 128) weights and the (128,) bias in FP32."""
    f32 = torch.float32
    return [torch.empty_like(edge_states), torch.empty_like(query_projection), torch.empty_like(neighbor_projection), edge_states.new_empty(_WIDTH, _WIDTH, dtype=f32),
            edge_states.new_empty(_WIDTH, _WIDTH, dtype=f32), edge_states.new_empty(_WIDTH, dtype=f32)]


@opaque(fake=_node_bwd_fake, name="mpnn_msg_sm80_node_bwd")
def _node_bwd(grad_reduced: torch.Tensor, edge_states: torch.Tensor, query_projection: torch.Tensor, neighbor_projection: torch.Tensor, flat_neighbor_indices: torch.Tensor, edge_weight: torch.Tensor,
              hidden_weight: torch.Tensor, hidden_bias: torch.Tensor, edge_mask: torch.Tensor, neighbor_scale: int) -> list[torch.Tensor]:
    """Replays the forward (nothing was saved) and differentiates it: see ``kernels/mpnn_node_message/cuda/sm80.py``."""
    e, q, nb, idx, w1, w2, b2, mask, k = _node_prep(edge_states, query_projection, neighbor_projection, flat_neighbor_indices, edge_weight, hidden_weight, hidden_bias, edge_mask)
    dedge, dquery, dnb, dw1, dw2, db2 = _node().backward(e, q, nb, idx, w1, w2, b2, mask, _aligned(grad_reduced.reshape(-1, _WIDTH).contiguous()), k, neighbor_scale)
    # Autocast's Linear rounds a bias gradient to BF16 before the FP32 parameter gradient (the Triton path keeps that boundary). The rounding lives HERE, inside the op: a bf16 round trip in the traced
    # graph would be fused away by Inductor (it keeps fused intermediates in fp32), and compiled would no longer equal eager.
    db2 = db2.to(torch.bfloat16).to(torch.float32)
    return [dedge.reshape(edge_states.shape), dquery.reshape(query_projection.shape), dnb.reshape(neighbor_projection.shape), dw1, dw2, db2]


class _NodeMessage(torch.autograd.Function):
    """The autograd boundary of the fused node message: forward and backward are one opaque op each; only the inputs are saved."""

    @staticmethod
    def forward(ctx, edge_states, query_projection, neighbor_projection, flat_neighbor_indices, edge_weight, hidden_weight, hidden_bias, edge_mask, neighbor_scale):
        ctx.save_for_backward(edge_states, query_projection, neighbor_projection, flat_neighbor_indices, edge_weight, hidden_weight, hidden_bias, edge_mask)
        ctx.neighbor_scale = neighbor_scale
        ctx.dtypes = (query_projection.dtype, neighbor_projection.dtype, edge_weight.dtype, hidden_weight.dtype, hidden_bias.dtype)
        return _node_fwd(edge_states, query_projection, neighbor_projection, flat_neighbor_indices, edge_weight, hidden_weight, hidden_bias, edge_mask, neighbor_scale)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_reduced):
        saved = ctx.saved_tensors
        query_dtype, neighbor_dtype, edge_dtype, hidden_dtype, bias_dtype = ctx.dtypes
        grad_edge, grad_query, grad_neighbor, grad_edge_weight, grad_hidden_weight, grad_hidden_bias = _node_bwd(grad_reduced.contiguous(), *saved, ctx.neighbor_scale)
        return (grad_edge, grad_query.to(query_dtype), grad_neighbor.to(neighbor_dtype), None, grad_edge_weight.to(edge_dtype), grad_hidden_weight.to(hidden_dtype),
                grad_hidden_bias.to(bias_dtype), None, None)


def node_message_reduce(edge_states: torch.Tensor, query_projection: torch.Tensor, neighbor_projection: torch.Tensor, flat_neighbor_indices: torch.Tensor, edge_weight: torch.Tensor,
                        hidden_weight: torch.Tensor, hidden_bias: torch.Tensor, edge_mask: torch.Tensor, neighbor_scale: int) -> torch.Tensor:
    if not torch.is_grad_enabled():
        return _node_fwd(edge_states, query_projection, neighbor_projection, flat_neighbor_indices, edge_weight, hidden_weight, hidden_bias, edge_mask, neighbor_scale)
    return _NodeMessage.apply(edge_states, query_projection, neighbor_projection, flat_neighbor_indices, edge_weight, hidden_weight, hidden_bias, edge_mask, neighbor_scale)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------- relative position
def serves_relpos(bucket: torch.Tensor, table: torch.Tensor, bias: torch.Tensor) -> bool:
    """The gate of ``relative_position_embed`` (backend ``triton``) after its contract check passed: an A100, a 16-channel table of at most 79 buckets, bf16 or fp32."""
    return (_enabled() and _a100(bucket.device) and table.shape[1] == _rp().WIDTH and table.shape[0] <= _rp().MAX_BUCKETS and bucket.numel() > 0 and _relpos_loads())


def _relpos_reduce_fake(grad_output, bucket, buckets):
    """(buckets, width) and (width,) FP32."""
    width = grad_output.shape[-1]
    return [grad_output.new_empty((buckets, width), dtype=torch.float32), grad_output.new_empty((width,), dtype=torch.float32)]


@opaque(fake=_relpos_reduce_fake, name="mpnn_msg_sm80_relpos_reduce")
def _relpos_reduce(grad_output: torch.Tensor, bucket: torch.Tensor, buckets: int) -> list[torch.Tensor]:
    table, bias = _rp().bucket_reduce(_aligned(grad_output.contiguous()), bucket, buckets)
    return [table, bias]


class _RelativePosition(torch.autograd.Function):
    """The same boundary as the Triton path's: the forward is the plain ``F.embedding + bias`` (Dynamo traces it into the graph and fuses it with its neighbours), only the backward is ours."""

    @staticmethod
    def forward(ctx, bucket, table, bias):
        ctx.save_for_backward(bucket)
        ctx.buckets = table.shape[0]
        ctx.dtypes = (table.dtype, bias.dtype)
        return F.embedding(bucket, table) + bias

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (bucket,) = ctx.saved_tensors
        table_dtype, bias_dtype = ctx.dtypes
        grad_table, grad_bias = _relpos_reduce(grad_output, bucket, ctx.buckets)
        return None, grad_table.to(table_dtype), grad_bias.to(bias_dtype)


def relative_position_embed(bucket: torch.Tensor, table: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return _RelativePosition.apply(bucket, table, bias)
