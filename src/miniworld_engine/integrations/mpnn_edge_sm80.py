"""Automatic dispatch of the ProteinMPNN encoder edge side to the A100 (sm_80) hand-CUDA kernels (``kernels/mpnn_edge_tail/cuda``): the fused edge tail, the edge MLP, the
edge LayerNorm's compressed-save backward and the edge dropout's bit-packed mask.

The family interfaces (``kernels/mpnn_edge_*/interface.py``) ask this module whether a call is served: capability exactly 8.0, ``MINIWORLD_MPNN_EDGE_SM80`` not 0, the engine
backend not forced to Triton, the Triton contract of the family met (``edge_tail_supported`` ...) and the extension loadable (a failed build warns once and keeps the Triton / PyTorch
path).  On a served call the Triton policies ``triton`` / ``triton_compute`` / ``triton_memory`` / ``memory`` / ``bitpack`` run the CUDA kernels instead (``backend="cuda"`` names
them explicitly; ``MINIWORLD_MPNN_EDGE_SM80=0`` or ``engine_backend="triton"`` give the Triton kernels back).

Edge tail contract: bf16 ``[B, T, K, 128]`` edge states, ``[B, T, 128]`` query / neighbour projections, int64 flat neighbour indices, an edge block that is a slice of the packed
projection (any row stride that is a multiple of 8), hidden / output weights and biases bf16 or fp32 (autocast), LayerNorm affine bf16 or fp32, any K, dropout p in [0, 1).
Training (``save``) keeps the GELU outputs ``act1`` / ``act2`` (bf16), their derivatives ``d1`` / ``d2`` (fp16), ``values`` (bf16), the row statistics and the packed dropout decisions for the backward
("compute" policy, ``triton_compute`` / ``cuda``); the Triton ``triton`` policy (recompute) keeps only the inputs and replays the saving forward at the start of the backward.  Numerics and timings:
``docs/gpus/a100/mpnn/mpnn_edge.md``.
"""

from __future__ import annotations

import os
import warnings

import torch

from miniworld_engine import settings
from miniworld_engine.kernels._compile import opaque

ENV = "MINIWORLD_MPNN_EDGE_SM80"
_FAILED = False
_WIDTH = 128


def wanted() -> bool:
    """The switch and the engine backend; everything else about a call is `serves_*`."""
    return os.environ.get(ENV, "1") != "0" and settings.current().engine_backend != "triton"


def _ext():
    from miniworld_engine.kernels.mpnn_edge_tail.cuda import sm80

    return sm80.ext()


@torch.compiler.assume_constant_result
def _loads() -> bool:
    """Builds (first call) or loads the sm_80 extension; False, with one warning, when the toolchain fails (the callers then keep the Triton / PyTorch path).

    A process-level constant, so ``torch.compile`` evaluates it once at trace time instead of tracing the nvcc lookup / JIT build into the graph."""
    global _FAILED
    if _FAILED:
        return False
    try:
        _ext()
    except Exception as exc:  # a toolchain problem keeps the old path
        _FAILED = True
        warnings.warn(f"sm_80 MPNN edge kernels unavailable, keeping the Triton / PyTorch path: {exc!r}", RuntimeWarning, stacklevel=2)
        return False
    return True


def on_a100(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (8, 0)


def _aligned(t: torch.Tensor) -> torch.Tensor:
    """The tensor with a 16-byte aligned start (the row kernels load 16 bytes at a time; a view of a larger tensor may start anywhere).  Called inside the opaque ops only."""
    return t if t.data_ptr() % 16 == 0 else t.clone()


def _rows16(weight: torch.Tensor) -> torch.Tensor:
    """The weight with 16-byte aligned rows (the pack kernel loads 16 bytes at a time): a slice of the packed projection starts 256 B in with a 768 B row stride and is returned as is,
    anything else is copied.  Called inside the opaque ops only (`storage_offset` is not traceable)."""
    return weight if weight.storage_offset() % 8 == 0 and weight.stride(0) % 8 == 0 and weight.stride(1) == 1 else weight.clone()


# =================================================================================================================================== edge tail
def tail_serves(edge_states: torch.Tensor, edge_weight: torch.Tensor, hidden_weight: torch.Tensor, output_weight: torch.Tensor) -> bool:
    """Whether the CUDA edge tail runs this call (the caller has already checked ``edge_tail_supported``: bf16 activations, width 128, contiguity, parameter dtypes)."""
    if not wanted() or not on_a100(edge_states.device):
        return False
    return edge_states.shape[-2] >= 1 and _loads()


def _fwd_fake(edge, query, nbr, idx, w1, w2, w3, b2, b3, gamma, beta, seed, k, eps, p, save, row_base):
    """out, then (with ``save``) act1, d1, act2, d2, values, the row statistics and (with dropout) the packed keep words; empty tensors otherwise."""
    rows = edge.shape[0]
    n = rows if save else 0
    e, f32 = edge.new_empty, torch.float32
    return [torch.empty_like(edge), e((n, _WIDTH)), e((n, _WIDTH), dtype=torch.float16), e((n, _WIDTH)), e((n, _WIDTH), dtype=torch.float16), e((n, _WIDTH)),
            e((n, 2), dtype=f32), e((rows if save and p > 0.0 else 0, 4), dtype=torch.int32)]


@opaque(fake=_fwd_fake, name="mpnn_edge_tail_sm80_fwd")
def _fwd(edge: torch.Tensor, query: torch.Tensor, nbr: torch.Tensor, idx: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor, b2: torch.Tensor,
         b3: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, seed: torch.Tensor, k: int, eps: float, p: float, save: bool, row_base: int) -> list[torch.Tensor]:
    """The fused edge tail on flat ``[rows, 128]`` operands (weights packed per call: the pack is one 24-block launch); ``row_base`` is the index of the first row in the whole tensor (a chunk of a replayed
    forward draws the dropout decisions of its rows)."""
    ext = _ext()
    img, tab = ext.pack(False, False, False, _rows16(w1), _rows16(w2), _rows16(w3), b2, b3, gamma, beta)
    return list(ext.tail_fwd(edge, query, nbr, idx, img, tab, seed, k, eps, p, save, row_base))


def _bwd_fake(go, edge, idx, values, stats, keep, act1, d1, act2, d2, w1, w2, w3, b2, b3, gamma, beta, nodes, k, p):
    """grad_edge, grad_query, grad_neighbor, dW1e, dW2, dW3, db2, db3, dgamma, dbeta -- the parameter gradients in the parameters' dtypes."""
    rows = edge.shape[0]
    e = edge.new_empty
    return [torch.empty_like(edge), e((rows // k, _WIDTH)), e((nodes, _WIDTH)),
            torch.empty((_WIDTH, _WIDTH), dtype=w1.dtype, device=w1.device), torch.empty((_WIDTH, _WIDTH), dtype=w2.dtype, device=w1.device),
            torch.empty((_WIDTH, _WIDTH), dtype=w3.dtype, device=w1.device), torch.empty((_WIDTH,), dtype=b2.dtype, device=w1.device),
            torch.empty((_WIDTH,), dtype=b3.dtype, device=w1.device), torch.empty((_WIDTH,), dtype=gamma.dtype, device=w1.device),
            torch.empty((_WIDTH,), dtype=beta.dtype, device=w1.device)]


@opaque(fake=_bwd_fake, name="mpnn_edge_tail_sm80_bwd")
def _bwd(go: torch.Tensor, edge: torch.Tensor, idx: torch.Tensor, values: torch.Tensor, stats: torch.Tensor, keep: torch.Tensor, act1: torch.Tensor, d1: torch.Tensor,
         act2: torch.Tensor, d2: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor, b2: torch.Tensor, b3: torch.Tensor, gamma: torch.Tensor,
         beta: torch.Tensor, nodes: int, k: int, p: float) -> list[torch.Tensor]:
    """LayerNorm / dropout backward + dX x 3 (one kernel), the weight gradients, the two reductions of the first product's gradient, the fixed-order sums."""
    ext = _ext()
    img, tab = ext.pack(True, True, True, _rows16(w3), _rows16(w2), _rows16(w1), b2, b3, gamma, beta)       # backward image order: dX3, dX2, dX1
    return list(ext.tail_bwd(go, edge, idx, values, stats, keep, act1, d1, act2, d2, img, tab, k, nodes, p, w1.dtype == torch.float32, b2.dtype == torch.float32,
                             gamma.dtype == torch.float32))


#: rows of the slice of the edge tensor a replayed forward + backward works on at a time (the recompute policy): the saved activations of a slice are 1.3 KB per row
_RECOMPUTE_CHUNK_ROWS = 1 << 19


def _replay_bwd_fake(go, edge, query, nbr, idx, w1, w2, w3, b2, b3, gamma, beta, seed, nodes, k, eps, p):
    """The outputs of ``_bwd_fake``: the replay is a backward whose saved operands are rebuilt inside it."""
    return _bwd_fake(go, edge, idx, edge, edge, edge, edge, edge, edge, edge, w1, w2, w3, b2, b3, gamma, beta, nodes, k, p)


@opaque(fake=_replay_bwd_fake, name="mpnn_edge_tail_sm80_bwd_replay")
def _replay_bwd(go: torch.Tensor, edge: torch.Tensor, query: torch.Tensor, nbr: torch.Tensor, idx: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor,
                b2: torch.Tensor, b3: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, seed: torch.Tensor, nodes: int, k: int, eps: float, p: float) -> list[torch.Tensor]:
    """The recompute policy's backward: the saving forward is replayed and consumed slice by slice (whole nodes: a multiple of K rows), so the saved activations of one slice exist at a time and the
    peak memory of the op does not grow with the edge tensor.  The replay is the same kernel with the same seed (the dropout decisions are keyed by the row of the whole tensor), hence the same
    activations bit for bit.  The neighbour gradient and the parameter gradients are summed over the slices in fp32 and rounded once; a tensor that fits one slice takes the plain path."""
    rows = edge.shape[0]
    step = max(k, _RECOMPUTE_CHUNK_ROWS // k * k)
    if step >= rows:
        _, act1, d1, act2, d2, values, stats, keep = _fwd(edge, query, nbr, idx, w1, w2, w3, b2, b3, gamma, beta, seed, k, eps, p, True, 0)
        return _bwd(go, edge, idx, values, stats, keep, act1, d1, act2, d2, w1, w2, w3, b2, b3, gamma, beta, nodes, k, p)
    gedge, gquery = torch.empty_like(edge), torch.empty_like(query)
    gnbr = torch.zeros((nodes, _WIDTH), dtype=torch.float32, device=edge.device)
    summed: list[torch.Tensor | None] = [None] * 7                 # dW1e, dW2, dW3, db2, db3, dgamma, dbeta
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        n0, n1 = r0 // k, r1 // k
        e, ix = edge[r0:r1], idx[r0:r1]
        _, act1, d1, act2, d2, values, stats, keep = _fwd(e, query[n0:n1], nbr, ix, w1, w2, w3, b2, b3, gamma, beta, seed, k, eps, p, True, r0)
        ge, gq, gn, *pg = _bwd(go[r0:r1], e, ix, values, stats, keep, act1, d1, act2, d2, w1, w2, w3, b2, b3, gamma, beta, nodes, k, p)
        gedge[r0:r1] = ge
        gquery[n0:n1] = gq
        gnbr.add_(gn)                                                    # fp32 += bf16 in one pass
        summed = [g.float() if s is None else s + g.float() for s, g in zip(summed, pg, strict=True)]
    dtypes = (w1.dtype, w2.dtype, w3.dtype, b2.dtype, b3.dtype, gamma.dtype, beta.dtype)
    return [gedge, gquery, gnbr.to(edge.dtype), *[s.to(dt) for s, dt in zip(summed, dtypes, strict=True)]]


class _TailUpdate(torch.autograd.Function):
    """The autograd boundary: the forward saves the backward's operands only when a gradient is wanted (``save``), the backward is one opaque op.

    ``recompute`` (the Triton ``triton`` policy, the memory-lean one): the forward is the inference kernel and keeps nothing but its inputs; the backward replays the saving forward
    (same kernel, same seed: the same activations and dropout decisions, bit for bit) slice by slice and runs the same backward on each (``_replay_bwd``, one opaque op)."""

    @staticmethod
    def forward(ctx, edge, query, nbr, idx, w1, w2, b2, w3, b3, gamma, beta, seed, eps, p, save, k, recompute):
        out, act1, d1, act2, d2, values, stats, keep = _fwd(edge, query, nbr, idx, w1, w2, w3, b2, b3, gamma, beta, seed, k, eps, p, save and not recompute, 0)
        if save:
            if recompute:
                ctx.save_for_backward(edge, query, nbr, idx, w1, w2, w3, b2, b3, gamma, beta, seed)
            else:
                ctx.save_for_backward(edge, idx, w1, w2, w3, b2, b3, gamma, beta, values, stats, keep, act1, d1, act2, d2)
            ctx.k, ctx.p, ctx.eps, ctx.nodes, ctx.recompute = k, p, eps, nbr.shape[0], recompute
            ctx.dtypes = (query.dtype, nbr.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.recompute:
            edge, query, nbr, idx, w1, w2, w3, b2, b3, gamma, beta, seed = ctx.saved_tensors
            gedge, gquery, gnbr, dw1, dw2, dw3, db2, db3, dgamma, dbeta = _replay_bwd(grad_out.contiguous(), edge, query, nbr, idx, w1, w2, w3, b2, b3, gamma, beta, seed,
                                                                                      ctx.nodes, ctx.k, ctx.eps, ctx.p)
        else:
            edge, idx, w1, w2, w3, b2, b3, gamma, beta, values, stats, keep, act1, d1, act2, d2 = ctx.saved_tensors
            gedge, gquery, gnbr, dw1, dw2, dw3, db2, db3, dgamma, dbeta = _bwd(grad_out.contiguous(), edge, idx, values, stats, keep, act1, d1, act2, d2, w1, w2, w3, b2, b3,
                                                                               gamma, beta, ctx.nodes, ctx.k, ctx.p)
        qdt, ndt = ctx.dtypes
        return gedge, gquery.to(qdt), gnbr.to(ndt), None, dw1, dw2, db2, dw3, db3, dgamma, dbeta, None, None, None, None, None, None


def edge_tail_update(edge_states: torch.Tensor, query_projection: torch.Tensor, neighbor_projection: torch.Tensor, flat_neighbor_indices: torch.Tensor,
                     edge_weight: torch.Tensor, hidden_weight: torch.Tensor, hidden_bias: torch.Tensor, output_weight: torch.Tensor, output_bias: torch.Tensor,
                     norm_weight: torch.Tensor, norm_bias: torch.Tensor, seed: torch.Tensor, eps: float, dropout_probability: float, *, recompute: bool = False) -> torch.Tensor:
    """The whole encoder edge tail on the CUDA kernels (the caller has checked ``tail_serves``); ``recompute`` picks the replaying backward over the saved activations."""
    width = edge_states.shape[-1]
    k = edge_states.shape[-2]
    tensors = (edge_states, query_projection, neighbor_projection, edge_weight, hidden_weight, hidden_bias, output_weight, output_bias, norm_weight, norm_bias)
    save = torch.is_grad_enabled() and any(t.requires_grad for t in tensors)
    out = _TailUpdate.apply(edge_states.reshape(-1, width), query_projection.reshape(-1, width), neighbor_projection.reshape(-1, width),
                            flat_neighbor_indices.reshape(-1), edge_weight, hidden_weight, hidden_bias, output_weight, output_bias, norm_weight, norm_bias, seed, eps,
                            dropout_probability, save, k, recompute)
    return out.reshape(edge_states.shape)


# =================================================================================================================================== edge MLP
def mlp_serves(preactivation: torch.Tensor, hidden_weight: torch.Tensor, output_weight: torch.Tensor) -> bool:
    """Whether the CUDA edge MLP runs this call (the caller has checked the family's Triton contract: bf16 ``[..., 128]``, contiguous 128 x 128 parameters)."""
    return wanted() and on_a100(preactivation.device) and preactivation.ndim >= 1 and _loads()


def _mlp_fwd_fake(x, wh, bh, wo, bo, save):
    return [torch.empty_like(x), x.new_empty((x.shape[0] if save else 0, _WIDTH))]


@opaque(fake=_mlp_fwd_fake, name="mpnn_edge_mlp_sm80_fwd")
def _mlp_fwd(x: torch.Tensor, wh: torch.Tensor, bh: torch.Tensor, wo: torch.Tensor, bo: torch.Tensor, save: bool) -> list[torch.Tensor]:
    """Both projections and both GELUs in one kernel; with ``save`` also the projection ``hid`` (the "compute" policy's saved tensor)."""
    ext = _ext()
    wh, wo = _rows16(wh), _rows16(wo)
    img, tab = ext.pack(False, False, False, wh, wo, wo, bh, bo, bh, bo)
    return list(ext.mlp_fwd(x, img, tab, save))


def _mlp_bwd_fake(go, x, hid, wh, bh, wo, bo, recompute):
    d = {"device": x.device}
    return [torch.empty_like(x), torch.empty((_WIDTH, _WIDTH), dtype=wh.dtype, **d), torch.empty((_WIDTH, _WIDTH), dtype=wo.dtype, **d),
            torch.empty((_WIDTH,), dtype=bh.dtype, **d), torch.empty((_WIDTH,), dtype=bo.dtype, **d)]


@opaque(fake=_mlp_bwd_fake, name="mpnn_edge_mlp_sm80_bwd")
def _mlp_bwd(go: torch.Tensor, x: torch.Tensor, hid: torch.Tensor, wh: torch.Tensor, bh: torch.Tensor, wo: torch.Tensor, bo: torch.Tensor, recompute: bool) -> list[torch.Tensor]:
    """grad_x, dWh, dWo, dbh, dbo (parameter dtypes); ``recompute`` rebuilds the projection from x (the "memory" policy), otherwise ``hid`` is the saved one."""
    ext = _ext()
    wh, wo = _rows16(wh), _rows16(wo)
    img, tab = ext.pack(True, True, False, wo, wh, wh, bh, bo, bh, bo)       # backward image: Wo^T (dX3), Wh^T (dX2), and Wh itself (the recompute)
    return list(ext.mlp_bwd(go, x, hid, img, tab, recompute, wh.dtype == torch.float32, bh.dtype == torch.float32))


class _MlpUpdate(torch.autograd.Function):
    """``mode``: 0 inference (nothing saved), 1 the "compute" policy (the projection is saved), 2 the "memory" policy (only the input is kept; backward recomputes the projection)."""

    @staticmethod
    def forward(ctx, x, wh, bh, wo, bo, mode):
        out, hid = _mlp_fwd(x, wh, bh, wo, bo, mode == 1)
        if mode:
            ctx.save_for_backward(x, hid, wh, bh, wo, bo)
            ctx.recompute = mode == 2
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, hid, wh, bh, wo, bo = ctx.saved_tensors
        gx, dwh, dwo, dbh, dbo = _mlp_bwd(grad_out.contiguous(), x, hid, wh, bh, wo, bo, ctx.recompute)
        return gx, dwh, dbh, dwo, dbo, None


def edge_mlp_update(preactivation: torch.Tensor, hidden_weight: torch.Tensor, hidden_bias: torch.Tensor, output_weight: torch.Tensor, output_bias: torch.Tensor,
                    *, memory: bool = False) -> torch.Tensor:
    """The edge MLP on the CUDA kernels (the caller has checked ``mlp_serves``); ``memory`` picks the recompute policy over the saved projection."""
    tensors = (preactivation, hidden_weight, hidden_bias, output_weight, output_bias)
    grad = torch.is_grad_enabled() and any(t.requires_grad for t in tensors)
    out = _MlpUpdate.apply(preactivation.reshape(-1, _WIDTH), hidden_weight, hidden_bias, output_weight, output_bias, (2 if memory else 1) if grad else 0)
    return out.reshape(preactivation.shape)


# =================================================================================================================================== edge LayerNorm
def norm_serves(values: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether the CUDA backward of the compressed-save LayerNorm runs this call (the caller has checked ``_memory_supported``)."""
    return wanted() and on_a100(values.device) and _loads()


def _norm_bwd_fake(dy, x, mean, rstd, weight):
    return [torch.empty_like(dy), torch.empty_like(weight), torch.empty_like(weight)]


@opaque(fake=_norm_bwd_fake, name="mpnn_edge_layernorm_sm80_bwd")
def _norm_bwd(dy: torch.Tensor, x: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, weight: torch.Tensor) -> list[torch.Tensor]:
    """dx (dy's dtype), dw, db (the weight's dtype) of the LayerNorm from the bf16 copy of its input and the forward's statistics."""
    return list(_ext().ln_bwd(_aligned(dy), _aligned(x), mean, rstd, weight))


class _MemoryLayerNorm(torch.autograd.Function):
    """The native forward, a bf16 copy of the input and the statistics kept for the backward, which is one CUDA kernel (the Triton path's contract, same saves)."""

    @staticmethod
    def forward(ctx, values, weight, bias, eps):
        native_norm = values.dtype == torch.bfloat16 and weight.dtype == torch.float32 and not torch.is_autocast_enabled(values.device.type)
        output, mean, rstd = torch.native_layer_norm(values.float() if native_norm else values, (values.shape[-1],), weight, bias, eps)
        ctx.save_for_backward(values.to(torch.bfloat16), weight, mean, rstd)
        ctx.input_dtype = values.dtype
        return output.to(values.dtype) if native_norm else output

    @staticmethod
    def backward(ctx, grad_output):
        saved, weight, mean, rstd = ctx.saved_tensors
        dx, dw, db = _norm_bwd(grad_output.contiguous(), saved, mean.reshape(-1), rstd.reshape(-1), weight)
        return dx.to(ctx.input_dtype), dw, db, None


def edge_layer_norm_memory(values: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    return _MemoryLayerNorm.apply(values, weight, bias, eps)


# =================================================================================================================================== edge dropout
def dropout_serves(values: torch.Tensor) -> bool:
    """Whether the CUDA pack / backward kernels of the bit-packed dropout mask run this call (the caller has checked ``_bitpack_supported``)."""
    return wanted() and on_a100(values.device) and _loads()


def _pack_fake(mask):
    return torch.empty(((mask.numel() + 7) // 8,), device=mask.device, dtype=torch.uint8)


@opaque(fake=_pack_fake, name="mpnn_edge_dropout_sm80_pack")
def _pack(mask: torch.Tensor) -> torch.Tensor:
    """ATen's boolean dropout mask to one bit per element."""
    return _ext().dropout_pack(mask)


def _dropout_bwd_fake(grad, packed, scale):
    return torch.empty_like(grad)


@opaque(fake=_dropout_bwd_fake, name="mpnn_edge_dropout_sm80_bwd")
def _dropout_bwd(grad: torch.Tensor, packed: torch.Tensor, scale: float) -> torch.Tensor:
    """grad * scale where the bit is set, 0 elsewhere (``scale`` is a runtime scalar: the bf16 rounding boundary of ``native_dropout_backward``)."""
    return _ext().dropout_bwd(_aligned(grad), packed, scale)


class _BitpackDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, probability):
        output, mask = torch.ops.aten.native_dropout.default(values, probability, True)      # the same native operation as F.dropout: its values and Philox state
        ctx.save_for_backward(_pack(mask))
        ctx.scale = 1.0 / (1.0 - probability)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        (packed,) = ctx.saved_tensors
        return _dropout_bwd(grad_output.contiguous(), packed, ctx.scale), None


def edge_dropout_bitpack(values: torch.Tensor, probability: float) -> torch.Tensor:
    return _BitpackDropout.apply(values, probability)
