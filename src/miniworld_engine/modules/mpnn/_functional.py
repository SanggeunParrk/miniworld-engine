"""Tensor primitives shared by the ProteinMPNN layers: initialiser, LayerNorm (a :class:`BackendModule`), neighbour gathers and the packed
projection."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.weak import WeakTensorKeyDictionary

from miniworld_engine.modules.primitives import BackendModule


def lecun_normal_(module: nn.Linear | nn.Embedding, scale: float = 1.0) -> None:
    """Initialize ``module.weight`` with the truncated LeCun distribution."""
    normal = torch.distributions.normal.Normal(0, 1)
    lower_cdf = normal.cdf(torch.tensor(-2.0))
    upper_cdf = normal.cdf(torch.tensor(2.0))
    probabilities = lower_cdf + (upper_cdf - lower_cdf) * torch.rand_like(module.weight)
    values = math.sqrt(2.0) * torch.erfinv(torch.clamp(2.0 * probabilities - 1.0, -1.0 + 1e-8, 1.0 - 1e-8))
    values.clamp_(-2.0, 2.0)
    stddev = math.sqrt(scale / module.weight.shape[-1]) / 0.87962566103423978
    with torch.no_grad():
        module.weight.copy_(stddev * values)


class MPNNLayerNorm(BackendModule, nn.LayerNorm):
    """``nn.LayerNorm`` (state-dict compatible) whose affine stays FP32 under a cast to bf16 / fp16 (FP64 tests keep FP64); a bf16 / fp16
    input outside autocast is normalised in fp32 and returned in its dtype."""

    backends = ("pytorch",)

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
        *,
        backend: str = "pytorch",
    ) -> None:
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias, device=device, dtype=dtype)
        self.set_backend(backend)

    def _apply(self, fn, *args, **kwargs):
        def keep_norm_precision(tensor):
            converted = fn(tensor)
            if converted.dtype in {torch.bfloat16, torch.float16}:
                # Do not round an existing FP32 gamma through BF16 on repeated casts.
                return tensor.to(device=converted.device, dtype=torch.float32)
            return converted

        return super()._apply(keep_norm_precision, *args, **kwargs)

    def _forward_pytorch(self, input: torch.Tensor) -> torch.Tensor:
        if input.dtype in {torch.bfloat16, torch.float16} and not torch.is_autocast_enabled(input.device.type):
            return F.layer_norm(input.float(), self.normalized_shape, self.weight, self.bias, self.eps).to(input.dtype)
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)


#: The width every ProteinMPNN message kernel is built for (hidden / output projections [128, 128], activations [..., 128]).
KERNEL_WIDTH = 128
#: Signed 32-bit element offsets (with a padded tail tile) the message kernels index with.
_INT32_MAX = 2**31 - 1


class EdgeLayerNorm(MPNNLayerNorm):
    """The LayerNorm of the encoder edge messages (a state-dict twin of :class:`MPNNLayerNorm`).

    ``"triton"``: with gradients enabled, the memory LayerNorm (``kernels/mpnn_edge_layernorm``): the native forward, bit for bit, and a
    bf16 copy of the edge-sized input saved for its Triton backward. Range: a bf16 CUDA input [..., 128] within 32-bit offsets, contiguous,
    an fp32 / bf16 affine of one dtype, or (under bf16 autocast) an fp32 input and affine; not under deterministic algorithms. Without
    gradients nothing is saved, so the call is the reference statement itself.
    """

    backends = ("pytorch", "triton", "cuda")

    def _supports_cuda(self, input: torch.Tensor) -> str | None:
        """A100: the same contract, the backward on ``kernels/integrations/mpnn_edge_sm80.edge_layer_norm_memory``."""
        if not torch.is_grad_enabled():
            return None
        from miniworld_engine.kernels.integrations import mpnn_edge_sm80

        return mpnn_sm80_unsupported(input, mpnn_edge_sm80._loads) or self._supports_triton(input)

    def _forward_cuda(self, input: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return self._forward_pytorch(input)
        from miniworld_engine.kernels.integrations import mpnn_edge_sm80

        return mpnn_edge_sm80.edge_layer_norm_memory(input, self.weight, self.bias, self.eps)

    def _supports_triton(self, input: torch.Tensor) -> str | None:
        if not torch.is_grad_enabled():
            return None
        weight, bias = self.weight, self.bias
        if weight is None or bias is None:
            return "the memory LayerNorm needs the affine weight and bias"
        if input.device.type != "cuda" or weight.device != input.device or bias.device != input.device:
            return "the Triton kernels run on CUDA tensors only"
        autocast_bf16 = torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16
        native = input.dtype is torch.bfloat16 and weight.dtype in {torch.bfloat16, torch.float32} and bias.dtype == weight.dtype
        autocast = (
            autocast_bf16
            and input.dtype in {torch.float32, torch.bfloat16}
            and weight.dtype is torch.float32
            and bias.dtype is torch.float32
        )
        if not (native or autocast):
            return f"input {input.dtype}, affine {weight.dtype}: the memory LayerNorm takes bf16 input (or fp32 under bf16 autocast)"
        if torch.are_deterministic_algorithms_enabled():
            return "the memory LayerNorm's backward is not deterministic"
        if input.dim() == 0 or input.shape[-1] != KERNEL_WIDTH or input.numel() == 0 or input.numel() > _INT32_MAX:
            return f"input {tuple(input.shape)}: the memory LayerNorm is built for non-empty [..., {KERNEL_WIDTH}] within 32-bit offsets"
        if not (input.is_contiguous() and weight.is_contiguous() and bias.is_contiguous()):
            return "the input and the affine must be contiguous"
        return None

    def _forward_triton(self, input: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return self._forward_pytorch(input)
        from miniworld_engine.kernels.mpnn_edge_layernorm.triton import edge_layer_norm_memory

        return edge_layer_norm_memory(input, self.weight, self.bias, self.eps)


def bf16_parameter_dtype_unsupported(*params: torch.Tensor) -> str | None:
    """Why the MPNN kernels cannot take the dtypes of these projection parameters, or ``None``: they compute in bf16, so every parameter
    is bf16, or (under CUDA bf16 autocast) fp32 / bf16 masters of one dtype."""
    dtypes = {p.dtype for p in params}
    autocast_bf16 = torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16
    if dtypes != {torch.bfloat16} and not (autocast_bf16 and len(dtypes) == 1 and dtypes <= {torch.float32, torch.bfloat16}):
        return f"projection parameters {sorted(map(str, dtypes))}: the kernels take bf16 parameters (fp32 masters only under bf16 autocast)"
    return None


def bf16_projection_unsupported(*params: torch.Tensor) -> str | None:
    """:func:`bf16_parameter_dtype_unsupported` plus: every parameter contiguous on a CUDA device."""
    why = bf16_parameter_dtype_unsupported(*params)
    if why is not None:
        return why
    if not all(p.is_cuda and p.is_contiguous() for p in params):
        return "the projection parameters must be contiguous CUDA tensors"
    return None


def mpnn_triton_unsupported(x: torch.Tensor) -> str | None:
    """The activation side of the MPNN Triton kernels: a non-empty bf16 CUDA tensor [..., 128] within 32-bit offsets."""
    if x.device.type != "cuda":
        return "the Triton kernels run on CUDA tensors only"
    if x.dtype is not torch.bfloat16:
        return f"dtype {x.dtype}: the MPNN Triton kernels take bfloat16 activations"
    if x.dim() < 2 or x.shape[-1] != KERNEL_WIDTH or x.numel() == 0:
        return f"activation {tuple(x.shape)}: the kernels are built for [..., {KERNEL_WIDTH}]"
    if x.numel() > _INT32_MAX - (KERNEL_WIDTH * KERNEL_WIDTH - 1):
        return "the activation exceeds the kernels' 32-bit offsets"
    return None


def mpnn_sm80_unsupported(x: torch.Tensor, loads=None) -> str | None:
    """Why the ``"cuda"`` backends of the MPNN classes cannot run on ``x``'s card: they are the A100 (sm_80) kernels
    (``kernels/integrations/mpnn_msg_sm80.py``, ``mpnn_edge_sm80.py``); every other card has the Triton kernels only. ``loads``: the
    extension's loader (a failed build refuses). Device only: capture-safe."""
    if x.device.type != "cuda":
        return "the CUDA kernels run on CUDA tensors only"
    cc = torch.cuda.get_device_capability(x.device)
    if cc not in ((8, 0), (12, 0)):
        return f"the MPNN CUDA kernels are the A100 (sm_80) and RTX 5090 (sm_120a) builds, got sm_{cc[0]}{cc[1]} (that card has the Triton kernels only)"
    if loads is not None and not loads():
        return "the sm_80 MPNN extension failed to build"
    return None


def projection_input(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Match native low-precision GEMMs without changing the autocast path."""
    if not torch.is_autocast_enabled(values.device.type) and weight.dtype in {torch.bfloat16, torch.float16}:
        return values.to(weight.dtype)
    return values


_FLAT_NEIGHBOR_CACHE: WeakTensorKeyDictionary = WeakTensorKeyDictionary()


def _flat_neighbor_indices(neighbor_indices: torch.Tensor, batch: int, length: int) -> torch.Tensor:
    """Return ``neighbor_indices`` shifted into a flattened ``[B*L]`` node axis.

    Every projection in every layer gathers with the same index tensor, so the shifted copy is built once per distinct index tensor
    rather than once per call (``F.embedding`` keeps each copy alive for backward). The memo covers eager execution only and is skipped
    while tracing, where a Python-level cache would be an invisible guard.
    """
    batch_offsets = (torch.arange(batch, device=neighbor_indices.device) * length).view(batch, 1, 1)
    if torch.compiler.is_compiling():
        return neighbor_indices + batch_offsets
    cached = _FLAT_NEIGHBOR_CACHE.get(neighbor_indices)
    if cached is not None:
        version, flat = cached
        if version == neighbor_indices._version and flat.shape == neighbor_indices.shape:
            return flat
    flat = neighbor_indices + batch_offsets
    _FLAT_NEIGHBOR_CACHE[neighbor_indices] = (neighbor_indices._version, flat)
    return flat


def gather_neighbors(values: torch.Tensor, neighbor_indices: torch.Tensor) -> torch.Tensor:
    """Gather node values ``[B, L, C]`` at ``neighbor_indices`` ``[B, L, K]`` with a fusible backward."""
    batch, length, channels = values.shape
    if values.is_floating_point():
        if batch == 1:
            return F.embedding(neighbor_indices, values[0])
        flattened = values.reshape(batch * length, channels)
        return F.embedding(_flat_neighbor_indices(neighbor_indices, batch, length), flattened)

    flattened_indices = neighbor_indices.reshape(batch, -1)
    flattened_indices = flattened_indices.unsqueeze(-1).expand(-1, -1, channels)
    gathered = torch.gather(values, 1, flattened_indices)
    return gathered.reshape(*neighbor_indices.shape, channels)


def project_packed_neighbor_inputs(
    query_nodes: torch.Tensor,
    edge_features: torch.Tensor,
    neighbor_nodes: torch.Tensor,
    neighbor_indices: torch.Tensor,
    projection: nn.Linear,
    node_width: int,
    edge_width: int,
) -> torch.Tensor:
    """Apply a packed ``[query, edge, neighbor]`` projection by blocks."""
    weight = projection.weight
    output = F.linear(query_nodes, weight[:, :node_width], projection.bias).unsqueeze(2)
    output = output + F.linear(edge_features, weight[:, node_width : node_width + edge_width])
    neighbor_projection = F.linear(neighbor_nodes, weight[:, node_width + edge_width :])
    return output + gather_neighbors(neighbor_projection, neighbor_indices)
