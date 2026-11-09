"""Normalisations: LayerNorm, RMSNorm, AdaptiveLayerNorm (AlphaFold 3 Algorithm 26), Chai-1's IdentityAdaptiveLayerNorm and ESMFold2's
AdaptiveRMSNorm.

Each is a :class:`~miniworld_engine.modules.primitives.BackendModule`; ``_forward_pytorch`` is the reference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size

from miniworld_engine.modules.primitives import BackendModule, Linear, master_linear, random_input, stream_shape, wide

_shape_t = int | list[int] | Size

#: The epsilon of :class:`AdaptiveRMSNorm` (and of the q / k RMSNorms beside it): float32's machine epsilon whatever the input dtype
#: (a configured constant of the models' math, also in a float64 call; only the arithmetic around it widens).
RMS_EPS = torch.finfo(torch.float32).eps

#: Activation dtypes the engine's norm kernels (CUDA general / pipe, Triton) take.
_NORM_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def h100_unsupported(*tensors: torch.Tensor) -> str | None:
    """Why the engine's ``"cuda"`` backends (hand-written sm_90 kernels) cannot take these tensors: not CUDA, not an H100 (sm_90), or a
    fake / traced tensor (the raw extension calls are opaque launches; their wrappers are not entered under tracing here). ``None`` if they
    can. Shapes / device only: capture-safe."""
    t = tensors[0]
    if t.device.type != "cuda":
        return "the CUDA kernels run on CUDA tensors only"
    cc = torch.cuda.get_device_capability(t.device)
    if cc != (9, 0):
        return f"the hand-written CUDA kernels are the H100 (sm_90) ones, got sm_{cc[0]}{cc[1]}"
    return None


def triton_unsupported(*tensors: torch.Tensor) -> str | None:
    """Why the engine's ``"triton"`` backends cannot take these tensors (not CUDA), or ``None``."""
    if tensors[0].device.type != "cuda":
        return "the Triton kernels run on CUDA tensors only"
    return None


def _as_rows3(x: torch.Tensor) -> torch.Tensor:
    """``x`` with at least three dimensions (the Triton launchers key on the pre-flatten ``(B, L, D)`` / ``(B, L, L, D)`` shape): an
    unbatched ``[L, D]`` (or a single row) becomes one batch of it."""
    return x if x.dim() >= 3 else x.reshape(1, -1, x.shape[-1])


class _Fp32ParamsMixin(nn.Module):
    """Pin this module's floating-point params/buffers to at least fp32 under ANY dtype cast (``.double()`` makes them fp64).

    Norm affine params (gamma init 1.0, beta 0.0) stagnate when stored bf16: at value 1.0
    the bf16 ULP is 2**-7 = 0.0078 > Adam's per-step update (~lr = 1.8e-3), so updates
    round back to 1.0 and gamma never trains. Keeping gamma/beta fp32 (trunk stays bf16)
    lets the update land. Overriding ``_apply`` -- the funnel for ``.to``/``.bfloat16`` --
    makes the pin survive later bulk ``.to(torch.bfloat16)`` of the parent trunk.

    Declared as an ``nn.Module`` subclass, not a bare mixin: ``super()._apply`` only exists
    because this class is always mixed in FRONT of one (``LayerNorm(_Fp32ParamsMixin,
    BackendModule, nn.LayerNorm)``)."""

    def _apply(self, fn, *args, **kwargs):
        def fp32_fn(t):
            out = fn(t)
            if isinstance(out, torch.Tensor) and out.is_floating_point():
                out = out.to(wide(out))
            return out

        return super()._apply(fp32_fn, *args, **kwargs)


class LayerNorm(_Fp32ParamsMixin, BackendModule, nn.LayerNorm):
    """A LayerNorm with an fp32-pinned affine (see :class:`_Fp32ParamsMixin`), computed in fp32 and returned in the input dtype.

    Backends: ``"anthropic"`` is the af3_torch kit's LayerNorm (``af3_torch_opt.ln_rows``: xfold's fastnn Triton kernel, several rows per
    program; rows wider than 512 bytes take that kernel's stock one-row launch, as the kit does) -- the same statement (fp32 statistics and
    affine, the result stored in the input dtype), forward only.

    ``"cuda"`` (H100): the engine's CUDA LayerNorm (``kernels/layernorm/cuda/general.layernorm_cuda``): rows of 144-2048 bytes (16-byte
    multiples) on the pipelined kernels (``cuda/pipe``), every other width on the general kernels; fp32 statistics, forward and backward
    (dx and the affine gradients, no atomics). ``"triton"``: the Triton LayerNorm (``kernels/layernorm/triton/main.triton_layernorm``:
    the fused forward that saves the row statistics, the persistent dx / dgamma / dbeta backward). Both: one normalised dimension,
    bf16 / fp16 / fp32 activations, the result in the input dtype."""
    version = 1   # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "anthropic", "cuda", "triton")

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
        *,
        backend: str = "pytorch",
    ) -> None:
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias, device=device, dtype=dtype)
        # fp32-pin the affine from construction, even when a bf16 dtype is requested; _apply keeps it fp32 through later .to() casts.
        with torch.no_grad():
            if self.weight is not None and self.weight.is_floating_point():
                self.weight.data = self.weight.data.to(wide(self.weight))
            if self.bias is not None and self.bias.is_floating_point():
                self.bias.data = self.bias.data.to(wide(self.bias))
        self.set_backend(backend)

    def _forward_pytorch(self, input: torch.Tensor) -> torch.Tensor:
        """Normalise in at least fp32 (a fp32-pinned affine never dtype-mismatches a bf16 activation; fp64 stays fp64); restore the
        activation dtype."""
        w = wide(input, *(p for p in (self.weight, self.bias) if p is not None))
        return F.layer_norm(input.to(w), self.normalized_shape, self.weight, self.bias, self.eps).to(input.dtype)

    def make_inputs(self, *, length, samples, depth, dtype, device, generator, stream=None) -> dict:
        shape = stream_shape(stream, samples=samples, length=length, depth=depth, width=self.normalized_shape[-1])
        return {"input": random_input(shape, dtype=dtype, device=device, generator=generator)}

    _KIT_LN = (("af3_torch_opt.ln_rows", "af3_torch/opt"), ("xfold.fastnn.layer_norm", "af3_torch/opt/forward/af3t/af3_torch"))

    def _supports_anthropic(self, input: torch.Tensor) -> str | None:
        from miniworld_engine.modules import _anthropic

        for module, path in self._KIT_LN:
            why = _anthropic.unavailable(module, input.device, path=path)
            if why is not None:
                return why
        if torch.is_grad_enabled():
            return "the kit's LayerNorm kernel has no backward (inference only: call under torch.no_grad() / inference_mode)"
        if input.dtype not in (torch.bfloat16, torch.float32):
            return f"dtype {input.dtype}: the backend is verified for bfloat16 / float32 activations"
        if len(self.normalized_shape) != 1:
            return "the kernel normalises the last dimension only"
        if self.weight is not None and self.weight.dtype != torch.float32:
            return "the affine must be fp32 (the module's pinned layout)"
        return None

    def _forward_anthropic(self, input: torch.Tensor) -> torch.Tensor:
        from miniworld_engine.modules import _anthropic

        (rows, rows_path), (stock, stock_path) = self._KIT_LN
        y = _anthropic.load(rows, path=rows_path).layer_norm_rows(input, self.normalized_shape, self.weight, self.bias, self.eps)
        if y is None:   # a row wider than the kit's 512-byte blocking: xfold's own one-row-per-program launch of the same kernel body
            y = _anthropic.load(stock, path=stock_path).LayerNormTritonFunc.apply(input, self.normalized_shape, self.weight, self.bias, self.eps)
        return y

    # ------------------------------------------------------------------------------------------------------------- engine kernels
    def _supports_engine(self, input: torch.Tensor) -> str | None:
        """The range both engine backends share: one normalised dimension, a listed dtype, a non-empty input."""
        if len(self.normalized_shape) != 1:
            return "the kernels normalise the last dimension only"
        if input.dtype not in _NORM_DTYPES:
            return f"dtype {input.dtype}: the kernels take bfloat16 / float16 / float32 activations"
        if input.numel() == 0:
            return "empty input"
        if input.shape[-1] != self.normalized_shape[0]:
            return f"input width {input.shape[-1]} != normalized_shape {self.normalized_shape[0]}"
        return None

    def _supports_cuda(self, input: torch.Tensor) -> str | None:
        from miniworld_engine.kernels.layernorm.cuda.general import supported

        why = h100_unsupported(input) or self._supports_engine(input)
        if why is not None:
            return why
        if not supported(input):
            return f"width {input.shape[-1]} ({input.dtype}) is beyond the general CUDA LayerNorm's builds"
        return None

    def _forward_cuda(self, input: torch.Tensor) -> torch.Tensor:
        from miniworld_engine.kernels.layernorm.cuda.general import layernorm_cuda

        return layernorm_cuda(input, self.weight, self.bias, self.eps)

    def _supports_triton(self, input: torch.Tensor) -> str | None:
        return triton_unsupported(input) or self._supports_engine(input)

    def _forward_triton(self, input: torch.Tensor) -> torch.Tensor:
        from miniworld_engine.kernels.layernorm.triton.main import triton_layernorm

        return triton_layernorm(_as_rows3(input), self.weight, self.bias, self.eps).reshape(input.shape)


class RMSNorm(BackendModule, nn.RMSNorm):
    """``torch.nn.RMSNorm`` (state-dict compatible), with :meth:`effective_eps`.

    ``eps=None`` keeps torch's accumulation-dtype epsilon: BF16/FP16 accumulate in FP32, so the epsilon is float32's.

    ``"cuda"`` (H100): the engine's CUDA RMSNorm (``kernels/rmsnorm/cuda.rmsnorm_cuda``: the pipelined kernels for 144-2048-byte rows,
    the general LayerNorm extension built without mean / bias for every other width), forward and backward. ``"triton"``: the Triton
    RMSNorm (``kernels/rmsnorm/triton/main.triton_rmsnorm``), forward and backward. Both: one normalised dimension, bf16 / fp16 / fp32
    activations, fp32 statistics, the result in the input dtype, the epsilon :meth:`effective_eps`.
    """

    backends = ("pytorch", "cuda", "triton")

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float | None = None,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
        *,
        backend: str = "pytorch",
    ) -> None:
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype)
        self.set_backend(backend)

    def effective_eps(self, dtype):
        if self.eps is not None:
            return self.eps
        return torch.finfo(torch.float32 if dtype in (torch.bfloat16, torch.float16) else dtype).eps

    def _forward_pytorch(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)

    def _supports_engine(self, x: torch.Tensor) -> str | None:
        if len(self.normalized_shape) != 1:
            return "the kernels normalise the last dimension only"
        if x.dtype not in _NORM_DTYPES:
            return f"dtype {x.dtype}: the kernels take bfloat16 / float16 / float32 activations"
        if x.numel() == 0:
            return "empty input"
        if x.shape[-1] != self.normalized_shape[0]:
            return f"input width {x.shape[-1]} != normalized_shape {self.normalized_shape[0]}"
        return None

    def _supports_cuda(self, x: torch.Tensor) -> str | None:
        from miniworld_engine.kernels.layernorm.cuda.general import supported

        why = h100_unsupported(x) or self._supports_engine(x)
        if why is not None:
            return why
        if not supported(x):
            return f"width {x.shape[-1]} ({x.dtype}) is beyond the general CUDA RMSNorm's builds"
        return None

    def _forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        from miniworld_engine.kernels.rmsnorm.cuda import rmsnorm_cuda

        return rmsnorm_cuda(x, self.weight, self.effective_eps(x.dtype))

    def _supports_triton(self, x: torch.Tensor) -> str | None:
        return triton_unsupported(x) or self._supports_engine(x)

    def _forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        from miniworld_engine.kernels.rmsnorm.triton.main import triton_rmsnorm

        return triton_rmsnorm(_as_rows3(x), self.weight, self.effective_eps(x.dtype)).reshape(x.shape)


class AdaptiveLayerNorm(BackendModule):
    """Adaptive LayerNorm::

        out = sigmoid(W_s LN_c(cond) + b_s) * LN(x) + W_b LN_c(cond)

    ``LN`` has no affine (``ln_in``), ``LN_c`` a weight and no bias (``ln_cond``, fp32-pinned whatever ``dtype`` the rest runs in: a bf16
    gamma at 1.0 cannot move by one Adam step). ``to_scale`` starts at zero weight / unit bias, ``to_bias`` at zero.
    """

    backends = ("pytorch",)

    def __init__(
        self,
        d_hidden: int,
        d_cond: int,
        *,
        dtype: torch.dtype = torch.float32,
        backend: str = "pytorch",
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.ln_in = nn.LayerNorm(d_hidden, elementwise_affine=False, dtype=dtype)
        self.ln_cond = LayerNorm(d_cond, bias=False)
        self.to_scale = Linear(d_cond, d_hidden, init="gating", dtype=dtype)
        self.to_bias = Linear(d_cond, d_hidden, bias=False, init="zero", dtype=dtype)
        self.set_backend(backend)

    def _forward_pytorch(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """``x`` ``[..., d_hidden]`` conditioned on ``cond`` ``[..., d_cond]``."""
        x_norm = self.ln_in(x)
        cond_norm = self.ln_cond(cond)
        scale = master_linear(self.to_scale, cond_norm)
        bias = master_linear(self.to_bias, cond_norm)
        return torch.sigmoid(scale) * x_norm + bias

    def forward_with_stream(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(self(x, cond), x)``, for a caller that adds ``x`` to something later (a residual stream)."""
        return self(x, cond), x

    def forward_with_stream_cond(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(self(x, cond), x, cond)``, for a caller that also feeds ``cond`` to a second consumer."""
        return self(x, cond), x, cond


class IdentityAdaptiveLayerNorm(BackendModule):
    """Chai-1's AdaLN: the identity-centred scale of the RAW conditioning, no LayerNorm of it and no biases::

        out = (1 + W_s cond) * LN(x) + W_b cond

    ``LN`` has no affine (``ln_in``, as :class:`AdaptiveLayerNorm`'s) and the epsilon ``eps`` (Chai-1's release: 0.1; FoldForge's fast
    mode: 1e-5); ``to_scale`` / ``to_bias`` are bias-free and zero-initialised (the identity). A part of the Chai-1 DiT blocks, not a
    requirements module of its own.
    """

    backends = ("pytorch",)

    def __init__(self, d_hidden: int, d_cond: int, *, eps: float = 1e-5, dtype: torch.dtype = torch.float32,
                 backend: str = "pytorch") -> None:
        super().__init__()
        self.dtype = dtype
        self.ln_in = nn.LayerNorm(d_hidden, eps=eps, elementwise_affine=False, dtype=dtype)
        self.to_scale = Linear(d_cond, d_hidden, bias=False, init="zero", dtype=dtype)
        self.to_bias = Linear(d_cond, d_hidden, bias=False, init="zero", dtype=dtype)
        self.set_backend(backend)

    def _forward_pytorch(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """``x`` ``[..., d_hidden]`` conditioned on ``cond`` ``[..., d_cond]``."""
        x_norm = self.ln_in(x)
        scale = master_linear(self.to_scale, cond)
        bias = master_linear(self.to_bias, cond)
        return (scale + 1) * x_norm + bias


class AdaptiveRMSNorm(BackendModule):
    """ESMFold2's atom AdaLN: an affine-free RMSNorm of the activation, SiLU (not a LayerNorm) of the conditioning, and the
    identity-centred scale with no bias on either projection:

        out = (1 + W_s silu(c)) * x / sqrt(mean(x^2) + eps) + W_b silu(c)          eps = float32 epsilon

    Parameters ``to_scale.weight`` and ``to_bias.weight`` (both zero-initialised: the identity). The norm runs in float32 and the result
    comes back in ``x``'s dtype. A part of the ESMFold2 atom block, not a requirements module of its own.
    """

    backends = ("pytorch",)

    def __init__(self, d_hidden: int, d_cond: int, *, dtype: torch.dtype = torch.float32, backend: str = "pytorch") -> None:
        super().__init__()
        self.to_scale = Linear(d_cond, d_hidden, bias=False, init="zero", dtype=dtype)
        self.to_bias = Linear(d_cond, d_hidden, bias=False, init="zero", dtype=dtype)
        self.set_backend(backend)

    def _forward_pytorch(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        w = wide(x)
        xw = x.to(w)
        norm = xw * torch.rsqrt(xw.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        context = torch.nn.functional.silu(cond)
        scale = master_linear(self.to_scale, context)
        bias = master_linear(self.to_bias, context)
        return ((scale.to(w) + 1.0) * norm + bias.to(w)).to(x.dtype)
