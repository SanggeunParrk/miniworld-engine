"""Normalisations: LayerNorm, RMSNorm, AdaptiveLayerNorm (AlphaFold 3 Algorithm 26), Chai-1's IdentityAdaptiveLayerNorm and ESMFold2's
AdaptiveRMSNorm.

Each is a :class:`~miniworld_engine.modules.primitives.BackendModule`; ``_forward_pytorch`` is the reference.
"""

from typing import ClassVar

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


def a100(t: torch.Tensor) -> bool:
    """A CUDA tensor on an A100 (capability 8.0): the ``"cuda"`` backends of the classes that have sm_80 builds route there to them (the H100
    path is unchanged). Device only: capture-safe."""
    return t.device.type == "cuda" and torch.cuda.get_device_capability(t.device) == (8, 0)


def sm120(t: torch.Tensor) -> bool:
    """A CUDA tensor on a GeForce RTX 5090 (capability 12.0, card key ``5090_sm120``): the ``"cuda"`` backends of the classes that have sm_120a
    builds route there to them (the H100 and A100 paths are unchanged). Device only: capture-safe."""
    return t.device.type == "cuda" and torch.cuda.get_device_capability(t.device) == (12, 0)


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


# ----------------------------------------------------------------------------------------------- Protenix's fast_layernorm (anthropic, train)
# The release ships Protenix 2.0's apex-derived ``fast_layer_norm_cuda_v2`` (forward + backward) as a stream-correct prebuilt .so for
# cp311 + torch 2.13.0+cu130 / 2.7.1+cu126 only, together with its patched sources (``fastln_prebuilt_cu130/patched_src``: the stock sources
# with the current stream at every launch). The same sources are built here for this environment through the engine's extension cache
# (``kernels/_nvcc.load_extension``: sm_80 / sm_90, the build under the cache, never in the release tree), after checking each file's
# sha256 against the release manifest.
_FASTLN_SRC = "protenix_v2/opt/forward/flashpairformer/third_party/fastln_prebuilt_cu130/patched_src"
_FASTLN_FILES = ("compat.h", "type_shim.h", "layer_norm_cuda.cpp", "layer_norm_cuda_kernel.cu")


def _fastln():
    """The built extension (cached per process; the build is cached on disk by its sources and flags)."""
    hit = _FASTLN_STATE.get("ext")
    if hit is not None:
        return hit
    import hashlib

    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension
    from miniworld_engine.modules import _anthropic

    base = _anthropic.root() / _FASTLN_SRC
    manifest = _anthropic._manifest()
    for name in _FASTLN_FILES:
        rel = f"{_FASTLN_SRC}/{name}"
        want = manifest.get(rel)
        if want is None or hashlib.sha256((base / name).read_bytes()).hexdigest() != want:
            raise _anthropic.ReleaseMissing(f"{rel} is missing or differs from release {_anthropic.REVISION[:8]}")
    from pathlib import Path

    home = ensure_cuda_home()
    # the binding is a .cpp (host compiler): it needs the toolkit's headers, which a conda toolkit keeps under targets/<arch>/include
    cuda_inc = [
        str(p)
        for p in ((Path(home) / "include", Path(home) / "targets" / "x86_64-linux" / "include") if home else ())
        if (p / "cuda_runtime.h").is_file()
    ]
    ext = load_extension(
        name="anthropic_protenix_fast_layer_norm_v1",
        sources=[str(base / "layer_norm_cuda.cpp"), str(base / "layer_norm_cuda_kernel.cu")],
        extra_include_paths=[str(base), *cuda_inc],
        extra_cuda_cflags=[*host_flags(), "-O3", *gencodes("80", "90")],
        extra_cflags=["-O3"],
        verbose=False,
    )
    _FASTLN_STATE["ext"] = ext
    return ext


_FASTLN_STATE: dict = {}


def _fastln_reason() -> str | None:
    if "why" not in _FASTLN_STATE:
        try:
            _fastln()
            _FASTLN_STATE["why"] = None
        except Exception as e:  # the reason the backend cannot serve the training call
            _FASTLN_STATE["why"] = f"Protenix fast_layernorm unavailable: {type(e).__name__}: {str(e)[:200]}"
    return _FASTLN_STATE["why"]


class _FastLayerNormFunction(torch.autograd.Function):
    """Protenix's ``FusedLayerNormAffineFunction`` (protenix/model/layer_norm/layer_norm.py) on the built extension: the affine cast to the
    activation's dtype (the master layout's bf16 compute copy), fp32 statistics saved for the backward, the affine gradients returned in
    the parameters' dtype."""

    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps):
        ext = _fastln()
        d = input.dtype
        x = input.contiguous()
        if weight is None and bias is None:
            out, mean, invvar = ext.forward_none_affine(x, normalized_shape, eps)
        elif weight is None:
            out, mean, invvar = ext.forward_with_bias_affine(x, normalized_shape, bias.to(d), eps)
        elif bias is None:
            out, mean, invvar = ext.forward_with_weight_affine(x, normalized_shape, weight.to(d), eps)
        else:
            out, mean, invvar = ext.forward_with_both_affine(x, normalized_shape, weight.to(d), bias.to(d), eps)
        ctx.save_for_backward(x, weight, bias, mean, invvar)
        ctx.shape, ctx.eps = normalized_shape, eps
        return out.view(input.shape)

    @staticmethod
    def backward(ctx, dout):
        ext = _fastln()
        x, weight, bias, mean, invvar = ctx.saved_tensors
        d = dout.dtype
        g = dout.contiguous()
        if weight is None and bias is None:
            dx, dw, db = ext.backward_none_affine(g, mean, invvar, x, ctx.shape, ctx.eps)
        elif weight is None:
            dx, dw, db = ext.backward_with_bias_affine(g, mean, invvar, x, ctx.shape, bias.to(d), ctx.eps)
        elif bias is None:
            dx, dw, db = ext.backward_with_weight_affine(g, mean, invvar, x, ctx.shape, weight.to(d), ctx.eps)
        else:
            dx, dw, db = ext.backward_with_both_affine(g, mean, invvar, x, ctx.shape, weight.to(d), bias.to(d), ctx.eps)
        dw = None if weight is None else dw.to(weight.dtype)
        db = None if bias is None else db.to(bias.dtype)
        return dx, dw, db, None, None


class LayerNorm(_Fp32ParamsMixin, BackendModule, nn.LayerNorm):
    """A LayerNorm with an fp32-pinned affine (see :class:`_Fp32ParamsMixin`), computed in fp32 and returned in the input dtype.

    Backends: ``"anthropic"`` is the af3_torch kit's LayerNorm (``af3_torch_opt.ln_rows``: xfold's fastnn Triton kernel, several rows per
    program; rows wider than 512 bytes take that kernel's stock one-row launch, as the kit does) -- the same statement (fp32 statistics and
    affine, the result stored in the input dtype) for a call without a gradient; a call with a gradient runs Protenix's fast_layernorm
    (``FusedLayerNormAffineFunction`` on ``fast_layer_norm_cuda_v2``, forward and backward), built for this environment from the release's
    patched sources (:func:`_fastln`). ``"cueq"``: cuEquivariance's ``layer_norm_transpose`` ("nd->nd"), forward and backward.

    ``"cuda"`` (H100): the engine's CUDA LayerNorm (``kernels/layernorm/cuda/general.layernorm_cuda``): rows of 144-2048 bytes (16-byte
    multiples) on the pipelined kernels (``cuda/pipe``), rows that are not a 16-byte multiple (267 / 451 / 651 / 831 / 833) on the row
    kernels (``cuda/general/ln_rows.cu``), every other width on the general kernels; fp32 statistics, forward and backward
    (dx and the affine gradients, no atomics). ``"triton"``: the Triton LayerNorm (``kernels/layernorm/triton/main.triton_layernorm``:
    the fused forward that saves the row statistics, the persistent dx / dgamma / dbeta backward). Both: one normalised dimension,
    bf16 / fp16 / fp32 activations, the result in the input dtype."""

    version = 1  # human-managed: bump when a change alters this class's accuracy or speed results

    backends = ("pytorch", "anthropic", "cueq", "cuda", "triton")
    backend_versions: ClassVar[dict[str, int]] = {
        "anthropic": 2,
        "cuda:h100_sm90": 2,
        "cuda:a100_sm80": 2,
    }  # 2026-11-10: kits; H100 odd-width rows kernels; 2026-11-11: A100 staged odd-width forward

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

    def _wants_grad(self, input: torch.Tensor) -> bool:
        return torch.is_grad_enabled() and (input.requires_grad or any(p.requires_grad for p in self.parameters()))

    def _supports_anthropic(self, input: torch.Tensor) -> str | None:
        from miniworld_engine.modules import _anthropic

        if input.dtype not in (torch.bfloat16, torch.float32):
            return f"dtype {input.dtype}: the backend is verified for bfloat16 / float32 activations"
        if len(self.normalized_shape) != 1:
            return "the kernel normalises the last dimension only"
        if self.weight is not None and self.weight.dtype != torch.float32:
            return "the affine must be fp32 (the module's pinned layout)"
        if self._wants_grad(input):  # training: Protenix's fast_layernorm (forward + backward), built for this environment
            if getattr(input.device, "type", None) != "cuda":
                return "the release's kernels run on CUDA tensors only"
            return _fastln_reason()
        for module, path in self._KIT_LN:
            why = _anthropic.unavailable(module, input.device, path=path)
            if why is not None:
                return why
        return None

    def _forward_anthropic(self, input: torch.Tensor) -> torch.Tensor:
        from miniworld_engine.modules import _anthropic

        if self._wants_grad(input):
            return _FastLayerNormFunction.apply(input, self.weight, self.bias, tuple(self.normalized_shape), self.eps)
        (rows, rows_path), (stock, stock_path) = self._KIT_LN
        y = _anthropic.load(rows, path=rows_path).layer_norm_rows(input, self.normalized_shape, self.weight, self.bias, self.eps)
        if y is None:  # a row wider than the kit's 512-byte blocking: xfold's own one-row-per-program launch of the same kernel body
            y = _anthropic.load(stock, path=stock_path).LayerNormTritonFunc.apply(
                input, self.normalized_shape, self.weight, self.bias, self.eps
            )
        return y

    # ------------------------------------------------------------------------------------------------------------- "cueq"
    # NVIDIA cuEquivariance's fused LayerNorm ``cuequivariance_ops_torch.layer_norm_transpose`` (Triton, forward and backward: dx and the
    # affine gradients) in its "nd->nd" layout: the rows [M, D], the affine handed to the op in the activation's dtype (``.to(input.dtype)``:
    # the master layout's fp32 gamma / beta cast like the engine's other library calls; the gradient reaches the fp32 parameters through the
    # cast); no offset -> zeros, no affine -> elementwise_affine=False.
    def _supports_cueq(self, input: torch.Tensor) -> str | None:
        from miniworld_engine.modules import _cueq

        why = _cueq.unavailable(input.device)
        if why is not None:
            return why
        if len(self.normalized_shape) != 1:
            return "the op normalises the last dimension only"
        if input.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return f"dtype {input.dtype} (bfloat16 / float16 / float32 only)"
        if input.numel() == 0 or input.shape[-1] != self.normalized_shape[0]:
            return f"input {tuple(input.shape)} for normalized_shape {tuple(self.normalized_shape)}"
        return None

    def _forward_cueq(self, input: torch.Tensor) -> torch.Tensor:
        import importlib

        lnt = importlib.import_module("cuequivariance_ops_torch").layer_norm_transpose
        d = self.normalized_shape[0]
        affine = self.weight is not None
        w = self.weight.to(input.dtype) if affine else input.new_ones(d)
        b = self.bias.to(input.dtype) if self.bias is not None else input.new_zeros(d)
        y = lnt(input.reshape(-1, d), w, b, eps=self.eps, elementwise_affine=affine, layout="nd->nd")
        return y.reshape(input.shape)

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
        if a100(input):  # A100: the sm_80 row kernels (kernels/layernorm/cuda/sm80.py)
            from miniworld_engine.kernels.layernorm.cuda import sm80

            return self._supports_engine(input) or sm80.refusal(input, self.weight, self.bias)
        from miniworld_engine.kernels.layernorm.cuda.general import supported

        why = h100_unsupported(input) or self._supports_engine(input)
        if why is not None:
            return why
        if not supported(input):
            return f"width {input.shape[-1]} ({input.dtype}) is beyond the general CUDA LayerNorm's builds"
        return None

    def _forward_cuda(self, input: torch.Tensor) -> torch.Tensor:
        if a100(input):
            from miniworld_engine.kernels.layernorm.cuda import sm80

            return sm80.layernorm(input, self.weight, self.bias, self.eps)
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
        if a100(x):  # A100: the RMS mode of the sm_80 row kernels (kernels/rmsnorm/cuda/sm80.py)
            from miniworld_engine.kernels.rmsnorm.cuda import sm80

            return self._supports_engine(x) or sm80.refusal(x, self.weight)
        from miniworld_engine.kernels.layernorm.cuda.general import supported

        why = h100_unsupported(x) or self._supports_engine(x)
        if why is not None:
            return why
        if not supported(x):
            return f"width {x.shape[-1]} ({x.dtype}) is beyond the general CUDA RMSNorm's builds"
        return None

    def _forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        if a100(x):
            from miniworld_engine.kernels.rmsnorm.cuda import sm80

            return sm80.rmsnorm(x, self.weight, self.effective_eps(x.dtype))
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

    def __init__(
        self, d_hidden: int, d_cond: int, *, eps: float = 1e-5, dtype: torch.dtype = torch.float32, backend: str = "pytorch"
    ) -> None:
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
