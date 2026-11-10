"""Shared PyTorch building blocks of the modules: the backend mechanism, initialisers, Linear, Dropout, fp32 master weights.

Backends
--------
Every module has one forward per *backend*, and ``"pytorch"`` is the reference: the single source of truth of the op's math. A kernel is
added later as another backend of the same module and is checked against ``"pytorch"`` (same parameters, same inputs). The mechanism is
:class:`BackendModule`, and it is all there is::

    class Transition(BackendModule):
        backends = ("pytorch",)                       # each name has a ``_forward_<name>``; "pytorch" is always there

        def __init__(self, d_hidden, *, backend="pytorch"):
            super().__init__()
            ...
            self.set_backend(backend)                 # validated against ``backends``

        def _forward_pytorch(self, x):                # the reference
            ...

    Transition(128)(x)                                # forward() calls self._forward_<self.backend>(x)
    Transition(128, backend="cuda")                   # ValueError until a "cuda" backend is added to ``backends``

``backend`` is a plain attribute (not in the state dict): a checkpoint loads into any backend, and ``set_backend`` switches it later.

A non-reference backend declares what it covers in ``_supports_<name>(*args, **kwargs) -> str | None`` (None = runs; a string = why not).
``forward`` raises :class:`BackendUnsupported` with that reason instead of falling back: a backend never silently becomes another one.

Every class keeps the public name and the parameter / buffer names of the engine's earlier modules, so their state dicts load unchanged.
"""

import contextlib
import math
from enum import Enum
from functools import partial
from typing import ClassVar, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _trunc_normal_init(
    weights: torch.Tensor,
    scale: float = 1.0,
    a: float = -2,
    b: float = 2,
    fan: Literal["in", "out", "avg"] = "in",
) -> None:
    """Initialize weights ``[fan_out, fan_in]`` with a truncated normal distribution."""
    fan_out, fan_in = weights.shape
    if fan == "in":
        f = fan_in
    elif fan == "out":
        f = fan_out
    elif fan == "avg":
        f = (fan_in + fan_out) / 2
    else:
        msg = f"Invalid fan option: {fan}. Choose from 'in', 'out', 'avg'."
        raise ValueError(msg)

    # Imported lazily so that importing the package does not load scipy.stats.
    from scipy.stats import truncnorm

    scale = scale / max(1, f)
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=weights.numel())
    samples = np.reshape(samples, weights.shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device, dtype=weights.dtype))


class InitType(Enum):
    """Enum for initialization types."""

    DEFAULT = partial(_trunc_normal_init, scale=1.0)
    RELU = partial(_trunc_normal_init, scale=2.0)
    NORMAL = partial(nn.init.kaiming_normal_, nonlinearity="linear")
    GLOROT = partial(nn.init.xavier_uniform_)
    GATING = partial(nn.init.zeros_)
    ZERO = partial(nn.init.zeros_)  # noqa: PIE796 -- a distinct NAME for the same
    # initialiser: GATING says why a tensor is zeroed, ZERO says only that it is.
    ONE = partial(nn.init.ones_)

    def apply(self, data: torch.Tensor) -> None:
        """Apply the initialization to weights and bias."""
        self.value(data)


class BackendUnsupported(RuntimeError):
    """A backend was asked for a call outside its stated range. Raised, never silently replaced by another backend."""


class BackendModule(nn.Module):
    """A module with one forward per backend (see the module docstring): ``forward`` calls ``self._forward_<self.backend>``.

    A subclass lists its backends in ``backends`` (``"pytorch"``, the reference, always among them), defines ``_forward_<name>`` for each,
    and calls :meth:`set_backend` with its constructor's ``backend`` argument.
    """

    #: The backends this class implements, each as a method ``_forward_<name>``. ``"pytorch"`` is the reference.
    backends: tuple[str, ...] = ("pytorch",)
    #: The backend ``forward`` runs.
    backend: str = "pytorch"
    #: Human-managed version of the class's results: bump it when a change to the class's math, its reference or a backend adapter can
    #: change an accuracy or speed result (not for a refactor or a comment). A subclass inherits it; the judge
    #: (miniworld_engine.judge) owes a new measurement of every case of a class whose version changed.
    version: int = 1
    #: Human-managed version of ONE backend's adapter in this class (backend name -> version, missing = 1): bump a backend's entry when
    #: its adapter changes that backend's results only (e.g. an external library wired to other kernels); the judge then owes the cases
    #: of that backend alone. A subclass inherits it.
    backend_versions: ClassVar[dict[str, int]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        missing = [name for name in ("pytorch", *cls.backends) if not callable(getattr(cls, f"_forward_{name}", None))]
        if missing:
            raise TypeError(f"{cls.__name__}: no _forward_{missing[0]} (backends {cls.backends}; 'pytorch' is required)")

    def set_backend(self, backend: str = "pytorch"):
        """Select the backend ``forward`` runs; ``ValueError`` if this class has no such backend. Returns ``self``."""
        if backend not in self.backends:
            raise ValueError(f"{type(self).__name__} has no backend {backend!r}; available: {', '.join(self.backends)}")
        self.backend = backend
        return self

    def unsupported(self, *args, **kwargs) -> str | None:
        """Why the selected backend cannot run this call (a short reason), or ``None`` if it can.

        A non-reference backend states its own range in ``_supports_<name>(*args, **kwargs) -> str | None`` (shapes, dtypes, modes, device, the
        upstream library being importable); without one it claims every call. ``"pytorch"`` runs everything."""
        check = getattr(self, f"_supports_{self.backend}", None)
        return None if check is None else check(*args, **kwargs)

    def forward(self, *args, **kwargs):
        reason = self.unsupported(*args, **kwargs) if self.backend != "pytorch" else None
        if reason is not None:
            raise BackendUnsupported(f"{type(self).__name__} backend {self.backend!r}: {reason}")
        return getattr(self, f"_forward_{self.backend}")(*args, **kwargs)

    def extra_repr(self) -> str:
        inner = super().extra_repr()
        return f"{inner}, backend={self.backend}" if inner else f"backend={self.backend}"

    def make_inputs(
        self,
        *,
        length: int,
        samples: int,
        depth: int | None,
        dtype: torch.dtype,
        device,
        generator: torch.Generator,
        stream: str | None = None,
    ) -> dict:
        """The forward kwargs of one measurement call (``self(**inputs)``): random activations of the right shapes in ``dtype``, key masks
        with some padding (:func:`key_mask`), and any structural inputs. ``length`` / ``samples`` / ``depth`` / ``stream`` are the
        requirements row's (``depth`` the MSA depth, ``stream`` e.g. ``token_pair``: the layout of a row-wise op's input). The only
        module-specific part of the measurement code (``miniworld_engine.measure``); it never changes the forward math."""
        raise NotImplementedError(f"{type(self).__name__} has no make_inputs: a requirement class defines its measurement inputs")


# ------------------------------------------------------------------------------------------------------------ compute precision floor
def wide(*refs: torch.Tensor | torch.dtype) -> torch.dtype:
    """The "at least fp32" compute dtype of ``refs`` (tensors or dtypes): float32 for bf16 / fp16 / fp32 (and integer / bool) operands,
    float64 when any operand is float64. A statement the modules pin to fp32 for low-precision inputs uses this, never a bare
    ``.float()``: bf16 / fp16 / fp32 calls compute exactly as in fp32, and a float64 call (the accuracy gold) stays float64."""
    out = torch.float32
    for r in refs:
        out = torch.promote_types(out, r.dtype if isinstance(r, torch.Tensor) else r)
    return out


# --------------------------------------------------------------------------------------------------- measurement inputs (make_inputs)
def random_input(shape, *, dtype: torch.dtype, device, generator: torch.Generator, scale: float = 1.0) -> torch.Tensor:
    """Standard-normal values (times ``scale``) drawn in fp32 from ``generator`` (on ``device``), rounded to ``dtype``."""
    return (scale * torch.randn(tuple(shape), device=device, generator=generator)).to(dtype)


def key_mask(rows: int, length: int, *, device, pad: int | None = None) -> torch.Tensor:
    """A bool key mask ``[rows, length]``: real positions first, the last ``pad`` (default ``length // 8``) padding."""
    pad = length // 8 if pad is None else min(pad, length - 1)
    return (torch.arange(length, device=device) < length - pad).expand(rows, length).contiguous()


def stream_shape(stream: str | None, *, samples: int, length: int, depth: int | None, width: int) -> tuple[int, ...]:
    """The activation shape of a row-wise op (LayerNorm, Transition) on a requirements stream: ``token_pair`` ``[S, L, L, w]``, an MSA
    (``msa_token`` or a depth) ``[S, depth, L, w]``, any other ``[S, L, w]``."""
    if stream == "token_pair":
        return (samples, length, length, width)
    if stream == "msa_token" or depth:
        return (samples, depth or 1, length, width)
    return (samples, length, width)


def pair_inputs(d_pair: int, *, length: int, samples: int, dtype, device, generator) -> dict:
    """``pair`` ``[S, L, L, d_pair]`` and its key ``mask`` ``[S, L]``: the inputs of every pair-track update."""
    return {"pair": random_input((samples, length, length, d_pair), dtype=dtype, device=device, generator=generator),
            "mask": key_mask(samples, length, device=device)}


class MasterWeights(nn.Module):
    """fp32 master weights under bf16 compute: ``module.set_master_weights()`` (or the constructor's ``master_weights=True``).

    The parameters stay fp32 (the master: what the optimizer updates and an fp32 checkpoint loads into) while the module is called with bf16
    activations; a bulk ``.bfloat16()`` / ``.to(torch.bfloat16)`` / ``.half()`` of a model that contains it no longer rounds them
    (the same pin ``LayerNorm`` has for its affine). On a CUDA device a bf16 activation next to fp32 master matrices runs the module's
    statements under bf16 autocast (:func:`master_autocast`), or a projection as a bf16 GEMM whose weight / bias gradients come back fp32 and
    unrounded onto the masters (:func:`master_linear`).

    ``master_weights`` is False by default: a module cast to bf16 keeps bf16 matrices. ``set_master_weights(False)`` only clears the pin (it
    does not cast down).
    """

    #: ``True`` after :meth:`set_master_weights`: floating parameters / buffers stay fp32 under a cast to bf16 / fp16.
    master_weights: bool = False

    def set_master_weights(self, enabled: bool = True):
        """Pin (or unpin) the fp32 masters of this module and every :class:`MasterWeights` below it; ``True`` also upcasts parameters that are
        not fp32 yet (in place: the same ``Parameter`` objects, an optimizer built earlier stays valid). Returns ``self``."""
        for module in self.modules():
            if isinstance(module, MasterWeights):
                module.master_weights = bool(enabled)
        if enabled:
            self.float()
        return self

    def _apply(self, fn, *args, **kwargs):
        if self.master_weights:
            inner = fn

            def pinned(t):
                out = inner(t)
                if isinstance(out, torch.Tensor) and out.dtype in (torch.bfloat16, torch.float16):
                    out = out.to(torch.float32)
                return out

            fn = pinned
        return super()._apply(fn, *args, **kwargs)


def master_autocast(x: torch.Tensor, module: nn.Module):
    """Context for a module's statements: a bf16 CUDA activation next to fp32 master matrices (the mixed-precision layout) runs them under bf16
    autocast, as the caller's own autocast would; anything else runs as is."""
    if (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and not torch.is_autocast_enabled("cuda")
        and any(p.dtype == torch.float32 for p in module.parameters() if p.ndim >= 2)
    ):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class _MasterLinear(torch.autograd.Function):
    """``F.linear`` with bf16 compute copies of fp32 masters: the GEMMs in bf16, the weight / bias gradients fp32 onto the masters."""

    @staticmethod
    def forward(ctx, x, weight, bias, compute_weight, compute_bias):
        ctx.save_for_backward(x, weight, bias, compute_weight, compute_bias)
        with torch.autocast("cuda", enabled=False):
            return F.linear(x, compute_weight, compute_bias)

    @staticmethod
    def backward(ctx, dy):
        x, _weight, bias, compute_weight, _ = ctx.saved_tensors
        x2 = x.reshape(-1, x.shape[-1])
        dy2 = dy.reshape(-1, dy.shape[-1])
        with torch.autocast("cuda", enabled=False):
            dx = torch.mm(dy2, compute_weight).reshape(x.shape)
            dw = torch.mm(dy2.t(), x2, out_dtype=torch.float32)
            db = dy2.sum(0, dtype=torch.float32) if bias is not None else None
        return dx, dw, db, None, None


def master_linear(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """``module(x)`` where a bf16 CUDA activation meets fp32 master parameters: bf16 GEMM, fp32 (unrounded) weight / bias gradients.
    Any other combination is the module itself."""
    w = module.weight
    if x.is_cuda and x.dtype == torch.bfloat16 and w.dtype == torch.float32:
        b = module.bias
        return _MasterLinear.apply(x, w, b, w.to(torch.bfloat16), None if b is None else b.to(torch.bfloat16))
    return module(x)


class Linear(nn.Linear):
    """A Linear layer with built-in nonstandard initializations.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool
        If set to `False`, the layer will not learn an additive bias.
    dtype : torch.dtype, optional
        The desired data type of the layer.
    init : str
        The initializer to use. Supported options are:
        - "default": LeCun fan-in truncated normal initialization.
        - "relu": He initialization using a truncated normal distribution.
        - "normal": Normal initialization with standard deviation `1/sqrt(fan_in)`.
        - "glorot": Fan-average Glorot uniform initialization.
        - "gating": Weights=0, Bias=1.
        - "zero": Weights=0, Bias=0.
        - "one": Weights=1, Bias=0.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        dtype: torch.dtype | None = None,
        init: Literal[
            "default",
            "relu",
            "normal",
            "glorot",
            "gating",
            "zero",
            "one",
        ] = "default",
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, dtype=dtype)
        self.init = init
        InitType[init.upper()].apply(self.weight)
        if self.bias is not None:
            if init == "gating":
                InitType["ONE"].apply(self.bias)
            if init in {"zero", "one"}:
                InitType["ZERO"].apply(self.bias)


class Dropout(nn.Module):
    """Dropout layer that can drop entire rows or columns.

    Parameters
    ----------
    broadcast_dim : int, optional
        Dimension to broadcast the dropout mask over. If None, standard dropout is
        applied.
    p_drop : float
        Probability of dropping a unit.

    """

    def __init__(
        self,
        broadcast_dim: int | None = None,
        p_drop: float = 0.15,
    ) -> None:
        super().__init__()
        self.broadcast_dim = broadcast_dim
        self.p_drop = p_drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if not self.training or self.p_drop == 0.0:
            return x
        shape = list(x.shape)
        if self.broadcast_dim is not None:
            shape[self.broadcast_dim] = 1
        mask = torch.rand(shape, device=x.device, dtype=x.dtype) > self.p_drop
        return x * mask / (1.0 - self.p_drop)


# --- magnitude-preserving (EDM2) primitives (for SWA/DiT) ---
def magnitude_normalize(
    x: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Per-output-channel L2 normalization (EDM2 ``normalize``, Algorithm 1).

    Normalizes every output-channel slice (``dim 0``) of a weight tensor to unit
    norm. ``alpha = sqrt(numel_per_channel)`` keeps ``eps`` scale-relative so the
    behaviour is shape-agnostic. Ref: Karras et al., arXiv:2312.02696.
    """
    dim = list(range(1, x.ndim))
    n = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    alpha = math.sqrt(n.numel() / x.numel())
    return x / torch.add(eps, n, alpha=alpha)


class MPLinear(Linear):
    """Magnitude-preserving Linear with EDM2 forced weight normalization.

    Two pieces (Karras et al., arXiv:2312.02696, §2.2-2.3, Algorithm 1):

    * **forced weight normalization** -- during training the *stored* weight is
      projected back onto the unit hypersphere at the start of each forward
      (``copy_``), pinning ``||w||`` so the effective learning rate stays equal
      across layers and uncontrolled magnitude growth is eliminated;
    * **on-use weight normalization** -- the forward always uses
      ``normalize(w) / sqrt(fan_in)``, so output magnitude is preserved and the
      loss gradient is projected onto the tangent plane.

    The projection happens before the normalized weight participates in the
    autograd graph.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        dtype: torch.dtype | None = None,
        init: Literal["default", "relu", "normal", "glorot", "one"] = "normal",
    ) -> None:
        if init in {"zero", "gating"}:
            msg = f"MPLinear is incompatible with init={init!r} (cannot normalize a zero/gating weight)."
            raise ValueError(msg)
        super().__init__(in_features, out_features, bias=bias, dtype=dtype, init=init)
        with torch.no_grad():
            self.weight.copy_(magnitude_normalize(self.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with forced + on-use weight normalization."""
        if self.training:
            with torch.no_grad():
                self.weight.copy_(magnitude_normalize(self.weight))
        weight = magnitude_normalize(self.weight) / math.sqrt(self.in_features)
        return nn.functional.linear(x, weight, self.bias)
