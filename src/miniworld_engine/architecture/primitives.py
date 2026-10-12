import math
import numbers
from enum import Enum
from functools import partial
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from scipy.stats import truncnorm
from torch import Size
from torch.nn.parameter import Parameter

from miniworld_engine.architecture.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)


def _trunc_normal_init(
    weights: Float[torch.Tensor, "fan_out fan_in"],
    scale: float = 1.0,
    a: float = -2,
    b: float = 2,
    fan: Literal["in", "out", "avg"] = "in",
) -> None:
    """Initialize weights with truncated normal distribution."""
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
    ZERO = partial(nn.init.zeros_)  # noqa: PIE796
    ONE = partial(nn.init.ones_)

    def apply(self, data: torch.Tensor) -> None:
        """Apply the initialization to weights and bias."""
        self.value(data)


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

    * **forced weight normalization** — during training the *stored* weight is
      projected back onto the unit hypersphere at the start of each forward
      (``copy_``), pinning ``||w||`` so the effective learning rate stays equal
      across layers and uncontrolled magnitude growth is eliminated;
    * **on-use weight normalization** — the forward always uses
      ``normalize(w) / sqrt(fan_in)``, so output magnitude is preserved and the
      loss gradient is projected onto the tangent plane.

    The projection happens before the normalized weight participates in the
    autograd graph; small MLP checks show this compiles cleanly with Inductor.
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


def convert_linears_to_mp(
    root: nn.Module,
    exclude_substrings: tuple[str, ...] = (),
) -> int:
    """Recursively replace ``Linear`` children with ``MPLinear`` (full-MP).

    For every ``Linear`` whose qualified name does NOT contain any string in
    ``exclude_substrings`` (e.g. ``"final_denoising"``), swap it for an
    ``MPLinear`` of the same shape (unit-Gaussian re-init; EDM2 on-use
    normalization makes the init magnitude irrelevant). ``init="zero"`` /
    ``"gating"`` Linears are turned into MP layers too (they cannot stay
    zero-init under MP). Bias is preserved if present.

    Any module exposing ``self.implementation`` (e.g. ``Transition``) is forced
    onto the PyTorch path, because its miniworld_engine kernel consumes raw ``.weight`` and
    would bypass MPLinear's on-use normalization. Returns the number of layers
    converted. Intended for the diffusion module only (never the frozen trunk).
    """
    converted = 0
    for name, child in list(root.named_modules()):
        if hasattr(child, "implementation"):
            child.implementation = ImplementationType.PYTORCH
    for name, child in list(root.named_modules()):
        if not isinstance(child, Linear) or isinstance(child, MPLinear):
            continue
        if any(s in name for s in exclude_substrings):
            continue
        parent = root
        *path, attr = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        new = MPLinear(
            child.in_features,
            child.out_features,
            bias=child.bias is not None,
            dtype=child.weight.dtype,
            init="normal",
        )
        setattr(parent, attr, new)
        converted += 1
    return converted


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

    def forward(self, x: Float[torch.Tensor, "*"]) -> Float[torch.Tensor, "*"]:
        """Forward pass."""
        if not self.training or self.p_drop == 0.0:
            return x
        shape = list(x.shape)
        if self.broadcast_dim is not None:
            shape[self.broadcast_dim] = 1
        mask = torch.rand(shape, device=x.device, dtype=x.dtype) > self.p_drop
        return x * mask / (1.0 - self.p_drop)


_shape_t = int | list[int] | Size


class _Fp32ParamsMixin:
    """Pin this module's floating-point params/buffers to fp32 under ANY dtype cast.

    Norm affine params (LayerNorm/RMSNorm gamma initialised to 1.0, beta to 0.0) stagnate
    when stored bf16: at value 1.0 the bf16 ULP is 2**-7 = 0.0078, larger than Adam's
    per-step update (~lr = 1.8e-3), so every update rounds back to 1.0 and gamma never
    trains. Keeping gamma/beta in fp32 (while the surrounding trunk stays bf16) lets the
    optimiser update land. Overriding ``_apply`` — the funnel for ``.to``/``.bfloat16``/
    ``.cuda`` — makes the fp32 pin survive later bulk ``.to(torch.bfloat16)`` casts of the
    parent trunk module. Kernels already upcast the norm weight to fp32 internally, so a
    fp32 weight flows through the fused path (and its grad returns fp32) unchanged."""

    def _apply(self, fn, *args, **kwargs):
        def fp32_fn(t):
            out = fn(t)
            if isinstance(out, torch.Tensor) and out.is_floating_point():
                out = out.to(torch.float32)
            return out

        return super()._apply(fp32_fn, *args, **kwargs)  # type: ignore[misc]


class LayerNorm(_Fp32ParamsMixin, nn.LayerNorm):
    """LayerNorm with optional miniworld_engine implementation and dtype preservation.

    Affine params are fp32-pinned (see :class:`_Fp32ParamsMixin`)."""

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        *,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            **factory_kwargs,
        )
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.implementation = implementation
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()
        # fp32-pin affine params from construction, even when a bf16 dtype is requested
        # (norm gamma stagnates in bf16 at 1.0; see _Fp32ParamsMixin). _apply keeps them
        # fp32 through later .to() casts; this covers the never-cast case.
        with torch.no_grad():
            if self.weight is not None and self.weight.is_floating_point():
                self.weight.data = self.weight.data.float()
            if self.bias is not None and self.bias.is_floating_point():
                self.bias.data = self.bias.data.float()

    def forward(self, input: Float[torch.Tensor, "*"]) -> Float[torch.Tensor, "*"]:
        """Forward pass."""
        # Match the engine dispatch policy: cuEquivariance supplies no standalone
        # LayerNorm kernel, so its model backend uses the PyTorch reference here.
        if self.implementation in {
            ImplementationType.PYTORCH,
            ImplementationType.CUEQUIVARIANCE,
        }:
            orig_type = input.dtype
            # Compute in fp32 so a fp32-pinned affine weight never dtype-mismatches a bf16
            # activation (and for numerical stability); restore the activation dtype.
            output = super().forward(input.float())
            return output.to(orig_type)
        if self.implementation == ImplementationType.MINIWORLD_ENGINE:
            from miniworld_engine import ops

            return ops.layer_norm(
                input,
                self.weight,
                self.bias,
                self.eps,
            )
        raise InvalidImplementationError(self.implementation)
