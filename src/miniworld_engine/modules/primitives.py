# vendored from team-gm psk/benchmark : src/team_gm/modules/primitives.py
import math
from enum import Enum
from functools import partial
from typing import Literal, Union

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Size
from torch.nn.parameter import Parameter

from miniworld_engine import kernels
from miniworld_engine.modules.dispatch import KernelBackend, resolve_layernorm
from miniworld_engine.modules.exceptions import (
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

    # Imported here, not at module scope: scipy is in the `baselines` extra, not the lean
    # core, and this is the only thing in `miniworld_engine.modules` that wants it. At
    # module scope it made `import miniworld_engine.modules.primitives` -- and so every
    # module built on it -- fail outright on a core-only install.
    from scipy.stats import truncnorm  # noqa: PLC0415

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


_shape_t = Union[int, list[int], Size]


class _Fp32ParamsMixin(nn.Module):
    """Pin this module's floating-point params/buffers to fp32 under ANY dtype cast.

    Norm affine params (gamma init 1.0, beta 0.0) stagnate when stored bf16: at value 1.0
    the bf16 ULP is 2**-7 = 0.0078 > Adam's per-step update (~lr = 1.8e-3), so updates
    round back to 1.0 and gamma never trains. Keeping gamma/beta fp32 (trunk stays bf16)
    lets the update land. Overriding ``_apply`` — the funnel for ``.to``/``.bfloat16`` —
    makes the pin survive later bulk ``.to(torch.bfloat16)`` of the parent trunk. The fused
    kernels already upcast the norm weight to fp32 internally and return grads in the
    parameter dtype, so a fp32 weight flows through unchanged.

    Declared as an ``nn.Module`` subclass, not a bare mixin: ``super()._apply`` only exists
    because this class is always mixed in FRONT of one (``LayerNorm(_Fp32ParamsMixin,
    nn.LayerNorm)``), and stating that is what makes the delegation checkable. The MRO is
    unchanged -- ``nn.Module`` still resolves after the concrete norm class."""

    def _apply(self, fn, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        def fp32_fn(t):  # noqa: ANN001, ANN202
            out = fn(t)
            if isinstance(out, torch.Tensor) and out.is_floating_point():
                out = out.to(torch.float32)
            return out

        return super()._apply(fp32_fn, *args, **kwargs)


class LayerNorm(_Fp32ParamsMixin, nn.LayerNorm):
    """A LayerNorm layer with precision control (fp32-pinned affine; see mixin)."""

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
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
        # A new name rather than rebinding the parameter: `_shape_t` admits a bare int, and
        # `tuple(int)` is not a thing -- the int case has to become a 1-tuple first. Test the
        # sequence side rather than `numbers.Integral`, whose ABC registration is invisible
        # to a static checker and leaves an `int & ~Integral` shard in the else branch.
        shape: tuple[int, ...] = (
            tuple(normalized_shape)
            if isinstance(normalized_shape, (list, Size))
            else (int(normalized_shape),)
        )
        self.normalized_shape = shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.implementation = ImplementationType(implementation)
        # Resolve the public option (incl. MINIWORLD auto) to a concrete internal
        # backend once; forward dispatches on this, never on ImplementationType.
        self._backend = resolve_layernorm(self.implementation)
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

    def forward(self, input: Float[torch.Tensor, "*"]) -> Float[torch.Tensor, "*"]:  # noqa: A002
        """Forward pass. Routes on the resolved internal backend (``_backend``)."""
        backend = self._backend
        if backend == KernelBackend.PYTORCH:
            # Compute in fp32 so a fp32-pinned affine weight never dtype-mismatches a bf16
            # activation (and for stability); restore the activation dtype.
            orig_type = input.dtype
            return super().forward(input.float()).to(orig_type)
        if backend in {KernelBackend.TRITON, KernelBackend.CUEQUIVARIANCE}:
            return kernels.triton_layernorm(
                input,
                self.weight,
                self.bias,
                self.eps,
            )
        if backend == KernelBackend.CUDA:
            # miniworld = our auto-routing LayerNorm: forward fused triton (HBM-bound,
            # at the bandwidth wall) + backward auto-dispatched per shape
            # (persistent / partial / atomic). See kernels/layernorm.
            return kernels.layernorm_kernel(
                input,
                self.weight,
                self.bias,
                self.eps,
            )
        raise InvalidImplementationError(self.implementation)


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

    def forward(self, x: Float[torch.Tensor, "*"]) -> Float[torch.Tensor, "*"]:
        """Forward pass."""
        if not self.training or self.p_drop == 0.0:
            return x
        shape = list(x.shape)
        if self.broadcast_dim is not None:
            shape[self.broadcast_dim] = 1
        mask = torch.rand(shape, device=x.device, dtype=x.dtype) > self.p_drop
        return x * mask / (1.0 - self.p_drop)


# --- magnitude-preserving (EDM2) primitives, ported from team-gm (for SWA/DiT) ---
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
