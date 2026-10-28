"""The architecture blocks' primitives. ``Linear``, ``MPLinear``, ``Dropout``, ``InitType`` and the initialisers are
``modules.primitives``' (one implementation); ``Linear`` / ``MPLinear`` are subclasses only so ``convert_linears_to_mp`` can tell
the Linears architecture code builds (converted) from the engine modules' (left alone), as it always has. ``LayerNorm`` stays its
own: it routes on the architecture ``ImplementationType``."""

import numbers

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Size
from torch.nn.parameter import Parameter

from miniworld_engine.architecture.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)
from miniworld_engine.modules import primitives as _modules_primitives
from miniworld_engine.modules.primitives import (  # noqa: F401 -- re-exported
    Dropout,
    InitType,
    _Fp32ParamsMixin,
    _trunc_normal_init,
    magnitude_normalize,
)


class Linear(_modules_primitives.Linear):
    """``modules.primitives.Linear``, as built by architecture code (see the module docstring)."""


class MPLinear(_modules_primitives.MPLinear, Linear):
    """``modules.primitives.MPLinear``, as built by architecture code (see the module docstring)."""


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




_shape_t = int | list[int] | Size


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
            from miniworld_engine.modules.dispatch import restricted_card

            if input.is_cuda and restricted_card(input.device):
                # H100 / B200 have no Triton fallback: the general CUDA LayerNorm, or the PyTorch statements for what it does not take
                from miniworld_engine.kernels.layernorm.cuda.general import (
                    layernorm_cuda,
                    supported,
                )

                if len(self.normalized_shape) == 1 and supported(input):
                    return layernorm_cuda(input, self.weight, self.bias, self.eps)
                return super().forward(input.float()).to(input.dtype)
            return ops.layer_norm(
                input,
                self.weight,
                self.bias,
                self.eps,
            )
        raise InvalidImplementationError(self.implementation)
