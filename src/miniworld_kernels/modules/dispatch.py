"""Central backend dispatch for the model-level modules.

This is the single home for two things the modules used to each re-implement:

1. **Public vs internal backend split.**
   :class:`~miniworld_kernels.modules.exceptions.ImplementationType` is the
   *public* module option. The user-facing values are:

     * ``PYTORCH``  — the readable reference (also the autograd oracle).
     * ``MINIWORLD`` — "ours (auto)": pick the fastest correct kernel for the
       running GPU / shape / mode. This is what production should select.
     * ``CUEQUIVARIANCE`` — the vendor baseline (comparison only).

   The concrete *technology* backends (``TRITON`` / ``CUTE`` / ``CUDA``) remain
   accepted for explicit per-impl benchmarking, but they are **internal**: the
   module never branches on ``ImplementationType`` directly in ``forward`` — it
   resolves to a :class:`KernelBackend` here and dispatches on that.

2. **GPU-architecture policy.**
   The "B200 (sm_100) → X, H100 (sm_90) → Y" knowledge used to live inline in
   four different files (trimul dispatch, transition fused, layernorm_linear, the
   calibration gates). The capability helpers below are now the single source of
   truth for the *module layer*. (Kernel-internal shape/arch sub-dispatch — e.g.
   transition's b2b-vs-cute-vs-triton-by-d, the cute sm100-vs-sm90 kernel pick —
   still lives next to those kernels; this module only resolves the *family*.)

Selecting ``MINIWORLD`` can only ever pick a *correct* backend: every resolver
returns a backend the op actually supports for the given inputs, falling back to
a slower-but-valid path (ultimately PYTORCH) rather than a wrong kernel. A wrong
*performance* pick is at worst slower; it is never incorrect.

Debug/manual pins (env) stay honoured for A/B work but never change the default
policy silently — see the per-op resolvers.
"""

from __future__ import annotations

import os
import warnings

import torch

from miniworld_kernels.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)

try:  # Py3.11+: StrEnum; fall back to (str, Enum) for older interpreters.
    from enum import StrEnum as _StrEnum
except ImportError:  # pragma: no cover
    from enum import Enum

    class _StrEnum(str, Enum):  # type: ignore[no-redef]
        pass


class KernelBackend(_StrEnum):
    """Concrete technology backend a module actually executes.

    Internal to the dispatch layer: :func:`resolve_*` turns the public
    ``ImplementationType`` (including the ``MINIWORLD`` auto family) into one of
    these, and modules branch their private ``_backend_forward`` on it.
    """

    PYTORCH = "pytorch"
    TRITON = "triton"
    CUDA = "cuda"
    CUTE = "cute"
    CUEQUIVARIANCE = "cuequivariance"


# --------------------------------------------------------------------------- #
# GPU-architecture policy (single source of truth for the module layer)
# --------------------------------------------------------------------------- #
def capability(device: torch.device | None = None) -> tuple[int, int]:
    """CUDA compute capability ``(major, minor)`` for ``device`` (current if None).

    Returns ``(0, 0)`` when CUDA is unavailable so callers can treat "no GPU" as
    "pre-Hopper" and fall through to the portable paths.
    """
    if not torch.cuda.is_available():
        return (0, 0)
    idx = None
    if device is not None and getattr(device, "type", None) == "cuda":
        idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()
    return torch.cuda.get_device_capability(idx)


def is_sm100(device: torch.device | None = None) -> bool:
    """True on Blackwell / B200 (sm_100, major == 10)."""
    return capability(device)[0] >= 10  # noqa: PLR2004


def is_sm90plus(device: torch.device | None = None) -> bool:
    """True on Hopper (sm_90) and newer — i.e. archs with a supported cute GEMM."""
    return capability(device)[0] >= 9  # noqa: PLR2004


def is_sm90(device: torch.device | None = None) -> bool:
    """True on Hopper *exactly* (sm_9x, major == 9) — NOT Blackwell.

    Distinct from :func:`is_sm90plus` (>= 9): the hand-CUDA b2b and the quack cute
    ``transition_fused`` kernels are Hopper WGMMA/TMA (sm_90a) code that neither
    builds/launches on pre-Hopper (sm_80 / A100) nor on Blackwell (sm_100). Guards
    that route to those kernels must use this, not ``not is_sm100`` — otherwise
    pre-Hopper GPUs (which are also "not sm100") get routed into a Hopper-only
    kernel and crash. A100 falls through to the portable Triton path instead.
    """
    return capability(device)[0] == 9  # noqa: PLR2004


# --------------------------------------------------------------------------- #
# Public ImplementationType -> internal KernelBackend
# --------------------------------------------------------------------------- #
_CONCRETE = {
    ImplementationType.PYTORCH: KernelBackend.PYTORCH,
    ImplementationType.TRITON: KernelBackend.TRITON,
    ImplementationType.CUDA: KernelBackend.CUDA,
    ImplementationType.CUTE: KernelBackend.CUTE,
    ImplementationType.CUEQUIVARIANCE: KernelBackend.CUEQUIVARIANCE,
}


def to_kernel_backend(impl: ImplementationType) -> KernelBackend:
    """Map an already-concrete ``ImplementationType`` to its ``KernelBackend``.

    ``MINIWORLD`` is *not* concrete — call the op's ``resolve_*`` instead. Passing
    it here is a programming error (raises ``InvalidImplementationError``).
    """
    try:
        return _CONCRETE[impl]
    except KeyError:
        raise InvalidImplementationError(impl) from None


def _coerce(impl: ImplementationType | str) -> ImplementationType:
    return ImplementationType(impl)


# --------------------------------------------------------------------------- #
# Per-op resolvers.  Each maps the PUBLIC option (incl. MINIWORLD auto) to the
# concrete KernelBackend the module should run.  These encode ONLY the family
# choice made at the module layer; the exact kernel within a family (shape / arch
# sub-dispatch) is chosen next to the kernels, unchanged.
# --------------------------------------------------------------------------- #
def resolve_transition(
    impl: ImplementationType | str, device: torch.device | None = None
) -> KernelBackend:
    """Transition: no cuequivariance kernel, so MINIWORLD (and the CUEQUIVARIANCE
    request, which shares the same path) resolves to the TRITON family, whose
    internal dispatch picks hand-CUDA b2b (d in {128,256}, n==4) / cute split
    (d>=512) / triton fused per shape+mode."""
    impl = _coerce(impl)
    if impl == ImplementationType.MINIWORLD:
        return KernelBackend.TRITON
    return to_kernel_backend(impl)


def resolve_triangle_attention(
    impl: ImplementationType | str, device: torch.device | None = None
) -> KernelBackend:
    """TriangleAttention: the only tensor-core kernels are the triton ones (which
    themselves per-GPU dispatch fused vs split), so MINIWORLD -> TRITON."""
    impl = _coerce(impl)
    if impl == ImplementationType.MINIWORLD:
        return KernelBackend.TRITON
    return to_kernel_backend(impl)


def resolve_adaptive_layernorm(
    impl: ImplementationType | str, device: torch.device | None = None
) -> KernelBackend:
    """AdaptiveLayerNorm: only PYTORCH + TRITON exist, MINIWORLD -> TRITON."""
    impl = _coerce(impl)
    if impl == ImplementationType.MINIWORLD:
        return KernelBackend.TRITON
    return to_kernel_backend(impl)


def resolve_conditioned_transition(
    impl: ImplementationType | str, device: torch.device | None = None
) -> KernelBackend:
    """ConditionedTransition: TRITON is the only fused family (the inference
    d_hidden sub-dispatch lives in the kernel), MINIWORLD -> TRITON."""
    impl = _coerce(impl)
    if impl == ImplementationType.MINIWORLD:
        return KernelBackend.TRITON
    return to_kernel_backend(impl)


def resolve_augmented_attention(
    impl: ImplementationType | str, device: torch.device | None = None
) -> KernelBackend:
    """AugmentedAttention: only PYTORCH + TRITON exist, MINIWORLD -> TRITON."""
    impl = _coerce(impl)
    if impl == ImplementationType.MINIWORLD:
        return KernelBackend.TRITON
    return to_kernel_backend(impl)


def resolve_layernorm(
    impl: ImplementationType | str, device: torch.device | None = None
) -> KernelBackend:
    """LayerNorm (primitives): MINIWORLD is our auto-routing LayerNorm
    (``layernorm_kernel``: fused triton forward + per-shape auto-dispatched
    backward). It is a *distinct* kernel from the legacy vendored ``triton_layernorm``
    (the TRITON/CUEQUIVARIANCE path), so MINIWORLD maps to the CUDA-family entry
    (``layernorm_kernel``), matching the prior ``{MINIWORLD, CUDA}`` grouping."""
    impl = _coerce(impl)
    if impl == ImplementationType.MINIWORLD:
        return KernelBackend.CUDA
    return to_kernel_backend(impl)


def resolve_triangle_multiplication(
    impl: ImplementationType | str, device: torch.device | None = None
) -> KernelBackend:
    """TriangleMultiplication: MINIWORLD is an *architecture capability* choice
    (not a shape crossover):

      * sm_100 (B200) / sm_90 (H100) and other cute-capable archs -> CUTE (the
        exact cute out_layout — ``bdll_sm100`` vs ``bdll_direct`` — is picked by
        :func:`trimul_out_layout`).
      * pre-Hopper (no tcgen05 / no supported cute GEMM) -> TRITON.

    Env override ``MINIWORLD_TRIMUL_IMPL`` (debug / manual pin) still wins. A wrong
    layout pick can only be slower, never incorrect (same bf16 in / fp32 acc math).
    """
    impl = _coerce(impl)
    if impl != ImplementationType.MINIWORLD:
        return to_kernel_backend(impl)
    override = os.environ.get("MINIWORLD_TRIMUL_IMPL")
    if override:
        return to_kernel_backend(ImplementationType(override.strip().lower()))
    return KernelBackend.CUTE if is_sm90plus(device) else KernelBackend.TRITON


# --------------------------------------------------------------------------- #
# dtype correctness-guard
#
# The fused kernels (triton / cute / hand-CUDA) are bf16-only. If a module is
# asked to run a fast backend on an input dtype the kernel can't handle, we must
# fall back to a dtype-agnostic path (the pytorch reference) rather than hand the
# wrong dtype to the kernel — a wrong backend for the dtype is a correctness bug,
# a slower one is not. PYTORCH / CUEQUIVARIANCE are treated as dtype-agnostic
# (the reference handles any dtype; an explicit cuequiv request is the caller's
# own choice). Extend _FAST_KERNEL_DTYPES if a kernel gains fp16/fp32 support.
# --------------------------------------------------------------------------- #
_FAST_KERNEL_DTYPES = frozenset({torch.bfloat16})
_DTYPE_AGNOSTIC = frozenset({KernelBackend.PYTORCH, KernelBackend.CUEQUIVARIANCE})
_dtype_warned: set[tuple[str, str, str]] = set()


def backend_supports_dtype(backend: KernelBackend, dtype: torch.dtype) -> bool:
    """True if ``backend`` can run inputs of ``dtype``. Fused kernels are bf16-only."""
    if backend in _DTYPE_AGNOSTIC:
        return True
    return dtype in _FAST_KERNEL_DTYPES


def guard_dtype(
    backend: KernelBackend, dtype: torch.dtype, *, op: str
) -> KernelBackend:
    """Return ``backend`` if it can run ``dtype``, else fall back to PYTORCH with a
    one-time warning. Used at the top of a module's forward so an unsupported dtype
    degrades to the (correct, slower) reference instead of crashing in a bf16 kernel.
    """
    if backend_supports_dtype(backend, dtype):
        return backend
    key = (op, backend.value, str(dtype))
    if key not in _dtype_warned:
        _dtype_warned.add(key)
        warnings.warn(
            f"{op}: the '{backend.value}' fast path is bf16-only but got {dtype}; "
            f"falling back to the PyTorch reference (slower). Cast inputs to "
            f"torch.bfloat16 to use the fused kernels.",
            stacklevel=3,
        )
    return KernelBackend.PYTORCH


def trimul_out_layout(device: torch.device | None = None) -> str:
    """cute tm1 ``out_layout`` for the running GPU:
    sm_100 -> ``bdll_sm100`` (our tcgen05 gate GEMM); else -> ``bdll_direct``
    (quack M-major, the H100 / pre-existing path). Env-overridable for debug."""
    override = os.environ.get("MINIWORLD_TRIMUL_OUT_LAYOUT")
    if override:
        return override.strip()
    return "bdll_sm100" if is_sm100(device) else "bdll_direct"
