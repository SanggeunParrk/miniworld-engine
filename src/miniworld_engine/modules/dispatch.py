"""Central backend dispatch for the model-level modules.

This is the single home for two things the modules used to each re-implement:

1. **Public vs internal backend split.**
   :class:`~miniworld_engine.modules.exceptions.ImplementationType` is the
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

import sys
import warnings
from collections.abc import Callable
from enum import Enum

import torch

from miniworld_engine.modules.exceptions import (
    ImplementationType,
    InvalidImplementationError,
)

# Py3.11+ has enum.StrEnum; 3.10 (this package's floor) does not. Guarded on sys.version_info
# rather than try/except ImportError because a type checker can follow the first and not the
# second: with the try form, `_StrEnum` was Unknown on the checked 3.10 target, so `KernelBackend`
# was Unknown, so every function returning one was an invalid-return-type -- six findings from
# one import. Same runtime behaviour, legible to both.
if sys.version_info >= (3, 11):
    from enum import StrEnum as _StrEnum
else:  # pragma: no cover -- exercised only on 3.10
    class _StrEnum(str, Enum):
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
# Train-vs-inference kernel choice (single source of truth for the module layer)
# --------------------------------------------------------------------------- #
def needs_backward(module: torch.nn.Module, *inputs: torch.Tensor) -> bool:
    """Does this forward have to SAVE what a backward will need?

    Two module families ask this to pick between a training kernel that saves activations and an
    inference one that does not, and they each had their own answer. Both were wrong in a different
    direction, so the condition lives here now and neither restates it.

    Gradients must be RECORDABLE. Under ``torch.no_grad()`` / ``inference_mode`` nothing can learn
    through this call whatever the tensors say, so saving activations is pure waste -- and both
    families were paying it. A module-level bench asking for `mode=inference` measured the TRAINING
    forward, because parameters always require grad and a module is in train mode until someone
    calls ``.eval()``: neither is a statement about the call in front of it, and neither had a
    ``torch.is_grad_enabled()`` in front of it either.

    And then EVERY tensor that can carry one counts, not just the first argument. An ``@opaque``
    custom op is opaque to autograd as well as to Dynamo (kernels/_compile.py): its output has no
    ``grad_fn``, so an inference kernel reached with a live gradient gives right numbers and learns
    nothing, with no error. ConditionedTransition tested ``x.requires_grad`` alone, which sent a
    detached x with a live conditioning tensor -- or a live WEIGHT, and the projections are
    parameters, so that is the common case -- down the inference path.
    """
    if not torch.is_grad_enabled():
        return False
    return (module.training
            or any(t is not None and t.requires_grad for t in inputs)
            or any(p.requires_grad for p in module.parameters()))


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
    return capability(device)[0] >= 10


def is_sm90plus(device: torch.device | None = None) -> bool:
    """True on Hopper (sm_90) and newer — i.e. archs with a supported cute GEMM."""
    return capability(device)[0] >= 9


def is_sm90(device: torch.device | None = None) -> bool:
    """True on Hopper *exactly* (sm_9x, major == 9) — NOT Blackwell.

    Distinct from :func:`is_sm90plus` (>= 9): the hand-CUDA b2b and the quack cute
    ``transition_fused`` kernels are Hopper WGMMA/TMA (sm_90a) code that neither
    builds/launches on pre-Hopper (sm_80 / A100) nor on Blackwell (sm_100). Guards
    that route to those kernels must use this, not ``not is_sm100`` — otherwise
    pre-Hopper GPUs (which are also "not sm100") get routed into a Hopper-only
    kernel and crash. A100 falls through to the portable Triton path instead.
    """
    return capability(device)[0] == 9


def is_sm86(device: torch.device | None = None) -> bool:
    """True on the GA10x Ampere *workstation* cards (RTX A5000 / A6000 / A40), sm_86.

    Distinct from A100 (sm_80): although both are pre-Hopper (``not is_sm90plus``), the
    workstation parts have far less shared memory (~100 KB/SM vs A100's 164 KB) and lower
    bf16 tensor-core / bandwidth. Some transition dispatch that favours the fused b2b at
    small ``d`` on A100 loses to the shape-general split on sm_86, so a few routes gate on
    this specifically to keep A100's tuned choice untouched while fixing sm_86.
    """
    return capability(device) == (8, 6)


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
# MINIWORLD (auto) backend policy — SINGLE source of truth.
#
# One declarative table maps each op to the module-layer backend the repo knows is best,
# per arch. A value is either a fixed ``KernelBackend`` or a ``callable(device)->KernelBackend``
# for arch-dependent choices. An op that is absent — or a GPU the callable does not specially
# recognize — falls back to ``_DEFAULT_BACKEND`` (TRITON), the portable path: brand-new GPUs
# get Triton by default, while op/arch pairs the repo has developed + measured a faster
# backend for (e.g. trimul cute on Hopper+) follow that. This encodes ONLY the module-layer
# family choice; kernel-internal shape/arch sub-dispatch still lives next to the kernels.
#
# Concrete (non-MINIWORLD) requests pass straight through ``to_kernel_backend``.
# --------------------------------------------------------------------------- #
_DEFAULT_BACKEND = KernelBackend.TRITON  # unknown op / unknown GPU -> portable Triton

# Ops that actually have a cuequivariance kernel. cuequivariance only ships a triangle
# multiplication kernel; for every OTHER op an explicit CUEQUIVARIANCE request has no cueq
# kernel to run, so it must fall back to the PYTORCH reference (the honest baseline) — NOT
# the miniworld Triton fused path. Routing a no-cueq op's CUEQUIVARIANCE request into the
# Triton family was a bug (e.g. the bf16 transition fused kernel OOMs shared memory at
# d>=256 on H100). PYTORCH is the default fallback for cueq on these ops.
_CUEQ_OPS = frozenset({"triangle_multiplication"})


def _trimul_known_best(device: torch.device | None) -> KernelBackend:
    """trimul: cute on Hopper+ (the measured winner; out_layout via ``trimul_out_layout``),
    Triton pre-Hopper. ``settings.trimul_impl`` pins a backend for debug/A-B."""
    from miniworld_engine import settings

    override = settings.current().trimul_impl
    if override:
        return to_kernel_backend(ImplementationType(override.strip().lower()))
    return KernelBackend.CUTE if is_sm90plus(device) else KernelBackend.TRITON


#: A fixed backend, or a device-dependent chooser. Spelled out rather than `object` so that
#: `resolve`'s `best(device) if callable(best) else best` narrows to `KernelBackend` on both
#: arms instead of returning `object`.
_KnownBest = KernelBackend | Callable[[torch.device | None], KernelBackend]

_MINIWORLD_KNOWN_BEST: dict[str, _KnownBest] = {
    "triangle_multiplication": _trimul_known_best,
    # layernorm's MINIWORLD is the auto-routing layernorm_kernel (fused triton fwd +
    # per-shape auto-dispatched backward), grouped under the CUDA-family entry.
    "layernorm": KernelBackend.CUDA,
    # These have no faster module-layer backend than the TRITON family (whose kernels do
    # their own shape/arch sub-dispatch, incl. cute on Hopper+ internally). Listed
    # explicitly so the policy is auditable rather than implicit-by-omission.
    "transition": KernelBackend.TRITON,
    "triangle_attention": KernelBackend.TRITON,
    "adaptive_layernorm": KernelBackend.TRITON,
    "conditioned_transition": KernelBackend.TRITON,
    "augmented_attention": KernelBackend.TRITON,
}


def resolve(
    op: str, impl: ImplementationType | str, device: torch.device | None = None
) -> KernelBackend:
    """Resolve the PUBLIC option to a concrete ``KernelBackend`` for ``op`` on ``device``.

    ``MINIWORLD`` -> the known-best table (unknown op / unknown GPU -> TRITON default);
    any concrete backend passes through unchanged. Single entry point behind the per-op
    ``resolve_*`` wrappers kept below for the modules / benchmark harness."""
    impl = _coerce(impl)
    # cuequivariance only has a trimul kernel: for any other op, an explicit CUEQUIVARIANCE
    # request falls back to the PYTORCH reference (default), never the Triton fused path.
    if impl == ImplementationType.CUEQUIVARIANCE and op not in _CUEQ_OPS:
        return KernelBackend.PYTORCH
    if impl != ImplementationType.MINIWORLD:
        return to_kernel_backend(impl)
    best = _MINIWORLD_KNOWN_BEST.get(op, _DEFAULT_BACKEND)
    return best(device) if callable(best) else best


# Thin per-op wrappers (stable API for the modules + benchmark harness + tests).
def resolve_transition(impl, device=None):
    return resolve("transition", impl, device)


def resolve_triangle_attention(impl, device=None):
    return resolve("triangle_attention", impl, device)


def resolve_adaptive_layernorm(impl, device=None):
    return resolve("adaptive_layernorm", impl, device)


def resolve_conditioned_transition(impl, device=None):
    return resolve("conditioned_transition", impl, device)


def resolve_augmented_attention(impl, device=None):
    return resolve("augmented_attention", impl, device)


def resolve_layernorm(impl, device=None):
    return resolve("layernorm", impl, device)


def resolve_triangle_multiplication(impl, device=None):
    return resolve("triangle_multiplication", impl, device)


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
    """cute tm1 ``out_layout`` for the running GPU: ``bdll_sm100`` on sm_100 (the hand-rolled
    tcgen05 dual-B gated collective — one A load, dual-TMEM proj+gate accumulators, fused GLU
    epilogue, M-major TMA store straight into d-major [B,D,L,L]); ``bdll_direct_wide`` elsewhere
    (one wide quack ``gemm_act`` + triton GLU fold).

    The bdll_sm100 collective's launch was migrated to cutlass-dsl 4.5.2's TVM-FFI convention
    (``make_fake_stream(use_tvm_ffi_env_stream=True)`` at compile, ``options='--enable-tvm-ffi'``,
    ``from_dlpack(enable_tvm_ffi=True)`` tensors, and a runtime call of the data tensors only —
    stream via env, max_active_clusters baked as Constexpr). The DEVICE kernel/algorithm is
    unchanged; only the launch ABI moved. Result on quack 0.5.0 / cutlass 4.5.2, L=384 inference:
    ``bdll_sm100`` 0.115 ms (fastest — beats even the old 4.4.2 bdll_sm100 at 0.160 ms),
    ``bdll_direct_wide`` 0.184 ms, ``bdll_direct`` 0.194 ms, triton 0.284 ms, pytorch 1.14 ms.
    d-major [B,D,L,L] feeds the efficient d-major einsum (the d-last ``blld`` einsum is ~9x
    slower). Overridable via ``settings.trimul_out_layout`` for debug/A-B."""
    from miniworld_engine import settings

    override = settings.current().trimul_out_layout
    if override:
        return override.strip()
    return "bdll_sm100" if is_sm100(device) else "bdll_direct_wide"
