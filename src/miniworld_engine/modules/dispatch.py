"""Central backend dispatch for the model-level modules.

This is the single home for two things the modules used to each re-implement:

1. **Public vs internal backend split.**
   :class:`~miniworld_engine.modules.exceptions.ImplementationType` is the
   *public* module option. The user-facing values are:

     * ``PYTORCH``  — the readable reference (also the autograd oracle).
     * ``MINIWORLD`` — "ours (auto)": pick the fastest correct kernel for the
       running GPU / shape / mode. This is what production should select.
     * ``CUEQUIVARIANCE`` — the vendor baseline (comparison only).
     * ``ANTHROPIC`` — the Anthropic release's kernels (comparison / opt-in).

   The concrete technology backends (:class:`KernelBackend`)
   are internal: the module never branches on ``ImplementationType`` directly in ``forward`` — it
   resolves to a :class:`KernelBackend` here and dispatches on that. What ``MINIWORLD`` resolves to:

     * H100 (sm_90) / B200 (sm_100): the native CUDA path; a call it does not serve runs the PyTorch
       reference. Never portable (:func:`restricted_card`).
     * every other card: a native CUDA path that serves the call, else the portable family IF the engine
       holds a record for that exact card (:mod:`miniworld_engine.autotune.records`), else a
       :class:`UnrecordedCardWarning` (once per op and card) and the PyTorch reference.

   ``settings.force_portable_kernels`` is the internal A/B switch that forces the portable family on every card.

2. **GPU-architecture policy.**
   The "B200 (sm_100) → X, H100 (sm_90) → Y" knowledge used to live inline in
   four different files (trimul dispatch, transition fused, layernorm_linear, the
   calibration gates). The capability helpers below are now the single source of
   truth for the *module layer*. (Kernel-internal shape/arch sub-dispatch — e.g.
   transition's CUDA-vs-portable-by-d, trimul's hand-CUDA contract —
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
    PORTABLE = "portable"
    CUDA = "cuda"
    CUEQUIVARIANCE = "cuequivariance"
    ANTHROPIC = "anthropic"


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

    ``module.training`` is kept although it is strictly redundant: with grad enabled and nothing
    requiring one, no graph is built and no backward can follow, so the saved activations are
    waste. It costs a module whose parameters are ALL frozen and whose inputs are detached -- a
    frozen submodule during fine-tuning -- and it buys the safe direction of a mistake. The
    expensive direction is the other one: an inference kernel reached with a live gradient returns
    right numbers and silently learns nothing, which is the bug this function was written for.
    Dropping it is a performance change and wants a measurement, not an argument.
    """
    if not torch.is_grad_enabled():
        return False
    return (module.training
            or any(isinstance(t, torch.Tensor) and t.requires_grad for t in inputs)
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
    """True on Hopper (sm_90) and newer."""
    return capability(device)[0] >= 9


def is_sm90(device: torch.device | None = None) -> bool:
    """True on Hopper *exactly* (sm_9x, major == 9) — NOT Blackwell.

    Distinct from :func:`is_sm90plus` (>= 9): the hand-CUDA kernels are Hopper WGMMA/TMA
    (sm_90a) code that neither
    builds/launches on pre-Hopper (sm_80 / A100) nor on Blackwell (sm_100). Guards
    that route to those kernels must use this, not ``not is_sm100`` — otherwise
    pre-Hopper GPUs (which are also "not sm100") get routed into a Hopper-only
    kernel and crash. A100 falls through to the portable path instead.
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
    ImplementationType.CUEQUIVARIANCE: KernelBackend.CUEQUIVARIANCE,
    ImplementationType.ANTHROPIC: KernelBackend.ANTHROPIC,
}


def to_kernel_backend(impl: ImplementationType | KernelBackend) -> KernelBackend:
    """Map an already-concrete ``ImplementationType`` to its ``KernelBackend`` (a ``KernelBackend`` passes through).

    ``MINIWORLD`` is *not* concrete — call the op's ``resolve_*`` instead. Passing
    it here is a programming error (raises ``InvalidImplementationError``).
    """
    if isinstance(impl, KernelBackend):
        return impl
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
# recognize — falls back to ``_DEFAULT_BACKEND`` (PORTABLE), the portable path: brand-new GPUs
# get portable by default, while op/arch pairs the repo has developed + measured a faster
# backend for follow that. This encodes ONLY the module-layer
# family choice; kernel-internal shape/arch sub-dispatch still lives next to the kernels.
#
# Concrete (non-MINIWORLD) requests pass straight through ``to_kernel_backend``.
# --------------------------------------------------------------------------- #
_DEFAULT_BACKEND = KernelBackend.PORTABLE  # unknown op / unknown GPU -> portable

# Ops that actually have a cuequivariance kernel to run. cuequivariance_torch (0.9.1 here) ships
# fused kernels for triangle_multiplication AND triangle self-attention (``cet.triangle_attention``,
# wired in triangle_attention/module.py); an explicit CUEQUIVARIANCE request on those runs the
# real vendor kernel. For every OTHER op cueq has no matching kernel, so the request falls back to
# the PYTORCH reference (the honest baseline) -- NOT the miniworld portable fused path (routing a
# no-cueq op's request into the portable family was a bug: the bf16 transition fused kernel OOMs
# shared memory at d>=256 on H100).
#
# CAUTION -- ``triangle_attention`` is the op name for THREE module shapes, and only one of them
# has a cueq kernel: `TriangleAttention(use_self_attention=True)` (the real branch),
# `TriangleAttention(use_self_attention=False)` (bias-only: `_kernel_bias_only_attention` takes no
# backend and always runs its einsum, so a CUEQUIVARIANCE request there is a LABEL, not a kernel),
# and `BidirectionalTriangleAttention` (no cueq kernel at all -- its forward routes CUEQUIVARIANCE
# to the pytorch reference explicitly, because adding this name here would otherwise make it raise).
_CUEQ_OPS = frozenset({"triangle_multiplication", "triangle_attention"})

# The ops whose module has an ``anthropic`` branch (``KernelBackend.ANTHROPIC``): the Anthropic TriMul payload and its TriangleAttention
# adapter (``integrations.anthropic``). Other ops reject an explicit ANTHROPIC request in ``resolve``.
_ANTHROPIC_OPS = frozenset({"triangle_multiplication", "triangle_attention"})


#: ``settings.trimul_impl`` values -> the backend they pin.
_TRIMUL_ROUTE_BACKENDS = {
    "portable": KernelBackend.PORTABLE,
    "native": KernelBackend.CUDA,
    "pytorch": KernelBackend.PYTORCH,
    "cuequivariance": KernelBackend.CUEQUIVARIANCE,
    "anthropic": KernelBackend.ANTHROPIC,
}


def _trimul_known_best(device: torch.device | None) -> KernelBackend:
    """trimul: PORTABLE family on every arch. The hand-CUDA H100 kernels are not a module-layer
    backend: the modules try ``integrations.trimul_h100`` first and fall through to this
    family for every shape it does not serve. ``settings.trimul_impl`` pins a backend for
    debug/A-B."""
    from miniworld_engine import settings

    override = settings.current().trimul_impl
    if override:
        return _TRIMUL_ROUTE_BACKENDS[override.strip().lower()]
    return KernelBackend.PORTABLE


def restricted_card(device: torch.device | None = None) -> bool:
    """True on H100 (sm_90) and B200 (sm_100): the cards with **no portable fallback**, on purpose. They run hand-written CUDA and
    cuBLAS; a call no CUDA path serves runs the PyTorch reference (compiled by ``torch.compile`` where the caller compiles), never an
    untuned portable kernel. ``settings.portable_on_hopper_blackwell`` (or ``force_portable_kernels=True``) lifts it for A/B work."""
    if not torch.cuda.is_available():
        return False
    from miniworld_engine import settings

    current = settings.current()
    if current.portable_on_hopper_blackwell or current.force_portable_kernels:
        return False
    index = device.index if device is not None and device.type == "cuda" else None
    return torch.cuda.get_device_capability(index)[0] in (9, 10)


def fallback_autocast(x: torch.Tensor, module: torch.nn.Module, active: bool = True):
    """Context for a PyTorch fallback leg. A bf16 activation next to fp32 master parameters (the engine's mixed-precision layout) runs the
    reference under bf16 autocast, as the caller's own autocast would; anything else runs as is."""
    import contextlib

    if (
        active
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and not torch.is_autocast_enabled("cuda")
        and any(p.dtype == torch.float32 for p in module.parameters() if p.ndim >= 2)
    ):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def portable_forced(impl: ImplementationType | str = ImplementationType.MINIWORLD) -> bool:
    """Is the portable family forced for ``impl``'s modules (``settings.force_portable_kernels``, ``MINIWORLD`` only)?"""
    from miniworld_engine import settings

    return settings.current().force_portable_kernels and _coerce(impl) == ImplementationType.MINIWORLD


def portable_leg_allowed(device: torch.device | None = None, op: str | None = None, *, training: bool = False) -> bool:
    """May a module run its portable leg on ``device``? False on H100 / B200 (see :func:`restricted_card`): the module takes its
    PyTorch leg. With ``op``, also False on a card without a record for what the call launches (:func:`route_for_card`, forward
    kernels only unless ``training``: one warning per op, card and mode). ``force_portable_kernels`` allows it everywhere."""
    if restricted_card(device):
        return False
    if op is None:
        return True
    from miniworld_engine import settings

    if settings.current().force_portable_kernels:
        return True
    return _card_gate(op, device, False, "training" if training else "inference") is not KernelBackend.PYTORCH


def _torch_on_hopper_blackwell(default: KernelBackend) -> Callable[[torch.device | None], KernelBackend]:
    """H100 and B200 run this op as PyTorch ops (``torch.compile`` fuses them); every other card keeps ``default``. The settings
    ``portable_on_hopper_blackwell`` / ``b200_portable_kernels`` (B200 only) bring the engine's portable kernels back for A/B work. Resolved once,
    when the module is built."""

    def pick(device: torch.device | None) -> KernelBackend:
        from miniworld_engine import settings

        if restricted_card(device):
            index = device.index if device is not None and device.type == "cuda" else None
            if torch.cuda.get_device_capability(index) == (10, 0) and settings.current().b200_portable_kernels:
                return default
            return KernelBackend.PYTORCH
        return default

    return pick


def _cuda_on_hopper_blackwell(default: KernelBackend) -> Callable[[torch.device | None], KernelBackend]:
    """Like :func:`_torch_on_hopper_blackwell`, but H100 and B200 run the op on its general CUDA kernel rather than as PyTorch ops."""
    torch_pick = _torch_on_hopper_blackwell(default)

    def pick(device: torch.device | None) -> KernelBackend:
        backend = torch_pick(device)
        return KernelBackend.CUDA if backend is KernelBackend.PYTORCH else backend

    return pick


#: A fixed backend, or a device-dependent chooser. Spelled out rather than `object` so that
#: `resolve`'s `best(device) if callable(best) else best` narrows to `KernelBackend` on both
#: arms instead of returning `object`.
_KnownBest = KernelBackend | Callable[[torch.device | None], KernelBackend]

_MINIWORLD_KNOWN_BEST: dict[str, _KnownBest] = {
    "triangle_multiplication": _trimul_known_best,
    # layernorm's MINIWORLD is the CUDA family on every card: the auto-routing layernorm_kernel
    # (fused portable fwd + per-shape auto-dispatched backward), and on H100 / B200 (no portable
    # there, see restricted_card) the general CUDA LayerNorm (kernels/layernorm/cuda/general).
    "layernorm": KernelBackend.CUDA,
    # rmsnorm: portable on other cards; H100 / B200 the general CUDA RMSNorm (kernels/rmsnorm/cuda).
    "rmsnorm": _cuda_on_hopper_blackwell(KernelBackend.PORTABLE),
    # These have no faster module-layer backend than the PORTABLE family (whose kernels do
    # their own shape/arch sub-dispatch, incl. hand-CUDA on H100 internally). Listed
    # explicitly so the policy is auditable rather than implicit-by-omission.
    "transition": KernelBackend.PORTABLE,
    "triangle_attention": KernelBackend.PORTABLE,
    # No CUDA path yet: portable was their implementation; on H100 / B200 they run the PyTorch reference (no portable fallback there).
    "adaptive_layernorm": _torch_on_hopper_blackwell(KernelBackend.PORTABLE),
    "conditioned_transition": _torch_on_hopper_blackwell(KernelBackend.PORTABLE),
    "augmented_attention": KernelBackend.PORTABLE,
    # The MSA pair. Their native H100 paths (integrations.opm_train / pwa_train) and the Anthropic payload are chosen inside
    # the modules before this is asked; what is resolved here is the portable path every other GPU gets.
    "outer_product_mean": KernelBackend.PORTABLE,
    "msa_pair_weighted_averaging": KernelBackend.PORTABLE,
    # The SWA atom attention's pointwise ops and the engine SWADiTBlock: no module-level resolve, but the same per-card rule.
    "swa_atom_attention": KernelBackend.PORTABLE,
    "swa_dit": KernelBackend.PORTABLE,
    # The attention cores of the portable DiT paths: the portable window / token kernels where the card is recorded (same per-card rule).
    "local_dit": KernelBackend.PORTABLE,
    "dit_variant": KernelBackend.PORTABLE,
    "dit_variant_atom": KernelBackend.PORTABLE,
}


class UnrecordedCardWarning(UserWarning):
    """``miniworld`` has no CUDA path for this op on this card and the engine holds no record that this card was run
    and checked here, so the call runs the PyTorch reference. Verify output and speed yourself, or record the card."""


class KernelRefusedWarning(UserWarning):
    """A portable kernel refused this call (a shape or dtype guard): ``miniworld`` runs the PyTorch reference for it instead."""


_refused_warned: set[tuple[str, str, str]] = set()
_unrecorded_warned: set[tuple[str, str, str]] = set()
_assume_recorded = False


def assume_every_card_recorded(value: bool = True) -> None:
    """Route ``miniworld`` as if the card held a record for every op. For ``autotune.derive``, which asks what the modules launch."""
    global _assume_recorded
    _assume_recorded = value
_card_cache: dict[object, tuple[str, tuple[int, int]]] = {}


def _card_of(device: torch.device | None) -> tuple[str, tuple[int, int]] | None:
    """``(name, capability)`` of ``device`` (the current one if None), or None when there is no card to ask: no CUDA, or a
    non-CUDA device."""
    if not torch.cuda.is_available():
        return None
    if device is not None and device.type != "cuda":
        return None
    if torch.compiler.is_compiling():
        # Dynamo cannot trace the driver query: use what an eager call (a module build) cached; a cold cache reads as "no card", i.e. the
        # kernel family stays, as it did before the per-card rule.
        index = device.index if device is not None and device.index is not None else 0
        return _card_cache.get(index)
    index = device.index if device is not None and device.type == "cuda" and device.index is not None else torch.cuda.current_device()
    card = _card_cache.get(index)
    if card is None:
        card = _card_cache[index] = (torch.cuda.get_device_name(index), tuple(torch.cuda.get_device_capability(index)))
    return card


def route_for_card(
    op: str,
    name: str,
    capability: tuple[int, int],
    *,
    native_cuda: bool = False,
    dtype: str = "bf16",
    mode: str = "inference",
    warn: bool = True,
) -> KernelBackend:
    """Where ``miniworld`` runs ``op`` on the card ``name`` / ``capability`` for a call in ``mode`` (``"inference"``: no backward is
    recorded, only the forward kernels run; ``"training"``: all of them), given the forced-portable switch is off.

    The pure rule (no GPU needed; the manifests are the only input, see :mod:`miniworld_engine.autotune.records`):

      * H100 / B200 (capability major 9 or 10): ``PORTABLE`` is returned unchanged -- these cards never reach a portable kernel
        through ``miniworld`` (:func:`restricted_card`), the modules take their CUDA or PyTorch leg, and no record is consulted.
      * ``native_cuda`` (a hand-written CUDA path of the engine serves this call on the card): ``PORTABLE``, the family whose
        module-level dispatch tries the CUDA path first; a record is not needed for what does not launch.
      * an autotune BUILD (``settings.capture``): ``PORTABLE``. The build is how a card's portable configs are captured and its
        record is made; refusing it for lack of a record would make a new card unbuildable.
      * every portable kernel the op launches in ``mode`` has an ``ok`` manifest row for this exact card and ``dtype``: ``PORTABLE``.
        A missing backward kernel therefore blocks training calls only.
      * otherwise: :class:`UnrecordedCardWarning` once per (op, card, mode) when ``warn``, and ``PYTORCH``.
    """
    from miniworld_engine import settings

    if capability[0] in (9, 10) or native_cuda or settings.current().capture or _assume_recorded:
        return KernelBackend.PORTABLE
    from miniworld_engine.autotune import records

    holes = records.missing(op, name, capability, dtype, mode)
    if not holes:
        return KernelBackend.PORTABLE
    if warn:
        _warn_unrecorded(op, name, capability, mode, holes)
    return KernelBackend.PYTORCH


def _warn_unrecorded(op: str, name: str, capability: tuple[int, int], mode: str, holes: tuple[str, ...]) -> None:
    from miniworld_engine.autotune import records

    key = (op, records.card_key(name, capability), mode)
    if key in _unrecorded_warned:
        return
    _unrecorded_warned.add(key)
    shown = ", ".join(holes[:3]) + (f", ... ({len(holes)} in all)" if len(holes) > 3 else "")
    calls = "calls that need a backward (training)" if mode == "training" else "this call"
    warnings.warn(
        f"{op}: this GPU ({name}, sm_{capability[0]}{capability[1]}) is not recorded in the engine for {calls} (no 'ok' row in "
        f"autotune/manifests for: {shown}) and miniworld has no native CUDA path for it here, so it runs the PyTorch reference: "
        f"correct, but not the engine's speed, and nothing was validated on this card. Verify the output and speed yourself, or "
        f"record the card with `python -m miniworld_engine.autotune.run_all`. Only to force the engine's portable kernels anyway "
        f"(A/B work), pass settings.force_portable_kernels=True.",
        UnrecordedCardWarning,
        stacklevel=5,
    )


def warn_refused(op: str, device: torch.device | None, reason: str) -> None:
    """``miniworld`` meets a call the op's kernels REFUSE (the guard that refused is ``reason``): it never raises and never
    diverges silently -- one :class:`KernelRefusedWarning` per (op, card, reason) and the caller runs the PyTorch reference. A CPU tensor
    is not a refusal worth a warning (there is no kernel to run): the caller does not call this for it."""
    card = _card_of(device)
    key = (op, card[0] if card else "no GPU", reason)
    if key in _refused_warned:
        return
    _refused_warned.add(key)
    warnings.warn(
        f"{op}: the engine's kernels refuse this call ({reason}); running the PyTorch reference instead (correct, but not the engine's "
        f"speed). Nothing was validated for this shape on this card: verify the output and speed yourself.",
        KernelRefusedWarning,
        stacklevel=3,
    )


def _card_gate(op: str, device: torch.device | None, native_cuda: bool, mode: str = "inference") -> KernelBackend:
    """:func:`route_for_card` for the card ``device`` names; no card (CPU build, non-CUDA device) keeps ``PORTABLE``, the family
    the module falls to once its tensors are on a card."""
    card = _card_of(device)
    if card is None:
        return KernelBackend.PORTABLE
    return route_for_card(op, card[0], card[1], native_cuda=native_cuda, mode=mode)


class TrainGate:
    """What a module built on a card recorded for INFERENCE but not for TRAINING remembers: its training calls take the PyTorch
    reference (:func:`call_backend`), with one warning. Plain attributes, so a compiled forward reads it as constants."""

    __slots__ = ("capability", "holes", "name", "op")

    def __init__(self, op: str, name: str, capability: tuple[int, int], holes: tuple[str, ...]) -> None:
        self.op, self.name, self.capability, self.holes = op, name, capability, holes

    def note(self) -> None:
        if not torch.compiler.is_compiling():
            _warn_unrecorded(self.op, self.name, self.capability, "training", self.holes)


def train_gate(
    op: str,
    impl: ImplementationType | str,
    backend: KernelBackend,
    device: torch.device | None = None,
    *,
    native_cuda: bool = False,
) -> TrainGate | None:
    """The :class:`TrainGate` of a ``miniworld`` module whose build-time ``backend`` is the portable family but whose card lacks ``ok``
    rows for the kernels a backward launches; None when nothing narrows (store it as ``module._train_gate``)."""
    from miniworld_engine import settings

    if backend is not KernelBackend.PORTABLE or _coerce(impl) != ImplementationType.MINIWORLD:
        return None
    if settings.current().force_portable_kernels or settings.current().capture or _assume_recorded:
        return None
    card = _card_of(device)
    if card is None or card[1][0] in (9, 10) or native_cuda:
        return None
    from miniworld_engine.autotune import records

    holes = records.missing(op, card[0], card[1], "bf16", "training")
    return TrainGate(op, card[0], card[1], holes) if holes else None


def call_backend(module: torch.nn.Module, backend: KernelBackend, *tensors: torch.Tensor) -> KernelBackend:
    """``backend`` for THIS call: a module whose card is recorded for inference only (``module._train_gate``) runs the PyTorch
    reference when the call needs a backward (:func:`needs_backward`)."""
    gate = getattr(module, "_train_gate", None)
    if gate is None or backend is not KernelBackend.PORTABLE:
        return backend
    if not needs_backward(module, *tensors):
        return backend
    gate.note()
    return KernelBackend.PYTORCH


def portable_blocked(op: str, device: torch.device | None, *tensors: torch.Tensor) -> bool:
    """For the weights-as-args ``ops.*`` entry points: must this call NOT run ``op``'s portable kernels? True on a CUDA card other than
    H100 / B200 that has no ``ok`` rows for what the call launches (inference; training when a tensor needs a gradient), unless the
    A/B switch is on: the caller then runs the PyTorch statements, with the same warning as the modules. False everywhere else
    (H100 / B200 keep their own rule, no card keeps the kernel)."""
    from miniworld_engine import settings

    if settings.current().force_portable_kernels:
        return False
    card = _card_of(device)
    if card is None or card[1][0] in (9, 10):
        return False
    training = torch.is_grad_enabled() and any(isinstance(t, torch.Tensor) and t.requires_grad for t in tensors)
    mode = "training" if training else "inference"
    return route_for_card(op, card[0], card[1], mode=mode) is KernelBackend.PYTORCH


def resolve(
    op: str,
    impl: ImplementationType | str,
    device: torch.device | None = None,
    *,
    native_cuda: bool = False,
    training: bool = False,
) -> KernelBackend:
    """Resolve the PUBLIC option to a concrete ``KernelBackend`` for ``op`` on ``device``.

    ``MINIWORLD`` -> the known-best table (unknown op / unknown GPU -> PORTABLE default), and a PORTABLE result is then
    gated per card (:func:`route_for_card`: recorded card -> PORTABLE, else a warning and PYTORCH; H100 / B200 untouched);
    any concrete backend passes through unchanged. ``native_cuda`` says the caller knows a hand-written CUDA path of the
    engine serves this very call on this card. Single entry point behind the per-op ``resolve_*`` wrappers kept below for
    the modules / benchmark harness."""
    impl = _coerce(impl)
    from miniworld_engine import settings
    if settings.current().force_portable_kernels:
        if impl == ImplementationType.ANTHROPIC:
            raise ValueError(f"{op}: {impl.value} conflicts with force_portable_kernels=True")
        if impl == ImplementationType.MINIWORLD:
            return KernelBackend.PORTABLE
    # An explicit CUEQUIVARIANCE request for an op with no cueq kernel (anything outside
    # _CUEQ_OPS) falls back to the PYTORCH reference, never the portable fused path.
    if impl == ImplementationType.CUEQUIVARIANCE and op not in _CUEQ_OPS:
        return KernelBackend.PYTORCH
    # The Anthropic payload is wired into the modules of _ANTHROPIC_OPS (TriMul, TriangleAttention); naming it for another op
    # here is a mistake, not a fallback (integrations.anthropic_trimul states what it serves and refuses the rest at forward
    # time). The MSA paths (integrations.anthropic_msa) route inside their own modules and never reach this resolver.
    if impl == ImplementationType.ANTHROPIC and op not in _ANTHROPIC_OPS:
        raise ValueError(f"{op}: the anthropic payload is wired for {', '.join(sorted(_ANTHROPIC_OPS))} only")
    if impl != ImplementationType.MINIWORLD:
        return to_kernel_backend(impl)
    best = _MINIWORLD_KNOWN_BEST.get(op, _DEFAULT_BACKEND)
    backend = best(device) if callable(best) else best
    mode = "training" if training else "inference"
    if backend is KernelBackend.PORTABLE and not (op == "triangle_multiplication" and settings.current().trimul_impl):
        backend = _card_gate(op, device, native_cuda, mode)
    elif backend is KernelBackend.CUDA and op == "layernorm" and _card_gate(op, device, native_cuda, mode) is KernelBackend.PYTORCH:
        # LayerNorm is the CUDA family on every card but its forward is portable off H100 / B200: the same record rule.
        backend = KernelBackend.PYTORCH
    return backend


# Thin per-op wrappers (stable API for the modules + benchmark harness + tests).
def resolve_transition(impl, device=None, **kwargs):
    return resolve("transition", impl, device, **kwargs)


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


def resolve_rmsnorm(impl, device=None):
    return resolve("rmsnorm", impl, device)


def resolve_triangle_multiplication(impl, device=None):
    return resolve("triangle_multiplication", impl, device)


# --------------------------------------------------------------------------- #
# dtype correctness-guard
#
# The fused kernels (portable / hand-CUDA) are bf16-only. If a module is
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
