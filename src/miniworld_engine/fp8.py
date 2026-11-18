"""The engine's FP8 (e4m3) inference routes on H100 (engine 3.0.0; opt-in, off by default: the default stays the bf16 / fp16 engine).

The user-facing name is the backend **``engine-fp8``**: the engine ``cuda`` backend with the ``safe`` preset's FP8 routes on for the calls of
the modules that select it (every other module, and every module by default, runs the bf16 / fp16 engine, bit-identical to 2.5.14)::

    TriangleMultiplication(128, backend="engine-fp8")         # or module.set_backend("engine-fp8"); a composite passes it to its parts
    miniworld_engine.fp8.set_model("af3")                     # + the TriAttn FP8 routes validated for that model (or MINIWORLD_FP8_MODEL=af3)

``engine-fp8`` is not a class's ``backends`` entry (the judge, the registry and the measurers never see it): ``set_backend`` resolves it to
``cuda`` (``module.backend == "cuda"``, ``module.engine_fp8 is True``, ``module.backend_name == "engine-fp8"``) on every class that has a
``cuda`` backend; ``engine_fp8`` (:data:`BACKEND_ID`) is accepted too, for places where ``-`` cannot stand. The module's ``_forward_cuda``
(an instance attribute ``set_backend`` installs, so ``forward`` and a runtime that calls ``_forward_<backend>`` directly both reach it) runs
inside :class:`BackendScope`, which turns the routes on for the call and restores the switches after it (a module on another backend called
through its ``forward`` inside it runs with the switches as they were before). Its range is ``cuda``'s; calls the FP8 routes do not take (fp32, cards other than
H100) run the default kernels.

``engine-fp8`` accelerates inference; training under ``engine-fp8`` runs the default bf16 path: every FP8 route is a forward (inference)
path with no FP8 backward, so a call in train mode (``module.training``, dropout live) or with autograd on (``torch.is_grad_enabled()``)
does not open the FP8 scope and runs exactly the ``cuda`` backend (same bytes, outputs and gradients). Run inference under
``torch.no_grad()`` / ``torch.inference_mode()`` with the model in ``eval()`` to get the FP8 routes.

The lower-level switch (process wide, every ``cuda`` module) stays::

    MINIWORLD_FP8=safe python run.py                          # environment: before the first import of the engine's kernels
    MINIWORLD_FP8=safe MINIWORLD_FP8_MODEL=af3 python run.py   # + the TriAttn FP8 routes validated for that model
    miniworld_engine.fp8.enable("safe")                       # API: at run time (after import), before the first forward
    miniworld_engine.fp8.enable("safe", model="af3")          # + the TriAttn FP8 routes validated for that model

``"safe"`` is the set validated end to end on AF3 (FoldBench, 27 targets, 0 flagged, lDDT +0.0000 against bf16, whole forward 0.97-0.80x;
docs/fp8.md "fp8safe2"): TriMul e4m3 planes (static scales, calibrated), OuterProductMean e4m3 product (weight-bound scales), PWA e4m3 key
contraction, Transition e4m3 GEMMs (calibrated), token DiT e4m3 expand + SwiGLU (not ``qkvg``), atom DiT e4m3 row kernels (``pre`` + ``post``):
these routes do not depend on the model.

TriAttn FP8 runs **only for a validated model** (:data:`TRIATTN_MODELS`; decision 2026-11-17): the FP8 PV route on the call-site kinds checked
end to end for that model, and e4m3 q / k on the fused d_pair-128 route only where that was checked too. The model is named once, either
``MINIWORLD_FP8_MODEL=<model>`` next to ``MINIWORLD_FP8`` (read at import of the TriAttn kernels) or ``enable(preset, model=<model>)`` (run
time; :func:`enable` does not read ``MINIWORLD_FP8_MODEL``). No model, or a model not in :data:`TRIATTN_MODELS` (e.g. Boltz-2, OpenFold3):
TriAttn keeps the default fp16 core, bit-identical to FP8 off. q / k e4m3 is in ``safe`` for af3 although its module error (judge form) is
above the module tolerance: the end-level AF3 FoldBench run with it on flagged 0 / 27 targets.

Each route keeps its own switch (``MINIWORLD_TRIMUL_FP8``, ``MINIWORLD_OPM_FP8``, ``MINIWORLD_PWA_FP8``, ``MINIWORLD_TRANSITION_FP8``,
``MINIWORLD_TOKEN_DIT_FP8``, ``MINIWORLD_ATOMDIT_FP8`` and their ``_GUARD`` / gate variables; TriAttn: ``h100_sm90.enable_fp8_pv_for``):
with ``MINIWORLD_FP8`` set, a per-route variable that is set explicitly (also to the empty string = off) wins over the preset.

FP8 is outside the judge: no registry row, tolerance or ``backend_versions`` entry exists for it (registry dtypes ``float32|master``), no
class lists ``engine-fp8`` in its ``backends``, and the judge tokens cover the default route only. Do not run the judge / measure tools with
any FP8 switch set.

Guard contract (static scales: TriAttn V / q k, TriMul planes, Transition h, PWA v). The scales are taken on a process's first eager call
of each site (calibration) and kept for the process. A value past the e4m3 range saturates; in eager mode TriMul / PWA / Transition (guard
``sync``, the default) redo such a call themselves, under CUDA-graph capture they record it. After every forward, outside any graph, call
:func:`check` once (one host sync per route): True means some route saturated and was rescaled -- run that forward again (at most 3 times).
OuterProductMean (``bound``), token DiT and atom DiT (exact per-row scales) cannot saturate.

The routes take inference calls in bf16 (``master`` modules) on H100 (sm_90a) only; everything else (training, fp32, other GPUs) runs the
default route, whose outputs are bit-identical with FP8 off.
"""

from __future__ import annotations

import os
from types import MappingProxyType
from typing import Any

#: The FP8 backend's public name (engine 3.0.0): what ``backend=`` / ``set_backend`` and a runtime's backend option take.
BACKEND = "engine-fp8"
#: Its identifier where ``-`` cannot stand (Python names, config keys); ``set_backend`` takes it as well.
BACKEND_ID = "engine_fp8"
#: The engine backend whose forward ``engine-fp8`` runs, and the preset it switches on.
BASE_BACKEND = "cuda"
BACKEND_PRESET = "safe"

#: the per-route environment variables of each preset (what ``MINIWORLD_FP8=<preset>`` makes the default of each route module)
PRESETS = MappingProxyType(
    {
        "safe": MappingProxyType(
            {
                "MINIWORLD_TRIMUL_FP8": "calib",
                "MINIWORLD_OPM_FP8": "bound",
                "MINIWORLD_PWA_FP8": "1",
                "MINIWORLD_TRANSITION_FP8": "calib",
                "MINIWORLD_TOKEN_DIT_FP8": "expand",
                "MINIWORLD_ATOMDIT_FP8": "1",
            }
        ),
    }
)
#: TriAttn FP8 per validated model: model -> (call-site kinds of the FP8 PV route, e4m3 q / k on the fused route). Only end-level evidence for
#: exactly these routes puts a model here (docs/fp8.md): af3 = FoldBench 27 targets with the whole ``safe`` set incl. q / k (0 flagged) and 4
#: targets x 4 trunk seeds with PV; protenix1 / opendde = PV on the listed kinds, 4 trunk seeds per target (no q / k run: PV only). Boltz-2 is out
#: on purpose (9c90 bistable); OpenFold3 is not added. ``h100_sm90.FP8_PV_WHITELIST`` is derived from this table. Extend only after an end-level check.
TRIATTN_MODELS = MappingProxyType(
    {
        "af3": (frozenset({"fused", "unfused"}), True),
        "protenix1": (frozenset({"fused"}), False),
        "opendde": (frozenset({"unfused"}), False),
    }
)
_NO_TRIATTN: tuple[frozenset[str], bool] = (frozenset(), False)


def preset() -> str | None:
    """The preset ``MINIWORLD_FP8`` names (None: unset, ``0``, ``off`` or ``none``)."""
    v = os.environ.get("MINIWORLD_FP8", "").strip().lower()
    if v in ("", "0", "off", "none"):
        return None
    if v not in PRESETS:
        raise ValueError(f"MINIWORLD_FP8={v!r}: unknown preset (use one of {', '.join(PRESETS)}, or off)")
    return v


def model() -> str | None:
    """The model ``MINIWORLD_FP8_MODEL`` names (lower case; None: unset or empty). Only read with ``MINIWORLD_FP8`` set, for the TriAttn part."""
    v = os.environ.get("MINIWORLD_FP8_MODEL", "").strip().lower()
    return v or None


def triattn_for(model_name: str | None) -> tuple[frozenset[str], bool]:
    """The TriAttn FP8 part for ``model_name``: (call-site kinds, e4m3 q / k); nothing for None or a model not in :data:`TRIATTN_MODELS`."""
    if model_name is None:
        return _NO_TRIATTN
    return TRIATTN_MODELS.get(model_name.strip().lower(), _NO_TRIATTN)


def env(name: str, default: str | None = None) -> str | None:
    """A route module's switch: the variable when it is set (also when set to ""), else the ``MINIWORLD_FP8`` preset's value, else ``default``."""
    if name in os.environ:
        return os.environ[name]
    p = preset()
    if p is not None and name in PRESETS[p]:
        return PRESETS[p][name]
    return default


_MODULES: list[tuple[Any, ...]] = []


def _modules() -> tuple[Any, ...]:
    if _MODULES:
        return _MODULES[0]
    from miniworld_engine.kernels.conditioned_transition.cuda import token_dit_fp8 as td
    from miniworld_engine.kernels.integrations import local_dit_h100 as ad
    from miniworld_engine.kernels.integrations import opm_fp8 as opm
    from miniworld_engine.kernels.integrations import pwa_fp8 as pwa
    from miniworld_engine.kernels.transition.cuda import fp8_sm90a as tr
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as ta
    from miniworld_engine.kernels.trimul_inproj.cuda import h100_b200style_fp8 as tm

    _MODULES.append((ta, tm, opm, pwa, tr, td, ad))
    return _MODULES[0]


# ------------------------------------------------------------------------------------------------------------- the engine-fp8 backend
def resolve_backend(name: str) -> tuple[str, bool]:
    """``(the backend whose forward runs, FP8 on)`` of a backend name: ``engine-fp8`` / ``engine_fp8`` -> ``("cuda", True)``; any other name
    -> ``(name, False)`` (checked against the class's ``backends`` by ``set_backend``)."""
    if name in (BACKEND, BACKEND_ID):
        return BASE_BACKEND, True
    return name, False


_MODEL: list[str | None] = [None]


def set_model(name: str | None) -> None:
    """The model ``engine-fp8`` modules run for (its TriAttn FP8 routes, :data:`TRIATTN_MODELS`); None: ``MINIWORLD_FP8_MODEL`` (read at each
    scope entry). A model outside :data:`TRIATTN_MODELS` keeps TriAttn on the default fp16 core. Process wide."""
    _MODEL[0] = name.strip().lower() if name else None


def backend_model() -> str | None:
    """The model of the ``engine-fp8`` backend: :func:`set_model`'s, else ``MINIWORLD_FP8_MODEL``."""
    return _MODEL[0] or model()


_TA_SWITCHES = ("FP8_PV", "FP8_PV_FUSED", "FP8_PV_UNFUSED", "FP8_VSCALE_AUTO", "FP8_PV_HL", "FP8_PV_MIN_S", "FP8_PV_FRONT_MIN_S", "FP8_QK")


def _snapshot() -> tuple:
    """The route switches as they are (what :func:`enable` / :func:`disable` set; calibrated scales and guards are not switches)."""
    ta, tm, opm, pwa, tr, td, ad = _modules()
    return tuple(getattr(ta, a) for a in _TA_SWITCHES), tm.POLICY, opm.POLICY, pwa.ENABLED, tr.POLICY, td.PLACES, ad.FP8_ROWS


def _restore(snap: tuple) -> None:
    ta, tm, opm, pwa, tr, td, ad = _modules()
    for a, v in zip(_TA_SWITCHES, snap[0], strict=True):
        setattr(ta, a, v)
    tm.POLICY, opm.POLICY, pwa.ENABLED, tr.POLICY, td.PLACES, ad.FP8_ROWS = snap[1:]


#: The ``engine-fp8`` scopes open now, innermost last (True: an ``engine-fp8`` module's call; False: a module on another backend called
#: inside one). Empty = no ``engine-fp8`` call is running: the switches are the process's own (default: every route off).
SCOPES: list[bool] = []
_BASE: list[tuple] = []  # the switches before the outermost scope


class BackendScope:
    """``with BackendScope(on):`` the route switches of one module call: ``on`` = the ``engine-fp8`` preset (``safe`` + :func:`backend_model`'s
    TriAttn part); off (inside an ``engine-fp8`` call) = the switches as they were before the outermost scope. The switches are restored on
    exit. An ``engine-fp8`` module's ``_forward_cuda`` opens it (``modules.primitives._EngineFP8Forward``); ``BackendModule.forward`` opens
    it off for a module on another (non-reference) backend only while an ``engine-fp8`` call is running, so the default route never enters it."""

    __slots__ = ("on", "pushed", "snap")

    def __init__(self, on: bool) -> None:
        self.on = bool(on)
        self.pushed = False
        self.snap: tuple | None = None

    def __enter__(self) -> BackendScope:
        self.snap = None
        self.pushed = bool(SCOPES) or self.on
        if not self.pushed:  # off with no engine-fp8 call running: the switches are already the process's own
            return self
        if not SCOPES or SCOPES[-1] != self.on:  # else the enclosing scope already has this state
            self.snap = _snapshot()
            if not SCOPES:
                _BASE.append(self.snap)
            if self.on:
                enable(BACKEND_PRESET, model=backend_model())
            else:
                _restore(_BASE[-1])
        SCOPES.append(self.on)
        return self

    def __exit__(self, *exc) -> None:
        if not self.pushed:
            return
        SCOPES.pop()
        if self.snap is not None:
            _restore(self.snap)
            if not SCOPES:
                _BASE.pop()


def enable_triattn(kinds, qk: bool) -> None:
    """The TriAttn FP8 PV route on the call-site kinds ``kinds`` (``fused``: the d_pair-128 front writes the e4m3 V^T planes; ``unfused``:
    the other sites, front-made planes or the standalone pass, S-gated), hi plane, static per-site scales; ``qk``: e4m3 q / k on the fused route."""
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as ta

    ta.disable_fp8_pv()
    ta.FP8_QK = False
    if not kinds:
        return
    ta.FP8_PV, ta.FP8_VSCALE_AUTO, ta.FP8_PV_HL = True, True, False
    ta.FP8_PV_FUSED = "fused" in kinds
    ta.FP8_PV_UNFUSED = "unfused" in kinds
    ta.FP8_PV_MIN_S = dict(ta.FP8_PV_UNFUSED_MIN_S) if ta.FP8_PV_UNFUSED else {}
    ta.FP8_PV_FRONT_MIN_S = dict(ta.FP8_PV_UNFUSED_FRONT_MIN_S) if ta.FP8_PV_UNFUSED else {}
    ta.FP8_QK = bool(qk) and ta.FP8_PV_FUSED


def enable(name: str = "safe", *, model: str | None = None, guard: str | None = None) -> None:
    """Switch every route of preset ``name`` on (run time; replaces the routes' current switches). ``model``: the TriAttn FP8 routes
    :data:`TRIATTN_MODELS` lists for that model; None (default) or an unlisted model: TriAttn stays on the default fp16 core (the other routes
    are on regardless of ``model``). ``guard``: ``"sync"`` / ``"lazy"`` for TriMul / OPM / PWA / Transition (None: as they are)."""
    if name not in PRESETS:
        raise ValueError(f"unknown FP8 preset {name!r} (use one of {', '.join(PRESETS)})")
    _ta, tm, opm, pwa, tr, td, ad = _modules()
    env_of = PRESETS[name]
    tm.POLICY = env_of.get("MINIWORLD_TRIMUL_FP8") or None
    opm.POLICY = env_of.get("MINIWORLD_OPM_FP8") or None
    pwa.ENABLED = env_of.get("MINIWORLD_PWA_FP8") == "1"
    tr.POLICY = env_of.get("MINIWORLD_TRANSITION_FP8") or None
    td.PLACES = td._parse(env_of.get("MINIWORLD_TOKEN_DIT_FP8", ""))
    ad.enable_fp8(ad._fp8_parts(env_of.get("MINIWORLD_ATOMDIT_FP8", "")))
    enable_triattn(*triattn_for(model))
    if guard is not None:
        if guard not in ("sync", "lazy", "off"):
            raise ValueError(f"guard must be sync, lazy or off, got {guard!r}")
        tm.GUARD = opm.GUARD = pwa.GUARD = tr.GUARD = guard


def disable() -> None:
    """Every FP8 route off (the default route: the bf16 / fp16 kernels, bit-identical to the engine without FP8). Calibrated scales are kept."""
    _ta, tm, opm, pwa, tr, td, ad = _modules()
    tm.POLICY = opm.POLICY = tr.POLICY = None
    pwa.ENABLED = False
    td.PLACES = frozenset()
    ad.enable_fp8(False)
    enable_triattn(frozenset(), False)


def status() -> dict:
    """The switches in force, per route."""
    ta, tm, opm, pwa, tr, td, ad = _modules()
    kinds = sorted(k for k, on in (("fused", ta.FP8_PV_FUSED), ("unfused", ta.FP8_PV_UNFUSED)) if on) if ta.FP8_PV else []
    return {
        "triattn": {"pv": kinds, "qk": bool(ta.FP8_QK and ta.FP8_PV)},
        "trimul": tm.POLICY,
        "opm": opm.POLICY,
        "pwa": pwa.ENABLED,
        "transition": tr.POLICY,
        "token_dit": sorted(td.PLACES),
        "atom_dit": sorted(ad.FP8_ROWS),
        "guard": {"trimul": tm.GUARD, "opm": opm.GUARD, "pwa": pwa.GUARD, "transition": tr.GUARD},
    }


def check() -> bool:
    """The lazy guard of every static-scale route, once per forward, outside any graph (one host sync per route that ran): True when some
    call saturated the e4m3 range since the last check; the saturated routes' scales were raised in place (captured graphs follow), so the
    caller runs that forward again."""
    ta, tm, opm, pwa, tr, _td, _ad = _modules()
    hit = False
    for m in (tm, opm, tr):
        peak = m.overflow_peak()
        if peak > 448.0:
            m.rescale_all(peak)
            hit = True
    if pwa.take_overflow():
        pwa.rescale_all()
        hit = True
    if ta.take_overflow():
        ta.rescale_all(4.0)
        hit = True
    return hit
