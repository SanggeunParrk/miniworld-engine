"""One switch for the engine's FP8 (e4m3) inference routes on H100 (engine 3.0.0; every route is opt-in and off by default).

    MINIWORLD_FP8=safe python run.py           # environment: before the first import of the engine's kernels
    miniworld_engine.fp8.enable("safe")        # API: at run time (after import), before the first forward

``"safe"`` is the set validated end to end on AF3 (FoldBench, 27 targets, 0 flagged, lDDT +0.0000 against bf16, whole forward 0.97-0.80x;
docs/fp8.md "fp8safe2"): TriAttn FP8 PV (fused d_pair-128 front + the unfused / front-made sites) with e4m3 q / k on the fused route,
TriMul e4m3 planes (static scales, calibrated), OuterProductMean e4m3 product (weight-bound scales), PWA e4m3 key contraction,
Transition e4m3 GEMMs (calibrated), token DiT e4m3 expand + SwiGLU (not ``qkvg``), atom DiT e4m3 row kernels (``pre`` + ``post``).
Each route keeps its own switch (``MINIWORLD_TRIMUL_FP8``, ``MINIWORLD_OPM_FP8``, ``MINIWORLD_PWA_FP8``, ``MINIWORLD_TRANSITION_FP8``,
``MINIWORLD_TOKEN_DIT_FP8``, ``MINIWORLD_ATOMDIT_FP8`` and their ``_GUARD`` / gate variables; TriAttn: ``h100_sm90.enable_fp8_pv_for``):
with ``MINIWORLD_FP8`` set, a per-route variable that is set explicitly (also to the empty string = off) wins over the preset.

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
#: the TriAttn part of each preset: (call-site kinds of the FP8 PV route, e4m3 q / k on the fused route)
TRIATTN = MappingProxyType({"safe": (frozenset({"fused", "unfused"}), True)})


def preset() -> str | None:
    """The preset ``MINIWORLD_FP8`` names (None: unset, ``0``, ``off`` or ``none``)."""
    v = os.environ.get("MINIWORLD_FP8", "").strip().lower()
    if v in ("", "0", "off", "none"):
        return None
    if v not in PRESETS:
        raise ValueError(f"MINIWORLD_FP8={v!r}: unknown preset (use one of {', '.join(PRESETS)}, or off)")
    return v


def env(name: str, default: str | None = None) -> str | None:
    """A route module's switch: the variable when it is set (also when set to ""), else the ``MINIWORLD_FP8`` preset's value, else ``default``."""
    if name in os.environ:
        return os.environ[name]
    p = preset()
    if p is not None and name in PRESETS[p]:
        return PRESETS[p][name]
    return default


def _modules() -> tuple[Any, ...]:
    from miniworld_engine.kernels.conditioned_transition.cuda import token_dit_fp8 as td
    from miniworld_engine.kernels.integrations import local_dit_h100 as ad
    from miniworld_engine.kernels.integrations import opm_fp8 as opm
    from miniworld_engine.kernels.integrations import pwa_fp8 as pwa
    from miniworld_engine.kernels.transition.cuda import fp8_sm90a as tr
    from miniworld_engine.kernels.triangle_attention.cuda import h100_sm90 as ta
    from miniworld_engine.kernels.trimul_inproj.cuda import h100_b200style_fp8 as tm

    return ta, tm, opm, pwa, tr, td, ad


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
    """Switch every route of preset ``name`` on (run time; replaces the routes' current switches). ``model``: the TriAttn part only on the
    call-site kinds ``h100_sm90.FP8_PV_WHITELIST`` lists for that model (a model it does not list keeps the fp16 TriAttn core; the other
    routes are on regardless); None: the preset's kinds. ``guard``: ``"sync"`` / ``"lazy"`` for TriMul / OPM / PWA / Transition (None: as they are)."""
    if name not in PRESETS:
        raise ValueError(f"unknown FP8 preset {name!r} (use one of {', '.join(PRESETS)})")
    ta, tm, opm, pwa, tr, td, ad = _modules()
    env_of = PRESETS[name]
    tm.POLICY = env_of.get("MINIWORLD_TRIMUL_FP8") or None
    opm.POLICY = env_of.get("MINIWORLD_OPM_FP8") or None
    pwa.ENABLED = env_of.get("MINIWORLD_PWA_FP8") == "1"
    tr.POLICY = env_of.get("MINIWORLD_TRANSITION_FP8") or None
    td.PLACES = td._parse(env_of.get("MINIWORLD_TOKEN_DIT_FP8", ""))
    ad.enable_fp8(ad._fp8_parts(env_of.get("MINIWORLD_ATOMDIT_FP8", "")))
    kinds, qk = TRIATTN[name]
    if model is not None:
        kinds = ta.FP8_PV_WHITELIST.get(model, frozenset())
    enable_triattn(kinds, qk)
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
