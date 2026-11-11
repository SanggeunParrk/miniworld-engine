"""Prebuild every candidate schedule of the native CUDA ``choose`` sites, for tuning (opt-in), on a CPU node (no GPU).

    python -m miniworld_engine.kernels._prebuild_cuda [--arch sm90a] [--list] [--check] [--jobs N] [--match REGEX]
                                                      [--stop-after SECONDS] [--cutlass DIR] [--mathdx DIR]

A native kernel picks its schedule at run time with ``_launch_config.choose(family, op, dims, dtype, candidates, run=...)``. Each
candidate is a compiled artifact of its own (or several share one). ``_prebuild`` builds what the runtime needs: each site's default
and the configs the launch-config tables hold. The pretuner times every candidate a site passes, and skips one whose artifact is
absent (``not prebuilt``: it compiles nothing). This tool builds the rest: for every site in :data:`SITES` and every requirements
shape that reaches it (``_prebuild.family_rows`` of the site's family, as the family's ``prebuild`` recipe reads them), the FULL
candidate list the site passes to ``choose``, through the family's own loader. Planning, content keys, the cache, the parallel
builder and resume are ``_prebuild``'s own (``record`` / ``build`` / ``present``): an artifact already present is skipped, a killed
job leaves nothing partial, ``--stop-after`` and a resubmit continue. No ``prebuilt-<arch>.json`` is written: candidates are not part
of a bundle.

``--list`` prints per site the keys it reaches (``op dtype|dims``) with their candidate counts, then the artifacts and whether each is
built. ``--check`` cross-checks :data:`SITES` against the code: every ``choose(`` call under kernels/ must be one of the sites.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import functools as ft
import os
import re
import sys
import time
from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import cast

from miniworld_engine.kernels import _nvcc, _prebuild
from miniworld_engine.kernels._prebuild import KERNELS, Target, family_rows, log


@dataclasses.dataclass
class Want:
    """One key a ``choose`` site is reached with: ``op``, ``label`` (``dtype|dims``), the candidates it passes, and per candidate the
    family loader that builds it with its arguments (``(loader, args)``; equal pairs are one build)."""

    op: str
    label: str
    candidates: list[dict]
    builds: list[tuple[Callable, tuple]]


def _uncached(fn):
    """``fn`` past its ``lru_cache`` (under planning a cached stand-in would record nothing)."""
    return getattr(fn, "__wrapped__", fn)


def _trimul_unit(native_residual: bool, cz: int, ch: int):
    """The packaged K1 / K3 unit of ``(cz, ch)``, as the TriMul recipe plans it."""
    from miniworld_engine.kernels.trimul_inproj.cuda import h100_inference as HI

    # the method on a bare namespace: only `unit` runs, without the device setup of `_Kernels()`
    return HI._Kernels.unit(cast("HI._Kernels", SimpleNamespace(_units={}, native_residual=native_residual)), cz, ch)


# --------------------------------------------------------------------------------------------------------------- sites
def transition_sm90(t: Target) -> Iterable[Want]:
    """``transition/cuda/fused_sm90a.py::_select_schedule``: the D = 128 fused Transition (n = 4 / 2). Candidates
    ``transition_candidates(sms, backward, n)``, one build per ``(ctas, dw_repl, save, n)``. From the rows (the Transition recipe's
    ``_shapes``): eval -> the non-saving forward; train -> the saving forward and the backward; ``float32`` rows never reach it (bf16
    activations only), and a row count that is not a whole number of 128-row tiles is not taken (``supported``)."""
    if t.arch != "sm90":
        return
    from miniworld_engine.kernels.transition.cuda import _shapes
    from miniworld_engine.kernels.transition.cuda import fused_sm90a as F

    for (d, n), per in sorted(_shapes(family_rows("transition")).items()):
        if d != 128 or n not in F.NS:
            continue
        for (dtype, mode), counts in sorted(per.items()):
            if dtype == "float32":
                continue
            save = mode == "train"
            for backward in (False, True) if save else (False,):
                grid = F.transition_candidates(t.sms, backward=backward, n=n)
                op = "transition_bwd_residual_sm90_cuda" if backward else "transition_fwd_residual_sm90_cuda"
                flag = f"FP32_DW={int(dtype == 'master')}" if backward else f"SAVE={int(save)}"
                for m in sorted(c for c in counts if c % F.ROWS == 0):
                    yield Want(
                        op,
                        f"bfloat16|{flag},K=128,M={m},ND={128 * n},SMS={t.sms} ({dtype} {mode})",
                        grid,
                        [(_uncached(F._build_ext), (c["ctas"], c["dw_repl"], save, n)) for c in grid],
                    )


def trimul_sm90(t: Target) -> Iterable[Want]:
    """``trimul_inproj/cuda/h100_inference.py::_select_config``: the packaged K1 / K3 TriMul inference. Candidates
    ``trimul_candidates(cz, ch, L, direction)`` (+ the L128 short-tile schedule the site puts first); every K1 / K3 variant is an
    instantiation inside the one unit of its ``(cz, ch)`` (the D384 / L512 outgoing native-residual unit apart), so the candidates of
    a key share that artifact. From the rows (the TriMul recipe's ``_cells``): the eval cells the tile table has, their lengths, one
    direction (0) for a bidirectional cell, both (1, 2) for a single-direction one."""
    if t.arch != "sm90":
        return
    from miniworld_engine.kernels.trimul_inproj.cuda import _cells
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_infer_kernel as K
    from miniworld_engine.kernels.trimul_inproj.cuda import h100_inference as HI

    for c in _cells(family_rows("trimul_inproj")):
        mode = c.modes.get("eval")
        if mode is None or ("sm_90a", c.d, c.ch, "b") not in K.TILE_TABLE:
            continue
        for length in sorted(mode.lengths):
            for direction in (0,) if c.bidir else (1, 2):
                grid = HI.trimul_candidates(c.d, c.ch, length, direction)
                native = c.d == c.ch == 384 and length == 512 and direction == 1
                yield Want(
                    "trimul_fwd_sm90_cuda",
                    f"bfloat16|CH={c.ch},CZ={c.d},DIR={direction},L={length}",
                    grid,
                    [(_trimul_unit, (native, c.d, c.ch))] * len(grid),
                )


#: ``kernels/``-relative ``file::function`` of each ``choose`` call -> (family, the function enumerating what it is reached with).
SITES: dict[str, tuple[str, Callable[[Target], Iterable[Want]]]] = {
    "transition/cuda/fused_sm90a.py::_select_schedule": ("transition", transition_sm90),
    "trimul_inproj/cuda/h100_inference.py::_select_config": ("trimul_inproj", trimul_sm90),
}


# --------------------------------------------------------------------------------------------------------------- plan
def plan(t: Target) -> tuple[list[_nvcc.Spec], dict[str, list[tuple[Want, list[str]]]], list[str]]:
    """Every artifact of every candidate of every site (deduplicated by key, plan order); per site its wants with the keys of
    their candidates' artifacts; the problems found."""
    specs: dict[str, _nvcc.Spec] = {}
    per_site: dict[str, list[tuple[Want, list[str]]]] = {}
    problems: list[str] = []
    with _nvcc.plan_session():  # the loaders' caches forget the stand-ins when the plan ends
        for site, (family, enumerate_) in SITES.items():
            try:
                wants = list(enumerate_(t))
            except Exception as exc:  # reported, the other sites still plan
                problems.append(f"{site}: enumeration failed: {type(exc).__name__}: {exc}")
                continue
            by_build: dict[tuple, list[str]] = {}
            rows = per_site.setdefault(site, [])
            for w in wants:
                keys: list[str] = []
                for fn, args in w.builds:
                    if (fn, args) not in by_build:
                        got = _prebuild.record(family, [ft.partial(fn, *args)], problems)
                        by_build[fn, args] = [s.key for s in got]
                        for s in got:
                            specs.setdefault(s.key, s)
                    keys += [k for k in by_build[fn, args] if k not in keys]
                rows.append((w, keys))
    return list(specs.values()), per_site, problems


def choose_sites() -> set[str]:
    """``file::function`` of every ``choose(`` / ``_launch_config.choose(`` call under kernels/ (static, from the source)."""
    out = set()
    skip = {KERNELS / "_launch_config.py", KERNELS / "_pretune.py", KERNELS / "_prebuild_cuda.py"}
    for path in sorted(KERNELS.rglob("*.py")):
        if path in skip:
            continue
        tree = ast.parse(path.read_text())

        def visit(node: ast.AST, fn: str) -> None:
            for child in ast.iter_child_nodes(node):
                name = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
                if isinstance(child, ast.Call):
                    f = child.func
                    if (isinstance(f, ast.Name) and f.id == "choose") or (isinstance(f, ast.Attribute) and f.attr == "choose"):
                        out.add(f"{path.relative_to(KERNELS).as_posix()}::{fn}")  # noqa: B023 -- visit runs inside this iteration
                visit(child, name)

        visit(tree, "<module>")
    return out


# --------------------------------------------------------------------------------------------------------------- entry point
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m miniworld_engine.kernels._prebuild_cuda", description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--arch", default="sm90a", help="target arch (default sm90a = H100)")
    ap.add_argument("--jobs", type=int, default=0, help=f"parallel builds (default: min(CPUs, memory / {_prebuild.PEAK_GB} GB))")
    ap.add_argument("--match", default=None, help="keep only artifacts whose name matches this regular expression")
    ap.add_argument("--list", action="store_true", help="print the sites' keys, candidates and artifacts; build nothing")
    ap.add_argument("--check", action="store_true", help="cross-check SITES against the choose() calls in the code; build nothing")
    ap.add_argument("--stop-after", type=float, default=None, help="start no build after this many seconds")
    ap.add_argument("--cutlass", default=None, help="CUTLASS 4.2 checkout (sets CUTLASS_PATH for the builds)")
    ap.add_argument("--mathdx", default=None, help="mathdx root (sets MATHDX_HOME for the builds)")
    args = ap.parse_args(argv)

    if args.check:
        code = choose_sites()
        missing, stale = sorted(code - SITES.keys()), sorted(SITES.keys() - code)
        log(f"choose() sites: {len(code)} in the code, {len(code & SITES.keys())} in SITES")
        for s in missing:
            log(f"  not in SITES (its candidates are never prebuilt): {s}")
        for s in stale:
            log(f"  in SITES but no choose() call there: {s}")
        return 1 if missing or stale else 0
    t = _prebuild.target(args.arch)
    if args.cutlass:
        os.environ["CUTLASS_PATH"] = args.cutlass
    if args.mathdx:
        os.environ["MATHDX_HOME"] = args.mathdx
    _nvcc.ensure_cuda_home()  # before planning: the key classifies include directories under CUDA_HOME
    started = time.monotonic()
    specs, per_site, problems = plan(t)
    if args.match:
        specs = [s for s in specs if re.search(args.match, s.name)]
    absent = {s.key for s in specs if _prebuild.present(s) is None}
    names = {s.key: s.name for s in specs}
    for site, rows in per_site.items():
        cands = sum(len(w.candidates) for w, _ in rows)
        arts = {k for _, keys in rows for k in keys if k in names}
        log(f"{site}: {len(rows)} keys, {cands} candidates, {len(arts)} artifacts ({len(arts & absent)} not built)")
        if args.list:
            for w, keys in rows:
                keys = [k for k in keys if k in names]
                log(
                    f"  {w.op} {w.label}: {len(w.candidates)} candidates -> {len(keys)} artifacts, "
                    f"{sum(k in absent for k in keys)} not built"
                )
    log(
        f"plan for {t.arch}: {len(specs)} artifacts in {time.monotonic() - started:.1f} s; {len(absent)} not built yet; "
        f"cache {_nvcc.engine_jit_root()}"
    )
    for p in problems:
        log(f"problem: {p}")
    if args.list:
        for s in specs:
            log(f"  {'have' if s.key not in absent else 'todo'} {s.kind:7s} {s.name}  [{s.key[:8]}]  {s.site}")
        return 0
    results = _prebuild.build(
        specs, args.jobs or _prebuild.default_jobs(), args.stop_after, {"sm80": "8.0", "sm90": "9.0a", "sm100": "10.0a"}[t.arch]
    )
    failed = {k: v for k, v in results.items() if v[1] is not None}
    left = sum(_prebuild.present(s) is None for s in specs)
    log(
        f"done: {len(results) - len(failed)} built, {len(failed)} failed, {left} of {len(specs)} still absent; "
        f"{sum(v[0] for v in results.values()):.0f} s of builds in {time.monotonic() - started:.0f} s"
    )
    return 1 if failed or problems or left else 0


if __name__ == "__main__":
    sys.exit(main())
