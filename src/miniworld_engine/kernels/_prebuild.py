"""Prebuild every hand-written CUDA / C++ artifact of the engine ahead of time, in parallel, on a CPU node (no GPU).

    python -m miniworld_engine.kernels._prebuild [--arch sm90a] [--jobs N] [--only FAMILY ...] [--list] [--check]
                                                 [--stop-after SECONDS] [--cutlass DIR] [--mathdx DIR]

**Build list from the requirements.** Each family names one function in :data:`RECIPES`, ``prebuild(target, rows)`` (a family with no
shape-dependent build, ``prebuild(target)``), which yields calls of the family's own loaders (the functions the runtime calls) for every
variant the requirements ask for on ``target``: ``rows`` are :func:`family_rows` (the requirements_module.csv rows of the module classes
that launch the family's kernels, by the derive rule of ``requirements/README.md``), so a kernel no requirements row needs is not built.
``--rows`` prints them, ``--compare INDEX.json`` diffs the plan against an earlier ``prebuilt-<arch>.json``. The calls
run under ``_nvcc.planning()``: every ``load_extension`` / ``load_inline_extension`` / ``cubin`` records what it would build
(name, content key, the arguments that rebuild it) and returns a stand-in, so a planned artifact has the key the runtime
computes. Nothing is recorded from a run of the models. ``--check`` cross-checks the plan against the code (every call site of
a build helper reached, every ``.cu`` / ``.cpp`` file compiled by some artifact) and ``requirements/requirements_kernel.csv``.

**Arch.** ``--arch`` (default ``sm90a``: H100) picks the recipes' variants (and the target's SM count, which a few builds are
specialized for). No device is queried: the arch an artifact is built for is the one its flags request.

**Cache.** The artifacts go where the runtime looks: ``_nvcc.engine_jit_root()`` (``~/.cache/miniworld_engine_jit``), at
``torch_extensions/<name>-<key16>/`` and ``<family>/<stem>-<key16>.cubin``. The key (``_codegraph``) covers the sources and
headers reached, flags, torch / CUDA / Python versions. A key present in any root (``_nvcc.artifact_roots()``) is skipped, so a
job that hits its time limit is resubmitted and continues. A build writes into a private temporary path that is renamed into
place when complete, under a flock the kernel drops if the builder dies: a killed job leaves no partial artifact and no lock.
``prebuilt-<arch>.json`` in the cache root lists the plan's artifacts (relative paths): a bundle for ``<arch>`` is a copy of
those paths into ``miniworld_engine/_prebuilt/`` (``_nvcc.PACKAGED_ROOT``, same layout, looked up first).

**Parallel.** A process pool of builds (each torch extension with ``MAX_JOBS=1``). The worker count defaults to
min(CPUs, memory / :data:`PEAK_GB`); PEAK_GB is the measured peak RSS of the heaviest single build plus margin.

Triton kernels are out of scope (they JIT in seconds). The other tuning candidates of the native ``choose`` sites (only the defaults and
the tables' entries are planned here) are built by ``_prebuild_cuda`` (opt-in, for the pretuner).
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures as cf
import dataclasses
import hashlib
import importlib
import inspect
import json
import multiprocessing
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

from miniworld_engine.kernels import _nvcc

KERNELS = Path(__file__).resolve().parent

#: Peak resident memory of one build, GB: the worst single artifact measured on the H100 prebuild (2026-11-09: 4.2 GB, a
#: cuBLASDx transition extension; CUTLASS triangle-attention and TriMul cubins stay under 3 GB) plus margin.
PEAK_GB = 5.0


@dataclasses.dataclass(frozen=True)
class Target:
    arch: str  # "sm80" / "sm90" / "sm100"
    sms: int  # SM count of the card the SM-count-specialized builds are made for
    gpu: str  # launch-config table key (_launch_config.gpu_key) of that card


TARGETS = {"sm80": Target("sm80", 108, "a100_sm80"), "sm90": Target("sm90", 132, "h100_sm90"), "sm100": Target("sm100", 148, "b200_sm100")}


def target(arch: str) -> Target:
    """``sm90a`` / ``sm_90a`` / ``90a`` / ``sm90`` -> the sm90 target."""
    key = "sm" + arch.lower().removeprefix("sm").removeprefix("_").rstrip("a")
    if key not in TARGETS:
        raise SystemExit(f"unknown arch {arch!r}; known: {', '.join(TARGETS)}")
    return TARGETS[key]


#: family -> "module:function"; ``function(target)`` yields zero-argument calls of the family's loaders.
RECIPES: dict[str, str] = {
    "cast": "miniworld_engine.kernels._cast_cuda:prebuild",
    "layernorm": "miniworld_engine.kernels.layernorm.cuda:prebuild",
    "sampler_ops": "miniworld_engine.kernels.sampler_ops.cuda:prebuild",
    "outer_product_mean": "miniworld_engine.kernels.outer_product_mean.cuda:prebuild",
    "dit_variants": "miniworld_engine.kernels.dit_variants.cuda:prebuild",
    "transition": "miniworld_engine.kernels.transition.cuda:prebuild",
    "triangle_attention": "miniworld_engine.kernels.triangle_attention.cuda:prebuild",
    "augmented_attention": "miniworld_engine.kernels.augmented_attention.cuda:prebuild",
    "swa_dit": "miniworld_engine.kernels.swa_dit.cuda:prebuild",
    "smooth_lddt": "miniworld_engine.kernels.smooth_lddt.cuda:prebuild",
    "bias_only_dit": "miniworld_engine.kernels.bias_only_dit.cuda:prebuild",
    "conditioned_transition": "miniworld_engine.kernels.conditioned_transition.cuda:prebuild",
    "integrations": "miniworld_engine.kernels.integrations:prebuild",
    "trimul_inproj": "miniworld_engine.kernels.trimul_inproj.cuda:prebuild",
    "pair_embed_adaln": "miniworld_engine.kernels.pair_embed_adaln.cuda:prebuild",
    "pair_embed_geom": "miniworld_engine.kernels.pair_embed_geom.cuda:prebuild",
    "token_pair_init": "miniworld_engine.kernels.token_pair_init.cuda:prebuild",
}


def _kv(text: str) -> dict[str, str]:
    """``a=1;b=x|y`` -> ``{"a": "1", "b": "x|y"}`` (the csv's dims / options cell)."""
    return dict(kv.split("=", 1) for kv in text.split(";") if kv)


_ROLE_MODES = {"forward": ("eval", "train"), "forward_infer": ("eval",), "forward_train": ("train",), "backward": ("train",)}


def family_rows(family: str) -> list[dict]:
    """The requirements every kernel of ``family`` must serve, by the derive rule (``requirements/README.md``): for each
    requirements_kernel.csv row of the family, the requirements_module.csv rows of the module classes that launch it, restricted to
    the modes its role runs in (``forward``: the row's modes; ``forward_infer``: eval; ``forward_train`` / ``backward``: train).

    One dict per (module row, dtype, mode): ``module``, ``model`` (selector keys), ``stream``, ``lengths`` and ``depths`` (ints),
    ``samples`` (ints: the row's eval_samples / train_samples for this mode, ``[1]`` when empty), ``dims`` and
    ``options`` (cell strings; a ``|`` lists the values the row covers), ``dtype`` (``bfloat16`` / ``float32`` / ``master``),
    ``mode`` (``eval`` / ``train``), ``kernels`` (the family's kernel names that need it). A family's ``prebuild(target, rows)``
    expands its loader variants from these instead of listing them."""
    import csv

    root = KERNELS.parent / "requirements"
    kernels = [r for r in csv.DictReader((root / "requirements_kernel.csv").open()) if r["family"] == family]
    modules = list(csv.DictReader((root / "requirements_module.csv").open()))
    out: dict[tuple, dict] = {}
    for k in kernels:
        classes = set(k["modules"].split("|"))
        for r in modules:
            if r["module"] not in classes:
                continue
            for mode in _ROLE_MODES[k["role"]]:
                if mode not in r["modes"].split("|"):
                    continue
                for dtype in r["dtypes"].split("|"):
                    key = (r["module"], r["stream"], r["dims"], r["options"], r["lengths"], r["depths"], dtype, mode)
                    row = out.setdefault(key, {
                        "module": r["module"], "model": r["model"], "stream": r["stream"],
                        "lengths": [int(x) for x in r["lengths"].split("|") if x],
                        "depths": [int(x) for x in r["depths"].split("|") if x],
                        "samples": [int(x) for x in r["eval_samples" if mode == "eval" else "train_samples"].split("|") if x] or [1],
                        "dims": r["dims"], "options": r["options"], "dtype": dtype, "mode": mode, "kernels": set()})
                    row["kernels"].add(k["name"])
    return list(out.values())


def requirement_dims(*modules: str) -> list[dict[str, str]]:
    """The ``dims`` (constructor arguments, as strings) of every requirements_module.csv row of ``modules``: the shapes a
    family whose builds are specialized per model width serves (``requirements/README.md``)."""
    import csv

    rows = csv.DictReader((KERNELS.parent / "requirements" / "requirements_module.csv").open())
    return [dict(kv.split("=", 1) for kv in r["dims"].split(";") if kv) for r in rows if r["module"] in modules]



# --------------------------------------------------------------------------------------------------- planning
def plan(t: Target, only: Iterable[str] = ()) -> tuple[list[_nvcc.Spec], list[str]]:
    """Every artifact the recipes ask for on ``t`` (deduplicated by key, plan order), and the problems found."""
    specs: dict[str, _nvcc.Spec] = {}
    problems: list[str] = []
    only = set(only)
    for family, where in RECIPES.items():
        if only and family not in only:
            continue
        mod, fn = where.split(":")
        try:
            recipe = getattr(importlib.import_module(mod), fn)
            # a recipe taking ``rows`` expands its variants from the requirements (family_rows); one taking only the target is
            # a hand-written list (being replaced)
            calls = list(recipe(t, family_rows(family)) if len(inspect.signature(recipe).parameters) > 1 else recipe(t))
        except Exception as exc:  # reported, the other families still plan
            problems.append(f"{family}: recipe {where} failed: {type(exc).__name__}: {exc}")
            continue
        for s in record(family, calls, problems):
            specs.setdefault(s.key, s)
    return list(specs.values()), problems


def record(family: str, calls: Iterable, problems: list[str]) -> list[_nvcc.Spec]:
    """Run ``calls`` (zero-argument calls of a family's loaders) under ``_nvcc.planning()``: the specs they record, each tagged with
    ``family``, in call order. A call that raises without recording anything is appended to ``problems``. (Also the planner of
    ``_prebuild_cuda``.)"""
    out: list[_nvcc.Spec] = []
    for i, call in enumerate(calls):
        error = None
        with _nvcc.planning() as got:
            try:
                call()
            except _nvcc.Planned:
                pass  # the loader used its stand-in (after recording, or one an lru_cache kept from an earlier call)
            except Exception as exc:  # reported
                error = f"{type(exc).__name__}: {exc}"
        if error is not None and not got:
            problems.append(f"{family}: call {i} ({getattr(call, '__name__', call)}) recorded nothing: {error}")
        for s in got:
            s.call["_family"] = family
        out += got
    return out


def present(spec: _nvcc.Spec) -> Path | None:
    for root in _nvcc.artifact_roots():
        p = root / spec.rel
        if (p / "artifact.json").is_file() if spec.kind == "ext" else p.is_file():
            return p
    return None


# --------------------------------------------------------------------------------------------------- building
def _worker_init(arch_list: str) -> None:
    _nvcc._BUILD_ONLY = True
    # torch asks the GPU for the arch of a CUDA build that names none (an inline C++ extension with_cuda): there is no GPU
    # here, so name the target (every .cu of the engine passes its own -gencode / -arch, which torch then leaves alone)
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
    os.environ["MAX_JOBS"] = "1"  # one compiler per worker: the pool is the parallelism


def _build(spec: _nvcc.Spec) -> tuple[str, float, str | None, int]:
    """Build one artifact (in a worker). Returns (key, seconds, error or None, peak child RSS in MB so far)."""
    import resource

    from miniworld_engine.kernels import _codegraph as cg

    started = time.perf_counter()
    err = None
    try:
        for path, text in spec.overlay.items():
            cg.publish(path, text)
        c = {k: v for k, v in spec.call.items() if not k.startswith("_")}
        if spec.kind == "ext":
            _nvcc._load(c["kind_label"], c["name"], c["kwargs"], c["sources"], c["inline"])
        elif spec.kind == "cubin":
            _nvcc.cubin(c.pop("name"), c.pop("flags"), **c)
        else:
            raise RuntimeError(f"unknown kind {spec.kind}")
        if present(spec) is None:
            err = f"built, but not at {spec.rel}: the worker computed another key than the plan"
    except Exception as exc:  # reported per artifact
        err = f"{type(exc).__name__}: {exc}"
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024
    return spec.key, time.perf_counter() - started, err, rss


def _derive(spec: _nvcc.Spec, by_key: dict[str, _nvcc.Spec]) -> str | None:
    """Make a derived cubin (a post-link patch) in this process, once its base is built. Returns an error or None."""
    base = by_key.get(spec.call["base_key"])
    if base is None or present(base) is None:
        return "its base cubin is not built"
    c = {k: v for k, v in base.call.items() if not k.startswith("_")}
    for path, text in base.overlay.items():
        importlib.import_module("miniworld_engine.kernels._codegraph").publish(path, text)
    base_path = _nvcc.cubin(c.pop("name"), c.pop("flags"), **c)
    where, args = spec.call["replay"]
    mod, fn = where.split(":")
    getattr(importlib.import_module(mod), fn)(base_path, *args)
    return None if present(spec) else "the replay did not produce it (its patch declined)"


def default_jobs() -> int:
    cpus = len(os.sched_getaffinity(0))
    mem_mb = int(os.environ.get("SLURM_MEM_PER_NODE") or 0)
    if not mem_mb:
        mem_mb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2**20
    return max(1, min(cpus, int(mem_mb / 1024 / PEAK_GB)))


def build(specs: list[_nvcc.Spec], jobs: int, stop_after: float | None = None, arch_list: str = "9.0") -> dict[str, tuple[float, str | None]]:
    """Build the absent ones; one log line per finished artifact. Returns key -> (seconds, error)."""
    _sweep_tmp()
    todo = [s for s in specs if s.kind != "derived" and present(s) is None]
    results: dict[str, tuple[float, str | None]] = {}
    t0 = time.monotonic()
    log(f"{len(todo)} to build with {jobs} workers ({len(specs) - len(todo)} present or derived)")
    if todo:
        ctx = multiprocessing.get_context("spawn")
        with cf.ProcessPoolExecutor(jobs, mp_context=ctx, initializer=_worker_init, initargs=(arch_list,)) as pool:
            pending = {}
            queue = list(todo)
            while queue or pending:
                while queue and len(pending) < jobs and (stop_after is None or time.monotonic() - t0 < stop_after):
                    s = queue.pop(0)
                    pending[pool.submit(_build, s)] = s
                if not pending:
                    log(f"stop after {stop_after:.0f} s: {len(queue)} not started (rerun to continue)")
                    break
                done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                for f in done:
                    s = pending.pop(f)
                    key, seconds, err, rss = f.result()
                    results[key] = (seconds, err)
                    status = "ok" if err is None else "FAIL"
                    log(f"{status} {seconds:7.1f} s  {s.name}  [{key[:8]}]  ({len(results)}/{len(todo)}, "
                        f"worker peak child RSS {rss} MB, {time.monotonic() - t0:.0f} s)"
                        + ("" if err is None else "\n    " + _head(err)))
    by_key = {s.key: s for s in specs}
    for s in specs:
        if s.kind == "derived" and present(s) is None:
            started = time.perf_counter()
            try:
                err = _derive(s, by_key)
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
            results[s.key] = (time.perf_counter() - started, err)
            log(f"{'ok' if err is None else 'FAIL'} {results[s.key][0]:7.1f} s  {s.name}  [{s.key[:8]}]  (derived)"
                + ("" if err is None else "\n    " + _head(err)))
    return results


def _sweep_tmp(age: float = 3600.0) -> None:
    """Remove private build directories / files a killed build left (older than ``age``; a live build's are younger)."""
    import shutil

    root = _nvcc.engine_jit_root()
    now = time.time()
    for p in list(root.glob(".*.tmp-*")) + list(root.glob("*/.*.tmp-*")):
        try:
            if now - p.stat().st_mtime > age:
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        except OSError:  # noqa: PERF203 -- per path: one vanished path must not stop the sweep
            pass


def _head(err: str, lines: int = 12) -> str:
    keep = [ln for ln in err.splitlines() if ln.strip()]
    errs = [ln for ln in keep if "error" in ln.lower()]
    return "\n    ".join((errs or keep)[:lines])


def log(msg: str) -> None:
    print(f"[prebuild] {msg}", flush=True)


#: Human-managed version of how the prebuilder builds (compiler flags, toolchain handling, what it plans): bump it when a change can make
#: the built kernels compute or run differently, not for refactors. Recorded in every measurement; the judge asks once whether the
#: results made with the previous builds still stand.
PREBUILD_VERSION = "1"


def _artifact_sha(p: Path) -> str:
    """Hash of the BUILT artifact (a cubin file; for an extension the compiled ``*.so`` files, else its ``artifact.json``): the identity of
    what a kernel launch runs, next to the content ``key`` (the identity of what was asked for). Equal binaries under different keys
    (a comment, a rename) are visible by comparing this."""
    h = hashlib.sha256()
    files = [p] if p.is_file() else sorted(p.glob("*.so")) or [p / "artifact.json"]
    for f in files:
        h.update(f.name.encode() if len(files) > 1 else b"")
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _file_sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def prebuild_meta(t: Target, specs: list[_nvcc.Spec]) -> dict:
    """The run's identity: the prebuilder's version, the toolchain, the requirements it planned from and the plan itself."""
    import torch

    env: dict = {"host": platform.node(), "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda}
    try:
        env["nvcc"] = subprocess.run([_nvcc.nvcc_path() or "nvcc", "--version"], capture_output=True, text=True,
                                     timeout=20).stdout.strip().splitlines()[-1]
    except Exception:  # recorded as absent
        env["nvcc"] = None
    req = KERNELS.parent / "requirements"
    return {"prebuild_version": PREBUILD_VERSION, "arch": t.arch, "sms": t.sms, "env": env,
            "requirements": {"module_sha": _file_sha(req / "requirements_module.csv"), "kernel_sha": _file_sha(req / "requirements_kernel.csv")},
            "plan_sha": hashlib.sha256("\n".join(sorted(s.key for s in specs)).encode()).hexdigest()[:16], "artifacts": len(specs)}


def write_index(t: Target, specs: list[_nvcc.Spec]) -> Path:
    """``prebuilt-<arch>.json`` in the cache root: the plan's artifacts that exist, by relative path, each with its content ``key`` and the
    ``sha`` of the built binary, and ``meta`` (version, toolchain, requirements hashes, plan hash)."""
    out = _nvcc.engine_jit_root() / f"prebuilt-{t.arch}.json"
    rows = []
    for s in specs:
        p = present(s)
        if p is not None:
            rows.append({"kind": s.kind, "name": s.name, "key": s.key, "sha": _artifact_sha(p), "path": s.rel, "site": s.site,
                         "family": s.call.get("_family")})
    _nvcc._write_json(out, {"arch": t.arch, "meta": prebuild_meta(t, specs), "artifacts": rows})
    return out


# --------------------------------------------------------------------------------------------------- cross-check
_BUILD_CALLS = {"load_extension", "load_inline_extension", "cubin", "derived_cubin", "compile", "compile_text"}


def call_sites() -> set[str]:
    """``file::function`` of every call of a build helper in kernels/ (static, from the source): ``load_extension`` /
    ``load_inline_extension`` (also under an alias), ``_nvcc.cubin`` / ``_nvcc.derived_cubin``, TriMul's ``T.compile`` /
    ``T.compile_text``."""
    sites = set()
    skip = {KERNELS / "_nvcc.py", KERNELS / "_prebuild.py", KERNELS / "trimul_inproj/cuda/_h100_runtime.py"}
    for path in sorted(KERNELS.rglob("*.py")):
        if path in skip or path.name == "setup.py":
            continue
        tree = ast.parse(path.read_text())
        alias = {a.asname or a.name: a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                 for a in n.names if a.name in ("load_extension", "load_inline_extension")}

        def visit(node: ast.AST, fn: str) -> None:
            for child in ast.iter_child_nodes(node):
                name = fn
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = child.name
                if isinstance(child, ast.Call):
                    f = child.func
                    hit = (isinstance(f, ast.Name) and f.id in alias) or (  # noqa: B023 -- visit runs inside this iteration
                        isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and (
                            (f.value.id == "_nvcc" and f.attr in ("cubin", "derived_cubin", "load_extension"))
                            or (f.value.id == "T" and f.attr in ("compile", "compile_text"))))
                    if hit:
                        sites.add(f"{path.relative_to(KERNELS).as_posix()}::{fn}")  # noqa: B023 -- visit runs inside this iteration
                visit(child, name)

        visit(tree, "<module>")
    return sites


_GLOBAL = re.compile(r"__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?(?:static\s+|inline\s+)*void\s+"
                     r"(?:__launch_bounds__\s*\([^)]*\)\s*)?(\w+)")


def check(t: Target, specs: list[_nvcc.Spec], problems: list[str]) -> int:
    """Print the cross-check; returns the number of findings."""
    findings = 0
    planned = {s.site for s in specs}
    static = call_sites()
    missing = sorted(static - planned)
    log(f"call sites: {len(static)} in the code, {len(static & planned)} reached by the plan")
    for site in missing:
        log(f"  not reached: {site}")
    findings += len(missing)
    reached = {Path(p).resolve() for s in specs for p in s.files}
    natives = sorted(p for p in KERNELS.rglob("*") if p.suffix in (".cu", ".cpp") and "third_party" not in p.parts)
    orphans = [p for p in natives if p.resolve() not in reached]
    log(f"native sources: {len(natives)} .cu/.cpp files, {len(natives) - len(orphans)} compiled by a planned artifact")
    for p in orphans:
        log(f"  not compiled: {p.relative_to(KERNELS)}")
    findings += len(orphans)
    findings += _requirements(specs)
    for p in problems:
        log(f"problem: {p}")
    return findings + len(problems)


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if t}


def _requirements(specs: list[_nvcc.Spec]) -> int:
    """Each requirements_kernel.csv row against the kernels the plan compiles. The csv names kernels by role, not by symbol
    (``requirements/README.md``), so the match is by name tokens: every token of the row's name (family prefix and ``_sm90``
    dropped) must occur in the tokens of one ``__global__`` function name plus its file path, in the row's family first,
    then anywhere. A row named ``*_sm90`` is an H100 hand-CUDA kernel: one without a match is a finding. Other rows may be
    Triton kernels (out of scope) and are matched against the ``@triton.jit`` functions the same way."""
    import csv

    rows = list(csv.DictReader((KERNELS.parent / "requirements" / "requirements_kernel.csv").open()))

    def candidates(items: Iterable[tuple[Path, str]]) -> dict[str, list[tuple[str, set[str]]]]:
        out: dict[str, list[tuple[str, set[str]]]] = {}
        for path, fn in items:
            rel = path.resolve().relative_to(KERNELS)
            out.setdefault(rel.parts[0], []).append((f"{rel.as_posix()}::{fn}", _tokens(fn) | _tokens(rel.with_suffix("").as_posix())))
        return out

    files = {Path(p) for s in specs for p in s.files}
    cuda = candidates((p, g) for p in sorted(files) for g in _GLOBAL.findall(p.read_text(errors="replace")))
    triton = candidates((p, fn) for p in sorted(KERNELS.rglob("*.py"))
                        for fn in re.findall(r"@triton\.jit[^\n]*\n(?:[^\n]*\n){0,3}?def\s+(\w+)", p.read_text(errors="replace")))

    def hit(want: set[str], fam: str, table: dict) -> str | None:
        def ok(w: str, have: set[str]) -> bool:
            return w in have or (len(w) >= 3 and any(len(h) >= 3 and (w in h or h in w) for h in have))

        for pool in (table.get(fam, []), [c for f, cs in table.items() if f != fam for c in cs]):
            for name, have in pool:
                if all(ok(w, have) for w in want):
                    return name
        return None

    prefixes = {"conditioned_transition": ("cond_transition",)}
    counts = {"cuda": 0, "triton": 0, "none": 0}
    findings = []
    for r in rows:
        name, fam = r["name"], r["family"]
        stem = name
        for pre in (*prefixes.get(fam, ()), fam):
            if stem.startswith(pre + "_"):
                stem = stem[len(pre) + 1:]
                break
        h100 = stem.endswith("_sm90")
        want = _tokens(stem.removesuffix("_sm90")) or _tokens(fam)
        if hit(want, fam, cuda):
            counts["cuda"] += 1
        elif not h100 and hit(want, fam, triton):
            counts["triton"] += 1
        else:
            counts["none"] += 1
            if h100:
                findings.append(f"{name} ({fam}, {r['role']})")
    log(f"requirements_kernel.csv: {len(rows)} rows; {counts['cuda']} match a compiled __global__ function, "
        f"{counts['triton']} a Triton kernel (out of scope), {counts['none']} neither "
        f"({len(findings)} of them named *_sm90, i.e. H100 hand CUDA)")
    per_family: dict[str, int] = {}
    for s in specs:
        for p in s.files:
            fam = Path(p).resolve().relative_to(KERNELS).parts[0]
            per_family[fam] = per_family.get(fam, 0) + 1
    #: requirements families whose files live under another kernels/ directory (the training glue of PWA / OPM: kernels/integrations/csrc)
    lives_in = {"pair_weighted_averaging": "integrations", "outer_product_mean": "integrations"}
    findings.extend(f"family {fam}: has *_sm90 rows but no planned artifact compiles a file of kernels/{fam}/"
                    for fam in sorted({r["family"] for r in rows if r["name"].endswith("_sm90")})
                    if not per_family.get(fam) and not per_family.get(lives_in.get(fam, fam)))
    log("H100 families with *_sm90 rows and the planned artifacts compiling their files: " + ", ".join(
        f"{fam} {per_family.get(fam, 0)}" for fam in sorted({r["family"] for r in rows if r["name"].endswith("_sm90")})))
    for f in findings:
        log(f"  no compiled kernel matches: {f}")
    planned = {s.call.get("_family") for s in specs}
    required = {r["family"] for r in rows}
    # sampler-style families with one shape-independent build, and integrations (its kernels are listed under outer_product_mean /
    # pair_weighted_averaging), have no rows of their own by design
    for fam in sorted(f for f in planned if f and f not in required and f != "integrations"):
        log(f"  recipe without a requirements row: {fam}")
    for fam in sorted(f for f in required if f not in planned and f in cuda):
        log(f"  requirements family whose kernels are compiled only by other recipes: {fam}")
    return len(findings)


# --------------------------------------------------------------------------------------------------- entry point
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m miniworld_engine.kernels._prebuild", description=__doc__.split("\n\n")[0])
    ap.add_argument("--arch", default="sm90a", help="target arch (default sm90a = H100)")
    ap.add_argument("--jobs", type=int, default=0, help=f"parallel builds (default: min(CPUs, memory / {PEAK_GB} GB))")
    ap.add_argument("--only", nargs="*", default=(), help="families to plan (default: all)")
    ap.add_argument("--match", default=None, help="keep only artifacts whose name matches this regular expression")
    ap.add_argument("--reverse", action="store_true", help="build in reverse plan order (a second job then meets the first one in the middle)")
    ap.add_argument("--rows", action="store_true", help="print, per family, the requirements rows its kernels must serve; build nothing")
    ap.add_argument("--compare", default=None, metavar="INDEX.json",
                    help="diff the plan against a prebuilt-<arch>.json index (artifacts only in one of them, by family); build nothing")
    ap.add_argument("--list", action="store_true", help="print the plan and what is missing; build nothing")
    ap.add_argument("--check", action="store_true", help="cross-check the plan against the code and requirements")
    ap.add_argument("--stop-after", type=float, default=None, help="start no build after this many seconds")
    ap.add_argument("--cutlass", default=None, help="CUTLASS 4.2 checkout (sets CUTLASS_PATH for the builds)")
    ap.add_argument("--mathdx", default=None, help="mathdx root (sets MATHDX_HOME for the builds)")
    args = ap.parse_args(argv)

    t = target(args.arch)
    if args.cutlass:
        os.environ["CUTLASS_PATH"] = args.cutlass
    if args.mathdx:
        os.environ["MATHDX_HOME"] = args.mathdx
    if args.rows:
        for family in (args.only or RECIPES):
            rows = family_rows(family)
            combos = sorted({(r["module"], r["dtype"], r["mode"]) for r in rows})
            log(f"{family}: {len(rows)} rows; " + ", ".join(f"{m} {d}/{mo}" for m, d, mo in combos))
        return 0
    _nvcc.ensure_cuda_home()  # before planning: the key classifies include directories under CUDA_HOME
    started = time.monotonic()
    specs, problems = plan(t, args.only)
    if args.match:
        specs = [s for s in specs if re.search(args.match, s.name)]
    if args.reverse:
        specs = specs[::-1]
    by_kind: dict[str, int] = {}
    for s in specs:
        by_kind[s.kind] = by_kind.get(s.kind, 0) + 1
    absent = [s for s in specs if present(s) is None]
    log(f"plan for {t.arch}: {len(specs)} artifacts ({', '.join(f'{v} {k}' for k, v in sorted(by_kind.items()))}) from "
        f"{len(RECIPES) if not args.only else len(args.only)} families in {time.monotonic() - started:.1f} s; "
        f"{len(absent)} not built yet; cache {_nvcc.engine_jit_root()}")
    for p in problems:
        log(f"problem: {p}")
    if args.compare:
        old = {a["key"]: a for a in json.loads(Path(args.compare).read_text())["artifacts"]}
        new = {s.key: s for s in specs}
        fams = sorted({a.get("family") or "?" for a in old.values()} | {s.call.get("_family") or "?" for s in specs})
        log(f"compare with {args.compare}: {len(old)} artifacts there, {len(new)} planned, {len(old.keys() & new.keys())} in both")
        for fam in fams:
            gone = [a["name"] for k, a in old.items() if k not in new and (a.get("family") or "?") == fam]
            added = [s.name for k, s in new.items() if k not in old and (s.call.get("_family") or "?") == fam]
            if gone or added:
                log(f"  {fam}: {len(gone)} only in the index; {len(added)} only in the plan")
                for n in sorted(gone):
                    log(f"    - {n}")
                for n in sorted(added):
                    log(f"    + {n}")
        return 0
    if args.list or args.check:
        for s in specs:
            log(f"  {'have' if present(s) else 'todo'} {s.kind:7s} {s.name}  [{s.key[:8]}]  {s.site}")
        return check(t, specs, problems) and 1 if args.check else 0
    results = build(specs, args.jobs or default_jobs(), args.stop_after, {"sm80": "8.0", "sm90": "9.0a", "sm100": "10.0a"}[t.arch])
    failed = {k: v for k, v in results.items() if v[1] is not None}
    index = write_index(t, specs)
    left = sum(present(s) is None for s in specs)
    total = sum(v[0] for v in results.values())
    log(f"done: {len(results) - len(failed)} built, {len(failed)} failed, {left} of {len(specs)} still absent; "
        f"{total:.0f} s of builds in {time.monotonic() - started:.0f} s; index {index}")
    return 1 if failed or problems or left else 0


if __name__ == "__main__":
    sys.exit(main())
