"""Per-GPU Triton autotune-config cache — see package docstring.

ONE cache location, always: ``src/miniworld_engine/autotune/data/<op>/<gpu_key>.json``,
committed to git and shipped inside the package. Reads (dispatch/prune) and writes
(builder / RUN_AUTOTUNE regen) both target this in-repo path so a tuned cache is
versioned with the kernels and shared across every machine that checks out the repo.
There is deliberately NO ``$MINIWORLD_ENGINE_CACHE_DIR`` / ``$XDG_CACHE_HOME`` / ``~/.cache``
override: a stale per-user cache must never shadow the repo's committed configs.

Cache JSON schema (v1)::

    {"schema": 1, "gpu": "<gpu_key>", "op": "<op>",
     "config_space_hash": "<12-hex>",          # hash of the kernel's FULL config grid
     "provenance": {"triton": "...", "torch": "...", "built_utc": "..."},
     "entries": {"<dtype>|<bucket>": [{"kwargs": {...}, "num_warps": N, "num_stages": N,
                                       "ms": <median>}, ... top-K]}}

``config_space_hash`` invalidates an entry when the kernel's grid changes, so a stale cache
degrades to a warn + full-grid fallback instead of silently pinning old tiles.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import torch

from miniworld_engine._atomic import write_json

SCHEMA = 1

#: How a bucket key is CONSTRUCTED, independent of the file layout `SCHEMA` describes. Bumped when
#: the same bucket string starts meaning something else, because nothing else catches that:
#: `config_space_hash` sees the config grid, `op_identity` sees the kernel source and its
#: `key=[...]`, and neither moves when a launcher changes what it puts IN the key.
#:
#:   1 -> a `level=both` kernel keyed on its LENGTH, so a pair L=1024 (1,048,576 rows) and an atom
#:        A=1024 (1,024 rows) shared bucket `shape_key=1024`.
#:   2 -> it keys on its ROW COUNT (autotune/shape_key.py::BOTH_ROWS).
#:
#: An entry written under an older scheme is a miss, not a wrong answer: the reader falls back to
#: the bounded heuristic subset and warns, and the next build overwrites it.
KEY_SCHEME = 3

#: How ``driver_identity`` is COMPUTED. Bumped whenever `_scoped_driver_source`'s scope changes,
#: because nothing else catches that: the stored hash and the live hash would simply disagree and
#: every stamped cache would read as "build driver changed" -- a false STALE the tools cannot tell
#: from a real one, and one no rebuild-free fix can clear. A stamp whose scheme is absent or older
#: than this is SKIPPED, not failed: the guard goes quiet for that cache until it is re-stamped.
#:
#:   1 -> the op's own driver function + the driver module's shared scope (imports, module
#:        constants, `_`-private helpers). The first shipped scope; an earlier unreleased revision
#:        hashed the whole driver FILE, which flagged 50% of caches on this repo's history because
#:        one op's edit invalidated every other op in the same file.
#:   2 -> ... plus the definitions IMPORTED FROM SIBLING DRIVER MODULES, transitively
#:        (`_imported_driver_scope`). Scheme 1 was blind to them, and that blindness is not
#:        theoretical: `drivers/adaln.py` takes its shapes from `drivers/conditioned_transition.py`
#:        via `from .conditioned_transition import _D, _DC, _M, _SHAPE_KEY`, so an edit to `_M`
#:        rewrote every bucket adaln builds while the import line -- all scheme 1 hashed -- stayed
#:        byte-identical. `cache-status` reported adaln OK against a cache the edit had already
#:        filled with buckets nothing queries.
DRIVER_ID_SCHEME = 2

#: Which levels each scheme bump actually changed. Scheme 2 re-based only `level=both` kernels;
#: a token or atom kernel's bucket means exactly what it did before, so invalidating its entries
#: would discard a finished build to fix something that was never wrong. 68 of the 103 declared
#: kernels are in that position.
#: Which levels a scheme bump re-buckets. 2: `level=both` moved from length to ROW COUNT.
#: 3: `atom_key` gained the token rungs 384 and 768 (shape_key.ATOM_KEY_BUCKETS), so a length that
#: used to floor into 256 or 512 now has its own bucket -- every `level=atom` entry is against the
#: old boundaries.
_SCHEME_AFFECTS = {2: frozenset({"both"}), 3: frozenset({"atom"})}


@functools.lru_cache(maxsize=1)
def _levels() -> dict[str, str]:
    """kernel -> its `level` column, from registry.csv. Read once."""
    import csv

    reg = Path(__file__).resolve().parents[1] / "kernels" / "registry.csv"
    if not reg.is_file():
        return {}
    return {r["kernel"]: r["level"] for r in csv.DictReader(reg.open())}


def _scheme_stale(op: str, stored) -> bool:
    """Was ``op``'s bucket construction changed by a scheme bump this entry predates?"""
    stored = stored if isinstance(stored, int) else 1     # absent = written before the field
    if stored >= KEY_SCHEME:
        return False
    level = _levels().get(op)
    if level is None:
        return True          # unknown op: the strict reading
    return any(level in _SCHEME_AFFECTS.get(v, frozenset())
               for v in range(stored + 1, KEY_SCHEME + 1))
# The one and only cache root: the in-repo ``data/`` dir, committed to git. Both reads
# and writes go here — no env override, no ~/.cache — so a tuned cache is versioned with
# the kernels and a stale per-user cache can never shadow the repo's committed configs.
_CACHE_ROOT = Path(__file__).parent / "data"






def gpu_key(device_index: int | None = None) -> str:
    """Stable per-GPU key: name + compute capability, e.g. ``NVIDIA A100 80GB PCIe (sm80)``."""
    if not torch.cuda.is_available():
        return "cpu"
    if device_index is None:
        device_index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device_index)
    cc = torch.cuda.get_device_capability(device_index)
    return f"{name} (sm{cc[0]}{cc[1]})".replace("/", "_")


def shape_bucket(**dims: int) -> str:
    """Canonical shape-bucket string from the size-defining dims, e.g. ``H2=1024`` or
    ``K=512,ND=2048``. Callers pass the dims that actually change the best config (NOT the
    batched/row count M, which is handled by tile-M autotune); keep it small + stable."""
    return ",".join(f"{k}={int(v)}" for k, v in sorted(dims.items()))


# --------------------------------------------------------------------------- #
# config (de)serialization + config-space hash
# --------------------------------------------------------------------------- #
def as_cfg_dict(config) -> dict:
    """Normalize a config from ANY backend to ``{kwargs, num_warps, num_stages}``.

    - triton.Config: ``.kwargs`` / ``.num_warps`` / ``.num_stages``.
    - plain dict (cute/cuda tile params, e.g. ``{"tile_m":128,"tile_n":128,"cluster":(1,1)}``):
      the whole dict is the kwargs; num_warps/num_stages default 0 (unused off-Triton).
      A dict already shaped ``{"kwargs":..., "num_warps":..., "num_stages":...}`` passes through.
    """
    if hasattr(config, "kwargs"):  # triton.Config
        return {"kwargs": dict(config.kwargs), "num_warps": int(config.num_warps),
                "num_stages": int(config.num_stages)}
    if isinstance(config, dict) and "kwargs" in config:
        return {"kwargs": dict(config["kwargs"]),
                "num_warps": int(config.get("num_warps", 0)),
                "num_stages": int(config.get("num_stages", 0))}
    if isinstance(config, dict):
        return {"kwargs": dict(config), "num_warps": 0, "num_stages": 0}
    raise TypeError(f"unsupported config type: {type(config)!r}")


def _json_safe(kwargs: dict) -> dict:
    """cute configs may carry tuples (cluster shapes); JSON has no tuples -> lists."""
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in kwargs.items()}


def _sig(config) -> tuple:
    """Hashable signature (sorted kwargs, num_warps, num_stages) for any-backend config."""
    d = as_cfg_dict(config)
    kw = tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v)
                      for k, v in d["kwargs"].items()))
    return (kw, d["num_warps"], d["num_stages"])


def _sig_from_dict(d: dict) -> tuple:
    """Signature of a stored/JSON config dict; JSON turned tuples into lists -> re-tuple them
    so it compares equal to a live config's ``_sig``."""
    kw = tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v)
                      for k, v in d["kwargs"].items()))
    return (kw, int(d["num_warps"]), int(d["num_stages"]))






def config_to_dict(config, ms: float | None = None) -> dict:
    """Serialize any-backend config to a JSON-safe ``{kwargs, num_warps, num_stages[, ms]}``."""
    c = as_cfg_dict(config)
    d = {"kwargs": dict(sorted(_json_safe(c["kwargs"]).items())),
         "num_warps": c["num_warps"], "num_stages": c["num_stages"]}
    if ms is not None:
        d["ms"] = float(ms)
    return d


def config_space_hash(configs) -> str:
    """12-hex hash of the kernel's full config grid; changes iff the grid changes."""
    sigs = sorted(_sig(c) for c in configs)
    return hashlib.sha1(repr(sigs).encode()).hexdigest()[:12]


_env_identity_cache: str | None = None


def env_identity() -> str:
    """12-hex of everything OUTSIDE the kernel that invalidates a measurement.

    A tuned config is a claim about a compiler and a device, not just about a grid. Triton picks
    different schedules across releases and ptxas emits different code across toolkits, so a cache
    built on triton 3.3 / cuda 12.4 says nothing about triton 3.4 -- yet the grid hash is identical
    across both, and without this the stale entries would be served as if they were measured here.
    This is the ``triton_X/cuda_Y/gpu_Z`` path segment triton-dejavu keys its storage on, flattened
    into a field because our layout already keys the FILE on the GPU."""
    global _env_identity_cache
    if _env_identity_cache is not None:
        return _env_identity_cache
    parts = []
    try:
        import triton
        parts.append(f"triton={getattr(triton, '__version__', '?')}")
    except Exception:  # identity degrades to "unknown", never raises
        parts.append("triton=?")
    parts.append(f"torch={torch.__version__}")
    parts.append(f"cuda={getattr(torch.version, 'cuda', None)}")
    # ptxas is what actually turns the IR into SASS; a toolkit bump changes the code for an
    # unchanged grid, and torch.version.cuda is the BUILD-time toolkit, which can differ from the
    # ptxas triton actually invokes. Tried in order, because triton has moved this three times and
    # a probe that silently misses records "ptxas=?" for everyone -- which is exactly what the
    # first version of this did: it imported `_path_to_binary` from
    # `triton.backends.nvidia.driver`, where it does not exist in triton 3.6, so the component
    # this function exists to capture was absent from every identity it produced. Found by `ty`.
    parts.append(f"ptxas={_ptxas_version()}")
    _env_identity_cache = hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]
    return _env_identity_cache


def _ptxas_version() -> str:
    """The ptxas release string, or ``"?"`` if it genuinely cannot be found."""
    import subprocess

    def _release(path) -> str | None:
        try:
            out = subprocess.run([str(path), "--version"], capture_output=True,
                                 text=True, timeout=10, check=False).stdout
        except Exception:
            return None
        return _pick(out)

    def _pick(text: str) -> str | None:
        """The release line only. get_ptxas_version() hands back the whole banner; storing all
        five lines works but makes every identity mismatch unreadable in a warning."""
        return next((ln.strip() for ln in str(text).splitlines() if "release" in ln), None)

    try:  # triton >= 3.3: the backend exposes the version directly
        from triton.backends.nvidia.compiler import get_ptxas_version
        rel = _pick(get_ptxas_version())
        if rel:
            return rel
    except Exception:
        pass
    for getter in (
        lambda: __import__("triton.knobs", fromlist=["nvidia"]).nvidia.ptxas.path,
        lambda: __import__("triton.backends.nvidia.compiler",
                           fromlist=["get_ptxas"]).get_ptxas(),
    ):
        try:
            rel = _release(getter())
        except Exception:
            continue
        if rel:
            return rel
    return "?"




def op_identity(autotuner) -> str:
    """12-hex of the kernel SOURCE and the autotuner's key list, for one autotuner.

    Two things change what a measurement means without touching the config grid. Editing the
    kernel body makes the winning tile a claim about code that no longer exists; editing
    ``key=[...]`` re-partitions the entries, so a stored bucket answers a question the runtime is
    no longer asking. dejavu hashes both (the JIT function, line numbers excluded, and the key
    list); we had neither, so both edits degraded to silently serving the wrong config."""
    parts = []
    fn = getattr(autotuner, "fn", None)
    # An Autotuner wraps a JITFunction, which may itself wrap a Heuristics/Autotuner. Walk to the
    # JITFunction that actually owns the source.
    for _ in range(8):
        if fn is None or hasattr(fn, "src"):
            break
        fn = getattr(fn, "fn", None)
    src = getattr(fn, "src", None)
    if isinstance(src, str):
        # Line numbers are not semantics: moving a kernel down a file must not invalidate it.
        body = "\n".join(ln.rstrip() for ln in src.splitlines() if ln.strip())
        parts.append(f"src={hashlib.sha1(body.encode()).hexdigest()}")
    else:
        parts.append("src=?")
    parts.append(f"keys={list(getattr(autotuner, 'keys', ()) or ())}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


@functools.lru_cache(maxsize=512)
def _registry_driver(op: str) -> tuple[str, str] | None:
    """``(module, function)`` of ``op``'s registry driver -- the BUILD code -- e.g.
    ``("miniworld_engine.kernels.drivers.adaln", "adaln_gemm_gate")``, or None."""
    import csv as _csv

    reg = Path(__file__).resolve().parent.parent / "kernels" / "registry.csv"
    try:
        with reg.open(encoding="utf-8") as h:
            for row in _csv.DictReader(h):
                if row.get("kernel") == op:
                    drv = (row.get("driver") or "").strip()
                    if ":" in drv:
                        mod, fn = drv.split(":", 1)
                        return (mod.strip(), fn.strip())
                    return None
    except OSError:
        return None
    return None


def _scoped_driver_source(src: str, fn_name: str) -> str | None:
    """``fn_name``'s source plus the module's SHARED scope -- imports, module constants and the
    ``_``-private helpers a driver calls (``_x`` / ``_bdll`` / ``_w``) -- and nothing else.

    Granularity matters here, measured on this repo's history: hashing the whole driver FILE
    flagged 50% of caches (one op's edit invalidating every other op in the same file), while this
    scope flags 6% -- the same as hashing the function alone, but without being blind to a helper
    change, which the function alone would miss."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    shared: list[str] = []
    target: str | None = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == fn_name:
                target = ast.get_source_segment(src, node)
            elif node.name.startswith("_"):          # shared private helper
                shared.append(ast.get_source_segment(src, node) or "")
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.Import, ast.ImportFrom)):
            shared.append(ast.get_source_segment(src, node) or "")
    if target is None:
        return None
    return "\n".join(shared) + "\n" + target


#: Package the sibling driver modules live in. An import from anywhere else (torch, the kernels
#: themselves) is build-independent or already covered by ``op_identity``.
_DRIVER_PKG = "miniworld_engine.kernels.drivers"


def _module_level_defs(tree, src: str) -> dict[str, str]:
    """name -> its top-level definition's source text, for constants and functions alike."""
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = ast.get_source_segment(src, node) or ""
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out[node.target.id] = ast.get_source_segment(src, node) or ""
        elif isinstance(node, ast.Assign):
            text = ast.get_source_segment(src, node) or ""
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    out[tgt.id] = text
    return out


def _live_driver_source(mod_name: str) -> str | None:
    """This checkout's source for a driver module. The default reader; the git backfill swaps in
    one that reads the same module out of a historical tree, so a stamp computed for an old commit
    hashes THAT commit's siblings rather than today's."""
    try:
        import importlib
        import inspect

        return inspect.getsource(importlib.import_module(mod_name))
    except Exception:
        return None


def _imported_driver_scope(mod_name: str, src: str, _seen: frozenset[str] = frozenset(),
                           read_source=_live_driver_source) -> str:
    """The definitions this driver module IMPORTS FROM SIBLING DRIVER MODULES, transitively.

    The hole this closes, and it is not hypothetical -- it is how a cache-destroying driver edit
    shipped while the guard reported OK. ``drivers/adaln.py`` gets its shapes with
    ``from .conditioned_transition import _D, _DC, _M, _SHAPE_KEY``. Changing ``_M``'s definition
    changes every bucket adaln builds, but the IMPORT STATEMENT's text -- all
    :func:`_scoped_driver_source` sees of it -- is byte-identical, so adaln's stamp did not move
    and ``cache-status`` called its now-junk cache fresh.

    Scope is the imported names plus the module-level names those transitively reference (so
    ``_M = ragged(_N * _BASE)`` still moves when ``_BASE`` does), NOT the whole file: a sibling
    module holds several families' shapes, and hashing all of it would flag every importer on an
    unrelated edit -- the same over-sensitivity that made whole-file hashing unusable."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ""
    chunks: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        target = (f"{_DRIVER_PKG}.{node.module}" if node.level and node.module
                  else node.module or "")
        if not target.startswith(_DRIVER_PKG) or target == mod_name or target in _seen:
            continue
        sib_src = read_source(target)
        if sib_src is None:
            continue
        try:
            sib_tree = ast.parse(sib_src)
        except SyntaxError:
            continue
        defs = _module_level_defs(sib_tree, sib_src)
        # transitive closure over module-level names the imported definitions reference
        wanted = {a.name for a in node.names} & set(defs)
        frontier = set(wanted)
        while frontier:
            nxt: set[str] = set()
            for name in frontier:
                for sub in ast.walk(ast.parse(defs[name])):
                    if isinstance(sub, ast.Name) and sub.id in defs and sub.id not in wanted:
                        nxt.add(sub.id)
            wanted |= nxt
            frontier = nxt
        # the sibling's own import lines: `_M = ragged(...)` means nothing without `ragged`
        imports = [ast.get_source_segment(sib_src, n) or ""
                   for n in sib_tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        chunks.append(f"# from {target}\n" + "\n".join(imports + [defs[n] for n in sorted(wanted)]))
        chunks.append(_imported_driver_scope(target, sib_src, _seen | {mod_name, target},
                                             read_source))
    return "\n".join(c for c in chunks if c)


def driver_identity(op: str) -> str | None:
    """12-hex of the BUILD code that produced ``op``'s cache: its registry driver function plus the
    driver module's shared scope. ``op_identity`` fingerprints the KERNEL; this fingerprints the
    code that *drives* it in the build. The driver decides which (shape, dtype, flag) buckets get
    tuned and with what arguments, so a driver edit -- adding an ``ADD_RESIDUAL=1`` call, changing
    a swept width -- changes what the cache covers while the kernel source and the config grid stay
    byte-identical, a drift neither ``op_identity`` nor ``config_space_hash`` can see.

    Scope is deliberate (see :func:`_scoped_driver_source`): other ops' driver functions in the
    same file are excluded, their edits are not this op's business. None when the op has no
    registry driver (a dispatch-only cache) or the module cannot be read here."""
    ref = _registry_driver(op)
    if ref is None:
        return None
    mod_name, fn_name = ref
    try:
        import importlib
        import inspect

        src = inspect.getsource(importlib.import_module(mod_name))
    except Exception:
        return None
    scoped = _scoped_driver_source(src, fn_name)
    if scoped is None:
        return None
    scoped += "\n" + _imported_driver_scope(mod_name, src)
    body = "\n".join(ln.rstrip() for ln in scoped.splitlines() if ln.strip())
    return hashlib.sha1(body.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# load / store
# --------------------------------------------------------------------------- #
_load_cache: dict[tuple[str, str], dict | None] = {}


def _load(op: str, gk: str) -> dict | None:
    """Load the in-repo cache for (op, gpu). Memoized; missing/corrupt -> None."""
    key = (op, gk)
    if key in _load_cache:
        return _load_cache[key]
    result = None
    fp = _CACHE_ROOT / op / f"{gk}.json"
    if fp.exists():
        try:
            result = json.loads(fp.read_text())
        except Exception:  # corrupt cache -> treat as miss
            result = None
    _load_cache[key] = result
    return result


@functools.lru_cache(maxsize=1)
def _build_revs() -> dict[str, int]:
    """kernel -> registry.csv's ``build_rev``. Read once."""
    import csv

    reg = Path(__file__).resolve().parents[1] / "kernels" / "registry.csv"
    if not reg.is_file():
        return {}
    out: dict[str, int] = {}
    for r in csv.DictReader(reg.open()):
        try:
            out[r["kernel"]] = int((r.get("build_rev") or "1").strip() or 1)
        except ValueError:
            out[r["kernel"]] = 1
    return out


def _stored_rev(data: dict) -> int:
    """The ``build_rev`` a cache file records, defaulting to 1 when absent.

    Spelled out rather than `int(data.get("build_rev", 1) or 1)`: `or 1` is falsy-tested, so a
    stored **0** reads back as 1 and a cache built under revision 0 would compare equal to one
    built under revision 1. Revisions start at 1 by convention, which is exactly why the bug would
    have stayed invisible until someone used 0.
    """
    v = data.get("build_rev", 1)
    try:
        return int(v)
    except (TypeError, ValueError):
        return 1


def build_rev(op: str) -> int:
    """The DECLARED measurement revision for ``op`` -- the one invalidator a human writes.

    Every other fingerprint here answers "did something change?" automatically, and that turned out
    to be the wrong question twice over. `op_identity` hashes the kernel's source, so reformatting a
    comment discarded hours of measurement; `driver_identity` hashes the build driver, so correcting
    which SHAPES get built deleted 32 of 38 tuned buckets that the edit said nothing against. Both
    conflate "the code moved" with "the numbers are wrong".

    `build_rev` asks the question that actually matters and can only be answered by a person: did
    the way this kernel is MEASURED change, so that everything recorded under the old way is void?
    Bump it in registry.csv and the next build discards that kernel's cache. Leave it and edits are
    additive -- a narrowed ladder keeps its winners, a corrected driver keeps its buckets.

    Two automatic invalidators remain, because neither is a person's to notice: `env_identity` (a
    different triton/cuda/ptxas really does make a recorded time a claim about another compiler) and
    `key_scheme` (a bucket string that now means something else is mislabelled, not merely stale).
    """
    return _build_revs().get(op, 1)


def _entries_survive(data: dict, configs) -> bool:
    """Is every stored winner still a legal config in the CURRENT grid?

    The reset rule this answers. `store_ranked_configs` used to clear the file whenever
    `config_space_hash` moved, which conflates two very different edits:

    * the grid GREW -- a config nobody benched now exists, so every stored winner is a claim about
      a smaller space than the one being searched. Genuinely stale.
    * the grid was NARROWED and every stored winner survived the cut -- the winner beat everything
      in the OLD space, and the new space is a subset of it, so the same config is still the best
      of the new space. Strictly MORE valid than before, and deleting it is pure loss.

    Narrowing a ladder to what the cache proves it needs is the repository's own recommended
    maintenance (`test_a_fully_measured_kernel_carries_its_own_ladder` derives such a narrowing and
    demands it). Charging a full rebuild for it made that maintenance cost 25 GPU-hours of
    re-measuring configs whose winners were already known -- so the tests asked for a change the
    cache then punished.

    Membership of every winner is the exact test: it holds iff no winner was cut, which for a
    subset edit is the whole question. If the grid grew, a winner can still be a member while the
    new values were never benched -- so the caller only reaches here when the space did not grow.
    """
    if not configs:
        return False            # cannot tell -> reset, the safe direction
    live = {_sig(c) for c in configs}
    for ranked in (data.get("entries") or {}).values():
        for cfg in ranked:
            if _sig_from_dict(cfg) not in live:
                return False
    return True


def configs_to_bench(op: str, gk: str, configs) -> list:
    """The configs a build still has to MEASURE -- the current grid minus the space already searched.

    The other half of the incremental policy. `store_ranked_configs` keeps a cache when only the
    grid moved; this is what makes that cheap instead of merely non-destructive. Each cache records
    `config_space`, the set of configs the build that wrote it actually searched, so a later build
    can bench the difference and nothing else:

      * grid WIDENED  -> the added configs were never timed. Bench exactly those and let the write
        merge them against the stored winners, which are still valid for everything they beat.
      * grid NARROWED -> nothing new to time. Returns empty, and the write drops any stored config
        the new grid no longer contains.

    Returns the FULL grid whenever the difference cannot be trusted: no cache, no recorded space (a
    file written before this field), or a fingerprint that invalidates the measurements outright
    (`build_rev`, `env_identity`, `key_scheme`) -- in which case the caller is about to reset the
    file anyway and every config needs timing again.

    The saving this exists for, measured on this repository: collapsing GROUP_M to its single
    winning value across 16 kernels is a narrowing the cache can already price, and under the old
    "any grid change resets" rule it cost a 25 GPU-hour rebuild to apply a change that removed
    nothing any winner needed.
    """
    configs = list(configs)
    data = _load(op, gk)
    if data is None:
        return configs
    if (_stored_rev(data) != build_rev(op)
            or _scheme_stale(op, data.get("key_scheme"))
            or data.get("env_identity") != env_identity()):
        return configs                      # about to be reset: everything needs re-measuring
    searched = data.get("config_space")
    if not isinstance(searched, list):
        return configs                      # predates the field: cannot tell, so measure it all
    seen = set(searched)
    todo = [c for c in configs if repr(_sig(c)) not in seen]
    return todo


def store_ranked_configs(
    op: str, gk: str, dtype: str, bucket: str, ranked: list[tuple[object, float]],
    config_space_h: str, *, top_k: int = 5, op_id: str = "", env_id: str = "",
    configs=None,
) -> Path:
    """Persist the top-K (config, ms) for (op, gpu, dtype, bucket) to the in-repo cache.

    ``ranked`` is a list of ``(triton.Config, median_ms)`` sorted fastest-first. Resets the
    file's entries if the config-space hash changed (kernel grid was edited). Writes into the
    committed ``data/`` tree so the builder's output is ready to ``git add`` and share."""
    fp = _CACHE_ROOT / op / f"{gk}.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] | None = None
    if fp.exists():
        try:
            data = json.loads(fp.read_text())
        except Exception:
            data = None
    env_id = env_id or env_identity()
    driver_id = driver_identity(op)
    # WHICH fingerprint changes make a stored WINNER wrong -- which is a narrower question than
    # "did anything about the build change", and conflating the two has cost this repository real
    # measurements twice.
    #
    #   env_identity / op_identity / key_scheme -> yes. A different compiler, a different kernel,
    #       or a bucket that means something else all make a recorded time a claim about something
    #       that no longer exists.
    #   config_space_hash -> ONLY when the space grew, or when a winner was cut out of it. A pure
    #       narrowing that keeps every winner leaves each entry the best of a SMALLER space.
    #   driver_identity -> NEVER. The driver decides which buckets get built, not whether a
    #       measured winner is right. Resetting on it deleted 32 of 38 tuned buckets from
    #       cond_transition_expand_swiglu the moment its driver was corrected -- data that had
    #       cost hours and that the edit said nothing against. It stays a recorded stamp, so
    #       `dev cache-status` still reports the coverage drift; it just no longer destroys.
    rev = build_rev(op)
    reset = True
    if (data is None
            or _stored_rev(data) != rev                       # declared: the METHOD changed
            or _scheme_stale(op, data.get("key_scheme"))      # automatic: the key means something else
            or data.get("env_identity") != env_id             # automatic: another compiler
            or (op_id and data.get("op_identity") not in (None, op_id))):  # automatic: another kernel
        # `op_identity` IS here, and demoting it was a mistake worth naming: with no reset and
        # no re-stamp the reader refuses the entries forever, including the ones a rebuild has
        # just measured; with no reset and a re-stamp the pre-edit winner is served as if it had
        # been measured against the current kernel. Neither is acceptable, and the field it was
        # supposed to be replaced by -- `build_rev` -- solves a different problem: it lets a person
        # invalidate on purpose, not avoid invalidating when the kernel really did change.
        #
        # A GRID change is deliberately NOT here. Widening it leaves the old winners valid for the
        # configs they beat and needs only the added configs benched; narrowing it leaves each
        # surviving winner the best of a smaller space. Either way the answer is an incremental
        # build against `config_space`, not a reset -- see `configs_to_bench`.
        import datetime as _dt  # stamp only when writing
        try:
            import triton as _triton
            triton_ver = getattr(_triton, "__version__", "?")
        except Exception:
            triton_ver = "?"
        # Annotated because the literal's value types join to `int | str | dict`, which makes
        # `data["entries"][key] = ...` below a subscript-assign on an int as far as a checker
        # can tell. The shape is genuinely heterogeneous; say so once.
        data = {
            "schema": SCHEMA, "key_scheme": KEY_SCHEME,
            "gpu": gk, "op": op, "config_space_hash": config_space_h,
            "build_rev": rev,
            "env_identity": env_id, "op_identity": op_id, "driver_identity": driver_id,
            "driver_id_scheme": DRIVER_ID_SCHEME if driver_id else None,
            "provenance": {"triton": triton_ver, "torch": torch.__version__,
                           "built_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")},
            "entries": {},
        }
    else:
        reset = False
    if op_id:
        data["op_identity"] = op_id
    if driver_id:
        data["driver_identity"] = driver_id
        data["driver_id_scheme"] = DRIVER_ID_SCHEME
    # Stamp it on every write, not only when the file is reset. A token/atom file that predates
    # the field is valid under scheme 2 -- the bump re-based both-level keys only -- but "valid
    # because the field is missing" is a fact you have to know `_SCHEME_AFFECTS` to reconstruct.
    # Writing it down means a file says which scheme its keys are in.
    data["key_scheme"] = KEY_SCHEME
    data["build_rev"] = rev
    data["config_space_hash"] = config_space_h
    if configs is not None:
        # The SPACE itself, not just its hash. A hash says "different"; the incremental build needs
        # to know HOW -- which configs were added (bench those) and which were removed (drop any
        # entry whose winner went with them).
        data["config_space"] = sorted(repr(x) for x in {_sig(c) for c in configs})
    # MERGE, not overwrite. Under the incremental policy a build may have benched only the configs
    # the grid ADDED, so `ranked` can be a strict subset of the search -- overwriting would throw
    # away the previous winner, which is usually the one that is still best. New measurements win
    # on a tie because they were taken now; anything the current grid no longer contains is
    # dropped, since it is no longer a config this kernel can be launched with.
    key = f"{dtype}|{bucket}"
    live = {_sig(c) for c in configs} if configs else None
    merged: dict[tuple, dict] = {}
    if not reset:
        for cfg in data["entries"].get(key, []):
            sig = _sig_from_dict(cfg)
            if live is None or sig in live:
                merged[sig] = cfg
    for c, ms in ranked:
        merged[_sig(c)] = config_to_dict(c, ms)
    ordered = sorted(merged.values(), key=lambda d: (d.get("ms") is None, d.get("ms", float("inf"))))
    data["entries"][key] = ordered[:top_k]
    write_json(fp, data, indent=2, sort_keys=True)
    _load_cache.pop((op, gk), None)  # invalidate memo
    return fp


# --------------------------------------------------------------------------- #
# runtime prune hook
# --------------------------------------------------------------------------- #
_warned: set[tuple] = set()


#: Every (op, gpu, dtype|bucket) this process failed to find in the cache. A build can only report
#: what it captured; this records what a RUN actually asked for and did not get, which is the only
#: direct measure of whether the cache covers a workload.
#:
#: Read by ``miniworld-engine dev audit --replay``. That flag is new: the comment here named
#: ``miniworld-engine audit``, which is the STATIC check in ``build/audit.py`` and has never
#: touched this set -- and ``builder.audit``, the replay it meant, had no caller anywhere in the
#: repo. The one measurement the coverage claim rests on was unreachable from every command.
_CACHE_MISSES: set = set()


def cache_misses() -> frozenset:
    """(op, gpu, key) triples this process looked up and did not find."""
    return frozenset(_CACHE_MISSES)


def clear_cache_misses() -> None:
    """Forget what this process has missed so far.

    The set only ever grew, so a caller that measured, improved the cache, and measured again got
    the first answer back both times -- it cannot report an improvement, only more misses. A
    before/after over one process is the natural way to check a cache fill and it was silently
    unable to fail in the useful direction."""
    _CACHE_MISSES.clear()


def _warn_once(op: str, gk: str, tag: str, reason: str, fallback: str = "") -> None:
    _CACHE_MISSES.add((op, gk, tag))
    key = (op, gk, tag, reason)
    if key in _warned:
        return
    _warned.add(key)
    # Say which fallback actually happened. "Falling back to the full autotune grid" was printed
    # even where the grid is 205,266 configs, and it is no longer true on the Triton path: a miss
    # now searches a bounded subset. A warning that misdescribes the recovery is worse than none.
    what = fallback or "the full autotune grid"
    warnings.warn(
        f"[miniworld.autotune] {reason} for op '{op}' on '{gk}' ({tag}). Falling back to "
        f"{what} — this run may be slower and the chosen config may be suboptimal. "
        f"Build a tuned cache for this GPU with the autotune cache-builder "
        f"(see docs/operations/dispatch-cache.md).",
        stacklevel=4,
    )


def _named_get(named_args, kwargs, name):
    if hasattr(named_args, "get") and name in named_args:
        return named_args[name]
    return kwargs.get(name)


def key_bucket_of(*key_names: str):
    """Build a ``bucket_of`` from the kernel's autotune ``key`` names (constexpr dims reliably
    present in named_args), e.g. ``key_bucket_of("GROUP_M", "n", "N")``."""
    def f(named_args, kwargs):
        return shape_bucket(**{k: _named_get(named_args, kwargs, k) for k in key_names})
    return f


def operand_bytes(named_args, kwargs, arg_name: str, default: int = 2) -> int:
    """Bytes per element of a kernel operand, for shared-memory estimates.

    Every hand-written smem estimator in this repo multiplied its element COUNT by a literal 2
    with a "bf16 = 2 B" comment. That is right up to the moment a kernel runs in fp32, where it
    reports exactly half the real footprint -- so an over-limit config survives the prune and the
    launch dies with OutOfResources instead of the tuner quietly rejecting it. It is the wrong
    direction to be wrong in: the prune exists precisely so an unlaunchable config never reaches
    a launch. Read the operand instead of assuming it.

    ``default`` is the bf16 2 B the estimators used to hardcode, kept for the case where the
    operand is not introspectable -- an unknown estimate must not become a crash.
    """
    t = named_args.get(arg_name) if hasattr(named_args, "get") else None
    if t is None:
        t = kwargs.get(arg_name)
    size = getattr(t, "element_size", None)
    try:
        return int(size()) if callable(size) else default
    except Exception:  # never let an estimate break a launch
        return default


def tensor_dtype_of(arg_name: str, default: str = "bfloat16"):
    """Build a ``dtype_of`` that reads the dtype of a tensor kernel-arg (falls back to
    ``default`` — production bf16 — if it isn't introspectable in named_args)."""
    def f(named_args, kwargs):
        t = named_args.get(arg_name) if hasattr(named_args, "get") else None
        return str(getattr(t, "dtype", default)).replace("torch.", "")
    return f


_smem_limit_cache: dict[int, int] = {}




# A COMPILE MONSTER is a config triton spends 10-20 MIN compiling (make_llir + ptxas), always
# register-spill bound so it never wins. The guard is a per-config COMPILE TIMEOUT (fork +
# SIGKILL) in ``autotune/capture.py``, which judges each config by how long it actually takes to
# compile on THIS GPU. There is deliberately NO static ``num_warps``/``num_stages`` pre-filter:
# one used to drop the ``num_warps>=16`` / ``num_stages>=5`` tail as "monsters on every arch", but
# neither claim survived measurement. ``num_stages`` has no compiler maximum and its cost is
# exactly one operand tile of shared memory per stage, so the usable ceiling is
# ``smem_limit / operand_tile`` -- it differs per config and is far above 5 for small tiles. A
# static rule also silently shrinks whatever grid a tuning run declares, which makes the run a lie
# about the space it searched. Judging by real compile time costs one timeout per bad config and
# is arch-agnostic by construction.

#: Every op that wires itself to the cache, filled as the kernel modules import. This is the only
#: complete list of what a build OUGHT to produce -- chasing missing ops one at a time (a dispatch
#: switch here, a dropout flag there) finds them one incident at a time, and each one is silent
#: until production hits the uncached path. Comparing this set against a built cache turns every
#: such hole into a single loud check; see ``miniworld-engine coverage``.
_REGISTERED_OPS: set[str] = set()






def dtype_of_args(nargs) -> str:
    """Every distinct floating dtype among the tensor operands, sorted and ``+``-joined.

    Not "the dtype of the first tensor operand", which is what this used to be. That reads
    whichever argument the kernel happens to declare first, and a kernel that upcasts one operand
    unconditionally then reports the same dtype no matter what it was given:
    ``layernorm_linear/triton/pair_bias.py`` upcasts ``dout`` to fp32 at every launch, so a bf16
    run and an fp32 run of ``layernorm_linear_bwd_fp32_triton`` recorded the byte-identical key
    ``float32|...`` while their measured errors differed by four orders of magnitude
    (2.8e-03 against 5.9e-07). Two activation regimes, one bucket.

    The set cannot collide that way: the bf16 run of that kernel is ``bfloat16+float32`` and the
    fp32 run is ``float32``. It also needs no rule about which operand "counts", which is the part
    that cannot be got right in general -- an fp32 accumulator, an fp32 stats buffer and an fp32
    activation are indistinguishable by position or by size.

    Integer operands are ignored: index and count tensors do not vary with the compute dtype.
    """
    seen = set()
    for v in (nargs or {}).values():
        dt = getattr(v, "dtype", None)
        if dt is None or not hasattr(v, "shape"):
            continue
        if getattr(dt, "is_floating_point", False):
            seen.add(str(dt).replace("torch.", ""))
    return "+".join(sorted(seen)) if seen else "any"


def bucket_of_autotuner(autotuner, nargs, meta=None) -> str:
    """Shape bucket from the autotuner's OWN ``key=[...]``, with no per-kernel wiring.

    This used to come from a ``make_cache_prune`` object carrying a hand-written
    ``key_bucket_of("N", "K", "DT")`` per kernel. Those were removed in fcd3c7a and nothing
    replaced them, so every capture since has recorded the single bucket ``any|any`` -- one config
    per op for every shape. That silently discards the whole point of the shape key: ``shape_key``
    is in every kernel's ``key`` list, so Triton re-tunes per shape bucket in-process, and then the
    cache threw the distinction away on the way to disk.

    ``autotuner.keys`` is the same list the kernel declared, so deriving the bucket from it covers
    every op automatically and cannot drift from what Triton actually re-tunes on -- which the
    per-kernel version could, and did.
    """
    keys = getattr(autotuner, "keys", None) or []
    dims = {}
    for k in keys:
        v = _named_get(nargs, meta, k)
        if isinstance(v, bool):        # flags partition the space too, but int() would alias them
            dims[k] = int(v)
        elif isinstance(v, int):
            dims[k] = v
    return shape_bucket(**dims) if dims else "any"


# --------------------------------------------------------------------------- #
# runtime cache READER (triton: narrow the grid to the cached top-K)
# --------------------------------------------------------------------------- #
_reader_installed = False


def heuristic_subset(configs: list, cap: int = 24) -> list:
    """A small, sane slice of ``configs`` for when there is no cached answer.

    The alternative is the full grid, and the full grid is what makes an untuned GPU unusable: the
    shipped set is 205,266 configs, so the first production forward on a card nobody has built for
    runs a tuning sweep inside itself. Nothing in the ecosystem does that -- triton-dejavu takes a
    user heuristic on miss (TRITON_DEJAVU_FORCE_FALLBACK), Liger-Kernel skips autotuning and picks
    num_warps from the row width, vLLM ships a default config and warns. A miss should cost a
    bounded search, not an unbounded one.

    The slice is the industry-standard centre of the space -- num_warps in {4, 8} and num_stages
    in {2, 3, 4}, which is vLLM's whole search space for fused_moe -- crossed with the middle
    value of every block axis. Measured over this repo's own 374-bucket sweep, warps=4/stages=2
    alone lands within 5% of the full-grid winner in 83% of buckets; widening it slightly and
    letting Triton pick among the survivors is meant to recover most of the rest without a sweep.
    It will not beat a built cache, and it does not try to: `build all` is still the answer, and
    the caller still warns that the cache is missing.
    """
    if not configs or cap <= 0 or len(configs) <= cap:
        return list(configs)          # already a cheap search; sorting it buys nothing
    axes: dict = {}
    for c in configs:
        for k, v in getattr(c, "kwargs", {}).items():
            if isinstance(v, int):
                axes.setdefault(k, set()).add(v)

    def score(c) -> tuple:
        kw = getattr(c, "kwargs", {})
        # distance from the middle of each block axis, then the standard warps/stages first
        off = sum(abs(sorted(axes[k]).index(v) - len(axes[k]) // 2)
                  for k, v in kw.items() if k in axes and isinstance(v, int))
        return (0 if c.num_warps in (4, 8) else 1,
                0 if c.num_stages in (2, 3, 4) else 1,
                off,
                abs(c.num_warps - 4), abs(c.num_stages - 2))

    # The ranking above puts every axis at its middle AT ONCE, which is the largest tile in the
    # "reasonable" region -- the offsets compound. When that corner does not fit in shared memory
    # nothing in the subset does, and the launch dies with OutOfResources instead of being slow:
    # measured on an A5000, adaln_bwd_dw_triton's stale-cache fallback asked for 294,912 B against
    # a 101,376 B limit, and all 24 candidates were over. So reserve a quarter of the cap for the
    # SMALLEST tiles. They are rarely the winner -- that is what the other three quarters are for --
    # but they are what makes the subset launchable at all, and a slow kernel beats a dead one.
    ranked = sorted(configs, key=score)
    floor = cap // 4
    if floor:
        def volume(c) -> int:
            v = 1
            for k, x in getattr(c, "kwargs", {}).items():
                if k in axes and isinstance(x, int) and x > 0:
                    v *= x
            return v * max(1, c.num_stages)
        # Still inside the industry centre: warps in {4,8}, stages in {2,3,4}. The floor is about
        # TILE SIZE, not about widening the warps/stages search, and
        # test_the_fallback_prefers_the_industry_centre_of_the_space pins that.
        centre = [c for c in configs if c.num_warps in (4, 8) and c.num_stages in (2, 3, 4)]
        smallest = sorted(centre or configs, key=volume)[:floor]
        keep, seen = [], set()
        for c in smallest + ranked:
            if id(c) in seen:
                continue
            seen.add(id(c))
            keep.append(c)
            if len(keep) >= cap:
                break
        return keep or list(configs)
    return ranked[:cap] or list(configs)


def _miss(op, gk, what, why, configs):
    """One exit for every cache miss: warn once, then hand back a BOUNDED search.

    Returning None here meant "keep the full grid", and the full grid is 205,266 configs. That is
    correct for a build and ruinous for a forward, which is the same call site."""
    from miniworld_engine import settings  # avoid an import cycle

    cur = settings.current()
    cap = 0 if getattr(cur, "run_autotune", False) else getattr(cur, "autotune_miss_cap", 0)
    if cap <= 0 or len(configs) <= cap:
        # A build wants the whole space on purpose, and a grid already smaller than the cap is
        # already a cheap search.
        _warn_once(op, gk, what, why, f"the full grid ({len(configs)} configs)")
        return None
    _warn_once(op, gk, what, why,
               f"a heuristic {cap} of {len(configs)} configs (run `miniworld-engine build all` "
               f"to tune this GPU properly)")
    return heuristic_subset(configs, cap)


def _cached_subset(autotuner, configs, nargs, meta):
    """The cached top-K for this (op, gpu, dtype, bucket), or a bounded fallback on a miss."""
    from miniworld_engine.autotune.configs import op_of  # avoid an import cycle

    op = op_of(getattr(autotuner, "configs", None) or [])
    if op is None:
        return None
    gk = gpu_key()
    dtype = dtype_of_args(nargs)
    bucket = bucket_of_autotuner(autotuner, nargs, meta)
    data = _load(op, gk)
    if data is None:
        return _miss(op, gk, dtype, "no tuned autotune cache", configs)
    if data.get("config_space_hash") != config_space_hash(configs):
        return _miss(op, gk, dtype,
                     "tuned autotune cache is STALE (kernel config grid changed)", configs)
    if _scheme_stale(op, data.get("key_scheme")):
        return _miss(op, gk, dtype,
                     "tuned autotune cache is STALE (bucket keys mean something else now; see "
                     "cache.KEY_SCHEME)", configs)
    if data.get("env_identity") != env_identity():
        return _miss(op, gk, dtype,
                     "tuned autotune cache is STALE (triton/cuda/ptxas changed since it was built)",
                     configs)
    stored_op_id = data.get("op_identity")
    if stored_op_id and stored_op_id != op_identity(autotuner):
        return _miss(op, gk, dtype,
                     "tuned autotune cache is STALE (kernel source or autotune key list changed)",
                     configs)
    # NOTE: driver_identity is deliberately NOT checked here. It is a BUILD-time fingerprint --
    # `dev cache-status` and the CI gate catch driver drift before anything ships -- and checking it
    # per launch would (a) import build-harness driver modules inside a production `prune_configs`,
    # dragging their import-time env-var surface into a consumer process, and (b) fail CLOSED: the
    # helper returns None on any import error, and `stored and stored != None` would then declare
    # every stamped cache stale and silently drop every launch to the heuristic subset.
    entry = data.get("entries", {}).get(f"{dtype}|{bucket}")
    if not entry:
        return _miss(op, gk, f"{dtype}|{bucket}",
                     "no tuned autotune cache entry for this shape", configs)
    # Intersect rather than trust: the cache names configs, the grid decides what is launchable.
    want = {_sig_from_dict(c) for c in entry}
    keep = [c for c in configs if _sig(c) in want]
    if not keep:
        # Every tuned config for this shape is outside the current grid -- a narrowing that cut
        # them all. The caller falls back to the FULL grid, which this module's own docstring calls
        # ruinous inside a forward, so it must not happen quietly. Unreachable before the grid
        # stopped resetting the file; reachable now, and this is the only thing that says so.
        return _miss(op, gk, f"{dtype}|{bucket}",
                     "every tuned config for this shape was removed from the config grid", configs)
    return keep


def install_cache_reader() -> None:
    """Narrow every Triton autotuner's grid to the cached top-K for the shape it is running.

    Without this the shipped cache is INERT for Triton. ``select_config`` is the only other reader
    and only the CuTe/CUDA paths call it, so since fcd3c7a -- which deleted the per-kernel
    ``make_cache_prune`` objects that used to do this and put nothing in their place -- every
    process has re-benched the full grid in-process and the committed ``data/*.json`` has been
    written but never read back. It went unnoticed because every shipped config set holds ONE
    config per op, which makes a full sweep free. It stops being free the moment a real search
    grid is installed: ``configs/grid`` is 205,266 configs, and unread means each process would
    bench all of them, per shape bucket, inside a production forward.

    Installed by patching ``Autotuner.__init__`` rather than by wiring each kernel, for the same
    reason the write side derives its bucket from ``autotuner.keys``: the per-kernel version could
    drift from what Triton actually re-tunes on, and did. Read and write now call ONE pair of
    functions -- ``dtype_of_args`` / ``bucket_of_autotuner`` -- so a cache entry cannot be written
    under one key and looked up under another.

    A miss, a stale hash, or an unknown shape warns once and keeps the full grid: config choice is
    performance-only, so the worst case is slow, never wrong.
    """
    global _reader_installed
    if _reader_installed:
        return
    from triton.runtime.autotuner import Autotuner

    orig_init = Autotuner.__init__

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        base = self.early_config_prune

        def prune(configs, nargs, **meta):
            cfgs = list(base(configs, nargs, **meta)) if base else list(configs)
            from miniworld_engine import settings
            cur = settings.current()
            if not cfgs or (cur.run_autotune and not cur.fill_gaps):
                return cfgs                          # a BUILD re-benches the whole grid on purpose
            try:
                hit = _cached_subset(self, cfgs, nargs, meta)
            except Exception:  # never break a launch
                return cfgs
            return hit or cfgs

        self.early_config_prune = prune

    Autotuner.__init__ = __init__
    _reader_installed = True


# --------------------------------------------------------------------------- #
# backend-agnostic config selection (cute / cuda: pick ONE, no autotune loop)
# --------------------------------------------------------------------------- #
def select_config(
    op: str, *, dtype: str, bucket: str, candidates=None, device_index: int | None = None,
) -> dict | None:
    """Return the cached **best** config (``{kwargs, num_warps, num_stages}``) for the running
    ``(gpu, dtype, shape-bucket)``, or ``None`` (warn-once) on a miss/stale cache.

    This is the cute/cuda counterpart of :func:`configs_for`: those backends fix their
    tile/cluster/stage config at build time and have no Triton autotune loop, so they call this
    to *pick one* config from the shipped cache instead of narrowing a grid. Callers apply the
    returned ``kwargs`` (e.g. ``tile_m``/``tile_n``/``cluster``) and, on ``None``, fall back to
    the kernel's own ``default_config``. ``candidates`` (optional) enables the same
    ``config_space_hash`` staleness check as the Triton path.

    INVARIANT (as everywhere in this module): every candidate computes the same math, so a
    miss/stale/None result only costs speed, never correctness.
    """
    try:
        gk = gpu_key(device_index)
    except Exception:
        return None
    data = _load(op, gk)
    # NOT _miss(): this path returns ONE config dict, and None means "use the kernel's own
    # default_config" -- already a bounded answer. The heuristic subset is for the Triton path,
    # where None means "sweep the whole grid".
    if data is None:
        _warn_once(op, gk, dtype, "no tuned autotune cache")
        return None
    if candidates is not None and data.get("config_space_hash") != config_space_hash(candidates):
        _warn_once(op, gk, dtype, "tuned autotune cache is STALE (kernel config grid changed)")
        return None
    if _scheme_stale(op, data.get("key_scheme")):
        _warn_once(op, gk, dtype, "tuned autotune cache is STALE (bucket keys mean something "
                                  "else now; see cache.KEY_SCHEME)")
        return None
    entry = data.get("entries", {}).get(f"{dtype}|{bucket}")
    if not entry:
        _warn_once(op, gk, f"{dtype}|{bucket}", "no tuned autotune cache entry for this shape")
        return None
    # Intersect with the live candidate space, exactly as the triton reader does. This used to be
    # implied by the reset -- a changed grid emptied the file, so a stored config could not outlive
    # the space it came from. Now that a grid edit is non-destructive, a narrowed ladder can leave
    # entries naming configs this kernel can no longer be launched with, and returning the fastest
    # stored one would hand the launcher a config that is not on its list.
    # `candidates` arrives as cache dicts (`cute_config._as_cache_dicts`), so one shape only.
    live = {_sig_from_dict(c) for c in (candidates or [])}
    for cfg in entry:
        if not live or _sig_from_dict(cfg) in live:
            best = dict(cfg)
            best.pop("ms", None)
            best["kwargs"] = {k: (tuple(v) if isinstance(v, list) else v)
                              for k, v in best["kwargs"].items()}
            return best
    _warn_once(op, gk, f"{dtype}|{bucket}",
               "every tuned config for this shape is outside the current candidate space")
    return None
