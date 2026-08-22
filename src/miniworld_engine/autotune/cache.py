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

import hashlib
import json
import warnings
from pathlib import Path

import torch

SCHEMA = 1
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
        return {"kwargs": {k: v for k, v in config.items()}, "num_warps": 0, "num_stages": 0}
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
        import triton  # noqa: PLC0415
        parts.append(f"triton={getattr(triton, '__version__', '?')}")
    except Exception:  # noqa: BLE001 -- identity degrades to "unknown", never raises
        parts.append("triton=?")
    parts.append(f"torch={torch.__version__}")
    parts.append(f"cuda={getattr(torch.version, 'cuda', None)}")
    try:
        # ptxas is what actually turns the IR into SASS; a toolkit bump changes the code for an
        # unchanged grid. torch.version.cuda is the build-time toolkit and can differ from it.
        import subprocess  # noqa: PLC0415

        from triton.backends.nvidia.driver import _path_to_binary  # noqa: PLC0415
        ptxas, _ = _path_to_binary("ptxas")
        out = subprocess.run([ptxas, "--version"], capture_output=True, text=True,  # noqa: S603
                             timeout=10, check=False).stdout
        rel = [ln for ln in out.splitlines() if "release" in ln]
        parts.append(f"ptxas={rel[0].strip() if rel else '?'}")
    except Exception:  # noqa: BLE001
        parts.append("ptxas=?")
    _env_identity_cache = hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]
    return _env_identity_cache


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
        except Exception:  # noqa: BLE001 -- corrupt cache -> treat as miss
            result = None
    _load_cache[key] = result
    return result


def store_ranked_configs(
    op: str, gk: str, dtype: str, bucket: str, ranked: list[tuple[object, float]],
    config_space_h: str, *, top_k: int = 5, op_id: str = "", env_id: str = "",
) -> Path:
    """Persist the top-K (config, ms) for (op, gpu, dtype, bucket) to the in-repo cache.

    ``ranked`` is a list of ``(triton.Config, median_ms)`` sorted fastest-first. Resets the
    file's entries if the config-space hash changed (kernel grid was edited). Writes into the
    committed ``data/`` tree so the builder's output is ready to ``git add`` and share."""
    fp = _CACHE_ROOT / op / f"{gk}.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    data = None
    if fp.exists():
        try:
            data = json.loads(fp.read_text())
        except Exception:  # noqa: BLE001
            data = None
    env_id = env_id or env_identity()
    if (data is None or data.get("config_space_hash") != config_space_h
            or data.get("env_identity") != env_id
            or (op_id and data.get("op_identity") not in (None, op_id))):
        import datetime as _dt  # noqa: PLC0415 -- stamp only when writing
        try:
            import triton as _triton  # noqa: PLC0415
            triton_ver = getattr(_triton, "__version__", "?")
        except Exception:  # noqa: BLE001
            triton_ver = "?"
        data = {
            "schema": SCHEMA, "gpu": gk, "op": op, "config_space_hash": config_space_h,
            "env_identity": env_id, "op_identity": op_id,
            "provenance": {"triton": triton_ver, "torch": torch.__version__,
                           "built_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")},
            "entries": {},
        }
    if op_id:
        data["op_identity"] = op_id
    data["entries"][f"{dtype}|{bucket}"] = [config_to_dict(c, ms) for c, ms in ranked[:top_k]]
    tmp = fp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(fp)
    _load_cache.pop((op, gk), None)  # invalidate memo
    return fp


# --------------------------------------------------------------------------- #
# runtime prune hook
# --------------------------------------------------------------------------- #
_warned: set[tuple] = set()


#: Every (op, gpu, dtype|bucket) this process failed to find in the cache. A build can only report
#: what it captured; this records what a RUN actually asked for and did not get, which is the only
#: direct measure of whether the cache covers a workload -- ``miniworld-engine audit`` replays the
#: build matrix with capture off and requires this to come back empty.
_CACHE_MISSES: set = set()


def cache_misses() -> frozenset:
    """(op, gpu, key) triples this process looked up and did not find."""
    return frozenset(_CACHE_MISSES)


def _warn_once(op: str, gk: str, tag: str, reason: str) -> None:
    _CACHE_MISSES.add((op, gk, tag))
    key = (op, gk, tag, reason)
    if key in _warned:
        return
    _warned.add(key)
    warnings.warn(
        f"[miniworld.autotune] {reason} for op '{op}' on '{gk}' ({tag}). Falling back to the "
        f"full autotune grid — this run may be slower and the chosen config may be suboptimal. "
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
    except Exception:  # noqa: BLE001 -- never let an estimate break a launch
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


def _cached_subset(autotuner, configs, nargs, meta):
    """The cached top-K for this (op, gpu, dtype, bucket), or None to keep the full grid."""
    from .configs import op_of                       # noqa: PLC0415 -- avoid an import cycle

    op = op_of(getattr(autotuner, "configs", None) or [])
    if op is None:
        return None
    gk = gpu_key()
    dtype = dtype_of_args(nargs)
    bucket = bucket_of_autotuner(autotuner, nargs, meta)
    data = _load(op, gk)
    if data is None:
        _warn_once(op, gk, dtype, "no tuned autotune cache")
        return None
    if data.get("config_space_hash") != config_space_hash(configs):
        _warn_once(op, gk, dtype, "tuned autotune cache is STALE (kernel config grid changed)")
        return None
    if data.get("env_identity") != env_identity():
        _warn_once(op, gk, dtype,
                   "tuned autotune cache is STALE (triton/cuda/ptxas changed since it was built)")
        return None
    stored_op_id = data.get("op_identity")
    if stored_op_id and stored_op_id != op_identity(autotuner):
        _warn_once(op, gk, dtype,
                   "tuned autotune cache is STALE (kernel source or autotune key list changed)")
        return None
    entry = data.get("entries", {}).get(f"{dtype}|{bucket}")
    if not entry:
        _warn_once(op, gk, f"{dtype}|{bucket}", "no tuned autotune cache entry for this shape")
        return None
    # Intersect rather than trust: the cache names configs, the grid decides what is launchable.
    want = {_sig_from_dict(c) for c in entry}
    keep = [c for c in configs if _sig(c) in want]
    return keep or None


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
    from triton.runtime.autotuner import Autotuner   # noqa: PLC0415

    orig_init = Autotuner.__init__

    def __init__(self, *a, **kw):                    # noqa: ANN001, ANN002, ANN003, N807
        orig_init(self, *a, **kw)
        base = self.early_config_prune

        def prune(configs, nargs, **meta):           # noqa: ANN001, ANN003
            cfgs = list(base(configs, nargs, **meta)) if base else list(configs)
            from .. import settings                  # noqa: PLC0415
            if settings.current().run_autotune or not cfgs:
                return cfgs                          # a BUILD re-benches the whole grid on purpose
            try:
                hit = _cached_subset(self, cfgs, nargs, meta)
            except Exception:                        # noqa: BLE001 -- never break a launch
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
    except Exception:  # noqa: BLE001
        return None
    data = _load(op, gk)
    if data is None:
        _warn_once(op, gk, dtype, "no tuned autotune cache")
        return None
    if candidates is not None and data.get("config_space_hash") != config_space_hash(candidates):
        _warn_once(op, gk, dtype, "tuned autotune cache is STALE (kernel config grid changed)")
        return None
    entry = data.get("entries", {}).get(f"{dtype}|{bucket}")
    if not entry:
        _warn_once(op, gk, f"{dtype}|{bucket}", "no tuned autotune cache entry for this shape")
        return None
    best = dict(entry[0])  # fastest-first; drop the stored ms
    best.pop("ms", None)
    best["kwargs"] = {k: (tuple(v) if isinstance(v, list) else v) for k, v in best["kwargs"].items()}
    return best
