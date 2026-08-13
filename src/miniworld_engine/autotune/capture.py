"""Generic autotune-cache BUILDER via Triton-autotuner instrumentation.

Instead of hand-replicating each kernel's launch in a per-kernel builder (fragile, and how the
int32-offset bug slipped in), this patches ``triton.runtime.autotuner.Autotuner._bench`` to
record every (config -> measured ms) as it is benchmarked during a REAL module forward/backward
run. Each timing is keyed by the SAME ``(op, dtype, shape-bucket)`` the runtime cache-prune uses
(read off the kernel's ``make_cache_prune`` hook via its ``_miniworld_op`` /
``_miniworld_dtype_of`` / ``_miniworld_bucket_of`` tags), so a built cache is guaranteed to hit.

Usage (on the target GPU, with the full grid unlocked so every config is benched):

    MINIWORLD_RUN_AUTOTUNE=1
    from miniworld_engine.autotune import capture
    capture.install()
    ... run each wired module fwd+bwd across representative shapes ...
    capture.flush(top_k=5)     # writes <cache-root>/autotune/<op>/<gpu>.json for every op seen

Any wired kernel that fires during the run is captured automatically — no per-kernel code. The
cross-check against the hand-built pilot caches (transition_split_fwd / trimul_bidir_front)
validates the capture path. Config choice is performance-only, so this never affects numerics.
"""

from __future__ import annotations

from pathlib import Path

from .cache import (
    as_cfg_dict,
    config_space_hash,
    config_to_dict,
    gpu_key,
    store_ranked_configs,
)
from .cache import _sig_from_dict as _sig_from_dict

# op -> {"grid": [configs] | None, "entries": {(dtype, bucket): {sig: (config, ms)}}}
_CAPTURE: dict = {}
_orig_bench = None
_orig_compile = None
_orig_prune = None
# Wall-clock budget (seconds) for compiling a SINGLE config during a build, enforced by forking the
# whole triton compile into a child and SIGKILLing it on overrun (see install()). A healthy config
# compiles in a few seconds; a register-spill config runs 10-20 min across make_llir (libLLVM) and
# make_cubin (ptxas). This is the SOLE compile-time guard and is arch-agnostic BY CONSTRUCTION: it
# judges each config by how long it actually takes to compile on THIS GPU, so every arch keeps only
# the configs that compile in time on it — no static num_warps/num_stages heuristic (which would be
# tuned to one arch). This is the arch-specific BACKSTOP; a static ``_drop_compile_monsters`` prune
# in cache.py already removes the always-bad ``num_warps>=16`` / ``num_stages>=5`` tail up front, so
# this only has to catch a given arch's residual oddball blowups. The full grid /
# ``config_space_hash`` is untouched, so all caches stay valid. 60s cleanly separates the two
# populations: a healthy config (even at large D) compiles in a few to a few-tens of seconds, while
# a register-spill monster runs 10-20 MINUTES — so 60s never kills a usable config yet kills fast.
_COMPILE_BUDGET_S = 60


# --------------------------------------------------------------------------- #
# parallel PRE-compilation of an autotune round (build-time only)
# --------------------------------------------------------------------------- #
# A build is host-bound, and not because compilation is slow -- because triton runs it SERIALLY.
# ``Autotuner.run`` times a round with one comprehension:
#
#     timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
#
# so each config compiles (seconds, ONE core) and is then timed (milliseconds, GPU) before the next
# one starts. A 800-config round spends ~40 minutes compiling with the GPU ~99% idle and 31 of 32
# cores idle. Nothing about the work requires that: the compiles are independent.
#
# So: pre-compile the WHOLE round in parallel, then let triton run its serial loop, where every
# compile is now an on-disk cache hit. Timing stays strictly serial and exclusive -- that part must
# not be parallelised, since concurrent benchmarking on one GPU corrupts the very measurements the
# cache is built from.
#
# The parent BLOCKS until the round is compiled. An earlier attempt fired the pool off with
# map_async so the parent could "bench meanwhile", and that is exactly what defeated it: the parent
# raced ahead compiling config 2, 3, 4 itself while the workers were still importing torch, so the
# same configs were compiled twice and the wall clock did not move (260s vs 271s). Waiting is the
# point.
#
# Workers are SPAWNED, never forked: the parent has initialised CUDA by now, and a forked child may
# not re-enter the driver ("Cannot re-initialize CUDA in forked subprocess"). A spawned worker
# rebuilds the ASTSource from plain data plus a re-imported JITFunction and compiles against the
# target the parent already resolved, so it never touches the driver at all.
#
# A mis-synthesised variant is harmless: triton keys the cache by the arguments, so it just writes
# an entry nobody looks up. Worst case is wasted CPU, never a wrong binary.
_PRECOMPILE_POOL = None
_PRECOMPILE = {"rounds": 0, "configs": 0, "compiled": 0, "failed": 0, "seconds": 0.0}
#: autotuner id -> configs of the round it is about to time, set by the patched prune_configs.
#: Keyed per autotuner because rounds interleave: a module fires several kernels, and a single
#: "pending" slot loses every round but the last (which is how an earlier version silently
#: pre-compiled only the first round of a whole build).
_ROUND: dict = {}
#: id of the autotuner whose _bench is currently on the stack, so the compile hook knows which
#: armed round the compile it is servicing belongs to.
_CURRENT: dict = {}


def precompile_summary() -> str:
    """One line on what the pre-compile actually did.

    A pool whose workers all die on arrival is indistinguishable from no pool at all -- the build
    just stays slow. Reporting the counts makes the difference visible instead of inferred.
    """
    p = _PRECOMPILE
    return (f"  [precompile] jobs={_compile_jobs()} rounds={p['rounds']} configs={p['configs']} "
            f"-> compiled={p['compiled']} failed={p['failed']} in {p['seconds']:.0f}s")


def _compile_jobs() -> int:
    """Workers for pre-compilation: ``settings.compile_jobs``, else one per usable core (cap 32)."""
    import os  # noqa: PLC0415

    from miniworld_engine import settings  # noqa: PLC0415

    want = settings.current().compile_jobs
    if want is not None:
        return max(1, int(want))
    try:
        cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = os.cpu_count() or 1
    return max(1, min(32, cores))


def _worker_compile(payload: tuple) -> bool:
    """Compile ONE config in a spawned worker. Must stay importable at module level (spawn).

    Touches no CUDA: it rebuilds the ASTSource from plain data plus a re-imported JITFunction and
    compiles against the ``target`` the parent resolved. The result lands in triton's on-disk cache;
    the return value is only for accounting.
    """
    module_path, fn_name, signature, constants, attrs, target, options = payload
    import os  # noqa: PLC0415
    import signal  # noqa: PLC0415
    import time  # noqa: PLC0415

    # Same wall-clock budget the serial path enforces, for the same reason -- and enforced the same
    # way, by running the compile in a child and SIGKILLing it on overrun (make_llir and ptxas both
    # block in native code, where a Python signal cannot reach them).
    #
    # This is not only about speed. The budget is what DECIDES whether a register-spill config is
    # usable on this GPU. If a worker compiled a 20-minute monster to completion, the serial pass
    # would find it in the on-disk cache, sail through, and keep a config the serial build drops --
    # so the same inputs would yield different caches depending on whether pre-compilation ran.
    # Forking here is safe: a worker never initialises CUDA, so it has no driver state to inherit.
    pid = os.fork()
    if pid == 0:
        try:
            import importlib  # noqa: PLC0415

            from triton.compiler.compiler import ASTSource, compile  # noqa: PLC0415

            from triton.runtime.jit import JITFunction  # noqa: PLC0415

            fn = getattr(importlib.import_module(module_path), fn_name)
            # The module attribute is whatever the decorators left there -- for an autotuned kernel
            # that is an Autotuner (or a Heuristics) wrapping the JITFunction, not the JITFunction
            # itself. ASTSource needs the inner one; triton unwraps the same way in
            # Autotuner.check_disk_cache.
            while not isinstance(fn, JITFunction):
                fn = fn.fn
            src = ASTSource(fn=fn, signature=signature, attrs=attrs)
            src.constants = constants   # already keyed by arg-index tuple, as ASTSource stores it
            compile(src, target=target, options=options)
            os._exit(0)
        except BaseException as exc:  # noqa: BLE001 -- serial pass recompiles whatever fails here
            # A child cannot report back through memory, and "every worker failed" is otherwise
            # indistinguishable from "the pool never ran". One line, from one child, says why.
            if os.environ.get("_MW_PRECOMPILE_FIRST") == "1":
                import sys  # noqa: PLC0415
                print(f"  [precompile] child failed: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
            os._exit(3)

    deadline = time.monotonic() + _COMPILE_BUDGET_S
    while True:
        try:
            done, status = os.waitpid(pid, os.WNOHANG)
        except OSError:
            return False
        if done == pid:
            return os.waitstatus_to_exitcode(status) == 0
        if time.monotonic() > deadline:
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except OSError:
                pass
            return False
        time.sleep(0.05)


def _precompile_round(src, target, options, configs) -> None:
    """Compile every config of this round in parallel, blocking until done. Never raises."""
    global _PRECOMPILE_POOL
    jobs = _compile_jobs()
    if jobs < 2 or len(configs) < 2:
        return
    import time  # noqa: PLC0415

    started = time.monotonic()
    try:
        import multiprocessing as mp  # noqa: PLC0415

        arg_names = list(src.fn.arg_names)
        base_constants = dict(src.constants)
        payloads = []
        for config in configs:
            constants = dict(base_constants)
            for name, value in config.kwargs.items():
                if name in arg_names:
                    constants[(arg_names.index(name),)] = value
            opts = dict(options)
            for knob in ("num_warps", "num_stages", "num_ctas", "maxnreg"):
                value = getattr(config, knob, None)
                if value is not None:
                    opts[knob] = value
            payloads.append((src.fn.module, src.fn.__name__, src.signature, constants,
                             src.attrs, target, opts))
        import os as _os  # noqa: PLC0415
        _os.environ["_MW_PRECOMPILE_FIRST"] = "1" if _PRECOMPILE["rounds"] == 0 else "0"

        if _PRECOMPILE_POOL is None:
            _PRECOMPILE_POOL = mp.get_context("spawn").Pool(jobs)

        # Bound the wait so one register-spill monster cannot stall the build: stragglers keep
        # compiling into the shared on-disk cache, and the serial pass picks up whatever landed.
        budget = _COMPILE_BUDGET_S * (len(payloads) // jobs + 2)
        results = _PRECOMPILE_POOL.map_async(_worker_compile, payloads, chunksize=1)
        try:
            done = results.get(timeout=budget)
        except Exception:  # noqa: BLE001 -- timeout: proceed, the serial pass still works
            done = []
        ok = sum(1 for d in done if d)
        bad = sum(1 for d in done if not d)
        _PRECOMPILE["compiled"] += ok
        _PRECOMPILE["failed"] += bad
        _PRECOMPILE["rounds"] += 1
        _PRECOMPILE["configs"] += len(payloads)
        # Printed per round, not just in the final summary: when this stalled before, nothing was
        # visible until the shard finished, so a dead pool and a slow one looked identical.
        print(f"  [precompile] round {_PRECOMPILE['rounds']}: {len(payloads)} configs on {jobs} "
              f"workers -> {ok} compiled, {bad} failed, "
              f"{time.monotonic() - started:.0f}s", flush=True)
    except Exception:  # noqa: BLE001 -- an optimisation; a build must never fail because of it
        pass
    _PRECOMPILE["seconds"] += time.monotonic() - started


def shutdown_precompile() -> None:
    global _PRECOMPILE_POOL
    if _PRECOMPILE_POOL is not None:
        try:
            _PRECOMPILE_POOL.terminate()
        except Exception:  # noqa: BLE001
            pass
        _PRECOMPILE_POOL = None


def _median(t) -> float:
    """do_bench(quantiles=(0.5,0.2,0.8)) returns [median, q20, q80]; be tolerant of a scalar."""
    if isinstance(t, (list, tuple)):
        return float(t[0])
    return float(t)


def _sig(config) -> tuple:
    d = as_cfg_dict(config)
    kw = tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v)
                      for k, v in d["kwargs"].items()))
    return (kw, d["num_warps"], d["num_stages"])


def _record_one(autotuner, config, meta, ms) -> None:
    ecp = getattr(autotuner, "early_config_prune", None)
    op = getattr(ecp, "_miniworld_op", None)
    if not op:
        return
    if ms == float("inf"):
        return
    nargs = getattr(autotuner, "nargs", None) or {}
    dtype = str(ecp._miniworld_dtype_of(nargs, meta))   # noqa: SLF001
    bucket = str(ecp._miniworld_bucket_of(nargs, meta))  # noqa: SLF001
    slot = _CAPTURE.setdefault(op, {"grid": None, "entries": {}})
    if slot["grid"] is None:
        slot["grid"] = list(autotuner.configs)
    ent = slot["entries"].setdefault((dtype, bucket), {})
    sig = _sig(config)
    prev = ent.get(sig)
    if prev is None or ms < prev[1]:   # keep the fastest reading for this config
        ent[sig] = (config, ms)


def install() -> None:
    """Patch Autotuner._bench (capture timings) and triton.compile (bound compile time). Idempotent.

    A register-spill config makes triton's compile pipeline run for 10-20 MINUTES — split across
    ``make_llir`` (in-process libLLVM) and ``make_cubin`` (ptxas subprocess). NEITHER can be bounded
    by a Python SIGALRM (both block in native code; PEP 475 restarts the syscall — confirmed by two
    py-spy stacks), and a subprocess timeout would only catch the ptxas half. The only guard that
    covers BOTH halves and is arch-agnostic is to run the WHOLE compile in a forked child bounded by
    wall-clock: the child compiles into triton's on-disk cache and exits; the parent waits up to
    ``_COMPILE_BUDGET_S`` and SIGKILLs the child on overrun (a hard kill of a process, immune to PEP
    475). On success the parent re-calls compile — now an instant disk-cache hit. On overrun/failure
    it raises, so ``_bench`` records +inf and the config is skipped. Each GPU thus keeps exactly the
    configs that compile in time on IT — no static num_warps/num_stages heuristic (which would be
    tuned to one arch). The child does a PURE ``compile`` (never ``warmup``), so it touches neither
    torch nor the CUDA driver and is safe to fork after the parent has initialized CUDA.
    """
    global _orig_bench, _orig_compile
    if _orig_bench is not None:
        return
    import os  # noqa: PLC0415
    import signal  # noqa: PLC0415
    import time  # noqa: PLC0415
    import triton.compiler as _tc  # noqa: PLC0415
    import triton.compiler.compiler as _tcc  # noqa: PLC0415
    from triton.runtime.autotuner import Autotuner

    _orig_compile = _tcc.compile

    def _fork_compile(*args, **kwargs):  # noqa: ANN002, ANN003
        # This is the only place fully-resolved compile arguments exist, so it is where a round's
        # fan-out has to start -- but only for the round currently being timed, and only once.
        src = args[0] if args else kwargs.get("src")
        current = _CURRENT.get("id")
        armed = _ROUND.pop(current, None) if current is not None else None
        if armed and src is not None and kwargs.get("target") is not None:
            _precompile_round(src, kwargs["target"], kwargs.get("options"), armed)
        if _COMPILE_BUDGET_S <= 0:
            return _orig_compile(*args, **kwargs)
        pid = os.fork()
        if pid == 0:  # child: compile into the on-disk cache, no torch/CUDA touched, then exit
            try:
                _orig_compile(*args, **kwargs)
                os._exit(0)
            except BaseException:  # noqa: BLE001 -- any failure -> parent treats config as unusable
                os._exit(3)
        deadline = time.monotonic() + _COMPILE_BUDGET_S
        while True:
            w, st = os.waitpid(pid, os.WNOHANG)
            if w == pid:
                break
            if time.monotonic() > deadline:
                try:
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
                except OSError:
                    pass
                raise RuntimeError(
                    f"triton compile exceeded {_COMPILE_BUDGET_S}s (register-spill config); skipped")
            time.sleep(0.05)
        if os.waitstatus_to_exitcode(st) != 0:
            raise RuntimeError("triton compile failed in isolated child; config skipped")
        return _orig_compile(*args, **kwargs)  # disk-cache hit -> instant

    _tcc.compile = _fork_compile
    _tc.compile = _fork_compile  # the name create_binder rebinds via `from ..compiler import compile`

    # The pruned config list is the round's work item; hand it to the compile hook, which is where
    # the fully-resolved compile arguments to clone from first appear.
    global _orig_prune
    _orig_prune = Autotuner.prune_configs

    def prune_configs(self, kwargs):  # noqa: ANN001, ANN202
        pruned = _orig_prune(self, kwargs)
        _ROUND[id(self)] = list(pruned)     # per autotuner: rounds interleave across kernels
        return pruned

    Autotuner.prune_configs = prune_configs

    _orig_bench = Autotuner._bench

    def _bench(self, *args, config, **meta):
        # Tell the compile hook which round it is servicing: the first config that actually
        # compiles triggers the fan-out for the whole round.
        previous = _CURRENT.get("id")
        _CURRENT["id"] = id(self)
        try:
            res = _orig_bench(self, *args, config=config, **meta)
        except Exception:  # noqa: BLE001 -- a config that fails to compile/run simply loses
            # Match triton's own sentinel SHAPE, not just its value: do_bench(quantiles=...) hands
            # back [median, q20, q80], and triton returns [inf, inf, inf] for a config it could not
            # build. Returning a bare float mixes scalars and lists in the timings dict, and the
            # `min(timings, key=timings.get)` that picks the winner then dies with
            # "'<' not supported between instances of 'float' and 'list'" -- taking the whole shard
            # with it. Only shows up once at least one config fails, which is why it surfaced when
            # the compile budget started dropping configs.
            res = [float("inf")] * 3
        finally:
            _CURRENT["id"] = previous
        try:
            _record_one(self, config, meta, _median(res))
        except Exception:  # noqa: BLE001 -- capture must never perturb a real bench
            pass
        return res

    Autotuner._bench = _bench


def uninstall() -> None:
    global _orig_bench, _orig_compile, _orig_prune
    shutdown_precompile()
    if _orig_bench is not None:
        from triton.runtime.autotuner import Autotuner
        Autotuner._bench = _orig_bench
        _orig_bench = None
        if _orig_prune is not None:
            Autotuner.prune_configs = _orig_prune
            _orig_prune = None
    if _orig_compile is not None:
        import triton.compiler as _tc  # noqa: PLC0415
        import triton.compiler.compiler as _tcc  # noqa: PLC0415
        _tcc.compile = _orig_compile
        _tc.compile = _orig_compile
        _orig_compile = None


def reset() -> None:
    _CAPTURE.clear()


def flush(top_k: int = 5, gpu: str | None = None) -> list:
    """Write the captured top-K per (op, dtype, bucket) to the runtime cache. Returns a list of
    (op, dtype, bucket, n_configs, path) for logging."""
    gk = gpu or gpu_key()
    written = []
    for op, slot in sorted(_CAPTURE.items()):
        grid = slot["grid"] or []
        if not grid:
            continue
        csh = config_space_hash(grid)
        for (dtype, bucket), ent in sorted(slot["entries"].items()):
            ranked = sorted(ent.values(), key=lambda cm: cm[1])   # (config, ms) fastest-first
            fp = store_ranked_configs(op, gk, dtype, bucket, ranked, csh, top_k=top_k)
            written.append((op, dtype, bucket, len(ranked), str(fp)))
    return written


def dump_shard(path: str) -> int:
    """Serialize this process's captured timings to a standalone JSON SHARD file (NOT the
    in-repo cache). Parallel capture jobs each ``dump_shard`` to their OWN file; a single
    ``merge_shards`` writer folds them into the committed cache — so no env var and no
    concurrent writers ever touch the in-repo tree. Returns the number of ops dumped."""
    import json  # noqa: PLC0415
    out: dict = {}
    for op, slot in _CAPTURE.items():
        grid = slot["grid"] or []
        entries = {f"{d}|{b}": [config_to_dict(c, ms) for c, ms in ent.values()]
                   for (d, b), ent in slot["entries"].items()}
        out[op] = {"grid": [config_to_dict(c) for c in grid], "entries": entries}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    return len(out)


def merge_shards(shard_paths, top_k: int = 5, gpu: str | None = None, only_ops=None) -> list:
    """Fold shard files (from ``dump_shard``) into the in-repo cache as the SOLE writer.

    Unions buckets across shards, keeping the fastest reading per config. ``only_ops`` (a set)
    restricts the write to those op names — so a targeted build never rewrites an op that already
    has a good cache. Returns ``(op, dtype|bucket, n_configs, path)`` rows for logging."""
    import json  # noqa: PLC0415
    gk = gpu or gpu_key()
    agg: dict = {}
    for sp in shard_paths:
        try:
            d = json.loads(Path(sp).read_text())
        except Exception:  # noqa: BLE001 -- skip unreadable/partial shard
            continue
        for op, slot in d.items():
            if only_ops is not None and op not in only_ops:
                continue
            a = agg.setdefault(op, {"grid": slot.get("grid", []), "entries": {}})
            if not a["grid"]:
                a["grid"] = slot.get("grid", [])
            for bk, lst in slot.get("entries", {}).items():
                ent = a["entries"].setdefault(bk, {})
                for cd in lst:
                    sig = _sig_from_dict(cd)
                    ms = float(cd.get("ms", float("inf")))
                    prev = ent.get(sig)
                    if prev is None or ms < prev[1]:
                        ent[sig] = (cd, ms)
    written = []
    for op, a in sorted(agg.items()):
        grid = a["grid"]
        if not grid:
            continue
        csh = config_space_hash(grid)
        for bk, ent in sorted(a["entries"].items()):
            dtype, bucket = bk.split("|", 1)
            ranked = sorted(ent.values(), key=lambda cm: cm[1])
            fp = store_ranked_configs(op, gk, dtype, bucket, list(ranked), csh, top_k=top_k)
            written.append((op, bk, len(ranked), str(fp)))
    return written


def summary() -> str:
    lines = []
    for op, slot in sorted(_CAPTURE.items()):
        n_buckets = len(slot["entries"])
        n_grid = len(slot["grid"] or [])
        lines.append(f"  {op}: grid={n_grid} buckets={n_buckets}")
        for (dtype, bucket), ent in sorted(slot["entries"].items()):
            best = min(ent.values(), key=lambda cm: cm[1]) if ent else None
            tag = f"{dtype}|{bucket}"
            if best:
                lines.append(f"    {tag}: {len(ent)} configs, best {best[1]:.4f}ms "
                             f"{as_cfg_dict(best[0])['kwargs']}")
    return "\n".join(lines) if lines else "  (nothing captured)"
