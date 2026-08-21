"""Generic autotune-cache BUILDER via Triton-autotuner instrumentation.

Instead of hand-replicating each kernel's launch in a per-kernel builder (fragile, and how the
int32-offset bug slipped in), this patches ``triton.runtime.autotuner.Autotuner._bench`` to
record every (config -> measured ms) as it is benchmarked during a REAL module forward/backward
run. The op behind a live autotuner is recovered by identity of the config list it was handed
(:func:`autotune.configs.op_of`) -- the only back-reference left now that the prune objects that
used to carry the name are gone.

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
    bucket_of_autotuner as _bucket_of,
    dtype_of_args as _dtype_of,
    gpu_key,
    store_ranked_configs,
)
from .cache import _sig_from_dict as _sig_from_dict

# op -> {"grid": [configs] | None, "entries": {(dtype, bucket): {sig: (config, ms)}}}
_CAPTURE: dict = {}
_orig_bench = None
_orig_run = None
_SINGLE_SEEN: set = set()
_orig_compile = None
_orig_prune = None
# Wall-clock budget (seconds) for compiling a SINGLE config during a build, enforced by forking the
# whole triton compile into a child and SIGKILLing it on overrun (see install()). A healthy config
# compiles in a few seconds; a register-spill config runs 10-20 min across make_llir (libLLVM) and
# make_cubin (ptxas). This is the SOLE compile-time guard and is arch-agnostic BY CONSTRUCTION: it
# judges each config by how long it actually takes to compile on THIS GPU, so every arch keeps only
# the configs that compile in time on it — no static num_warps/num_stages heuristic (which would be
# tuned to one arch). This is the SOLE guard: cache.py used to pre-filter the ``num_warps>=16`` /
# ``num_stages>=5`` tail statically, but that rule shrank the searched space behind the caller's
# back and its premise did not hold -- ``num_stages`` has no compiler maximum, and its ceiling
# (``smem_limit / operand_tile``) is far above 5 for small tiles. The full grid /
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

#: autotuner id -> best median (ms) seen so far in the CURRENT round. Reset by prune_configs.
_BEST: dict = {}

#: configs abandoned by the bench budget, for the summary. Counted, never silent: a build that
#: skipped nine tenths of its grid and one that measured all of it produce the same cache file.
#:
#: The seconds are split so the remaining bench time can be ATTRIBUTED rather than guessed. Three
#: candidates look identical in a total: slow kernels running to completion before being judged,
#: the fixed per-config cost of probing at all (two launches, two syncs, python round-trips), and
#: the full do_bench of the configs that survive. They call for different fixes, so they are
#: measured apart. ``kernel_ms`` is device time from cuda events; the ``*_s`` are host wall-clock,
#: and the gap between them IS the fixed overhead.
_ABANDONED: dict = {"skipped": 0, "measured": 0, "warm_s": 0.0, "probe_s": 0.0,
                    "bench_s": 0.0, "kernel_ms": 0.0, "skipped_kernel_ms": 0.0}

#: Where a compile() call actually spends its time. The guard forks a child, waits for it, and then
#: recompiles in-process expecting a cache hit -- three costs that look like one. They have
#: different fixes: fork is proportional to the parent's RSS, the wait is quantised by the poll
#: interval, and the in-process call is a lookup in ~/.triton/cache, which currently holds 88k+
#: entries. Measured apart so the fix targets the real one.
_COMPILE_T: dict = {"calls": 0, "fork_s": 0.0, "wait_s": 0.0, "parent_s": 0.0, "polls": 0}

#: First-launch cost per config, split. warm_s minus compile() left ~500s unexplained on a unit
#: whose kernels ran 2.2s of device time; these are the three things a first launch does that a
#: second one does not -- load the cubin into the context, build the launcher stub, and run the
#: autotuner's pre_hook (which zeroes reset_to_zero tensors).
_LAUNCH_T: dict = {"init_handles": 0, "init_s": 0.0, "launcher_s": 0.0,
                   "prehook_calls": 0, "prehook_s": 0.0, "run_calls": 0, "run_s": 0.0}

#: (device_ms, config kwargs) for every probed config. The budget knows a config is slow; it does
#: not know WHY, and "drop the slow ones" is only actionable once the slow ones share an attribute.
#: Kept in memory and dumped next to the shard.
_PROBE_LOG: list = []
_CURRENT_CFG: dict = {}

def _sig_line(cfg: dict) -> str:
    return ",".join(f"{k}={cfg[k]}" for k in sorted(cfg))


#: sig -> median ms for every probe this unit has already decided, across attempts. A kill costs a
#: restart, and a restart that re-benched everything up to the killed config would make the guard
#: cost more than the hang it prevents -- quadratic in the number of bad configs. Replaying the
#: decision instead makes a restart cost only the configs never reached.
_PROBE_DONE: dict = {}
_PROBE_FILE: list = []


def load_probe_state(shard_path) -> int:
    """Load the probes and compiles this unit already settled. Returns how many probes replay.

    Kept after the bench watchdog was removed: it no longer serves kill-and-restart, it serves any
    restart. A unit that dies to an OOM, a node failure, or a job time limit resumes without
    re-benching or re-compiling what it had already decided.
    """
    from pathlib import Path as _P  # noqa: PLC0415

    stem = str(shard_path)[: -len(".json")] if str(shard_path).endswith(".json") else str(shard_path)
    cf = _P(stem + ".compiled")
    _COMPILED_FILE.append(cf)
    if cf.exists():
        _COMPILED.update(ln.strip() for ln in cf.read_text().splitlines() if ln.strip())
    pf = _P(stem + ".probes")
    _PROBE_FILE.append(pf)
    if pf.exists():
        for ln in pf.read_text().splitlines():
            sig, _, ms = ln.rpartition("\t")
            if sig:
                try:
                    _PROBE_DONE[sig] = float(ms)
                except ValueError:
                    continue
    return len(_PROBE_DONE)


#: sigs whose compile is already in the on-disk triton cache from an earlier attempt at this unit.
#: The probe replay made a restart skip re-BENCHING; without the same trick for compiling, every
#: restart still re-submitted the whole round to a fresh spawn pool -- 446 s a time, which is what
#: turned 87 legitimate kills into 7799 s. The triton cache is already warm for these; the cost
#: being avoided is the pool spawn and the job round-trip, not the compile itself.
_COMPILED: set = set()
_COMPILED_FILE: list = []


def _mark_compiled(sigs) -> None:
    """Record configs whose compile is SETTLED -- succeeded or permanently failed.

    Failures have to be recorded too. A config that fails to compile fails deterministically (it
    spills registers, or wants more smem than the card has), so retrying it on every restart is
    pure loss: 38 such configs cost 206 s per attempt and, across ~90 attempts, dominated the
    restart budget the compiled-list was added to remove. A settled failure is as reusable a fact
    as a settled success -- the bench scores it +inf either way.
    """
    new_sigs = [x for x in sigs if x not in _COMPILED]
    if not new_sigs:
        return
    _COMPILED.update(new_sigs)
    if _COMPILED_FILE:
        try:
            with _COMPILED_FILE[0].open("a") as fh:
                fh.write("".join(x + "\n" for x in new_sigs))
        except OSError:
            pass


def _cfg_sig(config) -> str:
    d = dict(config.kwargs)
    d["num_warps"] = config.num_warps
    d["num_stages"] = config.num_stages
    return _sig_line(d)


def _remember(sig: str, ms: float) -> None:
    _PROBE_DONE[sig] = ms
    if _PROBE_FILE:
        try:
            with _PROBE_FILE[0].open("a") as fh:
                fh.write(f"{sig}\t{ms}\n")
        except OSError:
            pass


def probe_log_summary(top: int = 0) -> str:
    """Per-axis mean device time over every probed config -- which axis value costs the time."""
    import collections  # noqa: PLC0415

    if not _PROBE_LOG:
        return ""
    per = collections.defaultdict(list)
    for ms, cfg in _PROBE_LOG:
        for k, v in cfg.items():
            per[(k, v)].append(ms)
    lines = [f"  [probe-log] {len(_PROBE_LOG)} configs probed; mean device ms by axis value:"]
    for (k, v), vals in sorted(per.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        lines.append(f"      {k}={v!s:<6} n={len(vals):<5} mean={sum(vals) / len(vals):8.2f}ms"
                     f"  min={min(vals):7.3f}ms")
    if top:
        worst = sorted(_PROBE_LOG, key=lambda r: -r[0])[:top]
        lines.append("    slowest configs:")
        lines += [f"      {ms:9.1f}ms  {cfg}" for ms, cfg in worst]
    return "\n".join(lines)


def _install_launch_probes() -> None:
    """Time the per-config first-launch path. Idempotent; build-only."""
    import time  # noqa: PLC0415

    from triton.compiler.compiler import CompiledKernel  # noqa: PLC0415
    from triton.runtime.autotuner import Autotuner  # noqa: PLC0415
    from triton.runtime.jit import JITFunction  # noqa: PLC0415

    if getattr(CompiledKernel, "_mw_probed", False):
        return
    CompiledKernel._mw_probed = True

    orig_init = CompiledKernel._init_handles

    def init_handles(self):  # noqa: ANN001, ANN202
        if self.module is not None:
            return orig_init(self)
        t = time.monotonic()
        try:
            return orig_init(self)
        finally:
            _LAUNCH_T["init_handles"] += 1
            _LAUNCH_T["init_s"] += time.monotonic() - t

    CompiledKernel._init_handles = init_handles

    orig_run = JITFunction.run

    def run(self, *a, **k):  # noqa: ANN001, ANN002, ANN003, ANN202
        t = time.monotonic()
        try:
            return orig_run(self, *a, **k)
        finally:
            _LAUNCH_T["run_calls"] += 1
            _LAUNCH_T["run_s"] += time.monotonic() - t

    JITFunction.run = run

    orig_pre = Autotuner.pre_hook if hasattr(Autotuner, "pre_hook") else None
    if callable(orig_pre):
        def pre_hook(self, *a, **k):  # noqa: ANN001, ANN002, ANN003, ANN202
            t = time.monotonic()
            try:
                return orig_pre(self, *a, **k)
            finally:
                _LAUNCH_T["prehook_calls"] += 1
                _LAUNCH_T["prehook_s"] += time.monotonic() - t

        Autotuner.pre_hook = pre_hook


def _budget_ms(autotuner) -> float | None:
    """Bench budget for the next config, or None when the feature is off."""
    from miniworld_engine import settings  # noqa: PLC0415

    cur = settings.current()
    factor = cur.bench_budget_factor
    if not factor:
        return None
    best = _BEST.get(id(autotuner))
    cap = cur.bench_budget_cap_ms
    return cap if best is None else min(cap, best * factor)


def _budgeted_do_bench(orig, budget: float):
    """``do_bench`` that probes once and gives up if the config is already out of the running.

    One untimed call first: the timed one must not include a JIT compile (the precompile round
    normally removes that, but a cache miss here would read as a slow kernel and abandon a config
    for the wrong reason). Then one timed launch -- enough to separate a config that is 10x the
    best from one that might win, which is the only distinction the budget needs to make.
    """
    import time  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from miniworld_engine import settings as _settings  # noqa: PLC0415

    skip_warm = _settings.current().bench_budget_skip_warm

    def f(kernel_call, quantiles=None, **kw):  # noqa: ANN001, ANN003, ANN202
        sig = _sig_line(_CURRENT_CFG)
        if sig in _PROBE_DONE:          # decided on an earlier attempt: replay, do not relaunch
            m = _PROBE_DONE[sig]
            _ABANDONED["skipped" if m == float("inf") else "measured"] += 1
            return [m] * 3
        if not skip_warm:
            t0 = time.perf_counter()
            kernel_call()                   # warm: compile / cache / allocator
            torch.cuda.synchronize()
            _ABANDONED["warm_s"] += time.perf_counter() - t0

        t1 = time.perf_counter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        kernel_call()
        end.record()
        torch.cuda.synchronize()
        probe_wall = time.perf_counter() - t1
        dev_ms = start.elapsed_time(end)
        _ABANDONED["probe_s"] += probe_wall
        _ABANDONED["kernel_ms"] += dev_ms

        if dev_ms > budget:
            _ABANDONED["skipped"] += 1
            _ABANDONED["skipped_kernel_ms"] += dev_ms
            _PROBE_LOG.append((dev_ms, dict(_CURRENT_CFG)))
            _remember(sig, float("inf"))
            return [float("inf")] * 3
        _PROBE_LOG.append((dev_ms, dict(_CURRENT_CFG)))
        _ABANDONED["measured"] += 1
        t2 = time.perf_counter()
        res = orig(kernel_call, quantiles=quantiles, **kw)
        _ABANDONED["bench_s"] += time.perf_counter() - t2
        _remember(sig, _median(res))
        return res

    return f


def precompile_summary() -> str:
    """One line on what the pre-compile actually did.

    A pool whose workers all die on arrival is indistinguishable from no pool at all -- the build
    just stays slow. Reporting the counts makes the difference visible instead of inferred.
    """
    p = _PRECOMPILE
    budget = ""
    a = _ABANDONED
    if a["skipped"] or a["measured"]:
        tot = a["skipped"] + a["measured"]
        host = a["warm_s"] + a["probe_s"] + a["bench_s"]
        dev = a["kernel_ms"] / 1000.0
        budget = (
            f"\n  [bench-budget] {a['skipped']}/{tot} abandoned"
            f" | warm {a['warm_s']:.0f}s + probe {a['probe_s']:.0f}s"
            f" + full-bench {a['bench_s']:.0f}s = {host:.0f}s host"
            f" | device {dev:.1f}s (of which abandoned kernels"
            f" {a['skipped_kernel_ms'] / 1000.0:.1f}s)"
            f" | fixed overhead {host - dev:.0f}s")
    c = _COMPILE_T
    if c["calls"]:
        budget += (f"\n  [compile-guard] {c['calls']} compile() calls"
                   f" | fork {c['fork_s']:.0f}s + wait {c['wait_s']:.0f}s"
                   f" ({c['polls']} polls x 50ms) + parent-recompile {c['parent_s']:.0f}s"
                   f" = {c['fork_s'] + c['wait_s'] + c['parent_s']:.0f}s")
    lt = _LAUNCH_T
    if lt["run_calls"]:
        budget += (f"\n  [first-launch] fn.run {lt['run_calls']} calls {lt['run_s']:.0f}s"
                   f" | init_handles {lt['init_handles']} x -> {lt['init_s']:.0f}s"
                   f" | pre_hook {lt['prehook_calls']} x -> {lt['prehook_s']:.0f}s")
    return (f"  [precompile] jobs={_compile_jobs()} rounds={p['rounds']} configs={p['configs']} "
            f"-> compiled={p['compiled']} failed={p['failed']} in {p['seconds']:.0f}s{budget}")


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
    # A single config still goes through the pool. The parallelism is pointless there, but the
    # TIMEOUT is not: the worker forks and gets SIGKILLed past _COMPILE_BUDGET_S, which is the only
    # bound that exists on a compile. Bailing out at len(configs) < 2 sent every pinned-config-set
    # run straight into triton's serial in-process compile, where a pathological config stalls with
    # no output and nothing can interrupt it.
    if jobs < 1 or not configs:
        return
    import time  # noqa: PLC0415

    started = time.monotonic()
    todo = [c for c in configs if _cfg_sig(c) not in _COMPILED]
    skipped = len(configs) - len(todo)
    if not todo:
        print(f"  [precompile] round skipped: all {len(configs)} configs already compiled "
              f"on an earlier attempt", flush=True)
        return
    configs = todo
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
        # settled = attempted and answered, pass or fail. `done` is shorter than `configs` only
        # if the pool timed out; zip stops at the shorter one, so an unanswered config stays
        # unsettled and is retried, which is the intent.
        _mark_compiled(_cfg_sig(c) for c, _ in zip(configs, done))
        _PRECOMPILE["compiled"] += ok
        _PRECOMPILE["failed"] += bad
        _PRECOMPILE["rounds"] += 1
        _PRECOMPILE["configs"] += len(payloads)
        # Printed per round, not just in the final summary: when this stalled before, nothing was
        # visible until the shard finished, so a dead pool and a slow one looked identical.
        print(f"  [precompile] round {_PRECOMPILE['rounds']}: {len(payloads)} configs "
              f"({skipped} already compiled) on {jobs} "
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


#: ops whose autotuner actually launched this process, in first-launch order.
_LAUNCHED: dict[str, int] = {}


#: autotuner ids already attributed to an op, so the hot path costs one set lookup.
_NOTED: set = set()
_REC_INSTALLED = False


def launched_ops() -> dict[str, int]:
    """op -> number of distinct autotuners that ran for it. See :func:`install_launch_recorder`."""
    return dict(_LAUNCHED)


def install_launch_recorder() -> None:
    """Record which ops actually launch. Idempotent, and safe inside a timing benchmark.

    Separate from :func:`install`: capture pins dispatch switches and forces a full-grid search,
    which is right for a build and wrong for a measurement. Coverage has to be answerable during
    an ordinary bench, otherwise "we ran the kernels" stays an assumption. Attribution happens
    once per autotuner (memoised on id), so the per-launch cost is a set membership test.
    """
    global _REC_INSTALLED
    if _REC_INSTALLED:
        return
    _REC_INSTALLED = True
    from triton.runtime.autotuner import Autotuner  # noqa: PLC0415

    prev = Autotuner.run

    def run(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        key = id(self)
        if key not in _NOTED:
            _NOTED.add(key)
            op = _op_name(self)
            if op:
                _LAUNCHED[op] = _LAUNCHED.get(op, 0) + 1
        return prev(self, *args, **kwargs)

    Autotuner.run = run


def _op_name(autotuner) -> str | None:
    """The op behind a live autotuner, by identity of the config list it was handed."""
    from miniworld_engine.autotune.configs import op_of  # noqa: PLC0415
    return op_of(getattr(autotuner, "configs", None))


#: Recording errors, kept so a build cannot look healthy while capturing nothing.
_RECORD_ERRORS: dict[str, int] = {}


def _record_failed(exc: BaseException) -> None:
    key = f"{type(exc).__name__}: {exc}"
    first = key not in _RECORD_ERRORS
    _RECORD_ERRORS[key] = _RECORD_ERRORS.get(key, 0) + 1
    if first:
        import warnings  # noqa: PLC0415

        warnings.warn(f"[miniworld.autotune] capture could not record a timing -- {key}. "
                      f"This shard will be missing measurements.", stacklevel=3)


def record_errors() -> dict[str, int]:
    """Recording failures seen this process, by message. Empty is what a healthy build looks like."""
    return dict(_RECORD_ERRORS)


def _record_one(autotuner, config, meta, ms, *, unmeasured: bool = False, nargs=None) -> None:
    op = _op_name(autotuner)
    if not op:
        return
    # inf is how a FAILED config scores and must never be stored. NaN reaches here from exactly
    # one place and on purpose: an op with a single config runs no tuning loop, so the sole config
    # is the winner by default and there is no measurement to record (`unmeasured=True`). That is
    # why every one of the 37 caches built from the one-config sets holds `"ms": NaN`.
    #
    # A blanket `isfinite` filter here looks right and is wrong -- it silently stops the
    # single-config path recording anything at all. What actually needs handling is the ORDERING:
    # `sorted` by a NaN key neither raises nor orders, so an unmeasured entry merged alongside
    # measured ones could land at the head and be read as the winner. The rankers sort NaN last
    # (see `_rank`).
    if ms != ms and not unmeasured:
        return
    if ms == float("inf"):
        return
    # `nargs` is passed in by the single-config path, which records AFTER Autotuner.run has
    # finished -- and run() ends with `self.nargs = None`. Reading it off the autotuner there
    # yields nothing, so only KEYWORD arguments reach the bucket and every positional one
    # disappears: a kernel launched with a positional shape_key recorded a bucket with no shape
    # in it at all. Reconstructing the binding at the call site is the fix; making kernels pass
    # arguments by keyword to satisfy the recorder would be fixing production to suit the
    # measurement.
    if nargs is None:
        nargs = getattr(autotuner, "nargs", None) or {}
    # Always derive the key here; never off `early_config_prune`. That branch used to read
    # `_miniworld_dtype_of` / `_miniworld_bucket_of` from the per-kernel prune OBJECTS, and once
    # those were deleted in fcd3c7a the `if ecp` guard was only ever false -- until the cache
    # reader started installing a prune FUNCTION on every autotuner, at which point `ecp` became
    # truthy everywhere and every call raised AttributeError into the caller's `except: pass`.
    # Capture then silently recorded nothing at all: a build that looked like it ran and produced
    # an empty shard. Read and write share these two functions, which is the point.
    dtype = _dtype_of(nargs)
    bucket = _bucket_of(autotuner, nargs, meta)
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
        _COMPILE_T["calls"] += 1
        _t_fork = time.monotonic()
        pid = os.fork()
        if pid == 0:  # child: compile into the on-disk cache, no torch/CUDA touched, then exit
            try:
                _orig_compile(*args, **kwargs)
                os._exit(0)
            except BaseException:  # noqa: BLE001 -- any failure -> parent treats config as unusable
                os._exit(3)
        _COMPILE_T["fork_s"] += time.monotonic() - _t_fork
        _t_wait = time.monotonic()
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
            # Adaptive backoff, not a flat 50ms. Almost every compile here is an on-disk cache
            # hit that finishes in single-digit milliseconds, but a flat 50ms poll charges the
            # full tick to each one: 89,695 polls = 1848s of pure sleeping on one unit, 34% of its
            # wall clock. Start at 1ms so a hit is noticed almost immediately, and back off to
            # 50ms once it is clear this is a real compile, where the tick is irrelevant next to
            # the seconds it will take.
            waited = time.monotonic() - _t_wait
            time.sleep(0.001 if waited < 0.05 else 0.01 if waited < 0.5 else 0.05)  # noqa: ASYNC251
            _COMPILE_T["polls"] += 1
        _COMPILE_T["wait_s"] += time.monotonic() - _t_wait
        if os.waitstatus_to_exitcode(st) != 0:
            # Raising is right DURING a tuning round: _bench catches it, the config scores +inf and
            # is pruned, which is how the budget filters register-spill configs. It is wrong for a
            # launch outside tuning -- the winning config's own compile, or a kernel that is not
            # autotuned at all -- where the exception escapes into the module forward and takes the
            # whole unit down. Outside a round, compile in-process instead.
            if _CURRENT.get("id") is None:
                return _orig_compile(*args, **kwargs)
            raise RuntimeError("triton compile failed in isolated child; config skipped")
        _t_parent = time.monotonic()
        try:
            return _orig_compile(*args, **kwargs)  # expected: ~/.triton/cache hit
        finally:
            _COMPILE_T["parent_s"] += time.monotonic() - _t_parent

    _install_launch_probes()

    _tcc.compile = _fork_compile
    _tc.compile = _fork_compile  # the name create_binder rebinds via `from ..compiler import compile`

    # The pruned config list is the round's work item; hand it to the compile hook, which is where
    # the fully-resolved compile arguments to clone from first appear.
    global _orig_prune
    _orig_prune = Autotuner.prune_configs

    def prune_configs(self, kwargs):  # noqa: ANN001, ANN202
        pruned = _orig_prune(self, kwargs)
        _ROUND[id(self)] = list(pruned)     # per autotuner: rounds interleave across kernels
        # A round is exactly one prune_configs + one sweep of the pruned list for ONE autotune key,
        # so this is where the running best resets. Keeping it per autotuner rather than global:
        # rounds from different kernels interleave, and a fast elementwise kernel's best would
        # otherwise set an impossible budget for a GEMM.
        _BEST.pop(id(self), None)
        return pruned

    Autotuner.prune_configs = prune_configs

    _orig_bench = Autotuner._bench

    def _bench(self, *args, config, **meta):
        # Tell the compile hook which round it is servicing: the first config that actually
        # compiles triggers the fan-out for the whole round.
        previous = _CURRENT.get("id")
        _CURRENT["id"] = id(self)
        budget = _budget_ms(self)
        _CURRENT_CFG.clear()
        _CURRENT_CFG.update(config.kwargs)
        _CURRENT_CFG["num_warps"] = config.num_warps
        _CURRENT_CFG["num_stages"] = config.num_stages
        saved_do_bench = getattr(self, "do_bench", None)
        if budget is not None and saved_do_bench is not None:
            self.do_bench = _budgeted_do_bench(saved_do_bench, budget)
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
            if budget is not None and saved_do_bench is not None:
                self.do_bench = saved_do_bench
        med = _median(res)
        if med != float("inf"):
            prev = _BEST.get(id(self))
            if prev is None or med < prev:
                _BEST[id(self)] = med
        try:
            _record_one(self, config, meta, med)
        except Exception as exc:  # noqa: BLE001 -- capture must never perturb a real bench
            # ...but it must not fail SILENTLY either. Swallowing without a word is how a
            # recorder that raised on every single call produced an empty shard from a build
            # that otherwise looked healthy.
            _record_failed(exc)
        return res

    Autotuner._bench = _bench

    # A single-config autotuner never reaches _bench: triton gates the whole tuning path on
    # `if len(self.configs) > 1`, so with one config it just launches. That is the normal shape of
    # a pinned config set, and without this hook such a build captures nothing and every unit is
    # reported EMPTY even though the kernel ran fine. Time it once per (autotuner, key) and record
    # it, so a pinned build produces a cache entry naming the config it actually used.
    global _orig_run
    if _orig_run is None:
        _orig_run = Autotuner.run

    def run(self, *args, **kwargs):
        # triton gates its whole tuning path on len(configs) > 1, so a pinned single config never
        # reaches prune_configs -- which is where a round is armed for the compile hook. Arm it
        # here instead, or the one compile that matters runs serially in-process with no timeout.
        cfgs0 = getattr(self, "configs", None) or []
        # Snapshot the positional binding NOW: triton builds exactly this
        # (`self.nargs = dict(zip(self.arg_names, args))`) at the top of run() and clears it at
        # the bottom, so by the time the single-config record below fires it is gone.
        nargs0 = dict(zip(getattr(self, "arg_names", ()) or (), args))
        previous = _CURRENT.get("id")
        if len(cfgs0) == 1 and id(self) not in _ROUND:
            _ROUND[id(self)] = list(cfgs0)
            _CURRENT["id"] = id(self)
        try:
            out = _orig_run(self, *args, **kwargs)
        finally:
            if _CURRENT.get("id") == id(self):
                _CURRENT["id"] = previous
        cfgs = getattr(self, "configs", None) or []
        if len(cfgs) == 1:
            mark = (id(self), tuple(sorted((k, str(v)) for k, v in kwargs.items()
                                           if isinstance(v, (int, float, str, bool)))))
            if mark not in _SINGLE_SEEN:
                _SINGLE_SEEN.add(mark)
                try:
                    _record_one(self, cfgs[0], kwargs, float("nan"), unmeasured=True,
                                nargs=nargs0)
                except Exception as exc:  # noqa: BLE001 -- must not perturb a real run
                    _record_failed(exc)
        return out

    Autotuner.run = run


def uninstall() -> None:
    global _orig_bench, _orig_compile, _orig_prune, _orig_run
    shutdown_precompile()
    if _orig_bench is not None:
        from triton.runtime.autotuner import Autotuner
        Autotuner._bench = _orig_bench
        _orig_bench = None
        if _orig_run is not None:
            Autotuner.run = _orig_run
            _orig_run = None
        _SINGLE_SEEN.clear()
        _LAUNCHED.clear()
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


def _rank(pairs):
    """(config, ms) fastest first, with unmeasured (NaN) entries LAST.

    `sorted` on a NaN key neither raises nor orders: comparisons against NaN are all False, so the
    result depends on input order and an unmeasured entry can end up at the head, where
    `store_ranked_configs` reads it as the winner.
    """
    return sorted(pairs, key=lambda cm: (cm[1] != cm[1], cm[1]))


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
            ranked = _rank(ent.values())
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
            a = agg.setdefault(op, {"grid": {}, "entries": {}})
            # UNION the grids, do not take the first shard's. Shards that split by SHAPE all carry
            # the same full grid, so taking one was harmless. Shards that split the CONFIG SET
            # carry DIFFERENT slices, and then the first shard's slice hashes to something no later
            # full-grid run reproduces -- ``store_ranked_configs`` sees a changed
            # ``config_space_hash`` and RESETS every entry, discarding the whole build. Which shard
            # supplied it also depended on file order, so the hash was not even deterministic.
            for cd in slot.get("grid", []):
                a["grid"].setdefault(_sig_from_dict(cd), cd)
            for bk, lst in slot.get("entries", {}).items():
                ent = a["entries"].setdefault(bk, {})
                for cd in lst:
                    sig = _sig_from_dict(cd)
                    ms = float(cd.get("ms", float("inf")))
                    prev = ent.get(sig)
                    # `ms < prev[1]` is False when either side is NaN, so an unmeasured entry
                    # never displaces a measured one -- but a measured one must displace it.
                    if prev is None or ms < prev[1] or prev[1] != prev[1]:
                        ent[sig] = (cd, ms)
    written = []
    for op, a in sorted(agg.items()):
        grid = list(a["grid"].values())   # dedup by signature; config_space_hash sorts them
        if not grid:
            continue
        csh = config_space_hash(grid)
        for bk, ent in sorted(a["entries"].items()):
            dtype, bucket = bk.split("|", 1)
            ranked = _rank(ent.values())
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
