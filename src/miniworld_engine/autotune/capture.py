"""Generic autotune-cache BUILDER via Triton-autotuner instrumentation.

Instead of hand-replicating each kernel's launch in a per-kernel builder (fragile, and how the
int32-offset bug slipped in), this patches ``triton.runtime.autotuner.Autotuner._bench`` to
record every (config -> measured ms) as it is benchmarked during a REAL module forward/backward
run. The op behind a live autotuner is recovered by identity of the config list it was handed
(:func:`autotune.configs.op_of`) -- the only back-reference left now that the prune objects that
used to carry the name are gone.

Usage (on the target GPU, with the full grid unlocked so every config is benched):

    from miniworld_engine import settings
    from miniworld_engine.autotune import capture
    settings.configure(run_autotune=True)   # was MINIWORLD_RUN_AUTOTUNE=1
    capture.install()
    ... run each wired module fwd+bwd across representative shapes ...
    capture.flush(top_k=5)     # writes <cache-root>/autotune/<op>/<gpu>.json for every op seen

Any wired kernel that fires during the run is captured automatically — no per-kernel code. The
cross-check that validated this capture path was against two hand-built pilot caches
(`transition_split_fwd` / `trimul_bidir_front`, names since retired -- see
docs/kernels/rename-map.tsv); the script that built them is gone, the cross-check is history. Config choice is performance-only, so this never affects numerics.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from miniworld_engine.autotune.cache import _scheme_stale as _scheme_stale
from miniworld_engine.autotune.cache import _sig_from_dict as _sig_from_dict
from miniworld_engine.autotune.cache import (
    as_cfg_dict,
    config_space_hash,
    config_to_dict,
    gpu_key,
    op_identity,
    store_ranked_configs,
)
from miniworld_engine.autotune.cache import bucket_of_autotuner as _bucket_of
from miniworld_engine.autotune.cache import dtype_of_args as _dtype_of

# op -> {"grid": [configs] | None, "entries": {(dtype, bucket): {sig: (config, ms)}}}
_CAPTURE: dict = {}
#: op -> configs this card could not run. They score +inf and are correctly not stored, but the
#: unit reported only what survived -- "grid=864 ... 527 configs" and nothing about the other 337,
#: which is a 39% hole in the searched space with no line anywhere saying so. Reconstructing it
#: from the shard is possible and I got it wrong doing exactly that: `prune_configs` returns the
#: full list, so nothing is pruned; triton itself returns [inf, inf, inf] for a config it cannot
#: launch (OutOfResources: shared memory, e.g. 514,048 bytes required against an A6000's 101,376),
#: without raising, so no exception reaches this module either. The count belongs in the report.
_UNUSABLE: dict = {}
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
#: autotuner id -> (configs, round id) of the round it is about to time, set by the patched
#: prune_configs. The round id is triton's own autotune key (see :func:`_round_id`): one autotuner
#: tunes once per key, and two keys of one kernel compile to different cache entries.
#: Keyed per autotuner because rounds interleave: a module fires several kernels, and a single
#: "pending" slot loses every round but the last (which is how an earlier version silently
#: pre-compiled only the first round of a whole build).
_ROUND: dict = {}
#: id of the autotuner whose _bench is currently on the stack, so the compile hook knows which
#: armed round the compile it is servicing belongs to.
_CURRENT: dict = {}
#: autotuner id -> configs of its round still to be timed. Separate from _ROUND, which the first
#: compile of the round pops: this one has to survive to the LAST config, because that is when
#: the card's bench lock is released.
_ROUND_LEFT: dict = {}

#: The open, flock-held file for this card's bench lock, plus what it cost. See
#: :func:`_bench_lock_acquire`.
_BENCH_LOCK: dict = {"fh": None, "waited": 0.0, "held": 0.0, "since": 0.0, "rounds": 0}

#: Time inside ``Autotuner._bench`` -- do_bench's launches AND the synchronize that waits for the
#: device. Reported on its own because inferring it cost a wrong answer: the only bench-shaped
#: number the build printed was ``fn.run``, and the first compile of a round happens INSIDE
#: fn.run, so fn.run was carrying the whole precompile pool. Read as "bench", it said the GPU was
#: 1.2% of the build; measured here it is 20%.
_BENCH_T: dict = {"calls": 0, "seconds": 0.0}

#: Where a compile() call actually spends its time. The guard forks a child, waits for it, and then
#: recompiles in-process expecting a cache hit -- three costs that look like one. They have
#: different fixes: fork is proportional to the parent's RSS, the wait is quantised by the poll
#: interval, and the in-process call is a lookup in ~/.triton/cache, which currently holds 88k+
#: entries. Measured apart so the fix targets the real one.
_COMPILE_T: dict = {"calls": 0, "fork_s": 0.0, "wait_s": 0.0, "parent_s": 0.0, "polls": 0,
                    "forkless": 0, "forkless_s": 0.0, "known_bad": 0}

#: First-launch cost per config, split. warm_s minus compile() left ~500s unexplained on a unit
#: whose kernels ran 2.2s of device time; these are the three things a first launch does that a
#: second one does not -- load the cubin into the context, build the launcher stub, and run the
#: autotuner's pre_hook (which zeroes reset_to_zero tensors).
_LAUNCH_T: dict = {"init_handles": 0, "init_s": 0.0,
                   "prehook_calls": 0, "prehook_s": 0.0, "run_calls": 0, "run_s": 0.0,
                   # ``JITFunction.compile`` on the parent's FIRST launch of each config. The
                   # precompile round runs in forked children, so the parent's kernel_cache is
                   # empty and it re-enters compile() once per config. That is a disk-cache HIT
                   # (no LLVM), but a hit still deserialises the metadata json + every IR level
                   # (.ttir/.ttgir/.llir/.ptx/.cubin) -- ~6 reads per config off the cache dir.
                   # ckinit_s is exactly those reads.
                   "ckinit_calls": 0, "ckinit_s": 0.0}

#: (device_ms, config kwargs) for every probed config. The budget knows a config is slow; it does
_CURRENT_CFG: dict = {}

def _sig_line(cfg: dict) -> str:
    return ",".join(f"{k}={cfg[k]}" for k in sorted(cfg))


def load_compile_state(shard_path) -> int:
    """Load the compiles this unit already settled. Returns how many replay.

    Kept after the bench watchdog and then the bench budget were removed: it no longer serves
    kill-and-restart or probe replay, it serves any restart. A unit that dies to an OOM, a node
    failure, or a job time limit resumes without recompiling what it had already settled.
    """
    from pathlib import Path as _P

    stem = str(shard_path).removesuffix(".json")
    cf = _P(stem + ".compiled")
    _COMPILED_FILE.append(cf)
    if cf.exists():
        for raw in cf.read_text().splitlines():
            ln = raw.strip()
            if not ln:
                continue
            if ln[0] in "+-" and ln.count("\t") == 2:  # "<+|-><kernel>\t<round>\t<cfg sig>"
                (_COMPILE_OK if ln[0] == "+" else _COMPILE_BAD).add(ln[1:])
            # A row from an older layout -- a bare sig, or one that names no round -- cannot
            # answer either question under the current key, so it is dropped. That costs one
            # round of recompiling on the first restart of a unit whose `.compiled` predates the
            # layout; every later restart reads rows it can use.
    return len(_COMPILE_OK) + len(_COMPILE_BAD)


_COMPILED_FILE: list = []

#: "<kernel>\t<round>\t<cfg sig>" for compiles the precompile POOL already settled, split by
#: outcome. Two questions are answered off this one record, with the SAME key:
#:
#:   * should the ROUND recompile this? -- avoids re-submitting a warm round to a fresh spawn
#:     pool on every restart, 446 s a time, which is what turned 87 legitimate kills into 7799 s.
#:   * must the SERIAL pass fork a child to find out? -- a config the pool compiled cannot be a
#:     compile monster, that is exactly what the pool just proved under the same SIGKILL budget,
#:     so re-proving it costs a fork + a pipe + a poll loop (179 ms measured) to protect a
#:     triton.compile that is a warm on-disk cache hit (3.5 ms). A config the pool FAILED needs no
#:     child either: the answer is known and the bench scores it +inf.
#:
#: EVERY component of the key is there because leaving it out cost a measured build:
#:
#:   * <kernel> -- rounds interleave across the kernels of one unit (see `prune_configs`), so a
#:     second kernel reusing the first one's tile axes produced identical sigs, its round was
#:     skipped as "already compiled", and every config fell through to the fork the pool was
#:     supposed to avoid: 864 forks, 4166 s, on a unit that should have spent ~400 s.
#:   * <round> -- one kernel is tuned once per autotune key, and two keys of the same kernel
#:     compile to DIFFERENT cache entries (different signature, constants, specialisation). Keyed
#:     without it, the second key's round was skipped, the settled-set told the serial pass the
#:     configs were warm, and the parent then really compiled all 14996 of them one at a time on
#:     ONE core while fifteen pool workers sat idle. Measured on the A6000 rebuild: 1084 ms per
#:     "warm hit" against 3.5 ms on single-key units, and 15.8 of the build's 16.7 serial CPU-hours
#:     in the three two-key units -- 9.0 h of them in adaln_gemm_gate-L256 alone.
_COMPILE_OK: set = set()
_COMPILE_BAD: set = set()


def _round_id(autotuner, kwargs) -> str:
    """Triton's own autotune key, stringified. This is what makes two rounds of one kernel two
    rounds: ``Autotuner.run`` tunes once per key, and each key reaches ``compile`` with its own
    signature and specialisation. Reproduced from triton's ``run`` rather than intercepted,
    because ``prune_configs`` -- the only hook that fires once per round -- is not handed it."""
    try:
        all_args = {**(getattr(autotuner, "nargs", None) or {}), **kwargs}
        names = set(getattr(autotuner, "arg_names", ()) or ())
        args = {k: v for k, v in all_args.items() if k in names}
        # An autotune key is normally a scalar (a length, a flag). Anything else is named by
        # type: str() on a tensor would serialise the tensor into a dict key on every round.
        parts = [str(args[k]) if isinstance(args[k], (int, float, str, bool))
                 else type(args[k]).__name__
                 for k in (getattr(autotuner, "keys", ()) or ()) if k in args]
        parts += [str(a.dtype) for a in args.values() if hasattr(a, "dtype")]
        return "|".join(parts)
    except Exception:   # an identity, not a correctness gate: worst case two rounds share one
        return ""


def _settled(kernel: str, rnd: str, sig: str) -> bool:
    key = f"{kernel}\t{rnd}\t{sig}"
    return key in _COMPILE_OK or key in _COMPILE_BAD


def _mark_settled(keys_ok) -> None:
    """Record compile outcomes so the serial pass can skip the fork. ``keys_ok`` yields full
    ``"<kernel>\t<round>\t<cfg sig>"`` keys. Appended to the ``.compiled`` file, tagged, so a
    restart inherits them."""
    rows = []
    for key, ok in keys_ok:
        target = _COMPILE_OK if ok else _COMPILE_BAD
        if key in target:
            continue
        target.add(key)
        rows.append(("+" if ok else "-") + key)
    if rows and _COMPILED_FILE:
        try:
            with _COMPILED_FILE[0].open("a") as fh:
                fh.write("".join(r + "\n" for r in rows))
        except OSError:
            pass


def _bench_lock_acquire() -> None:
    """Take this card's bench lock, if the build gave one (``MINIWORLD_BENCH_LOCK``).

    A build can put more than one unit on a card so that one compiles while the other measures --
    compile was 72% of the A6000 rebuild and bench 20%, and today neither overlaps the other. What
    two units on one card must NOT do is measure at the same time: two kernels sharing the SMs
    both read slower, by an amount that drifts over the round, and that changes which config wins.

    Taken once per ROUND, not once per config. 542,146 configs were timed in that rebuild; a lock
    operation apiece would cost more than the contention it prevents. Held across the round's
    whole sweep and released when its last config has been timed.

    Blocking on purpose. flock is released by the kernel when the holder dies, so a unit that is
    killed mid-round cannot strand the card.
    """
    import os
    import time

    if _BENCH_LOCK["fh"] is not None:      # already ours; rounds do not nest, but be safe
        return
    path = os.environ.get("MINIWORLD_BENCH_LOCK")
    if not path:
        return
    import fcntl

    started = time.monotonic()
    try:
        fh = open(path, "w")   # noqa: SIM115 -- held until the round ends, closed by the release
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    except OSError:            # no lock is worse than a lock, but far better than no build
        return
    _BENCH_LOCK["fh"] = fh
    _BENCH_LOCK["waited"] += time.monotonic() - started
    _BENCH_LOCK["since"] = time.monotonic()
    _BENCH_LOCK["rounds"] += 1


def _bench_lock_release() -> None:
    import time

    fh = _BENCH_LOCK["fh"]
    if fh is None:
        return
    _BENCH_LOCK["fh"] = None
    _BENCH_LOCK["held"] += time.monotonic() - _BENCH_LOCK["since"]
    try:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()
    except OSError:
        pass


def _cfg_sig(config) -> str:
    d = dict(config.kwargs)
    d["num_warps"] = config.num_warps
    d["num_stages"] = config.num_stages
    return _sig_line(d)


#: :func:`_install_launch_probes` ran. A module global, matching ``_REC_INSTALLED`` -- the flag
#: used to be stamped onto ``CompiledKernel`` itself, which typed as an unresolved attribute on a
#: third-party class for no gain over a local one.
_PROBES_INSTALLED = False


def _install_launch_probes() -> None:
    """Time the per-config first-launch path. Idempotent; build-only."""
    import time

    from triton.compiler.compiler import CompiledKernel
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction

    global _PROBES_INSTALLED
    if _PROBES_INSTALLED:
        return
    _PROBES_INSTALLED = True

    orig_init = CompiledKernel._init_handles

    def init_handles(self):
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

    def run(self, *a, **k):
        t = time.monotonic()
        try:
            return orig_run(self, *a, **k)
        finally:
            _LAUNCH_T["run_calls"] += 1
            _LAUNCH_T["run_s"] += time.monotonic() - t

    JITFunction.run = run

    # NOTE: ``JITFunction.compile`` is an INSTANCE attribute (``self.compile = compile`` in
    # JITFunction.__init__), so it cannot be hooked on the class. CompiledKernel.__init__ is the
    # class-level chokepoint every compile path goes through, and it is the part that deserialises
    # the metadata json + every IR level (.ttir/.ttgir/.llir/.ptx/.cubin) off the cache dir.
    orig_ckinit = CompiledKernel.__init__

    def ckinit(self, *a, **k):
        t = time.monotonic()
        try:
            return orig_ckinit(self, *a, **k)
        finally:
            _LAUNCH_T["ckinit_calls"] += 1
            _LAUNCH_T["ckinit_s"] += time.monotonic() - t

    CompiledKernel.__init__ = ckinit

    # ``pre_hook`` is an INSTANCE attribute as well -- ``self.pre_hook = ...`` runs on every path
    # through ``Autotuner.__init__``, and the class carries no default. The previous
    # ``Autotuner.pre_hook = pre_hook`` was dead twice over: ``hasattr(Autotuner, "pre_hook")`` is
    # False so the guard never opened, and an instance would have shadowed the class attribute
    # anyway. Wrap per instance on first ``run`` instead -- that also catches autotuners built
    # before this installer ran, which is most of them (the decorators fire at kernel import).
    _pre_wrapped: set[int] = set()
    orig_at_run = Autotuner.run

    def at_run(self, *a, **k):
        if id(self) not in _pre_wrapped:
            _pre_wrapped.add(id(self))
            inner = self.pre_hook

            def pre_hook(*ha, **hk):
                t = time.monotonic()
                try:
                    return inner(*ha, **hk)
                finally:
                    _LAUNCH_T["prehook_calls"] += 1
                    _LAUNCH_T["prehook_s"] += time.monotonic() - t

            self.pre_hook = pre_hook
        return orig_at_run(self, *a, **k)

    Autotuner.run = at_run


def precompile_summary() -> str:
    """One line on what the pre-compile actually did.

    A pool whose workers all die on arrival is indistinguishable from no pool at all -- the build
    just stays slow. Reporting the counts makes the difference visible instead of inferred.
    """
    p = _PRECOMPILE
    budget = ""
    c = _COMPILE_T
    if c["calls"] or c["forkless"] or c["known_bad"]:
        budget += (f"\n  [compile-guard] forkless {c['forkless']} x -> {c['forkless_s']:.0f}s"
                   f" (pool-settled) | known-bad {c['known_bad']} x -> 0s"
                   f"\n  [compile-guard] {c['calls']} forked compile() calls"
                   f" | fork {c['fork_s']:.0f}s + wait {c['wait_s']:.0f}s"
                   f" ({c['polls']} polls x 50ms) + parent-recompile {c['parent_s']:.0f}s"
                   f" = {c['fork_s'] + c['wait_s'] + c['parent_s']:.0f}s")
    b = _BENCH_T
    if b["calls"]:
        # `fn.run` is NOT this number: a round's first compile happens inside it. Print both, and
        # print what the pool did apart from either, so no one has to infer the split again.
        pool_and_guard = p["seconds"] + c["forkless_s"] + c["fork_s"] + c["wait_s"] + c["parent_s"]
        budget += (f"\n  [bench] {b['calls']} configs timed {b['seconds']:.0f}s"
                   f" | compile {pool_and_guard:.0f}s (pool {p['seconds']:.0f}s"
                   f" + guard {pool_and_guard - p['seconds']:.0f}s)")
    lk = _BENCH_LOCK
    if lk["rounds"]:
        budget += (f"\n  [bench-lock] {lk['rounds']} rounds | held {lk['held']:.0f}s"
                   f" | waited on the other unit {lk['waited']:.0f}s")
    lt = _LAUNCH_T
    if lt["run_calls"]:
        budget += (f"\n  [first-launch] fn.run {lt['run_calls']} calls {lt['run_s']:.0f}s"
                   f" | CompiledKernel.__init__ {lt['ckinit_calls']} x -> {lt['ckinit_s']:.0f}s"
                   f" | init_handles {lt['init_handles']} x -> {lt['init_s']:.0f}s"
                   f" | pre_hook {lt['prehook_calls']} x -> {lt['prehook_s']:.0f}s")
    return (f"  [precompile] jobs={_compile_jobs()} rounds={p['rounds']} configs={p['configs']} "
            f"-> compiled={p['compiled']} failed={p['failed']} in {p['seconds']:.0f}s{budget}")


def _compile_jobs() -> int:
    """Workers for pre-compilation: ``settings.compile_jobs``, else one per usable core (cap 32)."""
    import os

    from miniworld_engine import settings

    want = settings.current().compile_jobs
    if want is not None:
        return max(1, int(want))
    try:
        cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = os.cpu_count() or 1
    return max(1, min(32, cores))


#: (module, fn) -> JITFunction, memoized per worker process. See _resolve_jit.
_JIT_CACHE: dict = {}


def _resolve_jit(module_path: str, fn_name: str):
    """The JITFunction behind ``module_path.fn_name``, imported at most once per process.

    The module attribute is whatever the decorators left there -- for an autotuned kernel that is
    an Autotuner (or a Heuristics) wrapping the JITFunction, not the JITFunction itself. ASTSource
    needs the inner one; triton unwraps the same way in ``Autotuner.check_disk_cache``.
    """
    hit = _JIT_CACHE.get((module_path, fn_name))
    if hit is not None:
        return hit
    import importlib

    from triton.runtime.jit import JITFunction

    fn = getattr(importlib.import_module(module_path), fn_name)
    while not isinstance(fn, JITFunction):
        fn = fn.fn
    _JIT_CACHE[(module_path, fn_name)] = fn
    return fn


def _compile_payload(payload, prefetched=None) -> None:
    """The compile itself, with no process management. Raises on failure.

    Also records how much shared memory the compiler ended up needing, when the payload carries a
    config signature and ``MINIWORLD_SMEM_LOG`` names a file. That number is the reason a config
    later scores +inf, and until now nothing anywhere held it: triton catches its own
    `OutOfResources` inside `Autotuner._bench` and returns `[inf, inf, inf]`, so the wrapper above
    it cannot tell shared-memory overflow from a register-spill kill from a genuine slow config.
    Everything arrived as one undifferentiated "inf", which is no basis for deciding what not to
    compile next time.

    It is written from the CHILD, appended a line at a time, because the pipe back to the parent
    carries exactly one byte per config and the stall detector counts those bytes to know which
    config hung. Widening that protocol to carry a number would put a measurement inside the one
    mechanism that stops a compile from running forever.
    """
    import os

    module_path, fn_name, signature, constants, attrs, target, options = payload[:7]
    sig = payload[7] if len(payload) > 7 else None
    from triton.compiler.compiler import ASTSource, compile

    fn = prefetched if prefetched is not None else _resolve_jit(module_path, fn_name)
    src = ASTSource(fn=fn, signature=signature, attrs=attrs)
    src.constants = constants   # already keyed by arg-index tuple, as ASTSource stores it
    kernel = compile(src, target=target, options=options)
    log = os.environ.get("MINIWORLD_SMEM_LOG")
    if sig and log:
        shared = getattr(getattr(kernel, "metadata", None), "shared", None)
        if shared is not None:
            try:
                with open(log, "a") as fh:
                    fh.write(f"{fn_name}\t{sig}\t{int(shared)}\n")
            except OSError:
                pass


def _worker_compile(chunk: list) -> list:
    """Compile a CHUNK of configs in ONE forked child, returning one row per config.

    Thin wrapper: ``_compile_chunk`` does the work and returns ``(ok, seconds)``, so the retry can
    recurse without the accounting tuple being wrapped a second time. Must stay importable at
    module level (spawn).
    """
    rows = _compile_chunk(chunk)
    return [(ok, 0.0, seconds, 0.0) for ok, seconds in rows]


def _compile_chunk(chunk: list) -> list:
    """Compile a CHUNK of configs in ONE forked child; returns ``(ok, seconds)`` per config.

    Touches no CUDA: it rebuilds each ASTSource from plain data plus a re-imported JITFunction and
    compiles against the ``target`` the parent resolved. Results land in triton's on-disk cache;
    the return value is only for accounting.

    ONE fork per chunk, not per config. Forking a worker that holds torch + triton + the kernel
    module costs ~1 s of page-table work and teardown, against 7.7 ms for the compile it guards:
    measured on one unit, 1944 configs, warm cache -- 1966 worker-seconds with a fork per config
    against 15 worker-seconds with none. The guard still has to exist (make_llir and ptxas block
    in native code, where no Python signal can reach them, so a register-spill monster can only be
    stopped by killing the process that is in it), so it is kept and amortised instead of dropped:

      * the child reports each finished config through a pipe, one byte, pass or fail;
      * the parent's deadline resets on every byte, so the budget is still PER CONFIG -- a chunk
        of 32 gets 32 chances to make progress, not one 32x-longer rope;
      * on a stall the child is SIGKILLed and the config it stalled on is known exactly (it is the
        next one after the last byte), so that one is recorded failed and the REST of the chunk is
        retried rather than condemned with it.

    Determinism is what makes this safe to keep: a config that compiles only because it was never
    given its own budget would be kept by a pre-compiled build and dropped by a serial one, and
    the two would produce different caches from the same inputs.
    """
    import os
    import signal
    import time

    if not chunk:
        return []

    # Resolve (i.e. IMPORT) the kernel modules in the WORKER, before the fork. It used to be
    # imported inside the child, which threw the import away on every ``os._exit`` -- so the same
    # module was imported once PER CONFIG. Measured on layernorm_fwd_saveact_triton against a warm
    # cache: 3082 ms per config as-shipped, of which 2589 ms was that import and 36 ms was the
    # compile. The disk cache was being hit the whole time; the import was hiding it.
    prefetched = []
    for payload in chunk:
        try:
            prefetched.append(_resolve_jit(payload[0], payload[1]))
        except BaseException:  # noqa: PERF203 -- fall back to resolving in the child
            prefetched.append(None)

    rfd, wfd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(rfd)
        try:
            for payload, pre in zip(chunk, prefetched, strict=False):
                try:
                    _compile_payload(payload, pre)
                    os.write(wfd, b"\x01")
                except BaseException as exc:  # noqa: PERF203 -- one bad config, not a bad chunk
                    if os.environ.get("_MW_PRECOMPILE_FIRST") == "1":
                        import sys
                        print(f"  [precompile] child failed: {type(exc).__name__}: {exc}",
                              file=sys.stderr, flush=True)
                    os.write(wfd, b"\x00")
        finally:
            os._exit(0)

    os.close(wfd)
    os.set_blocking(rfd, False)
    results: list = []
    last = time.monotonic()
    killed_at = None
    while len(results) < len(chunk):
        try:
            # ONE byte per read, not the whole backlog: a batched read timestamps every config in
            # it identically, so a 0.6 s compile whose neighbour finished in the same poll window
            # is charged 0.3 s. Draining one at a time costs one syscall per config and charges
            # each the gap since the previous one, which is what it took.
            buf = os.read(rfd, 1)
        except BlockingIOError:
            buf = b""
        except OSError:
            buf = b""
        if buf:
            # Per-config seconds, not the chunk average: the whole compile budget question turns
            # on the TAIL. Measured on the A6000 rebuild, 13,875 configs of 869,844 ran the full
            # 60 s before being killed -- 1.6% of the configs and 54% of all compile CPU. Nothing
            # in the build said so, because the only number it kept was the chunk mean (0.83 s).
            now = time.monotonic()
            results.append((buf[0] == 1, now - last))
            last = now
            continue
        try:
            done, _status = os.waitpid(pid, os.WNOHANG)
        except OSError:
            break
        if done == pid:            # exited without reporting the rest: they did not compile
            break
        if time.monotonic() - last > _COMPILE_BUDGET_S:
            # Stalled on the config right after the last byte. Kill, record it failed, and hand
            # the untouched remainder back for a fresh chunk -- one monster must not condemn 31
            # configs that were never attempted.
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except OSError:
                pass
            killed_at = len(results)
            break
        # Adaptive backoff: most compiles here are warm on-disk cache hits finishing in single-digit
        # milliseconds, and a flat 50 ms tick would charge the full tick to every one of them.
        waited = time.monotonic() - last
        time.sleep(0.001 if waited < 0.05 else 0.01 if waited < 0.5 else 0.05)
    os.close(rfd)
    if killed_at is None:
        with contextlib.suppress(OSError):
            os.waitpid(pid, 0)
    else:
        # The config that stalled. It is charged the whole budget, which is what it cost, and
        # it is NAMED: these are 1.6% of a build's configs and 54% of its compile CPU, and
        # deciding whether they can be predicted (the way shared-memory overflow now is) needs
        # to know WHICH configs they were. Same file as the smem log, tagged, because the two
        # answer the same question -- why a config scored +inf -- for different reasons.
        results.append((False, float(_COMPILE_BUDGET_S)))
        _log_killed(chunk[killed_at])
        rest = chunk[len(results):]
        if rest:
            results.extend(_compile_chunk(rest))    # never attempted; give them their own child
    results.extend([(False, 0.0)] * (len(chunk) - len(results)))
    return results


def _log_killed(payload) -> None:
    """Name a config the compile budget killed, into ``MINIWORLD_SMEM_LOG`` with a ``!`` tag.

    A config that overflows shared memory and a config that spills registers both arrive at the
    bench as +inf, and only the first one leaves a shared-memory reading. Without this the second
    kind is invisible: the build reports a count and nothing that could be fitted or checked.
    """
    import os

    log = os.environ.get("MINIWORLD_SMEM_LOG")
    if not log:
        return
    try:
        fn_name, sig = payload[1], payload[7]
    except (IndexError, TypeError):
        return
    try:
        with open(log, "a") as fh:
            fh.write(f"!{fn_name}\t{sig}\t{int(_COMPILE_BUDGET_S)}\n")
    except OSError:
        pass


def _chunk_cost(config) -> int:
    """Rough compile cost of one config, used ONLY to balance chunks against each other.

    Big tiles and deep pipelines are what make triton's compiler slow, and they are also what
    runs past the SIGKILL budget -- so this orders a round expensive-first and the chunks are then
    dealt round-robin, which puts one monster in each chunk instead of thirty-two in one. An
    ordering that is wrong for some kernel costs nothing beyond the imbalance we already have.
    """
    volume = 1
    for value in config.kwargs.values():
        if isinstance(value, int) and value > 1:
            volume *= value
    return volume * max(getattr(config, "num_stages", 1) or 1, 1)


def _balanced_chunks(items, costs, n_chunks: int) -> list:
    """Deal ``items`` into ``n_chunks`` lists, most expensive first, round-robin.

    The measured failure this replaces: a round's chunks were contiguous slices of 32, one worker
    drew the slice holding thirteen configs that each ran the full 60 s budget before being
    killed, and fifteen workers waited ~800 s for it. Aggregate pool occupancy over the A6000
    rebuild was 50% -- 429 CPU-hours of workers waiting on a tail.
    """
    order = sorted(range(len(items)), key=lambda i: -costs[i])
    chunks: list = [[] for _ in range(max(1, n_chunks))]
    for rank, i in enumerate(order):
        chunks[rank % len(chunks)].append(items[i])
    return [c for c in chunks if c]


def _precompile_round(src, target, options, configs, rnd: str = "") -> None:
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
    import time

    started = time.monotonic()
    kname = getattr(getattr(src, "fn", None), "__name__", "?")
    todo = [c for c in configs if not _settled(kname, rnd, _cfg_sig(c))]
    skipped = len(configs) - len(todo)
    if not todo:
        print(f"  [precompile] {kname}: round skipped, all {len(configs)} configs already "
              f"compiled for THIS kernel and key on an earlier attempt", flush=True)
        return
    configs = todo
    try:
        import multiprocessing as mp

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
                             src.attrs, target, opts, _cfg_sig(config)))
        keys = [f"{kname}\t{rnd}\t{_cfg_sig(c)}" for c in configs]
        import os as _os
        _os.environ["_MW_PRECOMPILE_FIRST"] = "1" if _PRECOMPILE["rounds"] == 0 else "0"

        if _PRECOMPILE_POOL is None:
            _PRECOMPILE_POOL = mp.get_context("spawn").Pool(jobs)

        # Bound the wait so one register-spill monster cannot stall the build: stragglers keep
        # compiling into the shared on-disk cache, and the serial pass picks up whatever landed.
        budget = _COMPILE_BUDGET_S * (len(payloads) // jobs + 2)
        # One fork per CHUNK, not per config -- see _worker_compile. Sized so every worker gets
        # several chunks (stragglers stay cheap to rebalance) while the ~1 s fork is amortised
        # over enough configs to disappear: 1944 configs / 8 workers / 8 chunks each = 30.
        csize = max(1, min(32, len(payloads) // max(jobs * 8, 1) or 1))
        n_chunks = max(1, -(-len(payloads) // csize))
        costs = [_chunk_cost(c) for c in configs]
        # Chunk the KEYS alongside the payloads, on the same permutation: `done` comes back in
        # chunk order, and zipping it against the original config order is what would silently
        # record every outcome against the wrong config.
        pairs = _balanced_chunks(list(zip(payloads, keys, strict=True)), costs, n_chunks)
        chunks = [[pl for pl, _ in c] for c in pairs]
        sent = [k for c in pairs for _, k in c]
        results = _PRECOMPILE_POOL.map_async(_worker_compile, chunks, chunksize=1)
        try:
            done = [r for chunk_res in results.get(timeout=budget) for r in chunk_res]
        except Exception:  # timeout: proceed, the serial pass still works
            done = []
        ok = sum(1 for d in done if d and d[0])
        bad = sum(1 for d in done if not (d and d[0]))
        wt = sum(d[2] for d in done if d)
        # The tail is the whole story of a build's compile cost, and averaging it away is how it
        # stayed invisible: on the A6000 rebuild 1.6% of configs ran the full 60 s budget before
        # being killed and took 54% of all compile CPU, while the reported per-config number was
        # the chunk mean, 0.83 s. Print the shape so the budget can be chosen from evidence.
        secs = sorted(d[2] for d in done if d)
        if secs:
            def _q(f):
                return secs[min(len(secs) - 1, int(f * len(secs)))]
            over = [x for x in secs if x >= _COMPILE_BUDGET_S]
            print(f"  [precompile-time] p50 {_q(0.5):.2f}s p90 {_q(0.9):.2f}s p99 {_q(0.99):.2f}s"
                  f" max {secs[-1]:.0f}s | {len(over)} at the {_COMPILE_BUDGET_S}s budget"
                  f" = {100 * sum(over) / max(sum(secs), 1e-9):.0f}% of the round's CPU",
                  flush=True)
        print(f"  [precompile-worker] {len(chunks)} chunks x ~{csize} | child time"
              f" {wt:.0f} worker-s -> {wt / max(jobs, 1):.0f}s of the round", flush=True)
        # settled = attempted and answered, pass or fail. `done` is shorter than `sent` only
        # if the pool timed out; zip stops at the shorter one, so an unanswered config stays
        # unsettled and is retried, which is the intent.
        _mark_settled((k, bool(d and d[0])) for k, d in zip(sent, done, strict=False))
        _PRECOMPILE["compiled"] += ok
        _PRECOMPILE["failed"] += bad
        _PRECOMPILE["rounds"] += 1
        _PRECOMPILE["configs"] += len(payloads)
        # Printed per round, not just in the final summary: when this stalled before, nothing was
        # visible until the shard finished, so a dead pool and a slow one looked identical.
        wall = time.monotonic() - started
        # Occupancy is the number the chunk deal exists to move: 15 workers that finish in
        # sequence are not 15 workers. The A6000 rebuild averaged 50% across 461 rounds.
        print(f"  [precompile] round {_PRECOMPILE['rounds']}: {len(payloads)} configs "
              f"({skipped} already compiled) on {jobs} "
              f"workers -> {ok} compiled, {bad} failed, "
              f"{wall:.0f}s at {100 * wt / max(wall, 1e-9) / jobs:.0f}% occupancy", flush=True)
    except Exception:  # an optimisation; a build must never fail because of it
        pass
    _PRECOMPILE["seconds"] += time.monotonic() - started


def shutdown_precompile() -> None:
    global _PRECOMPILE_POOL
    _bench_lock_release()   # a round that died mid-sweep must not hold the card to process exit
    if _PRECOMPILE_POOL is not None:
        with contextlib.suppress(Exception):
            _PRECOMPILE_POOL.terminate()
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
    from triton.runtime.autotuner import Autotuner

    prev = Autotuner.run

    def run(self, *args, **kwargs):
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
    from miniworld_engine.autotune.configs import op_of
    return op_of(getattr(autotuner, "configs", None))


#: Recording errors, kept so a build cannot look healthy while capturing nothing.
_RECORD_ERRORS: dict[str, int] = {}


def _record_failed(exc: BaseException) -> None:
    key = f"{type(exc).__name__}: {exc}"
    first = key not in _RECORD_ERRORS
    _RECORD_ERRORS[key] = _RECORD_ERRORS.get(key, 0) + 1
    if first:
        import warnings

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
        _UNUSABLE[op] = _UNUSABLE.get(op, 0) + 1
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
    slot = _CAPTURE.setdefault(op, {"grid": None, "op_id": "", "entries": {}})
    if slot["grid"] is None:
        slot["grid"] = list(autotuner.configs)
        # The autotuner is the ONLY place the kernel source and the key list are both reachable;
        # the writer runs long after it is gone, so snapshot the identity here.
        slot["op_id"] = op_identity(autotuner)
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
    import os
    import signal
    import time

    import triton.compiler as _tc
    import triton.compiler.compiler as _tcc
    from triton.runtime.autotuner import Autotuner

    _orig_compile = _tcc.compile

    def _fork_compile(*args, **kwargs):
        # This is the only place fully-resolved compile arguments exist, so it is where a round's
        # fan-out has to start -- but only for the round currently being timed, and only once.
        src = args[0] if args else kwargs.get("src")
        current = _CURRENT.get("id")
        armed = _ROUND.pop(current, None) if current is not None else None
        if armed and src is not None and kwargs.get("target") is not None:
            _precompile_round(src, kwargs["target"], kwargs.get("options"), armed[0], armed[1])
        if _COMPILE_BUDGET_S <= 0:
            return _orig_compile(*args, **kwargs)
        # The pool already answered this one. Forking again costs 179 ms to re-derive a fact we
        # hold, and it is the single largest line item in a unit: 1944 forks = 348 s of the 366 s
        # this unit spent, against 6.8 s of actual triton.compile. Measured with cProfile
        # (posix.read 186 s in the child pipe, select.poll 79 s, time.sleep 47 s, fork 28 s).
        # Only inside a round, and only when _CURRENT_CFG names the config being compiled --
        # outside one, the sig would be stale and the shortcut would trust the wrong answer.
        settled = None
        if _CURRENT.get("id") is not None and _CURRENT_CFG and src is not None:
            name = getattr(getattr(src, "fn", None), "__name__", None)
            if name:
                settled = f"{name}\t{_CURRENT.get('round', '')}\t{_sig_line(_CURRENT_CFG)}"
        if settled is not None and settled in _COMPILE_BAD:
            _COMPILE_T["known_bad"] += 1
            raise RuntimeError("triton compile failed in the precompile pool; config skipped")
        if settled is not None and settled in _COMPILE_OK:
            _COMPILE_T["forkless"] += 1
            _t_skip = time.monotonic()
            try:
                return _orig_compile(*args, **kwargs)  # warm on-disk cache hit
            finally:
                _COMPILE_T["forkless_s"] += time.monotonic() - _t_skip
        _COMPILE_T["calls"] += 1
        _t_fork = time.monotonic()
        pid = os.fork()
        if pid == 0:  # child: compile into the on-disk cache, no torch/CUDA touched, then exit
            try:
                _orig_compile(*args, **kwargs)
                os._exit(0)
            except BaseException:  # any failure -> parent treats config as unusable
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
            time.sleep(0.001 if waited < 0.05 else 0.01 if waited < 0.5 else 0.05)
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
            if settled is not None:
                _mark_settled([(settled, False)])
            raise RuntimeError("triton compile failed in isolated child; config skipped")
        # The child surviving IS the proof the fast path needs. Recording it here (not only from
        # the pool) is what makes a RESTART cheap: a unit whose `.compiled` predates the outcome
        # tags, or whose round was skipped because every config was already settled, would
        # otherwise fork all over again on every attempt.
        if settled is not None:
            _mark_settled([(settled, True)])
        _t_parent = time.monotonic()
        try:
            return _orig_compile(*args, **kwargs)  # expected: ~/.triton/cache hit
        finally:
            _COMPILE_T["parent_s"] += time.monotonic() - _t_parent

    _install_launch_probes()

    # Deliberate monkeypatch of triton's module-level `compile`. `_fork_compile(*args, **kwargs)`
    # forwards everything, but a checker compares it against the concrete signature and calls the
    # rebind invalid -- which is the nature of a monkeypatch, not a defect in one.
    _tcc.compile = _fork_compile  # ty: ignore[invalid-assignment]
    # the name create_binder rebinds via `from ..compiler import compile`
    _tc.compile = _fork_compile  # ty: ignore[invalid-assignment]

    # The pruned config list is the round's work item; hand it to the compile hook, which is where
    # the fully-resolved compile arguments to clone from first appear.
    global _orig_prune
    _orig_prune = Autotuner.prune_configs

    def prune_configs(self, kwargs):
        pruned = _orig_prune(self, kwargs)
        # per autotuner AND per autotune key: rounds interleave across the kernels of a unit, and
        # one kernel is tuned once per key.
        _ROUND[id(self)] = (list(pruned), _round_id(self, kwargs))
        _ROUND_LEFT[id(self)] = len(pruned)
        # A round is exactly one prune_configs + one sweep of the pruned list for ONE autotune key,
        # so this is where the running best resets. Keeping it per autotuner rather than global:
        # rounds from different kernels interleave, and a fast elementwise kernel's best would
        # otherwise set an impossible budget for a GEMM.
        return pruned

    Autotuner.prune_configs = prune_configs

    _orig_bench = Autotuner._bench

    def _bench(self, *args, config, **meta):
        # Tell the compile hook which round it is servicing: the first config that actually
        # compiles triggers the fan-out for the whole round.
        previous = _CURRENT.get("id")
        previous_round = _CURRENT.get("round", "")
        _CURRENT["id"] = id(self)
        # The armed entry is popped by the first compile of the round, so read the id from
        # whichever of the two still holds it.
        armed = _ROUND.get(id(self))
        _CURRENT["round"] = armed[1] if armed else previous_round
        # One acquire per round, released below when its last config has been timed.
        if id(self) in _ROUND_LEFT:
            _bench_lock_acquire()
        _CURRENT_CFG.clear()
        _CURRENT_CFG.update(config.kwargs)
        _CURRENT_CFG["num_warps"] = config.num_warps
        _CURRENT_CFG["num_stages"] = config.num_stages
        _t_bench = time.monotonic()
        try:
            res = _orig_bench(self, *args, config=config, **meta)
        except Exception:  # a config that fails to compile/run simply loses
            # Match triton's own sentinel SHAPE, not just its value: do_bench(quantiles=...) hands
            # back [median, q20, q80], and triton returns [inf, inf, inf] for a config it could not
            # build. Returning a bare float mixes scalars and lists in the timings dict, and the
            # `min(timings, key=timings.get)` that picks the winner then dies with
            # "'<' not supported between instances of 'float' and 'list'" -- taking the whole shard
            # with it. Only shows up once at least one config fails, which is why it surfaced when
            # the compile budget started dropping configs.
            res = [float("inf")] * 3
        finally:
            _BENCH_T["calls"] += 1
            _BENCH_T["seconds"] += time.monotonic() - _t_bench
            _CURRENT["id"] = previous
            _CURRENT["round"] = previous_round
            if id(self) in _ROUND_LEFT:
                left = _ROUND_LEFT[id(self)] - 1
                if left <= 0:
                    del _ROUND_LEFT[id(self)]
                    _bench_lock_release()
                else:
                    _ROUND_LEFT[id(self)] = left
        med = _median(res)
        try:
            _record_one(self, config, meta, med)
        except Exception as exc:  # capture must never perturb a real bench
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
        nargs0 = dict(zip(getattr(self, "arg_names", ()) or (), args, strict=False))
        previous = _CURRENT.get("id")
        previous_round = _CURRENT.get("round", "")
        if len(cfgs0) == 1 and id(self) not in _ROUND:
            # `self.nargs` is set by _orig_run, which has not run yet -- hand _round_id the same
            # binding it will build.
            rnd = _round_id(self, {**nargs0, **kwargs})
            _ROUND[id(self)] = (list(cfgs0), rnd)
            _CURRENT["id"] = id(self)
            _CURRENT["round"] = rnd
        try:
            out = _orig_run(self, *args, **kwargs)
        finally:
            if _CURRENT.get("id") == id(self):
                _CURRENT["id"] = previous
                _CURRENT["round"] = previous_round
        cfgs = getattr(self, "configs", None) or []
        if len(cfgs) == 1:
            mark = (id(self), tuple(sorted((k, str(v)) for k, v in kwargs.items()
                                           if isinstance(v, (int, float, str, bool)))))
            if mark not in _SINGLE_SEEN:
                _SINGLE_SEEN.add(mark)
                try:
                    _record_one(self, cfgs[0], kwargs, float("nan"), unmeasured=True,
                                nargs=nargs0)
                except Exception as exc:  # must not perturb a real run
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
        import triton.compiler as _tc
        import triton.compiler.compiler as _tcc
        _tcc.compile = _orig_compile
        _tc.compile = _orig_compile
        _orig_compile = None


def reset() -> None:
    _CAPTURE.clear()
    _UNUSABLE.clear()


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
            fp = store_ranked_configs(op, gk, dtype, bucket, ranked, csh, top_k=top_k,
                                      op_id=slot.get("op_id", ""))
            written.append((op, dtype, bucket, len(ranked), str(fp)))
    return written


def dump_shard(path: str) -> int:
    """Serialize this process's captured timings to a standalone JSON SHARD file (NOT the
    in-repo cache). Parallel capture jobs each ``dump_shard`` to their OWN file; a single
    ``merge_shards`` writer folds them into the committed cache — so no env var and no
    concurrent writers ever touch the in-repo tree. Returns the number of ops dumped."""
    from miniworld_engine._atomic import write_json
    from miniworld_engine.autotune.cache import KEY_SCHEME

    # `_key_scheme`, leading underscore: every other top-level key in a shard is an OP NAME, and
    # `merge_shards` iterates them as such. Without this a shard carries no record of what its
    # bucket strings MEAN, so a merge after a scheme bump folds old and new keys into one file --
    # dead buckets at best, and at worst an old pair measurement landing on the bucket an atom
    # launch now reads (both wrote `shape_key=256`).
    out: dict = {"_key_scheme": KEY_SCHEME}
    for op, slot in _CAPTURE.items():
        grid = slot["grid"] or []
        entries = {f"{d}|{b}": [config_to_dict(c, ms) for c, ms in ent.values()]
                   for (d, b), ent in slot["entries"].items()}
        out[op] = {"grid": [config_to_dict(c) for c in grid], "entries": entries,
                   "op_id": slot.get("op_id", "")}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Atomic, like store_ranked_configs. A bare write_text truncates and then writes, so two
    # processes on the same shard interleave: the shorter write lands inside the longer one and
    # the tail of the first survives past its end -- "Extra data: line 1 column 359431". Two
    # writers happen whenever a job is cancelled mid-unit and its claim is reclaimed while the
    # old process is still running, which is exactly what a --reclaim restart does. The result
    # is unparseable, and merge_shards drops unparseable shards, so the unit's whole measurement
    # disappears from the cache with nothing said.
    write_json(p, out)
    return len(out)


#: shards merge_shards could not parse, reported by the caller rather than swallowed.
_MERGE_SKIPPED: list = []


def merge_shards(shard_paths, top_k: int = 5, gpu: str | None = None, only_ops=None) -> list:
    """Fold shard files (from ``dump_shard``) into the in-repo cache as the SOLE writer.

    Unions buckets across shards, keeping the fastest reading per config. ``only_ops`` (a set)
    restricts the write to those op names — so a targeted build never rewrites an op that already
    has a good cache. Returns ``(op, dtype|bucket, n_configs, path)`` rows for logging."""
    import json
    gk = gpu or gpu_key()
    _MERGE_SKIPPED.clear()
    agg: dict = {}
    for sp in shard_paths:
        try:
            d = json.loads(Path(sp).read_text())
        except Exception as exc:  # unreadable/partial shard
            # Never silent: a dropped shard is a whole unit's measurement missing from the cache,
            # and it used to look identical to a unit that was never run.
            _MERGE_SKIPPED.append((str(sp), f"{type(exc).__name__}: {exc}"))
            continue
        shard_scheme = d.get("_key_scheme")
        for op, slot in d.items():
            if op.startswith("_"):
                continue                      # metadata, not an op
            if only_ops is not None and op not in only_ops:
                continue
            if _scheme_stale(op, shard_scheme):
                # Same rule the reader applies to a cache file. A shard written before a bump
                # that re-based THIS op's keys describes buckets that no longer mean what they
                # say; merging it would put them back in the file the bump just cleaned.
                _MERGE_SKIPPED.append(
                    (f"{sp}::{op}", f"key scheme {shard_scheme} predates the bump that re-based "
                                    f"this op's buckets"))
                continue
            a = agg.setdefault(op, {"grid": {}, "entries": {}, "op_id": ""})
            a["op_id"] = a["op_id"] or (slot.get("op_id") or "")
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
            fp = store_ranked_configs(op, gk, dtype, bucket, list(ranked), csh, top_k=top_k,
                                      op_id=a.get("op_id", ""))
            written.append((op, bk, len(ranked), str(fp)))
    return written


def summary() -> str:
    lines = []
    for op, slot in sorted(_CAPTURE.items()):
        n_buckets = len(slot["entries"])
        n_grid = len(slot["grid"] or [])
        bad = _UNUSABLE.get(op, 0)
        # "unusable" is stated even when 0: a reader has to be able to tell "nothing was dropped"
        # from "dropping is not reported", and for a long time those looked identical.
        lines.append(f"  {op}: grid={n_grid} buckets={n_buckets} unusable={bad}"
                     + (f" ({bad * 100 // max(n_grid * n_buckets, 1)}% of the searched space "
                        f"could not run on this card)" if bad else ""))
        for (dtype, bucket), ent in sorted(slot["entries"].items()):
            best = min(ent.values(), key=lambda cm: cm[1]) if ent else None
            tag = f"{dtype}|{bucket}"
            if best:
                lines.append(f"    {tag}: {len(ent)} configs, best {best[1]:.4f}ms "
                             f"{as_cfg_dict(best[0])['kwargs']}")
    return "\n".join(lines) if lines else "  (nothing captured)"
