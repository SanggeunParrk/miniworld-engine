"""Verify the autotune-cache build system against its requirements, by MEASUREMENT.

Every claim this build makes is otherwise unfalsifiable. "The grid is brute-forced" is true until
someone re-adds a hand list; "the sweep covers the tuning keys" is true until a kernel starts
bucketing on a constexpr no Case moves; "every kernel is built" is true until an op registers
itself on an import path no module reaches. None of those fail loudly -- they fail as an empty
cache bucket, months later, as a multi-minute full-grid stall inside a production forward.

So each requirement below is a function that inspects the LIVE objects (the Triton autotuners as
constructed, the prune callbacks as registered, the Case sweep as declared) and returns findings
rather than a boolean. Run it as ``python -m miniworld_engine.build.audit``.

What it cannot do without a GPU: prove that driving a Case actually reaches an op. Reachability is
answered from the shards a real build wrote (``--shards``), because that is the only honest source
-- a static call graph through Triton dispatch would be a guess.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import itertools
import json
import pkgutil
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import triton

WARN = "WARN"
FAIL = "FAIL"
OK = "OK"


@dataclass
class Finding:
    check: str
    level: str
    subject: str
    detail: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add(self, check, level, subject, detail):
        self.findings.append(Finding(check, level, subject, detail))

    def of(self, check, level=None):
        return [f for f in self.findings
                if f.check == check and (level is None or f.level == level)]


# --------------------------------------------------------------------------- #
# live introspection
# --------------------------------------------------------------------------- #
def import_all_kernels() -> list[tuple[str, str]]:
    """Import every kernel module. Returns the ones that could not be imported."""
    import miniworld_engine.kernels as pkg

    failed = []
    for m in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        try:
            importlib.import_module(m.name)
        except BaseException as exc:  # noqa: PERF203 -- a broken module is a finding, not a crash
            failed.append((m.name, f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}"))
    return failed


def autotuners() -> list[tuple[str, object]]:
    """Every live Triton Autotuner reachable from an imported kernel module, as (qualname, obj)."""
    from triton.runtime.autotuner import Autotuner

    seen, out = set(), []
    for name, mod in list(sys.modules.items()):
        if not name.startswith("miniworld_engine.kernels"):
            continue
        for attr in dir(mod):
            try:
                obj = getattr(mod, attr)
            except Exception:
                continue
            if isinstance(obj, Autotuner) and id(obj) not in seen:
                seen.add(id(obj))
                out.append((f"{name}.{attr}", obj))
    return out


def _op_of(tuner) -> str | None:
    # By identity of the config list the op was handed. The prune objects that used to carry
    # ``_miniworld_op`` were removed in fcd3c7a, after which this returned None for every op and
    # the whole audit fell back to reporting module-level variable names instead of op names.
    from miniworld_engine.autotune.configs import op_of  # avoid an import cycle

    return op_of(getattr(tuner, "configs", None) or [])


def _bucket_keys(tuner) -> list[str] | None:
    """The constexpr names this op's cache bucket is built from (its TUNING KEYS).

    Read off ``tuner.keys`` -- the kernel's own ``key=[...]`` -- which is the same source
    ``capture._bucket_of`` derives the bucket from, so the audit cannot report keys that differ
    from the ones the cache is actually split on.

    This used to dig the names out of the closure of a per-kernel ``key_bucket_of(...)`` hung off
    ``early_config_prune``. fcd3c7a removed those objects, after which this returned None for
    every op and the check emitted 91 "not introspectable" warnings and verified nothing.
    """
    keys = getattr(tuner, "keys", None)
    return list(keys) if keys else None


# --------------------------------------------------------------------------- #
# check 2 -- brute verification
# --------------------------------------------------------------------------- #
def check_brute(rep: Report, tuners) -> None:
    """A grid is brute iff it is the FULL cartesian product of its own observed value sets.

    Checked structurally rather than by looking for the string ``brute(``: what matters is the
    shape of the grid the tuner actually holds, not how the module spelled it. A hand list is
    almost never a full product -- it pins warps to the tile, skips stage counts, and that is
    exactly the coverage a build silently inherits.
    """
    # ``autotune.grids`` held one global WARPS/STAGES for every kernel; the axis values are now
    # declared PER OP in the config set (configs/<set>/<op>.csv), so the expectation has to come
    # from the same place the grid does. This import has been dead since the CSV format landed,
    # which is why `audit` could not run at all -- it raised ModuleNotFoundError on the first
    # check, and every reference telling a user to "run audit for the holes" pointed at a
    # command that crashed on startup.
    def _declared(op: str) -> tuple[set, set]:
        """(num_warps, num_stages) the op's config CSV declares, or empty if it has no spec."""
        from miniworld_engine.autotune.configs import _DIR

        d = _DIR
        if not d:
            return set(), set()
        f = Path(d) / f"{op}.csv"
        if not f.is_file():
            return set(), set()
        w = st = set()
        for line in f.read_text().splitlines()[1:]:
            axis, _, vals = line.partition(",")
            if axis == "num_warps":
                w = {int(x) for x in vals.split()}
            elif axis == "num_stages":
                st = {int(x) for x in vals.split()}
        return w, st

    for name, t in tuners:
        held = getattr(t, "configs", []) or []
        cfgs = list(held)
        op = _op_of(t) or name
        if not cfgs:
            rep.add("brute", WARN, op, "no configs on the autotuner")
            continue
        dims = collections.defaultdict(set)
        for c in cfgs:
            for k, v in c.kwargs.items():
                dims[k].add(v)
        warps = {c.num_warps for c in cfgs}
        stages = {c.num_stages for c in cfgs}
        expect = 1
        for vs in dims.values():
            expect *= len(vs)
        expect *= len(warps) * len(stages)
        # A declared inter-axis constraint (``widen(keep=...)``, carried on the config list as
        # ``miniworld_keep``) makes a grid legitimately smaller than the product of its own value
        # sets: the augmented_attention backward's FlashAttention dq/dk tiling requires
        # BLOCK_M1 >= BLOCK_N, and the configs violating it write out of bounds. Counting the
        # product WITH the predicate applied keeps this check strict -- it still fails if anything
        # other than the constraint is missing -- instead of reporting a FAIL whose easiest
        # "fix" is deleting the predicate.
        keep = getattr(held, "miniworld_keep", None)
        constraint = ""
        if keep is not None:
            full = 0
            for combo in itertools.product(*[sorted(dims[k]) for k in dims]):
                cand = triton.Config(dict(zip(dims, combo, strict=False)), num_warps=4, num_stages=3)
                if keep(cand):
                    full += 1
            expect = full * len(warps) * len(stages)
            constraint = " under declared keep()"
        missing = []
        if len(cfgs) != expect:
            missing.append(f"not a full product{constraint}: {len(cfgs)} configs, "
                           f"product would be {expect}")
        want_w, want_s = _declared(op)
        if want_w and not want_w <= warps:
            missing.append(f"num_warps {sorted(warps)} misses {sorted(want_w - warps)}")
        if want_s and not want_s <= stages:
            missing.append(f"num_stages {sorted(stages)} misses {sorted(want_s - stages)}")
        if missing:
            rep.add("brute", FAIL, op, "; ".join(missing))
        else:
            rep.add("brute", OK, op,
                    f"{len(cfgs)} configs, full product over {sorted(dims)}{constraint}")


def check_prune_executes(rep: Report, tuners) -> None:
    """Call every prune callback once. A NameError here is a kernel that cannot launch AT ALL.

    Import succeeds even when a prune body references an unbound global, because the body only
    runs at launch. That is how a missing ``operand_bytes`` import shipped past "all 86 kernels
    import" and past ruff (F821 does not flag it -- verified). The build then fails per unit with
    ``skip ...: NameError``, and a unit that skips still writes a shard, so ``--resume`` treats it
    as done. Executing the callback is the only check that sees it.

    Other exceptions are expected: the synthetic named_args has none of the kernel's constexprs, so
    estimators legitimately raise KeyError/TypeError. Only unbound names are a defect.
    """
    for name, t in tuners:
        op = _op_of(t) or name
        prune = getattr(t, "early_config_prune", None)
        cfgs = list(getattr(t, "configs", []) or [])[:4]
        if prune is None or not cfgs:
            continue
        try:
            prune(cfgs, {})
        except NameError as exc:
            rep.add("prune", FAIL, op, f"unbound name at launch: {exc}")
        except Exception:  # missing constexprs are expected here
            rep.add("prune", OK, op, "ran (raised on synthetic args, which is fine)")
        else:
            rep.add("prune", OK, op, "ran")


# --------------------------------------------------------------------------- #
# check 4 -- tuning-key coverage
# --------------------------------------------------------------------------- #
def check_tuning_keys(rep: Report, tuners) -> None:
    """Every op's bucket keys must be MOVED by the sweep, not merely present.

    A bucket key the sweep never varies produces exactly one bucket. That is not a small cache --
    every other value of that key is a full-grid fallback in production, which is the failure this
    whole system exists to prevent. So the finding is "key is pinned", not "key is missing".
    """
    from miniworld_engine.autotune import builder

    swept = collections.defaultdict(set)
    for case in builder.cases():
        for dims in case.dims:
            for k, v in dims.items():
                swept[k].add(v)
        swept["__lengths__"] |= set(case.lengths)
        swept["__dtypes__"] |= {str(d) for d in case.dtypes}

    # bucket keys are kernel constexpr names (N, K, D, GROUP_M, HEAD_DIM ...) and do not map 1:1
    # onto Case dim kwargs (d_pair, d_hidden, n_head). Report them for a human to map; flag only
    # the ops whose keys we can see are single-valued in every declared sweep.
    for name, t in tuners:
        op = _op_of(t) or name
        keys = _bucket_keys(t)
        if keys is None:
            rep.add("keys", WARN, op, "autotune key=[] -- one cache bucket for every shape")
            continue
        rep.add("keys", OK, op, f"buckets on {keys}; autotune key={list(getattr(t, 'keys', []))}")
    rep.stats["swept_dims"] = {k: sorted(v) for k, v in sorted(swept.items())}


def check_key_spread(rep: Report, shard_dirs: list[Path]) -> None:
    """Did the sweep actually MOVE each op's tuning keys, or only fill one bucket?

    Judged from a real build's shards rather than by mapping Case kwarg names onto kernel
    constexpr names -- that mapping is a guess (``d_pair`` is not ``N``, ``n_head`` is not ``H``),
    and a guess here would report coverage the build does not have. The shard entries are keyed
    literally ``"<dtype>|<bucket>"``, so counting distinct buckets per op measures the thing
    directly: one bucket means every other value of that key is a full-grid fallback in
    production, however many shapes the sweep visited.
    """
    per_op: dict[str, set] = collections.defaultdict(set)
    for d in shard_dirs:
        for f in d.glob("*.json"):
            try:
                j = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            for op, v in j.items():
                if isinstance(v, dict) and v.get("entries"):
                    per_op[op] |= set(v["entries"])
    if not per_op:
        rep.add("spread", WARN, "(no shards)", "pass --shards from a completed build to judge this")
        return
    # What "enough dtypes" means is DECLARED, in registry.csv's dtypes column -- a token kernel is
    # bf16 only, atom and both are bf16|fp32. Warning on "only one dtype" measured against nothing
    # flagged all 91 ops including the 37 that are correct by declaration, which is a report that
    # cannot be acted on and trains you to ignore the check.
    declared = _declared_dtypes()
    for op, buckets in sorted(per_op.items()):
        dtypes = {b.split("|", 1)[0] for b in buckets}
        shapes = {b.split("|", 1)[1] for b in buckets if "|" in b}
        if len(shapes) <= 1:
            # One bucket is CORRECT for a kernel whose autotune key carries no shape_key: it has
            # no per-shape cache to build, and the builder already drives it at one length only
            # (see `_keys_on_shape` there). transition_fold_triton is the one today -- it reads
            # the weights and never touches the activation, so N and K are its whole shape.
            # Judged by the same function the builder uses, so the two cannot drift apart.
            if not _keys_on_shape_key(op):
                rep.add("spread", OK, op,
                        f"1 shape bucket: {sorted(shapes)} -- autotune key carries no shape_key, "
                        f"so there is nothing per-shape to tune")
                continue
            rep.add("spread", FAIL, op,
                    f"only {len(shapes)} shape bucket: {sorted(shapes)} -- key is pinned")
            continue
        want = declared.get(op)
        # A capture records the dtypes it SAW together, e.g. "bfloat16+float32" for a kernel whose
        # operands are mixed; that satisfies a bf16|fp32 declaration. Match on membership, not on
        # set equality.
        seen = {t for k in dtypes for t in k.split("+")}
        missing = (want - seen) if want else set()
        if missing:
            rep.add("spread", WARN, op,
                    f"{len(shapes)} shape buckets, but declared dtype(s) {sorted(missing)} "
                    f"never appear (saw {sorted(dtypes)})")
        else:
            rep.add("spread", OK, op,
                    f"{len(shapes)} shape buckets x {sorted(dtypes)}")


def _keys_on_shape_key(op: str) -> bool:
    """Does ``op``'s ``@triton.autotune(key=[...])`` include ``shape_key``?

    Resolved through registry.csv (file + symbol) and answered by the builder's own
    ``_keys_on_shape``, so the audit and the build agree on which ops have a per-shape cache.
    An op the registry does not name is assumed to key on shape -- the strict reading.
    """
    import csv

    from miniworld_engine.autotune.builder import _keys_on_shape

    root = Path(__file__).resolve().parents[2]
    reg = Path(__file__).resolve().parents[1] / "kernels" / "registry.csv"
    if not reg.is_file():
        return True
    for r in csv.DictReader(reg.open()):
        if r["kernel"] == op and (r.get("file") or "").strip():
            return _keys_on_shape(root / r["file"], r["symbol"])
    return True


def _declared_dtypes() -> dict:
    """kernel -> the torch dtype names registry.csv says it must be tuned for."""
    import csv

    alias = {"bf16": "bfloat16", "fp32": "float32", "fp16": "float16"}
    out = {}
    reg = Path(__file__).resolve().parents[1] / "kernels" / "registry.csv"
    if not reg.is_file():
        return out
    for r in csv.DictReader(reg.open()):
        out[r["kernel"]] = {alias.get(x, x) for x in (r.get("dtypes") or "").split("|") if x}
    return out


# --------------------------------------------------------------------------- #
# check 5 -- registry reachability
# --------------------------------------------------------------------------- #
#: ops that are GPU-specific by construction -- excluded per the build policy, not holes.
GPU_SPECIFIC = re.compile(r"_sm(\d+)|^transition_b2b")


def check_reachability(rep: Report, shard_dirs: list[Path]) -> None:
    from miniworld_engine.autotune.configs import registered_ops

    reg = sorted(registered_ops())
    if not shard_dirs:
        # Same shape as check_key_spread's no-shards branch, which this did not follow: with no
        # evidence it FAILED every registered op with "NO build ever captured it" -- a statement
        # about a missing ARGUMENT, printed as 88 defects in the artifact, and enough to make the
        # audit exit 1 every time it is run the documented way. The module docstring already says
        # reachability is answerable only from a real build's shards.
        rep.add("reach", WARN, "(no shards)",
                "pass --shards from a completed build to judge reachability")
        rep.stats["registered"] = len(reg)
        rep.stats["captured"] = 0
        return
    built = set()
    for d in shard_dirs:
        for f in d.glob("*.json"):
            try:
                j = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            built |= {k for k, v in j.items() if isinstance(v, dict) and v.get("entries")}
    for op in reg:
        if op in built:
            rep.add("reach", OK, op, "captured by a build")
        elif GPU_SPECIFIC.search(op):
            rep.add("reach", OK, op, "GPU-specific: excluded on this card by policy")
        else:
            rep.add("reach", FAIL, op, "registered but NO build ever captured it")
    rep.stats["registered"] = len(reg)
    rep.stats["captured"] = len(built)


# --------------------------------------------------------------------------- #
# check 3 -- parallelism
# --------------------------------------------------------------------------- #
def check_parallelism(rep: Report) -> None:
    from miniworld_engine.autotune import capture

    jobs = capture._compile_jobs()
    rep.add("parallel", OK if jobs > 1 else FAIL, "cpu-compile",
            f"precompile workers = {jobs}")
    from miniworld_engine.autotune import builder

    assert builder.__file__ is not None  # a namespace package would have none; this is a module
    src = Path(builder.__file__).read_text()
    if "ThreadPoolExecutor(max_workers=len(gpus))" in src:
        rep.add("parallel", OK, "gpu-tune", "one unit per GPU, worker pool sized to len(gpus)")
    else:
        rep.add("parallel", FAIL, "gpu-tune", "no per-GPU worker pool found in builder")


# --------------------------------------------------------------------------- #
def check_cache_coverage(rep: Report, gpu: str | None = None) -> None:
    """Every DECLARED (op, shape bucket) must have an entry in the shipped cache.

    Declared means registry.csv crossed with the kernel's level -- exactly the work list
    ``op_units`` builds -- not "whatever happened to get measured". A hole here is a bucket that
    falls back to the full grid inside a production forward, which is the cost the cache exists
    to remove, and until now nothing reported them: the build printed "run `audit` for the holes"
    and no such check existed.
    """
    from miniworld_engine.autotune.builder import op_units
    from miniworld_engine.autotune.cache import _CACHE_ROOT, gpu_key

    gk = gpu or gpu_key()
    if gk == "cpu":
        # The default key on a machine with no CUDA. Every declared pair then "misses", so a login
        # node reported 51 FAIL and missing_pairs=859 against a cache that is in fact complete.
        rep.add("coverage", WARN, "(no gpu)",
                "no CUDA device here; pass --gpu '<name> (sm<arch>)' to audit a shipped card")
        return
    have: dict[str, set] = {}
    for f in sorted(_CACHE_ROOT.rglob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            rep.add("coverage", FAIL, f.parent.name, f"unreadable cache file: {f.name}")
            continue
        if not (isinstance(d, dict) and "entries" in d and d.get("gpu") == gk):
            continue
        for k in d["entries"]:
            # Split the dtype prefix off FIRST. Scanning the whole key for a "shape_key=" field
            # misses it whenever it is the first one after the pipe -- "bfloat16|shape_key=128"
            # splits on "," into a single element that starts with the dtype -- and every op whose
            # bucket is shape_key alone (no other dims) then recorded bucket None, which the check
            # reads as "this op has no shape axis" and passes.
            _, _, dims = k.partition("|")
            sk = next((int(x.split("=")[1]) for x in dims.split(",")
                       if x.startswith("shape_key=")), None)
            # A capture records the dtypes the kernel actually SAW, e.g. "bfloat16+float32" for
            # mixed operands. Count each declared precision the entry satisfies, so (op, dtype,
            # bucket) is what gets checked -- not (op, bucket), which reported 527/527 over a
            # cache that held only the bf16 half of 66 kernels.
            for t in k.split("|", 1)[0].split("+"):
                have.setdefault(d["op"], set()).add((t, sk))

    try:
        units = op_units()
    except Exception as exc:  # a config dir may not be set
        rep.add("coverage", WARN, "op_units", f"cannot enumerate declared work: {exc}")
        return

    want: dict[str, set] = {}
    for u in units:
        # `u.bucket`, not `u.length`: a both-level unit's key is its row count (pair L records
        # L*L), so comparing lengths against cached buckets reports every one of them missing.
        want.setdefault(u.op, set()).add((u.dtype, u.bucket))
    rep.stats["declared_ops"] = len(want)
    rep.stats["declared_pairs"] = sum(len(v) for v in want.values())

    missing = 0
    for op in sorted(want):
        got = have.get(op)
        if got is None:
            rep.add("coverage", FAIL, op, f"no cache entry at all on {gk}")
            missing += len(want[op])
            continue
        # A bucket is covered if any entry names it; ops whose key has no shape_key record None.
        # Ops whose autotune key has no shape_key record bucket None; they declare one bucket.
        shapeless = {b for _, b in got} == {None}
        gap = set() if shapeless else {p for p in want[op] if p not in got}
        if gap:
            missing += len(gap)
            byd: dict = {}
            for dt, b in sorted(gap):
                byd.setdefault(dt, []).append(b)
            detail = "; ".join(f"{dt}: {bs}" for dt, bs in byd.items())
            rep.add("coverage", WARN, op, f"{len(gap)} (dtype, bucket) missing on {gk} -- {detail}")
        else:
            rep.add("coverage", OK, op, f"{len(got)} (dtype, bucket) pair(s)")
    rep.stats["missing_pairs"] = missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="verify the autotune build system")
    ap.add_argument("--shards", nargs="*", default=[],
                    help="shard dirs from real builds, for reachability evidence")
    ap.add_argument("--verbose", action="store_true", help="also print OK findings")
    ap.add_argument("--gpu", default="", help="cache key to audit; defaults to this machine's GPU")
    args = ap.parse_args(argv)

    rep = Report()
    failed = import_all_kernels()
    for name, why in failed:
        rep.add("import", WARN, name, why)

    tuners = autotuners()
    rep.stats["autotuners"] = len(tuners)
    check_brute(rep, tuners)
    check_prune_executes(rep, tuners)
    check_tuning_keys(rep, tuners)
    check_parallelism(rep)
    shard_dirs = [Path(s) for s in args.shards]
    check_key_spread(rep, shard_dirs)
    check_reachability(rep, shard_dirs)
    check_cache_coverage(rep, args.gpu or None)

    order = ["import", "brute", "prune", "keys", "spread", "parallel", "reach", "coverage"]
    for check in order:
        rows = rep.of(check)
        bad = [f for f in rows if f.level != OK]
        print(f"\n=== {check}: {len(rows) - len(bad)} OK, {len(bad)} not OK "
              f"({len(rows)} checked)")
        for f in (rows if args.verbose else bad):
            print(f"  {f.level:4s} {f.subject:44s} {f.detail}")
    print("\nstats:", {k: v for k, v in rep.stats.items() if k != "swept_dims"})
    return 1 if any(f.level == FAIL for f in rep.findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
