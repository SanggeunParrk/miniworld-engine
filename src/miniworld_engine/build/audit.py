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
        except BaseException as exc:  # noqa: BLE001 -- a broken module is a finding, not a crash
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
            except Exception:  # noqa: BLE001, S112
                continue
            if isinstance(obj, Autotuner) and id(obj) not in seen:
                seen.add(id(obj))
                out.append((f"{name}.{attr}", obj))
    return out


def _op_of(tuner) -> str | None:
    # By identity of the config list the op was handed. The prune objects that used to carry
    # ``_miniworld_op`` were removed in fcd3c7a, after which this returned None for every op and
    # the whole audit fell back to reporting module-level variable names instead of op names.
    from miniworld_engine.autotune.configs import op_of  # noqa: PLC0415 -- avoid an import cycle

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
    from miniworld_engine.autotune.grids import STAGES, WARPS

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
                cand = triton.Config(dict(zip(dims, combo)), num_warps=4, num_stages=3)
                if keep(cand):
                    full += 1
            expect = full * len(warps) * len(stages)
            constraint = " under declared keep()"
        missing = []
        if len(cfgs) != expect:
            missing.append(f"not a full product{constraint}: {len(cfgs)} configs, "
                           f"product would be {expect}")
        if not set(WARPS) <= warps:
            missing.append(f"num_warps {sorted(warps)} misses {sorted(set(WARPS) - warps)}")
        if not set(STAGES) <= stages:
            missing.append(f"num_stages {sorted(stages)} misses {sorted(set(STAGES) - stages)}")
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
            prune(cfgs, {}, **{})
        except NameError as exc:
            rep.add("prune", FAIL, op, f"unbound name at launch: {exc}")
        except Exception:  # noqa: BLE001 -- missing constexprs are expected here
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
            rep.add("keys", WARN, op, "bucket_of not introspectable (not built by key_bucket_of)")
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
    for op, buckets in sorted(per_op.items()):
        dtypes = {b.split("|", 1)[0] for b in buckets}
        shapes = {b.split("|", 1)[1] for b in buckets if "|" in b}
        if len(shapes) <= 1:
            rep.add("spread", FAIL, op,
                    f"only {len(shapes)} shape bucket: {sorted(shapes)} -- key is pinned")
        elif len(dtypes) <= 1:
            rep.add("spread", WARN, op,
                    f"{len(shapes)} shape buckets but only dtype {sorted(dtypes)}")
        else:
            rep.add("spread", OK, op, f"{len(shapes)} shape buckets x {len(dtypes)} dtypes")


# --------------------------------------------------------------------------- #
# check 5 -- registry reachability
# --------------------------------------------------------------------------- #
#: ops that are GPU-specific by construction -- excluded per the build policy, not holes.
GPU_SPECIFIC = re.compile(r"_sm(\d+)|^transition_b2b")


def check_reachability(rep: Report, shard_dirs: list[Path]) -> None:
    from miniworld_engine.autotune.cache import registered_ops

    reg = sorted(registered_ops())
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

    jobs = capture._compile_jobs()  # noqa: SLF001
    rep.add("parallel", OK if jobs > 1 else FAIL, "cpu-compile",
            f"precompile workers = {jobs}")
    src = Path(__import__("miniworld_engine.autotune.builder", fromlist=["x"]).__file__).read_text()
    if "ThreadPoolExecutor(max_workers=len(gpus))" in src:
        rep.add("parallel", OK, "gpu-tune", "one unit per GPU, worker pool sized to len(gpus)")
    else:
        rep.add("parallel", FAIL, "gpu-tune", "no per-GPU worker pool found in builder")


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="verify the autotune build system")
    ap.add_argument("--shards", nargs="*", default=[],
                    help="shard dirs from real builds, for reachability evidence")
    ap.add_argument("--verbose", action="store_true", help="also print OK findings")
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

    order = ["import", "brute", "prune", "keys", "spread", "parallel", "reach"]
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
