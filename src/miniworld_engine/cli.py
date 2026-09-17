"""``miniworld-engine`` command line: build autotune caches and run benchmarks.

    miniworld-engine build all                # the DECLARED work list: 2,101 (op, dtype, bucket)
    miniworld-engine build all --resume       # skip what a previous run already claimed
    miniworld-engine bench_kernel all         # the kernel benches
    miniworld-engine bench_module all         # the module benches

Three commands, because there are three things to do: make this card's cache, measure a kernel,
measure a module. `build` decomposes, runs and merges in one go -- the merge is a step of it, not
a thing to remember.

The pieces underneath are still reachable, under `dev`, where they do not clutter the answer to
"what can I run":

    miniworld-engine dev merge --shards <shard-dir>   # shards from ANOTHER machine, or a re-merge
    miniworld-engine dev audit                        # build-system + cache-coverage checks
    miniworld-engine dev cache-status                 # are the committed caches stale? (no GPU)
    miniworld-engine dev cache-backfill               # recover driver_identity from git history
    miniworld-engine dev space-backfill               # recover each entry's searched config space
    miniworld-engine dev capture all                  # shards without merging (see below)

`dev audit`'s checks are already run by the CPU suite (`test_registry_complete`,
`test_declared_dtype_coverage`, `test_spread_shape_key`); the command adds only `--shards`, i.e.
evidence from a real build that a test cannot have. `dev capture` is the older two-step path:
`build --per-module` is the same decomposition, and neither is the default, because driving
production MODULES only reaches the kernels a module's own shapes dispatch to -- 48 of 91 triton
kernels on an A6000 -- while `build`'s per-op sweep drives the registry's declared list and
reaches all of them.

Everything the run depends on is an argument. The engine used to take these as environment
variables, which meant a run's behaviour lived in shell state that nothing recorded: a capture that
benched the PyTorch reference and reported it as ours, and one that skipped every kernel on the
losing side of a dispatch decision, both looked like successful runs.

What a run leaves behind: one shard JSON per unit, holding each op's config grid, its ranked
entries and its `op_id`; one log per unit under `<shards>/logs/`, opening with the config set the
unit resolved and closing with its precompile / compile-guard / launch accounting; and, in each
merged `data/<op>/<gpu>.json`, a `provenance` block naming the build time and the torch and triton
it was built with. The invoking argv is NOT among them -- reconstruct a build from the config set
named in its logs.

Multi-GPU runs inside a SINGLE job: one worker per GPU, each pulling from a shared queue of
(target, mode, sweep axis, dispatch pin) units and running each as a subprocess pinned to its card.
Workers pull rather than take a fixed slice because capture time varies ~30x between units of the
same target, so any static split leaves cards idle while one grinds through the tail. Subprocesses
rather than threads because a capture can hard-crash the CUDA context, and one dead card should not
take the fleet with it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from queue import Empty, Queue


@dataclasses.dataclass(frozen=True)
class ModuleTarget:
    """One module-level bench target: what fills its cache, and what its bench needs on argv.

    The two used to be separate dicts keyed by the same names -- ``TARGETS`` for the bench args and
    ``MODULE_BUILD_CASES`` for the build cases -- and they had already drifted apart:
    ``triangle_multiplication_bidirectional`` was in the second and not the first, and
    ``bench_module all`` reads the first, so `all` silently meant 8 of the 9 module targets.
    """

    #: The build case(s) that drive the same kernels, i.e. what fills this target's cache.
    cases: tuple[str, ...]
    #: Extra bench.py argv this target needs beyond the flags every bench run passes.
    bench_args: str = ""


#: The two halves of the model, as `build` selectors. A kernel declares which one launches it in
#: registry.csv's `stack` column, and `build trunk` / `build diffusion` run the same per-op sweep
#: `build all` runs over just that half -- the trunk being the model's Pairformer / MSA / template
#: stack and the diffusion being token_dit / atom_dit / the SWA atom transformer. A kernel both
#: sides launch is `both` and is built by EITHER: it has to be tuned for whichever half is built
#: first, and building one kernel twice costs build time where missing one costs a production
#: cache miss.
#:
#: A THIRD name space, alongside the cases `build <case>` takes and the kernels `--per-op` takes.
#: It has to be: a case names a module and an op names a kernel, and neither can say "half the
#: model" -- the trunk is 33 kernels across 7 families, and no module or kernel name spells that.
STACKS: tuple[str, ...] = ("trunk", "diffusion")


#: Every module-level bench target. `bench_module <name>` and `dev capture <name>` take these.
#:
#: A module target is named after the production MODULE it benches, spelled the way the engine
#: spells it -- while a kernel target (:data:`KERNEL_TARGETS`) is named after the kernel FAMILY it
#: benches. The two are separate namespaces, which is why `triangle_attention` can be both.
MODULE_TARGETS: dict[str, ModuleTarget] = {
    "transition": ModuleTarget(("transition",)),
    "triangle_multiplication": ModuleTarget(("triangle_multiplication",)),
    "triangle_multiplication_bidirectional": ModuleTarget(
        ("triangle_multiplication_bidirectional",)),
    "triangle_attention": ModuleTarget(
        ("triangle_attention_bidirectional", "triangle_attention_heads")),
    # fp32 stays -- every file in this kernel family states "fp32 io with TF32 tensor cores".
    # `d_single_token=384` does NOT: the bench builds ConditionedTransition(d_hidden=768,
    # d_cond=384), the model's `token_dit`, and pinning d_single_token to 384 made it 384/384 --
    # square, and a combination the model never builds. It was written when the bench had the two
    # roles swapped (d_hidden=d_pair, d_cond=d_single_token) and 384 was the only way to get a
    # sane d_cond out of it.
    "conditioned_transition": ModuleTarget(("conditioned_transition",), "precision=32"),
    "adaptive_layernorm": ModuleTarget(("adaptive_layernorm",)),
    "augmented_attention_token": ModuleTarget(("augmented_attention",)),
    "augmented_attention_atom": ModuleTarget(("augmented_attention",)),
    # ESMFold2 SWA atom attention. The build case already exists (builder.py `swa_atom_attention`,
    # driving SWA3DRoPEAttention); this exposes it to `bench_module` and the capture path.
    "swa_atom_attention": ModuleTarget(("swa_atom_attention",)),
    # The BLOCKS, not the parts. A per-part result does not compose: our kernels are opaque
    # `custom_op`s, so a part pays its launch overhead once and a block pays it once per part,
    # while `torch.compile` fuses across parts in the reference and cannot fuse across ours.
    # Both effects only land on a block. `dit` is the token track (pair-bias attention),
    # `dit_atom` uses the same pair-bias block at atom widths; `swa_dit` uses windowed 3D-RoPE.
    "dit": ModuleTarget(("augmented_attention", "adaptive_layernorm", "conditioned_transition"),
                        "precision=32"),
    "dit_atom": ModuleTarget(("augmented_attention", "adaptive_layernorm", "conditioned_transition")),
    "swa_dit": ModuleTarget(
        ("swa_atom_attention", "adaptive_layernorm", "conditioned_transition"), "precision=32"),
}

#: Named groups so `capture pairformer` means something. A group is just a set of targets.
#: `pairformer` holds the bidirectional trimul because that is what a ``PairformerBlock`` builds
#: (see modules/pairformer/module.py); leaving it out was the same drift as above.
GROUPS: dict[str, tuple[str, ...]] = {
    "all": tuple(MODULE_TARGETS),
    "pairformer": ("transition", "triangle_attention",
                   "triangle_multiplication", "triangle_multiplication_bidirectional"),
    "diffusion": ("conditioned_transition", "adaptive_layernorm",
                  "augmented_attention_token", "augmented_attention_atom",
                  "swa_atom_attention", "dit", "dit_atom", "swa_dit"),
    "attention": ("triangle_attention",
                  "augmented_attention_token", "augmented_attention_atom"),
}

#: Shape ladders. Atom-level modules use a different one.
SHAPES = {
    "default": {"seq_lens": (384, 512, 640, 768, 896, 1024), "d_pairs": (128, 256, 512)},
    "augmented_attention_atom": {"seq_lens": (128, 256, 384), "d_pairs": (16, 32, 64)},
}

#: Dispatch switches a build must sweep BOTH sides of, and the targets that consult each. The card
#: picks one side for the shapes being swept, so the other side's kernels never fire and never get
#: captured — yet they still run in production at other shapes, and would then have no cached
#: configs at all. Not hypothetical: it is why the A6000 cache was missing bias_only_sigmul_* and
#: triangle_attention_bwd_* entirely.
#: Swept independently, NOT as a cross product — each switch selects among its own kernels, so one
#: pinned run per side per switch covers them, while a cross product would multiply build time over
#: combinations no shape ever takes.
#: switch -> (values, applicable targets, applicable modes)
PINS: dict[str, tuple[tuple, tuple[str, ...], tuple[str, ...]]] = {
    # bias_only gate epilogue: fused_gate_out vs sigmoid_gate_fused + the split backward
    "gate_backend": (("fused", "split"), ("triangle_attention",),
                     ("inference", "training")),
    # inference LN+proj concat fusion (layernorm_linear) -- consulted on the inference path only
    "infer_concat": ((True, False), ("triangle_attention",),
                     ("inference",)),
    # Row-broadcast dropout in the trimul residual epilogue. USE_DROPOUT is part of
    # trimul_gate_elem_mul / _bwd_ew's autotune KEY, so dropout on and off are DIFFERENT cache
    # buckets. Every cache built so far was built with it off, so training -- the only place it is
    # live -- found no entry and fell back to the full grid, which presents as a hang. Training
    # only: the module gates dropout on self.training, so pinning it for inference does nothing.
    #: Only the ON value: the unpinned training run already covers dropout=0, so sweeping 0 here
    #: would just rebuild the same shard under a second name.
    "dropout": ((0.25,), ("triangle_multiplication",), ("training",)),
    # transition's forward runs the hand-CUDA fused b2b when it applies, and that kernel is not
    # autotuned -- so the TRITON expand-gate forward never fires during a build and
    # transition_expand_gate_fwd stays absent from the cache, while its backward (which has no CUDA
    # counterpart) is captured normally. The CUDA path only covers fixed shapes, so production
    # falls back to the triton one elsewhere and finds nothing cached. Sweeping it off captures
    # that side. Off only: the default (on) is already covered by the unpinned runs.
    "transition_cuda_b2b": ((False,), ("transition",), ("inference", "training")),
}


@dataclasses.dataclass(frozen=True)
class Job:
    """One capture unit: a bench invocation that writes exactly one shard."""

    target: str
    mode: str
    axis: str
    pin: tuple[str, object] | None      # (switch name, value) or None for the card's own choice
    shard: Path

    def bench_args(self, impl: str, shapes: dict | None = None) -> list[str]:
        shapes = shapes or SHAPES.get(self.target, SHAPES["default"])
        seq_lens, d_pairs = shapes["seq_lens"], shapes["d_pairs"]
        fixed_seq = seq_lens[0]
        if self.axis == "seq_len":
            shape = (f"min_seq_len={seq_lens[0]} max_seq_len={seq_lens[-1]} seq_len_step="
                     f"{seq_lens[1] - seq_lens[0]} d_pair_values=[{d_pairs[0]}]")
        else:
            shape = (f"min_seq_len={fixed_seq} max_seq_len={fixed_seq} seq_len_step=128 "
                     f"d_pair_values=[{','.join(str(d) for d in d_pairs)}]")
        args = [
            f"target={self.target}",
            "level=module",
            f"implementations=[{impl}]",
            f"mode={self.mode}",
            "metric=time",
            "compile=false",
            "cudagraph=manual",
            f"sweep_axis={self.axis}",
            f"name_suffix=build_{self.pin[1] if self.pin else impl}",
            f"+autotune_shard={self.shard}",
            "mask_prob=0.0",
            f"sweep_seq_len={fixed_seq}",
            *shape.split(),
        ]
        if MODULE_TARGETS[self.target].bench_args:
            args.extend(MODULE_TARGETS[self.target].bench_args.split())
        if self.pin:
            name, value = self.pin
            # dropout is a plain bench setting; the others are capture-time dispatch pins
            # `+` appends: hydra's schema is the yaml, not BenchConfig, so a bare `dropout=`
            # fails with "Key 'dropout' is not in struct".
            # bools go over as hydra's `true`/`false` so the bench's typed field accepts them;
            # `1` arrives as an int and fails pydantic validation.
            rendered = str(value).lower() if isinstance(value, bool) else value
            args.append(f"+dropout={value}" if name == "dropout"
                        else f"+pin_{name}={rendered}")
        return args

    @property
    def label(self) -> str:
        pin = f" {self.pin[0]}={self.pin[1]}" if self.pin else ""
        return f"{self.target} {self.mode}/{self.axis}{pin}"


def build_jobs(targets: tuple[str, ...], shard_dir: Path, sweep_dispatch: bool) -> list[Job]:
    jobs = []
    for target in targets:
        for mode in ("inference", "training"):
            pins: list[tuple[str, object] | None] = [None]
            if sweep_dispatch:
                for switch, (values, targets_, modes) in PINS.items():
                    if target in targets_ and mode in modes:
                        pins.extend((switch, v) for v in values)
            for pin in pins:
                for axis in ("seq_len", "d_pair"):
                    tag = f"-{pin[0]}{pin[1]}" if pin else ""
                    name = f"{target}-{mode}-{axis}{tag}"
                    jobs.append(Job(target, mode, axis, pin, shard_dir / f"{name}.json"))
    return jobs


def _compile_jobs_per_worker(n_workers: int) -> int:
    """Cores each GPU worker may use to pre-compile a round.

    The workers share this job's CPU allocation, so hand each an equal slice: letting every worker
    spawn a pool sized to the whole machine would oversubscribe it many times over.
    """
    import os

    try:
        cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = os.cpu_count() or 1
    return max(1, cores // max(1, n_workers))


def run_worker(device: int, queue: Queue, impl: str, repo: Path, log_dir: Path,
               shapes: dict | None = None, compile_jobs: int = 0) -> list[dict]:
    """Drain the shared queue on one GPU, as subprocesses so a crash cannot take the fleet down."""
    results = []
    while True:
        try:
            job = queue.get_nowait()
        except Empty:
            return results
        log = log_dir / f"gpu{device}-{job.shard.stem}.log"
        cmd = [sys.executable, "-u", "benchmarks/runners/bench.py",
               *job.bench_args(impl, shapes if job.target not in SHAPES else None)]
        if compile_jobs:
            cmd.append(f"+compile_jobs={compile_jobs}")
        started = time.monotonic()
        with log.open("w") as handle:
            proc = subprocess.run(
                cmd, cwd=repo, stdout=handle, stderr=subprocess.STDOUT, check=False,
                env=_worker_env(device),
            )
        ops = 0
        if job.shard.exists():
            try:
                raw = json.loads(job.shard.read_text())
                ops = sum(1 for v in raw.values() if isinstance(v, dict) and "entries" in v)
            except (OSError, ValueError):
                ops = 0
        results.append({
            "label": job.label, "gpu": device, "rc": proc.returncode,
            "seconds": round(time.monotonic() - started, 1), "ops": ops,
            "shard": str(job.shard), "log": str(log),
        })
        status = "ok" if proc.returncode == 0 and ops else "EMPTY" if not proc.returncode else "FAIL"
        print(f"  [gpu{device}] {status:5s} {job.label}  {results[-1]['seconds']}s  {ops} ops",
              flush=True)


def _worker_env(device: int) -> dict:
    """A worker inherits the parent env plus the one variable CUDA itself defines."""
    import os

    env = dict(os.environ)
    # Device selection is CUDA's own interface, not an engine switch: the alternative is for every
    # worker to see all GPUs and rely on each kernel honouring a device argument.
    env["CUDA_VISIBLE_DEVICES"] = _bench_visible_device(device)
    return env


def cmd_capture(args: argparse.Namespace) -> int:
    import concurrent.futures as cf

    repo = Path(__file__).resolve().parents[2]
    targets = GROUPS.get(args.target, (args.target,))
    unknown = [t for t in targets if t not in MODULE_TARGETS]
    if unknown:
        print(f"unknown target(s): {', '.join(unknown)}\n"
              f"targets: {', '.join(MODULE_TARGETS)}\ngroups : {', '.join(GROUPS)}",
              file=sys.stderr)
        return 2

    shard_dir = Path(args.shards).expanduser()
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir = shard_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    jobs = build_jobs(tuple(targets), shard_dir, sweep_dispatch=not args.no_sweep_dispatch)
    if args.resume:
        jobs = [j for j in jobs if not j.shard.exists()]
    if not jobs:
        print("nothing to do (every shard already exists; drop --resume to rebuild)")
        return 0

    shapes = None
    if args.seq_lens or args.d_pairs:
        base = SHAPES["default"]
        shapes = {
            "seq_lens": tuple(int(x) for x in args.seq_lens.split(",")) if args.seq_lens
            else base["seq_lens"],
            "d_pairs": tuple(int(x) for x in args.d_pairs.split(",")) if args.d_pairs
            else base["d_pairs"],
        }
        print(f"shape ladder overridden: {shapes}", flush=True)

    gpus = _resolve_gpus(args.gpus)
    print(f"capture: {len(targets)} target(s), {len(jobs)} shard(s), {len(gpus)} gpu(s) -> "
          f"{shard_dir}", flush=True)

    # Workers PULL from a shared queue rather than getting a fixed slice. Capture time varies by
    # ~30x between shards of the SAME target (transition: 20s for inference/seq_len, 761s for
    # training/d_pair), so any static split leaves cards idle while one grinds through the tail.
    queue: Queue[Job] = Queue()
    for job in jobs:
        queue.put(job)

    started = time.monotonic()
    results: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        per_worker = _compile_jobs_per_worker(len(gpus))
        print(f"pre-compile: {per_worker} core(s) per gpu worker", flush=True)
        futures = [pool.submit(run_worker, dev, queue, args.impl, repo, log_dir, shapes, per_worker)
                   for dev in gpus]
        for future in cf.as_completed(futures):
            results.extend(future.result())

    failed = [r for r in results if r["rc"] != 0]
    empty = [r for r in results if r["rc"] == 0 and not r["ops"]]
    (shard_dir / "capture_report.json").write_text(json.dumps(results, indent=2))
    print(f"\ndone in {round((time.monotonic() - started) / 60, 1)} min: "
          f"{len(results) - len(failed) - len(empty)} ok, {len(empty)} empty, {len(failed)} failed")
    for r in empty:
        print(f"  EMPTY {r['label']}  (exit 0 but captured nothing) -> {r['log']}")
    for r in failed:
        print(f"  FAIL  {r['label']}  exit={r['rc']} -> {r['log']}")
    # An empty shard is a silent failure: the run "succeeded" and the merge would just skip it.
    return 1 if (failed or empty) else 0


def cmd_cache_status(args: argparse.Namespace) -> int:
    """Scan every committed cache's fingerprints against the current grids -- no GPU, no kernel
    launch. Returns 1 if any cache is grid/scheme-stale, so CI and a pre-bench check both catch a
    grid edit that was never followed by a rebuild (the failure mode that silently benched the
    heuristic fallback)."""
    from miniworld_engine.autotune import cache_status

    rows = cache_status.scan(gpu_substr=args.gpu or None)
    print(cache_status.format_report(rows, gpu_substr=args.gpu or None))
    if args.verbose:
        for r in rows:
            if r.verdict == "OK":
                print(f"  OK    {r.op:44} {r.gpu}")
    return 1 if any(r.stale for r in rows) else 0


def cmd_space_backfill(args: argparse.Namespace) -> int:
    """Recover ``entry_grids`` for caches written before the field existed.

    Without it every shipped cache answers "what have you already searched?" with silence, and
    ``build`` re-measures kernels whose search was finished months ago -- which is what it did.
    """
    from miniworld_engine.autotune import space_backfill

    rows = space_backfill.backfill(apply=args.apply)
    print(space_backfill.format_report(rows, applied=args.apply))
    if not args.apply:
        print("\n(dry run -- pass --apply to write)")
    return 0


def cmd_cache_backfill(args: argparse.Namespace) -> int:
    """Recover ``driver_identity`` for caches written before the field existed, from the commit
    that last wrote each one -- so the guard turns on now instead of after every cache is rebuilt."""
    from miniworld_engine.autotune import cache_backfill

    rows = cache_backfill.backfill(apply=args.apply)
    print(cache_backfill.format_report(rows, applied=args.apply))
    if not args.apply:
        print("\n(dry run -- pass --apply to write)")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    """Fold shards into the in-repo cache. Sole writer, so it never runs inside a capture worker."""
    from miniworld_engine.autotune import capture
    from miniworld_engine.autotune.cache import gpu_key

    shard_dir = Path(args.shards).expanduser()
    paths = sorted(str(p) for p in shard_dir.glob("*.json"))
    if not paths:
        print(f"no shard files under {shard_dir}", file=sys.stderr)
        return 1

    # Default to THIS machine's key rather than a hand-typed string: a typo silently writes the
    # cache under a GPU nothing will ever look up.
    gpu = args.gpu or gpu_key()
    if gpu == "cpu":
        print('no CUDA device visible; pass --gpu "<gpu key>" to merge from a login node',
              file=sys.stderr)
        return 2

    written = capture.merge_shards(paths, top_k=args.top_k, gpu=gpu, only_ops=None)
    if capture._MERGE_SKIPPED:
        # Two reasons land here and they are not the same news. An unreadable shard is a whole
        # unit's measurements lost; a shard skipped because its bucket keys predate a scheme bump
        # is one the build is deliberately not using. Both are named, never silent.
        print(f"WARNING: {len(capture._MERGE_SKIPPED)} shard(s)/op(s) were NOT merged; each is "
              f"either a lost measurement or one this scheme cannot read:", file=sys.stderr)
        for sp, why in capture._MERGE_SKIPPED[:10]:
            print(f"  {sp}\n      {why}", file=sys.stderr)
    ops = sorted({w[0] for w in written})
    print(f"merged {len(ops)} ops / {len(written)} buckets from {len(paths)} shards into {gpu!r}")
    for op in ops:
        print(f"  {op}")
    return 0


#: Named config sets live here, one directory per set, one ``<op>.csv`` per op inside it.
CONFIG_ROOT = "configs"
#: The config set a bare ``build all`` / ``bench module`` uses. It was the literal string
#: "default", and ``configs/default`` has never existed in this repo -- so both commands failed at
#: argument resolution before doing any work. Building the cache MEANS searching, so the full
#: search grid is the only sensible default; the pinned single-config sets (blk*, warp*, mixed*)
#: exist for A/B runs and have to be asked for by name.
DEFAULT_CONFIG_SET = "grid"


def resolve_config_dir(config_type: str, repo: Path) -> Path | int:
    """Map the ``config_type`` positional to a config directory, or an exit code.

    A config set IS a directory of ``<op>.csv`` files, so the argument is either a path to one or a
    short name resolving to ``configs/<name>``. There is no second mechanism: the same directory
    drives a build, a bench and any accuracy run, which is what keeps them measuring the same thing.
    """
    # Two places: an explicit path, or a short name against the ONE packaged root. There used to
    # be a third -- the repo's own `configs/<name>`, where the A/B sets lived while `grid` was
    # packaged -- so a short name resolved against a different root depending on the caller, and a
    # wheel install could reach only one of them. Every set is packaged now, so the branch is gone.
    from miniworld_engine.autotune import configs as _configs  # heavy; import at use

    packaged_root = _configs.CONFIG_ROOT
    given = Path(config_type).expanduser()
    if given.is_absolute():
        candidates = [given]
    else:
        # `configs/<name>` is the form every script and doc used while the sets lived at the repo
        # root. Moving them into the package would otherwise turn a working command line into
        # "unknown config set", so the prefix is stripped and the name resolved where the sets
        # actually are. A real relative directory still wins if it exists.
        stripped = Path(*given.parts[1:]) if given.parts[:1] == (CONFIG_ROOT,) else given
        candidates = [given, packaged_root / stripped, packaged_root / config_type]
    for c in candidates:
        if c.is_dir():
            return c
    have = sorted(d.name for d in packaged_root.glob("*") if d.is_dir())
    print(f"unknown config set {config_type!r}; have: {', '.join(have) or '(none)'}\n"
          f"a config set is a short name from that list, or a path to a directory of <op>.csv. "
          f"They live in the package now ({packaged_root}), so the old repo-relative form "
          f"`{CONFIG_ROOT}/<name>` no longer resolves. The default is "
          f"{DEFAULT_CONFIG_SET!r}.", file=sys.stderr)
    return 2


def apply_config_dir(directory: Path) -> int:
    """Select ``directory`` as the config set, then import the kernels. Non-zero if unusable.

    ORDER MATTERS. Triton's ``Autotuner.__init__`` keeps the list it is handed only when that list
    is non-empty; hand it an empty one and it substitutes ``[Config({})]`` of its own and drops the
    reference, so filling the list afterwards has no effect and every kernel launches with no tile
    at all (``dynamic_func() missing ... 'BLOCK_M1'``). So the directory has to be set BEFORE the
    kernel modules import and their decorators run.
    """
    from miniworld_engine.autotune.configs import (
        missing_ops,
        registered_ops,
        use_config_dir,
    )
    from miniworld_engine.build.audit import import_all_kernels

    os.environ["MINIWORLD_CONFIG_DIR"] = str(directory)  # inherited by every child process
    use_config_dir(directory, require_all=False)   # sets the directory; nothing registered yet
    import_all_kernels()                           # decorators now read it and arrive non-empty
    missing = missing_ops()
    total = len(registered_ops())
    print(f"config set {directory}: {total - len(missing)}/{total} ops covered", flush=True)
    if missing:
        print(f"  {len(missing)} op(s) have no CSV and cannot launch: "
              f"{', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}", file=sys.stderr)
        return 1
    return 0


#: Kernel-level bench target -> the build case(s) that drive the same kernels.
#:
#: A kernel target is named after the kernel FAMILY in ``kernels/registry.csv`` that it benches --
#: `triangle_attention`, `bias_only_attention`, `augmented_attention`, `fused_ln_mask`,
#: `layernorm`, `adaln` -- except for the four that bench a fused op SHAPE implemented by several
#: families rather than one family (`dual_gemm_epilogue`, `gemm_epilogue`, `gemm_gate`,
#: `transition_b2b`), which are named after the shape. No abbreviations, in either case: the
#: targets used to read `tri_attn` / `bias_attn` / `aug_attn` / `ln_mask`, which is why
#: `bench_kernel triangle_attention` -- the family's own name -- came back "unknown target".
#:
#: Target names and BUILD CASE names are still different name spaces and always were: a target is
#: a measurement, a case is a module driven to fill a cache, and `build` names 21 of the latter.
#: A bench that auto-builds has to cross that gap explicitly; deriving it by string match would
#: silently build nothing for every target whose name differs.
#:
#: Rows come from what each ``bench_kernel_*`` function actually imports, not from its name. A
#: wrong entry degrades to the old behaviour and says so: the build fills the wrong case and the
#: bench then prints the engine's own per-op "no tuned autotune cache" warning, so it cannot
#: silently produce a fast-looking number from an untuned kernel.
KERNEL_TARGETS: dict[str, tuple[str, ...]] = {
    "dual_gemm_epilogue": ("triangle_multiplication_bidirectional", "triangle_multiplication"),
    "dual_gemm_epilogue_bwd": ("triangle_multiplication",),
    "gemm_epilogue": ("layernorm_linear_native",),
    # gemm_epilogue_bwd imports adaln, augmented_attention, bias_only_attention,
    # conditioned_transition, layernorm and layernorm_linear -- it benches the shared GEMM-epilogue
    # backward across all of them, so its cache comes from all of their cases.
    "gemm_epilogue_bwd": ("adaptive_layernorm", "augmented_attention", "attention_pair_bias",
                          "conditioned_transition", "layernorm_native",
                          "layernorm_linear_native"),
    "gemm_gate": ("triangle_multiplication",),
    "gemm_gate_bwd": ("triangle_multiplication",),
    "transition_b2b": ("transition",),
    "transition_b2b_bwd": ("transition",),
    "layernorm": ("layernorm_native", "triangle_multiplication"),
    "layernorm_bwd": ("layernorm_native", "triangle_multiplication"),
    "fused_ln_mask": ("triangle_multiplication",),
    "adaln": ("adaptive_layernorm", "layernorm_linear_native"),
    "adaln_bwd": ("adaptive_layernorm",),
    "triangle_attention": ("triangle_attention_bidirectional", "triangle_attention_heads"),
    "bias_only_attention": ("attention_pair_bias",),
    "augmented_attention": ("augmented_attention",),
    "conditioned_transition_tail": ("conditioned_transition",),
}


def is_bad_unit(result: dict) -> bool:
    """Did this unit fail, as opposed to answer "not on this GPU"?

    A unit that skipped a shape this card cannot hold is not a bad unit. It is a permanent,
    correct answer, and counting it as a failure is what made a resumed job that picked up only
    the leftover OOM shapes report "0 ok, 9 failed" and refuse to merge.

    One function, not an expression inlined at the call site, because the rule has to agree with
    the `skipped` flag the builder sets -- see tests/registry/test_permanent_skip_classification.py, which
    drives both ends of it.
    """
    return (result.get("claimed_elsewhere", False)
            or ((result["rc"] != 0 or not result["ops"]) and not result.get("skipped")))


def _merge_built_shards(args: argparse.Namespace, results: list) -> int:
    """Fold this build's shards into the in-repo cache and report the units that failed.

    Split out of the build command so the partial-merge policy is testable without a GPU:
    which units count as bad, whether a bad one blocks the merge, and what the exit code is.
    """
    from miniworld_engine.autotune import capture  # heavy; import at use

    bad = [r for r in results if is_bad_unit(r)]
    skipped = [r for r in results if r.get("skipped")]
    # capture.merge_shards is the sole writer of the in-repo cache; skipping it would leave the
    # bench reading the OLD cache while the shards just built sat unread.
    #
    # Merge happens even when some units failed. It used to return here instead, which threw away
    # every good measurement in the run: one shape that OOMs on this card, or one kernel that hangs
    # its compiler, and 526 of 527 units went unwritten. A missing (op, bucket) entry is not a
    # wrong one -- the reader warns and falls back to the full grid for exactly that entry -- so a
    # partial cache is strictly better than no cache, and `audit` is what reports the holes.
    # --strict restores the old all-or-nothing behaviour for CI.
    shard_dir = Path(args.shards).expanduser()
    shards = sorted(str(x) for x in shard_dir.glob("*.json"))
    if skipped:
        print(f"{len(skipped)} unit(s) skipped a shape this GPU cannot hold (OOM / smem); their "
              f"entries are absent by design, not by failure.", file=sys.stderr)
    if bad:
        held = [r for r in bad if r.get("claimed_elsewhere")]
        failed = [r for r in bad if not r.get("claimed_elsewhere")]
        if failed:
            print(f"build produced {len(failed)} failed or empty unit(s) of {len(results)}; "
                  "their required cache entries need verification after merging:", file=sys.stderr)
            for r in failed[:10]:
                print(f"  {r['label']} rc={r['rc']} ops={r['ops']} -> {r['log']}", file=sys.stderr)
            if len(failed) > 10:
                print(f"  ... and {len(failed) - 10} more", file=sys.stderr)
        if held:
            print(f"{len(held)} unit(s) are claimed elsewhere; completion is unverified "
                  "by this invocation. After all workers finish, rerun with resume to verify "
                  "completion.", file=sys.stderr)
        if getattr(args, "strict", False):
            print("--strict: not merging.", file=sys.stderr)
            return 1
    # The SHARD DIR decides, not this run's tally. With --resume a late job picks up only the
    # leftovers -- which are exactly the units that keep failing -- so gating the merge on "did
    # this run succeed?" left 526 finished shards unmerged while the job reported rc=1.
    if not shards:
        # Two different outcomes reach here. If units RAN and still left no shard, the build
        # produced nothing and that is a failure. If none ran, there was nothing to produce --
        # a fresh shard dir where the committed cache already answered every unit, which is the
        # case the cache-skip exists to create. Reporting rc=1 for that made the ideal run
        # ("clone, build, everything already tuned") look like a broken one to CI and to a
        # Slurm script reading the exit code.
        if results:
            print("no shards to merge.", file=sys.stderr)
            return 1
        print("nothing to merge: no unit needed building on this card.")
        return 0
    written = capture.merge_shards(shards)
    if capture._MERGE_SKIPPED:
        # Two reasons land here and they are not the same news. An unreadable shard is a whole
        # unit's measurements lost; a shard skipped because its bucket keys predate a scheme bump
        # is one the build is deliberately not using. Both are named, never silent.
        print(f"WARNING: {len(capture._MERGE_SKIPPED)} shard(s)/op(s) were NOT merged; each is "
              f"either a lost measurement or one this scheme cannot read:", file=sys.stderr)
        for sp, why in capture._MERGE_SKIPPED[:10]:
            print(f"  {sp}\n      {why}", file=sys.stderr)
    print(f"=== merged {len(written)} op file(s) into the in-repo cache"
          f"{f' ({len(bad)} unit(s) unresolved -- verify completion and coverage)' if bad else ''}",
          flush=True)
    return 1 if bad and getattr(args, "strict", False) else 0


def _bench_build_first(args: argparse.Namespace, targets: tuple[str, ...], repo: Path,
                       mapping: dict[str, tuple[str, ...]], table: str,
                       config_type: str = DEFAULT_CONFIG_SET) -> int:
    """Build the cache for ``targets`` before benching them, using config set ``config_type``.

    Benching an unbuilt kernel does not measure the engine: with no configs the launch fails, and
    with a full grid the autotuner sweeps mid-measurement, so the number is a tuning run with a
    benchmark wrapped around it.
    """
    from miniworld_engine.autotune import builder

    if getattr(args, "no_build", False):
        # `build all` decomposes into one unit per (op, dtype, shape bucket) -- 2,101 of them, with
        # no redundancy. The CASE decomposition this function uses is the older one: 1,738 module
        # units for the same 91 ops, of which more than half re-tune a bucket another unit already
        # covered. After a completed `build all` the pre-bench build is therefore days of work for
        # nothing, and `audit` is what says whether the cache is complete enough to skip it.
        directory = resolve_config_dir(config_type, repo)
        if isinstance(directory, int):
            return directory
        print(f"=== --no-build: benching against the cache in data/  (config set {config_type})",
              flush=True)
        return apply_config_dir(directory)

    cases: list[str] = []
    for target in targets:
        mapped = mapping.get(target)
        if not mapped:
            print(f"target {target!r} has no build case mapping; add it to {table} or the bench "
                  f"would measure an untuned kernel", file=sys.stderr)
            return 2
        cases.extend(mapped)
    names = tuple(dict.fromkeys(cases))

    directory = resolve_config_dir(config_type, repo)
    if isinstance(directory, int):
        return directory
    rc = apply_config_dir(directory)
    if rc:
        return rc

    selected = [c for c in builder.cases() if c.name in names]
    print(f"=== build before bench: {', '.join(names)}  (config set {config_type})", flush=True)
    results = builder.build_all(selected, Path(args.shards).expanduser(),
                                _resolve_gpus(args.gpus), args.compile_jobs,
                                resume=args.resume, config_dir=directory,
                                units_per_gpu=getattr(args, "units_per_gpu", 1),
                                keep_ir=getattr(args, "keep_ir", False),
                                predict=getattr(args, "predict_unusable", False),
                                bench_clear_mb=getattr(args, "bench_clear_mb", 0),
                                bench_rep_ms=getattr(args, "bench_rep_ms", 0),
                                unit_timeout_seconds=getattr(args, "unit_timeout_seconds",
                                                          builder.DEFAULT_UNIT_TIMEOUT_SECONDS),
                                pin_cores=getattr(args, "pin_cores", False))
    rc = _merge_built_shards(args, results)
    return rc or int(any(row.get("timed_out") for row in results))


#: A shard with no measurements is `{"_key_scheme": <n>}` and nothing else -- 18 bytes on the
#: A6000 rebuild. Anything above this carries at least one config, and the margin is four orders
#: of magnitude, so the count below never has to open a file to be right about it.
_EMPTY_SHARD_BYTES = 64


def _report_finished_units(units: list, shard_dir: Path, resume: bool) -> int:
    """Say how much of this sweep is already measured on disk. Always 0 -- it decides nothing.

    `trunk` and `diffusion` are not disjoint. A kernel either side launches carries `stack=both` in
    registry.csv and both sweeps include it -- measured on the shipped registry: trunk 1269 units,
    diffusion 1353, union 1713, so 909 units are in BOTH. A unit's identity (its shard stem) is
    (op, dtype, side, length, width) and carries no stack, so those 909 write the same shard file
    whichever sweep produced them, and running the second half without resuming re-benches all 909
    for an identical result.

    This used to REFUSE (exit 2) unless `--resume` was passed, so that re-benching took a flag
    rather than forgetting one. `--resume` is the default now, which serves that purpose better:
    forgetting a flag skips the finished work instead of redoing it, and the flag you have to
    remember (`--no-resume`) is the one that costs GPU-hours.

    What is left is the part a default cannot do. Nothing here decides the shards are still GOOD --
    a kernel edited since they were written makes them measurements of code that no longer exists,
    and that judgement is the operator's. So the count is printed rather than assumed away, with
    the two ways to act on it.
    """
    if not shard_dir.is_dir():
        return 0
    # Size, not `_shard_has_entries`. That parses every shard, and a sweep's shard directory is
    # 2,079 files running to megabytes on a shared filesystem: measured at 17 MINUTES of pure
    # startup before a single unit ran, and `build_all` then parses them all again for the resume
    # filter that actually decides anything. This line only prints a count, so it takes the cheap
    # answer -- an empty shard is the 18 bytes of `{"_key_scheme": N}` and one with measurements is
    # never close to that. The exact test stays where it decides work.
    done = [u for u in units
            if (shard_dir / f"{u.stem}.json").exists()
            and (shard_dir / f"{u.stem}.json").stat().st_size > _EMPTY_SHARD_BYTES]
    if not done:
        return 0
    verb = "SKIPPING" if resume else "RE-RUNNING"
    print(f"{verb} {len(done)} of {len(units)} unit(s) that already have measurements under "
          f"{shard_dir} (e.g. {', '.join(u.stem for u in done[:3])}).\n"
          f"  --no-resume re-runs them; `dev cache-status` says whether their kernels have "
          f"changed since.", file=sys.stderr)
    return 0


def _op_names(case: str) -> list[str]:
    """`--per-op`'s positional, as the list of kernels it names. One name is a list of one."""
    return [n for n in (part.strip() for part in case.split(",")) if n]


def _reject_unknown_build_target(args: argparse.Namespace, repo: Path) -> int:
    """0 if `build`'s positional names something real, 2 (having said what is real) if not.

    Which name space applies is the same choice `cmd_build` makes below: `--per-op` (and `all`,
    which defaults to it) selects ops, anything else selects cases. Read from the declarations,
    never by importing: :data:`~miniworld_engine.autotune.builder.CASE_NAMES` is a literal tuple
    and the ops are the first column of ``kernels/registry.csv``.
    """
    from miniworld_engine.autotune.builder import CASE_NAMES  # literal, no imports

    if args.case in ("all", *STACKS):
        return 0
    if args.per_op:
        rows = (Path(__file__).resolve().parent / "kernels" / "registry.csv").read_text()
        ops = {line.split(",", 1)[0] for line in rows.splitlines()[1:] if line.strip()}
        # A COMMA LIST is one sweep over several kernels, and it is not a convenience. `--per-op`
        # took one name, so tuning a related set meant one command per kernel -- and each command
        # re-imports every kernel module before it can build its work list, which is minutes of
        # triton compilation paid once per name, plus a GPU pool that drains and refills between
        # them. Named together they are one pool over one interleaved work list.
        unknown = sorted(set(_op_names(args.case)) - ops)
        if not unknown:
            return 0
        print(f"unknown op(s) {unknown}; --per-op takes kernels from registry.csv "
              f"({len(ops)} of them) as a comma list, or 'all'", file=sys.stderr)
        return 2
    if args.case in CASE_NAMES:
        return 0
    print(f"unknown case {args.case!r}; have: all, {', '.join(CASE_NAMES)}", file=sys.stderr)
    return 2


def cmd_build(args: argparse.Namespace) -> int:
    """Build the cache. The builder owns decomposition and multi-GPU execution; this only parses."""
    from miniworld_engine.autotune import builder

    repo = Path(__file__).resolve().parents[2]
    # Reject a name we already know is not a target BEFORE anything imports. Everything below --
    # apply_config_dir, cases(), op_units() -- imports every kernel module, which is minutes of
    # triton compilation, and `build <typo>` used to spend all of it before printing "unknown
    # case". Both name spaces are declared and readable without importing anything: CASE_NAMES is
    # a literal, and the per-op sweep's names are the first column of registry.csv.
    rc = _reject_unknown_build_target(args, repo)
    if rc:
        return rc

    from miniworld_engine.autotune.preflight import cache_write_access
    try:
        cache_write_access()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    # Select the config directory BEFORE cases(), which imports the kernel modules. An op that
    # calls configs_for() with no directory chosen gets triton's substitute Config({}) and can
    # never be refilled, so use_config_dir then refuses with "15 op(s) registered before a config
    # directory was selected" and the whole command dies -- `build` was unusable without
    # MINIWORLD_CONFIG_DIR already exported. `_bench_build_first` has always had these two in the
    # right order; this was a plain ordering slip.
    directory = resolve_config_dir(args.config_type, repo)
    if isinstance(directory, int):
        return directory
    rc = apply_config_dir(directory)
    if rc:
        return rc

    # Module dispatch defines the main plan; drivers cover its unreachable alternatives.
    def _op_pass():
        # `all`, a STACK (`trunk` / `diffusion`), or one kernel name. A stack is the same sweep
        # `all` runs, narrowed by registry.csv's `stack` column to the kernels one half of the
        # model launches -- so a trunk-first build tunes the Pairformer / MSA / template stack and
        # nothing else, and the DiT stack can follow later.
        stack = args.case if args.case in STACKS else None
        only = None if args.case == "all" or stack else set(_op_names(args.case))
        units = builder.op_units(only, config_dir=directory, stack=stack)
        if not units:
            print(f"no triton op with a driver matched {args.case!r}", file=sys.stderr)
            return None
        if only and len(only) > len({u.op for u in units}):
            # Named and not built is not the same as not named. A kernel with no driver, or none
            # this config set has a grid for, is silently dropped by op_units -- and asking for
            # twenty and getting nineteen must not read as success.
            missing = sorted(only - {u.op for u in units})
            print(f"named but not built: {missing} -- no driver, marked developed=no, or this "
                  f"config set declares no grid for them", file=sys.stderr)
            return None
        what = f"stack {stack!r}" if stack else ("all ops" if only is None else args.case)
        print(f"per-op sweep: {len(units)} (op, shape, width) items — {what}", flush=True)
        return units

    def _module_pass():
        units = [c for c in builder.cases() if args.case in ("all", c.name)]
        if not units:
            names = ", ".join(c.name for c in builder.cases())
            print(f"unknown case {args.case!r}; have: all, {names}", file=sys.stderr)
            return None
        return units

    # A NAMED target drives its MODULE: the two name spaces are different -- `build <case>` names a
    # module and `--per-op <kernel>` names a kernel -- so running the op sweep for a case name
    # filters `op_units` by a name no kernel has and returns "no triton op with a driver matched".
    # Which is what this did for one commit, turning `build gated_projection grid` into exit 2.
    # The MODULE pass is the default now, including for `all`. It used to be the op pass, because
    # the module pass reached only the 48 of 91 kernels some module happened to dispatch to while
    # the op pass drove every kernel with a driver from a DECLARED ladder. That trade is gone:
    # the ladders were hand-written in `op_units` and drifted from what the modules launch (146
    # replay misses), and the module sweep is now enumerated from `registry_module.csv`, which
    # `dev derive` runs to produce `registry_kernel.csv` -- so what this builds is exactly what
    # was predicted, and `dev coverage` checks it without a GPU. The op pass is still here under
    # `--per-op`: it is the only way to reach a kernel no module dispatches to, and `dev derive`
    # names those (17 on sm86, all alternative implementations kept for A/B).
    # A STACK (`trunk` / `diffusion`) stays on the op pass: it narrows by registry.csv's `stack`
    # column, which is a property of a KERNEL, and a module does not belong to one half of the
    # model -- `transition` is launched by both.
    module_pass = args.per_module or (not args.per_op and args.case not in STACKS)

    if module_pass:
        from miniworld_engine.autotune import plan
        sm = builder.device_sm()
        if sm is None:
            print("Module builds require an available CUDA GPU.", file=sys.stderr)
            return 2
        try:
            if args.case == "all":
                from miniworld_engine.autotune.preflight import native_dependencies
                native_dependencies(sm)
            plan.ensure(sm, workers=min(4, max(1, args.compile_jobs or
                        len(os.sched_getaffinity(0)))))
        except (OSError, ValueError) as exc:
            print(f"Build preflight failed: {exc}", file=sys.stderr)
            return 1
    selected = (_module_pass if module_pass else _op_pass)()
    if selected is None:
        return 2

    # Op units only: a Case has no shard of its own (build_all decomposes it into Units first), so
    # there is nothing to compare until that happens, and `--resume` already filters there.
    if not module_pass:
        _report_finished_units(selected, Path(args.shards).expanduser(), args.resume)

    # Build what is missing. That is what a build IS, so it is the default and `--rebuild` is the
    # word for the other thing. It used to be the reverse -- the cheap, correct behaviour needed
    # `--fill-gaps` and the ruinous one was the shorter flag next to it -- and the cost is on
    # record: `layernorm_bwd_split` spent 5h14m re-timing 64 already-tuned units to fill three
    # missing keys, because the run said `--rebuild-cached` when it meant "reach the new buckets".
    fill_gaps = not bool(getattr(args, "rebuild", False))
    # Only model-reachable shapes belong to the default build. Explicit --per-op remains
    # available for isolated implementation diagnostics.
    results: list = builder.build_all(selected, Path(args.shards).expanduser(),
                                      _resolve_gpus(args.gpus), args.compile_jobs,
                                      resume=args.resume, reclaim=args.reclaim,
                                      config_dir=directory, fill_gaps=fill_gaps,
                                      units_per_gpu=getattr(args, "units_per_gpu", 1),
                                      keep_ir=getattr(args, "keep_ir", False),
                                      predict=getattr(args, "predict_unusable", False),
                                      bench_clear_mb=getattr(args, "bench_clear_mb", 0),
                                      bench_rep_ms=getattr(args, "bench_rep_ms", 0),
                                      unit_timeout_seconds=getattr(args, "unit_timeout_seconds",
                                                                builder.DEFAULT_UNIT_TIMEOUT_SECONDS),
                                      pin_cores=getattr(args, "pin_cores", False),
                                      # --fill-gaps has to reach the units too: the unit-level
                                      # skip drops any op whose cache answers at ANY bucket,
                                      # which is exactly the op that owes a new one.
                                      # every unit runs; what it BENCHES is fill_gaps'
                                      # business, so the unit-level skip only gets in the way --
                                      # it drops any op the cache answers at ANY bucket, which is
                                      # exactly the op that owes a new one.
                                      skip_cached=False)
    held = [r for r in results if r.get("claimed_elsewhere")]
    skipped = [r for r in results if r.get("skipped") and not r.get("claimed_elsewhere")]
    finished = [r for r in results if not r.get("claimed_elsewhere") and not r.get("skipped")]
    failed = [r for r in finished if r["rc"] != 0]
    empty = [r for r in finished if r["rc"] == 0 and not r["ops"]]
    successful = [r for r in finished if r["rc"] == 0 and r["ops"] > 0]
    print(f"\n{len(successful)} ok, {len(empty)} empty, {len(failed)} failed, "
          f"{len(skipped)} skipped, {len(held)} claimed elsewhere")
    for r in empty + failed:
        print(f"  {'EMPTY' if r in empty else 'FAIL '} {r['label']} -> {r['log']}")
    for r in held:
        print(f"  HELD  {r['label']} (claimed elsewhere; completion not verified here)")
    dead = _ops_that_measured_nothing(results)
    if dead:
        # A unit that dies at ONE shape is ordinary -- the shape does not fit the card, and the
        # build says so and moves on. A kernel that dies at EVERY shape is a broken driver, and
        # under "fail only if nothing succeeded" it was invisible: `trimul_outproj_layernorm_
        # gemm_gate_triton` called its op without the required `eps`, so every one of its units
        # raised before a single config was timed, on every card, for as long as the op has
        # existed -- and every build reported success and shipped a kernel with no cache at all.
        print(f"\nDRIVER FAILURE: {len(dead)} kernel(s) measured NOTHING at any shape this "
              f"build ran. This is a broken driver, not a shape that does not fit:",
              file=sys.stderr)
        for op, log in sorted(dead.items()):
            print(f"  {op} -> {log}", file=sys.stderr)
    # THE per-op sweep is the path that actually builds the shipped cache, and it was the one path
    # that never merged: `cmd_build` and `_bench_build_first` both folded their shards in, this
    # returned straight to the shell. A 527-unit sweep therefore finished, wrote 145 GB of shards,
    # and left `data/` untouched -- the build looked complete and shipped nothing. Same policy as
    # the other two: merge what succeeded, name the holes, fail only if nothing succeeded.
    rc = _merge_built_shards(args, results)
    # After the merge, so a build that half-worked still ships what it measured -- the same rule
    # the merge itself follows. The exit code is the only thing a batch job's caller sees.
    sm = builder.device_sm()
    if module_pass and args.case == "all" and sm is not None:
        from miniworld_engine.autotune import derive, plan
        from miniworld_engine.autotune.cache import gpu_key
        try:
            plan.load(sm)
            report = derive.coverage(sm, gpu_key())
        except (OSError, ValueError) as exc:
            print(f"Build cannot certify coverage: {exc}", file=sys.stderr)
            return 1
        if report["missing"]:
            print(f"Build incomplete: {len(report['missing'])} required keys are not usable.",
                  file=sys.stderr)
            return 1
    # Module coverage cannot certify the alternative driver pass. Preserve good
    # shards, but never call a full build complete when a planned unit failed.
    incomplete = args.case == "all" and any(is_bad_unit(r) for r in results)
    if incomplete:
        reasons = []
        if failed or empty:
            reasons.append(f"{len(failed)} failed and {len(empty)} empty planned units")
        if held:
            reasons.append(f"{len(held)} units claimed elsewhere, with completion unverified "
                           "by this invocation")
        print(f"Build incomplete: {'; '.join(reasons)}; successful measurements were preserved.",
              file=sys.stderr)
    status = rc or (1 if dead or incomplete else 0)
    # Failed coverage or units need these compiled artifacts on resume. Merge alone
    # proves only that partial timings were saved, not that the build is complete.
    if not status and _should_prune(args):
        _empty_triton_cache(dry_run=False)
    return status


def _ops_that_measured_nothing(results: list) -> dict:
    """Kernels every one of whose units produced no measurement, as op -> a log to read.

    Grouped by OP and not by unit on purpose. The per-unit report already names each failure, and
    a failure there is usually legitimate (a shape too big for the card). What no per-unit line can
    say is that a kernel came out of the whole build with nothing at all, which is what a driver
    that cannot call its own op looks like.
    """
    by_op: dict[str, list] = {}
    for r in results:
        # A PERMANENT skip is not a measurement that failed to happen -- it is the card saying the
        # shape does not fit, which `builder` already records as `skipped` and which the existing
        # merge rule (`is_bad_unit`) deliberately does not count against a build. Reading only
        # `ops` made every op whose driven shapes are all too big for this card a DRIVER FAILURE,
        # which is the exact case the rule above says is ordinary.
        if r.get("skipped"):
            continue
        # `OpUnit.label` is "<op>[<dtype>] <side> L=<n> D=<n>". A module `Unit`'s label is
        # "<case>[<impl>/<dtype>] dims#<n> L=<n> <mode>" -- it has a bracket too, so splitting on
        # one alone grouped a CASE under a kernel's heading and reported "DRIVER FAILURE: transition".
        # The op half of an OpUnit label is a bare identifier; the case half of a module label is
        # not, because its bracket carries "<impl>/<dtype>".
        head, sep, rest = str(r.get("label", "")).partition("[")
        if not sep or "/" in rest.split("]", 1)[0]:
            continue
        if head.strip():
            by_op.setdefault(head.strip(), []).append(r)
    return {op: rs[0].get("log", "") for op, rs in by_op.items()
            if not any(r.get("ops") for r in rs)}


def _should_prune(args: argparse.Namespace) -> bool:
    """Prune by DEFAULT whenever the build was handed its own cache directory.

    Ownership is the condition, and the only one. A build pointed at a $TRITON_CACHE_DIR of its
    own is the scenario that drained two nodes -- 118 GB of compile cache on the filesystem that
    holds SlurmdSpoolDir, with cleanup left to a flag somebody had to remember -- so there the
    cache is emptied after a successful merge unless --keep-triton-cache says otherwise (the
    honest reason to keep it: re-measuring unchanged kernels, where a warm cache skips the
    recompile). With the variable UNSET the cache is ~/.triton/cache, shared with every other
    Triton workload on the machine; the build does not own that and never touches it --
    `_empty_triton_cache` refuses even an explicit --prune-cache there.

    --prune-cache survives as a no-op-when-defaulted for the job scripts that already pass it.
    """
    if getattr(args, "keep_triton_cache", False):
        return False
    if getattr(args, "prune_cache", False):
        return True
    return bool(os.environ.get("TRITON_CACHE_DIR"))


def _empty_triton_cache(dry_run: bool) -> int:
    """Empty the triton cache the build filled. Returns a process exit code.

    A build artifact, not an output: what ships is `autotune/data/`, which names configs. Measured
    on the A6000 rebuild, the cache was 221,487 entries and 40 GB of a filesystem shared with the
    rest of the lab.
    """
    import os

    from miniworld_engine.autotune import triton_cache

    raw = os.environ.get("TRITON_CACHE_DIR")
    if not raw:
        print("TRITON_CACHE_DIR is not set, so this build shares the default cache "
              "(~/.triton/cache) with everything else on this machine. Refusing to empty it: "
              "point the build at its own directory and this becomes safe.")
        return 2
    directory = Path(raw).expanduser()
    try:
        entries, total = triton_cache.clear(directory, dry_run=dry_run)
    except ValueError as exc:
        print(f"  {exc}")
        return 2
    verb = "would remove" if dry_run else "removed"
    print(f"  triton cache: {verb} {entries:,} entries, {total / 1024**3:.1f} GB from {directory}")
    return 0


def cmd_install_flash(args: argparse.Namespace) -> int:
    """Install whichever FlashAttention this machine's card can actually run.

    Two incompatible lines and no way to say so in the metadata: FA4 is sm_90+, FA2 covers
    sm_80/86/89, and a wheel for the wrong one installs cleanly and then never loads. PEP 508
    markers cannot branch on a GPU, so `pyproject.toml` has to expose both as extras
    (`[flash]` / `[flash2]`) and leave the choice to whoever is installing. This makes that
    choice from the device instead of from the reader.

    NOT run on import, and not a side effect of anything else. Installing packages is the
    caller's decision; this only makes it one command instead of a lookup table.
    """
    import subprocess
    import sys

    cap = args.arch
    if not cap:
        try:
            import torch

            if not torch.cuda.is_available():
                print("no CUDA device visible; run this on the machine that will train, or "
                      "pass --arch (e.g. --arch sm80)")
                return 2
            major, minor = torch.cuda.get_device_capability()
            cap = f"sm{major}{minor}"
        except ImportError:
            print("torch is not importable, so the card cannot be detected; pass --arch")
            return 2
    sm = int(cap.removeprefix("sm"))
    if sm >= 90:
        # Prerelease wheel, hence --pre; see the `flash` extra's note in pyproject.toml.
        cmd = [sys.executable, "-m", "pip", "install", "--pre", "flash-attn-4"]
    elif sm >= 80:
        # No prebuilt wheel matches every torch/CUDA pair, so this compiles (~40 min).
        cmd = [sys.executable, "-m", "pip", "install", "--no-build-isolation",
               "flash-attn>=2.8,<3"]
    else:
        print(f"{cap}: neither FlashAttention line supports this card. swa_atom_attention will "
              f"use its SDPA band, which needs an [N, S, S] mask -- 24 GiB at A=48, S=8192.")
        return 1
    print(f"{cap} -> {' '.join(cmd[3:])}")
    if args.dry_run:
        return 0
    return subprocess.call(cmd)


def cmd_buckets(args: argparse.Namespace) -> int:
    """Measure which of each row's declared widths actually reach its cache key.

    `op_units` crosses every kernel with a ladder of channel widths on the assumption that a
    different width is a different bucket. For 17 of the 91 triton ops it is not -- they file every
    declared width into ONE, so 217 of 2,079 units were re-timing a bucket a sibling unit had
    already timed and overwriting it. The bucket is a packed integer built from constexprs that do
    not exist until the launcher builds them, so the only way to answer this is to launch and read
    the key back, which is what this does. `op_units` then drops the duplicate units.

    Re-run it after adding a kernel, changing a driver's shape derivation, or moving a `width`
    column; an op the file does not cover keeps its full ladder, so a stale file loses no coverage.
    """
    import csv
    from pathlib import Path

    from miniworld_engine.autotune import builder, width_evidence

    rows = {r["kernel"]: r for r in csv.DictReader(
        (Path(builder.__file__).resolve().parent.parent
         / "kernels" / "registry.csv").open(newline=""))}
    # The DECLARED ladders, not the plan: `op_units` narrows the plan with the evidence this
    # command writes, so probing the plan would re-measure only what the last run left standing and
    # a collapse could never be revisited -- a driver that started honouring its width would keep
    # the one rung it was cut down to, forever.
    from unittest.mock import patch

    def no_evidence(path: Path | None = None) -> dict[str, dict[str, list[str]]]:
        return {}

    with patch.object(width_evidence, "load", no_evidence):
        declared = builder.op_units(None)

    want: dict[tuple[str, str], set[int]] = {}
    for u in declared:
        r = rows.get(u.op)
        if not r or (r.get("developed") or "yes").strip() == "no":
            continue
        if r["backend"] != "triton" or not (r.get("driver") or "").strip():
            continue
        want.setdefault((u.op, r["driver"]), set()).add(u.width or 0)
    only = set(args.ops.split(",")) if args.ops else None
    todo = [(op, ref, tuple(sorted(ws))) for (op, ref), ws in sorted(want.items())
            if only is None or op in only]
    print(f"probing {sum(len(w) for _, _, w in todo)} (op, width) launches "
          f"across {len(todo)} ops", flush=True)
    out = Path(args.out) if args.out else width_evidence.EVIDENCE
    got = width_evidence.measure(todo, out=out)
    collapsed = [op for op, per in got.items()
                 if len(per) > 1 and len({tuple(v) for v in per.values()}) == 1]
    print(f"wrote {out} -- {len(got)} ops; {len(collapsed)} file every declared width into one "
          f"bucket: {sorted(collapsed)}", flush=True)
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    """Is the built cache exactly what `registry_kernel.csv` says it should be?

    No GPU, no replay: `dev derive` already ran the same dispatch, so the answer is a set
    difference between two files.
    """
    from miniworld_engine.autotune import derive, plan

    try:
        plan.load(args.arch)
    except (OSError, ValueError) as exc:
        print(f"Coverage is unverified: {exc}", file=sys.stderr)
        return 2
    rep = derive.coverage(args.arch, args.gpu)
    print(f"{rep['gpu']} [{rep['arch']}]: derived {rep['want']} buckets, cache holds {rep['have']}")
    if rep["missing"]:
        print(f"\n  MISSING -- production reaches these with no entry ({len(rep['missing'])}):")
        for op, key in rep["missing"][:40]:
            print(f"    {op}  {key}")
        if len(rep["missing"]) > 40:
            print(f"    ... {len(rep['missing']) - 40} more")
    if rep["extra"]:
        by_op: dict = {}
        for op, key in rep["extra"]:
            by_op.setdefault(op, []).append(key)
        print(f"\n  OUTSIDE PLAN -- preserved, not classified as dead ({len(rep['extra'])} in "
              f"{len(by_op)} kernels):")
        for op, keys in sorted(by_op.items())[:40]:
            print(f"    {op}  {len(keys)} buckets")
    if not rep["missing"] and not rep["extra"]:
        print("  exact match")
    return 1 if rep["missing"] else 0


def cmd_derive(args: argparse.Namespace) -> int:
    """Derive `registry_kernel.csv` from `registry_module.csv` by running the modules on fake
    tensors -- no kernel compiled, no memory allocated, every branch decided by Python.

    This is the whole replacement for the hand-written per-axis ladders in `builder`. Those said
    WHICH axis a kernel buckets on and then guessed the VALUES, and every guess that drifted from
    what the modules launch was a bucket production reaches with no cache entry. Here the values
    are not guessed: the modules are run, and what they launch is what gets written.
    """
    from miniworld_engine.autotune import derive, plan

    source = plan.source_identity()
    out = Path(args.out) if args.out else derive.REGISTRY_KERNEL
    evidence = Path(args.per_unit) if getattr(args, "per_unit", "") else out.with_suffix(".units.json")
    rows = derive.module_rows()
    work = derive.units(rows, arch=args.arch or derive.sm_tag())
    print(f"{len(rows)} module rows -> {len(work)} invocations to record", flush=True)

    def progress(i, total, unit, launches, error):
        if error:
            print(f"  [{i}/{total}] {unit.label}: {error}", flush=True)
        elif i % 25 == 0 or i == total:
            print(f"  [{i}/{total}] {unit.label}: {len(launches)} launches", flush=True)

    entries, skipped = derive.derive_all(
        rows, on_unit=progress, arch=args.arch or None,
        per_unit=evidence, workers=getattr(args, "workers", 1))
    arch = args.arch or derive.sm_tag()
    out = Path(args.out) if args.out else derive.REGISTRY_KERNEL
    if skipped:
        from miniworld_engine._atomic import write_json
        report = out.with_suffix(".errors.json")
        write_json(report, [{"unit": u.label, "reason": why} for u, why in skipped])
        print(f"derivation FAILED: {len(skipped)} invocations; details: {report}",
              file=sys.stderr)
        print(f"Existing registry preserved: {out}", file=sys.stderr)
        return 1
    if source != plan.source_identity():
        print("Dispatch sources changed during derivation; existing registry preserved.", file=sys.stderr)
        return 1
    written = derive.write_kernel_registry(entries, arch, out)
    plan.stamp(evidence, out, source)
    kernels = {op for op, _, _ in entries if not op.startswith("<")}
    unresolved = sorted({op for op, _, b in entries
                         if op.startswith("<") or str(b).startswith("<")})
    print(f"\n{written} rows written to {out}")
    print(f"  {len(kernels)} kernels, arch {arch}")
    if unresolved:
        print(f"  {len(unresolved)} launches with no cache entry to build:")
        for name in unresolved[:20]:
            print(f"    {name}")
    if skipped:
        print(f"  {len(skipped)} invocations the modules refused (shape not supported):")
        for unit, why in skipped[:20]:
            print(f"    {unit.label}: {why}")
    return 0


def cmd_callers(args: argparse.Namespace) -> int:
    """Record which KERNEL each production case dispatches into.

    `registry.csv` says which family a kernel belongs to. It does not say who CALLS it, and for the
    shared kernels those are different questions: `layernorm_fwd_saveact_triton` is reached by the
    transition, the trimul and the DiT, each at its own width. Without that link, "does the plan
    cover production" cannot be answered without running production -- which is `dev audit
    --replay`, a card and half an hour, and it reports the answer as a list of missed keys rather
    than as a rule anything can check.

    With it the question is arithmetic: a case declares dims and lengths, this file says which
    kernels that case reaches, and a test can compare that against `op_units` at import time.

    Measured, not declared, for the same reason `dev buckets` is: a hand-written "called by" column
    would be one more list to drift, and drifting lists are what every miss in this repo has come
    from. Re-run it when a module's dispatch changes.
    """
    import json
    from pathlib import Path

    import torch

    if not torch.cuda.is_available():
        print("dev callers runs the production cases; this machine has no CUDA device")
        return 2
    from miniworld_engine import settings
    from miniworld_engine.autotune import builder, capture

    settings.configure(run_autotune=False, capture=False)
    capture.install_launch_recorder()
    out: dict[str, dict] = {}
    for case in builder.cases():
        ops: set[str] = set()
        ran = 0
        for di in range(len(case.dims)):
            for length in case.lengths_for(di):
                for dtype in case.dtypes:
                    for train in ((False, True) if case.train else (False,)):
                        for impl in case.impls:
                            capture.clear_launched_ops()
                            if builder.run_case(case, length, di, train=train, impl=impl,
                                                dtype=dtype):
                                ran += 1
                                ops |= set(capture.launched_ops())
        out[case.name] = {"ops": sorted(ops), "ran": ran,
                          "lengths": sorted(case.lengths),
                          "dims": [dict(d) for d in case.dims]}
        print(f"  {case.name:32s} {ran:3d} run(s), {len(ops):3d} kernel(s)", flush=True)
    path = Path(args.out) if args.out else (
        Path(builder.__file__).resolve().parent.parent / "kernels" / "case_callers.json")
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path}: {len(out)} cases, "
          f"{len({o for v in out.values() for o in v['ops']})} distinct kernels", flush=True)
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Verify the build system AND that every declared (op, bucket) is in the shipped cache."""
    from miniworld_engine.build import audit as _audit  # imports every kernel

    if args.replay:
        return _replay_audit()
    argv = ["--gpu", args.gpu] if args.gpu else []
    if args.shards:
        argv += ["--shards", *args.shards]
    if args.verbose:
        argv += ["--verbose"]
    return _audit.main(argv)


def _replay_audit() -> int:
    """The direct coverage measurement: run the build matrix against the finished cache and print
    every lookup it did not serve.

    `builder.audit` has always existed and `cache.py` has always pointed at it as "the only direct
    measure of whether the cache covers a workload" -- while nothing called it, from any command
    or any test. The static check (`dev audit` without this flag) compares DECLARED work against
    the cache, and declared work is (op, dtype, shape bucket); the cache key also carries the
    kernel's constexprs, so an op can be 100% covered by that measure and still miss at run time.
    Measured: one pass of the GPU suite on an A6000 hit 14 such (op, key) misses against a cache
    the static check reports as complete.

    Needs a card: it launches the modules.
    """
    import torch

    if not torch.cuda.is_available():
        print("--replay runs the build matrix; this machine has no CUDA device")
        return 2
    from miniworld_engine.autotune import builder

    misses = builder.audit(builder.cases())
    for op, gpu, key in misses:
        print(f"  MISS {op:44s} {gpu}  {key}")
    print(f"\nreplay: {len(misses)} lookup(s) the cache did not serve")
    return 1 if misses else 0


def _bench_cmd(args: argparse.Namespace, target: str, config_dir: Path | None,
               level: str) -> tuple[list[str], dict | None]:
    """The argv and env for one target's bench.py run.

    `level` is bench.py's namespace selector, not decoration: a kernel and the module built out of
    it share a name (`triangle_attention` is both), so a target is only identified by the pair.
    """
    # bench.py requires a mode. For a kernel target it is not the caller's to choose: the name
    # says it (`*_bwd` = training, else inference), which is why bench_kernel has no --mode.
    mode = ("training" if target.endswith("_bwd") else "inference") if level == "kernel" \
        else args.mode
    cmd = [sys.executable, "-u", "benchmarks/runners/bench.py",
           f"target={target}", f"level={level}",
           f"implementations=[{args.impl}]", f"mode={mode}", "metric=time",
           f"sweep_axis={getattr(args, 'sweep_axis', 'seq_len')}",
           f"cudagraph={getattr(args, 'cudagraph', 'manual')}",
           f"compile={getattr(args, 'compile', 'true')}"]
    module_target = MODULE_TARGETS.get(target) if level == "module" else None
    if module_target is not None and module_target.bench_args:
        cmd.extend(module_target.bench_args.split())
    env = None
    wrap = getattr(args, "compile_wrap", "")
    if wrap:
        env = {**os.environ, "MINIWORLD_COMPILE_WRAP": wrap}
    if config_dir is not None:
        # The child must have the set in its ENVIRONMENT, not on its argv: bench.py's own header
        # imports kernel modules, so anything read inside main() lands after the autotuners have
        # already been handed empty lists. +config_dir is kept so the child can assert they agree.
        cmd.append(f"+config_dir={config_dir}")
        env = {**(env or os.environ), "MINIWORLD_CONFIG_DIR": str(config_dir)}
    return cmd, env


def _bench_visible_device(gpu: int) -> str:
    """Map a logical device to its token in the parent's CUDA visibility mask."""
    from miniworld_engine.autotune.builder import visible_device

    return visible_device(gpu)


def _run_bench(args: argparse.Namespace, targets: tuple[str, ...], repo: Path,
               config_dir: Path | None = None, *, level: str) -> int:
    """Bench every target, one process each, spread across the visible GPUs.

    Targets are independent processes, so on an N-GPU node N of them run at once -- pinned with
    CUDA_VISIBLE_DEVICES so each child sees exactly one card and cannot contend for memory with a
    sibling. Running them serially on one card was leaving the rest of the node idle for the whole
    sweep, and `all` is 17 targets.

    Each child's output is captured and printed as one block when it finishes: interleaving live
    output from several benches produces a log where no line can be attributed to a target.
    """
    started = time.time()   # coverage must count THIS run's .ops, not history
    gpus = _resolve_gpus(args.gpus) or [0]
    devices = {gpu: _bench_visible_device(gpu) for gpu in gpus}
    if len(set(devices.values())) != len(gpus):
        raise ValueError("benchmark GPUs must identify distinct visible devices")
    jobs = [(t, *_bench_cmd(args, t, config_dir, level)) for t in targets]
    rc = 0

    def run(job: tuple, gpu: int) -> tuple[str, int, str]:
        target, cmd, env = job
        env = {**(env or os.environ), "CUDA_VISIBLE_DEVICES": devices[gpu]}
        done = subprocess.run(cmd, cwd=repo, check=False, env=env,
                              capture_output=True, text=True)
        return target, done.returncode, (done.stdout or "") + (done.stderr or "")

    if len(gpus) == 1 or len(jobs) == 1:
        for job in jobs:
            target, code, out = run(job, gpus[0])
            print(f"=== bench {target}  (gpu {gpus[0]}, rc={code})", flush=True)
            print(out, flush=True)
            rc |= code
    else:
        import concurrent.futures as cf
        import threading

        print_lock = threading.Lock()
        # A worker owns a GPU for its entire queue. Submitting individual jobs
        # round-robin lets an early finisher steal another busy GPU's next job.
        def run_queue(gpu: int, queue: list[tuple]) -> int:
            queue_rc = 0
            for job in queue:
                target, code, out = run(job, gpu)
                with print_lock:
                    print(f"=== bench {target}  (gpu {gpu}, rc={code})", flush=True)
                    print(out, flush=True)
                queue_rc |= code
            return queue_rc

        with cf.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = {pool.submit(run_queue, gpu, jobs[index::len(gpus)]): gpu
                       for index, gpu in enumerate(gpus)}
            for fut in cf.as_completed(futures):
                rc |= fut.result()
    if len(targets) > 1:
        rc |= _report_coverage(targets, repo, level, since=started)
    return rc


def _report_coverage(targets: tuple[str, ...], repo: Path, level: str, *,
                     since: float = 0.0) -> int:
    """Compare the kernels that actually launched against everything the repo declares.

    The denominator is ``kernels/registry.csv`` -- declared data, not something derived from this
    run. Deriving it from the run would drop every unreachable kernel out of numerator and
    denominator together and coverage would read 100% forever.

    Nothing here decides whether a kernel *could* run. A kernel either launched or it did not.
    """
    from miniworld_engine.autotune import devices
    from miniworld_engine.autotune.cache import gpu_key

    key = gpu_key()
    launched: set[str] = set()
    for target in targets:
        # bench.py owns the layout: benchmarks/{kernels,modules}/<target>/artifacts/<gpu>/<run>.ops
        # -- one directory per (level, target), with no exceptions, so the path is derived and not
        # guessed. It used to be guessed from `target in KERNEL_TARGETS`, which was wrong for
        # augmented_attention_token/_atom: they shared one directory named after neither of them,
        # so `base` did not exist and both targets' kernels were dropped from coverage in silence.
        base = repo / "benchmarks" / f"{level}s" / target
        if not base.is_dir():
            continue
        for f in base.glob("artifacts/*/*.ops"):
            # only this run: a .ops written before a rename still names kernels that no longer
            # exist, and counting it makes the coverage report describe history, not the run
            if f.stat().st_mtime < since:
                continue
            launched |= {ln.strip() for ln in f.read_text().split("\n") if ln.strip()}

    declared = devices.registered_kernels()
    hit = declared & launched
    # weak: the bench knows the kernel LAUNCHED and nothing about its numbers. Without this it
    # overwrote `run_all`'s measured `rel out=2.8e-03` with "launched by bench", and the declared
    # rtol bands are calibrated from exactly those numbers.
    devices.record(key, dict.fromkeys(hit, (True, "launched by bench")), weak=True)
    missed = sorted(declared - launched)
    print(f"\n=== coverage on {key}")
    print(f"    declared: {len(declared)}   launched: {len(hit)}   never launched: {len(missed)}")
    stray = sorted(launched - declared)
    if stray:
        print("    launched but NOT in registry -- add them:", file=sys.stderr)
        for op in stray:
            print(f"      {op}", file=sys.stderr)
    if missed:
        print("    no bench reaches these kernels:", file=sys.stderr)
        for op in missed:
            print(f"      {op}", file=sys.stderr)
    return 1 if (missed or stray) else 0

def cmd_bench_kernel(args: argparse.Namespace) -> int:
    """Kernel-level bench: the ``bench_kernel_*`` entry points, with a config_type."""
    repo = Path(__file__).resolve().parents[2]
    # 'all' is the honest default unit of work: a sweep that names one target measures one
    # target, and calling that "the kernels" is how a broken implementation stays hidden.
    targets = tuple(KERNEL_TARGETS) if args.target == "all" else (args.target,)
    unknown = [t for t in targets if t not in KERNEL_TARGETS]
    if unknown:
        print(f"unknown kernel target(s): {', '.join(unknown)}; have: all, "
              f"{', '.join(sorted(KERNEL_TARGETS))}", file=sys.stderr)
        return 2
    # A config set means "measure AT these configs". There is then nothing to tune, so there is
    # nothing to build: a build exists to SEARCH for the best config and write a cache, and its
    # unit list is a cross product over impls, dtypes, shapes, train/eval and setting switches --
    # hundreds of module runs that say nothing about whether a pinned config computes the right
    # answer. Building here also imports the failure modes of paths the run does not even use
    # (a cute impl the card cannot run, a case whose arguments are mis-ordered), and a single bad
    # unit aborts a measurement that would otherwise have succeeded.
    if args.config_type:
        directory = resolve_config_dir(args.config_type, repo)
        if isinstance(directory, int):
            return directory
        rc = apply_config_dir(directory)
        if rc:
            return rc
    else:
        # No config set: the autotuner would search its full grid mid-measurement, which times a
        # search rather than a kernel. Build first so the search happens once, up front.
        rc = _bench_build_first(args, targets, repo, KERNEL_TARGETS, "KERNEL_TARGETS")
        if rc:
            return rc
        directory = None
    return _run_bench(args, targets, repo, directory, level="kernel")


def cmd_bench_module(args: argparse.Namespace) -> int:
    """Module-level bench: takes NO config_type.

    A module bench is the production-shaped measurement -- whole module, its own dispatch decisions
    -- so the config space is not the caller's to pick: it is whatever the cache holds. That is the
    ``grid`` set, passed here as a constant rather than an argument so the two cannot disagree.

    Unlike ``bench_kernel``, this does NOT run a pre-bench build. Warmup resolves
    missing buckets using the runtime fallback policy, which may sample only a
    subset of the grid. This does not certify complete autotune coverage; build
    and audit the cache separately when a fully tuned comparison is required.
    ``--no-build`` is accepted for symmetry with ``bench_kernel``.
    """
    repo = Path(__file__).resolve().parents[2]
    targets = GROUPS.get(args.target, (args.target,))
    unknown = [t for t in targets if t not in MODULE_TARGETS]
    if unknown:
        print(f"unknown module target(s): {', '.join(unknown)}; have: "
              f"{', '.join(sorted(MODULE_TARGETS))}; groups: {', '.join(GROUPS)}",
              file=sys.stderr)
        return 2
    directory = resolve_config_dir(DEFAULT_CONFIG_SET, repo)
    if isinstance(directory, int):
        return directory
    rc = apply_config_dir(directory)
    if rc:
        return rc
    print("=== bench against the cache in data/ (grid config set); cache misses use the "
          "runtime fallback policy during warmup, without certifying full-grid tuning", flush=True)
    return _run_bench(args, targets, repo, directory, level="module")


def _resolve_gpus(spec: str) -> list[int]:
    """`--gpus 4` -> [0,1,2,3]; `--gpus 0,3` -> [0,3]; `--gpus all` -> every visible device."""
    if spec == "all":
        import torch
        return list(range(torch.cuda.device_count()))
    if "," in spec:
        return [int(x) for x in spec.split(",") if x.strip() != ""]
    count = int(spec)
    return list(range(count))


def build_parser() -> argparse.ArgumentParser:
    """The whole CLI, built but not run.

    Separate from :func:`main` so a test can parse a command line without executing it -- which
    is what lets tests/layout/test_cli_documented_commands.py check that every `miniworld-engine ...`
    line in the README and the docs is a command this parser accepts. The module docstring used
    to advertise `miniworld-engine bench all`; the subcommands are `bench_kernel` and
    `bench_module`, and there has never been a `bench`.
    """
    parser = argparse.ArgumentParser(prog="miniworld-engine", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # The top level is what a USER does: make this card's cache, and measure. Everything else --
    # the pieces `build` runs internally, and the checks the test suite already runs -- lives under
    # `dev`, so `--help` answers "what can I do with this" instead of listing the implementation.
    #
    #   build         make this GPU's autotune cache (decompose -> run -> merge, in one command)
    #   bench_kernel  measure one kernel
    #   bench_module  measure one module
    #   dev capture   shards without merging: the older two-step path, superseded by
    #                 `build --per-module` (which reaches 91 kernels where driving modules reaches
    #                  48) and kept only for its shape-ladder overrides
    #   dev merge     fold shards in by hand -- `build` does this itself; this is for shards from
    #                 another machine, or a re-merge after a cache-key scheme bump
    #   dev audit     the build-system checks. tests/registry/test_registry_complete.py,
    #                 test_declared_dtype_coverage.py and test_spread_shape_key.py already drive
    #                 these; the command adds only `--shards`, i.e. evidence from a real build
    #                 that a CPU test cannot have.
    sub = parser.add_subparsers(dest="command", required=True)
    dev_parser = sub.add_parser(
        "dev", help="build internals and build-system checks (not needed for normal use)")
    dev = dev_parser.add_subparsers(dest="dev_command", required=True)

    cap = dev.add_parser("capture", help="write shards without merging (see `build`)")
    cap.add_argument("target", help=f"module target or group ({', '.join(GROUPS)})")
    cap.add_argument("--gpus", default="all", help="count, comma list, or 'all' (default: all)")
    cap.add_argument("--shards", default="~/.cache/miniworld-shards", help="where shards go")
    cap.add_argument("--impl", default="miniworld",
                     help="implementations=[...]. Use miniworld: for triangle_multiplication, "
                          "'triton' builds the fp32 dtv1 baseline and captures nothing")
    cap.add_argument("--no-sweep-dispatch", action="store_true",
                     help="capture only the branch this card picks (leaves the other uncached)")
    cap.add_argument("--resume", action="store_true", help="skip shards that already exist")
    cap.add_argument("--seq-lens", default="", help="comma list overriding the seq_len ladder")
    cap.add_argument("--d-pairs", default="", help="comma list overriding the d_pair ladder")
    cap.set_defaults(func=cmd_capture)

    prn = dev.add_parser("prune-cache",
                         help="empty $TRITON_CACHE_DIR -- the build artifact, not the output")
    prn.add_argument("--dry-run", action="store_true", help="say what would go, remove nothing")
    prn.set_defaults(func=lambda a: _empty_triton_cache(a.dry_run))

    mrg = dev.add_parser("merge", help="fold shards into the in-repo cache by hand")
    mrg.add_argument("--shards", default="~/.cache/miniworld-shards", help="dir holding the shards")
    mrg.add_argument("--gpu", default="", help="cache key; defaults to this machine's GPU")
    mrg.add_argument("--top-k", type=int, default=5, help="configs kept per bucket")
    mrg.set_defaults(func=cmd_merge)

    cst = dev.add_parser("cache-status",
                         help="up-front staleness scan of the committed caches (no GPU); "
                              "exits non-zero if any is grid/scheme-stale")
    cst.add_argument("--gpu", default="", help="substring filter on the GPU key (e.g. A100)")
    cst.add_argument("--verbose", action="store_true", help="list OK caches too")
    cst.set_defaults(func=cmd_cache_status)

    spb = dev.add_parser("space-backfill",
                         help="recover, from git history, WHICH configs each cached entry was "
                              "already searched with, so a rebuild measures only the difference "
                              "(dry-run unless --apply)")
    spb.add_argument("--apply", action="store_true", help="write the recovered spaces")
    spb.set_defaults(func=cmd_space_backfill)

    bkf = dev.add_parser("cache-backfill",
                         help="recover driver_identity for pre-field caches from git history "
                              "(dry-run unless --apply)")
    bkf.add_argument("--apply", action="store_true", help="write the recovered hashes")
    bkf.set_defaults(func=cmd_cache_backfill)

    bld = sub.add_parser("build", help="build the cache by driving the production modules")
    bld.add_argument("case", nargs="?", default="all",
                     help="build case, a stack (trunk / diffusion -- half the model, the same "
                          "per-op sweep `all` runs narrowed to the kernels that half launches), "
                          "or 'all' (the default). With --per-op it is a kernel name "
                          "from registry.csv instead -- two name spaces, because a case drives a "
                          "module and an op is one kernel. Neither is a BENCH target; see "
                          "`bench_kernel` / `bench_module`.")
    bld.add_argument("config_type", nargs="?", default=DEFAULT_CONFIG_SET,
                     help="config set: a directory of <op>.csv files, or a short name resolving to configs/<name> (e.g. accuracy). Every kernel's grid comes from here.")
    bld.add_argument("--shards", default="~/.cache/miniworld-build", help="dir for the shards")
    # Filling the gaps is what a build IS. It was opt-in, and the two ways to get a complete
    # cache were `--fill-gaps` (bench only the missing keys) and `--rebuild-cached` (re-measure
    # every config the cache already holds) -- so the cheap, correct thing needed a flag and the
    # ruinous thing was one word away. It cost 5h14m once: `layernorm_bwd_split` re-timed 64
    # tuned units to fill 3 missing keys.
    bld.add_argument("--fill-gaps", action="store_true",
                     help="accepted and ignored -- this is the default now. Kept so the job "
                          "scripts that pass it keep working")
    bld.add_argument("--rebuild", action="store_true",
                     help="re-measure EVERY config of every key, including the ones the committed "
                          "cache already holds. The default builds only what is missing, so reach "
                          "for this only when the measurements themselves are suspect (a new "
                          "triton, a changed bench setting) -- not to fill a gap")
    bld.add_argument("--gpus", default="all", help="count, comma list, or 'all'")
    bld.add_argument("--compile-jobs", type=int, default=0, help="0 = one per core")
    bld.add_argument("--unit-timeout-seconds", type=float, default=7200.0,
                         help="maximum wall seconds per cache-build unit, including compilation "
                              "(default 7200); timed-out units release their GPU slot and retry on resume")
    bld.add_argument("--units-per-gpu", type=int, default=1,
                     help="units to run on each card at once (default 1). A unit alternates "
                          "between compiling and measuring; >1 lets one unit's compile overlap "
                          "another's measurement. Units sharing a card never measure at the "
                          "same time.")
    bld.add_argument("--keep-ir", action="store_true",
                     help="let triton keep every IR level in its cache. Off, the build writes "
                          "only the cubin and metadata a launch needs: 71 KB an entry instead of "
                          "187, which was 40 GB over one rebuild.")
    # The three below default ON. They were opt-in while they were being trusted, and every build
    # that has run since has passed all three -- so the flag that had to be remembered was the one
    # that made the build correct and cheap, and forgetting it was silent. Off is now the thing you
    # ask for, which is the right way round for a default nobody has reason to change.
    bld.add_argument("--predict-unusable", action=argparse.BooleanOptionalAction, default=True,
                     help="probe a slice of each round first and skip the configs the probes "
                          "prove cannot pay off -- over the card's shared memory, or past the "
                          "compile budget. Fitted and validated per kernel; a kernel neither "
                          "model describes compiles its whole grid. 9-29%% of a searched space "
                          "could not run on the card it was searched for. Default on; "
                          "--no-predict-unusable compiles everything.")
    bld.add_argument("--bench-clear-mb", type=int, default=0,
                     help="MB zeroed before each timed iteration (0 = triton's 256, which is 40x "
                          "an A6000's L2 and 97%% of a bench iteration). Set with --bench-rep-ms: "
                          "alone it buys more iterations, not less time.")
    bld.add_argument("--bench-rep-ms", type=int, default=25,
                     help="ms of measurement per config (0 = triton's 100). 16 MB at 10 ms was "
                          "7x cheaper than the default with 30%% more samples, on one kernel. "
                          "Default 25: bench is 97%% of a unit's wall time, so this number is "
                          "most of what a build costs.")
    bld.add_argument("--pin-cores", action=argparse.BooleanOptionalAction, default=True,
                     help="give each unit slot its own cores instead of pooling the node's. A "
                          "unit that is MEASURING otherwise competes with every other unit's "
                          "compile workers, and the measurement is what a build produces. "
                          "Default on; --no-pin-cores pools them.")
    # Ownership decides the default: see _should_prune. The pair is mutually exclusive so a
    # script cannot say both and silently get one of them.
    _prune = bld.add_mutually_exclusive_group()
    _prune.add_argument("--prune-cache", action="store_true",
                        help="after a successful merge, empty $TRITON_CACHE_DIR. This is the "
                             "DEFAULT whenever $TRITON_CACHE_DIR is set -- the flag remains for "
                             "the job scripts that pass it -- and it is refused when unset, "
                             "because ~/.triton/cache is shared with everything else here.")
    _prune.add_argument("--keep-triton-cache", action="store_true",
                        help="keep $TRITON_CACHE_DIR after the merge, for re-measuring "
                             "unchanged kernels where the warm cache skips the recompile.")
    # Default ON. A shard with entries is a finished measurement, and re-running its unit is the
    # same disaster as re-tuning a cached kernel: the plain command is the one people run after a
    # build is killed, and it must not throw away the hours that survived.
    bld.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                     help="skip units whose shard already has entries (default: on; "
                          "--no-resume re-runs them)")
    bld.add_argument("--per-op", action="store_true",
                     help="work item = (op, shape bucket) driven through its registry driver, "
                          "instead of (case, dims, length, mode) driving a whole module. No "
                          "redundancy: each (op, bucket) is tuned exactly once. This is an explicit "
                          "diagnostic sweep outside the default model plan; use it to build "
                          "one kernel by name. The positional takes one kernel name or a "
                          "comma list of them, and a list is ONE sweep -- one import, one GPU "
                          "pool, one interleaved work list.")
    bld.add_argument("--per-module", action="store_true",
                     help="force the module-unit decomposition. This is the DEFAULT for "
                          "`build all`: the units come from registry_module.csv, and `dev derive` "
                          "runs the same enumeration to write registry_kernel.csv, so what the "
                          "build produces is stated in a file before it starts.")
    bld.add_argument("--strict", action="store_true",
                     help="fail without merging if ANY unit failed (default: merge what "
                          "succeeded and report the holes)")
    bld.add_argument("--rebuild-cached", dest="rebuild", action="store_true",
                     help=argparse.SUPPRESS)   # the old spelling of --rebuild
    bld.add_argument("--reclaim", action="store_true",
                     help="first delete claims left by a killed build (they are otherwise "
                          "skipped silently forever). Do NOT use while another build runs "
                          "against the same --shards dir.")
    bld.set_defaults(func=cmd_build)

    def _bench_common(parser_) -> None:
        """Options both bench subcommands share: how to run the bench, and the pre-bench build."""
        parser_.add_argument("--impl", default="all",
                             help="comma list of implementation names, or 'all' (default) for "
                                  "every implementation the target defines")
        # Forwarded to the pre-bench build. Same defaults as `build` so the two agree.
        parser_.add_argument("--shards", default="~/.cache/miniworld-build",
                             help="dir for the shards")
        parser_.add_argument("--gpus", default="all", help="count, comma list, or 'all'")
        parser_.add_argument("--compile-jobs", type=int, default=0, help="0 = one per core")
        parser_.add_argument("--unit-timeout-seconds", type=float, default=7200.0,
                             help="maximum wall seconds per cache-build unit, including compilation "
                                  "(default 7200); timed-out units release their GPU slot and retry on resume")
        parser_.add_argument("--resume", action="store_true",
                             help="pre-bench build skips units whose shard already has entries")
        parser_.add_argument("--no-build", action="store_true",
                             help="bench against the cache already in data/, skipping the "
                                  "pre-bench build (use after `build all` + a clean `audit`)")
        # bench.py's config defaults to seq_len, so without this every run swept one axis and the
        # d_pair half of the matrix -- which docs/benchmarks.md and the README both call for, and
        # which is where the width-dependent kernels separate -- could only be reached by
        # invoking bench.py directly.
        parser_.add_argument("--sweep-axis", default="seq_len", choices=("seq_len", "d_pair"),
                             help="which axis to sweep (default: seq_len)")
        # Default "auto" picks the empirically-best regime per (mode, module): inference is
        # launch-bound for the small/many-launch modules so it captures a manual CUDA graph, while
        # training is backward-dominated (compute-bound) so it stays compile-only (a graph removes no
        # meaningful launch overhead there and its copy/replay only adds cost). swa_atom_attention is
        # launch-bound in its backward too, so it takes manual in both modes. Measured across every
        # module at the fixed real shape; see BenchConfig.cudagraph. Pass an explicit value to force
        # one regime for all runs (e.g. `--cudagraph manual` to reproduce the older committed tables).
        parser_.add_argument("--cudagraph", default="auto",
                             choices=("disabled", "manual", "graphed", "auto"),
                             help="CUDA-graph mode (default: auto -- inference=manual, "
                                  "training=compile-only, per measured best)")
        parser_.add_argument("--compile", default="true", choices=("true", "false"),
                             help="torch.compile the module under test (default: true)")
        # Passed to the child through the ENVIRONMENT, not argv: settings.compile_wrap is read
        # when the kernel modules import, which happens in bench.py's header, before anything
        # reads its Hydra config. See settings._compile_wrap_from_env.
        parser_.add_argument("--compile-wrap", default="",
                             choices=("", "disable", "custom_op"),
                             help="how kernel entry points are exposed to torch.compile "
                                  "(default: leave settings alone)")

    flash = dev.add_parser(
        "install-flash",
        help="install the FlashAttention line this card can run (FA4 on sm90+, FA2 on sm80+)")
    flash.add_argument("--arch", default="",
                       help="skip detection and install for this arch, e.g. sm80")
    flash.add_argument("--dry-run", action="store_true", help="print the command, install nothing")
    flash.set_defaults(func=cmd_install_flash)

    bkt = dev.add_parser("buckets",
                         help="measure which declared widths actually reach each kernel's cache "
                              "key, so the build stops re-timing one bucket N times (needs a GPU)")
    bkt.add_argument("--ops", default="", help="comma-separated ops; default every driven op")
    bkt.add_argument("--out", default="", help="where to write the evidence; defaults in-repo")
    bkt.set_defaults(func=cmd_buckets)

    cal = dev.add_parser("callers",
                         help="record which kernel each production case dispatches into, so the "
                              "plan can be checked against production without a replay (needs a GPU)")
    cal.add_argument("--out", default="", help="where to write it; defaults in-repo")
    cal.set_defaults(func=cmd_callers)

    cov = dev.add_parser("coverage",
                         help="compare the built cache against registry_kernel.csv -- the offline "
                              "replacement for `audit --replay` (no GPU, no module run)")
    cov.add_argument("--arch", default="sm86", help="which derived arch to compare against")
    cov.add_argument("--gpu", default="NVIDIA RTX A6000 (sm86)",
                     help="cache key of the card whose entries to check")
    cov.set_defaults(func=cmd_coverage)

    der = dev.add_parser("derive",
                         help="derive registry_kernel.csv from registry_module.csv by running the "
                              "modules on fake tensors (needs a GPU present, runs no kernel)")
    der.add_argument("--arch", default="",
                     help="derive FOR this arch (sm86/sm90/sm100) instead of the card this runs "
                          "on; nothing is launched, so any card can derive any arch")
    der.add_argument("--per-unit", default="",
                     help="also write, per unit, the keys it produced -- the evidence for which "
                          "axis of the sweep earns its units")
    der.add_argument("--out", default="",
                     help="write here instead of the in-repo registry_kernel.csv")
    der.set_defaults(func=cmd_derive)
    der.add_argument("--workers", type=int, default=1,
                     help="isolated derivation processes sharing the allocated GPU")

    aud = dev.add_parser("audit",
                         help="verify the build system and the shipped cache's coverage")
    aud.add_argument("--gpu", default="", help="cache key to audit; defaults to this machine's GPU")
    aud.add_argument("--shards", nargs="*", default=[],
                     help="shard dirs from real builds, for reachability evidence")
    aud.add_argument("--verbose", action="store_true", help="also print OK findings")
    aud.add_argument("--replay", action="store_true",
                     help="instead of the static checks, drive the build matrix against the "
                          "finished cache and report every lookup it did not serve (needs a GPU)")
    aud.set_defaults(func=cmd_audit)

    bk = sub.add_parser("bench_kernel",
                        help="build the cache, then bench ONE kernel-level target")
    bk.add_argument("target", help=f"kernel target, or 'all' "
                                   f"({', '.join(sorted(KERNEL_TARGETS))})")
    bk.add_argument("config_type", nargs="?", default=None,
                    help="config set: a directory of <op>.csv files, or a short name resolving to configs/<name> (e.g. accuracy). Every kernel's grid comes from here.")
    _bench_common(bk)
    bk.set_defaults(func=cmd_bench_kernel)

    bm = sub.add_parser("bench_module",
                        help="build the cache, then bench a module-level target (no config arg)")
    bm.add_argument("target", help=f"module target or group "
                                   f"({', '.join(sorted(MODULE_TARGETS))}; "
                                   f"groups: {', '.join(GROUPS)})")
    _bench_common(bm)
    # Only the module bench takes a mode: a module genuinely runs in both regimes. A kernel target's
    # name already says which one it is (`*_bwd` or not), so offering the choice there can only
    # produce a wrong answer -- a forward re-timed under training, or a backward asked for inference.
    bm.add_argument("--mode", default="inference", choices=("inference", "training"))
    bm.set_defaults(func=cmd_bench_module, config_type=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
