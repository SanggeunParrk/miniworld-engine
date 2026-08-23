"""``miniworld-engine`` command line: capture autotune caches and run benchmarks.

    miniworld-engine capture all              # every kernel, every dispatch branch
    miniworld-engine capture pairformer       # one module's kernels
    miniworld-engine capture transition --gpus 4
    miniworld-engine bench all

Everything the run depends on is an argument. The engine used to take these as environment
variables, which meant a run's behaviour lived in shell state that nothing recorded: a capture that
benched the PyTorch reference and reported it as ours, and one that skipped every kernel on the
losing side of a dispatch decision, both looked like successful runs. Arguments are echoed into
each shard, so a cache can be traced back to the command that built it.

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

#: Bench targets that make up "all", with the hydra kernel name and any extra bench args.
TARGETS: dict[str, str] = {
    "transition": "",
    "triangle_attention": "",
    "bias_only_attention": "",
    "triangle_multiplication": "",
    "conditioned_transition": "precision=32 d_single_token=384",
    "adaptive_layernorm": "",
    "augmented_attention_token": "",
    "augmented_attention_atom": "",
}

#: Named groups so `capture pairformer` means something. A group is just a set of targets.
GROUPS: dict[str, tuple[str, ...]] = {
    "all": tuple(TARGETS),
    "pairformer": (
        "transition", "triangle_attention", "bias_only_attention", "triangle_multiplication",
    ),
    "diffusion": ("conditioned_transition", "adaptive_layernorm",
                  "augmented_attention_token", "augmented_attention_atom"),
    "attention": ("triangle_attention", "bias_only_attention",
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
    "gate_backend": (("fused", "split"), ("bias_only_attention", "triangle_attention"),
                     ("inference", "training")),
    # inference LN+proj concat fusion (layernorm_linear) -- consulted on the inference path only
    "infer_concat": ((True, False), ("bias_only_attention", "triangle_attention"),
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
            f"kernel={self.target}",
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
        if TARGETS[self.target]:
            args.extend(TARGETS[self.target].split())
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
            proc = subprocess.run(  # noqa: S603
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
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    return env


def cmd_capture(args: argparse.Namespace) -> int:
    import concurrent.futures as cf

    repo = Path(__file__).resolve().parents[2]
    targets = GROUPS.get(args.what, (args.what,))
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        print(f"unknown target(s): {', '.join(unknown)}\n"
              f"targets: {', '.join(TARGETS)}\ngroups : {', '.join(GROUPS)}", file=sys.stderr)
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
        print("no CUDA device visible; pass --gpu \"<gpu key>\" to merge from a login node",
              file=sys.stderr)
        return 2

    written = capture.merge_shards(paths, top_k=args.top_k, gpu=gpu, only_ops=None)
    if capture._MERGE_SKIPPED:
        print(f"WARNING: {len(capture._MERGE_SKIPPED)} shard(s) were unreadable, so a whole "
              f"unit's measurements are MISSING from the cache:", file=sys.stderr)
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
    given = Path(config_type).expanduser()
    candidates = [given] if given.is_absolute() or given.parts[:1] == (CONFIG_ROOT,) else [
        repo / CONFIG_ROOT / config_type, given,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    have = sorted(d.name for d in (repo / CONFIG_ROOT).glob("*") if d.is_dir())
    print(f"unknown config set {config_type!r}; have: {', '.join(have) or '(none)'}\n"
          f"a config set is a directory of <op>.csv under {CONFIG_ROOT}/ "
          f"(the default is {DEFAULT_CONFIG_SET!r})", file=sys.stderr)
    return 2


def apply_config_dir(directory: Path) -> int:
    """Select ``directory`` as the config set, then import the kernels. Non-zero if unusable.

    ORDER MATTERS. Triton's ``Autotuner.__init__`` keeps the list it is handed only when that list
    is non-empty; hand it an empty one and it substitutes ``[Config({})]`` of its own and drops the
    reference, so filling the list afterwards has no effect and every kernel launches with no tile
    at all (``dynamic_func() missing ... 'BLOCK_M1'``). So the directory has to be set BEFORE the
    kernel modules import and their decorators run.
    """
    from miniworld_engine.autotune.configs import missing_ops, registered_ops, use_config_dir
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


#: bench target -> the build case(s) that drive the same kernels.
#:
#: The name spaces are not the same and never were. ``benchmarks/runners/bench.py`` has two families
#: of entry point -- ``bench_kernel_*`` (17 kernel-level targets) and ``bench_<module>`` (the
#: module-level ones) -- while ``build`` names the 23 cases that drive the production modules. That
#: gap is why ``bench adaln_bwd`` was rejected as an unknown target: the CLI only ever listed the
#: module family. A bench that auto-builds has to cross the gap explicitly; deriving it by string
#: match would silently build nothing for every target whose name differs.
#:
#: Kernel-target rows come from what each ``bench_kernel_*`` function actually imports, not from its
#: name. A wrong entry degrades to today's behaviour and says so: the build fills the wrong case and
#: the bench then prints the engine's own per-op "no tuned autotune cache" warning, so it cannot
#: silently produce a fast-looking number from an untuned kernel.
MODULE_BUILD_CASES: dict[str, tuple[str, ...]] = {
    "transition": ("transition",),
    "triangle_multiplication": ("triangle_multiplication",),
    "triangle_multiplication_bidirectional": ("triangle_multiplication_bidir",),
    "triangle_attention": ("triangle_attention_bidir", "triangle_attention_heads"),
    # bench's bias_only_attention target drives kernels.bias_only_attention.triton.main directly;
    # AttentionPairBias is the production module that dispatches those kernels, so its case is what
    # fills their cache.
    "bias_only_attention": ("attention_pair_bias",),
    "conditioned_transition": ("conditioned_transition",),
    "adaptive_layernorm": ("adaptive_layernorm",),
    "augmented_attention_token": ("augmented_attention",),
    "augmented_attention_atom": ("augmented_attention",),
}

KERNEL_BUILD_CASES: dict[str, tuple[str, ...]] = {
    "dual_gemm_epil": ("tm1_triton", "triangle_multiplication"),
    "dual_gemm_epil_bwd": ("triangle_multiplication",),
    "gemm_epil": ("layernorm_linear_pair_bias",),
    "gemm_gate": ("tm2_triton",),
    "gate_bwd": ("triangle_multiplication",),
    "transition_b2b": ("transition",),
    "transition_b2b_bwd": ("transition",),
    "layernorm": ("layernorm_lowreg", "layernorm_transpose"),
    "layernorm_bwd": ("layernorm_lowreg", "layernorm_transpose"),
    "ln_mask": ("layernorm_lowreg",),
    "adaln": ("adaptive_layernorm", "layernorm_linear_pair_bias"),
    "adaln_bwd": ("adaptive_layernorm",),
    "tri_attn": ("triangle_attention_bidir", "triangle_attention_heads"),
    "bias_attn": ("attention_pair_bias",),
    "aug_attn": ("augmented_attention",),
    "cond_transition_tail": ("conditioned_transition",),
    # gemm_epil_bwd imports adaln, augmented_attention, bias_only_attention,
    # conditioned_transition, layernorm and layernorm_linear -- it benches the shared GEMM-epilogue
    # backward across all of them, so its cache comes from all of their cases.
    "gemm_epil_bwd": ("adaptive_layernorm", "augmented_attention", "attention_pair_bias",
                      "conditioned_transition", "layernorm_lowreg",
                      "layernorm_linear_pair_bias"),
}


def _merge_built_shards(args: argparse.Namespace, results: list) -> int:
    """Fold this build's shards into the in-repo cache and report the units that failed.

    Split out of the build command so the partial-merge policy is testable without a GPU:
    which units count as bad, whether a bad one blocks the merge, and what the exit code is.
    """
    from miniworld_engine.autotune import capture  # noqa: PLC0415 -- heavy; import at use

    # A unit that skipped a shape this card cannot hold is not a bad unit. It is a permanent,
    # correct answer -- "not on this GPU" -- and counting it as a failure is what made a resumed
    # job that picked up only the leftover OOM shapes report "0 ok, 9 failed" and refuse to merge.
    bad = [r for r in results if (r["rc"] != 0 or not r["ops"]) and not r.get("skipped")]
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
        print(f"build produced {len(bad)} bad unit(s) of {len(results)}; their (op, bucket) entries "
              f"will be MISSING from the cache and will fall back to the full grid at runtime:",
              file=sys.stderr)
        for r in bad[:10]:
            print(f"  {r['label']} rc={r['rc']} ops={r['ops']} -> {r['log']}", file=sys.stderr)
        if len(bad) > 10:
            print(f"  ... and {len(bad) - 10} more", file=sys.stderr)
        if getattr(args, "strict", False):
            print("--strict: not merging.", file=sys.stderr)
            return 1
    # The SHARD DIR decides, not this run's tally. With --resume a late job picks up only the
    # leftovers -- which are exactly the units that keep failing -- so gating the merge on "did
    # this run succeed?" left 526 finished shards unmerged while the job reported rc=1.
    if not shards:
        print("no shards to merge.", file=sys.stderr)
        return 1
    written = capture.merge_shards(shards)
    if capture._MERGE_SKIPPED:
        print(f"WARNING: {len(capture._MERGE_SKIPPED)} shard(s) were unreadable, so a whole "
              f"unit's measurements are MISSING from the cache:", file=sys.stderr)
        for sp, why in capture._MERGE_SKIPPED[:10]:
            print(f"  {sp}\n      {why}", file=sys.stderr)
    print(f"=== merged {len(written)} op file(s) into the in-repo cache"
          f"{f' ({len(bad)} unit(s) missing -- run `audit` for the holes)' if bad else ''}",
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
                                resume=args.resume, config_dir=directory)
    return _merge_built_shards(args, results)


def cmd_build(args: argparse.Namespace) -> int:
    """Build the cache. The builder owns decomposition and multi-GPU execution; this only parses."""
    from miniworld_engine.autotune import builder

    # Select the config directory BEFORE cases(), which imports the kernel modules. An op that
    # calls configs_for() with no directory chosen gets triton's substitute Config({}) and can
    # never be refilled, so use_config_dir then refuses with "15 op(s) registered before a config
    # directory was selected" and the whole command dies -- `build` was unusable without
    # MINIWORLD_CONFIG_DIR already exported. `_bench_build_first` has always had these two in the
    # right order; this was a plain ordering slip.
    repo = Path(__file__).resolve().parents[2]
    directory = resolve_config_dir(args.config_type, repo)
    if isinstance(directory, int):
        return directory
    rc = apply_config_dir(directory)
    if rc:
        return rc

    # `build all` on a fresh card must produce a COMPLETE cache with no one curating a list, so
    # the per-op sweep is the default for it: its coverage is declared (registry.csv x level),
    # while driving modules only reaches the kernels some module happens to dispatch to -- 48 of
    # 91 triton kernels, measured, leaving 43 with working drivers untuned and invisible.
    # --per-module asks for the old decomposition, which is still what you want when the question
    # is "does this module's real dispatch path work", not "is every kernel tuned".
    per_op = args.per_op or (args.case == "all" and not args.per_module)
    if per_op:
        # One item per (op, shape bucket) instead of one per module unit. Same harness: the GPU
        # pool, the O_EXCL claims, --resume, the shards and the merge are all unchanged -- only
        # what a work item IS differs. Driving modules re-tunes an op once per unit that reaches
        # it; 3,385 units, 1,950 of them one case, and a single 15,552-config op inside it costs
        # 244 GPU-h of pure re-benching.
        only = None if args.case == "all" else {args.case}
        selected = builder.op_units(only, config_dir=directory)
        if not selected:
            print(f"no triton op with a driver matched {args.case!r}", file=sys.stderr)
            return 2
        print(f"per-op sweep: {len(selected)} (op, shape) items", flush=True)
    else:
        selected = [c for c in builder.cases() if args.case in ("all", c.name)]
        if not selected:
            names = ", ".join(c.name for c in builder.cases())
            print(f"unknown case {args.case!r}; have: all, {names}", file=sys.stderr)
            return 2

    results = builder.build_all(selected, Path(args.shards).expanduser(),
                                _resolve_gpus(args.gpus), args.compile_jobs,
                                resume=args.resume, reclaim=args.reclaim,
                                bench_budget=args.bench_budget, config_dir=directory)
    failed = [r for r in results if r["rc"] != 0]
    empty = [r for r in results if r["rc"] == 0 and not r["ops"]]
    print(f"\n{len(results) - len(failed) - len(empty)} ok, {len(empty)} empty, "
          f"{len(failed)} failed")
    for r in empty + failed:
        print(f"  {'EMPTY' if r in empty else 'FAIL '} {r['label']} -> {r['log']}")
    # THE per-op sweep is the path that actually builds the shipped cache, and it was the one path
    # that never merged: `cmd_build` and `_bench_build_first` both folded their shards in, this
    # returned straight to the shell. A 527-unit sweep therefore finished, wrote 145 GB of shards,
    # and left `data/` untouched -- the build looked complete and shipped nothing. Same policy as
    # the other two: merge what succeeded, name the holes, fail only if nothing succeeded.
    return _merge_built_shards(args, results)


def _bench_cmd(args: argparse.Namespace, target: str, config_dir: Path | None,
               per_target_mode: bool) -> tuple[list[str], dict | None]:
    """The argv and env for one target's bench.py run."""
    # bench.py requires a mode. For a kernel target it is not the caller's to choose: the name
    # says it (`*_bwd` = training, else inference), which is why bench_kernel has no --mode.
    mode = ("training" if target.endswith("_bwd") else "inference") if per_target_mode \
        else args.mode
    cmd = [sys.executable, "-u", "benchmarks/runners/bench.py", f"kernel={target}",
           f"implementations=[{args.impl}]", f"mode={mode}", "metric=time"]
    extra = TARGETS.get(target, "")
    if extra:
        cmd.extend(extra.split())
    env = None
    if config_dir is not None:
        # The child must have the set in its ENVIRONMENT, not on its argv: bench.py's own header
        # imports kernel modules, so anything read inside main() lands after the autotuners have
        # already been handed empty lists. +config_dir is kept so the child can assert they agree.
        cmd.append(f"+config_dir={config_dir}")
        env = {**os.environ, "MINIWORLD_CONFIG_DIR": str(config_dir)}
    return cmd, env


def _run_bench(args: argparse.Namespace, targets: tuple[str, ...], repo: Path,
               config_dir: Path | None = None, *, per_target_mode: bool = False) -> int:
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
    jobs = [(t, *_bench_cmd(args, t, config_dir, per_target_mode)) for t in targets]
    rc = 0

    def run(job: tuple, gpu: int) -> tuple[str, int, str]:
        target, cmd, env = job
        env = {**(env or os.environ), "CUDA_VISIBLE_DEVICES": str(gpu)}
        done = subprocess.run(cmd, cwd=repo, check=False, env=env,  # noqa: S603
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
        import itertools

        with cf.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = {pool.submit(run, job, gpu): job[0]
                       for job, gpu in zip(jobs, itertools.cycle(gpus))}
            for fut in cf.as_completed(futures):
                target, code, out = fut.result()
                print(f"=== bench {target}  (rc={code})", flush=True)
                print(out, flush=True)
                rc |= code
    if len(targets) > 1:
        rc |= _report_coverage(targets, repo, since=started)
    return rc


def _report_coverage(targets: tuple[str, ...], repo: Path, *, since: float = 0.0) -> int:
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
        family = "kernels" if target in KERNEL_BUILD_CASES else "modules"
        base = repo / "benchmarks" / family / target
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
    devices.record(key, {k: (True, "launched by bench") for k in hit})
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
    targets = tuple(KERNEL_BUILD_CASES) if args.what == "all" else (args.what,)
    unknown = [t for t in targets if t not in KERNEL_BUILD_CASES]
    if unknown:
        print(f"unknown kernel target(s): {', '.join(unknown)}; have: all, "
              f"{', '.join(sorted(KERNEL_BUILD_CASES))}", file=sys.stderr)
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
        rc = _bench_build_first(args, targets, repo, KERNEL_BUILD_CASES, "KERNEL_BUILD_CASES")
        if rc:
            return rc
        directory = None
    return _run_bench(args, targets, repo, directory, per_target_mode=True)


def cmd_bench_module(args: argparse.Namespace) -> int:
    """Module-level bench: takes NO config_type.

    A module bench is the production-shaped measurement -- whole module, its own dispatch decisions
    -- so the config space is not the caller's to pick: it is whatever the cache holds. That is
    ``default``, passed here as a constant rather than an argument so the two cannot disagree.
    """
    repo = Path(__file__).resolve().parents[2]
    targets = GROUPS.get(args.what, (args.what,))
    unknown = [t for t in targets if t not in MODULE_BUILD_CASES]
    if unknown:
        print(f"unknown module target(s): {', '.join(unknown)}; have: "
              f"{', '.join(sorted(MODULE_BUILD_CASES))}; groups: {', '.join(GROUPS)}",
              file=sys.stderr)
        return 2
    # A module bench takes no config set, so the build uses the default one -- see the docstring.
    rc = _bench_build_first(args, targets, repo, MODULE_BUILD_CASES, "MODULE_BUILD_CASES")
    if rc:
        return rc
    directory = resolve_config_dir(DEFAULT_CONFIG_SET, repo)
    if isinstance(directory, int):
        return directory
    rc = apply_config_dir(directory)
    if rc:
        return rc
    return _run_bench(args, targets, repo, directory)


def _resolve_gpus(spec: str) -> list[int]:
    """`--gpus 4` -> [0,1,2,3]; `--gpus 0,3` -> [0,3]; `--gpus all` -> every visible device."""
    if spec == "all":
        import torch
        return list(range(torch.cuda.device_count()))
    if "," in spec:
        return [int(x) for x in spec.split(",") if x.strip() != ""]
    count = int(spec)
    return list(range(count))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="miniworld-engine", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="build autotune caches by capturing benched configs")
    cap.add_argument("what", help=f"target or group ({', '.join(GROUPS)})")
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

    mrg = sub.add_parser("merge", help="fold captured shards into the in-repo cache")
    mrg.add_argument("--shards", default="~/.cache/miniworld-shards", help="dir holding the shards")
    mrg.add_argument("--gpu", default="", help="cache key; defaults to this machine's GPU")
    mrg.add_argument("--top-k", type=int, default=5, help="configs kept per bucket")
    mrg.set_defaults(func=cmd_merge)

    bld = sub.add_parser("build", help="build the cache by driving the production modules")
    bld.add_argument("case", nargs="?", default="all", help="kernel case, or 'all'")
    bld.add_argument("config_type", nargs="?", default=DEFAULT_CONFIG_SET,
                     help="config set: a directory of <op>.csv files, or a short name resolving to configs/<name> (e.g. accuracy). Every kernel's grid comes from here.")
    bld.add_argument("--shards", default="~/.cache/miniworld-build", help="dir for the shards")
    bld.add_argument("--gpus", default="all", help="count, comma list, or 'all'")
    bld.add_argument("--compile-jobs", type=int, default=0, help="0 = one per core")
    bld.add_argument("--resume", action="store_true",
                     help="skip units whose shard already has entries")
    bld.add_argument("--bench-budget", type=float, default=0.0,
                     help="abandon a config once one launch exceeds this factor x the round's "
                          "best (0 = off). Post-hoc: it skips the full do_bench of a config that "
                          "is already out of the running, it does not shorten the launch itself.")
    bld.add_argument("--per-op", action="store_true",
                     help="work item = (op, shape bucket) driven through its registry driver, "
                          "instead of (case, dims, length, mode) driving a whole module. No "
                          "redundancy: each (op, bucket) is tuned exactly once. This is the "
                          "DEFAULT for `build all`.")
    bld.add_argument("--per-module", action="store_true",
                     help="force the module-unit decomposition for `build all`. Reaches only the "
                          "kernels a module dispatches to, so it does not produce a complete "
                          "cache; use it to exercise real dispatch paths, not to tune.")
    bld.add_argument("--strict", action="store_true",
                     help="fail without merging if ANY unit failed (default: merge what "
                          "succeeded and report the holes)")
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
        parser_.add_argument("--resume", action="store_true",
                             help="pre-bench build skips units whose shard already has entries")

    bk = sub.add_parser("bench_kernel",
                        help="build the cache, then bench ONE kernel-level target")
    bk.add_argument("what", help=f"kernel target, or 'all' "
                                 f"({', '.join(sorted(KERNEL_BUILD_CASES))})")
    bk.add_argument("config_type", nargs="?", default=None,
                    help="config set: a directory of <op>.csv files, or a short name resolving to configs/<name> (e.g. accuracy). Every kernel's grid comes from here.")
    _bench_common(bk)
    bk.set_defaults(func=cmd_bench_kernel)

    bm = sub.add_parser("bench_module",
                        help="build the cache, then bench a module-level target (no config arg)")
    bm.add_argument("what", help=f"module target or group "
                                f"({', '.join(sorted(MODULE_BUILD_CASES))}; "
                                f"groups: {', '.join(GROUPS)})")
    _bench_common(bm)
    # Only the module bench takes a mode: a module genuinely runs in both regimes. A kernel target's
    # name already says which one it is (`*_bwd` or not), so offering the choice there can only
    # produce a wrong answer -- a forward re-timed under training, or a backward asked for inference.
    bm.add_argument("--mode", default="inference", choices=("inference", "training"))
    bm.set_defaults(func=cmd_bench_module, config_type=None)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
