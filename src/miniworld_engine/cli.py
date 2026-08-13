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
    ops = sorted({w[0] for w in written})
    print(f"merged {len(ops)} ops / {len(written)} buckets from {len(paths)} shards into {gpu!r}")
    for op in ops:
        print(f"  {op}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    targets = GROUPS.get(args.what, (args.what,))
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        print(f"unknown target(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    rc = 0
    for target in targets:
        cmd = [sys.executable, "-u", "benchmarks/runners/bench.py", f"kernel={target}",
               f"implementations=[{args.impl}]", f"mode={args.mode}", "metric=time"]
        if TARGETS[target]:
            cmd.extend(TARGETS[target].split())
        print(f"=== bench {target}", flush=True)
        rc |= subprocess.run(cmd, cwd=repo, check=False).returncode  # noqa: S603
    return rc


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

    ben = sub.add_parser("bench", help="run benchmarks")
    ben.add_argument("what", help=f"target or group ({', '.join(GROUPS)})")
    ben.add_argument("--impl", default="miniworld")
    ben.add_argument("--mode", default="inference", choices=("inference", "training"))
    ben.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
