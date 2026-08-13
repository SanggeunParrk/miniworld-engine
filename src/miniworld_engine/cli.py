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

#: Device-calibrated dispatch switches a build must sweep BOTH sides of. The card picks one side
#: for the shapes swept here, so the other side's kernels never fire and never get captured — yet
#: they still run in production at other shapes, and would then have no cached configs at all.
GATE_PINS = ("fused", "split")


@dataclasses.dataclass(frozen=True)
class Job:
    """One capture unit: a bench invocation that writes exactly one shard."""

    target: str
    mode: str
    axis: str
    gate: str | None
    shard: Path

    def bench_args(self, impl: str) -> list[str]:
        shapes = SHAPES.get(self.target, SHAPES["default"])
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
            f"name_suffix=build_{self.gate or impl}",
            f"+autotune_shard={self.shard}",
            "mask_prob=0.0",
            f"sweep_seq_len={fixed_seq}",
            *shape.split(),
        ]
        if TARGETS[self.target]:
            args.extend(TARGETS[self.target].split())
        if self.gate:
            args.append(f"+pin_gate_backend={self.gate}")
        return args

    @property
    def label(self) -> str:
        pin = f" gate={self.gate}" if self.gate else ""
        return f"{self.target} {self.mode}/{self.axis}{pin}"


def build_jobs(targets: tuple[str, ...], shard_dir: Path, sweep_gate: bool) -> list[Job]:
    jobs = []
    for target in targets:
        # Only the bias_only gate epilogue has a second branch worth sweeping; pinning it for
        # kernels that never consult it would just duplicate identical work.
        gates = GATE_PINS if (sweep_gate and target in
                              ("bias_only_attention", "triangle_attention")) else (None,)
        for gate in gates:
            for mode in ("inference", "training"):
                for axis in ("seq_len", "d_pair"):
                    name = f"{target}-{mode}-{axis}" + (f"-gate{gate}" if gate else "")
                    jobs.append(Job(target, mode, axis, gate, shard_dir / f"{name}.json"))
    return jobs


def run_worker(device: int, queue: Queue, impl: str, repo: Path, log_dir: Path) -> list[dict]:
    """Drain the shared queue on one GPU, as subprocesses so a crash cannot take the fleet down."""
    results = []
    while True:
        try:
            job = queue.get_nowait()
        except Empty:
            return results
        log = log_dir / f"gpu{device}-{job.shard.stem}.log"
        cmd = [sys.executable, "-u", "benchmarks/runners/bench.py", *job.bench_args(impl)]
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

    jobs = build_jobs(tuple(targets), shard_dir, sweep_gate=not args.no_sweep_dispatch)
    if args.resume:
        jobs = [j for j in jobs if not j.shard.exists()]
    if not jobs:
        print("nothing to do (every shard already exists; drop --resume to rebuild)")
        return 0

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
        futures = [pool.submit(run_worker, dev, queue, args.impl, repo, log_dir) for dev in gpus]
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
    cap.set_defaults(func=cmd_capture)

    ben = sub.add_parser("bench", help="run benchmarks")
    ben.add_argument("what", help=f"target or group ({', '.join(GROUPS)})")
    ben.add_argument("--impl", default="miniworld")
    ben.add_argument("--mode", default="inference", choices=("inference", "training"))
    ben.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
