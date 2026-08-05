#!/usr/bin/env python3
"""Dedicated parallel builder for the in-repo Triton autotune cache.

The capture itself happens inside a REAL ``benchmarks/runners/bench.py`` run (the intended path
per ``autotune/capture.py`` — never hand-replicate kernel launches). This tool adds the two things
a *parallel* build needs and nothing else:

  * SHARDING — each parallel bench run dumps its timings to its OWN shard file (pass
    ``autotune_shard=<file>`` to bench.py); parallel jobs therefore never race on the committed
    ``src/miniworld_engine/autotune/data`` tree, and no environment variable is involved.
  * MERGE — a single-writer fold of all shard files into the in-repo cache, optionally restricted
    to the ops you actually rebuilt (so an existing good cache is left untouched).

Typical flow (see ``submits/build_autotune_cache.sbatch`` for the slurm launcher):

    # 1) fan out capture jobs, each writing one shard (parallel, isolated):
    python benchmarks/runners/bench.py kernel=transition implementations=[triton] \
        mode=inference metric=time compile=false cudagraph=manual sweep_axis=seq_len \
        autotune_shard=$SHARDS/transition-inference-seq.json \
        mask_prob=0.0 min_seq_len=256 max_seq_len=512 seq_len_step=128 \
        d_pair_values=[64,128,256,512] sweep_seq_len=384

    # 2) fold the shards into the committed cache (only the ops you built):
    python submits/build_autotune_cache.py merge --shards $SHARDS \
        --gpu "NVIDIA H100 80GB HBM3 (sm90)" \
        --only-ops transition_split_fwd,transition_split_bwd,transition_fold_swiglu
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _shard_paths(spec: str) -> list[str]:
    """A shard spec is either a glob or a directory (recursively globbed for *.json)."""
    if any(ch in spec for ch in "*?["):
        return sorted(glob.glob(spec, recursive=True))
    return sorted(glob.glob(str(Path(spec) / "**" / "*.json"), recursive=True))


def cmd_merge(args: argparse.Namespace) -> int:
    from miniworld_engine.autotune import capture
    from miniworld_engine.autotune.cache import gpu_key

    paths = _shard_paths(args.shards)
    if not paths:
        print(f"!! no shard files under {args.shards!r}", flush=True)
        return 1
    gpu = args.gpu or gpu_key()
    if gpu == "cpu":
        print("!! no --gpu given and no CUDA device — pass --gpu \"<gpu_key>\" on a login node",
              flush=True)
        return 2
    only = {o for o in args.only_ops.split(",") if o} or None
    written = capture.merge_shards(paths, top_k=args.top_k, gpu=gpu, only_ops=only)
    for op, bk, n, fp in written:
        print(f"  merged {op} [{bk}] ({n} configs)", flush=True)
    ops = sorted({w[0] for w in written})
    print(f"MERGED {len(ops)} ops / {len(written)} buckets from {len(paths)} shards "
          f"into gpu={gpu!r}", flush=True)
    print("  ops: " + ", ".join(ops), flush=True)
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Print one bench.py capture command per shard for a (target × mode × axis × impl) matrix.
    Feed the lines to a slurm array / xargs; each writes its own shard json under --shards."""
    targets = args.targets.split(",")
    modes = args.modes.split(",")
    axes = args.axes.split(",")
    impls = args.impls.split(",")
    shdir = Path(args.shards)
    for t in targets:
        for mode in modes:
            for axis in axes:
                for impl in impls:
                    shard = shdir / f"{t}-{mode}-{axis}-{impl}.json"
                    extra = f" {args.extra}" if args.extra else ""
                    print(
                        f"kernel={t} implementations=[{impl}] mode={mode} metric=time "
                        f"compile=false cudagraph=manual sweep_axis={axis} "
                        f"name_suffix=build_{impl} +autotune_shard={shard} "
                        f"{args.shapes}{extra}"
                    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("merge", help="fold shard files into the in-repo cache (sole writer)")
    m.add_argument("--shards", required=True, help="dir (recursive) or glob of shard *.json")
    m.add_argument("--only-ops", default="", help="comma list; restrict the write to these op names")
    m.add_argument("--gpu", default="", help="gpu_key to write under (default: current CUDA device)")
    m.add_argument("--top-k", type=int, default=5)
    m.set_defaults(func=cmd_merge)

    p = sub.add_parser("plan", help="print bench.py capture commands (one per shard) for a matrix")
    p.add_argument("--targets", required=True, help="comma list of bench kernel targets")
    p.add_argument("--modes", default="inference,training")
    p.add_argument("--axes", default="seq_len,d_pair")
    p.add_argument("--impls", default="triton")
    p.add_argument("--shards", required=True, help="dir where shard files will be written")
    p.add_argument("--shapes", required=True, help="hydra shape args (min/max_seq_len, d_pair_values,…)")
    p.add_argument("--extra", default="", help="extra hydra args (e.g. precision=32)")
    p.set_defaults(func=cmd_plan)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
