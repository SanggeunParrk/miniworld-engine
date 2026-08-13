#!/usr/bin/env python3
"""Plan autotune-capture shards at ONE SHAPE per shard, instead of one sweep per shard.

``build_autotune_cache.py plan`` emits one command per (target, mode, axis), and that command
then walks every shape in the sweep. For the heavy targets that is the whole problem: a single
``triangle_attention`` sweep took 1-2 HOURS, all of it host-side Triton compilation with the GPU
sitting at 0% utilisation, and nothing about holding a GPU makes it go faster.

Compilation is the serial resource, so the only axis that actually parallelises is *processes*.
Splitting a 6-shape sweep into 6 single-shape shards turns one 90-minute task into six that run
at once. The compile work is duplicated across them (a shape change does not change a kernel's
constexprs, so the shapes in one sweep largely share compiled binaries), but they are competing
for idle CPUs, not for the scarce thing — and Triton's on-disk cache is shared, so a shard that
starts later can hit what an earlier one already compiled.

Shards merge exactly as before: each writes its own json and a single-writer merge folds them in.

    python submits/plan_per_shape.py --targets triangle_attention,bias_only_attention \
        --shards $HOME/.cache/mwk-shards/a6000 --impl miniworld > cmds.txt
    CMDFILE=cmds.txt sbatch --array=1-$(wc -l < cmds.txt)%10 --gres=gpu:A6000:1 \
        submits/build_shard_ampere.sbatch
"""

from __future__ import annotations

import argparse
from pathlib import Path

#: Same shape ladder the Ampere capture uses, split into its individual points.
SEQ_LENS = (384, 512, 640, 768, 896, 1024)
D_PAIRS = (128, 256, 512)
ATOM_SEQ_LENS = (128, 256, 384)
ATOM_D_PAIRS = (16, 32, 64)
#: Targets whose module needs a different shape ladder or extra hydra args.
ATOM_TARGETS = {"augmented_attention_atom"}
EXTRA = {"conditioned_transition": "precision=32 d_single_token=384"}


def commands(target: str, mode: str, shards: Path, impl: str) -> list[str]:
    atom = target in ATOM_TARGETS
    seq_lens = ATOM_SEQ_LENS if atom else SEQ_LENS
    d_pairs = ATOM_D_PAIRS if atom else D_PAIRS
    fixed_seq = seq_lens[-1 if atom else 0]
    extra = EXTRA.get(target, "")
    out = []

    def line(axis: str, tag: str, shape: str) -> str:
        shard = shards / f"{target}-{mode}-{axis}-{tag}-{impl}.json"
        return (
            f"kernel={target} implementations=[{impl}] mode={mode} metric=time "
            f"compile=false cudagraph=manual sweep_axis={axis} name_suffix=build_{tag} "
            f"+autotune_shard={shard} mask_prob=0.0 {shape}"
            + (f" {extra}" if extra else "")
        )

    # One shard per seq_len point; d_pair is held at the ladder's first value by the runner.
    for n in seq_lens:
        out.append(line(
            "seq_len", f"L{n}",
            f"min_seq_len={n} max_seq_len={n} seq_len_step=128 "
            f"d_pair_values=[{d_pairs[0]}] sweep_seq_len={fixed_seq}",
        ))
    # One shard per d_pair point, at the sweep's fixed seq_len.
    for d in d_pairs:
        out.append(line(
            "d_pair", f"d{d}",
            f"min_seq_len={fixed_seq} max_seq_len={fixed_seq} seq_len_step=128 "
            f"d_pair_values=[{d}] sweep_seq_len={fixed_seq}",
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", required=True, help="comma list of bench kernel targets")
    ap.add_argument("--modes", default="inference,training")
    ap.add_argument("--shards", required=True, help="dir where shard files will be written")
    ap.add_argument("--impl", default="miniworld",
                    help="implementations=[...]; use miniworld — for trimul, `triton` builds the "
                         "fp32 dtv1 baseline instead of the fused kernels and captures nothing")
    args = ap.parse_args()

    shards = Path(args.shards)
    for target in args.targets.split(","):
        for mode in args.modes.split(","):
            for cmd in commands(target, mode, shards, args.impl):
                print(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
