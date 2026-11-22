"""Per-kernel GPU time of one cell (torch.profiler on a CUDA-graph-free loop, L2 flushed): ``prof_cell.py CONFIG --module M --rows R --lengths L --modes eval --backend cuda``."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "paper"))
from paper_common import cfg as cfgmod  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    cfgmod.add_selection_args(p)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()
    from torch.profiler import ProfilerActivity, profile

    from miniworld_engine.measure import config
    from miniworld_engine.measure.speed import make_step, unpadded_masks

    cfg = cfgmod.load(args.config, args)
    dev = torch.device("cuda")
    flush = torch.empty(256 << 20, dtype=torch.int8, device=dev)
    for case in cfgmod.limit(cfg, args.max_cases):
        base = config.build(case, device=dev)
        backend = (args.backend or ["cuda"])[0]
        module = copy.deepcopy(base)
        with unpadded_masks():
            inputs = config.inputs(case, module, device=dev)
        step = make_step(module, inputs, backend=backend, train=case.mode == "train", autocast=config.autocast(case),
                         seed=config.seed(case), compile_mode=None, keep_expanded=config.keeps_expanded(case))
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                flush.zero_()
                step()
            torch.cuda.synchronize()
        print(f"== {case.module} r{case.row} {case.mode} {case.dtype} L{case.length} {backend}")
        rows = [(e.key, e.device_time_total / args.iters, e.count / args.iters) for e in prof.key_averages() if e.device_time_total > 0]
        for k, t, c in sorted(rows, key=lambda r: -r[1])[:14]:
            print(f"   {t:9.1f} us  x{c:4.1f}  {k[:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
