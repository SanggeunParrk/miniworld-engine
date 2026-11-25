"""Same-process A-B-A-B timing of variants of one cell (CUDA graph, L2 flush), for thin-margin work.

    python tools/bench/ab_cells.py CONFIG.yaml --module M --rows 158 --lengths 128,256 --modes eval \
        --variant cuda --variant "cuda|import x; x.FLAG = 1" --variant triton [--rounds 8] [--reps 20]

A variant is ``backend`` or ``backend|statement``: the statement runs (exec) just before the graph of that variant is captured and is undone
by the next ``|reset`` statement of the variant list, so it must set module-level switches that the forward reads at capture time. The
graphs of all variants are replayed interleaved (A B C A B C ...) and the median per variant over rounds is printed.
"""

from __future__ import annotations

import argparse
import copy
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "paper"))
from paper_common import cfg as cfgmod


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    cfgmod.add_selection_args(p)
    p.add_argument("--variant", action="append", required=True)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--reset", default="", help="statement run after each capture (undoes the variants' switches)")
    args = p.parse_args()
    from foldjet_engine.measure import config
    from foldjet_engine.measure.speed import _capture, _Timer, make_step, unpadded_masks

    cfg = cfgmod.load(args.config, args)
    dev = torch.device("cuda")
    timer = _Timer(dev, True)
    side = torch.cuda.Stream(dev)
    for case in cfgmod.limit(cfg, args.max_cases):
        base = config.build(case, device=dev)
        graphs, names = [], []
        for v in args.variant:
            backend, _, stmt = v.partition("|")
            try:
                if stmt:
                    exec(stmt, {})
                module = copy.deepcopy(base)
                with unpadded_masks():
                    inputs = config.inputs(case, module, device=dev)
                step = make_step(
                    module,
                    inputs,
                    backend=backend,
                    train=case.mode == "train",
                    autocast=config.autocast(case),
                    seed=config.seed(case),
                    compile_mode=None,
                    keep_expanded=config.keeps_expanded(case),
                )
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(5):
                        step()
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                g = _capture(step, side)
                torch.cuda.synchronize()
                for _ in range(5):
                    g.replay()
                torch.cuda.synchronize()
                graphs.append((g, module, inputs, step))
                names.append(v)
            except Exception as exc:
                print(f"  variant {v!r} failed: {type(exc).__name__}: {str(exc)[:200]}")
            finally:
                if args.reset:
                    exec(args.reset, {})
        times = [[] for _ in graphs]
        for _ in range(args.rounds):
            for i, (g, *_r) in enumerate(graphs):
                times[i].append(statistics.median(timer.run(g.replay, args.reps, head_start_ms=2.0)))
        label = f"{case.module} r{case.row} {case.mode} {case.dtype} L{case.length} d{case.depth} s{case.samples}"
        base_t = statistics.median(times[0]) if times else 0
        print(f"{label}")
        for n, t in zip(names, times):
            m = statistics.median(t)
            print(f"   {n:60s} {m * 1e3:9.2f} us  min {min(t) * 1e3:9.2f}  ratio_vs_first {m / base_t:.4f}")
        del graphs, base
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
