"""Fair speed protocol for OuterProductMean at large MSA depth (H100 power cap: 700 W; clocks drop 15-18 % after ~50-80 ms of load).

Per (row, L, S) case every backend of this stack is captured once into a CUDA graph (the requirements construction of measure.config,
bf16 autocast "master" dtype, unpadded masks), then for each of --reps repetitions, the backends in a rotated order:
  cool       idle --idle s (GPU asleep), then a burst of K timed calls (K * t <= ~40 ms; 256 MB L2 flush before each, as the measurer)
  sustained  back-to-back replays for --sustain s (the clock settles at the cap), then K2 timed calls in the same stream
The PyTorch backend runs in both stacks as the drift reference: compare backend/PyTorch ratios across the engine and the kit stack.
One JSONL line per (case, backend) with every rep's cool / sustained medians, the SM clock (NVML) and power after the sustained phase.
Resumable: finished (row, L, S, backend) are skipped.

  python fair.py --rows 55 57 --lengths 768 --depths 8192 16384 --backends pytorch cuda triton --out OUT.jsonl
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import statistics
import sys
import time

import torch

from miniworld_engine.measure import config, speed
from miniworld_engine.modules.primitives import BackendUnsupported, unpadded_masks


def _nvml():
    try:
        return torch.cuda.clock_rate() / 1000.0, torch.cuda.power_draw() / 1000.0
    except Exception:
        return None, None


def _done(path):
    keys = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            keys.add((r["row"], r["length"], r["depth"], r["backend"]))
    return keys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, nargs="+", required=True)
    p.add_argument("--lengths", type=int, nargs="+", required=True)
    p.add_argument("--depths", type=int, nargs="+", required=True)
    p.add_argument("--backends", nargs="+", required=True)
    p.add_argument("--reps", type=int, default=4)
    p.add_argument("--idle", type=float, default=2.0)
    p.add_argument("--sustain", type=float, default=0.6)
    p.add_argument("--deadline", type=float, default=500.0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    t0 = time.time()
    dev = torch.device("cuda")
    timer = speed._Timer(dev, True)
    done = _done(a.out)
    cfg = config.Config(modules=("OuterProductMean",), rows=tuple(a.rows), lengths=tuple(a.lengths), depths="min", modes=("eval",), samples="min")
    print(f"[fair] torch {torch.__version__} {torch.cuda.get_device_name()}", flush=True)
    with open(a.out, "a") as sink:
        for case0 in config.cases(cfg):
            for depth in a.depths:
                case = dataclasses.replace(case0, depth=depth)
                todo = [b for b in a.backends if b == "pytorch" or b in case.cls.backends]
                if all((case.row, case.length, depth, b) in done for b in todo):
                    continue
                if time.time() - t0 > a.deadline:
                    print("[fair] DEADLINE (resume to continue)", flush=True)
                    return 0
                tc = time.time()
                base = config.build(case, device=dev)
                graphs, recs = {}, {}
                for b in todo:
                    rec = {"row": case.row, "length": case.length, "depth": depth, "backend": b, "kwargs": case.kwargs, "torch": torch.__version__}
                    recs[b] = rec
                    try:
                        if b not in type(base).backends:
                            raise BackendUnsupported(f"no backend {b}")
                        module = copy.deepcopy(base)
                        with unpadded_masks():
                            inputs = config.inputs(case, module, device=dev)
                        step = speed.make_step(module, inputs, backend=b, train=False, autocast=config.autocast(case),
                                               seed=config.seed(case), compile_mode=None)
                        side = torch.cuda.Stream(dev)
                        side.wait_stream(torch.cuda.current_stream(dev))
                        with torch.cuda.stream(side):
                            for _ in range(3):
                                step()
                        torch.cuda.current_stream(dev).wait_stream(side)
                        torch.cuda.synchronize(dev)
                        g = speed._capture(step, side)
                        torch.cuda.synchronize(dev)
                        g.replay()
                        torch.cuda.synchronize(dev)
                        graphs[b] = (g, module, inputs, step)
                        rec.update(status="ok", cool=[], sus=[], sus_ghz=[], sus_w=[])
                    except BackendUnsupported as exc:
                        rec.update(status="unsupported", reason=str(exc)[:300])
                    except Exception as exc:
                        rec.update(status="failed", reason=speed._reason(exc)[:600])
                        torch.cuda.synchronize(dev)
                order = list(graphs)
                if order:
                    # call time estimate -> burst sizes
                    est = {}
                    for b in order:
                        ms = timer.run(graphs[b][0].replay, 2)
                        est[b] = max(min(ms), 0.05)
                    for rep in range(a.reps):
                        rot = order[rep % len(order):] + order[: rep % len(order)]
                        for b in rot:
                            g = graphs[b][0]
                            k = max(1, min(5, int(40.0 / est[b])))
                            torch.cuda.synchronize(dev)
                            time.sleep(a.idle)
                            recs[b]["cool"].append(statistics.median(timer.run(g.replay, k)))
                        for b in rot:
                            g = graphs[b][0]
                            n_warm = max(2, int(a.sustain * 1000.0 / est[b]))
                            k2 = max(3, min(10, int(100.0 / est[b])))
                            for _ in range(n_warm):  # back-to-back: the CPU runs ahead of the GPU
                                timer.flush.zero_()
                                g.replay()
                            starts = [torch.cuda.Event(enable_timing=True) for _ in range(k2)]
                            ends = [torch.cuda.Event(enable_timing=True) for _ in range(k2)]
                            for s_, e_ in zip(starts, ends):
                                timer.flush.zero_()
                                s_.record()
                                g.replay()
                                e_.record()
                            # NVML while the last calls run
                            ends[-2].synchronize()
                            ghz, w = _nvml()
                            torch.cuda.synchronize(dev)
                            recs[b]["sus"].append(statistics.median(s_.elapsed_time(e_) for s_, e_ in zip(starts, ends)))
                            recs[b]["sus_ghz"].append(ghz)
                            recs[b]["sus_w"].append(w)
                for b in todo:
                    rec = recs[b]
                    if rec["status"] == "ok":
                        rec["cool_ms"] = statistics.median(rec["cool"])
                        rec["sus_ms"] = statistics.median(rec["sus"])
                    if (case.row, case.length, depth, b) not in done:
                        sink.write(json.dumps(rec, default=str) + "\n")
                        sink.flush()
                    msg = (f"cool {rec['cool_ms']:.3f} sus {rec['sus_ms']:.3f} ms | cool {['%.2f' % x for x in rec['cool']]} sus "
                           f"{['%.2f' % x for x in rec['sus']]} GHz {rec['sus_ghz']}" if rec["status"] == "ok" else f"{rec['status']}: {rec.get('reason', '')[:160]}")
                    print(f"row{case.row} L{case.length} S{depth} {b:<9} {msg}", flush=True)
                for v in graphs.values():
                    del v
                graphs = None
                del base
                torch.cuda.synchronize(dev)
                torch.cuda.empty_cache()
                print(f"[fair] case {time.time() - tc:.1f} s (total {time.time() - t0:.0f} s)", flush=True)
    print("[fair] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
