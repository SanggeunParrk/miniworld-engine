"""Speed of OuterProductMean / MSAPairWeightedAveraging at large MSA depth (measurement only).

Reuses the committed measurer (foldjet_engine.measure.speed: graph timing, 256 MB L2 flush before every call, CUDA events, percentiles)
and the requirements cases (measure.config: same module construction, seeded weights, bf16 autocast "master" dtype, inputs without padding),
only overriding the MSA depth, adding peak memory and the error against the PyTorch backend. One JSONL record per
(module, row, L, depth, backend), written as each finishes; records already in --out are skipped (resumable).

  python bench_msa_depth.py --module OuterProductMean --rows 56 58 --lengths 256 384 --depths 1024 8192 16384 \
      --backends pytorch cuda triton anthropic --out OUT.jsonl
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import sys
import time

import torch

from foldjet_engine.measure import config, speed
from foldjet_engine.modules.primitives import BackendUnsupported, unpadded_masks


def _first(x):
    return x[0] if isinstance(x, (tuple, list)) else x


def _done(path: str) -> set:
    keys = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            keys.add((r["row"], r["length"], r["depth"], r["backend"], json.dumps(r["kwargs"], sort_keys=True)))
    return keys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--module", required=True)
    p.add_argument("--rows", type=int, nargs="+", required=True)
    p.add_argument("--lengths", type=int, nargs="+", required=True)
    p.add_argument("--depths", type=int, nargs="+", required=True)
    p.add_argument("--backends", nargs="+", default=["pytorch", "cuda", "triton", "anthropic"])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--deadline", type=float, default=0, help="seconds after start at which to stop starting new cases")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    t0 = time.time()
    dev = torch.device("cuda")
    timer = speed._Timer(dev, True)
    done = _done(a.out)
    cfg = config.Config(modules=(a.module,), rows=tuple(a.rows), lengths=tuple(a.lengths), depths="min", modes=("eval",), samples="min")
    env = {**speed._environment(dev), "warmup": a.warmup, "iterations": a.iterations, "flush_l2": True, "timing": "graph",
           "speed_version": speed.SPEED_VERSION, "config_version": config.CONFIG_VERSION}
    print(f"[bench] {env}", flush=True)
    base_cases = list(config.cases(cfg))
    with open(a.out, "a") as sink:
        for case0 in base_cases:
            for depth in a.depths:
                case = dataclasses.replace(case0, depth=depth)
                if a.deadline and time.time() - t0 > a.deadline:
                    print("[bench] DEADLINE: stopping (resume to continue)", flush=True)
                    return 0
                todo = ["pytorch", *(b for b in a.backends if b != "pytorch" and b in case.cls.backends)]
                keyfor = lambda b: (case.row, case.length, depth, b, json.dumps(case.kwargs, sort_keys=True))  # noqa: E731
                if all(keyfor(b) in done for b in todo):
                    continue
                base = config.build(case, device=dev)
                ref_cpu = None
                # the reference (pytorch) always runs first, also when it is already recorded (the errors need its output)
                for backend in todo:
                    rec = {**config.describe(case, cfg), "backend": backend, "timing": "graph", "env": env}
                    rec["label"] = f"{case.module} row{case.row} L{case.length} S{depth} {case.kwargs}"
                    skip = keyfor(backend) in done
                    if skip and backend != "pytorch":
                        continue
                    module = inputs = step = None
                    try:
                        if backend not in type(base).backends:
                            raise BackendUnsupported(f"{type(base).__name__} has no backend {backend!r}")
                        torch._dynamo.reset()
                        module = copy.deepcopy(base)
                        with unpadded_masks():
                            inputs = config.inputs(case, module, device=dev)
                        torch.cuda.synchronize(dev)
                        before = torch.cuda.memory_allocated(dev)
                        torch.cuda.reset_peak_memory_stats(dev)
                        step = speed.make_step(module, inputs, backend=backend, train=False, autocast=config.autocast(case),
                                               seed=config.seed(case), compile_mode=None)
                        stats = speed.measure(step, timer, timing="graph", warmup=a.warmup, iterations=a.iterations)
                        torch.cuda.synchronize(dev)
                        peak = torch.cuda.max_memory_allocated(dev)
                        rec.update(status="ok", **stats, peak_gb=peak / 2**30, peak_extra_gb=(peak - before) / 2**30, inputs_gb=before / 2**30)
                        # outputs: one more eager call; error vs the pytorch backend (information only)
                        out = _first(step()).float()
                        if backend == "pytorch":
                            ref_cpu = out.cpu()
                        elif ref_cpu is not None:
                            ref = ref_cpu.to(dev)
                            d = (out - ref).abs()
                            rec.update(max_abs_err=float(d.max()), ref_max_abs=float(ref.abs().max()),
                                       max_rel_to_refmax=float(d.max() / ref.abs().max().clamp_min(1e-30)),
                                       mean_abs_err=float(d.mean()), ref_mean_abs=float(ref.abs().mean()))
                            del ref, d
                        del out
                    except BackendUnsupported as exc:
                        rec.update(status="unsupported", reason=str(exc))
                    except Exception as exc:  # recorded, never silent
                        oom = isinstance(exc, torch.cuda.OutOfMemoryError)
                        rec.update(status="skipped" if oom else "failed", reason=speed._reason(exc))
                        if oom:
                            rec["reason"] = "OOM: " + rec["reason"]
                    finally:
                        step = module = inputs = None
                        torch._dynamo.reset()
                        torch.cuda.synchronize(dev)
                        torch.cuda.empty_cache()
                    if skip:  # only the reference output was needed
                        continue
                    sink.write(json.dumps(rec, default=str) + "\n")
                    sink.flush()
                    line = f"{rec['label']} | {backend:<9} | "
                    line += (f"{rec['median_ms']:.4f} ms p10 {rec['p10_ms']:.4f} p90 {rec['p90_ms']:.4f} peak {rec['peak_gb']:.2f} GB "
                             f"err {rec.get('max_abs_err', float('nan')):.3g}" if rec["status"] == "ok" else f"{rec['status']}: {rec.get('reason', '')[:200]}")
                    print(line, flush=True)
                del base
                torch.cuda.empty_cache()
    print("[bench] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
