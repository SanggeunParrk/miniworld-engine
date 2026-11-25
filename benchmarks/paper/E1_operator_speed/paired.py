"""E1, paired: the engine and the external backends of a case timed in ONE process, in interleaved rounds (re-measure of single cells).

    python E1_operator_speed/paired.py CONFIG.yaml --out records_paired/<card>/paired.r1.jsonl --rounds 6 [--module M ...] [selection overrides]

Why: the E1 campaign (``rtx5090.sh`` etc.) times the engine stack (pytorch / cuda / triton / cueq) and the Anthropic kit (pytorch /
anthropic) in two separate processes, one backend after the other, one timing run each. The engine-vs-kit ratio is then a cross-process
ratio, and on a card without locked clocks (RTX 5090) the clock state of the two processes differs by several percent (the kit process's own
pytorch reference ran 4-9 % faster than the engine process's on the same cases). Here every backend of a case is built once (same seeded
weights / inputs as ``foldjet_engine.measure.speed.run``) and timed ``--rounds`` times, the backend order rotated every round, so all
backends of a case share the clock state; the per-backend number is the median of the per-round medians, and a ratio is taken per round
(``baseline_round / cuda_round``) before its median (``paired_summary.py``).

Timing of one round = ``speed.measure`` unchanged (CUDA graph, warmup / iterations from the config, L2 flush): the step is captured afresh
every round. Statuses as in ``speed.run``: ``unsupported`` (``_supports_<name>`` refuses), ``failed``, ``skipped`` (out of memory); a step
that cannot be captured is timed eagerly (``timed_as: eager``, an upper bound). All steps of a case are held at once; when that does not fit
in memory the case is redone building each backend's step per round (``held: false``), where an out-of-memory backend is ``skipped``.
Resumable: cases already in ``--out`` are skipped; ``--budget-seconds`` stops before the next case; ``<out>.done`` marks a finished file.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmarks/paper

import torch

from paper_common import cfg as cfgmod
from paper_common import provenance
from paper_common.run_measure import case_key


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--out", required=True)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--budget-seconds", type=float, default=0.0)
    cfgmod.add_selection_args(p)
    args = p.parse_args(argv)
    out = Path(args.out)
    if Path(f"{out}.done").exists():
        print(f"[paired] {out}.done exists: nothing to do")
        return 0
    from foldjet_engine.measure import config, speed
    from foldjet_engine.modules.primitives import BackendUnsupported, unpadded_masks

    t0 = time.time()
    cfg = cfgmod.load(args.config, args)
    st = config.speed_settings(cfg)
    warmup, iterations = int(st.get("warmup", 5)), int(st.get("iterations", 100))
    flush_l2 = bool(st.get("flush_l2", True))
    dev = torch.device("cuda")
    timer = speed._Timer(dev, flush_l2)
    env = {**speed._environment(dev), "flush_l2": flush_l2, "warmup": warmup, "iterations": iterations, "rounds": args.rounds}
    have = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                have.add(case_key(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
    cases = [c for c in cfgmod.limit(cfg, args.max_cases) if case_key(c) not in have]
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_run = config.run_meta(cfg, "speed", speed.SPEED_VERSION)
    Path(f"{out}.meta.json").write_text(json.dumps(provenance.describe() | {"config": str(Path(args.config).resolve()), "paired": env}, indent=1, default=str))
    print(f"[paired] {len(cases)} case(s) to measure, {len(have)} already in {out.name}; {env}", flush=True)

    def build(base, case, backend):
        module = copy.deepcopy(base)
        with unpadded_masks():
            inputs = config.inputs(case, module, device=dev)
        step = speed.make_step(module, inputs, backend=backend, train=case.mode == "train", autocast=config.autocast(case),
                               seed=config.seed(case), compile_mode=None, keep_expanded=config.keeps_expanded(case))  # fmt: skip
        return module, inputs, step

    def time_one(step, rec):
        """One round of one backend: graph (or eager once the capture has failed); returns the median ms."""
        if rec.get("timed_as") != "eager":
            try:
                return speed.measure(step, timer, timing="graph", warmup=warmup, iterations=iterations)["median_ms"]
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as exc:  # cannot be captured: eager from now on (speed.run's rule)
                rec.update(timed_as="eager", capture_error=speed._reason(exc))
                torch.cuda.synchronize(dev)
        return speed.measure(step, timer, timing="eager", warmup=warmup, iterations=iterations)["median_ms"]

    def cleanup():
        torch._dynamo.reset()
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()

    budget_hit = False
    with open(out, "a") as sink:
        for case in cases:
            if args.budget_seconds and time.time() - t0 > args.budget_seconds:
                budget_hit = True
                print("[paired] budget used up: stopping before the next case", flush=True)
                break
            base = config.build(case, device=dev)
            backends = config.backends_of(cfg, case)
            recs = {b: {**config.describe(case, cfg), "meta": meta_run, "backend": b, "timing": "graph", "paired": True, "env": env} for b in backends}
            rounds = {b: [] for b in backends}
            active = []
            for held in (True, False):
                steps, active, oom = {}, [], False
                for b in backends:
                    for key in ("status", "reason", "timed_as", "capture_error"):
                        recs[b].pop(key, None)
                    rounds[b] = []
                    if b not in type(base).backends:
                        recs[b].update(status="unsupported", reason=f"{type(base).__name__} has no backend {b!r}")
                        continue
                    try:
                        if held:
                            steps[b] = build(base, case, b)
                        else:  # check it builds at all (unsupported / failed are decided here), then free it
                            build(base, case, b)
                            cleanup()
                        active.append(b)
                    except BackendUnsupported as exc:
                        recs[b].update(status="unsupported", reason=str(exc))
                    except torch.cuda.OutOfMemoryError as exc:
                        if held:
                            oom = True
                            break
                        recs[b].update(status="skipped", reason=speed._reason(exc))
                    except Exception as exc:
                        recs[b].update(status="failed", reason=speed._reason(exc))
                    cleanup()
                if held and not oom:
                    gone = set()
                    try:
                        for r in range(args.rounds):
                            k = r % max(len(active), 1)
                            for b in active[k:] + active[:k]:
                                if b in gone:
                                    continue
                                try:
                                    rounds[b].append(time_one(steps[b][2], recs[b]))
                                except torch.cuda.OutOfMemoryError:
                                    raise
                                except Exception as exc:  # not even eager runs
                                    recs[b].update(status="failed", reason=speed._reason(exc))
                                    gone.add(b)
                                    cleanup()
                    except torch.cuda.OutOfMemoryError:
                        oom = True
                    active = [b for b in active if b not in gone]
                    steps = None
                    cleanup()
                    if not oom:
                        break
                    print(f"[paired] {case.label} L={case.length}: does not fit with every step held, per-round build instead", flush=True)
                    continue
                if held:  # oom while building the held steps
                    steps = None
                    cleanup()
                    print(f"[paired] {case.label} L={case.length}: does not fit with every step held, per-round build instead", flush=True)
                    continue
                gone = set()
                for r in range(args.rounds):
                    k = r % max(len(active), 1)
                    for b in active[k:] + active[:k]:
                        if b in gone:
                            continue
                        try:
                            step = build(base, case, b)[2]
                            rounds[b].append(time_one(step, recs[b]))
                        except torch.cuda.OutOfMemoryError as exc:
                            recs[b].update(status="skipped", reason=speed._reason(exc))
                            gone.add(b)
                        except Exception as exc:
                            recs[b].update(status="failed", reason=speed._reason(exc))
                            gone.add(b)
                        step = None
                        cleanup()
                active = [b for b in active if b not in gone]
                break
            for b in backends:
                rec = recs[b]
                if b in active and len(rounds[b]) == args.rounds:
                    rec.update(status="ok", median_ms=statistics.median(rounds[b]), rounds_ms=rounds[b], held=bool(held))
                    rec.setdefault("timed_as", "graph")
                    rec["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
                ref = recs["pytorch"].get("median_ms") if recs["pytorch"].get("status") == "ok" else None
                if rec.get("status") == "ok" and ref:
                    rec["speedup_vs_pytorch"] = ref / rec["median_ms"]
                sink.write(json.dumps(rec, default=str) + "\n")
                sink.flush()
                line = f"{case.label} L={case.length} {case.mode} {case.dtype} | {b:<9} | "
                line += f"{rec['median_ms']:9.4f} ms rounds {' '.join(f'{v:.4f}' for v in rec['rounds_ms'])}" if rec.get("status") == "ok" else f"{rec.get('status')}: {str(rec.get('reason', ''))[:120]}"
                print(line + (" | eager" if rec.get("timed_as") == "eager" else ""), flush=True)
            del base
            cleanup()
    if not budget_hit:
        Path(f"{out}.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
