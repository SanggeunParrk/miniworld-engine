"""Run one of the measurers (speed, accuracy, memory) on a config of the harness; raw records plus a provenance sidecar.

    python paper_common/run_measure.py CONFIG.yaml --kind speed --out records/<card>/<stage>.r<k>.jsonl [--ext trifast ds4sci] [selection overrides]

``--kind speed``: miniworld_engine.measure.speed.run (timing graph, L2 flush, warmup / iterations from the config), unchanged;
``--kind accuracy``: miniworld_engine.measure.accuracy.run (float64 gold, judge light), unchanged; ``--kind memory``: paper_common/memory.py (E6).

Resumable: cases whose records (every backend) are already in ``--out`` are skipped, ``--budget-seconds N`` stops starting new cases after N
seconds (exit 0, no ``.done`` marker), and ``<out>.done`` marks a finished file (a later call exits at once): chain 10-minute jobs
(paper_common/chain.sh) until the marker exists. ``--max-cases N`` caps the number of cases (smoke runs).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmarks/paper: paper_common importable from any cwd / kit-stack wrapper

import torch

from paper_common import cfg as cfgmod
from paper_common import csvio, provenance


def case_key(case_or_rec) -> tuple:
    g = (lambda k: case_or_rec[k]) if isinstance(case_or_rec, dict) else (lambda k: getattr(case_or_rec, k))
    kw = g("kwargs")
    return (g("row"), json.dumps(kw, sort_keys=True), g("length"), g("depth"), g("samples"), g("dtype"), g("mode"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--kind", choices=["speed", "accuracy", "memory"], default="speed")
    p.add_argument("--out", required=True, help="raw JSONL records (appended); <out>.meta.json gets the provenance")
    p.add_argument("--ext", nargs="*", default=[], help="E5 external backends to install first (names in E5_other_kernels/externals.py)")
    p.add_argument("--budget-seconds", type=float, default=0.0, help="stop starting new cases after this many seconds (0 = no limit)")
    cfgmod.add_selection_args(p)
    args = p.parse_args(argv)
    out = Path(args.out)
    done_marker = Path(f"{out}.done")
    if done_marker.exists():
        print(f"[paper] {done_marker} exists: nothing to do")
        return 0
    if not torch.cuda.is_available():
        print("[paper] no CUDA device", file=sys.stderr)
        return 2
    from miniworld_engine.measure import accuracy, config, speed

    t0 = time.time()
    cfg = cfgmod.load(args.config, args)
    if args.ext:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "E5_other_kernels"))
        import externals

        externals.install(args.ext)
    have: dict[tuple, set] = {}  # case_key -> backends already recorded (each counts once, however many duplicate records)
    if out.exists():
        for rec in csvio.read_jsonl(str(out)):
            if rec.get("timing", "graph") == "graph" and rec.get("status") != "error":
                have.setdefault(case_key(rec), set()).add(rec["backend"])
    todo, budget_hit = [], False
    missing: dict[tuple, list[str]] = {}
    for case in cfgmod.limit(cfg, args.max_cases):
        miss = [b for b in config.backends_of(cfg, case) if b not in have.get(case_key(case), ())]
        if miss:
            missing[case_key(case)] = miss
            todo.append(case)
    total = len(todo)
    print(f"[paper] {total} case(s) to measure ({sum(map(len, have.values()))} records already in {out.name})", flush=True)
    # speed.run measures config.backends_of(cfg, case): only the backends a case still lacks (resume per backend, not per case)
    _all_backends = config.backends_of
    config.backends_of = lambda c, case: missing.get(case_key(case), _all_backends(c, case))

    def gen(_cfg):
        nonlocal budget_hit
        for case in todo:
            if args.budget_seconds and time.time() - t0 > args.budget_seconds:
                budget_hit = True
                print("[paper] budget used up: stopping before the next case", flush=True)
                return
            yield case

    config.cases = gen  # speed.run reads config.cases(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = provenance.describe() | {"config": str(Path(args.config).resolve()), "settings": dataclasses.asdict(cfg).get("speed", {})}
    Path(f"{out}.meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"[paper] {json.dumps(meta, default=str)}", flush=True)
    if total:
        if args.kind == "speed":
            speed.run(cfg, str(out))
        elif args.kind == "accuracy":
            accuracy.run(cfg, str(out))
        else:
            from paper_common import memory

            memory.run(cfg, str(out))
    if not budget_hit:
        done_marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
