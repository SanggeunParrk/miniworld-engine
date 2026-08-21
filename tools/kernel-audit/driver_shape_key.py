"""Does each driver's shape actually reach the kernel's shape_key?

The per-op sweep tunes one (op, shape bucket) per work item by driving the kernel at a chosen L.
That only works if the bucket the kernel RECORDS moves when L moves. It does not always: a driver
that hands the kernel an already-flattened (M, N) activation makes `length_of` return M rather
than L, so for a pair kernel M = L*L lands in the clamped top bucket at any L >= 91 -- every shape
collapses onto one entry and the sweep tunes the same bucket over and over.

Measured that way on gated_projection_gate_triton: eight units at L=128..8192 all recorded
shape_key=8192.

Run each driver at two lengths with a one-config grid and compare the buckets it records.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHILD = r'''
import json, sys
sys.path.insert(0, "src")
import torch
from miniworld_engine import settings
from miniworld_engine.autotune import capture
from miniworld_engine.autotune.builder import _run_one_driver

op, out = sys.argv[1], sys.argv[2]
settings.configure(run_autotune=True, capture=True)
capture.install()
ran = _run_one_driver(op)
buckets = sorted({b for slot in capture._CAPTURE.values() for _d, b in slot["entries"]})
json.dump({"ran": ran, "buckets": buckets}, open(out, "w"))
'''


def probe(op: str, length: int, tmp: Path) -> dict:
    out = tmp / f"{op}-{length}.json"
    env = {**os.environ, "PYTHONPATH": "src",
           "MINIWORLD_CONFIG_DIR": str(REPO / ".bench/onecfg"),
           "MINIWORLD_DRIVER_LENGTH": str(length)}
    r = subprocess.run([".pixi/envs/default/bin/python", "-c", CHILD, op, str(out)],  # noqa: S603
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    if r.returncode != 0 or not out.is_file():
        return {"ran": 0, "buckets": [], "err": (r.stderr or r.stdout)[-160:]}
    return json.loads(out.read_text())


def main() -> int:
    lo, hi = (int(a) for a in (sys.argv[1:3] or ("256", "512")))
    tmp = REPO / ".bench/_shapekey"
    tmp.mkdir(parents=True, exist_ok=True)
    reg = [r for r in csv.DictReader((REPO / "src/miniworld_engine/kernels/registry.csv").open())
           if r["backend"] == "triton" and (r["driver"] or "").strip()]
    moved, stuck, dead = [], [], []
    for i, r in enumerate(reg, 1):
        op = r["kernel"]
        a, b = probe(op, lo, tmp), probe(op, hi, tmp)
        if not a["ran"] or not b["ran"]:
            dead.append((op, a.get("err", "")[:80]))
        elif a["buckets"] == b["buckets"]:
            stuck.append((op, r["level"], a["buckets"][:1]))
        else:
            moved.append((op, a["buckets"][:1], b["buckets"][:1]))
        if i % 10 == 0:
            print(f"  {i}/{len(reg)}  moved={len(moved)} stuck={len(stuck)} dead={len(dead)}",
                  flush=True)

    print(f"\nL={lo} vs L={hi} over {len(reg)} triton drivers")
    print(f"  bucket MOVED with the driver length : {len(moved)}")
    print(f"  bucket STUCK (sweep would tune one bucket N times) : {len(stuck)}")
    print(f"  driver did not run at one of them : {len(dead)}")
    if stuck:
        print("\nSTUCK:")
        for op, lvl, bk in stuck:
            print(f"  {op:48s} level={lvl:5s} {bk}")
    if dead:
        print("\nDID NOT RUN:")
        for op, err in dead[:15]:
            print(f"  {op:48s} {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
