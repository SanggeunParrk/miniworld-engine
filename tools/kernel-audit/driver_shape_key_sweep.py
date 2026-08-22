"""Does each driver reach EVERY shape bucket of its level -- not just move between two?

``driver_shape_key.py`` runs one PAIR of lengths (256 vs 512) and asks whether the recorded bucket
moved. That is weak evidence. A driver can move between two lengths and still be broken elsewhere:
clamped at the top of its level's bucket set, collapsing two adjacent buckets onto one, or skipping
a bucket entirely. The per-op sweep creates one work item per (kernel, bucket), so any bucket that
cannot be REACHED is a work item that tunes the wrong entry.

This drives every (kernel, bucket) that ``builder.op_units()`` enumerates -- the exact work list
the sweep will use -- and records the ``shape_key=`` the launch actually put in its cache bucket.

  IDENTITY  every driven L recorded shape_key == L                          (expected)
  COLLAPSE  two or more driven lengths recorded the SAME shape_key
  WRONG     some driven L recorded a shape_key that is not L
  NO_KEY    the kernel's ``key=[...]`` has no ``shape_key`` (op_units already gives it one item)
  SKIPPED   the driver could not run at that L -- OOM at the big atom shapes is expected

One child process per (kernel, length): the drivers read ``MINIWORLD_DRIVER_LENGTH`` at IMPORT
time, so the length cannot be changed inside a live process.

    # one shard of a job array
    python tools/kernel-audit/driver_shape_key_sweep.py --shard 3 --nshards 24
    # collate
    python tools/kernel-audit/driver_shape_key_sweep.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / ".bench/_shapekey_sweep"
PY = str(REPO / ".pixi/envs/default/bin/python")
CONFIG_DIR = str(REPO / ".bench/onecfg")

# The child runs ONE driver at ONE length and dumps the cache buckets every autotuner recorded.
# MINIWORLD_CONFIG_DIR is already in the environment before this text is executed, which is what
# `autotune.configs` needs -- it reads the directory at import time.
CHILD = r'''
import contextlib, io, json, sys
sys.path.insert(0, "src")
import torch
from miniworld_engine import settings
from miniworld_engine.autotune import capture
from miniworld_engine.autotune.builder import _run_one_driver

op, out = sys.argv[1], sys.argv[2]
settings.configure(run_autotune=True, capture=True)
capture.install()
buf = io.StringIO()
err = ""
try:
    with contextlib.redirect_stdout(buf):
        ran = _run_one_driver(op)
except BaseException as exc:                     # noqa: BLE001 -- a hard failure is data too
    ran, err = 0, f"{type(exc).__name__}: {exc}"
slots = {name: sorted({b for _d, b in slot["entries"]})
         for name, slot in capture._CAPTURE.items()}
json.dump({"ran": ran, "slots": slots, "log": buf.getvalue()[-600:], "err": err[-600:],
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
          open(out, "w"))
'''

SHAPE_KEY = re.compile(r"\bshape_key=(\d+)")


def items() -> list[tuple[str, int]]:
    """The sweep's own work list: one (kernel, bucket) per op_units item. Reused, not rebuilt --
    a probe over a list the sweep does not use proves nothing about the sweep."""
    sys.path.insert(0, str(REPO / "src"))
    os.environ.setdefault("MINIWORLD_CONFIG_DIR", CONFIG_DIR)
    from miniworld_engine.autotune.builder import op_units

    return [(u.op, u.length) for u in op_units(config_dir=Path(CONFIG_DIR))]


def probe(op: str, length: int, tmp: Path) -> dict:
    out = tmp / f"{op}-{length}.json"
    out.unlink(missing_ok=True)
    env = {**os.environ, "PYTHONPATH": "src", "MINIWORLD_CONFIG_DIR": CONFIG_DIR,
           "MINIWORLD_DRIVER_LENGTH": str(length)}
    try:
        r = subprocess.run([PY, "-c", CHILD, op, str(out)],  # noqa: S603
                           cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return {"op": op, "length": length, "ran": 0, "slots": {}, "err": "TIMEOUT 1800s"}
    if not out.is_file():
        return {"op": op, "length": length, "ran": 0, "slots": {},
                "err": f"rc={r.returncode} " + (r.stderr or r.stdout)[-600:]}
    rec = json.loads(out.read_text())
    rec.update(op=op, length=length)
    return rec


def run_shard(shard: int, nshards: int) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"tmp-{shard}"
    tmp.mkdir(exist_ok=True)
    work = [it for i, it in enumerate(items()) if i % nshards == shard]
    path = OUT / f"shard-{shard}.jsonl"
    with path.open("w") as fh:
        for n, (op, length) in enumerate(work, 1):
            rec = probe(op, length, tmp)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            got = SHAPE_KEY.search(" ".join(rec["slots"].get(op, []))) if rec["slots"] else None
            print(f"[{shard}] {n}/{len(work)} {op} L={length} ran={rec['ran']} "
                  f"shape_key={got.group(1) if got else '-'}", flush=True)
    return 0


def _key_of(rec: dict) -> tuple[str | None, str]:
    """(shape_key recorded by the TARGET op, the raw bucket string) for one probe."""
    buckets = rec.get("slots", {}).get(rec["op"], [])
    if not buckets:
        return None, ""
    # One driver, one op, one length -> one bucket. More than one means the driver launched the
    # same kernel at two different shapes in a single call, which is itself worth naming.
    keys = {m.group(1) for b in buckets if (m := SHAPE_KEY.search(b))}
    if not keys:
        return "NO_KEY", ";".join(buckets)
    return ("|".join(sorted(keys, key=int)) if len(keys) > 1 else keys.pop()), ";".join(buckets)


def report() -> int:
    recs = [json.loads(ln) for f in sorted(OUT.glob("shard-*.jsonl"))
            for ln in f.read_text().splitlines() if ln.strip()]
    by_op: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_op[r["op"]].append(r)

    classes: dict[str, list[str]] = defaultdict(list)
    detail: list[str] = []
    ooms: list[str] = []
    fails: list[str] = []
    for op in sorted(by_op):
        got: dict[int, str] = {}
        for r in sorted(by_op[op], key=lambda r: r["length"]):
            L = r["length"]
            k, _raw = _key_of(r)
            if not r["ran"] or k is None:
                why = (r.get("err") or r.get("log") or "").strip().replace("\n", " ")
                (ooms if _is_oom(why) else fails).append(f"{op:46s} L={L:<5d} {why[-150:]}")
                continue
            got[L] = k
        if not got:
            classes["ALL_SKIPPED"].append(op)
            continue
        if all(v == "NO_KEY" for v in got.values()):
            classes["NO_KEY"].append(op)
            continue
        wrong = {L: v for L, v in got.items() if v != str(L)}
        dupes = {v: [L for L in got if got[L] == v] for v in set(got.values())}
        dupes = {v: Ls for v, Ls in dupes.items() if len(Ls) > 1}
        if wrong:
            classes["WRONG"].append(op)
            detail.append(f"WRONG    {op}: " + ", ".join(
                f"L={L} -> shape_key={v}" for L, v in sorted(wrong.items())))
        if dupes:
            classes["COLLAPSE"].append(op)
            detail.append(f"COLLAPSE {op}: " + ", ".join(
                f"L={sorted(Ls)} all -> shape_key={v}" for v, Ls in sorted(dupes.items())))
        if not wrong and not dupes:
            classes["IDENTITY"].append(op)

    print(f"{len(recs)} probes over {len(by_op)} kernels\n")
    for name in ("IDENTITY", "NO_KEY", "COLLAPSE", "WRONG", "ALL_SKIPPED"):
        ops = sorted(set(classes.get(name, [])))
        print(f"  {name:12s} {len(ops):3d}   {' '.join(ops) if name != 'IDENTITY' else ''}")
    if detail:
        print("\nFINDINGS")
        for line in sorted(detail):
            print("  " + line)
    print(f"\nOOM / out-of-resources at a shape ({len(ooms)}) -- expected at the big atom buckets")
    for line in ooms:
        print("  " + line)
    print(f"\nREAL FAILURES ({len(fails)})")
    for line in fails:
        print("  " + line)
    return 0


def _is_oom(msg: str) -> bool:
    m = msg.lower()
    return any(s in m for s in ("out of memory", "outofmemory", "outofresources",
                                "out of resource", "cuda_error_out_of_memory"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--count", action="store_true")
    a = ap.parse_args()
    if a.count:
        print(len(items()))
        return 0
    if a.report:
        return report()
    return run_shard(a.shard, a.nshards)


if __name__ == "__main__":
    raise SystemExit(main())
