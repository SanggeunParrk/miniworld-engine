"""Drive every kernel ``registry.csv`` declares ``bf16|fp32`` in fp32, and say what happened.

The ``dtypes`` column is a DECLARATION. Every driver, checker and measurement in this repo has
run bf16 (``drivers.BF16``), so nothing has ever executed the fp32 half of it. This runs the
registry's own drivers and checkers with ``MINIWORLD_DRIVER_DTYPE=fp32`` and records, per kernel:

    ran        the driver returned and the checker matched the torch reference
    WRONG      the driver ran but the numbers disagree with the reference
    raised     the driver or the checker raised -- the exception text is the reason

It also records the checker's worst relative error in BOTH dtypes. That is the part a pass/fail
cannot say: a kernel that accepts fp32 tensors but casts them to bf16 inside still "runs" and
still passes a 5e-2 band, and the only visible trace is an fp32 error the size of the bf16 one.

One child process per family, so a kernel that corrupts the CUDA context (an OOB store, a failed
CUDA graph) cannot take the other families' verdicts with it; a family whose child dies is
re-driven one kernel per process to pin the blame.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / ".pixi/envs/default/bin/python")
REGISTRY = REPO / "src/miniworld_engine/kernels/registry.csv"

CHILD = r'''
import json, sys, traceback
sys.path.insert(0, "src")
import torch
from miniworld_engine.autotune.run_all import _resolve

def rel(check):
    got = _resolve(check)()
    pairs = got if isinstance(got, dict) else {"out": got}
    worst, detail = 0.0, []
    for name, (actual, expected) in pairs.items():
        a, e = actual.float(), expected.float()
        if a.shape != e.shape:
            raise AssertionError(f"{name}: shape {tuple(a.shape)} vs reference {tuple(e.shape)}")
        r = (a - e).abs().max().item() / (e.abs().max().item() or 1.0)
        worst = max(worst, r); detail.append(f"{name}={r:.2e}")
    return worst, " ".join(detail)

for line in json.loads(sys.argv[1]):
    kern, drv, chk = line
    rec = {"kernel": kern}
    try:
        _resolve(drv)()
        torch.cuda.synchronize()
        rec["drove"] = True
    except Exception as exc:
        rec["drove"] = False
        rec["error"] = f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:400] if str(exc).strip() else ''}"
        print("RESULT " + json.dumps(rec), flush=True)
        continue
    try:
        worst, detail = rel(chk)
        rec["rel"] = worst
        rec["detail"] = detail
        rec["ok"] = worst < 5e-2 and worst == worst
    except Exception as exc:
        rec["ok"] = False
        rec["error"] = f"checker {type(exc).__name__}: {str(exc).strip().splitlines()[0][:400] if str(exc).strip() else ''}"
    print("RESULT " + json.dumps(rec), flush=True)
'''


#: ``--observed``: drive with the capture hook installed and report the dtype the AUTOTUNER saw.
#: The pass/fail above cannot distinguish "the kernel ran in fp32" from "the driver hardcodes
#: ``torch.bfloat16`` so the env override never reached it" -- both look like a clean pass.
#: ``cache.dtype_of_args`` reads the dtype off the kernel's own first tensor operand and is what
#: keys the cache entry, so the captured bucket IS the answer, for every triton op at once.
OBSERVE = r'''
import json, sys
sys.path.insert(0, "src")
import torch
from miniworld_engine import settings
from miniworld_engine.autotune import capture
from miniworld_engine.autotune.run_all import _resolve
settings.configure(run_autotune=True, capture=True)
capture.install()
for kern, drv, chk in json.loads(sys.argv[1]):
    try:
        _resolve(drv)()
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"SKIP {kern} {type(exc).__name__}", flush=True)
capture.dump_shard(sys.argv[2])
'''


def rows() -> list[dict]:
    with REGISTRY.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _child(work: list[tuple[str, str, str]], dtype: str, cfg: str,
           code: str = CHILD, extra: list[str] = ()) -> tuple[dict, int, str]:
    env = {**os.environ, "PYTHONPATH": "src", "MINIWORLD_CONFIG_DIR": cfg,
           "MINIWORLD_DRIVER_DTYPE": dtype}
    proc = subprocess.run([PY, "-c", code, json.dumps(work), *extra],  # noqa: S603
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=5400)
    out = {}
    for ln in proc.stdout.splitlines():
        if ln.startswith("RESULT "):
            r = json.loads(ln[len("RESULT "):])
            out[r["kernel"]] = r
    return out, proc.returncode, (proc.stderr or "")[-2000:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dtype", default="fp32", choices=("fp32", "bf16"))
    ap.add_argument("--out", default="")
    ap.add_argument("--families", default="")
    ap.add_argument("--observed", default="", help="capture-hook mode: write op -> dtype seen here")
    args = ap.parse_args()
    cfg = os.environ.get("MINIWORLD_CONFIG_DIR") or str(REPO / ".bench/onecfg")

    want = {f for f in args.families.split(",") if f}
    todo = [r for r in rows() if "fp32" in r["dtypes"] and r["driver"].strip()
            and (not want or r["family"] in want)]
    by_family: dict[str, list[dict]] = collections.OrderedDict()
    for r in todo:
        by_family.setdefault(r["family"], []).append(r)

    if args.observed:
        shards = REPO / ".bench/_fp32observe" / args.dtype
        shards.mkdir(parents=True, exist_ok=True)
        seen: dict[str, list[str]] = {}
        for fam, rs in by_family.items():
            work = [(r["kernel"], r["driver"], r["check"]) for r in rs]
            sh = shards / f"{fam}.json"
            _, rc, err = _child(work, args.dtype, cfg, code=OBSERVE, extra=[str(sh)])
            if not sh.is_file():
                print(f"  [{fam}] no shard (rc={rc}) {err.strip().splitlines()[-1][:160] if err.strip() else ''}",
                      flush=True)
                continue
            for op, slot in json.loads(sh.read_text()).items():
                dts = sorted({b.split("|", 1)[0] for b in slot.get("entries", {})})
                seen.setdefault(op, []).extend(dts)
        for op in sorted(seen):
            print(f"  {op:46s} autotuner saw {sorted(set(seen[op]))}", flush=True)
        Path(args.observed).write_text(json.dumps({k: sorted(set(v)) for k, v in seen.items()},
                                                  indent=1, sort_keys=True))
        print(f"  -> {args.observed}   ({len(seen)} ops observed)")
        return 0

    results: dict[str, dict] = {}
    for fam, rs in by_family.items():
        work = [(r["kernel"], r["driver"], r["check"]) for r in rs]
        got, rc, err = _child(work, args.dtype, cfg)
        missing = [w for w in work if w[0] not in got]
        if missing:
            print(f"  [{fam}] child rc={rc}, {len(missing)} kernel(s) unaccounted -> re-driving alone",
                  flush=True)
            if err.strip():
                print("    child stderr tail: " + err.strip().splitlines()[-1][:200], flush=True)
            for w in missing:
                one, rc1, err1 = _child([w], args.dtype, cfg)
                if w[0] in one:
                    got[w[0]] = one[w[0]]
                else:
                    tail = [ln for ln in err1.splitlines() if ln.strip()]
                    got[w[0]] = {"kernel": w[0], "drove": False, "ok": False,
                                 "error": f"process died rc={rc1}: {tail[-1][:300] if tail else ''}"}
        for r in rs:
            rec = got[r["kernel"]]
            rec["family"] = fam
            results[r["kernel"]] = rec
            verdict = ("ran " if rec.get("ok") else ("WRONG" if rec.get("drove") else "raise"))
            note = rec.get("error") or f"rel {rec.get('detail','')}"
            print(f"  [{verdict}] {r['kernel']:46s} {note[:100]}", flush=True)

    n_ok = sum(1 for v in results.values() if v.get("ok"))
    n_wrong = sum(1 for v in results.values() if v.get("drove") and not v.get("ok"))
    n_raise = sum(1 for v in results.values() if not v.get("drove"))
    print(f"\n  {args.dtype}: {len(results)} declared-fp32 kernels driven   "
          f"ran+correct {n_ok}   wrong-numbers {n_wrong}   raised {n_raise}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1, sort_keys=True))
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
