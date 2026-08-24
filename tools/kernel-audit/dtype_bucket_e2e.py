"""Does the autotune cache actually partition on DTYPE?

``cache.dtype_of_args`` reads the dtype of the first tensor operand and every entry is keyed
``"<dtype>|<bucket>"``, so the partition is claimed by construction. Nothing has tested it,
because nothing has ever driven a kernel in anything but bf16 -- with one dtype in play a cache
that ignores dtype and a cache that honours it are indistinguishable.

Same structure as ``multibucket_e2e.py``, one axis over: drive ONE op at ONE length in bf16 and
in fp32, dump a shard per dtype, merge, and check

  DISTINCT   the merged cache holds one entry per DTYPE, not one entry total
  CORRECT    each entry's dtype component names the dtype it was driven at, and the shape
             component is IDENTICAL across the two -- so the split is dtype and nothing else
  READ BACK  a fresh process at each dtype narrows the grid using THAT dtype's entry

The op's committed cache file is saved before the merge and restored after, so proving the
writer works does not leave a probe's timings in the repo.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OP = "layernorm_fwd_saveact_triton"      # level=both, declared bf16|fp32, cheap to drive
DTYPES = ("bf16", "fp32")
LENGTH = 512

BUILD = r"""
import sys; sys.path.insert(0, "src")
import torch
from miniworld_engine import settings
from miniworld_engine.autotune import capture
from miniworld_engine.autotune.builder import _run_one_driver
settings.configure(run_autotune=True, capture=True)
capture.install()
ran = _run_one_driver(sys.argv[1])
n = capture.dump_shard(sys.argv[2])
print(f"ran={ran} ops={n} errors={capture.record_errors()}")
"""

READ = r"""
import json, sys; sys.path.insert(0, "src")
import torch
from miniworld_engine import settings
from miniworld_engine.autotune import capture
from miniworld_engine.autotune.configs import op_of
from miniworld_engine.autotune.builder import _run_one_driver
settings.configure(run_autotune=False, capture=False)
from triton.runtime.autotuner import Autotuner
seen = {}
orig = Autotuner.prune_configs
def prune_configs(self, kwargs):
    keep = orig(self, kwargs)
    op = op_of(getattr(self, "configs", None) or [])
    if op == sys.argv[1]:
        seen["n"] = (len(self.configs), len(keep))
        seen["kept"] = [str(c) for c in keep[:3]]
    return keep
Autotuner.prune_configs = prune_configs
_run_one_driver(sys.argv[1])
json.dump(seen, open(sys.argv[2], "w"))
"""


def _run(code: str, args: list[str], dtype: str, cfg: Path) -> str:
    env = {**os.environ, "PYTHONPATH": "src", "MINIWORLD_CONFIG_DIR": str(cfg),
           "MINIWORLD_DRIVER_LENGTH": str(LENGTH), "MINIWORLD_DRIVER_DTYPE": dtype}
    r = subprocess.run([".pixi/envs/default/bin/python", "-c", code, *args],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=5400)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise SystemExit(f"child failed at dtype={dtype}")
    return r.stdout


def main() -> int:
    tmp = REPO / ".bench/_dtypebucket"
    tmp.mkdir(parents=True, exist_ok=True)
    cfg = Path(os.environ.get("MINIWORLD_DTYPE_CFG") or (REPO / ".bench/dtypecfg"))

    shards = {}
    for dt in DTYPES:
        sh = tmp / f"{dt}.json"
        out = _run(BUILD, [OP, str(sh)], dt, cfg)
        d = json.loads(sh.read_text()).get(OP)
        print(f"  {dt}: {out.strip()}  shard buckets={list(d['entries']) if d else None}")
        shards[dt] = sh

    sys.path.insert(0, str(REPO / "src"))
    from miniworld_engine.autotune import capture
    from miniworld_engine.autotune.cache import gpu_key

    live = REPO / "src/miniworld_engine/autotune/data" / OP / f"{gpu_key()}.json"
    backup = tmp / "committed_backup.json"
    had = live.is_file()
    if had:
        shutil.copy2(live, backup)

    ok = False
    try:
        written = capture.merge_shards([str(s) for s in shards.values()], top_k=2, only_ops={OP})
        print(f"\nmerged {len(written)} entr(ies) into {live}")
        entries = json.loads(live.read_text())["entries"]
        print(f"  cache buckets: {sorted(entries)}")

        got_dtypes = {b.split("|", 1)[0] for b in entries}
        shapes = {b.split("|", 1)[1] for b in entries}
        want_dtypes = {"bfloat16", "float32"}
        distinct = len(entries) == 2
        correct = got_dtypes == want_dtypes and len(shapes) == 1
        print(f"\n  DISTINCT: {len(entries)} entries for {len(DTYPES)} dtypes -> "
              f"{'PASS' if distinct else 'FAIL'}")
        print(f"  CORRECT : dtype parts {sorted(got_dtypes)} (want {sorted(want_dtypes)}), "
              f"shape part(s) {sorted(shapes)} -> {'PASS' if correct else 'FAIL'}")
        winners = {k: v[0]["kwargs"] for k, v in entries.items()}
        print(f"  winners per bucket: {json.dumps(winners, sort_keys=True)}")

        read_ok = True
        for dt in DTYPES:
            probe = tmp / f"read-{dt}.json"
            _run(READ, [OP, str(probe)], dt, cfg)
            s = json.loads(probe.read_text())
            full, keep = s.get("n", (0, 0))
            tag = "NARROWED" if 0 < keep < full else "full grid"
            print(f"  READ BACK {dt}: {full} -> {keep}  {tag}")
            read_ok = read_ok and 0 < keep < full
        print(f"\n  READ BACK: {'PASS' if read_ok else 'FAIL'}")
        ok = distinct and correct and read_ok
    finally:
        if had:
            shutil.copy2(backup, live)
            print(f"  restored committed cache {live}")
        elif live.is_file():
            live.unlink()
            print(f"  removed probe-created cache {live}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
