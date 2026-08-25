"""The whole chain, across MORE THAN ONE shape bucket.

The earlier reader probe drove a single shape, so it could not tell a cache that discriminates by
shape from one that does not -- which is exactly the bug that shipped (`any|any`) and the one the
driver sweep just fixed (every length clamped to one bucket). This drives one op at several
lengths, merges, and then checks three things that a one-bucket run cannot:

  DISTINCT   the merged cache holds one entry PER shape bucket, not one entry total
  CORRECT    each entry's bucket names the length it was driven at
  READ BACK  a fresh process at length L narrows the grid using L's entry -- and the winners for
             two different lengths are not forced to be the same config
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OP = "layernorm_fwd_saveact_triton"          # level=both, real grid, cheap to drive
LENGTHS = (256, 512, 1024)

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


def _run(code: str, args: list[str], length: int, cfg: Path) -> str:
    env = {**os.environ, "PYTHONPATH": "src", "MINIWORLD_CONFIG_DIR": str(cfg),
           "MINIWORLD_DRIVER_LENGTH": str(length)}
    r = subprocess.run([".pixi/envs/default/bin/python", "-c", code, *args],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-1500:])
        raise SystemExit(f"child failed at L={length}")
    return r.stdout


def main() -> int:
    tmp = REPO / ".bench/_multibucket"
    tmp.mkdir(parents=True, exist_ok=True)
    cfg = REPO / "configs/grid"

    shards = []
    for L in LENGTHS:
        sh = tmp / f"L{L}.json"
        out = _run(BUILD, [OP, str(sh)], L, cfg)
        d = json.loads(sh.read_text()).get(OP)
        print(f"  L={L:5d}  {out.strip()}  buckets={list(d['entries']) if d else None}")
        shards.append(sh)

    sys.path.insert(0, str(REPO / "src"))
    from miniworld_engine.autotune import capture

    written = capture.merge_shards([str(s) for s in shards], top_k=3, only_ops={OP})
    print(f"\nmerged {len(written)} entr(ies)")
    fp = Path(written[0][-1])
    entries = json.loads(fp.read_text())["entries"]
    print(f"  cache buckets: {sorted(entries)}")

    # Compare the shape_key FIELD, not the whole bucket string: the string also carries the
    # kernel's other key entries (N, HAS_ROWSCALE, ...), and hardcoding them made this report FAIL
    # on a cache that was in fact correct.
    got_keys = {int(part.split("=")[1])
                for b in entries for part in b.split(",") if part.startswith("shape_key=")}
    print(f"\n  DISTINCT: {len(entries)} entries for {len(LENGTHS)} lengths -> "
          f"{'PASS' if len(entries) == len(LENGTHS) else 'FAIL'}")
    print(f"  CORRECT : shape_key values {sorted(got_keys)} vs driven {sorted(LENGTHS)} -> "
          f"{'PASS' if got_keys == set(LENGTHS) else 'FAIL'}")
    got = got_keys
    want = set(LENGTHS)
    winners = {k: v[0]["kwargs"] for k, v in entries.items()}
    print(f"  winners per bucket: {json.dumps(winners, sort_keys=True)}")

    ok = True
    for L in LENGTHS:
        probe = tmp / f"read-L{L}.json"
        _run(READ, [OP, str(probe)], L, cfg)
        s = json.loads(probe.read_text())
        full, keep = s.get("n", (0, 0))
        tag = "NARROWED" if 0 < keep < full else "full grid"
        print(f"  READ BACK L={L:5d}: {full} -> {keep}  {tag}")
        ok = ok and 0 < keep < full
    print(f"\n  READ BACK: {'PASS' if ok else 'FAIL'}")
    return 0 if (len(entries) == len(LENGTHS) and got == want and ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
