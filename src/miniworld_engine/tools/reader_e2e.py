"""Does the shipped cache actually narrow a real kernel's grid in a fresh process?

The unit tests drive `_cached_subset` directly. This drives the whole loop on real hardware:

  pass 1  run_autotune=True  -> bench a multi-config grid, capture, merge into the cache
  pass 2  run_autotune=False -> a NEW process launches the same kernel and must bench only the
                                cached top-K, not the whole grid

Pass 2 is the half that has not existed since fcd3c7a. Two processes, because the config
directory is chosen at `autotune.configs` import time and the reader patches Autotuner.__init__.

This is the probe that caught the reader silently disabling capture: it reported 8 configs
compiled and 0 ops captured, where every unit test passed. The unit tests build a fake autotuner
with `early_config_prune = None` -- precisely the case that still worked.

WARNING: pass 1 merges through the real `merge_shards`, so it writes into the COMMITTED cache
tree. Revert afterwards:

    git checkout -- src/miniworld_engine/autotune/data/ && git status --short
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CASE = "gated_projection"
N_CFG = 8


def _spec_dir(tmp: Path) -> Path:
    """A config dir with a real multi-config grid for every op (slice keeps it cheap)."""
    from miniworld_engine.tools.gen_grid import op_axes
    from miniworld_engine.tools.gen_shards import full_grid

    grid, axes = full_grid(), op_axes()
    d = tmp / "cfg"
    d.mkdir(parents=True, exist_ok=True)
    for op, cfgs in grid.items():
        rows = cfgs[:N_CFG]
        cols = [*axes[op], "num_warps", "num_stages"]
        with (d / f"{op}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows({c: r[c] for c in cols} for r in rows)
    return d


CHILD = r"""
import json, os, sys
sys.path.insert(0, "src")
import torch
from miniworld_engine import settings
from miniworld_engine.autotune import capture
from miniworld_engine.autotune.builder import cases, run_case

mode, out = sys.argv[1], sys.argv[2]
seen = {}

if mode == "build":
    settings.configure(run_autotune=True, capture=True)
    capture.install()
else:
    # Count what SURVIVES the prune for each op: that is the reader's whole effect.
    settings.configure(run_autotune=False, capture=False)
    from triton.runtime.autotuner import Autotuner
    from miniworld_engine.autotune.configs import op_of
    orig = Autotuner.prune_configs
    def prune_configs(self, kwargs):
        keep = orig(self, kwargs)
        op = op_of(getattr(self, "configs", None) or [])
        if op:
            seen[op] = (len(self.configs), len(keep))
        return keep
    Autotuner.prune_configs = prune_configs

case = next(c for c in cases() if c.name == sys.argv[3])
run_case(case, 256, 0, train=False, dtype=torch.bfloat16)
torch.cuda.synchronize()

if mode == "build":
    print("capture summary:"); print(capture.summary())
    print("launched ops:", capture.launched_ops())
    capture.dump_shard(out)
else:
    json.dump(seen, open(out, "w"))
"""


def _run(mode: str, out: Path, cfg: Path) -> str:
    env = {**os.environ, "PYTHONPATH": "src", "MINIWORLD_CONFIG_DIR": str(cfg)}
    r = subprocess.run([".pixi/envs/default/bin/python", "-c", CHILD, mode, str(out), CASE],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise SystemExit(f"{mode} pass failed")
    return r.stdout


def main() -> int:
    tmp = REPO / ".bench/_reader"
    tmp.mkdir(parents=True, exist_ok=True)
    cfg = _spec_dir(tmp)

    shard = tmp / "shard.json"
    out = _run("build", shard, cfg)
    for ln in out.splitlines()[-25:]:
        print("    |", ln)
    captured = json.loads(shard.read_text())
    print(f"pass 1 (build): captured {len(captured)} op(s)")

    sys.path.insert(0, str(REPO / "src"))
    from miniworld_engine.autotune import capture as cap

    written = cap.merge_shards([str(shard)], top_k=3)
    print(f"pass 1: merged {len(written)} entr(ies) into the cache")
    for op, bk, n, _ in written[:4]:
        print(f"    {op}  {bk}  {n} configs ranked")

    probe = tmp / "probe.json"
    _run("read", probe, cfg)
    seen = json.loads(probe.read_text())
    print(f"\npass 2 (fresh process, run_autotune=False): {len(seen)} op(s) launched")
    narrowed = {o: v for o, v in seen.items() if v[1] < v[0]}
    print("    grid -> pruned, per op:")
    for op, (full, keep) in sorted(seen.items()):
        tag = "NARROWED" if keep < full else "full grid"
        print(f"      {op:46s} {full:3d} -> {keep:3d}   {tag}")
    print(f"\n  ops the cache narrowed: {len(narrowed)}/{len(seen)}")
    return 0 if narrowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
