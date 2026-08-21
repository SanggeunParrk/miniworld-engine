"""End-to-end: two CONFIG-SET shards -> capture -> merge -> is the cache actually right?

Everything fixed today only matters if this holds on real hardware with a real module run:

  1. a shard dir that carries a slice for its target op and one filler for the rest LAUNCHES
     (nothing dies with "dynamic_func() missing required positional arguments")
  2. captured entries are split by SHAPE BUCKET, not lumped into ``any|any``
  3. the merged cache records the hash of the FULL grid, so a later full-grid run keeps it
     instead of resetting every entry

Each shard runs in its OWN process, because the config directory is chosen at
``autotune.configs`` import time -- necessarily before any kernel module imports.

WARNING: ``merge_shards`` is the real thing and writes into the COMMITTED cache tree
(``src/miniworld_engine/autotune/data/``). This probe's numbers come from a 6-config slice, so
those writes are junk -- revert them afterwards:

    git checkout -- src/miniworld_engine/autotune/data/ && git status --short
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OP = ""
CASE = "gated_projection"
DIMS, LENGTH, MODE = "0", "256", "eval"


def build_shards(tmp: Path, n_each: int = 6) -> list[Path]:
    """Two dirs: shard 0 gets the first n_each configs of OP's GENERATED grid, shard 1 the next.

    Source is gen_grid, not configs/accuracy -- accuracy holds exactly one config per op, so
    slicing it cannot produce two disjoint shards at all. Every other op keeps its accuracy config
    so the module can still run; only OP's two slices matter to the merge under test.
    """
    sys.path.insert(0, str(REPO / "tools/kernel-audit"))
    from gen_shards import full_grid                                     # noqa: PLC0415

    grid = full_grid()[OP]
    assert len(grid) >= 2 * n_each, f"{OP} grid has only {len(grid)} configs"
    axes = [k for k in grid[0] if k not in ("num_warps", "num_stages")]
    cols = [*axes, "num_warps", "num_stages"]
    src = REPO / "configs/accuracy"
    out = []
    for i in range(2):
        d = tmp / f"cfg{i}"
        d.mkdir(parents=True, exist_ok=True)
        for p in src.glob("*.csv"):
            allrows = list(csv.DictReader(p.open(newline="")))
            if p.stem == OP:
                with (d / p.name).open("w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=cols)
                    w.writeheader()
                    w.writerows(grid[i * n_each:(i + 1) * n_each])
            else:
                with (d / p.name).open("w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(allrows[0]))
                    w.writeheader()
                    w.writerows(allrows[:1])
        out.append(d)
    return out


def _run(shard: Path, cfg: Path | None) -> tuple[int, str]:
    import os                                                            # noqa: PLC0415
    cmd = [".pixi/envs/default/bin/python", "-m", "miniworld_engine.autotune.builder",
           "--case", CASE, "--dims", DIMS, "--length", LENGTH, "--mode", MODE,
           "--shard", str(shard)]
    if cfg is not None:
        cmd += ["--config-dir", str(cfg)]
    r = subprocess.run(cmd, cwd=REPO, env={**os.environ, "PYTHONPATH": "src"},   # noqa: S603
                       capture_output=True, text=True, timeout=3600)
    return r.returncode, r.stdout + r.stderr


def discover_op(tmp: Path) -> str:
    """Which op does this case actually dispatch to? Run it once and look.

    Picking the op by name failed twice (layernorm_linear_pair_bias -> nothing;
    gated_projection -> gate_triton, not gate_gemm_triton). The case's dispatch depends on dims,
    length and mode, so the only reliable source is a real run.
    """
    sh = tmp / "discover.json"
    rc, out = _run(sh, REPO / "configs/accuracy")
    if rc != 0:
        raise SystemExit(f"discovery run failed:\n{out[-2500:]}")
    fired = sorted(json.loads(sh.read_text()))
    sys.path.insert(0, str(REPO / "tools/kernel-audit"))
    from gen_shards import full_grid                                     # noqa: PLC0415

    grid = full_grid()
    cand = [(len(grid.get(o, [])), o) for o in fired if len(grid.get(o, [])) >= 12]
    print(f"  case {CASE} fired {len(fired)} op(s): {fired}")
    if not cand:
        raise SystemExit(f"none of {fired} has a grid big enough to split")
    n, op = max(cand)
    print(f"  -> testing {op} (grid {n:,})")
    return op


def main() -> int:
    global OP                                                            # noqa: PLW0603
    tmp = REPO / ".bench/_e2e"
    tmp.mkdir(parents=True, exist_ok=True)
    OP = discover_op(tmp)
    cfgs = build_shards(tmp)
    shards = []
    for i, cfg in enumerate(cfgs):
        sh = tmp / f"shard{i}.json"
        print(f"--- shard {i}: {cfg.name}", flush=True)
        rc, out = _run(sh, cfg)
        tail = [ln for ln in out.splitlines() if OP in ln or "unit ran" in ln][-4:]
        print("\n".join("    " + t for t in tail) or f"    rc={rc}", flush=True)
        if rc != 0:
            print(out[-2500:])
            return 1
        shards.append(sh)

    sys.path.insert(0, str(REPO / "src"))
    from miniworld_engine.autotune import cache, capture

    # what each shard carried + measured, before merging
    per = []
    for sh in shards:
        d = json.loads(sh.read_text()).get(OP)
        if d is None:
            # Naming the op is not enough -- the CASE has to actually dispatch to it. Say which
            # ops did fire, so the next pick is informed instead of another guess.
            print(f"!! {sh.name} captured nothing for {OP}; it captured: "
                  f"{sorted(json.loads(sh.read_text()))}")
            return 1
        per.append(d)
        print(f"  {sh.name}: grid={len(d['grid'])} buckets={list(d['entries'])}")

    written = capture.merge_shards([str(s) for s in shards], only_ops={OP})
    print(f"\nmerge wrote {len(written)} entr(ies)")
    fp = Path(written[0][-1])
    data = json.loads(fp.read_text())

    class _C:
        def __init__(s, d):
            s.kwargs, s.num_warps, s.num_stages, s.maxnreg = (
                d["kwargs"], d["num_warps"], d["num_stages"], None)

    union = {cache._sig_from_dict(c): c for d in per for c in d["grid"]}   # noqa: SLF001
    want = cache.config_space_hash([_C(c) for c in union.values()])
    print(f"  union of shard grids : {len(union)} configs -> hash {want}")
    print(f"  cache config_space_hash: {data['config_space_hash']}")
    print(f"  HASH MATCHES UNION   : {data['config_space_hash'] == want}")
    print(f"  entries (buckets)    : {list(data['entries'])}")
    bad = [b for b in data["entries"] if b.endswith("|any") or b.startswith("any|")]
    print(f"  degenerate 'any' buckets: {bad or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
