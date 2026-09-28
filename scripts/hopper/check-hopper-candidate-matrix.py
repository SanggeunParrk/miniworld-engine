"""CPU-only candidate audit. Run on a compute node, never the login node."""
import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path

from miniworld_engine.autotune.cute_config import (
    config_to_kwargs,
    fused_lnl_candidates,
    gated_sm90_candidates,
    lnbwd_candidates,
    plain_sm90_candidates,
    tm2_candidates,
)
from miniworld_engine.autotune.hopper_cuda_config import (
    candidates,
    layernorm_candidates,
)
from miniworld_engine.autotune.native_compile import run_tasks, task_for, task_id


def meta(rows, cols, dtype="bfloat16", major="k"):
    return ((rows, cols), (cols, 1) if major == "k" else (1, rows), f"torch.{dtype}")


def cases():
    vec = meta(1, 128, "float32")
    x, w, y = meta(128, 128), meta(128, 128), meta(128, 128)
    tasks = []
    def add(op, grid, ts, extra=()):
        tasks.extend(task_for(op, c, repr((ts, extra))) for c in grid)
    plain = [config_to_kwargs(c) for c in plain_sm90_candidates()]
    gated = [config_to_kwargs(c) for c in gated_sm90_candidates()]
    fused = [config_to_kwargs(c) for c in fused_lnl_candidates()]
    for major in ("k", "m"):
        a = meta(128, 128, major=major)
        add("layernorm_linear_fwd_foldstats_sm90_cute", plain, [a, w, y, vec, vec, vec, vec])
        for gate, ws in product((None, y), (False, True)):
            add("layernorm_linear_fwd_sm90_cute", fused, [a, w, y, vec, vec, gate, None], (ws, 1e-5))
    add("transition_swiglu_fwd_sm90_cute", gated, [x, w, y, vec, vec, vec, vec], ("None",))
    add("transition_gate_bwd_sm90_cute", gated, [x, w, y, y, y, vec, vec, vec, vec])
    for width in (64, 128, 192, 256):
        grid = [config_to_kwargs(c) for c in lnbwd_candidates(width)]
        a, weight, normalized = meta(128, 4 * width), meta(4 * width, width), meta(128, width)
        add("layernorm_linear_bwd_dx_sm90_cute", grid, [a, weight, normalized, vec, vec])
        add("transition_bwd_dx_sm90_cute", grid, [a, weight, normalized, vec, vec, vec])
    for width in (16, 32, 48, 64, 128, 192):
        kp, npad = (width + 63) // 64 * 64, (width + 15) // 16 * 16
        grid = [{"tile_m": c.tile_m} for c in tm2_candidates()
                if (2 * (c.tile_m + npad) * kp + c.tile_m * npad) * 2 + 4096 <= 232448]
        a, weight = meta(128, width), meta(width, width)
        add("trimul_outproj_gemm_gate_sm90_cute", grid, [a, a, weight, weight])
    for kind, width in product(("b2b", "expand_gate", "gatebwd"), (128, 256, 512)):
        if kind == "b2b" and width == 512:
            continue
        op = {"b2b": "transition_fwd_b2b_sm90_cuda", "gatebwd": "transition_bwd_gate_sm90_cuda",
              "expand_gate": "transition_expand_gate_sm90_cuda"}[kind]
        add(op, candidates(kind, width), [meta(128, width)])
    add("layernorm_fwd_cuda", layernorm_candidates("fwd", 128, 2), [x])
    add("layernorm_bwd_split_cuda", layernorm_candidates("bwd", 256, 2), [x])
    return list({task_id(t): t for t in tasks}.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("all", "cute", "cuda"), default="all")
    args = parser.parse_args()
    tasks = [t for t in cases() if args.backend == "all" or t["op"].endswith(args.backend)]
    print(f"compile {len(tasks)} distinct candidates with {args.jobs} workers", flush=True)
    results = run_tasks(tasks, jobs=args.jobs, timeout=args.timeout, directory=args.output)
    counts = Counter(r["status"] for r in results.values())
    (args.output / "summary.json").write_text(json.dumps({"counts": counts, "results": results}, indent=2))
    print(dict(counts), flush=True)
    return int(any(r["status"] != "ok" for r in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
