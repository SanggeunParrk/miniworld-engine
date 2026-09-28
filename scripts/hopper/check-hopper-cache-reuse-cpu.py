"""Verify real cross-process CuTe object reuse on a CPU compute node.

The first pass compiles through the production native worker. Each second-pass
process has cute.compile replaced by a failure: successful completion therefore
requires loading the persistent object, not just recompiling the same ABI.
"""
import argparse
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

from miniworld_engine.autotune.native_compile import compile_task, run_tasks, task_id


def reuse(request):
    import cutlass.cute as cute
    import quack.cache

    assert quack.cache.CACHE_ENABLED, "persistent Quack cache must be enabled"

    def forbidden_compile(*args, **kwargs):
        raise AssertionError("persistent cache miss: second process tried to compile")

    cute.compile = forbidden_compile
    compile_task(json.loads(request.read_text()))
    print("PASS persistent object loaded with compiler disabled", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--reuse", type=Path)
    args = parser.parse_args()
    if args.reuse is not None:
        reuse(args.reuse)
        return
    if args.output is None:
        parser.error("--output is required")
    matrix = runpy.run_path(str(Path(__file__).with_name("check-hopper-candidate-matrix.py")))
    selected = {}
    for task in matrix["cases"]():
        op = task["op"]
        if not op.endswith("sm90_cute") or op.startswith("trimul_"):
            continue  # TM2 has no persistent runtime object cache
        if op == "layernorm_linear_fwd_sm90_cute":
            key = (op, task["tensors"][0][1][-1] == 1,
                   task["tensors"][5] is None, bool(task["extra"][0]))
        else:
            key = (op,)
        selected.setdefault(key, task)
    results = run_tasks(selected.values(), jobs=args.jobs, directory=args.output)
    assert all(r["status"] == "ok" for r in results.values()), "warm pass failed; inspect logs"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CUTE_DSL_ARCH="sm_90a",
               PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for task in selected.values():
        key = task_id(task)
        with (args.output / f"{key}.reuse.log").open("w") as log:
            subprocess.run([sys.executable, __file__, "--reuse",
                            str(args.output / f"{key}.request.json")],
                           env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=180)
    summary = {"warmed": len(results), "loaded_without_compiler": len(results)}
    (args.output / "reuse-summary.json").write_text(json.dumps(summary, indent=2))
    print(f"PASS {summary}", flush=True)


if __name__ == "__main__":
    main()
