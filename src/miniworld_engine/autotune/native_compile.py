"""Isolated CPU precompilation for native candidates, shared by build and audit.

Only tensor metadata crosses the process boundary. Each compiler runs in a fresh
interpreter with CUDA devices hidden; a crash or timeout cannot kill the tuner.
"""
from __future__ import annotations

import ast
import concurrent.futures
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def task_for(op, config, bucket):
    from miniworld_engine.autotune.native import BUILD_OPS
    if op not in BUILD_OPS:
        return None
    if op == "trimul_fwd_sm90_cuda":
        return None  # payload build and driver context are owned by the allocated GPU process
    tensors, extra = ast.literal_eval(bucket)
    if op.startswith("layernorm_") and op.endswith("cuda"):
        from miniworld_engine.autotune.hopper_cuda_config import layernorm_candidates
        config = (layernorm_candidates("compile", 128, 2)[0] if op == "layernorm_fwd_cuda"
                  else {k: config[k] for k in ("warps", "min_blocks")})
        tensors, extra = [], ()  # these CUDA extensions compile every supported dtype/width
    config = dict(config)
    return {"op": op, "config": config, "tensors": tensors, "extra": extra}


def task_id(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()[:24]


def compile_task(task):
    """Compile only. No tensor allocation on CUDA, launch, timing or cache ranking."""
    op, c, ts = task["op"], task["config"], task["tensors"]
    if op in ("transition_fwd_residual_sm90_cuda", "transition_bwd_residual_sm90_cuda"):
        from miniworld_engine.kernels.transition.cuda.fused_sm90a import _ext
        _ext(c["ctas"], c["dw_repl"], bool(task["extra"][0]))
        return
    if op.endswith("sm90_cuda"):
        from miniworld_engine.kernels.transition.cuda import _ext
        kind = {"transition_fwd_b2b_sm90_cuda": "b2b", "transition_bwd_gate_sm90_cuda": "gatebwd",
                "transition_expand_gate_sm90_cuda": "expand_gate"}[op]
        _ext(kind, ts[0][0][-1], c)
        return
    if op in ("layernorm_fwd_cuda", "layernorm_bwd_split_cuda"):
        from miniworld_engine.kernels.layernorm.cuda import _ext
        _ext(c)
        return
    raise ValueError(f"no native compile contract for {op}")


def _run_one(task, directory, timeout, env):
    from miniworld_engine._atomic import write_json
    key = task_id(task)
    request = directory / f"{key}.request.json"
    log = directory / f"{key}.log"
    write_json(request, task)
    started = time.monotonic()
    with log.open("w") as output:
        process = subprocess.Popen([sys.executable, "-m", __name__, "--worker", str(request)],
                                   stdout=output, stderr=subprocess.STDOUT, env=env,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            status = "ok" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code, status = process.returncode, "timeout"
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    result = {"task": task, "status": status, "returncode": code,
              "seconds": time.monotonic() - started, "log": str(log)}
    write_json(directory / f"{key}.result.json", result)
    return result


def run_tasks(tasks, *, jobs, directory, timeout=300):
    """Run distinct tasks with bounded CPU concurrency and per-candidate diagnostics."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MAX_JOBS="1")
    unique = {task_id(task): task for task in tasks}
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    jobs = max(1, min(jobs, cores, len(unique) or 1))
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_run_one, task, directory, timeout, env): key
                   for key, task in unique.items()}
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            results[key] = future.result()
            r = results[key]
            print(f"[native compile] {len(results)}/{len(unique)} {r['task']['op']} "
                  f"{r['status']} {r['seconds']:.1f}s ({key})", flush=True)
    return results


def precompile(op, candidates, bucket):
    """Warm persistent native objects before taking the GPU benchmark lock."""
    tasks = [task_for(op, c, bucket) for c in candidates]
    if not tasks or any(t is None for t in tasks):
        return {}
    from miniworld_engine.autotune import capture
    from miniworld_engine.autotune.native import source_identity
    directory = Path(os.environ.get("XDG_CACHE_HOME", "/tmp")) / "miniworld-native-compile"
    directory = directory / source_identity() / str(os.getpid())
    results = run_tasks(tasks, jobs=capture._compile_jobs(), directory=directory)
    return {i: results[task_id(task)] for i, task in enumerate(tasks)}


if __name__ == "__main__":
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("usage: python -m miniworld_engine.autotune.native_compile --worker task.json")
    compile_task(json.loads(Path(sys.argv[2]).read_text()))
