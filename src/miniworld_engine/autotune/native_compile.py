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
    tensors, extra = ast.literal_eval(bucket)
    if op == "transition_swiglu_fwd_sm90_cute" and extra not in ((), ("None",)):
        return None  # arbitrary activation callables have no portable compile contract
    if op.startswith("layernorm_") and op.endswith("cuda"):
        from miniworld_engine.autotune.hopper_cuda_config import layernorm_candidates
        config = (layernorm_candidates("compile", 128, 2)[0] if op == "layernorm_fwd_cuda"
                  else {k: config[k] for k in ("warps", "min_blocks")})
        tensors, extra = [], ()  # these CUDA extensions compile every supported dtype/width
    config = dict(config)
    if op.endswith("sm90_cute"):
        config.pop("max_swizzle_size", None)
    return {"op": op, "config": config, "tensors": tensors, "extra": extra}


def task_id(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()[:24]


def _dtype(meta):
    import cutlass
    names = {"bfloat16": "BFloat16", "float16": "Float16", "float32": "Float32"}
    return getattr(cutlass, names[meta[2].removeprefix("torch.")]) if meta is not None else None


def _major(meta, first, last):
    return (last if meta[1][-1] == 1 else first) if meta is not None else None


def compile_task(task):
    """Compile only. No tensor allocation on CUDA, launch, timing or cache ranking."""
    op, c, ts = task["op"], task["config"], task["tensors"]
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
    import cutlass
    from quack.cache import compile_only_mode
    if op == "trimul_outproj_gemm_gate_sm90_cute":
        import cutlass.cute as cute
        from quack.compile_utils import make_fake_tensor

        from miniworld_engine.kernels.tm2.cute.tm2_cute_kernel import TM2DualKernel
        k, n, tm = ts[0][0][-1], ts[2][0][0], c["tile_m"]
        kp, npad = (k + 63) // 64 * 64, (n + 15) // 16 * 16
        x = make_fake_tensor(cutlass.BFloat16, (tm, kp), leading_dim=1, divisibility=8)
        w = make_fake_tensor(cutlass.BFloat16, (npad, kp), leading_dim=1, divisibility=8)
        y = make_fake_tensor(cutlass.BFloat16, (tm, npad), leading_dim=1, divisibility=8)
        cute.compile(TM2DualKernel(npad, kp, tm), x, x, w, w, y)
        return
    tile = (c["tile_m"], c["tile_n"])
    cluster = (c["cluster_m"], c["cluster_n"], 1)
    pp, dyn, device = c["pingpong"], c["is_dynamic_persistent"], (9, 0)
    a, b = ts[:2]
    with compile_only_mode():
        if op == "transition_squeeze_residual_sm90_cute":
            from quack.gemm import _compile_gemm
            _compile_gemm(
                a_dtype=_dtype(a), b_dtype=_dtype(b), d_dtype=_dtype(ts[2]), c_dtype=_dtype(ts[2]),
                a_major=_major(a, "m", "k"), b_major=_major(b, "n", "k"),
                d_major="n", c_major=_major(ts[2], "m", "n"),
                tile_shape_mn=tile, cluster_shape_mnk=cluster, pingpong=pp,
                persistent=True, is_dynamic_persistent=dyn,
                rowvec_dtype=None, colvec_dtype=None, colvec_ndim=0,
                alpha_mode=0, beta_mode=0, add_to_output=False, concat_layout=None,
                varlen_m=False, varlen_k=False, gather_A=False, use_tma_gather=False,
                has_batch_idx_permute=False, device_capacity=device, rounding_mode=0,
                sr_seed_mode=0, has_trace_ptr=False, num_warps=None,
            )
        elif op == "trimul_inproj_masked_sm90_cute":
            from miniworld_engine.kernels.trimul_inproj.cute.masked_front import _compile_masked_front
            _compile_masked_front(_dtype(a), _major(a, "m", "k"), _major(b, "k", "n"),
                                  task["extra"][0], tile, cluster, pp, dyn, device)
        elif op == "layernorm_linear_fwd_foldstats_sm90_cute":
            from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear import (
                _compile_gemm_lnl,
            )
            _compile_gemm_lnl(_dtype(a), _dtype(b), _dtype(ts[2]),
                              _major(a, "m", "k"), _major(b, "n", "k"), _major(ts[2], "m", "n"),
                              _dtype(ts[3]), tile, cluster, pp, True, dyn, device)
        elif op == "layernorm_linear_fwd_sm90_cute":
            from miniworld_engine.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
                _compile_fused,
            )
            gate = ts[5]
            if len(ts) > 6 and ts[6] is not None:
                raise ValueError("M2 debug-output ABI is not enabled")
            _compile_fused(_dtype(a), _dtype(b), _dtype(ts[2]),
                           _major(a, "m", "k"), _major(b, "n", "k"), _major(ts[2], "m", "n"),
                           _dtype(ts[3]), device, (*tile, *cluster[:2], pp),
                           _dtype(gate), _major(gate, "m", "n"), bool(task["extra"][0]))
        elif op == "transition_swiglu_fwd_sm90_cute":
            from miniworld_engine.kernels.transition.cute.gemm_transition_swiglu import (
                _compile_gemm_ln_swiglu,
            )
            _compile_gemm_ln_swiglu(_dtype(a), _dtype(b), _dtype(ts[2]),
                                   _major(a, "m", "k"), _major(b, "n", "k"), _major(ts[2], "m", "n"),
                                   _dtype(ts[3]), tile, cluster, pp, dyn, device, None)
        elif op == "transition_gate_bwd_sm90_cute":
            from miniworld_engine.kernels.transition.cute.backward_gatebwd import (
                _compile_gemm_dln_gatebwd,
            )
            _compile_gemm_dln_gatebwd(
                _dtype(a), _dtype(b), _dtype(ts[2]), _dtype(ts[4]), _dtype(ts[3]),
                _major(a, "m", "k"), _major(b, "n", "k"), _major(ts[2], "m", "n"),
                _major(ts[4], "m", "n"), _major(ts[3], "m", "n"),
                _dtype(ts[5]), tile, cluster, pp, dyn, device)
        elif op == "trimul_output_bwd_rows_sm90_cute":
            from miniworld_engine.kernels.layernorm_linear.cute.dgrad_ln_rows import _compile
            _compile(_dtype(a), _dtype(b), _dtype(a), _dtype(ts[2]),
                     _major(a, "m", "k"), "k", "m", _major(ts[2], "m", "n"),
                     cutlass.Float32, tile, cluster, pp, True, dyn, device)
        elif op == "layernorm_linear_bwd_dx_sm90_cute":
            from miniworld_engine.kernels.layernorm_linear.cute.dgrad_lnbwd import (
                _compile,
            )
            _compile(_dtype(a), _dtype(b), _dtype(a), _dtype(ts[2]),
                     _major(a, "m", "k"), "k", "n", _major(ts[2], "m", "n"),
                     _dtype(ts[4]), tile, cluster, pp, True, dyn, device)
        elif op == "transition_bwd_dx_sm90_cute":
            from miniworld_engine.kernels.transition.cute.dab_lnbwd import _compile
            _compile(1, _dtype(a), _dtype(b), _dtype(a), _dtype(ts[2]),
                     "k", "k", "n", "n", _dtype(ts[4]), tile, cluster, pp, True, dyn, device)
        else:
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
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CUTE_DSL_ARCH="sm_90a",
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
    # TM2 currently uses a process-local CuTe callable, so exporting a separate
    # object would not warm its launch path. It still participates in CPU audits.
    if op == "trimul_outproj_gemm_gate_sm90_cute":
        return {}
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
