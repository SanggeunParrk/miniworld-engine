"""Configuration selection and build-time measurement for native kernels.

Launchers supply pure, repeatable calls with an explicit configuration. Measurements
join the builder's ordinary shards; workers never publish to the runtime cache.
"""
from __future__ import annotations

import ast
import hashlib
import math
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from miniworld_engine import settings
from miniworld_engine.autotune.cache import (
    as_cfg_dict,
    build_rev,
    config_space_hash,
    gpu_key,
    select_config,
)

_WINNERS: dict = {}
BUILD_OPS = frozenset({
    "trimul_output_bwd_rows_sm90_cute",
    "trimul_inproj_masked_sm90_cute",
    "transition_squeeze_residual_sm90_cute",
    "layernorm_linear_fwd_foldstats_sm90_cute", "layernorm_linear_fwd_sm90_cute",
    "transition_swiglu_fwd_sm90_cute", "transition_gate_bwd_sm90_cute",
    "transition_bwd_dx_sm90_cute", "layernorm_linear_bwd_dx_sm90_cute",
    "trimul_outproj_gemm_gate_sm90_cute", "transition_fwd_b2b_sm90_cuda",
    "transition_expand_gate_sm90_cuda", "transition_bwd_gate_sm90_cuda",
    "layernorm_fwd_cuda", "layernorm_bwd_split_cuda",
})


def native_shape_supported(op, width, dtype):
    if op in ("layernorm_fwd_cuda", "layernorm_bwd_split_cuda"):
        if op == "layernorm_bwd_split_cuda":
            alignment = 128 if dtype == "bfloat16" else 64
            if width % alignment:
                return False
        return width <= 1024 and dtype in ("bfloat16", "float32")
    if dtype != "bfloat16":
        return False
    if op == "trimul_output_bwd_rows_sm90_cute":
        return width == 128
    if op == "transition_squeeze_residual_sm90_cute":
        return width == 512  # only this width is enabled in production dispatch
    if op.endswith("sm90_cuda"):
        return width in ((128, 256) if "b2b" in op else (128, 256, 512))
    if op in ("transition_bwd_dx_sm90_cute", "layernorm_linear_bwd_dx_sm90_cute"):
        return width <= 256  # full-N WGMMA epilogue reduction
    if op == "trimul_outproj_gemm_gate_sm90_cute":
        # Smallest m64 CTA: all four operands plus output staging must fit
        # Hopper's 227 KiB opt-in limit. This is an allocation constraint.
        padded = ((width + 63) // 64) * 64
        return (2 * (64 + padded) * padded + 64 * padded) * 2 + 4096 <= 232448
    return True


def build_ops_for_arch(arch):
    # WGMMA custom kernels explicitly require Hopper; SM100 is not a superset
    # of their launch ABI. Portable CUDA LayerNorm retains its own build pass.
    portable = {"layernorm_fwd_cuda", "layernorm_bwd_split_cuda"}
    return set(BUILD_OPS) if arch == "sm90" else portable


@lru_cache(None)
def source_identity() -> str:
    """Conservative native implementation identity, independent of the search grid.

    Changing candidate policy preserves measurements of unchanged kernels. Kernel,
    launcher, conversion/validation and benchmark changes still invalidate them.
    """
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    from importlib.metadata import PackageNotFoundError, version
    for package in ("quack-kernels", "nvidia-cutlass-dsl", "nvidia-mathdx"):
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "absent"
        digest.update(f"{package}={installed}".encode())
    paths = sorted((root / "kernels").rglob("*.py"))
    paths += sorted((root / "kernels").rglob("*.cu"))
    paths += [Path(__file__), Path(__file__).with_name("cute_config.py"),
              Path(__file__).with_name("hopper_cuda_config.py"),
              Path(__file__).with_name("native_compile.py"),
              Path(__file__).with_name("native_history.py")]
    for path in paths:
        if "notes" not in path.parts:
            digest.update(str(path.relative_to(root)).encode())
            if path.name == "cute_config.py":
                tree = ast.parse(path.read_text())
                # These helpers affect what a stored kwargs dictionary executes.
                keep = {"_TUNABLE_FIELDS", "config_to_kwargs", "kwargs_to_config",
                        "validate_hopper_config", "resolve_config"}
                for node in tree.body:
                    names = {getattr(node, "name", "")}
                    if isinstance(node, ast.Assign):
                        names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
                    if names & keep:
                        digest.update(ast.dump(node, include_attributes=False).encode())
            else:
                digest.update(path.read_bytes())
    return digest.hexdigest()


def policy_identity():
    """Search policy invalidates completed build units, never compatible timings."""
    digest = hashlib.sha256()
    for name in ("cute_config.py", "hopper_cuda_config.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def candidates_for(op, bucket):
    """CPU-readable declared grid for one exact native workload."""
    from miniworld_engine.autotune import hopper_cuda_config as cuda
    tensors, extra = ast.literal_eval(bucket)
    if op.endswith("sm90_cute"):
        from miniworld_engine.autotune import cute_config as cute
        if op in ("trimul_inproj_masked_sm90_cute", "transition_swiglu_fwd_sm90_cute",
                  "transition_gate_bwd_sm90_cute"):
            grid = cute.gated_sm90_candidates()
        elif op in ("trimul_output_bwd_rows_sm90_cute", "layernorm_linear_fwd_foldstats_sm90_cute",
                    "transition_squeeze_residual_sm90_cute"):
            grid = cute.plain_sm90_candidates()
        elif op == "layernorm_linear_fwd_sm90_cute":
            grid = cute.fused_lnl_candidates()
        elif op in ("layernorm_linear_bwd_dx_sm90_cute", "transition_bwd_dx_sm90_cute"):
            grid = cute.lnbwd_candidates(tensors[1][0][-1])
        elif op == "trimul_outproj_gemm_gate_sm90_cute":
            k, n = tensors[0][0][-1], tensors[2][0][0]
            kp, npad = (k + 63) // 64 * 64, (n + 15) // 16 * 16
            return [{"tile_m": c.tile_m} for c in cute.tm2_candidates()
                    if (2 * (c.tile_m + npad) * kp + c.tile_m * npad) * 2 + 4096 <= 232448]
        else:
            raise ValueError(f"no native grid for {op}")
        return [cute.config_to_kwargs(c) for c in grid]
    width = tensors[0][0][-1]
    if op.startswith("layernorm_"):
        kind = "fwd" if op == "layernorm_fwd_cuda" else "bwd"
        return cuda.layernorm_candidates(kind, width, 4 if "float32" in tensors[0][2] else 2)
    kind = {"transition_fwd_b2b_sm90_cuda": "b2b", "transition_bwd_gate_sm90_cuda": "gatebwd",
            "transition_expand_gate_sm90_cuda": "expand_gate"}[op]
    return cuda.candidates(kind, width)


def pending_candidates(op, data):
    """Count missing per-entry coverage without treating a file grid as evidence."""
    from miniworld_engine.autotune.cache import entry_space, _sig_from_dict
    pending = 0
    for key in data.get("entries", {}):
        _, bucket = key.split("|", 1)
        wanted = {repr(_sig_from_dict(as_cfg_dict({"kwargs": c})))
                  for c in candidates_for(op, bucket)}
        pending += len(wanted - (entry_space(data, key) or set()))
    return pending


def tensor_key(*tensors, extra=()) -> str:
    """Exact extents/strides/dtypes: native constraints need more than an M bucket."""
    parts = [(tuple(t.shape), tuple(t.stride()), str(t.dtype)) if t is not None else None
             for t in tensors]
    return repr((parts, extra))


def choose_config(op, candidates, *, dtype, bucket, device_index=None, run=None):
    """Resolve one config; in a build, measure every candidate once per exact workload.

    The first candidate is the documented cache-miss default. Invalid candidates are
    reported; an entirely failed round raises rather than claiming build completion.
    ``run`` must overwrite its outputs and must not mutate any input.
    """
    import torch

    dtype = dtype.removeprefix("torch.")
    if not candidates:
        raise ValueError(f"{op}: no configuration supports this workload")
    grid = [as_cfg_dict({"kwargs": dict(c)}) for c in candidates]
    if torch.compiler.is_compiling():
        return dict(candidates[0])
    identity = source_identity()
    if not settings.current().run_autotune or run is None:
        best = select_config(op, dtype=dtype, bucket=bucket, candidates=grid,
                             device_index=device_index, op_id=identity)
        return dict(best["kwargs"] if best else candidates[0])

    from miniworld_engine.autotune import capture
    # Portable CUDA builds also use this selector in environments without CuTe.
    try:
        from quack.cache import is_compile_only
    except ImportError:
        is_compile_only = None
    if is_compile_only is not None and is_compile_only():
        return dict(candidates[0])
    from miniworld_engine.autotune import cache, native_history
    from miniworld_engine.autotune.native_compile import precompile
    from triton.testing import do_bench

    measurement = {"scheme": 1, "kind": "native", "implementation": identity,
                   "bench_clear_mb": settings.current().bench_clear_mb,
                   "bench_rep_ms": settings.current().bench_rep_ms}
    gk = gpu_key(device_index)
    history_key = (op, gk, dtype, bucket, identity, cache.env_identity(),
                   build_rev(op), cache.KEY_SCHEME, measurement)
    key = (*history_key[:-1], cache.workload_id(measurement),
           config_space_hash(grid), capture._INCREMENTAL)
    if key in _WINNERS:
        return dict(_WINNERS[key])
    signature = lambda c: repr(cache._sig_from_dict(as_cfg_dict({"kwargs": dict(c)})))
    known = {}
    if capture._INCREMENTAL:
        data = cache._load(op, gk)
        if data and not cache.measurement_mismatch(op, data, identity):
            record = cache.workload_record(data, f"{dtype}|{bucket}", measurement) or {}
            rows = record.get("timings", record.get("entries", []))
            for row in rows:
                ms = row.get("ms")
                if isinstance(ms, (int, float)) and math.isfinite(ms) and ms > 0:
                    known[signature(row["kwargs"])] = {"status": "ok", "ms": ms}
            for sig in record.get("searched", []):
                known.setdefault(sig, {"status": "observed_failure"})

    ranked, failures, reused, measured, retryable = [], [], 0, 0, 0
    with native_history.session(capture._ROUND_CACHE_DIR, history_key) as history:
        if capture._INCREMENTAL:
            known.update(history.records)
        pending = [c for c in candidates if not native_history.reusable(known.get(signature(c)))]
        compiled = precompile(op, pending, bucket) if pending else {}
        pending_index = {signature(c): i for i, c in enumerate(pending)}
        with ExitStack() as stack:
            if device_index is not None:
                stack.enter_context(torch.cuda.device(device_index))
            capture._bench_lock_acquire()
            capture._NATIVE_LOCK_HELD = True
            def release():
                capture._NATIVE_LOCK_HELD = False
                capture._bench_lock_release()
            stack.callback(release)
            bencher = SimpleNamespace(_do_bench=do_bench)
            capture._use_a_smaller_bench_budget(bencher)
            for c in candidates:
                sig = signature(c)
                record = known.get(sig)
                if native_history.reusable(record):
                    reused += 1
                else:
                    result = compiled.get(pending_index[sig])
                    try:
                        if result is not None and result["status"] != "ok":
                            record = {"status": native_history.compile_failure_status(result),
                                      "diagnostic": str(result)}
                        else:
                            run(c)
                            torch.cuda.synchronize(device_index)
                            ms = float(bencher._do_bench(lambda c=c: run(c), quantiles=None,
                                                        return_mode="median"))
                            if not math.isfinite(ms) or ms <= 0:
                                raise RuntimeError(f"invalid measurement {ms}")
                            record = {"status": "ok", "ms": ms}
                            measured += 1
                    except Exception as exc:
                        # A poisoned CUDA context cannot supply trustworthy later timings.
                        if capture._fatal_cuda_error(exc):
                            raise
                        # Unknown launch failures/OOM can be transient. Never permanently
                        # exclude them or claim complete coverage; retry on the next build.
                        record = {"status": "retryable_failure",
                                  "diagnostic": f"{type(exc).__name__}: {exc}"}
                    history.record(sig, {**record, "config": dict(c)})
                if record["status"] == "ok":
                    ranked.append((record["ms"], c))
                    capture.record_native(op, grid, dtype, bucket, c, record["ms"], identity,
                                          measurement=measurement)
                else:
                    failures.append(f"{c}: {record.get('diagnostic', record['status'])}")
                    retryable += record["status"] == "retryable_failure"
                    if record["status"] == "observed_failure":
                        capture.record_native(op, grid, dtype, bucket, c, float("inf"), identity,
                                              measurement=measurement)
    capture._SKIPPED[op] = capture._SKIPPED.get(op, 0) + reused
    print(f"[native] {op}: {measured} new, {reused} reused, "
          f"{len(failures)} failed / {len(candidates)} candidates", flush=True)
    for failure in failures:
        print(f"[native] rejected {failure}", flush=True)
    if retryable:
        # Preserve positive checkpoints/shards, but prevent the builder from marking
        # this unit complete and skipping its failed candidates on --resume.
        raise RuntimeError(f"{op}: incomplete native tuning: {retryable} retryable failures ({bucket})")
    if not ranked:
        raise RuntimeError(f"{op}: every native configuration failed ({bucket})")
    winner = min(ranked, key=lambda item: item[0])[1]
    _WINNERS[key] = dict(winner)
    return dict(winner)


def reset():
    _WINNERS.clear()
