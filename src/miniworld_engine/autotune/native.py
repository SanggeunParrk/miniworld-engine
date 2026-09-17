"""Configuration selection and build-time measurement for native kernels.

Launchers supply pure, repeatable calls with an explicit configuration. Measurements
join the builder's ordinary shards; workers never publish to the runtime cache.
"""
from __future__ import annotations

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
    """Native source and configuration policy both affect measured winners."""
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
              Path(__file__).with_name("native_compile.py")]
    for path in paths:
        if "notes" not in path.parts:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


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
    key = (op, gpu_key(device_index), dtype, bucket, config_space_hash(grid), identity)
    key += (build_rev(op), settings.current().bench_clear_mb, settings.current().bench_rep_ms)
    if key in _WINNERS:
        return dict(_WINNERS[key])
    from triton.testing import do_bench

    from miniworld_engine.autotune.native_compile import precompile

    compiled = precompile(op, candidates, bucket)

    ranked, failures = [], []
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
        for index, c in enumerate(candidates):
            try:
                result = compiled.get(index)
                if result is not None and result["status"] != "ok":
                    raise RuntimeError(f"CPU precompile {result['status']}; see {result['log']}")
                run(c)  # compile and surface launch errors before timing
                torch.cuda.synchronize(device_index)
                ms = float(bencher._do_bench(lambda c=c: run(c), quantiles=None,
                                            return_mode="median"))
                if not math.isfinite(ms) or ms <= 0:
                    raise RuntimeError(f"invalid measurement {ms}")
            except Exception as exc:
                failures.append(f"{c}: {type(exc).__name__}: {exc}")
                capture.record_native(op, grid, dtype, bucket, c, float("inf"), identity)
                continue
            ranked.append((ms, c))
            capture.record_native(op, grid, dtype, bucket, c, ms, identity)
    print(f"[native] {op}: {len(ranked)}/{len(candidates)} measured; "
          f"{len(failures)} failed", flush=True)
    if failures:
        # Full diagnostics go to the unit log, including unsupported candidates.
        for failure in failures:
            print(f"[native] rejected {failure}", flush=True)
    if not ranked:
        raise RuntimeError(f"{op}: every native configuration failed ({bucket})")
    winner = min(ranked, key=lambda item: item[0])[1]
    _WINNERS[key] = dict(winner)
    return dict(winner)


def reset():
    _WINNERS.clear()
