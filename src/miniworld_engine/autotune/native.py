"""Configuration selection and build-time measurement for native kernels.

Launchers supply pure, repeatable calls with an explicit configuration. Measurements
join the builder's ordinary shards; workers never publish to the runtime cache.
"""
from __future__ import annotations

import ast
import hashlib
import math
from collections.abc import Sequence
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
    "trimul_fwd_sm90_cuda",
    "transition_fwd_residual_sm90_cuda", "transition_bwd_residual_sm90_cuda",
    "transition_fwd_b2b_sm90_cuda",
    "transition_expand_gate_sm90_cuda", "transition_bwd_gate_sm90_cuda",
    "layernorm_fwd_cuda", "layernorm_bwd_split_cuda",
})


def native_shape_supported(op, width, dtype):
    if op == "trimul_fwd_sm90_cuda":
        return width in (64, 128, 256, 384, 512) and dtype == "bfloat16"
    if op in ("transition_fwd_residual_sm90_cuda", "transition_bwd_residual_sm90_cuda"):
        return width == 128 and dtype == "bfloat16"
    if op in ("layernorm_fwd_cuda", "layernorm_bwd_split_cuda"):
        if op == "layernorm_bwd_split_cuda":
            alignment = 128 if dtype == "bfloat16" else 64
            if width % alignment:
                return False
        return width <= 1024 and dtype in ("bfloat16", "float32")
    if dtype != "bfloat16":
        return False
    if op.endswith("sm90_cuda"):
        return width in ((128, 256) if "b2b" in op else (128, 256, 512))
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
    for package in ("nvidia-mathdx",):
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "absent"
        digest.update(f"{package}={installed}".encode())
    paths = sorted((root / "kernels").rglob("*.py"))
    paths += sorted((root / "kernels").rglob("*.cu"))
    paths += [Path(__file__),
              Path(__file__).with_name("hopper_cuda_config.py"),
              Path(__file__).with_name("native_compile.py"),
              Path(__file__).with_name("native_history.py"),
              Path(__file__).with_name("fused_config.py")]
    for path in paths:
        if "notes" not in path.parts:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def policy_identity():
    """Search policy invalidates completed build units, never compatible timings."""
    digest = hashlib.sha256()
    for name in ("hopper_cuda_config.py", "fused_config.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def candidates_for(op, bucket):
    """CPU-readable declared grid for one exact native workload."""
    tensors, extra = ast.literal_eval(bucket)
    if op == "trimul_fwd_sm90_cuda":
        from miniworld_engine.autotune.fused_config import trimul_candidates
        return trimul_candidates(tensors[0][0][-1], tensors[1][0][0], tensors[0][0][1], extra[0])
    if op in ("transition_fwd_residual_sm90_cuda", "transition_bwd_residual_sm90_cuda"):
        from miniworld_engine.autotune.fused_config import transition_candidates
        return transition_candidates(extra[-1], backward="bwd" in op)
    from miniworld_engine.autotune import hopper_cuda_config as cuda
    width = tensors[0][0][-1]
    if op.startswith("layernorm_"):
        kind = "fwd" if op == "layernorm_fwd_cuda" else "bwd"
        return cuda.layernorm_candidates(kind, width, 4 if "float32" in tensors[0][2] else 2)
    kind = {"transition_fwd_b2b_sm90_cuda": "b2b", "transition_bwd_gate_sm90_cuda": "gatebwd",
            "transition_expand_gate_sm90_cuda": "expand_gate"}[op]
    return cuda.candidates(kind, width)


def pending_candidates(op, data):
    """Count missing per-entry coverage without treating a file grid as evidence."""
    from miniworld_engine.autotune.cache import _sig_from_dict, entry_space
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


class _CacheConfigView(Sequence):
    """Re-iterable canonical cache configs without eager whole-grid conversion."""
    def __init__(self, candidates):
        self._candidates = candidates

    def __len__(self):
        return len(self._candidates)

    def cache_signatures(self):
        prepared = getattr(self._candidates, "cache_signatures", None)
        if prepared is not None:
            return prepared()
        from miniworld_engine.autotune.cache import _sig_from_dict
        return frozenset(_sig_from_dict(c) for c in self)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [as_cfg_dict({"kwargs": dict(c)}) for c in self._candidates[index]]
        return as_cfg_dict({"kwargs": dict(self._candidates[index])})


def choose_config(op, candidates, *, dtype, bucket, device_index=None, run=None, validate=None):
    """Resolve one config; in a build, measure every candidate once per exact workload.

    The first candidate is the documented cache-miss default. Invalid candidates are
    reported; an entirely failed round raises rather than claiming build completion.
    ``run`` must overwrite its outputs and must not mutate any input.
    """
    import torch

    dtype = dtype.removeprefix("torch.")
    if not candidates:
        raise ValueError(f"{op}: no configuration supports this workload")
    if torch.compiler.is_compiling():
        return dict(candidates[0])
    identity = source_identity()
    if not settings.current().run_autotune or run is None:
        best = select_config(op, dtype=dtype, bucket=bucket, candidates=_CacheConfigView(candidates),
                             device_index=device_index, op_id=identity)
        return dict(best["kwargs"] if best else candidates[0])

    # A build needs all candidates; runtime misses return before materializing
    # them. No selected-config memoization: publication/invalidation stays live.
    grid = list(_CacheConfigView(candidates))

    from triton.testing import do_bench

    from miniworld_engine.autotune import cache, capture, native_history
    from miniworld_engine.autotune.native_compile import precompile

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
                            output = run(c)
                            if validate is not None:
                                validate(c, output)
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
                assert isinstance(record, dict), "candidate must produce a result"
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
