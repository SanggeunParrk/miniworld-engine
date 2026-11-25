"""E6: peak GPU memory of one step per case and backend.

For every case of the config and every backend the case's class has: the module and its inputs are built as the speed measurer builds them
(foldjet_engine.measure.config / speed.make_step), the step (eval: the forward under no_grad; train: forward + backward, the production
step of speed.make_step) is run once warm (so lazily built weights, kernel tables and calibration are in place), the cache is emptied, then
``torch.cuda.reset_peak_memory_stats()`` and one step, and ``torch.cuda.max_memory_allocated()`` is the peak. The step runs eagerly on the
default stream (no CUDA graph: a graph's private pool would hide the allocator's peak).

Recorded in GiB: ``peak_gib`` = the peak of that step, absolute (inputs, parameters, gradients and the step's temporaries);
``input_gib`` = the input tensors, ``param_gib`` = the parameters, ``base_gib`` = memory in use just before the step (inputs + parameters +
the upstream gradients of a train step + the other backends' leftovers: none, each run frees its copy first), so ``peak_gib - base_gib`` is
what the step itself adds. A case that does not fit is ``skipped`` (out of memory), a backend that refuses it ``unsupported``.
"""

from __future__ import annotations

import copy
import gc
import json
import os
import sys
import traceback

import torch

GIB = 1024**3


def _bytes(tree) -> int:
    from torch.utils import _pytree

    seen, total = set(), 0
    for t in _pytree.tree_leaves(tree):
        if isinstance(t, torch.Tensor) and t.untyped_storage().data_ptr() not in seen:
            seen.add(t.untyped_storage().data_ptr())
            total += t.untyped_storage().nbytes()
    return total


def measure_backend(base, case, backend: str, dev) -> dict:
    from foldjet_engine.measure import config
    from foldjet_engine.measure.speed import make_step
    from foldjet_engine.modules.primitives import unpadded_masks

    module = copy.deepcopy(base)
    with unpadded_masks():
        inputs = config.inputs(case, module, device=dev)
    torch.cuda.synchronize(dev)
    gc.collect()
    torch.cuda.empty_cache()
    in_bytes = _bytes(inputs)
    par_bytes = _bytes(list(module.parameters()))
    step = make_step(
        module,
        inputs,
        backend=backend,
        train=case.mode == "train",
        autocast=config.autocast(case),
        seed=config.seed(case),
        compile_mode=None,
        keep_expanded=config.keeps_expanded(case),
    )
    step()  # warm: lazily built state (weights packs, tables, calibration) is not the step's memory
    torch.cuda.synchronize(dev)
    gc.collect()
    torch.cuda.empty_cache()
    base_bytes = torch.cuda.memory_allocated(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    step()
    torch.cuda.synchronize(dev)
    peak = torch.cuda.max_memory_allocated(dev)
    return {
        "peak_gib": peak / GIB,
        "base_gib": base_bytes / GIB,
        "input_gib": in_bytes / GIB,
        "param_gib": par_bytes / GIB,
    }


def run(cfg, out_path: str) -> list[dict]:
    from foldjet_engine.measure import config
    from foldjet_engine.modules.primitives import BackendUnsupported

    dev = torch.device(cfg.device)
    meta = config.run_meta(cfg, "memory", "1")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    records = []
    print(f"[memory] records -> {out_path}", flush=True)
    with open(out_path, "a") as sink:
        for case in config.cases(cfg):
            base = config.build(case, device=dev)
            for backend in config.backends_of(cfg, case):
                rec = {**config.describe(case, cfg), "meta": meta, "backend": backend, "timing": "graph"}
                try:
                    if backend not in type(base).backends:
                        raise BackendUnsupported(f"{type(base).__name__} has no backend {backend!r}")
                    rec.update(status="ok", **measure_backend(base, case, backend, dev))
                except BackendUnsupported as exc:
                    rec.update(status="unsupported", reason=str(exc))
                except Exception as exc:  # recorded, never re-routed
                    oom = isinstance(exc, torch.cuda.OutOfMemoryError)
                    rec.update(status="skipped" if oom else "failed", reason=f"{type(exc).__name__}: {str(exc).strip()[:500]}")
                    if os.environ.get("SPEED_TRACEBACK"):
                        traceback.print_exc()
                finally:
                    gc.collect()
                    torch.cuda.synchronize(dev)
                    torch.cuda.empty_cache()
                records.append(rec)
                sink.write(json.dumps(rec, default=str) + "\n")
                sink.flush()
                tail = (
                    f"{rec['peak_gib']:.3f} GiB (inputs {rec['input_gib']:.3f}, params {rec['param_gib']:.4f})"
                    if rec["status"] == "ok"
                    else rec["status"]
                )
                print(f"{rec['label']} | {backend:<9} | {tail}", flush=True)
            del base
            torch.cuda.empty_cache()
    return records


if __name__ == "__main__":
    sys.exit("use paper_common/run_measure.py --kind memory")
