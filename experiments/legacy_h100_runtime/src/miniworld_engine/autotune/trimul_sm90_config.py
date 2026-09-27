"""SM90 implementations share the corresponding Triton CSV search domains.

Candidate rejection is explicit: hardware-infeasible is not a measured loss.
No tile axis is renamed, silently rounded, or substituted by a fixed launch.
"""
from __future__ import annotations

import ast
import csv
import itertools
from pathlib import Path

TRITON_OPS = {
    "layernorm_bwd_split_sm90_cute": "layernorm_bwd_split_triton",
    "trimul_inproj_gemm_gate_mmajor_sm90_cute": "trimul_gemm_gate_mmajor_triton",
    "trimul_output_f567_train_sm90_cute": "trimul_output_f567_train_triton",
    "trimul_input_dual_bwd_sm90_cute": "trimul_input_dual_bwd_triton",
}


def declared_configs(op, directory=None):
    """Expand exactly the declared domain, including hardware-invalid candidates."""
    directory = Path(directory) if directory else Path(__file__).with_name("configs") / "grid"
    path = directory / (TRITON_OPS[op] + ".csv")
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty config space: {path}")
    if "axis" not in rows[0]:
        return [{k: int(v) for k, v in row.items() if v.strip()} for row in rows]
    axes = [(row["axis"], [int(v) for v in row["values"].split()]) for row in rows]
    if any(axis == "slice" for axis, _ in axes):
        raise ValueError("Use an unsliced domain for SM90 parity search")
    names = [axis for axis, _ in axes]
    return [dict(zip(names, values)) for values in itertools.product(*(values for _, values in axes))]


def partition_configs(op, feasibility, directory=None):
    """Return executable configs and a reason for each excluded declaration.

    ``feasibility(config)`` returns None on success and a diagnostic string otherwise.
    The caller closes over shape/layout constraints; it must not benchmark here.
    """
    kept, rejected = [], []
    for config in declared_configs(op, directory):
        reason = feasibility(config)
        if reason is None:
            kept.append(config)
        else:
            rejected.append({"config": config, "reason": str(reason), "status": "infeasible"})
    return kept, rejected


def partition_for_bucket(op, bucket):
    """Reproduce runtime feasibility for native cache coverage/build inspection."""
    tensors, extra = ast.literal_eval(bucket)
    if op == "trimul_inproj_gemm_gate_mmajor_sm90_cute":
        from miniworld_engine.kernels.trimul_inproj.cute.parity_front import front_config_rejection
        m, k = tensors[0][0]
        h2 = tensors[1][0][1] // 4
        save_preact = bool(extra[0]) if extra else True
        feasible = lambda c: front_config_rejection(
            c, m=m, k=k, h2=h2, save_preact=save_preact
        )
    elif op == "trimul_output_f567_train_sm90_cute":
        from miniworld_engine.kernels.trimul_inproj.cute.parity_f567 import feasibility
        kp, kg = tensors[0][0][1], tensors[1][0][1]
        feasible = lambda c: feasibility(c, smem_limit=extra[1], kp=kp, kg=kg)
    elif op == "trimul_input_dual_bwd_sm90_cute":
        from miniworld_engine.kernels.trimul_inproj.cute.parity_dual_bwd import feasibility
        feasible = lambda c: feasibility(c, smem_limit=extra[1])
    elif op == "layernorm_bwd_split_sm90_cute":
        from miniworld_engine.kernels.layernorm.cute.tma_backward import config_rejection
        shape, strides, dtype = tensors[0]
        feasible = lambda c: config_rejection(
            c, n=shape[1], itemsize=4 if "float32" in dtype else 2,
            m_major=strides[0] == 1, smem_limit=extra[0]
        )
    else:
        raise ValueError(f"Unknown SM90 TriMul op: {op}")
    return partition_configs(op, feasible)


def resolve(op, tensors, *, extra=(), feasibility, run, directory=None):
    """Use the engine's native cache/history with the same Triton config axes.

    Runtime cache misses use the first feasible CSV config; explicit native build
    mode measures all feasible candidates. Unsupported sets fail visibly.
    """
    from miniworld_engine.autotune.native import choose_config, tensor_key

    configs, rejected = partition_configs(op, feasibility, directory)
    if not configs:
        reasons = sorted({r["reason"] for r in rejected})
        raise ValueError(f"{op}: no feasible configuration: {reasons}")
    first = next(t for t in tensors if t is not None)
    return choose_config(op, configs, dtype=str(first.dtype),
                         bucket=tensor_key(*tensors, extra=extra),
                         device_index=first.device.index, run=run)
