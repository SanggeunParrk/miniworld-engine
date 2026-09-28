"""Build policy, independent of kernels, torch and device initialization.

Production builds follow dispatch instead of forcing fallback implementations.
Token training uses L384/L768; atom counts and inference retain their registry
ladders. Config breadth and dispatch coverage are independent choices.
"""
import os

TRAIN_TOKEN_LENGTHS = (384, 768)
TOKEN_STREAMS = frozenset({"token_pair", "token_single", "msa_token"})


def scope():
    value = os.environ.get("MINIWORLD_BUILD_SCOPE", "production")
    if value not in ("production", "all"):
        raise ValueError("MINIWORLD_BUILD_SCOPE must be production or all")
    return value


def mode():
    value = os.environ.get("MINIWORLD_BUILD_MODE", "both")
    if value not in ("both", "train", "eval"):
        raise ValueError("MINIWORLD_BUILD_MODE must be both, train or eval")
    return value


def identity():
    return (scope(), mode(), TRAIN_TOKEN_LENGTHS)


def allows(stream, length, run_mode, impl, option=None, impls=()):
    if mode() != "both" and run_mode != mode():
        return False
    if run_mode == "train" and stream in TOKEN_STREAMS and length not in TRAIN_TOKEN_LENGTHS:
        return False
    if scope() == "all":
        return True
    # miniworld is the module's automatic backend selector. Do not separately
    # drive Triton/CuTe alternatives when that selector already owns the call.
    if "miniworld" in impls and impl != "miniworld":
        return False
    # Dropout changes the workload, not the implementation being compared.
    return option is None or option[0] == "p_drop"


def filter_op_units(units):
    """Apply mode/crop policy to diagnostic drivers using declared stream levels."""
    import csv
    from pathlib import Path

    with (Path(__file__).resolve().parents[1] / "kernels" / "registry" / "registry.csv").open() as fh:
        levels = {row["kernel"]: row["level"] for row in csv.DictReader(fh)}
    selected = []
    run_mode = mode()
    for unit in units:
        training = any(tag in unit.op.split("_") for tag in ("bwd", "backward", "train"))
        inference = "inference" in unit.op or unit.op == "trimul_fwd_sm90_cuda"
        if (run_mode == "eval" and training) or (run_mode == "train" and inference):
            continue
        token = levels.get(unit.op) == "token" or (
            levels.get(unit.op) == "both" and unit.side in ("pair", "token", "msa"))
        if token and (run_mode == "train" or training):
            if unit.length not in TRAIN_TOKEN_LENGTHS:
                continue
        selected.append(unit)
    return selected
