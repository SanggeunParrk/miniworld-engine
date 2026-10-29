"""Tunable schedules of packaged fused CUDA kernels, with measured defaults first.

Candidate declarations do not certify correctness or performance on a new GPU.
The build validates outputs against the retained default before timing a candidate.
"""

from itertools import product
from typing import Any


def transition_candidates(sms, *, backward=False, n=4):
    """Schedules of the D128 fused Transition. The backward gives the weight role ``2 n`` hidden slices times ``dw_repl``
    replicas; the best split moves with the slice count (8 at n = 4, 15 at n = 2)."""
    if sms <= 8:
        raise ValueError("fused Transition requires more than 8 SMs")
    slices = 2 * n
    replica = min(
        15 if n == 2 else 8, (sms - 1) // slices
    )  # measured on H100 (n = 2: 14-15 best, 16 is 4 % slower)
    default = {"ctas": sms, "dw_repl": replica}
    ctas = (sms, max(16, sms // 2), sms * 2)
    replicas = (
        ((replica, 2, 4, 8, 12) + ((16, 24) if n < 4 else ()))
        if backward
        else (replica,)
    )
    grid = [
        {"ctas": c, "dw_repl": r} for c, r in product(ctas, replicas) if c > slices * r
    ]
    return [default] + [c for c in grid if c != default]


def trimul_candidates(cz, ch, length, direction):
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_infer_kernel import (
        TILE_TABLE,
        lookup,
    )

    # One heterogeneous dict per arch/width (tile tuples, variant lists, mode tags, per-N overrides).
    table: dict[str, Any] | None = TILE_TABLE.get(("sm_90a", cz, ch, "b"))
    if table is None:
        return []
    default = lookup(
        "sm_90a",
        cz,
        ch,
        "b",
        length,
        "incoming" if direction == 2 else "outgoing",
        True,
    )

    def pack(k1, k3):
        return {
            **{f"k1_{i}": v for i, v in enumerate(k1)},
            **{f"k3_{i}": v for i, v in enumerate(k3)},
        }

    first = pack(default["k1"], default["k3"])
    grid = [
        pack(a, b)
        for a, b in product(
            table.get("k1_variants", [table["k1"]]),
            table.get("k3_variants", [table["k3"]]),
        )
    ]
    return [first] + [c for c in grid if c != first]


def unpack_trimul(config):
    # Some K3 variants carry a fifth layout/mode tag. Preserve the entire tuple;
    # truncating it would silently change the previously measured default.
    return {
        part: tuple(config[key] for key in sorted(config) if key.startswith(part + "_"))
        for part in ("k1", "k3")
    }


def validator(run, default, *, output_only=False):
    """Lazily compare complete outputs/saved tensors/gradients before publication."""
    reference = []

    def check(config, output):
        import torch

        if not reference:
            reference.append(run(default))
        actual, expected = (
            (output[0], reference[0][0]) if output_only else (output, reference[0])
        )
        _assert_matches(actual, expected)

    return check


def _assert_matches(actual, expected):
    """5e-3 absolute and relative, widened for a 16-bit tensor to one rounding step of its dtype at the tensor's own
    scale: a weight gradient summed over 147 456 rows reaches ~256, where one bf16 step is 2.0, and a schedule that
    reorders the sum (``dw_repl``) lands one step away from the default on some elements."""
    import torch

    if isinstance(actual, (tuple, list)):
        assert isinstance(expected, (tuple, list)) and len(actual) == len(expected), "output structure differs"
        for a, e in zip(actual, expected):
            _assert_matches(a, e)
        return
    if not isinstance(expected, torch.Tensor) or expected.dtype not in (torch.bfloat16, torch.float16) or not expected.numel():
        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
        return
    step = torch.finfo(expected.dtype).eps
    scale = float(expected.detach().abs().max().float())
    torch.testing.assert_close(actual, expected, rtol=max(5e-3, step), atol=max(5e-3, step * scale))
