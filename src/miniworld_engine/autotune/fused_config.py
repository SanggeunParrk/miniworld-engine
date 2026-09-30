"""Tunable schedules of packaged fused CUDA kernels, with measured defaults first.

Candidate declarations do not certify correctness or performance on a new GPU.
The build validates outputs against the retained default before timing a candidate.
"""
from itertools import product


def transition_candidates(sms, *, backward=False):
    if sms <= 8:
        raise ValueError("fused Transition requires more than 8 SMs")
    replica = min(8, (sms - 1) // 8)
    default = {"ctas": sms, "dw_repl": replica}
    ctas = (sms, max(16, sms // 2), sms * 2)
    replicas = (replica, 2, 4, 8, 12) if backward else (replica,)
    grid = [{"ctas": c, "dw_repl": r} for c, r in product(ctas, replicas) if c > 8 * r]
    return [default] + [c for c in grid if c != default]


def trimul_candidates(cz, ch, length, direction):
    from miniworld_engine.kernels.trimul_inproj.cuda._h100_infer_kernel import (
        TILE_TABLE,
        lookup,
    )
    table = TILE_TABLE.get(("sm_90a", cz, ch, "b"))
    if table is None:
        return []
    default = lookup("sm_90a", cz, ch, "b", length,
                     "incoming" if direction == 2 else "outgoing", True)
    def pack(k1, k3):
        return {**{f"k1_{i}": v for i, v in enumerate(k1)},
                **{f"k3_{i}": v for i, v in enumerate(k3)}}
    first = pack(default["k1"], default["k3"])
    grid = [pack(a, b) for a, b in product(table.get("k1_variants", [table["k1"]]),
                                          table.get("k3_variants", [table["k3"]]))]
    return [first] + [c for c in grid if c != first]


def unpack_trimul(config):
    # Some K3 variants carry a fifth layout/mode tag. Preserve the entire tuple;
    # truncating it would silently change the previously measured default.
    return {part: tuple(config[key] for key in sorted(config)
                        if key.startswith(part + "_")) for part in ("k1", "k3")}


def validator(run, default, *, output_only=False):
    """Lazily compare complete outputs/saved tensors/gradients before publication."""
    reference = []
    def check(config, output):
        import torch
        if not reference:
            reference.append(run(default))
        actual, expected = (output[0], reference[0][0]) if output_only else (output, reference[0])
        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
    return check
