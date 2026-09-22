"""Search spaces for the hand-CUDA Hopper transition kernels.

BN and KT select supported WGMMA descriptors; stages and consumer warpgroups
are performance choices. B2B's two consumers and DN=128 are instruction/layout
constraints of its register-resident squeeze, not search axes.
"""
from itertools import product


def candidates(kind, width):
    if kind not in ("b2b", "expand_gate", "gatebwd"):
        raise ValueError(f"unknown Hopper CUDA kernel {kind}")
    if width not in (128, 256, 512) or (kind == "b2b" and width == 512):
        raise ValueError(f"unsupported Hopper {kind} width {width}")
    if kind == "b2b":
        default = {"bn": 128 if width == 128 else 64,
                       "stages": 2 if width == 128 else 1, "warpgroups": 2, "kt": width}
        grid = [{"bn": bn, "stages": st, "warpgroups": 2, "kt": width}
                for bn, st in product((64, 128), (1, 2, 3))]
    else:
        default = {"bn": 128 if kind == "expand_gate" and width < 512 else 64,
                       "stages": 3 if width == 128 else 2,
                       "warpgroups": 2, "kt": 64 if width == 512 else 128}
        grid = [{"bn": bn, "stages": st, "warpgroups": wg, "kt": kt}
                for bn, st, wg, kt in product((64, 128), (1, 2, 3), (1, 2), (64, 128))]
    return [{**c, "min_blocks": mb} for c in [default] + [c for c in grid if c != default]
            for mb in (1, 2) if _fits_layout(kind, width, c)]


def _fits_layout(kind, width, config):
    """Host mirror of CUDA static_asserts, not a performance heuristic."""
    rows, stages, bn, kt = 64 * config["warpgroups"], config["stages"], config["bn"], config["kt"]
    weight_elements = (3 * width * bn if kind == "b2b" else 2 * kt * bn)
    # Two 64-bit TMA barriers per stage, plus per-row FP32 mean/rstd.
    smem = 2 * (rows * width + stages * weight_elements) + 8 * rows + 16 * stages
    if smem > 227 * 1024:
        return False
    if kind == "b2b":
        # B2B's m64n128 squeeze stores both consumers through ONE Wa stage.
        return rows * 128 <= width * bn
    # Expand and gate backward overlay warpgroup output tiles on Wa staging.
    return rows <= stages * kt


def defines(kind, width, config):
    if config not in candidates(kind, width):
        raise ValueError(f"invalid Hopper {kind} config: {config}")
    return [f"-DMW_TRANSITION_{k.upper()}={v}" for k, v in
            {"width": width, **config}.items()]


def layernorm_candidates(kind, width, element_bytes):
    if kind == "fwd":
        return [{"block": b} for b in (128, 32, 64, 256, 512, 1024)]
    if kind == "compile":
        return [{"warps": 4, "min_blocks": 2}]
    if width > 1024:
        return []
    return [{"warps": w, "min_blocks": mb, "waves": waves,
             "reduce_block": rb, "tx_bytes": tx}
            for w, mb, waves, rb, tx in product((4, 8), (2, 1), (4, 8), (256, 128), (16, 8))
            if width % (32 * (tx // element_bytes)) == 0]
