"""Per-target benchmark precision policy, grounded in ``kernels/registry.csv``.

The rule (user contract): bench every target at the dtype(s) it actually supports.

  * ``dtypes=bf16``       -> bench bf16 only            -> precisions = [BF16]
  * ``dtypes=bf16|fp32``  -> bench fp32 AND bf16        -> precisions = [FP32, BF16]
  * ``dtypes=fp32``       -> bench fp32 only            -> precisions = [FP32]

``FP32`` is Lightning's ``32`` (:data:`FP32_PRECISION`); ``BF16`` is the string ``"bf16"`` --
"full bf16" (bf16 weights + bf16 activations, no autocast), which is what the harness already
runs since its fabric is a no-op shim.

Two namespaces resolve differently:

* **kernel** targets are named after a registry *family* (``bench.py`` line ~2599), so their
  declared dtypes come straight from the family's rows. The three gemm building-block probes
  (``dual_gemm_epilogue`` / ``gemm_epilogue`` / ``gemm_gate`` and their ``_bwd``) are not family
  names -- every gemm kernel in the registry is bf16 -- so they are bf16-only.

* **module** targets compose several kernel families, so a family-union would be wrong: a
  triangle-multiplication module pulls in ``layernorm`` (which *is* bf16|fp32) yet its own fused
  tm1/tm2/trimul_inproj GEMMs are bf16-only, so the module has no end-to-end fp32 miniworld path
  (an fp32 run falls to torch). What decides a module's precisions is whether *miniworld itself*
  has an fp32 kernel end to end -- true for the diffusion blocks (adaln / conditioned_transition /
  augmented_attention, all registry bf16|fp32), false for the trunk/fused blocks. So modules carry
  an explicit map. It cannot be derived from the registry -- ``transition``'s family declares
  bf16|fp32 for its cuda backward while the module's fused forward is bf16-only -- so the map states
  where it DELIBERATELY differs (:data:`MODULE_REGISTRY_EXCEPTIONS`), and a test pins that list: a
  new divergence has to be declared, it cannot appear by accident.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

FP32_PRECISION = 32
BF16 = "bf16"

_REGISTRY = (
    Path(__file__).resolve().parents[2]
    / "src" / "miniworld_engine" / "kernels" / "registry.csv"
)

# kernel bench target -> registry family whose dtypes it inherits. A target absent here (the gemm
# building-block probes) is bf16-only. ``_bwd`` targets share their forward family's dtypes.
KERNEL_TARGET_FAMILY: dict[str, str] = {
    "adaln": "adaln",
    "adaln_bwd": "adaln",
    "augmented_attention": "augmented_attention",
    "bias_only_attention": "bias_only_attention",
    "conditioned_transition_tail": "conditioned_transition",
    "fused_ln_mask": "fused_ln_mask",
    "layernorm": "layernorm",
    "layernorm_bwd": "layernorm",
    "transition_b2b": "transition",
    "transition_b2b_bwd": "transition",
    "triangle_attention": "triangle_attention",
}

#: Module targets whose precision set DELIBERATELY differs from a naive registry-family union, and
#: why. A module composes several families, so the union over-claims: what decides the module's
#: precisions is whether *miniworld itself* has an fp32 kernel END TO END.
MODULE_REGISTRY_EXCEPTIONS: dict[str, str] = {
    "transition": "family declares bf16|fp32 (the cuda backward), but the fused forward is "
                  "bf16-only -- an fp32 run falls to torch, so the module has no fp32 path",
    "triangle_multiplication": "pulls in layernorm (bf16|fp32) but its own tm1/tm2/trimul_inproj "
                               "GEMMs are bf16-only",
    "triangle_multiplication_bidirectional": "same as triangle_multiplication",
    "triangle_attention": "pulls in layernorm (bf16|fp32); its own attention kernels are bf16-only",
    "swa_atom_attention": "flash-backed; bf16-only, and it has no registry family of its own",
}

# module bench target -> does miniworld have an end-to-end fp32 kernel? The diffusion blocks do
# (their kernels are registry bf16|fp32); the trunk/fused blocks and flash-based swa do not.
MODULE_SUPPORTS_FP32: dict[str, bool] = {
    "triangle_multiplication": False,
    "triangle_multiplication_bidirectional": False,
    "triangle_attention": False,
    "transition": False,
    "swa_atom_attention": False,
    "conditioned_transition": True,
    "adaptive_layernorm": True,
    "augmented_attention_token": True,
    "augmented_attention_atom": True,
}


@lru_cache(maxsize=1)
def _family_dtypes() -> dict[str, set[str]]:
    """family -> the union of dtype tokens (``bf16`` / ``fp32``) its registry rows declare."""
    out: dict[str, set[str]] = {}
    with _REGISTRY.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            toks = {t.strip() for t in (row.get("dtypes") or "bf16").split("|") if t.strip()}
            out.setdefault(row["family"], set()).update(toks)
    return out


def _family_supports_fp32(family: str) -> bool:
    return "fp32" in _family_dtypes().get(family, {"bf16"})


def declared_precisions(level: str, target: str) -> list[int | str]:
    """The precisions to bench ``(level, target)`` at, per the registry dtype policy.

    Returns a subset of ``[FP32_PRECISION, BF16]`` in that order. bf16 is always present unless a
    target is fp32-only in the registry (the lone ``augmented_attention`` fp32 kernel is still
    reached through the bf16|fp32 family, so in practice every target benches bf16).
    """
    if level == "kernel":
        fam = KERNEL_TARGET_FAMILY.get(target)
        both = fam is not None and _family_supports_fp32(fam)
    elif level == "module":
        if target not in MODULE_SUPPORTS_FP32:
            raise KeyError(f"module target {target!r} has no precision policy entry")
        both = MODULE_SUPPORTS_FP32[target]
    else:
        raise ValueError(f"unknown level {level!r}")
    return [FP32_PRECISION, BF16] if both else [BF16]
