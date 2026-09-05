"""The registry-driven bench precision policy (benchmarks/runners/bench_policy.py) must stay
consistent with kernels/registry.csv and cover every bench target."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

import sys
_RUNNERS = Path(__file__).resolve().parents[2] / "benchmarks" / "runners"
sys.path.insert(0, str(_RUNNERS))
import bench_policy as bp  # noqa: E402

_REG = Path(__file__).resolve().parents[2] / "src" / "miniworld_engine" / "kernels" / "registry.csv"
_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"


def _families() -> set[str]:
    with _REG.open(encoding="utf-8") as h:
        return {r["family"] for r in csv.DictReader(h)}


def _targets(level: str) -> set[str]:
    out = set()
    for y in (_BENCH / f"{level}s").glob("*/configs/bench.yaml"):
        for line in y.read_text().splitlines():
            if line.startswith("target:"):
                out.add(line.split(":", 1)[1].strip())
    return out


def test_kernel_target_families_exist_in_registry():
    fams = _families()
    for tgt, fam in bp.KERNEL_TARGET_FAMILY.items():
        assert fam in fams, f"{tgt} -> unknown family {fam}"


def test_kernel_family_fp32_matches_registry():
    for tgt, fam in bp.KERNEL_TARGET_FAMILY.items():
        both = bp.FP32_PRECISION in bp.declared_precisions("kernel", tgt)
        assert both == bp._family_supports_fp32(fam), f"{tgt}: policy/registry fp32 mismatch"


def test_every_module_target_has_a_policy():
    for tgt in _targets("module"):
        # must not raise
        prec = bp.declared_precisions("module", tgt)
        assert bp.BF16 in prec, f"{tgt}: bf16 must always be benched"


def test_every_kernel_target_resolves():
    for tgt in _targets("kernel"):
        prec = bp.declared_precisions("kernel", tgt)
        assert bp.BF16 in prec


def test_declared_precisions_order_and_values():
    assert bp.declared_precisions("module", "triangle_multiplication") == [bp.BF16]
    assert bp.declared_precisions("module", "conditioned_transition") == [bp.FP32_PRECISION, bp.BF16]
    assert bp.declared_precisions("kernel", "gemm_gate") == [bp.BF16]
    assert bp.declared_precisions("kernel", "adaln") == [bp.FP32_PRECISION, bp.BF16]


def test_unknown_level_and_module_raise():
    with pytest.raises(ValueError):
        bp.declared_precisions("trunk", "x")
    with pytest.raises(KeyError):
        bp.declared_precisions("module", "does_not_exist")


def test_module_map_divergence_from_registry_is_declared():
    """A module's precision set may differ from a naive registry-family union -- a module composes
    several families, and the union over-claims. But every such divergence must be DECLARED in
    MODULE_REGISTRY_EXCEPTIONS with a reason, so a new one cannot appear by accident (the failure
    the old docstring claimed was covered and was not)."""
    fam_of = {
        "triangle_multiplication": "triangle_multiplication",
        "triangle_multiplication_bidirectional": "trimul_inproj",
        "triangle_attention": "triangle_attention",
        "transition": "transition",
        "conditioned_transition": "conditioned_transition",
        "adaptive_layernorm": "adaln",
        "augmented_attention_token": "augmented_attention",
        "augmented_attention_atom": "augmented_attention",
    }
    diverged = []
    for target, supports_fp32 in bp.MODULE_SUPPORTS_FP32.items():
        fam = fam_of.get(target)
        if fam is None:                       # no registry family at all (swa)
            if target not in bp.MODULE_REGISTRY_EXCEPTIONS:
                diverged.append(f"{target}: no registry family and no declared exception")
            continue
        if bp._family_supports_fp32(fam) != supports_fp32:
            diverged.append(f"{target}: registry family {fam!r} fp32="
                            f"{bp._family_supports_fp32(fam)} but map says {supports_fp32}")
    undeclared = [d for d in diverged if d.split(":")[0] not in bp.MODULE_REGISTRY_EXCEPTIONS]
    assert not undeclared, (
        "module precision map diverges from the registry without a declared reason -- add it to "
        "MODULE_REGISTRY_EXCEPTIONS (with why) or fix the map:\n  " + "\n  ".join(undeclared))


def test_every_declared_exception_is_a_real_target():
    for target in bp.MODULE_REGISTRY_EXCEPTIONS:
        assert target in bp.MODULE_SUPPORTS_FP32, f"stale exception for removed target {target!r}"
