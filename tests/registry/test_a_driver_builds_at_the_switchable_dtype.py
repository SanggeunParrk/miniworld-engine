"""A driver must build its ACTIVATIONS at the precision the build asked for, not at a fixed one.

`MINIWORLD_DRIVER_DTYPE` switches `drivers.BF16`, and every `dtype=BF16` default argument closes
over it. A driver that names `FP32` or `torch.float32` for an activation instead is pinned: the
kernel can then only ever be built at that precision, whatever `registry.csv` declares.

That is not hypothetical. Every conditioned_transition driver was pinned to fp32, and the `dtypes`
column said fp32 to match. Both halves were true and together they meant the model's own precision
-- the model runs bf16 -- was never tuned: all eight kernels have float32 entries and nothing else in
the shipped cache on both cards, so in a bf16 run every lookup is a miss and falls back to a
heuristic subset of the grid.

fp32 is still the right dtype for some operands. LayerNorm statistics are fp32 whatever the
activation is; a sigma is fp32. Those are not the failure -- pinning the thing the kernel computes
ON is. So every fixed-precision site in a driver is listed here with which of the two it is, and a
new one has to be classified rather than added silently.
"""
from __future__ import annotations

import re

from paths import ROOT

DRIVERS = ROOT / "src" / "miniworld_engine" / "kernels" / "drivers"

#: `file:symbol` -> why this site names a fixed precision. An ACTIVATION here is a bug unless the
#: reason says the kernel genuinely runs at one precision and the registry declares the same.
ALLOWED = {
    "adaln.py:FP32": "a LayerNorm statistic (mean/rstd) is fp32 whatever the activation is",
    "triangle_multiplication.py:torch.float32": "sigma is fp32 by definition, not an activation",
    "rope.py:torch.float32": "cos/sin are fp32 angle tensors (rotation precision), not activations",
    "fused_ln_mask.py:norm_affine":
        "the masked LN's gamma/beta, fp32 in production via `primitives._Fp32ParamsMixin` -- "
        "`dev audit --replay` showed this op keying `bfloat16+float32` against a bf16-only cache",
    "layernorm_linear.py:norm_affine":
        "same: `layernorm_linear_triton_fwd` is handed a `primitives.LayerNorm` parameter, so its "
        "production key is `bfloat16+float32` and the bf16 driver recorded a bucket nothing asks for",
    "trimul_inproj.py:norm_affine":
        "the LN_out affine (gamma/beta), which `primitives.LayerNorm`'s `_Fp32ParamsMixin` pins to "
        "fp32 through the trunk's bulk .to(bfloat16) -- so it is fp32 whatever the ACTIVATION is, "
        "the same class as adaln.py's statistic. Driving it at BF16 was not a pinned activation "
        "but the mirror-image bug: `dtype_of_args` keys on the SET of float operand dtypes, so "
        "production launched `bfloat16+float32` while the driver recorded `bfloat16`, and every "
        "trimul_outproj_layernorm_gemm_gate lookup missed on the dtype axis alone",
}

#: ``norm_affine`` is swept alongside the literals on purpose. It is `drivers.norm_affine`, which
#: builds a norm gamma/beta at fp32 because that is where `primitives._Fp32ParamsMixin` pins it --
#: a real fixed-precision choice, just a named one. Naming it must not buy an exemption from
#: declaring it, or the guard would go quiet exactly as sites move to the helper.
#: The import line and the docstrings that DISCUSS the names are not uses of them.
_SKIP = re.compile(r"^\s*(#|from |import )|^\s*$")


def _sites() -> dict[str, list[int]]:
    """`file:name` -> the line numbers naming a fixed precision, docstrings and imports aside."""
    out: dict[str, list[int]] = {}
    for path in sorted(DRIVERS.glob("*.py")):
        if path.name == "__init__.py":
            continue                      # where BF16/FP32 are DEFINED
        in_doc = False
        for n, line in enumerate(path.read_text().splitlines(), start=1):
            ticks = line.count('"""')
            if ticks:
                in_doc = in_doc != (ticks % 2 == 1)
                continue
            if in_doc or _SKIP.match(line):
                continue
            for name in ("torch.float32", "FP32", "norm_affine"):
                if name == "FP32" and "torch.float32" in line:
                    continue              # counted once, under its own name
                if re.search(rf"\b{re.escape(name)}\b", line):
                    out.setdefault(f"{path.name}:{name}", []).append(n)
    return out


def test_there_are_drivers_to_check() -> None:
    """Guard the guard: a moved package would make the sweep below find nothing."""
    assert list(DRIVERS.glob("*.py")), f"no driver modules under {DRIVERS}"


def test_every_fixed_precision_site_is_classified() -> None:
    """A new one is a decision -- an fp32 statistic, or a kernel pinned away from the model's own
    precision. Both are fine; being neither, silently, is what cost eight kernels their cache."""
    undeclared = sorted(k for k in _sites() if k not in ALLOWED)
    assert not undeclared, (
        f"driver sites naming a fixed precision with no entry in ALLOWED: {undeclared}. If it is a "
        f"statistic (fp32 whatever the activation is), say so. If it is an ACTIVATION, it pins the "
        f"kernel to one precision -- check that registry.csv's `dtypes` says the same, and that "
        f"the model does not run it at the other one.")


def test_no_entry_outlives_its_site() -> None:
    """An allowance for a line that no longer exists is a place to put the next one."""
    stale = sorted(k for k in ALLOWED if k not in _sites())
    assert not stale, f"ALLOWED names sites the drivers no longer have: {stale}"


def test_every_reason_says_something() -> None:
    for key, why in ALLOWED.items():
        assert len(why) > 40, f"{key}: {why!r} is not a reason"
