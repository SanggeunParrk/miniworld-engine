"""No path builds, declares, or stores a LayerNorm at bf16.

bf16 layernorm destabilises training. That is an operator decision from experience, not something
this repository can measure, so the job here is to make it structural: a normalisation is fp32
everywhere, and reintroducing bf16 has to fail rather than pass quietly.

Three places could put it back, and each has its own check below:

NOT in the list: `registry.csv`'s `dtypes` cell. That column says which OPERAND MIX the build
drives, and a bf16 cell there does not mean the normalisation is computed in bf16 -- it means the
ACTIVATION is bf16, which is what production presents and what the kernel is supposed to be tuned
for. `cache.dtype_of_args` keys an entry on every distinct float operand dtype, so a real launch
records `bfloat16+float32`: bf16 activation, fp32 gamma/beta/stats. Declaring these rows fp32-only
made the drivers build an all-fp32 activation and record `float32`, a key production never looks
up -- and the only eight cache files in the repo whose config_space_hash still matches are
layernorm files keyed `bfloat16+float32`. It would have overwritten the one live cache there is
with misses.

  * the drivers -- a bf16 activation at a layernorm site would tune the kernel at a precision the
    registry does not declare, which is how the column and the driver drifted apart before.
  * the kernels -- `main.py` used to save x for the backward as `x_2d.to(torch.bfloat16)`, halving
    the activation. Every load in these files widens with `.to(tl.float32)` and mean/rstd are
    stored fp32, so that one cast was the narrowest precision in the whole normalisation, on the
    one value the backward cannot recover.

What this does NOT claim is that the surrounding stream is fp32. It is not: token and pair
activations are bf16 and stay bf16. The normalisation over them is what has to be wide.
"""
from __future__ import annotations

import ast
import re

from paths import ROOT, registry_rows

PKG = ROOT / "src" / "miniworld_engine"

KERNELS = PKG / "kernels"
#: Where a LayerNorm actually lives -- DERIVED, not listed. It was `("layernorm", "layernorm_linear")`
#: written out, which covered two of the seven families that normalise: `adaln` computes the
#: statistics for the DiT, `fused_ln_mask` is layernorm_fwd with a row scale, `transition` owns
#: `layernorm_bwd_foldstats`, `trimul_inproj` owns `trimul_outproj_layernorm_gemm_gate`, and
#: `mpnn_edge_tail` folds the normalisation into its third projection. Eleven kernels with the
#: `layernorm` role sat outside the check, including three added this month.
#:
#: `notes/` is scratch and reference.py is the fp64/bf16 oracle a checker compares against, so
#: neither is a path production takes.
def _dirs() -> tuple[str, ...]:
    """Every family owning a registry kernel whose name carries the `layernorm` role."""
    return tuple(sorted({r["file"].split("kernels/")[1].split("/")[0]
                         for r in registry_rows() if "layernorm" in r["kernel"]}))


DIRS = _dirs()
#: A NARROWING of a tensor, which is the thing that loses precision. A `dtype=torch.bfloat16`
#: annotation or default is a declaration about someone else's operand -- the cute GEMM's second
#: weight, say -- and says nothing about the normalisation, so it is not matched here.
NARROW = re.compile(r"\.to\(\s*torch\.bfloat16\s*\)|\.bfloat16\(\)")
#: Narrowings that are NOT a normalisation's input, with the reason each is not. The rule is about
#: "a saved or cast ACTIVATION the backward cannot recover"; once the family list stopped being two
#: hardcoded directories, the regex started reaching casts that are neither.
NOT_A_NORMALISATION = {
    "adaln/triton/inference.py:273":
        "GEMM operands: the conditioning matmul that PRODUCES scale and bias, cast the way every "
        "other GEMM in the package casts its operands. The normalisation it feeds is fp32.",
    "adaln/triton/training.py:63":
        "the same matmul on the training path, and it widens the result back with .float() on the "
        "same line.",
    "mpnn_edge_tail/triton/main.py:1108":
        "autocast's own boundary: Linear rounds a BIAS GRADIENT to bf16 before the fp32 parameter "
        "gradient, and this reproduces it deliberately. Not an activation, and not an input.",
    "mpnn_edge_tail/triton/main.py:1110":
        "the second of that pair, for the output bias.",
}

#: A driver naming a fixed precision. `BF16` is NOT one: it is the name
#: `MINIWORLD_DRIVER_DTYPE` switches, so a driver written against it builds fp32 for an fp32 row.
#: The pin is what would break the declaration.
PINNED = re.compile(r"torch\.bfloat16")


def _rows() -> list[dict]:
    return registry_rows()


def test_no_layernorm_driver_builds_a_bf16_activation() -> None:
    """The driver is what a unit actually runs, so it has to agree with the column.

    It looks for a PIN, not for the word bf16. `drivers.BF16` is the name `MINIWORLD_DRIVER_DTYPE`
    switches, so a driver written against it builds fp32 tensors for an fp32 row -- that is the
    mechanism working. `torch.bfloat16` written out is what would hold the kernel at bf16 whatever
    registry.csv says, which is how the column and the driver drifted apart before.
    """
    drivers = KERNELS / "drivers"
    bad = []
    for name in ("layernorm.py", "layernorm_linear.py"):
        f = drivers / name
        if not f.is_file():
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or '"""' in stripped:
                continue
            if PINNED.search(line):
                bad.append(f"{name}:{i}: {stripped}")
    assert not bad, ("a layernorm driver naming bf16 -- it would tune the kernel at a precision "
                     "registry.csv does not declare:\n  " + "\n  ".join(bad))


def test_no_layernorm_kernel_narrows_an_activation_to_bf16() -> None:
    """A saved or cast activation is the value the backward cannot recover. Loads widening to
    fp32 inside the kernel are fine and are what these files already do."""
    bad = []
    for d in DIRS:
        for f in sorted((KERNELS / d).rglob("*.py")):
            if "notes" in f.parts or f.name == "reference.py":
                continue
            for i, line in enumerate(f.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if NARROW.search(line) and (f"{f.relative_to(KERNELS)}:{i}") not in NOT_A_NORMALISATION:
                    bad.append(f"{f.relative_to(KERNELS)}:{i}: {stripped}")
    assert not bad, ("a layernorm kernel narrowing to bf16:\n  " + "\n  ".join(bad))


def test_no_layernorm_kernel_saves_a_narrowed_activation() -> None:
    """The bug this file was written for, checked without an exemption anywhere.

    `layernorm/triton/main.py` used to save x for the backward as `x_2d.to(torch.bfloat16)`. Every
    load in these files widens to fp32 and mean/rstd are stored fp32, so that one cast was the
    narrowest precision in the whole normalisation, on the one value the backward cannot recover.

    A narrowing INSIDE `save_for_backward` is that bug and nothing else. The four the check above
    tolerates are all outside it -- two GEMM operands and two gradient-rounding boundaries -- so
    this one needs no list, and a new one cannot be waved through by adding a line to it.
    """
    bad = []
    for d in DIRS:
        for f in sorted((KERNELS / d).rglob("*.py")):
            if "notes" in f.parts or f.name == "reference.py":
                continue
            tree = ast.parse(f.read_text())
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "save_for_backward"):
                    continue
                bad.extend(f"{f.relative_to(KERNELS)}:{node.lineno}: {ast.unparse(arg)}"
                           for arg in node.args if NARROW.search(ast.unparse(arg)))
    assert not bad, ("a narrowed activation saved for a layernorm backward:\n  "
                     + "\n  ".join(bad))
