"""What "deterministic" means here, enforced rather than described.

A library that autotunes has a nuanced answer, which is exactly why it has to be written down. The
config a kernel launches with is selected per shape bucket from the tuned cache, so:

* **Within one process and one cache state** the selection is fixed, and repeated calls on identical
  inputs must be bitwise identical. That is what this file asserts.
* **Across cache states** -- a rebuild, a different GPU, a different config set -- a different
  config may win, and a different tile shape means a different reduction order. Results may then
  differ in the last bits. That is not a bug; it is what tuning is.

Consumers cannot read that off the source, and the second half is the half that produces "your
library is non-deterministic" reports. See README's Determinism section.

The check runs each kernel's own checker twice and compares what the KERNEL produced (the checkers
seed their inputs, so both calls see the same tensors). A sample rather than all 99: the point is
the property, and one kernel per family covers the distinct launch paths -- atomics, split
reductions, persistent grids -- without paying for the whole suite twice.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

REGISTRY = Path(__file__).resolve().parents[1] / "src/miniworld_engine/kernels/registry.csv"

#: One kernel per family that has a checker, chosen for launch-path variety rather than coverage:
#: an atomic accumulation, a split reduction, a persistent grid, a fused epilogue. Declared rather
#: than derived so a family that loses its checker is a visible change here.
SAMPLE = (
    "adaln_fwd_triton",
    "layernorm_fwd_saveact_triton",
    "layernorm_bwd_atomic_triton",          # atomics: the interesting case for run-to-run order
    "triangle_attention_fwd_triton",
    "bias_only_attention_fwd_triton",
    "augmented_attention_fwd_triton",
    "transition_expand_swiglu_triton",
    "cond_transition_swiglu_triton",
    "gated_projection_gate_triton",
    "trimul_gemm_gate_triton",
)


def _checkers() -> dict[str, str]:
    with REGISTRY.open(newline="") as fh:
        return {r["kernel"]: (r.get("check") or "").strip() for r in csv.DictReader(fh)}


def test_the_sample_still_names_real_kernels_with_checkers() -> None:
    """Guard the guard: a renamed kernel would silently shrink this file to nothing."""
    declared = _checkers()
    missing = [k for k in SAMPLE if not declared.get(k)]
    assert not missing, (
        f"{missing} are in SAMPLE but have no checker in registry.csv -- either they were renamed "
        f"or their checker was dropped, and this file would have quietly stopped covering them.")


def _actuals(checker: str):
    """Run a checker and return only what the KERNEL produced, as a flat list of tensors."""
    from miniworld_engine.autotune.run_all import _resolve

    got = _resolve(checker)()
    pairs = got if isinstance(got, dict) else {"out": got}
    return [actual for actual, _expected in pairs.values()]


@pytest.mark.parametrize("kernel", SAMPLE)
def test_two_calls_in_one_process_are_bitwise_identical(kernel: str) -> None:
    """The half of the determinism statement that IS a promise."""
    import torch

    from miniworld_engine.autotune.run_all import is_arch_gated, meets_arch

    with REGISTRY.open(newline="") as fh:
        rows = {r["kernel"]: r for r in csv.DictReader(fh)}
    row = rows[kernel]
    if not meets_arch(row):
        pytest.skip(f"needs {row['arch']}, this card is lower")

    try:
        first = _actuals(row["check"])
        second = _actuals(row["check"])
    except Exception as exc:
        if is_arch_gated(str(exc)):
            pytest.skip(f"not runnable on this device: {exc}")
        raise

    assert len(first) == len(second) == max(len(first), 1)
    for i, (a, b) in enumerate(zip(first, second, strict=True)):
        assert a.shape == b.shape, f"output {i}: {a.shape} vs {b.shape}"
        assert a.dtype == b.dtype, f"output {i}: {a.dtype} vs {b.dtype}"
        # Bitwise, not allclose. A tolerance here would hide exactly what is being asked: whether
        # the same inputs and the same cached config give the same bytes.
        assert torch.equal(a, b), (
            f"{kernel} output {i} differs between two calls in one process: "
            f"max|diff| = {(a.float() - b.float()).abs().max().item():.3e}. Either the kernel "
            f"reads uninitialised memory, or its reduction order is not fixed for a given config.")
