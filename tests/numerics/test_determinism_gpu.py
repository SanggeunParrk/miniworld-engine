"""What "deterministic" means here, enforced rather than described.

A library that autotunes has a nuanced answer, which is exactly why it has to be written down. The
config a kernel launches with is selected per shape bucket from the tuned cache, so:

* **Within one process and one cache state** the config selection is fixed, so most kernels return
  bitwise-identical output for identical input. Most, not all -- see below.
* **Across cache states** -- a rebuild, a different GPU, a different config set -- a different
  config may win, and a different tile shape means a different reduction order. Results may then
  differ in the last bits. That is not a bug; it is what tuning is.
* **A kernel whose accumulation order is not fixed** is not bitwise reproducible even within one
  process, AND WHETHER IT IS DEPENDS ON THE PRECISION. It is also not a coin flip: the output is a
  DISTRIBUTION, and two runs can coincide. `augmented_attention_bwd_atomic_triton` in bf16 gives 7
  distinct outputs in 40 runs, the commonest 22 times -- so two consecutive calls agree about 37%
  of the time. Every claim in this file therefore takes `SAMPLES` runs, not two. Measured:
  `augmented_attention_bwd_atomic_triton` returns different bytes for provably identical input in
  both precisions (the checker's torch-computed reference side matched across the two calls, the
  kernel's did not). `layernorm_bwd_atomic_triton` -- also atomics -- reproduces in bf16 and does
  NOT in fp32: the atomics reorder the same additions either way, and bf16 has too few bits left
  for the difference to survive rounding.

This file used to say the second one "IS reproducible, so uses-atomics does not predict it". That
was measured in bf16, which was the only precision this suite ever ran -- the drivers ignored the
declared dtype, so every check was bf16 whatever the registry said. The first fp32 run contradicted
it. The lesson is not about atomics: a property measured in one precision was written down as a
property of the kernel.

The first bullet used to read "must be bitwise identical", full stop. It was a promise the library
does not keep.

Consumers cannot read that off the source, and the second half is the half that produces "your
library is non-deterministic" reports. See README's Determinism section.

The check runs each kernel's own checker twice and compares what the KERNEL produced. Both calls
must see the same tensors, and that is the harness's job, not the checker's: only two of the
fourteen checker modules ever called `checks._fixed()`, so ~100 checkers built their inputs from
unseeded `torch.randn`. This file's first version trusted them and compared two different inputs,
reporting differences up to 1.6e+04 -- output-scale, not reduction-scale, which is what gave the
lie away. `run_all.run_checker` now seeds at the single invocation point.

A sample rather than all 99: the point is the property, and one kernel per family covers the
distinct launch paths -- atomics, split reductions, persistent grids -- without paying for the
whole suite twice.
"""
from __future__ import annotations

import csv

import pytest

pytestmark = pytest.mark.gpu

from paths import REGISTRY

#: Kernels that are NOT bitwise reproducible, with what makes them so, as
#: `kernel -> {precision: reason}`. An exception list nothing verifies becomes a place to put
#: failures, so `test_a_declared_exception_is_still_one` asserts each entry still differs at the
#: precision it claims -- if one becomes reproducible there, the entry goes. Reproducibility is a property of the kernel AND the precision
#: it runs at: an atomic accumulation reorders the same additions, and whether that changes the
#: bytes depends on how many bits are there to change. bf16 rounds the difference away where fp32
#: keeps it, which is why this is not one flag per kernel.
NOT_BITWISE = {
    "augmented_attention_bwd_atomic_triton": {
        "bf16": "unordered atomic accumulation into dK/dV; the order varies run to run",
        "fp32": "unordered atomic accumulation into dK/dV; the order varies run to run",
    },
    "layernorm_bwd_atomic_triton": {
        # bf16 is NOT listed: it reproduces there, and this file's header used to cite exactly that
        # as proof that "uses atomics" does not predict non-reproducibility. It was measured in the
        # only precision the suite ever ran. The first fp32 run of this suite -- possible only once
        # the drivers started honouring the declared dtype -- shows the same kernel differing.
        "fp32": "unordered atomic accumulation into dgamma/dbeta; in fp32 the reordered additions "
                "survive rounding, where in bf16 they do not",
    },
}

#: How many times a kernel is run before this file says anything about it.
#:
#: TWO IS NOT ENOUGH, and the number comes from a measurement rather than a feeling.
#: `augmented_attention_bwd_atomic_triton` accumulates dK/dV with unordered atomics, so its output
#: is a DISTRIBUTION, not a coin flip. Measured on an A6000, one config (`configs/blk64`), 40 runs:
#:
#:     bf16  40 runs -> 7 distinct outputs, the commonest 22 times, then 9, 4, 2, 1
#:     fp32  40 runs -> 40 distinct outputs
#:
#: So in bf16 two consecutive calls land on the same bytes about 37% of the time, and
#: `test_a_declared_exception_is_still_one` -- which asserted that two calls DIFFER -- failed on
#: roughly one run in three. It had passed on the runs before it, which is the worst way for a
#: test to be wrong. At 12 samples the chance of every one coinciding is about (22/40)^11, under
#: 0.1%. The same number does the opposite job for the positive half: a kernel that is quietly
#: non-reproducible had a 37% chance of slipping through two calls.
#:
#: (Under the full `grid` config set both precisions give 40 distinct outputs in 40 runs -- the
#: autotuner re-benches and can pick a different config, which adds its own variation. The number
#: above is the harder case, one fixed config.)
SAMPLES = 12

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
    """Run a checker on the fixed seed and return only what the KERNEL produced."""
    from miniworld_engine.autotune.run_all import run_checker

    got = run_checker(checker)
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
        runs = [_actuals(row["check"]) for _ in range(SAMPLES)]
    except Exception as exc:
        if is_arch_gated(str(exc)):
            pytest.skip(f"not runnable on this device: {exc}")
        raise

    why = _excused(kernel)
    if why is not None:
        pytest.skip(f"{kernel} is declared non-reproducible at this precision: {why}")
    first = runs[0]
    assert first, "the checker returned nothing to compare"
    for n, later in enumerate(runs[1:], start=2):
        assert len(later) == len(first)
        for i, (a, b) in enumerate(zip(first, later, strict=True)):
            assert a.shape == b.shape, f"output {i}: {a.shape} vs {b.shape}"
            assert a.dtype == b.dtype, f"output {i}: {a.dtype} vs {b.dtype}"
            # Bitwise, not allclose. A tolerance here would hide exactly what is being asked:
            # whether the same inputs and the same cached config give the same bytes.
            assert torch.equal(a, b), (
                f"{kernel} output {i} differs between call 1 and call {n} of {SAMPLES} in one "
                f"process: max|diff| = {(a.float() - b.float()).abs().max().item():.3e}. Either "
                f"the kernel reads uninitialised memory, or its reduction order is not fixed for "
                f"a given config.")


def _excused(kernel: str) -> str | None:
    """The reason this kernel is not bitwise reproducible AT THE RUNNING PRECISION, if any."""
    from miniworld_engine.kernels.drivers import DTYPE_MODE

    return NOT_BITWISE.get(kernel, {}).get(DTYPE_MODE)


@pytest.mark.parametrize("kernel", sorted(NOT_BITWISE))
def test_a_declared_exception_is_still_one(kernel: str) -> None:
    """The other half of the statement. A kernel listed as non-reproducible must still be
    non-reproducible -- and its difference must be reduction-order sized, not output sized, or it
    is a bug wearing an exception's clothes."""
    import torch

    from miniworld_engine.autotune.run_all import declared_rtol, meets_arch

    with REGISTRY.open(newline="") as fh:
        rows = {r["kernel"]: r for r in csv.DictReader(fh)}
    row = rows[kernel]
    if not meets_arch(row):
        pytest.skip(f"needs {row['arch']}, this card is lower")
    why = _excused(kernel)
    if why is None:
        pytest.skip(f"{kernel} is only excused at other precisions; this process is another")

    runs = [_actuals(row["check"]) for _ in range(SAMPLES)]
    first = runs[0]
    differs = sorted({i for later in runs[1:]
                      for i, (a, b) in enumerate(zip(first, later, strict=True))
                      if not torch.equal(a, b)})
    assert differs, (
        f"{kernel} is listed in NOT_BITWISE ({why}) but reproduced exactly across {SAMPLES} runs. "
        f"Move it into SAMPLE -- the exception list must not outlive its reason.")
    from miniworld_engine.autotune.run_all import DEFAULT_RTOL
    from miniworld_engine.kernels.drivers import DTYPE_MODE

    # `or` would turn a declared band of 0.0 -- which test_declared_tolerance says must mean
    # EXACT -- into the default, and 5e-2 was the default's value copied by hand.
    declared = declared_rtol(row, DTYPE_MODE)
    band = DEFAULT_RTOL if declared is None else declared
    worst = max(runs[1:], key=lambda later: max(
        (later[i].float() - first[i].float()).abs().max().item() for i in differs))
    for i in differs:
        a, b = first[i].float(), worst[i].float()
        scale = b.abs().max().item()
        rel = (a - b).abs().max().item() / scale if scale else 0.0
        assert rel <= band, (
            f"{kernel} output {i} differs by rel {rel:.3e}, past its declared band {band:.0e}. "
            f"Reduction order costs last bits; this is too large to be that.")
