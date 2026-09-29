"""The claim `compile_budget`'s safety rests on: a config that takes a minute to compile is slow.

`compile_budget` skips configs it cannot prove will be killed -- 35 of 18,393 in its own fixture
compiled fine and would be skipped anyway. That is only acceptable if such a config was never
going to be chosen. The reason to expect it holds is mechanical rather than statistical: one cause
produces both symptoms.

    the kernel needs more registers than a thread has
      -> ptxas grinds looking for an allocation          the compile is slow
      -> the overflow is spilled to memory               the kernel is slow

So this is not "slow compiles happen to lose". It is the same register pressure, measured twice.

The fixture is `(bench microseconds, compile milliseconds)` for every config of five
(op, shape bucket) pairs of one A6000 run -- two reductions, two attention kernels and a
GEMM-shaped one, 5,526 configs. Buckets with no slow compile at all are left out; they would make
every assertion here vacuously true. Nothing else is kept: the question needs two numbers per
config and the config signature would only make the file bigger.

    op                              configs   over 30 s   best rank of one   off the best time
    augmented_attention_fwd             587       9       559  (bottom  5%)        25.2x
    augmented_attention_bwd_split       534       7       521  (bottom  2%)        28.5x
    layernorm_stats  K=128             1434      18      1225  (bottom 15%)         6.4x
    layernorm_stats  K=256             1434      18      1327  (bottom  7%)         6.4x
    transition_fwd_b2b_ktiled          1537      12      1290  (bottom 16%)        32.7x

What this does NOT contain is the three kernels `compile_budget` actually produces false positives
on -- `_dgrad_epi`, `_dx_swiglubwd_kernel`, `fused_sigmoid_gate_fwd_kernel`. Their grids are the
large ones and had not finished benching when this was written. So the argument that those 35
configs cost nothing is an inference from the property measured here, not a measurement of them.
"""
from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "compile_budget" / "bench_vs_compile.csv.gz"
SLOW_COMPILE_S = 30.0


@pytest.fixture(scope="module")
def buckets():
    out = defaultdict(list)
    with gzip.open(DATA, "rt") as fh:
        for row in csv.DictReader(fh):
            out[row["op_bucket"]].append((float(row["bench_us"]), float(row["compile_ms"]) / 1000))
    for v in out.values():
        v.sort()
    assert len(out) >= 3, f"only {len(out)} buckets in the fixture"
    return dict(out)


def test_the_fixture_contains_slow_compiles_at_all(buckets) -> None:
    """Otherwise every assertion below is vacuously true."""
    for name, rows in buckets.items():
        slow = [c for _, c in rows if c > SLOW_COMPILE_S]
        assert len(slow) >= 5, f"{name}: only {len(slow)} configs compile slowly"


def test_no_slow_compile_is_anywhere_near_the_fastest_config(buckets) -> None:
    """Measured: the best-placed slow compile sat at rank 1225 of 1434, 1327 of 1434 and 1290 of
    1537 -- the bottom 7-16% by measured time in all three. If this ever fails, the false
    positives `compile_budget` accepts have started to cost something and its rule needs the
    tightening that `test_compile_budget_predicts_the_kills` prices out."""
    for name, rows in buckets.items():
        ranks = [i for i, (_, c) in enumerate(rows) if c > SLOW_COMPILE_S]
        assert ranks, name
        assert min(ranks) > 0.7 * len(rows), (
            f"{name}: a config compiling in {rows[min(ranks)][1]:.0f}s ranked "
            f"{min(ranks) + 1} of {len(rows)}")


def test_a_slow_compile_is_multiples_off_the_best_time(buckets) -> None:
    """Not marginally worse -- 6.4x, 6.4x and 32.7x off the best in the three buckets."""
    for name, rows in buckets.items():
        best = rows[0][0]
        slow = [b for b, c in rows if c > SLOW_COMPILE_S]
        assert min(slow) > 3 * best, (
            f"{name}: a slow compile came within {min(slow) / best:.1f}x of the best config")


def test_the_two_are_related_across_the_whole_grid_not_just_the_tail(buckets) -> None:
    """The mechanical story predicts a trend, not only a bad tail: the slowest-compiling fifth of
    a grid should be slower to RUN than the fastest-compiling fifth."""
    for name, rows in buckets.items():
        by_compile = sorted(rows, key=lambda r: r[1])
        q = len(rows) // 5
        cheap = sorted(b for b, _ in by_compile[:q])
        dear = sorted(b for b, _ in by_compile[-q:])
        assert dear[len(dear) // 2] > cheap[len(cheap) // 2], (
            f"{name}: the slowest-compiling fifth runs no slower than the fastest-compiling fifth")
