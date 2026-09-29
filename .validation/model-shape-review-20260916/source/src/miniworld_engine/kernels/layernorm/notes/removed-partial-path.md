# The `partial` backward path, and why it is gone

`layer_norm` backward used to pick between three triton paths — `atomic`, `persistent` and
`partial` — plus a hand-CUDA one for bf16 at 128 <= N <= 512. `partial` is removed.

## It never won

`autotune/data/ln_bwd_dispatch/` records all three timings per (d, M) bucket, measured on the real
tensors. Over 49 buckets across an A5000 and an A6000 it won **zero**. Using it alone would cost:

| path | A5000 median / worst | A6000 median / worst | buckets over the 1.059x noise floor |
|---|---|---|---|
| `atomic` | 1.0000 / 1.0402 | 1.0000 / 1.0000 | 0 of 49 |
| `persistent` | 1.3862 / 3.2863 | 1.4566 / 3.3581 | 32 of 49 |
| `partial` | 1.9560 / 3.8035 | 1.4590 / 3.8249 | **47 of 49** |

`persistent` stays. It wins three A5000 buckets outright, and while those margins (up to 4.0%) sit
inside the noise floor, the two cards measured here are the same architecture — the case for a
grid-stride backward is about SM count and L2, and sm_86 twice is not evidence about a card with
either in quantity.

## It was superseded by design, not just by measurement

`triton/persistent.py` was written to fix three named weaknesses in this path, and its docstring
lists them: a scalar row loop instead of a vectorized 2D tile, `cdiv(M, block_m)` partial dw/db
rows (~16k at M=1M) feeding a large buffer and a final reduce, and `BLOCK_N = next_pow2(N)`
register pressure. So `persistent` is the same algorithm done properly, which is why the numbers
above are not close.

## What was actually deleted

`triton/partial.py` (177 lines). Its two opaque ops, `_partial_fwd` and `_partial_bwd`, were
launched from nowhere — the file survived because `compile_native` imported `_bwd_block_m` from it
to size a buffer for `_bwd_partial_impl`, and that function launched **`_ln_bwd_persistent`**, the
persistent kernel, with a different grid. So the path was not a third kernel at all; it was the
persistent kernel run at a grid the measurements reject. That is also why it never had a row in
`registry.csv` and was never tuned as a kernel in its own right.

Also gone: `_use_partial_reduction`, `_bwd_partial_impl`, `partial` from `_VALID_BWD_PATHS`, from
the calibration `impls` dict, from `_static_bwd_path`, from the dispatch chain, and the
`ln_partial_reduction` switch and `ln_bwd_path` value in `autotune/builder.py`. The recorded
`partial` timings are pruned from `ln_bwd_dispatch`, because a stale time for a path that no longer
exists would re-derive a winner nothing can run.

Kept: the `triton-partial` labels in `viz/style.py`. Those name series in archived benchmark plots,
which still say what they said.
