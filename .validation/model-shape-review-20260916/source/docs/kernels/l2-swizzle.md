# GROUP_M: the L2 swizzle axis, and the 18 kernels that do not use it

> **Current reference.** Every op name below is a `registry.csv` kernel and every cited path
> is the file that registry row names — both checked. It was not always: this file carried 21
> pre-rename names until they were mapped through `docs/kernels/rename-map.tsv`.

## What the axis is

`GROUP_M` decides how a flattened 1-D program id is unpacked into a 2-D `(pid_m, pid_n)`:

```python
width      = GROUP_M * grid_n
group_id   = pid // width
first_m    = group_id * GROUP_M
group_size = min(grid_m - first_m, GROUP_M)
pid_m = first_m + (pid % group_size)
pid_n = (pid % width) // group_size
```

`GROUP_M` row-blocks are issued before moving along N, so one column tile of B stays L2-resident
across those rows instead of being re-read per row. It sizes no array. Every value >= 1 covers the
output tiles exactly once, so it is **result-invariant and performance-only** — which is why it
belongs in the CSV like any other tuned value.

It is NOT a tile, so it is deliberately not spelled `BLOCK_*`. The rule is one-directional:
`BLOCK_*` always means a tuning target, but not every tuning target is a `BLOCK_*`. Naming a
program-order group count `BLOCK_` would claim a tile extent that does not exist. `GROUP_M` also
matches upstream Triton, whose matmul tutorial calls it `GROUP_SIZE_M`.

## The same name also serves a second, unrelated role

`GROUP_M` / `GROUP_N` appear in 48 places across 102 autotuned ops, in two roles that never overlap:

| role | count | in the CSV | in `key=[...]` | read by the kernel body | declaration |
|---|---|---|---|---|---|
| L2 swizzle (above) | 5 | yes | no | yes | `tl.constexpr` |
| autotune cache bucket | 43 | no | yes | **no** | plain runtime int |

The bucket role exists only so the autotuner keys its cache on `get_seq_group(M)` instead of a raw
token count. It is not a tuning value and the kernel never reads it. It used to be declared
`tl.constexpr`, which baked the bucket id into the compiled kernel and forced a separate compile per
sequence-length bucket for byte-identical code; it is a runtime argument now, and `key=[...]` keys on
runtime arguments perfectly well (`key=['M','NX','NC']` elsewhere in this repo does exactly that).

One name for two roles is what let five real defects hide: a mechanical "GROUP_M is a kernel
parameter but not a CSV column" check produced 34 hits, 29 of them correct-by-design, and the 5
genuine ones — kernels where nobody supplied the value at all — sat inside that noise.
**Renaming the bucket role (e.g. `seq_group`) is open work.**

## Where the swizzle applies, and where it is missing

The swizzle only helps a kernel that flattens a 2-D `(M, N)` output into a 1-D grid and whose
K-loop reads both an A row tile and a B column tile. 23 of 102 ops decode `pid` into
`(pid_m, pid_n)` that way. The other 79 have a genuinely 1-D grid (elementwise, reduction, or
flash-attention over one axis) and have nothing to reorder — the axis is correctly absent there.

Of those 23, **5 implement the swizzle**:

| op | file |
|---|---|
| `adaln_gemm_gate_triton` | `kernels/adaln/triton/fused3.py` |
| `adaln_fused3_gemm_gate_train` | `kernels/adaln/triton/fused3.py` |
| `cond_transition_bwd_gemm_triton` | `kernels/conditioned_transition/triton/train_fused.py` |
| `cond_transition_bwd_gate_squeeze_dx_triton` | `kernels/conditioned_transition/triton/train_fused.py` |
| `cond_transition_bwd_swiglu_dx_packed_triton` | `kernels/conditioned_transition/triton/train_fused.py` |

The remaining **18 use plain row-major order** — `pid_m = pid // num_pid_n; pid_n = pid % num_pid_n`:

| op | file |
|---|---|
| `gated_projection_gate_gemm_triton` | `kernels/bias_only_attention/triton/gate_out.py` |
| `cond_transition_expand_swiglu_triton` | `kernels/conditioned_transition/triton/composed.py` |
| `cond_transition_expand_swiglu_saveact_triton` | `kernels/conditioned_transition/triton/train_fused.py` |
| `cond_transition_bwd_dw_triton` | `kernels/conditioned_transition/triton/train_fused.py` |
| `layernorm_bwd_split_mmajor_triton` | `kernels/layernorm_linear/triton/mmajor_bwd.py` |
| `layernorm_partial_bwd` | `kernels/layernorm/triton/partial.py` |
| `layernorm_bwd_split_triton` | `kernels/layernorm/triton/persistent.py` |
| `trimul_gemm_gate_triton`, `trimul_bwd_gate_recompute_triton` | `kernels/tm1/triton/main.py` |
| `trimul_outproj_gemm_gate_triton`, `trimul_outproj_bwd_gate_recompute_triton` | `kernels/tm2/triton/main.py` |
| `transition_bwd_swiglu_recompute_triton` | `kernels/transition/triton/fused.py` |
| `transition_expand_swiglu_triton` | `kernels/transition/triton/main.py` |
| `trimul_transpose_triton` | `kernels/trimul_inproj/cute/front_sm100.py` |
| `trimul_gemm_gate_saveact_triton`, `trimul_outproj_gemm_gate_saveact_triton` | `modules/triangle_multiplication/baseline_dtv1.py` |
| `trimul_gemm_gate_packed_mmajor_triton`, `trimul_outproj_gemm_sigmoid_triton` | `kernels/trimul_inproj/triton/front.py` |

These are GEMMs; the swizzle is applicable to every one of them. Nothing about their shapes rules it
out, and the inconsistency looks unintended rather than measured. Adding it costs, per kernel, the
six lines above plus a `GROUP_M` column in each config set — and it must be **measured**, because
`GROUP_M=1` (plain row-major, i.e. what they do today) genuinely wins for some shapes.

Correctness is not at stake either way: the axis cannot change results.

## How to verify a change here

A swizzle edit must not move any number. Bench the op before and after at the same config set and
require the accuracy columns to be bit-identical, not merely close — the tile→output mapping is a
permutation, so any drift means the unpacking is wrong (an output tile written twice, or never).
