# Bias-only DiT, a pair of its own per sample (`bodit_psb`): the H100 CUDA inference path, 2026-10-27

Branch `perf/final-psb-inf` (from `perf/final-psb`, `8d074bc9`). H100 (cssb node, 80 GB HBM3), CUDA-graph replay, compiled, `benchmarks/runners/ladder.py`
groups `bodit_psb` (bf16) and `bodit_psb_fp32`, mode inference, `single` [1, 4, L, 768], `pair` [4, L, L, 128], 1 block, 16 heads x 48, key mask
(bench default). Median of 3 repeats; times in ms. Before this work `miniworld` had no CUDA path for a per-sample pair in inference and ran the PyTorch
reference (the ladder reported the rows as routing gaps).

| row | L | CUDA ms | portable ms | PyTorch ms | CUDA / portable | CUDA / PyTorch |
|---|---|---|---|---|---|---|
| inference bf16 | 256 | 0.1082 | 0.1332 | 0.2167 | 0.812 | 0.499 |
| inference bf16 | 512 | 0.2490 | 0.2861 | 0.5102 | 0.870 | 0.488 |
| inference fp32 | 256 | 0.1963 | 0.2178 | 0.3399 | 0.901 | 0.578 |
| inference fp32 | 512 | 0.4480 | 0.5101 | 0.8923 | 0.878 | 0.502 |

(Before: CUDA = the PyTorch reference, 1.38-1.65x the portable time. The first CUDA build of this branch, with the fp32 core of the training
path, measured 0.2013 / 0.4887 ms for fp32 at L 256 / 512 = 0.922 / 0.958 of portable: the fp32 core below is what brought the L 512 row under 0.95.)

## What runs

`BiasOnlyDiTBlock.forward` -> `integrations.bias_only_dit.serves_psb` / `update_psb` (no gradient, H100, single [1, B, L, d] with B > 1,
bf16, or fp32 on TF32 with an fp32 block and no autocast; `pair` [B, L, L, 128] or `bias=` [B, H, L, L]) -> opaque op `bias_only_dit_infer_psb` ->
`FusedBiasOnlyDiT.step_psb` (`kernels/bias_only_dit/cuda/runner.py`), the B = 1 12-launch step with a conditioning per sample and the attention core
replaced. Per block, eager kernel times at L 512 (profile, us; bf16 / fp32):

| kernel | bf16 | fp32 |
|---|---|---|
| pair LayerNorm + bias projection, one pass (`pair_bias_k` / `pair_bias_tc_k`) | 108.8 | 207.2 |
| attention core (`psb_fwd` / `psb32i_fwd`) | 48.4 | 58.0 |
| expand + SwiGLU (bf16 `gemm_swiglu_sm90`) | 20.2 | - |
| fp32 TF32 GEMMs at 128x128 tiles (v / g, expand, squeeze, table), SwiGLU rows separate (10.5) | - | 86.2 |
| conditioning-table, v / g, out, squeeze GEMMs (cuBLAS) | 47 | 38 (64x128 + 64x256) |
| row passes (adaln, residual + gate, LN rows) | 27 | 48 |

The hoist (`pair_bias_all` with no gradient -> `pair_bias_all_infer`, `_hoist_fwd_infer`) is the training-rows pair-bias kernel on the flattened
[B L L, 128] pair: LayerNorm on chip, tensor-core projection, bias written head-major [H, B, L, L] (the layout the core reads). Standalone, CUDA-graph, B = 4:

| pair kernel (us) | bf16 L 256 | bf16 L 512 | fp32 L 256 | fp32 L 512 |
|---|---|---|---|---|
| LN0 rows + cuBLAS GEMM (`_hoist_fwd`, the training hoist) | 80.6 | 300.5 | 149.9 | 561.8 |
| portable Triton one-pass (`pair_bias`) | 37.2 | 115.5 | 65.2 | 221.8 |
| `pair_bias_k` / `pair_bias_tc_k` (used) | 30.5 | 108.0 | 56.8 | 207.9 |
| `apb.pair_bias` per sample (TMA ring, bf16) | 39.3 | 116.2 | - | - |
| read floor at 3 TB/s | 22 | 89 | 45 | 179 |

## The fp32 inference core (`bo_psb32_inf.cu`)

`bo_psb32.cu`'s forward keeps v of the (head, sample) resident (114 KB at L 512), so one CTA of 16 query rows runs per SM, and every phase of a
tile (softmax, P v on three warps, gated store with a dependent global load) waits for the one before. 99 us at B = 4, L 512 for a 22 us read.
`psb32i_fwd` (no statistics; inference only): a 3-deep ring of bias tiles (2-deep with two CTAs per SM at L <= 256), the product on all eight warps (4 key quarters
x 2 column halves, partial sums added in a fixed order through shared memory), two rows softmaxed together per warp, the gate values loaded right
after the barrier. Steps, B = 4 L 512 (us): 99.2 -> 82.2 (prefetch depth, eight warps) -> 58.0 (gate load hoisted, two-row softmax); L 256: 25.3 ->
23.2 -> 20.4. Still latency-bound (the floor is 22 / 6 us): v-resident shared memory caps a CTA at one tile and one or two CTAs per SM.

## Tests

`tests/gpu/sm90/test_bias_only_dit_psb_infer_gpu.py` (new, 40 tests): block inference bf16 and fp32, hoisted bias and pair-fed, B 2-5, L 128-768, mask none /
shared / per sample, against the fp32-IEEE PyTorch module (< 8e-3 bf16, < 2.5e-3 fp32), against the same samples one at a time, bitwise repeatable; the
deterministic sample chunking; head layouts 24 x 32, 12 x 64, 16 x 64; L off the tile and the switch keep the module composition; `psb32i_fwd` against
`psb32_fwd` (1e-5) and an fp64 softmax (2e-3), bitwise repeatable. The training psb tests, `test_h100_bias_only_dit_gpu.py` and the other files that touch the
touched modules: see the results at the end of this record.

## Not done / left

* More than one block: `pair_bias_all` with nb > 1 blocks has 32+ weight rows and keeps LN0 + cuBLAS (3 passes over the pair); a multi-block one-pass kernel
  (N = 16 nb outputs per row tile) would cut the hoist of a deep stack to one read.
* The bf16 core `psb_fwd` (48 us at L 512 for a 11 us read) has the same dependent-load epilogue as the fp32 core had; not touched (rows already 0.81 / 0.87).
* The pair read is at 83-86 % of the HBM floor in both dtypes; the front of the block (conditioning tables, AdaLN, v / g) does not depend on the bias and
  could overlap the pair kernel, but the pair kernel fills every SM's registers.
* L > 768 and the fp32 core beyond L 512 (batched-GEMM core) are served but not tuned.

## Test results (H100, this branch)

* `test_bias_only_dit_psb_infer_gpu.py` + `test_bias_only_dit_psb_gpu.py` (training psb, untouched code): 115 passed.
* `test_h100_bias_only_dit_gpu.py`: 84 passed, 36 skipped (the same count as 2026-10-19).
* `tests/cpu` + `tests/architecture`: 4373 passed, 6 failed, 2 errors -- the same 6 failures and 2 errors on the base commit `8d074bc9`
  (`test_compact_search_space`, `test_the_sweep_page_matches_the_repo`, `test_key_flags_are_built_on_both_sides`, `test_no_stale_caches`,
  `test_a_ladder_offers_what_wins`: stale autotune caches / sweep page, unrelated to this change).
