# Prebuilt-bundle gaps found by the empty-cache wheel check (2.4.4 candidate), 2026-11-04

Never edited after the fact (`docs/records/`). Written while preparing 2.4.4.

## How it was measured

The 2.4.4 split wheels were installed (cp312, torch 2.13 + cu129), the engine kernel cache empty per file (`XDG_CACHE_HOME`), `CUTLASS_PATH`
unset / nonexistent, `MINIWORLD_ENGINE_JIT_FALLBACK=0`, and every `tests/gpu/sm90/test_*.py` the code graph names (75 files, 77 queue entries with two extras) was run once, one file per
GPU, as a work queue on 4 H100s. A kernel the bundle lacks raises `PrebuiltMissing` (or `CutlassMissing` for the CUTLASS ones). 54 queue entries ran clean
(Error 0, PrebuiltMissing 0); 23 had failing tests. The same failing files were then run against the **2.4.3** wheels with the same recipe:
the per-file Error / PrebuiltMissing / passed / failed counts are identical on 2.4.3 for every file except `test_trimul_h100_long_l_gpu.py`,
which fails fewer tests on 2.4.4 (62, against 73) because 2.4.4 added the `trimul_wide_train` coverage. **No regression against 2.4.3.**

## The rule these tests expose

With JIT on (the warm-cache runs the repository's suites use) all of these pass. With JIT off they reach shapes, widths, batch sizes or options that
`registry_module.csv` does not declare, so no coverage case builds their kernel. The registry is the contract: what it declares is covered and bundled;
what a test adds beyond it is not. Next step per family: a registry decision (declare the shape and cover it) or cut the case from the test.

## Classes

* **a**: the failing parameters are outside `registry_module.csv` (evidence: the parameter against the registry rows).
* **a-guess**: the missing kernel name encodes a variant the registry does not declare; the parameters were not each matched against the registry.
* **b-candidate**: declared or MiniWorld-reachable but not covered (fixed in 2.4.4 where noted).
* **unknown**: not classified.

| file | failing tests | class | missing kernels (first of each kind) and why |
|---|---|---|---|
| `test_augmented_attention_psb_gpu.py` | 4 | a | head layouts 16x64, 24x32 (`augattn_sm90_attn_fwd_DHP64_ROW_WIDTH1024_PSB1`, `..._DHP32_ROW_WIDTH768_PSB1`); declared: token 16 heads x 48, atom 4 heads x 32 |
| `test_augmented_attention_bf16_sm90_gpu.py` | 1 | a-guess | `augattn_sm90_attn_dq`: A = 4 samples (not a multiple of 3, the per-sample-atomics backward); MiniWorld trains 48 |
| `test_h100_apb_training_gpu.py` | 35 | a | 12 / 16 / 24 heads, d 384 / 512, bf16 parameters; declared APB training: d_single 384, d_pair 128, 8 heads (`augattn_..._DHP16/32_ROW_WIDTH384/512_NWG1_APB...`) |
| `test_h100_token_dit_layouts_gpu.py` | 30 | a | 12 / 16 / 24 heads, 768 / 1024 wide (`token_dit_train_rows_cuda_{768,1024}_{32,64}_rpb2`, `augattn_..._tf32_NHP{12,16,24}`) |
| `test_token_dit_batched_pairs_gpu.py` | 4 | a-guess | `token_dit_train_rows_cuda_rpb2_{n4,dc256}`, `..._768_64_rpb2` |
| `test_token_dit_pair256_gpu.py` | 5 | a-guess | `augattn_sm90_attn_dq_NWG2_BWD_BF161`, `augattn_sm90_attn_tf32_NHP24_DHP32` |
| `test_ops_cuda_wiring_gpu.py` | 2 | a-guess | `augattn_sm90_attn_dq_NWG2_..._BWD_BF161` (bf16 training backward variant) |
| `test_ops_no_triton_gpu.py` | 2 | a-guess | `augattn_sm90_attn_dq_NWG2_BWD_BF161` |
| `test_h100_bias_only_dit_gpu.py` | 5 | a | `bias_only_dit_inf3_sm90a`: the `inf3_h100` three-kernel step, off by default on Hopper (measured slower); the test forces the cluster size (`_FORCE_CL` 8 / 16). `bias_only_dit` has no `registry_module.csv` row |
| `test_h100_trimul_batch_gpu.py` | 2 | a | training batches B 2 / 8 (`d64_train/d64_bwd_wg.cu`); the registry declares no batch |
| `test_h100_trimul_batch_wide_gpu.py` | 75 | a | D256 batches B 2..16 (`mw_b200style_k1_direct`, `k1ssregtune.cu`, `k1sssave.cu`, `b1_fused.cu`) |
| `test_trimul_h100_b200style_gpu.py` | 56 | a-guess | the b200style variants (`k1sssave.cu`, `k3.cu`, `k1ssregtune.cu`, `mw_b200style_k1_direct`) |
| `test_trimul_h100_bidir_hidden2x_gpu.py` | 17 | a | bidirectional hidden 2x; the registry's bidirectional TriMul is d_pair 128 / hidden 128 (`trimul_h100/generated/unit`) |
| `test_trimul_h100_d64_b200gate_gpu.py` | 6 | a-guess | `d64_train/uni_bwd_wg.cu`, `d64_bwd_wg.cu` |
| `test_trimul_h100_hidden2x_gpu.py` | 10 | a | hidden 2x single direction (`b1_fused.cu`, `d64_bwd_wg.cu`, `output/save_k3.cu`); not in the registry |
| `test_trimul_h100_long_l_gpu.py` | 62 | a | training at L 1152 .. 2560; the registry declares 128 .. 768 (`generated/mma_n` per length, ...) |
| `test_trimul_h100_small_buckets_gpu.py` | 5 | a-guess | `b7_768/joint.cu`, `b1/b1_fused.cu` |
| `test_trimul_single_h100_gpu.py` | 1 | a-guess | `single_output/kernel.cu` |
| `test_transition_fused_sm90a_gpu.py` | 2 | a-guess | `transition_fused_sm90a_c66_r8_s0_input_barrier_v1_fwdtune1` |
| `test_transition_n2_sm90a_gpu.py` | 3 | a-guess | `transition_fused_sm90a_c66_r15_s1_..._n2`, `transition_wide_sm90a_d512_c132_n2` |
| `test_transition_wide_sm90a_gpu.py` | 5 | a | d 512 and `saved_ab_v1` at d 384 (`transition_wide_sm90a_d384_c132_saved_ab_v1`, `..._d512_c132`); declared pair widths 128 / 256 / 384 |
| `test_triattn_h100_b200style_gpu.py` | 74 | a | `CutlassMissing: triattn_h100_b200style_module_v1` (74 parameters, widths 64 .. 512). The b200style `h100_triattn` module is the explicit benchmark baseline ("Explicit entry point only: no production dispatch changes"); `TriangleAttention` serves every registered width with `triattn_h100_sm90` first and reaches the b200style path only with `_h100_cuda = False` or `_h100_b200style_cuda`. An environment difference of the wheel run (no CUTLASS, nothing bundled) rather than an engine bug |
| `test_h100_master_weights_gpu.py` | 12 | b-candidate | see "Fixed in 2.4.4" below |

## Fixed in 2.4.4

* `trimul_wide_train` coverage (bidirectional TriMul hidden 256 / 384 / 512, L 384 / 768, masked, dropout, training, bf16 and **master** (fp32
  parameters, bf16 activations: the `-DMASTER_FP32=1` builds)): `front_d256`, `front_pre_d512` and the wide-training kernels were in no bundle up to
  2.4.3. `test_h100_module_wiring_gpu.py` (nine failures from a wheel before) now passes from the wheels.
* `msa_module` coverage (see CHANGELOG 2.4.4).

## Known, not fixed

* On cu130 the post-link register-pool cubin `trimul_h100/mw_d256_register_budget_source.pool120` is not in the bundle (the patch is built for the cu129
  SASS and declines on cu130, the caller keeps the base cubin). With JIT fallback off `derived_cubin` raises `PrebuiltMissing` before it asks the patch;
  with JIT on nothing is compiled. The H100 nodes here have driver 575 (CUDA <= 12.9): cu130 could not be run.
* PWA training at a length outside the native scope (N 128 / 256 / 512 / 640) followed in the same process by a native one (N 384 / 768) fails in
  `pwa_glue3` with `CUDA error: invalid argument` (also on v2.4.3).
