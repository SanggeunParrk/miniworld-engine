# H100 ops gate: performance of the routing against the previous Triton path (2026-11-01)

Branch `fix/h100-ops-gate`. Follow-up to [h100_ops_gate_20261101.md](h100_ops_gate_20261101.md), whose **Performance** section said "not measured".

Two measurements of the same harness. **Before** (job 41372, branch at 8eea3f6d7) routed the `ops.*` facades that lost their Triton kernel to PyTorch
statements: 208 of 342 rows were more than 5% slower than the Triton path, 86 of them more than 2x (layer_norm_linear up to 10.5x, triangle attention
bf16 up to 12.4x, TriMul up to 10.8x), and `ops.triangle_attention` at L = 1024 ran out of memory (the PyTorch core builds 16 GiB of scores). **After**
(this file's tables; jobs 41387 and 41393) the facades are wired to the hand-CUDA kernels that already serve the same ops
(`kernels/_h100_ops.py`, thin wrappers; no new kernel, no Triton): 107 rows are still more than 5% slower, 6 more than 2x, none out of memory.

## What serves each facade now (H100)

| facade | before (41372) | now |
|---|---|---|
| `ops.triangle_attention` bf16, 32-channel heads | PyTorch core (OOM at L1024) | the sm_90 core of the module path (`h100_sm90`: forward, delta, key-side and query-side backward) as an autograd function of the projected q / k / v / bias (`_TriangleCore`; L padded to 128 with masked keys); QK norm on the general CUDA RMSNorm (`rmsnorm_cuda`) |
| `ops.triangle_attention` fp32 / other heads | PyTorch core | PyTorch core chunked over the rows (score budget 4 GiB per chunk, checkpointed in training, a function of the shapes only): L1024 runs, bit-reproducibly |
| `ops.triangle_multiplicative_update`, `ops.bidirectional_...` bf16 | PyTorch statements | the module's hand-CUDA TriMul (`integrations.trimul_h100`: K1 / K3 inference, B1..B7 training, chunked batches) through a module-shaped view of the weights; fp32 stays the statements (TriMul is bf16 only) |
| `ops.layer_norm_linear` d_pair 128 -> 16 | `F.layer_norm` + `F.linear` | no gradient, bf16: `apb.pair_bias` (LayerNorm + projection, weight folded); otherwise the token DiT's `pair_bias` / `pair_bias_bwd` row kernels (one pass each way, TF32 projection) |
| `ops.rms_norm_modulation` | fp32 statements (a dozen launches) | activation-dtype statements: one RMSNorm, one GEMM over [Ws; Wsh; Wg] with the 1 of `1 + scale` in the bias epilogue, one `addcmul` |
| `ops.conditioned_transition` bf16 | PyTorch statements | `ct_inference` / `ct_tail_train` (the module's tail: expand + SwiGLU GEMM, squeeze + gate GEMM) |
| `ops.swiglu_ffn` | PyTorch statements | bf16 inference: `ct_inference` with a gate of exactly one (`cond = x`, `Wsc = 0`, bias 30); training and fp32: cuBLAS + the CUDA SwiGLU row kernels (`_SwiGLUFFN`) |
| `ops.gated_linear` bf16 | PyTorch statements | the CUDA sigmoid-gate row kernels around one GEMM (`_GatedLinear`) |
| `ops.gated_residual` | `x + gate * branch` | one `addcmul` (two products backward) |
| `ops.augmented_attention_pair_bias`, calls the sm_90 core does not take | PyTorch reference (einsum, L^2 fp32 scores) | `F.scaled_dot_product_attention` with the additive pair bias (cuDNN attention without a gradient at bf16, memory-efficient kernel otherwise) |
| `BiasOnlyAttention._attention_weights` (shared pair, shapes the CUDA block does not take) | `ln_pair` + `to_bias` statements | `ops.layer_norm_linear`'s CUDA path |

## Method

* One H100 80GB HBM3 per case (node01; before: 8 GPUs, one Slurm job as a work queue, job **41372**; after: 4 GPUs, job **41387** for the full harness and **41393** for the
  `augmented_attention_pair_bias` groups after the padded-core experiment was removed), torch 2.13 / cu129, same checkout for both sides.
* **new** = default settings (H100: no Triton through `ops.*` / the fallbacks). **old** = `settings.portable_on_hopper_blackwell=True` (the Triton path the branch replaced), same inputs,
  modules rebuilt per side (the module routing is resolved at build).
* CUDA-graph timing, warm (3 eager warm-ups, 10 replay warm-ups, median of 5 batches of replays, each side run twice, the better kept). Never eager.
* `infer` = `no_grad` forward; `fwd+bwd` = forward with autograd plus `autograd.grad` over every floating input (and the module parameters).
* Shapes: pair ops on `(1, L, L, 128)`, token ops on `(4, L, D)` with D = 768 (cond 384), atom-style `D = 128`; trimul / triangle attention D128 (4 heads); augmented-attention op on `(2, 1, H, L, D)`;
  `BiasOnlyDiTBlock(16 x 48)` with 3 samples (shared pair / pair per sample), bf16 = autocast over fp32 masters, fp32 = plain; `AugmentedAttentionPairBias(768, 384, 128, 16)` with 2 samples. L = 128 / 256 / 512 / 1024 (plus L100 / L200).
  Weights random, the same on both sides.
* `new/old` > 1 means the new path is slower. Latencies below ~0.02 ms are within graph-timing noise (about +-10%).
* The first attempt of some cases raised `During CUDA graph capture, autograd node AccumulateGrad has a stale reference to the default stream` (8 cases, both sides: new TriMul facade, old Triton fp32 swiglu); the
  second attempt of each ran and is the one kept (best of two), so every row below is a successful capture. Not diagnosed (a harness warm-up quirk or the TriMul caller hooks); the modules' own capture tests pass.
* Harness and logs outside the checkout: `~/miniworld-engine-scratch/h100-ops-gate-logs/perf/` (`bench_gate.py`, `agg_final.py`, `out_41372`, `out_41387`, `out_aug_41393`, `out_final`).
* `dev code-graph --diff main` ran first as its own job before each test / measurement submission (**41376**, **41384**, **41392**): no CUDA source changed, nothing rebuilt; it reaches 192 CPU / 195 GPU test files, of which the ones named in
  the record of the gate (and `test_ops_cuda_wiring_gpu.py`) ran. `dev restamp-native --op trimul_fwd_sm90_cuda` (owed because `_h100_pack.py` moved) ran in job 41377.

## Result in one picture

new / old latency of fwd+bwd per op and L, before (left) and after (right) (blue < 1, new faster; red > 1, new slower).

![new / old](figures/h100_ops_gate_perf_20261101.png)

## Still slower than the previous Triton path by more than 5%

Ratios over L = 128 ... 1024 (L100 / L200 where listed), before in the sixth column. Rows within 1.05 are not listed.

| op / path | shapes | dtype | mode | new/old now | new/old before | rows > 1.05 |
|---|---|---|---|---|---|---|
| layer_norm | pair D128 | bf16 | infer | 1.00 - 1.12 | 1.00 - 1.09 | 1 of 4 (L 256) |
| layer_norm | token D768 | bf16 | infer | 0.93 - 1.09 | 0.91 - 1.11 | 1 of 4 (L 512) |
| layer_norm | token D768 | fp32 | fwd+bwd | 0.97 - 1.32 | 1.06 - 1.32 | 3 of 4 (L 256, 512, 1024) |
| swiglu_ffn | pair D128 h512 | bf16 | infer | 1.28 - 1.39 | 1.96 - 2.28 | 4 of 4 (L 128, 256, 512, 1024) |
| swiglu_ffn | pair D128 h512 | bf16 | fwd+bwd | 1.05 - 1.21 | 1.46 - 1.64 | 4 of 4 (L 128, 256, 512, 1024) |
| swiglu_ffn | token D768 h1536 | bf16 | infer | 1.24 - 1.30 | 1.13 - 1.23 | 4 of 4 (L 128, 256, 512, 1024) |
| conditioned_transition | atom D128 c128 n2 | fp32 | fwd+bwd | 1.00 - 1.08 | 1.00 - 1.09 | 2 of 4 (L 512, 1024) |
| rms_norm_modulation | D128 c128 | bf16 | infer | 1.51 - 2.00 | 6.91 - 8.63 | 4 of 4 (L 128, 256, 512, 1024) |
| rms_norm_modulation | D128 c128 | bf16 | fwd+bwd | 1.67 - 1.95 | 5.11 - 5.97 | 4 of 4 (L 128, 256, 512, 1024) |
| layer_norm_linear | pair D128 ->16 | bf16 | infer | 0.94 - 1.77 | 5.64 - 10.49 | 2 of 4 (L 128, 256) |
| layer_norm_linear | pair D128 ->16 | bf16 | fwd+bwd | 1.11 - 1.70 | 4.48 - 8.54 | 4 of 4 (L 128, 256, 512, 1024) |
| gated_residual | pair D128 | bf16 | fwd+bwd | 0.83 - 1.15 | 1.03 - 1.34 | 3 of 4 (L 256, 512, 1024) |
| gated_residual | pair D128 | fp32 | fwd+bwd | 1.03 - 1.13 | 1.10 - 1.37 | 3 of 4 (L 256, 512, 1024) |
| gated_residual | token D768 | bf16 | fwd+bwd | 0.83 - 1.26 | 1.01 - 1.56 | 1 of 4 (L 128) |
| gated_residual | token D768 | fp32 | fwd+bwd | 0.93 - 1.38 | 1.15 - 1.65 | 3 of 4 (L 128, 256, 1024) |
| gated_linear | pair D128 | bf16 | infer | 1.14 - 1.64 | 1.54 - 2.23 | 4 of 4 (L 128, 256, 512, 1024) |
| gated_linear | pair D128 | bf16 | fwd+bwd | 1.09 - 1.19 | 1.31 - 1.60 | 4 of 4 (L 128, 256, 512, 1024) |
| gated_linear | token D768 | bf16 | infer | 1.03 - 1.07 | 1.33 - 1.35 | 2 of 4 (L 128, 512) |
| gated_linear | token D768 | bf16 | fwd+bwd | 1.07 - 1.17 | 1.22 - 1.28 | 4 of 4 (L 128, 256, 512, 1024) |
| triangle_attention | D128 H4 | bf16 | infer | 0.73 - 1.07 | 5.03 - 12.38 | 1 of 4 (L 128) |
| augmented_attention_pair_bias [supported] | bf16 16x48 B1 | bf16 | infer | 0.83 - 1.34 | 0.84 - 1.35 | 3 of 4 (L 128, 256, 512) |
| augmented_attention_pair_bias [supported] | bf16 16x48 B1 | bf16 | fwd+bwd | 0.78 - 1.48 | 0.77 - 1.52 | 2 of 4 (L 128, 256) |
| augmented_attention_pair_bias [supported] | bf16 4x32 B1 | bf16 | infer | 1.19 - 1.40 | 1.24 - 1.41 | 4 of 4 (L 128, 256, 512, 1024) |
| augmented_attention_pair_bias [supported] | bf16 4x32 B1 | bf16 | fwd+bwd | 0.96 - 1.58 | 1.00 - 1.62 | 3 of 4 (L 128, 256, 512) |
| augmented_attention_pair_bias [unsupported] | fp32 16x48 | fp32 | infer | 1.55 - 3.60 | 3.09 - 7.55 | 4 of 4 (L 128, 256, 512, 1024) |
| augmented_attention_pair_bias [unsupported] | fp32 16x48 | fp32 | fwd+bwd | 1.80 - 2.24 | 2.47 - 4.21 | 4 of 4 (L 128, 256, 512, 1024) |
| augmented_attention_pair_bias [unsupported] | bf16 8x64 | bf16 | infer | 0.99 - 2.07 | 3.94 - 18.31 | 3 of 4 (L 256, 512, 1024) |
| augmented_attention_pair_bias [unsupported] | bf16 8x64 | bf16 | fwd+bwd | 1.39 - 1.66 | 3.74 - 7.00 | 4 of 4 (L 128, 256, 512, 1024) |
| augmented_attention_pair_bias [unsupported] | bf16 16x48 L200 | bf16 | infer | 1.09 | 5.03 | 1 of 1 (L 200) |
| augmented_attention_pair_bias [unsupported] | bf16 16x48 L200 | bf16 | fwd+bwd | 1.18 | 3.95 | 1 of 1 (L 200) |
| BiasOnlyDiTBlock | shared pair | bf16 | fwd+bwd | 0.96 - 1.16 | 0.97 - 1.32 | 1 of 5 (L 1024) |
| BiasOnlyDiTBlock | shared pair | fp32 | infer | 1.00 - 1.42 | 1.00 - 1.45 | 1 of 5 (L 100) |
| BiasOnlyDiTBlock | shared pair | fp32 | fwd+bwd | 1.00 - 1.13 | 1.00 - 1.39 | 2 of 5 (L 100, 1024) |
| BiasOnlyDiTBlock | pair per sample | fp32 | infer | 0.99 - 1.41 | 1.00 - 1.40 | 2 of 5 (L 100, 1024) |
| BiasOnlyDiTBlock | pair per sample | fp32 | fwd+bwd | 0.99 - 1.08 | 1.00 - 1.08 | 2 of 5 (L 100, 1024) |
| AugmentedAttentionPairBias module | A2 D768 | bf16 | infer | 0.91 - 1.27 | 0.86 - 1.26 | 1 of 5 (L 100) |
| AugmentedAttentionPairBias module | A2 D768 | bf16 | fwd+bwd | 0.99 - 1.24 | 0.99 - 1.26 | 1 of 5 (L 100) |
| AugmentedAttentionPairBias module | A2 D768 | fp32 | infer | 1.24 - 1.39 | 1.26 - 1.40 | 5 of 5 (L 100, 128, 256, 512, 1024) |
| AugmentedAttentionPairBias module | A2 D768 | fp32 | fwd+bwd | 1.11 - 1.30 | 1.12 - 1.29 | 5 of 5 (L 100, 128, 256, 512, 1024) |
Fully within 1.05 now (every L, both modes): `ops.triangle_multiplicative_update` outgoing / incoming and `ops.bidirectional_triangle_multiplicative_update` (3.1 - 10.8x before), `ops.triangle_attention` bf16 with and without QK norm except
the bf16 L128 inference row at 1.07 (3.7 - 12.4x before; L1024: 0.73 / 0.68, the sm_90 core is faster than the Triton core), `ops.swiglu_ffn` and `ops.conditioned_transition` fp32 (0.6 - 1.0), `ops.conditioned_transition` bf16 token / atom
(1.07 - 1.36 before), `ops.layer_norm` bf16, `ops.rms_norm_modulation` D768 bf16 and fp32 (2.0 - 6.7x before) and fp32 D128, `ops.layer_norm_linear` bf16 inference from L512 (0.94 at L1024),
`BiasOnlyDiTBlock` L128 ... 512 (1.00, the same CUDA block), `AugmentedAttentionPairBias` module bf16 L128 ... 1024.

### Why the rest is slower (measured causes; nothing here is a regression of a CUDA path, each row is a call whose old path was one fused Triton kernel)

* **`ops.swiglu_ffn` bf16** (pair 1.05 - 1.39, token inference 1.24 - 1.30; was 1.06 - 2.28): no CUDA kernel serves the bare SwiGLU FFN with the hidden activation on chip. The fused CUDA Transition (`transition_b2b`) carries the LayerNorm and the residual and
  cannot be neutralised; the nearest kernel is `ct_inference`, whose expand + SwiGLU GEMM writes the hidden activation to HBM once (the Triton unit kept it in registers). Under 2x, so no new kernel.
* **`ops.rms_norm_modulation` D128 c128 bf16** (1.5 - 2.0, 10 - 14 us against 7 us; was 5 - 8.6): four launches (RMSNorm, weight concatenation, GEMM, `addcmul`) against one Triton kernel that does the 128 x 128 projection in-register. The RMS-adaLN CUDA kernels of
  the SWA block are bound to that block (fixed widths, silu on the conditioning). A one-launch row kernel would close it; it was not written (the largest row is 2.0x, only at L1024 inference).
* **`ops.layer_norm_linear` bf16** (inference L128 / L256 1.77 / 1.21, training 1.11 - 1.70; was 4.5 - 10.5): the CUDA kernels are persistent one-pass kernels with a TF32 projection and saved statistics (from the source, not profiled); at L128 the call is 8 us against 5 us, and the inference rows from L512 are at or under 1.0 (the `apb` kernel), so the remaining rows are small-L latency and the training backward.
* **`ops.gated_residual` / `ops.gated_linear`** (1.0 - 1.4 / 1.03 - 1.64; was 1.0 - 1.65 / 1.2 - 2.2): the Triton units were one elementwise (or gate + GEMM) pass; the CUDA row kernels of the conditioned transition and `addcmul` are one pass each but the gated
  linear still writes and re-reads the gated activation before the GEMM, and the backward of `addcmul` is two products (the Triton backward was one kernel).
* **`ops.augmented_attention_pair_bias`**: supported shapes (bf16 16 x 48 / 4 x 32, L % 128 == 0, B = 1) run the sm_90 core: faster than Triton from L512 (16 x 48), 1.2 - 1.6x slower at small L (the head-major to token-major copies of q, k, v and the scale pass,
  against Triton kernels that read the head-major layout directly; unchanged since the first measurement). Calls the core does not take: fp32 (1.55 - 3.6) has no CUDA core at the default matmul precision (the `attn_tf32` cores serve fp32 only
  under TF32, by the engine's rule that fp32 keeps full-fp32 math otherwise; the Triton fp32 kernel ran TF32 dots, so the comparison is not like for like) and SDPA's memory-efficient kernel is the fastest exact-fp32 form measured
  (job 41381: the math backend is 1.4 - 1.5x slower again at fp32); bf16 8 x 64 (0.99 - 2.07) has no CUDA core for 64-channel heads (cuDNN attention, inference; the memory-efficient kernel with a gradient: bias gradients are not supported by cuDNN); bf16 16 x 48 at L200 (1.09 / 1.18)
  is SDPA (a padded sm_90 core was measured and was slower, 2.6x, and removed).
* **`BiasOnlyDiTBlock`** L1024 shared pair (bf16 1.16, fp32 1.11; was 1.32 / 1.39): the pair LayerNorm + bias projection is now the CUDA pair-bias kernel; the rest of the gap is in the block's training fallback (PyTorch einsums where the Triton `core_t` ran; not isolated).
  fp32 L100 inference 1.4 (0.28 against 0.20 ms) is the fp32 statements leg of the block. Pair per sample fp32: 1.07 - 1.41, same legs.
* **`AugmentedAttentionPairBias` module** fp32 (1.11 - 1.39) and bf16 L100 (1.24 - 1.27): module paths, not facades; with the A/B switch the module runs the portable Triton family, without it the module's own CUDA / PyTorch legs. Not diagnosed further here.
* **`ops.layer_norm`** token D768 fp32 fwd+bwd (L256 / L512 / L1024: 1.11 / 1.11 / 1.32, 0.015 - 0.04 ms): the fp32 LayerNorm backward of the general CUDA kernel against the Triton one on a small tensor (not diagnosed).
* **`ops.conditioned_transition` atom fp32 fwd+bwd** L512 / L1024 (1.08 / 1.06): fp32 at the default matmul precision runs the exact statements (the CUDA tail needs TF32).

## Out of memory

`ops.triangle_attention` at L = 1024 (4 heads, bf16 and fp32, with and without QK norm; the PyTorch core built 16 GiB of fp32 scores per tensor): bf16 runs the sm_90 core (4.7 ms inference, 18.4 ms fwd+bwd, 0.73x the Triton core);
fp32 runs the chunked PyTorch core (92 ms inference, 370 ms fwd+bwd; the previous Triton path was the same PyTorch statements for fp32 and ran out of memory too, so the "old" column is the chunked core as well: 1.00).
Both are bit-reproducible (`tests/gpu/sm90/test_ops_cuda_wiring_gpu.py::test_triangle_attention_at_l1024_does_not_run_out_of_memory`, job 41377 / 41387 / 41393).

## Full table (infer and fwd+bwd, every case)

| op | variant | L | dtype | mode | new ms | old (Triton) ms | new/old | before (job 41372) | note |
|---|---|---|---|---|---|---|---|---|---|
| layer_norm | pair D128 | 128 | bf16 | infer | 0.004 | 0.004 | 1.00 | 1.00 | |
| layer_norm | pair D128 | 128 | bf16 | fwd+bwd | 0.012 | 0.015 | 0.83 | 0.87 | |
| layer_norm | pair D128 | 128 | fp32 | infer | 0.006 | 0.006 | 1.01 | 1.01 | |
| layer_norm | pair D128 | 128 | fp32 | fwd+bwd | 0.015 | 0.019 | 0.78 | 0.87 | |
| layer_norm | pair D128 | 256 | bf16 | infer | 0.010 | 0.009 | 1.12 | 1.09 | |
| layer_norm | pair D128 | 256 | bf16 | fwd+bwd | 0.037 | 0.042 | 0.88 | 0.87 | |
| layer_norm | pair D128 | 256 | fp32 | infer | 0.024 | 0.025 | 0.95 | 0.95 | |
| layer_norm | pair D128 | 256 | fp32 | fwd+bwd | 0.065 | 0.069 | 0.94 | 0.96 | |
| layer_norm | pair D128 | 512 | bf16 | infer | 0.049 | 0.049 | 1.01 | 1.00 | |
| layer_norm | pair D128 | 512 | bf16 | fwd+bwd | 0.124 | 0.126 | 0.98 | 0.98 | |
| layer_norm | pair D128 | 512 | fp32 | infer | 0.092 | 0.092 | 0.99 | 0.99 | |
| layer_norm | pair D128 | 512 | fp32 | fwd+bwd | 0.233 | 0.234 | 1.00 | 0.99 | |
| layer_norm | pair D128 | 1024 | bf16 | infer | 0.183 | 0.182 | 1.01 | 1.00 | |
| layer_norm | pair D128 | 1024 | bf16 | fwd+bwd | 0.461 | 0.462 | 1.00 | 1.01 | |
| layer_norm | pair D128 | 1024 | fp32 | infer | 0.356 | 0.359 | 0.99 | 0.99 | |
| layer_norm | pair D128 | 1024 | fp32 | fwd+bwd | 0.910 | 0.906 | 1.00 | 1.01 | |
| layer_norm | token D768 | 128 | bf16 | infer | 0.003 | 0.004 | 0.95 | 0.91 | |
| layer_norm | token D768 | 128 | bf16 | fwd+bwd | 0.010 | 0.012 | 0.80 | 0.79 | |
| layer_norm | token D768 | 128 | fp32 | infer | 0.003 | 0.005 | 0.64 | 0.65 | |
| layer_norm | token D768 | 128 | fp32 | fwd+bwd | 0.013 | 0.013 | 0.97 | 1.06 | |
| layer_norm | token D768 | 256 | bf16 | infer | 0.004 | 0.004 | 0.93 | 0.91 | |
| layer_norm | token D768 | 256 | bf16 | fwd+bwd | 0.011 | 0.014 | 0.77 | 0.75 | |
| layer_norm | token D768 | 256 | fp32 | infer | 0.004 | 0.005 | 0.72 | 0.71 | |
| layer_norm | token D768 | 256 | fp32 | fwd+bwd | 0.015 | 0.013 | 1.11 | 1.10 | |
| layer_norm | token D768 | 512 | bf16 | infer | 0.005 | 0.004 | 1.09 | 1.11 | |
| layer_norm | token D768 | 512 | bf16 | fwd+bwd | 0.013 | 0.018 | 0.74 | 0.75 | |
| layer_norm | token D768 | 512 | fp32 | infer | 0.005 | 0.006 | 0.83 | 0.82 | |
| layer_norm | token D768 | 512 | fp32 | fwd+bwd | 0.019 | 0.017 | 1.11 | 1.09 | |
| layer_norm | token D768 | 1024 | bf16 | infer | 0.005 | 0.006 | 0.96 | 0.98 | |
| layer_norm | token D768 | 1024 | bf16 | fwd+bwd | 0.016 | 0.022 | 0.70 | 0.71 | |
| layer_norm | token D768 | 1024 | fp32 | infer | 0.008 | 0.008 | 0.98 | 0.96 | |
| layer_norm | token D768 | 1024 | fp32 | fwd+bwd | 0.041 | 0.031 | 1.32 | 1.32 | |
| swiglu_ffn | pair D128 h512 | 128 | bf16 | infer | 0.034 | 0.026 | 1.28 | 1.96 | |
| swiglu_ffn | pair D128 h512 | 128 | bf16 | fwd+bwd | 0.122 | 0.116 | 1.05 | 1.46 | |
| swiglu_ffn | pair D128 h512 | 128 | fp32 | infer | 0.222 | 0.244 | 0.91 | 0.93 | |
| swiglu_ffn | pair D128 h512 | 128 | fp32 | fwd+bwd | 0.586 | 0.604 | 0.97 | 1.09 | |
| swiglu_ffn | pair D128 h512 | 256 | bf16 | infer | 0.127 | 0.098 | 1.29 | 2.21 | |
| swiglu_ffn | pair D128 h512 | 256 | bf16 | fwd+bwd | 0.480 | 0.400 | 1.20 | 1.64 | |
| swiglu_ffn | pair D128 h512 | 256 | fp32 | infer | 0.819 | 0.836 | 0.98 | 1.02 | |
| swiglu_ffn | pair D128 h512 | 256 | fp32 | fwd+bwd | 2.138 | 2.101 | 1.02 | 1.14 | |
| swiglu_ffn | pair D128 h512 | 512 | bf16 | infer | 0.501 | 0.361 | 1.39 | 2.24 | |
| swiglu_ffn | pair D128 h512 | 512 | bf16 | fwd+bwd | 1.776 | 1.464 | 1.21 | 1.63 | |
| swiglu_ffn | pair D128 h512 | 512 | fp32 | infer | 2.905 | 3.220 | 0.90 | 1.02 | |
| swiglu_ffn | pair D128 h512 | 512 | fp32 | fwd+bwd | 8.072 | 8.099 | 1.00 | 1.15 | |
| swiglu_ffn | pair D128 h512 | 1024 | bf16 | infer | 1.976 | 1.437 | 1.38 | 2.28 | |
| swiglu_ffn | pair D128 h512 | 1024 | bf16 | fwd+bwd | 6.951 | 5.850 | 1.19 | 1.63 | |
| swiglu_ffn | pair D128 h512 | 1024 | fp32 | infer | 11.425 | 12.686 | 0.90 | 1.01 | |
| swiglu_ffn | pair D128 h512 | 1024 | fp32 | fwd+bwd | 32.533 | 32.824 | 0.99 | 1.17 | |
| swiglu_ffn | token D768 h1536 | 128 | bf16 | infer | 0.023 | 0.018 | 1.27 | 1.13 | |
| swiglu_ffn | token D768 h1536 | 128 | bf16 | fwd+bwd | 0.047 | 0.052 | 0.91 | 1.16 | |
| swiglu_ffn | token D768 h1536 | 128 | fp32 | infer | 0.115 | 0.190 | 0.61 | 0.62 | |
| swiglu_ffn | token D768 h1536 | 128 | fp32 | fwd+bwd | 0.328 | 0.423 | 0.77 | 0.89 | |
| swiglu_ffn | token D768 h1536 | 256 | bf16 | infer | 0.028 | 0.022 | 1.30 | 1.18 | |
| swiglu_ffn | token D768 h1536 | 256 | bf16 | fwd+bwd | 0.068 | 0.069 | 0.99 | 1.14 | |
| swiglu_ffn | token D768 h1536 | 256 | fp32 | infer | 0.204 | 0.286 | 0.71 | 0.71 | |
| swiglu_ffn | token D768 h1536 | 256 | fp32 | fwd+bwd | 0.597 | 0.699 | 0.85 | 0.89 | |
| swiglu_ffn | token D768 h1536 | 512 | bf16 | infer | 0.040 | 0.033 | 1.24 | 1.23 | |
| swiglu_ffn | token D768 h1536 | 512 | bf16 | fwd+bwd | 0.104 | 0.114 | 0.92 | 1.07 | |
| swiglu_ffn | token D768 h1536 | 512 | fp32 | infer | 0.368 | 0.548 | 0.67 | 0.72 | |
| swiglu_ffn | token D768 h1536 | 512 | fp32 | fwd+bwd | 1.077 | 1.290 | 0.83 | 0.90 | |
| swiglu_ffn | token D768 h1536 | 1024 | bf16 | infer | 0.073 | 0.057 | 1.28 | 1.18 | |
| swiglu_ffn | token D768 h1536 | 1024 | bf16 | fwd+bwd | 0.185 | 0.202 | 0.92 | 1.06 | |
| swiglu_ffn | token D768 h1536 | 1024 | fp32 | infer | 0.651 | 0.940 | 0.69 | 0.78 | |
| swiglu_ffn | token D768 h1536 | 1024 | fp32 | fwd+bwd | 1.961 | 2.304 | 0.85 | 0.93 | |
| conditioned_transition | token D768 c384 n2 | 128 | bf16 | infer | 0.021 | 0.026 | 0.81 | 1.10 | |
| conditioned_transition | token D768 c384 n2 | 128 | bf16 | fwd+bwd | 0.075 | 0.086 | 0.86 | 1.08 | |
| conditioned_transition | token D768 c384 n2 | 128 | fp32 | infer | 0.142 | 0.213 | 0.67 | 0.67 | |
| conditioned_transition | token D768 c384 n2 | 128 | fp32 | fwd+bwd | 0.447 | 0.496 | 0.90 | 0.91 | |
| conditioned_transition | token D768 c384 n2 | 256 | bf16 | infer | 0.027 | 0.031 | 0.88 | 1.14 | |
| conditioned_transition | token D768 c384 n2 | 256 | bf16 | fwd+bwd | 0.100 | 0.109 | 0.91 | 1.10 | |
| conditioned_transition | token D768 c384 n2 | 256 | fp32 | infer | 0.235 | 0.317 | 0.74 | 0.74 | |
| conditioned_transition | token D768 c384 n2 | 256 | fp32 | fwd+bwd | 0.722 | 0.790 | 0.91 | 0.91 | |
| conditioned_transition | token D768 c384 n2 | 512 | bf16 | infer | 0.038 | 0.045 | 0.85 | 1.20 | |
| conditioned_transition | token D768 c384 n2 | 512 | bf16 | fwd+bwd | 0.144 | 0.161 | 0.89 | 1.07 | |
| conditioned_transition | token D768 c384 n2 | 512 | fp32 | infer | 0.440 | 0.594 | 0.74 | 0.74 | |
| conditioned_transition | token D768 c384 n2 | 512 | fp32 | fwd+bwd | 1.310 | 1.439 | 0.91 | 0.91 | |
| conditioned_transition | token D768 c384 n2 | 1024 | bf16 | infer | 0.069 | 0.076 | 0.92 | 1.13 | |
| conditioned_transition | token D768 c384 n2 | 1024 | bf16 | fwd+bwd | 0.235 | 0.263 | 0.89 | 1.07 | |
| conditioned_transition | token D768 c384 n2 | 1024 | fp32 | infer | 0.817 | 1.031 | 0.79 | 0.79 | |
| conditioned_transition | token D768 c384 n2 | 1024 | fp32 | fwd+bwd | 2.405 | 2.564 | 0.94 | 0.94 | |
| conditioned_transition | atom D128 c128 n2 | 128 | bf16 | infer | 0.010 | 0.014 | 0.69 | 1.29 | |
| conditioned_transition | atom D128 c128 n2 | 128 | bf16 | fwd+bwd | 0.039 | 0.050 | 0.79 | 1.20 | |
| conditioned_transition | atom D128 c128 n2 | 128 | fp32 | infer | 0.029 | 0.039 | 0.75 | 0.77 | |
| conditioned_transition | atom D128 c128 n2 | 128 | fp32 | fwd+bwd | 0.100 | 0.100 | 1.00 | 1.00 | |
| conditioned_transition | atom D128 c128 n2 | 256 | bf16 | infer | 0.010 | 0.015 | 0.66 | 1.31 | |
| conditioned_transition | atom D128 c128 n2 | 256 | bf16 | fwd+bwd | 0.043 | 0.054 | 0.80 | 1.23 | |
| conditioned_transition | atom D128 c128 n2 | 256 | fp32 | infer | 0.033 | 0.040 | 0.83 | 0.83 | |
| conditioned_transition | atom D128 c128 n2 | 256 | fp32 | fwd+bwd | 0.109 | 0.107 | 1.02 | 1.02 | |
| conditioned_transition | atom D128 c128 n2 | 512 | bf16 | infer | 0.010 | 0.016 | 0.65 | 1.30 | |
| conditioned_transition | atom D128 c128 n2 | 512 | bf16 | fwd+bwd | 0.051 | 0.061 | 0.83 | 1.24 | |
| conditioned_transition | atom D128 c128 n2 | 512 | fp32 | infer | 0.038 | 0.041 | 0.92 | 0.93 | |
| conditioned_transition | atom D128 c128 n2 | 512 | fp32 | fwd+bwd | 0.134 | 0.124 | 1.08 | 1.09 | |
| conditioned_transition | atom D128 c128 n2 | 1024 | bf16 | infer | 0.011 | 0.017 | 0.63 | 1.36 | |
| conditioned_transition | atom D128 c128 n2 | 1024 | bf16 | fwd+bwd | 0.059 | 0.073 | 0.81 | 1.20 | |
| conditioned_transition | atom D128 c128 n2 | 1024 | fp32 | infer | 0.052 | 0.056 | 0.92 | 0.93 | |
| conditioned_transition | atom D128 c128 n2 | 1024 | fp32 | fwd+bwd | 0.191 | 0.180 | 1.06 | 1.06 | |
| rms_norm_modulation | D768 c384 | 128 | bf16 | infer | 0.013 | 0.046 | 0.29 | 1.85 | |
| rms_norm_modulation | D768 c384 | 128 | bf16 | fwd+bwd | 0.051 | 0.120 | 0.43 | 2.02 | |
| rms_norm_modulation | D768 c384 | 128 | fp32 | infer | 0.043 | 0.525 | 0.08 | 0.12 | |
| rms_norm_modulation | D768 c384 | 128 | fp32 | fwd+bwd | 0.139 | 1.099 | 0.13 | 0.17 | |
| rms_norm_modulation | D768 c384 | 256 | bf16 | infer | 0.016 | 0.046 | 0.34 | 2.71 | |
| rms_norm_modulation | D768 c384 | 256 | bf16 | fwd+bwd | 0.070 | 0.123 | 0.57 | 2.63 | |
| rms_norm_modulation | D768 c384 | 256 | fp32 | infer | 0.066 | 0.525 | 0.13 | 0.17 | |
| rms_norm_modulation | D768 c384 | 256 | fp32 | fwd+bwd | 0.232 | 1.161 | 0.20 | 0.23 | |
| rms_norm_modulation | D768 c384 | 512 | bf16 | infer | 0.022 | 0.046 | 0.47 | 4.10 | |
| rms_norm_modulation | D768 c384 | 512 | bf16 | fwd+bwd | 0.103 | 0.145 | 0.71 | 3.71 | |
| rms_norm_modulation | D768 c384 | 512 | fp32 | infer | 0.118 | 0.532 | 0.22 | 0.29 | |
| rms_norm_modulation | D768 c384 | 512 | fp32 | fwd+bwd | 0.402 | 1.296 | 0.31 | 0.37 | |
| rms_norm_modulation | D768 c384 | 1024 | bf16 | infer | 0.038 | 0.047 | 0.82 | 6.65 | |
| rms_norm_modulation | D768 c384 | 1024 | bf16 | fwd+bwd | 0.178 | 0.185 | 0.96 | 4.60 | |
| rms_norm_modulation | D768 c384 | 1024 | fp32 | infer | 0.235 | 1.004 | 0.23 | 0.26 | |
| rms_norm_modulation | D768 c384 | 1024 | fp32 | fwd+bwd | 0.786 | 2.058 | 0.38 | 0.37 | |
| rms_norm_modulation | D128 c128 | 128 | bf16 | infer | 0.010 | 0.007 | 1.51 | 6.91 | |
| rms_norm_modulation | D128 c128 | 128 | bf16 | fwd+bwd | 0.039 | 0.023 | 1.67 | 5.11 | |
| rms_norm_modulation | D128 c128 | 128 | fp32 | infer | 0.013 | 0.032 | 0.39 | 0.85 | |
| rms_norm_modulation | D128 c128 | 128 | fp32 | fwd+bwd | 0.049 | 0.072 | 0.68 | 1.17 | |
| rms_norm_modulation | D128 c128 | 256 | bf16 | infer | 0.011 | 0.007 | 1.60 | 7.27 | |
| rms_norm_modulation | D128 c128 | 256 | bf16 | fwd+bwd | 0.041 | 0.024 | 1.72 | 5.21 | |
| rms_norm_modulation | D128 c128 | 256 | fp32 | infer | 0.014 | 0.032 | 0.45 | 0.87 | |
| rms_norm_modulation | D128 c128 | 256 | fp32 | fwd+bwd | 0.055 | 0.076 | 0.72 | 1.17 | |
| rms_norm_modulation | D128 c128 | 512 | bf16 | infer | 0.012 | 0.007 | 1.81 | 7.70 | |
| rms_norm_modulation | D128 c128 | 512 | bf16 | fwd+bwd | 0.047 | 0.026 | 1.79 | 5.14 | |
| rms_norm_modulation | D128 c128 | 512 | fp32 | infer | 0.019 | 0.032 | 0.58 | 1.03 | |
| rms_norm_modulation | D128 c128 | 512 | fp32 | fwd+bwd | 0.069 | 0.082 | 0.84 | 1.22 | |
| rms_norm_modulation | D128 c128 | 1024 | bf16 | infer | 0.014 | 0.007 | 2.00 | 8.63 | |
| rms_norm_modulation | D128 c128 | 1024 | bf16 | fwd+bwd | 0.055 | 0.028 | 1.95 | 5.97 | |
| rms_norm_modulation | D128 c128 | 1024 | fp32 | infer | 0.025 | 0.058 | 0.43 | 0.65 | |
| rms_norm_modulation | D128 c128 | 1024 | fp32 | fwd+bwd | 0.106 | 0.121 | 0.88 | 1.00 | |
| layer_norm_linear | pair D128 ->16 | 128 | bf16 | infer | 0.008 | 0.005 | 1.77 | 5.64 | |
| layer_norm_linear | pair D128 ->16 | 128 | bf16 | fwd+bwd | 0.032 | 0.019 | 1.70 | 5.28 | |
| layer_norm_linear | pair D128 ->16 | 128 | fp32 | infer | 0.008 | 0.017 | 0.49 | 1.94 | |
| layer_norm_linear | pair D128 ->16 | 128 | fp32 | fwd+bwd | 0.035 | 0.041 | 0.85 | 2.76 | |
| layer_norm_linear | pair D128 ->16 | 256 | bf16 | infer | 0.011 | 0.009 | 1.21 | 8.69 | |
| layer_norm_linear | pair D128 ->16 | 256 | bf16 | fwd+bwd | 0.065 | 0.049 | 1.33 | 8.54 | |
| layer_norm_linear | pair D128 ->16 | 256 | fp32 | infer | 0.019 | 0.065 | 0.29 | 1.91 | |
| layer_norm_linear | pair D128 ->16 | 256 | fp32 | fwd+bwd | 0.080 | 0.158 | 0.51 | 3.20 | |
| layer_norm_linear | pair D128 ->16 | 512 | bf16 | infer | 0.034 | 0.033 | 1.02 | 9.40 | |
| layer_norm_linear | pair D128 ->16 | 512 | bf16 | fwd+bwd | 0.181 | 0.160 | 1.14 | 4.48 | |
| layer_norm_linear | pair D128 ->16 | 512 | fp32 | infer | 0.066 | 0.215 | 0.31 | 2.18 | |
| layer_norm_linear | pair D128 ->16 | 512 | fp32 | fwd+bwd | 0.211 | 0.518 | 0.41 | 1.92 | |
| layer_norm_linear | pair D128 ->16 | 1024 | bf16 | infer | 0.109 | 0.116 | 0.94 | 10.49 | |
| layer_norm_linear | pair D128 ->16 | 1024 | bf16 | fwd+bwd | 0.645 | 0.583 | 1.11 | 4.56 | |
| layer_norm_linear | pair D128 ->16 | 1024 | fp32 | infer | 0.240 | 0.801 | 0.30 | 2.27 | |
| layer_norm_linear | pair D128 ->16 | 1024 | fp32 | fwd+bwd | 0.751 | 1.953 | 0.38 | 1.93 | |
| gated_residual | pair D128 | 128 | bf16 | infer | 0.004 | 0.007 | 0.57 | 0.97 | |
| gated_residual | pair D128 | 128 | bf16 | fwd+bwd | 0.011 | 0.013 | 0.83 | 1.03 | |
| gated_residual | pair D128 | 128 | fp32 | infer | 0.007 | 0.008 | 0.99 | 1.68 | |
| gated_residual | pair D128 | 128 | fp32 | fwd+bwd | 0.028 | 0.027 | 1.03 | 1.10 | |
| gated_residual | pair D128 | 256 | bf16 | infer | 0.024 | 0.025 | 0.99 | 1.39 | |
| gated_residual | pair D128 | 256 | bf16 | fwd+bwd | 0.062 | 0.054 | 1.15 | 1.32 | |
| gated_residual | pair D128 | 256 | fp32 | infer | 0.046 | 0.046 | 1.00 | 1.55 | |
| gated_residual | pair D128 | 256 | fp32 | fwd+bwd | 0.116 | 0.103 | 1.13 | 1.37 | |
| gated_residual | pair D128 | 512 | bf16 | infer | 0.088 | 0.089 | 0.99 | 1.51 | |
| gated_residual | pair D128 | 512 | bf16 | fwd+bwd | 0.223 | 0.200 | 1.11 | 1.34 | |
| gated_residual | pair D128 | 512 | fp32 | infer | 0.173 | 0.173 | 1.00 | 1.54 | |
| gated_residual | pair D128 | 512 | fp32 | fwd+bwd | 0.438 | 0.393 | 1.11 | 1.35 | |
| gated_residual | pair D128 | 1024 | bf16 | infer | 0.343 | 0.348 | 0.99 | 1.50 | |
| gated_residual | pair D128 | 1024 | bf16 | fwd+bwd | 0.865 | 0.785 | 1.10 | 1.33 | |
| gated_residual | pair D128 | 1024 | fp32 | infer | 0.683 | 0.682 | 1.00 | 1.53 | |
| gated_residual | pair D128 | 1024 | fp32 | fwd+bwd | 1.725 | 1.556 | 1.11 | 1.34 | |
| gated_residual | token D768 | 128 | bf16 | infer | 0.003 | 0.003 | 0.84 | 1.34 | |
| gated_residual | token D768 | 128 | bf16 | fwd+bwd | 0.006 | 0.005 | 1.26 | 1.56 | |
| gated_residual | token D768 | 128 | fp32 | infer | 0.003 | 0.003 | 1.00 | 1.50 | |
| gated_residual | token D768 | 128 | fp32 | fwd+bwd | 0.007 | 0.005 | 1.38 | 1.65 | |
| gated_residual | token D768 | 256 | bf16 | infer | 0.003 | 0.004 | 0.76 | 1.25 | |
| gated_residual | token D768 | 256 | bf16 | fwd+bwd | 0.007 | 0.007 | 1.01 | 1.30 | |
| gated_residual | token D768 | 256 | fp32 | infer | 0.004 | 0.004 | 0.89 | 1.38 | |
| gated_residual | token D768 | 256 | fp32 | fwd+bwd | 0.009 | 0.007 | 1.21 | 1.44 | |
| gated_residual | token D768 | 512 | bf16 | infer | 0.004 | 0.006 | 0.63 | 1.01 | |
| gated_residual | token D768 | 512 | bf16 | fwd+bwd | 0.009 | 0.011 | 0.83 | 1.01 | |
| gated_residual | token D768 | 512 | fp32 | infer | 0.006 | 0.006 | 0.94 | 1.43 | |
| gated_residual | token D768 | 512 | fp32 | fwd+bwd | 0.015 | 0.017 | 0.93 | 1.15 | |
| gated_residual | token D768 | 1024 | bf16 | infer | 0.005 | 0.010 | 0.51 | 1.05 | |
| gated_residual | token D768 | 1024 | bf16 | fwd+bwd | 0.020 | 0.019 | 1.04 | 1.22 | |
| gated_residual | token D768 | 1024 | fp32 | infer | 0.018 | 0.018 | 1.00 | 1.38 | |
| gated_residual | token D768 | 1024 | fp32 | fwd+bwd | 0.047 | 0.042 | 1.11 | 1.27 | |
| gated_linear | pair D128 | 128 | bf16 | infer | 0.009 | 0.008 | 1.14 | 1.54 | |
| gated_linear | pair D128 | 128 | bf16 | fwd+bwd | 0.030 | 0.027 | 1.10 | 1.31 | |
| gated_linear | pair D128 | 256 | bf16 | infer | 0.032 | 0.025 | 1.29 | 1.54 | |
| gated_linear | pair D128 | 256 | bf16 | fwd+bwd | 0.097 | 0.089 | 1.09 | 1.32 | |
| gated_linear | pair D128 | 512 | bf16 | infer | 0.122 | 0.077 | 1.58 | 2.17 | |
| gated_linear | pair D128 | 512 | bf16 | fwd+bwd | 0.347 | 0.298 | 1.16 | 1.60 | |
| gated_linear | pair D128 | 1024 | bf16 | infer | 0.470 | 0.287 | 1.64 | 2.23 | |
| gated_linear | pair D128 | 1024 | bf16 | fwd+bwd | 1.302 | 1.090 | 1.19 | 1.60 | |
| gated_linear | token D768 | 128 | bf16 | infer | 0.007 | 0.007 | 1.05 | 1.34 | |
| gated_linear | token D768 | 128 | bf16 | fwd+bwd | 0.019 | 0.018 | 1.07 | 1.28 | |
| gated_linear | token D768 | 256 | bf16 | infer | 0.008 | 0.008 | 1.04 | 1.35 | |
| gated_linear | token D768 | 256 | bf16 | fwd+bwd | 0.022 | 0.020 | 1.08 | 1.22 | |
| gated_linear | token D768 | 512 | bf16 | infer | 0.010 | 0.010 | 1.07 | 1.33 | |
| gated_linear | token D768 | 512 | bf16 | fwd+bwd | 0.031 | 0.027 | 1.13 | 1.22 | |
| gated_linear | token D768 | 1024 | bf16 | infer | 0.014 | 0.014 | 1.03 | 1.35 | |
| gated_linear | token D768 | 1024 | bf16 | fwd+bwd | 0.056 | 0.048 | 1.17 | 1.28 | |
| triangle_attention | D128 H4 | 128 | bf16 | infer | 0.081 | 0.076 | 1.07 | 5.03 | |
| triangle_attention | D128 H4 | 128 | bf16 | fwd+bwd | 0.302 | 0.303 | 1.00 | 3.67 | |
| triangle_attention | D128 H4 | 128 | fp32 | infer | 0.397 | 0.397 | 1.00 | 1.00 | |
| triangle_attention | D128 H4 | 128 | fp32 | fwd+bwd | 1.239 | 1.239 | 1.00 | 1.00 | |
| triangle_attention | D128 H4 | 256 | bf16 | infer | 0.236 | 0.251 | 0.94 | 8.55 | |
| triangle_attention | D128 H4 | 256 | bf16 | fwd+bwd | 1.018 | 1.145 | 0.89 | 5.54 | |
| triangle_attention | D128 H4 | 256 | fp32 | infer | 2.051 | 2.052 | 1.00 | 1.00 | |
| triangle_attention | D128 H4 | 256 | fp32 | fwd+bwd | 6.444 | 6.445 | 1.00 | 1.00 | |
| triangle_attention | D128 H4 | 512 | bf16 | infer | 1.016 | 1.197 | 0.85 | 12.38 | |
| triangle_attention | D128 H4 | 512 | bf16 | fwd+bwd | 3.592 | 4.497 | 0.80 | 9.24 | |
| triangle_attention | D128 H4 | 512 | fp32 | infer | 12.983 | 12.982 | 1.00 | 1.00 | |
| triangle_attention | D128 H4 | 512 | fp32 | fwd+bwd | 39.049 | 39.046 | 1.00 | 1.00 | |
| triangle_attention | D128 H4 | 1024 | bf16 | infer | 4.683 | 6.372 | 0.73 | - | |
| triangle_attention | D128 H4 | 1024 | bf16 | fwd+bwd | 18.359 | 25.316 | 0.73 | - | |
| triangle_attention | D128 H4 | 1024 | fp32 | infer | 91.943 | 91.945 | 1.00 | - | |
| triangle_attention | D128 H4 | 1024 | fp32 | fwd+bwd | 370.258 | 370.376 | 1.00 | - | |
| triangle_attention | D128 H4 +QKnorm | 128 | bf16 | infer | 0.092 | 0.094 | 0.98 | 4.75 | |
| triangle_attention | D128 H4 +QKnorm | 128 | bf16 | fwd+bwd | 0.336 | 0.366 | 0.92 | 4.36 | |
| triangle_attention | D128 H4 +QKnorm | 128 | fp32 | infer | 0.406 | 0.491 | 0.83 | 1.00 | |
| triangle_attention | D128 H4 +QKnorm | 128 | fp32 | fwd+bwd | 1.276 | 1.727 | 0.74 | 1.00 | |
| triangle_attention | D128 H4 +QKnorm | 256 | bf16 | infer | 0.269 | 0.313 | 0.86 | 7.84 | |
| triangle_attention | D128 H4 +QKnorm | 256 | bf16 | fwd+bwd | 1.102 | 1.319 | 0.84 | 5.34 | |
| triangle_attention | D128 H4 +QKnorm | 256 | fp32 | infer | 2.097 | 2.453 | 0.85 | 1.00 | |
| triangle_attention | D128 H4 +QKnorm | 256 | fp32 | fwd+bwd | 6.589 | 7.330 | 0.90 | 1.00 | |
| triangle_attention | D128 H4 +QKnorm | 512 | bf16 | infer | 1.114 | 1.446 | 0.77 | 10.87 | |
| triangle_attention | D128 H4 +QKnorm | 512 | bf16 | fwd+bwd | 3.824 | 5.173 | 0.74 | 8.53 | |
| triangle_attention | D128 H4 +QKnorm | 512 | fp32 | infer | 13.173 | 14.548 | 0.91 | 1.00 | |
| triangle_attention | D128 H4 +QKnorm | 512 | fp32 | fwd+bwd | 39.532 | 42.415 | 0.93 | 1.00 | |
| triangle_attention | D128 H4 +QKnorm | 1024 | bf16 | infer | 5.057 | 7.404 | 0.68 | - | |
| triangle_attention | D128 H4 +QKnorm | 1024 | bf16 | fwd+bwd | 19.295 | 28.151 | 0.69 | - | |
| triangle_attention | D128 H4 +QKnorm | 1024 | fp32 | infer | 92.432 | 98.123 | 0.94 | - | |
| triangle_attention | D128 H4 +QKnorm | 1024 | fp32 | fwd+bwd | 371.810 | 383.260 | 0.97 | - | |
| triangle_multiplicative_update | outgoing D128 | 128 | bf16 | infer | 0.024 | 0.054 | 0.43 | 3.12 | |
| triangle_multiplicative_update | outgoing D128 | 128 | bf16 | fwd+bwd | 0.160 | 0.193 | 0.83 | 3.58 | |
| triangle_multiplicative_update | outgoing D128 | 256 | bf16 | infer | 0.069 | 0.137 | 0.51 | 4.17 | |
| triangle_multiplicative_update | outgoing D128 | 256 | bf16 | fwd+bwd | 0.391 | 0.523 | 0.75 | 4.75 | |
| triangle_multiplicative_update | outgoing D128 | 512 | bf16 | infer | 0.251 | 0.455 | 0.55 | 5.49 | |
| triangle_multiplicative_update | outgoing D128 | 512 | bf16 | fwd+bwd | 1.260 | 1.733 | 0.73 | 5.41 | |
| triangle_multiplicative_update | outgoing D128 | 1024 | bf16 | infer | 1.107 | 1.831 | 0.60 | 8.87 | |
| triangle_multiplicative_update | outgoing D128 | 1024 | bf16 | fwd+bwd | 5.106 | 6.874 | 0.74 | 10.80 | |
| triangle_multiplicative_update | incoming D128 | 128 | bf16 | infer | 0.023 | 0.054 | 0.42 | 3.24 | |
| triangle_multiplicative_update | incoming D128 | 128 | bf16 | fwd+bwd | 0.160 | 0.192 | 0.83 | 3.60 | |
| triangle_multiplicative_update | incoming D128 | 256 | bf16 | infer | 0.069 | 0.138 | 0.50 | 4.17 | |
| triangle_multiplicative_update | incoming D128 | 256 | bf16 | fwd+bwd | 0.391 | 0.524 | 0.75 | 4.74 | |
| triangle_multiplicative_update | incoming D128 | 512 | bf16 | infer | 0.253 | 0.457 | 0.55 | 5.47 | |
| triangle_multiplicative_update | incoming D128 | 512 | bf16 | fwd+bwd | 1.268 | 1.734 | 0.73 | 5.42 | |
| triangle_multiplicative_update | incoming D128 | 1024 | bf16 | infer | 1.108 | 1.826 | 0.61 | 8.84 | |
| triangle_multiplicative_update | incoming D128 | 1024 | bf16 | fwd+bwd | 5.119 | 6.854 | 0.75 | 10.76 | |
| bidirectional_triangle_multiplicative_update | D128 | 128 | bf16 | infer | 0.037 | 0.082 | 0.45 | 3.03 | |
| bidirectional_triangle_multiplicative_update | D128 | 128 | bf16 | fwd+bwd | 0.190 | 0.253 | 0.75 | 3.34 | |
| bidirectional_triangle_multiplicative_update | D128 | 256 | bf16 | infer | 0.118 | 0.221 | 0.53 | 4.17 | |
| bidirectional_triangle_multiplicative_update | D128 | 256 | bf16 | fwd+bwd | 0.480 | 0.798 | 0.60 | 3.99 | |
| bidirectional_triangle_multiplicative_update | D128 | 512 | bf16 | infer | 0.430 | 0.753 | 0.57 | 5.31 | |
| bidirectional_triangle_multiplicative_update | D128 | 512 | bf16 | fwd+bwd | 1.709 | 2.800 | 0.61 | 4.24 | |
| bidirectional_triangle_multiplicative_update | D128 | 1024 | bf16 | infer | 1.906 | 3.170 | 0.60 | 8.93 | |
| bidirectional_triangle_multiplicative_update | D128 | 1024 | bf16 | fwd+bwd | 7.708 | 12.011 | 0.64 | 7.11 | |
| augmented_attention_pair_bias [supported] | bf16 16x48 B1 | 128 | bf16 | infer | 0.018 | 0.014 | 1.34 | 1.35 | |
| augmented_attention_pair_bias [supported] | bf16 16x48 B1 | 128 | bf16 | fwd+bwd | 0.050 | 0.034 | 1.48 | 1.52 | |
| augmented_attention_pair_bias [supported] | bf16 16x48 B1 | 256 | bf16 | infer | 0.021 | 0.020 | 1.07 | 1.29 | |
| augmented_attention_pair_bias [supported] | bf16 16x48 B1 | 256 | bf16 | fwd+bwd | 0.066 | 0.063 | 1.05 | 1.13 | |
| augmented_attention_pair_bias [supported] | bf16 16x48 B1 | 512 | bf16 | infer | 0.029 | 0.027 | 1.08 | 1.07 | |
| augmented_attention_pair_bias [supported] | bf16 16x48 B1 | 512 | bf16 | fwd+bwd | 0.118 | 0.133 | 0.89 | 0.88 | |
| augmented_attention_pair_bias [supported] | bf16 16x48 B1 | 1024 | bf16 | infer | 0.060 | 0.073 | 0.83 | 0.84 | |
| augmented_attention_pair_bias [supported] | bf16 16x48 B1 | 1024 | bf16 | fwd+bwd | 0.353 | 0.454 | 0.78 | 0.77 | |
| augmented_attention_pair_bias [supported] | bf16 4x32 B1 | 128 | bf16 | infer | 0.017 | 0.012 | 1.40 | 1.41 | |
| augmented_attention_pair_bias [supported] | bf16 4x32 B1 | 128 | bf16 | fwd+bwd | 0.045 | 0.028 | 1.58 | 1.62 | |
| augmented_attention_pair_bias [supported] | bf16 4x32 B1 | 256 | bf16 | infer | 0.019 | 0.014 | 1.38 | 1.39 | |
| augmented_attention_pair_bias [supported] | bf16 4x32 B1 | 256 | bf16 | fwd+bwd | 0.055 | 0.036 | 1.55 | 1.56 | |
| augmented_attention_pair_bias [supported] | bf16 4x32 B1 | 512 | bf16 | infer | 0.024 | 0.019 | 1.29 | 1.31 | |
| augmented_attention_pair_bias [supported] | bf16 4x32 B1 | 512 | bf16 | fwd+bwd | 0.075 | 0.056 | 1.33 | 1.32 | |
| augmented_attention_pair_bias [supported] | bf16 4x32 B1 | 1024 | bf16 | infer | 0.034 | 0.028 | 1.19 | 1.24 | |
| augmented_attention_pair_bias [supported] | bf16 4x32 B1 | 1024 | bf16 | fwd+bwd | 0.133 | 0.138 | 0.96 | 1.00 | |
| augmented_attention_pair_bias [unsupported] | fp32 16x48 | 128 | fp32 | infer | 0.023 | 0.015 | 1.55 | 3.09 | |
| augmented_attention_pair_bias [unsupported] | fp32 16x48 | 128 | fp32 | fwd+bwd | 0.074 | 0.041 | 1.80 | 2.47 | |
| augmented_attention_pair_bias [unsupported] | fp32 16x48 | 256 | fp32 | infer | 0.043 | 0.021 | 2.04 | 3.97 | |
| augmented_attention_pair_bias [unsupported] | fp32 16x48 | 256 | fp32 | fwd+bwd | 0.151 | 0.078 | 1.93 | 2.75 | |
| augmented_attention_pair_bias [unsupported] | fp32 16x48 | 512 | fp32 | infer | 0.132 | 0.047 | 2.82 | 6.08 | |
| augmented_attention_pair_bias [unsupported] | fp32 16x48 | 512 | fp32 | fwd+bwd | 0.390 | 0.191 | 2.04 | 3.98 | |
| augmented_attention_pair_bias [unsupported] | fp32 16x48 | 1024 | fp32 | infer | 0.534 | 0.148 | 3.60 | 7.55 | |
| augmented_attention_pair_bias [unsupported] | fp32 16x48 | 1024 | fp32 | fwd+bwd | 1.502 | 0.671 | 2.24 | 4.21 | |
| augmented_attention_pair_bias [unsupported] | bf16 8x64 | 128 | bf16 | infer | 0.013 | 0.013 | 0.99 | 3.94 | |
| augmented_attention_pair_bias [unsupported] | bf16 8x64 | 128 | bf16 | fwd+bwd | 0.043 | 0.031 | 1.39 | 3.74 | |
| augmented_attention_pair_bias [unsupported] | bf16 8x64 | 256 | bf16 | infer | 0.017 | 0.015 | 1.14 | 4.78 | |
| augmented_attention_pair_bias [unsupported] | bf16 8x64 | 256 | bf16 | fwd+bwd | 0.070 | 0.042 | 1.66 | 4.48 | |
| augmented_attention_pair_bias [unsupported] | bf16 8x64 | 512 | bf16 | infer | 0.030 | 0.021 | 1.44 | 8.68 | |
| augmented_attention_pair_bias [unsupported] | bf16 8x64 | 512 | bf16 | fwd+bwd | 0.142 | 0.088 | 1.61 | 5.17 | |
| augmented_attention_pair_bias [unsupported] | bf16 8x64 | 1024 | bf16 | infer | 0.080 | 0.039 | 2.07 | 18.31 | |
| augmented_attention_pair_bias [unsupported] | bf16 8x64 | 1024 | bf16 | fwd+bwd | 0.383 | 0.254 | 1.51 | 7.00 | |
| augmented_attention_pair_bias [unsupported] | bf16 16x48 L200 | 200 | bf16 | infer | 0.019 | 0.017 | 1.09 | 5.03 | |
| augmented_attention_pair_bias [unsupported] | bf16 16x48 L200 | 200 | bf16 | fwd+bwd | 0.075 | 0.064 | 1.18 | 3.95 | |
| BiasOnlyDiTBlock | shared pair | 100 | bf16 | infer | 0.106 | 0.149 | 0.71 | 0.78 | |
| BiasOnlyDiTBlock | shared pair | 100 | bf16 | fwd+bwd | 0.342 | 0.357 | 0.96 | 0.97 | |
| BiasOnlyDiTBlock | shared pair | 100 | fp32 | infer | 0.277 | 0.196 | 1.42 | 1.45 | |
| BiasOnlyDiTBlock | shared pair | 100 | fp32 | fwd+bwd | 0.788 | 0.696 | 1.13 | 1.14 | |
| BiasOnlyDiTBlock | shared pair | 128 | bf16 | infer | 0.143 | 0.143 | 1.00 | 1.00 | |
| BiasOnlyDiTBlock | shared pair | 128 | bf16 | fwd+bwd | 0.241 | 0.241 | 1.00 | 1.00 | |
| BiasOnlyDiTBlock | shared pair | 128 | fp32 | infer | 0.167 | 0.167 | 1.00 | 1.00 | |
| BiasOnlyDiTBlock | shared pair | 128 | fp32 | fwd+bwd | 0.440 | 0.440 | 1.00 | 1.00 | |
| BiasOnlyDiTBlock | shared pair | 256 | bf16 | infer | 0.173 | 0.173 | 1.00 | 1.00 | |
| BiasOnlyDiTBlock | shared pair | 256 | bf16 | fwd+bwd | 0.293 | 0.293 | 1.00 | 1.00 | |
| BiasOnlyDiTBlock | shared pair | 256 | fp32 | infer | 0.235 | 0.236 | 1.00 | 1.00 | |
| BiasOnlyDiTBlock | shared pair | 256 | fp32 | fwd+bwd | 0.588 | 0.588 | 1.00 | 1.00 | |
| BiasOnlyDiTBlock | shared pair | 512 | bf16 | infer | 0.264 | 0.263 | 1.00 | 1.00 | |
| BiasOnlyDiTBlock | shared pair | 512 | bf16 | fwd+bwd | 0.445 | 0.443 | 1.00 | 1.00 | |
| BiasOnlyDiTBlock | shared pair | 512 | fp32 | infer | 0.403 | 0.402 | 1.00 | 1.00 | |
| BiasOnlyDiTBlock | shared pair | 512 | fp32 | fwd+bwd | 0.965 | 0.960 | 1.01 | 1.00 | |
| BiasOnlyDiTBlock | shared pair | 1024 | bf16 | infer | 0.563 | 0.564 | 1.00 | 1.00 | |
| BiasOnlyDiTBlock | shared pair | 1024 | bf16 | fwd+bwd | 1.772 | 1.523 | 1.16 | 1.32 | |
| BiasOnlyDiTBlock | shared pair | 1024 | fp32 | infer | 0.959 | 0.963 | 1.00 | 1.00 | |
| BiasOnlyDiTBlock | shared pair | 1024 | fp32 | fwd+bwd | 5.378 | 4.825 | 1.11 | 1.39 | |
| BiasOnlyDiTBlock | pair per sample | 100 | bf16 | infer | 0.114 | 0.156 | 0.73 | 0.80 | |
| BiasOnlyDiTBlock | pair per sample | 100 | bf16 | fwd+bwd | 0.353 | 0.394 | 0.90 | 0.91 | |
| BiasOnlyDiTBlock | pair per sample | 100 | fp32 | infer | 0.298 | 0.212 | 1.41 | 1.40 | |
| BiasOnlyDiTBlock | pair per sample | 100 | fp32 | fwd+bwd | 0.810 | 0.753 | 1.08 | 1.08 | |
| BiasOnlyDiTBlock | pair per sample | 128 | bf16 | infer | 0.152 | 0.152 | 1.00 | 1.00 | |
| BiasOnlyDiTBlock | pair per sample | 128 | bf16 | fwd+bwd | 0.257 | 0.256 | 1.00 | 1.00 | |
| BiasOnlyDiTBlock | pair per sample | 128 | fp32 | infer | 0.159 | 0.160 | 0.99 | 1.00 | |
| BiasOnlyDiTBlock | pair per sample | 128 | fp32 | fwd+bwd | 0.453 | 0.452 | 1.00 | 1.00 | |
| BiasOnlyDiTBlock | pair per sample | 256 | bf16 | infer | 0.188 | 0.188 | 1.00 | 1.00 | |
| BiasOnlyDiTBlock | pair per sample | 256 | bf16 | fwd+bwd | 0.346 | 0.347 | 1.00 | 1.00 | |
| BiasOnlyDiTBlock | pair per sample | 256 | fp32 | infer | 0.230 | 0.230 | 1.00 | 1.00 | |
| BiasOnlyDiTBlock | pair per sample | 256 | fp32 | fwd+bwd | 0.667 | 0.675 | 0.99 | 1.02 | |
| BiasOnlyDiTBlock | pair per sample | 512 | bf16 | infer | 0.296 | 0.296 | 1.00 | 1.00 | |
| BiasOnlyDiTBlock | pair per sample | 512 | bf16 | fwd+bwd | 0.648 | 0.647 | 1.00 | 1.00 | |
| BiasOnlyDiTBlock | pair per sample | 512 | fp32 | infer | 0.434 | 0.430 | 1.01 | 1.00 | |
| BiasOnlyDiTBlock | pair per sample | 512 | fp32 | fwd+bwd | 1.345 | 1.345 | 1.00 | 1.00 | |
| BiasOnlyDiTBlock | pair per sample | 1024 | bf16 | infer | 1.897 | 1.935 | 0.98 | 0.98 | |
| BiasOnlyDiTBlock | pair per sample | 1024 | bf16 | fwd+bwd | 3.417 | 3.386 | 1.01 | 1.01 | |
| BiasOnlyDiTBlock | pair per sample | 1024 | fp32 | infer | 4.766 | 4.104 | 1.16 | 1.17 | |
| BiasOnlyDiTBlock | pair per sample | 1024 | fp32 | fwd+bwd | 7.657 | 7.147 | 1.07 | 1.07 | |
| AugmentedAttentionPairBias module | A2 D768 | 100 | bf16 | infer | 0.100 | 0.079 | 1.27 | 1.26 | |
| AugmentedAttentionPairBias module | A2 D768 | 100 | bf16 | fwd+bwd | 0.313 | 0.252 | 1.24 | 1.26 | |
| AugmentedAttentionPairBias module | A2 D768 | 100 | fp32 | infer | 0.197 | 0.158 | 1.25 | 1.26 | |
| AugmentedAttentionPairBias module | A2 D768 | 100 | fp32 | fwd+bwd | 0.510 | 0.460 | 1.11 | 1.12 | |
| AugmentedAttentionPairBias module | A2 D768 | 128 | bf16 | infer | 0.069 | 0.076 | 0.91 | 0.86 | |
| AugmentedAttentionPairBias module | A2 D768 | 128 | bf16 | fwd+bwd | 0.274 | 0.276 | 0.99 | 1.00 | |
| AugmentedAttentionPairBias module | A2 D768 | 128 | fp32 | infer | 0.206 | 0.166 | 1.24 | 1.27 | |
| AugmentedAttentionPairBias module | A2 D768 | 128 | fp32 | fwd+bwd | 0.551 | 0.480 | 1.15 | 1.14 | |
| AugmentedAttentionPairBias module | A2 D768 | 256 | bf16 | infer | 0.091 | 0.098 | 0.92 | 0.94 | |
| AugmentedAttentionPairBias module | A2 D768 | 256 | bf16 | fwd+bwd | 0.329 | 0.332 | 0.99 | 0.99 | |
| AugmentedAttentionPairBias module | A2 D768 | 256 | fp32 | infer | 0.309 | 0.242 | 1.28 | 1.28 | |
| AugmentedAttentionPairBias module | A2 D768 | 256 | fp32 | fwd+bwd | 0.871 | 0.739 | 1.18 | 1.18 | |
| AugmentedAttentionPairBias module | A2 D768 | 512 | bf16 | infer | 0.125 | 0.134 | 0.94 | 0.95 | |
| AugmentedAttentionPairBias module | A2 D768 | 512 | bf16 | fwd+bwd | 0.469 | 0.476 | 0.99 | 0.99 | |
| AugmentedAttentionPairBias module | A2 D768 | 512 | fp32 | infer | 0.695 | 0.503 | 1.38 | 1.38 | |
| AugmentedAttentionPairBias module | A2 D768 | 512 | fp32 | fwd+bwd | 1.873 | 1.490 | 1.26 | 1.26 | |
| AugmentedAttentionPairBias module | A2 D768 | 1024 | bf16 | infer | 0.231 | 0.237 | 0.97 | 0.97 | |
| AugmentedAttentionPairBias module | A2 D768 | 1024 | bf16 | fwd+bwd | 1.021 | 1.028 | 0.99 | 0.99 | |
| AugmentedAttentionPairBias module | A2 D768 | 1024 | fp32 | infer | 2.210 | 1.585 | 1.39 | 1.40 | |
| AugmentedAttentionPairBias module | A2 D768 | 1024 | fp32 | fwd+bwd | 5.346 | 4.115 | 1.30 | 1.29 | |