# Token DiT with `extra_pair_bias`: H100 ladder fix (2026-10-27)

Branch `perf/final-xbias` (from `integ/ladder-all2` 1361277f). Ladder groups `dit_xbias_bf16`, `dit_xbias_fp32`: `DiTBlock(extra_pair_bias=)`, d768 16 heads, A = 2,
shared conditioning, an [L, L] term added to every head's logits. Rule: hand CUDA < portable (Triton) < PyTorch (best of graph-eager / torch.compile).

## Cause (from the per-row kernel lists of the integration run)

* The fused CUDA training block did not take the term: `DiTBlock` skipped it whenever `extra_pair_bias` was given, so the "CUDA" rung of training was the composed module
  (CUDA attention core + PyTorch glue), 0.998-1.006x PyTorch. fp32 training had no CUDA rung (the PyTorch reference ran).
* fp32 inference: `serves_extra_bias` admitted bf16 only, so the fp32 call ran the composed path (`mwln` + `expand_swiglu` + cuBLAS TF32), slower than portable.
* Portable training: LayerNorm kernel + GEMM + their backwards for the pair bias (the pair LayerNorm path is ~45 % of the step at L512) next to the same PyTorch statements.
  The fused LayerNorm + projection Triton kernel that exists (`layernorm_linear/triton/pair_bias.py`) was slower than that (fp32 FMA dots, per-tile atomics, 57 / 68 us fwd / bwd at L256).

## What changed

* `integrations/token_dit_train.py`: the fused H100 training block takes `extra` ([L, L], bf16 or fp32): bf16 and fp32/TF32, padded L, gradient to the term (`dextra` = sum over heads of
  dbias). `modules/dit/module.py` routes `extra_pair_bias` to it (inference to the existing runner).
* bf16 pair path of the block: ONE pass over the pair each way, reusing the existing `pair_bias` / `pair_bias_bwd` row kernels of `token_dit_train_rows.cu` (TF32 mma; they were the B200 fp32
  path): forward writes the bf16 head-major bias in log2 units with the extra term added before the single rounding, no LN(pair) kept; backward recomputes LN(pair) from (mean, rstd) and writes
  dpair and the dWf partials. The kernels gained an optional extra term, a bf16 bias output and a bf16 dbias input (no new kernel).
* fp32 inference: `kernels/conditioned_transition/cuda/token_inference` folds the term into the TF32 core's bias in key-permuted order; `serves_extra_bias` admits fp32.
* Portable: `pair_bias.py` (registered `layernorm_linear_*_fp32_triton`): tensor-core dots when the input is 16-bit (LayerNorm output rounded to the input dtype, as LayerNorm -> Linear does),
  persistent one-tile backward (parameter sums added once per program), fp16/bf16 output without a cast pass, bf16 `dout` read as is, the 128-wide tile added to the config space
  (`axes.csv`, `configs/grid|default`). `AugmentedAttentionPairBias` uses it for training on the Triton core (not on H100/B200 unless portable is forced, not for fp32).
* Test `tests/gpu/sm90/test_dit_extra_pair_bias_h100_gpu.py` (routing asserted, vs fp64, inference and training, bf16 / fp32, masked, ragged L, bitwise repeat).

## Ladder (compile + CUDA graph, median of 3; H100; ms)

before = integration run `out2_J2` (commit e35029a1), after = this branch. c/p = CUDA / portable, p/t = portable / PyTorch (both must be < 1).

| row | L | PyTorch before -> after | portable before -> after | CUDA before -> after | c/p before -> after | p/t before -> after |
|---|---|---|---|---|---|---|
| bf16 inference | 256 | 0.152 -> 0.153 | 0.124 -> 0.124 | 0.062 -> 0.062 | 0.502 -> 0.501 | 0.813 -> 0.811 |
| bf16 inference | 512 | 0.265 -> 0.265 | 0.208 -> 0.209 | 0.075 -> 0.075 | 0.361 -> 0.359 | 0.785 -> 0.791 |
| bf16 training | 256 | 0.423 -> 0.424 | 0.440 -> 0.415 | 0.425 -> 0.337 | 0.968 -> 0.813 | 1.039 -> 0.978 |
| bf16 training | 512 | 0.702 -> 0.702 | 0.712 -> 0.634 | 0.698 -> 0.550 | 0.980 -> 0.867 | 1.014 -> 0.904 |
| fp32 inference | 256 | 0.205 -> 0.206 | 0.176 -> 0.177 | 0.195 -> 0.102 | 1.112 -> 0.578 | 0.858 -> 0.861 |
| fp32 training | 256 | 0.682 -> 0.686 | 0.633 -> 0.632 | 0.687 (PyTorch, no CUDA rung) -> 0.592 | n.a. -> 0.936 | 0.928 -> 0.922 |

All six rows OK. Not yet a "clear win" (<= 0.95): bf16 training L256 portable / PyTorch 0.978 (the weight casts, 52 us in three foreach copies, and ~190 us of cuBLAS GEMMs are common to both rungs;
the remaining portable slack is the pair-bias backward, 39 us against an ~11 us HBM floor, and the Triton attention backward), fp32 training L256 CUDA / portable 0.936.
Neighbouring groups (token DiT, DiT variants, aat, apb): no violation in the re-run (`out/f`).

## Parity

`test_dit_extra_pair_bias_h100_gpu.py` 13 passed: output and every gradient (inputs, pair, the term, all parameters) within 1.5x the bf16 PyTorch error vs fp64 (measured 0.8-0.95x);
fp32 (TF32 engine against the full-fp32 PyTorch block) within the absolute floor of `tests/gpu/any/test_dit_extra_pair_bias_portable_gpu.py`. Related files, all passed:
`test_token_dit_train_gpu` (60), `test_token_dit_rows_cuda_gpu` (16 + 4 skipped), `test_h100_token_dit_capture_gpu` (18), `test_h100_token_dit_layouts_gpu` (40), `test_token_dit_psb_gpu` (12),
`test_token_dit_batched_pairs_gpu` (12), `test_dit_extra_pair_bias_portable_gpu` and `test_normalization_atomic_gradients_gpu` (29 together).
Pair-bias Triton kernel vs fp64 (random pair, L256 / 512): dz / dgamma / dW 2.3e-3 / 1.6e-3 / 1.6e-3 against 2.9e-3 / 2.2e-3 / 2.3e-3 for LayerNorm -> Linear in bf16.
