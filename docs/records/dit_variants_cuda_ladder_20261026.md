# DiT variant stacks (Boltz-2 / RoseTTAFold3 / Chai-1): hand-CUDA training on H100, ladder before / after (2026-10-26)

Branch `perf/ladder-fix-ditvar` (from `perf/ladder-audit`). The ladder rule is hand CUDA < portable (Triton) < PyTorch. In
[performance_ladder_20261025.md](performance_ladder_20261025.md) the `miniworld` rung of the six variant stacks (token L384, atom 512 tokens = 4096 atoms) in
TRAINING ran the PyTorch reference (no CUDA training path existed), 1.5-1.6x (token) and 2.1-2.3x (atom) slower than the portable kernels. fp32 rows were not in the ladder
(groups `dit_*_fp32`, `local_*_fp32` added here; same behaviour: reference).

## What changed

* `kernels/dit_variants/cuda/train.py` + `train_rows.cu` + `edge.cu`: one autograd `Function` per block (opaque fwd / bwd ops), hand-CUDA row kernels forward and backward
  (LN + AdaLN modulation, qk prep / kq-norm, residual + gates, SwiGLU / up-gate, their backward, gate backward, qkv backward, AdaLN backward with the table gradients summed over
  the samples as fixed-order partial rows, pair LayerNorm rows, tiled transposes), cuBLAS GEMMs, and the existing sm_90a attention cores (token bf16 `attn_fwd` / `attn_dqb` / `attn_dkv`,
  token fp32 TF32 `attn_tf32`, atom `lattn90` bf16 / TF32). The slid key window's five edge windows (atoms) have their own fp32 register-blocked kernels (forward, backward compute / apply) and edge bias.
  bf16 (autocast, fp32 master parameters) and fp32 (TF32; attention operands rounded to nearest TF32). Conditioning tables and the pair bias are autograd ops outside the block.
  Switch: `settings.dit_variant_train_cuda` (default on). Bitwise repeatable (test).
* fp32 (TF32) INFERENCE: `kernels/dit_variants/cuda/infer32.py`, a hand-CUDA path built from the training rows (`train_rows.cu`, `edge.cu`), cuBLAS TF32 and the TF32 attention cores without anything the backward needs, and with the stack-level work hoisted: folded fp32 weights cached per parameter version, the conditioning tables of every block from two GEMMs, the pair bias of every block from one pair LayerNorm + one stacked GEMM (token, key-permuted for the TF32 core) or `pair_bias_mma` + the edge tables (atom), cached per pair (`_capture.lookup_inputs`). An earlier attempt that ran the training path under no_grad was 1.02-1.19x slower than portable; the portable routing that was tried for it is removed (no exception remains).
* Bench `execution_path` reports the training route; ladder gained `--modes` / `--kinds` and the fp32 variant groups; bundle coverage builds the training extensions.

## Ladder, training (compile + CUDA graph, median of 3 repeats of the medians; H100, torch 2.13, one-layer stack, A = 48)

before = same harness at the base engine (`MINIWORLD` rung = PyTorch reference), after = this branch. Ratio = CUDA / portable (< 1 required).

| row | dtype | PyTorch ms | portable ms | CUDA ms before | ratio before | CUDA ms after | ratio after |
|---|---|---|---|---|---|---|---|
| Boltz-2 token L384 | bf16 | 10.32 | 3.728 | 5.608 | 1.50 | 3.054 | 0.819 |
| RoseTTAFold3 token | bf16 | 10.59 | 3.575 | 5.418 | 1.52 | 2.786 | 0.779 |
| Chai-1 token | bf16 | 9.91 | 3.221 | 5.122 | 1.59 | 2.637 | 0.819 |
| Boltz-2 token | fp32 | 12.75 | 7.487 | 8.540 | 1.14 | 6.630 | 0.886 |
| RoseTTAFold3 token | fp32 | 12.23 | 6.824 | 7.806 | 1.14 | 6.071 | 0.890 |
| Chai-1 token | fp32 | 11.38 | 6.299 | 7.431 | 1.18 | 5.828 | 0.925 |
| Boltz-2 atom (4096) | bf16 | 20.40 | 4.430 | 9.351 | 2.11 | 3.481 | 0.786 |
| RoseTTAFold3 atom | bf16 | 21.54 | 4.161 | 9.355 | 2.25 | 3.392 | 0.815 |
| Chai-1 atom | bf16 | 19.20 | 3.600 | 7.990 | 2.22 | 2.853 | 0.793 |
| Boltz-2 atom | fp32 | 21.59 | 7.431 | 11.409 | 1.54 | 5.835 | 0.785 |
| RoseTTAFold3 atom | fp32 | 21.73 | 6.847 | 10.734 | 1.57 | 5.541 | 0.809 |
| Chai-1 atom | fp32 | 19.35 | 5.788 | 9.350 | 1.62 | 4.770 | 0.824 |

All 12 training rows: OK (CUDA < portable < PyTorch).

## Ladder, inference (final run; hand CUDA rung, CUDA / portable)

| row | dtype | portable ms | CUDA ms | CUDA / portable |
|---|---|---|---|---|
| Boltz-2 token | bf16 | 0.215 | 0.118 | 0.549 |
| RF3 token | bf16 | 0.197 | 0.108 | 0.548 |
| Chai-1 token | bf16 | 0.207 | 0.108 | 0.523 |
| Boltz-2 atom | bf16 | 0.194 | 0.159 | 0.820 |
| RF3 atom | bf16 | 0.192 | 0.154 | 0.804 |
| Chai-1 atom | bf16 | 0.163 | 0.129 | 0.794 |
| Boltz-2 token | fp32 | 0.406 | 0.236 | 0.581 |
| RF3 token | fp32 | 0.376 | 0.206 | 0.548 |
| Chai-1 token | fp32 | 0.436 | 0.213 | 0.488 |
| Boltz-2 atom | fp32 | 0.360 | 0.290 | 0.806 |
| RF3 atom | fp32 | 0.345 | 0.273 | 0.792 |
| Chai-1 atom | fp32 | 0.303 | 0.228 | 0.751 |

Before, the fp32 inference rows ran the PyTorch reference (0.43-0.50 ms, 1.07-1.42x slower than portable). All 24 variant-stack rows (12 training, 12 inference) are OK.

## Parity and tests

`tests/gpu/sm90/test_dit_variant_train_h100_gpu.py` (output and every gradient within 1.5x the same-dtype PyTorch error vs fp64; measured 0.8-1.0x; no mask, padded key window, padding atoms,
Chai-1 `uid`; bitwise repeat; routing). Run with the existing `test_dit_variant_h100_gpu.py`, `test_dit_variants_portable_gpu.py`: green. CPU suite: same 2 failures + 2 errors as the base (`tests/cpu/registry`).
`dev derive --arch sm86` re-run (units json).

## Open

* fp32 token rows are the tightest (Chai-1 0.925): TF32 GEMMs are 3.0 of 5.8 ms, the TF32 attention cores 1.3 ms, and the large fp32 row passes (SwiGLU forward / backward over [M, 3072], the pair LayerNorm) already run at 85-100 % of HBM roofline; further gain needs fusing the SwiGLU into the expand GEMM epilogue for training (an sm90 `SAVE_AB` build), not done.
* The "other options" pytest cases were added after the last full pytest run; they passed in the parity script, not yet in pytest together with the rest.
* Edge kernels (atoms) cost ~230 us of ~3 ms; row kernels run at 50-65 % of HBM roofline (no ILP across samples yet).
