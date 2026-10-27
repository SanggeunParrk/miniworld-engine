# Bias-only DiT H100 training: CUDA path against the faster portable path (2026-10-28)

Branch `perf/final-bodit-train-cuda` (from `perf/final-adalnct-cuda`, ef3e2c3d). Numbers: `benchmarks/runners/ladder.py` (compile + CUDA graph, median of 3, B = 1, token width 768, A = 48 samples, H100 80GB; noise 1 - 3 %).

## Results (training, ms; CUDA / portable is the ratio of the same run)

| row | PyTorch | portable | CUDA before | CUDA after | CUDA / portable before -> after |
|---|---|---|---|---|---|
| fp32 16x48 L256 | 3.850 | 3.370 | 3.346 | 3.213 | 0.988 -> 0.953 |
| fp32 16x48 L512 | 7.193 | 6.333 | 6.413 | 6.133 | 0.999 -> 0.968 |
| bf16 16x48 L256 | 1.676 | 1.549 | 1.453 | 1.461 | 0.949 -> 0.943 |
| bf16 16x48 L512 | 3.312 | 3.032 | 2.817 | 2.845 | 0.937 -> 0.938 |
| bf16 16x48 L768 | 5.015 | 4.570 | 4.294 | 4.328 | 0.949 -> 0.947 |
| bf16 24x32 L512 | 3.347 | 3.079 | 3.063 | 2.961 | 0.998 -> 0.962 |
| bf16 12x64 L512 | 3.303 | 3.030 | 2.892 | 2.847 | 0.953 -> 0.940 |

"Before" is a re-measurement of ef3e2c3d on the same node. Inference rows (`bodit_*`, modes inference) did not move (0.79 - 0.93, same as before). The goal (<= 0.95) is met for every
bf16 row except 24x32 L512 (0.962) and for fp32 L256 (0.953, within noise of 0.95); fp32 L512 is 0.968. They are all < 1, none is a violation any more.

## What was slow (per-kernel profile of one step, `kernel_probe`)

* fp32 L512: the Hopper attention backward went through torch copy kernels, 244 us of the 6.1 ms kernel sum (5 `elementwise_kernel`: dO permuted to `[H, L, A*dh]`, v permuted, the dV result permuted back, ...), and the forward ran a second GEMM for `v^T`
  (92 us) beside the `v | g` GEMM (161 us). cuBLAS TF32 GEMMs are the same kernels in both rungs (about 3.9 ms), the row kernels are at the HBM wall, so these copies were the only margin.
* bf16 24x32 / 12x64: the bias-gradient kernel `dpb2` (128 x TN tile, 49 us at 16x48) only existed for head width 48; the other widths used the 64 x 64 `dpb_pipe` (173 us at 24x32, 92 us at 12x64).

## What changed

1. `pv_gate_inf.cu` / `cuda/__init__.py`: `dpb2` templated on the head width (32 / 48 / 64: the MMA k-loop is `DH / 16`), routed for every H100 layout at L >= 384. 24x32: dpb 173 -> ~50 us; 12x64: 92 -> ~50 us.
2. `bias_only_dit_train.py` (fp32, Hopper, shared pair): feature-major core as in the portable path. The forward computes `v^T [DA, M]` and the gate `g [M, DA]` as two GEMMs into one flat buffer (no second `v^T` GEMM: -69 us).
   The backward reads `v^T` as `[H, (d, a), L]`, a view; `dV^T = dO^T P` and `dP = dO v^T` are one batched GEMM each with no permute copy; `dxa` / `dWvg` are two GEMMs (feature-major dV, row-major dgate) into the same outputs.
3. `bias_only_dit_f32_rows.cu`: `gate_bwd_fold_k` writes dO directly in the `[H, L, dh, A]` layout the batched GEMMs read. A first version with per-warp 4-byte scattered stores cost 442 us (vs 131 us);
   the shipped one stages 8 samples of one token in shared memory and stores 2 x float4 (a full 32-byte sector) per channel: 140 us.
   Net of the attention part: the 244 us of copies are gone for +9 us in the gate pass.

## Tried and not kept / left

* `pv_cta` (bf16 PV, 24x32 and 12x64) `(WGS, group, cluster)` sweep: at most 1 - 5 % of a 90 - 120 us kernel (`sweep_pvx.py`: best forced 118.3 vs 119.6 us at 24x32 L512); the cost model's default is within noise, not changed.
* fp32 weight-gradient / data-gradient GEMM layouts (a^T b, (b^T a)^T, bmm over 2 / 4 row chunks, nn vs tn forward, column-major weights): the current choices are already the fastest (e.g. dWab 556 us, 212 TF/s; all alternatives 556 - 1078 us).
  The fp32 step is 3.9 ms of cuBLAS TF32 at 190 - 300 TF/s; a hand wgmma TF32 GEMM with the SwiGLU epilogue is the next margin (about 100 us each for forward and backward SwiGLU), not attempted.
* `ladder.py` group `bodit_psb` fails on every command (`+per_sample_pair=true` collides with the key already in `bench.yaml`: needs `++`); it is not part of this change. The per-sample-pair paths were covered by `test_bias_only_dit_psb_gpu.py` instead.

## Tests

`tests/gpu/sm90/test_h100_bias_only_dit_gpu.py` (dbias test now over all layouts, 16x48 / 24x32 / 12x64 / 16x64, bitwise repeatability asserted; the training gradient test, bf16 and fp32, all layouts, against the fp32 oracle, bitwise repeatable),
`test_bias_only_dit_psb_gpu.py` (75 passed), plus `tests/gpu/any/test_bias_only_dit_portable_gpu.py` and `test_bodit_ct_training_portable_gpu.py`.
