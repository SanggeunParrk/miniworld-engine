# Triton paths for the model variants: PWA value_dim 8, OPM projection biases / groups, TriangleAttention variants (2026-10-24)

Gap: on a card with no native CUDA path (A100 / A5000 / A6000, anything recorded later) these calls had no Triton kernel and ran the
PyTorch statements: `MSAPairWeightedAveraging` with `msa_value_dim` 8 (AF3, OpenFold3, Protenix, OpenDDE), `OuterProductMean`
`projection_bias` (RoseTTAFold3) and `groups=8` (Chai-1), `TriangleAttention` with gate / out-projection biases (RoseTTAFold3) and
`untransposed_output` / `dual_output` (Chai-1). Now they run Triton, forward and backward.

Measured on **one H100 80GB HBM3 only**, with `settings.force_triton_kernels=True` (the A/B switch: the native CUDA paths step aside).
The kernels have **no manifest rows for any other card**: under the routing rule A100 / A5000 / A6000 still warn and run PyTorch until
`python -m miniworld_engine.autotune.run_all` has recorded them there. Autotune used the default config lists (no tuned cache for these
kernels on this card), so the Triton times are an untuned floor.

Method: bf16 modules, B = 1, MSA depth S as stated, CUDA-graph replay of the forward (`no_grad`) and of forward + `autograd.grad` over the
inputs and all parameters, 20 replays after warm-up. Baselines: the module's PyTorch statements (`implementation="pytorch"`) replayed in a
graph ("graph-eager") and the same module under `torch.compile` (default mode) replayed in a graph ("compiled"). Script:
`~/miniworld-engine-scratch/gapsA/bench_gaps.py`.

## Speed (ms; speed-up = baseline / Triton)

| case | Triton fwd | vs graph-eager | vs compiled | Triton fwd+bwd | vs graph-eager | vs compiled |
|---|---|---|---|---|---|---|
| PWA S512 L256 8 heads x 8 | 0.096 | 5.6x | 1.30x | 0.407 | 3.4x | 1.08x |
| PWA S512 L384 8 x 8 | 0.152 | 6.1x | 1.65x | 0.631 | 3.1x | 1.18x |
| PWA S1024 L256 8 x 8 | 0.167 | 5.7x | 2.00x | 0.719 | 3.1x | 1.17x |
| PWA S512 L384 8 x 32 (existing width, for reference) | 0.272 | 5.3x | 1.98x | 1.035 | 3.1x | 1.40x |
| OPM RF3 (bias, divide before) S512 L256 | 0.221 | 5.0x | 1.49x | 0.841 | 3.1x | 1.01x |
| OPM RF3 S512 L384 | 0.458 | 4.9x | 1.52x | 1.690 | 3.2x | 1.07x |
| OPM RF3 (divide after) S512 L256 | 0.221 | 3.4x | 1.50x | 0.911 | 2.0x | 0.99x |
| OPM RF3 (divide after) S512 L384 | 0.451 | 3.2x | 1.53x | 1.873 | 1.8x | 0.99x |
| OPM Chai-1 (8 groups, 64 hidden, pair 256) S512 L256 | 0.218 | 3.5x | 1.32x | 0.785 | 2.5x | 1.18x |
| OPM Chai-1 S512 L384 | 0.447 | 3.2x | 1.30x | 1.494 | 2.1x | 1.21x |
| TriAttn RF3 template (64 / 256, 4 x 64, biases) L256 | 0.367 | 6.7x | 2.72x | 1.475 | 4.4x | 2.00x |
| TriAttn RF3 template L384 | 0.627 | 11.5x | 6.21x | 2.579 | 7.2x | 3.97x |
| TriAttn RF3 128 / 4 x 32, ending, biases L256 | 0.372 | 5.9x | 2.68x | 1.494 | 3.9x | 1.88x |
| TriAttn RF3 128 L384 | 0.573 | 11.4x | 5.53x | 2.495 | 6.7x | 3.32x |
| TriAttn Chai-1 untransposed ending (256, 4 x 64) L256 | 0.583 | 4.8x | 2.32x | 2.033 | 3.6x | 1.82x |
| TriAttn Chai-1 untransposed L384 | 0.937 | 8.5x | 4.30x | 3.419 | 5.9x | 3.10x |
| TriAttn Chai-1 dual (confidence head) L256 | 0.620 | 4.4x | 4.41x * | 2.200 | 3.3x | 3.30x * |
| TriAttn Chai-1 dual L384 | 1.014 | 7.7x | 7.66x * | 3.759 | 5.3x | 5.35x * |

\* `torch.compile` of the dual module did not improve on eager (its time equals graph-eager): read those two columns as graph-eager.

Honest reading: the Triton path beats the PyTorch statements everywhere; against `torch.compile` it wins by 1.1-7x in the forward and
1.1-5x in forward + backward, **except** OuterProductMean RoseTTAFold3 training, which is at parity with compiled PyTorch (1.01-1.07x;
0.99x with the division after the projection). The OPM backward is dominated by its cuBLAS GEMMs and the untuned stages around them.

## Parity (bf16 Triton vs fp32 autograd of the PyTorch statements, relative L2; same inputs and cotangent)

The bf16 statements' own error against the same fp32 reference is the yardstick: the Triton path matches or beats it everywhere.

| case | output: Triton / bf16 PyTorch | worst gradient (inputs and all parameters): Triton / bf16 PyTorch |
|---|---|---|
| PWA 8 x 8 | 2.43e-3 / 2.48e-3 | 7.4e-3 / 8.5e-3 |
| PWA 4 x 8 | 2.43e-3 / 2.48e-3 | 7.2e-3 / 8.5e-3 |
| PWA 4 x 12 | 2.43e-3 / 2.48e-3 | 7.2e-3 / 8.5e-3 |
| PWA 8 x 32 (existing) | 2.42e-3 / 2.46e-3 | 7.3e-3 / 9.0e-3 |
| OPM RF3 divide-before | 2.41e-3 / 2.48e-3 | 7.2e-3 / 6.8e-3 |
| OPM RF3 divide-after | 2.42e-3 / 2.46e-3 | 7.2e-3 / 8.1e-3 |
| OPM Chai-1 (K = 8) | 4.91e-3 / 4.91e-3 | 6.9e-3 / 7.5e-3 |
| OPM Chai-1 (K = 4) | 4.85e-3 / 4.85e-3 | 7.0e-3 / 7.5e-3 |
| TriAttn RF3 template | 2.69e-3 / 2.71e-3 | 9.4e-3 / 9.4e-3 |
| TriAttn RF3 ending | 2.68e-3 / 2.69e-3 | 9.6e-3 / 9.6e-3 |
| TriAttn Chai-1 untransposed | 2.59e-3 / 2.60e-3 | 9.8e-3 / 9.8e-3 |
| TriAttn Chai-1 dual (starting / ending) | 2.83e-3 / 2.85e-3 | 1.01e-2 / 0.99e-2, 1.04e-2 / 1.10e-2 |
| TriAttn RF3 8 x 32 + dual + biases | 2.99e-3 / 3.01e-3 | 9.8e-3 / 9.7e-3 |

A head of 8 channels against the same head zero-padded to 16 (`test_narrow_head_is_the_zero_padded_head`): output within 2e-3, every
gradient within 5e-3 (the two cuBLAS contractions differ in size, so not bitwise).

Registry `rtol` (max |a-e| / max |e|, the repo's metric): measured worst per kernel 4.3e-3 (`bwd_epilogue_ln_dw`), 5.1e-3
(`epilogue_ln`), 6.1e-3 (`bwd_epilogue_ln_dx`); bands 1.8e-2 / 1.8e-2 / 2.5e-2 (about 4x). The existing kernels, now driven at the
variants too, measured at most 7.7e-3 against their 1.3e-2.

## How

* **PWA, value_dim 8.** Five kernels carry the head width `C` as a `tl.dot` operand (`gate_gemm`, `bwd_gate_gemm`, `bwd_layernorm_gemm_dw`,
  `bwd_layernorm_gemm_dx_dlnw`; the projection kernel only needs `BN | n_head C`). `CP = next_pow2(max(16, C))` is a constexpr; every
  load of a `C`-wide slice is masked to `cc < C` (zeros) and every store / atomic likewise, so padded columns are exact zeros that contribute
  nothing. `v` and `o` keep the compact `[NH, L, S C]` layout, so the two cuBLAS contractions are not inflated. For `C` >= 16 a power of two
  the generated code is the old one. `refusal` accepts `d_hidden` 1..128 with `n_head * d_hidden` a multiple of 16.
* **OPM projection biases.** The bias is added in the projection kernel's epilogue (fp32 accumulator, before the bf16 cast, as the bf16
  Linear does), then the mask zeroes the row; the backward kernel accumulates `db_left` / `db_right` from the masked `dA` / `dB`.
* **OPM groups.** The projection kernel stores `a`, `b` group-major `[G, S L, K]`, so `O[g] = a_g^T b_g` is one batched GEMM `[G, L K, L K]`
  (8x less work and memory than the full `[L CH, L CH]` product). The pair's products are `O[g, i K + k, j K + l]`. The first version fused
  gather + LayerNorm + projection in one Triton kernel: 435 us at L256 -- the gather was strided 16-byte runs. The kernels now read the
  tile the way O is laid out (for fixed `(g, k, i)` the `(j, l)` run is contiguous: a `[G K, BJ K]` tile, coalesced), normalise per
  `j`, and write rows of `y [L L, P]`; the projection `y W^T + b` is cuBLAS (bias in its epilogue) and the backward `dy = dz W` too
  (47 us vs 435 us for the gather kernel at L256). dW_out is a split-K kernel over `(dz, y)`. Remaining cost: the residual add (33 us at
  L256, a separate pass) and `db_out`.
* **TriAttn variants.** No new attention kernel: the core (`triton_triangle_attention_pair_bias`, flash-style, with its own backward)
  is shared; the variants differ in the epilogue and in orientation, which are cuBLAS projections with their biases in the epilogue plus the
  family's fused `sigmoid(gate) * o`, composed in `kernels/triangle_attention/triton/variants.py`. A row whose keys are all masked
  gives exactly 0 from the core, as the statements' `nan_to_num` does.

## Not done / limits

* Only an H100 was available: no A100 / A5000 / A6000 number and no manifest row. The routing rule therefore still sends those cards
  to PyTorch (one warning) for these variants until they are recorded.
* Parameter gradients of the new OPM kernels use fp32 atomics (as the existing OPM and PWA kernels do): not bitwise deterministic
  across runs; no determinism test covers these families.
* OPM RoseTTAFold3 training is at parity with compiled PyTorch, not faster. The Chai-1 OPM's `y` is kept for the backward (`[L L, 512]` bf16:
  67 MB at L256, 150 MB at L384).
* `TriangleAttention` variants need bf16 activations *and* bf16 parameters (the fp32-master layout is not taken here).
* Autotune caches for the new / changed kernels were not built (optional); the runs above used the default config lists.
