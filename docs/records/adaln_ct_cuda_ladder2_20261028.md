# AdaLN / ConditionedTransition H100 hand-CUDA: second round against the faster portable rung (2026-10-28)

Branch `perf/final-adalnct-cuda` = `perf/ladder-fix-adaln-ct-cuda` (d6e42640) + merges of `perf/final-adaln-triton` (bbaf835b) and `perf/final-ct-fp32` (a2a1db21), whose
portable Triton kernels got faster (K-major fp32 dgrad weights, cuBLAS dgrad + fused GEMM + gate at the atom width, `dsb` folded into the dx kernel; see
`adaln_portable_training_20261027.md`, `portable_ct_fp32_training_20261027.md`). Goal: CUDA / portable <= 0.95 and below PyTorch in every AdaLN / ConditionedTransition / bias-only B = 1 row.
One H100 80GB HBM3, torch 2.13 + cu129, `benchmarks/runners/ladder.py` (all four regimes, the faster one for the CUDA and the portable rung, medians of 3; rows are +-1-3 % noisy),
`settings.force_portable_kernels` for the portable rung. "Before" = the merged tree before this round (ladder jobs 37154-37200, ratio column), "after" = the tip of this round.
The portable rung ran with a STALE tuned autotune cache (warning in every log: the Triton / ptxas version moved since it was built), i.e. on the full config grid; the cache is
not touched here.

## Result (ms; PyTorch = the fastest of compiled / eager-in-graph / eager)

Columns: op, mode, dtype, shape, L (atoms for the d128 rows), PyTorch, portable, CUDA, CUDA / portable after, CUDA / portable before -> after (`-`: the row did not run before: the bf16 atom inference
rows hung, see below).

| op | mode | dtype | shape | L | PyTorch | portable | CUDA | CUDA / portable | before -> after |
|---|---|---|---|---|---|---|---|---|---|
| AdaLN | inference | bf16 | d128 cond128 A5/A48 (atoms) | 1024 | 0.021 | 0.013 | 0.011 | 0.794 | - -> 0.794 |
| AdaLN | inference | bf16 | d128 cond128 A5/A48 (atoms) | 4096 | 0.031 | 0.025 | 0.019 | 0.748 | - -> 0.748 |
| AdaLN | inference | bf16 | d128 cond128 A5/A48 (atoms) | 8192 | 0.046 | 0.038 | 0.027 | 0.712 | - -> 0.712 |
| AdaLN | inference | bf16 | d768 cond384 A5/A48 | 1024 | 0.041 | 0.032 | 0.027 | 0.858 | 0.846 -> 0.858 |
| AdaLN | inference | bf16 | d768 cond384 A5/A48 | 384 | 0.027 | 0.018 | 0.017 | 0.939 | 0.940 -> 0.939 |
| AdaLN | inference | bf16 | d768 cond384 A5/A48 | 768 | 0.036 | 0.027 | 0.024 | 0.874 | 0.867 -> 0.874 |
| AdaLN | inference | bf16 | d768 cond768 A5/A48 (ESMFold2) | 128 | 0.025 | 0.017 | 0.015 | 0.893 | 0.895 -> 0.893 |
| AdaLN | inference | bf16 | d768 cond768 A5/A48 (ESMFold2) | 384 | 0.032 | 0.023 | 0.021 | 0.927 | 0.922 -> 0.927 |
| AdaLN | inference | bf16 | d768 cond768 A5/A48 (ESMFold2) | 768 | 0.043 | 0.037 | 0.032 | 0.852 | 0.852 -> 0.852 |
| AdaLN | inference | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | 0.022 | 0.017 | 0.015 | 0.877 | 0.906 -> 0.877 |
| AdaLN | inference | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | 0.045 | 0.035 | 0.029 | 0.835 | 1.010 -> 0.835 |
| AdaLN | inference | fp32 | d128 cond128 A5/A48 (atoms) | 8192 | 0.081 | 0.063 | 0.046 | 0.735 | 1.051 -> 0.735 |
| AdaLN | inference | fp32 | d768 cond384 A5/A48 | 128 | 0.022 | 0.016 | 0.014 | 0.871 | 0.974 -> 0.871 |
| AdaLN | inference | fp32 | d768 cond384 A5/A48 | 384 | 0.034 | 0.025 | 0.023 | 0.925 | 0.930 -> 0.925 |
| AdaLN | inference | fp32 | d768 cond384 A5/A48 | 768 | 0.051 | 0.044 | 0.036 | 0.807 | 0.816 -> 0.807 |
| AdaLN | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | 0.026 | 0.022 | 0.019 | 0.866 | 0.966 -> 0.866 |
| AdaLN | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | 0.044 | 0.037 | 0.034 | 0.903 | 0.889 -> 0.903 |
| AdaLN | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | 0.070 | 0.066 | 0.056 | 0.840 | 0.834 -> 0.840 |
| AdaLN | training | bf16 | d128 cond128 A5/A48 (atoms) | 1024 | 0.188 | 0.134 | 0.122 | 0.908 | - -> 0.908 |
| AdaLN | training | bf16 | d128 cond128 A5/A48 (atoms) | 4096 | 0.693 | 0.478 | 0.376 | 0.786 | - -> 0.786 |
| AdaLN | training | bf16 | d128 cond128 A5/A48 (atoms) | 8192 | 1.310 | 0.909 | 0.704 | 0.774 | - -> 0.774 |
| AdaLN | training | bf16 | d768 cond384 A5/A48 | 1024 | 0.885 | 0.693 | 0.633 | 0.913 | 0.917 -> 0.913 |
| AdaLN | training | bf16 | d768 cond384 A5/A48 | 384 | 0.373 | 0.293 | 0.259 | 0.883 | 0.880 -> 0.883 |
| AdaLN | training | bf16 | d768 cond384 A5/A48 | 768 | 0.692 | 0.534 | 0.486 | 0.911 | 0.912 -> 0.911 |
| AdaLN | training | bf16 | d768 cond768 A5/A48 (ESMFold2) | 128 | 0.182 | 0.157 | 0.142 | 0.901 | 0.899 -> 0.901 |
| AdaLN | training | bf16 | d768 cond768 A5/A48 (ESMFold2) | 384 | 0.494 | 0.409 | 0.366 | 0.896 | 0.895 -> 0.896 |
| AdaLN | training | bf16 | d768 cond768 A5/A48 (ESMFold2) | 768 | 0.906 | 0.757 | 0.699 | 0.923 | 0.921 -> 0.923 |
| AdaLN | training | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | 0.292 | 0.240 | 0.217 | 0.904 | 1.079 -> 0.904 |
| AdaLN | training | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | 1.048 | 0.841 | 0.724 | 0.862 | 1.034 -> 0.862 |
| AdaLN | training | fp32 | d128 cond128 A5/A48 (atoms) | 8192 | 1.983 | 1.623 | 1.399 | 0.862 | 1.038 -> 0.862 |
| AdaLN | training | fp32 | d768 cond384 A5/A48 | 128 | 0.260 | 0.230 | 0.232 | 1.008 | 1.055 -> 1.008 |
| AdaLN | training | fp32 | d768 cond384 A5/A48 | 384 | 0.639 | 0.548 | 0.549 | 1.002 | 1.022 -> 1.002 |
| AdaLN | training | fp32 | d768 cond384 A5/A48 | 768 | 1.170 | 1.027 | 1.004 | 0.978 | 0.988 -> 0.978 |
| AdaLN | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | 0.375 | 0.329 | 0.305 | 0.929 | 1.060 -> 0.929 |
| AdaLN | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | 0.938 | 0.781 | 0.758 | 0.971 | 1.065 -> 0.971 |
| AdaLN | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | 1.618 | 1.442 | 1.391 | 0.964 | 1.059 -> 0.964 |
| bias-only DiT | inference | bf16 | 12x64 | 512 | 0.283 | 0.145 | 0.125 | 0.860 | 0.862 -> 0.860 |
| bias-only DiT | inference | bf16 | 24x32 | 512 | 0.281 | 0.154 | 0.126 | 0.821 | 0.811 -> 0.821 |
| bias-only DiT | inference | bf16 | d768 16x48 A5/A48 | 256 | 0.156 | 0.096 | 0.076 | 0.786 | 0.800 -> 0.786 |
| bias-only DiT | inference | bf16 | d768 16x48 A5/A48 | 512 | 0.281 | 0.148 | 0.122 | 0.823 | 0.821 -> 0.823 |
| bias-only DiT | inference | bf16 | d768 16x48 A5/A48 | 768 | 0.429 | 0.218 | 0.173 | 0.791 | 0.792 -> 0.791 |
| bias-only DiT | inference | fp32 | d768 16x48 A5/A48 | 256 | 0.216 | 0.149 | 0.136 | 0.912 | 0.906 -> 0.912 |
| bias-only DiT | inference | fp32 | d768 16x48 A5/A48 | 512 | 0.437 | 0.247 | 0.231 | 0.934 | 0.942 -> 0.934 |
| bias-only DiT | training | bf16 | 12x64 | 512 | 3.307 | 3.037 | 2.888 | 0.951 | 0.957 -> 0.951 |
| bias-only DiT | training | bf16 | 24x32 | 512 | 3.347 | 3.058 | 3.054 | 0.999 | 0.999 -> 0.999 |
| bias-only DiT | training | bf16 | d768 16x48 A5/A48 | 256 | 1.671 | 1.544 | 1.458 | 0.944 | 0.941 -> 0.944 |
| bias-only DiT | training | bf16 | d768 16x48 A5/A48 | 512 | 3.306 | 3.037 | 2.842 | 0.936 | 0.936 -> 0.936 |
| bias-only DiT | training | bf16 | d768 16x48 A5/A48 | 768 | 5.021 | 4.576 | 4.310 | 0.942 | 0.951 -> 0.942 |
| bias-only DiT | training | fp32 | d768 16x48 A5/A48 | 256 | 3.869 | 3.384 | 3.361 | 0.993 | 0.986 -> 0.993 |
| bias-only DiT | training | fp32 | d768 16x48 A5/A48 | 512 | 7.265 | 6.370 | 6.420 | 1.008 | 1.002 -> 1.008 |
| ConditionedTransition | inference | bf16 | d768 A5/A48 | 1024 | 0.148 | 0.126 | 0.109 | 0.866 | 0.862 -> 0.866 |
| ConditionedTransition | inference | bf16 | d768 A5/A48 | 384 | 0.083 | 0.058 | 0.054 | 0.925 | 0.926 -> 0.925 |
| ConditionedTransition | inference | bf16 | d768 A5/A48 | 768 | 0.123 | 0.101 | 0.091 | 0.903 | 0.906 -> 0.903 |
| ConditionedTransition | inference | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | 0.056 | 0.042 | 0.033 | 0.785 | 0.830 -> 0.785 |
| ConditionedTransition | inference | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | 0.134 | 0.106 | 0.076 | 0.714 | 0.748 -> 0.714 |
| ConditionedTransition | inference | fp32 | d128 cond128 A5/A48 (atoms) | 8192 | 0.258 | 0.189 | 0.133 | 0.705 | 0.801 -> 0.705 |
| ConditionedTransition | inference | fp32 | d768 cond384 A5/A48 | 128 | 0.067 | 0.053 | 0.049 | 0.925 | 1.085 -> 0.925 |
| ConditionedTransition | inference | fp32 | d768 cond384 A5/A48 | 384 | 0.138 | 0.110 | 0.090 | 0.812 | 0.816 -> 0.812 |
| ConditionedTransition | inference | fp32 | d768 cond384 A5/A48 | 768 | 0.201 | 0.203 | 0.160 | 0.788 | 0.789 -> 0.788 |
| ConditionedTransition | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | 0.073 | 0.061 | 0.057 | 0.941 | 1.115 -> 0.941 |
| ConditionedTransition | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | 0.151 | 0.129 | 0.105 | 0.814 | 0.817 -> 0.814 |
| ConditionedTransition | inference | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | 0.228 | 0.239 | 0.191 | 0.798 | 0.792 -> 0.798 |
| ConditionedTransition | training | bf16 | d768 A5/A48 | 1024 | 2.976 | 2.917 | 2.822 | 0.967 | 0.967 -> 0.967 |
| ConditionedTransition | training | bf16 | d768 A5/A48 | 384 | 1.242 | 1.187 | 1.145 | 0.965 | 0.965 -> 0.965 |
| ConditionedTransition | training | bf16 | d768 A5/A48 | 768 | 2.300 | 2.235 | 2.164 | 0.968 | 0.963 -> 0.968 |
| ConditionedTransition | training | fp32 | d128 cond128 A5/A48 (atoms) | 1024 | 0.922 | 0.796 | 0.784 | 0.985 | 1.045 -> 0.985 |
| ConditionedTransition | training | fp32 | d128 cond128 A5/A48 (atoms) | 4096 | 3.321 | 2.872 | 2.761 | 0.961 | 1.015 -> 0.961 |
| ConditionedTransition | training | fp32 | d128 cond128 A5/A48 (atoms) | 8192 | 6.361 | 5.594 | 5.377 | 0.961 | 1.015 -> 0.961 |
| ConditionedTransition | training | fp32 | d768 cond384 A5/A48 | 128 | 1.183 | 1.067 | 1.031 | 0.966 | 0.977 -> 0.966 |
| ConditionedTransition | training | fp32 | d768 cond384 A5/A48 | 384 | 2.884 | 2.637 | 2.615 | 0.992 | 0.995 -> 0.992 |
| ConditionedTransition | training | fp32 | d768 cond384 A5/A48 | 768 | 5.450 | 4.968 | 4.920 | 0.990 | 0.991 -> 0.990 |
| ConditionedTransition | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 128 | 1.354 | 1.208 | 1.154 | 0.956 | 0.980 -> 0.956 |
| ConditionedTransition | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 384 | 3.324 | 3.007 | 2.957 | 0.984 | 1.011 -> 0.984 |
| ConditionedTransition | training | fp32 | d768 cond768 A5/A48 (ESMFold2) | 768 | 6.065 | 5.558 | 5.486 | 0.987 | 1.013 -> 0.987 |

CUDA < PyTorch holds in every row (the largest CUDA / PyTorch is 0.948, ConditionedTransition bf16 training L1024).

### Rows still at CUDA / portable >= 0.95 (what is left)

| rows | CUDA / portable | why |
|---|---|---|
| AdaLN training fp32 d768 cond384 L128 / L384 / L768 | 1.008 / 1.002 / 0.978 | cuBLAS TF32 GEMMs and row passes at the HBM rate on both sides; the only gap CUDA has is the weight-gradient GEMM (see below) |
| AdaLN training fp32 d768 cond768 L384 / L768 | 0.971 / 0.964 | same; L128 is 0.929 |
| ConditionedTransition training bf16 L384 / L768 / L1024 | 0.965 / 0.968 / 0.967 | the step is cuBLAS GEMMs + five elementwise passes at 86 % of the HBM rate (profile: CUDA 1153 us, 28 launches); the fused wide kernels are 37 - 52 us SLOWER (re-measured today) |
| ConditionedTransition training fp32 d768 cond384 L128 / L384 / L768 | 0.966 / 0.992 / 0.990 | as above; the three TF32 weight-gradient GEMMs run 190 TFLOP/s against 370 for the forward GEMMs (MN-major operands) |
| ConditionedTransition training fp32 d768 cond768 L128 / L384 / L768 | 0.956 / 0.984 / 0.987 | same |
| ConditionedTransition training fp32 atoms L1024 / L4096 / L8192 | 0.985 / 0.961 / 0.961 | the AdaLN part is at 0.86 now; the tail (`swiglu` / `sigmul` passes at the HBM rate + cuBLAS) is the portable rung's own cost |
| bias-only DiT training fp32 L256 / L512, bf16 24x32 L512, bf16 12x64 L512 | 0.993 / 1.008 / 0.999 / 0.951 | not touched in this round (`integrations/bias_only_dit_train.py`, its own kernels); re-measured on the combined code |

## What moved

### 1. Atom width (d = c = 128): ONE launch forward, fused backward (`adaln_sm90.cu`)
* `adaln_atom_k` replaces `ln_rows` + the wgmma GEMM + the epilogue: two warpgroups, each an independent stream of 64-row tiles, the [Wscale ; Wbias] weights resident in shared memory
  (128 KB fp32 / 64 KB bf16), cond tile (TMA) -> LayerNorm in place in the 128-B swizzled tile (two lanes a row, 16 rows a warp at once: one shuffle for each reduction) -> wgmma -> x tile TMA-loaded into the SAME
  buffer -> statistics of x, gate, modulate in place -> TMA store. Training saves LN(cond) (TMA store), the statistics and the scale pre-activation. cond is read once, x once, y once.
  fp32 inference M = 20480: 31.6 -> 22 us (portable 26.3), bf16 M = 20480: 24.4 -> 12.6 us (portable 18.6); training forward at 4096 atoms fp32 320 -> 239 us, bf16 194 -> 119 us.
  First version 1.5x slower: the LayerNorm passes were one warp a row (16 rows in sequence, two reductions of five shuffles each); the half-row-a-lane form took bf16 194 -> 128 us. One CTA a tile while
  there are fewer tiles than SMs (M = 5120: 14.5 -> 10.3 us).
* `adaln_dgrad_atom_k` (backward): dcn = [dscale | dy] [Ws ; Wb] (K = 256, the weights resident as their K-major transpose) with the LayerNorm backward of cond and the block partial of the LN-weight gradient in its
  epilogue: `dcn` never reaches HBM. fp32 at 4096 atoms: cuBLAS dgrad 107 + `ln_cond_bwd` 99 -> 147 us; bf16 119 -> 77 us. The fp32 operands are rounded to the TF32 grid to nearest (`adaln_bwd` rounds `dsb`, the
  weight pack rounds the weights): a TMA-fed fp32 operand is truncated, which measured 8.5e-4 (against 6e-6) from the cuBLAS path.
* fp32 training at the atom width now takes the same wgmma forward as bf16 (it was the row kernels + cuBLAS).
* A DEADLOCK that was already in `perf/ladder-fix-adaln-ct-cuda` is fixed: the producer asked for the x tile of the next tile at slab `KB / 2`, the consumers release the staging from slab 1 on; at `KB = 2`
  (bf16, NC = 128) the producer waited for a release that needed the slab it had not yet issued. Every bf16 atom inference row with two or more tiles a CTA (M = 5 x 4096) hung the ladder (baseline jobs 37157 / 37198
  timed out at 40 min); the guard is `KB > 2` with the request after the loop for `KB <= 2`.

### 2. fp32 token width
* One-launch weight packing (`pack_f32_k`): [Ws ; Wb], its K-major transpose, the bias and, for the ConditionedTransition tail, [Wa ; Wb]^T, Ws^T, Wsc^T in ONE kernel (RNA-rounded) instead of cat / fill / transposing-copy
  PyTorch kernels (about 40 - 60 us a step at 18432 rows); the forward hands the pack to the backward (it was packed again there).
* Feature-major backward (`adaln_bwd_t_k`, `tpose.cuh`): a TF32 weight-gradient GEMM wants the contracted dimension (the rows) contiguous in its left operand; the row pass writes [dscale ; dy]^T through 32-row shared-memory
  tiles (rotated 33-float rows: conflict-free on both sides). `dW` 228 -> 141 us at d768 / c768 / M = 18432; the pass itself is 157 against 123 us, and the dgrad GEMM with a column-major operand +8 us: AdaLN c768 L384
  840 -> 758 us. Selected for a cond of 768 columns only (at 384 the net is -15 us, measured). The same trick for the ConditionedTransition tail (`swiglu_bwd_t`, `sigmul_bwd_t`, `CT_FM=1`) is implemented and OFF: dWab 430 -> 296, dWs 237 -> 146 us
  are paid back by the transposed row passes (+67 / +24 us) and by the data-gradient GEMMs reading column-major operands (+55 / +21 us): 0 net at L384.
* fp32 inference at M <= 640 rows (L128): 32-column GEMM tiles (60 -> 120 CTAs): 10.9 -> 9.2 us, bitwise the same result.

### 3. ConditionedTransition inference at small M
The squeeze (`squeeze_gate`) had 30 tiles of 128 columns (5 row blocks x 6) on 132 SMs, each streaming K = 1920; 64-column tiles: fp32 L128 42.8 -> 34.9 us (c384), 48.5 -> 39.4 (c768), bf16 26.3 -> 21.8 us; the rule
`tiles(128) * 2 <= SMs`. (A 2-CTA cluster with multicast measured slower.)

### 4. torch.compile
The routing flags are read at import (`ADALN_ATOM`, `ADALN_FM`, `CT_FM`, `ADALN_FP32_WG`): the routing functions run inside Dynamo traces, which must not read the environment. The CT fp32 weight pack is an opaque op
(a direct call of the extension was a graph break: the compiled regime of ct_fp32_c768 ran 5 % slower and its no-compile regime failed in one ladder run).

## Tried and rejected (measured)
* Row-kernel grid size of `adaln_bwd` / `ln_cond_bwd` (1 .. 8 CTAs a SM): no change (130 us for 340 MB = 2.6 TB/s: at the rate).
* TF32 weight-gradient layouts, all variants measured at M = 18432 .. 36864 (`dW` d768 c768: row-major 246, feature-major `D` @ cn 151, `D` @ cn^T 145; c384: 128 / 112 / 111): no layout of the row-major data is fast; the K-major copy costs what it saves except at c768.
* Feature-major CT backward (above), the wide Transition kernels in the CT training tail (+37 / +52 / +44 us at L384 / 768 / 1024, bf16), TMA multicast cluster for the squeeze / expand (slower), 2-CTA tiles.
* Fusing dgrad + the LayerNorm backward at the token width: the weights (2.3 MB) do not fit shared memory, the L2 -> SM traffic of the re-read weights exceeds the saved `dcn` round trip.

## Tests
* new `tests/gpu/sm90/test_adaln_ct_cuda_widths_gpu.py` (19): atom width (inference eager / graph, training gradients, `adaln_atom_k` / `adaln_dgrad_atom_k` launched, one launch not two, tail rows 3 x 700), the 768-wide fp32 cond (`adaln_bwd_t_k`,
  `pack_f32_k`), the small-M squeeze tile, the 32-column tile: fp64 parity within 1.5x the PyTorch error for the output and every gradient, bitwise repeat.
* H100 run on the tip: `test_adaln_ct_cuda_gpu` (24) + the new file (19): 43 passed; `test_engine_adaln_alignment`, `test_portable_ct_adaln_residual`, `test_conditioned_transition_precision`, `test_determinism`: 60 passed, 1 skipped;
  `test_h100_bias_only_dit`, `test_bias_only_dit_psb`, `test_h100_module_wiring`, `test_h100_weight_pack`: 203 passed, 36 skipped; `test_token_dit_*`, `test_dit_variant_train`, `test_pair_embed_adaln`, `test_bias_only_dit_portable`,
  `test_bodit_ct_training_portable`: those runs hit the 24 min job cap at about 70 % (no failure before it; not repeated: they do not import the changed modules). CPU suite: the same 2 failures + 2 errors as the base (`test_key_flags_are_built_on_both_sides`, `test_no_stale_caches`, `test_a_ladder_offers_what_wins`);
  `dev derive --arch sm86` re-run (registry units / csv in this commit).
