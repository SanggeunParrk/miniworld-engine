# float32 registry rows: ladder groups, H100 results and fixes (2026-10-27)

Branch `perf/final-fp32-rows` (from `integ/ladder-all2` b93a655e, plus the registry dtype patch: single-direction TriMul rows bf16 only; the four conditioning
Transition rows d_hidden = 128 / 256 / 384 / 768, n = 2 gain float32). Rule: hand CUDA < portable (Triton) < PyTorch (best of compile + graph, graph-eager, eager); fp32 = TF32 tensor cores.
Timings: `benchmarks/runners/ladder.py` (compile + CUDA graph, median of 3), H100 80GB HBM3. "c/p" = CUDA / portable, "p/t" = portable / PyTorch (both must be < 1).

## Ladder groups (the registry rows are the contract)

Added (`benchmarks/runners/ladder.py`; `bench.py` gained `transition_n` and `transition_stream` = pair | single):

| registry row | group |
|---|---|
| transition token_pair d128 n2 / d256 n2 | `trc_fp32_pair_D128`, `trc_fp32_pair_D256` (L 128, 384, 768) |
| transition token_single d384 n2 / d768 n2 | `trc_fp32_single_D384`, `trc_fp32_single_D768` |
| adaptive_layernorm token d768 c768 / atom d128 c128 | `adaln_fp32_c768`, `adaln_fp32_atom` (`adaln_fp32` = d768 c384, L now 128 / 384 / 768) |
| conditioned_transition token d768 c768 / atom d128 c128 | `ct_fp32_c768`, `ct_fp32_atom` (`ct_fp32` likewise) |
| augmented_attention token d768 c768 pair256 / atom d128 pair16 | `aat_fp32_c768`, `aaa_fp32` (`aat_fp32` = c384 pair128; L 128 / 384 / 768), and `aat_bf16_c768` for the bf16 twin |
| token DiT / atom DiT fp32 | `dit_fp32`, `local_fp32` (already there; confirmed) |

The atom lengths of the registry (1024 .. 8192) are run at 1024 / 4096 / 8192 for adaLN and the conditioned transition; the dense atom attention is run at 128 / 384 / 1024 (its [A, N, N] logits do not fit
the GPU in any rung at N = 4096).

## Fixes

* Transition, fp32 forward at >= 16 384 rows (every pair-stream row with L >= 128): the fp16-intermediate path computed its range scales with ~20 small torch kernels (~40 us of a ~35 us call).
  Now `h16_scales` (memset + `h16_absmax_kernel` + `h16_apply_kernel`, `generic_sm90a/bind.cu`). trc pair D128 L128 inference c/p 1.57 -> 0.92, D256 1.12 -> 0.79.
* AugmentedAttentionPairBias fp32 on the TF32 core (`fused_pair_tf32.py`, `pair_rows.cu`): layouts 16 x 48 with d_pair 256 (ESMFold2 token DiT) and the dense atom stack 4 x 32 with d_pair 16 had no CUDA path
  (the PyTorch reference ran). New generic pair->bias pass (`pair_bias_rows32`: any of the three layouts, key-permuted, masked, small-grid variant), `bwd256_k` (two warps per row: one warp spilled 40x),
  `atom32_fwd_k` / `atom32_bwd_k`, `dd_k` templated on (heads, width). d_pair 128 is untouched.

## Results (final tree; ms)

### Transition (trunk n=4 `tr_fp32_*`, conditioning n=2 `trc_fp32_*`)
| group | mode | L | PyTorch ms | portable ms | CUDA ms | CUDA/portable | portable/PyTorch | verdict |
|---|---|---|---|---|---|---|---|---|
| tr_fp32_D128 | infer | 384 | 0.8447 | 0.47226 | 0.32782 | 0.694 | 0.559 | OK |
| tr_fp32_D128 | infer | 768 | 3.29126 | 1.76357 | 1.21157 | 0.687 | 0.536 | OK |
| tr_fp32_D128 | train | 384 | 2.3665 | 1.8713 | 1.56106 | 0.834 | 0.791 | OK |
| tr_fp32_D128 | train | 768 | 9.13859 | 7.15165 | 6.03654 | 0.844 | 0.783 | OK |
| tr_fp32_D256 | infer | 384 | 1.69232 | 1.1916 | 0.8404 | 0.705 | 0.704 | OK |
| tr_fp32_D256 | train | 384 | 4.94709 | 5.06112 | 3.87318 | 0.765 | 1.023 | VIOLATION(portable_vs_pytorch) |
| trc_fp32_pair_D128 | infer | 128 | 0.06571 | 0.04611 | 0.04237 | 0.919 | 0.702 | OK |
| trc_fp32_pair_D128 | infer | 384 | 0.54277 | 0.32333 | 0.22816 | 0.706 | 0.596 | OK |
| trc_fp32_pair_D128 | infer | 768 | 2.06128 | 1.20006 | 0.82008 | 0.683 | 0.582 | OK |
| trc_fp32_pair_D128 | train | 128 | 0.19808 | 0.17376 | 0.14486 | 0.834 | 0.877 | OK |
| trc_fp32_pair_D128 | train | 384 | 1.48214 | 1.14422 | 0.98736 | 0.863 | 0.772 | OK |
| trc_fp32_pair_D128 | train | 768 | 5.53136 | 4.28597 | 3.75994 | 0.877 | 0.775 | OK |
| trc_fp32_pair_D256 | infer | 128 | 0.13082 | 0.1027 | 0.0807 | 0.786 | 0.785 | OK |
| trc_fp32_pair_D256 | infer | 384 | 1.06211 | 0.75475 | 0.52622 | 0.697 | 0.711 | OK |
| trc_fp32_pair_D256 | infer | 768 | 4.14723 | 2.9008 | 2.03514 | 0.702 | 0.699 | OK |
| trc_fp32_pair_D256 | train | 128 | 0.42246 | 0.39632 | 0.30914 | 0.780 | 0.938 | OK |
| trc_fp32_pair_D256 | train | 384 | 3.22538 | 2.74147 | 2.19162 | 0.799 | 0.850 | OK |
| trc_fp32_pair_D256 | train | 768 | 12.29266 | 10.52659 | 8.62694 | 0.820 | 0.856 | OK |
| trc_fp32_single_D384 | infer | 128 | 0.02688 | 0.02851 | 0.02486 | 0.872 | 1.061 | VIOLATION(portable_vs_pytorch) |
| trc_fp32_single_D384 | infer | 384 | 0.02934 | 0.02851 | 0.02525 | 0.886 | 0.972 | OK |
| trc_fp32_single_D384 | infer | 768 | 0.03062 | 0.0312 | 0.02566 | 0.823 | 1.019 | VIOLATION(portable_vs_pytorch) |
| trc_fp32_single_D384 | train | 128 | 0.06109 | 0.08102 | 0.0633 | 0.781 | 1.326 | VIOLATION(portable_vs_pytorch) |
| trc_fp32_single_D384 | train | 384 | 0.08112 | 0.0937 | 0.07069 | 0.754 | 1.155 | VIOLATION(portable_vs_pytorch) |
| trc_fp32_single_D384 | train | 768 | 0.10146 | 0.11738 | 0.08109 | 0.691 | 1.157 | VIOLATION(portable_vs_pytorch) |
| trc_fp32_single_D768 | infer | 128 | 0.03322 | 0.04054 | 0.04422 | 1.091 | 1.221 | VIOLATION(cuda_vs_portable)+VIOLATION(portable_vs_pytorch) |
| trc_fp32_single_D768 | infer | 384 | 0.04234 | 0.04541 | 0.04512 | 0.994 | 1.073 | VIOLATION(portable_vs_pytorch) |
| trc_fp32_single_D768 | infer | 768 | 0.05165 | 0.05248 | 0.05514 | 1.051 | 1.016 | VIOLATION(cuda_vs_portable)+VIOLATION(portable_vs_pytorch) |
| trc_fp32_single_D768 | train | 128 | 0.08701 | 0.12464 | 0.1183 | 0.949 | 1.433 | VIOLATION(portable_vs_pytorch) |
| trc_fp32_single_D768 | train | 384 | 0.13194 | 0.16637 | 0.13885 | 0.835 | 1.261 | VIOLATION(portable_vs_pytorch) |
| trc_fp32_single_D768 | train | 768 | 0.18315 | 0.22397 | 0.19626 | 0.876 | 1.223 | VIOLATION(portable_vs_pytorch) |
### AdaptiveLayerNorm (other agent: kernels/adaln; numbers only)
| group | mode | L | PyTorch ms | portable ms | CUDA ms | CUDA/portable | portable/PyTorch | verdict |
|---|---|---|---|---|---|---|---|---|
| adaln_fp32 | infer | 128 | 0.0224 | 0.0161 | 0.01542 | 0.958 | 0.719 | OK |
| adaln_fp32 | infer | 384 | 0.03437 | 0.02525 | 0.02355 | 0.933 | 0.735 | OK |
| adaln_fp32 | infer | 768 | 0.05037 | 0.044 | 0.04045 | 0.919 | 0.874 | OK |
| adaln_fp32 | train | 128 | 0.25901 | 0.25642 | 0.2447 | 0.954 | 0.990 | OK |
| adaln_fp32 | train | 384 | 0.63979 | 0.59357 | 0.55699 | 0.938 | 0.928 | OK |
| adaln_fp32 | train | 768 | 1.16739 | 1.13648 | 1.00394 | 0.883 | 0.974 | OK |
| adaln_fp32_c768 | infer | 128 | 0.02624 | 0.02147 | 0.02061 | 0.960 | 0.818 | OK |
| adaln_fp32_c768 | infer | 384 | 0.04381 | 0.03715 | 0.0337 | 0.907 | 0.848 | OK |
| adaln_fp32_c768 | infer | 768 | 0.0703 | 0.06592 | 0.05942 | 0.901 | 0.938 | OK |
| adaln_fp32_c768 | train | 128 | 0.3707 | 0.36202 | 0.34722 | 0.959 | 0.977 | OK |
| adaln_fp32_c768 | train | 384 | 0.93341 | 0.88352 | 0.83093 | 0.940 | 0.947 | OK |
| adaln_fp32_c768 | train | 768 | 1.60568 | 1.60941 | 1.50634 | 0.936 | 1.002 | VIOLATION(portable_vs_pytorch) |
| adaln_fp32_atom | infer | 1024 | 0.02186 | 0.01722 | 0.01552 | 0.901 | 0.788 | OK |
| adaln_fp32_atom | infer | 4096 | 0.0447 | 0.03501 | 0.03792 | 1.083 | 0.783 | VIOLATION(cuda_vs_portable) |
| adaln_fp32_atom | infer | 8192 | 0.08173 | 0.06288 | 0.06966 | 1.108 | 0.769 | VIOLATION(cuda_vs_portable) |
| adaln_fp32_atom | train | 1024 | 0.29242 | 0.3016 | 0.26051 | 0.864 | 1.031 | VIOLATION(portable_vs_pytorch) |
| adaln_fp32_atom | train | 4096 | 1.04794 | 1.09982 | 0.87475 | 0.795 | 1.050 | VIOLATION(portable_vs_pytorch) |
| adaln_fp32_atom | train | 8192 | 1.98038 | 2.16451 | 1.68475 | 0.778 | 1.093 | VIOLATION(portable_vs_pytorch) |
### ConditionedTransition (other agent: kernels/cond_transition; numbers only)
| group | mode | L | PyTorch ms | portable ms | CUDA ms | CUDA/portable | portable/PyTorch | verdict |
|---|---|---|---|---|---|---|---|---|
| ct_fp32 | infer | 128 | 0.0673 | 0.05306 | 0.05542 | 1.045 | 0.788 | VIOLATION(cuda_vs_portable) |
| ct_fp32 | infer | 384 | 0.13818 | 0.11275 | 0.08928 | 0.792 | 0.816 | OK |
| ct_fp32 | infer | 768 | 0.20045 | 0.2015 | 0.16797 | 0.834 | 1.005 | VIOLATION(portable_vs_pytorch) |
| ct_fp32 | train | 128 | 1.18939 | 1.14429 | 1.04186 | 0.910 | 0.962 | OK |
| ct_fp32 | train | 384 | 2.88493 | 2.79173 | 2.61842 | 0.938 | 0.968 | OK |
| ct_fp32 | train | 768 | 5.44214 | 5.30894 | 4.90854 | 0.925 | 0.976 | OK |
| ct_fp32_c768 | infer | 128 | 0.07235 | 0.0607 | 0.06541 | 1.077 | 0.839 | VIOLATION(cuda_vs_portable) |
| ct_fp32_c768 | infer | 384 | 0.15005 | 0.13072 | 0.10426 | 0.798 | 0.871 | OK |
| ct_fp32_c768 | infer | 768 | 0.22755 | 0.23818 | 0.19722 | 0.828 | 1.047 | VIOLATION(portable_vs_pytorch) |
| ct_fp32_c768 | train | 128 | 1.35251 | 1.29667 | 1.19634 | 0.923 | 0.959 | OK |
| ct_fp32_c768 | train | 384 | 3.31133 | 3.20066 | 3.03366 | 0.948 | 0.967 | OK |
| ct_fp32_c768 | train | 768 | 6.06341 | 5.92752 | 5.60534 | 0.946 | 0.978 | OK |
| ct_fp32_atom | infer | 1024 | 0.05501 | 0.04178 | 0.03459 | 0.828 | 0.759 | OK |
| ct_fp32_atom | infer | 4096 | 0.13507 | 0.10624 | 0.08368 | 0.788 | 0.787 | OK |
| ct_fp32_atom | infer | 8192 | 0.25722 | 0.18934 | 0.1591 | 0.840 | 0.736 | OK |
| ct_fp32_atom | train | 1024 | 0.9227 | 0.8929 | 0.83285 | 0.933 | 0.968 | OK |
| ct_fp32_atom | train | 4096 | 3.31706 | 3.35133 | 2.91782 | 0.871 | 1.010 | VIOLATION(portable_vs_pytorch) |
| ct_fp32_atom | train | 8192 | 6.35917 | 6.51891 | 5.66653 | 0.869 | 1.025 | VIOLATION(portable_vs_pytorch) |
### AugmentedAttention token (`aat_fp32` c384 pair128, `aat_fp32_c768` pair256)
| group | mode | L | PyTorch ms | portable ms | CUDA ms | CUDA/portable | portable/PyTorch | verdict |
|---|---|---|---|---|---|---|---|---|
| aat_fp32 | infer | 128 | 0.10474 | 0.09069 | 0.08723 | 0.962 | 0.866 | OK |
| aat_fp32 | infer | 384 | 0.33507 | 0.27186 | 0.19478 | 0.717 | 0.811 | OK |
| aat_fp32 | infer | 768 | 1.03562 | 0.73158 | 0.4327 | 0.591 | 0.706 | OK |
| aat_fp32 | train | 128 | 1.64314 | 1.50592 | 1.55686 | 1.034 | 0.916 | VIOLATION(cuda_vs_portable) |
| aat_fp32 | train | 384 | 6.16368 | 5.17493 | 4.69656 | 0.908 | 0.840 | OK |
| aat_fp32 | train | 768 | 18.58688 | 13.65974 | 10.99206 | 0.805 | 0.735 | OK |
| aat_fp32_c768 | infer | 128 | 0.12061 | 0.10736 | 0.10704 | 0.997 | 0.890 | OK |
| aat_fp32_c768 | infer | 384 | 0.42186 | 0.3629 | 0.28722 | 0.791 | 0.860 | OK |
| aat_fp32_c768 | infer | 768 | 1.35805 | 1.05802 | 0.74029 | 0.700 | 0.779 | OK |
| aat_fp32_c768 | train | 128 | 1.8409 | 1.68942 | 1.72317 | 1.020 | 0.918 | VIOLATION(cuda_vs_portable) |
| aat_fp32_c768 | train | 384 | 6.80702 | 5.7825 | 5.31056 | 0.918 | 0.849 | OK |
| aat_fp32_c768 | train | 768 | 20.06219 | 15.11344 | 12.46285 | 0.825 | 0.753 | OK |
### AugmentedAttention atom (`aaa_fp32`; last training row: PyTorch out of memory)
| group | mode | L | PyTorch ms | portable ms | CUDA ms | CUDA/portable | portable/PyTorch | verdict |
|---|---|---|---|---|---|---|---|---|
| aaa_fp32 | infer | 128 | 0.37774 | 0.25995 | 0.13165 | 0.506 | 0.688 | OK |
| aaa_fp32 | infer | 384 | 2.9081 | 1.79597 | 0.63158 | 0.352 | 0.618 | OK |
| aaa_fp32 | infer | 1024 | 19.7601 | 11.96728 | 3.73501 | 0.312 | 0.606 | OK |
| aaa_fp32 | train | 128 | 6.01315 | 3.6377 | 2.92874 | 0.805 | 0.605 | OK |
| aaa_fp32 | train | 384 | 49.67064 | 24.99322 | 19.59379 | 0.784 | 0.503 | OK |
| aaa_fp32 | train | 1024 |  | 172.37132 | 125.67847 | 0.729 | 0.000 | OK |
### AugmentedAttention token bf16 d768 c768 pair256 (no CUDA core for 16 heads x 48 with d_pair 256 in bf16: `fused_pair.LAYOUTS`)
| group | mode | L | PyTorch ms | portable ms | CUDA ms | CUDA/portable | portable/PyTorch | verdict |
|---|---|---|---|---|---|---|---|---|
| aat_bf16_c768 | infer | 128 | 0.10045 | 0.08666 | 0.0943 | 1.088 | 0.863 | VIOLATION(cuda_vs_portable) |
| aat_bf16_c768 | infer | 384 | 0.25486 | 0.20298 | 0.26352 | 1.298 | 0.796 | VIOLATION(cuda_vs_portable) |
| aat_bf16_c768 | infer | 768 | 0.71114 | 0.53203 | 0.70458 | 1.324 | 0.748 | n.a.(cuda rung missing) |
| aat_bf16_c768 | train | 128 | 0.944 | 0.84176 | 0.89349 | 1.061 | 0.892 | OK |
| aat_bf16_c768 | train | 384 |  | 2.95008 | 2.81674 | 0.955 | 0.341 | OK |
| aat_bf16_c768 | train | 768 |  | 8.24678 | 6.65226 | 0.807 | 0.305 | OK |
### Diffusion transformer blocks (token DiT, atom DiT)
| group | mode | L | PyTorch ms | portable ms | CUDA ms | CUDA/portable | portable/PyTorch | verdict |
|---|---|---|---|---|---|---|---|---|
| dit_fp32 | infer | 384 | 0.474 | 0.38064 | 0.1977 | 0.519 | 0.803 | OK |
| dit_fp32 | train | 384 | 9.14626 | 8.06803 | 7.17827 | 0.890 | 0.882 | OK |
| local_fp32 | infer | 512 | 0.53411 | 0.3737 | 0.15242 | 0.408 | 0.700 | OK |
| local_fp32 | train | 512 | 12.4377 | 10.49744 | 5.77872 | 0.550 | 0.844 | OK |

## Open

* Not mine to edit (reported): adaLN and ConditionedTransition have no H100 CUDA path of their own in the standalone modules; every violation in their tables is theirs (kernels/adaln, kernels/cond_transition).
* Owned by the Triton TMA-removal agent (kernels/transition/triton): `tr_fp32_D256` training portable / PyTorch 1.023, the whole `trc_fp32_single_*` portable / PyTorch column (tiny single-stream rows, 25 - 220 us),
  `ct_fp32` training.
* CUDA vs portable, Transition single stream D768 inference (L128 1.09, L768 1.05): the TF32 weight copies (`prep`, 13 us of 40) cost more than the portable kernel's in-kernel rounding at 128 rows. A cache of the rounded
  weights keyed on (pointer, version) would remove it but goes stale under CUDA-graph replay after an in-place weight update; not done.
* AugmentedAttention token fp32 training at L128, A = 48: c/p 1.034 (d_pair 128) and 1.020 (d_pair 256); the TF32 path spends ~50 us more in fp32 glue passes (q/k/v rounding+transpose, 3 transposes, dO rounding, ~190 MB at 3 TB/s)
  than the Triton one. L384 / L768 are 0.91 / 0.81.
* bf16 `aat_bf16_c768` (d_pair 256, 16 heads): the bf16 CUDA core has no 16 x 48 / 256 layout (inference c/p 1.09 - 1.32, L768 no CUDA rung). Not in this task (bf16).
