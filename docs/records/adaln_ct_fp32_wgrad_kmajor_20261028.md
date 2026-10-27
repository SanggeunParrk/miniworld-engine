# AdaLN / ConditionedTransition H100 fp32 training: K-major operand for the TF32 weight-gradient GEMMs (2026-10-28)

Branch `perf/final-adaln-wgrad` (from `perf/final-adalnct-cuda` ef3e2c3d). One H100 80GB HBM3, torch 2.13 + cu129, `benchmarks/runners/ladder.py` (`--kinds rungs`, medians of 3,
+-1-3 % noise), same protocol as `adaln_ct_cuda_ladder2_20261028.md`. "Before" = the ratio column of that record.

## Result (ms; PyTorch = the fastest of its forms)

| op | mode | dtype | shape | L | PyTorch | portable | CUDA | CUDA / portable before -> after |
|---|---|---|---|---|---|---|---|---|
| AdaLN | training | fp32 | d768 cond384 | 128 | 0.261 | 0.231 | 0.224 | 1.008 -> 0.969 |
| AdaLN | training | fp32 | d768 cond384 | 384 | 0.641 | 0.549 | 0.532 | 1.002 -> 0.969 |
| AdaLN | training | fp32 | d768 cond384 | 768 | 1.170 | 1.019 | 1.005 | 0.978 -> 0.986 (route unchanged: see below) |
| AdaLN | training | fp32 | d768 cond768 | 128 | 0.373 | 0.327 | 0.307 | 0.929 -> 0.938 (unchanged route, noise) |
| AdaLN | training | fp32 | d768 cond768 | 384 | 0.933 | 0.781 | 0.756 | 0.971 -> 0.968 (unchanged route) |
| AdaLN | training | fp32 | d768 cond768 | 768 | 1.618 | 1.438 | 1.392 | 0.964 -> 0.968 (unchanged route) |
| ConditionedTransition | training | fp32 | d768 cond384 | 128 | 1.189 | 1.066 | 0.977 | 0.966 -> 0.916 |
| ConditionedTransition | training | fp32 | d768 cond384 | 384 | 2.889 | 2.634 | 2.460 | 0.992 -> 0.934 |
| ConditionedTransition | training | fp32 | d768 cond384 | 768 | 5.443 | 4.973 | 4.703 | 0.990 -> 0.946 |
| ConditionedTransition | training | fp32 | d768 cond768 | 128 | 1.347 | 1.212 | 1.107 | 0.956 -> 0.914 |
| ConditionedTransition | training | fp32 | d768 cond768 | 384 | 3.310 | 3.006 | 2.829 | 0.984 -> 0.941 |
| ConditionedTransition | training | fp32 | d768 cond768 | 768 | 6.066 | 5.554 | 5.282 | 0.987 -> 0.951 |
| ConditionedTransition | training | fp32 | d128 atoms | 1024 / 4096 / 8192 | 0.923 / 3.320 / 6.359 | 0.797 / 2.880 / 5.615 | 0.784 / 2.760 / 5.377 | 0.985 / 0.961 / 0.961 -> 0.984 / 0.958 / 0.958 (unchanged route) |
| ConditionedTransition | training | bf16 | d768 | 384 / 768 / 1024 | 1.242 / 2.300 / 2.975 | 1.187 / 2.242 / 2.910 | 1.147 / 2.164 / 2.826 | 0.965 / 0.968 / 0.967 -> 0.966 / 0.965 / 0.971 (unchanged code, noise) |

CUDA < PyTorch holds in every row (largest CUDA / PyTorch: 0.950, CT bf16 L1024). Eight rows moved by more than the noise (AdaLN cond384 L128 / L384, CT fp32 token widths); the rest are unchanged routes re-measured.

## What was found

Profiles of the real step (CUDA graph, cold L2, `prof_train.py`) and a layout sweep of the three CT weight-gradient GEMMs at M = 18432 (cuBLAS TF32, `G^T X` with G, X row-major [M, P], [M, Q]):

| GEMM | both operands MN-major (what the code ran) | left K-major | right K-major (`G^T @ XT^T`) | both K-major |
|---|---|---|---|---|
| dWab 3072 x 768 | 432 us (201 TFLOP/s) | 343 | 301 (289) | 294 (296) |
| dWs 768 x 1536 | 237 (184) | 150 | 147 (295) | 146 (298) |
| AdaLN dW 1536 x 384 | 147 (148) | | 97 (224) | |
| AdaLN dW 1536 x 768 | 227 (191) | | 144 (303) | |

A cuBLAS TF32 GEMM reaches ~295 TFLOP/s as soon as ONE operand is K-major (rows contiguous), not both; the big gradient can stay as it is. So the cheap lever is a K-major copy of the
SMALLER operand, produced where the data is already in registers or shared memory:

* CT tail backward (`ct_train.py`): `dWab = dab^T x` with `x^T` (`transpose_f32`, one `pack_f32` launch, 43 us for 56 MB) as the right operand: 430 -> 296 us. `dWs = dout^T h` with
  `dout^T` as the K-major LEFT operand, written by the gate-backward row pass itself (`sigmul_bwd_x_k`, new in `ct_rows.cu`: the row-major `dout`, `dscale` and the column-sum partials as before plus the
  feature-major `dout` through the 32-row shared-memory tiles of `tpose.cuh`; 21 us more than `sigmul_bwd_k`): 237 -> 150 us, straight out in [D, ND] (no transposing copy after it).
  Gated to token widths (x.shape[1] >= 256): at the atom width (d = 128) the copy costs more than it saves (measured +13 us at 4096 atoms).
* AdaLN backward (`adaln/cuda/__init__.py`), fp32, cond width 256 / 384 / 512, `M <= 24576`: `ln_cond_fwd_t_k` (new in `adaln_rows.cu`) writes LN(cond) and, through shared-memory tiles, LN(cond)^T; the weight gradient is
  `dsb^T @ (cn^T)^T` = [2 d, dc] directly (the old `mm(cn.t(), dsb).t()` needed a 6.8 us transposing copy after the 112 us GEMM): GEMM 119 -> 90 us, the forward row pass 21 -> 39 us.
  At M = 36864 cuBLAS picks a slower kernel for this layout (dW 222 against 197 us) and the transposed row pass costs 68 against 40 us: measured 1050 against 1008 us, so the route stops at 24576 rows (L768 keeps the old path).
  The 768-wide cond keeps its feature-major route (`adaln_bwd_t_k`): the new route would save ~5 us there at most.

Numerics: the transposed operands are exact copies (`transpose_f32` rounds to the TF32 grid to nearest like the other packed fp32 operands; `ln_cond_fwd_t_k` copies without rounding, as `cn` itself); the
fixed-order block partials are unchanged, the GEMM accumulation order differs only inside cuBLAS. The tests below check module error against fp64 <= 1.5x the PyTorch TF32 error for the output and every gradient and a bitwise repeat.

## Looked at and not done

* A hand wgmma TF32 weight-gradient kernel (loading MN-major tiles, transposing in shared memory): the cuBLAS K-major kernels already run at 295 TFLOP/s; an estimate of the shared-memory and L2 traffic
  of a 128 x 192 tile with software transposition puts such a kernel at about 250 TFLOP/s without cluster multicast. Not built; the K-major copy gets the cuBLAS rate.
* cublasLt algorithm sweep for the MN-major GEMMs (24 heuristic candidates, dWab / dWs / dWsc / AdaLN dW): the best candidate is 1 - 8 % faster than the default (dWs 217 against 236 us), nothing like the 2.0 - 3.5x of the layout change.
* AdaLN dcn from two GEMMs (`dscale Ws + dy Wb`) to avoid writing the `dy` half of `dsb`: the partial results cost as many bytes as the copy saves.
* Remaining after this round (CUDA / portable >= 0.95): AdaLN fp32 cond384 (0.969 / 0.969 / 0.986: the row passes are at the HBM rate; L768 would need a better dW kernel choice), AdaLN fp32 cond768 (0.938 / 0.968 / 0.968),
  CT fp32 cond768 L768 (0.951), CT fp32 atoms (0.984 / 0.958 / 0.958), CT bf16 (0.966 / 0.965 / 0.971: bf16 wgrad GEMMs are already at ~800 TFLOP/s; what is left is ~40 us of launch gaps and five elementwise passes at the HBM rate).
  Next candidates: x^T written by the AdaLN forward row pass (saves the 43 us transposition), PDL between the custom row kernels and their cuBLAS neighbours (~1 us per launch).

## Tests (H100)

`test_adaln_ct_cuda_gpu` + `test_adaln_ct_cuda_widths_gpu` (now with the 384-wide cond: `ln_cond_fwd_t_k` / `sigmul_bwd_x_k` launched, fp64 parity, bitwise repeat) +
`test_engine_adaln_alignment` + `test_portable_ct_adaln_residual_gpu` + `test_conditioned_transition_precision_gpu` + `test_determinism_gpu`: 112 passed, 1 skipped.
`dev derive --arch sm86` re-run (units file regenerated by the tool).
