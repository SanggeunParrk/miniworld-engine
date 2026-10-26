# Portable Transition against PyTorch on H100: ladder before / after (2026-10-26)

Goal: remove the `portable_vs_pytorch` violations of the Transition rows of the performance ladder (portable = Triton family, reached on
H100 only through `settings.force_portable_kernels`; the hand-CUDA path and its H100 default are untouched).

**Result: every Transition row now has portable < PyTorch except `tr_fp32_D256` training (1.033, was 1.065). Inference rows are
0.48-0.85 of PyTorch; bf16 training rows 0.84-0.98 (0.92-0.98 at D256-D768). The CUDA rung is faster than portable everywhere (ratio
0.45-0.997).** Near-misses still >= 0.95: training D768 0.959, D512 0.976, D384 L768 0.980, D128 L128 0.959, and the cuda/portable tie
`tr_fp32_D128` inference (0.961 at L384, 0.997 at L768: the CUDA rung did not get slower, portable got faster).

Measured on one H100 80GB HBM3 (node02), torch 2.13.0+cu129, `benchmarks/runners/ladder.py run --group '^(tr_)' --repeats 3` at the commit that
carries this record (before: `~/miniworld-engine-scratch/ladder/out_full_A`, the ladder run at 12ae4ac8). ratio = portable / best PyTorch
(`pt_compiled` in every row), must be < 1. Times in ms, CUDA-graph replay.

| group | mode | L | PT best ms | portable before | ratio before | portable after | ratio after | CUDA before | CUDA after | cuda/portable after | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tr_D128 | inference | 128 | 0.05715 | 0.04013 | 0.70213 | 0.04454 | 0.7794 | 0.02256 | 0.02259 | 0.50718 | OK |
| tr_D128 | inference | 384 | 0.45178 | 0.28403 | 0.62881 | 0.24995 | 0.55327 | 0.1127 | 0.11261 | 0.45052 | OK |
| tr_D128 | inference | 768 | 1.71098 | 1.0808 | 0.63166 | 0.89888 | 0.52536 | 0.4089 | 0.40829 | 0.45422 | OK |
| tr_D128 | inference | 1024 | 3.03262 | 1.91342 | 0.63096 | 1.58576 | 0.5229 | 0.74077 | 0.72378 | 0.45642 | OK |
| tr_D128 | training | 128 | 0.15994 | 0.15184 | 0.9456 | 0.15331 | 0.95858 | 0.0855 | 0.08539 | 0.55698 | OK |
| tr_D128 | training | 384 | 1.21163 | 1.05102 | 0.86663 | 1.01192 | 0.83517 | 0.54557 | 0.53846 | 0.53212 | OK |
| tr_D128 | training | 768 | 4.51293 | 3.99142 | 0.88427 | 3.88973 | 0.86191 | 2.14944 | 2.10558 | 0.54132 | OK |
| tr_D128 | training | 1024 | 7.95598 | 7.04643 | 0.88557 | 6.88837 | 0.86581 | 3.79789 | 3.75456 | 0.54506 | OK |
| tr_D256 | inference | 384 | 0.88166 | 0.81725 | 0.92677 | 0.59654 | 0.67661 | 0.40698 | 0.39917 | 0.66913 | OK |
| tr_D256 | inference | 768 | 3.40438 | 3.3352 | 0.97701 | 2.30829 | 0.67803 | 1.59293 | 1.59942 | 0.6929 | OK |
| tr_D256 | training | 384 | 2.34179 | 2.57963 | 1.10389 | 2.15536 | 0.92039 | 1.84368 | 1.82792 | 0.84808 | OK |
| tr_D256 | training | 768 | 9.01984 | 10.16478 | 1.127 | 8.3735 | 0.92834 | 7.16554 | 7.0272 | 0.83922 | OK |
| tr_D384 | inference | 384 | 1.52818 | 1.59302 | 1.04067 | 1.12504 | 0.7362 | 0.96509 | 0.95942 | 0.85279 | OK |
| tr_D384 | inference | 768 | 5.96987 | 6.26278 | 1.05021 | 4.37186 | 0.73232 | 3.78282 | 3.73952 | 0.85536 | OK |
| tr_D384 | training | 384 | 4.12429 | 4.95818 | 1.21326 | 3.8735 | 0.93919 | 3.6375 | 3.63248 | 0.93778 | OK |
| tr_D384 | training | 768 | 15.86875 | 19.93955 | 1.24285 | 15.55474 | 0.98021 | 13.65914 | 13.46355 | 0.86556 | OK |
| tr_D512 | inference | 384 | 2.29194 | 2.61818 | 1.11877 | 1.81126 | 0.79028 | 1.58691 | 1.56595 | 0.86456 | OK |
| tr_D512 | training | 384 | 6.21974 | 7.87082 | 1.26959 | 6.06878 | 0.97573 | 6.02837 | 5.70885 | 0.94069 | OK |
| tr_D768 | inference | 384 | 4.35306 | 5.4025 | 1.24015 | 3.68541 | 0.84663 | 3.37702 | 3.32994 | 0.90355 | OK |
| tr_D768 | training | 384 | 12.29586 | 16.56752 | 1.35052 | 11.79445 | 0.95922 | 11.02454 | 10.90886 | 0.92492 | OK |
| tr_fp32_D128 | inference | 384 | 0.84464 | 0.48819 | 0.57718 | 0.42227 | 0.49994 | 0.40714 | 0.40574 | 0.96086 | OK |
| tr_fp32_D128 | inference | 768 | 3.27888 | 1.86707 | 0.56781 | 1.59171 | 0.48544 | 1.58835 | 1.58717 | 0.99715 | OK |
| tr_fp32_D128 | training | 384 | 2.36624 | 1.98643 | 0.83729 | 1.89242 | 0.79976 | 1.63805 | 1.63869 | 0.86592 | OK |
| tr_fp32_D128 | training | 768 | 9.13531 | 7.74194 | 0.84653 | 7.31245 | 0.80046 | 6.4143 | 6.42365 | 0.87845 | OK |
| tr_fp32_D256 | inference | 384 | 1.69128 | 1.27469 | 0.75195 | 1.13478 | 0.67096 | 0.99043 | 0.97104 | 0.8557 | OK |
| tr_fp32_D256 | training | 384 | 4.93187 | 5.25474 | 1.06477 | 5.09686 | 1.03345 | 4.04851 | 3.9752 | 0.77993 | VIOLATION(portable_vs_pytorch) |

## What changed (all in `kernels/transition/triton/`)

* `main.py` `transition_fwd_kernel`: the two expand weights are row-interleaved (`torch.stack((wa, wb), 1)`, ~10 MB, a few us), so a/b come out of
  ONE dot of N = 2*BLOCK_N (`tl.split` in the epilogue); on sm_90 with aligned operands the kernel runs a persistent grid with in-kernel tensor maps
  (`USE_TMA`, `_tma.py`) and `flatten=True`. The pointer kernel is unchanged in arithmetic and still serves every other card, ragged widths and
  sliced operands. `MINIWORLD_PORTABLE_NO_TMA=1` forces it (A/B and tests). `STORE_AB` (training forward) also stores a and b in the activation dtype.
* `residual.py` `_squeeze_residual_kernel`: same two modes (tensor-map loads/stores, the residual through a map).
* `training.py` (new) `_squeeze_swiglu_bwd_kernel`, registry row `transition_bwd_squeeze_swiglu_triton`: with the forward's saved halves (`SAVED`, the training
  path) it is one GEMM `dh = dy @ Ws` with the SwiGLU gate backward in the epilogue writing `[dA|dB]`; without them it recomputes a, b in fp32 first
  (kept, driven by the driver and the checker). `dx`, `dW*` are the same stacked cuBLAS GEMMs as before. The grad flag is decided in `triton_transition` /
  `transition_residual` (`ctx.needs_input_grad` is true for parameters even under `no_grad`), so inference never stores the halves.
  The old gate-backward kernel stays for `fused.py`.
* Config sets: `default/` carries the H100 tiles, at most 24-25 rows per op: on a cache miss the engine searches the whole default set only when it has
  <= `autotune_miss_cap` rows, otherwise a heuristic subset that never contained the big tiles (most of the original gap: the H100 portable path ran
  128x128 / 64x64 pointer tiles). `grid/` widened (BLOCK_M1 256, GROUP_M 16, warps 8) and gained the new op.
* Registry / drivers / checks / `key_gaps_allowed.csv` / README + support counts / `dev derive --arch sm86` / sweep page. The committed A100 / A5000 /
  A6000 autotune caches of the changed kernels are now stale by construction (source identity) and need a rebuild on those cards.

## Numerics: bf16 halves, measured before the test band moved

The training forward keeps a and b rounded to the activation dtype, as the PyTorch statements hold them (the previous commit recomputed them in fp32
inside the backward kernel: exact against the hand-CUDA kernel but 1.1-1.2x slower). Errors against an fp64 run of the same inputs and cotangent,
module level, bf16, worst over 8 shapes (D128..D768, L 16..96) x 2 seeds (`~/miniworld-engine-scratch/pvp-a/scripts/err.py`):

| tensor | portable (new) | bf16 PyTorch statements | ratio (max over cases) | hand-CUDA kernel* |
|---|---|---|---|---|
| out | 3.16e-3 | 4.14e-3 | 0.77 | 2.8e-3 |
| dx | 4.09e-3 | 4.87e-3 | 0.84 | 3.5e-3 |
| ln_in.weight | 4.17e-3 | 5.97e-3 | 0.88 | 3.6-3.8e-3 |
| ln_in.bias | 4.22e-3 | 5.78e-3 | 0.86 | 3.4-3.6e-3 |
| expand_a.weight | 4.29e-3 | 5.07e-3 | 0.85 | 3.95e-3 |
| expand_b.weight | 4.28e-3 | 4.84e-3 | 0.89 | 3.95e-3 |
| squeeze.weight | 3.41e-3 | 4.86e-3 | 0.70 | 3.4e-3 |

(*) two shapes only (`scripts/err2.py`). The yardstick (portable <= 1.5 x the bf16 PyTorch statements' own error) holds with a margin: the worst ratio is 0.89.
The portable path is therefore as accurate as PyTorch bf16 and differs from the hand-CUDA kernel (which keeps a, b in fp32) by up to 4.2e-3 on dx.

Test bands that moved (`tests/gpu/sm90/test_transition_fused_sm90a_gpu.py::test_agrees_with_the_triton_path_to_bf16`, CUDA kernel vs the portable path; comment in the test names this record):

| tensor | before | after | basis (1.5 x bf16 PyTorch's worst error, rounded up) |
|---|---|---|---|
| dx | 5e-4 | 7.5e-3 | 1.5 x 4.87e-3 = 7.3e-3 |
| expand_a.weight, expand_b.weight | 2e-3 | 7.5e-3 | 1.5 x 5.07e-3 = 7.6e-3 (7.5e-3 held at the measured 3.6e-3 difference) |
| ln_in.weight, ln_in.bias | 2e-3 | 9e-3 | 1.5 x 5.97e-3 = 9.0e-3 |
| out, squeeze.weight (default 2e-3) | 4e-3 / 2e-3 | unchanged | measured difference 2.5e-3 / 3e-4 |

No other band moved; `test_is_no_less_accurate_than_the_triton_path` (the fused kernel against fp32 no worse than the portable path x 1.5) still passes.

## Tests

* CPU suite: 4333 passed, 2 failed, 2 errors; all four are on the base commit too (perf/ladder-audit, run unmodified): `test_key_flags_are_built_on_both_sides`
  (mpnn `DROPOUT` one-sided), `test_no_stale_caches` (A5000/A6000 caches of other ops), `test_a_ladder_offers_what_wins` (2 errors, needs >40 cached kernels).
* GPU, all 25 files of tests/gpu/{any,sm90} that mention transition, `--maxfail=1000`, final kernels: 649 passed, 11 failed, 21 skipped. 10 of the failures
  pre-exist on the base commit (run there: `test_numerical` rows `transition_{fwd_b2b,expand_gate,bwd_gate}_sm90_cuda` -- cuBLASDx headers not found in this
  environment -- and seven mpnn kernels, bands 5e-2). The 11th, `test_mpnn_transition_recompute::test_transition_update_recompute_fullgraph_rng_and_gradient_parity`,
  passes alone on this branch (6 passed) and failed inside the long run; its own comment says it is order dependent (Inductor RNG codegen after other tests compiled
  in the process); it concerns the MPNN edge dropout, not the Transition kernels.
* No rtol changed except the band above.

## Not done

The user also asked for every near-miss (ratio >= 0.95) to become a clear win (<= 0.95). Left: training D768 0.959, D512 0.976, D384 L768 0.980, D128 L128 0.959 and
`tr_fp32_D256` training 1.033 (a violation), plus the cuda/portable tie of `tr_fp32_D128` inference. Where the time goes at D768 training (ms, graph): LN 0.17, expand with
a/b stores ~3.3, squeeze 1.05, dh-gate ~2.4, three cuBLAS GEMMs 4.6 (the same ones PyTorch runs), LN backward 0.5. The Triton persistent kernels do not overlap epilogue
stores with the next tile's loads (a store-only probe of the dh kernel ran at 1.4 ms against 2.4 with the a/b reads); the remaining margin needs a warp-specialised kernel.

## Follow-up attempts (2026-10-26), none adopted

Targets: every ratio <= 0.95; open rows are listed above. Tried, measured on H100, reverted:

* Exhaustive tile / warps / stages / GROUP_M sweeps (BM 64-256, BN 64-256, BK 32/64, stages 2-5, 1/4/16) of the expand+AB, squeeze and saved-halves backward kernels at
  D768 L384: best 2.459 / 1.068 / 2.328 ms, the same configs the 24-row default sets already hold. The tile space is exhausted; the gap is the epilogue, not the schedule.
* `warp_specialize=True` (with and without `flatten`) on the expand and backward persistent loops: 0-1 % (2.346 vs 2.386 ms expand), the backward did not improve.
* fp32 D256 training: backward that recomputes the halves instead of reading them: 5.20 ms against 5.10 (saved) and 4.93 (PyTorch); at D128 it is 1.745 against 1.909 (D128
  already passes). Not adopted.
* CUDA `tr_fp32_D128` inference (generic sm90a, TF32): per-kernel profile at L768 is swiglu GEMM 0.795 ms, squeeze GEMM 0.761 ms, LayerNorm 0.197 ms. The squeeze is bound by
  the fp32 `h` (M x 512 x 4 B = 1.2 GB written and read back). Walking the rows in 4096-16384-row chunks so `h` stays in L2 made it slower (2.2-2.9 ms: smaller launches plus an
  output copy). What would win is a TF32-equivalent half-width `h` (fp16 operands for the squeeze, same 10-bit mantissa; ~0.6 ms saved) -- a new mixed-type instantiation of the
  GEMM template with its own epilogue and TMA maps, not attempted here.

## Final state (2026-10-26, fp16 halves / fp16 h)

Two range-safe fp16 stores were added (fp16 has the TF32 mantissa, 10 bits, in half the bytes of fp32; fp16's range is smaller, so both carry a device scalar
that makes overflow impossible and is 1 for ordinary magnitudes -- no host sync, graph capturable, no data dependent fallback):

* portable, fp32 activations, training (`main._launch_expand`, `residual.half_scale`): the saved expand halves are fp16 divided by
  `max(1, bound / 60000)` with `bound = (sqrt(D) max|gamma| + max|beta|) * max row-L1 norm of Wa, Wb` (LayerNorm output is `z gamma + beta`, `|z| <= sqrt(D)`).
  Only the residual (LayerNorm) path uses it; `triton_transition` (no LayerNorm, no bound) keeps fp32 halves. fp32 D256 training 5.11 -> 4.89 ms.
  Error against fp64 equals the fp32 halves' (`tests/gpu/any/test_transition_fp16_halves_gpu.py`: <= 1.5x, ordinary and 2e4-scaled expand weights; measured, fp16 vs
  fp32 halves: out 1.78e-3 / 1.78e-3, dx 1.44e-3 / 1.43e-3, weight grads 1.64-1.69e-3 / 1.64-1.67e-3 at ordinary scale, identical at 2e4; PyTorch TF32 itself 4.0-5.8e-4 --
  the portable fp32 path was 3-4x the PyTorch TF32 error before this change, Triton's tf32 dot truncates; unchanged here). A first version with a running-max flag and an in-kernel
  recompute fallback was 5% SLOWER than fp32 halves (a runtime branch around a K loop inside the persistent loop) and was dropped for the bound.
* CUDA generic sm90a, fp32 activations, forward, >= 16384 rows (`fused_generic_sm90a._fwd_h16`, `tr90.cuh`, `bind.cu`): `h = silu(a) b` is written by the expand GEMM as fp16 through
  the staged TMA epilogue (`Tt<__half>`, `swiglu_h16`) and read by the squeeze GEMM on f16 wgmma with Ws as fp16 (`squeeze_res_h16`); `h` and `Ws` are divided by device scalars from the
  same kind of bound (`h <= |a||b|`, `max|Ws|`). fp32 D128 inference L768 1.587 -> 1.236 ms (swiglu 0.758 -> 0.567, squeeze 0.795 -> 0.484, +0.04 for the scale reductions).
  Error against fp64 = the PyTorch TF32 statements' (3.31e-4 vs 3.31e-4 at ordinary weights, 5.13e-4 vs 5.13e-4 at 2e4-scaled ones;
  `tests/gpu/sm90/test_transition_generic_h16_gpu.py`). Training forwards of the fp32 path use it too (the backward recomputes, unchanged).
  The jit extension revision is v29.

Ladder `tr_` rows at the final commit (3 repeats, ms; ratio = portable / PyTorch best; **bold** = still >= 0.95): no `VIOLATION` of any kind (26 of 26 OK).

| row | PT best | portable before (ratio) | portable now (ratio) | CUDA before | CUDA now | CUDA/portable |
|---|---|---|---|---|---|---|
| tr_D128 inference L128 | 0.05674 | 0.04013 (0.70213) | 0.04474 (0.78849) | 0.02256 | 0.02272 | 0.50787 |
| tr_D128 inference L384 | 0.45206 | 0.28403 (0.62881) | 0.25136 (0.55603) | 0.1127 | 0.11322 | 0.45041 |
| tr_D128 inference L768 | 1.71059 | 1.0808 (0.63166) | 0.90054 (0.52645) | 0.4089 | 0.40869 | 0.45382 |
| tr_D128 inference L1024 | 3.03277 | 1.91342 (0.63096) | 1.58997 (0.52426) | 0.74077 | 0.7383 | 0.46435 |
| tr_D128 training L128 | 0.16054 | 0.15184 (0.9456) | 0.15376 (**0.95774**) | 0.0855 | 0.0855 | 0.55609 |
| tr_D128 training L384 | 1.21418 | 1.05102 (0.86663) | 1.0137 (0.83488) | 0.54557 | 0.54139 | 0.53408 |
| tr_D128 training L768 | 4.5144 | 3.99142 (0.88427) | 3.8672 (0.85664) | 2.14944 | 2.12459 | 0.54939 |
| tr_D128 training L1024 | 7.95731 | 7.04643 (0.88557) | 6.8731 (0.86375) | 3.79789 | 3.75144 | 0.54581 |
| tr_D256 inference L384 | 0.88022 | 0.81725 (0.92677) | 0.59275 (0.67341) | 0.40698 | 0.40432 | 0.68211 |
| tr_D256 inference L768 | 3.40576 | 3.3352 (0.97701) | 2.29346 (0.67341) | 1.59293 | 1.59274 | 0.69447 |
| tr_D256 training L384 | 2.34006 | 2.57963 (1.10389) | 2.15066 (0.91906) | 1.84368 | 1.82032 | 0.8464 |
| tr_D256 training L768 | 8.99136 | 10.16478 (1.127) | 8.29827 (0.92292) | 7.16554 | 7.03949 | 0.84831 |
| tr_D384 inference L384 | 1.50861 | 1.59302 (1.04067) | 1.11875 (0.74158) | 0.96509 | 0.9529 | 0.85175 |
| tr_D384 inference L768 | 5.9239 | 6.26278 (1.05021) | 4.50278 (0.7601) | 3.78282 | 3.78294 | 0.84013 |
| tr_D384 training L384 | 4.08166 | 4.95818 (1.21326) | 3.86374 (0.94661) | 3.6375 | 3.60317 | 0.93256 |
| tr_D384 training L768 | 15.80589 | 19.93955 (1.24285) | 15.57312 (**0.98527**) | 13.65914 | 13.53117 | 0.86888 |
| tr_D512 inference L384 | 2.288 | 2.61818 (1.11877) | 1.81846 (0.79478) | 1.58691 | 1.56702 | 0.86173 |
| tr_D512 training L384 | 6.14086 | 7.87082 (1.26959) | 6.00083 (**0.9772**) | 6.02837 | 5.97406 | **0.99554** |
| tr_D768 inference L384 | 4.32843 | 5.4025 (1.24015) | 3.7067 (0.85636) | 3.37702 | 3.37474 | 0.91044 |
| tr_D768 training L384 | 12.17885 | 16.56752 (1.35052) | 11.88312 (**0.97572**) | 11.02454 | 10.97123 | 0.92326 |
| tr_fp32_D128 inference L384 | 0.84824 | 0.48819 (0.57718) | 0.41718 (0.49182) | 0.40714 | 0.36381 | 0.87206 |
| tr_fp32_D128 inference L768 | 3.28317 | 1.86707 (0.56781) | 1.58198 (0.48185) | 1.58835 | 1.23578 | 0.78116 |
| tr_fp32_D128 training L384 | 2.36739 | 1.98643 (0.83729) | 1.79574 (0.75853) | 1.63805 | 1.5976 | 0.88966 |
| tr_fp32_D128 training L768 | 9.13896 | 7.74194 (0.84653) | 6.83723 (0.74814) | 6.4143 | 6.09272 | 0.89111 |
| tr_fp32_D256 inference L384 | 1.69198 | 1.27469 (0.75195) | 1.11094 (0.65659) | 0.99043 | 0.8671 | 0.78051 |
| tr_fp32_D256 training L384 | 4.91738 | 5.25474 (1.06477) | 4.88107 (**0.99262**) | 4.04851 | 3.87645 | 0.79418 |

Still at or above the 0.95 the user asked for (all are strict wins, none is a violation): portable training D768 0.976, D512 0.977, D384 L768 0.985, D128 L128 0.958, fp32 D256 0.993,
and the CUDA/portable pair of D512 training (CUDA 5.974 vs portable 6.001 ms, 0.9955). Tried without gain for these (all measured, reverted): exhaustive tile sweeps, `warp_specialize`,
dropping `b` from the saved activations (`b sigma(a) = h / a`: 12.02 vs 11.96 ms), fp32 recompute backward, cuBLAS squeeze for fp32, row-chunking `h` in L2 for the CUDA fp32 path.

Tests at the final commit: CPU 4333 passed, 2 failed + 2 errors (all four also fail on perf/ladder-audit); GPU, 27 transition-related files of tests/gpu/{any,sm90} with `--maxfail=1000`: 653 passed, 11 failed
(the 10 that fail on the base commit -- three `*_sm90_cuda` numerical rows without cuBLASDx headers, seven mpnn kernels -- and `test_mpnn_transition_recompute` which passes alone, order dependent).
