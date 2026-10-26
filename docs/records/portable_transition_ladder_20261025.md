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
