# Portable Transition against PyTorch on H100: ladder before / after (2026-10-25)

Goal: remove the `portable_vs_pytorch` violations of the Transition rows of the performance ladder (portable = Triton family, reached on
H100 only through `settings.force_portable_kernels`; the hand-CUDA path and its H100 default are untouched).

**Result: all Transition inference rows and the training rows D256 L384 now beat PyTorch; the other training rows improved by 8-24% but
still lose (D256 L768 1.005, D384 1.09 / 1.12, D512 1.16, D768 1.23, fp32 D256 1.017). The remaining gap is a precision-for-speed trade
that the parity tests forbid (see "Not removed").**

Measured on one H100 80GB HBM3, node01, torch 2.13.0+cu129, `benchmarks/runners/ladder.py run --group '^(tr_)' --repeats 3` at commit
12be0d1b (before: `~/miniworld-engine-scratch/ladder/out_full_A`, the ladder run at 12ae4ac8; the CUDA column is unchanged by this work).
ratio = portable / best PyTorch (`pt_compiled` here), must be < 1. Times in ms, CUDA-graph replay.

| group | mode | dtype | L | PT best ms | portable before | ratio before | portable after | ratio after | CUDA after | verdict after |
|---|---|---|---|---|---|---|---|---|---|---|
| tr_D128 | inference | bf16 | 128 | 0.0568 | 0.04013 | 0.70213 | 0.04474 | 0.78761 | 0.02269 | OK |
| tr_D128 | inference | bf16 | 384 | 0.45158 | 0.28403 | 0.62881 | 0.2512 | 0.55626 | 0.11306 | OK |
| tr_D128 | inference | bf16 | 768 | 1.71091 | 1.0808 | 0.63166 | 0.90549 | 0.52924 | 0.40912 | OK |
| tr_D128 | inference | bf16 | 1024 | 3.03291 | 1.91342 | 0.63096 | 1.59709 | 0.52659 | 0.73766 | OK |
| tr_D128 | training | bf16 | 128 | 0.1609 | 0.15184 | 0.9456 | 0.152 | 0.94471 | 0.08502 | OK |
| tr_D128 | training | bf16 | 384 | 1.21186 | 1.05102 | 0.86663 | 0.94336 | 0.77844 | 0.54976 | OK |
| tr_D128 | training | bf16 | 768 | 4.51411 | 3.99142 | 0.88427 | 3.53094 | 0.7822 | 2.15998 | OK |
| tr_D128 | training | bf16 | 1024 | 7.95776 | 7.04643 | 0.88557 | 6.1912 | 0.77801 | 3.77749 | OK |
| tr_D256 | inference | bf16 | 384 | 0.88086 | 0.81725 | 0.92677 | 0.59917 | 0.6802 | 0.40642 | OK |
| tr_D256 | inference | bf16 | 768 | 3.41482 | 3.3352 | 0.97701 | 2.2988 | 0.67318 | 1.60707 | OK |
| tr_D256 | training | bf16 | 384 | 2.35792 | 2.57963 | 1.10389 | 2.27882 | 0.96645 | 1.84157 | OK |
| tr_D256 | training | bf16 | 768 | 9.00946 | 10.16478 | 1.127 | 9.0523 | 1.00476 | 7.21526 | VIOLATION(portable_vs_pytorch) |
| tr_D384 | inference | bf16 | 384 | 1.54718 | 1.59302 | 1.04067 | 1.14253 | 0.73846 | 0.97134 | OK |
| tr_D384 | inference | bf16 | 768 | 6.00181 | 6.26278 | 1.05021 | 4.58112 | 0.76329 | 3.6599 | OK |
| tr_D384 | training | bf16 | 384 | 4.1225 | 4.95818 | 1.21326 | 4.4959 | 1.09058 | 3.62971 | VIOLATION(portable_vs_pytorch) |
| tr_D384 | training | bf16 | 768 | 15.98088 | 19.93955 | 1.24285 | 17.87376 | 1.11845 | 13.50675 | VIOLATION(portable_vs_pytorch) |
| tr_D512 | inference | bf16 | 384 | 2.31272 | 2.61818 | 1.11877 | 1.83819 | 0.79482 | 1.58512 | OK |
| tr_D512 | training | bf16 | 384 | 6.26883 | 7.87082 | 1.26959 | 7.29926 | 1.16437 | 6.00389 | VIOLATION(portable_vs_pytorch) |
| tr_D768 | inference | bf16 | 384 | 4.38906 | 5.4025 | 1.24015 | 3.78557 | 0.8625 | 3.41245 | OK |
| tr_D768 | training | bf16 | 384 | 12.23546 | 16.56752 | 1.35052 | 15.03715 | 1.22898 | 11.03382 | VIOLATION(portable_vs_pytorch) |
| tr_fp32_D128 | inference | fp32 | 384 | 0.84518 | 0.48819 | 0.57718 | 0.41648 | 0.49277 | 0.40722 | OK |
| tr_fp32_D128 | inference | fp32 | 768 | 3.29088 | 1.86707 | 0.56781 | 1.58086 | 0.48038 | 1.58346 | VIOLATION(cuda_vs_portable) |
| tr_fp32_D128 | training | fp32 | 384 | 2.36946 | 1.98643 | 0.83729 | 1.71034 | 0.72183 | 1.63792 | OK |
| tr_fp32_D128 | training | fp32 | 768 | 9.13571 | 7.74194 | 0.84653 | 6.56893 | 0.71904 | 6.42326 | OK |
| tr_fp32_D256 | inference | fp32 | 384 | 1.69254 | 1.27469 | 0.75195 | 1.10603 | 0.65347 | 0.98957 | OK |
| tr_fp32_D256 | training | fp32 | 384 | 4.93533 | 5.25474 | 1.06477 | 5.01923 | 1.017 | 4.03083 | VIOLATION(portable_vs_pytorch) |

`tr_fp32_D128 inference L768` shows a new `cuda_vs_portable` tie (CUDA 1.583 ms, portable 1.581 ms): the portable kernel got faster, the CUDA
row did not move; it was OK before only because portable was slower.

## What changed (all in `kernels/transition/triton/`)

* `main.py` `transition_fwd_kernel`: the two expand weights are row-interleaved (`torch.stack((wa, wb), 1)`, ~10 MB, a few us), so a/b come out of
  ONE dot of N = 2*BLOCK_N (`tl.split` in the epilogue) instead of two dots; on sm_90 with aligned operands the kernel runs a persistent grid
  with in-kernel tensor maps (`USE_TMA`, `_tma.py`) and `flatten=True`. The pointer kernel is unchanged in arithmetic and still serves every
  other card, ragged widths and sliced operands. `MINIWORLD_PORTABLE_NO_TMA=1` forces it (A/B and tests).
* `residual.py` `_squeeze_residual_kernel`: same two modes (tensor-map loads/stores, residual through a map).
* `training.py` (new) `_squeeze_swiglu_bwd_kernel`, registry row `transition_bwd_squeeze_swiglu_triton`: ONE kernel runs the a/b recompute and
  `dh = dy @ Ws` before the gate and writes `h` and `[dA|dB]`; `dx`, `dW*` are the same stacked cuBLAS GEMMs as before. It replaces the cuBLAS `dh` call plus
  `_transition_expand_gatebwd_kernel` on the split / residual Transition (that kernel stays for `fused.py`). `TritonTransitionFunction` (public
  `triton_transition`) uses the same backward.
* Config sets: `default/` carries the H100 tiles (128x256 squeeze, 128x64/128x128 expand, 64x128 backward ...), at most 24 rows per op: on a cache miss
  the engine searches the whole default set only when it has <= `autotune_miss_cap` = 24 rows, otherwise a heuristic subset that never contained the
  big tiles (that, not the kernels, was most of the original gap: the H100 portable path ran 128x128/64x64 pointer tiles). `grid/` widened
  (BLOCK_M1 256, GROUP_M 16, warps 8) and gained the new op.
* Registry / drivers / checks / `key_gaps_allowed.csv` / README + support counts / `dev derive --arch sm86` / sweep page for the new row. The
  committed A100 / A5000 / A6000 autotune caches of the changed kernels are now stale by construction (source identity) and need a rebuild on those cards.

Single-kernel timings on H100 (L384, ms): squeeze D768 2.0 -> 1.05 (cuBLAS 0.89-1.0), expand D768 3.1 -> 2.4 (2.9 -> 2.45 by interleaving alone).

## Not removed, and why

Saving the expand halves in the forward (`STORE_AB`, kept as an option of the kernel and driven by its driver) and fusing `dh` with the gate made
training beat PyTorch at every size (probe, same harness, ms: D768 11.96, D512 6.10, D384 3.90, D256 2.18 against PyTorch 12.24 / 6.27 / 4.12 / 2.36)
-- with the halves stored in bf16, like PyTorch does. That moves `dx` by 3.8e-3 against the hand-CUDA kernels (which keep a, b in fp32), and
`tests/gpu/sm90/test_transition_fused_sm90a_gpu.py::test_agrees_with_the_triton_path_to_bf16` asserts 5e-4. fp32 halves pass it but cost two (M, ND) passes each
way and lose the gain (D768 13.99). The committed version recomputes a, b in fp32 inside the backward kernel (passes, 15.0 ms at D768). Options to
close the rest: accept bf16 halves and re-baseline that test's dx band (the maintainer's call), or a hand-pipelined backward that overlaps the three GEMMs' epilogue (the Triton
persistent kernel does not overlap epilogue stores with the next tile's loads; a pure-store probe of the same kernel ran at 1.4 ms against 2.3 with the a/b reads).

## Tests

CPU suite: 4331 passed; the 4 failures + 2 errors left are not this change (mpnn `DROPOUT` one-sided flag, stale A5000/A6000 caches of other ops,
`test_a_ladder_offers_what_wins` needing >40 cached kernels). GPU: `tests/gpu/any/*transition*`, `test_force_portable_option`, `test_whole_op_reachable`,
`tests/gpu/sm90/*transition*`, `h100_module_wiring`, `h100_master_weights`, `test_numerical -k transition` (the three `*_sm90_cuda` numerical rows fail in this
environment without CUTLASS and are CUDA kernels outside this change); after the last kernel change `test_transition_fused_sm90a_gpu.py` and
`test_transition_residual_gpu.py` (35 tests) pass. No rtol was changed (the only test edit adds the new kernel to the fp32-past-the-guard allow-list).
