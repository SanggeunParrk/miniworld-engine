# A6000 single-direction trimul training: historical reversal and present cost

Recorded 2026-09-10. This follows the [direction/reference comparison](a6000-trimul-directions-and-pytorch-reference.md). Investigation only: no production kernel, dispatch policy, or tuning cache changed.

## Why the old table showed a large win

The archived A6000 L384/D128/depth1 training CSV named `triangle_multiplication_n_layers=1_training_time_bf16-mixed_compile_seq_len.csv` records MiniWorld at 4.8143/4.8241 ms with **BF16 inputs**, and cuEquivariance at 9.1341 ms with **FP32 inputs**. Both rows label the precision BF16-mixed and use graph disabled. The graph-enabled `...compile_cudagraph-manual_seq_len_L_sweep.csv` similarly records 4.5640 ms BF16 MiniWorld versus 9.2457 ms FP32 cuEquivariance.
Those were mismatched precision comparisons. MiniWorld stayed around 4.8 ms; after the reference dtype was corrected, cuEquivariance became roughly 4.6 ms. This supports an invalid old speedup rather than evidence that MiniWorld suddenly doubled its latency. It does not establish that every historical result on every GPU was invalid. The old files lack current executed-compile/source evidence, so their compile labels alone are not proof.
The current controlled outgoing comparison is 4.8456 ms MiniWorld versus 4.6290 ms cuEquivariance with BF16 on both sides, actual compile, graph disabled, mask probability 0.125. Historical CSV hashes/rows and all diagnostic artifacts are preserved in the [JSON record](a6000-trimul-training-cause.json).

## Present bottleneck is in the training forward pass

Two implementations were profiled in one allocation, job 1672662, gpu04 A6000 UUID `ff295d93-3acc-0b21-ca80-6b90531154e9`. Each used the native L384 training callable, five warmed steps and executed compile evidence. This is the same physical GPU as the mask A/B below; it is a different allocation/device UUID from the previous full direction matrix, so instrumented sums are not subtracted from that matrix’s timings.
CUDA runtime/driver correlations assign each GPU event to its launching compiled forward/backward CPU range. GPU user annotations are excluded. Times below sum actual CUDA kernel/memory events; they are diagnostic device times, not end-to-end wall-clock benchmarks.

| Phase (ms/step) | MiniWorld | cuEquivariance |
|---|---:|---:|
| Training forward | 1.6667 | 1.1722 |
| Backward | 3.0756 | 3.4553 |
| Outside compiled ranges (gradient copies) | 0.0121 | 0.0000 |

MiniWorld spends more in training forward while its backward is faster in this profile. Training uses a different path from inference: `_UniBackHalfTriton` stores projection intermediates, materializes output normalization/projection, and runs a separate output-gate GEMM and elementwise gate. `_uni_infer` skips preactivation saves and uses the fused output back half.
The input projection saves a BF16 `[4*128,384*384]` preactivation tensor: 144 MiB, in addition to left/right outputs. The vendor wrapper saves inputs and weights and recomputes gated-projection intermediates during backward. This explains a forward/backward tradeoff; the individual saved tensor’s net cost was not separately ablated.

## Mask handling causally changes the ordering

With `mask_prob=0`, an all-valid mask and `mask=None` describe the same mathematical operation. The diagnostic wrapper passes None only in the latter case; its hash and runtime override are explicitly recorded. Both alternatives use the existing native CLI and timer, BF16, actual compile, graph OFF, L384, outgoing/depth1, no dropout. Three fresh CLI processes per alternative rotate implementation order and alternate A/B order. These diagnostic rows are not substitutes for the production mask=.125 comparison.

| Equivalent all-valid workload, native median ms | MiniWorld | cuEquivariance |
|---|---:|---:|
| Explicit all-valid mask | 4.7985 | 4.6280 |
| mask=None | 4.3878 | 4.5537 |
| Time removed | 0.4106 | 0.0742 |

MiniWorld is slower with the explicit all-valid mask and faster without it in every paired repetition. Its additional masking cost is large enough to reverse the ranking. This includes the change in generated execution when the mask is absent, not a promise that a future fused-mask patch will recover precisely the same number of milliseconds.
Production code applies the mask separately to `left` and `right` after projection (`unidirectional.py:74`), then to `d_left` and `d_right` before projection backward (`unidirectional.py:132`). These reread and rewrite large pair tensors. The actual trace contains two forward mask launches plus a generated backward mask kernel. cuEquivariance passes the mask into its gated-projection forward and backward kernels.

## Concrete next change

Fold the left/right mask into the input projection stores and the gradient mask into `_dconcat_kernel` loads. Preserve unmasked `x_n` for the output gate: folding the mask into input normalization would change the function. Then recheck output, input and all parameter gradients for outgoing/incoming and masked inputs before using the same native timing regime.
After that, separately evaluate the training output projection/gate materialization and the save-versus-recompute choice. No tile sweep or broad cache rebuild is justified by this evidence alone. Small-L tuning remains deferred.

## Validation

- All 12 mask A/B rows passed native output/input-gradient checks, actual compile and graph-OFF checks; CSV hashes and run/config/source identities were reread after completion.
- Both final profiles record one executed compiled graph, identical GPU UUID, unchanged source and cache. The original source/cache manifest still matches after the investigation.
- The all-valid experiment establishes masking overhead, not permission to remove nontrivial masks from real workloads. The proposed kernel fusion has not been implemented or benchmarked.
