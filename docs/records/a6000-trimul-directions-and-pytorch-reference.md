# A6000 triangle directions and independent PyTorch reference

Recorded 2026-09-10. The native benchmark CLI produced 192 accepted rows, covering 64 groups with three process repetitions each.
Source identity: `6df41ce4c744c85f51927609d783209c66950f22d2e8af1624b48191ec808b22`. Raw CSV paths/hashes, invocation arguments, resolved shapes, run/config identities, GPU UUIDs and all repetitions are preserved in the [JSON record](a6000-trimul-directions-and-pytorch-reference.json).

## Findings

- L384 outgoing training: MiniWorld 4.8456 ms versus cuEquivariance 4.6290 ms, 4.68% slower. Incoming shows 4.79% slower. This deficit reproduced across all three process repetitions; it is not explained by a false compile flag.
- The immediately preceding L384 table recorded outgoing training at MiniWorld 4.8343 ms versus cuEquivariance 4.5962 ms, consistent with the present deficit. Earlier, unspecified benchmark regimes cannot establish a regression without matching their code, direction, shapes, precision and graph policy.
- The real MiniWorld outgoing→incoming sequence takes 2.1248 ms inference and 9.6645 ms training. Bidirectional takes 2.0797/8.0814 ms: 2.12% faster in inference, 16.38% faster in training.
- Multiplying the outgoing time by two was not a measurement of outgoing→incoming. Bidirectional also has different semantics: it shares input normalization and normalizes the concatenated 2h contractions, whereas a sequential pair includes two residual updates and feeds the first result into the second.
- The preceding table compared single-direction training measured on gpu01 GPU `ecfb7c52-fbb1-de4f-fc55-270af8e02983` with bidirectional training on gpu01 GPU `13b61e95-69d6-f2ca-fe4c-e4593d53b746`. Thus its cross-module comparison was not controlled for physical GPU. All four current triangle workloads use gpu04 GPU `e1054d59-b429-c2a3-2b82-dad9e3512734`. This establishes a flaw in the earlier comparison, without claiming the exact cause of every historical millisecond is proven.
- A further three-process control benchmarks bidirectional training at L384 alone on that same gpu04 GPU: PyTorch 26.9588, cuEquivariance 8.8018, MiniWorld 8.1101 ms. Compare with the L128→L384 sweep values below; this checks whether preceding L128 compilation explains the change. These nine control rows are additional to the 192 matrix rows.
- The previous cuEquivariance bidirectional implementation incorrectly normalized the two halves separately. It now composes vendor primitives around the same shared normalization as the PyTorch/MiniWorld bidirectional module. This enables an equivalent baseline.
- SWA Attention and the nested attention in SWA DiT used MiniWorld RMSNorm, RoPE and sigmoid-gate kernels under the PyTorch label. Those paths now use torch operations. FlashAttention remains the allowed exception. Previous SWA DiT PyTorch timings describe the old hybrid reference and are superseded here.
- No kernel implementation or tile/dispatch-cache entry was changed during these comparisons. Cache misses abort; source/config/cache identities were checked before and after runs. L128 tuning remains deferred.

## Measurement conditions

- RTX A6000, all triangle comparisons on one gpu04 GPU; SWA comparisons on one gpu01 GPU. No concurrent timing jobs on the same host from this experiment. Exact UUIDs are in JSON.
- BF16 activations and trunk weights, FP32 normalization affine parameters; TF32 disabled; dropout 0; mask probability 0.125. Pair shape `[1,L,L,128]`, hidden width 128 per direction.
- Inference: actual `torch.compile`, then manual CUDA Graph capture/replay. Training: actual `torch.compile`, CUDA Graph disabled, forward and backward with gradients reset, no optimizer step.
- A=5 inference / A=48 training. Pair operations have no augmentation axis. SWA uses `[A,L*8,128]` atoms (L384 means 3072 atoms).
- Single and bidirectional depth 1; sequential depth 2 with `trimul_direction=alternating`. CSV direction is explicit for every triangle row.
- Existing `benchmarks/runners/bench.py` and its official timer were used through a cache-miss guard. Warmup 10 ms / repetition budget 100 ms; table values are medians of three native CLI processes, with implementation order rotated. Compilation and graph setup are outside steady-state timing.
- Every accepted row records executed compiled-graph evidence, requested/actual graph policy, input shapes/dtypes, and matching run/config/source identity. The external orchestration script does not implement a replacement timer.

## Timings (ms)

The bidirectional cuEquivariance column is a vendor-primitive composition, not a native fused bidirectional API. Its shared input/output norms and input gated projection use vendor primitives; contractions and final output projection/gate use torch. The installed vendor dual-input GEMM requires equal input widths, while this module has d_pair versus 2h.
The [NVIDIA single-direction API](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_multiplicative_update.html) exposes outgoing/incoming updates. Two complete calls would change this bidirectional module’s output-normalization semantics.

## L=128 inference

| Operation | PyTorch | cuEquivariance | MiniWorld |
|---|---:|---:|---:|
| outgoing | 0.2365 | 0.1331 | 0.1270 |
| incoming | 0.2365 | 0.1382 | 0.1331 |
| outgoing→incoming | 0.4772 | 0.2724 | 0.2540 |
| triangle_multiplication_bidirectional | 0.4219 | 0.2673 | 0.2488 |
| swa_atom_attention | 0.1372 | — | 0.1454 |
| swa_dit | 0.2273 | — | 0.1997 |

## L=128 training

| Operation | PyTorch | cuEquivariance | MiniWorld |
|---|---:|---:|---:|
| outgoing | 0.8484 | 1.6118 | 1.5841 |
| incoming | 0.8509 | 1.5985 | 1.6056 |
| outgoing→incoming | 1.7213 | 2.6255 | 2.5928 |
| triangle_multiplication_bidirectional | 1.3783 | 1.4597 | 1.6640 |
| swa_atom_attention | 5.2838 | — | 5.4554 |
| swa_dit | 7.9217 | — | 8.3026 |

## L=384 inference

| Operation | PyTorch | cuEquivariance | MiniWorld |
|---|---:|---:|---:|
| outgoing | 4.9116 | 1.1448 | 1.0557 |
| incoming | 4.9183 | 1.1244 | 1.0445 |
| outgoing→incoming | 9.8181 | 2.3081 | 2.1248 |
| triangle_multiplication_bidirectional | 9.6881 | 2.7607 | 2.0797 |
| swa_atom_attention | 0.3645 | — | 0.4045 |
| swa_dit | 0.6195 | — | 0.5704 |

## L=384 training

| Operation | PyTorch | cuEquivariance | MiniWorld |
|---|---:|---:|---:|
| outgoing | 14.1368 | 4.6290 | 4.8456 |
| incoming | 14.1358 | 4.6259 | 4.8476 |
| outgoing→incoming | 28.4713 | 9.2785 | 9.6645 |
| triangle_multiplication_bidirectional | 26.9978 | 8.8090 | 8.0814 |
| swa_atom_attention | 14.3606 | — | 10.1120 |
| swa_dit | 21.7559 | — | 16.5509 |


## Diagnostic profiling

Profiles used the native training callable at L384 on a separate A6000 (gpu03), five warmed steps, compile execution verified for each case. The table below sums actual CUDA kernel/memcpy events only; profiler annotations are excluded. These are instrumented diagnostic sums, not substitutes for the unprofiled timing table and not values to subtract from that table.

| Case | Sum of CUDA event time per step (ms) |
|---|---:|
| mw_out | 4.7737 |
| cueq_out | 4.5612 |
| mw_in | 4.8054 |
| mw_sequence | 9.6728 |
| mw_bidir | 8.1641 |
| cueq_bidir | 8.7868 |

Outgoing has approximately 4.774 ms of MiniWorld GPU work versus 4.561 ms for cuEquivariance in this diagnostic. The difference therefore also exists in device work, rather than solely in Python launch gaps. MiniWorld’s input projection/front takes about 0.559 ms, its projection-gradient preparation (`_dconcat_kernel`) 0.525 ms, and its two separate output-gate elementwise kernels about 0.264/0.261 ms. The vendor uses a different gated-GEMM and normalization decomposition. These identify optimization targets, but do not prove one isolated kernel causes the entire measured difference.
Bidirectional doubles projection channel width and still performs two contractions. Shared normalization and output gating save work, but concatenation/layout conversion and wide projection/backward remain. Calling it fused does not imply a twofold speedup over two complete modules. CPU/profiler overhead and GPU sums are not interchangeable.

## Validation and limits

- All 11 native module benchmark targets × inference/training (22 paths) were executed with Fake CUDA tensors and a Triton launch recorder under the PyTorch implementation. No non-FA MiniWorld launch was observed. This is a dispatch audit, not a numerical GPU test.
- Actual CUDA forward/backward tests for standalone SWA and SWA DiT reject every MiniWorld custom op except the FlashAttention bridge; both passed.
- Equivalent cuEquivariance bidirectional output, input gradient and every parameter gradient passed actual GPU comparisons at L128 and L384 with nonzero randomized weights and masks. CPU primitive-composition tests additionally check the shared normalization widths and gradients.
- CPU regression run: 200 passes and one stale YAML expectation (previously excluding cuEquivariance). After updating that expectation, the affected 47-test subset passed. Four CUDA tests passed. The 47-test rerun overlaps the first run; these are not 251 distinct CPU passes.
- Native triangle rows compare BF16 implementations to an FP32 PyTorch reference: every output relative-Frobenius error is below 0.02 and every training input-gradient relative-Frobenius error below 0.03. Parameter-gradient coverage comes from the separate numerical tests, not from the CSV accuracy columns.
- `torch.compile` may generate Triton kernels for pure torch operations. Those are part of the compiled PyTorch baseline, not calls to repository MiniWorld kernels. This audit covers the documented benchmark workloads and tested options, not every possible module argument combination.

| Implementation | Maximum triangle output relative Frobenius | Maximum input-gradient relative Frobenius |
|---|---:|---:|
| pytorch | 0.004085 | 0.005961 |
| cuequivariance | 0.004215 | 0.006105 |
| miniworld | 0.004149 | 0.006303 |

Local diagnostic traces and test logs remain under `/home/psk6950/practice/miniworld-engine/.scratch/a6000-2026-09/mw-baseline-direction/` and the `mw-*-tests.out` files. The JSON record preserves diagnostic summaries and raw native CSV identities.
