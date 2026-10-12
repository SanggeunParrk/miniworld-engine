# B200-style H100 TriMul optimization — 2026-10-07

Follow-up to [the original port](trimul_b200style_h100_20261007.md). The earlier port's poor timings were evidence about that implementation, not an inherent limit of the B200 fusion algorithm. This record preserves the B200 fusion and recomputation identities while adapting tensor layout and producer/consumer work to Hopper.

## Changes and actual selection

* D512 front and backward front recomputation now keep the normalized input resident in shared memory, stream packed weights, and use SS WGMMA. Four lanes normalize each token, avoiding scalar shared-memory row-stride conflicts. The selected tile is 128 tokens, N128, four weight stages, two consumer warpgroups and `wait_group 1`.
* Backward reuses saved input statistics, recomputes preactivations and directly produces packed gate/projection gradients. The resident normalized input is also emitted by TMA for the weight-gradient GEMM, removing the separate normalization kernel and its extra input read. No front preactivation or normalized input is saved by forward.
* D256/384 retain the RS front. Their recomputation epilogue now uses transposed ldmatrix loads and stmatrix writes instead of scattered scalar shared-memory gradient transfers. TMA staging and conservative barriers protect the shared-buffer transition.
* Folded K3 uses N256/three stages at D256/512 and N192/four stages at D384, retaining fused output statistics, both GEMMs, gate, dropout, residual and training saves.
* Warp-private gradient buffers, N256 SS tiles, deeper WGMMA queues and cuBLASLt alternatives remain diagnostic configurations. They are not selected. The cuBLASLt wrapper defaults to the existing PyTorch GEMMs and allocates no workspace in the selected path.

This is Hopper TMA/WGMMA implementation of the B200 algorithm; Blackwell TMEM/tcgen05 instructions are not copied. Plane packing uses 32-channel blocks. The existing explicit flag `MINIWORLD_TRIMUL_H100_B200STYLE=1` selects the optimized candidate through the real module integration. The flag still defaults to zero. The serving contract is unchanged: H100, B1, contiguous BF16 pairs, D256/384/512, supported L128..768 buckets, outgoing/incoming/bidirectional, BF16 matrices or FP32 masters, FP32 normalization affines and optional row dropout. Unsupported module shapes retain existing dispatch.

## Bounded tuning and verification

The D512/L768 bidirectional training search declared 35 configurations and measured 34 complete F+B configurations: five RS schedules, seven SS shapes, five valid pipeline/warp-private combinations and seventeen cuBLASLt combinations. One single-stage/zero-wait configuration failed compilation and was rejected. Front/pipeline configurations passed their independent FP32 output/all-gradient pilot before full-module timing. The cuBLASLt pilot used heuristic 0 at L128; its other wide-layout heuristics were timed but are not individually numerically qualified or selected. The input-gradient sweep held weight-gradient heuristic 7 fixed; this is a measured combination search, not an exhaustive claim about cuBLASLt or all shapes. D256/384 do not have exhaustive shape/config searches.

Final frozen source passes 49 tests, including every input/parameter gradient, nonzero weights/upstream gradients, nonidentity affines, both matrix dtypes, all three directions, L128/384/768, actual module selection, zero masks, inference/training equality and changed-input/weights/mask/dropout/upstream-gradient CUDA Graph replay. Relative Frobenius thresholds are output <0.006 and each gradient <0.01.

Final memcheck and synccheck each pass 25 numerical/graph/zero-mask tests with zero errors. Racecheck passes five full native F+B/changed-graph safety scenarios with zero errors and warnings, without kernel filters or suppression. Its numerical oracle is checked separately by the full suite. Sanitizer sizes are L128; wide numerical and timing coverage is separate. Register inspection of all four selected SS variants (outgoing/bidirectional, forward/recomputation) found no stack/local allocation or local-memory spill instructions; the highest SASS register indices are 88 forward and 148 backward, with 229888 bytes dynamic shared memory. This is not a SOL measurement. FP32-master performance, B>1 and SOL90 are unverified.

## Paired performance

Both comparisons use the unchanged `benchmarks/runners/bench.py`, BF16 activations/matrices, FP32 norm affines, B1, one full residual-inclusive module, mask probability 0.1, inference dropout=0 and training dropout=0.25. Training measures complete forward and every input/parameter backward, excludes optimizer. Compilation is requested and manual CUDA Graph steps are measured with 90 alternating AB/BA CUDA-event pairs after ten paired warmups. Each pair uses equal seeded tensors and the same allocated H100. Separate eager profiles establish actual dispatch; they are not final performance measurements.

The previous-port comparison loads frozen snapshot-v4 algorithm code with distinct custom-op/cache names and its original CUDA sources. The current-default comparison uses the unchanged default selector. These are separate same-GPU paired comparisons; absolute times from the two jobs should not be treated as one three-way paired experiment. External kernels were not remeasured here, so earlier vendor timings are not combined into new ratios.

Speedup is reference time / optimized time. Above 1 favors the optimized candidate.

| Mode | Direction | vs previous port, geomean | vs current default, geomean |
|---|---|---:|---:|
| inference | Outgoing | 1.116x | 0.890x |
| inference | Bidirectional | 1.173x | 0.856x |
| training | Outgoing | 1.086x | 0.861x |
| training | Bidirectional | 1.112x | 0.841x |

### Current default

| Mode | Direction | D | L | Reference ms | Optimized ms | Speedup |
|---|---|---:|---:|---:|---:|---:|
| inference | Outgoing | 256 | 384 | 0.351488 | 0.453184 | 0.776x |
| inference | Outgoing | 256 | 768 | 1.373664 | 1.755808 | 0.782x |
| inference | Outgoing | 384 | 384 | 0.753664 | 0.782016 | 0.964x |
| inference | Outgoing | 384 | 768 | 3.093888 | 3.081440 | 1.004x |
| inference | Outgoing | 512 | 384 | 1.256480 | 1.359712 | 0.924x |
| inference | Outgoing | 512 | 768 | 4.931424 | 5.380224 | 0.917x |
| inference | Bidi | 256 | 384 | 0.637856 | 0.764608 | 0.834x |
| inference | Bidi | 256 | 768 | 2.457696 | 2.922944 | 0.841x |
| inference | Bidi | 384 | 384 | 1.127616 | 1.303648 | 0.865x |
| inference | Bidi | 384 | 768 | 4.663264 | 5.315744 | 0.877x |
| inference | Bidi | 512 | 384 | 1.891648 | 2.226624 | 0.850x |
| inference | Bidi | 512 | 768 | 7.786848 | 8.964352 | 0.869x |
| training | Outgoing | 256 | 384 | 1.735744 | 2.088896 | 0.831x |
| training | Outgoing | 256 | 768 | 6.732992 | 7.684832 | 0.876x |
| training | Outgoing | 384 | 384 | 2.799584 | 3.336576 | 0.839x |
| training | Outgoing | 384 | 768 | 11.275904 | 13.007488 | 0.867x |
| training | Outgoing | 512 | 384 | 4.309408 | 5.009312 | 0.860x |
| training | Outgoing | 512 | 768 | 17.716703 | 19.851328 | 0.892x |
| training | Bidi | 256 | 384 | 2.615648 | 3.166720 | 0.826x |
| training | Bidi | 256 | 768 | 10.666048 | 12.409696 | 0.859x |
| training | Bidi | 384 | 384 | 4.527264 | 5.409824 | 0.837x |
| training | Bidi | 384 | 768 | 18.949823 | 21.684481 | 0.874x |
| training | Bidi | 512 | 384 | 6.956448 | 8.326624 | 0.835x |
| training | Bidi | 512 | 768 | 28.063168 | 34.435776 | 0.815x |

### Previous B200 port

| Mode | Direction | D | L | Reference ms | Optimized ms | Speedup |
|---|---|---:|---:|---:|---:|---:|
| inference | Outgoing | 256 | 384 | 0.446784 | 0.450752 | 0.991x |
| inference | Outgoing | 256 | 768 | 1.750336 | 1.737632 | 1.007x |
| inference | Outgoing | 384 | 384 | 0.821376 | 0.802880 | 1.023x |
| inference | Outgoing | 384 | 768 | 3.155616 | 3.090624 | 1.021x |
| inference | Outgoing | 512 | 384 | 1.815072 | 1.342144 | 1.352x |
| inference | Outgoing | 512 | 768 | 7.113664 | 5.184128 | 1.372x |
| inference | Bidi | 256 | 384 | 0.726016 | 0.712096 | 1.020x |
| inference | Bidi | 256 | 768 | 2.936608 | 2.851584 | 1.030x |
| inference | Bidi | 384 | 384 | 1.330976 | 1.305344 | 1.020x |
| inference | Bidi | 384 | 768 | 5.395968 | 5.260672 | 1.026x |
| inference | Bidi | 512 | 384 | 3.278368 | 2.136448 | 1.534x |
| inference | Bidi | 512 | 768 | 13.035680 | 8.453568 | 1.542x |
| training | Outgoing | 256 | 384 | 2.145632 | 2.094880 | 1.024x |
| training | Outgoing | 256 | 768 | 7.844320 | 7.660256 | 1.024x |
| training | Outgoing | 384 | 384 | 3.389088 | 3.333632 | 1.017x |
| training | Outgoing | 384 | 768 | 13.129888 | 12.913696 | 1.017x |
| training | Outgoing | 512 | 384 | 6.049024 | 4.975968 | 1.216x |
| training | Outgoing | 512 | 768 | 24.127169 | 19.366495 | 1.246x |
| training | Bidi | 256 | 384 | 3.236064 | 3.129952 | 1.034x |
| training | Bidi | 256 | 768 | 12.670304 | 12.171936 | 1.041x |
| training | Bidi | 384 | 384 | 5.503296 | 5.376736 | 1.024x |
| training | Bidi | 384 | 768 | 21.992191 | 21.410879 | 1.027x |
| training | Bidi | 512 | 384 | 10.527360 | 8.167712 | 1.289x |
| training | Bidi | 512 | 768 | 42.330078 | 32.589825 | 1.299x |

## Evidence and limits

The candidate does not qualify as a broadly faster replacement for the project default; default dispatch remains unchanged. The remaining performance gap is about this implementation. No inherent limit of B200 fusion on H100 or exhaustive optimality is claimed. Small ratios near 1 require repeat evidence before promotion.

Paired jobs 27495/27498; full numerical job 27498; sanitizer job 27493. [Default paired samples](../../runs/benchmarks-h100/benchmarks/modules/triangle_multiplication/results/h100/b200style_opt_20261007/paired_default.json), [previous-port paired samples](../../runs/benchmarks-h100/benchmarks/modules/triangle_multiplication/results/h100/b200style_opt_20261007/paired_previous_port.json), [search results](../../runs/benchmarks-h100/benchmarks/modules/triangle_multiplication/results/h100/b200style_opt_20261007/search.json), [source manifest and validation/runner evidence](../../runs/benchmarks-h100/benchmarks/modules/triangle_multiplication/results/h100/b200style_opt_20261007/).

The previous dated records are unchanged. Other sessions' edits are preserved. No release installation, canonical completion marker, commit or push is changed. Work used local Slurm normal_h100 allocations with at most two of this task's GPUs concurrently.
