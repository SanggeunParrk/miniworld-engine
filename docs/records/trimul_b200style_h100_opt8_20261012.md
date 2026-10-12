# TriMul H100 opt8 — D64 B200-style gate-gradient placement, 2026-10-12

D64 training now places the output-gate input-gradient GEMM in B7, as in the B200 small-width algorithm. Removing dynamic accumulator indexing makes the resulting CUDA implementation faster than installed opt6 in **24/24 tested shapes: 1.009949–1.044120x**. A fresh complete forward+backward comparison against **cuEquivariance 0.12.0 wins 24/24: 1.908789–3.075971x**.

The selected implementation is installed behind the existing `MINIWORLD_TRIMUL_H100_B200STYLE=1` flag. Its default remains zero, and the original D64 implementation remains available with the flag disabled. Installation includes separate CUDA sources and Python entry points, plus a flag/gradient/graph dispatch regression test. No commit or push occurred.

**SOL90 remains unmet.** This closes the D64 gate-gradient placement comparison left open in [opt6](trimul_b200style_h100_opt6_20261008.md). It does not close the normalized-input storage comparison: H100 still saves LayerNorm statistics and reconstructs normalized input, whereas B200 small saves the normalized input. The implementations are not identical in all storage, scheduling, folding or rounding details.

## Scope and measurement

All new timings are D64, batch 1, BF16 activations and linear weights, FP32 LayerNorm parameters, hidden width 64 per direction, one module layer, dropout 0.25 and mask probability 0.1. Lengths are **128, 256, 384, 512, 640, 768, 896, 1024**, with outgoing, incoming and bidirectional modules. FP32 master weights receive correctness coverage, not the performance claim.

The unchanged official `benchmarks/runners/bench.py` captures compiled manual CUDA Graphs. Each cell uses **90 alternating CUDA-event pairs** in one process on the same H100. The observer asserts matching input/parameter tensors, then checks the measured graphs and the candidate compiled invocation for the intended kernel and absence of Triton kernels. This is complete module **forward+backward**, including dropout and preparation, rather than a backward-kernel timing. Dropout probability and graph regime match; independent graph replays are not claimed to share an identical dropout draw.

The runner SHA256 is `9c414e84c20b74a3c94f16d6928066a0910e0ce306ea8d4d15dbc43463f9181a`. Work ran on node02 through `h100/cssb/normal_h100` with one allocated H100 per GPU job and explicit memory requests. Because shared `/home` remained absent on compute nodes, the existing PyTorch 2.13.0+cu129 environment, checkout and existing CUTLASS 4.2 headers were staged in a private `/tmp` directory. No system mount or package installation was changed. All compilation, imports, tests, profiling and archive generation ran on compute resources.

| Direction | Shapes | Opt6 / selected F+B | cuEquivariance / selected F+B |
|---|---:|---:|---:|
| Outgoing | 8 | 1.011603–1.044120x | 2.089208–3.075971x |
| Incoming | 8 | 1.030261–1.042054x | 2.089286–3.064157x |
| Bidirectional | 8 | 1.009949–1.036633x | 1.908789–2.955256x |

Jobs **28085** and **28089** provide the native and vendor grids. These are ratios of paired sample medians, not confidence intervals. Absolute times from separate jobs should not be subtracted to attribute an optimization.

After installation, job **28109** compares the public flag-off and flag-on module paths at L128/L1024 in all three directions. All six repeated cells improve, **1.013522–1.039419x**, and profiles confirm the actual graph dispatch. This checks the packaged selector as well as the private prototype.

Inference is unchanged by this D64 training selection. The proposed wide K3 producer-statistics optimization was separately rejected after full-module measurement; see [opt7 full-module follow-up](trimul_b200style_h100_opt7_full_20261012.md). The previous opt6 inference/vendor results retain their original scope, including unsupported Anthropic cells. This round does not claim a fresh comparison of every other width.

## What changed and why the first ports lost

Opt6 B1 computes `dxg = dGout @ Wg`, rounds it to BF16 and writes `[M,64]`. B7 reads that tensor and adds it to its input-projection gradient. The new B1 writes BF16 **dGout** in the same-sized buffer. B7 performs the gate GEMM and accumulates it in FP32 with the input-projection contribution before input-LayerNorm backward. This moves the computation without reducing its FLOPs or eliminating the `[M,64]` global intermediate. It also changes the intermediate rounding point, so full gradient validation is required.

The first implementation removes the unfolded Wg resident tile from B1, reducing its dynamic shared memory by 8,192 bytes. Single-direction B7 has space for resident Wg. Bidirectional B7 originally requests **227,904 bytes**, so adding another 8,192 bytes would exceed the per-block dynamic shared-memory limit. The first bidirectional port instead reloads Wg into the normalized-input shared buffer after its final consumer completes. It is correct and sanitizer-clean, but regresses F+B: single-direction ratios are **0.980738–0.986755x**, and bidirectional ratios are **0.954653–0.954844x**.

The second implementation keeps each thread's four input-LayerNorm affine partials in registers. Their 4,096-byte persistent shared array is replaced by an end-of-kernel reduction in the dead DL staging region. This permits resident Wg while requesting **232,000 bytes** of dynamic shared memory. The final reduction keeps its original order. The bidirectional F+B ratios recover to **0.995653–0.998131x**, but still do not improve opt6.

NCU then exposed a different cost shared by the original and new B7 implementations: runtime indexing into `dxn[32]` during warpgroup exchange forces a **128-byte local stack array**, even though PTXAS reports zero register spill bytes for that array. Both array alternatives can be indexed at compile time. Selecting between their values, instead of selecting an index, keeps the accumulator in registers. This removes the B7 stack allocation and measured local-memory accesses. It is the main source of the final gain.

## Separate the indexing repair from the fusion decision

A control applies the identical accumulator-index repair to opt6 while leaving the gate GEMM in B1. Its output and all gradients are bitwise equal to original opt6 in **12 cases**: L128/L384, three directions, BF16/FP32 master weights.

Job **28082** compares this repaired control with the repaired B200-style placement at L384/L768. The control/selected ratios are **0.996712–1.006982x**. The short single-direction cells slightly favor the control; the longer cells and both bidirectional cells favor the B200-style placement. This does not establish a large standalone fusion speedup. It shows that, after addressing accumulator traffic and Wg residency, the B200-style placement can be retained without the substantial regressions of the first port. The 3–5% representative-case improvement over unmodified opt6 must not be attributed entirely to moving the GEMM.

## NCU evidence and remaining utilization gap

NCU profiles B1/B7 from full training invocations after warm-up, with baseline and candidate in one process and kernel replay. These durations are diagnostics, distinct from the paired graph medians above. The final scalar-index candidate at L384 gives:

| Direction / kernel | Opt6 µs | Selected µs | Opt6 / selected SM throughput | Selected DRAM throughput |
|---|---:|---:|---:|---:|
| Outgoing B1 | 51.744 | 50.144 | 36.00 / 36.61% | 49.07% |
| Outgoing B7 | 96.800 | 87.424 | 32.18 / 36.73% | 38.18% |
| Bidirectional B1 | 65.536 | 64.704 | 36.12 / 36.12% | 55.40% |
| Bidirectional B7 | 143.360 | 126.016 | 32.73 / 37.13% | 36.04% |

The selected B7 kernels have zero stack/spill bytes and zero measured local load/store sectors. Eligible warps per scheduler rise from roughly **0.473 to 0.531** for outgoing B7 and **0.465 to 0.531** for bidirectional B7. DRAM reads stay near **96.2 MB** and **134.0 MB**, respectively. Thus the large local-memory reduction is not an equal-sized HBM-read reduction. The selected bidirectional **B1 still has an 8-byte spill**; the complete implementation is not described as spill-free.

Measured B1/B7 utilization remains below both SOL90 and the requested 70–80% stopping range. Neither the failed initial port nor these counters establish a hard limit on further fusion. Saved-XN policy and additional pipeline work remain open.

## Correctness, safety and installation

- The final prototype passes **48 strict FP32-oracle cases**, covering all 24 shapes with both BF16 and FP32 master weights. Maximum output relative error is **0.003604**, below 0.006; maximum gradient relative error is **0.006754**, below 0.01. No tolerance was relaxed.
- The existing D64 suites pass **51 tests**, with six existing skips and two stale rejection-message tests deselected. Those two tests also fail on unmodified opt6 because they expect `native H100 D64` while the current module rejects with the generic native-TriMul message. Their raw failures are preserved; the tests and comparator were not rewritten to manufacture a pass.
- The expanded 15-case suite covers changed graph input/weight/mask/dropout/dy values, zero-mask behavior, FP32 master gradients and outstanding-forward ownership. The selected scalar-index variant passes all 15.
- Memcheck, racecheck and synccheck pass at L128 for all three directions. Job **28107** repeats all three tools on the installed kernels at L256, exercising multiple B1/B7 persistent tiles and changed-input graph replay. All **18 direction/tool combinations** across the two lengths have zero errors/hazards.
- The packaged selector passes **12 new tests** in job **28097**, checking both flag states, both weight dtypes and all directions, actual CUDA Graph kernel names, all gradients, and changed-input zero-dropout replay. After copying the selected files into the checkout, job **28105** passes the same 12 checks again on the returned live bytes.
- The six installed files match `d64-selected-manifest.json`; the other **13 opt6 manifest files** retain their previous hashes. Existing uncommitted work was preserved. CUDA source bytes match the measured scalar-index prototype.

The training configuration remains the explicit native configuration used in opt6:

```bash
export MINIWORLD_TRIMUL_H100_B200STYLE=1
export MINIWORLD_TRIMUL_H100_NATIVE_BUCKETS=1
export MINIWORLD_TRIMUL_H100_PRESERVE_D128=1
export MINIWORLD_TRIMUL_H100_TRAIN_CUDA=1
export MINIWORLD_TRIMUL_H100_TRAIN_FUSED_PREP=1
export MINIWORLD_TRIMUL_H100_CUDA_AUX=1
export TRIMUL_NATIVE_DRIVER=cuda_bindings
```

The new entry points are `h100_d64_b200gate.py` and `h100_uni_d64_b200gate.py`, selected only in the D64 training branch of `integrations/trimul_h100.py`. The original entry points are retained. The default B200-style flag remains **0**.

## Frozen evidence

`runs/benchmarks-h100/benchmarks/modules/triangle_multiplication/results/h100/b200style_opt8_20261012/` contains **683 files** listed in `SHA256.json`: **74 paired cells**, **28 NCU kernel captures in seven reports**, raw samples and actual dispatch profiles, source variants and generators, compiler stack/spill logs, strict numerical errors, sanitizers, official runner artifacts, the selected source bundle and installation receipt.

The archive preserves the first regressions, the shared-memory repair, the scalar-index control, stale baseline test failures, the initial missing-CUTLASS failure, and corrected observer/generator setup failures. `results/d64-b7gate-selected-summary.json` separates the native grid, vendor grid, fusion control and installed-path repeat. `d64-install-receipt.json` records the verified source hashes and post-install test job.
