# Normalization and H100 qualification — 2026-09-17

## Production changes

- Keep the current fusion boundaries and FP32 reductions. Use relaxed GPU-scope atomics for parameter-gradient sums, whose initialization and consumers are separated by kernel boundaries.
- Apply this to common LN, strided output LN, TriMul input LN plus residual, Transition folded-stat LN, AdaLN conditioning LN, pair-bias LN/projection, RMSNorm, and RMSNorm modulation backward.
- Remove stale Transition LN width-equivalence evidence: the old record collapsed K=128/256/384/512/768 into one key, while the current launcher produces five distinct keys. The stale record silently omitted production widths from the build. The runtime key probe and a build-plan regression test verify the fix.
- Expand the persistent LN backward warp grid from 1/2 to 1/2/4/8/16/32. Preserve the existing BM/BK/stage axes.
- The CuTe TMA LN prototype below is an experiment. It is **not wired into production** and does not change defaults. Existing F2/F567/B9+B10 H100 kernels remain explicit opt-ins.

## Separate normalization kernels

BF16 official registry checker inputs; old and new code use identical selected configs. Times include resetting atomic accumulators. Seven alternating CUDA-graph measurements per arm. These are common Triton improvements, not H100-only gains.

|Kernel|L384 old → new, µs|Speedup|L768 old → new, µs|Speedup|
|---|---:|---:|---:|---:|
|_transition_ln_bwd_kernel|109.79 → 105.95|1.036×|406.72 → 394.82|1.030×|
|_dgrad_condln_kernel|14.21 → 13.15|1.080×|13.92 → 13.25|1.051×|
|_layer_norm_linear_bwd|175.52 → 130.53|1.345×|608.86 → 431.14|1.412×|
|rmsnorm_bwd_kernel weighted|24.46 → 22.99|1.064×|70.40 → 66.14|1.064×|
|rmsnorm_bwd_kernel unweighted|16.29 → 16.22|1.004×|45.06 → 44.98|1.002×|
|rmsnorm_adamod_bwd_kernel unweighted|89.06 → 88.83|1.003×|331.07 → 331.04|1.000×|
|rmsnorm_adamod_bwd_kernel weighted|109.50 → 103.62|1.057×|390.96 → 373.15|1.048×|

## Latest complete bidirectional TriMul module

Official benchmark fixture, B=1, D=hidden=128, dropout=0.25, static compile plus CUDA graph, forward and backward, no optimizer. Physical tensors are BF16 with FP32 normalization parameters. Each replay generates a fresh dropout mask. RNG reset, finite gradients and gradient overwrite were separately checked. Two independent captures, twelve alternating timing rounds per capture. Front/F567/B9+B10 configs are explicitly pinned to previously measured candidates in both arms.

|L|Latest pure Triton, ms|Latest H100 F2/F567/B9+B10, ms|Speedup|
|---:|---:|---:|---:|
|384|1.584624|1.532520|1.0340×|
|768|6.136128|5.983348|1.0255×|

## Experimental TMA B4/F4

- Implemented explicit CuTe TMA loads, shared-memory stage rings, mbarriers, and B4 TMA output stores. B4 retains the persistent per-CTA parameter partials and the same two final reductions.
- No WGMMA is appropriate here: LayerNorm contains no matrix product.
- Current prototype supports BF16 m-major inputs with N=BK=256 and aligned leading stride. It does **not** implement the full Triton feature-tile grid, so it is not a drop-in production replacement.
- L768 B4 winner: BM32/BK256/8 warps/2 stages. Independent captures measured 0.3987–0.4000 ms including final reductions versus updated Triton 0.4730–0.4751 ms: **1.186–1.189×**.
- F4 remains on Triton: best first-sweep TMA result 0.2480 ms versus Triton 0.2319 ms. Additional BM16 and deeper-pipeline candidates did not improve the result. The exploratory F4 sweep used a short shape-key label and therefore a heuristic-cache Triton baseline (the reproducer now keys by actual row count), not an exhaustive new F4 tune; this result is sufficient to reject the slower prototype but does not establish the optimal F4 configuration.
- B4 prototype passes FP32-autograd comparison, row-tail testing, CUDA-graph output-poison/replay checks, memcheck and racecheck (zero hazards). Sanitizer timings are not used as performance evidence.
- An early timing run launched on a stream outside graph capture and was discarded. Published timing uses the current capture stream and explicitly proves the graph overwrites poisoned outputs.

NCU full-set, identical large tensors; these profiler timings are separate from normal benchmark times:

|B4|NCU µs|DRAM throughput|HBM utilization|Registers/thread|Occupancy|
|---|---:|---:|---:|---:|---:|
|Updated Triton|483.23|1.935 TB/s|57.71%|255|12.50%|
|CuTe TMA prototype|416.32|2.218 TB/s|66.16%|120|24.54%|

This is not the HBM roofline. TMA B4 still has shared-memory and scheduling costs; the bandwidth percentage alone is not an end-to-end speedup limit.

In the same H100 TriMul module, changing only B4 yielded 5.9753→5.9246 ms and 5.9819→5.9057 ms across two captures: approximately 0.9–1.3% additional speedup, not 18% for the whole module. Maximum paired gradient relative-L2 difference was 2.10e-4.

## Validation

- Related numerical suite: 123 passed. Six conditioned-transition forward precision-floor assertions failed; the unchanged previous checkout `b06857c0` reproduces exactly the same six failures (12 pass there). No tolerances were relaxed.
- Additional atomic-gradient and registry tests: 20 passed, 1 skipped. Width-policy/build-plan regression: 7 passed. They cover pair-bias scalar and dot branches, ragged dimensions, BF16/FP32, and Transition replica/non-replica sums.
- Official kernel checkers passed BF16 aligned L384/L768 plus BF16 and supported FP32 ragged inputs. Transition folded-stat LN declares BF16 only; FP32 was skipped explicitly.
- Actual MiniWorld MiniPairformer two-block integration was tested separately. This is not a whole-model, optimizer, data-loader or multi-GPU training benchmark. See the integration JSON files for precision and scope.


Actual MiniWorld two-block MiniPairformer, BF16 autocast with FP32 parameters/gradients, dropout=0.25, static compile plus CUDA graph:

|L|Pure Triton, ms|H100 F2/F567/B9+B10, ms|Speedup|
|---:|---:|---:|---:|
|384|5.360896|5.148384|1.0413×|
|768|20.894433|20.174112|1.0357×|

Both arms produced matching outputs and all gradients (maximum relative-L2 below 1.90e-6). This excludes the new TMA B4 prototype. Existing caches/heuristic misses are part of this integration check; it is not a fully retuned whole-model training claim.

## Cache and remaining work

- Source changes invalidate old timings. Only measured new entries may be reused; no source-hash relabeling was done.
- An overnight two-GPU rebuild covers the nine changed Triton kernels using the declared shape/dtype/config ladders (914 declared units after the width fix; 92 L384/L768 units are scheduled first). Coverage is pending until the builder reports completion; this report does not claim all caches are finished.
- Remaining H100 work: complete the B4 feature-tile/layout/dtype/config space, qualify production dispatch and fallback, improve F4, and measure the full training step.
- Existing training and previous cache jobs were preserved.

## Reproduce

From the engine repository with CUDA dependencies available:

```bash
export PYTHONPATH="$PWD:$PWD/src"
python docs/records/normalization-h100-20260917/focus_tma.py
python docs/records/normalization-h100-20260917/bench_tma.py --length 768
python docs/records/normalization-h100-20260917/module_latest.py --length 768
MINIWORLD_DRIVER_LENGTH=384 MINIWORLD_DRIVER_DTYPE=bf16 python docs/records/normalization-h100-20260917/norm_probe.py
```

Raw JSON, configuration choices, original-kernel snapshots and experiment sources are kept alongside this report. Large NCU binaries and local compiler artifacts remain in the workspace run directory.
