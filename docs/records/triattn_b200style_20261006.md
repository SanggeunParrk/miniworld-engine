# H100 TriangleAttention CUDA candidate following B200

Implemented on `h100/master-weights`, baseline engine commit `4ee68216232a2085f8fce7737948afb2d338d8d8`. This is an opt-in candidate, not a replacement of the installed default. All GPU work uses local Slurm partition `h100`, account `cssb`, QoS `normal_h100`; no Vast instance is used.

## Scope and entry points

The accepted A100-derived comparison matrix has TriangleAttention C64/128/256/384/512 with H=C/32, Dhead32, d_hidden=C and L128/256/384/512/640/768. These are the 30 validated full-module shape cells. BF16 activations, BF16 parameters with FP32 LayerNorm affines, and FP32 master parameters are supported. Starting/ending, mask and broadcast dropout are included. B2 is an edge test; larger batch counts use a per-batch native attention launch loop, not a newly tuned batched scheduler. The explicit admission also allows L896/1024, but these are outside the measured matrix. QK normalization, non-self-attention, and d_hidden!=C are outside this candidate's scope.

```python
from miniworld_engine.kernels.triangle_attention.cuda import h100_triattn
out = h100_triattn.forward(module, pair, mask)
# Or opt into the candidate on the existing MiniWorld module backend:
module._h100_b200style_cuda = True
out = module(pair, mask)
```

Set `CUTLASS_PATH` to CUTLASS 4.2 headers before first use. The CUDA sources are JIT-built for sm90a; this implementation was tested with PyTorch 2.13.0+cu129 and CUDA 12.9. Isolate `TORCH_EXTENSIONS_DIR` when building alongside other work.

## Algorithm and measured H100 exceptions

The implementation follows B200's front/core/tail boundaries and grouped backward layout. Hopper uses TMA/WGMMA; B200 tcgen05/TMEM instructions are not available on H100. The native attention core retains the qualified Hopper 64-query/64-key tile and grouped dK/dV plus shared-bias backward, generalized from H4 to H2/4/8/12/16.

Both fused and grouped alternatives are implemented:

- CUDA fused frontend: LN statistics, five projections, Q/K/V/G layout and masked bias. TMA double-buffers weight tiles; C>=384 normalizes input in 64-column chunks to bound shared memory. It avoids a global normalized activation tensor.
- CUDA fused tail: sigmoid/gate, output projection, broadcast dropout and residual. Output BF16 rounding and dropout axis match the module contract.
- Fused projection dgrad/LN/residual backward: TMA/WGMMA grouped product, BF16-rounded dZ retained in shared memory, normalized activation recomputed for weight gradients.

Directly transplanting these fused stages was slower on the measured H100 shapes. The candidate therefore defaults to CUDA LN plus one packed cuBLAS frontend GEMM; Q/K/V/G remain strided views, with no repacking. The tail uses a CUDA gate product, cuBLAS output projection, and CUDA dropout/residual epilogue. Backward defaults to one grouped cuBLAS dgrad plus CUDA LN/residual. These exceptions add global intermediates, but must be judged by full-module latency rather than nominal traffic alone.

The backward packs Dall with 16-byte vector copies and compile-time channel widths. A warp computes each gate/delta head, without duplicating the already scaled output gradient. LN affine partials accumulate across 32 rows for small inputs and 128 rows for large inputs, preserving small-input occupancy while reducing large-input global storage. Weight-gradient GEMMs split the long reduction into batched cuBLAS products and sum FP32 partials. One parameter-pack kernel stages either BF16 or FP32 master weights; one finishing kernel writes all eight parameter gradients in the original parameter dtypes. Weight caches track parameter versions and CUDA Graph capture scope.

To compare the alternatives explicitly:

| Setting | Default | Alternative |
|---|---|---|
| `MINIWORLD_TRIATTN_H100_FRONTEND` | `grouped` | `fused` |
| `MINIWORLD_TRIATTN_H100_TAIL` | `grouped` | `fused` |
| `MINIWORLD_TRIATTN_H100_PROJECT_LN` | `0` | `1` |
| `MINIWORLD_TRIATTN_H100_SPLIT_WGRAD` | `1` | `0` |

The default candidate uses CUDA/cuBLAS for the entire eager/manual-CUDA-Graph forward and backward; profiler checks reject Triton kernels. This statement applies to the explicit admitted path, not every existing module dispatch. Fullgraph `torch.compile` qualification and general promotion remain outstanding.

## Validation and timing

Results and final source/binary hashes are recorded alongside the paired timing JSON. Correctness compares the nonzero residual update and all input/parameter gradients against an independent chunked PyTorch implementation and its FP32 oracle, using the BF16 reference error as the numerical allowance. There is no relaxed tolerance for zero gradients. Tests cover all-mask/one-key/None mask, zero projection weights, changed inputs/dy/master weights/masks in graph replay, and stochastic dropout replay. The original narrow path and the default dispatch remain intact.

The performance baseline is the current MiniWorld engine on the same H100, not PyTorch, not standalone Triton, and not B200 hardware. Each cell alternates baseline/candidate graph timing order across seven batches of 30 replays and reports the median. Full F+B includes identical gradient zeroing. Compilation/autotuning is warmed before capture. The baseline uses its normal shipped-cache/full-grid-or-miss-cap fallback policy; stale-cache warnings are retained in logs. Timings are B1, starting attention, mixed mask, dropout zero, nonzero randomized weights, and include the residual.


Final correctness: job21626 passed all 74 tests (30 full F+B shape cells, 30 forward cells, 8 edge cases, and 6 graph/dispatch/residual cases). One PyTorch AccumulateGrad stream-mismatch warning remains in the training graph test; replay output and every gradient match the eager execution exactly. The precision contract CPU tests passed (2 tests), and Python Ruff checks passed.

Implementation is complete as an explicit research candidate. Whole-matrix performance qualification, the <=2% BF16 versus master-weight timing requirement, fullgraph compile support, and default promotion are not complete. Earlier revisions improved some short-length cases but regressed others, especially the existing highly fused C128/L384 training path. Do not use those earlier timings as performance evidence for this final revision. The user requested wrap-up before the complete final sweep finished.

The final timing sweep was stopped at the user request: 46 completed measurements are preserved in `runs/benchmarks-h100/benchmarks/modules/triangle_attention/results/h100/b200style_20261006/partial_timings.json`. This is a partial matrix; neither whole-matrix speedup nor the <=2% master-weight requirement is established. Benchmark jobs21627/21628/21629 were canceled; other sessions' jobs were preserved.

Final sanitizer job21631 completed: memcheck (zero-update plus changed-input training graph and dropout graph tests) 0 errors; racecheck (zero-update full F+B) 0 hazards/errors/warnings; synccheck (zero-update full F+B) 0 errors. No candidate GPU jobs remain running.
