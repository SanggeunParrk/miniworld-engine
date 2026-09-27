# Triton bidirectional inference: remove contraction cat

`_bidir_infer` now reuses `packed_forward(lf, rf, h)`. It allocates one final `tri` and passes `tri[:h]` / `tri[h:]` as the two cuBLAS bmm output slices. Separate outgoing/incoming buffers and the following activation cat are removed. Small weight-packing cats are unaffected. No device kernel or tuning grid changed.

## Verification

- Installed MiniWorld cu128 package: 7 tests passed.
- Module output is bitwise equal to the previous split contraction for holes/no mask/all-invalid mask, including nonzero output and LN bias parameters.
- Fixed-shape fullgraph compile and CUDA graph replay pass, including changed input and mask.
- ATen profiler at L128/384/768 records exactly 2 bmm and 0 cat in the contraction block.
- Source patch persisted in both MiniWorld and team-gm patch stacks.

## Contraction-only measurements

H100, BF16, B1, h128. CUDA graph; 10 alternating rounds, each 100 replays. This is **not full-module or full-model inference timing**. Peak memory is additional live allocation inside the contraction, after warming both paths and cuBLAS.

| L | Before ms | After ms | Speedup | Before peak MiB | After peak MiB |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.016426 | 0.010005 | 1.642x | 16 | 8 |
| 384 | 0.139725 | 0.087294 | 1.601x | 144 | 72 |
| 768 | 0.615814 | 0.417281 | 1.476x | 576 | 288 |

Eliminated cat reads+writes: `2 * (2*h*L*L*sizeof(bfloat16))`, i.e. 144 MiB at L384 and 576 MiB at L768. This is logical tensor traffic, not hardware-counter measured HBM traffic.

Evidence: `check.log`, `contraction-benchmark.json`, `measure.py`.
