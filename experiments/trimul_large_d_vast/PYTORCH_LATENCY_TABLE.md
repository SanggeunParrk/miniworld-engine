# PyTorch vs our TriMul kernels: Vast H100 (2026-09-27)

The current comparison baseline is the pure PyTorch reference. Native selections
are the same qualified paths used previously: installed specialized D128, frozen
D256/D384 wide checkpoints, D512 checkpoint24 with split4 only at L384. These
wide plans remain explicit research entries, not installed production dispatch.

Conditions: batch1, BF16 activations/linear weights, FP32 LN affine parameters,
same seeded nonzero inputs/weights/upstream gradients, pair mask and supplied
row-dropout scale, residual and all 11 input/parameter gradients. The supplied
dropout scale is fixed for timing; RNG generation is outside both workloads.
Both paths use manual CUDA Graph replay. Each row is a same-GPU paired comparison
with alternating order and 75 samples; compilation and CPU launch time are excluded.
Latency is complete forward + backward in milliseconds; speedup = PyTorch/native.

The reference is the frozen fixture.ref: F.layer_norm on FP32 input then cast
back to BF16, F.linear, sigmoid/multiply, two einsums, concatenation and residual.
This matches the existing PyTorch module LN policy (modules/primitives.py).
The engine Triton LN path is not called. torch.compile uses default Inductor
with fullgraph=True, dynamic=False and triton.cudagraphs=False; Inductor may
generate Triton kernels internally. This is not a claim of globally optimal
PyTorch tuning or a fully Triton-free compiled runtime.

## PyTorch with torch.compile

| L | D | PyTorch ms | Our kernel ms | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 384 | 128 | 3.989 | 1.018 | 3.92x |
| 384 | 256 | 7.844 | 2.580 | 3.04x |
| 384 | 384 | 12.437 | 4.458 | 2.79x |
| 384 | 512 | 16.829 | 6.803 | 2.47x |
| 768 | 128 | 35.323 | 3.797 | 9.30x |
| 768 | 256 | 73.225 | 10.387 | 7.05x |
| 768 | 384 | 113.653 | 17.951 | 6.33x |
| 768 | 512 | 153.176 | 26.795 | 5.72x |

## PyTorch eager (also captured in CUDA Graph)

| L | D | PyTorch ms | Our kernel ms | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 384 | 128 | 6.996 | 1.020 | 6.86x |
| 384 | 256 | 13.390 | 2.580 | 5.19x |
| 384 | 384 | 20.564 | 4.455 | 4.62x |
| 384 | 512 | 27.864 | 6.795 | 4.10x |
| 768 | 128 | 47.082 | 3.800 | 12.39x |
| 768 | 256 | 96.317 | 10.401 | 9.26x |
| 768 | 384 | 150.468 | 17.959 | 8.38x |
| 768 | 512 | 208.769 | 26.729 | 7.81x |

Native medians differ slightly between the compiled and eager tables because
each is a separate paired run; use the speedup in its own row.

## Validation and retained diagnostic failures

All 16 full-workload comparisons passed the existing independent BF16 reference
bounds (relative L2 output <0.005, each gradient <0.01), normal graph replay,
changed-input/weight/dy/mask/dropout cross-implementation checks and changed
reference graph-versus-fresh-call checks (<5e-6; observed zero). These reference
bounds are separate from the stricter unchanged native-candidate qualification.

The initial optional backward-only test reused a retained compiled forward.
D256/L384 and D512/L384 failed its graph check with large errors; matching the
capture stream did not resolve D256. Those records remain incomplete in v1/v2
and are excluded. Their root cause is not established. The requested complete
F+B graph recomputes forward state on every replay and passes all checks above.
No BWD-only comparison is claimed for the full matrix. Current harness defaults
to full F+B; --with-backward is an optional unqualified diagnostic.

## Reproduction

```sh
scripts/vast-sync.sh run 0 bash experiments/trimul_large_d_vast/env.sh python experiments/trimul_large_d_vast/compare_pytorch.py --width 256 --length 384 --mode compiled
scripts/vast-sync.sh run 1 bash experiments/trimul_large_d_vast/env.sh python experiments/trimul_large_d_vast/compare_pytorch.py --width 512 --length 768 --mode eager
```

Set TRIMUL_SHORT_RESULTS to a fresh output directory when repeating. Each run
records source hashes, reference source text, loaded cuBLAS paths, GPU/runtime
identity, every paired sample and all validation errors. Historical harnesses
are retained locally under .bench/pytorch-harness-archive and on Vast in the
immutable v2/v3 experiment copies; original shared v1 is also in sync backups.

## Raw records

- compiled L384/D128: `.bench/vast-20260927/results/trimul-pytorch-v1/pytorch-compiled-D128-L384.json`
- compiled L384/D256: `.bench/vast-20260927/results/trimul-pytorch-v3/pytorch-compiled-D256-L384.json`
- compiled L384/D384: `.bench/vast-20260927/results/trimul-pytorch-v1/pytorch-compiled-D384-L384.json`
- compiled L384/D512: `.bench/vast-20260927/results/trimul-pytorch-v3/pytorch-compiled-D512-L384.json`
- compiled L768/D128: `.bench/vast-20260927/results/trimul-pytorch-v1/pytorch-compiled-D128-L768.json`
- compiled L768/D256: `.bench/vast-20260927/results/trimul-pytorch-v3/pytorch-compiled-D256-L768.json`
- compiled L768/D384: `.bench/vast-20260927/results/trimul-pytorch-v1/pytorch-compiled-D384-L768.json`
- compiled L768/D512: `.bench/vast-20260927/results/trimul-pytorch-v3/pytorch-compiled-D512-L768.json`
- eager L384/D128: `.bench/vast-20260927/results/trimul-pytorch-v1/pytorch-eager-D128-L384.json`
- eager L384/D256: `.bench/vast-20260927/results/trimul-pytorch-v3/pytorch-eager-D256-L384.json`
- eager L384/D384: `.bench/vast-20260927/results/trimul-pytorch-v1/pytorch-eager-D384-L384.json`
- eager L384/D512: `.bench/vast-20260927/results/trimul-pytorch-v3/pytorch-eager-D512-L384.json`
- eager L768/D128: `.bench/vast-20260927/results/trimul-pytorch-v1/pytorch-eager-D128-L768.json`
- eager L768/D256: `.bench/vast-20260927/results/trimul-pytorch-v3/pytorch-eager-D256-L768.json`
- eager L768/D384: `.bench/vast-20260927/results/trimul-pytorch-v1/pytorch-eager-D384-L768.json`
- eager L768/D512: `.bench/vast-20260927/results/trimul-pytorch-v3/pytorch-eager-D512-L768.json`
