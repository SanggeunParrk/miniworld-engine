# TriMul latency table, Vast H100, 2026-09-27

Previous Triton comparison. The user selected PyTorch as the new comparator;
see [PYTORCH_LATENCY_TABLE.md](PYTORCH_LATENCY_TABLE.md) for the current table.

BF16, batch1, CUDA Graph replay medians. Speedup = Triton/native.
Each row is a paired comparison on the same GPU. D512/L384 uses the qualified
split4 selection; D128 uses installed specialized native kernels; other wide
cells use the frozen qualified checkpoints. These are explicit research wide
plans, not a claim that production dispatch changed. Default Triton output-LN
dispatch is retained. Rows were measured across separate processes/runs.

## Full forward + backward

| L | D | Triton ms | Native ms | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 384 | 128 | 1.624 | 1.026 | 1.58x |
| 384 | 256 | 3.680 | 2.600 | 1.42x |
| 384 | 384 | 6.483 | 4.504 | 1.44x |
| 384 | 512 | 9.845 | 6.858 | 1.44x |
| 768 | 128 | 6.417 | 3.894 | 1.65x |
| 768 | 256 | 18.286 | 10.441 | 1.75x |
| 768 | 384 | 28.499 | 18.656 | 1.53x |
| 768 | 512 | 44.529 | 28.023 | 1.59x |

## Backward only

| L | D | Triton ms | Native ms | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 384 | 128 | 1.086 | 0.728 | 1.49x |
| 384 | 256 | 2.377 | 1.740 | 1.37x |
| 384 | 384 | 4.148 | 2.843 | 1.46x |
| 384 | 512 | 6.016 | 4.406 | 1.37x |
| 768 | 128 | 4.292 | 2.741 | 1.57x |
| 768 | 256 | 12.854 | 7.117 | 1.81x |
| 768 | 384 | 18.625 | 12.051 | 1.55x |
| 768 | 512 | 28.692 | 17.890 | 1.60x |

## Output-LN diagnostic, L768 full workload

| D | Default Triton ms | Forced atomic Triton ms |
| ---: | ---: | ---: |
| 128 | 6.357 | 6.258 |
| 256 | 18.227 | 14.922 |
| 512 | 43.390 | 40.155 |

This is a separate controlled intervention, not the Triton baseline in the
main tables. All three interventions have strict_equivalent=false; they were
not installed or promoted. Do not divide these latencies by native timings
from other runs and call the result a paired speedup. See LENGTH_SCALING.md.

## Raw paired records

- L384/D128: `.bench/vast-20260927/results/trimul-short-large-d/length-profile-v1/length-profile-D128-L384.json`
- L384/D256: `.bench/vast-20260927/results/trimul-short-large-d/length-profile-v1/length-profile-D256-L384.json`
- L384/D384: `.bench/vast-20260927/results/trimul-short-large-d/vs-triton-baseline-D384.json`
- L384/D512: `.bench/vast-20260927/results/trimul-short-large-d/vs-triton-split4-D512.json`
- L768/D128: `.bench/vast-20260927/results/trimul-short-large-d/length-profile-v1/length-profile-D128-L768.json`
- L768/D256: `.bench/vast-20260927/results/trimul-short-large-d/length-profile-v1/length-profile-D256-L768.json`
- L768/D384: `.bench/vast-20260927/results/trimul-short-large-d/length-profile-v1/length-profile-D384-L768.json`
- L768/D512: `.bench/vast-20260927/results/trimul-short-large-d/length-profile-v1/length-profile-D512-L768.json`
