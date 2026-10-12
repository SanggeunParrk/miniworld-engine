# H100 TriMul comparison — 2026-10-07

Frozen checkout HEAD: `5a2302dfa5e3f3d13bf9976ff59e8f545d9be5ed`. This compares the default paths of that checkout, without enabling experimental selectors.

Hardware: NVIDIA H100 80GB HBM3; driver 575.51.03; torch=2.13.0+cu129; triton=3.7.1; cuequivariance=0.12.0; cuequivariance-torch=0.12.0; cuequivariance-ops-torch-cu12=0.12.0. Each shape compares implementations on the same GPU; outgoing and bidirectional training ran on separate H100s. GPU UUIDs, Slurm job IDs and source hashes are in the manifest.

B=1; L384/L768; D64/128/256/384/512; hidden=D per direction; one residual-inclusive layer. All measured implementations use BF16 activations and matrix parameters with FP32 normalization affines. The harness precision label is `bf16-mixed`, but this TriMul harness explicitly pins matrix parameters to BF16: these are not FP32-master timing claims. TF32 is allowed. Masks are supplied and all true (`mask_prob=0`). Inference dropout is zero. Training includes forward, every input/parameter backward, and fresh-gradient handling with stochastic broadcast dropout 0.25; optimizer is excluded.

Every result uses manual CUDA Graph replay. Engine, cuEquivariance, PyTorch, Triton and DTv1 request compilation; the CSV records observed compile evidence and scope, including partial graphs. Anthropic native inference is graph-only and uses unchanged upstream sources rebuilt with CUDA12.9. It is labeled `anthropic_hybrid` by the external-payload adapter, but has no project optimization overlay. The upstream payload has no backward; no Anthropic training ratio is claimed.

Timing uses the unchanged shared runner: median CUDA-event latency, 10 ms warmup and 100 ms sampling after compilation/autotuning/capture warmup. Comparisons are sequential on the same allocated H100, not alternating 90-pair timing or an exhaustive fresh autotune. Shipped caches use the current default miss-cap-24/full-grid fallback, with stale-cache warnings retained. Graph replay profiles are collected after timing; later cells also have separately labeled eager dispatch profiles before timing. No production source or dispatch is modified.

**Speedup = fastest successful measured comparison implementation / current default.** Values above 1 favor our kernel; values below 1 favor the comparison. The comparison set includes cuEquivariance, Anthropic, compiled PyTorch, DTv1 and the engine Triton path. Bidirectional vendor baselines are compositions with one shared output normalization over both directions; they are not two sequential unidirectional layers.

## Single direction: outgoing — inference

| D | L | Ours ms | Fastest comparison | Comparison ms | Speedup | Fastest external / ours |
|---:|---:|---:|---|---:|---:|---:|
| 64 | 384 | 0.079520 | Anthropic CUDA12.9 rebuild | 0.092448 | 1.163x | 1.163x |
| 64 | 768 | 0.301184 | Anthropic CUDA12.9 rebuild | 0.328544 | 1.091x | 1.091x |
| 128 | 384 | 0.146512 | Anthropic CUDA12.9 rebuild | 0.168480 | 1.150x | 1.150x |
| 128 | 768 | 0.562784 | Anthropic CUDA12.9 rebuild | 0.618304 | 1.099x | 1.099x |
| 256 | 384 | 0.354784 | Anthropic CUDA12.9 rebuild | 0.367456 | 1.036x | 1.036x |
| 256 | 768 | 1.417088 | Anthropic CUDA12.9 rebuild | 1.509552 | 1.065x | 1.065x |
| 384 | 384 | 0.750704 | Anthropic CUDA12.9 rebuild | 0.784640 | 1.045x | 1.045x |
| 384 | 768 | 3.153920 | Anthropic CUDA12.9 rebuild | 3.224624 | 1.022x | 1.022x |
| 512 | 384 | 1.278128 | DTv1 | 1.987728 | 1.555x | 1.555x |
| 512 | 768 | 5.060512 | DTv1 | 8.066272 | 1.594x | 1.594x |

Geometric mean: 1.167x; range 1.022–1.594x; wins 10/10.

## Single direction: outgoing — training

| D | L | Ours ms | Fastest comparison | Comparison ms | Speedup | Fastest external / ours |
|---:|---:|---:|---|---:|---:|---:|
| 64 | 384 | 0.269936 | engine Triton | 0.535168 | 1.983x | 2.433x |
| 64 | 768 | 0.956048 | engine Triton | 1.895856 | 1.983x | 2.463x |
| 128 | 384 | 0.772080 | engine Triton | 1.017120 | 1.317x | 1.759x |
| 128 | 768 | 2.791104 | engine Triton | 3.780640 | 1.355x | 1.823x |
| 256 | 384 | 1.729088 | engine Triton | 2.195840 | 1.270x | 1.559x |
| 256 | 768 | 6.733920 | engine Triton | 8.945632 | 1.328x | 1.547x |
| 384 | 384 | 2.793808 | engine Triton | 3.641968 | 1.304x | 1.513x |
| 384 | 768 | 11.357424 | engine Triton | 15.348896 | 1.351x | 1.475x |
| 512 | 384 | 4.334768 | engine Triton | 5.497344 | 1.268x | 1.373x |
| 512 | 768 | 17.935265 | engine Triton | 23.433856 | 1.307x | 1.333x |

Geometric mean: 1.425x; range 1.268–1.983x; wins 10/10.

## Bidirectional — inference

| D | L | Ours ms | Fastest comparison | Comparison ms | Speedup | Fastest external / ours |
|---:|---:|---:|---|---:|---:|---:|
| 64 | 384 | 0.118688 | Anthropic CUDA12.9 rebuild | 0.138032 | 1.163x | 1.163x |
| 64 | 768 | 0.464480 | Anthropic CUDA12.9 rebuild | 0.507808 | 1.093x | 1.093x |
| 128 | 384 | 0.249120 | Anthropic CUDA12.9 rebuild | 0.284992 | 1.144x | 1.144x |
| 128 | 768 | 0.971552 | Anthropic CUDA12.9 rebuild | 1.120496 | 1.153x | 1.153x |
| 256 | 384 | 0.601024 | engine Triton | 1.084256 | 1.804x | 2.144x |
| 256 | 768 | 2.323936 | engine Triton | 4.347104 | 1.871x | 2.301x |
| 384 | 384 | 1.152192 | engine Triton | 2.111360 | 1.832x | 1.902x |
| 384 | 768 | 4.931168 | engine Triton | 8.502176 | 1.724x | 1.835x |
| 512 | 384 | 1.853344 | DTv1 | 3.195616 | 1.724x | 1.724x |
| 512 | 768 | 8.100928 | DTv1 | 13.419040 | 1.656x | 1.656x |

Geometric mean: 1.482x; range 1.093–1.871x; wins 10/10.

## Bidirectional — training

| D | L | Ours ms | Fastest comparison | Comparison ms | Speedup | Fastest external / ours |
|---:|---:|---:|---|---:|---:|---:|
| 64 | 384 | 0.408128 | engine Triton | 0.817376 | 2.003x | 2.600x |
| 64 | 768 | 1.511936 | engine Triton | 3.036064 | 2.008x | 2.638x |
| 128 | 384 | 1.001376 | engine Triton | 1.624336 | 1.622x | 2.115x |
| 128 | 768 | 4.071648 | engine Triton | 6.568544 | 1.613x | 1.998x |
| 256 | 384 | 2.651872 | engine Triton | 3.709200 | 1.399x | 1.580x |
| 256 | 768 | 10.976672 | engine Triton | 16.343936 | 1.489x | 1.523x |
| 384 | 384 | 4.614752 | engine Triton | 6.518016 | 1.412x | 1.471x |
| 384 | 768 | 19.535200 | DTv1 | 27.215839 | 1.393x | 1.393x |
| 512 | 384 | 7.073280 | DTv1 | 9.832560 | 1.390x | 1.390x |
| 512 | 768 | 29.318336 | DTv1 | 39.831327 | 1.359x | 1.359x |

Geometric mean: 1.553x; range 1.359–2.008x; wins 10/10.

## Speedup against each comparison implementation

Geometric means of comparison latency / engine latency. Each column covers ten cells, except Anthropic inference: eight outgoing cells (D64–384) and four bidirectional cells (D64–128). Missing coverage is not counted as a win.

| Comparison | Outgoing inference | Bidirectional inference | Outgoing F+B | Bidirectional F+B |
|---|---:|---:|---:|---:|
| Anthropic CUDA12.9 rebuild | 1.083x (8 cells) | 1.138x (4 cells) | — | — |
| cuEquivariance 0.12 | 2.609x (10 cells) | 2.598x (10 cells) | 2.574x (10 cells) | 2.291x (10 cells) |
| engine Triton | 1.608x (10 cells) | 1.746x (10 cells) | 1.425x (10 cells) | 1.574x (10 cells) |
| DTv1 | 2.028x (10 cells) | 2.117x (10 cells) | 1.689x (10 cells) | 1.750x (10 cells) |
| compiled PyTorch | 5.269x (10 cells) | 6.250x (10 cells) | 4.142x (10 cells) | 4.908x (10 cells) |

## Full implementation latency inventory

| Variant | Mode | D | L | Engine | cuEquivariance | PyTorch compiled | DTv1 | Triton | Anthropic rebuild |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| outgoing | inference | 64 | 384 | 0.079520 | 0.159232 | 0.362448 | 0.172448 | 0.105824 | 0.092448 |
| outgoing | inference | 64 | 768 | 0.301184 | 0.614208 | 2.707616 | 0.630816 | 0.382912 | 0.328544 |
| outgoing | inference | 128 | 384 | 0.146512 | 0.279552 | 0.756448 | 0.369632 | 0.259328 | 0.168480 |
| outgoing | inference | 128 | 768 | 0.562784 | 1.088912 | 6.210464 | 1.400176 | 0.967632 | 0.618304 |
| outgoing | inference | 256 | 384 | 0.354784 | 0.716976 | 1.442112 | 0.814880 | 0.649120 | 0.367456 |
| outgoing | inference | 256 | 768 | 1.417088 | 2.858176 | 12.374416 | 3.211344 | 2.479840 | 1.509552 |
| outgoing | inference | 384 | 384 | 0.750704 | 2.906272 | 2.124896 | 1.370560 | 1.188896 | 0.784640 |
| outgoing | inference | 384 | 768 | 3.153920 | 11.653600 | 19.002111 | 5.528912 | 4.684832 | 3.224624 |
| outgoing | inference | 512 | 384 | 1.278128 | 5.174048 | 2.927904 | 1.987728 | 2.185696 | — |
| outgoing | inference | 512 | 768 | 5.060512 | 20.755360 | 25.761057 | 8.066272 | 8.795392 | — |
| outgoing | training | 64 | 384 | 0.269936 | 0.740992 | 1.241664 | 0.656864 | 0.535168 | — |
| outgoing | training | 64 | 768 | 0.956048 | 2.792032 | 8.130352 | 2.355104 | 1.895856 | — |
| outgoing | training | 128 | 384 | 0.772080 | 1.430112 | 2.395088 | 1.357728 | 1.017120 | — |
| outgoing | training | 128 | 768 | 2.791104 | 5.510880 | 18.832384 | 5.089312 | 3.780640 | — |
| outgoing | training | 256 | 384 | 1.729088 | 3.383072 | 4.656192 | 2.695936 | 2.195840 | — |
| outgoing | training | 256 | 768 | 6.733920 | 19.400768 | 37.849009 | 10.417152 | 8.945632 | — |
| outgoing | training | 384 | 384 | 2.793808 | 8.156960 | 6.861568 | 4.226848 | 3.641968 | — |
| outgoing | training | 384 | 768 | 11.357424 | 31.658400 | 58.865601 | 16.746752 | 15.348896 | — |
| outgoing | training | 512 | 384 | 4.334768 | 13.537632 | 9.311024 | 5.952624 | 5.497344 | — |
| outgoing | training | 512 | 768 | 17.935265 | 54.554657 | 78.610046 | 23.915584 | 23.433856 | — |
| bidirectional | inference | 64 | 384 | 0.118688 | 0.264800 | 0.672160 | 0.292512 | 0.181920 | 0.138032 |
| bidirectional | inference | 64 | 768 | 0.464480 | 1.063120 | 5.798112 | 1.112320 | 0.673648 | 0.507808 |
| bidirectional | inference | 128 | 384 | 0.249120 | 0.519344 | 1.293536 | 0.624480 | 0.436800 | 0.284992 |
| bidirectional | inference | 128 | 768 | 0.971552 | 1.988176 | 11.721456 | 2.416928 | 1.658016 | 1.120496 |
| bidirectional | inference | 256 | 384 | 0.601024 | 1.674304 | 2.553984 | 1.288736 | 1.084256 | — |
| bidirectional | inference | 256 | 768 | 2.323936 | 6.695712 | 24.102128 | 5.347936 | 4.347104 | — |
| bidirectional | inference | 384 | 384 | 1.152192 | 3.477200 | 3.940688 | 2.191296 | 2.111360 | — |
| bidirectional | inference | 384 | 768 | 4.931168 | 14.046560 | 37.168240 | 9.048096 | 8.502176 | — |
| bidirectional | inference | 512 | 384 | 1.853344 | 5.895264 | 5.410576 | 3.195616 | 3.670352 | — |
| bidirectional | inference | 512 | 768 | 8.100928 | 23.738352 | 50.120865 | 13.419040 | 15.169680 | — |
| bidirectional | training | 64 | 384 | 0.408128 | 1.133024 | 2.023904 | 1.061024 | 0.817376 | — |
| bidirectional | training | 64 | 768 | 1.511936 | 4.249056 | 17.118977 | 3.989072 | 3.036064 | — |
| bidirectional | training | 128 | 384 | 1.001376 | 2.297728 | 3.976752 | 2.117792 | 1.624336 | — |
| bidirectional | training | 128 | 768 | 4.071648 | 8.760768 | 35.260143 | 8.136592 | 6.568544 | — |
| bidirectional | training | 256 | 384 | 2.651872 | 5.512992 | 7.909408 | 4.188896 | 3.709200 | — |
| bidirectional | training | 256 | 768 | 10.976672 | 21.747680 | 73.315613 | 16.717537 | 16.343936 | — |
| bidirectional | training | 384 | 384 | 4.614752 | 10.176512 | 12.504800 | 6.788656 | 6.518016 | — |
| bidirectional | training | 384 | 768 | 19.535200 | 41.252464 | 116.849091 | 27.215839 | 28.046240 | — |
| bidirectional | training | 512 | 384 | 7.073280 | 16.328848 | 17.037664 | 9.832560 | 9.885056 | — |
| bidirectional | training | 512 | 768 | 29.318336 | 68.555008 | 158.203461 | 39.831327 | 43.929024 | — |

## Numerical and coverage limits

The official runner records output/input-gradient comparison to its FP32 reference, checks all timed gradients for finiteness, and checks graph replay. Read-only instrumentation also records every parameter-gradient discrepancy from that same runner reference. These are diagnostic metrics, not a new all-parameter numerical qualification or sanitizer campaign. No tolerance was changed. Recorded earlier experiments are not substituted for current default timings. Some post-timing CUDA Graph profiler captures contain no kernel events; separately labeled eager dispatch traces supplement later cells. Empty traces are not counted as native/CUDA-only evidence. This campaign does not cover incoming-only modules, other batches, six-bucket inference coverage, FP32 activations or FP32-master performance, optimizer steps, whole-model training or SOL90.

Completed comparative cells: 40/40. Raw status counts: {'ok': 212, 'unsupported': 8}.

Failures and unavailable rows are retained in the raw CSVs:

- triangle_multiplication/inference/D512/L384/anthropic_hybrid: unsupported; UnsupportedBenchmark: miniworld_engine.integrations.anthropic_trimul.PayloadUnavailable: the anthropic TriMul payload cannot serve this call: the payload has no unit for (c_z=512, 
- triangle_multiplication/inference/D512/L768/anthropic_hybrid: unsupported; UnsupportedBenchmark: miniworld_engine.integrations.anthropic_trimul.PayloadUnavailable: the anthropic TriMul payload cannot serve this call: the payload has no unit for (c_z=512, 
- triangle_multiplication_bidirectional/inference/D256/L384/anthropic_hybrid: unsupported; UnsupportedBenchmark: miniworld_engine.integrations.anthropic_trimul.PayloadUnavailable: the anthropic TriMul payload cannot serve this call: the payload has no unit for (c_z=256, 
- triangle_multiplication_bidirectional/inference/D256/L768/anthropic_hybrid: unsupported; UnsupportedBenchmark: miniworld_engine.integrations.anthropic_trimul.PayloadUnavailable: the anthropic TriMul payload cannot serve this call: the payload has no unit for (c_z=256, 
- triangle_multiplication_bidirectional/inference/D512/L384/anthropic_hybrid: unsupported; UnsupportedBenchmark: miniworld_engine.integrations.anthropic_trimul.PayloadUnavailable: the anthropic TriMul payload cannot serve this call: the payload has no unit for (c_z=512, 
- triangle_multiplication_bidirectional/inference/D512/L768/anthropic_hybrid: unsupported; UnsupportedBenchmark: miniworld_engine.integrations.anthropic_trimul.PayloadUnavailable: the anthropic TriMul payload cannot serve this call: the payload has no unit for (c_z=512, 
- triangle_multiplication_bidirectional/inference/D384/L384/anthropic_hybrid: unsupported; UnsupportedBenchmark: miniworld_engine.integrations.anthropic_trimul.PayloadUnavailable: the anthropic TriMul payload cannot serve this call: the payload has no unit for (c_z=384, 
- triangle_multiplication_bidirectional/inference/D384/L768/anthropic_hybrid: unsupported; UnsupportedBenchmark: miniworld_engine.integrations.anthropic_trimul.PayloadUnavailable: the anthropic TriMul payload cannot serve this call: the payload has no unit for (c_z=384, 

## Artifacts

- [Comparison chart](../../benchmarks/modules/triangle_multiplication/artifacts/NVIDIA%20H100%2080GB%20HBM3/h100_trimul_compare_20261007_speedups.svg)
- [Summary CSV](../../benchmarks/modules/triangle_multiplication/artifacts/NVIDIA%20H100%2080GB%20HBM3/h100_trimul_compare_20261007_summary.csv)
- Full raw runner CSVs, `.run.json` compile/source provenance, profiler/parameter diagnostics and source-hash manifest are copied to the respective module artifact directories.
- Frozen source, job logs and measurement wrapper: `/home/psk6950/miniworld-engine-scratch/trimul-compare-20261007/`.
