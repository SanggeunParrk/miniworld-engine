# Bias-only triangle attention (single-dir): triton vs torch.compile (H100, bf16)

_Source: `src/miniworld_kernels/modules/triangle_attention/benchmark/bias_only.out`_

## forward: speedup vs PyTorch default (×)

_higher is better; table rows = d, columns = M, bold = fastest. See `bias_only_fwd_speedup.png`._

_speedup of Triton vs PyTorch (bold = best)_

| d (=d_in=d_out) | M=65536 | M=147456 | M=262144 | M=589824 | M=1048576 |
|---|---|---|---|---|---|
| 128 | **0.73×** | **0.92×** | **1.52×** | **1.50×** | **1.47×** |
| 256 | **1.92×** | **1.36×** | **1.33×** | **1.33×** | — |
| 512 | **1.38×** | **1.65×** | **1.65×** | — | — |

![forward speedup](bias_only_fwd_speedup.png)

### forward latency (ms)

_absolute latency, log scale, lower is better; rows = d, columns = M_

_backends: pytorch / triton (bold = fastest)_

| d (=d_in=d_out) | M=65536 | M=147456 | M=262144 | M=589824 | M=1048576 |
|---|---|---|---|---|---|
| 128 | **0.1570** / 0.2160 | **0.3090** / 0.3370 | 0.8720 / **0.5720** | 1.9400 / **1.2940** | 3.4920 / **2.3720** |
| 256 | 0.7290 / **0.3800** | 0.9840 / **0.7260** | 1.6830 / **1.2650** | 3.8090 / **2.8560** | — |
| 512 | 0.9640 / **0.7010** | 2.0940 / **1.2670** | 3.6810 / **2.2260** | — | — |

![forward latency](bias_only_fwd_latency.png)

## forward + backward: speedup vs PyTorch default (×)

_higher is better; table rows = d, columns = M, bold = fastest. See `bias_only_fwd_bwd_speedup.png`._

_speedup of Triton vs PyTorch (bold = best)_

| d (=d_in=d_out) | M=65536 | M=147456 | M=262144 | M=589824 | M=1048576 |
|---|---|---|---|---|---|
| 128 | **1.01×** | **0.46×** | **1.03×** | **1.04×** | **1.03×** |
| 256 | **1.00×** | **1.00×** | **1.00×** | **0.99×** | — |
| 512 | **1.00×** | **1.07×** | **1.07×** | — | — |

![forward + backward speedup](bias_only_fwd_bwd_speedup.png)

### forward + backward latency (ms)

_absolute latency, log scale, lower is better; rows = d, columns = M_

_backends: pytorch / triton (bold = fastest)_

| d (=d_in=d_out) | M=65536 | M=147456 | M=262144 | M=589824 | M=1048576 |
|---|---|---|---|---|---|
| 128 | 0.6090 / **0.6030** | **0.9250** / 2.0140 | 2.2840 / **2.2100** | 5.0330 / **4.8330** | 9.0410 / **8.7820** |
| 256 | **1.6050** / 1.6080 | **2.5520** / 2.5590 | 4.4230 / **4.4060** | **9.8160** / 9.8740 | — |
| 512 | **2.6460** / 2.6510 | 5.2540 / **4.9230** | 9.2190 / **8.5780** | — | — |

![forward + backward latency](bias_only_fwd_bwd_latency.png)
