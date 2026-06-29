# Bidirectional bias-only triangle attention: triton vs torch.compile (H100, bf16)

_Source: `src/miniworld_kernels/modules/triangle_attention/benchmark/bidir.out`_

## forward: speedup vs PyTorch default (×)

_higher is better; table rows = d, columns = M, bold = fastest. See `bidir_fwd_speedup.png`._

_speedup of Triton vs PyTorch (bold = best)_

| d (=d_in=d_out) | M=65536 | M=147456 | M=262144 | M=589824 | M=1048576 |
|---|---|---|---|---|---|
| 128 | **0.69×** | **0.54×** | **0.56×** | **0.58×** | **0.60×** |
| 256 | **0.69×** | **0.66×** | **0.68×** | **0.71×** | — |
| 512 | **1.15×** | **1.13×** | **1.13×** | — | — |

![forward speedup](bidir_fwd_speedup.png)

### forward latency (ms)

_absolute latency, log scale, lower is better; rows = d, columns = M_

_backends: pytorch / triton (bold = fastest)_

| d (=d_in=d_out) | M=65536 | M=147456 | M=262144 | M=589824 | M=1048576 |
|---|---|---|---|---|---|
| 128 | **0.3070** / 0.4450 | **0.5670** / 1.0430 | **0.9310** / 1.6750 | **2.0440** / 3.5020 | **3.7150** / 6.1530 |
| 256 | **0.4980** / 0.7190 | **1.0200** / 1.5360 | **1.7380** / 2.5420 | **3.8700** / 5.4400 | — |
| 512 | 1.4910 / **1.2970** | 3.2200 / **2.8420** | 5.5550 / **4.9160** | — | — |

![forward latency](bidir_fwd_latency.png)

## forward + backward: speedup vs PyTorch default (×)

_higher is better; table rows = d, columns = M, bold = fastest. See `bidir_fwd_bwd_speedup.png`._

_speedup of Triton vs PyTorch (bold = best)_

| d (=d_in=d_out) | M=65536 | M=147456 | M=262144 | M=589824 | M=1048576 |
|---|---|---|---|---|---|
| 128 | **0.56×** | **0.92×** | **1.05×** | **1.05×** | **1.04×** |
| 256 | **0.69×** | **0.70×** | **0.72×** | **0.74×** | — |
| 512 | **1.16×** | **1.11×** | **1.11×** | — | — |

![forward + backward speedup](bidir_fwd_bwd_speedup.png)

### forward + backward latency (ms)

_absolute latency, log scale, lower is better; rows = d, columns = M_

_backends: pytorch / triton (bold = fastest)_

| d (=d_in=d_out) | M=65536 | M=147456 | M=262144 | M=589824 | M=1048576 |
|---|---|---|---|---|---|
| 128 | **0.8060** / 1.4470 | **1.5670** / 1.7060 | 2.6410 / **2.5140** | 5.7520 / **5.4780** | 10.4900 / **10.0390** |
| 256 | **1.3900** / 2.0000 | **2.8430** / 4.0620 | **4.8900** / 6.7880 | **11.0530** / 14.9260 | — |
| 512 | 4.2370 / **3.6460** | 8.5920 / **7.7600** | 14.9110 / **13.4580** | — | — |

![forward + backward latency](bidir_fwd_bwd_latency.png)
