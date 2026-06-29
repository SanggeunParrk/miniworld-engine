# TriMul forward (H100 bf16, D=128; each kernel best regime)

_Source: `src/miniworld_kernels/modules/triangle_multiplication/benchmark/trimul_forward.csv`_

## Speedup vs default (×)

_higher is better. See `trimul_forward_speedup.png`._

_speedup vs PyTorch (bold = fastest at that point)_

| L | cuEquivariance | NVIDIA dt-v1 | ours |
|---|---|---|---|
| 128 | 4.01× | 4.25× | **5.18×** |
| 256 | 4.68× | 4.54× | **6.26×** |
| 384 | 5.12× | 4.96× | **6.42×** |
| 512 | 5.46× | 5.39× | **7.54×** |
| 768 | 7.72× | 7.57× | **10.73×** |
| 1024 | 8.24× | 8.08× | **11.33×** |

![speedup](trimul_forward_speedup.png)

### Absolute latency (ms)

_log scale, lower is better. See `trimul_forward_latency.png`._

_absolute latency (ms), lower is better; bold = fastest_

| L | PyTorch | cuEquivariance | NVIDIA dt-v1 | ours |
|---|---|---|---|---|
| 128 | 0.2850 | 0.0710 | 0.0670 | **0.0550** |
| 256 | 0.9080 | 0.1940 | 0.2000 | **0.1450** |
| 384 | 1.9650 | 0.3840 | 0.3960 | **0.3060** |
| 512 | 3.6140 | 0.6620 | 0.6700 | **0.4790** |
| 768 | 11.0300 | 1.4280 | 1.4570 | **1.0280** |
| 1024 | 21.1500 | 2.5660 | 2.6180 | **1.8670** |

![latency](trimul_forward_latency.png)
