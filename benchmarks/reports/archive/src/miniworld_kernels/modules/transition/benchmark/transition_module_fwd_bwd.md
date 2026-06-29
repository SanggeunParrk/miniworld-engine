# Transition module forward+backward (H100 bf16, d=128, n=4)

_Source: `src/miniworld_kernels/modules/transition/benchmark/transition_module_fwd_bwd.csv`_

## Speedup vs default (×)

_higher is better. See `transition_module_fwd_bwd_speedup.png`._

_speedup vs PyTorch (bold = fastest at that point)_

| seq_len | Triton (prev) | Triton b2b (ours) | cute (ours) |
|---|---|---|---|
| 384 | 1.64× | **1.92×** | 1.65× |
| 512 | 1.65× | **1.93×** | 1.67× |
| 640 | 1.64× | **1.94×** | 1.69× |
| 768 | 1.66× | **1.95×** | 1.69× |
| 896 | 1.66× | **1.96×** | 1.68× |
| 1024 | 1.66× | **1.97×** | 1.69× |

![speedup](transition_module_fwd_bwd_speedup.png)

### Absolute latency (ms)

_log scale, lower is better. See `transition_module_fwd_bwd_latency.png`._

_absolute latency (ms), lower is better; bold = fastest_

| seq_len | PyTorch | Triton (prev) | Triton b2b (ours) | cute (ours) |
|---|---|---|---|---|
| 384 | 2.3372 | 1.4261 | **1.2168** | 1.4139 |
| 512 | 4.0369 | 2.4428 | **2.0909** | 2.4236 |
| 640 | 6.2392 | 3.7930 | **3.2232** | 3.6899 |
| 768 | 8.8785 | 5.3620 | **4.5475** | 5.2617 |
| 896 | 12.0162 | 7.2299 | **6.1329** | 7.1413 |
| 1024 | 15.6812 | 9.4182 | **7.9680** | 9.2956 |

![latency](transition_module_fwd_bwd_latency.png)
