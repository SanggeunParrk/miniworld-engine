# Transition module forward (H100 bf16, d=128, n=4)

_Source: `src/miniworld_kernels/modules/transition/benchmark/transition_module_forward.csv`_

## Speedup vs default (×)

_higher is better. See `transition_module_forward_speedup.png`._

_speedup vs PyTorch (bold = fastest at that point)_

| seq_len | Triton (prev) | Triton b2b (ours) | cute (ours) |
|---|---|---|---|
| 384 | 2.05× | **4.23×** | 2.86× |
| 512 | 2.11× | **4.38×** | 2.97× |
| 640 | 2.07× | **4.46×** | 3.07× |
| 768 | 2.07× | **4.45×** | 3.09× |
| 896 | 2.07× | **4.56×** | 3.10× |
| 1024 | 2.07× | **4.59×** | 3.12× |

![speedup](transition_module_forward_speedup.png)

### Absolute latency (ms)

_log scale, lower is better. See `transition_module_forward_latency.png`._

_absolute latency (ms), lower is better; bold = fastest_

| seq_len | PyTorch | Triton (prev) | Triton b2b (ours) | cute (ours) |
|---|---|---|---|---|
| 384 | 0.7834 | 0.3830 | **0.1852** | 0.2737 |
| 512 | 1.3601 | 0.6447 | **0.3106** | 0.4584 |
| 640 | 2.1069 | 1.0175 | **0.4723** | 0.6853 |
| 768 | 3.0140 | 1.4555 | **0.6772** | 0.9739 |
| 896 | 4.0796 | 1.9716 | **0.8949** | 1.3163 |
| 1024 | 5.3393 | 2.5747 | **1.1621** | 1.7101 |

![latency](transition_module_forward_latency.png)
