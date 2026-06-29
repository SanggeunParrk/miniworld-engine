# Transition forward (H100 bf16, d=128, n=4)

_Source: `src/miniworld_kernels/kernels/transition/benchmark/transition_forward.csv`_

## Speedup vs default (×)

_higher is better. See `transition_forward_speedup.png`._

_speedup vs PyTorch (bold = fastest at that point)_

| seq_len | Triton (orig) | Triton b2b (ours) | cute (ours) |
|---|---|---|---|
| 384 | 3.93× | **5.26×** | 3.97× |
| 512 | 4.01× | **5.43×** | 3.92× |
| 640 | 4.04× | **5.50×** | 3.96× |
| 768 | 3.97× | **5.51×** | 3.97× |
| 896 | 4.03× | **5.37×** | 4.09× |
| 1024 | 3.96× | **5.36×** | 4.10× |

![speedup](transition_forward_speedup.png)

### Absolute latency (ms)

_log scale, lower is better. See `transition_forward_latency.png`._

_absolute latency (ms), lower is better; bold = fastest_

| seq_len | PyTorch | Triton (orig) | Triton b2b (ours) | cute (ours) |
|---|---|---|---|---|
| 384 | 0.9897 | 0.2519 | **0.1880** | 0.2494 |
| 512 | 1.7115 | 0.4267 | **0.3153** | 0.4367 |
| 640 | 2.6374 | 0.6522 | **0.4798** | 0.6655 |
| 768 | 3.7640 | 0.9486 | **0.6831** | 0.9486 |
| 896 | 5.1017 | 1.2665 | **0.9495** | 1.2488 |
| 1024 | 6.6664 | 1.6837 | **1.2436** | 1.6255 |

![latency](transition_forward_latency.png)
