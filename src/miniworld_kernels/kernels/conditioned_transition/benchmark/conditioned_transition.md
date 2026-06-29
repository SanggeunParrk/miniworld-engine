# ConditionedTransition tail — bench (H100, fp32 / TF32)

_Source: `/home/psk6950/miniworld-kernels/_ct_tmp/ct_bench_9903.out`_

AdaLN is out of scope; `x` is the post-AdaLN activation. Inference dispatch: atom (d<=128) -> fused b2b, token (d>=256) -> composed 2-kernel. Training: autograd Function (cuBLAS GEMMs + fused triton elementwise). Baselines: torch eager and torch.compile, both TF32.

## Inference: correctness + latency

_cos vs torch eager TF32 reference; us per call; speedup = baseline/ours (higher better)._

| stream | M | d | cos | maxerr | ours_us | eager_us | compile_us | vs_eager | vs_compile |
|---|---|---|---|---|---|---|---|---|---|
| atom | 2048 | 128 | 0.999999 | 5.52e-03 | 35.6 | 56.7 | 67.1 | 1.59x | 1.89x |
| atom | 4096 | 128 | 0.999999 | 6.62e-03 | 35.7 | 56.9 | 73.5 | 1.59x | 2.06x |
| atom | 8192 | 128 | 0.999999 | 5.47e-03 | 35.3 | 67.6 | 72.4 | 1.92x | 2.05x |
| token | 384 | 768 | 0.999999 | 4.83e-03 | 101.2 | 108.2 | 92.6 | 1.07x | 0.91x |
| token | 512 | 768 | 0.999999 | 4.83e-03 | 58.5 | 68.7 | 92.0 | 1.18x | 1.57x |
| token | 768 | 768 | 0.999999 | 6.24e-03 | 58.2 | 71.6 | 94.5 | 1.23x | 1.62x |
| token | 1024 | 768 | 0.999999 | 6.24e-03 | 62.0 | 79.7 | 95.0 | 1.29x | 1.53x |

## Training (fwd+bwd): correctness + latency

_cos_y = output cosine; cos_min = worst grad cosine (over dx,dcond,dWa,dWb,dWs,dWsc,db_sc); us per fwd+bwd._

| stream | M | d | cos_y | cos_min | ours_us | eager_us | compile_us | vs_eager | vs_compile |
|---|---|---|---|---|---|---|---|---|---|
| atom | 2048 | 128 | 1.00000 | 1.00000 | 463.7 | 448.3 | 393.5 | 0.97x | 0.85x |
| atom | 4096 | 128 | 1.00000 | 1.00000 | 338.4 | 327.8 | 395.0 | 0.97x | 1.17x |
| atom | 8192 | 128 | 1.00000 | 1.00000 | 354.6 | 323.3 | 391.2 | 0.91x | 1.10x |
| token | 384 | 768 | 1.00000 | 1.00000 | 361.3 | 333.1 | 422.1 | 0.92x | 1.17x |
| token | 512 | 768 | 1.00000 | 1.00000 | 363.2 | 342.7 | 434.6 | 0.94x | 1.20x |
| token | 768 | 768 | 1.00000 | 1.00000 | 370.5 | 342.0 | 425.5 | 0.92x | 1.15x |
| token | 1024 | 768 | 1.00000 | 1.00000 | 360.5 | 346.0 | 421.1 | 0.96x | 1.17x |

![inference](ct_inference.png)

![fwd+bwd](ct_fwd_bwd.png)
