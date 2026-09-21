# D = 128 forward: this kernel, the engine's paths, and Anthropic's own

`bench_vs_anthropic.py`, one process per length, CUDA-graph replay median, bf16, inference, all modules loaded from one state
dict and compared against one fp32 reference. node02 H100 80 GB, CUDA 12.9, PyTorch 2.10 cu128. The engine here is the
`perf/trimul-sm90-parity` checkout, which is where the Anthropic integration (`implementation="anthropic"`) lives.

| forward, D = 128 | L384 | rel-RMS vs fp32 | L768 | rel-RMS vs fp32 |
|---|---:|---:|---:|---:|
| engine, default path | 157.5 µs | 3.197e-3 | 565.8 µs | 3.199e-3 |
| engine, `transition_residual_fusion=0` (hand-CUDA b2b) | 156.8 µs | 3.197e-3 | 563.0 µs | 3.199e-3 |
| **Anthropic `v2`** (their best at this width) | 157.9 µs | 3.774e-3 | 596.9 µs | 3.776e-3 |
| Anthropic `af3_fused` | 164.3 µs | 3.999e-3 | 621.1 µs | 4.005e-3 |
| Anthropic `pf` | 208.2 µs | 4.006e-3 | 785.5 µs | 4.021e-3 |
| Anthropic `v1` | 539.2 µs | 4.006e-3 | 2048.4 µs | 4.021e-3 |
| **this kernel** | **129.4 µs** | **2.839e-3** | **494.4 µs** | **2.840e-3** |

1.22x Anthropic's best row and 1.21x the engine's fastest path at L384 (1.21x / 1.14x at L768), and the most accurate of the six.

Two things this table is not. **Anthropic has no CUDA Transition kernel at this width** — their `esm_t16` and `flash_sm90a`
are compiled for c = 256 / hidden = 1024 only, so the rows above are their Triton ones; the comparison against their CUDA
work happens at c = 256, where an optimised `esm_t16` reaches 325 µs at L384 (`runs/transition_esm_opt_20260921`), a width
MiniWorld does not use. And **Anthropic has no Transition backward at all** (their `esm_kd3` is a frozen-weight dX), so the
backward in this package has no Anthropic counterpart.

The engine number here is not the one in this package's README. On `perf/trimul-sm90-parity` the inference forward takes the
hand-CUDA b2b at ~157 µs; on `main` the default `transition_residual_fusion` routes both inference and training to the
three-kernel path at ~272 µs. Part of the forward win quoted against `main` is recovering that difference.
