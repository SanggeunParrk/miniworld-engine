# D = 128 forward: this kernel, the engine's paths, and Anthropic's own

`bench_vs_anthropic.py`, one process per length, CUDA-graph replay median, bf16, inference (forward only, no saves), all
modules loaded from one state dict and compared against one fp32 reference. node02 H100 80 GB, CUDA 12.9, PyTorch 2.10 cu128.
The engine here is the `perf/trimul-sm90-parity` checkout, which is what the Anthropic integration
(`implementation="anthropic"`) lives in.

| forward, D = 128 | L384 | rel-RMS vs fp32 | L768 | rel-RMS vs fp32 |
|---|---:|---:|---:|---:|
| engine, default path | 158.5 µs | 3.210e-3 | 566.3 µs | 3.212e-3 |
| engine, `transition_residual_fusion=0` (hand-CUDA b2b) | 157.9 µs | 3.210e-3 | 569.5 µs | 3.212e-3 |
| **Anthropic `v2`** (their best at this width) | 158.7 µs | 3.793e-3 | 597.4 µs | 3.794e-3 |
| Anthropic `af3_fused` | 165.5 µs | 4.022e-3 | 621.0 µs | 4.024e-3 |
| Anthropic `pf` | 208.2 µs | 4.006e-3 | 785.5 µs | 4.021e-3 |
| Anthropic `v1` | 539.2 µs | 4.006e-3 | 2048.4 µs | 4.021e-3 |
| **this kernel** (`-DFWD_SAVE=0`) | **145.9 µs** | **2.851e-3** | **536.3 µs** | **2.852e-3** |

So at MiniWorld's pair width this kernel is 1.09x Anthropic's best row and 1.08x the engine's own fastest path at L384
(1.11x and 1.06x at L768), and it is the most accurate of the six.

Two things this table is not. **Anthropic has no CUDA Transition kernel at this width** — their `esm_t16` and `flash_sm90a`
are compiled for c = 256 / hidden = 1024 only, so the rows above are their Triton ones; the comparison against their CUDA
work happens at c = 256, where an optimised `esm_t16` reaches 325 µs at L384 (`runs/transition_esm_opt_20260921`), a width
MiniWorld does not use. And **Anthropic has no Transition backward at all** (their `esm_kd3` is a frozen-weight dX), so the
2.0x backward in this package has no Anthropic counterpart to be measured against.

Note also that the engine number here is not the one in this package's README. On `perf/trimul-sm90-parity` the inference
forward takes the hand-CUDA b2b at ~158 µs; on `main` the default `transition_residual_fusion` routes both inference and
training to the three-kernel path at ~278 µs. Part of the forward win quoted against `main` is recovering that difference.
