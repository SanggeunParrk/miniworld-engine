# ConditionedTransition inference vs torch.compile — H100, fp32 / TF32

_Primary baseline = `torch.compile(reference, mode="reduce-overhead")` of the pure-pytorch
reference (graph-break check: graph_count=1, break_count=0 — compiles cleanly, no free win).
OURS measured under our own CUDA graph; COMPILE uses its own cudagraphs — apples-to-apples.
Eager is a context column. Source: `inference_vs_compile_cudagraph.out`._

Dispatch: atom (d≤128) → fused b2b single kernel; token (d≥256) → composed 2-kernel.

| stream | M | d | cos | ours us | compile us | eager us | **vs compile** | vs eager |
|---|---|---|---|---|---|---|---|---|
| atom | 2048 | 128 | 0.999999 | 25.1 | 76.5 | 31.0 | **3.05×** | 1.23× |
| atom | 4096 | 128 | 0.999999 | 25.4 | 53.3 | 39.3 | **2.10×** | 1.55× |
| atom | 8192 | 128 | 0.999999 | 27.2 | 69.8 | 57.6 | **2.56×** | 2.12× |
| token | 384 | 768 | 0.999999 | 30.0 | 67.6 | 50.6 | **2.25×** | 1.69× |
| token | 512 | 768 | 0.999999 | 31.3 | 69.6 | 51.6 | **2.23×** | 1.65× |
| token | 768 | 768 | 0.999999 | 53.3 | 78.7 | 60.7 | **1.48×** | 1.14× |
| token | 1024 | 768 | 0.999999 | 60.9 | 86.6 | 68.0 | **1.42×** | 1.11× |

**Ours beats torch.compile at every shape (1.42–3.05×).** The earlier "token M=384 = 0.91×
compile" was a measurement artifact (not apples-to-apples graph-captured); under correct
CUDA-graph-vs-cudagraph it is **2.25×**. Lowest margins are token M≥768 (1.42–1.48×) where the
composed path's `h` HBM round-trip + 2 GEMMs scale with M — still a clear win, so no change.
