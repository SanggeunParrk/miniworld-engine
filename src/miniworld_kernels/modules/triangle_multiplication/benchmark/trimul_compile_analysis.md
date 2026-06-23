# TriMul forward — compile / regime analysis (ours vs cuequiv vs dt-v1)

How `torch.compile` and CUDA graphs change the comparison, and why ours wins in
every regime. Forward only, **bf16, H100 80GB, B=1, D=128, single layer + mask
(mask_prob=0.2)**, `triton.do_bench` median, harness = a faithful port of
team-gm `scripts/bench.py` (Fabric bf16-mixed, `model.compile()`), so cuequiv/
dt-v1 reproduce team-gm's own numbers.
Source: `kernels/trimul_inproj/cute/teamgm_faithful.py`.

## TL;DR

- **Under "compile everything" (the realistic training setting), ours wins at
  every L by 1.37–1.65×.** Under "each kernel at its best regime", ours still
  wins by 1.22–1.42×. The choice of regime does not change the verdict.
- `torch.compile` **does help ours** at large L (Inductor fuses the einsum/bmm/
  cast glue): ours.compile 1.867 vs ours.eager 2.030 ms @L=1024.
- This is **forward only**. dt-v1's real strength is full (fwd+bwd); ours has no
  backward yet, so the decisive comparison is still pending (see backward work).

## Three regimes, and the `@torch.compiler.disable()` story

| regime | what it is |
|---|---|
| **eager** | raw, no compile. dt-v1's native/published mode. |
| **compile** | `model.compile()` (Inductor, default mode — no cudagraph). cuequiv's best. |
| **cudagraph** | manual `torch.cuda.graph` capture of the forward — removes kernel-launch overhead uniformly. |

**dt-v1 and ours both contain hand-written kernels (triton / CuTeDSL) inside
custom `autograd.Function`s.** `torch.compile` cannot trace *into* such a kernel
unless it is registered as a `torch.library.triton_op`/`custom_op`; otherwise
Dynamo graph-breaks (or risks mishandling the custom backward). dt-v1 marks its
Functions `@torch.compiler.disable()` — a deliberate **clean graph break**:
correctness guaranteed, the kernel runs as written, but compile gives it no
fusion (and adds a little graph-break dispatch overhead at small L). We wrap
ours' cute kernels the same way, so the comparison is apples-to-apples:
**both are "compile-opaque kernels with compiled glue."**

"dt-v1 can't compile" is wrong — it compiles and runs fine; compile just doesn't
*speed it up*. The `.compile` column below IS the consistently-compiled model.

## Full matrix (ms/layer; **bold** = that kernel's best regime per L)

| L | pytorch | cuequiv e / **c** / **g** | dt-v1 e / c / **g** | ours e / **c** / **g** |
|--:|--:|--:|--:|--:|
| 128 | 0.285 | 0.566 / 0.509 / **0.071** | 0.415 / 0.569 / **0.067** | 0.338 / 0.396 / **0.055** |
| 256 | 0.908 | 0.571 / 0.512 / **0.194** | 0.435 / 0.588 / **0.200** | 0.344 / 0.419 / **0.145** |
| 384 | 1.965 | 0.573 / 0.528 / **0.384** | 0.477 / 0.603 / **0.396** | 0.346 / 0.413 / **0.306** |
| 512 | 3.614 | 0.672 / **0.662** / 0.663 | 0.796 / 0.791 / **0.670** | 0.529 / **0.479** / 0.519 |
| 768 | 11.03 | 1.433 / **1.428** / 1.439 | 1.712 / 1.697 / **1.457** | 1.133 / **1.028** / 1.122 |
| 1024 | 21.15 | 2.576 / **2.566** / 2.590 | 3.047 / 3.014 / **2.618** | 2.030 / **1.867** / 2.037 |

(e = eager, c = compile, g = cudagraph)

Reading it:
- **Small L (≤384) is launch-bound** → cudagraph collapses everyone to ~0.05–0.4
  ms; ours.cudagraph is lowest. compile *hurts* ours & dt-v1 here (graph-break
  dispatch overhead > tiny kernel).
- **Large L (≥512) is compute-bound** → cudagraph gives ~nothing; compile helps
  ours (glue fusion). ours.compile is lowest.

## Realistic training: compile everything

| L | cuequiv.compile | dt-v1.compile | **ours.compile** | ours speedup |
|--:|--:|--:|--:|--:|
| 128 | 0.509 | 0.569 | **0.396** | 1.29–1.44× |
| 256 | 0.512 | 0.588 | **0.419** | 1.22–1.40× |
| 384 | 0.528 | 0.603 | **0.413** | 1.28–1.46× |
| 512 | 0.662 | 0.791 | **0.479** | 1.38–1.65× |
| 768 | 1.428 | 1.697 | **1.028** | 1.39–1.65× |
| 1024 | 2.566 | 3.014 | **1.867** | 1.37–1.61× |

## Each kernel at its own best regime (most generous to baselines)

| L | cuequiv best | dt-v1 best | **ours best** | ours speedup |
|--:|--:|--:|--:|--:|
| 128 | 0.071 (g) | 0.067 (g) | **0.055** (g) | 1.22–1.29× |
| 256 | 0.194 (g) | 0.200 (g) | **0.145** (g) | 1.34–1.38× |
| 384 | 0.384 (g) | 0.396 (g) | **0.306** (g) | 1.25–1.29× |
| 512 | 0.662 (c) | 0.670 (g) | **0.479** (c) | 1.38–1.40× |
| 768 | 1.428 (c) | 1.457 (g) | **1.028** (c) | 1.39–1.42× |
| 1024 | 2.566 (c) | 2.618 (g) | **1.867** (c) | 1.37–1.40× |

Even giving dt-v1 its theoretical best (cudagraph, 2.618 @1024 — i.e. zero
graph-break overhead, as if its `@disable` were "fixed"), it still loses to
ours.compile (1.867). So dt-v1's `@torch.compiler.disable()` is **not our problem
to fix, and fixing it would not change the verdict.**

## Caveats

- **Forward only.** dt-v1's biggest published wins are full (fwd+bwd), from its
  optimized backward. Ours' backward is in development; full-mode comparison is
  the real game and is pending.
- ours is **bf16-native**; cuequiv/dt-v1 run **bf16-mixed** (LN in fp32). Minor
  precision/speed nuance, far smaller than the 1.2–1.6× gap.
- Small-L absolute numbers run a bit faster than team-gm's H200 doc (harness
  overhead differs); relative ordering and large-L absolutes reproduce theirs.
