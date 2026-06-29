# ConditionedTransition training — fused-prologue triton dgrad (wgrad=cuBLAS) — H100, fp32/TF32

_Sources: `training_fused_v2.out` (e2e eager + CUDA-graph), `training_fused_perstage_cudagraph.out`
(per-stage CUDA-graph micro). All dgrad matmuls are `tl.dot(input_precision="tf32")` triton
kernels with the producing elementwise FUSED into the GEMM prologue. wgrad GEMMs stay on cuBLAS
(per directive). AdaLN out of scope._

## What was built (this phase)

`train_fused.py` / `ConditionedTransitionTailFusedFunction`:
- **Forward:** the fused inference kernels, emitting saved tensors (`h`, packed `ab=(M,2ND)`,
  `out`, `scale`).
- **dgrad with elementwise FUSED into the prologue:**
  - `_dh_gatebwd`: `dh = (sigmoid(scale)*dy) @ Ws` — gate-bwd forms `dout` in-register per
    K-tile; also emits materialized `dout, dscale` for the cuBLAS wgrads.
  - `_dx_swiglubwd`: `dx = dab @ Wcat` as **one concatenated GEMM** (fix: was 2 dots/tile) —
    swiglu-bwd forms `dab=[da|db]` in-register per reduction tile; emits `dab` for cuBLAS dWa/dWb.
  - `_dgemm` (GROUP_M-swizzled, autotuned) for `dcond = dscale @ Wsc`.
- **wgrad:** cuBLAS (TF32) — `dWs, dWsc, dWa, dWb` (reductions over M; cuBLAS's domain, left there).

## Correctness — cos vs autograd-through-torch reference (TF32)

All 7 grads (dx, dcond, dWa, dWb, dWs, dWsc, db_sc) + cos_y = **1.00000**, every shape. PASS.

## The measurement trap: EAGER fwd+bwd is autograd-overhead-bound, not kernel-bound

Eager `autograd.grad` fwd+bwd is ~330–390 us for **every** path (fused, cuBLAS-train, eager-ref)
regardless of shape — that floor is the autograd engine + Python dispatch, which masks the actual
kernel work. Under **CUDA graph** (kernels only) the same paths are 90–270 us. So eager fwd+bwd
ratios are meaningless here; CUDA-graph numbers are the truth (cf. the repo's standing note:
measure graph-break kernels under CUDA-graph, not eager).

## CUDA-graph end-to-end fwd+bwd (us) — true kernel cost

| stream | M | d | fused-triton | cuBLAS-train | eager-ref | vs eager | vs cuBLAS-train |
|---|---|---|---|---|---|---|---|
| atom | 2048 | 128 | 106.9 | 89.3 | 110.6 | **1.03×** | 0.84× |
| atom | 4096 | 128 | 147.8 | 116.8 | 142.8 | 0.97× | 0.79× |
| atom | 8192 | 128 | 243.5 | 163.7 | 200.2 | 0.82× | 0.67× |
| token | 384 | 768 | 356.5 | 147.2 | 174.3 | 0.49× | 0.41× |
| token | 512 | 768 | 368.0 | 166.3 | 196.2 | 0.53× | 0.45× |
| token | 768 | 768 | 568.5 | 211.4 | 234.9 | 0.41× | 0.37× |
| token | 1024 | 768 | 600.7 | 232.8 | 266.8 | 0.44× | 0.39× |

Fused-triton **beats torch-eager at atom M=2048 (1.03×)** and is near-parity at 4096, but loses
at large atom and badly at token. cuBLAS-train is fastest everywhere.

## Per-stage CUDA-graph micro — fused-triton dgrad vs (cuBLAS GEMM + separate elementwise)

Isolates the question with launch AND autograd overhead removed: does folding the elementwise
into a triton GEMM beat a cuBLAS GEMM + a separate elementwise kernel?

| stream | M | d | stage | fused-triton us | cuBLAS+elem us | ratio (fused/cuBLAS) |
|---|---|---|---|---|---|---|
| atom | 2048 | 128 | dh+gatebwd | 14.9 | 7.5 | 1.97× |
| atom | 2048 | 128 | dx+swiglubwd | 27.1 | 11.3 | 2.40× |
| atom | 2048 | 128 | dcond | 9.0 | 5.5 | 1.63× |
| atom | 8192 | 128 | dh+gatebwd | 49.3 | 13.4 | 3.67× |
| atom | 8192 | 128 | dx+swiglubwd | 55.9 | 26.6 | 2.10× |
| token | 384 | 768 | dh+gatebwd | 94.7 | 13.9 | 6.83× |
| token | 384 | 768 | dx+swiglubwd | 132.4 | 24.4 | 5.43× |
| token | 1024 | 768 | dh+gatebwd | 153.3 | 21.5 | 7.12× |
| token | 1024 | 768 | dx+swiglubwd | 233.2 | 38.2 | 6.11× |

**The fused-triton dgrad loses every stage, 1.6–7.1×, even though "cuBLAS+elem" pays an extra
separate elementwise launch.** The fusion successfully removes the elementwise HBM pass, but the
triton GEMM itself is multiples slower than the cuBLAS GEMM at these shapes, and the gap WIDENS
with M and with d=768 — the opposite of a launch-overhead/under-tuning artifact (which would
shrink as a fraction at large M). This is a genuine per-kernel compute gap, measured with launch
and autograd overhead removed.

## Why inference-triton beats cuBLAS but backward-triton does not

The inference win was a **b2b fusion that eliminates a large HBM intermediate** (expand→SwiGLU→
squeeze in one kernel; the (M,ND) `h` is never written) — a memory-traffic win across a GEMM
*pair*; the individual GEMMs were not faster than cuBLAS. Backward has no fusible GEMM pair:
`dh`, `dcond`, `dx` each feed a different cuBLAS wgrad that needs the operand materialized, so
there is no large intermediate to eliminate — only small elementwise passes, which cannot offset
GEMMs that are 2–7× slower.

## What was pushed (not a hand-wave "ceiling")

Per the course-correction's concrete suggestions, all implemented and measured: `dx` rebuilt as
one concatenated GEMM (was 2 dots/tile → ~2× faster but still 2.5–3.6× off cuBLAS); GROUP_M
L2-swizzle + inference-grade autotune configs (BLOCK_K 64/128, wide M tiles); gate-bwd and
swiglu-bwd folded into the consuming GEMM prologue; CUDA-graph measurement to remove the launch
floor; per-stage isolation. The triton-GEMM-vs-cuBLAS-GEMM gap survives all of it.

## Decision

- **wgrad stays cuBLAS** (directive).
- The **fused-prologue dgrad path** (`cond_transition_train_fused`) is correct, fully fuses the
  elementwise into the GEMM as specified, shipped and selectable. Under CUDA graph it
  reaches/beats torch-eager at atom M≤4096 but does not beat the cuBLAS-GEMM hybrid.
- **Production default stays `cond_transition_train`** (cuBLAS GEMMs + fused-triton elementwise) —
  fastest correct path under CUDA graph at every measured shape.
- Open lever (stated as evidence, not a ceiling claim): closing the triton-vs-cuBLAS *single-GEMM*
  gap at these M-heavy/short-K shapes needs a much deeper SM90 WGMMA triton GEMM (warp-specialized
  / TMA / persistent) than autotuning the standard tiled kernel provides — the measured per-stage
  gap is the evidence for why the standard path doesn't reach it.
