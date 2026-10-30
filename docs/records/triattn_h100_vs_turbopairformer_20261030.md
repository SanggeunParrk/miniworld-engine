# H100 triangle-attention operator: the engine core against TurboPairFormer and cuEquivariance (2026-10-30)

TurboPairFormer (arXiv 2610.05854, Lambda / Stevens / OpenFold; `pip install turbopairformer==0.1.3.post1`, Apache-2.0) is a
training kernel for the triangle-attention operator on H100 / H200. This record measures it against the engine's sm90 core at the
operator level, under the paper's own accuracy and determinism protocol.

## Setup

- One H100 80GB HBM3 per task, Slurm job 39364 (the TurboPairFormer accuracy rows were re-run in a follow-up 1-GPU job after a
  harness fix).
- **Engine:** main at f6536836 (2.4.0), torch 2.13.0+cu129, kernels from the 2.4.0 bundle (code-graph keys `triattn_h100_core_r2_v12`
  995d75e2, `triattn_h100_bwd_v12` 30876aa8; JIT fallback off).
  - The operator is the module's core without its front and tail: `triattn_fwd` + `triattn_delta` + `triattn_bwd_kv` + `bwd_q`.
  - The bias goes in as the core's operand `bf16(bias / scale)`.
- **TurboPairFormer:** its wheel is cp314 and pins torch 2.10.0, so it runs in a separate pixi env (Python 3.14, torch 2.10.0+cu128).
  It uses the package defaults: single reconstruction plus the output-residual "shadow".
- **cuEquivariance:** `cuequivariance_torch.triangle_attention` 0.12, engine env.
- **Layouts:**
  - The canonical tensors are token-major `[B, O, N, H, D]`: what a projection writes, the engine core's layout, and TurboPairFormer's
    output layout.
  - `turbo` gets the `[B, O, H, N, D]` view of them, so its fused pack runs, as inside a module.
  - `turbo_hmajor` gets a contiguous head-major copy made outside the timing (its best case).
  - cuEquivariance gets its head-major layout.
- **Timing:** CUDA graph replays, median of 7 x 20.
  - Forward without grad, forward with saved state, and forward + backward; the backward is the difference of the last two.
  - Peak is `max_memory_allocated` above the inputs over one forward + backward.
- **Harness:** `~/miniworld-engine-scratch/turbopf/` (`opbench.py`, `run.sbatch`, `collect.py`; tables in `results/39364/tables.md`).
  Every implementation gets identical bf16 inputs from numpy PCG64.

## Time (ms), B = 1, H = 4, D = 32

| N | forward: engine | forward: turbo | forward: turbo_hmajor | forward: cuEq | fwd+bwd: engine | fwd+bwd: turbo | fwd+bwd: turbo_hmajor | fwd+bwd: cuEq | turbo / engine (fwd+bwd) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.060 | 0.155 | 0.121 | 0.113 | 0.249 | 0.566 | 0.530 | 0.687 | 2.28x |
| 384 | 0.157 | 0.392 | 0.313 | 0.341 | 0.677 | 1.526 | 1.465 | 2.061 | 2.25x |
| 512 | 0.315 | 0.800 | 0.660 | 0.767 | 1.476 | 3.257 | 3.111 | 4.687 | 2.21x |
| 640 | 0.563 | 1.426 | 1.232 | 1.481 | 2.751 | 6.023 | 5.763 | 8.927 | 2.19x |
| 768 | 0.905 | 2.352 | 2.046 | 2.567 | 4.646 | 10.419 | 10.041 | 15.259 | 2.24x |
| 1024 | 2.059 | 5.139 | 4.582 | 5.943 | 10.729 | 23.366 | 22.717 | 36.505 | 2.18x |
| 1536 | 6.763 | 16.483 | 15.144 | 22.767 | 35.844 | 80.142 | 78.632 | 131.762 | 2.24x |
| 2048 | 15.881 | 39.097 | 36.827 | 60.578 | 82.620 | 179.697 | 176.596 | 321.505 | 2.17x |

- TurboPairFormer is 1.46-1.79x faster than cuEquivariance on forward + backward. The paper claims 35.6-45.6 % less time, so the
  harness reproduces their claim.
- With H = 8 (N 384 / 768 / 1536) the engine is 2.20-2.29x faster on forward + backward and 2.37-2.63x on the forward.

**Peak memory, forward + backward (MiB):**

| N | engine | turbo | cuEq |
|---:|---:|---:|---:|
| 768 | 657 | 3161 | 909 |
| 1536 | 2520 | 18801 | 3636 |
| 2048 | 4480 | 41424 | 6464 |

## Determinism

Five forward + backward calls per input, N 384 / 640 / 768, seeds 0-9, bias 0 and N(0, 1): 60 cases.

| implementation | bitwise identical in all five outputs |
|---|---|
| engine | 60 / 60 |
| turbo | 60 / 60 |
| cuEquivariance | 0 / 60 (dQ and dbias differ every case) |

## Accuracy

This is the paper's protocol:
- q / k / v ~ N(0, 1) with 0.1 % N(0, 10^2) outliers, dO ~ N(0, 1), bf16, zero bias.
- Absolute RMSE against an fp64 reference computed from the same bf16 inputs.
- 10 seeds.

The table divides each RMSE by the bf16 rounding floor of the reference (1.0 = as good as storing the exact result in bf16).

| N | impl | O | dQ | dK | dV | dbias |
|---:|---|---:|---:|---:|---:|---:|
| 384 | engine | 1.10 | 2.62 | 2.05 | 1.39 | 1.44 (fp32; 1.76 rounded to bf16) |
| 384 | turbo | 1.04 | 1.76 | 1.42 | 1.34 | 1.42 (bf16) |
| 384 | cuEq | 1.04 | 2.53 | 1.98 | 1.34 | 1.35 (fp32) |
| 768 | engine | 1.09 | 2.69 | 2.24 | 1.49 | 1.70 (fp32; 1.97 rounded to bf16) |
| 768 | turbo | 1.02 | 1.76 | 1.42 | 1.33 | 1.42 (bf16) |
| 768 | cuEq | 1.02 | 2.57 | 2.13 | 1.33 | 1.57 (fp32) |

The engine is at cuEquivariance's level. TurboPairFormer's dQ and dK errors are about a third lower; the paper's accuracy claim holds.
Its package source shows where the difference comes from:
- The forward stores the bf16 rounding residual of O (`turbopairformer_triangle_forward_shadow_out`).
- The delta kernel computes rowsum(dO * (O + residual)).
- The engine's `triattn_delta` reads the rounded bf16 O.

A bias ~ N(0, 1) run (3 seeds) is in the harness tables. It is not comparable for the engine: the harness rounds the bf16 bias once
more (into `bias / scale`), which the module's front does not do.

## Not measured here

- The modules around the operator (LayerNorm, projections, gate, out-projection).
- TurboPairFormer's TriMul.
- Head dim 16.
- Training-step time.
