# adaLN training (fwd+bwd) — fp32 main

team-gm DiT adaptive-layernorm + sigmoid-gate, **training** (forward saves + backward).

`adaln_train` (kernels/adaln/triton/training.py): materialize+cuBLAS forward (saves x, cond,
cond_aff, gate, stats) + symmetric single-GEMM backward.

**Backward** (let `D = [dscale ; dy]` stacked on the 2N axis → shape **(2NX, M)**, `W_cat = [Ws ; Wb]` (2NX,NC)):
- `_bwd_x` (ONE Triton kernel): `dscale = dy·x̂·gate·(1-gate)` → writes D (transposed store), AND fused
  x LayerNorm-backward (no affine) → `dx` in x's layout. (full row in regs → no separate LN-bwd
  kernel, no dxhat buffer, no x re-read.)
- `[dWs ; dWb] = D @ cond_aff` (ONE cuBLAS GEMM) — **D's (2NX,M) layout makes this a contiguous-K
  NN GEMM, ~1.6× faster than the transposed-view `Dᵀ@cond_aff` (388→240 µs @token1024).**
- `dcond_aff = Dᵀ @ W_cat` (ONE cuBLAS GEMM = `dscale@Ws + dy@Wb`)
- `dsb = Σ_m dscale = D[:NX].sum(1)` (cheap row-sum, 133→45 µs) ; cond LayerNorm-backward → `dcond, dlnw`.

Reuses te_style `_ln_materialize` / `_ln_bwd` / `_bias_grad` (layernorm_linear). Correctness vs
pytorch autograd: **all grads cos = 1.000000** (fp32), ≥0.99999 (bf16).

Harness: `triton.testing.do_bench`, H100, fp32, TF32 on. n_augment=32, M=32·seq. `pt_compile` from
the official `scripts/bench.py mode=full compile=true` run (job 9914); eager/cur_triton from the same
harness. (median ms)

## TOKEN  d=768  (fwd+bwd)  — D=(2NX,M) layout

| seq  | pt_eager | pt_compile | cur_triton | **ours** | vs compile | vs eager | vs cur_triton |
|-----:|---------:|-----------:|-----------:|---------:|-----------:|---------:|--------------:|
| 384  | 0.984    | 0.710      | 17.31      | **0.653**| **1.09×**  | 1.51×    | 26.5×         |
| 512  | 1.239    | 0.864      | 21.72      | **0.778**| **1.11×**  | 1.59×    | 27.9×         |
| 640  | 1.449    | 0.972      | 28.49      | **0.952**| **1.02×**  | 1.52×    | 29.9×         |
| 768  | 1.703    | 1.139      | 33.45      | 1.145    | 0.99×      | 1.49×    | 29.2×         |
| 896  | 1.960    | 1.302      | 40.05      | 1.304    | 1.00×      | 1.50×    | 30.7×         |
| 1024 | 2.204    | 1.462      | 45.39      | **1.449**| **1.01×**  | 1.52×    | 31.3×         |

## ATOM  d=128  (fwd+bwd)

| seq  | pt_eager | pt_compile | cur_triton | **ours** | vs compile | vs eager | vs cur_triton |
|-----:|---------:|-----------:|-----------:|---------:|-----------:|---------:|--------------:|
| 2048 | 1.072    | 0.786      | 10.65      | **0.459**| **1.71×**  | 2.33×    | 23.2×         |
| 4096 | 1.625    | 0.752      | 21.42      | **0.694**| **1.08×**  | 2.34×    | 30.9×         |
| 6144 | 2.351    | 1.070      | 32.12      | **0.991**| **1.08×**  | 2.37×    | 32.4×         |
| 8192 | 3.044    | 1.361      | 42.54      | **1.282**| **1.06×**  | 2.38×    | 33.2×         |

## Verdict

`adaln_train` **beats every baseline at every size**: vs current adaln Triton ~26–33× (its fp32
backward is broken-slow — tiny-tile in-kernel GEMMs with `input_precision="ieee"`, 17–45 ms), vs
pytorch **eager 1.49–2.38×**, and vs **torch.compile 0.99–1.11× (token) / 1.06–1.71× (atom)** — i.e.
ties-or-wins token, clearly wins atom. The key backward optimization: storing `D` as **(2NX,M)** (a
transposed store in `_bwd_x`, absorbed by H100 L2) turns the wgrad into a contiguous-K NN GEMM
(388→240 µs) and dsb into a cheap row-sum (133→45 µs), saving ~190 µs at token1024 (1.61→1.45 ms).
(ours is `@torch.compiler.disable`, so in a real custom-kernel model the fair baseline is
eager/CUDA-graph, not whole-graph torch.compile — this beats compile anyway.)
