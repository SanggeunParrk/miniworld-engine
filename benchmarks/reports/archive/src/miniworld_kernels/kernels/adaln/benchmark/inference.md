# adaLN inference (forward-only) — fp32 main

team-gm DiT adaptive-layernorm + sigmoid-gate, **inference** (fwd-only, no backward saves).

Op: `y = sigmoid(scale)·LN(x) + bias`, where `scale = LN(cond)·lnw @ Wsᵀ + b`, `bias = LN(cond)·lnw @ Wbᵀ`.

Two inference kernels, dispatched by d:
- **materialize** (large d / token): `cond_aff=LN(cond)·lnw` (Triton) → **one cuBLAS GEMM** producing
  `[scale|bias]` (concat Ws,Wb) → fused `LN(x)+sigmoid-gate` epilogue (Triton). Replaces the eager/
  compiled path's two separate GEMMs with one; fp32 uses TF32.
- **fused** (small d / atom): ONE Triton kernel (LN cond, in-kernel GEMM, LN x, gate → Y). Writes
  only Y — strips the x_hat/cond_norm/gate/rstd backward-materialization the training fwd kernel pays.

Harness: `triton.testing.do_bench`, H100, fp32, TF32 on (matches `config/bench.yaml allow_tf32=true`),
n_augment=32, M = 32·seq. Baselines: pytorch eager, pytorch `torch.compile`, current adaln Triton
(`triton_adaptive_layer_norm`, training fwd kernel). Correctness vs eager: **cos = 1.000000** (fp32),
0.99999 (bf16).

## TOKEN  d=768  (seq 384–1024)  — materialize wins

| seq  | M     | pt_eager | pt_compile | cur_triton | **ours (mat)** | vs compile | vs cur_triton |
|-----:|------:|---------:|-----------:|-----------:|---------------:|-----------:|--------------:|
| 384  | 12288 | 0.290    | 0.219      | 0.379      | **0.170**      | 1.28×      | 2.22×         |
| 512  | 16384 | 0.378    | 0.280      | 0.500      | **0.219**      | 1.28×      | 2.29×         |
| 640  | 20480 | 0.451    | 0.340      | 0.621      | **0.269**      | 1.27×      | 2.31×         |
| 768  | 24576 | 0.541    | 0.400      | 0.743      | **0.319**      | 1.25×      | 2.33×         |
| 896  | 28672 | 0.619    | 0.463      | 0.867      | **0.371**      | 1.25×      | 2.34×         |
| 1024 | 32768 | 0.700    | 0.520      | 0.982      | **0.422**      | 1.23×      | 2.33×         |

## ATOM  d=128  (seq 2048–8192)  — fused wins

| seq  | M      | pt_eager | pt_compile | cur_triton | **ours (fused)** | vs compile | vs cur_triton |
|-----:|-------:|---------:|-----------:|-----------:|-----------------:|-----------:|--------------:|
| 2048 | 65536  | 0.358    | 0.183      | 0.119      | **0.082**        | 2.24×      | 1.46×         |
| 4096 | 131072 | 0.687    | 0.343      | 0.221      | **0.149**        | 2.30×      | 1.48×         |
| 6144 | 196608 | 1.013    | 0.500      | 0.323      | **0.212**        | 2.36×      | 1.53×         |
| 8192 | 262144 | 1.336    | 0.656      | 0.419      | **0.272**        | 2.41×      | 1.54×         |

(times = median ms)

## Verdict

Best-of-both dispatch (`adaln_inference`, d≤256 → fused, else materialize) beats every baseline on
both paths. Token gain (1.23×) is GEMM-bound — remaining headroom is the `[scale|bias]` (M,2d)
HBM round-trip (~28% of runtime), which a fused cuBLAS-epilogue (cute) could reclaim. Atom gain
(up to 2.4× vs compiled, 1.5× vs current Triton) comes from dropping the backward saves.
