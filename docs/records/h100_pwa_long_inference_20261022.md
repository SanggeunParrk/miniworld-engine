# PWA H100 inference above L=1024 (2026-10-22)

MiniWorld-shape MSA pair-weighted averaging (d_msa 64, d_pair 128, 8 x 32, S 1024, bf16), CUDA graph, inference, H100.

Cause: `pwa_train.refusal` refused every N > 1024 ("the CUDA-only H100 path needs positive S and N <= 1024"), so the module fell through to the
PyTorch statements (3 direct_copy kernels 2.0 ms, un-fused sigmoid/mul/add 1.3 ms, cuBLAS GEMMs, 5.9 ms at L=1280). The cap was `pair_fwd3`'s
`N <= 1024` check and the dynamic-shared-memory attribute sized for N = 1028; nothing else in the inference path depends on N <= 1024.

Fix: `pair3.cu` pair_fwd3 accepts N <= 2048 (attribute sized for N = 2052: 64 KiB tiles + H*(N+4)*4 B logits, 132 KiB), `pwa_train.NMAX_INFER = 2048`
applies when autograd is off; training keeps N <= 1024 (pair_bwd3 is unchanged).

Numerics: update (out - msa) vs the PYTORCH implementation, same bf16 weights, 10 % masked keys: relative error 0.016 (L 256), 0.023 (1024), 0.024 (1280),
0.027 (2048), cos >= 0.99962, the same level and trend as the kernels already shipped at L <= 1024.

| L | before ours ms | after ours ms | Anthropic ms |
|---|---|---|---|
| 256 | 0.240 | 0.242 | 0.584 |
| 512 | 0.628 | 0.630 | 1.550 |
| 768 | 1.046 | 1.052 | 2.907 |
| 1024 | 1.549 | 1.596 | 4.731 |
| 1280 | 5.966 | 2.261 | 6.944 |
| 1536 | 7.583 | 3.024 | 9.601 |
| 1792 | n/a | 3.822 | 12.884 |
| 2048 | n/a | 4.938 | 16.528 |
