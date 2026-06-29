# Transition forward benchmark (H100, bf16)

`triton old` = original: separate triton LayerNorm + expand kernel (h to HBM) + cuBLAS squeeze.  
`triton b2b` = new back-to-back fused (LN+expand+SwiGLU+squeeze, h never in HBM) for d≤128; expand+cuBLAS squeeze for d≥256.  
`cute` = fused LN+dual-GEMM+SwiGLU expand kernel + torch.matmul squeeze.

## A. Full forward vs seq_len, d=128, n=4

| seq_len | M=L² | pytorch | triton old | triton b2b | cute | b2b vs old | b2b vs pytorch |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 384 | 147456 | 0.9897 | 0.2519 | 0.1880 | 0.2494 | 1.34× | 5.26× |
| 512 | 262144 | 1.7115 | 0.4267 | 0.3153 | 0.4367 | 1.35× | 5.43× |
| 640 | 409600 | 2.6374 | 0.6522 | 0.4798 | 0.6655 | 1.36× | 5.50× |
| 768 | 589824 | 3.7640 | 0.9486 | 0.6831 | 0.9486 | 1.39× | 5.51× |
| 896 | 802816 | 5.1017 | 1.2665 | 0.9495 | 1.2488 | 1.33× | 5.37× |
| 1024 | 1048576 | 6.6664 | 1.6837 | 1.2436 | 1.6255 | 1.35× | 5.36× |

## B. Crossover vs d (= GEMM K), M=262144

| d | n·d | pytorch | triton old | triton b2b | cute | best |
|--:|--:|--:|--:|--:|--:|:--|
| 128 | 512 | 1.7099 | 0.4303 | 0.3243 | 0.4305 | **triton b2b** |
| 256 | 1024 | 3.0680 | 1.1082 | 1.3705 | 1.0091 | **cute** |
| 512 | 2048 | 6.6659 | 3.4722 | 5.9447 | 2.9175 | **cute** |

## Note: cute squeeze fusion (h fully off HBM)

A TRUE single-kernel cute b2b (squeeze fused, h never in HBM) is *mechanically* expressible
as an epilogue extension (split-K atomic squeeze) and is numerically correct (cos>0.9999),
but the per-element `atomicAdd` makes the gmem atomic count equal the whole squeeze FLOP
count → **~500-1000× slower** (117 ms vs 0.25 ms @ d=128). A FAST fused squeeze needs a 2nd
WGMMA (`sH @ Ws`) in the epilogue (reduce in regs/smem, one write per output) — a large fork
of quack's composable TileStore epilogue, no payoff at the model's d=128 (triton b2b already
wins). So: **d=128 → triton b2b; d≥256 → cute composed (expand + cuBLAS squeeze).**
