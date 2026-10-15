# H100 AttentionPairBias: kernel work, 2026-10-13

One optimisation round on `AttentionPairBias` (d_single 384, d_pair 128, 8 heads x 48, bf16, B = 1) on H100 (sm_90a, cluster `cssb`,
partition `h100`, one GPU per job), branch `h100/apb-opt` on `h100/cuda-sweep` d3a9e21d, torch 2.13 + cu129. Numbers: `bench.py
target=attention_pair_bias level=module precision=bf16-mixed cudagraph=manual +apb_n_head=8 d_single=384`, fp32 master parameters,
no key mask, ms. Per-kernel numbers: torch profiler kernel sums and Nsight Compute (`--set full`) of one eager step.

## Start

| L | training ours | Triton | note |
|---|---|---|---|
| 384 | 0.498 | 0.255 | fp32 pair-weight shadow active (qualified L 384 / 768 only) |
| 512 | 0.216 | 0.363 | no shadow at L 512 |
| 768 | 1.280 | 0.639 | shadow active |

Inference was already 1.9-2.5x the fastest other row (0.0225 / 0.0416 / 0.0902 ms at L 128 / 384 / 768).

## What was found

| Finding | Evidence |
|---|---|
| The natural-unit attention builds (`APB_NATURAL`) divided every score and `lse` by a float constant (`x / RSQDV`, `lse / LOG2E`): IEEE division, 64 per thread per key block. | L768 profiler: attn_fwd 59 / attn_dq 62 / attn_dkv 84 us. After multiplying by the reciprocal (ncu): 15.9 / 20.1 / 18.5 us (ncu base clock), 12.3 / 14.5 / 13.4 us (profiler). |
| The fp32 pair-weight "shadow" (born as a repair of the LN-beta gradient, which no longer exists; narrowed to dWb) is 60-70 % of the step at L384 / 768. Its only effect is dWb: 1.6-1.7e-2 -> 0.8e-2 (master) / 1.9e-2 -> 1.45e-2 (bf16 params) against fp64; every other gradient of every bf16 implementation, the shadow path included, is at 1.7-1.9e-2 (cosine to the fp32 module 0.9998-0.9999). | `acc.py`: ours / bf16 PyTorch / Triton / shadow vs fp64, L 384 / 768, mask on / off, master and bf16 parameters. |
| `acc_reduce_k` summed 264 block partials sequentially per column: 12 us (+ 2.7 us zero fill). | profiler, L768. |
| Pair kernels: 87-93 % of the measured copy bandwidth; stage depth 2 / 3 / 4 identical within 1 us. | `pb.py`: pair_bias 58.6 us (2.74 TB/s), pair_bias_bwd 124.8 us in isolation (2.57 TB/s), 302 MB copy 102.5 us (2.95 TB/s). |

## Changes

| Change | Effect |
|---|---|
| reciprocal instead of division in `attn_fwd.cu`, `attn_dq.cu`, `attn_dkv.cu` | attention cores 205 -> 42 us at L768 |
| `settings.apb_h100_exact_dwb` (default off); the CUTLASS-dependent oracle builds only when it is on | training 1.28 -> ~0.29 ms at L768; default build needs no `CUTLASS_PATH` |
| `acc_reduce_k` 16-warp fixed-order tree over the rows each grid wrote; no `zero_acc` | 12 + 2.7 -> 3 us |
| `prep_k` reads fp32 / bf16 weights and packs Wq|k|v|g, Wo, Wf and Wb in one pass; `cast_leaves` no longer launched | -5.7 us |
| `attn_fwd` MODE 3: O fp32 and og = sigmoid(g) O bf16 from the epilogue; `gate_rows` not launched | -2.7 us |
| key splits (`APB_KS`, 2 warpgroups per tile, merged in smem) | 1-2 us per core; default 1 |

## Results (ms, CUDA graph)

| L | inf ours | inf Triton | train ours | train Triton | PyTorch compiled (train) | cuEquivariance (train) |
|---|---|---|---|---|---|---|
| 128 | 0.0225 | 0.0431 | 0.0661 | 0.1116 | 0.1364 | 0.1251 |
| 256 | 0.0300 | 0.0580 | 0.0865 | 0.1629 | 0.2071 | 0.1785 |
| 384 | 0.0419 | 0.0886 | 0.1226 | 0.2546 | 0.3089 | 0.2643 |
| 512 | 0.0558 | 0.1279 | 0.1656 | 0.3648 | 0.4334 | 0.3688 |
| 640 | 0.0722 | 0.1716 | 0.2153 | 0.4924 | 0.5887 | 0.5190 |
| 768 | 0.0902 | 0.2238 | 0.2716 | 0.6431 | 0.7379 | 0.6528 |

Tests: `tests/gpu/sm90/test_h100_apb_training_gpu.py` (accuracy vs the fp32 module within 1.3x of the bf16 PyTorch module for the
output and every gradient, five layouts, L 128 / 384, mask on / off, bf16 and fp32-master parameters; bitwise repeatability; live
CUDA-graph replay; `torch.compile`; the opt-in shadow) and `tests/gpu/sm100/test_b200_apb_gpu.py -k apb_h100` (inference): 105 passed.
Registry: `miniworld-engine dev derive --arch sm86` (the default-arch derive crashes in the TriAttn CUTLASS TMA path on H100,
also on an unmodified HEAD).

## Not done

- Overlap of `pair_bias_bwd` with the dK / dV -> qkv_bwd -> GEMM -> ln_bwd chain on a second stream (estimated 10 %).
- bf16 dbias between `attn_dq` and `pair_bias_bwd` (3 % of the pair bytes).
- fp32 activations, QK-norm, B > 1 (PyTorch path).
- SoL: pair kernels 87-93 % of copy bandwidth; step ~55 % (L768) / ~30 % (L384) of the HBM time roofline; not 90 %.
