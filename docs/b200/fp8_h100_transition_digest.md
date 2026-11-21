# H100 (sm90a) FP8 Transition digest (engine 3.0.3 docs/fp8.md "Transition forward", "D 256", "token counts", "D 64"; sources fp8_sm90a.py, transition_fp8_fwd_sm90a.cu, transition_fp8_d256_sm90a.cu/.cuh, wide/fp8_bind.cu)

## What is e4m3 per width (inference forward, bf16 x [..,D], n 2/4; off by default, MINIWORLD_TRANSITION_FP8=calib; training, D64, D768, fp32 activations stay default)
| width | kernel (H100) | e4m3 operands | h path | notes |
|---|---|---|---|---|
| D128 | the 2.5.12 fused D128 inference kernel (LNFRAG/XFIRST/DECOUPLE/SKEW), both GEMMs `wgmma m64n128k32 .e4m3` | xn (A, quantised in registers from the fp32 LayerNorm, no bf16 rounding first), [Wa;Wb] and Ws (B) | the fp32 C fragment of the expand IS the k32 e4m3 A fragment of the squeeze after a fixed K permutation (cols {2t,2t+1,8+2t,9+2t} of each 16-block) -> weights packed in that K order, no shuffle; h never leaves registers | 24 KB weight slots, 4 in the ring; F8G1/F8G2 give partial placements |
| D256 | the 2.5.12 D256 fwd (producer WG + 2 ping-pong consumer WGs, 64x256 fp32 out accumulator held across 32-unit hidden chunks) | same as D128 | chunk [a|b] fragment = one 32-wide h block = one k32 A fragment; expand 8x m64n64k32 SS, squeeze ONE m64n256k32 RS | xn8 in its own 32 KB buffer, x stays resident for the residual (the 2.5.12 second x load is gone) |
| D384 / D512 | wide 2-kernel chain: wide_lnsg_f8 (LN + expand + SwiGLU) + squeeze_f8 (+residual) | xn8 written in place over the resident x tile (128-B K slabs), h e4m3 (stmatrix, TMA store), Ws8 | h through HBM as e4m3 (half the bytes) | G1=0 = bf16 expand + e4m3 h/squeeze (the "h8" variant) |
Not on H100: D64, D768 (stays default), training.

## Scales (all static, power of two, no per-row scale)
* xn: sx = pow2 floor(448 / (sqrt(D-1) max|gamma| + max|beta|)): the LayerNorm bound, cannot saturate, no calibration.
* weights: per-tensor pow2 (Wa and Wb share one scale s1, Ws s2), packed once per parameter version (no per-call prepare kernel), cache keyed by wa data_ptr, weakref-finalised.
* h: sh calibrated on the FIRST EAGER call per module: a probe call with sh = 1 (pre-cast peak exact even when it would saturate), sh = 2^floor(log2(448 / (amax HEADROOM))), HEADROOM 16. Peak max|h sh| is observed in the epilogue and atomicMax-ed into a guard buffer.
* guard: `sync` (eager: redo an overflowing call with a lower sh, <= 4 retries, then default route), `lazy` (no host sync; CUDA graphs: overflow_peak() + rescale_all(peak) + redo = the TriMul contract), `off`. A capture with no calibration takes the default route.
* silu(a) = (a/2)(1 + tanh(a/2)): ONE MUFU op; epilogue h = silu(a ca) (b cb) with ca = 1/(sx s1), cb = sh/(sx s1); cast cvt.rn.satfinite.e4m3x2; squeeze out = x + acc/(sh s2).

## Gates and why (ROW_GATE, MINROWS env)
* D128 16384 rows (= pair L128); D256 8192 rows (fused kernel has <= 48 tiles for 132 SMs below that; below 8192 default runs generic kernels); D384/512 65536 rows (= L256; the wide chain is 1.5-3.4x slower at token counts 256-4096: h via HBM, 2-32 CTAs, latency bound; cuBLAS e4m3-vs-bf16 bound saves only 0-17 % of the module at token counts).
* Everything off unless POLICY set; with it unset the dispatcher sees one `POLICY is None` check, default bytes unchanged (sha1, separate processes, 14/14, 18/18).

## H100 speedups (module time, graph, cool / sustained, vs 2.5.12 default)
D128 n4 full 0.66-0.75 (L128..2048; 0.69-0.75 sustained), n2 0.68-0.76; G1-only 0.74-0.87, G2-only 0.77-0.90, bf16 twin 1.03-1.12 (n4)/0.98-1.04 (n2): the e4m3 gain over the same structure is 1.4-1.5x.
D256 n4 0.69-0.73, n2 0.74-0.76 (h8 0.91-0.93). D384 n4 full 0.64-0.69 (h8 0.85-0.90), n2 0.71-0.74; D512 n4 0.56-0.62 (h8 0.80-0.85), n2 0.63-0.66.
Kernel counters (D128 n4 L1024): default tensor 67 %, MUFU 45 %, DRAM 19 %; e4m3 kernel tensor 43 %, MUFU 29 %, FMA 28 %, no unit saturated -> the per-warpgroup chain expand -> SwiGLU -> squeeze is the limit; ablation: no SwiGLU math 0.398 ms, full 0.478 ms, HBM floor 0.195 ms (default 0.682).

## Accuracy numbers (judge = max|y - y64|/max|y64|; tolerance 8e-3; rms; update rms = rms(y-y64)/rms(y64-x))
PyTorch bf16 0.0044-0.0068 | 2.5.12 default 0.0035-0.0042 (rms 0.0027, update 0.0052) | D128 full e4m3 0.041-0.045 (rms 0.034, update 0.066) | G1-only 0.033-0.045 | G2-only 0.026-0.037 (rms 0.0195, update 0.038) | D256 full 0.035-0.040 | D384/512 full 0.035-0.042 (h8 0.020-0.029).
= about 10x the default error; ALL FP8 variants are outside the 8e-3 tolerance (3-5x) by construction (3 mantissa bits on the two activation inputs): documented, opt-in. Peak |h sh| 15-28 of 448 (headroom 16).

## Traps
1. Scale floors: H100 amax floor 1e-30 (B200 uses 1e-12: a product of scales must stay finite in fp32).
2. The bf16 twin built with the tanh sigmoid alone is SLOWER (1.03-1.12) than the default: tanh only pays with e4m3 GEMMs (SIGT=0 ex2+rcp 0.86-0.94 vs tanh 0.68-0.80).
3. cuBLASLt e4m3 vs bf16: squeeze 1.46-1.68x, expand 1.37-1.50x at D384/512 but only 1.06-1.14x at D64-256 (bf16 output bound) -> the epilogue/HBM, not the MMA, bounds narrow widths.
4. h row-permutation (K order) must be mirrored in the packed weights; a graph replay needs a calibrated module (eager first), the scale vector is a tensor refreshed in place.
5. Token-count single transitions (rows 256-4096) lose on e4m3 on every H100 chain (x1.5-3.4): gates matter.

## Transfers unchanged vs redo for tcgen05
| item | status |
|---|---|
| scales (pow2 LN bound, per-tensor weights, static sh, HEADROOM 16, probe calibration), guard sync/lazy/off, rescale_all, caches, env switches, packing per parameter version, route-stats | transfers unchanged (already in fp8_sm100a.py) |
| tanh silu + satfinite e4m3x2 epilogue, fp32 LN quantisation (no bf16 round) | transfers (device code) |
| register fragment reuse (expand C fragment = squeeze A fragment, K permutation) | REDO: TMEM accumulators; h goes TMEM->regs->e4m3->TMEM(A operand of a TS MMA) and the K order is the natural one (no permutation) |
| wgmma m64nNk32 SS/RS, 64B/128B swizzle slabs, stmatrix h staging | REDO: tcgen05 kind::f8f6f4, 128-B swizzle K=128 B per block, desc_k128 +32 B per k32 step, M=256 2-CTA pairs |
| row gates (16384 / 8192 / 65536) | RE-MEASURE on B200 (tile = 128 rows x 148 SMs, 2-CTA pairs) |
| expected gains (0.66-0.75 D128..256) | NOT transferable: B200 bf16 default is already a 2-CTA TMEM-resident fused kernel (D128 n4 L768 215.6 us); e4m3 halves tensor time but the fixed epilogue/LN/HBM/SwiGLU MUFU chain stays |
