# H100 MSA-family FP8 design digest: OuterProductMean, PWA (8 x 32, wide, BIG), Chai-1 grouped OPM (engine 3.0.3; agent G, 2026-11-19)

Sources read: docs/fp8.md (OPM forward, PWA forward, "The wide widths", "The BIG build", "Chai-1's option block / Parity extension: grouped OPM"), `opm_fp8.py`, `pwa_fp8.py`, release notes 3.0.x.

## 1. What is e4m3 and what is not
| module | e4m3 | stays bf16 / fp32 |
|---|---|---|
| OPM (groups 1, d_hidden 32) | the grouped product O = A2 . BT^T only: both operands A2 [(i,c), s], BT [(j,e), s] (fp32 accumulation, bf16 O) | LN, projections (fp32 acc, one cast at the drain), count table (exact), O (bf16), epilogue W_o, bias, residual. "O as e4m3" was NOT done (third rounding, <= 2 % at depth 16384, 15-25 % of the FP8 module at S 1024-2048); "fused kernel on e4m3 wgmma" NOT done ("a kernel rewrite") |
| PWA 8 x 32 / wide / BIG | the key contraction o[h] = w[h] . v[h] (K = j = N): w (a probability, scale 2^8 fixed) and v (static pow2 scale) | LN_m, projections (v is the e4m3 OUTPUT of `ln_vt8`, no extra cast), gate pass `gate_out_w` / `gate_w` + `out_w` (o stays bf16; "o as e4m3" tried on 8 x 32: no gain, the gate pass is latency bound), softmax pair side |
| Chai-1 grouped OPM (8 x 8, product LN eps 0.1) | the 8 per-group products (cast + transpose pass `chai_f8.cu`, 8 `_scaled_mm`) | product LayerNorm epilogue |

## 2. Scales
* OPM: policy `bound` (recommended): static power-of-two scales from the WEIGHTS alone, |LN(x)_k| <= sqrt(d_msa - 1)|gamma_k| + |beta_k| (+ |projection bias|), scale = next pow2 of max_c sum_k |W_ck| (...) x (1 + 2^-6): the cast cannot saturate, no calibration, no guard (chosen scales 2^-2 / 2^-1 / 2^0 at d_msa 64 / 128 / 256, scaled amax 21-26 / 11 / 5-6 of 448). `calib` (16x headroom over the first eager call) exists. Scales are cuBLASLt scale POINTERS (device scalars: a captured graph follows an in-place rescale).
* PWA: w fixed 2^8; v static pow2 from the first eager call's amax x HEADROOM 16 (key: the value weight, weakref-dropped), `take_overflow` / `rescale_all` / `GUARD sync|lazy`, `ln_vt8` records max |v / scale| (peak 23.5-23.8 of 448 on the BIG build).
* floors: all pow2 scales floored at 2^-30 (the OpenDDE trap: clamp_min(1e-30) -> 2^-99 scale saturates later input).

## 3. Gates and why (H100)
* OPM `_gate`: padded S >= 2048 at every L, S >= 1024 at padded L >= 256; below: bf16 (L128 S1024 +5 %, S512 +0..34 %). The FP8 composition REPLACES the fused bf16 kernel from S 1024 / L 256.
* PWA 8 x 32: N >= 384 and S >= 1024 below N 512 (N <= 256 loses at every depth: the fused `pwa_fwd3` beats split + FP8; N384 gains only from S1024).
* PWA wide (8 x 8 / 8 x 16): `WIDE_MIN_N` 256, `WIDE_MIN_SN` 3 x 2^16; N only a multiple of 16.
* PWA BIG (256 x 8 x 32, IntelliFold2): N % 64 == 0 (needs y from `ln_vt8` for `gate_w`; the N400 row loses 1.031), S x N >= 2^14, no N floor (every measured shape from S64 N256 gains).
* grouped OPM: gate closed ("F8_MIN_S never"): net loss at every S.

## 4. Accuracy (H100, fp64 gold, random weights)
OPM judge max|y - y64|/max|y64| 0.6-1.4 % at depth 1024-4096 (bf16 0.3-0.6 %), inside 8e-3 only at depth >= 8192; PWA judge 0.30-0.37 % (bf16 0.28 %); BIG update rms 3.5-3.6 % vs 2.55-3.1 % (judge 0.31-0.35 %); grouped OPM 8x the default's error (rms 3.6e-2 vs 4.6e-3 at S 1024: the product LayerNorm eps 0.1 does not hide the operand rounding). FP8 stays outside the judge; acceptance = fp64 error band + no saturation + end level (RMSD inside the bf16 trunk-seed spread, 9/10 targets IntelliFold2 BIG).

## 5. H100 speedups (graph, cool / sustained, FP8 / default)
OPM d_msa 64: S 16384 0.58-0.66, S 4096 L384 0.62, S 2048 L384 0.69, S 1024 L384 0.85; wide widths 0.58-0.86; PWA 8 x 32: S1024 N768 0.795, N2048 0.714; wide AF3 64 x 8: S1024 N768 0.865, N2048 0.794 (S 4096 N2048 0.74); BIG: 0.84-0.89 (0.96 at S64); stage shares H100: ln_vg 21 %, contraction 45 %, gate pass 30 % (8 x 32); wide 64 x 8 S1024 N768: contraction 30 %, gate 31 %, e4m3 GEMM only 0.65x (o write bytes).

## 6. Traps recorded
* Hopper e4m3 wgmma takes K-major operands only: v must be written j-contiguous (`ln_vt8` = ln_vg with the projection turned around). (B200: MN-major B is legal for cuBLASLt and tcgen05: no transposed v.)
* the grouped product's scale pointers must be separately allocated (a slice of a [2] tensor made cuBLASLt answer NOT_SUPPORTED).
* cuBLASLt fast accumulation: product 4-7 % faster but max error 3.5-4.5 % at K 16384: not recommended.
* the e4m3 byte 127 is NaN: satfinite cast or clamp at 126.
* `ln_vt8` for N % 16 reads the next row's tokens in the last block (clipped on store, excluded from amax).

## 7. Transfers unchanged vs redo for tcgen05
| transfers unchanged | redo for tcgen05 (B200) |
|---|---|
| e4m3 only on the product / contraction operands; `bound` OPM scales, w 2^8, v static pow2 + 16x headroom; scale floor 2^-30; guard contract (`take_overflow`, `rescale_all`, sync / lazy); route-stats keys | v / A2 / BT e4m3 written by the TMEM DRAIN (satfinite cvt, 1-byte rows, u8 TMA maps, 32 B / 64 B swizzle) instead of wgmma register tiles |
| gate SHAPES as starting values (OPM S >= 1024, PWA N^2 S work) | contraction = cuBLASLt e4m3 (MN-major B accepted) or a tcgen05 `kind::f8f6f4` kernel; scalar dequantisation scales; no `ln_vt8` transposed v |
| acceptance protocol (fp64 gold, large-weight stress x8 / x4, saturation through captured graphs) | gates RE-MEASURED against the B200 chains (cuBLAS bf16 1.3-1.66 PF/s under the cap: contraction share, gate pass, wide prologue shares differ) |
| BIG: N % 64 keeps y for `gate_w`; grouped OPM stays closed unless the cast is free | the B200 BIG build is the (256, 8 x 32) WS build of `pwa_wide_sm100.cu` (ring-streamed weights): the e4m3 v drain is the same code as the other builds |
