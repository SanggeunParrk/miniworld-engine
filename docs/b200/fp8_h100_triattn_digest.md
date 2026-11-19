# H100 TriAttn FP8 (engine 3.0.3): digest for the B200 port (agent D, 2026-11-19)

Source: `$W/b2511/fp8_ref` (docs/fp8.md "TriAttn forward: FP8 PV, per-model whitelist", "TriAttn: where else FP8 can go", "Parity extension: TriAttn d64 PV",
"TriAttn whitelist: ...", the RoseTTAFold3 / Boltz-2 sections; `fp8.py`; `h100_sm90.py`; `triattn_h16_sm90.cu`; tests `test_triattn_f8pv_h100.py`, `test_triattn_f8qk_h100.py`,
`test_triattn_d64_fp8.py`, `test_triattn_h16_fp8_extreme.py`). Everything is opt-in; with FP8 off the default route is bit-identical to the base.

## 1. What is quantised, scale granularity

| route (H100 name) | operand(s) in e4m3 | scale | where it is made | notes |
|---|---|---|---|---|
| FP8 PV, core `triattn_h16_sm90.cu -DTA_F8PV=1` (32- and 16-channel heads) | **P** (the softmax numerator), **V** (as V^T planes, K-major) | P: no scale, the byte IS p = 2^(x - m + 8) (largest probability 2^8 = 256); V: per head, static, power of two, 4x the calibration |V| absmax | P: in the softmax warps, from the fp16 logit, no `ex2` (byte = 8 x' + 1024 inside an fp16 mantissa, PRMT-packed); V^T: front kernel (`fused`, `front-made`) or a standalone `vt_quant` pass (`pv_pass`, `unfused`) | row sum l comes from the PV MMA itself (`[O | l] += P [V | 1]`, 8 ones rows, `m64n40k32` e4m3, fp32 accumulation): self-normalising, l is the sum of the QUANTISED P |
| FP8 PV hi + lo (`FP8_PV_HL`, `-DTA_F8HL`) | V as hi + lo/16 (two e4m3 planes) | same | pass / front | V error 2^-8 instead of 2^-4; module error about half; gates x 2; NOT the default |
| `pv_d64` (Chai-1, 64-channel heads) `triattn_d64_sm90.cu -DTE_F8PV=1` | P, V^T | static per-site V scale (first eager call); P `p = 2^(x + 8)`, `x <= 8.75` clamp | `vt_quant` (now 64 channels) | ones rows rewritten by the producer before each stage (the Q tile shares the K / V ring); gate `FP8_PV_D64_MIN_S = 1024` |
| FP8 QK (`FP8_QK`, `-DTA_F8QK`) | q, k (e4m3 rows) | per head power of two `sq`: q8 = q / sq, k8 = k * sq (product 1: bias and softmax scale untouched); refused if amax_q amax_k > 224^2 | the d_pair-128 front writes e4m3 q / k | experimental; in the `safe` whitelist only for af3 / protenix1 / openfold3 / rosettafold3 (fused d_pair-128 route only) |
| bidirectional / training | nothing | | | e5m2 backward no-go, e4m3 backward no speed gain; bias-only chain: scaled_mm P / V planes 131 vs 171 us but V planes cost more: no-go |

Call-site kinds (`FP8_PV_WHITELIST` = `fp8.TRIATTN_MODELS`): **fused** (d_pair 128 x 4 x 32: the front writes V^T), **front-made** (d_pair 64: 4x32, 2x32, 4x16 and 256x8x32 fronts write the planes),
**unfused** (standalone `vt_quant` pass: d_pair 384 x 12 x 32, template stacks, any shape without a one-kernel front), **d64** (64-channel heads).
Gates (below them the fp16 core runs, bit-identical to FP8 off): fused S >= 256; pass: 32-channel 1024, 16-channel 512; front-made d_pair 64 from L256, 256:8:32 from L768; d64 from 1024; all x 2 with hi + lo.

## 2. Saturation, NaN-byte, scale rules (the traps)
* **e4m3 NaN byte.** A byte formed by hand: 127 = NaN, 128+ = negative. The P conversion clamps x <= 8.75 (byte 8 x + 56 <= 126). `TA_CLAMPV` 3 (shipped): lower clamp = RELU of the first HFMA2, one HMNMX2 for the upper clamp.
  The 32-channel core had a latent edge (non-finite outputs at logits in the hundreds); fixed by `fp8c/h16-clamp`. The IntelliFold2 d64 attempt returned non-finite coordinates on 5 of 6 targets (the same edge, not verified).
  `cvt.rn.satfinite.e4m3x2` (the V path) is safe.
* **P offset / window.** p = 2^(x - m + 8) must stay below 448: the offset m rises as soon as a logit exceeds it by `TA_TAU` = 0.5 log2 units (exact tile-maximum tree, `vote`); O / l rescaled in registers. Lower window: x >= -7
  below the largest probability (byte 0): the tail below 2^-15 of the maximum is dropped (3 % worst case at 4096 keys).
* **V scale floors.** Static scale from the first eager call of a site (probe scale 1: the pre-cast peak is exact), headroom 4x (power of two); saturation sets a shared overflow flag; `take_overflow()` -> `rescale_all()` -> redo
  (`fp8.check()` once per forward outside any graph; the flag buffer and the scale tensors are updated in place, captured graphs follow). 3.0.3: pow2 scale floored (amax >= 1e-30, a scale of 2^-99 saturated later input and `1/scale = inf`).
* **Zero `ones` rows (d64).** The Q tile is carried through the K | V ring: the producer must rewrite the ones rows before each stage's V^T load (without it: err rms 3000 %).
* q / k scale cannot widen after saturation: `rescale_all` turns e4m3 q / k off for the site.
* An `fp8.py` guard: a TriAttn call from a site not reached through `_infer` has no site key and runs the dynamic scale.

## 3. Whitelist and why
| model | PV sites | q / k | evidence |
|---|---|---|---|
| af3 | fused + unfused | yes | FoldBench 27 targets, whole `safe` incl. q / k: 0 flagged (lDDT +0.0000); PV 4 targets x 4 trunk seeds |
| protenix1 | fused | yes | 9 + 3 targets x 4 seeds PV; 25 targets x 4 seeds q / k + PV |
| opendde | unfused | no | 16 + 3 targets x 4 seeds |
| protenix2 | unfused (256:8:32 from L768, 64:2:32 from L256) | no | 11 targets, 10 / 11 inside (7y37: bf16 bimodal) |
| openfold3 | fused + unfused | yes | 12 targets x 4 seeds, 11 / 12 inside each (7zau: bf16 spread 10.2 A) |
| rosettafold3 | fused | yes | 12 / 12 inside (PV, q / k, whole `safe`) |
| chai1 | d64 + unfused (template stack) | no | 12 / 12 inside with every gate removed |
| **Boltz-2** | NOT whitelisted | | 9c90 is bistable: relative noise of 0.5 / 1 / 2 % on the TriAttn output flips 2 / 1 / 4 of 4 draws; V on e4m3 flips it (hi + lo too); the bf16 seeds agree because that path is deterministic. Owner decision pending. KEEP the policy. |
| **IntelliFold2** | NOT whitelisted | | 10 / 12 inside, two large targets (8fxv 27 A, 7zau 22 A) flip; its trunk is 512:8:64 (d64 route: non-finite). Gain only 0.934 -> 0.918 of bf16 at n = 16 |
Acceptance (end level): same-trunk-seed RMSD of FP8 to bf16 <= max(1.1 x the largest bf16 trunk-seed pair RMSD, 0.5 A); a target whose bf16 seeds differ by > 5 A is a weak probe.
Module acceptance: judge form `max|y - y64| / max|y64|`; tolerance 8e-3: FP8 PV 4.8-7.8e-3 (d64), 0.43-0.66 % (d128); FP8 QK crosses it (0.85-0.95 %) but is whitelisted on end-level evidence. FP8 is outside the registry judge.

## 4. H100 numbers
Module time (CUDA graph, vs the 2.5.12 default, cool / sustained): d_pair 128 x 4 x 32 FP8 PV / default = 1.086 (L128), 0.934 (256), 0.909 (512), 0.878 (768), 0.865 (1024), 0.844 (1536), 0.845 (2048);
+ QK: 0.975 .. 0.929 of FP8 PV (0.804 at L1024 against the default, 0.795 at 2048). Core alone: 1024: 1.750 -> 1.449 ms (-17 %), 2048: 13.51 -> 10.87 (-20 %); 256:8:32 L1024 3.582 -> 3.122; 384:12:32 L1024 5.960 -> 4.811 (+ vt_quant 0.54).
Error (attention output, real activations): default 0.17-0.18 % rel L2, FP8 PV 2.07-2.40 %, PV + QK 3.12-3.96 % (max-norm 0.52 % / 6.2 % / 23.6 %). d64 core: err rms 4.2 % (random), logits to thousands stay finite.
Chai-1 d64 module (d_pair 256 x 4 x 64): 1.122 (L128), 1.042 (256), 1.059 (384), 1.047 (512), 1.009 (768), 0.986 (1024), 0.944 (1536), 0.930 (2048) of default: the V^T pass costs more than the core gains below ~1000 tokens.
5I28 (warm model s, FP8 / bf16): protenix2 n = 16 0.835, openfold3 0.777, rosettafold3 0.801 (n 8: 0.874), chai1 0.883 (n 16), af3 / opendde similar; at n <= 2 the gates keep the default core (0.97-1.00).

## 5. Traps (for the port)
* The H100's single trick that makes the core faster is NOT the e4m3 MMA (the QK^T removal ablation is worth 0-8 %): it is that P is formed WITHOUT `ex2` (byte arithmetic) and the row sum leaves the FMA pipe (the PV MMA with a ones tile). A port must keep both.
* P offset exact-tile-maximum + register rescale of O / l; on the B200 O / l live in TMEM (rescale = tcgen05.ld / st after the previous PV retired).
* V^T planes K-major (wgmma e4m3 needs K-major); on tcgen05 `kind::f8f6f4` the same (V is MN-major in the bf16 core: a plane pass is required).
* Static V scale + guard protocol; calibration keyed by the weight pack address; first eager call runs the pass.
* e4m3 q / k is judge-form out of tolerance; keep it behind its own switch.

## 6. Transfers unchanged vs redone for tcgen05
| transfers unchanged | redone for tcgen05 |
|---|---|
| p = 2^(x - m + 8) as e4m3 BYTES (8 x' + 1024 in an fp16 mantissa, relu / min clamp, PRMT), no `ex2` | the A operand: P goes to TMEM (4 e4m3 a 32-bit column, 8 columns a K = 32 MMA), not registers; PV = `tcgen05.mma.kind::f8f6f4` (ts) |
| row sum from the PV MMA with ones (`[V | 1]`), sum of quantised P | N = 48 (32 + 16 ones rows, M128 needs N % 16 = 0); O / l accumulators 48 TMEM columns a row (R = 2 rows a task: 96 columns) |
| V^T e4m3 planes, per-head static power-of-two scale (headroom 4), saturation flag, calibration on the first eager call, `take_overflow` / `rescale_all` / `fp8.check` | V^T tile: TMA box 128 B x 32 rows, 128 B swizzle, K-major; ones rows preset once per ring stage (B operand = 48 rows, SBO 1024) |
| exact tile maximum -> raise the offset by max + margin (TAU 0.5), window [-15, +0.5] | offset raise rescales O / l in TMEM (tcgen05.ld / st of 48 columns, after the previous PV of the row retired); the shift enters the byte conversion's constant |
| NaN-byte rule (clamp to 126 or prove x <= 8.75), no ex2 for the tail | softmax loop is B200's: logits S + bias (FHADD.BF16), packed fp16 conversion (the QS core's MUFU / S-return bound is gone: measured below) |
| model whitelist (af3 protenix1 opendde protenix2 openfold3 rosettafold3 chai1; NOT Boltz-2, IntelliFold2), policy, acceptance (end level) | gates re-measured against the B200 bf16 core (B200 default is faster than H100's: its S gates cannot be carried over) |
| QK e4m3 (own switch, not default) | not built unless measured: the QK^T MMA is 34 % busy and a QK^T removal buys ~2.4 % (docs/b200/triattn.md part 4) |
