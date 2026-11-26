# H100 TriMul FP8 design digest (engine 3.0.3; agent B, 2026-11-19)

Sources read: `_h100_fp8.py`, `h100_b200style_fp8.py`, docs/fp8.md (TriMul sections, RF3/IF2/D512, "where else e4m3 can go", bidirectional, history), tests/gpu/test_trimul_fp8_{planes,bidir,widths,k1,rf3,d512}.py.

## 1. What is e4m3 (and what is not)
- ONLY the contraction operands: the K1-output planes a and b (plane value = sigmoid(g) * p, bf16 on the default route).  Everything else stays bf16/fp32:
  LN_in, the four projections (K1 compute), the contraction accumulator (fp32), t (bf16 output of the e4m3 GEMM), LN_out, K3 projection and gate, residual.
- Placement study (fp64 emulation, `emul.py`, update rms error / bf16 floor 0.65 %): planes 3.72 %; + K1 gate projections 4.25 %; + all K1 projections 6.8 % (the
  experimental `FOLDJET_ENGINE_TRIMUL_FP8_K1=1`, never default); + t as K3 operand 4.5 % (built, no gain, removed); + K3 projection 5.3 %; + K3 gate 4.0 %.
  => the planes are the one placement that pays; value paths (projections, t, W_o) cost 0.8-3 % each, gate paths (sigmoid slope) only 0.3-0.5 %.
- "t as e4m3" implemented and REMOVED: contraction is tensor-bound (no faster), K3 saves <= 3.6-6.3 % bytes, measured slower in 15 of 16 cells.

## 2. Scale granularity and where scales come from
- Power-of-two per-TENSOR scales: one `sa` for the a planes and one `sb` for the b planes (per direction in the bidirectional block).  Per-channel mode exists
  (`FOLDJET_ENGINE_TRIMUL_FP8_SCALE=channel`, K3_CSCALE=1) but pow2 rounding leaves the e4m3 error unchanged, so channel-uniform costs nothing in accuracy (default `tensor`).
- `t = sa*sb * sum_k qa qb` exactly (pow2); the factor `csc = sa*sb` is folded on the H100 into K3: the projection operand `wpf = wpf0 * csc` (host, per call/refresh) and the
  LN_out statistics (mean*cs, var*cs^2).  K1 applies the scale through PRE-SCALED weights (W_p * 1/s; `qinv`), so the gate epilogue value is already the scaled plane value and is only
  cast (`cvt.rn.satfinite.e4m3x2`), amax of the scaled value goes to the guard buffer.
- Calibration ("calib" policy): FIRST EAGER CALL of a module = exact channel amax of the bf16 planes (bf16 K1 pre-pass, numerics only), `pow2_scale(amax)` = smallest 2^e with
  amax/2^e <= 448, floor 2^-30 (all-zero channel -> 1; the clamp_min(1e-30) -> 2^-99 scale saturated later input = OpenDDE trap), then HEADROOM 16x (scales * 16) for later inputs.
  Static from then on, cached per module (key = data_ptr of to_left.weight, `weakref.finalize` drops it when the weight is freed; a stale entry gave 137 % error / NaN).
  Bidirectional: `cache["bd"][1|2]` one state per direction.  `dynamic` policy = exact amax every call (numerics only).
- Peak scaled plane value observed: 17-23 of 448 at 16x headroom (random weights, N(0,1) inputs).

## 3. Saturation protocol (guard)
- `GUARD`: `sync` (default, eager: the call reads `LAST_AMAX`, if > 448 raises scales by `2^ceil(log2(peak/448*2))`, `refresh()`, redoes the call, <= 4 tries, then bf16 fallback `_default`),
  `lazy` (no host sync; REQUIRED under capture: kernel atomically maxes the scaled peak into a per-device buffer `_lazy_buffer`), `off`.
- `overflow_peak(reset=True)` (one host sync: read once per forward outside graphs) and `rescale_all(peak)` (all calibrated modules: sa,sb *= f; `refresh` rebuilds qinv, csc, prescaled weights
  and the folded K3 operand IN PLACE so captured graphs follow) -> caller redoes the forward (<= 3 times).  `fp8.check()` loops these for (tm, opm, tr).
- A capture with NO calibration takes the bf16 route (`capture_fallbacks`); eager warm-ups calibrate.  Unguarded saturation = 35 % update error (guarded 3.8 %).
- e4m3 NaN byte: every cast is `cvt.rn.satfinite`; hand-formed bytes must clamp to 126 (= 448).  No NaN/inf/saturation in any measured case.

## 4. Gates (why)
- `serves`: POLICY on; pair 4-D, square, 1..16 samples (B>1 = the one-sample route per sample, shared scales; the batched default is one launch, so small-L B>1 gains less), D in {64,128,256,384,512},
  `module.d_hidden == D` (hidden = pair), bf16, L >= 16, LayerNorm eps 1e-5 both, mask None or bool [B*L].
- D512 (IntelliFold2 trunk): one direction only, B = 1, L % 128 == 0, L <= 2048 (MAX_L_512; what was measured), env `FOLDJET_ENGINE_TRIMUL_FP8_D512=0` off.  Bidirectional D512 not built.
- `MIN_L[(D, dir)]` (dir 0 bidir, 1 out, 2 in), measured on H100 graph replay vs a same-cubin control: (128,1) 144, (128,2) 192, (256,1) 224, (256,2) 144, (128,0) 256, (256,0) 1024,
  (64,1|2) 512, (64,0) 2048, (384,1|2) 128, (384,0) 1024, (512,*) 0.  Reason: below the gate the e4m3 route's extra kernel (cuBLASLt e4m3 + 0.8 us workspace memset, second K1 launch in the
  bidirectional block recomputing LN_in) costs as much as the tensor-rate gain; D64 is byte-bound.  Env `FOLDJET_ENGINE_TRIMUL_FP8_MINL=<n>` replaces every gate (0 = none).
- RF3 (`TriangleMultiplicationRF3`): `_LeftScaled` hands a per-call product W_a*(1/n) (new tensor every call -> never cached, captures fell back).  Fix = `module_route` takes the PARAMETER
  `left_base_weight` for calibration/cache key and the factor `left_length_factor` as device scalar `lfac`; `lfac` multiplies the K3 channel scale `csc` at run time (K3 reads csc for the LN_out
  statistics) and rebuilds `wpf = wpf0*csc`; planes/scales/prescaled weights are the plain module's: ONE calibration whatever the length factor.  bf16 fallback multiplies `to_left.weight` by lfac.
- Env switches: `FOLDJET_ENGINE_TRIMUL_FP8={calib|dynamic}` (via `fp8.env`; preset `safe` = calib), `_GUARD={sync|lazy|off}`, `_HEADROOM`=16, `_SCALE={tensor|channel}`, `_MINL`, `_D512`, `_FASTACC`, `_K1`, `_RECORD`.
- Training: not served (inference only; no FP8 backward; e5m2 no-go).

## 5. Measured on H100 (graph replay, FP8/default; cool/sustained)
- D128 out: L256 0.91, L512 0.85, L1024 0.82, L2048 0.75; D128 in similar; D256 out: 0.96 / 0.93 / 0.95 / 0.84; bidir D128 0.95 (L256) .. 0.72 (L2048), bidir D256 1.01-1.04 up to L768, 0.975 L1024, 0.82 L2048.
- D64: +13.5 % at L128, ~0.98 L256, 0.90 L512, 0.86 L1024, 0.77 L2048; D384: 0.93-0.96 L128-L256, 0.89 L1024, 0.82 L2048; bidir D64 0.78 only at L2048 (+8 % sustained at L1024); bidir D384 0.92 L1024.
- D512 (packed K1 with e4m3 store, cuBLASLt e4m3): 0.76 L128, 0.87 L256, 0.89-0.94 L384-1024, 0.84-0.89 L2048.  E4m3 contraction GEMM = 1.55x of bf16 (not 2x) at D512 L1024.
- Stage view D128 L2048: K1 1.38 -> 1.08 ms (byte-bound: planes halve), contraction 2.83 -> 1.68, K3 unchanged: module -26 %.  The contraction reaches 85-87 % of the e4m3 tensor floor.
  FASTACC (cuBLASLt fast accumulation) +7-8 % of the contraction at L >= 1024 (opt-in).
- Accuracy vs fp64 gold (random weights, 3 masked tokens): FP8 update rms 3.6-3.8 % (max 3.1-5.2 %), engine form max|y-y64|/max|y64| 2.2-3.5 %; bf16 default 0.63-0.66 % (0.4-0.6 % engine form).
  FP8 fails the engine's 8e-3 module form by design (outside the judge); acceptance in tests = update error < 6 % (bidir stress < 8 %) and fp16/bf16 reference < 2 %.

## 6. Rejected / traps
- e4m3 t, e4m3 K3 operands, K3_MC (2-CTA multicast), K1 TMN_K1_WTMA / PIPE, direct n64 F8 K1 at D512 (1.37x slower than the packed K1): removed, no gain.
- FP8-compute K1 (all projections e4m3): 6.9 % error, failed the Protenix pLDDT criterion; a gate-projection-only K1 is the untested next lever (emulated 4.2 %).
- Incoming direction: e4m3 GEMM needs K-major operands -> K1 reads the pair TRANSPOSED and stores LN stats at the swapped token (`stats_t`) so planes come out K-major.  (eager host transpose cost +17 us: gone.)
- Eager measurements below L768 are host-bound for both routes: use graph replay.  Per-call zero-alloc of amax (`setdefault(.., torch.zeros(1))`) cost host time: allocate once.
- Calibration cache keyed by address of a tensor: stale state after free / reuse (fixed with weakref.finalize); `weights_changed()` does not invalidate (inference only).
- Sustained vs cool: some cells gain cool and lose sustained (bidir D64 L1024 -7.8 % / +5.3 %): H100 power cap; on B200 judge by sustained energy.
- cuBLASLt e4m3 heuristics and `cvt.satfinite` depend on the CUDA 12 toolchain.  `test_prebuild_plan::test_plan_is_reentrant` order dependence (not this change).

## 7. Transfers unchanged to B200 vs must be redone for tcgen05
| transfers unchanged | must be redone for tcgen05 (B200) |
|---|---|
| e4m3 on the planes only (a, b); t, LN, projections, K3 stay bf16/fp32 | K1 epilogue: the plane store is `stmatrix.trans` of bf16 pairs into [64 ch][32/64 tok] boxes + TMA; e4m3 needs a byte path (satfinite cvt, lane exchange, 1-byte rows) |
| power-of-two per-tensor scales (a, b), per direction when bidirectional; `t = sa sb sum qa qb` exact | contraction: `tcgen05.mma kind::f8f6f4` e4m3 (K = 32 per MMA, 128 B swizzle = 128 e4m3 per row, idesc), fp32 TMEM accumulator, MN-major (incoming) operands are legal for 8-bit |
| calibration on the first eager call from the exact bf16 amax, `pow2_scale` (floor 2^-30), HEADROOM 16x, static afterwards | scale application: apply `sa sb` in the contraction EPILOGUE (TMEM -> fp32 * csc -> bf16): no K3 change at all (the H100 folds it into K3's weights because cuBLASLt returns raw) |
| guard: amax of the scaled value to a per-device buffer, `overflow_peak`/`rescale_all`/`refresh` in place, sync/lazy/off, capture-without-calibration -> bf16 route, `weakref.finalize` cache | `qinv` reaches the epilogue as a DEVICE scalar array (read at run time) so `rescale_all` updates it in place under graphs (H100 pre-scales weights instead) |
| gates `MIN_L` as starting values (re-measure: the B200 bf16 chain is k1w -> ctr/cuBLAS -> k3w, sustained power-cap) | TMA maps (uint8 planes), pipelining/ring, TMEM double buffer, PDL chaining |
| RF3: calibrate on the parameter (`left_base_weight`), `lfac` runtime factor outside the planes' scales | incoming direction: no transposed K1 read needed if the e4m3 MMA takes MN-major operands (ctr already does for bf16) |
| accuracy acceptance: update rms < 6 % (bidir < 8 %) vs fp64, no NaN/sat, route-ran check (differs from bf16), graph == eager, default bit-identical (sha1, separate procs); + stress x{1e-3,1,40} x offset{0,25} | amax-to-guard from a TMA-store epilogue (per-CTA max + one atomic) |
