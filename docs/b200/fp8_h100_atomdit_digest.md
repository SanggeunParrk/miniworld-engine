# H100 (engine 3.0.3) FP8 digest for the atom DiT rows route and the token DiT placements (agent E, n14-e, 2026-11-19)

Sources read: `docs/fp8.md` (atom DiT sections L2871-3000 "Where FP8 can go / What is built / Module time / Error against fp64", ESMFold2 atom block L2984-3040,
RoseTTAFold3 KQLN L3042-3135, Boltz-2 up gate L3199-3330, token DiT L749-870 and L2611-2790), `sm90_atom/{f8.py, f8w90.cuh, pre90_f8.cu, post90_f8.cu, post90_f8up.cu}`,
`conditioned_transition/cuda/token_dit_fp8.py` (+ `token_dit_fp8_rows.cu`), tests `tests/gpu/test_atom_dit_fp8_rows.py`.

## 1. Atom DiT "rows" route (H100)

| question | H100 answer |
|---|---|
| what is e4m3 | the A operand of every row-kernel product (x1 / xkv for q, gate, k, v; gated o for Wo; x2 for [Wa; Wb]; h for the squeeze Wd) and the B operand (weights). NOT: attention (32-wide heads, ablation <= 1.5 %), conditioning tables, pair bias (hoisted per fold), residual stream, SwiGLU nonlinearity, outputs (bf16). |
| activation scale | DYNAMIC per row, power of two `2^floor(log2(448/amax))` from the row's exact amax, computed in the row kernel from the tile already in registers, cast once (`cvt.rn.satfinite.e4m3x2`) into 16 registers, reused by every product that reads it. No calibration, no static scale, no overflow guard (a cast of a row scaled to its own max cannot saturate). |
| weight scale | static power of two per 64-row unit / matrix (q lo/hi, gate, k, v; Wo units; Wa_j / Wb_j halves; Wd), once per weight version (`f8.pack8`, cached like the bf16 pack); K axis permuted to the accumulator order of the previous product (slot `16h+4t+2a+b`) so the accumulator pairs are the A fragment without shuffles. |
| epilogue | `acc * (1/row scale) * (1/weight scale)` in fp32 (exact power-of-two arithmetic); the squeeze over the 4 hidden chunks (each chunk its own row scale) is summed in fp32 registers AFTER dequantisation (rescaling the wgmma accumulator between chunks gave 95 % wrong output). |
| per-sample conditioning | falls back to bf16 (`why_cond`): AF3-family per-sample tables lost 1.09-1.12 (A = 1: the row kernels are a 1-sample, latency-bound grid and the e4m3 casts add to the critical path). 9 of 27 block calls per run (every model; the eager first call + the blocks that carry per-sample cond) run bf16. Route is therefore gated on a conditioning shared by the samples (`cond` an expanded view, stride 0, A > 1). |
| options | classes served: AdaLN (Protenix / OpenDDE, AF3 / OpenFold3 / IntelliFold2), Chai-1 (`IDENT`/`PARS`/`NOOUT`), ESMFold2 (`RMS`, branch esm-opendde), RoseTTAFold3 (`KQLN`, branch rf3-kqln: affine LayerNorm of the flat q / k on the register tile, `kq_ln_unit8`), Boltz-2 (`UP`, `post90_f8up.cu`: extra Wu_j chunk; gate `FP8_UP_MIN_N = 2048` atoms). fp32 and training never. |
| switches | `MINIWORLD_ATOMDIT_FP8=1/all/pre/post`, `local_dit_h100.enable_fp8()`; `safe` preset turns it on. Default off. |
| gates and why | shared conditioning (above); UP from 2048 atoms (below, the 3rd accumulator + 2nd weight stream loses); `pre` alone never wins on the H100 (0.94-0.99 module) — the gain is in `post` (0.91-0.97) because its products are 229 kflop / row against 131 k and it is the serial `cp.async -> wgmma -> epilogue` chain. Register trade: `pre90_f8` 236 regs, `post90_f8` 253; `f8._minb` caps at 168 (3 CTAs/SM) where the grid is one wave shorter. |
| accuracy | judge = max|y - y64| / max|y64|, tol = max(err_pytorch, 8e-3). FP8 pre+post judge 4.8-5.7e-2 (6-11x tol; rms 2.9 %, 7x the bf16 route's 0.4 %); pre only 1.2-2.1e-2 (rms 0.8 %); post only 4.6-5.4e-2; Chai-1 twice as large (identity AdaLN, raw c). Boltz-2 (% of output): pre 1.6 %, post 5 %, both 5.3 %. FP8 stays OUTSIDE the judge; acceptance = end level (200-step diffusion, RMSD to bf16 <= max(1.1x bf16 pair max, 0.5 A), finite, route stats, `check()` unused). |
| speedups (module, graph, sustained, A = 5 shared) | OpenFold3 1024 / 2048 / 4096 / 8192 / 12288 / 16384 atoms: 0.916 / 0.944 / 0.915 / 0.974 / 0.961 / 0.949; Protenix 4096 0.905, 16384 0.956; Chai-1 4096 0.955. Ablation upper bound: e4m3 products 0.87-0.91 of the step, weight bytes 0.95-0.97, attention <= 0.99-1.0. End to end: 0.3-1.2 % of a forward (atom DiT is 3-12 % of a warm forward). |
| traps | (1) per-sample cond loses; (2) `pre` alone loses; (3) rescaling a wgmma accumulator between K chunks silently wrong; (4) UP below 2048 atoms loses; (5) Chai-1 identity AdaLN scale is unbounded: error twice as large; (6) ESMFold2 census was 0 until `f8.OPTIONS` allowed `RMS` and `_infer_h100` got an FP8 branch (route counters are the test); (7) the H100 5I28 path never reaches `AtomDiTESMFold2Pair` for some models. |

## 2. Token DiT placements (H100)

| placement | e4m3 tensors / scale origin | gate (rows M = A B L) | H100 result | why |
|---|---|---|---|---|
| expand (+SwiGLU) | xt from the quantising AdaLN pass (`resgate_adaln_cf_q8`; Chai-1 `adaln_in_rows_cf_par_q8`) x [Wa; Wb]; sx per row (max/448), sw per output channel | 640 (`MIN_ROWS_PER`) | wins at every M (0.94-0.98); in `safe` | error 5.4 % rms accepted at end level |
| qkvg | xa from `adaln_in_cf_q8` x Wqkvg via `torch._scaled_mm` row-wise, bf16 out + bias | 2560 | loses below 2560 (1.006-1.019), 0.97-0.99 above; NOT in `safe` (the one placement that moved FoldBench); never for Chai-1 identity AdaLN | small M latency; error 1-1.3 % at x1 but 3.5-3.9 % at x0.05 input |
| out | og (attention output) needs its OWN row pass (16 head CTAs write a row, nobody owns the row max) | 2560 | 1.02-1.05 loss | row pass 5-9 us |
| squeeze | h needs its own row pass (a row spans 12 expand tiles) | 2560 | loss, error 3.7 % | same |
| up (Boltz-2) | expand reads [Wa 64 | Wb 64 | Wu 64] n192 tiles, three channel scales, `silu(a) b u` | `MIN_ROWS_UP` 2560 | kernel -27 %, block +2.7 % at 1920 because the quantising pass costs a third reduction (row passes 12.6 -> 14.0 us); 0.998 at 2560, 0.954 at 7680 | |
| RoseTTAFold3 | parallel non-identity block (k-q LN): `rows_q8(xt)` standalone, expand only | via `active(parallel=...)` | expand only | transition does not depend on the attention part |
| Chai-1 identity | expand / out / squeeze; never qkvg | | whole gain 0.6 % of kernel time | |

Scale floors: weights `clamp_min(1e-30)`; activation scale 1 when the row amax is 0. No saturation by construction -> no calibration / guard / `take_overflow`.
Stress protocol: input x 0.05 and x 20, 3 masked keys, random weights, LN 1 +- 0.1, graphs replayed.

## 3. Transfers unchanged vs redo for the B200

| item | atom DiT | token DiT |
|---|---|---|
| unchanged | per-row dynamic scale max/448 from the row the AdaLN thread owns; weight per-output-channel scale; e4m3 `cvt.rn.satfinite`; dequantising epilogue in fp32; gating on shared conditioning; option matrix (plain / IDENT / PARS / KQLN / UP / RMS); acceptance protocol (fp64 gold, stress, graph == eager, route stats); no calibration / no guard | whole recipe, row kernels `token_dit_fp8_rows.cu` (arch neutral), `_scaled_mm` for qkvg / out / squeeze, gates as starting values, route-stats keys |
| redo for tcgen05 | wgmma `m64nNk32` register-A -> `tcgen05.mma kind::f8f6f4` (K = 32/step, A from TMEM for the transposed 32/16-row tiles, weights in TMEM via `tcgen05.cp`, accumulators in TMEM, scale in the TMEM->register epilogue where lane = channel and column = row: the row scale is per COLUMN (smem float2), the weight scale per lane); register-held 16-reg A fragment disappears (B operand = 128-B swizzled e4m3 rows in smem written by the AdaLN threads, thread ak's 16 channels = chunk ak of the row, K axis permuted in the pack); power-of-two scales dropped for exact max/448 (no accuracy penalty; epilogue multiply is fp32 anyway) | 2-CTA M256 N256 tile, TMEM double buffer (`gemm_swiglu_f8_sm100.cu`), up form needs N = 192 tile split a64+b32 / b32+u64 over the CTA pair; thresholds re-measured (bf16 B200 expand kernel only from 3840 rows) |
| B200 chain differences | pre / post are tcgen05 transposed-tile kernels already TMA / row-thread bound (tensor time 13-16 % of the block); MMAs overlap the row work (H100: serial chain); side streams (pair bias), fold, pair32, vmn and TF32 stages are fp32-only and stay untouched; PDL, EB prefetch of the pair bias inside `pre_fwd` (`-DEB`), `ADA1` / `RES1` roundings kept | transition on a side stream beside the attention core in the parallel block (Chai-1): expand grid capped at 96 CTAs |
