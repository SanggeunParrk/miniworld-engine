# FP8 (e4m3) on the B200 (sm_100a) — opt-in inference routes (engine 3.0.3 + B200 kernel set, agent A, 2026-11-19)

Everything here is **opt-in and off by default**. With FP8 off the B200 default routes are bit-identical to the tree without it (section 6).
Scope of this document (agent A): the foundation facts (tcgen05 e4m3 vs bf16 under the power cap), the framework (`src/miniworld_engine/fp8.py`) and the **Transition**
route (`kernels/transition/cuda/fp8_sm100a.py`, `sm100/fp8/tf8_gemm.cu`, `sm100/fp8/tf8_ln.cu`). TriMul (agent B) and the token / atom DiT (agent C) have their own sections
in their trees; the shared facts are in section 2 and `work/n13-fp8-a/mb_results.txt`.

## 0. Summary

* e4m3 `tcgen05.mma kind::f8f6f4` runs at **1.9x the bf16 rate under the ~1000 W cap** (2.06-2.17 vs 1.08-1.17 PFLOP/s on 4096^3 / 8192^3, same tile and pipeline), energy per GEMM
  0.52x. A GEMM shrinks by that factor only where it is tensor-bound (K >= 512); at K <= 384 the epilogue and the bf16 output bound it (x0.8-0.95).
* The Transition on e4m3 (LayerNorm + quantize, expand + SwiGLU with e4m3 h, squeeze + residual: 3 PDL launches) is **faster than the B200 default route at D 512 and D 768**
  (sustained time and energy x0.47-0.88; D 768 n 4 x0.63, D 512 n 4 x0.77 at large M) and is **not served** at D 128 / 256 / 384 (x1.0-2.0: measured losses, section 4.6).
* Accuracy = the H100 FP8 band (judge error 0.034-0.039 against fp64 at x ~ N(0,1), update rms 6.6 %), no NaN, saturation handled by the 3.0.3 guard (sync redo / lazy +
  `fp8.check()`, also through captured graphs).
* Framework: `fp8.py` serves capability (10, 0) for the slots in `fp8.B200_ROUTES` (Transition now); `MINIWORLD_FP8=safe` / `engine-fp8` turn the route on on a B200.

## 1. H100 design digest (what 3.0.3 does and what transfers)

### 1.1 Framework (`src/miniworld_engine/fp8.py`, 3.0.3)
* `PRESETS["safe"]` maps per-route env names to values (TRIMUL=calib, OPM=bound, PWA=1, TRANSITION=calib, TOKEN_DIT=expand, ATOMDIT=1; TriAttn per validated model).
  `MINIWORLD_FP8=safe` selects it; `env(name, default)`: a variable present in `os.environ` (even `""` = off) wins, else the preset value, else `default`. Route modules read it **at import**
  (`POLICY = fp8.env("MINIWORLD_X_FP8") or None`); `enable()` / `disable()` set the module attributes at run time.
* `engine-fp8` backend: `resolve_backend` -> (`cuda`, FP8 on); `set_backend` installs a `_forward_cuda` wrapped in `BackendScope(on=True)` which `enable("safe")`s for the call and restores the
  switches on exit (stack `SCOPES`, `_BASE`); training (module.training or autograd enabled) never opens the scope. Every route's `serves()` was H100-only (`capability == (9, 0)`).
* `route_stats()` / `reset_route_stats()`: per-route plain-int counters bumped at Python call time (a graph replay does not recount). Transition keys: `calls retries fallbacks capture_fallbacks calibrations`.
* Guard protocol (static-scale routes): scales calibrated on the first eager call of a module (probe call with the scale 1: the pre-cast peak is exact even when it saturates), headroom 16; `sync`
  (eager: the wrapper redoes an overflowing call, up to 4 times), `lazy` (kernel max-atomics the scaled peak into a per-device buffer; `fp8.check()` once per forward outside any graph: peak > 448 ->
  `rescale_all(peak)` lowers the static scale in place, captured graphs follow, rerun), `off`. A capture of an uncalibrated module takes the default route (`capture_fallbacks`).
* Traps recorded in 3.0.x: scale floors (3.0.3: pow2 scale floored at 2^-30 / `1e-30` on amax: a scale of 2^-99 saturates later input, `1 / scale = inf` = a real-weight NaN); the e4m3 NaN
  byte (a hand-formed byte must be clamped to 126; `cvt.rn.satfinite.e4m3x2` is safe); FP8 is outside the judge (no registry row / `backend_versions` entry).

### 1.2 Transition on the H100 (`fp8_sm90a.py`, `transition_fp8_fwd_sm90a.cu`, `transition_fp8_d256_sm90a.cu`, `wide/fp8_bind.cu`, tests `test_transition_fp8*_h100.py`, docs/fp8.md "Transition forward")
| tensor | e4m3 | scale | where |
|---|---|---|---|
| xn = LayerNorm(x), expand A | yes | ONE pow2 scale from the LayerNorm bound `sx = 2^floor(log2(448 / (sqrt(D-1) max|gamma| + max|beta|)))`: cannot saturate | host per weight version; xn quantized from the fp32 LayerNorm in the LN pass |
| Wa, Wb, Ws | yes | per-tensor pow2, `448 / amax` | host, per parameter version, cached (key: data_ptr + `_version` of the 5 parameters) |
| h = silu(a) b | yes | static pow2 `sh`, calibrated (`HEADROOM` 16 over the calibration amax; measured peaks 15-28 of 448), guarded | expand epilogue, `cvt.rn.satfinite`, max|h sh| -> guard buffer |
| x, y, residual, LayerNorm math | bf16 / fp32 | | |
* silu by ONE `tanh.approx` (`silu(a) = (a/2)(1 + tanh(a/2))`): pays with e4m3 GEMMs (0.68-0.80 vs 0.86-0.94 with ex2 + rcp), not with bf16.
* Gates (rows = L^2, whole 128-row tiles): D 128 16384, D 256 8192, D 384 / 512 65536 (below: the default's generic kernels are faster, 128-row tiles leave SMs idle; at token counts the wide FP8
  chain was 1.5-3.4x SLOWER; cuBLAS e4m3 bounds the gain there at 0-17 %). Speed (cool / sustained, graph, vs 2.5.12): D128 0.66-0.76, D256 0.69-0.77, D384 n4 0.64-0.69, D512 n4 0.56-0.62.
* Accuracy acceptance (fp64 gold, weights N(0, 1/fan_in), gamma 1 +- 0.1, beta +- 0.1, x ~ N(0,1)): judge `max|y - y64| / max|y64|`: bf16 PyTorch 0.0044-0.0068, default 0.0035-0.0042,
  FP8 full 0.035-0.045 (rms 0.034, update rms 0.066), h + squeeze only 0.020-0.029; no NaN, no saturation. All FP8 variants are 3-5x outside the 8e-3 judge tolerance, accepted because FP8 is outside the judge.
* Tests to mirror: fragment permutation (wgmma only); error vs fp64; bf16 twin = default bytes; guard sync retry and lazy `overflow_peak` / `rescale_all`; default untouched when off; graph = eager.
* Rejected / not done on the H100: D 64 (bound 0.72-0.76 on paper), per-channel weight scales, e4m3 x / y storage, e5m2, training (4-11 % step time, gradient error 7-10x bf16).

### 1.3 Transfers unchanged vs redone for tcgen05
| transfers unchanged (used as is) | redone for tcgen05 |
|---|---|
| quantization recipe: LayerNorm-bound xn scale, per-tensor pow2 weight scales shared by Wa / Wb, static calibrated h scale with headroom 16 | MMA `kind::f8f6f4`, instruction descriptor, K = 32 per MMA |
| pack once per weight version, cache keyed by (data_ptr, `_version`), calibrate on the first eager call, capture fallback | smem layout (128 e4m3 per 128-B swizzle row, no K permutation: the H100's `_perm_k` was a wgmma register-fragment trick) |
| guard protocol (`sync` / `lazy`, `overflow_peak`, `rescale_all`, `fp8.check()`), route stats keys, `MINIWORLD_TRANSITION_FP8*` switches | epilogue (TMEM -> registers, scales, satfinite cast, st.global), pipelining (TMA ring, TMEM double buffer, PDL chain) |
| tanh silu | tile shapes (128 x BN, BN 128 / 192 / 256), launch structure (3 launches instead of one fused kernel) |
| acceptance: fp64 gold, judge band 0.02-0.045, no NaN | gates: re-measured against the B200 default route (v66), not carried over |
Deviations from the H100 recipe (each with a measured reason): the scale floor `AMAX_FLOOR = 1e-12` instead of `1e-30` (so that `1/(sx s1)` stays finite in fp32 for tiny gamma / weights),
the row gates (re-measured), D 128 / 256 / 384 not served (loses on B200 against v66).

## 2. Foundation facts: tcgen05 e4m3 vs bf16 under the power cap
Kernel `sm100/fp8/tf8_gemm.cu` (`-DE4=1` e4m3 `kind::f8f6f4`, `-DE4=0` bf16 `kind::f16`; MODE 2 = bf16-out GEMM): same 128 x BN tile, 128-B swizzle K-major operands, 4 MMAs per 128-B K block (K = 32 e4m3 /
16 bf16 per MMA, 128 / 64 elements per block), ring of 4-6 stages, TMEM double buffer, 8 epilogue warps, fp32 scale multiply, direct `st.global` of bf16. Method: a CUDA graph of 8 launches
replayed back to back for 2 s (sustained, no L2 flush), `nvidia-smi` power / SM clock every 0.2 s over the window, J = mean W x ms (`work/n13-fp8-a/mb.py`, `mb_raw*.jsonl`, `mb_table2.txt`; clean runs: another
user's job shared GPU 0 for ~15 min and its rows were rerun).
| GEMM (M N K) | e4m3 TFLOP/s | W | MHz | mJ | bf16 TFLOP/s | W | MHz | mJ | time x | energy x |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096 4096 4096 | 2156 | 862 | 1602 | 54.9 | 1167 | 887 | 1573 | 104.4 | 0.54 | 0.53 |
| 8192 8192 8192 | 2058 | 941 | 1106 | 502.7 | 1083 | 957 | 1078 | 972.0 | 0.53 | 0.52 |
| expand D 128 (65536, 1024, K 128) | 354 | 791 | 1965 | 38.4 | 324 | 862 | 1950 | 45.7 | 0.92 | 0.84 |
| expand D 256 (65536, 2048, K 256) | 672 | 884 | 1894 | 90.4 | 530 | 892 | 1785 | 115.5 | 0.79 | 0.78 |
| expand D 384 (65536, 3072, K 384) | 770 | 902 | 1949 | 181.0 | 612 | 914 | 1773 | 230.8 | 0.79 | 0.78 |
| expand D 512 (65536, 4096, K 512) | 1061 | 917 | 1746 | 237.6 | 782 | 927 | 1587 | 326.0 | 0.74 | 0.73 |
| expand D 768 (65536, 6144, K 768) | 1235 | 940 | 1719 | 470.9 | 910 | 953 | 1520 | 648.3 | 0.74 | 0.73 |
| squeeze D 128 (65536, 128, K 512) | 733 | 665 | 1965 | 7.8 | 583 | 771 | 1965 | 11.3 | 0.80 | 0.69 |
| squeeze D 256 (65536, 256, K 1024) | 1413 | 852 | 1850 | 20.7 | 774 | 875 | 1604 | 38.8 | 0.55 | 0.53 |
| squeeze D 384 (65536, 384, K 1536) | 1392 | 871 | 1627 | 48.4 | 787 | 888 | 1646 | 87.2 | 0.57 | 0.56 |
| squeeze D 512 (65536, 512, K 2048) | 1793 | 883 | 1447 | 67.7 | 1036 | 885 | 1400 | 117.5 | 0.58 | 0.58 |
| squeeze D 768 (65536, 768, K 3072) | 1968 | 899 | 1455 | 141.3 | 1112 | 915 | 1413 | 254.6 | 0.56 | 0.55 |
(the expand rows are the bf16-output stand-in with N = 2 x 4 D, so the output bytes are the bf16 ones; the whole table: `mb_table2.txt`.)
Facts: (a) compute-bound: **x1.9 the bf16 rate and x0.52 the energy under the cap**; both dtypes run near 860-950 W, the cap lowers the clock (1.1 GHz at 8192^3 for e4m3) and the e4m3 rate stays
2.06 PF/s. (b) K <= 384: the tile is epilogue / output-bound, gain only x0.8-0.95 (D 128 expand 0.92-0.99). (c) low arithmetic intensity (ai(e4) 120 flop/B at D 128): operand bytes halve but a bf16 output does not,
so a route keeps the byte saving only if the NEXT kernel takes e4m3 straight from this kernel's epilogue (the Transition's h) -- an extra quantization pass costs more than it saves. (d) Quantization pass `tf8_rowquant`
(bf16 read + e4m3 write + per-row scale, sustained without L2 flush): [16384,128] 4 us, [65536,384] 15 us (5.2 TB/s, L2 hits), [65536,512] 18 us, [65536,768] 31 us, [262144,128] 56 us (HBM): ~10-20 % of an e4m3 expand GEMM as a standalone pass; the LayerNorm + quantize kernel `tf8_ln` of the Transition takes 25-35 us at 65536 rows (eager, with its launch floor). cuBLAS bf16 on this box: 1319 / 1350 TFLOP/s at 4096^3 / 8192^3, so the e4m3 kernel is x1.55 cuBLAS bf16 (the 1-CTA kernel is 82-87 % of cuBLAS in bf16).
(e) The fp32 scale multiply in the epilogue is in both builds (1 FMUL / element) and not separable. (f) Instruction facts, verified by running: idesc = `(1<<4) | (N>>3)<<17 | (M>>4)<<24` with A / B format 0 = e4m3,
descriptors as bf16 (`desc_k128`, +2 x 16 B per MMA K step = 32 B), TMA maps of an e4m3 tensor = a bf16 view `[rows, K/2]` with a 64-element box (the host extension needs no change). (g) Kernel error against an fp32
reference of the same e4m3 inputs 2.4e-3 (the bf16 output rounding).

## 3. Framework change (`fp8.py`)
* `B200_ROUTES: {slot: module path}` (Transition: `kernels/transition/cuda/fp8_sm100a`); `SLOTS`; `arch()` ("sm100" on a B200); `serves_sm100(slot)`. `_modules()` is cached per arch and returns the B200 module in
  the slot's place on a B200, the H100 module otherwise (so `enable` / `disable` / `status` / `check` / `route_stats` / `BackendScope` need no other change). A slot without an entry keeps the H100 module, whose `serves()` is
  False on a B200. A B200 route module must offer the interface of its slot (`POLICY`, `GUARD`, `GUARD_STATS`, `overflow_peak`, `rescale_all`, ...) and read its switch through `fp8.env` with the preset's variable name.
* Same preset semantics: `MINIWORLD_FP8=safe` sets `MINIWORLD_TRANSITION_FP8=calib` for both archs; a per-route variable set explicitly (also `""`) wins; default off (`tests/cpu/test_fp8_switch.py`: `test_b200_route_table`,
  `test_b200_transition_route_follows_the_preset`). The token DiT's B200 route (agent C) lives in the H100 module `token_dit_fp8` itself, so it needed no entry.
* Guards: the `and not b200` guard of `conditioned_transition/cuda/__init__.py` is agent C's (removed only for the token-DiT expand route that runs on B200). Nothing else changed in the dispatch except one
  `POLICY is not None` test at the top of `route_sm100a.transition_sm100a` (and its private `_fp8=False` keyword for the guard's fallback).

## 4. Transition FP8 on the B200 (`fp8_sm100a.py`)
### 4.1 Design (the H100 recipe on tcgen05; one switch `MINIWORLD_TRANSITION_FP8=calib`, also `MINIWORLD_FP8=safe` / `engine-fp8`)
Three launches (PDL chain, `route_sm100a.transition_sm100a` takes the route first when `fp8_sm100a.POLICY is not None`, no grad, `serves()`):
1. `tf8_ln` (`sm100/fp8/tf8_ln.cu`): one warp per row, fp32 two-pass LayerNorm, xn quantized straight to e4m3 with the LayerNorm-bound scale `sx` (never saturates).
2. `tf8_gemm` MODE 0 (expand): A = xn8 [M, D], B = Wa8 / Wb8 [n D, D], two fp32 accumulators in TMEM (`kind::f8f6f4`, BN 128 per accumulator, 2 x 2 x 128 = 512 columns), epilogue
   `h = silu(acc_a ca) * (acc_b ca sh)` with the tanh silu, `cvt.rn.satfinite.e4m3x2`, st.global of the e4m3 h [M, n D] (half the bytes of the bf16 round trip), per-CTA max |h| -> `atomicMax` on the guard buffer.
3. `tf8_gemm` MODE 1 (squeeze): A = h8, B = Ws8 [D, n D], BN 256 when D % 256 == 0 and every SM keeps a tile, 192 when D % 192 == 0 (D 384), else 128; `out = x + acc / (sh s2)` in bf16.
Scales = the H100's: Wa and Wb share one per-tensor pow2 scale `s1`, Ws one `s2`, `sx` from the LayerNorm bound, `sh` static (HEADROOM 16, calibrated on the first eager call, probe with sh = 1), kernel vector
`[sx, 1/(sx s1), sh/(sx s1), 1/(sh s2)]` refreshed in place. Floors: amax >= 1e-12. Weights packed once per (data_ptr, `_version`) of the 5 parameters (fp32 masters or bf16). Any row count (TMA zero-fills the tail
tile, stores / residual loads are row-masked), D a multiple of 128 (`tf8_ln` D <= 1024), n 2 / 4. `fp8.route_stats()["transition"]`: `calls retries fallbacks capture_fallbacks calibrations`.
Cubins (`sm100/fp8/`): `ln`, `swiglu128`, `resid128`, `resid192`, `resid256` (prebuild group `transition_sm100a/fp8/*`, +5 artifacts in `_prebuild --arch sm100`). No host extension change.
Kernel times at 65536 rows (eager, us; a launch floor is in each): D 384 ln 33 / swiglu 96 / resid 62; D 512 34 / 143 / 75; D 768 43 / 282 / 140. Epilogue ablation of the expand kernel (`ABL`, wrong output,
timing only): D 768 full 288 us = no-math 251 (MMA-bound, 2.15 PF/s); D 512 143 vs 123; D 384 93 vs 74; D 256 53 vs 38: below K = 512 the silu + store epilogue and the tile prologue cost 20-30 %.

### 4.2 Gates (rows = batch x L^2 x ..., measured against the B200 default route)
`ROW_GATE = {(512, 4): 1024, (512, 2): 4096, (768, 4): 4096, (768, 2): 4096}`; `WIDTHS = 512, 768`; `MINIWORLD_TRANSITION_FP8_MINROWS` / `_WIDTHS` override (experiments only). Reasons: D 768 n 4 x1.07 at 1024
rows, 0.90 at 2048, 0.60-0.65 from 4096; D 768 n 2 x1.30 at 1024, 1.16 at 2048, 0.51-0.76 from 4096; D 512 n 4 x0.50 from 1024 rows (the default is a slow path there: 45 us at 1024 rows); D 512 n 2 x0.63 at 4096, 0.88 at 16384+.
Not served: **D 384** (x0.93-1.09 at 4096-262144 rows, 1.50 at 1024 rows, n 2: 1.09-1.21), **D 256** (x1.17 n 4 65536 rows, 1.36 n 4 262144, 1.46 n 2), **D 128** (x2.0): see 4.6.

### 4.3 Results: B200 default route vs e4m3 route, same process, interleaved, 3 rounds
Graph medians (CUDA graph of one call, 256 MB L2 flush before each replay, medians of 30 replays after a 0.15 s idle gap) and a sustained window (graph replayed back to back for 3 s per round, `nvidia-smi` power
every 0.2 s, J = mean W x ms per step, no L2 flush); master (fp32) weights, bf16 x, lazy guard, calibrated eagerly; no overflow in any case (peak 17 of 448). Script `work/n13-fp8-a/ab.py`, rows `ab_final.jsonl`.
The sustained columns of the 1024- / 4096-row cases are launch-bound (python replay loop, 350-700 W): read them as ratios only.

| D | n | rows | default ms (graph, L2 flushed) | e4m3 ms | x graph | default ms (sustained) | e4m3 ms | x sustained | W default / e4m3 | J default | J e4m3 | x J |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 4 | 1024 | 0.0451 | 0.0225 | 0.50 | 0.0389 | 0.0184 | 0.47 | 355 / 417 | 13.801 mJ | 7.682 mJ | 0.56 |
| 512 | 4 | 4096 | 0.0553 | 0.0328 | 0.59 | 0.0479 | 0.0281 | 0.59 | 609 / 689 | 29.124 mJ | 19.344 mJ | 0.66 |
| 512 | 4 | 16384 | 0.0903 | 0.0758 | 0.84 | 0.0979 | 0.0751 | 0.77 | 876 / 874 | 85.829 mJ | 65.602 mJ | 0.76 |
| 512 | 4 | 65536 | 0.3113 | 0.2417 | 0.78 | 0.3716 | 0.2854 | 0.77 | 897 / 899 | 333.250 mJ | 256.549 mJ | 0.77 |
| 512 | 4 | 262144 | 1.2524 | 0.9544 | 0.76 | 1.4551 | 1.1173 | 0.77 | 920 / 916 | 1338.833 mJ | 1022.924 mJ | 0.76 |
| 512 | 2 | 4096 | 0.0389 | 0.0246 | 0.63 | 0.0328 | 0.0205 | 0.62 | 534 / 578 | 17.498 mJ | 11.829 mJ | 0.68 |
| 512 | 2 | 16384 | 0.0573 | 0.0532 | 0.93 | 0.0534 | 0.0470 | 0.88 | 870 / 843 | 46.497 mJ | 39.580 mJ | 0.85 |
| 512 | 2 | 65536 | 0.1762 | 0.1618 | 0.92 | 0.2079 | 0.1775 | 0.85 | 890 / 890 | 185.067 mJ | 158.007 mJ | 0.85 |
| 768 | 4 | 4096 | 0.0821 | 0.0492 | 0.60 | 0.0749 | 0.0430 | 0.57 | 867 / 849 | 64.929 mJ | 36.512 mJ | 0.56 |
| 768 | 4 | 16384 | 0.2007 | 0.1311 | 0.65 | 0.2302 | 0.1457 | 0.63 | 890 / 885 | 204.948 mJ | 128.899 mJ | 0.63 |
| 768 | 4 | 65536 | 0.7374 | 0.4629 | 0.63 | 0.8710 | 0.5544 | 0.64 | 908 / 905 | 790.520 mJ | 501.989 mJ | 0.64 |
| 768 | 4 | 131072 | 1.4766 | 0.9259 | 0.63 | 1.7236 | 1.0931 | 0.63 | 920 / 913 | 1586.009 mJ | 998.013 mJ | 0.63 |
| 768 | 2 | 4096 | 0.0655 | 0.0348 | 0.53 | 0.0574 | 0.0290 | 0.51 | 591 / 720 | 33.902 mJ | 20.872 mJ | 0.62 |
| 768 | 2 | 16384 | 0.1126 | 0.0901 | 0.80 | 0.1188 | 0.0884 | 0.74 | 873 / 872 | 103.721 mJ | 77.092 mJ | 0.74 |
| 768 | 2 | 65536 | 0.3789 | 0.2888 | 0.76 | 0.4401 | 0.3345 | 0.76 | 896 / 895 | 394.329 mJ | 299.413 mJ | 0.76 |

(`x` columns = e4m3 route / default route; < 1 = the e4m3 route is faster / cheaper. D 384 and D 256 / 128: section 4.6.)

### 4.4 Accuracy against the fp64 gold (judge form `max|y - y64| / max|y64|`; non-zero random weights N(0, 1/fan_in), gamma 1 +- 0.1, beta +- 0.1, 16384 rows, calibrated on the case)
Next to each case the B200 DEFAULT route's error and the ratio. The default's error is bf16 (0.0035-0.0050 at x ~ N(0,1)). `scale` = x std, `off` = mean offset (x is bf16-rounded before both routes and the gold).
No NaN / inf and no guard retry in any of the 36 cases. (`work/n13-fp8-a/acc.py`, rows `acc_raw.jsonl`; the update rms column is `rms(y - y64) / rms(y64 - x)`.)
```
WARNING: foreign pid 3966857 on GPU 0: .venv/bin/python /home/khidi_hjju/tmp/distill/known/famtest/predict.py --part 0 --parts 8 
   D  n  xscale  off | judge def judge fp8  ratio |  rms def  rms fp8  ratio | upd8  nan retries
 384  4   0.001    0 |    0.0049    0.0591   12.0 |   0.0044   0.0650   14.8 | 0.065 0 0
 384  4   0.001   25 |    0.0007    0.0007    1.0 |   0.0002   0.0002    1.0 | 1.000 0 0
 384  4       1    0 |    0.0036    0.0383   10.8 |   0.0027   0.0344   12.6 | 0.066 0 0
 384  4       1   25 |    0.0023    0.0085    3.6 |   0.0014   0.0022    1.5 | 0.088 0 0
 384  4      40    0 |    0.0024    0.0031    1.3 |   0.0017   0.0019    1.2 | 0.127 0 0
 384  4      40   25 |    0.0021    0.0026    1.3 |   0.0017   0.0019    1.1 | 0.143 0 0
 384  2   0.001    0 |    0.0050    0.0648   13.0 |   0.0043   0.0652   15.0 | 0.065 0 0
 384  2   0.001   25 |    0.0006    0.0006    1.0 |   0.0002   0.0002    1.0 | 1.000 0 0
 384  2       1    0 |    0.0036    0.0374   10.3 |   0.0027   0.0341   12.6 | 0.066 0 0
 384  2       1   25 |    0.0024    0.0081    3.4 |   0.0014   0.0021    1.5 | 0.089 0 0
 384  2      40    0 |    0.0025    0.0031    1.3 |   0.0017   0.0019    1.2 | 0.128 0 0
 384  2      40   25 |    0.0021    0.0025    1.2 |   0.0017   0.0019    1.1 | 0.145 0 0
 512  4   0.001    0 |    0.0044    0.0639   14.4 |   0.0043   0.0650   15.0 | 0.065 0 0
 512  4   0.001   25 |    0.0006    0.0006    1.0 |   0.0002   0.0002    1.0 | 1.000 0 0
 512  4       1    0 |    0.0035    0.0369   10.4 |   0.0027   0.0341   12.7 | 0.066 0 0
 512  4       1   25 |    0.0024    0.0079    3.3 |   0.0014   0.0021    1.5 | 0.089 0 0
 512  4      40    0 |    0.0023    0.0028    1.2 |   0.0017   0.0019    1.2 | 0.127 0 0
 512  4      40   25 |    0.0022    0.0028    1.3 |   0.0017   0.0019    1.1 | 0.144 0 0
 512  2   0.001    0 |    0.0047    0.0636   13.5 |   0.0043   0.0651   15.1 | 0.065 0 0
 512  2   0.001   25 |    0.0007    0.0007    1.0 |   0.0002   0.0002    1.0 | 1.000 0 0
 512  2       1    0 |    0.0037    0.0353    9.6 |   0.0027   0.0342   12.6 | 0.066 0 0
 512  2       1   25 |    0.0024    0.0086    3.6 |   0.0014   0.0022    1.5 | 0.088 0 0
 512  2      40    0 |    0.0021    0.0025    1.2 |   0.0017   0.0019    1.2 | 0.127 0 0
 512  2      40   25 |    0.0021    0.0027    1.3 |   0.0017   0.0019    1.1 | 0.144 0 0
 768  4   0.001    0 |    0.0046    0.0668   14.6 |   0.0044   0.0651   14.8 | 0.065 0 0
 768  4   0.001   25 |    0.0007    0.0007    1.0 |   0.0002   0.0002    1.0 | 1.000 0 0
 768  4       1    0 |    0.0036    0.0342    9.5 |   0.0027   0.0343   12.6 | 0.066 0 0
 768  4       1   25 |    0.0024    0.0082    3.5 |   0.0014   0.0022    1.5 | 0.088 0 0
 768  4      40    0 |    0.0024    0.0031    1.3 |   0.0017   0.0019    1.2 | 0.127 0 0
 768  4      40   25 |    0.0022    0.0027    1.2 |   0.0017   0.0019    1.1 | 0.144 0 0
 768  2   0.001    0 |    0.0045    0.0659   14.8 |   0.0044   0.0652   14.8 | 0.065 0 0
 768  2   0.001   25 |    0.0006    0.0006    1.0 |   0.0002   0.0002    1.0 | 1.000 0 0
 768  2       1    0 |    0.0034    0.0360   10.6 |   0.0027   0.0341   12.6 | 0.066 0 0
 768  2       1   25 |    0.0024    0.0082    3.4 |   0.0014   0.0021    1.5 | 0.089 0 0
 768  2      40    0 |    0.0023    0.0027    1.2 |   0.0017   0.0019    1.2 | 0.128 0 0
 768  2      40   25 |    0.0022    0.0028    1.2 |   0.0017   0.0019    1.1 | 0.144 0 0
```
Reading: at x ~ N(0,1) the e4m3 route is at the H100 FP8 level (judge 0.034-0.039, update rms 6.6 %; the H100 doc: 0.035-0.045 / 6.6 %), 9.5-10.8x the bf16 default's judge error (10x was the H100's ratio too). At
x std 1e-3 the LayerNorm eps dominates (var 1e-6 << eps), y ~ x and the update is tiny, so the judge form (relative to max|y|) reads 0.059-0.067 for the same 6.5 % update error. Large activations (std 40) and offsets (25) are
benign: the error is dominated by the bf16 input rounding of x that both routes share (ratio 1.1-1.3). Offset 25 with std 1 (x quantized to 0.125 steps by bf16): 3.3-3.6x on the judge form (rms 1.5x).
All FP8 judge errors exceed the 8e-3 judge tolerance (as on the H100; FP8 is outside the judge).

### 4.5 Saturation and the guard (tests `test_sync_guard_redoes_a_saturated_call`, `test_lazy_guard_check_and_rerun_through_a_graph`)
`h` is the only static-scale tensor (xn: LayerNorm bound, weights: exact amax). LayerNorm makes h independent of the x scale and offset, so only a direction change (or a different distribution) can raise it. Provoked by calibrating on x of
std 1e-4 (eps dominates: h ~ 1000x smaller) and then running std 1: `sync` guard: the call detects peak > 448, lowers `sh` by 2^ceil(log2(peak / 448 x 2)) and redoes the call (`retries` >= 1, `fallbacks` 0), result inside
the band; `lazy`: the kernel records the peak (also inside a captured graph: the guard buffer is created before capture), `fp8.check()` returns True once, the scale is lowered in place, the captured graph is replayed
again and `check()` is False (error inside the band). Capture of a module that was never calibrated: the default route (`capture_fallbacks`). e4m3 conversion is `cvt.rn.satfinite` everywhere, so no NaN byte exists.

### 4.6 Measured losses and rejected variants
* **D 384 (x0.93-1.09; 1.5 at 1024 rows), D 256 (x1.17-1.46), D 128 (x2.0)** against the B200 default (v66 tuned fused / wide kernels, 0.0328 ms at D 128 n 4 65536 rows = one fused kernel with LayerNorm in registers): the
  three-launch chain pays ~25-35 us of LayerNorm + launch floor and the expand epilogue at K <= 384 (microbenchmark x0.79-0.95); the default's D 128 / 256 is a one-launch 2-CTA fused kernel. A fused e4m3 forward
  (LayerNorm -> expand -> SwiGLU -> squeeze in one kernel with h in TMEM / smem as e4m3) is the open item for D 128 / 256 / 384 (not attempted: `tr_fwd_sm100.cu` is 2-CTA with 8 roles).
* Squeeze BN 192 for D 384 (`resid192`): 65 -> 62 us only (the squeeze at D 384 is L2-intake-bound: 1536 tiles x 392 KB; the lever would be the CTA pair, not built).
* Expand epilogue in f16x2 (half the MUFU and ALU ops): counted at ~15 % fewer instructions per element, not built (the epilogue ablation bounds the gain at < 6 % of the D 512 route).
* A separate quantization pass for the squeeze operand (instead of the expand epilogue writing e4m3 h): rejected on arithmetic (a pass costs 20-30 % of the GEMM it feeds, section 2d).
* The H100 K permutation (`_perm_k`), the fp32 `-DABL` etc.: not needed (descriptors read the natural order).

## 5. Not done / open
* Fused one-kernel e4m3 Transition (D 128 / 256 / 384 and the AF3 pair transition): not built (see 4.6). TriMul / token DiT / atom DiT: agents B / C.
* Training (out of scope: 3.0.3 measured 4-11 % step time, gradient error 7-10x bf16). Non-power-of-two D (not multiples of 128), D > 1024.
* Squeeze CTA-pair (`cta_group::2`) and a multicast variant for the L2-intake-bound D 384 squeeze; epilogue pipelining for K <= 384.
* End-level (structure) accuracy of the e4m3 Transition on the B200 was not measured (module level only, like the H100 branch before `fp8safe2`).
* No `_prebuild` Transition FP8 cubin is in the default plan unless an `eval` row at D 512 / 768 exists (it does: +5 artifacts).

## 6. Verification
* Bit-identity with FP8 off: `work/n13-fp8-a/sha.py` (12 inference shapes D 64..768 n 2 / 4, eager and captured graph, + a training forward / backward at D 128 / 384 / 512) run in separate processes on the base tree
  (`n13-base`) and this tree: every inference sha1 (eager and graph) and every training y / x-grad / parameter-grad sha1 is identical, except ONE parameter gradient of each training case, which also differs between two runs of
  the BASE tree (atomic accumulation order): `sha_base.json`, `sha_base2.json`, `sha_fp8off.json`. The only default-path change is `POLICY is not None` at the top of `route_sm100a.transition_sm100a`.
* Tests: `tests/gpu/test_transition_fp8_b200.py` (24: route off = default bytes, error vs fp64 for D 128 / 256 / 384 / 512 / 768 x n, large-activation stress, sync guard, lazy guard + `fp8.check()` through a graph,
  graph = eager bytes, uncalibrated capture, gates, ragged rows, route_stats + framework, module `engine-fp8` backend, training stays default, an inf input does not break the guard) 24 passed; `tests/cpu/test_fp8_switch.py` (+2),
  `test_capture_api.py` (+3 listed caches), `test_prebuild_plan.py`: pass; B200 regression (`test_transition_b200_uninit`, `test_trimul_b200`, `test_triattn_b200`, `test_triattn_b200_capture_bits`, `test_misc_b200`,
  ...): 384 passed, 10 skipped (H100-only); the final combined run (this file + `test_fp8_switch` + `test_capture_api` + the B200 transition / trimul / triattn / capture-bits / misc tests): 446 passed. `test_plan_is_reentrant` fails identically on the BASE tree when a GPU test ran earlier in the same pytest process (order-dependent, not this change).
* `_prebuild --arch sm100 --check` (full output `prebuild_check.txt`, base `prebuild_check_base.txt`): 804 -> 809 artifacts (the 5 `transition_sm100a/fp8/*` cubins), native sources compiled by a planned artifact 207 -> 209,
  no other line differs (rc = 1 on both: pre-existing H100-family notes).
* Non-sm100 CUDA sources are byte-identical to `n13-base`; `backend_versions` unchanged (FP8 off changes nothing).

## 7. Fused e4m3 Transition for D 128 (and D 256 / 384): ablation first, decision (2026-11-19, follow-up)
Question: what can an e4m3 MMA (~1.9x tensor rate) save at best in the B200 default D 128 n 4 fused forward `tr_fwd_sm100.cu` (2-CTA, 8 roles, h bf16 in TMEM)? Timing-only variants of a copy
of the kernel (`work/n13-fp8-a/tr_fwd_abl.cu`, kept out of the tree; wrong results; flags `-DABLK=1` halves the MMA instruction count of both GEMMs = what a 2x tensor rate removes at best, `-DABLE=1` removes the SwiGLU
arithmetic, `-DABLE=2` also the LayerNorm statistics), `transition_fused_sm100a` through a CUDA graph, L2 flushed, medians of 50 (5 interleaved rounds, 0.1 s gap), us, `work/n13-fp8-a/trab.py`:
| rows (D 128, n 4) | default | half MMA | no SwiGLU | no SwiGLU + no LN | half MMA + no SwiGLU + no LN |
|---|---|---|---|---|---|
| 4096 | 16.4 | 14.3 (-13 %) | 15.9 | 14.3 | 14.2 |
| 16384 | 16.4 | 16.4 (0 %) | 16.4 | 16.4 | 14.3 |
| 65536 | 34.8 | 32.3 (-7.2 %) | 32.7 | 30.7 | 28.6 |
| 262144 | 88.4 | 82.2 (-7.0 %) | 86.1 | 79.9 | 73.6 |
Reading: the best case of a 2x tensor rate is **7 % of the kernel at 65536-262144 rows** (the AF3 pair-transition range), below the 8 % bar; the 13 % at 4096 rows is on a kernel whose floor is
14.2 us even with the MMA, SwiGLU and LayerNorm all removed (latency / launch bound, not tensor bound). Removing every arithmetic stage together is only 17 % at 262144 rows: the kernel is bound by its
producer / consumer chain and HBM (134 MB of x + y in 88 us = 1.5 TB/s), not by the tensor stage. An e4m3 version would in addition add the xn quantization and h conversion (more epilogue work) and cost the 10x accuracy
of 4.4. **Decision: declined with evidence for D 128** (D 256 is the same kernel family, `tg_fused_h16` / wide, with an even larger epilogue share).
D 384 (chain of `fp8_sm100a`): epilogue share measured in 4.1 (expand: silu + store = 19 us of 93 us; no-math ceiling 74 us vs the 70 us tensor roofline) and squeeze 62 us vs 39 us roofline (L2-intake bound).
Best cases: CTA-pair squeeze -23 us (12 % of the 186 us route), epilogue pipelining -15 us (8 %); at a typical 50-70 % realization 6-8 % each, i.e. a 12-15 % total at best against a default that the chain
only matches (x0.93-1.09). Not built (the break-even is within noise of the default's own variation).


# Agent B: TriMul FP8 (B200)

# B200 FP8 (e4m3) -- opt-in routes (docs/b200/fp8.md)

FP8 on B200 is DEVELOPED for tcgen05, not ported (docs/fp8.md of the H100 engine is the recipe reference: quantisation points, scale granularity, saturation
protocol, gates). Everything here is OPT-IN and OFF by default: with the switches unset the B200 default routes are bit-identical to the tree without it.

## 2026-11-19 -- TriMul e4m3 planes (agent B, branch tree n13-fp8-b)

### Design (what is e4m3, what is not)
| tensor | dtype | scale | where made |
|---|---|---|---|
| planes a / b (front output) | e4m3 | one power-of-two per TENSOR: a and b (per direction in the bidirectional block: 4 scales), static, HEADROOM 16 | `k1w_f8.cu`: plane value * `qinv[chunk]` -> `cvt.rn.satfinite.e4m3x2`, max |scaled value| -> guard (per group atomicMax) |
| contraction accumulator | fp32 (TMEM) | - | `ctr8.cu` `tcgen05.mma kind::f8f6f4` (e4m3 x e4m3, K = 32 per MMA, M128 N128/256) |
| t (contraction output) | bf16 | scale `sa sb` applied in ctr8's epilogue (`csc[2]` device scalars, exact: powers of two) | `ctr8.cu` |
| LN_in / projections / gates / K3 (LN_out, W_o, gate, residual) | bf16 / fp32 | - | unchanged B200 kernels (k3g / k3h / k3hf / k3w) -- K3 sees t at its TRUE scale, no folded operands |
Weights stay bf16 (e4m3 projection operands cost 0.8-3 % update error each on H100: accuracy map in docs/fp8.md; not built). t is not e4m3 (H100: tried, no gain).

Chain: `k1w8` (front: LN_in + the four projections + gate + mask, e4m3 plane store) -> `ctr8` (contraction, PDL-chained exactly like `ctr`) -> default K3.
The B200 cell logic (k1w FOLD / plain, k3hf / k3h / k3g / k3w, XF / OST flags, PDL triggers) is the bf16 chain's: `b200_fp8._body` mirrors `b200_infer._inference_body`
(k1h / M64 fronts and the one-launch L128 kernels are not used on the FP8 route).

Design decisions forced by tcgen05 (the H100 recipe otherwise transfers unchanged):
* **qinv / csc are device arrays read at run time** (the H100 pre-scales the packed weights per calibration). `rescale_all` / `refresh` update them IN PLACE, so captured graphs follow.
* **sa sb is applied in the contraction epilogue** (the H100 folds it into K3's projection operand because cuBLASLt returns the raw sum). K3 is untouched. RoseTTAFold3's length factor f rides on `csc[0]` (`lfac` from the mask at run time: one calibration for any length).
* **Token-shuffled plane layout (the trick that makes the e4m3 front epilogue cheap):** the front's TMEM fragment gives a thread tokens tr and tr + 8; those two bytes are ONE 16-bit element,
  so a single `stmatrix.trans` (+ 4 PRMT) writes the channel-major byte rows instead of a lane exchange + byte stores (first version: k1w8 +35-49 % over k1w; this: +4-12 %).
  The price: inside every aligned 16-token group the plane's contiguous axis is stored in the order p -> token (p >> 1) + 8 (p & 1). The contraction takes both operands in the same order for free
  where the contiguous axis is the contracted one (outgoing: k); for the incoming direction (the contiguous axis is the output index) ctr8's epilogue undoes it (staging row = natural row; the accumulator columns 2n / 2n + 1 become columns n / n + 8 of each 16-group). The bf16 chain is untouched.
* e4m3 MN-major operands (incoming direction) are legal for `kind::f8f6f4`: the same 128 B swizzle box [128 k rows][128 B], LBO = one box, 4096 B per K32 MMA -- no transposed front (the H100 needs one: wgmma e4m3 takes K-major only).
* **Any L:** ctr8 uses 3-D tensor maps ([plane][row][col]) so tiles clip at L (out-of-range rows / columns read zero and are not stored); L % 16 != 0 is padded on the host with masked tokens like the default (`_padded_inference`). The route declines lengths far from a multiple of 128 (tile quantisation: L400 D256 measured +5 % slower than bf16): `L / ceil128(L) >= 0.9`.

### Scales, calibration, guard (the H100 contract)
* First eager call of a module: a probe launch of the front with qinv = 1 gives the exact amax per group (a, b; per direction), `pow2_scale` (smallest 2^e with amax / 2^e <= 448, floor 2^-30, all-zero -> 1) x HEADROOM 16 (`MINIWORLD_TRIMUL_FP8_HEADROOM`); static afterwards, cached per module (key: data_ptr of `to_left.weight`, dropped by `weakref.finalize`).
* `GUARD`: `sync` (default; eager: the call reads the peak, > 448 -> scales x 2^ceil(log2(peak / 448 x 2)), refresh, redo <= 4 times, then the bf16 route), `lazy` (capturable; kernel atomically maxes the scaled peak into a per-device buffer), `off`. A capture before any calibration takes the bf16 route (`capture_fallbacks`).
* `overflow_peak()` (one host sync, once per forward outside graphs), `rescale_all(peak)` (in place; captured graphs follow; tested: stale scales 64x too small -> peak > 448 on replay -> `rescale_all` -> the SAME graph is accurate again).
* LN_in makes the planes invariant to the input scale and offset: input stress (x scale 1e-3 .. 40, offsets 0 / 25) never saturates (peak 17-23 of 448 at 16x headroom); saturation can only come from stale scales (weights changed after calibration) or heavy outlier channels.
* Switches (same names as H100, read via `fp8.env`): `MINIWORLD_TRIMUL_FP8` {calib, dynamic}, `_GUARD`, `_HEADROOM`, `_MINL` (replaces every gate, 0 = none). Counters: `b200_fp8.GUARD_STATS` (calls, retries, fallbacks, capture_fallbacks, calibrations). Registration for `fp8.py`: `work/n13-fp8-b/FP8_REGISTER.md`.

### Served
bf16 master modules, inference (no grad, no live dropout), TriangleMultiplication outgoing / incoming, Bidirectional, RF3 (D <= 128, one sample), D 64 / 128 / 256 / 384 / 512 with the bf16 route's hidden widths, any L with `L / ceil128(L) >= 0.9` and L >= MIN_L (below the gate the bf16 route runs unchanged), B up to one default launch.
MIN_L: see the follow-up section (D64 out/in 512, D64 bidir 384, D128 512, D256 out/in 384 and bidir 256, D384 512, D512 128).

### Results (module level, CUDA-graph replay, B = 1, bf16 master, `bench_mod.py`; interleaved A/B in ONE process, 3 rounds)
`gap` = the first replay after 0.15 s idle (the gap measurer's regime); `sus` = back-to-back replays for 1.5 s per arm (clock / power capped), power from nvidia-smi every 0.2 s; energy ratio = (sus time x mean power) FP8 / bf16.
| D | dir | L | gap bf16 / FP8 | gap ratio | sus bf16 / FP8 | sus time ratio | power bf16 / FP8 | sus energy ratio |
|---|---|---|---|---|---|---|---|---|
| D128 | out | 128 | 18.0 / 17.7 us | 0.981 | 17.2 / 16.9 us | 0.984 | 500 / 468 W | 0.919 |
| D128 | out | 256 | 41.4 / 41.0 us | 0.990 | 39.2 / 38.2 us | 0.975 | 717 / 652 W | 0.886 |
| D128 | out | 512 | 133.3 / 129.8 us | 0.974 | 144.4 / 133.6 us | 0.925 | 819 / 801 W | 0.905 |
| D128 | out | 1024 | 546.4 / 512.9 us | 0.939 | 630.5 / 576.6 us | 0.915 | 846 / 814 W | 0.880 |
| D128 | out | 2048 | 2787.3 / 2375.5 us | 0.852 | 3209.0 / 2715.0 us | 0.846 | 880 / 850 W | 0.818 |
| D128 | bidir | 128 | 25.0 / 23.8 us | 0.950 | 21.8 / 21.9 us | 1.006 | 578 / 551 W | 0.959 |
| D128 | bidir | 256 | 61.6 / 60.5 us | 0.982 | 61.3 / 57.9 us | 0.945 | 824 / 808 W | 0.927 |
| D128 | bidir | 512 | 213.6 / 200.3 us | 0.938 | 243.7 / 223.7 us | 0.918 | 824 / 842 W | 0.939 |
| D128 | bidir | 1024 | 1010.3 / 927.4 us | 0.918 | 1168.9 / 1042.0 us | 0.891 | 854 / 852 W | 0.889 |
| D128 | bidir | 2048 | 5222.2 / 4396.7 us | 0.842 | 6085.9 / 5043.5 us | 0.829 | 933 / 897 W | 0.796 |
| D64 | out | 128 | 12.5 / 14.4 us | 1.153 | 11.7 / 13.6 us | 1.163 | 417 / 371 W | 1.035 |
| D64 | out | 256 | 25.0 / 24.7 us | 0.989 | 22.9 / 23.4 us | 1.022 | 549 / 534 W | 0.994 |
| D64 | out | 512 | 67.2 / 65.1 us | 0.968 | 66.5 / 62.3 us | 0.938 | 778 / 826 W | 0.995 |
| D64 | out | 1024 | 257.8 / 244.3 us | 0.948 | 298.6 / 268.5 us | 0.899 | 835 / 808 W | 0.871 |
| D64 | out | 2048 | 1342.1 / 1129.4 us | 0.842 | 1533.3 / 1289.3 us | 0.841 | 857 / 841 W | 0.825 |
| D64 | bidir | 128 | 15.0 / 15.7 us | 1.044 | 14.4 / 14.9 us | 1.036 | 457 / 426 W | 0.966 |
| D64 | bidir | 256 | 32.8 / 32.7 us | 0.999 | 30.8 / 30.2 us | 0.981 | 686 / 604 W | 0.865 |
| D64 | bidir | 512 | 108.7 / 98.0 us | 0.902 | 112.1 / 98.5 us | 0.879 | 814 / 819 W | 0.885 |
| D64 | bidir | 1024 | 431.9 / 395.6 us | 0.916 | 500.3 / 446.0 us | 0.892 | 821 / 834 W | 0.905 |
| D64 | bidir | 2048 | 2383.6 / 1947.7 us | 0.817 | 2746.2 / 2217.0 us | 0.807 | 858 / 864 W | 0.812 |
| D256 | out | 128 | 29.7 / 29.5 us | 0.995 | 27.2 / 27.0 us | 0.993 | 687 / 679 W | 0.982 |
| D256 | out | 256 | 82.8 / 82.4 us | 0.996 | 88.2 / 85.6 us | 0.970 | 823 / 812 W | 0.958 |
| D256 | out | 512 | 299.4 / 282.2 us | 0.943 | 344.5 / 323.2 us | 0.938 | 831 / 841 W | 0.949 |
| D256 | out | 1024 | 1285.2 / 1177.2 us | 0.916 | 1497.6 / 1357.9 us | 0.907 | 856 / 839 W | 0.888 |
| D256 | out | 2048 | 6373.0 / 5466.4 us | 0.858 | 7455.9 / 6346.0 us | 0.851 | 921 / 898 W | 0.830 |
| D256 | bidir | 128 | 42.1 / 41.7 us | 0.991 | 39.3 / 38.3 us | 0.974 | 802 / 766 W | 0.930 |
| D256 | bidir | 256 | 140.7 / 129.3 us | 0.919 | 152.3 / 139.1 us | 0.913 | 845 / 800 W | 0.865 |
| D256 | bidir | 512 | 491.7 / 453.0 us | 0.921 | 569.1 / 522.6 us | 0.918 | 814 / 829 W | 0.936 |
| D256 | bidir | 1024 | 2229.8 / 2003.9 us | 0.899 | 2613.3 / 2321.9 us | 0.889 | 860 / 873 W | 0.902 |
| D256 | bidir | 2048 | 11835.4 / 9770.8 us | 0.826 | 13602.0 / 11322.5 us | 0.832 | 978 / 954 W | 0.813 |
| D512 | out | 128 | 69.6 / 67.0 us | 0.962 | 68.9 / 64.8 us | 0.940 | 797 / 813 W | 0.958 |
| D512 | out | 256 | 241.1 / 227.5 us | 0.944 | 268.1 / 251.3 us | 0.937 | 808 / 832 W | 0.965 |
| D512 | out | 512 | 895.3 / 841.9 us | 0.940 | 1028.7 / 970.0 us | 0.943 | 857 / 839 W | 0.923 |
| D512 | out | 1024 | 3947.3 / 3563.6 us | 0.903 | 4547.3 / 4109.4 us | 0.904 | 902 / 889 W | 0.891 |
| D512 | out | 2048 | 19642.7 / 16727.1 us | 0.852 | 21334.4 / 18421.6 us | 0.863 | 996 / 998 W | 0.865 |

More cells (same protocol, SUS 2 s): 
| case | gap bf16 / FP8 | gap ratio | sus time | sus energy |
|---|---|---|---|---|
| D384 out L512 | 550.9 / 519.1 us | 0.942 | 0.940 | 0.921 |
| D384 bidir L512 | 951.1 / 890.7 us | 0.937 | 0.928 | 0.940 |
| D384 out L1024 | 2403 / 2212 us | 0.921 | 0.911 | 0.901 |
| D128 out L3072 | 7717 / 6807 us | 0.882 | 0.818 | 0.801 |
| D256 in L3072 | 18441 / 14718 us | 0.798 | 0.824 | 0.824 |
| D128 out L1000 (padded 1008) | 1182 / 1179 us | 0.998 | 0.998 | 0.923 |
| D128 out L300 (padded 304), D256 out L400 | 116 / 121 us, 206 / 221 us | 1.036, 1.073 | 1.046, 1.011 | 0.965, 1.003 | (declined by the 0.9 tile-efficiency gate)

Reading: the gain is 5-10 % from L512, 10-18 % at L1024-3072 (time and sustained energy); at L128-256 level (D128 / D256 / D512 gain 1-6 %, D64 one direction loses 2-16 % -> gate 512).
It is smaller than the H100's (-15 .. -25 %) because (a) the B200 bf16 contraction is already at 1.3-1.45 PFLOP/s sustained, (b) the power cap: FP8 runs ~2.0 PFLOP/s in ctr8 at the same ~900-950 W, (c) the e4m3 front epilogue costs 4-12 % over k1w.
Per-kernel device time (profiler, graph-free eager, us; D128 out L1024): k1w 164 -> k1w8 183, contraction (cuBLAS nvjet) 175 -> ctr8 144, k3g 209 -> 196 (data-dependent power); D256 out L1024: k1w 444 -> 459, 379 -> 308, k3w 454 -> 430. D128 L512: 48.5 -> 56.3, 34.4 -> 28.4, 53.4 -> 53.6.
Contraction alone (microbenchmark, 20-launch graphs, burst, L2-warm): ctr8 reaches 1.7-2.0 PFLOP/s at L >= 1024 against cuBLAS bf16 1.35-1.45 (D128 L1024 138 vs 190 us, L2048 1089 vs 1631 us), 1.2-1.3 PF at L512 (26.4 vs 33.6 us), L384 15.9 vs 19.4 us, L256 level, L128 4.1 vs 3.3 us (cuBLAS) / 4.6 us (ctr bf16). cfg KB = 2 (256 k a stage) is slower at every L: default KB 1, BN 256 (BN 128 when 256 would waste a tile column).

### Accuracy against the float64 module (random N(0, 1/fan_in) weights, 5 masked tokens; update error = rms(y - y64) / rms(y64 - x))
Benign inputs (x ~ N(0, 1)); FP8 = static calibration on the same call, HEADROOM 16:
| case | bf16 default | FP8 | ratio | peak scaled plane value (of 448) |
|---|---|---|---|---|
| D128 out L256 / in L256 / bidir L256 | 0.0058 / 0.0058 / 0.0054 | 0.0379 / 0.0379 / 0.0378 | 6.5 / 6.5 / 7.0 | 19.1 / 16.7 / 20.1 |
| D64 out L256 / bidir L256 | 0.0059 / 0.0060 | 0.0381 / 0.0379 | 6.5 / 6.4 | 17.3 / 17.7 |
| D256 out L256 / bidir L256 | 0.0059 / 0.0060 | 0.0379 / 0.0378 | 6.5 / 6.4 | 18.5 / 18.0 |
| D384 out L256, D512 out L256 | 0.0058, 0.0064 | 0.0379, 0.0379 | 6.5, 5.9 | 19.1, 19.1 |
| D128 out L384 / L512 | 0.0058 / 0.0066 | 0.0379 / 0.0380 | 6.5 / 5.8 | 19.1 |
(the H100 e4m3 planes: 3.7-3.8 % update error, bf16 0.6 %: the error is the e4m3 storage of the planes, the same on both cards: tcgen05's fp32 accumulation adds nothing visible.) It fails the engine's 8e-3 module form by design (FP8 is outside the judge, as on H100).
Stress (x scale {1e-3, 1, 40} x mean offset {0, 25}, FP8 calibrated on the stressed input; `work/n13-fp8-b/stress_table.txt`, all widths, both directions, bidirectional):
x scale 1e-3, offset 0: bf16 0.0045-0.0052, FP8 0.0377-0.0381 (ratio 7.3-8.4); x 1, offset 0: 0.0054-0.0064 / 0.0378-0.0382 (5.9-7.0); x 1, offset 25: 0.0475-0.0673 / 0.0604-0.0771 (1.15-1.27); x 40, offset 0: 0.087-0.122 / 0.094-0.128 (1.05-1.09); x 40, offset 25: 0.102-0.143 / 0.108-0.148 (1.03-1.07).
At large activations / offsets both routes are limited by the bf16 rounding of y itself (the update is small against y) so the ratio -> 1; x 1e-3 with offset 25 is degenerate (constant rows in bf16: the update is exactly zero, 0 / 0 in the metric; all outputs finite). No NaN, no inf, no saturation (peak 17-23).

### Verification
* FP8 off (POLICY None) vs the pre-change tree `n13-base`, separate processes, 24 cases (widths 64-512 both modules, L 128-1024 incl. dropout p 0.25 eval): eager **24 / 24 outputs bitwise equal**, CUDA-graph replay **24 / 24 equal** (`work/n13-fp8-b/bit_*.pt`, `bitg_*.pt`, xchk / xchk_g / xcmp of `n2-trimul/_tools`). The default extension `trimul_b200_v6` is untouched (its sources are byte-identical); the hook in `integrations/trimul_b200.forward` is skipped when `b200_fp8.POLICY` is None.
* `tests/gpu/test_trimul_fp8_b200.py`: 38 passed (accuracy vs float64 and vs the bf16 route for D64-D512 both directions and bidirectional at L256 / 384, hidden = 2 D forms, padded lengths 1000 / 1008, lengths far from a tile multiple keep the default; stress inputs; saturation + `rescale_all` through a captured graph; graph == eager and capture-before-calibration fallback; default untouched and gates; RF3 length factor with one calibration, also through a captured graph after the mask changed; B = 3 equal to per-sample calls). `tests/gpu/test_trimul_b200.py`: 179 passed. `tests/cpu -k "fp8 or prebuild"`: 60 passed.
* `_prebuild --arch sm100 --check` (full output `work/n13-fp8-b/prebuild_check.txt`, base `prebuild_check_base.txt`): rc 1 in both (the pre-existing "no compiled kernel matches" / "compiled only by other recipes" notes, identical); the plan goes 804 -> 805 artifacts (+ `trimul_b200_fp8_v1` ext), call sites 221 -> 222, native sources 530 -> 533.

### Rejected / measured no (B200, this session)
* Separate quantise kernel (bf16 planes -> e4m3): extra 3 B / plane element of traffic = the whole contraction saving at L1024 (planes pass 0.19 ms vs 0.2 ms saved) -- the quantisation must be in the front's epilogue.
* First front epilogue (lane exchange + 16-bit byte stores, 32-token boxes): k1w8 +35 % (L1024) / +49 % (L2048); 64-byte-row boxes + the stmatrix shuffle: +4.5-12 %.
* ctr8 with 256 k a stage (KB 2): 10-30 % slower than KB 1 at L >= 512.
* Atomics / amax of the guard: ablation (atomics off, amax off) moved k1w8 by 0-3 % (noise); kept, flushed once per guard group instead of per chunk.
* Not built: e4m3 weights in k1w / k3w (projection operands: 0.8-3 % update error each, H100 accuracy map), e4m3 t, per-channel scales (pow2 rounding: no accuracy gain), training.

### Open items
* k1w8 is 4-18 % slower than k1w (D256 the most): the epilogue is the cost centre; the next levers are fewer cvt / amax instructions (a packed fp16 max) and the weight-stream-bound D256 front.
* Small L (<= 256): FP8 is level (the chain's fixed latencies dominate); the one-launch L128 kernels and the M64 fronts have no e4m3 twin. D64 one direction is gated at L512.
* The gain at L512-1024 (7-10 %) is bounded by the front (k1w8) and K3, neither of which gets faster with e4m3 operands; a gate-projection-only e4m3 front (H100 emulation: +0.3-0.5 % error) is the next candidate but needs two MMA chains per weight block.
* `fp8.py` registration (agent A): `FP8_REGISTER.md`. B > one default launch (chunked batches), fp32 and training stay on the bf16 route.

## 2026-11-19 (2) -- follow-up: front epilogue cost, preset path, gates (n13-fp8-b)
* **Count first**: `cuobjdump -res-usage` of `k1w8` D128 P256: 126 registers, no stack / local (no spills); SASS of the D128 P256 kernel 3.8k instructions against 2.2k of k1w (IMAD 593 vs 292, LOP3 364 vs 168, FMUL2 160 vs 96, F2FP 96 vs 64, FMNMX3 32: the guard amax is already one 3-input max per two values; the cvt is native `F2FP` e4m3x2; no SHFL / STSM beyond the baseline; files `work/n13-fp8-b/sass/`).
* **Ablations** (7 extension variants in ONE process, interleaved, min of 8 rounds; k1w8 / k1w, D128 P256 L1024 / D256 P512 L1024; a colleague's job shared GPU 1, so absolute times were 2x the quiet ones): full epilogue 1.025 / 1.005; guard flush off 1.009 / 0.943; amax off 1.010 / 0.944; qinv multiply off 0.977 / 0.999; cvt off 1.033 / 1.004; stmatrix off 0.993 / 0.930; scale as packed FMUL2 1.015 / 1.006. Reading: with the lanes' token shuffle the e4m3 front is within +0.5-2.5 % of k1w in isolation (the +4-18 % of the first measurement was clock / foreign-load noise); the only kept change is the packed FMUL2 scale (`F8_ABL` 128, default). The qinv multiply could be folded into the packed weights (H100 way) for another ~1-4 % of the FRONT = ~1 % of the module: not done (it forces a weight copy per rescale).
* **Gate-projection-only e4m3 front (H100 emulation: +0.3-0.5 % update error on top of the planes, 4.25 % vs 3.72 %)**: NOT implemented. It needs LN_in(z) quantised to e4m3 plus W_ag / W_bg e4m3 in a second MMA chain with a second xn image in TMEM per weight block (k1w's TMEM columns are full at D128: 3 x 128 accumulators + xn) -- a front rewrite, and the front is only 25-35 % of the module with the MMA a part of it; the upper bound (half of the gate chain at e4m3 rate) is at most ~3-5 % of the module at D256+ and ~1-2 % at D128 (H100's own estimate: -7 % of the module at D256 L2048). Below the cost of the rewrite for what is left of the session.
* **Gates re-measured** (graph replay, same protocol, GPU shared with a foreign job: +-3 %): D128 out L384 1.004 gap / 0.968 energy, D128 bidir L384 1.025 / 0.938, D256 out L384 0.969 / 0.975, D64 out L384 0.986 / 0.948, D64 out L512 0.980 / 0.861, D64 bidir L256 1.025 / 0.870, D64 bidir L384 0.976 / 0.939. New MIN_L = served only where the gap ratio is <= 0.99: D64 out / in 512, D64 bidir 384, D128 every direction 512, D256 out / in 384, D256 bidir 256, D384 512, D512 128 (`b200_fp8.MIN_L`).
* **`MINIWORLD_FP8=safe` preset path verified end to end** with agent A's fp8.py (copy of the tree n13-fp8-bx = n13-fp8-b + A's `fp8.py` / `fp8_sm100a.py`, `work/n13-fp8-b/dev/py/t_preset.py`): `fp8.arch()` = sm100, `status()["trimul"]` = calib, `_modules()[1]` = `b200_fp8`; eager calls take the route (`route_stats()["trimul"]` calls 2, calibrations 1, fallbacks 0); a captured graph takes the route (`capture_fallbacks` 0, graph == eager bitwise); `check()` False when clean, True after the scales were made 64x too small (the replay saturated), False after the in-place rescale; `fp8.disable()` returns to the bf16 route.


# Agent C: token DiT / atom DiT FP8 (B200)

# FP8 (e4m3) on B200 (sm_100a): token DiT and atom DiT inference

Opt-in, OFF by default; the default route is bit-identical to the pre-change tree (section "Verification"). Product code: `kernels/conditioned_transition/cuda/token_dit_fp8_b200.py`,
`gemm_swiglu_f8_sm100.cu`, `token_dit_fp8_bind_sm100.cu` (+ the arch-neutral 3.0.3 `token_dit_fp8_rows.cu`, unchanged); call sites in `token_dit_fp8.py` (`ext()`, `active(..., b200=True)`) and
`conditioned_transition/cuda/__init__.py` (`f8` selection, the `and not b200` guard removed). Agent C, 2026-11-19, engine 3.0.3 + the B200 kernel set (branch b200/3.0.3).

## 2026-11-19 -- token DiT: e4m3 expand (+ SwiGLU) on tcgen05; atom DiT: measured, not built

### H100 design that was reused unchanged (digest: `$W/b2511/work/n13-fp8-c/h100_digest.md`)
Quantisation recipe of 3.0.3: activations get a DYNAMIC per-row scale (row max / 448) made by the row pass that produces the row (the quantising AdaLN passes `resgate_adaln_cf_q8`,
`adaln_in_rows_cf_par_q8` for Chai-1's identity block: the H100 `token_dit_fp8_rows.cu`, compiled for sm_100a as it is), weights a static per-output-channel scale (max / 448, once per weight pack);
no calibration, no overflow guard, no `check()` hook (a row's own maximum maps to 448: nothing saturates whatever a diffusion step's noise). Forms: default block and Chai-1's identity /
parallel block; `qkvg` never for the identity AdaLN. `safe` = the expand placement only. Counters: `route_stats()` keys unchanged.

### What was built (B200 specific)
`gemm_swiglu_f8_sm100.cu` = `gemm_swiglu2_sm100.cu` (2-CTA tcgen05, (tile pair, 128-unit chunk) items, TMEM double buffer, h staged for TMA stores) with e4m3 operands:
* X (e4m3 [M, 768], row scale sx[M]) and Wa / Wb (e4m3 [H, 768], channel scales) stream in 128-element K-blocks (one 128 B swizzle row = the same 32 KB per stage as bf16): NKB = 6, four
  `tcgen05.mma.cta_group::2.kind::f8f6f4` (M 256, N 256, K 32) steps per block, fp32 accumulators in TMEM (the unscaled e4m3 product).
* the epilogue applies the scales in fp32 before silu: `a = acc_a sx[row] swa[unit]`, `b = acc_b sx[row] swb[unit]`, `h = bf16(a sigmoid(a) b)` (TMEM lane = row, so the row scale is one
  register; the unit scales are float2 loads from L1). The sigmoid is `tanh.approx` (`SIG_TANH`, -7 % kernel time, error 2^-11 against 5 % of e4m3 operand error).
* h stays bf16 (the squeeze GEMM is the bf16 one); the e4m3 weights are made once per weight pack (`_capture.lookup`, like every pack).
Gates (`token_dit_fp8.active(..., b200=True)`): bf16 only, D 768, M = A B L a multiple of 128 and >= `MIN_ROWS_B200["expand"]` = 1024; served blocks: the default block (Protenix / MiniWorld /
OpenDDE ...) and Chai-1's identity block; NOT served: RoseTTAFold3 (k-q LN parallel block), Boltz-2 (up gate: no 3-operand kernel), fp32, training.
Chai-1 (parallel schedule): the transition runs on a side stream beside the attention core; a persistent expand grid over all 148 SMs waits for the core's CTAs (Chai-1 L512 A5:
1.044 with 148 CTAs, 0.964 with 96, 1.001 with 32): the grid is capped at `B200_PAR_CTAS` = 96 there.
Switches: `MINIWORLD_TOKEN_DIT_FP8=expand` (module `PLACES`), `MINIWORLD_TOKEN_DIT_FP8_MIN_ROWS`, `MINIWORLD_TOKEN_DIT_FP8_B200` (placements the B200 serves, default `expand`),
`MINIWORLD_TOKEN_DIT_FP8_B200_DEFS` (extra -D of the cubin, default `SIG_TANH`), `MINIWORLD_TOKEN_DIT_FP8_B200_CTAS` (grid override for experiments).
Prebuild: `_prebuild --arch sm100` lists `token_dit_fp8_rows_sm100` and `augattn_sm100/gemm_swiglu_f8_sm100` for width-768 bf16 inference token rows.

### Kernel alone (graph, 50 back-to-back, median of 7; bf16 route = cuBLAS + `swiglu_rows` below 3840 rows, the sm_100a bf16 SwiGLU GEMM from there)
| M | bf16 route us | bf16 SwiGLU kernel us | e4m3 kernel us | e4m3 / route |
|---|---|---|---|---|
| 640 | 7.7 | 9.3 | 8.1 | 1.05 |
| 1280 | 10.5 | 9.5 | 8.3 | 0.79 |
| 1920 | 12.1 | 13.1 | 10.3 | 0.85 |
| 2560 | 16.1 | 13.3 | 10.5 | 0.65 |
| 3200 | 17.6 | 16.7 | 12.3 | 0.70 |
| 3840 | 16.7 | 16.7 | 12.6 | 0.76 |
| 5120 | 21.0 | 21.0 | 15.0 | 0.71 |
The e4m3 kernel is 21-29 % faster than the bf16 SwiGLU kernel but reaches only 1.6 PF/s of e4m3 at M = 5120 (epilogue / ring bound, not tensor bound); in a block it is 12 us of ~80.

### Placements measured and rejected on B200 (cuBLAS e4m3 `_scaled_mm` row-wise, bf16 out, graph timing; bf16 / e4m3 mm / the separate quantising pass, us)
| placement (K x N) | M 1280 | M 1920 | M 2560 | M 3840 | M 5120 |
|---|---|---|---|---|---|
| qkvg 768 x 3072 | 7.2 / 6.2 / +1.9 | 7.8 / 8.1 / +2.4 | 10.8 / 10.0 / +2.8 | 13.2 / 11.7 / +3.5 | 18.4 / 15.0 / +4.2 |
| out 768 x 768 | 4.1 / 4.0 / +1.9 | 4.5 / 4.7 / +2.4 | 4.9 / 4.9 / +2.8 | 5.5 / 5.9 / +3.5 | 7.1 / 6.8 / +4.2 |
| squeeze 1536 x 768 | 5.4 / 5.0 / +2.4 | 5.8 / 5.8 / +3.0 | 6.7 / 5.9 / +3.6 | 7.8 / 7.1 / +4.8 | 10.7 / 8.4 / +5.9 |
The e4m3 product saves at most 0-3 us, the quantising pass (qkvg: inside the AdaLN pass at +0.4 us per H100 data, but out / squeeze need their own) costs as much: no placement beyond expand
(as on H100: `out` / `squeeze` lost, `qkvg` lost below 2560 rows). Block-scaled MXFP8 (`_scaled_mm` with e8m0 block-32 scales, 128x4 swizzled layout; torch 2.13) is faster:
squeeze 5.9 -> 4.9 (M 1920), qkvg 7.7 -> 6.5, out 4.5 -> 3.8 us; an expand epilogue could write h as MXFP8 (32 units per thread = one block). Upper bound of the whole set ~3-4 us of an
80 us block (4 %) against another 3.7 % squeeze error and a swizzled-scale epilogue: not built (open item).

### Block results (B200 GPU 2, `dev.sh -x`, graph replay of one call under `static_weights()` + `static_inputs()` = the sampler's graphs; `_tools/blk2.py`)
Master weights (fp32 params, bf16 activations), A = 5 samples (A = 1 where marked), shared conditioning, 2 blocks per call (n = 2); e4m3 expand against the default in the SAME process, 9 rounds
interleaved with a 0.15 s idle gap before every timed 20-replay burst, order reversed every round, median; "ctrl" = a third capture of the DEFAULT timed the same way (address / order
effect of the A/B: +-4 % on single cells, so single ratios below ~3 % are noise); every row gate removed in the harness (`MIN_ROWS = 0`). `e4m3 / def` and `e4m3 / ctrl`. Raw rows: `$W/b2511/work/n13-fp8-c/final2/*.jsonl`.

| block (A:L) | M = A L | default us (2 blocks) | e4m3 us | e4m3 / def | ctrl / def | e4m3 / ctrl |
|---|---|---|---|---|---|---|
| Protenix 5:128 | 640 | 47.4 | 49.5 | 1.045 | 1.001 | 1.043 |
| Protenix 1:768 | 768 | 60.8 | 61.4 | 1.009 | 1.017 | 0.993 |
| Protenix 1:1024 | 1024 | 74.8 | 71.6 | 0.957 | 1.034 | 0.926 |
| Protenix 5:256 | 1280 | 67.4 | 64.0 | 0.950 | 1.040 | 0.914 |
| Protenix 1:1536 | 1536 | 98.9 | 96.4 | 0.975 | 1.003 | 0.973 |
| Protenix 5:384 | 1920 | 83.8 | 81.8 | 0.976 | 1.008 | 0.969 |
| Protenix 1:2048 | 2048 | 123.2 | 118.9 | 0.965 | 0.987 | 0.979 |
| Protenix 5:512 | 2560 | 116.7 | 109.6 | 0.939 | 0.984 | 0.955 |
| Protenix 5:768 | 3840 | 145.3 | 140.7 | 0.969 | 0.992 | 0.976 |
| Protenix 5:1024 | 5120 | 200.4 | 196.9 | 0.982 | 1.003 | 0.979 |
| Protenix 5:1536 | 7680 | 295.9 | 291.2 | 0.984 | 1.004 | 0.980 |
| MiniWorld 5:128 | 640 | 49.7 | 51.6 | 1.037 | 1.020 | 1.017 |
| MiniWorld 5:384 | 1920 | 88.7 | 87.6 | 0.987 | 1.005 | 0.983 |
| MiniWorld 5:640 | 3200 | 136.6 | 132.4 | 0.969 | 1.014 | 0.956 |
| MiniWorld 5:1024 | 5120 | 210.0 | 204.3 | 0.973 | 1.007 | 0.966 |
| Chai-1 5:128 | 640 | 38.1 | 37.2 | 0.974 | 1.014 | 0.961 |
| Chai-1 5:256 | 1280 | 56.9 | 53.4 | 0.940 | 0.994 | 0.945 |
| Chai-1 5:384 | 1920 | 82.2 | 71.9 | 0.874 | 0.964 | 0.907 |
| Chai-1 5:512 | 2560 | 95.2 | 90.7 | 0.952 | 1.009 | 0.944 |
| Chai-1 5:1024 | 5120 | 181.0 | 172.1 | 0.951 | 0.998 | 0.953 |
| Chai-1 5:1536 | 7680 | 277.9 | 265.9 | 0.957 | 1.001 | 0.956 |
| RoseTTAFold3 5:128 / 384 / 640 | 640 / 1920 / 3200 | 41.5 / 87.7 / 126.0 | declined (route is the default) | 1.043 / 0.969 / 0.982 | 1.042 / 0.988 / 0.990 | 1.001 / 0.981 / 0.992 |
| Boltz-2 | any | declined (up gate not built) | | | | |
The RoseTTAFold3 rows ARE the default route (the route declines it): their spread (0.97-1.05) is the noise floor of this A/B. The parallel block and the plain block at M = 640 (A5 L128: 1.043 over
the control, reproduced in three runs) lose; the gate `MIN_ROWS_B200["expand"]` = 1024 keeps them on bf16. Block-level result: plain blocks -2 .. -5 % from 1280 rows, Chai-1 -5 .. -9 %.
Why so little: the e4m3 expand kernel is 12 of ~80 us of a block; a quantising AdaLN pass is +0.4 us (the extra row-max reduction), the saving is the expand kernel (-2 us) plus a unified one-launch
expand against cuBLAS + row pass below 3840 rows (-2 us).

### Sustained time and energy per block (power cap ~1000 W; 8 s back-to-back replays per mode, nvidia-smi 0.2 s, two passes def / e4m3 / def / e4m3, mean of the pairs; `ENERGY=1`)
| block (A:L) | M | def us / W | e4m3 us / W | time | energy per call |
|---|---|---|---|---|---|
| Protenix 5:256 | 1280 | 64.7 / 709 | 62.3 / 674 | 0.963 | 0.917 |
| Protenix 5:512 | 2560 | 113.2 / 831 | 107.5 / 798 | 0.950 | 0.911 |
| Protenix 5:1536 | 7680 | 308.7 / 975 | 299.7 / 972 | 0.971 | 0.967 |
| Chai-1 5:256 | 1280 | 55.5 / 775 | 52.0 / 713 | 0.937 | 0.860 |
| Chai-1 5:384 | 1920 | 78.9 / 845 | 72.4 / 810 | 0.918 | 0.880 |
| Chai-1 5:512 (148 CTAs, before the grid cap) | 2560 | 92.6 / 933 | 96.3 / 835 | 1.040 | 0.929 |
| Chai-1 5:1024 | 5120 | 181.9 / 972 | 176.5 / 975 | 0.970 | 0.974 |
e4m3 saves energy before it saves time (the tensor pipe runs at half the cycles for the same work): 3-14 % less energy per block at 1280-2560 rows where the block is not yet at the cap;
at the cap (7680+ rows) time and energy agree at -3 %. (The Chai-1 5:512 row is the 148-CTA grid before the `B200_PAR_CTAS` cap: +4 % time, -7 % energy; the cap turned it into -5 % time.)

### Accuracy (random weights as the H100 protocol; fp64 PyTorch gold; update error = rms of (y - y64) over rms of (y64 - x); judge = max|y - y64| / max|y64|; `tests/gpu/test_token_dit_fp8_b200.py`)
| case | bf16 default route: rms / judge | e4m3 expand: rms / judge | ratio (rms) |
|---|---|---|---|
| Protenix 5:384, input x 1 | 0.0070 / 0.0050 | 0.0535 / 0.0306 | 7.6 |
| Protenix, input x 0.05 | 0.0072 / 0.0074 | 0.0536 / 0.0490 | 7.5 |
| Protenix, input x 20 | 0.109 / 0.0047 | 0.121 / 0.0048 | 1.1 |
| Chai-1 5:384, input x 1 | 0.0067 / 0.0061 | 0.0495 / 0.0453 | 7.4 |
| Chai-1, input x 0.05 | 0.0069 / 0.0084 | 0.0496 / 0.0558 | 7.2 |
| Chai-1, input x 20 | 0.042 / 0.0047 | 0.064 / 0.0061 | 1.5 |
Every block of the sweep above has e4m3 update rms 4.7-5.4 % and judge 2.5-5.1e-2 (table right-hand columns of `final2`), i.e. the H100 route's figures (5.4 % / 3-3.4e-2; Chai-1 4.7-6.4e-2): the 3.0.3 acceptance
is not the judge (the expand fails the 8e-3 judge 3-4x on H100 too, "FP8 outside the judge"), it is the end-level protocol of the 3.0.3 docs (not re-run here: same quantisation recipe, same
operand error). Saturation: impossible by construction (per-row scale = row max / 448; the stress inputs x 0.05 and x 20 stay finite and at the same relative error; graph replays are deterministic and equal
the eager route within 2 % of max |y|: test). The only B200-specific error source is the tanh.approx sigmoid (2^-11 relative; below the e4m3 rounding of a and b by 2 orders of magnitude).

### Verification
* FP8 OFF, bit-identity of the default route: the merge agent's scripts `$W/b2511/jobs/n12-bid/bid.py` (eager) and `bidg.py` (CUDA graph, replay twice), 398 measurer cases each (AttentionPairBias + five TokenDiT classes, bf16 / master /
  fp32, eval + train), sha1 of outputs and parameter gradients, tree n13-fp8-c against n13-base: **398 / 398 identical in both, 0 mismatches, 0 errors, 0 nondeterministic**
  (`$W/b2511/work/n13-fp8-c/bid/cmp.txt`, final tree). Non-sm100 CUDA sources byte-identical (diff list: only new files under `conditioned_transition/cuda/`), `backend_versions["cuda:b200_sm100"]` untouched.
* tests (B200): `tests/gpu/test_token_dit_fp8_b200_kernel.py` (the cubin against a torch reference on the same e4m3 operands at M 128 .. 5120), `tests/gpu/test_token_dit_fp8_b200.py` (accuracy next to the bf16 route at input x 1 / 0.05 / 20
  for Protenix and Chai-1, route on / off bit-equal below the gate, RoseTTAFold3 / Boltz-2 not served, graph == eager, route stats); the B200 suites trimul / triattn / capture_bits / transition_uninit / misc / local_dit / token_dit /
  atom_dit and `tests/cpu/test_fp8_switch.py`: 570 passed, 351 skipped (H100-only) (`$W/b2511/work/n13-fp8-c/tests/pytest.log`).
* `_prebuild --arch sm100 --check`: 804 -> 806 artifacts (739 cubin, 67 ext), 0 not built yet; the two new ones (`token_dit_fp8_rows_sm100`, `augattn_sm100/gemm_swiglu_f8_sm100`) are in the plan; rc = 1 only from the unchanged `*_sm90` / mpnn lines, and the
  line "not compiled: conditioned_transition/cuda/token_dit_fp8_rows.cu" is gone (full outputs: `$W/b2511/work/n13-fp8-c/prebuild/`).

### Rejected / not built (with data)
* `qkvg`, `out`, `squeeze` on B200 (cuBLAS e4m3 `_scaled_mm`): kernel table above; in the block (Protenix 5:384 / 768 / 1024 / 1536, e4m3 / default): all four placements 1.079 / 1.046 / 1.021 / 1.036; `qkvg + expand`
  1.007 / 0.986 / 0.983 / 0.983 against `expand` alone 0.976 / 0.969 / 0.982 / 0.984: no gain beyond the expand (and 1-3 % more error than expand alone: 5.4 % -> 6.6 % update rms with out / squeeze).
* RoseTTAFold3 `expand`: the parallel block with k-q LN needs a standalone quantising pass (`rows_q8(xt)`), which costs as much as the e4m3 expand saves (the transition runs beside the attention core on a side stream anyway); declined on B200.
* Boltz-2 (up gate): the bf16 B200 route is cuBLAS + `swiglu_rows` (unfused, [a | b | u] through HBM); a three-operand e4m3 tcgen05 kernel (64-unit chunks, N = 192 per pair step, B rows split a64 + b32 | b32 + u64) was NOT built (see open items); not served.
* A 148-CTA persistent expand grid in the parallel (Chai-1) schedule: serialises with the attention core (1.044); capped at 96 CTAs (0.964 / 0.965 at 96 / 64, 1.001 at 32).
* H100 transfers not taken: `ht = 64` tiling, wgmma n192 up kernel, the sm90 weight packs.

### Atom DiT: measured, no B200 route built
The 3.0.3 atom DiT route (`local_dit_h100.FP8_ROWS`: e4m3 `pre90_f8` / `post90_f8` row kernels) is an H100 wgmma kernel family; `MINIWORLD_ATOMDIT_FP8` stays unserved on B200 (`serves()` is H100 only).
What the B200 chain looks like (docs/b200/atom-dit.md, "Second pass" / "Third pass"): the bf16 inference stages (`qkvg_fwd1t / fwd3`, `attn_fwd`, `ffn_fwd1t / fwd3`, K = 128 / 256, d_atom 128, hidden 256) are row-thread and TMA-latency bound, not tensor bound
("9500 -> 5650 ticks per tile, all of it row-thread work now"; "a cold TMA load lands ~2.3 us after its issue"); the modulation GEMMs run per row inside the kernels from smem operands.
Measured bf16 block time (graph, B200 GPU 2, A = 5, master, shared conditioning, `_tools/blk.py` default column): AtomDiTProtenix / OpenFold3 / Boltz2 / Chai1 5 x 1024 atoms 28.4-31.4 us, 5 x 4096 atoms 49.5-53.8 us, 5 x 8192 atoms
83.7-90.6 us (dense AF3-class classes with pair bias; Protenix D5 variant 65 / 191 / 343 us). Tensor time of the e4m3-capable GEMMs at 1.3 PF/s bf16: ~0.45 MF per row (qkvg 131 k, Wo 33 k, a | b 131 k, squeeze 66 k, modulation
~100 k) -> 20480 rows 9 GF = 7 us of 52 us (13 %); 40960 rows 18 GF = 14 us of 89 us (16 %). Halving the MMA time bounds the gain at ~6-8 % even if the MMAs were serial with the row work, and they are overlapped on B200 (the kernels
hide them under the epilogue and the TMA); the H100 ablation, where the row kernels ARE serial `cp.async -> wgmma -> epilogue` chains, found 9-13 % and delivered 5-10 % at 5 % update error (post: judge 5e-2). The B200 e4m3 route would
additionally add a per-row amax + cast to the row-thread critical path of qkvg / FFN (the part that is binding) and re-derive the modulation / smem layouts of kernels that were tuned for tens of rounds. Not built: the measured bound does not justify the
loss of the "bit-identical bf16 cubins" and the accuracy cost; no atom DiT file of the tree changes.
(This is an analytic bound from measured block times and the documented stage structure, not an ablation of the B200 kernels.)

### Open items (exact)
1. Boltz-2 up-gate e4m3 expand (3-operand tcgen05 kernel); bound ~3 % of a block (unfused bf16 path today).
2. MXFP8 (block-32 e8m0 scales, `kind::mxf8f6f4` / cuBLAS block-scaled `_scaled_mm`): h written as MXFP8 by the expand epilogue (one thread = 32 units = one scale block) feeding a block-scaled squeeze; measured GEMM
   gain 0.3-1.3 us per GEMM (table above), not built; needs the 128x4 swizzled scale layout in the epilogue.
3. The e4m3 expand at 64-unit chunks (more items than SM pairs below 2560 rows: ~1 us of 10 us at M 1920); PDL for the expand kernel (the row passes of the 3.0.3 file never trigger their dependents early).
4. End-level (200-step diffusion, FoldForge RMSD) acceptance is NOT re-run on B200: the quantisation recipe / operand error are the 3.0.3 ones; only the module-level fp64 protocol was run.
5. Activation scales of rows whose max is denormal (|row max| < 1e-35) are not floored (the H100 row kernels are shared byte for byte); AdaLN outputs cannot be such rows.
6. `fp8.py` registration: see `$W/b2511/work/n13-fp8-c/FP8_REGISTER.md` (agent A).

### H100 token / atom DiT design digest (3.0.3 read, basis of the recipe)
#### H100 token / atom DiT FP8 design digest (read from 3.0.3 fp8_ref; basis of the B200 recipe)

#### Placements (token DiT, bf16 master inference, width 768, CondFold route)
| placement | e4m3 tensors | scale | where made | in `safe` | why / why not |
|---|---|---|---|---|---|
| expand (+SwiGLU) | xt (AdaLN rows) x [Wa;Wb] | act: dynamic per-row max/448; weight: static per-output-channel (max/448), per weight pack (`_capture.lookup`) | quantising AdaLN row pass (`adaln_in_rows_cf_q8` qkvg-less: `resgate_adaln_cf_q8`; Chai-1 `adaln_in_rows_cf_par_q8`; RF3 parallel: quant in the one LN pass) ; epilogue multiplies a,b by sx[row]*sw[unit] before silu(a)*b | YES | wins at every M (0.94-0.98 module); error 5.4 % rms (both a and b e4m3) but accepted: end-level RMSD inside 47/48 |
| qkvg | xa (adaln_in_cf_q8) x Wqkvg | per-row / per-channel, `torch._scaled_mm` rowwise bf16 out + bias | row pass | NO | loses below 2560 rows (1.006-1.019), wins 0.97-0.99 above; error 1-1.3 % at x1 but 3.5-3.9 % at small input (x0.05, attention update dominates); Chai-1 identity AdaLN scale unbounded: never qkvg |
| out | og (attention output) | needs own row pass (16 head CTAs write a row) | extra pass | NO | 1.02-1.05 loss (row pass 5-9 us outweighs) |
| squeeze | h (e4m3 hidden) | own row pass (h row spans 12 tiles) | extra pass | NO | loss; error 3.7 % |
Scale floor: weights `clamp_min(1e-30)` (train NaN fix); activation scale 1.0 if row amax == 0. No saturation by construction -> no calibration / guard / take_overflow.
Forms: plain (AF3/Protenix/MiniWorld), identity+parallel (Chai-1: sig=false, eps 0.1; expand/out/squeeze, never qkvg), parallel (RF3, kq-LN: expand only, quant in the LN pass), up (Boltz-2: [Wa64|Wb64|Wu64] n192 tiles, silu(a) b u, three channel scales).
Gates: MIN_ROWS_PER = {qkvg 2560, expand 640, out 2560, squeeze 2560}; MIN_ROWS_UP = 2560 (kernel -27 % but graph block +2.7 % at 1920: the quantising row pass costs a third reduction, row passes got SLOWER 12.6->14.0 us); M % 128 == 0; D == 768; bf16 only (fp32 declined).
Accuracy acceptance (H100): judge = max|y-y64|/max|y64| vs max(err of PyTorch bf16, 8e-3); expand FAILS the judge by 3-4x (3e-2) -> FP8 stays outside the judge, acceptance = end level (RMSD to bf16 <= max(1.1x bf16 pair max, 0.5 A)) + finite + route stats. Stress cases: x0.05 and x20 input, 3 masked keys, random weights LN 1+-0.1.
Speedups H100 (module graph, sustained, qkvg+expand / expand): 5:256 .941/.977 ; 5:512 .948/.967 ; 5:1024 .927/.949 ; 5:2048 .939/.973; A1 768 .992/.976. Boltz-2 expand: 1.056 at M640, .998 at 2560, .954 at 7680. Chai-1: whole gain 0.6 % of kernel time.
Traps: row pass slower than bf16 (3rd block reduction); out/squeeze lose; qkvg loses small M; Boltz up gate bistable end-level case 7y37; per-row scale is independent of step noise level but not of row shape.
#### Atom DiT (H100)
pre90_f8 (q/g/k/v proj K=128) + post90_f8 (Wo, [a,b], SwiGLU, squeeze Wd K=256) in the fused row kernels, A operand per-row power-of-two scale cast once to 16 regs, weights per 64-row-unit pow2 scales, epilogue acc*inv*inv exact; fp32 sum across the 4 hidden chunks (rescaling the wgmma accumulator gave wrong results). Attention (window, 32-wide heads) NOT e4m3: ablation <=1.5 %. Upper bound from ablation: row-kernel e4m3 products = 9-13 % of step (weight bytes only 3-6 %). Measured pre+post .895-.968 module; error post 5 % rms (judge 5e-2), pre 1.6 %. Gates: shared conditioning only (per-sample tables lost 1.09-1.12 -> 9 of 27 calls/run fallback), UP from 2048 atoms (Boltz), pre alone never wins (post carries it). Cond tables / pair bias are hoisted per fold: no FP8 there.
#### Transfers unchanged to B200
quantization recipe (per-row dynamic act max/448, per-channel weight max/448, floor), form matrix (plain/identity/parallel/up), the quantising AdaLN row kernels token_dit_fp8_rows.cu (CUDA, arch-neutral: only gencode changes), accuracy protocol + stress cases, gates as STARTING values (expand 640 -> re-measure; up 2560), safe = expand only, fp32 never, qkvg never for identity, route-stats keys (calls/declined/expand/...), shared-conditioning-only for atom rows.
#### Must be redone for tcgen05
MMA instruction (kind::f8f6f4, K=32/step, 128 elems per 128B swizzle block -> NKB=6 for D=768, same smem bytes per stage as bf16), accumulator in TMEM (epilogue scale in TMEM->reg path, per-lane row scale), 2-CTA pair tiling of gemm_swiglu2_sm100 (not wgmma n192), TMA box in elements, PDL, fold/side streams/pair32/vmn B200 chain (the e4m3 path must live beside _b200_swiglu and the parallel side-stream fork), thresholds (B200 bf16 expand kernel only from 3840 rows; below cuBLAS+rows), power-cap energy judgement.

## 2026-11-19 (second session, agent C) -- OuterProductMean and MSAPairWeightedAveraging: e4m3 on B200 (opt-in, default OFF)

Product code: `kernels/integrations/opm_fp8_b200.py` + `csrc/sm100/opm_fp8_b200.cu` (cast helper), `pwa_fp8_b200.py`; hooks in `opm_sm100.py` (`_forward`, `_product_and_counts`) and `pwa_sm100.py` (`_contract`, `_forward`);
two new template flags / entry points in the sm_100a sources `csrc/sm100/opm_sm100.cu` (`opm_prologue_sm100<WT, F8>`, `opm_prologue_f8`) and `csrc/sm100/pwa_sm100.cu` (`ln_vg_body<WT, F8>`, `ln_vg_f8`): the F8 = false instantiations are the
default kernels (the default route stays bit-identical, see Verification). Switches are the H100 route's: `MINIWORLD_OPM_FP8=bound` (`opm_fp8.POLICY`), `MINIWORLD_PWA_FP8=1` (`pwa_fp8.ENABLED`), so `fp8.enable` already reaches them.

### Decision from the B200 blockers tables (which stage is tensor / HBM bound, where e4m3 can win)
Stage times, B200 GPU 2/3, graph replay, `torch.profiler`, default route (`_tools/prof3.py`):
| module / case | stage | us | share |
|---|---|---|---|
| OPM 64/128, L384 depth 2048 (485 us) | cuBLAS product `nvjet_tst_128x256_64x6` (K = S) | 371.7 | 77 % |
| | epilogue (O round trip, W_o 1024 -> 128) | 64.6 | 13 % |
| | prologue (LN + projections, writes A2 / BT) | 40.7 | 8 % |
| OPM 64/128, L256 depth 1024 (140 us) | product / epilogue / prologue | 84.0 / 34.7 / 17.0 | 60 / 25 / 12 % |
| PWA 8x8, L768 depth 1024 (208 us) | gate_out_w / bmm / ln_vg_w / pair_fwd | 73.2 / 52.6 / 42.6 / 39.7 | bmm 25 % |
| PWA 8x8, L1536 depth 1024 (518 us) | bmm 175.4 / gate_out 141.1 / pair_fwd 119.9 / ln_vg 81.8 | | bmm 34 % |
The OPM product is tensor bound (cuBLAS bf16 1.3-1.66 PF/s at the power cap) and 60-77 % of the call from depth 1024: the e4m3 product bound is -30 .. -40 % of the module. The PWA contraction is 16-34 % of the module (the rest is the
latency / memory bound gate pass, `ln_vg`, `pair_fwd`): bound ~-8 .. -17 % for the e4m3 product alone. Both above the 5 % bar: built. (Agent A's `mb_results.txt` was not needed; micro-benchmarks of this section: `_tools/bench_opm.py`, `bench_pwa.py`.)
cuBLAS e4m3 `_scaled_mm` against bf16 for the OPM product (K = S, M = N = 32 L; GEMM only, us): L128 / S1024 27.1 -> 16.2, L256 / S1024 87.4 -> 51.6, L384 / S2048 371.7 -> 195.9, L768 / S4096 2997 -> 1578: 0.52-0.60 at S >= 1024 (0.72-0.84 at S 512).

### OuterProductMean: design (H100 recipe unchanged where not instruction specific)
* e4m3 stage: the grouped product O[(i,c), (j,e)] = sum_s A2[(i,c), s] BT[(j,e), s] only (as H100). Policy `bound`: static power-of-two scales from the weights alone,
  |LN(x)_k| <= sqrt(d_msa - 1)|gamma_k| + |beta_k| (`opm_fp8.bound_scales`, reused): no input can saturate the cast, no calibration / guard / `check()` hook. Scale floor: `pow2_scale` of the H100 TriMul helper (2^-30).
* B200 specific: d_msa 64 / d_pair 128 (the narrow prologue): the prologue drain writes e4m3 A2 / BT itself (`opm_prologue_f8`: value x q, saturating `cvt.rn.satfinite.e4m3x2`, 64 B swizzled staging, u8 TMA maps: half the A2 / BT bytes,
  no cast pass; the fp32 accumulator is cast once: no double rounding). Other widths and the projection-bias variants: one cast launch (`opm_cast2`, bf16 A2 / BT -> e4m3, 6 TB/s) after the unchanged bf16 prologue.
  The product is `torch._scaled_mm` (cuBLASLt e4m3, fp32 accumulation, scalar dequantisation scales, bf16 O); count table (side stream) and epilogue are the default's.
* Gate (`opm_fp8_b200._gate`): padded S >= 1024 at every L, or S >= 512 from N 256 where the fused bf16 kernel does not serve; below it (and where the fused kernel serves: S <= 512, N >= 512) the bf16 route runs. Chai-1's grouped OPM stays bf16 (below).
* Not served: training, Chai-1 grouped OPM (declined, see "Rejected"), calibration policy (`bound` only).

### OuterProductMean results (B200 GPU 3, `dev.sh -x`, graph replay of one call under `static_weights()` + `static_inputs()`, master weights, one sample; `_tools/blk3.py`: 9 rounds interleaved, 0.15 s idle gap before every timed 10-replay burst, order reversed every round, median; "ctrl" = a third capture of the default timed the same way)
Module time e4m3 / default (e4m3 us); fp64-gold error as rms / judge (max |y - y64| / max |y64|) of the bf16 route and of the e4m3 route; sustained energy per call = (time x power) of the e4m3 route over the default's (8 s back-to-back, nvidia-smi 0.2 s, mean of two def / e4m3 passes):
| case (d_msa / d_pair, L, depth S) | default us | e4m3 / default | ctrl | gold bf16 rms / judge | gold e4m3 rms / judge | energy ratio |
|---|---|---|---|---|---|---|
| 64/128 L128 S1024 | 48.9 | 0.782 (38.2) | 1.03 | 0.0041 / 0.0047 | 0.0208 / 0.0127 | 0.74 |
| 64/128 L128 S2048 | 78.2 | 0.721 | 1.00 | 0.0040 / 0.0053 | 0.0161 / 0.0112 | 0.70 |
| 64/128 L256 S1024 | 138.1 | 0.755 | 1.01 | 0.0041 / 0.0044 | 0.0210 / 0.0129 | 0.73 |
| 64/128 L256 S2048 | 233.8 | 0.657 | 1.00 | 0.0040 / 0.0045 | 0.0163 / 0.0101 | 0.65 |
| 64/128 L384 S1024 | 284.6 | 0.708 | 1.00 | 0.0041 / 0.0046 | 0.0210 / 0.0137 | 0.71 |
| 64/128 L384 S2048 | 493.9 | 0.634 | 1.00 | 0.0040 / 0.0050 | 0.0163 / 0.0106 | 0.64 |
| 64/128 L768 S1024 / S2048 (sweep) | 1072 / 1863 | 0.693 / 0.620 | 1.00 | | | |
| 128/256 L256 S1024 / S2048 | 153.7 / 257.9 | 0.815 / 0.736 | 1.00 | 0.0044 / 0.0052 | 0.0270 / 0.0169 (S2048: 0.0222 / 0.0131) | |
| 128/256 L384 S1024 / S2048 | 312.0 / 532.0 | 0.762 / 0.698 | 1.00 | | | |
| 128/384 L256 S1024 / S2048 | 167.4 / 271.5 | 0.823 / 0.748 | 1.00 | 0.0044 / 0.0052 | 0.0271 / 0.0169 | |
| 128/384 L384 S1024 / S2048 | 350.2 / 571.4 | 0.783 / 0.720 | 1.00 | | | |
| 256/512 L256 S1024 / S2048 | 204.2 / 331.4 | 0.854 / 0.789 | 0.99 | 0.0045 / 0.0041 | 0.0258 / 0.0194 | |
| 256/512 L384 S1024 / S2048 | 405.9 / 660.7 | 0.815 / 0.758 | 1.00 | | | |
Short depths (64/128, e4m3 / default): S 512: L128 0.97 (ctrl 1.06: noise), L256 0.877, L384 0.872, L768 1.00 (the fused bf16 kernel serves: unchanged); S 1024 at L128: 0.757 (the gate takes every L at S >= 1024).
The first version (separate cast pass) gave 0.80 / 0.71 / 0.74 / 0.67 at L256 S1024 / L256 S2048 / L384 S1024 / L384 S2048; the fused prologue (e4m3 written by the drain, no cast launch) takes another 4-6 points
(L384 S2048: product 372 -> 188 us, cast 25 us gone, the prologue writes half the A2 / BT bytes). Energy tracks time at the cap (both routes ~965-975 W): the e4m3 route's energy per call is its time ratio.
Accuracy: e4m3 update error = one e4m3 rounding of A2 / BT (2.7 % rms each) averaged by the depth: judge 1.0-2.0 % against 0.4-0.5 % for the bf16 route (tolerance 8e-3: 1.3-2.4x it; the H100 route: 1.03-1.72x at depth 1024-4096, inside at depth >= 8192);
update rms 1.6-2.7 % (ratio to the bf16 route 4-6x). Large-weight stress (LN weights x 8, biases x 4; `test_large_weight_stress`): judge 1.3 % (bf16 0.46 %), finite: the bound policy's scales follow the weights. No saturation by construction.
The e4m3 judge error exceeds the 8e-3 judge, like every FP8 route ("FP8 outside the judge"); the end-level check (RMSD) was NOT run on B200.

### MSAPairWeightedAveraging: design
* e4m3 stage: the key contraction o[h] = w[h] . v[h] (K = j = N) only. w (a probability <= 1): fixed scale 2^8 (cannot saturate); v: static power-of-two scale from the weights alone (`bound` policy: |y_k| <= sqrt(d_msa - 1)|gamma_k| + |beta_k|,
  max_c sum_k |Wv_ck| (...) x (1 + 2^-6)): no calibration, no guard (the H100 route needed 16 x headroom + `take_overflow()` because its v scale came from a first-call amax). Fp32 accumulation, bf16 o exactly where cuBLAS bmm writes it; `pwa_gate_out` is the default's.
* B200 specific: the default `ln_vg` output v is head-major [H, N, S C] = an MN-major B operand, which cuBLASLt's e4m3 GEMM takes as it is on Blackwell (measured: NN == TN speed) -- Hopper needed v transposed to K-major (`ln_vt8`, a new kernel).
  Eight per-head `torch._scaled_mm` GEMMs with `out=` o[h]. (a) MiniWorld's MSA side (d_msa 64, 8 x 32, inference at d_pair 128): `ln_vg_f8` -- the drain of `ln_vg` writes v as e4m3 (v x q, saturating cast, 32 B rows, no swizzle, u8 TMA map) instead of bf16: no cast pass over v
  (v is 0.4-1 GB at N 768-2048), half its bytes; `pwa_dual` (pair_fwd + ln_vg in one launch) is replaced by the split launches while the route serves. (b) every other build (Chai-1 d_pair 256, 8 x 16, IntelliFold 256/512): one cast launch for v (`opm_cast1`).
* Gates: fused (a): N^2 S >= 5e8 (padded); cast route (b): head dim 32 from N 1280, 16 from N 1792, 8 never (the below-5 % bar). Below them bf16 (cuBLAS bmm) runs unchanged.
* Not served: training, p_drop / gate / out passes stay bf16 (as H100: gate pass is latency bound, o e4m3 lost on H100).

### MSAPairWeightedAveraging results (B200 GPU 3, same protocol; e4m3 / default; d_msa 64 / d_pair 128 / 8 x 32 (MiniWorld, fused v), depth 1024)
| N (= L) | default us | e4m3 us | e4m3 / default | ctrl | gold bf16 rms / judge | gold e4m3 rms / judge | energy ratio |
|---|---|---|---|---|---|---|---|
| 768 | 467.7 | 413.5 | 0.884 | 1.00 | 0.00175 / 0.00295 | 0.00217 / 0.00350 | 0.86 |
| 1024 | 707.6 | 559.8 | 0.791 | 1.00 | 0.00175 / 0.00285 | 0.00207 / 0.00316 | 0.79 |
| 1536 | 1342.5 | 960.7 | 0.716 | 1.00 | 0.00175 / 0.00294 | 0.00197 / 0.00319 | 0.72 |
| 2048 | 2096.1 | 1449.6 | 0.692 | 1.00 | 0.00174 / 0.00285 | 0.00192 / 0.00303 | 0.71 |
Depth sweep (e4m3 / default, fused, gate removed, `pwa/s2_fused.jsonl`): L512: S384 1.25, S1024 0.99, S2048 0.87; L768: 1.02 / 0.86 / 0.79; L1024: 0.96 / 0.79 / 0.75; L1280: 0.97 / 0.76 / 0.72; L1536: 0.83 / 0.72 / 0.69; L2048: 0.79 / 0.69 / 0.66; L256-L384 lose (1.03-1.45).
The first (cast) version of the 8 x 32 route: 0.90 / 0.89 / 0.84 at N 768 / 1536 / 2048 (1.10 at N 768): the fused v halves its cost.
Other builds (cast route, S1024): Chai-1 d_pair 256 8x32 N 1536 / 2048: 0.885 / 0.847; 128/256 8x16 N 1536 / 2048: 0.962 / 0.907; 256/512 8x32 N 1536 / 2048: 0.940 / 0.911; 8x8 builds 1.01-1.13 at N 1024-1536, 0.95-0.97 at N 2048 (not served).
Accuracy: PWA's update is small next to the residual, so the e4m3 contraction adds little: judge 0.30-0.35 % (bf16 0.28-0.29 %, tolerance 8e-3), update rms 0.19-0.22 % (bf16 0.175 %); large-weight stress (LN weights / biases x 8): 0.92 % judge (bf16 0.57 %), finite.
H100 route: judge 0.30-0.37 %. Energy at the cap = time ratio.

### Verification (OuterProductMean / MSAPairWeightedAveraging)
* FP8 OFF, bit-identity of the default route: the merge agent's `bid.py` (eager) and `bidg.py` (CUDA graph, replay twice) over the measurer cases of `OuterProductMean` and `MSAPairWeightedAveraging` (both modes: inference + training; L 128 / 384 / 640, every depth /
  width / variant, bf16 / master / fp32: 174 cases, `_tools/msabid.yaml`), sha1 of outputs and parameter gradients, tree n13-fp8-c (final) against n13-base: **174 / 174 identical in both, 0 mismatches, 0 errors**
  (`$W/b2511/work/n13-fp8-c/bid_msa2/cmp.txt`; the 30 cases flagged non-deterministic are the training cases with dropout draws, flagged identically in both trees). The default kernels are the F8 = false instantiations of
  `opm_prologue_sm100` / `ln_vg_body` (default template argument, defaulted trailing kernel parameters): same code, same bits. Their artifacts are re-keyed (source text changed): `miniworld_opm_sm100`, `miniworld_pwa_sm100`.
* Tests: `tests/gpu/test_opm_fp8_b200.py` (accuracy at 64/128 plain / projection bias / count_offset and 128/256, large-weight stress, below-gate and off bit-equal, graph == eager), `tests/gpu/test_pwa_fp8_b200.py` (accuracy + stress at N 1280 depth 512, below-gate
  and head-dim-8 bit-equal, graph == eager); with the token DiT tests, the B200 suites (trimul / triattn / capture_bits / transition_uninit / misc / local_dit / token_dit / atom_dit / opm / pwa / engine_fp8 / route_stats / i64_opm_pwa) and
  `tests/cpu/test_fp8_switch.py`: **583 passed, 491 skipped (H100-only), 0 failed** (`$W/b2511/work/n13-fp8-c/tests2/pytest.log`).
* `_prebuild --arch sm100 --check`: 807 artifacts (739 cubin, 68 ext), 0 not built yet (base 804 / 66); new: `opm_fp8_b200` (the cast helper), plus the re-keyed opm / pwa extensions and the token DiT pair of the first session; rc 1 only from the unchanged `*_sm90` / mpnn lines (`prebuild2/`).

### Rejected / not built (with data)
* Chai-1's grouped OPM (8 groups of 8, product LayerNorm): kept bf16. Product only, B200, us (bf16 `bmm` / eight e4m3 GEMMs / the separate cast): N256 S1024 46.9 / 56.5 / 53.5 (a loss), N384 S1024 100 / 81 / 80, N512 176 / 125 / 106, N384 S4096 383 / 217 / 312, N768 S1024 394 / 244 / 159.
  The cast costs what the GEMM saves below S 4096 and the H100 census puts the grouped OPM at 0.7 % of Chai-1's kernel time: no route.
* OPM O as e4m3 (the epilogue reads 302 MB at L384 S2048: 13 % of the module, 25 % at L256 S1024): not built (adds a third e4m3 rounding and an epilogue rewrite; the H100 doc estimated <= 2 % there).
* PWA 8 x 8 builds: the e4m3 contraction is 5 % of the module at best at N 2048 (0.95-0.97), a loss below: not served. PWA o as e4m3 / gate-out in e4m3: lost on H100 (the gate pass is latency bound), not retried.
* OPM / PWA training: no e4m3 placement (as 3.0.3).

### Open items (OPM / PWA)
1. OPM wide prologue (`opm_wprologue_sm100`: d_msa 128 / 256, projection biases): write e4m3 A2 / BT from its drain as the narrow prologue now does (today one cast launch: ~6-8 % of the module at S 2048).
2. PWA wide builds (`ln_vg_w`, Chai-1 d_pair 256 / 8 x 16 / IntelliFold 256-512): fused e4m3 v (the cast pass is ~15 % of the 8 x 32 module at N 2048).
3. OPM O as e4m3 end to end (cuBLASLt e4m3 D + an e4m3-reading epilogue); an e4m3 version of the fused bf16 kernel for S <= 512 (N >= 512), a new tcgen05 kernel.
4. PWA: the eight per-head `_scaled_mm` launches -> one batched / grouped e4m3 GEMM; `pwa_dual` (pair_fwd + ln_vg in one launch) has no e4m3 form: the route runs the split launches.
5. End-level (diffusion RMSD) acceptance on B200 not run; judge errors above the 8e-3 tolerance for OPM (1.0-2.0 %), inside it for PWA (0.30-0.35 %).
6. `fp8.py` registration: `$W/b2511/work/n13-fp8-c/FP8_REGISTER.md` (OPM / PWA section).
