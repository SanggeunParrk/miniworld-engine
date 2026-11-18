# FP8 training kernels (H100, opt-in, 3.1.0 line, EXPERIMENTAL)

FP8 inside the GEMMs of the training step (forward + backward). Everything here is opt-in, default OFF, separate from the inference
`safe` preset (`miniworld_engine.fp8`, docs/fp8.md). With the switches off the training route is the 2.5.14 one, bit for bit.

## MSAPairWeightedAveraging (`kernels/integrations/pwa_fp8_train.py`, branch `fp8t/pwa`, base 1a2bbe42e = rel/3.0.0)
**Result (H100, 8 x 32 widths):** FP8 o / dv / dw (cuBLASLt, operands written by fused producers, do e4m3, delayed per-tensor scales) makes
the training step (fwd + bwd) **0.85-0.91x** of the 2.5.14 default (cool; 0.88-0.92 sustained) at the registry training shapes; gradient error
against fp64 rises from 0.3-0.7 % (default) to 2-5 % rms; a synthetic 1000-step distillation ends ~1 % higher in loss. Opt-in only
(`MINIWORLD_PWA_FP8_TRAIN=1`), default bytes unchanged. The `pwa_wide` (AF3-family) widths are not served (upper bound 4-7 %).

### Where the training step's time goes (2.5.14 default, measured)

Step = forward + backward of one module call (`torch.autograd.grad` of the output, all inputs and parameters), master weights, p_drop 0.25,
one CUDA graph (median of 3 x 20 replays); stage times = torch profiler kernel times of eager steps (mean of 5). Scripts / raw rows:
`~/miniworld-engine-scratch/fp8t-pwa/{scripts,res}` (`prof.py`, `res/prof0.jsonl`).

8 x 32 (`pwa_train`, native kernels), S 1024, N 768, d_pair 128: step 3.17 ms.

| stage | work | us | share | bound (ncu not run; from the GEMM rate / bytes) | FP8 placement |
|---|---|---|---|---|---|
| `pair_fwd3` | LN_z, z [N^2,128] x Wb, softmax | 68 | 2 % | memory (z read) | none (z is the module input) |
| `ln_vg` | LN_m, y [S N, 64] x Wv^T -> v [H][N][S C] | 216 | 7 % | memory (v write) | v as e4m3 (vT + natural) |
| contraction o = w v | per head [N,N] x [N, S C], 309 GFLOP | 446 | 14 % | tensor (cuBLAS bf16 ~690 TFLOP/s) | **w e4m3, v e4m3** |
| `pwa_gateout` | gate GEMM, sigmoid, out GEMM, dropout, residual | 325 | 10 % | latency (inference ncu: DRAM 61 %, tensor 20 %) | none (o e4m3 tried in inference: no gain) |
| `pwa_glue3` | do = (dout Wo) g, dgp, dWo partials | 643 | 20 % | latency / memory (reads o, y, dout; writes do, dgp) | do written as FP8 (natural + transposed) |
| dv = w^T do | per head [N,N]^T x [N, S C], 309 GFLOP | 422 | 13 % | tensor | **w e4m3, do e5m2 / e4m3** |
| dw = do v^T | per head [N, S C] x [S C, N], 309 GFLOP, K = 32768 | 340 | 11 % | tensor | **do e5m2 / e4m3, v e4m3** |
| `dgv_bwd` | dy = [dgp\|dv][Wg;Wv], LN backward, dWgv | 422 | 13 % | memory (reads dgv [S N 512] + dv) | none (dWgv: 2 x 25 GFLOP inside a memory-bound pass) |
| `pair_bwd3` | softmax / proj_z / LN_z backward | 231 | 7 % | memory | none |
| casts, reduces, mask, transposes | | ~50 | 2 % | launch | none |

The three key contractions are 38 % of the step (1208 us). The projection GEMMs (Wv, Wg, Wo and their dgrad / wgrad) live inside the fused
memory / latency-bound kernels (`ln_vg`, `pwa_gateout`, `pwa_glue3`, `dgv_bwd`), 25-51 GFLOP each at 15-20 % tensor use: e4m3 operands there
cannot shorten an MMA-bound part (there is none). Other shapes: S 2048 N 768 contractions 896 + 838 + 671 = 2405 of 5923 us (41 %);
S 4096 N 768 4808 of 11443 us (42 %); S 1024 N 384 (fused `pwa_fwd2` forward 302 us, dv 152, dw 158) 22 % of 1410 us (+ the fused forward).
Not native (plain builds; default at every shape but the six): S 1024 N 512 2.12 ms, S 384 N 768 1.67 ms -- the plain forward / `pwa_plain2`
dv / Triton pair backward.

FP8 GEMMs alone (`gemm.py`, `res/gemm0.jsonl`, CUDA graph, us, bf16 cuBLAS bmm -> cuBLASLt FP8 with fp32 accumulation):

| C:S:N | GFLOP | o | dv | dw | sum bf16 -> FP8 |
|---|---|---|---|---|---|
| 32:1024:384 | 77 | 161 -> 115 | 161 -> 115 | 166 -> 76 | 488 -> 306 (0.63x) |
| 32:1024:768 | 309 | 454 -> 300 | 512 -> 331 | 416 -> 243 | 1382 -> 874 (0.63x) |
| 32:2048:768 | 618 | 907 -> 698 | 1000 -> 649 | 813 -> 481 | 2720 -> 1828 (0.67x) |
| 32:4096:768 | 1237 | 1988 -> 1321 | 1988 -> 1323 | 2111 -> 1120 | 6087 -> 3764 (0.62x) |
| 32:384:768 | 116 | 178 -> 113 | 178 -> 125 | 156 -> 93 | 512 -> 331 (0.65x) |
| 32:1024:512 | 137 | 235 -> 164 | 247 -> 167 | 194 -> 104 | 676 -> 435 (0.64x) |
| 32:1024:1024 | 550 | 827 -> 563 | 824 -> 520 | 727 -> 418 | 2378 -> 1501 (0.63x) |
| 8:1024:768 (AF3 width) | 77 | 122 -> 79 | 122 -> 80 | 104 -> 63 | 348 -> 222 (0.64x) |
| 16:1024:768 | 155 | 233 -> 153 | 249 -> 164 | 204 -> 134 | 686 -> 451 (0.66x) |

Operand layouts. Hopper's FP8 MMA takes K-major operands only, and each contraction runs over a different index: o over j (w row-major,
v needs j-contiguous = vT), dv over i (w needs w^T, do needs do^T), dw over S*C (do and v as they are). So v must exist as vT and natural,
do as natural and do^T, w as w and w^T. A standalone quantize / transpose pass over a [H, N, S C] tensor costs more than the whole GEMM
gain (PyTorch: 2.1 ms quantize, 1.4 ms byte transpose at S1024 N768 against a 0.5 ms saving), so the producers write them:

- `ln_vt8n` (`csrc/pwa_vt8.cu` built with `-DVT_NAT=1`): `ln_vt8` (inference FP8 route's producer: y bf16 = `ln_vg`'s bytes, vT e4m3) plus
  v e4m3 natural from a second wgmma with the roles swapped (rows n, columns hc: consecutive channels pack into one e4m3x2), one more TMA
  box store. Replaces `ln_vg`; v bf16 is never written (training saves v e4m3: half the bytes).
- `pwa_glue3_f8` (`csrc/pwa_glue3_f8.cu`): `pwa_glue3` with the do epilogue writing e5m2 (or e4m3) do [64 i][64 x] and do^T [64 x][64 i]
  tiles (two 4 KiB TMA boxes in the bf16 tile's staging) and the amax; no bf16 do.
- `w_quant` (inference route) / `w_quant_t` (new): w and w^T as e4m3 of 256 w (exact power of two; w <= 1 cannot saturate), N^2 H bytes.
- `scale_update`: the delayed scale of one site in one launch.

Upper bound at S1024 N768: the contractions at the isolated FP8 rate save 1208 x (1 - 0.63) = ~450 us = 14 % of the step; producers
cost ~0 if fused at equal bytes (vT + v natural e4m3 = v bf16 bytes; do + do^T FP8 = do bf16 bytes) -> ~0.86x.

### Enable

| env / attribute | effect |
|---|---|
| `MINIWORLD_PWA_FP8_TRAIN=1` (or `all`; or `MINIWORLD_FP8_TRAIN=1`) / `pwa_fp8_train.PLACES` | FP8 o, dv, dw (fused producers). A subset (`fwd`, `dv`, `dw`, comma list) runs the PyTorch-quantized prototype (ablation only: its quantize passes cost 2+ ms) |
| `MINIWORLD_PWA_FP8_TRAIN_DO` | `e4m3` (default) or `e5m2` for do |
| `MINIWORLD_PWA_FP8_TRAIN_MARGIN` / `_HISTORY` | delayed-scale headroom (2) / amax history length (16 calls) |
| `MINIWORLD_PWA_FP8_TRAIN_IMPL=proto` | PyTorch quantize passes instead of the fused producers |
| `pwa_fp8_train.overflow_count()` / `reset()` | saturated calls since reset (one host sync) / forget the scale histories |

Scope: `pwa_train` widths (d_msa 64, 8 x 32; d_pair 128 / 256 / 384 / 512), H100, N % 128 == 0, N <= 1024, S even. With the switch on the
route runs the native kernels at every such shape (the bf16 default runs them only at the six qualified shapes). Not served: the `pwa_wide`
widths (AF3 family 8 x 8 / 8 x 16, BIG), B200, inference (the inference FP8 route is `pwa_fp8`). Scales: v and do per-tensor power of two,
delayed (max of the last 16 calls' amax x 2 / FMAX; recorded by the producers, one `scale_update` launch per site, CUDA-graph safe; first call
of a module seeds from its own tensor, eager; an all-zero first tensor -- zero-initialised `to_out` -- re-seeds on the next call); w fixed 2^-8.
Saturation: satfinite cast, counted (`overflow_count`), the history covers it on the next call; 0 overflows in every run here. The parameter-
gradient reductions are unchanged (native fixed-order partials): the FP8 step is deterministic (test: eager == eager == graph replay, bytes).

### Step time against the 2.5.14 default (`speed.py`, `res/sp2.jsonl`, `res/sp1.jsonl`)

One process; fwd + bwd (`autograd.grad` of the output w.r.t. msa, pair and every parameter) of one module call, master weights, p_drop 0.15,
each variant one CUDA graph; def / ctl (same-cubin control) / FP8 interleaved; cool = 2 s idle + short burst, median of 3; sustained = after
1 s of back-to-back replays, mean of 3. Same-cubin control 0.986 .. 1.006. Cells FP8 / default, `cool / sustained`.

| S:N[:d_pair] | default ms (cool) | FP8 do e4m3 | FP8 do e5m2 | native bf16 (attribution) |
|---|---|---|---|---|
| 1024:768 | 3.27 | 0.880 / 0.906 | 0.875 / 0.900 | 1.00 |
| 1024:384 | 1.44 | 0.910 / 0.918 | 0.908 / 0.916 | 1.00 |
| 2048:768 | 6.18 | 0.860 / 0.879 | 0.852 / 0.883 | 1.00 |
| 4096:768 | 11.77 | 0.871 / 0.884 | 0.861 / 0.883 | 1.00 |
| 2048:384 | 2.75 | 0.888 / 0.895 | 0.881 / 0.890 | 1.00 |
| 1024:768:256 (Chai-1) | 3.47 | 0.889 / 0.907 | 0.886 / 0.900 | 1.00 |
| 384:768 (plain default) | 1.70 | 0.801 / 0.808 | 0.796 / 0.806 | 0.877 |
| 1024:512 (plain default) | 2.16 | 0.810 / 0.825 | 0.810 / 0.828 | 0.895 |

At shapes outside the six the default runs the plain builds; the "native bf16" column (`NATIVE_ONLY`, sp1) is the same route with no FP8:
there about 0.88-0.90 of the gain is the native kernels and FP8 adds ~0.90-0.91x on top.

Per stage at S1024 N768 (profiler, us; default -> FP8 e4m3): `ln_vg` 217 -> `ln_vt8n` 259 (+42: the natural-v byte scatter; launch tuned
1 block/SM x 4 stages, 310 at 2 x 3); o 448 -> 293; `pwa_glue3` 652 -> `pwa_glue3_f8` 658; `w_quant_t` +16 (+ `w_quant` ~7); dv 448 -> 293;
dw 387 -> 233; `scale_update` 2 x ~2; every other stage unchanged. Step kernel sum 3273 -> 2872 us (0.877). Against the upper bound
(~0.86x) the producers give back ~2 % (the natural v / do^T byte stores).

### Gradient error against fp64 PyTorch (`err.py`, `res/err1.jsonl`)

Module in float64 on the PyTorch backend, same inputs; random weights (fan-in^-0.5, `to_out` included), LN weights 1 +- 0.1, biases +- 0.1,
3 masked keys, p_drop 0, upstream gradient N(0, 1); FP8 = second call (delayed scale). rms relative error in % (max relative in the raw rows,
same order of magnitude). upd = the output update y - x.

S1024 N768, d_pair 128:

| tensor | PyTorch bf16 | default 2.5.14 | FP8 do e4m3 | FP8 do e5m2 | FP8 dv+dw only (do e4m3) |
|---|---|---|---|---|---|
| update (fwd) | 2.38 | 2.37 | 2.89 | 2.89 | 2.37 |
| d msa | 0.17 | 0.17 | 0.21 | 0.25 | 0.21 |
| d pair | 0.71 | 0.58 | 3.80 | 5.93 | 3.82 |
| d ln_msa.weight / bias | 0.46 / 0.39 | 0.36 / 0.35 | 2.50 / 2.57 | 4.17 / 4.82 | 2.44 / 2.67 |
| d to_value | 0.43 | 0.28 | 2.75 | 5.36 | 2.79 |
| d ln_pair.weight | 0.81 | 0.59 | 4.14 | 6.97 | 4.23 |
| d to_bias | 0.69 | 0.55 | 3.80 | 5.96 | 3.83 |
| d to_gate | 0.59 | 0.40 | 1.69 | 1.69 | 0.40 |
| d to_out | 0.45 | 0.32 | 1.56 | 1.56 | 0.32 |

The same picture at every case (1024:384, 2048:768, 384:768, 1024:512, 1024:768:256): FP8 e4m3 do 2.1-5.0 % rms on the gradients behind dv
(to_value, ln_msa) and dw (pair, to_bias, ln_pair), 1.6-2.2 % on to_gate / to_out (from the FP8 forward o), d msa 0.21-0.25 %; e5m2 do
4-7 %. Default 0.3-0.7 %. No NaN / inf, no saturation. The error is the per-element rounding of the e4m3 / e5m2 operands (rel. ulp/2 = 2^-4 /
2^-3), which a dot product does not average away (random signs: the error grows with sqrt(K) like the sum) -- so it does not shrink with S or N.
Placement attribution (`proto` per placement, 1024:384): fwd alone -> to_gate / to_out 2.2 %, update 3.1 %; dv alone -> to_value / ln_msa
~5.8 % (e5m2); dw alone -> pair / ln_pair / to_bias ~6 % (e5m2).

### Training dynamics (`train.py`, `res/tr3*.jsonl`, `res/tr2*.jsonl`)

2-layer PWA stack (cuda backend, master weights, 8 x 32, p_drop 0.15) distilled onto a random fp32 PyTorch teacher stack; fresh random msa /
pair per step (same stream for every variant), loss = relative MSE of the update, Adam lr 1e-3, 1000 steps. Mean loss per 200-step window:

| S:N | variant | 1-200 | 201-400 | 401-600 | 601-800 | 801-1000 |
|---|---|---|---|---|---|---|
| 1024:384 (native default) | default (dropout seed 5) | 0.6265 | 0.3260 | 0.2084 | 0.1642 | 0.1568 |
| | default (dropout seed 6) | 0.6268 | 0.3270 | 0.2103 | 0.1660 | 0.1573 |
| | FP8 e5m2 (seed 5) | 0.6269 | 0.3264 | 0.2084 | 0.1656 | 0.1585 |
| | FP8 e5m2 (seed 6) | 0.6272 | 0.3281 | 0.2098 | 0.1665 | 0.1585 |
| | FP8 e4m3 (seed 5) | 0.6269 | 0.3263 | 0.2085 | 0.1656 | 0.1585 |
| | FP8 forward o only (proto) | 0.6269 | 0.3255 | 0.2077 | 0.1657 | 0.1587 |
| | FP8 dv + dw only (proto, e4m3 do) | 0.6265 | 0.3259 | 0.2082 | 0.1643 | 0.1568 |
| 512:384 (plain default) | default | 0.6330 | 0.3322 | 0.2213 | 0.1757 | 0.1618 |
| | FP8 e5m2 | 0.6332 | 0.3304 | 0.2211 | 0.1767 | 0.1637 |
| | FP8 e4m3 | 0.6332 | 0.3304 | 0.2211 | 0.1766 | 0.1637 |

Gap FP8 / default in the last window: +1.1 % (seed 5), +0.8 % (seed 6) at S1024 N384; +1.2 % at S512 N384; the dropout-seed spread of the
default is 0.3 % there (and 0.7 % in window 4). So FP8 trains to a ~1 % higher loss on this task -- small, systematic, the same for e4m3 and
e5m2 do; no divergence, NaN or overflow. The placement runs locate it: **the gap is the forward e4m3 contraction** (o only: +1.2 %); FP8 dv + dw alone end equal to the default (0.1568 vs 0.1568) despite their 2-4 % gradient error. Synthetic task, 1000 steps: not evidence for a real model's training (that needs an end-level run).

### Bit-identity of the default (switch off)

`bit.py` (sha1 of the output and all 10 gradients, training with p_drop 0.15 + inference, separate processes, deterministic parameters):
this branch vs the base tree (1a2bbe42e = rel/3.0.0 = 2.5.14 + inference FP8), 7 cases -- 8 x 32 S1024 N384 / N768 (native), d_pair 256
N768, AF3 64 x 8 N384 (wide), 128 x 16 d_pair 256 (wide): every hash equal (`res/bit_wt3.jsonl` vs `res/bit_base3.jsonl`). The two plain-build
cases (S1024 N512, S384 N768) are equal in output, d msa, d pair and 7 of 8 parameter gradients; d ln_msa.weight / bias differ -- and differ
between two runs of the base tree too (`bit_base3` vs `bit_base4`): the plain `dgv_bwd` build sums them with float atomics (documented, not a
change). The inference FP8 producer `ln_vt8` (`pwa_vt8.cu` without `-DVT_NAT`) compiles from the same preprocessed text as before.

### Tests

`tests/gpu/test_pwa_fp8_train.py` (new): fused producers vs PyTorch quantization (vT8 = transpose of v8, amax, w_quant_t = w_quant^T),
fused vs proto error against fp64 (both formats), route gradients within max(8 %, 6 x default error) of fp64 at 3 shapes (d_pair 128 / 256)
and the default bytes unchanged after the route ran (qualified shape), overflow 0, graph replay == eager == eager (deterministic). Plus the
files importing what changed: `test_pwa_fp8.py`, `test_pwa_fp8_wide.py`, `test_pwa_long_inference.py`, `test_i64_offsets.py::test_pwa_train_pair_past_2_31`;
CPU: `test_capture_api.py` (the new module dicts listed), `test_fp8_switch.py`; ruff check / format, pyright clean on the touched files.
Results (job 57855, final tree): `test_pwa_fp8_train.py` 7 passed; `test_pwa_fp8.py` + `test_pwa_fp8_wide.py` + `test_pwa_long_inference.py` 28 passed;
`test_i64_offsets.py::test_pwa_train_pair_past_2_31` 1 passed; CPU `test_capture_api.py` + `test_fp8_switch.py` 25 passed (job 57856).

### What did not pay / not done (numbers)

- **Wide widths (AF3 8 x 8, 128 x 8 / 16, BIG)**: the three contractions are 18 % (AF3 S1024 N768: 120 + 112 + 90 of 1823 us), 11 %
  (128 x 8 d_pair 256) and 12 % (BIG) of the step; FP8 GEMMs alone are 0.64x -> upper bound 4-7 % before the producers (new `ln_vg_w` / `gate_bwd_w`
  FP8 epilogues in `pwa_wide.cu`, which would re-key the default build or need copies). Not built. A larger, accuracy-free target there: the
  dropout mask is applied by two PyTorch elementwise passes over [S, N, D] in the backward (130 + 66 us = 11 % of the AF3 step; 13 % at 128 x 8;
  BIG also a 596 us `dgv` permute-copy) -- folding it into `gate_bwd_w` like `pwa_glue3` does is a bf16 change, outside this FP8 task.
- **Swapped-role wgmma for the natural v** (second m64n64 MMA, consecutive channels pack into halfwords): 335 us vs 290 for the byte scatter;
  reverted. Launch tuning of the byte-scatter build: 1 x 4 -> 245-259 us (from 310). Still +42 us over `ln_vg`.
- **Unfused quantize / transpose passes** (`IMPL=proto`): 2.1 ms (quantize) + 1.4 ms (byte transpose) per [H, N, S C] tensor at S1024 N768
  against a 0.5 ms GEMM saving -- the producers must write the FP8 layouts.
- **FP8 in the projection GEMMs** (Wv / Wg / Wo dgrad / wgrad, 25-51 GFLOP each): they live inside memory / latency-bound fused kernels at
  15-20 % tensor use; no MMA-bound time to remove (as in the inference study).
- **Forward contraction at N < 512**: the default's fused `pwa_fwd2/3` forward is replaced by split FP8 + `pwa_gateout`; still a win in the
  step (0.91 at S1024 N384) because the backward gains dominate.
- Not measured: fast accumulation (`MINIWORLD_PWA_FP8_TRAIN_FASTACC=1` exists, 1-3 % in the inference study), end-level training.

### Recommendation

Speed: 0.85-0.91x of the 2.5.14 step at the 8 x 32 training shapes (cool; 0.88-0.92 sustained), 0.80-0.83 at shapes the default serves with
the plain builds (of which ~0.88-0.90 is the native kernels alone -- a bf16 change worth making separately). Accuracy: gradient error
2-5 % rms (e4m3 do) against 0.3-0.7 % default, ~1 % higher loss on a synthetic distillation run. Keep it opt-in (3.1.0, e4m3 do). The loss gap comes from the forward FP8 contraction, not the backward: a backward-only variant (bf16 forward, FP8 dv / dw) would keep the default loss curve here but needs a producer that writes bf16 v for the forward and e4m3 v natural for dw (`ln_vg` + an e4m3 copy, ~+0.07 ms) and keeps ~2/3 of the GEMM saving -- the next step if the full route's 1 % does not pass an end-level check; ship as
default only after an end-level training run (e.g. Boltz-2 / MiniWorld trunk fine-tune loss curves) shows no gap. The wide (AF3-family)
widths are not worth an FP8 route (<= 4-7 % bound); their dropout passes are the better bf16 target.

### Progress log

- 2026-11-18 00:20 branch `fp8t/pwa` from 1a2bbe42e; 01:05 stage profile + GEMM bench; 01:10 first fused route (0.886 / 0.882 at S1024 N768);
  01:25 swapped-role natural-v store (slower, reverted); 01:33 launch tuning (1 x 4); 01:35 sp2 / training curves / bit-identity; 08:00
  (after an API outage 02:00-05:10) tests fixed (fused-vs-proto against fp64; default bytes checked at a qualified shape), docs.

## Rounds 2 and 3: backward-only variant, baseline = the 3.0.1 default (branch `fp8t3/pwa` = `fp8t2/pwa` + origin/main f5be1df46)

Everything below is measured against the **3.0.1 default** (the branch was merged with origin/main 3.0.1 first; the plain-build shapes are
a little faster there than in 2.5.14: S1024 N512 1.94 ms against 2.14). Scripts / raw rows: `~/miniworld-engine-scratch/fp8t3-pwa/{scripts,res}`;
the step-time benchmark is `tools/fp8_train/pwa_step_bench.py` (+ `run_pwa_step_bench.sbatch`).

### Enable (added)

| env / attribute | effect |
|---|---|
| `MINIWORLD_PWA_FP8_TRAIN=bwd` | **backward-only**: bf16 forward (default numerics), FP8 dv / dw (producers `ln_vt8n -DVT_BF` + `pwa_glue3_f8` + `w_quant_t`) |
| `MINIWORLD_PWA_FP8_TRAIN_MIN_S` / `pwa_fp8_train.MIN_S` | smallest MSA depth served (default 1024: measured S 1024 .. 4096; below it the default route runs) |

Gates (`serves`): H100, N % 128 == 0, N <= 1024, S >= 1024, S even; `bwd` additionally N >= 512 (its forward needs the split head-major
contraction, N >= 512). Everything else runs the default route (bit-identical, see below).

### What was wrong with the round-2 producer, and the fix

Round 2's `bwd` variant (bf16 v for the forward + e4m3 natural v for dw from one swapped-role wgmma) gave only 0.994 of the default
(S1024 N768, cool): GEMMs saved 151 (dv 447 -> 297) + 153 (dw 388 -> 235) us, but `ln_vt8` (VT_BF) took 443 us against `ln_vg`'s 218.
Micro-benchmark of the producer alone (`mb.py`, S1024 N768, us): base 460-470; **no smem staging stores 468** (the bank conflicts of the
scatter are NOT the cost); **no v TMA stores 225** (the global stores are); swizzled staging (TMA 64B / 32B swizzle) 468 (no change).
The stores are 64 B (bf16) / 32 B (e4m3) row pieces at a stride of S*C bytes, and items were ordered n-block fastest, so the 132 concurrent CTAs
wrote 12 distant rows. Ordering the items **s fastest** (`VT_MAP=1`: concurrent CTAs write adjacent pieces of the same rows, which the L2
merges) takes the producer to **335 us** (swizzled staging 335, plain 337; cfg (1,4) best), bytes identical. Applying the same order to the
FP8 forward's `ln_vt8n` made it slower (the full route's step 0.904 against 0.887): left off there. The same order in `pwa_glue3_f8`
(`G3_MAP`): 662 against 659 us, no effect, off. 335 us against the memory-bound ideal ~290 (806 MB of v at ~2.8 TB/s): ~45 us left.

### Step time, fwd + bwd of one module call, against the 3.0.1 default (`bench_final.jsonl`, job 58927)

One process; `autograd.grad` of the output w.r.t. msa, pair and every parameter, master weights, p_drop 0.15, each variant one CUDA graph,
def / ctl (same-cubin control) / variants interleaved; cool = 2 s idle + short burst, median of 3; sustained = after 1 s of back-to-back replays,
mean of 3. Cells = variant / default, `cool / sustained` (ctl = control, 1.00 means the measurement is clean; 1.019 once at 2048:384).

| S:N[:d_pair] | default ms (cool) | ctl cool / sus | `bwd` (bf16 fwd, FP8 dv + dw) | full (`1`, FP8 o + dv + dw, do e4m3) |
|---|---|---|---|---|
| 1024:768 | 3.256 | 0.993 / 1.003 | **0.954 / 0.964** | 0.886 / 0.899 |
| 2048:768 | 6.075 | 0.993 / 0.997 | **0.951 / 0.958** | 0.881 / 0.892 |
| 4096:768 | 11.732 | 0.993 / 1.002 | **0.958 / 0.968** | 0.889 / 0.914 |
| 1024:768:256 (Chai-1) | 3.467 | 0.996 / 1.006 | **0.956 / 0.971** | 0.894 / 0.911 |
| 1024:512 | 1.942 | 0.998 / 1.005 | **0.972 / 0.982** | 0.914 / 0.925 |
| 1024:384 | 1.398 | 0.992 / 0.998 | not served (N < 512): default | 0.940 / 0.949 |
| 2048:384 | 2.620 | 1.019 / 1.004 | not served (N < 512): default | 0.922 / 0.927 |

Per stage at S1024 N768 (profiler, us; default -> `bwd`): `ln_vg` 219 -> `ln_vt8` (VT_BF, mapped) 329; contraction dv 447 -> 297; dw 388 -> 235;
`pwa_glue3` 612 -> `pwa_glue3_f8` 658; `w_quant_t` +16; `scale_update` 2 x 3. Gain 304 us of GEMM, cost 110 + 46 + 16 + 6 = 178 us of producers
(net ~126 us = 3.9 %; the measured 4.6 % cool also includes the smaller v bytes the dw GEMM reads).

### Gradient error against fp64 (`err3.jsonl`, rms relative error in %, same protocol as above; default / PyTorch / `bwd` / full e4m3)

| S:N | tensor | PyTorch bf16 | default 3.0.1 | `bwd` | full |
|---|---|---|---|---|---|
| 1024:768 | update (fwd) | 2.38 | 2.37 | 2.37 | 2.89 |
| | d msa | 0.17 | 0.17 | 0.21 | 0.21 |
| | d pair | 0.71 | 0.58 | 3.80 | 3.80 |
| | d to_value | 0.43 | 0.28 | 2.75 | 2.75 |
| | d ln_msa.weight | 0.46 | 0.36 | 2.30 | 2.50 |
| | d to_gate / to_out | 0.59 / 0.45 | 0.40 / 0.32 | 0.40 / 0.32 | 1.69 / 1.56 |
| 2048:768 | d pair / d to_value / d ln_msa.w | 0.71 / 0.45 / 0.62 | 0.58 / 0.31 / 0.44 | 3.79 / 3.01 / 2.41 | 3.79 / 3.01 / 2.66 |
| 1024:512 | d pair / d to_value / d ln_msa.w | 0.71 / 0.44 / 0.52 | 0.58 / 0.29 / 0.39 | 3.79 / 2.71 / 2.78 | 3.79 / 2.71 / 3.22 |

The `bwd` variant's forward (update, to_gate, to_out) equals the default's; the dv / dw gradients (pair, to_bias, ln_pair, to_value, ln_msa) carry
the e4m3 rounding (2.3-4.1 % rms, max 2-3 %), 4-8x the default's error; the full route adds the forward contraction (update 2.89, to_gate / to_out 1.6-2.0 %).

### Training curve (`tr3.jsonl`, `train.py`: 2-layer PWA stack distilled to an fp32 PyTorch teacher, S1024 N512, p_drop 0.15, Adam 1e-3, 1000 steps, mean loss per 200 steps)

| variant | 1-200 | 201-400 | 401-600 | 601-800 | 801-1000 |
|---|---|---|---|---|---|
| default | 0.6270 | 0.3329 | 0.2364 | 0.1950 | 0.1666 |
| `bwd` (FP8 dv + dw) | 0.6271 | 0.3333 | 0.2365 | 0.1952 | 0.1668 |
| full (FP8 o + dv + dw, e4m3) | 0.6273 | 0.3326 | 0.2374 | 0.1955 | 0.1671 |

Final window: `bwd` +0.1 %, full +0.3 % (one dropout seed; the default's own seed spread is ~0.3 %); no divergence, overflow 0. (The 3.0.0-line runs
above saw +1 % for the full route at S1024 N384; same direction, `bwd` indistinguishable from the default in both.) Synthetic task: not an end-level check.

### Default route bit-identical with the switch off

`bit.py` (sha1 of output + all gradients, training p_drop 0.15 + inference, 7 cases: native, d_pair 256, plain builds, wide): this branch against a
detached origin/main 3.0.1 worktree: `cmp` of the two result files equal (`bit_wt5` / `bit_base5`). Tests (job 58922): `test_pwa_fp8_train.py`
+ `test_pwa_fp8.py` + `test_pwa_fp8_wide.py` + `test_i64_offsets.py::test_pwa_train_pair_past_2_31` + CPU `test_capture_api.py`, `test_fp8_switch.py`: 57 passed;
ruff format / check, pyright clean.

### Verdict

- `bwd`: wins 2.8-4.9 % (cool; 1.8-4.2 % sustained) at S >= 1024, N 512-1024, d_pair 128 / 256, forward numerics unchanged, training curve = default;
  gradient error 2.3-4 % rms on the dv / dw side. Gated off for N < 512 (no win there: 0.99-1.00) and S < 1024.
- full: wins 6-12 % at every gated shape incl. N 384; forward error and +0.3-1 % loss.
- Remaining producer headroom: `ln_vt8` (VT_BF) 335 us vs ~290 ideal, `pwa_glue3_f8` +46 us over `pwa_glue3` (its do / doT stores are 64 B pieces,
  tile order did not help); worth ~1.5 % more at most. Not a default: opt-in, off. FP8 projection GEMMs, wide widths: unchanged no-go (above).

### Progress log (rounds 2-3)
- 2026-11-18 18:33 branch fp8t3/pwa = fp8t2/pwa + 3.0.1; 18:36 baseline: `bwd` 0.994 of default (producer 443 us); 18:52 producer micro-benchmark:
  smem scatter / swizzle irrelevant, stores dominate; 23:47 s-fastest item order 460 -> 335 us; 00:10 gradient error, training curve; 00:22 tests, bit-identity;
  00:23 `bench_final` (step table above); S gate (MIN_S 1024) added after.

## Round 4: saved-activation bytes, memory, finer scales (branch `fp8t4/pwa` = `fp8t3/pwa` 1269b85c6 + round 4)

Scripts / raw rows: `~/miniworld-engine-scratch/fp8t4-pwa/{scripts,res}` (`mem.py`, `errstudy.py`, `speed.py`, `err.py`, `train.py`). Baseline = the 3.0.1 default.

### 1. Saved activation bytes and peak memory (`mem.py`; eager fwd + bwd, MiB above the inputs + parameters; `saved` = allocated after the forward minus before, output excluded)

Saved by the module for the backward (S1024 N768, default): w 9 + v 384 + o 384 + y 96 = 873 MiB; the inputs (m 96, z 144) are the caller's. v and o are each S x N x 256 x 2 B and are 88 % of it.

| S:N[:d_pair] | default saved / peak | `bwd` saved / peak | `bwd` + O8 saved / peak | peak vs default (`bwd` / `bwd`+O8) |
|---|---|---|---|---|
| 1024:768 | 873 / 2776 | 681 / **2318** | 489 / **2130** | -16.5 % / -23.3 % |
| 4096:768 | 3465 / 10552 | 2697 / 9230 | 1929 / 8466 | -12.5 % / -19.8 % |
| 1024:512 | 580 / 1813 | 452 / 1542 | 324 / 1421 | -14.9 % / -21.6 % |
| 1024:384, 2048:384 (`bwd` not served; full route) | 434 / 1343, 866 / 2639 | full: 338 / 1247, 674 / 2447 | | -7 % |
| 1024:1024, 1024:768:256 | 1168 / 3785, 873 / 2924 | 912 / 3529, 681 / 2732 (before the frees below) | | |

What moved the peak: (a) FP8 v (round 1-3): saved -192 MiB (-22 %), peak -7 %; (b) **new, free**: the FP8 backward drops `doT8` right after the
dv GEMM and `do8` right after the dw GEMM (they were held until the function returned): peak -266 MiB (2584 -> 2318 at S1024 N768), no time
cost; (c) **O8** (below): saved o as e4m3, a further -192 MiB saved and -190 MiB peak. DRAM bytes per step (ncu, round 2 on the 2.5.14 base, S1024 N768,
read / write MB): default 4425 / 2376; `bwd` 3859 / 2579; full 3662 / 2375 -- the bytes drop only by the half-size v re-read (dw) and
the smaller dv / dw operand reads; the write side does not shrink (v bf16 + e4m3 are both written).

**Largest S that fits one H100 80 GB** (training step, `mem.py` OOM sweep, step 2048; N > 1024 is not served by the cuda training backend
(`BackendUnsupported`: pair backward holds N <= 1024), so there is no N sweep):

| N | default | `bwd` (frees + FP8 v) | `bwd` + O8 |
|---|---|---|---|
| 768 | 26624 (28672 OOM) | 30720 (32768 OOM) = +15 % | 32768 (34816 OOM) = +23 % |
| 1024 | 20480 (22528 OOM) | 22528 (24576 OOM) = +10 % | 24576 (26624 OOM) = +20 % |

### 2. FP8 storage of o (`O8`, `MINIWORLD_PWA_FP8_TRAIN_O8=1`, needs `bwd` or `1`; `csrc/pwa_o8.cu`)

o ([H, N, S*C], head-major) is quantised to e4m3 (delayed per-tensor pow2 scale, amax recorded by the kernel, same site machinery as v / do)
right before `save_for_backward`, and dequantised to bf16 in the backward just before `pwa_glue3_f8` (freed right after it). **A memory mode, not a speed
mode**: two memory-bound passes (403 MB read + 201 MB write each) cost ~420 us at S1024 N768 (step 1.07-1.11x of the default, i.e. +12-16 points over `bwd`).
Fusing the quantisation into `pwa_gateout` (which reads o anyway) and the dequantisation into `pwa_glue3_f8` (o arrives by TMA into swizzled smem
tiles that the consumers read as bf16) would remove most of it but means new copies of two persistent TMA kernels; not done (default `pwa_gateout`
cannot be re-keyed). Error (below) and curve: gate / out gradients go from 0.4 / 0.3 % to 2.6 / 2.5 % rms (same level as the other FP8 gradients).

### 3. Finer-grained scales: no-go (`errstudy.py`, `res/errstudy.json`)

Quantise-dequantise emulation, exact fp64 products, w = softmax(N(0, 1.5)), v with outlier channels (every 97th x8), do with lognormal row scales; rms relative error of the
contraction (S*C = 4096, N = 768, 8 heads), e4m3 both operands:

| contraction | per-tensor scale | per-row scale | per-128-block scale (not one GEMM) |
|---|---|---|---|
| dw = do v^T | 3.748 % | 3.748 % | 3.748 % |
| dv = w^T do | 3.713 % | 3.713 % | 3.713 % |
| o = w v | 3.730 % | 3.730 % | 3.730 % |

The error is the **mantissa rounding**, independent of the scale grain: one e4m3 rounding of a Gaussian vector is 2.65 % relative rms, two operands give sqrt(2) x 2.65 = 3.75 %.
An exact v alone: 2.65 % (the rounding of do remains); v as e4m3 hi + lo (2 GEMMs): 2.65 %; both operands hi + lo (4 GEMMs): 0.09 %. DeepSeek-V3's fine-grained
scales fix outliers / dynamic range and keep the loss within 0.25 %; they do not change the 3 mantissa bits, and the module's own error (3.8 % on d pair) is that floor.
Reaching <= 1.5 % rms needs >= 3 FP8 GEMMs per contraction (hi*hi + hi*lo + lo*hi) = more than the bf16 GEMM time saved. Per-row scales are also
more traffic (scale vectors) for no gain. No-go; the delayed per-tensor scale stays.

### 4. The remaining producer gaps

- `ln_vt8` (VT_BF) 335 us: bytes x 100 + y 100 + v bf16 403 + v e4m3 201 = 804 MB = 2.4 TB/s, against `ln_vg`'s 603 MB in 218 us = 2.77 TB/s: ~290 us at the same rate, i.e. ~45 us (1.4 % of the step) left. The stores are 64 B / 32 B row pieces (item = 64 tokens x one s); an item of 2 s would double the pieces (128 B / 64 B) but the smem staging then grows to 96 KiB: not tried, <= 1 % of the step.
- `pwa_glue3_f8` +46 us over `pwa_glue3`: the byte scatter of do / doT into smem has 4-way bank conflicts; 64B-swizzled staging tiles (TMA SWIZZLE_64B, `MINIWORLD_PWA_FP8_TRAIN_G3SWZ=1`)
  gave 664 us against 659: not the cost (the kernel is latency-bound on its o / dout stream); left off. Tile order (round 3): no effect.

### 5. Step time against the 3.0.1 default (`bench_final4.jsonl`, `tools/fp8_train/pwa_step_bench.py`; same-process interleaved CUDA graphs, ctl = same-cubin control)

Cells = variant / default, `cool / sustained`; round 3 in brackets for `bwd`.

| S:N[:d_pair] | default ms (cool) | ctl cool / sus | `bwd` (round 4 [round 3]) | `bwd` + O8 | full (`1`) |
|---|---|---|---|---|---|
| 1024:768 | 3.231 | 0.991 / 1.003 | **0.951 / 0.962** [0.954 / 0.964] | 1.081 / 1.088 | 0.890 / 0.897 |
| 2048:768 | 6.059 | 0.990 / 1.002 | **0.947 / 0.961** [0.951 / 0.958] | 1.083 / 1.098 | 0.875 / 0.889 |
| 4096:768 | 11.668 | 0.992 / 0.993 | **0.949 / 0.959** [0.958 / 0.968] | 1.093 / 1.106 | 0.878 / 0.895 |
| 1024:768:256 | 3.455 | 0.990 / 0.996 | **0.952 / 0.962** [0.956 / 0.971] | 1.071 / 1.081 | 0.888 / 0.902 |
| 1024:512 | 1.950 | 0.992 / 0.997 | **0.962 / 0.978** [0.972 / 0.982] | 1.106 / 1.117 | 0.906 / 0.912 |
| 1024:384 | 1.419 | 0.998 / 0.995 | default (N < 512) | default | 0.927 / 0.930 |
| 2048:384 | 2.716 | 0.990 / 0.995 | default (N < 512) | default | 0.900 / 0.902 |

`bwd` is 0.4-1 point better than round 3 (fewer live buffers); full unchanged.

### 6. Accuracy and curve of O8 (`err5.jsonl`, `tr5.jsonl`)

rms % against fp64 (S1024 N768): `bwd`+O8: update 2.37, d msa 0.22, d pair 3.80, d to_value 2.75, d ln_msa.w 2.71, d to_gate 2.63, d to_out 2.50 (the `bwd` route without O8: 0.40 / 0.32 on the last two;
full + O8: update 2.89, to_gate 3.12, to_out 2.90). 2048:768 and 1024:512 the same (to_gate 2.7, to_out 2.5-2.6).
Distillation curve (S1024 N512, 1000 steps, window means): default 0.6270 / 0.3329 / 0.2364 / 0.1950 / 0.1666; `bwd` + O8 0.6271 / 0.3361 / 0.2379 / 0.1965 / 0.1678 (+0.7 % in the last window,
`bwd` alone +0.1 %, full +0.3 %; seed spread ~0.3 %).

### 7. Verdict (round 4)

- Ship (opt-in) the `bwd` route as is, now with the free peak-memory gain (-12 .. -17 % peak, +10-15 % max S) at 0.947-0.962 of the default step.
- O8 = a memory mode (peak -20 .. -23 %, max S +20-23 %) at +8-11 % step time over the default and +0.7 % loss; useful only when S is the limit.
- Finer scales: no-go (rounding floor); <= 1.5 % rms is not reachable with e4m3 x e4m3 in one GEMM.
- Producers: `ln_vt8` is ~45 us (1.4 %) above its memory rate, `pwa_glue3_f8` latency-bound (swizzle tried): no further cheap gain.
- Default route bit-identical with the switch off: `bit.py` sha1 of all 7 cases equal to the 3.0.1 worktree (`bit_wt6` vs `bit_base5`).
