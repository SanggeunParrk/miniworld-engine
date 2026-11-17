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
