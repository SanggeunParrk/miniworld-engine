# Transition, LayerNorm, RMSNorm and InputFeatureEmbedder on B200 (sm_100a)

Card key `b200_sm100` (NVIDIA B200, capability 10.0, 148 SMs). This page covers the `"cuda"` backend of `modules.Transition`,
`modules.LayerNorm` (with `modules.RMSNorm`, which has no requirement row) and `modules.InputFeatureEmbedder` on the B200: what runs,
what was measured, and what is still not served (gap analysis). Judging: [measure](../measure.md), [judge](../judge.md).

## Transition (`modules/transition.py`, `cuda:b200_sm100` 28, `triton:b200_sm100` 22)

`Transition._supports_cuda` / `_forward_cuda` take a `cap == (10, 0)` branch (`modules.norm.b200`) to
`kernels/transition/cuda/route_sm100a.transition_sm100a`, which tries, in order:

| build | takes | source |
|---|---|---|
| `fused_sm100a` | D 128, n 4; bf16 activations; whole 128-row tiles | `sm100/tr_fwd_sm100.cu`, `sm100/tr_bwd_sm100.cu` (one persistent forward, a two-role backward + partial reduce) |
| `fused_wide_sm100a` | D 64 / 128 / 256 / 384 / 512 with n 2 / 4, D 768 with n 2; bf16; whole 128-row tiles; D 384 at most 16 tiles: the small-M PDL chains (`sm100/widths/tsx_*.cu`) | `sm100/widths/*.cu` |
| (zero-row padding) | a bf16 call whose row count is not a multiple of 128 is padded to the next tile for the two tuned builds and cut back | `route_sm100a._padded` |
| `fused_generic_sm100a` | D a multiple of 64 up to 1024, hidden a multiple of 128, any row count; bf16, or fp32 on TF32 tensor cores (only where torch allows TF32); fp32 inference at D 64 / 128 / 256 (D 256 up to 2^20 rows) in the one-launch `tg_fused_h16.cu` (fp16 operands) | `sm100/generic/tg_*.cu` |

fp32-master parameters are cast inside (one launch, or once per parameter version through `_capture.param_forms` at inference);
their gradients come back fp32. The kernels are the ones of the B200 port branch `origin/b200/ladder-final` (dc425caa5), brought over
unchanged except: the `settings` knobs of that tree are module constants (`DW_REPL`, `REPL`, `SMALL_TILES`, `SMALL_FWD_TILES`;
`$MINIWORLD_TRANSITION_SM100_NVCC` overrides the toolkit), a build failure goes through `_nvcc.build_failed` (2.5.x's policy), and
`fused_generic_sm100a._plan` plans an unsplit GEMM when K is a single K-block (D 64 bf16 crashed with `TypeError` there).
Cubins are built by `kernel_toolchain()` (the newest nvcc that knows sm_100a: CUDA 13.1 on the box) under
`engine_jit_root/transition_sm100a/`; the host binding `transition_fused_sm100a` (`sm100/transition_sm100.cu`) is built for
`sm_100a` only.

`route_sm100a.GENERIC` lists (D, n, training) cells and row ranges that skip the tuned builds for the generic kernels (measured faster there, below).

### Measured on the B200 (2026-11-16)

Records (B200 box): `$W/b2511/work/transition/acc.Transition.jsonl`, `acc.Transition.big.jsonl`, `spd.Transition.jsonl`
(`measure.accuracy` / `measure.speed`, the requirement rows 25-40 and 209-221, every length, `float32` and `master`, eval and train;
speed: CUDA graph, L2 flushed, GPU 1 exclusive); the PyTorch / Anthropic / cuEquivariance baselines from `$W/b2511/out_base/`.

Accuracy (`cuda`, 307 requirement cases): **254 GREEN, 49 YELLOW, 0 RED, 0 unsupported**, 4 skipped (the fp64 gold / reference of
rows 31 L 2976 / 3968 and 32 L 2048 does not fit beside the other jobs on the card; row 31 L 2048 re-measured GREEN). `triton`:
252 GREEN, 51 YELLOW. The YELLOW cases:

| cases | cuda err / tol | triton | why |
|---|---|---|---|
| fp32 eval rows 27 / 29 / 35 / 37 (13 cases) | 1.00-1.04 | YELLOW, the same error | TF32: the reference and both backends round alike; within 4 % of the tolerance |
| fp32 train rows 211 / 213 / 217 / 219 / 33 (30 cases) | 1.01-2.5 (row 33 D 512 n 2: up to 2.5 at L 768) | YELLOW, the same order | TF32 gradients; Triton is as far |
| master train row 209 (D 64, n 2, L 256-768; 5 cases) | 1.03-1.09 (`grad:x`, `grad:ln_in.weight`) | GREEN (0.79-0.89) | the wide D 64 n 2 backward's bf16 rounding of d_xn; the generic kernels are GREEN there but 1.6-2.1x slower (measured), so the tuned build stays |
| master train row 34 (D 512, n 4, L 768) | 1.18 (`grad:x`) | YELLOW, the same error | the generic kernels are GREEN but 1.4x slower |

Every YELLOW is within 5x the tolerance (no RED), as on the other cards (H100 168, A100 70, RTX 5090 43 YELLOW `cuda` tokens).

Speed, `cuda` against each baseline (ratio = baseline / cuda time; > 1: cuda faster; `spd.Transition.jsonl`, measured before the
`GENERIC` cells below were set):

| dtype, mode | vs triton: median (min-max), wins | vs pytorch: median (min-max), wins | vs anthropic |
|---|---|---|---|
| float32 eval (42) | 2.67 (1.38-3.46), 42 / 42 | 1.98 (1.05-4.62), 42 / 42 | 0.89 (0.71-0.90): the kit's D 128 n 2 fp32 cell is ~10 % faster (9 cases) |
| float32 train (30) | 1.93 (1.60-2.06), 30 / 30 | 1.35 (1.14-3.57), 30 / 30 | -- |
| master eval (139) | 2.30 (0.93-5.68), 132 / 139 | 4.88 (0.93-18.7), 134 / 139 | 2.65 (0.85-13.0), 126 / 128 |
| master train (96) | 2.00 (0.76-3.82), 90 / 96 | 3.53 (0.74-12.0), 90 / 96 | -- |

(cuEquivariance has no Transition.)

The 13 cases where the tuned builds lost to Triton (and PyTorch) in that run -- D 768 n 2 on the single stream (rows 37 eval, 219 train,
master) and D 384 n 4 eval at L 2976 (row 36) -- now run the generic kernels (`route_sm100a.GENERIC`), measured
(`spd.knob2_off.jsonl` / `spd.knob2_on.jsonl`, us):

| case | tuned | generic (now) | triton | pytorch |
|---|---|---|---|---|
| row 37 [1, L, 768] n 2 eval, L 128-2048 | 59.3-65.4 | **30.6-43.0** | 55.2-67.5 | 55.2-75.5 |
| row 219 [1, L, 768] n 2 train, L 128-768 | 167.8-174.1 | **61.3-75.7** | 128.8-135.0 | 124.8-141.2 |
| row 36 [1, 2976 / 3968, 384] n 4 eval | 59.2 / 55.2 | **38.8 / 40.8** | 55.2 / 59.3 | 69.6 / 79.8 |

Their accuracy on the generic kernels (`acc.knob3.jsonl`): every master case GREEN. With that, `cuda` beats Triton and PyTorch on
every measured Transition case on the B200. It is still behind the Anthropic kit on the fp32 D 128 n 2 pair cell (row 27, 9 cases,
0.71-0.90x) and ties it at row 27 master L 128.

## LayerNorm / RMSNorm (`modules/norm.py`, LayerNorm `cuda:b200_sm100` 22)

The 2.5.11 LayerNorm CUDA kernels already compile for sm_100 (`kernels/layernorm/cuda/general` and `cuda/pipe` pass
`gencodes("80", "90", "100")`), so the B200 route builds nothing new and no other card's build changes. On capability (10, 0)
`LayerNorm` / `RMSNorm` call `kernels/layernorm/cuda/b200.py`:

* `sm100` / `skew` (`cuda/sm100`, sm_100a only; rounds 2 and 3 below) for the widths of `b200.SM100_ROWS` and the bf16 odd widths of
  `b200.SM100_SKEW`;
* `pipe` (`cuda/pipe/ln_pipe.cu`) for rows of 144 .. 2048 bytes that are a 16-byte multiple;
* `rows` (`cuda/general/ln_rows.cu`) for rows that are not a 16-byte multiple (267 / 451 / 651 / 831 / 833), 32 .. 1024 elements;
* `general` (`cuda/general/ln_fwd.cu`, `ln_bwd.cu`) for every other width (bf16 64, 2560 ...).

The H100 gates (`pipe.serves`, `general.rows_serves`, capability 9.0) are untouched: the B200 route calls the same opaque ops
directly. `b200.SLOWER` lists the (width, dtype, mode) cells from which row count on the route refuses because the Triton
LayerNorm is faster there (measured).

The A100 / RTX 5090 row kernels (`cuda/sm80/norm_rows.cu`) were also built for sm_100a (`-gencode arch=compute_100a,code=sm_100a`,
a separate extension): every launch failed (`norm_fwd: launch failed`), so that build was dropped and is not shipped.

### Measured on the B200 (2026-11-16)

Records: `$W/b2511/work/transition/acc.LayerNorm.jsonl`, `spd.LayerNorm.jsonl` (the 44 requirement rows, every length, `float32`
and `master`, eval and train), measured before `SLOWER` was set (every case served); baselines from `$W/b2511/out_base/`.

Accuracy (`cuda`, 656 requirement cases): **656 GREEN** (0 YELLOW / RED). Triton: 656 GREEN.

Speed (baseline / cuda time):

| dtype, mode | vs triton: median (min-max), wins | vs pytorch | vs cueq | vs anthropic |
|---|---|---|---|---|
| float32 eval (190) | 1.53 (0.86-5.85), 187 | 2.69 (1.00-18.2), 182 | 1.00 (0.85-1.65), 124 | 1.25 (0.89-3.10), 139 |
| float32 train (132) | 1.73 (0.93-6.31), 128 | 2.35 (0.76-15.4), 110 | 2.13 (1.07-3.19), 132 | -- |
| master eval (196) | 1.51 (0.77-8.88), 142 | 4.67 (1.76-29.0), 196 | 1.33 (0.72-9.62), 129 | 1.49 (0.98-4.40), 137 |
| master train (138) | 1.92 (0.66-7.84), 134 | 4.28 (1.28-21.1), 138 | 2.63 (1.09-9.63), 138 | -- |

Where the CUDA kernels lose to Triton (now refused through `b200.SLOWER`; one tie and three wins within 2 us at the thresholds are refused with them). Re-measured accuracy with the final route
(`acc.LayerNorm.final.jsonl`): **594 GREEN, 60 unsupported (`SLOWER`), 2 skipped** (gold OOM beside the other jobs), 0 YELLOW / RED.

| cell | rows refused | cuda / triton (us), examples |
|---|---|---|
| D 64 bf16 eval (row 1, `general`: 128-byte rows are below `pipe`'s 144) | from 65 536 (pair L >= 256) | L 768 38.8 / 30.6, L 2048 219 / 168 |
| D 128 bf16 eval (rows 2 / 3, `pipe`) | from 147 456 (pair L >= 384; S 5: L >= 384) | L 768 69.4 / 61.3, L 2048 432 / 358 |
| D 256 bf16 eval (rows 4 / 5, `pipe`) | from 65 536 (pair L >= 256) | L 768 131 / 111, L 2048 878 / 700 |
| D 384 bf16 eval (rows 7 / 8, `pipe`) | from 147 456 (pair L >= 384) | L 768 199 / 184, L 2048 1360 / 1218 |
| D 64 bf16 train (row 189, `general`) | up to 262 144 (pair L <= 512) | L 128 24.4 / 16.3, L 512 63.2 / 59.3 |
| D 64 fp32 train (row 189, `general`) | from 147 456 (pair L >= 384) | L 768 143 / 133 |

Two fp32 eval D 64 cases (L 256: 14.0 / 12.1, L 768: 53.2 / 53.1) and one D 128 fp32 eval case (L 128 pair: 12.2 / 12.1) are within the
timer's 2 us steps, not refused. cuEquivariance's `layer_norm_transpose` is faster than the CUDA route on the same large bf16 pair
inference cells (e.g. D 128 L 2048 S 5: 1653 vs 1917 us) and on some fp32 eval cells (0.85x at worst).

Slower than PyTorch (served, cuda is 2-6x faster than Triton there): fp32 training of the single stream at D 768 (rows 202 / 203,
L 128-768: 18-24 us against PyTorch 16-22 us; Triton 55-57 us) and D 831 / 833 at L 128 / 256 (rows 204-206: 22.4 against 18-20 us;
Triton 133-141 us): the forward + dx + affine-reduce launches of a few hundred rows against PyTorch's fused native kernel.

## InputFeatureEmbedder (`modules/embedder.py`, `cuda:b200_sm100` 1)

A composition: its `cuda` backend hands `cuda` to its atom transformer (`SWAAtomTransformer`) and its norms (`atom_norm` /
`atom_out_norm`: LayerNorm, or an affine-free RMSNorm for the legacy features). Its card gate now admits the B200; the norms run the
route above. Measured after merging the atom-DiT port (`acc.InputFeatureEmbedder.final.jsonl`, `spd.InputFeatureEmbedder.jsonl`), 207
requirement cases: **29 GREEN, 52 YELLOW, 0 RED, 126 unsupported** -- every refusal is `SWAAtomTransformer`'s (bf16 global attention and
bf16 training below 8192 rows on the B200), and the YELLOW cases are its numerics (Triton: 48 GREEN, 96 YELLOW; the norms are GREEN on
their own rows). Speed of the 81 served cases: vs Triton 1.20-1.37 median (wins 76; 5 fp32 legacy global-window cases of row 136 are
0.75-0.90x), vs PyTorch 6.8-10x median (wins 81), vs Anthropic 1.25-9.4x median (wins 76). Closing the rest is the atom-DiT port's
work (the norms add nothing to it).

## Optimization round (2026-11-17)

Records (B200 box, `$W/b2511/agents/opt-transition/out/`): `spd.Tr.final.jsonl` / `acc.Tr.final.jsonl` (every Transition requirement
case, `cuda` + `triton`), `spd.r209.jsonl` / `acc.r209.jsonl` (row 209 after the route change), `spd_tr_f32_after2.jsonl` (Triton fp32
before the table re-tune), `spd.LN.final2.jsonl` / `acc.LN.final.jsonl` (LayerNorm, `cuda`, `master`), `cmp.LN.txt`; baselines and the
pre-round `cuda` / `triton` medians from `$W/b2511/out_base/speed.*.jsonl` (the ladder run of 2026-11-16). Ladder rule: graph timing,
L2 flushed, a violation when the faster rung is more than 2 % slower.

### Transition

* **Triton fp32 (`triton:b200_sm100` 1).** The ladder had 8-9 Triton < PyTorch misses, all fp32: row 33 (pair [1, L^2, 512], n 2, eval,
  L 256-512, up to 1.09x) and row 37 (single [1, L, 768], n 2, eval, up to 1.29x). The expand kernel rounded the x tile to the TF32 grid
  in registers (`ROUND_X`); on sm_100 that keeps Triton 3.7 from feeding `tl.dot` from the shared-memory tiles it loaded and the kernel ran
  at ~2.7x cuBLAS' TF32 time. On the B200 the activation is now rounded in a launcher pass (one `tf32_grid_many` over x) wherever K > 128
  (`triton/main.py: ROUND_X_IN_KERNEL_SM100`; at K 128 the extra pass measured 4-9 % slower, so D 128 keeps the in-kernel rounding).
  The B200 table was re-tuned for the affected rows (29 / 33 / 35 / 37 / 213 / 217 / 219: `kernels/_pretune` on GPU 1, merged into
  `transition/configs/b200_sm100.json`, version 2) and the M 1024 bucket of the d 768 expand takes the 128 x 64 tile first. The TF32
  weight forms are now cached on the B200 (`_tf32.tf32_cached(by_address=True)`, Transition launchers only): through the opaque ops the
  weights arrive as fresh alias tensors each call, so the identity key never hit and the stacked [Wa; Wb] and rounded Ws were rebuilt
  every call (a stack + two rounding launches, ~8 us of a 70 us call). Measured (triton, us, before -> after; PyTorch):

  | case | before | after | PyTorch |
  |---|---|---|---|
  | row 33 [1, 256^2, 512] n2 eval | 847.7 | 636.9 | 779.1 |
  | row 33 [1, 512^2, 512] n2 eval | 3163.5 | 2466.2 | 3097.0 |
  | row 29 [1, 2048^2, 256] n2 eval | 17219.6 | 15132.7 | 23720.4 |
  | row 37 [1, 2048, 768] n2 eval | 90.0 | 71.7 (final run: within 2 %) | 69.5 |
  | row 33 [1, 768^2, 512] n2 train | 19651.9 | 17608.0 | 21510.1 |

  Over the 72 fp32 cases: geomean 0.94x the old Triton time (0.75-1.05x; the D 128 rows unchanged). Accuracy (`acc.Tr.final.jsonl`):
  Triton 253 GREEN / 51 YELLOW / 0 RED (before: 252 / 51), `cuda` 255 / 49 / 0, 3 skipped (gold OOM). Final ladder on Transition
  (`spd.Tr.final.jsonl` against the PyTorch baselines): Triton < PyTorch misses 9 -> 1 (row 37 L 1536, 1.07x; L 1024 / 2048 fixed).
* **`cuda` row 209 (D 64 n 2 master training) -> generic kernels (`cuda:b200_sm100` 2).** The wide build's backward was YELLOW there
  (1.03-1.09x the tolerance, `grad:x` / `grad:ln_in.weight`) where Triton is GREEN, which the round's accuracy rule does not allow.
  `route_sm100a.GENERIC[(64, 2, True)]` now sends it to the generic kernels: 6 / 6 GREEN, 65.4-546.8 us against Triton 73.6-700.3 us
  (still 1.1-1.3x faster; the wide build was 30.6-294.9 us).
* **`cuda` vs Triton**: the re-tuned Triton fp32 now beats the generic `cuda` kernels on row 37 at L 128-384 ([1, L, 768] n 2 fp32
  eval: 38.8 vs 34.7 us). Profiled: `cuda` = expand + SwiGLU tcgen05 GEMM 12.4 us + cuBLAS squeeze 7.5 + LayerNorm 2.6 + a 1.7 us copy;
  launch / latency bound at 128-384 rows. Open (gap table below).

### LayerNorm

* **sm_100a row kernels (`cuda:b200_sm100` 2).** The B200 port branch's sub-warp row kernels (`origin/b200/ladder-final`
  `kernels/layernorm/cuda/general/ln_rows.cu`: a row on TPR <= 32 lanes x CH 16-byte vectors, shuffle-only reductions, resident grid,
  per-CTA dw / db partials + one fixed-order column pass) are now an sm_100a-only extension (`kernels/layernorm/cuda/sm100/`,
  `layernorm_rows_sm100`; built for `compute_100a` alone, prebuilt under `target.arch == "sm100"`). Added there: (16, 3) / (8, 3) / (32, 6)
  layouts (bf16 384, fp32 768), RPT 4 / 8 and 4-warp CTAs, a non-persistent forward (`ctas` 0), layout checks. `b200.SM100_ROWS` routes
  bf16 D 64 (every size, eval + train), D 128 / 256 (eval + train) and D 384 (eval) from 65 536 rows to them. D 64 bf16 training
  (row 189, master) is no longer refused: 14.2 / 24.4 / 34.7 / 51.2 us at L 128-512 against Triton 18.3 / 26.5 / 42.9 / 59.3, L 768
  96.1 against 94.2 (tie). Training D 256 (rows 192 / 193): 7-14 % faster than before (e.g. L 768 307 -> 274 us), still behind Triton
  (245 us). Accuracy (`master`, `cuda`): 281 GREEN, 0 YELLOW / RED, 53 unsupported (`SLOWER`, below). Over the 281 served master cases
  the route is 0.99x the old time (0.77-1.31x: the >1.05x cases are single 2 us timer steps at 6-20 us).
* **Refused bf16 pair inference stays refused** (53 cases: D 64 from 65 536 rows, D 128 / 384 from 147 456, D 256 from 65 536). Measured
  ([L^2, D] bf16, fp32 affine, us; `lnbench.py`):

  | D, L | old route | sm100 rows (best launch) | Triton | cuEquivariance |
  |---|---|---|---|---|
  | 64, 2048 | 219.1 | 178.3 | 167.9 | 172.0 |
  | 128, 2048 | 434.2 | 350.2 | 323.6 | 325.6 |
  | 256, 2048 | 878.6 | 692.3 | 618.3 | 630.8 |
  | 384, 2048 | 1359.9 | 1201 (rpt 2) | 931.8 | 1130.6 |

  Tried and rejected (slower than the default `(8, 2, 1, 2, 16)` at D 128 L 2048, 350 us): 2 / 4 / 8 rows a thread (376-516 us),
  4-warp CTAs, non-persistent one-tile-a-CTA grids (362-467 us), the `ld.global.nc.L1::no_allocate` / `st.global.cs` hints on 2-byte
  rows (362 us), the H100 `pipe` direct2 forward with bf16 weights (377 us). Triton's kernel (64 rows x 64-column K tiles, a second
  pass over the L1/L2-resident row) runs at 6.6 TB/s; a `torch` copy of the same bytes takes 739 us here.

### LayerNorm round 3 (2026-11-17): B200 `cuda` against H100 `cuda` (`cuda:b200_sm100` 3)

Target of the round: B200 `cuda` median <= H100 `cuda` median / 1.5 on every case both serve (graph timing, L2 flushed). 599 LayerNorm
cases are served by both cards. Records: `$W/b2511/agents/r3-layernorm/_scratch/out/spd_skew.jsonl`, `spd_small.jsonl` (speed, `cuda`),
`acc_skew.jsonl`, `acc_small.jsonl` (accuracy); the other cases from the tokens of both cards.

| b200 / h100 over the 599 cases | not >= 1.5x faster | slower than H100 | worst | geomean |
|---|---|---|---|---|
| before (`cuda:b200_sm100` 2) | 397 | 227 | 1.302 | 0.795 |
| after (`cuda:b200_sm100` 3) | 375 | 215 | 1.302 | 0.775 |

By B200 time (after; the misses are the cases above 0.667):

| B200 median | eval: cases / misses (before) | train: cases / misses (before) |
|---|---|---|
| > 100 us | 74 / **0** (9) | 76 / **0** (10) |
| 33-100 us | 29 / 3 (3) | 31 / 3 (5) |
| 16.5-33 us | 17 / 5 (7) | 69 / 67 (77) |
| 8.3-16.4 us | 53 / 47 (48) | 90 / 90 (79) |
| <= 8.2 us (one launch) | 160 / 160 (159) | -- |

**What bounds the small cases.** The measurer's events tick in 2.048 us steps on this card and an empty graph replay is 3 ticks:
`a.add_(1)` on one element and `torch.cuda._sleep(0 .. 3000 cycles)` measure 6.02 us, `_sleep(4000)` 8.06 us, `_sleep(8000)` 10.1 us
(`_scratch/floor.py`). The single-launch LayerNorm forwards (2.5-2.9 us of kernel, CUPTI) land on the 8.06 us tick (some replays on
6.02), against 6.2-6.8 us on the H100: the ratio there is 0.9-1.3 whatever the kernel does, and 1.5x would need a 4.2 us replay,
below the empty replay. A training step is three launches (forward, backward, column pass: 12.2-16.3 us here, 10-16 us on the H100).
Everything at or below ~33 us is in this regime: 369 of the 375 misses. Fusing the column pass into the backward was not built: the
partial rows are per CTA, so the last CTA would have to add e.g. 256 partial rows x 768 columns alone (the pass spreads them over 48 CTAs).

**Built / changed.**

* **Skewed-row kernels** (`kernels/layernorm/cuda/sm100/ln_skew_sm100.cuh`, in the `layernorm_rows_sm100` extension, sm_100a only): bf16
  rows of an odd width (267 / 651: a row is not a whole number of 16-byte vectors; its start cycles through 8 offsets). The H100-shaped
  route staged them through shared memory (forward, 3.3-4.1 TB/s here) and ran the backward a warp per row, element-wise (1.4-2.2 TB/s).
  Now a row is TPR lanes x CH aligned 16-byte vectors in registers (the aligned row kernels' form): the first / last vector of a row is
  loaded whole and masked to the row's elements, and stored element by element (its other elements are a neighbour row's); a CTA takes
  the rows of one phase (row mod 8), so the affine parameters are one phase-shifted copy in shared memory and the dw / db sums of the
  backward are per-thread registers folded like the aligned kernels'; one fixed-order column pass. A tensor that ends mid-vector at the
  end of its storage is padded on the Python side (the kernels read whole vectors). Launches: 267 on 8 x 5 lanes x vectors (12 CTAs a
  SM forward, 4 backward), 651 forward 8 x 11 (6), backward 32 x 3 (6). Routed (`b200.SM100_SKEW`) from 16 384 rows in training, from
  65 536 rows in inference (the L 128 forward is 0.7-4 us behind the staged kernel).

  Kernel durations (us, L2 flushed; old = staged forward / warp-per-row backward + its two column passes):

  | [L^2, D] bf16 | fwd old | fwd skew | bwd old | bwd skew |
  |---|---|---|---|---|
  | 267, L 128 / 256 / 512 / 768 / 2048 | 11.1 / 26.6 / 79.7 / 165 / 1095 | 11.8 / 23.9 / 67.7 / 140 / 945 | 28.0 / 83.2 / 287 / 627 / -- | 14.0 / 37.1 / 116 / 246 / -- |
  | 651, L 128 / 256 / 512 / 768 | 18.1 / 55.5 / 200 / 443 | 22.6 / 49.0 / 149 / 317 | 44.5 / 134 / 481 / 1057 | 26.5 / 74.2 / 257 / 565 |

  Measured cases (b200 / h100, before -> after): master train 267 L 128-768 0.90 / 0.82 / 0.88 / 0.88 / 0.86 / 0.86 -> 0.76 / 0.53 / 0.48 /
  0.44 / 0.43 / 0.42; master train 651 0.83 / 0.85 / 0.84 / 0.84 / 0.84 / 0.84 -> 0.65 / 0.56 / 0.52 / 0.51 / 0.50 / 0.50; master eval 267
  L 256-2048 0.89 / 0.86 / 0.81 / 0.78 / 0.76 / 0.74 / 0.73 / 0.70 -> 0.77 / 0.72 / 0.69 / 0.66 / 0.65 / 0.64 / 0.63 / 0.61; master eval 651
  L 256-768 0.84 / 0.82 / 0.82 / 0.82 / 0.82 -> 0.72 / 0.64 / 0.62 / 0.60 / 0.60. Accuracy GREEN on the 27 cases (rows 6 / 11 / 194), the
  same y / dx errors as the old route (`_scratch/skew_check.py`: 324 shapes incl. odd row counts, views, missing weight / bias; dw / db
  within 2e-6 relative; replays bit-identical).

  Design notes (measured, [589824, 267] / [589824, 651] forward): a first build kept each thread's affine values in registers with the 8
  phases inside one CTA: 168 registers (251-255 backward), so only 384 (256) threads a SM were resident; 170 / 360 us against 140 / 317
  now. A not-inlined element-wise tail load inside the row loop (never taken) and float masks: 474 -> 359 us at 651 without them; the tail is
  now the host's contract. Partial stores as at most three aligned 2 / 4 / 8-byte stores of packed words: slower (360 -> 524 us, register
  pressure). Dropping the masks altogether (wrong result) would be another 15 %.
* **fp32 D 768 on the sub-warp row kernels** (`b200.SM100_ROWS`, `sm100.DIRECT_BELOW`): the token streams ran the general kernels (a
  128-thread CTA per row, two column-pass launches in training). Now 32 lanes x 6 vectors, forward on a non-persistent grid below 3 072
  rows. Training [1, L, 768] 16.3-24.4 -> 12.2-14.2 us (b200 / h100 1.00-1.26 -> 0.68-0.87), inference [5, L, 768] from L 384 10.1-20.4 ->
  8.1-18.3 us (0.73-0.99 -> 0.63-0.86). 30 cases GREEN.
* **Tried and not adopted.** A TMA bulk-copy forward (16-32 KB row tiles into shared memory, mbarrier ring; the next idea of the previous
  round): 1075 us at [589824, 651] against the staged kernel's 441, 384 against 345 at [4194304, 128], 283 against 325 at
  [1048576, 384] where Triton is at 240 -- dropped (the prototype is in the branch's stash). Staged-forward geometry on the B200 (2 / 4 / 8
  warps, 4-98 KB tiles): at most 10 % (442 -> 397 us at 651). The skewed kernels for the 451 / 831 / 833 token streams: 10.1 us against
  6.1-8.1 (forward), 31-41 against 16-20 (training; an earlier build). bf16 D 128 / 256 on the row kernels from 16 384 rows, bf16 D 768 and fp32 / bf16
  D 384 token streams on them: single timer steps either way.

**Still above 0.667 and not launch-bound** (6 cases, 33-100 us): master eval 267 L 384 / 512 (0.72 / 0.69), 651 L 256 (0.72), fp32 train
267 / 651 L 128 (0.69 / 0.70: the element-wise fp32 backward; a 4-byte skewed build would take them), master train D 64 L 512 (0.67).

### LayerNorm round 3, part 2 (2026-11-17): the 20 us exemption, `cuda:b200_sm100` 4

Counted with the user's exemption (cases whose H100 `cuda` median is below 20 us are exempt; `$W/b2511/vsh100_15.py` rule): 599 cases
both cards serve at the start, 309 exempt, 290 eligible. Records: `$W/b2511/agents/r3-layernorm/_scratch/out/spd_p2{a,b,c}.jsonl`,
`spd_p3{a,b,c}.jsonl` (speed, `cuda`; `spd_p3b` also `triton`), `acc_p2.jsonl`, `acc_p3.jsonl` (accuracy); merged over the part-1
records and the tokens by `_scratch/list20.py`.

| eligible cases (H100 >= 20 us) | both serve | above 0.667 | slower than H100 | worst | geomean |
|---|---|---|---|---|---|
| start of part 2 (`cuda:b200_sm100` 3) | 290 | 66 | 15 | 1.162 | 0.577 |
| after (`cuda:b200_sm100` 4) | 293 | 51 | **0** | 0.973 | 0.555 |

Refused cells with an H100 median >= 20 us: 145 -> 142 (three bf16 pair-inference cells served, below). The others are 51 fp32 / 39
master training cases at L >= 1024 the B200 tokens do not have (not measured on the B200, no refusal), 51 bf16 inference cells where
Triton is faster (below) and 4 fp32 D 64 training cells (Triton faster from 147 456 rows).

**Built / changed** (accuracy: 152 + 105 cases GREEN, the rest the known refusals; `_scratch/skew_check.py` 756 shapes incl. fp32, views,
odd row counts, no weight / bias: 0 bad, replays bit-identical; `_scratch/wide_check.py` bf16 2560: 0 bad, bit-identical replays):

* **Skewed-row kernels for fp32 and the token streams.** `ln_skew_sm100.cuh` is templated on the element type (V = 8 bf16 / 4 fp32
  elements a 16-byte vector, V row phases); fp32 267 / 651 (pair streams) and 451 / 831 / 833 in both dtypes (token streams) train on it
  (`b200.SM100_SKEW`; their inference stays on the old route, which is faster below 65 536 rows). The affine table copy issues all its
  loads before the first store (one load round, not S / threads). Training steps: fp32 [16384, 267] 36.7 -> 30.5 us, [16384, 651]
  73.6 -> 57.1, fp32 [L, 833] 22.4 -> 16.2 (L 384) / 18.2 (L 768), [128, 831] 20.4 -> 14.1, master [L, 831 / 833] 22.4 -> 16.2-20.3.
* **First tile before the table** (`ONE` build): when the grid covers every row tile of a phase, a CTA issues its tile's loads before
  the affine-table copy and the barrier, so the two cold DRAM round trips overlap (fp32 [768, 833] training 18.2 -> 16.2 us). Applied
  to every multi-tile launch it was slower (bf16 [16384, 651] training 44.8 -> 59.2): the dispatcher picks the build by grid coverage.
* **Next-tile prefetch** (`PF`, `sm100.SKEW_PF`, `sm100.ROWS_PF`): a resident forward keeps the next tile's loads in flight while it
  reduces and stores the current one. Taken for 651 (bf16 [65536 / 262144, 651] inference 46.8 / 145 -> 42.9 / 131 us; fp32 651 forward
  at 8 CTAs a SM: [65536, 651] 77.7 -> 63.4) and the bf16 384 row kernels (16 x 3: [65536 / 147456 / 589824, 384] 28.4 / 54.9 / 188 ->
  26.5 / 51.0 / 172 in the A/B; the measurer's L 256 medians 28.6 -> 28.3). Not taken: 267 (best launch unchanged; at 12 CTAs a SM the
  extra registers cost 10-45 %), bf16 64 / 128 / 256 (a timer step slower at 1-4 of 5 sizes).
* **bf16 2560 training on the row kernels** (`SM100_ROWS`, layout 160 lanes x 2 vectors: one row a five-warp CTA, row sums through
  shared memory; backward four rows a thread, one CTA a SM, one partial row a CTA, one column pass). The general kernels it replaces wrote a
  partial row per row and ran two column passes (four launches; 10.8 us backward at 384 rows). Training [1, L, 2560] L 128-768: 22.4 /
  24.4 / 28.5 / 28.5 / 32.6 / 32.7 -> 18.2 / 18.2 / 20.3 / 20.3 / 22.4 / 24.4 us (b200 / h100 1.06-1.15 -> 0.77-0.86: no longer slower).
  A first build (64 lanes x 5, one row a two-warp CTA, partial row per row) was 28.5 / 42.8 us at L 384 / 768 and was replaced.
* **Refusal spans**: bf16 256 inference is served from 65 536 to 147 455 rows ([65536, 256] 20.2 us against Triton 22.0; r4 / r5 L 256),
  bf16 128 inference up to 262 143 rows ([147456, 128] 20.4 against 20.4; r3 L 384: 0.62 of the H100).

**Tried and dropped (measured):**

* **The backward as one thread-block cluster** (16 CTAs, dw / db summed over DSMEM in rank order, no partial rows, no column-pass
  launch) for the short token streams: 22.3-42.9 us against 16.3-18.3 for the three-launch step ([384 / 768, 833] bf16 / fp32, [1024,
  451]): 16 SMs stream the rows one dependent tile after another. One launch less does not pay for 1/9 of the machine.
* **Fused column tail** (part 1's stash: in-kernel barrier, last CTA adds the partial rows) and **PDL on the column pass**: 2 us slower /
  no gain (part 1's measurement, stash `r3-layernorm part 2: fused column tail`).
* **col_reduce with 16 loads a round** (the predecessor's uncommitted edit): fp32 768 training 14.2 -> 16.2 us and bf16 D 64-256 one
  timer step slower: reverted.

**What is still above 0.667 (51 cases) and why.** All but six are training steps or launches at 14-31 us on the B200 (H100 20-43 us):
three launches (forward, backward, one fixed-order column pass) with an L2-flushed cold start each. Kernel durations of such a step
(CUPTI, [384, 833] master): forward 4.7, backward 5.0, column pass 3.4 us; the graph adds ~3 us; 1.5x of the H100's 20.7 would be 13.8
us. The H100 runs the same three launches (its column pass is a PDL dependent). The measurer's 2.05 us tick moves single cases across
0.667 between runs (fp32 768 L 640: 14.3 / 16.3 in two runs). Cases above 33 us: master eval 267 L 384 / 512 (0.72 / 0.69: skewed
forward at 3.6 TB/s; the prefetch build is slower for 267), 651 L 128 eval (0.80), master train 267 L 128 (0.71), D 512 L 128 train
(0.72-0.73, the pipe kernels), D 128 L 256 train (0.72).

**H100 technique audit (LayerNorm `cuda`)**

| H100 (docs/h100/layernorm.md) | B200 |
|---|---|
| pipe kernels (warp-per-row-group, cp.async ring, persistent) | used: the pipe extension's sm_100 build serves bf16 D 128 / 256 below 65 536 rows, D 384 / 512 training |
| `direct2_fwd` PDL launch below 8 MB | used where the pipe route runs (same source); master has an fp32 affine, so `direct_fwd` as on the H100 |
| `bulk_fwd` (cp.async.bulk 16 KB tiles, 8-48 MB inference) | not applicable under `master` (fp32 affine, as on the H100); a TMA-bulk forward for the row kernels lost (part 1: 1075 vs 441 us at [589824, 651]) |
| `directch_fwd` from 48 MB | not used: the sm100 row kernels are faster at those sizes, and the largest cells are refused for Triton |
| `pipe_reduce` as a PDL dependent | used on the pipe route; sm100 `col_reduce` plain launch (PDL measured no gain, part 1) |
| odd widths: staged forward + warp-per-row backward + column pass | replaced by the skewed-row kernels (faster, measured above and in part 1) |
| general kernels for 2560 (4 launches in training) | replaced by the 160 x 2 row kernels (3 launches) |
| (mean, rstd) as one float2 | not ported: 8 bytes a row against 1-5 KB of row traffic |
| training launch count 3 | 3 on every B200 route (was 4 for 2560 and fp32 768 before round 3) |

### LayerNorm round 6 (2026-11-17, BRIEF6): the true state, then the training launches (`cuda:b200_sm100` 20)

Measured first (every requirement case, `cuda` / `triton` / `pytorch`, one measurer run per (length, dtype) group on an exclusive slot of
GPU 2, graph timing, L2 flushed; `$W/b2511/agents/r3-layernorm/_scratch/out/s6_before`, `s6_after2`, `s6_final`, tables by `_scratch/s6tab.py`,
per-case comparison `_scratch/s6cmp.py`). 844 cases; the H100 tokens give the 1.5x rule (H100 `cuda` >= 20 us).

| | (1) eligible | above 0.667 | slower than H100 | worst | ladder cuda > triton, L >= 384 (L <= 256) | cuda refused |
|---|---|---|---|---|---|---|
| before (`cuda:b200_sm100` 4) | 293 | 46 | 0 | 0.923 | 81 (3) | 78 |
| after (`cuda:b200_sm100` 20; final run on the merged base) | 297 | 46 | 0 | 0.899 | 14 (1) | 74 |

The old training rows at L >= 1024 (split off the requirements on 2026-11-12; H100 tokens exist, never measured on the B200; the old rows in
`_scratch/cfg/p4big/req_oldtrain.csv`): 116 cases, accuracy GREEN on all, (1) above 0.667: 25 -> 23 of 87 -> 90 eligible (worst 0.965, none slower
than the H100), ladder 27 -> 11, refused 3 -> 0.

**Built / changed** (accuracy: 764 GREEN, 0 YELLOW / RED, 74 unsupported (the bf16 pair-inference refusals), 2 skipped and 4 errors on both
`pytorch` and `cuda` (fp32 eval D 256 L 2048 S 5: the gold reference does not fit beside the other agents' jobs); repeated training steps bitwise
equal, `_scratch/detcheck.py`):

* **fp32 training of D 64 / 128 / 256 / 384 / 512 on the row kernels** (`b200.SM100_ROWS`; D 256-512 from 65 536 rows: at 16 384 rows the
  pipe kernels stay a timer step ahead). The forward is the non-persistent grid on four-warp CTAs (`sm100.FWD_DIRECT4`), the backward two
  rows a thread on four-warp CTAs at four CTAs a SM (`sm100.BWD_LAUNCH`; was two-warp CTAs, the most vectors a thread). Training steps
  (us, before -> after, Triton): D 128 L 768 270.1 -> 239.4 (239.5), L 512 130.9 -> 114.6 (116.6), D 256 L 768 520.0 -> 460.8 (456.8),
  D 384 L 512 350.1 -> 315.2 (327.7), D 512 L 512 448.4 -> 417.6 (415.9), atom D 128 [8192 x 48] 186.3 -> 165.7 (165.8).
* **fp32 D 64 training served** (was refused from 147 456 rows): L 384-768 44.9 / 66.4 / 94.0 / 128.7 us against Triton 45.1 / 69.6 / 98.1 /
  130.9; old rows L 1024-2048 now served too.
* **bf16 backward launches** (`BWD_LAUNCH`: D 64 4 x 2, D 128 8 x 2, D 256 16 x 2, all two rows a thread, four-warp CTAs, 4 a SM): master
  D 64 training L 512 / 768 52.5 / 96.1 -> 44.9 / 79.9 us; master D 256 L 768 274.4 -> 268.2 (Triton 243.7: still behind).
* **Lean forward** (`fwd_lean`, `ctas` -1: one-warp CTAs, RPT 2 / 4 rows a thread, all row loads first, the affine values read at use one
  16-byte chunk at a time, `__launch_bounds__(32, 64)`): bf16 384 / 512 inference from 65 536 rows (`sm100.LEAN_FROM`); master eval D 512
  -7 % on every size (e.g. L 1024 350.0 -> 325.4 us, Triton 332.6; L 2048 1369 -> 1268), D 384 [147456 / 589824 / 2359296] 51.0 / 172 / 681 ->
  46.9 / 156 / 590 in the sweep (Triton 44.8 / 143 / 538: those cells stay refused).
* **D 128 inference from 16 384 rows on the row kernels** (`b200.SM100_EVAL_FROM`): the atom streams [5 x 4096-7168, 128] 12.1 -> 10.0 us
  (one timer step, both dtypes); fp32 [2949120, 128] 452 -> 444 (Triton 445).

**Refused bf16 pair inference (74 cases) stays refused, measured reason.** The B200 copy ceiling for the same bytes (`_scratch/fwdlab`,
non-persistent 256-thread copy, 16-byte vectors) is 6.6-6.9 TB/s and Triton's LayerNorm (one-warp programs of 8 rows, `ld/st.global.v4.b32`,
no cache hints: read from its PTX) runs at it: [4194304, 128] Triton 321 us, copy 316. The best row-kernel forward measured 329-346 us (lean
r4, graph timing); load / store hint variants (`ld.global.nc.L1::no_allocate.L2::256B`, `L2::128B`, `L1::evict_first`, `st.cs`,
`st.L1::no_allocate`, `st.wt`, an L2 evict-first policy) move it by <= 1 %; layouts (TPR x CH), 1-4 rows a thread, 1-4 warps, resident /
non-persistent / lean grids: best 2-5 % behind Triton (D 64 [4194304] 169 vs 168 in the lab, 174-176 in the measurer). The registers the
register-resident row needs (68-126 a thread, `cuobjdump`) cap residency below Triton's.

**Still above 0.667 (46 cases), why.** 41 are training steps at 14-30 us (three launches: forward, backward, column pass) where the B200
tick (2.05 us) and the launch floor dominate (H100 20-43 us); 5 are single forwards of 14-28 us (D 651 L 128, D 384 L 128 fp32, D 267 L 256,
D 128 atom). The 1.5x line for the largest of them (master train D 256 / D 384 L 128, 24-28 us) would need 17.5-21 us.

**Ladder still violated (14 at L >= 384, the class is a ladder exception):** master training D 256 L 384-768 (1.03-1.10x Triton: the bf16
backward, best launch in the sweep 266 vs 244 us), fp32 eval D 651 L 512-768 (1.02-1.04, staged forward),
master eval D 512 L 2048 (1.022), atom D 128 fp32 eval [5 x 4096 / 7168] (one timer step).

Not kept: PF (next-tile prefetch) builds for the other layouts (built, no gain in the sweep at any size, removed again), `fwd_lean` for
bf16 64 / 128 / 256 (level with the resident grid in the measurer). TMA / bulk-copy producers were not retried (part 1: 1075 vs 441 us; one
issuing lane is capped at ~1 TMA a ~600 clk).

### LayerNorm round 7 (2026-11-18): programmatic dependent launches (`cuda:b200_sm100` 21)

Before = the round-6 final run (`_scratch/out/s6_final`), after = `_scratch/out/s6_r7b` (every requirement case, `cuda` / `triton` /
`pytorch`, one measurer run per (length, dtype) group, exclusive slot on GPU 2, graph timing, L2 flushed; tables `_scratch/s7_r7b.txt`,
per case `_scratch/s7_cmpb.txt`). H100 LayerNorm tokens are unchanged by the 2.5.13 / 2.5.14 merge.

| | (1) eligible | above 0.667 | slower than H100 | worst | ladder cuda > triton, L >= 384 (L <= 256) | cuda refused |
|---|---|---|---|---|---|---|
| before (`cuda:b200_sm100` 20) | 297 | 46 | 0 | 0.899 | 14 (1) | 74 |
| after (`cuda:b200_sm100` 21) | 297 | 45 | 0 | 0.893 | 15 (1) | 74 |

Per case: 68 `cuda` medians faster by more than 5 % (all small training steps, one 2.05 us timer step each, e.g. master train [512, 833]
18.2 -> 14.2 us, [768, 451] 18.2 -> 14.9, [16384, 64] 16.2 -> 14.1, [128, 2560] 18.2 -> 16.2), 10 slower by one timer step (fp32 768 / 384
training at 384-768 rows, 12.1-14.3 -> 14.0-16.2). The tick flips are noise of the timer: the same A/B on cells whose route is not touched
(`pipe`, fp32 [384 / 512, 384]) flips 12.1 <-> 14.1 us between runs. The extra ladder pair (master eval D 512 L 1536, 1.022) is on an
unchanged kernel (large activations take no PDL build). Accuracy: 764 GREEN, 0 YELLOW / RED, 74 unsupported (the bf16 pair-inference
refusals), as in round 6; `_scratch/detcheck.py` bitwise equal repeated training steps (BAD 0), `_scratch/skew_check.py` 756 shapes 0 bad.

**Built.** Every kernel of the sm100 extension launches through `mwln::launch` (`cudaLaunchKernelEx`, `ProgrammaticStreamSerialization`)
when its activation is under 8 MB (`pdl_on`; bits of `pdl_mode`, `set_pdl` for A/B): forwards wait at entry and then trigger; the
backward loads x (saved by the forward) before `griddepcontrol.wait`, the statistics (the forward's) and dy (the previous kernel's)
after it, then triggers; the column pass is a dependent of the backward. Kernel durations of the step (CUPTI, eager, L2 flushed, master
[384, 833]): forward 4.35, backward 5.09, column pass 3.17 us; the graph adds ~6 us (an empty replay measures 6.02 us).

**Tried and dropped (measured):**

* **The wait / trigger in every build** (first version): 8-40 % slower on the large HBM-bound training steps (master train D 651 L 512
  386 -> 536 us, D 256 L 768 268 -> 300) and 1.7x on the lean forward (master eval D 512 L 1024 325 -> 545). The PDL instructions are now
  a separate build (`PDL` template flag) for the (rows a thread, warps) launches the tables use (`pdl_rl`) and only taken under 8 MB; the
  large cases run the unchanged builds (A/B: [589824, 256] 268.2 vs 269.9, [262144, 651] 389.0 vs 389.0 us).
* **PDL on the forward alone** (bit 1): no change in the measurer (the forward is the graph's first node); kept under 8 MB for callers
  whose forward follows another kernel.
* **bf16 D 256 backward launches** for the ladder pair master train D 256 L 384-768 ([589824, 256], step us; base 16 x 2, 2 rows, 4 warps,
  4 CTAs a SM = 268.1): 2 rows at 2 / 3 / 5 / 6 / 8 CTAs 327.5 / 286.7 / 309.1 / 292.8 / 272.2, 1 row at 4 / 8 CTAs 294.4 / 298.8, 4 rows at
  1 / 2 CTAs 433.6 / 319.3, two-warp CTAs 321.3-330.7, 8 x 4 layout 270.3-311.1, 32 x 1 286.5-315.2. Nothing beats the current launch;
  Triton 243.7.

**Still above 0.667 (45), why.** The same regime as round 6: training steps of 14-30 us made of three launches (forward, backward,
column pass) whose kernels are each 2.5-11 us of mostly cold-DRAM latency and launch ramp, plus the ~6 us graph floor; PDL hides one
timer step of the launch gaps, 1.5x of the H100 would need 2-4 more (e.g. master train D 833 L 384-768: 16.2 us, the line is 13.5-14.0,
i.e. the 12.2 us tick). A fused column tail (round 3 part 2: in-kernel barrier, last CTA) measured as long as the separate pass.

### LayerNorm round 8 (2026-11-18): the column pass inside the backward (`cuda:b200_sm100` 22)

Before = round 7 (`_scratch/out/s6_r7b`, same LayerNorm code as the merged 2.5.14 base), after = `_scratch/out/s6_r8` (every requirement
case, `cuda` / `triton` / `pytorch`, one measurer run per (length, dtype) group, exclusive slot on GPU 2; tables `_scratch/s8_r8.txt`,
per case `_scratch/s8_cmp.txt`).

| | (1) eligible | above 0.667 | slower than H100 | worst | ladder cuda > triton, L >= 384 (L <= 256) | cuda refused |
|---|---|---|---|---|---|---|
| before (`cuda:b200_sm100` 21) | 297 | 45 | 0 | 0.893 | 15 (1) | 74 |
| after (`cuda:b200_sm100` 22) | 297 | 45 | 0 | 0.915 | 16 (1) | 74 |

Per case: 13 `cuda` medians faster by more than 5 %, all fp32 D 768 training (rows 202 / 203, L 256-768: 14.1-16.2 -> 12.1-14.2 us; L 640 /
768 0.784 -> 0.688 / 0.677 of the H100, one 2 us tick above the line); the 11 slower ones are on routes this round does not change (eval
D 768 / 128, master train D 833 L 768 16.2 -> 18.1, D 128 L 256 28.7 -> 30.5: single timer ticks; the worst case, master train D 256 L 128,
is the unchanged `pipe` route, 23.5 -> 24.0). Accuracy: 764 GREEN, 0 YELLOW / RED, 74 unsupported (as round 7); `_scratch/detcheck.py`
bitwise equal repeated steps incl. the fused cells (BAD 0); `_scratch/fuse_check8.py` 246 shapes (bf16 / fp32, rows / skew, RMS, no affine,
retain_graph, 200 graph replays) 0 bad, dw / db within 1.6x and dx within 2.1x of the separate-pass error against fp64 (the dx bits differ:
other FMA contraction in the build without the partial sums); `_scratch/skew_check.py` 756 shapes 0 bad.

**Measured first: what the column pass costs** (`_scratch/r8bound.py`, graph timing): the same training step without affine gradients
(no partial rows, no column pass) is 4-6 us shorter: [384, 833] master 14.3 -> 10.1, [640, 768] fp32 16.2 -> 10.0, [384, 2560] master
18.2 -> 12.1, [16384, 256] master 24.4 -> 20.3 (the `pipe` route). The H100 runs the same three launches (forward, backward, PDL column pass).

**Built: the fused backward** (`ln_common.cuh` `col_role`, `rows::bwd` / `skew::bwd` template `FUSE`): extra column CTAs in the backward's
grid sum dw / db concurrently with the dx CTAs (which drop their partial sums); a column range (16 lanes x one 16-byte vector each, two
row slots a warp; odd widths read the aligned vectors and realign them with a shuffle from the next lane) is one thread-block cluster of K
CTAs, each the rows of one group; the K partials are added over DSMEM after a cluster barrier in rank order. No partial rows in memory,
no counters, no column-pass launch: two launches a step, bitwise deterministic. Taken where it measured faster (`sm100.FUSE_K`): fp32 768
below 4096 rows, K 16.

Tried on the way (A/B `_scratch/r8ab.py`, graph timing, us, `k0` = separate pass):

| build | [384, 833] master | [768, 833] master | [640, 768] fp32 | [384, 2560] master | [16384, 64] master |
|---|---|---|---|---|---|
| separate column pass (round 7) | 14.2 | 16.2 | 16.2 | 20.2 | 14.1 |
| v1: column CTAs, scalar loads, last-arriving CTA of a range adds the slab rows (counter behind rstd, zeroed by the forward), K 16 | 14.1 | 18.2 | 12.1 | 26.4 | 51.0 |
| v1, column CTAs return at once (lower bound) / skip the slab + counter tail | 10.0 / 14.1 | 12.1 / 16.2 | -- | 14.1 / 20.3 | 10.0 / 46.8 |
| v2 (kept): 16-byte loads, cluster DSMEM fold, K 16 | 16.2 | 22.3 | 14.1 | 24.4 | 28.5 |
| v2, column CTAs only meet at the cluster barriers | 12.1 | 12.1 | 12.1 | 18.4 | 12.1 |

The fold itself (counter tail or cluster barriers) costs ~2 us and the column loads of the bf16 odd widths another 2-4 us, so the
skewed widths, bf16 2560 (one dx CTA a SM: the column CTAs do not fit beside it) and the 16 384-row cells (too many rows a column CTA)
stay on the separate pass. fp32 768 takes v2 (v1 was within one tick; v2 needs no forward change and no counter).

**Also measured and not taken:**

* **(b) bf16 D 256 backward with the next tile in flight** (`rows::bwd<PF>`: x / dy / statistics of tile t + grid loaded before tile t is
  reduced, Triton's two-stage schedule; Triton's B200 config is one-warp programs of 4 rows x 256, 8 a SM, `num_stages` 2, one atomic
  flush a program): every launch slower, [589824, 256] best 300.8 vs 270.1 us without (Triton 243.6), [262144, 256] 145.3 vs 130.8 (118.8).
  The template stays, no dispatch builds it.
* **(c) four-warp lean forward** (`fwd_lean` LW 4, `__launch_bounds__(128, 16)`: a 32-register cap so 2048 threads a SM fit; one-warp CTAs
  stop at 32 a SM): spills, slower on every refused cell ([1048576, 384] 401-2730 vs 270 us). The best one-warp lean launch stays 2-14 %
  behind Triton on the 74 refused bf16 cells ([4194304, 256] 657 vs 617, [4194304, 128] 336 vs 321, [147456, 384] 46.9 vs 44.9): they stay
  refused. The LW 4 dispatch was removed again.
* The TMA box-count finding (coordinator, Transition part 14) does not apply: no LayerNorm kernel on the B200 route issues TMA (the TMA
  bulk forward of round 3 lost to the register-row kernels, 1075 vs 441 us at [589824, 651]).


### InputFeatureEmbedder

The ladder's 18 `cuda` < Triton cases (worst 1.64x, row 136 fp32 L 1024 train) are the `SWAAtomTransformer` child's: the embedder's own
norms are LayerNorms of [atoms, 128] (1-8k rows, 5-15 us per call) against 100-500 us gaps. Nothing to change on the norm side; the
atom-DiT round (merged in b200-2.5.x) owns them.

## Round 3 (2026-11-17): Transition, B200 `cuda` against H100 `cuda`

Rule of the round: on every requirement case both cards' `cuda` serve, the B200 median at most the H100 median / 1.5 (ratio B200 / H100
<= 0.667; graph timing, L2 flushed). Records (B200 box, `$W/b2511/agents/r3-transition/out/`): `s_*.w2.jsonl` (every Transition case,
`cuda`, re-measured in one session), `s_f32e.w3.jsonl` / `a_f32e.w3.jsonl` (fp32 inference rows 27 / 29 on the final kernel),
`s_e4.w4.jsonl` / `a_gen_e.w4.jsonl` / `a_gen_t.w4.jsonl` (single-stream rows after the in-place squeeze), `s_209.w1.jsonl` /
`a_209.w1.jsonl` (row 209), `cmp.*.txt` (the comparisons). The H100 side is its speed tokens (`judge/tokens/h100_sm90`).

Every Transition requirement case (307), B200 `cuda` median / H100 `cuda` median, before (the B200 speed tokens of 2026-11-16 / 17)
-> after (the re-measure of this round):

| | before | after |
|---|---|---|
| B200 slower than the H100 (ratio > 1) | 63 | **16** |
| ratio > 0.667 (misses 1.5x) | 122 | **97** |
| worst ratio | 1.390 | 1.189 |
| ladder: `cuda` more than 2 % behind the Triton / PyTorch token | 3 (row 37 fp32) / 0 | **0 / 0** |

Part of the "before -> after" on cells this round did not touch is the weight-form fix below and the re-measure itself (one session,
GPU 1 exclusive; the box's event timer steps in 2.05 us, which is 5-15 % of a 15-40 us case); the two were not separated by an A/B.

Cells with a miss before or after (cases; slower than the H100; missing 1.5x; worst ratio; ratio range after):

| row, dtype, mode, shape | cases | slower: before -> after | miss 1.5x: before -> after | worst ratio: before -> after | ratio after |
|---|---|---|---|---|---|
| 25 master eval, D 64 n 2 | 9 | 1 -> 1 | 5 -> 3 | 1.390 -> 1.189 | 0.622-1.189 |
| 26 master eval, D 64 n 4 | 9 | 1 -> 1 | 3 -> 2 | 1.187 -> 1.016 | 0.491-1.016 |
| 27 float32 eval, D 128 n 2 | 9 | 5 -> 0 | 9 -> 3 | 1.353 -> 0.966 | 0.623-0.966 |
| 27 master eval, D 128 n 2 | 9 | 0 -> 0 | 4 -> 2 | 0.973 -> 0.864 | 0.623-0.864 |
| 28 master eval, D 128 n 4 | 9 | 1 -> 0 | 2 -> 1 | 1.039 -> 0.776 | 0.438-0.776 |
| 29 float32 eval, D 256 n 2 | 9 | 3 -> 0 | 9 -> 4 | 1.324 -> 0.982 | 0.591-0.982 |
| 29 master eval, D 256 n 2 | 9 | 0 -> 0 | 1 -> 1 | 0.809 -> 0.720 | 0.491-0.720 |
| 30 master eval, D 256 n 4 | 9 | 0 -> 0 | 1 -> 0 | 0.713 -> 0.640 | 0.454-0.640 |
| 33 float32 eval, D 512 n 2 | 6 | 0 -> 0 | 6 -> 6 | 0.938 -> 0.924 | 0.687-0.924 |
| 35 float32 eval, D 384 n 2 | 9 | 9 -> 4 | 9 -> 9 | 1.293 -> 1.090 | 0.965-1.090 |
| 35 master eval, D 384 n 2 | 9 | 8 -> 1 | 9 -> 9 | 1.178 -> 1.063 | 0.845-1.063 |
| 36 master eval, D 384 n 4 | 11 | 10 -> 1 | 11 -> 11 | 1.194 -> 1.118 | 0.797-1.118 |
| 37 float32 eval, D 768 n 2 | 9 | 6 -> 1 | 9 -> 9 | 1.286 -> 1.014 | 0.768-1.014 |
| 37 master eval, D 768 n 2 | 9 | 6 -> 2 | 7 -> 6 | 1.352 -> 1.037 | 0.484-1.037 |
| 38 master eval, D 64 n 4 | 9 | 0 -> 0 | 3 -> 2 | 0.875 -> 0.964 | 0.471-0.964 |
| 209 master train, D 64 n 2 | 6 | 6 -> 0 | 6 -> 0 | 1.320 -> 0.580 | 0.545-0.580 |
| 210 master train, D 64 n 4 | 6 | 0 -> 0 | 1 -> 1 | 0.756 -> 0.757 | 0.503-0.757 |
| 211 float32 train, D 128 n 2 | 6 | 0 -> 0 | 1 -> 1 | 0.834 -> 0.834 | 0.590-0.834 |
| 211 master train, D 128 n 2 | 6 | 0 -> 0 | 1 -> 1 | 0.854 -> 0.886 | 0.574-0.886 |
| 212 master train, D 128 n 4 | 6 | 0 -> 0 | 1 -> 1 | 0.715 -> 0.715 | 0.443-0.715 |
| 213 float32 train, D 256 n 2 | 6 | 0 -> 0 | 4 -> 4 | 0.779 -> 0.786 | 0.645-0.786 |
| 213 master train, D 256 n 2 | 6 | 0 -> 0 | 1 -> 1 | 0.814 -> 0.814 | 0.501-0.814 |
| 214 master train, D 256 n 4 | 6 | 0 -> 0 | 1 -> 1 | 0.708 -> 0.710 | 0.500-0.710 |
| 217 float32 train, D 384 n 2 | 6 | 3 -> 3 | 6 -> 6 | 1.121 -> 1.085 | 0.838-1.085 |
| 218 master train, D 384 n 4 | 6 | 0 -> 0 | 1 -> 2 | 0.678 -> 0.678 | 0.596-0.678 |
| 219 float32 train, D 768 n 2 | 6 | 1 -> 1 | 5 -> 5 | 1.093 -> 1.093 | 0.618-1.093 |
| 219 master train, D 768 n 2 | 6 | 3 -> 1 | 6 -> 6 | 1.034 -> 1.032 | 0.899-1.032 |

### What changed (`cuda:b200_sm100` 3)

* **fp32 inference at D 64 / 128 / 256 in one launch: `sm100/generic/tg_fused_h16.cu`** (rows 27 / 29). The whole forward (LayerNorm,
  expand, SwiGLU, squeeze, residual) in one persistent tcgen05 kernel with the products on the fp16 tensor cores (`kind::f16`, fp32
  accumulation): fp16 has TF32's mantissa, so xn, the weights and h are rounded as the TF32 path rounds them, at twice the tensor rate
  and half the operand bytes -- the scheme of the H100's `transition_fwd_f32io` / `f32xs`. The fp16 weight forms and four device scales
  (each 1 unless an operand bound passes 30000) are made once per parameter version (`_h16_forms`). xn lives in tensor memory, h in
  shared memory; D 256 keeps one [a | b] accumulator and reads x from global memory (512 TMEM columns and 227 KB of shared memory are
  full). It replaced the TF32 `tg_fused_f32.cu` (D 64 / 128, removed) and the TF32 GEMM chain at D 256.
  Accuracy: row 27 YELLOW 1.0001 / 1.0006 x tol on all 9 lengths -- the H100 tokens' own values (1.00005 / 1.00056); row 29 GREEN 0.855
  at L 128-1024 (H100: YELLOW 1.124). At D 256 past 2^20 rows the kernel measured YELLOW 1.050 (L 1536; H100 1.040, Triton 1.000) and
  1.161 (L 2048; no H100 token, Triton 1.002): `FUSED_H16_MAX_ROWS = {256: 1 << 20}` keeps those two cases on the TF32 chain (GREEN
  1.000 / YELLOW 1.002, 0.95x the H100 time). The rms error of the fp16 kernel equals the TF32 chain's and PyTorch's to four digits at
  every size tried (e.g. [1, 1536^2, 256]: 3.891e-05 each; max 3.25e-04 against 2.96e-04): the light is a tail draw of the max norm.
  Lifting the cap serves the two cases at ~0.58x the H100 time.
* **Its SwiGLU epilogue** folds the scales into two constants and multiplies in fp32x2 (`mul.rn.f32x2`): D 128 3 % faster, same errors.
* **D 256: one LayerNorm statistics pass.** Diagnostic builds with one stage disabled each ([1, 1024^2, D] fp32, us; base 1548 at D 256,
  467 at D 128): the LayerNorm's three passes over the global row were 31 % of the D 256 kernel (consistent with little L1 left beside 224 KB
  of shared memory, every pass an L2 round trip per lane), the residual x loads of the output stage 17 % (D 128: 20 %), the expand and
  the squeeze MMAs 4-5 % each, the SwiGLU 3 % (D 128: 15 %), the output store 4 % (9 %). The statistics now take one pass (sums of
  x - c0 and (x - c0)^2 about the row's first element): 1548 -> 1327 us at L 1024, 410 -> 354 at L 512, same max / rms error.
* **D 64 n 2 training back on the wide two-role build** (row 209): `tbwd_d64.cu -DDXN32` keeps d_xn in fp32 through the LayerNorm
  backward and rounds dx once (the bf16 d_xn was the YELLOW of `grad:x` / `grad:ln_in.weight` that had sent the row to the generic
  kernels). 6 / 6 GREEN (0.61-0.93 x tol; the H100 tokens are YELLOW 1.05-1.09 at L 384-768), 28.8 / 53.1 / 90.0 / 143.4 / 210.9 /
  292.8 us at L 128-768 (generic kernels: 67.4-544.7; H100 51.1-507.2).
* **Inference weight forms read by a CUDA-graph capture as the warm-up made them** (`_capture.param_forms(capture_reads_eager=True)`,
  as the H100 builds): the captured graph no longer re-makes the casts / TF32 forms per replay. Not measured on its own; the master /
  fp32 inference cases under ~40 us re-measured 2-6 us below their tokens (e.g. row 37 master L 256: 30.6 -> 24.5 us).
* **The cuBLAS squeeze of the generic kernels at a few row tiles accumulates in place** onto a copy of x that the LayerNorm launch
  writes (`tg_ln_fwd` XC), instead of `torch.addmm(x, ...)` copying x in a launch of its own (~1 us of a 22-30 us call).
* **Prebuild**: the generic cubins (`fused_generic_sm100a._specs`: bf16 where a (D, n) has no tuned build or `route_sm100a.GENERIC`
  sends it, fp32 for the float32 rows) are in the sm100 plan (`kernels/transition/cuda/__init__.py`); they had been built at first use.

### Tried and rejected

| change | measured | why not |
|---|---|---|
| 2-CTA clusters multicasting the weight stages of `tg_fused_h16` (the predecessor's WIP), 4-CTA clusters | fp32 [1, 1024^2, 128] n 2: single CTAs 485.3 us, pairs 489.5, quads 935.9; D 256: 1556.5 / 1562.6 / 2977.3 | the weight intake is not the bound (the row loads are); a cluster couples its tiles |
| sigmoid through `tanh.approx` (the bf16 kernels' form) in the fp32 kernel | -- | relative error 2^-11: fp32 accuracy |
| the fp16-operand scheme for the fp32 rows at D >= 384 (rows 33 / 35 / 37) | not built | the H100 and Triton run TF32 there with err / tol 1.000-1.03; fp16 rounding has the same rms but an independent max (0.86-1.16 x at D 256), which the accuracy rule does not admit case by case |
| RESID tile / split-K PLAIN + `tg_splitsum` instead of the cuBLAS squeeze at a few row tiles | fp32 [1, 256, 768]: cuBLAS 30.7, RESID 38.9, split 30.7; [1, 1024, 768]: 41.0 / 45.1 / 38.9; bf16 [1, 256, 768]: 24.6 / 26.6 / 24.5 | no consistent win over one launch that splits K inside |
| the small-M PDL chain (`tsx_*`) at D 768 | not built | `tsx_expand` keeps the tile's whole xn (196 KB at D 768) beside its weight K-blocks in shared memory; needs a K-block ring first |

### What still misses 1.5x, and why

* **The replay floor.** One 128-element kernel in a CUDA graph, timed like the measurer: B200 box 6.1 / 7.2 / 8.2 us for 1 / 2 / 3 kernels,
  H100 node 5.3 / 6.4 / 7.8 us (`b200cfg/floor.py`). A case the H100 serves in T us needs the B200 below T / 1.5; with a 6 us floor that is
  not reachable below T ~ 9 us at all, and a single launch has T / 1.5 - 6 us for its whole kernel: 0.8 us at the H100's 10.2 us
  (row 25 L 128), 4.9 us at 16.4 us (row 35), 9 us at 22.6 us (row 37). The B200 kernels there take 6-17 us of device time over
  1-3 launches. These cases are bound by launch / startup latency, not by the tensor rate or HBM.
  Census after this round: of the 97 misses, 51 are cases the H100 serves in under 40 us (40 of them under 30 us; every one of the 35
  cases under 25 us misses).
* **Single stream, [1, L <= 3968, 384 / 768]** (rows 35 / 36 / 37 eval, 217 / 219 train; 61 misses, 14 of them still slower than the
  H100 at 1.01-1.12x): 14-37 us inference in three launches (LayerNorm -> expand + SwiGLU -> squeeze: the `tsx_*` PDL chain at D 384
  bf16, the generic kernels + one cuBLAS launch elsewhere; up to 59 us at fp32 D 768 L 2048), 61-96 us training steps of ~12 launches on the generic kernels. 1.5x needs
  the whole forward in one launch whose device time is under ~5-9 us: a cluster-per-tile kernel that keeps h in distributed shared
  memory (the forward analogue of `tsx_gdx.cu`), and for training the small-M chain (`tsx_gate` / `tsx_gdx` / `tsx_dw`) at D 768 and
  in fp32 (TF32). Smaller steps towards the H100 time (not tried): the copies of the three gradient views per step (dWa / dWb of
  the stacked [dWa; dWb], dgamma of [dgamma | dbeta]) written to their own buffers, PDL between the generic launches (the remaining 14).
* **fp32 pair inference at D 512** (row 33, 6 cases at 0.69-0.92): the TF32 GEMM chain moves 10.8 GB at L 768 (x, xn and h each written
  and read back) in 3.56 ms, ~3.0 TB/s, the rate of a plain copy on this card (2.9 TB/s, round 2). Needs xn and h kept on chip (a fused TF32 kernel in the `tg_fused_h16`
  shape; D 512 needs the y accumulator split over a CTA pair), since fp16 operands are closed by the accuracy rule above.
* **fp32 pair inference at D 128 / 256, L 128-384** (rows 27 / 29, 5 cases at 0.68-0.98, none slower than the H100; plus the two capped
  D 256 cases at 0.95): the kernel reaches 0.59-0.66 from L 512; below that the 6 us floor and the first tile's latency are 20-40 % of
  the call. The next 20 % of the kernel is the residual x that the output stage reloads from global memory per lane (keep the x tile in
  shared memory until the output stage: needs 64 KB the D 128 build does not have beside three weight stages).
* **fp32 pair training at D 256** (row 213, 4 cases at 0.69-0.79), **the smallest lengths of the other cells** (rows 25-29 / 38 eval, 210-214 / 218 train:
  19 cases at 0.67-1.19, two slower than the H100: row 25 / 26 L 128 at 12.2 us against 10.2 / 12.0): the generic TF32 training chain
  (no tuned fp32 training build), and one-launch kernels at 12-50 us whose startup (weights, TMEM, first tile) is the floor argument above.
## Round 3, part 2 (2026-11-17): the small-M chains beyond D 384 bf16 (`cuda:b200_sm100` 4)

Rule since noon: a case the H100 serves in under 20 us is exempt from the 1.5x target (launch floor); from 20 us it still needs
B200 / H100 <= 0.667. Records (`$W/b2511/agents/r3-transition/out/`): `s_n_e.w6.jsonl` / `s_n_t.w6.jsonl` (rows 35-37 eval, 215-219
train), `s_f_e.w7.jsonl` / `s_f_t.w7.jsonl` (their float32 cases on the final builds), `acc.w6.txt` / `acc.w7.txt`, `final2.txt`
(the per-cell table over all 307 cases and every remaining miss).

| | part 1 | part 2 |
|---|---|---|
| B200 slower than the H100 | 16 | **4** (row 25 / 26 L 128: 12.2 us against 10.2 / 12.0, exempt; rows 35 master / 36 L 2048 at 1.06 / 1.04, one 2.05 us timer step) |
| ratio > 0.667 | 97 | 97 (14 of them exempt: H100 under 20 us) |
| ladder / RED | 0 / 0 | 0 / 0 |

| cell | slower than H100: part 1 -> 2 | ratio range: part 1 -> 2 |
|---|---|---|
| 35 float32 eval, [1, L, 384] n 2 | 4 -> 0 | 0.965-1.090 -> 0.784-0.881 |
| 37 float32 eval, [1, L, 768] n 2 | 1 -> 0 | 0.768-1.014 -> 0.716-0.890 |
| 37 master eval, [1, L, 768] n 2 | 2 -> 0 | 0.484-1.037 -> 0.484-0.950 |
| 217 float32 train, D 384 n 2 | 3 -> 0 | 0.838-1.085 -> 0.785-0.975 |
| 219 float32 train, D 768 n 2 | 1 -> 0 | 0.618-1.093 -> 0.593-0.959 |
| 219 master train, D 768 n 2 | 1 -> 0 | 0.899-1.032 -> 0.794-0.929 |

### What changed

* **`tsx_ln.cu` / `tsx_expand.cu` / `tsx_squeeze.cu` take fp32 on the TF32 tensor cores (`-DF32=1`) and ring their K-blocks** (D 768 and
  fp32 do not fit shared memory whole). `tsx_expand.cu -DSK=<n>` splits K over a cluster: the fp32 partials of [a | b] meet in
  distributed shared memory (st.async on the owner's barrier, added in rank order) before the SwiGLU, as the squeeze's did;
  `-DCH=128` is a 128-unit build for 13-16 tiles. The squeeze splits up to 8 ways.
* **`fused_small_sm100a.py`** runs the chain (three programmatic dependent launches) for bf16 D 768 n 2 (up to 6 row tiles), fp32
  D 384 n 2 (up to 16) and fp32 D 768 n 2 (up to 6), whole 128-row tiles. `_pick` takes, among the builds whose clusters fit one wave,
  the least per-CTA operand bytes plus a cost per extra cluster CTA: a second wave costs more than any split saves (fp32 [1, 768, 768]:
  expand (64, 1) 30.7 us, (64, 2) 36.9, (64, 4) 55.3; at 16 tiles of D 768 the chain is 106-162 us against the generic kernels' 57).
  Measured (us, L2 flushed, generic -> chain): fp32 [1, 128, 384] 22.5 -> 16.4-18.3, fp32 [1, 128, 768] 28.9 -> 20.5-24.3, bf16
  [1, 128, 768] 22.5 -> 18.4, fp32 [1, 2048, 384] 32.6 -> 26.5.
* **The generic training step's forward is the chain in those cells** (`fused_generic_sm100a._fwd_launch`): `tsx_expand` stores a
  packed [a | b] (`ldab` / `boff`) for GATE_AB and `tsx_ln` the mean (`save == 2`). fp32 [1, 128, 384] training 61.4 -> 55.1 us,
  [1, 128, 768] 83.9 -> 73.6, bf16 [1, 128, 768] 61.4 -> 55.2.
* **Accuracy rule on the fp32 builds.** The fp32 `tsx_ln` uses `tg_ln_fwd`'s arithmetic (the same xn bits). With the expand's K split
  the max-norm error moved above the Triton / H100 tokens on four YELLOW cases (fp32 [1, 128, 768] out 1.0371 x tol against 1.0336;
  grad:ln_in.weight at [1, 128, 384] 1.0298 against 1.0198), and with the 4-way squeeze at D 384 on two (1.0033 / 1.0067 against
  1.0031): fp32 keeps the expand unsplit (`EXPAND_F32`) and D 384 the 2-way squeeze (`SQUEEZE_F32`). After that every YELLOW case is at
  or below the larger of the old B200 `cuda` and the Triton token (rows 35 / 37 float32: 7 GREEN / 11 YELLOW as before; 217 / 219
  float32 12 YELLOW as before; every master case GREEN). It costs one timer step at fp32 [1, 128, 768] (20.5 -> 24.3 us).
  Two calls are bit-equal on every build (`b200cfg/sxab.py`, `tcheckt.py`).
* The generic backward returns [dgamma | dbeta] packed (no clone launch). Prebuild: the 31 chain cubins are in the sm100 plan.

### Not built / still open

* **A one-launch cluster-per-tile forward.** A cluster is at most 8-16 CTAs and every CTA of a tile takes the tile's whole A operand, so
  one cluster per tile puts 590 KB-1.2 MB of operand intake on each of 8 SMs where the chain spreads a tile over 48-96; it would save
  ~2 us of launch floor and lose more in intake. The misses of rows 35 / 36 / 37 eval are 16-30 us against H100 times of 20-34 us with
  an 8.2 us floor for three launches.
* **The backward of the small-M chain in TF32 and at D 768** (`tsx_gate` / `tsx_dxn` / `tsx_dw`): rows 217 / 219 still run the generic
  backward (two tcgen05 GEMMs 13-18 us, two cuBLAS GEMMs 9-12, `tg_ln_bwds` 5-8, `tg_prep` 3-6.5 of a 55-96 us step). `tsx_dxn` at
  D 768 would put 1.2 MB of intake on each CTA of its 12-CTA cluster (a 24-CTA cluster does not exist); its TF32 form needs MN-major
  TF32 operands. The bf16 D 384 chain's whole step is 28.7 us, which is the size of the gain.
* **Three weight-gradient copies per generic training step** (`aten::copy_` of dWa, dWb, dWs from AccumulateGrad, 5-7 us of the step;
  the wide chain's gradients are taken without a copy). Cause not found.
* Items 2-4 of the brief (a fused TF32 kernel for row 33 and the capped D 256 cases, a tuned fp32 training build for row 213, the
  first-tile latency of rows 27 / 29) were not started.

## Round 3, part 3 (2026-11-17): what bounds the small-M chain (no kernel change kept, `cuda:b200_sm100` stays 4)

CPU gate on the part-2 tree: pyright 0 errors, CPU pytest 3384 passed / 2 skipped; ruff 2 findings (RUF021 in
`attention_pair_bias_b200._pair_tc`, fixed: parentheses only); `ruff format --check` lists 66 files repo-wide (pre-existing on `main`).

Timing below: `b200cfg/sxab.py` (eval, CUDA graph, L2 flushed like `measure.speed`; "amortised" = 20 calls in one graph, no flush),
GPU 1 exclusive. Base (fused_small picks): fp32 [1, 128 / 768 / 2048, 384] 16.4 / 18.4 / 24.6 us (amortised 11.7 / 13.1 / 19.8),
fp32 [1, 128 / 768, 768] 22.5 / 28.7 (15.2 / 22.4), bf16 [1, 128 / 768, 768] 18.4 / 22.5 (10.6 / 15.2). Eager kernel times
(`b200cfg/kl.py`, fp32 parameters as the measurer): fp32 [1, 128, 384] tsx_ln 2.4 + tsx_expand 6.4 + tsx_squeeze 6.4 us;
fp32 [1, 128, 768] 2.8 + 9.2 + 6.3.

| tried | measured | verdict |
|---|---|---|
| xn shared by TMA multicast over 2 / 4 / 8 hidden chunks of one tile (`tsx_expand -DMC`, empty barriers fed by multicast commits; no K split, same bits) | fp32 [1, 128, 384] 18.4 (all builds) against 16.4-18.3; [1, 768, 384] (32, 1, 4) 24.6, others 20.5 against 18.4; [1, 128, 768] 22.5-24.6 against 22.5 | per-CTA operand intake is not the bound at these sizes; reverted |
| h shared by TMA multicast over 2 / 4 output blocks (`tsx_squeeze -DMC`, with and without the K split) | fp32 D 384, L 128 / 768 / 2048: best multicast build 18.4 / 20.5 / 30.7 against the pick's 16.4 / 18.4 / 24.6 | slower; reverted |
| the LayerNorm launch prefetches the three weight matrices into L2 (`cp.async.bulk.prefetch.L2`, 8 KB pieces over its blocks) | +6 to +8 us on every case (fp32 [1, 128, 384] 16.4 -> 24.6, amortised 11.7 -> 16.5), reproduced twice | the prefetches delay the chain; reverted |
| a deeper squeeze ring (6 -> 12 / 24 stages, the whole hidden slice in shared memory) | equal within one timer step on all 7 cases, twice | not the bound; reverted |

The flushed median sits 5-8 us above the warm amortised time on every chain case: the chain is bound by cold-HBM latency and three
launch latencies (graph floor 8.2 us for three kernels), not by the per-CTA intake or the ring depth. 1.5x on rows 35 / 36 / 37 eval
(H100 20-40 us) needs B200 14-26 us, i.e. ~2 us below the chain at every length.

**The "three weight-gradient copies per generic training step" of part 2 do not occur in the measured step.** They came from
`b200cfg/cp.py` building `Transition` with its default bf16 projection weights: the backward's fp32 dWa / dWb / dWs were cast to the
parameters' bf16 (`aten::to` -> `copy_`). With the measurer's fp32 parameters (`float32` / `master`) the step has no copy at all
(`b200cfg/kl.py`: fp32 [1, 128, 384] training = 11 kernels, 49.5 us device: tg_prep 3.0, tsx_ln 2.4, tsx_expand 7.1, tsx_squeeze 6.3,
tg_round_tf32 1.6, GATE 8.1, two cuBLAS weight gradients 3.9 + 4.9, PLAIN dxn 4.7, tg_ln_bwds 5.3, tg_colsum 2.2; measured step 55.1 us
against the H100's 56.5, target 37.7). Folding tg_colsum / tg_round_tf32 / tg_prep into neighbours would save ~5 us of it; the target
needs the TF32 small-M backward chain (part 2, "not built"), which remains open.

Not started in part 3: the two-launch chain (LayerNorm in the expand's prologue: every expand CTA would normalise its whole tile,
~4k cycles per 128 x 384 tile on its critical path, against ~1 us of launch latency saved), the fused TF32 kernel for row 33, the
first-tile work of rows 27 / 29. Every per-case number of part 2 (`final2.txt`) stands: 83 non-exempt misses, 2 non-exempt cases
slower than the H100 (row 35 master / row 36 at L 2048, one timer step), ladder clean, no RED.

### Part 3, second pass (built and measured, none kept)

| built | measured (us, L2 flushed, GPU 1 exclusive, twice) | verdict |
|---|---|---|
| `tsx_gate.cu -DF32=1`: the small-M gate on the TF32 tensor cores (dy on the TF32 grid, Ws^T from `tg_prep` as a K-major B operand, 32-element K-blocks through a ring, weights before griddepcontrol.wait, the saved a / b loaded before it), replacing `tg_gemm` GATE_AB in the generic fp32 training step of the chain cells | gradients bit-identical to GATE_AB, deterministic; fp32 training [1, 128 / 768 / 2048, 384] 55.1 / 63.5 / 100.4 -> 55.3 / 63.5 / 100.8-101.7, [1, 128 / 768, 768] 73.7 / 98.3 -> 73.7 / 96.3 (`b200cfg/tt.py`) | a second implementation of GATE_AB that wins only one timer step at one case; parked on branch `b200r3-transition-tf32gate` (649b5e502) as the first kernel of a TF32 gate -> dxn -> dw chain |
| earlier PDL trigger: `tsx_expand` lets the squeeze launch before its own griddepcontrol.wait (parameter weights only), `tsx_squeeze` loads the residual x before its wait (both bit-identical) | fp32 D 384 L 128 / 768 / 2048 16.4 / 18.4 / 26.6, D 768 L 128 / 768 22.5 / 30.7, bf16 D 768 18.4 / 22.5 (all equal to the base); bf16 D 384 n 2 L 128 / 2048 14.3 / 20.5-22.5 against 14.3 / 20.5, n 4 14.3-16.4 / 28.7 against 14.3 / 28.7 (`b200cfg/early.sh`, base tree `agents/r3-transition-b`) | no gain; reverted |

The gate replaces 8.1 of the 49.5 us device time of the fp32 [1, 128, 384] step with a kernel of the same length (its 12 CTAs ingest
the 196 KB dy tile each), so the step time does not move. What the 37.7 us target needs is the rest of the chain: d_xn + LayerNorm
backward + dgamma / dbeta partials in one cluster launch (`tsx_dxn` in TF32, replacing PLAIN 4.7 + `tg_ln_bwds` 5.3 + `tg_colsum` 2.2)
and the weight gradients without cuBLAS (8.8). Not built: the TF32 `tsx_dxn` / `tsx_dw`, the two-launch forward, the fused TF32 D 512
kernel, the rows 27 / 29 first-tile work.

## Round 3, part 4 (2026-11-17): fp32 inference on the fp16 tensor cores (`cuda:b200_sm100` 5)

Records (B200 box, `$W/b2511/agents/r3-transition/out/`): `s_fin.jsonl` / `a_fin.jsonl` (rows 27 / 29 / 33 / 35 / 37 float32 eval, every
length, final code), `s_tr.jsonl` / `a_tr.jsonl` (rows 33 / 211-214 / 217 / 219 training, master and float32), `a_h16.jsonl` /
`a_h16b.jsonl` / `a_v2.jsonl` / `a_29.jsonl` (the accuracy A/B below). Probes: `b200cfg/bound.py` (pieces of the chain), `b200cfg/sxab.py`
(`-t`: TF32 chain, `-T`: TF32 generic, `-g`: no chain), `b200cfg/runm.py` (a measurer with flag overrides), `b200cfg/accx.py` (err / tol
against the YELLOW limit = the largest of the H100 cuda, B200 / H100 Triton and old B200 cuda tokens), `b200cfg/p4sum.py`.

### First: what the LayerNorm launch of the chain costs (brief item 1, measured before building it)

`b200cfg/bound.py` times pieces of the small-M chain in a CUDA graph with the L2 flushed (us; the timer steps in 2.05 us, an empty graph
is 6.1): the expand + squeeze WITHOUT the LayerNorm launch (the expand reading x as its xn: the lower bound of a LayerNorm folded into the
first GEMM for free) against the whole chain.

| cell | chain | expand + squeeze, no LayerNorm | expand alone | squeeze alone |
|---|---|---|---|---|
| fp32 [1, 128, 384] n2 (TF32) | 16.4 | 16.4 | 10.2 | 10.2 |
| fp32 [1, 2048, 384] n2 | 26.6 | 24.6 | 16.4 | 16.4 |
| fp32 [1, 128, 768] n2 | 22.5 | 20.5 | 14.3 | 12.3 |
| bf16 [1, 128, 768] n2 | 18.4 | 16.4 | 10.7 | 10.2 |
| bf16 [1, 768, 768] n2 | 22.5 | 20.5 | 12.3 | 12.3 |

A free LayerNorm is worth at most one timer step; a real one in the expand's loader costs every expand CTA a pass over its whole
128 x D tile (the H100's `ln_swiglu_gemm` does that; part 3 estimated ~4k cycles per 128 x 384 tile). Not built. The two GEMM launches
are the time: ~4 us each above the graph floor, and halving the fp32 operand bytes is what moved them.

### What changed

* **The small-M chain on the fp16 tensor cores for fp32 inference** (`tsx_ln.cu` / `tsx_expand.cu` / `tsx_squeeze.cu` `-DH16=1`):
  tsx_ln writes xn = fp16(LN(x) / s_x) (the F32 build's statistics), the expand runs kind::f16 with fp16 A / B and stores
  h = fp16(silu(a) b / s_h) with a, b rescaled by s_x s_w, the squeeze reads the fp32 residual and writes fp32 out = x + acc s_h s_ws.
  fp16 has TF32's 10-bit mantissa and rounds to nearest (the H100 f32io forward's scheme, `tg_fused_h16`'s): half the TF32 chain's
  operand bytes per CTA. `fused_small_sm100a` takes it for fp32 inference at D 384 n2 from 2 to 16 row tiles and D 768 n2 up to 8
  (`CELLS_H16`, `H16_MIN_TILES`; the 8-tile D 768 call measured 24.6 us against 30.7 on the generic fp16 forward). The training forward
  keeps the TF32 chain (its backward takes TF32 xn / a / b).
* **fp32 D 512 inference in three fp16 launches** (`fused_generic_sm100a._fwd_h16x3_launch`, `GENERIC_H16_WIDTHS`): tsx_ln `-DH16` ->
  `tg_gemm -DH16` SWIGLU (h fp16) -> RESID (fp32 residual and out), the scales passed in the `bias` slot -- the H100's `transition_lnsg_f32`
  + `squeeze_res_h16` structure. `tg_gemm`'s other builds are unchanged bit for bit (the RESID scale is a constexpr 1 there).
* **The fp16 h scale from a Cauchy-Schwarz bound** (`_h16_forms`, `H16_BOUND_L2`): |a_j| <= ||xn||_2 ||Wa_j||_2 with
  ||xn||_2 <= sqrt(D) max|gamma| + ||beta||_2, instead of the element bound times the rows' L1 norms. Both are rigorous; the L1 one gave
  s_h = 6-22 for N(0, 1 / D) weights at D 384-768 and pushed the small h of the single-track rows into fp16's subnormals (rows 35 / 37
  float32 measured YELLOW 1.05-1.15 x tol with it, `a_h16.jsonl`); with the L2 one s_h = 1 there and the same cases are 0.98-1.007.
  It also lifts `tg_fused_h16`'s D 256 cap of 2^20 rows (row 29 float32 L 1536 / 2048: GREEN 0.9998 / 0.9995 where the cap had been set
  for YELLOW 1.050 / 1.161), so those two cases run the one-launch fp16 forward now.
* **`tg_colsum` with 32 warps in the Transition backward** (the dgamma | dbeta sum of ~600 partial rows ran in 16 blocks of 8 warps;
  other launches keep 256 threads and their bits): fp32 [1, 128, 128, 128] n2 training 116.7 -> 110.6 us, [1, 128, 128, 256] 227.3 -> 221.6.

Kept out by the accuracy rule (measured, then routed back to TF32): fp32 [1, 128, 384] (one tile: YELLOW 1.0014 where every token is
GREEN, 18.2 us TF32 against 14.3 fp16) and D 768 above 8 tiles (L 1536 / 2048: 1.0054 / 1.0067 against limits 1.0036 / 1.0031; the
chain and the generic fp16 forward give the same bits there: the error is the fp16 products', not the split).

Measured (`measure.speed`, graph, L2 flushed; us, TF32 -> fp16; H100 cuda):

| case | before | now | H100 | ratio |
|---|---|---|---|---|
| row 35 fp32 [1, 256-768, 384] | 18.3 | 14.3 | 22.1-23.3 | 0.62-0.65 |
| row 35 fp32 [1, 1024 / 1536 / 2048, 384] | 20.4 / 22.4 / 26.5 | 16.2 / 16.3 / 20.3 | 23.1 / 28.6 / 30.2 | 0.70 / 0.57 / 0.67 |
| row 37 fp32 [1, 128-768, 768] | 24.3-30.6 | 16.4-22.5 | 30.2-42.7 | 0.53-0.64 |
| row 37 fp32 [1, 1024, 768] | 38.8 | 24.6 | 44.2 | 0.56 |
| row 33 fp32 [1, L, L, 512], L 128-768 | 131-3564 | 82-2050 (L 384: 501 against 844) | 142-4879 | 0.42-0.58 |
| row 29 fp32 L 1536 / 2048 | 4873 / 8695 | one-launch fp16 | 5131 / 9213 | below 0.667 |

Accuracy (`a_fin.jsonl`, 42 float32 eval cases of rows 27 / 29 / 33 / 35 / 37): 42 / 42 at or below the YELLOW limit (`accx.py`), 0 RED;
training (`a_tr.jsonl`, 72 cases): 72 / 72. Deterministic (two calls bit-equal, `sxab.py` / `tcheckt.py`). CPU gate: ruff, pyright on
the changed files clean; `tests/cpu/test_judge.py test_capture_api.py test_prebuild_plan.py test_requirements.py` 1227 passed.

### Misses of 1.5x against the H100 (H100 >= 20 us), part 3 -> part 4

| dtype, mode | misses | worst ratio |
|---|---|---|
| float32 eval | 31 -> **10** | 0.981 -> 0.978 |
| float32 train | 16 -> 16 | 0.975 -> 0.976 |
| master eval | 23 -> 23 | 1.057 -> 1.057 (not re-measured: bf16 chains unchanged) |
| master train | 12 -> 12 | 0.929 -> 0.935 |
| total | 82 -> **61** | |

What remains and why (measured):

* float32 eval: row 27 L 128 / 256 / 384 (0.96 / 0.80 / 0.68) and row 29 L 128 / 256 (0.98 / 0.75) on `tg_fused_h16`: the kernel runs at
  ~4x its HBM bound at these sizes (row 27 L 384, 147 456 rows: 80 us against ~20 us of traffic; 7.8 tiles per persistent CTA, the
  per-tile pipeline is the time); row 35 L 128 (accuracy rule, above), L 1024 / 2048 (0.70 / 0.67: the 8 / 16-tile chain); row 37 L 1536 /
  2048 (accuracy rule keeps TF32: 0.76 / 0.89).
* float32 / master training at small M (rows 217 / 219 and 211 / 213 L 128): the generic step is 10-11 launches (fp32 [1, 128, 384]:
  tg_prep 3.0, tsx_ln 2.4, expand 7.2, squeeze 6.2, tg_round_tf32 1.6, GATE_AB 8.1, two cuBLAS dW 4.0 + 4.8, PLAIN d_xn 4.7, tg_ln_bwds 5.3,
  tg_colsum 2.3 = 49.7 us device; bf16 [1, 128, 768] 10 launches, 50.4 us). 1.5x needs 37.7 / 39.6 us: a 5-launch step (the TF32 / D 768
  `tsx_dxn` + `tsx_dw` of the bf16 D 384 chain), not built.
* master eval rows 35 / 36 / 37 (bf16 chains, 16-37 us against H100 20-39): three launches at ~4 us each above the 6.1 us floor;
  the LayerNorm fold is worth at most one timer step (table above).

### H100 techniques -> B200 (audit, Transition)

| H100 (`docs/h100/transition.md`) | B200 |
|---|---|
| D 128 one fused forward + two-role backward (`fused_sm90a`) | used (`fused_sm100a`: `tr_fwd_sm100` / `tr_bwd_sm100`) |
| D 64 one fused forward + fused two-role backward with partial reduce | used (`tfwd_d64` / `tbwd_d64`) |
| D 256 one fused forward; gate -> cuBLAS fp32 dW -> d_xn + LN-bwd -> reduce | used (`tfwd_d256`, wide backward) |
| D 384 / 512 `ln_swiglu_gemm`: LayerNorm inside the expand GEMM | not applicable at small M (measured: a free LayerNorm is <= one 2 us timer step, table above); the large-M D 384 forward is the wide `tfwd_d384` |
| D 768 split expand saving h, a, b; `gate_saved` | used (the chain saves [a \| b] for GATE_AB) |
| D 384 keeps h for the backward (`save_h_enabled`) | not applicable: GATE_AB recomputes h from the saved a, b in its epilogue (no extra GEMM) |
| fp32 f32io: fp16 products at TF32 rounding, device scales, one-launch D 64 / 128 / 256 forward | used (`tg_fused_h16`, now without the D 256 row cap) |
| fp32 D 512 `lnsg_f32` (fp16 h) + `squeeze_res_h16` | used since this part (three launches, `GENERIC_H16_WIDTHS`) |
| fp16 forward for fp32 single-track rows (D 384 / 768) | H100 keeps them generic TF32; B200 now runs them on fp16 (2-16 / up to 8 tiles) |
| fp32 TRAINING forward on fp16 (F32_TRAIN_SHAPES D 128 / 256 / 512, saving xn) | missing: the B200 training forward is TF32; it needs per-step fp16 forms + scales on the device (the H100's `f32_forms_stats` / `apply`, two launches) |
| fp32 D 128 fused TF32 backward (`bwd_f32dx` + `bwd_f32dw`, from 65 536 rows) | missing (B200: generic GATE -> cuBLAS -> PLAIN -> ln_bwd -> colsum) |
| generic: PDL chains, one-launch weight prep (`prep_bwd`), fixed-order affine reduce | used (`tg_prep`, `tg_ln_bwd` + `tg_colsum`; the forward chains PDL) |
| `prefers_generic` row table | used (`route_sm100a.GENERIC`) |

## Round 3, part 5 (2026-11-17): the weight gradients on `tg_dw` (`cuda:b200_sm100` 6)

Records (B200 box, `$W/b2511/agents/r3-transition/out/`): `sp_p5_{train,eval}_<L>.jsonl` (rows 33 / 35-37 / 209-221, master + float32,
one exclusive measurer process per length: no small case timed behind a large one), `a_p5e.p5.jsonl` / `a_p5t.p5b.jsonl` /
`a_p5f.p5c.jsonl` (accuracy, `.accx` = err / tol against the YELLOW limit), `p6dw.txt` (tg_dw against cuBLAS, below), `sp_p6_<L>.jsonl` /
`a_p6.jsonl` (rows 33 / 211 / 213 float32 training after the fp32 route fix), `p6prof.txt` (per-kernel device times), `p6sum.txt` (misses).
Probes: `b200cfg/p6dw.sh`, `p6prof.sh`, `p6spd.sh`, `p5sum.py`.

### What changed

* **`tg_dw.cu`: both weight gradients of the backward in one launch** (dWs = dy^T h, [dWa; dWb] = [dA | dB]^T xn; MN-major tcgen05 over
  the rows straight from the row-major activations, bf16 or kind::tf32 with the 32-B-atom swizzle for fp32; a deterministic row split whose
  fp32 slabs `tg_wsum` adds in slab order, with the dgamma | dbeta partial sum in the same launch; unsplit, the dgamma | dbeta sum is folded
  into `tg_dw` itself and `tg_colsum` is gone). It replaces the two cuBLAS GEMMs (+ cuBLAS' own split-K reduce) in the generic backward and
  in the wide (D 256 / 384 / 512) backward. For fp32, dy is split into two TF32 values (`tg_split_tf32`, hi + lo, dWs from both) and h is
  rounded to nearest TF32, so the TF32 dWs keeps the accuracy of cuBLAS' fp32-input GEMM.
* The generic GEMM / row launches are programmatic dependent launches (griddepcontrol.wait before the first dependent load).
* **fp32 route of `tg_dw` (`DW_MAX_F32_ELEMS` = 2^22 x elements, takeover fix)**: as committed, fp32 took `tg_dw` at every row count up to
  D 256 and up to 16 384 rows above; measured against cuBLAS (`p6dw.sh`, graph, L2 flushed, us, twice each, tg_dw -> cuBLAS):

  | cell (training step) | tg_dw | cuBLAS |
  |---|---|---|
  | fp32 D 512 n 2, 16 384 rows (row 33 L 128) | 708 | **539** |
  | bf16 D 512 n 2, 16 384 rows | **248** | 259 |
  | fp32 D 256 n 2, 16 384 / 65 536 / 147 456 / 262 144 / 409 600 / 589 824 rows | **227** / 760 / 1641 / 2848 / 4416 / 6372 | 232 / **703** / **1595** / **2637** / **3976** / **5618** |
  | fp32 D 128 n 2, 16 384 / 65 536 / 147 456 / 262 144 / 589 824 rows | **104** / **305** / 633 / 1075 / 2353 | 115 / 309 / **629** / **1050** / **2227** |

  The dy split pass and the slab traffic grow with M (fp32 D 256 65 536 rows: `tg_split_tf32` 80 us, `tg_wsum` 35 us, `tg_dw` 186 us); the
  fp32 rule is now tg_dw up to M x D = 2^22 (D 128 to 32 768 rows, D 256 to 16 384, D 384 / 768 small M), cuBLAS above. bf16 keeps
  `DW_MAX_ROWS` = 16 384.

Accuracy: training rows 33 / 209-221 (master + float32, `a_p5t.p5b`, `a_p5f.p5c`, `a_p6`): every case at or below its YELLOW limit, 0 RED;
eval rows 33 / 35-37 unchanged. Deterministic: two calls bit-equal (`p5tc1.txt` / `p5tc2.txt`: fp32 [1, 128, 384] / [1, 512, 768]).
CPU gate: `tests/cpu/test_judge.py test_capture_api.py test_prebuild_plan.py test_requirements.py` 1227 passed; ruff clean on the changed files.

Measured (`measure.speed`, us, part 4 -> part 5; H100 cuda): row 217 fp32 [1, L, 384] L 128-768 55.2-63.5 -> 47.0-57.3 (56.5-80.6);
row 219 fp32 [1, L, 768] 73.7-96.1 -> 65.6-90.0 (76.7-136.9); row 219 master 53.1-67.6 -> 49.1-63.4 (59.4-82.4); row 211 fp32 L 128
110.6 -> 98.2 (140.0); row 33 fp32 L 128 526 (token) -> 695 (tg_dw) -> 524 (cuBLAS again); row 213 fp32 L 256-768 (cuBLAS again)
675 / 1531 / 2557 / 3865 / 5496 against the tokens 670 / 1527 / 2478 / 3792 / 5369 (within 1-3 %).

### Misses of 1.5x against the H100 (H100 >= 20 us), part 4 -> part 5

| dtype, mode | misses | worst ratio |
|---|---|---|
| float32 eval | 10 -> 10 | 0.978 -> 0.979 |
| float32 train | 16 -> 17 | 0.976 -> **0.855** |
| master eval | 23 -> 23 | 1.057 -> 1.049 |
| master train | 12 -> 12 | 0.935 -> 0.886 |
| total | 61 -> 62 | |

The float32 train count rises by one although no case got slower than its token: row 213 L 640 / 768 (H100 5.69 / 8.22 ms) sit at
0.680 / 0.669 against tokens of 0.667 / 0.653 (1-2 % run-to-run on 4-5 ms cases) and were not on the part-4 list. No Transition case is slower than the H100 in float32 / master training
(worst 0.886).

### Not built in part 5 (the brief items), and the measured reason

Per-kernel device times now (`p6prof.txt`, tb.py, bf16 parameter casts of the probe excluded):

| step | kernels (us) |
|---|---|
| fp32 [1, 128, 384] train (51.2 graph) | GATE_AB + PLAIN 13.1, tsx_expand 7.1, tsx_squeeze 6.6, tg_dw 6.2, tg_ln_bwds 5.1, tg_prep 3.1, tsx_ln 2.4, tg_split_tf32 1.8 |
| fp32 [1, 128, 768] train (71.7) | GATE_AB + PLAIN 17.4, tg_dw 11.1, tsx_expand 10.2, tg_ln_bwds 7.6, tsx_squeeze 7.1, tg_prep 6.4, tsx_ln 2.9, split 1.9 |
| bf16 [1, 2048, 384] n 4 eval (28.7) | tsx_expand 14.4, tsx_squeeze 12.2, tsx_ln 2.3 |
| bf16 [1, 2048, 384] n 2 eval (20.5) | tsx_expand 9.2, tsx_squeeze 6.9, tsx_ln 2.3 |
| bf16 [1, 640, 768] eval (20.8) | tsx_squeeze 7.7, tsx_expand 7.5, tsx_ln 2.6 |

* fp32 training forward on fp16 operands (H100 `F32_TRAIN_SHAPES`, item 1): not built. The forward is 16 of 45 us at D 384 small M and the
  backward would need the saved xn on the TF32 grid as it is now; the per-step forms launch (`_h16_forms` is ~10 torch ops today) has to
  become one kernel first.
* 5-launch small-M training step (item 2): `tg_dw` is the weight-gradient half of it (11 -> 9 launches); `tsx_dxn` (d_xn + LayerNorm
  backward + dgamma | dbeta in one cluster launch, replacing PLAIN + tg_ln_bwds) is the remaining ~8-10 us. Row 217 L 128 needs 37.7 us
  (now 47.0), row 219 fp32 L 128 51.1 (now 65.6). The D 128 TF32 fused backward (`f32dx` / `f32dw`) remains missing.
* master eval rows 35 / 36 / 37 (item 3): the H100 runs them as `ln_swiglu_gemm` + squeeze (two launches); the B200 three (above). The
  LayerNorm launch is 2.3-2.6 us of device time (one timer step, part 4 bound table), so matching the H100's launch count does not reach
  1.5x; the n 4 L 2048 case (28.7 against 27.3 on the H100; target 18.2) needs the expand + squeeze in one launch with the hidden split
  over CTAs (16 row tiles x 8 hidden slices, a deterministic cluster reduction of the squeeze partials): not built.
* rows 27 / 29 small L (item 4) and the hi + lo fp16 squeeze for rows 35 L 128 / 37 L 1536-2048 (item 5): not started.

### H100 techniques -> B200 (audit, Transition), changes of part 5

| H100 | B200 |
|---|---|
| generic backward: cuBLAS TF32 dW | B200 goes further: one `tg_dw` launch for both weight gradients (bf16 to 16 384 rows, fp32 to 2^22 x elements), cuBLAS above (measured) |
| fp32 TRAINING forward on fp16 (F32_TRAIN_SHAPES) | missing (see above) |
| fp32 D 128 fused TF32 backward (`f32dx` + `f32dw`) | missing |
| D 384 / 512 `ln_swiglu_gemm` (2 launches) | B200 3 launches (LayerNorm launch 2.3-2.6 us measured; a fold is <= one timer step) |

## Round 3, part 6 (2026-11-17): the one-launch small-M forward built and measured (rejected), side-stream overlaps (`cuda:b200_sm100` stays 6)

Probes (B200 box, `$W/b2511/agents/r3-transition/b200cfg/`): `one.py` (graph median, L2 flushed, error vs fp64, max |diff| to the chain,
determinism, per build), `onep.py` (the kernel built with `-DPROBE=1`, launched directly: per-CTA `%globaltimer` stamps at the phase
boundaries), `tb.py -noside / -noprep` (A/B of the side-stream changes, per-kernel profile with `-p`), `side1.sh` / `side2.sh`. The
rejected kernel and its routing: `r3tr6_one_launch_rejected.patch` / `tsx_one_rejected.cu` in the coordinator's scratch folder (`b2/`).

### Item 1: master eval rows 35 / 36 / 37 in ONE launch (`tsx_one.cu`, built, correct, slower: not routed)

What the H100 does there: `ln_swiglu_gemm` (LayerNorm inside the expand GEMM, h to L2) + the squeeze GEMM, two launches. The B200 runs
three (`tsx_ln` -> `tsx_expand` -> `tsx_squeeze`, PDL). The kernel built here does the whole forward in one: one cluster of SK CTAs per
128-row tile, CTA k owns the hidden slice k (H / SK units, whole 64-unit steps):

1. LayerNorm in the loader: CTA k normalises 128 / SK rows (tsx_ln's bf16 arithmetic: the same xn bits) into its shared xn tile
   (K-major, 128-B swizzle) and bulk-copies them (`cp.async.bulk.shared::cluster`) into every other CTA's tile;
2. expand per 64-unit step (M 128, N 128 into tensor memory), SwiGLU by 8 epilogue warps into a shared h K-block (two buffers),
3. squeeze of each h block straight into a [128 x 384] fp32 accumulator in tensor memory (three N 128 pieces): h never leaves the SM;
   every weight operand (a [Wa_c; Wb_c] K-block or a 128-row Ws piece) is one 16 KB slot of one ring in the order the tensor pipe takes them;
4. the SK fp32 partials of y reduced over the cluster in distributed shared memory, rank order (deterministic), + residual, bf16 store.

Correct on every case tried: error vs fp64 equal to the chain's (e.g. [1, 2048, 384] n 4: 3.468e-3 both, PyTorch 5.952e-3), max |diff| to
the chain one bf16 ulp of the output (the squeeze's hidden split adds the partials in another order), two calls bit-equal. Speed (graph,
L2 flushed, us; chain -> best one-launch build, SK in brackets):

| case | chain | one launch, version 1 (two rings, st.async exchange) | version 4 (one ring, bulk-copy exchange) |
|---|---|---|---|
| [1, 2048, 384] n 4 (row 36, H100 27.3) | **28.7** | 41.0 (4) | 32.8 (6) |
| [1, 1024, 384] n 4 | **22.5** | 32.8 (8) | 32.8 (6 / 8) |
| [1, 640, 384] n 4 | **18.4** | 32.8 (8) | -- |
| [1, 2048, 384] n 2 (row 35, H100 21.1) | **20.6** | -- | 28.7 (6) |
| [1, 128, 384] n 2 | **14.3** | -- | 26.6 (6 / 12) |

Where the time goes (`onep.py`, per-CTA stamps, median over the CTAs, us from the first CTA's entry; [1, 2048, 384] n 4, SK 6, 96 CTAs):

| phase | version 2 (one ring, st.async push) | version 3 (staged partial, ld.shared::cluster pull) | version 4 (bulk copies, two rounds) |
|---|---|---|---|
| setup (barriers, 512 TMEM columns, cluster barrier) | 0.5 | 0.5 | 0.5 |
| LayerNorm of 16-22 rows + their bulk copies | 2.8 | 2.8 | 2.8 |
| xn of the other ranks in (82 KB) | 0.8 | 0.8 | 0.8 |
| expand + SwiGLU + squeeze (576 KB of weights per CTA) | 10.7 | 10.7 | 10.7 |
| the partials' reduction (160 KB out / in per CTA) | 9.6 | 4.4 staging + 18.2 pull | 3.3 + 5.3 |

Two bounds make this design lose at these sizes, both measured:

* **weight streaming per CTA**: the ring (96 KB beside the resident 96 KB xn tile and two h buffers) turns over at ~0.3 us per 16 KB slot
  whatever the load: 1 tile (6 CTAs) and 16 tiles (96 CTAs) give the same ~55 GB/s per SM, i.e. an effective ~1.7 us per slot round trip
  with 6 slots in flight. Every tile's cluster streams all 3 D H x 2 B of weights (3.5 MB at n 4) over its SK SMs; the chain spreads the
  same bytes over 384 expand CTAs (2.6 waves) and keeps many more bytes in flight. A version with two rings (one per operand) was worse
  (each load waited on its own ring: 26 us per CTA at [1, 640, 384] n 4).
* **the hidden split's partial reduction**: y partials are 128 x D fp32 per CTA (196 KB), more than the h tile the chain sends through L2.
  Distributed shared memory moved them at ~18 GB/s per SM with 16-B st.async stores, slower with ld.shared::cluster pulls, and ~100 GB/s with
  bulk copies (the xn exchange measures the same), which still costs 8.6 us in two rounds (staging slots + receive slots must fit the freed
  operand buffers) plus three cluster barriers.

Not routed (one implementation per op: the chain stays). What it would take: halve the per-CTA weight bytes with `cta_group::2` (a CTA
pair, M 256 over two tiles, each CTA loading half of every weight slot) and keep xn out of shared memory (A from tensor memory, which needs
the squeeze accumulator split over a CTA pair to free the columns) -- the expected floor then is still ~14-18 us device for [1, 2048, 384] n 4
against 22.7 us of the chain, short of the 18.2 us graph target.

### Item 2: the small-M training step

`tsx_dxn` for fp32 / D 768 (d_xn + LayerNorm backward + dgamma | dbeta in one cluster launch) was not built: the two bounds above apply to
it directly. d_xn's A operand per tile is [dA | dB] fp32 (128 x 2 H x 4 B = 786 KB at D 384 n 2, 1.5 MB at D 768), which one cluster per tile
would stream at ~55 GB/s per SM (16 CTAs: 245 KB each, ~4.5 us) and then reduce its K-split partials through distributed shared memory
(the slow part above); the split-K PLAIN GEMM + `tg_ln_bwds` it would replace take 6.5 + 5.1 us now and spread the operand over many CTAs.
Built instead (both A/B'd, both rejected):

* `tg_dw` on a side stream right after GATE (it needs dy, h, [dA | dB], xn only), beside PLAIN + `tg_ln_bwds`, the dgamma | dbeta sum moved
  to `tg_colsum` (`fused_generic_sm100a.DW_SIDE`): graph, us, single call L2 flushed (the measurer's number) -> amortised over 20 calls:

  | step | tg_dw last | side stream |
  |---|---|---|
  | fp32 [1, 128, 384] n 2 | 51.2 / 45.3 | 51.2 / 40.9 |
  | fp32 [1, 512, 384] n 2 | 61.4 / 50.9 | 61.4 / 45.8 |
  | fp32 [1, 128, 768] n 2 | 75.8 / 64.1 | 77.8 / 63.8 |
  | bf16 [1, 128, 768] n 2 | 55.3 / 46.0 | 59.4 / 43.6 |
  | bf16 [1, 512, 768] n 2 | 67.6 / 56.8 | 67.6 / 57.1 |

  The overlap shows in back-to-back calls (4-5 us) but not in the single flushed call the requirement timing takes (the fork / join
  of the graph and the extra tg_colsum launch eat it). Reverted.
* the training forward's weight copies (`tg_prep`, 3.1 us at D 384 / 6.4 at D 768) on a side stream beside `tsx_ln`, joined before the
  expand (`fused_generic_sm100a.PREP_SIDE`, one custom op making the copies and running the chain): fp32 [1, 128, 384] n 2 single call
  51.2 -> 55.3 us (twice each, alternating), amortised 45.2 -> 45.0. Reverted.

  Both side-stream changes: `r3tr6_side_streams_rejected.patch` (`b2/`). A fork / join inside a captured single call cost ~4 us here,
  more than the launch it hides.

### Items 3-5

Not started: `_h16_forms` in one launch + the fp32 training forward on fp16 operands (item 3), the D 128 TF32 fused backward `f32dx` /
`f32dw` (item 4), `tg_fused_h16` at small L and the hi + lo fp16 squeeze (item 5). The time went into item 1 (four versions with stamps).

### Misses of 1.5x against the H100 (H100 >= 20 us), part 5 -> part 6

No kernel change was kept (`cuda:b200_sm100` stays 6; no token changes): float32 eval 10, float32 train 17, master eval 23 (worst 1.049,
row 36 [1, 2048, 384] n 4), master train 12 -- 62 in total, as after part 5.

### H100 techniques -> B200 (audit, Transition), changes of part 6

| H100 | B200 |
|---|---|
| D 384 / 512 `ln_swiglu_gemm` + squeeze (2 launches) | one-launch forward built (LayerNorm split over the cluster, h on chip, DSMEM reduction): correct, 1.1-1.9x slower than the 3-launch chain (per-CTA weight streaming ~55 GB/s per SM, 196 KB fp32 partials per CTA through DSMEM); not routed |
| fp32 TRAINING forward on fp16 (F32_TRAIN_SHAPES), `f32_forms` in two launches | missing (not started) |
| fp32 D 128 fused TF32 backward (`f32dx` + `f32dw`) | missing (not started) |
| generic backward, one stream | kept: tg_dw / tg_prep on a side stream measured slower per single call (fork / join ~4 us) |

## Round 3, part 7 (2026-11-17): the H100's fp32 training forward on fp16, the accuracy-rule follow-up (`cuda:b200_sm100` 7)

Accuracy rule of this part (coordinator, 2026-11-17 evening): the engine standard only -- GREEN or YELLOW (<= 5 tol) pass, RED fails;
"no case worse than its current B200 token" is withdrawn. Records (B200 box, `$W/b2511/agents/r3-transition/out/`): `p7probe.txt` (TF32
against fp16 forward), `p7chk.txt` / `p7ab1-3.txt` (training A/B, `tcheckt.py` error and determinism), `p7prof.txt` (per-kernel times),
`a_p7.jsonl` / `a_p7t.jsonl` (accuracy), `sp_p7e_<L>` / `sp_p7f_<L>` / `sp_p7g_<L>.jsonl` (speed, one exclusive measurer process per
length), `p7final.txt` / `p7fin2.txt` (summaries). Probes: `b200cfg/runp.py` (flag overrides around a script), `p7ab.sh`, `p7prof.sh`,
`p7final.sh`, `p7fin2.sh`, `p7sum.py`.

### What changed

* **fp32 training forward on the fp16 tensor cores (the H100's `F32_TRAIN_SHAPES`)**, D 64 / 128 / 256 from 65 536 rows
  (`fused_generic_sm100a.TRAIN_H16`, `TRAIN_H16_MIN_ROWS`): `tg_fused_h16` gets a training build (`-DSAVE_BUILD=1`, cubin `fusedh<D>s`;
  the inference cubin is unchanged) whose LayerNorm warps also write xn on the TF32 grid, mean and rstd; the backward is the generic TF32
  recompute backward as before (the H100's structure: fp16 forward, native-TF32 backward).
* **`tg_prep16`: the step's fp16 weight forms and scales in ONE launch** beside the backward's TF32 copies (`tg_rows.cu`; replaces
  `tg_prep` + `_h16_forms`' ~10 torch launches; the H100 needs two, `f32_forms_stats` + `f32_forms_apply`). tg_prep's grid (one block per
  32 x 32 tile): every block writes its TF32 copies / transposes, the fp16 forms at scale 1 and its tile's max |w| and largest 32-column
  row sum of squares; the last block (ticket counter, reset by that block) reduces them in a fixed order into sc and, only where a weight
  scale is not 1 (|w| > 30000), rewrites the fp16 forms divided by it. The row-norm bound is the per-32-row-tile sum of those maxima
  (rigorous, <= ~2x the exact norm; s_h stays 1 for ordinary weights). Two designs measured first and dropped: a software grid barrier
  between a bounds pass and the conversion with one block per SM (37.5 us at D 768 n 2: 16 tiles per block in sequence), four per SM with
  pipelined loads (23.8 us); the last-block form is 18.2 / 8.1 / 6.9 us at D 768 / 384 / 128 against tg_prep's 6.4 / 3.1 / 3.1.
* **fp32 inference, the YELLOW cells part 4 had kept on TF32** (allowed now): the fp16 chain from one row tile at D 384
  (`H16_MIN_TILES` empty: row 35 L 128 18.2 -> 14.2 us) and up to 12 tiles at D 768 (`CELLS_H16`: row 37 L 1536 44.9 -> 26.6 us), the
  three-launch fp16 generic forward at D 768 above (`GENERIC_H16_WIDTHS` + 768: row 37 L 2048 57.2 -> 38.8 us; the chain measured 43.0).

Built and measured slower, not kept:

* the fp32 training forward on the fp16 small-M chains (D 384 / 768 single track, rows 217 / 219: `tsx_ln -DH16` with saves of xn and the
  statistics, `tsx_expand -DH16` saving [a | b] fp32 for GATE_AB): correct and deterministic, but graph, us, TF32 -> fp16: [1, 128, 384]
  51.2 -> 53.2, [1, 128, 768] 71.7 -> 79.8, [1, 512, 768] 92.1 -> 96.3. The forward saves 3.5 - 6 us (expand 10.3 -> 6.4, squeeze
  7.3 -> 4.9 at D 768) and tg_prep16 costs 5 - 12 us more than tg_prep (its scale reduction is a serial tail of the launch). Diff:
  `b2/r3tr7_chain_train_h16_rejected.patch` in the coordinator's scratch folder.
* the fused training forward below 65 536 rows: D 128 / 256 at 16 384 rows 104.4 / 227.3 -> 104.5 / 229.4 us (forward -8 us, tg_prep16 +4).

Measured (`measure.speed`, us, before -> after, H100 cuda; one process per length): row 213 fp32 train L 256 / 384 / 512 / 640 / 768
674.7 / 1530.8 / 2557.4 / 3865.5 / 5496.0 -> 661.5 / 1487.3 / 2427.6 / 3647.5 / 5167.6 (955.4 / 2037.8 / 3602.0 / 5688.0 / 8221.3);
row 211 fp32 train L 256-768 unchanged within noise (already below 0.667); rows 27 / 29 fp32 eval unchanged (20.4 / 40.8 at L 128).

Accuracy (`a_p7`, `a_p7t`, err / tol): rows 211 / 213 fp32 train 1.004-1.039 (all 12 YELLOW as before, limit of the old rule 1.26-1.47);
row 35 fp32 eval L 128 1.0014, row 37 L 1536 / 2048 1.0054 / 1.0067 (YELLOW, inside the standard); every other case of rows 27 / 29 / 35 / 37
as before. Deterministic: two training steps bit-equal (`tcheckt.py`: D 128 / 256 n 2 at 16 384 and 65 536 rows, D 64 at 4 096, D 384 / 768 at 128 / 768 rows).
CPU gate: ruff check / format clean on the changed Python files.

### Misses of 1.5x against the H100 (H100 >= 20 us), part 6 -> part 7

| dtype, mode | misses | worst ratio |
|---|---|---|
| float32 eval | 10 -> **7** | 0.979 -> 0.980 (row 29 L 128) |
| float32 train | 17 -> **15** | 0.855 -> 0.855 (row 219 L 128) |
| master eval | 23 -> 23 | 1.050 (row 36 n 4 L 2048) |
| master train | 12 -> 12 | 0.885 (row 211 L 128) |
| total | 62 -> **57** | |

Flipped: row 35 fp32 eval L 128 (0.823 -> 0.643), row 37 fp32 eval L 1536 / 2048 (0.734 / 0.859 -> 0.434 / 0.583), row 213 fp32 train
L 640 / 768 (0.680 / 0.669 -> 0.641 / 0.629). Closer, still missing: row 213 L 256 / 384 / 512 (0.706 / 0.751 / 0.710 -> 0.692 / 0.730 / 0.674).

### Not built in part 7, and the measured reason

* **H100 `f32dx` / `f32dw`** (the D 128 TF32 fused backward): the H100 runs it from 65 536 rows only (`_F32_FUSED_BWD_MIN_ROWS`; below,
  its own generic backward measured faster). On the B200 every row-211 fp32 case from 65 536 rows (L >= 256) is already below 0.667; the
  one miss is L 128 (16 384 rows, 98.3 against 140.0, needs 93.3), the size at which the H100 itself keeps the generic backward. The
  port is also no structural copy: the bf16 B200 two-role backward (`tr_bwd_sm100.cu`) fills the shared memory with bf16 operands
  (inputs 2 x 64 KB + rings); TF32 doubles every operand.
* **master training at small M (item 3)**: the H100's small-M bf16 step (`prefers_generic`: `ln_fwd` -> SWIGLU_SAVE -> squeeze; `gate` ->
  cuBLAS d_xn -> `ln_bwd` + `reduce_cols`, the dW GEMMs on a side stream) is 9+ launches with a stream fork; the B200 step is 8 launches
  (tg_prep, tsx_ln, expand, squeeze, GATE_AB, PLAIN split-K, tg_ln_bwds, tg_dw) on one stream, and the side-stream dW was A/B'd in part 6
  (slower per single flushed call: fork / join ~4 us). Nothing of the H100's step is missing on the B200; the 12 cases stay.
* **rows 217 / 219 fp32 training**: the fp16 forward does not pay at a single track (above); what remains is the 8-9-launch step
  (~5 us per launch above the 6 us graph floor).

### H100 techniques -> B200 (audit, Transition), changes of part 7

| H100 | B200 |
|---|---|
| fp32 TRAINING forward on fp16 (`F32_TRAIN_SHAPES` D 128 / 256 / 512), saving xn | used at D 64 / 128 / 256 from 65 536 rows (`tg_fused_h16 -DSAVE_BUILD`); below and at the single-track D 384 / 768 measured slower (the per-step forms); D 512 (row 33, already 0.60-0.65) not ported |
| `f32_forms_stats` + `f32_forms_apply` (two launches per step) | one launch (`tg_prep16`), with the backward's TF32 copies |
| fp32 D 128 fused TF32 backward (`f32dx` + `f32dw`, from 65 536 rows) | not applicable: the B200 generic backward already meets 1.5x on every row-211 case from 65 536 rows; the L 128 miss is below the H100's own threshold |
| bf16 small-M training: dW on a side stream | not applicable (measured slower in part 6, single flushed call) |

## Round 3, part 8 (2026-11-17 evening): true state, D 384 inference routes, wide training copies (`cuda:b200_sm100` 20)

Targets (BRIEF6): (1) B200 `cuda` <= H100 `cuda` / 1.5 on every case with an H100 median >= 20 us; (2) ladder `cuda` < `triton` <
`pytorch` (2 % margin). Measured with `measure.speed` (graph, L2 flushed), `cuda` + `triton` + `pytorch` on all 307 requirement cases,
one exclusive measurer process per (mode, length) on GPU 1 (`b200cfg/m3.sh`, `m3all.sh`; summary `b200cfg/viol.py`). Records (B200
box, `$W/b2511/agents/r3-transition/out/`): `m_m0_*` (base, b200-2.5.x 83e4c1ddf), `m_m3_*` (final), `viol_m0.txt` / `viol_m3.txt`,
`acc9_m3.txt` (accuracy of rows 33-37 / 40 / 213-216). Other agents' processes ran on GPU 1 outside the exclusive slot during part of
the session (`dev.sh` warned of foreign pids): large cases moved +-5 % between runs that no change reached.

| | base (m0) | final (m3) |
|---|---|---|
| (1) misses of 1.5x (H100 >= 20 us) | 59 | **54** |
| (1) of them slower than the H100 | 1 (row 36 n 4 L 2048, 28.7 vs 27.3) | **0** |
| (2) ladder violations | 1 (Triton row 37 fp32 L 1024, 51.1 vs PyTorch 49.0) | 1 (the same, 51.1 vs 49.1) |
| accuracy of the changed cells | -- | 94 GREEN / 31 YELLOW / 0 RED |

### What changed

* **bf16 D 384 inference from 8 row tiles (n 4) / 16 (n 2) on the generic kernels with the cuBLAS squeeze** (`route_sm100a.GENERIC`;
  the cancelled predecessor's part-8 finding, re-measured): the small-M chain's squeeze takes one 64-column block per CTA; cuBLAS'
  64 x 96 two-CTA tiles take it in 5.7 against 10.8 us at [1, 2048, 384] n 4. `fused_generic_sm100a`: the cuBLAS squeeze also where
  the RESID tiles leave SMs idle (bf16 inference, row tiles x column tiles < SMs; `SQUEEZE_SMALL_CUBLAS`). Measured (us, base -> m3):
  row 36 L 1024 / 1536 / 2048 / 2976 / 3968 22.4 / 24.5 / 28.7 / 32.6 / 34.7 -> 20.4 / 20.5 / 24.5 / 28.5 / 30.6, row 35 L 2048 20.4
  -> 18.5, row 37 (D 768, generic) L 1024 / 1536 / 2048 28.5 / 28.5 / 34.7 -> 22.5 / 24.5 / 30.7. Kept on the chain (A/B, `b200cfg/p9.sh`):
  n 4 L 512-768 (chain 16.4-18.2, generic 18.4), n 2 L 768-1536 (14.3-16.4 against 16.4-18.4).
* **fp32 D 256 training: d_xn on cuBLAS** (`_DXN_CUBLAS_F32`, the predecessor's `p8dxn` A/B: [65 536 / 147 456 / 262 144, 256] 657 / 1471 /
  2425 -> 629 / 1384 / 2271 us). Row 213 fp32 L 256-768 -4.4 to -6.2 %.
* **Wide-build training at D >= 256 from fp32 masters** (`fused_wide_sm100a`, rows 33 / 34 / 40 / 213 / 214 / 215 / 216): the profile of
  [16384, 256] n 2 (124.9 us) had ~23 us of copies (`ws.t().contiguous()`, `torch.cat((wa, wb)).t().contiguous()`, the cast launch, two
  output clones) and a 2-block `transition_lnbwd_w_reduce` of 15.5 us.
  * `T_FORMS`: the bf16 forms and Ws^T / [Wa; Wb]^T in ONE `tg_prep` launch in the forward (round-to-nearest, the same bits as `.to`),
    passed to the backward op;
  * `LN_SUM_IN_DW`: the dgamma | dbeta column sum inside `tg_dw` / `tg_wsum` (`weight_grads(part=, dgb=)`), the weight gradients
    moved after the LayerNorm backward; cuBLAS + the old reduce where `_dw_plan` keeps cuBLAS;
  * the backward op returns dgamma | dbeta and [dWa; dWb] packed (`_packed`), split by the caller (no clone launches);
  * `LNBWD_PER_SM` 4 -> 2 (fewer partial rows for the column sum).

  Output and dx bit-identical; dgamma / dbeta differ in summation order (max |diff| 1-5e-4 on O(10) values), deterministic over two steps
  (`b200cfg/p16.py`, `p18.py`, `p20.py`). Graph, us, base -> final: [16384, 256] n 2 122.9 -> 100.4, n 4 176-178 -> 149.3, [16384, 512] n 2
  239.5 -> 210.9, [65536, 256] n 2 321.5 -> 305.2; `measure.speed` row 213 master L 128 124.8 -> 100.2 (H100 160.8), row 214 L 128 176.0 ->
  149.4 (253.6), row 33 master train L 128 235.7 -> 208.9, row 40 L 256-640 -4 to -6 %.

Flipped below 0.667: row 36 n 4 L 2048 (1.049 -> 0.895: no longer slower than the H100, still a miss), row 213 master L 128 (0.776 ->
0.623), row 214 master L 128 (0.694 -> 0.589), row 213 fp32 L 256 / 512 (0.692 / 0.671 -> 0.662 / 0.630).

### Built and measured, not kept

| tried | measured | why not |
|---|---|---|
| Triton row 37 fp32 L 1024: every `candidates.json` config (24 expand, 23 squeeze) as rank 1 of the shared M 1024 bucket (`b200cfg/p13.py`) | 51.2 us for every config at L 1024 (PyTorch 49.2-51.2 in the same process); L 1536 51.2-53.2 | the bucket's GEMM configs do not bound the case; swapping the first two expand entries cost L 1536 51.2 -> 59.4 |
| Triton fp32: x rounded to TF32 inside the expand kernel (`ROUND_X_IN_KERNEL_SM100`) | L 512 / 1024 / 2048: 43.0 / 51.2 / 67.6 -> 51.2 / 71.6 / 114.7 | slower, as recorded for triton:b200_sm100 1 |
| generic route for D 384 n 4 from 4 tiles / n 2 from 4 tiles | n 4 L 512-768 18.4 against the chain's 16.4-18.2; n 2 L 768 / 1024 / 1536 16.4 / 16.4 / 18.4 against 14.3 / 16.4 / 16.4 | slower |
| `LNBWD_PER_SM` 1 | [16384, 512] 213.0, [65536, 256] 311.3 against 210.9 / 305.2 at 2 | 2 is the best everywhere measured |

The remaining ladder pair (Triton row 37 fp32 L 1024) is one 2.05 us timer tick: Triton read 51.1-51.2 us in every process, PyTorch
49.0-51.2. The Triton launches are LayerNorm (LayerNorm class) -> `tf32_grid_many` -> expand -> squeeze, 37.8 us of device time
against PyTorch's 43.8 in 8 launches; the gap between device time and the flushed graph median is not in the GEMM configs. What it
would take: the TF32 rounding of x inside the LayerNorm's store (LayerNorm class; one launch fewer, ~2 us).

### What still misses 1.5x (54 cases), and why

* **Launch-floor cells, H100 20-40 us** (rows 25-30 / 35-38 eval L 128-768, 210-212 / 217-219 train L 128): 3-launch inference at
  16-22 us and 8-12-launch training steps; the per-part history (parts 2-7) bounds them: the one-launch forward was built and is slower,
  the side streams are slower per single flushed call.
* **Row 36 n 4 L 1024-3968** (0.77-0.90): the generic kernels + cuBLAS in 3 launches, 20-31 us against the H100's 23-39.
* **fp32 rows 27 / 29 eval L 128-384, 211 / 213 / 217 / 219 train**: as in part 7 (first-tile latency, TF32 small-M backward not built).

## Round 3, part 9 (2026-11-17 night): inference only -- the one-launch pair forwards and the Triton fp32 LayerNorm (`cuda:b200_sm100` 22, `triton:b200_sm100` 22)

Scope from this part on: Transition INFERENCE (eval) cases, `cuda` and its Triton rung (the training cases moved to agent
`r3-transition-train`, odd versions). Records (B200 box, `$W/b2511/agents/r3-transition/out/`): `m_p9_eval_<L>.jsonl` (cuda, every eval
case, one exclusive process per length), `t_p9_<L>.jsonl` (triton + pytorch of the fp32 eval rows 33 / 35 / 37), `viol_m3e.txt` (base,
part-8 records) / `viol_p9t.txt` (after), `acc10_p9.txt` (accuracy of rows 25 / 27 / 29 / 30 eval), `acc11_p9.txt` (training rows 209-214
on the edited sources: all GREEN). Probes: `b200cfg/pf.py` (per-CTA `%globaltimer` stamps of the fused forwards built with `-DPROBE=1`),
`b200cfg/pv.py` (a forward build with extra `-D` flags against the routed one: eager L2-flushed median, back-to-back time, error vs fp64,
max |diff|), `b200cfg/tq.py` (Triton A/B).

| eval cases (181; 167 with an H100 median >= 20 us) | base (part 8) | part 9 |
|---|---|---|
| (1) misses of 1.5x | 31 | **28** |
| (1) of them slower than the H100 | 0 | 0 |
| (2) ladder violations | 1 (Triton row 37 fp32 L 1024, 51.1 vs PyTorch 49.1) | **0** (Triton 49.0 vs 51.1) |
| accuracy of the changed cells | -- | 45 GREEN / 9 YELLOW (row 27 fp32, untouched `tg_fused_h16`, as before) / 0 RED |

### What the stamps showed

The one-launch pair forwards (`tfwd_d64` / `tr_fwd_sm100` / `tfwd_d256`, master eval rows 25-30 / 38) were 3-4x off both bounds (e.g.
[147 456, 64] n 2: 19.9 us device for 38 MB and 7 GFLOP). Per-CTA stamps: D 64 n 2 runs one tile per ~2.0 us, the LayerNorm (0.8-1.0 us) and
the epilogue (0.8-1.0 us) of warps 8-11 in sequence; D 256 at one tile per CTA ([16 384, 256]) spends 1.0 us in setup, 2.6 us waiting
for x, 3.6 us in the LayerNorm (4 warps, three passes over a 256-column row), ~1.1 us per hidden chunk, 2.7 us in the epilogue; D 128 n 2
~4.5 us per tile, the LayerNorm + epilogue warps again the serial part.

### What changed (inference cubins `fwdi`, `fused_wide_sm100a.FWD_INFER_FLAGS`; the training forward keeps `fwd`)

* **D 64 n 2, `-DROLES=1`**: the epilogue on its own warpgroup (warps 4-7), every SwiGLU chunk on warps 12-15, the LayerNorm alone on 8-11
  (same bits). Back-to-back, us: [147 456, 64] 20.85 -> 18.57, [65 536, 64] 14.12 -> 12.34.
* **D 128 n 2, `-DLNH=1`**: the LayerNorm and the epilogue split over warps 4-7 / 8-11, every SwiGLU chunk on 12-15. Each half owns the
  leaves l = h, h + 2, .. of the sm_90a reduction tree (the first / second 8 bytes of every 16-byte chunk), whose sums are exactly the two
  subtrees `tree32` adds last: the same mean / rstd / xn bits. [65 536, 128] 23.24 -> 22.65, [147 456, 128] 41.28 -> 39.00.
* **D 256 n 2 / n 4, `-DXPAR=1 -DWPAR=1 -DLN2=1`**: the LayerNorm of a tile on two warpgroups (warps 4-7 columns 0-127 after their even
  SwiGLU chunks, 8-11 columns 128-255), each half row in registers, the row sums exchanged through 512 B of shared memory (summation order
  changes: max |diff| one bf16 ulp of the output, error vs fp64 equal to four digits); the epilogue split the same way; tile 0's x boxes before
  the cluster barrier and every x / weight box issued from a lane of its own (BRIEF6: one lane issues ~one TMA load per 600 clk).
  [16 384, 256] n 2 22.60 -> 20.35, n 4 30.79 -> 28.71, [65 536, 256] n 2 61.68 -> 59.36.
* **Triton fp32 inference with K > 128 (rows 33 / 35 / 37)**: the LayerNorm stores xn on the TF32 grid (`layernorm/triton/main.py`
  `layer_norm_fwd_fused(TF32_OUT=True)`, `_ln_fwd(tf32_out=True)`: an additive constexpr of the LayerNorm family's kernel, default off, the
  same kernel for every other caller) and the expand skips its launcher rounding pass (`_expand_swiglu(x_on_grid=True)`); B200 only (it
  applies exactly where `main._round_x_in_kernel` is false). The same output bits, one launch fewer: [1, 512 / 1024 / 1536, 768] 43.0 /
  51.2 / 53.2 -> 38.9 / 49.2 / 49.2 us (PyTorch 41.0 / 51.2 / 55.3).

`measure.speed` (us, base -> part 9, H100 cuda): row 25 L 256 18.3 -> 16.3 (20.4), row 25 L 384 26.5 -> below 0.667 of 35.8, row 29 master
L 128 26.5 -> 24.5 (36.9), row 30 L 128 36.6 -> 32.7 (54.4), row 27 master L 128 16.3 -> 14.4 (exempt).

### Built and measured, not kept

| tried | measured (back-to-back, us) | why not |
|---|---|---|
| SwiGLU sigmoid as `tanh.approx.f16x2` (one MUFU op per two units) in the D 64 / 128 inference forwards | [147 456, 64] n 2 24.6 -> 26.6 (eager median), [65 536, 128] 28.7 -> 30.7 | the conversions cost more than the halved MUFU work |
| D 64 `ROLES=1` at n 4 | [98 304, 64] 20.56 -> 22.48 | one SwiGLU warpgroup cannot keep up with four chunks a tile |
| D 64: each SwiGLU warpgroup also writes half the epilogue | n 2 21.23 -> 20.72; n 4 deadlocks (SwiGLU(i + 1) waits for the squeeze of tile i + 1, which waits for epilogue(i)) | removed |
| D 128 `LNH` at n 4 (`fused_sm100a`, row 28) | [16 384, 128] 12.89 -> 13.86, [65 536, 128] 28.62 -> 30.59 | slower |
| D 128 epilogue re-reading the residual x from global memory, x stage released after the LayerNorm, y staged in the dead xn buffer | [65 536, 128] 22.65 -> 24.72 | slower |
| gamma / beta loads without a memory clobber, LayerNorm / epilogue stores without one, residual loads batched beside the TMEM loads | equal or slower (D 128 n 2 22.6 -> 26.8) | no gain |

### What still misses 1.5x (28 eval cases), and why

* Launch-floor cells, H100 20-40 us: rows 35 / 36 / 37 master (3-launch chain or generic + cuBLAS, 16-31 us), rows 25 / 26 / 28 / 38 at
  their smallest pair sizes (one launch of 11-18 us device plus the ~6 us replay floor), as in part 8.
* Row 27 master L 256 (28.5 against 27.7 needed): the D 128 n 2 forward is still ~4.5 us a tile; the stamps put the time in the x-stage
  turnover (two 32 KB stages, the next x load waits for the epilogue's store) and the per-chunk SwiGLU; three restructurings above did not
  move it.
* fp32 rows 27 / 29 L 128-384 and 35 L 1024 / 2048: as in part 7 / 8 (first-tile latency of `tg_fused_h16`, the fp16 chain).

## Round 3, part 10 (2026-11-17 night): the fp32 inference forward `tg_fused_h16` (`cuda:b200_sm100` 24)

Scope: Transition INFERENCE, `cuda` (the Triton rung unchanged). Records (B200 box, `$W/b2511/agents/r3-transition/out/`): `m_p10_eval_<L>.jsonl`
(cuda, every eval case, one exclusive process per length), `viol_p9b.txt` (base: part-9 records) / `viol_p10.txt` (after), `hv_p10.txt`
(kernel A/B against the part-9 build), `acc10_p10.txt` (eval rows 25 / 27 / 29 / 30: 45 GREEN / 9 YELLOW / 0 RED, as part 9),
`acc12_p10.txt` (fp32 training rows 211 / 213 L 256 on the unchanged SAVE build: 2 YELLOW, as before). Probes: `b200cfg/hp.py`
(per-CTA `%globaltimer` stamps of `tg_fused_h16 -DPROBE=1`), `b200cfg/hv.py` (current build against a `-D` variant: eager L2-flushed
median, back-to-back, error vs fp64, max |diff|), `b200cfg/pv2.py` (two builds of the bf16 fused wide forwards), `b200cfg/rg.py` (ptxas).

| eval cases (181; 167 with an H100 median >= 20 us) | base (part 9) | part 10 |
|---|---|---|
| (1) misses of 1.5x | 28 | **26** (row 27 fp32 L 256 / 384 cleared) |
| (1) of them slower than the H100 | 0 | 0 |
| (2) ladder violations | 0 | 0 |
| worst row | 29 fp32 L 128 0.980 | 35 master L 2048 0.966 (row 29 fp32 L 128 0.833) |

### What the stamps showed (D 256 [16 384, 256] n 2, one tile per CTA; D 128 [16 384, 128])

Part 9 called this "first-tile latency"; the stamps put it in three places, none of them the TMA issue rate:
* the LayerNorm. D 128: gamma / beta read by `__ldg` in the store pass (no L1 beside 224 KB of shared memory: an L2 round trip per 32
  columns). D 256 (x from global memory, a row per lane): every load touches 32 rows, 32 L1 wavefronts; the statistics pass took 9 us,
  the store pass 6 us, of a 33 us kernel.
* the epilogue's residual, read a row per lane (the same 32-wavefront pattern): 2.8 us of the D 128 tile, 6 us of the D 256 tile.
* the chunk loop: ~0.8-1.0 us a chunk at D 128 and 1.5 us at D 256, set by the MMA / SwiGLU pipeline (see the rejected items).

### What changed (inference build only; `SAVE_BUILD` keeps every old path, flags default to `!SAVE_BUILD`)

* `LNS`: gamma | beta staged in shared memory by two bulk copies on their own barrier (no thread stalls on the kernel's first cold loads;
  the stamps showed ~4 us for them), at D 256 the CTA's x tiles prefetched into L2 (`cp.async.bulk.prefetch.L2`). Same bits.
* `RXC` + `YST`: the residual slab [32 rows][32 columns] read coalesced (8 lanes a 128-byte row segment) and passed through the warp's
  staging slab; y written from the slab by coalesced stores (y's pointer in `XN` in the inference build), no TMA store / read wait. Same bits.
* `XLN` (D 256): the LayerNorm reads its rows in the tcgen05 16x256b fragment order (a lane quad holds 16 columns of rows r and r + 8),
  8 wavefronts a load instead of 32, statistics summed by quad shuffles, xn stored by `tcgen05.st.16x256b.x4`; the store pass loads the
  next 64 columns while it converts the current ones. xn within the LayerNorm's summation order (max |diff| 1.4e-3 of y, error vs fp64
  equal to four digits).
* `MLANE`: weight / x boxes from a lane each (BRIEF6). Measured the same time as one lane at every width; kept (same bits).

Kernel, eager L2-flushed / back-to-back us, part 9 -> part 10 (`hv_p10.txt`): D 256 [16 384] 41.0 / 33.0 -> 35.1 / 26.9, [65 536]
112.8 / 109.3 -> 100.4 / 96.3; D 128 [16 384] 20.5 / 15.1 -> 18.4 / 12.8, [65 536] 45.3 / 39.9 -> 38.9 / 33.8, [147 456] 79.9 / 74.7 ->
71.7 / 66.4; D 64 [65 536] 26.6 / 22.0 -> 24.6 / 19.5, n 4 [16 384] equal. `measure.speed` (graph, us, H100): row 27 fp32 L 256 44.9 -> 38.9
(58.4), L 384 79.7 -> 71.7 (118.0), row 29 fp32 L 128 40.9 -> 34.7 (41.7), L 256 112.6 -> 100.4 (149.4), row 27 fp32 L 128 20.4 -> 18.3 (21.1).

### Built and measured, not kept

| tried | measured | why not |
|---|---|---|
| D 256 expand as two N = 64 halves with their own barriers and SwiGLU warpgroups (`ABS`, 4 KB boxes, 32-row maps) | [16 384] b2b 26.8 -> 30.9, [65 536] 96.2 -> 112.3 | slower: the chunk period is not the single [a \| b] buffer |
| SwiGLU warps load the chunk's a / b at once and release [a \| b] before the math | equal at every width | the next expand waits for its weight stage / the MMA pipe, not for the release |
| [Wa_c ; Wb_c] interleaved, one 16 KB box a K-block | equal | box count / issue rate is not the bound (also with 32 CTAs: the chunk period stays 1.3-1.5 us at D 256) |
| D 256 with 8 LayerNorm warps (one 16-row half each, packed xn made before the xn-buffer wait) | [65 536] 94.4 -> 97.0, [147 456] 179.1 -> 196.5 (96 registers, spills) | the register cap of 576 threads |
| bf16 D 128 n 2 forward (`tr_fwd_sm100 LNH`) with the weights resident (HID 256: one ring stage per chunk, loaded once) | [65 536] b2b 22.67 -> 22.66, [147 456] 39.2 -> 38.3, [16 384] 10.4 -> 10.6 | no gain at the routed sizes (row 27 master L 256 stays 28.5 against 27.7) |

### What still misses 1.5x (26 eval cases), and why

* fp32 row 29 L 128 (34.7 against 27.8) and L 256 (100.4 against 99.6), row 27 L 128 (18.3 against 14.1): D 256 is now the chunk loop
  (8 x ~1.5 us, the MMA pipe ~70 % busy with in-order expand / squeeze and one [a | b] buffer) plus the steady-state y epilogue on the
  SwiGLU warps (the next tile's first squeeze waits for it: TMEM holds y 256 + [a | b] 128 + xn 128 columns, nothing double-buffered).
  What it would take: the y epilogue on the LayerNorm warps without shared memory (16x256b TMEM loads, coalesced residual / stores),
  or cta_group::2 (half the weights per SM) as the bf16 D 128 forward does.
* Rows 35 / 36 / 37 master, rows 25 / 26 / 28 / 38 smallest sizes, row 27 master L 256, fp32 row 35 L 1024 / 2048: as in part 9 (the chain
  and the generic GEMMs already launch with PDL and load weights before `griddepcontrol.wait`; nothing left to overlap there).

## Round 3, training part (2026-11-17 night, r3-transition-train): the training cases (`cuda:b200_sm100` 25)

Training only (float32 and master; the inference agent owns the forward). Measured with `measure.speed` (graph, L2 flushed), `cuda` +
`triton` + `pytorch` on all 126 training requirement cases, one exclusive measurer process per length on GPU 1; A/B runs alternate a
base tree (b200-2.5.x ddd984081) and the new tree per case. Probes (agent tree `$W/b2511/agents/r3-transition-train/b200cfg/`): `tb.py`
(graph median + per-kernel profile, now with the measurer's fp32 masters: the older probe built `Transition` with bf16 weights and showed
three weight copies that the measured step does not have), `gsave.py` (gradients of one step saved per tree: bit equality to the base,
determinism over two steps), `abspd.sh` / `abt.sh` (base / new alternated), `skp.py` (cuBLAS split GEMMs), `prb.py` (the D 128 backward
built with `-DPROBE=1`: per-CTA `%globaltimer` and the `TRB` trace points), `rp.py` (replica sweep), `acc.sh`.

| | base (m0) | final (m2) |
|---|---|---|
| (1) training misses of 1.5x (H100 >= 20 us) | 23 | 23 (see the A/B below: 2 fixed, 3 border cases within one 2.05 us timer tick) |
| (1) of them slower than the H100 | 0 | 0 |
| (2) ladder violations (training) | 0 | 0 |

### What changed

* **fp32 weight gradients on cuBLAS over row slabs** (`fused_generic_sm100a.WGRAD_SPLIT`, `_slab_products`): cuBLAS' TF32 kernels do not
  split K, and dWab = dab^T xn is [1024, 256] = 4 output tiles over 147 456 rows (0.29 PFLOP/s). One `torch.bmm` over 16 row slabs (8 from
  16 384 to 65 535 rows), the slabs added in slab order by the existing `tg_wsum` launch together with the dgamma | dbeta partials (its
  slab addressing generalised: WS / WS2 with their own slab strides; tg_dw's callers pass the old layout, same bits), so the `tg_colsum`
  launch goes too. GEMMs alone (graph, us, mm -> slabs): [147 456, 256] dWab 290.8 -> 135.2, dWs 98.4 -> 94.2; [262 144, 256] 370.7 ->
  215.6, 159.6 -> 143.4; [589 824, 256] 625.1 -> 460.8, 301.0 -> 286.7; [65 536, 128] 51.2 -> 47.1. `DW_MAX_F32_ELEMS` 2^22 -> 2^21: fp32
  [16 384, 256] now takes the slab cuBLAS instead of tg_dw (+ tg_wsum + the dy hi / lo split): step 206.8 -> 190.5 us; D 128 at 16 384
  rows keeps tg_dw (96.2 either way). Steps (tb.py): [147 456, 256] n 2 fp32 1384.4 -> 1210.4 us, [65 536, 256] 626.7 -> 602.1,
  [65 536, 128] 284.7 -> 276.5. Accuracy (rows 33 / 211 / 213 float32 train, every length): YELLOW 1.00-1.07 x tol, no RED;
  deterministic over two steps.
* **Partial reductions with their loads batched** (same order, bit-identical; `gsave.py` on 11 shapes): the replica / slab / partial-row
  loops of `transition_bwd_reduce` (D 128), `transition_bwd_d64_reduce`, `transition_bwd_d384_reduce`, `tg_wsum`
  (slabs and dgamma | dbeta partials), `tsx_dw`'s tile sum and `transition_lnbwd_w_reduce` ran with runtime trip counts and one load in flight (R x the L2 latency).
  Measured: tg_wsum [16 384, 128] fp32 8.2 -> 4.7 us (step 98.3 -> 96.2), transition_bwd_reduce D 128 n 2 4.5 -> 3.2 us (step 53.2 ->
  51.2); D 128 n 4 / D 64 n 4 unchanged (their reduce time is the backward's tail, below).
* `transition_fused_bwd` makes x's device current first (as the generic surface does): its first call on autograd's device thread
  failed with "invalid device context" in a fresh process whose first backward was this one (`tb.py` D 128 n 4).

### A/B of the border cases (measure.speed, base / new alternated, two rounds each, us; H100 / 1.5 in brackets)

| case | base | new |
|---|---|---|
| row 211 fp32 L 128 / 256 / 384 / 640 | 99.2-100.2 / 286.7 / 562.3-563.0 / 1383.3 | 96.2 / 270.2 / 532.4 / 1357.2-1357.7 |
| row 213 fp32 L 128 / 256 / 384 / 640 | 207.7-208.8 / 630.7 / 1389.1-1401.7 / 3431.4-3471.4 | 190.5 / 598.0 / 1214.5-1217.4 / 3334.5-3354.6 |
| row 211 master L 128 / 213 master L 128 | 53.2 / 98.3 | 51.2 / 96.2 |
| row 217 master L 128 / 384 | 26.5 / 28.4-28.5 | 26.5-26.6 / 26.5 |
| row 218 master L 256 | 32.8-33.6 | 33.0-34.5 (one timer tick either way) |
| row 219 fp32 L 640 / master L 640 | 90.1 / 59.5 | 90.0 / 61.3 (tb.py: 59.4 both trees, twice) |
| rows 217 / 219 fp32 L 128-384, 219 master L 128-384, 218 L 128 / 384 / 640 | equal within 0.1 us | |


### Built and measured, not kept

| tried | measured | verdict |
|---|---|---|
| multi-lane TMA issue (one load per lane, BRIEF6 finding) in `tr_bwd_sm100` (DW and DX producers, weight rings), `tg_gemm` (GATE / GATE_AB / PLAIN), `tsx_gdx` and `tsx_dw` | bit-identical everywhere; D 128 n 2 [16 384] bwd 23.7 -> 23.5 us, n 4 [98 304] 145.5 -> 150.1; tg_gemm fp32 [128, 384] 13.1 -> 13.1, [16 384, 128] 52.2 -> 52.3; tsx_gdx 15.8 -> 15.5, tsx_dw 6.1 -> 6.4, step 32.8 -> 32.8 | these producers are not issue-bound; reverted (`b2/r3trt_mlane_*_rejected.diff`) |
| replica count of the two-role backward at 16 384 rows (`rp.py`) | D 128 n 4: 5 / 6 / 7 / 8 / 9 / 10 / 11 / 12 -> 67.6-69.6 / 63.5 / 59.4-61.4 / 59.4-61.2 / 61.4 / 61.4 / 73.7 / 73.7-75.7 us (two rounds); n 2: 8-20 -> 51.2-53.2 (14 at 51.2); D 64 n 4: 10-28 -> 32.8 (20) .. 41.0 | the tuned values are the best within one timer tick |

### Where the D 128 two-role backward spends its time (`prb.py`, [16 384, 128] n 4, 72 DW + 76 DX CTAs)

| role | CTAs | end of the CTAs (us after the first start) |
|---|---|---|
| DW (15 tiles each) | 72 | 26.7-29.3 (median 27.9) |
| DX, one tile | 24 | 21.3-22.3 |
| DX, two tiles | 52 | 28.7-32.9 |

DX CTA 0's trace (`TRB`): a hidden chunk every ~1.5 us (16 chunks: 1.2 -> 24.7 us); per chunk the gate warps compute for 0.7-1.0 us
(issue-bound: 128 rows x 64 units per chunk on 8 warps) and wait 0.1-0.8 us for the next [a | b] in alternate chunks (the two-slot
[Wa; Wb] ring: a slot lives from the a | b product to the d_xn product of its chunk, then reloads); the LayerNorm-backward epilogue of a
tile runs on 4 warps for 4.6 us (last tile) to 8 us (first tile, beside the next tile's chunks). The step is not MMA-bound (MMA work is
~11-12 us per role); what it would take: a DX role that does not recompute the gate (the DW role's [dA | dB] through global memory, d_xn
over K = 1024 per tile: ~3-4 us per tile instead of ~10), a three-slot weight ring (no shared memory left for it in this layout), and the
last tile's epilogue spread over the idle gate warps (~3 us of the tail).

### What still misses 1.5x, and why

* rows 217 / 219 (single track, fp32 and master) and 210-212 (L 128): 8-11-launch steps at ~5 us per launch at a few row tiles
  (fp32 [1, 128, 384]: GATE_AB + PLAIN 13.1, expand 7.1, squeeze 6.5, tg_dw 6.3, ln_bwds 5.1, prep 3.0, ln 2.4, split 1.9 us); the
  D 128 / D 64 two-role backwards as above.
* row 211 fp32 L 128 (95.9 against 93.3): memory-bound generic step at 16 384 rows (4 tg_gemm 52.3, tg_dw 18.0, ln_bwd 8.1, split 4.9,
  wsum 4.7 us).

## Round 3, training part 2 (2026-11-18, r3-transition-train): the single-track training step in four launches (`cuda:b200_sm100` 27)

Training only. The single-track cells whose forward is the small-M chain (`fused_small_sm100a.CELLS`: fp32 D 384 / 768, bf16 D 768, n 2,
1-6 row tiles; rows 217 float32 and 219 float32 / master) ran 9-10 launches per step (tg_prep, tsx_ln / expand / squeeze, tg_split_tf32,
GATE_AB, PLAIN split K, tg_ln_bwds, tg_dw), each a few us of fixed latency at one to six row tiles (`b200cfg/skl.py`: not launching
tg_dw + tg_ln_bwds alone took fp32 [1, 128, 384] 47.1 -> 34.8 us). Now four:

* **`generic/tg_bwd_small.cu`: the whole backward in ONE persistent launch** (one CTA per SM, 320 threads, software grid barriers on
  monotonic counters -- `(GEN + 1) G`, no reset and no exit ticket): P1 the gate's dh = dy Ws split over K (fp32 slabs, all CTAs) and the
  gate elementwise (h, dA, dB from the slabs and the forward's [a | b]); P2 the d_xn split-K tiles on CTAs [0, NPC) beside the dW tiles on
  the rest; P3 the LayerNorm backward (one row per warp, the slabs added in slab order) and per-CTA dgamma | dbeta rows; P4 their sum.
  The GEMM phases are `tg_gemm.cu` / `tg_dw.cu` themselves: their kernels became `tg_gemm_roles` / `tg_dw_roles` device functions
  (bit-identical for every other caller: `gsave.py` on [16384, 128] n 4 bf16, [16384, 128 / 256] n 2 fp32, [147456, 256] n 2 fp32 equal to
  the round-3 gradients), included in namespaces with new build options: `BMN` (B operand MN-major: Ws and [Wa; Wb] read as stored, no
  transposed copies), `GSTORE` (epilogue rows stored from registers through `store_t`'s transpose instead of staging + TMA store + the bulk
  completion wait). `_small_plan` / `_gate_split` pick the tile widths, K slices and the d_xn / dW CTA split.
* **The weight forms in the chain's LayerNorm launch** (`fused_small_sm100a.chain(cast=)`, the existing `tsx_ln` cast blocks with the expand's
  `wpre = 0`): no tg_prep launch, and only Ws / [Wa; Wb] (no Ws^T / [Wa; Wb]^T).
* **fp32 dy as it is** (`SB_DYRAW`; the MMAs truncate it to TF32, no dy_lo term in dWs, no split phase): rows 217 / 219 float32 train
  YELLOW 1.37-1.60 x tol (with the split 1.00-1.08), master GREEN 0.59-0.90; inside the engine standard. Deterministic over two steps on 9
  shapes. `-DDYRAW=0` / `SB_DYRAW = False` keeps the split (A/B).

Phases measured with `-DPROBE` builds (`b200cfg/sbprb.py`, %globaltimer per CTA, eager, L2 flushed): a `__syncthreads` right after the
GEMM roles released early while lane 0 of a warp was still in its store wait (warp-granular arrival of a diverged warp): every roles call
is followed by `__syncwarp()`. The epilogue's bulk stores dominated the gate (fp32 D 768: MMA done 7.4 us, stored 11.4 -> 10.2 with
`store_t` rows); splitting the gate's K took fp32 [1, 128, 768] 55.3 -> 53.2 and [1, 128, 384] 38.9 -> 36.9.

Single call, graph, L2 flushed (`b200cfg/sbchk.py`, generic -> one launch, us): fp32 [1, 128, 384] 47.1 -> 36.9, [1, 768, 384] 57.5 -> 47.1,
[1, 128, 768] 67.6 -> 53.2, [1, 512, 768] 86.0 -> 76.9; bf16 [1, 128, 768] 49.2 -> 41.0, [1, 256, 768] 53.3 -> 45.1, [1, 768, 768] 63.5 -> 57.3.

Built and measured, not kept: one producer lane per ring slot in the roles (`MLANE`, BRIEF6's TMA issue finding) -- level on all seven
shapes (`docs/b200/rejected/transition_train_mlane_tg_bwd_small.diff`); the last CTA summing dgamma | dbeta through a ticket instead of barrier 3 -- its
serial column sums took 3 us; raw fp32 weights in the forward / backward (no forms at all) -- YELLOW up to 4.16 x tol, too close to RED.

Final run on the merged base (05fe8a72b, `measure.speed` cuda / triton / pytorch, all 126 training cases): 1.5x misses 24 -> 14 (none slower
than the H100; row 217 float32 all six and row 219 float32 L 256 / 384 / 512 / 640 / 768 fixed), ladder 0 -> 0.

What still misses 1.5x: row 219 master L 128-768 (0.69-0.81) and float32 L 128 (0.70): the phases are per-SM-intake bound (each d_xn /
dW / gate tile streams (128 + width) x K operands at ~100 GB/s per SM; the bf16 D 768 phase 2 alone is 13-18 us at L 512-768) and the
forward's three chain launches + the cast take ~20 us at D 768; rows 210-212 L 128 (two-role D 64 / 128 backwards, not changed); row 217
master L 128 / row 218 L 384 at 0.67 (wide D 384 chains, not changed).

## Round 3, part 11 (2026-11-18): `tg_fused_h16` D 256 inference -- the LayerNorm off the tile boundary (`cuda:b200_sm100` 26)

Scope: Transition INFERENCE, `cuda` (Triton unchanged). Base: `b200-2.5.x` b070dfb11 (engine 2.5.14 merged: the H100 Transition tokens
were re-measured, `cuda:h100_sm90` 6, so the 1.5x targets are stricter than in part 10). Records (B200 box,
`$W/b2511/agents/r3-transition/out/`): `m_b14_eval_<L>` / `viol_b14.txt` (base, cuda + triton + pytorch, one exclusive process per length),
`m_p11c_eval_<L>` / `viol_p11c.txt` (after, the same procedure), `acc10_p11.txt` (eval rows 25 / 27 / 29 / 30: 45 GREEN / 9 YELLOW (row 27
fp32, untouched D 128 path, as before) / 0 RED; row 29 fp32 9 / 9 GREEN). Bit-identity: `b200cfg/savecmp.py` (training build SAVE_BUILD=1
at D 64 / 128 / 256: y, xn, mean, rstd; inference at D 64 / 128) against the base source: all bit-identical, so the training forward
(`fusedh{d}s`) is unchanged. Probes: `b200cfg/hvb.py` (A/B against another source file), `b200cfg/hpx.py` (stamps + clock64: the SM ran at
1.69-1.77 GHz in these kernels), `b200cfg/cc.sh` (ptxas only, no GPU).

| eval cases (181; 166 with an H100 median >= 20 us) | base (b14) | part 11 (p11c) |
|---|---|---|
| (1) misses of 1.5x | 28 | **28** (row 29 fp32 L 128 0.819 -> 0.722, still a miss) |
| (1) of them slower than the H100 | 0 | 0 |
| (2) ladder violations | 0 | 0 |
| row 29 fp32 (us, L 128 / 256 / 512 / 1024 / 2048) | 34.7 / 100.4 / 311.3 / 1169.8 / 5152.3 | **30.6 / 87.9 / 276.4 / 1124.8 / 4797.4** |

Other cases moved by one 2.048 us timer tick between processes (row 35 master L 2048 20.4 -> 18.5, row 36 n 4 L 768 18.2 -> 17.2; and in a
cuda-only process row 37 master L 1024 22.4 -> 24.5): their routes were not touched.

### What the stamps showed ([16 384 / 65 536, 256] n 2)

* Steady state: a tile every ~22 us against a chunk loop of ~13 us. After the previous tile's last expand freed the (single) xn buffer,
  the LayerNorm warps ran three of their four passes (group 0's store, group 1's statistics and store) while the MMA idled; then the
  y epilogue (7.7 us at 65 536 rows) on the SwiGLU warps held the next tile's first SwiGLU.
* The first tile: x in at ~4.9 us (cold, L2 flushed), statistics + store to 8.5 us, chunk loop to 22 us, epilogue to 26.5 us.
* The chunk loop: 1.55 us a chunk at 1.77 GHz against 0.87 us of tensor work (measured: doubling the expand added 0.58 us a chunk,
  i.e. the full ~4 K FMA / clk; N = 256 and two N = 128 MMAs cost the same). Weights are not the bound (no weight loads after the first
  round: equal), nor the [a | b] release point (below).

### What changed (D 256 inference build only; flags default on there, off in SAVE_BUILD)

* Both 16-row groups' statistics before the wait for the xn buffer (same bits): [65 536, 256] n 2 96.4 -> 84.3 us back-to-back.
* `LNA`: at a CTA's first tile the SwiGLU warps of half 0 (idle until the first expand) normalize the first 16-row group of each quarter,
  the LayerNorm warps the second (barrier `xa0`): [16 384] 26.8 -> 24.8 back-to-back, eager flushed 34.8 -> 30.7.
* `XS0`: the first tile's x by TMA into the second expand stage + squeeze ring (128 KB, unused until that tile's LayerNorm is done; the
  producer holds those two loads back on `xfull` / `xa0`), the first tile's LayerNorm reading it there: [65 536] 84.3 -> 81.0, [16 384] equal.

Kernel A/B against the base source (`hvb.py`, eager L2-flushed / back-to-back us, max |diff| 0 everywhere): [16 384, 256] n 2 35.8 / 26.9 ->
29.7 / 24.8, [65 536] 101.4 / 96.0 -> 89.1 / 81.8, [147 456] 189.4 / 182.9 -> 168.6 / 159.0, [16 384, 256] n 4 46.1 / 37.6 -> 39.9 / 33.7.

### Built and measured, not kept (diff: `docs/b200/rejected/transition_p11_yln_ab2_abr.diff`)

| tried | measured (back-to-back, us) | why not |
|---|---|---|
| `YLN` (brief item b): the y epilogue on the LayerNorm warps (16x256b TMEM loads, residual / y 8 bytes a lane), y in two 128-column halves with their own full / empty barriers (the squeeze as two N = 128 MMAs) so the next tile's first squeeze waits for half 0 only | [16 384] 26.8 -> 28.3, [65 536] 96.2 -> 103.0, [147 456] 182.8 -> 204.3 (on the part-10 LayerNorm) | 4 warps write y at half the rate of 8, and the LayerNorm warps were already the serial part; no shared memory left for a staged epilogue (224 KB used) |
| `AB2`: [a \| b] as two 64-column halves (expand as two N = 64 MMAs, 32-row weight boxes) so the expand of half s + 1 runs under the SwiGLU of half s | [16 384] 24.8 -> 26.8, [65 536] 81.9 -> 92.7 | the N = 64 MMAs cost more than the overlap gave |
| `ABR`: the SwiGLU warps load the chunk's a / b, release [a \| b], then do the math | equal at every size (as part 10's variant) | the chunk period is not the release point |
| first-tile weight loads held until the xn buffer is full | equal | weights do not compete with the x read |
| no L2 bulk prefetch of the first x tile / statistics unrolled 4 / 16 | equal within 0.7 us | -- |
| (a) the CTA pair (`cta_group::2`, half the weights / hidden per CTA) | not built | the measured bounds above say it does not address them: the weights are not a bound (no-load experiment: equal), the per-SM tensor work is the same, and TMEM per CTA stays y 256 + [a \| b] 128 + xn 128 (the pair splits B's N, not the accumulators); the hidden split would need 256 CTAs for the 128 tiles of L 128 |

Epilogue cost measured by removing its parts (timing only): no residual loads [65 536] 84.2 -> 79.2, no y stores 84.3 -> 70.1, neither 68.7;
[16 384] no stores 24.9 -> 22.8. The y stores are the larger part.

### What still misses 1.5x (28 eval cases), and why

* Row 29 fp32 L 128 (30.6 against 28.3; on the 2.048 us timer grid it needs 26.6): one tile per CTA (128 tiles on 148 SMs): x in at ~4.9
  us (cold), LayerNorm to 8.5 us, chunk loop 13 us (the MMA pipe ~56 % busy: expand, SwiGLU and squeeze stay in series with one [a | b]
  buffer, and the variants that double-buffer it cost more), epilogue 4.5 us (the y stores, ~2 us of it).
* Row 27 fp32 L 128 / 256 (D 128): untouched this part (0.887 / 0.674).
* Rows 35 / 36 / 37 master, rows 25 / 26 / 28 / 38 smallest sizes, fp32 row 35 L 1024 / 2048: as in parts 9 / 10 (launch-bound chains,
  item c not reached this part).

## Round 3, part 12 (2026-11-18): inference -- three B200 builds measured and rejected (`cuda:b200_sm100` stays 26)

Scope: Transition INFERENCE, `cuda`. Base: `b200-2.5.x` 7c74a8902 (engine 2.5.14, part 11 + the training agent's part 9; the inference
routes are byte-identical to part 11, so part 11's `viol_p11c` measurement is the true state: 28 misses of 1.5x among the 166 eval cases with an
H100 median >= 20 us, none slower than the H100, ladder 0). No source change is kept: the violation table is unchanged (below). Probes
(B200 box, `$W/b2511/agents/r3-transition/b200cfg/`): `tl.py` (per-kernel timeline of one L2-flushed graph replay), `lnab.py` / `mcab.py` /
`qmcab.py` (A/B, outputs compared: max |diff| 0 for every variant), `stp.py` (per-CTA `%globaltimer` stamps), `mean.py` / `fine.py`
(the measurer's timing and a fine time: 20 [flush, call] per graph, 10 replays, the flush graph subtracted).

What the timer shows: on this card the graph timing of one call is always a whole number of 2.048 us ticks and the same tick every
replay (the mean of 400 replays equals the median to 0.05 us), i.e. a case measures ceil(T / 2.048 us) ticks; a miss of one tick needs
the true time below the grid point, not "a bit faster". The replay adds ~4.6 us to the device time (`fine.py`: [16 384, 128] n 4 one kernel
11.8 us fine, 16.4 measured).

### Where the misses spend their time (`tl.py`, kernels of one flushed replay, us from the flush end: start / duration)

| cases | route | phases |
|---|---|---|
| row 37 master L 128-768 (D 768 n 2 single stream, 0.81-0.90) | small-M chain `tsx_ln` -> `tsx_expand` -> `tsx_squeeze` (PDL) | L 128: ln 3.1 / 4.2, expand 4.2 / 7.3 (waits for ln, its own work ~4.2), squeeze 7.9 / 8.6 (~5 after the expand) |
| row 37 master L 1024-2048, row 36 n 4 L 1024-3968, row 35 n 2 L 2048 (0.79-0.90) | generic `tg_ln_fwd` -> `tg_gemm` SWIGLU -> cuBLAS squeeze | row 37 L 1024: 3.4 / 3.1, 5.1 / 11.0, 16.1 / 6.1; L 2048: 3.9 / 3.7, 6.8 / 15.8, 22.9 / 7.3; row 36 L 3968: 3.5 / 4.7, 7.8 / 13.7, 21.8 / 7.2 |
| row 35 master L 1536, row 36 L 768 (0.75-0.79) | D 384 small-M chain | L 1536 n 2: 3.1 / 2.7, 5.2 / 5.4, 7.4 / 8.1 |
| rows 25 / 26 / 28 / 38 smallest pair sizes (0.69-0.80) | one fused kernel (`transition_fwd_d64_sm100`, `transition_fwd2_sm100`), one or two tiles per CTA | the replay (~4.6 us) plus one tile's latency (x in, LayerNorm, the chunk loop, the epilogue) |
| row 27 / 29 fp32 L 128-256, row 35 fp32 L 1024 / 2048 (0.67-0.89) | `tg_fused_h16` one tile per CTA / the fp16 chain | part 11 stamps: first-tile latency (x in ~4.9 us, LayerNorm to 8.5, chunk loop 13, epilogue 4.5) |

### Built and measured, not kept

| tried | measured | why not |
|---|---|---|
| (a) the LayerNorm inside the small-M expand (`tsx_expand.cu -DLNF`, bf16 inference: per-row statistics from global memory by the 8 epilogue warps, each x K-block TMA-loaded in the swizzled layout and normalised in place, handed to the MMA by an 8-warp barrier; the chain in two launches; the same xn bits, outputs max \|diff\| 0). Diff: `docs/b200/rejected/transition_p12_lnf_in_expand.diff` | graph median us (base -> LNF): [1, 128 / 384 / 768, 768] 18.4 / 18.4 / 22.5 -> 26.6 / 34.8 / 38.9; [1, 768 / 1536, 384] n 2 14.3 / 16.4 -> 22.6 / 24.6; [1, 768, 384] n 4 18.4 -> 26.6. Without the statistics and the normalisation (timing builds, wrong numbers) 16.4 at [1, 128, 768] and 14.3 at [1, 768, 384] | stamps ([1, 128, 768], 96 CTAs): statistics 5.3 us after griddepcontrol.wait (every CTA reads the whole 128-row tile: 96 x 196 KB), then the normalisation paces the MMA at ~0.5 us a K-block; gamma / beta staged in shared memory changed nothing. Even the free bound (no statistics, no normalisation) is one tick at D 768 and none at D 384: the separate LayerNorm launch is not what stands between these cases and 1.5x |
| (a) the tile's xn / h shared by TMA multicast over a cluster of the CTAs that read it (`tsx_expand.cu` / `tsx_squeeze.cu -DMC`: one CTA issues each K-block to the cluster, stages freed by multicast tcgen05.commit to every CTA's empty barrier). Diff: `docs/b200/rejected/transition_p12_tma_multicast_expand_squeeze.diff` | fine us, base -> best multicast build: expand [1, 128 / 384 / 768, 768] 12.9 / 15.1 / 16.7 -> 13.4 / 15.9 / 17.8, [1, 768 / 1536, 384] n 2 10.1 / 12.2 -> 10.8 / 13.3, n 4 12.8 -> 13.8; squeeze 12.9 / 14.8 / 16.7 -> 15.8 / 17.1 / 18.5, D 384 10.1 / 12.3 / 12.8 -> 11.4 / 12.9 / 15.2 (larger clusters slower still) | the operand intake of the tile's A rows is not the bound (one tile per cluster: equal or slower), and the coupled ring (a stage waits for every CTA of the cluster) and the cluster placement cost more |
| one issuing lane per weight ring slot in the D 128 pair forward (`tr_fwd_sm100.cu -DWML`, the SWA agent's ~600 clocks per TMA issue). Diff: `docs/b200/rejected/transition_p12_fwd2_multilane_weights.diff` | fine us [16 384 / 65 536 / 589 824, 128] n 4: 11.80 / 27.69 / 168.1 -> 11.84 / 27.73 / 170.0 | the weight issue is not the bound of this kernel (as part 10 found for `tg_fused_h16`'s MLANE) |
| the expand / squeeze prologue TMA loads from one lane per stage (same idea in the small-M chain) | mean us 17.76 -> 17.59 ([1, 128, 768]), equal elsewhere; graph medians unchanged | no tick moved |

### Misses of 1.5x against the H100 (H100 >= 20 us), part 11 -> part 12

Unchanged (no route or kernel changed): 28 of 166 eval cases, none slower than the H100, ladder 0. Worst: row 37 master L 384 0.903
(20.4 us against 22.5), row 36 master n 4 L 2048 0.898, row 27 fp32 L 128 0.887. What the measurements above say each needs: the D 768 /
D 384 single-stream chains a shorter expand + squeeze (neither the LayerNorm launch nor the A-operand intake is the bound); the one-tile
pair kernels and `tg_fused_h16` a shorter single-tile latency (the replay alone is ~4.6 us of the 16.4 us cases).

## Round 3, part 13 (2026-11-18): inference -- the hidden split of one tile over a cluster, re-measured with a multi-lane producer (rejected, `cuda:b200_sm100` stays 26)

Scope: Transition INFERENCE, `cuda`. Base: `b200-2.5.x` 05fe8a72b (engine 2.5.14; the merge changed no Transition source or H100 token, so
part 11's `viol_p11c` table is the true state: 28 misses of 1.5x among 166 eval cases with an H100 median >= 20 us, none slower than
the H100, ladder 0). Idea tested: cut the single-tile latency by splitting the HIDDEN dimension of each 128-row tile over a cluster (LayerNorm
-> expand slice -> SwiGLU -> squeeze partial on chip -> partials reduced in distributed shared memory in rank order, one launch). That is
part 6's `tsx_one.cu`; part 6 blamed its weight phase on a ~55 GB/s-per-SM ring turnover, which the SWA agent's later finding (one thread
issues ~one TMA per 600 clocks) would explain. Rebuilt with (a) lane s of the producer warp issuing ring slot s (`-DML=1`) and (b) the
partial staging loading a whole column block from tensor memory before one wait. Probes: `b200cfg/onep13.py` (per-CTA `%globaltimer`
stamps, ML 0 / 1), `b200cfg/one.py` (graph median, L2 flushed; error vs fp64; determinism). Diff: `docs/b200/rejected/transition_p13_one_launch_multilane.diff`.

Stamps, [1, 2048, 384] n 4, SK 6 (96 CTAs), us from the first CTA's entry, median over CTAs (ML 0 -> ML 1):

| phase | ML 0 | ML 1 |
|---|---|---|
| LayerNorm rows sent | 3.58 | 3.33 |
| MMA warp past xn | 4.35 | 4.10 |
| expand + SwiGLU + squeeze done (590 KB of weights per CTA) | 15.10 (10.75) | 14.85 (10.75) |
| first half of the partials received | 18.94 | 18.69 |
| stored | 24.32 | 23.81 |

The weight phase is the same 10.75 us with one or six issuing lanes: the per-CTA intake is not TMA-issue bound here. It is the pipeline:
the 384-column squeeze accumulator and the 128-column expand accumulator fill the 512 tensor-memory columns, so the expand of step c + 1
waits for the SwiGLU warps to drain step c, and the squeeze slots of the ring wait behind it (the chain's expand, ~200 KB per CTA in ~4.2 us,
runs at the same ~50 GB/s per SM). The partial reduction (128 x 384 fp32 per CTA, 5/6 of it moved) stays at ~8.5 us.

Graph medians (L2 flushed, us; current route -> one launch at SK 4 / 6 / 8 / 12; outputs within one bf16 ulp of the route, error vs fp64
3.2-3.5e-3 against PyTorch's 4.6-6.0e-3, deterministic):

| case (H100 / 1.5) | route | SK 4 | SK 6 | SK 8 | SK 12 |
|---|---|---|---|---|---|
| [1, 768, 384] n 4 (15.3) | **16.4** (chain) | 38.9 | 30.7 | 30.7 | 28.7 |
| [1, 1024, 384] n 4 (15.6) | **20.5** (generic) | 38.9 | 32.8 | 32.8 | 49.2 |
| [1, 2048, 384] n 4 (18.1) | **24.6** (generic) | 38.9 | 32.8 | 57.3 | 71.7 |
| [1, 1536, 384] n 2 (13.8) | **16.4** (chain) | 30.7 | 28.7 | -- | 45.1 |
| [1, 2048, 384] n 2 (14.1) | **20.5** (generic) | 30.7 | 28.7 | -- | 65.5 |

Why the cluster-per-tile split cannot win these cases (measured above, and by arithmetic for D 768): every tile's cluster streams ALL the
weights (3 D H x 2 B: 3.5 MB at D 384 n 4, 7.1 MB at D 768 n 2) over its SK SMs, so the per-CTA weight bytes are tiles x W / CTAs, the same as
the chain's, with fewer SMs at a few tiles (row 37 L 128 is ONE tile: at most 16 SMs, 442 KB each at SK 16 = ~8 us at the measured ~55 GB/s
per SM, against the whole chain's ~9 us on 148 SMs); and it adds the partial exchange (128 x D fp32 per CTA, more bytes per tile than the h
round trip through L2 it removes). At D 768 the squeeze accumulator (768 fp32 columns) does not fit tensor memory either. The output-column
variant (h slices all-gathered over DSMEM) moves 15 x 24 KB in per CTA at D 768 SK 16 (~3.6 us at the measured ~100 GB/s bulk-copy rate)
and needs the full 393 KB h per CTA streamed through shared memory: not built. Nothing routed; the route is byte-identical to part 12.

### Misses of 1.5x against the H100 (H100 >= 20 us), part 12 -> part 13

Unchanged (no route or kernel changed): 28 of 166 eval cases, none slower than the H100, ladder 0 (worst row 37 master L 384 0.903,
row 36 master n 4 L 2048 0.898, row 27 fp32 L 128 0.887).

## Round 3, part 14 (2026-11-18): inference -- the CTA pair (`cta_group::2`) for the small-M chain (`cuda:b200_sm100` 28)

Scope: Transition INFERENCE, `cuda`. Base: `b200-2.5.x` d586ed4b7 (engine 2.5.14). True state before (`out/viol_r14m0.txt`, cuda + triton +
pytorch, one exclusive process per length on GPU 0): 28 misses of 1.5x among 166 eval cases with an H100 median >= 20 us, none slower than
the H100, ladder 0 (the part-11 table). Probes (B200 box, `$W/b2511/agents/r3-transition/b200cfg/`): `sxp.py` (the chain's three
launches captured in a graph, built with `-DTSX_STAMP=1`: per-CTA `%globaltimer` stamps per phase, L2 flushed), `sxsum.sh`, `pcab.py`
(module route against the pair chain: graph median and mean of 300 flushed replays, outputs compared), `det14.py`, `acc14.yaml`.

### What the stamps showed (current chain, [1, 128, 768], us from the first expand CTA's entry, medians over CTAs)

| phase | LN done (expand's griddepcontrol.wait) | expand first K-block | last K-block | partials exchanged | h stored | squeeze wait | squeeze K loop | partials exchanged | stored |
|---|---|---|---|---|---|---|---|---|---|
| tsx_ln -> tsx_expand (32, 2) -> tsx_squeeze (64, 4) | 2.9 | 3.4 | 4.4 | 5.6 (1.0) | 6.0 | 7.1 | 7.5 -> 8.3 | 10.0 (1.5) | 10.4 |

* The two K-split exchanges in distributed shared memory cost 2.5 us of the 10.4.
* Operand intake: one lane per ring stage instead of lane 0 (the SWA finding) changed nothing ([1, 384, 768] expand 3.87 -> 3.94 us;
  `docs/b200/rejected/transition_p14_sx_multilane_stamps.diff`). What did: the NUMBER of TMA boxes per SM. The same pair kernel with its
  64-row xn K-blocks as one 8 KB box each, then as one 3-D box of 2 / 4 K-blocks (16 / 32 KB): [1, 384, 768] chain end 12.45 -> 9.70 us
  (expand K loop 1.9 -> 0.54 us), [1, 128, 768] 9.54 -> 8.64. The per-SM box rate (~0.15-0.2 us a box), not the bytes, paced these
  kernels.

### What changed (`fused_small_sm100a.PAIR*`, `sm100/widths/tsx_expand2.cu`, `tsx_squeeze2.cu`)

* **The CTA pair.** Measured layout (`docs/b200/rejected/transition_p14_pair_m128_layout_test.diff`): an M = 128 `cta_group::2` product
  takes A rows 0-63 from the leader's shared memory and 64-127 from the peer's, B's N halves from each, and leaves in EACH CTA its own 64
  rows of the whole N, folded over the 128 lanes (lanes 0-63 the leader's B columns, 64-127 the peer's). So a pair over one 128-row tile
  gives each SM half the tile's rows and half the pair's weights, with the whole K in tensor memory: the K split and its exchange are gone.
  - `tsx_expand2.cu` (`-DUH` hidden units per CTA, `-DKBX` K-blocks per 3-D box): one pair per (tile, 2 UH units), [a | b] of 4 UH
    columns, SwiGLU of the CTA's 64 rows x 2 UH units; weights before griddepcontrol.wait.
  - `tsx_squeeze2.cu` (`-DNBW` output columns per pair, `-DKBX`): one pair per (tile, NBW columns), the whole hidden width, + x.
  - host `tmap(..., kb_box)` (`transition_sm100.cu`, additive): a 3-D map (element, row, K-block) whose box lands as kb_box consecutive
    128-B-swizzled K-block slabs.
* Routed: bf16 inference, D 768 n 2, 1-6 row tiles (row 37 master L 128-768) in `fused_small_sm100a.chain` (`PAIR_CELLS`), picks
  `PAIR_PICK` from the sweep (UH, NB): 1 tile (16, 32), 2-3 tiles (32, 32), 4-6 tiles (64, 64); KBX 4. Output within one bf16 ulp of the
  K-split chain (max |diff| 0.031), bitwise deterministic over 5 calls, accuracy row 37 master eval 18 / 18 GREEN.

`pcab.py` (graph median us, route -> pair): [1, 128 / 256 / 384 / 512 / 640 / 768, 768] 18.4 / 18.4 / 20.5 / 20.5 / 20.5 / 22.5 ->
14.3 / 16.4 / 16.4 / 18.4 / 18.4 / 20.5.

### Built and measured, not kept

| tried | measured | why not |
|---|---|---|
| pair chain on D 384 n 2 / n 4 (rows 35 / 36, [1, 768-1536, 384]) | 14.3 / 16.4 / 16.4 against the same | equal; from 12 tiles (two waves) slower: [1, 1536 / 2048, 384] n 4 20.5 / 24.6 -> 22.5 / 30.7, n 2 L 2048 20.4 -> 22.5 |
| the LayerNorm inside the pair expand (`tsx_expand2l.cu`: x by 3-D TMA, 8 warps normalise the CTA's 64 rows in shared memory, released to the leader; weights as 3-D boxes, two N = 2 UH products a K-block) | graph [1, 128 / 384 / 768, 768] 16.4 / 18.4 / 20.5 -> 18.4 / 20.5 / 22.5 | x in at 1.9 us, but the in-place normalisation 3.6 us (stores 2.4, shuffle trees 1.7: timing builds `-DLNX`); the separate LayerNorm launch is cheaper. `docs/b200/rejected/transition_p14_pair_expand_ln_fused.diff` |
| per-tile release counters instead of the squeeze's grid-completion wait (expand CTAs `red.release` per tile, LayerNorm adds the targets, squeeze spins with acquire + `fence.proxy.async.global`) | +2 us every case ([1, 128, 768] mean 16.4 -> 17.2, [1, 768, 384] 14.3 -> 16.4) | the spinning early squeeze CTAs cost more than the ~1.2 us completion hop. `docs/b200/rejected/transition_p14_pair_counters_ln.diff` |
| squeeze NB 16 (96 CTAs at one tile) / KBX 8 | [1, 128, 768] 16.4 against 14.3 at NB 32; KBX 8 equal to 4 | -- |

### Misses of 1.5x against the H100 (H100 >= 20 us), before -> after

`out/viol_r14m0.txt` (before) -> `out/viol_r14p1.txt` (after; cuda + triton + pytorch, every eval length in its own exclusive process):
28 -> **27** misses of 1.5x among the 166 eval cases with an H100 median >= 20 us, none slower than the H100, ladder 0 -> 0.

| row 37 master [1, L, 768] (H100 us) | before | after |
|---|---|---|
| L 128 (22.3) | 18.3 (0.819) | **14.3 (0.64: cleared)** |
| L 256 (22.2) | 18.3 (0.823) | 16.3 (0.732) |
| L 384 (22.5) | 20.5 (0.908) | 18.2 (0.807) |
| L 512 (23.1) | 20.4 (0.881) | 18.4 (0.797) |
| L 640 (23.5) | 20.5 (0.871) | 20.3 (0.862; the A/B process read 18.4) |
| L 768 (27.7) | 22.4 (0.808) | 20.4 (0.734) |

Worst now: row 36 master n 4 L 2048 0.899, row 27 fp32 L 128 0.887, row 35 master n 2 L 2048 0.874.

Measured floor of the remaining row-37 cases (pair chain, [1, 384, 768]): replay ~4.6 us + LayerNorm launch to its last row 2.9 us + pair
expand 1.7 us + the expand -> squeeze grid-completion hop ~1.2 us + pair squeeze 2.3-3.0 us (h of the tile per SM through 6 boxes) =
~13.6 us device, 18.2 measured; the H100 / 1.5 grid point needs 14.3 (one tick below 15.0). The pieces each attempt in this part
removed (in-expand LayerNorm, counters instead of the completion hop) measured slower; what is left is the squeeze's h intake (a K split
of the pair squeeze with a 2-way exchange, not built) and the LayerNorm launch's cold x read.

## Round 3, training part 3 (2026-11-18, r3-transition-train): `tg_bwd_small` barriers, phases and plans; faster replica reduces (`cuda:b200_sm100` 29)

Training only. Base `b200-2.5.x` 1d4b36754 (engine 2.5.14 H100 tokens), merged up to 80572b7d4. Probes (agent tree `b200cfg/`): `sbprb.py`
(per-CTA `%globaltimer` phase stamps of a `-DPROBE` build; stamps 18-21 now time barrier 4), `sbsweep.py` (last-CTA end of the probe build
over phase-2 plans, `GATES=` gate plans, `KS=` / `NPCS=` lists), `fx.py` / `j_fx.sh` (graph step, alternated module-attribute sets, mean
and median of 300 replays), `j_sb.sh` (probe per `-D` flag set), `final3.sh`.

| | base (m4) | final (m7) |
|---|---|---|
| (1) training misses of 1.5x (H100 >= 20 us) | 14 | 10 |
| (1) of them slower than the H100 | 0 | 0 |
| (2) ladder violations (training) | 0 | 0 |

Fixed: row 219 master L 128 / 768, float32 L 128, row 217 master L 768 (border). Row 219 master L 640 61.3 -> 53.0, L 512 53.1 -> 49.0,
L 384 47.0 -> 45.0, L 256 44.9; row 212 61.3 -> 59.3.

### What changed

* **Barriers** (`tg_bwd_small.cu`): the arrival is one `atom.add.release.gpu` (`BREL`; was `__threadfence` + `atomicAdd`); the
  `fence.proxy.async.global` pair only at the barrier whose data the next phase TMA-loads (barrier 1; `PFSEL`). The probe showed the
  ~4 us between "P2 end" and barrier 2 is not the proxy fence (PFSEL level there): the stamp after a diverged warp's `__syncthreads` is
  early; the real per-CTA phase end is the `b2 fence` stamp.
* **Gate elementwise** (`EWU` 2): two vectors per thread per load round, slab loads predicated on `KSG` (the unused ones re-read slab 0
  before): bar 1 -0.3 / -0.8 us at L 128 / 640. `EWU` 3 level, 4 slower (registers).
* **LayerNorm phase on several warps per row** (`ln_rows<WPR>`, `LNW`): at D 768 and <= 2 rows per CTA, 6 / 3 warps per row, all d_xn
  slabs in one load round, the row sums met in shared memory in part order (dx bits change; deterministic). bf16 / fp32 [128, 768]
  -1.5 / -2.2 us. Two warps per row (L 512 / 640) +1.4 us, D 384 [384, 384] +0.7 us, and the build that carries the multi-warp path ran
  the one-row-per-warp phase 2 us slower at L 640 / 768 (kernel-wide code generation): the host builds `-DLNW=0` there.
* **Measured plans** (`SB_PLAN_TUNED`, `SB_GATE_TUNED`): the model's K split ignores the slabs' write + LayerNorm read traffic, and the
  split gate's elementwise pass costs more than the unsplit tiles' intake from 384 rows. Last-CTA end (us, auto -> tuned): phase 2
  bf16 [512 / 640 / 768, 768] 30.3 -> 29.4, 36.8 -> 32.7, 34.5 -> 34.1; gate (64-wide unsplit GATE_AB tiles) [384 / 512 / 640 / 768]
  24.3 -> 22.2, 26.8 -> 23.8, 28.5 -> 26.8, 29.1 -> 27.1, [256] (64, 3) 22.1 -> 21.3, fp32 [128, 768] (64, 6) 22.5 -> 21.2.
* **Replica reduces** (`tr_bwd_sm100.cu`, `widths/tbwd_d64.cu`, `RQ` 16 / `GQ` 32 loads in flight, same summation order): D 128 n 4
  [16 384] tail after the backward 3.8 -> 2.4 us (row 212 61.3 -> 59.3); D 128 n 2 / D 64 n 4 level.

Accuracy (rows 210-212 / 217 / 219, float32 + master train, every length): 30 GREEN, 18 YELLOW (row 211 float32 1.00-1.04, rows 217 / 219
float32 1.37-1.59 x tol: the DYRAW numbers of part 2), no RED. Deterministic over two steps on 9 shapes (`gsave.py`).

### Built and measured, not kept

| tried | measured | verdict |
|---|---|---|
| d_xn tiles on 2-CTA clusters, each CTA TMA-multicasting half of the [Wa; Wb] panel (`tg_gemm.cu -DMCB`, per-CTA intake (128 + BNP / 2) K) | correct; bf16 [640, 768] (256, 2, 256, 36) 26.98 vs 26.7 unpaired; no plan of 256-640 rows better | level: the d_xn operand bytes are not what bounds phase 2 (`rejected/transition_train_p3_mcb_dxn_pairs.diff`) |
| dW tiles with 2 K-blocks per TMA box / ring stage (box 64 x 128 rows, part 14's box-count finding) | bit-identical; graph means level within 0.3 us on six shapes | the dW CTAs are not the critical role (`rejected/transition_train_p3_dw_kblock_boxes.diff`) |
| proxy fences only where TMA reads next (`PFSEL`) | level | kept (fewer fences, same time) |
| two warps per LayerNorm row; `EWU` 3 / 4 | +1.4 us; level / slower | off |

### What still misses 1.5x, and why

* row 219 master L 256 / 384 / 512 / 640 (0.67-0.70): forward chain + per-step cast ~15-20 us (the LayerNorm launch with the fp32 -> bf16
  cast of the 14 MB of masters is 6.3 us at L 128; the expand waits for the whole launch), backward 21-27 us with a 1.5-2 us cost per grid
  barrier (three barriers). The H100 casts per step too (its version-keyed form cache is inference only).
* rows 210 / 211 / 212 L 128 (two-role D 64 / D 128 backward): the DX role recomputes the gate (0.7-1.0 us per chunk) and the replica
  reduce is a separate tail launch (2.4-3.5 us); the H100 builds dA / dB in registers as the next wgmma's A fragment -- the B200 analogue
  (A operand from tensor memory, `tcgen05.mma` ts) was not built in this part.
* rows 217 master L 128, 218 L 384 (0.670): the D 384 `tsx_gdx` + `tsx_dw` pair; `SX_DW_NT_FORCE` 128 / 192 and the two-launch gate +
  d_xn measured level or slower (graph means 26.6 / 29.0 / 33.0 us unchanged).

## Round 3, part 15 (2026-11-18): inference -- part 14's box-count finding applied to the other kernels, the pair squeeze K split (all rejected, `cuda:b200_sm100` stays 28)

Scope: Transition INFERENCE, `cuda`. Base: `b200-2.5.x` 80572b7d4 (engine 2.5.14 + part 14); the Transition sources and H100 tokens are
those of part 14, so `out/viol_r14p1.txt` is the true state: 27 misses of 1.5x among 166 eval cases with an H100 median >= 20 us, none
slower than the H100, ladder 0. No source change is kept: the table is unchanged. Probes (B200 box, `$W/b2511/agents/r3-transition/b200cfg/`):
`kbab.py` (module A/B of the KBX / WKBX / SWIGLU_KBX builds, graph median and mean of 300 L2-flushed replays, outputs compared), `sqk.py`
(pair squeeze unsplit against `-DSQK=2`, determinism by a repeat replay), `fine.py` / `mean.py` (fine device time of one call, median).

### Built and measured, not kept (all bit-identical to the route or within one bf16 ulp; deterministic)

| tried | measured (graph median, us; A / B / A / B in one process) | verdict |
|---|---|---|
| D 384 small-M chain (`tsx_expand.cu` / `tsx_squeeze.cu -DKBX=3`): xn, Wa, Wb and h, Ws of a ring stage as ONE 3-D box each (a and b as two N = CH products); the same build in the fp16 chain of fp32 rows 35 (`-DH16`) | [1, 768, 384] n 4 16.4 / 16.4; [1, 1536, 384] n 4 20.5 / 20.5, n 2 16.4 / 16.4; fp32 [1, 1024 / 2048, 384] 16.4 / 16.4, 18.4-20.5 / 18.4; max \|diff\| 0. Taking the generic ranges (`--gen`): [1, 1024, 384] n 4 20.5 -> 22.5, L 1536 / 2048 n 4 and L 2048 n 2 equal | level: these chains load the whole K in their prologue (6 K-blocks), the box count is not their bound |
| pair chain (row 37) with the WEIGHT K-blocks as 3-D boxes too (`tsx_expand2.cu` / `tsx_squeeze2.cu -DWKBX=1`: [Wa_0; Wa_1] and [Wb_0; Wb_1] as two N = 2 UH pair products, folded as the N = 4 UH one) | [1, 128, 768] 14.3 -> 16.4 (mean 14.5-14.7 -> 16.0-16.4); L 256-768 equal (mean +0.0-0.4) | slower at one tile, level elsewhere (the weights load before griddepcontrol.wait, under the LayerNorm) |
| generic SWIGLU GEMM (`tg_gemm.cu -DKBX=2`, single CTA, bf16): A, Wa, Wb of 2 K-blocks per stage as one 3-D box each (rows 35 / 36 / 37 from L 1024) | [1, 1024 / 1536 / 2048 / 2976 / 3968, 384] n 4 20.5 / 20.5-22.4 / 24.6 / 32.8 / 30.7 against 20.5 / 20.5 / 24.6 / 32.8 / 30.7; n 2 L 2048 20.5 / 20.5; [1, 1024 / 1536 / 2048, 768] 24.6 / 24.7-26.6 / 32.8 against 24.6 / 24.7 / 32.8 | level or a tick slower (half the ring depth); the first variant timed in a process reads one tick lower in every case (the clock, not the build) |
| pair squeeze with the hidden width split over two pairs of a 4-CTA cluster (`tsx_squeeze2.cu -DSQK=2`: pair 1 sends its fp32 partial to pair 0 by `st.async`, added in pair order) | [1, 256, 768] (NB 64) 16.4 / 16.4 (mean 16.44 against 16.40); L 128 14.3 -> 16.4; L 384 / 512 / 640 / 768 +4.1 / +4.1 / +4.1-6.0 / +8.0 (4-CTA clusters fit ~3/4 of the SMs: a second wave); NB 32 slower still | the h / Ws intake per SM is not what is left of the squeeze at L 256; at L 384+ the cluster shape costs a wave |
| a persistent pair kernel for 12+ tiles (row 36 n 4 L 2048, item 3) | not built | part 14: the pair chain is 22.5 / 30.7 us at 12 / 16 tiles against the generic 20.5 / 24.6; even one ideal wave of it would not reach the 16.4 us grid point that row 36 L 2048 needs (24.5 now) |

Diffs: `docs/b200/rejected/transition_p15_kbx_chains_pair_weights.diff`, `transition_p15_kbx_tg_gemm_swiglu.diff`,
`transition_p15_pair_squeeze_ksplit.diff`. The part-14 gain came from the 64-row xn K-blocks of the pair EXPAND (8 KB boxes issued after
griddepcontrol.wait, on the critical path); boxes that are already 16 KB, or that load under the previous launch, gain nothing.

### How far the one-tick cases are from their grid point (`fine.py`: device time of one call; graph median in brackets)

| case (needed median) | fine us (median) |
|---|---|
| row 37 master [1, 128, 768] (solved, 14.3) | 10.49 (14.3) |
| row 37 master [1, 256, 768] (14.3) | 11.43 (16.4) |
| row 37 master [1, 384, 768] (14.3) | 12.60 (16.4) |
| row 37 master [1, 768, 768] (18.4) | 15.37 (20.5) |
| row 37 master [1, 1536, 768] (22.5) | 19.62 (24.6) |
| row 36 master [1, 768, 384] n 4 (14.3) | 12.15 (16.4) |
| row 35 fp32 [1, 1024 / 2048, 384] (14.3 / 18.4) | 11.43 / 14.65 (16.4 / 18.4) |
| row 27 master [1, 256, 256, 128] (26.6) | 22.92 (28.7) |
| row 26 master [1, 256, 256, 64] n 4 (16.4) | 14.39 (18.4) |
| row 28 master [1, 128, 128, 128] n 4 (14.3) | 11.80 (16.4) |
| row 27 fp32 [1, 128, 128, 128] (12.3) | 12.67 (18.4) |

The 14.3 grid point lies between 10.49 and 11.43 us of device time: row 37 L 256 and row 35 fp32 L 1024 need 0.5-0.9 us less, the others
1.5-2 us.

## Round 3, training part 4 (2026-11-18, r3-transition-train): the DX role fed by the DW role's [dA | dB] (built, rejected); programmatic-dependent launches (`cuda:b200_sm100` 31)

Training only. Base `b200-2.5.x` 08cc8fc89 (engine 2.5.14 H100 tokens). Probes (agent tree `b200cfg/`): `dxgchk.py` (one step with the
DXG backward off / on: bits and determinism), `dxprb.py` + `dxpa.py` (`-DPROBE` stamps of the DXG input role and of DW CTA 0), `fx.py`
(now also `f4.` = `fused_sm100a` and `;`-separated overrides), `final4.sh`.

| | base (m8) | final (m9) |
|---|---|---|
| (1) training misses of 1.5x (H100 >= 20 us) | 11 | 12 (row 212 59.3 -> 57.2 us, still 0.07 us above H100 / 1.5; row 218 master L 512 34.8 -> 36.6: the D 384 cells are not changed here, its median flips between two 2.05 us timer ticks, p10 34.7 both runs) |
| (1) of them slower than the H100 | 0 | 0 |
| (2) ladder violations (training) | 0 | 0 |

### The brief's item 1: TMEM-resident [dA | dB] and the `.ts` d_xn product

The D 128 two-role backward's input (DX) role ALREADY runs d_xn with the A operand in tensor memory (`umma_ts`, `tr_bwd_sm100.cu` since the
b200/ladder-final port: the gate writes [dA | dB] back over dh as bf16 and the d_xn product reads it from TMEM -- the H100 register-A
fragment's counterpart). What it repeats is the gate itself (a | b and dh MMAs + the SwiGLU backward per chunk), because the [dA | dB] the
weight (DW) role computes lives in OTHER CTAs: a DW CTA owns one 64-unit hidden slice of every tile it takes (its dWab / dWs accumulators
are 128 + 64 of its 512 TMEM columns, kept over all its tiles), while d_xn of a tile needs all slices. A CTA owning both would need the
dW accumulators of every slice (8 x 192 columns at n 4): no single-CTA TMEM form exists. So the redesign was built with the [dA | dB]
handed over through global memory (`-DDXG`, `docs/b200/rejected/transition_train_p4_dxg_dab_through_global.diff`):

* DW role: an idle warp stores the (tile, slice) [dA | dB] (one 32 KB 3-D TMA box) after the gate, then releases a per-tile counter;
  `g_empty` counts the store's read so the gate's buffer is not overwritten early.
* DX role: no gate. Per half-chunk [dA_j] / [dB_j] (16 KB, ring slots with their own issuing lanes, the lane polls the tile's counter
  before its first load) and Wa_j / Wb_j (16 KB 3-D boxes, multicast over the pair); `umma_ss` d_xn in the TMEM-A build's K order (d_xn
  bit-identical in the first version); x / dy staged by TMA (two buffers); the LayerNorm backward split by columns over two warpgroups with
  each row's dgamma | dbeta terms summed over the CTA's tiles in TMEM (one row reduction at the end instead of 8 warp reductions per tile).
* Correct and deterministic (dx bit-identical in the first version, then within one bf16 ulp; weight gradients within 1e-6 relative).

Measured (graph, L2 flushed, `fx.py`, step us; base = the one-kernel build):

| [16 384, 128] | base | DXG best (replicas) | other replica counts |
|---|---|---|---|
| n 2 (row 211) | 51.1-51.2 | 49.15 (20, 24) | 28: 51.2, 32: 67.6 |
| n 4 (row 212) | 57.3 | 61.4 (13) | 11: 63.5, 15: 70.0 |

Why (probe, n 2, 28 replicas): the DX tile is bound by its per-SM intake -- 128 KB [dA | dB] + 128 KB [Wa; Wb] + 64 KB x / dy per tile
at the ~100 GB/s an SM takes in (2.8 us from a tile's counter to its d_xn, the epilogue 3.3 us per tile after the TMEM accumulation; the
first epilogue, x / dy read row-per-thread from global memory, was L1-bound: 5.8-10 us per tile, 59.4 -> 41.0 us with it skipped); the DW
role runs its gate and its dW products in series (one h / [dA | dB] buffer: no shared memory for a second) at 2.0 us per (tile, slice),
and the store adds ~2 us before a tile's counter. At n 4 the DX tile needs 576 KB: slower than the gate recompute it replaces. Not routed.

### What changed (`cuda:b200_sm100` 31)

* **Programmatic-dependent launches** (same bits on every gradient, `gsave.py` against the part-3 files): `tr_bwd_sm100.cu` and
  `widths/tbwd_d64.cu` wait (`griddepcontrol.wait`) after their prologue (barrier init, TMEM allocation, cluster sync; gamma now loaded by
  the epilogue warps after the wait) and are launched as programmatic dependents (`fused_wide_sm100a.BWD_PDL`; D 128 n 4 in
  `transition_sm100.cu`), the D 128 n 2 reduce too; `tr_fwd_sm100.cu` initialises its barriers and allocates tensor memory, then
  waits (a no-op for the inference launches, which stay ordinary launches), and its training launches (D 128 n 2 / n 4, save = 1) are
  programmatic dependents; `transition_cast3` triggers its dependents at entry. `measure.speed` (cuda, master train, L 128), base tree
  (b200-2.5.x) / this tree alternated, three rounds: row 212 59.3 / 59.2 / 57.3 -> 57.2 / 57.2 / 57.2 us, row 211 51.1 / 51.2 / 51.1 ->
  50.9 / 49.2 / 49.2, rows 209 / 210 level (28.6 / 32.7). (`fx.py` graph means, BWD_PDL on / off: [16 384, 64] n 2 29.2 / 30.2, n 4
  32.8 / 33.1, [65 536, 128] n 2 97.8 / 98.3, n 4 134.1 / 134.4; fx.py's n 2 graphs also alternate between 49.15 and 51.2 us for
  identical code, so its n 2 A/B is not evidence.)
* **D 128 n 4 with 8 weight-role replicas up to 128 row tiles** (`fused_sm100a.DW_REPL_SMALL`): three alternated rounds, mean 56.7-56.8 ->
  56.4-56.6 us (9 stays above; 7 / 10 slower).

Accuracy (rows 209-212 master train, every length): 24 / 24 GREEN (dx of D 128 n 4 within one bf16 ulp of part 3: the 8-replica dW sums regrouped; D 64 and D 128 n 2 gradients bit-identical to part 3). Deterministic over two steps on 8 shapes.

### Built and measured, not kept

| tried | measured | verdict |
|---|---|---|
| DXG: [dA | dB] through global memory, no gate in the DX role (above) | n 2 51.2 -> 49.15, n 4 57.3 -> 61.4 | rejected (diff above): per-SM intake bound |
| DXG epilogue reading x / dy row-per-thread from global memory | 5.8-10 us per tile (L1-bound) | replaced by TMA staging within DXG |
| the last tile's LayerNorm backward on all twelve math warps (columns 0-63 epilogue warps, 64-95 / 96-127 the two idle gate warpgroups, row sums met in the dead weight ring; dgamma bits unchanged, dx within one bf16 ulp) | step 49.2 -> 59.4 us inlined (more spills), 53.2 noinline; n 4 57.3 -> 61.4 | rejected (`rejected/transition_train_p4_last_tile_epilogue_12warps.diff`) |
| n 2 / D 64 n 4 replica counts with the PDL launches | n 2 12 / 14 / 16: 49.2 / 49.2 / 51.2; D 64 n 4 17 / 20 / 23: 34.8 / 32.8 / 36.9 | unchanged |

### The brief's items 2-4

* Row 219 forward cast: the fp32 -> bf16 cast blocks already run in the SAME launch as the LayerNorm blocks (`tsx_ln`, concurrent), and
  the expand needs both xn and the bf16 weights, so a separate cast launch (PDL) cannot let the expand start earlier; what remains is the
  cast's own memory time (14 MB fp32 read + 7 MB written per step at L2-flushed rates, 4.7-4.8 us for the launch at L 256-640) or an
  expand that converts fp32 weights on its way into shared memory (bit-identical forms; `tsx_expand.cu`, the inference agent's file).
  Not built.
* Row 219 backward barriers (probe, [512, 768] bf16): gate end 7.2 -> barrier 1 8.2, phase 2 (d_xn CTAs, intake-bound) to 17.7 -> barrier
  2 18.7, LayerNorm 21.9 -> barrier 3 23.2, dgamma 23.3 us. A phase-2 -> phase-3 flag per row tile (LayerNorm rows on the dW CTAs that end
  at ~12 us) saves at most ~1 us (the d_xn CTAs end within 2 us of each other); not built in this part.
* Rows 217 L 128 / 218 L 384 / 217 L 768: within one 2 us timer tick of the bound; their small-M kernels (`tsx_gdx` flags, `tsx_dw`
  dWs before its wait) already overlap; not changed.

### What still misses 1.5x, and why

* row 211 master L 128 (51.1 against 40.1): the two-role D 128 n 2 backward is ~23 us at 16 384 rows and both roles are near their own bounds (DW gate + dW products in series at ~2 us per (tile, slice); DX gate recompute, or with DXG the per-SM intake); with ~10 us of forward + cast and the replay floor, 40 us needs a backward of ~12 us -- a different decomposition (e.g. cta_group::2 DX pairs that split the weight panel by N, halving the per-SM weight intake), not built.
* row 212 master L 128 (57.2 against 57.1): as row 211; 0.07 us.
* row 210 master L 128 (32.6 against 29.2): D 64 two-role backward + its reduce; the PDL launches gave 0.3 us.
* row 211 float32 L 128 (98.2 against 92.6): the generic TF32 step (unchanged).
* row 219 master L 256-640 (0.67-0.70): see items 2-3 above (cast already concurrent with the LayerNorm; phase 2 intake-bound, barriers ~1 us each).
* rows 217 L 128 / 768, 218 L 384 / 512 (0.669-0.671): one timer tick.

## Round 3, training part 5 (2026-11-18, r3-transition-train): box counts, in-kernel sums, gate reorders (`cuda:b200_sm100` 33)

Training only. Base `b200-2.5.x` 72b1c4ef2 (engine 2.5.14 H100 tokens; the Transition sources of part 4, so `out/viol_m10.txt` is the
true state before). Probes (agent tree `b200cfg/`): `p5a.sh` .. `p5e.sh` (gradient bits against part 4 + determinism with `gsave.py`, then
`fx.py` in-process A/B through `fused_wide_sm100a.BWD_AB_FLAGS` (new: an extra "bwdab" cubin of the D 64 / D 128 n 2 backward built with
those flags) and the D 128 n 4 step alternated against a second tree `r3-transition-trb` holding the A/B build), `resid.py`, `p5h.sh`
(replica sweep + `measure.speed` rows 210-212 L 128 alternated), `final5.sh`.

| | base (m10) | final (m11) |
|---|---|---|
| (1) training misses of 1.5x (H100 >= 20 us) | 12 | 11 (row 218 master L 512 back on its lower tick: not changed here) |
| (1) of them slower than the H100 | 0 | 0 |
| (2) ladder violations (training) | 0 | 0 |

### What changed (`cuda:b200_sm100` 33; every gradient bit-identical to part 4 on 7 shapes, deterministic)

* **DW gate before its wait (`GOV`, `tr_bwd_sm100.cu` and `widths/tbwd_d64.cu`)**: the D 128 weight role waited for the previous tile's dW
  products (`g_empty`) BEFORE loading dh / a / b from tensor memory, so its `gate_read` arrival (and with it the next tile's dh / [a | b]
  products) queued behind them; now the loads, the arrival and all 16 gate pairs run first and only the shared-memory stores wait (D 64
  already loaded first; its arithmetic now moves before the wait too).
* **DX gate's a | b part before dh_j (`GPRE`, D 128)**: s, l = a s and u = l (1 - s) + s (gate_pair16's own ops) are computed while dh_j
  is still in the tensor pipe; after `dh_full` only dB = g l and dA = (g b) u remain.
* `fx.py` (graph, L2 flushed, mean of 300 replays): [16 384, 128] n 2 50.4-50.6 -> 49.3-49.4 us; n 4 55.5-55.6 -> 54.4-54.9 us (median
  55.3 either way); D 64 level (32.8 / 28.7). `measure.speed` (median): row 212 57.31 / 57.39 -> 57.28 / 57.22 us (needs 57.13: its
  samples sit on two timer ticks, 55.3 and 57.3, p10 56.7 -> 55.3), row 211 master 51.17 both, row 210 32.7 both.

## Round 3, part 16 (2026-11-18): inference -- the near misses, phase by phase (`cuda:b200_sm100` 32)

Scope: Transition INFERENCE, `cuda`. Base: `b200-2.5.x` 7b360e407 (engine 2.5.14 + part 15 + training part 3; no inference route changed, so
`out/viol_r14p1.txt` was the true state: 27 misses of 1.5x among 166 eval cases with an H100 median >= 20 us, none slower than the H100,
ladder 0). Probes (B200 box, `$W/b2511/agents/r3-transition/b200cfg/`): `fab16.py` (fine device time of one call + the measurer's graph
median / mean of 400 flushed replays, under build variants set before the first build: extra `-D` flags per chain kernel, inference flags of
the fused forwards; output compared bitwise against a pinned reference, determinism by a repeat), `p24.sh` (variants alternated, two rounds,
one exclusive process each), `tl.py` (kernel timeline), `sxp.py` (per-CTA `%globaltimer` stamps; `NOFLUSH=1` keeps the L2 warm; the D 384
`tsx_expand` / `tsx_squeeze` now carry the same `TSX_STAMP` points as the pair kernels, off by default).

### What the stamps showed (us from the first expand CTA's entry, medians over CTAs)

| case | LN done (expand wait) | expand first stage | expand stored | squeeze entry | squeeze wait | squeeze K loop | stored |
|---|---|---|---|---|---|---|---|
| [1, 256, 768] pair (32, 32), 96 + 96 CTAs | 1.09 | 2.53 | 4.03 (max 4.48) | 2.62 (co-resident) | 5.50 | 6.18 -> 7.46 | 7.87 |
| [1, 384, 768] pair (32, 32), 144 + 144 | 1.12 | 2.62 | 4.16 (max 4.48) | 5.22 (after the expand CTAs leave) | 6.18 | 6.94 -> 8.26 | 8.67 |
| same, L2 warm (`NOFLUSH`) | 1.12 | 1.73 | 3.30 | 4.35 | 5.34 | 5.89 -> 7.20 | 7.65 |
| [1, 768, 384] n 4 chain (64, 1) / (32, 2), 144 + 144 | 0.67 | 1.34 | 3.49 (max 3.84) | 4.61 | 5.12 | 5.54 -> 8.10 | 9.18 |

* The LayerNorm launch (`tsx_ln`, bf16 D 768): the SASS kept 5 of the 6 gamma / beta chunks behind `rsqrt`, i.e. a second DRAM round trip
  (the L2 is flushed) after the row statistics. At D 384 the compiler had hoisted them already (no gain there).
* The pair expand's first stage lands 1.4 us after its wait with a cold L2 and 0.6 us with a warm one: the cold weight intake (each SM
  takes 192 KB, ~100 GB/s per SM) paces it, not the issue rate (one box per lane changed nothing, below).
* From 3 tiles the squeeze CTAs cannot be resident beside the expand's (192 KB + 200 KB of shared memory): they enter ~0.9 us after the
  expand CTAs leave and pass their wait ~1.5 us after the last expand CTA (0.9 us when co-resident at 2 tiles).

### What changed (`widths/tsx_ln.cu -DLNPF`, default on in the bf16 and fp16 builds, off in the TF32 build)

gamma / beta (and the fp16 build's scale) are loaded into registers at kernel entry, before griddepcontrol.wait (they are parameters),
instead of after the statistics. Bitwise identical (max |diff| 0 against the old build on every case below, deterministic). Fine us /
graph median (old -> new, `fab16.py`, two alternated rounds):

| case (H100 / 1.5) | fine | graph median |
|---|---|---|
| row 37 master [1, 128, 768] (14.9) | 10.37 -> 9.43 | 14.3-16.4 (mean 15.6) -> 14.3 (mean 14.4) |
| row 37 master [1, 256, 768] (14.8) | 11.42 -> 10.49 | 16.4 -> **14.3** |
| row 37 master [1, 384, 768] (15.0) | 12.62 -> 11.38 | 16.4 -> 16.4 |
| row 37 master [1, 512 / 640, 768] (15.4 / 15.7) | 14.27 / 14.62 -> 13.13 / 13.59 | 18.4 -> 18.4 |
| row 37 master [1, 768, 768] (18.5) | 15.21 -> 14.16 | 20.5 -> **18.4** |
| row 35 fp32 [1, 1024, 384] (15.6) | 11.42 -> 11.05 | 16.4 -> 16.4 |
| row 35 fp32 [1, 2048, 384] (20.1) | 14.62 -> 14.16 | 18.4 (20.4 in r14p1) -> 18.4 |
| row 36 master [1, 768, 384] n 4 (15.3) | 12.14 -> 12.14 | 16.4 -> 16.4 |

### Built and measured, not kept

| tried | measured | verdict |
|---|---|---|
| item 1: the D 128 backward's TMA boxes (`BX`): dy / xn / x / dx as one 64 x 128 x 2 3-D box per tile (12 -> 3 boxes in the DX role, 8 -> 2 received per DW tile), Ws_j one 64 x 128 box, Wa_j / Wb_j one 64 x 64 x 2 box each landing as [Wa kb0][Wa kb1][Wb kb0][Wb kb1] (a \| b as two N = 64 products, d_xn with an 8 KB d-block stride): per chunk 6 -> 3 received boxes | bit-identical; n 2 49.66 -> 49.98, n 4 55.48-55.53 -> 55.54-55.57 us (three alternated rounds) | level: the box count is not this kernel's bound (diff `rejected/transition_train_p5_bx_boxes_ired_sums.diff`) |
| item 3: the partial sums inside the backward (`IRED`: the R replicas of a slice meet on a counter and each sums 1 / R of it, the first 16 DW CTAs then sum dgamma \| dbeta after the DX counter; no reduce launch; same order, same bits; 74 resident 2-CTA clusters checked) | n 2 48.95 -> 49.75, n 4 55.76 -> 58.67 us | slower: the PDL reduce already overlaps the backward's tail (same diff). A first version with monotonic launch-numbered counters hung: launches with other R / ndx share the counter buffer; self-resetting counters fixed it |
| DW gate on three warpgroups (`G3`: the partial-store warps 8-11 take units 48-63 until the last tile) | n 2 49.06 -> 49.82, n 4 54.7 -> 56.0 | slower (`rejected/transition_train_p5_dw_gate3_gprew_l2prefetch.diff`) |
| DW gate's a \| b part under the dh product (`GPREW`, its own a \| b commit) | n 2 49.26 -> 49.40, n 4 54.6 -> 55.2 | slower (same diff) |
| L2 prefetch of the later tiles' inputs from idle lanes (`PFL`: DW warp 3, DX warp 3 lanes 1-31) | n 2 50.04 -> 50.32, n 4 54.9 -> 56.2 | slower (same diff) |
| replicas after GOV / GPRE | n 4 7 / 8 / 9: 57.2 / 54.7 / 55.4; n 2 12 / 14 / 16 / 18: 49.3 / 49.4 / 50.1 / 50.5 | unchanged (8, 14) |
| item 2: `cta_group::2` DX pairs | not built | the probe (`prb.py`, eager) shows the WEIGHT role is the longer one after GOV / GPRE: n 2 DW ends 20.6-22.5 us vs DX 16.1-16.8 (one tile) / 20.4-21.9 (two); n 4 DW 31.6-34.1 vs DX 21.7-31.8. DW CTA 0's trace: each tile's xn / dy load is issued only when the tile two before freed its slot (two 64 KB input slots, no shared memory for a third), ~1.6 us apart; halving the DX weight intake does not move that |
| item 4: row 211 fp32 (generic TF32) | not changed | its tg_* launches are already programmatic dependents (round 3 part 5); part 15's K-block boxes on tg_gemm measured level |

### What still misses 1.5x, and why

* row 212 master L 128 (57.2 against 57.13): mean -0.9 us, the median still on the upper of two ticks.
* row 211 master L 128 (51.1 against 40.1) and row 212: the D 128 DW role is paced by its two input slots (load of tile i + 2 waits for
  tile i's dW products, then the DRAM / L2 latency) on top of the duplicated a | b / dh products and gate in both roles.
* row 210 (32.6 against 29.2), row 211 fp32, rows 219 / 217 / 218: as part 4.

| gamma / beta of the D 128 fused forward (`tr_fwd_sm100.cu -DGBR`) through the LayerNorm threads' registers, so the cold load no longer holds the setup `__syncthreads` (the weight producer's first TMA) | fine [1, 128, 128, 128] n 4 11.76-12.08 -> 12.06-12.07; [1, 256, 256, 128] n 2 23.38-23.42 -> 23.33-23.36 | level; not kept (`rejected/transition_p16_fwd2_gamma_registers.diff`) |
| an L2 prefetch of the squeeze's Ws by the pair expand CTAs (`cp.async.bulk.prefetch.L2`, each CTA 1 / grid of Ws; before / after the wait) | fine [1, 128-768, 768]: +0.0-0.6 us before the wait, -0.03..+0.5 after | the extra DRAM traffic competes with the expand's own weights; `rejected/transition_p16_ws_l2_prefetch.diff` |
| one weight box per lane in the pair chain's first ring round (expand: 24 lanes, squeeze Ws: 20 lanes) | fine within +-0.05 us on [1, 128-768, 768] | level: per-SM intake, not the per-thread issue rate; `rejected/transition_p16_pair_one_box_per_lane.diff` |
| lane s issues ring stage s in the D 384 / fp16 chain (`tsx_squeeze -DQML`, `tsx_expand -DEQML`: the lone producer lane had issued six 16 KB h boxes after the wait) | fine [1, 768, 384] n 4 12.09-12.11 -> 12.10-12.12; fp32 [1, 1024 / 2048, 384] +0.07 / +0.0-0.1 | level (part 12 found the same); `rejected/transition_p16_sx_lane_per_stage.diff` |
| the pair chain past 6 tiles (row 37 L 1024-2048) | not reached by `PAIR_CELLS` alone (the generic route is chosen before `chain`); part 14 measured the two-wave pair chain slower | not pursued |
| other pair picks at 2 tiles (all squeeze CTAs resident beside the expand): (UH, NB) (32, 64) / (64, 32) / (64, 64) / (16, 64) | fine 10.46 -> 10.69 / 12.00 / 12.65 / 13.61 | slower; `PAIR_PICK` unchanged |
| a preferred shared-memory carveout of 100 % for `tsx_ln` (a `transition_carveout` cubin symbol read by the host) | level in the slow state below | not kept |

### The measurer reads row 37 L 256 / L 768 one tick above the probes: a slow state of the card, not the case

`measure.speed` (`m4r.sh`, rows 35 / 37 alone) reads row 37 master L 256 / L 768 at 16.3 / 20.4 us, while the same case built by the
measurer's own `config.build` / `inputs` / `make_step` / `measure` in a fresh process (`b200cfg/mprobe2.py`) reads 14.2 / 18.3. In one process
the first cuda measurement reads 14.2, later ones 16.3 (the squeeze's first CTA starts ~1.3 us later in the kernel timeline); a pytorch
graph, a side-stream matmul or layer_norm before it moved the switch earlier or later but none of allocation shifts, cuBLAS init, eager
forwards on the default stream, or a carveout preference removed it; with nothing in between it can still switch at the third measurement.
It behaves as the clock / power state the brief describes. The official timing (pytorch, then cuda, in a long process) is taken in that
state, so these cases need ~0.3 us more margin than the fresh-process fine times suggest.

### Misses of 1.5x against the H100 (H100 >= 20 us), before -> after (`out/viol_r14p1.txt` -> `out/viol_r16p1.txt`, cuda + triton + pytorch, every eval length in its own exclusive process)

27 -> 27 misses among the 166 eval cases with an H100 median >= 20 us, none slower than the H100, ladder 0 -> 0. Moved (us, ratio):

| case (H100 / 1.5) | before | after |
|---|---|---|
| row 37 master L 384 (15.0) | 18.2 (0.807) | 16.3 (0.724) |
| row 37 master L 512 (15.4) | 18.4 (0.797) | 18.3 (0.793) |
| row 37 master L 640 (15.7) | 20.3 (0.862) | 18.3 (0.779) |
| row 37 master L 768 (18.5) | 20.4 (0.734) | 19.4 (0.699; 18.4 / 20.4 alternating; rows-only run 20.4) |
| row 37 master L 1024 (19.1) | 24.3 (0.849) | 22.5 (0.788) |
| row 37 master L 128 (14.9) | 14.3 (cleared) | 14.3 (fine 10.4 -> 9.4: no longer on the edge) |
| row 37 master L 256 (14.8) | 16.3 | 16.3 (fresh-process 14.2) |

## Round 3, part 17 (2026-11-18): inference -- chunk-major DSMEM exchange, the fp32 (fp16-operand) chain on CTA pairs (`cuda:b200_sm100` 36)

Scope: Transition INFERENCE, `cuda`. Base: `b200-2.5.x` f2d941e47 (engine 2.5.14; no inference source changed since part 16, so
`out/viol_r16p1.txt` is the true state before: 27 misses of 1.5x among 166 eval cases with an H100 median >= 20 us, none slower than
the H100, ladder 0). Probes (B200 box, `b200cfg/`): `fab17.py` (fab16 with RANDOM weights: fab16 kept the zero-initialised squeeze, so
its "max |d| 0" only compared x; fab17 also prints the error against an fp64 reference), `p17.sh` / `sw17*.sh` (variants alternated, two
rounds, exclusive, GPU 1), `sxp.py` stamps, `run17.sh` (accuracy rows 35-37 + measure.speed every eval length, cuda / triton / pytorch).

### What changed

* **Chunk-major receive layout for the K-split partials** (`tsx_expand.cu` / `tsx_squeeze.cu -DCHM`, default 1; `tsx_store.cuh chm16`):
  the receive area is [chunk][128 rows] x 16 B, so one warp `st.async` writes one contiguous 512-B run (was 32 scattered 16-B chunks
  in a row-major swizzled area); the reads match. Same bits (random weights, max |d| 0 on every case; training chains too).
  Stamps [1, 1024, 384] n 2 squeeze (64, 2): acc_full -> red_full 1.06 -> 0.64 us. Fine us (CHM 0 -> 1): fp32 [1, 1024, 384] 11.15 ->
  10.61, fp32 [1, 128, 768] 11.16 -> 10.24, fp32 [1, 512, 768] 15.25 -> 14.66, fp32 [1, 768, 384] 9.60 -> 9.31; bf16 [1, 768, 384] n 4
  (squeeze (32, 2), 8 KB a CTA) level.
* **fp16-operand pair squeeze** (`tsx_squeeze2.cu -DH16`, fp32 x / out, y = acc sc[3] + x; NB 128 added: two 32-column epilogue
  passes, 64 TMEM columns) routed by `fused_small_sm100a.H16_PAIR_SQ` (D 384 1-3 tiles 32, 8 / 12 tiles 64, 16 tiles 128; D 768 1-3
  32, 4-6 64, 7-12 128): no hidden split, so no exchange; at 16 tiles NB 128 halves the per-SM Ws intake of the unsplit (64, 1) squeeze.
* **fp16-operand pair expand** (`tsx_expand2.cu -DH16`, h = fp16(silu(a s1) (b s1) s2), tsx_expand's scheme) routed by `H16_PAIR_EX`
  (D 384 1 tile UH 16, 2-12 tiles 32; D 768 1 tile 16, 2-3 tiles 32, 4-6 tiles 64).

Fine device time of the fp32 chain (us, part 16 build -> part 17; random weights): D 384 L 128 / 256 / 384 / 512 / 640 / 768 / 1024 /
1536 / 2048: 8.15 / 8.13 / 8.27 / 8.9 / 9.1 / 9.6 / 11.15 / 11.6 / 14.3 -> 7.44 / 7.71 / 7.86 / 8.53 / 8.88 / 9.25 / 9.78 / 11.13 /
13.74; D 768 L 128 / 256 / 384 / 512 / 1024 / 1536: 11.16 / 12.5 / 13.9 / 15.25 / 19.7 / 20.9 -> 9.5 / 10.65 / 11.75 / 13.6-14.2 / 17.73 /
19.1. Outputs within 6e-6 of the K-split chain, error vs fp64 unchanged to three digits, deterministic. Accuracy rows 35 / 36 / 37
master + float32 eval: 35 GREEN / 12 YELLOW / 0 RED (row 35 fp32 L 128 / 256 GREEN 0.998 -> YELLOW 1.0009 x tol; the rest as before).

### Built and measured, not kept

| tried | measured | verdict |
|---|---|---|
| other K-split picks of the fp16 chain with the cheaper exchange (expand (128, 1) / (64, 2) / (32, 1), squeeze (32, 1) / (64, 4) / (32, 2)) at fp32 [1, 1024 / 2048, 384] | 10.56 / 14.19 against 11.0-14.3 / 14.2-20.9 | the automatic picks stay best |
| pair squeeze / expand of the fp16 chain where not routed (D 384 4-6 tiles squeeze, 16 tiles expand; D 768 8 / 12 tiles expand) | level, or +0.8-3 us | not routed |
| brief item 2 (`tg_fused_h16` setmaxnreg + 8 LayerNorm warps, row 29 fp32 L 128) | not built | row 29 L 128 is one tile per CTA: the first-tile LayerNorm already runs on 8 warps (part 11 `LNA`); the 8-warp variant addresses the steady state, and the case needs ~4 us (26.6 grid point) |
| brief item 3 (D 64 64-row tiles) | probe only: [M, 64] n 4 fine 7.07 / 9.64 / 12.23 / 14.67 / 14.89 us at 1 / 2 / 3 / 3.46 / 4 rounds of 148 tiles | a perfectly balanced last round would give ~13.4-13.7 at row 26 L 256 (needs ~12.4 for the 16.4 grid point), row 25 / 38 need 3-4 us: no case clears; not built |

### Misses of 1.5x against the H100 (H100 >= 20 us), before -> after (`out/viol_r16p1.txt` -> `out/viol_r17p2.txt` / `viol_r17p3.txt`)

27 -> **25** misses among 166 eval cases, none slower than the H100, ladder 0 -> 0. Cleared: row 35 float32 L 1024 (16.3 -> 14.3) and
L 2048 (20.4 -> 18.4). Row 37 master L 256 reads 16.3 again (part 16's slow state: route untouched). Untouched routes read one tick
slower than in part 16 in this card state (row 35 master L 2048 18.4 -> 20.4, row 26 L 256 18.4 -> 20.3, row 28 L 128 16.3 -> 18.2;
two processes each).

## Round 3, part 18 (2026-11-18): inference -- accuracy audit of parts 9-17, the bf16 chain with the CTA-pair squeeze past 6 tiles (`cuda:b200_sm100` 38)

Base: `b200-2.5.x` ea6821710 (engine 2.5.14 + part 17). Probes (B200 box, `b200cfg/`): `aud.py` (`measure.accuracy` run unchanged --
random weights, fp64 gold, tol = max(err_pytorch, fixed) -- and observed: per cuda call the fp64 rms error, an output sample and the
whole output's fp64 sum / sum of squares, saved per build), `audcmp.py` (build against build), `audrun.sh`; `fab17.py` / `p17.sh` A/B.

### The audit (part 17 found that `fab16.py` kept the zero-initialised squeeze: its "max |d| 0" compared only x)

`measure.accuracy` rows 25-38, eval, master + float32, every length and depth (190 cuda cases measured, 3 skipped: the fp64 gold of row
31 L 2976 / 3968 and row 32 L 2048 does not fit), on seven builds of the inference route: before part 9 (`ea7ee3d1f^1`), after parts 9 /
10 / 11 / 14 / 16 / 17 (the merge commits; the training merges and engine 2.5.14 in between included). Every build: 163 GREEN, 27 YELLOW
(fp32 rows 33 / 35 / 37 at 1.00-1.007 x tol), 0 RED; the worst case row 37 float32 L 2048 1.0067 x tol in every build. Per change (cases
whose output changed at all; the rest equal over the whole output):

| change | cases changed | max \|d\| vs the build before | err / tol, before -> after | fp64 rms |
|---|---|---|---|---|
| part 9 (fwdi: D 64 ROLES, D 128 LNH, D 256 LN2 / XPAR / WPAR) | rows 29 / 30 master, every L (18) | 0.016-0.031 (one bf16 ulp) | 0.808 / 0.545 / 0.47-0.48, unchanged | unchanged |
| part 10 (tg_fused_h16 LNS / RXC / YST / XLN / MLANE) | row 29 float32, every L (9) | 0.9e-3-1.1e-3 (XLN's summation order; LNS / RXC / YST / MLANE same bits) | 0.999-1.000, unchanged | 2.70e-4, unchanged |
| part 11 (tg_fused_h16 D 256 LN before the wait, first x tile by TMA) | none | 0 | -- | -- |
| part 14 (bf16 D 768 pair chain, KBX boxes) | row 37 master L 128-768 (6) | 0.0078-0.016 (one bf16 ulp) | 0.435-0.463, unchanged | unchanged |
| part 16 (tsx_ln LNPF) | none | 0 | -- | -- |
| part 17 (CHM exchange; fp16 pair squeeze / expand) | row 35 float32 L 128-384 / 1024 / 1536, row 37 float32 L 128-768 (11) | 2.6e-6-6e-6; row 37 float32 L 128 1.9e-4 (the D 768 one-tile fp16 pair expand) | row 37 fp32 L 128 1.002 -> 1.005 (YELLOW both), the rest within 0.001 | unchanged |

Every adopted change is inside the standard; the "same bits" claims of parts 11 / 16 and of part 17's CHM hold (now checked with random
weights). Claims that rested on probes with the zero squeeze (rejected items only): part 12's multicast expand (`mcab.py` / `qmcab.py`)
and part 16's rejected list (`fab16.py`: GBR, the Ws L2 prefetch, one box per lane in the pair chain, QML / EQML, other pair picks, the
carveout) -- rejected anyway, their output claims unverified.

### What changed (`fused_small_sm100a.INFER_CELLS` / `BF16_PAIR_SQ`, `cuda:b200_sm100` 38)

bf16 inference at D 768 n 2 7-12 tiles, D 384 n 2 16 tiles, D 384 n 4 8-12 tiles runs the small-M chain (tsx_ln -> the K-split
tsx_expand pick -> the squeeze on a CTA pair, `tsx_squeeze2.cu` bf16, NB 128 / 128 / 64) instead of the generic kernels with the cuBLAS
squeeze. `fab17.py` (graph median of 400 flushed replays, fine device time in brackets, us): D 768 8 / 12 tiles 22.5 / 24.5 (18.7 / 19.7)
-> 20.5 / 22.5 (16.7 / 18.0); D 384 n 2 16 tiles 20.5 (15.2) -> 16.5-18.4 (12.6); D 384 n 4 8 / 12 tiles 20.5 / 20.5 (15.8 / 16.3) ->
18.4 / 18.4 (12.8 / 13.7). Accuracy: the five changed cases GREEN, err / tol 0.35-0.55 (was 0.43-0.56), one bf16 ulp from the generic
output; deterministic.

### Built and measured, not kept

| tried | measured | verdict |
|---|---|---|
| the K-split squeeze of the chain at these tiles | D 768 8 / 12 tiles 24.4 / 24.6; D 384 n 4 12 tiles 18.5 | the pair squeeze is faster or equal |
| the pair expand (`tsx_expand2.cu` bf16, UH 16 / 32 / 64) instead of the K-split expand | +1-17 us on every case (D 768 12 tiles UH 64 26.6, UH 16 45.1) | slower: 2 UH units x 2 CTAs a tile need tiles x H / UH CTAs, past one wave; diff `rejected/transition_p18_bf16_chain_pair_expand.diff` |
| the chain at D 768 16 tiles, D 384 n 4 16 / 31 tiles | 32.8 -> 34.7-38.9; 24.5 -> 24.6; 30.7 -> 36.9 | not taken: the expand grid is past one wave |

### Misses of 1.5x against the H100 (H100 >= 20 us), before -> after (`out/viol_r18a.txt` -> `out/viol_r18b.txt`)

25 -> **24** misses among 166 eval cases, none slower than the H100, ladder 0 -> 0. Cleared: row 37 master L 1536 (24.6 -> 22.5, 0.616).
Moved: row 35 master L 2048 20.4 -> 18.3 (0.964 -> 0.867), row 36 L 1024 20.4 -> 18.3 (0.869 -> 0.781), L 1536 20.5 -> 18.4 (0.777 ->
0.698), row 37 L 1024 24.4 -> 22.4 (0.855 -> 0.783); row 36 L 2048 read 22.6 (0.829; route untouched, 24.5 before).

## Round 3, part 19 (2026-11-18): inference -- the persistent pair expand (`cuda:b200_sm100` 40)

Base: `b200-2.5.x` f1ea32732 (engine 2.5.14 + part 18; no inference source changed since, so `out/viol_r18b.txt` is the state before:
24 misses of 1.5x among 166 eval cases with an H100 median >= 20 us, none slower than the H100, ladder 0). Probes (B200 box, `b200cfg/`):
`fab19.py` (fab17 + `px=UH:TG:SQ` persistent-expand override, `xp=` extra flags, `WATCHDOG` for first runs of new kernels, `TL=1` kernel
timeline), `p19.sh` (shared first pass / two alternated exclusive rounds, GPU 1), `sxp19.py` (per-CTA `%globaltimer` stamps of
`tsx_ln -> tsx_expand2p [-> squeeze]`), `bld19.py`.

### What changed (`sm100/widths/tsx_expand2p.cu`, `fused_small_sm100a.PEX`, `tsx_ln.cu` save 3)

* **The persistent pair expand** (`tsx_expand2p.cu`, bf16 inference). Grid = (hidden chunk j of 2 UH units, group g) pairs, at most one
  wave (UH 32: 24 chunks x 3 groups = 72 pairs at H 1536). Each pair keeps its weight panel ([Wa_r; Wb_r], the whole K) RESIDENT in
  shared memory (loaded before griddepcontrol.wait, one barrier per ring stage so the first products start on the first K-blocks) and
  loops over the 256-row blocks g, g + 3, ...; the products are **M = 256** pair products (each CTA one 128-row tile: the tensor cores at
  full rate; the M = 128 pair products of `tsx_expand2.cu` run them at half rate). The blocks' xn K-blocks stream through an A ring
  (3-D boxes of KBX K-blocks, lane s issues ring slot s across block boundaries); the accumulator is double-buffered in TMEM (2 x 4 UH
  columns), so the SwiGLU of block k overlaps the products of block k + 1. SwiGLU sigmoid 0.5 tanh(a / 2) + 0.5 (one MUFU op, the D 64 /
  128 kernels' TR_SIG 1 form; `-DESIG=0` the kit form). Rows past m are zero-filled by TMA and not stored, so a ragged row count
  (row 36 L 2976) runs too.
* **Squeeze**: cuBLAS `addmm_` onto the x copy that the LayerNorm launch writes while it has the row in registers (`tsx_ln.cu` save 3,
  the generic route's scheme: `torch.addmm(x, ...)` had cost a 1.9 us copy launch).
* Routed (`PEX`, `INFER_CELLS`): bf16 inference D 768 n 2 13-16 tiles, D 384 n 4 13-31 tiles (and ragged rows of those), UH 32, 3 groups.

Stamps ([1, 2048, 768], us from the first expand CTA's entry, medians): LayerNorm done 1.2, first products 2.6, per 256-row block
~2.0-2.4 us of products (3 blocks), epilogue of the last 0.9, end 10.3. Without the xn loads and h stores (probe flags `XP_NOLOAD`,
`XP_NOSTORE`) a block still takes 1.9-2.3 us: the products themselves (M 256 x N 128 x K 768 per pair) pace it at this card's clock.

Fine device time (`fab19.py`, L2 flushed) / graph median (400 flushed replays), us, route before -> after:

| case (H100 / 1.5) | before | after |
|---|---|---|
| row 37 master [1, 2048, 768] (25.0) | 25.9 / 30.7 (generic) | 20.8 / 26.6 (mean 25.9-26.0) |
| row 36 master [1, 2048, 384] n 4 (18.1) | 18.8 / 24.6 (generic) | 15.1 / 20.5 |
| row 36 master [1, 2976, 384] n 4 (23.9) | 25.6 / 30.7 (generic) | 23.2 / 28.7 |
| row 36 master [1, 3968, 384] n 4 (25.9) | 25.6 / 30.7 (generic) | 20.2 / 26.6 |
| [1, 1792, 768] / [1, 2560, 384] n 4 (no requirement row; the routed ranges) | 25.6 / 30.7, 19.2 / 24.6 | 21.6 / 26.6, 17.8 / 22.5 |

Accuracy: `measure.accuracy` rows 35-37 eval master + float32 (`out/acc17_r19a.txt`): 42 GREEN, 15 YELLOW (all float32, routes
unchanged), 0 RED; the changed cases GREEN (row 37 L 2048 0.443 -> 0.475 x tol, row 36 L 2048 0.471 -> 0.498 x tol = the Triton value;
`measure.accuracy` reads L 2976 / 3968 at the L 2048 shape). Random weights against fp64 (`fab19.py`, real shapes, max |err| / max |ref|):
[1, 2048, 768] 4.39e-3, [1, 2048 / 2976 / 3968, 384] n 4 3.97e-3 / 4.03e-3 / 4.03e-3, equal to the generic route's (same cuBLAS squeeze);
outputs within one bf16 ulp (0.0156) of the generic route; bitwise deterministic over repeated calls.

### Built and measured, not kept

| tried | measured | verdict |
|---|---|---|
| the persistent expand with M = 128 pair products (64 rows a CTA, `tsx_expand2.cu`'s layout), UH 32 / 64 | [1, 2048, 768] with the K-split squeeze 29.6 / 33.5 fine (route 25.9); [1, 1024, 768] UH 32 15.5 against 16.6 | replaced by the M = 256 form |
| UH 64 (N 256) at M = 256 | [1, 2048, 768] 29.0-30.5 against 20.8-21.2 (UH 32); D 384 n 4 L 2048 / 3968 17.8-18.0 / 21.2-22.1 against 15.1-16.5 / 20.2-21.8 | D 768: two A stages beside a 192 KB panel; D 384: the SwiGLU epilogue (128 x 128 units a CTA, MUFU-bound) takes ~2 us a block |
| 2 groups instead of 3 (48 pairs) | [1, 1536, 384] n 4 14.4 against 13.4; [1, 1536, 768] 20.1 against 18.2 | slower |
| the chain's squeeze after the persistent expand: K-split `tsx_squeeze` / pair `tsx_squeeze2` NB 64 / 96 (NB 96 built for this) / 128 | [1, 2048, 768] 30.5 / 28.3 / 24.5 / 25.6 against cuBLAS 20.8; D 384 n 4 L 2048 NB 96 15.0 = cuBLAS 15.1 (graph 20.5 both), L 3968 23.4 against 20.2 | cuBLAS kept (`rejected/transition_p19_pair_squeeze_nb96.diff`) |
| an M = 256 pair squeeze (`tsx_squeeze2w.cu`: one pair per (256-row block, NB columns), Ws as 3-D boxes, weights before the wait; NB 64 / 96 / 128, KBX 2-4) | [1, 2048, 768] 21.4-22.8 (NB 96 best) against cuBLAS 20.8; D 384 n 4 L 2048 16.1-17.6 against 15.1; L 3968 20.6-21.2 against 20.2-21.3 | level or slower: 32-64 pairs, each CTA takes 384 KB of h at ~130 GB/s (the per-SM intake); the stamps show its K loop 3.9-4.0 us after a 1.2 us completion hop (`rejected/transition_p19_tsx_squeeze2w.cu`) |
| **the LayerNorm folded into the persistent expand** (`-DFOLD`: raw x as A, W' = bf16(W gamma), a = rs (x W'^T) - rs mean u + v in the epilogue, four statistics warps on the A ring, the x copy for cuBLAS from those warps; no `tsx_ln` launch) | [1, 2048, 768] fine 25.96-27.2 against 20.8; the expand starts 2.4 us earlier but every block takes 3.6-4.4 us instead of 2.2: the statistics warps hold each ring stage (read after the products' commit: +0.3-0.5 us a stage turnaround with one ring stage fewer) and, read in parallel with a forwarded `full` (cluster-scope release arrive from the MMA thread), the MMA thread stalls on the release | rejected (`rejected/transition_p19_folded_persistent_expand.diff`); accuracy was fine (e64 3.2e-3 with the pair squeeze) |
| the SwiGLU sigmoid as f16x2 tanh (`-DESIG=2`) | level with the f32 tanh | not kept |

Not built: (2) of the brief, the LayerNorm fold in `tg_fused_h16` (row 27 / 29 fp32 L 128): the fold measured here shows the cost of
statistics warps that share the operand ring; `tg_fused_h16`'s tile is resident (no ring), so the TriMul pass-14 form (raw x into TMEM by
`tcgen05.cp`, statistics in parallel, K16 correction MMA) would apply there, but it was not reached in this part.

### Misses of 1.5x against the H100 (H100 >= 20 us), before -> after (`out/viol_r18b.txt` -> `out/viol_r19a.txt`)

24 -> **23** misses among the 166 eval cases, none slower than the H100, ladder 0 -> 0 (the one-count drop is row 37 L 768, route untouched,
reading under 0.667 in this run). Moved (us, ratio): row 37 master L 2048 30.6 -> 26.5 (0.816 -> 0.706), row 36 master L 2048 22.6 -> 20.4
(0.829 -> 0.747), L 2976 30.5 -> 26.5 (0.849 -> 0.738), L 3968 30.6 -> 26.5 (0.788 -> 0.682). Still above 0.667: these four need one
(L 2048 / 3968 / D 768) to two ticks more; the remaining time is the cuBLAS squeeze (~7 us) and the products of three 256-row blocks
per pair.

## Round 3, part 20 (2026-11-18): inference -- `tg_fused_h16` (fp32 rows 27 / 29): tanh sigmoid, the last tile's epilogue through the weight ring (`cuda:b200_sm100` 42)

Base: `b200-2.5.x` a87d671b0 (engine 2.5.14 + part 19; `out/viol_r19a.txt` is the state before: 23 misses of 1.5x among 166 eval cases
with an H100 median >= 20 us, none slower than the H100, ladder 0). Probes (B200 box, `b200cfg/`): `hvf.py` (the routed `tg_fused_h16`
build against a build with extra `-D` flags or another source (`HVSRC`): fine device time = graphs of [flush, call] x 20 minus [flush] x
20, the measurer-like graph median, error against an fp64 reference with RANDOM weights / gamma / beta, max |cur - var|, determinism),
`hp.py` (per-CTA `%globaltimer` stamps, new slots for the LayerNorm sub-phases), `run20.sh` (`measure.accuracy` rows 25 / 27 / 29 / 30
eval, then `measure.speed` cuda / triton / pytorch every eval length, one exclusive process each, GPU 1). Timing probes added to the
source (compile-time, off): `TBLN` 1 / 2 (no LayerNorm statistics / no LayerNorm), `TBEPI` 1 / 2 / 3 (no y epilogue / no residual loads
/ no y stores), `TBW` (weight rings never refilled), `HSIG 9` (no sigmoid).

### Bounds first (fine us, [16 384, 256] n 2 = row 29 L 128 one tile per CTA; [65 536, 128] n 2 = row 27 L 256; part 19 build)

| probe | row 29 L 128 | row 27 L 256 | row 27 L 128 |
|---|---|---|---|
| routed | 25.5-25.9 (graph 30.7) | 33.8-34.1 (38.9) | 13.0-13.2 (18.4) |
| no LayerNorm statistics (`TBLN 1`, the LN-fold's best case) | 23.9 (28.7) | 32.3 (36.9) | 12.4 (16.4) |
| no LayerNorm (`TBLN 2`) | 23.0 (28.7) | 30.4 (34.8) | -- |
| no y epilogue (`TBEPI 1`) | 22.3 (26.6) | 26.2 (30.7) | 11.0 (16.4) |
| no residual loads / no y stores (`TBEPI 2 / 3`) | 25.1 / 25.1 | 32.3 / 30.3 | -- |
| no sigmoid (`HSIG 9`) | 25.0 | 31.0 | -- |

Stamps ([16 384, 256], one tile): x landed 4.9 us (cold, L2 flushed: the 16.8 MB of x, read against the flush's dirty lines; contiguous
16 KB boxes instead of the [128 rows][32 columns] boxes land at the same 4.86 us), LayerNorm statistics to 6.4, xn stored 8.45 (two passes
on 8 warps), chunk loop 8.7 -> 19.7 (8 chunks, 1.28-1.38 us each), epilogue to 22.3-26.4.

### What changed (`generic/tg_fused_h16.cu`, inference build only; the training build `SAVE_BUILD` keeps its sigmoid and epilogue)

* **`HSIG 1`: the SwiGLU sigmoid as 0.5 tanh.approx(z / 2) + 0.5** (one MUFU op a unit instead of ex2 + rcp; the bf16 kernels' form).
  Error against fp64 (random weights, real shapes): [16 384, 256] rms 3.652e-4 -> 3.652e-4, max/max|ref| 3.788e-4 -> 3.783e-4;
  [65 536, 128] rms 3.879e-4 -> 3.880e-4 (the fp16 rounding of h dominates).
* **`RXT`: a CTA's LAST tile takes its residual from a TMA copy into the weight ring** (free once the tile's last expand has committed:
  `rxgo`; lanes 16 .. 16 + D / 32 - 1 of the producer warp issue the [128][32] boxes on `rxfull`) and writes y by TMA stores from there
  (each lane adds its row's accumulator in place; `cp.async.bulk.wait_group.read` before the exit): no global residual loads, no slab
  round trip. Same arithmetic: max |diff| 0 against the old epilogue alone (ragged [16 000, 256] too).
* A `__syncwarp()` before the final `__syncthreads()`: the producer / MMA warps leave their roles lane by lane and bar.sync counted a
  diverged warp's partial arrival as the whole warp -- the stamps showed the final barrier passing (and warp 1 freeing the tensor memory)
  before the epilogue warps had finished.

Fine us, base source -> new (`hvf.py HVSRC=` the part 19 source; random weights; error vs fp64 equal to 3 digits in every case):

| rows | [16 384, 256] | [65 536, 256] | [147 456, 256] | [262 144, 256] | [589 824, 256] | [1 048 576, 256] |
|---|---|---|---|---|---|---|
| D 256 n 2 (row 29) | 25.5 -> 23.0 | 81.9 -> 76.8 | 161.9 -> 153.0 | 284.6 -> 261.3 | 665.2 -> 621.1 | 1187.6 -> 1137.3 |
| D 128 n 2 (row 27) | 13.2 -> 12.2 | 34.2 -> 32.0 | 65.8 -> 60.2 | 110.0 -> 102.6 | 240.6 -> 224.8 | 423.6 -> 399.1 |

D 64 n 2 [65 536] 19.2 -> 18.0, n 4 [16 384] 9.9 -> 9.4 (no fp32 requirement row).

Accuracy (`measure.accuracy` rows 25 / 27 / 29 / 30 eval, random weights, fp64 gold, tol = max(err_pytorch, fixed); `out/acc10_r20a.txt`):
53 GREEN / 18 YELLOW / 0 RED, as part 11; row 27 float32 YELLOW 1.0001 / 1.0006 x tol (unchanged), row 29 float32 GREEN 0.9990 (L 1536
0.9998 -> 0.9939).

### Built and measured, not kept (diffs `rejected/transition_p20_tg_fused_h16_experiments.diff`, `rejected/transition_p20_ln12_first_tile_12_warps.diff`)

| tried | measured (fine us) | verdict |
|---|---|---|
| `EPW`: the y epilogue reading 64 accumulator columns per `tcgen05.wait::ld`, yempty released once in registers | [65 536, 128] 33.8 -> 34.9, [16 384, 256] 25.9 -> 27.1 | slower |
| `SWW` / `ABR`: the SwiGLU's a / b loaded before one wait / [a \| b] released before the math | level (22.97 against 23.33; [65 536, 128] 32.4) | the stamps: the next expand waits for the in-order MMA thread / SwiGLU period, not the release |
| `RYI`: the one-tile CTA copies x into the y accumulator by `tcgen05.cp` as it lands (squeezes accumulate onto it) and stores y from TMEM in the 16x256b order | 23.3 -> 24.7 | the copies read XS0 beside the LayerNorm passes (statistics 6.4 -> 6.9 us), and the epilogue stayed 2.5 us: it is the y write itself (16.8 MB at ~6.7 TB/s), not the residual |
| `XEARLY`: XS0's x issued right after the barrier init | x landed 4.86 -> 4.61, fine level | -- |
| f16x2 tanh (`HSIG 2`) | level; rms 3.652e-4 -> 3.990e-4 | worse accuracy for nothing |
| `TBW` (weights never refilled) as a bound | 22.97 -> 22.88 | the weight stream is not the chunk loop's bound; it is the single [a \| b] buffer (TMEM: y 256 + [a \| b] 128 + xn 128 columns) |
| `LN12`: the first tile's LayerNorm on 12 warps (both SwiGLU halves + the LN warps split the columns, partial (s, q) through the idle h buffers, f32x2 arithmetic) | [16 384, 256] 22.94 -> 22.77 / 23.35 -> 23.13 | the SwiGLU-side warps finish 0.8 us after the LN warps (spills: 128 B stack); level |

### The brief's items

1. **LN_in fold in `tg_fused_h16`** (row 29 fp32 L 128): not built. Its best case is the one-pass LayerNorm, measured as `TBLN 1` on top
   of the kept changes: fine 21.45-22.21 us = graph 26.59 (the 26.6 grid point needed, 0.0-0.7 us of margin), before any correction cost.
   The correction does not fit: TMEM is full at D 256 (y 256 + [a \| b] 128 + raw x 128 columns: no 8 columns for TriMul's K16 correction
   rows in the A operand), and shared memory has 2.8 KB free (224 KB of rings + the 2 KB gamma / beta) against 4 KB of correction rows per
   chunk (8 KB double-buffered with the weight stage); the per-unit u / v in the SwiGLU epilogue (2 FFMA2 + vector loads a pair) is what
   TriMul measured slower, and the SwiGLU is half of this kernel's chunk period. fp32 x also has no weight-derived fp16 range bound (the
   current xn has one): the fold would need a per-tile overflow fallback.
2. **Squeeze fused into the persistent expand** (rows 35-37 large L): not built -- the byte count rules it out. Row 37 L 2048 (D 768 n 2,
   H 1536): h is 2048 x 1536 x 2 B = 6.3 MB each way (the brief's 25 MB is 4 D per row; n is 2 here). Fused, each of the 24 hidden-slice
   pairs makes a y partial [256 x 768] fp32 per 256-row block: 24 x 786 KB = 18.9 MB a block, 151 MB for 8 blocks to reduce (24x the h
   bytes); a 16-CTA cluster (8 pairs) leaves 3 cluster partials a block through L2 (19 MB). Capacity: a CTA's y partial is 768 TMEM
   columns (256 left beside the double-buffered [a \| b]), its Ws slice 48 KB beside a 192 KB resident panel. Row 36 L 3968 (D 384 n 4):
   h 12.2 MB against 146 MB of partials.
3. **Rows 25 L 256 / 37 L 512 phase traces** (`tl.py`, one flushed replay; `pf.py` stamps): row 25 [65 536, 64] n 2: one kernel of 11.65 us
   starting 2.6 us into the replay (graph 16.4); x of the first tile lands at 2.05 us, a tile every ~1.2 us after the first is done at 5.6,
   68 of 148 CTAs take a fourth tile (512 tiles = 3.46 a CTA; the last round ends at 9.5-10.2). The 12.3 grid point needs the kernel at
   ~7.6 us: even a perfectly balanced last round (5.6 + 2.46 x 1.2 = 8.6) is one tick short. Row 37 [1, 512, 768]: tsx_ln 2.46 + 2.78,
   the pair expand 3.58 + 8.16 (ends 11.74), the pair squeeze 5.82 + 10.78 (ends 16.6, 4.9 us after the expand); fine 13.24 us, graph 18.5;
   the 14.3 grid point needs ~3.5 us less (part 16's stamps at L 384: the expand stored 1.5 us after its first stage, the squeeze's K
   loop ~1.3 us after its wait, the rest launch / LayerNorm / wait hops): not reachable by a phase tweak.

### Misses of 1.5x against the H100 (H100 >= 20 us), before -> after (`out/viol_r19a.txt` -> `out/viol_r20a.txt`)

23 -> **22** misses among 166 eval cases, none slower than the H100, ladder 0 -> 0. Cleared: row 27 float32 L 256 (38.9 -> 36.8 us,
0.673 -> 0.637). Moved: row 27 float32 L 128 18.3 -> 16.2 (0.887 -> 0.786), row 29 float32 L 128 30.6 -> 28.5 (0.722 -> 0.674, the
target is 28.27: one tick). Every other row 27 / 29 float32 length 3-11 % faster (row 29 L 256 88.0 -> 81.9, L 2048 4760 -> 4639; row
27 L 512 116.6 -> 108.5, L 1536 915.5 -> 864.2).

## Gap analysis (B200 `cuda`)

**Transition** (307 requirement cases): every case routes. Not fully closed:

| cases | state | why | what it would take |
|---|---|---|---|
| row 31 L 2976 / 3968, row 32 L 2048 (master eval) | accuracy `skipped` | the fp64 gold / reference (24-90 GB) did not fit beside the other jobs on GPU 1 | re-measure on an idle card (speed measured: served) |
| fp32 rows 27 / 29 / 33 / 35 / 37 / 211 / 213 / 217 / 219 (43 cases) | YELLOW | TF32 rounding; Triton is YELLOW on the same cases with the same order of error | an fp32 path without TF32 (3xTF32 split products) if the rule should be GREEN |
| row 209 master train (D 64 n 2) | closed 2026-11-17 (round 3): the wide build with `tbwd_d64.cu -DDXN32`, 6 / 6 GREEN, 28.8-292.8 us (0.55-0.58x the H100 time) | the bf16 d_xn of the wide backward had been YELLOW; round 2 had moved the row to the generic kernels | -- |
| row 34 master train (D 512 n 4, L 768) | YELLOW (1.18x) | same as Triton | as above (generic GREEN, 1.4x slower) |
| row 27 fp32 eval (D 128 n 2, 9 cases) + row 25 master L 128 | closed 2026-11-17 (round 3): `tg_fused_h16`, 1.00-1.40x the Anthropic kit (L 128: 20.4 us, a tie); row 25 L 128 a tie (12.2 us); round 3 part 20: L 128 16.2 us, L 256 36.8 (1.5x the H100) | was the generic TF32 forward against the kit's fused fp32 cell (0.71-0.90x) | a faster first tile (the L 128 ties) |
| fp32 training at every width | generic backward; the forward on the fp16 one-launch `tg_fused_h16` at D 64 / 128 / 256 from 65 536 rows (part 7), TF32 generic / small-M chains elsewhere | the tuned builds are bf16 | a TF32 fused D 128 backward (H100 `f32dx` / `f32dw`, not built: round 3 part 7) |
| row 37 fp32 eval [1, L, 768] n 2, L 128-384 | closed 2026-11-17 (round 3): `cuda` 30.6-32.7 us against the Triton token 34.7 (the ladder holds on all 307 cases) | weight forms re-made inside the captured graph; the addmm copy launch | 1.5x against the H100 (30.2-33.4 us there): see round 3 |
| Triton row 37 fp32 eval L 1536 | 1.07x PyTorch (the last Triton < PyTorch miss) | M-bucketed launch table (M 1024 bucket serves 1024-2047 rows) | a separate L 1536 key or a split-K expand on sm_100 |

**LayerNorm** (656 requirement cases): after 2026-11-17 the master half is 281 served GREEN / 53 refused (was 277 / 57); fp32 unchanged (D 64 fp32 training from 147 456 rows refused). 79 served cases lose to the pretuned Triton (exception class; master 27 -> 24).

| cases refused | why | what it would take |
|---|---|---|
| bf16 pair inference, D 64 from 65 536 rows, D 128 from 147 456, D 256 from 65 536, D 384 from 147 456 (rows 1-5, 7, 8) | the sm_100a row kernels closed half the distance (D 128 L 2048: 434 -> 350 us) but Triton runs these at 6.6 TB/s (324 us); every launch / hint variant tried is listed in the optimization round | the TMA bulk-copy forward was prototyped in round 3 and is slower (384 us at D 128 L 2048, 283 at D 384 L 1024 against Triton 240); open: a forward that beats 6.6 TB/s (a second in-cache pass like Triton's, or fewer instructions per element than the register form), then lift `SLOWER` |
| D 64 fp32 training from 147 456 rows (row 189) | bf16 closed 2026-11-17 (sm_100a rows); fp32 still on the general kernels | an fp32 entry in `SM100_ROWS` after measuring the (4, 4) layout |

Against the H100 (round 3, 599 cases both cards serve): 375 are not 1.5x faster, 215 are slower. 369 of the 375 are at or below 33 us,
where the replay floor of this card (6.02 us empty, 2.048 us event ticks) meets the H100's 6.2-6.8 us per launch; what it would take is
fewer launches than the H100 per call (a training step in two launches), which no LayerNorm kernel change inside one launch gives. The
6 others (33-100 us) are listed in the round-3 section.

Served but slower than PyTorch (Triton is 2-6x slower still): fp32 training of the single stream at D 768 (rows 202 / 203) and
D 831 / 833 at L 128 / 256 (rows 204-206): 2-4 us behind at a few hundred rows (three launches against one native kernel). The A100
row kernels (`cuda/sm80/norm_rows.cu`) built for sm_100a did not launch (`norm_fwd: launch failed`) and are not used.

**RMSNorm** (no requirement row): routed on the B200 to `pipe` (16-byte rows of 144-2048 bytes) or the general kernels' RMS mode; it
serves `InputFeatureEmbedder`'s legacy-feature `atom_out_norm`.

**InputFeatureEmbedder**: 126 of 207 cases unsupported, all by `SWAAtomTransformer` (the atom-DiT port); see above.

## S90 round (2026-11-18, `s90-transition-inf`): Transition INFERENCE, SOL% against measured floors (no kernel change kept; `cuda:b200_sm100` stays 42)

Base: `b200-2.5.x` 35fa126c1 (engine 2.5.14 + parts 1-20). Scope: every eval case of rows 25-40 (`master` + `float32`, 181 cases; rows 209-221 have no eval
case). Files: `docs/b200/s90_transition_inf/` (`sol_inf.py` the calculator = `sol_transition.py`'s inference formulas + three extra columns, `peaks.json`,
`sol_base0.tsv` the per-case table, `s90m.sh` / `s90t.py` / `ceil.py` the probes). Records: B200 box `$W/b2511/agents/s90-transition-inf/out/m_base0_eval_<L>.jsonl`
(`measure.speed`, cuda, graph, L2 flushed, one exclusive process per length, GPU 2). The base tree measures 25 misses of 1.5x among the 166 non-exempt eval
cases (part 20 read 22: the difference is single 2.048 us ticks of unchanged routes), none slower than the H100, ladder 0 -- unchanged, nothing was routed.

### SOL% per case, before (= after: no change kept)

SOL% = t_sol / t_measured, t_sol the study's floor (bf16 1570 TF/s, fp16 1533, TF32 836, HBM 6565 GB/s, MUFU 16/clk/SM). Three further readings:
"+floor" adds the MEASURED graph-replay floor below (6.1 us); "sustained" uses the sustained cuBLAS peaks measured below (bf16 1365, fp16 1290 TF/s);
"fp16 kind::f16" (float32 only) is the fp16 tensor peak, which is what the float32 inference paths already run (fp16 operands at TF32 rounding under
`_h16_forms` scales, rows 27 / 29 `tg_fused_h16`, rows 33 / 35 / 37 the fp16 chains): the TF32 column of the study is not the engine's binding peak (row 33 float32 reads 101-105 % of it).

| dtype | cases | SOL% min / median / max | >= 90 / 70 / 50 % | +6.1 us floor: min / median / max | >= 90 / 70 / 50 | vs sustained peak: median / max | >= 90 / 70 / 50 |
|---|---|---|---|---|---|---|---|
| master | 139 | 4.0 / 52.6 / 93.1 | 3 / 52 / 73 | 43.0 / 62.4 / 94.1 | 3 / 56 / 119 | 60.5 / 107.1 | 32 / 65 / 81 |
| float32 (TF32 peak) | 42 | 4.7 / 53.8 / 105.5 | 10 / 14 / 23 | 47.8 / 63.3 / 106.6 | 11 / 16 / 40 | -- | -- |
| float32 (fp16 kind::f16 peak) | 42 | 4.7 / 32.4 / 57.5 | 0 / 0 / 9 | -- | -- | 35.2 / 68.3 | 0 / 0 / 12 |
| all | 181 | 4.0 / 53.3 / 105.5 | 13 / 66 / 96 | 43.0 / 63.2 / 106.6 | 14 / 72 / 159 | 53.3 / 107.1 | 32 / 65 / 93 |

Per row (SOL% pure, min - max over lengths; master unless noted): 25 5.4 - 49.7, 26 8.5 - 51.2, 27 12.7 - 53.3 (float32 23.8 - 64.3; fp16 16.1 - 42.6), 28 22.5 - 85.6,
29 33.5 - 69.0 (float32 54.2 - 96.2; fp16 29.5 - 52.4), 30 47.4 - 83.0, 31 45.6 - 78.6, 32 64.6 - 93.1, 33 53.5 - 75.2 (float32 75.5 - 105.5; fp16 41.2 - 57.5),
34 71.4 - 85.5, 35 4.0 - 12.7 (float32 4.7 - 23.6), 36 6.9 - 33.8, 37 13.1 - 35.0 (float32 14.8 - 53.4; fp16 13.0 - 29.1), 38 16.9 - 45.4, 39 64.4 - 86.6, 40 75.5 - 83.6.
Lowest 15 (all rows 35 / 36 and row 25 L 128): 4.0 - 7.7 % (+floor 45-56 %). The 13 cases >= 90 % are rows 29 / 33 float32 (TF32 peak; fp16 peak 41-57 %) and
row 32 master L 384-640 (91.2-93.1 %). Counts >= 90 / 70 / 50 % (pure, all 181): 13 / 66 / 96.

### Measured floors (B200 box, GPU 2 exclusive)

1. **Graph-replay floor** (`floor.py`, measure.speed's flush + graph timing, k tiny kernels): 1 / 2 / 3 / 4 / 6 / 8 kernels 6.1 / 8.2 / 8.2 / 10.2 / 12.3 / 14.3 us
   (2.048 us event ticks). Every inference case pays >= 6.1 us for ONE kernel: rows 35 / 36 / 37 (t_sol 0.6-9 us), row 25 / 26 / 38 L 128 (t_sol 0.7-3 us) can reach at most
   t_sol / (t_sol + 6.1) = 9-13 % at L 128 even with a perfect one-launch kernel. With that floor added they sit at 45-59 % today (the 3-launch PDL chain 14.1 us against the
   3-trivial-kernel floor 8.2 us: ~6 us of LayerNorm launch + expand + squeeze + two completion hops, each measured in parts 14-19; the one-launch / fused alternatives
   measured slower: parts 6, 14, 19). 90 % of ANY of these cases is therefore unreachable by arithmetic, not only by design.
2. **Sustained tensor peaks** (`ceil.py`, graph replays after 1 s of the same load, run length 0.25 - 16 ms: no decay with length): cuBLAS square 8192 GEMM bf16 1365-1385,
   fp16 1281-1353 (1290 at 16 ms), TF32 716-753 TFLOP/s -- 87 / 84 / 86 % of the "best of 5" peaks of the study, which are cold-card bursts. K = 256 expand shapes
   ([2^20 / 2^22, 256] x [256, 2048]) reach only 830 (bf16) / 800 (fp16) / 449 (TF32) TFLOP/s in cuBLAS (output write bound), against the engine's row 30 L 2048 at 1175 TFLOP/s. The
   large-L plateaus of rows 28-31 (62-83 % pure) read 70-107 % of the sustained peak; the L >= 1536 step down is the card's power state, which cuBLAS shows too.
3. **Memory floor of the D 64 n 2 forward** (`tfwd_d64.cu` with the whole compute removed: `TBLN=1 TBSW=2 TBEPI=1`, [1 048 576, 64], back to back): 1.04 us a tile with the 4 x stages of
   the kernel, **0.82 us a tile with 8 stages** (45.5 us for 268 MB = 5.9 TB/s, the HBM copy peak 6.5): the x TMA stage lifetime paces the empty pipeline. The routed kernel (`ROLES=1`)
   runs 1.59 us a tile (88.0 us); its stage stamps (`pf.py`, [1 048 576, 64]): one SwiGLU chunk every 0.77 us (0.51 us of work), LayerNorm done every 1.0-1.8 us, epilogue 1.0 us long.

### Built and measured, not kept (diff `rejected/transition_s90_d64_ob2_nxs8_tbprobes.diff`; D 64 n 2 inference forward, [1 048 576, 64], back to back, us a tile)

| tried | routed (ROLES=1) 1.59-1.61 | verdict |
|---|---|---|
| ablations: no LayerNorm statistics / no sigmoid / no SwiGLU TMEM loads + math / no epilogue loads + adds / all three | 1.415 / 1.498 / 1.449 / 1.497 / 1.038 | the stages are balanced (single removals save 0.09-0.18, all three 0.55): no one stage is the bottleneck, so no single-stage rewrite pays more than ~10 % |
| `OB2`: the squeeze accumulator double-buffered in tensor memory (2 x 64 columns fit: [a \| b] 256 + h 128 + 2 x 64 = 512), epilogue frees its buffer right after the TMEM loads (breaks the out_full -> epilogue -> out_empty -> squeeze recurrence) | 1.604 | level: the recurrence is not the cycle |
| `NXS_=8` x stages (4 -> 8, 214 KB) | 1.604 (empty pipeline 1.04 -> 0.82) | level: the real kernel is paced by its compute chain, not by the stage lifetime |
| `NXS_=2` | empty pipeline 1.61 | confirms the stage count paces only the empty pipeline |

What it would take beyond this: more issue slots for the three compute roles at once (the CTA has 16 warps with all registers in use: SwiGLU 152 regs a thread), i.e. a
register-lean SwiGLU / LayerNorm on f32x2 arithmetic with 20-24 warps; its ceiling at the measured 0.82 us a tile is 88 -> ~58 us at [1 048 576, 64] (row 25 L 1024: 93.8 -> ~65 us),
not built in this round (single-stage changes measured 0.09-0.18 us each; the ring needs all three).

### Per-case reasons for the cases that stay below 90 % (measured, not inferred)

* **Rows 35 / 36 / 37 and row 25 / 26 / 38 L 128 (launch / replay floor)**: replay floor 6.1 us (1 kernel), 8.2 us (3 trivial kernels); chain stamps and phase costs in parts 14-19
  (LayerNorm launch to last row 2.9 us, pair expand 1.7, completion hop 1.2, pair squeeze 2.3-3.0 at [1, 384, 768]).
* **Rows 25 / 26 / 38 (D 64) from L 384 and row 27 / 28 (D 128) at L 128-384 (45-53 %)**: the persistent forward's tile period 1.59 us against the 0.82 us memory-only period (above).
* **Rows 29-31, 33, 34 L >= 1536 (62-75 %)**: power-state / sustained peak (cuBLAS 1365 TF/s square, 830 at K = 256); against the sustained peak they are 70-107 %.
* **Float32 rows** (27 / 29 / 33 / 35 / 37): the fp16 tensor peak is 1.83x the TF32 peak the study scored them against; at the fp16 peak they are 4.7-57.5 % (median 32 %): the fused
  `tg_fused_h16` (D 64-256) and the 3-launch fp16 chains (D 384-768) are first-tile / launch-latency bound at small L (parts 10, 17, 20) and 41-57 % of the fp16 peak at large L, where the
  x read + y write in fp32 (row 29 L 2048: 8.6 GB, 1.31 ms at the HBM peak = 61 % of the 2.15 ms fp16 tensor time) leave little room to overlap, calculated from the formulas, not separately measured.

### Summary

No source or route changed; the SOL table is the same before and after. Nothing to prebuild, no version bump.

### S90 follow-up: the 20-warp D 64 n 2 forward built (`SW2`, rejected; `cuda:b200_sm100` stays 42)

Built as asked: `tfwd_d64.cu -DSW2=1` (with `ROLES=1`): 640 threads, the SwiGLU of every chunk split by units over two warpgroups (12-15 units 0-31, 16-19 units 32-63; `ab_empty` /
`h_full` count 16), 104 registers in the 16 compute warps, 40 in the producer / MMA warps. First try deadlocked (`setmaxnreg_inc<112>` needs 512 x (112 - 96) = 8192 registers released by the four
producer warps, which free only 5120 at 56): the 640-thread launch is allocated 96 a thread, so the pool allows 104 with the producers at 40. Correct afterwards (error vs fp64 max/max|ref| 4.0e-3,
rms 2.6e-3, random weights, [1 048 576, 64] n 2, one shape; the routed kernel 3.8e-3 / 2.6e-3 on its own random draw). Timing ([1 048 576, 64], back to back, us a tile):

| variant | us a tile |
|---|---|
| routed `ROLES=1` | 1.596 |
| `SW2` (SwiGLU twice as wide) | 1.599 |
| `SW2` + no LayerNorm statistics | 1.360 |
| `SW2` + no epilogue loads / adds | 1.394 |
| `SW2` + neither | 1.209 |

So halving the SwiGLU alone gains nothing: LayerNorm (-0.24) and the epilogue (-0.20) are what paces the ring once it is split. Splitting those two as well needs 28 warps (4 + 8 + 8 + 8); at 65 536
registers that leaves ~74 a thread, below what the SwiGLU half needs (spills) and 4 warps short even at 24 warps (88 regs: the pool the producers release is 7168). Not built. Even with both free the ring
stops at 1.21 us a tile (88 -> ~67 us at [1 048 576, 64], row 25 L 1024 93.8 -> ~70 us), i.e. a ceiling of -24 %, only reachable with a 28-warp layout this register file does not hold. Diff with all
probe flags (`TBLN`, `TBSW`, `TBEPI`, `OB2`, `NXS_`, `SW2`): `rejected/transition_s90_d64_ob2_nxs8_tbprobes.diff`. D 128 n 2 (`tr_fwd_sm100.cu`, ~4.5 us a tile) was not touched: its ring has the
same three roles and a longer tile, so the same register argument applies.

Float32 at large L, x read + y write (brief item c): the residual is added to fp32 x, so y = x + f(x) needs x in fp32 (4 B read) and the fp32 y (4 B write); a bf16 / fp16-staged x would round the
residual (fp16: 2^-11 relative, 5e-4, against the float32 tolerance 1e-5 of the fixed table), so it is not exact and not allowed; the in-kernel front cast already exists (xn is made fp16 in the kernel).
No byte saving without an accuracy loss; not built. 1.5x / ladder counts: no change to any route, so 25 / 0 as measured above.

## S90 round (2026-11-18, s90-transition-train): Transition TRAINING, SOL toward 90 % (`cuda:b200_sm100` 43)

Training only (`modules.Transition`, master and float32, 126 token cases). Base `b200-2.5.x` 35fa126c1. SOL = the study's calculator
(`docs/b200/sol_transition`, `min(recompute, save a,b)`, bf16 1570 / TF32 836 TFLOP/s, HBM 6565 GB/s), re-scored here with `b200cfg/solscore.py`
on fresh `measure.speed` runs (cuda, graph, L2 flushed, one exclusive process per length, GPU 1): `out/m_m0_*` before, `out/m_m1_*` after.
Probes (untracked, `b200cfg/`): `fx.py` / `fxj.sh` (module step, graph, L2 flushed, in-process A/B over module attributes), `gsave.py` (bits against the
old policy + determinism, `S90_OLD=1` restores the base policy), `acc.sh` (measure.accuracy), `prof1.sh` (`tb.py -p`), `solscore.py`, `floor.py`.

### SOL before -> after (all 126 training cases)

| dtype train | cases | SOL% min / median / max before | after | >= 90 | >= 70 | >= 50 before -> after |
|---|---|---|---|---|---|---|
| master | 96 | 4.4 / 40.6 / 66.3 | 4.3 / 41.8 / 69.1 | 0 -> 0 | 0 -> 0 | 27 -> 26 (run-to-run, see below) |
| float32 | 30 | 3.3 / 23.4 / 40.6 | 3.3 / 24.5 / 55.9 | 0 -> 0 | 0 -> 0 | 0 -> 5 |

No case reaches 90 % or 70 %. Per row (SOL% min - max over L 128..768), before -> after:

| row | dtype | D n stream | before | after |
|---|---|---|---|---|
| 033 | float32 | 512 n2 pair | 34.5 - 40.6 | 45.9 - 55.9 |
| 033 / 034 | master | 512 n2 / n4 pair | 47.6 - 63.7 / 54.0 - 66.3 | 47.6 - 63.8 / 54.0 - 69.1 |
| 040 / 221 | master | D256 n4 msa / D128 n4 msa | 45.4 - 51.0 / 46.5 - 50.5 | 48.5 - 51.2 / 46.5 - 51.8 |
| 209 / 210 | master | D64 n2 / n4 pair | 6.6 - 23.3 / 11.5 - 34.1 | 6.6 - 23.3 / 11.5 - 35.0 |
| 211 | float32 / master | D128 n2 pair | 14.4 - 25.6 / 14.7 - 40.1 | 14.7 - 26.9 / 14.8 - 43.9 |
| 212 | master | D128 n4 pair | 23.9 - 50.4 | 24.3 - 51.9 |
| 213 | float32 / master | D256 n2 pair | 23.5 - 34.5 / 25.1 - 43.3 | 24.6 - 37.2 / 25.1 - 43.1 |
| 214 / 215 / 216 | master | D256 n4 / D384 n2 / D384 n4 pair | 33.4 - 51.2 / 35.6 - 54.6 / 46.4 - 63.1 | 33.4 - 47.7 / 35.6 - 54.3 / 46.3 - 63.5 |
| 217 / 218 / 219 | float32 + master | single stream D384 / D768 | 3.3 - 23.3 | 3.3 - 23.2 (unchanged) |
| 220 | master | D64 n4 msa | 20.5 - 31.8 | 20.5 - 32.4 |

Rows 033 master / 034 / 214 / 215 / 216 / 040 were not touched; their before / after differences (-7 % ... +11 % on single lengths, e.g. 214 L512 1540 -> 1654 us,
034 L384 2851 -> 2564 us) are the run-to-run spread of the shared, power-capped card (other agents' exclusive jobs run on the same GPU between measurer
processes) and bound what a single measurement can resolve at the large lengths.

15 lowest after (all single stream or the narrow L128 chains): rows 217 float32 L128 / L256 / L384 (3.3 / 5.0 / 6.3 %), 217 master L128 / L256 / L384 / L512 (4.3 / 4.6 / 6.0 / 7.0 %),
209 master L128 (6.6 %), 218 master L128 / L256 (6.8 / 7.0 %), 217 master L640 / L768, 217 float32 L512 / L640, 219 float32 L128 (9.4 %).

Requirement cases (brief 6 conditions, H100 >= 20 us): 1.5x misses 10 -> 8 (none slower than the H100; rows 212 L128 and 218 L384, both one timer tick from the
line, now on the right side; row 211 float32 L128 0.707 -> 0.693), ladder violations 0 -> 0 (cuda, triton, pytorch measured on all 126 cases, `out/viol_m0.txt`, `out/viol_m1.txt`).

### What changed (`cuda:b200_sm100` 43; training only)

1. **fp32: keep a, b (GATE_AB) from D 256 up, d_xn on cuBLAS at D 128 / 512** (`fused_generic_sm100a._AB_F32_MIN_D`, `_DXN_CUBLAS_F32`). Module step, graph, L2 flushed, us:

   | case (fp32, n 2) | old | new |
   |---|---|---|
   | D 512, 16 384 / 65 536 / 147 456 / 262 144 / 409 600 / 589 824 rows | 533 / 1800 / 3923 / 7078 / 11 015 / 15 583 | 402 / 1290 / 2749 / 4848 / 7618 / 11 044 |
   | D 256, 16 384 / 65 536 / 147 456 / 409 600 | 196.6 / 612.4 / 1243.2 / 3297 | 188.5 / 563.2 / 1157.3 / 3087 |
   | D 128, 16 384 / 147 456 / 262 144 / 409 600 / 589 824 | 98.3 / 546.8 / 917.2 / 1398.9 / 1994.6 | 96.3 / 522.9 / 876.5 / 1337.3 / 1902.6 |

   Why: the TF32 recompute GATE tile (three accumulators, fp32 a, b recomputed) ran at ~30 % of the cuBLAS TF32 rate (4 `tg_gemm` launches 5476 us against
   cutlass dW GEMMs at 94 % of the 836 TFLOP/s peak, `tb.py -p` [262 144, 512] n 2), so saving a, b (2 x M x H x 4 B written and read) wins from D 256; at D 128
   the fp16 fused forward + recompute stays ahead (keep a, b + TF32 forward: 276 -> 303 us at 65 536 rows), so D 128 only takes the cuBLAS d_xn.
   At D 256 the fp16 forward (`_train_h16`) gives way to the TF32 forward that keeps a, b. Accuracy (basis: `measure.accuracy`, random weights through the
   measurer's own inputs, vs the engine's fp64 / PyTorch reference, rows 033 / 211 / 213 float32 train, every length, `out/acc_s90a.log`): YELLOW 1.00-1.06 x tol,
   one GREEN, no RED (unchanged from the base's 1.00-1.07). Bits (`gsave.py`, [16 384 / 65 536, 128 / 256 / 512] n 2, two steps each): deterministic everywhere;
   output and every gradient bit-identical to the old policy except D 256 at 65 536 rows (the forward moved from fp16 to TF32 operands: max |d| 1.7e-3 y, 4.5e-2 dW on values of O(10)).
2. **bf16 two-role backward: weight-role replicas by tile count** (`fused_wide_sm100a.REPL_TILES`, `fused_sm100a.DW_REPL_LARGE`): the counts (D 128 n 2 14, D 128 n 4 9,
   D 64 n 4 20) were tuned at 128 tiles; the role balance moves with the tile count. D 128 n 2: 15 from 129 tiles, 16 from 2048; D 128 n 4: 10 from 3200 tiles; D 64 n 4: 19.
   Step (us): D 128 n 2 at 65 536 / 147 456 / 262 144 / 409 600 / 589 824 rows 98.1 / 186.4 / 309.2 / 475.0 / 673.8 -> 96.1 / 182.3 / 292.9 / 436.2 / 616.2; D 128 n 4 at
   409 600 / 589 824 654.0 / 942.2 -> 647.2 / 913.4 (262 144 rows: 10 is 430.1 against 426.0, so not below 3200 tiles); D 64 n 4 at 65 536 / 147 456 / 262 144 / 589 824
   65.5 / 114.7 / 194.5 / 395.3 -> 63.5 / 112.6 / 190.5 / 385.1. 128 tiles (L 128) unchanged. Bits: y and dx bit-identical, the fp32 weight gradients regrouped
   (max |d| 8.5e-4 / 2.5e-2 on [409 600, 128] n 4); deterministic over two steps on 7 shapes. Accuracy (`measure.accuracy`, rows 209-212 / 220 / 221 master train, every length): 36 / 36 GREEN (0.61-0.99 x tol).

### Recompute (22 MDH) against save a, b (18 MDH + 4 M H b bytes), measured in step time

| width / path | schedule in the base | the other schedule, measured | verdict |
|---|---|---|---|
| fp32 D 512 n 2 (generic TF32) | recompute | keep a, b: -22 to -25 % at every row count (7078 -> 5384 us at 262 144 rows) | **keep a, b** (done) |
| fp32 D 256 n 2 | fp16 forward + recompute | TF32 forward keeping a, b: -4 to -7 % (1243 -> 1157 us at 147 456) | **keep a, b** (done) |
| fp32 D 128 n 2 | fp16 forward + recompute | keep a, b + TF32 forward: +10 to +14 % (276 -> 303 us at 65 536) | recompute stays |
| bf16 D 256 n 2 / n 4 (`tgate_w`) | recompute | ln + swiglu_ab + squeeze + `tgate_ab` (all built for DIM 256): n 4 16 384 / 65 536 / 147 456 / 589 824 rows 145.4 / 473.1 / 905.2 / 3545.6 -> 151.6 / 474.9 / 927.7 / 3587.1; n 2 96.3 / 563.2 -> 106.5 / 571.4 | recompute stays (4-10 % faster); `rejected/transition_s90_d256_save_ab.diff` |
| bf16 D 512 n 2 / n 4 (`tgate_ab`) | save a, b | recompute (`tgate_w` DIM 512, forward without the a, b stores): n 2 204.8 / 669.7 / 2544.7 -> 229.4 / 776.2 / 2914.3; n 4 364.5 / 2560.4 / 4612.6 -> 411.7 / 3041.8 / 5399.1 (16 384 / 147 456 / 262 144 rows) | save a, b stays (12-17 % faster); `rejected/transition_s90_d512_recompute.diff` |
| bf16 D 384 | save a, b (fused forward writes them) | not built (the fused D 384 forward has no no-save training variant) | unchanged |
| bf16 D 64 / D 128 two-role backward | recompute inside the one kernel (DX role recomputes the gate) | the part 4 DXG build ([dA | dB] through global memory) was 49.15 vs 51.2 us at n 2 and 61.4 vs 57.3 at n 4 | unchanged (docs above) |

So the SOL calculator's `min(A, B)` is not the schedule the hardware prefers: on the measured card the choice follows the GEMM kernels' efficiency, not the
FLOP / byte count (D 256 bf16 saves 4 MDH but loses 4-10 %; fp32 D 512 recompute loses 25 %).

### Rejected in this round

* `tg_round_tf32` / `tg_split_tf32` with four 16-B loads in flight per thread (they showed 622 us for 537 MB in the profile): step level at four shapes
  (D 512 [262 144] 4848 -> 5005 us, [65 536] 1290 -> 1288, D 256 [147 456] 1157 -> 1159, [16 384] 401.5 -> 401.4): the profile duration is a programmatic dependent
  launch waiting for the forward's last kernel, not the kernel's own time (`rejected/transition_s90_round_tf32_unroll4.diff`).
* D 64 n 2 replicas 26 / 28 / 32 / 34 / 38 (30 is within 1 % at every row count), D 64 n 4 14-26 outside 19-20, D 128 n 2 10-12 / 17-18, D 128 n 4 7-8 / 11-12.

### Why the cases below 90 % stay there (measured, not inferred)

* **Single stream, rows 217 / 218 / 219 (4-23 %)**: the replay floor measured here (`b200cfg/floor.py`, graph of k tiny dependent kernels, L2 flushed, events like
  the measurer): 1 kernel 6.14 us, 2 8.19, 3 10.24, 4 12.29, 6 14.40, 8 18.43, 12 23.36 us (timer tick 2.048 us): ~6 us for the first launch and ~2 us per further
  dependent kernel. Row 217 master is 26.5 us = the floor of ~8-10 launches plus 1.2 us of tensor work; the step is already 4 launches with grid barriers
  (parts 2-4: 3-5 us per fixed phase, ~1-2 us per grid barrier). t_sol is 1.2-10 us, so these rows can reach at best ~25-40 % even at the launch floor (the +5 us column of the study).
* **D 64 pair rows 209 / 210 (7-35 %)**: [589 824, 64] n 2 is 290.8 us for 67.7 us of tensor time and 57 us of HBM time: forward 68 us for 18 us of tensor work (26 %),
  backward 215 us for 49 us (23 %) (`tb.py -p`); K = 64 products and one SwiGLU per hidden element dominate; the replica sweep above moves it < 1 %.
* **Wide master rows (033 / 034 / 214-216 at 47-69 %)**: the fwd / bwd tile kernels sit at ~51 % of the cuBLAS-measured bf16 peak ([262 144, 384] n 4: fwd 1166 us for 591 us of
  tensor time, bwd 1227 us for 591) while the cuBLAS dW GEMMs run at 94 % (630 us for 591): the gap is the kernels' epilogue / pipeline, not the schedule (above).
  `tgate_w` at D 256 n 4 [98 304]: 228 us for 98 us of tensor work (43 %).
* Not touched in this round: float32 D 128 L128 (96.2 us, H100 / 1.5 = 92.6), rows 219 master L256-640 (0.70-0.71), rows 210 / 211 L128: the parts 3-5 bounds above stand.

### S90 part 2 (2026-11-18): where the wide bf16 gate and the D 64 backward spend their time (ablations; no adopted change, `cuda:b200_sm100` stays 43)

No counters (ncu blocked), and a stage trace of one CTA would only restate what the build-time ablations below already isolate, so each stage of the kernel was
switched off in turn (probe flags in `rejected/transition_s90_ablation_probes.diff`: `-DNOA` no dy / xn loads, `-DNOGATE` no gate arithmetic, `-DNOST` no TMA stores,
`-DNOMMA` no tcgen05 products; wrong results, timing only; module step, graph, L2 flushed, `fx.py`, us).

**`tgate_w.cu` D 256 n 4 (the gate, [147 456, 256] step 903 us of which the gate is ~340):**

| variant | step | delta |
|---|---|---|
| base | 903 | |
| no gate arithmetic | 941 / 471 vs 936 / 473 (two shapes) | 0 |
| no A (dy, xn) loads | 868 (147 456) / 457 (65 536) | -35 / -16 |
| no MMA | 879 | -25 |
| no TMA stores (h, dA, dB: 906 MB written) | 784 | **-119** |
| no MMA + no stores | 762 | -141 |
| none of MMA, stores, A loads, gate math | 680 | -223 (B loads + barrier / TMEM pipeline remain: ~117 us of the gate) |
| ring 4 -> 3 stages / 2 stages (NST_) | 967 / 1128 | +64 / +225 |

Cause: the gate is bound by its own M x H output writes (h, dA, dB = 3 x M x H x 2 B = 906 MB = 139 us at 6.5 TB/s; removing them is -119 us) plus a TMA-fed K loop that is
very sensitive to ring depth (the producer re-streams the 128 KB dy / xn A panels for every 64-unit chunk: 176 KB per item and CTA, ~64 GB/s per SM, against ring depth 4 at
the 232 KB shared-memory limit). The arithmetic of the gate and the tcgen05 products are not on the critical path (0 and -25 us). The unfused schedule moves each M x H intermediate
through HBM about eight times a step (3 writes in the gate, 2 reads by d_xn, 3 by the dW GEMMs: ~1.6 GB = 250 us of the 674 us step at [98 304, 256] n 4); that, not the tile
kernels' main loops, is what the SOL calculator's byte count (x, dy, dx only) does not contain. What would remove it: d_xn fused into the gate (TMEM: 384 columns for the
double-buffered [a | b] / dh plus 256 for the d_xn accumulator, 640 > 512) or h out of the gate (the D 256 forward has no shared or tensor memory left to stage it, header of
`tfwd_d256.cu`). Neither was built.

**`tbwd_d64.cu` D 64 n 2 [589 824] (backward 215 us, tensor 49 us, memory 46 us):** no gate arithmetic -22 us, no per-tile dgamma / dbeta column reductions (`reduce_scatter32`) -18 us,
both -41 us (291 -> 250 us step); the remaining ~180 us of the backward is per-tile serial latency of the DX role (chunk MMAs, gate, d_xn product, LayerNorm backward on four warps,
dx TMA store: ~4 us per tile at 52 tiles per CTA), which the replica sweeps (part 1) cannot move. Not changed.

### S90 follow-up 2: the 24-warp D 64 n 2 forward (`SW2` + `LN2`, built, bit-identical, 26 % slower: rejected; `cuda:b200_sm100` stays 43 on the merged base)

Base merged: `b200-2.5.x` 5e7dc3865 (Transition `cuda:b200_sm100` 43; an inference change would be 44, none is adopted). `LN2` (in `rejected/transition_s90_d64_ob2_nxs8_tbprobes.diff`, which holds
every probe flag: `TBLN`, `TBSW`, `TBEPI`, `OB2`, `NXS_`, `SW2`, `LN2`): 768 threads, the LayerNorm of a tile split by columns over warps 8-11 (columns 0-31) and 20-23 (32-63); the 16 partial sums of a
row's tree are 8 + 8 and the tree's first level pairs l with l + 8, so each half exchanges its 8 partials through shared memory (`ex1` / `ex2`, 8 KB each, one 64-thread named barrier per warp pair) and both
compute the same tree: **bit-identical to the 16-warp kernel (max |diff| 0 against `ROLES=1`, [1 048 576, 64] n 2, random weights; error vs fp64 3.6e-3 / rms 2.6e-3)**. Registers per role: producer / MMA warps 32
(`setmaxnreg_dec<32>`), every compute warp 88 (`setmaxnreg_inc<88>`; launch allocation 80 a thread, `cuobjdump -res-usage` REG 80, STACK 16 B). 96 deadlocks: the pool is 768 x 80 = 61 440 launched, so
65 536 - 61 440 + 128 x (80 - 32) = 10 240 registers are free for 640 compute threads, i.e. exactly +16 only if nothing else holds a register; at +16 the inc blocks forever, +8 runs. SwiGLU half-warps (units 32 a
thread) fit in 88 with a 16-byte stack; the full SwiGLU needs 152 (hp 32 + 2 x 2 x 16 loads).

| variant ([1 048 576, 64], back to back) | us a tile |
|---|---|
| routed `ROLES=1` | 1.589 |
| `SW2` + `LN2`, one 256-thread barrier per exchange | 2.044 |
| `SW2` + `LN2`, 64-thread pair barriers, last-barrier -> 4 arrivals per CTA | 2.003 |
| same, SwiGLU math removed (`TBSW=2`) | 1.960 |

Stamps (`pf2.py`, medians over CTAs, us): LayerNorm done every 1.9-2.1 us (3.07, 4.61, 6.40, 8.45, 10.50) against ~1.0-1.8 for `ROLES=1`; the expand starts 0-0.2 us after each, the epilogue follows at the same
2.0 period; the SwiGLU chunk period is unchanged (0.77). So splitting the LayerNorm made the LayerNorm stage slower (the ring waits on it), and removing the SwiGLU math does not help (1.96): the cost is in
the split LayerNorm path itself (two exchanges, 4 + 4 warps sharing issue slots with 8 spinning waiters), not in a resource; I could not find the cause with the remaining stamps. No further D 64 variant is
adopted. D 128 not started.

Angles of the brief, by calculation (no kernel built; stated as such):
* (2) LayerNorm statistics out of the ring: a separate pass has to read x again (+134 MB per 1 048 576 rows, +50 % of the ring's 268 MB: memory floor 45 -> 68 us), or write and re-read xn (+100 %). The most the
  ring gains from a free LayerNorm is 0.24 us a tile = 13 us at that size. A net loss. The previous kernel's epilogue does not exist for an inference call.
* (1) role split across the two SMs of a cluster: each SM keeps the same total issue work per row (the roles move, they do not shrink); the xn exchange adds a 16 KB DSMEM packet a tile. It relaxes the
  register file (1.5 roles an SM instead of 3), which `SW2` already showed is not the binding resource for the SwiGLU alone. Not built.
* (4) rows 35-37 one-launch single kernel and (5) the float32 TMEM / MMA shape: not re-measured in this round (the part 6 / 14 / 19 stamps stand); no new clock64 evidence.

### S90 follow-up 3: the one-launch chain re-measured by a skeleton, D 128 stage ablation (no change kept; `cuda:b200_sm100` stays 43)

Base merged again: `b200-2.5.x` tip (Transition 43). Probes: `docs/b200/s90_transition_inf/skel.cu` / `skel.py` (a standalone one-launch skeleton), `s90t.py`; D 128 probe flags in `rejected/transition_s90_d128_stage_probes.diff`.

**(4) Why the one-launch chain loses: measured with a data-movement skeleton.** `skel.cu`: ONE cluster of CS CTAs, one 128-row tile, D 384, n 2 (row 35 L 128, H 768): every CTA bulk-loads its 1/CS slice of
Wa / Wb / Ws (110 KB at CS 16, 221 KB at CS 8) and its 1/CS rows of x, all-gathers xn (98 KB) and then h (184 KB at CS 16) into its peers by DSMEM bulk copies (`cp.async.bulk.shared::cluster.shared::cta`,
mbarrier complete_tx), stores its y slice: NO arithmetic, destination buffers overlaid (an optimistic floor: a real kernel could not hold them). Graph median, L2 flushed, like `floor.py`:

| skeleton | graph us | clock64 (cycles from CTA entry, median over CTAs): x in / xn gathered / weights in / h gathered / end |
|---|---|---|
| CS 16 (non-portable cluster) | **24.6** | 2873 / 10617 / 10679 / 30040 / 43323 |
| CS 8 | **18.4** | 3652 / 8631 / 8693 / 17289 / 24441 |
| the 3-launch chain, same case (measured above) | **14.1** | -- |
| one trivial kernel | 6.1 | -- |

The data movement of the one-launch design alone (zero math, no LayerNorm, no MMA) is already 4.3-10.5 us slower than the whole chain. The cause is the DSMEM exchange: an all-gather of 94 KB (xn) takes ~7 900
cycles and of 184 KB (h) ~18 700 cycles, i.e. 10-12 B/clk (~19-23 GB/s) received per SM, whatever the issue pattern (15 pushes from one thread: 24.6 us; one lane per peer: 24.6 us, identical). The chain moves h through
L2 with 100s of CTAs and does not pay this. Removable costs tested: multi-lane bulk issue (no change); fewer, larger CTAs (CS 8 is faster but its weights stream per CTA is 221 KB and the gather per CTA is the same
total bytes); tensor-map prefetch and early dependents are moot in a single launch. Conclusion for rows 35-37 and L 128 of rows 25 / 26 / 38: a one-launch cluster design cannot reach the chain's 14.1 us; the
replay floor (6.1 us for one kernel, 8.2 for three) is not the blocker, the exchange is. (Part 6's earlier kernel had the same exchange rates: 18 GB/s st.async, ~100 GB/s bulk copies at larger boxes.)

**(6) D 128 n 2 inference forward (`tr_fwd_sm100.cu -DLNH=1`, [262 144, 128], back to back, us a tile; ramp included: 13.8 tiles a CTA):** routed 5.08; no sigmoid 5.03; no SwiGLU loads / math 5.00; no epilogue
loads / adds 4.72; neither 4.48. The three compute stages together cost 0.6 us of 5.1 (12 %): the ring is NOT bound by them. The tensor work alone is 2.37 us a tile (25.2 MFLOP at 10.6 TF/s an SM): the MMA pipe is
~53 % busy with all three stages removed, so the limit is the MMA / TMEM chain itself (chunk dependency expand -> SwiGLU -> squeeze with two TMEM slots, M 256 N 128 pair products at K 128, the squeeze at K 64), not
issue slots or registers. Not pursued further in this round (needs MMA-level probes: per-instruction `tcgen05.commit` latency, N / K shape efficiency; cuBLAS reaches 830 TF/s on the K 256 expand shape, 1365 on large K).

**(5) float32**: the stage breakdown of `tg_fused_h16` exists in part 20 (ablation flags `TBLN`, `TBEPI`, `TBW`, `HSIG 9`): [65 536, 128] 33.8 us routed; no LayerNorm statistics 32.3, no LayerNorm 30.4, no y epilogue 26.2
(the biggest stage, 22 %: the y write itself, 16.8 MB at ~6.7 TB/s), no sigmoid 31.0; [16 384, 256] routed 25.5 -> epilogue 22.3. Re-running it with new flags was not done in this round.

### S90 follow-up 4 (`cuda:b200_sm100` 46): D 128 n 2 inference forward with 3 x stages, corrected one-launch skeleton

**One-launch skeleton, corrected exchange** (`skel2.cu`: 512 threads, chunk-major 16-B `st.async` all-gathers, one warp-instruction = one contiguous 512-B run; `XFULL=1`: every CTA loads the whole x tile itself instead of
gathering xn; `MODE 1`: bf16 y partials reduce-scattered instead of h all-gathered; same row 35 L 128 case, graph median, L2 flushed, zero math):

| exchange | CS 16 | CS 8 |
|---|---|---|
| bulk copies (first skeleton): xn gather + h gather | 24.6 | 18.4 |
| `st.async`: xn gather + h gather (184 KB in) | 16.5 | 16.4 |
| `st.async`: xn gather + bf16 partial reduce-scatter (92 KB in) | 14.3 | 14.3 |
| whole x per CTA (no xn gather) + bf16 partial reduce-scatter | **12.3** | **12.3** |

`st.async` is 1.5-1.7x better than bulk copies, but the xn gather still takes ~8 500 cycles (94 KB, ~11 B/clk a SM) while the weights stream, and the best variant (12.3 us) is above the ~10 us bar and 1.8 x the 6.1 us
replay floor with NO LayerNorm, MMA or SwiGLU in it (those add >= 3.5 us on the chain's own stamps): the one-launch kernel would land at ~15-16 us against the chain's 14.1. Not built; numbers recorded.

**D 128 n 2: where the 4.5 us tile goes** (`tr_fwd_sm100.cu -DLNH=1`, [1 048 576, 128], back to back, us a tile; probe flags in `rejected/transition_s90_d128_stage_probes.diff`): routed 4.59; no MMA instructions 4.16; no weight TMA 4.52;
neither 4.12; neither + no SwiGLU math + no epilogue 3.69; no weights + no SwiGLU + no epilogue (MMA kept) 3.99. Everything but LayerNorm, the x load and the y store removed still takes 3.69 us: so no compute stage is the
bound. Stamps (`pf2.py`, tiles 2-4): the LayerNorm of tile 2 starts 4.6 us after tile 1's, i.e. when its x stage arrives: tile i's x stage is freed by the epilogue of tile i - 2 (stage 0 epilogue end 9.73 us), the TMA load
of 32 KB from HBM then takes 1.8 us (LayerNorm starts 11.52), LayerNorm computes 1.3 us (12.8), the chunk chain + epilogue 6.4 us (19.2): **the x stage lifetime is ~9.5 us and with 2 stages the tile period is half of it**.

**Fix: 3 x stages (`NXS_=3`) beside a 2-slot weight ring (`NW_=2`)** (3 x 32 KB x + 2 x 24 KB ring + 2 x 32 KB xn = 208 KB; 3 stages with a 3-slot ring need 234.5 KB > 227): `tr_fwd_sm100.cu`, the x index is now `i % NXS`
(phases `i / NXS`), the xn index stays `i & 1`; `fused_wide_sm100a.FWD_INFER_FLAGS[(128, 2)] = (-DLNH=1, -DNXS_=3, -DNW_=2)`. The training forward (`fwd`) and D 128 n 4 (`fused_sm100a`) keep NXS 2 / NW 4. Same bits (max |diff| 0
against the routed kernel at [1 048 576, 128], [262 144, 128]).

| | before | after |
|---|---|---|
| us a tile, [1 048 576, 128] n 2 | 4.59 | 3.91 |
| eager median, [262 144, 128] n 2 | 76.7 | 66.5 |
| D 128 n 4 [1 048 576, 128] (not adopted there) | 5.78 | 5.96 (NXS 3 / NW 2) |

`measure.accuracy` row 27 master eval, every length (random weights, fp64 gold, tol = max(err_pytorch, fixed)): **18 / 18 GREEN**. `measure.speed` (cuda, graph, L2 flushed, one exclusive process a length, GPU 2), row 27 master, us before -> after:
L 128 16.13 -> 16.11, 256 28.45 -> 26.43, 384 47.06 -> 42.83, 512 75.58 -> 65.50, 640 110.61 -> 96.08, 768 155.42 -> 132.99, 1024 259.98 -> 220.85, 1536 561.57 -> 479.23, 2048 985.89 -> 851.74 (SOL 47-53 % -> 55-62 %, row 27 L 256: 28.8 -> 31.0 %).
All 181 eval cases (`out/sol_v44.tsv` = `docs/b200/s90_transition_inf/sol_v44.tsv`): master SOL% median 52.6 -> 59.2, all-case median 53.3 -> 55.5; cases >= 90 / 70 / 50 %: 13 / 66 / 96 -> 13 / 65 / 99 (a 70 % boundary case moved down by noise elsewhere). 1.5x misses 25 -> 24, slower than the H100: 0, ladder 0
(cuda vs pytorch records; Triton not re-measured). Version 46 (inference, even; 45 is the training merge); no new cubin name (the `fwdi` cubin of D 128 n 2 has new flags, so the sm100 prebuild plan, derived from `_specs`, builds it).

The same stage-lifetime reading explains the D 64 empty pipeline (8 stages: 1.04 -> 0.82 us a tile) but not its real kernel, whose compute chain, not the stage count, paces it (`NXS_=8` level). Rows 28 / 39 / 34 (D 128 n 4 and wider) keep 2 stages.
### S90 part 3 (2026-11-18): d_xn fused into the D 256 gate (`tgate_wx.cu`; built, correct, slower; rejected, `cuda:b200_sm100` stays 43)

Built as `-DDIM=256` variant of `tgate_w.cu` (diff `rejected/transition_s90_gate_dxn_fused_d256.diff`, flag `fused_wide_sm100a.D256_FUSE`): per chunk the gate's staged dA_j / dB_j tiles
are the A operand of d_xn += [dA_j | dB_j] [Wa_j^T; Wb_j^T] (M 256, N 256 over the pair, K 2 x 64, B = the K-major tiles of [Wa; Wb]^T, 32 KB per chunk and CTA through a second TMA
buffer), accumulated over the tile's 16 chunks in 256 TMEM columns; [a | b] + dh are single-buffered (192 + 256 = 448 columns), the ring is 3 stages (212 KB of shared memory:
ring 132 + staging 48 + B operand 32); the accumulator is drained to the bf16 d_xn tensor (row per thread) at the tile's last chunk and the separate d_xn GEMM launch goes.
Correct: y and dWa / dWb / dWs bit-identical to the unfused path, dx / dgamma regrouped (d_xn summed in chunk order: max |dx| 6e-2 on values O(10), one bf16 ulp), deterministic over two steps ([16 384 / 65 536, 256] n 2 / n 4).

Step, graph, L2 flushed (us): [98 304, 256] n 4 669.7 -> 741.4, [147 456, 256] n 4 899.2 -> 999.4, [16 384, 256] n 4 145.4 -> 149.5: **slower (+7 to +11 %)**.
Ablations of the fused kernel ([98 304], step 737): no B-operand loads 739, no d_xn products 756, no drain stores 704, none of the three 672 -- i.e. the fused skeleton with neither the
products nor the drain still costs the old gate + d_xn GEMM (329 us: gate 228 + GEMM 101) in one kernel. The two structural costs: the 3-stage ring (the unfused gate at 3 stages is +64 us
at [147 456], part 2) and the single-buffered [a | b] / dh (the products of chunk q + 1 wait for chunk q's tensor-memory reads, the d_xn products queue behind them), ~55 us; the drain
sits on the critical path (-33 us when its stores are dropped: the MMA warp's first d_xn product of the next tile waits for it). The d_xn products themselves cost ~0 (they hide under the store-bound gate).
D 384 (N 384 + 192 = 576 columns) and D 512 do not fit tensor memory even single-buffered; D 128 is already the two-role kernel. What would still be needed to win: 4+ ring stages
(shared memory: nothing left at 212 KB) or a 32-unit chunk (96-column [a | b] / dh buffers, double-buffered next to the 256-column accumulator = 448), which halves the tcgen05 N of the
gate products; not built.

### S90 part 4 (2026-11-18): the fused gate with 32-unit chunks, double-buffered tensor memory, 4 ring stages, h from registers (built, correct, slower; rejected)

Diff `rejected/transition_s90_gate_dxn_fused_d256_c32.diff` (replaces the part 3 kernel; same flag `D256_FUSE`). Items are 32-unit chunks: [a | b] + dh = 96 columns double-buffered (the two gate
warpgroups take the even / odd chunks) next to the 256-column d_xn accumulator (448 of 512); two chunks fill one 64-unit staging pair (dA | dB, 32 KB) that is TMA-stored and multiplied into d_xn;
h goes from registers to global memory (no h staging: 16 KB freed), so the ring is 4 stages of 38 KB (222 KB in all); each warpgroup drains half of the previous tile's d_xn columns after its first
chunk of the next tile (the drain overlaps the next tile's chunks). Correct (y and dW bit-identical to the unfused path, dx / dgamma regrouped as in part 3, deterministic, [16 384 / 65 536, 256] n 2 / n 4).
Step (us): [98 304, 256] n 4 669.6 -> 757.8, [147 456] 907.3 -> 1034.2 (**+13 %**; part 3's 64-unit kernel was +11 / +11 %).
Ablations ([98 304], step 758): no dy / xn loads 688 (-70), no d_xn products 737, none of {A loads, d_xn products, drain stores} 651 (the fused skeleton alone is 19 us below the unfused gate + d_xn GEMM).
So halving the chunk doubles the A (dy, xn) re-streaming per hidden unit (304 KB per 64 units and CTA against 176 KB), which the ring cannot hide (-70 us without it), and removing every extra
cost of the fusion would still leave it at ~651 us against 670 for the unfused path: there is no configuration of this tile layout that wins by more than the ~3 % that all ablations together leave.

### S90 part 5 (2026-11-18): the D 64 two-role backward (rows 209 / 210 / 220), D 64 forward ablation, fp32 D 512 pair tiles (`cuda:b200_sm100` 45)

Base: `b200-2.5.x` bb9d44e32 + this branch. Probes: `b200cfg/prb64.py` (`%globaltimer` stamps of DX CTA 0, written past the end of the `dgbw` partial-sum tensor and read back, not printf: a
printf per event spread a 4 us tile over 140 us), `prb64.sh` (per role CTA end times), `fe.py` / `fej.sh` (eval forward with build flags), `fx.py` (module step A/B).

**Where the D 64 backward spends its time** ([589 824, 64] n 2, 215 us, 52 DX tiles per CTA): both roles end together (DX CTAs 212.7-216.8 us, DW CTAs 213.0-214.8: the replica split is
balanced). DX CTA 0 per tile (4.0 us period): input tile seen 79.07 us, the tile's two chunks gated 79.5 -> 81.0 (0.64 us each), d_xn done 81.3, **epilogue sees d_xn 82.5, x tile seen 83.3
(0.74 us wait), d_xn read 82.56, dx store issued 86.0 (LayerNorm backward 2.7 us on 4 warps)**, next tile's input seen 82.94: a tile buffer lives ~7 us (load, chunks, epilogue, store) and there are two.
Ablations (step 291): no LayerNorm-backward arithmetic -4 us only (286.7), because the DW role is the co-limiter: with it and replicas 40 the step is 223 us (-23 %), i.e. a free epilogue would be
worth 68 us; no gate arithmetic -22; no dgamma / dbeta reductions -18. A deeper pipeline alone does nothing: d_xn accumulator double-buffered (`-DDXN2`) 290.8 -> 290.8; three [a | b] buffers + one dh
(`-DAB3`, TMEM 384 + 64 + 64) 290.8 -> 290.8 / 386.9 -> 386.9 (n 4); three input tile buffers (`-DTB3`) 288.7 / 378.9; both LayerNorm-backward groups alternating tiles with the gate on one
warpgroup (`-DEPI2`) 292.9.
What bound it: the producer issues in order -- tile i + 1's input, then the reload of tile i's dead xn slot with x -- and the reload waits for tile i's products while the next input waits for a free
tile buffer, so x arrived late (the 0.74-1.8 us waits) and buffer reuse was serialised. **`-DXSEP`**: the DX input stage carries x (dy | xn | x, 48 KB; no reload, no `xn_dead`): n 2 290.8 -> 266.2 us
(-8.5 %), n 4 387.0 -> 370.7 (-4.2 %). With XSEP the other pieces pay off together (n 2: `-DEPI2 -DDXN2 -DXSEP -DTB3`, replicas 34): 241.6 us (-17 %); alone EPI2 / DXN2 / TB3 add nothing or lose
(n 4: EPI2 worse, TB3 does not fit shared memory). Adopted: n 2 `EPI2 DXN2 XSEP TB3` + replicas 34 from 129 tiles (`REPL_TILES`), n 4 `XSEP` (`D64_BWD_FLAGS`).
Step, graph, L2 flushed (us, old -> new): D 64 n 2 [16 384 / 65 536 / 147 456 / 262 144 / 589 824] 28.64 / 51.17 / 88.06 / 143.3 / 290.8 -> 28.64 / 49.15 / 79.8 / 122.8 / 241.5;
n 4 [16 384 / 65 536 / 147 456 / 589 824] 32.77 / 63.5 / 112.5 / 385.0 -> 32.77 / 63.5 / 112.6 / 370.7. Accuracy (`measure.accuracy`, rows 209 / 210 / 220 master train, every length): 18 / 18 GREEN
(0.60-0.93 x tol, the same numbers as before); y and dx bit-identical to the old build on seven shapes, the fp32 weight gradients regrouped (replica count; max |d| 1.2e-2 at [589 824, 64]), deterministic
over two steps. Not built: a lighter LayerNorm-backward epilogue (the dgamma / dbeta shuffles, 64 gamma loads per row) -- with the DW role as co-limiter a faster DX epilogue is only worth it together with a re-split.

**D 64 forward** (eval build, [589 824, 64] n 2: 57.3 us against 23 us memory and 18 us tensor time; flags in `rejected/transition_s90_tfwd_d64_ablation_probes.diff`): no SwiGLU arithmetic 53.3, no
LayerNorm 51.2, no epilogue add 55.3, none of the three 45.1: the skeleton (x loads, xn / y stores, product chain) is 45 us, 2x the copy time; x stages 4 -> 8 57.3 -> 57.3 (2: 83.9). The
three arithmetic stages together are 12 us; nothing in the arithmetic is the largest stage, so no change.

**fp32 D 512 tile shape**: `GATE_AB` as CTA pairs (`_BNS2_F32["gate_ab"]` 64 / 128 / 256; the pair builds of `swiglu_ab` / `resid` are already on) [262 144 / 65 536, 512] n 2 step 5331 / 1380 (64), 5005 / 1266 (128),
4974 / 1270 (256) against ~4850-5000 / 1288: within the run-to-run spread, not adopted. The fp16-exact / bf16-split storage of a, b, h, dab (the ~14 GB per step) was not built.

**Not built**: the single-stream rows 217-219 one-launch skeleton with chunk-major st.async packets.

SOL after part 5 (m2, all 126): master 4.3 / 41.3 / 66.2 %, float32 3.3 / 24.5 / 56.8 %; >= 90: 0, >= 70: 0, >= 50: 32. Rows 209 / 210 / 220 (min - max): 6.6 - 23.3 -> 6.5 - 28.0, 11.5 - 35.0 -> 11.5 - 36.4,
20.5 - 32.4 -> 21.2 - 34.1. 1.5x misses 9 (row 212 L128 on the line at 0.668, the others as before), ladder 0.

Post-merge check (tree with Transition 45 merged, version 46): row 27 master eval 18 / 18 GREEN; `measure.speed` L 512 / 2048 as above (see report).

### S90 follow-up 5 (no change kept; `cuda:b200_sm100` stays 46): buffer-lifetime lesson applied to D 64 n 2 and to the fp32 epilogue

Diff of both experiments: `rejected/transition_s90_h16_ye_d64_nhs4.diff` (`tfwd_d64.cu -DNHS_=4`, `tg_fused_h16.cu -DYE`).

* **D 64 n 2 inference forward** (`tfwd_d64.cu -DROLES=1`, [1 048 576, 64], back to back, us a tile): the x stage count is not the bound (4 -> 8 stages: 1.589 -> 1.592 with the real compute; it IS the bound of the empty pipeline, 1.04 -> 0.82), the producer order is already
  ahead (4 stages, the next tile's x load issued as soon as its stage's epilogue freed it: no dead-slot reload), the y accumulator double buffer (`OB2`) was level, and the h slot recurrence (SwiGLU store -> `h_full` -> squeeze -> `sq_done` -> next store, once a tile at the 2 h slots)
  was tested by **4 h slots in tensor memory** (`NHS_=4`: T_H 256 .. 383 fits; h_full / sq_done x4): 1.586 -> 1.589 us a tile, bit-identical. The per-tile timeline (`pf.py` stamps, one CTA): LayerNorm done -> expand start 0.26 us, SwiGLU chunk start 0.26 us after the previous chunk's end (the SwiGLU
  warpgroup is busy 0.51 of every 0.77 us per chunk), epilogue 1.0 us, all with period 1.54 us; no wait is idle by more than ~0.26 us anywhere, so the chain of per-chunk handoffs (each a remote arrive + wake) is what sets 1.54 us, and none of the recurrences tried (xn 2 stages, AB 2 slots, h 2 -> 4 slots,
  y 1 -> 2 buffers, x 4 -> 8 stages) is the single binding one. Not resolved.
* **fp32 y epilogue of `tg_fused_h16`** (22 % of the kernel when removed, part 20 `TBEPI`): `YE` loads all accumulator columns of a tile into registers first and frees the y accumulator right after the TMEM loads (not after the residual adds and stores), so the next tile's first squeeze no longer waits for them
  (D <= 128: 64 registers; D 256 keeps its layout). Fine us (`hvf.py`, random weights, outputs bit-identical, max |cur - var| 0), YE -> base: [16 384, 128] n 2 11.75 -> 11.65, [65 536, 128] 32.54 -> 32.17, [147 456, 128] 61.60 -> 61.03, [589 824, 128] 225.85 -> 224.04, D 64 [65 536] 18.13 -> 18.10, [262 144] 52.97 -> 52.76:
  level or 1 % slower. At [65 536, 128] the kernel moves ~67 MB through HBM (y write + the x read; the residual re-read is an L2 hit) = 10 us at 6.5 TB/s of its 32.5 us, and its fp16 tensor time is 8.4 us: it is neither HBM nor tensor bound, the epilogue's 7.6 us is serial (the y store / residual pass of the CTA's tiles), not overlap-limited at the accumulator.
  Not built: the K-loop / `kind::f16` ablation of the float32 path (the part 20 table is the stage breakdown: no LayerNorm statistics 32.3, no LayerNorm 30.4, no sigmoid 31.0, no y epilogue 26.2 at [65 536, 128]).

### S90 follow-up 6 (no change kept; `cuda:b200_sm100` stays 46): packed f16x2 SwiGLU at D 64; D 256 stage analysis

* **(a) SwiGLU of a unit pair in packed f16x2** (`tfwd_d64.cu -DSWH=1`, built and measured; probe in this note only: `A16 = f2h2(a0, a1)`, `B16 = f2h2(b0, b1)`, `S16 = hfma2(htanh2(hmul2(A16, 0.5)), 0.5, 0.5)`, `H16 = hmul2(hmul2(A16, S16), B16)`, then f16x2 -> two f32 -> bf16x2 for the MMA operand): half the MUFU ops
  but 11 instead of 7 instructions a pair. [1 048 576, 64] n 2, back to back: **1.588 -> 1.906 us a tile (+20 %)**; error vs fp64 rms 2.598e-3 -> 2.622e-3 (max|diff| to the fp32 path 0.031 = one bf16 ulp; GREEN-level), so slower with no accuracy case to answer for. The ablation bound already said why: removing the sigmoid entirely gains 0.09 us a tile, the whole SwiGLU 0.14.
  (b) wide / pipelined TMEM loads (already 16-column loads with the next pair in flight), (c) SwiGLU split over two warpgroups (`SW2`: level, then LN-bound) and (d) 32-unit chunks (doubles the per-chunk handoffs, each a remote arrive + wake of the ~0.26 us that sets the period) were not built: bound by the same ablation (<= 0.14 us of 1.59) and by the handoff count.
* **(2) D 256 inference forward** (`tfwd_d256.cu`): the x-lifetime lesson is already in this kernel: x(i) is loaded TWICE, into the xn buffer (for the LayerNorm, issued once xn(i - 1) is dead, i.e. during tile i - 1's tail) and into the x buffer (for the residual, once the epilogue of tile i - 1 has stored); the second load hits L2. A third buffer cannot exist (231.9 KB of 232 KB used; ring 2 slots of 48 KB, x 64 KB, xn 64 KB), so the remaining gap
  (row 29 master L 2048: 15.3 us a tile against 9.5 us of tensor time, 62 %) is not the load. D 384 / D 512 run the small-M chain / generic GEMM chain (tiles of 128 x 384-512 with L2-resident h): no per-tile ring to lengthen. No measurement of those two was made in this round.
* **(4) D 128**: the TMEM of the D 128 n 2 forward is full (a | b 256 + h 64 + y 128 + xn 2 x 64 = 512), so more chunk slots do not fit; expand and squeeze are already issued by two different warps (1 and 2) with one commit a chunk and operand (the granularity of the commits is per chunk). Not built.

### S90 follow-up 7 (`cuda:b200_sm100` 48): D 256 inference forward -- the per-tile timeline, the idle-tensor-pipe bubble, L2 prefetch of the next x; D 384 / D 512 breakdowns

**D 256 n 2 timeline** (`tfwd_d256.cu` built with `-DPROBE=1`, `pf3.py`: `%globaltimer` stamps to global memory, median over the 74 leader CTAs, [589 824, 256] = 31 tiles a CTA; the stamps now cover tiles 0-5 of every phase, `docs/b200/s90_transition_inf/d256t.sh`). Tile period 14.2 us against 9.5 us of tensor work (62-67 %):

| phase of tile 1 (us on the CTA's clock) | stamp |
|---|---|
| last expand of tile 0 done, xn buffer free; the next x is loaded INTO it | ~12.5 |
| LayerNorm(1) starts (after the load landed: waited ~3.6 us) | 16.13 |
| LayerNorm(1) done (2.05 us, 8 warps, 3 passes) | 18.18 |
| SwiGLU(1) chunk 8 | 18.43 |
| epilogue(0) 18.18 -> 20.99 (runs on the SwiGLU / LayerNorm warps, between chunks 9 and 10: 3.3 us gap) | |

The tensor pipe idles from the last expand of tile i (~12.5) to the first expand of tile i + 1 (LayerNorm done, 18.2): **~5.7 us of the 14.2 us tile = 3.6 us of exposed x load + 2.05 us of LayerNorm**, both behind the single 64 KB xn buffer (the kernel's own comment: smem is full: ring 2 x 48 KB, x 64 KB, xn 64 KB).
(D 384 `fwd_d384`: 29.9 us a tile at row 31 L 1536 against 21.3 us of tensor time, the same structure, 8.6 us bubble.)

**Built (`PF2`, adopted, `cuda:b200_sm100` 48): the x boxes of tile i + 1 prefetched into L2 (`cp.async.bulk.prefetch.tensor.2d.L2.global`, one box a lane) when the producer starts tile i**, so the load into the xn buffer hits L2. Same bits (max |diff| 0 against the routed kernel on all shapes tried).
us a tile, back to back: [589 824, 256] n 2 14.00 -> 13.60, [147 456, 256] n 2 15.28 -> 14.81, [16 384, 256] n 2 20.58 -> 20.57 (one tile a CTA), [262 144, 256] n 4 22.85 -> 22.60 (`XPAR`; inference builds `fwdi` of (256, 2) / (256, 4)). `measure.accuracy` rows 29 / 30 master eval every length: **36 / 36 GREEN**.
`measure.speed` rows 29 / 30 master (us, 46 -> 48): row 29 L 256 69.41 -> 67.58, 384 124.80 -> 120.78, 512 206.72 -> 200.48, 640 313.22 -> 302.91, 768 444.32 -> 429.97, 1024 761.60 -> 741.31, 1536 1861.65 -> 1793.84, 2048 3368.70 -> 3268.38; row 30 L 256 104.22 -> 102.30, 384 192.38 -> 188.26, 512 321.41 -> 315.14, 640 497.31 -> 487.41,
768 724.94 -> 706.43, 1024 1304.91 -> 1246.11, 1536 3140.03 -> 3149.17, 2048 5597.04 -> 5675.38 (row 30 L >= 1536 level within noise). 1.5x misses 24 -> 24, ladder 0. Full table: `docs/b200/s90_transition_inf/sol_v48.tsv`: all-case SOL% median 55.5 -> 56.0; >= 90 / 70 / 50: 13 / 67 / 98.

**The lever behind the timeline (not built)**: the 5.7 us bubble is hidden only if tile i + 1's xn is normalised while tile i's expands run, which needs a second xn buffer (64 KB). With the residual read from global memory (L2 hit, as `tg_fused_h16` does) the 64 KB x buffer becomes that second xn buffer, but the epilogue then needs a y staging area (8 warps x 4 KB) that the 232 KB do not hold beside
ring + 2 xn (96 + 128 + 3 = 227 KB + 32 KB); direct 16-B row stores from registers would do without. The ceiling is the tensor time: 14.2 -> ~10 us a tile (row 29 L 2048 64 -> ~90 %). A rewrite of the epilogue / LayerNorm barrier protocol of this kernel; not done in this round.

**D 384 / D 512 inference, measured now** (`tl.py`, kernels of one flushed graph replay, us; [rows, D] with `measure.speed`'s lengths):
| case | kernels (duration) | tensor SOL | binding |
|---|---|---|---|
| row 33 D 512 n 2 L 512 (700 us; 74.8 %) | `transition_ln_w` 93.7, `transition_swiglu_w_sm100` 393.3, `transition_gemm_nd_sm100` 253.7 | expand 350, squeeze 175 | LayerNorm pass 13 % of the step (268 MB x in + xn out: 82 us of HBM at 6.5 TB/s, it runs at 88 %), expand at 89 % of its tensor SOL, squeeze at 69 % (254 vs 175: h in 537 MB + residual + y out: 164 us of HBM overlapping the tensor work) |
| row 34 D 512 n 4 L 640 (1990.7 us; 75.5 %) | ln 151.0, swiglu 1341.7, gemm 700.6 | expand 1045, squeeze 523 | expand at 78 %, squeeze at 75 % |
| row 31 D 384 n 2 L 1536 (3625 us; 70 %) | one kernel `transition_fwd_d384_sm100` 3728.8 | 18 432 tiles / 148 = 124.5 a CTA: 21.3 us a tile | 29.9 us a tile: the bubble above (x load + LayerNorm behind one 96 KB xn buffer) |
| row 31 D 384 n 2 L 512 (385 us; 77 %) | one kernel 379.7 | | same |

What binds the cases under 70 % (rows 29 / 31 / 33 at large L): D 256 / D 384 the xn-buffer bubble (5.7 / 8.6 us a tile), D 512 the separate LayerNorm pass (13 %) and the squeeze GEMM's 69 % of tensor SOL. `PF2` for D 384 was not built (its producer is one lane that also issues every chunk's Ws boxes; a lane-parallel prefetch needs a tile-pacing barrier).

### S90 follow-up 8 (`cuda:b200_sm100` 50): D 256 inference forward with two xn buffers and no x buffer (`tfwd_d256.cu -DX2=1`)

The lever of follow-up 7, built. What changed (`X2`, needs `XPAR WPAR LN2`; the training forward `fwd` and the one-tile-a-CTA calls keep the 2.5.14 / PF2 build, `fwdi1`):
* **Two xn buffers** (XN[i & 1] at the old x / xn buffers' places, 2 x 64 KB): x(i) is loaded into XN[i & 1] as soon as the last expand of tile i - 2 is done (`xn_empty[2]`, `xl_full[2]`, `xn_full[2]`; tiles 0 and 1 at setup), so the LayerNorm of tile i + 1 runs while tile i expands (the loop of the LayerNorm warps is `ln2(i + 1); epi2(i)`) and the tensor pipe never waits for a load or a LayerNorm.
* **No x buffer.** The residual x is re-read from global memory and y written by 128-B row segments from registers (`tsx_store.cuh` `load_t<32>` / `store_t<32>`: groups of 8 lanes cover one 128-B segment of a row, a warp instruction covers four rows x 128 B). The residual loads of tile i are issued before the wait for its accumulator; the y accumulator is released (`out_empty`) after the last TMEM load, before the last K-block's adds and stores.
  The kernel's `rstd` / `c1` arguments carry x / y in this build (`fused_wide_sm100a.FWD_X2`, `X2_RESID`; the inference call needs neither).
* **Roles:** warps 4-7 every SwiGLU chunk (one warpgroup; was two alternating), warps 8-15 the two LayerNorm / epilogue halves.
Registers 128 (STACK 8 B). Same bits as the 2.5.14 build at every shape tried (max |diff| 0: n 2 at 384 / 1 152 / 16 384 / 20 480 / 65 536 / 262 144 / 589 824 rows; n 4 equal in the same runs). Back to back, us a tile (base PF2 build -> X2): n 2 [589 824, 256] 13.53 -> 11.88, [262 144] 13.83 -> 12.04,
[65 536] 16.99 -> 14.68, [20 480] 30.4 -> 27.1; n 4 [589 824] 24.55 -> 21.53, [262 144] 22.58 -> 21.17, [65 536] 26.43 -> 24.11. One tile a CTA ([16 384, 256]) is level (20.39 / 20.55), so the host routes those calls to `fwdi1` (X2 off): the first X2 version cost row 29 L 128 +1.9 us.
Ablations of the X2 kernel ([147 456, 256], 12.87 -> ): no SwiGLU loads / math 11.17, no residual loads / stores 10.41: the epilogue's instruction cost (shuffles of the row transposition, 128 registers) and the single SwiGLU warpgroup are what is left above the 9.5 us of tensor time. (The first X2 version, residual loaded after the accumulator wait and `out_empty` at the end of the epilogue: 14.71 -> 13.33, then 12.87, then 12.53 with the two changes above.)

`measure.accuracy` rows 29 / 30 master eval, every length: **GREEN on every case** (72 records, the file holds two runs). `measure.speed` (cuda, graph, L2 flushed, us; 48 -> 50) and SOL% (pure):
row 29 L 128 24.53 -> 24.51 (33.5 %), 256 67.58 -> 59.17 (55.5), 384 120.78 -> 100.32 (73.6), 512 200.48 -> 161.66 (81.2), 640 302.91 -> 245.84 (83.4), 768 429.97 -> 356.00 (83.0), 1024 741.31 -> 629.58 (83.4), 1536 1793.84 -> 1505.15 (78.5), 2048 3268.38 -> 2687.94 (78.2; was 64 %);
row 30 L 256 102.30 -> 92.16 (71.2), 384 188.26 -> 167.95 (87.9), 512 315.14 -> 282.54 (92.9), 640 487.41 -> 442.14 (92.8), 768 706.43 -> 647.10 (91.3), 1024 1246.11 -> 1146.62 (91.6), 1536 3149.17 -> 3020.14 (78.3), 2048 5675.38 -> 5214.14 (80.6; was 74 %).
All 181 eval cases (`sol_v50.tsv`): master SOL% median 59.8, all-case median 55.9 (53.3 at the start of the round), cases >= 90 / 70 / 50 %: 21 / 74 / 100 (13 / 66 / 96 at the start). **1.5x misses 25 -> 22, ladder 0.** Version 50 (inference, even); `fwdi1` is a new cubin of the `_specs` list the sm100 prebuild plan is derived from.

### S90 follow-up 9 (no change kept): D 384 prefetch, second xn buffer at D 384

* `tfwd_d384.cu -DPF2=1` (inference cubin `fwdi`: lanes 1-12 of the producer warp prefetch tile i + 1's 12 x boxes into L2 once tile i's x has landed; diff `rejected/transition_s90_d384_pf2.diff`): rows 31 / 32 master eval `measure.speed` (us, v50 -> PF2): row 31 L 512 380.9 -> 385.0, 768 848.2 -> 858.0 (857.4 again), 1024 1591.6 -> 1594.1, 1536 3597.6 -> 3816.3 (3801.0), 2048 6718.3 -> 6737.2;
  row 32 L 512 636.8 -> 649.2, 768 1460.0 -> 1555.9 (1522.9), 1024 2863.4 -> 2720.5, 1536 6473.6 -> 6569.0 (6606.1), 2048 11603.9 -> 11930.0. Level to 6 % slower; not adopted (the run-to-run spread of these long cases is ~5 %).
* A second xn buffer at D 384 does not fit: 2 x 96 KB + the 72 KB weight ring = 264 KB against 227 KB (the D 256 build had 64 KB buffers and freed its 64 KB x buffer by re-reading the residual; the D 384 kernel already has no x buffer). The ring would have to shrink to ~35 KB (one 24 KB Ws slot + one 8 KB K-block stage), which starves the chunk loop (its weight stream paces it). Not built.

### S90 follow-up 10 (no change kept; `cuda:b200_sm100` stays 50): the residual re-read / row-store epilogue at D 128 and D 64, and the final re-score with the lowest-15 blockers

**The D 256 lesson does not transfer to D 128 / D 64 (both built `RX=1`, bit-identical, slower; diffs `rejected/transition_s90_d128_rx_resid_reread.diff`, `rejected/transition_s90_d64_rx_resid_reread.diff`):** the residual x re-read from global memory in 128-B row segments before the accumulator wait (`load_t<32>`), y from registers by `store_t<32>`, the x stage released right after its LayerNorm, the accumulator released after the TMEM loads.
* D 128 n 2 (`tr_fwd_sm100.cu -DLNH=1 -DRX=1`, [1 048 576, 128] us a tile; base = `NXS_=3 NW_=2`, 3.90): RX 4.43-4.45 (3 x stages + 2-slot ring), 4.42-4.52 (2 stages + 4-slot ring, 2 + 3-slot ring); [262 144, 128] 4.43 -> 4.97-5.07. Loading tile i's residual before the LayerNorm of tile i + 1 (in flight during it) instead of at the epilogue: 4.45, no change.
* D 64 n 2 (`tfwd_d64.cu -DROLES=1 -DRX=1`, [1 048 576, 64]): 1.592 -> 1.876 us a tile (+18 %).
The staged TMA-store epilogue is cheaper than the register transposition + 128-B segment stores at these tile sizes (D 256's tile is 12 us, so the epilogue's ~2.5 us hides behind the tensor pipe once the xn buffer is doubled; D 128 / D 64 tiles are 1.6-3.9 us and the epilogue instruction time lands on the chain).

**Re-score of all 181 eval cases (v50, `sol_v50.tsv`)**: master SOL% min / median / max 4.0 / 59.8 / 93.3 (>= 90 / 70 / 50 %: 12 / 60 / 77), float32 4.7 / 53.7 / 103.9 against the TF32 peak and 4.7 / 32.4 / 56.6 against the fp16 peak (9 cases >= 50 %), all 181 4.0 / 55.9 / 103.9 (21 / 74 / 100). Start of the round: 53.3 median, 13 / 66 / 96. With the 6.1 us replay floor added: median 63.2, 24 / 80 / 159. Against the sustained peaks: 43 cases >= 90 %.
Per row, pure SOL% min - max: 25 5.4 - 49.6, 26 8.5 - 51.1, 27 12.7 - 61.7 (float32 23.9 - 64.2), 28 22.5 - 86.1, 29 33.5 - 83.4 (float32 54.1 - 96.1), 30 47.4 - 92.9, 31 45.5 - 78.6, 32 64.4 - 92.8, 33 53.6 - 75.2 (float32 75.5 - 103.9), 34 71.4 - 85.8, 35 4.0 - 12.6 (float32 4.7 - 23.6), 36 6.9 - 36.5, 37 13.1 - 34.8 (float32 14.1 - 53.4), 38 16.9 - 45.4, 39 64.6 - 86.3, 40 87.9 - 93.3.

**The lowest 15 and what blocks each (all measured earlier in this round)**: rows 35 (master L 128-768, 1024; float32 L 128-384) and 36 (master L 128-512) and row 25 master L 128 (4.0 - 7.4 %; with the 6.1 us replay floor 45 - 56 %):
* replay floor, measured: one tiny kernel 6.1 us, three 8.2 us (`floor.py`); t_sol of these cases is 0.6 - 1.2 us, so even a perfect single kernel reads 9 - 16 %.
* the 3-launch chain (LayerNorm 2.9 us, expand 1.7, completion hop 1.2, squeeze 2.3 - 3.0; 14.1 us total) cannot become one launch: the data-movement skeleton of the one-launch design (no arithmetic) measured 12.3 us (best variant: `st.async` partial reduce-scatter, whole x per CTA) to 24.6 us, above the chain even before LayerNorm / MMA / SwiGLU are added;
* row 25 L 128 (D 64, 128 tiles one a CTA): one kernel of ~8 us device + the 6.1 us replay; its tile latency (x in 2 us, LayerNorm, the chunk chain, the epilogue) is the D 64 per-chunk handoff chain, measured 0.77 us a chunk with no single resource (OB2, NXS 8, NHS 4, SW2, LN2, RX, SWH all level or slower).

### S90 follow-up 11 (no change kept; `cuda:b200_sm100` stays 50): D 256 role split, float32 MMA ablation

* **(3) D 256 `X3`** (built, bit-identical except one bf16 ulp from the single-warpgroup LayerNorm's summation order; diff `rejected/transition_s90_d256_x3_roles.diff`): with the LayerNorm off the critical path (two xn buffers) it takes ONE warpgroup (warps 8-11, whole rows), the SwiGLU goes back to two warpgroups alternating chunks (warps 4-7 even, 12-15 odd, then epilogue half 1), the epilogue halves on warps 8-11 / 12-15.
  us a tile: [147 456, 256] n 2 12.48 (X2) -> 14.96, [589 824] 11.88 -> 14.08. Slower: one warpgroup's whole-row LayerNorm (~4 us) plus its epilogue half no longer hides, and the odd-chunk warps wait behind the epilogue's global loads. The TMEM-transposed load (`tcgen05.ld 16x256b`) was not built: its fragment layout gives each lane 2 consecutive columns of two rows (4-byte bf16 pairs, 16 B only per 4 lanes), so a row-major y store would be 4-B granular without a shuffle; the `store_t` butterfly (3 x 4 exchanges of 4 shuffles a 64-column block) is what it would replace.
* **(5) float32 `tg_fused_h16` MMA ablation** (`-DTBMMA=1`, the expand / squeeze `tcgen05.mma` removed, commits kept; probe diff `rejected/transition_s90_h16_tbmma_probe.diff`; fine us, `hvf.py`):

| case | routed | no MMA | no MMA + no epilogue | + no LayerNorm statistics + no sigmoid |
|---|---|---|---|---|
| [16 384, 128] n 2 | 11.67 | 10.31 | | |
| [65 536, 128] n 2 | 32.27 | 27.60 | 20.32 | 15.01 |
| [262 144, 128] n 2 | 102.38 | 83.68 | | |
| [16 384, 256] n 2 | 23.28 | 19.18 | | |
| [65 536, 256] n 2 | 75.62 | 62.10 | | |

  The MMA instructions account for 12-18 % of the kernel (4.7 us of 32.3 at [65 536, 128]; the fp16 tensor time there is 8.4 us): the tensor pipe is not what binds it; the serial stages around it are (epilogue 7.3 us with the MMA removed, LayerNorm + sigmoid + the x / weight streams the rest 15 us). The K loop is short (K = D = 128 / 256, four K16 steps a K-block) and the accumulator layout is one [a | b] chunk of N = 128; none of it is the limit.
  Can the fp32 x read (4 B an element) be halved? Not exactly: fp32 = 24 mantissa bits; two fp16 (hi + lo) hold ~22 bits and still cost 4 B, a bf16 / fp16 stage 8 / 11 bits: the residual add `y = x + f(x)` needs the exact fp32 x (float32 tolerance 1e-5), so x and y stay 4 B each way (8 B an element, 67 MB at [65 536, 128] = 10 us of HBM against the 32 us kernel). No byte saving is available without changing results; the fix is overlap (stage serialisation), not bytes.
* (4) D 384 and the wider `tcgen05.ld` at D 64: not built. D 384's 2 x 48 KB xn halves would need the LayerNorm statistics pass first (an extra x read, 12 K-block loads again) before the expand reads xn halves; the weight stream (72 KB ring) paces the chunk loop, so a `cta_group::2` weight split is the only route and it changes the kernel's whole TMA layout. The D 64 SwiGLU is bounded by its ablation (whole SwiGLU 0.14 us of 1.59 a tile).

### S90 follow-up 12 (no change kept; `cuda:b200_sm100` stays 50): float32 `tg_fused_h16` per-tile timeline and the epilogue-off-the-SwiGLU-warps overlap (`EP2`)

**Timeline** (`hp.py`, the kernel built with `-DPROBE=1`, `%globaltimer` stamps to global memory, median over 740 CTA samples, [262 144, 128] n 2 float32 = 13.8 tiles a CTA, us from kernel entry; `docs/b200/s90_transition_inf/hp1.sh`):

| | tile 0 | tile 1 | tile 2 | tile 3 | tile 4 | tile 5 |
|---|---|---|---|---|---|---|
| LayerNorm done (xn in TMEM, `LN arrive`) | 5.38 | 9.73 | 14.85 | 20.74 | 25.86 | 32.00 |
| MMA starts the tile (`MMA xn tile`) | 5.38 | 10.24 | 17.66 | 24.83 | 32.00 | 39.42 |
| epilogue end (`epi end tile`) | 12.80 | 20.22 | 27.39 | 34.56 | 42.50 | 49.92 |

Tile 0 chain: expand chunks ready 5.63 / 6.14 / 7.17 / 8.19, squeeze chunks 6.66 / 7.68 / 8.96 / 9.73, `yfull` 9.98: 4.6 us of MMA chain. The steady-state period is 7.2-7.4 us a tile; the LayerNorm warps run AHEAD (LN done 3-7 us before the MMA starts the tile at tiles 3-5), so the LayerNorm / x load is NOT what gates the tile (unlike D 256 bf16 before X2).
What gates it: warps 2-9 run the SwiGLU chunks of tile i + 1 only AFTER the y epilogue of tile i, which stages in the h buffers the chunks need (`epi end` 20.22 -> the next tile's chunk work, MMA start 17.66 -> 24.83 follows): SwiGLU(i), epilogue(i), SwiGLU(i + 1) are serial on the same eight warps.

**Built: `EP2`** (D <= 128 inference, `M % 128 == 0`): the y epilogue moves to the LayerNorm warps 10-13 (after the LayerNorm of the next tile), residual x re-read from global memory in 128-B segments (`load_t<32>` on fp32 rows: 32 words = 128 B) before the accumulator wait, y by `store_t<32>` from registers (no staging), the accumulator released (`yempty`, 4 arrivals) after the last TMEM load; the SwiGLU warps skip the epilogue (and `RXT` is off). Correct (error vs fp64 equal to the routed kernel's: rms 3.811e-4 / 3.880e-4 / 4.018e-4 / 3.708e-4 / 3.848e-4 / 4.150e-4; deterministic).
Fine us, routed -> EP2 (`hvf.py`): [16 384, 128] n 2 11.66 -> 14.28, [65 536, 128] 32.27 -> 35.97, [262 144, 128] 102.41 -> 105.13, [589 824, 128] 223.15 -> 221.98, D 64 [65 536] 18.03 -> 19.14, [262 144] 52.51 -> 52.57. No gain: the four LayerNorm warps now carry LayerNorm(i + 1) + epilogue(i) in series (a tile every ~6 us) and the first / last tile's epilogue lost its eight warps. The serialisation moves, it does not vanish.
Diff: `rejected/transition_s90_h16_ep2_epilogue_on_ln_warps.diff`. What it would take: an epilogue split across BOTH groups (half of the 32-column blocks on the SwiGLU warps in their own staging, half on the LayerNorm warps), which needs 16 KB of staging the 224 KB of shared memory do not have at D 128 (weights 96 + 32, h 32, x 64).

**D 384 (`cta_group::2` weight stream / xn halves): not built** in this round; the work is a new TMA layout for the weight ring and the pair-split MMAs of the whole `tfwd_d384.cu` (~500 lines), and the only measured number on it is the 8.6 us / tile bubble (29.9 us a tile against 21.3 us of tensor time).

### S90 follow-up 13 (`cuda:b200_sm100` 52): float32 `tg_fused_h16` D 128 -- the epilogue split over both warp groups (`EP3`); D 384 per-tile timeline

**`EP3`** (`tg_fused_h16.cu`, D 128 inference, `M % 128 == 0`; routed from 98 304 rows by `fused_generic_sm100a.EP3_F32` / `EP3_MIN_ROWS`, cubin `fusedh128e`): the y epilogue of a tile is split over BOTH warp groups and uses no staging, so the h buffers the SwiGLU chunks need are not touched and SwiGLU(i + 1) no longer waits for epilogue(i):
the 32-column K-blocks 0 and 1 of tile i by the SwiGLU warps (half 0 / half 1) right after their chunks, K-blocks 2 and 3 by the LayerNorm warps after the LayerNorm of the next tile. The residual x is re-read from global memory in 128-B row segments before the accumulator wait (`load_t<32>` on fp32 rows), y by `store_t<32>` from registers, every warp releases the accumulator (`yempty`, 12 arrivals) after its last TMEM load (the first version, EP2: all of it on the LayerNorm warps, moved the serialisation and gained nothing).
Splits tried at [262 144, 128] (fine us): SwiGLU warps 1 + 1 blocks / LayerNorm 2: 92.9 (91.2 in the first run), 2 + 1 / 1: 116.5, 1 + 2 / 1: 116.6, 2 + 2 / 0: 100.7, 1 + 0 / 3: 99.2, 0 + 2 / 2: 105.6: the 1 + 1 + 2 split kept. Same arithmetic (`fmaf(acc, s_y, x)`): bit-identical (rms error vs fp64 unchanged, 3.6e-4 - 4.2e-4, deterministic).
Fine us, routed -> EP3 (`hvf.py`, random weights): [16 384, 128] n 2 11.66 -> 12.58 (not routed), [32 768] 18.97 -> 19.92, [65 536] 32.26 -> 32.53, [98 304] 45.00 -> 43.62, [147 456] 60.51 -> 57.44, [262 144] 103.07 -> 91.21, [589 824] 223.24 -> 196.46, [1 048 576] 399.60 -> 353.30.
`measure.accuracy` row 27 float32 eval, every length: cuda YELLOW 1.0001 - 1.0006 x tol, as before (the documented TF32-rounding YELLOW; PyTorch GREEN). `measure.speed` row 27 float32 (us, 50 -> 52), pure SOL% against the TF32 / fp16 peak: L 128 16.16 -> 16.16, 256 36.77 -> 36.67, 384 65.38 -> 63.17 (54.9 %), 512 108.54 -> 98.14 (62.8), 640 163.66 -> 145.38 (66.3), 768 229.36 -> 201.54 (68.9),
1024 395.04 -> 341.79 (72.2), 1536 864.21 -> 759.92 (73.0), 2048 1574.59 -> 1366.66 (72.2). All 181 cases (`sol_v52.tsv`): all-case SOL median 55.6, float32 median 54.5 (53.7), 21 / 77 / 100 cases >= 90 / 70 / 50 %; float32 against the fp16 peak 33.0 % median. 1.5x misses 24 (the run-to-run level at 25 / 22), ladder 0. Version 52 (inference, even).

**D 384 per-tile timeline** (`tfwd_d384.cu` built with `-DPROBE=1` stamps, `pf384.py`, [2 359 296, 384] n 2 = 124.5 tiles a CTA, median over CTAs, us): tile period 26.1 us; expand starts the tile at 24.83 / 51.46 / 77.70, the last expand is issued at 44.29 / 70.66 (19 us of expand + squeeze), the SwiGLU of the last chunk is done 1.3 us later (45.57), the x tile of the next tile LANDS at 48.38 (3.5 us after the last expand), the LayerNorm (12 warps) is done at 51.46 (3.1 us later) and the expand of the next tile starts at once:
**the bubble is 7.2 us = 3.5 us of x load + 3.1 us of LayerNorm behind the single 96 KB xn buffer**, as at D 256 before X2. Tried and measured level or slower: the 12 x boxes from 12 lanes (`XPAR`: expand start 24.83 / 51.46, unchanged), the 12 expand weight boxes of a stage from 6 lanes (`WPAR`: period 27.3 us, slower), the L2 prefetch of the next x (`PF2`, follow-up 9: level to 6 % slower). Diff of the stamps and the two lane variants: `rejected/transition_s90_d384_stamps_xpar_wpar.diff`.
Why the load takes 3.5 us whatever the issue pattern: per tile and CTA the kernel pulls 0.88 MB of weights (half of Wa, Wb, Ws of 3 x 384 x 768 x 2 B) + 96 KB of x through the L2 at 34 GB/s a SM (18 432 tiles x 0.88 MB = 16 GB in 3.6 ms = 4.5 TB/s over the 148 SMs): the weight re-stream of every 128-row tile (a cluster pair already halves it by sharing M = 256) keeps the TMA / L2 path busy, and the x tile queues behind the next chunks' weights. The remaining levers are bytes through the L2: a 4-CTA cluster with the weight boxes multicast to two pairs (halves the weight traffic per CTA again, M = 512 a pass), or 256-row tiles per CTA (TMEM: a 384-column accumulator per tile does not fit twice). The xn-halves variant (LayerNorm statistics from the landed K-blocks as they arrive, normalisation after the last) could overlap only the first of the LayerNorm's three passes (~0.7 us of the 7.2); not built, a 4-CTA multicast cluster is the structural fix and was not built in this round.

### S90 follow-up 14 (`cuda:b200_sm100` 54): D 512 n 2 inference with the LayerNorm folded into the expand GEMM; D 384 x-priority test

**D 512 LayerNorm fold** (`widths/tstat_w.cu` new, `widths/tswiglu_w.cu -DFOLD`, `fused_wide_sm100a.FOLD_LN` / `FOLD_MIN_ROWS` / `FOLD_MAX_ROWS`; new cubins `stat`, `swiglu_fold`, `swiglu_fold_is` in the D 512 `_specs`): the three-kernel chain `transition_ln_w` (x in, xn out: 268 MB + 268 MB at L 512) -> expand GEMM on xn -> squeeze becomes
`transition_ln_stats_w` -> expand GEMM on the RAW x -> squeeze. `tstat_w`: ONE launch, row blocks read x once and write only rstd and c1 = mean rstd (no xn), 32 prep blocks make Wp = bf16(W gamma) of [Wa ; Wb] and per weight row u_n = sum Wp_nk (of the rounded values), v_n = sum beta_k W_nk;
the expand's epilogue computes a_n = rstd (x . Wp_n) - c1 u_n + v_n (two FMAs a unit: this thread's row statistics from global memory, the item's 64 columns' u / v staged in shared memory a warpgroup at a time, prefetched an item ahead, `named_bar_sync` per warpgroup), then silu(a) b as before.
Launch count unchanged (3); the xn buffer is gone. First version (u / v by uniform global loads in the epilogue): n 4 swiglu 1342 -> 1588 us, n 2 405: the staging above brought it level (n 2 L 512: swiglu 393 -> 391 us, n 4 1342 -> 1329).
Kernel timeline (`tl.py`, one flushed replay, us): row 33 L 512 ln 93.7 + swiglu 393.3 + gemm 253.7 (median 700.4) -> stats 68.0 + swiglu 390.7 + gemm 249.8 (679.5); the statistics pass reads x at 3.9 TB/s (60 % of the HBM peak; 41 us would be the peak), 26 us under the LayerNorm pass it replaced.
`measure.accuracy` rows 33 / 34 master eval, every length, folded path active: **24 / 24 GREEN** (error vs fp64 0.0051 against PyTorch 0.0086 at one case). `measure.speed` row 33 master (us, 52 -> 54): L 128 61.26 -> 61.47 (below the 65 536-row threshold: not folded), 256 191.73 -> 186.00, 384 393.02 -> 378.91, 512 706.34 -> 677.76, 640 1105.79 -> 1069.33, 768 1602.30 -> 1598.74 (level: not folded above 524 288 rows).
Row 34 (n 4) measured +/- 3 % with the fold (L 256 317.2 -> 325.4, 512 1275.7 -> 1257.2, 640 2007.0 -> 2075.5, 768 3230.0 -> 3328.8) and stays on the LayerNorm launch. All 181 eval cases (`sol_v54.tsv`): all-case SOL median 56.4 (55.6), master 59.4, >= 90 / 70 / 50 %: 23 / 77 / 100; 1.5x misses 24, ladder 0. Version 54.
Why only 3-5 %, not the 13 % of the LayerNorm pass: the statistics pass still reads all of x (68 us for the 268 MB), the expand's epilogue is longer, and the two prep roles share the launch; the saved bytes are the 268 MB of xn written and the 268 MB read back by the expand's A operand (the A operand now streams x, same bytes), i.e. only the xn WRITE (+ the LayerNorm's own re-read) is removed.

**D 384 x-priority test** (`tfwd_d384.cu`, stamps as follow-up 13; diff not kept: flags `XFIRST` = the weight producer holds a tile's first chunk back until its x boxes are issued, `XEF` = `evict_first` L2 hint on x): expand-start period us at tiles 1-4: base 24.83 / 51.33 / 77.57 / 103.17 (26.1 a tile); XFIRST 24.58 / 51.20 / 77.82 / 103.17 (same), XEF 24.06 / 51.46 / 78.85 / 105.34 (27.0, slower), both 24.06 / 51.20 / 78.21 / 104.70; x landing 48.13 / 48.00 / 48.26 / 47.87 after the last expand at ~44.3: the x load takes ~3.5 us whatever it queues behind.
**The D 384 kernel already is a `cta_group::2` pair** (header of `tfwd_d384.cu`: the two CTAs run their own 128-row tiles in lockstep, the leader issues M = 256 products whose B operand is split by N across the pair, each CTA loads its half of every weight stage), so the 0.88 MB a CTA pulls per tile IS the pair-halved weight stream; the only larger token tile left is a 4-CTA cluster with weight multicast (M = 512 per weight read), not built.
The bursts are synchronised: all 148 SMs cross a tile boundary together, so the 96 KB x loads of all SMs (14 MB) land in the same ~3 us window (4.7 TB/s of L2 / HBM) -- a staggered start of the cluster pairs (a different phase per pair) would spread them; not tried.
### S90 parts 6-7 (2026-11-18): D 64 training forward timeline, D 256 training forward on X2, the L128 cells re-measured, fp32 pair tiles interleaved (`cuda:b200_sm100` 53)

**D 64 n 2 training forward** (`b200cfg/prbf.py`: per-tile `%globaltimer` stamps of CTA 0 written past the end of the saved `c1` tensor): the build ran LayerNorm(i + 1) and the epilogue(i) on the SAME four
warps (`ROLES 0`), tile period 2.1 us (LN done 7.97 -> epilogue seen 10.21 -> y store 11.26 us: the pair is the chain), SwiGLU on two warpgroups. `-DROLES=1` for the training build (`FWD_TRAIN_FLAGS`, n 2 only: the
inference build already had it): period 1.7 us, [589 824 / 147 456, 64] n 2 step 239.7 / 77.8 -> 227.3 / 75.8 us, [65 536] / [16 384] level (47.1, 28.7); n 4 370.7 -> 378.9 (slower, not applied). y bit-identical
to the old build ([16 384 / 65 536 / 589 824]), deterministic. What is left: one SwiGLU warpgroup now paces the tile (0.8 us a chunk, two chunks a tile = the 1.7 us period); a third [a | b] buffer
(TMEM 384 + 64 + 64) would be the next step.

**D 256 training forward on the inference X2 build** (two xn buffers, no x buffer, residual re-read from L2, y from registers; the kernel now takes x / y as two extra arguments, `rstd` / `c1` are the saved statistics;
the xn save store reads the right buffer `XB(i)`): the training call takes the inference cubin from the second tile of a CTA (`X2_TRAIN`); one tile a CTA keeps the old build. Step, graph, L2 flushed (us, old -> new):
n 4 [65 536 / 98 304 / 147 456, 256] 475.1 / 671.7 / 913.4 -> 458.7 / 647.1 / 882.7, n 2 [147 456] 565.2 -> 532.5, [16 384] n 4 145.4 / n 2 96.3 (level, old build). y / dx differ from the old build by one bf16 ulp
(the LN2 summation order, max 3.1e-2 on y at 147 456 rows), deterministic; `measure.accuracy` rows 213 / 214 / 40 master train, every length: 18 / 18 GREEN (0.73-0.89 x tol).

**The L128 cells, three `measure.speed` runs each** (cuda, graph, L2 flushed, one exclusive process, `out/m_r{1,2,3}_train_128.jsonl`): the numbers repeat to 0.1 us (row 217 master 26.51 / 26.50 / 26.50 against the 26.36 line, row 211 master
49.02 / 49.06 / 49.06 against 40.09, row 211 float32 96.16 / 96.13 / 96.13 against 92.58, row 210 32.64 / 32.66 / 32.64 against 29.23): the 1.5x misses at L128 are real, not noise (row 212 now passes: 55.2 against 57.14 after the part 5
merge); row 217 master is 0.14 us (one third of a timer tick) from its line.

**fp32 D 512 `GATE_AB` pair tiles, interleaved in ONE process** (the cubins of all widths loaded first, the dict switched per captured graph; us, with -> without): [262 144, 512] n 2 N 128 5054 -> 4995 (+1.2 %), N 256 4974 -> 4987
(-0.3 %); [65 536] N 128 1280 -> 1296 (-1.3 %), N 256 1282 -> 1300 (-1.4 %): level within the spread, not adopted. The fp16-exact / bf16-split storage of a, b, h, dab was not built (it needs fp16-operand GEMMs with device scales
as the inference `tg_fused_h16` and a GATE_AB reading fp16: a new family, not a storage change; bf16 hi + lo halves are as many bytes as fp32).

**Single-stream rows 217-219, the launch chain** (`tb.py -p`, durations of PDL kernels include their wait): row 217 master [1, 128, 384] n 2 26.6 us = `tsx_ln` 2.6 + `tsx_expand` 4.7 + `tsx_squeeze` 4.4 (forward 11.7) + `tsx_gdx` 11.2 +
`tsx_dw` 4.7; D 384 n 4 32.8 us (3.2 + 4.8 + 6.0 | 15.7 + 5.6); D 768 n 2 38.9 (5.6 + 5.3 + 5.0 | `tg_bwd_small` 18.9); fp32 D 384 34.9 (3.3 + 6.6 + 6.2 | 17.8), fp32 D 768 49.2 (6.8 + 9.7 + 6.2 | 22.4). A one-launch forward replaces the
three forward kernels: the replay floor is 6.14 us for one launch (part 1) plus LayerNorm + expand + squeeze work that is >= 4 us even when overlapped, and its exchange of h through distributed shared memory moves ~184 KB into each CTA
(12 KB from each of 15 CTAs at a 16-CTA cluster) at ~40 B/clk with chunk-major 512 B `st.async` packets = ~2.4 us at 1.9 GHz: ~12-13 us against the 11.7 us chain, i.e. no gain even at the packet rate that works (the bulk-copy rate of 10-12 B/clk
would be 4x worse; part 6 of the inference rounds measured the same skeleton 1.1-1.9x slower). Not built for that reason.

After (m3, all 126, `measure.speed` cuda / triton / pytorch + `solscore.py`): master SOL% 4.4 / 42.8 / 65.6 (min / median / max), float32 3.3 / 24.5 / 55.6; cases >= 90 / 70 / 50 %: 0 / 0 / 34 (27 at the start of S90). Rows 209 6.6-29.5 %, 210 11.5-36.3, 214 33.4-52.6 (was 33.4-51.2), 213 master 25.1-44.5, 033 float32 45.8-55.6 (was 34.5-40.6).
1.5x misses 8 (none slower than the H100), ladder 0.

### S90 part 8 (2026-11-18): software grid barrier measured, the row 217 chain timeline, trigger placements (no change kept; `cuda:b200_sm100` stays 53)

**Software grid barrier microbenchmark** (`b200cfg/barrier.py`: G CTAs x 128 threads resident, one `atom.add.release.gpu` per CTA + `ld.acquire.gpu` spin on one counter, 1000 barriers in one launch, GPU 1 exclusive): 148 CTAs 1.217 us a barrier
(2377 clk), 74 CTAs 1.174, 32 CTAs 1.122, 16 CTAs 1.102 us. Against the PDL kernel boundaries of the chain below (start-to-start 2.0 / 1.9 / 3.5 / 4.5 us) a barrier is ~1.2 us.

**Row 217 master timeline** (nsys node trace of one flushed graph replay, `b200cfg/tl.sh`, median of 17 replays; us from the first kernel's start, [1, 128, 384] n 2): `tsx_ln` 0.00-3.33, `tsx_expand` 2.02-6.98 (start +2.0 after
the LayerNorm's), `tsx_squeeze` 3.90-9.70, `tsx_gdx` 7.39-19.84 (starts 3.5 after the squeeze's start, ends 10.1 after the squeeze's end), `tsx_dw` 11.90-21.54 (ends 1.7 after gdx); the step is 21.5 us from the first start
plus the ~5 us fixed cost of a replay. n 4 [1, 128, 384]: ln 0-4.3, expand 2.8-8.0, squeeze 4.9-12.6, gdx 8.5-26.5, dw 15.0-28.2 (gdx 18.0 us, the tile's [dA | dB] chain). Five kernels, four boundaries: 3.0 us each on average
(start-to-start), which matches the ~3 us per boundary of the TriAttn finding; the critical path is the 12-CTA gdx cluster (a single tile in flight: dy -> gate products -> [dA | dB] over the cluster -> d_xn -> LayerNorm backward).

**Tried, level** (`rejected/transition_s90_sx_trigger_probes.diff`, `SX_X` flags per chain kernel): `griddepcontrol.launch_dependents` right after every `griddepcontrol.wait` in `tsx_squeeze` (the next launch scheduled as early as possible):
[1, 128, 384] n 2 / n 4 26.59 / 32.77 -> 26.56 / 32.77; `tsx_gdx` triggering the weight-gradient kernel as soon as its flag is cleared, before the gate products are done: 26.62 -> 26.62 (n 2 L 128 and L 512), n 4 32.77 level.
The dependent's prologue is not what the boundary costs.

**What a barrier-fused kernel would be worth, and why it was not built**: replacing the three forward boundaries (ln -> expand -> squeeze: 2.0 + 1.9 us of start offsets, ~6.7 us from the first start to the squeeze's end) by two software barriers
(2 x 1.2 us) in one persistent kernel is worth ~2-3 us of the 26.5 (row 217 would pass its 26.36 line); the three kernels would have to become device functions of one kernel (different grids: expand `tiles x h / ch` CTAs of 384 threads, squeeze
`tiles x d / nb x sk` clusters, one LayerNorm launch that also casts the 3.5 MB of fp32 masters), their shared memory unioned, mbarriers re-initialised per phase and TMEM allocated once -- the refactor `tg_bwd_small` did for the fp32 GEMM roles, on three tcgen05 kernels
with cluster-pair issue. Not built in this round; the barrier cost (1.2 us) and the gap (2.0-3.5 us per boundary) are now measured numbers for it.

### S90 follow-up 15 (`cuda:b200_sm100` 56): merge with Transition 53 (training), fold statistics pass at 88 % of HBM, D 384 stagger / weight-traffic tests

**Merge** of `b200-2.5.x` (Transition 53: D 64 n 2 training forward `ROLES=1` in `FWD_TRAIN_FLAGS`; the D 256 training forward on the X2 build with explicit x / y arguments, `X2_TRAIN`) into the branch: `fused_wide_sm100a.py` keeps both (the D 512 fold routing for inference row 33, training's `FWD_TRAIN_FLAGS` / `X2_TRAIN` / `x2_cell` and the trailing `x, out` kernel arguments; `fwdi` and `fwdi1` stay the D 256 X2 / one-tile cubins used by inference AND training, `fwd` the training single-buffer build), `transition.py` version 54 (then 56), the docs both sections.
Checked on the merged tree (`runmerge.sh`, `det.sh`): `measure.accuracy` inference rows 29 / 30 / 33 / 34 master eval, every length: **60 / 60 GREEN**; training rows 213 / 214 / 40 master train: **36 / 36 GREEN**; `tcheckt.py` (cuda vs PyTorch vs fp64 gradients, two steps bitwise): D 256 n 2 [65 536] / [16 384], n 4 [16 384], D 64 n 2 [65 536] **det True** (grad errors 0.49 - 0.90 x the PyTorch statement).

**Fold statistics pass at 88 % of HBM** (`STAT_LPR = 16`, `STAT_BPS = 8` in `fused_wide_sm100a.py`: two rows a warp, four 16-B loads a lane, 8 blocks of 256 threads an SM instead of 32 lanes a row and 4 blocks): `transition_ln_stats_w` row 33 L 512 68.0 -> 47.4 us (268 MB x: 5.7 TB/s), L 256 17.5 -> 21.9 us at 65 536 rows (a launch-bound size, unchanged in the step); step (`tl.py`) 679.5 -> 663.6 us.
`measure.accuracy` row 33 master eval: 12 / 12 GREEN (the summation order of the row statistics changed with the lane count). `measure.speed` row 33 master (us, 54 -> 56): L 256 186.00 -> 190.27, 384 378.91 -> 376.61, 512 677.76 -> 664.46, 640 1069.33 -> 1047.54, 768 (not folded) 1598.74 -> 1675.04 (the run-to-run spread of the long cases, ~5 %). All 181 cases (`sol_v56.tsv`): all-case SOL median 55.6, >= 90 / 70 / 50 % 22 / 77 / 100, 1.5x misses 24, ladder 0.
**Row 34 (n 4) fold**: with the faster statistics pass the statistics saving is larger (76 against the LayerNorm launch's 149 us at L 640) but the expand takes **1437.7 us against 1328.1** (+8 %) although n 2's is level (390.7 / 373.9 against 393.3): the fold's epilogue (row statistics from global memory, the item's u / v staged in shared memory with a warpgroup barrier per item) is slower per item at n 4's 16 chunks a tile; the cause was not isolated. n 4 stays on the LayerNorm launch (`FOLD_N4 = False`).

**D 384 stagger and the weight-traffic test** (graph medians of the module call, L2 flushed, us; `ab384.py` / `stag.sh`, `-DSTAG=<us>` = a start delay of 0 .. 7 x STAG us for the 8 pair phases):

| | [262 144, 384] n 2 | [1 048 576] n 2 | [2 359 296] n 2 | [262 144] n 4 | [1 048 576] n 4 |
|---|---|---|---|---|---|
| base | 383.2 | 1495.0 | 3648.0 | 634.9 | 2845.7 |
| STAG 1 | 380.6 | 1476.6 | 3730.9 | 630.9 | 2851.0 |
| STAG 2 | 385.0 | 1486.9 | 3668.0 | 636.9 | 2589.7 |
| STAG 3 | 393.2 | 1523.2 | 3558.9 | 641.0 | 2801.2 |
| STAG 4 | 399.4 | 1491.5 | 3629.0 | 649.2 | 2803.8 |

Staggering the pairs' phases changes nothing outside the 3 % run-to-run spread (the 2589.7 is a single outlier; STAG 1 / 3 / 4 read 2851 / 2801 / 2804): the 96 KB x loads of all SMs landing together is not what slows the load.
**Weight stream removed** (`TBW=1` timing probe in `tfwd_d384.cu`: expand and Ws stages loaded for the first chunks only, later stages merely signalled; wrong numbers; stamps [2 359 296, 384] n 2): expand-start period 24.83 / 51.33 / 77.31 / 102.78 -> 23.04 / 48.64 / 73.86 / 98.05, i.e. 25.9 -> 25.0 us a tile (-3 %). **The weight stream through the L2 is not what binds the D 384 tile** (the 0.88 MB a CTA per tile at 34 GB/s a SM was my earlier hypothesis; it is wrong): the tile is ~21 us of tensor work plus the ~4 - 7 us x-load + LayerNorm bubble behind the single xn buffer. A 4-CTA weight-multicast cluster would only cut the 3 %; not built. The only remaining lever of the 25 us is the bubble itself (a second xn buffer does not fit: 2 x 96 KB + the 72 KB ring).

### S90 part 9 (2026-11-18): the per-CTA timeline of `tsx_gdx` (row 217 / 218 backward body), st.async exchanges and multi-lane TMA issue built, both level (no behaviour change)

Stamps (`b200cfg/prb_gdx.py`: `-DPROBE=1`, `GSTW(k, warp)` = `%globaltimer` of lane 0 of one warp into the tail of the `flags` tensor, which the host replaces by a larger view so the weight-gradient kernel sees the same flags; 13 points a CTA):
`tsx_gdx` D 384 n 2 [1, 128, 384] (one tile, one 12-CTA cluster), us after the first CTA's entry, median over the 12 CTAs (min - max in brackets where it spreads):

| phase | stamp | what happens in it |
|---|---|---|
| 0.00 | entry | |
| 0.58 | setup done | barrier init, tensor maps, TMEM, the Ws chunk and the dedicated d_xn weight blocks issued (all before `griddepcontrol.wait`) |
| 0.64 | `griddepcontrol.wait` returns | the squeeze's output is complete |
| 2.14 | gate products issued (2.11-2.24) | the 6 dy K-blocks (96 KB a CTA, every CTA loads the whole tile) + the Ws chunk (48 KB): **1.5 us at ~100 GB/s, the per-SM L2 intake** |
| 2.21 | gate products done | tcgen05 time ~0.07 us |
| 3.65 | gate stores done (3.39-3.68) | TMEM loads, SwiGLU backward, `store_t` rows to global `dab` (1.4 us) |
| ~4.93 | cluster barrier released | fence + barrier.cluster release / acquire over 12 CTAs: **1.25 us** after the last CTA's stores |
| 7.52 | last d_xn K-block landed | 12 K-blocks of [dA | dB] (16 KB each, the same K half is loaded by the 6 CTAs of a half) + 12 weight blocks: **2.6 us for 288 KB = 110 GB/s intake** |
| 7.71 | d_xn accumulator complete | |
| 8.51 | partner partial received | TMEM load, stage, one 16 KB DSMEM bulk copy, add (0.8 us) |
| 9.54 | every CTA's row sums received | LayerNorm-backward row sums, 11 DSMEM bulk copies, column partials (1.0 us) |
| 9.95 | `dx` stored (9.89-9.98) | |

So the 12.4 us kernel body of the nsys trace (7.39 -> 19.84) is ~10 us of this chain plus the kernel's tail: five latency / intake-bound phases in a row (dy + Ws intake 1.5, gate arithmetic + stores 1.4, cluster barrier 1.25, d_xn operand intake 2.6, two DSMEM exchanges 1.8),
the tensor work is 0.07 + ~0.1 us. Per-SM intake is the limit of both load phases (96-144 KB a CTA at ~100 GB/s).

Built and measured (both bit-identical to the old kernel on [1, 128 / 512, 384] n 2 and n 4, deterministic; `fx.py` step, us, base -> variant: `26.62 -> 26.62`, `26.62 -> 26.62`, n 4 `32.77 -> 32.77`):
* `-DSTA` (`SX_X["gdx"]`): the partner's partial straight from the registers into its shared memory with `st.async` v4 (no staging, no bulk copy), the row sums by `st.async` v2.f32 from the 128 row-owning threads instead of 11 serial 1 KB bulk copies from one thread: red_full seen 8.51 -> 8.74, rs_full 9.54 -> 9.86 (level to slightly later).
* `-DMLN`: one lane per TMA issue (the 6 dy K-blocks, the first STG [dA | dB] blocks): phases unchanged (dy landed 2.11 -> 2.11-2.18, last d_xn block 7.52 -> 7.58): the issue rate is not the limit, the intake is.
* (part 8) the early `griddepcontrol.launch_dependents` placements: level.

What would still move it (not built): the d_xn phase is intake-bound at KS = 2 (each CTA takes the K half of A = 192 KB + the 96 KB weight half); KS = 4 with 4 column blocks of 96 (16 CTAs, the cluster limit) takes 96 + 72 = 168 KB (~ -1.1 us) but needs a 4-way
partial reduction through DSMEM (+~0.5 us) and a 16-CTA cluster that has to be resident (7 clusters of 12 fit 148 SMs; 9 of 16): ~-0.6 us at best. The gate's dy tile (96 KB a CTA for 64 hidden units) is the same: a K split of the gate needs the same exchange.
`tg_bwd_small` (rows 219, fp32): its phase stamps are in part 3 / 4 above (gate end 7.2 -> barrier 8.2, d_xn CTAs intake-bound to 17.7, LayerNorm 21.9, dgamma 23.3 us) -- the same intake-bound structure.

### S90 follow-up 16 (no change kept; `cuda:b200_sm100` stays 56): the LayerNorm fold in the fused D 384 kernel, and the D 512 n 4 fold epilogue

**D 384 fold built** (`tfwd_d384.cu -DFOLD=1`, inference cubin `fwdi`, `fused_wide_sm100a.FOLD_D384`; diff `rejected/transition_s90_d384_fold_in_kernel.diff`): the A operand is the raw x tile in the xn buffer (no apply pass; the lead thread arrives on `xn_full` the moment the x has landed, so the expand of a tile starts at once), the weights are `Wp = bf16(W gamma)` from one prep launch (`transition_ln_stats_w` with 0 row blocks: 148 blocks, `u`, `v` of every weight row), the two statistics passes over the x tile run in the SwiGLU warps (12 warps) while the first chunk's MMAs run and leave rstd / c1 in shared memory,
and the SwiGLU warps correct the accumulators after the [a | b] hand-back: `a_n = rstd (x . Wp_n) - c1 u_n + v_n`, `b` likewise (u / v of the chunk's 32 units staged per warpgroup in shared memory, prefetched a chunk ahead, one `named_bar_sync` per chunk; float4 broadcast loads in the second version). `measure.accuracy` rows 31 / 32 master eval: **34 / 34 GREEN** (0.42 - 0.59 x tol).
`measure.speed` (us, v56 -> fold; the float4 version): row 31 L 384 227.2 -> 227.2, 512 381.7 -> 382.1, 768 852.0 -> 850.8, 1536 3733.5 -> 3724.8, 2048 6716.2 -> 6644.1 (L 1024 1483.7 -> 1622.5 is a one-off spread); row 32 (n 4) L 384 366.4 -> 388.9, 512 635.5 -> 675.6, 768 1462.6 -> 1532.6, 1024 2873.7 -> 2998.3, 1536 6420.3 -> 6872.9, 2048 11584.3 -> 12306.3 (+5 - 7 %).
**Not adopted**: n 2 level (the prep launch is 5 - 10 us of a 227 - 380 us step and the tile did not get shorter), n 4 worse. The expected 3.1 us of the bubble did not disappear: the statistics passes (2 us) are still behind the x landing and now the SwiGLU of chunk 0 waits for them, and the correction adds 128 shared-memory loads + FMAs a thread to every chunk's SwiGLU, the chain that sets the chunk period (the scalar-load first version: n 2 level, n 4 +6 - 8 %; float4 loads: n 4 +4 - 7 %). The x tile in two K-block halves (the expand starting on the first K-blocks) was not built: the bubble's other half is the 3.5 us x load, which the stagger and the weight-removal probes showed not to be queueing.

**D 512 n 4 fold epilogue with float4 loads** (`tswiglu_w.cu -DFOLD`; diff `rejected/transition_s90_d512_fold_float4_epilogue.diff`): the item's u / v read as four float4 broadcasts a 32-column step instead of 64 float2 loads: n 4 L 640 expand 1437.7 -> **1309.1 us** (below the unfolded kernel's 1328.1), statistics pass 76.0 us (149 us for the LayerNorm launch), gemm 669; kernel sum 2054 against 2162 us. The graph medians (`measure.speed` row 34 master, us, v56 -> n 4 folded): L 256 317.2 -> 329.6 / 330.2, 384 692.0 -> 688.0, 512 1273.6 -> 1251.2 / 1269.6, 640 2064.0 -> 1988.5 / 2123.1: within the run-to-run spread (the two runs of one case differ by up to 6 %), no consistent gain, so n 4 stays unfolded (`FOLD_N4 = False`) and the n 2 path is untouched (same bits; version stays 56). The scalar-load epilogue was the cause of the earlier n 4 slowdown (instruction count: 128 LDS and 128 FMAs per chunk step against ~300 instructions of SwiGLU).

### S90 follow-up 17 (`cuda:b200_sm100` 58): the fused D 384 inference kernel with the LayerNorm folded into the expand through a K16 CORRECTION MMA

The scalar-epilogue fold of follow-up 16 added 128 loads + FMAs a thread to every SwiGLU chunk. The rank-1 terms go into the tensor core instead (`tfwd_d384.cu -DFOLD=1`, inference cubin `fwdi`, `FOLD_D384`, `FOLD_D384_MIN_ROWS = 65536`; prep `tstat_w.cu`):
* **Exact form**: a_n = rstd (x . Wp_n - mean u_n + q v_n), q = 1 / rstd, Wp = bf16(W gamma), u_n = sum Wp_n, v_n = sum beta W_n. The last two terms are ONE extra K16 MMA per chunk accumulated into [a | b]: A columns [-mean_hi, -mean_hi, -mean_lo, q_hi, q_hi, q_lo, 0, 0] (bf16 hi / lo pairs of -mean and q),
  B columns [u_hi, u_lo, u_hi, v_hi, v_lo, v_hi, 0, 0] (the same for u and v): mean u = mean_hi u_hi + mean_hi u_lo + mean_lo u_hi (about 16 mantissa bits; the dropped lo x lo term is 2^-16 relative). The main K-blocks run on the RAW x tile and start the moment the x has landed (the lead thread arrives on `xn_full` at landing); the SwiGLU is `a = rstd acc` (64 multiplies a chunk step instead of 128 loads + FMAs).
* **Operands** (no-swizzle K-major, `desc_none`: 8-row x 16-B core matrices, LBO = distance of the K-adjacent core matrix = the zero plane, SBO = 128 B): the A plane is 128 rows x 16 B written by the statistics thread of each row (together with rstd) after the two statistics passes (the SwiGLU warps run them while the chunk's main K-blocks run;
  the K16 MMA of the tile's first chunk waits `aug_full`, the others do not), the B plane 64 rows x 16 B per chunk (the CTA's 32 units of Wa then Wb; a prep image made by `transition_ln_stats_w` with `aug_on`), bulk-copied (`cp.async.bulk`) into a 2-slot ring by warp 3 lane 0 and forwarded to the leader's barrier by lane 1; a shared 2 KB zero plane is the second K core matrix of both operands. gamma / beta are not staged any more, their 3 KB hold the planes: shared memory 230.4 + 1.0 KB of 232.4.
  The K16 MMA is issued after the last stage's main MMAs and before the `ex_done` commit. The training build (`fwd`) keeps its original code path (`#if FOLD` around the changed region).
* A first version with a one-slot B ring: [262 144, 384] n 2 434 us (the 1 KB plane's load latency sat in front of every chunk's K16); two slots: **366.6 us (kernel 354.8 against 375.4 unfolded)**.
* **Accuracy** (`chk384.py`, random non-zero weights and gamma / beta, real shapes, fp64 reference): n 2 [16 384, 384] fold 3.383e-03 / rms 3.580e-03 against the unfolded kernel's 3.478e-03 / 3.630e-03; with x offset by a mean of 3: 3.853e-03 / 6.192e-03 against 3.869e-03 / 6.220e-03; [65 536] 3.280e-03 / 3.580e-03 against 3.453e-03 / 3.629e-03; n 4 [16 384] 3.533e-03 / 4.816e-03 against 3.434e-03 / 4.891e-03; max |fold - unfolded| 0.031 - 0.0625 (one bf16 ulp of the output).
  `measure.accuracy` rows 31 / 32 master eval: **34 / 34 GREEN** (0.41 - 0.59 x tol); the D 384 training forward (unchanged build): `tcheckt.py` 384 n 2 / n 4 [16 384] det True, gradient errors 0.53 - 0.95 x the PyTorch statement.
* **Speed**, in-process A/B (`fold_ab.py`, graph medians, L2 flushed, fold / unfolded, three alternating rounds each): n 2 [16 384, 384] 1.133, [65 536] 0.967, [262 144] 0.948, [589 824] 0.947, [1 048 576] 0.920, [2 359 296] 0.927; n 4 [16 384] 1.141, [65 536] 1.000, [262 144] 0.908, [589 824] 0.938, [1 048 576] 0.952, [2 359 296] 0.986 (the prep launch, 4.5 us, loses below ~65 536 rows).
  `measure.speed` (us, v56 -> v58), row 31 master: L 128 38.93 -> 39.14, 256 122.88 -> 120.83, 384 227.22 -> 219.14, 512 381.71 -> 366.37, 640 587.55 -> 565.25, 768 851.95 -> 814.91, 1024 1483.74 -> 1433.65, 1536 3733.49 -> 3564.94, 2048 6716.24 -> 6407.41, 2976 14323.63 -> 13773.71, 3968 25714.48 -> 24679.25 (SOL 70 - 72 % -> 72 - 82 %);
  row 32 master: L 128 57.31 -> 57.26, 256 190.42 -> 190.40, 384 366.37 -> 362.37, 512 635.52 -> 630.30, 640 992.90 -> 983.10, 768 1462.62 -> 1438.51, 1024 2873.66 -> 3113.17 (a one-off against the A/B's 0.952), 1536 6420.34 -> 6333.18, 2048 11584.32 -> 11420.03.
  All 181 cases (`sol_v58.tsv`): all-case SOL median 56.4, master 59.9, >= 90 / 70 / 50 % 22 / 79 / 99; 1.5x misses 24, ladder 0.
* **Not built**: the statistics moved off the SwiGLU warps (they run while the first chunk's main K-blocks run and cost the chunk period nothing now that the correction is in the MMA); the K16 correction at D 512 n 4 (the same machinery on `tswiglu_w.cu`; the float4 epilogue there already removed the expand slowdown, the step stayed within noise) and at D 256.

### S90 follow-up 18 (no change kept; `cuda:b200_sm100` stays 58): the K16 correction MMA at D 512 (rows 33 / 34), and why D 256 `tcgen05.ld 16x256b` was not built

**D 512 K16 correction** (`tswiglu_w.cu -DFOLD` rewritten, `tstat_w.cu` writes rstd plus the A plane of every row `[-mean_hi -mean_hi -mean_lo q_hi q_hi q_lo 0 0]` (16 B a row, M x 16 B) and the B plane images of every (chunk, CTA rank) (128 rows x 16 B); diff `rejected/transition_s90_d512_k16_correction.diff`): per item (128-row tile x 128-unit chunk) the producer bulk-copies the item's A plane (the 128 rows, 2 KB contiguous) and B plane (2 KB)
into a 2-slot ring, lane 1 forwards the landing to the leader's barrier, ONE K16 MMA (no-swizzle K-major, `desc_none`, the zero plane as the second K core matrix) is issued after the item's eight K-blocks and before the `acc_full` commit, and the SwiGLU epilogue is `a = rstd acc` (128 multiplies a thread an item; the float4 / float2 u, v staging is gone). Shared memory +10 KB.
Correct at the first run: `chk512.py` (random non-zero weights / gamma / beta, fp64 reference, fold vs unfolded vs fp64, max/max|ref| and rms): n 2 [65 536, 512] 3.396e-03 / 3.540e-03 against 3.576e-03 / 3.587e-03 (x offset by a mean of 3: 3.881e-03 / 6.179e-03 against 3.645e-03 / 6.207e-03), n 4 [65 536] 4.246e-03 / 4.747e-03 against 3.942e-03 / 4.818e-03, n 4 [98 304] mean 3 3.632e-03 / 6.907e-03 against 3.702e-03 / 6.955e-03; max |fold - unfolded| 0.031 - 0.0625.
**Slower than the float4 scalar epilogue it was meant to replace**: kernel times (`tl.py`, us): n 2 L 512 expand 401.9 (scalar float4 fold 376.1, unfolded 393.3), n 4 L 640 expand 1393.4 (float4 fold 1309.1, unfolded 1328.1); in-process A/B (`fold_ab512.py`, graph medians, fold / unfolded, three alternating rounds; FOLD_MAX_ROWS lifted):
n 2 [65 536] 1.032, [147 456] 0.992, [262 144] 0.929, [409 600] 0.976, [589 824] 0.960; n 4 [65 536] 1.069, [147 456] 1.026, [262 144] 1.026, [409 600] 1.019, [589 824] 1.007. The expand GEMM of D 512 is tensor-bound (89 % of its tensor SOL): the extra K16 MMA (33 instead of 32 per item, +3 % of its tensor time) costs more than the epilogue it removes,
and the scalar float4 epilogue already cost ~nothing there (the SwiGLU warps are not the chain at D 512, unlike D 384 where the SwiGLU chunk period is). Not adopted; the D 512 n 2 fold of version 54 / 56 (scalar epilogue) stays, n 4 stays unfolded.

**D 256 `tcgen05.ld 16x256b` for the epilogue: not built.** The 16x256b fragment gives a lane 2 consecutive columns of two rows per 8-column block (thread t: rows t / 4 and t / 4 + 8 of a 16-row half, columns 2 (t % 4), + 1), repeated x4 over 8-column blocks: a row-major y store from it is 4-byte granular (a quad of lanes writes 16 B of one row, 8 rows a warp instruction) or needs the same quad exchanges `store_t` already does to build 128-B segments; a bf16 pair is 4 B, so the transposition cost moves rather than disappears.
Float32 D 256 with `EP3` (rows 29 / 33 float32): the single-tile-a-CTA lengths (L 128 / 256) that sit lowest do not benefit (an epilogue split needs >= 2 tiles a CTA), and the multi-tile lengths are already at 84 - 96 % of the TF32 peak: not built.

### S90 follow-up 19 (`cuda:b200_sm100` 60): the D 512 squeeze GEMM -- what it is, what binds it, the per-part accumulator

**Which kernel**: `transition_gemm_nd_sm100` (`widths/tgemm_nd.cu`, custom, NOT cuBLAS): out[M][512] = h[M][K] Ws^T + x, K = n x 512 (1024 / 2048), cta_group::2 pairs (M = 256), N = 512 as two parts of 256 (each CTA holds half of every part's B rows),
a 3-stage ring of 48 KB stages (A 16 KB + B 32 KB per K-block and CTA), an epilogue on two warpgroups that take alternate 64-column blocks (residual x block by TMA into a 2-slot ring, add, TMA store). The whole-row accumulator (512 columns) fills tensor memory: **single-buffered**.
It is the kernel of rows 33 / 34 only (D 384 / 256 / 128 have the squeeze inside their fused kernels; D 768 uses it per half).
**Probes** (`-DTBW / TBA / TBE`: B boxes / A boxes loaded for the first ring fill only, epilogue releases the accumulator right after its TMEM loads and stores nothing; `sqprobe.py`, kernel durations of one flushed replay, us; diff `rejected/transition_s90_d512_squeeze_probes_tbw_tba_tbe.diff`), original kernel:

| row 33 L 512 (M 262 144, K 1024; tensor SOL 175) | squeeze us |
|---|---|
| as is | 244.3 |
| no B (Ws) refills | 237.3 |
| no A (h) loads | 224.4 |
| no epilogue | **176.0** (the tensor SOL) |
| no A, no B | 219.2 |
| nothing | 159.4 |

and row 34 L 640 (n 4, K 2048): 651.6 / 627.4 / 572.1 / 542.9 / 550.1 / 457.4. **The epilogue is the bound: with one accumulator the next tile's MMAs wait until the epilogue has read all 512 columns (4 blocks a warpgroup, each with a TMA store + slot refill round trip: ~5 us a tile of 17.9)**; the B stream costs 3 %, the A stream 8 %.
A register epilogue (`RX`: residual into registers by 128-B row segments, `store_t`, no x ring, 4 ring stages) was built and is slower (272.7 us at row 33 L 512; diff `rejected/transition_s90_d512_squeeze_rx_epilogue.diff`).
**Built: `PARTS`** (`tgemm_nd.cu -DPARTS=1`, cubin `squeezep`, D 512 n 2 inference): an item is (tile, N part) with its own 256-column accumulator buffer (it % 2): the K loop runs per part, the epilogue of part p drains buffer b while the MMAs of the next part / tile fill the other; the h tile is read once per part; a stage is 32 KB (A 16 + B 16), five stages.
Probes of the new kernel (row 33 L 512): as is 234.2, no epilogue 180.0, no A loads 178.8, no B 222.7, no A no B 166.7: the epilogue no longer is the whole gap, the doubled h stream (now ~64 B / clk / SM of L2 ingest against ~47 before) is the other half. Kernel 244 -> 235 us (n 2), n 4 level (651.6 -> 684.7): kept for n 2 only (`SQ_PARTS`, `h == 2 d`).
Bit-identical arithmetic (same K order); `measure.accuracy` rows 33 / 34 master eval: **24 / 24 GREEN**. `measure.speed` row 33 master (us, v58 -> v60): L 128 61.39 -> 59.36, 256 190.46 -> 186.38, 384 376.88 -> 368.66, 512 665.41 -> 656.21, 640 1047.95 -> 1026.90, 768 1600.32 -> 1580.30 (SOL 74 - 80 %); in the first run of the new kernel 649 / 650 at L 512.
Row 34 stays on the single-accumulator kernel (L 256 317.2 -> 319.3, 640 2002.7 -> 2055.1 with PARTS). All 181 cases (`sol_v60.tsv`): all-case SOL median 55.6, master 59.4, >= 90 / 70 / 50 % 21 / 79 / 100, 1.5x misses 24, ladder 0.
**What is left of the squeeze**: 235 us against 175 us of tensor time (the A stream through the L2 at two reads a tile and the epilogue's remaining share); a 4-CTA cluster multicasting B (half the B bytes a CTA) or an M = 256-a-CTA tile with a split accumulator would be the next lever, not built.

### S90 follow-up 20 (no change kept; `cuda:b200_sm100` stays 60): the D 512 squeeze is memory- and tensor-bound at once; the register epilogue with an L2 prefetch is still slower

**Register epilogue (`RX`) again, now with the residual tile (128 KB, contiguous rows) prefetched into the L2 a tile ahead by `cp.async.bulk.prefetch.L2` and a 4th ring stage in the freed room** (diff `rejected/transition_s90_d512_squeeze_rx_l2prefetch.diff`; in-process A/B `sqab.py`, step / squeeze us, three alternating rounds, outputs bit-identical):

| case | original | PARTS | RX + L2 prefetch |
|---|---|---|---|
| n 2 [262 144, 512] | 702.5 / 242.0 | 697.4 / 228.9 | 743.5 / 274.9 |
| n 2 [409 600] | 1120.3 / 391.1 | 1123.3 / 363.2 | 1219.6 / 431.4 |
| n 4 [262 144] | 1278.0 / 418.9 | 1336.4 / 434.7 | 1442.8 / 447.5 |
| n 4 [409 600] | 2168.9 / 693.2 | 2078.3 / 654.2 | 2264.6 / 765.4 |
| n 4 [98 304] | 473.1 / 180.4 | 475.2 / 181.2 | 489.5 / 196.0 |

The row-segment transposition (`load_t` / `store_t`: ~100 shuffles a 64-column block) costs more than the TMA store + refill round trip it replaces at this tile size (D 256's 12 us tile hid it; this one, 17 us, is ~13 us of MMA + the epilogue in series). The prefetch does not change that. PARTS kept for n 2 only (above: step -0.7 / +0.3 % here, -2 % in the measurer runs of follow-up 19; n 4 mixed -4.2 % / +4.6 % / +0.4 %: the run-to-run spread of these 1 - 3 ms steps is ~5 %).
**Why the epilogue probes overstate the gap**: the squeeze moves h in (537 MB at row 33 L 512), x in (268 MB) and y out (268 MB) = 1 073 MB = **165 us at the 6.5 TB/s HBM copy peak, against 175 us of tensor time**: the stage is balanced between the two roofs, so no schedule reaches 175 us; 190 us (88 % of the HBM peak) is the realistic floor. The `TBE` probe (no epilogue = no x loads / y stores = 25 % of the bytes and the serial wait) and `TBA` (no h = 50 % of the bytes)
therefore showed the byte-removal as much as the overlap: with `TBW` (B = Ws refills, 0.5 MB a tile from the L2, not HBM) the kernel only gained 3 %. At 235 us the kernel runs at ~79 % of that floor; the remaining levers (ring-of-columns accumulator slices keeping h read once; a 4-CTA cluster multicasting B) can recover at most ~10 % of the squeeze (~3 % of the step) and were not built.

## S90 follow-up 21: fused D 512 pair kernel (column-split pair, h by DSMEM) -- built, correct, REJECTED (1.6 - 1.9x slower)

Design (prototype kept in `docs/b200/s90_transition_inf/tfwd_d512_proto.cu`, host route in `fwd512_route.diff`; not in the tree, no cubin in the prebuild plan, no version bump): a 2-CTA cluster on ONE 128-row tile,
CTA q owns output columns 256 q.., expand of its 32 units per 64-unit chunk (M128 N64 K512, cta_group::1), SwiGLU, h pushed into both CTAs' h slots with chunk-major `st.async` (complete_tx on the peer barrier), squeeze M128 N256 K64 per chunk into 256 TMEM columns, residual + TMA store epilogue.

Budget (TMEM 512 columns; smem 227 KB = 232 448 B usable):

| buffer | TMEM columns | smem KB |
|---|---|---|
| squeeze accumulator (256 out columns) | 256 | |
| expand a / b accumulators, 2 slots x 64 | 128 | |
| spare | 128 | |
| xn tile 128 x 512 bf16 (A operand, whole K) | | 128 |
| expand B ring 3 x 8 KB (32 a-rows + 32 b-rows x K64) | | 24 |
| Ws stage (256 rows x K64), single | | 32 |
| h slots 2 x 16 KB | | 32 |
| barriers | | 1 |
| total | 384 | 217 (4 stages 225) |

A second Ws stage (+32), a second x tile (+128) or a deeper ring (+8 per stage beyond 4) do not fit. The pair serves 128 tokens per weight read (the current pair kernel 256), so the weight L2 traffic per token is 2x: 1.5 MB per CTA per tile, floor 17.7 us / tile at the measured 85 GB/s per SM (`wstream`, 17.7 / 20.3 / 31.6 / 50.7 us at 0 / 200 / 400 / 800 clk spin per 16 KB stage).

Result (correct: row 33 shape, random non-zero weights, fused vs generic chain max 3.1e-2, error vs fp64 3.58e-3 of max, base 3.40e-3, M 256 / 2048 / 65 536): graph medians, us, fused / current (L2 flushed, alternating in one process):
M 65 536 330 / 186; 147 456 725 / 371-375; 262 144 (L 512) 1241 / 648-657; 409 600 (L 640) 1928-2100 / 1022-1026; 589 824 2742-3072 / 1569-1610: 1.77 - 2.1x slower.
Ablations at M 262 144 (kernel + LN launch): no expand B stream (TBB) 1180, no Ws stream (TBWS) 1198, both 1170; no SwiGLU math 1155; no x tile load (TBX) 1163; no epilogue (TBEPI) 1098; everything off 1018 - 1050. Four B stages instead of 3: 1184 (-5 %).
Blocker: no single stage or stream is the bound (each removal moves < 8 %); the stripped kernel is still ~1.6x the current chain. What remains is the structure itself: the N = 64 expand (A re-read from smem for 64 B columns, 6 KB smem operand per 131 072 FMA, smem-operand bound, ~0.67 of the tensor peak), the in-order MMA warp (squeeze j - 1 waits for the SwiGLU + DSMEM exchange of chunk j - 1 inside the expand of chunk j, one expand time of slack, no third h slot fits) and the 2x weight traffic. Ideal estimate of the fused step was 580 - 670 us against 650 now, i.e. at best level; measured 1170. Not adopted.

## S90 consolidation (`cuda:b200_sm100` 60, Transition inference, 181 eval cases): re-score, blockers per case group

**Re-score at the tip (60, nothing changed since `sol_v60.tsv`; follow-up 21 adopted nothing).** All 181 cases: pure SOL% min / median / max 4.0 / 55.6 / 105.2, >= 90 / 70 / 50 %: 21 / 79 / 100; with the 6.1 us replay floor added 41.4 / 65.8 / 106.0 and 25 / 86 / 159; master 139 cases median 59.4 (+floor 65.1), float32 42 cases median 54.6 against TF32 and 33.0 against the fp16 peak.
**1.5x rule against the H100 `cuda` tokens** (`h100ratio.py`, the v60 `measure.speed` records vs `judge/tokens/h100_sm90/Transition.jsonl`, `h100ratio_v60.tsv`): 181 compared, 15 exempt (H100 < 20 us), 166 eligible, **24 not >= 1.5x faster** (ratio > 0.667), 0 slower than the H100, worst 0.877, geomean 0.517. The ladder (registry harness, not re-run here, not mine to touch) was 0 at v58 / v60.
All 24 misses are cases of 16 - 40 us (the 20 us exemption margin): rows 25 L 256, 26 L 256, 27 / 28 / 29 L 128, 35 L 1536 / 2048, 36 L 768 - 3968, 37 L 256 - 2048, 38 L 128 / 256.

### Transition inference: blockers per case group

Numbers: pure SOL% (against the tensor / HBM / MUFU peak; with the 6.1 us replay floor in brackets), v60. "Attempted" lists what was built or ablated and the result; all of it is in the sections above.

| group | cases | SOL% (+floor) | measured blocker (evidence) | attempted, result |
|---|---|---|---|---|
| rows 25 / 26 / 38 L 128 (D 64, one tile a CTA, 12.2 - 18.3 us) | 3 | 5.4 - 16.8 (50 - 59) | the replay floor (6.1 us, one kernel) + the tile's own latency ~8 us: x in 2 us, LayerNorm, the 0.77 us a chunk handoff chain (no single resource), epilogue (follow-ups 10 / 11, S90 parts 5 - 6) | 20 / 24-warp SW2 + LN2 ring (26 % slower), OB2, NXS 8, NHS 4, RX residual re-read (+18 %), packed f16x2 SwiGLU, ROLES=1 (adopted, training), one-launch skeleton (12.3 us, above the chain) -- all level or slower |
| D 64 / D 128 narrow, steady state: rows 25 / 26 L >= 256, 27 / 28 master, 38 L >= 256, 39 | 51 | 5 - 87 (50 - 88) | per-tile ring: D 64 1.6 us a tile = handoff chain 0.77 us a chunk; D 128 3.9 us a tile = x-stage lifetime + LN / SwiGLU / epilogue serial (follow-ups 4, 10) | 3 x stages + 2-slot ring (adopted, 46), LNH, RX (slower), SW2 (slower), EP3 for fp32 (below) |
| rows 35 / 36 / 37 master (D 384 / 768, single stream, 1 - 4 tiles, 14 - 29 us) | 29 | 4.0 - 35.0 (41 - 58) | the 3-launch PDL chain: LayerNorm 2.9 us, expand 1.7, completion hop 1.2, squeeze 2.3 - 3.0 = 14.1 us against t_sol 0.6 - 1.2 us and a 6.1 / 8.2 us floor; a one-launch data-movement skeleton (zero math) measures 12.3 us (CS 8 / 16), the xn gather ~8 500 cycles (follow-ups 3, 4, 10) | skeleton in 4 exchange variants (bulk copy 24.6 / 18.4, st.async 16.5, partial reduce-scatter 14.3, whole x per CTA 12.3 us), SQ split, K-split d_xn: the one-launch kernel would land at 15 - 16 us against 14.1: not built |
| rows 35 / 37 float32 | 18 | 4.9 - 53.4 (49 - 79) | the same chain with TF32 / fp16-operand products; row 37 L >= 1024 is tensor-bound (fp16 peak 33 %) | as above; float32 MMA ablation (follow-up 11) |
| D 256 L 128 - 256 (rows 29 / 30, master) | 4 | 33.0 - 71.3 (57 - 78) | one tile a CTA pair, no overlap possible: **timeline now taken (`pf256.py`, M 16 384, us)**: setup done 1.5, x landed 3.3, LayerNorm done 5.4, expand c0 starts 5.6, tensor chain 5.6 -> 15.6 (10 us = the tile's tensor time at ~10 TF/SM), epilogue 15.6 -> 17.9; n 4 (M 16 384) 5.6 -> 24.6 chain, end 26.9. Fixed head 5.6 us (setup 1.5 + x 1.8 + LN 2.1) + tail 2.3 us around 10 us of tensor time | X2 / PF2 / XPAR / WPAR / LN2 / `fwdi1` for the single tile (follow-ups 7, 8, 11: adopted for >= 2 tiles; X3 roles slower). Not built: LN stats inside the kernel with the fold (K16) for D 256 (the stat pass of the fold is a launch, +3 us at one tile, ceiling gain <= 2 us of 24.9) |
| D 384 L 128 - 256 (rows 31 / 32, master) | 4 | 45.8 - 77.6 (61 - 81) | tensor-bound plus head / tail: row 31 L 128 40.3 us = tensor 21.5 (128 of 148 SMs busy) + floor 6.1 + x load / LN ~5 + epilogue ~2.5 (estimate); the steady-state tile has the 7.2 us bubble (3.5 us x load + 3.1 us LN, follow-up 13) | X prefetch PF2, XPAR / WPAR, x-priority, stagger (level / slower), K16 LN fold (adopted, 58, from 65 536 rows: -3 - 5 %); **first-tile timeline NOT taken**: the stamps of `tfwd_d384.cu` live in `rejected/transition_s90_d384_stamps_xpar_wpar.diff`, not in the tree; no attempt at one tile a CTA beyond the fold threshold |
| D 384 steady-state bubble (rows 31 / 32 L >= 384) | 12 | 72 - 94 | 7.2 us of 26.1 us per tile (follow-up 13), x load queued behind the weight stream (4.5 TB/s of L2) | PF2, XPAR, WPAR, XFIRST / XEF, K16 fold; not built: 4-CTA cluster with B multicast, staggered pair phases |
| D 512 n 4 (row 34) | 6 | 71.3 - 86.1 (78 - 87), all >= 70 | squeeze balanced between HBM (165 us) and tensor (175 us) at L 512; fold costs the n 4 expand 1437 vs 1328 us (follow-ups 14, 16, 19, 20) | LayerNorm fold (+- 3 %, not applied), PARTS squeeze (n 4: 651 -> 685, not applied), RX register epilogue (slower), K16 fold (18), fused column-split pair kernel (21: 1.6 - 1.9x slower, measured budget + ablation) |
| D 512 n 2 (row 33 master) | 6 | 55.3 - 80.1 (66 - 82) | squeeze 235 us vs tensor 175 us: the A stream + the epilogue share; expand 391 us | fold (adopted, 54), PARTS (adopted, 60), fused pair kernel (rejected, 21) |
| float32 D 128 (row 27) | 9 | 23.7 - 72.6 (59 - 74); 33 % of the fp16 peak | **one tile per CTA (L 128, M 16 384): timeline taken (`pfh16.py`, kernel with `-DPROBE=1`, us from the first CTA)**: setup 0.5, first weight box 1.0, x staged + LN stats done 4.1, xn ready 4.9, expand c0 5.1, chunk chain (4 chunks of 64 units) 5.1 -> 9.0 (3.9 us for 2.5 us of fp16 tensor time), epilogue tail to 10.2 = 10.2 us a tile + the 6.1 us floor = the 16.3 us measured. Ablations at the same shape: no LayerNorm (TBLN 2) 8.45 us (LN costs 1.8 us), no epilogue (TBEPI 1) 9.0, everything off + weights loaded once (TBLN 2 + TBEPI 1 + TBW) 7.2 us: **the floor of this structure is ~7 us = the x HBM load (64 KB a CTA, 128 CTAs = 8 MB, ~2.3 us: 1.3 us at HBM rate + latency) + setup 1 + the dependent 4-chunk chain**. At 7.8 tiles a CTA (M 147 456) the first tile ends 13 us in a 52 us kernel (the rest is the steady chain) | EP3 (adopted, 52: -3 - 13 % from 98 304 rows; at one tile a CTA it is slower: 11.66 -> 12.58, not routed), EP2, MMA ablation (the MMA is 12 - 18 % of the kernel), 4-split of the epilogue, LN12 (level); not attempted now beyond timeline + ablation: no change to remove the 2.3 us x load (only bytes, x must stay fp32) or to overlap the 4-chunk chain at one tile |
| float32 D 256 (row 29) | 9 | 54.0 - 96.0 (75 - 97) | L 128: 28.5 us = floor 6.1 + 22.4: x staged 4.4 - 5.1, LN stats 6.1, xn stored 7.9, expand c0 9.0 -> last squeeze 20.0 -> epilogue end 22.5 (the Ws first box is issued at 9.2 us, not binding); no LayerNorm 17.9, no epilogue 18.4, all off 14.6: ~8 us of head + tail, 10 us tensor chain. L >= 384: 86 - 96 % | EP2 / EP3 (D 128 only), X2-style buffers do not apply (one tile) |
| float32 D 512 (row 33 fp32) | 6 | 75.4 - 105.2 | at the TF32 peak (above 90 % from L 384 up in the fp16 path); not below 70 % | -- |

**Where nothing (or only a timeline) was attempted**: D 384 L 128 - 256 first tile (no timeline: stamps not in the tree), float32 D 128 / D 256 at one tile a CTA (timeline + ablation taken now; no kernel change, ceiling 16.3 -> ~13 us = +20 % pure SOL at most, and every term of it is a measured serial latency: x HBM load 2.3 us, LayerNorm 1.8, chunk chain 3.9, epilogue 1.3), D 256 bf16 one tile a CTA beyond `fwdi1` (timeline now: 5.6 us head + 2.3 us tail around a 10 us tensor chain). The one-tile cases are bound by data-dependent latency in series (x in -> LN -> chain -> epilogue) that no buffer or role change of the kernel removes; the only structural cut left is overlapping the tail of tile 0 with its own head by splitting the 128-row tile into two 64-row halves on 256 CTAs (needs 2 CTAs per SM, i.e. half the smem and TMEM per tile: not possible with the current 200+ KB footprint).

### S90 part 10 (2026-11-18): `tsx_gdx` split over several clusters a tile (CC = 4 clusters of 12, last-cluster reduction): built, correct, deterministic, SLOWER (rejected; diff `rejected/transition_s90_gdx_multicluster_cc.diff`)

Design (`tsx_gdx.cu -DCC=<c>`, `fused_wide_sm100a.SX_GDX_CCS`): the tile's hidden units split over CC clusters (n 2: 192 units, 3 gate CTAs a cluster); every cluster is self-contained (its gate chunks -> cluster barrier -> d_xn over its own K slice, [dA | dB] of its units only:
per CTA 72 KB of operands instead of 288 KB), the partner pair-sum as before, then clusters 0 .. CC - 2 store their owned-columns partial (128 rows x 16 columns fp32 a thread) to a workspace and bump a per-(tile, rank) counter (`atom.add.release.gpu`),
the last cluster's CTA of the same rank spins on that counter (`ld.acquire.gpu`), sums the partials in cluster order (own last: bitwise deterministic, fixed order) and goes on with the row-sum exchange and the LayerNorm backward unchanged; the weight-gradient kernel waits for one flag a cluster
(`use_flags` = CC words a tile). All 48 CTAs co-resident (4 clusters of 12 <= the 7 resident), counters zero between launches (the master resets what it consumed), `griddepcontrol` and flag protocol unchanged.
Correct and deterministic on [1, 128, 384] n 2 / n 4 (y, dW bit-identical, dx / dgamma one rounding different: the d_xn sum order).

Step, graph, L2 flushed: n 2 [1, 128, 384] 26.62 -> **32.26 us**, n 4 32.77 -> 34.82 us. Per-CTA %globaltimer timeline of the CC = 4 kernel (us after entry, median): the intended gains happen -- d_xn operands landed 7.52 -> 5.54 (-2.0), partner partial 8.51 -> 6.46 (-2.1) --
but the cross-cluster step costs more than they save: partial stored 6.7-7.6 -> counter bumped 7.9-8.3 (the releasing increment drains the stores: 1.3 us) -> the last cluster's wait done 8.4-8.7 -> **its loads of the partials (12 x 16 B a thread, L2-resident lines written by other SMs) take 3.8-4.3 us**
(`__ldcg`, `ld.relaxed.gpu`: the same or worse; stamps 19 / 16 on all 12 master CTAs) -> row sums received 13.5 (was 9.54). A gpu-scope hand-off of 192 KB per cluster through L2 on this card is ~5 us (release 1.3 + first read 4), against the ~4 us the split saves: no net gain at any CC.
What this leaves for the single-tile chain: stay inside one cluster (DSMEM exchange: 0.8 + 1.0 us) or take a different decomposition that needs no cross-cluster data (more rows, not more K slices). Also done in this part: `tsx_gdx` stamps (`GSTW`, part 9) and the barrier microbenchmark (part 8) are the tools for it.

### S90 part 11 (2026-11-18): starting the d_xn operand stream before the whole-cluster barrier (`tsx_gdx`): two variants, level and +4 us (rejected; diff `rejected/transition_s90_gdx_rdy_per_block_signal.diff`)

The d_xn stream (A = [dA | dB] K-blocks from global `dab`, W weight blocks) starts after the cluster barrier (4.93 us in part 9's timeline, 1.25 us after the last gate store). Which part can start earlier: the 8 first W blocks already do (NDED stages before
`griddepcontrol.wait`, NGR stages after the gate products); the 12 A blocks (192 KB, the bulk of the 2.6 us) are the gate CTAs' own output, so they can only be loaded after the CTA that wrote them has finished.
* `-DNOREL` (relaxed `barrier.cluster.arrive` after an explicit `fence.acq_rel.gpu` instead of the release arrive): [1, 128, 384] n 2 / n 4 / [1, 512, 384] n 2 26.62 / 32.77 / 26.62 -> 26.62 / 32.77 / 26.62.
* `-DRDY` (per-source readiness: each gate CTA, after its stores, `fence.acq_rel.gpu` + a named barrier over the gate warps + 12 `mbarrier.arrive.release.cluster` to the barrier of that CTA in every CTA; the consumer loads A block kb as soon as `rdy[source]` is complete,
  the whole-cluster barrier kept for the flag): correct (bit-identical to the old kernel, deterministic) but **26.62 -> 30.72 us (n 2), 32.77 -> 38.94 (n 4), [1, 512] 26.62 -> 32.77**: twelve remote release-arrives per CTA cost far more than the hardware cluster barrier they replace.
Together with part 10 (gpu-scope hand-off through L2 ~5 us) and the DSMEM st.async / multi-lane variants (part 9): the cluster barrier is the cheap synchronisation primitive on this card; every attempt to replace it by finer-grained signalling or to move data between clusters cost more.
A 16-CTA cluster would not shorten the A stream either: A bytes a CTA depend on the K split (2 halves = 192 KB), not on the column blocks; only W shrinks (288 -> 264 KB, ~0.2 us) -- not built.

## S90 part 12: TMA op count (the TriMul box-size rule applied to the Transition inference streams) -- built for D 384, D 256, fp32 D 64 / 128 / 256, bit-identical, LEVEL, rejected (`cuda:b200_sm100` stays 60; merged tip with TriMul 72)

Lens (TriMul microbenchmark): a TMA op costs ~590 - 740 clk a SM whatever its size, 16 KB boxes ~28 B/clk, 32 KB ~53, 64 KB ~89, 96 KB ~92: merge boxes into 32 - 96 KB ops. Streams of the kernels (per CTA, before): D 384 x tile 12 boxes of 8 KB (96 KB); expand weights 6 boxes of 4 KB a stage (24 KB), Ws 3 x 8 KB a chunk; D 256 x tile 8 x 8 KB issued one lane each (XPAR), weight chunk 4 x 8 KB, Ws 16 KB; fp32 `tg_fused_h16` x tile XKB (2 / 4 / 8) x 16 KB, one lane each, weights 8 KB boxes. tgemm_nd stages are 32 - 48 KB already.
Built (diff `rejected/transition_s90_part12_tma_box_merging.diff`; the `tmap` of the extension got an `ab` mode: a 4-D map over a stacked [Wa; Wb] tensor):
* **D 384 `X3D`**: the x tile as ONE 96 KB 3-D box (the `kb_box` map: element, row, K-block), 12 ops -> 1.
* **D 384 `W4D`** (fold path, Wp stacked): the expand weights of a stage as ONE 24 KB 4-D box (K-block element, row, a / b, K-block), 6 ops -> 1 a stage.
* **D 256 `X3D` / `W3D`**: x tile one 64 KB box, the weight chunk of a CTA one 32 KB 3-D box (cubins `fwdi3`, `fwdi13`).
* **fp32 `tg_fused_h16` `X3D`**: x tile one box of XKB x 16 KB for D 64 / 128 / 256 (cubins `fusedh{d}x`, `fusedh128ex`).
All outputs bit-identical to the routed kernels (max |diff| 0 on every shape). Graph medians, us, L2 flushed, alternating 3 times in one process, new / old (`x3d_ab.py`, `d256_ab.py`, `f32_ab.py`):

| case | new / old (us) | ratio |
|---|---|---|
| D 384 n 2 [16 384] (X3D only) 38.9 / 38.9; [65 536] 122.8 / 122.9 (X3D + W4D 123.0 / 120.9); [147 456] 221.2 / 221.1; [262 144] 368.6 / 368.6 (X3D + W4D 367.9 / 366.6); [589 824] 815 / 829 | | 0.97 - 1.02 |
| D 384 n 4 [16 384] 58.1 / 59.3; [65 536] 192.7 / 192.5; [147 456] 364.5 / 362.6; [262 144] 630.7 / 632.9 | | 0.95 - 1.02 |
| D 256 n 2 [16 384] 24.5 / 24.5; [65 536] 59.4 / 59.4; [147 456] 102.4 / 100.4; [262 144] 161.8 / 163.8; [589 824] 356.2 / 362.5 | | 0.99 - 1.00 |
| D 256 n 4 [16 384] 34.8 / 34.8; [65 536] 94.2 / 94.2; [147 456] 167.9 / 168.0; [262 144] 284.7 / 284.7 | | 1.00 |
| fp32 D 128 n 2 [16 384] 16.4 / 16.4; [65 536] 36.9 / 36.9; [147 456] 63.5 / 63.5; [262 144] 98.3 / 98.3; fp32 D 256 [16 384] 28.7 / 28.7, [65 536] 80.5 / 81.9; fp32 D 64 [16 384] 12.3 / 12.3 | | 0.98 - 1.01 |

Level everywhere (the run-to-run level of the long cases is 3 - 5 %; the 2 us timer step hides the short ones: every short case is identical to the tick). **The op count of these streams is not what binds them**: the producers issue ahead of the consumers through rings that never run empty from op latency, the D 384 / D 256 x load that lands late (3.5 us, follow-up 13) waits for the HBM burst of all SMs crossing a tile boundary together (148 x 96 KB = 14 MB = 2.2 us at the HBM rate plus latency) and for the weight stream through the L2, not for 12 op slots, and a single 96 KB op moves the same bytes through the same L2 / HBM path. (The TriMul k1w weight stream is different: it is the whole critical path of that kernel at L128 - 512.) Not built: D 128 / D 64 bf16 weight boxes (4 KB boxes of the ring; same conclusion expected, the D 64 chunk chain is 0.77 us a chunk with no single resource, follow-up 10), tgemm_nd (32 - 48 KB stages already), tsx_* (single-stream chain, launch-bound).

### S90 part 12 (2026-11-18): large TMA boxes in `tsx_gdx` (the TMA microbenchmark rule): level (rejected; diff `rejected/transition_s90_gdx_bigbox_pfdy.diff`)

The rule from the inference TMA microbenchmark (a TMA op costs ~590-740 clk per SM whatever its size; 16 KB boxes cap at ~28 B/clk, 48 KB ~75) predicts that the gate phase of `tsx_gdx` (6 dy boxes of 16 KB + 6 Ws boxes of 8 KB = 12 ops) is op-count bound.
Built (`-DBIGBOX`, `fused_wide_sm100a.GDX_BB`, n 2): the dy tile as TWO 48 KB 3-D boxes (`kb_box` 3: box 64 x 128 x 3 K-blocks, the existing `tmap` option) and the gate's Ws chunk as two 24 KB 2-D boxes (192 rows): 4 ops instead of 12, the MMA warp waits per
3-K-block group. Bit-identical to the old kernel and deterministic ([1, 128 / 512, 384] n 2); **step 26.62 -> 26.62 us** (both lengths), and the per-CTA timeline (`prb_gdx.py`) does not move: gate products issued 2.14 -> 2.27 us (dy landed after
~1.7 us for 96 KB in either form). The gate phase is the cold read of the dy tile (the replay flushes L2: one HBM fetch shared by the 12 CTAs, ~1.5-2 us latency + the Ws chunk), not the op count; the earlier multi-lane issue (`-DMLN`, part 9) was level for the same reason.
`-DPFDY` (L2 prefetch of the dy tile by the setup thread before `griddepcontrol.wait`, 2 x 48 KB `cp.async.bulk.prefetch.tensor.3d`): 26.62 -> 26.62 (n 2 L 128), 26.69 (L 512). The d_xn stream (24 ops, 288 KB) has the same structure (the A tile is the gate CTAs' own output, written
~1 us before it is read, L2-resident) and was not restructured into 48 KB boxes: the dy result above says op count is not the limit there either, and the stage layout (A and W interleaved per stage) would have to change to make the boxes contiguous.

### S90 part 13 (2026-11-18): the backward gate launched early with an L2 prefetch of dy (training chain): mean -0.3 us, measurer level (rejected; diff `rejected/transition_s90_sq_early_trigger_gdx_dy_prefetch.diff`)

`tsx_squeeze.cu -DETRIG` (training chain only: new cubins `sx_sqt*`) issues `griddepcontrol.launch_dependents` right after each of its waits, so `tsx_gdx`'s 12-CTA cluster is resident and runs its setup while the squeeze finishes; `tsx_gdx.cu -DPFDY` prefetches the cold dy tile (6 x
16 KB `cp.async.bulk.prefetch.tensor.2d.L2`) before its `griddepcontrol.wait`. Either alone is level; together (`fx.py`, graph, L2 flushed, mean of 300 replays): [1, 128, 384] n 2 26.58 -> 26.28 us, n 4 32.78 -> 32.23, [1, 512, 384] n 2 26.80 -> 26.74; the median stays on its
2.05-us bucket (26.62 / 32.77). Adopted as defaults for a `measure.speed` run (cuda, graph, one exclusive process a length, rows 217 / 218 every length, against the same tree without them): row 217 master L 128 / 256 / 384 / 512 / 640 / 768 26.50 / 26.53 / 26.50 / 26.50 / 28.54 / 28.67 -> 26.50 / 26.50 / 26.54 / 26.54 / 28.54 / 28.64,
row 218 32.64 / 32.67 / 32.67 / 34.72 / 36.74 / 36.74 -> 32.64 / 32.64 / 32.64 / 34.72 / 36.74 / 36.77: unchanged (bit-identical gradients, deterministic, accuracy rows 217 / 218 GREEN 0.71-0.90 x tol, the uninitialised-memory test clean), so not adopted; row 217 L 128 stays 0.14 us from its line.

### S90 part 14 (2026-11-18): what the chain's first kernel (`tsx_ln`) costs: the weight cast, not the LayerNorm (probes only; diff `rejected/transition_s90_ln_cast_ablation_probes.diff`; no change kept, `cuda:b200_sm100` stays 60)

Before building the LayerNorm fold into `tsx_expand`, the first launch was ablated (fx.py step, graph, L2 flushed, [1, 128, 384] n 2 / n 4, mean and median; probe flags `SX_X["ln"]`):

| tsx_ln variant | n 2 mean / median | n 4 mean / median |
|---|---|---|
| base | 26.6 / 26.62 | 32.7 / 32.77 |
| no LayerNorm rows (`-DNOLN`, wrong results) | 26.65 / 26.62 | 32.82 / 32.77 |
| no weight cast (`-DNOCAST`) | 24.68 / 24.58 | 30.75 / 30.72 |
| neither | 24.64 / 24.58 | 30.77 / 30.72 |
| cast Wa + Wb only (`-DCASTN=2`) | 25.67 / 26.37 | 31.38 / 30.75 |
| cast Wa only (`-DCASTN=1`) | 25.17 / 24.61 | 31.99 / 32.48 |
| 8-element groups a thread 1 / 4 / 8 (`CASTU`; base 2) | 26.63 / 26.66 / 28.69 | 32.83 / 34.26 / 34.72 |

So the LayerNorm rows are free (they overlap the cast blocks; the LayerNorm fold would remove nothing from the chain), and the fp32 -> bf16 master cast inside that launch costs **~2.0 us (0.7 us a matrix)** on the critical path: the expand's weight loads wait for the whole launch (`wpre = 0`). The cast is volume-bound (~1.7 TB/s of cold reads + writes; more or fewer
elements a thread does not help). Moving a matrix's cast elsewhere does not pay either: the squeeze loads its Ws K-blocks BEFORE its `griddepcontrol.wait` (they were meant to be parameters or an earlier launch's cast), so a cast of Ws inside the expand launch would have to be waited for first (~0.7 us of cold loads on the squeeze's critical path, the saving).
What would remove the 2 us is an expand that converts the fp32 masters on their way into shared memory (TMA of the fp32 chunk, bf16 in the swizzled layout before the first MMA, the chunk's bf16 forms written for the later kernels by tile 0's CTAs and Ws by extra blocks): a rewrite of the expand's weight pipeline and the squeeze's prologue, not built. Row 217 / 218 / 219 stay at 26.50 / 32.64 / 38.78 us (L 128).

### S90 part 15 (2026-11-18): the fp32 master cast moved into the expand (`tsx_expand.cu -DCASTF`): built, bit-identical, level (rejected; diff `rejected/transition_s90_cast_in_expand_castf.diff`)

Built for the bf16 D 384 small-M training chain (rows 217 / 218): the LayerNorm launch casts only Ws (`tsx_ln.cu -DCASTN=1`, new `sx_ln_c1`) and issues bulk L2 prefetches of the fp32 Wa / Wb (`cp.async.bulk.prefetch.L2`, one per cast block); the expand (`-DCASTF=1`, cubins `sx_expand{32,64}_cf`)
converts Wa_j / Wb_j itself: its warps 4-11 (256 threads, before `griddepcontrol.wait`: the masters are parameters) load the chunk's fp32 rows (all K-blocks in flight where the registers allow), pack bf16 (`cvt.rn.bf16x2`, the cast's rounding) into the 128-B-swizzled operand
K-blocks of the ring (`fence.proxy.async` + one arrival per thread on the stage barrier, which now counts 1 + 256), and tile 0's CTAs also store the bf16 forms to global memory for the gate / d_xn / dW kernels. Gradients and outputs bit-identical to the old chain ([1, 128 / 512, 384] n 2, [1, 128, 384] n 4), deterministic.
Step (fx.py, graph, L2 flushed, cast in the expand against in the LayerNorm launch, interleaved): [1, 128, 384] n 2 26.62 / 26.59 against 26.62 / 26.59, n 4 32.77 against 32.77, [1, 512, 384] n 2 26.62 against 26.62 (means within 0.1-0.3 us either way). nsys node trace (n 2 L 128): the LayerNorm launch is still 3.1 us (3.3 before), the expand now starts 1.15 us after
it (2.0 before) but runs 6.6 us (5.0): the chain from the first start to the last end does not shorten (22.5 us against 21.5 in the trace). The weights' cold fetch (2.4 MB of fp32 for Wa / Wb, then conversion) is the same latency wherever it is placed; an L2 prefetch from the first kernel does not make it earlier than the LayerNorm's own cold x load and the 2 us the ablation (part 14) attributes to the cast
is the fetch + write of the weights on this card, not a separable launch. Which chains cast per step: the small-M chains of D 384 (rows 217 / 218; `tsx_ln` cast blocks), D 768 bf16 / fp32 (`fused_small_sm100a.chain(cast=)` / `SB_LN_CAST`, row 219 and the fp32 rows) and the wide / generic paths' `tg_prep` / `_bf16_forms` (one launch per forward);
the cast is needed once per optimizer step per parameter (the forms are saved for the backward within the step), it cannot be shared across layers (each layer's weights differ) and not across steps (the optimizer changes them): the per-step cost is the one measured here.


## N1 pass (2026-11-19, agent n1-transition, `cuda:b200_sm100` 62, inference): LayerNorm folded into the small-M expand, cached fold preps, the pair squeeze hole, the dead pad

Tools (all in `n1cfg/` of the tree; measurement only, nothing routed): `stampchain.py` (per-CTA `%globaltimer` stamps of tsx_ln / tsx_expand / tsx_squeeze written to GLOBAL memory in a CUDA-graph replay, L2 flushed; the
stamp macros `TSX_STAMP_ARG` / `TSX_ST` of `tsx_store.cuh` are live again behind `-DTSX_STAMP=1`, no-ops otherwise; `fused_wide_sm100a.SX_X` / `SX_ST` carry the flags and buffers), `span.py` (device span of the kernels of one
replay from CUPTI: the timer's 2.048 us tick hides sub-tick gains, the span does not), `foldv.py` / `f384.py` / `f512.py` / `ov.py` / `sweep.py` (in-process A/B, alternating 3 rounds, graph medians + span + accuracy against fp64),
`scan.py` (kernel inventory of one replay of every inference case), `prof.py`.

### Blockers of the D 384 n 2 small-M chain (row 35 L 1536: 12 tiles, 144 expand CTAs, graph 16.38 us), measured with stamps (us from the first LayerNorm CTA's entry, median over CTAs)

| phase | base chain (3 launches) | folded expand + pair squeeze (2 launches) |
|---|---|---|
| LayerNorm: x landed + statistics / xn stored | 1.28 / 1.28 | -- |
| expand: `griddepcontrol.wait` returns (completion + flush hop after the LayerNorm) | 2.56 (1.3 after the last xn store) | 0.77 (nothing to wait for but the prologue) |
| expand: first / last K-block landed | 3.33 / 4.35 | 1.79 / 3.07 |
| expand: MMA done / SwiGLU epilogue end | 4.61 / 5.38 | 3.58 / 4.35 |
| squeeze CTA entry (an SM frees only when the expand CTA exits: 230 KB of shared memory) / `griddepcontrol.wait` returns | 6.40 / 6.91 (hop 1.5 after the expand epilogue) | 5.38 / 5.89 |
| squeeze: first / last A (h) K-block landed | 7.42 / 9.98 (192 KB a CTA, unsplit) | 6.66 / 7.42 (96 KB a CTA, K halves on a CTA pair) |
| squeeze: end | 10.24 | 8.7 - 8.96 (partial exchange 0.77 + store 0.5) |

What bounds it: two completion hops (1.3 + 1.5 us, one per launch boundary), the per-SM operand intake of each stage (expand: x 96 KB + weights 96 KB a CTA in 2.3 us after the wait = ~85 GB/s a SM; squeeze 192 KB 2.5 us) and
the one-tile latency chain inside a CTA. The intake is not an op-count limit: the x tile as one 96 / 48 KB 3-D box and the weight chunk as one 48 KB box each (diff `rejected/transition_n1_fold_expand_merged_x_w_boxes.diff`, flags `X3D` /
`W3D`) measured 11.17 / 11.36 / 11.79 / 11.97 us of device span (x 3 / x 6 / + weights 3 / + weights 6) against 11.02 for one box a K-block: the MMAs can no longer start as the K-blocks land. A graph of two kernel nodes also
carries ~0.7 us less overhead than three (span 11.02 -> graph 14.37, against 12.18 -> 16.38).

### Shipped (all measured in one process, alternating, L2 flushed; accuracy basis below)

1. **LayerNorm folded into the small-M expand** (`tsx_expand.cu -DFOLD`, `fused_wide_sm100a._sx_forward(fold=)`, `SX_FOLD`, `SX_FOLD_MIN_TILES = {2: 4, 4: 5}`): the expand takes the RAW x as its A operand, Wp = bf16(W gamma) as B and one extra K16 correction MMA
   (-mean u + q v, the follow-up 17 form) accumulates into [a | b]; the row statistics come from the x tile in shared memory as its K-blocks land (shifted single pass, shift = the row's first element, the two K-block halves on warps 4-7 / 8-11,
   exchanged through shared memory; the first half writes the K16 A plane + rstd), the SwiGLU is `a = rstd acc`. Wp and the u / v correction rows are a weight form (`_sx_fold_forms`, torch ops, `_capture.param_forms`): made once per weights.
   The chain is expand + squeeze, two launches. Device span (us) unfolded -> folded, [1, M, 384] n 2: M 384 9.15 -> 8.58, 512 9.50 -> 9.18, 640 9.79 -> 9.22, 768 9.98 -> 9.52, 896 11.78 -> 11.12, 1024 11.82 -> 11.14, 1280 12.00 -> 11.26, 1536 12.48 -> 11.65;
   M 128 / 256 level (8.42 -> 8.38 / 8.58 -> 8.66, not taken); n 4: M 512 12.06 -> 11.65, 768 12.58 -> 12.32 (M 128 / 256 level).
2. **The 64-column CTA-pair squeeze at 7 - 12 tiles of n 2** (`_sx_sq(save=False)`, `SX_SQ_PAIR_INFER`): the old rule returned (32, 1) there (3 / 6 tiles are caught by the first branch; 7 - 12 tiles had never been measured: the span jumped 9.98 -> 11.78 us from 6 to 7 tiles).
   Span unfolded: 896 11.90 -> 11.17 (pair squeeze alone), 1024 11.92 -> 11.09, 1280 11.89 -> 11.63, 1536 12.13 -> 11.62; with the fold 10.46 / 10.50 / 10.70 / 10.98 (-1.4 to -1.5 us against the base). Training keeps `(32, 1)`.
3. **The fused D 384 inference kernel's fold prep cached** (`_f384_ok`, `_fwd_f384_launch`, `FOLD_D384_CACHED_MIN_ROWS = {2: 16384, 4: 32768}`): `tstat_w.cu` ran a 4.5 us prep launch (Wp, u, v, the correction images) plus a zero fill in front of EVERY call, so the fold only paid from
   65 536 rows. As a weight form it pays from 16 384 rows of n 2 and 32 768 of n 4. Graph us, unfolded -> cached: n 2 16 384 41.0 -> 38.9, 24 576 67.5 -> 63.5, 32 768 69.6 -> 67.6, 65 536 124.9 -> 116.7, 98 304 176.1 -> 166.0, 147 456 227.6 -> 217.1; n 4 32 768 106.5 -> 102.4,
   65 536 192.5 -> 185, 98 304 276.5 -> 272.4 (n 4 16 384 level: not taken). Rows 31 / 32 measure.speed (min of 3 runs, us): row 31 L 128 40.74 -> 38.82, L 256 120.64 -> 116.54; row 32 L 768 / 1024 -4.5 / -3.3 %.
4b. **The same for D 512 n 4** (`FOLD_D512_N4_CACHED`): span unfolded -> cached fold [M, 512] n 4: 16 384 rows 88.09 -> 87.23, 32 768 163.0 -> 161.3, 65 536 314.2 -> 307.0, 147 456 680.1 -> 667.5, 262 144 1261 -> 1246, 409 600 1996 -> 1959 (-1.0 - -2.3 %; the per-call prep fold of S90 was +20 us at 16 384 rows and the reason n 4 stayed on the LayerNorm launch). Official accuracy rows 33 / 34: 12 / 12 GREEN (row 34 err 0.0043 -> 0.0047, tol 0.008).
4. **The D 512 n 2 inference fold's prep cached** (`_f512_ok`, `_fwd_f512_launch`, `FOLD_D512_CACHED_MIN_ROWS = 16384`, up to `FOLD_MAX_ROWS`): the statistics launch has no prep blocks any more (they were 32 blocks of the same launch; the per-call prep fold was +7.6 us at 16 384 rows).
   Device span unfolded -> cached fold [M, 512] n 2: 16 384 55.7 -> 53.6, 32 768 97.4 -> 94.7, 65 536 182.6 -> 174.9, 147 456 382.1 -> 363.4, 262 144 686.6 -> 655.1. Row 33 measure.speed (min of 3): L 128 59.33 -> 57.34, L 256 186.35 -> 177.94.
5. **`route_sm100a`: the zero-row pad of a ragged row count only where a tuned build runs.** `_padded(x)` ran before the small-M chains (which take a ragged row count themselves) and before the generic ranges: 2 kernels (~4 us in a graph) for nothing.
   Row 36 L 2976 (23.25 tiles): measure.speed 26.46 -> 23.09 / 24.4 us (min of 3 runs 24.38 against 26.42). The kernel inventory (`scan.py`) of all 179 inference cases now shows no kernel outside the transition_* and GEMM-library names.

Accuracy basis: (a) `measure.accuracy` (the official measurer, cuda, master eval, random non-zero module weights, real shapes, every length the harness runs) rows 31 / 32 / 35 / 36 / 37 / 33 / 34: 46 (rows 31 / 32 / 35 / 36 / 37) + 12 (rows 33 / 34) cases GREEN, none YELLOW / RED, err 0.0030 - 0.0054 against tol 0.008 (the PyTorch reference's own error 0.0043 - 0.0084);
3 large rows-31 / 32 cases are skipped by the harness (memory). (b) `foldv.py` / `f384.py` / `f512.py`: random non-zero weights, gamma 1 + 0.2 randn, beta 0.2 randn, x ~ N(0, 1.5^2) (also offset by a mean of 3 / 20 and a first element of 50 - 3000 to stress the shifted statistics), first 4096 rows against fp64 on the master weights,
[M, 384] n 2 / n 4 for M 128 - 1536 (chain) and 16 384 - 147 456 (fused), [M, 512] n 2 16 384 - 262 144: max error / max |ref| folded 2.4e-3 - 3.4e-3 against 2.4e-3 - 3.3e-3 unfolded, rms equal to 4 digits (3.28e-3 both; offset 3: 5.99e-3 / 6.00e-3; spike 3000: 3.80e-2 both);
max |folded - unfolded| 0.0156 - 0.0625 (one bf16 ulp of the output). Determinism: every case replayed once more in the same process, max |diff| 0 (inference only; training untouched: `_sx_forward(fold=None)` is the old code, `_sx_sq` keeps (32, 1) for `save=True`).
Prebuild: `_prebuild --arch sm100 --check`: the four new routed cubins (`d384n2 / d384n4 sx_expand32f / sx_expand64f`) are in the plan from `fused_wide_sm100a._specs`; the D 512 cached fold and the fused D 384 fold use cubins that were already planned (`stat`, `swiglu_fold*`, `fwdi`).
`tsx_ln.cu` / `tsx_expand.cu` / `tsx_store.cuh` changed (stamp arguments, FOLD): every `tsx_*` cubin is re-keyed (SASS of the unchanged builds not compared: macros expand to nothing by default); non-sm100 sources untouched.

### Result against the H100 1.5x rule and the tick

`h100ratio.py` on three full `measure.speed` runs per tree (inference rows 25 - 40, 181 cases, one exclusive process each): misses 24 / 24 / 25 (base) -> 23 / 23 / 23 (new), none slower than the H100. The base tree and the new tree are separate processes and a run-to-run /
process-to-process difference of one 2.048 us tick is common at the tick edge (the same code gave 14.1 and 16.2 us at row 37 L 128 in two processes: the device span of that chain varies 9.95 - 10.6 us with the allocator history and the tick edge is at ~10.45), so only the in-process A/B
tables above are evidence for a change; the measure.speed pairs are consistent with them where the code changed (row 31 / 32 / 33 small L, row 36 L 640 / 768 -2 us, row 36 L 2976, row 35 L 128 / 256 -2 us) and show the same tick luck where it did not (row 37 L 128 / 384 / 512, +- 2 us in both directions).
Still above 0.667 after this pass (new tree, run 1): rows 25 / 26 / 27 / 28 / 38 L 128 - 256 (one-tile chains of D 64 / 128: not touched), 29 float32 L 128 (0.674), 35 float32 L 1024 / master L 1536 / 2048, 36 L 768 - 3968, 37 L 256 - 2048. Row 36 L 2976 / 3968 sit at 0.682 (24.4 / 26.5 against 23.9 / 25.9 needed).

### Built and rejected / not built (this pass)

* **Merged TMA boxes in the folded expand** (x tile as one 96 / 48 KB 3-D box, weight chunk as one 48 KB box each): slower (above), diff `rejected/transition_n1_fold_expand_merged_x_w_boxes.diff`.
* **The fold at 1 - 2 tiles** and **n 4 at 1 - 2 tiles**: level or slightly slower (span +0.08 / +0.16 us): not taken. The fold needs the whole x tile resident (NKB <= STG): the D 384 n 4 chains of 8 - 12 tiles use a 128-unit chunk whose ring has 4 stages, the persistent pair expand (13 - 31 tiles, D 768) and the D 768 pair chains
  have no fold: not built (a decoupled x buffer + weight ring would be needed; expected ~1.3 us at 8 - 12 tiles of n 4, i.e. one case, row 36 L 1536, across the 1.5x line).
* A measured hole not on a requirement length: D 384 n 4 at 7 tiles (L 896) takes the wide chain's two waves (CH 64: 168 CTAs): span 16.93 us against 12.51 (6 tiles) and 14.19 (8 tiles); `fused_small_sm100a.INFER_CELLS[(384, 4)] = (8, 31)` would need to start at 7.
* Persistent expand (`PEX`) at 7 - 12 tiles of D 768 / D 384 n 4: level (+-0.3 us) against the chain, the table stays.
* One-tile D 64 / 128 / 256 and fp32 one-tile cases, training rows 217 - 219: NOT re-measured or changed in this pass (the S90 timelines stand).

### N1: the D 384 fused kernel (rows 31 / 32) with the stamps back in (`n1cfg/pf384f.py`: the S90 `-DPROBE` stamps re-applied by hand to a scratch copy of `tfwd_d384.cu` for the K16-fold build, cached forms; NOT kept in the tree), us, median over CTAs, from the earliest tile-0 expand start

| [2 359 296, 384] n 2 (124.5 tiles a CTA) | tile 0 | tile 1 | tile 2 | tile 3 | tile 4 |
|---|---|---|---|---|---|
| MMA expand starts | 0.51 | 24.06 | 49.92 | 75.26 | 100.61 |
| last expand issued | 19.71 | 46.08 | 71.94 | 97.28 | 122.11 |
| x tile (A operand) landed | 0.26 | 23.81 | 49.66 | 75.26 | 100.35 |
| last SwiGLU chunk done | 21.25 | 47.36 | 73.22 | 98.56 | 123.39 |
| squeeze accumulator ready (`out_full`) | 25.86 | 51.46 | 77.31 | 102.40 | 127.23 |
| epilogue end | 31.74 | 57.34 | 82.94 | 108.03 | 132.86 |

Period 25.4 - 25.9 us. With the LayerNorm fold the bubble of follow-up 13 (3.5 us x load + 3.1 us LayerNorm) is only the x load: the next tile's x lands 4.1 us after the last expand issue and 0.25 us before its first expand starts. Releasing the x tile per
K-block (the last chunk frees K-blocks 0 - 2 half a chunk earlier) would overlap at most ~0.4 us of it (1.5 % of the period): not built. One tile a CTA ([16 384, 384]: 128 tiles on 64 pairs; expand starts 0.51, last expand 20.48, out_full 22.78, epilogue end 27.65): the epilogue of the last tile (4.9 us from the accumulator to the
last store) is exposed, 18 % of that case's kernel; not changed (rows 31 / 32 L 128 only). The L2 / TMA intake stays the bound (follow-up 13).

### N1: extra accuracy / memory checks of the shipped paths (`n1cfg/edge.py`, `n1cfg/poison.py`, inference, cuda, fp32 master weights under bf16 autocast, random non-zero weights, gamma 1 + 0.3 randn, beta 0.3 randn)

* `edge.py`: fold (chain, fused D 384 cached, D 512 cached) against the unfolded chains and fp64 (first 2048 rows) on [384 n 2 / 1280 n 2 / 640 n 4 / 16 384 n 2 / 32 768 n 4 / D 512 16 384 n 2] x x-scale {1e-3, 1, 40} x mean offset {0, 25} x LayerNorm eps {1e-5, 1e-2} = 72 cases:
  relative rms error folded / unfolded 0.84 - 1.009 (the fold is the more accurate one wherever eps matters or the scale is small: no xn bf16 rounding), max error 2.3e-3 - 6.1e-3 against 2.3e-3 - 7.6e-3 unfolded, every replay bit-identical to the next (rerun 0).
* `poison.py`: the same calls on clean memory, on allocator blocks refilled with 0xFF (eager) and inside a captured graph whose pool was poisoned first (1 GiB filled with 0xFF and freed inside the capture): 8 shapes (3 chain, 3 cached fold, ragged 2976 n 4, D 768) bitwise equal in all three.

### N1: D 64 inference forward with two SwiGLU warp groups (`tfwd_d64.cu -DROLES=2`, 640 threads: epilogue 4-7, LayerNorm 8-11, SwiGLU of every chunk split by units on 12-15 / 16-19): built, bit-identical, level (rejected; `rejected/transition_n1_d64_roles2_two_swiglu_groups.diff`)

Reason: the S90 table says the D 64 tile period (1.6 - 1.7 us = 0.77 us a chunk) has "no single resource"; the SwiGLU warp group of ROLES 1 is busy ~0.8 us a chunk, so halving it by splitting the 64 units of a chunk over two warp groups (each writes its half of the chunk's h to TMEM, `ab_empty` / `h_full` count 16) was
the one untested split. The register budget has to balance exactly inside the CTA (first build deadlocked with `setmaxnreg` 40 / 112 at 640 threads, bounded run: 96 launch registers a thread need dec 32 / inc 112). Device span (us, L2 flushed, ROLES 1 -> ROLES 2, in one process, `d64r.py`): n 2 [16 384 / 65 536 / 147 456 / 589 824 / 2 359 296 / 4 194 304, 64]
6.83 -> 6.78, 11.71 -> 11.52, 17.92 -> 17.74, 54.57 -> 53.77, 189.33 -> 187.15, 324.25 -> 320.72 (-1.1 % at most); n 4 (ROLES 0 / 1 / 2) [589 824] 82.29 / 90.05 / 84.67, [2 359 296] 291.26 / 321.65 / 304.53 (the shipped ROLES 0 stays best). So the SwiGLU is not what paces the D 64 tile: the dependent chain
expand MMA -> `ex_done` (cross-CTA commit) -> TMEM loads -> SwiGLU -> TMEM store -> `h_full` -> squeeze MMA -> `sq_done` is latency, not throughput. Outputs bit-identical to ROLES 1 (max |diff| 0), deterministic.

### N1: is the D 384 fused tile period L2-weight-stream bound? (timing-only ablation, wrong results, stamps as above, [2 359 296, 384] n 2, expand-start period us over tiles 0 - 4)

Follow-up 13 named the weight re-stream through the L2 (0.88 MB a CTA a tile, 4.5 TB/s) as the reason the x load takes 3.5 us and listed a 4-CTA cluster with weight multicast (half the weight bytes a CTA) as the remaining lever, not built. Before building it the premise was ablated:
`-DABW` loads only Wa of each expand stage (half of the expand weight bytes: 576 -> 288 KB a tile, the leader's `expect_tx` reduced to match), `-DABS` drops the third Ws box (8 of 24 KB a chunk). Expand starts at tile 1 / 2 / 3 / 4:
base 24.58 / 50.18 / 75.78 / 100.61 (period 25.4), ABW 22.53 / 47.87 / 72.45 / 96.26 (24.6, -3.2 %), ABS 24.06 / 49.79 / 75.52 / 100.86 (25.6, level), ABW + ABS 22.53 / 47.36 / 71.94 / 96.00 (24.5, -3.5 %). Halving the expand weight bytes buys 3 % of the period: the L2 stream is a minor term, a multicast
cluster (at best that ablation, before its handshakes, which cost 2 - 6x on every earlier try, HANDOFF section 5) is not worth building. The period is the dependent chain of the tile (expand MMAs 22 us after the first one, the last chunk's SwiGLU + squeeze + the x load of the next tile behind the single x buffer), not a bandwidth term.

### N1: final tree (runs 4 - 5 of the full inference `measure.speed`, 181 cases, one exclusive process each, `h100ratio.py` / `sol_inf.py`)

1.5x misses against the H100 `cuda` tokens: 21 / 21 (base: 24 / 24 / 25 in three runs), none slower than the H100 (worst 0.870, geomean 0.516). Gone in both runs against all three base runs: row 35 float32 L 1024, row 36 L 2976 (the dead pad: real), row 37 master L 128 (a tick: the same code measured 14.1 and 16.2 us in different processes); nothing new.
Remaining (new tree): rows 25 / 26 L 256, 27 float32 L 128, 28 L 128, 29 float32 L 128 (0.719 this run, 0.674 before: tick), 35 L 1536 / 2048, 36 L 768 / 1024 / 1536 / 2048 / 3968, 37 L 256 / 384 / 512 / 640 / 768 / 1024 / 2048, 38 L 128 / 256 (rows 37 and the chains of 16 - 31 tiles sit 1 - 3 ticks above the line, the one-tile D 64 / 128 chains 4 - 5).
Pure SOL (peaks.json, 6.1 us floor column not used): all-case median 57.3 / 57.3 (base 56.4 / 56.3 / 56.4), >= 90 / 70 / 50 %: 21 / 80 / 100 and 22 / 79 / 101 (base 20 / 80 / 100, 22 / 80 / 100, 21 / 80 / 101): the gain is real only where the code changed (rows 31 - 34 and 35 / 36 small-M chains, 1 - 5 %), well under the case-to-case noise in the counts.
Training rows 210 - 214 / 217 - 219 (L 128 / 384, master and float32): equal to the base to 0.1 us in separate processes (row 213 float32 L 384 +2.4 % and row 219 float32 L 128 -3.7 % are the only differences beyond that: run-to-run of those cases, no training code changed).

### N1: the folded expand for the 128-unit chunk (`tsx_expand.cu -DFOLD` with `XD`: the whole x tile resident with a barrier a K-block, the weights in a 3-stage ring of their own) and the cells of `fused_small`'s chains (`route_sm100a.FOLD_FIRST`, `fused_wide_sm100a.fold_first`, `FOLD_PAIR_SQ`)

The first fold needed the whole x tile AND the whole weight K loop in one ring (`NKB <= STG`), i.e. 64-unit chunks. The cells whose CTAs only fit one wave with 128-unit chunks (n 4 at 7 - 12 tiles, n 2 at 13 - 16 tiles) went to `fused_small_sm100a`'s chain (LayerNorm launch + expand + CTA-pair squeeze) or, at n 2 13 - 15 tiles
and n 4 7 tiles, to the two-wave 64-unit chain (+2 - 3 us holes). `XD` keeps the 96 KB x tile resident (its K-blocks land on `xfull[kb]`, the statistics warps and the MMA warp wait there) and gives the weights a 3-stage ring (`STG` from the budget; the zero plane grows to the B operand's 256 rows = 4 KB).
The folded chain's squeeze is `fused_small`'s pair squeeze where that table has one (`FOLD_PAIR_SQ`, `BF16_PAIR_SQ`: n 4 8 - 12 tiles), else the 64-column pair `tsx_squeeze` (the version without it was slower than fused_small's chain: n 4 1024 13.60 -> 13.95 us).
Device span (us, in-process, 3 alternating rounds, base chains -> fold + pair squeeze; `foldv.py o,n`): n 4 [M, 384] M 640 12.14 -> 11.92, 768 12.53 -> 12.40, 896 (7 tiles) 16.00 -> 13.81, 1024 13.63 -> 13.01, 1152 14.22 -> 13.18, 1280 14.13 -> 13.26, 1408 14.51 -> 13.54, 1536 14.90 -> 13.78;
n 2 M 1664 15.92 -> 12.80, 1792 15.94 -> 13.02, 1920 16.00 -> 13.12, 2048 (16 tiles, fused_small chain with the 128-column pair squeeze) 13.68 -> 12.38. Graph medians: n 4 896 20.48 -> 18.43, 1536 20.51 -> 18.46, n 2 1664 - 1920 20.48 -> 18.43, 2048 18.43 -> 16.38 (n 4 1024 - 1408 level at 18.43 / 18.46: the span gain does not cross the tick).
`measure.speed` (min of 4 new runs against min of 3 base runs, master eval, us): row 35 L 1536 16.35 -> 14.78 (means 16.90 -> 15.84), L 2048 18.21 -> 16.19 (18.26 -> 17.22); row 36 L 640 / 768 18.19 / 18.18 -> 16.16 / 16.16, L 1536 20.22 -> 18.21 (20.26 -> 19.24), L 2976 26.42 -> 22.56 (26.45 -> 23.32).
Accuracy: `measure.accuracy` rows 35 / 36 / 37 master eval, every length: 29 / 29 GREEN; `edge.py` (x scale 1e-3 / 1 / 40, mean offset 0 / 25, eps 1e-5 / 1e-2) on n 4 M 1024 / 1536 and n 2 M 1280 / 1664 / 2048: rms folded / unfolded 1.005 - 1.009, no case flagged; `poison.py` bitwise equal on clean, poisoned and poisoned-capture memory; replays bit-identical.
SASS of the chain's unchanged cubins (`sx_ln`, `sx_expand32 / 64`, `sx_sq64_2 / 32_2`, base against this tree): identical except the three `Bars`-relative shared-memory addresses of `sx_expand` (the barrier struct grew by the fold's barriers).
`n 4 L 128` (1 tile, unchanged code) measured 14.34 / 16.2 / 16.2 us in the three base processes and 16.2 four times in the new tree: the tick edge of that chain (device span 10.34 - 10.43 us, the edge at ~10.4 us), not a code difference (above).

### N1 second round (2026-11-19): ladder, the measured 1.5x list, and the training / D 512 / fp32 one-tile items (all measured; nothing further shipped, `cuda:b200_sm100` stays 62, training counter untouched)

**Ladder** (`ladder.py` on token-format files made from one `measure.speed` run of Transition + InputFeatureEmbedder, cuda / triton / pytorch, eval + train, every length, graph, L2 flushed: 1353 records, 307 Transition and 144 InputFeatureEmbedder cases with all three backends, `n1cfg/conv.py`, `n1out/lad.jsonl`): **no violation** (cuda < triton < pytorch with 2 % margin) for either module, no exception needed, no external-baseline miss.
**1.5x list** (`vsh100_15.py` on the same records against `judge/tokens/h100_sm90`, 15 exempt cases (H100 < 20 us)): Transition 23 eval + 9 training misses (this single run; the three-run eval picture of the earlier section is 21 - 22), none slower than the H100 outside the exempt cases (worst exempt ratio 1.186 = row 25 L 128, 12.2 against 10.2 us); InputFeatureEmbedder 5 eval + 23 training misses (ratio 0.67 - 0.73, L 3072 - 8192). List: `n1out/vsh.txt`.
Training misses: D 128 n 2 master L 128 (0.849), D 64 n 4 L 128 (0.743), row 219 D 768 master L 256 / 384 / 512 / 640 (0.70), row 211 float32 L 128 (0.692), row 217 master L 128 (0.669), row 218 L 384 (0.668).

**Training single-stream rows 217 - 219: the fp32 master cast.** Ceiling (timing-only, `-DCASTN=0` = the cast blocks convert nothing, `n1cfg/tab.py` with `XLN`): step span 21.25 -> 20.69 (row 217 L 128), 28.11 -> 27.01 (218), 22.90 -> 21.47 (217 L 384); graph 26.62 -> 24.58 / 32.77 -> 30.72 / 26.62 -> 26.62: the whole cast is 0.6 - 1.4 us of device time.
Built: S90's `-DCASTF` (the expand converts Wa / Wb itself from the fp32 masters, Ws in the LayerNorm launch, L2 prefetch of Wa / Wb from the first launch) re-ported onto the current `tsx_expand.cu` / `tsx_ln.cu` / `_sx_forward` (flag `SX_CAST_IN_EXPAND`): gradients and y bit-identical, deterministic, step span (off -> on, same process, `tab.py`) 21.38 -> 21.84 (217 L 128), 28.29 -> 28.19 (218 L 128), 22.50 -> 22.75 (217 L 384), graph medians equal (26.62 / 32.77 / 26.62): **level, rejected** (`rejected/transition_n1_castf_report_training.diff`). `CASTF` does not apply at D 768 (the 12 K-blocks do not fit one ring).
D 768 (row 219): the cast launch is 7.20 us against 2.45 us without the cast blocks (`n1cfg/castmb.py`, 21.3 MB moved: 4.4 TB/s of the ~6 TB/s HBM peak); elements a cast thread 1 / 2 / 4 / 8 groups of 8: 9.7 / 7.2 / 9.0 / 9.5 us (`castmb2.py`, the shipped 2 is the optimum, same as at D 384): ~1 us is left against the HBM roof. The training-miss rows 219 L 256 - 640 need 2.3 - 2.8 us.
**D 512 squeeze** (`tgemm_nd.cu -DPARTS=1`, [262 144, 512] n 2, `n1cfg/sq512.py`, kernel us, L2 flushed): base 228.4, `TBE` (no epilogue) 175.0, `TBA` (A / h loaded for the first ring fill only) 173.0, `TBW` (Ws loaded once) 218.1 - 220.2, all three 152.6 - 160.7. Removing either the epilogue's traffic (x in, out) or the h stream brings the kernel to the tensor bound (~175); with both, the bytes (537 MB h + 268 MB x + 268 MB out = 1.07 GB in 228 us = 4.7 TB/s, 78 % of the 6.0 - 6.5 TB/s copy rate) and the tensor pipe (75 % busy) are each at ~75 %:
the kernel is bound by the two streams competing for HBM and the MMA / epilogue hand-off, not by either alone. No flag-gated build beyond the existing probes was built (ring depth is at the smem limit: 5 stages + the 64 KB epilogue x ring; a third epilogue warp group needs 8 column blocks / 3 groups: `NEPI = 3` fails its own assert); lever left: fewer bytes (the h round trip: the fused pair kernel was 1.6 - 2.1x slower, S90 part 21).
**fp32 one-tile serial chain** (`tg_fused_h16.cu`, `n1cfg/f32one.py`, kernel us, L2 flushed, 3 interleaved rounds; probes give wrong numbers): D 128 [16 384] base 11.78, `TBLN=2` (no LayerNorm) 10.27, `TBEPI=1` (no y epilogue) 10.43, `TBW=1` 11.78, all three 8.78 (ceilings: LayerNorm 1.5 us, epilogue 1.35, together 3.0); D 256 [16 384] 23.52 / 19.90 / 20.85 / 23.39 / 17.05 (LayerNorm 3.6, epilogue 2.7, together 6.5); D 128 [65 536] 32.77 / 28.96 / 24.59 / 32.47 / 21.2.
The tree already has what the serial head allows (S90 / round 3: x tile prefetched into L2 before the setup, the LayerNorm in tcgen05 fragment order at D 256, the first x tile staged in a ring stage, EP3). Not built: a row-half split of the x boxes (LayerNorm of rows 0 - 63 under the load of rows 64 - 127: at most the second half's arrival lag, ~0.4 us) -- below the tick, row 29 float32 L 128 needs 2 us; the fold (K16 correction) does not apply to the fp16-operand path (the x tile has to be converted anyway).

**D 512 squeeze, built (flag-gated, `tgemm_nd.cu -DXEF=1 / -DOEF=1`, `sq512.py`, kernel us, L2 flushed, 3 alternating rounds, base -> flag)**: the epilogue's x loads with an L2 `evict_first` hint (`XEF`; the stores and the h stream compete with it for the 126 MB L2 and HBM): [65 536, 512] 72.9 -> 72.7, [147 456] 136.2 -> 136.2, [262 144] 228.4 -> 226.3 / 225.7, [409 600] 353.2 -> 346.1 (-2.0 %): **adopted** for the `squeezep` (PARTS, n 2) cubin (`fused_wide_sm100a._specs`), hint only: bit-identical output, rows 33 / 34 `measure.accuracy` 12 / 12 GREEN.
`OEF` (evict_first on the TMA stores of y): 228.4 -> 234.5 / 228.4 (level to +2.7 % slower), not adopted. The h (A) load is a `cta_group::2` TMA with no hinted form in `sm100.cuh`; a hinted 2-SM load was not built. Gain is limited to the large cases (rows 33 L 512 - 640, ~1 %), below the tick everywhere else.

### N1 third round (2026-11-19): InputFeatureEmbedder training / eval against the H100 (blockers measured; nothing shipped, versions untouched)

Misses (`n1out/vsh.txt`, one run): training 23 (master, ratio 0.67 - 0.73: global window 1 000 000 at L 3072 - 8192 and window 128 at L 4096 / 7168), eval 5 (master L 6144 x3 0.67 - 0.69, float32 L 5120 / 6144 0.67 / 0.70). Tools: `ifeprof.py` (kernel list and timeline of the measurer's own step, CUPTI), `ifeab.py` (same-process interleaved A/B of any module attribute on the measurer's step: outputs and gradients bitwise vs the first value, determinism, 3 rounds of graph medians), `ifeshape.py`, `mmk.py`.

**Blockers, esmfold2 / relu / global, master training L 4096** (measurer 696 us; timeline above, us; forward 153 -> 411 = 258, backward 445 -> 873 = 428; 86 kernels):

| group | us (3 blocks) | share of the step | what bounds it |
|---|---|---|---|
| global attention backward `gattn_bwd` (cluster split CS 2: 64 items x 2 = 128 of 148 CTAs) | 190 (63.6 a block, on the chain) | 27 % | ~3.5x its tensor / MUFU floor (21.5 GFLOP, 67 M exponentials); S4 probes of the SWA part 18: no single limit |
| TF32 FFN forward `ffn_fwd_tf32` (64-row tiles on CTA pairs, 128 CTAs) | 142 (47 a block, ~19 after the attention on the chain) | 20 % | serial MMA / SwiGLU chain a tile (weights 448 KB a tile) |
| global attention forward | 94 (31 a block) | 13 % | 64 items, 128 CTAs |
| cuBLAS TF32 GEMMs (dWmod [768,4096]x[4096,128], dc [4096,768]x[768,128], dx bmm) + split-K reduce | 47 + 41 + 28 + 29 = 145 (partly beside the dW kernels) | ~15 % | few-CTA GEMMs of deep K, ~10 us each |
| modulation / weight forms / roundings / casts (`mod_fwd` 39, `wprep` 24 + 16, `round` 15, `gcast` 28, `mod_silu` 22 + 13, `qkvg` 40) | ~200 | ~25 % | 87 dependent launches at 2 - 6 us |
| oproj bwd + ffn bwd gate + adaLN + dW | 45 + 47 + 36 + 53 | ~20 % | the TF32 backward's dependent stages |

Not bytes, not atomics (the dQ adders are counter-ordered in the kernel, deterministic), not the fp32 master cast (the weight forms are one cached launch: `wprep` 24 us of 696): launches and the long chains of the attention / FFN stages.
Eval master L 6144 (306 us): `gattn_fwd` 3 x 70 = 210 us (69 %): 4 heads x 24 key-pair items = 96 CTAs of 148 (cluster split needs items x CS <= 148: CS 1), MUFU floor 52 us at 96 CTAs; FFN 3 x 27, qkvg 3 x 12.

**A/B on the measurer's step (same process, interleaved, graph us, first -> second)**: `tf32_fwd.ROWS_FORCE` None / 32 / 64 / 128 (L 4096): 693 / 700 / 694 / 707 (default is already the 64-row pair build); `FFN_PAIR` True -> False 690 -> 700; `CSB_BWD` (global backward cluster split) True -> False 690 -> 764; `CSPLIT_FWD` True -> False 694 -> 725; `SPLIT_GAIN_FWD` 1.2 -> 0.5 (eval L 5120 / 6144 / 8192) 262 / 307 / 414 -> 262 / 307 / 419 (no change at 5120 / 6144); `atom_dit._B200_MT_BF16_MAX_ROWS` 0 -> 16384 (window-128 master training, the hybrid bf16 forward on the TF32 backward) L 3072 / 4096 / 6144 / 7168: 379 / 465 / 551 / 592 -> 506 / 543 / 658 / 709 (+30 %, stays off).
Every shipped default already is the best of its knob; gradients and outputs bit-reproducible in all runs. K-split of the deep-K backward GEMMs (`mmk.py`, bmm over 2 - 16 K slices + ordered sum): level or slower for all six shapes (e.g. [768,4096]x[4096,128] mm 16.4 us, 16.4 - 18.4 split; [128,4096]x[4096,133] 20.5 -> 18.4 at 16 slices).
What the 8 % (L 4096 696 -> 638 us) would take, beyond knobs: a global-attention backward / forward running at ~60 % instead of ~30 % of the tensor / MUFU floor on all 148 SMs (per-item granularity: 96 / 64 items do not tile 148 SMs; `docs/b200/atom-dit.md` part 22 "not reached"), or fewer / fused launches in the TF32 backward (the SWA agent's open items) -- both are rewrites of the SWA kernels, not done here.
Eval: 3 of the 5 misses are 1 - 1.7 % from the line (master L 6144 linear / relu need 300.5 / 301.6 us against 305.0 / 304.9, float32 L 5120 317.2 against 315.2: run-to-run level); master legacy L 6144 (305 against 294) and float32 L 6144 (360 against 342) need 4 - 5 %.


### N1 pass, round 4 (2026-11-18): InputFeatureEmbedder training glue (shipped) / gattn_fwd, gattn_bwd decomposition (measured, no change)

Shipped (training only; bitwise the same results; `SWAAtomTransformer` 27, `AtomDiTMiniWorld` 19, `InputFeatureEmbedder` 29 -- training versions are odd):
- the native global backward's glue: the three head casts (`swa_gcast`, nc 3) and dO16 + D (`swa_gdelta8`) were two launches (two independent kernels, 5.4 + 2.7 us at L 4096, one 0.6 us gap) -> ONE launch (`gattn_tf32.cu swa_gprep_tf32_sm100`: first 3 n8 threads cast, the next 8 N S H threads the delta; warp-uniform split, same per-element arithmetic), launched programmatically (PDL attribute + `griddepcontrol.wait`) with `griddepcontrol.launch_dependents` at its top, so `gattn_bwd` (waits itself) runs its prologue under it; switches `tf32_bwd.GPREP`, `tf32_fwd.GPREP_PDL`, `GPREP_IN_PDL`, `N1_GLUE` (master A/B switch).
- the forward cast launch (`swa_gcast_pdl_tf32_sm100`) triggers the dependent attention launch at once (`GCAST_PDL`).
- same-process interleaved A/B (graph, L2 flushed, `n1cfg/ifeab.py`, 3 rounds x 41 replays), N1_GLUE False -> True, bitwise-vs-first True and run-to-run deterministic on every case: ESMFold2 master train L 3072 587.9 -> 571.5 / 583.8 -> 569.3 us, L 4096 692.4 -> 672 (fused launch -9.5, gprep trigger -4, gprep launched with PDL -2 - -4, gcast trigger -4), L 8192 1443.9 -> 1425.5 / 1439.8 -> 1417.3 us, legacy L 3072 577.5 -> 565.2, L 8192 1431.5 -> 1407.0, float32 legacy L 3072 620.5 -> 606.2, L 8192 1470.5 -> 1454.0 us (-1.3 - -2.9 %). Eval is untouched (no cast launch there: A/B +-2 us tick noise).
- H100 1.5x list for IFE training (`n1out/vsh_ife.txt`, `vsh100_15.py`): 23 -> 21 misses of 72 (all master, ratios 0.671 - 0.740; 8 of them within 1.5 % of 0.667).
- accuracy (`measure.accuracy`, IFE L 3072 / 4096, master + float32 + bfloat16, train + eval): cuda 18 GREEN / 18 YELLOW / 0 RED (worst YELLOW 2.7x the pytorch error: float32 grad:pos); `SWAAtomTransformer` / `AtomDiTESMFold2` L 3072 train: no RED. Ladder (cuda < triton < pytorch, 2 %): 0 violations of 72 IFE train cases. `_prebuild --arch sm100 --check`: plan 793 artifacts, 0 not built; all 204 "no compiled kernel matches" lines are `*_sm90` names (augmented_attention / dit_variants), the 4 "compiled only by other recipes" lines are the mpnn families; rc 1 comes from those lines only.

Measured, not shipped:
- `gattn_bwd` decomposition (L 3072 / 4096 / 8192, same process, graph, L2 flushed; `n1cfg/gbb.py`, `gbv.py`): auto (cluster split CS 2 / CS 3 x 2) 50.8 / 65.8 / 211.7 us vs split mode (148 CTAs, workspace combine) 63.3 / 89.0 / 231.5 and whole items 80.4 / 105.8 / 211.7: the existing modes are already the best choice. The 64 items of L 4096 x 2 shares fill 128 of 148 SMs (ideal 148/128 -> -8 us a call, -3.5 % of the step); a launch has ONE cluster size, a mixed CS 2 / CS 3 grid (20 items with 3 shares) needs a per-item share table and non-uniform clusters -- not built (the split build's finer decomposition is 35 % slower at this shape). Build flags of the cluster build (PMOD 2 / 3 / 8, ST 4, SPIN_NS 16, PCNT 0, NSTG): PMOD / ST / SPIN within +-1.5 %, PCNT 0 (all exponentials on MUFU) -1.4 % (changes bits, 0.4 % of the step): not taken. Race hunt (`n1cfg/gbrace.py`, 200 runs each, SPIN_NS 64 / 16 / 1): 0 mismatches.
- `gattn_fwd` (eval L 6144): the auto path already uses all 148 SMs (split mode); a microbench-only "3rd object 11 - 30 % faster" mode could not be reproduced in the step (in-step 68.8 - 69.1 us with 0 / 3 / 6 throwaway objects). The five eval misses are unchanged (3 within 1 - 1.7 %).
- rejected variants: swa_gtail8 launched programmatically (+-0.3 us), swa_gcast launched programmatically (0), the forward's `save32` widening on a side stream joined by the backward (window 128 L 4096: 460.9 -> 459.3 us esmfold2, 448.5 -> 454.7 us legacy; `docs/b200/rejected/n1-ife-save32-side-stream.diff`).
- trace tools (`n1cfg/`): `ifeprof.py` (TL=1 timeline, TRACE=file chrome trace), `trace.py`, `excl.py` (exclusive / shared device time per kernel: L 4096 global train: gattn_bwd 185 us exclusive of 679 busy, ffn_fwd 56, qkvg 25, gtail8 11), `gbb.py`, `gbv.py`, `gbdet.py`, `gbrace.py`. The ~74 us gap after `swa_wpack_hybrid` seen in a profiled first replay is a cold-launch artifact (absent in warm back-to-back replays).
- (later, same round) inference: `swa_embed_rope2_sm100` (embed_glue.cu; the rope launch's seqused count on 16-byte loads: the one-block-a-row count was the launch's tail, 7.4 - 10.8 -> 3.2 - 3.4 us), used only without grad (`embedder._ROPE2`; a training forward measured +8 us at window 128 L 4096 / -4 us global, so train keeps the old launch). Same integers, bitwise equal, deterministic; eval step A/B: global L 4096 190.5 -> 186.4, window 114.7 -> 110.6, legacy 237.5 -> 231.3 us, L 8192 -10 - -12 us. `InputFeatureEmbedder` cuda:b200_sm100 30. Rejected: `swa_embed_scatter2` chunked/speculative loads (rejected/n1-ife-embed-scatter2-chunked-loads.diff; ablation: head detection 3.6, no loop 5.9, full 11.0 us at 6144 atoms; no variant faster at N >= 6144). Not re-measured after rope2: H100 1.5x list and ladder (train ladder 0 violations / 21 train misses were measured before rope2; eval list not rerun).
- final IFE numbers (after rope2, `n1out/sp_ife.jsonl`, `vsh100_15.py`): 144 cuda cases (eval + train, all dtypes / lengths): 26 not >= 1.5x faster than H100 cuda = 24 master train (ratios 0.667 - 0.730; the set near 0.667 moves by run) + 2 eval (1 master, 1 float32; were 5); IFE eval ladder (cuda < triton < pytorch) and train ladder: see violations in the hand-back (train 0 / 72 measured before rope2).

## N5 pass (2026-11-19, agent n5-transition, `cuda:b200_sm100` 64, inference): the LayerNorm folded into the CTA-pair expand (D 768 n 2, 4 - 6 tiles); the one-launch and tick questions re-measured

Tools (probes, not routed; `n5out/` of the tree): `pfold.py` (D 768 n 2 pair chain, three launches `u` against the folded chain `f`: error against fp64, device span, graph medians, 3 interleaved rounds; env `PFX` extra nvcc flags of the folded expand, `PKBX` K-blocks a stage, `MODES`), `pstamp.py` (per-CTA `%globaltimer` stamps of the folded expand: `-DTSX_STAMP=1` through `fused_small_sm100a.PAIR_FOLD_X`, `PAIR_STAMP`), `noln.py` / `noln2.py` (timing ablation: the chain with its `tsx_ln` launch skipped, wrong results), `poison_pair.py`, `rowv.py`, `miss.py` (the Transition 1.5x list of a gap-measurer file).

### Measurement phase (before any build): what is left of the 32 Transition misses

`miss.py` over `n3-verify/g1_transition.jsonl`: 32 cases; the graph medians sit on the 2.048 us grid, so a case flips only when the device span crosses a tick edge (graph = span + ~5.0 - 5.4 us; D 768 n 2: 18.43 at span <= ~14.9 us, 20.48 above). Options of the brief, measured / re-read before choosing:

* **(A) one persistent kernel for expand + squeeze (cluster, DSMEM h): not rebuilt.** Round 3 part 6 built it (`tsx_one.cu`: 41.0 / 32.8 us at [1, 2048, 384] n 4 against the chain's 28.7) and S90 follow-up 3 measured the data movement alone (`skel.cu`: no arithmetic, CS 8: 18.4 us, CS 16: 24.6 us, against 14.1 us of the whole 3-launch chain): the DSMEM all-gather of xn / h moves ~19 - 23 GB/s a SM and the weights stream per CTA at ~55 GB/s. A flag-synchronised variant (h through L2, no cluster) would replace the 1.5 us completion hop by a software barrier of 1.2 us (S90 part 8: 1.217 us a barrier at 148 CTAs) and cannot prefetch the squeeze weights (the expand's 230 KB shared memory is live until its epilogue): ~1.2 us at best for a >= 400-line fusion of two tcgen05 pair kernels with unioned shared memory: not started.
* **(B) one-tile D 128 / D 256 row-half pipeline: not built.** `f32one.py` ceilings (LN 1.5 / 3.6 us, epilogue 1.35 / 2.7 us at D 128 / 256) stand; the cases that remain (fp32 D 128 L 128 16.2 -> needs 13.8, D 256 L 128 28.4 -> needs 28.2) need 2 - 4 us.
* **(C) the LayerNorm launch of the D 768 n 2 pair chain** (rows 37, 1 - 6 tiles: `tsx_ln` -> `tsx_expand2` -> `tsx_squeeze2`, three launches; the N1 fold had only the single-CTA D 384 expand): ceiling by ablation (`noln.py`, the chain without `tsx_ln`, device span us, graph in brackets): [1, 128, 768] 9.95 -> 9.15 (14.34 -> 14.34), 256 11.14 -> 10.56 (14.34), 384 12.08 -> 11.20 (16.42 -> 16.38), **768 15.52 -> 14.14 (20.48 -> 18.43)**; the 16-tile case (PEX + cuBLAS addmm, row 37 L 2048): 22.43 -> 20.22 (26.62 -> 24.58, needs 25.0). Chosen: build the fold into the pair expand (`tsx_expand2.cu -DFOLD=1`).

### What changed (`cuda:b200_sm100` 62 -> 64; inference only; the training kernels and every non-fold cubin are untouched)

* `sm100/widths/tsx_expand2.cu -DFOLD=1` (macros only: the SASS of all 42 pre-existing cubins of the D 768 n 2 pair / small / H16 cells is identical to the base, `cuobjdump -sass`): the raw x is the A operand, resident (this CTA's 64 rows, `NKB` K-blocks, one 3-D box per `KBX` K-blocks on the leader's barrier `xfull[stage]`), Wp = bf16(W gamma) the B operand in its own ring (`full[s]`), loaded before `griddepcontrol.wait` as in the unfolded kernel. The row statistics come from the x tile in shared memory as the K-blocks land (shifted single pass, shift = the row's first element, the two K-block halves on threads 0-63 / 64-127 of warps 4-7, exchanged through shared memory; the peer reads through a relay `xready[stage]` the leader arrives on once the stage has landed: the peer's threads cannot wait on the leader's barrier). The rank-1 terms are applied in fp32 in the epilogue, a = rs (acc - mean u_n) + v_n (`aug` [H, 4] fp32 = u_a v_a u_b v_b, u_n = sum_k Wp_nk of the rounded Wp, v_n = sum_k beta_k W_nk: a weight form, bulk-loaded to shared memory before the wait): no K16 correction MMA, so the pair product has no extra operand planes. The epilogue is packed f32x2 with the one-MUFU sigmoid (`sigmoid2`: 0.5 tanh(a / 2) + 0.5; the pair epilogue at the two-op `sigmoid_kit` form is MUFU-bound: 128 units x 64 rows = 16 K MUFU ops a CTA = ~0.5 us), flag `SIGT`. x of the CTA's 64 rows is prefetched into L2 before the wait (`cp.async.bulk.prefetch.L2`; legal before `griddepcontrol.wait`: only the L2 line is touched).
* `fused_small_sm100a.py`: `PAIR_FOLD` / `PAIR_FOLD_TILES = {(768, 2): (4, 5, 6)}`, `pair_fold_takes`, `_pair_fold_forms` (Wp, **interleaved by UH-unit chunk** [Wa_c; Wb_c] so that a CTA's weight stage is ONE 3-D box instead of 2 x KBX 2-D boxes; one copy per UH build + `aug`; a weight form made once per parameter version), `pair_fold_chain` (two launches: folded expand + `tsx_squeeze2`), `_pair_fold_launch` (opaque op with a fake), `PAIR_FOLD_KBX = {16: 4, 32: 4, 64: 2}`, three new cubins `pair_d768n2/exf16 / exf32 / exf64` (in the prebuild plan: 796 -> 799 artifacts). `modules/transition.py`: `backend_versions["cuda:b200_sm100"]` 62 -> 64.

### Measured (B200 GPU 3, exclusive, L2 flushed, one process, `u` / `f` alternating, 3 rounds; us)

| [1, M, 768] n 2 | tiles / UH | span u | span f | graph u | graph f |
|---|---|---|---|---|---|
| M 128 | 1 / 16 | 9.98 | 10.54 - 10.75 | 14.34 | 14.34 |
| 256 | 2 / 32 | 11.20 | 11.68 - 12.16 | 14.34 | 16.38 |
| 384 | 3 / 32 | 12.11 - 12.18 | 12.54 - 12.88 | 16.38 - 16.42 | 16.38 - 18.46 |
| **512** | 4 / 64 | 14.08 - 14.19 | **13.44 - 13.50** | 18.43 - 20.48 | **18.43 - 18.46** |
| **640** | 5 / 64 | 14.53 - 15.01 | **13.87 - 14.03** | 18.43 - 20.51 | **18.43 - 18.46** |
| **768** | 6 / 64 | 15.04 - 15.41 | **14.30 - 14.42** | 18.43 - 20.51 | **18.43 - 18.46** |

Fold at 1 - 3 tiles is **slower** (+0.4 - 1.0 us) and not routed: there the LayerNorm launch overlaps the weight stream (the expand's weights are in shared memory when `griddepcontrol.wait` returns), so removing it moves the weight intake onto the critical path (the kernel is bound by the per-SM intake, A 96 KB + W 98 KB at UH 32). At UH 64 the weight panel is 196 KB a CTA and the LayerNorm is worth more than the lost overlap. The per-CTA stamps of the folded expand ([1, 768, 768], us from the first CTA's entry): `griddepcontrol.wait` returns 0.77, first stage landed 2.30, last 4.91, statistics done 4.78 (not critical), accumulator done 5.50, epilogue end 6.40, exit 6.59 -- the 290 KB of operands a CTA (144 CTAs: ~42 MB of L2 reads in 4.1 us) and the 0.6 us tail of the last stage's 8 UMMAs bound it.

Ablations of the first build (timing only, wrong results): no peer relay 10.72 against 10.69 (L 128; the relay costs nothing); no statistics pass 10.45 (-0.25 at L 128, -0.55 at L 256); L2 prefetch of x: L 512 13.63 vs 13.82, L 768 14.67 vs 14.64 (level: kept, it costs one instruction); `SIGT=0` (two-op sigmoid) L 512 13.63, L 768 14.59 against 13.41 / 14.18 (+0.2 / +0.4: adopted `SIGT=1`); per-K-block boxes for x and W (KBX 1, 12 + 24 TMA ops a CTA): L 768 14.70 against 14.53 with merged boxes (the interleaved Wp form); KBX 3 at UH 64: L 512 13.58, L 768 14.43 against 13.46 / 14.22 for KBX 2 (KBX 4 leaves one ring stage: refused by the build).

### Accuracy (basis: random non-zero weights, gamma 1 + 0.2 randn, beta 0.2 randn, x ~ N(0, 1.5^2); first 4096 rows against fp64 on the master weights; `pfold.py`) and the checks

* max error / max |ref|, u / f: [1, 512, 768] 2.753e-3 / 2.697e-3, [1, 768, 768] 2.753e-3 / 2.697e-3, rms 3.30e-3 / 3.28e-3 (equal to 3 digits); max |f - u| 0.031 (one bf16 ulp of the output); stress: x mean offset 3 / 20 / 100 (rms 6.0e-3 / 3.6e-2 / 0.145 equal in u and f), first element 50 and 3000 (err 2.50e-3 / 6.59e-4 in both), L 640 offset 3 + spike 3000: 6.589e-4 both. Reruns in the same process bit-equal (deterministic: fixed summation orders, no atomics).
* `measure.accuracy` (the official measurer, cuda + pytorch, row 37 every length, `acc37.jsonl`): master eval 9 / 9 GREEN (L 512 / 640 / 768 err 3.52e-3 / 3.75e-3 / 3.75e-3 against tol 8.0e-3, PyTorch 4.84e-3), float32 eval 4 GREEN + 5 YELLOW (the fp32 route is the H16 chain: SASS-identical cubins, route untouched by this pass). Rows 25 - 36 / 38 / 209 - 221: no code path of them changed (all 42 pre-existing cubins SASS-identical, the route only differs for bf16 D 768 n 2 at 4 - 6 whole tiles); `tests/gpu/test_transition_b200_uninit.py` 19 passed (training).
* memory: `poison_pair.py` (allocator blocks filled with 0xFF, eager and inside a capture whose pool was poisoned first, [1, M, 768] n 2 M 256 / 384 / 512 / 640 / 768): bitwise equal to clean memory.
* `_prebuild --arch sm100 --check`: plan 799 artifacts (base 796: the three `exf` cubins); the exit code 1 and the `no compiled kernel matches ... _sm90` lines are the same in the base tree (full outputs `prebuild_check_base.txt` / `prebuild_check.txt`, only the re-keyed `tsx_expand2` hashes and the three new cubins differ).

### Result against the H100 1.5x rule (gap-measurer, `GAP=0.15`, all 352 Transition requirement records; base `n3-verify/g1_transition.jsonl` -> this tree, two full runs)

| case | H100 | needed | base | run 1 | run 2 |
|---|---|---|---|---|---|
| **row 37 master eval L 768** | 27.7 | <= 18.47 | 20.14 | **18.30** | **18.22** |
| row 37 L 640 | 23.5 | <= 15.7 | 20.26 | 18.30 | 18.21 (still a miss) |
| row 37 L 512 | 23.1 | <= 15.4 | 18.35 | 18.30 | 18.21 (still a miss) |
| row 37 L 128 / 256 / 384 | -- | -- | 14.11 / 16.16 / 16.19 | 14.21 / 16.26 / 16.26 | 14.32 / 16.16 / 16.18 (unrouted, level) |

Misses of the list (H100 >= 20 us, B200 / H100 > 0.667): 32 -> 31 / 30; none slower than the H100 except the exempt row 25 L 128 D 64 n 4 case (12.1 against 12.0 us, H100 < 20 us); ladder (cuda < triton < pytorch, 2 %): 0 violations in both runs. The differences of the other cases between the three files are one-tick jitter of code this pass did not touch (e.g. training D 384 n 4 L 256 / L 512 swap in and out).

### Built / measured and not kept

* Fold at 1 - 3 tiles (`PAIR_FOLD_TILES` with 1 - 3; KBX 2 / 3 / 4 / 6 at UH 16 / 32 all +0.4 - 1.0 us): table above.
* The persistent pair expand (`PEX`, row 37 L 2048, 16 tiles, cuBLAS addmm squeeze): the LayerNorm launch is also the x -> out copy of the addmm; the ablation ceiling is 22.43 -> 20.22 us (graph 26.62 -> 24.58 against 25.0 needed) but a fold there needs stats from the streaming A ring (not resident) and a replacement for the copy: not built.
* D 768 L 1024 (8 tiles, wide chain): the ablation ceiling is 17.28 -> 16.11 (graph 22.5 -> 20.48, needs 19.1): out of reach.

### What still misses 1.5x, and why (30 cases of run 2; unchanged by this pass except row 37 L 768)

The cases 2 - 4 ticks over the line need the launch-boundary / one-tile-latency work of (A) / (B), measured above and in parts 6, 17, 19, S90 follow-ups 3 / 21: the DSMEM exchange (19 - 23 GB/s a SM) and the per-SM operand intake (~70 - 85 GB/s a SM; here 290 KB a CTA at 4.1 us) bound every fusion that keeps the hidden activation or the weights moving per CTA. The one-tick cases (row 37 L 2048, D 384 n 4 L 1536 / L 3968, D 384 fp32 L 1024, D 256 fp32 L 128) sit 0.5 - 1.4 us from their tick edge; the next lever for L 2048 is the fold in the persistent expand (ceiling 24.58 us), for the D 384 n 4 cases the one-MUFU sigmoid of this pass applied to `tsx_expand.cu` (12 tiles at CH 128: 32 K MUFU ops a CTA = ~1.1 us of epilogue; not built, accuracy rows 35 / 36 would have to be re-audited).

CPU gate: `tests/cpu/test_judge.py test_capture_api.py test_prebuild_plan.py test_requirements.py` 1226 passed, 1 failed (`test_every_module_level_dict_is_registered_or_listed`, naming `fused_wide_sm100a.SX_ST` of the N1 pass; the same single failure in the base tree). Rejected-variant diffs: none kept as files (every rejected variant of this pass is a flag or table entry of the tree: `PAIR_FOLD_TILES` with tiles 1 - 3, `PAIR_FOLD_KBX`, `-DSIGT=0`, `-DFABL=<bits>` probes, `PAIR_FOLD = False` for the three-launch chain).

## N9 pass (2026-11-19, agent n9-transition, `cuda:b200_sm100` 64 -> 66, inference): the bf16 expand epilogue on packed pairs with the one-MUFU tanh sigmoid; the FMA-pipe offload of the sigmoid measured and rejected

### SASS count (cuobjdump, `tsx_expand.cu` D 384 n 4, CH 64, SK 1, 16 pairs a thread, per 2 elements)
* before: 4 MUFU (2 EX2 + 2 RCP) + 6 FMUL + 2 FADD = 12 instructions; after: 2 MUFU.TANH + 2 FMUL2 + 1 FFMA2 = 5 (`ah = a / 2; a sigmoid(a) = ah (1 + tanh ah)`, the 0.5 folded so the pair costs mul2, 2 MUFU, fma2, mul2 by b). Kernel total 1144 -> 1024 instructions. FOLD build: 1984 -> 1824.
* tg_fused_h16 (fp32 one-tile) epilogue was already 8 instructions a pair (FMUL2, 2 MUFU.TANH, FFMA2, 3 FMUL2, F2FP); folding the 0.5 (7 a pair) measured 0.0 - 0.1 us: not kept.

### What changed
`tsx_expand.cu` (bf16 builds only: `SIGNEW = !F32 && !H16`, `SIGPAT = 0xFF`): the epilogue of the SK 1 chunk loop, the FOLD form and the K-split cluster form work on f32x2 pairs (`swiglu_pair`), sigmoid by `tanh.approx`. fp32 / fp16 (`H16`) builds are the old scalar code (byte-identical SASS). `sm100.cuh`: `sigmoid_z2`, `rcp_nr2`, `ex2_poly2` (packed Newton reciprocal / polynomial 2^x, modes selectable per pair position by `SIGPAT`; OFF, see below). `fused_small_sm100a.py`: `EXPAND_X` probe hook (empty). Version 64 -> 66.

### Measured (B200 GPU 2, gap-measurer GAP=0.15, all 352 Transition records, min of 3 runs each, base = n9-base; us; tick 2.048)
eval D384 n4 L128 16.16 -> 14.11, L256 16.16 -> 14.21 (H100 line 13.2 / 13.1); eval D384 n2 L1536 16.16 -> 14.24 (line 13.81); eval D768 n2 L1024 22.30 -> 20.48 (line 19.06); train D384 n4 L384 34.59 -> 32.62, L768 36.64 -> 34.80. Device span (same process family, 3 rounds): D384 n4 L1024 12.93 -> 12.16, n4 L512 11.78 -> 11.54, n2 L1536 11.04 -> 10.64, n2 L1024 10.64 -> 10.27, D768 n2 L1024 17.4 - 17.8 -> 16.7 - 17.1. D768 n2 L512 - 768 (pair kernel `tsx_expand2.cu`, already tanh) unchanged. H100 1.5x misses 30 -> 29 (nothing crossed the line; D384 n2 L1536 is 0.43 us above it). Some unrelated cells move +-1 tick between runs (d 64 / 128, float32, train D384 n2): same cubins, timer-tick noise.

### Rejected (data)
* MUFU -> FMA-pipe offload (`SIGPAT` mode 1 Newton reciprocal, 2 polynomial exponential, mixes): D384 span n4 L1024 base 12.93, all-kit packed 12.83, Newton 12.99, polynomial 13.18, half polynomial 12.74; tanh 12.3 (before the fold). The packed FMA path costs ~3x the issue slots of the MUFU pair it replaces. tg_fused_h16 (fp32, `HFMAPAT`): D128 L128 kernel 10.02 -> 10.54 (1/4 of the pairs), 12.2 (all): monotone slower, D256 18.96 -> 19.8 / 24.6. Diff: `rejected/transition_n9_tg_fused_h16_fma_offload_and_fold.diff`.
* `tanh.approx.f16x2` (one MUFU op a pair): SASS shows 2 MUFU a pair anyway plus conversions: not built further.
* tsx_expand2 non-fold bf16 tanh (`rejected/transition_n9_tsx_expand2_nonfold_tanh.diff`): 0.0 us in every D768 n2 cell (the non-fold pair is MUFU-neutral).
* tsx_squeeze*, tsx_ln, tgemm_nd have no sigmoid: untouched.

### Accuracy (basis: random non-zero weights, gamma 1 + 0.2 randn, beta 0.2 randn; fp64 gold from the module weights; `n9out_ab.py`) and checks
Stress x scale {1e-3, 1, 40} x mean {0, 25} on D384 n4 L512, n2 L1536, D768 n2 L512 / L1024: rms error vs gold identical to 3 digits with and without the change (e.g. scale 1.5: 2.512e-3 vs 2.512e-3; torch bf16 3.47e-3), max error within 13 % in one cell (scale 1: 2.01e-2 -> 2.28e-2, PyTorch bf16 3.5e-2); no NaN. `measure.accuracy` Transition all 614 records: cuda 247 GREEN / 57 YELLOW / 0 RED, identical colours to base, one error changed (row 36 L1536 3.99e-3 -> 3.67e-3). tanh.approx error: ~2^-11 relative on tanh (absolute 2.4e-4 on the sigmoid for a < -2; it is the form TR_SIG 1 / SIGT 1 already ship). `tests/gpu/test_transition_b200_uninit.py` 19 passed; poison-memory check (eager clean = eager poisoned = captured graph, bitwise) ALL EQUAL for 20 D 384 / 512 / 768 cells (`n1cfg/poison.py`, `poison_pair`); `_prebuild --arch sm100 --check`: 0 not built, 204 pre-existing "no compiled kernel matches" lines, identical to base (rc 1 as in base).

## FP8 (opt-in, 2026-11-19)
An opt-in e4m3 inference route for D 512 / 768 (`kernels/transition/cuda/fp8_sm100a.py`, `MINIWORLD_TRANSITION_FP8=calib` / `MINIWORLD_FP8=safe` / backend `engine-fp8`) is documented in `docs/b200/fp8.md`;
off by default, the default route above is bit-identical with it off.
