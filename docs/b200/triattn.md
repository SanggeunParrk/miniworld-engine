# Triangle attention on B200 (sm100)

Scope: the `"cuda"` backend of `TriangleAttention`, `TriangleAttentionAF3`, `TriangleAttentionRF3`, `TriangleAttentionChai1`,
`TriangleAttentionChai1Confidence` and `BidirectionalTriangleAttention` (`modules/triangle.py`) on an NVIDIA B200 (capability (10, 0),
card key `b200_sm100`). The requirement rows are the H100 page's ([../h100/triangle_attention.md](../h100/triangle_attention.md)): every
class `master` only (bf16 activations, fp32 master parameters), `eval` and `train`, starting and ending node, L 128-768 (train) and
128-2048 (eval; d_pair 384 also 2976 / 3968).

## Routing

`_TriangleAttentionBase._cuda_route` sends a call on capability (10, 0) to `_cuda_route_sm100` (self-attention) or, for the bias-only
form (`use_self_attention=False`), to the same `bias_only` route as H100 / A100. The H100 / A100 / RTX 5090 routes are unchanged.

| route | what runs | serves |
|---|---|---|
| `sm100` | `kernels/integrations/triattn_b200.py` -> `kernels/triangle_attention/cuda/b200_triattn.py` (`b200_sources/*.cu`, tcgen05 / TMEM / TMA): the whole update (LN, projections, core, gate, out-projection(s), residual, dropout), forward and backward, as opaque custom ops | bf16 pair, L >= 16 (any L: ragged lengths read / write the caller's tensors through bounded tensor maps), any B (chunked by `plan`), bool `[B, L]` key mask, no q / k norm |
| fused | `tri_front` / `triattn_fwd` / `tri_tail` (+ `tri_gate_bwd`, `triattn_bwd`, `tri_head_bwd`, `tri_wgrad`) | d_pair = d_hidden = 128, 4 heads x 32, no projection biases, standard output; AF3's ending-node bias orientation for inference only |
| wide | `tri_wfront` (2-CTA) / core / `tri_wtail` (+ `tri_scale_rows`, `tri_wgbwd`, core backward, `tri_db_rows`, `tri_whbwd`, cuBLAS, `tri_wfinish`) | d_pair 64 .. 512 (multiple of 64), heads of 16 (zero-padded to 32), 32, 64 (`triattn_dh64_sm100.cu`), 96 (padded to 128) and 128 (`triattn_dh128_sm100.cu`) channels, <= 16 heads, heads x slot <= 512; every variant: AF3 / RF3 / Chai-1 bias orientation, RF3 gate / out biases, Chai-1 untransposed / dual output |
| `bias_only` | `kernels/integrations/bidir_triattn_h100.py` with its glue / gate backward built for sm_100a (`kernels/triangle_attention/cuda/b200_bidir.py`), cuBLAS GEMMs | `BidirectionalTriangleAttention` and `TriangleAttention(use_self_attention=False)`, as on H100 |

Coverage of the requirement rows:

| class | rows (d_pair, heads x head width) | eval | train |
|---|---|---|---|
| `TriangleAttention` | 128, 4 x 32 | fused | fused |
| `TriangleAttention` | 64 (2 x 32, 4 x 16, 4 x 32), 256 (8 x 32), 384 (12 x 32) | wide | wide |
| `TriangleAttention` bias-only | 128, 4 x 32 | bias_only | bias_only |
| `TriangleAttentionAF3` | 64 (4 x 16), 128 (4 x 32), 256 (8 x 32), 512 (8 x 64) | fused at 128, else wide | starting 128 fused, else wide |
| `TriangleAttentionRF3` | 64 (4 x 64), 128 (4 x 32), with gate / out biases | 128: fused (since round 3, the biases in `tri_tailg`), 64: wide | wide |
| `TriangleAttentionChai1` | 64 (4 x 32), 256 (4 x 64), untransposed output | wide | wide |
| `TriangleAttentionChai1Confidence` | 256 (4 x 64), dual output | wide | wide |
| `BidirectionalTriangleAttention` | 128 | bias_only | bias_only |

Every requirement case routes (no `unsupported` record on the B200). `backend_versions["cuda:b200_sm100"] = 1` on
`_TriangleAttentionBase` (inherited by the five classes) and on `BidirectionalTriangleAttention`.

Kernel history, design and the earlier measurements: `docs/gpus/b200/triattn/triattn.md` on the branch `origin/b200/ladder-final`
(the kernels and the dispatch come from that branch unchanged, adapted to the 2.5.x module API: no `settings` (the deterministic fused
weight gradients are `MINIWORLD_TRIATTN_B200_DETERMINISTIC_WGRAD=1`), no `kernels._gemm` (`b200_triattn._mm32` keeps cuBLAS 12.9's
split-K tail fix), the route instead of the old `PORTABLE` backend gate).

## Measured on the B200 (2026-11-16)

NVIDIA B200 (148 SMs), torch 2.13.0+cu129, kernels built by nvcc 12.9 for sm_100a; GPU 1 of the shared node (other agents' runs on the
same GPU: accuracy in shared mode, speed in exclusive mode). Records: `$W/b2511/agents/triattn/_b2/out/acc_<class>.jsonl`,
`spd_<class>.jsonl` and `spdE_TriangleAttention{,AF3}.jsonl` (eval re-measured after the inference pack caching), baselines (anthropic /
cueq) from `$W/b2511/out_base/speed.*.jsonl` (GPU 0). Every requirement row, every length, `master`, eval and train, `depths: min`,
`samples: min`.

### Accuracy (`measure.accuracy`, lights vs the fp64 gold, `tol = max(err_pytorch, 8e-3)`)

| class | cuda GREEN | YELLOW | RED | unsupported | skipped | triton GREEN / YELLOW |
|---|---|---|---|---|---|---|
| `TriangleAttention` | 161 | 10 | 0 | 0 | 31 | 155 / 16 |
| `TriangleAttentionAF3` | 96 | 4 | 0 | 0 | 8 | 93 / 7 |
| `TriangleAttentionRF3` | 48 | 4 | 0 | 0 | 8 | 48 / 4 |
| `TriangleAttentionChai1` | 41 | 11 | 0 | 0 | 8 | 36 / 16 |
| `TriangleAttentionChai1Confidence` | 23 | 3 | 0 | 0 | 4 | 16 / 10 |
| `BidirectionalTriangleAttention` | 11 | 1 | 0 | 0 | 0 | (no triton backend) |

Every YELLOW is a training gradient (or the bias-only form's) at 1.00-1.36x its tol (31 of 33 under 1.22x; the largest: AF3 d_pair 256
ending node L128 `grad:to_query.weight` 1.33x, Chai-1 d_pair 64 ending L512 `grad:ln_pair.weight` 1.36x): the bf16 rounding spread, as on
the other cards (H100 tokens: 50 cuda YELLOW up to 1.43x, A100: 74 up to 1.98x) and fewer than triton's on the same cases. Skipped =
the fp64 gold / reference does not fit (every eval case at L >= 1536, d_pair 256 / 384 at L >= 1024; skipped for every backend), plus
four cases (r72 L640 / L768, r245 L768) whose gold ran out of memory because other agents held ~160 GB of GPU 1 (re-run, same result).
The GPU test `tests/gpu/test_triattn_b200.py` (71 cases) checks the same geometries against fp32 at L128 / ragged L200 with B = 2.

### Speed (`measure.speed`, CUDA graph, median ms)

cuda is faster than triton and than pytorch in all 472 timed cases (eval and train, every class): triton / cuda 1.18-4.64x,
pytorch / cuda 1.63-36x. Against the best of anthropic / cueq (where they run): 2.5-6.1x in training, 1.8-5.7x in inference except
`TriangleAttention` / `TriangleAttentionAF3` d_pair 256 x 8 heads, ending node, L128 eval (0.95x / 0.89x of anthropic: 0.086 vs 0.082 ms).

| class, row | L | mode | cuda | triton | pytorch | anthropic | cueq |
|---|---|---|---|---|---|---|---|
| TriangleAttention 128 x 4 (r74) | 384 | eval | 0.176 | 0.666 | 2.415 | 0.330 | 1.001 |
| TriangleAttention 128 x 4 (r74) | 768 | eval | 0.915 | 3.380 | 16.61 | 1.744 | 5.268 |
| TriangleAttention 128 x 4 (r243) | 384 | train | 0.657 | 2.152 | 5.404 | - | 3.470 |
| TriangleAttention 128 x 4 (r243) | 768 | train | 3.409 | 10.95 | 35.61 | - | 20.12 |
| TriangleAttention 64, 4 x 16 (r72) | 128 | eval | 0.033 | 0.094 | 0.168 | 0.039 | - |
| TriangleAttention 256 x 8 (r75) | 768 | eval | 1.797 | 6.086 | 32.50 | 3.747 | 9.863 |
| TriangleAttention 384 x 12 (r76) | 2048 | eval | 43.73 | 126.3 | - | 111.8 | 230.6 |
| TriangleAttention 384 x 12 (r76) | 3968 | eval | 308.2 | 867.3 | - | - | 1532 |
| TriangleAttentionAF3 512, 8 x 64 (r80) | 768 | eval | 3.952 | 8.846 | 36.36 | - | 14.23 |
| TriangleAttentionAF3 512, 8 x 64 (r80) | 768 | train | 14.81 | 30.60 | 80.60 | - | 52.79 |
| TriangleAttentionRF3 128 x 4 (r249) | 768 | train | 3.858 | 11.38 | 71.66 | - | 20.55 |
| TriangleAttentionChai1 256, 4 x 64 (r84) | 2048 | eval | 22.89 | 56.98 | - | 57.40 | 91.70 |
| TriangleAttentionChai1Confidence (r252) | 768 | train | 6.916 | 16.78 | 78.58 | - | 27.96 |
| BidirectionalTriangleAttention (r142) | 768 | train | 2.144 | - | 6.958 | - | - |
| TriangleAttention bias-only (r188) | 768 | train | 1.726 | 3.982 | 4.458 | - | - |

Inference weight packs (fused and wide path) go through `_capture.param_forms(capture_reads_eager=True)`, the H100 path's contract: a
captured inference step is served the pack its eager warm-up made, so a replay runs no packing kernel (d_pair 64 x 4 x 16 L128 eval
0.076 -> 0.031-0.033 ms; the branch's own capture-scoped cache repacked at every replay). Training packs stay capture-scoped.

## Optimization round (2026-11-17)

Branch `b200opt-triattn`. `backend_versions["cuda:b200_sm100"]`: `_TriangleAttentionBase` 1 -> 3 (inherited by the five
`TriangleAttention*` classes), `BidirectionalTriangleAttention` 1 -> 2. Records: `$W/b2511/agents/opt-triattn/_o/out/`
(`spd_<class>.jsonl` = every requirement case, cuda + the pytorch reference, after the changes; `acc_core.jsonl`, `acc_bo_v2.jsonl`;
`sum_before.txt` / `sum_after.txt` = the summaries below; "before" = the speed tokens of the ladder round in
`$W/eng-2511/src/miniworld_engine/judge/tokens/b200_sm100/`).

**Ladder** (`docs/ladder.md`, the coordinator's ladder of 2026-11-16 with the B200 Triton tables): no violation in any of the six
classes before or after (cuda < triton < pytorch in every timed case). The only miss against an external baseline was d_pair 256 x 8
heads, ending node, L128 eval (`TriangleAttention` r75 0.95x, `TriangleAttentionAF3` r79 0.89x of anthropic); fixed below.

### 1. Head-dim-32 core: overflowing tile sums shift P instead of recomputing S (cuda:b200_sm100 2)

The ending-node anomaly was not the 4-D tensor maps: a kernel profile showed the attention core at 45 us against 14 us for the
starting node of the same shape (front 18 us, tail 16-17 us in both). The core's softmax offset is the maximum of a row's first 32
keys and is raised only after a tile whose exponentials sum past 2^64; that branch recomputed the tile's raw S from q / k in global
memory, one fp32 dot product per key per row thread (32 x 128 FMAs). The measurement inputs of the ending-node case have a wider logit
spread and took the branch; a synthetic sweep (q / k / bias scaled) showed the core 3.2x slower at a moderate spread (q x3, bias x10)
and 11x (L128) / 20x (L768) at an extreme one.

Now a tile whose sum overflows but stays finite (< 2^127, so every p of the tile is finite too) raises the offset by the sum's binary
exponent e and scales the tile's P in TMEM by 2^-e (exact in bf16: P is the same as with the higher offset from the start), and l_r,
O_r by 2^-e as before (`TA_PSHIFT`, `triattn_sm100.cu`). Only a sum that is inf / NaN (a logit more than ~127 log2 units, ~88 nats,
above the first 32 keys' maximum) still takes the recompute. The steady state is unchanged (the code sits in the warp-uniform rare
branch): 14.4 us at q x1 / bias x1 before and after.

| case | before | after | anthropic |
|---|---|---|---|
| r75 `TriangleAttention` d_pair 256 x 8, ending, L128 eval | 0.084-0.086 ms | 0.055 ms | 0.082 ms |
| r79 `TriangleAttentionAF3` d_pair 256 x 8, ending, L128 eval | 0.092 ms | 0.055 ms | 0.082 ms |
| core alone, L128, 8 heads, q x3 / bias x10 | 46.2 us | 14.4 us | |

Accuracy (`acc_core.jsonl`: rows 72, 74, 75, 76, 79, 243, 244 at L128 / 384 / 768, eval and train): 53 GREEN, 1 YELLOW (r79 L128
training `grad:to_query.weight` 1.33x tol, the same value as before the change). Against fp32 at spreads up to q x10 / bias x100:
relative error 0.8-2.3e-3, no NaN, run-to-run bitwise. Tried and reverted: a 32-log2 headroom on the first offset (it widens the
cheap window to ~159 units but changed no measured case).

### 2. Bias-only forms: dropout and residual in one pass, no batch-buffer copies (cuda:b200_sm100 3 / Bidirectional 2)

A profile of `TriangleAttention(use_self_attention=False)` training at L768 showed the dropout of the update as broadcast torch ops:
two non-vectorised multiplies of 180 us each (forward update, backward dy), an add of 68 us and copies of the update / the fp32 LN
input gradient into the batch buffers (~150 us), against ~75 us for each GEMM. On the B200 (`_b200()`-gated in
`bidir_triattn_h100.py`; H100 / A100 / RTX 5090 run the previous statements) the forward now writes `x + update o drop` with the wide
path's `tri_scale_rows` (given an optional residual: one fp32 FMA, one rounding), the backward scales dy with the same kernel, and at
B = 1 the LN input gradient is used without the copy (also in `BidirectionalTriangleAttention`).

| case (r188, d_pair 128 x 4, master, train) | L128 | L384 | L768 |
|---|---|---|---|
| starting, before -> after (ms) | 0.173 -> 0.159 | 0.509 -> 0.411 | 1.726 -> 1.298 |
| ending, before -> after (ms) | 0.220 -> 0.204 | 0.585 -> 0.481 | 1.882 -> 1.433 |
| triton (ms), starting / ending | 0.235 / 0.259 | 1.069 / 1.233 | 3.982 / 4.682 |

Bidirectional training L768 2.14 -> 2.04 ms. Accuracy (`acc_bo_v2.jsonl`) unchanged (the same errors as before at every case); with
dropout active (the accuracy measurer runs without it) against the fp32 statements on the same draw: y / dx / dWo / dgamma within
1e-3 of the bf16 PyTorch statements' own error, bitwise deterministic run to run.

### 3. On engine 2.5.12's bias-only structure (2026-11-17, cuda:b200_sm100 4 / Bidirectional 3)

The merge of origin/main 2.5.12 replaced `bidir_triattn_h100.py` (planes GEMM `cgemm`, `logits`, the fused gate + out-projection `tail`,
`gate_bwd_fused`, `softmax_bwd_planes`, `_mm_f32_into`). Its new kernels are mma.sync / cuBLASLt code (no sm_90a-only instruction): they
build for sm_100a through `b200_bidir` unchanged and are correct on the B200 (accuracy as before: r188 10 GREEN / 2 YELLOW, r142 5 / 1,
the same errors; `tests/gpu/test_triattn_b200.py` 71 passed; dropout check vs fp32 at the bf16 statements' error, bitwise
deterministic). The dropout pass of change 2 is re-applied B200-only on top (the fused tail takes no dropout scale; the B = 1 copy is
gone upstream through `_mm_f32_into`). Records `$W/b2511/agents/opt-triattn2/_o/out/`.

| case (master, train unless noted) | 2.5.12 glue on B200 | + dropout pass |
|---|---|---|
| r188 starting L128 / L384 / L768 | 0.150 / 0.464 / 1.584 ms | 0.140 / 0.390 / 1.253 ms |
| r188 ending L128 / L384 / L768 | 0.194 / 0.532 / 1.717 ms | 0.179 / 0.452 / 1.364 ms |
| r188 eval starting / ending L768 | 0.413 / 0.422 ms (was 0.463 / 0.483) | unchanged |
| r142 eval / train L768 | 0.643 / 1.973 ms (was 0.727 / 2.04) | (no dropout) |

### Summary after the round (cuda speedup, min / geomean over the timed requirement cases)

| class | vs triton | vs pytorch | vs anthropic | vs cueq |
|---|---|---|---|---|
| `TriangleAttention` | 1.23 / 2.76 (was 1.16 / 2.73) | 1.74 / 8.66 | 1.21 / 2.31 (was 0.95) | 2.79 / 5.05 |
| `TriangleAttentionAF3` | 1.86 / 2.64 | 2.91 / 8.91 | 1.21 / 1.72 (was 0.89) | 2.65 / 4.65 |
| `TriangleAttentionRF3` | 1.84 / 2.64 | 4.38 / 13.4 | 1.84 / 2.31 | 3.04 / 4.54 |
| `TriangleAttentionChai1` | 2.11 / 2.82 | 3.90 / 14.1 | 2.51 / 3.94 | 2.76 / 4.81 |
| `TriangleAttentionChai1Confidence` | 2.09 / 2.67 | 3.58 / 10.3 | 2.51 / 3.89 | 2.60 / 4.16 |
| `BidirectionalTriangleAttention` | (no triton) | 2.31 / 3.54 | - | - |

GPU test `tests/gpu/test_triattn_b200.py`: 71 passed. CPU: ruff / ruff format / pyright clean on the changed files; `tests/cpu`
judge / requirements / modules / imports: 2650 passed.

## Round 3 (2026-11-17): against the H100 `cuda` medians

Target of the round: on every requirement case both cards serve, B200 `cuda` median <= H100 `cuda` median / 1.5 (ratio b200 / h100
<= 0.667; graph timing, L2 flushed; cases whose H100 median is under 20 us are exempt: none of these classes has one). Branch
`b200r3-triattn`. `backend_versions["cuda:b200_sm100"]`: `_TriangleAttentionBase` 4 -> 5. Records:
`$W/b2511/agents/r3-triattn/_r/out/` (`spd_<class>_v17.jsonl`: every requirement case, cuda, after the changes; `spd_tr_v18.jsonl`: training L128 / L256 on the final tree; `cb*.log`: the
core's variants; `ftl*.log`: the time buckets; `acc_core_v17.jsonl`; `packchk.log`); "before" = the speed tokens in
`$W/eng-2511/src/miniworld_engine/judge/tokens/b200_sm100/`, the H100 medians from `tokens/h100_sm90/`.

### What bounds the forward core (head dim 32, L768, 4 heads: 661 us)

Compile-time ablations of `triattn_fwd_sm100` (`TA_NOKV`, `TA_NOBIAS`, `TA_NOMMA`, `TA_FNOMATH`, `TA_NOEX`, `TA_FNOLD`,
`TA_FNOST`, `TA_FNOLOOP`: off by default) and clock64 time buckets of one CTA's warps (`TA_FTL`, `triattn_ftl`):

| variant (L768, 4 heads, graph, us) | time |
|---|---|
| as it was | 661 |
| no exponential (an FFMA2 instead) | 660 |
| no logits, no exponential | 560 |
| ... and no K / V / bias loads | 560 |
| ... and no MMAs (commits only) | 471 |
| ... and no TMEM loads / stores, no softmax arithmetic at all | 359 |
| the same skeleton with carried indices (no divisions) | 285 |

The kernel was not bound by the MUFU, the FMA pipe, TMA or TMEM traffic but by its one MMA warp: per 128 x 128 sub-step it spent
~1500 clk issuing (ten MMAs at ~70 clk, two commits, two barrier probes at ~90 clk and four runtime divisions for the sub-step's
indices) against ~1150 clk of softmax work a group, and the two softmax groups waited for their next S a quarter of the time. Two
changes, both with the same output bits:

- **Carried indices**: the issue streams keep (task, stage, row, ring slot, phase) as counters instead of dividing the flattened
  sub-step index by a runtime `SPT`: 661 -> 638 us.
- **Two issue warps** (`TA_SPLIT`, warp 14): one warp issues QK^T, another the PV GEMMs; QK^T(y) waits for PV(y - 3) to retire
  through the K / V stage barrier (the streams are ordered by barriers instead of one thread's issue order; a commit tracks the
  issuing thread's own MMAs): 638 -> 625 us at L768, 11.36 -> 10.88 ms at L2048, unchanged at L <= 384. The softmax groups now
  wait ~5 % of the time. The head-dim-64 core (`triattn_fwd64_sm100`, four QK^T MMAs of 16 KiB operands a sub-step) had the same
  structure and gains more: L768 4 heads 1115 -> 811 us, L256 55.0 -> 49.3 us.

With the issue path off the critical path the softmax groups are the bound (work ~2600 clk of a group's ~3100 clk a sub-step).
Measured on top of the two issue warps and not kept (L768, us; 625 as it is): 4 / 8 / 12 of 16 exponential pairs as an FMA-pipe
polynomial 700 / 834 / 900; predicated issue loops with precomputed descriptors (the backward kernels' form) 639; barrier waits with a
suspend hint (every warp class) 626-627; the bias pair widened by a byte permute instead of the shift ptxas turns into `IMAD.U32`
631; P packed by a byte permute (truncation) 623; the PV issue warp on the fourth sub-partition (warp 15) 625. Upper bounds
from ablations: no exponential 490, no bias add and no row-sum add 539, so moving the bias and the row sums to the tensor core
(the H100 kernel's form) is worth at most ~14 %. Rows per task: 2 up to L384 and 4 above is still the best split (L128 / 256 / 384
/ 512: 10.4 / 32.9 / 94.6 / 206.8 us with 2, 11.5 / 37.0 / 94.5 / 200.5 with 4).

### The wide path's weight packs in one launch

`pack_wide_weights` was torch statements (casts, zero fills, cats, transposes): a dozen and a half launches, which a training
capture replays at every step (training packs are capture-scoped). `tri_wpack` (`triattn_wide_sm100.cu`) writes `wcat`, `wo_p`,
`wo_t`, `wall_t` (and the dual output's `wo2_p`) in one launch of 3.4 us, bitwise equal to the statements (`_r/packchk.py`: twelve
widths x fp32 / bf16 x single / dual output, and mixed dtypes through fp32). d_pair 64 x 4 x 16 L128 training step under
`torch.profiler` (CUDA graph, warm L2): 153.9 -> 114.3 us.
The wide backward no longer zero-fills its bias-gradient output for a module without gate / out biases (it is never read then).

### Result (every requirement case, cuda, graph, L2 flushed; ratio = B200 median / H100 median, target <= 0.667)

| class | cases | over 0.667 before -> after | worst before -> after | geomean before -> after | B200 after / before (geomean) |
|---|---|---|---|---|---|
| `TriangleAttention` | 202 | 93 -> 88 | 1.025 -> 1.021 | 0.650 -> 0.635 | 0.977 |
| `TriangleAttentionAF3` | 108 | 44 -> 41 | 0.994 -> 0.989 | 0.626 -> 0.603 | 0.964 |
| `TriangleAttentionRF3` | 60 | 24 -> 18 | 1.039 -> 1.038 | 0.589 -> 0.545 | 0.925 |
| `TriangleAttentionChai1` | 60 | 18 -> 14 | 0.859 -> 0.855 | 0.544 -> 0.505 | 0.930 |
| `TriangleAttentionChai1Confidence` | 30 | 0 -> 0 | 0.583 -> 0.557 | 0.436 -> 0.399 | 0.915 |
| `BidirectionalTriangleAttention` | 12 | 2 (not re-measured: no change) | 0.773 | 0.514 | - |

The 161 cases still over 0.667 (`_r/out/vs_final.txt`): inference L >= 512 96 (0.67-0.82), L256 / L384 27 (0.67-0.82), L128 23
(0.69-1.02), training L128 15 (0.67-0.81; 8 of them within one 2.05 us timer step of the target). No case is slower than before
beyond a timer step. Accuracy: every change writes the same bits (`cb*.log`: err 0 against the previous build at L128 / 256 / 768
/ 2048; `packchk.log`); `acc_core_v17.jsonl` 53 GREEN, 1 YELLOW (r79 L128 training `grad:to_query.weight` 1.32x tol, as before);
`tests/gpu/test_triattn_b200.py` 71 passed; CPU judge / requirements / modules / imports / prebuild plan 2673 passed. The ladder
(cuda < triton < pytorch) is untouched: no case got slower.

### Part 2 (2026-11-17 afternoon): three softmax groups (cuda:b200_sm100 6)

`triattn_fwd_sm100` built with `TA_R=3` (`G3`): three softmax groups of four warps (warps 2-13; the former epilogue warps 10-13
are the third group), each owning one pair row of the task and one 128-column S / P buffer (group = row = buffer), and each
draining its own row at the end of the task (no epilogue warps, no row-sum handoff; the next task's first PV of the row waits for
the group's `pf` arrive, which comes after its drain in program order). A sub-partition now runs three softmax warps instead of
two, so its MUFU unit keeps working while one group waits for its PV + next QK^T. Same arithmetic per row as the 2 / 4-row builds:
the same output and LSE bits (`_r/out2/g3a.log`, `g3c.log`: err 0 against the 4-row / 2-row builds at L384 / 512 / 768 / 2048).
`_rows_per_task`: 2 up to L256, 3 from L384 on (core, 4 heads, graph, us):

| L | 2 rows | 4 rows | 3 rows (G3) |
|---|---|---|---|
| 128 | 10.3 | - | 12.4 |
| 256 | 32.9 | - | 34.9 |
| 384 | 94.4 | - | 90.3 |
| 512 | 206.7 | 201.0 | 193.0 |
| 768 | 631.6 | 625.4 | 583.4 |
| 2048 | 11067 | 10897 | 10664 |

Measured on top of G3 and not kept (L768, us; 583 as it is): 1 / 2 / 4 of 16 exponential pairs on the FMA pipe 598 / 600 / 674;
K / V ring of 4 / 6 stages 586 / 591; S / P in 64-key halves (six half buffers, per-half S-ready / P-ready / PV-done barriers,
QK^T of a half issued once that half's previous PV retired; with an unrolled half loop and the second half's first TMEM load
issued during the first half's last chunk) 628-633 (L2048 11.04-11.10 ms): the softmax groups' wait for S fell from ~18 % to ~6 %
of their time (clock64 buckets, `_r/out2/ftl_g3.log` / `ftl_hv.log`), but the per-half overhead (two votes / fences / arrives a
sub-step, N = 64 QK^T reading Q twice) cost more than the wait it hid (the code is in the scratch file
`halves_final_rejected.cu` of the session, not on the branch).
Ablations of G3 (L768, us): no exponential 494, no logits and no exponential 389, no softmax arithmetic at all 320, no TMEM loads
584, no TMEM stores 572, no bias loads 582. So the logits (bias unpack + two FFMA2 a pair) cost ~100 us, the exponentials ~90, the
row sum + pack ~70, and the skeleton (barriers, TMEM traffic, the S -> PV -> QK^T chain of each group) ~320.

**MUFU arithmetic** (L768 x 4 heads): 768^3 x 4 = 1.81e9 exponentials; one `ex2.approx.f32` a logit at 16 / clk / SM (a warp
instruction per 8 clk a sub-partition) on 148 SMs at ~1.9 GHz = 4.5e12 / s -> 0.40 ms floor. 583 us = 69 % MUFU busy. 0.667 x the
H100 needs ~0.52 ms at this size (the H100 runs 128 exponentials per row-tile on fp16 pairs, bias and row sum on its tensor cores,
which here would need the bias in TMEM before QK^T: an fp32 64 KiB tile per key tile and row through `tcgen05.cp`, more shared-memory
feed per sub-step than the tensor pipe has left).

Result (every requirement case, cuda, graph, L2 flushed; records `_r/out/spd_<class>_v19.jsonl`, `_r/out2/vs_v19v.txt`; "before"
= the round's start tokens, as in the table above):

| class | cases | over 0.667 before -> part 1 -> part 2 | worst part 2 | geomean part 2 | B200 part 2 / part 1 (geomean) |
|---|---|---|---|---|---|
| `TriangleAttention` | 202 | 93 -> 88 -> 79 | 1.025 | 0.626 | 0.986 |
| `TriangleAttentionAF3` | 108 | 44 -> 41 -> 41 | 0.994 | 0.596 | 0.987 |
| `TriangleAttentionRF3` | 60 | 24 -> 18 -> 18 | 1.036 | 0.541 | 0.994 |
| `TriangleAttentionChai1` | 60 | 18 -> 14 -> 4 | 0.857 | 0.502 | 0.992 |
| `TriangleAttentionChai1Confidence` | 30 | 0 -> 0 -> 0 | 0.555 | 0.398 | 0.998 |
| `BidirectionalTriangleAttention` | 12 | 2 (no change) | 0.773 | 0.514 | - |

Per length (part 2 / part 1, geomean over the classes' cases): eval L384 0.985, L512 0.984, L640 0.976, L768 0.969, L1024-2048
0.978, L2976 / 3968 0.89; training 0.995-0.998; L128 / 256 unchanged (within a timer step). Accuracy `acc_core_v19.jsonl`: 53 GREEN,
1 YELLOW (r79 L128 training `grad:to_query.weight`, as before); `tests/gpu/test_triattn_b200.py` (new: the three-row core at L400
with a padding row, inference and training; determinism / graph at L384) 80 passed. Still over 0.667: 142 (TriangleAttention 79,
AF3 41, RF3 18, Chai1 4) + Bidirectional 2: eval L128 23, training L128 15, eval L >= 256 104.

### Part 2b: the training dropout draw in one launch (cuda:b200_sm100 7)

The B200 paths drew the training dropout scale with the module's torch statements (`torch.rand` under capture: the draw and two
generator-state fills, then compare, cast, divide). The H100 draws it with one Philox kernel (`h100_sources/pack.cu`
`drop_scale`, the same Philox subsequences / counter advance / bf16 rounding as `torch.rand`: bitwise the same tensor); that source
is plain CUDA and is now also built for sm_100a (`b200_triattn._pack`, extension `triattn_b200_pack_v1`, in the sm100 prebuild
plan; the H100 build is unchanged) and used by the fused / wide paths (`integrations/triattn_b200.py`) and the bias-only form
(`bidir_triattn_h100.forward_single`, under `_b200()`). `test_drop_scale_is_the_torch_draw`: bitwise equal to the torch draw for
both broadcast axes and two p. Training L128 -3 to -7 us a step (geomean 0.970; L256 0.988; `_r/out/spd_tr_v20.jsonl`), bias-only
training L128 0.1824 / 0.1404 -> 0.1779 / 0.1349 ms (ending / starting; `spd_bo_tr_v20.jsonl`). Training cases over 0.667: 15 -> 10.

State after part 2b (`_r/out2/vs_v20bv.txt`): 137 + 2 (Bidirectional) over 0.667 -- TriangleAttention 76, AF3 39, RF3 18, Chai1 4;
**inference 127** (L128 23, L >= 256 104), **training 10** (all L128: d_pair 128 x 4 fused 3 at 0.78-0.79, bias-only 2 at 0.72 /
0.76, d_pair 64 x 2 x 32 2 at 0.74-0.75, d_pair 64 x 4 x 16 (d_hidden 128) 3 at 0.68-0.69). Slower than the H100: 2 (eval L128: RF3 d_pair 128 starting 1.036, d_pair 64 x 2 x 32 starting 1.025).

### Round 3, fronts and tails (2026-11-17, branch `b200r3-triattn-tail`, cuda:b200_sm100 8)

Scope: the fronts, tails and their Python routing (the cores are `b200r3-triattn`'s). Records `$W/b2511/agents/r3-triattn-tail/_t/out/`
(`spd_inf.jsonl` / `spd_trn.jsonl`: every case of the changed rows r71-74, r77, r78, r82, r83 eval and r72, r241, r242, r246, r250
train; `acc_inf.jsonl` / `acc_trn.jsonl`; `p768*.log` / `e2*.log`: kernel profiles and ablations; `chk1*.log` / `chk2.log` /
`chk3.log`: the checks below). "Before" = part 2b's records (`r3-triattn/_r/out/spd_<class>_v19.jsonl`, `spd_tr_v20.jsonl`) over the
round-start tokens.

**1. Inference tail that projects the gate from x (the H100 inference tail), taken.** `tailg_sm100.cuh` (`tri_tailg` in the mod
extension for d_pair 128, `tri_wtailg` in the wide one for d_pair 64 with heads x slot 64 / 128): `y = LN(x)` (the front's
arithmetic, two-pass fp32 statistics, one bf16 rounding), `g = y Wg^T` in fp32 in TMEM (never rounded or stored), `u = sigmoid(g [+
gb]) o o`, `out = x + u Wo^T [+ bo]`; the front writes q / k / v only (`tri_front<.., NOUT = 3>`, wide `nog`). Design after three
attempts (L768, d_pair 128 / 64 x 2 x 32, kernel us):

| tail | d128 | d64 x 2 x 32 |
|---|---|---|
| before (front writes g, `tri_tail` / `tri_wtail` read it) | 99.0 | 47.2 |
| v1: one y / u tile in shared memory, x / o stages of two tiles (serial chain LN -> gate GEMM -> u -> out GEMM per tile) | 102.9 | 64.7 |
| v3: y and u in TMEM (A operands of `tcgen05.mma`), one tile of LN / gate ahead of the GEMMs, residual re-read from global | 121.4 | 56.6 |
| v4 (kept): v3 with the residual from the tile's x slot and the result written over it, per-warp TMA stores of [32][64] | 98.8 | 48.4 |

Ablations of v3 (`TG_NO*`, timing only): no residual loads 80.1 / 46.5, no drain 71.3 / 40.7, no LN / gate arithmetic 114 / 49: the
per-row global residual loads (32 rows a warp instruction) were a third of the kernel. Module, L768 eval graph (`profs.sh`): d_pair 128
830 -> 793 us (front 148.5 -> 111.5), RoseTTAFold3 d_pair 128 904 -> 800 (see 3), d_pair 64 x 4 x 16 789 -> 736, 64 x 2 x 32 458 ->
409 (with 4), AF3 d_pair 64 ending 810 -> 760. At L128 (one tile a CTA) the longer per-tile chain lost 2 us on the d_pair 128 path
(34.7 -> 36.7 us), so d_pair 128 without biases keeps the 4-output front + `tri_tail` at L128. Accuracy: the update against the fp32
module (`chk1`, 37 cases, both nodes, AF3 / RF3 orientation, ragged L200, masks, B 2-3) at the previous tail's error (max 6-11e-3 of
the update, the difference between the two at one bf16 step), bitwise run to run.

**2. Compact 16-channel heads, partly.** In inference the gate stream is gone (1), so the remaining padded streams are q / k / v
written by the front and o read by the tail. Keeping q / k / v compact needs the core's loads to zero-pad (the H100 core's 3-D map
`[rows][H][16]` with a 32-channel box): a change in `triattn_sm100.cu`'s maps and load coordinates (the core owner's file; not done
here). The d_pair 64 x 4 x 16 rows got 1 and 4 instead: L768 eval 803 -> 744 us (ratio 0.747 -> 0.692).

**3. RoseTTAFold3 d_pair 128 on the fused path (inference), taken.** `_kind` sends RF3's d_pair 128 inference (gate / out biases) to
the fused path: the front is bias-free (no g), the biases are added in `tri_tailg` in fp32 before each rounding. L768 0.920 -> 0.808 ms
(ratio 0.762 -> 0.669), L128 starting 0.0387 -> 0.0346 ms (1.036 -> 0.927: no longer slower than the H100). Training stays wide (its
backward would need dbg / dbo in the fused gate backward).

**4. Fronts.** The wide front at d_pair 64 (2-CTA, weights streamed per 64-row block, ~3.3 TB/s at 64 x 2 x 32) is replaced by the
fused front's design (`tri_front_sm100<CX, HD, NOUT>`, now templated on d_pair 64 / 128 and heads x 32 = 64 / 128: one CTA, the weights
resident, the bias projection on the LayerNorm warps' CUDA cores, LN statistics for the backward written by the same warps;
`tri_front64`), inference and training: L768 64 x 2 x 32 front 113 -> 64.5 us, 64 x 4 x 16 137 -> 93 us. The fused front's drain is
its bound (ablation at d128: no drain 88 us, no stores 116, no LN 131, no MMA 134 of 148). Folding the LayerNorm into the weights (the
H100 d256 wide front) was not built: the B200 wide front at d256 is write-bound (q | k | v | g + bias 1.2 GB of its 1.5 GB at L768, 300
us = 5 TB/s), the fold only moves arithmetic. No launch the H100 lacks remains in inference (pack cached; front, core, tail).

**5. Training, measured, not changed beyond 4.** Launches per training step under a graph (profile, d_pair 64 x 4 x 16 L128): forward
wpack, drop_scale, front, core, tail; backward scale_rows, wgbwd, dbias fill, biasT, bwd_kv, bwd_q, db_rows, whbwd, two cuBLAS GEMMs
(+ split-K reduces), wfinish (+ sum): ~20, the H100 width-generic step ~20 (pack, ln_fwd, GEMM, front_bias, core, out GEMM, residual,
drop_scale; gate_bwd_in, du / dWo GEMMs, gate_bwd_out, bwd_kv, bwd_q, slab sum, assemble, two GEMMs, ln_bwd, torch sums). At L128
(128 tiles: one tile a CTA) every kernel is a single tile's latency (d128 fused: wgrad 17.5, gate_bwd 15.0, head_bwd 11.1 us); the
count is already at the H100's. Training d_pair 64 rows on `tri_front64`: L128 0.108 -> 0.106 ms (64 x 2 x 32), 0.124 -> 0.121 (64 x 4
x 32); training over 0.667 5 -> 5 of the changed rows. The fused d_pair 128 path's dWo / weight gradients use fp32 atomics by default
(`DETERMINISTIC_WGRAD` off: `chk3` shows `to_out.weight` differing by 2.7e-5 between two runs; the wide path and `tri_front64`
training are bitwise deterministic) -- unchanged here, listed under Gaps. The bias-only training L128 (0.72 / 0.76) and
BidirectionalTriangleAttention's per-head dP were not reached.

Result on the changed rows (cuda, graph, L2 flushed, ratio = B200 / H100 median; `_t/cmp2.py`):

| | cases | over 0.667 before -> after | worst before -> after | B200 after / before (geomean) |
|---|---|---|---|---|
| eval (r71-74, r77, r78, r82, r83) | 138 | 114 -> 95 | 1.036 -> 1.002 (L128 d128 ending; see 1: fixed after the run) | 0.936 |
| train (r72, r241, r242, r246, r250) | 60 | 5 -> 5 | 0.747 -> 0.734 | 0.985 |

Accuracy (`acc_inf` / `acc_trn`): eval 112 GREEN, 26 skipped (no fp64 gold at L >= 1536); train 48 GREEN, 12 YELLOW -- the same 12 cases
as before at the same ratios (1.01-1.34x tol, max Chai-1 d_pair 64 ending L512 `grad:ln_pair.weight` 1.34x, was 1.36x).
`tests/gpu/test_triattn_b200.py` 84 passed.


### Round 3, fronts and tails part 2 (2026-11-17, cuda:b200_sm100 10; 9 is `b200r3-triattn`'s)

Records `$W/b2511/agents/r3-triattn-tail/_t/out/` (`spd_t128*_v10.jsonl` / `spd_t128*_atom.jsonl`: the fused d_pair 128 training
rows r243 / r247, deterministic and atomic, L128 / L256 timed in their own measurer runs; `acc_t128_v10.jsonl`; `spd_trn*_v10.jsonl`,
`acc_trn_v10.jsonl`: the d_pair 64 training rows r72 / r241 / r242 / r246 / r250 after the merge). Kernel times from `_t/var.py`
(graph, profiler), `_t/e3.sh` / `ng.sh` / `e5.sh`.

**1. Deterministic weight gradients on the fused d_pair 128 path, by default (taken).** `DETERMINISTIC_WGRAD` now defaults on
(`MINIWORLD_TRIATTN_B200_DETERMINISTIC_WGRAD=0` opts back into the fp32 atomics): `tri_gate_bwd` writes its dWo per CTA, `tri_wgrad`
its M_t / vector sums per CTA, and `tri_slot_sum` adds the slots in CTA order. Every output and gradient is bitwise equal over three
steps (`_t/det3.py`: L128 / 200 / 384 / 768 starting, L256 ending B2, AF3 L128; dropout on). To halve the partials `tri_wgrad` runs in
NG tensor groups: CTA (split, group) accumulates 4 / NG of the four projections over the split's row tiles (its slot holds only its
projections; group 0 also sums B_h / cdb). Measured (d128 training, wgrad + slot sum, us):

| NG | L128 | L256 | L768 |
|---|---|---|---|
| 1 | 17.9 + 6.1 | 34.0 + 7.9 | 175.8 + 7.7 |
| 2 | 15.6 + 6.9 | 31.5 + 9.6 | 201.9 + 8.8 |
| 4 | 17.0 + 5.6 | 40.2 + 8.7 | 286.9 + 7.4 |

`wg_groups`: NG 2 up to 8 tiles an SM, else 1. In the deterministic form `tri_gate_bwd` is faster (L128 15.8 -> 11.7 us, no
same-address atomics); the slot sum (5.7-8 us) is the cost. Measurer, cuda graph, L2 flushed, deterministic vs atomic on r243 / r247
(24 cases): geomean +1.0 %; L128 starting 0.1226 -> 0.1249 ms, AF3 starting 0.1206 -> 0.1248, ending 0.1227 -> 0.1248; L768 +6 us of
3320 (var.py). More than one timer step at some L, so the atomic path stays as the opt-out, not as the default. Accuracy r243 / r247:
48 / 48 GREEN; the d_pair 64 training rows unchanged (the same 12 YELLOWs at the same ratios, one Chai-1 case now GREEN).

**2. Programmatic dependent launches in the d128 backward (rejected).** `tri_wgrad` does not read `tri_head_bwd`'s output, so it was
launched as its programmatic dependent (head_bwd triggers at its start; wgrad waits only before it exits), and `tri_slot_sum` /
`tri_param_finish` as dependents (`TB_PDL` bits). Graph us, d128 training, off / wgrad / sum+finish / both / + early wgrad trigger:
L128 114.6 / 114.0 / 114.1 / 113.5 / 117.7, L256 296.4 / 300.5 / 295.9 / 300.6 / 305.0, L768 3323.5 / 3327.8 / 3323.8 / 3327.7 / 3336.3.
At L128 head_bwd holds 128 of the 148 SMs and its shared memory (229 KiB) leaves no room for a wgrad CTA beside it, so the overlap is
the 20 free SMs; at L >= 256 the concurrent wgrad slows head_bwd. Under one timer step at best; not kept
(`scratchpad/b2/r3tt_pdl_bwd_rejected.diff`).

**Not reached in this part** (task items 2-5): M = 64 / channel-split tiles and fused gate_bwd + head_bwd / wgrad-in-head_bwd for
training L128 (head_bwd's resident weights take 128 KiB of shared memory and wgrad all 512 TMEM columns: a fused kernel needs the
four M_t accumulators split across a CTA pair or two passes); the bias-only / Bidirectional per-head dP kernel; the gate-from-x tail
at d_pair 256 / 384; inference L128 per-tile pipeline. After this part the training rows over 0.667 are the fused d_pair 128 L128 cases
(r243 / r247: 3, 0.788-0.795) and d_pair 64 L128 (5, 0.668-0.725; was 0.684-0.747 at round start).

### Part 3 (2026-11-17 evening): L128, L256 row split, live key tiles (cuda:b200_sm100 9)

The session that started part 3 was stopped mid-work (B200 access rules); its experiments were not committed. What it had measured
(`_r/out3/`, L768 x 4 heads core, 583-586 us as it is): the bias on the tensor core (`TA_BMMA`: eight M128 N16 K16 MMAs a sub-step,
A = the staged bias tile, B = c I16) 617 us at L768 (99.8 at L384, 10466 at L2048 vs 10603): the softmax groups' work fell from
2733 to 1936 clk a sub-step (clock64 events, `tl2b.log`), but the QK^T issue warp now issues ten MMAs a sub-step and a tcgen05 MMA
costs ~65-73 clk whatever its N (`mmaiss.log`: M128 N16 / N32 / N128 K16 73 / 68 / 65 clk an instruction), so the tensor pipe
(18 MMAs a sub-step) became the bound; one issue warp a softmax group (`TA_SELF`) 651 us; P released to the PV warp per 32-key chunk
(`TA_PCH`) 591 us; four groups on three S / P buffers (`TA_G4`) faulted. Patches in the session scratchpad (`git stash` on this
branch: "part 3 rejected core experiments").

This part (core forward, graph, us; records `_r/out4/`, the core bench `_r/cb4.py`, small L in their own runs):

- **What the L128 core is made of** (4 heads, 10.4 us): no logits / exponentials 8.3, no softmax arithmetic 8.3, also no MMAs 6.3,
  also no K / V / bias loads 6.3. Two thirds of the kernel are its skeleton (launch, set-up, Q load, barrier chain, drain), not
  the MUFU (the exponentials of L128 x 4 heads are ~1.9 us of MUFU time on 148 SMs). At L256: 32.9 -> 22.7 / 19.0 / 16.5 / 14.5.
- **One pass of four rows at L128 (`TA_L1`), rejected.** R = 4 with a softmax group and an S / P buffer a row, the row's O in its
  buffer's columns 64-95 (L128 has one key tile, so S 64-127 is read before P is handed over), so a CTA does its four rows at once
  instead of two tasks of two rows; the same bits. 2 / 4 / 8 / 12 heads: 10.37 / 10.36 / 16.49 / 23.17 us against 8.32 / 10.35 /
  14.48 / 20.59 (one issue warp instead of two: 22.13 at 12 heads). Four softmax warps a sub-partition share its MUFU and the
  latency of the single pass is not shorter than two overlapped tasks.
- **L256 with three rows at 12+ heads, taken.** The three-group build (part 2) at L256: 2 / 4 / 8 / 12 heads 22.65 / 34.94 / 65.76 /
  91.36 against 20.60 / 32.90 / 64.99 / 96.03 with two rows: `_rows_per_task(L, samples x heads)` takes three rows at L256 from
  12 on. d_pair 384 x 12 L256 eval (module, measurer): 0.2559 -> 0.2476 ms ending, 0.2427 -> 0.2354 starting (ratio vs H100 0.616 ->
  0.596, 0.597 -> 0.579); accuracy GREEN (2 / 2), the same bits as before (the three-row build writes the two-row bits).
- **Live key tiles (the H100 core's mask-counted key tiles), rejected.** The forward took the key mask, one warp a sample found the
  last live key, and the TMA / issue / softmax warps walked only the live tiles (per-task sub-step counts in the issue streams).
  Module outputs bitwise equal with and without (11 classes x both nodes, the measurer's 1/8-padded mask at L1024 and two
  different tail masks at L1000 B2; training: the same forward, the gradients as nondeterministic as without (fused d_pair 128
  fp32-atomic weight gradients)). Core, 4 heads: with the 1/8-padded mask L1024 0.884, L2048 0.913, 12 heads L1024 0.889 of the
  unmasked time; but against the kernel before the change (interleaved A/B) the unpadded case lost 3.7 % at L512, ~2 % at L1024,
  ~1 % at L2048 (the per-task sub-step counts in the issue warps), and the speed measurer times unpadded masks (`unpadded_masks`), so
  every requirement case would get slower. It would pay only as a separate build selected when a sample has a dead tail, which
  needs the mask's live length on the host (a sync) or from the front kernel (the fronts' owner).

State after part 3: the changed cases are the d_pair 384 L256 pair above (both were already under 0.667). The L128 cases (23
inference, 10 training) and the softmax-bound L >= 256 inference cases are as after the fronts / tails round: see Gaps.

### Part 4 (2026-11-17 night): the PV MMA count, 2-CTA, bias in the accumulator -- measured ceilings, nothing kept

Task: cut the head-dim-32 core's PV MMA count (O^T form or a `cta_group::2` M = 256 core), then use the freed tensor-pipe time for
the bias-into-accumulator form and the softmax groups. Records `$W/b2511/agents/r3-triattn/_r/out5/` (`j1*.log`, `mmab*.log`,
`j3_build.log`, `j6_build.log`) and the job logs `$W/b2511/jobs/r3ta_j{1..7}.log`, `r3ta_mmab.log`; the tools `_r/mmab.cu` / `mmab.py`
(MMA rate microbenchmark), `_r/cb5.py` (core variants, interleaved rounds against the committed source), `_r/ftl.py` (clock64
buckets); the rejected variants as diffs in `_r/out5/p4_*.diff` (bmma_proxy, defer, single_issuer, hiraw). Core forward, graph, us; L768 x 4 heads unless said.

**1. What one tcgen05 MMA costs** (one issuing thread, 240 back-to-back instructions, clk an instruction, issue = completion):

| form | clk | | form | clk |
|---|---|---|---|---|
| ss M128 N128 K16 (QK^T) | 64.5 | | 2-CTA ss M256 N128 | 64.9 |
| ss M128 N64 | 48.5 | | 2-CTA ts M256 N16 / N32 / N64 | 46.7 |
| ss / ts M128 N16 / N32 / N64 (PV: ts N32) | 46.7-46.9 | | 2-CTA ss M256 N256 | 129.0 |
| ts M128 N128 / N256 | 64.9 / 128.9 | | sub-step mix 2 ss N128 + 8 ts N32 | 46.7 avg |
| O^T: ss M64 N128 / N256 (A = V^T MN-major) | 64.5 / 128.7 | | the same mix, 2-CTA | 46.7 avg |

An MMA has a floor of ~46.7 clk whatever N up to 64 (N128 = 64.5 clk is the bf16 peak, 4096 MAC / clk); the 2-CTA instruction costs
the same and covers both SMs. So the O^T form does not cut the count (K = the 128 keys still takes eight K16 steps; M = 64 for the 32
channels wastes half the tile), while the 2-CTA form halves the PV instructions an SM (8 x 46.7 = 374 -> 187 clk a sub-step).
(Part 3 had M128 N16 / N32 at 73 / 68 clk; this benchmark issues from an unrolled loop with the form a template parameter -- its
first version, with a runtime switch in the loop, read 165-235 clk for every form, so issue-side overhead easily inflates these.)

**2. The tensor pipe is not the bound.** Per CTA sub-step (clock64 buckets, 747 sub-steps a CTA): 1500 clk; the tensor work is
2 x 64.5 + 8 x 46.7 = 503 clk (34 % busy). The softmax groups: work 916-931, waiting for S 272-292, bias 80, drain / bookkeeping
186-220. Upper bounds by issuing fewer MMAs (wrong output, timing only): PV 4 of 8 580.6 us, no PV at all 573.4, QK^T 1 of 2 578.8,
against 587.2 -- removing the whole PV stream buys 2.4 %, and the softmax groups' wait for S only falls 292 -> 249 clk (it is the
barrier hand-offs, not tensor time). A 2-CTA core (halving PV a SM) is bounded by the PV-4 number (-1.1 %) before the cross-CTA
barrier coupling it adds; **not built** on this measurement.

**3. Bias into the accumulator, revisited (rejected).** The predecessor's `TA_BMMA` (eight ss M128 N16 K16 a sub-step, A = the staged
bias tile, B = c I16 with c = bf16(1 / scale)): 621.2 us at L768 (587.2), 100.2 at L384 (90.6); accuracy as before (vs the fp32
reference 2.7e-3 like the base; c = bf16(sqrt 32) is 1.1e-4 off). With the overflow guard compiled out for timing (variants with
garbage S): no guard 605.0, BMMA 615.7, BMMA with half the bias and half the PV MMAs (the per-SM tensor load of a 2-CTA BMMA core)
597.0, **no bias anywhere** (the softmax's bias unpack + FFMA2 gone and no bias MMAs: the ceiling of the form at zero tensor cost)
577.0, the same with half the PV 580.4; at L384 95.1 against 93.7. So even free, the bias-in-accumulator form is worth <= 4.6 % at
L768 and nothing at L384, and its 2-CTA form ~1.3 %. It also needs a finite mask value (0 x -inf = NaN in the identity MMA: the fronts
write -inf for masked keys). The target (~520 us, -11 %) is not reachable through the tensor core.

**4. Softmax-group schedule variants (rejected).** Interleaved A/B against the committed source, three / two rounds:
- the deferred drain (`TA_DEFER`: a task's O_r / l_r drain in the next task's first sub-step, after its exponentials and before its P
  hand-over, so the wait for the row's last PV overlaps the softmax): L384 91.1 vs 90.5, L512 193.0 vs 193.3, L768 4 heads within the
  power-cap drift (583-613 both), 12 heads L384 +0.8 %, L768 1886 vs 1894 (-0.5 %); the same bits. Within a timer step / noise.
- one issue warp for both streams in the three-group build (`TA_SPLIT=0`, R = 3; built but never timed by the predecessor): L384
  91.9 vs 90.3, L512 197.2 vs 193.4, L768 618 vs 585 (+5.6 %), 12 heads L768 1926 vs 1893.
- the bias pair's high element read as the whole 32-bit word (one instruction a pair less): not timed -- a masked pair
  (0xff80ff80) reads as NaN and every masked tile would take the recompute branch.

**5. Head-dim-64 / 128 cores.** Nothing above won, so nothing to port. The three-group form (part 2's win) does not fit the
head-dim-64 core's TMEM: three S buffers + three 64-column O = 576 > 512 columns (a 96-key tile would fit, 288 + 192, but serves only L
multiple of 384).

**What bounds the core now, and what is left.** Per sub-step the SM spends ~1024 clk of MUFU (128 x 128 exponentials at 16 / clk)
of 1500; the tensor pipe 503; the three softmax warps a sub-partition issue roughly 530 instructions a sub-step each (estimate) and are latency bound
(the "no exponential" ablation, 494 us, still takes ~1255 clk a sub-step). The remaining levers are all blocked by the 512 TMEM columns (480
used: three 128-column S / P buffers + three 32-column O): a fourth softmax group, a separate P buffer (to start the next QK^T before
PV retires) or a ones column on V (row sums on the tensor core, N = 48: +48 columns) each need 32-192 more columns.

**L128: the interface a fused front + core + tail kernel needs** (to be assigned; not built here: the front / tail sources belong to
`b200r3-triattn-tail`). Work unit: one CTA a (sample, pair row n) of 128 tokens, all heads; the bias needs every row of the pair, so
it stays a separate first launch (or a grid barrier).
1. *Bias launch* (front owner): `tri_front_sm100<CX, HD, NOUT = 0>` -- LayerNorm + the bias projection only, writing
   `bias [B][H][128][128]` bf16 (masked keys -inf, `BT` geometry for AF3's ending node) and, in training, the LN statistics
   `(mean, rstd) [B*L*L]`. ~4 MiB at d_pair 128 x 4 heads: stays in L2 for the second launch.
2. *Front phase as device code* (front owner): `fr::ln_tile` (x tile [CX/64][128][64] 128B-swizzled in shared memory -> y in place,
   given LN warps) and `fr::proj` (y . W^T with W streamed from global / L2 in [64-K][HD] chunks through a caller-given ring, since the
   resident q|k|v|g weights (128 KiB at d128) do not fit beside the core's buffers), accumulating into caller-given TMEM columns
   (4 x HD fp32 = all 512 at d128: q | k | v first (384), g later or projected in the tail as `tri_tailg` does), and a drain that
   writes bf16 tiles in the *core's operand layouts* into caller-given shared memory: Q and K per head K-major [128][32] 64B-swizzled
   (the `desc_k64` layout), V per head [128 keys][32] 64B-swizzled read MN-major (`sdesc(.., 512, 4)`), q pre-multiplication none
   (the core applies scale x log2e in fp32). In training it must also store q / k / v / g rows to global (the backward reads them).
3. *Core phase* (core owner, this file): a device function `core_rowtile(sQ, sK, sV, bias rows from global, H heads, out: O_h in
   TMEM [128][32] fp32 + l, m)` running the four heads on the two S buffers left after the front's TMEM is drained; writes LSE in
   training.
4. *Tail phase as device code* (tail owner): `tailg` taking o from TMEM (fp32, [128][H x 32]) or from shared bf16 [128][HD]
   128B-swizzled, the residual x from the tile already resident (the front phase's x slot; LN recomputed for the gate), Wg / Wo
   streamed (not resident), dropout scale rows, and the `TRANS` geometry for the store.
5. *Budgets to agree on*: shared memory (x 32 KiB + Q / K / V 3 x 32 KiB + a weight ring 2 x 16 KiB + barriers: ~165 KiB at d128,
   ~200 KiB at d_pair 256), TMEM by phase (projection 384-512 -> core 2 x 128 S + 4 x 32 O -> tail gate / out accumulators), the
   warp roles (the front's 320 + drain warps vs the core's 480 threads: one CTA shape for all phases), and the barrier / named
   barrier ids. Gain estimate at L128 d_pair 128 inference (module ~35 us, front 10 + core 10.4 + tail ~10): the two inter-kernel
   start-up / drain latencies (~6-8 us) and the q / k / v / o round trips (~20 MiB); the core's own skeleton (6.3 of 10.4 us) shrinks
   only by its launch / set-up share.

State after part 4: no source change (cuda:b200_sm100 stays 10, taken by the fronts / tails part 2); the counts over 0.667 are those
after part 3 and the fronts / tails rounds.

### Part 5 (2026-11-17 night): more TMEM for the softmax (CTA pair) -- measured ceilings, nothing kept

Task: use a `cta_group::2` CTA pair for twice the TMEM per row tile, then use the freed columns for a fourth softmax
group, separate P buffers or row sums as a ones column of V. Records `$W/b2511/agents/r3-triattn/_r/out6/` and the job logs
`$W/b2511/jobs/r3ta_p5j1.log` / `r3ta_p5j2.log`; tools `_r/cb6.py` (core variants, interleaved rounds, resource usage) and `_r/ftl6.py`
(clock64 buckets of CTA 1, rows per variant); the variants are compile-time switches in `docs/b200/rejected/triattn_part5_tmem_ceilings.diff`
(`TA_G4X`, `TA_SEPPX`, `TA_NOGUARD`, `TA_PK`). Core forward, head dim 32, 4 heads, graph, us.

**1. A CTA pair does not add TMEM per SM.** Each SM still has 128 lanes x 512 columns, and the softmax warps of an SM read only
their own SM's TMEM. With `cta_group::2` M = 256 each CTA holds 128 rows x N columns, the same as now. With M = 128 (64 rows a CTA, CUTLASS
`tmem_frg_2sm` "2x2" layout) a CTA's S is 128 lanes x 64 columns, keys 0-63 in lanes 0-63 and keys 64-127 in lanes 64-127. So the columns per
S buffer halve, but so do the elements. The fp32 S still costs one column a key a lane. O gets more expensive: a row's O sits in two lane
halves, and the A = P operand of a CTA-pair PV must be duplicated in both lane halves (`TmemAllocMode::Duplicated`). That leaves one-CTA
PV MMAs with B = [V keys 0-63 | V keys 64-127] at N = 64, half of D unused: 64 columns a row slot instead of 32. A softmax warp would also do
32 rows x 64 keys a sub-step, the granularity of the rejected 64-key halves (part 2). So the pair gives no room for anything. Before
building it, the two levers it was meant to enable were measured with all the TMEM they need (the output is wrong; timing only):

**2. Ceilings with free TMEM.** Interleaved A/B, two rounds; `ng` = the base without the overflow guard and FLAGS atomics (the
variants have no guard), since the guard-free build schedules differently:

| variant | L384 | L768 (two rounds) | clk a sub-step (L768) | softmax wait for S / work |
|---|---|---|---|---|
| base (G3) | 90.3 / 90.6 | 586.3 / 619.6 | 1496 | 265 / 918 |
| ng | 94.8 / 95.1 | 608.7 / 626.6 | 1476 | 266 / 902 |
| G4X: four one-row groups, four S buffers (all 512 columns, O aliased), 608 threads | 99.3 / 99.3 | 641.1 / 664.9 | 1560 | 174 / 951 |
| SEPPX: G3, P in its own columns, QK^T(y + 3) released once the group has read S(y) | 94.9 / 95.0 | 614.8 / 612.3 | 1453 | 57 / 1119 |

(L768 drifts with the power cap between rounds; compare within a round.) A fourth softmax group is 5-10 % slower. Separate P buffers
remove the wait for S (266 -> 57 clk) but the softmax work grows by the same amount (902 -> 1119): -1.5 % clk, level in time.

**3. What the sub-partition is bound by: issue slots.** P's bf16 pack in three forms (L768 clk a sub-step; L384 us): `cvt` (F2FP, as it
is) 1495, 90.2-90.7; one byte permute (truncation) 1503, 92.2 (error vs fp32 3.6e-3 -> 7.8e-3); round-to-nearest in integer ops
(about 5 more ALU instructions a pair, 64 pairs a warp and sub-step) **1815**, 111.2. So the pack shares no pipe with the MUFU, and
about 320 extra instructions a warp-sub-step cost about 320 clk a sub-step. Each sub-partition issues about one instruction a clock for its
three softmax warps, so the forward core is bound by the instruction count of the softmax loop, not by latency, TMEM capacity
or the MUFU alone. This replaces part 4's "latency bound" reading. Part 4's "no exponential" ablation (an FFMA2 per pair instead of
two MUFU ops) took fewer issue slots too. More warps or buffers cannot help when the issue port is full. The lever left is fewer
instructions a key in the softmax loop. A SASS count of the loop (bias unpack + two FFMA2, two MUFU, the add, the pack a pair,
plus the TMEM / barrier overhead a chunk) is the next step. Moving the bias or the row sum to the tensor core removes instructions
but was measured slower or capped (part 4: <= 4.6 %).

**4. Head-dim-64 / 128 cores.** Nothing won, so nothing to port.

State after part 5: no source change (cuda:b200_sm100 stays 10; 12 was reserved for this part and is not used). The counts over 0.667
are those after part 3 and the fronts / tails rounds.

### Round 3, fronts and tails part 3 (2026-11-17 night, cuda:b200_sm100 11 / Bidirectional 4)

Records `$W/b2511/agents/r3-triattn-tail/_t/out/` (`spd_bo128_v11.jsonl` / `spd_bobig_v12.jsonl` / `acc_bo_v11.jsonl`: the bias-only
rows r142 / r188, native on; `spd_bo*_off.jsonl`: the same with `MINIWORLD_BIDIR_B200_NATIVE=0`; `spd_wtr{s,b}_tk{1,0}.jsonl`,
`acc_wtr_tk1.jsonl`: the wide training rows with the tall-K kernel on / off). Kernel profiles: `_t/var.py` (graph replay +
torch.profiler), scripts `_t/p2.sh` .. `p6.sh`; comparisons `_t/cmpbo.py`, `_t/cmpw.py`.

**1. Bias-only forms: the ending node's per-head contractions on tcgen05 (taken).** `BidirectionalTriangleAttention` and
`TriangleAttention(use_self_attention=False)` ran the ending node as one cuBLAS batched GEMM per head (P_e[h] shared by the head's
channels: `bmm(P.expand(dh, L, L), V)`), and in the backward two more per head plus a torch sum over the head's channels for dP
(`sum_d dO_d V_d^T`: a [dh, L, L] fp32 intermediate). At L128 x 4 heads that is 12 GEMM launches + 4 reduces + the softmax backward's
multiply / reduce, each a few us. New sm_100a kernels (`b200_sources/bidir_sm100.cu`, extension `b200_bidir.bo()`):
- `bo_pv`: out[c] = P[h] V[c] (forward, A K-major) or P[h]^T dO[c] (backward dV, A MN-major), B = the channel plane MN-major, every
  channel of every head in one launch (a CTA = one channel x one 128 x 128 tile; fp32 in TMEM, one bf16 rounding). The same bits as
  the batched GEMM (`_t/bo1.py`, `tests/gpu/test_triattn_b200.py::test_bias_only_native_ending_kernels`).
- `bo_dp`: dP[h] = sum over the head's channels and columns of dO[c] V[c]^T, K = dh x L in one accumulator; a cluster of up to 8
  CTAs splits the channels and adds the partial tiles through distributed shared memory in rank order (deterministic, no fp32
  intermediate). fp64 check 3e-5 relative at L768.
- `bo_rowdot`: the softmax backward's (p dp).sum(-1) in one launch (torch: a multiply and a 8-9 us reduce); `bo_colsum`
  (`sm100::colsum`): the LayerNorm backward's d gamma / d beta partials (torch's tall reduce: 8.5 us at L128); the first sample's
  weight-gradient products start the sums (no zero fill + add).
The native GEMMs run at L <= 256 (`BO_GEMM_MAX_L`): from L384 on cuBLAS's per-head GEMMs are faster (ending eval L768 0.422 vs 0.559
ms with the native one-tile-a-CTA kernel); the row dot / column sum / accumulation changes run at every L.

| measurer, cuda, graph, ms (ratio vs H100) | before | after |
|---|---|---|
| Bidirectional L128 train | 0.1964 (0.773) | 0.1392 (0.548) |
| Bidirectional L128 eval | 0.0551 (0.684) | 0.0490 (0.608) |
| Bidirectional L256 train | 0.3520 (0.587) | 0.3110 (0.518) |
| bias-only ending L128 train | 0.1780 (0.757) | 0.1224 (0.521) |
| bias-only starting L128 train | 0.1369 (0.725) | 0.1203 (0.637) |
| bias-only ending L128 eval | 0.0430 (0.659) | 0.0387 (0.594) |
| L >= 384 (both classes, train) | | 0.97-0.99 of before; eval within +-2 % (unchanged path) |

Over 0.667 on these rows: 4 -> 0 (Bidirectional train / eval L128 at 0.773 / 0.684 included). Accuracy (r142 / r188, every L, eval +
train): 30 GREEN, 6 YELLOW -- the same six cases at the same ratios as the tokens (1.02-1.19x tol). Training bitwise over three runs
(`_t/bo1.py`: L128, L256 B2, L200 ragged, both classes and nodes).

**2. Wide training at L128: the weight-gradient GEMMs in one tcgen05 launch, `tri_db_rows` widened (taken).** The wide backward
ran G = Dall^T [xhat | 1] and dWo = dy_s^T u as two cuBLAS split-K GEMMs plus two split-K reduce kernels (L128 d_pair 64 x 4 x 32:
7.7 + 5.6 + 2 x 2.3 us). `tallk_sm100.cu` runs both products in one launch: a cluster of up to 16 CTAs (non-portable cluster size)
shares each 128 x 128 output tile, each CTA a contiguous K range with both operands MN-major straight from the row-major tiles,
the partial tiles added through distributed shared memory in rank order (deterministic, no partial buffers). Microbenchmark (K =
16384, M 576 x N 128 + 64 x 128): split 2 / 4 / 8 / 16 37.4 / 21.4 / 13.9 / 11.4 us against cuBLAS 16.8; a K = 64 call still takes
~6-7 us (set-up, the cluster's drain / exchange), so it loses at long K: measurer, training, tallk on / off geomean L128 0.974, L256
1.019, L384-768 1.025-1.030, d_pair 384 / 512 at L128 +2 / +8 %. Taken where it wins: B Lp^2 <= 128^2 rows and d_pair <= 256
(`TALLK_MAX_ROWS`). `tri_db_rows` (dbias fp32 -> the head backward's [R][64] bf16 block) ran a thread a row (64 blocks at L128:
6.4 us); a thread per row x 16-byte piece, the piece slowest (coalesced head-plane loads): 3.0 us, the same bits.

| measurer, train L128 (ratio vs H100) | before | after |
|---|---|---|
| d_pair 64 x 4 x 32 starting / ending | 0.1226 / 0.1227 (0.688 / 0.681) | 0.1104 / 0.1130 (0.619 / 0.628) |
| Chai-1 d_pair 64 x 4 x 32 starting | 0.1211 (0.686) | 0.1104 (0.626) |
| d_pair 64 x 2 x 32 starting / ending | 0.1060 / 0.1064 (0.731 / 0.725) | 0.0984 / 0.0997 (0.679 / 0.680) |
| d_pair 256 x 8 starting | 0.2107 (0.646) | 0.2025 (0.621) |
| AF3 d_pair 512, d_pair 384 (cuBLAS kept) | 0.399-0.411, 0.297-0.301 | unchanged (+-1 %) |

Wide training rows over 0.667: 5 -> 2 (the d_pair 64 x 2 x 32 pair, 0.679 / 0.680). Accuracy (13 wide training rows, L128 / 384 /
768): 59 GREEN, 17 YELLOW, every YELLOW at the tokens' ratio or lower (two token YELLOWs now GREEN); three runs bitwise (`_t/tk1.py`:
d_pair 64 x 4 x 32 / 2 x 32 ragged L200 B2 / Chai-1 / AF3 L256 / d_pair 256 / RF3); `tests/gpu/test_triattn_b200.py` 87 passed.
Tried on top and rejected (d_pair 64 x 2 x 32 L128 training, graph): `tallk` as a programmatic dependent of `tri_whbwd`
(set-up before `griddepcontrol.wait`) 86.4 / 86.5 vs 86.9 / 86.9 us (`scratchpad/b2/tt_tallk_pdl_rejected.diff`); `tri_wfinish_sum`
folded into `tri_wfinish` as a last-block sum (one launch less) 87.3 us, the last block's serial sum costing what the launch saved
(`tt_wfinish_lastblock_rejected.diff`).

**3. Wide front at d_pair 256 / 384 (measured, nothing kept).** At L1536 eval the front is 2.56 ms of 17.7 at d_pair 384 (1.20 of
10.9 at d_pair 256): ~2.8 TFLOP of projections at 55 % of the tensor peak, 3.5 TB/s. Its shared memory is the W ring (the pair
streams 1.2 MB of W per 256-row pair tile at d_pair 384): a deeper drain ring (4 buffers a warp) costs W stages and is slower
(d_pair 384 2609 -> 2822 us, d_pair 256 1220 -> 1556), one x buffer at d_pair 256 for 6-8 W stages slower (1196 -> 1301-1306), the
stage cap 6 -> 8 changes nothing at d_pair 384 (the ring already fills the 227 KiB). Two row tiles a CTA sharing each W chunk would
halve the W inflow but need two x tiles (2 x 96 KiB at d_pair 384) or y in TMEM (384 of the 512 columns): not built.
(`scratchpad/b2/tt_wfront_knobs_rejected.diff`.)

**4. Gate-from-x tail at d_pair 256 / 384 (not built, measured reason).** The g stream it removes is 1.8 GB written + read at d_pair
384 L1536 (~0.5 ms of the front's and tail's 4.0 ms), but the gate GEMM (0.7 TFLOP) moves into the tail, and Wg + Wo (576 KiB at
d_pair 384) cannot stay resident: streamed per 128-row tile they are 6x the tile's own traffic through L2, split over a 4-CTA
cluster they need the u slices exchanged through distributed shared memory every tile. Upper bound ~3 % of the module; with the
front at its W-inflow limit (3) the d_pair 256 / 384 L >= 1536 cases at 0.671-0.694 stay core-bound.

**5. Fused L128 front + core + tail (the core owner's interface, part 4): not built.** Ceiling against the requirement: d_pair 128
L128 eval is 34.6 us (H100 36.6, target 24.4); the interface's estimate of the saving is 6-8 us (two kernels' start-up / drain and
the q / k / v / o round trips) -> 27-29 us (0.74-0.79), so no case crosses 0.667 even at the full estimate; d_pair 64 x 2 x 32 (26.5
vs 27.9, target 18.6) neither. The kernel (one CTA shape for the front's 320 + drain warps and the core's 480 threads, TMEM phases
512 -> 2 x 128 S + 4 x 32 O -> tail accumulators, weights streamed) is a multi-day build; the L128 inference rows stay at 0.82-0.99.

**6. Fused d_pair 128 training L128 (item 2 of the task): not built.** 124.6 us (H100 157.1; target 104.8): -20 us needed. The
non-core kernels are front 10.8, tail 8.1, gate_bwd 11.8, head_bwd 11.1, wgrad 15.6, slot sum 7.0, finish 3.5, pack 3.5 (graph
profile); the cluster split-K reduction of (2) applied to gate_bwd's dWo and wgrad's M_t partials would replace the slot sum (~7 us)
by in-kernel exchanges (~1-2 us each, measured on tallk's drain / exchange), i.e. -3 to -5 us; fusing gate_bwd + head_bwd + wgrad
needs the four [128][128] M_t accumulators (all 512 TMEM columns) beside head_bwd's two gy accumulators: a CTA pair (cta_group::2)
or two passes. Neither reaches -20 us; the three cases stay at 0.79.

### Training tails round 6 (2026-11-17 night, cuda:b200_sm100 30)

Scope: the training backward (my classes, training cases only). Records `$W/b2511/agents/r3-triattn-tail/_t/out6/`
(`spd_before_L*.jsonl` / `spd_after_L*.jsonl`: every training requirement case of the six classes, cuda / triton / pytorch, one measurer
run per length, exclusive; `acc_after.jsonl`); tables `_t/t6.py`, accuracy `_t/acc6.py`, checks `_t/pre1.py`, `_t/clu1.py`.

True state at the start (the four commits of fronts / tails part 3 on b200-2.5.x 83e4c1ddf; 198 cases, H100 median >= 20 us for all):
over 0.667: 5 (d_pair 128 fused L128 starting / ending / AF3 starting 0.794 / 0.798 / 0.792; d_pair 64 x 2 x 32 L128 0.677 / 0.679);
none slower than the H100; ladder violations 0. The part 3 commits were kept (their gains are in the before numbers).

**1. The backward's set-up passes in one launch (taken).** `tri_bwd_pre` (`triattn_wide_sm100.cu`): dy_s = dy o drop (tri_scale_rows'
body), the head-dim-32 core's bias^T x log2e (triattn_biasT's tiles, the same bits) and its zeroed dbias accumulator, block ranges of
one launch; `triattn_bwd` takes the bias^T as an optional argument (`bias_t`). Wide training (head dim 32) and the fused d_pair 128
path. L128 d_pair 64 x 2 x 32: scale 2.4 + biasT 1.5 + fill 1.2 us -> 2.5 us. Every output / gradient bitwise equal to the separate
launches, three runs bitwise (`pre1.py`: 13 configurations incl. ragged L200 B2, RF3, Chai-1 / Confidence, d_pair 384).

**2. Slot sum with more loads in flight (taken).** `tri_slot_sum` 8 lanes x 4 loads -> 16 lanes x 8 loads a column (fixed order):
d_pair 128 training L128 8.1 -> 6.8 us, L256 9.8 -> 8.3 (graph step -1.0 / -1.7 us). Other shapes measured: 16 x 4 8.9, 16 cols x 16
lanes 7.3, 16 x 32 10.8, 8 x 32 x 8 7.8 us (L128).

**3. Rejected (measured):**
- tall-K GEMM in K parts (`tallk` grid z = K part, planes added in order by `tri_wfinish`): d_pair 64 x 2 x 32 L128 tallk 11.4 us ->
  split 16 x 2 parts 17.4 (more than ~4 clusters of 16 do not run at once), 8 x 2 10.1 but tri_wfinish +1.2 (graph 87.2 -> 87.3),
  8 x 4 15.6, 4 x 4 10.7: the kernel is not load-bound (~10 us fixed). `scratchpad/b2/tt6_tallk_kparts_rejected.diff`.
- dgamma / dbeta of `tri_wfinish` in dedicated blocks over all rows (no `tri_wfinish_sum` launch): the serial row loop made wfinish
  3.5 -> 6.5 us (d_pair 64 x 2 x 32), 11.3 (x 4 x 32), 15.6 (RF3 d_pair 128): slower than the saved 2.2 us launch.
- cluster reductions of the fused d_pair 128 path's deterministic partials (gate_bwd dWo and wgrad M_t summed through distributed
  shared memory in rank order inside clusters of 8, one slot a cluster): slot sum 7.8 -> 4.8 us, but wgrad 16.4 -> 20.9 and gate_bwd
  11.8 -> 13.7 (the exchange runs after every role is done); graph L128 115.9 -> 119.0, L256 298.6 -> 301.5. Same gradients to
  2e-7 relative, deterministic. `scratchpad/b2/tt6_cluster_wgrad_gatebwd_rejected.diff`.

Result (measurer, cuda, graph, L2 flushed; ratio = B200 / H100):

| | before | after |
|---|---|---|
| over 0.667 | 5 | 3 |
| d_pair 64 x 2 x 32 L128 starting / ending | 0.0982 / 0.0996 ms (0.677 / 0.679) | 0.0947 / 0.0957 (0.653 / 0.653) |
| d_pair 128 fused L128 starting / ending / AF3 starting | 0.1248 / 0.1264 / 0.1247 (0.794 / 0.798 / 0.792) | 0.1220 / 0.1237 / 0.1236 (0.776 / 0.781 / 0.785) |
| ladder violations (cuda < triton < pytorch) | 0 | 0 |
| cuda after / before, 198 cases | | geomean 0.996 (0.959-1.019) |

Accuracy (`acc_after.jsonl`, the 15 training rows of the five TriangleAttention classes at L128 / 384): 46 GREEN, 14 YELLOW, every
YELLOW at the token's ratio (1.01-1.32x tol), one token YELLOW now GREEN; no RED. `tests/gpu/test_triattn_b200.py` 87 passed;
training bitwise over three runs (`pre1.py`, `det3.py`, `bo1.py`).

Left: the three fused d_pair 128 L128 training cases (0.78; -19 us needed of ~122). Kernel times (graph, L128): wgrad 15.7, gate_bwd
11.8, head_bwd 11.1, front 10.8, bwd_kv 10.7, bwd_q 10.4, core fwd 9.5, tail 8.1, slot sum 6.8, finish 3.6, pack 3.3: each kernel is
one tile a CTA on 128 of the 148 SMs; the measured levers here were 1-3 us each. The remaining routes (one kernel for gate_bwd +
head_bwd + wgrad needing the four M_t accumulators beside head_bwd's gy: a CTA pair; a fused front + core + tail) are the multi-day
builds of part 3 (5 / 6).

### Part 6 (2026-11-17 evening): inference -- true state, PDL chain front -> core -> tail (cuda:b200_sm100 20)

Records `$W/b2511/agents/r3-triattn/_r/t/` (`v0/e_<L>.jsonl`: every eval requirement case of the six classes, pytorch / cuda / triton,
one measurer run per length group in an exclusive slot; `v1/`: cuda after the change, same runs; `vt.py` / `vt*.sh`: module A/B in
separate processes, L2 flushed like the measurer; `chk.py` / `cmp.py`: bitwise PDL on / off). Timer resolution ~2.05 us.

**True state before (v0), eval, 274 cases with an H100 token**: 109 over 0.667 (none slower than the H100: worst 0.971, RF3 d_pair 128
L128 starting), ladder violations 0 (cuda < triton < pytorch on every case). By class: TriangleAttention 49 / 118 (worst 0.948),
AF3 38 / 60 (0.946), RF3 17 / 36 (0.971), Chai-1 4 / 36 (0.822), Chai-1 confidence 0 / 18, Bidirectional 1 / 6 (0.684). By length:
L128 23 / 33 (0.69-0.97: ~28 % faster needed), L256 13, L384-L768 10 each, L1024 7, L1536 10, L2048 15, L3968 1. Almost all misses are
on the head-dim-32 core (d_pair 128 x 4 x 32, d_pair 64 x 4 x 16 / 4 x 32); at L384-L1024 most need 1-5 %, at L2048 8-14 % (the core
is ~86 % of the module there; the H100 module scales sub-cubically from L768 to L2048, ours does not).

**1. Programmatic dependent launches, taken.** `sm100.cuh` `launch_pdl` / `pdl_wait` / `pdl_trigger`: the fronts (`tri_front`,
`tri_front64`, `tri_wfront`) trigger at entry; the cores (head dim 32 / 64 / 128) are PDL launches that wait after their set-up
(barriers, TMEM); the tails are PDL launches whose x and weight loads (and in `tri_tailg` the LayerNorm and the gate GEMM, which do not
depend on o) run before `griddepcontrol.wait` -- only the o loads and the stores wait (`tri_tailg`, `tri_tail`, `tri_wtail`). Every
dependent can only take an SM its primary has left (shared memory), so it overlaps the primary's ramp-down. Outputs bitwise equal with
and without (`chk.py`: 12 configurations x both nodes x L128 / L200 B2 / L384, key masks with a dead tail, eval outputs and training
forward outputs; `MINIWORLD_TRIATTN_B200_PDL=0` launches ordinarily). Module A/B (us): d_pair 64 x 2 x 32 L128 24.6-26.6 -> 22.5, L256
51.2 -> 47.2-48.3; d_pair 128 ending / AF3 / RF3 L128 34.8 -> 32.8; d_pair 64 hidden 128 L128 30.7 -> 28.7; d_pair 256 L128 53.2 ->
51.2; RF3 d_pair 128 L256 77.8-79.6 -> 75.8; AF3 d_pair 64 L384 153.4 -> 149.5; nothing at L >= 512. Measurer (v1 / v0, geomean):
L128 0.960, L256 0.985, L384 0.998, L512+ 1.000 +- 0.5 % (drift).

**2. Rejected (measured):**
- the bias pair's high key read as the whole 32-bit word (one LOP3 a pair less in the softmax loop; `rejected/triattn_part6_bias_hi_word.diff`):
  core L384 / L768 / L2048 90.2 / 601 / 10616 us against 90.3 / 591-607 / 10665-10763 -- level -- and the error against fp32 2-3x
  (3.6e-3 -> 6.2e-3 at L384). With part 5's pack variants this says the loop is not bound by the issue of its ALU instructions either;
  the predecessor's fp16 pre-scaled bias (`TA_HB`, FHADD mixed adds, its stash) was 4-12 % slower (spills at the 128-register cap).
- `ex2.approx.f16x2` / `.bf16x2` / `tanh.f16x2` on sm_100a: 8 instructions / clk / SM = 16 elements, the same rate as `ex2.f32`
  (microbenchmark, SASS `MUFU.EX2.F16` per half): no packed MUFU rate to take from the H100's fp16 core.
- `tri_tailg` x / o slots 2 / 2 instead of 3 / 1 at d_pair 128: L384 163.8 -> 176.1 us, L768 821 -> 893.
- fused front with 8 drain warps (`FR_DW=8`, one staging tile each) or two staging tiles a drain warp: level (L256 / 384 / 768).
- the gate-from-x tail at d_pair 128 L128 with the PDL chain: level.

After (v1 cuda + v0 triton / pytorch): 113 over 0.667 (3 fixed, 7 newly over by 0.3-2.5 % at L1024-L2048 and one untouched bias-only
L128 case: run drift of the power-capped clock, ratios 0.67-0.68), none slower than the H100, ladder 0. The 1.5x target at L128 needs
the fused front + core + tail kernel (part 4's interface), and at L >= 256 a faster core than parts 2-6 found.

### Part 7 (2026-11-18): the fused L128 kernel -- front + core + tail in one CTA a pair row (cuda:b200_sm100 42)

Records `$W/b2511/agents/r3-triattn/_r/t/r7b/` (before: every L128 eval case, pytorch / cuda / triton, one exclusive measurer run),
`_r/t/r7a/` (after: cuda; accuracy `acc_128.jsonl`); tools `_r/r7/l1.py` (fused vs chain vs fp32 PyTorch, 8 configurations x both nodes x
B1 L128 / B2 L128 9 dead keys / B1 L100 / B2 L113 5 dead keys; module A/B), `tl.py` (globaltimer phases of every CTA, `-DL1_TL=1`),
`pv.py` (compile-flag variants, interleaved), `ovh.py` (graph floor).

**Kernel** (`b200_sources/tri_l128_sm100.cu`, `tri_l128_sm100<CX, HD, GB>`, CX = d_pair 64 / 128, HD = heads x 32 = 64 / 128; extension
`triattn_b200_l128_v3`). One CTA = one pair row (b, n) of 128 tokens = one 128-query x 128-key tile of every head; 448 threads (warp 0
TMA, 1 tcgen05 issue, 2-9 LayerNorm then two softmax groups, 10-13 drain then epilogue):
- phase A: x and W (Wq | Wk, then Wv | Wg on their own barrier) by TMA before the block set-up; LN in place (tri_front's arithmetic); the
  four projections into all 512 TMEM columns (HD 128); q / k / v drained into the core's [128][32] 64B-swizzled operand tiles over the
  retired weight rows; g stays in TMEM (fp32).
- the bias: every CTA writes the bias rows of its own tokens (the front's bias drain, the same bits) and meets the others at a grid
  barrier (a cooperative launch, B L <= 148 CTAs; a monotonic per-grid-size counter) before its bias tiles' TMA; at B L > 148 a light
  bias kernel (`tri_l128_bias`, two threads a row) runs first and the fused kernel is its PDL dependent.
- phase B: per head S = Q K^T into a TMEM buffer, exact row maximum (one key tile: no running maximum, no overflow guard), the logits
  stored over S in the first pass, P = 2^(x - m) in the second (4 of 16 exponential pairs on the FMA pipe), P over S, O_h = P V (ts MMA);
  two softmax groups take heads g, g + 2.
- phase C: per head as soon as its PV retired, u = sigmoid(g [+ gb]) o bf16(O / l) (g rounded to bf16 as tri_tail at d_pair 128 without
  biases, fp32 as tri_tailg / tri_wtailg otherwise) into the out-projection's A tile over K; y = x + u . Wo^T [+ ob] per 32-channel chunk,
  each warp storing its 32 rows; the output may take another geometry than x (Chai-1's untransposed ending node).
Served: inference (no grad, no dropout) at padded L = 128: the fused d_pair 128 path (incl. RoseTTAFold3's gate / out biases) and the
wide path's d_pair 64 rows with 2 x 32 / 4 x 16 (padded) / 4 x 32 heads (out_mode 0 / 1). `MINIWORLD_TRIATTN_B200_L128=0` runs the
three-kernel chain; `MINIWORLD_TRIATTN_B200_L128_GB=0` the separate bias launch.

**Where the time went** (`tl.py`, d_pair 128 starting, graph after an L2 flush; a graph of one tiny kernel is 6.1 us here):
first version with the separate bias launch 28.1 us (the bias kernel 7.7 us and the fused CTAs could not start beside its blocks: the
register file), bias rows in the CTAs + grid barrier 23.7 (kernel 16.6), y signalled before the bias stores / Q K drained before V /
per-head epilogue / early TMA 23.0 (kernel 15.4: set-up 0.5, W 2.3, LN 4.4, projections 6.7, barrier 6.9, bias tiles 7.7, heads 9.7 /
10.2 / 12.5 / 13.1, stored 15.4), logits stored in the first pass 22.1 (the second pass is MUFU-bound: 128 ex2 a thread a head),
four of sixteen pairs on the FMA polynomial 21.5 (2 / 6 / 8 pairs: level / level / slower).

**Accuracy**: every configuration of `l1.py` has the chain's error against fp32 PyTorch (2.8-3.4e-3 relative to max, no non-finite),
fused vs chain within 1-2 bf16 ulps of the output; measurer accuracy `acc_128.jsonl`: 60 GREEN (every L128 eval row of TriangleAttention, AF3, RF3, Chai-1).

**Result** (measurer, eval L128, cuda, graph, L2 flushed; ratio vs the H100 token):

| case (eval L128) | before us (ratio) | after us (ratio) |
|---|---|---|
| TriangleAttention / AF3 d_pair 128 x 4 x 32, starting / ending | 32.5 (0.889-0.890); AF3 ending 32.7 (0.760) | 22.3-22.4 (0.609-0.610; AF3 ending 0.521) |
| RF3 d_pair 128 starting / ending | 32.6 (0.872) / 34.6 (0.781) | 22.3 (0.597) / 24.4 (0.551) |
| d_pair 64 x 2 x 32 starting / ending | 24.4 (0.874) / 24.5 (0.857) | 16.2 (0.580 / 0.566) |
| d_pair 64 x 4 x 16 (TA, AF3) | 28.4-28.5 (0.855-0.864) | 18.2-20.3 (0.551-0.615) |
| d_pair 64 x 4 x 32 (d_hidden 128: TA, Chai-1) | 28.4-30.4 (0.766-0.797) | 18.2-20.3 (0.494-0.546) |
| d_pair 256 x 8, d_pair 384 x 12 (not served: chain) | 51-77.6 (0.672-0.805) | unchanged (0.672-0.803) |

Over 0.667 at L128 eval: 22 -> 6 (the d_pair 256 / 384 rows); none slower than the H100; ladder (cuda < triton < pytorch) 0 -> 0.

Rejected / measured: the softmax's second pass recomputing the logits instead of storing them (+1 us); 6 / 8 polynomial pairs; the
separate bias launch as the default (28 us).
Left at L128: d_pair 256 x 8 and 384 x 12 (4 cases, 0.67-0.81): W4 alone is 512 KiB / 1.2 MiB at those widths -- a fused kernel needs
the weights streamed per K chunk (or a CTA cluster sharing them), a different design. L256 (step b) is not attempted: a pair row
there is two query tiles x two key tiles, the exact single-tile softmax and the one-row-a-CTA layout do not carry over (the core's
online softmax and a 2-CTA row, or 2 rows a CTA pair, would be needed).

### Part 8 (2026-11-18): L256 on a CTA pair, the core's O double-buffered -- measured, nothing kept

Records `$W/b2511/agents/r3-triattn/_r/t/v2/` (true state: every eval requirement case of the six classes, pytorch / cuda / triton, one
exclusive measurer run per length group), `_r/r8/` (tools: `l2.py` fused vs chain vs fp32, `tl2.py` globaltimer phases, `j*.sh`), job logs
`$W/b2511/jobs/r3ta_*.log`; the rejected sources as diffs in `docs/b200/rejected/triattn_part8_*.diff`.

**True state (eval, cuda vs the H100 `cuda` tokens of engine 2.5.14, H100 >= 20 us)**: 104 cases over 0.667 of 274 (L128 6, L256 14,
L384-L768 10 each, L1024 9, L1536 16, L2048 18, L3968 1); none slower than the H100 (worst 0.806, TriangleAttention d_pair 256 L128
ending); ladder (cuda < triton < pytorch) 0 violations. By class: TriangleAttention 47 / 118 (worst 0.806), AF3 35 / 60 (0.786),
RF3 17 / 36 (0.723), Chai-1 5 / 36 (0.738), Chai-1 confidence 0 / 18, Bidirectional 0 / 6. (Against the 2.5.12 tokens the same records
gave 95.)

**1. L256 fused on a CTA pair (`tri_l256_sm100`), built, correct, rejected.** A cluster of 2 a pair row of 256 tokens, rank r owning
tokens 128 r ..; every product a `cta_group::2` MMA issued by rank 0: [q | k] and [v | g] as M256 N 2HD products whose B halves are the
pair's W rows (rank 0 stages Wq / Wv, rank 1 Wk / Wg), S = Q K^T as M256 N256 (keys 0-127 are rank 0's K tile, 128-255 rank 1's: no K
exchange), exact maximum over 256 keys by two softmax warps a row (maxima met in shared memory), PV as M256 N32 ts products whose B half is
16 channels of all 256 keys (each CTA bulk-copies the other 16 channels of its V tile into its peer, `cp.async.bulk.shared::cluster`), u
over the Q tiles, the out-projection M256 N CX with half of Wo a CTA; the bias from `tri_l128_bias`, a first launch (PDL). Output error vs
fp32 PyTorch equal to the chain's on all 8 configurations x both nodes x (B1 L256, B2 L256 9 dead keys, B1 L200, B2 L227 5 dead keys):
2.8-3.2e-3, no non-finite. Module eval (graph, L2 flushed, us): d_pair 128 120.3-128.6 vs the chain 75.8-79.9, d_pair 64 x 2 x 32
59.4-61.4 vs 47.1-49.1, d_pair 64 x 4 heads 90.1-92.2 vs 64.5-71.7. Timeline (`tl2.py`, d_pair 128): a CTA lives ~21 us (LN 1.3,
projections + drain 3.1, ~3 us a head on the one 256-column S buffer, epilogue 4.4) and 512 CTAs on 148 SMs are 3.5 waves of one CTA an
SM, after the bias launch (~14 us until the median CTA of the first wave starts). The phases of a CTA run in series (the MUFU idles
through the projections and the epilogue) while the chain's core keeps three softmax groups fed: even at zero bias cost 3.5 x 21 us is
the chain's time. At L128 the design won because the grid is one wave; at L256 it is not.

**2. The two-row core's O double-buffered (`TA_OB2`), rejected.** With two rows a task (L <= 256: four sub-steps a task) the next
task's first PV waits for the epilogue warps to drain O; two O buffers by task parity fill the 512 TMEM columns exactly (3 x 128 S +
2 x 2 x 32 O). The same bits (err 0 vs the single buffer, deterministic). Core forward, graph, three interleaved rounds (us, single /
double): L128 2 / 4 / 8 / 12 heads 8.32 / 10.38 / 14.50 / 20.62 vs 8.32 / 10.39 / 14.67 / 20.66; L256 20.65 / 32.93 / 64.19 / 95.88 vs
20.92 / 33.36 / 64.86 / 96.68: level at L128, +0.5-1 % at L256. The drain is not on the critical path there.

**3. Measured for the next steps** (chain kernels without PDL, graph, warm L2, d_pair 128 / 64 x 4 heads, us): L256 core 33.5 / 33.2,
front 19.3 / 15.6, tail 18.3 / 12.6; L512 204 / 202, 54.5 / 44.9, 48.7 / 31.8; L768 597 / 593, 111 / 93, 99 / 62. At L256 the front and
tail move 3.5 / 2.8 TB/s (512 tiles on 148 persistent CTAs: 3 or 4 tiles a CTA, the 4-tile CTAs set the time) against 5.4 / 4.6 TB/s at
L768; the core is 2.2x its MUFU floor at L256 (14.9 us), 1.7x at L512, 1.5x at L768. The core's steady sub-step (SASS of the R = 3
build, `_r/r8/out/core_r3.sass`): per key pair 2 FFMA2 + 2 MUFU.EX2 + 1 FADD2 + 1 F2FP + the bias unpack (IMAD.U32 + LOP3), ~536
instructions a warp a 128-key sub-step. `add.rn.f32.bf16` compiles to one `FHADD.BF16` with a `.H1` half select (no unpack): the
predecessor's fp16 form of it (TA_HB) spilled at the 128-register cap and was 4-12 % slower, so it was not rebuilt here.

State after part 8: no source change (cuda:b200_sm100 stays 42); the violation counts are the true state above. L128 d_pair 256 / 384
(step b: a fused kernel would be 256 CTAs = 1.7 waves of a CTA pair a pair row splitting the heads, with Wq|Wk|Wv|Wg streamed; after
part 8's lesson on waves its gain is doubtful) and the core's instruction count (step c) were not built.

### Training tails round 7 (2026-11-18, cuda:b200_sm100 41 / 43)

Records `$W/b2511/agents/r3-triattn-tail/_t/out6/` (`spd_before7_L*.jsonl`, `spd_after7c_L*.jsonl`: every training requirement case of
the six classes, cuda / triton / pytorch, one measurer run per length, exclusive; `acc_after7.jsonl`), `_t/out7/` (determinism checks);
scripts `_t/t6.py`, `_t/detm.py`, `_t/cmpchk.py`, `_t/cbw.py` (the core's backward alone, base vs this tree), `_t/var.py` / `_t/p10.sh`.

**1. Deterministic dbias (the priority fix; cuda:b200_sm100 41).** The query-side backward kernel of all three attention cores (head dim
32 / 64 / 128) added every task's dbias partial into one fp32 buffer by TMA reduce-add, from many CTAs in a run-dependent order: dbias, and
through it dx, the LayerNorm and the bias-projection gradients, differed between runs (the inference agent's `chk.py`, masks, two processes:
91 of 322 tensors). Now each CTA runs one *part* (up to `per` consecutive row groups) of one (sample, head, query tile) unit and writes that
part's own slab (its first task stores, the later ones reduce-add after `bulk_wait<1>` -- the lead thread's older writes, among them the
previous task's write of the same stage, have completed); `triattn_db_slabs` sums the NP slabs of every unit in a fixed order (32 float4
columns x 8 slab lanes, lane sums added in lane order). The CTAs of one part are neighbours across the query tiles, so they step through
the same rows at once and share K / V in L2 as the old round-robin order did. No dbias zero fill any more (`tri_bwd_pre(zero=False)`).
Two processes 0 of 322 tensors differ; `detm.py` 14 configurations (masked / unmasked, fused / wide head dim 32 / 64 / 128, L128-384,
ragged L200, B2) x 5 runs bitwise; tests/gpu/test_triattn_b200.py 87 passed.

Core backward alone (`cbw.py`, 4 heads, us: kv + q + set-up launches): L128 23.1 -> 25.2 (slab sum 3.1), L512 595 -> 609 (bwd_q 292 ->
288, slab sum 16), L640 1098 -> 1110 (545 -> 536, 19), L768 1842 -> 1824 (920 -> 877, 26). Layouts measured on the way: row groups fastest in a CTA's
range (K / V re-read from HBM per query tile: bwd_q L640 546 -> 704 us), query tiles fastest in a CTA's range (38 slabs a unit: bwd_q
819 us, slab sum 128 us; `docs/b200/rejected/triattn_tail7_dbias_slabs_qtile_fastest.diff`).

**2. L128 launches (taken).** The core's bias^T is made by `tri_gate_bwd`'s CTAs beyond the row tiles (fewer tiles than SMs: L128 one
sample; same bits as `tri_bwd_pre`): no set-up launch, graph -1.0 to -1.5 us (GATE_BIAST A/B).

**3. Measured and rejected:**
- side streams (graph branches) for the small launches (drop draw beside pack / front / core, bias^T beside the gate backward, the dWo slot
  sum beside the attention backward): gate_bwd 11.8 -> 21.4 us next to the set-up kernel's blocks, graph 114.4 -> 123.6 (off; the forward
  draw's branch is kept behind `MINIWORLD_TRIATTN_B200_SIDE=1`, unmeasured alone).
- multi-lane TMA issue in the backward producers (gate_bwd / head_bwd / wgrad start-up loads on 8 lanes, bwd_kv / bwd_q stage loads on
  rotating lanes): every kernel within +-0.2 us -- at L128 these producers are not issue-bound
  (`docs/b200/rejected/triattn_tail7_multilane_tma_producers.diff`).
- dbias slabs summed inside tri_head_bwd (4 heads x 32 slabs of loads a row): head_bwd 11.1 -> 22.6 us against the 3 us launch.
- head_bwd's LN / Wb parameters read from global (no constant upload): head_bwd 11.1 -> 15.1 us against the 1.3 us copy.
- gate_bwd's dWo partials summed on head_bwd's 20 CTAs beyond the tiles: 8 MB on 20 SMs takes ~19 us, head_bwd 11 -> 19.3.
- (the three opt-in branches above, even switched off, made head_bwd 18.5 us: removed from the kernel.)
- the wide path's slabs summed by tri_db_rows (no triattn_db_slabs launch): a thread walks 32 slabs in series at L128, d_pair 64 x 2
  x 32 L128 0.672 -> 0.716 of the H100 (`docs/b200/rejected/triattn_tail7_db_rows_slab_sum.diff`).

**Result** (measurer, cuda, graph, L2 flushed, ratio = B200 / H100 against the 2.5.14 H100 tokens, `_t/t6.py before7 after7c`; 198
cases, none under 20 us at the H100): over 0.667 3 -> 5 (fused d_pair 128 L128 AF3 starting 0.791 -> 0.787, ending 0.787 -> 0.787,
starting 0.780 -> 0.784; d_pair 64 x 2 x 32 L128 starting 0.658 -> 0.672 and d_pair 128 L256 ending 0.660 -> 0.669 crossed by ~1 %:
the slab sum launch, ~3 us at L128); none slower than the H100; ladder violations 0 -> 0; cuda after / before geomean 1.009 (0.940 -
1.048) -- the price of deterministic dbias. Accuracy (`acc_after7c.jsonl`): 46 GREEN, 14 YELLOW (all at the token's ratio), no RED. The three fused d_pair 128 L128 training cases stay above 0.667 (one tile a CTA per kernel; ~19 us of ~122
needed); the per-kernel L128 picture is unchanged from round 6 (wgrad 15.7, gate_bwd 12.7, head_bwd 11.5, bwd_kv 10.8, bwd_q 10.4, slot sum
6.8, finish 3.5, slab sum 2.9 us).


### Part 9 (2026-11-18): the bias add without unpack (taken); last-wave half tiles, multi-lane weight streams (rejected) (cuda:b200_sm100 44)

Records `$W/b2511/agents/r3-triattn/_r/r9/` (tools: `cb7.py` core variants with ns a sub-step a SM, `sass.sh` registers / spills / opcode
counts, `chk9.py` bitwise eval / training outputs at L256 between processes, `acc9.yaml`, `j*.sh`), `_r/t/v3/` (after: every eval
requirement case, cuda, one exclusive measurer run per length group), job logs `$W/b2511/jobs/r3ta9_*.log`. "Before" = part 8's true
state (`_r/t/v2/`, the same sources: no change since).

**1. The bias added by `add.rn.f32.bf16` (taken for inference).** The softmax loop made a logit pair as unpack (IMAD.U32 + LOP3) + two
FFMA2 (S x scale log2 e - m, then + bias x log2 e). With the bias staged as bf16(bias x log2 e) it is FFMA2 + two FHADD.BF16 (the
fp32 + bf16 add selects the half in the instruction: no unpack): three instructions a pair instead of four. Core forward (`cb7.py`, 4
heads, graph, two interleaved rounds; ns a sub-step a SM = time x 148 / sub-steps):

| build | regs / spill | L512 | L768 | L1024 | L2048 |
|---|---|---|---|---|---|
| as it was (32-key chunks) | 128 / 0 | 193.1-193.3 us (872 ns) | 588-616 | 1434-1439 (810-813) | 10.77-10.80 ms (760-762) |
| HB, 32-key chunks | 128 / 36 B stores, 44 B loads | 205.5-207.7 (928-938) | 625-642 | 1405-1442 | - |
| 16-key chunks (same bits) | 128 / 0 | 195.6-195.8 (884) | 618-640 | 1444-1448 (815-818) | 10.81-10.87 ms |
| **HB, 16-key chunks** | 128 / 0 | **191.3-192.4 (864-869)** | 613-623 | **1384-1388 (781-784)** | **10.43-10.47 ms (736-739)** |

The HB form with the 32-key chunks spilled at the 128-register cap and lost 2-8 % (the predecessor's fp16 form spilled the same way);
in 16-key chunks (`TA_CK=16`: the S registers of a thread halve, 64 -> 32; the offset is still the maximum of the first 32 keys and
l_r sums the pairs in key order, so the 16-key build alone writes the same bits) it does not spill and is 3 % faster at L1024 / 2048,
1 % at L512 (L768 drifted with the power cap in both rounds). The pre-scaled bias is the fronts' business: the fused d_pair 128 front
takes its bias rows times log2 e (fp32, `param_forms` on Wb), the wide pack (`pack_wide_weights(l2e=True)`, inference only, cached
separately) multiplies Wb by log2 e in fp32 before its one bf16 rounding, so the bias is still rounded once (bf16(bias x log2 e)
instead of bf16(bias): the same relative error). Training keeps the unscaled build (its backward reads the bias; its forward bits are
unchanged); the L128 kernel computes its own bias from the plain pack. `_core(rows, hb=True)` = `triattn_b200_core_r*_hb_v1`
(`-DTA_HB=1 -DTA_CK=16`), in the prebuild list; `MINIWORLD_TRIATTN_B200_HB=0` turns it off. Accuracy (`acc9.jsonl`, measurer, eval
L256 / 512 / 1024 of the five self-attention classes): 86 GREEN, 0 YELLOW / RED (7 skipped: the fp64 reference out of memory),
cuda error 2.8-4.2e-3 against tol 8e-3, as before.

**2. Half tiles in the last wave (built, rejected).** The persistent row-tile kernels (`tri_front_sm100`: the fused d_pair 128 front and
`tri_front64`; `tri_tailg_sm100`: the inference tails at d_pair 64 / 128) ran 128-row tiles round robin; at L256 (512 tiles on 148
CTAs) a CTA does 3 or 4 tiles and the 4-tile CTAs set the time. `tailg::hsplit`: when the last wave holds at most half the CTAs
(ntiles mod 148 <= 74) its tiles run as two 64-row halves (items 0 .. nfull - 1 whole, then halves), so those CTAs do 3.5 tiles' work.
A half item loads its 64 rows by a 64-row tensor map (x, o), computes all 128 rows of the buffer (the upper 64 stale, never stored) and
stores / writes the bias of its 64 rows only. Bitwise equal on / off (`chk9.py`: 9 configurations x both nodes x L256 / L256 with 19
dead keys / L250 ragged with 7, eval; the training forward at L256 too); `MINIWORLD_TRIATTN_B200_HALF=0` turns it off. Applies at L256
B1 (r = 68), L768 (r = 20), not at L384 / 512 / 640 (r = 116 / 124 / 92 > 74).
Module eval (`vt.py`, graph, L2 flushed, median of 50, separate processes, two interleaved rounds, us; half on / off): d_pair 128 L256
starting 75.7-75.8 / 73.7, RF3 d_pair 128 ending 79.8-79.9 / 77.9-78.0, AF3 d_pair 64 x 4 x 16 65.5 / 63.5, d_pair 64 x 2 x 32 49.2 /
49.2: 2-3 % slower. A half item costs about a whole tile (the per-tile chain -- load, LayerNorm, GEMMs, drain -- is latency, not bytes,
at 3-4 tiles a CTA), so the last wave does not get shorter and the extra items add a step. Diff: `rejected/triattn_part9_half_tiles.diff`.

**3. The wide front's and tail's TMA streams from one lane a ring stage (built, rejected).** The SWA agent's finding (one thread issues about one
TMA load per ~600 clk whatever the box) applies to `tri_wfront` (the 2-CTA wide front streams every weight block of every tile:
d_pair 256 x 8 heads = 266 KiB a CTA a tile in 16 KiB boxes from one lane) and `tri_wtail` (g | o | Wo chunks a stage): `WF_ML` --
lane s of the producer warp waits on and fills ring stage s (the same loads, the same bits).
Module eval (us, one lane a stage / lane 0 only, two rounds): d_pair 256 x 8 L128 51.2 / 51.2, d_pair 384 x 12 L128 77.8-77.9 /
71.8-73.7, d_pair 256 ending L256 161.8-163.8 / 153.6: slower (the lanes' divergent spin waits share the producer warp's issue; the
MMA warp consumes the stages in order anyway, so the wide front at these sizes is bound by its 2-CTA GEMM chain, not by the stream).
Diff: `rejected/triattn_part9_wide_multilane_tma.diff`.

**4. HB at the other lengths** (module eval, HB on / off, two rounds, us): d_pair 128 L256 73.7 / 73.7, L384 161.8 / 163.9-164.3, L512
307.1 / 309.2, ending L1024 1722-1727 / 1755-1757, L2048 11.71 / 12.06 ms; d_pair 64 x 2 x 32 L256 45.1 / 47.1; AF3 d_pair 64 x 4 x 16
L256 67.6 / 67.6, L384 143.4 / 143.4-145.4, L1024 1587-1588 / 1618; d_pair 256 x 8 L256 145.6-146.5 / 151.5-151.6. Never slower: on at
every length above 128.

**Result** (measurer, every eval requirement case, cuda, graph, L2 flushed, one exclusive run per length group, `_r/t/v3/`; ratio vs the
2.5.14 H100 tokens, H100 >= 20 us; triton / pytorch from part 8's `v2`):

| class | cases | over 0.667 before -> after | worst before -> after |
|---|---|---|---|
| `TriangleAttention` | 118 | 47 -> 36 | 0.806 -> 0.803 |
| `TriangleAttentionAF3` | 60 | 35 -> 34 | 0.786 -> 0.775 |
| `TriangleAttentionRF3` | 36 | 17 -> 16 | 0.723 -> 0.707 |
| `TriangleAttentionChai1` | 36 | 5 -> 4 | 0.738 -> 0.711 |
| `TriangleAttentionChai1Confidence` | 18 | 0 -> 0 | - |
| `BidirectionalTriangleAttention` | 6 | 0 -> 0 | - |

104 -> 90 over 0.667 (L128 6 -> 6, L256 14 -> 10, L384 10 -> 9, L512 10, L640 10, L768 10, L1024 9 -> 7, L1536 16 -> 10, L2048 18 -> 17,
L3968 1); none slower than the H100; ladder (cuda < triton < pytorch) 0 -> 0. Left: the head-dim-32 core is still ~1.4-1.5x its MUFU
floor at L >= 768 (AF3 d_pair 64 x 4 x 16 0.72-0.78: 16-channel heads cost the B200 as much as 32-channel ones, the H100 not), and L128
d_pair 256 / 384 (0.71-0.80) still run the three-kernel chain. GPU test file not run (`pytest` missing from the B200 env); correctness
here: `chk9.py` bitwise (half tiles), the accuracy measurer (HB), `cb7.py` err 0 (16-key chunks vs 32).

### Training tails round 8 (2026-11-18, cuda:b200_sm100 47)

Records `$W/b2511/agents/r3-triattn-tail/_t/out6/` (`spd_before8_L*.jsonl` / `spd_after8_L*.jsonl`: every training requirement case of the
six classes, cuda / triton / pytorch, one exclusive measurer run per length; `acc_after8.jsonl`); scripts `_t/l1t.py` (fused vs chain vs
fp32, determinism), `_t/l1b.py` / `_t/l1tl.py` (core backward A/B, globaltimer phases), `_t/l1ftl.py` (tri_l128 phases, inference vs
training form), `_t/j8v.sh` / `_t/j8p.sh` / `_t/j8q.sh` (module graph A/B).

**1. The fused L128 kernel in a training form (taken for d_pair 64).** `tri_l128_sm100<CX, HD, GB, TR>` with TR: the drained q / k / v
tiles go out by per-warp TMA stores (box 32 x 32, 64B swizzle) straight from the core's operand tiles (the stores' reads complete
before the residual x / u take the tiles: barrier `qsd`), g / o of each head through a 16 KiB staging pair (TMA stores), the base-2 LSE
from the exact row maximum, the bias rows (already made for the grid barrier), the LN statistics (wide backward) and the dropout scale in
the epilogue (tri_tail's order: update o drop, then + x). Inference instantiations (TR = false) are the same code (bits unchanged).
Kernel alone (d_pair 128, `l1ftl.py`): inference 18.2 us, training form with register stores 24.8, q / k / v by TMA 22.7, g / o by TMA
20.8. Module training step (graph): d_pair 64 x 2 x 32 L128 83.4 -> 77.5 us, d_pair 64 x 4 x 32 (d_hidden 128) 99.8 -> 92.4 us;
d_pair 128 113.8 -> 117.8 (starting) / 114.7 -> 118.9 (ending): the d_pair 128 chain front -> core -> tail overlaps by PDL and the
cooperative launch does not (the profile shows the fused kernel 27.7 us in the graph vs 20.8 alone) -- d_pair 128 stays on the chain
(`MINIWORLD_TRIATTN_B200_L128_TRAIN_D128=1` opts in). Fused vs chain within 1-5e-3 of max (bf16), deterministic.

**2. The core backward's three launches as programmatic dependents (taken for the wide path at L128).** bwd_kv / bwd_q / db_slabs
trigger at entry and wait for the previous grid (`launch_pdl_if(pdl && S == 128)`, `triattn_bwd(pdl=True)` from the wide path):
d_pair 64 x 2 x 32 L128 85.5 -> 83.5 us. The same on every d_pair 128 backward kernel (gate_bwd, head_bwd, wgrad, slot sum, finish too):
L128 114.2 -> 119.6 us, L256 302.2 -> 318.0, L768 3308 -> 3337 (rejected; with the core three alone on the d_pair 128 path +1 us).

**3. Rejected: the L128 core backward in one cooperative kernel** (`docs/b200/rejected/triattn_tail8_l128_bwd_cooperative.diff`,
`tri_l128_bwd_sm100.cu`): R pair rows of one (sample, head) a CTA (B H (128 / R) <= SMs), query side (S, dP -> dS, dQ, dbias rows
in shared memory) and key side (S^T, dP^T -> P^T, dS^T, dV, dK) on separate warp groups, the CTA's dbias slab, a grid barrier and the
slab sum in slab order. Correct (same error as the chain vs fp32, 4e-7 dbias difference, deterministic), but 30.8 us against the chain's
26.7 (graph, L128 4 heads): each pair row's softmax pass costs 2-3.5 us (globaltimer) on 4 rows in series a CTA; with two warps a lane
quadrant it is still 2.2-3.8 us a row, the ablations (no exponentials / no dbias accumulation / no TMEM stores) each cut it little and
all three together to 0.5 us; in the module step +6 us. Not pursued further after the coordinator's barrier measurement (a grid barrier
~1.5 us, a cooperative launch serialises against the previous kernel).

**Result** (measurer, cuda, graph, L2 flushed, against the 2.5.14 H100 tokens; `_t/t6.py before8 after8`; 198 cases, none under
20 us at the H100): over 0.667 6 -> 4 (d_pair 64 x 2 x 32 L128 starting 0.674 -> 0.599, ending 0.668 -> 0.606; fused d_pair 128 L128
AF3 starting 0.792 -> 0.774, starting 0.782 -> 0.774, ending 0.784 -> 0.766, d_pair 128 L256 ending 0.670 -> 0.668 -- these four within
run-to-run noise of before); none slower than the H100; ladder violations 0 -> 0; cuda after / before geomean 0.993 (0.889 - 1.028).
Accuracy (`acc_after8.jsonl`, 15 training rows x L128 / 384): 106 GREEN, 14 YELLOW, no RED. tests/gpu/test_triattn_b200.py 87 passed. Training bitwise over 5 runs (`detm.py`,
14 configurations incl. d_pair 64 L128 masked / unmasked).
Left: the three fused d_pair 128 L128 training cases (0.77; ~16 us of ~121) and d_pair 128 L256 ending at 0.668.
### Part 10 (2026-11-18): exponentials on the FMA pipe, a native 16-channel-head core (cuda:b200_sm100 48)

Records `$W/b2511/agents/r3-triattn/_r/r10/` (tools: `pipe.cu` / `pipe.py` softmax instruction-mix microbenchmark, `cb10.py` core variants
on 16-channel heads, `ext10.py` extreme inputs, `j*.sh`; `out/acc10.jsonl`, `out/cmp_n_h1.txt`), `_r/t/v4/` (before: every eval
requirement case, pytorch / cuda / triton, on the 2.5.14 base), `_r/t/v5/` + `_r/t/v7/` (after, cuda; L128 / 256 / 768 re-measured on the
merged tree), job logs `$W/b2511/jobs/r3ta10_*.log`.

**1. What the softmax loop costs (microbenchmark, `pipe.py`).** 1 CTA an SM, 12 / 16 / 24 warps, 8 independent key pairs an iteration,
clk a warp key pair a sub-partition: MUFU ex2 alone 16.07 (the 16 / clk / SM floor); the loop's full mix since part 9 (FFMA2, two FHADD,
two MUFU, FADD2, F2FP) 16.36; everything but the MUFU 7.1; `ex2.approx.f16x2` 17.3-18.1 (two MUFU.EX2.F16 in SASS: no MUFU saving).
So in isolation the mix is MUFU-bound and its other instructions hide; moving exponentials to the FMA pipe pays: of 8 pairs, 1 / 2 / 4 /
8 on the degree-3 polynomial with the clamp 15.3-15.7 / 14.1-15.2 / 17.2-17.7 / 25.7; without the clamp 2 of 8 reads 13.2-13.6
(the clamp is needed: masked keys are -inf). Part 4's rejection (`TA_POLY`, 1 of 16 +2.6 %) was measured before part 9 took the
bias unpack out of the loop.

**2. `TA_POLY` in the inference (HB) core (taken).** Core, 4 heads, graph, us (two rounds, `cb7.py`): head dim 32 L256 31.9 / 30.9 /
30.8 / 31.1-31.3 (0 / 2 / 4 / 6 of 16 pairs), L768 582-646 / 561-595 / 586-601 / 617-625, L2048 10386-10446 / 10389-10439 / 10637 /
11091. Module A/B (`vt.py`, POLY on / off, two rounds): d_pair 256 x 8 L768 1605 / 1645, 1589 / 1630; d_pair 384 x 12 L640 1755 / 1817,
1782 / 1790; d_pair 128 L768 level; AF3 d_pair 64 L768 711 / 717, 694 / 719; Chai-1 d_pair 64 x 4 x 32 L2048 11498 / 11404, 11501 / 11432
(+0.7 %: a build without the polynomial at L >= 2048 was measured with the measurer and gave one miss more at L2048 than with it,
so it was not kept). Taken: 4 of 16 at two rows a task (L <= 256), 2 of 16 above (`POLY_INFER`, `MINIWORLD_TRIATTN_B200_POLY=0`
turns it off). Relative error of the polynomial 7.5e-5 (under P's bf16 rounding).

**3. 16-channel heads (AF3 / TriangleAttention d_pair 64 x 4 x 16), `TA_D16` (taken).** The head slot stays 32 channels, but the d_pair
64 front writes 1.0 into v's padded channel 16 (`tri_front64(v1=True)`, inference only), so the PV GEMM's channel 16 is the row sum
l = sum P (bf16 P, fp32 accumulation, as the H100's tensor-core row sums). The core then drops the FADD2 a pair, the per-tile overflow
guard (sum, vote, shared-memory row state) and one of the two QK^T MMAs (channels 16..31 of q / k are zero); its output channels 16..31
are written 0 as before. Overflow safety moves to the drain: a row whose l is not in [2^-100, 3e38) or whose O is not finite (a logit
~120 log2 units above the first-32-key offset, or a provisional offset after 32 masked keys) is recomputed exactly from q / k / v / the
bias in global memory (`d16_exact`, CUDA cores, two passes; rare and slow). `ext10.py` (L256 / L768, plain, a +150 nat spike on one key,
the first 40 keys masked, both, all but 3 keys masked): every output finite, error vs fp32 torch 2.3-3.9e-3 (the base core 3.0-5.5e-3),
the spike rows exact. Core (`cb10.py`, 4 heads, us, HB base -> D16 -> D16 + POLY): L256 31.7-32.2 -> 28.8 -> 28.8 (4), L512 191 -> 183,
L768 577 -> 547-555 -> 529-535 (2), L1024 1309-1316 -> 1235, L2048 10011-10019 -> 9500 -> 9162-9202 (2). Extension
`triattn_b200_core_r*_hb_d16_p*_v2_tt8` (prebuild list), front `triattn_b200_mod_v19_tt7`; `MINIWORLD_TRIATTN_B200_D16=0` turns it off.

**Correctness.** Measurer accuracy (`acc10.jsonl`, eval L256 / 512 / 1024 of the five self-attention classes): 172 GREEN, 0 YELLOW / RED.
Bitwise vs part 9 (`chk9.py`, 70 tensors: 54 eval outputs, 16 training outputs / gradients): only the 54 eval tensors differ (1-2 bf16
ulps of the output); training is bitwise unchanged (the HB / D16 builds are inference only; the training core build has two unused
kernel parameters more).

**Result** (measurer, every eval requirement case, cuda, graph, L2 flushed, one exclusive run per length group; ratio vs the 2.5.14
H100 tokens, H100 >= 20 us; triton / pytorch from `v4`):

| class | cases | over 0.667 before -> after | worst before -> after |
|---|---|---|---|
| `TriangleAttention` | 118 | 35 -> 22 | 0.805 -> 0.803 |
| `TriangleAttentionAF3` | 60 | 34 -> 32 | 0.775 -> 0.762 |
| `TriangleAttentionRF3` | 36 | 15 -> 15 | 0.707 -> 0.705 |
| `TriangleAttentionChai1` | 36 | 4 -> 4 | 0.697 -> 0.713 |
| `TriangleAttentionChai1Confidence` | 18 | 0 -> 0 | - |
| `BidirectionalTriangleAttention` | 6 | 0 -> 0 | - |

88 -> 73 over 0.667 (L128 6 -> 6, L256 10 -> 9, L384 9 -> 8, L512 10 -> 8, L640 9 -> 7, L768 10 -> 7, L1024 6 -> 3, L1536 11 -> 10,
L2048 16 -> 14, L3968 1 -> 1); none slower than the H100; ladder (cuda < triton < pytorch) 0 -> 0. AF3 d_pair 64 x 4 x 16 went 0.70-0.77
-> 0.67-0.71 (L1024 1576 -> 1458 us, L2048 11046 -> 10179) but most of those rows still need 1-7 %; TriangleAttention d_pair 64 x 4 x 16
is now under 0.667 except L256 ending (0.669). Chai-1 L2048 starting moved 0.667 -> 0.671 (within the run-to-run spread of these
11.5 ms cases; the module A/B shows +0.7 % from the polynomial there).
Left: L128 d_pair 256 / 384 (the three-kernel chain, 0.67-0.80) untouched; L >= 1536 head dim 32 cores at 0.68-0.72 (MUFU-bound loop
plus the per-sub-step skeleton); AF3 d_pair 64 x 4 x 16 0.67-0.71 (front / tail still on the padded 32-channel slots).

### Training tails round 9 (2026-11-18, cuda:b200_sm100 49; 50 with the inference part 10)

Records `$W/b2511/agents/r3-triattn-tail/_t/out6/` (`spd_after9_L*.jsonl`, `acc_after9.jsonl`; before = `spd_after8_*`, the same
training sources); scripts `_t/hw1.py` / `hw4.py` / `hw5.py` (module step A/B per split), `_t/hw3.py` (gradients vs the old build,
determinism), `_t/tl9.py` (globaltimer phases, `-DTB_TL=1`), `_t/box.py`, `_t/var.py` with mod defines.

**H100 path read first** (`h100_sm90.py` train_forward / train_backward, `h100_sources/triattn_mod_sm90.cu`): the same stages as the
B200 fused d_pair 128 path -- pack, front, core, tail; gate_bwd (per-CTA dWo partials, grid a multiple of L / 128), bwd_kv, bwd_q (dbias
into a zeroed fp32 buffer), head_bwd (one tile a CTA), wgrad (2 x splits CTAs, a CTA owns a PAIR of projections = the B200's NG = 2),
tri_grad_reduce, tri_param_finish; plain launches (no PDL / graph branches in the backward), 16 KiB-class TMA boxes. Nothing in its
training step is missing on the B200 path; the B200 has one launch more (the deterministic dbias slab sum) and one less (no dbias
memset).

**1. head_bwd + wgrad in one CTA-split launch (taken at 512-2048 row tiles).** Neither reads the other's outputs and neither fits beside the
other on an SM (head_bwd 226 KiB shared, wgrad 512 TMEM columns), so `tri_hw_bwd_sm100` runs head_bwd's body on CTAs 0 .. NH - 1 and
wgrad's on the rest (both kernels are now device bodies taking their CTA index / count; `tri_head_bwd` / `tri_wgrad` keep their own
launches). Module step (graph, d_pair 128, starting / ending, us; NH of 148): L256 294.7 -> 287.9 / 298.0 -> 290.6 (NH 60), L384
645.0 -> 633.4 / 653.3 -> 639.9 (64), L512 1229.8 -> 1218.4 / 1247.6 -> 1232.6 (64); L640 / 768 within 0.7 % (not taken); NH 32 / 80 / 96
lose 2-8 %. dx bitwise as before, parameter gradients regrouped (<= 2.5e-6 relative), five runs bitwise (`hw3.py`, `detm.py`).
At L128 (one tile a CTA) it does not pay: 111.6 us -> 109.9-113.3 by the split, the split kernel 25.8 us against 11.3 + 15.7. Phases
(`tl9.py`): head_bwd's CTA is 9 us of serial chain on one tile (W + d landed 2.0 us, gy 5.4, epilogue to the reduce 8.9), wgrad's 13-15 us
(x 1.8, xhat 3.6, all MMAs 6-11, flush +4): each role's time grows with its tiles a CTA, so splitting the SMs moves work, it does not
overlap it. The CTA counts the brief asked for: bwd_kv 148 CTAs (256 tasks), bwd_q 128 -- no idle SMs to give the wgrad at L128.

**2. Measured and left off (macros, default off; the default build is the previous code):**
- `HB_GIC` (head_bwd's channel half as a compile-time constant: c_hb's Wb / gamma as immediate constant operands): the CTA's epilogue
  3.6 -> 2.8 us, the kernel 11.5 -> 11.6 us, step 112.7-113.2 -> 112.9-113.8 (no gain).
- `WG_DST` (wgrad's M_t slots stored straight from the registers, no staging / TMA store): wgrad 15.5 -> 17.5 us.
- `HB_NR` / `HB_NX` / `WG_NR` (deeper d rings: head 4 slots with one x buffer, wgrad 6 / 8): head 11.4 -> 11.1, wgrad 15.6 -> 15.2-15.3, step
  -0.3 to -1.0 us (within the run spread; not taken).
- `HB_BOX3` (the box-count finding of transition part 14: both K halves of a tensor in ONE 3-D 32 KiB box instead of two 16 KiB boxes,
  bitwise the same): head_bwd 11.5 -> 12.2 us, step +1.1 to +1.5 us -- the MMA now waits for both halves and the next tensor for both
  slots; at 10 boxes a tile the per-box issue cost is not this kernel's bound.
- not built: M = 64 tiles for bwd_q / bwd_kv at L128 -- 512 (head, row, query-tile) units of the step over 148 SMs: bwd_q runs 4 rows a
  CTA on 128 CTAs, bwd_kv 2 x 2 rows on 108 of 148; finer units can at best reach 3.46 rows an SM (-13 % of a 10 us kernel, ~1.3 us)
  while doubling the per-unit set-up; the slab sum through a cluster: 32 slabs a (head, query tile) at L128 do not fit one cluster
  (<= 16), and the in-order DSMEM exchange after all roles is the pattern rejected in round 6 (+3-5 us).

**Result** (measurer, cuda / triton / pytorch, graph, L2 flushed, one exclusive run per length; vs the 2.5.14 H100 tokens; 198 cases):
over 0.667 4 -> 3 (d_pair 128 L256 ending 0.668 -> 0.649); the three fused d_pair 128 L128 cases unchanged (0.773 / 0.774 / 0.768); none
slower than the H100; ladder 0 -> 0; cuda after / before geomean 0.999 (0.971 - 1.024). Accuracy (`acc_after9.jsonl`, 15 training rows
x L128 / 384): 46 GREEN, 14 YELLOW (each at its token's ratio), no RED.

### Part 11 (2026-11-18): compact 16-channel heads end to end (cuda:b200_sm100 50)

Records `$W/b2511/agents/r3-triattn/_r/r11/` (tools: `chk11.py` compact vs padded core on extreme inputs / module bitwise, `t11.py`
module A/B in separate processes, `j1.sh`-`j4.sh`; `out/acc11.jsonl`, `out/n.pt`), `_r/t/v8/` (after: every eval requirement case,
cuda, one exclusive measurer run per length group), job logs `$W/b2511/jobs/r3ta11_*.log`. "Before" = part 10's records (`v4` triton /
pytorch, `v5` / `v7` cuda) on the same 2.5.14 H100 tokens.

**H100 path read first** (`h100_sm90_generic.py` `COMPACT`, `triattn_h16_sm90.cu`): AF3 / TriangleAttention d_pair 64 x 4 x 16 runs
three launches on the H100 -- the one-kernel front writing q / k / v / g with 16 channels a head (fp16), the fp16 core loading them
through 3-D maps `[rows][H][16]` whose 32-channel box zero-fills channels 16 .. 31 (one k16 score step, the gate in its epilogue),
cublasLt `x + u Wo^T` with K = 64. The B200 path (part 10) wrote 32-channel padded slots from the front, read them padded in the core
and the tail (K = 128): twice the bytes of q / k / v / o.

**1. Compact heads (taken).** `TA_D16C` (with `TA_D16`): the core's q / k / v maps are the H100's 3-D maps (box (32, 1, 128), 64B
swizzle: the smem tiles stay the padded [128][32] ones, the math is part 10's); v's 1.0 in channel 16 (the PV GEMM's row sum) is
written by the PV issue warp into each landed V tile (it waits the stage's barrier itself, 32 lanes x 4 rows, 16-bit stores at the
swizzled chunk 2 ^ ((row >> 1) & 3), `fence.proxy.async`, off the P critical path); out stores channels 0 .. 15. The front is
`tri_front64` at heads x 16 = 64 output columns (no kernel change: the front is head-agnostic) from the compact rows of the inference
pack (`pack_wide_weights(l2e=True)` adds `[Wq; Wk; Wv; Wg]` and Wo without their padded rows / columns, the same bf16 values); the
tail is `tri_wtailg<64, 64>` (K = 64) on the compact o, Wg and Wo. Inference only (`D16C_INFER`, `MINIWORLD_TRIATTN_B200_D16C=0`
turns it off); extension `triattn_b200_core_r*_hb_d16c_p*_v3_tt8` (prebuild list).
Correctness: compact == padded bit for bit (core: 10 configurations incl. a +150 spike, 40 masked keys, all but 3 masked, L256 / L768;
module: AF3 and TriangleAttention, both nodes, B1 L256 / B2 L250 with 7 dead keys / B1 L384, random weights: outputs equal);
`chk9.py` 70 tensors (eval + training) identical to part 10; measurer accuracy `acc11.jsonl` 172 GREEN.
Module eval (`t11.py`, graph, L2 flushed, two rounds, us, compact / padded): AF3 d_pair 64 L256 53.3 / 61.4, ending L384 122.9 /
145.4, L768 626.7 / 692-694, L1024 1354 / 1458-1460 (-7 to -15 %).

**2. The B200 TMA box-count finding applied (coordinator, Transition part 14)** -- measured:
- the core's bias tile (two 64-key halves) as one 3-D box `[S / 64][rows][64]` (taken: same bits, level -- d_pair 128 L384 / L768
  161.8 / 782 us as before; one box fewer a key tile);
- K | V of a sub-step as one box (rejected: level). The front put q | k | v in one buffer and the core loaded K | V through a map
  whose outer dimension steps from k to v (compact: 4-D): AF3 d_pair 64 L256 / 384 / 768 / 1024 53.2 / 122.9 / 626.7 / 1353.7 vs
  53.3 / 122.9 / 626.7 / 1353.7, d_pair 128 L384 / 768 161.8 / 782.4 vs 161.8 / 782.3 (two boxes a ~1 us sub-step are not the
  bound); `rejected/triattn_part11_kv_one_box.diff`;
- the wide front's weight stream (`tri_wfront`, 2-CTA) with KB = 2 / 3 K chunks a box (3-D map `[C / 64][rows][64]`, KB-chunk ring
  stages; rejected: level, d_pair 384 ending +0.5-2.6 %): d_pair 256 L128 51.2 / 51.2 / 51.2 (KB 1 / 2 / 3), ending 53.2 x 3, d_pair
  384 L128 73.7 x 3, ending 75.8 / 76.2-77.6 / 77.8, AF3 d_pair 256 L128 51.2 x 3, d_pair 256 L256 145.4 / 145.4 / 145.3, d_pair 384
  L640 1792-1846 / 1776-1832 / 1761-1858 (noise); bits equal. This front is bound by its 2-CTA GEMM chain, not by its 20 weight boxes
  a tile (as part 9's multi-lane issue already showed); `rejected/triattn_part11_wfront_kchunk_boxes.diff`.

**3. The 16-channel core's FMA-pipe exponentials at L >= 1536 (rejected)**: TA_POLY 2 / 3 / 4 / 6 of 16 pairs, core alone, 4 heads,
two rounds: L1536 3867 / 3895-3959 / 3997-4061 / 4075-4103 us, L2048 9160-9167 / 9200-9205 / 9339-9385 / 9690-9797: 2 (the
default) stays.

**Result** (measurer, every eval requirement case, cuda, graph, L2 flushed, one exclusive run per length group; ratio vs the 2.5.14
H100 tokens, H100 >= 20 us; triton / pytorch from `v4`):

| class | cases | over 0.667 before -> after | worst before -> after |
|---|---|---|---|
| `TriangleAttention` | 118 | 22 -> 24 | 0.803 -> 0.805 |
| `TriangleAttentionAF3` | 60 | 32 -> 20 | 0.762 -> 0.763 |
| `TriangleAttentionRF3` | 36 | 15 -> 16 | 0.705 -> 0.702 |
| `TriangleAttentionChai1` | 36 | 4 -> 2 | 0.713 -> 0.709 |
| `TriangleAttentionChai1Confidence` | 18 | 0 -> 0 | - |
| `BidirectionalTriangleAttention` | 6 | 0 -> 0 | - |

73 -> 62 over 0.667 (L128 6, L256 9 -> 6, L384 8 -> 6, L512 8 -> 6, L640 7 -> 5, L768 7 -> 5, L1024 3 -> 2, L1536 10 -> 11, L2048
14, L3968 1); none slower than the H100; ladder (cuda < triton < pytorch) 0 -> 0. AF3 d_pair 64 x 4 x 16 is under 0.667 at L256-1024
(0.59-0.63) and at L1536 / 2048 0.673-0.679 (core ~92 % of the module there, 646 ns a sub-step a SM). The TriangleAttention / RF3
movements (d_pair 128 L1024 ending 0.680 / 0.671, d_pair 256 L1536 0.673-0.677, L2048 ending 0.671) are run drift of paths this
part did not change (the same bits, the same kernels but for the bias box).
Left: d_pair 128 x 4 x 32 at L256-2048 (~40 cases, 0.67-0.72: the head-dim-32 core plus the per-tile front / tail at L256-768), L128
d_pair 256 / 384 (6 cases: the three-kernel wide chain), d_pair 64 x 4 x 32 / RF3 d_hidden 256 at L2048 (0.67-0.71).

| case | H100 (launches; fused stages; key techniques) | B200 after part 11 | H100 technique not ported, and why (measured) |
|---|---|---|---|
| d_pair 64 x 4 x 16 (AF3 / TA), L > 128 | 3: front (LN + q / k / v / g + bias, compact, fp16), fp16 core (compact 3-D loads, gate in the epilogue, row sums on the tensor core, key tiles past the mask skipped, 8 / 16 polynomial pairs), cublasLt out-proj + residual | 3, PDL-chained: `tri_front64` (compact q / k / v + bias, no g), TA_D16C core (compact 3-D loads, row sum from the PV GEMM's ones channel, 2 / 16 polynomial pairs), `tri_wtailg` (gate from x, K = 64, residual) | fp16 core / `ex2.f16x2`: two MUFU ops a pair on sm_100a (part 6 microbenchmark); 8 / 16 polynomial pairs: 4 / 6 of 16 slower here (part 10, this part); gate in the core epilogue: the gate-from-x tail saves the g stream (part "fronts and tails"); key-tile skipping: +1-4 % on the unpadded speed-measurer masks (part 3) |
| d_pair 128 x 4 x 32, L > 128 | 3: `tri_front` (fp16 q / k / v / bias, no g), fp16 core (bias as the accumulator's initial value, `[O | l] += P [V | 1]`), `tri_tail` (gate from x) | 3, PDL-chained: `tri_front` (no g), HB core (bias pre-scaled by log2 e added by FHADD, 2-4 / 16 polynomial pairs), `tri_tailg` (gate from x) | bias-in-accumulator: tensor pipe becomes the bound (part 3 / 4: L768 617 vs 584 us); ones column on V: 48 O columns a row exceed TMEM with three S buffers and three rows (part 4) |
| d_pair 128 / 64 x 32, L128 | 3 launches | 1 (`tri_l128`, part 7) | - |
| d_pair 256 x 8 / 384 x 12, L128 | 3: wide front (CTA groups of 4, LN folded into the pack), fp16 core with the gate, cublasLt out-proj | 3, PDL-chained: `tri_wfront` (2-CTA tcgen05), HB core, `tri_wtail` | LN fold: the front is GEMM-chain bound, the fold only moves arithmetic (audit below); fewer weight boxes: level (this part) |

### Training tails round 10 (2026-11-18, cuda:b200_sm100 53)

Task: fold the three small launches after the core backward (dbias slab sum 2.9 us, slot sum 6.8, param finish 3.6 at L128 d_pair
128) into the preceding kernels with a deterministic last-CTA / barrier pattern. Records `$W/b2511/agents/r3-triattn-tail/_t/out6/`
(`spd_after10_L*.jsonl`, `acc_after10.jsonl`; before = `spd_after9_*`); scripts `_t/tl10.py` (kernel timeline of one graph replay),
`_t/tl11.py` (globaltimer phases, `-DTB_TL=1`), `_t/wf1.py` / `wf2.py` / `wf3.py` (gradients vs the old path, determinism, interleaved
graph A/B).

**Sizing first: these are not launch-bound reductions.** At L128 the slot sum reads 74 x 256 KiB of M_t slots + 128 x 64 KiB of
gate_bwd dWo partials = 27 MB (6.8 us = 4 TB/s), the slab sum 32 slabs x 4 heads x 64 KiB = 8 MB; L256: 11 + 9.5 MB and 16 MB. One
CTA (or the last 8) cannot sum that in 1-2 us; a fold has to run on the whole grid behind a barrier.

**1. Everything in tri_wgrad (`WFold`, fold = 1; measured, opt-in only).** wgrad's grid is at most one CTA an SM (all resident): after
the flush a grid barrier (CUTLASS form: fence.acq_rel + relaxed red, acquire polls; counters reset by the finish ticket's last CTA),
each CTA sums a contiguous range of the M_t columns over the slots in slot order and writes dW_t = gamma M + beta cs directly plus its
dgamma / dbeta shares; the last CTA (ticket) adds the 64 shares a channel and writes dgamma | dbeta | dWb. The wgrad slots rebalanced
(128 tiles: 74 -> 64 splits, same 2 tiles a CTA, 19 -> 16 MB). Correct (dx bitwise, parameters <= 7e-7 relative, bf16 <= 1.4e-4),
deterministic. Phases at L128 (globaltimer from the first CTA): flush done 14.1-14.3 us, barrier passed 15.6-16.1 (the first form
with a generation word, SC fences and returned atomics: 3 us), slot ranges done 21.0 (5.1 us for 16 MB, ~3 TB/s: fewer loads in flight
than the 520-block slot-sum kernel; 16 loads a thread instead of 8 changes nothing), last CTA's finish 22.3 -> 24.8. Total after the
flush 10.5 us vs slot sum + finish 10.3: graph L128 +0.4 us, L256 +6, L384 +7 (fewer wgrad CTAs than the 128 column ranges at
512-2048 tiles: 2 rounds). NG = 4 (one projection a CTA, half the slot bytes) on top: -0.4 us. Kept behind
`MINIWORLD_TRIATTN_B200_WFOLD=1`.

**2. dbias slab sum inside bwd_q (rejected, `docs/b200/rejected/triattn_tail10_dbias_slab_fold_bwdq.diff`).** The CTAs of a
(b, h, query tile) unit meet at a per-unit barrier (grid2 <= SMs by construction), each sums 1 / NP of the unit's [128][S] block
over the NP slabs in slab order and writes dbias. Correct, deterministic; bwd_q 10.3 -> 13.4 us against bwd_q + slab sum 10.3 +
2.8: graph d_pair 128 L128 +0.2 to +1.0 us, L256 / L384 +0.5 to +1.3, d_pair 64 L128 / L256 +0.2 / +0.6 (`wf2.py`). Removed from the core
source (the inference core builds untouched).

**3. Taken: the gate backward's dWo partials summed in tri_wgrad's idle row warps (fold = 2, the default).** The dWo partials are
ready before wgrad starts; wgrad's row warps 2-5 are idle after their xhat loop while the MMAs and the flush run (L128: done at 6.1-7.7
us of a 14 us chain; L256: done by 41-52 us, flush 50-55). Each CTA sums its share of the float4 columns over the partials in CTA order
(4 slot lanes, added in lane order) and writes dWo in the parameters' dtype; tri_slot_sum loses the dWo segment (27 -> 19 MB at L128:
one wave of 520 blocks) and tri_param_finish its 16 cast blocks. Only dWo regrouped (<= 2e-7 relative fp32, 4e-6 bf16), every other
gradient bitwise; deterministic (`detm.py` 14 configurations x 5 runs). Module step (graph, interleaved, us, off -> on): L128 starting
113.5 -> 111.9, ending 114.0 -> 112.4, AF3 113.3 -> 111.8, L256 286.4 -> 285.2, L384 ending 634.0 -> 632.3, L512 1213.6 -> 1212.1,
L640 within noise. The H100 path (`h100_sm90.py`) keeps tri_grad_reduce / tri_param_finish as separate launches.

**Result** (measurer, cuda / triton / pytorch, graph, L2 flushed, one exclusive run per length; vs the 2.5.14 H100 tokens; 198 cases):
over 0.667 3 -> 3 (d_pair 128 L128 starting 0.774 -> 0.770, AF3 starting 0.773 -> 0.767, ending 0.768 -> 0.763); none slower than the
H100; ladder 0 -> 0; cuda after / before geomean 1.000 (0.973 - 1.028). Accuracy (`acc_after10.jsonl`, 15 training rows x L128 / 384):
46 GREEN, 14 YELLOW (each at its token's ratio), no RED.

Left: the three d_pair 128 L128 training cases need ~120.9 -> 104.8 us (-16 us). The tails measured here are bandwidth (27 + 8 MB of
fp32 partials), not launches; the remaining lever is fewer partial bytes (fewer, longer wgrad / bwd_q CTAs lengthen their one-tile chains;
the cluster / DSMEM reduction of round 6 cost +3 us), or the fused front + core + tail of part 3 (5 / 6).

### Training tails round 11 (2026-11-18): fewer partial bytes -- measured, nothing kept (no source change; cuda:b200_sm100 stays 54)

Task: cut the fp32 partial bytes of the L128 d_pair 128 training tail (M_t slots 19 MB, dbias slabs 8 MB) by summing them inside
clusters before HBM. Records `$W/b2511/agents/r3-triattn-tail/_t/` (`wcl1.py` / `ft1.py`: gradients vs the old path, determinism,
interleaved graph A/B; `tl12.py`: globaltimer phases, `-DTB_TL=1 -DWCLX=..`; `tl10.py` kernel timeline), `_t/out6/spd_after11_L*.jsonl`.

**1. tri_wgrad's M_t partials summed in a cluster over DSMEM (built, rejected; `rejected/triattn_tail11_wgrad_cluster_dsmem_mt.diff`).**
The SWA DW_CL pattern: CS CTAs of one tensor group (consecutive splits, nsp rounded to a multiple of CS with the same tiles a CTA,
`cudaOccupancyMaxActiveClusters` check), after a cluster barrier each CTA stages its [128 c][128 o] partial in the d ring as CS
column blocks, CS lanes send block o to rank o by one `cp.async.bulk.shared::cluster.shared::cta` each (mbarrier complete_tx), rank o
adds the CS blocks in rank order and stores one slot a cluster. Correct (dx bitwise, parameters <= 2e-7 relative fp32 / 1e-4 bf16),
bitwise deterministic for CS 4 / 8 / 16 (6 configurations incl. masked, ragged L200 B2, L640 / L768). But: graph step L128 112.0 ->
122.4 us (CS 4, 8, 16 alike), L640 / L768 within noise. Kernel timeline: tri_wgrad 16.9 -> 28.0 us, **tri_slot_sum 4.9 -> 5.1 us with 8x
fewer M_t bytes (19 -> 2.4 MB)**. Phases (CS 8, two rounds of 64 KiB a CTA): staging 64 KiB from TMEM to shared 3.5-4 us a round
(also with the TMEM loads removed: ~3 us of STS.128 alone), the 8 x 8 KiB bulk copies + rank-order sum ~3-4 us a round, the two cluster
barriers ~1 us; everything off (`WCLX=15`) +1 us. So: the slot sum is not bound by its bytes, and the exchange costs more than the
TMA-store flush it replaces. The dbias-slab form (same pattern on bwd_q) was not built: its sum (triattn_db_slabs, 2.8 us) is the same
kind of near-empty launch.

**2. bf16 partial slots: not built** -- with (1) showing the slot sum time independent of its bytes, halving them cannot pay.

**3. tri_slot_sum + tri_param_finish in one launch (built, rejected; `rejected/triattn_tail11_slot_finish_one_launch.diff`).** Block =
one M_t row (t, c): its 128 o summed over the slots (the same lane order: the same M bits), cs_t summed in the same pass, dW_t[o][c]
and the row's dgamma / dbeta shares written, the last block (ticket) finishes dgamma | dbeta | dWb. Correct, deterministic. First form
(two serial slot passes) 11.4 us; all loads of a lane in flight 8.8 us vs 5.1 + 3.6 us: graph L128 +0.2 to +1.0 us. Not taken.

**4. The L128 d_pair 128 training step, kernel by kernel** (graph timeline, us; bytes = HBM/L2 traffic of the kernel, floor = bytes / 4 TB/s):

| kernel | us | bytes | floor us | x floor |
|---|---|---|---|---|
| drop draw + pack | 2.1 + 3.4 | ~4 MB | 1.0 | 5 |
| front (LN + q/k/v/g + bias) | 10.7 (overlaps the core by PDL) | 4 in + 16.5 out | 5.1 | 2.1 |
| core forward | 17.4 (8.4 under the front) | 12.5 in + 4.1 out | 4.2 | 2.2 (net) |
| tail (gate, Wo, residual) | 10.0 | 12 in + 4 out | 4.0 | 2.5 |
| gate_bwd | 12.6 | 16 in + 12 out + 8 dWo partials | 9.0 | 1.4 |
| bwd_kv / bwd_q | 10.8 / 10.3 | ~21 / ~21 (+8 dbias slabs) | 5.3 / 7.3 | 2.0 / 1.4 |
| db_slabs | 2.8 | 8 in + 0.25 out | 2.1 | 1.3 |
| head_bwd | 11.6 | 20 in + 4 out | 6.0 | 1.9 |
| wgrad | 16.7 | 20 in + 8 dWo in + 19 slots out | 11.8 | 1.4 |
| slot sum + finish | 4.9 + 3.5 | 19 in (L2) | 4.8 | 1.7 (but byte-independent, see 1) |

No kernel is above 2.5x its byte floor; the ones near 2-2.5x (front, core, tail) are single-tile-per-CTA latency chains, already the
subject of parts 7 / 8 (fused L128 kernel, CTA-pair core: measured, not faster in the d_pair 128 training form). The small tails are
launch-shaped (~3-5 us whatever their work).

**Result** (measurer, cuda / triton / pytorch, graph, L2 flushed, one exclusive run per length, on b200-2.5.x a5e6d54b8 with
cuda:b200_sm100 54; vs the 2.5.14 H100 tokens; 198 cases): over 0.667 3 -> 3 (d_pair 128 L128 starting 0.770 -> 0.774, AF3 0.767 ->
0.773, ending 0.763 -> 0.768: run spread), none slower than the H100, ladder 0 -> 0, cuda after / before geomean 0.999 (0.965 - 1.027).

### Part 12 (2026-11-18): the H100 fp16 core's pieces one by one; HB core at L128 for the wide chain (cuda:b200_sm100 54)

Records `$W/b2511/agents/r3-triattn/_r/r12/` (tools: `cb12.py` HB core variants, timing + error vs fp32, `EXT=1` extreme inputs; `dbg.py`;
`l128p.py` wide L128 module A/B; `j1.sh`-`j4.sh`; `out/vs_before.txt` / `vs_after.txt`, `out/acc12.jsonl`), `_r/t/v9/` (after: L128 eval
cuda, one exclusive measurer run; the other lengths are part 11's `v8`, no source change on their path), job logs `$W/b2511/jobs/r3ta12_*.log`.
"Before" = part 11's records (`v4` triton / pytorch, `v8` cuda) against the 2.5.14 H100 tokens: 62 / 274 over 0.667, none slower than
the H100, ladder 0.

**The H100 d128 core (`triattn_h16_sm90.cu`) piece by piece, measured on the B200** (core alone, head dim 32, 4 heads, graph, two
interleaved rounds unless said; "base" = the HB core as served):

| H100 piece | B200 form built | result |
|---|---|---|
| row sums on the tensor core, `[O \| l] += P [V \| 1]` | `TA_ONES`: after a sub-step's PV, eight ts MMAs M128 N16 K16 with B = a 1 KiB ones tile; no FADD2 a pair, no per-tile overflow guard (a bad row is recomputed exactly at the drain, `d16_exact<32>`). TMEM: two rows accumulate l in columns 480-511; three rows (3 x 128 S + 3 x 32 O = 480: no room for three 16-column sums) pass each sub-step's partial through a two-slot ring (commit per row `lfu[r]`, the group reads + adds in key-tile order, `lem[slot]` frees it) | correct (extreme inputs: spike / 40 masked keys / all but 3 masked, every output finite; error vs fp32 2.4-3.0e-3 against the base's 3.1-4.9e-3), but not faster: two rows L256 4 heads 30.87 / 30.89 vs 30.81 / 30.89, L128 8 heads 14.50 vs 14.48, 12 heads 20.4 vs 19.4-19.8 (slower); three rows (ring) L384 89.3-89.5 vs 87.3-87.6 (+2 %), L512 187.8-187.9 vs 184.7 (+1.7 %), L768 591 / 567 vs 554 / 577 (power-cap drift), L1536 4461 / 4367 vs 4380 / 4513, L2048 10360-10367 vs 10367-10404 (level); the three-row ceiling without the ring (`TA_ONESX`, wrong sums) L1536 -2..-4 %, L768 within the drift. Rejected: the eight extra MMAs sit in front of QK^T(y + 3) in the in-order tensor pipe, eating what the softmax saves |
| fp16 P (probabilities as the PV's fp16 A operand) | `TA_PH` (timing only: P packed by `cvt.rn.f16x2.f32`, PV idesc fp16): same instruction count as bf16 P (one F2FP a pair either way) | level: L256 30.85-30.87 vs 30.81-30.89, L768 561 / 581 vs 558 / 610, L1536 4423-4520 vs 4500-4505. On the H100 fp16 P pays because P comes out of `ex2.f16x2` already packed; on sm_100a `ex2.f16x2` is two MUFU ops (part 6) and the scores sit in fp32 TMEM |
| bias fragment as the score accumulator's initial value | `TA_BMMA` (parts 3 / 4) | rejected before: L768 617-621 vs 584-587 us; ceiling at zero tensor cost <= 4.6 % |
| fp16 score accumulator + HFMA2 logits | not built: a tcgen05 fp16 accumulator still takes one 32-bit TMEM column an element (CUTLASS `tmem_frg_1sm<c_type>` with 32-bit storage), so no TMEM / load saving; HFMA2 needs the bias already in the accumulator (row above) | - |
| `ex2.approx.f16x2` | microbenchmark (parts 6 / 10) | two MUFU ops a pair on sm_100a: no saving |
| 8 / 16 polynomial pairs | `TA_POLY` (parts 10 / 11) | 2 / 16 (three rows) and 4 / 16 (two rows) are the B200 optimum |
| lazily raised offset (vote) | first-32-key offset + overflow guard / PSHIFT (round 3) | equivalent, kept |
| sub-step stall breakdown (asked: TMEM-load vs MUFU vs FMA vs barrier stalls of the ~730-830 ns sub-step) | `ncu` WarpStateStats / SourceCounters | not available: `ERR_NVGPUCTRPERM` (no access to the performance counters for this account on the B200); the clock64 buckets of parts 2 / 4 / 5 (`TA_FTL`) remain the breakdown: wait for S ~265, work ~918, bias / drain ~200-280 clk a sub-step of ~1500 |

A cross-sub-step prefetch of S (the tcgen05.ld of sub-step k + 1 before the exponentials of k) is not possible in this schedule: a group's
next S is QK^T(y + 3), issued only after PV(y) consumed this sub-step's P (part 5 measured separate P buffers: wait 266 -> 57 clk but the
work grew by the same).

**Taken: the HB core at L128 for the wide chain.** The inference integration took the HB core (bias bf16(bias x log2 e), FHADD adds, 4 / 16
polynomial pairs at two rows) only above L128 (the L128 kernel makes its own bias); d_pair 256 / 384 at L128 are not served by the L128
kernel and ran the plain build. Now `hb` is on at L128 whenever the L128 kernel does not serve the width
(`integrations/triattn_b200.py`). Module eval (`l128p.py`, graph, L2 flushed, two rounds in separate processes, us, HB off / on): d_pair
256 x 8 starting 51.20 / 49.15-49.18, d_pair 384 x 12 starting 73.73 / 71.68, ending 75.78 / 75.74-75.78.
The default builds' bits are unchanged by this part's source edits (`chk9.py`: 70 eval + training tensors identical to part 11).

**Result** (measurer, every eval requirement case, cuda, graph, L2 flushed; ratio vs the 2.5.14 H100 tokens, H100 >= 20 us; triton /
pytorch from `v4`): 62 -> 61 over 0.667 (TriangleAttention 24 -> 23: d_pair 384 L128 starting 0.672 -> under; d_pair 384 ending 0.714 ->
0.695, AF3 d_pair 256 ending 0.749 -> 0.722, d_pair 256 0.788 / 0.805 -> 0.785 / 0.803); none slower than the H100; ladder 0. Accuracy
(`acc12.jsonl`, eval L128 of the five self-attention classes): 64 GREEN.
Left: L128 d_pair 256 x 8 (4 cases 0.72-0.80) and d_pair 384 ending (0.695) need a different structure (the three-kernel wide chain: front
~16 / core ~14 / tail ~15 us at d_pair 256, PDL-overlapped); L256-L2048 d_pair 128 x 4 x 32 (0.67-0.72) are bound by the head-dim-32
core, which none of the H100 core's pieces above made faster on the B200; L256-768 tile quantisation of the fronts / tails (half tiles
measured and rejected in part 9).

### Part 13 (2026-11-18): the FA4 sm100 forward schedule against the head-dim-32 core -- measured, nothing kept

Records `$W/b2511/agents/r3-triattn/_r/r13/` (`cb13.py`: HB core variants, timing, bitwise vs the first variant, error vs fp32, `EXT=1`
extreme inputs incl. spikes late in a key tile; `j1.sh`), job log `$W/b2511/jobs/r3ta13_j1.log`; the variant as
`docs/b200/rejected/triattn_part13_splitp.diff`.

**What the FA4 forward (`flash_attn/cute/flash_fwd_sm100.py`) does, and where the core already is.** FA4: one MMA warp, one load warp, two
softmax warpgroups on two 128-row Q tiles, a correction warpgroup, S / P aliased in TMEM (P as bf16 over S, the PV MMA's A operand from TMEM,
`.ts`), the MMA warp's order per KV block `PV_i(j - 1)` then `QK^T_i(j)` into the same S buffer (the next S only after PV consumed P --
FA4 has no second S buffer either), `split_P_arrive` (P's first 3/4 released early, the last 1/4 committed separately), the row sum
updated after P's release, ex2 emulation on the FMA pipe, and a running maximum whose O rescale runs on the correction warps. The core:
three softmax groups (one more than FA4) with one 128-column S / P buffer each + three 32-column O = 480 of 512 TMEM columns, P already
in TMEM as the PV MMA's A operand (`mma_ts`, no shared-memory round trip), QK^T and PV on two issue warps, QK^T(y + 3) ordered after
PV(y) by the K / V stage barrier (FA4's order), the polynomial ex2 (part 10), and no running maximum (offset from the first 32 keys;
nothing to correct, so no correction warpgroup). A second S buffer per group does not fit (2 x 128 + 32 a group: 3 groups 864, 2 groups
576 columns); a separate P buffer (part 5 `SEPPX`) was measured level. The FA4 piece not yet built was the split P arrive.

**`TA_SPLITP` (built, rejected).** The softmax warps arrive a first P barrier after chunk `SPLITP - 1` of the 8 16-key chunks (wait::st, fence,
arrive); the PV warp issues those K16 MMAs then and the rest after the full arrive (same MMAs, same order: the same bits). The overflow
guard rewrites P / O, which an early-consumed P forbids: a warp arrives early only if every lane's partial sum is <= 2^64 and no offset is
provisional; a lane whose full-tile sum then passes 2^64 is marked (+inf row sum) and its row recomputed exactly at the drain
(`d16_exact<32>`). Correct: plain inputs bitwise equal to the base at L256 / 512 / 768 / 2048; extreme inputs (spike in chunk 4 / chunk 7 of a
tile / of the offset tile, a +70 nat spike in chunk 7 of every tile, 40 masked first keys, rows with 3 live keys) all finite and
deterministic, equal bits except the every-tile late spike (rows recomputed exactly: error vs fp32 2.6e-3, base 4.2e-3). Core, 4 heads,
graph, two interleaved rounds, us:

| L (rows) | base | split after 6 of 8 chunks | after 4 of 8 |
|---|---|---|---|
| 256 (2) | 30.84 / 30.90 | 33.51 / 34.55 | 33.36 / 33.06 |
| 512 (3) | 185.3 / 185.4 | 192.2 / 192.1 | 195.3 / 194.9 |
| 768 (3) | 569.4 / 593.4 | 586.2 / 624.0 | 610.7 / 633.5 |
| 2048 (3) | 10393 / 10431 | 10648 / 10667 | 10743 / 10766 |

+2.5-8 % everywhere. Like the 64-key halves (part 2) and `SEPPX` (part 5), starting PV / the next QK^T earlier does not pay: the mid-tile
store wait + fence + vote cost the softmax warps more than the shortened S -> PV -> QK^T chain gives back. Three measurements now agree that
the chain latency is not the core's bound (removing the wait entirely: <= 1.5 %); the time is in the softmax warps' own instruction stream
sharing the sub-partition with the MUFU.

State after part 13: no source change (default builds untouched; `cuda:b200_sm100` stays 54). The counts over 0.667 are part 12's.

### Part 14 (2026-11-18): the head-dim-64 core's softmax loop; ones inside the PV MMA; warpgroup registers (cuda:b200_sm100 56)

Records `$W/b2511/agents/r3-triattn/_r/r14/` (`cb14.py` head-dim-64 core variants, `cb13.py` head-dim-32 core variants, `EXT=1` extreme
inputs; `t14.py` module A/B in separate processes; `lst.py`; `j1.sh`-`j6.sh`; `out/acc14.jsonl`, `out/vs_after.txt`), `_r/t/v10/` (after:
every eval requirement case, cuda, one exclusive measurer run per length), job logs `$W/b2511/jobs/r3ta14_*.log`. "Before" = part 12
(`v9` cuda, `v4` triton / pytorch) against the 2.5.14 H100 tokens: 61 / 274 over 0.667, none slower than the H100, ladder 0.

**Where the 64-channel heads stood.** Every 64-channel-head eval case (AF3 d_pair 512 x 8, RF3 64 x 4 x 64, Chai-1 / confidence 256 x 4)
was already under 0.667 except RF3 d_hidden 256 L2048 (0.669 / 0.669); the rest 0.37-0.59 (`lst.py`). The Chai-1 misses at L1536 / 2048
are d_pair 64 x 4 x 32 (head dim 32), not the head-dim-64 core.

**1. The head-dim-64 core with the head-dim-32 inference loop (`TA_HB` in `triattn_dh64_sm100.cu`, taken).** The core still ran the
training loop: bias unpacked and scaled per pair (two FFMA2), a per-chunk running maximum (16 FMNMX a 32-key chunk + the raise branch).
The HB build: bias pre-scaled bf16(bias x log2 e) by the wide front (`pack_wide_weights(l2e=True)`), FFMA2 + two FHADD a pair, the
offset fixed at the first 32 keys with triattn_sm100.cu's overflow guard after the tile (PSHIFT; the exact recompute from q / k at 64
channels), TA_POLY of 16 pairs, 16-key chunks; the row state in registers (the shared memory is full). Inference only (no LSE); the
training build is the previous code (its kernel takes four unused pointer parameters more; `triattn_b200_dh64_v7_tt7`). Core alone
(`cb14.py`, 4 heads, graph, two rounds, us, base -> HB, POLY 2): L256 48.7-49.0 -> 47.3-47.4, L512 266.6-267.9 -> 249.8-256.2, L768
815-913 -> 732-787 (power-cap drift), L1536 6289-6407 -> 5786-5937, L2048 14839-14952 -> 13467-13549 (-3 / -6 / ~-7 / -7 / -9 %).
POLY 0 / 2 / 4 and 32-key chunks are within the run spread (the loop is not MUFU-bound with two softmax groups). Error vs an fp32
reference with the same (pre-scaled) bias: max 3.8-4.3e-3, rms 2.3e-3 (base 3.2-4.3e-3 / 2.3e-3); extreme inputs (spikes early / late /
in every tile, 40 masked first keys, rows with 3 live keys, q x 6) all finite and deterministic. Module A/B (`t14.py`, graph, L2 flushed,
two rounds, us, off / on): RF3 64 x 4 x 64 L768 1333 / 1262, L2048 17732 / 16713, 17538 / 16567; AF3 512 x 8 L768 3393 / 3250, 3367 /
3267; Chai-1 confidence 256 L768 1534 / 1442, 1499 / 1432; Chai-1 256 L1536 ending 8245 / 7894, 8260 / 7884 (-4 to -6 %).
`MINIWORLD_TRIATTN_B200_HB64=0` turns it off, `MINIWORLD_TRIATTN_B200_POLY64` sets the polynomial pairs; extension
`triattn_b200_dh64_hb_p2_v1` (prebuild list). Accuracy (`acc14.jsonl`, AF3 / RF3 / Chai-1 / confidence eval L256 / 512 / 768): 108 GREEN.
Default builds bitwise unchanged (`chk9.py` 70 eval + training tensors equal to part 12).

**2. The row sum inside the PV MMA, `[O | l] += P [V | 1]` (built twice, rejected).** The form that needs no front change: V stays
MN-major and the MMA's N grows by 16; the B operand's second MN block (channels 64 .. / 32 ..) is reached through the descriptor's
leading-byte offset, which is pointed at a small block of ones (the H100 d64 core's `desc_mv` trick). Correct on every extreme input
(bad rows recomputed exactly at the drain), error vs fp32 lower (2.4e-3 vs 3.8e-3).
- head dim 64, `TA_O1` (N = 80, the 2 KiB ones block in the row-sum handoff's place): 2 x 128 + 2 x 80 = 416 columns, i.e. two S
  buffers instead of three. Core, 4 heads: L256 49.7-49.9 vs 47.2, L768 767-780 vs 732, L2048 14476-14478 vs 13467-13513 (+5 to
  +7.5 %): the lost QK^T look-ahead costs more than the FADD2 and the guard. Kept in the source, default off.
- head dim 32, `TA_OV` (N = 48, the 1 KiB ones tile): two rows 3 x 128 + 2 x 48 = 480; three rows 2 x 48 + 32 = 512 with rows 0 / 1 on
  the ones and row 2 keeping its FADD2 sum and guard (its own instantiation of the softmax groups). Core, 4 heads: L256 (two rows)
  30.86-30.88 vs 30.85-30.87 (level, as part 12's `TA_ONES`), L512 196.8-203.8 vs 184.9-185.0, L768 598-620 vs 559-578, L1536
  4528-4541 vs 4364-4499, L2048 10600-10625 vs 10394-10421 (+1 to +10 %). `rejected/triattn_part14_ov_ones_in_pv.diff`.
So neither the FADD2 a pair nor the per-tile guard is what bounds the head-dim-32 / 64 loops.

**3. Warpgroup-aligned roles + `setmaxnreg` for the three-row head-dim-32 core (`TA_WG`, rejected).** 512 threads: warpgroup 0 = TMA /
QK^T / PV / idle, warpgroups 1-3 = the softmax groups, `setmaxnreg.dec` / `.inc` so 32-key chunks fit without the 128-register spill.
Same bits. Core, 4 heads, us: issue warps at 40 / 56 registers spill (L768 1093 / 2007 / 4162 vs 554-556: the issue loops' carried
counters); at 96 / 136 L768 568-598 (16-key) / 621-626 (32-key) vs 559-569, L2048 10539 / 10965 vs 10372. And the default core with
32-key chunks (spilling) is level with 16-key chunks at L1536 / 2048 (4365 / 10485 vs 4364 / 10394) and +5-8 % at L512 / 768: the
32-key chunk has no gain to unlock. `rejected/triattn_part14_wg_setmaxnreg.diff`.

**4. Module-to-module PDL (front as a dependent of the previous module, trigger in the tails): not built.** The speed measurer times one
module a graph, so it cannot change a requirement case; left for when a whole-trunk graph is the measured unit.

**Result** (measurer, every eval requirement case, cuda, graph, L2 flushed, one exclusive run per length, `v10`; triton / pytorch
from `v4`; vs the 2.5.14 H100 tokens, H100 >= 20 us): RF3 16 -> 14 over 0.667 (d_hidden 256 L2048 0.669 / 0.669 -> 0.631 / 0.617; every
64-channel-head case now 0.37-0.63); TriangleAttention 23 -> 24 and Chai-1 2 -> 4 by cases at 0.667-0.670 on paths this part did not
change (same bits: run drift at the boundary, e.g. Chai-1 d_pair 64 L1536 starting 0.668, L2048 starting 0.670, TriangleAttention d_pair
64 x 2 L2048 0.670); total 61 -> 62 / 274; none slower than the H100; ladder 0. What is left is the head-dim-32 core (d_pair 128 x 4 x 32
L256-2048, 0.67-0.72) and the L128 wide chain (d_pair 256 / 384): neither ones-in-PV nor more registers moved the head-dim-32 core.

| H100 2.5.13 `triattn_d64_sm90.cu` (head dim 64) | B200 head-dim-64 core after part 14 |
|---|---|
| fp16 q / k / v / bias from an fp16 front; the bias fragment as the fp16 score accumulator's initial value (no bias unpack) | bias pre-scaled bf16, added by FHADD without unpack (`TA_HB`); a tcgen05 fp16 accumulator still takes 32-bit TMEM columns, and bias-in-accumulator was rejected on the head-dim-32 core (tensor pipe, part 3 / 4) |
| HFMA2 logits, `ex2.approx.f16x2`, 8 / 16 pairs on the polynomial | `ex2.f16x2` is two MUFU ops on sm_100a; fp32 polynomial 0 / 2 / 4 of 16 measured level (2 kept) |
| lazy offset with a warp vote (full row maximum only when a lane passes offset + 8) | offset fixed at the first 32 keys, overflow guard after the tile (PSHIFT / exact recompute): no per-chunk maximum (taken) |
| `[O \| l] += P [V \| 1]` m64n72k16, ones through the V descriptor's leading-byte offset | built (`TA_O1`, N = 80, same trick): TMEM then holds two S buffers, +5-7.5 % (rejected) |
| Q tiles through the K \| V ring (a fourth K \| V stage) | not built: Q double-buffered beside three K \| V stages (the shared memory is full; the core is softmax-issue bound, not load bound) |
| two consumer warpgroups x 64 query rows, R = 2, dynamic task counter, live key tiles | two softmax groups x 128 rows (a third needs 3 x 128 + 3 x 64 = 576 TMEM columns); dynamic counter and live key tiles rejected on B200 (parts 3 / H100 audit) |

### H100 techniques -> B200 (audit, 2026-11-17)

| H100 technique (docs/h100/triangle_attention.md, the `cuda:h100_sm90` notes) | B200 |
|---|---|
| fused d128 front: LN in registers + q / k / v / g + head-major scaled / masked bias in one kernel | used (`tri_front`, tcgen05) |
| core: persistent grid, query tile fastest, bias shared by the task's pair rows | used (2 / 3 rows a task) |
| core: dynamic task counter (`TA_DYN`) | not applicable: measured 4-5 % slower on B200 (launch groups of ~36 864 tasks instead) |
| core: bias fragment as the score accumulator's initial value | built and rejected (round 3 part 3, `TA_BMMA`: the bias added by eight M128 N16 K16 MMAs a sub-step): softmax work -29 %, but the tensor pipe becomes the bound (an MMA costs ~65-73 clk whatever its N): L768 617 vs 584 us, L2048 10466 vs 10603; re-measured in part 4: 621 vs 587 us, and with the bias gone from the softmax at zero tensor cost (the ceiling of the form) only 577 vs 605 (guard off): <= 4.6 %, its 2-CTA form ~1.3 %; also needs a finite mask value (0 x -inf in the identity MMA) |
| core: row sums on the tensor core (`[O | l] += P [V | 1]`) | not applicable: a ones column on V makes O 48 columns a row (+48 TMEM columns: 528 > 512 with the three S / P buffers, part 4); the overflow guard would also need the MMA's sum |
| core: fp16 inference (`ex2.f16x2`, HFMA2 logits) | not applicable as a MUFU saving: `ex2.approx.f16x2` issues as two MUFU operations on sm_100a; the HFMA2 logits would only pay together with the bias-in-accumulator form |
| core: FA3 order (QK^T of y + 1 before the exponentials of y) | used (QK^T up to NS - 1 sub-steps ahead, two issue warps) |
| core: PV GEMM count (the H100 issues PV per 64-row warpgroup) | measured, not applicable: a tcgen05 MMA costs ~46.7 clk at any N <= 64, so the PV is 8 x 46.7 clk a sub-step; the O^T form keeps the 8 K16 steps (M = 64 for 32 channels); the 2-CTA form halves it per SM, but removing every PV MMA buys only 2.4 % (part 4) -- the tensor pipe is 34 % busy |
| one issue thread for QK^T and PV (in-order tensor pipe, no retire wait) | rejected for the three-group core (part 4): L768 618 vs 585 us |
| core: lazily raised running maximum / speculative offset + fix list | equivalent: offset from the first 32 keys, raised only by the overflow guard (P shifted by a power of two) |
| core: mask's live key tiles counted in the kernel (inference) | built and rejected (round 3 part 3): the same bits, 0.884 / 0.913 of the core at L1024 / 2048 with the 1/8-padded mask, but +1-4 % on unpadded inputs (the speed measurer times unpadded masks); see part 3 |
| core: every other exponential pair on the FMA polynomial (`TA_POLY16`) | rejected: 1 / 2 / 4 of 16 pairs 598 / 600 / 674 us vs 583 (L768) |
| tail: `sigmoid(g) o o` as the A operand, residual + dropout in place, one rounding; RF3 biases in fp32 | used on the fused (d128, no biases) and wide tails |
| tail: inference gate projected from x (front writes q / k / v only) | used since round 3 fronts / tails (`tailg_sm100.cuh`; y / u in TMEM, x / o rings; d_pair 128 L > 128 and RF3, d_pair 64 with heads x slot 64 / 128): L768 eval d128 830 -> 793 us; not at d_pair 256 / 384 (Wo and Wg, 256 KiB, do not fit beside the rings) nor Chai-1's untransposed / dual outputs |
| tail: ending node reads o transposed, x / out as stored | used (`TRANS` geometry of the tails) |
| wide front at d256 (CTA groups, LN folded into the weight pack) | B200 wide front is a 2-CTA tcgen05 kernel (`tri_wfront`); the LN fold not applied: write-bound (1.2 of 1.5 GB at L768 are q / k / v / g stores, 5 TB/s), the fold only moves arithmetic |
| one-kernel front at d64 (two weight stages, no LN statistics stored), AF3 flipped bias by the whole warpgroup | used since round 3 fronts / tails: d_pair 64 runs the fused front's design (`tri_front64`, weights resident, bias on the LN warps; inference and training): L768 64 x 2 x 32 front 113 -> 64.5 us. 16-channel heads stay zero-padded in q / k / v: compact needs the core's 3-D zero-padding loads (core owner) |
| `ln_bias_fwd` (bias from the LN kernel, d_pair 384) | not applicable: the B200 wide front makes the bias in its GEMM |
| training dropout scale in one Philox launch | used since part 2b (the same `pack.cu`) |
| backward: gate_bwd -> bwd_kv / bwd_q (P from LSE, deterministic dbias slabs) -> head_bwd -> wgrad (partials summed in order) | used (KV-owned + query-owned kernels, TMEM-resident operands, dbias by fp32 reduce-add; wgrad deterministic on both paths since cuda:b200_sm100 10) |
| bias-only: logits pass, planes GEMM, fused tail, fused gate backward | used (2.5.12 structure built for sm_100a); beyond the H100: the ending node's per-head contractions (O_e, dV_e, dP_e) as one tcgen05 launch each for every head at L <= 256 (the H100 runs a cuBLAS batched GEMM + a sum per head), the softmax backward's row dot and the LN parameter partials in one launch each (fronts / tails part 3) |
| width-generic training: G = D^T [xhat | 1] and dWo = dy_s^T u as cuBLAS GEMMs | B200 at L128, d_pair <= 256: both in one tcgen05 launch, cluster split-K reduced in rank order (`tallk_sm100.cu`); cuBLAS above (fronts / tails part 3) |
| PDL chains | inference: used since part 6 (front -> core -> tail, the dependents' set-up and x / weight loads before the wait: L128 -4 %, L256 -1.5 %); earlier: graph-node gaps are ~0.7 us a launch at L128 inference (measured round 3 part 1); in the d128 training backward (wgrad beside head_bwd, slot sum / finish as dependents) measured -0.6 us at L128, +4 us at L256 (fronts / tails part 2) |
| fused d128 tail with RF3 gate / out biases (fp32 inside the tail) | used for inference since round 3 fronts / tails (RF3 d_pair 128 on the fused path: L768 0.920 -> 0.808 ms); training stays wide |
| training launch count (width-generic ~20 a step) | B200 at the same count (~20); L128 bound by one tile a CTA per kernel, not by launches |
| more softmax warps / buffers than TMEM allows (H100: registers, not TMEM) | measured, not applicable (part 5): a CTA pair adds no TMEM per SM; with free TMEM a fourth softmax group is 5-10 % slower and separate P buffers are level (wait for S 266 -> 57 clk, work +217 clk); the sub-partition is issue-slot bound (+~5 ALU instructions a pair = +320 clk a sub-step) |

## Gaps

Every requirement case routes to the B200 kernels; none is refused, none violates the ladder, and cuda is ahead of anthropic / cueq
in every case they run (>= 1.21x). What is left:

- **1.5x against the H100 on the softmax-bound cases** (inference L >= 256, 104 cases after part 2):
  the forward core is bound by its softmax warps (three a sub-partition since part 2: L768 x 4 heads 583 us, MUFU 69 % busy; see
  part 2 for the ablation split and the rejected 64-key halves). The H100 kernel runs its
  inference softmax on fp16 pairs (`ex2.f16x2`, bias as the accumulator's initial value, row sums on the tensor core); on
  sm_100a `ex2.approx.f16x2` issues as two MUFU operations, the MUFU rate per SM is the H100's, and the measured ceiling of the
  H100 form here is ~14 % of the core (no bias add, no row-sum add: 625 -> 539 us). Reaching 0.667 on these rows needs ~520 us at
  L768 x 4 heads (now 583; the MUFU floor is ~400 us): the logits' bias unpack + FFMA2 (~100 us) would have to leave the
  softmax warps (bias in TMEM before QK^T, see part 2), and the S -> PV -> QK^T chain hidden without per-half overhead; plus the tail projecting the gate from x (the H100's inference tail: one [L, L, HD]
  stream less written and read, ~4 % of the module). d_pair 64 x 4 x 16 additionally pays the zero-padded 32-channel slots in the
  front and tail (the H100 keeps 16-channel heads compact there). RoseTTAFold3 d_pair 128 runs the wide front / tail (its
  biases), ~5 % slower than the fused d_pair 128 kernels: the fused tail / front would need the bias adds.
  Part 4 measured the tensor-core routes out: removing every PV MMA buys 2.4 %, the bias-in-accumulator form at zero tensor cost
  <= 4.6 %, a 2-CTA core ~1-1.3 %; the levers left (a fourth softmax group, separate P buffers, row sums on the MMA) all need TMEM
  columns beyond the 512 (480 in use). Part 5 measured them with free TMEM: a fourth group is slower and separate P buffers are level, and a CTA pair adds no TMEM per SM; the core is issue-slot bound per sub-partition, so the lever is fewer instructions a key in the softmax loop.
- **L128 inference: done for d_pair 64 / 128 by the fused kernel (part 7)**; d_pair 256 / 384 still run the chain (0.67-0.80: their weights do not fit one CTA).
- **L128** (23 inference cases, 15 training; part 3: the core is two thirds skeleton -- 6.3 of 10.4 us with no loads, MMAs or arithmetic --, and a one-pass four-row core was not faster): three kernels of 6-10 us each for ~3.5 us of steady work; a CTA of the core lives
  ~6.6 us of the kernel's 10.4 us (2 tasks of 2 sub-steps: start-up, first loads and the epilogue). The gaps between graph nodes
  are only ~0.7 us a launch (inference), so PDL would not pay; it needs one kernel for the whole L128 update, or M = 64 tiles.
  Training L128 is launch-count-bound: after the pack kernel a wide step is ~25 launches, 6 of them the reference's dropout draw
  (`torch.rand` under capture: the draw, two generator-state fills, compare, cast, divide), 4 cuBLAS.
  The interface a fused front + core + tail L128 kernel needs from the front / tail sources is written down in part 4 (to be assigned).

- **Fronts / tails after round 3 (fronts and tails)**: the gate-from-x tail covers d_pair 64 / 128 only (d_pair 256 / 384 would need
  Wo and Wg streamed: 256 KiB at d256), not Chai-1's untransposed / dual outputs (the tail's rows are not the front's there). 16-channel
  heads stay zero-padded in q / k / v / o (compact needs the core's 3-D zero-padding loads, as the H100 fp16 core has). The fused d_pair
  128 training path is deterministic since cuda:b200_sm100 10 (part 2 below; ~2-4 us a step at L128). Training at L128 is
  per-kernel latency (one tile a CTA), not launch count.
- **Extreme logit spreads**: a tile with a logit more than ~127 log2 units (~88 nats) above the row's first-32-key maximum still takes
  the recompute branch of the head-dim-32 core (11x at L128, ~20x at L768 when it happens on most tiles: synthetic q x10 inputs). No
  requirement case hits it. Closing it needs either a per-tile offset check inside the steady loop (measured on the branch at 4-15 %
  of the steady state) or a recompute through the MMA warp (the K tile is still in shared memory then). The head-dim-64 / 128 cores
  use a lazily raised running maximum and do not have this cliff.
- **Bias-only forms** stay a composition of cuBLAS GEMMs and the CUDA-core glue on channel planes (now without the torch broadcast
  ops); no tcgen05 kernel. A native path would fuse the LayerNorm into the planes GEMM and the gate into the out-projection
  (MN-major A from the planes), saving ~0.6 GB of the ~2 GB per L768 step (~30 %); not attempted (cuda already 2.2-4.0x triton).
  The ending direction's per-head contractions run on tcgen05 at L <= 256 since part 3 (all bias-only cases under 0.667); at L >= 384
  the native one-tile-a-CTA kernels lose to cuBLAS's per-head GEMMs (no K reuse of P across channels: L768 eval 0.559 vs 0.422 ms).
- **Small-L latency, kernel level** (L128, all widths): 3 kernels of ~7-18 us each (d_pair 256: front 18, core 14, tail 16 us); fewer, fused
  launches or M = 64 tiles would help. cuda now beats every other backend there.
- **Training-gradient YELLOWs** (33 cases, <= 1.36x tol, unchanged): bf16 rounding of the intermediates (dq / dk / dv / dg in bf16,
  dbias by fp32 reduce-add); tightening would need fp32 gradient rows. Same picture on H100 / A100.
- **Unjudged sizes**: eval at L >= 1536 (and d_pair 256 / 384 at L >= 1024) has no fp64 gold on a 178 GB card (skipped for every
  backend).
- **Heads of 96 channels** run zero-padded to 128 (no requirement row has them).

## Changes outside sm100-only code

- `kernels/integrations/bidir_triattn_h100.py` (shared by H100 / A100 / RTX 5090 / B200): the new statements run only under
  `_b200()`; the other cards' path is unchanged.
  Round 3 part 2b: `forward_single` draws the dropout scale with the sm_100a build of the H100 Philox kernel, also under `_b200()`.
- `kernels/triangle_attention/cuda/h100_sources/pack.cu` (the H100 pack / Philox source): unchanged; additionally built for sm_100a
  as `triattn_b200_pack_v1` (`b200_triattn._pack`); the H100 build and extension name are untouched.
- `triattn_wide_sm100.cu` / `triattn_sm100.cu` are sm_100a-only sources (extension names bumped: `triattn_b200_wide_v39`,
  `triattn_b200_core_r*_v18`, `triattn_b200_dh64_v4`, `triattn_b200_wide_v40`).
- Fronts / tails part 3: `bidir_triattn_h100.py` -- the native ending GEMMs (`_bo()`), the row dot, the column sum and the first-sample
  accumulation run only under `_b200()` (`_bo()` returns None elsewhere); the H100 / A100 / RTX 5090 statements are the ones before
  (the starting dP keeps its copy into dp there). New sm_100a-only sources `b200_sources/bidir_sm100.cu` (`triattn_b200_bidir_v3`) and
  `b200_sources/tallk_sm100.cu` (`triattn_b200_tallk_v2`); `triattn_wide_sm100.cu` `tri_db_rows` (`triattn_b200_wide_v42`).
- Part 7: new sm_100a-only source `b200_sources/tri_l128_sm100.cu` (extension `triattn_b200_l128_v3`); `b200_triattn.py` routes inference at padded L128 to it (B200-only module). No shared source changed.
- Training tails round 8: sm_100a-only sources only -- `tri_l128_sm100.cu` gains the training template form (extension
  `triattn_b200_l128_v3_tt8`; the inference instantiations unchanged), `triattn_sm100.cu` `triattn_bwd(pdl=)` (`triattn_b200_core_r*_tt8`).
- Training tails round 6: sm_100a-only sources only (`triattn_sm100.cu` `triattn_bwd(bias_t=)`, `triattn_wide_sm100.cu` `tri_bwd_pre`,
  `triattn_mod_sm100.cu` slot sum); extension names `triattn_b200_core_r*_v18_tt1`, `triattn_b200_wide_v42_tt1`, `triattn_b200_mod_v16_tt1`.
- Part 9: `triattn_sm100.cu` (sm_100a-only) gains the `TA_HB` / `TA_CK` knobs (defaults: the previous bits); the inference build
  `triattn_b200_core_r*_hb_v1` is a new extension name, added to the sm100 prebuild list (`triangle_attention/cuda/__init__.py`);
  `b200_triattn.py` / `integrations/triattn_b200.py` (B200-only modules): the inference bias rows times log2 e (`pack_wide_weights(l2e=)`,
  a separately cached inference pack). No shared source changed.
- Part 10: `triattn_sm100.cu` (sm_100a-only) gains `TA_D16` (default off: the previous bits) and two kernel parameters (v, bias pointers
  for the D16 drain's exact recompute; unused otherwise); `triattn_mod_sm100.cu` (sm_100a-only) `tri_front_sm100` gains a runtime `V1`
  parameter (0 everywhere but the d_pair 64 inference front with 16-channel heads); the HB builds take `TA_POLY` (`POLY_INFER`); new
  extension names `triattn_b200_core_r*_hb_[d16_]p*_v2_tt8`, `triattn_b200_mod_v19_tt7`. No shared source changed.
- Training tails round 9: sm100-only sources only -- `triattn_mod_sm100.cu` (head_bwd / wgrad bodies, `tri_hw_bwd`, experiment macros
  `TB_TL` / `HB_NR` / `HB_NX` / `HB_GIC` / `HB_BOX3` / `WG_NR` / `WG_DST`, all defaults the previous code), extension `triattn_b200_mod_v19_tt9`.

- Part 11: `triattn_sm100.cu` (sm_100a-only) gains `TA_D16C` (default off) and the bias tile as one 3-D box (all builds, same bits);
  `b200_triattn.py` / `integrations` (B200-only): `pack_wide_weights` returns a ninth element (the compact inference pack, or None),
  `wide_forward` reads `packed[:8]`; `triangle_attention/cuda/__init__.py` (the prebuild recipe, sm100 branch): one more `_core` build
  (`d16c=True`), additive. No shared source changed.
- Part 12: `triattn_sm100.cu` (sm_100a-only) gains `TA_ONES` / `TA_ONESX` / `TA_PH` / `TA_DBGW` (all default off: the default builds write
  the same bits, 70 tensors checked), `d16_exact` templated on the channel count; `integrations/triattn_b200.py` (B200-only) takes the HB
  build at L128 for widths the L128 kernel does not serve. No shared source changed. Rejected host wiring of TA_ONES:
  `rejected/triattn_part12_ones_host_wiring.diff`.
- Part 14: `triattn_dh64_sm100.cu` (sm_100a-only) gains `TA_HB` / `TA_CK` / `TA_POLY` / `TA_O1` (default off: the training build's code,
  four more unused kernel pointer parameters; extension `triattn_b200_dh64_v7_tt7`) and the inference build `triattn_b200_dh64_hb_p2_v1`
  (prebuild list, `triangle_attention/cuda/__init__.py`, additive); `b200_triattn.py` / `integrations/triattn_b200.py` (B200-only): the
  l2e pack and `hb` for 64-channel heads (`HB64_INFER`). No shared source changed.

### S90 round, training (2026-11-18, branch `b200s90-triattn-train`; measured, no source change, `cuda:b200_sm100` stays 56)

Goal of the round: SOL% (t_sol / t_measured, `docs/b200/sol_triattn.md`) toward 90 % for the TriangleAttention training cases (head widths 16 / 32,
bias-only row 188). Records and scripts: `docs/b200/triattn_s90_train/` (`solrun.py` = the SOL study's `sol.py` on fresh measurer records,
`sol_cases_base.tsv` all 84 cases, `timelines_base.txt` kernel timelines of one graph replay, `abl.py` / `ablation_bwd_kv.txt`, `pmb.py` /
`pmb_schedule_C.txt`, `lnmb.py` / `lnmb_bo_glue.txt`, `m.sh` + `ta512.yaml` the measurer runs). Base b200-2.5.x 35fa126c1 (engine 2.5.14 merged),
`cuda:b200_sm100` 56, one exclusive measurer run per length (graph, L2 flushed, GPU 3).

**SOL table (84 training cases, dtype `master` only; before = after, no source change).**

| group | cases | SOL% min / median / max |
|---|---|---|
| all | 84 | 3.4 / 18.8 / 25.7 |
| row 72 (D64 H4 dh16) | 12 | 5.3 / 17.0 / 23.6 |
| row 241 (D64 H2 dh32) | 12 | 3.4 / 13.6 / 20.5 |
| row 242 (D64 H4 dh32) | 12 | 5.9 / 17.0 / 23.2 |
| row 243 (D128 H4 dh32) | 12 | 7.5 / 17.4 / 23.2 |
| row 244 (D256 H8 dh32) | 12 | 11.3 / 21.5 / 22.1 |
| row 245 (D384 H12 dh32) | 12 | 16.6 / 24.2 / 25.7 |
| row 188 bias-only (D128 H4 dh32) | 12 | 4.2 / 18.6 / 24.9 |
| head width 32 self-attention (241-245) | 60 | 3.4 / 18.8 / 25.7 |

Cases >= 90 %: 0, >= 70 %: 0, >= 50 %: 0. Lowest 15 (row start L: us, SOL%): 241 E/S L128 88.7 / 86.7 (3.4 / 3.5), 188 E/S L128 121.5 / 119.4 (4.2 / 4.3),
72 E/S L128 101.0 / 100.0 (5.3 / 5.4), 242 E/S L128 101.0 / 99.1 (5.9 / 6.1), 243 E/S L128 121.7 / 121.5 (7.5 / 7.6), 72 E L256 289.5 (10.1),
241 E L256 209.5 (10.2), 72 S L256 284.2 (10.3), 188 E L256 242.3 (10.5), 241 S L256 202.4 (10.5). The earlier study's numbers moved up a little
(median 18.2 -> 18.8, row 72 L128 171 -> 100 us) by the tails rounds that landed since; the lowest groups are the same.

**1. The three legal schedules, in measured time (the SOL study picks the minimum of the floors; here is what each costs on this code).**

* A (save q, k, v, g, o + LSE; the backward recomputes S and P): the shipped schedule, 110.8 us (D128 L128), 1213 us (L512), `timelines_base.txt`.
* B (keep only the pair, run front + core again in the backward): every saved tensor is made again in HBM, so B saves no byte and re-runs the
  front and the core. From the timelines: L128 D128 +10.7 (front) + 17.5 (core; 8.5 overlapped by PDL, so +10 to +17) +3.4 (pack) = +24..31 us on 110.8
  (+22..28 %); L512 +63.1 + 212.1 + 3.3 = +278 us on 1213 (+23 %); D384 L512 +296 + 608 + 26 = +930 us on 4079 (+23 %). B loses at every length; the study chose B
  at L128 only because its floor model has no cost for the re-run's latency chain.
* C (save P in bf16: 2 L^3 H bytes written by the forward core, read by each of the two backward kernels). `pmb.py`, L2 flushed, 4 heads, written once
  and read once (fill / amax, 3.9 TB/s both ways at L >= 384; the HBM peak read of the study is 5.7 TB/s, write-only 3.55):

  | L | P bytes | write | read | C traffic (1 write + 2 reads), measured | A core fwd + bwd (measured) |
  |---|---|---|---|---|---|
  | 256 | 134 MB | 40 us | 55 us | 150 us | not timed |
  | 384 | 453 MB | 122 us | 134 us | 389 us | not timed |
  | 512 | 1.07 GB | 277 us | 286 us | 850 us (653 at the 5.7 TB/s read peak) | 212 + 585 = 797 |
  | 768 | 3.62 GB | 925 us | 918 us | 2760 us (2196 at peak) | ~660 (round 3 doc) + 1791 = ~2450 |

  The traffic alone is at or above the core time of A from L512 up, and the backward kernels keep their MMA / hand-off chain without the exp (below), so
  C cannot beat A by more than ~9 % of the module at L512 and not at all at L768, while it needs L^3 H x 2 bytes alive between forward and backward
  (1.07 GB at L512 H4, 3.2 GB at H12, 10.9 GB at L768 H12 per call). Not built. A stays: it is the cheapest of the three in measured time.

**2. What bounds the core backward (the largest block: 585 of 1213 us at L512 D128, 1684 of 4079 at D384, 48 % / 41 %).**
`abl.py` builds `triattn_bwd_kv_sm100` with its compile-time ablations (4 heads, graph, us, bwd_kv / bwd_q):

| variant | L512 kv | L768 kv | L512 q | L768 q |
|---|---|---|---|---|
| base | 299.8 | 914.6 | 285.6 | 876.2 |
| `TB_POLY=1` (one pair of 4 exps on the FMA pipe) | 317.1 | 978.9 | 311.6 | 956.5 |
| `TA_NOLD` (no S^T / dP^T TMEM loads) | 298.5 | 921.6 | | |
| `TA_NOLDS` (no bias / LSE / delta shared loads) | 298.3 | 909.8 | | |
| `TA_NOST` (no P^T / dS^T TMEM stores) | 298.1 | 914.2 | | |
| `TA_NOMATH` (no ex2 / softmax arithmetic) | 252.6 | 750.1 | | |
| `TA_NOMATH` + `NOLD` + `NOST` (only the MMAs and hand-offs left) | **242.3** | **714.0** | | |
| `TA_BIAS1` / `TA_QD1` (no bias / Q, dO traffic after the first) | 293.8 / 277.8 | 902.2 / 887.0 | | |

With the whole softmax, its TMEM traffic and its stores removed the key-side kernel still takes 81 % (L512) / 78 % (L768) of its time: the kernel is
bound by its tcgen05 chain (S^T and dP^T as shared-fed MMAs ~65 clk each, four grad MMAs, ~3 barrier round trips of 100-300 clk a stage, two groups
alternating) at ~1040 clk a 64 x 128 stage (443 stages an SM at L512) -- about twice the ~500 clk the MMAs themselves need -- and not by MUFU (115 us a
kernel at the 4.6 T ex2/s peak), nor by FP32 / TMEM / memory traffic. Moving exponentials to the FMA pipe makes it slower (+6 % kv, +9 % q). Raising the
rate needs a different MMA structure (K / V as TMEM operands, i.e. the ts form of S^T / dP^T: the 512 TMEM columns are all used by two groups of
S^T | dP^T 128, P^T | dS^T 64 and dV | dK 128; or a third softmax group): the multi-day redesign of the earlier parts 4 / 5 and training tails rounds 6-11, not started here.
The query-side kernel has no ablation macros; it runs within 5 % of the key-side one at every L and has the same chain.

**3. Per-kernel time against a byte floor (6 TB/s) -- why the glue is not the lever either.** D128 H4 L512, 1213 us: front 63 (floor 56 = 335 MB),
core fwd 212, tail 80 (45-56), gate_bwd 82 (56), bwd_kv / bwd_q 299 / 286, db_slabs 16, head_bwd + wgrad 174 (67 + the wgrad MMAs), slot sum + finish 9: every
non-core kernel is 1.1-2.6x its byte floor, none is >= 2x and fat. D384 H12 L512, 4079: wfront 296 (168), core fwd 608, wtail 281 (134), bwd_pre 85 (67), wgbwd 258,
core bwd 1684, whbwd 472, two split-K cuBLAS GEMMs 246 + 68. L128 (one tile a CTA, 111 us D128): 12 kernels of 2.8-17.5 us, floors 1-5 us (round 11's table).

**4. Row 188 (bias-only, cuBLAS glue), D128 L512 starting 569 us (SOL 21 %).** The step is 30 launches: ln_fwd 35.4 (floor at the measured clone
rate 29), c-plane GEMM 35.1, logits + softmax + copy 8.6, P.v GEMM 29.0, gate_u 30.8, out GEMM 17.5, scale + residual 35.1, backward scale 28.5, dWo split-K 32 + 3,
gate_bwd 67.4, dV GEMM 26.8, dP GEMM 41.5 (K = 16384, 64 CTAs: 2x its bytes / flops), dW split-K 37 + 3, dc^T w GEMM 36.6, ln_bwd 81.1, colsum 13. `lnmb.py`
(event timing, L2 flushed, standalone): ln_fwd 43.2 vs clone 29.0 us, ln_bwd 87.4 vs 72.6 at L512; L768 85.1 vs 54.5, 171.2 vs 136. Tuning every glue kernel to its
byte floor would give ~-60 us of 569 (10 %, SOL 21 -> 23 %), spread over eight kernels; not done. The big pieces (gate_u 30.8 and scale + residual 35.1 are AT their floors; the GEMMs are cuBLAS)
need a fused sm100 tail / head kernel pair (LN + projections, P.v + gate + Wo + dropout + residual), the multi-week build.

**Which resource blocks 90 % (every case below 90 %, i.e. all 84).** Self-attention training L >= 384: the tcgen05 / hand-off chain of the two core backward kernels (item 2: measured 78-81 % of
the kernel with no softmax), then the forward core (MUFU-bound loop at ~56 % of the ex2 peak, parts 10 / 13 / 14) and kernels at 1.1-2.6x their byte floors (item 3); the study's floors
also assume the dense 1704 TF peak for K = 32 MMAs and a ~4.6 T ex2/s MUFU with no overlap loss, which this code cannot reach (a K = 32 ss MMA costs ~65 clk, measured earlier, = ~35 % of the peak MMA rate).
L128 / L256: dependent-kernel chains of 10-14 tile-per-CTA kernels (2.8-17.5 us each against 1-5 us floors; the measured launch + fill of one dependent kernel here is 2-3 us). Row 188: unfused
cuBLAS + glue, 1.2-2x byte floors per kernel (item 4). Datasheet-peak SOL (study column) is lower still (median 15.7).

**Result.** No source change, no `backend_versions` bump, no new cubin (the ablation builds are measurement-only). 1.5x vs H100 misses and ladder violations are unchanged by construction
(last measured: 3 and 0 on the 198 training cases of the six classes, round 11). Nothing left running on the B200.

#### S90 training, part 2: the key-side backward's clock64 timeline and four restructuring experiments (measured, nothing adopted)

Tools: `docs/b200/triattn_s90_train/tlkv.py` (clock64 timeline of CTA 1 of `triattn_bwd_kv_sm100`, built with `-DTA_TL=1`, extra events added in the
experiment diff), `cmpb.py` (core backward, base build vs build with -D defines: bitwise equality of dq / dk / dv / dbias, determinism, graph time), `pipe2.py`
(the part 10 softmax-mix microbenchmark at 4 / 8 / 12 / 16 / 24 warps). Experiment source: `docs/b200/rejected/triattn_s90_train_kv_experiments.diff`
(TB_EARLY, TB_NOEPI, TB_SLEEP and the timeline events; the kernel source of this branch is unchanged). L512, 4 heads, GPU 3 exclusive.

**Timeline of one CTA (steady stages, clk).** Stage period 1144-1191 (64 q x 128 k = 8192 scores; 443 stages an SM); two softmax groups alternate, so a group's cycle is 2290:
wait for S^T 301-355, TMEM loads + bias / LSE / delta shared loads + wait for the previous gradient GEMMs + exp / dS arithmetic + TMEM stores 1514-1544, arrive 50, and
the per-task dK / dV drain amortised 390 (1560 a task of 16 stages: tmem wait 267-286, load 90, pack + st.shared + syncs ~420, the two TMA stores 750: ~375 clk a store issue,
the lane that issues them delays its group's next P^T arrive). S^T issued -> seen 440-600, P^T arrive -> grad MMA warp sees it 70-170. The S^T issuer is blocked ~750 clk a stage on the group's
`se` barrier (the group's S^T | dP^T buffer free); that barrier is released at ~1100 clk after the group started, but the slowest of the 4 warps of a group arrives 500-800 later.
With all softmax work removed (NOMATH + NOLD + NOST) the stage period is 939-959 and the group waits 980-1040 for S^T: the chain floor of the key kernel is ~240 us at L512, 80 % of the base.

**Instruction-mix microbenchmark (`pipe2.py`, clk per warp key pair a sub-partition).** The real softmax mix: 20.9 at 4 warps (one a sub-partition), 17.9 at 8 (two: this kernel),
16.4 at >= 12; MUFU alone 19.1 / 16.2 / 16.1. So with two warps a sub-partition the loop alone is within 10 % of the MUFU rate; the kernel's MUFU duty is 44 % (1024 of 2290 clk
a stage pair) because each group spends 1100 of its 2290 clk outside the exp loop and the two groups only overlap by ~400 clk.

**Experiments (core backward alone, `cmpb.py`, graph, base -> variant; all four bitwise equal to the base output, deterministic):**

| variant | what | L256 | L512 | L768 |
|---|---|---|---|---|
| `TB_EARLY=1` | both chunks of S^T and dP^T loaded into 128 registers first, the TMEM buffer released at once (instead of mid-math) | 100.6 -> 102.4 | 604.5 -> 615.8 | 1820.6 -> 1886.7 |
| `TB_SLEEP=1` / 2 / 4 / 7 | TMA / S^T MMA / grad MMA warp `wait`s with a suspend hint (no issue slots taken from the softmax warps) | 0.99-1.00 | 1.000 | 1.006 / 1.022 / 1.006 / 1.017 |
| `TB_NOEPI=1` (ablation) | no dK / dV drain at all | | kv 298.5 -> 286.9 (-3.9 %) | kv 914.6 -> 897.8 (-1.8 %) |
| `TB_POLY=1` (earlier table) | one pair of four exps on the FMA pipe | | kv 299.8 -> 317.1 | 914.6 -> 978.9 |

So: the early S^T release does not help (the register cost lengthens the math), spinning waits do not cost the softmax warps issue slots, and even a free drain gains 4 %: an
epilogue warp group (which would need setmaxnreg and a 512-thread layout) is not worth building. What is left is structural and measured above: softmax free = 240 us (-19 %),
so the largest possible gain of any softmax-side restructuring (a third group / 32-query stages with 4 groups, TMEM exactly full, 16-query register chunks at ~104 regs) is ~-19 % of the
key kernel and about the same for the query kernel (~-8 % of the module at L512), at the price of a full rewrite of both kernels; the MMA side (K / V as TMEM A operands) needs
TMEM that these two kernels do not have. Not started: the measured ceiling per kernel is below the cost and risk of the rewrite on top of 8 training-tails rounds.

## S90 round, inference (2026-11-18, branch `b200s90-triattn-inf`, cuda:b200_sm100 58)

SOL = `docs/b200/sol_triattn.md` formulas, re-scored on fresh measurer records (TriangleAttention, cuda, eval, graph, L2 flushed, exclusive, GPU 2; every length group its own run). Full per-case table: `triattn_s90_inf_sol.md`. Three floors: all exps on MUFU (the study's); the study's formula split (8 FP32 instrs / emulated exp); the measured-loop split (`t_mufu` x 0.884, from the softmax-loop microbenchmark below).
Eval, 118 cases: SOL% min / median / max = 5.6 / 41.4 / 65.7 (all-MUFU), 4.4 / 32.5 / 49.0 (formula split), 4.9 / 36.6 / 58.1 (measured-loop split); cases >= 90 / 70 / 50 %: 0 / 0 / 41. Bias-only row 188: 6.4 / 15.4 / 24.8.

**Taken: bias-only at one tile (row 188, L128).** `q = k = 0` makes the self-attention update the bias-only update, so `bidir_triattn_h100.forward_single` (B200, no grad, no dropout, L <= 128, d_pair 128 x 4) calls `fused_forward` (the one-kernel `tri_l128`) with zero Wq / Wk: eval L128 start / end 34.8 / 36.8 -> 20.5-21.3 / 22.4 us in the module A/B, measurer 36.7 -> 22.4 us (SOL 3.9 -> 6.4 %). Accuracy: measurer GREEN (err 2.8e-3 / 3.0e-3, tol 8e-3), `_s90/chk_a.py` (random non-zero weights, B1-3, L64-128, masks with dead keys, both nodes, vs fp32 gold) max error 2.4-3.6e-3 against the chain's 2.6-3.6e-3. `MINIWORLD_TRIATTN_B200_BO_L128=0` turns it off. No new cubin.

**Measured floors.** Softmax loop (`_s90/pipe_s90.py`, 16 warps, clk per warp key pair per sub-partition): MUFU only 16.07, all but MUFU 7.12, real mix 16.36; polynomial pairs of 8 (clamped, degree 3): 1/2/3/4/8 = 15.43/14.20/15.23/17.18/25.75; degree 2: 15.59/14.52/15.26/15.82/23.25. Best = 14.2 (0.884 of MUFU only): the loop is issue-bound at the optimum (poly pair ~18.6 issue clk), so the formula's 26 % cut is not reachable in isolation (issue model: 21 %), and the shipped core takes 2/16 pairs. The core at L2048 (4 heads) is 72 % of the all-MUFU floor, i.e. ~25 % outside the isolated loop (TMEM ld/st, barriers, MMA issue); parts 4/5/13/14 measured those as not removable.
Bias-only L768 starting (profiler, cold L2, us): LN 79.9, planes GEMM 77.8, logits+softmax+copy 15.5, PV bmm 77.7, tail_k 162.8 = 416 for 151 MB in / 151 MB out: the chain materialises 4 pair-sized streams (floor of the whole op 50 us memory, 102 us tensor). LN and the GEMM sit at 3.8 and 5.9 TB/s, the tail at ~3.7 TB/s of the 604 MB it moves.

**Not done (needs a new kernel, not attempted this round):** bias-only L >= 256 as front(LN + v|bias to planes) -> cuBLAS PV -> gate-from-x tail would cut the chain from 4 to ~2.5 pair streams (est. -40 %, unmeasured); a fully fused PV is L2-bandwidth bound (~100-150 us at L768, measured nothing).

### S90 inference, part 2 (cuda:b200_sm100 60): the bias-only chain at L >= 256 (row 188)

Chain before (L768 starting, profiler, cold L2, us): LN 79.9, planes GEMM 77.8, logits + softmax + copy 15.5, PV bmm 77.7, `tail_k` 162.8 (4 pair-sized streams, 416 us). Now, for B = 1, L % 128 == 0, d_pair 128 x 4 heads, inference: `tri_front(vpl)` (the existing LN + tcgen05 front with only the v GEMM; v drained as a channel-major plane [128][L^2] by a transposed staging tile and a TMA store; the masked head-major bias from the LN warps as before) -> `torch.softmax` of the bias (bf16, one launch) -> the same cuBLAS `bmm` on the plane (one launch for BOTH nodes: the ending node is the starting node of the transposed frame, `trans` in the front and the tail) -> `tri_tailg_pl` (`tri_tailg` with o loaded as plane boxes [64 channels][128 tokens]; gate from x, out-projection, residual). Sources: `triattn_mod_sm100.cu` (NOUT = 1 front, `tri_tailg_pl`), `tailg_sm100.cuh` (`OPL`), `b200_triattn.bias_only_planes`, `bidir_triattn_h100._b200_one_tile` (route). Extension names `triattn_b200_mod_v21_tt10`, `triattn_b200_wide_v45_tt7` (the wide module includes the tail header; no wide code change). `MINIWORLD_TRIATTN_B200_BO_PLANES=0` turns it off. Not added to the prebuild plan yet (build of the two extensions happens on first use; the prebuild recipe lists `_mod()` already, which now contains the new kernels).
Module A/B (`_s90/chk_b.py`, graph, L2 flushed, us, new / chain): L256 49.2 / 67.6 start, 51.2 / 77.7 end; L384 81.9 / 122.9, 88.1 / 131.1; L512 131.1 / 192.5, 137.2 / 206.8; L640 203.5 / 299.0, 213.0 / 317.4; L768 272.5 / 409.6, 290.5 / 421.9 (-27 .. -34 %). Accuracy: error vs fp32 gold (random non-zero weights, masks with dead keys, both nodes) max 2.4-3.6e-3, the chain's 2.6-3.6e-3; measurer accuracy L256 / 384 / 512 (TriangleAttention, all rows, cuda): max err / tol 0.43 / 0.49 / 0.44 (GREEN).
Kernels at L768 now (profiler): front 89 (50 us of traffic floor; timing-only flags `abl_front.py`: no LN 59, no drain 80, no MMA 82, no stores 86: the LN warps' CUDA-core bias projection is ~29 us), softmax 8, PV bmm 78 (the cuBLAS plane form; a token-major PV via strided bmm measured 248 us, `pvbench.py`), tail 108-117 (453 MB of traffic: 75 us floor).
Not done: the bias projection on the tensor core (N = 16 MMA from the staged fp32 rows, est. -15 us of the front); an own PV kernel.

### S90 inference: measured facts for the core (no change)
- SM clock under the L2048 core loop (nvidia-smi samples every 0.7 s during `clkrun.py`): 1867-1890 MHz at ~990 W (power-cap throttle bit set) against 1965 MHz: the clock explains ~4-5 % of the gap to the SOL floor, not the 28 %.
- The shipped split (2 of 16 pairs above L256, 4 of 16 below) is the measured optimum of the real core (parts 10/11: 3 / 4 / 6 of 16 are 3-8 % slower at L768 / L2048), although the isolated loop's best is 2 of 8 (14.2 vs 16.07 clk): the core cannot use the extra issue slots (the loop shares them with the TMEM ld/st, barrier and MMA-issue instructions).
Follow-up in the same version: the v-plane front's bias projection moved from the LN warps' CUDA cores to one N = 16 tcgen05 GEMM (bf16 bias rows staged in shared memory, accumulator at TMEM column 3 HD, drained by the drain warps; extension `triattn_b200_mod_v22_tt10`): module A/B L256 49.2 -> 47.1 us (start), L768 272.5 -> 262.1 (start) / 290.5 -> 276.5 (end). Final measurer grid (`_s90/out/final3`): eval 118 cases, SOL% all-MUFU min / median / max 5.6 / 41.4 / 66.4, row 188 6.4 / 24.0 / 39.1 (was 3.9 / 15.4 / 24.8); over 0.667 of the H100 token 20 -> 18 (none slower than the H100; ladder 0); measurer accuracy L256 / 512 max err / tol 0.43 / 0.44.

### S90 inference, part 3: the core on the current build (measured, no source change; cuda:b200_sm100 stays 60)

Records `_s90/ja.sh` / `ja2.sh` (jobs ta_a1 / ta_a2 on the B200), tools `_s90/ftl6.py` (the `TA_FTL` clock64 buckets, present in the current source), `_s90/cb13.py` (core variants, 20 back-to-back replays = sustained clocks), `abl_tail.py`.
**Timeline of one CTA, HB core, 3 rows a task, 2 of 16 polynomial pairs, 4 heads (clk per 128 x 128 sub-step of the SM; the three softmax groups each see every third sub-step):**

| L | sub-step | QK^T issue warp: issue / w_kvf / w_qf | softmax group: work / wait S / wait bias / other | TMA warp: w_bias_empty / w_kv_empty / issue |
|---|---|---|---|---|
| 512 | 1477 | 1213 / 141 / 121 | 838 / 332 / 115 / 187 | 421 / 601 / 431 |
| 1024 | 1320 | 1171 / 75 / 75 | 828 / 286 / 70 / 125 | 491 / 450 / 372 |
| 2048 | 1271 | 1143 / 66 / 62 | 827 / 280 / 56 / 96 | 532 / 388 / 347 |

At L2048 a sub-step takes 1271 clk against the MUFU floor 1024 (896 with 2 of 16 pairs on the FMA pipe): 81 % of the all-MUFU floor in the kernel, 72 % at module level. A group's cycle is 3 x 1271 = 3813 clk: 2481 clk of softmax work (the three groups share one MUFU, so it is stretched to ~the MUFU time of all three: 70 % MUFU busy), 840 waiting for its S, 170 for the bias tile, ~300 other. The QK^T warp's "issue" bucket is not issue: it contains the wait for PV(y - 3) to retire (the S buffer's release), so the ring S -> softmax -> PV -> QK^T -> S (three buffers, one per group, 480 of 512 TMEM columns) is the structure that leaves the MUFU idle ~30 % of the time. Its levers were measured before (parts 4, 5, 13: a 4th group, separate P buffers, split-P arrive, 2-CTA PV, ones in the PV MMA: all level or slower) and need TMEM columns the kernel does not have.
**Polynomial pairs on the current build, sustained (core alone, 4 heads, us, two rounds):** L512 p0 190.8-191.1, p2 184.6, p3 184.8-185.0, p4 184.1, p6 189.3-189.8; L1024 p0 1315-1341, p2 1325-1352, p3 1355-1359, p4 1376, p6 1428-1443; L2048 p0 10217-10261, p2 10262-10284, p3 10268-10283, p4 10506-10513, p6 10951. So the shipped 2 of 16 is within 0.3 % of the best at L512 (4: -0.3 %) and 0.4-0.8 % slower than none at L1024 / L2048 (inside the measurer's noise at those sizes); 6 of 16 is 3-7 % slower everywhere. Not changed. SM clock during the L2048 loop 1867-1890 MHz (see above): the power cap costs 4-5 %.
**Row 188 pieces at L768:** PV bmm 77.9 us for 116 GF = 1.49 PF/s, 87 % of the 1704 TF cuBLAS peak: an own PV kernel can win at most ~10 us (and the token-major strided form is 248 us); the plane tail `tri_tailg_pl` 110.6 us for 453 MB = 4.1 TB/s (68 % of 6.0). The timing-only tail ablations did not run (the variant builds raised `invalid argument` at launch; not investigated).

### S90 inference, part 4: the S ring and the plane tail -- measured, nothing adopted (cuda:b200_sm100 stays 60)

TMEM of the head-dim-32 core (`triattn_sm100.cu`): S / P 3 x 128 columns (P bf16 over the fp32 S, PV `.ts`), O 3 rows x 32 = 96, columns 480-511 hold nothing in the inference build (`COL_L`, used only by `TA_ONES`): 32 free columns, not the 128 a fourth S buffer needs; with 2 rows a task 64 free. Levers against the 840 clk S wait (22 % of a group's 3813 clk cycle at L2048):
1. **Finer S (64-key half buffers, six of them in the same 384 columns)**: built and measured in part 2 on the then core (code not kept): the groups' S wait fell from ~18 % to ~6 %, but the per-half overhead (two votes / fences / arrives a sub-step, N = 64 QK^T re-reading Q) made L768 628-633 us against 583 (L2048 11.04-11.10 ms against 10.66). Not re-built on the HB core.
2. **P aliasing S**: already how the core works (P is stored over S, the PV MMA reads it from TMEM); nothing left to free.
3. **Reclaiming TMEM**: 32 columns free (see above): a 4th buffer does not fit; 96-key buffers do not divide the 128-key tiles; an fp16 S (64 columns a buffer) would need fp16 q / k and fp16 accumulation (part 12 built the fp16 pieces and kept them off).
4. **QK^T(y + 3) not waiting for PV(y) to retire** (`TA_PVI`, new, default off, same bits): the QK^T warp waits for an mbarrier the PV warp arrives on right after issuing the PV MMAs (tcgen05 fence before / after, tensor pipe in order), instead of the K / V stage's retire barrier. Core, 4 heads, 3 rows, graph x 20, us (base / pvi): L512 184.4 / 184.7 and 184.4 / 184.9, L1024 1306.9 / 1315.5 and 1357.5 / 1340.6, L2048 10244 / 10229 and 10271 / 10257; L256 (2 heads) 20.55 / 20.58: level, same bits. So the S wait is not the PV GEMM's execution time (8 x ~47 clk) but the latency of the signalling chain (commit -> mbarrier -> warp wake -> issue -> commit) and the groups' phase alignment; rejected.

**Tail ablation of `tri_tailg_pl` at L768** (timing-only flags, one process each; the earlier `invalid argument` was several flagged extensions loaded in one process): base 110.6 us; no output stores 100.4; no residual loads 106.5; no drain (OUT released unread) 86.0; no LN arithmetic 96.4; no gate arithmetic 98.3. The drain (4 warps, 32 rows each, 4 chunks of TMEM load -> residual add -> smem -> TMA store) is the longest pole (-24.6 us); a second drain warp group per K chunk is the candidate (est. -10 us, and it serves every d_pair 128 inference tail); not built.
**Not built: (B)**, the cluster weight-split fused kernel for d_pair 256 / 384 at L128 (a 4-CTA cluster with 2 heads a CTA, DSMEM reduction of the out-projection partials, a bias exchange that needs a separate bias launch because 512 CTAs cannot be grid-synchronised on 148 SMs): a new ~1000-line tcgen05 / DSMEM kernel, not attempted in this session.

### S90 inference, part 5: second drain group for the plane tail, single-warp issue, the d256 L128 chain -- measured, nothing adopted (cuda:b200_sm100 stays 60)

- **`TG_DW2`** (new in `tailg_sm100.cuh`, default off, default build unchanged: 110.6 us / 24.5 us at L768 / L256 as before): a second drain warp group (warps 15-18, 608 threads, so 104 registers a thread) drains K chunk 1 while warps 10-13 drain chunk 0, the slot / OUT barriers count 8. `tri_tailg_pl`, graph, L2 flushed: L768 110.6 -> 120.8 us, L256 24.5 -> 26.6 us. Slower (the register cap of the larger block, plus twice the arrivals); rejected. The drain pole (86.0 us without it) needs a different split (the TMEM load / residual add / store pipelined inside the existing four warps), not more warps.
- **One warp issuing both QK^T and PV (`TA_SPLIT=0`)**: not available with three softmax groups (static assertion `R = 3 needs TA_SPLIT`); with 2 rows a task (L256, 2 heads) 20.57 us against 20.48 (level), and part 2 measured 638 against 625 us at L768 for the three-group-less core. Rejected. Staggering the groups' start and spin-polling variants were not built: the ring already starts the three groups in turn, and `wait_sleep` (suspend hint) was level in part 2.
- **(B), d_pair 256 / 384 at L128**: not built. Measured chain (profiler, eval L128, d_pair 256 x 8 heads): `tri_wfront` 18.0 us (ending 18.4), core 13.5, `tri_wtail` 16.4 (18.1): 47.9 us of kernels in a 49.2 us graph, i.e. the PDL-chained launches already overlap to ~1.3 us; a fused cluster kernel can only win by making the three phases themselves cheaper. Cost model of the cluster design (4 CTAs a row, 2 heads each, 160 KB of weight slices + 64 KB x tile per CTA already exceeds the 227 KB shared memory once the q / k / v tiles are added, so the weights would stream per K chunk; 128 rows x 4 CTAs = 512 CTA tasks on 148 SMs = 3.5 waves of a ~10 us row phase, the bias needs its own launch because the grid cannot synchronise): ~35 us against 49, a new ~1000-line tcgen05 / DSMEM kernel, unmeasured and not started.

### S90 inference, part 6: the tail's drain pole -- a pipelined drain and direct stores, both slower (cuda:b200_sm100 stays 60)

`tri_tailg_pl` (o plane) and `tri_tailg` (o token-major, the d_pair 128 inference tail of the fused path), graph, L2 flushed, us; knobs in `tailg_sm100.cuh`, all default off, the default build bitwise and in time unchanged (110.6 / 38.9 / 24.6 at L768 / 384 / 256):
- `TG_PD = 1`, a software pipeline inside the four drain warps (chunk c + 1's TMEM load issued before chunk c is processed, double v registers; the slot release of a tile deferred behind the next tile's first load): plane tail L768 110.8 -> 137.2, L384 38.9 -> 43.0, L256 24.6 -> 24.6; token-major tail L768 106.5 -> 133.1, L384 36.9 -> 41.0; same bits.
- `TG_PD = 2` (the chunk prefetch only, immediate release): L768 112.4 vs 110.6, L384 38.9 vs 38.9: level. So the TMEM-load latency is not the pole; the loss of `TG_PD = 1` is the delayed release of the x slot (~0.85 us a tile of the 3.5 us tile period): the x ring (three slots, 32 KiB each; a fourth does not fit: 233 984 B against 232 448) is the loop that bounds the kernel: slot freed -> TMA reload -> LN -> gate -> out GEMM -> drain -> store read -> slot freed.
- `TG_DS = 1` (the starting node stores y from registers, 16 B a lane to rows 256 B apart; the slot is only read, so it is released at once): plane tail L768 110.6 -> 139.3, L384 38.9 -> 45.1, L256 24.6 -> 28.7; token-major L768 106.7 -> 122.9; the L384 plane-tail checksum differed from the TMA-store path (not investigated: the variant is rejected). Uncoalesced stores cost more than the smem round trip saves.
- `TG_DW2` (part 5): 120.8.
Not done: the stage ablation of the wide front / tail at d_pair 256 / 384 (the wide kernels have no timing-only flags; adding them is a pass over two ~2 000-line sources) and any fix there.

### S90 inference, part 7: four x slots in the d_pair 128 tail (taken, cuda:b200_sm100 62); producer lanes of the wide front / tail (level, off)

`tailg_sm100.cuh` `TG_NXS4` (default on; CP = HD = 128): the x ring grows from 3 to 4 slots. Shared memory 233 984 -> 231 936 B: no 1 KiB alignment slack (the base is asserted 1 KiB-aligned, `__trap` otherwise, as `tri_wfront` already does), the LN partial sums in one [2][128] buffer (two more named barriers a tile: after the means were read, after the variances were read), barriers 512 B. Same bits (checksums equal). `tri_tailg_pl` (plane) / `tri_tailg` (token-major), graph, L2 flushed, us (3 slots -> 4 slots): L768 110.6 -> 102.2 / 107.7 -> 96.3, L384 38.9 -> 36.8 / 36.9 -> 34.8, L256 24.6 -> 24.4 / 24.4 -> 22.6. With four slots the deferred release (`TG_PD = 1`) is level (102.4 / 98.3 at L768): no further gain, `TG_PD` stays 0. Measurer (TriangleAttention eval, 118 cases, `_s90/out/after4`): over 0.667 of the H100 token 18 -> 17, none slower, ladder 0; row 188 SOL median 24.0 -> 25.2 %, all rows 41.4 -> 41.1 % (median noise); accuracy L256 / 384 / 512 max err / tol 0.43 / 0.49 / 0.44 (GREEN).
**Producer lanes** (`WF_ML` / `WT_ML` in `triattn_wide_sm100.cu`, default 1): the front's W stream (20 loads of 16 KiB a tile at d_pair 256) and the tail's stage stream issued by 2-4 lanes of the TMA warp instead of one (the B200 "one TMA load per ~600 clk a thread" finding). Module A/B (graph, L2 flushed, us, base / WF 4 / WF 2 / WT 3 / both): d_pair 256 L128 51.2 / 51.2 / 53.2 / 51.2 / 51.6, L256 149.6 / 161.8 / 153.6 / 151.6 / 161.8; d_pair 384 L128 75.8 / 79.8 / 75.8 / 75.8 / 78.1, L256 247.8 / 260.0 / 250.9 / 247.8 / 260.1: level or slower; the wide front and tail are not bound by the issue rate of their weight streams. Not adopted. The stage ablation (no-load / no-MMA / no-store) of the wide kernels was not built.

### S90 inference, part 8: the wide front / tail stage ablation, the drain as the largest stage, store boxes (cuda:b200_sm100 64)

Tools: timing-only flags in `triattn_wide_sm100.cu` (`WF_NOLOAD / NOMMA / NOLN / NODRAIN / NOSTORE`, `WT_NOLOAD / NOMMA / NOGATE / NODRAIN / NOSTORE`, `WF_TL / WT_TL` = `%globaltimer` stamps of CTAs 0 / 1 read by `tri_wtl_read`), `_s90/abl_wide.py` (one flagged extension per process; kernel durations from the profiler, L2 flushed, min of 7) and `jw.sh`. Kernel us, base -> without the stage (the other kernels of a flagged run are not meaningful):

| case | kernel | base | no load | no MMA | no LN / gate | no drain | no store |
|---|---|---|---|---|---|---|---|
| d_pair 256 L128 | front | 17.9 | 17.1 | 15.6 | 15.8 (LN) | 14.4 | 15.6 |
| d_pair 256 L128 | tail | 17.2 | 13.8 | 16.2 | 17.2 (gate) | 8.3 | 11.9 |
| d_pair 256 L256 | front | 45.8 | 42.0 | 41.2 | 43.9 | 34.8 | 38.1 |
| d_pair 256 L256 | tail | 40.9 | 34.9 | 41.2 | 43.0 | 31.3 | 34.5 |
| d_pair 384 L128 | front | 33.0 | 31.5 | 30.4 | 31.3 | 27.9 | 30.0 |
| d_pair 384 L128 | tail | 19.7 | 17.9 | 17.7 | 16.6 | 15.6 | 18.0 |
| d_pair 384 L256 | front | 97.6 | 86.7 | 86.1 | 89.0 | 79.3 | 85.4 |
| d_pair 384 L256 | tail | 61.6 | 49.3 | 56.7 | 50.0 | 57.3 | 58.8 |

Floors at d_pair 256 L128: front tensor 8.7 GF = 5.1 us at 1704 TF, tail 4.3 GF = 2.5 us, tail bytes 33 MB = 5.5 us at 6 TB/s. The weight stream (no load) and the MMA (no MMA) cost 0.8-2.3 us of the front and 1.0-3.4 of the tail at L128: neither is the largest stage; **the drain (TMEM read + staging + TMA stores) is, in all eight cases but d_pair 384 L256 tail**: 3.5 / 8.9 us at d_pair 256 L128, 11 / 9.6 at L256. The `%globaltimer` stamps of the d_pair 256 L128 tail (us after the first load issue): first stage landed 1.38, accumulator ready 5.79, drain finished 14.21 -- the drain alone takes 8.4 us for one 128 x 256 tile. Together with the producer-lane result of part 7 this rules out the weight stream, so **(B), the 4-CTA weight-split cluster kernel, does not pay** (it only removes weight-stream / MMA time, which is 1-3 us).
Fix built for the tail drain: stores of 64 columns x 32 rows (4 KiB, 128B swizzle, two TMEM slices a round, one commit; `WT_PAIR = 2`) instead of 32 x 32 (2 KiB): d_pair 256 L128 tail 17.0 -> 15.9 us (same bits); a single 16 KiB store of 128 rows x 64 columns for the four drain warps (`WT_PAIR = 3`, named barrier) 16.1: no better; a pair-round with the old 2 KiB boxes (`WT_PAIR = 1`) 18.8 (slower). The larger staging ring costs one stage of the stage ring at d_pair 256 (DR 24 -> 32 KiB), which loses at L >= 384 (measurer d_pair 256 L384 321 -> 326 us), so the 64-column drain is used from L <= 256 (template `PR`). Measurer (TriangleAttention, cuda, eval, us, before -> after): d_pair 256 L128 ending 53.1 -> 51.3, starting 51.1 -> 49.0; d_pair 384 L128 starting 73.5 -> 71.7; L256 d_pair 256 151.7 / 147.2 -> 149.3 / 145.4; L384 unchanged. Accuracy (all five self-attention classes, eval L128 / L256, cuda): 32 / 32 cases, max err / tol 0.67 / 0.68, none over tolerance.
Not done: the same large-box drain for the front (its drain is the largest front stage: 3.5 / 11 us at d_pair 256 L128 / L256, 5 / 18 us at d_pair 384; the front has eight drain warps and four output tensors, a bigger change).

**Merged-tree check (b200-2.5.x bb9d44e32, `cuda:b200_sm100` 60, this branch merged in; no training source change here, so no bump to 61).** Measurer, all 84 training cases, cuda, graph, L2 flushed:
SOL% min / median / max 3.4 / 18.8 / 26.0 (before the merge 3.4 / 18.8 / 25.7), >= 90 / 70 / 50 %: 0 / 0 / 0; row 188 training times within +-1 % of the pre-merge run at every L (e.g. L768 starting 1235.5 -> 1232.5 us,
L128 ending 121.5 -> 121.3). Accuracy (`measure.accuracy`, rows 188 / 242 / 243 training, L128 / 256 / 384 / 768, starting + ending, random weights, vs the fp64 gold, tol = max(err_pytorch, 8e-3)):
24 cuda records: 20 GREEN, 4 YELLOW (row 188: L256 ending 1.03, L384 ending 1.10, L768 starting 1.14 x tol; pytorch itself sits at 0.9-1.0 x tol there), no RED.

#### S90 training, part 3: the three-group key-side kernel (built, bitwise equal to the base, slower -- not adopted)

`triattn_bwd_kv3_sm100` (diff `docs/b200/rejected/triattn_s90_train_kv3_three_groups.diff`, `-DTB_KV3=1`, `TB_KGN` groups, `TB_EARLY3`): three softmax groups (12 warps + TMA, S^T MMA, grad MMA = 480 threads, 136 registers), stage y handled by
group y % 3 over the same two S^T | dP^T / P^T | dS^T TMEM tile pairs, 16-query register chunks, per-group dK / dV drain tiles (the group that ran a row's last stage drains it while the other two continue), ring
5 deep. Two findings on the way: (1) with several groups on one buffer the mbarrier parity waits alias (a group two phases ahead of the barrier passes `wait` at once: the first version returned
garbage dK / dV, and hung from L256 up); fixed with per-buffer monotone smem tickets (S^T loaded, P^T stored; every group waits for stage y - 2 of its buffer before `sf` and before the `ge` wait). (2) Correct and deterministic
(dq dk dv dbias bitwise equal to the base, L128 / 256 / 512 / 768, two runs equal), but the time does not improve:

| core backward alone (graph) | L256 | L512 | L768 |
|---|---|---|---|
| base | 100.7 | 603.7 | 1819.0 |
| 2 groups, the new code (TB_KGN=2, chunk 16, tickets) | 101.3 (1.006) | | |
| 3 groups | 101.9 (1.009) | 619.4 (1.026) | 1900.2 (1.045) |
| 2 groups, all S^T / dP^T chunks loaded first, buffer released early (TB_EARLY3) | | 639.4 (1.057) | 1992.4 (1.094) |
| 3 groups + early release (spills: 128 registers of S / dP) | 136.8 (1.36) | 878.1 (1.45) | 2766.4 (1.52) |
| TMA producer on 6 lanes, one load a lane (TB_ML, divergent issue) | 121.8 (1.21) | 786.8 (1.30) | 2497.4 (1.37) |
| ring depth 7 / 4 slots (TB_NQ) | | 1.001 / 1.008 | 1.005 / 1.013 |

Adding a third group (TMEM column map unchanged: S 256, P | dS 128, dV | dK 128) does not shorten the stage period, so the key kernel is not limited by the number of softmax groups: with the math removed it
still costs 242 us and with the Q / dO / bias loads also removed 624 clk a stage; the remaining ~940 clk a stage is the MMA / barrier chain and the feed of the stage's operand tiles (S^T and dP^T are shared-fed MMAs of ~65 clk,
grad MMAs, the S^T buffer round trip of MMA latency ~450 + the group's load time), which a softmax-side rewrite does not touch. Not adopted; no `backend_versions` change.

#### S90 training, part 4: L128 / L256 chains, row 188, the feed chain (cuda:b200_sm100 61)

**Adopted (61): the single-direction bias-only backward (row 188) writes the LayerNorm input gradient `dyln = dc^T W` as bf16** (`bidir_triattn_h100.DYLN_BF16`, B200 only,
`MINIWORLD_BIDIR_B200_DYLN_BF16=0` turns it off): 2 x 67 MB less traffic at L512 (the fp32 GEMM output and `ln_bwd_k`'s read; the kernel already took bf16). Graph step, row 188 (us, off -> on, two interleaved rounds):
starting L256 199.9 -> 193.7, L512 568.9 -> 554.5, L768 1201.6 -> 1164.9; ending L256 226.8 -> 221.1, L512 643.2 -> 628.3, L768 1315.2 -> 1280.0 (-2.4 .. -3.0 %). Measurer, all 84 cases (`sol_cases_final.tsv`): row 188 SOL% min / median / max
4.2 / 18.6 / 24.9 -> 4.2 / 19.1 / 25.7, every other row unchanged (all: 3.4 / 18.8 / 25.7). Accuracy (`measure.accuracy`, rows 188 / 242 / 243 training, L128 / 256 / 384 / 768, starting + ending, random weights, vs the fp64 gold,
tol = max(err_pytorch, 8e-3)): 24 cuda records, 20 GREEN, 4 YELLOW (row 188 at 1.0-1.14 x tol, as before), no RED; the bf16 GEMM output is deterministic. (`BidirectionalTriangleAttention` keeps its fp32 dyln: not measured.)

**Measured, not adopted:**
- the dropout draw as a graph branch beside the parameter pack on the wide / fused-L128 paths (`MINIWORLD_TRIATTN_B200_SIDE=1` extended beyond the fused d_pair 128 path, joined before the first reader): step graph, SIDE 0 -> 1, us: d_pair 64 x 2 L128 77.4 -> 78.3, 64 x 4 90.5 -> 90.2, 128 ending 111.7 -> 113.3,
  256 x 8 190.9 -> 191.4, row 188 L128 102.0 -> 102.0; L256 64 x 2 190.0 -> 191.5, 64 x 4 271.9 -> 274.0, 128 290.4 -> 289.5, 256 x 8 580 -> 579, row 188 200.3 -> 200.1: no gain (the 6.6 us rng / drop / pack start is not shortened by a branch).
- timeline at L128 (D64 x 4 x 32 90.3 us graph): wpack 3.3, fused fwd 16.8, bwd_pre 2.8, wgbwd 6.4, kv 11.0, q 14.0, db_slabs 11.9 (starts 9 us early under PDL), db_rows 3.4, whbwd 9.1, tallk 11.4, wfinish 4.8 + 2.3; every
  gap is already -0.4 .. 0 us (PDL), so a PDL change cannot shorten the chain; round 7 / 8 measured the cooperative core backward and graph branches (slower).
- the feed chain of the key-side kernel (part 3): the shared-memory-bandwidth hypothesis (per stage ~100 KB: TMA writes 24.5, MMA operands 24 + 16, bias tile reads 32) is not supported -- `TA_NOLDS` (no bias / LSE / delta loads at all) changes the kernel by 0.5 %; `cta_group::2` and N = 128 stages do not fit the TMEM
  map (S^T | dP^T for 128 queries is 256 columns a group: 512 for two); 128-byte operand swizzle needs K = 64 (these tiles are 64-byte rows); commits are already one a stage; K / V as TMEM operands needs 64 columns that two 64-query groups do not leave.

**Not built (scoped and measured only):** the plane-layout training head / tail of row 188. `tri_tailg_pl` computes the gate from x and has no dropout, no `u` output and no gate plane; the training backward needs y, the LN statistics, the g plane and `u`, so a training version is a new front variant (v and g planes, statistics, y) and
a new tail (g plane in, dropout, `u` out): forward share at L512 190 of 569 us, fusing gate_u (30.8) + out GEMM (17.5) + scale / residual (35.1) = 83 us into one pass has a 56 us byte floor (-27 us, 5 %).

#### S90 training, part 5: per-CTA %globaltimer timelines of the L128 training chain (D64 x 4 x 32, starting; rows 72 / 242 / 241 share it)

Instrumentation (`docs/b200/rejected/triattn_s90_train_cta_timeline_stamps_tallk_flags.diff`, `S90_ST` stamps of thread 0 of every CTA into a global array, no printf; `docs/b200/triattn_s90_train/cta_mod.py`, `cta_tl.py`): entry, prologue done (TMEM alloc + barrier init + sync, **including a PDL wait**), end.
One module training step, GPU 3 exclusive, microseconds from the kernel's first CTA entry:

| kernel | CTAs | prologue (entry -> synced) | body (median, min..max) | last CTA end | what it is made of |
|---|---|---|---|---|---|
| `tri_l128` (fused front + core + tail, training form) | 128 | 0.74 | 9.8 (9.5..10.0) | 10.7 | one tile a CTA, a serial chain; 128 of 148 SMs; no launch / prologue cost to speak of |
| `triattn_bwd_kv` | 148 | 0.54 | 4.6 (2.2..5.0) | 5.8 | 256 tasks over 148 CTAs: 108 CTAs run two tasks, 40 run one (first CTA ends at 2.7) |
| `triattn_bwd_q` | 128 | 1.15 (max 4.2: PDL wait on kv) | 4.7 (4.5..4.9) | 9.1 | 128 tasks, one wave; its CTAs entered while kv ran |
| `triattn_db_slabs` | 160 | (entries spread over 9.3: PDL wait on q) | | 12.4 | ~1.4 us of work after q's end |
| `tri_whbwd` | 128 | 1.44 (the wait) | 2.6 (2.4..2.8) | 4.3 | one tile a CTA |
| `tri_wgbwd` | 128 | 0.48 | 0.83 | 1.6 | |
| `tallk` (dW tall-K GEMM) | 16 (one cluster of 16) | 0.70 | 9.5 (9.50..9.54) | 10.3 | K = 16384 split over 16 ranks, 16 K steps of ~0.6 us = ~1100 clk each |

Reading: the profiler's per-kernel durations (wgbwd 6.4, whbwd 9.1, db_slabs 11.9, kv 11.0, q 14.0 us) are mostly the dependency wait of a kernel launched early by PDL; the work of the seven kernels above is ~44 us of the 90 us step.
The rest (~40 us over ~13 kernel boundaries) is the cost of a boundary -- grid completion, the dependent grid's release, cold first accesses -- about 3 us each. The kernels with real bodies are the fused forward (9.8, one tile a CTA, a serial chain),
tallk (9.5 with 16 CTAs on 148 SMs) and the core backward (4.6 + 4.7).
Ablations on tallk: the producer's four loads of a K step issued by four lanes at once (`TK_ML`): body 9.6 -> 9.9 us; ring depth 7 instead of 4 stages: 9.6 -> 9.7; so the kernel is not TMA-issue- or ring-bound (the MMA-less ablation hung and was dropped). Split into more K parts was
measured in training tails round 6 (8 x 2 parts: 11.4 -> 10.1 us, +1.2 us in `tri_wfinish`). Neither was adopted; the stamps, flags and these two builds are reverted (diff archived).
The inference agent's wide-tail drain pattern (serial `tcgen05.ld` slices with a wait each) is in `tri_wtail` which training shares -- it arrives with the merge (WT_PAIR = 2, `cuda:b200_sm100` 64); the training backward kernels above have short bodies (whbwd 2.6, wgbwd 0.8), so it is not their pattern.

**Merged tree (b200-2.5.x 6b66f5673, `cuda:b200_sm100` 64; this branch adds no kernel change on top).** Determinism (`detm.py`, 14 configurations x 5 runs, masked / unmasked, d_pair 64 / 128 / 256, ending node, B2, ragged L200, L384): all bitwise deterministic.
Accuracy (rows 72 / 188 / 241-245, L128 / 256 / 384 / 768, starting + ending): 47 GREEN, 7 YELLOW, 2 skipped (the fp64 reference of row 245 L768 ran out of memory), no RED. All 84 training cases (`measure.speed`, cuda, graph, L2 flushed): SOL% min / median / max 3.4 / 18.8 / 26.5, none >= 50 %
(row 245 max 25.7 -> 26.5, row 188 median 18.6 -> 19.1, the others within 0.2).

### S90 inference, part 9: the d_pair 256 tail drain re-timelined, eight drain warps (cuda:b200_sm100 66)

`%globaltimer` stamps per 32-column slice of warp 10 of CTA 0 (`WT_TL`, `_s90/abl_wide.py`, d_pair 256 L128 ending node, 32 x 32 boxes; us after the first load issue). Slice k: TMEM load issued / done / after pack + ring wait / after STS + fence / store issued:
k0 5.57 / 5.60 / 5.86 / 5.92 / 6.14; k1 6.14 / 6.14 / 6.40 / 6.46 / 6.69; k2 6.69 / 6.72 / 6.98 / 7.04 / 7.26; k3 7.26 / 7.30 / 7.55 / 7.62 / 7.84; k4 7.84 / 7.87 / 8.13 / 8.19 / 8.42; k5 8.42 / 8.45 / 8.70 / 8.80 / 8.99; k6 8.99 / 9.02 / 10.53 / 10.85 / 11.39; k7 11.39 / 11.49 / 12.64 / 12.86 / 13.47.
So the drain is a serial per-warp chain of ~0.57 us a slice: TMEM load 0.03, pack of 32 values + syncwarp 0.26, STS + fence 0.06, syncwarp + TMA store issue 0.22; at k6 / k7 the ring wait (`bulk_wait_read`, the store three slices back to have READ its staging) adds 1.5 / 1.2 us (the ending node's 64-byte row pieces drain at ~7 KB / us an SM). Inference has no residual loads in the drain (the residual rides the MMA: `fold`), and the TMEM load is not the chain; so the remedies that attack a chain of loads (a) / (b) do not apply, and the larger store boxes only shorten the ring wait (part 8: -1.1 us).
**Built: eight drain warps (`WT_DW8`, template `PR = 4`)** for the d_pair 256 tail at L <= 256 without dropout: warps 10-13 take the even slices, warps 14-17 the odd ones (two warps a TMEM quadrant, 576 threads, two 2 KiB staging buffers a warp, `ae` / slot counts 8). Tail kernel (profiler, L2 flushed, same bits): d_pair 256 L128 16.2 -> 12.5 us, L256 41.8 -> 40.7; d_pair 384 L128 19.4 -> 22.0 (worse: not used, the switch is C == 256). Measurer (TriangleAttention, cuda, eval): d_pair 256 L128 ending 53.1 -> 49.0 us, starting 51.1 -> 46.9 (-8 %), L256 and L384 unchanged; all five self-attention classes eval L128 / L256 accuracy max err / tol 0.67 / 0.68, none over tolerance.
**Front drain** (the largest front stage): the two-slice round now stores one 64-column x 32-row 4 KiB box (`WF_BIG`, runtime `BIGF`, L <= 256, same bits): d_pair 256 L128 18.3 -> 18.5 us (level), L256 48.3 -> 47.3, d_pair 384 L256 98.2 -> 96.0 (-2 %), L128 34.0 -> 33.0; the front's drain already ran on eight warps with two slices a round, so its chain is shorter than the tail's was; its stage remains the largest of the front (3.5 us at d_pair 256 L128) and has no further fix.

### S90 inference, part 10: the front drain per slice, the d_pair 384 tail, the kernel boundary, the full re-score (cuda:b200_sm100 stays 66)

**Front drain per round** (`WF_TL` stamps of warp 10, CTA 0, 64-column x 32-row store boxes, two slices a round; us): d_pair 256 L128 round 0: TMEM load issued 7.20, loaded 7.33, ring wait done 7.49, pack + STS done 7.84, fence + syncwarp 7.90, store issued 8.16; round 1: 8.22 / 8.35 / 8.90 (ring wait +0.55) / 9.22 / 9.28 / 9.44; d_pair 384 L128 the same shape (rounds of 0.9-1.1 us). Per round: load wait 0.13, ring wait 0.16-0.55, pack of 64 values + 8 STS 0.33, fence 0.06, store issue 0.16-0.26: the same serial chain as the tail had (0.57 us a slice), on eight warps already; the drain of one 128 x 256 block is ~1.5 us against ~0.5 us of MMA, so the front is drain-paced (no-drain ablation -3.5 us at d_pair 256 L128). No fix beyond the 64-column box of part 9 was found (level to -2 %); the pack + STS step (0.33 us for 32 `F2FP` + 8 `STS.128`) is not explained by instruction counts and was not isolated.
**d_pair 384 tail**: both 192-column halves of a tile as ONE item (`WT_MERGE`, one 384-column accumulator, g / o loaded and gated once): bit-identical, but tail L128 19.4 -> 32.5 us, L256 62.2 -> 82.5, L384 117.7 -> 152.5 (80 KiB stages leave two, and one accumulator loses the drain / MMA overlap); rejected, off. The eight-warp drain (`WT_DW8`) is slower at d_pair 384 (19.4 -> 22.0) and stays d_pair 256 only. The d_pair 384 tail's ablation (stage loads -12.3 us, gate -11.6 us at L256) matches its L2 traffic: 336 KiB of g / o / W per 192-column item, 7 items a CTA at L256 = ~38 GB/s an SM = ~5.7 TB/s of L2 reads, i.e. the L2 read rate; the merge halved the g / o reads but lost more than it saved.
**Kernel boundary** (`_s90/bnd.cu`: 148 CTAs x 128 threads, graph replay, empty kernels / ~3 us kernels): PDL chain 0.73-0.91 us a boundary (1.4-1.6 with 200 FMA iterations of work), a plain chain 1.3-1.4 (1.8), and ONE kernel with a software grid barrier (monotonic counter, acquire / release) 1.22 us (1.74): a software grid barrier does not beat PDL; merging kernels pays only where the phases overlap instead of meeting at a barrier. The per-boundary ~3 us of the training step is the ramp-down / ramp-up of real kernels, not the launch. At d_pair 256 L128 the chain (front 18.0 + core 13.4 + tail 12.5 = 43.9 us) sits in a 46.9-49.0 us graph: the boundaries cost ~2-3 us in all.
**Full re-score** (`_s90/out/after8`, TriangleAttention eval, 118 cases, measurer, graph, L2 flushed, exclusive): SOL% all-MUFU min / median / max 5.6 / 41.1 / 65.8 (formula split 33.1 median, measured-loop split 37.8), row 188 6.4 / 25.2 / 40.4, cases >= 90 / 70 / 50 %: 0 / 0 / 43; against the 2.5.14 H100 tokens: 18 of 118 over 0.667, none slower than the H100, ladder violations 0 (worst: row 75 L128 0.741 / 0.723, was 0.80 / 0.79 before part 9). Accuracy L256 / 384 / 512: max err / tol 0.43 / 0.49 / 0.44.
#### S90 training, part 6: the key-side / query-side kernels as two graph branches (adopted, cuda:b200_sm100 67), software grid barrier cost

**Grid barrier microbenchmark** (`docs/b200/triattn_s90_train/gb/`, one CTA an SM, `red.release.gpu` + `ld.acquire.gpu` spin on a monotonic counter, 2000 barriers): 1.17 us (32 CTAs), 1.19 (74), 1.24 (128), 1.25 (148 CTAs); with ~100 FMAs of work between barriers 1.47 .. 1.55 us.
Against the ~3 us a kernel boundary costs under PDL in the L128 chain (part 5), a persistent kernel with software barriers would save ~1.5 us a boundary at best, and each merged kernel needs the role bodies of two kernels with different shared-memory, thread and TMEM layouts: not built.

**Built and adopted (67): `triattn_bwd(..., which)`** (0 both kernels, 1 the key side, 2 the query side + slab sum): while a CUDA graph is being captured the two kernels are two branches (`bias_only_dit.cuda.side.fork / join`), so the 40 SMs the key side leaves idle in its
second task round run query-side CTAs (kv: 256 tasks on 148 CTAs; q: 128 CTAs); the wide path (outputs given) and the fused d_pair 128 path (dq from the query call, dk / dv from the key call) both. Eager calls never branch.
`MINIWORLD_TRIATTN_B200_KVQ_BRANCH=0` turns it off. Module training step, graph, off -> on (us):

| case | L128 | L256 | L384 | L512 | L768 |
|---|---|---|---|---|---|
| d_pair 64 x 2 x 32 | 77.5 -> 73.5 | 189.7 -> 184.8 | 391.0 -> 387.2 | 739.4 -> 722.4 | 1906 -> 1878 |
| d_pair 64 x 4 x 32 | 90.4 -> 89.5 | 271.9 -> 268.2 | 631.8 -> 624.6 (ending) | 1242 -> 1210 (ending) | 3366 -> 3331 (ending) |
| d_pair 128 x 4 x 32 (fused) | 110.5 -> 110.0 | 285.8 -> 281.7 | | 1211.9 -> 1184.6 | 3287.6 -> 3252.0 |
| d_pair 256 x 8 x 32 | 191.7 -> 186.9 | 583.8 -> 575.1 | 1299.5 -> 1278.6 | 2587.9 -> 2533.7 | 6916 -> 6868 |
| d_pair 384 x 12 x 32 | | | 2097.5 -> 2078.9 | 4081.5 -> 4019.3 | 11118 -> 11069 |
| row 188 | unchanged (not the self-attention core) | | | | |

(-0.4 .. -5.2 %.) Measurer, all 84 training cases: SOL% min / median / max 3.4 / 18.8 / 25.7 -> 3.5 / 19.2 / 26.6 (`sol_cases_m3.tsv`; this includes the bf16 dyln of part 4 for row 188 and the merged inference changes); none at 50 / 70 / 90 %.
Checks on the merged tree (b200-2.5.x 214b12457, TriAttn 66 -> 67): `detm.py` 14 configurations x 5 runs all deterministic; accuracy (rows 72 / 188 / 241-245, L128 / 256 / 384 / 768): 47 GREEN, 7 YELLOW, 2 skipped (fp64 reference OOM at row 245 L768), no RED;
under graph capture (`cmpgrad.py`, p_drop 0) the output, dx and 7 of 8 parameter gradients are bitwise equal with the branch on and off and three replays of the branched graph are bitwise equal in all 10 configurations.
Observation (not caused by the branch, found on the way): the gradient of `to_out.weight` differs between two *independently captured* graphs of the same step (off vs off: max abs diff 57.6, case d_pair 64 x 4 L128, p_drop 0) while every replay of one graph is bitwise equal and eager repeats are deterministic (`detm.py`): an input of the
dWo path is read from allocator state that differs between captures (probably an uninitialised pad region); worth a separate look before the next release.

### S90 inference, part 11: the front pack, the wide-path core ablation (measured, nothing adopted; cuda:b200_sm100 stays 67)

**Front drain pack** (`WF_NOPACK` / `WF_NOSTS` / `WF_NOFENCE`, timing only; front kernel us, base / without): d_pair 256 L128 18.0 / 18.0 (no bf16 conversion: the `F2FP.BF16.F32.PACK_AB` pack, one instruction per two values, costs nothing), 17.0 (no staging writes: -1.0), 17.4 (no `fence.proxy.async`: -0.6); L256 47.6 / 47.4 / 45.4 (-2.2) / -; the 128B-swizzled write is conflict-free (each quarter-warp phase of a `STS.128` covers eight distinct 16-byte chunks `(p * 4 + c) ^ (lane & 7)` of eight rows). So pack + staging + fence are ~1.6 us of the 18 us front, not the 0.33 us a round the stamps suggested (the stamps, a `%globaltimer` read plus a global store per point, inflate short segments); no fix there.
**Core ablation** (`triattn_fwd`, head dim 32, kernel us, profiler, L2 flushed, one flagged build a process; `_s90/abl_core.py`, `jy.sh`): d_pair 256 L128 / L256, d_pair 384 L128 / L256: base 13.3 / 71.4 / 22.4 / 113.4; no K | V loads 13.3 / 71.8 / 19.1 / 113.6; no bias loads 13.3 / 74.6 / 19.1 / 114.8; no MMAs 12.4 / 70.6 / 17.5 / 107.1; no logits and no exponentials 11.4 / 62.7 / 16.2 / 96.8; no exponential 12.9 / 73.1 / 18.2 / 106.9; no TMEM loads of S 13.1 / 73.9 / 18.7 / 113.4; no P stores 13.1 / 73.8 / 18.5 / 112.3; **the softmax warps doing no arithmetic at all (`TA_FNOLOOP`) 9.5 / 53.5 / 13.4 / 84.2**. Attention FLOPs at L128 are 8 heads x 128 x 128 x 128 x 4 = 0.5 GF (0.3 us at peak); MUFU 2.1 M exponentials = 0.5 us. So 70-75 % of the core at these shapes is skeleton: the per-task chain (Q load -> QK^T -> S wait -> P -> PV -> epilogue drain) that three groups of a CTA only partly overlap, ~3.5 tasks a CTA at L128 (d_pair 256: 512 tasks on 148 CTAs, 3.46 rounds). Clock64 buckets of CTA 1 at L128 (`ftl6.py`, 7-10 sub-steps a CTA): ~2200-2900 clk a sub-step against ~900 clk of steady-state work: the QK^T issue warp 1670 clk issue (waiting for S buffers), softmax groups 1385-1825 clk of work, the epilogue group 1290-1390 clk waiting for the bias tile.
Rows a task re-tested (core us, 2 / 3 / 4 rows): d_pair 256 L128 13.3 / 15.1 / 14.9, L256 75.0 / 76.8 / 74.2; d_pair 384 L128 18.8 / 21.3 / 22.0, L256 118.5 / 115.8 / 125.0: the current rule (2 rows to L256 below 12 heads, 3 beyond) is the best. No fix for the skeleton was found; it needs the per-task chain restructured (a one-kernel L128 path for the wide rows, as `tri_l128` is for d_pair 64 / 128).
#### S90 training, part 7: the dWo "nondeterminism" across graph captures -- root cause: the test setup, not a kernel (no kernel change)

Symptom (part 6): `to_out.weight`'s gradient differed between two independently captured graphs of the same step (max abs diff 57.6, d_pair 64 x 4 L128, p_drop 0, also with the kv / q branch off).
Reproduced eagerly and in graphs with the allocator's free memory pre-filled with different byte patterns (`poison.py`: a 40 GB block and 3000 small blocks filled with 0x00 / 0xFF / 0x7F and freed, so the step's `torch.empty` buffers reuse them): with the module's DEFAULT init the gradient of `to_out.weight` (and, for the bias-only form, `to_bias`) changed with the poison in every case (self-attention d_pair 64 / 128 / 256, ending node, bias-only).
Bisect: (1) with random non-zero weights written over every parameter all 10 tensors (y, dx, 8 parameter gradients) are bitwise equal under all three poisons, eager AND graph, six configurations (d_pair 64 x 4 with and without dropout, 128, 256 x 8, ending node L256, bias-only): the kernels do not read allocator state;
(2) the default-init failures were not poison-specific: two modules built after the same `torch.manual_seed(0)` have DIFFERENT parameters (`po12.py`: parameter checksums differ, the fp64 PyTorch gold differs from itself in dWo, rel 0.99). `Linear(init="default")` draws its truncated normal through scipy from numpy's GLOBAL random state (`primitives._trunc_normal_init`),
which `torch.manual_seed` does not seed. In the default init `to_out` and `to_gate` are zero, so dWo = dy^T (sigmoid(g) o) is the only non-zero gradient and it changes with every construction (y and dx stay equal because the update is multiplied by the zero out-projection); every other regime hides it because the other gradients are compared with random overwritten weights.
(3) With `np.random.seed(0)` next to `torch.manual_seed(0)` the default-init step is bitwise equal under poison, eager and graph, six configurations (`poison.py`, `poison.py graph`, and `graph rand`), and independently captured graphs agree.
(4) `compute-sanitizer --tool initcheck` (D128 and D64 L128 eager steps) reports only `triattn_db_slabs` reading uninitialised memory (1.87 M / 0.59 M 16-byte reads): a false positive -- those slabs are written by the query kernel's TMA stores / bulk reduce-adds, which initcheck does not track; with the poison the dbias is bitwise unchanged.
(5) The same poison test (`poison_gen.py`, np seeded, random weights) on TriangleMultiplication (d_pair 128 outgoing L128, incoming L256, d_pair 256) and Transition (128, n = 4, L128 / L256), eager and graph: bitwise equal under all three poisons.
Regression test: `tests/gpu/test_triattn_b200_capture_bits.py` (poison fill + eager / graph, four configurations; it seeds numpy and documents the trap). Recommendation outside this branch: seed numpy in `measure` / test setups that build modules with default init, or draw `_trunc_normal_init` from the torch generator.

### S90 inference, part 12: cross-task pipelining, bias ring depth, the QK^T warp's waits (measured, nothing adopted; cuda:b200_sm100 stays 67)

- **Cross-task pipelining already exists**: the TMA warp prefetches the next task's Q (two Q buffers) and keeps the K / V and bias rings running across tasks; the QK^T warp runs up to NS - 1 = 2 sub-steps (one two-row task) ahead across task boundaries; the single O accumulator is the only per-task serial point and its double-buffered form (`TA_OB2`, the 512 TMEM columns: 3 x 128 S + 2 x 2 x 32 O) was level at L128 / L256 (part 8: 8.32 / 8.32, 10.38 / 10.39, 14.50 / 14.67 us at 2 / 4 / 8 heads). The first QK^T waits for the cold Q tile once per CTA (~1.4 us), the CTA then runs 3.5 tasks.
- **QK^T issue warp, waits split per site** (`TA_FTL` bucket 5 = the wait for PV(y - 3) to retire, i.e. the S buffer being free; `_s90/ftl6.py`, d_pair 256 / 384 L128, clk a sub-step of the SM): total 2056 / 2190: issue 706 / 655 (the MMA issue and commits), **wait for the S buffer's PV to retire 761 / 971**, wait for K | V 300 / 261, wait for Q 240 / 236. So 37-44 % of the QK^T warp's time is the S-buffer hand-back of the three-buffer ring; the softmax groups' work is 1365-1877 clk of each ~2600 clk sub-step.
- **Bias ring depth** (`TA_NB`, new knob, default 2): 3 stages: core d_pair 256 L128 13.3 -> 13.7 us, L256 74.9 -> 75.9, d_pair 384 L128 21.8 -> 19.4 (the d_pair 384 L128 baseline itself measured 18.8-22.4 us in this session's runs: run-to-run spread larger than the change); K | V ring 3 stages: 13.7 / 79.1 / 22.5 (worse); 7 / 8 K | V stages do not fit the shared memory. Not adopted.
- **Task balance** at L128, d_pair 256: 512 two-row tasks on 148 CTAs (68 CTAs run four, 80 run three); the other rows a task (3 / 4) are slower at every size (part 11), and a row-split mix needs one kernel with a runtime row count (the softmax groups are compile-time, `NGR = 2 | 3`); not built.
#### S90 training, part 8: the tall-K GEMM (`tallk`) timeline, two reduction variants (no gain, not adopted)

Per-CTA %globaltimer stamps inside `tallk_sm100` (D64 x 4 x 32 L128 training step; `docs/b200/triattn_s90_train/cta_tk.py`; the stamps are per cluster rank, the grid is 16 ranks x tiles, so the earlier "16 CTAs" was only the rank slots the stamp array holds) -- microseconds from the entry:
prologue (TMEM alloc, barrier init, tensor-map prefetch) done 0.5; the first K block lands 1.4 (0.9 us of TMA latency after the prologue); the 16 K steps (4 MMAs of 128 x 128 x 16, MN-major operands, ~0.25 us each) finish at 5.4; the partial tile is drained to shared memory and the
cluster barrier passed at 6.2-6.5; the distributed-shared-memory reduction (each CTA adds its 8 rows over 16 ranks in rank order) ends at 9.4-9.7 (3.1 us); the closing cluster barrier and TMEM release end the kernel at 10.2-10.5.
Variants (`docs/b200/rejected/triattn_s90_train_tallk_reduction_variants.diff`): two elements a thread with all 2 x 16 remote loads in flight before the adds (the reduction 3.1 -> 3.1 us); rows past M neither loaded nor added (-0.1 us at most). The reduction is not bound by its load count or by two serial rounds; it is the ~3 us of cluster-barrier + `ld.shared::cluster` latency
(mapa + a 16-byte remote load a rank), which only a different reduction shape (fewer ranks per tile and more tiles, or a global last-arriver sum) would change; the K loop (4 us) is MMA-bound, so a K split into more clusters shortens it but adds a partial pass (round 6: 8 x 2 parts 11.4 -> 10.1 us, +1.2 us in `tri_wfinish`). No source change; `cuda:b200_sm100` stays 67.

#### S90 training, part 9: `tallk` reduction by pushes (st.async + complete_tx) -- bitwise equal, no gain, not adopted

`-DTK_PUSH=1` (diff `docs/b200/rejected/triattn_s90_train_tallk_push_reduction.diff`, `MINIWORLD_TRIATTN_B200_TALLK_PUSH=1`): every rank writes its partial tile to its own shared memory, then pushes each slice to the owning rank with `st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32` (16 B a store,
64 KiB received per owner in a separate `[rank][rows][128]` buffer, one split cluster barrier arrive at the prologue / wait before the pushes instead of the barrier after the drain, `expect_tx(65536)` on the owner's mbarrier); the owner waits one mbarrier and adds the slices in rank order (the same bits as the remote-load version).
Dw / dx / all parameter gradients bitwise equal to the remote-load build (4 configurations: d_pair 64 x 4 L128 and L384, 64 x 2 ending L256, 256 x 8 L128; `dump.py`). Module training step, graph, remote loads -> pushes (us): d_pair 64 x 2 L128 73.4 -> 75.3, 64 x 4 89.7 -> 90.2, 128 ending 110.5 -> 110.4, 256 x 8 187.5 -> 188.6, bias-only 102.0 -> 100.1.
No gain: the ~3 us between the partial tiles being in shared memory and the owner's sum is not the remote-load round trip (st.async has no reduce form for fp32, so the adds stay on the owner). Not tried: 8 ranks with more tiles (the K loop doubles to ~8 us) and a global last-arriver sum.

#### S90 training, part 10: cluster barrier and DSMEM microbenchmarks; tallk through global memory and a merged `tri_wfinish_sum` (no gain, not adopted)

**Cluster barrier cost** (`docs/b200/triattn_s90_train/cb/cb.cu`, 128 threads a CTA, 200 KiB shared memory so one CTA an SM, 2000 arrive / wait pairs, grid = one cluster or 96 CTAs): `barrier.cluster.arrive.release` + `wait.acquire`
0.204 (cluster 1), 0.209 (2), 0.194 (4), 0.206 (8), 0.210 us (16); `arrive.relaxed` + `wait` 0.070 / 0.073 / 0.059 / 0.070 / 0.072 us. A 16-CTA non-portable cluster costs the same as a 2-CTA one: the barriers of `tallk` (two release / acquire pairs) are ~0.4 us of its 10 us, not the 1.6 us window.
**Distributed shared memory read** (`dsm.cu`, an all-to-all read, 192 threads, 16 loads in flight): a round of reads costs ~1.1 us whatever the cluster size up to 16 KiB a CTA (the remote-load latency); 64 KiB a CTA: 2.8 us (cluster 2), 4.7 (4), 6.3 (8), 8.0 (16). `tallk` moves 64 KiB a CTA through DSMEM (one 128 x 128 fp32 tile): the ~3 us reduction is a latency + bandwidth cost of that all-to-all, not of the barriers.
**Built and measured (module step, graph, us; bitwise equal output where noted):**
- the partial tiles through global memory (`TK_GLOB`: each rank stores its tile with `st.global.cg`, one cluster barrier, each owner loads its slice of the 16 tiles with `ld.global.cg` and adds in rank order; the closing barrier dropped): d_pair 64 x 2 L128 73.5 -> 73.9 / 73.0, 64 x 4 89.7 -> 89.4, 128 ending 110.4 -> 110.5, 256 x 8 187.6 -> 188.0, bias-only 101.7 -> 100.7; L256 within +-0.5. (The first version missed a `__syncthreads` before reading its own shared tile: 2 of 4 configurations differed in bits; the timing was measured on it and is the same.)
- `tri_wfinish` with the last block (a self-resetting ticket) summing the dgamma / dbeta partials, no `tri_wfinish_sum` launch (`WF_LAST`): bitwise equal (4 configurations); step 73.5 -> 74.3 / 74.8, 89.8 -> 89.9, 110.4 -> 110.4, 187.3 -> 190.3 (D256 x 8 L128), bias-only 101.8 -> 100.5; L256 +-0.3 (the D256 L256 +2.7).
Neither changes the step: a boundary the chain no longer has is not a time the step no longer takes (PDL hides most of it), and the tail of `tallk` is not on the critical path as it looks. `cuda:b200_sm100` stays 67. Diff: `docs/b200/rejected/triattn_s90_train_tallk_glob_wfinish_last.diff`.

### S90 inference, part 13: TMA ops per stage (the per-op cost rule) in the wide front / tail / core (measured, nothing adopted; cuda:b200_sm100 stays 67)

Rule applied (TriMul microbenchmark: a TMA op costs ~590-740 clk an SM whatever its size, 16 KiB boxes ~28 B / clk, 32 KiB ~53): what each stream issues and what removing / merging it is worth (kernel us, profiler, L2 flushed, one flagged build a process):
- **Wide front W stream** (d_pair 256: 20 x 16 KiB per tile): removing it entirely (`WF_NOLOAD`) saves 0.8 us of 17.9 (L128), 3.8 of 45.8 (L256), 1.5 of 33.0 (d_pair 384 L128), 10.9 of 97.6 (d_pair 384 L256); 2- and 3-chunk 3-D boxes (32 / 48 KiB) were built in part 11 and measured level. An op-count fix is bounded by those numbers (<= 1.5-4 % of the front).
- **Core K | V / bias boxes** (8 KiB K and V a sub-step, 32 KiB bias a key tile): `TA_NOKV` / `TA_NOBIAS` change nothing at d_pair 256 (13.3 / 13.3 / 13.3 at L128, 71.4 / 71.8 / 74.6 at L256) and 3.3 us at d_pair 384 L128 (a baseline that itself spreads 18.8-22.4 us); the TMA warp's issue bucket is ~350 clk of a 1270-clk sub-step at L1024-2048 and ~1200 clk of 1840 at L128. K and V as one box was measured level (part 11).
- **Wide tail stage stream** (per K chunk g 16 KiB + o 16 KiB + W 24-32 KiB): the stage loads are the tail's one load-bound piece. Timing-only builds (`WT_NOGO` / `WT_NOW`), tail us base / no g | o / no W: d_pair 256 L128 12.4 / 11.7 / 11.9, L256 40.8 / 35.6 / 40.1; d_pair 384 L128 21.6 / 19.2 / 19.2, L256 63.0 / 56.5 / 60.6; i.e. the g and o ops cost more than W. g and o as ONE 32 KiB 3-D box (`WT_GO1`, timing only, the data of o wrong): d_pair 256 L128 12.4 -> 12.2, L256 41.1 -> 38.6, d_pair 384 L128 19.5 -> 19.6, L256 61.6 -> 59.6 (-2.0 .. -2.5 us at L256, ~1 % of the module). A real merge needs g and o in one tensor (the front writing g and the core writing o into chunk-interleaved halves of one [R][2 x HDp] buffer, 3-D store maps in both) -- three kernels' layouts for <= 2.5 us at L256 and nothing at L128; not built.

## TriAttn inference: blockers per case group (S90 consolidation, tip 67, 2026-11-18)

Basis: TriangleAttention eval, `cuda`, 118 cases (rows 71-76, 188; starting / ending), measurer graph timing, L2 flushed, exclusive GPU (`_s90/out/after9`, `triattn_s90_inf_sol.md`, calculator `docs/b200/sol_triattn/sol.py` + `_s90/sol2.py`). SOL% = t_sol / t_measured with the all-MUFU floor; the measured-loop split floor (0.884 of it, `_s90/pipe_s90.py`) is shown in the summary only. Whole grid: SOL min / median / max 5.6 / 41.7 / 66.4 (formula split 33.1 median, measured-loop split 38.0), cases >= 90 / 70 / 50 %: 0 / 0 / 43; against the 2.5.14 H100 tokens 16 of 118 are over 0.667, none slower than the H100, ladder violations 0 (worst: row 75 L128 0.742 / 0.722, row 76 L128 0.695, row 74 L2048 / L1536 / L256 0.69-0.71); accuracy L256 / 384 / 512 max err / tol 0.43 / 0.49 / 0.44. Case counts below: 2 + 10 + 8 + 4 + 12 + 48 + 34 = 118.

| group | cases | SOL% min / med / max | measured time | binding resource of the floor | measured blocker (evidence) | attempted -> result |
|---|---|---|---|---|---|---|
| G1 row 188 (bias-only) L128 | 2 | 6.4 / 6.4 / 6.4 | 22.3 us | memory (1.4 us) | the fused one-kernel L128 path's fixed chain (graph floor 6.1 us + the kernel's LN 4.4, projections 6.7, bias barrier, heads; part 7 timeline) | q = k = 0 through `tri_l128` (36.7 -> 22.3 us, part 1) |
| G2 row 188 L256-768 | 10 | 13.0 / 28.4 / 40.4 | 46.9-266 us | tensor (PV) | PV bmm at 87 % of bf16 peak (77.9 us for 116 GF at L768); front 89 / tail 110 us vs 50 / 75 us traffic floors; cuBLAS token-major PV 248 us | v-plane front + gate-from-x plane tail (410 -> 262 us at L768, -27..-36 %, part 2); tail four x slots (110.6 -> 102.2, part 7); pipelined drain / direct stores / second drain group rejected (parts 5-6); own PV kernel not built (estimated <= 10 us) |
| G3 d_pair 64 / 128 L128 | 8 | 5.6 / 9.0 / 10.0 | 16.2-22.3 us | MUFU 6, tensor 2 | one-kernel `tri_l128` (part 7): LN 4.4 us, projections 6.7, grid barrier for the bias 6.9, heads 9.7-13.1 us of a 15.4 us kernel in a 6.1 us launch floor | the fused kernel itself (28-33 -> 16-22 us, part 7); polynomial split 4 of 16 pairs (part 10); nothing further attempted |
| G4 d_pair 256 / 384 L128 | 4 | 15.5 / 16.2 / 22.5 | 46.9-75.6 us | tensor | three PDL-chained kernels: d_pair 256 front 18.0 + core 13.3 + tail 12.5 = 43.8 us in a 46.9-49.0 us graph (profiler, part 10); front drain-paced (drain -3.5 us of 18), core 70-75 % skeleton (softmax warps doing nothing: 9.5 of 13.3 us), tail drain (8.9 us of 17.2 before the fix); boundary 0.7-1.6 us with PDL, a software grid barrier 1.2-1.7 us (part 10) | 64-column store boxes (part 9), eight tail drain warps (16.2 -> 12.5 us, -8 % of the module, part 9), `WF_BIG` front boxes (level), producer lanes, W K-chunk boxes, `WT_MERGE`, `TA_NB`, rows a task (all level or worse, parts 7-13); **the one-kernel / cluster path was NOT built** |
| G5 L256 (rows 71-76) | 12 | 15.5 / 23.0 / 30.7 | 45-246 us | MUFU 8, tensor 4 | front + core + tail chain with the core's per-task skeleton; d_pair 128 ending L256 0.688 of the H100 | PDL chain (part 6), polynomial 4 / 16 (part 10), L256 CTA-pair fusion (rejected, part 8), `TA_OB2` (level), rows a task |
| G6 L384-768 | 48 | 26.7 / 41.9 / 63.0 | 88-2989 us | MUFU 46, tensor 2 | the head-dim-32 core: 3-group S -> softmax -> PV -> QK^T ring, 81 % of the all-MUFU floor in the kernel at L2048 (1271 of 1024 clk a sub-step), group cycle 3813 clk = work 2481 + S wait 840 + bias wait 170 + other 300; the QK^T warp waits 37-44 % for the S buffer's PV to retire; TMEM 480 / 512 columns used (32 free) | polynomial pairs 0 / 2 / 3 / 4 / 6 of 16 re-measured (shipped 2 is within 0.3-0.8 % of best), HB core, 16-channel compact heads, ones-in-PV, split-P, 64-key halves (part 2: wait 18 -> 6 % but +8 % time), `TA_PVI` (level), single-warp issue, `TA_NB` bias ring |
| G7 L1024-3968 | 34 | 44.6 / 59.0 / 66.4 | 0.83-275 ms | MUFU 34 | the same core ring; SM clock 1867-1890 MHz of 1965 at ~990 W (4-5 %); d_pair 128 L1536-2048 0.69-0.71 of the H100 | as G6; at sustained clocks no polynomial split beats 0 by more than 0.4 % (p0 / p2 10217-10261 / 10262-10284 us at L2048) |

Where nothing (or not everything) was attempted: **(a) G4's one-kernel path** -- expected gain stated honestly: the chain's kernels sum to 43.8 us of a 46.9-49.0 us graph, i.e. the three boundaries cost 2-3 us in all, and a software grid barrier is not cheaper than PDL (1.2-1.7 us against 0.7-1.6); a fused front + core + tail kernel would remove at most ~2-3 us (4-6 %) of four cases (3 % of the grid) while needing a cluster / streamed-weight design (the weights are 512 KiB - 1.2 MiB); not built, and it is not the dominant remaining group (G6 + G7: 82 of 118 cases, all bound by the core ring). **(b)** the S-buffer hand-back of the core (G5-G7, the largest remaining blocker): the levers that need TMEM columns (a 4th S buffer: 128 needed, 32 free; fp16 S) were not built; the 64-key-half form was measured only on the pre-HB core. **(c)** the front drain (3.5 us at d_pair 256 L128, 11 us at L256, 5 / 18 us at d_pair 384) has no fix beyond the 64-column box; its per-round 0.33 us pack + staging step is not explained by instruction counts (part 11 ablation: pack 0, staging writes 1.0 us, fence 0.6 us of the 18 us front). **(d)** g and o as one TMA box in the wide tail (-2.0 .. -2.5 us at L256 in a timing-only test) needs a combined layout written by the front and the core; not built. **(e)** G1-G3 SOL is dominated by the launch floor: with the +5 us floor their SOL is 28-37 %.
### TriAttn training: blockers per case group (S90 training round, tip 1f3ff60d7, `cuda:b200_sm100` 67)

**Re-score at the tip** (measurer, cuda, graph, L2 flushed, one exclusive run a length; `docs/b200/triattn_s90_train/sol_cases_tip.tsv`; SOL% = t_sol / t_measured of the SOL study's floors). 84 training cases, dtype `master`:

| group | cases | SOL% min / median / max, start of round -> tip |
|---|---|---|
| all | 84 | 3.4 / 18.8 / 25.7 -> 3.5 / 19.1 / 26.0 |
| row 72 (D64 H4 dh16) | 12 | 5.3 / 17.0 / 23.6 -> 5.5 / 17.4 / 23.9 |
| row 241 (D64 H2 dh32) | 12 | 3.4 / 13.6 / 20.5 -> 3.5 / 13.8 / 20.8 |
| row 242 (D64 H4 dh32) | 12 | 5.9 / 17.0 / 23.2 -> 5.9 / 17.4 / 23.3 |
| row 243 (D128 H4 dh32) | 12 | 7.5 / 17.4 / 23.2 -> 7.7 / 17.7 / 23.1 |
| row 244 (D256 H8 dh32) | 12 | 11.3 / 21.5 / 22.1 -> 11.5 / 21.4 / 22.3 |
| row 245 (D384 H12 dh32) | 12 | 16.6 / 24.2 / 25.7 -> 16.7 / 24.1 / 26.0 |
| row 188 bias-only | 12 | 4.2 / 18.6 / 24.9 -> 4.2 / 19.0 / 25.6 |

Cases >= 90 / 70 / 50 %: 0 / 0 / 0. H100 comparison (`t6.py`, the six classes, 198 training cases, B200 cuda / H100 cuda): over 0.667: 3 (the fused d_pair 128 L128 cases: TA starting 0.758, ending 0.747, AF3 starting 0.759), none slower than the H100, geomean 0.502, ladder (cuda < triton < pytorch) violations 0.

**Blockers by group (measured numbers; what was tried is in parts 1-10 above):**
- *Self-attention, L >= 384 (rows 72, 241-245: ~55 cases, 11-26 %)*: the core backward (kv 300 + q 286 us of 1213 at D128 L512; 41-48 % of the step). The key kernel is bound by its tcgen05 / barrier / operand chain: with all softmax work removed it is 242 us (81 %), 939-959 clk a 64 x 128 stage, 624 clk with the Q / dO / bias loads also removed; three softmax groups, early S^T release, spinning vs suspending waits, a multi-lane producer, deeper rings, the FMA-polynomial exp, a bias tile in one TMA op (-0.15 .. -0.4 %, bitwise equal) all measured flat or worse; the
  TMA op count is not the lever: removing every Q / dO / LSE / delta / bias load after the first stage gains 9 % of the kernel, so merging half of the ops can gain ~4 % of the kernel (~1 % of the step). Next in size: the forward core (MUFU loop at ~56 % of the ex2 peak), then kernels at 1.1-2.6x their byte floors (front, tail, gate backward, head backward + wgrad, wide whbwd 472 / wtail 281 us at D384 L512). The SOL study's own floors assume the dense 1704 TF for K = 32 MMAs, which this code cannot reach.
- *L128 / L256 (rows 72, 241-243: 24 cases at 3.5-10 %)*: chains of 10-14 kernels; per-CTA timelines (D64 L128): `tri_l128` 9.8 us body (one tile a CTA, a serial chain), `tallk` 9.5 (K loop 4.0 MMA-bound, DSMEM reduction 3.1: latency 1.1 us + 64 KiB a CTA at 2.8-8 us; barriers 0.2 us at any cluster size), bwd_kv 4.6 + bwd_q 4.7, whbwd 2.6, wgbwd 0.8, `tri_bwd_pre` 2.8; the seven kernels do ~44 of the 90 us, the rest is ramp-down / ramp-up around ~13 boundaries.
  Done: kv / q graph branches (-1 .. -5 %). Measured flat: draw on a branch, tallk reductions (shared loads, push, global), merged `tri_wfinish_sum`. Ablation upper bounds (timing only, outputs wrong): removing `tri_bwd_pre` gains nothing (+-2 us, so folding it into `tri_wgbwd` is not worth building); removing `tri_db_rows` gains 2.3 / 2.6 / 3.2 us at L128 (D64 x 2 / D64 x 4 / D256 x 8) and 5.7 / 7.6 / 8.0 us at L256 (3-4 %): the one L128 / L256 item left with a measured ceiling.
  Not built: `tri_db_rows` folded into `triattn_db_slabs` (strided 2-byte writes into the pre-zeroed pad columns; the slab sum inside `tri_db_rows` was rejected in round 7), the per-phase timeline of `tri_l128` (its body is one serial tile chain; the Transition-style overlap of its phases is a rewrite).
- *Row 188 bias-only (12 cases at 4-26 %)*: 30 launches of cuBLAS + glue; at D128 L512 568.9 -> 554.5 us after the bf16 `dyln` (part 4); the unbuilt plane head / tail (gate_u 30.8 + out GEMM 17.5 + scale / residual 35.1 us = 83 us against a 56 us byte floor, -27 us = 5 %); glue kernels ln_fwd 35 (floor 22-29), ln_bwd 81 (floor 56-73), gate_bwd 67 (56), dP GEMM 41 (K = 16384 on 64 CTAs, 2x its bytes / flops).
- *Determinism / initialisation read*: no kernel reads uninitialised memory (poison-fill proof, part 7); the earlier "dWo differs between captures" was numpy-seeded default init in the test setup.

### S90 inference, part 14: the fp16-S ring (rewrite-class experiment; timing-only builds, rejected; cuda:b200_sm100 stays 69)

Idea: S accumulated in fp16 (tcgen05 `kind::f16`, f16 D, q / k fp16) halves an S buffer to 64 TMEM columns, so the same 384 columns hold SIX buffers (two a softmax group) and the QK^T of a group's next sub-step no longer waits for PV(y - 3) to retire. Built as flagged timing-only experiments in `triattn_sm100.cu` (default off; inputs are bf16 bits read as fp16, results WRONG, timing valid): `TA_S16` (six 64-column buffers, `ID_S` f16 / f16, the softmax loads 8 columns = 16 scores and converts them in place with `cvt.f32.f16`) and `TA_H16T` (on top of it: the packed fp16 loop HFMA2 logits, HSUB2 offset, `ex2.approx.f16x2`, HADD2 row sum, P as fp16 pairs, PV on fp16 operands via `TA_PH`). Core, 4 heads, 3 rows, graph x 20, us (base / `TA_S16` / `TA_H16T`): L512 184.6 / 201.6 / 195.3; L1024 1300-1362 / 1392-1423 / 1315-1330; L2048 10228-10269 / 10583-10604 / 9946-9947 (-2.8 %). K | V ring of 6 stages: no help (H16T 202 / 1373 / 10346, base 187 / 1356-1366 / 10302-10349).
Timeline (`ftl6.py`, L1024, clk a sub-step; base / S16 / H16T): the softmax groups' wait for S 281 / 54 / 42, the QK^T warp's wait for the PV retire 710 / 106 / 101, as intended; but the sub-step goes 1318 / 1415 / 1374 clk: the softmax work rises 831 / 1155 / 1108 (the fp32 conversion costs ~320 clk a sub-step; even the packed loop's work window is MUFU-saturated, 3 x 1100 clk against the 3 x 1024 clk of 128 MUFU pairs, and carries no polynomial pairs, which the base loop uses 2 of 16 for), and the K | V ring becomes the wait (QK^T warp 652-739 clk for K | V, TMA 864-889 clk for a free stage). So removing the S wait moves the bound to the MUFU (74 % busy at 1024 / 1374) and the K | V cycle; it does not convert into time. A real build would also need fp16 q / k / v / bias from the fronts, an fp16 polynomial, accuracy against fp64 and the tails' o; the measured ceiling of the packed loop is -2.8 % at L2048 and +/- 0 at L1024: not built. The 2-CTA split of S across a pair (halving S columns a CTA) attacks the same wait and is bounded by the same MUFU / K | V ceiling: not built.
#### S90 training, part 11: `tri_db_rows` folded into the slab sum (adopted, cuda:b200_sm100 69); K / V as TMEM operands bounded; row-188 plane head / tail

**Adopted (69): `triattn_db_slabs_rows`.** The wide backward's slab sum (starting-node bias orientation, head dim 32) writes the head backward's dbias rows [R][64] directly -- bf16, the H heads of row (b, j, k) at `b S^2 + j S + k` (the same linear index as the plane, so the sum's four results are four 2-byte stores), and extra blocks of the same launch write the zero columns [H, 64) (16-byte stores from the first multiple of 8, 2-byte before it) --
instead of a fp32 plane and a `tri_db_rows` launch. `MINIWORLD_TRIATTN_B200_DBROWS=0` turns it off; `bt` (the transposed-bias orientation) keeps the old kernel. (A pybind default `torch::Tensor()` is rejected when a call omits it: every `triattn_bwd` call now passes `db_rows` explicitly.) The timing ablation that sized it (removing `tri_db_rows`: -2.3 .. -8 us at L128 / L256) was exceeded (the old kernel's strided reads were slower at large L).
Module step, graph, off -> on (us): d_pair 64 x 2 L128 73.6 -> 71.2, L256 184.7 -> 178.0, L384 387.2 -> 374.7, L512 722.4 -> 689.5, L768 1879.3 -> 1800.9; 64 x 4 (ending at L384+) 89.6 -> 88.4, 268.1 -> 261.0, 624.5 -> 612.7, 1208.9 -> 1180.2, 3330.6 -> 3256.9; 256 x 8 187.7 -> 186.0, 576.1 -> 569.7, 1280.9 -> 1271.4, 2537.0 -> 2516.5, 6869.9 -> 6809.1; 384 x 12 L384 2082.5 -> 2079.3, L512 4020.9 -> 4007.8, L768 11073.7 -> 11051.3 (-0.2 .. -4.2 %). The fused d_pair 128 path and row 188 do not use it.
Checks: bitwise equal output / gradients to the old kernel (4 configurations); allocator poison fill 0x00 / 0xFF / 0x7F bitwise equal, eager and graph, six configurations (`poison.py`); `detm.py` 14 configurations x 5 runs deterministic; accuracy (rows 72 / 188 / 241-245, L128 / 256 / 384 / 768): 47 GREEN, 7 YELLOW, 2 skipped (fp64 reference OOM), no RED.
Measurer, all 84 cases (`sol_cases_v69.tsv`): SOL% min / median / max 3.5 / 19.1 / 26.0 -> 3.6 / 19.5 / 25.8 (row 241 median 13.8 -> 14.4, row 242 17.4 -> 17.8, row 72 17.4 -> 17.8); none >= 50 %.

**K / V as TMEM operands (and with it the fp16 accumulator): bounded, not built.** The point of an fp16 S^T | dP^T accumulator would be the 128 TMEM columns it frees for K_r / V_r (64 columns) so the key kernel's four shared-fed MMAs per stage become TMEM-A ones (~65 -> ~30 clk each).
`TB_TSA` (timing ablation, wrong results: the S^T and dP^T MMAs issued in the ts form from TMEM columns of the dV | dK region), core key kernel, 4 heads: L512 299.2 -> 295.2 us (-1.3 %), L768 915.0 -> 901.2 (-1.5 %); with the softmax removed (the chain floor) 242.3 -> 238.4 (-1.6 %) and 710.6 -> 692.5 (-2.5 %).
The shared-fed S^T / dP^T MMAs are not what binds the chain: even a free operand feed gains <= 1.5 % of the kernel (< 0.5 % of the step), so neither the TMEM re-budget nor the fp16 accumulator (and its accuracy cost) is worth building. Diff: `docs/b200/rejected/triattn_s90_train_tsa_timing_ablation.diff`.

**Row-188 plane head / tail: rejected with numbers, not built.** Fusing gate_u (30.8 us) + the out GEMM (17.5) + scale / residual (35.1) at D128 L512 = 83.4 us against a 56 us byte floor is -27 us (4.7 % of 569 us; ~60 us / 5 % at L768); `tri_tailg_pl` has neither dropout nor a `u` / gate-plane output and the training backward needs y, the LN statistics, g and u, so it is a new front variant (v + g planes, statistics, y) and a new tail (g plane in, dropout, u out) plus the backward's transposed-frame pieces.
Value: 12 cases at 4-26 % SOL, +0.2 .. +1.3 SOL points on those cases, +0.1 on the 84-case median.

### N1 pass (2026-11-18, agent n1-triattn, GPU 1): the L128 training chain, the key-side backward's per-warp timeline, a merged key + query launch -- built and measured, nothing adopted (cuda:b200_sm100 stays 69)

Tools (all in `docs/b200/triattn_n1/`): `tl.py` (kernel timeline of one module training step, graph, grads reset per step so it matches the measurer), `ab.py` (module training step, variants of `b200_triattn` flags captured as separate graphs, interleaved >= 4 rounds, L2 flushed), `kq.py` / `kqp.py` / `kvb2.py` (core backward alone, which = 0 / 1 / 2 / 3), `kqs.py` / `kqm2.py` / `tlk.py` (per-CTA / per-warp %globaltimer stamps into GLOBAL memory, `triattn_n1_experiments.diff` holds the `N1_STAMP*` macros).

**Blockers table, d_pair 128 x 4 heads L128 starting (the three H100-1.5x misses, 0.747-0.759; step 110.8 us measurer / 117.4 us in `ab.py` with its extra grad bookkeeping), kernel durations in a graph replay (profiler), us:**
rng_scalar_write 1.5 -> drop_scale 2.0 -> tri_pack 3.4 (serial, 6.9 before the front) | tri_front 11.7 | triattn_fwd 18.2 (9.3 hidden by PDL) | tri_tail 10.0 (3.0 hidden) | tri_gate_bwd 12.6 | triattn_bwd kv + q (two graph branches) ~21 incl. db_slabs 2.6-4.9 | memcpy32_post 1.2 | tri_head_bwd 11.7 | tri_wgrad 16.6 | tri_slot_sum 5.0 | tri_param_finish 3.6. Needed: -12 us.
Floors of the same kernels (bytes / flops) are 1-5 us each; the chain is 13 dependent kernels.

**What a key-side kernel (kv) is made of at L128 (per-warp stamps before / after the last block barrier, CTA 1, us from the graph's first kernel):** the TMA warp is done at 5.1, the S^T MMA warp at 6.7, the grad-group warps at 8.2-9.2 and every warp leaves at 9.2; a one-task CTA (CTAs 108-147) leaves at 5.6-6.1. So a task is ~3.4 us (4 stages x 0.85 us = the chain-bound stage period measured at L512), the first one carries ~1.6 us of start-up. 1024 stages / 148 CTAs = 6.9 stages a CTA against the 8 the 2-task CTAs run: the kernel is at ~90 % of stages x stage period; there is no idle time to remove, only the stage period (chain) itself. (A trap found on the way: a thread-0 `%globaltimer` store placed after `__syncthreads()` was scheduled BEFORE the barrier completed -- body ends 4.4 us earlier than the real ones. Use per-warp stamps at the last instruction.)
Clock64 timeline of a 2-task CTA (`tlk.py`, warm): first S^T issued at 1.9k clk, a group's dK / dV drain of task 0 takes 2.0k clk (wait for the last grad MMAs + TMEM load + pack + store), during which the other task's first stage sits ready.

**Experiments (core backward alone, graph, L2 flushed, interleaved; module step by `ab.py`):**
| experiment | result |
|---|---|
| `triattn_bwd(which=3)`: key side and query side in ONE grid (CTAs split by work, 352 threads, bodies as device functions, slab count halved: per = 2), dq / dk / dv bitwise equal to which = 0, db within 2.4e-6 (other slab partition), deterministic | core alone L128 32.8 -> 28.6 us (its two launches + biasT); MODULE step L128 d_pair 128 117.4 / 117.5, d_pair 64 x 2 80.7 / 81.8, 64 x 4 97.0 / 96.8, 256 x 8 195.3 / 197.3 (off / on): level. per = 1 / 3 / 4 split: 53.2 / 34.8 / 40.9 us. Reason: at 5.0 + 3.4 (n - 1) us a CTA the two waves are as full as they get (sequential 8.4 + ~6 vs merged 11-15). Not kept. |
| `prefetch.tensormap` of the 7 maps at kernel entry (kv, q) | 10.75 -> 10.50 us (timer tick 0.26): none |
| dK / dV drain of task t after the first stage of task t + 1 (software pipelining of the grad-group warps; dk / dv bitwise equal, deterministic) | kv core L128 18.4 / 18.4, L256 55.5 / 55.3, L512 309.2 / 307.2 us (-0.7 %): none |
| ablation: no drain at all (timing only) | L128 -11 %, L256 -7 %, L512 -4.6 % of kv: the free-drain ceiling; the pipelined form takes <= 1/6 of it (the grad MMA warp waits for the accumulators' release (`de`) before the next task's first GEMM) |
| bare launch of the same kernels: empty kernel 1.4 us, prologue + TMEM alloc / dealloc only 2.1 us, whole kv body 10.4 us (profiler) | the launch is not the cost; the body (8 stages) is |
| `KQ_PDL=1` (merged launch as a programmatic dependent) | 117.5 / 117.5 |

**Fused dQ in the key-side backward (the idea of this pass), bounded, not built:** (1) TMEM: S^T | dP^T 2 x 128 + P^T | dS^T 2 x 64 + dV | dK 2 x 64 = 512 of 512 columns; the dQ accumulator (64 q x 32 fp32 = 32 columns a group) has no room without dropping a softmax group or serialising dP^T behind the exponentials (both lengthen the chain that already bounds the kernel at 81 %); (2) at L > 128 the sum over key blocks needs a deterministic cross-CTA reduction: fp32 slabs are 4 key blocks x L^2 x H x 32 x 4 B = 537 MB written + read at L512 H4 (~190 us at 5.7 TB/s), a 4-CTA cluster exchange costs ~6 KiB a stage at ~40 B/clk/SM (+15 % of the 1040-clk stage); at L128 (one key block) no reduction is needed but the dbias partials double (2 rows a CTA instead of 4: 16 MB instead of 8 MB of slabs) and the merged launch above shows the two waves cannot be packed tighter. Estimated -8 % of the core backward at best (~-4 % of the step) with a full TMEM re-map: not built; the 32-query / 4-group variant needs 4 x 128 + 4 x 32 + 4 x 64 columns for 4 groups = over 512.
**Result:** no source change in the shipped tree; the three L128 misses stay at 0.75 (they need a different structure: ~13 dependent kernels at ~3 us of boundary each; the only candidate of that size is a persistent backward megakernel, weeks of work). Experiment source: `docs/b200/rejected/triattn_n1_kq_merge_drain_stamps.diff`.

### N1 pass 2 (2026-11-18, agent n1-triattn): the bias-only paths LayerNorm forward / backward and gate backward on B200 kernels, the weight-gradient GEMMs as graph branches (ADOPTED: TriangleAttention cuda:b200_sm100 71 -- its own entry, AF3 / RF3 / Chai1 / Chai1Confidence stay at the base 69 -- and Bidirectional 5)

Found while timelining row 188 (`tl.py`, L512 starting, graph): its glue runs the H100 sources (`h100_sources/triattn_generic_sm90.cu`, `triattn_bidir_sm90.cu`) built for sm_100a unchanged; three of them sat 15-40 % above the clone-rate byte floor
because of their memory pattern (a row's loads are issued one dependent step after the other, the next row only after the previous one's stores, a 132 x 16 block cap from the H100, a partial-sum row for every 16 rows):
`ln_fwd_k<16>` 37.2 us (floor at the measured clone rate ~29), `ln_bwd_k<16>` 81.1 (floor 58), `gate_bwd_fused_k<128>` 66.6 (floor ~56).
New `b200_sources/lnbwd_sm100.cu` (shared sources stay byte-identical; routed by `bidir_triattn_h100._ln_fwd / _ln_bwd / _gate_bwd`, `MINIWORLD_BIDIR_B200_LNBWD=0` turns it off, `b200_bidir.lnb`, in the sm100 prebuild plan (`triangle_attention/cuda/__init__.py`)):
* `ln_bwd128` (C = 128, dyln bf16 or fp32): x / dyln / dy of a row step issued together and one step ahead of the arithmetic (the shared kernel loads dy only after both reductions), weights in registers, 2 persistent CTAs an SM (296 partial rows for d gamma / d beta instead of 2112: the column sum shrinks), streaming loads / stores. Per-row arithmetic in the shared kernel's order: dx bitwise equal; the fp32 d gamma / d beta partials regroup rows (max relative difference of the column sums 3e-7, deterministic for a fixed grid). (An ending-orientation `trans` variant that walks the PHYSICAL rows is in the file, 110.6 -> 84.0 us at L512, but no caller passes trans = true on this path.)
* the two weight-gradient GEMMs as branches of the captured graph (`B200_SIDE`, `MINIWORLD_BIDIR_B200_SIDE`, bit 0 dWo = dt^T u, bit 1 dW = dc^T y; one sample, graph capture only, `bias_only_dit.cuda.side`): both are tall-K split-K cuBLAS GEMMs (9.3 / 9.8 us at L128, 36 / 40 us at L512 with their reduce) that nothing reads before the function returns: dWo beside the gate backward and the core GEMMs (L <= 256 only: above it the core GEMMs fill the SMs and the branch slows the chain), dW beside the dyln GEMM and the LayerNorm backward (every length). Measurer, side off -> dWo + dW (L <= 256) -> dW alone (all L), ms ending / starting: L128 0.1113 / 0.1092 -> 0.1059 / 0.1039 (-4.9 %); L256 0.2195 / 0.1947 -> 0.2141 / 0.1900 (-2.5 %); L384 0.3940 / 0.3359 -> 0.3886 / 0.3293 (-1.4 / -2.0 %, dW alone); L512 -1.0 / -1.1 %; L640 -0.9 / -0.7 %; L768 -0.6 / -0.3 %. Bitwise equal output and gradients (graph vs graph, `ab2.py`). BidirectionalTriangleAttention takes the same two branches (-2 .. -4 us). Not kept: a third branch for the dV contraction beside the dP contraction (`rejected/triattn_n1_dv_branch.diff`: -2 .. +18 us, no consistent gain).
* `colsum256`: the d gamma / d beta partial rows summed in row order in ONE launch (8 CTAs of 32 columns, 8 row groups added in a fixed order); the shared `bo_colsum` is two launches (2.1 + 5.4 us at L512) -- ~4 us at every L.
* `ln_fwd128`: the next row step's load before this one's reductions, persistent 4 CTAs an SM. y / mean / rstd bitwise equal.
* `gate_bwd128` (K = 128): the shared kernel's arithmetic, the gate / o plane loads of a tile issued BEFORE its mma.sync GEMM (the shared kernel loads them after it: an exposed HBM round trip a tile). dO and dG bitwise equal.
Kernel alone (graph, L2 flushed, us, shared -> new): ln_bwd L384 / 512 / 768 53.2 / 81.9 / 161.8 -> 42.8 / 69.6 / 141.4 (bf16, starting layout; fp32 dyln within 1 us of it); ln_fwd 26.6 / 41.0 / 83.8 -> 22.6 / 34.8 / 69.5; gate_bwd 49.1 / 77.8 / 159.8 -> 47.1 / 71.8 / 149.5 (all include the 6 us graph floor; 3 CTAs an SM is worse for all three: 53.2 / 86.0 / 182.2 for the gate backward, 45.1 / 63.5 / 73.7 for ln_bwd).
Module step, row 188 `TriangleAttention` (use_self_attention = False, p_drop 0.25), graph, L2 flushed, `measure.speed` (2 runs each, ms, before -> after; ending / starting):
| L | ending | starting |
|---|---|---|
| 128 | 0.1192 -> 0.1061 (-11.0 %) | 0.1177 -> 0.1042 (-11.5 %) |
| 256 | 0.2361 -> 0.2142 (-9.3 %) | 0.2094 -> 0.1901 (-9.2 %) |
| 384 | 0.4184 -> 0.3884 (-7.2 %) | 0.3581 -> 0.3278 (-8.5 %) |
| 512 | 0.6507 -> 0.6105 (-6.2 %) | 0.5727 -> 0.5327 (-7.0 %) |
| 640 | 0.9876 -> 0.9427 (-4.5 %) | 0.8892 -> 0.8456 (-4.9 %) |
| 768 | 1.3010 -> 1.2467 (-4.2 %) | 1.1884 -> 1.1342 (-4.6 %) |
SOL% (t_sol of `sol_cases_base.tsv` / measured, new times): ending 4.2 / 10.5 / 15.9 / 18.8 / 19.3 / 22.9 -> 4.8 / 11.8 / 17.6 / 20.4 / 20.7 / 24.6 (base column from the study's own run; the base re-run of this pass reads 0.2-0.5 % lower times); starting 4.3 / 11.7 / 18.4 / 21.3 / 21.3 / 24.9 -> 4.9 / 13.3 / 20.9 / 23.4 / 23.0 / 27.1 (row 188 median 18.6 -> 20.6). `BidirectionalTriangleAttention` training step (`ab2.py`, d_pair 128 x 4, graph, same process): L128 137.2 -> 127.2, L256 305.2 -> 286.7, L384 579.6 -> ~550, L512 907 -> 877 us (-7.3 / -6.1 / -5.1 / -3.4 %), outputs and every parameter gradient bitwise equal to the shared kernels'; B = 2 / 3 and ragged L200 also bitwise equal and faster.
Accuracy basis: `measure.accuracy` (`acc188.yaml`, `accbi.yaml` in `docs/b200/triattn_n1/`), training, master dtype, random non-zero weights (the harness' seeded init), vs the fp64 gold, tol = max(err_pytorch, 8e-3): row 188 L128 / 384 / 768 starting + ending, 6 cases: 3 GREEN, 3 YELLOW (1.0-1.14 x tol: 9.675e-3 / 8.778e-3 / 9.664e-3, the same errors to the digit as the base tree), no RED; all six lengths L128-768, train + eval, starting + ending (24 cases, final tree): 17 GREEN, 7 YELLOW, 0 RED (every YELLOW a training gradient at 1.04-1.14 x tol, as in the base tree); BidirectionalTriangleAttention L128 / 384 / 768: 2 GREEN, 1 YELLOW (9.535e-3, identical to the base tree), no RED; eval (same files with modes: [eval]): 6 + 3 cases all GREEN. Determinism: `det.py`, three runs of the training step with N(0, 0.1) weights (gradients |max| 4e2-5e2), L384 starting / L512 ending / Bidirectional L384: output, dx and all six parameter gradients bitwise identical between runs. Prebuild plan (`_prebuild --arch sm100 --check`): 787 -> 788 artifacts, the new extension `triattn_b200_ln_v1` built, 0 not built, nothing else changed (`work/n1-triattn/prebuild_*.txt`).
Measured and NOT kept in this pass: (a) the d gamma / d beta column sum folded into `ln_bwd128` (last CTA to arrive sums the partial rows, ticket in a `__device__` word; FOLD in the source, off): microbench ln_bwd + the two `colsum` launches 94.2 -> 79.8 us at L512 looked better than the unfolded kernel + colsum (~77) only against the OLD kernel; in the module and the measurer it costs 1-5 us more than the unfolded form (L256 ending 0.2224 -> 0.2239, L512 0.6212 -> 0.6243, L768 1.2552 -> 1.2625 ms; the dead code in the kernel itself already cost ~1.5 us until it was made a template parameter); gm = 1 (one CTA an SM) 18.4 us at L128 but 118.9 at L512. (b) dP bmm split-K x4 as one batched bmm + sum (`dpk.py`): -6 us at L512, -2 us at L384, +8 us at L768, not adopted; the weight-gradient tall-K GEMMs (dWo, dW) as batched split-K (`tallk.py`) are slower than `torch.mm` at every split (+14 .. +100 us). (c) routing d_pair 128 L128 training through the wide path (`MINIWORLD_TRIATTN_B200_D128_WIDE`, `rejected/triattn_n1_d128_l128_wide_route.diff`): ab harness 114.1 vs 117.4 us, but measurer 0.1194 -> 0.1208 / 0.1215 ms (slower): rejected.
Still above its byte floor on this path (not touched): the dP bmm (K = 16384, 64 CTAs: 42 us vs ~25), the dWo split-K GEMM (36 us vs ~22), the L128 / L256 chains of ~27 kernels at 2-8 us each (logits + softmax + copy, bo_rowdot + smbwd, the two colsum launches, drop_scale + cast_w are 2 us launches that a fusion would save at ~2 us each; a fused logits + softmax + bf16 kernel (`patch_sm.py`, `smb.py`) is bitwise equal to the three-launch chain but measured no faster in a graph: removed).
Also measured and rejected in this pass: the same graph-branch idea on the WIDE backward's weight-gradient GEMMs (dWo, the bias column sum, dWo2 beside the G GEMM and `tri_wfinish`, `rejected/triattn_n1_wide_gemm_branches.diff`, graph step off -> on us: D64 x 4 L256 271.1 -> 271.0, L384 610.0 -> 616.6; D64 x 2 189.3 -> 192.2 / 386.5 -> 387.7; D256 x 8 581.3 -> 597.8 / 1293 -> 1329; D384 x 12 907.9 -> 957.2 / 2091 -> 2205: the bandwidth-bound GEMMs only compete); the existing `MINIWORLD_TRIATTN_B200_SIDE=1` (dropout draw beside pack / front / core) re-measured on the three L128 misses by the measurer: 0.1195 / 0.1174 -> 0.1210 / 0.1210 ms (worse, as before).
Also rejected: the fp32 -> bf16 weight cast (`cast_w_k`, 2 us) as a branch beside the LayerNorm forward (`rejected/triattn_n1_cast_branch.diff`: L128 95.4 -> 94.9 / 107.2 -> 105.2 us, L256 / L384 within noise); a pipeline-parameterised copy of the shared `tail_k` (K = 256, channels a stage 16, 4-5 stages, `patch_tail.py` / `tlb.py`): 122.9 us against the shared kernel's 116.8 at L512 (bitwise equal, slower; 3 stages and 32-channel stages faulted and were not pursued). The remaining K = 256 kernels of the Bidirectional class (`tail_k<256>` 112-122 us, `gate_bwd_fused_k<256>` 137 us at L512: 60 % above their byte floors) look MUFU / mma.sync bound (2 MUFU an element for the gate's sigmoid) rather than latency bound; not touched.

### N1 pass 3 (2026-11-18, agent n1-triattn): ladder + H100 1.5x re-check at 71 / 5, tuned cublasLt tall-K GEMMs (ADOPTED, same versions 71 / 5), Bidirectional K = 256 ablations (measured, no fix)

**Ladder and H100 1.5x** (fresh B200 cuda medians of this tree, measurer graph, L2 flushed, all 214 cases of TriangleAttention (84 train + 118 eval) and BidirectionalTriangleAttention (6 + 6); triton / pytorch rungs from the B200 tokens (they do not change), pytorch re-measured in the same runs; H100 cuda from the 2.5.14 H100 tokens; script `docs/b200/triattn_n1/lad.py`, output `work/n1-triattn/_n1_results/lad/ladder_h100.txt`):
ladder violations (cuda slower than triton or pytorch by > 2 %): **0 of 214**. Cases with H100 cuda >= 20 us and B200 / H100 > 0.667: **19** (2 training, 17 inference; none slower than the H100; BidirectionalTriangleAttention 0): training: d_pair 128 x 4 L128 ending 0.754 / starting 0.748; inference: d_pair 256 x 8 L128 0.742 / 0.724, d_pair 384 x 12 L128 0.696 / 0.673, d_pair 128 L256 ending 0.688, L384 ending 0.686 / starting 0.669, L512 ending 0.683, L768 ending 0.671, L1536 0.687 / 0.680, L2048 0.704 / 0.690, d_pair 64 (d_hidden 128) L2048 0.675 / 0.674, d_pair 256 L1536 ending 0.672, d_pair 384 L3968 starting 0.673. Geomean B200 / H100: TriangleAttention train 0.522, eval 0.569; Bidirectional train 0.463, eval 0.477. (AF3 / RF3 / Chai1 / Chai1Confidence were not re-measured: their code and versions are unchanged at 69.)

**Tuned cublasLt GEMMs (`b200_sources/ltmm_sm100.cu`, `b200_bidir.ltmm`, in the prebuild plan: 788 -> 789 artifacts, 0 not built):** the bias-only backward's tall-K contractions (dP = dos^T c, K = 16384 x 4 heads; dWo = dt^T u and dW = dc y, K = L L) go through `lt_mm`: the first call of a shape OUTSIDE a graph capture times the heuristic's 32 candidates on the call's own operands, drops every algorithm that is not deterministic (two runs bitwise equal) or not accurate (max error against the fp32 product > 1e-4 of max), keeps the fastest; in a capture before calibration it falls back to torch (the warm-up always calibrates). `MINIWORLD_BIDIR_B200_LT=0` turns it off. Kernel alone (graph, L2 flushed, us, torch -> tuned, includes the 6 us floor): L512 dP 49.2 -> 45.1, dWo 47.1 -> 41.0, dW 55.3 -> 53.2; L256 dWo 20.5 -> 18.4; L768 no change (the first candidate is the best). Relative error against the fp64 product <= 1.3e-5. Module step (ab2.py, same process, graph, us, off -> on, starting / ending): L128 92.0 -> 86.0 / 92.2 -> 88.1, L256 174.1 -> 165.9 / 196.6 -> 190.5, L384 -4 us, L512 -8 us. Not bitwise equal to the torch GEMMs (the split-K reduction differs; output / dx identical to ~1e-4 relative in the gradients), deterministic run to run.
Measurer (`measure.speed`, 2 runs, ms, base -> final tree, ending / starting): L128 0.1192 -> 0.1009 (-15.4 %) / 0.1177 -> 0.0996 (-15.4 %); L256 0.2361 -> 0.2067 (-12.4 %) / 0.2094 -> 0.1830 (-12.6 %); L384 0.4184 -> 0.3865 (-7.6 %) / 0.3581 -> 0.3269 (-8.7 %); L512 0.6507 -> 0.6050 (-7.0 %) / 0.5727 -> 0.5248 (-8.4 %); L640 0.9876 -> 0.9296 (-5.9 %) / 0.8892 -> 0.8238 (-7.4 %); L768 1.3010 -> 1.2457 (-4.3 %) / 1.1884 -> 1.1357 (-4.4 %). Accuracy (`measure.accuracy`, train, master, L128-768, 6 lengths x 2 orientations, non-zero seeded weights, fp64 gold): TriangleAttention row 188 no RED, 8 YELLOW at 1.03-1.2 x tol (pytorch 7-9e-3 at the same cases), Bidirectional 5 GREEN / 1 YELLOW (9.535e-3, unchanged); three-run bitwise determinism (`det.py`) of output, dx and all six parameter gradients for three configurations: identical. Versions unchanged (training odd): TriangleAttention 71, Bidirectional 5.

**Bidirectional K = 256 kernels (`tail_k<256>` 112-122 us, `gate_bwd_fused_k<256>` 137-147 us at L512), ablation table** (`exp_k256.cu` + `k256.py`: copies of the shared kernels with timing-only macros, graph, L2 flushed, us, includes the 6 us floor; base copy bitwise equal to the shared kernels): 
| variant | tail train | tail eval | gate_bwd |
|---|---|---|---|
| shared kernel | 129.0 | 116.7 | 147.5 |
| copy (= base) | 127.0 | 116.7 | 147.5 |
| no sigmoid (no MUFU) | 120.8 | 108.5 | 149.5 |
| no mma.sync | 112.6 | 98.3 | 133.1 |
| no stores | 112.6 | 112.6 | 96.3 |
| no plane / x loads | 96.3 | 88.1 | 83.3 |
| tanh-form sigmoid (1 MUFU) | 127.0 | 114.7 | 159.7 |
| plane loads hoisted before the GEMM (the lnbwd trick) | 127.0 | 116.7 | 200.7 (register spill) |
Reading: the gate backward moves 603 MB (g, o, dy read; dO, dG written): 100 us at 6 TB/s, 131 us at the measured clone rate (4.6 TB/s) against 141 us measured: within ~8 % of its floor, the hoist spills and the tanh form is slower: nothing to fix. The tail moves 402 MB (67-87 us) and takes 111: even with no loads at all it keeps 82 us (mma.sync 18 + MUFU 8 + the per-chunk barrier chain), so a better issue-ahead (earlier pipeline sweep: 5 stages = no change) cannot reach the floor; closing the 25-40 % needs the contraction on tcgen05 with TMA plane boxes (MN-major A operand) -- a new kernel, not attempted: BidirectionalTriangleAttention already runs 0.46-0.61 of the H100 time on all 12 cases.

### N1 pass 4 (2026-11-18, agent n1-triattn): the 19 remaining H100-1.5x misses -- ending-node tile order built and rejected; core / L128 chain analysis (no source change; versions stay 71 / 5)

**YELLOW note for row 188 after the cublasLt change (honest accounting).** Same measurer harness, same lengths: before the tuned GEMMs (3 lengths L128 / 384 / 768 x 2 orientations) 3 YELLOW of 6 training cases (9.675e-3, 8.778e-3, 9.664e-3), after: the same 3 (9.675e-3, 8.777e-3, 9.650e-3). Over all six lengths x 2 orientations (train, the final tree) 7 of 12 training cases are YELLOW (L128, L256 8.238e-3, L384, L512 8.358e-3 / 8.330e-3, L640 9.455e-3, L768), 5 GREEN, no RED; the pre-change 24-case run (train + eval) had 7 YELLOW as well, all training. All YELLOW are at 1.03-1.2 x tol (the pytorch errors of the same cases are 7.0-9.0e-3; tol = max(err_pytorch, 8e-3)), inside the 5-tol pass band. The members moved (L256, L640 are now YELLOW) because the tuned split-K reduction order changes the fp32 sum (module-level gradient differences ~1e-4 relative); the count did not grow.

**Ending-node tile order (built, A/B, rejected; `rejected/triattn_n1_tile_perm_ending.diff`).** Measured finding: d_pair 128 inference, ending vs starting at L512 (graph, us): front 59.3 vs 56.0, core 210.5 vs 213.9, tail 61.0 vs 55.0 -- the ending node costs ~10 us (3.5 %) more in the front / tail, which is why 8 of the 19 misses are ending cases (L256 .688, L384 .686, L512 .683, L768 .671, L1536 .687, L2048 .704, ...). Hypothesis: the ending node's x / y row tiles are 128 pieces of 256 B at the caller's [b, s, n], 128 rows apart in the order of consecutive CTAs. Lever built: `tperm` (sm100.cuh) makes the persistent front / tail visit tiles with n fastest (148 consecutive CTAs read 148 x 256 B contiguous a row), flag `TA_TPERM`. Result: front 59.2 (unchanged), tail 61.0 -> 67.8: slower; reverted. The cost is the per-row TMA request count of the transposed 4-D box (one 128 B piece a row), not DRAM page locality; a larger piece would need the kernels' tile to hold several n of one s, i.e. a different logical row order of the q / k / v planes (the core is per pair row): a redesign, not attempted.
**Group (a), d_pair 128 inference L384-L2048 (core 210 of 290 us at L512; MUFU ~74 % busy, 81 % of the all-MUFU floor):** nothing new built. The levers named as untried are already in the tree and tuned (POLY_INFER: 2 / 4 of 16 exponential pairs on the FMA pipe, "6 of 16 slower everywhere"; rows a task; HB bias; fp16-S ring built timing-only and rejected, part 14); the ending-node overhead above is the only part of these 10 cases that is not the core.
**Group (b), d_pair 256 / 384 inference L128 (three PDL-chained kernels, 43.8 us of a 46.9-49.0 us graph):** the one-kernel `tri_l128` takes d_pair <= 128 (weights 4 x 256 x 256 x 2 B = 512 KiB do not fit shared memory); a cluster-streamed weight design would remove at most the 2-3 us of the three boundaries (measured boundary 0.7-1.6 us with PDL, a software grid barrier is not cheaper): not built.
**Group (c), training L128 d_pair 128:** nothing from (b) transfers (the fused-training forward kernel itself takes 28.7 us against the 27.6 us chain); the measured experiments of passes 1-2 stand (merged key / query launch, drain pipelining, wide route, SIDE branches: none adopted).

### N1 pass 5 (2026-11-18, agent n1-triattn): the ending-node gap and the L128 d_pair 256 / 384 chain, measured (no source change; versions stay 71 / 5)

**Mechanism of the ending-node gap, measured (`tmab.cu` / `tmab_st.cu` / `tmab2.cu`, 148 CTAs a single TMA-issuing thread, 4-stage ring, same bytes per tile (32 KiB), L2 flushed, one launch each, us):**
| variant (x row tiles, 2048 tiles at L512) | load L512 | store L512 | load L1024 | store L1024 |
|---|---|---|---|---|
| 2-D contiguous box 64 x 128 (starting node) | 18.6 | 15.1 | 56.4 | 47.6 |
| 4-D transposed box 64 x 1 x 128 (ending node today) | 20.5 | 16.4 | 58.4 | 53.2 |
| same, tiles visited n-fastest | 20.5 | 16.4 | 55.5 | 49.6 |
| transposed box 128 x 1 x 128, no swizzle (256 B runs) | 20.5 | 16.4 | 57.5 | 51.2 |
| box 128 x 8 x 16 (8 n x 16 s: 2 KiB contiguous runs) | 20.5 | 15.4 | 57.3 | 48.0 |
| 128 x cp.async.bulk 256 B (1-D) | 55.3 | -- | 198.7 | -- |
So the transposed 128-byte-piece box costs +8-10 % over the contiguous box (1.3-2 us of a 67 MB pass at L512, +5-12 % at L1024); a box with 2 KiB runs recovers it for stores (15.4) and most of it for loads at L1024, and the per-row 1-D bulk copy is 3x worse. Whole-kernel ablation of the same d_pair 128 L512 inference (graph kernel times, us; `XDEFS` timing macros): front ending 59.3 vs starting 56.5; tail ending 59.3 vs 55.2 -- the tail's gap is 1.5 us with no output stores (`TG_NOST`: 54.8 vs 53.3) and ~0 with the LayerNorm arithmetic removed (`TG_NOLN`: 52.1 vs 52.7), i.e. the stores explain ~2.5 of the tail's 4 us and the geometry of the loads ~2 of the front's 2.8: the measured box-shape effect (1.3-2 us a pass) accounts for about half of the 7 us. The candidate fix (a tile of 8 n x 16 s: x / y as ONE 2 KiB-run box, the plane tile as 16 boxes of 16 rows) measured on the tail's input stream (x 32 KiB + o 32 KiB a tile): 33.7 us today vs 32.8 us at L512 and 98.3 vs 98.3 us at L1024: no gain once the 16 small plane boxes are paid, and it changes the front's plane stores the same way. Not implemented (nothing to gain); tile order n-fastest (kernel-level A/B, built) made the tail slower (61.0 -> 67.8 us): `rejected/triattn_n1_tile_perm_ending.diff`.

**L128 d_pair 256 / 384 inference chain (corrected targets and measured timeline).** Targets from the token pairs (H100 cuda 66.0 / 64.9 / 108.8 / 109.4 us): d_pair 256 ending 49.0 -> <= 44.0 us (-5.0), starting 47.0 -> <= 43.3 (-3.7); d_pair 384 ending 75.7 -> <= 72.6 (-3.1), starting 73.6 -> <= 72.9 (-0.7) -- i.e. 1-5 us, not 12-14 us. Timeline (graph 47.1 us, d_pair 256 x 8 starting): `tri_wfront` 0 -> 18.0 us (128 CTAs, one tile each: the four projections stream 512 KiB of weights a CTA), `triattn_fwd` launched at 2.8 us under PDL and computing from ~18 to 32 us (1024 (row, head) units over 148 CTAs: 4 task rounds of ~3.5 us at 2 rows a task; every front CTA finishes together, so a finer front -> core flag hand-off cannot start the core earlier), `tri_wtail` 28.8 -> 42.3 (weights preloaded under the core's tail), graph floor ~5. d_pair 384: front 31.6, core 3.5 -> 52, tail 48.1 -> 68.6. Levers measured by the span: rows a task for the 8 / 12-head core (`_rows_per_task` override, graph us, R = 2 / 3 / 4): d_pair 256 starting 47.1 / 47.1 / 49.2, ending 47.1 / 49.2 / 51.2; d_pair 384 starting 71.7 / 73.7 / 75.8, ending 75.8 / 75.8 / 80.0: two rows a task stays best. The d_pair 256 / 384 ending gap (d_pair 384: front 33.6 vs 31.6, tail 21.6 vs 20.5 us) is the transposed-box effect above. A one-kernel front + core (the `tri_l128` form) needs q | k | v | g of a row in TMEM: 4 x 256 = 1024 columns at d_pair 256 (512 available) and 512 KiB of weights that cannot stay resident: a two-pass head-group design with re-streamed weights, not attempted -- its saving is bounded by the measured front -> core boundary (the core starts computing ~0 us after the front ends under PDL, gap -15 us in the timeline), so the 1-5 us needed cannot come from merging the two launches.

### N2 pass (2026-11-18, agent n2-triattn, GPU 1): per-kernel timelines of the misses; the front's bias projection on the tensor core (ADOPTED, TriangleAttention cuda:b200_sm100 72); tail hand-off, tail o-slot, core flags (built, measured, rejected)

Tools (`docs/b200/triattn_n2/`): `abp2.py` (one variant per process -- two builds of one extension in one process break the tail launch with `invalid argument` --; macros as `-D` to the mod / core extensions; graph median of 60 L2-flushed replays, profiler kernel spans, and the stamp tables below), `bitscan.py` / `cmpscan.py` (bitwise scan of inference outputs between trees), `acc_inf.yaml`, `spd_inf.yaml`, `ladder_n2.txt`, `prebuild_check.txt`. Timing-only stamp builds (default off, same bits): `TG_STAMP` (start / end / tiles of every tail CTA, `tri_tail_stamps`), `TG_TL` / `FR_TL` (per-tile stage stamps of CTA 10 of the tail / front, `tri_tail_tl` / `tri_front_tl`), `TG_NOPDL` (the tail never waits for the core), `FR_NOBP` (no bias projection in the front's LayerNorm). All numbers: exclusive GPU 1, graph, L2 flushed, us, medians of 60 replays, A/B in alternating processes (2-3 rounds).

**Per-kernel spans of the miss cases** (profiler, one replay; the kernels overlap under PDL; start + duration): d_pair 128 L512 ending front 0 + 59.1, core 50.3 + 215.6, tail 247.5 + 64.4 (graph 301); starting 0 + 56.3, 50.3 + 213.0, 247.5 + 55.0 (291); L384 ending 37.1 / 105.4 / 39.4 (161.8); d_pair 256 L1536 ending (8 heads) front 1272 + core 8662 + tail 870 (10 781); d_pair 384 L3968 starting front 18 459 (2.6 TB/s of x + q / k / v / g, tensor floor 10.7 ms), core 228 613, tail 12 646 (3.8 TB/s) (graph 265.95 ms = 0.660 of the H100 in this run, 271.4 ms = 0.673 in the measurer: +-1.5 % between runs of the same build at these lengths: clocks 1867-1890 MHz at the power cap).

**Tail, per CTA stamps (`TG_STAMP`, L512 ending).** The tail CTAs start 0-14.6 us apart (the core's CTAs end one task, 11.5 us, apart: 18.5 or 19 tasks a CTA) and end 42.8-52.7 us after the first start; a CTA's run time is 36-44 us for 13-14 tiles. *Bound for a core -> tail hand-off:* `TG_NOPDL` (the tail never waits for the core; wrong results, timing only): graph 301.1 -> 303.1 (L512 ending, 3 rounds), 161.8 -> 161.8 (L384): no gain, the latest CTA end is the same (54.3 vs 52.7 us). *Arrival-ordered tile counts* (`TG_RANK` k: an atomicInc rank at the start, counts n_q a ramp of k tiles from the first to the last arrival, tiles base_i + q so every round is a prefix of the ranks; with `TG_NOPDL`): latest end 52.0 (k 4) / 51.5 (k 6) against 52.7 (base) and 54.3 (NOPDL, static); L384 28.9 / 31.0 against 29.4: at most 1.2 us (0.4 %) of 301 even with the dependency wait removed, because the late CTAs' time is their own throughput (3.1 vs 2.5 us a tile between CTAs), not their start. Rejected (`rejected/triattn_n2_tail_arrival_rank_and_o_early_release.diff`); a flag-based finer core -> tail hand-off (the other half of the idea (d)) is therefore bounded by the same 1.2 us.

**Tail, per tile (`TG_TL`, CTA 10, L512).** Steady state 2.56 us a tile (starting) / 2.63-2.8 (ending); the HBM floor is 96 KB / (6.0 TB/s / 148) = 2.37 us: the starting tail after the core ends runs at ~92 % of the roofline (201 MB in 35 us = 5.7 TB/s; the kernel's 54 us span includes 12-14 us of waiting for the core). The chain is LN 1.0 + gate step 1.0 + o wait 0.26 + 0.3; x lands 1.3 us after its issue, 4 tiles ahead, so the loads are not the limit. `TG_OEARLY` (the o tile read into registers first, its single slot freed at once so the next o load hides under the arithmetic; same bits): level or slower -- L512 ending 297.0 -> 297.1 / 299.0, starting 286.7 -> 288.7 / 288.8, L768 starting 751.6 -> 759.8 / 757.8, L256 ending 71.7 -> 73.7 -- the o wait was already ~0 in the timeline after it. `TG_NXS4=0` (3 x slots): 297.0 -> 300.8 / 301.0; `FR_WB=2`: 298.8 / 299.0; `FR_DW=8`, `FR_WB=4`: static assertion (shared memory). Rejected.

**Front, per tile (`FR_TL`, CTA 10, L512 ending, before the change).** Steady state 3.7 us a tile against an HBM floor of 3.2 (128 KB a tile): the LayerNorm warps (8 warps, two threads a row, 64 channels each) take 2.6-3.0 us of it (x landed -> y written), the MMA 1.0, the drain ~2.3; the x buffers (2) fill 3.5 us after their request (HBM queueing). In the LN loop 8 of ~22 instructions a channel pair are the bias projection (4 heads x 128 channels a row in FMAs on the CUDA cores, the y values unpacked again). Ablation `FR_NOBP` (no projection, wrong bias): front 59.5 -> 53-54.7, graph L512 ending 301.1 -> 295.1 / 295.7, L1024 1644.5 -> 1624.1.

**Adopted: `FR_TCB` (default 1) -- the q / k / v front's bias projection as one tensor-core GEMM.** For NOUT = 3 (the inference fronts of d_pair 128 x 4 heads and, through `tri_front64`, d_pair 64) the MMA warp issues, in the v half, one N = 16 GEMM y x [Wb hi | Wb lo]^T (K = CX, the bias rows as the existing NOUT = 1 form's B operand `sWb`: rows 0-3 the bf16 heads' weights, rows 4-7 their bf16 residue, so the product has fp32 weights' accuracy) into the g tensor's TMEM columns (3 HD .. 3 HD + 15, unused at NOUT 3); the drain reads them in the v half, adds hi + lo, applies the key mask and writes `bias[b, h, qi, ki]` exactly as the LN warps did. The LN loop loses the projection (STATS rows still written when requested). Front L512 ending 59.5 -> 53.3, starting 55.9 -> 52.2; the steady tile is now 3.0 us (LN 1.0-1.3, at the HBM floor); module (graph, us, `FR_TCB=0` -> 1): L256 ending 73.7 -> 71.7 (starting level 71.7), L384 ending 161.8 -> 157.9, L512 ending 301.1 -> 296.9-297.0 / starting 290.8 -> 286.7 (-1.4 %), L768 ending 785.9 -> 774.1-776.2 / starting 764 -> 758-762, L1536 starting 5087-5103 -> 5013-5048 (-1.1 .. -1.7 %); d_pair 64 L512 level (the front is already at 5 TB/s). The L512 starting front now takes 52.2 us for 268 MB (5.1 TB/s).
*Accuracy / bits (basis):* (1) `measure.accuracy` (`acc_inf.yaml`), TriangleAttention eval, rows 71-76 (d_pair 64 x 2 heads, d_pair 128 x 4, d_pair 64 x 4 (d_hidden 128), d_pair 256 x 8 ..., both orientations), L256 / 384 / 640, master dtype, the harness' seeded non-zero weights, fp64 gold, tol = max(err_pytorch, 8e-3): 36 cuda cases, 36 GREEN, max err / tol 0.49 (3.96e-3 / 8e-3), err_pytorch 3.1-5.3e-3; (2) `bitscan.py` (random non-zero parameters N(0, 0.15) / N(0, 0.5), bf16 inputs, key masks, ragged L130 / 300, B 1-2, both orientations, TriangleAttention d_pair 128 x 4 / 64 (d_hidden 128) x 4 / 256 x 8, AF3 and RF3 classes at d_pair 128: 60 cases) against the unmodified n2-base tree: 13 of 60 bitwise equal, every other case differs by at most one bf16 ulp of the output (max |diff| 0.031 at |y| ~7, 0.0156 at d_pair 64: the bias is rounded to bf16 from a differently ordered fp32 sum); (3) run-to-run: the same call twice, 120 outputs, bitwise equal; (4) `tests/gpu/test_triattn_b200.py` + `test_triattn_b200_capture_bits.py`: 95 passed. AF3 and RF3 at d_pair 128 run the same front (scan: 24 of 24 cases differ from the base by at most one ulp, none bitwise equal), so the base entry `_TriangleAttentionBase.backend_versions["cuda:b200_sm100"]` (AF3 / RF3 / Chai-1 / Chai-1 Confidence) is bumped 69 -> 70 as well; the TriangleAttention entry 71 -> 72; Bidirectional stays 5 (it never reaches this front).

**Core flags at L1024, d_pair 128 x 4, starting (graph, 2 rounds, base 1603.6 / 1611.8):** `TA_NKV` 6: 1628.2 / 1611.8, 4: 1632.3 / 1622.0; `TA_POLY` 1: 1654.8 / 1652.7 (+3 %), 3: 1636.4 / 1607.7; `TA_PVI` 1: 1601.5 / 1627.4: all inside the +-1 % between processes except POLY 1 (worse). The core's exponentials stay at 2 of 16 on the FMA pipe (the packed `ex2.f16x2` / `.bf16x2` issue as two MUFU operations on sm_100a: part 6). A larger structural lever was identified but NOT built: four softmax groups would hide the S hand-back wait (22 % of the group cycle) but need 4 x (BN + 32) <= 512 TMEM columns, i.e. 96-key tiles (exact for L = 384 / 768 / 1536 only, +12.5 % pad at L512): a rewrite of the key-tile loop, the bias tile layout (2 x 64-key halves) and the mask logic.

**Result (measurer, `spd_inf.yaml`, eval rows 71-76, 106 cases, graph, L2 flushed; ladder vs the B200 triton / pytorch tokens, H100 cuda from the 2.5.14 tokens; `ladder_n2.txt`):** ladder violations 0 of 106; cases with H100 >= 20 us and ratio > 0.667: 17 -> 13. Before (N1 pass 4 list) -> after: d_pair 128 L256 ending 0.688 -> 0.670 (71.7 us; the next timer tick, 69.6, would be 0.651), L384 ending 0.686 -> 0.668 (159.5; 157.7 would pass), L384 starting 0.669 -> (passes), L512 ending 0.683 -> 0.669 (294.9; 292.9 would pass), L768 ending 0.671 -> passes, L1536 0.687 / 0.680 -> 0.680 / 0.677, L2048 0.704 / 0.690 -> 0.694 / 0.688; unchanged: d_pair 256 L128 0.744 / 0.725, d_pair 384 L128 ending 0.696, d_pair 64 L2048 0.676 / 0.670, d_pair 384 L3968 starting 0.673. Three of the remaining cases (L256 / L384 / L512 ending) sit 0.3-1.2 us above a 2.048 us timer tick that would pass them.

**Adopted, second change of the pass (same versions 72 / 70): `TG_HALF` (default 1) -- the tail's gate GEMM on y / 2.** The tail's gate step was chain-bound (LN 1.0 + gate 1.0 us a tile, `TG_TL`), 5.5 instructions an element (`0.5 * x`, MUFU tanh, FFMA, FMUL by o, o unpack, pack). The tail now loads the LayerNorm's gamma | beta and the gate bias halved (a power-of-two scale is exact in fp32 and in the bf16 rounding of y), so the gate accumulator is exactly g / 2 and sigmoid(g) = 0.5 + 0.5 tanh(G): one FMUL less an element, bitwise the same output (bitscan: 60 / 60 equal to the build without it; accuracy and tests above are for the final tree). A/B, 2 rounds in alternating processes, `TG_HALF=0` -> 1: L512 ending 297.0 / 297.0 -> 294.9 / 294.9, starting 286.7 / 286.7 -> 284.7 / 284.7 (tail span 54.4 -> 51.6-51.8 us), L768 starting 752.2 / 753.7 -> 751.6 / 749.6, L384 ending 157.7 / 157.7 -> 155.7 / 157.7, L256 ending 71.7 -> 71.7 (level). The same idea one step further was NOT built: the core writing o / 2 so the gate step needs a single FFMA (u = o' + tanh(G) o'), -1 instruction an element more, but the o plane is produced by several cores (head-dim-32 HB core, TA_D16, dh64) and consumed by tailg with different routes (tri_tailg, tri_wtailg, the bias-only plane tail): it needs a runtime `ohalf` flag through every route.

**Not built / not concluded (honest list).** (a) the smem-staged 2 KiB-run store box and the 8 n x 16 s ending-node tile: N1 pass 5 measured the box shapes on the load / store stream (no gain once the 16 plane boxes are paid) and the drain writes the out tile over the x slot (a [32][256 B] re-layout overlaps other warps' unread residual rows); not built here. (b) a finer exp split / packed ex2: measured as flags above (`TA_POLY` 1 / 3 / 5 / 6 and `TA_NKV` 4 / 6 / `TA_PVI` level or worse at L256 / 384 / 512 / 1024), `ex2.f16x2` is two MUFU ops (part 6). (c) K | V ring vs S buffers: `TA_NKV` 4 / 6 level; the TMEM map (3 x 128 S + 3 x 32 O = 480) allows no fourth S buffer at BN = 128 (see the 96-key note above). (d) front -> core / core -> tail overlap: the tail bound above (<= 1.2 us); the front -> core side is bounded by the front's tile imbalance (one 3 us tile on 124 of 148 CTAs): not built. (e) d_pair 384 L3968 / 256 L1536: per-kernel spans above; the wide front (tensor floor 58 % at d384) and tail (3.8 TB/s, per-tile weight re-stream of the 2 x 192-column items) were not touched: the 3 % of the module they could give needs a pass over the 2 000-line wide sources. Training L128 d_pair 128 and L128 d_pair 256 / 384 inference: not attempted beyond the N1 timelines.

**Adopted, third change of the pass (same versions): `TG_HF2` (default 1, needs `TG_HALF`) -- the gate step's o / 2 in one packed bf16 multiply.** With G = g / 2 the product u = o sigmoid(g) = (o / 2) (1 + tanh G): the o pair is halved by one `mul.rn.bf16x2` (exact), unpacked once, and u = fma(h, tanh G, h): 8 instead of 9 instructions a channel pair (the old FMUL by o is gone; the first version of this idea, o / 2 written by the cores, would have removed one more instruction but needs a runtime flag through every o producer / tail route and was not built). A/B, 3 rounds in alternating processes, `TG_HF2=0` -> 1 (both with `TG_HALF`): L384 ending 157.7 / 157.7 / 157.7 -> 155.6 / 155.6 / 155.7; L512 ending 294.9 x 3 -> 292.9 x 3; L768 starting 751.6 / 749.6 / 749.6 -> 747.6 / 747.5 / 748.3; L512 starting 284.7 -> 284.7 / 284.6 / 284.6 (level). Not bitwise the same as `TG_HALF` alone (the fp32 rounding of u differs in the last bit before the bf16 pack): 48 of 60 scan cases differ from the previous build by at most one bf16 ulp of the output (0.0156 at |y| 7); against the n2-base tree every d_pair 128 / 64 case differs by <= 1 ulp (0.031), none bitwise equal, d_pair 256 (wide tail, `tri_wtail`, untouched) 12 / 12 equal. Accuracy of the final tree (`measure.accuracy`, `acc_inf.yaml`): 36 / 36 GREEN; tests 95 passed; `_prebuild --arch sm100 --check`: 0 todo (the plan's two triattn extensions `triattn_b200_mod_v23_tt11` and `triattn_b200_wide_v45_tt7` rebuilt for the new header; non-sm90 absences listed in `triattn_n2/prebuild_check.txt`: `b200_triattn.py::_core128`, `b200_bidir.py::bo` not reached, `b200_sources/triattn_dh128_sm100.cu` not compiled, unchanged by this pass).

**Final measurer (all three changes; `spd_inf.yaml`, TriangleAttention eval rows 71-76, 106 cases, graph, L2 flushed; `ladder_n2.txt`; the "Result" paragraph above is the state after `FR_TCB` only):** ladder violations 0 of 106 (cuda slower than the B200 triton / pytorch tokens by > 2 %); cases with H100 >= 20 us and B200 / H100 > 0.667: 17 (N1 pass 4) -> 12: d_pair 256 L128 0.743 / 0.724, d_pair 384 L128 ending 0.696, d_pair 128 L2048 ending 0.694 / starting 0.677, L1536 ending 0.683 / starting 0.677, d_pair 64 L2048 0.672 / 0.674, d_pair 384 L3968 starting 0.671, d_pair 128 L256 ending 0.669 (71.6 us; 71.3 needed, the timer tick is 2.048 us), d_pair 256 L1536 starting 0.6675. Passed by this pass: d_pair 384 L128 starting, d_pair 128 L384 ending and starting, L512 ending (292.9 us against 293.7), L768 ending. The 2 training misses (L128 d_pair 128, 0.754 / 0.748) were not touched. Run-to-run spread of the same build at L1536 / L2048 / L3968 is 1-2 % (the first measurer run of this tree gave L2048 0.694 / 0.688, L1536 0.680 / 0.677, d_pair 384 L3968 0.673; before the tail change: L512 ending 0.669), so the cases within 1 % of the line flip between runs.

### N7 pass (2026-11-19, agent n7-triattn): the d_pair 128 inference core's logit without the FFMA2 (TA_QS) and 5 of 16 exponentials on the FMA pipe -- core -6 % at L >= 1024, 13 of the 20 gap-measurer H100-1.5x misses gone (TriangleAttention 72 -> 74, base 70 -> 72)

**What bounds the loop (measured first, L1024 d_pair 128 x 4, core span of the profiler, same session, exclusive GPU).** base 1406.8 us; `TA_NOEX` (no exponential at all, one FFMA2 in its place) 1244.0 (-11.6 %: the MUFU is not the whole loop); `TA_FNOLOOP` (no arithmetic: only the S load / P store) 797.7 (-43 %): the logit / sum / pack instructions of the softmax warps are what the core waits for; `TA_POLY=0` +6 % (module 1583 -> 1628). So fewer FMA-pipe instructions per key pair AND fewer MUFU per pair both pay, and the poly share has to be re-tuned when the loop gets cheaper.
**SASS count (cuobjdump of the r3 inference core, one 16-key chunk = 8 key pairs; the compiler interleaves chunks, so +-3):** before (TA_POLY 2: one pair of 8 on the polynomial): MUFU 14, FHADD 16, FFMA2 11 (8 logit + 3 poly), FADD2 11 (8 row sum + 3 poly), FADD 2 (the poly pair's two -rr negations), FMNMX 2, IMAD/LEA 2, F2FP 8, LDS 2, LDTM 1, STTM 1: 66-72 instructions. After (QS, POLY 5 = 2-3 pairs of 8): MUFU 11-12, FHADD 16, FFMA2 8 (only the poly pairs: 4 each), FADD2 12, FMNMX 4-6, F2FP 8, LDS 2: 60-66. Packed ops were already in the loop (FFMA2 / FADD2 since part 9; FHADD.BF16 for the bias; `ex2.f16x2` is two MUFU, part 6; F2FP packs two); the instruction count itself fell only ~8 %, the saving is mostly that the MUFU share fell 14 -> 11 a chunk while the FMA pipe (FFMA2 and FADD2 are 2-clock instructions: the isolated mix of part 9 reads 7.1 clk a pair for FFMA2 + 2 FHADD + FADD2 + F2FP) got the logit FFMA2 back.
**The rewrite (`triattn_sm100.cu` `TA_QS`, `b200_triattn.py` `QS_INFER`, MINIWORLD_TRIATTN_B200_QS=0 turns it off):**
1. *q pre-scaled by scale * log2 e.* The host scales the q rows of the packed weight once (cached, `param_forms` `triattn_b200_pack_qs` / `triattn_b200_wide_l2e_qs`; one rounding of the scaled weight, as the unscaled one before), the core's `scl` is 1 and the logit of a key pair is `x = S + bias`: two FHADD.BF16, no FFMA2 and no `-m` operand (`logit2q`). Training keeps the unscaled build; 16-channel heads (TA_D16, the row sum comes from the PV GEMM and there is no overflow guard) and the head-dim-64 / 128 cores are not changed.
2. *Offset 0 for the typical row.* The offset used to be the maximum of the row's first 32 keys. Now a row whose first-32-key maximum lies inside +-`TA_QT` (24, log2 domain) keeps offset 0; every other row keeps the maximum as before. The first key tile (`j == 0`, which fixes the offset) and every tile of a warp with one raised row run the old loop (`chunk_loop(BC<false>)`, FFMA2 with scl 1 and `-m`); a later tile whose 32 rows all have offset 0 runs the FHADD-only loop (`BC<true>`, warp-uniform vote). The overflow guard / PSHIFT / recompute path is unchanged (a late spike > 64 still raises the offset there; tested below).
3. *`TA_PSUB` (default 1; bit-identical).* The polynomial exponential's `x - round(x)` is one FFMA2 (`fma(rr, -1, x)`, the same bits) instead of two negating FADDs + a FADD2: -2 instructions a polynomial pair.
4. *`POLY_INFER(rows, qs)`:* 5 of 16 exponential pairs on the FMA pipe for the QS builds at 3 rows a task (4 at 2 rows: not re-measured), 2 for the others as before. Sweep (L1024, module graph p25, us, alternating processes, QS builds, PSUB on): POLY 2 1553-1561, 3 1554-1555, 4 1497-1503, 5 1493-1499, 6 1497-1501, 7 1499, 8 1558-1561 (without PSUB: POLY 5 1513).
5. *`QS_MIN_L` = 512:* QS is taken above 512 padded tokens only. With 3-4 key tiles the first-tile old loop is 25-33 % of the tiles: L512 level (module 284.7 / 284.7 / 284.6 -> 284.9 / 286.7 / 284.7, core 213.3 / 214.8 / 210.8 -> 207.0 / 211.1 / 208.6) and L384 slower in the gap measurer (155.4 / 159.5 -> 157.7 / 161.7), so L <= 512 runs the old build (bitwise the old outputs).

**Measured (alternating processes, 3 rounds, graph, L2 flushed; module = the whole TriangleAttention graph, core = the profiler span of `triattn_fwd_sm100`):**
| case | before | after |
| L1024 d128 starting, GPU 1 exclusive: core us | 1372.0 / 1387.4 / 1349.2 / 1440.7 | 1295.3 / 1297.5 / 1281.5 / 1273.6 |
| same, module p25 us | 1581.0 / 1577.0 / 1579.0 | 1497.0 / 1493.0 / 1499.1 (-5.1 %) |
| L2048 d128 starting, GPU 3 exclusive: core us | 10023.9 / 10020.1 / 10023.9 | 8876.9 / 9719.6 / 9694.1 (two clock levels, SW power cap) |
| same, module p25 us | 11462.7 / 11430.9 / 11432.0 | 10855.0 / 10845.2 / 10853.4 (-5.1 %) |
| L512 (QS not taken since QS_MIN_L) | core 213.3 / 214.8 / 210.8 | level by construction |
Gap-measurer run (`spdgap.py`, GAP 0.15, every requirement case, `cmpg.py`): H100-1.5x misses 20 -> 7 (the 7: L128 training x2 0.754 / 0.748, L128 d_pair 256 x2 0.741 / 0.723, L128 d_pair 384 0.695, L256 d_pair 128 ending 0.669 (71.6 us, 71.3 needed), L2048 d_pair 128 ending 0.667 (11310.0 us, 11307.5 needed): none was touched). The cases the pass was for (b200 / h100, old -> new): d_pair 128 L1536 0.692 / 0.679 -> 0.650 / < 0.64, L2048 0.705 / 0.693 -> 0.667 / 0.663, d_pair 64 x 4 x 32 L2048 0.680 / 0.678 -> 0.646 / < 0.64, d_pair 256 L1536 0.682 -> 0.636, L2048 0.677 / 0.677 -> 0.642 / 0.643, d_pair 384 L3968 0.681 -> passed, L768 d_pair 128 ending 0.659 -> passed, L384 / L512 unchanged (0.659 / 0.660, 0.645 / 0.664). Ladder (cuda < triton < pytorch, 2 %): 0 violations; slower than H100: 0.
**Accuracy basis:** (1) `measure.accuracy` (`acc_inf.yaml`, rows 71-76, L256 / 384 / 640, master dtype, the harness' seeded non-zero weights, fp64 gold, tol = max(err_pytorch, 8e-3)): 36 / 36 cuda cases GREEN, max err 3.96e-3 (0.50 of tol; before 3.96e-3); (2) `bitscan.py` / `cmpscan.py` against n7-base (60 cases: TriangleAttention d_pair 128 / 64 (d_hidden 128) / 256, AF3, RF3, both orientations, masks, ragged L): 50 bitwise equal (every L <= 512 case), the 10 L640 cases differ by at most 0.0625 (one bf16 ulp at the 0.03125 level on |y| ~ 7); repeated calls bitwise equal; larger tolerance cases (`qs_chk.py`, `qs_chk2.py`: d_pair 128 L640 / 1030, 64, 256, 384 at L768; QS vs old max 0.125 at |y| 25, 0.375 at |y| 55, 1.0 at |y| 100: 1-2 bf16 ulp); (3) large logits (`qs_big.py`: Wq / Wk x3, `qs_big2.py`: bias x25, std ~140 in the log2 domain): no NaN, QS within 0.19 of the old core at bias x25 (QS and old both ~7.4 off the fp32 reference there) and within the old core's own error at Wq / Wk x3; (4) tests: `tests/gpu/test_triattn_b200.py` + `test_triattn_b200_capture_bits.py` 95 passed (training capture bits unchanged: QS is inference only); `tests/cpu` -k version/triangle/triattn/prebuild/judge 591 passed; `_prebuild --arch sm100 --check` after building the four changed HB cores: 0 not built, the same 160 "not reached" non-sm90 call sites as `triattn_n2/prebuild_check.txt` (rc 1 as before); (5) AF3 / RF3 / Chai-1 run the same front and core (scan: 60 of 60 cases through the same functions), so the base `_TriangleAttentionBase` entry is bumped 70 -> 72 with TriangleAttention 72 -> 74; Bidirectional stays 5.
**Rejected (`rejected/triattn_n7_qs_pure_offset0.diff`):** QS with offset 0 for every row from the first tile (provisional start, no first-32-key maximum): core faster still (module L1024 1584 -> ~1531, -3.1 % at POLY 2) but at large logits (Wq / Wk x3, logits ~9x) it was wrong (11.8 off the fp32 reference, NaN on the ending node, old core 2.6 / 2.1): the overflow / provisional rescue path of the first tile was meant for the rare row. Poly share: 0 (+6 %), 2 / 3 (-1.5 %), 8 (+4 %) are worse than 4-7; `ex2.f16x2` was already excluded (part 6); the polynomial clamp stays one FMNMX an element (no packed max). Row sum on the tensor core (`TA_ONES`) was rejected in part 12.
**Not built / unfinished:** the first key tile still runs the old loop (L <= 512 has no gain: a speculative offset-0 first tile needs the rescue path proven on every row); the d_pair 64 x 4 x 16 core (TA_D16: row sum from the PV GEMM, no overflow guard, so offset 0 needs a guard first), the head-dim-64 / 128 cores and the 2-row builds (L <= 256: POLY 4 not re-measured with QS, QS not taken there) are unchanged; L128 / L256 and the ending-node front / tail overheads are not touched. Tools (`docs/b200/triattn_n7/`): `abp7.py` (abp2 with the QS build), `ab2.sh`, `qs_chk.py`, `qs_chk2.py`, `qs_big.py`, `qs_big2.py`, `gchk.py` (CUDA-graph replay == eager, bitwise, d_pair 128 / 256 at L640).

### N9 pass (2026-11-19, agent n9-triattn-b): which cores really run QS, the 16-channel-head and head-dim-64 cores ported, and a TF32 trap in the bias-only training backward (TriangleAttention 74 -> 76, `_TriangleAttentionBase` 72 -> 74, Bidirectional 5 -> 6)

**The misses on the current tree first** (`spdgap.py`, GAP 0.15, `cmpg.py`, before any change, `triattn_n9` tools `tab.py`): 7 H100-1.5x misses, not the stale 20 / 13 list: training L128 d_pair 128 x2 (0.754 / 0.748), L128 d_pair 256 x2 (0.742 / 0.724), L128 d_pair 384 (0.696), L256 d_pair 128 ending (0.669, the d_pair 128 core's agent), and **BidirectionalTriangleAttention training L768 (0.734: 3272 us against 1400 at L640)**. The cases the N7 pass attacked all pass (d_pair 64 x 4 x 32 L2048 0.624 / 0.626, d_pair 256 L1536 / L2048 0.622 / 0.628, d_pair 384 L3968 0.571 / 0.619, d_pair 128 L2048 0.659 / 0.643).

**1. Which core serves which width, and whether it runs QS** (`cuobjdump -sass` of the `triattn_fwd*` kernel of each cached cubin, `triattn_n9/hist.sh`; whole-kernel counts, the kernel holds every loop copy: the first-key-tile loop, the FHADD-only loop, the 3 rows, the guard):

| width (requirement rows) | core | QS before | MUFU | FHADD | FFMA2 | FADD2 | F2FP | total |
| d_pair 128 x 4, d_pair 64 x 4 (d_hidden 128), d_pair 64 x 2, d_pair 256 x 8, d_pair 384 x 12 (head dim 32) | `triattn_sm100.cu` HB core (`_core(rows, hb, qs)`, `wide_forward` / `fused_forward`; one core for all five widths, only the grid differs) | yes, above 512 padded tokens (N7) | 212 | 416 | 216 | 196 | 156 | 3544 (old form, POLY 2: 132 / 176 / 136 / 84 / 92, 2688) |
| d_pair 64 x 4 x 16 (OpenFold3 / AF3 / RF3 trunk) | same source, `TA_D16` (row sum from the PV GEMM) | **no** (`qs_for` excluded it; `_core` flag `not d16`) | 126 | 160 | 112 | 16 (+ 84 FFMA) | 80 | 2744 |
| head dim 64 (Chai-1 / RF3 template, IntelliFold; no requirement row) | `triattn_dh64_sm100.cu` HB build (its own copy of the N2 loop) | **no** | 133 | 176 | 144 | 92 | 140 | 3088 |
| head dim 96 / 128 (no requirement row) | `triattn_dh128_sm100.cu`, the training build also serves inference (no HB build, bias unpacked: FMUL 324, no FHADD) | no | 84 | 0 | 64 | 32 | 176 | 2512 |

So the "wide paths follow the same rule" claim of N7 was right for d_pair 64 x 2 / 256 / 384 (they are head dim 32 and share the core and the pack: confirmed in the table and by the A/B below), wrong for the 16-channel-head core, the head-dim-64 core and the head-dim-128 core.

**2. Ported / extended.**
(a) *16-channel-head core (`TA_D16`)*: `TA_QS` now builds with `TA_D16` (and `TA_D16C`): its loop has no FADD2 (the sum is the PV GEMM's channel 16), so the logit FFMA2 was one of six instructions of a pair and the QS saving is larger than at d_pair 128. The weight pack scales q by `head_dim^-0.5 * log2 e` (hd 16: `pack_wide_weights`, the compact pack copies the scaled rows), `qs_for` takes head dim 16 above `QS_MIN_L16` = 256 (MINIWORLD_TRIATTN_B200_QS_MIN_L16), `D16QS_POLY` = 4 of 16 (`D16QS_POLY2` for two rows is unused). The D16 core has no per-tile overflow guard; offset 0 is covered by its existing drain check (a row whose PV-summed l left [2^-100, 3e38) is recomputed exactly from q / k in global memory; q is the pre-scaled one and `scl` 1, as in the d_pair 128 QS build).
(b) *head-dim-64 core*: the same four edits as N7 in `triattn_dh64_sm100.cu` (`TA_QS`: `logit2q`, the +-24 offset-0 rule, the FHADD-only loop for later tiles of a warp whose rows all have offset 0, `scl` 1 on the host), `_core64(hb, poly, qs)`, `POLY64QS_INFER` = 2, `QS_MIN_L64` = 512 (MINIWORLD_TRIATTN_B200_QS_MIN_L64). The non-QS HB build is now a lambda like the d32 one: bitwise the same outputs (60 cases of `bitscan9.py`, L <= 512 equal).
(c) *head dim 128*: not ported (no HB build exists: the inference path runs the training core; QS needs the bias pre-scaled in the front and the HB loop first; no requirement row has 96 / 128-channel heads).
(d) *`ltmm_sm100.cu` TF32 guard (the Bidirectional miss)*: see 4.
(e) `cuda/__init__.py`: the prebuild plan lists the three new builds (`triattn_b200_core_r3_hb_d16{,c}_p4_qs_v3_tt8`, `triattn_b200_dh64_hb_p2_qs_v1`).

**SASS after (whole kernel):** d16c QS r3 p4: MUFU 206, FHADD 288, FFMA2 208, FADD2 64, FFMA 84, F2FP 144, total 3312 (before 126 / 160 / 112 / 16 / 84 / 80, 2744: the FHADD-only loop is 2 FHADD + 1 MUFU-or-poly + 1 F2FP per pair, no FFMA2); dh64 QS p2: 245 / 304 / 168 / 180 / 204, total 3640 (before 3088). Instructions of the steady loop per exponential pair fall by the logit FFMA2 (1 of 4 on the FMA pipe for D16: 6 -> 5 issue slots a pair, the poly share moves 2 -> 4 of 16 pairs).

**3. Measured (alternating processes, `abp.py` = N7's `abp7.py`, GPU 1 exclusive, graph, L2 flushed, 3 rounds; core = profiler span of `triattn_fwd_sm100`, module = the graph; the SW power cap quantises to 2 us ticks):**
| case | QS off | QS on |
| d_pair 64 x 4 x 16 L768 core / module (3 rounds) | 547.8 / 548.5 / 546.9; 622.6 / 622.6 / 622.6 | 523.3 / 523.3 / 523.5; 598.2 / 598.2 / 599.6 (POLY 2) |
| same, POLY sweep (module): 0 / 1: 632.8, 2 / 3: 598.0, **4: 579.6**, 5: 581.6, 6: 585.3 | | core 504.5 at POLY 4 (-7.9 %), module 622.6 -> 579.6 (-6.9 %) |
| d_pair 64 x 4 x 16 L640 / L512 / L384 module (2 rounds) | 385.0 / 385.1; 227.3 / 227.3; 116.8 / 116.7 | 364.5 / 364.5; 219.1 / 219.1; 112.6 / 112.6 (-5.3 / -3.6 / -3.5 %) |
| same, L256 (QS_MIN_L16 0, 2 rows: POLY2 4 / 2 / 6) | 53.2 / 53.2 | 55.3 / 53.2 / 57.3: not taken (QS above 256) |
| head dim 64 (d_pair 128 x 2 x 64) L1024 core / module | 903.8 / 900.3 / 922.6; 1292.5 / 1284.1 / 1284.3 | 877.0 / 881.6 / 884.5; 1281.4 / 1269.5 / 1267.7 (core -2.7 .. -4.1 %, module -1.1 %); POLY64QS 0: 916.7, 4: 908.5, 6: 899.4 |
| head dim 64 L640 | core 255.0 / 257.6, module 419.9 / 419.8 | 253.4 / 254.8, 419.7 / 419.6 (level) |
Gap-measurer (`spdgap.py`, every requirement case; `tab0.txt` before, `tab3.txt` after, ratios b200 / h100): d_pair 64 x 4 x 16 eval starting 384 / 512 / 640 / 768: 118.7 / 227.2 / 386.9 / 622.5 -> 114.6 / 219.0 / 364.4 / 579.5 us (0.546 / 0.560 / 0.565 / 0.579 -> 0.527 / 0.540 / 0.532 / 0.540); ending 120.6 / 231.3 / 391.0 / 629.6 -> 116.6 / 223.1 / 368.6 / 587.6. No other eval row moved by more than noise (+-3 %). H100-1.5x misses 7 -> **6** (only the Bidirectional L768 one went; the 6 left are L128 training d_pair 128 x2, L128 d_pair 256 x2, L128 d_pair 384, L256 d_pair 128 ending), ladder (cuda < triton < pytorch, 2 %): 0 violations, slower than H100: 0.

**4. The Bidirectional (and bias-only TriangleAttention) training L768 miss: a TF32 trap, not a kernel.** In the measurer the cuda backend runs with `torch.backends.cuda.matmul.allow_tf32 = True`; `ltmm_sm100.cu`'s first call of a shape (the module's warm-up) times the cublasLt candidates and keeps only those whose result is within 1e-4 of the largest value of `torch::mm(a.float(), b.float())`, which with TF32 on is itself a TF32 product (error ~1e-3 at K = L L). At L768 (K = 589 824, the dW GEMM `[520, L L] x [L L, 128]`) every candidate was rejected, `chosen = -1`, and the call fell back to a bf16 -> fp32 cast (843 us) plus a TF32 cutlass GEMM (570 us): training step 3.26 ms instead of 1.87 ms (the same case run alone with `allow_tf32 = False`, or by hand in a graph with the module's flags, was 1.87 ms; `triattn_n9/tf32_probe.py` runs the step with `speed.make_step` and shows the two kernels). Fix: `Tf32Off` in `lt_mm` (TF32 off for the reference and the fallbacks, restored on exit; extension name `triattn_b200_ltmm_v2`). Measured (gap measurer, training, us): Bidirectional d_pair 128 L768 3272 -> 1910 (0.734 -> 0.43), TriangleAttention `use_self_attention=False` d_pair 128 L768 2006 / 1946 -> 1245 / 1136 (0.579 / 0.623 -> 0.36); every other training row within +-1 % (a candidate was already accepted there). Accuracy (`measure.accuracy` rows 142 / 188 training, L256 / 640 / 768, `acc_bi.yaml`, new tree vs n9-base): the same verdicts and errors within 1e-4 (L768 r188 6.615e-3 vs 6.525e-3, r188 9.664e-3 vs 9.650e-3; the YELLOW weight-gradient cases are the old ones); bitwise: step twice == step twice, graph replay == eager for the Bidirectional and the bias-only module at L512 / L768 with TF32 on (`bias_only_det.py`).

**Accuracy basis of the QS ports:** (1) `measure.accuracy` rows 71-76 (`acc_inf.yaml`, L256 / 384 / 640, fp64 gold, tol = max(err_pytorch, 8e-3)): 36 / 36 cuda GREEN, max err 3.96e-3 (0.50 of tol; unchanged); (2) `bitscan9.py` vs n9-base (144 cases: d_pair 128 / 64 x 4 / 64 x 2 / 256 / 384 / 128 x 2 x 64, AF3, RF3, both orientations, masks, ragged L, repeat-bitwise): 126 equal, 18 differ by 0.03125 (one bf16 ulp at |y| 6-7: d_pair 64 x 4 at L >= 384 padded, head dim 64 at L >= 640); every case repeat-bitwise; (3) QS vs old vs fp32 pytorch reference (`qs_chk16.py 64 4`, `qs_chk64.py`): typical weights L384-1030 and masked L700: QS-vs-old 0.0625 (hd 16, = ulp) / 0.125 (hd 64, ulp at 16-32), QS-vs-reference equal to the old core's; Wq / Wk x3: hd 16 QS 0.4375 = old 0.4375, hd 64 4.375 / 3.33 vs old 4.375 / 3.63; bias x25: hd 16 1.156 vs 1.156, hd 64 5.23 / 6.40 vs 5.22 / 6.37; no NaN anywhere (at Wq / Wk x8 the hd-64 old core is already NaN and QS equals it; hd 16 x8: QS-vs-old 8.6, both ~8 off the reference: the logits are ~9x the bf16 range); (4) graph == eager (`gchk9.py`, d_pair 64 x 4 / 64 x 2 / 256 at L640, both orientations); (5) tests: `tests/gpu/test_triattn_b200.py` + `test_triattn_b200_capture_bits.py` 95 passed (training bits unchanged), `tests/cpu -k "version or triangle or triattn or prebuild or judge"` 591 passed; `_prebuild --arch sm100 --check`: 0 not built (804 artifacts), the same 160 "not reached" non-sm90 call sites as before, rc 1 as before.

**Rejected / measured level (no code kept):**
- QS_MIN_L lower for the head-dim-32 core at d_pair 256 / 64 x 2 (L512: 585.8 / 585.7 -> 583.7 / 583.7 and 157.7 -> 155.7 / 157.5; L384: 311.3 -> 317.4 for d_pair 256 (+2 %), 86.8 / 86.2 -> 86.1 / 87.8): stays 512.
- QS for the 16-channel-head core at L256 (two rows a task): 53.2 -> 55.3 / 53.2 / 57.3 for POLY2 4 / 2 / 6: not taken (the threshold is 256).
- POLY share of the head-dim-32 QS build at d_pair 256 / 384 L1024 (4 / 5 / 6 of 16, 2 rounds, core span varies 2476-2700 us at equal POLY with the power-cap clock; best minimum 5): stays 5. POLY64QS 0 / 4 / 6: worse than 2.
- Core rows a task at L128 of the wide chain (d_pair 256 / 384): `rejected/triattn_n9_rows_exp.diff` (3 rows: 49.2 / 75.8, 4 rows: 51.2 / 79.9 us against 47.1 / 75.8 at 2).

**Not done / what is left.** L128 d_pair 256 / 384 (the three wide misses): the chain is front 18.7 / 33.2 us (single-tile 2-CTA clusters, 128 of 148 SMs, drain-paced, weight stream from L2), core 12.8 / ~16 us after the front (2 rows a task, 3.5 waves of latency-bound tasks), tail 11.2 / ~18 us after the core, +4 us graph ends; 47.1 / 75.7 us against 43.3-44.0 / 72.5 needed. Candidates (none free): the gate-from-x tail (`tri_wtailg` exists for C = 64 only: -25 % front N, longer tail chain), a fused L128 one-CTA kernel like `tri_l128` for C = 256 / 384 (smem: the 256 x 1040 / 384 x 1552 weights do not fit one CTA), the front's two-CTA W stream with TMA multicast already halves the L2 read. The 16-channel-head QS first tile still runs the old loop (L <= 256 gains nothing); QS for the head-dim-128 core needs an HB build first. Tools in `docs/b200/triattn_n9/`: `qs_chk16.py`, `qs_chk64.py`, `bitscan9.py`, `gchk9.py`, `hist.sh` (SASS opcode counts), `tab.py` (gap jsonl -> table with H100 ratios), `tf32_probe.py`, `bias_only_det.py`, `acc_bi.yaml`.
