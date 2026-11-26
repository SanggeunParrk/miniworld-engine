# Triangle multiplicative update on B200 (sm100)

Scope: the `cuda` backend of `TriangleMultiplication`, `TriangleMultiplicationRF3` and `BidirectionalTriangleMultiplication`
(`modules/triangle.py`) on a capability (10, 0) device (card key `b200_sm100`, `backend_versions["cuda:b200_sm100"] = 6` on all
three). The kernels are the B200 research branch's (`origin/b200/ladder-final`, 2026-11-02): tcgen05 / TMEM / TMA hand-CUDA in
`kernels/trimul_inproj/cuda/b200_sources/`, driven by `b200_infer.py` (inference) and `b200_train.py` (training), dispatched by
`kernels/integrations/trimul_b200.py`. The kernel-level design (tiles, warp roles, the batched-sample layout, chunking, the measured
coefficients) is the branch's `docs/gpus/b200/trimul/trimul.md`; this page covers the 2.5.11 integration and its state.

```
x = LN_in(pair)
a = sigmoid(x W_ag) o (x W_a),  b = sigmoid(x W_bg) o (x W_b)        (masked by m_i & m_j)
t = sum_k a_ik b_jk  (outgoing) | sum_k a_ki b_kj  (incoming)        (bidirectional: first d_hidden channels outgoing, second incoming)
y = pair + drop_row(sigmoid(x W_g) o (LN_out(t) W_o))
```

## Route

`_trimul_cuda_route` sends `cap == (10, 0)` to `trimul_b200.route(module, pair, mask, dropout_p, bidirectional=...)`:

| grad | route | kernels | contract (else a refusal reason) |
|---|---|---|---|
| off | `b200_infer` | `k1w` (front, every D; D >= 384 with `k1w_stats`, D <= 256 computes the LayerNorm statistics itself; in `b200_infer.fold_cell` -- D128 with the k3w output, D256 / D384 -- the LN_in-folded FOLD form, no `k1w_stats`), cuBLAS contraction, `k3g` (D <= 128 output) / `k3w` + `wide_fold_prep` + `wide_ln_stats` (D >= 256 output, and D128 in `K3W_D128_CELLS`; from L384 on, and at L128 on the tile-grouped schedule, k3w computes the LN_out statistics itself, no `wide_ln_stats`) | bf16 contiguous square pair, bool `[B, L]` mask, LN eps 1e-5, D in 64 / 128 / 256 / 384 / 512, hidden = D (or 2 D: one direction at D64 / D128, bidirectional at D64), any L (L % 16 != 0 padded on the host), a live row dropout through `k3g`'s training epilogue needs L % 128 at D <= 128 |
| on | `b200_train` | forward `k1w` + contraction + `k3g` (saving) / `k3w`; backward D64 `b1s` + `b7m`, D128 `b1g` + `b7g`, D256-512 `wide_gate_bwd` / `wide_lnout_bwd` / `k1wb` / `wide_lnin_bwd` + cuBLAS | as above; D64 / D128 L % 128 == 0 and L <= 10240, D256-512 L % 16 == 0 |

Every requirement case of the three classes is inside this contract: the route refuses none of them (rows' lengths are multiples of
128, or 16 for OpenDDE's 2976 / 3968). The row-dropout scale is the reference's torch draw (`torch.rand` of `functional.dropout_scale(pair, p, 1)`,
the compare / cast / scale in one kernel: `b200_infer.dropout_scale`); on one sample at D64 / D128 RoseTTAFold3's `1 / n` is formed in the
weight-forms launch from the token mask (`b200_train.forms_rf3_op`, read at every replay), otherwise
RoseTTAFold3's `1 / n` is folded into `W_a` (`_LeftScaled`, as on the other cards; one launch per sample when masks differ).

`b1s`, `b1g` and `b7g` spin on ring flags across CTAs: all three are launched cooperatively (`cudaLaunchAttributeCooperative`; `b1s` together with the programmatic-dependent attribute) with one CTA per SM, so every CTA is resident or the launch fails: never a partial-residency hang
(two such kernels on two streams cannot each hold part of the device).

## Changes to shared sources

`b200_sources/wide_aux.cu` and `wide_bwd.cu` are also read by the H100 (`h100_b200style_aux.py` compiles each file's text up to its
first `}  // namespace wide` / `wbwd`). That prefix is kept byte-identical to 2.5.11; the branch's additions (`trimul_forms`,
`grad_reduce`, the D64 8-plane `gate_bwd_fold` / `lnin_bwd64`) are appended in a reopened namespace that only the B200 extension
compiles (round 3 added `drop_scale` and the RoseTTAFold3 factor of `trimul_forms` / `grad_reduce` there). The B200 host `wide_gate_bwd` now uses the template's fixed-order per-block partials (`gate_bwd_kernel<D, true>` +
`colsum_kernel`, the H100 port's deterministic form) instead of an atomic flush: D256-512 training gradients are bit-identical run to
run (the branch's D64 / D128 backward already was). Nothing the H100 / A100 / RTX 5090 builds or runs changed.

## Accuracy (B200, 2026-11-16)

`measure.accuracy`, every row of the three classes, every length, backends `pytorch` / `cuda` / `triton`, dtype `master`. Records:
`$W/b2511/agents/trimul/_out/acc_<Class>.jsonl` on the B200 box.

| class | backend | eval GREEN | train GREEN | train YELLOW | RED | unsupported | skipped |
|---|---|---|---|---|---|---|---|
| TriangleMultiplication | cuda | 102 | 66 | 6 | 0 | 0 | 4 |
| TriangleMultiplication | triton | 102 | 53 | 19 | 0 | 0 | 4 |
| BidirectionalTriangleMultiplication | cuda | 30 | 26 | 3 | 0 | 0 | 1 |
| BidirectionalTriangleMultiplication | triton | 30 | 25 | 4 | 0 | 0 | 1 |
| TriangleMultiplicationRF3 | cuda | 36 | 18 | 6 | 0 | 0 | 0 |
| TriangleMultiplicationRF3 | triton | 36 | 16 | 8 | 0 | 0 | 0 |

Every eval cell is GREEN. The 15 cuda YELLOW cells are training weight / LayerNorm-affine gradients at 1.02-1.54 tol (max err
1.24e-2, RF3 D64 incoming L640 `grad:to_left.weight`), mostly D64; the bf16 operand rounding of a sum over L^2 tokens, the same
cells and magnitudes as the released H100 tokens (17 YELLOW, up to 1.22e-2) and fewer than Triton on this card (31). Skipped =
the reference ran out of memory (TriangleMultiplication D384 eval L2976 / L3968, both directions: still skipped when re-run alone
on the GPU; bidirectional D256 L768, skipped on the shared GPU, re-run exclusively: eval and train GREEN, `_out/acc_skip.jsonl`); the
cuda route serves them.

## Speed (B200, 2026-11-16)

`measure.speed` (CUDA graph, L2 flushed, median of 50, exclusive GPU 1), every row and length of the three classes, backends
`pytorch` / `cuda` / `triton`; `anthropic` / `cueq` from the baseline records on GPU 0 (`$W/b2511/out_base/speed.*.jsonl`, same card
model, other device). Records: `$W/b2511/agents/trimul/_out/spd_<Class>.jsonl`.

| class | cells | cuda vs triton | cuda vs pytorch | cuda slower than triton |
|---|---|---|---|---|
| TriangleMultiplication | 178 | 1.60-6.97x | 3.61-23.2x | 0 |
| BidirectionalTriangleMultiplication | 60 | 2.27-7.94x | 3.77-15.7x | 0 |
| TriangleMultiplicationRF3 | 60 | 1.63-3.47x | 3.84-20.6x | 0 |

Selected cells (ms; anthropic is inference only):

| case | cuda | triton | pytorch | anthropic | cueq |
|---|---|---|---|---|---|
| D64 out, eval L512 | 0.086 | 0.213 | 1.216 | 0.164 | 0.233 |
| D64 out, eval L2048 | 1.662 | 2.872 | 28.02 | 2.659 | 3.502 |
| D64 hidden 128, train L768 | 0.872 | 2.623 | 12.76 | - | 3.836 |
| D128 out, eval L1024 | 0.614 | 1.981 | 12.78 | 1.205 | 1.221 |
| D128 out, train L768 | 1.187 | 3.406 | 13.89 | - | 4.470 |
| D256 out, eval L2048 | 9.466 | 25.03 | 99.63 | 12.92 | 21.30 |
| D384 in, eval L3968 (OpenDDE refiner) | 75.99 | 237.3 | 509.5 | 140.9 | 185.0 |
| D512 out, train L768 | 10.02 | 26.56 | 60.43 | - | 29.42 |
| bidirectional D128, train L768 | 1.875 | 6.354 | 18.55 | - | - |
| bidirectional D512, train L768 | 16.50 | 54.57 | 86.94 | - | - |

cuda is the fastest backend in every measured cell (also against anthropic and cueq wherever they serve).

## B200 vs H100 (round 3, 2026-11-17)

Target: on every requirement case both cards' `cuda` serve, B200 median <= H100 median / 1.5 (ratio b200 / h100 <= 0.667; graph
timing, L2 flushed). H100 medians: the released `h100_sm90` speed tokens. "Before": the `b200_sm100` cuda speed tokens
(`cuda:b200_sm100` 2). "After": `cuda:b200_sm100` 3, `measure.speed` settings on an exclusive development GPU, every case once and
L128-384 / L1024-1536 twice (the cards are shared; the lower median of a repeated case is taken). Records on the B200 box:
`$W/b2511/agents/r3-trimul/_out/spd_v3.jsonl`, summary `_out/summ_v3.txt` (`_tools/summ.py`).

| class | cases | misses of 1.5x before -> after | slower than H100 before -> after | worst ratio before -> after | geomean speedup vs H100 |
|---|---|---|---|---|---|
| TriangleMultiplication | 178 | 59 -> 41 | 10 -> 2 (1.007, 1.004: ties) | 1.208 -> 1.007 | 1.52x -> 1.63x |
| TriangleMultiplicationRF3 | 60 | 24 -> 14 | 5 -> 0 | 1.066 -> 0.793 | 1.47x -> 1.65x |
| BidirectionalTriangleMultiplication | 60 | 11 -> 9 | 1 -> 0 | 1.055 -> 0.979 | 1.69x -> 1.76x |

The ladder holds on every case (cuda / triton <= 0.78, cuda / pytorch <= 0.27 against the `b200_sm100` tokens; no triton > pytorch).

What changed (all inside the sm_100a extension and its drivers; the H100-read prefixes of `wide_aux.cu` / `wide_bwd.cu` are
byte-identical):

| change | where | effect (ms, before -> after) |
|---|---|---|
| Inference graphs are served the eager weight forms (`param_forms(..., capture_reads_eager=True)`, the H100 inference forms' contract): no `trimul_forms` / cast launch per replay | `trimul_b200._inference_params`, `b200_train.kernel_forms` | D64 eval L128 0.0244 -> 0.0204-0.0224, D256 eval L128 0.0552 -> 0.0429 (4 bf16 cast launches + 1 fold launch fewer) |
| RoseTTAFold3 (one sample, D64 / D128): the length factor `bf16(1 / n)` is counted from the token mask and multiplied into `W_l` inside the forms launch (`forms_rf3_op`; `grad_reduce` scales dW_l by it) instead of ~7 torch launches + uncached forms; bitwise the `_LeftScaled` result (test) | `wide_aux.cu` `forms_kernel`, `b200_train`, `TriangleMultiplicationRF3._forward_cuda` | RF3 D64 eval L128 0.0389 -> 0.0244, D128 eval L256 0.0716 -> 0.0551, D64 train L128 0.0934 -> 0.0729 |
| Row-dropout scale: compare / cast / divide of the reference draw as one kernel, bitwise the reference (test) | `wide_aux.cu` `drop_scale_kernel`, `b200_infer.dropout_scale` | two launches fewer per training call (~3 us) |
| `k3w` (D256+ output): 8 epilogue warps (two per TMEM lane quarter), a tile-grouped schedule (a CTA runs both channel items of a token tile, the second reads its K-blocks from L2), the epilogue arithmetic on packed f32x2, a one-MUFU `tanh.approx` gate (`K3W_SIG` 2, k1w / k3g's form) | `k3w.cu` | k3w at D256 L2048 2.89 -> 1.97-2.03 ms; D256 eval L768 0.928 -> 0.791 |
| D256 `k1w` computes the LayerNorm statistics in its transform warps (the 4 K-blocks of a row are resident in the x ring; `K1W_INSTAT` 256) and writes them for `k3w` / the backward: no `k1w_stats` pass over x | `k1w.cu`, `b200_infer`, `b200_train` | D256 eval L768 0.791 -> 0.726, L2048 8.31 -> 7.80; bidirectional D256 eval L768 1.249 -> 1.189 |

Rejected (measured, reverted): a `tanh.approx` gate in `b7m` (D64 train L512 0.3095 -> 0.3084 / 0.3090 -> 0.3094: noise; b7m is not
MUFU-bound); D256 x / weight ring splits 7 / 4 and 6 / 5 for the in-kernel statistics (no better than 4 / 6); the in-kernel statistics at
D384 (x ring 6, weight ring 5: eval L256 0.164 -> 0.156, L768 1.37 -> 1.32, L2048 13.7 -> 13.2, but L128 0.0553 -> 0.0572 -- the only
D384 length below the target -- so not taken; D512 has no room for its 8 K-blocks next to a useful weight ring); `k3g` knobs on D64 / D128
(`K3_NWP` 2, `K3_EPW` 32, `K3_LNU` 8: <= 1 % of the module; `K3_NTH` 3 / `K3_NWP` 4 do not fit shared memory). 2.5.12's H100 items do
not carry over: the measured cuBLASLt table has no counterpart (the B200 contraction runs cuBLAS's `nvjet` kernels at 15-20 us for
64 x 512^3, about the HBM floor of its operands), "port K3" *is* this page's `k3w`, and the no-mask launch cells are an H100 launch-table
matter.

### Round 3, second pass (2026-11-17, `cuda:b200_sm100` 4)

What a replay costs at small L (measured, `torch.profiler` device times under graph replay with the L2 flush, D64 inference L128):
`measure.speed` median = about 4.8 us + the sum of the kernels' device times, whatever the launch count (one empty launch 6.0 us = 4.8 +
1.2-1.5; three launches of 15.6 us in total: 20.4 us; the timer steps in 2.05 us). An empty 128-CTA / 512-thread / 230 KB kernel takes
1.47 us, the TMEM allocation another 0.5 us; `k1w` on one tile per CTA is 6.0 us, cuBLAS 2.9, `k3g` 6.7. So the small-L cases are not
launch-bound: they are bound by the one-tile latency of each kernel (load -> LayerNorm -> product -> epilogue -> store, about 4 us of
it in `k1w` and 4.8 us in `k3g`), which the B200 does not run faster than the H100.

**The one-launch forward was built, measured and not taken.** `kf.cu` (kept outside the tree: `_tools/kf_experiment/` in the round's
worktree, `git stash` "kf one-launch forward experiment"): one cooperative grid running `k1w`'s body, the contraction as in-kernel
tcgen05 products (outgoing: both operands K-major; incoming: the same plane boxes read MN-major; fp32 accumulators -> bf16 rows of t)
and `k3g`'s body, with a grid barrier (a global arrival counter) between the phases. Output: bitwise the three-launch path on all
20 D64 / D128 shapes at L128 / L256 (the in-kernel products equal cuBLAS's bit for bit). Time: slower on 15 of 20, equal on 4, one timer step faster on 1 (bidirectional D64 L128 26.5 -> 24.5 us)
(D64 L128 20.35 -> 22.4 us, D128 L256 51.1 -> 57.2 us). Per phase (kernel time, D64 L128): empty kernel with TMEM 1.95 us (2.85 with
the counter's two atomics per CTA: 128 CTAs on one word cost 0.9 us), front 5.0, contraction 4.1 (cuBLAS: 1.4 above its launch),
output 6.7, two barriers 2.5; 19.1 us in one launch against 15.6 us in three. Fusing removes two kernel start-ups (about 2 us each) and
pays the barriers, a slower contraction and phases that run 1-2.6 us slower inside the larger kernel than alone. Even with free
barriers and cuBLAS's contraction rate the one launch would be 11.9 us of kernel time (16.7 us measured-equivalent, ratio 0.83 at D64
L128); the 1.5x target there (13.5 us) needs 8.7 us, i.e. one-tile latencies about three times shorter than `k1w` / `k3g` have.

Role profile of the D64 training kernels at L512 (`PW` clock probes, share of the kernel's time a role waits):

| kernel | finding | change | kernel time (us) |
|---|---|---|---|
| `k3g` (C64 H64, saving) | the MMA thread waits 40 % for the epilogue to free the single accumulator pair and 39 % for the LayerNorm warps; the epilogue waits 10 % for the residual x to be reloaded after the in-place xn save; warps 4-7 (LN_in, then half of LN_out) are busy 76 % | accumulators double buffered (64 columns each at C64); xn saved from two staging tiles, x not reloaded; at hidden == pair width LN_out runs on warps 8-11 only | 46.8 -> 39.0 (inference L512: 35.5 -> 33.8; C128 H128 inference: 60.3 -> 57.5; C64 H128 and C128 H256 keep the split: +5 % / +1 % with it) |
| `b7m` | the epilogue waits 21 % for its x / dy tiles, loaded only after the previous tile's dx store | x / dy tiles double buffered (two source chunks: shared memory allows it) | 81.3 -> 73.9 (L384: 49.4 -> 44.8) |
| `b1s` | front epilogue waits 48 % for its ring slot, LN CTAs wait 50 % for the front's flag | slots checked / returned by their own threads as early as possible: **slower** (79.6 -> 83.6), reverted; front CTAs 80 / 100 / 110 instead of 88: 94.0 / 88.2 / 102.6, the 88 stays | 79 (unchanged) |
| `k3g` two CTAs per SM (C64 H64: 256 TMEM columns, <= 64 registers, 113 KB each, grid 296) | "two tiles in flight" without re-pipelining | runs correctly, **slower** (training L512 39.1 -> 43.7, inference 33.8 -> 39.4), reverted (`_tools/k3g_dual_experiment/`) | -- |

The last row settles what the D64 training kernels are bound by: not hand-overs (double buffering bought 5-17 % per kernel and its
waits are gone) and not a single tile in flight (two CTAs on an SM are slower than one), but the SM's elementwise instruction
throughput -- LayerNorm, gate and epilogue arithmetic on 128 x 64 values per tile, about 6.4 k warp-instructions per `k3g` tile.
A further step needs less elementwise work per token (e.g. `k3w`'s LayerNorm-folded weights on raw operands at D64, one statistics
pass instead of three across `k1w` / `k3g` / `b7m`), not another pipeline shape.

State after this pass (`_out/spd_v4*.jsonl`, two sweeps of every case on an exclusive development GPU; summary of the lowest median
per case over these and the first pass's records: `_out/summ_v34.txt`; of this pass's records alone: `_out/summ_v4.txt`):

| class | cases | misses of 1.5x: first pass -> now | slower than H100 | worst ratio | geomean speedup vs H100 |
|---|---|---|---|---|---|
| TriangleMultiplication | 178 | 41 -> 38 | 2 -> 1 (1.007: a tie) | 1.007 | 1.63x -> 1.65x |
| TriangleMultiplicationRF3 | 60 | 14 -> 13 | 0 | 0.793 | 1.65x -> 1.68x |
| BidirectionalTriangleMultiplication | 60 | 9 -> 9 | 0 | 0.978 | 1.76x -> 1.77x |

No requirement case of the three classes has an H100 median below 20 us (the smallest is 20.2 us), so the 20 us exemption removes
none. The ladder holds on every case (cuda / triton <= 0.78, cuda / pytorch <= 0.27). D64 training moved by 5-6 % (L384 0.1968 ->
0.1865 ms, L512 0.3135 -> 0.2961, L640 0.4650 -> 0.4456, L768 0.6223 -> 0.5899): TriangleMultiplication L768 now passes (0.659),
L384-640 are at 0.70-0.72 and RoseTTAFold3 L384-768 at 0.67-0.68. The three inference cases the first pass put down to sweep bias,
each measured in its own process: RoseTTAFold3 D64 incoming L1024 0.3214 ms (0.602), D256 outgoing L1536 3.978 ms (0.634), D64
hidden 128 outgoing L1024 0.5211 ms (0.596) -- all pass; in a sweep the first two still come out at 0.667-0.672.

The 60 remaining: L128 28 and L256 14 (one-tile latency, see above: not reachable by fusing launches), D64 training L384-768 16
(elementwise throughput of `b1s` / `b7m` / `k3g` / `k1w`; needed: -7 % at L384, -6 % at L512 / L640 for TriangleMultiplication, -1 to
-2 % for RoseTTAFold3), and 2 inference cases that pass when measured alone.

Accuracy of `cuda:b200_sm100` 4 (`measure.accuracy`, every row, L128-768 eval + train, `_out/acc_v4s.jsonl`): eval 126 / 126 GREEN; training
119 GREEN, 7 YELLOW, 0 RED -- the same 7 YELLOW cells as before (no cell changed its light; the bidirectional D384 L768 cell skipped in
the first pass measured GREEN). Run-to-run determinism: the tests (20 + 5 repeated backwards, bitwise).

### Round 3, third pass (2026-11-17, `cuda:b200_sm100` 5)

**Where one tile's time goes** (in-kernel `clock64` timelines, `TM_TL` probes compiled out by default, `_tools/tl.py`; D64 eval L128,
medians over the CTAs, us after the CTA's first instruction; the device time adds ~1.5 us of launch / CTA start-up before and ~0.3 after):

| kernel | set-up done (barriers, tensor-map prefetch, TMEM) | operands landed | LayerNorm done | MMA issued | epilogue stores done | device time |
|---|---|---|---|---|---|---|
| `k1w` | 1.65 | x 2.08 | 2.62 | 3.05 | 4.16 | 6.0 |
| `k3g` | 0.90 | x 1.35, t 1.60 | xn 2.56, LN_out 2.69 | 2.97 | 3.88 | 6.5 |

The set-up alone is 0.9-1.65 us and a cold TMA box lands ~0.45-1.3 us after its issue; the LayerNorm / MMA / epilogue work of one tile is
~2 us per kernel. Only the set-up and the loads that do not depend on the previous kernel can be hidden -- by a programmatic dependent
launch (PDL), which this route had nowhere.

What changed (`cuda:b200_sm100` 5; every kernel calls `griddepcontrol.wait` before it reads anything the previous kernel writes or
writes anything it reads, so completion stays transitive):

| change | where | effect |
|---|---|---|
| `k3g` launched as a programmatic dependent of the contraction while every CTA has one tile (L128 at D <= 128): barrier init, TMEM, the x / weight loads and LN_in run during the cuBLAS GEMM, the t loads after the wait | `k3g.cu` (`K3_PDL`, `K3_PDL_TILES` 1) | D64 eval L128 20.4 -> 18.4 us (kernel start 0.35 us before the GEMM's end instead of 0.4 us after it) -- when the instantiated graph honours the edge, see below |
| one-tile CTAs: warps 4-7 (idle after the LayerNorms) run the epilogue of the upper half of the channels (`K3_LAT`, the predecessor's edit); needed for the PDL gain above | `k3g.cu` | with PDL: D64 eval L128 20.2 -> 18.4 us; without PDL no change |
| `k1w`: barrier init, tensor-map prefetch and TMEM allocation before the wait, the first ring fills right after it (`K1W_PDL`, `K1W_EARLY`); x keeps the L2 (no evict-first) while it is <= 24 MB (`K1W_XKEEP`, for `k3g` / the backward) | `k1w.cu` | training: k1w starts 0.5 us before the forms launch ends; D128 eval L256 53.1 -> 51.1 us (`K1W_EARLY` + `K1W_XKEEP` against neither) |
| training chain: `drop_scale`, `forms`, `grad_reduce` wait + trigger at entry; `b7m` waits only before the dA loads (weights, xn / x / dy / dGout first); `b1s` cooperative + programmatic, waits after its Wp' formation; `k3g` triggers at entry so `b1s` CTAs launch as `k3g`'s retire | `wide_aux.cu` (B200-only part), `b7m.cu`, `b1s.cu`, `k3g.cu` (`WA_PDL`, `B7M_PDL`, `B1S_PDL`, `K3_TRIG`) | D64 training L384 0.1880 -> 0.1855 ms, RF3 L384 0.1865 -> 0.1845, L128 0.0739 -> 0.0717 |

Rejected (measured, reverted / off): `k3w` as a programmatic dependent of `wide_ln_stats` (D256 / D384 eval L128, D256 L768: no
change); `b7m` triggering `grad_reduce` at entry (`B7M_TRIG`: the small grid sits on the SMs, training +2-4 us); `k3g`'s first loads
issued by thread 0 at entry (`K3_EARLY`: no gain at L128 with PDL, D64 eval L256 32.7 -> 34.7 us); PDL for multi-tile `k3g`
(`K3_PDL_TILES` 1000: no change at L256-512).

**The programmatic edge is bimodal per graph instantiation.** The same captured step gives 18.4 us on most captures and 20.3 us on some
(`_tools/pdlprobe2.py`: six captures of one step, 2 of 6 at 20.3, whether the stream is new or the graph is kept); in a long sweep
process the later captures are mostly the slow kind (`_tools/pdlprobe.py`; the graph's DOT dump is the same in both). The k1w ->
cuBLAS gap moves with it (0.19 vs 0.65 us), so it is the driver's scheduling of the instantiated chain, not the kernels. The measurer's
tokens can therefore come out at either value for D64 eval L128 (and D128 eval L256 with PDL: 49.0 / 51.1 us).

State after this pass (`_out/spd_v5{a,b}.jsonl`: two exclusive sweeps of every case; the cases still above 0.667 re-timed in one warmed
process, `_out/rem_v5.jsonl` (`_tools/remeas.py`); lowest median per case, summary `_out/summ_v5r.txt`):

| class | cases | above 0.667: second pass -> now | slower than H100 | worst ratio | geomean speedup vs H100 |
|---|---|---|---|---|---|
| TriangleMultiplication | 178 | 38 -> 37 | 1 -> 0 | 1.007 -> 0.976 | 1.65x -> 1.65x |
| TriangleMultiplicationRF3 | 60 | 13 -> 10 | 0 | 0.793 -> 0.781 | 1.68x -> 1.70x |
| BidirectionalTriangleMultiplication | 60 | 9 -> 9 | 0 | 0.978 -> 0.906 | 1.77x -> 1.77x |

The ladder holds on every case (cuda / triton <= 0.78, cuda / pytorch <= 0.27). Accuracy (`_out/acc_v5s.jsonl`, every row, L128-768
eval + train): 126 / 126 eval GREEN, training 119 GREEN + 7 YELLOW, no cell changed its light against `cuda:b200_sm100` 4. Tests: 74
passed (new: `test_b200_pdl_chain_graph_matches_eager`, ten replays of the PDL chain bitwise the eager step, eval and train).

The 56 remaining: **L128 (28) and L256 (14)** are one-tile latency (table above): the D64 eval L128 target is 13.5 us measured, i.e.
~8.7 us of kernel time for three kernels whose single-tile critical paths are ~4 us each plus ~1.5 us start-up; PDL hides about 2 us of
it when the edge is honoured. Reaching 1.5x there needs the per-tile path itself about halved (a 64-token / split-channel tile over
twice the SMs at L128, which needs M=64 tcgen05 tiles and a re-derived k1w / k3g, or the cuBLAS contraction replaced by an in-house
GEMM that triggers its dependents early, so both edges overlap) -- not started. **D64 training L384-640 (14: TriangleMultiplication 7, RF3 6, bidirectional 1; 0.670-0.710; every L768 case
now passes)** and the D128 / D256 / D384 L128-256 training cells are elementwise-throughput bound as the second
pass found; PDL took 1-2 us of the ~5-15 us each still needs. Five cases are within 1 % of the target (D128 eval L256 0.668 / 0.670,
RF3 train L512 0.670, hidden-128 train L512 0.671, RF3 train L640 0.673): one honoured programmatic edge or a quiet card flips them.

### Round 3, fourth pass (2026-11-17, `cuda:b200_sm100` 6)

**D256-512 training: the torch glue around the kernels.** An ordered trace of one graph replay (`_tools/proford.py`, D256 incoming
training L128, 53 launches, 0.250 ms) showed ~75 us that were not the kernels: six bf16 casts of the fp32 master weights (14 us), the
packed-W1 / W_g concatenation of the input-gradient GEMM (four cat + two copy launches, 13.5 us), ten `.to(float32, copy=True)` copies
of the parameter gradients (17.5 us), four elementwise launches forming dW_p (8.5 us), and the gate backward's deterministic column sum
(`colsum_kernel` over 2220 rows of per-block partials: 17.5 us at L128, plus two fills for S1 / S2).

| change | where | effect |
|---|---|---|
| `wide_forms`: the six casts and the packed [W1; W_g] operand ([4R + D, D], so `dxn = dcat [W1; W_g]` stays one GEMM) in one launch (`forms_kernel` with a `skip` offset: the D64 / D128 layout without k3g's permuted W_p); bitwise torch's `.to(bf16)` and `torch.cat(_pack_w1(...), W_g)` (test) | `wide_aux.cu`, `b200_train._kernel_leaves` (a 12th leaf) | 6 + 6 launches -> 1 |
| `wide_grad_pack`: the ten gradients in parameter order into one fp32 buffer (the D64 / D128 `grad_reduce` layout, split by the autograd function), dW_p = g_o (mm - r1) + r0 b_o with the torch expression's roundings; bitwise the replaced statements (test). The backward ops return one buffer (renamed `trimul_b200_train_bwd_g1` / `_into_g1`) | `wide_aux.cu`, `b200_train._backward_body` | 14 launches -> 1 |
| gate backward on 888 blocks (`WGB_BLOCKS`, a multiple of 3 so the thread count stays a multiple of D / 8) and every wide column sum on a 1024-thread kernel (`colsum32_kernel`, fixed order); S1 / S2 adjacent, one memset | `wide_bwd.cu` host functions (B200-only part), `wide_aux.cu` | gate backward + column sum 39 -> ~10 us at L128 |

D256 training L128 0.251 -> 0.177 ms (ratio 0.785 -> 0.55), D384 0.334 -> 0.261 (0.724 -> 0.565), bidirectional D256 0.316 -> 0.241
(0.699 -> 0.53), D256 L256 0.463, D512 L128 0.347. Gate grid A/B (D256 / D384 L128, D256 / D512 L768; ms): 444 blocks 0.1722 / 0.2684 /
3.599 / 9.439, 888 0.1753 / 0.2715 / 3.576 / 9.357, 1332 0.1793 / 0.2652 / 3.647 / 9.428, 2220 (the old grid, with the new column sum)
0.1865 / 0.2828 / 3.572 / 9.457: 888 kept.

**D256-512 inference: `wide_ln_stats` -> `k3w` as programmatic dependents.** The H100-read `ln_stats_kernel` cannot trigger its
dependents (byte-identical prefix), which is why the third pass saw no gain from a programmatic `k3w`; a B200-only copy
(`ln_stats_pdl_kernel`, the same arithmetic) waits for the contraction and triggers at entry, and `k3w` (the third pass's rejected
`K3W_PDL` diff, re-applied) waits only before its epilogue reads mu_o / rs_o -- with the cached inference fold only (the training forward
has `wide_fold_prep` between them). Two processes each: D384 eval L128 0.0552 -> 0.0532 / 0.0537 ms (passes), bidirectional D256 L128
0.0531 -> 0.0511, D256 L256 0.0900 -> 0.0880, D256 L128 0.0389 -> 0.0386, D384 L768 1.372 / 1.384 -> 1.369 / 1.373.

**D64 / D128 training: `grad_reduce` on 16 warps per block** (`GR_W`; each warp sums every 16th partial row of the block's 128
outputs, fixed order): D64 L128 0.0698 -> 0.0677 ms, L384 0.1854 / 0.1864 -> 0.1845, D128 L128 0.1128 -> 0.1102 / 0.1128,
bidirectional D128 L128 0.1497 -> 0.1455 / 0.1502; 32 warps slower (0.0688, 0.1148, 0.1557).

**cuBLASLt algorithm table for the contraction (2.5.12's H100 technique).** `_tools/ltsweep.py` timed every heuristic candidate
(cuBLASLt returns 8 per shape here, all algorithm 66 with split-K 1) against `torch.bmm` for L128-768 x 64 / 128 planes and the three
operand layouts the forward and backward use (CUPTI kernel time, L2 flushed; `_out/lt2.jsonl`): no gain at L128 (0.98-1.00) or L768
(1.00); L256 x 64 0.93-0.99, L384 x 64 0.95-0.96, L512 x 64 0.93-0.95, L640 x 128 0.91 (a^T b) / 0.96 (a b); every best candidate is
bitwise the default's output. In the module (`b200_infer.LT_B200`, served through `_h100_bmm.contract`, no workspace; two processes
each): D64 L512 training 0.2945 / 0.2952 -> 0.2909 / 0.2911 ms, RF3 0.2918 -> 0.2880, D64 L512 eval 0.0818 -> 0.0798 -- kept; the L384
keys made D64 L384 training 0.1844 -> 0.1866 and bidirectional L384 0.2684 -> 0.2705, the L256 key D64 L256 training 0.1138 -> 0.1148
-- not listed. The table keys L512 x 64 (both layouts) and L640 x 128 (a^T b).

Rejected this pass (measured, reverted or left off):

| experiment | numbers (graph median, ms) |
|---|---|
| b1s / b1g front-CTA count at L128 (`_front_ctas` 48 / 64 / 72 / 96 / 104 / 112 vs the table's 90 / 84) | D128 L128 0.1087 (table) vs 0.1087-0.1148; bidirectional D128 0.1481 vs 0.1496-0.1567; D64 0.0718 vs 0.0708-0.0738: the table stays |
| the H100's one-launch Philox dropout scale (`h100_sources/pack.cu` built for sm_100a: torch.rand + the scale kernel in one launch, the graph's generator-state fills as one-thread writes; bitwise the reference draw, eager and over graph replays) | D64 L128 training 0.0677 -> 0.0678 / 0.0683, L384 0.1826 -> 0.1845, D128 L128 0.1128 -> 0.1126 / 0.1132, bidirectional D128 0.1496 -> 0.1508, D256 0.1880 -> 0.1870: no gain (the draw and its fills overlap the forms launch already); not taken |
| a mask-free call (the H100's no-mask cells: k1w / k1wb skip the token-mask reads with mask = None) | D64 / D128 / bidirectional D64 / D256 eval L128 and D64 training L128: identical medians with and without the key mask (`_tools/nomask.py`); no requirement case is unmasked anyway (`key_mask` pads the last L / 8 tokens) |
| `GR_W` 32 | see above |
| `WGB_BLOCKS` 444 / 1332 / 2220 | see above |

#### H100 technique audit (TriMul, `cuda:h100_sm90` path vs this route)

| H100 technique | where on the H100 | B200 | evidence |
|---|---|---|---|
| fused front (LN_in + four gated projections + pair mask -> channel-major planes, one launch) | `_h100_infer_kernel` K1, `tmn_kernels.cuh` | used | `k1w` (tcgen05), LN statistics in-kernel up to D256 |
| fused output (LN_out + W_o + sigmoid gate + residual, one launch) | K3 | used | `k3g` (D <= 128), `k3w` (D >= 256) |
| port K3: LN affines folded into W_p / W_g, statistics applied in the epilogue (2.5.12) | `h100_b200style.rowpad_k3`, `mw_aux_fold` | used | `k3w` + `wide_fold_prep` (cached in inference) at D >= 256; at D <= 128 `k3g` normalises in-kernel (no fold) |
| PDL chains (griddepcontrol in K1 / K3) | packaged K1 / K3 only | used, wider | k3g after the contraction, the whole D64 / D128 training chain (third pass); `wide_ln_stats` -> `k3w` (this pass). The edge is bimodal per graph instantiation (third pass) |
| measured cuBLASLt algorithm table | `_h100_bmm.LT_CONFIGS`, `_h100_bmm_lt_bidir.LT_BIDIR` | used (this pass) | `LT_B200`: L512 x 64 / L640 x 128 keys; L128 / L256 / L384 / L768 measured, no module gain |
| zero-workspace cuBLAS (no stream-K semaphore memset node) | `_h100_bmm.bmm_out` | not applicable | the B200 traces of every case show no memset node next to the contraction (cuBLAS picks non-split nvjet kernels); the LT path runs without workspace anyway |
| no-mask launch cells | `configs/h100_sm90.json` MASK=0 cells | not applicable (measured) | mask = None: same medians; every requirement case carries a padded key mask |
| per-cell launch tables (tile shapes by L / B) | `TILE_TABLE`, JSON `choose` | used as analytic rules | front CTAs / b7g source groups / k3g PDL tile count / ln_stats tokens per lane by L; the L128 front-CTA sweep (this pass) found no better cell |
| row-padded port for long one-direction eval | `trimul_h100._rowpad_small` | not applicable | `k1w` / `k3g` serve any L (L % 16 host pad); the long one-direction cases pass (0.53-0.66) |
| inference weight forms cached per parameter version, read by captures | `_capture.param_forms(..., capture_reads_eager=True)` | used | `kernel_forms(eval_reads_eager=True)`, the wide fold cache |
| one-launch training weight prep | `fused_training_prepare`, `cast_weights` + `pack_into` | used, fewer launches | D64 / D128 `trimul_forms`; D256-512 `wide_forms` (this pass; the H100 wide path still casts + packs in two) |
| B samples in one launch / chunking | `trimul_h100._update_inference_body` | used | sample-major D64 planes, `_plan` chunks |
| D64 training B1 / 4 bmm / B2 / fin (deterministic fixed-order partials) | `h100_d64_b200gate` | used | `b1s` / bmm / `b7m` / `grad_reduce` (16 warps this pass) |
| D128 in-kernel partial reduction (B1 `PART_ONLY=2` grid barrier, B7 in-kernel reduce: no `fin` launch) | `h100_training` | not taken (measured) | ported for D64 (`b7m` grid barrier + in-kernel reduce, training pass): 12-14 us slower at L128-384 than the separate PDL-launched `grad_reduce`; D128 not built on that evidence |
| D64 B1 front + LN_out backward in one kernel (`h100_d64_b200gate` B1) | `h100_d64_b200gate` | used (training passes 1 and 2) | `b1u_kernel` at hidden 64 one direction, `b1v_kernel` at contraction width 128 (hidden 128, bidirectional; pass 2): no ring / flags / cooperative launch at any D64 shape |
| contraction-gradient GEMMs (four cuBLAS bmm, owned scratch at L768) | `h100_d64_b200gate._bmm`, `_h100_pack.owned_bmm` | B200 ahead (pass 2) | `cgb_kernel`: every product in one tcgen05 launch at L <= 256 (operand transposes as descriptor modes); cuBLAS from L384 (measured) |
| PDL across the training backward (H100: packaged K1 / K3 only) | - | used, wider (pass 2) | b1u / b1v / cgb trigger their dependents at entry: -2 us per case at L128 / L256; b1g / b7g as programmatic dependents measured neutral at L128 and 3-6 us slower at L256 (not taken) |
| D256+ backward glue | `h100_b200style.backward` (~25-30 launches incl. ~10 dtype copies) | B200 ahead | this pass: casts / cat / gradient copies / dW_p elementwise folded into two launches, the column sums re-gridded |
| Philox dropout scale in one launch + graph RNG fills as one-thread writes | `h100_triattn.drop_scale` (`pack.cu`) | not taken (measured) | built for sm_100a, bitwise the reference; no gain on this card (table above) |
| saved intermediates (D64: stats, recompute xn; D128: xn) | `h100_d64_b200gate`, `h100_training` | different trade | B200 D64 / D128 `k3g` saves xn through staging tiles (second pass); recomputing it in `b1s` was not measured |

State after this pass (`_out/spd_v6{a,b}.jsonl`: two exclusive sweeps of every case; the cases still above 0.667 re-timed twice in one
warmed process, `_out/rem_v6.jsonl`; lowest median per case, summary `_out/summ_v6r.txt`):

| class | cases | above 0.667: third pass -> now (eval / train) | slower than H100 | worst ratio eval / train | geomean speedup vs H100 |
|---|---|---|---|---|---|
| TriangleMultiplication | 178 | 37 (16 / 21) -> 32 (16 / 16) | 0 | 0.975 / 0.947 | 1.65x -> 1.68x |
| TriangleMultiplicationRF3 | 60 | 10 (0 / 10) -> 8 (0 / 8) | 0 | - / 0.771 | 1.70x -> 1.69x |
| BidirectionalTriangleMultiplication | 60 | 9 (4 / 5) -> 8 (4 / 4) | 0 | 0.902 / 0.866 | 1.77x -> 1.80x |

The ladder holds on every case (cuda / triton <= 0.78, cuda / pytorch <= 0.24). Accuracy (`_out/acc_v6s.jsonl`, every row, L128-768
eval + train): 126 / 126 eval GREEN, training 119 GREEN + 7 YELLOW -- the same 7 YELLOW cells as `cuda:b200_sm100` 5, no light changed.
Tests: 79 passed (new: the wide forms / gradient pack bitwise against the replaced statements, the dropout scale over consecutive draws
and over graph replays).

The 48 remaining, by measured cause:
- **One-tile latency at L128 / L256, inference (20: D64 / D64 hidden-128 / D128 / bidirectional D64 / D128 / D256 / D384; 0.667-0.975).**
  As the third pass measured: three kernels whose single-tile paths are ~4 us each plus ~1.5 us start-up; the PDL edges hide ~2 us when
  the instantiated graph honours them (D384 eval L128 is 0.0532-0.0537 ms with the edge and 0.0552 without it, i.e. 0.65 / 0.67; the
  bidirectional D256 L128 0.0511 sits on the line). Needed: M = 64 / split-channel tcgen05 tiles in `k1w` / `k3g` / `k3w` (more CTAs,
  shorter per-tile pipelines) -- not started.
- **D64 / D128 training at L128 / L256 (20; 0.714-0.947).** D64 L128 = 13 launches whose kernels are each one-tile-latency bound
  (`b1s` 15.5, `b7m` 13.5, `k3g` 7.4, `k1w` 6.1 us) plus the reference dropout draw (rand + two graph-RNG fills, which the H100's
  Philox launch did not shorten here); D128 L128 is `b1g` / `b7g` ~29-30 us each on one or two tiles per CTA. The H100's in-kernel
  D128 partial reduction would remove the `grad_reduce` launch (~3-5 us) -- short of the 5-11 us these cells need.
- **D64 training L384-640 (8 + RF3 L384 / L640 4 + bidirectional L384 1: 0.667-0.703).** Elementwise-throughput bound (`b1s` / `b7m` /
  `k3g` / `k1w`, second pass); this pass took 1-4 us (`grad_reduce`, the L512 cuBLASLt keys), 2-4 % short remain.
### Round 3, training pass (2026-11-17, `cuda:b200_sm100` 7, agent r3-trimul-train)

**The unified B1 for D64 one direction (`b1s.cu` `b1u_kernel`, `b200_train.B1U`).** At hidden = 64 every operand of the B1 stage fits
one CTA (TMEM: dWg 64 | A 64 | G 2 x 64 | Q 2 x 64 | dnr 2 x 64 = 512 columns), so each CTA runs the front epilogue (P / dP / dG / S1 /
S2, dWg) AND the LN_out backward of its own tiles with dPr kept in shared memory: no L2 ring, no ring flags, no cooperative launch, no
front / LN role split. tri / (xn | dy) rings 4 / 3 deep; Wg kept halved in shared memory for the gate tanh. Grid min(tiles, SMs), one
partial row per CTA into the unchanged `grad_reduce` layout (fixed order: deterministic). Hidden 128 and the bidirectional class keep the
ring b1s (their operands do not fit one CTA's TMEM). A / B in one warmed process per length, B1U False / True alternating, three rounds,
lowest median (`_tools/ab2.py`, ms):

| case | L128 | L256 | L384 | L512 | L640 | L768 |
|---|---|---|---|---|---|---|
| D64 outgoing | 0.0667 -> 0.0668 | 0.1097 -> 0.1057 | 0.1814 -> 0.1712 | 0.2920 -> 0.2612 | 0.4579 -> 0.4079 | 0.6109 -> 0.5409 |
| D64 incoming | 0.0706 -> 0.0686 | 0.1139 -> 0.1074 | 0.1855 -> 0.1725 | 0.2935 -> 0.2612 | 0.4578 -> 0.4056 | 0.6103 -> 0.5418 |
| RF3 D64 incoming | 0.0713 -> 0.0687 | 0.1138 -> 0.1076 | 0.1869 -> 0.1732 | 0.2920 -> 0.2633 | 0.4496 -> 0.4081 | 0.6012 -> 0.5418 |

Accuracy (`_out/acc_v7.jsonl`, L128-768 training, every row): 119 GREEN + 7 YELLOW, the same 7 YELLOW cells as `cuda:b200_sm100` 5 / 6.
Tests: 84 passed (new: unified vs ring b1s on y / dx / every parameter gradient, and five backwards bitwise equal).

Measured and not taken:

| experiment | numbers (ms) |
|---|---|
| H100 in-kernel partial reduction (grid barrier after the last partial store of b7m, every CTA then reduces 128 outputs per warp, no `grad_reduce` launch; `GR_IN_B7`) | D64 L128 0.0677 -> 0.0800, L256 0.1066 -> 0.1210, L384 0.1702 -> 0.1937, bidirectional D64 L128 0.0959 -> 0.1076, RF3 L128 0.0677 -> 0.0800: the barrier waits for the slowest CTA's tail and the reduction then runs on one CTA wave without overlap, while the separate launch starts as a PDL dependent. Rejected; the D128 port (`b7g`, the same reduction code) was not built on this evidence |
| ring b1s without the cooperative attribute (`B1S_COOP=0`, probe only) | bidirectional D64 L128 0.0934 -> 0.0903 / 0.0908, D64 hidden 128 L128 0.0872 -> 0.0851: ~3 us, but the ring flags need every CTA resident, which only the cooperative launch guarantees; not taken (correctness first) |
| b7m without the gate-gradient arithmetic / without the LN_in backward (timing ablations `B7M_ABL_DG` / `B7M_ABL_LN`, wrong results) | D64 L512 0.258 -> 0.252 / 0.235, L256 0.106 -> 0.102 / 0.0995: the LN_in backward is ~10 % of the step at L512, the gate arithmetic ~2 %; both are inside the TMEM read-out of b7m already (one read of each tensor), so no further fusion target |

State (two full training sweeps, one measurer process per length, small lengths first, `_out/spd_m1_{1,2}.jsonl`; the cases above 0.667
re-timed with L128 / L256 in their own process and L384+ in another, `_out/rem_m3.jsonl`; lowest median; `_out/summ_m3.txt`). Training
cases above 0.667: TriangleMultiplication 16 -> 12 (worst 0.947 -> 0.919), RF3 8 -> 4 (0.771 -> 0.738), bidirectional 4 -> 4 (0.866 ->
0.895, the L128 case varies 0.0934-0.0959 ms between runs). No case slower than the H100; ladder holds (cuda / triton <= 0.70, cuda /
pytorch <= 0.24). Every D64 one-direction case at L384-768 now passes (0.55-0.66).

Remaining 20 training cases, by measured cause:
- D64 / D64 hidden-128 / RF3 D64 / D128 / bidirectional at L128 (11; 0.693-0.919) and L256 (8; 0.669-0.794): one-tile latency of the
  13-launch chain (k1w 6, k3g 7, b1s / b1u 15, b7m 13 us at D64 L128; b1g / b7g ~30 us at D128). Meeting 0.667 at D64 L128 needs
  0.0497 ms against 0.0667 now: a fused small-L backward (B1 + both contractions + B7 in one persistent kernel) or M = 64 tcgen05 tiles
  in every stage; not built.
- D64 hidden 128 L384 (2; 0.677-0.678) and bidirectional D64 L384 (1; 0.687): the ring b1s (hidden 128 does not fit one CTA's TMEM for
  the unified B1). A two-CTA (cta_group::2) unified B1 would carry the L384-640 gain over.

### Round 3, fifth pass: inference at small L (2026-11-17, `cuda:b200_sm100` 8, agent r3-trimul)

**Kept (`cuda:b200_sm100` 8; 7 is the training pass above).** `ctr.cu`: the contraction on tcgen05 at small L (M128 N128, K stages of
128, both directions in one launch, a programmatic dependent of `k1w` that triggers `k3g` at entry; bitwise cuBLAS on 54 shapes) in
the measured `b200_infer.CTR_CELLS`: bidirectional L128 D64 26.5 -> 22.4 us, D128 32.6 -> 28.6 / 30.6, D256 51.0 -> 49.0, bidirectional
D64 L256 44.9 -> 43.0, one direction D128 L128 26.5 -> 24.4. `k3g` keeps a whole W_p resident when it fits the ring (D64 hidden 128 L256
42.8 -> 40.9). `k1w_stats` -> `k1w` as programmatic dependents at D384 / D512 (k1w loads x / weights / affine before the wait; D384 L128
55.2 -> 53.1 when the graph instantiation honours the edge).

**Small L timed on its own** (BRIEF4 measurement note: the L128 cases in one measurer process, the L256 cases in another, two
processes each, `_out/spd_m8a.jsonl`, `_out/summ_m8a.txt`): the same 19 eval cases stay above 0.667 as in the mixed sweep (none was a
power-cap artefact); D64 L128 reads 18.3 us alone vs 20.3 in a mixed run, the ratios move by < 0.03.

**Built and measured, rejected:**

| experiment | numbers (graph median, us; two processes / alternating variants in one process) |
|---|---|
| **one-launch fused inference** (`_tools/fused_experiment/fused_rejected.diff`): `k1w` / `ctr` / `k3g` refactored into device bodies and run as three phases of ONE cooperative kernel (512 threads, one TMEM allocation, mbarriers invalidated between phases, global arrival counters with release / acquire + `fence.proxy.async.global` in place of the kernel boundaries; only the TMA producer of a phase waits, so set-up / weight and x loads / LN_in overlap the previous phase on other CTAs; counters self-resetting). Bitwise the three-launch chain on all 12 L128 D64 / D128 eval cases of the three classes, with and without a token mask, eager and over 30 graph replays | chain -> fused: D64 out 18.3 -> 22.4, D64 in 20.4 -> 22.4, D64 hidden 128 22.4 -> 24.4, bidirectional D64 22.4 -> 24.4, D128 out / in 24.4 -> 26.7-28.5, bidirectional D128 30.6 -> 34.7, RF3 D64 22.4 -> 26.5, RF3 D128 26.6-28.4 -> 30.6-32.5. Slower everywhere |
| why (per-CTA `globaltimer` phase marks, `-DFUSED_TL`, D64 out L128) | k1w body done 4.9 us after entry, phase hand-off 0.8, ctr 4.1 after the hand-off, hand-off 1.0, k3g 5.6 after t is ready: the phases are each one tile's latency and add up serially (16.4 us), the same span as the PDL chain (15.8: k1w 6.2, ctr ends 4.5 after it, k3g 5.1 after that) -- the chain already overlaps each kernel's set-up and k3g's LN_in with the previous kernel on the SMs the previous kernel does not use, which a sequential per-CTA body cannot do. Bidirectional D128: 9.5 / 7.2 / 11.5 us per phase |
| fused, grid of every SM with the contraction on the CTAs past the token tiles (`rev`) | D64 22.4 (no change), D128 26.7, bidirectional D128 32.8-34.7: still 2-4 us slower |
| fused without the cooperative attribute (plain launch) | identical to the cooperative launch (22.4 / 26.8 / 34.7): the cooperative launch is not the cost |
| `ctr` in more cells: one direction L128 x 256 / 384 contracted (D256 / D384), L256 x 128 (D128, D64 hidden 128), bidirectional L256 x 128 | D384 53.1 / 55.2 -> 53.2 / 55.2, D256 38.8 -> 38.8, D128 L256 49.1 -> 51.1, D64 h128 L256 42.8 -> 42.9-43.0, bidirectional D128 L256 71.7 -> 69.7 (not a miss; within the graph-instantiation spread): not taken |

**M = 64 / channel-split tiles (task item 1):** not built as tcgen05 M = 64 kernels. The phase marks above (and the second pass's
clock64 buckets) show each L128 stage is one 128-token tile's dependency chain (cold TMA ~1.3 us, LN on one thread per row, MMA,
epilogue, TMA store) with the SMs mostly idle; the fourth pass's two / three warp-group LN split of `k1w` -- the "shorter per-tile
pipeline" half of the M = 64 idea, on the current tile -- measured slower (k1w 6.20 -> 6.91 us D64 L128). An M = 64 layout changes the
TMEM data path (lanes 0-15 per sub-partition) of every epilogue in k1w / k3g / k3w and was not attempted in this pass.

State: the 19 eval cases above 0.667 at L128 / L256 are unchanged by this pass (worst: D64 hidden 128 L128 0.930, D64 L128 0.876 /
0.904, bidirectional D64 L128 0.827, D128 L128 0.823 / 0.826, D64 L256 0.789 / 0.822, D256 L128 0.755 / 0.782, D64 hidden 128 L256 0.782,
bidirectional D64 L256 0.747, bidirectional D128 L128 0.726, D128 L256 0.696 / 0.698, D384 L128 0.670 / 0.674 -- D384 out L128 is 0.645
whenever the PDL edge is honoured). For the D64 cells the target (13.5-13.9 us) is below this chain's GPU span (15.8 us) plus the ~2.5-4.5 us
a graph replay costs; a fix has to shorten each stage's single-tile latency, not the number of launches.

Audit additions:

| H100 technique | B200 | evidence |
|---|---|---|
| fewer launches at small L (the H100 eval graph at L128: K1, cuBLAS, K3) | same count (k1w, ctr, k3g); one launch built and measured slower | table above |
| tile tables by L | used (`CTR_CELLS`, measured cells only) | L128 x 256 / 384 and L256 x 128 cells measured, no gain |

### Round 3, sixth pass: inference (2026-11-17, `cuda:b200_sm100` 20, agent r3-trimul)

**State measured first** (`_tools/spd3.py`, cuda / triton / pytorch, every eval case, one exclusive measurer run per length,
`_out/spd_st0.jsonl`, `_tools/ana.sh`): 172 eval cases against the H100 tokens (none under the 20 us floor), 20 above 0.667, one slower
than the H100 (D64 incoming L128 20.3 vs 20.2 us); ladder: no violation (cuda < triton < pytorch on every case).

**The timer grid.** The B200's event timestamps step by 2.048 us, so a replay median is round(t / 2.048 us) ticks: a case whose true
mean sits at a tick boundary reads either tick per capture (the "bimodal PDL edge" of the third pass is this: D64 L128 means 19.2-20.4 us,
boundary 19.46; D384 L128 54.3, boundary 54.27; D128 L256 49.7-51.1, boundary 50.2). `_tools/ab.py` therefore compares variants
by the mean of 200 replays over several fresh captures, alternating the variants in one process.

**Kept:** the cancelled predecessor's M = 64 tcgen05 front / output (`k1h.cu`, `k3h.cu`, stash "cancelled agent 2026-11-17 (r3-trimul)"),
re-measured and re-selected (`b200_infer.M64_CELLS` / `M64_FRONT`; debug phase marks left out): bidirectional D64 L128 22.40 -> 20.44 us,
L256 43.25 -> 42.13; D64 L256 34.4 -> 32.9 (both directions); hidden 128 L128 22.43 -> 21.55; hidden 128 L256 41.6 -> 40.8-41.1 with
k3h only. Bitwise test (`test_b200_m64_tiles`), 110 GPU tests passed, eval accuracy L128 / L256 all GREEN (`_out/acc_p6.jsonl`).

**Built, measured, rejected** (sources kept in `_tools/`):

| experiment | result (mean of graph replays, us) |
|---|---|
| `ctr` for the one-direction cells (D64 / D256 / D384 L128, D64 / D128 L256) | no change or +0.1-2 (D64 L256 32.6 -> 34.7, D128 L256 49.0 -> 51.1) |
| k1w chunk-split schedule (tiles x chunks over every SM in contiguous ranges; bitwise; `k1w_split_rejected.cu`) | L128 +0.4-2.1 everywhere (D384 55.2 -> 57.3, D128 26.4 -> 28.5); L256 D384 / D512 -1 to -4 %, D128 +1-2 % |
| k1w weight K-blocks multicast to a CTA pair (cluster 2, each CTA loads half; bitwise; `k1w_mc_rejected.cu`) | no gain (D384 L128 54.6 -> 55.7, D256 38.3 -> 38.8, D128 L256 50.6 -> 51.4): k1w is not L2-bound |
| k1w input statistics from global x in the transform warps, no `k1w_stats` launch at D384 / D512 (`k1w_inl_rejected.cu`) | L128 +1.4-3.3 (D384 54.6 -> 55.9 / 57.0), L256-768 -1.5 to -3 % (passing cases only) |
| cuBLASLt table at L1024-2048 x 64 / 128 planes (`ltsweep.py`) | no candidate faster than torch.bmm (1.02-1.06) |

Why the rest stays (`TM_TL` timelines, `_tools/j_tl.sh`): D384 L128 k1w is 12 chunks of ~1.0 us of MMA each on one 128-token tile per
SM (the MMA alone is ~0.78 us per chunk); D256 L128 / D128 L256 have the same shape; D64 L128 is three ~4-6 us single-tile chains.

**After** (L128 / L256 / L2048 re-measured, other lengths unchanged code: `_out/spd_after.jsonl`): still 20 above 0.667, one slower than
the H100 (D64 incoming L128, 20.3 vs 20.2, at the tick boundary); the M = 64 cells moved (bidirectional D64 L128 0.827 -> 0.755, L256
0.749 -> 0.714; D64 L256 0.837 / 0.826 -> 0.788 / 0.777; hidden 128 outgoing L128 0.919 -> 0.838). Ladder: no violation.

### Round 3, seventh pass: inference (2026-11-17, agent r3-trimul; no kernel change, `cuda:b200_sm100` stays 40)

**State measured first** (base ddd984081, `_tools/st.sh st7`: cuda / triton / pytorch, every eval case, one exclusive measurer run per
length, `_out/spd_st7.jsonl`, `_out/ana_st7.txt`): 172 eval cases, 20 above 0.667, one slower than the H100 (D64 incoming L128, 20.3 vs
20.2 us), ladder: no violation. Against the sixth pass's after-state only cases at a tick boundary moved (D64 hidden 128 incoming L2048
0.678 -> below, D256 outgoing L1536 below -> 0.681).

**Why D64 L128 reads 18.3 or 20.3 us: the measurer's L2 flush and where the buffers land (not the driver, not the clock).**
`_tools/bim2.py` (16 measurer captures in one process, SM clock read before / after each): the clock is 1820-1890 MHz in every capture,
the first three captures read 18.3 us, the rest 20.3. `_tools/bim3.py` (10 graphs captured and kept, each re-timed in three later
passes): every graph keeps its own time in every pass (18.49 ... 20.18 us, spread 1.7 us), and a second process reproduces the same
per-capture sequence -- the time belongs to the graph, i.e. to its buffers' addresses (each capture's private pool lands 20 MB further).
`_tools/addr.py` places planes / t / y at swept offsets of one arena (2 MB steps): each of the three buffers alone moves the replay mean
between 18.5 and 20.1 us in a ~16-24 MB periodic pattern; with the L2 flush off (`NOFLUSH=1`) every placement reads 17.96-18.08 us. So the
slow captures are the flush's dirty L2 lines (`flush.zero_()` before every replay) being written back where this call's lines are
allocated -- on the two-die B200 the cost depends on the address. The kernels themselves run the case in 18.0 us (H100 token 20.2).
The measurer cannot be changed; the next two rows tried to make the call less sensitive to it.

**Built, measured, rejected:**

| experiment | result |
|---|---|
| TMA issue from several lanes (coordinator's finding: one thread ~1 load / 600 clk): k1w x / weight rings and first fills (slot `s` by lane `s % ml`), k1h / k3h / k3g first fills (unit `u` by lane `u % ml`), ctr's four stage boxes from lanes 0-3, k3w stage ring; `ml` per kernel at launch (env), bitwise the single-lane kernels on 11 eval cases (`_tools/ckml.py`) -- diff `docs/b200/rejected/trimul_multilane_tma.diff` | per kernel (`_tools/ab.py`, mean of 200 replays, 3 captures each, us): k1w 4 / 8 lanes D256 L128 39.97 -> 45.04 / 48.91, D384 L128 59.36 -> 66.51 / 61.66, bidirectional D128 L128 29.64 -> 30.58 / 30.87; k3w 5 lanes D384 L128 59.36 -> 59.03, D256 40.0 -> 40.5; k1h / k3h / k3g / ctr: within the placement noise (+-1 us, see above) on every case. The lanes of one producer warp wait on different ring slots and serialise each other; k1w's weight stream at D384 L128 already delivers ~12 boxes per ~1 us chunk (not issue-bound) |
| t and y without buffers of their own at L128, D <= 128 (ctr writes t over the left planes it has read -- one item per channel at L128 -- and k3g / k3h write y over the first right planes), so after k1w no new L2 line is allocated; bitwise the base on 9 L128 cases -- diff `docs/b200/rejected/trimul_alias_t_y_in_planes.diff` | 6 captures each: D64 incoming 19.79 -> 20.21, outgoing 20.38 -> 20.29, hidden 128 20.51 -> 20.73 / 21.08 -> 21.11, bidirectional D64 21.20 -> 21.05, D128 24.56 -> 24.58 / 24.54 -> 24.51, bidirectional D128 29.73 -> 30.24, RF3 D64 22.44 -> 22.40: no gain |
| D64 one direction L128 on the M = 64 kernels with ctr in between, every launch a programmatic dependent (k1h -> ctr -> k3h; task item (b)) | incoming 19.97 -> 19.75, outgoing 19.89 -> 19.84 (4 captures each, within the placement noise); ctr incoming is not bitwise cuBLAS's on this shape (max 0.0156 on y) |

**Task item (a), channels of k1w over a CTA pair / two CTAs per token tile at L128 / L256 -- not built, bounded.** k1w at D384 L128 is
MMA-throughput-bound per SM (12 chunks of M128 N128 K384 at ~0.78 us = the SM's bf16 peak) on 128 of the 148 SMs. A pair or two CTAs per
tile only help if the extra CTAs run on SMs that are otherwise idle: two CTAs per SM share its tensor core, and 256 one-SM CTAs are two
waves. The best schedule spreads 128 x 12 chunk units over 148 SMs (10.4 instead of 12 per SM: at most ~1.6 us of k1w's ~13), which is the
sixth pass's chunk-split schedule (`_tools/k1w_split_rejected.cu`: L128 +0.4-2.1 us, the helper CTAs reload / re-normalise their x tiles).

State after this pass: unchanged (no kernel or route change): 20 above 0.667, one above 1.0 (D64 incoming L128, 18.0 us kernel time,
20.3 us under the flush at the slow placements), ladder 0.

### Round 3, eighth pass: inference (2026-11-17, `cuda:b200_sm100` 44 (42 before the training agent's 43 merged), agent r3-trimul)

**State measured first** (base 83e4c1ddf merged; L128 / L256 / L384 re-measured exclusively, `_out/spd_st8.jsonl`; the longer lengths
from the seventh pass's `_out/spd_st7.jsonl`, no TriMul source changed in between): the same 20 eval cases above 0.667, one slower than
the H100 (D64 incoming L128 20.4 vs 20.2 us), ladder 0.

**Per-kernel timelines** (`_tools/profall.py`, one graph replay, L2 flushed, us):

| case | front | contraction | output | reading |
|---|---|---|---|---|
| D256 L1536 | k1w 1664 | cuBLAS 1164 | ln_stats 204, k3w 1255 | k1w 3.6 GB in 1.66 ms (2.2 TB/s) at 0.74 PFLOP/s: neither bound; k3w 3.6 GB in 1.26 ms |
| D128 L256 | k1w 18.4 | cuBLAS 7.8 | k3g 20.0 | 512 tiles on 148 CTAs (up to 4 a CTA), each stage a one-tile chain |
| bidirectional D128 L128 | k1w 11.2 | ctr 15.2 (ends 6 us after k1w) | k3g 11.1 | k1w streams 32 weight boxes a tile from one thread |
| bidirectional D64 L256 | k1h 15.0 | ctr 13.3 | k3h 15.9 | each ~2.5 TB/s or less (L2-resident operands) |
| D64 hidden 128 L128 | k1h 6.8 | ctr 6.2 | k3h 9.5 | single-tile chains |

**Kept (the same bits, `cuda:b200_sm100` 42):**

| change | A/B (`_tools/ab.py`, mean of 200 L2-flushed graph replays, 4 alternating captures, us) |
|---|---|
| k1w weight ring issued by two warps (warp 3 even fills, warp 2 -- idle after the TMEM allocation -- odd fills, one thread each) at D >= 256 (`TRIMUL_K1W_NP` overrides) | eval L128: D384 60.4 -> 57.0 / 60.3 -> 57.3, D256 40.0 -> 39.0 / 39.9 -> 38.8, bidirectional D256 53.4 -> 51.3, D512 85.2 -> 79.5; L256: D256 98.5 -> 93.6, D384 181.7 -> 170.4; D512 L512 1086 -> 1019; D256 L768 854 -> 816; D384 L768 1621 -> 1506; training D256 L128 187.9 -> 187.4, D384 L256 785 -> 778. D <= 128: no change (D128 L256 50.2 -> 50.7, bidirectional D128 L128 29.8 -> 29.5, D64 equal) -- not used there |
| the D >= 256 inference front packed once with the forms (`b200_infer.pack_front`: chunk c's 64 gate and 64 projection rows as rows 128c .. 128c + 127 of one [2P, D] matrix; k1w takes it when `wl` has 2P rows) -- one [128][64] TMA box per weight slot instead of two [64][64] | on top of two warps: D384 L128 58.1 -> 57.3 / 58.5 -> 57.3, D512 80.8 -> 78.8, bidirectional D256 52.4 -> 51.4, D256 L256 95.7 -> 94.1, D256 L128 39.2 -> 38.6 / 39.2 (equal); packing with one issuing warp: no gain (D384 60.3 -> 59.8) -- the issue rate is set by the number of independent issuing threads, not by the instruction count |
| D <= 128: every weight K-block loaded once per CTA and kept when the front's K-blocks fit the ring (P / 64 x D / 64 <= 8: one direction D128, D64; `TRIMUL_K1W_WRES`) -- k1w re-streamed them per tile | eval: D128 L256 50.2 -> 49.6-50.0 / 50.2 -> 49.7, RF3 D128 L256 52.8 -> 52.1, D128 L384 86.8 -> 86.1, L512 141.1 -> 140.0, L768 304.7 -> 300.6, D64 L384 52.2 -> 51.3; training D128 L256 196.2 -> 196.6, D64 L384 156.9 -> 157.2 (noise) |

Bitwise (`_tools/ckml.py`, eager, three repeats each): every variant equals the base on 16 eval cases (D64 / D128 / D256 / D384 / D512,
bidirectional D128 / D256 / D384, RF3, hidden 128).

**Built, measured, rejected** (diff `docs/b200/rejected/trimul_two_warp_issue_k3w_ctr.diff`):

| experiment | result (us) |
|---|---|
| k3w stage ring issued by two warps (0 even, 2 odd) | D256 L1536 3935 -> 4003, D384 L768 1621 -> 1682: slower |
| ctr stage boxes issued by two warps (0 even, 3 odd) | bidirectional D64 L256 41.8 -> 42.2, D128 L128 30.6 -> 30.6, D64 L128 21.4 -> 21.8, hidden 128 L128 21.1 -> 21.2: no gain |
| k1w without its weight stream (timing probe: the ring filled once, WRONG output) at D256 L1536 | 3935 -> 3893 (-1 %): the large-L front is not weight-bound; D256 L1536 (0.681 in the sweep, 0.634 alone in the second pass) is the measurement spread of the first long case in a process |

**What stays and why.** D256 L128 needs 38.8 -> 33.0 us, bidirectional D128 L128 30.6 -> 28.1, D64 L256 32.7 -> 27.7, D64 hidden 128
L128 22.4 -> 16.1: each is three single-tile dependency chains (the fifth to seventh passes); the k1w issue change took 1-2 us of D256 L128.
D128 L256 sits on the 2.048 us timer grid at 24 ticks (49.2 us) with the threshold at 48.9 (the residency saves ~0.5 us of a needed ~1.6);
bidirectional D64 L256 is 20-21 ticks against 19.5.

**After** (base b070dfb11 = engine 2.5.14 merged, so against the RE-MEASURED 2.5.14 H100 tokens; `_tools/j_r8h.sh`: per length the new
build (cuda, triton, pytorch) and right after it the pass-8 flags off (`TRIMUL_K1W_NP=1 TRIMUL_PACK=0 TRIMUL_K1W_WRES=0`, cuda) in exclusive
measurer runs, `_out/spd_new.jsonl` / `_out/spd_old.jsonl`): 1.5x misses 25 (flags off, same session) -> 20; the earlier session's
before-state (`_out/spd_before.jsonl`) read 21 against the 2.5.14 tokens -- the node ran ~1-4 us slower in this session at L128. Cases
fixed: D256 L1536 both directions, D256 L640 both directions, bidirectional D256 L640 (and in-session: D256 L128 40.8 -> 38.8, D384 L128
61.4 -> 57.3, bidirectional D256 L128 55.2 -> 51.1 (0.669), D128 L256 51.1 -> 49.1 (0.671 / 0.674)). One case slower than the H100 (D64
incoming L128, 20.4 vs 20.2, unchanged path). Geomean new / old over the 172 eval cases 0.964 (74 faster by > 3 %); three read > 3 % slower,
all on paths this pass does not change or at one timer tick (D64 L256 outgoing: k1h, untouched; D128 L128 outgoing 24.5 -> 26.5 = one
2.048 us tick, ab.py mean 26.2 -> 25.9; RF3 D128 L1536 first long case of its process). Ladder: no violation. Tests: 130 passed
(`tests/gpu/test_trimul_b200.py`, merged tree).

### Round 3, ninth pass: D64 L128 inference in one cluster launch (2026-11-18, `cuda:b200_sm100` 46 on TriangleMultiplication / RF3, agent r3-trimul)

**Why another one-launch kernel after the rejected ones (pass 2 `kf`, pass 5 `fused`).** Those ran the chain's three phases in one grid
with TWO all-to-all exchanges through L2 (planes, then t) and lost: each phase is still one tile's latency, and the PDL chain already
overlaps set-ups. `tf_sm100.cu` removes one exchange: the contraction is done by the CTAs that own the OUTPUT tokens, so t never
leaves the cluster and the output phase needs no second barrier (the TriAttn L128 lesson: fewer cross-CTA dependencies, not fewer
launches).

**Kernel** (`b200_sources/tf_sm100.cu`, extension `trimul_b200_l128_v13`, `b200_infer._tf_ext`; served: D64, hidden 64, one direction,
L = 128, B = 1, inference without dropout, bf16 [64, 64] weights; `TRIMUL_B200_L128=0` runs the chain). 128 CTAs = 32 clusters of 4
(at most 15 clusters of 8 and 7 of 16 are co-resident on the 148 SMs -- `cudaOccupancyMaxActiveClusters`, `tf_clusters()`; the
counter waits need every CTA resident, so the route checks >= 32 clusters of 4 first). A cluster owns a 16 x 32 block of pair tokens,
a CTA its 8 x 16 sub-block (the x tile, a 3-D TMA box; incoming reads the box q-major) and 16 contraction channels.
1. front: x (TMA at entry), LN_in (thread = row), [a | b] gated projections (tcgen05 N 128 + 128) and the output gate g = xn Wg^T
   (N 64, kept in TMEM to the end), x copied into TMEM for the residual; a, b written with st.global straight from registers into
   BLOCKED planes (per channel, one 256-byte block per CTA sub-block, in the contraction's K-major orientation: incoming writes the
   plane transposed, so both directions contract T_c = A_c B_c^T).
2. one dependency, not a grid barrier: per 32-row plane block a monotonic counter (each launch adds exactly 32; the launch's base is
   read at entry before the CTA's own increment); a CTA waits only for the two blocks it reads (A rows I0.., B rows J0..).
3. contraction: A (16 rows) and B (32 rows) of its 16 channels, 192 KB over the front's dead weights / x tile / staging, four channel
   quarters on four barriers (four issuing lanes); mma.sync m16n8k16 (fp32 accumulators, one bf16 rounding), a warp per channel PAIR
   over the whole 16 x 32 block (no operand byte read twice: the ldmatrix traffic bounded this phase -- 384 KB per CTA with a warp per
   channel group x column tile was 1.3 us, the pair form 0.25 us).
4. t exchange in the cluster: each warp stages its pair per destination (128 tokens x 4 B) in its consumed operand slab and sends ONE
   512-byte DSMEM bulk copy (`cp.async.bulk.shared::cluster`) per destination, completing on the destination's mbarrier.
5. LN_out (thread = row, t tile [32 pairs][128 tokens]), p = xo Wp^T (tcgen05), y = x + bf16(sigmoid(g) p) (k3g's arithmetic) stored
   with st.global by warps 4-7 / 12-15.

**Where the time went** (globaltimer marks per CTA, `-DTF_TL`, `_tools/tftl.py`, D64 incoming L128, graph after an L2 flush, us from
the first CTA entry; CUPTI kernel time in brackets):

| version | x in | planes released | counters passed | operands | t full | y stored | kernel |
|---|---|---|---|---|---|---|---|
| first (8 x 16 row tiles, TMA plane store of 32-B rows, st.async v4 exchange, 8 contraction warps, exit atomic + reset) | 2.6 | 6.7 | 7.4 | 9.0 | 11.5 | 13.3 | 17.0 |
| blocked planes (256-B rows), channel quarters | 2.3 | 6.4 | 7.4 | 9.0 | 11.8 | 13.3 | 15.2 |
| monotonic counters, no exit atomic / cluster barrier | 2.6 | 5.9 | 6.7 | 7.7 | 10.2 | 12.0 | 13.9 |
| 16 contraction warps (A re-read by every column warp: slower) | 2.6 | 5.9 | 6.9 | 8.5 | 11.5 | 13.3 | 15.0 |
| warp per channel pair + 4-B st.async (4096 a CTA: the exchange took 4.6 us) | 2.6 | 5.9 | 6.9 | 7.9 | 13.3 | 14.9 | 17.3 |
| warp per pair + 512-B DSMEM bulk copies | 2.6 | 5.9 | 6.9 | 7.9 | 9.5 | 11.0 | 13.9 |
| early cluster arrive, packed staging stores | 2.6 | 5.9 | 6.7 | 7.7 | 9.5 | 11.0 | 13.5 |
| planes and y by st.global (no staging / TMA store) | 2.6 | 5.6 | 6.7 | 7.9 | 9.5 | 11.0 | 13.0 |

The plane hand-off (registers -> L2, fence, release, acquire, TMA load of 192 KB) is ~3.3 us of the 13 (write ack ~0.8, counter
propagation and the slowest CTA ~1.0, operands ~1.3); the cold x load ~2.6 us after entry; ~2 us of the CUPTI time lies outside the
first entry / last mark (launch of a 219 KB, 512-thread cluster grid). Not kept: the final epilogue split over warps 4-7 and 12-15
(level), sixteen contraction warps (slower, above).

**The chain it replaces** (CUPTI, same case, `_tools/proft.py`): B200 `k1w` 6.30 us, cuBLAS 2.91 (starts 0.2 after), `k3g` 6.27 (PDL,
starts during cuBLAS): span 15.4 us, measurer 18.3 (incoming) / span 16.6, 20.4 (outgoing). H100 (2.5.14 tree, node02): K1
`tmn_k1_z64_h64` 6.62, cuBLAS 3.68, K3 5.70: span 16.0 / 16.7 us, measurer 19.3 / 20.4 -- three launches on both cards, the same
one-tile latencies.

**Result** (measurer, eval, graph, L2 flushed, two exclusive processes, `_out/spd_p9a/b.jsonl`; ratio vs the 2.5.14 H100 token):

| case | before (us, ratio) | after |
|---|---|---|
| TriangleMultiplication D64 incoming L128 | 20.4 (1.006, slower than the H100) | 16.2 (0.802) |
| TriangleMultiplication D64 outgoing L128 | 20.3 (0.965) | 16.2 (0.771) |

Every other cell unchanged. Accuracy: the one-launch output's error against an fp64 reference equals the chain's (max / rms, plain
and masked, both directions, RF3 too; `_tools/tfchk.py`), `measure.accuracy` eval L128 of the three classes 21 / 21 GREEN
(`_out/acc_p9.jsonl`); deterministic; graph replay = eager. Not bitwise the chain (mma.sync sums the contraction in another order than
cuBLAS: <= 1-2 bf16 ulps of y).

**Limits / caveats.** The plane-row counters are module globals: two of these launches running concurrently on different streams of
one device would corrupt each other's hand-off (not a case of the engine's inference; the chain is used whenever `tf_supported()` is
false). Left at L128: hidden 128 / bidirectional D64 (contraction 128 channels: 384 KB of operands a CTA do not fit beside the rest;
needs two channel passes or 8-CTA clusters, which are not co-resident), D128 (front N 512 and t 128 channels: TMEM / smem), D256 / D384,
and L256 (4 x the tokens: not one wave).

### Round 3, tenth pass: the D64 L128 one-launch kernel (2026-11-18, `cuda:b200_sm100` 48 on TriangleMultiplication / RF3, agent r3-trimul)

**State first** (`_out/spd_p10base.jsonl`, cuda / triton / pytorch, eval L128 and L256 each in its own exclusive process): 20 cases
above 0.667 (the same 20 as pass 9; none slower than the H100), ladder 0.

**The floor of this measurement.** An empty kernel with the same launch (512 threads, 128 CTAs in clusters of 4, 219 KB shared memory;
`-DTF_EMPTY=1`; 1 KB shared memory `=2`, no cluster `=3` are the same): CUPTI 1.06 us, measurer 4.1 us (2 ticks). So the measurer reads
the kernel + ~3.4 us, on the 2.048 us grid: 8 ticks (16.4 us) until the kernel drops below ~11.9 us, 7 ticks (14.3, still 0.708 / 0.68)
below that; the 1.5x target needs 6 ticks (12.3 us), a kernel of ~10 us.

**Variants** (`-D` flags of `tf_sm100.cu`, `_tools/j_p10b.sh`: tfchk, proft CUPTI, tftl marks; D64 incoming, us from the first CTA entry):

| variant | x in | planes released | counters | operands | t full | end mark | CUPTI in / out | measurer |
|---|---|---|---|---|---|---|---|---|
| pass 9 (x by one TMA box, per-thread fence) | 2.56 | 5.38 | 6.40 | 7.42 | 9.22 | 10.75 | 12.61 / 12.67 | 16.3 |
| x rows by the LN threads into registers (`TF_XREG=1`) | 3.07 | 5.89 | 6.66 | 7.94 | 9.47 | 11.01 | 13.25 / 13.57 | 16.3 / 18.3 |
| operands from 8 lanes, a barrier per contraction warp (`TF_OPL=8`) | 2.56 | 5.38 | 6.40 | 7.42 | 9.22 | 10.75 | 12.54 / 12.64 | 16.3 |
| one fence a CTA after the barrier (`TF_ONEFENCE=1`) | 2.56 | 5.12 | 6.14 | 7.17 | 8.96 | 10.50 | 12.38 / 12.38 | 16.2 |
| + x by all 512 threads, coalesced, as the first instructions (`TF_XREG=2`) -- **taken** | 2.05 | 4.86 | 5.63 | 6.66 | 8.45 | 9.98 | 12.1-12.5 / 12.1 | 16.2 |
| + output epilogue g / x from TMEM before the wait for p (`TF_EPI=1`) | 2.05 | 4.86 | 5.63 | 6.66 | 8.45 | 9.98 | 12.64 / 12.10 | 16.2 |

Taken: `TF_XREG=2` + `TF_ONEFENCE=1` (the same bits as pass 9: no arithmetic changed; tfchk max / rms vs fp64 equal to the chain's,
graph == eager). Register rows were slower (each 16-byte lane load hits another 128-byte line: twice the sectors); 8 operand lanes and the
early epilogue loads are level. The cold x read is still ~2 us after entry: the measurer's L2 flush (a `zero_` of a large buffer) leaves
L2 full of dirty lines, so the first reads wait for write-backs.

**Why D64 L128 stays a miss.** After the change: x 2.0, front 2.8 (LN 0.3, MMA 0.3, plane epilogue 0.8, release fence ~1.0), hand-off
1.8 (counter propagation ~0.8, 192 KB operands per CTA ~1.0 -- 24 MB from L2 for the grid), contraction + DSMEM exchange 1.8, LN_out /
Wp / epilogue 1.5, and ~2.1 us of CUPTI time outside the marks (launch ~1, the y write drain at exit). 6 ticks need ~2 us less of these
2-3 us blocks; 7 ticks (no pass either) ~0.3 us less.

**Not built, with the reason measured here.** Hidden 128 L128 (H100 23.8 us -> needs 15.9, i.e. 7 ticks): a one-launch kernel with 128
contracted channels is necessarily slower than the D64 one, which reads 8 ticks. Bidirectional D64 L128 (H100 27.0 -> 18.0, 8 ticks
pass): needs both orientations' planes, 128 contracted channels in two 192 KB operand passes (+~1.3 us) and a second front half
(+~0.8 us): estimated ~14.5 us of kernel against a ~14 us limit for 8 ticks -- left for a dedicated pass. L256 (4 x the tokens: 512
CTA tiles on 148 SMs, so not one co-resident wave; the counter waits need co-residency): a different kernel.

**After** (`_out/spd_p10after.jsonl`, cuda / triton / pytorch, eval L128 / L256, exclusive): 20 -> 20 above 0.667 (D64 L128 0.802 /
0.772, 8 ticks as before), none slower than the H100, ladder 0. tfchk (D64 + RF3, both directions, with / without a token mask) OK;
`tests/gpu/test_trimul_b200.py` 142 passed; `measure.accuracy` eval L128 21 / 21 GREEN (`_out/acc_p10.jsonl`).

### Round 3, eleventh pass: inference (2026-11-18, agent r3-trimul)

**State first** (base d586ed4b7 merged; `_out/spd_p11base.jsonl`, cuda / triton / pytorch, eval L128 / L256 each in its own exclusive
process): the same 20 cases above 0.667 as pass 10, none slower than the H100, ladder 0.

**Built, measured, rejected: bidirectional D64 L128 in one launch** (`docs/b200/rejected/trimul_tfb_bidir_l128_sm100.cu`, route
`trimul_tfb_bidir_l128_route.diff`). `tf_sm100.cu`'s design with both directions in the same CTA tiles: four gated projections in
three TMEM slots (a_in reuses slot 0 after a_out's epilogue loaded it; x not kept in TMEM, re-read at the end), outgoing / incoming
planes released on two counter sets, eight 24 KB operand slots (one per contraction warp: pass 1 outgoing, pass 2 incoming into the
same slot; slots 4-7 lie beyond the front's weights and load as soon as the counters pass), one 1 KB DSMEM copy per destination with
t rows interleaved (row 2 q + dir), LN_out over 128 channels, Wp loaded late into slot 0's consumed B part (smem 232192 B). Correct
(tfchk: error vs fp64 equal to the chain's, plain and masked; graph == eager). But (CUPTI, graph after an L2 flush): kernel 21.4 us
against the chain's 17.1 us span (k1h 6.9 / ctr 6.0 / k3h 9.8; measurer 24.6 vs 20.3). Marks (us from the first CTA entry): x 3.1,
xn 3.6, outgoing planes released 7.2 (the 8 MB of plane writes -- twice the one-direction kernel's, the incoming ones as 4-byte
words 16 B apart -- and the second epilogue round), incoming released 9.2 (max 10.2), pass-2 operands 12.0, t full 14.6, LN_out
(128 channels, 192 4-byte smem reads a thread) done 16.9, y 17.9, end 18.7. The pass-9 estimate (~14.5 us) missed the write traffic
and the serial pass-2 dependency (incoming release -> counter -> 24 MB load); 8 timer ticks need <= ~13.9 us: not pursued.

**Measured, rejected: ctr triggering wide_ln_stats at entry** (D >= 256 chain; `ab.py`, 4 rounds, mean of 200 replays, us):
bidirectional D256 L128 50.65 -> 53.97 (ln_stats CTAs parked on the SMs during ctr), D256 / D384 L128 level; with ctr also in the
one-direction D256 / D384 L128 cells 40.8 / 59.1 (slower, as in pass 7).

**Measured, rejected: the LN_out statistics with more warps** (`docs/b200/rejected/trimul_ln_stats_nw_inference.diff`). The per-kernel
timeline (`_tools/proft.py`) puts `ln_stats_pdl_kernel<512, 2>` on the critical path of bidirectional D256 L128: ctr ends 29.9 us,
ln_stats 29.5 -> 34.7, k3w 33.5 -> 46.7 (D384 L128: cuBLAS ends 32.6, ln_stats 32.5 -> 36.9). A 16- / 32-warp form (each warp H / NWS
channels with every load in flight, Chan merge in warp order; an inference-only entry `wide_ln_stats_inf`) did not shorten it
(32 warps: 29.4 -> 34.4, t read at ~3.6 TB/s either way -- not load-latency bound). `ab.py` (4 rounds, mean of 200 replays, us,
8 / 16 / 32 warps): bidirectional D256 L128 50.70 / 49.74 / 50.61, D384 L128 56.77 / 56.20 / 56.69, D256 L128 38.86 / 38.78 / 38.92,
D256 L256 94.20 / 94.82 / 96.52. With 16 warps at L128 the full measurer sweep read the same values as before (bidirectional D256
51.1, D384 57.2): no case changes side of 0.667, so not taken. Removing ln_stats from the path needs the statistics inside k3w (from
its t stages) -- not built.

**After**: no source change kept; the violation table is the state above (20 above 0.667, worst hidden 128 incoming L128 0.933, none
slower than the H100, ladder 0). Tests on the 16-warp variant: 141 / 142 (batch == samples failed at D256 because the variant was
picked by token count; fixed by picking by L before it was dropped); accuracy eval L128 21 / 21 GREEN.

### Round 3, twelfth pass: the LN_out statistics inside k3w (2026-11-18, `cuda:b200_sm100` 50 on all three classes, agent r3-trimul)

**State first** (base b200-2.5.x 80572b7d4 merged; `_out/spd_st12.jsonl`: cuda / triton / pytorch, every eval case, each length its
own exclusive measurer run; `_out/ana_st12.txt`): 21 of 172 above 0.667 (the pass-11 20 plus D128 outgoing L1536 at 0.667 exactly),
none slower than the H100, ladder 0.

**The H100 technique and what it became here.** The H100 wide K3 (`h100_sources/inference/tmn_k3_wide.cuh`) LayerNorms the t tile it
already holds (ldmatrix.trans -> registers -> LN -> written back as the SS operand): no statistics launch. k3w keeps its folded form
(raw t as the MN-major A operand, the LayerNorm undone per token in the epilogue) and now computes the per-token statistics from its own
t stages (`OST`, `k3w.cu`): four more warps (512 threads) read every t stage from shared memory as it lands (thread = 8 tokens x 8
channels of a stage; pivot-shifted sums as `ln_stats_kernel`, lane pairs merged by shuffle and the four warps in order by Chan's
formula: deterministic), arrive on the stage's empty barrier next to the MMA commit (count 5; `fence.proxy.async` before the arrive:
generic reads before the refilling TMA), and hand mean / rstd to the epilogue through a double-buffered table (`sf` / `se` barriers);
items of a token tile the CTA has already seen (grouped schedule) copy the previous item's table and leave their stages unread. k3w
is then launched as a programmatic dependent of the contraction itself (its producers wait before reading t). Inference only
(`k3w_forward` pdl bit 1); `TRIMUL_K3W_OST=0` runs the ln_stats chain; `K3W_OST_MINL` (384) is the first length served.

Accuracy: error vs an fp64 reference equal to the ln_stats chain to 4 digits (max / rms, plain and with a random token mask, both
directions, D256 / D384 / D512, L128-640; `_tools/ostchk.py`), graph == eager, three repeated runs bitwise equal; `measure.accuracy`
eval L384 / L512 / L768 of the three classes 63 / 63 GREEN (`_out/acc_p12.jsonl`); `tests/gpu/test_trimul_b200.py` 142 passed.

**Why not at L <= 256** (`_tools/k3tl.py`, `-DK3W_TL` %globaltimer marks per CTA, and CUPTI timelines `_tools/proft.py`, us):

| case | ln_stats chain: ln_stats / k3w | OST: k3w | module (mean of 200 replays, 4 rounds) chain -> OST |
|---|---|---|---|
| bidirectional D256 L128 | 29.95 -> 35.17 / 34.01 -> 47.04 | 29.73 -> 46.34 | 51.02 -> 50.49 (level) |
| D256 L128 | 20.35 -> 23.87 / 22.69 -> 34.94 | 20.32 -> 34.34 | 39.02 -> 38.78 |
| D384 L128 | 33.28 -> 37.73 / 36.58 -> 54.40 | 33.73 -> 55.90 | 57.33 -> 57.68 |
| D512 L128 | 47.20 -> 52.67 / 51.23 -> 76.41 | 46.72 -> 78.49 | 78.16 -> 81.36 |
| D384 L256 | | | 184.6 -> 190.0 |
| D256 L384 / L512, bidirectional D256 L384 / L512 | | | 210.3 -> 204.9 / 372.4 -> 356.2, 344.0 -> 329.6 / 631.5 -> 594.0 |

At L128 k3w runs 1-2 items per CTA and its time is the stage stream plus the last item's epilogue (bidirectional D256: first stage
1.8 us after entry, 12 stages of item 0 by 4.6, item 1's by 8.5-9.0, its epilogue 9.0 -> 11.0); the statistics warps' reads and
their arrive on every stage lengthen the stream by about what the ln_stats launch cost, so the critical path does not shorten.

**Built, measured, rejected** (diffs `docs/b200/rejected/trimul_k3w_cluster_ns6_tb3_full.diff`, `trimul_k1w_weight_group_boxes.diff`):

| experiment | result |
|---|---|
| coordinator's box-count finding, k1w: two / three / four weight K-blocks of a chunk as one 3-D box ([KB][2P][64], up to 64 KB) into adjacent ring slots (`TRIMUL_K1W_WG`; bitwise) | slower everywhere it applies (ab.py, mean us, WG 1 / 2 / 3): bidirectional D256 L128 51.4 / 54.9 / 51.5 (3 = 1: the group does not divide the ring), D384 L128 57.8 / 61.4 / 61.3, D512 L128 86.5 / 92.1 / 86.1, D384 L256 172.4 / 188.6 / 184.4; k1w CUPTI D384 L128 22.4 -> 25.9 (WG 2) / 25.1 (WG 3). A group waits for WG free slots, so the 6-slot ring runs at 3 / 2 groups in flight. (An earlier run read WG 2 faster -- 48.4 vs 50.4 -- while a foreign process ran on GPU 0; not reproduced in two later runs) |
| k1w x ring as pair boxes | level (50.40 vs 50.42 us) |
| k3w t stage as one 3-D box ([M / 64][H][64], the two 64-token halves; 3 boxes a stage -> 2) | level (bidirectional D256 L128 51.02 -> 50.99, D256 L256 94.45 -> 94.49); bitwise |
| k3w in clusters of D / 128 CTAs (D256 / D512): the cluster shares each token tile's t / x blocks by TMA multicast (each CTA 1 / CL of the A block), commits multicast to the cluster's empty barriers; bitwise | k3w CUPTI: bidirectional D256 L128 13.1 -> 13.7, D256 12.3 -> 12.8, D512 25.6 -> 37.0 (clusters of 4 schedule worse): the stream is not bound by L2 reads of t / x |
| k3w with a sixth 32 KB stage (in the 1 KB alignment slack; bitwise) | slower: bidirectional D256 L128 51.7 -> 52.5, D256 L256 96.3 -> 98.0, D384 L256 172.5 -> 175.7 |
| k3w with 16 epilogue warps (`-DK3W_EPW=16`, 640 threads) | epilogue per item unchanged (~2.0-2.3 us): the epilogue is not bound by per-thread latency |
| k3w epilogue vectors (sp / ep / sg / eg) through ld.shared instead of the generic float pointer the compiler issued as LD.E | epilogue ~2.3 -> ~2.0-2.3 us per item (level); kept (same bits, training too) |

**H100 vs B200 at L128, D >= 256** (H100: the released 2.5.14 path and token medians; its per-kernel profile job for these cells
waited behind the cluster queue and was cancelled, so only the launch structure is compared; B200: CUPTI, this pass, us):

| case | H100 token | H100 path (launches) | B200 path: kernel spans | B200 module | ratio |
|---|---|---|---|---|---|
| bidirectional D256 | 76.4 | K1 (LN_in inside) / cuBLAS x2 / wide K3 (LN_out from its t tile) | k1w 0-22.8, ctr 1.5-30.3, ln_stats 30.0-35.2, k3w 34.0-47.0 | 51.2 | 0.670 |
| D256 one direction | 49.4 / 50.4 | K1 / cuBLAS / wide K3 | k1w 0-15.9 (LN_in inside), cuBLAS 16.1-20.9, ln_stats 20.4-23.9, k3w 22.7-34.9 | 38.9 | 0.77-0.79 |
| D384 | 82.0 | K1 (LN_in inside) / cuBLAS / wide K3 | k1w_stats 0-5.2, k1w 4.8-27.1, cuBLAS 27.4-33.3, ln_stats 33.3-37.7, k3w 36.6-54.4 | 57.3 | 0.699 |
| D512 | 161 | as D384 | k1w_stats 0-5.7, k1w 5.3-39.6, cuBLAS 40.1-47.3, ln_stats 47.2-52.7, k3w 51.2-76.4 | 77.8-79.3 | 0.48-0.49 |

The B200 path has two launches more than the H100 at D384 / D512 (k1w_stats, ln_stats) and one more at D256 (ln_stats); this pass
removes ln_stats from L384 on. At L128 the B200 front and output are each two items a CTA of ~3 us stage streams plus a ~2 us
epilogue; the H100 K3 is register-resident (LN, gate, residual from the fragments), the B200 k3w epilogue goes through TMEM loads
and a shared-memory residual / output tile.

**After** (`_out/spd_p12after.jsonl`: cuda, every eval case, each length its own exclusive run; triton / pytorch of the state sweep;
`_out/ana_p12after.txt`): 20 above 0.667 (D128 L1536 0.667 left the list: a path this pass did not change), none slower than the
H100, ladder 0. D >= 256 cells, after / before (state sweep): L128 0.98-1.00 (unchanged path), L256 0.98-1.00, L384 0.95-0.98, L512
0.94-0.98 (bidirectional D512 1.05 in the sweep, 1.008 in an in-process A/B), L640 0.86-0.98, L768 0.91-1.03 (D256 incoming 1.034 in
the sweep, 0.977 in-process), L1024 0.97-1.04 (D384 outgoing 1.036 in the sweep, 0.961 in-process), L1536-3968 0.93-0.99.

### Round 3, thirteenth pass: inference at L128, D >= 256 (2026-11-18, `cuda:b200_sm100` 52 on all three classes, agent r3-trimul)

**State first** (`_out/spd_st13.jsonl`: cuda / triton / pytorch, eval L128 and L256 each in its own exclusive run; the longer lengths
from pass 12, no inference source changed since): 20 above 0.667 (bidirectional D256 L128 at 0.668), none slower than the H100, ladder 0.

**Fresh per-kernel / in-kernel timelines** (`_tools/proft.py`, `tl.py`, us): D256 L128 k1w (one tile a CTA): set-up done 2.9 after entry,
x landed 3.4, xn in TMEM 5.9, MMA 6.1 -> 11.8 (8 chunks at 0.71 us, against ~0.53 of bf16 peak per chunk), stores done 12.9.

**Built, measured, rejected: k3w on a CTA pair (`cta_group::2`)** (diff `docs/b200/rejected/trimul_k3w_cta_pair_cg2.diff`). Two-CTA
clusters, the leader issues M = 256 (each CTA's 128 tokens) x N = 128 with each CTA loading its 64 of the weight rows (24 KB stages
instead of 32, 7-stage ring), commits multicast to both CTAs, the peer's epilogue arrives remotely on the leader's accumulator barrier,
`tcgen05.alloc.cta_group::2`. Bitwise equal to the one-CTA k3w (8 cases, L128 / L256, D256-512). Slower everywhere (ab.py, mean of
3 captures, us): bidirectional D256 L128 50.8 -> 52.9, D256 L128 38.7 -> 40.5, D384 L128 57.0 -> 59.3, D256 L256 92.6 -> 97.5. Per phase
(`-DK3W_TL`, bidirectional D256 L128, median / max over CTAs): the second item's last stage lands at 8.7 (max) against 9.2 -- the stage
stream is not shorter with a quarter fewer bytes per CTA (it is not L2-bound, as the pass-12 multicast found) -- while the last item's
epilogue ends 11.0 / 12.3 against 10.8 / 11.3: the slower CTA of a pair gates both. With the in-kernel statistics (OST, L384) the pair
form did not finish (not debugged: rejected on the L128 / L256 numbers first).

**Kept 1: k3w's in-kernel LN_out statistics at L128 on the tile-grouped schedule** (`b200_infer.K3W_OST_GROUP`, k3w pdl bit 3). Pass 12
found the OST form level at L <= 256 on the default schedule; there every CTA's second item is a fresh token tile, so the statistics
warps re-read it. With one token tile per CTA (grid = 128 at L128: `L * L / 128 <= SMs`, decided by L so a batch gives each sample's
bits) the CTA runs the tile's D / 128 channel items back to back, the statistics come from the first item's t stages only and the
later items' t / x blocks are L2 hits; the `wide_ln_stats` launch is gone. Rotated A/B (mean of 4-6 captures, us): bidirectional D256
50.1 -> 47.4, D256 38.7 -> 36.8, D384 55.7 -> 54.5, D512 77.4 -> 73.9, bidirectional D384 83.6 -> 79.9, bidirectional D512 121.7 ->
118.4, RF3 D256 69.3 -> 67.1. At L256 the grouped OST is slower (D256 92.2 -> 94.8, D384 167.1 -> 169.4, D512 254.0 -> 258.7; the
ungrouped OST also slower), so only one tile a CTA. Accuracy (`_tools/ostchk.py`, plain and masked, grouped): error vs fp64 equal to the
ln_stats chain to 4 digits, graph == eager, three runs bitwise.

**Kept 2: a third k1w weight issuer** (warp 0 after its x stream; flags bit 6, only with one tile a CTA -- with more tiles warp 0's
x stream waits on the transform and would deadlock the weight ring; D >= 256 inference calls, `grid = -1` from `b200_infer`; the same
bits). Rotated A/B (mean of 6): bidirectional D256 L128 51.3 -> 50.9, D384 56.9 -> 55.6, D512 78.3 -> 76.7,
bidirectional D384 85.6 -> 83.8. Five / seven issuers (the transform warps after the transform as well): no better than three.

**Measured, not used by default:** the k1w transform split over warps 4-7 and 8-11 at D <= 256 (each half the K-blocks, the
statistics as two half sums; flags bit 7, env `TRIMUL_K1W_TS=1`): D256 L128 38.8 -> 38.3, bidirectional 50.9 -> 50.5, D128 level --
but it changes the bits, and a batch (several tiles a CTA, no split) then differs from its samples (`test_b200_batch_equals_samples`
failed), so off. No weight boxes in the entry fill (`TRIMUL_K1W_EW=1`): level. The k3w grouped schedule without OST (grid 128): level.

**Not built, with the reason.** The last-item tail split (half-size last items): at L128 the grouped OST schedule above gives each CTA
one token tile (2-4 items) and removes the second tile's statistics; half items would re-stream the full A operand for half the output.
D64 L128 (`tf_sm100.cu`): the contraction is already split over the cluster's 4 CTAs by channel (16 each, independent outputs: no
reduction exists to do over DSMEM); its 192 KB a CTA is the A / B rows of those 16 channels for the cluster's 16 x 32 output block,
which only a larger block (8-CTA clusters, not co-resident) would shrink.

**After** (same session, alternating exclusive runs per length, after / before (pass-13 switches off by env), twice each, the lower
median; `_out/spd_p13A.jsonl` / `spd_p13B.jsonl`): 1.5x misses 20 -> 19 (bidirectional D256 L128 51.1 -> 49.0 us, 0.668 -> 0.641),
none slower than the H100, ladder 0 (triton / pytorch of the state sweep). L128 module medians before -> after (us): D256 38.9 / 40.8 ->
38.8 / 38.8, D384 57.2 -> 55.1 (0.697 -> 0.672: 0.5 us short), D512 77.8-79.7 -> 75.6, bidirectional D384 84.1 -> 81.6, bidirectional
D512 126.8 -> 118.8. L256 unchanged (paths unchanged). `tests/gpu/test_trimul_b200.py` 148 passed; `measure.accuracy` eval L128 of the
three classes 21 / 21 GREEN (`_out/acc_p13b.jsonl`).

### Round 3, fourteenth pass: the LN_in fold in the front, D128 on k3w (2026-11-18, `cuda:b200_sm100` 54 on all three classes, agent r3-trimul)

**State first** (base f2d941e47; `_out/spd_st14.jsonl`: cuda / triton / pytorch, eval L128 and L256 each in its own exclusive run; the
longer lengths unchanged since pass 12 / 13): 19 above 0.667 (all at L128 / L256), none slower than the H100, ladder 0.

**Kept 1: the LN_in-folded front (`k1w` FOLD, the H100 "port K3" fold applied to the front).** W' = bf16(W o gamma_in) packed as
`pack_front` (forms: `b200_infer.front_fold`), so the MMA needs no normalised x. The MMA thread copies each raw x K-block into TMEM with
`tcgen05.cp.128x256b` (the same A image the transform warps wrote: lane = token, 32 columns of bf16 pairs a K-block) as soon as it lands
and multiplies on the TS form as before; the slot is released by a commit after the copy plus the four statistics warps. Those warps
compute mu / rstd in ONE pass over the K-blocks as they land (pivot-shifted sums, the row's first value as pivot; also written for k3w)
and store a per-token K16 correction row into TMEM: [a1 a1 a2 a1 a2 a3 | c1 c1 c2 c1 c2 c3 | 0 0 0 0] with a = -mu, c = 1 / rstd, each
split into three bf16 parts. At every chunk's last K-block one more K16 MMA multiplies it with the forms' rows [u1 u2 u1 u3 u2 u1 |
v1 v2 v1 v3 v2 v1] (u = sum_k W', v = W beta_in, three parts each; a [128][16] SW32 box loaded with that K-block), i.e. adds -mu u + v / rs
to the fp32 accumulator; the epilogue only scales by rs (sigmoid(rs acc_g) rs acc_p) and masks. The statistics are off the MMA path
(D256 L128 timeline: x lands 2.7 us after entry, first accumulator 4.3 us -- the normalise-then-MMA front had xn in TMEM at 5.9) and D384
needs no `k1w_stats` launch. Served cells: `fold_cell` (D128 with the k3w output up to L768; D256 / D384 at L128; one direction D384 every
length and D256 to L512; bidirectional D384 to L512). Accuracy: error vs fp64 equal or lower than the normalise-then-MMA front (rms
3.11e-3 -> 2.94e-3 at D256 L128, plain and masked, every served shape; `_tools/foldchk.py`), deterministic, graph == eager, a batch
equals its samples (the fold is per token: schedule independent); test `test_b200_folded_front`.

| variant (built, measured) | result (rotated A/B, mean of 4 L2-flushed captures, us) |
|---|---|
| SS form: the MMA reads raw x from the x ring (row resident until the tile's last chunk), u / v applied in the epilogue from an fp32 table | slower everywhere: D256 L128 37.8 -> 40.4, D384 54.7 -> 57.8, D512 74.6 -> 87.2, bidirectional D256 48.5 -> 56.8 (SS doubles the shared-memory operand traffic; D512 needed a 3-slot weight ring) |
| cp form, u / v applied in the epilogue (two extra FFMA2 per value pair + the vector loads) | D256 L128 37.8 -> 39.7, bidirectional D256 48.6 -> 55.3: the front starts ~3 us earlier (marks) but the MMA stream drops from 0.71 to 1.24 us a chunk -- the epilogue (one warp per SM sub-partition per group) is latency-bound and the extra arithmetic gates acc_empty |
| **cp form + the correction MMA (kept)** | L128: D256 37.8 -> 35.8 / 38.2 -> 36.7, D384 54.7 -> 51.3 / 54.5 -> 50.9, bidirectional D256 48.7 -> 47.0, D384 80.1 -> 77.9, RF3 D256 123.5 -> 117.0; one direction D384 L256 167.6 -> 158.6, L512 633 -> 606, L768 1517 -> 1456, L1024 2679 -> 2544, L2048 13466 -> 13077, L3968 68259 -> 66603; D256 L256 92.6 -> 91.1, L512 329 -> 321 (L768 810 -> 816: not served); bidirectional D256 L256 151.4 -> 151.9, L384 302 -> 318 (not served); D512 not built (no TMEM columns left for the correction rows) |

**Kept 2: the D128 output on k3w** (`K3W_D128_CELLS`: one direction L128-384, bidirectional L128-768; `k3w_forward` instantiated at
(128, 128) / (128, 256), one channel item a token tile; its fold Wp' / Wg' / vectors and the folded front cached as inference forms next
to the shared D64 / D128 forms, `integrations.trimul_b200._inference_params`; RoseTTAFold3 keeps the plain front, its 1 / n is in W_l per
call). Both LayerNorms folded (the H100 K3 technique already used at D >= 256), the LN_out statistics in-kernel (OST), the grouped
schedule at L128. Rotated A/B (us, k3g -> k3w -> k3w + folded front): D128 L128 26.4 -> 25.6 -> 23.1, L256 50.7 -> 49.0 -> 47.3 / 50.6 ->
49.2 -> 47.1, L384 87.1 -> 87.1 -> 84.8, bidirectional D128 L128 30.6 -> 29.0 -> 28.5, L256 75.8 -> 71.2 -> 69.7, L512 232.8 -> 222.1, L768
557 -> 535, RF3 D128 L256 54.0 -> 51.3; one direction L512 / L768 slower on k3w (141.7 -> 144.7, 309 -> 316: not served). Accuracy: rms
vs fp64 3.2e-3 -> 3.0-3.1e-3 (k3g -> k3w), max within the same 3.0-3.6e-2.

**Not built:** (2) the LN_out fold inside `k3g` -- superseded at D128 by k3w (which is that fold); D64 / hidden 128 outputs stay on
k3g / k3h. (3) the 128-byte packed `st.async` t exchange in `tf_sm100` -- D64 L128 reads 8 timer ticks (16.2 us) and the target needs 6
(a ~2 us shorter kernel, pass 10); -0.3-0.6 us does not change a tick. (4) cuBLASLt for the D256-512 contractions -- not reached.

**After** (`_out/spd_p14.jsonl`: cuda + pytorch, every eval case of the three classes, each length its own exclusive run; triton of
`spd_st12` / `spd_st14`; `_out/ana_p14.txt`): 1.5x misses 19 -> 15 of 172 -- fixed D384 L128 both directions (55.1 -> 51.1 us, 0.672 ->
0.623) and D128 L256 both directions (51.1 -> 47.4, 0.70 -> 0.65); closer: D256 L128 38.8 -> 36.7 (0.728 / 0.743), bidirectional D128
L128 30.6 -> 28.5 (0.676), D128 L128 26.5 -> 24.4 / 22.4 (0.841 / 0.784). None slower than the H100, ladder 0. The D128 RoseTTAFold3
call keeps the plain front (its `_LeftScaled` and forms paths must give the same bits: `test_b200_rf3_factor_in_forms`).
`tests/gpu/test_trimul_b200.py` 168 passed; `measure.accuracy` eval L128-768 of the three classes 126 / 126 GREEN (`_out/acc_p14.jsonl`).
Left (15, all L128 / L256 at D <= 256): the single-tile chains of passes 5-13 (D64 / hidden 128 / bidirectional D64 / D128 L128, D64 /
hidden 128 / bidirectional D64 L256, D256 L128 2-3 us short).

### Round 3, fifteenth pass: k3hf, the D64 folded front, k3w before its wait (2026-11-18, `cuda:b200_sm100` 56 on all three classes, agent r3-trimul)

**State first** (base 4d285cb66; `_out/spd_st15.jsonl`: cuda / triton / pytorch, eval L128 and L256 each in its own exclusive run;
`_out/ana_st15.txt`): 15 of 172 above 0.667, all at L128 / L256, none slower than the H100, ladder 0.

**Timelines** (`_tools/proft.py` CUPTI, `tlm.py` / `k3tl.py` in-kernel marks; us): hidden 128 L128 k1h 0-6.9, ctr 5.5-11.6, k3h 7.3-17.1
(k3h: t lands 1.2 us after its entry, LN_out -- statistics pass + normalise pass + xo stores -- done 1.0 us later, accumulator 0.4 after
that, store 0.6 later); bidirectional D64 L256 k1h 0-14.9, ctr 12.0-24.9, k3h 22.4-37.9 (13 us after the contraction for ~3.5 tiles a CTA:
LN_out and the epilogue of every tile serial on warps 8-11); D128 L128 k1w 0-7.6, ctr 1.5-12.1, k3w 11.6-19.3; D256 L128 k1w 0-13.8,
cuBLAS 14.4-19.1, k3w 19.0-32.2.

**Kept 1: `k3hf` (`k3h.cu`), the D64 output with both LayerNorms folded at M = 64** -- k3w's form (p = rs_o (t Wp'^T - mu_o sp) + ep,
g = rs_i (x Wg'^T - mu_i sg) + eg) on k3h's 64-token tiles: the g MMA reads raw x (K-major, as loaded before griddepcontrol.wait), the p
MMA the raw channel-major t tile as the MN-major A operand the moment it lands; warps 4-7 / 8-11 only compute the per-token statistics
(LN_in two-pass on thread pairs, LN_out k3h's one-pass pivot form) into shared tables; both warp groups run every tile's epilogue (half
the channels each); accumulators (2 x [p | g], 256 TMEM columns) and t double buffered. Forms: `fold_op` cached with the D64 inference
forms (`integrations.trimul_b200._inference_params`). Served at every length (`b200_infer.k3hf_ok`: D64, hidden 64 / 128 one direction,
64 bidirectional). Rotated A/B (mean of 4 L2-flushed captures, us, k3h or k3g -> k3hf): hidden 128 L128 20.69 -> 20.36 / 21.23 -> 20.36,
bidirectional D64 L128 21.19 -> 20.41, L256 41.76 -> 38.84; D64 L256 32.91 -> 31.46 / 32.78 -> 31.04, hidden 128 L256 40.99 -> 38.66;
L384 51.27 -> 49.87, hidden 128 71.27 -> 66.04, bidirectional 75.90 -> 71.22; L512 79.69 -> 75.92 / 115.27 -> 107.52 / 119.72 -> 112.24;
L768 165.6 -> 154.9 / 245.9 -> 229.6 / 249.2 -> 232.7; L1024 315.0 -> 297.0, bidirectional 530.3 -> 504.0; L1536 837.7 -> 807.6 /
1465.7 -> 1443.0 / 1478.3 -> 1447.6; L2048 1659 -> 1597; L3968 9050 -> 8860, bidirectional 16925 -> 16717. Accuracy (`_tools/foldchk.py`,
plain and masked, L128 / L256, RF3 too): rms vs fp64 lower than k3h (3.18e-3 -> 3.05e-3 at hidden 128 L128), max within the same
2.8-3.8e-2, graph == eager, three runs bitwise.

**Kept 2: the D64 front on the LN_in-folded k1w** (pass 14's FOLD with D64 allowed: a 4-slot weight ring and a 2-slot x ring to fit the
correction boxes; forms: `front_fold` cached with the k3hf fold, empty for RoseTTAFold3, which keeps the plain front) in
`K1W_FOLD64_CELLS` (one direction L256 hidden 64 / 128, L384 hidden 64): hidden 128 L256 38.27 -> 38.00 / 38.27 -> 36.90, D64 L256
31.02 -> 30.63 (against k1h), L384 49.86 -> 49.03. Level at L128 (hidden 128 / bidirectional 20.36 / 20.34) and bidirectional L256;
slower at hidden 128 / bidirectional L384 and L768 (+1-3 %): not served there.

**Kept 3: k3w multiplies x before its griddepcontrol.wait (XF, k3w flags bit 4) with ctr triggering it after ctr's own wait (ctr
pdl bit 2).** With OST k3w is a programmatic dependent of the contraction and waited before every load; now a CTA's first item loads
and multiplies its x / Wg' stages first, waits, then streams the t stages (the stage order is mapped in the producer, MMA and statistics
warps alike: the same bits). Reading x before the wait is only safe when every kernel before the contraction has completed, so ctr
triggers its dependents right after its own wait (k1w complete) instead of at entry; behind cuBLAS (no early trigger) the order is the
same as before. Rotated A/B (us, bitwise): D128 L128 23.02 -> 22.41 / 24.26 -> 22.44, bidirectional 28.43 -> 27.55; not at L256+ (D128
46.97 -> 47.37, bidirectional 70.36 -> 71.14, D256 L384 184.8 -> 186.3): `XF_MAXL` 128.

**Kept 4: ctr for one direction D256 / D384 / D512 and bidirectional D384 at L128** -- measured level in passes 5-7 behind cuBLAS; with XF
the k3w CTAs that land on SMs a retiring ctr CTA frees do their set-up and x half meanwhile (rotated A/B, us, base / XF alone / ctr + XF):
D256 L128 36.15 / 36.01 / 34.81 and 36.09 / 36.16 / 34.82, D384 51.48 / 51.97 / 51.25, D512 74.97 / 74.05 / 73.72, bidirectional D384
78.53 / 78.74 / 73.60, RF3 D256 127.8 / 127.6 / 125.9; bidirectional D256 (ctr already) 47.05 / 46.99 / 46.98.

**Measured, not taken:** ctr at L256 one direction (hidden 128 38.12 -> 38.10 / 37.48 -> 37.63, D64 30.65 -> 30.59 / 30.70 -> 30.61).

**Not built, with the reason.** (1) the LN_in fold inside `k1h`: the k1h transform is ~0.5 us of its timeline (x landed 1.79 -> xn 2.32,
accumulator 2.63) and the cells it serves (D64 L128 hidden 128 / bidirectional) need 4-6 us (7-8 timer ticks against 10); the D64
folded k1w measured level there. (2) D128 L128 one direction (22.4 us, needs 19.1-19.4 = 9 ticks): its chain is k1w 7.6 / ctr ends 4.9
later / k3w 6.5 after the contraction (t lands, 2 stages, ~1.3-1.8 us epilogue); an 8-CTA cluster one-launch kernel needs 15 resident
8-clusters at least (`tf_clusters`: 15 is the maximum) and 2 x the D64 kernel's operand traffic, which pass 10 already measured at 8
ticks for D64. (3) D256 L128 (34.7 us, needs 16 ticks = 32.8): k1w 14.1, ctr ends 5.5 later (256 items on 148 CTAs: two waves), k3w 11.4
after it; no inter-CTA exchange is left on this path for a packed `st.async`, and the contraction is ctr now (no cuBLASLt pick).

**After** (`_out/spd_p15.jsonl`: cuda + pytorch, every eval case of the three classes, each length its own exclusive run; triton of
`spd_st12` / `spd_st14` / `spd_st15`; `_out/ana_p15.txt`): 1.5x misses 15 -> 13 of 172 -- fixed bidirectional D128 L128 (28.5 -> 26.6 us,
0.676 -> 0.629) and bidirectional D64 L256 (40.9 -> 38.8, 0.679 -> 0.644); closer: D256 L128 36.7 / 36.8 -> 34.7 (0.688 / 0.702), hidden
128 L256 40.8 / 42.3 -> 38.7 / 38.8 (0.701 / 0.704), D64 L256 33.6 / 32.7 -> 32.4 / 30.6, bidirectional D64 L128 22.4 -> 20.4 (0.754),
D128 L128 24.4 -> 22.4 (0.771 / 0.781), hidden 128 incoming L128 22.4 -> 20.4 (0.847). None slower than the H100, ladder 0. Geomean new /
old over the 172 eval cases 0.980 (58 faster by > 3 %); five read 3-5 % slower, all on paths this pass does not change (D256 incoming
L1024 / L2048, RF3 D128 L2048, bidirectional D512 L640, bidirectional D384 L512: k1w / cuBLAS / k3w(g) at L > 128). `tests/gpu/
test_trimul_b200.py` 168 passed; `measure.accuracy` eval L128-1024 of the three classes 119 / 119 GREEN (`_out/acc_p15.jsonl`).

### Round 3, sixteenth pass: the contraction inside the output kernel (2026-11-18, agent r3-trimul; rejected, no route change, `cuda:b200_sm100` stays 56)

**Built, measured, rejected: k3c** (diff `docs/b200/rejected/trimul_p16_k3c_contraction_in_output.diff`). The L128 D128 chain k1w -> ctr ->
k3w as two launches, k1w -> (contraction + output): `k3w_kernel` with a CT template flag, 128 CTAs (one token tile each, all resident),
CTA c computes ctr's item c (channel c's whole 128 x 128 plane: its A / B rows after griddepcontrol.wait on the front, ctr's eight
MMAs into TMEM columns 256-383, ctr's epilogue through a staging region, TMA store) and releases a monotonic counter of its 64-channel
block (red.release.gpu after the store's completion; base read at entry, +512 a launch); the output's t stages wait on those counters
(ld.acquire) instead of a launch boundary, while x / Wg' stages and the Wp' halves load before the wait (k1w got a "trigger after
its own wait" flag, bit 13, so x is safe to read early); 3-slot ring + a 64 KB contraction region (206 KB). Bitwise equal to ctr + k3w
(outgoing / incoming / RF3, plain and masked; graph == eager; three runs bitwise; `_tools/foldchk.py` CHK_ATTR=K3C).
Speed (rotated A/B, mean of 2-4 L2-flushed captures, us): D128 L128 22.45-22.68 -> 22.39-22.47 / 22.63-22.71 -> 22.41-22.44 (medians 22.4
both: one ~2 us timer step); CUPTI span (k1w start -> last end) 18.85 -> 18.50. Refinements, all level: eight-lane operand boxes with the
two K halves on two barriers, the t stages issued by four lanes behind their own counters, a relaxed counter add. In-kernel timeline
(`_tools/k3ctl.py`, -DK3W_TL, us after the first CTA's entry, median / max over CTAs): front done 4.6 / 5.1, operands landed 5.9 / 6.4,
accumulator 6.4 / 7.2, store issued 6.9 / 7.4, store complete 7.4 / 7.9, counter released 8.2 / 9.0, t blocks seen 9.2 / 9.7, p
accumulator 11.0, y stored 12.8 -- the same 4.9 us from the front's end to t as the separate ctr (k1w end -> ctr end 4.9, pass 15):
the launch boundary was not the cost; the all-to-all exchange of t through L2 (store + completion, release, slowest-CTA skew, reload)
is, and the fused form keeps it. Two contraction items a CTA (bidirectional D128, D256 one direction) measured slower (26.6 -> 28.6,
34.9 -> 38.4 / 34.8 -> 37.9) and raced (the second item's accumulator shares TMEM with the first output item's p, whose block counter
completes before the second item is read). The same fusion on k3hf (D64 / hidden 128 / bidirectional D64) not built: the D128 result
does not carry over.

### S90 round, inference (2026-11-18, agent trimul-inf, `cuda:b200_sm100` 58 on TriangleMultiplication / RF3 / Bidirectional)

Goal of the round: SOL% (t_sol / t_measured, `docs/b200/sol_trimul.md` calculator and `cases.csv` t_sol column) towards 90 %, eval, every case of the
three classes (172 cases). Tools: `_tools/s90_bw.py` / `s90_seg.py` (measured HBM floors), `_tools/var.py` + `s90run.sh` (nvcc-define ablations,
CUPTI per-kernel device time inside the measurer's graph replay), `_tools/ab_il.py` (interleaved A/B in one process), `_tools/s90chk.py` (bitwise / fp64),
`_tools/full.sh` + `soljoin.sh` (the grid and its SOL join). No counters (ncu is blocked): every claim below is a timing.

**Where the SOL% is lost (before; latest eval records = `spd_p15` of the SOL study, re-measured on the pass-16 tip as `_out/spd_base_*`).**
Median SOL% by (D, L): L128 D64 4-7 %, D128 9-18 %, D256 28-38 %, D384 41-52 %, D512 50-57 %; L256 D64 11-18 %, D128 23-32 %;
L384-768 D64 18-51 %, D128 32-60 %; L1024+ D64 43-59 %, D128 54-60 %, D256+ 63-74 %. The lowest groups:
1. **L128 / L256, D64 / D128**: the contraction-in-chain latency, not a resource: D64 L128 (one launch, `tf_kernel`) 16.3 us graph = 12.1 us kernel span
   + 4.1 us graph launch while t_sol is 0.7 us (SOL study per-kernel table); D128 L128 22.4 us: k1w 7.6 / ctr ends 4.9 us later / k3w 6.5 us after the
   contraction (pass 15 timeline) and pass 16 measured the 4.9 us as the t exchange through L2 (store + completion, release, skew, reload), kept by the
   fused form. Floors measured by earlier passes; nothing new built here (an 8-CTA cluster one-launch kernel measured 8 ticks at D64, rejected in pass 10).
2. **L384-L768 D64 (18-51 %)**: the three stages are at the *measured streaming floor* of their own bytes: L512 D64 front 26.0 us vs a Triton
   1-read-2-write kernel of the same bytes 27.0 us, cuBLAS contraction 16.2 us, k3hf 30.4 us vs the 2-read-1-write floor 29.1 us (all with ~4 us of launch in
   the floor): 72.6 us of floor against 75.6 us measured. The remaining distance to t_sol is the structural a / b / t round trip (`t_struct` of the SOL study,
   5 -> 6 C + 3 D bytes per token), which only a fused front + contraction + output (rejected in the first pass) removes. No change possible here.
3. **L1024+ D64 / D128 (43-60 %)**: the front kernel's plane stores (found below) and the output kernel's on-chip chain (not changed, below).

**Finding: the k1w plane stores wrote 64-byte rows.** A warp's [64 ch][32 tok] SW64 TMA box puts 64 contiguous bytes per channel plane (row stride L*L*2 bytes).
Measured write throughput of a plane store as a function of the contiguous segment (Triton streaming store of [64 ch][T tok] boxes, L2 flushed, 20 reps median;
`_tools/s90_seg.py`): L1024 P128 (268 MB) 64 B 61.3 us / 128 B 48.6 / 256 B 46.7 / 512 B 47.6; L1024 P256 111.0 / 86.0 / 83.9 / 83.9; **L2048 P128 (1074 MB) 337.5 / 185.8 /
158.8 / 160.9 us (3.2 TB/s at 64 B vs 6.7 TB/s at 256 B)**. Reads of the same boxes: L2048 64 B 327.5 us, 128 B 178.3, 256 B 172.7 (so the output kernels'
128-byte t reads are fine). Ablations of the front at D64 L1024 (CUPTI device time, us; `ABL_*` defines, removed again): base 102.3, no TMA plane store 76.1, tanh
replaced by the identity (no MUFU) 101.1, no LayerNorm statistics 101.6, no affine transform 100.7, no epilogue arithmetic (stores kept) 101.3,
no epilogue arithmetic and no store 63.2, nothing but the x read + MMA 29.3. D128 L1024: 202.6 / no MUFU 201.4 / no store 157.9 / neither 153.4. So the
transform, the statistics and the MUFU are not on the critical path of the front; the 64-byte stores are (25 us of 102 for the store alone at D64 L1024).

**Kept: planes as 128-byte rows (`k1w.cu` flags bit 14, host `TRIMUL_K1W_WST`, inference calls only).** The epilogue's two warps of a pair (tokens 64 p .. 64 p + 63) write
their stmatrix fragments into one [64 ch][64 tok] SWIZZLE_128B staging box (8 KB, chunk ^ (row & 7)), join on a named barrier (ids 2 + 2 p, 3 + 2 p), and the first warp issues
one TMA store of the 128-byte-row box (it waits for the previous store's smem read before the pair restages). The inference callers pass grid -2 (D <= 128) / -1 (D >= 256, as before);
training (grids > 0 / sample-major planes) and D >= 384 are untouched: D384 / D512 measured slower with it (front D384 L1024 1082 -> 1178 us, D512 L512 409 -> 466 us; the pair
join couples the warps that run one chunk apart there) and stay on the per-warp 64-byte boxes. Same bits: output y bitwise equal with the flag on and off (`_tools/s90chk.py`, 18
cases x plain / masked: D64 hidden 64 / 128 L128 / 384 / 512 / 640 / 1024 / 2048, D128 L256 / 1024, D256 L384 / 640, D384, D512, bidirectional D64 L512 / 1024, D128 L384, D256 L768,
RF3 D64 L512 / D128 L256; random non-zero weights, real shapes, vs the fp64 PyTorch module: max / rms unchanged, e.g. D64 L1024 3.18e-2 / 2.70e-3), graph == eager, three runs bitwise.
`tests/gpu/test_trimul_b200.py` 168 passed (before and after).

Interleaved A/B (one process, 5 rounds alternating the flag, the measurer's graph timing, L2 flushed, mean of the round medians, us): D64 L1024 TriMul 298.0 -> 273.9 (0.919), RF3 300.0 -> 276.3 (0.921),
D128 L1024 613.7 -> 575.2 (0.937), D256 L1024 1478.7 -> 1447.6 (0.979); level (0.99-1.01) at D64 L512 / L640 (TriMul and RF3), D128 L640 / L768 (RF3 D128 L640 too), D256 L640, bidirectional D64 L640, D128 L256 / L512, D128 L128.
Front kernel alone (CUPTI): D64 L1024 102.7 -> 79.0, L2048 547.7 -> 328.0; D128 L1024 204.9 -> 169.5, L2048 1192 -> 763.6; D256 L768 291 -> 277,
L1024 565.5 -> 514.2; bidirectional D64 L512 44.8 -> 39.9, bidirectional D128 L1024 403 -> 369.

**Grid** (`_out/spd_base_*` = flag off, `_out/spd_new_*` = default, each length its own exclusive run, 172 eval cases; single runs: cases whose code did not change (D >= 384, L <= 768)
differ by up to +-5 % run to run, RF3 mid L up to +-10 %, power state of the shared card): L1024 / 1536 / 2048 one-direction D64: 296.9 / 808 / 1636 -> 276.4 / 704 / 1517 us;
D128: 614 / 1754 / 3539 -> 588 / 1625 / 3196 us. SOL% (t_sol / t, pure) over the 172 cases, before -> after: TriMul 4.5 / 53.4 / 74.0 -> 4.5 / 57.0 / 74.3 (min / median / max, 106 cases),
RF3 3.6 / 33.2 / 58.1 -> 3.6 / 32.5 / 64.2 (36), Bidirectional 6.8 / 57.1 / 76.3 -> 6.8 / 57.1 / 73.5 (30), all 3.6 / 50.6 / 76.3 -> 3.6 / 52.5 / 74.3. Cases at >= 90 %: 0 -> 0; >= 70 %: 17 -> 15
(noise on unchanged D >= 384 cases); >= 50 %: 92 -> 95. 31 cases faster by > 5 %, 13 by > 10 % (all D64 / D128 L >= 1024).

**Back kernels (`k3hf`, `k3g`), measured, not changed.** At D64 L1024 k3hf runs 103 us against 72 us for a 2-read-1-write streaming kernel of the same bytes (D128 L1024 k3g 213-224 vs 130 us; D64 L2048 k3hf 440 vs
243 us). Not the y store: without the TMA store k3hf 103.9 -> 96.8 us, k3g 224.5 -> 213.3 us (-5 %). Not the t reads (128-byte rows read at full rate, above). The cause is the per-CTA chain: k3hf's LN_out warps run
LN_out(i) -> MMA -> epilogue(i) -> LN_out(i + 1) serially (a deeper tile pipeline deadlocks with one xo / accumulator buffer, see the code comment), two CTAs per SM overlap two chains: 1.87 us per tile per CTA
at 64 tokens. Removing it needs a double-buffered xo / accumulator (a rewrite of k3hf / k3g); a real gain of up to ~10 % at L >= 1024 D64 / D128 is left on the table.

### S90 round, inference, part 2 (2026-11-18, agent trimul-inf, `cuda:b200_sm100` 60)

**D384 / D512 plane stores (diagnosed with measurements).** The part-1 note that 128-byte plane stores were slower at D384 / D512 (front 1082 -> 1178 us, 409 -> 466 us) was run-to-run
noise of single runs (the same non-interleaved comparison read 1209 -> 1202 / 449 -> 421 us in the next run). Interleaved A/B in one process, 5 rounds, flag on / off (`-DWST_MAXD=512` build, `_tools/abil.sh`;
mean us, new / old): D384 L384 1.004, L512 0.986, L768 1.004, L1024 0.985; D512 L384 1.002, L512 1.003, L768 1.008; bidirectional D256 / D384 / D512 L512 1.006 / 0.995 / 1.007. Level: the wide fronts are tensor /
weight-stream bound (36-50 % HBM, MMA-limited), not store-limited, so the store layout does not matter there. The hypothesis "warp-pair join couples the warps" was tested with a variant that moves the wait-for-previous-store
and pair barrier from before the TMEM loads to before the first stmatrix (`KW_LATEA`: accumulator released earlier): D384 L512 front 261.4 -> 265.1 / 249.9 (D <= 256 build), D512 L512 420.7 -> 424.4 / 430.3: level. Diff archived
(`docs/b200/rejected/trimul_s90_k1w_wst_maxd_latea.diff`); the gate stays D <= 256.

**Output kernel `k3hf` (D64): what limits it, measured.** The kernel was already double buffered (x, t and the [p | g] accumulators, 256 TMEM columns); the claim that a single buffer is the cause is wrong. Tests (D64 L1024, kernel device time, us):
one accumulator buffer at two CTAs an SM 103.95 -> 103.60 (no change: accumulator buffering is not what limits). Ablations: epilogue arithmetic removed 80.3, LN_out statistics removed 92.8, both removed 76.6 (floor of the same bytes 72):
the arithmetic / statistics chain is 27 of the 104 us. clock64 timeline of tile 10 of a CTA (`-DTM_TL`, `_tools/tlf.py`, cycles from the LN_out warps' wait for t): statistics 653, LN_in / LN_out join 258, acc_full 274, fences + x_full 182,
epilogue arithmetic and shared stores 1306, barrier + store 210; the MMA of the tile had been issued ~1400 cycles earlier (t landed ~1700 before the statistics start). So there is no MMA / TMA starvation and no deadlock to remove: the
per-tile chain is issue / latency of the warps themselves, and the way to hide it is more resident warps, not deeper buffers. Built: **three CTAs an SM** (`k3hf_kernel<C, H, CPS = 3>`, D64 hidden 64: one 128-column accumulator buffer
per CTA = 384 TMEM columns, 53 KB of shared memory a CTA, 444 CTAs; hidden 128 keeps two, its shared memory does not fit three; four CTAs an SM measured 133 us: registers spill). Kernel 103.95 -> 96.61 us at L1024.
Interleaved A/B (env `TRIMUL_K3HF_CPS=0` = old), mean us old -> new: D64 L384 49.0 -> 48.2, L512 76.0 -> 74.1, L768 156.4 -> 152.5, L1024 272.5 -> 265.3, L2048 1407 -> 1346, L640 124.0 -> 120.2 (0.956-0.983); level at L128 / L256 (one wave) and
hidden 128 / bidirectional D64 (unchanged kernel). Bitwise equal to the two-CTA kernel (`_tools/s90chk.py`, TriMul L128-2048 plain / masked, hidden 128, RF3 L512, bidirectional L384, graph == eager, three runs).

**RF3 (class SOL 32 %).** Not a kernel gap: RF3 runs the same `k1w` / contraction / `k3hf`; same-shape device times equal (L640 D64: 34.8 / 36.1 / 44.5 us against 35.5 / 36.3 / 44.7 us). Its one extra launch is `forms_kernel` (3.4 us at L512,
3.85 us at L640, already a PDL primary that triggers dependents at entry), which is 15-20 % of the L128 call (20.4 against 16.3 us). The class median is low because its rows are D64 / D128 only (no D >= 256). Removing the launch means applying
the length factor inside the three front kernels (tf / k1h / k1w) on the accumulator instead of in the bf16 weights: not the same bits, three code paths; not built.

### S90 round, inference, part 3 (2026-11-18, agent trimul-inf; RF3 `cuda:b200_sm100` 62, TriMul / Bidirectional stay 60)

**RF3 length factor inside the front kernels (adopted, RF3 62).** The per-call `forms_kernel` launch (3.4-3.9 us, 15-20 % of the L128 call) existed to fold f = bf16(1 / max(n, 1)) into W_l. The front kernels
(`tf_kernel` D64 L128, `k1h`, `k1w`; flag `rf3` on the three) now multiply the left projection's accumulator by f before the gate product (planes bf16(sigmoid(g) (f p)), left chunks only); the token mask's kept-token count is made
in the kernel (every CTA counts the L mask bytes with `__syncthreads_count`, the same `1.0f / n` then bf16 rounding as the forms), the constant 1 / L for no mask; RF3 reads the plain cached forms (no `trimul_forms_rf3` launch). Training keeps the forms path.
Basis of the accuracy claim (`_tools/s90chk_rf3.py`, random non-zero weights, real shapes, RF3 D64 / D128 L128-1024, plain and 30 %-masked, fp64 PyTorch reference, forms-folded vs inline): rms vs fp64 equal or lower
(e.g. L1024 D64 masked 2.425e-3 -> 2.354e-3, L128 D64 3.108e-3 -> 3.108e-3), max vs fp64 within +-0.3e-2 of the forms path (<= 1.5 x + 1e-3 rule), inline-vs-folded max 3.1e-2 (one bf16 ulp of y), graph == eager, three runs bitwise; 20 / 20 OK.
`test_b200_rf3_factor_in_forms[*-masked-eval]` compared bits with the `_LeftScaled` path: it now allows max 8e-2 / rms 4e-3 for inference (power-of-two factors still match bitwise); training bits unchanged; 174 tests pass.
Interleaved A/B (5 rounds, mean us, forms -> inline): D64 L128 19.9 -> 17.4 (0.873), L256 32.6 -> 30.6, L384 51.5 -> 49.8, L512 76.9 -> 73.2, L640 118.7 -> 115.5, L1024 270.2 -> 262.1; D128 L128 28.5 -> 26.5 (0.930), L256 52.3 -> 49.0,
L384 90.8 -> 88.0, L512 147.4 -> 142.9, L768 307.2 -> 302.3. RF3 now costs what TriMul costs at the same shape (D64 L128 17.4 vs 16.3 us). env `TRIMUL_B200_RF3=0` restores the forms launch.

**`k3hf` split into statistics warps and epilogue warps (`k3hs`): built, measured, rejected.** Warps 4-7 / 8-11 compute the statistics up to two tiles ahead (double-buffered tables freed by `sti_free` / `sto_free`), warps 12-15 (one epilogue group, all channels) or 12-19
(two groups, half the channels each) run only the epilogue: the statistics of tile i + 1 overlap the epilogue of tile i; 512 / 640 threads, two CTAs an SM. Kernel time (us): D64 hidden 64 L1024 108.6 (one group) / 110.2 (two) against 103.9 (k3hf, two CTAs) and 96.7 (k3hf, three CTAs);
L512 31.9 / 32.5 against 30.4 / 28.5; L2048 479 / 483 against 440 / 417; hidden 128 L1024 126.9 / 130.9 against 124.6. Slower: the 64 / 51-register budget of the larger CTA costs more than the overlap gives; with the three-CTA kernel the same effect (more resident warps) is
already taken. Diff archived (`docs/b200/rejected/trimul_s90_k3hs_split_stats_epilogue.diff`). Remaining: k3hf 96.7 us at D64 L1024 against 72 us of streaming; 27 us of arithmetic + statistics issue (epilogue arithmetic removed -23.5, statistics -11).

**PDL prologue overlap (audit, no new work).** Every kernel of the inference chain already does its independent work before `griddepcontrol.wait`: k1w (barrier init, tensor-map prefetch, TMEM allocation; its x / weight fills are after the wait only because it can be the first kernel of the graph),
`ctr` (own kernel at L128: triggers dependents at its entry, a programmatic dependent of k1w), k3hf / k3h / k3g (x and the folded weights loaded before the wait, t after), k3w (x, Wg' halves before the wait: XF). The contraction itself is cuBLAS and does not trigger early
(only its tail overlaps the output kernel's prologue). No further overlap found without replacing cuBLAS.

**Not built:** the D128 / D256 one-launch cluster kernel and the contraction fused into the output kernel's front half (pass 16 measured the t exchange through L2 as 4.9 us of the chain, which a fused form keeps).

### S90 round, inference, part 4 (2026-11-18, agent trimul-inf, `cuda:b200_sm100` 64 on all three classes)

**k1h 128-byte plane rows: built, measured, rejected.** The M = 64 front (`k1h`, cells hidden 128 L128, bidirectional D64 L128 / L256) stores [64 ch][16 tok] SW32 boxes (32-byte rows); the four warps of an epilogue group (q = 0 .. 3) were made
to fill one [64 ch][64 tok] SW128 box (named barriers 2 + 2 g / 3 + 2 g). Bitwise equal (`_tools/s90chk.py`, 5 cases x plain / masked, graph == eager, three runs). Interleaved A/B (5 rounds, mean us, old -> new): hidden 128 L128 20.4 -> 20.3 / 20.3 -> 20.4,
bidirectional D64 L128 20.4 -> 20.4, **L256 38.8 -> 40.8 (+5.2 %)**: not adopted (`docs/b200/rejected/trimul_s90_k1h_128b_rows.diff`). At these sizes the front is latency / launch bound (1-2 tiles a CTA), not write-segment bound.

**Fused front + contraction (+ output) for L128 / L256 D64 / D128: not buildable, from the arithmetic of the existing design.** The D64 L128 one-launch kernel (`tf_kernel`, 128 CTAs in 16 clusters of 8) gets its t without an L2 exchange of t only by
RECOMPUTING the front: a cluster's 16 x 32 token block needs a[:, 16 rows, 128 k] and b[:, 32 rows, 128 k] = 3072 tokens of front projections against the 128 tokens it owns (24x redundancy; 12.1 us kernel). At L256 the same block needs 6144 tokens (48x): the front is 8 N D C
= 2.1 GFLOP of useful work, 100 GFLOP with redundancy = ~70 us of tensor time at the measured 1458 TFLOPs, against 30.6 us for the three-kernel chain. D128 L128 (hidden 128): the front is 4x the FLOPs at the same 24x: ~35 us against 22.4 us. The t of an output tile
needs all channels, and a per-channel contraction GEMM needs tiles of M >= 64 pairs: an output-tile-resident contraction (the "contraction inside the output kernel's front half") runs [4 i-rows x L] x [L x 16 j] per channel: M = 4 padded to 64 = 16x the contraction FLOPs
(L512 D64: 17.2 GFLOP -> 275 GFLOP = ~190 us against cuBLAS 16.3 us). The measured 4.9 us (pass 16) of t exchange through L2 therefore stays: it is cheaper than any form of redundancy.

**`k3hf` epilogue instructions.** The residual add y = bf16(x + bf16(sigmoid(g) p)) took 7 instructions per channel pair (two unpacks of the x word, the unpack of the rounded product, fadd2, pack); it is now `add.rn.bf16x2` of the x word and the packed product (the exact
fp32 sum of two bf16 values rounded once: the same bits except at the double-rounding ties of the old form). k3hf D64 L1024 96.63 -> 93.59 us (single runs, same process state; the epilogue arithmetic is 23.5 us of the kernel). fp64 error of the module outputs unchanged on every case
of `_tools/c_chk3.txt` (e.g. L1024 D64 3.177e-2 / 2.697e-3 max / rms before and after); 174 tests pass. LN_out statistics via the tensor core (ones column) not built: the sums of squares need t^2, which the MMA cannot form, so at most the mean pass (~half of the statistics, ~5 us) moves.

### S90 round, inference, part 5 (2026-11-18, agent trimul-inf; no version change)

**Own contraction kernel (`ctr`) for L256-512 vs cuBLAS: built to the (a)-(d) list, measured, not adopted (`TRIMUL_CTR_ALL=1` enables it for D <= 128, L <= 512; default off).**
`ctr.cu` already was the design asked for (persistent, a programmatic dependent of the front that prefetches its tensor maps and waits for the front only before its first plane load, triggers its dependents at entry / after its wait, TMEM accumulate, t stored as the output
kernel's [H ch][tokens] rows in 128-byte-row boxes, one TMA box per tile). Changes made now: (1) the 4 boxes of a stage issued by 4 lanes (6 at BN 256) instead of one thread: kernel D64 L256 9.19 -> 8.90 us, L384 16.3 -> 15.4, L512 29.5 -> 28.6; (2) a BN = 256 item
(M128 x N256, A / B operand traffic per output element 25 % lower; 6 boxes a stage, 2 stages, epilogue warps loop two column halves; L % 256 == 0): L256 D64 8.45 / 8.83 us, L512 25.97 us; (3) two CTAs an SM with one stage: worse (L256 9.85, L512 37.96 us).
Outputs bitwise equal to cuBLAS's (`_tools/s90chk.py` TRIMUL_CTR_ALL on / off: 9 cases x plain / masked, graph == eager, three runs). Against cuBLAS in the graph (CUPTI kernel time, us): L256 D64 **8.45 vs 5.40**, L384 15.4 vs 9.7, L512 26.0 vs 16.25. Interleaved module A/B
(5 rounds, mean us, cuBLAS -> ctr): D64 L256 out 29.8 -> 29.4 (0.985), in 30.6 -> 30.6, L384 47.8 -> 49.0 (1.025), L512 75.2 -> 77.7 (1.034); hidden 128 L256 37.7 -> 36.7 (0.975), L512 108.4 -> 114.5 (1.056); D128 L256 level, L384 1.053, L512 1.045.
Why it stays behind: operand traffic. At L512 D64 the 512 items read (128 + 256 rows) x 512 k x 2 B = 393 KB each = 201 MB in 26 us = 7.7 TB/s, the L2 -> SM rate; the MMA minimum is 13.6 us. cuBLAS (`nvjet ... 2cta`) runs 256-row tiles with cta_group::2 (B shared by two SMs), reaching 1055 TFLOPs (16.3 us). Matching it needs
cluster multicast / cta_group::2 in this kernel (a rewrite of the producer / MMA / epilogue protocol, measured a loser as a plain TMA multicast earlier in the brief); not built. ctr BN = 256 stays the default where L % 256 == 0 (bidirectional L256 cells: module level, 1.000 / 1.000 vs BN 128; kernel 5 % shorter);
output bits unchanged, so no version bump.

**First access after the L2 flush, per kernel (measured).** CUPTI device time of every kernel in a graph replay with / without the measurer's L2 flush before each replay (us, flush -> none): k1w D64 L256 10.14 -> 9.37, D128 L128 7.83 -> 7.57, D128 L256 16.44 -> 16.35; k3hf D64 L256 10.43 -> 10.10;
k3w D128 L128 10.51 -> 10.55; ctr D128 L128 11.01 -> 11.08; cuBLAS contraction L256 5.53 -> 5.56, L512 16.45 -> 16.60; at L512-1024 no kernel differs (k3hf 93.93 -> 93.84, k1w 80.93 -> 80.48). So the cold-access cost is 0.3-0.8 us per kernel at L <= 256 (a front's first x / weight boxes and the output kernel's first t box), nothing at
L >= 512: a prefetch issued by the previous kernel could win at most that (3 % of the 29 us D64 L256 call), and the kernels already load their independent operands (x, weights) before `griddepcontrol.wait`. Not built.

### S90 round, inference, part 6 (2026-11-18, agent trimul-inf, `cuda:b200_sm100` 66 on all three classes)

**`k3g` (D128 output) timeline and fixes.** `-DK3G_TLG` stamps %globaltimer (ns, to a global array, `k3g_tlg()`; `_tools/tlg.py`) per CTA for tiles 10 / 11. D128 L1024, tile 10, us from the tile's first t box issue: t issue 0, LN_out sees t 6.4 (it was
still on the previous tile), x issue 6.1, LN_in sees x 7.2, MMA xo_ready 9.5, epilogue acc_full 9.7, y store issued 12.5, x slot free 13.3; tile period 4.35. Reading: an x slot lives 7.2 us (issue -> free) and there are two, so the period cannot be under 3.6 us; the x load after the slot frees costs 1.0 us (HBM) in that chain; the t ring (two
halves = one tile) only started tile i + 1's t after tile i's LayerNorm had read it.
Built (kernel us, D128 outgoing, CUPTI in the graph, rotated builds): (1) the residual as one packed `add.rn.bf16x2` (the k3hf cut): L1024 228.8 / 216.2 -> 212.0 / 213.4; (2) four tri-tile half slots instead of two (`K3_NTH_BIG`; the next tile's t lands during this tile's LayerNorm, smem fits at C = H = 128 only): L768 122.0 -> 116.5, L1024
216-229 -> 206.4-207.1, L1536 534 -> 493-517, L2048 955 -> 924-927 (single runs, +-3 %); three slots 216.0 (L1024); (3) an L2 prefetch (`cp.async.bulk.prefetch.tensor.2d.L2`) of the next tile's x boxes issued with the current tile's x load (`K3_PFX`): L768 116.4 -> 113.8, L1024 207.4-218.1 -> 205.5-208.8, L1536 501-510 -> 497-503
(-1 to -3 %). Module medians (ms): L768 0.2980 -> 0.2949, L1024 0.5527 -> 0.5385, L1536 1.4391 -> 1.356. Outputs identical to the previous build (fp64 error of the module unchanged on 14 cases of `_tools/c_chk7.txt`, graph == eager, three runs; the packed add differs from the old pair only at double-rounding ties); 174 tests pass.
Remaining at D128 L1024: k3g 207 us against the 130 us streaming floor; the chain per tile (LN_in / LN_out 3.1 us after the loads, epilogue 2.8 us) with two x slots bounds the period at 3.6 us (measured 4.35); a third x slot does not fit (3 x 32 KB + 64 KB t + weights > 227 KB).

**`k3hf` L2 prefetch of the next x tile (K3HF_PFX): measured level, off.** D64 L512 27.78 vs 27.85 us, L1024 93.79 vs 94.19, L1536 222 vs 214 (noise); the D64 tile is 8 KB and the latency is covered by the double buffer.
**Not built in this part:** the `k3w` (D256-512) timeline and its prefetch; `k1w` next-tile prefetch.

**Full inference grid re-scored (172 cases, `_out/spd_now_*`, each length its own exclusive run; before = part 1's `spd_base_*`, the tip before the S90 changes; k3hf prefetch off, as shipped).** SOL% (t_sol / t, pure): TriMul 4.5 / 53.4 / 74.0 -> 4.5 / 58.3 / 74.3 (min / median / max, 106 cases), RF3 3.6 / 33.2 / 58.1 -> 4.0 / 33.9 / 70.1 (36), Bidirectional 6.8 / 57.1 / 76.3 -> 6.8 / 56.8 / 76.0 (30), all 3.6 / 50.6 / 76.3 -> 4.0 / 52.9 / 76.0.
Cases >= 90 %: 0 -> 0; >= 70 %: 17 -> 19; >= 50 %: 92 -> 96. Lowest 15 (SOL %, us before -> now) and the measured blocker of each:
RF3 D64 L128 in / out 3.6 -> 4.0 (20.4 -> 18.3): tf one-launch kernel 12.1 us span + 4.1 us graph floor + (RF3 now without the forms launch) ; TriMul D64 L128 4.5 (16.2 / 16.3 us): the same kernel, 24x front recomputation (24 x 2.1 GFLOP... 12.1 us); bidirectional D64 L128 6.8 (20.4) and hidden-128 D64 L128 6.8 (20.3-20.4): k1h 6.9 + ctr 6.1 + k3hf 10 us chain, t exchange through L2 4.9 us (pass 16);
RF3 D128 L128 9.0 -> 9.7 (28.5 -> 26.5): k1w 7.8 + ctr 11.0 (kernel) + k3w 10.5, overlapped to 22-26 us; TriMul D128 L128 10.5 / 10.6 (24.4): same chain (front 7.6 / contraction ends 4.9 later / k3w 6.5 after, pass 15); D128 incoming L128 11.4 -> 10.6 (22.5 -> 24.3 in this single run; interleaved A/B read 22.4 vs 22.5, run noise);
D64 L256 (RF3 30.6, TriMul 28.5-28.6, SOL 12.0-12.9): k1w FOLD 10.1 + cuBLAS 5.5 + k3hf 10.4 us with 0.3-0.8 us of first-access cost per kernel (measured, part 5): the sum, not one kernel, is the floor.

### S90 round, inference, part 7 (2026-11-18, agent trimul-inf, `cuda:b200_sm100` 68 on all three classes)

**`k3w` (D256-512 inference output) timeline (`-DK3W_TL`, %globaltimer marks per CTA, `_tools/k3tl.py`).** D256 L1024 (two channel items per token tile, 55 tiles a CTA): per item the epilogue runs from acc_full to the y store in 3.07 us (TMEM load done +0.26, first shared store +0.25 after that,
epilogue arithmetic and shared stores to the end 2.3 us, store issue 0.25); the item period is 4.8 us (534 us / 110 items); the MMA of an item (p: K = H, g: K = D, M128 x N128) is 1.7 us at the tensor peak: the kernel is epilogue-latency bound (8 epilogue warps, 32 channels
per step), as the earlier passes found for k3g / k3hf. D128 L128 (the critical-path body, one item per CTA): entry -> first stage landed 1.5 us (cold weight / t / x boxes), stages + MMA to acc_full +1.3 us (K = 512), acc_full -> y store issued +1.8 us (TMEM load 0.0-0.26, arithmetic 1.5): body 4.6 us from entry (the 10.5 us kernel duration includes the PDL wait for `ctr`);
the g half (x Wg') runs before t lands (XF), so after t lands the chain is the 2 p stages (0.65 us) + the epilogue 1.8 us + the t box latency (~1.3 us L2 round trip). Splitting the item N-wise over two CTAs halves the MMA (0.65 -> 0.3 us) and the epilogue arithmetic per CTA (1.5 -> 0.8 us) at the price of a second t / x box set per token tile (the t exchange through L2 doubles its readers): bound on the gain ~1.3 us of 22-26 us; not built.
Built: the residual as one packed `add.rn.bf16x2` (inference path only, `!SAVE`; the training save kernel's code is untouched): kernel us (rotated builds, two runs each) D256 L1024 578.5 -> 560.0 / 560.0, D384 L768 519.5 -> 488.7 / 485.8, D512 L512 354.8 -> 349.7 / 349.3, D256 L512 126.8 -> 124.3. Tried: 16 epilogue warps (`K3W_EPW=16`): slower (609.6 / 584.2 / 366.5 us). fp64 error of the module identical to the pre-change build on D256 L384, D384 L256, D512 L256, bidirectional D256 L768 (plain and masked), 174 tests pass.
`k1w` next-tile prefetch: not built; the front's x ring is already 4 slots (4 tiles ahead at D64, 2 at D128, 1 at D256 where the front is 70 % tensor bound). `k3w` residual x prefetch: the single residual tile is loaded while the item's stages stream (xr_full is never the last to land in the timeline), no gain expected.

### S90 round, inference, part 8 (2026-11-18, agent trimul-inf; analysis only, no version change)

**Is inference `k3w`'s MMA serialised with its epilogue? No: both TMEM accumulators are in use, and the MMA is not waiting for anything; it runs about twice as slowly while the epilogue runs.** Column map: `tmem + b * 256` (p at +0, g at +128, b = item & 1): 512 columns, double buffered; the MMA waits `acc_empty[b]` only for item i - 2.
Steady-state stamps (`-DK3W_T2`, %globaltimer, CTA stamps of items 10 / 11, `_tools/t2.py`; us from the item's first stage issue): D256 L1024 item 10: MMA first stage landed 3.58, last stage landed 6.14, MMA commit issued 6.40, LN_out statistics done 6.91, epilogue sees acc_full 10.50, y store 14.08; item 11: first stage 8.19, last 10.50, commit 10.75, stats 10.50, acc_full 15.62, y store 18.43; item period 5.12 us.
D384 L768: acc_full of items 10 / 11 at 6.66 / 11.26 (4.6), period 5.12 (two items of a different shape share the period figure only by timer granularity).
(a) one accumulator buffer: no; (b) the residual tile: xr is seen before acc_full; (c) the next tile's statistics: done before the MMA of the same item ends; (d) the commit is issued at 6.40 and the accumulator completes 4.1 us later: the MMA PIPE is busy for 4.1 us after its last issue, not the issue thread; (e) ring: the last stage of the item lands at 6.14, before the commit.
Ablations (kernel us, D256 L1024 / D384 L768 / D512 L512; base 539.7 / 483.2 / 342.6): no epilogue arithmetic (TMEM loads skipped, accumulators still released) 398.3 / 451.9 / 322.1; no LN_out statistics reads 513.4 / 481.2 / 330.6; neither 374.2 / 435.8 / 311.4; no weight-box loads 499-506 / 470-478 / 320;
A operand K-major instead of MN-major (wrong data, timing only) 560.6 / 475.6-499.8 / 344.2 (no change). So the skeleton (loads + MMA + store) alone is 374 us = 3.4 us per item at D256 (the ideal 1.7 us at the tensor peak: M128 N128 K16 SS instructions read 8 KB of shared memory every 64 clk and rewrite a 64 KB accumulator each);
with the epilogue running next to it the period is 5.1 us instead of max(3.4, 3.07): the MMA pipe slows by ~1.5x (4.1 us commit -> acc_full where the skeleton needs ~2.6) while 8 epilogue warps load TMEM (128 KB per item) and read / write the residual tile in shared memory. A second accumulator buffer does not exist to add; a wider item (N = 256: one A read for two channel halves, 25 % fewer operand bytes and half the accumulator rewrites per FLOP) needs 512 columns for ONE buffer (p 256 + g 256) and so gives up the overlap: estimated 3.4 us of MMA + 6.1 us of epilogue per tile, no gain.
No built fix: every cheap variant (EPW 16, K-major A, fewer loads, no stats reads) either costs time or gains at most 7 % (no weight loads, not a real configuration).

### S90 round, inference, part 9 (2026-11-18, agent trimul-inf, `cuda:b200_sm100` 70 on all three classes)

**`k3w` epilogue off shared memory: built, measured, rejected (`K3W_GREG=1`).** The residual x chunks are read from global (L2) into registers before the TMEM wait (4 x 16 B per 32 channels of the thread's row) and y is stored from registers (`st.global.v4`, 64 contiguous bytes a row per step); no residual TMA, no staging tile, no store barrier. Kernel us (D256 L1024 / D384 L768 / D512 L512), shared-memory
epilogue -> global-register epilogue: 544.3 / 495.2 / 345.1 -> 702.4 / 700.2 / 358.8 and 498.2 / 496.5 / 348.7 -> 700.6 / 680.7 / 391.6: **+30-40 % at D256 / D384, +3-12 % at D512**. The residual load latency (an L2 round trip a 32-channel step) and 64-byte partial-line row stores cost far more than the shared-memory traffic they save: the staging tile's
traffic (32 KB lds + 32 KB sts + the TMA read) is small next to the fold-vector reads.
**Where the epilogue's shared-memory traffic is: the per-channel fold vectors.** Each channel pair read four 64-bit broadcasts (sp, ep, sg, eg) = 8 B x 4 a pair, 1024 wavefronts per item for 8 warps against ~560 for the residual tile and 2048 clocks of MMA operand reads. Now one 128-bit broadcast pair per vector per 8-channel chunk (8 loads instead of 16 per chunk, `K3W_NOVS8` restores the old form): kernel us D512 L512 341.1 / 352.5 -> 324.2 / 324.5, D256 L1024 536.7 / 525.8 -> 537.3 / 561.7 and D384 L768 494.5 / 502.4 -> 495.6 / 522.3 (single runs, +-5 %: level).
Bits unchanged (same arithmetic): fp64 error of the module identical on D256 L384, D384 L256, D512 L256, bidirectional D256 L768 (plain and masked), graph == eager, 174 tests pass.
Not built: the A-from-TMEM (`.ts`) variant (LN_out writing the normalised A into TMEM). For `k3w` the A operand of the p GEMM is the raw t tile (the LayerNorms are folded into the weights), so there is no producer warp that could write it into TMEM without a copy (tcgen05.cp of a 128 x 256 B stage is itself a shared-memory read of the same bytes the MMA would read).

### TriMul inference: blockers per case group (tip `cuda:b200_sm100` 70, 2026-11-18)

Grid: `_out/spd_t70_*` (every length its own exclusive run, 172 eval cases, cuda, graph, L2 flush), joined with the SOL study's t_sol (`soljoin.sh`). SOL% pure, before (`spd_base`, the tip before S90) -> now: TriMul 4.5 / 53.4 / 74.0 -> 4.5 / 57.5 / 74.4 (min / median / max, 106 cases), RF3 3.6 / 33.2 / 58.1 -> 4.0 / 34.7 / 66.6 (36), Bidirectional 6.8 / 57.1 / 76.3 -> 6.8 / 57.1 / 73.7 (30), all 172: 3.6 / 50.6 / 76.3 -> 4.0 / 52.5 / 74.4; cases >= 90 %: 0 -> 0, >= 70 %: 17 -> 23, >= 50 %: 92 -> 95.
H100 ratio (cuda B200 / H100 > 0.667, H100 >= 20 us): 13 of 172, all at L128 / L256 (hidden-128 D64 L128 / L256 0.70-0.85, D128 L128 0.78-0.84, D64 L128 0.77-0.80, D256 L128 0.69-0.70, D64 L256 0.68-0.69, bidirectional D64 L128 0.75), none slower than the H100; ladder (cuda < triton < pytorch, 2 %): 0 violations (triton / pytorch records of pass 15, unchanged kernels).
Run-order artefact: in the grid RF3 is measured right after the TriMul cases of the same length; at L640-1536 it reads 10-25 % slower than TriMul of the same shape (L768 D64 172 vs 151 us; the previous grid's RF3 read 225 us) while interleaved A/B in one process measures them equal (L640 118.7 vs 117.9, L1024 276.3 vs 273.9); the RF3 mid-L SOL is under-reported by that amount.
Plane bytes: `k1w` writes 2 H planes per token (left | right), the gate multiply already applied (plane = bf16(sigmoid(g) p)); there are no separate gate planes, so the a / b round trip cannot be halved by folding the gate; fp16 planes are the same bytes; t is bf16. The structural floor of the three-stage form is the study's `t_struct`.

| group (D, hidden C; L) | cases | SOL % now | measured blocker | evidence | attempted (result) |
|---|---|---|---|---|---|
| D64 C64 L128 (TriMul, RF3) | 4 | 4.0-4.5 | one-launch `tf_kernel` 12.1 us span + 4.1 us graph floor; 24x front recomputation (3072 tokens of front for 128 owned) | per-kernel table (SOL study), arithmetic in part 4 | fused cluster kernel (pass 9-10), RF3 forms launch removed (part 3: 20.4 -> 18.3 us), PDL audit (part 2), 128 B rows (no effect at 1-2 tiles) |
| D64 C128 / bidirectional D64 L128 | 3 | 6.8 | k1h 6.9 + ctr 6.1 + k3hf 10 us chain; t exchange through L2 4.9 us | pass 15 / 16 timelines | k3hf (pass 15), k3c contraction inside the output (level, rejected), k1h 128 B rows (part 4: bidirectional L256 +5 %, rejected), fused form not buildable (part 4 FLOP count) |
| D64 L256 (C64, C128, bidirectional) | 7 | 12.0-19.0 | k1w FOLD 10.1 + cuBLAS 5.5 + k3hf 10.4 us; 0.3-0.8 us first access per kernel | part 5 per-kernel + flush / no-flush | custom ctr (part 5: 8.45 vs 5.4 us, module +0.0-2.5 %), k3hf 3 CTAs (level at L256), k3hs split (slower), k1h rows (slower), cold-access prefetch measured <= 3 % |
| D64 L384-768 | 20 | 20-50 | three kernels each at the measured streaming floor of its own bytes (L512: front 26.0 vs 27.0 us Triton 1R2W, k3hf 28 vs 29.1 us 2R1W, cuBLAS 16.3 us) | part 1 microbenchmarks | fused front + contraction + output (part 4: 16x tensor FLOPs / 24-48x front recompute), custom ctr (part 5, slower: L2 -> SM 7.7 TB/s vs cuBLAS 2-CTA), k3hf 3 CTAs (-2.6 %, adopted), plane gate fold (already folded, nothing to fold) |
| D64 L1024-2048 | 18 | 36-71 | k3hf 94 us vs 72 us streaming at L1024 (arithmetic 23.5 us + LN_out statistics 11 us issue-bound), k1w stores (fixed), cuBLAS contraction (excluded) | part 2 ablations / timeline | k1w 128 B plane rows (adopted: -8 % module), k3hf 3 CTAs (adopted), packed bf16 add (adopted), k3hs split (slower: 108-110 us), EPW / prefetch (level) |
| D128 L128 | 5 | 9.7-18.1 | chain k1w 7.8 + ctr (body 4.8) + k3w body 4.6 (entry 1.5, MMA 1.3, epilogue 1.8) | part 6 / 7 timeline | XF, ctr, k3w fold (passes 13-15), N-split of the single tile estimated <= 1.3 us (not built), RF3 forms removed (part 3: 28.5 -> 26.5) |
| D128 L256-768 | 30 | 24-61 | k1w 16.4 + cuBLAS 7.7 + k3w 19 us (L256); k3g 53 us (L512) | part 5 / 6 kernel times | custom ctr (slower), k3g packed add + prefetch (adopted, -2..-7 %), 4 tri slots (level), WST (level at <= 768) |
| D128 L1024-2048 | 12 | 57-67 | k3g 207 us vs 130 us streaming: two x slots (x slot life 7.2 us -> period 4.35 us), 3.1 us LN + 2.8 us epilogue chain | part 6 %globaltimer timeline | k3g NTH 4 / prefetch / packed add (adopted), 3 x tiles (training merge, level), k1w WST (adopted, -4..-6 %) |
| D256 L128 / D384 L128 / D512 L128 | 6 | 27.6-57 | k1w weight intake 36-41 B/clk per SM (D256 L128: 512 KB in 14.1k cycles; part 10 %clock64 marks) + cuBLAS/ctr 5.5 + k3w 11.4 us | part 10 timeline | third weight issuer, FOLD front, XF, ctr cells (passes 13-15), k3w packed add (adopted); N-split infeasible (one CTA an SM, 128 tiles / 148 SMs), cluster multicast measured 2-6x slower than the plain stream (part 11 microbenchmark); grouped 32 KB weight boxes built: module -4..-9 % |
| D256-512 L256-640 | ~25 | 44.8-69 | front 70-73 % TC; k3w epilogue-latency bound: MMA pipe 1.5x slower beside the epilogue (skeleton 3.4 us / item, with epilogue 5.1 us) | part 8 ablations (374 -> 540 us at D256 L1024) and %globaltimer stamps | EPW 16, K-major A, GREG (all slower), fold-vector 128-bit broadcasts (adopted, D512 -5 %), packed add (adopted, -3..-6 %) |
| D256-512 L768-3968 | ~40 | 59-74 | same (k3w / cuBLAS at 1.3-1.5 PFLOPs); the contraction is cuBLAS at 79-92 % of peak (excluded) | SOL study ctr column | as above |
| RF3 rows (D64, D128) | 36 | 4-67 | as the same-shape TriMul rows (identical kernels) + run-order artefact above | part 3 (same-shape kernel times within 3 %) | forms launch removed (part 3: -2.5 us at L128, -3..-6 % mid L) |

Cases below 70 %: every group above has an attempt. S90 attempt on the single-tile output chain at L128 (the last open entry): 16 epilogue warps in `k3w` (`K3W_EPW=16`, the single tile's epilogue arithmetic split twice as wide): module D128 L128 22.4 -> 24.4 us, D256 L128 34.7 -> 36.7, D384 L128 51.1 -> 53.2 (two runs each, identical): +2 us, rejected; the N-split over two CTAs (estimated <= 1.3 us of 22-26 us) is not built. The groups whose only attempts predate S90 (D256 / D384 / D512 at L128-L256 front weight-streaming, passes 13-15) stay as listed.

### S90 round, inference, part 10 (2026-11-18, agent trimul-inf; analysis, no version change)

**`k1w` front at D256 / D384 / D512 L128 / L256: weight intake per SM (`-DTM_TL` clock64 marks of tile 0, `_tools/k1tl.py`).** D256 L128: x landed 5176 cycles after entry (setup + cold box), the last MMAs of tile 0 issued at 19297 cycles, i.e. 14.1k cycles for the 32 weight K-blocks (16 KB each = 512 KB: every CTA streams the whole 2 P x D weight matrix) = **36 B/clk** per SM (~70 GB/s);
D512 L128 (2 MB of weights a CTA): MMAs issued at 51587 cycles = 41 B/clk; D256 L256: 20458 (the same stream for two tiles: 36 B/clk). The tile's compute (MMA, epilogue) hides under the stream: the front is **weight-intake bound at the per-SM TMA rate** (the one-thread-one-TMA-per-600-clk finding; three issuing lanes already, pass 13).
Why an N-split of the tile (each CTA a quarter of the output channels) does not shorten it here: L128 has 128 token tiles for 148 SMs and the kernel holds 192 KB of shared memory (6 x 16 KB weight ring, 4 x 16 KB x ring, 32 KB staging), so one CTA fits an SM: an S-way split means 128 S CTAs in ceil(128 S / 148) waves (S = 2: two waves of half-length streams = the same stream time, plus a second launch ramp); it only pays if 2 CTAs share an SM, which needs a ring of <= 56 KB per CTA (no room for a weight ring that hides the 1.3 us box latency) and 2 x 512 TMEM columns do not exist.
The lever the timeline points at is the opposite one: the same weight boxes are read by all 128 CTAs, so a 2-CTA cluster with TMA multicast (each CTA issues half of the boxes and both receive all of them, cross-CTA empty barriers) halves the boxes each CTA must issue and the L2 reads: if the per-SM limit is the issue rate the stream drops from 7.4 us to ~3.7 us at D256 L128 (13.8 -> ~10 us of the 34.7 us module) and 11 us at D512. Not built (new multicast helpers, a cluster launch for the front and the cross-CTA slot protocol in a 900-line kernel with 3 weight issuers); the earlier brief note that multicast lost was for the HBM-bound mid-L regime, not this issue-bound one.
**`k3w` / `k3g` / `k3hf` single-tile output chain at L128 (N-split of the item over 2-4 CTAs).** The k3w body at D128 L128 is 4.6 us from entry (part 7). An N-split to two CTAs needs NTILE = 64 items: 256 CTAs for 148 SMs and 160 KB of shared memory a CTA (5 x 32 KB ring): two waves, unless the ring shrinks to 3 x 24 KB and the accumulators to one 128-column buffer so that two CTAs share an SM (the TMEM, shared memory and 512-thread budgets all change: NTILE is a compile-time constant in 34 places of the kernel). Gain bound: MMA 1.3 -> 0.7 us and epilogue arithmetic 1.5 -> 0.8 us per CTA = ~1.3 us of 22 us; not built.
Attempted instead at L128 (part 10 measurement): 16 epilogue warps (+2 us, see the table note). A first ablation flag `K1W_ABL_NOW` (no weight boxes, `expect_tx(0)`) hung the kernel (the diff is kept as `docs/b200/rejected/trimul_s90_k1w_abl_now_hangs.diff`; the stuck process was killed by PID).

### S90 round, inference, part 11 (2026-11-18, agent trimul-inf, `cuda:b200_sm100` 72 on all three classes)

**Weight-stream microbenchmark (`_tools/wmc.cu` / `wmc.py`, 128 CTAs, one CTA an SM, every CTA streams the same 512 KB through a ring of stages with `cp.async.bulk`, consumer spin 0; B/clk per SM, warm L2 / cold L2).**
Plain (cluster 1): stage = box 16 KB: 28.5 / 27.8 B/clk (ring 4, 6 or 8 alike: 18380 cycles); the same 16 KB stage split into 2 / 4 / 8 boxes issued by 2 / 4 / 8 lanes: 26.2 / 22.5 / 17.5 B/clk (**more issuing lanes are slower**); stage 32 KB (one 32 KB box): 53.4 / 50.0; two 16 KB boxes of the 32 KB stage: 49.5; 4 / 8 boxes: 43 / 34;
48 KB box: 75.2 / 67.9; 64 KB box (ring 3): 88.7 / 69.1; 96 KB (ring 2): 91.9 / 69.1; 128 KB (ring 1): 64.3 / 48.6; ring depth 2 / 3 / 4 / 6 at 32 KB: 51.5 / 53.0 / 53.4 / 54.2 (cold 34.8 / 45.1 / 50.0 / 51.1). With 600 clk of consumer work a stage: 32 KB 30.0 B/clk, 64 KB 53.8, 96 KB 70.6.
So a TMA op costs ~590-740 clk per SM whatever its size and whichever lane issues it (the per-SM op rate, not the issue lanes, is the limit), and the stream saturates near 90 B/clk at 64-96 KB per op: **the right lever is fewer, larger ops**.
Multicast (cluster 2 / 4, each CTA issues 1 / cs of the boxes with `.multicast::cluster`, cross-CTA empty barriers through `mapa` + remote `mbarrier.arrive.release.cluster`): 16 KB stage 9.2 / 7.2 B/clk, 32 KB stage 17.8 / 13.3 B/clk: **2-6x slower than the plain stream** (a cluster-scope slot handshake per stage, ~1.8k cycles a stage whatever the ring depth). Verdict: multicast is not built; its cost is the handshake, not the L2 reads.
**Built from it: grouped weight boxes in `k1w`** (flags bit 15, inference only, packed forms): one 3-D TMA op ([k in block 64][row][2 K-blocks] = two consecutive 16 KB slot images) per two K-blocks; ring barriers per group (full 1, empty gw arrivals); `TRIMUL_K1W_GW=0` restores the 16 KB boxes. gw = 3 (48 KB, D384) failed with an illegal-address fault and is not used; gw = 4 does not fit the ring (6 slots at D256 / D384 / D512).
Bitwise equal to the 16 KB-box front (`_tools/s90chk.py`: D128 L128 / L256, D256 L128 / L256 / L512, D384 L128, D512 L256, bidirectional D256 L128 / D128 L256, plain and masked, graph == eager, three runs). Interleaved A/B (5 rounds, mean us, old -> new): D128 L128 24.4 -> 22.5 (0.919), L256 49.0 -> 47.1 (0.960); D256 L128 40.8 -> 38.8 (0.950), L256 111.1 -> 105.2 (0.947), L512 381.0 -> 360.3 (0.946);
D384 L128 61.8 -> 59.3 (0.959), L256 200.9 -> 187.7 (0.935); D512 L128 86.0 -> 79.3 (0.922), L256 290.8 -> 266.2 (0.916); bidirectional D256 L128 57.2 -> 53.5 (0.935).
Not taken from the microbenchmark: 64 KB groups (the weight ring would need 8 slots: 128 KB, no room with the x ring and staging) and a bigger k1w weight ring for D256 / D512.

### S90 round, inference, part 12 (2026-11-18, agent trimul-inf, `cuda:b200_sm100` 74 on all three classes)

**Regression found and fixed (mine, part 11).** A re-run of the grid at tip 73 read D256-512 L128 / L256 8-22 % slower than at tip 70 (geomean 1.135), also with `TRIMUL_K1W_GW=0`. Cause: the weight-group size reached the kernel as a runtime value from the flag word, so the MMA and loader loops did integer divisions (`s / gw`, `q % gw`); `k1w` D256 L256 52.9 us, D512 L256 146.7 us against 40.3 / 118.9 before. Now `gsh` / `gmk` (shift, mask; groups of 1 or 2 K-blocks): D256 L256 37.1 us, D384 L128 22.7, D512 L256 108.6.
Grid at tip 74 (`_out/spd_t74_*`, 172 cases, each length its own exclusive run): SOL% min / median / max TriMul 4.5 / 58.7 / 75.9, RF3 4.0 / 33.9 / 67.6, Bidirectional 6.8 / 56.1 / 76.7, all 4.0 / 52.8 / 76.7; >= 90 %: 0, >= 70 %: 34 (17 at the start of S90, 23 at tip 70), >= 50 %: 95; geomean t74 / t70 0.993 (L128 / L256 0.991, D256-512 L128 / L256 0.988); H100 ratio > 0.667: 13 (all L128 / L256), ladder 0 (pass-15 triton / pytorch records).
**The 48 KB group crash.** 3-block (D384, gw 3) and 4-block (D512 with an 8-slot ring, gw 4) groups fault ("unspecified launch failure"); gw 2 (32 KB) works everywhere. The same fault in two shapes points at the tensor-box size rather than the ring protocol (barrier counts, phases and the correction slot are the gw 2 code); a 32 KB tensor box is the working maximum here, a 48 / 64 KB one is not (non-tensor `cp.async.bulk` of 64-128 KB works, microbenchmark). Not investigated further.
**TMA box-size audit (box <= 16 KB and < 40 B/clk per SM on a critical-path stream).** k1w: x 16 KB boxes, weights 32 KB (now), plane stores 8 KB (SW128 [64 ch][64 tok]) / 16 KB; ctr: 16 KB boxes (3 / 4 / 6 per stage); k3w: t 8 KB x 2 + Wp' 16 KB, x 16 KB + Wg' 16 KB per stage; k3g: x / t 16 KB, y 16 KB; k3hf: t / x / y 8 KB (D64) per 64-token tile.
Rate per SM on the critical streams: k3g L1024 96 KB a tile per 4.35 us = 11.6 B/clk; k3hf 24 KB a tile x 3 CTAs per ~3.3k clk = 22 B/clk; k3w item 256 KB per 5.1 us = 26 B/clk; ctr L512 1.36 MB per 26 us = 27.6 B/clk (below the 28.5 B/clk ceiling of 16 KB ops: the only stream near it). Tested: ctr with 3 ops a stage (A one 3-D 32 KB op, B two 32 KB ops; stage bytes unchanged): 8.48 / 8.87 / 15.68 / 25.93 us against 8.45 / 8.83 / 15.4 / 25.97 (level; diff `docs/b200/rejected/trimul_s90_ctr_32kb_ops.diff`);
k3w with the two 8 KB t halves as one 16 KB 3-D op: D256 L1024 550 / 541 against 542 / 537 us (level; `trimul_s90_k3w_t3_merged_t_halves.diff`). So the op-rate rule applies to the L128 weight streams (done in k1w), not to these kernels, whose critical paths are the MMA pipe and the epilogue chains (parts 6-9). k3hf pair-tile boxes (two 64-token tiles one op) would need a four-slot ring and an adjacent-pair schedule: not built.
**D64 L512 byte accounting (TriMul D64 C64 out; N = 262 144 tokens; times CUPTI in the graph).**
front `k1w` 26.0 us: reads x 33.6 MB, writes 2 planes x 64 ch 67.1 MB = 100.7 MB = 3.9 TB/s; streaming floor of the same bytes (Triton 1R2W, event incl. ~4 us launch) 27.0 us, i.e. at its floor.
contraction cuBLAS 16.3 us: reads the 2 planes 67.1 MB + writes t 33.6 MB = 100.7 MB = 6.2 TB/s (the planes are L2-resident after the front: 67 MB of a 126 MB L2, so the HBM share is lower) and 17.2 GFLOP = 1055 TFLOPs, 72 % of the 1458 TFLOPs peak (FLOP floor 11.8 us): 38 % above its FLOP floor, at the byte rate of the HBM peak.
output `k3hf` 28.1 us: reads t 33.6 MB + x 33.6 MB, writes y 33.6 MB = 100.7 MB = 3.6 TB/s; streaming floor (2R1W) 29.1 us with launch: at its floor.
Sum 70.4 us of kernels (75.6 us graph) for 302 MB = 55 us at the 5.5 TB/s streaming rate, 49 us at the HBM read peak. No stage is more than 20 % above its own byte or stream floor except the contraction against its FLOP floor (38 %), which is cuBLAS (excluded); the group's SOL (27 %) is the structural a / b / t round trip (198 us vs 130 us of tensor time at L1024 in the study), not one slow stage.

### Round 3, training pass 2 (2026-11-17, `cuda:b200_sm100` 9, agent r3-trimul-train)

**The unified B1 at contraction width 128 (`b1s.cu` `b1v_kernel`, `b200_train.B1V`).** D64 with hidden 128 (one direction) and the
bidirectional D64 block contract over H = 128 channels; b1u's TMEM plan needs A [64 o][128 ch] plus two whole-tile dnr buffers of 128
columns there (704 columns), so they ran the ring b1s (front CTAs hand dPr to LN CTAs through L2, ring flags, cooperative launch; tri
read twice). A CTA pair (`cta_group::2`) does not help that budget -- a pair MMA doubles M (tokens) and keeps the per-CTA accumulator
columns -- so b1v keeps one CTA per tile and changes the plan instead:
- A^T = tri dPr [128 ch lanes][64 o] (M = 128 channels, the tri tile K-major as loaded, dPr the MN-major B operand): 64 columns and no
  padding rows;
- dnr in a two-slot ring of 64-channel halves (slot = half; the MMA of half 1 runs while the LN warps work on half 0, half 0 of the
  next tile while they work on half 1).
TMEM dWg 64 | A^T 64 | G 2 x 64 | Q 2 x 64 | dnr 2 x 64 = 512. Shared memory: Wp' (two 64-channel blocks) | Wg | dPr x 2 | 3 tri slots
of 32 KB | 2 (xn | dy) slots (219.5 KB). Front epilogue = b1u's (Q sums K = 128), LN backward = b1u's per 64-channel half with 1 / H,
H = 128; the LN tail reads A^T per (channel lane, 16 o) and sums dg_o / db_o over four column groups in a fixed order. Partials in
the ring b1s layout (grad_reduce unchanged). Same graph, A / B in one warmed process per length, B1V False / True alternating, three
rounds (`_tools/ab2.py`, lowest median, ms):

| case | L128 | L256 | L384 | L512 | L768 |
|---|---|---|---|---|---|
| D64 hidden 128 outgoing | 0.0873 -> 0.0810 | 0.1527 -> 0.1404 | 0.2572 -> 0.2346 | 0.4026 -> 0.3651 | 0.8624 -> 0.7871 |
| D64 hidden 128 incoming | 0.0874 -> 0.0791 | 0.1527 -> 0.1405 | 0.2572 -> 0.2346 | | |
| bidirectional D64 | 0.0954 -> 0.0872 | 0.1650 -> 0.1528 | 0.2715 -> 0.2490 | 0.4159 -> 0.3800 | 0.9002 -> 0.7974 |

y bitwise equal, dx / parameter gradients within 2.1e-3 .. 4.2e-3 (relative max) of the ring path, five backwards bitwise equal.

**The contraction backward in one launch (`b7m.cu` `cgb_kernel`, `b200_train.CGB_MAX_L` = 256).** The plane gradient was one cuBLAS
batched GEMM launch per product (two one direction, four bidirectional); cgb runs every product as 128 x 128 output tiles of one
persistent tcgen05 GEMM: a product group reads A / B from dt or the planes, K-major as stored or MN-major (the transposes are
descriptor modes, no copies), a 4-deep ring of 64-wide K blocks (16 KB A + 16 KB B), two TMEM accumulators (the next tile's MMAs during
this tile's epilogue), four epilogue warps fp32 -> bf16 -> SW128 staging -> TMA store; K summed in index order. Bitwise equal to the
cuBLAS products on every checked case (both layouts, all three directions). ms:

| case | L128 | L256 | L384 | L512 |
|---|---|---|---|---|
| D64 outgoing | 0.0667 -> 0.0648 | 0.1071 -> 0.1077 | 0.1712 -> 0.1752 | 0.2632 -> 0.2778 |
| D64 incoming | 0.0667 -> 0.0667 | 0.1076 -> 0.1076 | | |
| D64 hidden 128 outgoing | 0.0810 -> 0.0810 | 0.1404 -> 0.1404 | | |
| bidirectional D64 | 0.0891 -> 0.0810 | 0.1547 -> 0.1446 | 0.2500 -> 0.2517 | 0.3809 -> 0.4032 |
| D128 outgoing | 0.1138 -> 0.1117 | 0.1978 -> 0.1963 | 0.3392 -> 0.3473 | |
| bidirectional D128 | 0.1506 -> 0.1445 | 0.2888 -> 0.2838 | | |

From L384 cuBLAS's 192 x 128 two-CTA tiles win (the products are bandwidth-bound there and cgb re-reads each operand per 128-wide
tile): cgb serves L <= 256 only.

**b1u's tail and early PDL triggers.** A globaltimer probe per CTA (D64 L128, graph replay; experiment only, not kept) showed one tile
through b1u taking ~10 us after its `griddepcontrol.wait`, then a ~3.7 us tail, and 2-3.5 us of idle between each pair of kernels
(b1u -> cgb -> b7m) although the dependents are PDL launches: none of the training kernels let its dependents launch before it
retired. Two changes: b1u's `db_o = Wp^T s` comes from shared memory (F = Wp o s stored next to E in the tail; the 64-thread column
loop over Wp in global memory was most of the tail), and b1u / b1v / cgb call `griddepcontrol.launch_dependents` at entry (their
dependents cannot share an SM with them -- shared memory / registers -- so the early launch only puts the next grid's CTAs on the SMs
this grid leaves free and removes the launch latency at the boundary). Separate builds, alternating processes, two rounds (ms; "old"
= the pass-2 kernels before these two changes, from the A / B runs above):

| case | old | tail fix, no trigger | tail fix + triggers |
|---|---|---|---|
| D64 outgoing L128 | 0.0646 | 0.0646 | 0.0646 |
| RF3 D64 L128 | 0.0666 | 0.0647 | 0.0626 |
| D64 hidden 128 L128 | 0.0791 | 0.0789 | 0.0768 |
| bidirectional D64 L128 | 0.0811 | 0.0810 | 0.0789 |
| D64 outgoing L256 | 0.1056-0.1076 | 0.1016 | 0.0996-0.1009 |
| RF3 D64 L256 (both directions) | 0.1076-0.1096 | 0.1056 | 0.1036 |
| D64 hidden 128 L256 | 0.1404 | 0.1404 | 0.1384 |
| bidirectional D64 L256 | 0.1445 | 0.1460 | 0.1435 |

Measured and not taken (pass 2):

| experiment | numbers (ms) |
|---|---|
| the H100 Philox dropout scale (`b200_triattn.drop_scale`, one launch, bitwise the torch draw) for the TriMul route | D64 L128 0.0647 -> 0.0673, D128 0.1117 -> 0.1124, bidirectional D64 0.0829 -> 0.0816, RF3 L256 0.1096 -> 0.1084, D64 L256 0.1056 -> 0.1075: no gain (as in the fourth pass); the torch graph-RNG prologue fills stay outside the graph either way |
| output-side gradients (b1's partials) summed on a forked stream right after b1 (a parallel graph branch next to cgb / B7; the final grad_reduce over B7's partials only; `grad_reduce` region modes) | L128: D64 0.0646 -> 0.0687, D128 0.1116 -> 0.1138, bidirectional D128 0.1427 -> 0.1445, bidirectional D64 0.0830 -> 0.0851; L256 +0-2 us; L384+ neutral: the fork / join costs more than the overlap (removing grad_reduce altogether would save 0 / 7 / 4 us at D64 / D128 / bidirectional D64 L128) |
| b1v ring depths 2 tri / 3 (xn \| dy) slots instead of 3 / 2 | L384 hidden 128 0.2305 -> 0.2428, bidirectional 0.2490 -> 0.2634, L768 0.7732 -> 0.8327 (4 / 2 does not fit) |
| a `cta_group::2` pair for the H = 128 unified B1 | not built: a pair MMA doubles M (tokens per pair) and keeps the per-CTA accumulator columns, so it does not relieve the 704-column TMEM plan; the A^T orientation + half-tile dnr ring (b1v) does |
| D128: b1g / b7g as programmatic dependents (b1g also triggering cgb) | L128 D128 out 0.1077 -> 0.1097, in 0.1117 -> 0.1097, bidirectional 0.1423 -> 0.1404, RF3 0.1117 -> 0.1096 (within the 2 us timer step); L256 D128 0.1916 -> 0.1978, bidirectional 0.2828 -> 0.2857: not taken |

State after pass 2 (merged with `b200-2.5.x` e0455805e; two full training sweeps, one measurer process per length, `_out/spd_v9_{1,2}.jsonl`;
the cases above 0.667 re-timed with L128 / L256 in their own process, `_out/rem_v9r.jsonl`, and the L384+ ones one case per process,
because whole-length runs read the power-capped clock): training cases above 0.667 -- TriangleMultiplication 12 -> 10 (worst 0.919 ->
0.851), RF3 4 -> 2 (0.738 -> 0.682), bidirectional 4 -> 3 (0.895 -> 0.740); 20 -> 15. Every L384+ training case now passes (D64 hidden
128 L384 0.678 -> 0.604 / 0.613, bidirectional D64 L384 0.687 -> 0.627; one case per process: RF3 L384 0.606, D64 L640 0.636, RF3
L768 0.585, bidirectional D64 L512 0.585 / L768 0.538), and so do RF3 L256 (0.682 / 0.669 -> 0.656 / 0.650). No case slower than the
H100; ladder holds (cuda / triton <= 0.71, cuda / pytorch <= 0.24). Accuracy (`_out/acc_v9_*.jsonl`, every training row L128-768):
119 GREEN + 7 YELLOW, the same 7 cells as `cuda:b200_sm100` 7, no RED. Tests: 100 passed (new: b1v against the ring b1s and bitwise
repeatable over five backwards; cgb against the cuBLAS products in all three directions and both layouts, repeatable).

Launch count, D64 one direction L128 training step (graph, in order): RNG fills x 2 (replay prologue, outside the graph), rand,
drop_scale, forms, k1w, bmm, k3g | b1u, cgb, b7m, grad_reduce = 12 (was 13: two cuBLAS products -> one cgb). The H100 step
(`h100_uni_d64_b200gate`): pack / forms, K1, bmm, K3 | B1, bmm, bmm, B2, fin -- the same structure; the B200 step is 0.0626 ms against
the H100's 0.0745.

Remaining 15 training cases, by measured cause:
- D64 one direction L128 (2: 0.840 / 0.851; 0.0626 / 0.0636 ms against 0.0497 needed) and L256 (2: 0.729 / 0.730), D64 hidden 128 L128 / L256
  (4: 0.697-0.770), RF3 L128 (2: 0.681 / 0.682), bidirectional D64 L128 / L256 (2: 0.740 / 0.686): the forward (forms 2.6, k1w 6.3, bmm 3,
  k3g 7.5 us at D64 L128, the inference kernels) plus ~13 us of RNG / dropout launches leave ~30 us for the backward against 49.7 us in
  all; b1u alone is ~10 us for its single tile after its wait (load -> MMA -> front epilogue -> MMA -> LN backward), b7m ~13 us. Closing
  it takes the backward as one persistent kernel without the cooperative attribute (which itself cost ~3 us in the ring b1s) or M = 64
  tiles in every stage including the inference k1w / k3g -- not built. A forked-stream reduction and the Philox draw were measured and lose.
- D128 L128 one direction (2: 0.718) and bidirectional D128 L128 (1: 0.674): b1g / b7g ~29 / 30 us on one or two tiles per CTA with the
  ring hand-over; PDL for them was neutral (not taken); removing grad_reduce altogether would save 7 us (not reachable: the forked
  reduction loses 2 us).

### Round 3, training parts 3 and 4 (2026-11-17, `cuda:b200_sm100` 11 / 30, agent r3-trimul-train)

Part 3 (`cuda:b200_sm100` 11, a cancelled agent's three commits; re-measured before keeping them, A / B in one warmed process, three
rounds, ms): **b1w** (`b1g.cu`, the unified B1 at C = 128 for D128 one direction at L128: front epilogue and LN_out backward in one CTA,
dnr in G's TMEM columns, no ring / flags / cooperative launch) D128 out 0.1056 / 0.1097 -> 0.0954 / 0.0974 (min / median), in 0.1097 ->
0.0954, RF3 D128 0.1117 -> 0.0973; **the row-dropout scale made by the forms launch** (`DROP_IN_FORMS`, one launch fewer) D128 0.0964 ->
0.0944, bidirectional D64 0.0748 -> 0.0730, RF3 D64 0.0604 -> 0.0585, D64 L128 / L256 level; b1u / b7m partial stores coalesced. Kept.

Part 4 (`cuda:b200_sm100` 30):

| change | where | effect (ms, A / B in one process) |
|---|---|---|
| the training forward's contraction by `ctr.cu` (the inference route's kernel, bitwise cuBLAS; a programmatic dependent of k1w that triggers k3g) in `TRAIN_CTR_CELLS`: bidirectional L128 (D64, D128), one direction L128 at 128 contracted channels; one sample's sample-major planes are passed as the channel-major view | `b200_train._forward_small_body` | bidirectional D64 L128 0.0728 -> 0.0688, D128 0.1410 -> 0.1364 (both now pass), D64 hidden 128 L128 0.0708 -> 0.0693, D128 L128 0.0953 -> 0.0934, RF3 D128 0.0954 -> 0.0940 |
| `b7m`'s LN_in backward warps in the MMA fragment layout (`B7M_FRAG`): 4 lanes per row (16 columns each, x / dy by ldmatrix, dxn by tcgen05.ld 16x256b, dx by stmatrix), row sums by two xor shuffles, the dgi / dbi column sums kept in registers over every row and tile of the CTA and reduced once at the end (was one row per thread and a 32-lane shuffle tree per row and tile: an ablation of that tree alone measured D64 hidden 128 L256 0.1343 -> 0.1301, L128 0.0687 -> 0.0667) | `b7m.cu` | D64 L256 0.0976 -> 0.0933 / 0.0954, L384 incoming 0.1619 -> 0.1547, L512 0.2572 -> 0.2408, L640 0.4006 -> 0.3780, hidden 128 L256 0.1343 -> 0.1321, bidirectional D64 L256 0.1384 -> 0.1363, L640 0.5889 -> 0.5828; L128 level |

Role profile of `b7m` before the change (`PW` probes, `-DPROF=1`, `b7m_prof` now bound): the LN warps were the longest role at every
length (D64 L256: 35.5 kcycles, 30 % waiting; one tile at L128: ~7.5 kcycles after the dxn accumulator is ready, the kernel's tail).
After it, D64 L256: LN warps 33.4 kcycles (gate warps 28.9); at hidden 128 the gate / MMA chain (single pre-activation and dgp buffers at
NCH = 4, no shared memory left for a second) is now the longer one.

Measured and not taken: `ctr` in the cells one direction D64 L128 / L256, hidden 128 L256, D128 L256, bidirectional D64 / D128 L256
(level or up to 2 us slower, e.g. D128 L256 0.1957 -> 0.1977).

State (cuda training sweeps, one measurer process per length, three sweeps; the cases still above 0.667 re-timed three times in one
warmed process, `_out/rem_f.jsonl`; lowest median; triton / pytorch from the start-of-pass sweep `_out/spd_s0.jsonl`, which measured all
three backends; summary `_out/lad_fr.txt`): training cases above 0.667 -- TriangleMultiplication 8 -> 7, RF3 0 -> 0, bidirectional
3 -> 0 (11 -> 7); no case slower than the H100; ladder holds (cuda / triton <= 0.71, triton / pytorch <= 0.43). Accuracy
(`_out/acc_f_*.jsonl`, every training row L128-768): 119 GREEN + 7 YELLOW, the same 7 cells as the `b200_sm100` tokens with the same
errors (`_tools/acccmp.py`). Tests: 105 passed.

Sweep caveat: D64 chains read bimodally in long sweep processes (the programmatic-edge effect of the third pass): D64 L256 is
0.0974-0.0995 ms in a warmed process and in the first sweep of the pass, 0.110-0.114 in later sweeps of the same code (also before
this pass's changes: `_out/spd_s0r.jsonl`).

Remaining 7 (all TriangleMultiplication training): D64 L128 (2: 0.782 / 0.784; 0.0585 ms against 0.0497 needed), D64 hidden 128 L128 (2:
0.695 / 0.696; 0.0686 against 0.0658), D64 L256 (2: 0.685 / 0.686; 0.0974 against 0.0947), D64 hidden 128 outgoing L256 (1: 0.675;
0.1341 against 0.1324). Each L128 kernel is one tile per CTA (forward ~20 us of the inference kernels, then b1u ~9, cgb ~4, b7m ~11 incl.
the ~4 us LN tail, grad_reduce ~4 us); the L256 cells are within 2-3 % and move with the sweep's bimodality.

### Round 3, training part 5 (2026-11-17, `cuda:b200_sm100` 41, agent r3-trimul-train)

Start (three-backend sweep, one measurer process per length, `_out/s3_b0.jsonl`): 7 training cases above 0.667, all
TriangleMultiplication D64-family: D64 L128 x2 (0.758 / 0.755; 0.0565 ms), hidden 128 L128 x2 (0.707 / 0.698; 0.0688 / 0.0698), D64
L256 x2 (0.687 / 0.685; 0.0974), hidden 128 incoming L256 (0.669; 0.1328); ladder 0.

Graph timeline, D64 L128 (`_tools/gaps.py`, `GAPS_ALL=1`): the replay starts with the graph-RNG prologue (two 1.3 us elementwise
kernels) and a ~7 us launch gap before the first graph kernel; inside the graph: rand 2.1, forms 2.7, k1w 5, bmm 3, k3g 7.6 us (end to
end), then b1u +9.2, cgb +4.2, b7m +10.1, grad_reduce +3.7 us after the previous kernel's end.

**The fused small-L backward (`b7m.cu` `b7f`, built, bitwise equal, measured slower; compiled out by default: `-DB7F_BUILD=1`).**
b1u, cgb and b7m became device functions (`b1u_body` / `cgb_body` / `b7m_body`; the kernels call them, the same code and bits);
`b7f_kernel` runs them, then grad_reduce's arithmetic, as four phases of one persistent grid (one CTA per SM, 512 threads, TMEM
allocated once). No cooperative launch and no residency assumption: each phase's work is claimed from a global counter as virtual CTA
ids (the body runs as CTA v of a G-CTA grid, so tiles, partial rows and summation order are those of the separate kernels), a phase's
work is claimed only after the previous phase's claims ran out (deadlock-free with any number of resident CTAs), and dependencies are
completion counters (TMA stores waited, bar.sync, fence, atomic increment; consumer ld.acquire + fence.proxy.async). The counters live
in k3g's zeroed flag words and the last CTA out resets them. Between phases the MMA thread drains its tcgen05 ops and the phase's
mbarriers are invalidated. Every output bitwise equal to b1u -> cgb -> b7m -> grad_reduce (D64 L128 out / in, RF3; five calls bitwise
equal; `_tools/b7fchk.py`). Measured (ms, A / B in one process, `_tools/ab2.py`):

| variant (D64 outgoing L128) | ms |
|---|---|
| separate launches | 0.0564 |
| all four phases fused | 0.0749 - 0.0789 |
| b1u + cgb fused, b7m / grad_reduce separate | 0.0564 |
| b7m alone in the fused grid, all bodies compiled in | 0.0728 |
| b7m alone in the fused grid, only its body compiled | 0.0626 |

Phase marks (globaltimer per CTA, graph replay, L2 flushed, `-DB7F_TL=1`, `_tools/b7ftl.py`): b1u ~10 us (as the kernel), cgb ~7 us
(kernel 4.2), b7m ~22 us (kernel 10). Two measured causes: the four bodies in one kernel share one register allocation (128 at 512
threads, 44-48 bytes of spills against none in b7m_kernel<2, *>): ~10 us of b7m's phase; and b7m in the fused grid with only its own
code still takes 6 us more than the separate launch (claim / drain / release per item and the lost PDL overlap of b7m's weight loads
with cgb). Not taken. A fused small-L backward would need the phases' register budgets made equal first (b7m at 128 registers without
spills is already at the limit).

**Loads issued by several lanes (the coordinator's finding: one thread issues about one TMA per ~600 clk).** Separate builds, B7F off,
two alternating rounds, `_tools/j_ml2.sh` (ms):

| loader | D64 L128 | hidden 128 L128 out / in | D64 L256 | hidden 128 in L256 |
|---|---|---|---|---|
| none (single lane) | 0.0564 | 0.0687 / 0.0687 | 0.0954 | 0.1323 |
| b1u / b1v: tri halves, xn, dy from four lanes (`B1U_ML`, taken) | 0.0564 | 0.0667 / 0.0667 | 0.0934 | 0.1323 |
| b7m: weights, xn, dA chunk pairs from five lanes (`B7M_ML`) | 0.0565 | 0.0667 / 0.0667 | 0.0974 | 0.1342 |
| cgb: each K block's four loads from four lanes (`CGB_ML`) | 0.0564 | 0.0687 / 0.0687 | 0.0954 | 0.1343 |



Each row is that loader alone. b7m's multi-lane issue gives the same hidden 128 L128 gain as b1v's but costs 2 us at L256 (lane 0 now
waits both dA buffers of a chunk pair before xn goes out); cgb's costs 2 us at hidden 128 L256. Both are off (`B7M_ML` / `CGB_ML` 0).

State after part 5 (merged with `b200-2.5.x` 35282f48e; GPU tests 115 passed; three-backend training sweep one measurer process per
length `_out/s3_f1.jsonl`, cuda L128 / L256 twice more `_out/s3_f{2,3}.jsonl`, lowest median): training cases above 0.667 7 -> 6, all
TriangleMultiplication: D64 L128 in / out 0.782 / 0.759 (0.0585 / 0.0566 ms, need 0.0497), hidden 128 L128 out / in 0.697 / 0.696
(0.0687, need 0.0658), D64 L256 out / in 0.674 / 0.671 (0.0956 / 0.0955, need 0.0947 / 0.0948); hidden 128 incoming L256 passes. RF3 0,
bidirectional 0, no case slower than the H100; ladder 0 (cuda / triton <= 0.73, triton / pytorch <= 0.43). In the sweep processes
the b1u / b1v loader gain of the warm A / B (2 us at hidden 128 L128 and D64 L256) did not show: these cells sit on the graph timer's
~2 us step (values 0.0564 / 0.0585, 0.0667 / 0.0687, 0.0934 / 0.0955) and move by one step between processes.
What remains: D64 L128 needs 7-9 us off a 56 us step whose in-graph part is 47 us (forward 20 us, of it rand 2.1 and forms 2.7 before
k1w; backward 27 us in four kernels); the fused backward that would remove the three backward boundaries measured 18 us slower (above).

### Round 3, training part 6 (2026-11-17, `cuda:b200_sm100` 43, agent r3-trimul-train)

Start (merged `b200-2.5.x` 13e2c1ce8; three-backend training sweep, one measurer process per length, `_out/s3_b43.jsonl`): 6 cases above
0.667, all TriangleMultiplication -- D64 L128 in / out 0.782 / 0.756, hidden 128 L128 out / in 0.697 / 0.676, D64 L256 in / out 0.678 /
0.673; ladder 0, none slower than the H100.

Timeline of the D64 L128 step before (graph replay after the L2 flush, profiler, us from the flush's end; `_tools/tline.py`):
FillFunctor 2.48-4.51 and 6.37-8.06 (the graph RNG prologue: `CUDAGeneratorState::replay_prologue` fills the generator's graph-owned
seed and offset with two ordinary aten launches, 1.9 us of launch gap between them), 3.1 us graph-launch gap, rand 11.15-13.22 (the
dropout draw, torch's distribution kernel), forms 13.06-15.86, k1w from 14.69 ... grad_reduce to 58.51. 11 launches.

**The draw inside the forms launch (`PHILOX_IN_FORMS`, `forms_philox_op`).** The integration hands a `DeferredDraw` (shape only) to the
training call; `trimul_forms` reserves the default generator's Philox counter exactly as `torch.rand(n)` does (same increment, under the
generator's lock, graph-safe state inside a capture) and the forms kernel's dropout blocks compute torch.rand's element bits themselves
(`philox_rand_bf16`: torch's grid stride, subsequence = thread, component (e / S) % 4 of pass e / 4S, bf16 rounding with 1 -> 0), one
element per thread (eight per thread made the four draw CTAs the launch's tail: forms 2.7 -> 4.7 us). Where the forms come from the cache
the draw is made with torch.rand at that point (`realize_draw`); traced code keeps the torch draw. ds is bitwise the old path's and the
generator stands where torch.rand leaves it (tests: three draws in a row, a 655360-element draw over several passes, graph replays,
whole training steps out / hidden 128 in / RF3 / bidirectional D128 bitwise vs the torch draw). One launch fewer.

**The graph RNG prologue as one programmatic launch (`GRAPH_RNG_WRITER`, `wide_aux.cu` `rng_fill`).** For the scalars of a graph that
holds this draw (registered at capture, released with the graph through a CUDA user object) the CUDA `aten::fill_` is served here: the
seed fill is held, the offset fill writes both in ONE launch with the programmatic-serialization attribute (griddepcontrol.wait first,
so ordering is that of an ordinary launch). Every other fill goes to ATen. (The H100 TriAttn pack.cu overrides the same op with two
ordinary launches; whichever registers last serves the fills, the other's scalars then take ATen's fill: correct either way.)

A / B in one warmed process, alternating, three rounds (`_tools/ab2.py`, lowest / median ms):

| case | writer 0 (ATen) | 1 (one ordinary launch) | 2 (one programmatic launch) |
|---|---|---|---|
| D64 out L128 | 0.0570 / 0.0572 | 0.0559 / 0.0559 | 0.0560 / 0.0560 |
| D64 in L128 | 0.0589 / 0.0589 | 0.0559 / 0.0560 | 0.0559 / 0.0560 |
| hidden 128 out L128 | 0.0694 / 0.0703 | 0.0664 / 0.0672 | 0.0668 / 0.0673 |

| case | torch draw | draw in forms |
|---|---|---|
| D64 out L128 | 0.0563 / 0.0564 | 0.0559 / 0.0560 |
| hidden 128 out L128 | 0.0688 / 0.0696 | 0.0683 / 0.0683 |
| D64 out L256 | 0.0955 / 0.0965 | 0.0940 / 0.0949 |

In the profiler the programmatic writer starts under the flush (-0.4 us) and the step ends at 49.1 us after the flush (was 58.5); with the
measurer's events between the flush and the replay it gains only what the merged launch gains (mode 1 = mode 2 above): an event record
ends the programmatic overlap. Kept at 2 (a real training step has no event there).

**b7m's extra CTAs reduce the B1 partials (`B7M_XR`).** At L128 b7m runs one tile per CTA on 128 of 148 SMs; up to 32 extra CTAs (here
20) on the free SMs wait for the contraction backward (so the B1 kernel's partials are complete) and sum the output-side regions (dWg,
dWp, dg_o, db_o) with grad_reduce's arithmetic (16 warps, fixed order: the same bits, tested), and grad_reduce then runs only the dW1 /
LN_in blocks (`skip_o`). D64 L128 out 0.0553 -> 0.0552, in 0.0553 -> 0.0553, hidden 128 out 0.0678 -> 0.0675, in 0.0680 -> 0.0676
(medians): small; grad_reduce's tail after b7m stayed ~3.9 us -- it is latency-bound, not bandwidth-bound (more loads in flight,
`-DGR_U=8` / `-DGR_W=8 -DGR_U=16`, separate builds alternating: level or up to 0.7 us slower; not taken).

Timeline after (D64 L128): rng_scalar_write -0.40-1.54, forms 3.17-5.94, k1w 5.30-11.70, bmm 11.60-14.80, k3g 14.29-21.97, b1u to
31.09, cgb to 35.58, b7m to 45.55, grad_reduce to 49.09: 9 launches (was 11), the head before the first graph kernel 3.2 us (was 11.2).

State after (merged base, GPU tests 130 passed; sweep `_out/s3_a43.jsonl`): 6 cases above 0.667 -- D64 L128 out / in 0.754 / 0.752
(0.0562 / 0.0563 ms, need 0.0497 / 0.0499), hidden 128 L128 out / in 0.686 / 0.685 (0.0676, need 0.0658), D64 out L256 0.676 (0.0959;
0.0949 in the warmed A / B), hidden 128 in L256 0.670 (0.1329; passed at 0.669 in part 5: sweep noise at the 2 % level); D64 in L256 now
passes. RF3 0, bidirectional 0, ladder 0, none slower than the H100. Outputs bitwise those of `cuda:b200_sm100` 41 (the draw and the
reduction are bit-identical; accuracy tokens unaffected).

What remains: D64 L128 needs 6 us off a step whose in-graph part is ~46 us (forward 19 us of the inference kernels, backward 27 us in four
launches; grad_reduce ~3.9 us of launch / latency tail). The fused backward that would remove those boundaries measured 18 us slower in
part 5 (register budget); the measured event head (s -> writer -> graph launch, ~4 us) is the replay floor of a graph that draws.

### Round 3, training part 7 (2026-11-18, `cuda:b200_sm100` 45, agent r3-trimul-train)

Start (three-backend sweep `_out/s3_b45.jsonl`, against the 2.5.14 H100 tokens of the merged base b070dfb11): 4 training cases above
0.667, all TriangleMultiplication -- D64 L128 out / in 0.749 / 0.746 (0.0560 ms, need 0.0498), hidden 128 L128 out / in 0.702 / 0.692;
ladder 0, none slower than the H100.

**One graph RNG writer for TriMul and TriangleAttention (`kernels/_graph_rng_sm100.{cu,py}`).** Part 6's writer (`wide_aux.cu`) and the
H100 pack.cu built for sm_100a (TriangleAttention's `drop_scale`) each registered an `aten::fill_.Scalar` override; the last registration
served the fills and the other family's graphs took ATen's two fill launches. Now one tiny sm_100a extension holds the only override and
the registry (user objects per graph, seed fill held and written by the offset fill's one programmatic launch); `wide_aux.cu` and the B200
build of pack.cu register their captured scalars through its C function table, handed over by Python once per load (`connect`). The
B200 pack build is `triangle_attention/cuda/b200_sources/pack_sm100.cu` (`triattn_b200_pack_v2`): it includes `h100_sources/pack.cu`
UNCHANGED (byte-identical to origin/main: no H100 cache-key change) and reroutes it by two function-like macros -- its fill_ override
address resolves to a forwarder into the shared writer, its register_graph_rng calls resolve to a non-const-reference overload that
registers with the shared writer (pack.cu's own definitions are compiled under other names); table via
`torch.ops.mw_pack_sm100.set_graph_rng_api`. Same values. Test
`test_b200_graph_rng_writer_shared_with_triattn`: TriAttn-only, TriMul-only and mixed graphs captured in one process replay the draws of
ATen's fills, with no ATen fill kernel launched. One process holding both training steps, writer on / ATen (`_tools/rngab.py`, three
rounds, median ms): TriMul D64 L128 0.0561 / 0.0588, L256 0.0949 / 0.0969; TriangleAttention d64 x 4 L128 0.1032 / 0.1059, L256 0.2802 /
0.2823.

**Pairwise backward fusions (b7f phases with only the pair's bodies compiled, `-DB7F_BUILD=1 -DB7F_PAIR=1`).** ptxas: every pair is at
the 128-register cap of 512 threads and spills (b1u + cgb 28-32 B, cgb + b7m 48 B stores / 140 B loads, b7m + reduction 32 / 60; cgb
alone 115 registers, b7m alone 128 with 20 B). All bitwise equal to the separate launches; D64 L128 out / in (ms, median of three):
separate 0.0553 / 0.0553, b1u + cgb 0.0639 / 0.0645, cgb + b7m 0.0692 / 0.0700, b7m + reduction 0.0614 / 0.0619. Rejected: the claim /
drain framework and the spills cost more than the boundary.

**PDL on the remaining edge.** Every backward edge already is a programmatic launch; b7m -> grad_reduce had no explicit trigger (entry
trigger: 2-4 us slower, part 5). A late trigger (tid 384 when the LN warps get the last tile's accumulator, `-DB7M_TRIG=2`): D64 L128
0.0551-0.0560 -> 0.0580, hidden 128 0.0672-0.0684 -> 0.0692, L256 level. Rejected (off by default).

State after (merged base b070dfb11; GPU tests 136 passed (test_trimul_b200.py + TriAttn drop_scale); sweep `_out/s3_f57.jsonl` vs the
2.5.14 H100 tokens): 6 cases above 0.667 -- D64 L128 out / in 0.749 / 0.748 (0.0560 ms), hidden 128 L128 out / in 0.700 / 0.694, D64
L256 in / out 0.671 / 0.667 (0.0949 ms against 0.0946 at the start: the 2 % sweep step these two cells sit on; TriMul-alone code path
unchanged); RF3 0, bidirectional 0, ladder 0 (cuda / triton <= 0.72, triton / pytorch <= 0.42), none slower than the H100. Outputs
bitwise those of 43.

What remains: D64 L128 needs ~6 us off a step whose in-graph part is 46 us (timeline: writer -0.4-1.5, forms 3.2-5.7, k1w to 11.4, bmm to
14.6, k3g to 21.7, b1u to 30.9, cgb to 35.4, b7m to 45.6, grad_reduce to 49.1). The backward boundaries cannot be fused inside the
512-thread register budget (above); the forward (k1w / bmm / k3g, 16.5 us, the inference agent's kernels) and the forms launch on its
critical path (~2.5 us: k1w reads bf16 weight forms) are where the rest is.

### Round 3, training part 8 (2026-11-18, agent r3-trimul-train; no route change, `cuda:b200_sm100` stays 45)

The forward of the D64 L128 training step (forms 2.6 us, then k1w 6.3, cuBLAS contraction 3.1, k3g 7.6; profiler timeline after the
L2 flush, `_tools/tline.py`).

**(a) The front from the fp32 master weights (`k1wt.cu`, `forward_small_master`, `forms_philox_gs_op`; built, bitwise, off: `K1WT`).**
`k1wt.cu` is a training-only copy of k1w's kernel (k1w.cu, the inference agent's, unchanged): the four front matrices are read as the
stored fp32 masters and rounded to bf16 (RNE, the forms launch's rounding: the same operand bits) into the weight ring by the epilogue
warps, at one-ring residency (every weight K-block of a CTA resident: L128, 121 registers, no spills). Three orders were measured:
1. forms -> k1wt as a programmatic dependent that does not wait at entry (waits at its end): k1wt still started 2.1 us after the forms
   launch (the dependent launch latency, and the 192 forms blocks hold the register files k1wt needs): level.
2. k1wt first (waits for the work before it, then triggers), then the forms launch on the 20 SMs it leaves free (`forms_gs_kernel`:
   1024-thread blocks, grid stride, no wait at entry, griddepcontrol.wait at its end): k1wt 3.2 -> 10.3 us (6.9 us; k1w 6.3: the fp32
   loads and conversion), the forms launch must complete after k1wt, so the contraction starts at 11.1 (was 11.4); step end in the
   graph 48.4 us (was 49.1); graph-timed D64 L128 out / in 0.0555 / 0.0554 -> 0.0553 / 0.0553 ms, hidden 128 0.0676 -> 0.0676.
3. as 2 with the forms launch on a side stream (a graph branch joined before k3g): k1wt starts at 4.1 (no programmatic edge from the
   RNG writer) and the contraction only at 13.2 after the two-input join: 0.0560 -> 0.0580 ms. Rejected (code removed).
Order 2 is bitwise (test `test_b200_k1wt_master_front_bits`: eager and graph replays, y / dx / every gradient) but level within the timer's
step: kept off (`K1WT = False`, not prebuilt).

**(b) The contraction.** `ctr.cu` (the inference route's tcgen05 contraction, a programmatic dependent triggering k3g) in the D64 one-
direction L128 cell: 0.0560 -> 0.0559 / 0.0560 -> 0.0560 ms (level; part 4 measured the same). A fusion of the contraction into k3g needs
every right plane (2 MB at D64 L128) per output tile: not built.

State: unchanged from part 7 (six cases above 0.667 against the 2.5.14 H100 tokens; ladder 0). The D64 L128 step's forward is ~18 us of
launch-bound kernels at one tile per CTA (forms / front / contraction / output, each 3-7 us) and its backward ~27 us in four launches
whose fusion loses on registers (part 7): the remaining 6 us are not in one kernel.

### Round 3, training part 9 (2026-11-18, agent r3-trimul-train; no route change, `cuda:b200_sm100` stays 45)

The question: does a persistent cooperative kernel (stages separated by grid barriers instead of launches, as TriAttn L128
inference did) pay for the D64 / hidden 128 L128 training step? Start (base `b200-2.5.x` 703c1ad40, engine 2.5.14; three-backend
training sweep, one measurer process per length, `_out/s3_r9b0.jsonl`): 5 training cases above 0.667, all TriangleMultiplication --
D64 L128 out / in 0.748 / 0.748 (0.0559 / 0.0560 ms), hidden 128 L128 out / in 0.702 / 0.697, D64 L256 in 0.667 (0.0943 against
0.0943 needed; out passed this sweep at the same code); RF3 0, bidirectional 0, ladder 0, none slower than the H100.

**Boundary cost on this card (`docs/b200/rejected/trimul_train9_b7m_grad_reduce_grid_barrier.diff`, `bnd.cu`).** N stages of 128
CTAs x 512 threads (one per SM, 200 KB shared memory, TMEM allocation + mbarrier init per launch), graph replays, us per replay:

| stages | plain launches | PDL | PDL + entry trigger | one cooperative launch, generation barrier | one cooperative launch, monotonic counter (red.release / ld.acquire) |
|---|---|---|---|---|---|
| 1 | 4.10 | 4.10 | 4.11 | 4.11 | 4.11 |
| 4 | 12.29 | 10.25 | 8.21 | 10.26 | 8.21 |
| 8 | 20.49 | 17.41 | 14.34 | 20.48 | 14.36 |
| 8, 2 us work per stage | 34.85 | 32.80 | 30.72 | 36.87 | 30.74 |

A grid barrier costs what a programmatic launch boundary with an entry trigger costs (~1.5 us each); there is no ~5 us launch to win.
Every backward edge of the step is already such a launch; what the kernels take beyond it is their own one-tile latency chain.

**b7m + grad_reduce in one launch (built, bitwise, rejected).** b7m's CTAs (tiles and the B7M_XR extra CTAs, grid <= SMs) arrive on a
monotonic counter in k3g's zeroed flag words after their work, then sum grad_reduce's 128-output blocks themselves (grad_reduce's
arithmetic with 16 warps: the same bits; RF3 factor included); the last CTA out resets the counter. Bitwise equal to b7m + grad_reduce and
repeatable (D64 L128 out, hidden 128 in, RF3 L128 / L256 in, D64 L256, bidirectional D64). Measured (A / B one process, `_tools/ab2.py`,
median ms): D64 L128 out 0.0550 -> 0.0570, in 0.0553 -> 0.0572, hidden 128 out / in 0.0675 / 0.0673 -> 0.0714 / 0.0712, D64 L256 out /
in 0.0941 / 0.0940 -> 0.0966 / 0.0960. Profiler timeline (D64 L128, from the end of cgb): separate b7m 10.2 + grad_reduce 3.6 us = 13.8;
fused with the cooperative attribute 15.5 (b7m no longer starts under cgb: a cooperative launch waits for the whole device); fused
without it 14.7-15.1, of which the barrier alone (sums compiled out, wrong gradients) 12.2 - 10.2 = 2.0 us and the sums after it 2.5 us
(grad_reduce's own launch takes 3.6 us for both). The partials are 8 MB (128 CTA rows of 64 KB): they need every SM to sum quickly, so
the work cannot move onto the 20 free SMs before the barrier.

Not built, on these numbers: the cooperative forward (forms -> front -> contraction -> output). Its three internal edges are already
programmatic launches (k1w after forms, k3g after the contraction) or a cuBLAS launch that the PDL `ctr` replaced at no gain (parts 4 and 8);
a barrier per edge costs the same, and the cooperative attribute would forbid the PDL overlap with the backward's first kernel.

State: unchanged (sources as part 8; the rejected code is the diff above). What remains, measured: the D64 / hidden 128 L128 step is
one-tile latency chains (k1w 6.3, cuBLAS 3.1, k3g 7.6, b1u 9.3, cgb 5.1, b7m 9.7, grad_reduce 3.6 us after the previous kernel's end),
not launch boundaries; D64 L256 in sits on the 2 % boundary (0.0940-0.0949 ms between sweeps).


### Round 3, training part 10 (2026-11-18, `cuda:b200_sm100` 47, agent r3-trimul-train)

Start (base `b200-2.5.x` 8dbd94351 / 05fe8a72b, engine 2.5.14; three-backend training sweep, one measurer process per length,
`_out/s3_r10b.jsonl`): 6 training cases above 0.667, all TriangleMultiplication -- D64 L128 out / in 0.742 / 0.742 (0.0554 / 0.0556
ms), hidden 128 L128 out / in 0.700 / 0.690, D64 L256 out / in 0.667 / 0.667 (on the boundary); RF3 0, bidirectional 0, ladder 0,
none slower than the H100.

**Phase trace (`pt_tl.cuh`, `-DPT_TL=1`: per-CTA clock64 marks since the CTA's entry, globaltimer at entry / exit; `_tools/ptl.py`,
`b1u_tl` / `b7m_prof` read them).** D64 L128 one direction, median CTA, us after entry:
- b1u: griddepcontrol.wait returns 1.0, the tile's operands (tri, xn, dy: 64 KB from four lanes) 2.9, Q / G 3.3, dPr handed over 4.4,
  dnr 5.3, dTri handed 5.8, tail 6.5 -> 7.6 (partials), exit 7.7.
- b7m: wait 0.7, dA chunk 0 2.4 (1.7 us of load latency), chunk 0 dgp 3.7, last chunk dgp 4.9, dxn 6.5, dx store 9.3 (LN_in backward
  2.8 us on the four LN warps), dW1 flushed 7.4, lnpart 9.9, exit 10.0.
The kernels' critical paths are a series of latency-bound phases on one tile; the cluster split of the brief (2-4 CTAs per tile) needs
256-512 CTAs for the 128 tiles of L128 on 148 SMs (two waves, or two CTAs per SM at <= 113 KB shared memory and 64 registers per
thread: b7m holds 216 KB and 128 registers), so it was not built; the phases were shortened inside the CTA instead.

Built and measured (A / B in separate builds, two rounds; profiler step end in us after the L2 flush and graph median ms, `_tools/j_ab1{0,1,2,3}.sh`):

| change | D64 L128 | hidden 128 L128 | D64 L256 | taken |
|---|---|---|---|---|
| `B7M_FTS`: b7m's dW1 partial (32 KB per chunk) staged as 32-column SW128 boxes (conflict-free) in the dgp / xn tiles and written by TMA stores from one thread (was 256 threads' coalesced float4 copy-out, which ran on the LSUs against the LN_in backward); bitwise | 50.1 -> 49.0 us, 0.0560 -> 0.0552 ms | 63.5 -> 63.2 us | 87.2 -> 85.9 us, 0.0949 -> 0.0929 | yes |
| `B7M_LNH`: the CTA's last tile's LN_in backward split by row halves -- warps 12-15 rows 0-15 of each lane quarter, warps 0-3 (loader / MMA / alloc / x loader, their roles done) rows 16-31; gated by an mbarrier the LN warps arrive on once they hold the last tile's x / dy (a parity wait two phases back would pass early: built that way first, nondeterministic at L >= 256); dgi / dbi get a second set of four warp rows, added after the first (2e-7 relative change of ln_pair.weight / bias) | +2.3 us (slower) | 63.7 -> 61.8 us, 0.0702 -> 0.0681 ms | +2.6 us | NCH 4 only (hidden 128 / bidirectional D64) |
| `B7M_GOE`: the last chunk's dGout Wg MMAs and acc_full commit before its dW1 MMAs (bitwise; dxn 1.1 us earlier in the trace) | +0.3 us | -0.5 us | level | no (default 0) |
| `B7M_BWL`: the last dx store's completion waited after the dgi / dbi sums (bitwise) | level | level | level | no |
| `B7M_XNE`: the first tile's xn loaded before griddepcontrol.wait (bitwise) | level | level | level | no |
| `TM_NWC`: b1u / b1v / b7m / cgb end on `cp.async.bulk.wait_group.read` instead of the write completion (bitwise) | level | level | level | no (default 0) |

With FTS the split helps only at NCH 4; at NCH 2 the trace shows the LN rows no faster with half the rows per warp (2.4 us for 16 rows
against 2.8 for 32): the LN phase is bound by the shared-memory / TMEM traffic it shares with the dW1 flush, not by its arithmetic.

Same graph, default build against FTS = LNH = 0 (`_tools/j_ab12.sh`, two rounds, ms): D64 L128 out 0.0560 -> 0.0552, in 0.0569 ->
0.0553, hidden 128 L128 out / in 0.0702 -> 0.0679, D64 L256 out / in 0.0949 -> 0.0923, bidirectional D64 L128 0.0701 -> 0.0680, RF3 D64
L128 0.0580 -> 0.0557. Outputs bitwise for NCH 2 (D64 one direction, RF3) at L128-640; at NCH 4 ln_pair.weight / bias differ by
<= 2.7e-7 relative; three runs bitwise each (`_tools/dchk.py`).

State after (merged `b200-2.5.x` 05fe8a72b; GPU tests 140 passed (test_trimul_b200.py + TriAttn drop_scale); accuracy `_out/acc_f10`:
every training row L128-768 119 GREEN + 7 YELLOW, no RED; three-backend sweep `_out/s3_r10a.jsonl` against the 2.5.14 H100 tokens):
5 cases above 0.667 -- D64 L128 out / in 0.749 / 0.746 (0.0560 ms in the sweep, 0.0552 / 0.0553 in the A / B: the graph timer's ~2 us
step), hidden 128 L128 out / in 0.700 / 0.696 (0.0681 / 0.0682, need 0.0648 / 0.0653), D64 L256 out 0.667 (0.0949 in the sweep, 0.0923
in the A / B; in passes); RF3 0, bidirectional 0 (worst 0.652), ladder 0 (cuda / triton <= 0.72, triton / pytorch <= 0.42), none
slower than the H100.

What remains, measured: D64 L128 needs ~6 us, hidden 128 L128 ~2.5 us. In the hidden 128 step b7m alone is 16.4 us after cgb (its LN
rows and the 128 KB dW1 flush both end ~11.5 us after entry: they share the SM's shared-memory / TMEM bandwidth); D64 L128 is the chain
forms 2.7, k1w 6.4, cuBLAS 3.0, k3g 7.6 (inference kernels), b1u 9.5, cgb 4.6, b7m 9.5, grad_reduce 3.5 us.


### Round 3, training part 11 (2026-11-18, `cuda:b200_sm100` 49 on TriangleMultiplication / RF3 / Bidirectional, agent r3-trimul-train)

Start (base `b200-2.5.x` 05fe8a72b; three-backend training sweep `_out/s3_r11b.jsonl` against the 2.5.14 H100 tokens): 5 training cases
above 0.667, all TriangleMultiplication -- D64 L128 out / in 0.749 / 0.746 (0.0560 ms, need 0.0498), hidden 128 L128 out / in 0.702 /
0.697, D64 L256 out 0.667; RF3 0, bidirectional 0, ladder 0, none slower than the H100.

**The training forward in one cluster launch (`b200_sources/tf_train_sm100.cu`, extension `trimul_b200_tft_v1`, `b200_train.TRAIN_TF`).**
A copy of the inference agent's `tf_sm100.cu` (ninth pass; the inference kernel is untouched) that also writes the backward's saved set
in the buffers / layouts k1w, cuBLAS and k3g left for it, so b1u -> cgb -> b7m -> grad_reduce run unchanged: xn [M, 64] (LN_in,
unmasked; staged in the unused plane staging and written by ONE TMA store of the sub-block), the token-order planes [2, 1, 64, L, L]
(the blocked planes of the in-kernel contraction go to a scratch buffer; the token-order words are kept in registers and stored after
the plane-row release, off the counter path), t [1, 1, 64, L, L] (from the contraction fragments, a 4 x 4 quad transpose so a quad stores
64 contiguous bytes), LN_out mean / rstd, the flag words zeroed, and k3g's training epilogue y = bf16(fma(bf16(p) sigmoid(bf16(g)),
ds[j], x)). Served: D64, hidden 64, one direction, L = 128, B = 1 (TriangleMultiplication and RoseTTAFold3), mask [L] uint8 / bool or
none, bf16 [64, 64] forms. Launch order: forms (draw + weight forms) -> tft as a programmatic dependent (barrier init, TMEM allocation
and the x TMA before griddepcontrol.wait; the weight forms, LayerNorm vectors and ds after it) -> b1u.

Where the dependents are triggered decided it (D64 out L128, profiler timeline, us after the L2 flush, step end / graph median):

| tft trigger | b1u start | tft end | b1u end | cgb end | step end | ms |
|---|---|---|---|---|---|---|
| at entry | 8.2 | 22.4 | 35.4 | 39.9 | 53.4 | 0.0572-0.0582 |
| none (implicit at exit) | 21.8 | 22.3 | 31.9 | 39.6 | 52.9 | 0.0580 |
| when t is complete (LN_out) | 19.1 | 22.1 | 30.8 | 35.5 | 48.9 | 0.0539 |
| at the epilogue (taken, `TFT_TRIG` 3) | 19.9 | 22.5 | 30.9 | 35.5 | 48.9 | 0.0539 |
| chain (k1w, cuBLAS, k3g) | 15.7 | 21.6 | 31.0 | 35.7 | 48.6 | 0.0556-0.0560 |

With the entry trigger b1u's CTAs sat ~14 us on the free SMs and its own phases ran ~4 us longer; with no early trigger cgb ran ~3.5 us
longer after b1u. Experiments that rewrote the saved tensors by the chain kernels after tft (planes by k1w, t by cuBLAS, y / xn / mo / ro
by k3g, flags by a memset, a dummy allocation, a persistent scratch) located it in the launch / trigger timing, not in the data: the
backward runs at the chain's pace whenever its first kernel is a PDL dependent triggered a few us before its primary ends.
Then tft as a programmatic dependent of the forms launch: tft start 7.1 -> 6.3 us, D64 out 0.0539 -> 0.0536 ms.

Phase marks (`-DTFT_TL`, `_tools/tfttl.py`, median CTA, us from the first entry; outgoing / incoming): x in 2.3 / 2.3, planes released
5.9 / 6.4, counters passed 7.2 / 8.5, t full 10.5 / 11.8, y stored 12.5 / 14.1, exit 12.8 / 14.3 (the inference tf: 13.0 us CUPTI).

**Fewer TMA boxes in b7m (`B7M_DA1`, taken).** b7m's single loader lane issued 2 + 2 NCH + 1 boxes per tile; each dA chunk's two 64-token
halves are now ONE box (sample-major: a 5-D map (token mod 64, channel, token / 64, sample, chunk), box (64, 64, 2, 1, 1); channel-major:
a 3-D map, box (64, 64, 2)). Bitwise. D64 L128 out 0.0528-0.0537 -> 0.0516-0.0517 ms (b7m 14.3-14.9 -> 13.4-13.6 us), in 0.0540-0.0548
-> 0.0533-0.0540; hidden 128 / bidirectional L128 and D64 L256 level.

Built, bitwise, level, off: `B1U_T1` (b1u's tri tile in one 5-D box instead of two boxes from two lanes: D64 L128 out 0.0518 / 0.0516,
in 0.0539 / 0.0532-0.0539), `CGB_MN1` (cgb's MN-major operand's two 64-wide M blocks in one 3-D box: every L128 / L256 cell within
0.0002 ms).

Accuracy / tests: one-launch y and every gradient against fp64 carry the chain's error (`test_b200_l128_train_one_launch`: out / in /
RF3, masked or not, max <= 1.5x and rms <= 1.1x the chain's, three steps and a graph replay bitwise); relative differences to the chain
y 1.2e-3..2.5e-3, dx 2.4e-3..4.7e-3, parameter gradients <= 2.7e-4 (`_tools/tftchk.py`). `measure.accuracy` training L128-768
(`_out/acc_f11b.jsonl`): 119 GREEN + 7 YELLOW, the same 7 cells and errors as the `b200_sm100` tokens (`_tools/acccmp.py`); every L128
row GREEN. GPU tests 152 passed.

State after (merged `b200-2.5.x` 80572b7d4; sweep `_out/s3_r11b2.jsonl`): 4 cases above 0.667 -- D64 L128 in 0.746 (0.0559 ms in the
sweep, 0.0533-0.0540 in the A / B processes), D64 L128 out 0.706 (0.0528 in the sweep, 0.0516 in the timeline), hidden 128 L128 out / in
0.702 / 0.697 (0.0682, unchanged: tft does not serve hidden 128); D64 L256 out passes (0.0929-0.0939 ms). RF3 0, bidirectional 0,
ladder 0, none slower than the H100.

What remains, measured: D64 L128 needs ~1.8 us (out) / ~3.5 us (in) more. The step is now forms 2.5 us (starts 4.6 us after the flush:
graph launch head), tft 14.8, b1u to +8.6 after tft, cgb +5, b7m +8.6, grad_reduce +3.7. The forms launch is on the critical path only
because tft reads the bf16 weight forms and ds; moving it beside tft (tft reading the fp32 masters, k1wt-style, and drawing ds itself)
is the next step, not built. Hidden 128 L128 (contraction 128 channels: 384 KB of operands per CTA) has no one-launch forward.

### Round 3, training part 12 (2026-11-18, agent r3-trimul-train; no route change, `cuda:b200_sm100` stays at the base's)

Start (base `b200-2.5.x` 55c854eb6; three-backend training sweep, one measurer process per length, `_out/s3_r13b.jsonl`): 4 training
cases above 0.667, all TriangleMultiplication L128 -- D64 in 0.737 (0.0552 ms, need 0.0500), D64 out 0.709 (0.0530, need 0.0498),
hidden 128 out / in 0.694 / 0.694 (0.0675 / 0.0680, need 0.0648 / 0.0653); RF3 0, bidirectional 0, ladder 0, none slower than the H100.

Everything below is bitwise equal to the shipped route (tests `test_b200_l128_train_master_after_forms`, 10 cases: y, dx, every
gradient against the forms-fed route, eager with and without the forms cache, graph replays under one seed; `_tools/dchk.py` cmp),
measured with `_tools/tline.py` (graph median + profiler timeline after the L2 flush, two rounds, D64 L128 out / in) and kept OFF:

| variant | what | D64 L128 out / in, ms | hidden 128 L128 | why |
|---|---|---|---|---|
| shipped | forms (2.5 us) -> tft as its PDL dependent | 0.0512-0.0519 / 0.0533-0.0540 | 0.0678-0.0682 | |
| `TFT_MASTER`, `TFT_FM` 1 | tft reads the six fp32 masters itself (ld.global, RNE into the SW128 boxes by warps 8-15), after a wait-first forms launch on 8 blocks that it waits for only before reading ds | 0.0512 / 0.0533 | | tft starts 0.5 us earlier (5.6 vs 6.1), but W is ready 4.1 us after entry (bf16 forms: ~2 us): the front starts ~1 us later |
| `TFT_FM` 2 | the same, each CTA of a cluster TMA-multicasts a 16-row quarter of each master | 0.0520 / 0.0539 | | W at 3.6 us |
| `TFT_FM` 3 | the same, each CTA one 16 KB TMA box per master at entry | 0.0513-0.0520 / 0.0533-0.0539 | | W at 3.3 us (the cold 96 KB after the L2 flush) |
| `TFT_CHAIN` | tft FIRST (waits at entry, triggers at entry), the forms launch its dependent beside it (ds first, then a counter tft's epilogue waits on; it triggers the backward when all 128 tft CTAs reach their epilogue, a second counter) | 0.0573-0.0580 / 0.0574-0.0580 | | tft 17-18 us with the master conversion, the forms launch starts 2.7 us after tft, b1u starts ~2.5 us later through the hand-off. The first chain step deadlocked under CUDA lazy loading (the forms kernel's first launch waited for the running tft): its module is now loaded before tft (`trimul_forms_gs_preload`) |
| `K1WT_AF` | hidden 128: k1wt (masters) right after a wait-first forms launch, waiting for it at its exit | | 0.0679-0.0680 (20 forms blocks; 8 blocks: 0.0723, the forms launch gated k1wt's exit) | level |
| `TFT_XREG` / `TFT_ONEFENCE` | the inference kernel's pass-10 x load (512 threads, ld.global) and one release fence a CTA | 0.0513-0.0519 / 0.0534-0.0539 | | level (tft 14.6-14.8 -> 15.0 us) |

What the trace says (tfttl, us from the first CTA entry, shipped kernel, outgoing / incoming): x 2.3 / 2.1, front products 3.3 / 3.3,
planes released 5.9 / 6.1, counters passed 7.2 / 8.5, t full 10.2 / 11.5, exit 13.1 / 14.3. The forms launch costs the step only
the ~1.7 us between its start and tft's (launch latency; tft's x load is already issued before its wait), and every way of taking it
off delays the weights by about as much. The incoming direction's extra ~1.3 us is the counter hand-off (counters pass 2.3 us after
its own release against 1.3 for outgoing), not the plane writes.

(2) the hidden 128 one-launch forward was not built: at 32 contraction channels a CTA the operands are 384 KB, the t tile 32 KB and
Wp 16 KB, so the contraction needs at least three serial operand passes (~0.8 us each) in 227 KB, against the ~2.5 us the cell needs;
the inference agent's pass 10 reached the same estimate. (3) b1u and b7m already issue their tile loads from several lanes (one box
each: B1U_ML / B7M_ML, DA1); their first operands arrive 1.7-1.9 us after griddepcontrol.wait, i.e. load latency, not issue.

State after: unchanged (no default path changed): the 4 misses above, ladder 0, none slower than the H100.

### What was still below 1.5x after the first pass and why (superseded where the second pass says otherwise)

64 cases (41 + 14 + 9), three groups:

1. **L128 (28 cases, ratio 0.68-1.01) and L256 (15 cases, 0.69-0.87): launch-bound.** The measurer's floor on this card is 6.0 us for
   one empty launch, 8.0 us for two or three, 12.1 us for four or five (`$W/b2511/agents/r3-trimul/_tools/floor.py`; one 64 x 128^3 cuBLAS `bmm` alone: 8.1 us).
   D64 inference at L128 is `k1w` + `bmm` + `k3g` = 20.4 us against the H100's 20.2 us (three launches there too); the 1.5x target is
   13.5 us, i.e. two launches' worth of floor plus 5 us for the whole update. D64 training at L128 is 10 launches (draw, scale, k1w,
   3 bmm, k3g, b1s, b7m, grad_reduce; under capture also the forms launch and two int64 fill launches, most likely torch's
   graph-safe generator state -- the H100's `pack.cu` replaces those by one-thread writes), 73 us against 74.5 us.
   Closing these takes (a design sketch, not tried) one persistent launch for the whole forward (front, contraction and output in one tcgen05 kernel with a
   cross-CTA hand-over of the planes, feasible while L^2 D fits L2: the contraction is 64-128 products of 128^3 / 256^3) and likewise
   one for the backward: new kernels, not a retune.
2. **D64 training L384-768 (18 cases: TriangleMultiplication 0.69-0.76 and hidden 128 at L384 0.67, RF3 0.71, bidirectional L384 0.68).** At L512: `b7m` 82 us,
   `b1s` 78, `k3g` 47, `k1w` 25, three `bmm` 58 of 310 us. The three tcgen05 backward / saving kernels move bytes at 2.6-2.9 TB/s
   (38-43 % of the card's 6.75 TB/s; the research branch's roofline table) while the H100's K1 / K3 sit at 79-94 % of their floors
   (`docs/h100/trimul.md`) -- so
   the B200 runs them only ~1.35x faster. They do not look instruction-bound (an estimate: about 4 k warp-instructions per 128-token tile in ~6 k
   clock cycles) but dependency-bound: one tile per CTA walks load -> LN -> MMA -> epilogue -> store with single-buffered TMEM
   operands in `k3g` (xo / xn and the p / g accumulators; `b1s` / `b7m` were not analysed at that level). The step that would close it is two tiles in flight
   per CTA (double-buffered TMEM operands and accumulators; `k3g` already uses all 512 TMEM columns, so that needs another operand
   layout) -- a re-pipelining of `k3g` / `b1s` / `b7m`, measured with the kernels' `PW` clock
   probes. Needed: -10 % at L512 / L640, -4 % at L768.
3. **Three inference cases at 0.67-0.69 in the sweep** (RF3 D64 incoming L1024 0.690, D256 outgoing L1536 0.675, D64 hidden 128
   outgoing L1024 0.672): measurement spread, not a kernel gap. The first case of a kind in a sweep process comes out up to 15 % slow
   (RF3 D64 incoming L1024: 0.3685 ms twice in the sweep, 0.3214 ms = 0.602 measured in its own process next to outgoing 0.3218 and
   TriangleMultiplication 0.3193 / 0.3173, the same three kernels); their mirrored-direction cases pass.

### Accuracy (2026-11-17, `cuda:b200_sm100` 3)

`measure.accuracy`, cuda, every row of the three classes: L128-768 eval + train (`_out/acc_v2s.jsonl`) and L1024 / 1536 / 2048 eval
(`_out/acc_v2l.jsonl`). Eval: 168 / 168 GREEN. Training: 118 GREEN, 7 YELLOW, 0 RED, 1 skipped (bidirectional D384 L768: the reference
ran out of memory on the shared GPU; GREEN in the tokens). The 7 YELLOW cells are the 7 YELLOW cells of the `b200_sm100` cuda tokens
with the same errors (8.1e-3 .. 1.14e-2 at tol 8e-3: weight / LayerNorm-affine gradients), so no light changed
(`_tools/acccmp.py`). The D256+ outputs changed by bf16 ulps (tanh gate, statistics summed in another order); bit-identity is not
required.

## Gaps

- **Speed vs the H100:** 48 of 298 cases are below the 1.5x target (none slower than the H100; worst 0.975), see "Round 3, fourth pass" above.
- **Unsupported: none.** The route serves every requirement case (row / dims / dtype `master` / mode / length) of the three classes.
- **YELLOW (7 cuda training cells at L128-768 since `cuda:b200_sm100` 2, not refused; 15 in the first round).** Weight / LayerNorm-affine gradients at 1.01-1.43 tol (listed in the
  accuracy records). Cause: bf16 operands (planes, pre-activation gradients) of reductions over L^2 tokens; the same cells and
  magnitudes as the released H100 tokens. Removing them would take fp32 (or split bf16x2) pre-activation gradients into the
  weight-gradient GEMMs (`b7m` / `b7g` dW1, the D256+ `torch.mm(dcat.t(), xn)`), at a bandwidth cost; not attempted.
- **Not measurable: TriangleMultiplication D384 eval L2976 / L3968** (OpenDDE refiner): the bf16 PyTorch reference itself runs out of
  memory on 178 GB (re-run exclusively: still skipped), so there is no light; the cuda route serves them (timed: 37.5 / 76.0 ms).
- **Prebuilt kernels: none for sm_100a.** The B200 TriMul extension (`trimul_b200_v6`, 11 sources) JIT-compiles on first use (~75 s);
  wiring it into `kernels/_prebuild.py`'s sm100 target / the wheel is open.
- **Kernel registry:** `requirements_kernel.csv` has no B200 TriMul kernel names (no rows added, by rule), so the judge's per-kernel
  versions for these classes on `b200_sm100` list none of the sm_100a kernels.
- **Triton on this card** (not this route): at D384 / D512 inference the portable Triton TriMul is slower than PyTorch (e.g. D512 L128
  0.58 vs 0.42 ms), a Triton-rung issue recorded here for the ladder.

## Tests

`tests/gpu/test_trimul_b200.py` (skipped unless capability (10, 0)): route + accuracy against the fp32 statements for 16 width /
direction / hidden variants in eval and train; 20 repeated backwards bit-identical (the ring-flag kernels, and D256's per-block
partials); a batch of 3 equals its one-sample calls bitwise (y, dx); ragged L 200 inference (host pad); RF3 masked batch; CUDA-graph
replay equals eager in training; the fused row-dropout scale equals the reference draw bitwise; RoseTTAFold3's factor in the forms launch
equals the `_LeftScaled` path bitwise (y, dx, every parameter gradient); L384 training through the double-buffered `k3g` / `b7m` pipelines (fp32 statements, five runs bitwise); the D64 L128 one-launch inference (`tf_sm100.cu`) against the chain and an fp64 reference, plain / masked, outgoing / incoming / RF3, repeated runs and graph replays bitwise (pass 9); the programmatic-dependent chain (k3g after the contraction at L128, the training kernels) captured and replayed ten times bitwise the eager step, eval and train; the D256-512 weight forms and gradient pack bitwise the statements they replaced (fourth pass); the dropout scale over consecutive draws and graph replays. 79 passed on the B200.


### S90 round, training (2026-11-18, `cuda:b200_sm100` 57, agent s90-trimul-train; branch `b200s90-trimul-train`)

Goal of the round: SOL% (floor = max of the tensor / HBM / MUFU / FMA floors of `sol_trimul.md`, 1457.7 TFLOPs, 6116 GB/s) of the three
classes, training. Per-case table before -> after (all 126 training cases, `docs/b200/trimul_s90_train_sol.txt`; `_tools/solsc.py` is the
study's calculator; one cuda-only measurer process per length, exclusive slot, graph timing).

| | cases | min / median / max SOL % before | after | >= 90 / >= 70 / >= 50 % |
|---|---|---|---|---|
| all training | 126 | 11.8 / 38.8 / 53.4 | 11.8 / 38.2 / 53.8 | 0 / 0 / 4 -> 5 |
| D >= 256 (the changed code) | 54 | 15.1 / 40.4 / 53.4 | 15.7 / 40.3 / 53.8 | mean time change -1.8 % (L128-L384 -3 .. -4.6 %, L512-768 -1 .. -3 %) |
| D <= 128 (code untouched) | 72 | | | mean time change +1.0 % = the sweep's noise (single cells +-7 %, up to +-18 % at L640 for D64 bidirectional) |

The lowest 15 cases are unchanged in order: D64 / RF3 D64 L128 (11.8-12.4 %), D128 L128 (14.0-14.7 %), D256 L128 (15.1 -> 15.7 %),
D64 bidirectional L128 (14.7 %), D128 bidirectional L128 (14.9 %). No case reaches 90 % (best 53.8 %: bidirectional D512 L768) or 70 %.

**What changed (D256-512 training; D64 / D128 chains untouched).** Outputs: y bitwise the base's; dx differs by bf16 ulps and the weight /
LayerNorm gradients by <= 1.3e-4 relative (fp32 sums associated in another order; `_tools/dchk.py --cmp`), bitwise run to run (three
runs of every case, graph replays); `measure.accuracy` training L128-768 `_out/acc_a1.jsonl`: 119 GREEN + 7 YELLOW, the same 7 cells and
the same errors (8.63e-3 vs 8.70e-3 ... against the `b200_sm100` tokens) -- basis: fp64 gold of the measurer, random non-zero weights,
real shapes. GPU tests 174 passed (`tests/gpu/test_trimul_b200.py`, 6 new: wide draw bits / graph replays).
1. **Draw + scale in the wide forms launch** (`wide_forms(..., drop_out, p)`, `b200_train.wide_forms_philox_op`): the D256-512 call's
   two aten RNG-state fills, `torch.rand` launch and `drop_scale` launch are gone (28 -> 23 launches with 2 and 3 below); ds bitwise the
   torch draw (the generator advances alike; graph RNG writer registers the captured scalars).
2. **`colsum3`** (`wide_aux.cu`): the three per-block partial tables (gate_bwd r0 / r1, lnout_bwd dg_o / db_o, lnin_bwd dg_i / db_i)
   are reduced by ONE launch at the backward's end (tile x row-split grid, last-arriving block adds the splits in split order) instead of
   one colsum32 launch after each producer.
3. **`gate_bwd3`** (a row per warp, loads of the next row before the reductions; S1 / S2 are one warp reduction, no atomics, no
   zeroing: the memset launch is gone, block 0 zeroes colsum3's counters) and **`lnout_bwd2`** (16 channel rows per block, every load
   of a thread before its arithmetic). Micro-benchmark (`_tools/micro.py`, L2-flushed graph replays on random data, copy kernel 6.0 TB/s):
   gate D256 L512 170 -> 144 us (3.9 -> 4.7 TB/s), D512 L384 189 -> 149 us (4.0 -> 5.1 TB/s), lnout D256 L512 108 -> 93 us, D512 L384
   118 -> 101 us, D256 L128 12.6 -> 9.9 us.
Module graph medians (A / B, same process pairs): D256 L128 178.3 -> 172.5 us, L256 479.4 -> 471-474, L512 1568 -> 1534, D512 L128 359.6 -> 357.

**Measured, rejected (diffs / notes under `docs/b200/rejected/`).**
- gate_bwd2 (next iteration's loads issued first, software pipeline): D256 L512 170 -> 184 us, D512 L384 189 -> 204 us. Not load-issue bound
  (`trimul_s90_gate_bwd2_prefetch.cu`). lnout_bwd with 64 / 128 channel rows per block: 112 / 118 us (16: 93 us); gate_bwd3 with 2 / 4
  rows in flight: 146 / 187 us (1: 144 us).
- **The "RNG then idle 6.6 / 20 us" of the per-kernel study is a profiler artifact, not a graph-time loss.** In the CUPTI timeline of
  D256 L128 the draw chain (two fills, rand, drop_scale, forms) ends 21 us after the first launch against 8 us with the draw in the forms
  launch (13 us), and the profiled replay ends 195.3 us against 174.2 us; the measurer's graph median moves only 178.3 -> 176.4 us (A / B
  of the draw alone, two rounds: D256 L128 -1.5 us, L256 -3.5 us, D512 L128 -1.6 us). The timelines (`_tools/tline.py`) of the old
  chain overstate the tiny kernels' cost; every claim in this section is a graph-median A / B.
- `xn` saved by the forward instead of the backward's `ln_apply` pass (D256 L512: 47 us, 94 % of HBM, plus its launch): not built. It
  would add N D bf16 (134 MB at D256 L512, 604 MB at D512 L768) to every block's saved set against ~25 us net gain; the recompute is the
  memory-lean choice and the saved set is what the 48-block training runs hold.

**What blocks 90 % (measured numbers; floors per case class).**
- D256 L512 (1534 us, floor 566 us, 36.9 %): HBM-bound elementwise passes 415 us (gate_bwd 144 us at 4.7 TB/s, lnout_bwd 93, lnin_bwd
  103 at 5.2, ln_apply 47 at 5.7, ln_stats 28 at 4.8, vs a 6.0 TB/s copy), tensor kernels that are not at the cuBLAS rate: k1w 141 us
  (137 GFLOP = 67 % of the sustained peak), k3w training 223 us (68.7 GFLOP, 21 %: each 128 x 128 item re-streams 32 KB stages of which half
  are the L2-resident weights, 1.07 GB through L2 -> SM; rejected earlier: cta-pair, multicast, cluster variants), k1wb 227 us (recompute
  of the four projections + dgrad, 43 %), the cuBLAS dW1 / dxn GEMMs 255 us at 91 % (their own limit), the contraction cuBLAS 190 us
  (excluded by the brief).
- L128 (D64 / D128 / D256: 11.8 - 15.7 %): 5-10 kernels or 23 launches of 2-17 us whose durations are the kernels' own latency
  chains (D64: forms 2.5 + tft 14.8 + b1u 8.6 + cgb 5 + b7m 8.6 + grad_reduce 3.7 us, measured in part 11 / 12; D256: k1w 16.5, k3w 16.6, k1wb 17.0,
  cuBLAS dW1 + dxn 28, gate 13, lnout 9 us) against floors of 6.6 / 13.2 / 28.7 us; the 6 us launch floor of the measurer alone is 10-50 % of them.
- Not attempted: fusing gate_bwd / lnout_bwd / ln_apply / lnin_bwd into the GEMMs' prologues / epilogues (cuBLAS has no hook; the
  tcgen05 replacements of dgrad, dWp, dW1 and dxn are new kernels of the size of k1wb), TMA multi-lane issue in k3w (its producer lane
  issues 3 boxes of 16 KB per stage; part 5's multi-lane TMA variant was measured level, `rejected/trimul_multilane_tma.diff`).
- 1.5x-vs-H100 / ladder counts: not re-swept (the changed cells only got faster, D256-512 by 1-5 %; cuda stays below triton and pytorch there, whose
  paths are untouched); the cells that were above 0.667 before are the D64 / hidden 128 L128 / L256 ones, unchanged here.

#### S90 training, second pass: k3w training (D256 L512 etc.), measured (2026-11-18, no code change)

`_tools/micro3.py` (k3w_forward alone, random data, L2-flushed graph replays): training k3w (p / g saved through staging tiles + TMA
stores, 3-slot ring) 208.8 us at D256 L512, 54.8 us at L256, 279.3 us at D512 L384; the same kernel without the p / g save (5-slot ring)
149.2 / 41.4 / 195.4 us. The save costs 60 us at D256 L512 = the 268 MB of p and g writes at ~4.5 TB/s: the "+76 us vs inference" is
traffic, not a pipeline defect. Variants (diff `rejected/trimul_s90_k3w_direct_pg_and_ys.diff`): p / g by direct row stores from the
epilogue threads with the 5-slot ring: 236.1 / 62.0 / 328.2 us (slower: 16-byte row-strided stores); y staged in its own tile so the next
residual x tile loads during the store (non-save kernel, 4-slot ring): 149.2 -> 139.8 us, 41.4 -> 38.7, 195.4 -> 189.3 (the save kernel
has no shared memory for it: 3 x 32 KB ring + x + y + p + g tiles + vectors exceed the 227 KB limit by 256 bytes).
What would remove the 60 us (and gate_bwd's re-read of p and g, another ~45 us): not saving p / g at all and recomputing them in the
backward inside a fused "k3wb" (k3w's mainloop + epilogue with dy, ds and the sigmoid-gradient algebra producing dpr, dg and the row sums
S1 / S2 across the D / 128 channel items of a token tile): HBM traffic 670 MB against 940 MB now, +68 GFLOP of tensor work (47 us at the
sustained peak, overlappable); estimated -3 to -4 % of the D256 step. A new tcgen05 kernel, not built in this pass.

#### S90 training, third pass: k3wb (gate backward fused into a recompute-p/g backward kernel), built and REJECTED (2026-11-18, no code change)

Diff: `docs/b200/rejected/trimul_s90_k3wb_gate_bwd_fusion.diff` (k3w.cu `SAVE == 3`, `k3wb_forward`, `wide_r01_cols`, the forward without the
p / g save, `_contract` with 8 extra rows, grad_pack reading r0 / r1 from the dWp GEMM's extra columns, tests; version bumped to 59 and
reverted). What it was: forward k3w without the p / g save (5-slot ring); backward k3wb = k3w's mainloop recomputing p and g, an epilogue
with dy (TMA tile in place of the residual x) producing dpr, dg (rows of the [dpre | dg] operand, strided TMA store), per-token S1 / S2 (fixed-order
sums over the channel halves and items of a tile in registers, written once) -- no p / g in the saved set (-268 MB per block at D256 L512).
Measured (`_tools/tline.py`, D256 L512): forward k3w 223 -> 154 us, k3wb 201 us (215 us with p, g rounded to bf16 as the old backward
saw them) against gate_bwd3 144 us + colsum: net -7 us at best, +8 us with the rounding that keeps the accuracy lights. Per-channel column sums
r0 / r1 over tokens by lane shuffles (three reduce-scatter rounds + two butterflies) cost 52 us (254 -> 202 us) and were replaced by 8 extra
columns (hi / lo bf16 splits of 1 / rs_o and mu_o) of the dWp cuBLAS GEMM (to_out.weight differs by <= 6.5e-4 relative: bf16 dpr as the
GEMM operand). Without rounding p / g to bf16: 4 more cells YELLOW (D256 incoming L384 / L512 / L640, D512 L512, D384 L768: 8.2-9.5e-3 vs
tol 8e-3; they pass); with it the lights equal the tokens (119 GREEN + 7 YELLOW). Full sweep after (p2 vs the part 1 state p1, all 126
cases): D256 mean time -0.1 %, D384 +0.1 %, D512 +2.3 % -- the measurer does not improve, so not adopted. A packed-f32x2 rewrite of the
epilogue arithmetic was level (201 -> 205 us). Conclusion: the fused epilogues are bound by the CUDA-core work of 8 epilogue warps per SM
(~45 instructions per channel pair), not by HBM; the 2 x 134 MB it saves cost as much in epilogue issue slots as the separate kernel costs in
traffic. The same reasoning bounds the proposed dxn GEMM + lnin_bwd fusion (D = 256 output row per CTA tile, LN backward in the epilogue:
mainloop ~130 us against 125 + 103 us now, but a second TMEM pass and the dg_i column sums put the epilogue at k3wb's level): estimated
<= 2 % of the D256 step; not built.

### S90 round, training, part 2 (2026-11-18, `cuda:b200_sm100` 61, agent s90-trimul-train; base = b200-2.5.x with inference 60)

Adopted (bitwise the same outputs as before each change; `_tools/chk3.sh` / `chk4.sh`: three runs bitwise, old / new bitwise; 174 GPU tests;
`measure.accuracy` training L128-768 on the merged tree: the same lights as the tokens, 119 GREEN + 7 YELLOW):
1. **Side stream in the wide backward** (`SIDE_BWD`): `ln_apply`, the dWp GEMM and the dW1 GEMM run beside the input-gradient chain (events
   after gate_bwd and k1wb, joined before the gradient pack). A / B (graph median, two rounds): D256 L256 458 -> 445 us, D384 L256 754 -> 743,
   D256 L512 1512-1558 -> 1504-1540, D256 L128 level (172.5 -> 172.3). At L128 the overlapped cuBLAS split-K kernels fill the card and stretch
   (dW1 split-K 12 -> 24.7 us, dWp 7.9 -> 13.6 us) and lnin_bwd waits for SMs until the dW1 kernel ends: no gain there.
2. **k1wb da prefetch** (`K1WB_PF`): each epilogue warp requests its next chunk's da box as soon as the current one has been read by ldmatrix
   (a proxy fence first), before the dpre store. k1wb D256 L512 225 -> 196 us (4 rounds: 225.2 / 225.4 -> 199.0 / 194.6), D256 L128 17.3 -> 16.5, D384 L256
   100 -> 98.4-99.8, D512 L384 358 / 322 -> 351 / 340 (noise); graph medians D256 L512 1.505 / 1.518 -> 1.479 / 1.481 ms. Measured first:
   with the epilogue arithmetic stubbed out k1wb is 205 us against 219, so it is not issue-bound (6 %); it is HBM-bound at ~72 % (940 MB in
   219 us); the da load latency between chunks was the exposed piece.
All 126 training cases, the merged tree against the first S90 baseline (`docs/b200/trimul_s90_train_sol_p3.txt`): mean time change D256 -5.8 %,
D384 -2.2 %, D512 -2.9 % (parts 1 + 2 together); the D <= 128 cases (code untouched) +2.4 % = sweep noise; median SOL 38.8 -> 38.3 % is that
noise, 0 cases >= 90 / >= 70 %, 6 cases >= 50 %.

Not built in this pass (measured reasons): the L128 cluster-split fused forward for D128 / D256 -- the D128 L128 chain is 87 us (floor 13 us), its
kernels are already PDL-overlapped and one 128-token tile per CTA on 128 of 148 SMs, so a 2-4 CTA split needs 256-512 CTAs and a second wave
(the part 10 trace of the D64 chain says the same); `fold_prep` merged into `ln_stats` is ~2 us of 172 us at D256 L128.

#### S90 training, part 2 follow-up: 1.5x-vs-H100 / ladder counts, gate_bwd3 grid (2026-11-18, `cuda:b200_sm100` 61 tree, no code change)

Three-backend training sweep of all 126 cases (one measurer process per length, exclusive; `_tools/s0.sh`, `lad.py`, H100 tokens of 2.5.14):
1.5x misses 4 (all TriangleMultiplication L128, ratio > 0.667: D64 incoming 0.737 (0.0552 ms), D64 hidden 128 outgoing / incoming 0.694 / 0.693,
D64 outgoing 0.693), RF3 0, bidirectional 0 (worst 0.660); none slower than the H100; ladder violations 0 (cuda / triton <= 0.714, triton / pytorch
<= 0.420). Same four cells as before this round; no count got worse.
gate_bwd3 grid (`WGB_BLOCKS` 592 / 888 / 1184 / 1776, micro on random data): D256 L128 20.9 / 20.3 / 23.4 / 28.0 us, D256 L512 179 / 143.5 / 155.7 / 152.3 us,
D512 L128 28.9 / 33.3 / 37.7 / 46.4 us: 888 stays (a size-dependent grid would gain 4 us at D512 L128 only).

#### S90 training, part 2 follow-up: epilogue ablation of k3g, and the L128 tile split (M64 / two CTAs per SM) for k3w measured (2026-11-18, no code change)

**Epilogue ablation (issue-bound or HBM-bound?).** `-DK3G_NOMATH` replaces k3g's whole sigmoid / gate arithmetic by one FMA (`tline.py`, D64 / D128 L512):
k3g<64,64,1,1> 39.0 -> 36.4 us, k3g<128,128,1,0> 77.8 -> 72.6 us (-7 %): the training output kernel is not issue-bound; it moves ~134 MB (D64 L512) in 39 us
(3.4 TB/s, 56 % of the copy rate) and is limited by its load / MMA / store pipeline, not by epilogue arithmetic. (b1u / b7m / b7g / b1g were not
ablated: their epilogues are interleaved with the LN_in / LN_out backward phases that earlier rounds (parts 10-12) traced by clock64; those traces
call them latency-bound chains.)

**L128 tile split.** Lower bound of what a 64-token tile (half the per-CTA work, 256 CTAs) could give: k3w training with half / a quarter of the items
(`TRIMUL_K3W_ITEMS_DIV`, one item per CTA, CTAs on separate SMs): D256 L128 18.6 -> 11.4 / 10.7 us, D256 L256 54.9 -> 32.7, D512 L128 40.1 -> 23.1. Whether
that is reachable depends on two CTAs per SM overlapping, so it was built in the cheapest form that tests it (`rejected/trimul_s90_k3w_lite_two_ctas_per_sm.diff`):
128-token tiles, one channel item a CTA, 2-slot ring, p / g by direct row stores (no staging tiles), ONE accumulator buffer (256 TMEM columns), 100 KB of
shared memory -> 256 CTAs, two resident per SM. Bitwise the old kernel (y, p, g; D256 H256 / H512 L128, 192), but SLOWER: D256 H256 L128 18.5 -> 25.0 us,
H512 21.1 -> 30.6 us (D384 / D512 do not fit the item budget and are unchanged). Two co-resident CTAs share the SM's TMA / MMA / epilogue resources: one item
already keeps them busy, so the 128-token item latency is not hidden by a second CTA. The 148-SM grid executes 256 items as 2 sequential items on 108 SMs
whatever the tiling; a real M64 tile would additionally run its MMAs at half array utilisation. Not built.

### S90 round, training, part 3 (2026-11-18, `cuda:b200_sm100` 65, agent s90-trimul-train): per-tile %globaltimer timelines of k3g, b7m, b1u

Method: stamps (`-DK3G_TT` / `-DB7M_TT`, `_tools/tt.py`, `tt2.py`) written to global memory by one thread of CTA 0 per tile / chunk event, one eager training step, read
back; times are us from the first stamp.

**k3g (training output kernel), D64 L512 / D128 L512, tile period and x stage life.** Per tile (D64): x load issue -> x_full 1.35 us, LN_in 1.05, products 0.5, epilogue
1.3, store read-out 0.4: the x tile lives 4.55 us (issue to x_empty) against a tile period of 2.46 us with TWO x buffers (4.55 / 2 = 2.3 us: the period is the stage
life over the ring depth). D128: issue -> x_full 1.8, LN_in 2.5, MMA 1.7, epilogue 3.6: life 10.2 us, period 5.3 us, again life / 2. The producer waits ~3 us for the slot.
**Built:** `K3_NXB` x tiles in flight (shape-dependent: 3 where the shared memory allows; the xn staging tiles follow); bitwise the old outputs (six small-D training cases,
three runs each, old vs new). k3g D64 L512 38.9 -> 35.9 us, D64 L256 13.7 -> 13.4, D64 hidden 128 L384 29.9 -> 27.6, **D128 L512 78.1 -> 66.6 us (-15 %)**, D128 L256 25.7 -> 23.1;
four tiles in flight gave the same D64 numbers (D128 does not fit). Module graph medians: D128 L512 0.5330 -> 0.5207 ms, D128 L256 0.1948 -> 0.1930, D64 L512 0.2362 ->
0.2341, D64 hidden 128 L384 0.2177 -> 0.2155. Sweep (all 126 cases, part 2 tree -> this tree): D128 -1.3 % mean time, D64 / D256 / D384 / D512 +-0.5 % (noise).
Earlier in this pass: `K3_NTH` 3 / 4 (token-half ring of the tri tile) does not fit the shared memory (static_assert).

**b7m (D64 input-side backward), D64 L512 CTA 0: 14 tiles, 3.45 us per tile (2 chunks).** dA chunk: issue -> da_full 2.0-2.5 us, consumer 0.9-1.4 us to free it, so one dA buffer
cycles in ~3.3 us and two buffers give 2 chunks per 3.3 us = the tile period. Built a three-buffer dA ring (`B7M_NDA` 3; the x / dy double buffering must go to fit: 216 KB of 227 KB are used
at NCH = 2): bitwise the same outputs, b7m 63.6 -> 62.6 us at L512 but module 0.2374 -> 0.2403 ms and L256 0.0922 -> 0.0943: not adopted
(`rejected/trimul_s90_b7m_da_ring3_and_timeline_stamps.diff`, with the timeline stamps).
**b1u (D64 output-side backward):** NT = 4 tri / NX = 3 x stages already. The loader waited for the x slot before issuing the tile's tri loads; issuing the tri loads first
(the coordinator's producer-order finding) is bitwise and level (b1u 47.5 -> 46.7 us L512, 18.9 -> 19.3 L256; module 0.2358 -> 0.2349, 0.0933 -> 0.0930): not adopted
(`rejected/trimul_s90_b1u_tri_load_before_x_wait.diff`). b1g / b7g (D128) were not timelined.
State: 1.5x misses 4 (the D64 / hidden 128 L128 cells, unchanged), ladder 0 (sweep of the part 2 tree); accuracy lights = tokens (119 GREEN + 7 YELLOW), 174 GPU tests, determinism bitwise.

#### S90 training, part 3 follow-up: k3w training timeline (D256 L512, D512 L384), 1.5x / ladder on the 65 tree (2026-11-18, no code change, `cuda:b200_sm100` stays 65)

**k3w training (save kernel), per-item stamps of CTA 0 (`K3W_TT`, `tt3.py`), D256 L512, 28 items, 7.2 us per item.** Steady item: accumulators complete -> epilogue arithmetic
3.7 us -> the three y / p / g stores issued and drained out of shared memory 2.1 us -> the residual tile of the next item reloaded 1.4 us (single residual buffer: the y is written in
place, so the tile is dead only after its store has been read) -> the next epilogue starts: the epilogue chain (3.7 + 2.1 + 1.4) is the item period; the MMA of an item finishes
well before its epilogue can start (tensor pipe idle ~70 % of the time; D512 L384: 8.0 us per item, the same chain). Built and measured, both bitwise the old outputs and level:
(1) y stored as its own bulk group so the residual tile is released after y has been read, not after p / g too (`rejected/trimul_s90_k3w_store_bulk_group_split.diff`): k3w D256 L512 221.9 / 216.0 / 211.0 (off) vs 221.9 / 230.1 us (on) over two rounds, D256 L128 16.5, L256 51.8-52.0:
no change; (2) the six TMA stores issued by six threads, each draining its own group (`rejected/trimul_s90_k3w_store_split_multilane_and_timeline_stamps.diff`, with the stamps): 224.0 / 238.9 (on) vs
221.0 / 227.8 us (off). The drain time is not the store issue and not the p / g groups; the epilogue arithmetic itself (3.7 us per 128 x 128 item with 8 warps) and the residual reload remain,
and the shared memory of the save kernel (192 KB + vectors, 227 KB limit) has no room for a second residual / y buffer (the non-save inference kernel gained 6 % from it earlier). b1g / b7g (D128) were not
instrumented: both are cooperative front / LN (b1g) and source / consumer (b7g) designs with their own rings (b1g: three operand slots, the next tile's xn prefetched into L2 already; b7g: 12 ring slots per
source group, four consumer stages, three xn buffers), where a plain stage-count change is not the knob, and the time went to the k3w chain above.

**1.5x vs H100 / ladder, all 126 training cases, three-backend sweep of the 65 tree:** misses 4 (all TriangleMultiplication L128: D64 incoming 0.744 (0.0558 ms), D64 outgoing 0.718, D64 hidden 128 outgoing / incoming
0.701 / 0.695), RF3 0, bidirectional 0 (worst 0.662); none slower than the H100; ladder violations 0 (cuda / triton <= 0.714, triton / pytorch <= 0.420). The four cells are the same as before this round.

#### S90 training, k3w overlap attempts after the timeline (2026-11-18, no code change, `cuda:b200_sm100` stays 65)

The production k3w already double-buffers the accumulators in TMEM (`b = i & 1`, p and g 128 columns each, two buffers = all 512 columns): the MMA of item i + 1 runs during item i's
epilogue, and its commit is long done when the epilogue arrives (timeline: `epi_acc` stamps equal the epilogue's own arrival); the item period is the epilogue chain, not MMA -> epilogue
serialisation. Two attempts to shorten the chain, both bitwise the old outputs:
- 16 epilogue warps (`K3W_EPW=16`, 640 threads, ~100 registers): k3w D256 L128 16.6 -> 19.1 us, L256 52.0 -> 59.6, L512 ~ level (243.9 / 235.3 base vs 256.4 / 243.5), D512 L384 322 / 303 -> 357 / 353: slower
  (register-limited; the arithmetic is not a warp-count problem).
- two residual / y tiles with the stage ring cut from 3 to 2 slots (the same 192 KB; the next item's residual loads while this item's y / p / g drain; the tiles released at the next item's start;
  `rejected/trimul_s90_k3w_two_residual_tiles_ring2.diff`): D256 L512 k3w 193.7 / 237.0 us (on) against 218.2 / 223.0 (off), module 1.4972 / 1.4781 vs 1.4799 / 1.5136 ms (level); D256 L128 17.0 vs 16.5 us,
  L256 52.7 vs 51.7; D512 L384 367 / 373 vs 304 / 293 us (+22 %: the 512-wide stages need the third slot). Not adopted.

#### S90 training: k1wb per-chunk timeline, multi-lane weight loader rejected; merged tree (66) re-verified (2026-11-18, no code change, `cuda:b200_sm100` stays 66)

**k1wb D256 L512 (`K1WB_TT`, CTA 0, warp 8 = epilogue group 0, per chunk; 8 chunks a tile, ~2.0 us per chunk).** A chunk: MMA start -> commit 1.4 us (the weight K-blocks stream through the 4-slot ring, 4 x 16 KB per chunk
issued by one lane as 8 boxes: ~600 clk each = ~2.5 us), accumulators waited for by the epilogue ~1.1 us, then the da box ~1.2 us more (even with the part 2 prefetch), 0.64 us of arithmetic, store. Weight issue looked like the limit, so the
weight loader was made multi-lane (all boxes of a step requested by one lane, issued by 2 x blocks lanes; groups of 1 / 2 / 4 K-blocks): bitwise the same outputs but SLOWER: k1wb D256 L512 196-204 -> 215-223 us, D512
L384 310-366 -> 418-482 us, D256 L128 16.5 -> 18.6-19.4 us (`rejected/trimul_s90_k1wb_multilane_weight_loader_and_timeline_stamps.diff`). The single-lane loader is not the limit; the extra synchronisation costs more than it saves.

**Merged tree (inference part 4: k3g with Geo<C,H,SAVE> and the inference schedule next to the training one), training re-run:** six small-D training cases (D64 out / in, D128, D64 hidden 128, RF3 D128, bidirectional D64)
bitwise equal to the pre-merge build (three runs each, plus the cross-build comparison), 174 GPU tests, accuracy lights = tokens (119 GREEN + 7 YELLOW).

#### S90 training: uninitialised-memory check (2026-11-18, `cuda:b200_sm100` 66, clean)

`_tools/uninit.py`: for 12 training cases (D64 out / in, hidden 128, D128, D256 L128 / L256, D512, RF3 D128 / D64, bidirectional D64 / D128 / D256) one step (y, dx, every parameter gradient) compared bitwise:
eager vs eager after filling the allocator's free blocks with 0xFF (24 GB in 1 MB - 512 MB blocks); a capture vs a second fresh capture; a capture vs a capture whose pool is poisoned (0xFF blocks allocated, filled and freed inside the capture,
so every replay re-poisons before the step). All bitwise equal: no kernel of the three classes' training path reads uninitialised memory in these cases. Regression test:
`test_b200_train_graph_captures_equal_bits_with_poisoned_memory` (5 cases). Not run: compute-sanitizer initcheck (no difference to chase), a cross-process capture comparison (single-process captures and eager runs were bitwise stable across the earlier determinism checks).

#### S90 training: 128-bit fold-vector broadcasts (inference finding) in the training kernels (2026-11-18, merged tree 70, no code change)

The merged k3w epilogue (shared by the inference and the training save instantiation) already loads the four per-channel vectors as 128-bit broadcasts (`K3W_NOVS8` off), so the training k3w has the change. A / B on the 70 tree (graph timeline, two rounds):
k3w training D256 L512 211.8 / 204.0 us (128-bit) against 232.3 / 214.0 us (`-DK3W_NOVS8=1`, 64-bit), module 1.4975 / 1.5073 against 1.5084 / 1.4995 ms (level); D256 L128 16.5 / 16.6 against 16.7 / 16.6, L256 51.7 / 51.9 against 51.9 / 52.1;
D512 L384 318 / 286 against 286 / 293 us (noise). The other training kernels already read their per-channel vectors as 128-bit (k3g, k1wb, b7g: `lds128f`); b7m's LN gamma loads are 64-bit (`lds64g`) in a latency-bound chain and were not changed.

#### S90 training: L2 next-tile prefetch on b1u / b7m (2026-11-18, measured, rejected, no code change)

`cp.async.bulk.prefetch.tensor.2d.L2` of the next tile's boxes (b1u: xn and dy by two spare loader lanes; b7m: xn by the dA loader, x and dy by the epilogue-input loader), bitwise the same outputs, six small-D training cases deterministic: slower.
D64 L512: b1u 47.1 -> 49.2 us, b7m 58.7 -> 59.7 us (module 0.2341 -> 0.2342 ms); D64 L256: b1u 19.0 -> 19.4, b7m 23.1 -> 24.1 us (module 0.0926 -> 0.0947 ms); D64 hidden 128 L384 level (0.2156 -> 0.2165). The inputs are L2 /
HBM-streamed at the tile period already and the prefetch adds requests ahead of the demand loads (`rejected/trimul_s90_b1u_b7m_l2_next_tile_prefetch.diff`).

#### S90 training: k1wb weights as 3-D boxes of two K-blocks (inference k1w finding applied; 2026-11-18, measured, rejected, no code change)

Built from the TMA-op-cost model (a TMA op ~600 clk per SM whatever its size; the inference `k1w` gained 4-9 % from 32 KB 3-D weight boxes): k1wb's weight loader requested a chunk's 4 K-blocks as 8 boxes of 8 KB; the training weights are four separate
row-major tensors, so a K-block's gate and projection rows cannot share one box. Built: 3-D boxes [64 k][64 rows][2 K-blocks] (16 KB per op, 4 ops a chunk instead of 8), group slots [gate kb0 | gate kb1 | proj kb0 | proj kb1], and the MMA as two N = 64 products per K-block
(gate -> accumulator columns 0-63, projection -> 64-127; same A from TMEM) instead of one N = 128. Bitwise the same outputs (four wide training cases, three runs, old vs new). Slower or level: k1wb D256 L512 201.1 / 201.4 us against 200.0 / 199.3 (off), D256 L128 17.0 / 16.9 against 16.6 / 16.7,
D512 L384 370 / 402 against 343 / 337 us (+10-19 %); module D256 L512 1.5628 / 1.5509 against 1.5161 / 1.5294 ms. The weight stream is not the limit of this kernel (the multi-lane loader showed the same from the other side) and the doubled MMA count costs more than the halved op count saves
(`rejected/trimul_s90_k1wb_3d_weight_box_pairs.diff`).

### S90 round, training, part 4 (2026-11-18, `cuda:b200_sm100` 73, agent s90-trimul-train): LN_out statistics inside the training k3w

**The LN-statistics fold into the backward GEMMs (the coordinator's item 1) does not save bytes.** dW = sum_m dcat[m, o] xn[m, i] with xn = (x - mu_m) rs_m g_i + b_i: pulling rs_m into the GEMM means a row-scaled dcat' = rs_m dcat
(the dxn GEMM then needs 1 / rs_m in lnin_bwd, and bf16(dcat rs) changes dxn's rounding), and the mean / bias terms need sum_m dcat' mu_m and sum_m dcat[m, o] for all 2P + D = 1280 columns: either x augmented by two columns (a copy of x: the same
134 MB read + write as ln_apply, no saving) or full-width column sums of dcat in the producers (the lane-shuffle reductions measured 52 us for 512 columns in the k3wb attempt). Not built.

**What was taken instead: the forward's LN_out statistics.** The inference k3w computes mu_o / rs_o from the t stages it already streams (OST warps); the training save kernel now does the same and writes the statistics out for the backward (the item whose channel tile starts at 0 stores them),
so the separate `wide_ln_stats` launch (28 us at D256 L512, 29 us at D512 L384) is gone: `k3w_kernel<D, H, 1, 1>`, flags 2 | 8 (tile-grouped schedule), `b200_train.K3W_TRAIN_OST`, only for n >= `K3W_TRAIN_OST_MINL` = 384 (L128 level, L256 slower: k3w 61.3 us in-kernel against 52.0 + 6.5 us).
Interleaved A / B, four rounds, graph median: D256 L384 0.8892 -> 0.8821 ms (-0.8 %), D256 L512 1.5240 -> 1.5045 (-1.3 %), D512 L384 2.3157 -> 2.2346 (-3.5 %). The statistics come from pivot-shifted sums (not the ln_stats kernel's order): y / dx change by bf16 ulps (1.9e-3 / 4e-3 relative to max), parameter gradients <= 7e-4; bitwise run to run (four cases, three runs).
Accuracy lights = tokens (119 GREEN + 7 YELLOW; D512 L384 ln_pair.weight 8.79e-3 against 8.70e-3), 179 GPU tests. A sweep against the earlier S90 baseline is confounded by the inference merges in between and was not used for the decision.

### Finalisation pass (2026-11-18): the fused small-L backward `b7f` is removed

`b7m.cu`'s `b7f` (b1u, cgb, b7m and grad_reduce as phases of one persistent launch, `-DB7F_BUILD=1`, `E.b7f_backward`, `B7F` / `B7F_MAX_L` /
`B7F_PH` in `b200_train.py`) was measured slower than the separate launches (D64 L128 0.0564 -> 0.0749 ms, training part 5) and was never
routed (`B7F = False`; the default build compiled only a stub that raised). One implementation per op: the kernel, its binding, the
Python branch and the `B7F_TU` include guards in `b1s.cu` are deleted; the diffs that measured it stay under `rejected/`. The ring `b1s`
(front / LN CTA split) stays: the unified `b1u` / `b1v` replace it on every D64 route, but `test_b200_unified_b1_matches_ring_b1s` and
`..._h128_...` use it as the A / B reference (`B1U = False` / `B1V = False`). The compiled output of every other kernel is unchanged (cubins
bitwise equal before and after the removal of the default-off timing macros, see the finalisation report).

#### S90 training: the D64 / D128 L >= 512 group (72 of 126 cases): kernel table against byte floors, b7g ablations (2026-11-18, no code change)

Kernel times from `tline.py` (graph replay, exclusive, current tree), bytes = the compulsory formula bytes of `sol-trimul-kern` `per_kernel_tables.md` (read once / write once per tensor), floor = bytes / 6116 GB/s; ratio = time / floor. The three cuBLAS contraction GEMMs (54 us D64, 102 us D128 at L512; 70-106 % of the tensor peak) are excluded.

| case | kernel | us | MB | %HBM | time / floor |
|---|---|---|---|---|---|
| D64 L512 (0.238 ms) | b7m | 59.7-66.1 | 244.7 | 60-69 | 1.5-1.65 |
| | b1u | 48.4 | 174.8 | 59 | 1.69 |
| | k3g | 36.1 | 102.8 | 47 | 2.15 |
| | k1w | 25.7 | 136.3 | 87 | 1.15 |
| | grad_reduce | 6.4 | 14.7 | 37 | 2.7 |
| D128 L512 (0.521 ms) | b7g | 173.9 | 508.7 | 48 | 2.09 |
| | b1g | 111.6 | 357.2 | 52 | 1.91 |
| | k3g | 65.4 | 203.5 | 51 | 1.96 |
| | k1w | 49.5 | 270.7 | 89 | 1.12 |
| D64 L1024 (study timings, 0.92 ms) | b7m / b1u / k3g / k1w | 216 / 192 / 150 / 106 | 949 / 684 / 411 / 545 | 72 / 63 / 45 / 85 | 1.39 / 1.7 / 2.2 / 1.18 |
| D128 L1024 (study timings, 2.27 ms) | b7g / b1g / k3g / k1w | 709 / 414 / 316 / 215 | 1918 / 1370 / 814 / 1082 | 44 / 54 / 42 / 83 | 2.26 / 1.85 / 2.4 / 1.2 |

So the group does have headroom: every non-front kernel (k3g, b1u / b1g, b7m / b7g) sits at 1.4-2.3x its byte floor and 45-70 % of HBM; only k1w is within 1.2x. The D64 step is b7m + b1u + k3g + k1w = 171 us of 238 (the rest: three contractions 54 us, forms / grad_reduce / launch gaps ~13 us); D128 is b7g + b1g + k3g + k1w = 400 us of 521.
b7g ablations (D128 L512 b7g 173.9 us; `-D` builds, no correctness check): no xn loads in the sources (`B7_NOXN`) 177.0 (the xn stream is not the limit); ring slots 12 -> 8 (`B7_RD`) 213.4 (+23 %: the source / consumer hand-over through the ring is the pipeline), 16 does not build; consumer stages 4 -> 3 (`B7_NST`) 176.1 (level). The earlier rounds' per-tile timelines of the b1 / b7 families (docs parts 10-12) and the three S90 timelines of this round (k3g adopted; b7m / b1u level or worse; k3w, k1wb) found the same shape: the kernels are tile pipelines whose period is a chain (load latency -> LN / MMA -> epilogue -> store) rather than any one stream; stage-count, producer-order, prefetch, box-size and loader-lane changes were measured on b1u / b7m / k3g / k1wb / k3w. b1g / b7g are cooperative (front / LN CTAs, source / consumer CTAs with a flag ring) and were ablated, not stamped; no fix was found in this pass.

#### S90 training part 5: b7g ring timeline (both roles), source-group split (version 75, 2026-11-18)

Probe `-DB7G_TL=1` (`_tools/b7gtl.py`): %globaltimer stamps indexed [event][tile][chunk] from both roles (publisher: slot wait / dgp ready / flag published; EW start, waits, ready; watcher poll start / flag seen; loader per-stage issue; MMA per-stage landed / commit; epilogue start / acc ready / dx store). D128 L512, one direction, 18 source groups x 4 = 72 source CTAs, 76 consumer CTAs, 2048 tiles, timer resolution 0.25 us, medians over the middle half of the tiles:

- Source: EW compute 0.75 us (+0.25 wait, +0.5 fence / bar / mask between tiles) = group period 1.5 us per tile (12.2 tiles / us over 18 groups); the publisher waits for a free slot 2.3 us on the median but the slot is free before the tile is ready (ready -> slot seen: -0.75 us): the 12-slot ring is never the limit; EW ready -> flag published 7.3 us (two-tile fence batching + fence ~1 us + write completion), publish -> watcher sees it 1.0 us, slot life (publish -> returned) 4.75 us: slots needed = 4.75 / 1.5 = 3.2 of 12.
- Consumer: period 6.4 us per CTA (76 CTAs -> 11.9 tiles / us = the source rate); the stage stream of a tile (10 x 32 KB) takes 6.5 us, 1.0 us load latency per stage, in bursts gated by the chunk flags: chunk 0 is issued 0.5 us after its last chunk is published, i.e. the consumers wait for the sources (data are consumed 0.5-2.5 us after being published); slot reissue after the MMA commit 0.25 us. Epilogue per tile: acc ready -> dx store 2.5 us.
- So at 18 groups the sources were the bottleneck (consumers idle on the flags); the optimum had moved since the 2026-10-30 measurement: `B7_SG` scan (graph medians, three rounds): L512 18 / 20 / 21 / 22 groups = 0.5215 / 0.5105 / 0.5201 / 0.5126 ms; L384 0.3315 (18) -> 0.3257 (21); L768 1.1843 -> 1.1649 (22); L1024 2.2829 -> 2.2283 (20) / 2.2316 (22); L128 / L256 flat (kept 18). Bidirectional (8 chunks) 12 groups stays best (L128-L768 scan 9-15). Adopted: `_B7_SG4` = {384: 21, 512: 20, 768: 22, 1024: 22}, else 18 below L384 and 22 above. Version 75.
- Accuracy: the dW1 per-group partials regroup, so dW1 differs from version 73 in fp32 summation order: worst per-tensor relative rms of the ten parameter gradients, SG 18 vs 20 (L512) 2.5e-6, SG 18 vs 22 (L768) 9.6e-6, run-to-run bitwise at the new split (p_drop 0 for the comparison, graph-free eager step).
- Bytes: the dgp ring (33 MB of slots at 20 groups x 12 x 4 x 32 KB) is written and read through L2; whether its dirty lines write back to HBM was not measured here (no ncu); xn is read by the 4 chunk CTAs of a group (L2) and x / dy / dGout / dx once by the consumers: no stream is read twice from HBM by design.
- Rejected: the next tile's mask loads issued before the tile's waits (`B7_MPF`): 180.6 vs 176.8 us (diff in `docs/b200/rejected/b7g_mask_prefetch_and_stamps.diff`). Not done (no time / no go): cluster-local DSMEM ring, EW MUFU offload.

#### S90 training part 6 (handoff, 2026-11-18): shipped-default re-time on the merged tip, Python-side split scans

**Re-time of the shipped default, tip 75 vs tip 74 (f7e6fb7b3 = 4e3f83f0f + trimul-inf part 12)**, alternating processes A B A B A B on one GPU (`_tools/abtree.sh`, `_tools/med.py`; graph medians, L2-flushed; each process three fresh captures; median of the three rounds' medians): D128 one direction L384 0.3264 -> 0.3241 ms (-0.7 %), L512 0.5249 -> 0.5129 (-2.3 %), L768 1.1321 -> 1.1023 (-2.6 %; run-to-run spread of one tree up to 3 %); D64 L512 (not touched by the change) 0.2495 -> 0.2419 (noise). No new knob is a runtime value in a hot loop (the group count is a Python-side launch argument).

**Python-side tunables** (`_tools/scans.sh`, `ab2.py`: 3 interleaved rounds, adopt only >= 1.5 % consistent):

| constant | values | cases | result |
|---|---|---|---|
| `B7_SG` one direction (part 5) | 16-22 | D128 L128-L1024 | adopted: `_B7_SG4` (L384 21, L512 20, L768+ 22), version 75 |
| `B7_SG` bidirectional | 9-15 | D128 L128-L768 | 12 stays best |
| `FRONT_CAP_D128_4` (b1g front CTAs, one direction; now a module value, default 80) | 72 76 80 84 88 | D128 L256 / 384 / 512 / 768 / 1024 | L256 / L384 flat (<= 0.4 %); L512 76 -0.6 %; L768 88 -2.8 % by median but -0.6 % by min; L1024 84 -2.5 % median, -1.1 % min; no value wins at every L and the spread is the noise: not adopted |
| `FRONT_CAP_D128_2` (bidirectional, default 84) | 76 80 84 88 92 | bidirectional D128 L256 / 384 / 512 / 768 | L256 / L384 flat; L512 76 -1.3 %; L768 88 -2.3 % median, noisy: not adopted |
| `FRONT_CAP_D64` (default 90) | 80 90 96 | D64 L512 one direction and bidirectional | flat (D64 runs the unified b1u / b1v; the cap is not on its path) |
| `COLSUM3_SPLIT`, `CGB_MAX_L`, `B1W_MAX_L`, `TRAIN_CTR_CELLS`, `B1_RSF`, `B7M_XG`, `K3W_TRAIN_OST_MINL`, side-stream thresholds, every constant of `b200_infer.py` | | | NOT scanned (`_tools/scans.sh` has the cases `cs`, `cgb`, `b1w`, `ctr`, `sg8` ready) |

**Open:** the b1g ring timeline (b1g.cu: front publisher flag release at the `st_release` in warp 2, slot wait in warp 3, LN-side `spin_until` in warp 0 -- the same stamp design as `-DB7G_TL`, events [tile][chunk-less]); the unscanned constants above; b7g EW (MUFU) and consumer stage-stream latency (part 5).

### n1-trimul pass (2026-11-19, `cuda:b200_sm100` 77 on TriangleMultiplication / RF3 / Bidirectional; inference part 76, training part 77)

Python-side constant scans (graph medians, L2 flushed, exclusive GPU 0, `_tools/ab2.py` 3 interleaved rounds in one process; shipped-default
re-times in alternating processes vs the pristine tree, `_tools/spd.py` / `med2.py`). Adopted only >= 1.5 % consistent:

| constant | values / cases | result |
|---|---|---|
| `B1W_MAX_L` (training, D128 one direction, unified b1w instead of ring b1g) | 128 / 256 / 384 / 512 at L256-L512 | L256 0.1931 -> 0.1848 ms (-4.3 %, out and in, 3 runs); L384 / L512 level or worse (+0.7 %, noise): ADOPTED 256 |
| `CTR_CELLS` (inference, ctr.cu contraction, bitwise cuBLAS) | `TRIMUL_CTR_ALL=1` rotated A/B (4 rounds each order) at L256 | hidden-128 D64 L256 38.8 -> 36.7 us (-5.3 %, both directions), bidirectional D128 L256 69.5 -> 65.5 us (-5.8 %; -3.1 % in the re-time); D64 one-direction L256 outgoing +7 % (cuBLAS stays), D128 level: ADOPTED cells (256, one dir, 128) and (256, bidir, 128) |
| `COLSUM3_SPLIT` 1-4 (D256 L384 / L512) | | 1 is -1.5..-1.9 % but within the run spread; not adopted (D256 is outside the target) |
| `CGB_MAX_L` 256 / 512 | D64 / D128 L384 / L512, bidirectional | 512 is +3..+5 % slower everywhere: 256 stays |
| `B7_SG` bidirectional 11 / 12 / 13 | L384 / L512 / L1024 | 12 best (11 +1..+1.5 %, 13 +3..+4.5 %): stays |
| `B7M_XG` 0 / 16 / 32 / 48 | D64 / D128 L128, D64 L256 | D64 L128 0.0538 -> 0.0518 at 32 (already the default); flat otherwise |
| `TRAIN_CTR_CELLS` + L256 cells | D128 / D64 / bidirectional / hidden-128 train L256 | -0.8..-1.9 % by min, median flat: not adopted |
| `K3W_TRAIN_OST_MINL` 128 / 256 / 384 / 640 | D256 L256-L512, D384 L256 | within noise (+-1.5 %), 384 stays |
| `B1_RSF` 2 / 3 / 4 / 6 / 8 | D128 L512 / L768 | 2 +7 %, 3 +2 %, 4 = 6 = 8: 4 stays |
| inference env switches `TRIMUL_K3W_OST_MINL=256`, `K3W_D128=0`, `K1W_FOLD128_MAXL=384 / 1024`, `PACK=0` | L128-L768 all three classes | each is slower on the cells it changes (+4..+18 %); the shipped defaults stay |

Version 77: `B1W_MAX_L` 128 -> 256 (training) and the two L256 `CTR_CELLS` (inference). Accuracy: tests/gpu/test_trimul_b200.py 179 passed
(includes `test_b200_unified_b1_d128_matches_ring_b1g[in-256-1]`: rel <= 1e-2 against the ring path, bitwise repeatable five backwards);
measurer accuracy (`_tools/acc.sh`, fp64 gold, random non-zero weights, real shapes, L256 train + eval, 3 classes, 84 cases) 83 GREEN + 1
YELLOW; the YELLOW (D64 L256 train grad:to_left.weight 8.080e-3 vs 8.000e-3) is the identical value on the pristine tree (not caused by this pass).
Re-time vs pristine (alternating processes): D128 train L256 193.2 -> 184.8 us; D64 train L128 / L256 level (0.0537 / 0.0927 ms both trees).
No new kernel / build (prebuild plan check: no new trimul entry). Not done: the D64 / D128 L >= 512 kernels (b7g / b1g / k3g at 1.9-2.4x the
byte floor) were NOT improved in this pass; only their Python-side constants were scanned (all at their optimum).

#### n1-trimul pass, part 2 (2026-11-18/19): measured blockers of the D64 / D128 L >= 512 training group; b7g source EW parity split (`cuda:b200_sm100` stays 77)

**Power cap (new fact).** `nvidia-smi` during a looped D128 one-direction training step on GPU 0 (`_tools/loop.py`, six 1 s samples each):
L256 540 W, SM clock 1965 MHz, SW power cap not active; L512 992 W, 1710-1725 MHz, cap ACTIVE; L1024 987 W, 1515-1552 MHz, cap ACTIVE.
So the L >= 512 group (72 of 126 cases) runs at the 1000 W cap: step time ~ energy / 985 W, per-kernel times move with the neighbours'
power state (L1024 k3g 217-277 us, same source, in runs that differ only in another kernel's ablation; step medians 2.01-2.13 ms for
nominally identical trees). Timeline / issue-order changes that do not remove joules cannot pay there; run-to-run spread of a step
at L >= 768 is 3-6 %.

**Ablation builds (timing only, wrong results; macros default off, kept in the sources: `B7_ABL`, `B1G_ABL`, `K3_ABL`; `_tools/abl_*.sh`).**
D128 one direction L512, kernel time from the graph timeline (`tline.py`), us:

| kernel | variant | us | vs base |
|---|---|---|---|
| b7g 162.9-164.0 | no W1 / Wg loads (consumer stream halved, `B7_ABL=1`) | 161.4 | -1 % (L1024 667 -> 613, -8 %) |
| | no dgp ring loads (`2`) | 163.7 | 0 % (L1024 -2.7 %) |
| | no consumer MMA (`4`) | 157.4 | -3.6 % |
| | no dx TMA store (`8`) | 161.0 | -1.8 % (L1024 -3 %) |
| | no source EW math (`16`) | 150.9 | -7.5 % |
| | no ring dgp store (`32`) | 165.8 | 0 % |
| | no consumer output pass (`64`) / no c1 c2 pass (`128`) | 161.2 / 161.9 | -1 % / -0.6 % (L1024 -4.6 % / -2.4 %) |
| | no EW + no ring store (`48`) | 158.8 | -3 % |
| | 2+4+8+16+32 (keeps W1 loads, c1c2, output pass) | 125.0 | -23 % |
| | 1+4+8+16+32+64+128 (skeleton: loads, stats, flags, source MMAs) | 139.3 | -14 % |
| b1g 111.5 | no front MMAs (`B1G_ABL=1`) | 106.6 | -4.4 % |
| | no front epilogue math (`2`) / no LN-side MMAs (`4`) | 111.5 / 110.5 | 0 / -1 % |
| | no LN backward math (`8`) / no dTri store (`16`) / no ring dPr store (`32`) | 113.6 / 109.0 / 113.0 | +2 / -2.3 / +1 % |
| k3g 65.4-66.3 | no LN_out affine / stats (`K3_ABL=1 / 2`) | 63.0 / 63.9 | -3.6 / -2.2 % |
| | no LN_in stats / affine (`4 / 16`) | 64.3 / 62.4 | -3 / -5.8 % |
| | no epilogue math (`8`) | 62.7 | -5.4 % |
| | all five (`31`) | 56.9 | -14 % |

Reading: no single phase of any of the three kernels is the blocker. Removing any one phase (loads, MMAs, stores, elementwise
passes, ring hand-over) saves 0-8 %; removing every elementwise pass of k3g saves 14 %, and the bare skeleton of b7g (all math and
ring stores gone, loads / flags / source MMAs kept) is still 139 us of 163. The kernels are closed-loop tile pipelines whose roles
(80 source CTAs: 68 GFLOP of MMA + EW; 68 consumer CTAs: 43 GFLOP + LN backward) are co-limiting; one limit moved leaves the next at
1-3 %. The ring-slot count (`B7_RD` 12 -> 14 / 16) and `B1_RSF` 4 -> 6 / 8 are level (16 builds now; the 'does not build' of the earlier note is stale).

**Adopted: b7g source EW split by tile parity (`B7_SPLIT`, from `B7_SPLIT_MINL` = 256).** Warps 4-7 take the even tiles, warps 8-11
the odd tiles of a source CTA (both token groups of a quarter per tile, 128-thread named barriers 1 / 3, `da_free` / `pre_free`
arrival counts 4); the buffers were already parity-indexed, so the two groups run independent tile chains and the 0.5-0.75 us
inter-tile fence / barrier gap overlaps the other group's EW. Same arithmetic per element and the same MMA order: results are
BITWISE the same as the single-group kernel (five D128 training cases, one direction / in / RF3 / bidirectional B = 2, L256-L640,
y, dx and every parameter gradient `torch.equal`; run-to-run bitwise too). Kernel time (graph timeline, two processes each, us): D128
L256 57.0 / 58.0 -> 55.3 / 54.6, L512 162.9 -> 160.2, L768 372-374 -> 358-371, L1024 667-671 -> 622-626 (SG 20 / 22), bidirectional L256
82.6 -> 81.5 / 81.9, L512 283 -> 273.8 / 267.0; L128 and L384 level (bidirectional L128 +1 us, so L < 256 keeps the single group:
`B7_SPLIT_MINL`). SG scan with the split (L768 / L1024 20 / 22 / 24; L512 16 / 18 / 20 / 22): the optimum stays 20-22. STEP level
(alternating processes, graph medians, three rounds each order): D128 L256 train 0.1926 -> 0.1830 ms (together with `B1W_MAX_L`),
L512 0.5099 -> 0.5075 (-0.5 %), L768 / L1024 not distinguishable from the 3-6 % power-cap spread (min of three: L768 1.0693 / 1.0984 /
1.0871 base vs 1.0854 / 1.0825 / 1.0818; L1024 2.031 / 2.168 / 2.167 vs 2.072 / 2.120 / 2.040). So: bitwise-identical, small, real at the
kernel level, ~0.5 % at L512 and below the noise above; kept because it never loses (`_tools/split*.sh`, `B7_SPLIT=0` is the A / B build).

**Not built / why (each judged on the ablation table above):** a consumer epilogue parity split (c1c2 + output passes are -1..-4.6 %
in total at most); 2-CTA (`cta_group::2`) UMMA for the consumer stream (halves the W1 half of the stage bytes; removing ALL of it is -1 % at L512, -8 % at L1024);
W1-resident consumers (160 KB of operand + 224 KB of tile buffers do not fit); wider 32 KB A boxes (the dgp stream removed entirely is 0 %);
LN_out affine fold / single-pass LN_in statistics in k3g (the whole LN_out affine pass is 2.4 us of 66, and instruction removal cannot lower the
power-limited step much); fusing the contraction with its neighbours (the planes are read 4x by output tiles: batched GEMM structure).
The three cuBLAS contraction GEMMs (33.6 / 35.0 / 33.4 us at D128 L512) move ~201 / 210 / 200 MB: 6.0 TB/s, at the HBM roofline.

Validation of the final tree (version 77): `tests/gpu/test_trimul_b200.py` 179 passed; `_tools/splitchk.py` / `splitcmp.py` (random non-zero module weights,
real shapes, masked tokens, B = 1 / 2): y, dx and all parameter gradients bitwise equal to the single-group b7g and run-to-run bitwise (5 cases);
measurer accuracy (`_tools/acc.sh`, fp64 gold, L256 + L512, train + eval, three classes, 84 cases): 82 GREEN + 2 YELLOW, the same two YELLOW with the same
errors on the pristine tree (D64 L256 train grad:to_left.weight 8.080e-3, bidirectional D128 L512 train grad:ln_out.weight 8.716e-3, tol 8.000e-3);
`_prebuild --arch sm100 --check`: no new trimul entry (one extension, the new template instantiations live in it); non-sm100 sources: `h100_sources` identical to the base.

#### n1-trimul pass, part 3 (2026-11-18): sustained energy measurements, ladder / H100 lists at version 77, built variants

**Measurer for the power-capped group (`_tools/energy.sh` + `loop2.py`).** Graph replays back to back (no L2 flush) for 12-16 s on the exclusive GPU 0, `nvidia-smi`
power / SM clock sampled every 0.2 s (first 6 and last 2 samples dropped); reports ms/step, mean W, J/step. Repeatability: the same build three times
0.5551 / 0.5562 / 0.5562 ms (D128 L512), 2.3206 / 2.3206 / 2.3198 ms (L1024): +-0.1-0.2 %, i.e. 10x finer than the 3-6 % spread of the short graph medians. Caveat: ablations that change the DATA
(zeroed gate gradients, no MMA, no loads) also change the operand toggle energy, so their drops are upper bounds, not savings (B1G_ABL=2: -8 % / -11 %, all of it zeros flowing downstream).

**Sustained upper bounds per component (D128 one direction train; ms/step vs base 0.5555 / 2.3205; ablation builds):**

| ablation | L512 | L1024 |
|---|---|---|
| k3g: all five LN passes off (`K3_ABL=31`) | 0.5430 (-2.4 %) | 2.2608 (-2.6 %) |
| k3g: LN_out affine pass off (`K3_ABL=1`; the fold's whole gain) | 0.5522 (-0.7 %) | 2.2978 (-1.0 %) |
| b1g: front + LN-side MMAs off (`B1G_ABL=5`) | 0.5385 (-3.1 %) | 2.2456 (-3.3 %) |
| b1g: LN backward math off (`B1G_ABL=8`) | 0.5550 (0) | 2.3000 (-0.9 %) |
| b7g: source EW math off (`B7_ABL=16`) | 0.5443 (-2.0 %) | 2.2854 (-1.5 %) |
| b7g: consumer MMA off (`4`) | 0.5424 (-2.4 %) | 2.2606 (-2.6 %) |
| b7g: W1 / Wg stream off (`1`) | 0.5518 (-0.7 %) | 2.3038 (-0.7 %) |
| b7g: consumer output + c1 c2 passes off (`192`) | 0.5465 (-1.7 %) | 2.2704 (-2.1 %) |
| b7g: skeleton only (`253`) | 0.5109 (-8.1 %) | 2.1499 (-7.3 %) |

So the largest removable energy items are MMAs (10 % of the step together, not removable) and the elementwise passes (k3g 2.5 %, b7g EW 2 %, consumer passes 2 %); no single item is above 3 %.

**Built and measured (sustained, three interleaved rounds, base 0.5551-0.5562 / 2.3198-2.3206 ms):**
- k3g LN_in statistics in one pivot-shifted pass (`K3_LNIN1`): 0.5547 / 0.5555 / 0.5555 and 2.3139 / 2.3141 / 2.3147 ms (-0.1 % / -0.26 %), same lights at the measurer; kept (default 1, training and inference). The LN_out fold (the other k3g item) is bounded by `K3_ABL=1` (whole affine pass: -0.7 % / -1.0 %) and needs Wp' / bias plumbing through the forms buffer: not built, bounded above by that row.
- b7g source EW split (`B7_SPLIT`): sustained L512 0.5579 -> 0.5551 (-0.5 %), L384 +0.25 %, L768 +0.2 %, L1024 +0.25 %: the route is limited to L = 256 and L = 512 (`B7_SPLIT_MINL / MAXL`, `L % 256 == 0`).
- b7g consumer epilogue even / odd split (`B7_CSPLIT`, diff `rejected/n1_trimul_b7g_consumer_epilogue_parity_split.diff`): correct (deterministic, grads rel 2e-7 of the single group) but sustained +7.0 % (L512 0.5960) and +8.5 % (L1024 2.5203 ms, 980 W, clock UP 1632 MHz): the dy tile is one buffer, so the output passes of consecutive tiles serialize, and the 128-thread rows are twice as long. Rejected. Two bugs on the way (written down because both are easy to repeat): a parity `mbarrier.wait` cannot wait two phases ahead (the second group's dy wait returned early; fixed with a `dy_done` barrier), and the shared stage-ring scratch of the final dgi / dbi combine must wait for BOTH groups.
- L2 hints (`rejected/n1_trimul_b1g_tri_evict_last.diff`, `..._b7g_ring_store_l2_hint.diff`): b1g front tri loads with EVICT_LAST 0.5551 vs 0.5553 (L512), 2.3207 vs 2.3201 (L1024): null; b7g ring dgp stores EVICT_LAST +0.15 %, EVICT_FIRST +1 % / +1.1 %: no redundant HBM write-back of the ring or re-read of tri to remove. A redundant x vs xn read in b7g cannot be removed either: x-hat from xn needs 1 / g_in.
- Not built, with their measured bounds: 32 KB A boxes in the consumer stream (the WHOLE W1 stream removed is -0.7 %; the 32 KB-op variants of ctr and k3w were level, see above); LN_out fold (-0.7 / -1.0 % bound).
Final tree vs the pristine base, sustained (three rounds, alternating): D128 L512 0.5573 / 0.5578 / 0.5583 -> 0.5547 / 0.5555 / 0.5554 ms (-0.45 %, J/step 0.5537 -> 0.5494), L1024 2.3134 / 2.3163 / 2.3129 -> 2.3080 / 2.3091 / 2.3091 (-0.2 %, J/step 2.2937 -> 2.2870).

**Ladder and H100 lists at version 77** (`_tools/ladder2.py`: my cuda records, measured in length bands 128-512 / 640-1024 / 1280-3072 like the planned token runs, two rounds, minimum, against the b200 triton / pytorch tokens and the h100 cuda tokens): ladder cuda < triton < pytorch (2 %): 0 violations of 294 cases; slower than H100: 0 (the first all-lengths-in-one-process run read L128 hidden-128 D64 eval 26.5 us vs 23.8 us H100: a measurement-order artefact, the pristine tree reads the same 26.46 us in that process layout and 20.35 us alone, and the banded run 20.4 us).
H100 1.5x misses (H100 >= 20 us): 17 of 294 (eval 12: D64 L128 x 4 variants 0.77-0.85, D128 L128 0.77-0.78, bidirectional D64 L128 0.75, D64 L256 0.73-0.74, D256 L128 0.69-0.70, hidden-128 D64 L256 0.667; train 5: D64 L128 / L256 0.69-0.92 (the 0.916 row reads 53 us alone: another order artefact)). Eval L640 D64 (0.671 in the first run) passes in the banded run. The remaining gaps are 1-3 us at L128 / L256 where the L2 flush kernel's tail, the graph prologue (rng fill 1.4 us + 3 us graph start before the first kernel) and three launches already cost 6-8 us.

### n2-trimul pass (2026-11-18, `cuda:b200_sm100` 78 on TriangleMultiplication / RF3 / Bidirectional; inference part 78, training untouched)

Brief: the 1.5x-vs-H100 misses at L128 / L256 (eval 12, train 6-8 of 294 cases) are 0.6-5 us each; the only levers left are fewer / shorter launches
in the chain. Method: per-kernel device-span timelines from CUPTI (`_tools/tl3.py`: 24-40 graph replays after the measurer's L2 flush, median start /
duration per kernel, first start -> last end = "span"), the event-time distribution of the measurer (`_tools/hist.py`), launch-gap microbenchmarks
(`_tools/gap2.cu`), a weight-stream microbenchmark with a consumer (`_tools/wmc_b.py` / `wmc_r.py`), same-process interleaved A/B (`tl3.py` with
`TL_DEFS`, alternating processes, two rounds), bitwise comparison of outputs (`_tools/xchk.py` / `xcmp.py`).

**Launch counts and device-span timeline of the miss cases (tip 77, us, start offset / duration; measurer = event time of the case alone, graph, L2 flushed):**

| case | launches | per kernel (start / dur) | span | measurer | needed (H100 / 1.5) |
|---|---|---|---|---|---|
| D64 L128 one direction | 1 | tf_kernel 0 / 11.9 | 11.9 | 16.2-16.3 | 13.5 (in) / 14.1 (out) |
| hidden-128 D64 L128 | 3 | k1h 0 / 6.8, ctr 5.4 / 6.1, k3hf 7.5 / 8.7 | 16.2 | 20.35 | 15.9 / 16.0 |
| bidirectional D64 L128 | 3 | k1h 0 / 6.75, ctr 5.4 / 5.9, k3hf 7.4 / 8.8 | 16.2 | 20.3 | 18.0 |
| D128 L128 | 3 | k1w 0 / 8.1, ctr 1.6 / 11.1, k3w 8.7 / 10.3 | 19.0 | 22.4 (24.4 in one process order) | 19.1 / 19.4 |
| D256 L128 | 3 | k1w 0 / 14.1, ctr 2.9 / 17.8, k3w 18.1 / 12.9 | 30.7 | 34.7 | 32.9 / 33.6 |
| D64 L256 | 3 | k1w 0 / 10.6, nvjet 11.1 / 5.3, k3hf 16.5 / 10.3 | 26.7 | 28.5 | 27.7 / 27.9 |
| hidden-128 D64 L256 | 3 | k1w 0 / 14.4, ctr 11.3 / 11.4, k3hf 19.4 / 12.7 | 32.1 | 36.7 | 36.67 |
| train D64 L128 | 7 | rng 0 / 1.7, forms 5.0 / 2.4, tft 6.8 / 14.4, b1u 18.8 / 11.2, cgb 21.6 / 13.5, b7m 29.8 / 13.6, grad_reduce 42.9 / 4.2 | 47.1 | 51.7 (53.2 in sequence) | 49.8 / 50.0 |
| train hidden-128 L128 | 9 | rng, forms 2.7, k1w 7.8, ctr 11.9, k3g 12.6, b1v 17.2, cgb 15.5, b7m 17.8, grad_reduce 4.4 | 62.4 | 68.0 | 64.8 |
| train D64 L256 | 9 | rng, forms 2.6, k1w 11.4, nvjet 4.9, k3g 12.9, b1u 18.7, cgb 14.7, b7m 23.8, grad_reduce 5.2 | 87 | 92.5 alone / 101-107 in sequence | 94.3 |

Blockers measured (none of them a removable launch or Python glue: every miss chain already has only the kernels its dependency structure needs; no copy, cast, zero-fill or elementwise kernel is left in an inference graph; the training graphs have the torch RNG fill, `forms_kernel` (PDL, 0.6 us on the path) and `grad_reduce`):
- Launch floor: flush end -> first kernel 2.8 +- 0.5 us whatever the kernel (shared memory 0-200 KB, 128 / 512 threads, grid 128-1024, cluster 4, 700 B of parameters: `gap2.cu`), kernel -> kernel inside a graph 1.1 us (0.9 us with the PDL attribute: `gap.cu`); the training replay has the engine's `rng_scalar_write` (1.4-1.9 us) in front and the first graph kernel 3.1-3.7 us after it.
- Every dependent stage costs 4.6-5.1 us after its producer (b1u -> cgb -> b7m -> grad_reduce: 8.8 / 5.1 / 8.3 / 3.7 us after the previous kernel's end; ctr 4.7 us after k1h, k3hf 4.6 after ctr): a grid completion + L2 hand-off + a cold TMA load + MMA + epilogue + store; measured earlier as not removable by a flag or counter hand-off (HANDOFF section 5).
- The event timer has a 2.048 us tick (confirmed: D256 L128 reads 32.70-32.78 = 16 ticks or 34.5-34.9 = 17 ticks, D64 L128 16.0-16.5 = 8 ticks, a distribution of 400 replays blurred by +-0.2 us of jitter, `hist.py`); the reading is the true time rounded to a tick, so a gain of 0.6 us of span moved D256 L128 by one whole tick (2.0 us) in a fresh process and by none in a sequence process.
- Placement noise is larger than the gaps: the same chain reads 28.5 or 30.6 us (D64 L256 eval: a 1 MB offset of the allocator base makes `k1w` 10.2 -> 10.9 us), 92.5 us alone or 101-107 us inside the measurer's case sequence (train D64 L256), D128 L128 outgoing 22.4 alone and 24.4 in sequence. A flip at the 1.5x line in a token run is therefore partly the physical page layout of the process; the table below is from alternating runs and the sequence runs `_out/fin2_*.jsonl`.

**Built (version 78, inference; results bitwise equal to 77: 11 + 11 + 11 inference cases, repeat-bitwise, and 6 training cases outputs + gradients equal to the pristine tree, `_tools/xchk.py` / `xchk_g.py` + `xcmp.py`):**
0. `k3w.cu` epilogue without the dropout scale (all D >= 256 and the D128 cells, inference without `ds`): the epilogue multiplied every channel pair by the constant bf16 1.0 and unpacked its two bf16 words (the scale is read from global only when `ds` exists); the loop is now a generic lambda over `std::true_type / false_type` (`dsr == nullptr`), the multiply and the unpacking are gone (3 of 14 instructions a channel pair; the same bits, x 1.0 is exact). k3w body at D256 L128 13.0 -> 12.0 us (alternating processes), D384 L128 19.5 -> 18.6, D128 L128 9.9 -> 9.7. In fresh single-case processes (3 rounds x 2 directions, alone in the process) the measurer reads D256 L128 34.69 / 34.70 us (pristine tree) -> 32.70-32.78 us (this tree): 17 -> 16 event-timer ticks of 2.048 us, ratios 0.702 / 0.688 -> 0.662 / 0.651 (the 1.5x line needs 32.9 / 33.6); in the measurer's case sequence the same code reads 34.69 again (the cases before it change the state by one tick: the true time sits at ~33.8 us, 0.5 us from robust), see the placement note above.
1. `k3w.cu` (D256 one direction, `K3W_X2_256 = 1`): the item epilogue pipeline. The next item's per-channel vectors and mu_i / rs_i are prefetched into registers during the current item and written to a second vector buffer; the output TMA store is issued by the residual-loader warp (warp 3) from a `y_ready` mbarrier, so no epilogue thread waits for the store's read; two residual / output tiles (the stage ring shrinks from 5 to 4). The timeline showed the second item's epilogue starting 1.3 us after the first one's end (global load of the vectors behind the store-read wait, then the reload of the single residual tile). CUPTI span (alternating processes, us, old -> new): D256 L128 30.6 -> 30.2, L256 84.5 -> 82.0, L384 176.0 -> 168.8, L512 301.8 -> 286.0, L640 500.3 -> 473.3 / L1024 -4 %; measurer (`fin2` vs the pristine tree): D256 L256 -2.3..-4.4 %, L384 -3.7..-3.9 %, L512 -5.0..-5.3 %, L640 -5.3..-5.7 %. Bidirectional D256 (12 stages an item), D384 and D512: mode 1 is +2..3 % slower at L256-L512 (the ring is the limit), mode 2 (one tile, prefetch + warp-3 store) level: `K3W_X2_256B / 384 / 512 = 0`.
2. `ctr.cu` (L128 with more than 2 x SMs items: D384, D512, bidirectional D256): `ctr_kernel<128, 1, 2>`, a single-stage ring (K = 128 is one stage an item), 99 KB of shared memory, two CTAs per SM (256 TMEM columns each): D512 L128 75.6 -> 73.6 us, D384 L128 -0.4 us, bidirectional D256 L128 46.9 -> 45.0 us (CUPTI span -0.4..-0.9). D256 (256 items): +0.6 us, so it stays on one CTA per SM (env `TRIMUL_CTR_TWO=0` for A/B).

**Before / after of the miss cases** (measurer, graph, L2 flushed, exclusive GPU 0; sequence runs `_out/base_l*.jsonl` -> `_out/fin3_l*.jsonl`, fresh-process runs `tl3.py`): D256 L128 in the measurer's sequence 34.78 / 34.75 -> 34.69 / 34.69 us (0.702 / 0.688, still a miss), alone in a fresh process 34.69 / 34.70 -> 32.74 / 32.77 us (0.662 / 0.651: below the line); D64 L128, hidden-128 L128, D128 L128, bidirectional L128, D64 / hidden-128 L256 and all training rows are unchanged kernels (0.69-0.85, 0.67 edge) and read the same within +-2 us of placement noise. No case crossed 0.667 and none is slower than the H100 (`ladder2.py` on `_out/fin3_*.jsonl`: 147 cases L128-L640; the list shows bidirectional D128 L128 28.4 us and bidirectional D64 L256 38.8 us as new misses / worse readings in that sequence run: the same kernels read 26.5 / 36.7 us in fresh processes, base and new alike, `tl3.py` alternating: placement noise of +-2 us, not a change); ladder cuda < triton < pytorch (2 %): 0 violations. Whole-grid comparison L128-L640 (119 cases, pristine tree vs this tree, sequence runs): geomean 0.9945; D256 one direction -3.5 % (L256) .. -5.1 % (L512, L640), D384 L128 -3.9 %, D512 L128 -2.4 %.

**Accuracy basis.** The kernels' arithmetic and rounding are unchanged: outputs of the changed paths are bitwise equal to version 77 (`_tools/xchk.py` / `xcmp.py`, random non-zero weights from the measurer's `randomize`, real shapes: eval L128 D256 / D384 / D512 both directions, bidirectional D128-D384, L256 / L384 / L640 D256 and D384, 22 cases, each run twice in a process: repeat-bitwise). `measure.accuracy` (fp64 gold, `_tools/acc_n2.sh`) eval L128 / L256 / L384 / L512 of the three classes: 84 of 84 GREEN (TriMul 48, RF3 16, bidirectional 20); `tests/gpu/test_trimul_b200.py`: 179 passed (pytest shim). Training kernels untouched (no training version bump). `_prebuild --arch sm100 --check`: 795 artifacts, 0 not built (the full output `jobs/n2-trimul-pre2.log`: 204 "no compiled kernel matches" lines, all `*_sm90`, 0 non-sm90; 4 `requirements family ... other recipes` mpnn lines; rc 1 as on the pristine tree).

**Built, measured, rejected (diffs in `docs/b200/rejected/n2_*.diff`):**
- `k1w` sigmoid without the MUFU (ablation, wrong values) -0.4 us at D64 L256, 0 elsewhere; `tanh.approx.f16x2` +0.4 us (two elements per op is slower here): the front epilogue is not MUFU-bound (`n2_k1w_sigmoid_ablation_f16x2.diff`).
- Final TMA-store wait `.read` only in ctr / k1w / k1h / k3h / k3g / k3w: level (+-0.05 us on six chains; `n2_final_tma_store_wait_read_only.diff`).
- L2 prefetch of the weight groups at `k1w` entry (every CTA one group): -0.1..-0.2 us at D256 L128, level at D384 / D512 (`n2_k1w_weight_l2_prefetch_at_entry.diff`).
- `ctr` epilogue with both TMEM halves loaded before one wait: level on six chains (`n2_ctr_both_tmem_halves_one_wait.diff`).
- Two epilogue warp groups in `k3w` (items alternate between 2 x 8 warps; 768 threads): without `setmaxnreg` the launch bound caps the registers at 80 (STACK 136 B, +2.7 us); with `setmaxnreg` 40 / 64 / 88 and 16-channel steps STACK 216 B and k3w 12.5 -> 27.5 us (D256 L128); `setmaxnreg` 96 is refused by ptxas (C7600); the first run of the kernel with a 96 / 56 / 64 register split deadlocked in `setmaxnreg.inc` (the pool is only what the other warps release: 128 x 24 + 128 x 16 = 5120 < 8192), bounded by `dev.sh -t`, no process killed by hand (`n2_k3w_two_epilogue_groups_setmaxnreg.diff`).
- `k3w` epilogue halves (ro, ri, ep, eg pre-halved, y update `fma(tanh(g/2), p/2, p/2)`, 9 instead of 11 f32x2 / MUFU instructions a pair; not bitwise): D256 L128 -0.04 us, D384 -0.6 us span: not kept (`n2_k3w_epilogue_halves_one_fma.diff`); the same in `k3hf`: -0.1 us (`n2_k3hf_epilogue_halves_one_fma.diff`). `k3w` y written straight to global by the epilogue threads (no staging tile, no TMA store): +1.5 us at D256 L128 (`n2_k3w_y_direct_global_stores.diff`). `k3w` Wp' halves of the first item's t stages issued before `griddepcontrol.wait` (only the t boxes after it): +0.8 us span at D256 L128 (30.4 vs 29.6, `n2_k3w_wp_halves_before_wait.diff`).
- Existing switches re-measured on the new chains: `K3W_EPW=16` +2.5 us at D256 L128 (80 registers, spills), `K3W_GREG=1` +2.3 us at D256 L128 (+2.3 / +5 us D384 / D512), `ctr` two CTAs per SM at D256 L128 +0.6 us.
- Weight-stream microbenchmark with a consumer (`wmc_b.py`: 128 CTAs, 512 KB, consumer spin = MMA time of the stage): plain ring 16 KB x 6 23.7 B/clk, 32 KB x 3 32.3, 48 KB x 2 38.6, 64 KB x 2 / x 3 38.9 (spin 0: 53 / 70 / 78 B/clk): with the consumer in the loop the stream is bound by a per-stage fixed ~600 clk, not by the box size or the ring depth; staggering every CTA's start offset (`wmc_r.py`) changes nothing (not an L2 hot spot); multicast with a plain `mbarrier.arrive.shared::cluster` (the earlier "2-6x slower" result used `.release.cluster`) is level with the plain stream (52.0 vs 50.2 B/clk at 32 KB stages, cluster 2). Bigger groups for `k1w` D256 need an 8-slot ring (128 KB) that does not fit with the 64 KB x ring and the correction boxes (234.5 KB > 227 KB).

**What remains (all with measured blockers, none with a built fix):** D64 L128 one-launch kernel 11.9 us = 1.06 launch + 2.05 cold x + 2.8 front + 1.8 hand-off + 1.8 contraction / exchange + 1.5 output + 1.0 write drain (pass 10): 2.1-2.8 us short of the line at 16.2 us; hidden-128 / bidirectional D64 L128 are 3 x 4.6-4.9 us dependent stages (-4.4 / -2.4 us needed); D128 L128 -3.3; D256 L128 is flipped only in a fresh process (0.5 us short of robust; its k3w is stream-bound in item 0: 256 KB through the ring in 4.9 us, 27 B/clk, then two sequential item epilogues of 1.5 / 1.3 us); the lever with the right size is more epilogue warps with more registers (needs a leaner epilogue: 121 registers today) or a 2-SM tcgen05 weight stream (k1w weight intake 36 B/clk per SM is the front's limit; cta_group::2 halves the weight bytes a CTA receives: not built). Training D64 L128 / L256 are 7-9 dependent launches at 3.7-8.8 us each.

## FP8 (opt-in): e4m3 planes between the front and the contraction (2026-11-19, n13-fp8-b)

OFF by default (bit-identical to the bf16 chain, 24 / 24 cases eager + graph). `FOLDJET_ENGINE_TRIMUL_FP8=calib` (`b200_fp8.POLICY`): `k1w_f8.cu` writes the planes as e4m3 (power-of-two per-tensor scales, token order shuffled inside 16-token groups so the epilogue is one `stmatrix.trans`),
`ctr8.cu` contracts on `tcgen05 kind::f8f6f4` (1.7-2.0 PFLOP/s burst vs cuBLAS bf16 1.35-1.45; the scale sa sb applied in its epilogue; MN-major e4m3 operands for the incoming direction; 3-D maps clip at any L), K3 unchanged.
Module (graph replay, same process, 3 rounds): -5 .. -10 % from L512, -10 .. -18 % at L1024-3072 (time and sustained energy), level at L <= 256 (D64 one direction gated at L512); update error 3.8 % against float64 (bf16 0.6 %, the H100's e4m3-plane error). Details, tables, rejected variants: docs/b200/fp8.md.
