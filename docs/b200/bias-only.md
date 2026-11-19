# Bias-only token DiT and TokenPairInit on B200 (sm_100a): round 3

Scope: `TokenDiTMiniWorldBiasOnly` (`modules/token_dit.py`; `kernels/bias_only_dit/cuda`, `kernels/integrations/bias_only_dit*.py`) and
`TokenPairInit` (`modules/token_pair_init.py`; `kernels/token_pair_init/cuda/sm100`) on the NVIDIA B200 (capability 10.0, card key
`b200_sm100`), branch `b200r3-bias-only`. The wiring and the first optimization round are in [token-dit.md](token-dit.md); the H100 pages
are [../h100/bias_only_dit.md](../h100/bias_only_dit.md). Judged as in [../measure.md](../measure.md).

Round 3's rule (2026-11-17): on every requirement case both cards' `cuda` backends serve, the B200 median is at most the H100 median
/ 1.5 (ratio B200 / H100 <= 0.667; graph timing, L2 flushed, the measurer's speed tokens).

`backend_versions["cuda:b200_sm100"]`: `TokenDiTMiniWorldBiasOnly` 2 -> 3, `TokenPairInit` 1 -> 2. No class `version`, no other card's
entry, no requirement row changed. Every change is behind `torch.cuda.get_device_capability(...) == (10, 0)` or in an sm_100a-only file.

## Result

Measured on the B200 (GPU 3, `dev.sh -x`, `measure.speed`, graph timing, L2 flushed, 50 iterations; two full runs of the bias-only block, three
of TokenPairInit) against the H100 `cuda` speed tokens (`judge/tokens/h100_sm90`); "before" = the B200 tokens at the start of the round.

| class | cases | over 0.667 before -> after | slower than H100 before -> after | worst B200 / H100 before -> after | geomean before -> after |
|---|---|---|---|---|---|
| TokenDiTMiniWorldBiasOnly | 96 | 38 -> 0 | 4 -> 0 | 1.213 -> 0.661 | 0.618 -> 0.495 |
| TokenPairInit | 12 (11 under the rule) | 9 -> 0 (part 2; part 1: 2) | 1 -> 0 | 1.014 -> 0.661-0.666 (float32 eval L128, exempt; 0.626 under the rule) | 0.738 -> 0.497 |

A case whose H100 median is below 20 us is exempt from the rule since 2026-11-17 noon (the launch floor of a CUDA graph is ~6 us here and the
timer steps in 2.05 us): TokenPairInit float32 eval L 128 (H100 18.3 us; B200 12.1-12.2 us since part 2, inside 1.5x as well) is the one such
case of these classes. Part 2 (TokenPairInit, below) closed the last case under the rule: float32 eval L 384 24.4 -> 22.3-22.4 us (35.8 on the
H100, ratio 0.623-0.626).

The closest bias-only cases (us; B200 before -> after, H100, ratio):

| case | before | after | H100 | ratio |
|---|---|---|---|---|
| r104 float32 eval L128 S5 | 157.7 | 110.5 | 167.2 | 0.661 |
| r106 (pair 256) float32 eval L128 S5 | 153.6 | 115.0 | 175.4 | 0.656 |
| r104 master train L256 S1 | 292.9 | 165.7 | 257.0 | 0.645 |
| r105 (cond 768) float32 eval L128 S5 | 167.9 | 119.2 | 186.7 | 0.639 |
| r107 float32 eval L128 S5 | 162.7 | 123.3 | 195.6 | 0.630 |
| r104 master train L128 S1 | 276.4 | 143.2 | 227.8 | 0.629 |
| r104 master train L768 S1 | 450.6 | 311.2 | 514.0 | 0.605 |
| r104 master eval L128 S5 | 112.0 | 82.4 | 155.7 | 0.529 |
| r104 float32 eval L768 S5 | 460.6 | 305.7 | 592.5 | 0.516 |
| r104 master eval L768 S5 | 290.9 | 188.2 | 452.4 | 0.416 |

TokenPairInit (us, three runs' range where they differ; "part 1" = after the first half of the round, "part 2" = now, `_bo/out/p3_sp_tpi_{c,d,e}.jsonl`):

| case | before | part 1 | part 2 | H100 | ratio (part 2) |
|---|---|---|---|---|---|
| float32 eval L128 / L256 / L384 | 16.2 / 20.3 / 26.5 | 14.2-16.2 / 18.3 / 24.4 | 12.1-12.2 / 16.2-16.3 / 22.3-22.4 | 18.3 / 30.0 / 35.8 | 0.66 / 0.54 / 0.62 |
| master eval L128 / L256 / L384 | 26.3 / 28.5 / 34.7 | 16.2 / 18.3 / 26.5 | 12.0-12.2 / 16.2-16.3 / 24.4-24.5 | 26.0 / 36.1 / 46.7 | 0.47 / 0.45 / 0.52 |
| float32 train L128 / L256 / L384 | 51.0 / 77.7 / 141.3 | 40.8 / 57.2 / 118.7 | 46.9-47.0 / 55.3-56.8 / 79.6-79.8 | 78.8 / 132.4 / 186.0 | 0.60 / 0.42 / 0.43 |
| master train L128 / L256 / L384 | 63.3 / 86.0 / 155.5 | 41 / 59.4 / 121-123 | 46.9-47.1 / 57.2 / 81.7-81.8 | 88.4 / 141.0 / 199.6 | 0.53 / 0.41 / 0.41 |

The training step at L 128 is ~6 us slower than after part 1 (40.8 -> 47.0): part 1's backward used atomics (not repeatable); the repeatable
backward pays a second (reduce) launch, which dominates at the smallest length. It is still 0.60 of the H100.

The ladder (cuda < triton < pytorch) holds on every case of both classes against the B200 triton / pytorch tokens (no case within 2 % of triton).

Accuracy (`measure.accuracy`, every requirement case, fp64 gold): TokenDiTMiniWorldBiasOnly 34 GREEN / 62 YELLOW / 0 RED (before the round
35 / 61 / 0: r104 master train L128 went 0.99 -> 1.14 of its tolerance with the seven-launch backward tail; H100 cuda 1.01, B200 triton 1.40
on that case, both YELLOW); TokenPairInit 12 GREEN.

Determinism (`_bo/det.py`: the step twice eagerly and three CUDA-graph replays, outputs and every gradient bitwise; then every parameter
scaled in place and a replay against a fresh eager step): TokenDiTMiniWorldBiasOnly 96 / 96 cases bitwise repeatable, eager and replay the
same bits, the replay following the new weights; TokenPairInit (part 2): 12 / 12 cases clean, training included (z and all four gradients
bitwise repeatable; `_bo/out/p3_det_TokenPairInit.txt`; before part 2 d right / d w_rel / d w_bond were not, from `tpi_bwd`'s atomics).
TokenPairInit accuracy after part 2: 12 GREEN (`_bo/out/p3_acc_TokenPairInit.jsonl`); `_bo/tpi_check2.py` (fp32 / bf16 inputs, int32 / int64 /
int16 ids, odd L and B, L 1 to 512, two runs bitwise): all ok.

Records: `$W/b2511/agents/r3-bias-only/_bo/out/fin_sp_<Class>_{a,b,c}.jsonl`, `fin_cmp_{a,b}.txt`, `fin_acc_<Class>.jsonl`,
`fin_det_<Class>.txt`; the CPU gate (`gate.sh`: ruff 0, pyright 0 errors, 3382 CPU tests passed) in `_bo/gate/`.

## What a captured replay pays for

Three facts about the B200 under the measurer (a CUDA graph replayed with the L2 flushed) decided the changes; all measured on the box
(`_bo/tpit.py`, `_bo/tl.py` + nsys `--cuda-graph-trace=node`):

* **A replay re-runs the weight pack.** A pack is scoped to the capture (the weights may change between replays), so every kernel of it is
  part of the step. It is bandwidth bound: the B200 fills fresh memory at 7.3 TB/s (`zero_` of 75 MB: 10.3 us over the 6.1 us floor of a
  one-kernel graph), and a pack of 7 M fp32 elements moves 56 MB. Packing less is the only lever; ordering packs differently is not
  (two packs side by side share the bandwidth).
* **Small kernels are latency, not work.** A cuBLAS GEMM of 640 x 384 x 1536 is 5.9 us, a 640-row LayerNorm 2.6 us, a one-kernel graph
  6.1 us: launch + one or two memory round trips each (the flushed L2 makes the first touch of every line an HBM round trip, ~1 us).
  A step captured on one stream is a chain of them.
* **A graph runs its branches concurrently.** Kernels that do not fill the card overlap when they are on different streams of the capture.
  The event timer of the box has a 2.05 us quantum: medians move in steps of it.

## TokenDiTMiniWorldBiasOnly

| change | where | measured |
|---|---|---|
| The inference pack makes only what the three-kernel step reads: the AdaLN table weights in that step's order only, the tail's pair-packed `Wo` / `Wsq` for the cluster of the step's row count only (another cluster on first use: `_TailForms`), none of the 12-launch step's natural forms (`_step_natural` makes them on first use); the pack kernel issues four vector loads per thread before its stores | `runner._pack_b200`, `bo_pack_sm100.cu` (`UNR`) | r104 L128 S5: pack 36 (master) / 42 (fp32) us -> 14.5 us |
| The one-block conditioning table in one pass: the two table GEMMs write straight into the table, `bo_tab_sm100.cu` adds `[b2 | b1]` and applies the four sigmoids in place (was two biased GEMMs, a cat and a strided sigmoid) | `runner._tab_b200`, `bo_tab_sm100.cu` | 26 us of r104 master L512 S5 -> 17.7 |
| The pair hoist in one pass at pair 128 too: bf16 the training block's `pair_bias` kernel (LayerNorm on chip + the H biases), fp32 the tensor-core `pair_bias_tc` | `runner.hoist` (`_B200_HOIST1`, `_B200_HOIST_TC`) | r104 L512 S5: master LayerNorm rows + GEMM 39.0 + 13.1 us -> 21.4; fp32 FMA kernel 90.7 us -> 40.7 |
| **The pair part as a parallel branch of the captured graph** (`bias_only_dit/cuda/side.py`): inference -- the pair projection's pack and the hoist (pair bias + softmax) on their own stream beside the table pack, the conditioning LayerNorm and the table GEMMs, joined before the front kernel; training -- `pair_bias` + `softmax_t` beside the conditioning / AdaLN / value chain, and in the fused backward `pair_bias_bwd` beside the middle of the step and the weight gradients, finalize after the join (the one-launch `pair_bias_bwd_fin` waited for the weight gradients; eager calls run the same two launches on one stream, so eager and captured steps give the same bits) | `runner._pack_b200` / `hoist` / `_join_hoist`, `bias_only_dit_train._fwd_impl` / `_bwd_fused_step` | r104 (us): fp32 eval L128 116.6 -> 110.5, master train L128 154.6 -> 143.3, L768 333.7 -> 311.2; 96 cases: 6 over 0.667 -> 0 |
| Training backward tail as its seven launches (`tail_mode` "off"; the three `bo_bwd_tail` launches are a serial chain on 10-30 CTA pairs, ~120 us at every L on torch 2.13's cuBLAS), without the four transposed weights in the pack | `bwd_fused.tail_mode`, `bias_only_dit_train._pack_one` | r104 master S1 graph step 274.5 / 323.6 / 452.6 -> 182.9 / 227.4 / 360.5 us at L128 / L384 / L768 |
| `bo_wgrad`: fp32 outputs leave through the warp's 4 KB of the idle operand stages (whole 128-byte lines per store instruction), the unfold's `Wraw` rows and column sums through a second 4 KB | `bo_wgrad.cu` | 11.5 us of a 27.6 us launch at M 128 -> 15.6 us launch |
| fp32 backward at <= 1024 rows: weight gradients as one plain GEMM each, the data-gradient GEMMs on the weights as packed (no column-major copies), `dh` + the SwiGLU backward as cuBLAS + rows | `bias_only_dit_train._small32` / `_wgrad32` / `_cm` | r104 fp32 L128: 16 of 50 launches gone (~110 us); 307.2 -> 213 us |

Inside a capture the side streams follow three rules (`side.py`): a fork is ordered after everything launched before it; a tensor a side
kernel touches stays referenced until the join; nothing on the main stream reads a side kernel's output before the join. Eager calls and
the other cards never fork. Only CUDA-core kernels of the engine run on a side stream (no cuBLAS call).

Rejected (measured, reverted):

| tried | result |
|---|---|
| The step-weight pack forked after the table pack (so the two do not overlap) | r104-r107 fp32 eval L128: 110.5 / 116.7 / 118.5 / 123.3 us against 110.5 / 119.2 / 114.8 / 123.1: no gain (the packs are bandwidth bound either way) |
| The training pack as two launches, the late one (Wraw, Wvg, Wab, Wo, Wsq) on a side stream | r104 master train L128-L768: 147.2 / 163.8 / 190.2 / 223.0 / 266.0 / 311.1 us against 143.3 / 165.7 / 190.3 / 223.2 / 264.1 / 311.2: no gain |
| The conditioning LayerNorm rows on a stream forked before the pack | r104 fp32 eval L128 113.1 against 110.5: no gain |
| The two table GEMMs as parallel branches of the graph (disjoint columns of the table; means of 300 replays, `_bo/ab_bo.py`) | fp32 slower: r104 L128 110.0 -> 110.6, L384 153.1 -> 157.0, r106 L128 112.5 -> 114.8 us; master 0-2 us faster (r104 L384 102.9 -> 101.5, r106 L384 114.1 -> 112.2): reverted |

## TokenPairInit

| change | where | measured (graph, us) |
|---|---|---|
| `tpi_fwd_w` reads bf16 `left` / `right` as they are (exact widening in the kernel): no cast launches before it | `token_pair_init.cu` (`ld_lr`), `modules/token_pair_init.py` | master eval L128 / L256 / L384: 26.3 / 28.5 / 34.7 -> 16.2 / 18.3 / 26.5 |
| A training step on fp32 weights runs the one-launch forward (`tpi_fwd_w`) and a one-kernel backward on the same tensors (`tpi_bwd_w`: the five id tensors as they are, `dleft` in the input's dtype, `dright` and the table gradient in one zero-filled buffer; the weight gradients are transposed views of the table gradient): no `stack` / cast / `cat` prep, no bf16 casts, one fill. The backward's kernel is loaded during the forward (autograd's device thread has no driver context for a module load) | `function.TokenPairInitFunctionW`, `sm100.backward_w` | train L128 / L256 / L384: fp32 51.0 / 77.7 / 141.3 -> 40.8 / 57.2 / 118.7; master 63.3 / 86.0 / 155.5 -> 41 / 59.4 / 121-123 |
| `tpi_fwd_w`: 16 warps a CTA (128 registers a thread) and every load issued one step before its value is needed, none used before the loads beside it are issued -- the first two items' ids and bond, the first item's `left` rows and first `right` group and the warp's class rows before the table is staged; inside an item the second `right` group before the first is written, the next item's rows before the second; work items in 32-bit arithmetic | `token_pair_init.cu` | fp32 eval L128 (harness means, us): 16.4 -> 14.4 |
| Three table rows per pair: the chain-class, same-entity and bond rows are added once per CTA into 4 (2 s_max + 2) combined rows (the benchmark's ids change the class at every position, so the five-row sum ran per pair) | `token_pair_init.cu` | fp32 eval L128 / L256 / L384 (same build, 5 rows -> 3, means): 14.5 / 18.4 / 26.8 -> 14.4 / 16.6 / 24.7; measurer 16.2 / 20.3 / 26.5 -> 14.2-16.2 / 18.3 / 24.4 |
| Part 2. **Bitwise-repeatable backward** in two launches (the H100 structure): `tpi_bwd_w` -- streams of four warps, each with a PRIVATE class table in shared memory (two table rows per pair: merged residue / token class, chain class x bond), units of 32 positions of a row split evenly and statically over the streams, row sums per unit and column sums per (stream, column chunk) as partials -- then `tpi_bwd_red`, the partials added in a fixed order (`dleft` / `dright` in the inputs' dtypes, the table gradient over the CTAs; terms loaded 32 at a time). No atomics, nothing zero-filled | `token_pair_init.cu` (`bwd_body`, `tpi_bwd_red`), `sm100._bwd_plan` / `_bwd_launch` | train fp32 L128 / L256 / L384: 40.8 / 57.2 / 118.7 -> 47.0 / 55.3-56.8 / 79.7; master 41 / 59.4 / 122 -> 47.0 / 57.2 / 81.7; repeatable 12 / 12 |
| Part 2. The weights reach shared memory by **bulk copies** (`cp.async.bulk` + mbarrier, issued first by one thread), and one pass builds the table rows from that raw copy (one barrier). The per-lane transposing loads (32 floats of a channel per load, each straddling two lines) kept the warps waiting on the SM's outstanding misses, and four warps of every CTA took two round trips (`-DTPI_TRACE`) | `token_pair_init.cu` | L384 a launch back to back (`_bo/tpiab.py`: 20 launches after one flush): 19.35 -> 18.53 us |
| Part 2. **Two table rows per pair**: the 2 (2 r_max + 2) - 1 merged residue x token rows (the token class is "other" unless the residue offset is 0 in the same chain), built in the same pass; the shared-memory reads share the LSU with the z stores (z without any table read: 17.6 us at L384, three rows 18.5) | `token_pair_init.cu` | measurer float32 eval L384 24.4 -> 22.3-22.4 us; back to back 18.53 -> 17.82 (with the merged rows read once per thread / four in flight) |
| Part 2. A **branch-free item path** (runtime `pipe`, `sm100.fwd_w_pipe`) where the grid has at least two warps per item: every pair's rows looked up, the reads of the next positions issued ahead of this one's stores (a warp's single item is latency-bound) | `token_pair_init.cu`, `sm100.fwd_w_pipe` | float32 eval L128 14.3 -> 12.1-12.2 us (back to back 9.2 -> 6.4); L256 / L384 without it (the path measured slower there: 10.65 -> 10.96, 17.8 -> 18.4) |

`tpi_fwd_w`'s z differs from `tpi_fwd`'s in the last bit or two (the order of the table rows' sum); both are exact fp32 sums of the same
terms. `tpi_bwd_w` is `tpi_bwd`'s body on another id loader.

Rejected (harness means of 200 replays, `_bo/tpit.py`): 32 warps a CTA (64 registers a thread: the held loads spill; L128 / L384 20.5 /
32.8 us against 14.3 / 24.5), 8 / 12 / 20 / 24 warps (slower from L256), the class rows staged channel-per-lane (strided loads: every
scalar its own line, staging 4-6 us against 2), sixteen `right` loads in flight per item (slower at every L), two CTAs per SM (72 KB of
table each: 22.5 / 26.7 / 30.7 us), a touch of the second `right` group's lines before the staging (14.6 / 18.4 / 25.5 against 14.4 /
16.6 / 24.7), the weight gradients written in the weights' own layout by the backward kernel (four scalar atomics per chunk on 128 lines
of the weight: train +4 / +14 / +4 us at L128 / L256 / L384).

Rejected in part 2 (`_bo/tpiab.py`, back to back / single flushed replay, us): the weights loaded per lane with all of a warp's 40 loads in flight
before any store (L384 24.5 -> 26.6: the SM's outstanding misses, not the round trips, limit the staging), plain or `L1::no_allocate` z stores
instead of `st.global.cs` (no change), the first item's ids issued before the bulk copies (L384 17.82 -> 18.02, L128 6.35 -> 6.55), the
branch-free path at L384 / L256 (above), an smem transpose of the raw copy with a division per element (2.2 us of prologue: replaced by a
thread-per-channel pass).

## Gaps

| class | cases | why | what it would take |
|---|---|---|---|
| TokenPairInit | none over 0.667 (float32 eval L128 0.661-0.666 is exempt: H100 18.3 us; float32 eval L384 0.623-0.626) | the z write is the floor: a fill of the same 75 MB is 16.4 us under the measurer, the kernel 22.3; the rest is the prologue (first id loads issued ~1.2 us after entry -- a cold instruction cache --, the table built at ~3.1 us) and ~2.6 TB/s less than a fill during the stores (LSU shared by the table reads, the right loads and the stores) | the table kept outside the replay for static weights is not allowed (a replay follows in-place weight updates); fewer LSU ops per z store (a 4-row item halves the right loads but leaves warps idle at L384) |
| TokenDiTMiniWorldBiasOnly | margins (unchanged by part 2; re-measured after the merge, `_bo/out/p3_fin_sp_bo.jsonl`): float32 eval L 128 0.62-0.66 on the four rows (r104 110.4 us against 111.4 allowed, r106 115.0 against 116.9), master train L 256 0.645, L128 0.629; r104 master train L128 accuracy YELLOW 1.14 x tol (H100 1.01, B200 triton 1.40: inside the rule, not brought back) | at 5 x 128 rows the step is the tail kernel (40 us: five clusters of 8 CTAs, 40 of 148 SMs busy), the front (19 us), the two table GEMMs on cuBLAS (5.9 + 8.0 us), the table pass (5.9 us) and the weight pack (14.5 us: 56 MB per replay, bandwidth bound) | front / tail builds for few rows that spread a 128-row tile over more SMs (64-row tiles or more column groups); a table GEMM kernel of the engine (both GEMMs + bias + sigmoid in one launch); static weights (no pack per replay) |
| TokenDiTMiniWorldBiasOnly | YELLOW: 62 of 96 (float32 TF32 paths and master training gradients at 1.0-1.9 x tol, as before the round and as on H100 / triton for most) | TF32 tensor cores against the TF32 reference; bf16 gradients | not a speed item |

## H100 technique audit (2026-11-17, rule: port every H100 hand-CUDA technique that fits sm_100a)

TokenPairInit (`kernels/token_pair_init/cuda/sm90/token_pair_init.cu` against `sm100/token_pair_init.cu`):

| H100 technique | B200 |
|---|---|
| class-row table by ONE bulk (TMA) copy overlapping the bins' packing | used (part 2): bulk copies of the raw weights, overlapping the id / left / right loads; the table rows built from them in shared memory |
| `right[j]` loaded once for the R rows of a CTA | used: one register copy of `right[j]` serves the item's two rows (R > 2: a 4-row item halves the right loads but leaves warps idle at L 384 -- not taken) |
| `tpi_prep`: the ids / table glue in ONE launch before the forward | superseded: the B200 forward reads the five id tensors and the weights as they are (no prep launch: one launch where the H100 has two) |
| backward: private per-warp class-bin gradient tables in shared memory, no atomics; a fixed-order reduce launch (dright over row blocks, dW over CTAs) | used (part 2): `tpi_bwd_w` streams with private tables + `tpi_bwd_red`; bitwise repeatable 12 / 12 |
| backward reads dz once | used |

TokenDiTMiniWorldBiasOnly (`docs/h100/bias_only_dit.md`, `bias_only_dit/cuda/runner.py` `_step`, `integrations/bias_only_dit_train.py`):

| H100 technique | B200 |
|---|---|
| inference 12-launch step (`_step`) | superseded by the three-kernel tcgen05 step (`bo_front_*`, `pv_gate`, `bo_tail_*`, PDL-chained) |
| hoist of the pair bias once per pair / mask, `pair_bias_all` (one LN0 + one projection for every block), fp32 `pair_bias_tc` (TF32, split fp32) | used (`runner.hoist`, `_B200_HOIST1` / `_B200_HOIST_TC`; on a side stream of the capture) |
| `cond_tables`: both conditioning-table GEMMs in ONE launch (bf16, <= 384 table rows) | missing on B200: two cuBLAS GEMMs + `bo_tab_sm100` (the one-GEMM fp32 form was measured in part 2: GEMM 13.9 -> 10.8 us but the table pass 5.9 -> 9.6, no gain, reverted); an sm_100a tcgen05 port of `cond_tables` is open (Gaps) |
| `pvx` / `pvx32`: samples of a group as the N dimension, P read once per (head, rows, group); gate by TMA before the epilogue | used (`pv_gate_inf`, `bo_pv_gate_tf32_sm100`) |
| `dpb2` / `dpb_pipe` (dbias tiles, dsum precomputed) | used as `dpb_sm100` / `dpbx2_sm100` (CTA pairs at L768) / `dpb32_sm100` (fp32) |
| `gemm_swiglu` (expand GEMM + SwiGLU epilogue) | used: fused inside `bo_tail_*` (inference), `gemm_swiglu2_sm100` keeping a \| b (training) |
| training fp32: `gate_bwd_fold` (dO head-major, samples folded) so dV^T and dP are one batched GEMM each | used (shared path, `bias_only_dit_train.py`) |
| training: weight pack in one launch, `finalize` fixed-order reduction of per-block partials | used (`bo_pack_sm100`, per-replay pack; fixed-order finalize) |
| weight gradients one cuBLAS GEMM each | used (fp32 <= 1024 rows: `_wgrad32`); bf16 `bo_wgrad` |
