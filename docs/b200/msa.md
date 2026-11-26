# OuterProductMean and MSAPairWeightedAveraging on B200 (sm100)

Scope: the `"cuda"` backend of `modules.OuterProductMean` and `modules.MSAPairWeightedAveraging` (`modules/msa.py`) on an NVIDIA B200
(capability (10, 0), card key `b200_sm100`). The route is `modules.norm.b200(t)` in `_supports_cuda` / `_forward_cuda`, next to the A100 /
RTX 5090 routes; the H100, A100 and RTX 5090 paths are unchanged. `backend_versions["cuda:b200_sm100"] = 1` for both classes.

The kernels are the B200 branch's (`origin/b200/ladder-final`, `docs/gpus/b200/opm/opm.md`, `docs/gpus/b200/pwa/pwa.md` there), brought
into the 2.5.x layout as sm_100a-only files:

| file | what |
|---|---|
| `kernels/integrations/opm_sm100.py` | OuterProductMean (groups 1) on B200: forward / backward ops, padding, batch, `count_offset`, interchain mask, projection biases |
| `kernels/outer_product_mean/chai_sm100.py` | Chai-1 OuterProductMean (8 groups of 8, product LayerNorm) on B200 |
| `kernels/integrations/pwa_sm100.py` | MSAPairWeightedAveraging on B200 (MiniWorld MSA-side build and the wide build) |
| `kernels/integrations/csrc/sm100/opm_sm100.cu` | OPM at d_msa 64 / d_pair 128: prologue, epilogue, counts, dWo, prologue backward, weight cast (built by `opm_train._ext(arch="sm100")`) |
| `kernels/integrations/csrc/sm100/opm_wide_sm100.cu` | OPM wide widths (d_msa 128 / 256, d_pair 256 / 384 / 512), projection biases, CTA-pair dgrad / epilogue, Chai-1 passes |
| `kernels/integrations/csrc/sm100/pwa_sm100.cu` | PWA pair side (`pair_fwd`, `pair_bwd`, `pair_dw_fix`), MiniWorld MSA side (`ln_vg`, `pwa_gate_out`, `pwa_glue2`, `dv_bwd`) |
| `kernels/integrations/csrc/sm100/pwa_wide_sm100.cu` | PWA MSA side at the other widths, one build per (d_msa, d_hidden): `ln_vg_w`, `gate_out_w`, `glue_w`, `dvb_w` |

The prebuild (`kernels/integrations/__init__.py`, target `sm100`) lists all of them.

## OuterProductMean

Forward: prologue (LayerNorm + both projections (+ RoseTTAFold3's biases) x mask -> A2 [(i,c), s], BT [(j,e), s], the LayerNorm
statistics in training, the mask as bit rows), cuBLAS O = A2 BT^T, the pair divisor table [N, N] fp32 (`opm_norm`: max(n, 1) off the
bit rows; with `count_offset`: n + offset from one cuBLAS GEMM of the mask, exact integer counts), epilogue (either normalisation order,
+ bias, + residual, interchain mask). Backward: CTA-pair dgrad at every c_z (`opm_wdgrad`), dA / dB on cuBLAS with dWo beside them on a
side stream (16 CTAs, SM carve-out for the GEMMs), prologue backward (projection-bias gradients = masked column sums of dA / dB).
A length not a multiple of 64 / a depth not a multiple of 128 is padded with masked rows (256 until 2026-11-17); a batch runs one stack at a time.
Chai-1: `opm_chai_prologue`, cuBLAS batched products, `opm_chai_epilogue` (product LayerNorm + W' GEMM), `opm_chai_lnbwd`,
`opm_chai_prologue_bwd`; any length / depth (padded to 32 / 128), d_pair 128 / 256.

The 2.5.11 H100 large-depth route (cuBLAS product + count-table epilogue past S 4096) is not needed on B200 for the requirement rows:
their deepest MSA is 2048. The B200 route is "cuBLAS product + epilogue on a count table" at depth > 512 and in training; shallow
inference takes the branch's fused product + projection kernel (below).

## MSAPairWeightedAveraging

MiniWorld's MSA side (d_msa 64, 8 x 32: Boltz-2 / RoseTTAFold3 / MiniWorld at d_pair 128; Chai-1's d_pair 256 in training) runs
`pwa_sm100.cu`; the other widths (d_msa 64 / 128 at 8 x 8 / 16, d_msa 64 x 32 inference, IntelliFold-v2 256 / 512 / 8 x 32) run
`pwa_wide_sm100.cu` with the same pair kernels. S and N are padded to 128 (pad keys masked); N after the pad <= 2048.

### Accuracy work (2026-11-16)

As brought over, 29 of the 70 training cases were YELLOW (grad of `pair`, `ln_pair.weight`, `to_bias.weight` at 1.0-1.6x the module's
own error; the H100 path has the same tensors YELLOW). Attribution by emulating the pair backward in PyTorch (`_b2/exp_pwa.py` style:
swap one rounding at a time) showed three sources, in order: the bf16 row of logits in `pair_fwd`, the bf16 softmax w16 read by the
softmax backward, and the single bf16 rounding of Wb o gamma in `pair_fwd`'s folded LayerNorm. Fixes, all in `pwa_sm100.cu`:

- `pair_fwd`: Wb o gamma as two bf16 terms, hi in the B operand's rows 0-7 and lo in rows 8-15 (the padding of the 8 heads to N 16), so
  the same single MMA gives z . hi and z . lo, summed in fp32 (no extra tensor-core work).
- `pair_fwd` in training (N <= 1024): the row of logits in fp32 (`PF<..., L32>`, the z ring two units shorter) and an fp32 copy of the
  softmax (`w32`).
- `pair_dw_fix`: dw <- (w32 / w16) (dw - sum_j w32 dw), so that `pair_bwd`'s w16 (dw - sum_j w16 dw) is the softmax backward with the
  fp32 w; dw = do . v^T now comes out of cuBLAS in fp32.

The inference forward shares the first fix (Wb o gamma hi + lo).

## Measured (B200, GPU 1)

Records on the B200: `$W/b2511/agents/msa/_b2/out/` (`acc_all_OuterProductMean.jsonl`, `acc_pwa_v2.jsonl`, `spd_*.jsonl`).

- Accuracy (cuda, master): OuterProductMean 263 GREEN, 0 YELLOW / RED, 4 skipped (the reference ran out of memory on the shared GPU: r56 L2048, r57 L2048 D384 / D512, r58 L1536); MSAPairWeightedAveraging 122 GREEN, 0 YELLOW / RED, 18 skipped (same cause; re-run queued as `re_acc.jsonl`). Before the PWA fixes 29 of the 70 training cases were YELLOW.
- Speed (CUDA graph, median ms): cuda faster than triton on every measured case. OPM inference 1.1-2.9x (triton / cuda), training 1.6-6.1x; PWA inference 1.6-8.3x. PyTorch is 2.4-11x slower than cuda. PWA training speed: still running when this was written (`spd_MSAPairWeightedAveraging_train.jsonl`).

## Gaps

- Every requirement row of both classes routes on the B200 (eval and train, all lengths and depths); no refusals at the requirement shapes.
- PWA training past N 1024 keeps the bf16 row of logits (no shared-memory room for the fp32 row at NMAX 2048); no requirement row trains past 768.
- The triton backend is slower than PyTorch at some cases (OPM training at d_msa 128 / 256: e.g. 256/512 L128 2.0 vs 1.16 ms; 128/384 L640+). Fixing that is a Triton job, not part of the cuda route.
- The 2.5.11 H100 large-depth OPM route (S > 4096) has no B200 counterpart; no requirement row goes deeper than 2048. The branch kernel `opm_fused_infer` (d_pair 64 only) is not used.

## Optimization round (2026-11-17)

Ladder at the start of the round (`$W/b2511/ladder/ladder.txt`, tuned B200 triton tables, graph timing): no violation for either class
(OuterProductMean 222 cases, MSAPairWeightedAveraging 98 with cuda + triton + pytorch). The closest cases were OuterProductMean
inference at d_msa 64 / d_pair 128: r57 (depth 384) cuda / triton 0.95-1.02 at every length, r55 / r56 / r58 / r63 at L 1536 / 2048
0.90-0.97; training r236 (depth 384) 0.90. MSAPairWeightedAveraging: cuda / triton <= 0.71 everywhere.

### Changes

- **Triton PWA pair backward on sm_100 (the "misaligned address" of r232 / r233 training).** `_pwa_bwd_layernorm_gemm_softmax_kernel`
  computes `dzy = dl . Wb` as a [BJ, DZP] tile. With BJ >= 64 Triton 3.7 lowers it to tcgen05 with the fp32 accumulator in tensor memory.
  At DZP 512 (d_pair 384 / 512) that accumulator takes all 512 TMEM columns, and the A operand's TMEM copy lands past them: the kernel
  compiled for sm_100 reports 520 columns at BJ 128. TTGIR checked on the B200's CPU: BJ <= 32 compiles to mma.sync with no TMEM.
  Fix: an `early_config_prune` (`_sm100_tmem_safe`) keeps only BJ <= 32 configs on capability 10.x when DZP >= 512, and a second
  default (BJ 32) gives the prune something to keep. Every other GPU keeps every config, with BJ 64 still first. Verified on the GPU:
  128/384/8x8 and 256/512/8x32 training at L768 now run. The pretune re-tuned rows 232-233 with the fix (`63767be9b` in b200-2.5.x).
- **OuterProductMean: depth padded to 128 instead of 256.** Every sm_100a kernel takes S % 128; the multiple of 256 came from the H100
  integration. At depth 384 a third of the work was padding. A / B at S 384 (min of 3): inference 0.88-0.91, training 0.84-0.98
  (L 128-768). Outputs are bitwise equal in inference and the gradients are within 7e-7 relative.
- **OuterProductMean: the branch's fused product + projection kernel (`opm_fused_infer`, `opm_fused_sm100<128>`) for shallow inference.**
  It is served for c_z 128, the clamp-1 divisor, no interchain mask, with or without RoseTTAFold3's projection biases (through the wide
  prologue), at depth <= 512 (S 384: every length; S 512: from N 512) (`opm_sm100._fused_serves`). The depth is padded to 128 there.
  Fused / unfused through the module, same session, min of 3:
  S 384 0.77-0.89 at every length; S 512 0.93-0.98 from N 512 (1.01 at N 256);
  S 1024 / 2048 0.97-1.17 and unstable from run to run, so those are not served (the branch saw the same "long-K gap" of the fused mainloop).
  `cuda:b200_sm100` 2.

### Measured after (B200 GPU 1, measure.speed graph timing, L2 flushed; records `$W/b2511/agents/opt-msa/_opt/out/`)

| case | before cuda / triton (tokens) | after cuda (ms) | triton (ms) | cuda / triton after |
|---|---|---|---|---|
| r57 L128 D384 eval | 0.059 / 0.066 | 0.041 | 0.061 | 0.67 |
| r57 L768 D384 eval | 0.677 / 0.732 | 0.539 | 0.680 | 0.79 |
| r57 L1024 D384 eval | 1.211 / 1.201 | 0.965 | 1.175 | 0.82 |
| r57 L2048 D384 eval | 5.122 / 5.244 | 4.091 | 5.062 | 0.81 |
| r236 L128 D384 train | 0.143 / 0.207 | 0.131 | 0.207 | 0.63 |
| r236 L768 D384 train | 2.751 / 3.056 | 2.071 | 2.955 | 0.70 |

Accuracy (measure.accuracy, cuda, master): r55-r58 eval 36 / 36 GREEN, r57 + r236 15 / 15 GREEN (`acc_opm_eval.jsonl`,
`acc_d384.jsonl`). Runs are deterministic (same bits twice through the module, fused and unfused).

### What is left (OuterProductMean inference at L 1536 / 2048, depth 1024: cuda / triton 0.90-0.97)

Profile at L2048 D1024 (d_msa 64 / d_pair 128, cuda 6.80 ms): cuBLAS product 5.22 ms (65536 x 65536 x 1024, 1.69 PFLOP/s, about the
B200's power-capped bf16 rate), epilogue 1.43 ms (reads the 8.6 GB product at about 6 TB/s), prologue 0.09 ms, counts 0.04 ms. Triton
runs the same cuBLAS product (5.26 ms) plus a 1.93 ms epilogue, so the gap is the epilogue only. That is close to the HBM bound.
The only way past it is not to write the product at all (the fused kernel), and its mainloop loses about 5-15 % against cuBLAS at
K >= 1024. Closing that would take a fused mainloop at cuBLAS efficiency for long K (e.g. 2-CTA 256 x 256 tiles with a deeper stage
ring, or split-K over S with an fp32 reduction of the 128-column z accumulator), not a launch-config change.

## Round 3 (2026-11-17): B200 cuda >= 1.5x H100 cuda on every case

Target: B200 cuda median <= H100 cuda median / 1.5 (ratio <= 0.667; graph timing, L2 flushed). At the start (`vs_h100_15x.txt`):
OuterProductMean 12 of 222 cases over the ratio (worst 0.908), MSAPairWeightedAveraging 9 of 98 (worst 0.846). Every one is an
inference case at L 128-384 except one OuterProductMean training case at L 128: 50-130 us in total, four to seven launches, each launch
a few us of fixed cost.

### OuterProductMean (`cuda:b200_sm100` 3)

- `opm_norm` writes max(n, dmin) + offset itself, so the `count_offset` table comes off the prologue's bit rows (it was a cuBLAS GEMM of
  the mask + a clamp + an add: three launches more; the same fp32 values).
- The count table runs on a side stream beside the cuBLAS product (`_product_and_counts`).
- The projection biases reach `opm_wprologue` as two segments of the cast buffer (no `cat` + copy launches); `opm_cast_bf16` is vectorized.

Per-kernel at L128 depth 1024 (64 / 128, projection biases): cast 2.3, prologue 12.2, counts 4.5 beside the product 25.1, epilogue
10.0 us; graph 59.4 -> 55.2 us. 128 / 384 with `count_offset`: 77.8 -> 65.4 us.
After (measure.speed, `_r3/out/spd_opm_wip1.jsonl`, every row at L128, inference and training): 0 cases over 0.667, worst 0.645
(r56) / 0.640 (r58); training worst 0.624. Accuracy 32 / 32 GREEN (`acc_opm_v3.jsonl`). The fused kernel at depth 1024 measured
1.02-1.04 of this path at L 128 / 256 again, so it stays at depth <= 512.

### MSAPairWeightedAveraging (`cuda:b200_sm100` 2)

- **Inference: `pair_fwd` on a side stream on SMs of its own.** The pair side (z, key mask -> w) and the MSA side (weight cast, `ln_vg`)
  are independent until the bmm; they are two branches of the CUDA graph, joined before the bmm. Side / serial 0.96-0.98 at L 128-768.
  With both kernels asking for the whole GPU they queue on each other's SMs, so the grids are disjoint (`_PAIR_CTAS`): `pair_fwd` 32
  CTAs up to N 256, 48 up to N 512, `ln_vg` the rest; past N 512 and at d_msa 256 (IntelliFold-v2, a long MSA side: split / whole
  1.00-1.06) each asks for the whole GPU. Split / whole GPU each: N 128 0.94-0.97, N 256 0.94-0.98, N 384 0.94-0.98, N 512 0.93-1.01.
  A narrower `pair_fwd` starves (16 CTAs at N 384: 1.2-1.3); a narrowed `pair_fwd` beside a whole-GPU `ln_vg` is worse than no split
  (N 128: 1.1-1.3). Outputs are bitwise equal to the serial order at every split. Training keeps the serial order (not measured).
- **Wide build: one weight-cast launch** (`opm_cast_bf16` over Wv, Wg, Wo) instead of three `.to(bf16)`.
- **Wide build, d_hidden 8: the head-major o / v tiles as flat TMA rows.** A head's 128 MSA rows of 16 bytes are 2 KB contiguous in
  o / v and in the unswizzled staging tile: a (256, 4) box of 512-byte rows instead of 128 rows of 16 bytes (`head_map`). 64 / 8 L128
  depth 1024: graph 53.1 -> 51.1 us.
- **Wide build, d_msa 64 / 8 x 8: deeper stage rings** (`ln_vg_w` 3 -> 6 x stages, `gate_out_w` 2 -> 4): L128 16.2 -> 15.7 and
  18.7 -> 16.7 us, L256 20.8 -> 18.9 and 35.8 -> 29.9 us. d_msa 128 / 8 x 8 with 4 / 3 measured the same as 3 / 2 (kept).

Rejected (measured, reverted):
- `ln_vg` / `ln_vg_w` casting the fp32 Wv into shared memory themselves (no cast launch before the MSA side, Wg / Wo cast on the side
  stream): 0 to +4 us on every case (L128 64 / 32 / 256 79.7 -> 81.7, L256 64 / 32 / 128 71.5 -> 75.6); bitwise the same output.
- The wide (64, 32) build instead of MiniWorld's kernels for 64 / 8 x 32 at d_pair 128: the same time (L128 48.9 / 48.9, L256 71.5 / 71.5).

After (measure.speed, every inference row at L 128-1024: `_r3/out/spd_pwa_f2.jsonl`; training `spd_pwa_f1.jsonl`), 9 -> 4 over 0.667:

| case | B200 before (ms) | B200 after (ms) | H100 (ms) | ratio after |
|---|---|---|---|---|
| r49 64 / 8 x 32 / 128 L128 depth 384 | 0.0511 | 0.0496 | 0.0604 | 0.821 |
| r51 128 / 8 x 8 / 256 L256 | 0.1330 | 0.1433 (0.119-0.121 in every other run) | 0.1910 | 0.750 (0.63) |
| r51 128 / 8 x 8 / 256 L128 | 0.0798 | 0.0742 | 0.1076 | 0.689 |
| r48 64 / 8 x 8 / 128 L128 | 0.0593 | 0.0496 | 0.0725 | 0.684 |
| r53 128 / 8 x 8 / 384 L128 | 0.0818 | 0.0742 | 0.1148 | 0.647 |
| r48 L256 / L384 | 0.0921 / 0.1269 | 0.0754 / 0.1049 | 0.1266 / 0.1895 | 0.596 / 0.554 |
| r49 L256 | 0.0757 | 0.0721 | 0.1094 | 0.660 |
| r50 64 / 8 x 32 / 256 L128 | 0.0859 | 0.0783 | 0.1265 | 0.619 |

r51 L256 measured 0.1432 / 0.1433 ms in the two runs of the whole class (rows 48-54, L 128-1024, in one process) and 0.1187-0.1214 ms
in six other runs (the row alone, the row after r48 / r49 / r50, every row at L 128-384, r50 + r51 at every length, the whole class
with a profiler attached). The previous token (0.1330) against 0.1216 for the same code in a shorter run shows the same dependence on
what ran before in the process; it is not understood (the replayed graph is the same kernels with the same grids; only the buffer
addresses differ).
Training is unchanged in time (0.95-1.01 of the tokens, every ratio <= 0.584). Accuracy: 56 / 56 GREEN (28 inference + 28 training
cases at L 128 / 256 / 384 / 768, `acc_pwa_f1.jsonl`).

### What is left and what it would take

The four cases are bound by fixed launch cost and by HBM traffic, not by tensor-core time. r49 at L128 depth 384: cast 1.7 us,
`ln_vg` 14.8 (m 6 MB in, v 25 MB out), bmm 10.6 (25 MB in, 25 MB out: cuBLAS at 4.7 TB/s), `pwa_gate_out` 15.0 (o 25 MB, m 6 MB in,
6 MB out), about 5 us around the graph: 118 MB of traffic for 12 MB of input + output. 0.667 needs 40.3 us.

- A fused contraction + gate pass that never writes o: tile = one MSA row x 128 tokens, per head one tcgen05 GEMM
  o_h = w_h [128 n x N j] . v_h[s] [N j x C] straight into the TMEM columns the gate reads, then the existing gate / out-projection.
  It removes the bmm launch and 50 MB of the 118 MB (r49), and needs w streamed per head (32 KB per head and tile at N 128, from L2) and
  v written s-major by `ln_vg`. A rewrite of `pwa_gate_out` / `gate_out_w`; worth it only up to N of a few hundred (w per tile grows
  with N).
- d_msa 128: `ln_vg_w` moves 51 MB in 26 us (1.9 TB/s) against 33 MB in 16 us at d_msa 64; the one-thread-a-row LayerNorm over a
  128-wide row in shared memory is the suspect (no Nsight Compute counters on this box: ERR_NVGPUCTRPERM).

## Round 3, part 2 (2026-11-17 afternoon): PWA fused contraction, the wide-build LayerNorm, process-history timing

### MSAPairWeightedAveraging (`cuda:b200_sm100` 3 -> 4)

- **Version 3 hung in training.** The two-thread register LayerNorm of `ln_vg_w` / `gate_out_w` (version 3) sized every
  `(2 + LNWARPS + 4) * 32` kernel in `pwa_wide_sm100.cu` for eight LayerNorm warps, `glue_w` (the training backward, which has no
  LayerNorm) included: its warps 10-13 ran the drain branch for heads 8-11 and arrived on barriers counted for eight warps; every
  training call of the wide build (d_msa 64 / 8 x 8, 128 / 8 x 8 / 16, 64 / 8 x 32 at d_pair 256) deadlocked. `glue_w` is back to 320
  threads. Determinism (`_r3/det.py`, every width, eval + train) and accuracy (56 / 56 GREEN: 28 inference + 28 training cases at
  L 128 / 256 / 384 / 768, `_r3/out/acc_pwa_v4.jsonl`) after the fix.
- `gate_out_w` keeps the one-thread LayerNorm at d_hidden 32 (64 / 8 x 32 at d_pair 256, L256 depth 1024: 147.2 us two threads a row
  against 143.1 us one); d_hidden 8 / 16 keep the two-thread form (69.4 / 71.5, 106.4 / 110.4, 129.0 / 131.0 us).
- `ln_vg_w` at d_msa 128 (item 2 of the task): the two-thread register LayerNorm of version 3 took it from 22.6 to 17.7 us at L128
  depth 1024 (the one-thread-a-row LayerNorm over a 128-wide row in shared memory was the bound, as suspected).
- From the H100 path (audit below): the backward's seven parameter gradients leave in one `_cast_cuda.cast(..., force=True)` launch
  instead of seven `.to(copy=True)` (training, every row and length: geomean 0.983 of the version-4 times, `full_pwa_train_v9.jsonl`;
  L128 depth 384 0.1752 -> 0.1661 ms).

Rejected (measured, reverted; the patch is kept in the worktree as `_r3/rejected/pwa_cgate.patch`):
- **The fused contraction + gate pass** (`pwa_cgate`, inference at N 128, the (64, 8 x 32) MSA side at every d_pair): tile = two MSA
  rows x all 128 tokens; per head O_h = w_h [128 i x 128 j] . v_h[j, s0 .. s0 + 1, :] on tcgen05 (B MN-major: the two rows of head-major
  v are 64 contiguous values, so v stays as `ln_vg` writes it), the gate GEMM of both rows, u = bf16(sigmoid(bf16 G) bf16(O)) back to
  TMEM, out += u . Wo_h^T over the heads; w_h (32 KiB) + v_h (16 KiB) + Wg_h (4 KiB) streamed per head through a 2-3 stage ring.
  Outputs bitwise equal to cuBLAS bmm + `pwa_gate_out` at d_pair 128. Kernel time against bmm + gate pass (64 / 8 x 32 / 128, L128):
  depth 128 16.2 vs 5.2 + 9.1 us, depth 384 26.4-28.1 vs 10.7 + 15.3, depth 1024 47.6-52.8 vs 17.9 + 30.2; graph 36.6 / 48.9 /
  75.6-79.8 vs 34.6 / 48.9 / 75.6 us. A globaltimer trace of one CTA (`CG_TRACE` build): 1.1 us a head step, 8 steps a tile, every
  step waiting on its stage's TMA: a 52 KiB stage lands 2.2-2.7 us after it is issued with 148 CTAs loading at once, and three stages in
  flight (all the shared memory there is next to the y tile and Wo) cover only ~3 of those. Variants measured the same: head order
  rotated per CTA (all CTAs on different w_h), w loaded once instead of per head (wrong results, the cost of streaming w: none),
  two stages + two y buffers, two stages + one. Making it win needs fewer, larger steps per tile (four MSA rows a tile halves the
  tiles but needs 896 TMEM columns) or w resident (a CTA pair holding four heads each, partial outs exchanged over DSMEM).

After (B200 GPU 3, measure.speed graph timing, L2 flushed, every row and length; records `$W/b2511/agents/r3-msa/_r3/out/`:
inference `full_pwa_eval_v5.jsonl` and `full_pwa_eval_v5b.jsonl` (two whole-class runs), training `full_pwa_train_v5.jsonl`):

| | above 0.667 | slower than the H100 | worst |
|---|---|---|---|
| inference, 56 cases (run v5 / v5b) | 1 / 2 | 0 | r49 64 / 8 x 32 / 128 L128 depth 384: 0.0476 ms vs 0.0604 (0.788) |
| training, 42 cases | 0 | 0 | r228 L128 depth 384: 0.587 |

The second inference run also had r51 128 / 8 x 8 / 256 L128 at 0.670 (0.0721 ms; 0.0660 in the first). Ladder: cuda < triton on
every case (worst cuda / triton 0.665 inference, 0.563 training), cuda < pytorch everywhere.

r49 at L128 depth 384 (48.9 us graph against 40.3 for 0.667): pair_fwd 13.3 us on its 32 SMs beside ln_vg 11.4 us, the bmm 10.5 us
(25 MB in, 25 MB out), the gate pass 15.0 us; the fused pass above was the attempt at the last two.

### r51 L256 (item 3): the slow whole-class value is gone
r51 128 / 8 x 8 / 256 at L256 measured 0.143 ms in every whole-class run before version 3 and 0.119-0.121 ms in every other
configuration. Since version 3 it measures 0.1090 ms in all three whole-class runs (v3, v5, v5b; the diagnosis in `_r3/spdx.py`, armed at
0.115 ms, never fired). The change that coincides is the inference weight cast leaving the graph (`_infer_weights`: one launch and three
allocations fewer per call inside the captured graph); a mechanism is not proven.

### OuterProductMean (`cuda:b200_sm100` 3 -> 4)

- **Inference reads cached bf16 parameter forms** (`opm_sm100._infer_flat`, the H100 path's `param_forms`): the flat
  `opm_cast_bf16` buffer (mode -1, or mode 2 with Wo in the fused kernel's k order) is kept on the module, keyed by (storage, version,
  dtype) of the parameters; a miss during a CUDA-graph capture casts inside the graph as before. One launch fewer per call / replay:
  L128 inference (16 cases) geomean 0.962, worst ratio to the H100 0.646 -> 0.621 (`opm128_g3a.jsonl` -> `opm128_v9.jsonl`).
- **The backward's parameter gradients in one cast launch** (OPM 6-7 tensors, Chai-1 8), as the H100 path: training (96 cases)
  geomean 0.995, every case <= 0.663 of the H100 (`full_opm_train_v9.jsonl`).
- Accuracy 32 / 32 GREEN at L128 (eval + train, `acc_opm_v9.jsonl`).

### Whole-class timing runs and the L128 cases (item 3, OuterProductMean too)

OPM inference at L128 measured 0 of 16 cases over 0.667 in every run of the L128 cases alone (three runs on GPU 3 and GPU 0: worst
0.645 / 0.645 / 0.645) and 3-6 over (worst 0.70-1.00, a different case each time: r64 0.1124 / 0.0817 / 0.0717 ms, r61 0.0716 / 0.1185)
in the four whole-class runs. Most of those runs logged another agent's process on the same GPU during the exclusive slot (`dev.sh`:
"foreign pid ... on GPU 3": `_bo/det.py`, `_bo/sp_TokenDiT...` speed runs, `_r/cb3.py` builds); a whole-class run lasts 4-5 minutes
against ~1 for the L128 rows, and the L128 cases (50-120 us) are the ones a concurrent kernel moves most. With the count table on the
main stream (no graph branch) the whole-class run was no better (6 over), so the side-stream branch is not the cause. A token run on a
GPU nothing else touches should land on the L128-alone values; this could not be shown on the shared development GPUs.

### H100 technique audit (the H100 `cuda` route at capability (9, 0) against this B200 route)

OuterProductMean:

| H100 technique | B200 |
|---|---|
| fused product + projection in inference (`opm_fused_kernel`, every depth) | used at depth <= 512 (`opm_fused_sm100`, CTA pair); at depth 1024 measured 0.97-1.17 of cuBLAS + epilogue and unstable: not served |
| counts from the mask bits in the kernel | the count table from the prologue's bit rows (`opm_norm`) on a side stream beside the product: one launch, hidden |
| inference weight forms cached per parameter version (`param_forms`) | **adopted in this round** (`_infer_flat`) |
| one-launch weight pack | used (`opm_cast_bf16`, vectorized; its buffer is kept for the backward, which the H100 re-packs) |
| persistent TMA prologue (LN + both projections + mask + stats + bits) | used (`opm_prologue_sm100`, two CTAs an SM, TMEM accumulators) |
| epilogue (permute, / count, Wo, bias, residual) on the tensor core | used (`opm_epilogue_sm100`, 4-D TMA permute, interchain mask) |
| dgrad into the grouped layout, dbias beside it | used (`opm_wdgrad_sm100`, CTA pairs) |
| dWo split-K + deterministic reduce | used (`opm_dwo_sm100` + `colsum`; on a side stream with an SM carve-out for dA / dB) |
| fused prologue backward + reduce + finish | used |
| all parameter gradients cast in one launch | **adopted in this round** (was six copies) |
| cuBLAS-owned graph workspaces, cuBLASLt algorithm index (depth > 4096 shapes) | not applicable: no requirement case of the B200 route needs them |
| PDL | not used on either card |

Launches per call (64 / 128 / 32, L128 depth 1024): inference H100 2 (prologue + fused), B200 4 (prologue, counts, cuBLAS, epilogue; the
cast is gone); training forward 5 / 5; backward H100 13, B200 10 (was 15).

MSAPairWeightedAveraging:

| H100 technique | B200 |
|---|---|
| pair side LN -> proj -> mask -> softmax in one kernel (`pair_fwd3`) | used (`pair_fwd_sm100`, LN folded into the projection, fp32 row kept for the backward), on a side stream beside the MSA side in inference |
| contraction + gate + out + residual + dropout in ONE kernel below N 512 (`pwa_fwd3`, o never written in inference) | **missing**: built and measured this round (`pwa_cgate`, above), no faster than cuBLAS bmm + the gate pass (TMA stage latency); rejected |
| `ln_vg` (TMA + tensor core, persistent, head-major v) | used (`ln_vg` / `ln_vg_w`, tcgen05) |
| N >= 512: cuBLAS bmm + gate pass | used at every N |
| backward glue with the dWo partials in the kernel | used, further fused (`pwa_glue2`: dgp never leaves the chip) |
| dv on cuBLAS, `dgv_bwd` (dWgv + LN backward + residual) | used (`dv_bwd` + `bwd_finish`); one extra launch (`pair_dw_fix`) for the pair-side accuracy |
| deterministic split reductions | used everywhere (the H100's non-native training route uses `torch.sum` / atomics) |
| weight casts | inference cached (`_infer_weights`), training one launch (`pwa_wcast` / `pwa_wprep`); the H100 casts per call |
| one-launch gradient cast | **adopted in this round** (was seven copies) |
| TMA cluster multicast of W | not applicable: present in `pwa_fwd3.cu` but every H100 call passes cluster size 1 |
| PDL | not used on either card |

Launches per call (64 / 8 x 32 / 128, L128 depth 384): inference H100 5, B200 4 (pair_fwd beside ln_vg, bmm, gate pass); training
backward H100 ~26 (non-native route at this shape), B200 11 (was 17).
Still missing and plausible: the wide build's backward re-casts Wv / Wg / Wo (three `.to`) where the forward's buffer could be saved;
`pair_fwd` beside `ln_vg` in training too (serial now, not measured); PDL between the fixed-cost launches of the L128 cases.

### Final whole-class inference run (merged tree, `full_pwa_eval_v10.jsonl`, `full_opm_eval_v10.jsonl`; another agent's speed run was on the GPU)

| class | above 0.667 before (round-3 start) -> now | slower than the H100 | worst now | ladder |
|---|---|---|---|---|
| MSAPairWeightedAveraging inference (56) | 9 -> 2 (1-2 in the v5 / v5b runs) | 0 | r49 L128 depth 384 0.821 (0.788 in v5 / v5b) | 0 misses |
| MSAPairWeightedAveraging training (42) | 0 -> 0 | 0 | 0.557 | 0 misses |
| OuterProductMean inference (126) | 11 -> 4 (0 of the 16 L128 cases in every L128-only run, worst 0.621) | 0 | r63 L128 0.880 (L128-only: 0.59) | 1 (r63 L128 cuda / triton 1.023; 0.71 in the L128-only runs) |
| OuterProductMean training (96) | 1 -> 0 | 0 | 0.663 | 0 misses |

## Round 3, part 3 (2026-11-17 evening)

### The whole-class L128 instability is the SM clock after the previous case (OuterProductMean and MSAPairWeightedAveraging)

A whole-class speed run measures the rows in order, each row's lengths ascending, pytorch before cuda: every L128 case is timed right after
the previous row's L2048 (or L768 / L1024) case and its own short pytorch L128 baseline. Reproduced on an idle GPU (GPU 2 / 3, nothing
else on the card during the `-x` slot; `_r3/order.py` runs cases in a chosen order and, for every cuda L128 timing, times the same graph
again and a fresh capture of the same step):

| sequence (one process) | r63 L128 cuda t1 / again / fresh capture (us) |
|---|---|
| r63 L128 first | 53.0 / 53.0 / 51.2 |
| after r62 L768 | 74.9 / 67.6 / 67.4 |
| after r61 L2048 | 75.6 / 61.3 / 61.2 |
| after 10 s idle | 52.3 / 51.1 / 51.6 |
| r64 L128 alone / after r56 L2048 | 61.2 / 100.1 (again 63.5, fresh 63.4) |

A fresh capture is as slow as the timed graph, and both recover with time: no state of the process (weight caches, launch tables,
allocator, workspaces, graph contents) is involved. `nvidia-smi` at 100 ms beside the run: during the large case the SM clock falls to
1140-1400 MHz with throttle reason 0x4 (SW power cap; ~940 W). `_r3/clk.py` times the small graph after 1.5 s of the large one, with
the SM clock read on the GPU (`torch.cuda._sleep` of 200k cycles timed by events) before every 20 timed replays:

| ms after the large case | SM clock (MHz) | r63-like L128 graph (us) |
|---|---|---|
| 5-80 | 920-1410 | 69.5-83.8 |
| 85-115 | 1740-1770 | 55 |
| 120-150 | 1800-1840 | 53 |
| >= 180 | 1850-1890 | 51.2 |

(second round: 990-1180 MHz -> 78-82 us for 77 ms, 1507 MHz -> 63 us, 1846 MHz -> 51.2 us at 145 ms). The power governor keeps the
clock down for ~80-180 ms after a power-capped kernel stream ends; a launch- and latency-bound 50-100 us case timed in that window runs
1.3-1.9x slower. Which L128 case is hit depends on how long its preceding steps took (deepcopy, inputs, pytorch baseline), hence "a
different case each time". Nothing in the cuda route can change this; the remedy is in the measurer (a pause, or waiting for the clock
to read back its maximum, before a case's timed replays — e.g. `_Timer.cycles_per_ms()` re-read until it is within 2 % of the idle
value) and is the coordinator's decision (a measurer change re-times every card). The same effect applies to the H100 tokens
(their whole-class order is the same).

### OuterProductMean inference: the H100's two-launch form (prologue + fused product / projection) at depth 1024

Kernel times through the module (B200, idle GPU, CUDA graph, L2 flushed; `_r3/profopm.py`, 64 / 128 / 32):

| case | fused path (prologue + `opm_fused_sm100`) | current path (prologue, counts beside cuBLAS, epilogue) |
|---|---|---|
| L128 depth 1024 | 55.1 us (11.5 + 39.5) | 51.0 us (11.5, 4.5 beside 25.2, 10.2) |
| L256 depth 1024 | 141.1 us (17.1 + 122.9) | 132.9-141.2 us (17.4, 4.4 beside 82.5, 34.4) |
| depth 384, L 128 / 256 / 512 / 768 / 1024 / 2048 | 36.6 / 87.9 / 253.8 / 549 / 986 / 4086 us | 36.6 / 89.9 / 287 / 588-637 / 1113-1165 / 4595-4672 us |

Why the fused kernel loses at depth 1024: its mainloop runs one cta_group::2 256 x 256 x 64 step every ~0.56 us (39.5 us for the 64
steps a CTA pair does at L128: 4 passes x 16 k blocks), against ~0.39 us a step for cuBLAS' `nvjet 128x256_64x6_2x2_2cta` (25.2 us,
4 waves x 16 steps) and ~0.28 us of tensor time. Each CTA has 4 stages of 32 KiB in flight (128 KiB; the Opair chunks (64 KiB) and the
weight slice (32 KiB) take the rest of the 224 KiB), which does not cover the L2 latency of 148 SMs loading at once; cuBLAS has 6
stages and a 2 x 2 cluster that multicasts A. On the B200 the product is short enough at depth 384 that the ring covers it (fused <= unfused at every length there: 1.00 at
L128, 0.88-0.98 above), which is what the route already serves. The "unstable 0.97-1.17" of the earlier A / B is most likely the clock effect above (in this
round's idle-GPU runs both paths repeat within 1 %); the kernel itself is steadily slower at depth 1024. Serving `count_offset` in the fused kernel (the H100 passes dmin / doff) was
built (one fp32 divisor in the epilogue, as `opm_norm`) and reverted: the only `count_offset` row (r63) is at depth 1024, where the fused
kernel loses. Making the fused path win at depth 1024 needs a deeper ring: e.g. Opair in TMEM instead of shared memory, or the 2 x 2
cluster multicast of A2 between two CTA pairs on the same i block (24 KiB a stage from L2 instead of 32).

### MSAPairWeightedAveraging (`cuda:b200_sm100` 5)

- **Training: `pair_fwd` on the side stream beside the weight cast + `ln_vg` at d_msa 64 / 8 x 32** (MiniWorld's MSA side; was serial in
  training). Train step (forward + backward graph, `_r3/proftr.py`): L128 depth 384 167.0 -> 162.0 us, L384 323.8 -> 319.6 us. The wide
  build keeps the serial order (128 / 8 x 8 / 256 L128 158.8 -> 166.0 us with the side stream; 64 / 8 x 8 0.99 / 1.004).
- **Wide build backward: Wg / Wv / Wo cast in one launch** (`_bf16_params`, `opm_cast_bf16`) instead of three `.to(bf16)` (audit item).
- Accuracy 56 / 56 GREEN (28 inference + 28 training cases at L 128 / 256 / 384 / 768, `_r3/out/acc_pwa_v11.jsonl`); determinism
  (`_r3/det.py`, every width, eval + train) OK.

Not taken (measured):
- PDL between the fixed-cost launches: no two of our own kernels are adjacent on the L128 paths (PWA inference ln_vg / pair_fwd -> cuBLAS
  bmm -> gate pass; OPM prologue -> cuBLAS -> epilogue); the only adjacent pair is the fused OPM path (prologue -> fused kernel, 0.2 us
  apart in the trace), which is not served at depth 1024.
- r49 L128 depth 384 (item 3): trace (idle GPU) ln_vg 11.7 us beside pair_fwd 14.3 us (32 SMs), the bmm 10.4, the gate pass 15.1, graph
  47.0 us. A wider pair_fwd split: 40 / 48 / 56 CTAs 45.1 / 44.8 / 46.8 us against 46.9 (one 2-us timer step; r50 L128 depth 1024 73.5
  at 32 / 40 / 48, 75.6 at 56): not kept. The 40.3 us target needs the bmm + gate pass (25.5 us, 75 MB of traffic) replaced by one pass
  that never writes o: the resident-w forms (a CTA pair holding 4 heads' w_h each, or a 4-CTA cluster holding 2 heads each, per-head
  partial outs reduced over DSMEM in a fixed order; 128 + 96 KiB per CTA for the pair form does not fit next to the m / y / v / u
  tiles, the 4-CTA form fits in ~200 KiB) were designed but not built in this part.

### Whole-class runs after this part (B200 GPU 3, nothing else on the card; `_r3/out/full_*_v11.jsonl`)

| class / mode | above 0.667 (part 2 final -> now) | slower than the H100 | worst | ladder |
|---|---|---|---|---|
| MSAPairWeightedAveraging inference (56) | 2 -> 2 | 0 | r49 L128 D384 0.821, r50 L128 D1024 0.776 | 0 misses |
| MSAPairWeightedAveraging training (42) | 0 -> 0 | 0 | r232 L256 D1024 0.659 | 0 misses |
| OuterProductMean inference (126) | 4 -> 7 (all L128 depth 1024, code unchanged: the clock effect) | 1 (r56 L128 1.049) | r56 L128 1.049 | 1 (r56 L128 cuda / triton 1.097) |
| OuterProductMean training (96) | 0 -> 0 | 0 | r277 L128 D1024 0.654 | 0 misses |

The OuterProductMean L128 numbers of this run (r56 89.8 us, r64 94.2 us) are the clock window: the same cases measure 53.2 / 61.2 us
when not preceded by a long case (table above).

### H100 technique audit, updated (part 3)

| class | H100 technique | B200 now |
|---|---|---|
| OPM | inference in 2 launches (prologue + fused, every depth, dmin / doff in the kernel) | fused at depth <= 512 (fused <= unfused at every length there); depth 1024 measured 55.1 vs 51.0 us (L128), 141 vs 133-141 (L256): not applicable at depth 1024 until the fused mainloop has a deeper ring (reason above); dmin / doff built and reverted (no served row) |
| OPM | PDL | not applicable on the served paths (no adjacent own kernels) |
| PWA | side stream for the independent pair side | inference (all widths) and now training at 64 / 8 x 32; the wide build's training measured slower with it |
| PWA | weight casts (training) | one launch everywhere now (the wide backward's three `.to` gone) |
| PWA | contraction + gate in one kernel (`pwa_fwd3`) | missing (part 2's streamed form rejected; the resident-w cluster form not built) |

## Round 3, part 4 (2026-11-17 night)

### MSAPairWeightedAveraging: the w-resident contraction + gate kernel (`pwa_cwr`), built and rejected

Built as designed in part 3 (patch kept in the worktree as `_r3/rejected/pwa_cwr.patch`; inference at N 128, (d_msa 64, 8 x 32), every
d_pair, replacing cuBLAS bmm + the gate pass): a 4-CTA cluster works on one pair of MSA rows; CTA c holds w of heads 2c, 2c + 1 **in
tensor memory** (the O GEMM's A operand from TMEM, 2 x 64 columns, loaded once per CTA: no head step waits for a w stage), and per head
O_h = w_h . v_h[j, s0 .. s0 + 1, :] (N 64, B = v head-major's two rows, a 16 KiB TMA box in a 5-deep ring), G_h = y_s . Wg_h^T for both
rows (every CTA LayerNorms both rows in place), u_h = bf16(sigmoid(bf16 G) bf16(O)) back over O's columns, P_s += u_h . Wo[:, h]^T (the
CTA's partial out over its two heads). The four partials are summed over DSMEM in rank order (CTA c finalizes output channels 16c .. +15
of both rows: `st.async` into the receiver's buffer with complete_tx on its mbarrier, the buffer handed back by a remote arrive), then
bf16 + residual. Outputs: within one bf16 ulp of bmm + the gate pass (mean |diff| 3e-9 against |out| 0.8: the partial sums in another
order), deterministic. Time (B200 GPU 3, CUDA graph, L2 flushed, `_r3/cw.py`):

| case | bmm + gate pass (us) | `pwa_cwr` (us) | graph before / with it (us) |
|---|---|---|---|
| r49 64 / 8 x 32 / 128 L128 depth 384 | 10.3 + 14.7 | 53.5 | 46.9 / 75.6 |
| r49 shape, depth 1024 | 17.9 + 30.0 | 126.3 | 75.6 / 153.4 |
| r50 64 / 8 x 32 / 256 L128 depth 1024 (wide build front) | 17.7 + 28.5 | 127.2 | 73.5 / 153.4 |
| r50 shape, depth 384 | 10.8 + 14.3 | 54.5 | 48.9 / 75.8 |

Globaltimer trace of CTA 0 (`-DCW_TRACE=1`, depth 384, us from kernel start): w in TMEM at 2.6; head steps issued at 3.78, 4.64, 5.86,
7.14, 10.56, 11.87, 16.10, ...; a row pair's partial final at 6.94, 14.78, 19.65, 25.66, 31.65, 37.44; all three peers' slices received
3.4-6 us after (10.30, 16.93, 23.33, ...); y of the third row pair ready only at 10.40. Steady state ~6 us a row pair, 5-6 row pairs a
cluster (6 for CTA 0 at depth 384). Why it cannot win:
- **The tensor pipe is the floor, and the head split multiplies the MMA count.** Splitting heads over the cluster cuts the gate GEMM to
  N 32 per head and the out GEMM to K 32 per head: 40 small tcgen05 MMAs a row pair per CTA (O 2 x 8 at N 64, G 2 x 2 x 4 at N 32,
  out 2 x 2 x 2 at N 64), 160 for the cluster, where the gate pass does a row's G in 4 MMAs at N 256. A head step (O + G + the previous
  step's out, 20 MMAs) takes 1.1-1.3 us from issue to drained (~100 clk an MMA; one commit tracks every earlier MMA, so a step's buffer
  frees only when the next step's MMAs retire): >= 2.4 us a row pair even with every other phase hidden, i.e. >= 12.5 us at depth 384
  and >= 34 us at depth 1024 against 25.0 / 47.9 us for bmm + the gate pass.
- The larger-N forms do not fit: four rows a step (O at N 128) or G of both heads at once with double-buffered steps need w 128 + O 256
  + G 256 + P 256 TMEM columns (512 exist), and shared memory is full (v ring 80 KiB, y 64 KiB, receive buffer 48 KiB, Wg / Wo slices).
- The DSMEM sum (48 KiB out and in per CTA a row pair) and the per-row-pair LayerNorm chain add 3-5 us of latency per row pair on top.
So both forms of the one-pass contraction (part 2 streamed w, now resident w) lose to cuBLAS bmm + the gate pass on the B200; r49 / r50 at
L128 stay where part 3 left them (r49 depth 384 0.78-0.82 in whole-class runs).

### Both classes timed the right way: every length in its own measurer run (part 4)

Code unchanged since part 3 (`cuda:b200_sm100` OPM 4, PWA 5; tree merged with `b200-2.5.x` 8d27e53b7). One `measure.speed` run per
(class, mode, length), each in its own exclusive `dev.sh` slot on GPU 3 (no foreign process on the card in any of the 34 runs), cuda and
triton measured together at L 128 / 256, cuda alone above (records `$W/b2511/agents/r3-msa/_r3/out/len/p4_*.jsonl`, `_r3/len.sh`,
comparison `_r3/cmplen.py`; "before" = the B200 tokens at the start of round 3):

| class / mode | cases | above 0.667: round-3 start -> now | slower than the H100 | worst now | ladder (cuda < triton < pytorch) |
|---|---|---|---|---|---|
| MSAPairWeightedAveraging inference | 56 | 9 -> 1 | 0 | r49 64 / 8 x 32 / 128 L128 depth 384: 0.0476 ms vs 0.0604 (0.789) | 0 misses (worst cuda / triton 0.667) |
| MSAPairWeightedAveraging training | 42 | 0 -> 0 | 0 | 0.544 (r227 L128) | 0 misses |
| OuterProductMean inference | 126 | 11 -> 0 | 0 | 0.623 (r56 L128 depth 1024, 0.0532 ms) | 0 misses (worst r63 L2048 cuda / triton 1.011, inside 2 %) |
| OuterProductMean training | 96 | 1 -> 0 | 0 | 0.613 (r70 L128) | 0 misses |

Every OuterProductMean L128 case is <= 0.623 of the H100 and ahead of triton (worst cuda / triton 0.696 inference, 0.700 training) when it is
not timed in the clock window after a long case: the 7 inference misses and the r56 ladder miss of part 3's whole-class run were that
window, nothing to fix in the route. r50 L128 depth 1024 is 0.0763 ms (0.603). The one case left is r49 at L128 depth 384 (47.6 us vs
40.3 for 0.667): pair_fwd 13-14 us beside ln_vg 11.5, then bmm 10.3 + the gate pass 14.7; both one-pass contraction + gate forms are
measured slower (parts 2 and 4), the pair_fwd split is at its optimum (part 3).

### H100 technique audit (part 4)

| class | H100 technique | B200 now |
|---|---|---|
| PWA | contraction + gate + out + residual in one kernel below N 512 (`pwa_fwd3`) | not applicable: built twice on sm_100a, streamed w (part 2, 26-28 vs 25.5 us) and w resident in TMEM over a 4-CTA cluster (part 4, 53.5 vs 25.0 us at depth 384, 126 vs 48 us at 1024): the head split over CTAs multiplies the small-N tcgen05 MMA count (40 a row pair a CTA, ~100 clk each) and adds a DSMEM sum; the H100 form keeps all heads in one CTA's registers, which TMEM / shared memory cannot hold next to the stages |
| OPM | everything else | unchanged from part 3 |

## Round 3, part 5 (2026-11-17 night, restart brief): the true state, pair_fwd + ln_vg in one launch

### True state at the start (B200 GPU 3, every length in its own `measure.speed` run, cuda + triton + pytorch together; `_r3/out/len/m0_*`)

| class / mode | cases | above 0.667 | slower than the H100 | worst | ladder (cuda < triton < pytorch) |
|---|---|---|---|---|---|
| MSAPairWeightedAveraging inference | 56 | 1 | 0 | r49 64 / 8 x 32 / 128 L128 depth 384: 47.6 us vs 60.4 (0.787) | 0 / 56 + 0 / 56 (worst cuda / triton 0.666, triton / pytorch 0.445) |
| MSAPairWeightedAveraging training | 42 | 0 | 0 | 0.544 | 0 + 0 |
| OuterProductMean inference | 126 | 0 | 0 | 0.627 (r56 L128 depth 1024) | 0 + 0 (worst cuda / triton 0.983, r63 L1536) |
| OuterProductMean training | 96 | 0 | 0 | 0.620 | 0 + 0 |

### Taken: `pwa_dual` (`cuda:b200_sm100` 20)

Inference at (d_msa 64, 8 x 32, d_pair 128), N <= 512: `pair_fwd` and `ln_vg` become one launch on disjoint CTAs (`pwa_dual_fwd_sm100`:
CTAs [0, npair) run pair_fwd's body over the rows i, the rest ln_vg's over the (token, 128-row) tiles; both bodies are the old kernels'
code as `__device__` functions taking the block index / count), instead of two kernels on two streams joined by events before the bmm.
The graph keeps one branch: the event join cost 2.4 us between pair_fwd's end and the bmm's start (trace), and the fork another edge.
Outputs bitwise equal to the two-stream path at every case measured. Graph (L2 flushed, `_r3/ab_dual2.py`): L128 depth 128 34.6 -> 30.5,
depth 384 44.8-46.9 -> 40.8-42.8, depth 1024 73.6-75.5 -> 71.6-73.5, L256 depth 384 71.5-73.5 -> 65.3, L384 98.1 -> 94.0, L512 depth 384
133-135 -> 126.8-128.8 us. CTA split per N (`_DUAL_EXTRA`, CTAs moved from the ln_vg part to the pair part): dual kernel at L128 depth
384 13.1 / 12.8 / 12.9 / 13.7 / 14.3 us at +0 / +8 / +16 / +24 / +32; L256 / L512 +16 best.

### Measured and rejected (diffs in `_r3/rejected/`)

- **PDL on the gate pass** (launched with a programmatic dependency on the cuBLAS bmm; the weights, the msa tiles' LayerNorm and the first
  gate GEMMs before `griddepcontrol.wait`): the gate CTAs became resident only 0.3-0.6 us before the bmm ended (both kernels take a whole
  SM's shared memory) and the gate pass ended at the same time: graph 44.8 / 44.9 us, no case changed.
- **Single-rounding gate (PRE form)**: Wv / 2 and Wg / 2 (exact) so that u = fma(tanh(G / 2), o / 2, o / 2), three instructions a pair
  instead of six; max |diff| to the module's double rounding 1.6e-2 (one bf16 ulp at |u| 2-4, 1.4 % of the elements): graph 40.7 / 40.7
  us at L128 depth 384, the same at every other case. The drain is not issue-bound (an earlier probe that suggested it carried a global
  flag load per element in the loop).
- **Two TMEM loads per wait in the ln_vg / gate drains**: no change (gate 13.8 / 14.2 us).
- **TMA issue from several lanes** (the coordinator's finding: one thread issues one TMA per ~600 clk): the gate pass's nine stage boxes
  from nine lanes, ln_vg's eight v stores from eight lanes, pair_fwd's two z boxes from two lanes: every kernel time within 0.2 us of before
  at every case (L128-512): none of these producers is at that rate here.
- **`cp.async.bulk.wait_group.read` instead of the full wait at ln_vg's exit**: no change.
- cuBLAS forms of the contraction (`_r3/bmmv.py`, N 128 S 384: bmm / bmm(out=) / matmul 16.2 us, two 4-head bmm 20.3, transposed
  operands 16.2): nothing faster than the plain bmm.

### Where r49 L128 depth 384 stands (globaltimer trace of the gate pass, `_r3/rejected/pwa_sm100_trace_probe.cu`)

Graph ~40.8 us (script), 42.9 us in measure.speed (0.711): dual 12.8 us -> bmm 10.2-10.4 -> gate pass 13.8-15.0, ~3.4 us before the first kernel.
Each kernel has a large fixed part: pair_fwd ~7 us + 1.4 us a row of 128 keys (depth 128: 6 / 4 / 3 / 2 rows a CTA = 15.2 / 12.4 / 11.1 /
9.7 us), the gate pass ~6 us + 2.5 us a tile (depth 128: 8.9 us for one tile a CTA). Gate pass trace (median CTA, us from entry): setup
0.5-0.8, msa tile landed 2.3, gate GEMM ready 3.3, o landed 3.6, drain half 0 done 4.6, tile 0 drained 5.6, out of tile 0 5.9, tile 1
8.5-9.0, tile 2 11.3-11.8, end 12.5-13.1: the drain takes ~1 us a half tile (8 warps; not issue-bound, not the o loads), and the
2-stage ring refills a stage ~3 us after it is freed.

### After (version 20, tree merged with `b200-2.5.x` ddd984081; `_r3/out/len/f1_*`, every length its own run, cuda + triton + pytorch)

| class / mode | above 0.667: start -> now | worst | ladder |
|---|---|---|---|
| MSAPairWeightedAveraging inference (56) | 1 -> 1 | r49 L128 depth 384 47.6 -> 42.9 us (0.787 -> 0.711; 40.3 needed) | 0 violations (worst cuda / triton 0.603) |
| MSAPairWeightedAveraging training (42) | 0 -> 0 | 0.544 | 0 violations |
| OuterProductMean (no source change) | 0 / 0 | 0.627 / 0.620 | 0 violations |

Accuracy 56 / 56 GREEN (eval + train, L 128-768, `_r3/out/acc_pwa_v20.jsonl`); inference outputs bitwise equal to version 5.
What r49 still needs (2.6 us): the fixed parts above; the three kernels each take a whole SM's shared memory, so none overlaps the next.

## Round 3, part 6 (2026-11-18): contraction + gate pass in one launch at N 128 (`cuda:b200_sm100` 21)

`pwa_ctr_gate_sm100` replaces cuBLAS bmm + `pwa_gate_out_sm100` for inference at (d_msa 64, 8 x 32, d_pair 128), N 128 (the contraction has
one 128-deep K block). One CTA per SM; phase 0 computes the CTA's contiguous share of o in 64-column units (w_h resident, v boxes in an
8-slot ring, one producer lane a slot, groups of four units as one MMA chain per head at N <= 256), writes bf16 o into the unit's own ring
slot and TMA-stores it over the same v columns (o in place: v is a temporary); a self-resetting grid counter; then the gate pass's pipeline
(Wg and the first two msa tiles load into retired ring slots, their LayerNorm and gate GEMMs run before the counter; the o tile's two
head halves on their own barriers). Launched as a PDL dependent of `pwa_dual` (which triggers its dependents at entry): barriers, TMEM and
Wo are set up on the SMs pwa_dual frees, `griddepcontrol.wait` before w / v. Outputs bitwise equal to bmm + gate pass at every case
measured; accuracy 56 / 56 GREEN (`_r3/out/acc_pwa_v21.jsonl`), determinism OK.

What each step was worth (globaltimer traces, `_r3/cgtrace.py`; graph script `_r3/ab_cg.py`, L128 depth 384, bmm + gate 25.2 us):
- first form (4-slot ring, N 64 MMA a unit, direct 32-byte o stores): 23.3-23.8 us. Trace: phase 0 ~10 us. The v boxes land ~0.8 us apart
  (not the strided pattern: contiguous probe boxes landed the same) because each unit's MMA chain (8 x N 64) took ~0.5 us and the MMA warp
  serialized on it: a chain of small tcgen05 MMAs costs ~100+ clk an MMA whatever N.
- 10-slot ring: no change (the loads were not the limit). Groups of four units as one N 256 chain: MMA no longer the limit, the drain was:
  32-byte stores from the TMEM row layout touch 32 lines a warp instruction (~32 GB/s an SM).
- drain into the slot + TMA store (in place over v): store read-back of 64 KB ~1.5 us (also with the stores issued from four lanes: the
  SM's TMA / L2 path, not the issue rate); kernel 22.6-23.2 us, graph 40.8 us.
- PDL behind pwa_dual: graph 38.9-40.7; o halves on their own barriers: 38.8-38.9 us.

| L128, 64 / 8 x 32 / 128 | before (v20) | after (v21) |
|---|---|---|
| depth 128 graph | 32.6 | 28.5 |
| depth 256 | 34.8 | 34.6 |
| depth 384 | 42.8 | 38.9 |
| depth 1024 | 73.5 | 71.4 |

measure.speed (every L128 eval row, cuda + triton + pytorch, `_r3/out/len/h1{off,on}_*`): r49 L128 depth 384 42.8 -> 38.8 us (0.709 ->
0.642 of the H100); 0 cases above 0.667, ladder 0 violations (worst cuda / triton 0.574).

## 2026-11-19 (agent C, n13-fp8-c): opt-in e4m3 products on B200 (`FOLDJET_ENGINE_OPM_FP8=bound`, `FOLDJET_ENGINE_PWA_FP8=1`, default OFF)
OPM: the grouped product (60-77 % of the call from depth 1024, tensor bound) on e4m3, the narrow prologue writing e4m3 A2 / BT itself: module 0.63-0.82 of the bf16 route at depth >= 1024 (0.62 at L768 S2048). PWA: the key contraction on e4m3 with v written as e4m3 by `ln_vg_f8`
(8 x 32, N^2 S >= 5e8): 0.88 at N 768 .. 0.69 at N 2048 (depth 1024); other builds from N 1280-1792. Default routes bit-identical (174 / 174). Details, tables, rejected placements: `docs/b200/fp8.md`.
