# Token DiT family on B200 (sm_100a)

Scope: `AttentionPairBias` (`modules/attention.py`), the token DiT blocks `TokenDiTProtenix`, `TokenDiTMiniWorld`,
`TokenDiTMiniWorldBiasOnly`, `TokenDiTBoltz2`, `TokenDiTRoseTTAFold3`, `TokenDiTChai1` (`modules/token_dit.py`,
`modules/dit_variant_cuda.py`) and `TokenPairInit` (`modules/token_pair_init.py`) on the NVIDIA B200 (capability 10.0, card key
`b200_sm100`), engine 2.5.12 + branches `b200-token-dit`, `b200opt-token-dit`, `b200r3-token-dit`. The H100 pages: [../h100/token_dit.md](../h100/token_dit.md),
[../h100/attention_pair_bias.md](../h100/attention_pair_bias.md), [../h100/bias_only_dit.md](../h100/bias_only_dit.md),
[../h100/dit_variants.md](../h100/dit_variants.md). Accuracy and speed are judged as in [../measure.md](../measure.md).

The 2.5.0 redesign kept the B200 (sm_100a) code paths inside the shared H100 / B200 integrations but refused capability (10, 0)
at the class gates. This branch opens those gates where the B200 path measures accurate, adds the pieces the B200 path lacked
(the fp32 APB inference core binding, the wide pair / 768 conditioning on the B200 token runner, the model options on B200, the
TokenPairInit sm_100a kernels from `origin/b200/ladder-final`), and refuses what is still missing. Every route is behind a
`cap == (10, 0)` test; the H100 / A100 / RTX 5090 paths and builds are unchanged.

## Routes

| class | eval (inference) | train |
|---|---|---|
| `AttentionPairBias` | bf16 / master: `kernels/integrations/attention_pair_bias_b200.py` sm_100a path (`apb_rows.cu` sm_100 build, `ApbInferenceCore` = `attn_inf -DQPAIR`), one step per batch row, L padded to 128; fp32: the same packs and fp32 pair pass (`pair_bias32_fwd`, keys un-permuted) on the B200 TF32 gated core (`attn_inf_tf32` built `-DNHEAD -DDHP` per head layout: `_b200_tf32_core`; k and v^T gathered into the pair pass's key order); fp32 at L <= 128 refused (slower than Triton) | bf16 / master: the sm_100a training Function (`attn_fwd2` / `attn_dkv` / `attn_dqb -DQPAIR -DNHEAD`, `gate_rows`, `pair_bias_bwd`); fp32: `forward32_b200` / `backward32_b200` on the sm_100a TF32 cores (`attn_fwd_tf32` / `attn_dkv_tf32` / `attn_dqb_tf32` built `-DNHEAD -DDHP -DRSQDV` per head layout, a zero second sample since the cores pair samples, q / k / v / dO rounded to TF32 to nearest first: without it 2 RED) |
| `TokenDiTProtenix`, `TokenDiTMiniWorld` | `kernels/integrations/token_dit.py` -> `FusedTokenDiTComposed` (CUDA rows, B200 `GatedInferenceCore` bf16 / TF32, `gemm_swiglu` sm_100a bf16); new on B200: pair 256 / 512 (`_hoist`: CUDA `layernorm_rows` + one GEMM) and cond 768 (`_cond`: `cond_rows`) | `kernels/integrations/token_dit_train.py` B200 path (sm_100a `attn_fwd2` / `attn_dkv` / `attn_dqb`, TF32 twins, `gemm_swiglu2 -DSAVE_AB`); pair 256 / 512 and cond 768 opened |
| `TokenDiTBoltz2`, `TokenDiTRoseTTAFold3`, `TokenDiTChai1` | the model options on B200: `conditioned_transition.cuda.token_inference` (the H100 composition: `token_pack_vector`, CondFold rows, option rows) with the B200 pieces -- `GatedInferenceCore` bf16 / TF32 (`_b200_core`), the fp32 bias in natural key order, cuBLAS + `swiglu_rows` for the expand | the training block with the options' row bits (`OPT_IDENT` / `OPT_PAR` / `OPT_UP`); refused: RoseTTAFold3 (flat q / k LayerNorm), Chai-1 fp32 (identity AdaLN) |
| `TokenDiTMiniWorldBiasOnly` | `kernels/integrations/bias_only_dit.py` B200 three-kernel step (`bo_front_*`, `pv_gate_*`, `bo_tail*`); cond 768 / pair 256 opened | `bias_only_dit_train.py` B200 path; cond 768 / pair 256 fp32 only |
| `TokenPairInit` | `kernels/token_pair_init/cuda/sm100`: ONE launch (`tpi_fwd_w`: the fp32 weights bulk-copied (cp.async.bulk) into shared memory and turned into merged residue x token rows and combined chain / entity / bond rows there -- two table rows per pair; the ids (int32 / int64) and bf16 / fp32 `left` / `right` read as they are -- no prep or cast launches) | fp32 weights (the requirement cases): `tpi_fwd_w` and a two-launch backward on the same tensors -- `tpi_bwd_w` (streams of four warps with a private class table each, two table rows per pair, row / column partial sums; no atomics) + `tpi_bwd_red` (the partials added in a fixed order; `dleft` / `dright` written in the inputs' dtypes): bitwise repeatable, nothing zero-filled. Other weights / ids: ids and table by torch ops, `tpi_fwd` + `tpi_bwd` / `tpi_bwd_red` (the same backward on the packed ids). Round 3: [bias-only.md](bias-only.md) |

`backend_versions["cuda:b200_sm100"] = 1` on `AttentionPairBias`, `TokenDiTProtenix` (inherited by the four subclasses),
`TokenDiTMiniWorldBiasOnly` and `TokenPairInit`. (The table is the first wiring round. Since round 3 the token DiT blocks of every class run `conditioned_transition.cuda.token_inference` with the B200 cores at L a multiple of 128 -- the composed runner serves the other lengths of the default options -- and the training block runs the native row kernels: see Round 3; `cuda:b200_sm100` is 3 there.)

## Accuracy (B200, 2026-11-16)

Every requirement row of the class, every length, both dtypes and modes (the plan configs `plan0/cfg/accuracy.<Class>.yaml`,
backends `cuda` + `pytorch`), fp64 gold, fixed tolerances bf16 / master 8e-3, float32 1e-5. Records:
`$W/b2511/agents/token-dit/_td/out/acc_full<Class>.jsonl`; APB fp32 training re-measured after the TF32 rounding fix in
`acc_apbt.jsonl` (rows 41-47, 146-148, 222-226, L 128 / 384 / 640: 27 YELLOW, 0 RED).

| class | GREEN | YELLOW | RED | unsupported | skipped |
|---|---|---|---|---|---|
| AttentionPairBias | 94 | 170 | 2 -> 0 (fp32 train, fixed) | 0 (then fp32 eval L 128 refused: speed) | 2 (L 3968: fp64 reference OOM) |
| TokenDiTProtenix | 140 | 124 | 0 | 4 | 0 |
| TokenDiTMiniWorld | 13 | 11 | 0 | 0 | 0 |
| TokenDiTMiniWorldBiasOnly | 32 | 46 | 0 | 18 | 0 |
| TokenDiTBoltz2 | 27 | 21 | 0 | 0 | 0 |
| TokenDiTRoseTTAFold3 | 29 | 7 | 0 | 12 | 0 |
| TokenDiTChai1 | 79 | 23 | 0 | 12 | 0 |
| TokenPairInit | 12 | 0 | 0 | 0 | 0 |

YELLOW: almost all float32 (TF32 tensor cores against the TF32 PyTorch reference, 1.1-4x its error) and part of the master
training gradients (1.0-1.6x). The H100 / A100 tokens of these classes are YELLOW on the same cells (H100 TokenDiTProtenix float32
eval 54 of 82, AttentionPairBias float32 all); they are served, as on those cards.

## Speed (B200, graph timing, 2026-11-16)

`measure.speed`, graph timing, L2 flushed, 30 iterations, `dev.sh -x` (GPU 1 alone). cuda speedup over triton (medians; over
pytorch in brackets where it differs). Records: `$W/b2511/agents/token-dit/_td/out/spd_{tpi,apb,apb32,tdp,tdm,tdv,tdt}.jsonl`.

| class (rows) | dtype | mode | L 128 | L 384 | L 768 |
|---|---|---|---|---|---|
| AttentionPairBias (41, 44, 46, 47, 146) | master | eval | 2.0-2.3x | 5.3-6.6x | 10.5-12.6x |
| AttentionPairBias (47, 146, 222) | master | train | 2.5-3.2x | 4.6-5.6x | 6.9-9.4x |
| AttentionPairBias (41, 44, 46, 47, 146) | float32 | eval | 0.86-0.96x (refused) | 1.6-1.9x | 2.1-3.2x |
| AttentionPairBias (47, 146, 222) | float32 | train | 1.3-1.4x | 1.5-1.7x | 1.9-2.5x |
| TokenDiTProtenix (99-102, 143, 179) | master | eval | 1.7-1.8x | 2.6-3.4x | 3.9-5.0x |
| TokenDiTProtenix (99-102, 143, 179) | float32 | eval | 1.2-1.4x | 1.7-2.0x | 2.0-2.7x |
| TokenDiTProtenix (101, 143, 260, 286) | master | train | 1.6-1.8x | 1.9-2.3x | 2.7-3.0x |
| TokenDiTProtenix (101, 143, 260, 286) | float32 | train | 1.5-1.7x | 1.4-2.1x | 1.06-2.7x (r101 pair 512: 1.06x, pytorch 1.03x) |
| TokenDiTMiniWorld (103) | master / float32 | eval | 2.0x / 1.4x | 3.1x / 2.0x | 4.6x / 2.7x |
| TokenDiTMiniWorld (103, S 48) | master / float32 | train | 2.5x / 2.1x | 3.7x / 2.5x | 4.3x / 2.6x |
| TokenDiTMiniWorldBiasOnly (104, 105) | master | eval | 1.08-1.19x | 1.08-1.15x | 1.09-1.18x (3.3x) |
| TokenDiTMiniWorldBiasOnly (104, 105) | float32 | eval | 1.2-1.4x (pytorch 0.84-0.86x) | 1.35-1.55x | 1.35-1.53x |
| TokenDiTMiniWorldBiasOnly (104, 105) | both | train | 2.3-2.7x | 2.3-2.6x | 2.0-2.4x |
| TokenDiTBoltz2 (108 / 263) | master | eval / train | 3.0x / 1.6x | 4.1x / 1.9x | 6.2x / 2.7x |
| TokenDiTBoltz2 (108 / 263) | float32 | eval / train | 1.8x / 1.5x | 2.2x / 1.6x | 2.7x / 2.0x |
| TokenDiTRoseTTAFold3 (109) | master / float32 | eval | 3.1x / 1.8x | 4.6x / 2.3x | 6.9x / 2.8x |
| TokenDiTChai1 (110 / 265) | master | eval / train | 2.5x / 1.5x | 3.9x / 2.0x | 5.0x / 2.8x |
| TokenDiTChai1 (110) | float32 | eval | 1.5x | 1.9x | 2.3x |
| TokenPairInit (159) | float32 / master | eval | 1.37x / 1.16x (9.2x / 6.6x) | 1.8x / 1.6x | -- |
| TokenPairInit (159) | float32 / master | train | 3.2x / 2.9x | 4.4x / 4.2x | -- |

Exceptions to "cuda > triton > pytorch": bias-only fp32 inference at L 128 (pytorch 0.178 ms < cuda 0.207 < triton 0.246) and
TokenDiTProtenix r101 fp32 training at L 768 (cuda 3.31 / pytorch 3.41 / triton 3.51 ms). Anthropic / cueq baselines are in
`$W/b2511/out_base` (not re-measured here).

## Optimization round (2026-11-17, branch `b200opt-token-dit`)

Start: the B200 ladder of 2026-11-17 (`$W/b2511/ladder/ladder.txt`, triton pretuned for the B200). Measured after each change on GPU 1
(`dev.sh -x`, graph timing, L2 flushed, 30 iterations); records `$W/b2511/agents/opt-token-dit/_tdo/out/`. `cuda:b200_sm100` is 2 on
`AttentionPairBias`, `TokenDiTProtenix` (and its four subclasses) and `TokenDiTMiniWorldBiasOnly`.

| change | effect |
|---|---|
| bias-only inference: the runner's weight pack in ONE launch per block (`bias_only_dit/cuda/bo_pack_sm100.cu`, `pack_sm100.py`): casts, folded LN weights, the AdaLN table weights in both orders, the tail kernels' pair packs (z order), TF32 rounding -- bit for bit the torch pack (checked on rows 104-107, both dtypes, fp32 and bf16 parameters) | a captured step re-ran ~40 torch kernels per replay (~100 of 248 us at r104 master L384). Rows 104-107 eval: master 0.69-0.86x triton -> 1.10-1.33x; float32 1.03-1.58x from L 256 (19 REAL ladder violations -> 0) |
| bias-only bf16 training at cond 768 / pair 256 (rows 105-107 master): the per-step backward launches (the fused sm_100 backward is built for cond 384 / pair 128) | served: 1.40-1.64x triton, YELLOW / GREEN (refused before) |
| token DiT training, RoseTTAFold3 (row 264): the flat q / k LayerNorm makes its q / k copies on B200 (they were raw column views of qkvg: no LayerNorm at all) and leaves q unscaled for the B200 cores | 12 RED -> 12 YELLOW (err / tol master 1.2-1.5, float32 2.6-3.8; H100 cuda tokens 1.05-1.8 / 2.3-3.7); 1.52-2.71x triton |
| token DiT training, Chai-1 fp32 (rows 265 / 266): the fp32 cond is the identity AdaLN's raw copy | served: 1.36-2.38x triton, GREEN / YELLOW |
| token DiT inference, FoldForge R18 `token_folds*` (rows 163 / 187, B = 5 trunk_seeds): one B = 1 step per fold on the sm_100a runner, the runner's pair-bias cache multi-entry (64 MB) | served: 1.15x (L 128) / 1.85-1.89x (L 256) triton, GREEN |
| AttentionPairBias fp32 inference: the TF32 core built `-DKPERM=1` (reads the pair pass's key-permuted bias, k / v in natural order: no gathers), the row-wise stages (LN rows, q / k / g and v^T GEMMs, to_out) batched over B, the inference packs in one launch (`bo_pack_sm100.cu` modes 3 / 4, bit for bit) | 0.086 -> ~0.06 ms at r41 L256 (anthropic 0.090); L 128 served (1.12-1.76x triton, was refused); 1.16-3.8x triton overall, ahead of anthropic at every fp32 length measured (r41 L256 0.061 vs 0.090, L384 0.115 vs 0.141) |
| tuning: `_launch_config.untracked` -- the tuner's resets / restores no longer bump autograd version counters | the B200 pretune of bias-only triton training (rows 104-106) no longer raises "modified by an inplace operation" (cherry-picked as 11c7c72c5) |

Branch features (`origin/b200/ladder-final`) not adopted this round: the APB key-split / query-split cores (`-DKSPLIT` / `-DQSPLIT`),
per-sample bias planes (`-DBPS`) and the C++ launch sequences. The master APB rows already run 2-12x triton and ahead of anthropic /
cueq here; the C++ host path removes host time, which graph timing does not count. The AdaLN cta_group::2 / split-K / folded LN(cond)
kernels belong to the AdaLN module family (the token DiT runs its own row kernels and cond tables).

## Round 3 (2026-11-17, branch `b200r3-token-dit`): B200 `cuda` against the H100 `cuda`

Target: on every requirement case both cards serve, B200 median <= H100 median / 1.5 (graph timing, L2 flushed). Start (the tokens of
2026-11-17): TokenDiTProtenix 217 of 268 cases above 0.667 (113 slower than the H100, worst 1.83), TokenDiTChai1 63, TokenDiTRoseTTAFold3 40,
TokenDiTBoltz2 37, TokenDiTMiniWorld 13. `cuda:b200_sm100` is 3 and `triton:b200_sm100` 2 on `TokenDiTProtenix` and its four subclasses.
Measured with the measurer's graph timing (`measure.speed.measure`: warmup 5, 30 iterations, L2 flushed) of the `cuda` backend alone
(`_r3/qspd.py`; the H100 side is its tokens), `dev.sh -x`; records `$W/b2511/agents/r3-token-dit/_r3/out/f{a,b,c,d,e}.fm.jsonl`,
the comparison `_r3/out/fin.fm.txt` (the merged tree, base be0fbe7df); per-kernel profiles (torch profiler under graph replay) `_r3/out/p{a,b}*.txt`.

### What changed

Inference (`conditioned_transition.cuda.token_inference`, `kernels/integrations/token_dit.py`):

| change | effect (B200) |
|---|---|
| every block at L a multiple of 128 runs `token_inference` with the B200 cores -- the default options too (they ran the composed runner: its own row passes, two conditioning GEMMs) | the H100 composition: the folded conditioning (one GEMM), PDL row passes |
| bf16 default options: the weight pack in ONE launch (`bias_only_dit/cuda/bo_pack_sm100.cu` through `_b200_pack`) + one GEMV; it replaced `token_pack_vector`, the [Wa; Wb] cast / cat and the fold operands (~9 kernels a replay) | r144 master L128: pack ~35 -> 17 us |
| one sample, bf16: `attn_inf -DQPAIR=1` (a work item pairs two 128-query tiles of a head; the sample-pair schedule computed a dropped partner) | r144 master L768: core 34 -> 14.5 us, step 320 -> 125 us |
| FoldForge R18 folds (B trunk_seeds), L a multiple of 128: ONE `token_inference` call (GEMMs and row passes over every fold's rows, one core launch per fold with its bias planes) instead of one step per fold | r163 L128 414 -> 176 us, L256 545 -> 262 us |
| fp32: the pair bias in one pass instead of LN rows + GEMM + mask fill -- from L 256 AttentionPairBias's TMA-fed TF32 pass `pair_bias32_tc` (natural key order; merged from `b200r3-apb`: 86 against 131 us at L 768, 286 against 485 at L 1536), below it the training rows' `pair_bias`; masked keys -1e30, as every inference bias. The fp32 training forward takes the same pass with the row stats | r99 fp32 L2048 2131 -> 1262 us |
| bf16 wide pair (256 / 512 at 16 heads): `apb.pair_bias` (one bulk-loaded pass) instead of LN rows + GEMM + mask fill | r100 master L768 391 -> 234 us, L2048 1709 -> 868 us; Chai-1 master L2048 1680 -> 836 us |
| model options (Boltz-2 / RoseTTAFold3 / Chai-1), bf16: the folded conditioning's operands made once with the static weight pack (a replay re-ran a cat, a row sum and two casts) | r108 master L128 86 -> 65 us |
| bf16 from 3840 rows: the sm_100a expand GEMM with the SwiGLU epilogue (`gemm_swiglu.GemmSwiglu`, as the composed runner had it) | ~4 us at S 5, L768 |

Training (`kernels/integrations/token_dit_train.py`):

| change | effect (B200) |
|---|---|
| the native row kernels of `token_dit_train_rows.cu` on B200 too: weight pack (`train_pack90`), conditioning prep / backward, final gradients (`train_final90`); on sm_100 the pack and the final gradients take 8 vectors a thread (`#if __CUDA_ARCH__ >= 1000`, the host asks the device: 18.1 -> 12.5 and 18.7 -> 13.0 us). They replaced the torch pack and the per-leaf gradient copies (~60 small kernels) | r144 master L128 403 -> 211 us with the two rows below |
| no padded sample at A = 1: bf16 the `attn_fwd2 -DQPAIR=1` forward; fp32 the TF32 cores' own odd A (the launch grid was A // 2, it is (A + 1) // 2); a sample-only pad no longer copies the pair (its backward copied the pair gradient: 2 x 43 us at L512 fp32) | every GEMM and row pass over L rows instead of 2 L |
| bf16: the pair bias in one pass forward (`apb` `pair_bias_fwd_stats`: bulk-loaded tiles, with the row stats: 57 -> 19 us at L512) and one pass backward (`pair_bias_bwd`: dpair + the dWf partials) instead of LN rows + GEMM + mask fill and two GEMMs + the LN backward rows | r144 master L512 414 -> 342 us |
| the model options through the native pack (`_pack_options` on top); Boltz-2 through the native final gradients (the up-gate leaf's gradient is the third block of dWab) | r263 master L128 427 -> 233 us; RoseTTAFold3 422 -> 238; Chai-1 381 -> 240 |
| fp32 weight-gradient GEMMs unsplit below 4096 rows (the batched GEMM + its sum cost more than the one GEMM); the per-block partial sums as one gemv (`_colsum`) instead of a strided `sum(0)` | r144 fp32 L512: ~60 + 16 us of 690 |

Triton (`triton:b200_sm100` 2): the Triton attention core hands dbias back as the head-major view on the B200 for the token DiT blocks
(`TritonAugmentedAttentionHeadMajor`: no sum over A = 1, no transposing copy, `to_bias`'s backward on the head-major operand), as the
A100 / H100 always had it. The default (`TritonAugmentedAttentionFunction`, what `AttentionPairBias` calls) is unchanged on every card.

### Rejected (measured, reverted or not built)

| idea | why not |
|---|---|
| 8 vectors a thread in `bo_pack_sm100` (`-DUNR=8`) and in `token_pack_vector` | no gain: 17.2 -> 17.0-17.4 us and 12.3 -> 12.7-13.0 us (the bf16 pack reads 33 MB of fp32 leaves from HBM after the L2 flush and writes 16.5 MB); reverted |
| a warp's four pair rows loaded up front in `pair_bias` / `pair_bias_bwd` (sm_100) | within noise: fp32 L768 123-125 -> 115-128 us, bwd L512 84.7 -> 82.8-83.7; reverted |
| AttentionPairBias's fp32 one-pass pair bias (`apb` `pair_bias32_fwd`, key-permuted, with the TF32 core built `-DKPERM=1`) for fp32 inference | slower than the training rows' `pair_bias` on this card: 14.3 / 53.2 / 180.2 / 680.9 us against 12.3 / 38.9 / 120.8 / 440.5 at L 128 / 384 / 768 / 1536 (pair 128, 16 heads; pair 256: 348 against 195 at L 768) |
| a static (per weight version) pack for the default-option blocks, as the model options have | (taken in part 2 for `token_inference`, which never trains: see "Round 3, part 2") |
| the query-tile-pair schedule (`-DQPAIR`) for the TF32 cores | `attn_inf_tf32` / `attn_fwd_tf32` at 16 x 48 hold 3 stages in 217 KB of shared memory; two bias tiles a stage do not fit without dropping a stage. One-sample fp32 calls still compute a dropped partner sample |
| the folded conditioning in fp32 | the three `*_cf` row kernels are bf16-typed; fp32 keeps `cond_rows` + two conditioning GEMMs (~8 us at L128) |
| one core launch for all R18 folds | `attn_inf` pairs samples that share one bias plane; a per-fold plane needs a second bias tile per stage (as -DQPAIR) |

### Result against the H100 tokens

192 of the classes' cases re-measured (rows 99-101, 103, 108-110, 143-145, 163, 179, 187, 260-265, 286; L 128 / 256 / 384 / 768, and 2048
for the inference rows that have it; both dtypes); ratio = B200 ms / H100 ms, target <= 0.667. "before" = the same cases in the
B200 tokens of 2026-11-17.

| class | dtype mode | cases | before: > 0.667 / > 1.0 / worst / geomean | after: > 0.667 / > 1.0 / worst / geomean |
|---|---|---|---|---|
| TokenDiTProtenix | master eval | 35 | 32 / 28 / 1.83 / 1.145 | 18 / 0 / 0.783 / 0.618 |
| TokenDiTProtenix | float32 eval | 31 | 28 / 17 / 1.37 / 0.939 | 13 / 0 / 0.812 / 0.615 |
| TokenDiTProtenix | master train | 28 | 25 / 21 / 1.52 / 1.128 | 16 / 0 / 0.805 / 0.677 |
| TokenDiTProtenix | float32 train | 28 | 19 / 4 / 1.03 / 0.694 | 5 / 0 / 0.740 / 0.486 |
| TokenDiTMiniWorld | master eval / train | 4 / 4 | 4 / 3 / 1.20 / 1.082; 1 / 0 / 0.75 / 0.590 | 2 / 0 / 0.727 / 0.671; 0 / 0 / 0.616 / 0.545 |
| TokenDiTMiniWorld | float32 eval / train | 4 / 4 | 4 / 1 / 1.02 / 0.865; 0 / 0 / 0.42 / 0.405 | 1 / 0 / 0.740 / 0.625; 0 / 0 / 0.403 / 0.393 |
| TokenDiTBoltz2 | master eval / train | 5 / 4 | 2 / 0 / 0.78 / 0.674; 4 / 4 / 1.16 / 1.142 | 0 / 0 / 0.596 / 0.577; 0 / 0 / 0.631 / 0.616 |
| TokenDiTBoltz2 | float32 eval / train | 5 / 4 | 4 / 0 / 0.90 / 0.760; 3 / 0 / 0.88 / 0.760 | 2 / 0 / 0.786 / 0.631; 0 / 0 / 0.641 / 0.502 |
| TokenDiTRoseTTAFold3 | master eval / train | 5 / 4 | 3 / 0 / 0.84 / 0.712; 4 / 4 / 1.28 / 1.227 | 1 / 0 / 0.685 / 0.611; 3 / 0 / 0.724 / 0.680 |
| TokenDiTRoseTTAFold3 | float32 eval / train | 5 / 4 | 5 / 0 / 0.95 / 0.788; 3 / 0 / 0.94 / 0.792 | 1 / 0 / 0.802 / 0.640; 1 / 0 / 0.715 / 0.540 |
| TokenDiTChai1 | master eval / train | 5 / 4 | 2 / 0 / 0.87 / 0.687; 4 / 4 / 1.15 / 1.137 | 0 / 0 / 0.604 / 0.442; 2 / 0 / 0.712 / 0.666 |
| TokenDiTChai1 | float32 eval / train | 5 / 4 | 3 / 0 / 0.79 / 0.700; 3 / 0 / 0.92 / 0.730 | 1 / 0 / 0.758 / 0.560; 1 / 0 / 0.693 / 0.486 |
| all | | 192 | 153 / 86 / 1.83 | 67 / 0 / 0.812 |


No measured case is slower than the H100 any more; 67 of 192 are still short of 1.5x. Where they are:

| cases still above 0.667 | ratio | why |
|---|---|---|
| inference, L 128 (default options, both dtypes; the model options in fp32) and L 256 master (default options) | 0.68-0.82 | a step is ~12 kernels of 3-10 us; the bf16 weight pack is 17 us of an 82 us step on its own (33 MB of fp32 leaves read from HBM after the L2 flush, 16.5 MB written: bandwidth at this size, the same with 1 or 8 vectors a thread); the H100 step has the same kernel count |
| one-sample inference (rows 143-145), fp32 at every L, master at L <= 384 (24 x 32, 12 x 64) | 0.67-0.86 | the TF32 core has no query-tile-pair schedule (a dropped partner sample is computed); otherwise as L 128 |
| R18 folds (rows 163 / 187) | 0.72-0.79 | one core launch per fold (5), the pair pass per fold |
| master training at one sample, L <= 384 (pair 256 / 512 and Chai-1 up to L 768) | 0.67-0.81 | ~75 launches a step (22 cuBLAS GEMMs of 3-8 us); wide pairs and Chai-1: `pair_bias_bwd` (per-row TF32 `mma.sync` tiles, 3.3 rows / ns) is 80 of 390 us at L 512 already at pair 128, and Chai-1 keeps the torch final gradients (identity AdaLN) |
| fp32 training at L 128 | 0.70-0.74 | launch-bound as above |

### Accuracy, determinism, ladder

Accuracy (cuda + pytorch against the fp64 gold; `_r3/out/acc{t2,x,w,e}.jsonl`: rows 99-103, 108-110, 112, 144, 260-265, L 128 / 384, both
dtypes, eval and train, 64 cuda cases): 0 RED; the lights are those of the B200 tokens except r101 master train L128 (YELLOW at 1.06 x
tol where the token was GREEN at 0.96; the H100 cuda token of the case is YELLOW at 1.15). The bf16 training gradients moved within
their YELLOW band (the pair bias is now one TF32 / bf16 pass per direction). The Triton backend's values are bit for bit its tokens'
(`_r3/out/acctri.jsonl`, 20 cases).

Determinism (`_r3/det.py`, the same module and inputs twice): inference is bitwise repeatable on every case checked (rows 99, 103,
108, 110, 144, 163, both dtypes, L 128 / 384: 21 cases). Training is not, and was not before this round (the round-2 tree: 23 of 24
training cases differ run to run; this tree: 18 of 24 -- bf16 at one key chunk, L 128, is repeatable now): the bf16 `attn_dqb` adds its
dQ partials with TMA reduce-adds in arrival order, the TF32 `attn_dqb_tf32` adds its sample groups' dbias the same way. See Gaps.

Triton against PyTorch, float32 training (`measure.speed`, `_r3/out/sptri.jsonl`), the cases of the ladder's `triton<pytorch` violations:

| case | pytorch ms | triton ms | triton / pytorch (before) |
|---|---|---|---|
| TokenDiTProtenix r260 L384 / 512 / 768 | 0.933 / 1.256 / 2.192 | 0.926 / 1.234 / 2.168 | 0.992 / 0.982 / 0.989 (L512: 1.049) |
| TokenDiTRoseTTAFold3 r264 L384 / 512 / 768 | 0.935 / 1.248 / 2.149 | 0.946 / 1.253 / 2.184 | 1.012 / 1.004 / 1.016 (1.029 / 1.026 / 1.039) |
| TokenDiTChai1 r265 L384 / 512 / 768 | 0.942 / 1.332 / 2.440 | 0.955 / 1.342 / 2.475 | 1.014 / 1.008 / 1.014 (1.037 / 1.026 / 1.030) |
| TokenDiTChai1 r266 L384 / 512 / 768 | 0.942 / 1.333 / 2.439 | 0.955 / 1.342 / 2.475 | 1.014 / 1.007 / 1.015 (1.041 / 1.025 / 1.028) |

All inside the ladder's 2 % band now (Protenix ahead of PyTorch); RoseTTAFold3 and Chai-1 stay 0.4-1.6 % behind: what is left is the
Triton core itself against PyTorch's TF32 cuBLAS attention at one sample (L 640 not re-measured).

### Round 3, part 2 (2026-11-17): the shared source restored, deterministic training, side streams -- `cuda:b200_sm100` 4

**The shared row source.** `token_dit_train_rows.cu` is byte-identical to origin/main (7c544cf2f) again. The B200 build is a translation
unit of its own, `token_dit_train_rows_sm100.cu` (`train.rows_ext_sm100`, `gencodes("100a")` only, extension names
`token_dit_train_rows_sm100*`): it includes the shared file with its `PYBIND11_MODULE` renamed into a function, then replaces four
bindings with B200 kernels -- `pair_bias` (the masked keys' value as an argument), `train_pack90` / `train_final90` (8 vectors a thread,
loads first) and `unfold_lnw` (the cond-LN weight gradients as per-block partial rows added in block order; the shared kernel adds them with
float atomics). The other cards' prebuild plans (`_prebuild --arch sm90a / sm80 / sm120 --list`, artifact names and keys) equal origin/main's
line for line: 513 / 72 / 64 artifacts (`_r3/out/plan/`); `--arch sm100 --list` has 464 entries and holds every build a run compiled.

**Deterministic training.** The B200 training cores now run their deterministic builds: bf16 `attn_dkv -DDET=1` (dK / dV in block order)
and `attn_dqb -DDQPART=1 -DDQDET=1` (each 128-key chunk's dQ stored to its own slice, the slices added in chunk order; `DQDET` orders the
two dQ MMA issuers of a step -- without it, the token DiT at 48 samples, L 384 still differed run to run); fp32 `attn_dkv_tf32 -DDET=1` and
`attn_dqb_tf32 -DDQPART=1 -DDET=1 -DDBPART=1` (dQ slices per 64-key chunk, each sample group's dbias its own slice, both added in a fixed
order: `sm100.backward_tf32(det=True)`). With `unfold_lnw` above: **every training case checked is bitwise repeatable** (`_r3/det.py`,
four runs: rows 101, 103, 144, 260, 263, 264, 265 at L 128 / 256 / 384 / 768, both dtypes: 56 of 56; inference rows 99, 103, 108, 110, 144,
163 at L 128 / 384: 21 of 21; before: 18 of 24 training cases differed).

**Side streams in captured steps** (`bias_only_dit.cuda.side`, `MINIWORLD_TOKEN_B200_STREAMS=0` turns them off): training forward -- the
gate GEMM and the pair -> bias pass beside the conditioning / AdaLN / q k v chain, joined before the core; training backward -- the six
weight-gradient GEMMs and the pair backward (bf16 pair 128: AttentionPairBias's one-pass `pair_bias_bwd`, 46 against 80 us at L 512)
beside the chain, joined before the final gradients; fp32 inference -- the pair pass and the v^T GEMM beside the chain. Captured == eager
bit for bit and replays equal on 53 cases (`_r3/capeq.py`). On / off, the same tree, 60 cases (`_r3/out/ss{0,1}.jsonl`): training
2-6 % faster at L 128 - 384, fp32 inference 3-10 % faster; bf16 inference was 1-10 % *slower* at L 128 / 384 with the fork, so bf16
inference does not fork.

**Static inference packs for the default options too.** `token_inference` is inference only (grad off), so on the B200 the default-option
blocks keep their weight pack across captures and replays per weight pointer / version, as the model-option blocks always did (the
round-3 "rejected" entry assumed a training replay; this path never trains). A replay no longer repacks: the bf16 one-launch pack was 17
of 82 us and the fp32 `token_pack_vector` 13 of 104 us at L 128 (e.g. r99 master eval L 128 0.0818 -> 0.063 ms). As before for the model
options: an in-place weight change after a capture needs a new capture.

Accuracy (`_r3/out/acc{x,w,t2,e}.p2.jsonl`, 64 cuda cases): 0 RED, every light as before; the errors are those of the previous records
except r144 master train L 128 (7.6e-3 -> 1.3e-2, GREEN: the bf16 one-pass pair backward).

Speed (the 192 cases of round 3, `_r3/out/fin.p2.txt`; ratio = B200 / H100 cuda):

| class | dtype mode | cases | part 1: > 0.667 / > 1.0 / worst / geomean | part 2: > 0.667 / > 1.0 / worst / geomean |
|---|---|---|---|---|
| TokenDiTProtenix | master eval | 35 | 18 / 0 / 0.783 / 0.618 | 1 / 0 / 0.674 / 0.519 |
| TokenDiTProtenix | float32 eval | 31 | 13 / 0 / 0.812 / 0.615 | 3 / 0 / 0.707 / 0.558 |
| TokenDiTProtenix | master train | 28 | 16 / 0 / 0.805 / 0.677 | 12 / 0 / 0.767 / 0.632 |
| TokenDiTProtenix | float32 train | 28 | 5 / 0 / 0.740 / 0.486 | 4 / 0 / 0.723 / 0.476 |
| TokenDiTMiniWorld | all four | 16 | 3 / 0 / 0.740 | 0 / 0 / 0.621 |
| TokenDiTBoltz2 | all four | 18 | 2 / 0 / 0.786 | 2 / 0 / 0.729 (fp32 eval L 128 / 256) |
| TokenDiTRoseTTAFold3 | all four | 18 | 6 / 0 / 0.802 | 6 / 0 / 0.767 |
| TokenDiTChai1 | all four | 18 | 4 / 0 / 0.758 | 3 / 0 / 0.709 |
| all | | 192 | 67 / 0 / 0.812 / 0.586 | 31 / 0 / 0.767 / 0.544 |

(Part 2's 192-case sweep ran bf16 inference with the fork; the full sweep below has bf16 inference without it.)

**Every requirement case of the five classes** (514 cases, 502 with an H100 cuda token; `_r3/out/full.p3.jsonl` / `full.p3.txt`: the
192 above, the other 322 measured the same way, bf16 inference re-measured after the fork was dropped): **46 of 502 above 0.667, none
slower than the H100, worst 0.767, geomean 0.504.** Per class (> 0.667 / > 1.0 / worst / geomean): Protenix master eval 2 of 88
(0.683), fp32 eval 5 of 84 (0.709), master train 14 of 48 (0.767), fp32 train 5 of 48 (0.723); Chai-1 7 of 114 (0.709);
RoseTTAFold3 9 of 48 (0.767); Boltz-2 4 of 48 (0.744, fp32 eval); MiniWorld 0 of 24. Where the 46 are:

| cases still above 0.667 | ratio | why |
|---|---|---|
| master training at one sample (eight for row 286), L 128 / 256 (Protenix rows 101, 143-145, 260, 261 -- 261 also L 384 --, 286 at L 128; RoseTTAFold3 264; Chai-1 265 / 266 at L 128), 18 cases | 0.67-0.77 | ~75 launches a step on few SMs: 22 cuBLAS GEMMs of 3-15 us, the weight pack (12.5 us) and the final gradients (13 us + 10 us of partial reductions) every step; the side streams overlap ~5 % |
| one-sample fp32 training, L 128 (rows 143-145, 260, 261, 264, 265), 7 cases | 0.67-0.72 | the same, TF32 |
| fp32 inference at L 128 / 256 with five samples (Boltz-2 108 / 183, Chai-1 110 / 111 / 185 / 186, RoseTTAFold3 109 / 184), 10 cases | 0.67-0.77 | ~12 kernels of 3-10 us; fp32 keeps `cond_rows` + two conditioning GEMMs (the folded `*_cf` rows are bf16-typed) |
| one-sample fp32 inference (rows 143 / 144, L 256-640), 5 cases | 0.67-0.71 | the TF32 core has no query-tile-pair schedule (a dropped partner sample is computed): 3 stages in 217 KB of shared memory |
| bf16 inference: R18 folds at L 128 (rows 163 / 187) and RoseTTAFold3 184 / 109 at L 128-512, 6 cases | 0.68-0.70 | the folds: one core launch and one pair pass per fold; RoseTTAFold3: the flat q / k LayerNorm rows on top of the default composition |

Ladder (`_r3/out/tp{,p}.jsonl`): cuda is below triton on every case (`full.p3.txt`: no `LADDER(c>t)`); triton against PyTorch on the
float32 one-sample training rows (260 / 264 / 265 / 266, L 384 / 512 / 640 / 768): 0.981-1.017, all inside the 2 % band, L 640 included.

### Round 3, part 3 (2026-11-17): the final-gradient tail, R18 folds in one launch -- `cuda:b200_sm100` 5

Measured as before (`_r3/qspd.py`, graph timing, L2 flushed, `dev.sh -x`; GPU 1 was shared with other agents' foreign processes part of
the time, so single cases move by ~1-2 %); records `$W/b2511/agents/r3-token-dit/_r3/out/` (`q*.N.jsonl`: A/B pairs, the variant in
`_r3/abq.sh`'s argument order; `full.p5.jsonl` / `full.p5.txt`: every requirement case after the part). Timelines of one captured step
(`_r3/tl.py`: every kernel's start / end and stream) found where the time goes.

| change | effect (B200) |
|---|---|
| training: every weight gradient straight into its leaf (one cuBLAS GEMM per leaf, `torch.mm(..., out=leaf)`, on the side streams) -- no ~57 MB copy of the gradients into the leaves | the step ended in a 25 us tail after the last row pass: the cond-LN partials (6.3), the partial-row reductions (3.9), one final kernel copying every gradient (13.8) |
| training: the rest of the final gradients beside the chain (`token_dit_train_rows_sm100.cu`): `final_lnw100` (the four cond-AdaLN leaves dWn diag(w) and the cond-LN partial rows, the raw leaves read in place; 8 rows a block), `final_reduce100` (the partial-row reductions), `final_rest100` (only the leaves no GEMM wrote: biases, cond-LN weights, the pair weights) -- fixed orders, bitwise repeatable | r144 master train L 128 200.6 -> 181.2 us, fp32 258.0 -> 231.7 |
| training, the option models (Chai-1's identity AdaLN, RoseTTAFold3's flat q / k LayerNorm): the same leaf-direct GEMMs, `final_lnw100` beside the chain (the identity AdaLN: dWn's four blocks straight into the leaves), the small leaves in one multi-tensor copy on side stream 1 (`_rest_b200`) | their ~48 us tail (unfold_lnw, its reduction, the sums, a 25-leaf copy) is gone: RF3 r264 master L 128 233 -> 197 us, fp32 285 -> 240; Chai-1 r265 master 222 -> 203, fp32 270 -> 245 |
| training: the conditioning prep on a side stream, issued before the weight pack (the pack's full grid would hold every SM: issued after it, the prep queued behind it) | r144 master L 128 181.2 -> 176.3 us, fp32 231.7 -> 225.3 |
| inference, R18 folds (B trunk_seeds): ONE core launch for every fold (`attn_inf -DFOLDARG=1`: an extra argument NB, a work item is (fold, head, query tile, sample pair) -- both warpgroups still read one bias plane; planes [H][NB]) and ONE pair pass (`apb.pair_bias_batch`) -- outputs bitwise those of the per-fold launches | r163 master L 128 147.5 -> 116.7 us, L 256 241.6 -> 192.4 (5 core launches of 7.8 us with 48 work items each, 5 pair passes + a stack) |
| fp32 inference inside a capture: the gates' GEMM beside the chain (read after the join before the core) and the pair pass issued after the conditioning GEMMs (issued first, its persistent grid held the SMs while the conditioning rows and GEMMs queued: ~11 us at L 128); from L 256 the v^T GEMM on the chain (the side stream's pair pass outlasts the chain to the core) | Chai-1 r110 L 128 102.3 -> 97.6 us, r185 98.2 -> 92.2; Boltz-2 r108 / r183 L 256 149.2 / 139.1 -> 141.2 / 128.8 |
| one-sample fp32 inference: `attn_inf_tf32 -DQPAIR=1` (two 128-query tiles of a head share their k / v^T blocks; two stages at 48 wide, a stage holds two bias tiles) -- outputs bitwise those of the sample-pair schedule | 0-2 % (r143 L 256 94.1 -> 92.2 us): at one sample the core has 12-24 work items for 148 SMs, so the dropped partner cost latency, not throughput |

Rejected (measured, reverted):

| idea | why not |
|---|---|
| a few-row split-K GEMM (`_r3/rejected/tdt_gemm_sm100.cu`: tcgen05, a cluster of KS CTAs shares a 128 x 64 tile, 1 / KS of the K loop each, the fp32 partials reduced through distributed shared memory in rank order) for the token DiT's 12 chain GEMMs at 128 rows | slower than cuBLAS on every shape (graph chains of 10, L2 flushed, M 128: 5.3-7.2 against 4.1-7.2 us with PDL; M 384: 7-15 against 4.3-7.4): in-kernel stamps (`_r3/gprof.py`) put the first TMA data 1.2-1.4 us after the start and the cluster reduction + barriers at 1.5-2.5 us -- cuBLAS's 4.1 us for a 128-row GEMM is the launch floor (2.05 us) plus one HBM round trip; the lever at this size is fewer launches, not a faster GEMM |
| the weight pack split in two: what the first kernels read on the chain, Wraw / q k v g / Wo / Wab / Wsq on a side stream (grid of one block per SM, or four) | +7 us at r144 master L 128 (181.2 -> 188.3; four blocks per SM 184.5): the late pack on few SMs took 24 us and the chain's kernels still queued behind it |
| the gate in the training core's epilogue (`attn_fwd2 -DOGATE`: og = sigmoid(g) O stored by the softmax warps, the H100 core's gated output) | r144 master L 128 180.5 against 178.0 us without it, r101 202.0 / 200.9: the per-row g loads and og stores in the epilogue cost what the 3 us gate pass did |
| the expand + SwiGLU of fp32 inference on the B200 Transition's TF32 SWIGLU GEMM (`tg_gemm.cu`, [W_a; W_b] on the TF32 grid in the static pack) | equal to cuBLAS + `swiglu_rows` at 640 rows (r108 104.5 / 104.5 us, r110 98.1 / 96.4): the fp32 operands bound the per-SM intake either way |
| the dcg GEMM (dGg Wg) on a side stream as soon as dGg is complete | level (176.3 / 176.3 us) |
| bf16 pair 128 training: the pair backward on AttentionPairBias's tcgen05 `pair_bwd` (`sm100.apb_pair_bwd`, merged from b200r3-apb) instead of its mma.sync one-pass `pair_bias_bwd` | level within +-2 % (`_r3/out/pb.speed`: master train L 128 r143 180.0 -> 178.3 us, r261 184.6 -> 186.4, r263 196.9 -> 197.7; L 512 r144 271.2 -> 276.5, r263 301.1 -> 299.2): on the token DiT's side stream the pair backward is off the critical path; and the dpair / dWf error larger (r144 master L 128 1.30e-2 against 7.8e-3, still GREEN) -- not taken |
| RoseTTAFold3 bf16 inference: the flat q / k LayerNorm one warp per (row, q \| k) with PDL (`_r3/rejected/qk_ln_sm100.cu`, `qkln_warp.diff`: no block barriers, the affine read before the wait) instead of the shared block-per-row `qk_ln_rows` | the pass takes 4.0 us either way at 1280 rows (timeline, r109 / r184 L 256 S 5): a ~4 MB in-place pass right behind its GEMM sits at the short-kernel floor, like `swiglu_rows` (3.8 us); A / B over every length level within +-1 % (L 256 77.6 / 77.7 against 77.7 / 77.7 us) -- reverted. Only removing the pass (the statistics in a neighbouring kernel) would gain |

Also: bf16 inference forks too now (the pair pass on a side stream, issued after the folded conditioning GEMM; part 2 had measured the
fork 1-10 % slower with the pair pass issued first, where its grid held the SMs the chain needed): r99 master L 512 139.3 -> 131.0 us,
r184 L 512 127.0 -> 120.8, r144 L 128 54.4 -> 51.3; fp32 one sample: v^T on the chain (r144 L 256 98.0 -> 88.0 us, r143 L 128 77.9 ->
71.7; five samples keep it beside the chain, 4-6 % slower there).

**Every requirement case of the five classes** (`full.p5.jsonl` / `full.p5.txt`, measured one length per run -- the B200 runs a short
case slow right after a long one, its clock down under the power cap): **20 of 502 above 0.667 (part 2: 46), none slower than the H100,
worst 0.743 (0.767)**; `cuda < triton` on every case. Per class (> 0.667, part 2 -> part 3): Protenix 26 -> 10, RoseTTAFold3 9 -> 5,
Chai-1 7 -> 2, Boltz-2 4 -> 3, MiniWorld 0 -> 0.

| cases still above 0.667 | ratio | why (measured) |
|---|---|---|
| fp32 inference, five samples, L 128 (Boltz-2 108 / 183, Chai-1 110 / 111, RoseTTAFold3 109 / 184) and Boltz-2 183 L 256 | 0.68-0.74 | the TF32 GEMMs at 640 rows: expand 14.7 us, cond 10.4, q k g 11.0, squeeze 8.8 of a ~100 us step (r108 timeline) -- fp32 operands double each SM's intake, and a tcgen05 TF32 GEMM (the Transition's `tg_gemm`) measured the same as cuBLAS there; the H100 step has the same composition |
| one-sample master training L 128 (rows 101, 143, 144, 260, 261, 286) and r101 L 256 | 0.68-0.71 | ~70 launches of 2-8 us: the chain after the final-gradient change is forward 75 us (pack 12.5, 6 GEMMs, core, 6 row passes) + backward ~100 us (6 dX GEMMs, the cores, 6 row passes); r101 (pair 512) adds the rows' pair backward (22 us on its side stream) |
| one-sample fp32 inference, r144 (12 x 64) L 256 / 512 / 640 | 0.67-0.71 | the TF32 core at 12-24 work items for 148 SMs (17 us at L 256): the query-tile pairs did not shorten it (latency per item, not throughput); a key split (`-DKSPLIT` for the TF32 core + a combine pass) would |
| RoseTTAFold3 bf16 inference 109 / 184 at L 256 / 512 | 0.667-0.70 | the flat q / k LayerNorm pass on top of the default composition (3 us at L 128) |

Accuracy (`_r3/out/acc{5,9,t3}.p5.jsonl`, 44 cuda cases: rows 101 / 143 / 144 / 163 / 187 / 263-265 / 286 and the fp32 five-sample rows,
eval + train): 0 RED, no light worse than the B200 token except three master training cases now YELLOW (r101 L 128 9.6e-3, r286 L 128
8.5e-3, r286 L 384 8.3e-3) -- each no worse than the H100 cuda token of the case (11.6e-3, 12.4e-3, 8.2e-3) or the B200 triton token
(9.4e-3 at r286 L 384); the cond-LN weight gradients are summed in another fixed order (8-row partials). Determinism (`_r3/det.py`, four
runs; `_r3/out/fin5.check`): 42 of 42 cases bitwise repeatable (rows 101 / 108 / 143 / 144 / 163 / 263-265, L 128 / 256, both dtypes, eval
and train); captured == eager and replays equal on 22 cases (incl. the folds). Plans: `_prebuild --arch sm90a / sm80 / sm120 --list` equal
to origin/main's (513 / 72 / 64 artifacts), `--arch sm100` 484 entries with the new builds.

**H100 audit** (the H100 hand-CUDA token DiT -- `docs/h100/token_dit.md`, `token_dit_train.py` `_native90` branches,
`conditioned_transition.cuda.token_inference`): every technique, and whether the B200 path has it.

| H100 technique | B200 |
|---|---|
| inference: weight pack once per weight version (static per (pointer, version)) | used (part 2: the default-option blocks too) |
| inference: hoisted pair bias, one pass (LN(pair) Wf + key mask) | used (`apb.pair_bias`; fp32 `pair_bias32_tc` from L 256) |
| inference: cond LayerNorm rows + two GEMMs | bf16: beyond it (the folded conditioning: one GEMM, no LN pass); fp32: as H100, the gates' GEMM beside the chain (part 3) |
| inference: gated core writes sigmoid(g) o over q | used (`attn_inf`, `attn_inf_tf32`) |
| inference: one sample -- the H100 core's per-sample schedule | B200 beyond it: bf16 `attn_inf -DQPAIR`, fp32 `attn_inf_tf32 -DQPAIR` (part 3) |
| inference: B trunk_seeds in ONE core launch (group s % B), one bias build per trajectory | used (part 3: `attn_inf -DFOLDARG`, `apb.pair_bias_batch`); was one launch + one pair pass per fold |
| inference: expand + SwiGLU in one GEMM (bf16 always; TF32 from 768 rows) | bf16 from 3840 rows (`gemm_swiglu2_sm100`; below it cuBLAS + `swiglu_rows` is faster: 13.7 against 14.0 us at L 384 S 5); fp32: not applicable -- the B200 Transition's TF32 SWIGLU GEMM (`tg_gemm.cu`) measured equal to cuBLAS + `swiglu_rows` at 640 rows (r108 104.5 against 104.5 us, r110 98.1 / 96.4), not taken |
| inference: PDL row passes | used (the `*_cf` passes, `_PDL`) |
| training: one opaque op forward / backward, autocast off | used |
| training: weight pack per step (`train_pack90`) | used (`train_pack100`, 8 vectors a thread); the pack beside the chain on a side stream measured slower (+7 us at r144 L 128: a full grid holds every SM) -- the conditioning prep runs beside it instead (part 3) |
| training: bf16 pair bias in one pass forward, one-pass backward | used (`apb.pair_bias_fwd_stats`, `apb` `pair_bias_bwd` at pair 128; the rows' `pair_bias_bwd` at 256 / 512) |
| training: the core writes og (the gate in its epilogue), O not kept | missing: built and measured (`attn_fwd2 -DOGATE`, part 3) -- no gain (r144 master L 128 180.5 against 178.0 us without it, r101 202.0 / 200.9): the per-row g loads / og stores in the epilogue cost what the 3 us gate pass did; reverted |
| training: bf16 backward with three samples per CTA (A a multiple of 3) | not applicable to the missing cases (one sample); the B200 cores have their own deterministic dQ slices |
| training: fp32 backward on the TF32 cores, tiled transposes for the token-contracting products | used in substance (`attn_dqb_tf32` / `attn_dkv_tf32` read MN-major operands: no transposes) |
| training: weight gradients cuBLAS + one fixed-order `final` kernel | B200 beyond it (part 3): the GEMMs write the leaves, the rest of the final gradients beside the chain -- the H100 still pays the final copy |
| training: graph-owned cuBLAS scratch (A = 5, L 384 / 768) | not applicable (an H100 capture workaround) |
| training: `train_bias90` (bias + key masks in one pass) | used in substance (the pair pass writes the masked bias itself) |

### Round 3, part 4 (2026-11-17, evening): Chai-1 fp32 gate columns -- `cuda:b200_sm100` 6 (TokenDiTChai1 only)

Records `$W/b2511/agents/r3-token-dit/_r3/out/` (`cg*.{1,2}.jsonl`: 1 = new, 2 = `MINIWORLD_TOKEN_B200_NOGATE_GEMM=0`; each length group
its own run; `miss6.jsonl` / `miss6.txt`: the part-3 misses re-measured alone, one length group per run).

| change | effect (B200) |
|---|---|
| fp32 inference without a gating query (Chai-1): the static pack's gate rows are zero (`to_gate.weight = 0`) and their bias +1e4, so ONE contiguous q \| k \| g GEMM (bias in cuBLAS's epilogue) writes the gate columns -- before, every call filled them through a strided view (5.0 us) and `addmm(..., out=qkvg[:, :2D])` copied the bias into the strided view first (6.0 us) | Chai-1 fp32 eval, rows 110 / 111 / 185 / 186: L 128 98.8 / 98.3 / 90.6 / 92.1 -> 98.0 / 96.9 / 90.0 / 91.9 us, L 256 143.4 / 145.3 / 137.1 / 137.2 -> 133.1 / 133.2 / 127.0 / 135.2, L 640 334.7 -> 319.2 (r110), L 1024 587.6 -> 573.5, L 2048 1789 -> 1759; accuracy: the same lights and errors as the B200 tokens (12 cases L 128 / 256 / 1024); capture == eager, replays equal, four runs bitwise equal (L 128 / 256) |

The part-3 misses re-measured alone (`miss6.txt`): 15 of the 20 still above 0.667 -- r111 fp32 L 128, r144 fp32 L 512, RoseTTAFold3 r109
master L 256 (77.7 us, 0.635) and r184 master L 512 (122.8, 0.667) pass when timed alone; Chai-1 r110 fp32 L 128 sits at 0.667 (96.2
against 144.3 us). Still above: five-sample fp32 inference L 128 (Boltz-2 108 / 183 0.73, RoseTTAFold3 109 / 184 0.72-0.74, Chai-1 r110
0.667-0.68), Boltz-2 183 fp32 L 256 0.687, Protenix one-sample master training L 128 (rows 101 / 143 / 144 / 260 / 261 / 286, 0.67-0.69)
and r101 L 256 0.688, Protenix r144 fp32 one sample L 256 / 640 (0.675 / 0.721), RoseTTAFold3 r184 master L 256 0.697.

### Round 3, part 5 (2026-11-17, night): one-sample training restructured -- built, measured, all rejected (no version bump)

Target: the 15 misses of `miss6.txt` (Protenix one-sample master training L 128 rows 101 / 143 / 144 / 260 / 261 / 286 at 0.67-0.69,
r101 L 256 0.688; fp32 inference r144 L 256 / 640, five-sample fp32 inference L 128, RoseTTAFold3 r184 master L 256). Base 8d27e53b7
(APB part 6 merged). Records `$W/b2511/agents/r3-token-dit/_r3/out/` (`q6*.N.jsonl`, `s7q6*.N.jsonl`, `s8q6*.N.jsonl`,
`s9q6*.N.jsonl`: A/B pairs in `_r3/j6.sh` - `j9.sh` order, one length group per run, `dev.sh -x`, graph timing, L2 flushed); timelines
`tl_t6{a,b,c,d}.log`; the code of every rejected piece in [rejected/token_dit_part5_train.diff](rejected/token_dit_part5_train.diff) (627 lines).

Built (all behind B200 env switches, measured against the same tree with them off):

| piece | what | effect (us, off -> on; r101 / r143 / r144 / r260 / r261 / r286 at L 128, r101 / r143 at L 256) |
|---|---|---|
| grouped weight gradients `wgrad100` (sm100 TU) | every `C_i = A_i^T B_i` of a group in ONE launch: a CTA per 128 x 128 tile walks all M rows (mma.sync m16n8k16 bf16 with ldmatrix.trans, m16n8k8 TF32), fp32 leaves written directly, no split (bitwise repeatable); optional epilogue = `final_lnw100` (leaf = acc w[col], cond-LN partial rows). Three launches a step (after res_adaln_b_bwd: squeeze, expand a / b, to_out, both to_scale, the transition AdaLN's cond leaves; after qknorm_bwd: q / k / v / g; after adaln_a_bwd: the attention AdaLN's cond leaves) + `final_rest100` straight on the row partials: ~13 cuBLAS GEMMs + `final_lnw100` + `final_reduce100` -> 3 launches. Unit test vs fp64: bf16 rel 2-5e-7, TF32 3e-4, repeat bitwise | with the cond split below (rows' pair backward as before): 205.0 / 177.2 / 178.3 / 180.1 / 186.5 / 251.9 -> 202.8 / 180.3 / 180.1 / 178.3 / 186.4 / 251.9 (level within +-2 %); L 256 287.0 / 216.8 -> 295.1 / 227.3 (+8 / +10): rejected |
| conditioning backward in parts (`cond_bwd3`) | dGg Wg and dG[:, 2D:] Wn[2D:] (fp32) on a fourth side stream after res_adaln_b_bwd, only dG[:, :2D] Wn[:2D] after adaln_a_bwd, `cond_bwd3` adds the fp32 parts | 205.0 / 176.3 / 177.7 / 179.7 / 186.4 / 251.9 -> 206.9 / 182.4 / 180.1 / 180.3 / 186.6 / 254.0; L 256 288.8 / 217.0 -> 289.3 / 223.3: rejected |
| pair backward issued after the dxa GEMM | its full grid no longer ahead of qknorm_bwd / the dxa GEMM | with the two above: level to +2 at L 128, L 256 r101 289.3 -> 287.4: rejected (no gain) |
| `pair_bias_bwd100` (sm100 rebinding of the rows' pass, pair 256 / 512) | each warp's four rows of the NEXT tile loaded into registers during the dWf / dbias / dxh phases (same products, same values) | alone: L 128 level (+-1 %), r101 L 256 287.0 -> 292.0: rejected |
| the sm_100a expand GEMM + SwiGLU epilogue (`gemm_swiglu2 -DSAVE_AB`) in training from 128 rows | one launch for expand + swiglu_rows | 202.9 / 176.1 / 178.2 / 180.2 / 186.5 / 251.8 -> 207.0 / 185.9 / 184.3 / 182.4 / 190.4 / 253.8; L 256 288.9 / 216.3 -> 290.8 / 224.0: rejected |

Launch count of the one-sample master step (r143 L 128, graph timeline): 48 kernels before; 41 by construction with the grouped
gradients (the H100 step: ~45 -- its cuBLAS weight gradients + one `final` kernel). The halving the brief asks for was not reached and
the 7 removed launches bought nothing: **why (measured)**: the timelines' side-stream branches (weight gradients, pair backward,
conditioning GEMMs) are not the critical path -- the step is the forward / backward *chain* of ~30 dependent kernels of 2-9 us
(pack, 6 GEMMs, core, 5 row passes forward; 6 dX GEMMs, two cores, 7 row passes backward). Every side kernel moved or merged only
changed which chain kernel waited for SMs: fewer, larger side launches (the grouped kernel's 324 tiles after res_adaln_b_bwd) delayed the
attention backward at L 256. (The profiler's timelines also inflate the gaps: r101 L 256 showed qknorm_bwd 57 us behind the pair
backward, but issuing the pair backward late gained 2 us of the 289.) What would gain is a shorter chain -- row passes folded into the
GEMMs' prologues / epilogues (custom tcgen05 GEMMs with LN / gate epilogues for 128-row operands, which the round-3 split-K GEMM showed
cost 1.2-2.5 us of setup each) or a persistent step kernel; neither is a part-5-sized change.

Not built this part, with the reason:

| item | why |
|---|---|
| key split for the one-sample TF32 inference core (r144 fp32 L 256 / 640) | `attn_inf_tf32.cu` is AttentionPairBias's fp32 inference core too (agent `r3-apb`, which merged a KPERM / device_index fix of it in part 6 tonight) -- a `-DKSPLIT` there plus a fixed-order combine is an APB-kernel change: requested from `r3-apb` (the bf16 `attn_inf -DKSPLIT` + `ksplit_combine` exists as the model) |
| RoseTTAFold3's flat q / k LayerNorm folded into a neighbour (r184 master L 256, needs 3.3 us) | the neighbours are cuBLAS (the q / k GEMM) and the shared bf16 core (APB); a custom tcgen05 q / k GEMM with the LN in its epilogue needs 768 fp32 accumulator columns per row (TMEM holds 512) -- two passes over N with a staged first half |
| fp32 640-row TF32 GEMMs (five-sample fp32 inference L 128) | measured cause stands (60 CTAs of 1.2 MB operands each: ~80 GB/s per SM); a cluster-multicast / split-K TF32 GEMM is the fix; not built |

State after part 5: unchanged from part 4 (`miss6.txt`: 15 cases above 0.667, none slower than the H100). Accuracy of the training
pieces while on (`_r3/out/acc6.jsonl`, `acc7.jsonl`, 12 cases rows 101 / 143 / 263-265 / 286 L 128, both dtypes): 0 RED, all inside the
engine standard (r101 / r286 master YELLOW 1.06 / 1.06 x tol, as the part-3 records); determinism bitwise on every case printed (rows 263-265, both dtypes),
captured == eager on rows 143 / 263. Nothing of it is in the tree.

### Round 3, part 6 (2026-11-17, night): true state, the v^T placement, dchat in one launch -- `cuda:b200_sm100` 20

Base ddd984081 (b200-2.5.x with APB's fp32 one-sample key split). Records `$W/b2511/agents/r3-token-dit/_r3/out/`: `m0/` (before: cuda,
triton, pytorch on every requirement case of the seven classes, 646 cases, one `dev.sh -x` run per length and backend, `_r3/m/meas.sh`),
`m1/` (after: cuda), A/B pairs (A B A B, one length group per run) in `$W/b2511/agents/r3-token-dit-x/_r3/out/ab*/`, the triton /
pytorch alternation `abk1/`; tables `_r3/m/viol.sh` / `_r3/m/abcmp.sh`.

Violations, before -> after (H100 >= 20 us; ratio = B200 cuda / H100 cuda, target <= 0.667; ladder: 2 % band):

| class dtype mode | 1.5x misses | worst ratio | ladder t > p |
|---|---|---|---|
| TokenDiTBoltz2 float32 eval | 3 -> 2 | 0.729 -> 0.729 | 0 |
| TokenDiTBoltz2 master / float32 train | 0 | 0.548 / 0.532 | 1 / 1 -> 0 / 0 |
| TokenDiTChai1 float32 eval | 1 -> 0 | 0.667 -> 0.667 (r110 L 128 96.2 -> 94.1 us) | 0 |
| TokenDiTProtenix float32 eval | 2 -> 0 | 0.711 -> 0.656 | 0 |
| TokenDiTProtenix master train | 8 -> 5 | 0.699 -> 0.697 | 4 -> 0 |
| TokenDiTProtenix float32 train | 0 | 0.625 | 1 -> 0 (1 of 2 pytorch runs within 3.5 %) |
| TokenDiTRoseTTAFold3 float32 eval | 2 -> 2 | 0.743 -> 0.727 | 0 |
| TokenDiTRoseTTAFold3 master eval | 1 -> 1 | 0.679 -> 0.680 | 0 |
| every other class / dtype / mode | 0 | <= 0.665 | 0 |

Total 17 -> 11 misses, none slower than the H100 before or after; cuda < triton everywhere. The seven "triton slower than pytorch"
training cases of the before run were the order of the runs (triton timed right after the cuda run of a length): re-measured
alternating (triton, pytorch, triton, pytorch) triton is 1-6.6 % faster on all 30 cases (rows 143 / 144 / 260 / 261 / 263, L 128 - 384,
both dtypes), except one pytorch run of r144 fp32 L 256 (1044.9 against 1095.7 us the other run; triton 1083.3).
r144 fp32 L 256 / 640 and Protenix r144 / r260 master L 128 left the miss list with the merged base / within the run-to-run spread.

Taken:

| change | effect (us, before -> after) |
|---|---|
| fp32 inference, several samples (`token_inference`, `_B200_VT_RULE`): the v^T GEMM on the chain at L 128 when the samples share the conditioning or there is no gating query; on a stream of its own at L 256 with a shared conditioning (beside the q \| k rows and the pair pass). Bits unchanged | Chai-1 r110 / r111 L 128 96.1 / 97.6 -> 94.1 / 94.2; shared-cond rows 179-186 L 128 2-11 % faster (r182 92.0 -> 81.9); Boltz-2 r183 L 256 132.8 -> 126.9 |
| bf16 training up to 256 rows: dchat = dG Wn ([M, 3072] x [3072, 384]) as one split-K cluster launch (`tdt_epi_sm100.cu` EPI 0, K split 8, partial tiles reduced in rank order over DSMEM: deterministic) instead of cuBLAS split K + its reduce kernel on the step's tail | r101 L 128 203.9 -> 198.0, r144 L 256 214.2 -> 207.0, r261 L 128 186.6 -> 174-185 (A/B `ab5`); level elsewhere; at 1024 rows (r286 S 8) 6 % slower, so only <= 256 rows |
| `tdt_epi_sm100.cu` producer: the first NST stages issued by NST lanes (one thread issues ~1 TMA load per 600 clk) | the dchat shape measured level (6.14 us at M 128 either way: these CTAs are not issue bound) |

Built, measured, rejected (code in [rejected/token_dit_part6.diff](rejected/token_dit_part6.diff)):

| piece | result |
|---|---|
| fp32 inference: the pair pass issued right after the first conditioning GEMM, the gates' GEMM on a second side stream | L 128 2-11 % slower on 16 of 20 cases (RoseTTAFold3 r109 90.1 -> 100.2): the pair pass takes the SMs the conditioning / q \| k chain needs (as part 3 found) |
| v^T on the chain / on its own stream for every multi-sample length | per-sample rows with a gate 2-9 % slower at L 128 (r109 90.1 -> 98.2), every row 1.3-3 % slower at L 640: only the rule above is taken |
| the training weight pack in two launches (`train_pack100r`: Wo / [W_a; W_b] / W_s on a side stream, 64 or 148 CTAs) | 1-9 % slower (r145 L 128 198.5 -> 216.9): the head part still takes 8 us of the 12.8 (latency, not bytes), the tail (15-31 us) slows the chain's kernels beside it |
| `tdt_epi` epilogue routes: res_c in the squeeze GEMM, SwiGLU in the expand GEMM, swiglu_bwd in the dh GEMM (the cancelled agent's work, measured now) | res_c + dchat: -2 to +2.6 % at one sample, +9.7 % at 1024 rows; alone (chain of 10, L2 flushed): res_c 6.76 against 7.17 us at M 128 but 17.4 against 9.2 at M 1024; swiglu 6.96 / 8.80 against 6.55 / 6.76 (cuBLAS + rows), swiglu_bwd 7.17 / 9.84 against 6.14 / 6.76 |
| dcg = dGg Wg by `tdt_epi.mm` too | level to +3.7 % (r144 / r260 L 128) |
| fp32 GEMMs on other kernels: `b200_atom.gemm32` (persistent kind::tf32) / fp16 operands with fp32 output (cuBLAS) at the 640-row shapes | gemm32 9.8-24.8 against cuBLAS 5.3-9.8 us; fp16 -> fp32 0.9-1.3x of TF32 (expand 640 rows 12.7 against 9.8) |

What remains (11 cases) and the measured reason:

* five-sample fp32 inference at L 128 with per-sample conditioning or the flat q / k norm: Boltz-2 r108 / r183 (0.729 / 0.713), RoseTTAFold3
  r109 / r184 (0.719 / 0.727). The chain is six cuBLAS TF32 GEMMs at 640 rows (6-15 us each, ~50 of ~88 us: 60-120 CTAs of
  128 x 256 / 128 x 128 tiles, latency bound) plus 8-9 row passes; no faster 640-row TF32 GEMM was found (above). The rna rows (q / k / v^T to
  TF32 for the core, 2.3-4.4 us) would fold into the TF32 core's operand loads -- an APB core change (requested from `r3-apb`).
* one-sample master training at L 128 (Protenix r101 0.691, r143 0.674, r261 0.669; r261 L 256 0.670) and r286 S 8 L 128 (0.697): the chain of
  ~30 dependent kernels (part 5); the A/B pairs of this part move these by +-3 % run to run (r143: 168.3 / 178.4 us in two runs of the same
  tree) -- r143 / r261 sit inside that spread of the target. What would gain: the gate (`gate_o`, 2.9 us) folded into the forward core's
  epilogue and `bias_transpose` (2.1 us) into the backward core (APB kernels), or a persistent step kernel.
* RoseTTAFold3 r184 master eval L 256 (0.680): the flat q / k LayerNorm pass (3.9 us) between the q \| k GEMM and the core (part 5's analysis).

Accuracy (`measure.accuracy`, `_r3/out/fin/acc{T,E}.jsonl`: master training rows 101 / 103 / 143 / 144 / 260 / 261 / 263-265 / 286 L 128 /
256, fp32 inference rows 108 / 110 / 179 / 183 / 184 L 128 / 256): 0 RED; 17 GREEN / 13 YELLOW; fp32 inference the same errors as the B200
tokens (L 128 bitwise the same error values); training r101 L 128 GREEN -> YELLOW (1.06 tol, part 3's level), r103 L 256 / r261 / r286 L 256
YELLOW -> GREEN. Determinism (`_r3/det.py`, the 20 training cases): output and every gradient bitwise equal over two runs.

### Round 3, part 7 (2026-11-17/18): the backward core pair beside each other, the 640-row TF32 GEMMs measured -- `cuda:b200_sm100` 21

Base 7bc2db519 (APB v21), then b070dfb11 (engine 2.5.14; no token DiT change). Records `$W/b2511/agents/r3-token-dit/_r3/out/p7m0/`
(before: cuda, every requirement case of the seven classes, 646 cases, one `dev.sh -x` run per length; triton / pytorch of part 6's
`m0` -- those rungs did not change), `$W/b2511/agents/r3-token-dit-x/_r3/out/p7m1/` (after), A/B pairs (base tree against work tree,
A B A B) `ab_p7{a,c,t,t2}/`, env A/B `e_*/`, timelines `$W/b2511/jobs/tdt7_tl.log`, GEMM benches `$W/b2511/jobs/tdt7_g32{,b,c}.log`.

Taken (bf16 training; the same bits -- eager outputs and every gradient bitwise equal to the base tree on rows 101 / 143 / 144 / 261 /
286, L 128 / 256; captured == eager and replays equal on 40 cases; run-to-run bitwise on 40):

| change | effect (us, base -> work, A B A B means) |
|---|---|
| `attn_dqb` launched on side stream 2 beside `attn_dkv` (`sm100.backward(side=...)`; with `det` attn_dkv fills no dQ, the two read only their inputs; joined before dQ / dbias are read). The two grids are 16 + 16 CTAs at L 128 one sample: they ran back to back on the chain (6.3 + 8.9 us) | with the next row, master train L 128: r101 198.8 -> 191.9, r143 177.3 -> 170.2, r261 186.5 -> 177.1, r286 S 8 250.4 -> 243.4, r286 S 48 676.5 -> 651.4; L 256: r101 281.3 -> 273.6, r143 219.3 -> 211.2, r261 232.7 -> 225.5 |
| one sample: `attn_dkv -DBIASNT=1` reads the bias as stored (AttentionPairBias's flag, offered by `r3-apb`): no `bias_transpose` launch (2.0 us at L 128) on the chain | (in the row above) |

Built, measured, rejected (code in [rejected/token_dit_part7_gemm32.diff](rejected/token_dit_part7_gemm32.diff) and
[rejected/token_dit_part7_inference.diff](rejected/token_dit_part7_inference.diff)):

| piece | result |
|---|---|
| (a) TF32 `tdt_epi` (`-DTF32`, kind::tf32, K split over a cluster with the rank-order DSMEM reduce, multi-lane producer) at the six 640-row GEMMs + v^T | every shape slower than cuBLAS: g1 K 768 N 3072 12.3 against 9.4 us, qkvg 12.3 / 9.4, ya 8.2 / 6.6, squeeze 11.3 / 8.4; the K split makes it slower (ks 2: 20-30 us); the multi-lane producer changed nothing (not issue bound); error 5.4e-4 (truncated operands) against cuBLAS's 2.1e-4 |
| (a) `gemm32mc_sm100` -- weight-major tiles (128 weight rows x 128-256 activation rows; TMEM lane = output column, coalesced stores), a CW x CX cluster multicasting both operands (each CTA loads 1 / CX of the weight box and 1 / CW of the activation box) | best config per shape (of bx 128-256, cw 1-8, cx 1-8): g1 K 768 11.5 / 9.4, qkvg 11.3 / 9.6, ya 10.4 / 6.6, vT 10.4 / 6.6, squeeze 17.0 / 8.2; only Boltz-2's up-gate expand (N 4608) 12.7 against 13.3. The multicast configs are never faster than cw = cx = 1: the per-SM L2 intake is not what bounds these GEMMs (part 5's reading of the cuBLAS kernels does not carry over); a CTA spends ~0.4 us per 32-wide K block (pipeline latency, 5-6 stages in flight) |
| (c) RoseTTAFold3's flat q / k LayerNorm with q and k of a row in one pass (`qk_ln100`: both halves loaded first, the sums paired; the same bits) | level: r184 master L 256 75.7 / 75.7, fp32 L 128 89.9 / 89.9, L 256 112.6 / 112.6. In the timelines the pass is 3.8-6.0 us beside the pair pass either way (it is the pair pass's neighbour, not its own latency) |
| the fp32 pair pass (`pair_bias_m_k`) with each warp's rows loaded before the first is reduced and the next tile's in flight under the MMA (the same bits) | Boltz-2 r108 / r183 level, RoseTTAFold3 r109 / r184 2.1 us slower (the faster side pass takes the SMs of the main chain's qkvg GEMM) |
| five samples, per-sample conditioning with a gate at L 128: v^T on a stream of its own beside the pair pass (`VT_OWN128`) | Boltz-2 r108 102.4 -> 100.3 / 102.3 -> 102.1, r183 96.0 -> 94.1 / 95.9 -> 94.8; Protenix r99-r101 / RoseTTAFold3 r109 4-6 % slower |
| (b) one-sample training row passes into GEMM epilogues | not rebuilt: part 6 measured the `tdt_epi` epilogue forms at <= 256 rows (res_c -2 to +2.6 % at the step, swiglu / swiglu_bwd 0.4-3 us slower); this part removed two chain launches instead (above) |

The five-sample fp32 L 128 timeline (Boltz-2 r108, `tdt7_tl.log`): the core waits for the side branch (pair pass 16-17 us beside the
q | k | g GEMM, then v^T 6.7 + rna 2.3) about 12 us after the main chain's q / k rows are done; making either side kernel faster or
moving v^T beside the pair pass did not move the step (rows above): the two branches share the SMs, so the step is bound by the
total work of the fork, not by one branch.

Split (work tree, `MINIWORLD_TOKEN_B200_DQB_SIDE=0` against on, A B A B): r101 / r143 / r286 S 8 / r286 S 48 at L 128 196.8 / 176.3 /
249.7 / 675.8 -> 191.6 / 170.4 / 241.7 / 642.4 (the side dqb); the rest of the base -> work difference (~2 us at one sample) is the
dropped transpose.

Violations before (`p7m0`) -> after (`p7m1`, work tree on b070dfb11; ratio = B200 cuda / H100 cuda, H100 >= 20 us; ladder 2 % band):

| class dtype mode | 1.5x misses | worst ratio | ladder |
|---|---|---|---|
| TokenDiTProtenix master train | 5 -> 2 | 0.700 -> 0.680 | 0 |
| TokenDiTBoltz2 float32 eval | 2 -> 2 | 0.728 -> 0.729 | 0 |
| TokenDiTRoseTTAFold3 float32 eval | 2 -> 2 | 0.726 -> 0.726 | 0 |
| TokenDiTRoseTTAFold3 master eval | 1 -> 1 | 0.680 (r184 L 256) -> 0.667 (r184 L 512, 122.75 against 122.7: inside the timer's 2 us step; A B A B base / work 118.6 / 120.9 / 120.8 / 121.3, eval is unchanged by this part) | 0 |
| TokenDiTProtenix float32 train | 0 | 0.625 | t > p 1 -> 0: r144 L 256 alternated triton / pytorch 1057.8 / 1111.8, 1103.7 / 1110.9 |
| every other class / dtype / mode | 0 | <= 0.666 | 0 |

Total 11 -> 8 in the sweeps (Protenix r143 / r261 L 128 and r101 L 256 out), none slower than the H100, cuda < triton everywhere.
The two Protenix master training misses left: r286 S 8 L 128 243.4 us (0.674 in the A/B; the sweep 245.6, 0.680) and r101 L 128
(sweep 199.0, 0.678; the A/B runs 190.7-192.4, 0.650-0.655: the one-sample L 128 steps move by +-4 us run to run). The fp32
five-sample L 128 inference (Boltz-2 r108 / r183, RoseTTAFold3 r109 / r184, 0.71-0.73) is unchanged: no faster 640-row TF32 GEMM was
found (above) and the fork's two branches are bound by their shared SMs.

Shared sources: `kernels/augmented_attention/cuda/sm100/__init__.py` (`backward`: `bias_nt` / `side`, both default off -- the other
callers launch what they did); `prebuild_sm100.py`: the token DiT's `attn_dkv -DDET=1 -DBIASNT=1` build added to the B200 plan.

### Round 3, part 8 (2026-11-18): v MN-major in the TF32 core, the parallel block's transition beside the attention -- `cuda:b200_sm100` 22

Base 6facf3101 -> 80572b7d4 (b200-2.5.x on engine 2.5.14; no token DiT change in between). Records: before `$W/b2511/agents/r3-token-dit/_r3/out/p8m0/`
(cuda, triton, pytorch, every requirement case of the seven classes, one `dev.sh -x` run per length and backend, GPU 3), after
`$W/b2511/agents/r3-token-dit-x/_r3/out/p8m1/` (cuda, every case) with the fp32 eval rows re-measured after the VMN length rule in
`$W/b2511/agents/r3-token-dit-y/_r3/out/p8m2/`; env A/B `r3-token-dit-{x,y}/_r3/out/x*`; timelines `$W/b2511/jobs/tdt8_x3.log`.

What the H100 does for the target cases (fp32, five samples, L 128; `token_inference` non-B200 branch, read 2026-11-18): one stream,
16 launches -- cond_rows, the two conditioning GEMMs, the key-permuted one-pass pair bias (`pair_bias_tok_cuda`), adaln_in_rows, the
q | k | g GEMM, rna (or the flat q / k LayerNorm), the v^T GEMM + its rna, `attn_tf32_inf` (wgmma, v^T K-major), the out projection,
resgate_adaln, the expand GEMM + swiglu_rows (the TF32 wgmma SwiGLU GEMM only from 768 rows), the squeeze GEMM, resgate_out. No fork,
no PDL on the fp32 rows. The B200 path had the same launches on two streams (pair pass + v^T beside the chain); after this part it has
14 (no v^T GEMM, one rna pass for q | k | v) and, for the parallel blocks, the transition on a third stream.

Taken:

| change | effect (us, graph timing) |
|---|---|
| fp32 inference, several samples, L <= 128 (`token_inference` `vmn`, `attn_inf_tf32 -DVMN`): ONE q \| k \| v \| g GEMM (the static pack keeps the reordered weight / bias), q \| k \| v rounded in one rna pass, the TF32 core reads v MN-major from its column view (32-column atoms, 128-B swizzle with 32-B atomicity, as attn_fwd_tf32.cu; the 48-wide tile is two atoms, so the 16 x 48 VMN build keeps two stages) -- no v^T GEMM, no v^T rna. Also under VMN: both samples' k in one 3-D box and v in one 4-D box (3 TMA boxes a stage instead of 8). Outputs bitwise equal to the v^T core (eager dumps, 38 cases) | Boltz-2 r108 L128 104.4 -> 98.0, r183 94.0 -> 89.9, Chai-1 r110 / r185 L128 87.8 / 80.1 -> 77.8 / 73.4, Protenix r99 / r100 94.0 / 96.0 -> 89.9; RoseTTAFold3 r184 (with the next row) 83.7 -> 71.6 |
| B200 inside a capture, parallel blocks (RoseTTAFold3, Chai-1), bf16 and fp32: the transition (expand, SwiGLU, squeeze) on a stream of its own, forked after the q \| k rows, joined before the output pass (the same kernels, the same bits) | RoseTTAFold3 r109 / r184 fp32 L128 90.1 / 83.7 -> 81.7 / 75.6 (VMN off / on), master r184 L256 / L512 75.6 / 122.8 -> 71.6 / 109.2; Chai-1 master 5-10 % faster at L 128 - 512 |

Rule from the per-row A/B (`x6`, VMN on / off with the transition fork): VMN loses from L 256 (L 512 7-11 % slower: the two-stage core)
and on the parallel block with a gate and a per-sample conditioning (RoseTTAFold3 r109 L128 80.7 -> 83.7: there v^T beside the chain
overlaps better) -- both keep the v^T path.

Built, measured, rejected (code in [rejected/token_dit_part8_fold32.diff](rejected/token_dit_part8_fold32.diff)):

| piece | result |
|---|---|
| fp32 folded conditioning (`fold32_in / mid / out` in token_dit_train_rows_sm100.cu: G = c [W1; W2]^T in one TF32 GEMM, the cond LayerNorm applied in the row passes, no fp32 residual copy) | level: Boltz-2 r108 98.1 / 98.0, r183 89.9 / 90.1, RoseTTAFold3 r109 88.1 / 90.1 (fold off / on, VMN on); the 6 D GEMM (15.1 us at 640 rows) costs what cond_rows + g1 + the side g2 cost; accuracy 0 RED but more YELLOW |
| fold + the gates' GEMM on the side stream | capture error (unjoined work) -- not pursued after the fold measured level |
| fold + the pair pass issued after the first row pass | 6-8 % slower (r108 104.4, r184 82.2) |
| the transition forked right after the AdaLN pass (instead of after the q \| k rows) | fp32 r109 / r184 83.7 / 77.8 against 81.7 / 75.6; bf16 mixed |

Accuracy (`measure.accuracy`, fp32 eval rows 99-102 / 108-111 / 143-145 / 179-186 L 128 / 256, master eval of the option models L 128 /
256): 0 RED; fp32 14 + 3 + 1 + 6 YELLOW, worst 1.27 tol (Protenix r143 one sample: a path this part does not change); master 0 YELLOW.

Violations before (`p8m0`) -> after (`p8m1` + `p8m2`; ratio = B200 cuda / H100 cuda of the 2.5.14 tokens, H100 >= 20 us; ladder 2 % band):

| class dtype mode | 1.5x misses | worst ratio | ladder |
|---|---|---|---|
| TokenDiTBoltz2 float32 eval | 2 -> 2 | 0.729 -> 0.686 (r108 L128 104.4 -> 98.3; r183 L128 0.681 -> 0.681, 89.9 in both sweeps, 94.0 -> 89.9 in the A/B pairs) | 0 |
| TokenDiTRoseTTAFold3 float32 eval | 2 -> 0 | 0.726 -> 0.639 (r109 L128 90.1 -> 80.0, r184 L128 87.9 -> 73.5) | 0 |
| TokenDiTRoseTTAFold3 master eval | 2 -> 0 | 0.696 -> 0.618 (r184 L256 77.6 -> 68.0, L512 122.8 -> 109.2) | 0 |
| TokenDiTProtenix master train | 2 -> 1 | 0.680 -> 0.674 (r286 S 8 L128 245.5 -> 243.7, training is unchanged by this part; r101 L128 198.7 -> 194.8 us, 0.677 -> 0.663: the one-sample training step moves by +-4 us run to run) | 0 |
| every other class / dtype / mode | 0 | <= 0.662 | 0 |

Total 10 -> 3, none slower than the H100, cuda < triton everywhere. The sweep's five "triton slower than pytorch" training cases (r145 /
r261 L128, r263 L256, r265 L384, r264 L512: run order, triton timed right after cuda) re-measured alternating triton / pytorch twice
(`r3-token-dit-x/_r3/out/abk8{a,b,c,d}`): triton faster on all but r264 fp32 L512 (1254.2 / 1247.2, 0.6 %: inside the band). Over the
646 cases after / before geomean 0.988; no case left the target.

What remains (3) and the measured reason:

* Boltz-2 fp32 five samples L128 (r108 0.686, r183 0.681): sequential block, so the transition cannot leave the chain; the chain is now
  q | k | v | g GEMM 11.7 -> rna 3.7 -> core 11.0 -> out projection 6.5 -> resgate 4.2 -> up-gate expand 14.4 -> swiglu 3.1 -> squeeze 8.8 ->
  out 3.7 us (`tdt8_x3.log`); the 640-row TF32 GEMMs are cuBLAS (no faster one found in parts 5-7), the core's VMN build keeps two stages
  (1.4 us over the three-stage v^T build) and merging its k / v boxes (8 -> 3 a stage) measured level (98.0 / 97.9): the core is not TMA-issue bound.
* Protenix master training r286 S 8 L128 (0.674): unchanged by this part (part 7's state).

### Round 3, part 9 (2026-11-18): CTA-pair TF32 GEMMs for the 640-row fp32 chain -- `cuda:b200_sm100` 23

Base 719b99c9e (b200-2.5.x, engine 2.5.14; the token DiT sources are part 8's, so part 8's `p8m1` / `p8m2` sweep is the state before:
3 misses, ladder 0). Records `$W/b2511/agents/r3-token-dit/_r3/out/p9*` and `$W/b2511/jobs/p9*.log`; probes `_r3/m/p9g.py` (GEMM sweep
against cuBLAS, graph chains of 10, L2 flushed), `p9p.py` / `p9s.py` (timing builds: `-DNOLOAD` / `-DNOMMA` / `-DNOSTORE` / `-DNOEPI`,
`-DSTAMP` per-CTA `%globaltimer` phase stamps), A/B in the step `_r3/m/xw.sh` (env variants interleaved, min over rounds).

Built: `conditioned_transition/cuda/tdt_pair32_sm100.cu` + `pair32.py` (sm_100a only). C = A W^T, fp32 operands on kind::tf32, one
2-CTA cluster per (2 HR rows, 2 NH weight rows): CTA r loads HR rows of A and NH weight rows, the leader issues M = 2 HR, N = 2 NH
products (`cta_group::2`), the whole K in tensor memory (no K split). HR 64 (the Transition part-14 folded layout) or 128 (each CTA its
128 rows x the whole N). Operands as 3-D TMA boxes of KBX K-blocks, lane s of the producer warp owning ring slot s, weights issued before
`griddepcontrol.wait`. Epilogues: bias + TF32 rounding of the first `rnac` columns (q | k | v), SwiGLU(-up) of a regrouped weight
(`pack_swiglu`: each weight half [Wa | Wb | Wu] of NH / 3 units; h rounded to TF32), and `x + sigmoid(gl) acc`. Row segments are
transposed inside lane groups before the global accesses (tsx_store.cuh's butterfly). Errors against fp64 equal cuBLAS's when the
operands are TF32-rounded (squeeze 1.7e-6 both); with a raw fp32 A, 2.9e-4 against cuBLAS's 1.5e-4 (the tensor cores read the TF32
bits, cuBLAS rounds).

GEMMs alone (`p9g.py` / `p9p.py`, M 640, us, best config (HR, NH, KBX, NG)): q | k | v | g K 768 N 3072 10.45 (128, 64, 2, 2) against
cuBLAS 9.42 (+ the rna pass 3.7 in the step); out projection N 768 5.94 (64, 32, 4, 1) / 6.75; up-gate expand + SwiGLU N 4608 15.56
(128, 96, 2, 1) against cuBLAS + `swiglu_rows` 16.38; squeeze + `x + sigmoid(gl) y` K 1536 12.49 (64, 32, 4, 1) against cuBLAS + the row
pass 10.85. What bounds them (stamps): ~0.4-0.5 us a ring stage of 48 KB per CTA (~100 GB/s per SM, the same with KBX 4: not the box
count here), the first stage ~1 us after the dependency wait, and an epilogue of 0.5-0.6 us per 32-column chunk per warp even with the
TMEM loads, the transposes and the stores compiled out (one warp per scheduler: latency bound); more epilogue warps (NG 2 / 4) gain
0.2-0.8 us. The per-SM intake, not the MMA (NOMMA builds as slow), paces the K loop.

In the step (A/B min over 2-4 interleaved rounds, the measurer's ~2 us timer grid): the pair q | k | v | g with the fused rounding is
-2 us at Boltz-2 r108 / Protenix r102 / r182 (a conditioning per sample) but +2-4 us where the pair pass runs beside it with a shared
conditioning (r183) and on Chai-1 (no gating query: 77.9 / 77.5 -> 80.8 / 79.7) -- the 2-CTA clusters wait for SM pairs the side
stream's persistent pair pass holds; the out projection on the pair kernel moved cases +-2-4 us both ways (left on cuBLAS, bit 2 of
`MINIWORLD_TOKEN_B200_PAIR32`); the expand + SwiGLU (after the join, nothing beside it) helps every Boltz-2 row.

Taken (`token_inference`, B200 fp32, several samples, L <= 128 (the VMN path), d 768, M a multiple of 128; `MINIWORLD_TOKEN_B200_PAIR32`
bits, default 5):

| piece | effect (us, A/B min, cuBLAS path -> pair path) |
|---|---|
| Boltz-2 up-gate expand with silu(a) b u in the pair GEMM's epilogue (no [a \| b \| u] in HBM, no `swiglu_rows`); its input xt rounded to TF32 by `resgate_adaln_rows_r` (the existing rounding variant) | Boltz-2 r183 L128 88.5 / 89.1 -> 85.8 / 85.9; with the row below r108 96.3 / 97.3 -> 94.0 / 94.1 |
| gated blocks with a conditioning per sample and no q / k norm: q \| k \| v \| g on the pair GEMM (128, 128, 2, 1) with the TF32 rounding of q \| k \| v in its epilogue (no `rna_rows` pass) | r108 (above), Protenix r102 94.0 -> 89.9, r182 87.9 -> 81.8 |

Rejected: the out projection on the pair GEMM (+-2-4 us both ways), the fused squeeze + residual (12.5 against 10.9 us alone), the
q | k | v | g pair GEMM with a shared conditioning or without a gating query (+2-4 us, above), HR 64 x NH 128 for q | k | v | g in the step.

Accuracy (`measure.accuracy`, fp32 eval L 128 rows of the five classes, `_r3/out/p9/acc2.jsonl`): 0 RED, 8 GREEN / 12 YELLOW (the
fp32 TF32 paths as before); Boltz-2 r108 / r183 errors 4.6e-4 / 4.7e-4 against PyTorch's 4.5e-4 / 4.5e-4 (without the xt rounding the
pair expand had 8.8e-4 / 1.05e-3). Inference only: training paths unchanged.

Violations before (part 8's sweep) -> after (ratio = B200 cuda / H100 cuda of the 2.5.14 tokens, H100 >= 20 us; ladder 2 % band):

| class dtype mode | 1.5x misses | worst ratio | ladder |
|---|---|---|---|
| TokenDiTBoltz2 float32 eval | 2 -> 0 | 0.686 -> 0.657 (r108 L128 98.0 -> 94.1 us, H100 143.2; r183 89.9 -> 85.8, H100 131.9: 0.650) | 0 |
| TokenDiTProtenix master train | 1 -> 1 | 0.674 (r286 S 8 L128, not worked on in this part) | 0 |
| every other class / dtype / mode | 0 | <= 0.662 | 0 |

The fp32 eval L 128 rows of the other classes in the final A/B (`p9fin`: cuBLAS path / pair path) stay well inside the target (Chai-1
back on the cuBLAS path by the gate rule after that run; MiniWorld r103 86.2 / 90.0, 0.61; Protenix r179 79.7 / 84.0, 0.61 -- the
2-4 us moves of the timer grid); no other case reaches the new code (``p32`` is None off the VMN path).

Not done in this part: the VMN core's third stage (the 16 x 48 VMN build needs 234,496 B of shared memory for three stages, 2,048 over
the 232,448 limit; the smallest change found is to take the 16-column B boxes of g in / o out (8 KB per warpgroup) out of the staging
buffers -- a g / o pointer pair in the APB-shared core's VMN build -- not built), and the r286 S 8 training phase trace.

### Round 3, part 10 (2026-11-18): the r286 S 8 training phase trace -- `cuda:b200_sm100` 24

Base b200-2.5.x f2d941e47 (engine 2.5.14). Records `$W/b2511/agents/r3-token-dit/_r3/out/p10/` (accuracy `acc*.jsonl`, determinism
`*.check`), env A/B `_r3/out/p10x{1..5}/` (`_r3/m/xw.sh`, interleaved rounds, min), training sweep `_r3/out/p10m1/`, timeline
`$W/b2511/jobs/r3tdt_tl0.log` (`_r3/m/tlb.py`, one captured step, L2 flushed).

Phase trace of Protenix r286 master training S 8 L 128 (1024 rows; 49 kernels; the profiler's span carries a ~32 us gap between the
conditioning prep and the weight pack that the graph timing does not show): forward chain pack 12.6 -> cond GEMM 5.5 -> adaln_a 7.4 ->
q | k | v | g 7.2 -> attn_fwd2 6.9 -> gate_o 3.3 -> out 4.1 -> res_adaln_b 6.9 -> expand 6.9 -> swiglu 3.6 -> squeeze 5.4 -> res_c 3.9;
backward res_c_bwd 4.1 -> dh 6.0 -> swiglu_bwd 4.3 -> dxt 8.6 -> res_adaln_b_bwd 12.1 -> dog 6.7 -> gate_o_bwd 9.6 -> **bias_transpose
2.4** -> attn_dkv 14.3 || attn_dqb 20.2 (16 CTAs: dbias summed over the 8 samples) -> pair_bwd || qknorm_bwd 11.2 -> dxa 9.2 ->
adaln_a_bwd 10.5 -> dchat 12.2 -> dcg 6.1 -> cond_bwd 7.5 -> join -> the expand backward of the shared conditioning (torch sum over
the samples, 5.9). The step ended on side stream 2: the four q | k | v | g weight gradients (5.0 / 8.0 / 9.0 / 8.8 us, serial) and then
both to_scale weight gradients (5.5 + 5.4) -- 353.8 against 346.2 (main) and 352.3 (side stream 1's final_rest100).

Taken (training, every subclass; outputs and gradients the same values -- accuracy records identical to part 9's, determinism bitwise):

| change | effect (r286 master L128 S 8 / S 48, us, A/B min of 3 interleaved rounds) |
|---|---|
| attn_dkv -DBIASNT at every sample count (the bias is shared by the samples; the dkv tile does not depend on the sample: `sm100.backward` drops its A = 1 guard -- only the token DiT passes `bias_nt`) -- no bias_transpose on the chain | with the row below: 243.6 / 652.1 -> 239.7 / 627.6 (BIASNT alone, part-10 run 1: 243.5 -> 241.5 at S 8) |
| both to_scale weight gradients on side stream 0 right after dWo (dGg is complete after res_adaln_b_bwd; stream 0 idles there) instead of behind the q \| k \| v \| g gradients at the end of side stream 2 | (above) S 48 -13 us alone |

Built, measured, rejected (code in [rejected/token_dit_part10_shc_wgm.diff](rejected/token_dit_part10_shc_wgm.diff),
[rejected/token_dit_part10_goptr.diff](rejected/token_dit_part10_goptr.diff)):

| piece | result |
|---|---|
| shared conditioning as its [1, 1, L, cond] base, d cond summed over the samples in one row pass (`cond_bwd_shared100`, fixed sample order) -- no expand-backward reduction | S 8 241.5 -> 243.8, S 48 640.8 -> 650.1: the one-pass kernel (L blocks walking the samples) on the chain costs more than the reduction it removes |
| q \| k \| v \| g, expand a \| b, to_scale weight gradients as ONE GEMM each into a buffer whose row blocks are the leaves | not measurable: the leaves come back from the custom op as views of one buffer, which `torch.library` refuses (outputs may not alias); would need the buffers as extra op outputs split outside -- not built further after the to_scale move closed the miss |
| PDL: attn_fwd2 / attn_dkv -DPDLW=1 as programmatic dependents (APB v20's builds; the audit's item 1 for the cores) | S 8 241.4 -> 247.8, S 48 650.4 -> 669.6: rejected. The row passes of the chain live in the shared `token_dit_train_rows.cu` (kept byte-identical to origin for the other cards), so no early trigger / first-touch-before-wait was added there |
| the VMN fp32 core with the g / o B columns straight from / to global (`attn_inf_tf32 -DGOPTR`: XG 16 KB a warpgroup less, three stages at 48 wide; outputs bitwise equal) | fp32 eval L128 S5 A/B GOPTR off / on: r99 89.9 / 85.7, r100 87.7 / 90.1, r183 84.0 / 87.9, r108 98.1 / 98.1, r110 77.8 / 77.6, r184 69.4 / 69.4 -- level within the 2-4 us timer grid (the core is not stage bound, as part 8's merged-box test found): rejected |

Accuracy (`measure.accuracy`, master + fp32 training rows 101 / 143 / 263-265 / 286 L 128, S 1 / 8 / 48): 0 RED, the same error values
as before the change; fp32 eval L 128 VMN rows 0 RED. Determinism (`_r3/det.py`, the 14 training cases): output and every gradient
bitwise equal over two runs.

Violations, training (every requirement case of the token DiT classes, cuda, one `dev.sh -x` run per length; before = part 8's `p8m1`,
after = `p10m1`; triton / pytorch rungs from `p8m0`, unchanged):

| class dtype mode | 1.5x misses | worst ratio | ladder c > t | ladder t > p (run order, see part 8) |
|---|---|---|---|---|
| TokenDiTProtenix master train | 1 -> 0 (r286 S 8 L128 243.7 -> 238.5 us, H100 361.3: 0.674 -> 0.660) | 0.674 -> 0.660 | 0 | 1 -> 1 (r261 L128) |
| every other class / dtype / mode (train) | 0 | <= 0.625 | 0 | as before |

r286 master, all lengths (us, before -> after; H100 S 8): L128 S8 243.7 -> 238.5 (361.3), S48 651.2 -> 626.6; L256 S8 346.0 -> 337.7
(565.2); L384 S8 456.5 -> 446.4 (816.5); L512 S8 591.9 -> 577.5 (1110.2); L640 S8 753.3 -> 728.0 (1464.7); L768 S8 891.9 -> 860.1
(1824.3), S48 4001.2 -> 3856.2. Over the 168 training cases after / before geomean 0.996 (five one-sample cases +4-5 %, the known +-4 us
run-to-run spread of the one-sample steps; none leaves the target).

Not done in this part, with the reason: the audit's PDL item for the row passes (they are compiled from the shared
`token_dit_train_rows.cu`, byte-identical to origin for the other cards; the measured PDL on the two cores was +6 / +19 us); `tdt_epi`'s
DSMEM exchange as 128-B `st.async` (it serves only the one-sample <= 256-row dchat, no miss there); the `tdt_pair32` generic SwiGLU for
Protenix / Chai-1 fp32 640-row expand (no miss there).

### Outside sm100-only code

* `kernels/conditioned_transition/cuda/__init__.py` (part 9): `token_inference` gains B200-only branches (`p32` is None off the B200
  fp32 VMN path; the non-B200 calls and their order are those of origin) and `_prebuild_sm100` plans the two `tdt_pair32_sm100` builds.

* `kernels/augmented_attention/cuda/sm100/attn_inf.cu` (sm_100a, shared with AttentionPairBias's inference core): `-DFOLDARG` (default
  0; the item decomposition writes the head as `(cid / mt) % NHEAD` and a fold index that is 0 without the flag -- the existing builds
  compute what they did). `attn_inf_tf32.cu` (shared with AttentionPairBias fp32, `-DKPERM`): `-DQPAIR` (default 0; the stage layout
  and loads are those of before without it). Their cubin keys change with the source text.
* `kernels/conditioned_transition/cuda/token_dit_train_rows.cu` (the extension every card builds): byte-identical to origin/main again
  since part 2 (see "Round 3, part 2"); the B200 behaviour lives in `token_dit_train_rows_sm100.cu`, built for sm_100a only.
* `kernels/augmented_attention/cuda/sm100/attn_dqb.cu` / `attn_dqb_tf32.cu` (sm_100a sources, shared with AttentionPairBias and the
  bias-only DiT on the B200): two new build flags, `-DDQDET` / `-DDBPART`, default 0 -- the existing builds compute what they did (their
  cubin keys change with the source text).
* `kernels/augmented_attention/triton/main.py`: the backward body moved into `_backward(ctx, grad_output, head_major)`; the existing
  Function passes None (the card's default, as before); `TritonAugmentedAttentionHeadMajor` is new. `modules/attention.py`
  `_triton_core(..., head_major=False)`; only `AugmentedAttentionPairBias` on a B200 passes True.
* `kernels/augmented_attention/cuda/sm100/attn_inf_tf32.cu` (shared with AttentionPairBias fp32): `-DVMN` (part 8, default 0 -- the existing builds compute what they did; under it v MN-major, k / v merged TMA boxes, two stages at 48 wide); `GatedInferenceCore(..., vmn=False)` binds it; `prebuild_sm100.py` adds the VMN build.
* `token_dit_train.py`: `_pair_fused` no longer tests the card (H100 and B200 are the only cards that reach it; the H100 branch is
  unchanged), `_mm32`'s split and the gemv sums are behind `not _sm90`.

## Gaps (after round 3 part 3, 2026-11-17)

| class | cases | why | what it would take |
|---|---|---|---|
| TokenDiTMiniWorldBiasOnly | float32 eval L 128 (r104, r106) / L 256 (r104): cuda 0.91-0.97x triton (small-L ladder exception) | launch / pack bound at 5 x 128 rows: the TF32 three-kernel step + the one-launch pack (~35 us of fp32 weights) | static weights (no pack per replay) or the pack folded into the front kernel's prologue |
| TokenDiTChai1 / TokenDiTRoseTTAFold3 | triton 0.4-1.6 % behind pytorch on float32 training at one sample (inside the ladder band since round 3; was 2.5-4.9 %) | the Triton core against PyTorch's TF32 cuBLAS attention at A = 1 | a fp32 variant of the Triton augmented-attention core |
| model options, inference | L not a multiple of 128 (no requirement row) | `token_inference` does not pad | pad as the default-option runner does |
| every class | B200 `cuda` not 1.5x ahead of the H100 on 20 of 502 cases (none slower, worst 0.743; see "Round 3, part 3" for where) | five-sample fp32 inference at 640 rows (TF32 GEMMs bound by the fp32 operand intake), one-sample master training at L 128 (~70 launches), the one-sample TF32 core (12-24 work items), RoseTTAFold3's flat q / k LayerNorm pass | a persistent row + GEMM step kernel for one-sample training (the launches, not the GEMMs: a faster few-row GEMM measured no better than cuBLAS), TMA-multicast TF32 GEMMs for 640-row fp32 steps, a key split for the TF32 inference core, the flat q / k LayerNorm folded into a neighbouring pass |

No requirement case of these classes is refused on the B200 any more. YELLOW cells: the float32 TF32 paths (as on H100 / A100) and part of
the master training gradients; regression accuracy after the round (rows 99 / 101 / 103 / 108 / 110 / 143 / 179 / 260 / 263, L 128 /
384, both dtypes and modes): 0 RED.

## Round 3: AttentionPairBias against the H100 (2026-11-17, branch `b200r3-apb`)

Target (user, 2026-11-17): on every requirement case both cards' `cuda` serve, B200 median <= H100 median / 1.5 (ratio <= 0.667;
graph timing, L2 flushed, the speed tokens). `backend_versions["cuda:b200_sm100"]` of `AttentionPairBias` is 3. Measured on GPU 2
(`dev.sh -x`, the measurer, 30 iterations) against the H100 cuda speed tokens; records `$W/b2511/agents/r3-apb/_apb/out/spd_final.jsonl`
(table `spd_final.txt`), accuracy `acc_f1 / acc_f2 / acc_g1.jsonl`.

| dtype / mode | cases | not >= 1.5x: before -> after | slower than H100: before -> after | worst ratio after | geomean ratio after |
|---|---|---|---|---|---|
| master eval | 80 | 70 -> 38 | 25 -> 0 | 0.935 | 0.673 |
| master train | 54 | 53 -> 33 | 13 -> 2 (rows 147 / 222 L 256: 1.008 / 1.004, under one 2 us timer step) | 1.008 | 0.732 |
| float32 eval | 80 | 63 -> 19 | 0 -> 0 | 0.787 | 0.523 |
| float32 train | 54 | 45 -> 0 | 0 -> 0 | 0.607 | 0.500 |
| all | 268 | 231 -> 90 | 38 -> 2 | 1.008 (was 1.505) | |

What changed (all inside `cap == (10, 0)` code and `apb_rows_sm100.cu`, which only the sm_100 build compiles):

| change | effect |
|---|---|
| bf16 / master: fp32 master weights go straight into one pack launch at any pair width (`prep_dp`: W q / k / v / g, Wo, Wf, Wb, the gate constant); no cast launches | master eval 16 x 128 L 128 42.8 -> ~27 us |
| the pack and LayerNorm(single) of the first row in one launch (`prep_ln`) | -2 us per step (one launch) |
| the pack / pair pass, and in the backward the independent stages, on side streams (`MINIWORLD_APB_B200_STREAMS`, default on) | graph timing, on / off: master eval L 128 24.5 / 26.5 us, master train 16 x 512 L 128 91.9 / 100.3 us, L 512 307 / 326 us |
| a wide pair's backward (d_pair 256 / 384 / 512) in ONE launch (`pair_bias_bwd_wide1`: block b takes chunk b % NC; was one launch per 128 columns, each paying ~12 us of per-block Wf setup) | master train 16 x 512: L 128 128 -> 100 us, L 512 354 -> 313 us; same dpair bits |
| fp32: the pair pass on the TF32 tensor cores (`pair_bias32_tc`: TMA 32-column boxes, 3 stages, `mma.sync m16n8k8` tf32, row statistics about a pivot, slices added in a fixed order): training always (natural key order + row stats: the gather and its copy go), inference at L >= 256 where it is faster (`_tc32`) | alone at L 512: 16 x 128 86 -> 47 us, 16 x 512 244 -> 135 us, 24 x 128 148 -> 51 us (3.1-4.1 TB/s); fp32 eval 16 x 512 L 512 287 -> 176 us |
| fp32 training glue in single launches: `pack2_rna` (q / k / v rounded to TF32 + the zero second sample), `gate32`, `gate32_bwd` (dO rounded, D, dg, the zeros), the packs (`bo_pack_sm100`), dq / dk / dv written into their column blocks | (77 launches per step before, L 128) fp32 train 16 x 512 L 128 270 -> 160 us, L 512 970 -> 670 us |
| determinism: the backward cores build `-DDET=1` (`attn_dkv` / `attn_dkv_tf32`: dK / dV accumulated in block order; `_apb_defs` backward builds, `_b200_tf32_defs`), the bf16 backward's parameter-gradient sums are block partial rows reduced in a fixed order by `finalize` (as on H100; they were float atomics) | master training bitwise repeatable (15 cases x 3 runs); cost ~2 us per step. The base branch (0f2c5f268) was not repeatable in training either |

Accuracy after the round (cuda, both dtypes, every row; L 128-1024): 0 RED; the YELLOW cells are at their previous B200 token ratios
(float32: the TF32 paths, err / tol 1.2-2.9 eval, 1.6-2.5 train, the H100 tokens 1.1-2.2 / 2.6-4.1; master train 1.0-1.5).

What bounds the remaining 90 cases (measured on this B200: a 256 MB fill takes 69 us = 3.9 TB/s, so the bandwidth-bound passes -- the
pair kernels run at 3.1-4.2 TB/s -- have ~1.4x on the H100, not the nominal 2.4x; a captured launch has the same ~2 us floor on both cards):

| cases | why | what it would take |
|---|---|---|
| master eval, d_pair 128 (rows 41 / 146 / 147 / 148; 23 cases, L 128-768: 0.67-0.94) and wide pairs (rows 42-46; 15 cases: 0.67-0.71, row 46 L 2976 0.82) | 5 launches (pack + LN, pair pass, q / k / v / g GEMM, core, to_out) of 2-7 us each at L <= 256: launch-bound like the H100's step (H100 28-39 us, B200 24-35 us at d_pair 128); from L 512 the pair pass runs at the HBM rate and the core has 32 work items for 148 SMs; the wide rows sit at the bandwidth ratio (0.67-0.70) | the pair pass folded into the pack / LN launch (one launch less: ~2 us), the branch's key-split inference core (`attn_inf -DKSPLIT`, not in this tree; an estimate: 13 -> ~7 us at L 512 plus its combine pass); L 128 would still need the two cuBLAS GEMMs fused into the row / core kernels |
| master train, d_pair 128 (rows 146 / 147 / 148 / 222; 24 cases, every L: 0.70-1.01) and d_pair 256 / 384 (rows 223-226; 9 cases at L 128-512: 0.67-0.85) | about 20 launches per step; the three attention cores at B = 1 fill 16-32 SMs; the pair forward + backward are at the HBM rate | the branch's `-DKSPLIT` forward (gate in its epilogue, no O kept) and `-DQSPLIT` dK / dV cores, bias^T read inside `attn_dkv` (no transpose launch), the gate / qkv row passes fused (each ~2-3 us) |
| float32 eval, rows 44 / 46 (8 heads x d_pair 384; 17 cases: 0.67-0.79) and rows 146 / 148 at L 128 (0.68) | the fp32 FMA pair pass stays (the TF32 tensor-core pass is slower at 8 x 384: 125 vs 106 us at L 512) and is half the step | a 6-warp 384-wide tile for `pair_bias32_tc` with the Wf fragments in registers (8 heads: one mma per step) |
| float32 training at L >= 384: dQ (to_query.weight, and dx through it) differs run to run by ~2e-5 relative | `attn_dqb_tf32` reduce-adds one dQ partial per 64-key chunk in arrival order (three or more addends from L 192 on); dK / dV are repeatable now. Same on the base branch | the branch's `-DDQPART=1 -DDET=1` in `attn_dqb_tf32` (a slice per chunk, summed in order) |

### Round 3, continued (2026-11-17, `cuda:b200_sm100` 4)

Same target and measurement; records `$W/b2511/agents/r3-apb/_apb/out/spd_Fe.jsonl` (eval) / `spd_Ft.jsonl` (train), tables `spd_Fe.txt` /
`spd_Ft.txt`; accuracy `acc_A1.jsonl` (eval) / `acc_A2.jsonl` (train).

| dtype / mode | cases | not >= 1.5x: v3 -> v4 | slower than H100: v3 -> v4 | worst ratio v3 -> v4 |
|---|---|---|---|---|
| master eval | 80 | 38 -> 30 | 0 -> 0 | 0.935 -> 0.938 |
| master train | 54 | 33 -> 34 | 2 -> 0 | 1.008 -> 0.987 |
| float32 eval | 80 | 19 -> 3 | 0 -> 0 | 0.787 -> 0.681 |
| float32 train | 54 | 0 -> 0 | 0 -> 0 | 0.607 -> 0.629 |
| all | 268 | 90 -> 67 | 2 -> 0 | 1.008 -> 0.987 |

| change | effect |
|---|---|
| `planes_fix` (apb_rows_sm100.cu): `extra_pair_bias` and the L % 128 pad in ONE launch on the bias planes. fp32 added the extra planes through a sort (the key permutation, re-made inside every capture), a gather and four elementwise launches; a padded call (L 2976 -> 3072) copied the whole pair into a padded one. Now the pair pass runs at L (L % 32 == 0) and the launch writes the padded planes (pad keys `NEG`, pad query rows 0), adding the extra planes in the planes' units and key order | row 46 (8 x 384, extra bias) fp32 eval: L 256 122.7 -> 72.2 us, L 768 344 -> 275 us, L 2976 14.86 -> 3.49 ms (ratio 0.696 -> 0.163); master L 2976 11.02 -> 1.66 ms (0.817 -> 0.123) |
| fp32 inference on three streams: the packs + the pair pass, LN + the q / k / g GEMM, the v^T GEMM (was one stream; B = 1) | fp32 eval L 128-384: -4 us (8 x 384 L 128 49.6 -> 45.5 us, 12 x 128 L 128 45.4 -> 41.4 us) |
| the key-split bf16 inference core from L 512 (`sm100/attn_inf_ks.cu` = the branch's `attn_inf.cu`, built `-DKSPLIT=1`, + `ksplit_combine`; `sm100.apb_ksplit`, `MINIWORLD_APB_B200_KSPLIT` forces) | master eval step, unsplit -> split: 16 x 128 L 512 48.9 -> 44.8 us, L 768 67.7 -> 65.4, L 1024 96.1 -> 92.1; 12 x 128 L 768 69.4 -> 63.3; slower below L 512 (the combine launch: L 256 32.6 -> 34.6), 24 heads level |
| the pair backward leaves 8 SMs free (`sm_reserve`: the extension's SM count minus 8 while the partial rows are sized and the pass is launched; d_pair 128, a wide pair up to L 256). It filled every SM for half the backward, so `qkv_bwd` / dW / dxa / `ln_bwd` on the main stream sat queued until it ended | master train 12 x 128 L 256 / 512 / 768: 92.1 / 141.1 / 212.9 -> 86.0 / 134.8 / 209.0 us; a wide pair from L 512 loses (204.8 -> 207.2 us): off there |
| fp32 training dQ bitwise repeatable: `attn_dqb_tf32 -DDQPART=1 -DDET=1` (ported from the branch; a slice per 64-key chunk, `torch.sum` in chunk order; `sm100.backward_tf32(dq_parts=True)`, default off: the token DiT's build is unchanged) | `det2.py`: fp32 train L 128-768 x 3 runs bitwise equal (was ~2e-5 relative on dQ from L 384); fp32 train worst ratio 0.607 -> 0.629 |

Measured and rejected: an SM reserve on the FORWARD pair pass (so the q / k / v / g GEMM overlaps it): 8 / 16 / 32 / 56 SMs -> level, +2 %,
+10 %, +30 % on the pair-dominated steps. That sweep also settles what bounds the pair passes: their time scales with 1 / (SMs they
run on), so they are bound per SM (~28 GB/s of pair per SM, 4.1-4.4 TB/s over 148 SMs, with `cp.async.bulk`, `__ldg` and TMA alike),
not by a card-wide bandwidth. A kernel launched beside a pair pass on another stream therefore takes its SMs' share of that rate:
overlapping only helps where the pass is short of work for all SMs (L 128) or where the queued work is a serial tail (the backward).

Accuracy (cuda, 126 cases: every eval row at L 128 / 512 / 768 / 1024, every train row at L 128 / 384 / 768): 0 RED. master eval all GREEN
(the key-split lengths included); float32 eval YELLOW at 1.0-2.8 x tol (the TF32 paths, as before; H100 tokens 1.1-2.2); train YELLOW at
1.0-2.1 (master 1.0-1.5, float32 1.5-2.1; H100 float32 tokens 2.6-4.1). Determinism: `det2.py` rows 41 / 45 / 146 / 147 / 223 / 225, both
dtypes, eval + train, L 256 / 512 / 768: every output and gradient bitwise equal over three runs; B = 2 against two B = 1 calls 0.0.

What is still above 0.667 (67 cases) and why:

| cases | ratio | why (timeline of one captured step, `_apb/tl.py`) | what it would take |
|---|---|---|---|
| master eval d_pair 128 (rows 41 / 146 / 147 / 148), L 128-640: 20 | 0.67-0.94 | L 128: start 4 us, pack + LN 3.4, [pair 4.3 beside GEMM 6.4], core 6.3, to_out 3.0 = 24.5 us against the H100's 28.4 (five launches on both cards; the target is 18.9). L 512: pack 3.6, pair 18.0 (148 SMs busy: the GEMM queues behind it, +2.6), core 12.5 (8 with the key split + 3 combine), to_out 3.4 | one kernel for LN + the q / k / v / g GEMM reading the fp32 masters (-5 us at L <= 256), Wf made inside the pair pass (no wait for the pack: -3.5 us), to_out inside the core's combine pass; all three together reach ~0.70 at L 128, not 0.667 |
| master eval wide pairs (row 45: L 384-2048, rows 44 / 46 / 47 at L 128): 10 | 0.667-0.694 | the pair pass is 70-85 % of the step and runs at the per-SM rate above; the H100 runs the same kernel on 132 SMs | a pair pass with a higher per-SM rate (the bf16 pass holds 4 warps per SM at d_pair 384: a 2-stage ring with two blocks per SM, or tcgen05 with the row statistics in a side pass) |
| float32 eval row 44 (8 x 384) L 512-768: 3 | 0.673-0.681 | the fp32 FMA pair pass is 80 % (215 of 268 us at L 768, 4.2 TB/s); the two GEMMs queue behind it | as above for fp32 (the TF32 tensor-core pass is slower at 8 heads: a 384-wide tile for it) |
| master train d_pair 128 (rows 146 / 147 / 148 / 222), every L: 24 | 0.69-0.99 | 19 launches; L 512: forward 47 us, then dO GEMM / transpose / gate backward 14 us before the cores start, dqb 14 + dkv 16, the pair backward 46 (per-SM rate), reduce + finalize 7.5 | the branch's `attn_fwd2 -DKSPLIT` (gate folded into the combine) and `attn_dkv -DQSPLIT` (their maps differ from this tree's: `attn_fwd2` / `attn_dkv` take the batch as a third map coordinate there), the bias transpose read inside `attn_dkv`, gate backward + dO GEMM fused |
| master train d_pair 256 / 384 (rows 223-226), L 128-512: 10 | 0.668-0.81 | as above; from L 384 the pair forward + backward dominate (0.67-0.69) | as above + the pair backward's per-SM rate |

### Round 3, part 3 (2026-11-17, `cuda:b200_sm100` 5)

Records `$W/b2511/agents/r3-apb/_apb/out/spd_Ge.jsonl` (master eval) / `spd_Gt.jsonl` (master train), tables `spd_Ge.txt` / `spd_Gt.txt`;
accuracy `acc_w1` (wide eval), `acc_w2` (wide train), `acc_u1` (d_pair 128 eval on the unified core), `acc_td` (token DiT on it).
float32 was not touched and not re-measured (its v4 numbers stand).

| dtype / mode | cases | not >= 1.5x: v4 -> v5 | slower than H100 | worst ratio v4 -> v5 |
|---|---|---|---|---|
| master eval | 80 | 30 -> 28 | 0 -> 0 | 0.938 -> 0.938 |
| master train | 48 measured (of 54) | 34 -> 32 | 0 -> 0 | 0.987 -> 0.987 |

What bounds a pair pass (measured with standalone probes, `_apb/mb/ld.cu` / `pb.cu`, L2 flushed): a load-only ring of 32 KB tiles moves
6.1-6.6 TB/s of a 1 GB pair and the same on 74 SMs (5.0 TB/s on 37), with `cp.async.bulk` and with TMA boxes alike -- the card's rate, not
an SM's. The shared mma.sync pass is below it because an SM's rate is (bytes it has in flight) / (load latency + the time a landed tile
waits in its stage): 128-144 KB in flight and a tile's whole compute as the wait. So the lever is in-flight bytes and a short wait.

| change | effect |
|---|---|
| a wide bf16 pair's pair -> bias pass on tcgen05 (`apb/apb_pair_fwd_sm100.cuh`, cubin `sm100/pair_fwd.cu`, `sm100.apb_pair_fwd`): per 16 pair columns two MMAs issued by one thread -- X X^T (its diagonal is every row's sum of squares) and X [Wf ; 1]^T (the projections and the row sum) --, 192 KB of pair in flight per SM (TMA boxes of 64 columns, 2-4 stages; d_pair 512 in two K halves), a stage reloaded by a loader warp as soon as its MMAs commit, two TMEM accumulators so a tile's epilogue runs under the next tile's MMAs, Wf formed in the kernel from to_bias.weight / ln_pair.weight (inference: the pass starts with the step, no wait for the pack). Routed where it wins (`_pair_tc`): d_pair 384 from L 384, 256 / 512 from L 512; inference and the training forward (row stats + unmasked bias written by the same pass) | master eval 16 x 384 (row 45): L 384 61.3 -> 57.7 us, L 512 81.9 -> 72.1, L 1024 235.5 -> 174.3, L 2048 829 -> 572 (ratio 0.668 -> 0.460); 8 x 384 L 2048 738 -> 551; 16 x 512 L 768 161.7 -> 141.8; master train 16 x 256 L 512 205.7 -> 202.8, 8 x 384 L 512 202.8 us (ratio 0.548) |
| the key-split inference core is the `-DKSPLIT=1` build of the shared `sm100/attn_inf.cu` (the superset source; `attn_inf_ks.cu` is gone). Without `-DKSPLIT` / `-DKMASK` it is the previous kernel statement for statement (the token DiT's builds: master eval GREEN on TokenDiTProtenix / TokenDiTMiniWorld, 26 cases) | one source |

Lessons from the probes (both cost a factor before they were found): two threads of ONE warp on different loops take turns (an MMA
thread and a loader thread in one warp: 89 us against 74 with the loader in its own warp); global reads that are used one after
another after an L2 flush are each a round trip to memory (~2 us): issue them all first (Wf's rows: 6.9 -> 3.4 us of start-up).

Measured and rejected:
- the tcgen05 pass at d_pair 128 (32 KB tiles, 4 stages; 6 stages are slower): level at L 1024 (92.0 vs 94.8 us step), slower below
  (L 128 24.5 -> 29.2 us, L 512 44.9 -> 49.6): one block per SM and ~3.4 us of start-up (TMEM, Wf) against two blocks per SM; a wide
  pair below L 384 loses the same way (16 x 384 L 128 28.5 -> 31.2 us).
- LayerNorm(single) + the q | k | v | g GEMM in one launch from the fp32 masters (`ln_qkvg`, three variants: plain rows, one bulk
  copy, TMA + 128-B swizzle): 15-30 us per launch at L 128 against 3.4 + 6.4 us for pack + LN -> cuBLAS. A block's reads (rows,
  96 KB of weights, the LN vectors) arrive as successive round trips (~8 us per block), and its 150 KB of shared memory cannot sit
  beside a pair-pass block on one SM, so the blocks that find no free SM start when the pair pass ends. Removed.

Accuracy: wide eval 15 / 15 GREEN (rows 42 / 44-47, L 384 / 512 / 768), d_pair 128 eval 12 / 12 GREEN, wide train (rows 223-226, L 128 /
384 / 512) 0 RED, 10 YELLOW at exactly their B200 token ratios (1.06-1.46). Determinism (`det2.py`, rows 225 / 226 master train L 384 /
512, three runs): bitwise equal. `_prebuild --arch sm100 --list` has the four `pair_fwd` builds.

Still above 0.667 (60 of the 128 master cases measured; fp32 eval row 44 L 512-768, 3 cases at 0.673-0.681, unchanged):

| cases | ratio | why |
|---|---|---|
| master eval d_pair 128, L 128-640: 20 | 0.675-0.938 | launch-bound as before (five launches, 24.5 us at L 128 against a target of 18.9); neither new kernel helps there (above) |
| master eval wide pairs at L 128 / 256 (rows 42-46): 8 | 0.670-0.706 | inside one 2 us timer step of the target (v4 measured the same code at 0.64-0.69: 26.6 vs 28.5 us on row 42 L 128); the old pass serves them, 5 launches |
| master train d_pair 128, every L: 24 | 0.687-0.987 | unchanged: 19 launches, the pair forward / backward at d_pair 128 |
| master train wide, L 128-384 (rows 223-226): 8 | 0.689-0.807 | launch-bound at L 128 / 256; the pair BACKWARD (still the mma.sync pass) is the larger half from L 384 |

What it would take: a tcgen05 pair backward on the same ring (dWf = dbias^T X as an MMA, dpair = Wf^T dbias with the LN backward in
the epilogue) for the training rows; for the launch-bound steps the start-up of the tcgen05 pass must drop under ~1.5 us (Wf by one
bulk copy, two blocks per SM sharing TMEM) before it can replace the pack -> pair chain; a TF32 (`kind::tf32`) build of the same pass
for fp32 row 44.

### Round 3, part 4 (2026-11-17, `cuda:b200_sm100` 6 and 7)

Records `$W/b2511/agents/r3-apb/_apb/out/`: every case `spd_Ie` (eval) / `spd_It` (train), tables `spd_Ie.txt` / `spd_It.txt`; the cases
that measured beside another session's process on GPU 2 re-measured alone twice (`spd_J*`); A / B sets `spd_V*` / `spd_W*` (pair pass),
`spd_U*` (pair backward, reserve); accuracy `acc_X1` (d_pair 128 training), `acc_X2` / `acc_X3` (the re-routed wide cases).

| dtype / mode | cases | not >= 1.5x: start of part 4 -> v6 -> v7 | slower than H100 | worst ratio v7 |
|---|---|---|---|---|
| master eval | 80 | 28 -> 23 -> 20 | 0 | 0.883 (row 146 L 256) |
| master train | 54 | 32 -> 33 -> 29 | 0 | 0.951 (row 148 L 128) |
| float32 eval | 80 | 3 -> 3 -> 2 (row 44 L 512 / 768: 0.669 / 0.667) | 0 | 0.669 |
| float32 train | 54 | 0 -> 0 -> 0 | 0 | 0.629 |
| all | 268 | 63 -> 59 -> 51 | 0 | |

Ladder against the B200 Triton tokens: 268 / 268 cuda faster than triton (worst cuda / triton 0.629), cuda vs triton geomean 5.44x,
vs pytorch 5.79x.

What changed:

| change | effect |
|---|---|
| v6: inference packs through `param_forms(capture_reads_eager=True)` (bf16 + fp32: no pack launch per replay), the d_pair 128 inference pair pass leaves 8 SMs at L 384-640 (the q / k / v / g GEMM runs under it), the bf16 inference pair pass starts when LayerNorm(single) has ended, the training backward issues dxa before dW q / k / v / g | master eval 16 x 128 L 128 / 256 27.1 / 34.7 -> 24.4 / 32.6 us, 12 x 128 L 512 47.0 -> 43.1; master train 16 x 128 L 512 143.1 -> 139.1 |
| v7: the d_pair 128 training pair backward on tcgen05 (`sm100/pair_bwd.cu`, `sm100.apb_pair_bwd`): D = a Wf and dWf^T += X^T a as MMAs (a = rstd dbias in bf16, the operand of both; X^T and a read MN-major), Wf . x and the row sums from a third MMA, the row sums of squares in the epilogue (no Gram MMA), phase A of tile n + 1 before phase B of tile n (two a / D buffers), a store warp that hands the stage back, Wf read before the first tiles' loads; partial rows into the same accumulator columns as `pair_bias_bwd` | alone (graph replay, L2 flushed): 16 heads L 512 43.0 -> 38.9 us, 24 heads 51.2 -> 49.2, 12 heads L 128 12.3 -> 12.3; in the step it is bound by the card's memory rate (a block moves ~5.3 TB/s in steady state) and shares it with dkv / the GEMMs, so it gives up SMs to them: 32 at <= 16 heads and L <= 512, else 8 (`_pbtc_reserve`): master train 12 x 128 L 384 / 512 / 768 102.2 / 133.0 / 210.7 -> 98.0 / 130.8 / 196.6, 16 x 128 (row 222) 112.6 / 139.3 / 229.3 -> 110.2 / 137.0 / 216.9, 24 x 128 L 384 122.7 -> 118.7 |
| v7: the tcgen05 pair -> bias pass reads Wf (to_bias.weight, ln_pair.weight) before it issues its first tiles (behind 148 SMs' tile loads those reads took ~5 us of start-up); re-measured thresholds (`_pair_tc`, now per head count): d_pair 512 at every L, d_pair 384 from L 256 at 16 heads (384 at 8), d_pair 256 from L 256 in training (768 in inference). The d_pair 256 threshold A / B (task item 4): training L 128 / 256 / 384 79.8 / 108.4 / 151.3 -> 81.8 / 106.3 / 147.5 us; inference L 512 / 768 / 1024 59.4 / 95.4 / 141.3 -> 59.2 / 94.2 / 134.9 | master eval 16 x 384 L 256 42.8 -> 40.8 us (ratio 0.680 -> 0.648), 16 x 512 L 128 / 256 30.8 / 46.9 -> 30.5 / 42.8; master train 16 x 256 L 256 / 384 108.4 / 151.3 -> 104.4 / 147.1 (0.668 / 0.660), 16 x 384 L 256 139.3 -> 122.7 |

Accuracy (cuda): d_pair 128 training rows 146 / 147 / 148 / 222 at L 128 / 384 / 768: 0 RED, 2 GREEN, 10 YELLOW at 1.05-1.41 (the B200
tokens 1.00-1.49; one cell moved: row 148 L 128 1.00 -> 1.25 on grad:pair, the H100 Triton token there is 1.29); the re-routed wide
cases (rows 42 / 45 / 47 eval L 128-512: 9 / 9 GREEN; rows 223 / 224 / 226 / 47 train L 256 / 384: 1 GREEN, 7 YELLOW, each at its B200
token). Determinism (`det2.py`): d_pair 128 training rows at L 256 / 512, rows 223 / 226 / 47 training L 256, rows 45 / 47 eval L 256:
every output and gradient bitwise equal over three runs. The new pair backward against the mma.sync pass and an fp64 reference
(`_apb/pbchk.py`, 8-24 heads, L 128-768): dpair 2.36e-3 vs 2.35e-3 relative, dWf 1.70e-3 vs 1.70e-3, run to run bitwise equal.

Measured and rejected:
- the gate in `attn_fwd2`'s epilogue (`-DGATE`: og = sigmoid(g) O written by the core, no `gate_rows` launch): the core grew 11.2 -> 14.8
  us (each thread's g loads and og stores are 32-96 B per row, uncoalesced) against the 2.5 us launch it saved. Removed.
- the tcgen05 pair backward with a Gram MMA for the row statistics (as the forward pass): 0.5 us of tensor time per tile on the critical
  path; a stage handed back one tile late by the epilogue (instead of a store warp): the stages sat out the epilogue's queue (2 us per
  tile, load-latency bound). Both replaced in the kept kernel.
- the tcgen05 pair -> bias pass at d_pair 128 (eval), even with the start-up fix: 16 x 128 L 128 / 512 / 768 24.5 / 44.8 / 65.5 -> 26.4 /
  46.9 / 68.1 us (one block per SM, the shared pass runs two).
- a larger SM reserve for the pair backward (56): 16 x 128 L 512 / 768 139.2 / 223.2 -> 145.2 / 244.9 us.

Still above 0.667 (51):

| cases | ratio | why (timelines `_apb/tl.py`) |
|---|---|---|
| master eval d_pair 128 (rows 41 / 146 / 147 / 148), L 128-640: 19 | 0.675-0.883 | five launches; L 128: LayerNorm 2.2, [pair 5.3 beside the q / k / v / g GEMM 7.3], core 6.6, to_out 3.3 = 24.4 us against a target of 19.4 (the H100's step has the same five launches). From L 512 the pair pass (19.6 us at L 512, 67 MB: 3.4 TB/s) is the step; the d_pair 128 tcgen05 pass loses (above) |
| master eval 8 x 384 L 128 (row 44): 1 | 0.684 | launch-bound as above (28.4 us against 27.7) |
| master train d_pair 128 (rows 146 / 147 / 148 / 222): 23 | 0.685-0.951 | ~19 launches. L 512, 12 heads (130.8 us, target 117.7): forward 44.6 (pair 19.0 with the q / k / v / g GEMM starved behind it, core 11.2), dO / dWo GEMMs, the bias transpose and the gate backward 14, attn_dqb 14.2, then the pair backward ~44 us (151 MB at the card's rate, sharing it with dkv and the GEMMs) and acc_reduce + finalize 8. At L 128 / 256 the step is launch latency (69.5 us for 19 launches against the H100's 75.9) |
| master train d_pair 256 / 384 (rows 223-226) L 128 / 256: 6 | 0.668-0.775 | launch-bound at L 128; L 256 at the bandwidth ratio (0.668-0.675) |
| float32 eval row 44 (8 x 384) L 512 / 768: 2 | 0.667-0.669 | at the threshold (one timer step); the fp32 FMA pair pass is 80 % of the step at the card's rate |

What it would take: for training, fewer bytes rather than faster kernels -- the pair is read twice and dpair written once (151 MB at
d_pair 128, L 512), so the remaining lever is the launch tail (one acc_reduce + finalize launch, the bias transpose read inside attn_dkv,
the gate backward folded into the dO GEMM's epilogue: ~10 us at every L, enough for L >= 384 but not for L 128 / 256); for small-L
inference a single-launch step (LayerNorm + q / k / v / g + core + to_out with roles; the pair pass beside it).

### Round 3, part 5 (2026-11-17, `cuda:b200_sm100` 8)

Records `$W/b2511/agents/r3-apb/_apb/out/`: master training `spd_Lu{128,384,512,640,768}` + `spd_Lt2` (one length per measurer run,
BRIEF4: small L never behind L >= 768), master eval `spd_Ev*`, float32 row 44 `spd_Ef*`; A / B (interleaved graph replays, `_apb/ab.py`)
`j8_*`, `j10`, `j14_*`; accuracy `acc_K1`-`acc_K4`. Measured on GPU 3 (GPUs 0-3 lent; the part 4 numbers were GPU 1).

What changed (training, bf16 / master, every pair width):

| change | effect (A / B, part 4 forms -> part 5) |
|---|---|
| `finalize_red` (`apb_rows_sm100.cu`): acc_reduce + finalize in one launch (copies, row-kernel columns and the dWf / dWb / dln_pair columns as block ranges, fixed order: deterministic) | 12 heads L 512 129.0 -> 127.0 us; at 24 heads its 24 column sums per thread lost 2-10 us, so <= 16 heads only |
| `gate_bwd2`: the gate backward with a warp per row at W = 384 (every lane busy) | 12 heads L 512 131.1 -> 129.0; 24 heads no gain: 12 heads only |
| bias^T made by the forward beside its core (`glue.cu` bias_transpose64_bf16: 64 x 64 tiles, 16-byte loads / stores, a 64-block persistent grid that leaves the core its SMs), saved for attn_dkv: the backward has no transpose launch | 12 / 16 heads L 512 -2 us; 24 heads L 128 -2 us, L 768 +10 us: <= 16 heads or L <= 256 |
| pair backward (`pair_bwd.cu`) with a = a_hi + a_lo (`PB_SPLIT`, two bf16 parts, the low one at head slots 16.. / 32..; D = a Wf and dWf see a at ~fp32 precision) | row 148 L 128 grad:pair 1.25 -> 1.00 x tol (the task's accuracy cell: the a = rstd dbias rounding to bf16 was the cause); also row 146 L 768 1.49 -> 1.41. No measurable cost at <= 16 heads; 24 heads L 768 +8 us (K of D = a Wf doubles), so <= 16 heads only |
| harness: `_apb/ab.py` kept each graph's step alive (graph A's backward reads the measurer step's upstream gradient, a default-pool tensor; building B's step freed it and the empty_cache at B's capture unmapped it: the "illegal memory access" of part 4's last A / B runs was this, not the engine -- found by logging every kernel argument's block in the allocator snapshot, `_apb/ab4.py`) | -- |

All together (A / B, the same process): d_pair 128 at <= 16 heads 2-6 % faster at every L (row 146 L 128 69.6 -> 65.6, L 512 133.1 ->
125.0, row 222 L 640 174.1 -> 165.9 us), the wide pairs 1.5-4.5 % (row 223 L 384 147.6 -> 143.4, row 47 L 128 96.3 -> 92.2).

Against the H100 cuda tokens (one length per run):

| dtype / mode | cases | above 0.667: part 4 -> part 5 | worst ratio part 5 |
|---|---|---|---|
| master train | 54 | 29 -> 26 (d_pair 128: 23 -> 22, 0.656-0.934; d_pair 256 / 384: 6 -> 4, all at L 128 (0.708-0.792): rows 223 / 224 L 256 now 0.654 / 0.650). Rows 41-45 at L >= 1024 not re-measured (they passed in part 4 and run the same tail) | 0.934 (row 148 L 128) |
| master eval | 80 | 20 -> 21 (unchanged code; GPU 3 reads 1-2 us slower at L 128 than GPU 1 did: rows 42 / 45 / 47 L 128 0.669-0.705) | 0.884 (row 146 L 256) |
| float32 eval | 80 | 2 -> 1-2 (row 44 L 512 0.668 / 0.669, L 768 0.665 / 0.668: one timer step) | 0.669 |
| float32 train | 54 | 0 -> 0 | |

Accuracy (cuda, master training, rows 146 / 147 / 148 / 222 at L 128 / 384 / 768 and the wide rows 47 / 223 / 224 / 226 at L 256 / 384):
0 RED; every cell at or below its B200 token (row 148 L 128 1.00, was 1.25 with the part 4 pair backward). Determinism (`det2.py`, rows
146 / 148 / 222 at L 256 / 512, rows 47 / 147 / 223 at L 256): every output and gradient bitwise equal over three runs.

Not built in part 5 (the coordinator stopped the round for the B200 access change): the dK / dV core reading the bias as stored (the
forward-made bias^T removes the backward's transpose launch instead, at <= 16 heads), the single fused small-L inference kernel, fewer
training launches at L 128 / 256 (still ~17 launches, 65-76 us against 76-81 us on the H100), the TF32 tcgen05 pair pass for row 44.

### Round 3, part 6 (2026-11-17, `cuda:b200_sm100` 9)

Records `$W/b2511/agents/r3-apb/_apb/out/`: speed `spd_P6e{128,256,384,512,640}` (master eval), `spd_P6t{128,...,768}` (master train),
`spd_P6f*` / `spd_P6f2_*` (float32 eval), one length per measurer run; accuracy `acc_P6a1`-`acc_P6a4`; A / B (`_apb/p6_chk.py`: the same
module and inputs, the switch off / on, graph replays interleaved, L2 flushed) and timelines (`_apb/tl.py`) in `$W/b2511/jobs/apb_j2*.log`.
GPU 3. Base: `b200-2.5.x` 3cebc8cb3 merged first.

The four items of parts 4 / 5, built:

| item | built | result |
|---|---|---|
| 1. small-L inference in <= 3 launches | `inf_front` (apb_rows_sm100.cu): LayerNorm(single) + the q / k / v / g GEMM + the pair pass in ONE launch with block roles (GEMM blocks first in the grid: 32 rows x 64 columns, the tile's weight rows by bulk copies, ln_rows_k's row math into shared memory, mma.sync; then pair_bias_fwd_k's pass over the remaining blocks), then the core launched as a programmatic dependent (`attn_inf -DPDLW=1`: its barriers / TMEM set up under the front's tail; the front triggers at its start), then to_out (cuBLAS). Launches 5 -> 3 (H100: 5). d_pair 128, L <= 256 | taken. A / B (step): L 128 26.6 -> 22.5 (front) -> 20.4 us (+ PDL) at 12 / 16 heads, 24 heads 26.6 -> 20.4, d_single 512 26.6 -> 22.5; L 256 34.6 -> 28.6-30.7. L 384 level or slower (16 x 512: 41.0 -> 45.1): off from L 384. Same bits as the unfused step at L 128 (the GEMM sums in the same order), one bf16 step at L 256. Timeline L 128: front 6.8, core 6.6, to_out 3.1 us |
| 2. small-L training launches | `trn_front` (pack + LN + GEMM + pair pass in one launch: the pack blocks first, GEMM / pair blocks waiting on a counter), `out_gate` (gate + to_out in one row GEMM), `dog_gate` (the dO GEMM with the gate backward in its epilogue): 18 -> 15 launches | rejected, removed (`scratchpad/r3apb6_rejected_*`). Each hand GEMM was slower than cuBLAS + the row pass it replaced: out_gate 8.7-14.8 us vs gate 2.5 + to_out 2.9, dog_gate 7.2-18.2 vs 3.5 + 2.8, trn_front 13.8-20.4 vs prep_ln 3.7 -> [pair 4.2 beside GEMM 6.5]; master train L 128 65.5 -> 69.6-81.9 us. What bounds them: a block's operand bytes arrive at a per-SM rate (16-B cp.async: ~14 GB/s per block for 72 KB; one bulk copy per row: the TMA unit serialises the 384 row copies of a Wo block, 18 us); cuBLAS spreads the same GEMMs over 96+ CTAs with tiny N tiles |
| 3. TF32 tcgen05 pair pass (fp32) | `apb_pair_fwd32_sm100.cuh` / `sm100/pair_fwd32.cu` (`sm100.apb_pair_fwd32`): apb_pair_fwd_sm100.cuh's schedule on fp32 tiles -- per 32 columns X X^T (row sums of squares) and X [Wf; 1]^T as kind::tf32 MMAs, 32 KB units in 5-6 stages, a loader warp, two TMEM accumulators; Wf rounded to TF32 to nearest, x read truncated (its scale cancels in LN to first order); keys written in the TF32 core's key_perm order | taken: 8 x 384 from L 256, every other layout from L 512. A / B: 8 x 384 L 256 / 384 / 512 / 768 64.5 / 98.3 / 145.4 / 264.2 -> 55.3 / 83.9 / 120.8 / 206.8 us; L 512: 16 x 128 84.0 -> 73.7, 16 x 384 167.9 -> 118.9, 16 x 512 172.0 -> 143.4. Row 44 L 512 / 768: ratio 0.669 / 0.665 -> 0.555 / 0.514. Accuracy YELLOW 1.23-1.68 x tol (tokens 1.34-1.79) |
| 4. dK / dV with the bias as stored | `attn_dkv -DBIASNT=1`: TMA box of 128 keys x 64 queries of the bias as stored (no swizzle), thread r reads its key column with 16-bit loads (a warp 64 contiguous bytes); `apb_backward_stages(bias_nt=True)` | taken where the forward makes no bias^T (24 heads, L > 256): no transpose launch. A / B row 147 L 384 / 512 / 768 120.5 / 157.7 / 270.3 -> 118.8 / 153.6 / 258.1 us, same bits |

Also: the merge of `b200-2.5.x` (token DiT part 3) made `GatedInferenceCore` read `self.device_index` and route one-sample fp32 calls
at L > 128 to an `attn_inf_tf32 -DQPAIR` build without `-DKPERM`; AttentionPairBias's fp32 core (built through `__new__`) raised
AttributeError on every fp32 inference call at the base tip. `_b200_tf32_core` now sets `device_index` and `tf32_qpair = False`
(the class honours the flag; its default keeps the token DiT's behaviour).

Against the H100 cuda tokens (one length per run; part 5 -> part 6):

| dtype / mode | cases | above 0.667 | worst ratio | d_pair 128 at L 128 / 256 |
|---|---|---|---|---|
| master eval | 80 | 21 -> 21 (L 128: 7 -> 6; r42 L 256 0.663 -> 0.677, the same code, one timer step) | 0.884 -> 0.804 | 0.80-0.86 / 0.81-0.88 -> 0.70-0.76 / 0.78-0.80 |
| master train | 54 | 26 -> 26 | 0.934 -> 0.935 | unchanged (0.86-0.94 / 0.82-0.87); row 147 L 768 0.685 (BIASNT) |
| float32 eval | 80 | 1-2 -> 0 | 0.669 -> < 0.667 | |
| float32 train | 54 | 0 -> 0 | | |

Accuracy (engine standard): master eval rows 41 / 146 / 147 / 148 L 128 / 256: 8 / 8 GREEN; master train rows 146 / 147 / 148 / 222 at
L 128 / 384 / 768: 2 GREEN, 10 YELLOW 1.00-1.43 x tol, each at its B200 token (row 146 L 768 1.49 -> 1.41); float32 eval (tcgen05 pass):
YELLOW 1.23-1.68 x tol, 0 RED (`acc_P6a3` / `acc_P6a4`). Determinism (`det2.py`): master train rows 146 / 147 / 222 L 256 / 512, master
eval rows 146 / 148 L 128, float32 eval rows 44 / 45 / 147 L 512: every output and gradient bitwise equal over three runs.

What remains above 0.667 and why:

| cases | ratio | why (timelines) |
|---|---|---|
| master eval d_pair 128 L 128 / 256 (10) | 0.70-0.80 | 3 launches now: start 3.2 + front 6.8 + core 6.6 + to_out 3.1 us at L 128 (target 19.4); the front is its GEMM role (6.0 us alone, the pair role 4.9): latency of the operand loads per block; the core has 16 work items for 148 SMs |
| master eval d_pair 128 L 384-640 (10) | 0.68-0.79 | the shared pair pass at the per-SM rate (part 4); the front loses there (fewer pair blocks beside the GEMM blocks) |
| master eval wide L 128 / 256 (r42 / 43) | 0.67-0.71 | at the line, one timer step |
| master train d_pair 128 (22) | 0.67-0.94 | ~18 launches; L 128: forward to the core 14.7 us, cores 12 us (dqb beside dkv), then qkv_bwd -> dxa -> ln_bwd -> finalize_red (~19 us of chained launches, finalize_red 5.5 us). The fused training forms built here lost (item 2) |
| master train d_pair 256 / 384 L 128 (4) | 0.71-0.79 | as above |

#### AttentionPairBias: H100 techniques on the B200 (audit, 2026-11-17)

| H100 technique (`docs/h100/attention_pair_bias.md`) | B200 |
|---|---|
| one `prep` launch: casts of fp32 masters folded in, q / k / v / g + bias, Wo, Wf packed | used (`prep_dp`, any d_pair; `prep_ln` adds LayerNorm(single) to the same launch) |
| packs cached per parameter version, capture-scoped | used, further: `param_forms(capture_reads_eager)`, no pack launch per replay (inference) |
| q pre-scaled by log2 e / sqrt(dh), gate constant in the packed bias, Wf in log2 units | used |
| `ln_rows` writes the residual seed, `to_out` accumulates onto it (beta = 1) | used |
| gated inference core (sigmoid(g) o in the epilogue) | used (`attn_inf -DQPAIR`; from L 512 the `-DKSPLIT` build + `ksplit_combine`, one launch more, measured faster) |
| training core with the gate in its epilogue (`attn_fwd_o_og`) | not applicable (measured): `attn_fwd2 -DGATE` core 11.2 -> 14.8 us against the 2.5 us `gate_rows` launch it removes (uncoalesced per-row g / og) |
| `apb_dqb` (dQ + full dbias in one kernel) + `attn_dkv` | used (`attn_dqb -DDQPART -DDET`, `attn_dkv -DDET`) |
| `attn_dkv` reading the bias as stored | used (part 6: `attn_dkv -DBIASNT=1`) where the forward makes no bias^T (24 heads, L > 256); at <= 16 heads or L <= 256 the forward-made bias^T (part 5) stays |
| row passes `gate_bwd`, `qkv_bwd`, `ln_bwd` | used (the same sources) |
| per-block partial rows, fixed-order `acc_reduce` + `finalize` (deterministic) | used; the tcgen05 pair backward writes its partial rows into the same columns |
| `pair_bias_fwd` (LN folded into the projection, TMA ring) | used at d_pair 128; wider pairs on the tcgen05 pass where it wins |
| `pair_bias_bwd` (LN recomputed) | replaced at d_pair 128 by the tcgen05 pair backward (training) |
| `pair_bias_bwd_wide` per 128-column chunk (one launch each) | used, in one launch (`pair_bias_bwd_wide1`) |
| TF32 path: fp32 pair kernels, TF32 core | used; inference: the pair pass on tcgen05 kind::tf32 (`pair_fwd32`, part 6) from L 512 (8 x 384 from L 256), `pair_bias32_tc` (mma.sync TF32) below; training `pair_bias32_tc`; TF32 tcgen05 cores |
| launch count, inference | B200 3 at d_pair 128 L <= 256 (part 6: `inf_front` = LN + GEMM + pair, core, to_out) vs H100 5; else 5 (+ `ksplit_combine` from L 512) |
| launch count, training forward | equal: 6 (prep + LN fused on B200, pair, GEMM, core, gate, to_out vs prep, LN, GEMM, pair, gated core, to_out); part 5 adds bias^T beside the core at <= 16 heads (7; the backward loses it) |
| launch count, training backward | B200 11 vs H100 12 at <= 16 heads (part 5: no transpose launch, acc_reduce + finalize as one `finalize_red`); 24 heads 13 |
| `EXACT_DWB` fp32 pair-weight shadow | not applicable: opt-in and off on the H100 as well |
| PDL chains | none on the H100 path; B200 (part 6): the inference core after `inf_front` as a programmatic dependent (-2 us at L 128 / 256, 12 / 24 heads) |

### Round 3, v20 (2026-11-17 evening, `cuda:b200_sm100` 20)

Restart under BRIEF6 (agent r3-apb, GPU 3, base `b200-2.5.x` 83e4c1ddf merged). Records `$W/b2511/agents/r3-apb/_apb/out/`: before `B_<L>.jsonl`
(cuda, triton, pytorch; one measurer run per length, `dev.sh -x`, 50 iterations, graph timing), after `A_<L>.jsonl` + `C_128.jsonl`;
tables `B_summ.txt` / `F_summ.txt` (`_apb/v20/summ.py`); A / B `_apb/v20/ab.py` (interleaved graph replays, L2 flushed, trimmed means:
the B200 event timer steps by ~2.05 us, so medians of single replays only move in 2 us steps); accuracy `acc_v20`.

| dtype / mode | cases | above 0.667: before -> after | slower than H100 | worst ratio before -> after | ladder violations |
|---|---|---|---|---|---|
| master eval | 80 | 23 -> 21 | 0 -> 0 | 0.801 -> 0.801 (r148 L 256) | 0 -> 0 |
| master train | 54 | 27 -> 25 | 0 -> 0 | 0.910 -> 0.858 (r148 L 128) | 0 -> 0 |
| float32 eval | 80 | 0 -> 0 | 0 | 0.632 -> 0.630 | 0 -> 0 |
| float32 train | 54 | 0 -> 0 | 0 | 0.639 -> 0.631 | 0 -> 0 |

Taken:

| change | effect (A / B, trimmed means) |
|---|---|
| training at L <= 256: the main stream's own kernels as programmatic dependents -- attn_fwd2 / attn_dkv `-DPDLW=1` (griddepcontrol.wait + launch_dependents after the prologue), gate_rows / gate_bwd / qkv_bwd / ln_bwd / finalize_red through the extension's `pdl_go` (`pdl(1)`); the shared row kernels open with `APB_PDL_ENTRY()` (empty in apb_rows.cu, defined only by apb_rows_sm100.cu) | bitwise the same values; L 128 12 x 128 63.5 -> 61.4 us, 16 x 256 77.8 -> 73.7, 16 x 384 90.1 -> 86.0; L 256 24 x 128 88.1 -> 84.0, 16 x 256 104.4 -> 98.4; L 384-768 level to 8 us slower: off there |
| the tcgen05 pair backward leaves 96 SMs at L 128 (was 32) | 12 x 128 61.5 -> 59.4, 16 x 128 (d 512) 71.7 -> 69.6, 16 x 128 65.5 -> 63.4, 24 x 128 70.1 -> 67.6; at L 256 64 / 96 lose 2-4 us (24 heads L 256: 64 -1.8 us, L 384 / 512 +9 / +23: not taken) |
| inference: the core after the q / k / v / g GEMM (and the key-split combine after the core) as programmatic dependents (`attn_inf -DKSPLIT=1 -DPDLW=1`, `ksplit_combine` via `pdl_go`) | -0.3 to -2.9 us on most rows / lengths (16 x 128 d 512 L 512 47.5 -> 44.6, 8 x 384 L 128 / 256 -1.1 / -1.2), a few +0.3 to +1.1 (16 x 256 L 256) |
| the inference key split from L 384 (`_KSPLIT_MIN_L`, was 512) | L 384: 16 x 128 40.9 -> 39.0, 12 x 128 39.2 -> 37.6, 16 x 256 50.8 -> 49.7, 8 x 384 57.4 -> 55.9; 24 heads / d 512 level; L 256 +1.6-2.1 (kept unsplit) |
| the fused front (`inf_front`) for d_pair 256 at L 128 | 16 x 256 + gate constant 28.67 -> 26.62 us, 16 x 256 level (same bits) |
| part 7 (cancelled agent, commit 781c90e60): `attn_inf_tf32 -DKSPLIT` + `ksplit_combine32` for the token DiT's one-sample fp32 inference kept (TokenDiTProtenix S 1 L 256 92.1 -> 85.6, L 640 174.4 -> 150.0 us); its PDL pair backward (`attn_dqb -DPDLT`, `pair_bwd -DPB_PDL`) measured -0.8 .. +3.4 us (level) and reverted. Its prebuild edit had swallowed `if apb_rows:` into a comment (the extension call appended per row): fixed | |

Built, measured, rejected (removed):
- Wf packed alone on a side stream (`wf_pack`) so the training pair pass starts beside the pack + LayerNorm launch: +2-3.6 us at L 128-512
  (the pack launch on the main stream and the extra launch cost more than the overlap).
- to_out as an extension kernel (`out_proj`, mma.sync 32 x 64 tiles, Wo rows bulk-copied before griddepcontrol.wait under the core's tail,
  the core triggering early): +1 to +5 us against cuBLAS at every eval case (L 128-1024).
- a 3-stage ring for the shared bf16 pair pass (`FWD_STAGES 3` in the sm100 build): level at 12 / 16 heads, slower at 24 heads (the
  second block per SM no longer fits: 24 x 128 eval L 128 20.5 -> 24.6, L 512 51 -> 55 us).

Accuracy (master, every row, L 128 / 256 / 384 / 768 / 1024, 82 cases): 54 GREEN, 28 YELLOW (all training, 1.00-1.46 x tol, each at
its B200 token but r147 L 256 grad:pair 1.00 -> 1.02, r222 L 256 1.06 -> 1.10), 0 RED. Determinism (`det2.py`): master train rows 146 /
147 / 148 / 222 / 223 at L 128 / 256, master eval rows 41 / 45 / 147 at L 384 / 512: bitwise equal over three runs.

What remains (46 cases) and why (timelines `_apb/v20/tl*.sh`, `$W/b2511/jobs/apb20_tl*.log`):
- master train d_pair 128 (rows 146-148 / 222, every L, 0.67-0.86) and d_pair 256 / 384 at L 128 (0.69-0.75): r148 L 128 ends at 65.6 us
  after the flush, 18 launches: forward 28 us (prep_ln 6.5, [pair 5.0 | qkvg GEMM 6.9], core 5.9, gate, to_out 3.2), backward 38 us with
  the chain gate_bwd -> attn_dqb (8.7 us, 16 work items) -> qkv_bwd -> dxa (7.4 us, sharing the card with the pair backward and dW) ->
  ln_bwd -> finalize_red (6.8 us, waits for dW q|k|v|g). The target (53.9 us) needs ~12 us more; the remaining lever is fewer kernels on
  that chain (dqb with more work items at L 128, finalize without the dW copies), not launch latency.
- master eval d_pair 128 (rows 41 / 146-148, L 128-640, 0.67-0.80): L 128-256 is front + core + to_out (3 launches; r41 L 256: front
  14.5, core ends 8 us after it, to_out 3.5); L 384-640 the shared pair pass (r147 L 384 18 us for 38 MB) with the GEMM queued behind it.

### Round 3, v21 (2026-11-17 night, `cuda:b200_sm100` 21)

Agent r3-apb, GPU 3, base `b200-2.5.x` ddd984081. Records `$W/b2511/agents/r3-apb/_apb/out/`: before `B_<L>.jsonl` (cuda, triton, pytorch;
one measurer run per length, `dev.sh -x`, 50 iterations, graph timing), after `A_<L>.jsonl` (cuda); tables `B_summ.txt` / `A_summ.txt`
(`_apb/v21/summ.py`; the ladder pairs use the B triton / pytorch medians, which no change here touches); A / B `_apb/p7_chk.py`
(interleaved graph replays, L2 flushed, trimmed means; the event timer steps by ~2.05 us). Rejected diffs: `docs/b200/rejected/apb_v21_*.diff`.

Rule fix first: v20 had added an (empty) `APB_PDL_ENTRY()` hook to the SHARED `apb/apb_rows.cu`, which the H100 / A100 / RTX 5090 builds
compile (their source hash and build keys moved). `apb_rows.cu` is byte-identical to origin/main again; the PDL forms of the five row
kernels the B200 launches through `pdl_go` (gate_rows, gate_bwd, gate_bwd_hl, qkv_bwd, ln_bwd) are B200 copies under a `_pk` name in
`apb/apb_rows_pdl_sm100.cuh` (included only by `apb_rows_sm100.cu`; the same statements: training outputs and gradients bitwise equal
to v20, `det2.py` / `p7_chk.py`).

| dtype / mode | cases | above 0.667: before -> after | slower than H100 | worst ratio before -> after | ladder violations |
|---|---|---|---|---|---|
| master eval | 80 | 21 -> 17 | 0 -> 0 | 0.801 -> 0.803 (r148 L 256, the same code) | 0 -> 0 |
| master train | 54 | 26 -> 26 | 0 -> 0 | 0.858 -> 0.835 (r148 L 128) | 0 -> 0 |
| float32 eval | 80 | 0 -> 0 | 0 | 0.630 -> 0.643 | 0 -> 0 |
| float32 train | 54 | 0 -> 0 | 0 | 0.631 -> 0.639 | 0 -> 0 |

Taken:

| change | effect |
|---|---|
| `pair_bias_fwd2` (`apb/apb_pair_fwd_rows_sm100.cuh`): the shared d_pair 128 pair -> bias pass with each warp's 16 rows (4 KB) loaded by its own lane of warp 0 onto its own mbarrier (the shared kernel: one 32 KB bulk copy per tile from thread 0, every warp waiting for the whole tile); 2 stages, 2 blocks per SM. Inference wherever the shared pass ran (d_pair 128, L >= 384), training forward from L 384 | the same bits (bias bitwise equal, 16 layouts x L 128-1024). Standalone (L2 flushed): 24 x 128 L 640 / 1024 38.9 / 77.5 -> 36.9 / 72.8 us, 12 x 128 L 384 18.3 -> 16.8, 16 heads level to -1.6. Step (A / B): master eval 24 x 128 L 512 / 768 51.2 / 73.7 -> 49.2 / 71.6, 12 x 128 L 512 45.1 -> 43.0, 16 x 128 (d 512) L 640 / 768 57.2 / 65.5 -> 55.3 / 63.5; train L 768 12 x 128 194.4 -> 190.5, 24 x 128 254.0 -> 251.9 |

What bounds the shared pair pass (task item (b), probes in `_apb/v21/pf2probe.py` / `pf2st.py`, standalone, L2 flushed): with the mma and
the stores removed (loads only) the pass still takes 28.2 us for 105 MB (16 x 128, L 640: 3.7 TB/s) at 2 blocks x 2 stages x 32 KB in
flight, 28.7 us at one block x 6 stages x 32 KB (192 KB in flight per SM), 59.5-62.3 us for 268 MB at L 1024; the full pass is 30.7 / 60.9
us. So neither the issuing thread (8 lanes issuing changed nothing in the loads-only time) nor the bytes in flight is the limit, and the
compute is 2-8 % of it at <= 16 heads; deeper rings in one block per SM (4 / 6 stages) lose 10-40 % because a single block no longer
overlaps its compute / stores with the other block's. A device copy of the same 105 MB (read + write) takes 39 us.

Built, measured, rejected (removed; diffs `docs/b200/rejected/apb_v21_*.diff`):
- (a) attn_dqb on the main stream as a programmatic dependent of the gate backward (`-DPDLW=1`), attn_dkv on the side stream (dqb is the
  longer core on the dq -> qkv_bwd -> dxa chain): L 128 level (r222 63.5 -> 61.4, the rest equal), L 256 +-2, L 384 +2 to +8 us.
- (a) dxa split: the gate quarter (dg ready after the gate backward) as its own GEMM on side stream 0 beside the cores, the q | k | v part
  addmm'd onto it after qkv_bwd (K 4 W -> 3 W on the chain): +6 to +12 us at L 128-512 (the side GEMM takes SMs from dqb / dkv / the
  pair backward, the addmm reads the partial).
- (a) finalize without the dW copies: dW q | k | v | g as four GEMMs and dWo written straight into the parameter gradients (no head
  padding, W == d_single), `finalize_red` without its copy blocks: bitwise the same gradients, +4 to +10 us (12 x 128, 16 x 128 d 512;
  four small cuBLAS launches on the side stream end later than one GEMM + the copy blocks, and finalize waits for them).
- (b) `pair_bias_fwd2` with 4 / 6 stages in one block per SM (192 KB in flight): 10-40 % slower (above).

Timeline after (r148 master train L 128, 67.1 us): forward 25 us (prep_ln 6.6, [pair 4.5 | qkvg GEMM 6.7], core 5.8, gate, to_out 3.1),
backward: dO GEMM 5.3, gate_bwd 3.3, [dkv 7.0 | dqb 8.7], qkv_bwd 3.0, [dxa 7.3 | pair backward 12.1 | dW 6.2], ln_bwd 2.9, finalize_red
6.6. 18 kernels; the H100 runs 18 as well at 80.9 us.

Still above 0.667 (43) and why:
- master train d_pair 128 (rows 146-148 / 222, every L, 0.669-0.835) and d_pair 256 / 384 at L 128 (0.69-0.75): at L 128 / 256 a chain of
  ~12 dependent launches of 3-9 us each in the backward (above); the three ways to shorten it built here all lost (more launches beside
  the chain take its SMs; the chain kernels are latency-bound). From L 384 the pair forward + backward run at the card's read rate for
  this pattern (above).
- master eval d_pair 128 L 128-256 (0.69-0.80): three launches (front, core, to_out) as in v20; L 384-640 (0.68-0.76): the pair pass at
  ~3.7 TB/s (above) plus core and to_out.

### Round 3, v22 (2026-11-17 night, `cuda:b200_sm100` 22)

Agent r3-apb, GPU 3, base `b200-2.5.x` 7bc2db519. Records `$W/b2511/agents/r3-apb/_apb/out/`: before `B22_<L>.jsonl` (cuda, triton, pytorch;
one measurer run per length, `dev.sh -x`, 50 iterations, graph timing), after `A22_<L>.jsonl` / `A22b_<L>.jsonl` (cuda); tables
`B22_summ.txt` / `A22b_summ.txt` (`_apb/v22/summ.py`); A / B `_apb/v22/ab.py` (interleaved graph replays, L2 flushed as the measurer,
trimmed means); the flush probe `_apb/v22/flushprobe.py` (log `$W/b2511/jobs/apb22_j1.log`). Rejected diff:
`docs/b200/rejected/apb_v22_qkvbf_lnfin_pbevf.diff`.

**(a) Is the pair pass's ~3.7 TB/s the measurer's dirty L2?** The same `pair_bias_fwd2` graph timed after four flushes -- W: `zero_` of 256 MB
(the measurer), R: a 256 MB read (clean L2), WR: zero_ then the read, N: none -- in forms plain / L2 evict_first / evict_last / evict_normal,
and the loads alone (no mma, no stores), plain and evict_first; references: a read of the pair (amax) and a device copy (us):

| case, flush | plain | evict_first | evict_last / normal | loads alone | loads alone, evict_first | read of the pair |
|---|---|---|---|---|---|---|
| 16 x 128 L 640 (105 MB), W | 30.8 | 28.7 | 30.8 | 28.7 | 24.6 | 35.1 (3.0 TB/s) |
| same, R / WR | 28.7 | 28.7 | 28.7 | 22.5 / 22.8 | 22.5 | 28.7 (3.7 TB/s) |
| same, N | 28.1 | 27.3 | 28.1 | 19.5 | 18.4 | 23.0 (4.6 TB/s) |
| 16 x 128 L 1024 (268 MB), W | 61.5 | 57.4 | 61.5 | 58.4 | 50.1 | 67.9 (4.0 TB/s) |
| same, R | 54.0 | 54.9 | 55.3 | 45.1 | 45.1 | 51.4 (5.2 TB/s) |
| 24 x 128 L 640, W | 34.8 | 34.8 | 34.8 | 30.0 | 26.6 | 36.8 |
| same, R | 32.8 | 32.8 | 32.8 | 24.6 | 24.6 | 28.6 |

So the dirty lines are part of it, not all: they cost the loads 6-13 us (22.5 -> 28.7 at L 640, 45 -> 58 at L 1024) and the full pass 2-7 us
(which hides part of it under its compute / stores); with a clean L2 the pass still reads 28.7 / 54 us (the plain read of the same bytes
3.7-5.2 TB/s). The pattern that suffers least is evict_first on the read-once pair tiles: the loads get back 4-8 us, the pass 2-4 us at
<= 16 heads (24 heads: level, its mma / stores bound it). evict_last / evict_normal policies change nothing.

Taken: `pair_bias_fwd2` with the pair loads under `createpolicy.fractional.L2::evict_first` (`bulk_g2s_h<1>`, stages code 12; the same bits),
inference from L 512. Step A / B (master eval, trimmed means): r41 L 768 / 1024 63.6 / 90.5 -> 61.4 / 86.7 us, r146 L 384 / 768 36.9 / 61.4 ->
35.0 / 57.3, r147 L 512 / 768 50.7 / 69.6 -> 49.2 / 67.7; L 384 level to +1 us at r41 / r147 (so from L 512). The training forward stays on
plain loads: A / B there was L 512 level, L 768 -1 to -9 us, but the measurer read r146 L 640 / 768 and r222 L 768 5-9 us slower.

**(b) Fewer dependent launches in the L 128 training backward, built and measured (both bitwise equal to v21, both rejected):**
- `QKVBF` (attn_dkv / attn_dqb `-DQKVBF=1`, L 128 = one key chunk): dK | dV | dQ written as bf16 straight into dqkvg and dQ's 4-row
  column sums (qkv_bwd's values and order, by shuffles) into the dbq columns of acc: no qkv_bwd launch. 6 rows L 128: +0.5 to +2.6 us
  (r146 59.4 -> 61.4, r148 67.6 -> 68.9, r223 74.2 -> 76.7).
- `ln_fin` (ln_bwd + finalize_red in one launch: the LayerNorm-backward blocks first, each 128-thread quarter one of ln_bwd's 4-row blocks
  with the same column sums; the finalize blocks of the LayerNorm columns wait on an in-kernel counter with acquire loads, the last waiter
  resets it): +2 to +7 us at <= 16 heads (r148 L 128 / 256 69.0 / 84.7 -> 75.8 / 91.3), level at 24 heads.
Why: at L 128 the backward's critical path is not the main stream's dO -> gate_bwd -> dkv -> qkv_bwd -> dxa -> ln_bwd chain but the side
stream's attn_dqb (8.7 us) -> tcgen05 pair backward (12.1 us, 52 SMs) -> finalize. A heavier dqb epilogue (QKVBF) delays the pair backward;
ln_fin can only start after the join, so ln_bwd no longer overlaps the pair backward. Also measured and rejected: evict_first on the
tcgen05 pair backward's pair tiles (`PB_EVF`): -5.8 to +5.1 us, level.

| dtype / mode | cases | above 0.667: before -> after | slower than H100 | worst ratio | ladder violations |
|---|---|---|---|---|---|
| master eval | 80 | 19 -> 16 (r41 L 512 / 640, r146 L 512 / 640 below; r147 L 512 0.723 -> 0.696) | 0 -> 0 | 0.801 -> 0.801 (r148 L 256) | 0 -> 0 |
| master train | 54 | 24 -> 26 (the same code as v21: r147 / r222 L 768 and r148 L 640 read 0.668-0.673 against 0.664-0.672 before -- the run-to-run spread at the line) | 0 -> 0 | 0.858 -> 0.858 (r148 L 128) | 0 -> 0 |
| float32 eval | 80 | 0 -> 0 | 0 | 0.636 | 0 -> 0 |
| float32 train | 54 | 0 -> 0 | 0 | 0.631 | 0 -> 0 |

Accuracy: unchanged (the same bits: bias / outputs bitwise equal off / on in every A / B above). Determinism (`det2.py`): master train rows
146 / 147 / 148 / 222 L 512 / 768, master eval rows 41 / 146 / 147 / 148 L 512 (41 also L 1024): bitwise equal over three runs.

Still above 0.667 (42) and why: master train d_pair 128 at L 128 / 256 (0.78-0.86): the side-stream chain dqb -> pair backward -> finalize
(above), each kernel latency-bound with 16 / 128 work items; the forward 25 us of 6 launches. Master eval d_pair 128 L 128-256 (0.69-0.80):
front + core + to_out (v20). L 384-640: the pair pass, at most ~2 us closer to the clean-L2 time with evict_first.

### Round 3, v23 (2026-11-18, `cuda:b200_sm100` 23)

Agent r3-apb, GPU 3, base `b200-2.5.x` b070dfb11 (engine 2.5.14) then 05fe8a72b. Records `$W/b2511/agents/r3-apb/_apb/out/`: before `B23_<L>.jsonl`
(cuda, triton, pytorch, one measurer run per length, `dev.sh -x`, 50 iterations), after `A23_128/256.jsonl` (cuda; the other lengths run
unchanged code); tables `B23_summ.txt` / `A23_summ.txt` (`_apb/v23/summ.py`); A / B `_apb/v23/ab.py`; bitwise probe `_apb/v23/dqchk.py`.

| dtype / mode | cases | above 0.667: before -> after | slower than H100 | worst ratio before -> after | ladder |
|---|---|---|---|---|---|
| master eval | 80 | 18 -> 19 (r45 L 128 0.704: d_pair 384, a path not changed here -- run-to-run) | 0 -> 0 | 0.803 -> 0.786 (L 256 0.74-0.80 -> 0.74-0.76) | 0 -> 0 |
| master train | 54 | 23 -> 21 (r224 / r226 L 128 below; r223 L 128 0.709 -> 0.668) | 0 -> 0 | 0.858 -> 0.800 | 0 -> 0 |
| float32 eval / train | 80 / 54 | 0 -> 0 | 0 | 0.644 / 0.639 | 0 -> 0 |

Taken:
- (a) `dqb_small` (apb_rows_sm100.cu): the training dQ + dbias pass at L 128 / 256 with the QUERY rows split -- a CTA per (head, 32 queries),
  2 warps, mma.sync, the dQ partial per 128-key chunk into attn_dqb -DDQPART's slices. NH L / 32 CTAs instead of attn_dqb's NH (L 128) work
  items. Bitwise equal to attn_dqb (DQ / DB, 5 head layouts, L 128 / 256; every training output / gradient equal off / on). It heads the
  side stream's dqb -> pair backward chain: L 128 r146 59.4 -> 55.3, r148 67.9 -> 63.8, r222 62.8 -> 57.3, r223 74.2 -> 69.4, r226 86.0 -> 80.3 us;
  L 256 -1.3 to -3.3 us.
- (b) `small_core` (mode 1): the inference core as a CTA per (head, 64 queries), mma.sync, exact one-pass softmax, PDL dependent of the front,
  then cuBLAS to_out; at L 256 (d_pair 128) and d_pair 256 L 128: L 256 r41 30.7 -> 29.5, r146 28.7 -> 27.3, r148 32.4 -> 30.7, r42 / r43
  L 128 26.6 -> 24.6 us. Accuracy GREEN (18 / 18 master eval cases L 128 / 256), deterministic.

Built, measured, rejected:
- (b) the fully fused small-L inference (core + to_out in one launch, `MINIWORLD_APB_B200_SMALL=2`, kept opt-in): a cluster of the NH heads
  of a 64-query block exchanges the gated o for to_out. Per-CTA stamps (globaltimer) at L 128, after griddepcontrol.wait: loads 2.6 us,
  core 1.0-1.3, exchange 3.3 (bulk shared::cta -> shared::cluster pushes) / 4.6 (st.shared::cluster v4) / 3.0 (global + counter), to_out
  1.0: every form 0-4 us slower than attn_inf + cuBLAS (each CTA must receive the block's whole o, 64 KB). At d_pair 128 L 128 the core-only
  mode is level to one timer step slower, so not routed there.
- `dkv_small` (key rows split, the bias read as stored, no transpose launch): bitwise equal to attn_dkv, level (-1.8 .. +1.6 us): removed
  (`docs/b200/rejected/apb_v23_dkv_small.diff`).
- (c) the front at small L: standalone (L2 flushed), the pair role alone and the GEMM role alone each take ~7 us at L 128 and 9-11 us at
  L 256 for 4 / 16 MB -- a fixed first-touch latency after the flush, not bandwidth; no form built here changed it.

Still above 0.667 and why: master train d_pair 128 L 128-768 (0.68-0.80): after dqb_small the L 128 chain is the pair backward (12 us,
tcgen05) and the main stream's dkv -> qkv_bwd -> dxa -> ln_bwd -> finalize; L >= 384 the pair passes at the read rate. Master eval d_pair 128
L 128 (0.69-0.73): front 7 us + core + to_out, three launches each mostly latency; L 256-640: the front / pair pass (above).

### Round 3, v24 (2026-11-18, `cuda:b200_sm100` 24)

Agent r3-apb, GPU 3, base `b200-2.5.x` 80572b7d4 (engine 2.5.14). Records `$W/b2511/agents/r3-apb/_apb/out/`: before = `A23_*` (v23's after);
after `A24b_*` / `A24c_*` (cuda, triton, pytorch at L 128 / 256 / 384, two measurer runs; the other lengths run unchanged code); tables
`A24b_summ.txt` / `A24c_summ.txt`. Probes `_apb/v24/`: `frp.py` (inf_front standalone under CUPTI, L2 flushed: ns durations instead of the
2.05-us event steps), `frtl.py` (per-block stage stamps, a probe build with `-DFR_TL=1`), `mp.py` (micro-kernels: empty / one 32 KB bulk
copy / 8 x 4 KB copies / 8 ldg.v4 per thread), `frcmp.py` (inf_front against the v23 source built as a probe extension: bitwise).

What the front's 7 us at L 128 is (stamps, 16 x 128 D 384, after the flush; CUPTI kernel 7.1 us, an empty 128-block kernel 1.1 us, a kernel
that only reads 32 KB per block 3.2 us whether by one bulk copy, eight 4 KB copies or 8 ldg.v4 per thread): GEMM role -- copies issued
2.0 us after entry (the 64 row copies from warp 0, two per lane: a bulk-copy issue holds its thread ~1 us after the flush), LayerNorm rows
in shared memory 4.3 us (the rows' loads were issued only after warp 0's copies), weights landed 4.3, end 5.6; pair role -- Wf fragments
3.1 us (thread 0 issued the two ring copies before its own Wf loads), tile landed 3.3, end 4.4. So: first-touch latency (~2 us per
dependent round trip after the 256 MB zero_ flush), paid twice in a row in the GEMM role; neither bandwidth nor CTA count.

Taken (all inf_front outputs bitwise equal to v23 -- bias, qkvg, y; 12 layouts x mask on / off, `frcmp.py`):
- every first-touch load before anything waits: the GEMM tile's rows, LayerNorm's weights and the epilogue bias into registers first, then
  the weight rows; the pair role's Wf words before the ring's copies, the first copies from NSF lanes. Stamps: GEMM end 5.6 -> 4.6 us
  (alone 3.6), pair end level; CUPTI 7.10 -> 6.75 us (16 x 128 L 128).
- the GEMM tile's 64 weight rows as ONE 3-D tensor-map box (dims 64 x NC x D / 64, 128-B swizzle; 48 / 64 KB) instead of 64 row copies, the
  fragments by ldmatrix (`front_tmw`, default 1): CUPTI L 128 16 x 128 6.43 -> 6.21, D 512 level, L 256 -0.3 to -0.6 us.
- `inf_fronto_k` (pblocks <= -100): the roles overlapped in one wave -- at L 256 the GEMM tiles (192-256) and the pair blocks did not fit in
  one wave and most pair blocks started after the GEMM blocks ended (stamps: pair work from 6 us on). Blocks [0, G) take a GEMM tile, then
  join the pair walk; the other P - G blocks walk KA = 6 pair tiles each from the start. Routed at L 256 and L 384, d_pair 128, D 384,
  12 / 24 heads (`_front_pb`; and `_front` now takes L 384 there). Step A / B (master eval, interleaved replays, trimmed means): L 256
  12 x 128 28.5 -> 26.7, 24 x 128 30.6 -> 27.7 us; L 384 (the front instead of LN + GEMM + pair_bias_fwd2) 12 x 128 36.9 -> 35.2, 24 x 128
  40.2 -> 37.0. Not routed: 16 x 128 L 256 +0.8, 16 x 128 d 512 +1 to +8, L 512 / 640 level to +6, KA 10-16 slower.

Built, measured, rejected:
- the front as a work queue (an atomic counter per launch, GEMM items first then pair tiles, the last block resetting it): the same bits,
  L 128 6.8 -> 10.1 us, L 256 14.6 -> 16.3 (the claims cost a round trip each after the flush) -- `docs/b200/rejected/apb_v24_front_queue.diff`.
- the GEMM tile's mma phase (1.3 us between weights landed and the stores): ldmatrix fragments (kept with the tensor-map layout, level) and
  two accumulator chains per m-tile (the bits change; level): neither moved it.
- the tcgen05 pair backward over more SMs at L 128 (task (b); reserve 96 -> 64 / 20, i.e. 52 -> 84 / 128 CTAs, one or two tiles each):
  master train +1.0 to +4.1 us (r146 53.3 -> 57.0 / 57.4) -- the dxa / dW GEMMs beside it lose the SMs. Its loads are 3 boxes per 40 KB
  tile (2 x 16 KB pair, 8 KB dbias), already few and large.

| dtype / mode | cases | above 0.667: before (A23) -> after (A24b / A24c) | slower than H100 | worst ratio | ladder |
|---|---|---|---|---|---|
| master eval | 80 | 19 -> 19 / 19 (out: r147 L 256 0.747 -> 0.667 / 0.66x; down: r146 L 256 0.763 -> 0.714, r146 L 384 0.756 -> 0.712, r147 L 384 0.786 -> 0.706, r148 L 128 0.733 -> 0.669 in A24c; in: r46 L 128 / L 256 0.669-0.677, r47 L 128 0.667, paths not changed here -- run-to-run) | 0 | 0.786 -> 0.764 (r41 L 384, unchanged path) | 0 -> 1 / 1 (triton vs pytorch on unchanged rungs, a different case each run: r41 L 256 2.4 %, r148 L 128 4.6 %) |
| master train | 54 | 21 -> 21 (no training change) | 0 | 0.800 -> 0.818 (run-to-run) | 0 |
| float32 eval / train | 80 / 54 | 0 -> 0 | 0 | 0.631 / 0.639 | 0 |

Accuracy: master eval rows 41-46 / 146-148 at L 128 / 256 / 384, 27 cases GREEN (the L 384 front replaces cuBLAS with the front's mma GEMM).
Determinism: master eval rows 41 / 146 / 147 / 148 at L 128-384 bitwise over three runs. `apb_rows.cu` untouched (byte-identical to 2.5.14).

Still above 0.667 and why: master eval d_pair 128 L 128 (0.69-0.72, one 2.05-us timer step from the line): front ~6 us (two dependent
first-touch round trips + the GEMM tile's 1.3 us mma phase) + core 5.6 us after it + cuBLAS to_out 3.1 us; L 256 / 384 at 16 heads and
d 512: the front's roles add up (each alone 7-10 us; together 13-16) -- both latency-bound per block with two blocks per SM, so sharing the
SMs serialises them. Master train: unchanged (the v23 notes).

### Round 3, v25 (2026-11-18, `cuda:b200_sm100` 25) -- training

Agent r3-apb, GPU 3 (shared with r3-triattn / r3-token-dit this round), base `b200-2.5.x` 783647647. Before = v24's measured state (`A24c_*` /
`B23_*`, the APB code unchanged since). Probes `_apb/v25/`: `ab.py` (A / B of any module flag, interleaved graph replays, L2 flushed, trimmed
means, outputs + every gradient compared), `pbst*.py` (the tcgen05 pair backward alone under CUPTI after the 256 MB flush, bytes / us),
timelines `_apb/tl.py` (`$W/b2511/jobs/apb25_j1.log`). Rejected diffs: `docs/b200/rejected/apb_v25_*.diff`.

Training stages against the HBM rate (step timelines, after the flush; % of 8 TB/s; biggest gap first among the memory-heavy stages):

| stage (case) | bytes | us | TB/s | % |
|---|---|---|---|---|
| qkv_bwd (r146 L 768, beside the pair backward, 8 SMs free) | ~11 MB | 29.6 | 0.37 | 5 |
| bias_transpose64 (r146 L 768 / r148 L 384, side stream beside the core) | 28 / 9.4 MB | 31.6 / 11.7 | 0.9 / 0.8 | 11 / 10 |
| prep_ln (pack of the fp32 masters + LN, every L) | 7.8 MB | 6.9-7.9 | 1.0-1.1 | 13 |
| pair backward tcgen05 (r148 L 384 in the step / alone) | 85 MB | 34.1 / 26 | 2.5 / 3.3 | 31 / 41 |
| pair_bias_fwd2 training forward (r148 L 384) | 42 MB | 13.3 | 3.2 | 40 |
| pair backward (r146 L 768 in the step / alone) | 330 MB | 81.9 / 68.7 | 4.0 / 4.8 | 50 / 60 |
| pair_bias_fwd2 (r146 L 768) | 165 MB | 34.3 | 4.8 | 60 |
For reference the plain read of the pair after the measurer's flush runs at 3.0-4.0 TB/s (v22), so the pair passes from L 768 sit at that
ceiling; the training forward pass already has the 8-lane issue (v21). The pair backward's trace (block 0, 16 x 768): ~1.9 us per tile,
the epilogue busy ~1.4 us of it (phase A 0.7-1.0, B 0.55), a tile's load 2.5 us from issue, the store read 1.3 us -- a stage-ring cycle.

Taken: `backward_nc` + `finalize_red_nc` (16 x 32 at d_single 512 only, rows 148): the dW q|k|v|g / dWo GEMM outputs are the gradients
(views), finalize without its 640 copy blocks runs on the main stream beside the GEMMs. Bitwise the same gradients. r148 L 128 63.5 ->
61.0-61.3, L 384 -0.1 / -3.2, L 512 -2.2, L 768 -2.8 us; L 256 -0.6 / +2.0 (level). Not routed for 12 x 32 / 24 x 16 at 384 (level to +1.6).

Built, measured, rejected (all bitwise equal to v24 unless noted):
- pair backward loads from 3 lanes / stores from 2 (`PB_ML`): +0 to +1.5 us alone; B(n) before A(n+1) when tile n+1 is late (`PB_RO`):
  +4 to +27 %; the epilogue's shared loads batched ahead of the in-place stores (`PB_BL`): level (-1 to +1 %); 16 epilogue warps of 32
  columns (`PB_EW=16`, sum of squares per quarter: dz 2e-3 rel, deterministic): 5-15 % slower at every size. The per-SM rate is not
  set by the issuing lane nor by the epilogue's instruction latency.
- the PDL row kernels with their independent loads before griddepcontrol.wait (gate_rows / gate_bwd / qkv_bwd / ln_bwd `*_pf`, qkv_bwd's
  dq chunks 4 in flight): L 128 / 256 -0.1 to +1.9 us, L 384 / 768 (without PDL) level to +3.7; attn_dkv `-DPDLW=2` (K / V / q / bias / LSE
  before the wait): -0.6 to +2.0. PDL at L >= 384 with them: +0.5 to +4.9.
- finalize split (finalize_red_nc on main, the copies `fin_copy` on side 0 after dW; 16 x 24 / any heads): level to +2.9 us.
- pair_bias_fwd2 in the training forward at L 256: level; its evict_first loads in training at L 512 / 768: -3.6 to +3.7 (level).
- not built: (3) the gate in the to_out prologue needs a hand to_out GEMM (rejected twice before: v20 out_proj +1-5 us, part 6 out_gate
  8.7-14.8 vs 5.4 us); (4) the eval front as a programmatic dependent of the caller's last kernel cannot show in the per-class tokens
  (the measured graph starts after the flush, no programmatic predecessor).

### Round 3, v26 (2026-11-18, `cuda:b200_sm100` 26)

Agent r3-apb, GPU 2, base `b200-2.5.x` f2d941e47 (engine 2.5.14). Records `$W/b2511/agents/r3-apb/_apb/out/`: before `B26_<L>.jsonl` (cuda, triton,
pytorch, every dtype / mode, one measurer run per length, `dev.sh -x`, 50 iterations), after `A26_<L>.jsonl` (cuda, master eval L 128-640),
`A26t_128` / `A26u_128` (master train L 128, two runs), `A26v_128` (master eval L 128 rerun); tables `B26_summ.txt` / `A26_summ.txt`
(`_apb/v26/summ.py`); A / B `_apb/v25/ab.py` (interleaved graph replays, L2 flushed, trimmed means, outputs and every gradient compared).
Another agent's probes (`_tools/tl.py`, not under dev.sh) ran on GPU 2 during some of these runs. Rejected diff: `docs/b200/rejected/apb_v26_inf_pdlw2.diff`.

| dtype / mode | cases | above 0.667: before -> after | slower than H100 | worst ratio before -> after | ladder |
|---|---|---|---|---|---|
| master eval | 80 | 19 -> 9 (rerun of L 128: 10, r44 0.670 on an unchanged path) | 0 | 0.764 -> 0.745 (r41 L 256) | 0 -> 0 |
| master train | 54 | 24 -> 24 (L 128: r146 0.725 -> 0.672, r222 0.747 -> 0.693, r148 0.757 -> 0.706, r147 0.747 -> 0.746) | 0 | 0.801 -> 0.801 (L 256, unchanged) | 0 -> 0 |
| float32 eval / train | 80 / 54 | 0 -> 0 (not touched) | 0 | 0.630 / 0.644 | 0 |

Taken (`small_core`, apb_rows_sm100.cu; `apb_rows.cu` untouched):
- (1) the fused small-L inference with the o exchange as `st.async` 16-B stores onto each peer's mbarrier (complete_tx; the peers' barriers
  armed before a split cluster arrive in the prologue, a warp writing 512 contiguous bytes per destination) and 16 queries per CTA
  instead of 64 (the exchange per CTA is (H - 1) slots of QB rows: 64 rows = 60-75 KB took 2.0-2.8 us even as st.async -- stamps,
  globaltimer); 24 heads as 2 heads per CTA (a cluster of 12). Routed at d_pair 128, L 128 (`_small_fused`). Master eval (A / B, trimmed
  means, attn_inf + cuBLAS to_out -> fused): 12 x 128 20.5 -> 18.4 (another run 18.4 -> 16.4), 16 x 128 (d 512) 22.5 -> 18.5, 24 x 128
  20.5 -> 19.5, 16 x 128 level; st.async alone at 64 queries level (r146 / r148) to +1.2 (r41); 32 queries about half the gain; at L 256 every
  fused form loses 2-10 us (the core-only form stays there); d_pair 256 L 128 level (not routed).
- the core-only `small_core` at L 384 / 512 / 640 with 128-key chunks (running maximum; L <= 256 keeps one chunk, the same statements),
  no key split and no combine launch, where the CTAs fit one wave (`_small_long`: L 384 32 queries per CTA up to 16 heads, 64 at 24; L 512
  64 up to 16 heads; L 640 12 heads): r41 L 384 36.9 -> 34.8, r146 35.9 -> 32.8, r147 37.9 -> 35.7, r148 39.0 -> 34.8; L 512 r41 43.0 -> 41.0,
  r146 43.0 -> 40.5, r148 44.5 -> 41.0; L 640 r146 53.3 -> 49.1 (16 / 24 heads at L 640 and 24 at L 512: two waves, +1 to +8 us, not routed).
- 32 queries per CTA for the core-only form at L 256 up to 16 heads (bitwise the same values): r41 28.7 -> 28.0, r146 26.7 -> 25.0, r148
  30.7 -> 29.0 (24 heads +0.7: 64 stays).
- training forward at L 128 (d_pair 128) in one launch (`small_core_train`, the fused form with attn_fwd2's scaling, writing O fp32, LSE
  and og for the backward; replaces attn_fwd2 -> gate_rows -> cuBLAS to_out): master train r146 53.6 -> 51.3, r148 60.4 -> 57.4, r222 58.6 ->
  54.2, r147 60.3 -> 59.4 us; every gradient within 6e-3 (relative to its max) of the attn_fwd2 path. At L 256 (`MINIWORLD_APB_B200_TRF256=1`):
  r146 -2.1, the others +0.7 to +3.9 (not routed).

Measured and rejected: (3) attn_inf `-DPDLW=2` (the tensor maps prefetched before griddepcontrol.wait, launch_dependents right after it, so
the key-split combine launches under the core): L 128-640 level to +1.8 us (diff archived). (2) the tcgen05 d_pair 128 pair pass as two
blocks per SM: not built -- v21 / v22 measured the shared pass's loads alone at its own time (28.2 vs 30.7 us at 16 x 128 L 640, the
compute 2-8 %), so a faster compute engine cannot move it; the L 384-640 gap was the chain after the pair pass (GEMM tail, core 7 us,
combine 3 us, to_out 3.4 us: timeline r41 L 384), which the long small_core above shortened.

Accuracy: master eval rows 41 / 146 / 147 / 148 / 42 / 43 L 128 and 41 / 146 / 147 / 148 L 384 / 512 / 640: 18 / 18 GREEN; master train rows
146 / 147 / 148 / 222 L 128: 0 RED, 1 GREEN, 3 YELLOW at their B200 tokens (1.16 / 1.02 / 1.07 against 1.20 / 1.00 / 1.07). Determinism
(`det2.py`): those eval cases and the four training rows at L 128 bitwise over three runs.

Still above 0.667 and why: master eval r41 L 256 (0.745) / r148 L 256 (0.700) / r147 L 256 (0.667): front 13-14 us + core + to_out;
r147 L 512 / 640 (24 heads: the long core needs two waves); r41 / r146 L 384 (0.684 / 0.669: the pair pass ~14 us at ~2.7 TB/s plus the GEMM tail);
r147 L 128 (one 2.05-us timer step: 19.5 us against 19.7). Master train: unchanged at L >= 256 (the v25 notes); L 128 now 51-59 us against
50.6-52.9 (the backward's dqb -> tcgen05 pair backward 12 us -> finalize chain).

### Round 3, v27 (2026-11-18, `cuda:b200_sm100` 27)

Agent r3-apb, GPU 2, base `b200-2.5.x` b74fcc08a (engine 2.5.14). Records `$W/b2511/agents/r3-apb/_apb/out/`: before `B27_<L>.jsonl` (cuda,
triton, pytorch, every dtype / mode, one measurer run per length, `dev.sh -x`, 50 iterations), after `A27c_<L>.jsonl` (L 128-768, the same
procedure), `A27d_256.jsonl` (L 256 again after the last route change), `A27_<L>.jsonl` (cuda only; L >= 1024 unchanged code); tables
`B27_summ.txt` / `A27d_summ.txt` (`_apb/v27/summ.py`), per case `_apb/v27/ba.py`. A / B `_apb/v25/ab.py`; stage stamps `_apb/v27/sbtl.py`
(a `-DSBW_TL=1` build). Cuda-only measurer runs read 1-4 us slower on unchanged paths (float32 eval r146 L 128 39.2 -> 42.8 us cuda-only,
39.3 with triton / pytorch interleaved): before / after are compared with the same procedure.

| dtype / mode | cases | above 0.667: before -> after | slower than H100 | worst ratio before -> after | ladder |
|---|---|---|---|---|---|
| master eval | 80 | 10 -> 9 (out: r44 L 128 0.684 -> 0.634, r42 L 256 0.696 -> 0.622; in: r147 L 384 0.666 -> 0.667, unchanged path) | 0 | 0.747 -> 0.750 (r148 L 256, unchanged path) | 0 -> 0 |
| master train | 54 | 25 -> 22 (out: r146 / r223 / r224 L 128; L 256 0.78-0.80 -> 0.74) | 0 | 0.800 -> 0.810 (r147 L 256 24 heads, unchanged path: 81.7 -> 83.7 run-to-run) | 0 -> 0 |
| float32 eval / train | 80 / 54 | 0 -> 0 (not touched) | 0 | 0.630 / 0.645 -> 0.630 / 0.639 | 0 |

Taken:
- `small_core_bwd` (apb_rows_sm100.cu): the training attention backward at L 128 (any head count) and L 256 (12 / 16 heads) in ONE launch
  instead of gate_bwd + attn_dkv + dqb_small + qkv_bwd (four launches on two streams). A CTA per (head, 32-row block), 512 threads: it makes
  dO = dog sigmoid(g) and D = sum(dog og) for all L rows of its head itself (redundant over the head's CTAs: no exchange), then 8 query warps
  (2 slabs of 16 rows x 4 key quarters: dbias fp32 for the pair backward, dQ partials) and 8 key warps (2 slabs of 16 keys x 4 query
  quarters: dK / dV partials, the bias read as stored); the quarter partials are summed in quarter order (deterministic) into dqkvg, dbq's
  4-row column sums written in qkv_bwd's acc layout. The forward then skips bias^T. Kernel 9.8 us at r148 L 256 (eval stamps: entry->wait
  in the step, gate 1.5, loops 2.3, sums 1.5), 5.9 us at L 128; it moves the pair backward's start ~2 us earlier and frees the SMs of the
  old dkv / dqb pair. With it the tcgen05 pair backward leaves 80 SMs at L 128 (96 before) and 48 at L 256 (32 before; sweep 16 / 48 / 56 /
  64 / 72 / 84: 48 best: r146 -3.1, r147 -2.5, r148 -2.2, r222 -6.0 us).
- finalize_red in its three parts (`finalize_part`: the dWf | dWb | dln_pair.weight blocks on side 1 right after the pair backward, the dW
  copies on side 0 right after the dW GEMMs, the row columns on main right after ln_bwd; the same blocks, bitwise the same gradients) for 16
  heads up to L 512 (A / B: L 128 -0.5 .. -1.9, L 256 -0.6 .. -3.0, L 384 -0.2 .. -1.6 us; 12 heads +1.5 at L 128, +4.1 at L 768: not routed).
- small_core on the non-front inference path (d_pair 256 / 384 / 512 at L 128 / 256, where inf_front does not serve; also with an extra
  bias, which planes_fix has folded into the planes): fused with to_out at L 128 (16 queries per CTA), core-only at L 256. A / B: L 128 r44
  -3.1, r45 -2.0, r46 -2.0, r47 -2.1 us; L 256 r42 -2.2, r43 -4.1, r44 -2.9, r45 -4.5, r46 -3.7, r47 -4.3 us.
- Master train after (A27c / A27d): L 128 r146 51.0 -> 48.9, r147 59.2 -> 54.9, r148 57.1 -> 55.2, r222 55.1 -> 53.3, r223 / r224 71.5 -> 67.5;
  L 256 r146 75.6 -> 71.5, r148 83.7 -> 77.9, r222 79.7 -> 73.9.

Built, measured, rejected (`docs/b200/rejected/apb_v27_bwd_dog_cluster_early_wf.diff`):
- the first small_core_bwd forms: 2 warps per role over all L (14.3 us at L 256: the serial chain) and 16-row CTAs (21.7 us: twice the
  redundant loads).
- dog = dy Wo inside small_core_bwd (`small_core_bwd2`: each CTA its rows x its head's Wo columns, the head's CTAs a cluster reading each
  other's dO rows / D by ld.shared::cluster; no dO GEMM launch): L 128 level (-1.1 .. +0.3 us), L 256 +5 to +9 us (clusters of 8 CTAs with
  200 KB of shared memory wait for free SMs beside the dWo GEMM).
- Wf / Wb by a launch of their own ahead of the training pair pass, the weight pack on main beside it (`prep_wf`): +2 to +4 us at L 128-384
  (the pack and the pair pass share the memory system; the q|k|v|g GEMM starts later), -10.7 at r146 L 768 only.
- small_core_bwd at L 256 for 24 heads (two 256-thread CTAs a SM: +2.1 us) and 8 heads (head width 48: +4 us); the pair backward reserve 56
  at L 384 / 512 (+0.4 to +11.7 us).

Accuracy: master train rows 146-148 / 222-226 L 128 / 256: 0 RED, 5 GREEN, 11 YELLOW (worst 1.20 tol, r146 L 128 grad to_query.bias; its
B200 token 1.20); master eval rows 41-47 / 146-148 L 128 / 256: 20 / 20 GREEN. Determinism (`_apb/det2.py`, three runs, bitwise): those
training rows at L 128 / 256 (16 cases, output + 12 gradients each) and eval rows 42-47 at L 128 / 256. `apb_rows.cu` untouched.

Still above 0.667 and why: master train L 256 (0.74; r147 24 heads 0.81, old path): forward 38 us (prep_ln 6.3 -> pair pass 9.6 -> core ->
to_out, the pack at ~1.2 TB/s heads it) + dO GEMM 5.4 + small_core_bwd 9.8 + the pair backward 18.8 us at 100 SMs; L 384-768 unchanged (the
pair passes at the read rate, v25); L 128 r148 / r222 / r147 0.68-0.70 (1-2 us over). Master eval: r148 / r41 L 256 (front 13-14 us), r147
24 heads L 128-640, r41 / r146 L 384 (unchanged).

### Round 3, v28 (2026-11-18, `cuda:b200_sm100` 28) -- the training forward front, the pair passes

Agent r3-apb, GPU 2, base `b200-2.5.x` 37f9ba886 (engine 2.5.14). Records `$W/b2511/agents/r3-apb/_apb/out/`: before `B28_<L>.jsonl` (cuda,
triton, pytorch, every dtype / mode, L 128-768, one measurer run per length, `dev.sh -x`, 50 iterations), after `A28b_<L>.jsonl` (the same
procedure); tables `B28_summ.txt` / `A28b_summ.txt` (`_apb/v28/summ.py`), per case `_apb/v28/ba.py`. A / B `_apb/v28/abn.py` (N flag sets,
interleaved graph replays, L2 flushed, trimmed means, output + every gradient compared bitwise); timelines `_apb/tl.py`; the pair backward
alone with its per-block / per-tile trace `_apb/v28/pbtr.py`, `pbws.py`; finalize alone `_apb/v28/fin.py`. L >= 1024 not re-measured (no
route there changed). Rejected diffs: `docs/b200/rejected/apb_v28_*.diff`.

| dtype / mode | cases (L 128-768) | above 0.667: before -> after | slower than H100 | worst ratio before -> after | ladder |
|---|---|---|---|---|---|
| master train | 54 | 21 -> 19 (out: r148 L 128 0.682 -> 0.658, r222 L 640 0.672 -> 0.665) | 0 | 0.810 -> 0.790 (r147 L 256) | 0 -> 0 |
| master eval | 60 | 8 -> 8 (no eval path changed) | 0 | 0.750 -> 0.751 (r148 L 256) | 0 -> 0 |
| float32 eval / train | 60 / 54 | 0 -> 0 (not touched) | 0 | 0.630 / 0.639 | 0 |

The front, measured (timeline r148 master train L 256, after the flush): prep_ln 3.4-9.7 us, the pair pass 10.1-19.8, the q|k|v|g GEMM
17.0-23.5 (queued 7 us behind the pass's persistent grid), the core from 23.4. prep_ln at d_single 512 had 64 registers a thread -> 4 blocks
per SM (592 resident) for 689 blocks: two waves (D 384: 48 registers, one wave, 3.9 us).

Taken (every output and gradient bitwise equal to v27, `abn.py`; determinism `det2.py` rows 146 / 147 / 148 / 222 L 128 / 256 / 384, three runs):
- prep_ln_k `__launch_bounds__(256, 5)` at D 512 (`prep_mb`): 6.2 -> 4.4 us; r148 L 128-512 -2.0 to -2.8 us.
- the q|k|v|g GEMM captured before the side stream's pair pass (`_gemm_first`, d_pair 128, L 256-768 at <= 16 heads, L 256 at 24): the GEMM
  takes its SMs first (3.6 us instead of 6.0 behind the pass; the pass 9.1 -> 11.7 beside it, the core starts 0.7 us earlier). r147 L 256
  -2.8, r222 L 384 -1.3, L 768 -2.4 to -5 (r146 / r148 / r222); L 128 +1.3 to +2.7 and wide pairs (r223 / r225 / r226 / r47 L 256 / 384:
  +3 to +14 us) not routed.
- pair_bias_fwd2 in the training forward from L 256 at 16 heads (`_pf2_tmin`; v21 had it from 384): with the GEMM first r148 L 256 -2.0, r222 -0.4.
- Step (measurer, before -> after): r148 L 128 / 256 / 384 / 512 / 768 55.2 / 78.8 / 108.3 / 135.2 / 219.7 -> 53.2 / 75.6 / 104.4 / 131.2 / 217.2;
  r222 L 384 / 512 104.4 / 133.2 -> 102.3 / 131.1; r147 L 256 83.8 -> 81.7; r146 L 768 196.4 -> 192.4 us.

Built, measured, rejected:
- (1b) the training pair pass from the masters (`pair_bias_fwd2m`: Wf = bf16(rbf(wb) lpw ws) made in its prologue -- bitwise the pack's
  words, the same bias), so it needs no pack before it: captured first on the side stream with the pack + LayerNorm on main, or after it;
  2 / 1 blocks per SM; 40 SMs reserved; the tcgen05 apb_pair_fwd from the masters instead (not bitwise, 3e-3). L 256 2 blocks per SM +4 to
  +5 us (the pass's persistent grid takes every SM and prep_ln queues behind it), 1 block per SM level to -1, reserve 40: prep_ln 15 us on 40
  SMs; tcgen05 -1.9 .. +0.6 (mixed); L 128 level to +2. The pack and the pass are each latency-bound and share the SM slots: overlapping
  them does not beat running them back to back (v20 wf_pack / v27 prep_wf lost the same way).
- (3) the pair backward's dpair stored from registers (`PB_DST`, st.global per row, the stage back to the loader right after phase B): bitwise,
  +5 to +15 us (uncoalesced 16-B row stores).
- (3) phases A and B on their own warps (`PB_WS`, 608 threads, A(n + 1) beside B(n), (k, c0) handed in shared memory four tiles deep, D[ab]
  freed by an mbarrier): bitwise, level alone (16 x 256 18.9 -> 18.6, 16 x 768 71.5 -> 69.8, 8 x 384 22.0 -> 22.8 us). The trace shows why: the
  tile period stays ~1.5 us with either form -- a tile's load lands ~3 us after its issue, so 4 stages x 40 KB per SM bound the rate, not the
  epilogue; with smem at 225 of 227 KB no fifth stage fits.
- finalize_part 4 inside the pair backward (`PB_FIN`: the last 16 blocks take a ticket, wait for the rest and reduce the partial rows in
  finalize_red_k's order -- bitwise): +1 to +2.3 us (finalize part 4 alone is 3.9 us in L2, 6.8 after the flush: a slow kernel, but 16 x 352
  threads of the pair backward reduce it slower still).
- pair_bias_fwd2 at L 256 for 12 / 24 heads: +1.4 / +0.8.

Still above 0.667 and why: master train L 256 (r147 0.79, r222 0.74, r146 0.74, r148 0.72): forward to the core ~20 us (prep 4.4 -> [GEMM 3.6
| pair pass ~11]) + core / gate / to_out ~11, backward dO GEMM 5.3 -> small_core_bwd 9.9 -> pair backward 18.8 (ring-bound at ~1.5 us a tile
after ~3 us of first-load latency) -> finalize 5 us; L 384-640 the pair passes at the post-flush read rate; L 128 r222 / r147 0.67-0.70.
Master eval unchanged (r148 / r41 L 256 front, r147 24 heads, r41 / r146 L 384).

### Round 3, v29 (2026-11-18, `cuda:b200_sm100` 29) -- the pair backward's bound, 24-head finalize parts

Agent r3-apb, GPU 2, base `b200-2.5.x` baf7a5e85 (engine 2.5.14). Records `$W/b2511/agents/r3-apb/_apb/out/`: before `B29_<L>.jsonl` (cuda,
triton, pytorch, every dtype / mode, L 128-768, one measurer run per length, `dev.sh -x`, 50 iterations), after `A29b_<L>.jsonl` (L 256 / 512
re-run after the last route change, the other lengths `A29_<L>`); tables `B29_summ.txt` / `A29b_summ.txt`. Probes `_apb/v29/`: `pbv.py` (pair
backward variants alone, CUPTI after the 256 MB flush, interleaved, dz + partial rows compared), `pbtr.py` (per-tile trace, `PB_TX` stamps
inside phase A / B), `fin2.py` (finalize dWf blocks alone), `abn.py` (step A / B). Rejected diffs: `docs/b200/rejected/apb_v29_*.diff`.

| dtype / mode | cases (L 128-768) | above 0.667: before -> after | slower than H100 | worst ratio before -> after | ladder |
|---|---|---|---|---|---|
| master train | 54 | 20 -> 20 (out: r147 L 768 0.667 -> 0.665; in: r222 L 640 0.665 -> 0.672, unchanged path) | 0 | 0.791 -> 0.773 (r147 L 256) | 0 -> 0 |
| master eval | 60 | 8 -> 8 (no eval path changed) | 0 | 0.750 -> 0.752 (r148 L 256) | 0 -> 0 |
| float32 eval / train | 60 / 54 | 0 -> 0 | 0 | 0.630 / 0.639 -> 0.630 / 0.642 | 0 |

What bounds the tcgen05 pair backward (trace, 16 x 768, 100 CTAs): the tile period is 1.71-1.77 us and is the EPILOGUE's: phase A 0.9-1.0 us
(sum of squares 0.26, the halves' barrier + dbias loads 0.17, S1 / S2 0.31, a stores 0.20) + phase B 0.57 + ~0.2 us of waits, back to back on
the 8 epilogue warps; a tile's load lands 1.4-1.7 us after its issue (v28's ~3 us was a ring held by the epilogue, not the cause) and with
5 stages the loads simply wait longer (3.3-3.7 us) at the same period. The per-SM rate is fixed (~37 GB/s, read + write): 16 x 768 93.5 us on
100 SMs, 131 us on 68, 69.6 us on 148.

Built, measured, rejected (all bitwise equal to v28 unless noted; alone, CUPTI, 9 sizes x reserve 0 / 48 / 80):
- `PB_R5`: five 32 KB pair stages + a 4-slot dbias ring freed after phase A (a / Wf^T in the 64-B swizzle: 16 + 8 KB instead of 32 + 16,
  dWf^T MMA N = 32): -2 .. +3 % (16 x 768 r48 93.6 -> 95.5, 16 x 384 r0 24.3 -> 24.0).
- `PB_X3`: the pair tile as one 3-D 32 KB box for the load and the store (2 + 1 TMA ops a tile instead of 3 + 2): -1 .. +1 %.
- `PB_SQ` (1: 8-way sum of squares + 4-way S1 / S2; 2: the whole row's sum per thread, no barrier between the halves; not bitwise, dz 2e-3):
  phase A 0.94 -> 0.88 us, period unchanged; +0 .. +5 % alone.
- `PB_NS` (the MMA thread's poll with __nanosleep 32 / 100 / 250 ns): level.
- finalize_red_k's dWf blocks with Wb / ln_pair.weight read before griddepcontrol.wait and a 5-step tree for the 32 group sums (fp64 error
  2.6e-7 -> 9e-8): part 4 alone 3.86 -> 3.91 us (L2), 6.77 -> 6.67 (flushed): level -- the kernel's ~4 us is not its reductions.

Taken: 24-head training (r147, d_pair 128) with finalize_red in its three parts per stream (`_fsplit`, L 256 / 384 / 640 / 768) and the pair
backward reserve 16 at L 256 (A / B, one launch -> parts: L 256 79.9 -> 76.1 us, L 384 114.7 -> 111.5, L 640 201.4 -> 197.6, L 768 248.7 ->
247.0; L 512 149.7 -> 148.5 by A / B but 149.5 -> 153.7 in the measurer: not routed; L 128 +0.4 / +1.3: not routed; 12 heads L 256 +1.1: not
routed). Measurer (B29 -> A29b): r147 L 256 81.8 -> 79.9, L 384 116.5 -> 114.8, L 640 202.6 -> 197.1, L 768 250.7 -> 249.9 us. Gradients 2-4e-7
(relative to their max) from the one-launch form (the partial-row count changes with the reserve); determinism `det2.py` r147 L 256 / 384 /
512 / 768: bitwise over three runs. `apb_rows.cu` untouched.

Still above 0.667 and why: master train 16 / 12 heads L 256-512 (0.70-0.74): the backward tail is small_core_bwd -> the pair backward
(18.4 us on 100 SMs, epilogue-bound at a fixed per-SM rate) -> finalize part 4 (5.8 us in the step, 3.9 alone); r147 24 heads (0.72-0.77):
no small_core_bwd / pair_bias_fwd2 at 24 heads (both measured slower, v27 / v28), bias transpose + dkv / dqb + qkv_bwd before the pair
backward. Master eval (r148 / r41 L 256 front, r147 24 heads, r41 / r146 L 384): unchanged; item (3) not built -- the front's v24 stamps
show first-touch latency (two dependent round trips), not the transfer, and the pair backward's measurements above found no gain from finer
stages.
