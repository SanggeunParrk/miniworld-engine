# Atom DiT classes on B200 (sm_100a, card `b200_sm100`)

## Scope

The `"cuda"` backend on an NVIDIA B200 (capability 10.0) of the atom DiT classes in `modules/atom_dit.py`:

| class | B200 route | `cuda:b200_sm100` |
|---|---|---|
| `AtomDiTProtenix` | the tcgen05 local DiT (`kernels/integrations/local_dit_b200`; chained K / V AdaLN, `pad` / `slide`, bf16 / master / fp32); B > 1 folds in one inference call | 4 |
| `AtomDiTOpenFold3` | the tcgen05 local DiT (parallel K / V AdaLN, `pad` / `slide` / `slide_qblock`, `mask_form="keys"` / `"af3"`) | 4 |
| `AtomDiTBoltz2`, `AtomDiTRoseTTAFold3`, `AtomDiTChai1` | the tcgen05 local DiT's variant builds (`local_dit_b200._run_variant`, round 3: IDENT / PARS / KQLN / UP, `pad` / `slide`, bf16 / master / fp32, inference and training); the calls it refuses on the `dit_variants` sm_80 path | 3 |
| `AtomDiTESMFold2` (= `AtomDiTMiniWorld`), `SWAAtomTransformer` | the fused SWA block's sm_100a stages (`kernels/swa_dit/cuda/sm100`; bf16 inference on row kernels with the modulation inside: few-tile up to 4736 rows, many-tile above); training as one Function with the modulation (`_SWATrainB200`); global attention: forward on the native `gattn_fwd.cu` (inference and training), backward on `gattn_bwd.cu`; split builds where whole items leave SMs idle | 11 (`SWAAtomTransformer` 16) |
| `AtomDiTESMFold2Pair` | inference: the tcgen05 ESMFold2 block (`kernels/integrations/local_dit_esm_b200`, bf16 / master / fp32); training: `kernels/integrations/local_dit_esm.py` sm_80 path built for sm_100a | 2 |

Requirement rows: every row of these classes in `requirements/requirements_module.csv` (46 rows: `atom_single`,
`atom_single_shared_cond`, `atom_folds`, `atom_folds_shared_cond`, Chai-1's `atom_single_fp32_cond` / `atom_single_fp32`; lengths
1024-16384 eval, 1024-8192 train; `float32 | master`). No row was added or changed.

## Round 3 (2026-11-17, branch `b200r3-swa-global`): a native global attention forward in place of FlashAttention-4's

Classes: `SWAAtomTransformer` (row 132) and `InputFeatureEmbedder` (rows 133, 134, 136) with `swa_window_size=1000000`; both
`cuda:b200_sm100` 6. `AtomDiTESMFold2` has no global-attention row.

### What changed

- **`kernels/swa_dit/cuda/sm100/gattn_fwd.cu`** (new, sm_100a only): the global attention forward on tcgen05. Query-tile pairs (two
  softmax warpgroups of 128 rows on the same 64-key K | V stages, the schedule of `augmented_attention/cuda/sm100/attn_inf.cu`), S and P
  double-buffered in TMEM, the two warpgroups taking turns on the exponentials, persistent grid, PDL. Specific to this block:
  - **no running maximum**: the heads are RMS-normalized without a weight (`SWA3DRoPEAttention`: `F.rms_norm`, no affine), so
    `|q . k| scale <= sqrt(32)` and `P = 2^(s scale log2 e)` is taken directly (exponent in [-8.2, 8.2]): no row maximum, no O rescale.
    The contract is written in the kernel header and in `sm100._native_global`;
  - the row sum on the FP32 pipe (a P x ones MMA was tried: +4 % -- the tensor pipe is the busier one);
  - one commit fewer per block: QK(G + 2)'s `s_full` commit follows PV(G) and doubles as the P-buffer-free signal;
  - every row of O written (zeros past seqused): the callers no longer zero O first (`sm100.global_out`);
  - lse in FA4's convention (natural log, [N, H, S]); measured compatible with FA4's backward (`_tools/gchk.py`: dQ / dK / dV from
    the native forward within 0.4 - 0.7 % of FA4's own), but see the training note below.
- **Route** (`sm100.attn_global_fwd` -> `_native_global`): B200 with FA4 installed, S a multiple of 128, **inference only**
  (`sm100.block_fwd` / `block_fwd_c` and the TF32 block's global branch pass `native=not save`). Training keeps FA4's forward and
  backward: with the native forward, FA4's backward put S48 master training at YELLOW 1.13 / 1.09 x tol (L 1024 / 4096) where the
  tokens are GREEN 0.96 / 0.95 -- a few % of the step for an accuracy step back, not taken.
- The cubin is in the sm100 prebuild plan (`sm100.CUBINS["gattn_fwd"]`; `_prebuild --arch sm100 --list` shows `swa_dit_sm100/gattn_fwd`).

### FA4 against the native forward, the attention alone (`_tools/gb.py`; graph timing, L2 flushed; H 4, D 32; us)

FA4 needs O zeroed before it (it skips the rows past seqused): "FA4 + zero" is what a block paid. Floor = exponentials at 16 / clk / SM.

| N x S | FA4 | FA4 + zero | native | MUFU floor | | N x S | FA4 | FA4 + zero | native |
|---|---|---|---|---|---|---|---|---|---|
| 1 x 1024 | 22.5 | 22.5 | 20.5 | 0.9 | | 5 x 5120 | 190.5 | 194.0 | 180.8 |
| 1 x 2048 | 32.8 | 34.8 | 30.7 | 3.7 | | 5 x 8192 | 475.7 | 480.3 | 460.8 |
| 1 x 4096 | 55.3 | 57.3 | 55.3 | 14.9 | | 48 x 1024 | 106.5 | 110.6 | 81.9 |
| 1 x 6144 | 79.8 | 81.9 | 77.8 | 33.6 | | 48 x 2048 | 311.3 | 319.4 | 262.3 |
| 1 x 8192 | 102.4 | 104.4 | 102.4 | 59.7 | | 48 x 4096 | 1047.6 | 1062.8 | 951.8 |
| 5 x 1024 | 22.5 | 24.6 | 20.5 | 4.7 | | 48 x 8192 | 3928.0 | 3954.7 | 3743.7 |
| 5 x 4096 | 155.7 | 159.7 | 147.5 | 74.6 | | | | | |

Never slower than FA4 + zero (0 - 26 %); at N 1 S >= 4096 it ties FA4 itself. Accuracy against fp64 (`_tools/gchk.py`, full and
ragged seqused incl. 0 and 1): O max error 4 - 7.5e-4 (FA4 3.6 - 7.8e-4), lse error 1e-6 (FA4 5e-6); dead rows exactly zero.
Determinism: the same inputs twice -> the same bits (every case). Stress: 3000 graph replays per shape alternating full / ragged seqused,
0 mismatches (1 x 1024, 1 x 4096, 5 x 2048, 48 x 1024, 3 x 3072).

### Where the time goes (probes in the development copy, N 1 S 8192)

The kernel is not exponent-bound: halving the MUFU work (`ex2.approx.f16x2`) changed nothing (106 -> 108 us), no exponentials at all
84 us, no softmax work at all (MMA + TMA stream) 53 - 65 us, no MMAs and no softmax 41 us -- and that floor did not move with deeper
K | V rings (6 -> 20 stages), half the TMA bytes, cluster multicast of K | V (CL 2 / 4, kept as a build option, not routed: 0 - 2 % at
N 1 - 48, worse at N 5 S 1024) or plain arrives in place of the commits. The per-block control chain (barrier waits / commits of the
MMA warps, ~600 clk per 64-key block) is the floor; FA4 sits at the same place. A larger key block (BN 96 / 128 with P aliasing S in
TMEM) or an M 256 pair (cta_group::2) would halve the per-key chain; not done in this round.
Also found: keeping the `tcgen05.fence::after_thread_sync` before the exponent turn is worth 4 - 6 % (without it the compiler moves the
turn's arithmetic ahead of the named barrier); and one CTA per SM must be forced (the whole shared memory), or two CTAs share an SM's
TMEM and the persistent grid leaves SMs idle (+4 %).

### Measured (B200 GPU 1, `dev.sh -x`, the measurer on the global rows; the same tree with `NATIVE_GLOBAL` off as "before")

Records `$W/b2511/work/r3-swa-global/spd_off.jsonl` (FA4) / `spd_on.jsonl` (native), ratios against the H100 `cuda` tokens:

| class | dtype mode | cases | above 0.667 before -> after | worst before -> after |
|---|---|---|---|---|
| `InputFeatureEmbedder` | master eval (3 rows) | 24 | 16 -> 9 (L 4096 - 6144) | 0.827 -> 0.757 |
| `InputFeatureEmbedder` | float32 eval | 8 | 6 -> 6 | 0.892 -> 0.881 |
| `SWAAtomTransformer` | master eval S 5 | 8 | 6 -> 4 | 0.848 -> 0.722 |
| `SWAAtomTransformer` | float32 eval S 5 | 8 | 6 -> 5 | 0.797 -> 0.797 |
| both | training (FA4, unchanged) | 38 + 18 not served | unchanged | |

Per case (ms, before -> after): IFE master linear L 1024 0.154 -> 0.119, L 4096 0.266 -> 0.246, L 8192 0.440 -> 0.418; SWA master
L 1024 0.055 -> 0.047, L 4096 0.211 -> 0.191, L 8192 0.558 -> 0.532. No case slower than H100 among the eval rows.
Accuracy (`acc1.jsonl` / `acc2.jsonl`, L 1024 / 2048 / 4096): every eval case GREEN or YELLOW no worse than the B200 tokens (fp32
eval L 4096: YELLOW 1.25 against the token's 1.40); training identical to the tokens (FA4 path).

### Still above 0.667, and why

- master eval L 4096 - 6144 (IFE, SWA L 4096 / 6144): the attention is still ~60 % of a block there and runs at the per-block control
  floor described above; the H100's FA3-class kernel at D 32 is relatively better at these sizes. Needs the larger key block / M 256.
- float32 eval: the TF32 row stages and the fp32 -> bf16 casts around the attention dominate; the attention change moves 1 - 3 %.
- training (S48 SWA, IFE L 7168 / 8192): FA4's backward is ~70 % of the step; a native deterministic backward on tcgen05 (dQ / dK dV
  passes on the no-max softmax) would be the lever.

### Part 2 (2026-11-17, swa-global): a native deterministic global attention backward; master one-sample training on the TF32 stages

`cuda:b200_sm100` 10 for `SWAAtomTransformer` and `InputFeatureEmbedder` (9 left to the parallel window branch `b200r3-swa`).

What changed:
- **`kernels/swa_dit/cuda/sm100/gattn_bwd.cu`** (new, sm_100a only): the global attention backward on tcgen05, deterministic, in place
  of FlashAttention-4's (`sm100.GattnBwd`, `attn_global_bwd_native`, route `native_global_train`: B200 with FA4 installed, S a multiple
  of 256). Key-major items (sample, head, key pair: two 128-key tiles, one per softmax warpgroup, on the same 64-query blocks),
  S^T / dP^T in TMEM, P^T to TMEM and dS^T to shared memory, dV += P^T dO and dK += dS^T q accumulated in TMEM, dQ_b = dS K per block
  into a double-buffered TMEM slot. dQ is reduced deterministically: the items of a head add their dQ_b into an fp32 accumulator in a
  fixed order enforced by a per-(n, h, query block) counter (ld.acquire / red.release, FA4's deterministic scheme), but each item starts
  its query sweep at a rotated offset so that in the steady state no item waits; rank 0 stores, middle ranks bulk-add
  (`cp.reduce.async.bulk .add.f32`), the last rank scales, writes dQ bf16 and resets the counter. Every row written (zeros past
  seqused): no zero fills. A quarter of the exponentials run on the FMA pipe (FA4's degree-3 polynomial).
- Training forward stays FA4's (its row-max-subtracted P): with the native no-max forward the S48 master gradients were YELLOW
  1.10 - 1.13 x tol with either backward (`NATIVE_TRAIN_FWD` stays False); FA4's natural-log LSE is converted to the negated log2 the
  kernel adds. The fp32 block's global backward (`tf32_bwd._global_attn_bwd`) takes the same kernel (`swa_gdelta_tf32`: dO cast and
  D = sum dO16 o in one launch).
- **Master one-sample global training at L 7168 / 8192** (embedder) now runs on the TF32 row stages + the native backward
  (`atom_dit._B200_MT_TF32_GLOBAL`, `_b200_master_tf32`) instead of the bf16 stages: faster and more accurate (below).
- Tried, not taken: `gattn_fwd.cu` with 96-key blocks (P aliased onto S in TMEM; `-DBNK=96` kept as a build option): 0 - 4 % on the
  attention alone (N 1 S 6144 77.8 -> 75.8 us, N 5 S 4096 147.4 -> 147.4), no case crosses 0.667. 128-key blocks do not fit the
  four-S-buffer TMEM layout (would need single-buffered S). The same TF32 master route below 7168 rows: serves the refused embedder
  cases faster than Triton (0.75 - 0.97 x) but slower than the H100 (1.08 - 1.17 x) and linear L 1024 at 1.48 x tol (H100 1.36,
  Triton 1.10): kept refused.

The backward alone (`_tools/gbw.py`; graph timing, L2 flushed; us; FA4 includes the three zero fills it needs):

| N x S | native bwd | FA4 bwd | | N x S | native bwd | FA4 bwd |
|---|---|---|---|---|---|---|
| 1 x 1024 | 36.8 | 47.1 | | 48 x 1024 | 180.0 | 294.9 |
| 5 x 2048 | 114.4 | 163.8 | | 48 x 2048 | 561.2 | 1038.4 |
| 1 x 7168 | 176.2 | 290.7 | | 48 x 4096 | 2085.9 | 3346.3 |
| 1 x 8192 | 200.6 | 327.7 | | 48 x 8192 | 8806.2 | 9747.2 |

dQ / dK / dV against fp64 equal to FA4's (max error 2.4 - 4.5e-3 of max |ref|, full and ragged seqused incl. 0 / 1); same bits twice
on every case; 100 graph replays alternating full / ragged seqused per shape, 0 mismatches. Module determinism (`_tools/det.py`, 30
cases incl. embedder master L 7168 / 8192 and fp32 L 8192 training): same bits over 3 runs.

Measured (B200 GPU 1, `dev.sh -x`, every global case of both classes, L 1024 - 3072 / 4096 - 6144 / 7168 - 8192 in separate runs;
records `$W/b2511/work/r3-swa-global3/fin_*.jsonl`, `fin_ifem_7168_8192.jsonl`; ratios against the H100 `cuda` tokens; "before" = the
same tree with the FA4 backward, `st_fa4_*.jsonl`, for training, and part 1 above for eval):

| class | dtype mode | cases | above 0.667 before -> after | worst before -> after |
|---|---|---|---|---|
| `SWAAtomTransformer` | master train S 48 | 8 | all (5 of 5 lengths timed on FA4) -> 8 | 0.853 -> 0.741 (L 1024 - 8192: 0.67 - 0.74) |
| `SWAAtomTransformer` | float32 train S 48 | 8 | 0 -> 0 | 0.615 -> 0.586 |
| `SWAAtomTransformer` | master eval S 5 | 8 | 4 -> 4 | 0.722 -> 0.723 |
| `SWAAtomTransformer` | float32 eval S 5 | 8 | 5 -> 5 | 0.797 -> 0.754 |
| `InputFeatureEmbedder` | master train L 7168 / 8192 | 6 | 6 -> 6 | 1.048 -> 0.830 (none slower than the H100 any more) |
| `InputFeatureEmbedder` | float32 train | 8 | all (5 of 5 lengths timed on FA4) -> 6 (L 7168 / 8192 now 0.62 / 0.61) | 0.885 -> 0.805 |
| `InputFeatureEmbedder` | master eval | 24 | 9 -> 9 | 0.757 -> 0.767 (unchanged path) |
| `InputFeatureEmbedder` | float32 eval | 8 | 6 -> 6 | 0.881 -> 0.848 |

Training steps (ms, FA4 backward -> native): SWA master S 48 L 1024 1.016 -> 0.913, L 4096 6.344 -> 5.108, L 8192 17.42 -> 16.49;
SWA fp32 S 48 L 4096 6.825 -> 5.616; embedder master L 8192 2.13 -> 1.65 - 1.69, fp32 L 8192 1.973 -> 1.620. Ladder: no cuda case
slower than Triton (worst cuda / triton 0.84).

Accuracy (`acc_t.jsonl`, `acc_ife_fin.jsonl`, `acc_e32.jsonl`; e / tol of the worst tensor): SWA master training GREEN 0.96 / 0.92 /
0.95 at L 1024 / 2048 / 4096 (= the tokens; H100 0.85 / 1.14 / -); SWA fp32 training YELLOW 3.13 - 3.58 (tokens 3.03 - 3.58, H100
3.90 - 4.62). Embedder master L 7168 / 8192: linear GREEN 0.85 / 0.99 (H100 1.28 / 1.33), relu 1.16 / 1.10 and 1.14 / 1.19 (H100
1.64 / 1.82 and 1.62 / 1.48) -- the relu L 7168 case the FA4 route had at 1.98 x is now below the H100. fp32 eval unchanged or better.
Open (not from this change, identical with FA4's backward): embedder fp32 training L 7168 / 8192 YELLOW 2.45 / 2.51 x tol against
the B200 tokens' 1.77 / 2.03 (H100 2.22 / 2.03, Triton 2.22 / 2.63): the worst tensor is a Wqkv / to_atom_single_cond gradient.

Still above 0.667, and why:
- SWA master training S 48 (0.67 - 0.74): the attention backward is now ~40 % of the step; the backward runs at ~2.9 x the MUFU floor
  (the per-block control chain, as the forward) and FA4's forward at the same place. Next levers: the native forward in training (needs
  a max-subtracted variant to keep the gradients GREEN), a 2-CTA (M 256) backward.
- Embedder master training L 7168 / 8192 (0.81 - 0.83) and fp32 training L 1024 - 6144 (0.67 - 0.81): one sample, three blocks; the
  TF32 row stages and the attention are each about half; small grids (S / 256 items per head x 4 heads) leave SMs idle in the backward.
- master eval L 4096 - 6144 and the fp32 eval rows: as part 1 (the forward's per-block control floor; the TF32 row stages and casts).

H100 audit (global attention): the H100 runs FlashAttention-4 (sm_90 build) forward and backward between composed / native stages
(`dispatch._h100_global*`). B200: the same stage structure; forward native (inference) / FA4 (training), backward native for all
training -- beyond the H100 there. The H100's fp32 inference fusion (`_swa_dit_fwd_fp32_global`: fp32 qkvg / out-projection kernels
around FA4, four launches) corresponds to the B200's TF32 row stages + `GlobalGlue` (one cast launch, no O zero fill now).

### Part 3 (2026-11-17, swa-global): native training forward, split global attention (forward and backward), embedder master training from 5120 rows

`cuda:b200_sm100` 13 for `SWAAtomTransformer` and `InputFeatureEmbedder` (12 is the window branch's). Records
`$W/b2511/work/r3-swa-global4/` (`bef_*` / `aft_*`: the same tree with the training forward on FA4 and no split / with both; `mtg_*`:
embedder master training on the TF32 route from 1024 rows; `fin_*`: the final tree after the merge of base 8d27e53b7; accuracy
`acc_t_new`, `acc_t2_new`, `acc_e_new`, `acc_mtg`; determinism `det_new.log`, `det_fin.log`).

What changed:
- **Training forward on the native `gattn_fwd.cu`** (`sm100.NATIVE_TRAIN_FWD = True`; it saves the negated log2 LSE the native backward
  reads). The S48 master `gate_proj.weight` gradient is YELLOW 1.13 / 1.09 x tol at L 1024 / 4096 on it (FA4's forward: GREEN 0.96 /
  0.95). That is seed noise, not a defect: FA4 (bf16 V) keeps its first block's row maximum (rescale threshold 8), so its P is rounded
  like the native one; `_tools/accdiag.py` (the measurer's gold / reference / rule, three seeds) gives the native forward's worst tensor
  0.93 / 0.94 at seeds 1 / 2 where FA4's is 0.94 / YELLOW 1.02 (`grad:c`). Inside the engine standard (YELLOW <= 5 tol). Tried, not
  routed: **fp16 P** (`-DP16=1`: P in fp16, each stage's V converted bf16 -> fp16 in shared memory by the idle warp 3, `vc_full`
  barrier; O error rms 1.13e-4 -> 7.95e-5) moved L 1024 only to 1.07 and costs 2 % of the kernel; a mixed f16 x bf16 MMA (`-DMIXV=1`)
  is an illegal instruction on sm_100a.
- **Split mode** of both kernels (`-DSPLITB=1` builds `gattn_fwd_s` / `gattn_bwd_s`; host `GattnFwd.splits` / `GattnBwd.splits`,
  `_split_pays`): where whole items (forward: tile pairs, backward: key pairs) leave SMs idle -- one sample (S / 64 items for 148 SMs)
  or five samples at L 2048 / 4096 -- the (item, key block) / (item, query step) sequence of all items is cut into 148 equal contiguous
  ranges, one per CTA (thread 0 builds the CTA's piece list from a per-sample prefix in shared memory, after `griddepcontrol.wait`).
  An item cut by a range boundary runs as pieces on consecutive CTAs; each piece writes its fp32 partial (forward: unnormalized O and
  row sum -- no running maximum, so partials add directly; backward: dK | dV) to a workspace slot (CTA, first / last piece of its
  range, warpgroup) and counts on a per-(item, warpgroup) counter; the piece completing the count adds every partial from the
  workspace in CTA order (deterministic whoever comes last), stores, and resets the counter. The backward's dQ adders of a query block
  are ordered by (step within their CTA's range, CTA): every CTA runs its range in that order and waits only on smaller keys (no
  waits-for cycle; every CTA resident), and the CTAs advance together, so it is the arrival order in the steady state.
  Built separately because the split code merely present costs the whole-item loop ~10 % (126 -> 168 registers); the backward's
  combine reads its own partial back from the workspace too (an inline accumulator spilled 400 - 700 B in the row loop, a
  `__noinline__` call spilled the kernel's live registers).
  Routed where the busiest whole-item CTA has >= 1.3 x the even share and >= 12 blocks per CTA remain (`SPLIT_GAIN`,
  `SPLIT_MIN_BLOCKS`): one sample L 3072 - 7168 (forward and backward), five samples L 2048 / 4096 (forward).
- **Embedder master training from 5120 rows** on the TF32 row stages (`atom_dit._B200_MT_TF32_GLOBAL_MIN_ROWS = 5120`, was 7168):
  6 of the 18 refused cases served.
- `tf32_fwd.FA2_TRAIN_FWD` (the window branch's FA2 forward for fp32 one-sequence training, switched off by the coordinator) removed:
  the native training forward makes it dead code.

The attention alone (`_tools/gcmp.py`, `_tools/gbt.py`; graph timing, L2 flushed; us; whole items -> split build):

| N x S | forward | backward | | N x S | forward | backward |
|---|---|---|---|---|---|---|
| 1 x 1024 | 20.4 -> 38.8 (not split) | 36.8 -> 71.6 (not split) | | 1 x 7168 | 90.0 -> 86.0 | - |
| 1 x 2048 | 30.6 -> 34.7 (not split) | 59.4 -> 69.5 (not split) | | 1 x 8192 | 100.4 -> 104.4 (not split) | 200.7 -> 223.2 (not split) |
| 1 x 3072 | 42.9 -> 36.8 | 81.9 -> 67.5 | | 5 x 2048 | 55.2 -> 49.1 | 112.6 -> 94.2 |
| 1 x 4096 | 55.2 -> 47.0 | 106.4 -> 92.1 | | 5 x 4096 | 147.4 -> 123.0 | - |
| 1 x 6144 | 77.8 -> 69.5 | 153.6 -> 131.1 | | 2 x 8192 | 190.5 -> 182.2 (not split) | - |

Checks (`_tools/gsp.py`, `_tools/gbsp.py`): O / LSE and dQ / dK / dV against fp64 equal to the whole-item kernels' (full and ragged
seqused incl. 0 / 1), split vs whole within 2e-4 (forward O) / 5e-3 relative (backward); same bits twice on every case; 100 - 200 graph
replays per shape alternating full / ragged seqused, 0 mismatches, every counter zero afterwards. Found on the way: two launches of
the same split kernel on two streams at once share its workspace and counters (one test ran an eager launch on a side stream without
`wait_stream`): the counters then stay off for good. The engine never does that (one stream per forward / backward; the measurer's side
stream waits on the main one), as for the backward's dQ counters since part 2.

Measured (B200, `dev.sh -x`, the measurer on every global case of both classes; L 1024 in its own run, then 2048 - 3072, 4096 -
6144, 7168 - 8192; ratios against the H100 `cuda` tokens; "before" = the same tree with the FA4 training forward and no split, i.e.
part 2 on the current base, `bef_*`; "after" = the final tree, `fin_*`):

| class | dtype mode | cases | above 0.667 before -> after | worst before -> after | refused before -> after |
|---|---|---|---|---|---|
| `SWAAtomTransformer` | master train S 48 | 8 | 8 -> 6 (L 2048 / 3072 now 0.65) | 0.743 -> 0.733 | 0 -> 0 |
| `SWAAtomTransformer` | float32 train S 48 | 8 | 0 -> 0 | 0.587 -> 0.580 | 0 -> 0 |
| `SWAAtomTransformer` | master eval S 5 | 8 | 4 -> 3 (L 2048 0.69 -> 0.66, L 4096 0.66 -> 0.59) | 0.722 -> 0.720 | 0 -> 0 |
| `SWAAtomTransformer` | float32 eval S 5 | 8 | 5 -> 5 (L 2048 / 4096 0.74 -> 0.69 / 0.68) | 0.752 -> 0.775 (L 1024, unchanged path: 2 us timer noise) | 0 -> 0 |
| `InputFeatureEmbedder` | master train | 24 | 6 -> 12 (6 newly served) | 0.827 -> 0.867 (L 6144, newly served) | 18 -> 12 |
| `InputFeatureEmbedder` | float32 train | 8 | 6 -> 4 (L 3072 / 4096 0.80 / 0.77 -> 0.74 / 0.70) | 0.800 -> 0.771 | 0 -> 0 |
| `InputFeatureEmbedder` | master eval | 24 | 9 -> 3 (only L 6144: 0.68 - 0.70) | 0.767 -> 0.697 | 0 -> 0 |
| `InputFeatureEmbedder` | float32 eval | 8 | 6 -> 6 (L 3072 - 6144 each 5 - 8 % lower) | 0.854 -> 0.804 | 0 -> 0 |

No case slower than the H100. Accuracy (every light inside the engine standard, no RED): S48 master training as above (GREEN 0.92 at
L 2048, YELLOW 1.13 / 1.09 at L 1024 / 4096), fp32 training YELLOW 3.16 / 3.53 / 3.33; embedder master training L 5120 / 6144 (newly
served) YELLOW 0.99 - 1.27 / 0.80 - 1.42 x tol (H100 1.18 - 1.58 / 1.16 - 1.58, Triton 1.14 - 1.50 / 1.24 - 1.63), L 7168 / 8192
0.86 - 1.25 / 1.01 - 1.38; embedder fp32 training L 7168 YELLOW 2.95 (was 2.45 on FA4's forward; H100 2.22), L 8192 1.94; every eval
case unchanged (eval forward on the same kernels). Training determinism: same bits over 3 runs on all 30 cases of `_tools/det.py`.

Still above 0.667, and why:
- SWA master training S 48 L 1024 and 4096 - 8192 (0.67 - 0.73): 768 - 6144 items fill the grid, so neither split helps; the step is
  ~40 % attention backward at ~2.9 x its MUFU floor (the per-block control chain). The 2-CTA (`cta_group::2`) backward was not built
  this round: the time went to the split modes, which serve more cases; it remains the lever (a pair of CTAs on 256 keys x 64
  queries per MMA: half the MMA issues and B-operand loads per key, the same row-thread work).
- Embedder master training L 1024 - 4096 (12 refused): on the TF32 route with both splits it is 1.03 - 1.11 x the H100 (`mtg_*`),
  still slower -- at 1024 - 4096 rows the TF32 row stages (not attention) are most of the step. Embedder master training L 5120 - 8192
  (0.79 - 0.87) and fp32 training L 1024 / 2048 (0.75 / 0.77, below the split threshold: 6.9 blocks per CTA): the TF32 row stages.
- Eval at L 1024 (S 5 / one sample) and L 6144 / 8192 S 5: whole items already fill the grid (or pieces would be too short); the
  forward's per-block control floor. The M 256 (`cta_group::2`) inference forward and the fp32 eval casts folded into the attention
  kernel were not built this round.

H100 audit rows (global attention, part 3): the H100 serves these one-sample cases through FlashAttention-4's sm_90 build with whole
items; the B200 split modes have no H100 counterpart (beyond it). Training forward: H100 FA4 (sm_90) -> B200 native
(`gattn_fwd.cu`), same stage structure.


### Part 4 (2026-11-17, swa-global): fp32 inference folded into the attention kernel; the backward / forward limits measured

`cuda:b200_sm100` 15 for `SWAAtomTransformer` and `InputFeatureEmbedder` (14 and 16 are the window branch's). Records
`$W/b2511/work/r3-swa-global5/` (`fin7_*`: the final tree after the merge of base f1bccc33b, global-attention cases only, `_tools/spd_ab.py GONLY=1`
-- the window cases of the same rows are the window branch's; `gf32*`, `gfp*`, `gbv*`, `ab4*`: the kernel / block experiments; accuracy `acc_e32c`). Rejected code:
`docs/b200/rejected/swa_global_part4_*.diff`.

What changed:
- **fp32 global inference on `gattn_fwd32`** (`gattn_fwd.cu -DF32IO=2`, `sm100.F32IO_MODE`, `tf32_fwd.F32IO_GLOBAL`, at most
  `F32IO_MAX_ROWS` = 20480 rows N S): the attention kernel reads the TF32 qkvg stage's fp32 q heads itself (TMA into a raw staging slot, warp 3
  -- idle in the bf16 build -- rounds them to the bf16 64-B swizzled tiles, `q_full` by arrive) and writes O as fp32 holding the bf16
  values (per row, plain stores) for the TF32 FFN stage: the q cast and the `O16.float()` launch are gone; K | V still come from the
  cast launch (two arrays instead of three). The same bits as the cast route (`_tools/gf32.py`: O equal bitwise, whole and split builds,
  full and ragged seqused incl. 0 / 1; determinism; 100 graph replays alternating full / ragged per shape, 0 mismatches).
  `-DF32IO=1` (K | V read fp32 too) is kept as a build option, not routed: K | V are then rounded once per ITEM (each tile pair of a
  head re-reads every key block), and the single converter warp falls behind -- 5 x 4096 143 -> 170 us, 5 x 8192 491 -> 649 us.
- `gattn_fwd.cu` `-DNOMMA=1|2|3`: timing probe without the QK / PV MMAs (below).

The attention alone (`_tools/gf32.py`; graph timing, L2 flushed; us; "cast route" = cast launch + `gattn_fwd` + `O16.float()`):

| N x S | cast route | F32IO 2 | | N x S | cast route | F32IO 2 |
|---|---|---|---|---|---|---|
| 5 x 1024 | 28.6 | 28.6 | | 1 x 4096 | 55.2 | 53.0 |
| 1 x 1024 | 26.6 | 26.5 | | 1 x 6144 | 79.8 | 77.7 |
| 5 x 2048 | 61.4 | 61.3 | | 5 x 4096 | 143.3 | 139.3 |
| 3 x 3072 | 57.3 | 59.3 | | 5 x 8192 | 491.5 | 540.5 (not routed) |

The block (`_tools/ab4.py`, the same process alternating routes, graph, L2 flushed; us): SWA fp32 eval S 5 L 1024 67.5 / 69.5 ->
65.4 / 67.5, L 2048 106.5 / 110.5 -> 106.2 / 106.5, L 4096 225 / 227 -> 219 / 221, L 8192 584 -> 621 and, in the measurer, S 5 L 6144 0.410 -> 0.434 ms (hence `F32IO_MAX_ROWS`: N S <= 20480);
embedder fp32 eval L 4096 278 / 287 -> 275 / 275, L 6144 353 / 360 -> 348 / 349, L 1024 186 / 194 -> 188 / 190 (level).

**SWA fp32 eval L 1024 at 0.775 (part 3) explained**: not a code change -- an order effect of the measurer. Every case is timed right
after its PyTorch reference (0.436 ms a replay, 55 replays), and the B200's power-cap clock drop after that (BRIEF4's measurement
note) reads the ~70 us `cuda` step 3 - 6 % slow, more or less from run to run: part 3's runs 69.5 (`bef`) / 71.6 (`fin`) us, three
fresh runs of the row alone 71.6 / 71.6 / 71.6 us; the same step timed without a reference before it (`_tools/ab4.py`) 67.5 - 69.5 us.
The kernel did not get slower: part 2's `gattn_fwd.cu` (8d27e53b7) swapped into the same block gives 67.5 / 69.5 / 67.6 us against
the current 67.5 / 69.5 / 68.8, and the attention alone 20.4 us = 20.4 us at 5 x 1024. (The H100 token, 92.4 us, was timed by the same
measurer; at 67.5 us the case would be 0.73.) This part's F32IO route: 69.5 us in the final record (0.752).


Built, measured and not taken:
- **Global attention backward, row-thread side** (`swa_global_part4_bwd_dsb2_dqw_dqs.diff`; all bitwise equal to the default build,
  0 graph-replay mismatches; whole build, us):

  | build | 48 x 1024 | 48 x 4096 | 48 x 8192 | 5 x 2048 | 3 x 3072 |
  |---|---|---|---|---|---|
  | default | 176.1 | 2087.7 | 8848 | 113.0 | 88.1 |
  | DSB2: dS^T double-buffered (the 32 KB dK / dV staging freed: dK / dV per row from registers), P^T released by the dV MMAs alone | 180.2 | 2088.6 | 8798 | 114.6 | 88.1 |
  | DQW: a fourth warpgroup reads dQ_b out of TMEM (512 threads) | 231.1 | 2938.8 | 12694 | 149.5 | 116.5 |
  | DQW + DSB2 | 190.5 | 2169.8 | 9163 | 118.6 | 91.1 |
  | DQS: dQ_b read out every block by both tiles, half the columns each | 192.5 | 2325.5 | 9888 | 124.9 | 98.1 |
  | DQS + DSB2 | 192.5 | 2289.6 | 9686 | 123.4 | 96.2 |

  Knobs on the default (`gbv3b`; 48 x 1024 / 48 x 4096): PCNT 0 180.3 / 2143, PCNT 2 188.3 / 2284, PMOD 8 PCNT 3 180.0 / 2137, mbarrier
  back-off 32 ns 180.2 / 2131, no dQ-poll back-off 176.1 / 2087, ST 4 186.4 / 2258 (ST 8, NSTG 6: no longer fit the shared memory) --
  the default is the local optimum.
- **The 2-CTA (`cta_group::2`, M 256) backward, not built to the end -- its bound measured first.** Removing MMA groups from the
  default build (`PROBE`, part 2's probe bits; 48 x 4096 2088 us): no dQ MMAs 2062 (-1 %), no dK / dV MMAs 2024 (-3 %), no S^T / dP^T
  MMAs 2001 (-4 %), no dK / dV / dQ MMAs 1943 (-7 %); no row math 1618 (-22 %). A pair of CTAs on 256 keys halves the MMA issues and
  the q / dO B-operand reads per CTA but cannot remove the MMAs, so it is bounded well inside those 4 - 7 %; and it costs more than
  the halves: every tcgen05 instruction of a kernel must carry the same `cta_group` (ptxas; `bias_only_dit/cuda/bo_pvdpb.cu` notes it),
  so dQ = dS K would have to be an M 128 pair product too, whose K-dimension (the keys) must be common to both CTAs -- each CTA would
  need the peer's dS^T (16 KB a block through DSMEM) and K rows split by head dimension, a per-block cluster exchange on the very chain
  that bounds the kernel. The clock trace (`_tools/gbtr.py`, 48 x 4096, CTA 0) shows that chain on the row threads: s_full wait ~170,
  math ~1200, P^T / dS^T free wait ~170, stores ~450, and the dQ_b read-out (~600, every other block per tile) -- 2250 / 3000 cycles
  for even / odd blocks; DSB2 / DQW / DQS above attack those pieces and lose (DQW: four more warps polling on the row warps' SMSPs;
  DQS: the read-out's fixed latency paid twice a block).
- **M 256 inference forward, bound measured** (`-DNOMMA`, `_tools/gfp.py`; us): no QK MMAs / no PV MMAs / neither -- 5 x 4096 122.9 ->
  118.8 / 120.8 / 110.6, 5 x 6144 280.4 -> 270.4 / 276.6 / 252.9, 5 x 8192 458.1 -> 444.4 / 454.6 / 414.1, 1 x 6144 69.6 -> 67.5 /
  67.6 / 63.5, 48 x 4096 947 -> 923 / 941 / 863. With no MMA at all the forward is 8 - 10 % faster; a CTA pair keeps all the MMA work
  (it halves issues and the K | V shared-memory reads per CTA), so it is bounded inside that -- SWA master eval L 6144 needs 7 % of the
  block (~12 % of its attention), L 8192 4.5 % (~8 %). The forward is bound on the softmax side: its clock trace (`_tools/gtr.py`,
  48 x 4096) has ~1000 of the ~1400 cycles a block in the other warpgroup's exponential turn (MUFU: 16 384 exponentials a block-pair
  = 1024 cycles at 16 / clk / SM).
- **Forward: a share of the exponentials on the FMA pipe** (`swa_global_part4_fwd_poly_exp.diff`; the backward's degree-3 `ex2_poly2`,
  unmasked blocks only): 1 / 2 of 4 and 3 of 8 pairs all slower -- 48 x 4096 947 -> 1010 / 1061 / 979 us, 5 x 8192 458 -> 487 / 514 /
  475; O error unchanged (2.3e-3 rms). The FMA pipe is as loaded as MUFU there. Also slower or level: no exponential turns (PP 0) 48 x
  4096 985, 5 x 8192 480; ST 4 / 8, QR 3: level.

Measured (B200, `dev.sh -x`, the measurer on every global case of both classes, global kwargs only; L 1024 in its own run, then 2048 -
3072, 4096 - 6144, 7168 - 8192; ratios against the H100 `cuda` tokens; "before" = part 3's final records `r3-swa-global4/fin_*`):

| class | dtype mode | cases | above 0.667 before -> after | worst before -> after | refused before -> after |
|---|---|---|---|---|---|
| `SWAAtomTransformer` | master train S 48 | 8 | 6 -> 3 (L 6144 / 7168 / 8192: 0.669 / 0.681 / 0.715) -- the base's window part 8 (bf16 row stages), not this part | 0.733 -> 0.715 | 0 -> 0 |
| `SWAAtomTransformer` | float32 train S 48 | 8 | 0 -> 0 | 0.580 -> 0.580 | 0 -> 0 |
| `SWAAtomTransformer` | master eval S 5 | 8 | 3 -> 3 | 0.720 -> 0.721 | 0 -> 0 |
| `SWAAtomTransformer` | float32 eval S 5 | 8 | 5 -> 4 (L 4096 0.684 -> 0.666; L 1024 0.775 -> 0.752, L 2048 0.688 -> 0.675) | 0.775 -> 0.752 | 0 -> 0 |
| `InputFeatureEmbedder` | master train | 24 | 12 -> 12 | 0.867 -> 0.868 | 12 -> 12 |
| `InputFeatureEmbedder` | float32 train | 8 | 4 -> 4 | 0.771 -> 0.769 | 0 -> 0 |
| `InputFeatureEmbedder` | master eval | 24 | 3 -> 3 | 0.697 -> 0.711 (unchanged path; run-to-run) | 0 -> 0 |
| `InputFeatureEmbedder` | float32 eval | 8 | 6 -> 6 (L 3072 / 4096 / 5120 / 6144: 0.762 / 0.787 / 0.697 / 0.735 -> 0.740 / 0.754 / 0.676 / 0.711) | 0.804 -> 0.804 (L 2048, level) | 0 -> 0 |

No case slower than the H100. The S 48 master training steps fell 3 - 10 % with the base merge (L 1024 0.876 -> 0.786 ms, L 8192
16.31 -> 15.90 ms): the window branch's part 8, merged into the base; this part changes no training route.


Accuracy: the F32IO route gives the cast route's bits (above); `acc_e32c` (fp32 eval L 1024 - 6144, every row): the same lights as the cast route (bitwise equal O): SWA S 5 GREEN 0.90 / 0.99 at L 1024 / 2048, YELLOW 1.06 / 1.25 / 0.88 GREEN / 1.04 at L 3072 - 6144 (H100 GREEN 0.61 - 0.89, YELLOW 1.13 at 6144); embedder YELLOW 1.32 - 1.61 (H100 1.32 - 1.72, Triton 1.44 - 1.82) -- every light inside the engine standard, no RED. Training determinism (`det7.log`, 30 cases on the merged tree): the same bits over 3 runs, 0 bad. Training
paths unchanged (no training kernel or route changed in this part).

Still above 0.667, and why:
- SWA master training S 48 (L 6144 - 8192 after the base merge): the global attention backward is ~40 % of the step and bound by its row-thread chain
  (trace above), the forward by MUFU; the MMA side of either is worth at most 4 - 10 % of the kernel (probes), so neither the 2-CTA
  backward nor the M 256 forward reaches the 0.4 - 7 % of the step these cases need on its own. What is left: the bf16 row stages of the
  step (qkvg / FFN / out-projection forward and backward, the window branch's kernels) and the backward's grid at L 8192 (S / 256 = 32
  items per head: whole heads per round leave 20 of 148 SMs idle, 13.5 %; the split build that fills them is capped at 4 items per SM).
- fp32 eval (SWA L 1024 / 6144 / 8192, embedder): the TF32 row stages (qkvg, FFN, modulation) are now most of a block; the attention
  is at the bf16 kernel's time plus the K | V cast.
- master eval L 6144 / 8192 (SWA S 5, embedder): the forward's MUFU / chain floor (above).
- Embedder master / fp32 training: the TF32 row stages (part 3), unchanged.

H100 audit rows (part 4): the H100's fp32 global inference (`_swa_dit_fwd_fp32_global`) runs its fp32 qkvg / out-projection kernels around
FlashAttention-4 (four launches, bf16 operands made outside the attention); the B200 now reads q fp32 and writes O fp32 inside its
attention kernel (beyond the H100; K | V still one cast launch). The
H100 has no 2-CTA attention backward or M 256 forward (FlashAttention-4 sm_90: 1-CTA wgmma); not ported, bound measured above.

## Round 3 (2026-11-17, branch `b200r3-local-dit`): B200 `cuda` against H100 `cuda`

Target: on every requirement case both cards' `cuda` serve, B200 median <= H100 median / 1.5 (ratio b200 / h100 <= 0.667; graph timing, L2
flushed). Classes of this branch: `AtomDiTOpenFold3`, `AtomDiTProtenix`, `AtomDiTESMFold2Pair`. Measured on the B200 (GPU 1, `dev.sh -x`,
`measure.speed`, every length and sample count of the rows) against the H100 speed tokens; records `$W/b2511/work/r3-local-dit/`
(`spd_<Class>_f1.jsonl`, training of OpenFold3 / Protenix `spdT_<Class>_f2.jsonl`; accuracy `acc_<Class>_f1.jsonl`, `accT_<Class>_f2.jsonl`).

| class | cases | ratio > 0.667 before -> after | slower than H100 before -> after | worst ratio | geomean ratio |
|---|---|---|---|---|---|
| `AtomDiTOpenFold3` | 110 | 68 -> 0 | 53 -> 0 | 1.940 -> 0.593 | 0.931 -> 0.469 |
| `AtomDiTProtenix` | 132 | 67 -> 35 | 18 -> 1 | 1.852 -> 1.084 | 0.693 -> 0.555 |
| `AtomDiTESMFold2Pair` | 56 | 52 -> 40 | 30 -> 0 | 1.554 -> 0.998 | 1.016 -> 0.725 |

The ladder holds on every case of the three classes (`cuda` below `triton` and `pytorch`: 0 violations; `cuda` against `triton` 2.15 x at
least, geomean 9.3 / 9.2 / 11.7 x; against `pytorch` geomean above 11 x).

### What changed

- **Inference packs read by a capture** (`_capture.param_forms(capture_reads_eager=True)`: `_weights`, `_kv_pack`, `_weights32`): a
  CUDA-graph replay of an inference call ran the bf16 casts / cats / transposes of the weight packs (they were scoped to the capture);
  now it reads the pack the eager warm-up made. No backward transposes in an inference pack.
- **`slide_qblock` and `mask_form="af3"` on the tcgen05 block** (`AtomDiTOpenFold3`; they ran the sm_80 `mma.sync` builds: 1.2 - 1.9 x
  the H100 time): `slide.cuh` `slide_nreal(.., qb)` rounds the real atom count up to a whole query window, every attention kernel
  (`lattn_fwd / dq / dkv` and the `_tf32` twins) takes the flag; AlphaFold 3's mask form is a widened key mask (the class widens it).
- **B > 1 folds in one call** (`atom_folds`, inference): `lattn_fwd` `NF` reads fold f's pair-bias rows for its row block; one launch
  sequence for the F folds (was one block call per fold).
- **The conditioning tables in one launch** (bf16 inference, both classes): `cond_fwd.cu` `-DKV=1` (kernel `cond_kv`) adds a third grid row
  that makes the K / V AdaLN's raw tables `rn(rn(LN_gkv(c)) [Wkv_scale; Wkv_shift]^T)` beside `mod`. It replaces `local_cond_ln` + a cuBLAS
  product: Protenix 1024 atoms bf16 47.1 -> 38.9 us (a removed launch is worth ~4 us of a replay at this size: the timer floor is 4 us + 2 us
  a trivial kernel).
- **fp32 training backward: side streams.** The column sums of `dmod / dp / dmkv` (the bias gradients) with their per-weight copies, and
  the pair-bias backward with `d pair` and its two weights' copies, run on two side streams beside the products (`_fork` / `_join`,
  `swa_dit.cuda.sm100._side_streams`; CUDA-graph capturable). OpenFold3 `mask_form="af3"` 128 atoms: 317 -> 281 us; Protenix 1024 x 48:
  1264 -> 1196 us; 5120 x 48: 5175 -> 5144 us.
- **Weight gradients taken, not copied, by autograd.** `_fused_wgrads` returns a view of each gradient buffer: `AtomDw` keeps the
  launch's destination views until its next call, so `AccumulateGrad` saw a second reference and copied all 16 dense weight gradients
  every step (16 `Memcpy DtoD`). bf16 and fp32 training: Protenix 1024 x 48 fp32 1196 -> 1156 us, OpenFold3 bf16 training worst ratio
  0.563 -> 0.500.
- **`AtomDiTESMFold2Pair` inference on the tcgen05 ESMFold2 block** (`kernels/integrations/local_dit_esm_b200.py`,
  `b200_atom_local/esm.py`, `lattn_esm*.cu`, `lesm.cu`; the `-DRMS=1` builds of `b200_atom`'s row kernels and of `fused_tf32.cu`):
  `origin/b200/ladder-final`'s ESMFold2 block on the 2.5 module. bf16: `cond_fwd` RMS (tables and the K / V tables, one launch) -> `pre_fwd`
  RMS -> q / k RMSNorm + rotary -> pair bias by position -> rank-window attention -> `post_fwd` RMS; fp32: `silu` -> `gemm_tf32` ->
  `atom_pre_tf32` RMS -> q / k norm -> pair bias -> TF32 attention -> `atom_post_tf32` RMS. 1024 atoms: bf16 94.1 -> 44.9 us, fp32
  100.4 -> 67.5 us.
  - Accuracy fix that made the bf16 route usable: the branch's RMS row kernels stored the AdaLN scale as `rn(1 + s)` and multiplied in
    bf16 (three roundings); the reference module computes `((1 + s) x_n + b)` in fp32 and rounds once, and `rn(1 + s)` costs `s` its low
    bits. `-DRAWSC=1` builds (`pre_rms_raw`, `post_rms_raw`) read the raw `s` (the inference pack leaves the identity bias out) and do the
    AdaLN in fp32 with one rounding: bf16 inference went from 16 of 30 cases YELLOW (e / tol up to 1.43) to PyTorch's error (relative L2
    2.96e-3 against 2.97e-3; worst e / tol 1.13, the level of the H100 and Triton tokens).
  - Training stays on the sm_80 path (`local_dit_esm`): the tcgen05 training half is ported and deterministic
    (`local_dit_esm_b200.ROUTE_TRAINING = False`) but its gradients are further from the gold than that path's (e / tol 1.82 bf16, 1.92
    fp32 against 1.58 / 1.39) and it is slower up to 2048 atoms; its forward still has the `rn(1 + s)` tables (the `RAWSC` form needs the
    RMS backward kernels to read the raw scale).

Tried and reverted (no win): the pair-bias launch of an inference call on a side stream beside the conditioning / projection launches
(1024 atoms 38.9 us either way; one CTA per SM for it: 4096 atoms 51.2 -> 55.3 us, 16384 atoms 122.8 -> 139.4 us); the TF32 attention core
for ESMFold2's bf16 inference (q / k normalised into fp32: same error, + 2 us).

### Accuracy (`cuda`, lights against the fp64 gold; e / tol = the worst tensor's error over its tolerance)

| class | dtype, mode | GREEN / YELLOW / RED | worst e / tol | B200 before | H100 token | B200 Triton token |
|---|---|---|---|---|---|---|
| OpenFold3 | master eval | 38 / 0 / 0 | 0.99 | 38 / 0, 0.99 | 38 / 0, 0.94 | 37 / 1, 1.04 |
| OpenFold3 | float32 eval | 28 / 10 / 0 | 1.15 | 23 / 15, 1.15 | 28 / 10, 1.27 | 16 / 22, 1.18 |
| OpenFold3 | master train | 0 / 17 / 0 | 1.40 | 1 / 16, 1.38 | 0 / 17, 2.24 | 0 / 17, 1.36 |
| OpenFold3 | float32 train | 0 / 17 / 0 | 1.89 | 0 / 17, 1.88 | 0 / 17, 2.15 | 0 / 17, 1.55 |
| Protenix | master eval | 44 / 0 / 0 | 1.00 | 44 / 0, 1.00 | 41 / 3, 1.09 | 40 / 4, 1.15 |
| Protenix | float32 eval | 12 / 28 / 0 | 1.10 | 12 / 28, 1.10 | 27 / 13, 1.29 | 32 / 8, 1.13 |
| Protenix | master train | 0 / 24 / 0 | 2.42 | 0 / 24, 2.42 | 0 / 24, 1.78 | 0 / 24, 1.62 |
| Protenix | float32 train | 0 / 24 / 0 | 2.14 | 0 / 24, 2.14 | 0 / 24, 2.29 | 0 / 24, 1.51 |
| ESMFold2Pair | master eval | 11 / 9 / 0 | 1.13 | 18 / 2, 1.14 | 13 / 7, 1.16 | 14 / 6, 1.19 |
| ESMFold2Pair | float32 eval | 15 / 5 / 0 | 1.19 | 14 / 6, 1.22 | 9 / 11, 1.13 | 8 / 12, 1.20 |
| ESMFold2Pair | train (sm_80 path, unchanged) | 0 / 16 / 0 | 1.58 | 0 / 16, 1.58 | 0 / 20, 1.66 | 0 / 16, 1.36 |

No RED. Run-to-run determinism checked (two calls, bitwise equal outputs and gradients) on every path changed. ESMFold2Pair master eval
has more YELLOW cases than the sm_80 path had (9 against 2 of 20) at the same worst e / tol as the H100 / Triton tokens; four of them are
YELLOW where every reference token is GREEN or less far ((165, 12288), (165, 16384), (177, 2048), (177, 6144): e / tol 1.06 - 1.13), and
one float32 case ((177, 7168): 1.19 against 1.08 / 1.10).

### Still above 0.667, and what each would take

- **Protenix `pad` inference, bf16 (13 cases, 0.70 - 0.76) and fp32 (17 cases, 0.67 - 1.08); ESMFold2Pair inference (27 cases, 0.67 -
  1.00).** A replay is 5 (Protenix bf16), 6 (ESMFold2 bf16, Protenix fp32) or 7 (ESMFold2 fp32) launches; the H100 runs the same
  decomposition. Up to ~4000 atoms the time is launch count x the per-launch latency of a tcgen05 kernel (bf16 7 - 11 us each: weights by
  TMA into TMEM, one tile through LayerNorm -> MMA -> epilogue -> TMA store; fp32 `atom_pre_tf32` 13 - 21 us and `atom_post_tf32` 20 - 27 us:
  448 KB of weights streamed through a 3-slot ring per CTA). Protenix 1024 atoms bf16: 38.8 us against the 35.2 us target. Closing it needs
  one launch fewer on each path: the pair bias inside the attention kernel (the CTA computes `LN(z) Wb` for its bias tile), or `cond_fwd`
  inside `pre_fwd`; ESMFold2: the q / k norm + rotary in `pre_fwd`'s epilogue; fp32: `ln` / `silu` as the A-operand phase of the tables'
  GEMM, and a fused pre / post with finer units at few rows (the fused TF32 kernels are a serial latency chain).
  At 8192 - 16384 atoms the kernels are throughput-bound and the pair bias is the largest (16384 atoms bf16: pre 32, pair bias 32,
  attention 29, cond 23, post 21 us of 120 us; target 110): `local_bias_fwd` is latency-bound per thread (one 32-B row per loop trip, 14
  trips), two rows a trip or a TMA-fed version would take it towards the ~13 us its 100 MB of traffic costs.
- **Protenix fp32 training, 48 samples (5 cases, 0.67 - 0.71).** Throughput: 5120 atoms x 48 = 5075 us, of which the 16 TF32 products
  1557, `atom_dw` 784, `f32_adaln_bwd` 653, the three column sums + their reductions 418, the attention backward 566. The column sums re-read
  `dmod` (755 MB); accumulating them in the kernels that write `dmod` (per-CTA partial rows, as the bf16 row kernels do) would give the
  ~6 % the worst case lacks.
- **ESMFold2Pair training (13 cases, 0.67 - 0.94)** runs the sm_80 builds. The tcgen05 training half is in the tree (not routed): it
  needs the `RAWSC` AdaLN in the training forward and the RMS backward kernels for the accuracy, and the pack / gradient glue of its master
  path off the replay (a 1024-atom step: 79 launches, 55 of them casts, cats and copies) for the speed.

Not a requirement case, found while checking: with a slid window and a key mask whose real atoms end before the last query window (e.g.
1000 atoms, 976 real), the padding-only query window keeps the clipped keys where the reference reads the slid keys `[R - 128, R)` (the
A100 fix of 2026-11-13, `dit_variants edge.cu` tail kernels); only rows of padding atoms differ. `slide.cuh` `slide_moved` would have to
take the windows past the last real one into the edge CTAs' group.

### Changes outside this branch's classes

- `b200_atom/cond_fwd.cu` (`-DKV`), `pre_fwd.cu` / `post_fwd.cu` (`-DRAWSC`), `fused_tf32.cu` (`-DRMS`): new build options, default 0; the
  default builds' code is unchanged (their cubin keys change with the source text).
- `b200_atom/__init__.py`: `cond_fwd(.., g3=)`, `pre_fwd_kv(.., raw=)`, `post_fwd(.., raw=)`, `pre32 / post32(.., rms=)`: new keyword
  arguments, defaults as before. `b200_atom_local/slide.cuh` `slide_nreal(.., qb = 0)` and the attention wrappers' `qblock=` / `folds=`.
- `local_dit_b200._fused_wgrads` returns views (also read by `local_dit_esm_b200`).
- `prebuild_sm100.py`: the ESMFold2 inference cubins when the rows have `AtomDiTESMFold2Pair` eval cases; `cond_kv` through
  `_ROW_KERNELS`.
- `tests/cpu/test_capture_api.py`: `local_dit_esm_b200._ESM_CONST` listed as not weight-derived.
- Versions: `cuda:b200_sm100` 3 on `AtomDiTProtenix` and `AtomDiTOpenFold3` (their own entries; `_LocalAtomDiT`'s stays 2 for Boltz-2 /
  RoseTTAFold3 / Chai-1), 2 on `AtomDiTESMFold2Pair`. No class `version`, no other card's entry.


### Round 3, second part (2026-11-17, after the merges of the base's atom-dit rounds 3 / 3b)

Records `$W/b2511/work/r3-local-dit/` (`spdE_<Class>_f6.jsonl` inference speed, `acc_<Class>_f5.jsonl`, `f6_acc.jsonl` accuracy).

- **The windowed pair bias inside the conditioning launch** (bf16 inference of Protenix / OpenFold3; `cond_fwd.cu` `-DPB=1`, kernel
  `cond_kv_pb`, `cond_fwd(.., pb=)`): the grid is (SMs, 4); rows y < 3 are the tables' CTAs with x < gxc (the others leave at once), row
  y = 3 runs the pair bias rows (`b200_atom_local/lbias_sm100.cuh`, `lbias.cu`'s arithmetic, four rows in flight per thread) on the SMs the
  tables leave and then on all of them. One launch fewer in a replay (5 -> 4); the bias is bit-identical. `_cond_tables(pb=, made=)` hands
  the bias to `_pair_bias`; a call that finds the tables hoisted but not the bias makes the bias by the stand-alone launch.
- **The attention as a programmatic dependent launch** (`-DLATTN_PDL`, the base's variant-path build, now every bf16 inference call):
  `post_fwd` stages its weights while the attention runs. The `LATTN_PDL` cubin is now in the sm100 prebuild plan (it was not).
- Same session, Protenix master eval, graph, L2 flushed (us): 1024 atoms 41.0 -> 36.9 (PDL) -> 34.7 (+ fused bias); 2048: 47.1 -> 47.1 ->
  43.0; 4096: 55.3 -> 53.2 -> 49.2; 8192: 79.7 -> 76.4 -> 69.6; 16384: 122.9 -> 120.8 -> 114.7; 1024 x 5 shared conditioning: 49.2 -> 45.1
  -> 43.0.
- Tried and reverted: the four-rows-in-flight pair bias as the stand-alone launch (`pb_f` / `pb_f32`: training, fp32, the variant blocks).
  No gain for the bf16 pair at 16384 atoms and slower for the fp32 pair (Protenix fp32 inference 12288 atoms 137 -> 149 us, 16384:
  164 -> 176 us), so the stand-alone launches keep `lbias.cu` / `lbias_tf32.cu`.

**The padding-only query windows of a slid key window** (what the atom-dit agent saw as "N % 32 != 0 is wrong": `make_inputs` masks the
last 24 atoms, so 2096 / 2100 / 2128 atoms have a query window of padding atoms only; every real row was right, those rows read their
clipped keys where the reference reads `[R - 128, R)`): fixed in the third part below.

Still above 0.667 after this part (H100 >= 20 us on every case of these classes, so none is exempt):

- Protenix: 35 -> 28 cases above 0.667 (master eval 13 -> 7, worst 0.760 -> 0.723: 2048 / 3072 / 5120 / 16384 atoms and the 5-sample
  shared-conditioning cases at 1024 - 3072, 0.685 - 0.723; fp32 eval 17 -> 16, worst 1.084, path unchanged; fp32 training 5, unchanged;
  master training 0). OpenFold3: 0 of 110 (worst 0.591). At 2048 atoms
  the K / V third of the conditioning launch takes a second tile per CTA (64 tiles x 3 on 148 SMs); `cond_fwd` inside `pre_fwd` (the base's
  round 3b works on that fusion for the variant blocks) is the next launch to remove.
- Protenix fp32 eval: the fused TF32 pre / post kernels (the atom-dit agent is redesigning `atom_post_tf32`); fp32
  training at 48 samples, ESMFold2Pair inference and training: as written above, not touched in this part.

### Round 3, third part (2026-11-17, after the merge of base 11c7f17b0: merge commit b2ce12ab0)

Records `$W/b2511/work/r3-local-dit/` (`tail*.out`, `oddv.out`, `v1.out`, `san.out`, `spd_sl_{base,new}.jsonl`, `acc_esm_res1*.jsonl`).

- **The slid key window's padding-only query windows, all six attention kernels** (`b200_atom_local/slide.cuh`, new `ltail.cuh`).
  The reference gives every query window whose clipped keys end past the real count, *including every window of padding queries past
  the last real one*, the slid keys `[max(R - 128, 0), R)`. `slide_moved` now says exactly that (it stopped at the last real window, so
  those windows ran on their clipped keys); edge group 1 (`wf = max(2, last - 2)`, four windows) takes every moved window up to `wf + 3`
  (`slide_mine` / `slide_tail0`), and is live for every `R > 0` (a system under 64 real atoms had windows 2 / 3 on their clipped keys too).
  The windows from `wf + 4` on (the TAIL: padding queries only, none in any requirement case) are computed by the group-1 edge CTAs
  themselves after their own windows, on CUDA cores in fp32 (`ltail.cuh`: `tail_fwd` O / LSE, `tail_dq` dQ, `tail_dkv` dbias -- the CTA's
  samples summed in order, at most two sample ranges per head added onto the zeroed tensor, so exact -- and the slid keys' dK / dV added
  into the CTA's own rows of group 1's partial, which `lslide.cu` adds as before). No launch of its own, no host read (the real count is
  the mask's population count on the device), CUDA-graph capturable; the kernels take the raw q / k / v / dO / LSE / D / O / dQ pointers
  beside their TMA maps. Without a tail window nothing runs.
  - Accuracy: `tests/gpu/test_atom_dit_slide_tail.py` (the H100 / A100 tail test: every block-local class, `slide` and `slide_qblock`,
    N / real = 1024 / 700, 2048 / 1536, 256 / 100, 1024 / 1000, fp32 and master, inference + training with dy on all rows and on the
    real rows only, every position and every parameter gradient against the PyTorch reference; determinism of two training calls):
    60 of 60 `cuda` cases pass on the B200 (the tree before this part: 12 of 12 Protenix cases fail). The atom-dit agent's reproducer
    (`r3-atom-dit/b200cfg/odd.py`, 2048 / 2080 / 2100 / 2176 atoms, five classes, pad + slide, master + fp32): the last 128 rows now at
    the error of the others (Chai-1 slide master 2100 atoms: tail 4.95e-3, head 4.89e-3; was 3e-2).
  - Speed (`spd_sl_{base,new2}.jsonl`: 142 cases of Protenix / OpenFold3 / Boltz-2 / Chai-1, every row at 1024 / 4096 / 16384 atoms,
    min samples, eval + train, same session, the tree before this part against this one): geomean 1.0028, best 0.962, worst one run
    1.082 (OpenFold3 master training 1024 atoms); that case re-measured twice each way: 216.9 -> 220.8 us (+1.8 %; row 117
    `slide_qblock`: 218.9 -> 223.1 us), per kernel `local_attn_dkv` 8.6 -> 8.9 us (the kernels' larger parameter list / the tail call
    site; its ratio to the H100 stays at 0.50). Inference unchanged (within +-1 %).
  - Found on the way: with the group reaching past the last real window, an edge CTA's fourth window could lie past `nwin` (R = N,
    e.g. `slide_qblock` rounding 4072 up to 4096): its dbias offset pointed past the tensor in the dK / dV kernels (an illegal address in
    OpenFold3 `slide_qblock` 4096 atoms master training). Every edge window is now bounded by `nwin` (`wok`, `wq < nwin`); compute-sanitizer
    memcheck clean on the training calls (`san.out`).
- **ESMFold2Pair bf16 inference: the transition's residual update with one rounding** (`post_fwd.cu -DRES3`, the `post_rms_raw` build).
  The atom-dit agent's `-DRES1` (both residual updates `rn(a + g u)` in fp32 with one rounding) moved ESM master eval from 18 / 12 to 28 / 2
  GREEN / YELLOW but made one GREEN case YELLOW (177, 8192 atoms: e / tol 1.00 -> 1.17, worse than every token); the attention's update
  alone: 26 / 4, worst 1.32; the transition's alone (`RES3`): 27 / 3, worst 1.13 -> 1.03, no case worse than before. The four cases worse
  than the reference tokens ((165, 12288), (165, 16384), (177, 2048), (177, 6144)): GREEN, GREEN, 1.116 -> 1.033, GREEN.
  Speed unchanged (56 cases: 40 above 0.667, worst 0.998, geomean 0.724, as before; `spd_esm_g7.jsonl`). ESM float32 eval is untouched
  (bitwise the same errors: 20 / 10, worst 1.35 at (165, 2048 atoms, 5 samples) -- the fp32 path's, not this part's).
- Versions: `cuda:b200_sm100` 5 on Protenix / OpenFold3 / Boltz-2 / RoseTTAFold3 / Chai-1 (the slid window), 3 on ESMFold2Pair (RES3).


### Round 3, part 4 (2026-11-17, after the merges of base 0862a4516 / 28432858c / 697a33a76)

Records `$W/b2511/work/r3-local-dit/p4/` (`s1_*` after the bias kernels, `s2_*` the final tree, `ab1.out` / `ab3.out` same-session A / B,
`acc_esm_eval.jsonl`, `acc_local.jsonl`); tools in the worktree's `_ld4/` (git-excluded): `lprof.py` (kernel list + graph median of one
call), `bb.py` (the pair-bias kernels alone), `ab.sh` / `ab3.sh` (base tree against this tree, interleaved), `cmp.py` (records against the
H100 speed tokens), `acmp.py` / `atok.py` (accuracy against older records / the reference tokens).

**What the profiles showed.** The windowed pair-bias kernels (`lesm.cu` `local_esm_bias`, `lbias.cu` `local_bias_fwd`) were the largest
single launch at large lengths (ESMFold2 8192 atoms bf16 24.8 us of 84; Protenix 16384 atoms: the bias rows made the conditioning launch
49.6 us) and ran at a third of the bandwidth their bytes need. Cause: registers (99 / 98 per thread: two 256-thread CTAs an SM, 16 warps),
not the per-thread load chain -- four rows in flight per thread changed nothing (8192 atoms 32.7 -> 30.7 us alone), the register cap did.

- **Pair-bias kernels at eight CTAs an SM.** `local_esm_bias`: `__launch_bounds__(256, 8)` (32 registers, no spill), the key-mask byte
  and the pair row loaded independently (the slot clamped into the window so the address is always valid), 32-bit indices, a 8 x SMs grid;
  `local_bias_fwd` (bf16 pair) the same. Alone (graph, L2 flushed, incl. the ~6 us floor), bf16 pair: ESM 4096 / 8192 / 16384 atoms
  20.5 / 32.7 / 55.3 -> 14.3 / 22.6 / 36.9 us; `local_bias_fwd` 8192 / 16384: 22.5 / 36.9 -> 20.5 / 28.7 us. Same values (same
  arithmetic per element). `lbias_tf32.cu` (fp32 pair, 48 registers) capped the same way was slower (16384: 36.9 -> 38.9 us): unchanged.
- **The pair-bias rows out of the conditioning launch** (`cond_fwd -DPB` and `lbias_sm100.cuh` removed; `_pair_bias` launches
  `pair_bias_fwd`). The conditioning CTAs take a whole SM (232 KB shared memory), so the bias rows of the same launch only started after
  the tables, one 384-thread CTA an SM. With the fast stand-alone kernel (same session, `MW_LD4_PB` sweep, us): Protenix master 1024 /
  2048 / 4096 / 8192 atoms level (34.8, 43.0, 49.2, 69.6), 5120 59.4 -> 58.8, 12288 92.1 -> 90.1, 16384 116.7 -> 110.4; 3072 x 5 65.5 ->
  63.5; OpenFold3 slide 16384 118.8 -> 112.6.
- **ESMFold2 bf16 inference: the q / k RMSNorm + rotary inside the projection kernel** (`pre_fwd.cu -DQKROPE=1`, the `pre_rms_raw_rope`
  build that replaces `pre_rms_raw`): after the staged Q / K rows of a tile are complete (the KQLN frame: eight threads a row, 16 channels
  each), the four threads of a head reduce the sum of squares (`shfl ^1 ^2`), scale, and rotate with the partner channel c ^ 16 on lane
  ^ 2, then the TMA store writes the normalised rows. One launch fewer (`local_esm_qkrope` 2.2 - 3.3 us). Same-session A / B (base tree =
  the merge before this part): ESM master 1024 atoms 43.0 -> 40.9, 2048: 49.2 -> 45.1, 5120: 71.5 -> 63.5, 8192: 84.0 -> 75.8 us (with
  the bias kernel). Accuracy: ESM master eval 18 / 2 GREEN / YELLOW, worst 1.033, every case within 5 % of its previous e / tol.

Tried and rejected (measured, reverted; diffs in the scratchpad `ld4_rejected_*.diff`):

- the pair bias on a side stream beside the conditioning / projection launches (fp32 Protenix, ESM bf16 / fp32): the bias CTAs hold the
  SMs the row kernels need (one CTA an SM by shared memory): ESM master 8192 atoms 83.9 -> 88.1 us; a narrow bias grid (16 - 64 CTAs) is
  slower still (ESM master 1024 atoms 43.0 -> 45.1 - 63.4 us, 8192: 84.0 -> 122.9 - 317.4 us).
- the fp32 conditioning LayerNorm as the A phase of the tables' GEMM (`gemm_tf32` reading c, the rounding warps normalising each slice
  with per-tile statistics; the H100 does the tables in one launch): the per-tile statistics are a cold row read ahead of the first MMA,
  the GEMM went 11.4 -> 26.4 us at 4096 atoms (block 71.7 -> 88.0 us). The silu A phase for ESMFold2's fp32 tables likewise (the GEMM
  re-applies it per output tile: 9.4 -> 24.3 us at 3072 atoms).
- the fp32 q / k norm + rotary in `atom_pre_tf32`'s epilogue (lane-per-row, per-atom cos / sin loads): the pre kernel 13.6 -> 18.5 us
  against the 2.8 us launch it saves.

**H100 techniques against the B200 path** (audit of `docs/h100/local_atom_dit.md`, `integrations/local_dit_h100.py`,
`integrations/local_dit_esm.py`; classes Protenix / OpenFold3 / Boltz-2 / RoseTTAFold3 / Chai-1 / ESMFold2Pair):

| H100 technique | B200 |
|---|---|
| one-launch weight pack (`pack90`, every forward / backward form) | used: inference packs read from the eager warm-up (no kernel in a replay); variant blocks one `var_pack` launch; Protenix / OpenFold3 training packs by torch ops inside the capture (missing as one launch; not measured as a cost in the training profiles) |
| PDL on every row kernel | used (row kernels, fused TF32 kernels, the attention as a dependent launch in bf16 inference) |
| conditioning tables in one launch (LN gamma folded into W) | bf16: used (`cond_fwd -DKV`, one launch with the K / V tables); fp32: missing (`ln32` + `gemm32`) -- the LN as the GEMM's A phase measured slower (above); folding gamma needs the LN's per-row statistics in the epilogue (not built) |
| 64-row CTAs and `_split_for` grid.y splits (~264 CTAs) at few rows | bf16: finer (32-row tiles); fp32 fused kernels: 128-row tiles split into halves -- the latency chain per unit is the fp32 gap (missing: a column / CTA-pair split of `atom_post_tf32`) |
| shared conditioning read modulo N (+ grouped partials in the backward) | used |
| attention CTA per 64 queries x head x sample chunk, the bias staged once and reused by the chunk | used (bias per window shared by the samples, folds in one launch) |
| slid edge windows as `edge.cu` kernels | used in a better form: edge CTAs inside the same attention launches, no extra launch |
| inference hoists (tables, pair bias, edge tables) | used |
| weight gradients: cuBLAS + `finalize90` | used in a better form: one grouped split-K `atom_dw` launch, fixed order |
| pair bias `pair_bias_mma` one launch | used; part 4: full occupancy (above) |
| ESMFold2: `qknorm_rope` separate launch | bf16: fused into the projection kernel (one launch fewer than the H100); fp32: separate (fused measured slower) |
| ESMFold2 training: `train_rows` + cuBLAS | same decomposition (the sm_80 build of the same path); the tcgen05 training port stays unrouted (gradient e / tol 1.82 / 1.92 against 1.58 / 1.39, slower up to 2048 atoms) |

### Round 3, part 5 (2026-11-17, after the merge of base 25d5a6a73)

Records `$W/b2511/work/r3-local-dit/p5/` (`s3_<Class>_eval_{small,large}.jsonl`: inference, lengths <= 3072 in their own measurer runs,
`cmp_after.txt`); before = part 4's `p4/s2_*.jsonl` (whole-class runs). Training records unchanged (no training code touched).

**Where fp32 inference spends its time at few tiles** (Protenix 1024 atoms, graph replay, profiler): `atom_post_tf32` 26.7, `atom_pre_tf32`
21.3, the tables' `atom_gemm_tf32` 9.6, the TF32 attention 9.5, the pair bias 4.4, `f32_ln_aff` 2.8 us; graph median 61.4 us against the
H100's 58.4 (target 39.0). 1024 atoms are 8 tiles of 128 rows: the fused kernels ran 16 half units on 16 of 148 SMs, each a serial chain.

- **The fused TF32 kernels in quarter units** (`fused_tf32.cu`, `nfull = -1`; chosen when four units a tile fit one wave, <= 37 tiles /
  4736 rows): `atom_post_tf32` runs every tile as four 32-row units (its row phases already took any multiple of 32 rows; the MMAs still
  run M = 128, the TMEM lanes past the unit are not read), `atom_pre_tf32` as four 64-row x 256-column units (row half x column half: the
  warp-per-row loop over the unit's rows, lane quadrants past the unit store nothing). Same outputs bit for bit (14 cases, the env switch
  `MW_LD5_PQ` / `MW_LD5_PRQ` = 0 against the default, incl. 1000 atoms and a shared conditioning; two calls bitwise equal).
  Same session, graph, L2 flushed (us; post quarters, then + pre quarters): Protenix fp32 1024 atoms 61.4 -> 59.4 -> 55.2, 2048: 63.5 ->
  61.4 -> 57.3, 4096: 71.7 -> 69.6 -> 65.5; Boltz-2 1024: 61.4 -> 59.4 -> 55.3; RoseTTAFold3 1024: 55.3 -> 53.2 -> 49.2; ESMFold2Pair
  1024: 63.5 -> 59.4 -> 55.3. Over every inference case: fp32 geomean 0.984 (best 0.872), bf16 unchanged (0.998, noise +-5 %).

Counts above 0.667 (H100 >= 20 us; before -> after, worst ratio after):

| class | fp32 eval | master eval | fp32 train | master train |
|---|---|---|---|---|
| Protenix | 17 -> 15, 0.945 | 9 -> 8, 0.755 | 5, 0.706 | 0 |
| ESMFold2Pair | 10 -> 9, 0.876 | 9 -> 9, 0.718 | 5, 0.861 | 8, 0.941 |
| Boltz-2 | 7 -> 6, 0.951 | 2 -> 2, 0.701 | 0 | 0 |
| RoseTTAFold3 | 7 -> 6, 0.878 | 7 -> 6, 0.730 | 0 | 0 |
| OpenFold3 / Chai-1 | 0 | 0 | 0 | 0 |

Total 86 -> 79 (master eval changes are run-to-run noise around 0.667; the code there is unchanged).

**What is left, and what it would take.**
- fp32 inference up to ~4000 atoms (worst 0.95 at 1024 atoms: 55 us against a 39 us target): seven launches, each a latency chain of a
  few units. The quarter units took the cheap part; what remains needs fewer stages per unit -- the post kernel split by hidden half over a
  CTA pair (each CTA one [a | b] half, t summed over DSMEM in fixed order: half the 448 KB weight stream and one MMA round less per CTA),
  and the conditioning tables (`ln32` + `gemm32`, ~12 us) as one launch with gamma folded into the weights (the H100 `cond90` form; the LN
  as the GEMM's A phase measured slower in part 4).
- The 5-sample shared-conditioning fp32 cases (1024 x 5 = 40 tiles: halves, not quarters; 0.85 - 0.88): the same.
- bf16 inference (2048 - 5120 atoms and 1024 - 3072 x 5, 0.67 - 0.76): five launches at 7 - 15 us each; `cond_fwd` inside `pre_fwd`
  remains the next launch to remove.
- ESMFold2Pair training (13 cases, up to 0.94): still the sm_80 build; the tcgen05 training half needs the `RAWSC` AdaLN in its forward and
  RMS backward kernels for accuracy before it can be routed.
- Versions: `cuda:b200_sm100` 7 on Protenix / OpenFold3 / Boltz-2 / RoseTTAFold3 / Chai-1, 5 on ESMFold2Pair. No shared file outside
  `b200_atom` (sm_100a only) touched.

### Round 3, part 6 (2026-11-17, after the merge of base 8d27e53b7)

Records `$W/b2511/work/r3-local-dit/p6/` (`spd_<Class>.jsonl`: fp32 inference at 1024 / 2048 / 3072 atoms, every sample count, in their
own measurer runs; `acc_<Class>.jsonl`: fp32 inference at 1024 / 2048 atoms); A / B logs `$W/b2511/jobs/ld6{a,b,c}.log` (same session,
graph, L2 flushed; `_ld4/sweep.sh` env switches, `_ld4/ab.sh` base tree against this tree).

- **`atom_post_tf32` on a CTA pair** (`fused_tf32.cu -DPAIR=1`, builds `post_pair` / `post_rms_pair`, launched as clusters of 2;
  `b200_atom._pair_units`): at <= 18 tiles (2304 rows: the pairs of every tile's quarter units fit one wave) each 32-row unit runs on two
  CTAs. Both make gated -> y = gated Wo^T -> x2; CTA rank r streams and multiplies only hidden half r ([Wa_r; Wb_r] and Ws_r: 12 weight
  slices instead of 20, one [a | b] / h / Ws round instead of two); rank 1 writes its t_1 = h_1 Ws_1^T into rank 0's tile H through DSMEM
  (`st.shared::cluster`, once rank 0's MMAs are done: `hfree`; per-thread release arrivals on `xfull`), rank 0 adds t = t_0 + t_1 (always
  in this order: deterministic) and writes the output. The PARS / UP builds (RoseTTAFold3 / Chai-1 / Boltz-2) are unchanged (EARLY already
  runs the PARS transition under the attention). Accuracy: the block output against the fp64 reference equal to the single-CTA kernel's
  (Protenix 1024 / 1000 / 4096 / 8192 atoms, 1024 x 5 shared, OpenFold3 2048, ESMFold2Pair 1024 / 3072: relative L2 identical to 4
  digits, two calls bitwise equal). Speed (MW_LD6_PAIR 0 / 1, same session, us): Protenix 1024 atoms 55.3 -> 53.2, ESMFold2Pair 1024
  55.3 -> 53.2, Protenix 2048 57.3 -> 57.3; halves on pairs (4096 atoms, 32 tiles) were slower (71.7 -> 73.5) and are not used.
- **Built and rejected: the fp32 conditioning tables in one launch** (`gemm_tf32.cu -DLNA`, the H100 `cond90` form: gamma folded into the
  weights, the GEMM reads c itself; the rounding warps centre each row on the mean of its first 32 values, accumulate the shifted sums and
  squares over the four K slices and hand rstd / (mean - shift) to the epilogue, out = rstd (acc - md wsum) + bias; ESMFold2: silu in the
  operand phase; with one item a CTA the eight epilogue warps share the operand phase). Accuracy as the two-launch form (tables relative L2
  2.49e-4 against 2.49e-4, also with a row mean of 3; deterministic). Speed: no gain anywhere and slower at several (MW_LD6_C32 0 / 1, us:
  Protenix 1024 53.2 / 53.2, 2048 55.3 / 57.2, 1024 x 5 63.6 - 65.5 / 65.5, RoseTTAFold3 1024 49.2 / 51.1, OpenFold3 1024 51.3 / 53.2,
  ESMFold2Pair 1024 / 2048 level; with the four rounding warps alone at several items a CTA, 4096 atoms 65.5 -> 71.7). The removed
  `f32_ln_aff` (2.9 us alone) costs about as much inside the GEMM's operand phase, which is the GEMM's critical path at these sizes (the
  same reason part 4's per-tile LN A phase lost). Diff: scratchpad `b2/ld6_rejected_cond32.diff`.

Counts above 0.667 (H100 >= 20 us; the fp32 inference cases <= 3072 atoms re-measured, the others as part 5):

| class | fp32 eval | master eval | fp32 train | master train |
|---|---|---|---|---|
| Protenix | 15 -> 15, worst 0.945 -> 0.909 | 8 | 5 | 0 |
| ESMFold2Pair | 9 -> 9, worst 0.876 -> 0.846 | 9 | 5 | 8 |
| Boltz-2 | 6 (unchanged path) | 2 | 0 | 0 |
| RoseTTAFold3 | 6 (unchanged path) | 6 | 0 | 0 |
| OpenFold3 / Chai-1 | 0 | 0 | 0 | 0 |

Accuracy (fp32 eval, 1024 / 2048 atoms, every sample count): Protenix 8 / 4 GREEN / YELLOW (worst e / tol 1.09), OpenFold3 12 / 0,
ESMFold2Pair 4 / 2 (worst 1.35); no RED.

**What is left, and the measured reason.**
- fp32 inference <= 4000 atoms (1024 atoms 53 us against 39): the profile after this part (graph replay, us, PDL overlap counted in each
  kernel): `atom_post_tf32` 21.6, `atom_pre_tf32` 18.2, tables GEMM 10.6 (+ `f32_ln_aff` 2.9), TF32 attention 9.6, pair bias 4.7. The
  chain is the serial latency of five dependent kernels each of one to four 128-row items per CTA; the two levers tried here took 2 us. The
  remaining one with a size to it is the pre kernel on pairs (column halves on two CTAs is what the quarter units already do) or the tables
  folded into `atom_pre_tf32` (s1 / b1 for its rows: one GEMM stage more on pre's chain, the other tables for post) -- not built.
- bf16 inference (0.67 - 0.76): `cond_fwd` inside `pre_fwd` -- not built in this part (pre's TMEM holds W_p and two accumulator sets: all 512
  columns; the tables' product would need the smem-operand form and a second chain stage).
- ESMFold2Pair training (13 cases, up to 0.94): not built in this part; the tcgen05 training half still needs the RAWSC AdaLN and the RMS
  backward kernels reading the raw scale.
- Versions: `cuda:b200_sm100` 8 on Protenix / OpenFold3, 6 on ESMFold2Pair. Shared-file edits: `fused_tf32.cu` (a `PAIR` build option,
  default 0; the other builds' code unchanged; two new barriers initialised in every build), `b200_atom/__init__.py` (`post32` picks the
  pair build, `fused_kernel32` launches PAIR builds as clusters), `prebuild_sm100.py` (the two PAIR cubins).

Audit rows changed by this part (the rest as part 4's table):

| H100 technique | B200 |
|---|---|
| conditioning tables in one launch, gamma folded (`cond90`) | fp32: built (`gemm_tf32 -DLNA`, shift-centred operand phase) and rejected: level or slower (above); bf16: used (`cond_fwd -DKV`) |
| `_split_for` grid.y splits at few rows | fp32 post: finer than the H100 -- 32-row quarter units on CTA pairs split by hidden half (`-DPAIR`), t summed over DSMEM in a fixed order |
| ESMFold2 training: `train_rows` + cuBLAS | unchanged (sm_80 build of the same decomposition); tcgen05 half not routed |

### Round 3, part 7 (2026-11-17 evening restart, base 83e4c1ddf)

Records `$W/b2511/agents/r3-local-dit/_ld7/out/` (`base_<L>.jsonl` the base tree, `fin_<L>.jsonl` this part's tree: cuda / triton /
pytorch, every requirement case of the six classes at min samples, one measurer run per length, ascending, GPU 0 exclusive;
`cmp_base.txt` / `cmp_fin.txt` against the released H100 `cuda` speed tokens; `cs*` / `up*` / `esmT*` A / B records). The work of two
cancelled agents was stashed (`git stash list`: "cancelled agent 2026-11-17 r3-local-dit part 7" and "... r3-local-dit-inf") and
measured before it was taken.

- **AtomDiTESMFold2Pair training on the tcgen05 ESMFold2 block (routed).** The cancelled agent's port: the AdaLNs on raw scale tables in
  the training forward and in the RMS backward kernels (`pre_fwd` / `post_bwd` / `pre_bwd -DRAWSC`), the training weight packs in one
  launch (`b200_atom/esm_pack.cu`), the q / k RMSNorm + rotary inside the projection kernel with the raw q | k saved for its backward
  (`pre_fwd -DQKSAVE`), the pair bias / d bq / dx1 / dc work on side streams at <= 4096 rows. Same session, graph (us), the old route
  (`MW_ESM_TT=0`, the sm_80 build) -> this route: master 1024 / 4096 / 8192 atoms 227 / 334 / 487 -> 125 / 174 / 274, fp32 235 / 406 /
  614 -> 190 / 258 / 453. All 16 training cases now under 0.667 (worst 0.628; were 12 above, worst 0.924). Accuracy (1024 - 8192 atoms):
  4 GREEN / 4 YELLOW per dtype, worst e / tol 1.48 master, 1.69 fp32 (the sm_80 route 1.58 / 1.39, H100 token 1.66); no RED; three
  training calls bitwise equal (master 1024, fp32 1024 / 4096).
- **Boltz-2's up-gated fp32 post kernel on CTA pairs** (`fused_tf32.cu -DUP -DPAIR`, build `post_up_pair`; the cancelled -inf agent's):
  at <= 18 tiles rank r takes the hidden quarters 2 r, 2 r + 1, t summed over DSMEM as the part-6 pair. 1024 atoms 57.9 -> 54.1 - 55.8 us,
  2048 59.9 -> 57.9 us (r121 slide 57.2 -> 53.1); lights equal to the B200 tokens (r120 / r171 L1024 YELLOW 1.137 before and after).
  Still above 0.667 (0.916 / 0.768).
- **Rejected: the conditioning tables inside `atom_pre_tf32`** (the -inf agent's `-DTAB`, pre units make their AdaLN table pair, extra
  CTAs make the post tables; bitwise the same output): Protenix fp32 1024 / 2048 / 3072 / 4096 atoms 53.2 / 55.3 / 61.4 / 65.5 us either
  way -- the table GEMM moves onto pre's chain. Diff `docs/b200/rejected/local_dit_part7_tab_in_pre.diff` (it carries the UP pair hunks
  that were taken).
- **Rejected: the fp32 backward's bias column sums folded into the table-gradient kernels** (`f32_tail_bwd_cs` / `f32_adaln_bwd_cs`:
  persistent grid, per-block partial rows, `col_jobs`; no second read of the 755 MB dmod at 48 x 5120 atoms). Same session (us): Protenix
  fp32 train 48 samples 4096 / 5120 / 8192 atoms 3878 / 5060 / 7529 -> 3862 / 5025 / 7486 (-0.4 to -0.7 %), but 1024 x 48 1157 -> 1174,
  8-sample shared conditioning 4096 / 8192 671 / 1150 -> 711 / 1184 (+3 to +6 %). Deterministic, accuracy YELLOW 1.94 worst. Diff
  `docs/b200/rejected/local_dit_part7_colsum_fold.diff`.

Violations (H100 >= 20 us everywhere; none exempt), base -> this part: 80 -> 68 cases above 0.667, none above 1.0, no ladder pair
(cuda < triton < pytorch on every case, both runs).

| class | fp32 eval | master eval | fp32 train | master train |
|---|---|---|---|---|
| Protenix | 16 -> 16, 0.909 | 9 -> 9, 0.754 | 5 -> 5, 0.707 | 0 |
| ESMFold2Pair | 9 -> 9, 0.846 | 9 -> 9, 0.718 | 4 -> 0, 0.628 | 8 -> 0, 0.505 |
| Boltz-2 | 6 -> 6, 0.949 -> 0.916 | 2 -> 2, 0.709 | 0 | 0 |
| RoseTTAFold3 | 6 -> 6, 0.882 | 6 -> 6, 0.730 | 0 | 0 |
| OpenFold3 / Chai-1 | 0 | 0 | 0 | 0 |

What is left: fp32 inference <= 6144 atoms (a five-kernel latency chain; Protenix 1024 atoms 53 us against 39: post 21.5, pre 16.5,
tables GEMM 9.7, attention 9.6, bias 4.5, LN 2.9 us), bf16 inference 0.67 - 0.75 (Protenix 2048 atoms 43.0 against 39.7: post 15.4,
pre 13.9, cond 9.3, attention 8.2, bias 6.9 us -- a PDL-early bias launch would write before its grid dependency resolves, unsafe with
the caching allocator, not built), Protenix fp32 training at 48 samples (throughput: 5110 us at 5120 atoms, of which the 16 TF32
products 1537, `atom_dw` 783, `f32_adaln_bwd` 653, the attention backward 566 us).

### Round 3, part 8 (2026-11-18, base b070dfb11 = engine 2.5.14; no AtomDiT H100 token changed by it)

Records `$W/b2511/agents/r3-local-dit/_ld7/out/` (`bef_<L>.jsonl` / `cmp_bef.txt` the start of this part, `fin8_<L>.jsonl` / `cmp_fin8.txt`
this part's tree: cuda / triton / pytorch, every requirement case of the six classes at min samples, one measurer run per length, GPU 0
exclusive), A / B and timeline logs `$W/b2511/jobs/ld8_*.log` (`_ld8/tl.py`: per-kernel start / end of one graph replay, L2 flushed).

**Where the fp32 inference replay spent its time** (timeline, us from the replay's first kernel; Protenix 1024 atoms): `f32_ln_aff` 0 - 2.8,
tables GEMM 1.0 - 10.8, `atom_pre_tf32` 4.2 - 20.2, **pair bias 20.3 - 24.7**, attention 24.4 - 34.1, `atom_post_tf32` 26.3 - 47.8. The
pair bias reads the pair only but ran after the projections (a plain launch): 4.4 us of the critical path at 1024 atoms, 13.3 at 6144
(bf16: 6.9 at 2048, 10.5 at 5120).

- **Taken: the fp32 pair bias beside `atom_pre_tf32`** (`lbias_tf32.cu local_bias_fwd_f32_early`, `b200_atom_local.pair_bias_early32`,
  `local_dit_b200._pair_bias_deferred`; the AF3 block and the variant block without a conformer mask). The bias kernel is the programmatic
  dependent of the projection kernel and writes before any grid-dependency wait; CTA 0 waits for the projection kernel before it exits, so
  the attention (the bias kernel's dependent) still sees complete projections. Why the early writes are safe with the caching allocator:
  the caller makes z, the parameters and the bias buffer BEFORE the chain's launches (z / parameters complete: >= 2 launches earlier; the
  buffer a fresh allocation no running kernel reads), and `atom_pre_tf32` lets its dependents start only after its own wait (everything
  before it is complete then). A hoisted bias needs no launch; a refresh of the hoisted entry launches at once. Used up to 6144 rows (16384
  with a shared conditioning): above, `atom_pre_tf32` holds every SM and the bias CTAs beside it ran 2 - 3 x slower (8192 atoms 98.3 ->
  100.5, 16384: 161.8 -> 176.1 us). Same session (own launch -> beside, us): Protenix 1024 51.2 -> 49.1, 6144 92.2 -> 86.0, 1024 x 5 shared
  63.5 -> 61.5, 2048 x 5 79.9 -> 75.8, 3072 x 5 90.1 -> 86.0; OpenFold3 2048 55.3 -> 49.2; Boltz-2 1024 53.2 -> 47.5 - 49.1; RoseTTAFold3 4096
  65.5 -> 61.4. Outputs bit for bit those of the base tree (12 cases incl. 1000 atoms, the 5-sample shared conditioning, Boltz-2 /
  RoseTTAFold3 / OpenFold3).
- **Taken: the fused TF32 kernels' weight streams from three lanes, before the grid-dependency wait** (`fused_tf32.cu`: lane s of warp 0
  issues ring slot s -- the SWA agent's finding that one thread issues about one TMA load per ~600 clk --, and warp 0 skips the wait: it
  reads only the constant weight packs, made >= 2 launches earlier). Protenix fp32 1024 atoms 53.2 -> 51.2, 6144: 92.2 -> 90.2 us; same bits.
- **Rejected (measured, diffs `docs/b200/rejected/local_dit_part8_multilane.diff`): multi-lane TMA issue elsewhere.** `atom_gemm_tf32` (stage
  s from lanes 2 s / 2 s + 1): fp32 training 5120 x 48 the 16 products 1555 -> 1683 us; `local_attn_dq_tf32` (six lanes) 226 -> 284 us,
  `dkv_tf32` 339 -> 344, `fwd_tf32` 209 -> 214; `atom_dw` (box i from lane i): level (the training step 5112 -> 5277 us together with the
  GEMM change); the bf16 `cond_fwd` / `pre_fwd` / `post_fwd` loads from 2 - 15 lanes: level (ESMFold2Pair master 1024 39.0 -> 40.7).
- **Rejected: the bf16 pair bias beside `pre_fwd`** (the same early kernel on bf16 z): the bias ran 3 x slower beside it (2048 atoms 6.9 ->
  23.6 us; replay 43.0 -> 55.3, 5120: 57.3 -> 86.0); `pre_fwd` also lets its dependents start before its own wait, which the early writes
  would need moved. Not used.

Violations (H100 >= 20 us everywhere), start of this part -> after: 68 -> 65 cases above 0.667, none above 1.0, no ladder pair (cuda <
triton < pytorch on every case).

| class | fp32 eval | master eval | fp32 train | master train |
|---|---|---|---|---|
| Protenix | 16 -> 13, 0.909 -> 0.855 | 9 -> 9, 0.754 -> 0.755 | 5 -> 5, 0.707 -> 0.706 | 0 |
| ESMFold2Pair | 9 -> 9, 0.846 | 9 -> 9, 0.718 | 0 | 0 |
| Boltz-2 | 6 -> 6, 0.916 -> 0.803 | 2 -> 2, 0.704 -> 0.703 | 0 | 0 |
| RoseTTAFold3 | 6 -> 5, 0.883 -> 0.819 | 6 -> 7, 0.729 -> 0.740 (unchanged path: run-to-run) | 0 | 0 |
| OpenFold3 / Chai-1 | 0 | 0 | 0 | 0 |

Accuracy (fp32 eval, 1024 / 6144 atoms, every sample count): Protenix 12 / 4 GREEN / YELLOW (worst e / tol 1.06), Boltz-2 10 / 6 (1.14),
RoseTTAFold3 12 / 4 (1.01), OpenFold3 14 / 2 (1.01); no RED (the outputs are the base tree's bit for bit).

Found on the way (not changed): two processes of the same tree give different `to_out` / `squeeze` weight gradients (fp32 and master
training; within a process repeated calls are bitwise equal, the lights the same), and different Chai-1 inference outputs (the conformer
ids of `make_inputs`?) -- the base tree does the same.

What is left, and the measured reason: fp32 inference at 1024 atoms (49 us against 39: a five-kernel latency chain, tables 1.0 - 10.9,
pre to 21.6, attention to 30.4, post to 44.4); fp32 training at 48 samples (throughput: 16 TF32 products 1555, `atom_dw` 782,
`f32_adaln_bwd` 652, the attention backward 565, `f32_colsum` 332, `f32_adaln` 310 us of 5112 at 5120 atoms; the row kernels near 4.6 - 4.9
TB/s); bf16 inference 0.67 - 0.76 (the bias, 7 - 10 us after `pre_fwd`, cannot run beside it as built).

### Round 3, part 9 (2026-11-18, base 05fe8a72b = engine 2.5.14): determinism, three overlap attempts (all rejected)

**Cross-process "non-determinism" (part 8's finding) -- not a kernel.** The part-8 probe seeded torch only; the modules' default `legacy`
initialisation draws its truncated normals from NumPy's global state (`primitives._trunc_normal_init` via scipy), so two processes built
different weights (the probe now saves the parameters: 8 of 23 Chai-1 tensors differ, `p.*`). With the weights set from a seeded torch
generator, two processes (random hash seeds, and `PYTHONHASHSEED=0`) give the same output, input gradients and every weight gradient bit for
bit on 23 cases: the six classes, fp32 / master, inference / training, 1024 - 6144 atoms, 1 - 48 samples, the shared conditioning
(`_ld9/dp.py`, `det1.sh`). Regression test `tests/gpu/test_local_dit_b200.py` (B200-gated: two interpreters, 11 cases, plus repeated
calls in one process).

True state before this part (merged tree, `dev.sh -x`, one measurer run per length; `_ld7/out/cmp_pre9.txt`): 63 / 674 cases above 0.667,
none slower than the H100, ladder 0 (Protenix fp32 eval 13 / 0.855, master eval 9 / 0.755, fp32 train 5 / 0.706; ESMFold2Pair fp32 eval 8 /
0.846, master eval 9 / 0.718; Boltz-2 6 / 0.803 + 2 / 0.709; RoseTTAFold3 5 / 0.819 + 6 / 0.740). Unchanged after this part (no kernel
change kept).

Rejected (measured, same-session env A / B, outputs bit for bit equal in every variant):
- **Table GEMM weights before the grid-dependency wait** (`gemm_tf32.cu` `bpre`: the first 3 B stages issued by lanes 0-2 before
  `griddepcontrol.wait`, the inference conditioning tables only): level everywhere (Protenix fp32 1024 49.1 / 47.1 vs 47.1 / 49.1 us,
  ESMFold2Pair 1024 53.2 both). Diff `docs/b200/rejected/local_dit_part9_bpre.diff`.
- **The bf16 pair bias beside `pre_fwd`, with a correct release point** (`pre_fwd -DLATEREL`: dependents start after its own wait;
  `local_bias_fwd_early`, the fp32 early kernel's contract; grid 2 / 4 / 8 CTAs an SM, trigger at the start or after the rows): the bias
  runs 20 - 24 us beside `pre_fwd` (6.9 alone) even past `pre_fwd`'s end; Protenix master 2048 43.0 -> 49.2 - 55.3 us, 5120 59.4 -> 86 - 117,
  16384 108.5 -> 164 - 291, OpenFold3 8192 69.6 -> 105 - 160. Diff `local_dit_part9_bf16_early_bias.diff`.
- **ESMFold2Pair fp32 inference: the pair bias forked onto a side stream before the tables** (it sat serially between the table GEMM and
  `atom_pre_tf32`, 5.9 us): the bias now runs at 0 - 2.2 us, but the fork / join breaks the PDL chain (`f32_silu` 2.5 -> 6.3 us) and the
  replay is level (1024 53.2 both, 2048 57.3 -> 59.4 - 61.4). Diff `local_dit_part9_esm_bias_fork.diff`.

### Round 3, part 10 (2026-11-18, base d586ed4b7 / 80572b7d4 = engine 2.5.14): phase traces, transposed quarter-unit products

Records `$W/b2511/agents/r3-local-dit/_ld7/out/` (`pre10_<L>.jsonl` / `cmp_pre10.txt` the start, `fin10_<L>.jsonl` / `cmp_fin10.txt` the end:
cuda / triton / pytorch, every requirement case of the six classes at min samples, one measurer run per length, GPU 0 exclusive); A / B
and trace logs `$W/b2511/jobs/ld10*.log`; tools `_ld10/` (git-excluded): `tr.py` / `tr2.py` (per-CTA `%globaltimer` phase stamps of the
`-DTRACE=1` builds of `fused_tf32.cu` / `post_fwd.cu` in one graph replay), `gprof.py` (per-kernel training profile with each `gemm32`
call's bytes and FLOP), `bt.sh` (two trees, seeded weights, bitwise compare), `ab.sh` (two trees, interleaved graph medians).

**What the traces showed** (Protenix fp32 1024 atoms, us after the post kernel's grid-dependency wait): gated 1.8, Wo MMAs 0.8-1.0,
x2 phase 1.9, [a | b] MMAs 1.2-1.6, h 1.7 (2 of 8 warps: lane = row, 32 rows), Ws MMAs 0.8-1.0, t exchange 1.4, output 0.9 -- 12.5 us.
The MMA phases were NOT weight stalls (a 12-granule resident ring changed nothing: `docs/b200/rejected/local_dit_part10_compact_ring.diff`)
but the TF32 tensor rate: an M = 128 MMA computes 128 rows for the unit's 32 (pre: 64) real ones. Also: the PAIR build's `cluster_sync`
after the setup's PDL wait held the weight producer until the attention ended.

- **Taken: `atom_post_tf32t` / `atom_pre_tf32t`** (`fused_tf32.cu`): the quarter units' products transposed, D^T [128 features, 32 rows]
  = W . X^T (weight slice = A operand, M = 128; the unit's 32 rows = B operand, N = 32): a quarter of the tensor work, accumulators lane =
  feature so all 8 compute warps work in the TMEM phases (column writes back into 32-row K-major tiles, conflict-free), every row input
  except o loaded into registers BEFORE the PDL wait (g from pre, s, the tables: complete before the chain's last launch), pre on 32-row
  units over all 512 columns (one load round a warp instead of two), the PAIR cluster barrier before the wait. Builds: default / RMS / PAIR
  / EARLY (parallel schedule) / UP (Boltz-2: a granule = two 64-row boxes of the quarter pack) post, default / RMS / KQLN pre (RoseTTAFold3:
  the row statistics of q / k by a transposing warp butterfly + a 4-warp shared-memory sum). Bitwise equal to the row form on every case
  checked (16 fp32 eval cases of all six classes, 1000 atoms, 5-sample shared conditioning) except KQLN (tree vs sequential sum: output
  rel. max 7.8e-6; RoseTTAFold3 fp32 eval accuracy 27 GREEN / 5 YELLOW, worst e / tol 1.008). Graph medians (us, base -> this part):
  Protenix fp32 1024 49.1 -> 38.9, 2048 51.2 -> 47.1, 4096 63.5 -> 59.4; ESMFold2Pair 1024 55.0 -> 49.2, 3072 63.5 -> 57.3; Boltz-2 1024
  49.1 -> 38.9; RoseTTAFold3 1024 45.0 -> 34.8, 2048 47.1 -> 38.9; Chai-1 1024 45.0 -> 36.8; OpenFold3 2048 49.2 -> 45.1. Post after the
  wait 12.5 -> ~5 us; MMA instructions of N = 32 cost ~37 ns each (issue-bound: 64 a pre unit = 2.8 us, the remaining pre cost).
- **Taken: `gemm_tf32` NT 128 with a 6-stage ring** (one 4 KB staging a warp; NT 256 keeps 3 stages + two stagings: its 4-stage form was
  slower). Training profile (Protenix fp32 5120 x 48): the TF32 products at 3.2-5.6 TB/s (memory-bound, K 128-512); the dx products K 256 /
  512 N 128 went 80 -> 64, 141 -> 111 us. Step 5115 -> 4979 us (4096 x 48: 3913 -> 3818; 7168 x 48: 6811 -> 6615; Boltz-2 4096 x 48 3780 ->
  3667). Bitwise equal.
- **Taken: bf16 post two tiles a CTA above half the SMs** (the trace: post CTAs placed after the attention's end spent ~3.5 us in setup +
  weight staging; 64 CTAs with both warpgroups busy instead of 128 with one): master eval 2048 atoms Protenix 43.0 -> 41.0, RoseTTAFold3 /
  Boltz-2 43.0 -> 38.9; at <= 74 tiles level or slower (ESMFold2Pair 1024: 38.9 -> 41.0), so not used there.
- **Taken: bf16 post on 3-D tensor maps (`-DBOX3`)** (the Transition agent's finding: the number of TMA boxes, not bytes or lanes): a tile's
  inputs in 4 boxes (was 14), weights 7 (was 26), stores 1-5 (was 2-10). Protenix master 5120 59.4 -> 57.4, 2048 39-41 -> 38.9 us; bitwise.
- Rejected (measured): the 12-granule compact ring (level, above); bf16 post a / sg / mod loads before the PDL wait (level, ESMFold2Pair 1024
  +2 us; `local_dit_part10_bf16_post_prein.diff`); look-ahead loads in the sample loop of `f32_adaln_bwd` / `f32_tail_bwd` (bitwise, the
  48-sample cases are not shared-conditioning so the loop has one sample: no effect; reverted).

Fp32 training profile (Protenix 5120 x 48, us of 4979): TF32 products 1411 (4.7 TB/s mean of 8.0 GB), `atom_dw` 782 (~70 % of HBM),
`f32_adaln_bwd` 749, attention dK/dV 340 + dQ 226 + fwd 209, `f32_colsum` 326 (side stream), `f32_adaln` 321, `atom_gemm_rg_tf32` 286,
`f32_swiglu_bwd` 268, `f32_ln_bwd` 236. Everything is bandwidth-bound at 60-75 % of HBM; the remaining 3-4 % to the target is the
intermediates the H100's fused `tr_bwd90` / `post_bwd90` / `pre_bwd90` keep on chip (dh, dx2, dgated, dx1: ~1.26 GB of writes + reads at
5120 x 48 ~ 190 us) -- not built in this part.

Violations (H100 >= 20 us everywhere), start of this part -> after: 64 -> 47 cases above 0.667, none above 1.0, ladder 0.

| class | fp32 eval | master eval | fp32 train | master train |
|---|---|---|---|---|
| Protenix | 13 -> 9, 0.855 -> 0.855 (1024 x 5) | 9 -> 8, 0.755 -> 0.726 | 5 -> 2, 0.706 -> 0.687 | 0 |
| ESMFold2Pair | 9 -> 9, 0.846 -> 0.808 | 9 -> 8, 0.718 -> 0.727 | 0 | 0 |
| Boltz-2 | 6 -> 2, 0.803 -> 0.795 | 2 -> 2, 0.703 -> 0.710 | 0 | 0 |
| RoseTTAFold3 | 5 -> 1, 0.819 -> 0.819 | 6 -> 6, 0.740 -> 0.740 | 0 | 0 |
| OpenFold3 / Chai-1 | 0 | 0 | 0 | 0 |

What is left: the 5-sample shared-conditioning fp32 cases at 1024 atoms (40 tiles: above the quarter-unit limit of 37, still the row-form
kernels; 0.795 - 0.855) -- the transposed kernels with 64-row units (N = 64, 80 CTAs) would cover them; bf16 inference 0.67 - 0.74 (five
launches of 7 - 9 us); fp32 training 5120 / 6144 x 48 (0.687 / 0.667: the fused backward above).
- Versions: `cuda:b200_sm100` 22 on the six classes. Shared files: `b200_atom/fused_tf32.cu` (new entry points, `-DTRACE` stamps compiled
  out by default, setup's PAIR barrier moved before the wait), `gemm_tf32.cu` (ring), `post_fwd.cu` (sm_100a half only: `-DBOX3`, `-DTRACE`),
  `b200_atom/__init__.py` (routing, maps, `_box3_kernel`), `prebuild_sm100.py` (BOX3 builds), `local_dit_esm_b200.py` (preloads). GPU tests
  `tests/gpu/test_local_dit_b200.py` + `test_atom_dit_slide_tail.py`: 123 passed.

### Round 3, part 11 (2026-11-18, base 9a44bf254 = engine 2.5.14): 64-row transposed units, the dh product with the SwiGLU backward

Records `$W/b2511/agents/r3-local-dit/_ld7/out/` (`fin11_<L>.jsonl` / `cmp_fin11.txt`: cuda / triton / pytorch, every requirement case of
the six classes at min samples, one measurer run per length, GPU 0 exclusive; the start is part 10's `cmp_fin10.txt`, same tree before
this part), A / B logs `$W/b2511/jobs/ld11{a..f}.log` (`_ld11/`: env switches through `_ld8/sw.sh`, bitwise checks `_ld11/bte.sh`).

- **Taken: the fused fp32 kernels on 64-row transposed units** (`fused_tf32.cu -DTU64=1`, builds `pre_w` / `pre_rms_w` / `pre_kq_w`,
  `post_w` / `post_rms_w` / `post_up_w` / `post_pars_w`; `b200_atom._wide`): for 38..74 tiles (the quarters no longer fit one wave, the
  64-row units do: 1024 atoms x 5 samples, 5120 - 9472 rows) the transposed kernels of part 10 with N = 64: 80 CTAs at 40 tiles where the
  row form ran 80 half units through M = 128 MMAs for 64 real rows. X / Y 32 KB, H 64 KB (EARLY: x2 shares X -- every MMA reading it is
  done at the last h's commit), a 6-granule ring, 64 TMEM columns a part; a TMEM-phase thread takes 32 rows in two 16-column halves; the
  row phases take 8 rows a warp in two rounds of 4 with only g held from before the wait (a1 recomputed from Y for the output, the same
  fmaf). Bitwise equal to the row form on every case checked (Protenix / ESMFold2Pair / Boltz-2 / Chai-1 1000 x 5, OpenFold3 5000, Protenix
  6000, Boltz-2 9000 atoms) except RoseTTAFold3's KQLN pre (tree sum, out rel. 1.5e-5; accuracy 22 GREEN / 10 YELLOW, worst e / tol 1.005).
  Same session (us; row form -> wide post -> wide post + pre): Protenix 1024 x 5 69.6 -> 65.5 -> 63.5, Boltz-2 1024 x 5 63.5 -> 57.3 -> 55.3,
  RoseTTAFold3 1024 x 5 63.5 -> 61.4 -> 55.3, ESMFold2Pair 1024 x 5 73.7 -> 69.6 -> 69.6, Protenix 7168 94.2 -> 90.1 -> 88.1, ESMFold2Pair
  6144 94.2 -> 90.1 -> 90.1. The wide pre was 2 us slower at 40 / 48 unshared one-sample tiles (Protenix 5120 75.8 -> 77.8, Boltz-2 6144
  73.7 -> 75.8: the early pair bias runs beside it there), so the pre takes it with several samples or from 52 tiles (`pre32(samples=)`).
- **Taken: the dh product with the SwiGLU backward as its epilogue** (`gemm_sb_tf32.cu`, `b200_atom.dh_swiglu_bwd32`; the Protenix /
  OpenFold3 and ESMFold2 fp32 backwards): dh [M, 256] stays in TMEM; the epilogue loads the a / b chunks of u by TMA (one 32 x 32 chunk
  ahead, as `gemm_rg_tf32`) and writes d[a | b] and h. f32_swiglu_bwd's arithmetic element for element: bitwise equal (Protenix 1000 x 2,
  ESMFold2Pair 1000 x 2, OpenFold3 2048 x 4 training, every gradient). Removes 2 KB a row of HBM traffic (0.5 GB at 5120 x 48): Protenix fp32
  training 5120 x 48 4979 -> 4911 us, 6144 x 48 5788 -> 5702.
- **Taken: K = 64 stages from 3-D boxes** (`gemm_tf32.cu -DK64=1`, `gemm_k64`, `b200_atom._tm32k`): the K >= 256, N 128 products of many
  tiles (the training step's K 256 / 512 ones at ~3.5 TB/s) load each operand of a stage as one [32 x 128 x 2 slices] box. Bitwise equal
  (Protenix / ESMFold2Pair / Boltz-2 / OpenFold3 2048 x 12 training). Smaller than the issue-rate estimate: 5120 x 48 4912 -> 4896, 6144 x 48
  5708 -> 5691, Boltz-2 5120 x 48 4789 -> 4781 us.
- **Taken: ESMFold2's bf16 attention as a programmatic dependent launch** (`lattn_esm.cu -DLATTN_PDL`, `esm._load_esm_pdl`): the setup runs
  under the bias kernel and `post_fwd` stages its weights while the attention runs (the timeline had post starting only after the
  attention's end, 13.4 us of post against Protenix's 9.2). Bitwise equal (1000 x 5 eval, 2000 x 2 training). ESMFold2Pair master eval (us):
  1024 x 5 51.2 -> 47.1, 2048 45.0 -> 41.0, 5120 63.5 -> 61.4, 16384 122.9 -> 120.8.
- Measured, not taken: 256-wide tiles for the fp32 conditioning tables at 1024 x 5 (one wave of 120 tiles instead of 240): level. The fp32
  training row kernels alone run at copy speed (`_ld11/p1.py`, 245760 rows: AdaLN backward with the gate 5.1 TB/s, without 4.6, tail 5.4,
  a clone 5.45 TB/s); in the step they share HBM with the side stream's column sums, so only fusion removes their bytes.

Violations (H100 >= 20 us everywhere), part 10's end -> this part: 47 -> 39 cases above 0.667, none above 1.0, ladder 0.

| class | fp32 eval | master eval | fp32 train | master train |
|---|---|---|---|---|
| Protenix | 9 -> 10, 0.855 -> 0.770 (1024 x 5) | 8 -> 8, 0.726 | 2 -> 1, 0.687 -> 0.675 (5120 x 48) | 0 |
| ESMFold2Pair | 9 -> 8, 0.808 -> 0.785 (1024, unchanged path) | 8 -> 3, 0.727 -> 0.679 | 0 | 0 |
| Boltz-2 | 2 -> 0, 0.795 -> 0.664 | 2 -> 2, 0.710 -> 0.679 | 0 | 0 |
| RoseTTAFold3 | 1 -> 1, 0.819 -> 0.710 | 6 -> 6, 0.740 | 0 | 0 |
| OpenFold3 / Chai-1 | 0 | 0 | 0 | 0 |

(Protenix fp32 eval 9 -> 10 although no case got slower: 1024 x 1 read 40.7 against 38.9 in the measurer, the same tree A / B'd against
part 10's at 38.9 / 38.9; four cases sit at 0.667 - 0.671.)

What is left: fp32 inference 1024 x 5 (57 against 49 us: conditioning tables 16 us before pre, attention 23 us, a five-kernel chain),
ESMFold2Pair fp32 1024 / 2048 / 5120 (its own path, not changed here); bf16 inference 0.67 - 0.74 (Protenix / RoseTTAFold3: the five
launches; the transposed small-N products for the bf16 pre / post are not built); fp32 training 5120 x 48 (4833 against 4777 us: the dx2 /
dgated / dx1 products with the AdaLN / gate backwards as epilogues, ~0.75 GB, not built).
- Versions: `cuda:b200_sm100` 23 on the six classes. Shared files (sm_100a only): `b200_atom/fused_tf32.cu` (`TU64` builds; the 32-row
  builds' code unchanged up to two one-trip loops), `gemm_tf32.cu` (`K64` build), new `gemm_sb_tf32.cu`, `b200_atom/__init__.py` (routing,
  `_tm32k`, `atom_kernel32` takes builds with defines), `b200_atom_local/lattn_esm.cu` + `esm.py` (PDL build), `prebuild_sm100.py` (the new
  cubins), `local_dit_b200.py` / `local_dit_esm_b200.py` (calls). GPU tests `tests/gpu/test_local_dit_b200.py` +
  `test_atom_dit_slide_tail.py`: 123 passed.

### Round 3, part 12 (2026-11-18, base d8887164b, then 234ae0ac3 = engine 2.5.14): the pair bias off the inference chains

Records `$W/b2511/agents/r3-local-dit/_ld7/out/` (`fin12_<L>.jsonl` / `cmp_fin12.txt`: cuda / triton / pytorch, every requirement case of
the six classes at min samples, one measurer run per length, GPU 0 exclusive; `fin12e_*` / `cmp_fin12e.txt` the ESMFold2Pair fp32 eval rows
after the final routing; the start is part 11's `cmp_fin11.txt`, the same code for these classes), A / B and timeline logs
`$W/b2511/jobs/ld12*.log` (`_ld12/`; bitwise checks `_ld12/bte.sh`).

**Where the bf16 inference replay spent its time** (timeline, us): Protenix 2048 atoms cond 0 - 9.2, pre 2.2 - 16.1, **pair bias 16.1 - 23.0**
(a plain launch after pre), attention to 30.6, post to 36.0; 5120 atoms the bias 20.5 - 31.0. The variant path (RoseTTAFold3 / Boltz-2) forks
the bias onto a side stream beside the tables, but its 8-CTA-an-SM grid held the SMs: cond started at 7.6 us (1024 x 5).

- **Taken: the bf16 pair bias made by the projection launch** (`pre_fwd.cu -DEB=1`; builds `prex_eb` / `prexp_eb` / `pre_a1_eb` /
  `pre_kq_a1_eb`; `b200_atom_local.pair_bias_in_pre`, `local_dit_b200._pair_bias_pre`): the compute warps of every CTA make rows
  [R b / G, R (b + 1) / G) of the bias (local_bias_fwd's arithmetic statement for statement, 4 rows a thread in flight, the 64 weights in
  the dynamic shared memory past the barriers) BEFORE their first tile -- the time they otherwise wait for the conditioning tables. No bias
  launch; the attention (PDL) waits for the projection grid's completion, so it sees the bias. Contract as part 8's early fp32 bias (z,
  parameters and the fresh buffer made before the chain). Used up to 12288 rows (16384 with a shared conditioning): Protenix 16384 atoms
  108.5 -> 114.7 us (every SM holds a projection CTA: the bias rides the critical path), 12288 level. Outputs bit for bit the base tree's
  (Protenix 1000 x 5 / 2048 / 5000, RoseTTAFold3 1000 x 5, Boltz-2 2048, OpenFold3 2048, Chai-1 1000 unchanged route). Same session (us, off
  -> on): Protenix master 1024 x 5 47.1 -> 43.0, 3072 x 5 75.8 -> 71.6, 4096 49.2 -> 45.1, 5120 57.4 -> 55.3; RoseTTAFold3 1024 x 5 45.1 ->
  41.0, 3072 x 5 69.6 -> 65.5, 5120 55.3 -> 53.2-55.0; Boltz-2 1024 x 5 45.1 -> 41.0; OpenFold3 8192 69.6 -> 67.5; 2048 x 1 level (38.9).
- **Taken: ESMFold2's fp32 pair bias beside `atom_pre_tf32t`** (`lesm.cu -DEARLY=1`, `esm.esm_bias_early`): the programmatic dependent of the
  projection kernel (launch_dependents at its start, CTA 0 waits at its end -- `local_bias_fwd_f32_early`'s contract); it ran serially
  between the table GEMM and pre (5.9 us at 1024 atoms). Used with one sample up to 6144 rows and with several samples always (7168 x 1
  92.2 -> 98.3, 8192 x 1 98.3 -> 104.4, 8192 x 5 286.7 -> 278.5). Same bits (1000 x 1, 3000 x 2). Us: 1024 49.2 -> 43.0, 2048 53.2 ->
  49.2, 5120 86.0 -> 79.9, 1024 x 5 69.6 -> 65.5.
- **Rejected: ESMFold2's bf16 bias inside the projection launch** (`-DEB=2`, lesm.cu's position-indexed elements, 160 per row): the
  projection kernel 13.7 -> 38.9 us at 5120 atoms, replay 61.4 -> 69.6, 16384: 120.8 -> 163.8. Diff
  `docs/b200/rejected/local_dit_part12_esm_bf16_bias_in_pre.diff`.

Violations (H100 >= 20 us everywhere), part 11's end -> this part: 39 -> 18 cases above 0.667, none above 1.0, ladder 0.

| class | fp32 eval | master eval | fp32 train | master train |
|---|---|---|---|---|
| Protenix | 10 -> 8, 0.770 -> 0.773 (1024 x 5, unchanged path) | 8 -> 1, 0.726 -> 0.671 (16384) | 1 -> 1, 0.675 (5120 x 48) | 0 |
| ESMFold2Pair | 8 -> 3, 0.785 -> 0.700 (5120) | 3 -> 3, 0.679 | 0 | 0 |
| Boltz-2 | 0 | 2 -> 0, 0.679 -> 0.630 | 0 | 0 |
| RoseTTAFold3 | 1 -> 1, 0.710 -> 0.684 (1024 x 5) | 6 -> 1, 0.740 -> 0.692 (1024: 36.7 against 35.3) | 0 | 0 |
| OpenFold3 / Chai-1 | 0 | 0 | 0 | 0 |

What is left: Protenix fp32 inference with five samples (1024 x 5 57 against 49 us, 2048 x 5 78 / 73, 4096 / 6144 / 16384 x 5 at 0.67 -
0.70) and one sample 5120 - 12288 (0.668 - 0.684); ESMFold2Pair fp32 5120 / 6144 / 12288 (0.68 - 0.70) and bf16 2048 x 5 / 5120 / 16384
(0.670 - 0.679: its bias kernel stays a launch after pre); RoseTTAFold3 fp32 1024 x 5, master 1024 (one 2-us timer step); fp32 training
5120 x 48 (4833 against 4777 us: the fused dx2 / dgated / dx1 epilogues, not built). The token-DiT agent's CTA-pair TF32 GEMM (part 9) was
not applied: its measured gain comes from epilogue fusions, and the table GEMM here has none to remove.
- Versions: `cuda:b200_sm100` 24 on the six classes. Shared files (sm_100a only): `b200_atom/pre_fwd.cu` (`EB` build), `b200_atom/__init__.py`
  (the four `_eb` builds, `eb=` arguments), `b200_atom_local/lesm.cu` (`EARLY` build) + `esm.py` + `__init__.py`, `prebuild_sm100.py`,
  `local_dit_b200.py` / `local_dit_esm_b200.py`. The GPU tests could not run: the shared env on the B200 has no pytest (`No module named
  pytest`); the bitwise A / B checks above stand in for them.

### Round 3, part 13 (2026-11-18, base f2d941e47 = engine 2.5.14): table GEMM analysis, bias launches, the fused gate backward

Records `$W/b2511/agents/r3-local-dit/_ld7/out/` (`fin13b_<L>.jsonl` / `cmp_fin13b.txt`: cuda / triton / pytorch, every requirement case of the
six classes at min samples, one measurer run per length, GPU 0 exclusive; the start is part 12's `cmp_fin12.txt` + `cmp_fin12e.txt`, the same
code for these classes), A / B logs `$W/b2511/jobs/ld13*.log` (`_ld13/`: `bt2.sh` two trees, `bte.sh` two env settings, both `_ld9/dp.py`
with seeded random non-zero weights, comparing output, d single, d pair and every weight gradient with torch.equal; `ab2.sh` / `_ld8/sw.sh`
interleaved graph medians).

**The fp32 conditioning table GEMM (Protenix 1024 x 5 shared conditioning).** The requirement case shares one conditioning across the 5
samples, so the tables are [1024, 1024] (mod 768 + mkv 256 columns) from K = 128: 0.27 GFLOP, ~4 MB written, ~4 MB (A twice) + 0.5 MB read.
Timeline: `f32_ln_aff` 0 - 2.9 us, `atom_gemm_tf32` (64 CTAs of 128 x 128: 43 % of the SMs, one tile each) 1.0 - 10.8 us, i.e. ~8 us after
the LN -- ~1 TB/s and ~34 TFLOP/s: latency, not bandwidth or tensor rate (the same 9.9 us at 1024 x 1). The "16 us" of part 11 was the
pre kernel's span, not the tables. At 6144 x 5 (tables [6144, 1024]) the GEMM is 16 us (~2.5 TB/s).
- **Rejected: the tables in ONE launch** (`ctab_tf32.cu`: a CTA per 128 x NT tile, its 8 warps make LN(c) * g / silu(c) / c in registers
  into the swizzled A tile, 4-lane weight boxes, NT 64 / 128 / 256 so the grid covers the SMs). Slower everywhere (us, two-launch -> one):
  Protenix 1024 x 5 55.3 -> 57.3, 1024 38.9 -> 41.0, 5120 75.8 -> 84.0, 6144 x 5 165.9 -> 174.1, ESMFold2Pair 5120 79.9 -> 92.0; NT 64 / 128
  worse still. Diff `docs/b200/rejected/local_dit_part13_ctab_one_launch.diff`.

- **Taken: the plain pair-bias launches trigger their PDL dependents and read a row by one 256-bit load** (`lbias.cu local_bias_fwd`,
  `lesm.cu local_esm_bias` non-EARLY: `griddepcontrol.launch_dependents` at the start -- the attention behind them is a PDL launch that
  waits for this grid before reading the bias; `ld.global.nc.v8.b32`). Bitwise equal (pre-change tree vs this, 5 cases: Protenix master
  16000, ESMFold2Pair master 16000 / 5000, fp32 12000, OpenFold3 master 8192). Us: Protenix master 16384 108.6 -> 106.5, ESMFold2Pair
  master 16384 120.8 -> 118.8, fp32 12288 145.4 -> 143.4, 2048 x 5 level.
- **Rejected: `pre_fwd` on 3-D tensor maps (BOX3)**: W in 4 boxes (was 16), inputs 2 / 3 (was 6 / 10), outputs 4 - 6 (was 8 - 12); bitwise
  equal (9 cases), level (ESMFold2Pair master 2048 x 5 55.3 / 55.3, 5120 61.4 / 61.2, RoseTTAFold3 1024 32.8 / 32.8, Protenix 2048 / 16384
  level, 5120 x 48 master train +0.2 %). Diff `local_dit_part13_pre_box3.diff`.
- **Taken: dgated = dyo Wo with the output gate's backward as its epilogue** (`gemm_rg_tf32.cu -DGATEB=1`, `b200_atom.gemm_gate_bwd32`,
  `local_dit_b200._gated_bwd`; Protenix / OpenFold3, the variant blocks, ESMFold2): o and the gate pre-activation g arrive by TMA one chunk
  ahead (gemm_rg's form), f32_gate_bwd's arithmetic element for element (D in its lane-tree order); dgated never leaves TMEM (~250 MB at
  5120 x 48). Bitwise equal (env A / B, 7 fp32 training cases of all six classes incl. the shared conditioning). Us: Protenix 5120 x 48
  4898 -> 4839, 6144 x 48 5687 -> 5629, 1024 x 48 1131 -> 1121, Boltz-2 5120 x 48 4785 -> 4725, ESMFold2Pair 4096 x 48 3575 -> 3538. The same
  epilogue in gemm_tf32's per-thread form was 8-18 us SLOWER (per-thread 128-B row loads; diff `local_dit_part13_gate_bwd_in_gemm_tf32.diff`).
- **Rejected: dx2 = dab Wu with the AdaLN 2 backward as its epilogue** (`gemm_ab_tf32.cu`: dx rows copied from TMEM to a swizzled smem
  buffer, then f32_adaln_bwd warp-per-row on them): bitwise equal, but 5120 x 48 4840 -> 4874, 1024 x 48 1122 -> 1151 -- 8 warps a CTA
  working the rows are latency-bound (the stand-alone row kernel has 64 warps an SM). Diff `local_dit_part13_dx2_adaln_bwd_epilogue.diff`.

Violations (H100 >= 20 us everywhere), part 12's end -> this part: 18 -> 13 cases above 0.667, none above 1.0, ladder 0.

| class | fp32 eval | master eval | fp32 train | master train |
|---|---|---|---|---|
| Protenix | 8 -> 7, 0.773 -> 0.746 (1024 x 5) | 1 -> 0 | 1 -> 1, 0.675 -> 0.669 (5120 x 48: 4793.6 against 4776.8) | 0 |
| ESMFold2Pair | 3 -> 2, 0.700 -> 0.685 | 3 -> 1, 0.679 -> 0.668 (16384: 120.7 against 120.5) | 0 | 0 |
| Boltz-2 | 0 -> 1, 0.664 -> 0.669 (1024: 40.8 against 40.7, unchanged path) | 0 | 0 | 0 |
| RoseTTAFold3 | 1 -> 1, 0.684 -> 0.683 (1024 x 5) | 1 -> 0 | 0 | 0 |
| OpenFold3 / Chai-1 | 0 | 0 | 0 | 0 |

What is left: Protenix fp32 inference with five shared-conditioning samples (1024 x 5 55 against 49 us: the five-kernel chain; the table
GEMM ends at 10.8 us, pre at 23, attention at 36, post at 51) and one sample 5120 - 12288 (0.670 - 0.685); ESMFold2Pair fp32 5120 / 6144;
RoseTTAFold3 fp32 1024 x 5; the 0.668 - 0.669 cases (Protenix fp32 train 5120 x 48, ESMFold2Pair master 16384, Boltz-2 fp32 1024) one
timer step or < 0.4 % away. Not built here: the fp16-exact kind::f16 transposed unit, the PAIR t exchange by st.async (both part 14).
- Versions: `cuda:b200_sm100` 25 on the six classes. Shared files (sm_100a only or bit-preserving): `b200_atom_local/lbias.cu` (the H100
  build's preprocessed code unchanged: `ATOM_HOPPER` guards), `lesm.cu`, `b200_atom/gemm_rg_tf32.cu` (`GATEB` build; the plain build's code
  unchanged), `b200_atom/__init__.py` (`gemm_rg_gb`, `gemm_gate_bwd32`, the `_pre_maps` helper -- same maps), `local_dit_b200.py` /
  `local_dit_esm_b200.py` (`_gated_bwd`).

### Round 3, part 14 (2026-11-18, base 98e15317b = engine 2.5.14): fp16-exact kind::f16 transposed units

Records `$W/b2511/agents/r3-local-dit/_ld7/out/` (`fin14_<L>.jsonl` / `cmp_fin14.txt`: cuda / triton / pytorch, every requirement case of the six
classes at min samples, one measurer run per length, GPU 0 exclusive; `fin14m_*` / `fin14e_*` the re-measured Protenix master 2048 and
ESMFold2Pair master 16384 rows; the start is part 13's `cmp_fin13b.txt`, the same code for these classes), accuracy `acc_f16b.jsonl` (this
tree) / `acc_f16b0.jsonl` (the same tree, `MW_LD14_F16=0`: the tf32 kernels), A / B and trace logs `$W/b2511/jobs/ld14*.log` (`_ld14/`:
`u1.py` pre32 / post32 alone against the tf32 build and an fp64 reference, `bte.sh` seeded bitwise compare of two env settings, `tr.py`
phase stamps with the shared-conditioning stream, `tl.sh` timelines, `ptxas.py` register / spill report).

- **Taken: `atom_pre_f16t` / `atom_post_f16t`** (`fused_tf32.cu -DF16=1`; builds `pre_h` / `pre_rms_h` / `pre_kq_h`, `post_h` / `post_rms_h` /
  `post_pair_h` / `post_rms_pair_h` / `post_pars_h` / `post_up_h` / `post_up_pair_h`, `_hw` the 64-row ones; `b200_atom._f16pack` /
  `_f16post` / `_f16pre`): the transposed units of parts 10 / 11 with kind::f16 products on fp16 operands that hold the tf32 operands
  EXACTLY. A tf32 value has 11 significant bits, as an fp16 one has, so it is an fp16 value whenever it lies in fp16's normal range; every
  operand is scaled by a power of two first. Weights: per pack block (Wp's q / k / v / g blocks, Wo, Wu, Ws) the largest magnitude into
  [2^14, 2^15); the host checks that every scaled element is an fp16 value bit for bit (`torch.equal(w16.float(), scaled)`) and runs the
  tf32 kernels otherwise (a block spanning more than ~2^29, or a non-finite one). Activations: per row, the largest magnitude of the
  tf32-rounded row into [2^14, 2^15) (gated, x2, x1, xkv: `redux.sync.max.u32` on the |x| bit patterns -- the 5-step shuffle butterfly cost
  ~1 us of the 64-row post); h (feature per lane: its row maximum is not at hand) from the bound |h| <= 2 (A1 X + 0.2785) B1 X [U1 X] (X the
  row's max |x2|, A1 / B1 / U1 the largest L1 norm of a Wa / Wb / Wu row, host). The accumulators are multiplied back by 2^-(e_w + e_row)
  (pow2 by pow2: exact). Range: no overflow; an element below 2^-29 of its row's maximum (h: of the bound) lands in fp16's subnormals and
  may lose low bits (error < 2^-39 of the row maximum). The packs are made once per weight-pack object (cached on the object and its
  version); a CUDA-graph capture never builds one (it gets the tf32 kernels), so a capture after an eager call uses them. Half the MMA
  instructions (K 16 a kind::f16 MMA), half the weight bytes (16 KB granules of [128 x 64 K]), half the operand tiles; the 32-row post
  keeps every row input but o in registers from before the wait (as `atom_post_tf32t`); the PAIR post moves t_1 to rank 0 by ONE 16 KB
  bulk DSMEM copy (`cp.async.bulk.shared::cluster`, xfull count 1 + expect_tx).
  Numerics (basis of the claims): (1) the operands are exact -- `u1.py`, the same inputs through both builds: pre32's P differs from the tf32
  build's by at most 3.3e-7 relative (fp32 ulps on 82 % of the elements: the tensor core sums the same exact products in its own K
  grouping, 16 against 8), against fp64 both 1.00e-4 max / 6.66e-6 rms (M 2048; 5120 / 12288 / 20480 alike); a single inexact operand would
  show as ~1e-4; post32 differs up to 1.1e-4 / 2.0e-4 relative only through the tf32 re-rounding of x2 / h that an ulp flips, against fp64
  F16 rms 7.9e-6 / 1.15e-5 / 9.9e-6 / 1.07e-5 vs tf32 9.3e-6 / 1.28e-5 / 1.21e-5 / 1.29e-5 (M 2048 / 5120 / 12288 / 20480); NOT bit-identical to
  the tf32 build (whole block outputs differ by 0.5e-4 - 2.3e-4 relative max, 12 cases of all six classes, `bte.sh`). (2) The engine standard
  against the fp64 gold (accuracy harness, fp32 eval, 1024 / 2048 / 5120 / 6144 / 12288 atoms at the requirement samples, 236 records): this
  tree and the tf32 build give the same lights per class (Protenix 28 / 12 GREEN / YELLOW, OpenFold3 31 / 5, Boltz-2 30 / 10, RoseTTAFold3 28
  / 12, Chai-1 31 / 9, ESMFold2Pair 19 / 1; no RED) and worst e / tol 1.102 vs 1.103, 1.145 vs 1.145, 1.146 vs 1.142, 1.008 vs 1.008, 1.162
  vs 1.162, 1.041 vs 1.041. (3) Deterministic: two calls in a process bitwise equal on all 12 cases; GPU tests 123 passed.
  Same session (tl.py graph medians, us, tf32 -> F16): Protenix 1024 x 5 (unshared) 63.5 -> 61.4, 5120 75.8 -> 71.8 - 73.6, 6144 81.9 -> 77.8,
  4096 59.4 -> 57.3, ESMFold2Pair 5120 79.9 -> 77.8, 6144 90.1 -> 88.0, RoseTTAFold3 1024 x 5 55.3 -> 53.2, Boltz-2 1024 38.9 -> 36.9, 1024 x 5
  57.3 -> 55.3, Chai-1 1024 38.7 -> 36.9; never slower. Phase trace (post, 64-row, Protenix 1024 x 5): the [a | b] MMAs 4.17 -> 2.52 us, the t
  MMAs 3.51 -> 1.57, the PAIR post after the wait 5.7 -> 4.1 us.
- **Taken: the 64-row F16 post for > 74 tiles too** (persistent: `min(SMs, units)` CTAs; `MW_LD14_BIG=0` the row form): Protenix 4096 x 5
  159.7 -> 149.5 - 151.6, 16384 x 5 485.4 -> 469.0, 6144 x 5 215.0 -> 210.9, 2048 x 5 96.3 -> 94.2, 12288 133.0 -> 131.1 (unshared stream).
- Measured, not taken: the 64-row F16 pre at 40 / 48 unshared tiles (`MW_LD14_PRW=1`: the early pair bias beside it ran 2 us longer --
  Protenix 5120 71.8 -> 75.8, ESMFold2Pair 5120 77.8 -> 84.0; part 11's rule kept); the 64-row F16 pre for > 74 tiles with all of Wp resident
  (`MW_LD14_BIG=1 / 2`: 2048 x 5 96.3 -> 98.3, 6144 x 5 215.0 -> 221.0, 12288 133.0 -> 135.2); the tf32 PAIR post's t_1 by one bulk copy
  (`-DXBULK=1`, `MW_LD14_XB=1`: bitwise equal on 4 cases, the exchange phase 0.65 - 0.72 us either way -- latency, not the store count:
  built, off).
- Rejected (diff `docs/b200/rejected/local_dit_part14_pdl_bias.diff`): the plain fp32 pair bias as a PDL launch (wait first; with / without
  triggering the attention at once; two rows a thread by 256-bit loads): bitwise equal, slower everywhere (12288 131.1 -> 139.3 - 143.4,
  6144 x 5 shared 165.9 -> 170.0 - 174.1, 8192 86.0 -> 92.1 - 96.3). The early bias with a capped grid (68 / 148 / 296 CTAs) or off: level or
  slower (5120: 71.7 -> 79.9 / 94.2 / 81.9 off).
- Timing probe caveat: a first `-DF16PROBE` build (kind::f16 on the tf32 bytes, garbage values) suggested 6 us at 1024 x 5; the real
  kernels gave ~1 us a kernel: garbage q / k made the attention faster. Probes with garbage operands do not time the chain.

Violations (H100 >= 20 us everywhere), part 13's end -> this part: 13 -> 6 cases above 0.667, none above 1.0, ladder 0 (fin14 read Protenix
master 2048 at 40.9 / 0.686 on an unchanged path; the re-measure gave 38.9 / 0.654).

| class | fp32 eval | master eval | fp32 train | master train |
|---|---|---|---|---|
| Protenix | 7 -> 4, 0.746 -> 0.717 (1024 x 5: 55.3 -> 53.2 against 49.4) | 0 | 1 -> 1, 0.669 (5120 x 48: 4790.2 against 4776.8) | 0 |
| ESMFold2Pair | 2 -> 0, 0.685 -> 0.665 | 1 -> 1, 0.668 (16384: 120.7 against 120.5) | 0 | 0 |
| Boltz-2 | 1 -> 0, 0.669 -> 0.662 | 0 | 0 | 0 |
| RoseTTAFold3 | 1 -> 0, 0.683 -> 0.656 | 0 | 0 | 0 |
| OpenFold3 / Chai-1 | 0 | 0 | 0 | 0 |

What is left, and the measured reason: Protenix fp32 inference with five shared-conditioning samples (1024 x 5 53.2 against 49.4: the
64-row pre's row phase takes 6.1 us for 8 rows a warp -- two dependent load rounds of ~3 us each, L2 latency, no register room to
overlap them; the post's x2 phase 3.3 us likewise; 2048 x 5 75.8 / 72.8 and 6144 x 5 167.8 / 159.9: the row-form tf32 pre runs 29 / 44 us,
and at 6144 x 5 the pair bias (14 us) is a plain launch after it) and 12288 x 1 (133.0 / 129.6: the plain pair bias reads 100 MB in 24 us
after pre; every faster form measured slower); ESMFold2Pair master 16384 (its pair bias moves ~142 MB in 30 us at ~4.7 TB/s after pre);
Protenix fp32 training 5120 x 48 (0.3 %: memory-bound products, unchanged here).
- Versions: `cuda:b200_sm100` 26 on the six classes. Shared files (sm_100a only): `b200_atom/fused_tf32.cu` (the `F16` kernels at the end,
  compiled only with `-DF16=1`; `setup()` takes xfull's count, default 256 as before; `-DXBULK` default 0: the tf32 builds' code unchanged;
  `-DTRACE` stamps compiled out), `b200_atom/__init__.py` (the `_h` / `_hw` builds, `_f16pack` / `_f16cached` / `_f16pre` / `_f16post` /
  `_tm16`, routing in `pre32` / `post32`), `prebuild_sm100.py` (the F16 cubins).

### Round 3, part 15 (2026-11-18, base 67a6339f9 = engine 2.5.14): one load round for the 64-row row phases; a capped early pair bias

Records `$W/b2511/agents/r3-local-dit/_ld7/out/` (`fin15_<L>.jsonl` / `cmp_fin15.txt`: cuda / triton / pytorch, every requirement case of the
six classes at min samples, one measurer run per length, GPU 0 exclusive; the start is part 14's `cmp_fin14.txt`, the same code for these
classes -- the merge of 67a6339f9 changed only other classes), A / B and trace logs `$W/b2511/jobs/ld15*.log` (`_ld15/`: `j1.sh` .. `fin.sh`,
`sass.py` the SASS of one entry point).

- **Taken: `-DPRE1R` / `-DPOST1R`** on the 64-row fp16-exact builds (`_hw`): the pre loads s of both 32-row rounds before the grid-dependency
  wait and the AdaLN 1 table rows of both right after it (one load round instead of two; the cross mode keeps the per-round form); the
  post loads x2's inputs (s, so, s2, b2) one round ahead, round 0 under the Wo MMAs. 164 / 168 registers (post: 32 B spill). Bitwise equal
  to part 14 (10 cases: Protenix 1000 x 5 shared / unshared, 4000 x 5, 5000; RoseTTAFold3 / Boltz-2 / ESMFold2Pair / Chai-1 1000 x 5
  shared; OpenFold3 5000); GPU tests 123 passed. Same session (graph medians, us, part 14 -> this, two rounds): Protenix 1024 x 5 shared
  53.2 -> 51.2 / 53.2, unshared 61.4 -> 59.4, RoseTTAFold3 1024 x 5 49.1 -> 47.1, ESMFold2Pair 1024 x 5 55.3 -> 53.2, Protenix 5120 73.7 ->
  71.7; Boltz-2 1024 x 5 level. The measurer's Protenix 1024 x 5 (shared) read 53.1 (53.2 before): the graph timer moves in ~2 us steps and
  the gain sits under one.
- **What the row phase costs, measured** (`-DTRACE` stamps, Protenix 1024 x 5 shared, us after the pre's first CTA): wait ends 5.7, every
  load issued 6.3 (SASS: 8 LDG before ACQBULK, 16 right after it), round 0's tables used 8.45, round 0 done 9.4, round 1's LayerNorm
  start 9.6 -> its row maxima 11.3, row phase done 11.8 -- the same as part 14's two-round form (11.7). So round 1 takes ~1.7 us with every
  operand already in registers: the row phase is NOT bound by the two load rounds (part 14's reading was wrong) but by the row arithmetic
  of 8 warps an SM (~430 SASS instructions a round, the shuffle / redux chains); with the bias launch off the trace is the same (11.6).
  More row warps need <= 128 registers a thread (not built).
- **Rejected: the early fp32 pair bias on a capped persistent grid** (`local_bias_fwd_f32_ec`: 16 / 24 / 32 / 48 CTAs, 4 rows a thread in
  flight, 256-bit loads; also above the old early-bias row limits, where the bias was a plain launch after pre). Bitwise equal; much
  slower at every cap (us, plain -> cap 16 / 24 / 32 / 48): Protenix 12288 131.0 -> 286.7 / 225.3 / 196.6 / 165.9, 8192 86.0 -> 192.5 / 157.7 /
  139.3 / 116.8, 6144 x 5 shared 165.9 -> 239.6 / 208.9 / 194.5 / 178.2, 4096 x 5 118.6 -> 163.9 / .. / 122.9; 2048 x 5 level. 100 MB of pair
  through a few SMs is far longer than the 30 - 45 us pre it would hide behind (the pre holds its SMs: 384 threads x 144 - 168 registers
  leave no room beside it). Diff `docs/b200/rejected/local_dit_part15_capped_early_bias.diff`.
- Not changed: ESMFold2Pair master 16384 (the timeline of this tree: cond 0 - 15.1, pre 2.1 - 30.8, bias 30.8 - 61.2, attention to 97.9,
  post to 115.6; graph 118.8 in the timeline run, 120.7 in the measurer against 120.5), Protenix fp32 training 5120 x 48 (4793.2 against
  4776.8: the memory-bound products, unchanged).

Violations (H100 >= 20 us everywhere), part 14's end -> this part: 7 -> 7 cases above 0.667 (part 14's count of 6 used the re-measured
Protenix master 2048; it read 40.2 / 0.675 here on an unchanged path), none above 1.0, ladder 0.

| class | fp32 eval | master eval | fp32 train | master train |
|---|---|---|---|---|
| Protenix | 4 -> 4, 0.717 -> 0.716 (1024 x 5 shared: 53.1 against 49.4); 2048 x 5 0.694 -> 0.692, 6144 x 5 0.699 -> 0.691, 12288 0.684 -> 0.674 | 1 -> 1, 0.686 -> 0.675 (2048, unchanged path) | 1 -> 1, 0.669 (5120 x 48) | 0 |
| ESMFold2Pair | 0 | 1 -> 1, 0.668 (16384: 120.7 against 120.5) | 0 | 0 |
| Boltz-2 / RoseTTAFold3 / OpenFold3 / Chai-1 | 0 | 0 | 0 | 0 |

- Versions: `cuda:b200_sm100` 27 on the six classes. Shared files (sm_100a only): `b200_atom/fused_tf32.cu` (`PRE1R` / `POST1R`, default 0:
  the other builds' code unchanged; `-DTRACE` stamps 19 - 22 compiled out), `b200_atom/__init__.py` (the `_hw` builds' defines;
  `MW_LD15_1R=0` strips them for A / B).

## Round 3 (2026-11-17, branch `b200r3-swa`): B200 `cuda` against H100 `cuda`

Target (user, 2026-11-17): on every requirement case both cards serve, B200 `cuda` median <= H100 `cuda` median / 1.5 (ratio B200 / H100
<= 0.667; graph timing, L2 flushed). Classes of this branch: `AtomDiTESMFold2`, `SWAAtomTransformer`, `InputFeatureEmbedder` (the SWA
blocks, their stack, and the embedder's own statements). The target is **not met on every case**; the table says where.

### What changed

- **Modulation on many rows** (`sm100.mod_fwd` / `mod_bwd`): from 4096 rows the forward is torch's bf16 `silu` and one cuBLAS GEMM with
  fp32 output (the same exact products as `mod_fwd.cu`, fp32 accumulation); from 16384 rows the backward is two TF32 cuBLAS GEMMs around
  one elementwise kernel (`mod_bwd_act.cu`). `dmod` by plain stores for A <= 8 (`*_st` builds: no zero fill, no `red.add`).
- **Inference weight forms**: the bf16 forms of the six fp32 masters as views of one buffer with `[Wqkv; Wg]` adjacent
  (`sm100.master_pack`, no `cat` per replay); the fp32 block's TF32 forms and the rounded `Wmod` through `dispatch.eval_form` (a capture
  reads the forms its eager warm-up made: no `wprep` / `round` launch in a replay; `tf32_fwd._forms`).
- **bf16 inference stacks on >= 4096 rows** make `silu(c)` once for their blocks (`SWAAtomTransformer._forward_cuda`, as the H100 stack).
- **fp32 global attention glue** (`gattn_tf32.cu`, `tf32_fwd.GlobalGlue`): the fp32 -> bf16 casts of the attention's operands and the zero
  fill of its outputs in one launch (forward 3 casts + 1 fill, backward 5 casts + 3 fills), and the backward's head-wise tail (rotary
  transpose + affine-free RMSNorm backward for q and k, the v columns; `tf32_bwd._head_tail`'s equations) in one launch. They were ~10
  (forward) and ~50 (backward) elementwise torch launches per block.
- **`InputFeatureEmbedder` on the B200** (`modules/embedder.py: _forward_b200`, `embed_glue.cu`): the 3D RoPE tables and `seqused` in one
  launch (when no gradient flows into the positions); inference: the reference features + bias-free projection (+ the `"esmfold2"`
  LayerNorm) in one launch (fp32 arithmetic on the fp32 weights, the result in the activation dtype) and the token mean as a zero fill,
  one scatter launch and one normalising launch around the read-out GEMM; training: the token mean as one Function
  (`_TokenMeanB200`, backward = the gather). The scatter adds each token's run of atoms once, so with a token's atoms contiguous the sums
  are bit-reproducible -- torch's `scatter_add` is not: `InputFeatureEmbedder` float32 training differed run to run on **every** backend
  (PyTorch included) before; on the B200 `cuda` backend it is now reproducible (`_tools/det.py`, 3 runs, outputs and all gradients).
  Outside the range (non-bool masks, non-int64 indices, `d_atom` != 128, `d_token` not a multiple of 8) the statements run as before.
- `backend_versions["cuda:b200_sm100"]` 3 for `AtomDiTMiniWorld`, `SWAAtomTransformer`, `InputFeatureEmbedder`.

### Second pass (2026-11-17): few tiles -- bf16 inference of at most one 32-row tile per SM (<= 4736 rows)

`backend_versions["cuda:b200_sm100"]` 4 for `AtomDiTMiniWorld`, `SWAAtomTransformer`, `InputFeatureEmbedder`.

What limits one sample of a few thousand atoms (measured with `clock64` traces inside the row kernels, `_tools/tr.py` / `tr1.py`, and
graph timing, L2 flushed, timer floor 6.0 us for one tiny launch; the timer's step is ~2 us):

- **A CTA takes ~83 GB/s through TMA, whatever the source**: 192 KB of weights are in shared memory ~2.5 us after the kernel starts with
  the L2 flushed and ~2.3 us with a hot L2; multicast over clusters of 2 / 4 / 8 CTAs changes nothing (`ffn_fwd2` at 1024 rows: 16.2 us
  with every cluster size, `_tools/mc.py`). `tcgen05.cp` moves a 32-KB unit into TMEM in ~0.35 us (48 of them: 2.1 us), and loads queue
  behind each other. So every byte of weights a CTA needs before its tile is latency, and the order of the loads is the order things
  become ready. A TMA instruction also costs its issuing thread ~50 ns (24 weight atoms from one thread: 1.3 us before the tile's inputs
  were even requested).
- **The launches themselves cost nothing**: the three stage kernels as roles of ONE launch (CTA ranges of one grid, each role waiting on a
  counter in global memory for its predecessor's stores; bit-identical) took 28.4 - 32.6 us for the 1024-row block against 28.5 us for
  the three PDL-chained launches, and 36.7 against 30.6 us with the modulation launch in front. Reverted (the patch is kept outside the
  tree). PDL already overlaps the prologues; what a chain pays is each stage's own critical path.
- The modulation launch and the weight staging were 6 us of the 1024-row block's 30.6 (24.5 us with neither; its fp32 `[R, 768]` written to HBM and read back by two kernels), and the
  qkvg stage read its 128 KB of W through TMA -> shared memory -> `tcgen05.cp` -> TMEM before requesting its tile.
- A thread-private fp32 stash in shared memory (128 threads, 128-B stride) cost 2.5 us for 16 loads per thread; registers: 0.25 us.

Built (`kernels/swa_dit/cuda/sm100/qkvg_fwd1t.cu`, `ffn_fwd1t.cu`; `sm100.block_fwd_c`, taken by `AtomDiTMiniWorld._forward_b200` for
bf16 inference when `N S <= 32 x SMs` = 4736 rows; one tile per CTA, no loop):

- **The modulation inside the row kernels**: `a = rn(silu(c))` in place on the c tile, `Wmod` rows as a bf16 A operand read in place from
  shared memory (`tcgen05.mma` smem x smem), the fp32 results read by the row threads straight from TMEM (thread = channel = lane,
  column = row): shift_a / scale_a in the qkvg kernel, gate_a / shift_f / scale_f / gate_f in the FFN kernel (gate_f kept, rounded, in
  registers before the att MMA reuses its columns). No modulation launch, no fp32 `[R, 768]`.
- **qkvg**: W = `[Wq; Wk; Wv; Wg]` is an smem A operand too (no TMEM copy); loads one per lane in the order inputs, Wmod[0:256], W (W lands
  while a, the modulation and x are made); x is written over q; the output staging tiles overlay Wmod once its MMAs are done; the TMA
  stores one per lane. It triggers its dependents after its own wait (not at the top), so the attention and FFN kernels start under it
  with q and c final.
- **FFN**: everything that does not depend on the attention runs before the wait for it: Wmod[256:768] and c, the four modulation MMAs,
  Wu / Wd through two staging slots into TMEM (a unit's `tcgen05.cp` under the next unit's load), Wo. After the wait only q, g, o are
  loaded.
- The outputs are **the same bits** as `mod_fwd` + `block_fwd` (`_tools/ft.py`: 1024 - 4608 rows, 1 - 4 samples; `_tools/stress.py`:
  20000 graph replays of three chained blocks on nine shapes with ragged `seqused`, no hang, the same bits every time).

Block alone (`_tools/ft2.py`, us, graph timing; H100 = the module's `h100_sm90` token):

| rows | before (4 launches) | few-tile block | H100 | target (H100 / 1.5) |
|---|---|---|---|---|
| 1024 | 30.5 | 24.4 | 31.7 | 21.1 |
| 2048 | 33.6 | 26.4 | 34.2 | 22.8 |
| 3072 | 36.8 | 30.5 | 39.1 | 26.1 |
| 4096 | 42.8 | 30.6 | 44.0 | 29.3 |

Timeline of one CTA at 1024 rows (clock ticks at ~1.7 GHz, each kernel alone): qkvg 11300 = TMEM allocation 730, inputs in 3700, a
4500, modulation MMAs 5600, x 6000, projection MMAs 7700, epilogue (head RMS, RoPE, staging) 10700, stores 11300; FFN after its wait
6800 = q / g / o in, gated 770, att MMA 560, q1 + RMS + y 1700, a|b MMAs 470, h 1600, ffn MMA 600, out 420, store 620. Beyond one
wave (> 4736 rows) the few-tile kernels do not win (5120 rows 47.1 against 48.9 us, 8192 rows 53.2 = 53.2, 16384 rows 94.1 against
86.0): a CTA reloads the weights per tile, so the multi-tile `*_fwd2` kernels keep those row counts.

Not done in this pass (steps 2 - 5 of the plan): the in-kernel modulation for the multi-tile kernels (> 4736 rows), the fused float32
training backward, the 32-row-tile bf16 backward, a native global attention.

Every requirement case after the second pass (`$W/b2511/work/r3-swa/spd5.jsonl`, `spd_ife5.jsonl`, table `final5_vs.txt`; the table of
the first pass below is unchanged outside the few-tile range, up to the timer's ~2 us step):

| class | compared | miss 1.5 x: first pass -> second | slower than H100 | worst ratio | geomean |
|---|---|---|---|---|---|
| `InputFeatureEmbedder` | 108 | 64 -> 63 | 8 -> 9 | 1.050 -> 1.050 | 0.743 -> 0.721 |
| `SWAAtomTransformer` | 90 | 52 -> 51 | 3 -> 4 | 1.029 -> 1.025 | 0.681 -> 0.681 |
| `AtomDiTESMFold2` | 30 | 25 -> 25 | 4 -> 2 | 1.109 -> 1.112 | 0.814 -> 0.785 |

`AtomDiTESMFold2` master inference of one sample: 1024 atoms 32.5 -> 24.5 us (ratio 1.026 -> 0.772), 2048: 34.6 -> 26.5 (1.012 ->
0.777), 3072: 38.7 -> 30.6 (0.989 -> 0.783), 4096: 42.8 -> 30.7 (0.973 -> 0.698); the embedder's master inference of one sample with
window attention (3 blocks): 0.47 - 0.52 of the H100 at 1024 - 4096 atoms (0.57 - 0.65 after the first pass). None of the four crosses 0.667: the 1024-row block is the
sum of three stages' own critical paths (6.2 + 6.1 + 6.1 us over the 6.0 us floor), and from 3072 rows the 3 x 96 - 144 CTAs no longer
fit the 148 SMs at once. The counts of "slower than H100" move by one on cases this pass does not touch (float32 / master training at
ratio 1.00 - 1.03). Ladder: 0 violations (worst 1.007); 48 cases refused as before. Accuracy (`acc5_*.jsonl`): no RED, the lights of
the first pass (the few-tile path has the old path's bits). Determinism: `_tools/det.py` 0 bad.

### Third pass (2026-11-17): many tiles -- bf16 inference on more rows than one tile per SM (> 4736 rows)

`backend_versions["cuda:b200_sm100"]` 5 for `AtomDiTMiniWorld`, `SWAAtomTransformer`, `InputFeatureEmbedder`.

Built (`kernels/swa_dit/cuda/sm100/qkvg_fwd3.cu`, `ffn_fwd3.cu`; `sm100.block_fwd_c` takes them outside the few-tile range, and
`AtomDiTMiniWorld._forward_b200` now takes `block_fwd_c` for every bf16 inference call in the kernels' range -- d_atom 128, hidden 256,
window 64 or global; a `SWAAtomTransformer` stack in that range no longer makes a shared `silu(c)`): the few-tile design as a loop over
a CTA's 32-row tiles (grid = min(tiles, SMs)), the modulation inside (no modulation launch, no fp32 `[R, 768]` in HBM: a qkvg tile reads
20 KB instead of 44 KB, an FFN tile 32 KB instead of 88 KB). The outputs are **the same bits** as `mod_fwd` + `block_fwd`
(`_tools/ft.py`: 1 x 5120 - 16384, 5 x 1024 - 8192 rows and, forced, the few-tile shapes; `_tools/stress.py`: 5000 graph replays of
three chained blocks on eleven shapes with ragged `seqused`, no hang, the same bits every time: `stress7.log`).

What was measured on the way (`clock64` per tile, `_tools/tr3.py`, TRACE builds; ticks at ~1.7 GHz):

- **A cold TMA load lands ~2.3 us (3900 ticks) after its issue, a TMA store is read out of shared memory ~1.9 us after its issue.**
  A tile stage that is released by the tile's own epilogue / store therefore stalls the row threads on every tile: the first
  `qkvg_fwd3` (two input stages, released at the store) spent 3900 of 9500 ticks per tile waiting for the next tile's inputs, and
  with one output staging set another 740 - 950 ticks waiting for the previous tile's stores. Loads have to be issued **two tiles
  ahead**, and what a store reads must not be needed again for a tile period.
- qkvg: the x / a tile is dead as soon as the projection MMAs have read it, so their `tcgen05.commit` arrives a second barrier
  (`xcfree`) that releases the buffer for tile T + 2; cos / sin sit in four small buffers; two output staging sets (the second is
  W's last slot: Wg moves on into TMEM, 64 columns, before the first projection). 9500 -> 5650 ticks per tile, all of it row-thread
  work now (a / RMS / x 1950, epilogue 3400).
- FFN: a tile was 7850 ticks of row-thread work and MMA waits in sequence. Now the MMA waits hide the next tile's work (`a(T + 1)`
  under the a | b MMAs, `gated(T + 1)` under the ffn MMA), and the next tile's modulation MMAs are issued behind the ffn MMA (three of
  them at once -- their columns are read -- the fourth, into the ffn columns, when tile T's ffn has been read). 7850 -> 6900 ticks.
- Every barrier is waited by its one consumer before its producer can complete the next phase (per-buffer barriers, or strict
  alternation), so no wait depends on a thread polling in time (the parity ABA the round-2 kernels rely on timing for).

Block alone (`_tools/ft3.py`, us, graph timing, L2 flushed; before = `mod_fwd` + `block_fwd`):

| rows (samples x atoms) | before | first build | overlapped tiles | H100 | target (H100 / 1.5) |
|---|---|---|---|---|---|
| 1 x 5120 | 46.9 | 40.7 | 38.7 | 55.9 | 37.3 |
| 1 x 8192 | 55.1 | 44.8 | 42.8 | 66.2 | 44.1 |
| 1 x 16384 | 83.8 | 65.3 | 59.2 | 119.6 | 79.7 |
| 5 x 1024 | 47.0 | 40.8 | 38.7 | 55.1 | 36.7 |
| 5 x 8192 | 171.8 | 122.7 | 106.3 | 265.7 | 177.1 |

Tried and reverted in this pass: the fp32 training block's weight-form launches (`wprep_tf32`, `wprep_bwd_tf32`) on a side stream
beside `round(Wmod)` + the modulation -- no change (ESMFold2 float32 training L 1024 159.9 -> 161.7 us; the 56 float32 training cases
move by -5.6 % to +2.0 % with no direction; `spd7_ft.jsonl`): those launches are not what the chain waits on. A plain `wait_group` in place of `wait_group.read`
after the stores: no change.

Every requirement case after the third pass (`$W/b2511/work/r3-swa/spd6.jsonl`, `spd_ife6.jsonl`, tables `final6_vs.txt`; groups by
`_tools/groups.py`):

| class | compared | miss 1.5 x: second pass -> third | slower than H100 | worst ratio | geomean |
|---|---|---|---|---|---|
| `InputFeatureEmbedder` | 108 | 63 -> 62 | 9 -> 8 | 1.050 -> 1.049 | 0.721 -> 0.706 |
| `SWAAtomTransformer` | 90 | 51 -> 41 | 4 -> 2 | 1.025 -> 1.029 | 0.681 -> 0.629 |
| `AtomDiTESMFold2` | 30 | 25 -> 22 | 2 -> 2 | 1.112 -> 1.110 | 0.785 -> 0.745 |

| class | dtype, mode, attention | cases | miss | slower | ratio | missing at L |
|---|---|---|---|---|---|---|
| `AtomDiTESMFold2` | master eval | 10 | 7 | 0 | 0.512 - 0.779 | 1024 - 7168 |
| `AtomDiTESMFold2` | float32 eval | 10 | 5 | 1 | 0.456 - 1.110 | 1024 - 5120 |
| `AtomDiTESMFold2` | float32 train | 8 | 8 | 1 | 0.708 - 1.018 | all |
| `AtomDiTESMFold2` | master train (L 7168, 8192) | 2 | 2 | 0 | 0.969 - 0.980 | both |
| `SWAAtomTransformer` | master eval, window, 1 block | 8 | 1 | 0 | 0.408 - 0.706 | 1024 |
| `SWAAtomTransformer` | master eval, window, 3 blocks | 8 | 1 | 0 | 0.410 - 0.725 | 1024 |
| `SWAAtomTransformer` | master eval, global | 8 | 6 | 0 | 0.617 - 0.847 | 1024 - 8192 |
| `SWAAtomTransformer` | float32 eval, window (1 / 3 blocks) | 16 | 2 | 0 | 0.416 - 0.700 | 1024 |
| `SWAAtomTransformer` | float32 eval, global | 8 | 6 | 0 | 0.633 - 0.796 | 1024 - 8192 |
| `SWAAtomTransformer` | master train, window S 48 | 8 | 1 | 0 | 0.570 - 0.685 | 1024 |
| `SWAAtomTransformer` | master train, global S 48 | 8 | 8 | 0 | 0.783 - 0.854 | all |
| `SWAAtomTransformer` | master train, window S 1 (L 7168, 8192) | 2 | 2 | 1 | 0.985 - 1.027 | both |
| `SWAAtomTransformer` | float32 train, window S 48 | 8 | 0 | 0 | 0.584 - 0.628 | |
| `SWAAtomTransformer` | float32 train, global S 48 | 8 | 6 | 0 | 0.587 - 0.707 | 3072 - 8192 |
| `SWAAtomTransformer` | float32 train, window S 1 (3 blocks) | 8 | 8 | 1 | 0.724 - 1.029 | all |
| `InputFeatureEmbedder` | master eval, window | 24 | 0 | 0 | 0.459 - 0.562 | |
| `InputFeatureEmbedder` | master eval, global | 24 | 16 | 0 | 0.579 - 0.813 | 1024 - 6144 |
| `InputFeatureEmbedder` | float32 eval, window | 16 | 4 | 0 | 0.508 - 0.746 | 1024 - 2048 |
| `InputFeatureEmbedder` | float32 eval, global | 8 | 6 | 0 | 0.611 - 0.898 | 1024 - 6144 |
| `InputFeatureEmbedder` | float32 train, window | 16 | 16 | 1 | 0.748 - 1.006 | all |
| `InputFeatureEmbedder` | float32 train, global | 8 | 8 | 1 | 0.823 - 1.027 | all |
| `InputFeatureEmbedder` | master train, window (L 7168, 8192) | 6 | 6 | 0 | 0.966 - 0.994 | both |
| `InputFeatureEmbedder` | master train, global (L 7168, 8192) | 6 | 6 | 6 | 1.018 - 1.049 | both |

No case of the three classes has an H100 median under 20 us (the exemption of 2026-11-17 noon does not apply; the smallest is 31.7 us).
Ladder: 0 violations (worst cuda / triton 1.012); 48 cases refused as before (master training of one sample below 7168 rows: 6
ESMFold2, 6 SWA, 36 embedder). Accuracy (`acc6_a..d.jsonl`, the `cuda` records): no RED; GREEN 93, YELLOW 52 (22 refused, 8 skipped), the lights of the
second pass -- one YELLOW where the B200 token was GREEN, the case already noted in the first pass (embedder float32 inference L 1024
window, 1.01 x tol; triton 1.74, H100 cuda 1.62). Determinism: `_tools/det.py` 0
bad. `_prebuild --arch sm100 --list` has the two new cubins (`swa_dit_sm100/qkvg_fwd3`, `ffn_fwd3`; `prebuild_list6.txt`).

Still above 0.667 after the third pass, and why:

| group | cases | why (numbers) | what it would take |
|---|---|---|---|
| bf16 inference, one tile per SM or a little more (ESMFold2 L 1024 - 7168; SWA 5 x 1024) | 9 | the block is three dependent kernels, each its own critical path over the 6 us floor: L 1024 24.5 us against a target of 21.1, 5 x 1024 / L 5120 38.7 against 36.7 / 37.3 (the 12 CTAs with a second tile add qkvg 3.3 + FFN 4.1 us) -- one or two 2-us timer steps | a faster qkvg epilogue (3400 of 5650 ticks per tile: RMS reduce-scatter, RoPE partner shuffles, 96 shared loads and 32 16-bit stores per thread) and q1 / y phase (1700 ticks); tiles of 40 - 64 rows so that 5120 - 8192 rows stay one tile per CTA |
| global attention (SWA master / float32 eval and S 48 training; embedder global -- 14 of them embedder training cases counted below too) | 62 | FlashAttention-4 (external) is 45 - 50 % of a step at one sample and most of it at S 48; the row stages around it are the ones above | a native tcgen05 global attention that beats FA4 (not started; nothing measured says it would) |
| float32 inference of one sample (ESMFold2 L 1024 - 5120, SWA / embedder window L 1024 - 2048) | 11 | 51 us flat from 1024 to 4096 rows: 128-row tiles (8 - 32 CTAs) whose 448 KB of fp32 weights stream through a ring per tile (`ffn_fwd_tf32` 192 MMAs and 56 weight slots per tile) behind a modulation launch | TF32 row kernels of the few-tile kind: 32-row transposed tiles over every SM, weights resident, the modulation inside -- new kernels (Wu + Wd alone are 768 TMEM columns in fp32, so the weights have to stay shared-memory operands and be split over CTA pairs) |
| float32 training of one sample | 40 | ~21 dependent launches per block at ~7.5 us each (ESMFold2 L 1024: 160 us for one block); taking launches off the side (weight forms) changes nothing | fewer links in the chain: the GEMMs inside `adaln_bwd` (2 x 2 launches), dQ + dK / dV in one kernel, `round` + `mod_fwd`, the modulation backward's four launches in one |
| master training of one sample (L 7168 / 8192; below 7168 rows refused) | 16 (+ 48 refused) | the backward's 128-row tiles: 12 dependent kernels, 219 us at L 8192 (`oproj_bwd` 42, `ffn_bwd_dy` 31, `dw` + reduce 55, `mod_bwd` 51, `qkvg_bwd` 25, dq / dkv 43 under the profiler) | 32-row-tile backward row kernels with the two-tiles-ahead loads (the training forward `qkvg_fwd2` releases its stage at the store: by the numbers above it would stall ~4 us per tile -- not traced; its saves X / PQ / PK share the stage with the next tile's x, so it needs a third stage or its own save buffers) |
| master training window S 48, L 1024 | 1 | 0.685: 706 us against 688 | the same `qkvg_fwd2` rework (10 tiles per CTA), if the stall is there |

### Part 4 (2026-11-17): float32 training -- the backward's dependent chain

`backend_versions["cuda:b200_sm100"]`: `AtomDiTMiniWorld` 6, `SWAAtomTransformer` 7, `InputFeatureEmbedder` 7 (6 is the swa-global
merge's for the last two).

Built (`kernels/swa_dit/cuda/sm100/tf32_bwd.py`, `_bwd_few`: float32 backward of at most one 128-row tile per SM, `SCHED = 2`; the
same kernels and the same bits as the one-after-the-other path, `_tools/rw.py`):

- **The backward's launches as their dependencies allow**: dK / dV beside dQ (side stream) when both passes fit the SMs; dWo / dWu / dWd
  slices beside the attention when they fit too, dWqkv | dWg and the reduction after it on side stream 1; `silu(c)` of the modulation's dW
  at the start and the dW GEMM beside dc on side stream 2. The current stream carries only the chain ffn_bwd_gate -> dy GEMM -> adaLN ->
  oproj -> dQ -> dx GEMM -> adaLN -> d mod Wmod -> dsilu.
- **Warp-per-row adaLN backward of one augment** (`swa_adaln_bwd1_tf32_sm100`): the per-atom kernel launched a block per atom whose seven
  other warps found no augment (8192 blocks at 8192 atoms: ~30 us, twice per block). Both adaLN kernels now PDL-chain (wait, then trigger).
- **Item kernels for few tiles**: `ffn_bwd_gate_tf32` takes (tile, chunk group) items (NG = 1 / 2 / 4 / 8 groups of the eight hidden
  chunks: an item streams 64 KB of weights per chunk instead of the tile's 512 KB), `oproj_bwd_fused_tf32` (tile, head group) items.
- **`dw_tf32`**: slices of 64 rows (was 256: one sample of 1024 atoms ran each problem on 4 CTAs), a stage's boxes issued one per lane
  (one thread issuing 20 loads per stage spent ~37 ns on each), and the partial sums leave by TMA stores from 128-B-swizzled tiles staged
  over the (then idle) ring: the rows' own 16-B stores (one row per lane, 512 B apart) were 10 of a 64-row slice's 14.5 us.
- `ffn_fwd_tf32 -DROWS=64 / 32` builds (smaller tile slots, deeper weight ring) for few tiles of one augment (`tile_rows`).
- `adaln_bwd` bind: absent operands passed as `None` (an 8-byte null pointer) -- a Python `0` went into the 8-byte pointer parameter as
  4 bytes.

Measured on the way (`_tools/tl.py`: one replay's kernel timeline under CUPTI; ESMFold2 float32 training, one sample of 1024 atoms, us
from the flush kernel's end): the forward is ~52 us (round 2, `mod_fwd` 7.6, `qkvg` 13, attention 7, `ffn_fwd` 21 after the attention),
the backward ~57 us: `ffn_bwd_gate` 8.5 after `ffn_fwd`, dy GEMM 5.3 + adaLN 3.0, `oproj` 9.6, dQ / dK dV 13, dx GEMM 5.1 + adaLN 2.1,
d mod Wmod 6.7 + dsilu 2.0. Every link costs ~4 - 5 us however little it computes (a cold TMA load lands ~2.3 us after its issue, and a
dependent kernel's first load waits for its predecessor's last store).

Tried and removed (patches kept outside the tree):

- **The chain's GEMMs as own split-K tcgen05 kernels** (`mm_tf32.cu`: (128-row tile, K-slice) CTAs, one 32-column K block each, partials
  stored by TMA, summed by the adaLN / dsilu kernels in slice order; PDL; weights loaded before `griddepcontrol.wait`): ESMFold2 float32
  training L 1024 110.6 -> 118.6 us, L 2048 120.6 -> 128.9, L 8192 188.3 -> 194.6. A stage is ~4 us whatever it computes, so a GEMM
  kernel of our own is not faster than cuBLAS's, and 16 partials made the adaLN read 8 MB at L 1024.
- **GEMM + adaLN backward in one launch** (`mm_adaln_tf32.cu`: a cluster of 2 / 4 / 8 CTAs per 128-row tile splits K, the partial
  [128][128] parked in shared memory, each CTA reduces its rows over distributed shared memory in rank order and runs the adaLN of
  them): L 1024 110.6 -> 114.5 (NS 8) / 118.6 (NS 4) / 129.0 (NS 2), L 2048 121.0 -> 134.9 / 126.8 / 135.0. The fused dy stage ended
  where cuBLAS + adaLN ended (8.4 us after `ffn_bwd_gate`); the dx one started late: clusters of 8 one-CTA-per-SM blocks wait for
  whole free GPC slices while `dw_tf32` holds SMs on the side stream (L 2048: 21.7 us after dK / dV).
- **d mod Wmod split** (columns 2C .. 6C right after the first adaLN, beside the chain; columns 0 .. 2C after the second; dsilu adds the
  slices): no gain -- the event wait in front of the late part breaks its PDL edge.
- **`ffn_fwd_tf32` cluster build** (a tile per cluster of NG CTAs, rank g streaming Wo and only 8 / NG hidden chunks of Wab / Wd: 160 KB
  at NG 4 instead of 448 KB, the partial ffn reduced over distributed shared memory): float32 inference L 1024 46.9 -> 69.4 (NG 8) / 47.1
  (NG 4) / 46.9 (NG 2) us, training L 1024 108.6 -> 131.1 (NG 8). Streaming a third of the weights per CTA changes nothing: the FFN
  stage of a few tiles (17.6 us after the attention in float32 inference, L 1024: `mod_fwd` 7.7, `qkvg` 11.4, attention 5.7, FFN 17.6) is
  not bound by its weight stream, so item 3 of the plan (float32 inference with the weights split over CTAs) was not pursued further.

Float32 training after part 4 (`$W/b2511/work/r3-swa/p4/spd8_ft.jsonl`; ratio B200 / H100; before = the third pass, `final6_vs.txt`):

| class | dtype, mode, attention | cases | above 0.667 before -> after | ratio after |
|---|---|---|---|---|
| `AtomDiTESMFold2` | float32 train | 8 | 8 -> 4 | 0.539 - 0.808 (L 1024 - 4096 above) |
| `SWAAtomTransformer` | float32 train, window S 48 | 8 | 0 -> 0 | 0.398 - 0.448 |
| `SWAAtomTransformer` | float32 train, global S 48 | 8 | 6 -> 0 | 0.457 - 0.615 |
| `SWAAtomTransformer` | float32 train, window S 1 (3 blocks) | 8 | 8 -> 4 | 0.533 - 0.717 (L 1024 - 4096 above) |
| `InputFeatureEmbedder` | float32 train, window | 16 | 16 -> 8 | 0.575 - 0.817 (L 1024 - 4096 above) |
| `InputFeatureEmbedder` | float32 train, global | 8 | 8 -> 8 | 0.741 - 0.908 |

None of the 48 float32 training cases is slower than the H100 any more (was 4). Accuracy (`acc8_ft.jsonl`, float32 training of the three
classes at L 1024 / 2048 / 4096 / 8192): no RED, no case worse than its B200 token (`_tools/acccmp.py`). Graph replays
(`_tools/stressft.py`: ESMFold2 L 1024 / 3072 / 8192, SWA 3 blocks L 1024 / 5120, embedder L 2048; 2000 replays each): no hang, the same
gradient bits at every 100th replay. `_prebuild --arch sm100 --list` lists `ffn_fwd_tf32_r64` / `_r32` (`p4/prebuild_list8.txt`).

Found on the way: the swa-global merge put its `kb` box edit into `AttnFwd2.bind` (window attention forward) and `AttnDq.bind` (window dQ backward) instead of only `GattnFwd.bind`, where
`self.cl` / `kb` do not exist -- every bf16 / master window inference block raised `AttributeError` and every bf16 / master
window training backward `NameError` on `b200-2.5.x` 0862a4516. Fixed: `AttnFwd2` without the line, `AttnDq` back to its [D, 64] K / V
boxes.

What is left of the float32 training cases above 0.667 (one sample, L 1024 - 4096; embedder global all lengths): the chain of ~17
dependent stages (6 forward, 11 backward) at ~4 - 5 us each against the H100's 157 us (ESMFold2 L 1024: 110 us, target 105); the
embedder's global cases keep FlashAttention-4's forward + backward in training. Shortening it further needs fewer stages, not faster
ones: e.g. the dy GEMM inside `ffn_bwd_gate` (its hidden chunks are dy's K: partial dy per item, summed by the adaLN), the dx GEMM inside
the attention backward's epilogues, the modulation's dsilu inside a kernel that already holds `dmod`.

Every requirement case after part 4 and the swa-global merge (`$W/b2511/work/r3-swa/p4/spd8_ft.jsonl`, `spd9.jsonl`, `spd_ife9.jsonl`,
`spd10_mt.jsonl`; table `p4/final10_vs.txt`; before = the third pass, `final6_vs.txt`):

| class | compared | miss 1.5 x: third pass -> part 4 | slower than H100 | worst ratio | geomean |
|---|---|---|---|---|---|
| `InputFeatureEmbedder` | 108 | 62 -> 44 | 8 -> 6 | 1.049 -> 1.047 | 0.706 -> 0.649 |
| `SWAAtomTransformer` | 90 | 41 -> 27 | 2 -> 2 | 1.029 -> 1.027 | 0.629 -> 0.575 |
| `AtomDiTESMFold2` | 30 | 22 -> 19 | 2 -> 1 | 1.110 -> 1.017 | 0.745 -> 0.682 |

The float32 training groups moved by this part's changes; the global inference groups by the swa-global merge. Still slower than the
H100: embedder master training global L 7168 / 8192 (6 cases, 1.018 - 1.047: FlashAttention-4's training forward + backward and the
128-row bf16 backward), SWA master training window one sample L 7168 / 8192 (1.010 / 1.027: the 128-row bf16 backward), ESMFold2 float32
inference L 1024 (1.017: the four-stage chain at ~4 - 5 us a stage). 48 cases refused as before (master training of one sample below 7168
rows). Ladder: 0 violations (worst cuda / triton 0.991). Determinism (`_tools/det.py`, `p4/det9.log`, `det10.log`): 0 bad. Master training
accuracy at L 7168 / 8192 (`acc10_mt.jsonl`): no RED; the embedder global relu L 7168 case is YELLOW 1.98 x tol (H100 1.64, triton 1.34;
no B200 token for it) -- a bf16 path this part does not touch.

### Part 5 (2026-11-17): master training of one sample on the TF32 kernels

`backend_versions["cuda:b200_sm100"]`: `AtomDiTMiniWorld` 7, `SWAAtomTransformer` 8, `InputFeatureEmbedder` 8.

- **Master training of window attention on at most 8192 rows** (`_B200_MT_TF32_MAX_ROWS`; every requirement case of one sample) runs
  the float32 training block of part 4 (`_SWATrainB200` with fp32 x / c and the fp32 master weights as they are: no `master_pack`
  cast): x and c cast to fp32, the output back to bf16, autograd casting the gradients. A stack (`SWAAtomTransformer`, the embedder's
  atom transformer) casts q and c once for all its blocks. Before, the bf16 stages refused these calls below 7168 rows (128-row tiles,
  8 - 48 CTAs a launch) and served L 7168 / 8192 at 0.96 - 1.06 x the H100. The S 48 calls (49152+ rows) stay on the bf16 stages
  (L 1024 window 0.708 ms bf16 against 0.823 TF32).
- **Global attention stays on the bf16 path** (refused below 7168 rows, as before): through the TF32 row stages it measured 1.07 - 1.36 x
  the H100 at L 1024 - 6144 and one case (embedder relu L 5120) at 1.66 x tol against the H100's 1.47 / triton's 1.41.
- **Bug fixed on the way**: `tf32_bwd.mm_tf32` (the backward's cuBLAS GEMMs) and `_SWATrainB200.backward` now run with autocast off. A
  master backward runs under the caller's bf16 autocast, which turned the TF32 GEMMs into bf16 ones -- in eager mode on side streams
  this gave `CUBLAS_STATUS_EXECUTION_FAILED` and run-to-run different gradients (`_tools/det_mt.py`). float32 calls never run under
  autocast: unchanged bits.
- Tried and removed: the four casts as two `torch._foreach_copy_` launches (x | c and dx | dc): ESMFold2 L 1024 0.131 -> 0.145 ms (the
  multi-tensor kernel runs a few CTAs per tensor).

Measured (`$W/b2511/work/r3-swa/p5/s6.jsonl`, `vs6.txt`; ratio B200 / H100 cuda; master training, one sample):

| class, case | cases | before | after |
|---|---|---|---|
| `AtomDiTESMFold2` | 8 | 6 refused, L 7168 / 8192 0.977 / 0.960 | 1.170 1.188 1.080 1.068 0.941 0.907 0.916 0.907 |
| `SWAAtomTransformer` 3 blocks, window | 8 | 6 refused, L 7168 / 8192 1.027 / 1.010 | 1.004 1.012 0.975 0.954 0.866 0.864 0.851 0.838 |
| `InputFeatureEmbedder` window (3 variants) | 24 | 18 refused, L 7168 / 8192 0.96 - 0.99 | 0.944 - 1.034 (L 1024 - 4096), 0.838 - 0.879 (L 5120 - 8192) |
| `InputFeatureEmbedder` global (3 variants) | 24 | 18 refused, L 7168 / 8192 1.018 - 1.047 | unchanged (bf16 path, FlashAttention-4) |

Every served case beats the B200 triton token (cuda / triton 0.76 - 0.99). Of the 30 newly served cases 8 are slower than the H100
(ESMFold2 L 1024 - 4096, SWA L 1024 / 2048, embedder linear L 3072 and relu L 2048): the float32 block's ~17 dependent stages plus the
casts (~20 us at L 1024: x, c, out, dy, dx, dc -- `p5/tl_mt1024.txt`) against the H100's bf16 64-row slab chain (ESMFold2 L 1024:
131 against 112 us). Accuracy (`p5/acc_mt6.jsonl`, all lengths): no RED; ESMFold2 and SWA S 1 GREEN on every case (e / tol 0.25 -
0.44, were up to 1.05 on the bf16 path); embedder window cases at or below the triton / H100 tokens. The three cases above both
tokens are on paths this part does not change (embedder global relu L 7168 1.98 and linear L 8192 1.34, SWA global S 48 L 5120 1.01).
Determinism (`_tools/det_mt.py`, `p5/det_mt3.log`, eager, 3 runs): same bits; graph replays (`_tools/stressmt.py`, `stressmt_e.py`
with `DT=master`, 2000 replays each: ESMFold2 L 1024 / 3072 / 8192, SWA L 1024 / 5120, embedder window L 2048 / 6144): no hang, the
same gradient bits.

What remains for master training: the bf16 block backward of few tiles (32 / 64-row tiles, item kernels as `_bwd_few` has for TF32)
with the casts gone (bf16 operands read in place) -- the H100 form; the TF32 route is the fallback that beats triton meanwhile.

### Part 6 (2026-11-17): master training without casts, dy inside the FFN gate kernel

`backend_versions["cuda:b200_sm100"]`: `AtomDiTMiniWorld` 8 (`AtomDiTESMFold2` inherits it), `SWAAtomTransformer` 9, `InputFeatureEmbedder` 9.

- **Master training reads bf16 in place** (`_B200_MT_IO16`): the TF32 kernels take x / c / dy as bf16 (TMA boxes of 32 bf16, 64-B
  swizzle, converted in the loaders of `mod_fwd_tf32`, `qkvg_fwd_tf32`, `ffn_fwd_tf32`, `ffn_bwd_gate_tf32`) and write out / dx / dc as
  bf16 (`ffn_fwd_tf32` staging, `adaln_bwd_tf32` `_r16` / `_xo16` entries, `mod_bwd_tf32` `_c16` entries). The six casts of part 5 are
  gone; the same bits as part 5 (`_tools/eq6.py`, `p6/eq2.log`: 0 differing tensors). A stack reads q in bf16 in its first block and
  writes bf16 from its last.
- **dy = [da | db] Wab inside `ffn_bwd_gate_tf32`** (DYF build, few tiles; the H100 fp32 block's "dy inside the FFN backward"): [da | db]
  of each hidden chunk is written back over its a | b TMEM columns and used as the A operand of a second MMA pass against Wab (MN-major,
  128-B swizzle with 32-B atoms); the a | b / dh accumulators are single-buffered to free 128 TMEM columns for dy. Each chunk group
  stores a partial; the adaLN backward sums the NG partials in group order (deterministic). The cuBLAS dy GEMM launch is gone (another
  K order: rel. diff <= 4e-4, `p6/dyf1.log`).
- Measured dead ends (stash / `scratchpad/b2/r3swa6_rejected_msplit_dxf.diff`): (1) the modulation backward split at column 2C
  (dmod[:, 2C:] Wmod on a side stream during the attention backward, K = 2C on the chain): cuBLAS picks an sm_80 kernel for the strided
  K = 256 slice and the side GEMMs delay dK / dV: ESMFold2 L 1024 108.8 -> 114.5 us. (2) dx = dP W + the qkvg adaLN backward + dc in
  one CUDA-core kernel (a warp per row, shuffle-broadcast GEMV, fp32 FMA): the kernel alone 42.7 us against 13 us for cuBLAS dx GEMM +
  adaLN + g Wmod + dsilu; 108.8 -> 143.2 us. The fold needs tensor cores (the qkvg-backward stage of the H100), which the earlier
  separate tcgen05 GEMM + adaLN kernel also did not win with.

Measured (`$W/b2511/work/r3-swa/p6/fin_a.jsonl` lengths 1024 / 2048, `fin_b.jsonl` 3072 - 8192, separate runs; `fin_vs.txt`;
ratio B200 / H100 cuda; "was" = the round's start tokens):

| class | cases | above 0.667 | slower than H100 | worst | geomean |
|---|---|---|---|---|---|
| `AtomDiTESMFold2` | 32 | 24 (24) | 1 (9) | 1.035 (1.223) | 0.739 (0.907) |
| `SWAAtomTransformer` | 96 | 33 (66) | 0 (6) | 0.963 (1.099) | 0.591 (0.765) |
| `InputFeatureEmbedder` | 126 | 64 (81) | 8 (59) | 1.049 (1.567) | 0.678 (1.071) |

The 8 master-training cases that part 5 left slower than the H100: ESMFold2 L 1024 - 4096 1.07 - 1.19 -> 0.91 - 0.97, SWA L 1024 / 2048
1.00 - 1.01 -> 0.95 - 0.97, embedder window L 2048 relu / L 3072 linear 1.03 -> 1.010 / 1.002 (within run noise of the H100). ESMFold2
float32 training L 1024: 110 -> 108.8 us. Ladder: 0 violations (cuda / triton worst 0.931). Accuracy (`p6/acc1.jsonl`): no case worse
than before (master ESMFold2 GREEN 0.37 - 0.44). Determinism (`p6/det1.log`, 3 eager runs) and 2000-replay graph stress
(`stressmt` / `stressmt_e` / `stressft`): same bits, no hang.

Still slower than the H100: ESMFold2 float32 inference L 1024 (1.035; 4 launches mod -> qkvg -> attention -> FFN, the modulation 8 us at
the chain head), embedder global master training L 7168 / 8192 (1.02 - 1.05, FlashAttention-4 path, agent `r3-swa-global`). Master
training of one sample at 0.79 - 0.97 is the TF32 block's ~15-stage dependent chain (L 1024: forward 52 us, backward 55 us); under
0.667 needs the bf16 slab chain of the H100 (64-row tcgen05 item kernels, bf16 operands at twice the TF32 rate) with the dx GEMM and the
modulation backward inside tensor-core stages -- not built in this part.

### Part 7 (2026-11-17): the float32 global training accuracy, the modulation inside the fp32 qkvg stage

`backend_versions["cuda:b200_sm100"]`: `AtomDiTMiniWorld` 9 (`AtomDiTESMFold2` inherits it), `SWAAtomTransformer` 12, `InputFeatureEmbedder` 12.

**Accuracy: InputFeatureEmbedder float32 training, global attention (row 136), L 7168 / 8192 at 2.45 / 2.52 x tol** (B200 tokens 1.77 /
2.03, H100 2.22 / 2.03). Bisected (records `$W/b2511/work/r3-swa/p6/bis_*.jsonl`, `p7/bis_*.jsonl`, `p7/a2_*` - `a4_*`):

- by commit: every tree from the wiring round's last commit before the token (1b644124d, 2026-11-16 23:39) to today gives 2.45 / 2.51 --
  none of the window-side changes of parts 4 - 6 moved these numbers;
- by switch on the current tree: few-tile schedule off, DYF off, backward PDL off, `dw_tf32` 256-row slices, the modulation as its own
  Function, the global glue on the old torch statements, FA4's backward in place of the native one: 2.45 / 2.51 every time;
- PyTorch's FlashAttention-2 for the global attention (forward + backward): 1.77 / 2.03, the token values exactly, at every length
  (L 1024 - 8192: 2.70 2.87 2.85 1.68 2.12 2.16 1.77 2.03 against the tokens 2.72 2.87 2.85 1.68 2.12 2.15 1.77 2.03). FA2's backward alone
  changes nothing: the forward is the difference.

The tokens were measured before FlashAttention-4 was installed on the B200 (the route then fell back to FA2); since then the fp32
block's global attention is FA4's forward. Both are bf16 attentions on the same TF32-valued heads, and these gradients' error is that
rounding: FA4's sample is better at L 1024 - 3072 (2.27 2.25 2.57) and worse at L 4096 - 8192 (1.87 2.13 2.21 2.45 2.52). Fix: float32
training of global attention on one sequence (N = 1) of at least 4096 atoms takes FA2's forward (`tf32_fwd.FA2_TRAIN_FWD`,
`FA2_TRAIN_MIN_S`; the backward stays the native `gattn_bwd.cu` on FA2's natural-log LSE). Cost: FA2's forward is slower
(graph timing, row 136, ms: L 4096 1.061 -> 1.214, 5120 1.211 -> 1.400, 6144 1.346 -> 1.561, 7168 1.481 -> 1.719, 8192 1.618 ->
1.887; against the H100 0.76 -> 0.88, 0.67 -> 0.78, 0.67 -> 0.78, 0.62 -> 0.72, 0.61 -> 0.71 -- none slower than the H100, two more above
0.667). After the fix (`p7/v1_SW_X-1.jsonl`) row 136 measures 2.27 2.25 2.57 1.68 2.12 2.15 1.77 2.02, at or below the B200 token at
every length. Master training and `SWAAtomTransformer` S 48 (N = 48) keep FA4: at S 48 FA2's sample is not better (L 1024 3.54 against
FA4's 3.13, token 3.03).

**ESMFold2 float32 inference L 1024: the modulation inside the qkvg stage** (`qkvg_fwd_tf32 -DMODF`, `mod_fwd_tf32 -DLATE_TRIGGER`,
`tf32_fwd.block_fwd_tf32_c`): for one augment on at most 1792 rows the qkvg stage loads c into ring slots 4 - 7, the row threads apply
silu + TF32 rounding in place, the MMA warp computes shift_a | scale_a = silu(c) Wmod[0, 2C)^T into TMEM columns 128 - 383 (Wmod rows
streamed ahead of W through the weight ring), and pass 2 reads them from TMEM. The modulation kernel runs beside it for the FFN's
columns: built with its PDL trigger after its own `griddepcontrol.wait`, so the qkvg stage needs no wait before its loads; the qkvg
stage waits once at its end, so its completion implies the modulation's (the FFN waits for the attention, the attention for qkvg).
The same bits as `mod_fwd_tf32` + `block_fwd_tf32` (`_tools/modf.py`: 0 differing elements; 2000 graph replays the same bits). Block
alone (graph replay, L2 flushed): 1024 rows 49.2 -> 47.1 us; 2048 rows (the modulation on 3 channel groups to fit beside qkvg)
49.1 -> 49.2, 3072 rows (1 group) 49.2 -> 53.2 -- with fewer groups the modulation ends after the attention, so the route is limited
to 6 groups (10 ntile <= 148 SMs). Measured with the measurer (`p7/v3_*.jsonl`, MODF off -> on): ESMFold2 float32 inference L 1024 S 1 48.9 -> 46.9 us
(H100 46.0: 1.065 -> 1.021, one 2-us timer step above the H100), L 2048 48.5 -> 47.1 (noise: the route does not serve it); embedder /
SWA float32 inference unchanged within noise (their blocks run 3 per step; the route serves their L 1024 blocks). Determinism
(`_tools/det7.py`, 3 eager runs): same bits; accuracy of the float32 inference rows L 1024 / 2048 (`p7/v2_SW_X-1.jsonl`): every case at or
below its B200 token except the known embedder legacy window L 1024 (1.01 against 0.99; triton 1.74, H100 1.62; the same bits as
before this part).

**Not built**: the bf16 64-row tcgen05 slab chain for master training (item 2 of the plan) and the S 48 window master training L 1024
case (0.688). Measured reasons: the slab chain is a new set of six bf16 64-row tcgen05 backward kernels
(FFN gate + dy, adaLN, out-projection, dQ / dK dV, dx + adaLN, modulation backward) -- not started in this part (the accuracy bisect and
the MODF build took it); parts 4 - 6 measured that each stage of the present chain costs ~4 - 5 us however little it computes, so the
gain has to come from fewer stages, which only that chain gives. The S 48 window case needs ~18 us of 706 (2.5 %); not profiled in this
part.
### Part 8 (2026-11-17): master training of one sample -- the bf16 forward on the TF32 backward; the bf16 slab backward measured

`backend_versions["cuda:b200_sm100"]`: `AtomDiTMiniWorld` 10 (`AtomDiTESMFold2` inherits it), `SWAAtomTransformer` 14, `InputFeatureEmbedder` 14.

**What serves now** (master training of window attention, one sample, <= 6144 rows; `_SWATrainB200`, `_b200_hybrid`):

- **Every weight form in one launch** (`wpack_bf16.cu swa_wpack_hybrid_sm100`, `sm100.master_pack_hybrid`): the bf16 forward forms
  (Wmod, [Wqkv; Wg] adjacent, Wo, Wu, Wd) and the TF32 backward's (W, WO, WAB, WD of `wprep_tf32`, WdT / WoT of `wprep_bwd_tf32`), the
  latter injected into the cache the backward's `_cached` calls read. Was three launches with ~2 us gaps (6.6 us of head at L 1024).
- **The bf16 forward** (the bf16 modulation / qkvg / attention / FFN stages: ~31 us at L 1024 against the TF32 forward's ~52) and its
  twelve saves widened to fp32 in one launch (`swa_save32_sm100`, exact) for the **TF32 backward** (part 6's chain, unchanged). The
  output stays bf16, also between a stack's blocks (the stack passes c in bf16 for the forward, fp32 for the backward's dc sums).
- **The modulation inside the qkvg stage** (<= 3072 rows, `sm100.block_fwd_train_c`): `mod_fwd -DLATE_TRIGGER` (triggers the next
  kernel after its own wait) beside `qkvg_fwd1t -DSAVE -DLATE_WAIT` (shift_a / scale_a from c itself, x / rn(p_q) / rn(p_k) saved, no
  wait before its loads, its store warp waits once): the chain is qkvg -> attention -> FFN. The same bits (`_tools/eqs.py`).
- From 7168 rows the TF32 forward stays: widening the saves is bandwidth-bound (8.3 us at 4096 rows). A/B in one process
  (`_tools/ab8.py`, hybrid / TF32): ESMFold2 L 3072 0.953, 4096 0.984, 5120 0.991, 6144 1.000, 7168 1.023, 8192 1.043; embedder linear
  L 4096 0.977, 6144 0.991, 8192 1.018.

ESMFold2 master train L 1024 (graph, L2 flushed): TF32 route 109.0 us -> hybrid 94.7 -> one-launch forms 91.9 -> modulation inside qkvg
87.9 (H100 111.8). Timeline at L 1024 (`_tools/tl8.py`): forms 2.9, modulation || qkvg 15, attention 5.4, FFN 6.9, widening 3.9, then the
TF32 backward (gate 10, adaLN 3.7, out-projection 9.3, dQ || dK dV 13.5, dx GEMM 5.9, adaLN 2.3, dc GEMM 7.2, dsilu 1.8).

**The bf16 64-row slab backward** (built, measured, not routed: `_B200_MT_BF16_MAX_ROWS = 0`): the one-launch forms with the backward
transposes (`master_pack_train`), pre-wait weight builds of the four TMEM-weight kernels (`-DWPRE`: `ffn_bwd_gate_wp`, `ffn_bwd_dy_stwp`,
`oproj_bwd_stwp`, `qkvg_bwd_stwp`), one-augment 64-row paths in `qkvg_bwd.cu` / `oproj_bwd.cu` (A1: the AT8 form with a per-row scale and
per-row dmod stores, in place of the generic row-at-a-time loop -- a TMEM load, a wait and two butterflies per row, 22 us of the step),
a dc kernel on 32-row slabs (Wmod^T in TMEM, dmod as two bf16 terms, the silu backward fused). ESMFold2 L 1024: 157.5 us (bf16 chain at
the round's start) -> 139.1 (WPRE) -> 137.0 (A1 paths + dc slab) against the TF32 route's 109. Every stage is 1.5 - 3 x what its work
predicts once inside the chain (`_tools/trk.py`, globaltimer stamps of CTA 0): `ffn_bwd_dy` pass 2 (32 rows of a thread: 3 shared loads,
a store, 3 dmod stores a row) 4.2 us alone, 11.5 us in the step (with its dmod stores compiled out too); the dc slab's 96 KB TMA load 5.3
us in the step (18 GB/s a CTA). Not resolved; the TF32 backward stages are the faster ones at one sample, so the route is the hybrid.

Tried and removed (patches in `scratchpad/b2/r3swa8_rejected_*`): (1) the twelve saves widened on a side stream beside the gate kernel
(gate reading y in bf16, event join before the adaLN): L 1024 91.9 -> 98.5 us (the join breaks the adaLN's PDL edge); (2) the qkvg
stage's / attention's saves widened on a side stream during the forward: L 1024 -> 98.4 (the event record after the qkvg launch breaks the
qkvg -> attention PDL edge: the attention started 4.7 us after the qkvg stage ended); (3) the dc slab kernel in the TF32 backward's tail:
19 - 21 us after the adaLN against cuBLAS TF32 + dsilu's ~9 (L 1024 102.4, L 4096 147.5 against 91.9 / 137.0); the kernel was removed
(`mod_dc_slab.cu`, one implementation per op).

Measured (`$W/b2511/work/r3-swa/p8/f_a.jsonl` L 1024 / 2048, `f_b.jsonl` 3072 - 8192, separate runs; `f_vs.txt`; ratio B200 / H100 cuda;
master training; before = part 6 / 7, `p6/fin_vs.txt`):

| class, case | cases | before | after |
|---|---|---|---|
| `AtomDiTESMFold2` one sample | 8 | 0.79 - 0.97 (L 1024 0.973, 2048 0.961) | 0.788 - 0.895 (L 1024 0.788, 2048 0.895, 3072 0.852, 4096 0.885) |
| `SWAAtomTransformer` 3 blocks, window, one sample | 8 | 0.84 - 0.97 | 0.765 - 0.885 |
| `InputFeatureEmbedder` window (3 variants) | 24 | 0.82 - 1.010 (2 above 1.0: relu L 2048 1.010, linear L 3072 1.002) | 0.817 - 0.939 (relu L 2048 0.877 / 0.884, linear L 3072 0.939) |

None of the master-training cases is slower than the H100 (was 2); none of the one-sample window cases is under 0.667 (target not met).
Ladder: 0 violations (worst cuda / triton 0.83). Accuracy (`p8/f_acc.jsonl`, L 1024 - 8192): no RED; ESMFold2 GREEN 0.37 - 1.0 except L
6144 YELLOW 1.03 (H100 0.99); embedder window cases YELLOW 1.01 - 1.99 -- relu L 1024 1.99 (triton 1.39, H100 1.49: allowed, inside the
standard), the others at or below the triton / H100 tokens within ~0.3. Training determinism (`_tools/eqh.py` / `eqs.py`, 3 eager runs,
outputs and every gradient, five cases): same bits. 2000 graph replays (`stressmt.py` / `stressmt_e.py`, `DT=master`): no hang, the same
gradient bits.

What remains above 0.667 at one sample (L 1024: 87.9 us against a 74.6 us target): the TF32 backward is ~55 us of ~10 dependent stages
at ~4 - 6 us each, its tail (dx GEMM + adaLN + dc GEMM + dsilu, ~17 us at L 1024 and ~38 us at L 4096 on cuBLAS TF32) the largest
piece; the widening of the saves (3.9 us at L 1024, 8.3 at L 4096, bandwidth); the forward's FFN stage. The H100 form (dx and the
modulation backward inside tensor-core slab stages) needs bf16 slab kernels that run in the step as fast as alone -- the measured gap above
(stages 1.5 - 3 x slower in the chain than alone) is the first thing to find.

Shared files: none outside `kernels/swa_dit/cuda/sm100` (sm_100a only) and `modules/atom_dit.py` / `modules/embedder.py` (B200 branches, versions).
New sm_100a cubins built on first use (the sm100 prebuild plan must pick them up): `swa_dit_sm100/wpack_bf16`, `qkvg_fwd1t_tr`, `mod_fwd_lt`,
`ffn_bwd_gate_wp`, `ffn_bwd_dy_stwp`, `oproj_bwd_stwp`, `qkvg_bwd_stwp` (the last four only on the unrouted bf16 route).

### Part 10 (2026-11-17, r3-swa restart): the true state, multi-lane TMA issue in the fp32 forward stages

`backend_versions["cuda:b200_sm100"]`: `AtomDiTMiniWorld` 11 (`AtomDiTESMFold2` inherits it), `SWAAtomTransformer` 16, `InputFeatureEmbedder` 16.
Records `$W/b2511/work/r3-swa/t0/` (before, base 83e4c1ddf) and `t1/` (after); each length group (1024 | 2048 - 3072 | 4096 - 6144 |
7168 - 8192 | 12288 - 16384) its own measurer run, `cuda`, `triton`, `pytorch`, `dev.sh -x` GPU 2; tables `ana.txt` (`_tools/t0/ana.py`).

**State at the start (276 requirement cases at min samples)**: 109 above 0.667 (H100 >= 20 us everywhere), 1 slower than the H100
(ESMFold2 float32 inference L 1024: 46.9 against 46.0 us), ladder 0 violations (cuda < triton < pytorch on every served case), 12 refused
(embedder master training, global attention, L 1024 - 4096: Triton 0.70 - 1.49 ms is itself 1.1 - 1.6 x the H100 cuda there).

**Found**: one thread issues a TMA load about every 600 clocks whatever its size (`_tools/t0/tmabw.py`: one CTA streaming [R rows][32 fp32]
boxes through a ring -- 6.8 / 13.5 / 27 / 54 B per clock for 4 / 8 / 16 / 32 KB boxes, the same for 1, 32 or 148 CTAs, hot or flushed L2;
eight lanes issuing 4-KB boxes at once: 36 B per clock; `prefetch.tensormap` and spinning waits change nothing). The TF32 forward stages
issued every ring load from lane 0: `ffn_fwd_tf32` streamed its 448 KB of weights at ~16 B per clock (clock trace `_tools/tr32.py`: the
hidden-chunk loop waited ~2900 clocks a chunk for its 48 KB with the MMAs removed as with them).

**Built (taken)**: the producer warps of `ffn_fwd_tf32` (activation and weight rings), `qkvg_fwd_tf32` (activation ring, cos / sin,
weight ring), `mod_fwd_tf32` (c tiles, Wmod ring) and `attn_fwd_tf32` (q / K / V) issue from several lanes (lane l: the loads l mod PL,
PL <= the ring's slot count -- with more lanes than slots a lane can wait on a slot two rounds ahead and the empty barrier's parity
aliases: PL 16 on 8 slots crashed). Same bits (`_tools/t0/dump.py` / `cmp.py`, base tree against this one: 23 cases, inference and
training, every output and gradient equal; two runs each bitwise equal). Graph timing alone, base -> new (`_tools/t0/abt.sh`, us):
ESMFold2 float32 inference L 1024 46.7 -> 44.9, L 8192 59.1 -> 57.3; SWA float32 inference 5 x 1024 (1 block) 151.2 -> 149.3, 5 x 2048
global 106.4 -> 104.3; embedder float32 inference L 2048 225.0 -> 218.8; float32 training L 1024 110.5 -> 108.9; FFN stage alone at
8192 rows (`tr32.py`, 128-row tiles) 43853 -> 36034 clocks.

**Measured, not taken** (diffs under `docs/b200/rejected/`): the same multi-lane issue in the TF32 backward stages (`ffn_bwd_gate_tf32`,
`oproj_bwd_fused_tf32`, `attn_dq_tf32`, `attn_dkv_tf32`; `swa_part10_bwd_multilane_tma.diff`): float32 / master training +1.5 - 2.5 %
(ESMFold2 float32 training L 1024 110.4 -> 112.4, master L 1024 87.9 -> 90.3 us); in the bf16 attention stages (`attn_fwd1h`, `attn_fwd3`,
`gattn_fwd`; `swa_part10_bf16_attn_multilane_tma.diff`): level (+-0.3 %); in `gattn_bwd` (4 lanes for a block's q / dO / lse / D):
level (S 48 L 7168 11409 -> 11388 us). The 32-row FFN build with a deeper activation ring (16 slots, 19 weight slots; the shift / scale
columns no longer wait ~3000 clocks behind the ring): <= 0.6 %, kept as the `-DNA32 / -DNW32` build option only. Weight loads before
`griddepcontrol.wait` in fp32 inference (`ffn_fwd_tf32` / `qkvg_fwd_tf32`, forms cached): level (0.98 - 1.01). Hidden chunks
streamed in a CTA-rotated order (no two CTAs reading the same weight lines at once): level.

**After** (`t1/ana.txt`): 107 above 0.667 (SWA float32 inference 5 x 1024 window 3 blocks 0.669 and 5 x 2048 global 0.675 now under),
1 slower than the H100 -- the same ESMFold2 float32 inference L 1024: the measurer reads 46.9 us (p10 44.8; its replays sit on the
timer's two ~2-us steps, 50 iterations) against 44.9 - 45.0 us for the same step timed alone with 30 iterations (H100 46.0); ladder 0;
12 refused (unchanged). Per class / dtype / mode: ESMFold2 float32 eval 5 / 10 (worst 1.020), float32 train 3 / 8 (0.707), master eval
8 / 10 (0.779), master train 8 / 8 (0.898); embedder float32 eval 8 / 24 (0.804), float32 train 12 / 24 (0.852), master eval 3 / 48
(0.710), master train 36 / 48 (0.949; 12 of them refused); SWA float32 eval 4 / 24 (0.729), float32 train 4 / 24 (0.708), master eval
5 / 24 (0.725), master train 11 / 24 (0.885).

**Measurement note**: many cases sit within 2 - 5 % of 0.667 and move with the measurer's order (each `cuda` step is timed right after
its PyTorch reference): SWA master training S 48 global L 7168 measures 11.4 ms alone (0.64) and 12.1 ms in the measurer (0.68),
ESMFold2 master inference L 8192 42.9 alone (0.648) and 44.8 in the measurer (0.677).

### Part 11 (2026-11-17, r3-swa): the fp32 FFN stage on CTA pairs, the embedder's training statements as kernels

`backend_versions["cuda:b200_sm100"]`: `AtomDiTMiniWorld` 12 (`AtomDiTESMFold2` inherits it), `SWAAtomTransformer` 17, `InputFeatureEmbedder` 17.
Records `$W/b2511/work/r3-swa/t2/` (`fin/` the final cuda run, one measurer run per length group, `dev.sh -x` GPU 2; triton / pytorch
from `t1/`, their paths unchanged; `fin/ana.txt`).

**The fp32 FFN stage on a CTA pair per tile** (`ffn_fwd_tf32 -DPAIR`, `tf32_fwd.FFN_PAIR`, `tile_rows`): a clock trace of the 32-row
build at 1024 rows (`_tools/tr32.py`) put the hidden-chunk loop at 14400 of 31000 clocks (8 chunks x ~1700: the a | b and ffn MMAs of a
chunk, M 128 with 32 real rows). The pair build launches 2 CTAs per tile in a cluster of 2: both make gated / att / q1 / y, rank r runs
hidden chunks 4 r .. 4 r + 3 (32 weight slots instead of 56) into a partial ffn, writes the other rank's output half of it into that
CTA's shared memory (`st.shared::cluster`, one remote mbarrier arrive per thread) and makes out for channels 64 r .. 64 r + 63 as
q1 + gate_f (p_own + p_peer) -- an fp32 add commutes, so both halves sum p0 + p1: deterministic (rel. diff to the one-CTA build 4e-7 in
inference, 1.3e-4 in training gradients). Rank 0 alone writes the q1 / att / y saves. Route: one augment whose 32-row tiles fit the SMs
as pairs (<= 2368 rows), else 64-row tiles on pairs (<= 4736 rows), else as before. Kernel alone 30500 -> 24700 clocks; in the step
(graph, L2 flushed): ESMFold2 float32 inference L 1024 44.9 -> 42.8 us (H100 46.0), L 2048 47.1 -> 44.8, L 3072 46.9 -> 44.9 (64-row
pairs; 32-row tiles alone 46.9), L 4096 49.0 -> 46.9; float32 training L 1024 108.7 -> 104.6, SWA float32 training S 1 L 1024 325 -> 311,
embedder float32 inference window L 1024 188 -> 178.

**The embedder's training statements** (`modules/embedder.py`, `embed_glue.cu`): a timeline of master training L 1024 (`_tools/tl.py`)
had ~190 of 436 us in the embedder's own torch launches around the three blocks. (1) The 3D RoPE tables in one launch also under a
gradient into the positions: no B200 block Function differentiates cos / sin (`_SWATrainB200`, `SWADiTBlockFunction` return None), so the
~25 statement launches built tables whose gradient was dropped -- the same bits (outputs and all 26 gradients), 436 -> 393 us.
(2) `_FeatB200`: features + projection + LayerNorm in one launch (fp32 arithmetic, the masked projection kept), backward one launch
(`swa_embed_featbwd_sm100`: LayerNorm backward, the positions' gradient, the features) + the weight gradient as one GEMM + one column sum
for the LayerNorm's weight / bias. (3) `_OutNormB200`: the terminal norm (LayerNorm / affine-free RMSNorm / identity) + the mask in one
launch and its backward in one; the ReLU folded into `_TokenMeanB200`'s scatter, its backward one launch (`swa_embed_tokgrad_sm100`;
was gather + divide + cast + ReLU mask). Backwards bind the device's primary context on the autograd worker thread (`sm100._ctx_current`).
Graph timing, base -> new: master window linear L 1024 436 -> 358 us, float32 L 1024 458 -> 392, float32 legacy global L 2048 726 -> 643,
master legacy window L 2048 479 -> 390, master global relu L 6144 1263 -> 1176. Deterministic (2 eager runs, every gradient). Accuracy
(`t2/acc_ife_f2.jsonl`, training L 1024 / 2048, both dtypes): no RED, no case above its B200 token (`acccmp.py`: 0 worse); master window
linear GREEN 0.93 / 0.95 (H100 1.19 / 1.33).

**Measured** (`t2/fin/ana.txt`; ratio B200 / H100 `cuda`; "above" = above 0.667; H100 >= 20 us on every case):

| class | dtype mode | cases | above 0.667 part 10 -> 11 | slower than H100 | worst part 10 -> 11 |
|---|---|---|---|---|---|
| `AtomDiTESMFold2` | float32 eval | 10 | 5 -> 5 | 1 -> 0 | 1.020 -> 0.932 |
| `AtomDiTESMFold2` | float32 train | 8 | 3 -> 0 | 0 | 0.707 -> 0.665 |
| `AtomDiTESMFold2` | master eval | 10 | 8 -> 8 | 0 | 0.779 -> 0.780 |
| `AtomDiTESMFold2` | master train | 8 | 8 -> 8 | 0 | 0.898 -> 0.886 |
| `InputFeatureEmbedder` | float32 eval | 24 | 8 -> 6 | 0 | 0.804 -> 0.769 |
| `InputFeatureEmbedder` | float32 train | 24 | 12 -> 1 | 0 | 0.852 -> 0.677 |
| `InputFeatureEmbedder` | master eval | 48 | 3 -> 3 | 0 | 0.710 -> 0.702 |
| `InputFeatureEmbedder` | master train | 48 | 36 + 12 refused -> 46 served, 0 refused | 0 | 0.949 -> 0.988 (the newly served L 2048) |
| `SWAAtomTransformer` | float32 eval | 24 | 4 -> 4 | 0 | 0.729 -> 0.730 |
| `SWAAtomTransformer` | float32 train | 24 | 4 -> 3 | 0 | 0.708 -> 0.678 |
| `SWAAtomTransformer` | master eval | 24 | 5 -> 5 | 0 | 0.725 -> 0.725 |
| `SWAAtomTransformer` | master train | 24 | 11 -> 11 | 0 | 0.885 -> 0.885 |
| total | | 276 | 107 + 12 refused -> 100, 0 refused | 1 -> 0 | |

Ladder: 0 violations (cuda < triton < pytorch on every served case, 2 % rule). **The 12 refused cases are served** (embedder master
training, global attention, L 1024 - 4096: `_B200_MT_TF32_GLOBAL_MIN_ROWS` 5120 -> 1024, `t2/fin2/`): 0.887 - 0.988 x the H100 (were 1.03 -
1.11 before this part's embedder kernels), Triton 1.26 - 1.58 x slower than them; accuracy (`t2/acc_g.jsonl`) GREEN 0.62 - 0.97 or YELLOW
1.01 - 1.22, every case below both the Triton and the H100 tokens; deterministic (2 eager runs, every gradient). Without them: 88 above 0.667. Accuracy of the pair FFN
(`t2/acc_pair.jsonl`, float32 L 1024 / 3072, the three classes): no RED; every case within 5 tol; against the B200 tokens two cases moved
(SWA S 5 global L 3072 inference 0.88 -> 1.06 -- not on the pair route, S 5 has 480 tiles; embedder window relu L 1024 inference 0.99 ->
1.02, the known case of part 3, triton 1.74, H100 1.62).

**Not taken / nothing left**: (d) the bf16 forward kernels' weight streams already issue one TMA load per lane (`qkvg_fwd1t`,
`ffn_fwd1t`, `qkvg_fwd3`, `ffn_fwd3`: 8 - 16 lanes, weights resident after one load); the bf16 backward producers that stream from lane 0
(`ffn_bwd_gate`, `dw_bf16`, `attn_dq` / `dkv`) serve S 48 master training, whose timeline (SWA S 48 global L 6144, 8.33 ms) is the
global attention (backward 3.95 ms, forward 1.88 ms at the MUFU floor: 48 x 4 x 6144^2 exponentials at 16 / clk / SM = 1.6 ms).
(c) 64-row bf16 tiles for 5120 - 8192 rows: not built -- by the part-3 tile traces a 64-row tile doubles the row threads' work
(qkvg epilogue 3400 ticks per 32 rows, FFN ~2000), which is about what the 12 - 108 second tiles cost now (qkvg +3.3, FFN +4.1 us).

**Still above 0.667, and why** (88 cases):

| group | cases | ratio | why (measured) |
|---|---|---|---|
| master training of one sample (ESMFold2 8, SWA 3 blocks 8, embedder window 16 / global 8) | 40 | 0.68 - 0.89 | bf16 forward + TF32 backward: ESMFold2 L 1024 88 us against a 74.6 us target; the backward is ~10 dependent stages of 4 - 6 us (parts 4 - 8); the embedder runs three such blocks plus its read-out GEMMs |
| bf16 inference of one sample (ESMFold2 L 1024 - 8192, SWA 5 x 1024) | 13 | 0.68 - 0.78 | three dependent stage kernels each above the ~6 us floor (L 1024 24.4 against 21.1); from 5120 rows a second 32-row tile on some CTAs |
| fp32 inference of one sample (ESMFold2 L 1024 - 5120, SWA 5 x 1024 window) | 6 | 0.67 - 0.93 | L 1024: qkvg with the modulation 16 us, attention 6, the pair FFN 14 after the attention |
| global attention (embedder fp32 / master eval, SWA S 5 eval, SWA S 48 master train, embedder fp32 train L 2048) | 26 | 0.67 - 0.77 | the attention is 45 - 70 % of the step at its per-block control / MUFU floors (swa-global parts 1 - 4) |
| fp32 training of one sample (SWA L 1024 - 3072) | 3 | 0.667 - 0.678 | within 2 % of the target: the TF32 backward chain |

### Part 12 (2026-11-18, r3-swa): qkvg_fwd1t on CTA pairs, K-split dx / dc GEMMs, the embedder's global training timeline

`backend_versions["cuda:b200_sm100"]`: `AtomDiTMiniWorld` 13 (`AtomDiTESMFold2` inherits it), `SWAAtomTransformer` 18, `InputFeatureEmbedder` 18.
Base `b200-2.5.x` b070dfb11 (engine 2.5.14 merged: no change to these classes or their H100 tokens). Records `$W/b2511/work/r3-swa/t3/`
(`before/` and `fin3/`: `cuda`, `triton`, `pytorch`, one measurer run per length group, `dev.sh -x` GPU 2; `eq/` the A/B dumps).

**qkvg_fwd1t on two CTAs per tile** (`QkvgFwd1t.bind`, `PAIR1T`; inference and the master-training forward, where 2 x tiles (+ the
modulation kernel's CTAs beside it in training) fit the SMs): rank r makes the projections p = r and r + 2 (rank 0 Q, V; rank 1 K, G):
half of W loaded (64 of 128 KB), half the MMAs, one epilogue pass per warpgroup instead of two; both make a and x (rank 0 stores the x
save). No exchange: the same bits (`t3/ab.sh`, six cases, outputs and every gradient). Graph timing alone: ESMFold2 master inference
L 1024 24.5 -> 22.4 us (it reads 22.4 / 24.4 between processes: the timer's 2-us step), embedder master inference L 1024 114.7 -> 110.4,
ESMFold2 master training L 1024 87.5 -> 86.0; in the measurer (master training only, `me/`): embedder window relu L 1024 346 -> 340 us,
legacy window 330 -> 325, window linear 358 -> 353; L 2048: level.

**dx = dP W and dc = dmod Wmod as K-split batched GEMMs** (`tf32_bwd._bwd_few`, `KSPLIT_DX` 2 / `KSPLIT_DC` 3 slices, from 2048 rows,
`_ks`: as many slices as cuBLAS's 128 x 64 tiles fit the SMs): one strided batched TF32 GEMM (no copies) writes [ks, M, C] partials,
which the adaLN backward (its `ndy` partial sums) and the dsilu kernel (`acc_sum`, new) add in slice order: deterministic, ~1e-6 relative
to the single GEMM. Timeline before (ESMFold2 master training L 1024): dx GEMM 5.0 + adaLN 2.3 + dc GEMM 7.2 + dsilu 1.8 us, the GEMMs on
16 CTAs. Measured (graph, alone): ESMFold2 float32 training L 2048 114.5 -> 112.5, L 3072 126.8 -> 125.1, master L 3072 120.6 -> 118.9,
SWA float32 3 blocks L 2048 348 -> 340 us; 1024 rows level or slower (SWA 311 -> 313), so not used there. 4 / 6 slices were slower
(ESMFold2 float32 L 1024 104.4 -> 108.6, master L 1024 86.1 -> 91.9). Accuracy (`t3/acc_ks.jsonl`, training L 2048 - 4096, both dtypes,
three classes): no RED; embedder float32 window cases YELLOW 1.74 / 2.68 (pos gradient, L 2048 / 3072; B200 tokens 1.43 / 1.94, H100
2.28 / 2.33) -- the same values with the split off (`acc_e0` / `acc_e1`): not this change.

**Rejected** (`docs/b200/rejected/swa_part12_ffn_fwd1t_pair.diff`): `ffn_fwd1t -DPAIR` (bf16 inference FFN stage on a cluster of 2: rank r
hidden units 128 r .., 96 instead of 192 KB of Wu / Wd, partial ffn exchanged over DSMEM): ESMFold2 master inference L 1024 22.4 -> 24.4,
L 2048 26.4 -> 28.5 (with qkvg single: 26.4 / 30.4) -- the FFN kernel launches under the qkvg / attention kernels (PDL) and its pairs do
not fit beside them.

**(c) Embedder global master training, timeline** (L 2048 esmfold2 linear, 710 us, H100 724, target 482; `_tools/tl.py`): per block
forward ~70 us (`gattn_fwd` 23, `qkvg_fwd_tf32` 13 - 19, `ffn_fwd_tf32` 18, modulation / weight forms / roundings ~15), backward ~120 us
(`gattn_bwd` 48.5, gate 12, adaLN 15, oproj 12.5, gtail 9, gdelta 4, the dx / dc GEMMs ~9); the embedder's own statements ~40 us. The
biggest stage is the global attention backward: 32 items (S / 256 key blocks x 4 heads) on 32 of 148 SMs at one sample of 2048 atoms;
its split mode was measured slower there by the swa-global rounds (`SPLIT_MIN_BLOCKS`: N 1 S 2048 59.4 whole items against 69.5 split). Not changed in this part: even a backward at the
SMs' full width (~15 us a block) would leave ~610 us (0.84 x the H100).

**Measured** (`t3/fin3/ana.txt` against `t3/before/ana.txt`, both all three backends; ratio B200 / H100 `cuda`; "above" = above 0.667):

| class | dtype mode | cases | above before -> after | worst before -> after |
|---|---|---|---|---|
| `AtomDiTESMFold2` | float32 eval | 10 | 5 -> 5 | 0.932 -> 0.931 |
| `AtomDiTESMFold2` | float32 train | 8 | 1 -> 0 | 0.668 -> 0.665 |
| `AtomDiTESMFold2` | master eval | 10 | 8 -> 8 | 0.781 -> 0.783 (L 1024 0.771 -> 0.705, 5120 0.728 -> 0.696, 7168 0.719 -> 0.689) |
| `AtomDiTESMFold2` | master train | 8 | 8 -> 8 | 0.885 -> 0.886 (L 1024 0.788 -> 0.770, 2048 0.827 -> 0.815, 3072 0.851 -> 0.838) |
| `InputFeatureEmbedder` | float32 eval | 24 | 6 -> 6 | 0.768 -> 0.762 |
| `InputFeatureEmbedder` | float32 train | 24 | 1 -> 1 | 0.679 -> 0.669 |
| `InputFeatureEmbedder` | master eval | 48 | 3 -> 3 | 0.692 -> 0.693 |
| `InputFeatureEmbedder` | master train | 48 | 46 -> 46 | 0.982 -> 0.974 |
| `SWAAtomTransformer` | float32 eval | 24 | 4 -> 4 | 0.730 -> 0.729 |
| `SWAAtomTransformer` | float32 train | 24 | 2 -> 1 | 0.676 -> 0.677 (L 2048 0.670 -> under) |
| `SWAAtomTransformer` | master eval | 24 | 5 -> 5 | 0.725 -> 0.725 |
| `SWAAtomTransformer` | master train | 24 | 11 -> 11 | 0.885 -> 0.890 |
| total | | 276 | 100 -> 98 | |

None slower than the H100 (0 -> 0); ladder 0 violations (cuda < triton < pytorch, 2 % rule) before and after. Measurement note: with
`cuda` alone in the measurer (`t3/fin/`, `fin_a2/`) the embedder window relu master training L 1024 case reads 373 us (0.720) with the
pair and 379 without it; with the three backends 340 / 346 -- the case's context in the run, not the change.

Still above 0.667 (98): as part 11's table; the one-sample master training chain (ESMFold2 L 1024 86.1 us against a 74.5 us target)
and the global attention backward (above) are the two largest pieces.

### Part 13 (2026-11-18, r3-swa): the embedder's LayerNorm gradient column sums; the global attention backward on more SMs (rejected)

`backend_versions["cuda:b200_sm100"]`: `InputFeatureEmbedder` 19 (`AtomDiTMiniWorld` 13 and `SWAAtomTransformer` 18 unchanged). Base
`b200-2.5.x` 05fe8a72b (engine 2.5.14). Records `$W/b2511/work/r3-swa/t4/` (`fin/`: embedder training, `cuda`, one measurer run per
length group, `dev.sh -x` GPU 2; the other classes / backends from `t3/fin3/`, their paths unchanged; `fin/ana.txt`).

**Found** (timelines `t4/tl_*.txt`, `_tools/tl.py`): every embedder training step with a LayerNorm (the `esmfold2` features' norm, the
terminal LayerNorm) spent ~25 us twice on `gb.sum(1)` -- the weight / bias gradient column sums of `_FeatB200` / `_OutNormB200`: a
memset, ~11 us of idle GPU and an 11-us `reduce_kernel` (window and global, L 1024 and 2048 alike), on the step's chain.

**Built (taken)**: `swa_embed_colsum_sm100` (`embed_glue.cu`, `embedder._colsum`): grid (R, 8), cluster (1, 8), 1024 threads -- each CTA
sums M / 8 rows (32 row groups x 32 float4 columns), the row groups in order in shared memory, rank 0 the 8 chunks in rank order over
DSMEM: a fixed order (the same bits over 3 runs, `_tools/det.py`, 0 bad). Graph timing: embedder master training window linear L 1024
354 -> 334 us, global L 2048 705 -> 658 (`fl.py`; in the step 2 x ~25 us -> 2 x ~4 us). Shared file: `modules/embedder.py` (only the B200 Functions `_FeatB200` / `_OutNormB200` call it; no other card's path changed). Accuracy (`t4/fin/acc_e.jsonl`, training L 1024 / 2048,
both dtypes, every case): no RED, none worse than its B200 token (`acccmp.py`: 0 worse).

**Measured, not taken** (`docs/b200/rejected/swa_part13_gattn_bwd_aligned_sliced_split.diff`): the global attention backward of one
sample (`gattn_bwd`, 32 items = 32 SMs at L 2048) on more CTAs. (1) Split mode on an item-aligned grid (items x CS CTAs, CS 2 - 4 pieces
of Qn / CS query blocks per key pair) with the sweep starts staggered so the dQ adders of a block never tie, and the bulk add released at
once (`EAGER`); (2) the same with sliced dQ (each key pair's dQ_b into its own fp32 slice, the last of a block's adders sums the slices in
key-pair order: no ordered chain at all, `SLICED`). Both correct, deterministic, 60 graph replays full / ragged 0 mismatches. The kernel
alone (`gbt.py`, L2 flushed, us; whole / old split / aligned / sliced): L 1024 36.7 / 71.7 / 38.7 (CS 2) / 59.1; L 2048 59.2 / 63.3 / 48.8 /
53.1; L 3072 83.7 / 67.4 / 57.2 / 69.6; L 4096 106.5 / 89.9 / 77.6 / 95.9. In the step (L2 hot, `fl.py`, us; routing as now / aligned CS 4 /
sliced): embedder master global L 1024 530 / 557 / 604, L 2048 658 / 678 / 709, L 3072 784 / 813 / 975, L 4096 876 / 881 / 1036, fp32 L 2048
635 / 656 / 687 -- slower everywhere: in the step the whole-item kernel runs 49 us at L 2048 (the attention inputs hot in L2), and a CTA's
fixed cost (K / V load, pipeline fill, the dK / dV combine through the workspace) on 4 - 8 blocks outweighs the SMs gained.

**Measured** (`t4/fin/ana.txt` against `t3/fin3/ana.txt`; ratio B200 / H100 `cuda`; "above" = above 0.667): 98 -> 97 above (embedder master
training window linear L 1024 0.677 -> 0.638), 0 slower than the H100, ladder 0 violations. Embedder master training 46 -> 45 (worst 0.974 ->
0.965; window esmfold2 L 2048 - 8192 -2 to -5 %, global esmfold2 -1 to -4 %; the legacy features have no LayerNorm and gain only the
terminal norm's sum where it is a LayerNorm).

Still above 0.667: as part 12. The embedder's remaining glue in a one-sample training step (`_tools/t4/ops.py`): the fp32 casts of
`single` (3.9 us), the read-out weight casts (1.8 + 3.6), the per-block 128 x 128 weight copies (3 x ~1.7), the gradient adds of x = c
(3 x ~1.7), the positions' casts (2.6 + 2.0) -- ~20 us; the global cases' gap is the TF32 row stages and the attention backward's
per-block chain (part 12), not the SM count.

### Part 14 (2026-11-18, r3-swa): the step's glue off the chain, the stack's dc summed in-kernel, deferred weight-gradient joins

`backend_versions["cuda:b200_sm100"]`: `AtomDiTMiniWorld` 14 (`AtomDiTESMFold2` inherits it), `SWAAtomTransformer` 19, `InputFeatureEmbedder` 20.
Base `b200-2.5.x` 80572b7d4 (engine 2.5.14). Records `$W/b2511/work/r3-swa/t5/` (`fin/`: `cuda` of the three classes, one measurer run
per length group, `dev.sh -x` GPU 2; triton / pytorch from `t3/before/`, their paths unchanged; `fin/ana.txt`; timelines `b2/` - `b7/`;
A/B probe `_tools/t5/p14.py`).

**Built (taken)** -- every item measured on the graph-replay timeline (`_tools/tl.py`) of embedder master training L 1024 window linear:
- `dw_reduce5` (dw_tf32.cu): the reduce writes dWg into a tensor of its own -- no `dwqkvg[3C:].clone()` launch per block (same bits).
- Weight forms cached per weight version (`_b200_hybrid_pack`, `_b200_side_forms`: `kernels._capture.param_forms` -- eager: once per
  optimizer update, invalidated by the parameter's `_version`; a CUDA-graph capture: once per capture, every replay recasts the weights
  as they are then, so an optimizer step inside the graph is seen). In a captured training step the forms are made on a side stream
  forked at the step's start (the stack's / the embedder's prefetch: three block packs + the read-out weight's bf16 form), off the chain.
- Positions read in place (bf16 widened in `swa_embed_rope` / `_feat` / `_featbwd`, flag bit 0; dpos written bf16, bit 2), the feature
  projection W read as the parameter itself (bit 1): no pos cast (x 2), no W^T copy, no dpos cast. Same bits.
- `_ReadoutB200`: the terminal norm + the read-out GEMM in one Function on the cached bf16 weight; backward dy = g Wr and the norm's
  backward on the current stream, the weight gradient g^T y with fp32 output (no bf16 rounding, no cast launch) on a side stream.
  `_FeatB200` backward: its weight GEMM beside the column sum.
- A master stack (`SWAAtomTransformer` TF32 route): c read in its own dtype (no fp32 copy on the hybrid route); the blocks' dc summed
  inside a new dsilu launch (`swa_mod_dsilu_acc_sm100`, `tf32_bwd.DcAcc`: fp32 running sum in the backward's order, one rounding;
  the same bits where x and c differ, the embedder (x = c) also adds the first block's dx -- 4e-3 of max relative difference in the
  outputs / gradients against the autograd adds, error vs fp64 GREEN / YELLOW as before); the non-final blocks' dc launches on a
  stream of their own (DEFER_DC); the non-final blocks' weight-gradient side streams joined by the first block's backward
  (`_b200_defer_ok`: every weight an fp32 leaf with `.grad` None and no hooks -- AccumulateGrad then steals the tensor, no kernel reads
  it before the stack's backward ends; otherwise joined as before). The next block's backward now starts after this block's dq.

Graph timing alone (us, before -> after): embedder master window linear L 1024 332 -> 297, global L 2048 658 -> 620, SWA master 3 blocks
L 1024 262 -> 235, ESMFold2 master L 1024 86 -> 86 (one block: its tail is the dc chain and the dW reduce, both ~17 us after dK / dV).
A/B (`p14.py`, switches off vs on, same process): deterministic over 3 runs, after an in-place weight update (eager) and in a captured
graph replayed after an in-place update both match the reference within 7e-3 (master) / 1e-6 (fp32); `.backward()` twice (second time
accumulating, no deferral) = 2 x one step exactly.

**Measured, not taken** (switches in the source, off): (1) `EARLY_DC` -- dc's GEMM split at the modulation column 2C, the FFN half early on
its own stream: ESMFold2 L 1024 86 -> 96 us (the early half beside the attention backward stretched dK / dV 12.5 -> 17 us; the late half,
a strided K slice, ran on cuBLAS's sm_80 64 x 64 kernel, 10 us against 6.7). (2) `DwTf32.DIRECT_STORE` (the dW partials by the 128
epilogue threads' coalesced stores instead of one thread's TMA stores): level (problem 0 ~10 us after the attention either way).
(3) `DwTf32.BOX3` (taken, level): a ring stage's operands as two 3-D-map boxes instead of up to 20 2-KB boxes (the Transition agent's box-
count finding) -- the same bits, dW problem 0 still ~9.6 us after dK / dV: that kernel is bound by its 256-KB partial per CTA (16 CTAs at
1024 rows), not by its box count. The H100 sums these partials inside an 8-CTA cluster over DSMEM (no workspace, no reduce launch): the
B200 counterpart is the next item for the one-block tail.

**Measured** (`t5/fin/ana.txt` against `t4/fin/ana.txt`; ratio B200 / H100 `cuda`): 97 -> 94 above 0.667, 0 slower than the H100, ladder 0
violations. Embedder master training 45 -> 41 (worst 0.965 -> 0.921; window L 1024 / 2048 now under, L 3072 - 8192 0.67 - 0.72, global
0.72 - 0.92), SWA master training 11 -> 11 (worst 0.890 -> 0.830; S 1 L 1024 0.754 -> 0.682), ESMFold2 master training 8 (0.886, unchanged),
SWA float32 training 1 -> 2 (L 4096 0.670: 425.9 against 424 us, the timer step). Accuracy (`t5/fin/acc.jsonl`, the three classes'
training at L 1024 / 2048, both dtypes): no RED; every case within 5 tol.

### Part 15 (2026-11-18, r3-swa): the weight gradients summed inside a cluster over DSMEM; the TF32 stack's forms and modulation ahead

`backend_versions["cuda:b200_sm100"]`: `AtomDiTMiniWorld` 15 (`AtomDiTESMFold2` inherits it), `SWAAtomTransformer` 20, `InputFeatureEmbedder` 21.
Base `b200-2.5.x` ac060f863 (engine 2.5.14). Records `$W/b2511/work/r3-swa/t6/` (probes `_tools/t6/dwg.py`, timelines `tl*_*.txt`,
`fin/`: the three classes' `cuda`, one measurer run per length group, `dev.sh -x` GPU 2; triton / pytorch from `t3/before/`).

**dW in one cluster per output tile, DSMEM ordered sum** (`dw_tf32.cu -DDW_CL`, `DwTf32._bind_cluster`; the H100's form): the 10 output
tiles of 128 rows (dWqkvg m-blocks 0 - 3, dWo, dWu m-blocks 0 - 3, dWd [128 x 256]) each get a cluster of CL CTAs, rank k the k-th row
slice of the reduction (A operand: one m-block's 4 column atoms per 3-D TMA box, 32 rows a ring stage). After the MMAs every CTA stages
its accumulator in its ring as CL column blocks and sends block o to rank o with one `cp.async.bulk` shared::cta -> shared::cluster
(completing on rank o's mbarrier); rank o adds the CL slots in rank order and writes the final dW (dWd: two rounds of 128 columns). No
workspace, no partial stores to global, no reduce launch; deterministic (fixed order; 3 runs equal), error vs fp64 3e-7 - 1.4e-6 of the
max (split-K form 3e-7 - 5e-7). CL = 4 up to 2048 rows, 8 above, up to 16384 rows (the few-tile backward); more rows keep the split-K
form. Graph replay, kernels alone (`dwg.py`, us; split-K + reduce -> cluster 4 / 8): problem 0 + reduce at 1024 rows 12.5 -> 9.2 / 8.5,
2048 13.9 -> 11.3 / 9.6, 4096 15.4 -> 15.2 / 11.6. First form measured and replaced: the accumulator rows sent by the epilogue threads'
own `st.shared::cluster.v4` stores -- ~10 us for a CTA's 64 KB (an epilogue-only build: 15.6 us against 10.9 with bulk copies); DSMEM
wants bulk transfers. 16-CTA clusters: slower (the six-cluster launch waits for whole GPCs: 21.8 against 14.1 us at 1024 rows).
In the step (graph, `tl.py`, us, split-K -> cluster): ESMFold2 master L 1024 85.9 -> 85.8 (its tail is the dc chain: dx GEMM 5.2 ->
adaLN 2.0 -> dc GEMM 6.9 -> dsilu 2.0 after dK / dV, ending after the dW), L 2048 102.2 -> 96.2 (the reduce no longer slows the dx GEMM
beside it: 13.2 -> 5.5 us), L 4096 139.0 -> 132.9, L 8192 184.3 -> 176.2; SWA master 3 blocks L 1024 236.3 -> 231.2, L 8192 540.5 ->
520.9; embedder master window L 1024 298.0 -> 292.8, L 4096 462.2 -> 457.7, global L 2048 620.3 -> 610.7.

**The TF32 forward route's forms and modulation ahead of the blocks** (`atom_dit._b200_prefetch_tf32`, `_B200_TF32_PREFETCH`; master-training
stacks not on the hybrid route -- global attention, the embedder's global variants): the timeline of embedder global L 2048 showed per
block boundary `round_tf32` (Wmod) -> `mod_fwd_tf32` -> `qkvg_fwd_tf32` on the chain (`wprep_tf32` beside), ~7 us lost each, and the
backward's `wprep_bwd_tf32` (~2.5 us) at the head of every block backward. Every block is conditioned on the same c, so the stack now
makes all blocks' forward forms and modulations on a side stream forked at its start (block i's event waited before block i), then the
backward's forms (waited once after the blocks); c is read in its own dtype as on the hybrid route (no fp32 copy). Graph timing (us,
off -> on): embedder master global L 1024 495.5 -> 477.3, L 2048 611.3 -> 597.9, L 4096 818.1 -> 804.8. Not taken: the `O16.float()`
copy after the global attention (3.5 us a block on the chain) -- removing it needs bf16 O in `ffn_fwd_tf32` and `oproj_bwd_fused_tf32`.

**Measured** (`t6/fin2/ana.txt` against `t5/fin/ana.txt`; ratio B200 / H100 `cuda`; above = above 0.667): 94 -> 81 above, 0 slower than the
H100, ladder 0 violations. `AtomDiTESMFold2` master train 8 -> 8 (worst 0.886 -> 0.871; L 2048 0.800 -> 0.766, L 8192 0.791 -> 0.752),
`InputFeatureEmbedder` master train 41 -> 31 (worst 0.921 -> 0.885), float32 train 1 -> 0, `SWAAtomTransformer` master train 11 -> 10
(worst 0.830 -> 0.778), float32 train 2 -> 1 (0.668). Accuracy (`t6/fin2/acc.jsonl`, the three classes' training at L 1024 / 2048, both
dtypes): no RED, every case within 5 tol; the one case `acccmp.py` lists as worse than its B200 token is listed the same in `t5/fin`
(not this part). Determinism (`_tools/det.py`): the same bits over 3 runs, 0 bad.

Still above 0.667: ESMFold2 one-sample master training (L 1024 85.8 us against 74.5) ends on the dc chain after dK / dV (dx GEMM ->
adaLN -> dc GEMM -> dsilu, 16.7 us: four dependent launches on 16 CTAs); the next step there is that chain as one tcgen05 kernel (K split
over a cluster, the partials summed by bulk DSMEM copies as above -- the per-thread DSMEM stores may be what made part 5's fused GEMM +
adaLN kernel lose). bf16 one-sample inference L 1024 (22.3 us, target 21.1): `qkvg_fwd1t` 7.8, the attention ends 5.0 and the FFN 6.0 us
later (graph timeline `t6/tl7_ei1024.txt`); not changed in this part. Embedder global master training: `gattn_bwd` 49 us a block at
L 2048 (32 items on 32 SMs) remains the largest stage (part 13).

### Part 16 (2026-11-18, r3-swa): the few-tile tail as one cluster kernel (built, measured, off); the global block's glue

`backend_versions["cuda:b200_sm100"]`: `SWAAtomTransformer` 21, `InputFeatureEmbedder` 22 (`AtomDiTMiniWorld` 15 unchanged: its path is
the same). Base `b200-2.5.x` e531831c6 (engine 2.5.14). Records `$W/b2511/work/r3-swa/t7/` (probes `_tools/t7/dcx.py`, `mb.py`, `ab.py`;
timelines `tl1_*` - `tl3_*`; `fin/`: the three classes' `cuda`, one measurer run per length group, `dev.sh -x` GPU 2).

**The tail as one kernel** (`dcx_tf32.cu`, `tf32_bwd.DcxTf32`, switch `DCX`, OFF): dx = dP W -> the qkvg adaLN backward (dq, dmod
columns 0 .. 2C) -> dc = (dmod Wmod) silu'(c) [+ prev] [+ dq] in one launch, one 8-CTA cluster per 128-row tile: rank k the K slices k
of both GEMMs (dx K 64, dc K 64 of the columns 2C .. 6C before the adaLN, K 32 of the adaLN's own columns after it), rows 16 k ..
16 k + 15 of the row stages; the dx and dc partials summed per row block in rank order (deterministic), the adaLN's columns all-gathered
into each rank's operand tile. Correct and deterministic on every one-sample training case of the three classes (`dcx.py`: against the
four-launch chain 5e-6 - 1e-3 max relative, the same bits over 3 runs; DcAcc stacks and the embedder's x = c fold included). Kernel alone
(`mb.py`, graph replay, 1024 rows, us; old chain = cuBLAS dx + adaln_bwd1 + cuBLAS g Wmod + dsilu, 13.4): bulk-copy exchanges 14.1,
`st.async` per row thread 14.2, staged + `st.async` 512-B rows by 4 warps 12.5, by all 8 warps rotated by rank (`-DSTA=3`) 10.7. The
trace (`-DTRACE=1`, all ranks of tile 0) shows where it goes: each of the two partial exchanges takes ~4000 clocks whatever the form --
bulk copies drain one 8-KB copy per ~470 clocks per SM (the ranks' blocks land in issue order), scattered 16-B `st.async` ~10 B/clk,
coalesced 512-B `st.async` rows ~16 B/clk per SM: an 8-CTA cluster moves ~112 B/clk over DSMEM in total (448 KB a partial exchange).
The 16-KB all-gather in coalesced 128-B `st.async` segments took ~400 clocks. In the step it does not pay: ESMFold2 master L 1024 85.8
-> 85.8 / 87.8 us (dWmod in one / two parts), L 2048 96.2 -> 110.4 (16 tiles x 8 CTAs beside the dW cluster kernel: the clusters wait
for whole GPCs), SWA 3 blocks L 1024 231.2 -> 239.4 (in a stack the dc phase is on the dq chain; the four-launch form puts it on a side
stream), embedder global L 2048 594.9 -> 628.9. The tail after dK / dV is then max(dcx 11.3 us, dW cluster 10.7) + the late dWmod rows
(6.1 us, cuBLAS: they need the adaLN's columns) -- against 16.8 for the four launches. What would make it pay: the late dWmod rows inside
the kernel (per-tile partial over the gathered columns, a fixed-order cross-tile sum) and a cheaper exchange than DSMEM's ~112 B/clk per
cluster. `DWM_SPLIT` (dWmod's rows 2C .. 6C early on a stream of their own) only with `DCX`: cuBLAS ran the strided transposed slice on its
sm_80 64 x 64 kernel (~9 us).

**Global attention, training forward** (`tf32_fwd.F32IO_TRAIN`, taken): the F32IO attention build (fp32 q heads rounded to bf16 in the
kernel, O written fp32 with the bf16 values) now also serves training where the native backward does (lse saved as its negated log2),
up to F32IO_MAX_ROWS rows: no q cast, no `O16.float()` copy (3.5 us a block at L 2048). The same bits (`ab.py`: outputs and every
gradient equal, embedder master L 2048 / float32 L 1024); embedder global master L 2048 632.7 -> 628.9 us (graph, `tl.py`).

**Global attention, backward glue** (`tf32_fwd.GLUE8`, taken): `gtail` (the head-wise tail of q / k, the v columns) and `gdelta` (dO16
and D) ran one thread per (row, head) -- 8192 threads at one sample of 2048 atoms, 32 blocks on 32 SMs: 8.7 / 4.2 us a block for ~6 /
~2 MB. Now eight threads per (row, head) (lane k: channels 4 k .. 4 k + 3, the rotary partner by `shfl_xor 4`, the head sums over the
8-lane group): 2.9 / 2.2 us. Embedder global master L 2048 594.9 -> 572.3 us, L 4096 796.7 -> 777.3 (graph, `tl3_*`). The head sums'
order changed (fixed: the same bits over 3 runs); against the one-thread form 6e-5 - 1.4e-4 max relative.

**Roofline of the global block** (embedder global master training, L 2048, one sample, per block; exposed us from the graph timeline
`tl2_g2048_f1.txt`; bytes = what the stage must read + write; FLOPs dense; achieved = bytes or FLOPs / exposed time):

| stage | exposed us | must move | FLOPs | achieved | bound |
|---|---|---|---|---|---|
| `qkvg_fwd_tf32` | 12.6 | ~8 MB (q in; Qh Kh Vh G X PQ PK saves) | 0.27 G TF32 | 0.6 TB/s | 16 tiles x NG 4 items, weight stream per CTA |
| `gcast` (fwd, K V) | 2.3 | 2 MB in, 1 MB out | - | 1.3 TB/s | launch + one wave |
| `gattn_fwd` (F32IO) | 28.4 | ~4 MB | 2.1 G bf16 | 75 TF/s | 32 items on 32 SMs (latency) |
| `ffn_fwd_tf32` (after the attention) | 16.4 | ~9 MB (g o q in; out q1 att y ffn) | 0.46 G TF32 | 0.55 TB/s | weight stream per CTA (448 KB), 64-CTA pairs |
| `gcast` (bwd, Q K V) | 4.1 | 3 MB in, 1.5 MB out | - | 1.1 TB/s | |
| `gdelta` -> `gdelta8` | 4.2 -> 2.2 | 2 MB in, 0.5 MB out | - | 0.6 -> 1.1 TB/s | was 32 SMs |
| `gattn_bwd` | 49 | ~6 MB | 5.3 G bf16 | 108 TF/s | 32 items on 32 SMs (part 13: splits lost in the step) |
| `gtail` -> `gtail8` | 8.7 -> 2.9 | ~6 MB | - | 0.7 -> 2.1 TB/s | was 32 SMs |

Every stage is latency-bound at one sample (HBM ~8 TB/s, dense bf16 ~2.2 PF/s); the two glue kernels were the cheap gaps, the attention
kernels' item counts the large ones.

**Measured** (`t7/fin/ana.txt` against `t6/fin2/ana.txt`; ratio B200 / H100 `cuda`; above = above 0.667): 81 -> 80 above, 0 slower
than the H100, ladder 0 violations (cuda < triton < pytorch, 2 %). `InputFeatureEmbedder` master train 31 -> 30 (worst 0.885 -> 0.848;
the global cases L 1024 - 7168 -3 to -6 %: esmfold2 linear L 2048 623.4 -> 598.8 us, legacy relu L 4096 831.3 -> 796.6, L 2048 619.3 ->
591.6; L 8192 level), every other class / dtype / mode the same count (ESMFold2 master train 8, worst 0.871: the four-launch tail stays).
Accuracy (`t7/fin/acc.jsonl`, the three classes' training at L 1024 / 2048, both dtypes): no RED, every case within 5 tol (the global
fp32 cases YELLOW 2.1 - 3.5 as before; `acccmp.py` lists one case worse than its B200 token: SWA master global L 1024 1.13 against 0.96,
the glue's head-sum order). Determinism (`_tools/det.py`): the same bits over 3 runs, 0 bad.

**bf16 one-sample inference** (L 1024, 22.3 us against 21.1): not changed. The FFN stage already launches under the attention (PDL
trigger at the attention's start) and streams its weights into TMEM / shared memory before its wait (`ffn_fwd1t.cu`); what follows the
wait (q / g / o loads, gated, att, q1 / y, a | b, h, ffn, out: 6.0 us) depends on the attention's O, and q alone (final before the wait)
is 8 of the 24 KB loaded after it.

### Part 17 (2026-11-18, r3-swa): the global attention forward on a cluster split

`backend_versions["cuda:b200_sm100"]`: `SWAAtomTransformer` 22, `InputFeatureEmbedder` 23 (`AtomDiTMiniWorld` 15 unchanged: window only).
Base `b200-2.5.x` 9ce271490 (engine 2.5.14). Records `$W/b2511/work/r3-swa/t8/` (probe `_tools/t8/gcs.py`; `cs1/`: the global
classes' `cuda`, one measurer run per length group 1024 / 2048 - 3072 / 4096 - 6144, `dev.sh -x` GPU 2, other cases from `t7/fin`;
`acc.jsonl`; timelines `tl_g2048_c0/1.txt`).

**Built (taken)** (`gattn_fwd.cu -DCSPLIT=1`, cubins `gattn_fwd_c` / `gattn_fwd32_c`, `GattnFwd.cluster_split`): flash-decoding's key
split inside one cluster. One item (sample, head, pair of 128-query tiles) per cluster of CS CTAs (the launch attribute; the kernel reads
`%cluster_nctarank`), rank k the k-th contiguous share of the item's key blocks (at least one block a rank; blocks past seqused masked).
No running maximum (the contract of the kernel), so the partials (unnormalized O, row sum l) add directly. After the item each softmax
thread sends its row's 32 fp32 + l to the row's owner (rank r / (128 / CS)) by `st.async` into a chunk-major receive area
([src][w][chunk][row] x 16 B: the lanes of a warp are consecutive rows, so one instruction writes one contiguous 512-B run), completing
on the owner's mbarrier (armed with the fixed byte count before the start-up cluster barrier); the owner sums the CS slots in rank order
(deterministic), normalizes and stores O (plain stores; fp32 or bf16) and LSE. Routed (`CSPLIT_SIZES` 8 / 4 / 2, `CSPLIT_MIN_BLOCKS`
4) where items x CS CTAs fit the SMs at once: one sample L 1024 (CS 4), 2048 (4), 3072 (2), 4096 (2), two samples L 2048 (2); everything
else as before (whole items / the old split). Not a persistent loop: past one wave (L >= 5120, five samples) the rounds do not shrink.

Kernel alone (`gcs.py`, graph timing, L2 flushed, ~6 us timer floor included; us; whole / old split / cluster):

| N x S | bf16 | F32IO (fp32 q, O fp32) |
|---|---|---|
| 1 x 1024 | 20.3 / 38.7 / 14.1 (CS 4; CS 8 20.3) | 24.4 / 63.4 / 16.2 |
| 1 x 2048 | 32.5 / 34.6 / 18.2 (CS 4) | 36.7 / 36.8 / 20.3 |
| 1 x 3072 | 42.8 / 36.8 / 30.5 (CS 2) | 49.0 / 40.8 / 32.5 |
| 1 x 4096 | 55.1 / 45.1 / 36.8 (CS 2) | 61.4 / 47.1 / 38.9 |
| 2 x 2048 | 32.6 / 36.7 / 24.4 (CS 2) | 36.7 / 38.7 / 26.4 |

Checks: O / LSE against the whole-item kernel 2e-3 - 4e-3 max relative of O (one bf16 step; the sum order), LSE 1e-6, full and ragged
seqused, CS 2 / 4 / 8, the same bits over 3 runs (`gcs.py`, 0 bad); `compute-sanitizer` memcheck clean; `_tools/t8/det8.py` (embedder
global training / inference, L 1024 - 4096) the same bits over 3 runs. In the step (embedder global master L 2048 linear, graph timeline):
`gattn_fwd` 28.4 -> 13.4 us a block, the step 572.4 -> 530.4 us. Found on the way: a cluster of 3 does not divide the 128 rows of a tile
(the owner index ran past the cluster): CS must divide 128.

**Measured** (`t8/cs1/ana.txt` against `t7/fin/ana.txt`; ratio B200 / H100 `cuda`; above = above 0.667): 80 -> 76 above, 0 slower than the
H100, ladder 0 violations (cuda < triton < pytorch, 2 %). `InputFeatureEmbedder` float32 eval 6 -> 2 (worst 0.768 -> 0.715), master
train 30 -> 31 (the global L 1024 - 4096 cases -4 to -9 %: esmfold2 linear L 2048 598.8 -> 546.7 us, 0.828 -> 0.756, L 1024 0.733 ->
0.690; the one added is window legacy L 3072 at 0.668, a path this part does not touch), master eval 3 -> 3; `SWAAtomTransformer`
float32 train 1 -> 0 (window, the timer step). Accuracy (`t8/acc.jsonl`, SWA + embedder, both dtypes, eval and train, L 1024 - 4096):
no RED, every case within 5 tol (`acccmp.py` lists 3 cases above their B200 token, all YELLOW <= 3.07 and below 5).

**Not built in this part** (measured reasons): the dc chain as one kernel with the late dWmod rows inside (part 16 put its best at
~6 us of the ESMFold2 one-sample step 85.8 against a 74.5 target -- it cannot reach the target alone); cta_group::2 for the TF32 qkvg
/ ffn stages (the fp32 FFN cluster build of the audit did not move float32 inference; part 11 pairs are in). The step after this part
(embedder global master L 2048, per block): forward qkvg 12.5 + K / V cast 2.4 + attention 13.4 + FFN 17 exposed; backward ~100 us of
which `gattn_bwd` 49 (32 items on 32 SMs) remains the largest single stage -- the next lever: the backward split the same way (key pairs
x query shares in a cluster, dK / dV reduce-scattered over DSMEM, the dQ chain's adders per query share; part 13's workspace version
lost in the step).

### Part 18 (2026-11-18, r3-swa): the global attention backward on a cluster split; the forward's F32IO exponentials and split gain

`backend_versions["cuda:b200_sm100"]`: `SWAAtomTransformer` 23, `InputFeatureEmbedder` 24 (`AtomDiTMiniWorld` 15 unchanged: window
only). Base `b200-2.5.x` f2d941e47 (engine 2.5.14). Records `$W/b2511/work/r3-swa/t9/` (probes `_tools/t9/gbc.py` (backward
correctness / determinism / timing), `gbv.py` (variant builds), `gtr.py` (TRACE timelines), `gfv.py` / `gfm.py` (forward variants /
whole vs split), `xab.py` (FFN pair A/B); `cs1/`, `f1/`: the two global classes' `cuda`, one measurer run per length group 1024 /
2048 - 3072 / 4096 - 6144 / 7168 - 8192, `dev.sh -x` GPU 2, other cases from `t8/cs1`; `f1/acc.jsonl`; timelines `tl_*`).

**Built (taken): `gattn_bwd.cu -DCSB=1`** (cubins `gattn_bwd_c`, per-size `gattn_bwd_c2` / `_c3` / `_c4` / `_c8` with `-DCSN`;
`GattnBwd.cluster_split`): one item (sample, head, key pair of 256 keys) per CS CTAs of a cluster (optionally two items a cluster of
2 CS), rank k the k-th contiguous share of the item's query blocks, swept from its own stagger `kp qs / Pn` inside the share. dK | dV:
after the last block every row thread sends its key row's fp32 partial (64 floats) by `st.async` to the row's owner rank (rank o owns
keys ceil(256 o / CS) ..; receive area [src][chunk 0..15][RS rows] x 16 B, a warp's rows one contiguous run), completing on the owner's
mbarrier armed with its byte count before the start-up cluster barrier; the owner's 256 threads sum the CS slots in rank order (row
fastest, conflict-free) and store bf16 by plain stores (no X staging: the area fits beside QR = 1). dQ: the counter-ordered bulk adds
as before, the adders of a block being rank k of each key pair's cluster, ordered by (step in the share, key pair) (ties when Pn > qs
broken by key pair). The exchange runs after the last `dq_out` (no rank waits on peers before its dQ adds are staged: no cycle with
the dQ chain). Routed where every cluster is co-resident (`cuOccupancyMaxActiveClusters`: 74 of 2, 45 of 3, 33 of 4, 22 of 6, 15 of 8
on 148 SMs): one sample L 1024 / 2048 (CS 4), 3072 / 4096 (CS 2: 48 clusters of 3 / 24 of 6 do not fit), two samples L 2048 (CS 2).
Checks: dQ / dK / dV against the whole-item kernel 1e-5 - 5e-3 max relative (one bf16 step; the sum orders), full and ragged
seqused (incl. a share without query blocks, L 2304 CS 3), CS 2 / 3 / 4 x one / two items a cluster, the same bits over 3 runs
(`gbc.py`, 0 bad); embedder determinism (`det8.py g`) the same bits.

Kernel alone (graph, L2 flushed, us; whole / old split / cluster): 1 x 1024 36.7 / 71.5 / 22.4 (CS 4), 1 x 2048 59.3 / 69.4 / 30.5
(CS 4; CS 3 34.6, CS 2 40.7), 1 x 3072 83.8 / 67.4 / 55.1 (CS 2), 1 x 4096 106.5 / 92.0 / 67.6 (CS 2), 2 x 2048 61.4 / 73.5 / 42.8.
In the step (embedder global master, esmfold2 linear, graph timeline, `gattn_bwd` per block / step us, off -> on): L 1024 29.7 ->
17.8 / 427.9 -> 391.1, L 2048 49.0 -> 24.4 / 531.5 -> 456.6, L 3072 67.7 -> 44.9 / 651.3 -> 583.6, L 4096 75.8 -> 56.5 / 751.0 -> 688.0.

Where a block goes now (TRACE, CTA 0, L 2048 CS 4, warm): ~1.5 us a block (rows: S^T / dP^T read + math 0.6 - 1.0, P / dS stores
0.15 - 0.25, dQ_b out 0.3 every other block), the MMA warp ~1.0 us a block (40 tcgen05.mma of K 16 + commits; no-math probe: rows
wait 0.65 us on p_free), the epilogue 3.0 us (last MMAs 0.6, send 0.4, peers 1.1, sum + stores 0.9). Probes (CSB L 2048): no counter
waits / no dQ traffic / no row math / no MMAs each ~28.4 against 30.5 us alone -- no single limit; the multi-lane TMA producer (`MLI`)
level (59.3 / 59.2 us whole, 30.6 / 30.5 CSB: not issue-bound, left off); `PCNT` 0 / 2 / 3 (exponentials on FMA) level / +6 %.

**Forward (taken)**: (a) F32IO builds take 1 of 8 exponential pairs on the FMA pipe (`FPCNT` / `FPMOD`, the backward's degree-3
polynomial; clamped at -126, masked entries selected to 0): S5 L 8192 522 -> 459 us alone, N 1 L 8192 114.7 -> 102.4, L 1024 - 6144
level; O within one bf16 step of the MUFU form, LSE 3e-5. bf16 builds keep MUFU only (1 of 4: level to +3 %, 2 of 4: +20 %).
(b) `SPLIT_GAIN_FWD` 1.2 (was the shared 1.3): N 5 S 6144 (busiest / even share 1.235) now runs the split build -- whole 280.4 /
split 245.5 us bf16, 321.5 / 253.7 F32IO; no other requirement case flips.

**Measured, not taken**: (1) `P16` (fp16 P, V converted) and `EXH` (P = `ex2.approx.f16x2` of the f16-rounded scores, P16 only) on
the forward's large shapes (us, base / P16 / EXH): S5 L 6144 243.5 / 245.8 / 302.8, S5 L 8192 455.0 / 464.9 / 563.0, S48 L 1024 81.6 /
83.1 / 96.3, S48 L 2048 261.9 / 266.2 / 320.3, N1 L 8192 102.1 / 102.4 / 120.8 -- these shapes are not exponential-bound (no-exp probe
S5 L 6144 278.5 -> 245.8 whole, 12 %); switches left in the source, off. (2) `ffn_fwd_tf32 -DXSTA=1` (the PAIR exchange by 128-B
`st.async` into a chunk-major tile, completing by bytes, instead of per-thread `st.shared::cluster.v4` + 256 remote arrives): the same
bits, ESMFold2 float32 eval L 1024 - 5120 42.8 / 42.9, 44.8 / 42.8, 44.8 / 44.6, 46.9 / 45.0, 53.0 / 53.1 us (old / new) -- level, off.
(3) A persistent cluster split past one wave (S5 L 1024: 80 forward items, 16 key blocks): CS 2 needs 160 CTAs; persistent clusters
quantize to whole items (80 / 74 clusters: 2 rounds of 8 blocks = the whole kernel's 16), CS 4 / 8 at 3 / 6 rounds would save ~4 of 16
blocks minus 3 - 6 exchanges -- not built.

**Measured** (`t9/f1/ana.txt` against `t8/cs1/ana.txt`; ratio B200 / H100 `cuda`; above = above 0.667): 76 -> 68 above, 0 slower than
the H100, ladder 0 violations (cuda < triton < pytorch, 2 %).

| class | dtype mode | above before -> after | worst before -> after |
|---|---|---|---|
| `AtomDiTESMFold2` | float32 eval / train | 5 / 0 -> 5 / 0 | 0.931 / 0.665 (unchanged path) |
| `AtomDiTESMFold2` | master eval / train | 8 / 8 -> 8 / 8 | 0.779 / 0.871 (unchanged path) |
| `InputFeatureEmbedder` | float32 eval / train | 2 / 0 -> 1 / 0 | 0.715 -> 0.703 / 0.608 |
| `InputFeatureEmbedder` | master eval | 3 -> 3 | 0.690 -> 0.687 |
| `InputFeatureEmbedder` | master train | 31 -> 26 | 0.825 -> 0.737 (global L 1024 / 2048 under; L 3072 0.70 - 0.73, 4096 0.72 - 0.74, 5120 - 8192 0.68 - 0.72) |
| `SWAAtomTransformer` | float32 eval / train | 4 / 0 -> 3 / 0 | 0.729 -> 0.723 / 0.665 |
| `SWAAtomTransformer` | master eval / train | 5 / 10 -> 4 / 10 | 0.715 -> 0.718 / 0.778 (S5 L 6144 under) |

Accuracy (`t9/f1/acc.jsonl`, SWA + embedder, both dtypes, eval and train, L 1024 / 4096 / 6144): no RED, every case within 5 tol;
`acccmp.py` lists 3 cases above their B200 token (SWA S48 global master train L 1024 / 4096 1.13 / 1.09, embedder window fp32 eval
L 1024 1.10; all YELLOW, the first listed the same in part 16). Determinism: the same bits over 3 runs.

Still above 0.667: embedder global training L 3072 - 8192 (`gattn_bwd` at CS 2 on 96 - 128 SMs, past one wave at L >= 5120 the old
split; per block the rows' math and the MMA warp's 40 issues are both ~1 - 1.5 us); the one-sample ESMFold2 master training tail
(L 1024 85.8 us against 74.5: after dK / dV dx GEMM 5.2 -> adaLN 2.1 -> dc GEMM 6.5 -> dsilu 2.7, dWmod's late rows beside dc; the
backward front ffn_bwd_gate 12 -> adaLN 3.5 -> oproj 9.4 -> attention 13); bf16 one-sample inference (`qkvg_fwd1t` epilogue); window
master training of one sample (0.70 - 0.78). Not built in this part: the `qkvg_fwd1t` epilogue rewrite and the DCX tail with in-kernel
late dWmod rows.

### Part 19 (2026-11-18, r3-swa): the saves widened under the gate kernel; the c gradient's GEMM split at column 2C

`backend_versions["cuda:b200_sm100"]`: `AtomDiTMiniWorld` 16 (`AtomDiTESMFold2` inherits it), `SWAAtomTransformer` 24, `InputFeatureEmbedder` 25.
Base `b200-2.5.x` e30107324 (engine 2.5.14). Records `$W/b2511/work/r3-swa/t10/` (probes `_tools/t10/abx.py` (A/B with input gradients),
`dcgp.py` / `dcgp2.py` (dcg alone), timelines `tl1_*` - `tl5_*`; `f1/`: the three classes' training `cuda`, one measurer run per length group
1024 / 2048 - 3072 / 4096 - 6144 / 7168 - 8192, `dev.sh -x` GPU 2, other cases from `t9/f1`; `f1/acc.jsonl`).

**Taken: the hybrid route's saves widened in the backward** (`sm100.save32_late`, `wpack_bf16.cu swa_save32_lw_sm100`, `SAVE_LATE`):
the bf16 forward kept its twelve saves in bf16; the backward's first kernel (`ffn_bwd_gate_tf32`, DYF) reads y in bf16 in place (its
YB16 path of part 8) and, with io16 bit 2, triggers its dependents only after its own `griddepcontrol.wait`; the next launch widens the
saves (exact, grid-stride, 3 blocks a SM so it stays resident beside the gate kernel, 4 loads in flight a thread), triggers at its start and
waits for the gate kernel only at its END -- the FFN adaLN after it still runs after the gate kernel, the widening (3.8 us at 1024 rows,
8.5 at 4096) runs under it. No event join (part 8's side-stream form broke the adaLN's PDL edge: 91.9 -> 98.5). Same bits (`abx.py`:
outputs, parameter and input gradients equal, 3 runs equal). ESMFold2 master training, graph median us (off -> on): L 1024 85.8 -> 83.9,
2048 96.2 -> 94.4, 3072 112.6 -> 108.6, 6144 163.8 -> 161.9, but L 4096 132.9 -> 137.0 (the gate kernel takes 128 SMs there and 38 MB
of widening runs on the rest: 20 us): routed where the gate kernel has at most 96 CTAs or the block at most 1024 rows (`save_late_ok`; L 2048, 128 gate CTAs: 94.4 / 98.2 in two runs against 96.2 -- left off).
With 6 blocks a SM the adaLN's CTAs found no room to launch early (L 1024 85.8, no gain).

**Taken (1024-row few-tile backwards): the c gradient's GEMM split at column 2C** (`dcg_tf32.cu`, `tf32_bwd.DcgTf32`, `DCG`): early E =
dmod[:, 2C:] Wmod[2C:] (the FFN / gate_a columns, final after the FFN adaLN) on a stream of its own forked after the out-projection,
one CTA a 128-row tile; the current stream waits for it where it already joins the dK / dV stream (in front of the cuBLAS dx GEMM, no
PDL edge lost); after the qkvg adaLN one launch does dmod[:, :2C] Wmod[:2C] (K 256, 4 column splits of 32) with E, silu', the stack's
running sum (`DcAcc.late`, the DEFER_DC stream as `step`) and the embedder's x fold in its epilogue -- in place of cuBLAS g Wmod (K 768) +
the dsilu launch on the chain. tcgen05 kind::tf32 on the fp32 operands (TF32 truncation; cuBLAS rounds): dc 7.4e-4 of the max against fp64
(cuBLAS + dsilu 3.5e-4); ESMFold2 float32 training L 1024 grad:cond YELLOW 1.16 (B200 token 1.06, H100 3.73, triton 3.64). Kernel alone
(back to back, us): early 5.5 - 6.0, late 5.7 (input loads before the accumulator wait; was 8.6 with them after it), cuBLAS K 768 + dsilu
7.0. Routed only where its CTAs fit beside the attention backward and the early weight gradients (`dcg_room`: 1024 rows): at 2048 rows
its 64 CTAs (225 KB of shared memory each) pushed dQ / dK dV off their SMs (dK / dV 145.9 -> 150.1 us end, step 98.2 -> 101.1). ESMFold2
master L 1024 83.9 -> 82.1 us (graph median; the late launch ends 6.7 us after the adaLN, the four-launch tail 8.7).

**Measured and dropped on the way**: (1) the late launch with 9 / 6 ring slots (all stages in flight): level -- a CTA's 128 KB of dmod
through TMA at ~80 GB/s is its time, not the ring depth; 2 / 1 column splits: 9.4 / 15.8 us (the per-row epilogue chunk). (2) the
widening's grid at 6 blocks a SM (see above).

**Checks**: `abx.py` with random non-zero weights (N(0, 0.1) for every parameter: the AdaLN-Zero init has zero modulation / gates)
and every floating input requiring grad, outputs + all parameter + input gradients, `torch.equal` and max relative difference against the
switch off in the same process, 3 runs with it on: SAVE_LATE equal on 10 cases (ESMFold2 master L 1024 / 2048 / 3072 / 6144, SWA 3 blocks
L 1024, embedder six variants L 2048); DCG within 1e-3 (fp32) / 4e-3 (master, one bf16 step) on 9 cases (ESMFold2 fp32 L 1024, SWA L 1024 both
dtypes, embedder six variants L 1024), the same bits over 3 runs; `dcgp.py` the kernel alone on random dmod / Wmod / c (M 1024 / 2048 / 4096)
against fp64: 6.8e-4 - 8.0e-4 of the max. Graph replays (`stressmt.py`, master and float32, 1000 replays x 5 cases): no hang, the same
gradient bits. Accuracy (`f1/acc.jsonl`, the three classes' training L 1024 / 2048 / 4096, both dtypes, against fp64): no RED, every case
within 5 tol; `acccmp.py` lists the 2 cases of part 18 above their B200 token (SWA S48 global master, paths not changed here).

**Measured** (`t10/f2/ana2.txt`: the three classes' training cuda, one measurer run per length group, ESMFold2 in runs of its own; other
cases from `t9/f1`): 68 -> 64 above 0.667, 0 slower than the H100, ladder 0 violations. ESMFold2 master train L 1024 86.0 -> 81.9 us
(0.769 -> 0.733), L 3072 112.7 -> 110.5, L 5120 155.6 -> 149.5, L 6144 165.9 -> 161.8, others level; float32 train 0 above (L 1024 0.652).
SWA master train window S 1 L 2048 0.714 -> 0.698. The ESMFold2 one-sample target (L 1024 74.5 us) is not reached: what remains after dK / dV
is the dx GEMM (5 us) -> adaLN (2) -> the late dc launch (6.7), the dW cluster kernel beside it ending ~10 us after dK / dV.

**Prebuild** (`f1/prebuild_list.txt`): the sm100 plan of `swa_dit` holds 106 cubins, among them `gattn_bwd_c`, `_c2` / `_c3` / `_c4` / `_c8`,
`gattn_fwd_c`, `gattn_fwd32_c`, `dcg_tf32`, `wpack_bf16` (both `CUBINS` tables are planned whole); a fresh process of embedder global master
L 2048 and ESMFold2 master L 1024 compiled nothing.

### Part 20 (2026-11-18, r3-swa): the TF32 backward's store round trips; what bounds the one-sample chain (probes)

`backend_versions["cuda:b200_sm100"]`: `AtomDiTMiniWorld` 17 (`AtomDiTESMFold2` inherits it), `SWAAtomTransformer` 25, `InputFeatureEmbedder` 26.
Base `b200-2.5.x` f61cd6d0d (engine 2.5.14). Records `$W/b2511/work/r3-swa/t11/` (probes `_tools/t11/ref.py` (outputs + every parameter
and input gradient of random-weight training cases saved from one build and compared bitwise with another), `tma_probe*.py` (TMA time
per SM), timelines `tl*_*.txt`, in-kernel `%globaltimer` / `clock64` stamps `trp*.txt` (a `-DTRP=1` development build, removed); `f1/`:
the three classes' training `cuda`, one measurer run per length group, `dev.sh -x` GPU 2; `f1/ana.txt`).

**Where the one-sample chain goes** (ESMFold2 master training L 1024, in-kernel stamps, us from the kernel's first instruction):
- FFN gate kernel (64 CTAs, 12 us): its inputs arrive ~1.3 us after its wait, then one k-block (y | dq2 | gate_f, 32 KB) every
  ~0.8 us -- 4.4 - 5.6 us before the tile is in TMEM. Not the issue rate: a TMA probe (`tma_probe.py`, 128 KB a CTA, 64 CTAs) takes
  ~1850 clocks L2-hot with 1, 2, 4 or 8 issuing lanes and boxes of 8 - 32 KB (4-KB boxes: 1 lane 3600, 8 lanes 2100 clocks) -- part
  10's "~600 clocks per issue a lane" holds for small boxes only. It is the source: with L2 flushed before the launch (the measurer
  flushes before every replay) the same loads take 6755 clocks (3-KB-stride rows of `mod`) / 5707 (64-B bf16 rows, 64 KB): the
  backward's first kernel reads dy (and its tiles) from HBM at ~13 - 30 B/clock per SM. The first k-block's row-thread pass also pays
  ~1000 clocks of instruction fetch (the code of each kernel is cold after the flush; the second pass over the same loop: 150 clocks).
- out-projection backward (32 CTAs, 9.4 us): phase A (dq1, gate_a) ends 2.6 - 3.8 us after its wait; then the head's three output
  tiles leave through ONE staging tile, a wait for the previous TMA store's read before each (~1 us each), and the CTAs of group 0
  stored all four DATT k-blocks the same way (1.3 us later than the other CTAs, which the next kernel waits for).
- attention backward: dK / dV launches on a side stream after the out-projection ends (no PDL): 1.5 us from its launch to its first
  load; its epilogue (head tail + stores) ~2.6 us.
- after dK / dV: dx GEMM 5 -> adaLN 2.3 -> late dc 7.5 us, and beside them the dW cluster kernel (ends 11 us after dK / dV) and the late
  dWmod cuBLAS GEMM (6.3 us after the adaLN): three chains of 9 - 15 us; shortening one of them moves the step end by <= 2 us.

**Taken** (the same bits: `ref.py`, every output, parameter and input gradient equal to the part-19 build on ESMFold2 master L 1024 /
4096, ESMFold2 float32 L 1024, SWA 3 blocks master / float32 L 1024, the embedder's six variants master L 1024; the same bits over 3 runs):
- `oproj_bwd_fused_tf32` (NG >= 4 only): one item a CTA with one head (NG 4, <= 4736 rows) stages the head's dO / dG / gated tiles in ring slots 2 - 7
  (never refilled after phase A: entries 8 / 9 go to slots 0 / 1) -- three stores without a round trip between them; the DATT stores of
  a tile shared by its items (k-block kb by group kb mod NG; was group 0 alone).
- `ffn_bwd_gate_tf32` (NG >= 4 only): the DFFN stores shared the same way (k-block kb by group kb mod NG); with one item a CTA the four dy-partial
  chunks stage in the spent activation ring (no store round trip; was one tile, a wait each).
- `adaln_bwd_tf32`: 4 - 8 dy partials loaded at once, then added in order (was a dependent load per add: eight L2 round trips).
The first build applied all three at every NG and also prefetched the attention epilogues' row operands into L1: SWA 3 blocks master
L 6144 (NG 2) measured +0.5 % (487.3 -> 489.4 - 490.5 us, HEAD tree against this one, interleaved); restricted to NG >= 4 (<= 2048
rows for the gate, <= 4736 for the out-projection) and without the prefetch: L 6144 484.3 / 485.3 against 488.3 / 487.3 (`_tools/t11/ab.sh`).
ESMFold2 master L 1024 timeline (graph, `tl18_e1024.txt` against `t10/tl5_e1024_d1.txt`): out-projection ends 8.1 us after the adaLN
(was 9.5), the backward ends at 148.9 (was 149.8 - 151.0).

**Measured, not taken** (diffs in `docs/b200/rejected/`):
- `swa_part20_fada_gate_adaln.diff` -- the FFN adaLN backward inside the gate kernel (FADA: a tile barrier on a self-advancing global
  counter after the dy partials, then the row code of `adaln_bwd1` on the bf16 saves, a row per half-warp with the same butterfly order:
  the same bits). Correct, deterministic, no hang -- and slower: the gate kernel ends 15.8 us after it starts against 12 + 3.8 for the
  two launches (the barrier: the partial stores' completion, the release and the poll ~1 us; then the partials' L2 latency, which the
  separate launch also pays but behind PDL). ESMFold2 L 1024 81.9 -> 83.9 us.
- `swa_part20_att_one_launch.diff` -- dQ and dK / dV passes as one launch on the main stream (PDL under the out-projection; the kernels
  as device bodies of (CTA index, CTA count), `attn_dqkv_tf32.cu`): the same bits; ESMFold2 L 1024 80.0 -> 83.9 us (the 64 CTAs launch
  under the out-projection beside the dW / dc side-stream kernels: 168 CTAs for 148 SMs, the attention ends 2.4 us later), L 2048 level.
- `swa_part20_multilane_tma_neutral.diff` -- a ring slot per issuing lane in the gate, out-projection, attention backward, `ffn_fwd2`
  and `attn_fwd1h` producers (the brief's "one TMA per ~600 clocks a lane"; the boxes here are 8 - 16 KB): level (82.9 / 82.0 us), as
  part 10 found for the attention.
- the gate's y | dq2 in one ring slot (8 entries a tile instead of 12): level -- the k-blocks arrive at the HBM rate either way.

**Measured** (`t11/f2/ana.txt` against `t10/f2/ana2.txt`; the three classes' training `cuda`; ratio B200 / H100 `cuda`): 64 -> 63 above
0.667, 0 slower than the H100, ladder 0 violations. ESMFold2 master train L 1024 81.9 -> 81.8 us (0.733 -> 0.732), L 2048 96.2 -> 94.3
(0.765 -> 0.749; A/B 96.2 -> 92.3 / 94.4); SWA 3 blocks master S 1 L 2048 264.1 -> 262.2 (0.698 -> 0.693), L 4096 371.1 -> 368.6; embedder
master train 22 -> 21 above (window L 4096 esmfold2 linear 0.671 -> below). ESMFold2 L 5120 149.5 -> 153.4 is run noise: 40 tiles, NG 2,
the code path is the part-19 one (151.6 in the first run of this part).

Still above 0.667 (ESMFold2 one-sample master training, target L 1024 74.5 us): the step is the sum of per-kernel latencies -- each
stage's first loads from HBM after the measurer's L2 flush (dy) or from L2 (~1 - 2 us a dependent round trip), the code fetch of each
kernel (~0.5 us), and three tail chains of 9 - 15 us after dK / dV (dx GEMM -> adaLN -> late dc; dW cluster; late dWmod) of which all
three must shorten together. Fusing a stage pair through a global barrier (FADA) or one launch for both attention passes did not pay
(above). Next: dx = dP W as per-head partials in the attention epilogues (summed by the adaLN, as the gate's dy), the late dWmod rows
and the dW tiles of dq / dk / dv beside it -- every tail chain at once.

### Part 21 (2026-11-18, r3-swa): dx in the attention epilogues, dWqkvg by readiness, the dW cluster's partials through L2

`backend_versions["cuda:b200_sm100"]`: `AtomDiTMiniWorld` 18 (`AtomDiTESMFold2` inherits it), `SWAAtomTransformer` 26, `InputFeatureEmbedder` 27.
Base `b200-2.5.x` 4c597d230 (engine 2.5.14). Records `$W/b2511/work/r3-swa/t12/` (timelines `tl*_e*.txt` from `_tools/tl.py`; probes
`_tools/t12/adcp.py`, `dwp.py`, `dwtr.py` (clock64 trace of the dW cluster kernel, TRACE build); `ref_dxp0.pt` = outputs + every parameter /
input gradient of the part-20 build (`_tools/t11/ref.py`); `f1/` A/B, stress and accuracy; `f2/`, `f3/` the three classes' training `cuda`,
one measurer run per length group, `dev.sh -x` GPU 2; `f3/ana.txt`, `f2/ana_before.txt` against the merged tree's `tokens/h100_sm90`).

**Taken**
- **DXP** (`attn_dq_tf32.cu` / `attn_dkv_tf32.cu -DDXP=1`, cubins `attn_dq_tf32_x` / `attn_dkv_tf32_x`, `tf32_bwd.DXP`): where both attention
  passes and the early weight gradients fit the SMs (one item a CTA; one sample of 1024 atoms), each pass's epilogue also writes its head's
  share of the qkvg backward's dx = dP W -- dQ pass [dpq_h | dg_h] [Wq_h; Wg_h], dK / dV pass [dpk_h | dpv_h] [Wk_h; Wv_h] (tcgen05 kind::tf32,
  M 128 x N 128 x K 64, TMEM columns 0 - 127 once the item's accumulators are read; W rows loaded into the unused second item slot / ring
  stages 8 - 11, the partial staged in the spent ring and stored by TMA) into `xp` [8, M, C]; the qkvg adaLN sums the eight partials in order
  (its `ndy` path). No cuBLAS dx GEMM after dK / dV (5.2 us). Max relative difference to the part-20 build 1.4e-3 (master) / 6.8e-6 - 2.6e-4
  (float32), the same bits over 3 runs. At 2048 rows (no early weight gradients) measured level / +2 us: routed off there.
- **DWS** (`tf32_bwd.DWS`, `DwTf32` `run.tiles`): dWqkvg's four 128-row m-blocks launched as their operands arrive -- dWg (tile 3) with the
  early tiles after the out-projection, dWq (tile 0) after the dQ pass on a stream of its own, dWk / dWv (tiles 1 - 2) after dK / dV on
  clusters of 8 (`DWS_CL`). 8 instead of 4 partials for tiles 1 - 2: max relative 8e-7 against off, deterministic.
- **DW_GX** (`dw_tf32.cu -DDW_GX=1`, cubins `dw_tf32_cl4_r32g` / `_cl8_r32g`, `DwTf32.GX`): the cluster build's partials meet in an L2
  workspace instead of DSMEM. A clock64 trace of the DSMEM form (CTA 0, 1024 rows, 2 tiles x 8 CTAs, L2 hot) had the MMAs done at 2995
  clocks, the partial landed at 8072 and the final cluster barrier at 14011: ~9700 of the kernel's ~14000 clocks were the 64-KB exchange.
  Now every rank stages its accumulator as SW128 [128][32] chunks and stores it with ONE bulk copy (64 / 128 KB), one cluster barrier,
  then rank o bulk-loads the o-th contiguous 1/CL of each rank's partial (8 KB a copy) and adds them in rank order (0 + s0 + s1 ..: the
  DSMEM form's order -- the same bits on every checked case, `ref.py`). Trace: 14011 -> 9163 clocks (the bulk store of 64 KB still takes
  ~4600 clocks: ~14 B/clock a SM; per-thread row stores were worse, ~6000). In the step: ESMFold2 master L 3072 - 8192 -1.3 to -2.2 %.

ESMFold2 master training L 1024 (graph timeline `tl7_1101_e1024.txt` against `tl0_e1024.txt`, us after the step start): dK / dV end
134.1 -> 135.1 (+1 us of epilogue partial), then dx GEMM (5.2) -> adaLN -> late dc (end 149.0) became adaLN 136.1 - 138.8 -> late dc (end
145.2); the dW tail 144.9 -> 143.0, the late dWmod cuBLAS 149.2 -> 145.4. Step 81.9 -> 77.9 us (graph median, `tlf_e1024.txt`); measurer
81.8 -> 79.8 us.

**Measured, not taken** (diffs in `docs/b200/rejected/`):
- `swa_part21_adc_kernel.diff` + `swa_part21_adc_dwmc_host.diff` (ADC): the qkvg adaLN backward with dc's late part in one launch
  (`adc_tf32.cu`: 16-row blocks, the [16 x 2C] x Wmod[0:2C] product on mma.sync TF32, Wmod by TMA before griddepcontrol.wait): correct
  (dc 2.5e-4 vs fp64), 8.5 us in the step against 2.6 + 6.4 for adaLN + `DcgTf32.late`; it also delays the dWmod GEMM: L 1024 78.1 -> 83.9.
  Probe (`adcp.py`, ADCX builds): the mma.sync loop ~2.3 us of it, unchanged with four independent accumulators and plain loads; the Wmod
  load is not it (an L2 prefetch of Wmod in the dQ pass changed nothing).
- DWMC (same host diff): dWmod on the dW cluster kernel, rows 2C .. 6C early (after the FFN adaLN or after the out-projection) and rows
  0 .. 2C after the adaLN: L 1024 78.1 -> 83.8 - 94.0 (the early launch beside the attention passes stretched dK / dV; the late one 11.7 us).
- `swa_part21_rsplit_qkvg1t.diff` (RSPLIT, item 3 of the brief): `qkvg_fwd1t`'s pair build with both warpgroups on both projections of the
  rank by rows (Q | K with the head RMS + RoPE, then V | G; 16 rows each) instead of one projection each: the same bits, master eval L 1024
  22.3 -> 22.3 us, L 2048 26.4 -> 26.4 (the qkvg kernel 69.9 - 76.9 both ways): its epilogue is not on the critical path.

**Measured** (`t12/f3/ana.txt` against `t12/f2/ana_before.txt` = the part-20 records; ratio B200 / H100 `cuda`, the merged tree's tokens):
63 -> 62 above 0.667, 0 slower than the H100, ladder 0 violations. ESMFold2 master train L 1024 81.8 -> 79.8 us (0.732 -> 0.714), L 3072
110.5 -> 108.4, L 4096 133.1 -> 130.8 (0.858 -> 0.843), L 5120 - 8192 -1.2 to -2.2 %; L 2048 94.3 -> 94.2. Embedder master train 21 -> 20
(window esmfold2 relu L 8192 0.668 -> below). SWA master train 10 -> 10. Accuracy (`t12/f1/acc.jsonl`, the three classes' training L 1024
/ 2048 / 4096, both dtypes): no RED; `acccmp.py` lists the 2 SWA S48 global master cases of parts 18 / 19 (paths not changed here).
A/B with random weights and input gradients (`_tools/t10/abx.py`, DXP and DWS, 9 cases each): bad 0, the same bits over 3 runs; graph
stress (`stressmt.py`, 1000 replays x 5 cases, master and float32): no hang, 0 differing checks.

Still above 0.667: ESMFold2 one-sample master training (L 1024 79.8 us against 74.5): after dK / dV the adaLN (2.5 us) feeds two 6-us chains
(the late dc launch and the dWmod cuBLAS GEMM), the dW tail ends beside them; a store of a 64-KB partial costs a CTA ~4600 clocks, a 128-KB
TMA read ~2000 (L2) -- the next lever is fewer bytes a CTA on those chains, not fewer launches.

### Part 22 (2026-11-18, r3-swa): the one-sample tail's per-CTA bytes (measured; nothing taken)

Base `b200-2.5.x` 3270a03ab (part 21 merged). No source change kept, no version bump. Records `$W/b2511/work/r3-swa/t13/` (bit-identity
against a tree of the base's sources, `agents/r3-swa-ref`: `ref21.pt` / `ref21b.pt` by `_tools/t11/ref.py`; clock64 traces
`_tools/t12/dcgtr.py`, `dwtr.py`; timelines `tl*_*.txt`; `f1/`: the three classes' training `cuda`, one measurer run per length group,
`dev.sh -x` GPU 2, `f1/ana.txt` / `f1/ana_before.txt`). Prebuild: the sm100 plan of `swa_dit` (`CUBINS` + `tf32_fwd.CUBINS`, 110 names)
holds part 21's `attn_dq_tf32_x`, `attn_dkv_tf32_x`, `dw_tf32_cl4_r32g`, `dw_tf32_cl8_r32g` (listed from a B200 process).

Where ESMFold2 master training L 1024 ends (graph timeline, us): dK / dV 134.5, qkvg adaLN 135.6 - 138.1, then three chains -- the dW tail
(dWk / dWv, 142.4), the late dc launch (`DcgTf32.late`, 145.2) and the dWmod cuBLAS GEMM (144.8).

**Measured, not taken** (diffs in `docs/b200/rejected/`):
- `swa_part22_dcg_st_presigmoid.diff`: the late dc launch's epilogue. Trace (CTA 0, 1024 rows, L2 hot, clocks): the 8 stages land ~480
  apart, the accumulator at 5757, the epilogue ends at 10008 -- 32 sigmoids (IEEE divide) a thread, one warp a sub-partition, after the GEMM.
  (a) s = sigmoid(c), t = 1 + c (1 - s) formed before the accumulator wait, the product kept as (g s) t: 10008 -> 9250; (b) s, t made by
  `swa_mod_silu_st_sm100` with a = silu(c) at the backward's start (off the chain) and loaded before the wait: -> 7543. The same bits on 11
  cases (ESMFold2 master L 1024 / 2048, float32 L 1024, SWA 3 blocks both dtypes L 1024, the embedder's six variants: `equal True`).
  In the step the late dc launch ends 145.2 -> 143.1, but the dWmod GEMM (144.8) then ends the step: 78.0 -> 77.8 us (graph, 3 interleaved
  pairs against the base tree), measurer 79.8 -> 79.8.
- `swa_part22_gx_own_slice.diff`: DW_GX ranks keep their own slice (store and load (CL - 1) / CL of the partial): the same bits; trace
  9163 -> 9156 clocks (2 tiles x 8 CTAs): the store phase is not bound by those 8 KB.
- `swa_part22_dwl_split_dwmod.diff` (DWL): dWmod's rows 2C .. 6C on cuBLAS after the FFN adaLN on a stream of its own, rows 0 .. 2C after
  the adaLN on the dW cluster kernel (GX, clusters of 8): ESMFold2 L 1024 77.9 -> 85.8 us, SWA 3 blocks L 1024 217.0 -> 229.2 (two
  interleaved runs each): the early cuBLAS beside the attention passes stretched dK / dV (134.5 -> 137.7) and the late cluster launch took
  8.5 us in the step (alone ~4.8). Found on the way: `bind_mod`'s closure held the returned dWmod, so AccumulateGrad copied it
  (`memcpy32_post`, 1.8 us at the step's end) -- part 21's DWMC numbers carry the same copy.

Full measurement (`t13/f1/ana.txt` against `ana_before.txt`; the build measured carried (a) + (b) and the own-slice change, all
bit-identical): 62 -> 63 above 0.667 (the one added: embedder window esmfold2 linear L 4096, 0.679, a boundary case on a path only the
bit-identical own-slice change touches), 0 slower than the H100, ladder 0. ESMFold2 master train L 1024 79.8 -> 79.8 us, L 5120 151.3 ->
149.5, L 8192 175.9 -> 174.2, others level. Graph stress (300 replays x 5 cases): 0 differing checks. Not taken: no condition moved.

Not reached in this part: the embedder global L 3072 - 8192 uneven cluster split (48 items, 45 co-resident clusters of 3: a mixed CS 3 /
CS 2 launch ends at the CS 2 items' time -- needs smaller items, not a split), kernel merges of the cold-start glue, a re-read of the newest
H100 SWA / embedder kernels.

### H100 techniques against the B200 path (audit, 2026-11-17; `docs/h100/swa_atom_dit.md`, `kernels/swa_dit/cuda/swa_fp32.cu`, `dispatch.py`)

The three classes share the block (`AtomDiTMiniWorld` / `AtomDiTESMFold2` one block, `SWAAtomTransformer` a stack of them,
`InputFeatureEmbedder` a 3-block stack plus its own statements), so one table serves all three.

| H100 technique | B200 | note (measured) |
|---|---|---|
| bf16 inference: 4 stage kernels (qkvg, attention, out-proj + FFN; modulation a cuBLAS GEMM in front) | used, beyond it | 3 kernels with the modulation inside (`*_fwd1t` / `*_fwd3`), no fp32 `[R, 768]` |
| one modulation `silu(c)` per stack | used | `SWAAtomTransformer` bf16 stacks; inside the row kernels on the few / many-tile paths |
| weight casts / packs cached per weight version (inference) | used | `master_pack`, `eval_form` (no weight launch in a replay) |
| PDL with weights copied before the gate (slab / attention / wgrad kernels) | used (inference row kernels, TF32 stages, adaLN backward) | not before the gate where a weight form is made in the same graph (`wprep_tf32` in training) |
| window attention as one pass over the 192 keys of a 64-query tile | used (own tcgen05 form) | `attn_fwd1h` / `attn_fwd3` / `attn_fwd_tf32` |
| attention backward dQ and dK / dV in one kernel | equivalent | two kernels launched side by side (dK / dV on a side stream when both fit): one link of the chain either way |
| fp32 block on TF32 with `cvt.rna` operands, saved fp32 intermediates | used | the B200 TF32 stages save q1 / att / y / ffn / x / p_q / p_k as the H100 does |
| fp32 FFN weights split tf32 hi + lo (two passes) | not applicable | accuracy already better than the H100 token on every float32 training case (err / tol 1.06 - 1.43 against 3.2 - 4.5, `acc8_ft.jsonl`) |
| fp32 row kernels on a pair of CTAs per 64-row tile, weights split between them | not applicable (measured) | the `ffn_fwd_tf32` cluster build (2 / 4 / 8 CTAs per tile, each a fraction of the weights): float32 inference L 1024 46.9 -> 46.9 / 47.1 / 69.4 us -- the stage is not bound by its weight stream on the B200 |
| fp32 backward: dy = [da \| db] Wab and the adaLN backward INSIDE the FFN backward kernel; dx = dP W and the adaLN backward inside the qkvg backward kernel (K split over the CTA pair, partials by `st.async`) -- 4 stage kernels per block backward | partly (part 6) | dy = [da | db] Wab now inside `ffn_bwd_gate_tf32` (DYF, few tiles; per-item partials summed by the adaLN backward); dx = dP W + the qkvg adaLN backward stay cuBLAS + `adaln_bwd` (a CUDA-core fold measured 143 against 109 us). Part 5 note: Inside `ffn_bwd_gate_tf32` the dy accumulator (128 TMEM columns) does not fit: y 0 | dffn 128 | a\|b x 2 256 | dh x 2 384 use all 512 columns for the whole tile, and [da | db] is staged SW64 for TMA stores, not as the K-major SW128 A operand a dy MMA needs -- the fold means y or dffn as a shared-memory operand and a re-staged dab, a rewrite of the kernel. A separate fused GEMM + adaLN kernel (cluster K split, DSMEM reduce) did not win (L 1024 110.6 -> 114.5 us; the stage costs what the two did): the gain needs the GEMM inside `ffn_bwd_gate_tf32` (its chunk items are dy's K) and inside a qkvg-backward stage, as the H100 does |
| five weight gradients in one launch, cluster of 8 per output tile, reduction over DSMEM (no workspace, no reduce launch) | partly | one launch + a deterministic reduce launch (`dw_tf32` + `dw_reduce_tf32`), both off the dependent chain on a side stream at few tiles |
| `dmod` by plain stores when one slab holds every augment | used | A <= 8 (`*_st` builds), A = 1 warp-per-row adaLN backward |
| bf16 / master training on 64-row slab kernels at any row count | partly (part 5) | master training of window attention on <= 8192 rows runs the B200 float32 training block (TF32 kernels, item kernels of few tiles; x / c cast to fp32, the output back to bf16): 30 of the 48 refused cases served, L 7168 / 8192 0.84 - 0.92 (were 0.96 - 1.06). The bf16 32-row-tile backward itself is not built; global attention below 7168 rows stays refused (18 cases) |
| global attention on FlashAttention-4 | used / beyond | forward on the native `gattn_fwd.cu` (inference and, since swa-global part 3, training), backward `gattn_bwd.cu`, split builds for small grids |
| fp32 modulation GEMM on cuBLAS TF32 | used | `mod_fwd_tf32` (own kernel) below 4096 rows, cuBLAS above; backward on cuBLAS + `mod_bwd_tf32` |
| bf16 inference: modulation inside the row kernels (B200 form, H100 has it in front) -- fp32 counterpart | used (part 7, few rows) | `qkvg_fwd_tf32 -DMODF`: shift_a / scale_a from c inside the qkvg stage, the modulation kernel beside it for the FFN; one augment, <= 1792 rows (more rows: the modulation on fewer channel groups ends after the attention, 3072 rows 49.2 -> 53.2 us) |
| bf16 / master training on 64-row slab kernels (dx and the modulation backward inside tensor-core stages) | partly (part 8) | built and measured, not routed: pre-wait weight builds, one-augment 64-row qkvg / out-projection backward paths, a dc slab kernel (ESMFold2 L 1024 157.5 -> 137.0 us, slower than the TF32 backward's 109: the stages run 1.5 - 3 x slower in the step than alone); master training of one sample serves the bf16 FORWARD on the TF32 backward (L 1024 109.0 -> 87.9 us) |
| bf16 forward weight forms in one launch per step; modulation inside the qkvg stage (training) | used (part 8) | `swa_wpack_hybrid_sm100` (bf16 + TF32 forms, was 3 launches); `mod_fwd -DLATE_TRIGGER` beside `qkvg_fwd1t -DSAVE -DLATE_WAIT` (<= 3072 rows; same bits, L 1024 91.9 -> 87.9 us) |

### Measured (B200, `dev.sh -x`, every requirement case at min samples, lengths all; records `$W/b2511/work/r3-swa/spd3.jsonl`,
`spd_ife3.jsonl`, table `final_vs.txt`; H100 from the `h100_sm90` speed tokens)

Before = `$W/b2511/ladder/vs_h100_15x.txt` (the B200 tokens of the round's start).

| class | compared | miss 1.5 x before -> after | slower than H100 before -> after | worst ratio before -> after | geomean after |
|---|---|---|---|---|---|
| `InputFeatureEmbedder` | 108 | 108 -> 64 | 66 -> 8 | 1.34 -> 1.050 | 0.743 |
| `SWAAtomTransformer` | 90 | 66 -> 52 | 6 -> 3 | 1.099 -> 1.029 | 0.681 |
| `AtomDiTESMFold2` | 30 | 26 -> 25 | 9 -> 4 | 1.223 -> 1.109 | 0.814 |

Ladder (`cuda` against the B200 `triton` tokens, 2 % rule): 0 violations on the three classes (worst 1.005; the embedder had 18, up to
1.64). 48 cases stay refused as before (bf16 / master training of one sample below 7168 rows: 12 SWA / ESMFold2, 36 embedder).

By group (ratio B200 / H100, min - max; "miss" = above 0.667):

| class | dtype, mode, attention | cases | miss | slower | ratio |
|---|---|---|---|---|---|
| `InputFeatureEmbedder` | master eval, window | 24 | 0 | 0 | 0.568 - 0.646 |
| `InputFeatureEmbedder` | master eval, global | 24 | 18 | 0 | 0.617 - 0.873 |
| `InputFeatureEmbedder` | float32 eval, window | 16 | 4 | 0 | 0.513 - 0.750 |
| `InputFeatureEmbedder` | float32 eval, global | 8 | 6 | 0 | 0.604 - 0.898 |
| `InputFeatureEmbedder` | float32 train, window | 16 | 16 | 1 | 0.720 - 1.005 |
| `InputFeatureEmbedder` | float32 train, global | 8 | 8 | 1 | 0.821 - 1.023 |
| `InputFeatureEmbedder` | master train, window (L 7168, 8192) | 6 | 6 | 0 | 0.963 - 0.995 |
| `InputFeatureEmbedder` | master train, global (L 7168, 8192) | 6 | 6 | 6 | 1.018 - 1.050 |
| `SWAAtomTransformer` | float32 eval, window (1 / 3 blocks) | 16 | 2 | 0 | 0.415 - 0.700 |
| `SWAAtomTransformer` | float32 eval, global | 8 | 6 | 0 | 0.640 - 0.796 |
| `SWAAtomTransformer` | float32 train, window S 48 | 8 | 0 | 0 | 0.590 - 0.632 |
| `SWAAtomTransformer` | float32 train, window S 1 (3 blocks) | 8 | 8 | 1 | 0.724 - 1.024 |
| `SWAAtomTransformer` | float32 train, global S 48 | 8 | 6 | 0 | 0.589 - 0.702 |
| `SWAAtomTransformer` | master eval, window 1 block | 8 | 4 | 0 | 0.647 - 0.888 |
| `SWAAtomTransformer` | master eval, window 3 blocks | 8 | 7 | 0 | 0.662 - 0.851 |
| `SWAAtomTransformer` | master eval, global | 8 | 8 | 1 | 0.737 - 1.003 |
| `SWAAtomTransformer` | master train, window S 48 | 8 | 1 | 0 | 0.571 - 0.685 |
| `SWAAtomTransformer` | master train, window S 1 (L 7168, 8192) | 2 | 2 | 1 | 0.986 - 1.029 |
| `SWAAtomTransformer` | master train, global S 48 | 8 | 8 | 0 | 0.783 - 0.853 |
| `AtomDiTESMFold2` | float32 eval | 10 | 5 | 1 | 0.456 - 1.109 |
| `AtomDiTESMFold2` | float32 train | 8 | 8 | 1 | 0.709 - 1.028 |
| `AtomDiTESMFold2` | master eval | 10 | 10 | 2 | 0.719 - 1.026 |
| `AtomDiTESMFold2` | master train (L 7168, 8192) | 2 | 2 | 0 | 0.969 - 0.980 |

Accuracy (`acc2.jsonl` - `acc4.jsonl`: the three classes at L 1024 / 2048, every dtype and mode, + master inference at L 4096 / 8192):
no RED; against the round's start one case moved GREEN -> YELLOW (embedder float32 inference L 1024 window legacy, 1.01 x tol; triton
1.74, H100 cuda 1.62 on the same case), several moved YELLOW -> GREEN (the glue computes the features and the token mean in fp32).
Determinism (`_tools/det.py`): the same bits over 3 runs for the embedder (float32 / master, inference and training, both attention
kinds), `SWAAtomTransformer` float32 global (training S 48, inference) and `AtomDiTESMFold2` float32 inference.

### Still above 0.667, and what it would take

| group | why | what it would take |
|---|---|---|
| one sample, <= 4096 atoms (ESMFold2 inference 33 - 51 us, SWA / embedder training) | launch-bound: 4 dependent launches per block (modulation, qkvg, attention, FFN) against the H100's 5 at the same latency per launch (L 1024 master: `qkvg_fwd2` 12 us, `ffn_fwd2` 12, `mod_fwd` 7, `attn_fwd1h` 6 -- each CTA loads its weights and modulation tile first) | the modulation inside the row kernels (`silu(c) Wmod^T` as one more tcgen05 GEMM in `qkvg_fwd2` / `ffn_fwd2`: 3 launches, and no fp32 `[R, 768]` in HBM), then qkvg + attention + FFN of a few-tile sequence in one cluster-synchronised launch |
| master inference on many rows (window, 1 / 3 blocks, 0.65 - 0.85) | HBM-bound on the fp32 modulation `[R, 768]` (126 MB at 40960 rows: written once, read by qkvg and FFN): `ffn_fwd2` 65 us, `qkvg_fwd2` 53, modulation 33, attention 25 | the same in-kernel modulation (reads `c` bf16 `[R, 128]` instead) |
| global attention (embedder master / float32, SWA rows 132: 0.60 - 1.00) | FlashAttention-4 (external) is 45 - 50 % of the step at one sample (L 4096: 3 x 45 us of 289) and most of it at S 48 | not ours: an own tcgen05 global attention, or FA4 getting faster on small problems |
| float32 training of one sample (3 blocks 0.72 - 1.02) | ~21 launches per block (TF32 stages, cuBLAS GEMMs, the modulation's backward) + the embedder's autograd statements | fused backward row kernels (the GEMM inside `adaln_bwd`), `round(Wmod)` + `wprep` + `wprep_bwd` in one launch, the embedder's features / LayerNorm / read-out backward as kernels |
| master training of one sample (L 7168 / 8192: 0.96 - 1.05; below 7168 rows refused) | the sm_100a backward's 128-row tiles (as before) | 32-row-tile backward row kernels (unchanged gap of the optimization round) |

### Changes outside sm_100a-only code (round 3)

- `modules/embedder.py`: `InputFeatureEmbedder._forward_cuda` now names its eight parameters (was `*args, **kwargs`) and takes
  `_forward_b200` only for `b200(pos)`; every other card calls `_forward_pytorch` with the same arguments as before.
- `modules/atom_dit.py`: `SWAAtomTransformer._forward_cuda` gained a `b200(q)` branch (bf16 inference stacks); `AtomDiTMiniWorld._forward_b200`
  an optional `silu` argument. `kernels/swa_dit/cuda/sm100` is sm_100a only. No shared `.cu` / `.cuh`, no Triton kernel, no requirement
  CSV changed.
- Two new sm_100a cubins are built on first use: `swa_dit_sm100/embed_glue` and `swa_dit_sm100_tf32/gattn_tf32` (the sm100 prebuild
  plan has to pick them up; not re-generated on this branch).
## Round 3b (2026-11-17, branch `b200r3-atom-dit`): the 30 cases above 1.5 x and the inference accuracy of the variant blocks

Continues the round-3 section below (same classes, same target: B200 `cuda` median <= H100 `cuda` median / 1.5; every paired case of
these classes has an H100 median >= 59 us, so the 20-us exemption touches none). `cuda:b200_sm100` = 4 on `AtomDiTBoltz2` /
`AtomDiTRoseTTAFold3` / `AtomDiTChai1`. Records: `$W/b2511/work/r3-atom-dit/` (`ab_fe1.jsonl`, `ab_fe2.jsonl` inference speed, `ab_ft.jsonl`
training speed, `acc_f4.jsonl` accuracy, `miss4.txt`, `tm*.log` the A / B runs, `pr3.log` the kernel timelines).

### What changed

- **fp32 inference, parallel schedule (RoseTTAFold3, Chai-1): the transition before the attention's end** (`atom_post_tf32 -DEARLY`, the
  default of `-DPARS`). In the parallel schedule x2 = LN(s) s2 + b2, [a | b], h and t read only the block input and the tables, which are
  complete when the kernel starts as the attention's programmatic dependent (every fp32-path kernel waits before it lets its dependents
  go). So the unit runs x2 -> [a | b]_0 -> h_0 -> t, [a | b]_1 -> h_1 -> t (16 of its 20 weight slices) under the attention, then
  `griddepcontrol.wait`, gated -> y = gated Wo^T (Wo last in the ring) and the output phase. Same arithmetic, same values. In-process A / B
  (`b200cfg/tm.py`, two captured graphs timed in alternating blocks, mean replay time): -7.5 / -8.4 / -2.0 us at 1024 / 2048 / 4096 atoms
  S 1 (RoseTTAFold3), -8.0 / -7.7 / -5.6 us (Chai-1), -2 .. -6 us at S 5. The gain ends where the attention's CTAs fill the SMs (the
  dependent's CTAs then start when the attention's leave).
- **bf16 / master: the AdaLN outputs and the residual updates with one rounding each** (`pre_fwd` / `post_fwd -DADA1 -DRES1`, the three
  variants' builds; forward of inference and training). Found stage by stage against fp64 (`b200cfg/stage_acc.py`: hooks on the fp64 and
  the bf16-autocast reference module against the kernels' saved activations): under bf16 autocast the reference's LayerNorm returns fp32
  and `sigmoid(scale) * x_norm + bias` stays fp32 until the next linear rounds it -- one rounding, where the kernels rounded LN(x), the
  product and the sum (x1 / x2 relative error 3.4e-3 against the reference's 2.2e-3 before its own rounding). `-DADA1`: x = rn(s LN(a) +
  b) in fp32. `-DRES1`: a2 = rn(a + so u) and a3 = rn(a2 + st t) as fp32 multiply-adds (t the unrounded accumulator), two bf16 roundings
  of the output stream less than the bf16 module (more precise than the reference there; Chai-1's reference keeps its whole stream fp32).
  Block output against fp64 (relative L2, 1024 atoms S 5; cuda before -> now, the bf16 reference): Boltz-2 4.94e-3 -> 4.46e-3 (4.67e-3),
  RoseTTAFold3 4.18e-3 -> 3.74e-3 (3.96e-3), Chai-1 5.70e-3 -> 4.89e-3 (4.89e-3). No speed trend in the A / B.
- **L2 prefetch of the first input tile** in `cond_fwd` / `pre_fwd` (`-DPF`, warp 3, a hint issued under the previous kernel): -0.3 us mean
  over the 27 master cases of the A / B (most faster, up to -2.2 us).

### Measured (B200 GPU 0, `dev.sh -x`; graph timing, L2 flushed; against the H100 `cuda` tokens)

Cases above 1.5 x: **30 -> 22** (of 376 paired; round 3 started at 376): Boltz-2 9 -> 9 (float32 7, master 2), RoseTTAFold3 20 -> 13
(float32 12 -> 6, master 8 -> 7), Chai-1 1 -> 0. Slower than the H100: 2 -> 1 (Boltz-2 float32, 1024 atoms S 1: 1.049; RoseTTAFold3's
1.109 -> 0.969). Every training case (96) and every `slide` case meets the target, as before. Ladder: 0 `cuda` > `triton` (min 3.4 x).

| class | dtype | paired | miss: round 3 -> now | worst: round 3 -> now | geomean b200 / h100 |
|---|---|---|---|---|---|
| `AtomDiTBoltz2` | float32 eval pad | 20 | 7 -> 7 | 1.051 -> 1.049 | 0.648 |
| `AtomDiTBoltz2` | master eval pad | 20 | 2 -> 2 | 0.701 -> 0.697 | 0.578 |
| `AtomDiTRoseTTAFold3` | float32 eval pad | 20 | 12 -> 6 | 1.109 -> 0.969 | 0.675 |
| `AtomDiTRoseTTAFold3` | master eval pad | 20 | 8 -> 7 | 0.731 -> 0.724 | 0.628 |
| `AtomDiTChai1` | float32 eval pad | 20 | 1 -> 0 | 0.745 -> 0.641 | 0.528 |
| `AtomDiTChai1` | master eval pad | 40 | 0 -> 0 | 0.621 -> 0.620 | 0.525 |

The 22 (us; all `pad` inference):

| class | dtype | L | S | B200 | H100 | ratio |
|---|---|---|---|---|---|---|
| `AtomDiTBoltz2` | float32 | 1024 | 1 | 64.0 | 61.0 | 1.049 |
| `AtomDiTRoseTTAFold3` | float32 | 1024 | 1 | 57.3 | 59.1 | 0.969 |
| `AtomDiTRoseTTAFold3` | float32 | 1024 | 5 | 63.4 | 74.8 | 0.848 |
| `AtomDiTBoltz2` | float32 | 1024 | 5 | 65.4 | 77.1 | 0.848 |
| `AtomDiTBoltz2` | float32 | 2048 | 1 | 63.4 | 75.3 | 0.842 |
| `AtomDiTRoseTTAFold3` | float32 | 2048 | 1 | 59.3 | 72.3 | 0.819 |
| `AtomDiTRoseTTAFold3` | float32 | 4096 | 1 | 71.6 | 94.8 | 0.755 |
| `AtomDiTBoltz2` | float32 | 3072 | 1 | 65.5 | 87.4 | 0.749 |
| `AtomDiTRoseTTAFold3` | float32 | 3072 | 1 | 61.4 | 82.3 | 0.746 |
| `AtomDiTRoseTTAFold3` | master | 2048 | 1 | 42.9 | 59.3 | 0.724 |
| `AtomDiTBoltz2` | float32 | 4096 | 1 | 71.5 | 99.8 | 0.717 |
| `AtomDiTBoltz2` | float32 | 5120 | 1 | 81.8 | 115.3 | 0.710 |
| `AtomDiTRoseTTAFold3` | master | 1024 | 5 | 43.0 | 61.6 | 0.698 |
| `AtomDiTBoltz2` | master | 1024 | 5 | 44.8 | 64.2 | 0.697 |
| `AtomDiTBoltz2` | master | 2048 | 1 | 42.9 | 62.0 | 0.692 |
| `AtomDiTBoltz2` | float32 | 6144 | 1 | 84.4 | 123.5 | 0.684 |
| `AtomDiTRoseTTAFold3` | master | 3072 | 5 | 65.4 | 96.2 | 0.680 |
| `AtomDiTRoseTTAFold3` | master | 3072 | 1 | 45.1 | 66.5 | 0.678 |
| `AtomDiTRoseTTAFold3` | float32 | 2048 | 5 | 77.7 | 114.8 | 0.677 |
| `AtomDiTRoseTTAFold3` | master | 2048 | 5 | 53.2 | 78.7 | 0.676 |
| `AtomDiTRoseTTAFold3` | master | 4096 | 1 | 49.0 | 72.6 | 0.675 |
| `AtomDiTRoseTTAFold3` | master | 16384 | 1 | 112.6 | 167.6 | 0.672 |

Why they stay (kernel timelines of graph replays, `b200cfg/prof3.py`; us from the replay's first kernel):

- *master, 4096 atoms S 1 (49 us):* pair bias 0 - 10.5 (side stream), `cond_fwd` 9.0 - 17.7, `pre_fwd` 15.7 - 25.4, attention 27.4 -
  35.4, `post_fwd` 29.3 - 45.7. The bias kernel fills the SMs, so the side stream buys nothing from ~3072 atoms on and `cond_fwd` starts
  when it leaves; the join costs the attention its dependent launch (2 us after `pre_fwd`). These cases miss by 0.5 - 3.5 us (one or two
  2.05-us timer steps). What it would take: a faster windowed pair bias (16 MB of pair per 4096 atoms: `b200_atom_local`, shared with
  AtomDiTProtenix / OpenFold3) or `cond_fwd` fused into `pre_fwd`.
- *float32:* `f32_ln_aff` 4 - 9 us and `atom_gemm_tf32` 9 - 11 us before `atom_pre_tf32` (12 - 17 us after its tables), the attention 10 -
  11 us, then `atom_post_tf32`: 17 - 20 us after the attention for Boltz-2 (sequential schedule with the up-gate: 28 weight slices per
  unit, nothing to run early), ~7 us for the parallel schedule where the SMs are free. What it would take: the weights resident for the
  unit (the round-3 note below) and the tables' LayerNorm + GEMM inside `atom_pre_tf32`.

Tried, measured and removed (in-process A / B means): the bf16 `post_fwd` with the same early transition (+0.1 us mean, -1.6 .. +0.8: its
weights' staging already overlaps the attention and the MMAs are short); the fused TF32 kernels' weight ring started before the wait
(+-1 us, no trend); the pair bias on the main stream ahead of the chain (Boltz-2 / RoseTTAFold3 master 0 .. -1 us, Chai-1 and float32 up to
+6 us).

### Accuracy (`acc_f4.jsonl`: every row at 1024 / 4096 / 8192 / 16384 atoms, min samples, inference + training; 148 cuda cases)

No RED, no unsupported case. The measurer's error is max |x - gold| / max |gold| with tol = max(err_pytorch, fixed): one worst element
decides a light, so a path at the reference's own rounding level lands on either side of tol. err / tol over the inference cases
(geomean; round 3 -> now): Boltz-2 master 0.98 -> 0.87 (16 / 16 GREEN, was 14), RoseTTAFold3 master 0.80 -> 0.71 (16 / 16 GREEN),
Chai-1 master 1.10 -> 1.02 (16 GREEN / 16 YELLOW, was 10 / 22; worst 1.53 where tol is the fixed 8e-3), float32 unchanged (same
arithmetic: Boltz-2 1.03, RoseTTAFold3 0.99, Chai-1 0.97; 27 GREEN / 21 YELLOW).

Inference cases YELLOW here while BOTH the H100 `cuda` token and the B200 `triton` token are GREEN: 14 before (Chai-1 master 4, Boltz-2
master 2, float32 8) -> 14 now, of another make-up: Boltz-2 master 0, Chai-1 master 6 (err / tol 1.01 - 1.53), float32 8 (Boltz-2 4,
RoseTTAFold3 3, Chai-1 1; err / tol 1.002 - 1.05). The bf16 double roundings are gone (Boltz-2 / RoseTTAFold3 now below the reference's
L2 error); what is left:
- *Chai-1 master:* its reference keeps the attention output and the residual stream in fp32 (no out projection: `so * out` is a bf16
  gate times an fp32 tensor) and returns fp32; the kernels' attention writes bf16 and the block returns bf16 (L2 error 4.89e-3 = the
  reference's 4.89e-3). The `atom_single_fp32` / `*_fp32_cond` streams add the entry rounding of `MasterEntryCast` (1.7e-3 of |single|
  before the LayerNorm). Below the reference would take an fp32 attention output and an fp32 `a` into `post_fwd`.
- *float32:* every product is TF32 with operands rounded to nearest, as the reference's; the fused inference kernels have no rounding of
  their own to remove (the eight cases are within 5 % of tol, the same eight as in round 3).
Training: YELLOW on every case, as on the H100 tokens and the B200 triton tokens. Determinism: two training steps bitwise equal on
every model x window x dtype (`quick_train.py`), inference twice bitwise equal (`quick_var.py`), captured graph = eager step bitwise
(`graph_check.py`).

### Found on the way (outside this round's changes)

- **The slid key window with a partial last query window (N % 32 != 0) is wrong** on the B200, variant blocks and AtomDiTProtenix /
  AtomDiTOpenFold3 alike (`b200cfg/odd.py`: 2100 atoms, the last 128 rows 4e-3 - 7e-3 off in fp32 and 6e-3 - 8e-3 in bf16, 3e-2 for Chai-1; 2048 /
  2080 / 2176 atoms are right; `pad` is right; AtomDiTOpenFold3 shows it with its default window too). No requirement length hits
  it (all multiples of 1024). The cause was not looked for: it is in the block's shared slide path (`b200_atom_local`).
- **`unspecified launch failure` under `torch.profiler`** when graph replays of several cases are profiled one replay at a time in one
  process (`b200cfg/prof3.py b200cfg/p_pr.yaml`): 5 of 6 runs of this branch, 4 of 4 on the base tree (`$W/eng-2511`), with this round's
  options off too; never without the profiler (measurer runs, `tm.py`: several thousand replays). Not diagnosed.

### Changes outside the variant path (round 3b)

- `b200_atom/pre_fwd.cu`, `post_fwd.cu`, `cond_fwd.cu`: `-DADA1`, `-DRES1`, `-DPF` (default 0: the default builds compile the same
  code; the PF builds take one more pointer argument); `b200_atom/fused_tf32.cu`: `-DEARLY` (default = `PARS && !UP`, so only the
  variant build `post_pars` changes), `setup(B, warp, mode = 0)`.
- `b200_atom/__init__.py`: `KERNELS_VAR` + `cond_pf`, `pre_a1`, `pre_kq_a1`, `post_pars_a1`, `post_up_a1` (`cond_id` is a PF build now);
  optional arguments with the old defaults: `cond_fwd(pf=None)`, `pre_fwd(ada1=False)`, `pre_fwd_kq(ada1=False)`, `post_fwd(ada1=False)`.
- `integrations/local_dit_b200.py`: `_cond_tables(pf=None)`; the variant calls pass the new flags (`var` only). The prebuild plan reads
  `KERNELS_VAR` / `KERNELS_FUSED32_VAR`, so `_prebuild --arch sm100 --list` covers the new builds.
- No H100 / A100 / RTX 5090 source or route change, no Triton change, no requirement row touched.

## Round 3 (2026-11-17, branch `b200r3-atom-dit`): Boltz-2 / RoseTTAFold3 / Chai-1 on the tcgen05 block

Target of the round: on every requirement case both cards' `cuda` serve, the B200 median <= the H100 median / 1.5 (graph timing, L2
flushed). The three variant classes started with every paired case missing (376 of 376; worst 2.70 x the H100 time): their row kernels
were the A100's `mma.sync` builds and cuBLAS (`dit_variants`), the H100 runs them as options of its fused local DiT.

### What changed

**The route.** `VariantAtomCuda._b200_tc` -> `_LocalAtomDiT._b200_tc` -> `local_dit_b200.reason()`: a `kv_norm == "shared"` block (one
AdaLN for q and k / v: these three models) goes to `_variant_reason` / `_run_variant`, the tcgen05 local DiT of the optimization round
with the models' differences as kernel builds and leaves, as the H100's `local_dit_h100.options` (the same `options()` / `_leaf` /
`_weights_of` are read):

| option | models | forward | backward |
|---|---|---|---|
| IDENT (identity AdaLN: tables of the raw c, raw scales) | Chai-1 | `cond_fwd -DIDENT` (fp32: one GEMM of c) | `post_bwd` / `pre_bwd` / `cond_bwd -DIDENT` (no sigmoid backward of the scales, no LayerNorm backward of c); fp32 `adaln_bwd32_id`, dc one K = 768 GEMM |
| PARS (parallel schedule: the transition normalises the block input) | RoseTTAFold3, Chai-1 | `post_fwd -DPARS`; fp32 `atom_post_tf32 -DPARS` | `post_bwd -DPARS` (the staged tile is the block input, the attention branch reads dy alone); fp32 `tail_bwd32` on the gate |
| KQLN (affine LayerNorm of the flat q / k) | RoseTTAFold3 | `pre_fwd -DKQLN` (on the staged rows; saves the raw q, k); fp32 `atom_pre_tf32 -DKQLN`, training `f32_kqln_fwd` | `kqln_bwd.cu` / `f32_kqln_bwd`: dq, dk in place before `pre_bwd`, the four LayerNorm leaves' gradients as partial rows |
| UP (up-gated SwiGLU `silu(a) b u`) | Boltz-2 | `post_fwd -DUP` (Wu in shared memory, hidden in two chunks); fp32 `atom_post_tf32 -DUP`, training one more GEMM | `tr_bwd_gate -DUP` (Wu resident in shared memory under the weights' staging units, a fourth accumulator; DAB = [da, db, du] [M, 768]) and `post_bwd -DUP` (Wu^T resident, its product accumulated onto dxa); one input stage per warpgroup in both; fp32 `f32_swiglu_up_bwd` + ONE K = 768 GEMM against [Wu^T, Wup^T] |
| no gating query / no out projection / no query bias | Chai-1, ... | constant leaves (gate weight 0, Wo = 2 I, bias 0): exact | the same kernels (the leaves' gradients are dropped) |
| conformer mask (`uid`) | Chai-1 | `conf_mask.cu`: -inf into the pair bias at the keys the attention reads, one launch, hoisted with the bias | the saved bias carries it |

Served: `"pad"` and `"slide"`, bf16 / master and fp32 (TF32), inference and training, one fold, shared conditioning (stride 0 over the
samples). Refused calls (another key window, geometry, `uid` shapes) keep the `dit_variants` path. `cuda:b200_sm100` = 3 on the three
classes (their own entries; `_LocalAtomDiT`'s stays 2).

**Launch count and overlap** (the small cases are launch-bound: one sample at 1024 atoms is 8 - 64 CTAs a kernel):

- *Weight packs in one launch* (`var_pack.cu`, `var_pack32.cu`): the stacked bf16 (or TF32-rounded fp32) operands, the vector pack and,
  for training, every transposed A operand (+ the fp32 conditioning backward's grouped operand and Boltz-2's [Wu^T, Wup^T]) straight
  from the parameters; a CTA per 16-column strip of a 128 x 128 unit. The torch packs were 16 (bf16 forward; `_pack` ~18) / ~12 (fp32
  training) launches of every captured step. With the L2 flushed a whole unit per CTA took 18 - 22 us a pack, the 16-column
  strips over all SMs 6 - 7 us. Inference packs are read from the eager warm-up in a capture (`param_forms(capture_reads_eager=True)`, the
  contract of `triattn_b200`, `swa_dit.dispatch`, `trimul_h100`): an inference replay runs no pack kernel.
- *The window attention as a programmatic dependent launch* (`lattn_fwd.cu -DLATTN_PDL`, `attn_fwd(pdl=True)`): `post_fwd` stages its
  weights into TMEM while the attention runs. 41.4 -> 35.3 us at 1024 atoms (Boltz-2 master eval). Against it: the join of the side
  stream before `pre_fwd` (39.4 us) and the pair bias on the main stream (34.7 us: level) -- the side stream + join before the attention
  stays. Checked: a captured graph replayed on new input contents equals the eager step bitwise (`graph_check.py`, 24 configurations x
  eval / train).
- *The pair bias (+ conformer mask) on a side stream* in every variant path (bf16 and fp32, inference and training; `_fork_bias`).
- *The small gradients in one launch* (`col_jobs.cu`): the nine bias / LayerNorm-weight gradients (+ KQLN's four) summed from the
  row kernels' partial rows into fresh tensors of the parameters' dtypes; it replaces three reductions and 9 - 13 slice + clone launches.
- *Weight-gradient buffers handed to autograd without a copy* (`_fused_wgrads(release=True)`, `AtomDw.release`): the launcher kept
  views of the destination buffers; a view holds its base, so `AccumulateGrad` could not take the buffer over and copied it -- one
  Memcpy per fused gradient, 13 - 16 a step (~20 us at 1024 atoms). Also `dpair` is the kernel's own buffer when the pair is unpadded.
  (The same flag would remove the same copies from AtomDiTProtenix / OpenFold3 training: `_fused_wgrads` is shared, default off.)
- *fp32 training's conditioning as grouped GEMMs*: the tables one LayerNorm launch + one grouped GEMM (the inference form; [cn1, cn2,
  c] is saved as one tensor), the backward one grouped GEMM [dcn1, dcn2, dc0] + `f32_ln_bwd2` (was three GEMMs each way).

A training step at 1024 atoms, one sample, `pad` (graph, L2 flushed; us): master Boltz-2 189 -> 136, RoseTTAFold3 201 -> 136, Chai-1 181 ->
130; float32 311 -> 201, 295 -> 194, 252 -> 190 (first tcgen05 version of this round -> final; the `dit_variants` path: 467 / 498 / 360 and 461 /
485 / 391).

### Measured (B200 GPU 0, `dev.sh -x`; records `$W/b2511/work/r3-atom-dit/`)

Speed: `spd_full2.jsonl` (inference, every length 1024 - 16384, every sample count) and `spd_full_t3.jsonl` (training, every length
1024 - 8192) -- graph timing, L2 flushed -- against the tokens of the same cases: the H100 `cuda` token (ratio = B200 / H100, target
<= 0.667), the B200 `cuda` token before this round, and the B200 `triton` / `pytorch` tokens (`b200cfg/final_cmp.py`; the round's scripts are scratch files of the agent tree, `$W/b2511/agents/r3-atom-dit/b200cfg/`). 376 cases have
an H100 token (the other 160 measured cases -- `atom_single` at S 5 -- have none).

| class | dtype | mode | window | paired cases | miss 1.5 x: before -> after | worst B200 / H100: before -> after | geomean B200 / H100 | x the previous B200 cuda (geomean) | x triton (min / geomean) / x pytorch (geomean) |
|---|---|---|---|---|---|---|---|---|---|
| `AtomDiTBoltz2` | float32 | eval | pad | 20 | 20 ->    7 | 2.047 -> 1.051 | 0.650 | 2.07 | 4.35 / 9.40 / 11.18 |
| `AtomDiTBoltz2` | float32 | eval | slide | 20 | 20 ->    0 | 1.140 -> 0.536 | 0.467 | 2.12 | 4.67 / 9.45 / 12.49 |
| `AtomDiTBoltz2` | float32 | train | pad | 8 | 8 ->    0 | 1.489 -> 0.648 | 0.602 | 2.29 | 3.79 / 5.91 / 6.46 |
| `AtomDiTBoltz2` | float32 | train | slide | 8 | 8 ->    0 | 1.116 -> 0.456 | 0.435 | 2.45 | 3.72 / 5.83 / 6.89 |
| `AtomDiTBoltz2` | master | eval | pad | 20 | 20 ->    2 | 2.521 -> 0.701 | 0.575 | 2.94 | 8.83 / 15.95 / 19.25 |
| `AtomDiTBoltz2` | master | eval | slide | 20 | 20 ->    0 | 1.305 -> 0.469 | 0.381 | 3.05 | 9.14 / 16.01 / 21.07 |
| `AtomDiTBoltz2` | master | train | pad | 8 | 8 ->    0 | 1.775 -> 0.522 | 0.494 | 3.33 | 6.42 / 9.44 / 10.29 |
| `AtomDiTBoltz2` | master | train | slide | 8 | 8 ->    0 | 1.253 -> 0.371 | 0.344 | 3.48 | 6.16 / 9.16 / 10.76 |
| `AtomDiTChai1` | float32 | eval | pad | 20 | 20 ->    1 | 1.592 -> 0.745 | 0.567 | 2.02 | 4.31 / 9.27 / 11.35 |
| `AtomDiTChai1` | float32 | eval | slide | 20 | 20 ->    0 | 0.990 -> 0.516 | 0.389 | 2.26 | 4.32 / 9.28 / 12.63 |
| `AtomDiTChai1` | float32 | train | pad | 8 | 8 ->    0 | 1.289 -> 0.628 | 0.561 | 1.95 | 3.44 / 5.74 / 6.39 |
| `AtomDiTChai1` | float32 | train | slide | 8 | 8 ->    0 | 0.854 -> 0.400 | 0.382 | 2.19 | 3.45 / 5.65 / 6.83 |
| `AtomDiTChai1` | master | eval | pad | 40 | 40 ->    0 | 1.794 -> 0.621 | 0.528 | 2.41 | 7.09 / 13.75 / 16.39 |
| `AtomDiTChai1` | master | eval | slide | 40 | 40 ->    0 | 1.035 -> 0.435 | 0.334 | 2.79 | 6.88 / 13.73 / 17.64 |
| `AtomDiTChai1` | master | train | pad | 8 | 8 ->    0 | 1.388 -> 0.501 | 0.470 | 2.58 | 5.71 / 8.96 / 9.93 |
| `AtomDiTChai1` | master | train | slide | 8 | 8 ->    0 | 0.898 -> 0.328 | 0.305 | 2.86 | 5.52 / 8.67 / 10.38 |
| `AtomDiTRoseTTAFold3` | float32 | eval | pad | 20 | 20 ->   12 | 2.110 -> 1.109 | 0.720 | 1.97 | 4.28 / 9.27 / 10.99 |
| `AtomDiTRoseTTAFold3` | float32 | eval | slide | 20 | 20 ->    0 | 1.153 -> 0.597 | 0.506 | 2.02 | 4.59 / 9.36 / 12.31 |
| `AtomDiTRoseTTAFold3` | float32 | train | pad | 8 | 8 ->    0 | 1.539 -> 0.616 | 0.585 | 2.41 | 3.93 / 6.07 / 6.62 |
| `AtomDiTRoseTTAFold3` | float32 | train | slide | 8 | 8 ->    0 | 1.118 -> 0.449 | 0.424 | 2.57 | 3.93 / 6.01 / 7.09 |
| `AtomDiTRoseTTAFold3` | master | eval | pad | 20 | 20 ->    8 | 2.704 -> 0.731 | 0.634 | 2.94 | 9.23 / 16.22 / 19.46 |
| `AtomDiTRoseTTAFold3` | master | eval | slide | 20 | 20 ->    0 | 1.398 -> 0.500 | 0.411 | 3.04 | 9.64 / 16.21 / 21.18 |
| `AtomDiTRoseTTAFold3` | master | train | pad | 8 | 8 ->    0 | 1.827 -> 0.514 | 0.494 | 3.45 | 6.53 / 9.49 / 10.34 |
| `AtomDiTRoseTTAFold3` | master | train | slide | 8 | 8 ->    0 | 1.290 -> 0.371 | 0.345 | 3.58 | 6.29 / 9.20 / 10.81 |

Totals: **376 -> 30 cases** above the 1.5 x target (AtomDiTBoltz2 112 -> 9, AtomDiTRoseTTAFold3 112 -> 20, AtomDiTChai1 152 -> 1);
every training case (96) and every `slide` case meets it. Two cases are still slower than the H100 (float32 inference, 1024 atoms, one
sample: Boltz-2 1.05, RoseTTAFold3 1.11). The ladder holds on every measured case: 0 `cuda` > `triton` (min 3.4 x), triton < pytorch
unchanged (no Triton change).

Accuracy (`acc_full2.jsonl`: every row at lengths 1024 / 4096 / 8192 / 16384, min samples, inference and training, 148 cuda cases;
`acc_full_t3.jsonl`: the 36 training cases again on the final packs): **no RED, no unsupported case**. Lights GREEN / YELLOW: Boltz-2
22 / 22, RoseTTAFold3 25 / 19, Chai-1 20 / 40. Training is YELLOW on every case here, on the H100 tokens (2 of them RED: RoseTTAFold3
float32) and on the B200 triton tokens. Inference is at the bf16 / TF32 rounding level of the reference (tol = max(err_pytorch,
fixed)): of the 112 inference cases 14 are YELLOW here while both the H100 `cuda` token and the B200 `triton` token of the case are
GREEN, at err / tol 1.002 - 1.36 (the worst: Chai-1 master, `atom_single_fp32`, 16384 atoms, 1.31e-2 against 9.6e-3; the others <=
1.19); 24 are GREEN here where one of those tokens is YELLOW (`b200cfg/lights2.py`). Determinism: two training steps bitwise equal on
every model x window x dtype (S 1 at 1024 atoms, S 5 at 2048, `quick_train.py`); a captured graph replayed on new inputs equals the
eager step bitwise (`graph_check.py`).

### Still above the 1.5 x target (30 cases, all `pad` inference; us)

| class | dtype | mode | L | S | B200 | H100 | B200 / H100 |
|---|---|---|---|---|---|---|---|
| `AtomDiTBoltz2` | float32 | eval | 1024 | 5 | 63.5 | 77.1 | 0.823 |
| `AtomDiTBoltz2` | float32 | eval | 1024 | 1 | 64.1 | 61.0 | 1.051 |
| `AtomDiTBoltz2` | float32 | eval | 2048 | 1 | 63.4 | 75.3 | 0.842 |
| `AtomDiTBoltz2` | float32 | eval | 3072 | 1 | 65.6 | 87.4 | 0.751 |
| `AtomDiTBoltz2` | float32 | eval | 4096 | 1 | 71.6 | 99.8 | 0.717 |
| `AtomDiTBoltz2` | float32 | eval | 5120 | 1 | 81.9 | 115.3 | 0.710 |
| `AtomDiTBoltz2` | float32 | eval | 6144 | 1 | 83.9 | 123.5 | 0.679 |
| `AtomDiTBoltz2` | master | eval | 1024 | 5 | 45.0 | 64.2 | 0.701 |
| `AtomDiTBoltz2` | master | eval | 2048 | 1 | 42.9 | 62.0 | 0.693 |
| `AtomDiTChai1` | float32 | eval | 1024 | 1 | 59.3 | 79.6 | 0.745 |
| `AtomDiTRoseTTAFold3` | float32 | eval | 1024 | 5 | 67.6 | 74.8 | 0.903 |
| `AtomDiTRoseTTAFold3` | float32 | eval | 1024 | 1 | 65.5 | 59.1 | 1.109 |
| `AtomDiTRoseTTAFold3` | float32 | eval | 2048 | 5 | 81.9 | 114.8 | 0.713 |
| `AtomDiTRoseTTAFold3` | float32 | eval | 2048 | 1 | 67.5 | 72.3 | 0.934 |
| `AtomDiTRoseTTAFold3` | float32 | eval | 3072 | 1 | 69.6 | 82.3 | 0.846 |
| `AtomDiTRoseTTAFold3` | float32 | eval | 4096 | 5 | 124.8 | 184.9 | 0.675 |
| `AtomDiTRoseTTAFold3` | float32 | eval | 4096 | 1 | 73.7 | 94.8 | 0.778 |
| `AtomDiTRoseTTAFold3` | float32 | eval | 5120 | 1 | 83.9 | 114.0 | 0.736 |
| `AtomDiTRoseTTAFold3` | float32 | eval | 6144 | 5 | 159.7 | 238.2 | 0.671 |
| `AtomDiTRoseTTAFold3` | float32 | eval | 6144 | 1 | 86.0 | 122.7 | 0.701 |
| `AtomDiTRoseTTAFold3` | float32 | eval | 12288 | 1 | 133.0 | 197.6 | 0.673 |
| `AtomDiTRoseTTAFold3` | float32 | eval | 16384 | 5 | 368.5 | 551.6 | 0.668 |
| `AtomDiTRoseTTAFold3` | master | eval | 1024 | 5 | 45.0 | 61.6 | 0.731 |
| `AtomDiTRoseTTAFold3` | master | eval | 2048 | 5 | 55.1 | 78.7 | 0.701 |
| `AtomDiTRoseTTAFold3` | master | eval | 2048 | 1 | 43.0 | 59.3 | 0.725 |
| `AtomDiTRoseTTAFold3` | master | eval | 3072 | 5 | 65.6 | 96.2 | 0.682 |
| `AtomDiTRoseTTAFold3` | master | eval | 3072 | 1 | 47.0 | 66.5 | 0.707 |
| `AtomDiTRoseTTAFold3` | master | eval | 4096 | 1 | 49.1 | 72.6 | 0.677 |
| `AtomDiTRoseTTAFold3` | master | eval | 5120 | 1 | 57.3 | 85.8 | 0.668 |
| `AtomDiTRoseTTAFold3` | master | eval | 16384 | 1 | 112.8 | 167.6 | 0.673 |

- **float32 inference, `pad` (20 cases; ratio 0.67 - 1.11).** The chain is tables (`f32_ln_aff` + one grouped `atom_gemm_tf32`; Chai-1:
  the GEMM only) -> `atom_pre_tf32` -> attention -> `atom_post_tf32`, with the bias on the side stream. The fused TF32 row kernels are
  latency chains per work unit: fp32 weights do not fit TMEM beside the accumulators, so `atom_post_tf32` streams Wo, Wu, Ws (448 KB)
  through a 3-slot ring of 32-KB slices, one TMA round trip per slice: ~17 us a unit at any size (its profile at 1024 atoms: post 25 -
  27 us including the wait on the attention, pre 17 - 24 us including the wait on the GEMM), against ~10 us for the bf16 `post_fwd`
  whose weights are TMEM-resident. What it would take: the weights resident for the unit instead of streamed -- the SwiGLU hidden in
  quarters so that [Wa_q; Wb_q] (64 KB) + Wo + Ws halves stay in shared memory next to one 64-KB operand tile (the stage layout of the
  `-DUP` build already runs quarters), or a bf16-operand `post` with an fp32 residual stream if the engine's fp32 standard allows it
  (it does not today: the TF32 products are the reference's). The H100 pays the same structure less: its wgmma kernels take A from
  registers and keep the weights in shared memory.
- **master inference, `pad` (10 cases; ratio 0.668 - 0.731; RoseTTAFold3 8, Boltz-2 2).** Four serial launches (`cond_fwd` -> `pre_fwd`
  -> attention -> `post_fwd`), each 7 - 11 us at these sizes with the L2 flushed (TMEM allocation, 130 - 230 KB of weights by TMA and
  `tcgen05.cp` per CTA, one or two tiles of work). They miss by 1 - 4 us (the timer's steps are ~2 us). Tried without a gain: two tiles
  per CTA in `pre_fwd` / `post_fwd` (40.8 against 42.9 us at 2048 atoms S 1 only, +2 us at 1024), the join before `pre_fwd`, the bias
  on the main stream. What it would take: `cond_fwd` + `pre_fwd` as one kernel at <= 148 tiles (the tables stay in shared memory:
  one launch and one weight staging less, ~5 us), or RoseTTAFold3's flat q / k LayerNorm moved off `pre_fwd`'s staged rows.
- The H100 has no token for `atom_single` at S 5 (160 measured cases); they follow the paired cases of the same lengths.

### Changes outside the variant path (this round)

- `kernels/augmented_attention/cuda/b200_atom/` (B200-only package, shared with AtomDiTProtenix / OpenFold3 / ESMFold2Pair): new
  builds of existing sources behind `-D` flags that default to 0 (`cond_fwd` IDENT; `pre_fwd` KQLN; `post_fwd` PARS / UP; `post_bwd`
  PARS / IDENT / UP; `pre_bwd` IDENT; `cond_bwd` IDENT; `tr_bwd_gate` UP; `fused_tf32` KQLN / PARS / UP) -- the default builds
  compile the same code; new sources `conf_mask.cu`, `kqln_bwd.cu`, `var_pack.cu`, `var_pack32.cu`, `col_jobs.cu`,
  `var_rows_tf32.cu`. `__init__.py`: optional arguments with the old defaults (`cond_fwd(ident)`, `post_fwd(pars, wup)`,
  `tr_bwd_gate(wup)`, `post_bwd(pars, ident, wupT)`, `pre_bwd(ident)`, `cond_bwd(ident)`, `pre32(kq)`, `post32(pars, up)`,
  `colsum32(partials)`), `pre_fwd` unchanged in signature and result (the KQLN form is `pre_fwd_kq`), `AtomDw.release()` (new method).
- `kernels/augmented_attention/cuda/b200_atom_local/`: `lattn_fwd.cu` `#ifdef LATTN_PDL` (two statements; the default build is
  unchanged), `cubin(defs=())`, `attn_fwd(pdl=False)`.
- `kernels/integrations/local_dit_b200.py`: `_block_fwd` / `_block_bwd` / `_cond_tables` / `_fwd_fake` / `_bwd_fake` take the variant
  flags with defaults that leave the Protenix / OpenFold3 calls as they were; `_fused_wgrads(release=False)`; `_cached32(eager=False)`;
  `run_block(uid=None)`.
- `kernels/augmented_attention/cuda/prebuild_sm100.py`: the plan builds the block's cubins for the three variant classes too, plus
  their variant builds.
- `modules/dit_variant_cuda.py` (`_b200_tc`, B200 only), `modules/atom_dit.py` (`run_block(..., uid)`, the three classes'
  `cuda:b200_sm100`). No H100 / A100 / RTX 5090 source or route changes; no Triton change; no requirement row touched.

## Optimization round (2026-11-17, branch `b200opt-atom-dit`)

### What changed

**Block-local classes on the tcgen05 local DiT of `origin/b200/ladder-final`.** The wiring round ran these blocks on the A100's
`mma.sync` kernels rebuilt for sm_100a. The branch's own B200 local DiT (dc425caa5: `integrations/local_dit.py`,
`kernels/augmented_attention/cuda/sm100_atom`, `sm100_atom_local`) is now ported as B200-only code:

- `kernels/augmented_attention/cuda/b200_atom/` (row kernels: `cond_fwd`, `pre_fwd` with the chained (`-DCROSS=1`) and parallel
  (`-DPARALLEL=1`) K / V AdaLN, `post_fwd`, `tr_bwd_gate`, `post_bwd`, `pre_bwd`, `cond_bwd`, the grouped split-K weight gradient
  `dw_atom`, the fp32 path's `gemm_tf32` / `gemm_rg_tf32` / `rows_tf32` / fused `fused_tf32`) and
  `kernels/augmented_attention/cuda/b200_atom_local/` (window attention `lattn_fwd` / `lattn_dq` / `lattn_dkv` and their `_tf32` twins,
  the slid window inside them `slide.cuh` + `lslide.cu`, pair bias `lbias` / `lbias_tf32`, `lcross`). The branch's sources, copied
  rather than merged into main's `sm100_atom` / `sm100_atom_local`, whose sources also carry the H100 `-DATOM_HOPPER` half that the
  H100 build compiles (those files are untouched). Cubin families `local_dit_b200`, `local_dit_b200_tf32`, `atom_local_b200`.
- `kernels/integrations/local_dit_b200.py`: the branch's B200 block (bf16 ops `local_dit_b200_fwd` / `_bwd`, fp32 ops
  `local_dit_b200_fwd_tf32` / `_bwd_tf32`; the Hopper and ESMFold2 parts dropped), 2.5.x hoists (`_capture.weight_cache`,
  `hoisted_table`, `param_forms`), `reason()` (the gate) and `run_block()`.
- `modules/atom_dit._LocalAtomDiT._b200_tc`: on a B200 the class takes `local_dit_b200` when `reason()` accepts the call:
  `AtomDiTProtenix` (chained K / V AdaLN) and `AtomDiTOpenFold3` (parallel), `"pad"` or `"slide"`, key-mask form, bf16 / master or
  fp32 (TF32), inference and training, one fold (B > 1 folds: one call per fold, as before). Everything else keeps the sm_80 paths.
- Determinism fix (needed: the branch kernels were not run-to-run deterministic with non-trivial weights): the bias / LayerNorm-weight
  column sums of `tr_bwd_gate`, `post_bwd`, `pre_bwd`, `cond_bwd` (atomics into one vector) and of `lcross` `cond_ln_bwd` and
  `rows_tf32` `f32_ln_bwd` now write one partial row per CTA (`COLS_STRIDE`), summed in a fixed order on the host. Checked: three
  identical training steps bitwise equal (Protenix L 1024 S 5 / S 48 master, S 48 float32, perturbed weights).

**Boltz-2 / RoseTTAFold3 / Chai-1 (the `dit_variants` atom path): the attention core swapped** (the branch's
`dit_variants/cuda/sm100.py` idea). `kernels/dit_variants/cuda/train.py` `_b200_core`: on a B200, N a multiple of 128 and the
`"slide"` window or a training call, the window attention and the pair bias run on `b200_atom_local` (`attn_fwd` / `attn_bwd` and
the `_tf32` twins, `pair_bias_fwd` / `_bwd`, natural-unit bias with the key mask passed apart; Chai-1's same-conformer mask folded
into the bias at the keys the kernel reads, the slid ones in the moved windows) in place of the sm_80 `mma.sync` wrappers; the slid
window's five edge windows are inside the attention launches (no `edge.cu` edge windows; its tail kernels stay). Row kernels and
cuBLAS unchanged. A `"pad"` inference call keeps the sm_80 core (measured level, 0.96 - 1.04 x).

**SWA blocks (ESMFold2, SWAAtomTransformer).**

- training: `modules/atom_dit._SWATrainB200`, the modulation and the block as one autograd Function (bf16 / master; fp32 on the TF32
  kernels, `tf32_bwd.block_bwd_tf32(modb=...)` overlapping the modulation backward likewise). bf16: one launch casts the six
  master weights (`sm100.master_pack`, views of one bf16 buffer: `[Wqkv; Wg]` and the backward's transposes from two cats instead
  of per-weight copies); the backward overlaps independent few-CTA kernels on two side streams (`sm100._fork_join`, CUDA-graph
  capturable): dQ beside dK / dV, and the modulation backward's `dc` / `dW` kernels beside the block's weight-gradient kernel.
- Programmatic dependent launch on every bf16 backward kernel (`griddepcontrol.wait` after each kernel's barrier / TMEM setup:
  `ffn_bwd_gate`, `ffn_bwd_dy`, `oproj_bwd`, `attn_dq`, `attn_dkv`, `attn_dkvq`, `qkvg_bwd`, `mod_bwd`; `dw_bf16` already had it).
- The bf16 weight-gradient split-K on few rows: down to 128 rows per CTA, at most 16 partials (`DW_MIN_ROWS`, `DW_MAX_SPLIT`).
- Global attention on FlashAttention-4 when it is installed (`flash_attn.cute`), with FA4's deterministic backward on the B200.
- Range: bf16 training served from 7168 rows (`_B200_BF16_MIN_ROWS`, was 8192); the bf16 global-attention inference threshold
  (`_B200_FA2_GLOBAL_MIN_ROWS` = 8192) now applies only without FA4.
- L 1024 S 1 ESMFold2 master training step (CUDA graph, warm L2): 205 -> 164 (PDL, dW split) -> 138 us (one Function, streams).

**FlashAttention-4.** Installed for these runs from `flash-attn-4==4.0.0b39` (+ `quack-kernels==0.6.5`, `torch_c_dlpack_ext==0.1.5`,
`--no-deps`) into `$W/b2511/agents/_fa4lib`, put on `PYTHONPATH` for the SWA measurements only (`b200cfg/run_spd.sh ... fa4`). It is NOT
yet in the shared B200 env (`$E/.pixi/envs/default`): the coordinator's accuracy units were running on it. To make the FA4 numbers
the env's: `$ENV/bin/pip install --pre flash-attn-4==4.0.0b39` (dry run: adds exactly those three packages, nothing upgraded).

Versions: `cuda:b200_sm100` 2 for `_LocalAtomDiT` (Protenix, OpenFold3, Boltz-2, RoseTTAFold3, Chai-1), `AtomDiTMiniWorld`
(ESMFold2) and `SWAAtomTransformer`. `AtomDiTESMFold2Pair` unchanged (1).

### Measured (B200 GPU 1, `dev.sh -x`; records `$W/b2511/work/opt-atom-dit/`)

Accuracy (`acc_local.jsonl`, `acc_swa.jsonl`; every requirement case at min samples, lengths all (SWA: 1024-8192), cuda against the
fp64 reference): no RED. cuda lights before -> after:

| class | GREEN / YELLOW before | GREEN / YELLOW after | unsupported after |
|---|---|---|---|
| `AtomDiTProtenix` | 61 / 71 | 56 / 76 | 0 |
| `AtomDiTOpenFold3` | 59 / 51 | 62 / 48 | 0 |
| `AtomDiTBoltz2` | 62 / 50 | 62 / 50 | 0 |
| `AtomDiTRoseTTAFold3` | 60 / 52 | 60 / 52 | 0 |
| `AtomDiTChai1` | 63 / 89 | 60 / 92 | 0 |
| `AtomDiTESMFold2` (L <= 8192) | | 7 / 9 | 3 (S 1 training < 7168 rows) |
| `SWAAtomTransformer` (L <= 8192) | | 26 / 23 | 3 (row 130 S 1 training < 7168 rows) |

The YELLOWs are at the pytorch-relative rate of the wiring round and of the H100 / A100 tokens (Protenix's master training YELLOWs are
`ln_cond` / bias gradients at 1.2 - 1.5 x tol, unchanged in kind). Row 132 float32 training (FA4 global attention) is YELLOW at
1.7e-3 - 2.8e-3 against tol 5.6e-4 - 8.6e-4 (within 5 x), the H100 token of the same case is YELLOW too.

Speed (`spd_local.jsonl`, `spd_var.jsonl`, `spd_swa*.jsonl`; graph timing, L2 flushed, lengths 1024 / 4096 / 8192 / 16384 at min
samples; triton / pytorch / anthropic from the B200 tokens of the pretuned ladder run). cuda speedup, geomean (min - max):

| class | mode | x the wiring round's cuda | x triton | x pytorch | x anthropic |
|---|---|---|---|---|---|
| `AtomDiTProtenix` | eval | 1.92 (1.20 - 2.98) | 8.67 (4.24 - 17.33) | 10.82 | 8.68 (4.05 - 17.38) |
| `AtomDiTProtenix` | train | 2.14 (1.63 - 3.14) | 4.97 (2.54 - 10.34) | 6.56 | - |
| `AtomDiTOpenFold3` | eval | 1.63 (0.97 - 3.03) | 5.79 (1.54 - 17.09) | 7.55 | 5.68 (1.67 - 16.80) |
| `AtomDiTOpenFold3` | train | 1.56 (1.00 - 3.11) | 3.15 (1.30 - 8.71) | 3.72 | - |
| `AtomDiTBoltz2` | eval | 1.14 (0.99 - 1.46) | 4.76 (2.16 - 9.64) | 5.99 | 4.63 (2.08 - 9.45) |
| `AtomDiTRoseTTAFold3` | eval | 1.14 (0.98 - 1.44) | 4.96 (2.23 - 9.78) | 6.21 | - |
| `AtomDiTChai1` | eval | 1.13 (0.98 - 1.40) | 4.81 (1.68 - 11.26) | 6.08 | - |
| `AtomDiTBoltz2` | train | 1.14 (1.00 - 1.29) | 2.45 (1.50 - 3.62) | 2.80 | - |
| `AtomDiTRoseTTAFold3` | train | 1.13 (1.01 - 1.26) | 2.42 (1.47 - 3.59) | 2.75 | - |
| `AtomDiTChai1` | train | 1.16 (1.01 - 1.34) | 2.79 (1.46 - 4.77) | 3.27 | - |

The variants' slid-window rows gain 1.2 - 1.5 x (Boltz-2 rows 121 / 172 / 272, RoseTTAFold3 123 / 174 / 274, Chai-1 124 / 126 / 128 /
175 / 275), the "pad" training rows 1.04 - 1.05 x; OpenFold3's `slide_qblock` rows keep the sm_80 path (level). Representative
(ms, cuda / triton / pytorch): Protenix master training L 8192 S 48 (row 267) 3.021 / 21.782 / 33.214 (wiring round 5.784), float32 training 7.558 / 20.492 / 31.301 (wiring round 14.115) -- the branch measured 2.98 / 7.44 for the same block.

SWA, cuda against the pretuned triton (ms):

| class | row | case | cuda | triton | pytorch | anthropic | cuda / triton |
|---|---|---|---|---|---|---|---|
| `AtomDiTESMFold2` | 270 | L 1024 S 1 float32 train | 0.1617 | 0.1618 | 0.6880 | 1.3916 | 1.000 (was 1.071) |
| `AtomDiTESMFold2` | 270 | L 2048 S 1 float32 train | 0.1700 | 0.1800 | 1.1150 | 1.5155 | 0.945 (was 1.023) |
| `AtomDiTESMFold2` | 270 | L 1024 - 4096 S 1 master train | refused | 0.135 - 0.176 | | | (cuda 0.164 - 0.195 measured) |
| `AtomDiTESMFold2` | 270 | L 7168 S 1 master train | 0.2292 | 0.2557 | 2.1509 | 2.3322 | 0.897 (was refused) |
| `AtomDiTESMFold2` | 270 | L 8192 S 1 master train | 0.2354 | 0.2396 | 2.4995 | 2.4929 | 0.982 (was 1.145) |
| `AtomDiTESMFold2` | 119 | L 1024 S 1 float32 eval | 0.0531 | 0.0511 | 0.2661 | 0.0511 | 1.040 (unchanged) |
| `SWAAtomTransformer` | 130 | L 1024 S 1 float32 train | 0.4731 | 0.4729 | 2.0347 | 3.9444 | 1.000 (was 1.087) |
| `SWAAtomTransformer` | 130 | L 2048 S 1 float32 train | 0.5130 | 0.5325 | 3.3514 | 4.5149 | 0.963 (was 1.044) |
| `SWAAtomTransformer` | 130 | L 1024 - 4096 S 1 master train | refused | 0.385 - 0.526 | | | (cuda 0.485 - 0.588 measured) |
| `SWAAtomTransformer` | 130 | L 7168 S 1 master train | 0.7116 | 0.7721 | 6.3886 | 7.1459 | 0.922 (was refused) |
| `SWAAtomTransformer` | 130 | L 8192 S 1 master train | 0.7464 | 0.7219 | 7.4240 | 7.4902 | 1.034 (was 1.132) |
| `SWAAtomTransformer` | 132 | L 1024 S 5 master eval (global, FA4) | 0.0716 | 0.0981 | 0.3440 | 0.1064 | 0.729 (was refused) |
| `SWAAtomTransformer` | 132 | L 8192 S 5 master eval (global, FA4) | 0.6798 | 1.2255 | 6.4950 | 1.7847 | 0.555 (was 1.052 ms) |
| `SWAAtomTransformer` | 132 | L 4096 S 48 master train (global, FA4) | 6.374 | 10.698 | | | 0.596 (was 10.098 ms) |

Over every measured SWA case (`spd_swa_final.jsonl`, lengths 1024 - 16384, all samples): ESMFold2 x triton 1.33 eval / 1.22 train
(geomean), SWAAtomTransformer 1.70 / 1.47; against the wiring round's cuda 1.00 - 1.16 x (the global-attention row with FA4 up to 1.71 x).

### Ladder after this round (my classes, against the B200 tokens' triton / pytorch)

- `AtomDiTProtenix`, `AtomDiTOpenFold3`, `AtomDiTBoltz2`, `AtomDiTRoseTTAFold3`, `AtomDiTChai1`: 0 cuda < triton violations (was 0;
  OpenFold3's small-L `mask_form=af3` L 128 case 1.072 is the small-L exception, on the unchanged sm_80 path).
- `AtomDiTESMFold2`: 4 -> 1 (row 119 float32 inference L 1024 S 1, 1.04: 0.0531 against 0.0511; Anthropic level with triton there).
- `SWAAtomTransformer`: 3 -> 1 (row 130 master training L 8192 S 1, 1.034).
- triton < pytorch: none for these classes.
- `InputFeatureEmbedder` (the transition agent's class) runs SWAAtomTransformer as its child: its float32 S 1 training cases gain what row
  130 gained; its bf16 S 1 cases under 7168 rows stay refused with the child.

## Gap analysis after the optimization round

| class | row | case | why | what it would take |
|---|---|---|---|---|
| `AtomDiTESMFold2` | 270 | master train L 1024 - 4096 S 1 (refused) | one sample: every sm_100a backward launch is 8 - 32 CTAs (64 / 128-row tiles), latency-bound; after PDL + streams 0.164 / 0.170 / 0.195 against triton 0.135 / 0.151 / 0.176 ms. The heaviest: `qkvg_bwd` 22 us, `ffn_bwd_dy` 18, `oproj_bwd` 14, `ffn_bwd_gate` 13 (L 1024) | 32-row-tile variants of the backward row kernels (N = 32 UMMA, as `qkvg_fwd2_a1` / `ffn_fwd2_a1` in the forward) or a split of each tile's output columns over 2 - 4 CTAs; then lower `_B200_BF16_MIN_ROWS` |
| `SWAAtomTransformer` | 130 | master train L 1024 - 4096 S 1 (refused); L 8192 S 1 served at 1.034 x | as above (3 blocks) | as above |
| `AtomDiTESMFold2` | 119 | float32 eval L 1024 S 1, 1.04 x triton / Anthropic | fp32 inference of one sample: launch-bound (modulation, three TF32 stages) | fold the modulation into the qkvg stage at A = 1, or PDL across the TF32 forward |
| `AtomDiTOpenFold3` | 117 / 169 | `slide_qblock` (IntelliFold2) | the tcgen05 slid window implements AlphaFold 3's rule only | a `qblock` variant of `slide.cuh` (rule against R rounded to a query window); the sm_80 path serves it now |
| `AtomDiTOpenFold3` | 118 / 170 | `mask_form="af3"` (L 128) | `local_dit_b200.reason` refuses the form (sm_80 path) | the widened key mask is a prefix: the tcgen05 path could take it as a key mask |
| `AtomDiTBoltz2`, `AtomDiTRoseTTAFold3`, `AtomDiTChai1` | all | **closed in round 3 (above)** -- was: row kernels on the sm_80 `mma.sync` builds + cuBLAS; the tcgen05 attention core only for "slide" and training | their options (up-gate transition, flat q / k LayerNorm, identity AdaLN, no out projection, parallel schedule) are not in the tcgen05 row kernels | tcgen05 row kernels with those options (the H100 `sm90_atom` family has them as `-D` builds: UP, KQLN, IDENT, NOOUT, PARS) |
| `AtomDiTESMFold2Pair` | all | sm_80 path built for sm_100a (unchanged, 6 - 19 x triton) | not ported: the branch's ESMFold2 tcgen05 kernels (`lattn_esm*`, `lesm`) were left out of `b200_atom_local` | port `_esm_block` / `_EsmBlock` of the branch's `local_dit.py` with those sources |
| all local classes | "slide" | padding query windows past the last real window (none in the requirement cases: the masks pad < 32 atoms) | the tcgen05 slide keeps such windows on their clipped keys; the reference (and the A100 / H100 paths, `edge.cu` tail kernels) read the slid keys `[R - 128, R)` | call `edge.cu`'s `tail_attn_fwd` / `tail_bwd` around the tcgen05 attention, as `dit_variants/cuda/train.py` does for the variants |
| `SWAAtomTransformer` | 132 | float32 global, YELLOW (training 1.7e-3 - 2.8e-3, within 5 x tol; the H100 token is YELLOW too) | TF32 row stages + bf16 global attention (FA4); the backward's head-wise tail is torch elementwise | an fp32 / TF32 global attention, and the tail as a kernel |

FlashAttention-4 is needed in the B200 env for the row 132 numbers above; without it the route behaves as the wiring round (FA2
fallback, the global bf16 inference refused below 8192 rows).

## Changes outside sm_100a-only code (this round)

- `kernels/dit_variants/cuda/train.py` (shared with A100 / RTX 5090): `_b200_core`, `_AtomPairBiasB200`, a `core` flag through
  `atom_fwd` / `atom_bwd` / `_AtomBlock` (default False). Off a B200 `_b200_core` is False and every statement runs as before (the
  slid-window branches were split as `slide and not core` / `slide`, same order of launches for `core` False).
- `kernels/swa_dit/autograd.py` `SWADiTBlockFunction`: a B200-only branch (capability 10.0, bf16 with fp32 masters) packs the weights
  by `sm100.master_pack`. `kernels/swa_dit/cuda/sm100/__init__.py`: `attn_global_bwd` passes `deterministic=True` to FA4 only on a B200
  (H100 calls unchanged); the rest of that module and `tf32_bwd.py` are sm_100a-only. The bf16 backward `.cu` files of
  `kernels/swa_dit/cuda/sm100` (PDL waits) are compiled for sm_100a only.
- `modules/atom_dit.py`: `_LocalAtomDiT._b200_tc` (B200 only), `_SWATrainB200`, `AtomDiTMiniWorld._b200_tf32_train` / `_b200_reason`
  thresholds, versions.
- `tests/cpu/test_capture_api.py`: the new module-level dicts listed (`b200_atom._TMAPS32`, `b200_atom_local._TMAPS`,
  `local_dit_b200._ZEROS32`).
- New B200-only files: `kernels/augmented_attention/cuda/b200_atom/*`, `.../b200_atom_local/*`, `kernels/integrations/local_dit_b200.py`.
  Main's `sm100_atom` / `sm100_atom_local` (their old B200 halves are now unused by any route; the H100 halves are) are untouched.

## Wiring round (2026-11-16): how the B200 was routed

**Block-local AF3 atom blocks (Protenix, OpenFold3, Boltz-2, RoseTTAFold3, Chai-1, ESMFold2Pair).** The A100 kernels of these blocks
(`mma.sync` / `ldmatrix` / `cp.async`, the `lattn90*.cu` window attention with its dependent-launch statements emptied, the
`train_rows.cu` / `edge.cu` row and edge kernels, `sm80_atom_rows`, `sm80_atom_local`) run unchanged on sm_100a, which keeps the
warp-level tensor-core instructions and the 227 KB of shared memory per block of the H100. They are built for sm_100a by the one
retargeting hook the RTX 5090 port introduced: `kernels/_sm120.py` now also answers `building_sm100()` (capability 10.0, or
`MINIWORLD_ENGINE_BUILD_ARCH=sm100` for a GPU-less prebuild) and compiles every `sm80_load_extension` / `sm80_cubin_flags` build as
`<name>_sm100` with `-gencode=arch=compute_100a,code=sm_100a` and `-DMWE_SM100` (no source reads that define). Cards other than the
B200 build exactly what they built before. Gates opened for capability 10.0: `modules/atom_dit._local_sm80`, `modules/dit_variant_cuda._atom_sm80`
(the token variants keep `_a100`), `kernels/dit_variants/cuda/train.atom_serves`, `kernels/integrations/local_dit_sm80_block.serves`
(the A100's 128-160 KB weight-resident row kernels fit the B200), `kernels/integrations/local_dit_esm._a100`.

Everything the A100 route serves, the B200 serves: bf16 / master and fp32 (TF32), inference and training, the clipped (`pad`), slid
(`slide`) and `slide_qblock` key windows, AF3's mask form, Chai-1's conformer mask and its fp32-entry streams (`MasterEntryCast`), the
shared conditioning (stride 0 over the samples) and the multi-fold streams (one call per fold).

**SWA atom blocks (ESMFold2, SWAAtomTransformer).** `AtomDiTMiniWorld._b200_reason` / `_forward_b200`: the dispatch's sm_100a stages
with `portable` off (`swa_dit_block_fwd` for inference, `SWADiTBlockFunction` with a gradient), the modulation `silu(c) W^T` on the
sm_100a `mod_fwd` / `mod_bwd` (bf16) or `mod_fwd_tf32` / `tf32_bwd.mod_bwd_tf32` (fp32) (`_SWAModulationB200`). Changes in
`kernels/swa_dit` (B200 only):

- fp32 training: the TF32 backward rebuilt on 2026-10-07 / 08 on `origin/b200/ladder-final` (`cuda/sm100/tf32_bwd.py` and
  `adaln_bwd_tf32`, `attn_dq_tf32`, `attn_dkv_tf32`, `dw_tf32`, `ffn_bwd_gate_tf32`, `mod_bwd_tf32`, `oproj_bwd_fused_tf32`,
  `wprep_bwd_tf32` `.cu`) with the TF32 forward that writes its saves (`tf32_fwd.py`, `qkvg_fwd_tf32.cu`, `attn_fwd_tf32.cu`,
  `ffn_fwd_tf32.cu` from the same branch); `dispatch._tf32_bwd_ready` is now true and `_swa_dit_block_bwd` sends fp32 heads to
  `tf32_bwd.block_bwd_tf32`. Every gradient column is written by one kernel (no float atomics): two identical training steps give
  bitwise-equal gradients (checked on the B200 for ESMFold2 L 1024 S 48 and SWAAtomTransformer L 2048 S 48, fp32 and master).
- global attention (`swa_window_size=1000000`) without FlashAttention-4 installed (the B200 environment has none):
  `cuda/sm100.attn_global_fwd` / `_bwd` fall back to PyTorch's bundled FlashAttention-2 (`aten._flash_attention_forward` /
  `_backward`, varlen with the padding rows as key-less sequences, as the A100 block does). B200 only (`_fa2_fallback`).
- fp32 global attention: `tf32_fwd.block_fwd_tf32` with a global half window runs the TF32 row stages around the bf16 global
  attention (`supported_tf32_global`, `dispatch._swa_dit_block_fwd`, saves included); its backward `tf32_bwd._global_attn_bwd` takes
  the bf16 global attention's backward (`attn_global_bwd`) and runs the head-wise tail of the qkvg backward (q / k RMSNorm + rotary
  backward, which the TF32 window kernels fuse into their epilogues) as fp32 elementwise torch (`_head_tail`). Deterministic (FA2's
  backward included: two identical steps bitwise equal at L 1024 S 48, fp32 and bf16).
- the bf16 backward's five weight gradients on the split-K `dw_bf16.cu` kernel of the branch (69b49d498; fixed-order reduction) in
  place of four cuBLAS GEMMs.
- bf16 training and bf16 global-attention inference with fewer than 8192 rows (`N S`) are refused by the B200 route
  (`_B200_BF16_MIN_ROWS`): measured slower than the Triton stages there (see Speed).

## Measured in the wiring round (2026-11-16)

Records: `$W/b2511/work/atom-dit/` on the B200 (`W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950`): accuracy `acc_full_local.jsonl`
(Protenix, OpenFold3, Boltz-2, RoseTTAFold3, Chai-1), `acc_full_swa.jsonl` (ESMFold2, ESMFold2Pair, SWAAtomTransformer),
`acc_132.jsonl` / `acc_132t.jsonl` (row 132 after the fp32 global changes), `acc_swa_train.jsonl` (SWA training after `dw_bf16`);
speed `spd_full_<class>.jsonl` (cuda / triton / pytorch, every row, lengths 1024 / 4096 / 8192 / 16384, min samples, CUDA-graph
timing, `dev.sh -x`), `spd_probe.jsonl` (lengths 2048 / 8192, all samples); the PyTorch / Anthropic baselines of the same box in
`$W/b2511/out_base/speed.*.jsonl`. Accuracy: every requirement case at min samples (the canonical accuracy run: lengths all, depths
min, samples min), backends pytorch / cuda / triton.

### Accuracy (lights of the `"cuda"` backend; `"triton"` in brackets)

| class | GREEN | YELLOW | RED | unsupported | skipped |
|---|---|---|---|---|---|
| `AtomDiTProtenix` | 61 (74) | 71 (58) | 0 (0) | 0 (0) | 0 (0) |
| `AtomDiTOpenFold3` | 59 (53) | 51 (57) | 0 (0) | 0 (0) | 0 (0) |
| `AtomDiTBoltz2` | 62 (58) | 50 (54) | 0 (0) | 0 (0) | 0 (0) |
| `AtomDiTRoseTTAFold3` | 60 (67) | 52 (45) | 0 (0) | 0 (0) | 0 (0) |
| `AtomDiTChai1` | 63 (66) | 89 (86) | 0 (0) | 0 (0) | 0 (0) |
| `AtomDiTESMFold2Pair` | 32 (29) | 24 (27) | 0 (0) | 0 (0) | 0 (0) |
| `AtomDiTESMFold2` | 16 (28) | 20 (8) | 0 (0) | 0 (0) | 0 (0) |
| `SWAAtomTransformer` | 41 (56) | 42 (27) | 0 (0) | 0 (0) | 13 (13) |

The 13 skipped SWAAtomTransformer cases are training at L 5120-8192 with 48 samples, where the fp64 reference ran out of memory
next to the other jobs on the card (every backend skipped alike). Row 132 float32 training (global attention, served last): YELLOW
at L 1024-4096 as the Triton backend (1.65e-3 - 2.55e-3 against Triton's 1.69e-3 - 2.37e-3), L 5120+ skipped by the reference.
The bf16 cases the row threshold now refuses (rows 130 / 270 master training at L 1024-7168 with S 1, row 132 master inference
L 1024 S 5: 7 ESMFold2 and 8 SWAAtomTransformer cases) were measured before it and are counted here as measured (none RED); on the
final route they are unsupported (`acc_thr.jsonl` checks the refusal).

Run-to-run determinism (two identical training steps, `det_swa.py`): bitwise-equal outputs and gradients for ESMFold2 L 1024 S 48,
SWAAtomTransformer L 2048 S 48 (float32 on the TF32 kernels, bf16 / master), Protenix L 1024 S 48 (float32, bf16 activations) and Boltz-2 L 1024 S 8 (float32).

No case is RED. YELLOW (within 5 x the tolerance) is at the rate of the `"triton"` backend on the B200 and of the same kernels on the
A100 (A100 tokens, cuda: Protenix 68 GREEN / 76 YELLOW, Chai-1 75 / 85, OpenFold3 66 / 48).

### Speed

`"cuda"` speedup over the other backends per class and mode (median of 50 CUDA-graph replays per case, L2 flushed; ratio = other /
cuda; Anthropic only where its kit serves the class):

| class | mode | cases | x triton: geomean (min - max) | x pytorch: geomean (min - max) | x anthropic: geomean (min - max), cases |
|---|---|---|---|---|---|
| `AtomDiTProtenix` | eval | 34 | 4.57 (1.55 - 14.52) | 5.65 (2.01 - 15.82) | 4.53 (1.63 - 14.50), 34 |
| `AtomDiTProtenix` | train | 18 | 2.41 (1.30 - 4.36) | 3.07 (1.62 - 5.75) | - |
| `AtomDiTOpenFold3` | eval | 28 | 3.58 (1.55 - 7.70) | 4.63 (1.99 - 10.95) | 3.48 (1.67 - 7.52), 28 |
| `AtomDiTOpenFold3` | train | 12 | 2.03 (1.30 - 2.83) | 2.38 (1.64 - 3.18) | - |
| `AtomDiTBoltz2` | eval | 32 | 4.23 (1.47 - 9.70) | 5.27 (1.94 - 10.61) | 4.63 (2.08 - 9.45), 16 |
| `AtomDiTBoltz2` | train | 12 | 2.18 (1.22 - 3.35) | 2.48 (1.55 - 3.59) | - |
| `AtomDiTRoseTTAFold3` | eval | 32 | 4.39 (1.54 - 9.83) | 5.45 (1.99 - 11.12) | - |
| `AtomDiTRoseTTAFold3` | train | 12 | 2.15 (1.18 - 3.33) | 2.43 (1.50 - 3.58) | - |
| `AtomDiTChai1` | eval | 48 | 4.31 (1.24 - 11.37) | 5.37 (1.74 - 12.77) | - |
| `AtomDiTChai1` | train | 12 | 2.42 (1.15 - 4.35) | 2.81 (1.54 - 4.67) | - |
| `AtomDiTESMFold2Pair` | eval | 16 | 9.16 (4.38 - 19.23) | 11.85 (5.54 - 21.94) | - |
| `AtomDiTESMFold2Pair` | train | 6 | 6.36 (4.29 - 9.01) | 7.32 (5.13 - 10.23) | - |
| `AtomDiTESMFold2` | eval | 8 | 1.54 (1.17 - 2.83) | 20.19 (5.01 - 118.90) | 1.33 (0.96 - 1.86), 8 |
| `AtomDiTESMFold2` | train | 6 | 1.09 (0.74 - 1.67) | 7.16 (2.44 - 30.33) | 10.26 (7.61 - 19.03), 6 |
| `SWAAtomTransformer` | eval | 18 | 1.61 (0.93 - 2.82) | 12.93 (3.26 - 68.75) | 1.50 (1.00 - 1.95), 18 |
| `SWAAtomTransformer` | train | 18 | 1.34 (0.76 - 2.04) | 7.16 (2.36 - 46.98) | 7.47 (2.86 - 18.87), 18 |

Five cases were slower than Triton, all bf16 SWA blocks on few rows, and are now refused by the B200 route (`_B200_BF16_MIN_ROWS`
= 8192 rows `N S`): master training with one sample (ESMFold2 row 270 L 1024 0.205 / 0.152 ms, L 4096 0.225 / 0.200;
SWAAtomTransformer row 130 L 1024 0.588 / 0.446, L 4096 0.668 / 0.597) and master global-attention inference row 132 L 1024 S 5
(0.106 / 0.099). At A = 1 every row tile of the sm_100a backward is 128 atoms of one sample, so L 1024 is 8 CTAs on 148 SMs; a
profile at L 2048 shows the backward kernels (qkvg_bwd 22 us, ffn_bwd_dy 19, oproj_bwd 14, ffn_bwd_gate 13, dq + dkv 19), the
modulation backward (`mod_bwd` 24) and the split-K weight gradients (21, 8 CTAs) each latency-bound. From 8192 rows the route wins
(L 8192 S 1: 1.07 / 1.07; every S 48 case 1.16 - 2.04 x). Against Anthropic one case is behind: ESMFold2 float32 L 1024 S 1 0.053
against 0.051 ms (its band core; every other ESMFold2 / SWA case is ahead or level).

Representative absolute times (ms; cuda / triton / pytorch): Protenix master inference L 8192 S 5 0.295 / 2.086 / 2.700, master
training L 8192 S 48 5.784 / 23.148 / 33.224, float32 training L 8192 S 48 14.115 / 22.757 / 31.319; Chai-1 master inference L 8192
S 5 0.305 / 1.819 / 2.432; ESMFold2Pair master training L 8192 S 1 0.490 / 4.426 / 5.002; SWAAtomTransformer (3 blocks) master
training L 8192 S 48 6.041 / 8.237 / 103.179, float32 training L 8192 S 48 7.131 / 14.527 / 334.976.

## Gap analysis of the wiring round (2026-11-16; the current state is in the optimization round above)

Every requirement case of the five block-local classes and of `AtomDiTESMFold2Pair` is served. Not served (refused, by speed):

| class | row | case | why | what it would take |
|---|---|---|---|---|
| `AtomDiTESMFold2` | 270 | master, train, L 1024-7168, S 1 (7 cases) | slower than the Triton stages (L 1024 0.74 x, L 4096 0.89 x): A = 1 gives the sm_100a backward one 128-atom tile per CTA, 8-56 CTAs | backward kernels with smaller row tiles at A = 1 (32-atom tiles as `qkvg_fwd2_a1` / `ffn_fwd2_a1` already do in the forward), the modulation and weight gradients on cuBLAS below a row count; then lower `_B200_BF16_MIN_ROWS` |
| `SWAAtomTransformer` | 130 | master, train, L 1024-7168, S 1 (7 cases) | as ESMFold2 (L 1024 0.76 x, L 4096 0.89 x) | as above |
| `SWAAtomTransformer` | 132 (global attention) | master, eval, L 1024, S 5 (5120 rows) | 0.93 x Triton: FlashAttention-2 (no FA4 installed) and its varlen setup (a few small torch ops) at a small size | FlashAttention-4 installed in the environment, or the varlen tables made by one kernel |

Served but with a known weakness:

- `SWAAtomTransformer` row 132 float32 (global attention): the TF32 row stages around the bf16 global attention (FlashAttention-2 here:
  no FlashAttention-4 in the B200 environment); the backward's head-wise RMSNorm + rotary tail is elementwise torch (fp32), not a
  kernel. Inference YELLOW in 5 of 8 lengths (at most 1.24 x the tolerance) where the Triton backend (fp32 row stages, bf16 attention)
  is GREEN: the TF32 rounding of the rows adds to the bf16 attention operands. A TF32 global attention kernel (fp32 operands) would
  close it. The row is engine-own (legacy r109, no FoldForge / MiniWorld constructor builds it).
- Speed lever not taken (every served case beats Triton and PyTorch): the five block-local classes run the A100's
  `mma.sync` kernels and cuBLAS (`dit_variants` rows, `local_dit_sm80_block`), not the B200's own tcgen05 local DiT
  (`origin/b200/ladder-final`: `integrations/local_dit.py`, `kernels/augmented_attention/cuda/sm100_atom/` row kernels, the
  `sm100_atom_local/lattn_{fwd,dq,dkv}*.cu` window attention with the slid window inside (`slide.cuh`, `lslide.cu`), the hoisted
  modulation tables). The branch measured the AF3 cross block (Protenix's chained K / V AdaLN) at A 48 N 8192 training in 2.98 ms
  bf16 / 7.44 ms fp32 against 5.78 / 14.11 ms here (Protenix row 267, L 8192, S 48), i.e. about 2x on training; inference A 5 N 4096
  0.061 ms bf16 (static weights and inputs) against 0.139 ms (L 2048) and 0.295 ms (L 8192) here (A 5, master, row 114). Porting it means: the branch's newer `sm100_atom` /
  `sm100_atom_local` sources (main's copies are older and also carry the H100 halves under `-DATOM_HOPPER=1`, which must stay
  byte-identical for the H100 build, so the sm_100a-only kernels would go to their own files), a B200 `run_block` over the module's
  parameter names (`kernels/integrations/local_dit.WEIGHTS` / `CROSS_WEIGHTS` are the branch's layout), and, for OpenFold3 / Boltz-2 /
  RoseTTAFold3 / Chai-1 whose block options (parallel K / V AdaLN, identity AdaLN, flat q / k LayerNorm, up-gated transition) the
  tcgen05 row kernels do not have, the branch's `dit_variants/cuda/sm100.py` route: the `dit_variants` rows around the tcgen05
  window attention and pair bias (only the attention core swapped).
- B200 prebuild: the sm_100a builds of the sm_80 sources compile on first use (`dit_variants_train_rows_sm80_sm100` ~60 s,
  `af3_local_rows_sm80_sm100`, the `atom_local_mma_sm80_sm100/*` cubins); no prebuild plan enumerates a `b200_sm100` target for the
  sm_80 builders (`kernels/_prebuild.py` knows sm80 / sm90 / sm120). The keys are those of `MINIWORLD_ENGINE_BUILD_ARCH=sm100`.
- Not a requirement case, the same on A100 / RTX 5090: a bf16 activation with fp32 parameters called WITHOUT autocast on the
  `dit_variants` path (Boltz-2 / RoseTTAFold3 / Chai-1 / OpenFold3) passes `_supports_cuda` and fails in `_atom_tables` (a bf16 x fp32
  `F.linear`); the master rows always run under bf16 autocast.

## Changes outside sm_100a-only code (wiring round)

- `kernels/_sm120.py`: `building_sm100()`, `_target()`; `_retarget(flags, arch)` takes the arch; `sm80_load_extension` /
  `sm80_cubin_flags` retarget to sm_100a on a B200. On every other card (and under `MINIWORLD_ENGINE_BUILD_ARCH=sm80 / sm90 /
  sm120`) the flags, names and keys are what they were.
- `modules/atom_dit.py`: `_local_sm80` (the `_LocalAtomDiT` gates; `_a100` itself unchanged, so the SWA classes do not take the sm_80
  SWA path on the B200), the B200 SWA route, `backend_versions["cuda:b200_sm100"]` entries.
- `modules/dit_variant_cuda.py`: `_atom_sm80`, `_card`, `_call_reason` (`a100=True` callers only), `_atom_serving`.
- `kernels/dit_variants/cuda/train.py` `atom_serves`, `kernels/integrations/local_dit_sm80_block.py` `serves`,
  `kernels/integrations/local_dit_esm.py` `_a100`: capability (10, 0) added.
- `kernels/swa_dit/dispatch.py`: `_tf32_bwd_ready` true (the TF32 path is B200-only: `tf32_fwd.is_b200`), the fp32-heads branches of
  `_swa_dit_block_bwd` (window and global), the fp32 global branch of `_swa_dit_block_fwd` (behind `tf32_fwd.supported_tf32_global`,
  B200 only). `kernels/swa_dit/cuda/sm100/__init__.py` (sm_100a-only module): the FA2 fallback of `attn_global_fwd` / `_bwd`
  (B200 only; H100 calls them only with FA4 installed), `DwBf16` in `block_bwd`. `tf32_fwd.py` / `tf32_bwd.py` and the `*_tf32.cu`,
  `dw_bf16.cu` sources: sm_100a only. H100 / A100 / RTX 5090 take none of these branches and build none of these sources.
- No `.cu` / `.cuh` source compiled for another card was changed (the sm_80 sources are only built a second time, for sm_100a).
- `tests/cpu/test_judge.py::test_card_of` asserts `set(judge.GPUS)` without `b200_sm100` (the base branch added the card): not
  changed here.

## N2 round (2026-11-18, agent n2-ife): the embedder's step glue, the weight-gradient GEMMs, the inference tail

`backend_versions["cuda:b200_sm100"]`: `InputFeatureEmbedder` 30 -> 32 (31 training, 32 inference), `SWAAtomTransformer` 27 -> 29 and the
`AtomDiTESMFold2`-class counter 27 -> 29, `AtomDiTMiniWorld` 19 -> 21 (the modulation weight gradient: training). Base: the merged tree (`n2-base`).
Tools: `n1cfg/ifeab.py` (same-process A/B of one module attribute, 3 interleaved rounds, bitwise compare, determinism), `ifeprof.py` + `trace.py`
(captured-graph kernel timeline), `gbb.py` / `gfa.py` (the attention kernels alone). Records and probes: `$W/b2511/work/n2-ife/`.

### Timeline first (L 4096 master global training, esmfold2 relu, graph, L2 flushed; us)
Step ~670: forward ~245 (prelude 44 before the first qkvg, 3 x 65 per block), backward ~430 (head 33, 3 x ~130 per block, tail ~59 after the last
`gtail`). Per block backward: `gattn_bwd` 64 | gtail 4 | dx GEMM 7 - 14 | adaLN 4 | ffn gate 15 | adaLN | oproj 14 - 19 | gprep 6 - 13; the side streams
carry dW (17), dWmod (cutlass sm80-style GEMM on 12 CTAs, 11 - 20), dc GEMM + reduce + dsilu (12 + 4 + 4), silu(c). Skip probes (wrong gradients, timing
only; `SWA_PROBE_SKIP`): no dWmod GEMM -22 us, no dc GEMM -15.5 us, both -28 us of the step: the two GEMM families cost ~4 % of the step through the tail and
through the chain stages they run beside. The tail after the last `gtail`: dx 7, adaLN 4, dc 12 + 4 + 4, featbwd 10.6, features dW GEMM 11 + 3 (cutlass
`s1688`, an align-1 / 2-CTA kernel at nf = 133 whatever the K split: 8.8 - 12.5 us alone in `mb1.py`), LayerNorm column sums 11.

### Built and kept (flag gated, A/B in one process, `ifeab.py`; bitwise unless stated)
| change | where | measured |
|---|---|---|
| features' weight gradient `dW = DP^T F` as one structured kernel (`swa_embed_dwf_sm100`: the 5 dense feature columns by row chunks over a cluster of 8, DSMEM sum in rank order; the 128 element one-hot columns by a block-ordered match list, 8 loads in flight; fixed order; `featbwd` writes no F) | `embed_glue.cu`, `embedder._FeatB200`, `_FEAT_DWK` | L 4096 esm 669.9 -> 662.6, legacy 667.5 -> 659.5; vs the GEMM: dW rel diff 3e-7 (`dwtest.py`, 7 shapes, random non-zero data, ragged mask, elements -3 .. 130), the other gradients bitwise, deterministic |
| modulation weight gradient dWmod = dmod^T silu(c) on the cluster dW kernel's one-problem build (`dw_tf32.cu -DDWM=1`: six 128-row tiles of 8 CTAs, partials through the global workspace, rank-order sum) in place of a cuBLAS GEMM, K = rows | `dw_tf32.cu`, `tf32_fwd.py` cubin `dw_tf32_cl8_r32g_dwm`, `tf32_bwd.DwModTf32`, `DWM_KERNEL` | L 4096 659.0 -> 647.0 (-12), L 8192 1409 -> 1389 (-20); operands truncated by the MMA like the other five weight gradients: error vs fp64 7.6e-4 against 2.9e-4 for cuBLAS TF32 (`dwmtest.py`, M 256 - 16384), deterministic |
| the RoPE launch on a side stream beside the features (training) | `embedder._forward_b200`, `_ROPE_SIDE` | L 4096 663.6 -> 658.0 |
| the read-out scatter on one thread a group of 8 atoms (`swa_embed_scatter8_sm100`; the run's loads in chunks of 8: the per-atom load chain was ~11 us) | `embed_glue.cu`, `_SCATTER8` | kernel 11.1 -> 8.2 us at 6144 atoms (`scmb.py`, bitwise equal on contiguous / ragged / long runs, bf16 / fp32); training L 4096 660.0 -> 657.8, eval L 6144 -2 us |
| inference read-out: the terminal norm on `swa_embed_outnorm_sm100` (it also zeroes the scatter's accumulator: one launch less), the master read-out weight's bf16 form cached (`eval_form`: no cast launch) | `embedder._forward_b200`, `_EVAL_OUTNORM`, `_EVAL_RW` | eval L 6144: float32 legacy 356.4 -> 354.5, master legacy 305.2 -> 303.1 -> 292.9 with the two scatter / cast items |
| silu(c) of the dWmod made once per stack | `tf32_bwd.SILU_ONCE` | L 4096 -0.8 (level) |
| `embed_colsum` loads unrolled 16 (was 4) | `embed_glue.cu` | in-step kernel 11 -> 7.5 us (off the tail now: it runs beside `dwf`) |

### Measured and not taken (diffs in `docs/b200/rejected/`)
* the TF32 route's weight forms (wprep + Wmod rounding) made at the start of the step on the side stream (`atom_dit._b200_prefetch_tf32_forms`, `_B200_EARLY_FORMS`) and block 0's modulation on the current stream (`_B200_MOD0_MAIN`): L 4096 -2.1 / -0.9 us but L 3072 +8, L 6144 +4.5 / +8, L 7168 +8 / +6, L 8192 +5 / +9.5 (graph A/B): left in the source, OFF (`SWA_EARLYFORMS` / `SWA_MOD0_MAIN` = 1 turns them on).
* features' dW as K-split `bmm` (4 / 8 / 16 / 32 chunks) and F padded to 136 floats: 671.9 vs 688.4 / 676.0 / 673.9 / 674.0 (the cutlass `s1688` kernel takes 8.8 - 12.5 us whatever K, `mb1.py`): `ife_n2_feat_dw_ksplit_bmm_pad.diff`.
* `DcgTf32` (early GEMM beside the attention backward + late launch) for global attention: +30 us (every block), +12 us (last block only): `gattn_bwd` waits for the SMs the early launch holds; `swa_n2_dcg_global.diff`.
* `DcgTf32.late` as the whole K = 6C GEMM + silu' + stack sum in one launch (the dc chain 12 + 4 + 4 -> one launch): not deterministic (3 replays differ) and +28 us (every block's dc then on the chain): `swa_n2_dcg_full_k6c.diff`.
* `DcxTf32` (dx -> adaLN -> dc -> dsilu cluster kernel) on two waves of clusters for M 4096 (`DCX_WAVES`; the tiles are independent): +86 us (every block), +43 us (last block only): `swa_n2_dcx_waves_m4096.diff`.
* flag sweep at L 4096 (all level, +-2 us): DWM_SPLIT, DCG, DXP, DWS, KSPLIT_DX 1/2/4, KSPLIT_DC 1/2/3/4, EARLY_DC, DCX (`ok` fails: 32 tiles x 8 CTAs > 148 SMs).
* `GattnFwd.splits` gain 1.15 for the single-round long-loop case (N 1 S 8192: 128 items of 148 SMs): the 102.4 -> 96.3 us seen once was the 60 / 67-us state of the next item (below), not the build; reverted.
* `ffn_fwd_tf32` build flags (L 4096 global, 659.6 base): MMA_INORDER=1 659.2, PL=4 658.0, PL=1 665.1. The FFN stage has one 64-row tile per CTA pair at these lengths, so there is no next tile to overlap with; the hidden chunks already double-buffer the a | b accumulator (header of `ffn_fwd_tf32.cu`).

### The task's five items
(a) `gattn_bwd` on 148 SMs: the even cut of the (item, query block) sequence over all 148 CTAs is the existing split build (`gattn_bwd_s`; equal contiguous ranges = LPT at block granularity, dK / dV combined in CTA order, dQ adders ordered): alone, L 4096 (`gbb.py`) auto (cluster CS 2, 128 SMs) 65.9, split 89.1, whole 105.9 us -- the combine and the workspace traffic cost more than the 15 % of SMs gain. Two concurrent launches (CS 2 + CS 3 remainder) cannot lower the busiest CTA: all four heads (64 items) need CS >= 3 for 27.7 blocks, 192 CTAs > 148, and a head's items must be resident together; so the busiest CTA stays at 32 blocks. Not built: a persistent work-stealing queue (dQ order chain per head across pieces).
(b) deep-K GEMMs: built dWmod (K 4096, -12 us) and the features' dW (K 4096, -8 us) as above; the dc GEMM (K 768) as a hand kernel was built (`DcgTf32` K = 6C) and rejected; dx (K 512) not built.
(c) glue fusions built: features dW + F removal, eval read-out tail, scatter8, silu once, RoPE side stream, early forms. Not built: featbwd + dsilu merge (autograd Function boundary), tokgrad into the read-out GEMM.
(d) FFN forward: flags only (above); the premise (tile n + 1) does not exist at these row counts.
(e) `gattn_fwd` L 6144 eval (A/B inside one process, `gfb.py`): split build 69.6 us alone (whole 77.8). Variants inside one placement (gfb.py, default 59.5 - 59.8 us): PP=0 62.6, FPMOD 8 / FPCNT 1 61.6, FPMOD 4 / FPCNT 1 60.9, ST=4 59.9, QR=3 60.5: none better. Found: the same code alone measures 67 us (the first two objects of a process) or 60 us (the third and later) in `gfb.py` / `gfy2.py` (identical SASS, equal cubins, ws / cnt / O / L / seq placement, padded module copies, preloading dummy modules in the step and the flush state do NOT explain it; in the step the kernel stays at ~69 us). Unresolved; worth 7 us a block (21 us at L 6144) if a deterministic way to the 60-us state is found.

### Measured end to end (`measure.speed`, `sp_ife.yaml`: every IFE case, graph, L2 flushed; the median of 3 full runs per tree, `medmerge.py`; B200 GPU 2, `dev.sh -x`)
| | base (merged tree) | N2 |
|---|---|---|
| cases not >= 1.5x faster than the H100 `cuda` (144 compared, none exempt) | 26 (24 training, 2 eval) | 12 (11 training, 1 eval) |
| worst ratio / geomean (b200 / h100) | 0.730 / 0.565 | 0.726 / 0.552 |
| master training geomean new / base (48 cases), best / worst | | 0.969 (0.905 / 1.032) |
| float32 training, master eval, float32 eval (new / base) | | 0.974, 0.979, 0.993 |
| L 4096 master global esmfold2 relu training (step, ms) | 0.6756 (ratio 0.706) | 0.6508 (0.680) |
| L 4096 master global legacy relu | 0.6859 (0.730) | 0.6447 (0.687) |
| L 6144 master global legacy / esmfold2 linear / relu | 1.0506 / 1.068 / 1.0606 | 1.0255 / 1.0403 / 1.0187 |
| L 8192 master window 128 (esmfold2 linear / relu, legacy) | 0.6490 / 0.6409 / 0.6287 | 0.5867 / 0.5800 / 0.5698 (-9.5 %) |
| L 6144 master eval legacy global (ratio 0.673 -> 0.664) | 0.2968 | 0.2928 |
| L 6144 float32 eval legacy global (ratio 0.679 -> 0.671: still 3.4 us short of 0.667) | 0.3480 | 0.3441 - 0.3460 |
Still above 0.667 (12): master training global L 3072 (esmfold2 relu 0.703, linear 0.674), 4096 (legacy 0.687, esmfold2 relu 0.680, linear 0.667), 6144 (legacy 0.686, esmfold2 linear 0.682, relu 0.670), 8192 (0.721 / 0.726 esmfold2, legacy 0.672), float32 eval L 6144 legacy 0.674. The remaining step is the dependent chain: `gattn_bwd` 3 x 63 us (128 of 148 SMs, MMA / row math both ~1 - 1.5 us a 64-query block), the FFN forward tails 3 x 19, the dc GEMM chain on the tail (12 + 4 + 4), the dx GEMM + adaLN between blocks.

Found, not resolved: **L 8192 esmfold2 global training in a full `measure.speed` run** reads 1.464 / 1.473 ms (N2) against 1.421 / 1.428 (base) -- every run of the whole yaml, also with every N2 switch off (1.504) -- while the same case alone, with lengths [3072 ..], or with `ifeprof` / `ifeab` reads 1.38 - 1.39 (N2) against 1.418 (base): it needs [eval, train] over all eight lengths in one process (any shorter list, train only, preloading dummy modules, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` do not change it). The base tree does not show it on the same sequence. Same-process records differ from fresh-process ones by up to 6 % there; the ranking above uses the full-run medians.

Accuracy (`measure.accuracy`, `acc_ife3.yaml`: InputFeatureEmbedder L 3072 / 4096 / 6144, master / float32 / bfloat16, train + eval, both feature sets, window 128 / global, relu / linear; the measurer's random non-zero weights and inputs against the fp64 gold; 54 cuda cases): 28 GREEN, 26 YELLOW, 0 RED, the same counts and the same worst-tensor error (identical to 4 digits, 0 of 54 cases more than 2 % worse) as the base tree. `acc_swa.yaml` (SWAAtomTransformer / AtomDiTMiniWorld / AtomDiTESMFold2 master + float32 training L 3072): the same statuses and errors as the base. Kernel-level (`dwtest.py`, `dwmtest.py`, `scmb.py`, random non-zero data): features dW 3e-7 relative to the GEMM, dWmod 7.6e-4 against fp64 (cuBLAS TF32 2.9e-4), scatter8 bitwise equal to the scatter. Determinism: every `ifeab` A/B states `deterministic True` (three graph replays bitwise), and `dethash.py` gives the same sha1 over the outputs and every parameter gradient (eager step 1, step 2, graph replay) in two separate processes for esmfold2 global master training L 4096, legacy window-128 training L 4096 and master / float32 eval L 6144.

Ladder (`lad_ife.yaml` train + `lad_ife_e.yaml` eval, cuda / triton / pytorch, 144 cases with all three timed): 0 violations (cuda < triton < pytorch, 2 % margin). `_prebuild --arch sm100 --check`: 796 artifacts, 0 not built; the only absences are the pre-existing sm90 `augmented_attention_local_esm_*` / `dit_variants_esm_*` kernels and the mpnn requirement families (as in the N1 check).

Changes outside sm100-only code: `modules/embedder.py`, `modules/atom_dit.py` (the early-forms function, off; the three version counters), shared `driver.py` unchanged (a padded-image probe was reverted). No non-sm100 CUDA source changed.
Tools of this round: docs/b200/n2_tools/ (probe and measurement scripts; records in $W/b2511/work/n2-ife/n2out/).

### N2 addendum (20:10 UTC): the L 8192 full-yaml regression
Bisected with tree copies (base files swapped in): the cause is in the embedder-side files, and in the end in WHERE the side-stream pair is first created: `embedder._forward_b200` fetched `sm100._side_streams(...)[1]` in every call (inference too, at the first eval case). Creating it only when the RoPE side stream is used (`s2r = None` otherwise) removes the +3 % of L 8192 esmfold2 global training in the full `measure.speed` yaml (1.464 -> 1.38 - 1.40 ms, base 1.421). Another case then changed state: master legacy global L 4096 / 8192 read 0.672 / 1.438 in the full yaml against 0.645 / 1.376 before the change (base 0.686 / 1.433). The state is deterministic per code version (three runs agree), not random, and not the hardware connections (`CUDA_DEVICE_MAX_CONNECTIONS` 2 / 32: unchanged), the allocator (expandable segments), preloaded modules or any N2 switch. Full-yaml median of 3 runs, final tree: 15 of 144 cases above 0.667 (base 26), worst 0.715, no case slower than base by more than 1.7 % (all at L <= 2048); N2 vs base geomean: master training 0.976, float32 training 0.973, master / float32 inference 0.980 / 0.991. RoPE side stream off (`SWA_ROPE_SIDE=0`) gives 20 misses. The two cluster-split levers of the coordinator list (work-stealing `gattn_bwd`, tcgen05 dc / dx kernel, FFN-tail PDL) were not built in this window.

## N7 pass (2026-11-19, agent n7-ife): instruction-count rewrite of the global attention's softmax loops -- measured, nothing taken

Merged tree base, no code of the engine changed (`backend_versions` untouched). Variants: `rejected/n7_gattn_fwd_fpfast_combpf_trace.diff`
(`-DFPFAST`, `-DCOMBPF`, `-DTRCTA` + epilogue stamps; all default off). Tools in `$W/b2511/work/n7-ife/`: `gf8.py` (build variants alone, kernel
durations from the profiler = ns resolution, interleaved rounds, L2 flushed, `GAP=0.15` idle gap; the event timer of `gfa32.py` / `gf7.py` steps
in 2.05 us, too coarse for 3 %), `tr8.py` / `tr7.py` (TRACE timelines of any CTA), `det7.py`, `pr7.py`.

**Instruction counts** (SASS of `gattn_fwd32`, F32IO 2, per exponential pair of one thread): MUFU pair = FMUL2 (scale) + 2 MUFU.EX2 + F2FP.BF16
(pack) + FADD2 (row sum) = 4 issue slots + 2 MUFU (16 clk of the SM's 16/clk MUFU, 8 a warp op x 2); the shipped FMA-pipe pair (`fex2_poly2`
with the -126 clamp and the masked-entry selects) = FMUL2, 2 FMNMX, 2 FSETP, 3 FADD2, 3 FFMA2, 2 IMAD (exponent insert), CS2R + 2 predicated
IMAD.MOV (the select), F2FP, FADD2 = 18 slots. Already packed: scale, sum, poly are all f32x2; the pack is one F2FP per pair; no running max; so
the 4 slots / pair are the floor of this form (the only further saving would be folding SC into q, i.e. into other stages' bf16 rounding).
Backward per pair: FFMA2 (x = s SC + l), 2 MUFU, FADD2 (dP - D), FMUL2, 2 F2FP = 6 slots; its poly (`ex2_poly2`, no clamp) = 9.

**Rewrite tried (`FPFAST=1`)**: the unmasked blocks' poly without clamp / selects (11 slots a pair instead of 18; bitwise the same O and LSE as the
clamped form in range), the masked block keeps the old form; first as a per-pair branch (90.6 vs 69.6 us: the branch split the unrolled
loop -- scheduling lost), then as one `expcc(BoolC<UNM>)` lambda per block (no longer slower), swept with `FPMOD / FPCNT`:

| N 1 S 6144 forward, kernel us (profiler, median; GAP 0.15) | base | 1/8 poly | 1/4 | 1/3 | 1/2 |
|---|---|---|---|---|---|
| bf16 build (training forward), FPFAST | 67.1 | 70.8 | 68.4 | 66.6 | 72.2 |
| F32IO 2 build (eval float32), FPFAST (event timer, 2 us steps) | 69.6 | 75.8 | 71.7 | -- | 73.7 (2/8) |
| poly off entirely (F32IO `FPCNT=0`) | 71.7 (= base within the timer step) | | | | |

Backward (`gbv.py`, 1 x 4096, CS 2, graph median): shipped PCNT 1 of 4 65.8, PCNT 0 (no poly) 64.9, PCNT 2 67.8, PMOD 8 66.8, PMOD 2 66.6.
So with a cheaper poly the exponential split still does not move either kernel: neither is issue- or MUFU-bound.

**Why (TRACE, `tr8.py`, N 1 S 6144 split mode, CTA 0 / 20 / 40 / 60 / 80 / 100 / 120 / 147 all alike; clk of the SM)**: one block of one
warpgroup (64 keys x 128 rows) = 1390 clk steady period: S read 100, mask 50, turn wait 7, exponentials 966, P store + arrive 36, gap to
the next block 150-180. MUFU floor of a block pair = 1024 clk, so the loop runs at 74 % of it. Removing the exponentials (`NOEXP`) leaves a period
of 1067 clk (S read waits 505 for the MMA warp), removing the QK or the PV MMAs 57-59 us of the 69: the loop is a handshake chain (S ->
softmax -> P -> PV -> QK(G + 2) -> S) at ~1.07 k clk a block of both warpgroups; MUFU is a second ~1.0 k chain and the two overlap to 1.39 k.
Poly offload shortens the period (1299 clk at 1/4 in the trace) but the chain floor and the unchanged power cost keep the kernel level.
Outside the loop (the same CTAs, clk): prologue 3.5-4.0 k (kernel entry to first S), epilogue 15-17 k (7.7 - 8.7 us of the 66.7 us kernel; CTA 147: 5.7 k):
last PV + O read 1650, stcg + `__threadfence` 2060, counter + two named barriers 1200, **combine of 2 - 3 pieces 7500** (2.6 - 3.5 k a
piece, + 750 between pieces), TMA store wait 2000, exit 800. (Kernel 66.7 us against 103 k clk = 52 us at 1965 MHz: the SM clock under this load is ~1.55 GHz.)

**Combine prefetch (`COMBPF=2/3/4`)**: the pieces' partials loaded together, added in the same CTA order: bitwise the same O and LSE (all
four), but 84 / 88 / 96 us (bf16) and 78 / 123 / 133 us (F32IO) against 69.6: the split build has 151 registers, the 168-register cap
(384 threads) spills (STACK 24 / 184 B) and the main loop pays. Needs registers the epilogue does not have; a shared-memory staging of the
partials (the stage ring is not provably free while the other warpgroup's last PV runs) is the next step, not built: bound 2.5 - 5 k clk of 103 k (1 - 2.5 %) on split shapes.

Other knobs on the same shape (kernel alone, 69.6 base, event timer): `ST` 4 / 8 69.6 / 69.6, `PP=0` 73.7, `BNK=96` (bf16) 71.7, `NOMMA` 1 / 2 67.6 / 67.6.

**Second item (persistent LPT backward, two items a CTA, dK / dV staged in a workspace in fixed order): not built.** Per the earlier estimate
the ideal bound is 3.7 % of the L 4096 step (190 us of 672: 128 of 148 SMs busy); the TRACE above shows the backward's per-block chain, and the
probes of part 18 (no counter waits / no dQ traffic / no row math / no MMAs each level) say its time is that chain, not SM count; with the
window left after the forward analysis it was not started rather than started unverified.

Verification: no source of the engine differs from the merged base (`diff -rq n7-base n7-ife`: only this file and the rejected diff), so accuracy,
determinism, ladder and miss counts (IFE 14, SWAAtomTransformer 17, AtomDiTESMFold2 19 above 0.667) are those of the base. Variant checks:
FPFAST O / LSE vs MUFU form 1 bf16 step / 3 - 1e-4 (poly error), `det7.py` 30 launches bitwise equal for base and FPFAST; COMBPF bitwise equal to base.

### N7 addendum (00:40 UTC): the split combine through shared memory is taken (bf16 build); the schedule rewrite is not built

**Taken: `gattn_fwd.cu` `CMBS` (default 2 for bf16 builds, 0 for F32IO).** The split build's combiner (the piece that completes an item's counter)
no longer loads the other pieces' partials through registers piece by piece: all FOREIGN pieces of a group (CMBS + 1 consecutive CTAs of the item)
go by `cp.async.cg` (16 B, one 128 B row a thread, plus the 128 `l` by lanes 0 .. 31) into a shared-memory staging area (O_CMB, 144 B row pitch:
conflict-free `lds128`; 2 x (CMBS + 1) x 18.9 KB, 114 KB of the 232 KB), this CTA's own piece goes registers -> its slot (no L2 round trip), then the
partials are added in CTA order from shared memory: the same fp32 adds in the same order, so O and LSE are bit-identical (below). Zero extra
registers (COMBPF, which kept them in registers, spilled). Why shared memory and not just more loads in flight: TRACE of the combine: three pieces took
7.5 k clk (2.6 - 3.5 k a piece); with the staging 5.9 k at CMBS 2 and, with the own piece from registers, post-loop 15.4 k -> 12.4 k (CTA 0, N 1 S 6144;
F32IO CMBS 1: 16.4 k -> 11.1 k). All CTAs reach their combine in the same microsecond (equal ranges), so the partial traffic (~20 MB through L2) is part of it.
Kernel alone (`gf8.py`, profiler ns, L2 flushed; us base -> CMBS 2, bf16): N 1 S 5120 50.5 -> 47.2, S 6144 64.6 -> 60.9, S 7168 80.0 -> 76.5, N 5 S 2048
43.2 -> 40.4, S 4096 119.0 -> 115.9, S 6144 239.3 -> 237.8; level (+- 0.5 %) at N 1 S 3072 / 4096 / 8192, N 5 S 8192, N 48 S 1024 (no split or
cluster split there). F32IO (CMBS 1 is the largest that fits beside the raw tiles): level at N 1 (S 4096 31.3 / 31.4, S 6144 65.5 / 65.1, S 7168 80.5 / 81.3),
slower at N 5 S 6144 (239.8 -> 253.2): the F32IO build keeps CMBS 0. Module level (gap-measurer, `GAP=0.15`, two runs each, new vs base): InputFeatureEmbedder
L 6144 train 1024.5 / 1031.8 / 1039.1 against 1023.5 / 1032.6 / 1040.2 us -- the 3 native forwards of a step gain ~7 us of 1.03 ms, inside the noise; misses above
0.667: IFE 14 -> 14, SWAAtomTransformer 17 -> 17, AtomDiTESMFold2 19 -> 19; ladder 0 violations, 0 slower than the H100 (both trees).
Versions: `SWAAtomTransformer` 29 -> 31 (30 inference / 31 training), `InputFeatureEmbedder` 32 -> 34 (33 training / 34 inference). Kernel source only: no host change.

Verification: (1) bits -- `det9.py`: default build against the CMBS=0 build, 8 shapes (N 1 S 5120 / 6144 / 7168 / 8192, N 5 S 2048 / 4096 / 6144, N 3 S 3072; full and ragged
seqused), 200 launches each with O / LSE poisoned before every launch: 1600 launches, 0 mismatches (the race hunt of the forward; `gbrace.py` is the backward's);
(2) module level -- `dethash.py` (sha1 of outputs + all parameter gradients: eager twice, then CUDA-graph replay) on InputFeatureEmbedder L 6144 master / bfloat16 train + eval (12 cases),
SWAAtomTransformer + AtomDiTESMFold2 L 6144 float32 / master train (8): eager = eager = graph in every case, two separate processes identical, and the InputFeatureEmbedder hashes
identical to the PRISTINE base tree's (`n7-base`); (3) accuracy basis -- bit-identical outputs, so the engine-standard fp64-gold result of the base stands;
`measure.accuracy` InputFeatureEmbedder L 6144 master / bfloat16, train + eval (12 cuda cases): 9 GREEN, 3 YELLOW (master train grads, as in N2), 0 RED;
(4) `_prebuild --arch sm100`: 2 cubins built (`gattn_fwd_c`, `gattn_fwd32_c`), `--check` 0 not built, the same output as the base apart from the changed cubin hashes (exit 1 in both trees: sm90 plan lines);
(5) `tests/gpu` (swa_h100_global_composed, mask_key_skip_attention, atom_dit_many_atoms, atom_dit_backward_autocast, accuracy_smoke): 18 passed, 215 skipped (sm_90 only).

**Correction of the pass above**: the "SM clock ~1.55 GHz" sentence is not an explanation. A sustained loop of the kernel (no flush, `pw2.py`) holds 1965 MHz at 671 W
with no throttle reason, and runs 57.5 us a launch against 64.5 us flushed; clk-vs-globaltimer ratios of the TRACE builds (1557 MHz base, 1349 MHz PP=0) are
not a frequency. What stands is the per-block TRACE: period 1390 clk (S read 100 incl. the wait, mask 50, turn wait 7, exponentials 966, P store 36, gap 150).

**Schedule rewrite (3-deep S, P over S, handshake off the critical path): not built; ablations say why.** Barrier table of the shipped schedule (per warpgroup w, buffer b = G & 1;
arrival count / signal / waiter and parity): `q_full[QR]` 1 + tx (producer TMA, 2 tiles) / MMA warps at the item's first block, (li / QR) & 1; `q_empty[QR]` 2
(`tc_commit` of both MMA warps after the item's last QK) / producer, ((li / QR) - 1) & 1; `kv_full[ST]` 1 + tx (producer) / both MMA warps (QK) and F32IO warp 3, (G / ST) & 1;
`kv_empty[ST]` 2 CL (each MMA warp's `tc_commit` after QK(G + 2): PV(G) read V) / producer, ((g / ST) - 1) & 1; `s_full[w][b]` 1 (`tc_commit` after QK(G); also says PV(G - 2) is done,
so P buffer b is free) / softmax warps of w, (G >> 1) & 1; `p_full[w][b]` 4 (lane 0 of each softmax warp after `tcgen05.wait::st`) / MMA warp w before PV(G), (G >> 1) & 1;
`o_done[w]` 1 (`tc_commit` after the item's last PV) / softmax epilogue, li & 1; `o_free[w]` 4 (softmax warps after reading O) / MMA warp w before the next item's first PV, (li - 1) & 1.
Named barriers: 3 / 4 = the exponential turn tokens (256 threads, `bar.arrive` / `bar.sync`), 1 + w = warpgroup sync (128).
What the traces say about a deeper S: the softmax warp's wait for S(G) is inside the 100-clk "S read" (TMEM load included): in steady state S(G) is already
there when the warpgroup comes back, the MMA warp is not on the critical path, so a third S buffer / in-place P would remove at most ~100 + 150 clk of 1390 and the barrier hops are not the 1067-clk NOEXP floor of the
chain -- that floor is the same loop with the exponentials removed (S read waits 505 for the MMA warp only there). The critical path is the warpgroup's own work: with the turn token
only one warp per SMSP issues exponentials, 64 MUFU in 966 clk (15 clk each, twice the pipe's 8): its own FMUL2 -> MUFU -> F2FP / FADD2 dependence chain. `PP=0` (both warpgroups exponentiating at once)
drops the period to 1165 - 1190 clk (gate of ~1200 met by a one-flag ablation) with the same bits, but the kernel follows only where it is short: N 1 S 3072 25.8 -> 24.5, S 4096 32.1 -> 30.8 us (-4 %),
and loses where the grid is full or long: N 1 S 5120 47.3 -> 49.2, N 5 S 2048 / 4096 / 6144 +3 / +3 / +4 %, N 48 S 1024 / 2048 +3 / +4 %, S 6144 64.5 -> 67.1. So clk per block is not the kernel time here
(prologue 3.5 - 4 k and epilogue 12 - 17 k clk and the CTA-to-CTA balance grow with it); no schedule was written on that basis.

## 2026-11-19 (agent C, n13-fp8-c): FP8 for the atom DiT inference stages -- measured, not built
The H100 e4m3 row kernels (`pre90_f8` / `post90_f8`) have no B200 counterpart; the B200 stages are row-thread / TMA-latency bound with the MMAs hidden, tensor time is 13-16 % of the block at 5 x 4096 .. 5 x 8192 atoms
(analytic, from the measured bf16 block times 52 / 89 us), so the bound of an e4m3 product is ~6-8 % before the per-row amax cast it would add to the binding row-thread path. Nothing in this file's kernels changed; numbers and reasoning in `docs/b200/fp8.md`.

## 2026-11-19 (agent E, n14-e): the FP8 atom DiT rows route is BUILT and measured (supersedes the analytic note above)
`b200_atom/{pre_fwd_f8.cu, post_fwd_f8.cu, f8.py}`: e4m3 q / k / v / gate projections and out projection + SwiGLU expand (Boltz-2's up gate included) on tcgen05 `kind::f8f6f4`, opt-in (`MINIWORLD_ATOMDIT_FP8_B200`), default off and bit-identical
(`pre_fwd.cu` / `post_fwd.cu` untouched). Measured (docs/b200/fp8.md, "Agent E"): 0.93-0.99 of a block at 1024 atoms x 5 samples, level time and -4 .. -5 % energy at 2048 x 5, +1.5 .. +6 % above; the tensor time the analytic note
bounded at 6-8 % is real (free-quantisation ablation of the built stages: 0.85-0.90 of a row stage at 5120 rows, 0.95-0.99 at >= 20480), the per-row amax + cast on the row-thread path eats it. Served from 0 to 10240 rows.
