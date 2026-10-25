# Performance ladder: CUDA < portable < PyTorch on one H100 (2026-10-25)

Branch `perf/ladder-audit` (base: `origin/main` at `ae6136cd`; the engine sources under `src/` are those of `ae6136cd`, only `benchmarks/`, `tests/`, the CLI entry
`ladder` and docs were added). The maintainer's rule: **the portable (Triton) kernels are always faster than PyTorch, and the hand CUDA kernels are always faster than the
portable ones.** This record is the first full measurement of that ladder, for every op the engine serves that has a hand-CUDA path, a portable path or both, in inference
and training, at the production shapes of MiniWorld / FoldForge / La-Proteina. Harness and gate: `benchmarks/runners/ladder.py` ([benchmarks/README.md](../../benchmarks/README.md#performance-ladder-cuda--portable--pytorch)).

## How it was measured

* **Card**: one H100 80GB HBM3 per job (nodes node01 and node02 of the `h100` partition, 1-GPU jobs, at most two at a time), torch 2.13.0+cu129, triton 3.7.1, the repo pixi env, a
  private JIT cache (`~/miniworld-engine-scratch/ladder/jit`, copied from a warm cache so no first-call nvcc is timed). Ampere cards (A100 / A5000 / A6000) are **not available**:
  the portable-vs-PyTorch rung can only be judged on Hopper here and **may rank differently on Ampere** (the portable kernels are tuned on those cards, not on sm_90; the committed
  autotune caches are the Ampere ones), and no CUDA rung exists there by definition.
* **Rungs** (each row = one `(op, mode, dtype, shape)`; every number is the median of **3 independent `do_bench` medians** of the same replayed callable):
  * `pytorch`: `implementation=PYTORCH`, `torch.compile`d and CUDA-graph replayed (`pt_c`); the same statements eager-in-a-graph (`pt_eg`) and plain eager (`pt_e`) are reported next to it
    and **the PyTorch rung is the fastest of the three** (it was `pt_c` in every row of this run, so the ratios are against the best a careful user gets).
  * `portable`: `implementations=portable` = `settings.force_portable_kernels=True` for the row (the Triton family, every `*_cuda` stage off).
  * `cuda`: `implementations=miniworld` = the engine's own routing on this card (the hand CUDA path on an H100; the PyTorch reference where no CUDA path serves the call, which is
    detected and reported as a **routing gap**, not as a CUDA measurement).
  * The two engine rungs are timed compiled and not compiled (both CUDA-graph replayed; the engine kernels are opaque launches) and the faster counts. Training rows are fwd + bwd of
    the op / block (for the pair ops: fwd + `autograd.grad` over the table / parameter inputs, as in the earlier records). When a graph capture of a row fails it is re-measured
    without the graph (`retry`), a row that hits a CUDA error in a shared process is re-measured alone (`isolated`).
  * Parameters: the module benches' default `bf16-mixed` (fp32 master parameters under bf16 activations, MiniWorld's trainer). Where the portable rung cannot take that layout
    (single-direction TriMul, see below) the group is re-run with bf16 parameters (`bf16 params` rows).
* **Verdict per row** (ratio = lower rung / upper rung, must be < 1; a tie is a violation, a ratio within the repeat spread of 1 carries a `within noise` note):
  `OK`, `VIOLATION(cuda_vs_portable)`, `VIOLATION(portable_vs_pytorch)`, `VIOLATION(cuda_vs_pytorch)` (ops with no portable rung by design), `n.a.` (a rung has no measurement; the
  reason is in the note), `n.a.(cuda rung missing)` (the CUDA row ran the PyTorch statements: routing gap).
* **Provenance** is in the header line of every generated table below (commit of the reporting run, card, torch) and in `provenance.json` of each run directory
  (`~/miniworld-engine-scratch/ladder/out_full_{A,B,C,D}`: A = trunk ops, B = diffusion / norms / pair ops, C and D = re-runs of groups that failed for harness reasons).
  Raw rows: [data/performance_ladder_20261025.csv](data/performance_ladder_20261025.csv); the debt list the gate test reads: [data/performance_ladder_known.txt](data/performance_ladder_known.txt).
* Harness checks: the known violations of the brief reproduce (pair_embed_geom forward 1.17-1.18x slower in CUDA, geom train 1.16-1.38x, pair_embed_ca training 1.3-1.73x,
  adaln fwd+bwd about equal, bias-only DiT bf16 1.03-1.08x slower than compiled PyTorch, DiT `extra_pair_bias` bf16 1.07-1.08x, OPM RF3 training 1.00-1.05x), so the numbers agree with
  the earlier records (`portable_pair_ops_20261025`, `portable_parity_modules_20261025`, `triton_gaps_a_20261024`, `triton_dit_portable_20261024`) within their run-to-run noise.
  `fp32_check_limit=0.01` loosens the harness' compiled-vs-eager consistency check for fp32 tensors (1e-4 -> 1e-2): the fp32 gradients of a bf16 computation differ by 3e-3 between a compiled
  and an eager step, which stopped the PyTorch rung of several training rows from being timed at all.

## Headline

305 ladder rows (every `(op, mode, dtype, shape)` of the grid, 3 repeats per rung; two rungs per group are re-run as `bf16 params` variants where the fp32-master layout has no portable rung).

| verdict | rows |
|---|---|
| OK (CUDA < portable < PyTorch where the rungs exist) | 159 |
| VIOLATION(cuda_vs_portable): the hand CUDA path is slower than (or ties) the portable kernels | 30 |
| VIOLATION(portable_vs_pytorch): the portable kernels are slower than (or tie) compiled + graphed PyTorch | 48 |
| VIOLATION(cuda_vs_pytorch) (ops with no portable rung by design) | 0 |
| n.a.(cuda rung missing): `miniworld` runs the PyTorch statements, a routing gap (the portable kernels are faster by more than 1 % in 28 of the 38) | 38 |
| n.a.(other rung missing): single-direction TriMul with fp32 master parameters (portable fails, 32 rows), SWA DiT (portable rejected by design, 6), `pair_embed_adaln` training (portable backward raises, 5) | 43 |

(a row can carry two violations, so the verdict counts add to more than the violating rows; the per-op table below has the split.) What holds, in every measured cell: TriAttn,
bidirectional TriMul (hidden = D, D64 .. D512), OuterProductMean and MSAPairWeightedAveraging at the MiniWorld shapes and PWA value_dim 8, the fused token DiT (bf16, fp32, qk-norm,
8 / 8 rows), the AF3 LocalDiT atom block (bf16 and fp32, 6 / 6), `pair_embed_adaln` inference, `smooth_lddt` (training and all but one tie), every sampler kernel (0.12-0.58 of PyTorch), SWA
attention / SWA DiT against PyTorch (0.03-0.09), `AttentionPairBias` at pair 128 (portable 0.87-0.99 of PyTorch, CUDA 0.35-0.40) and the DiT variant stacks / `AugmentedAttention` / the
ESMFold2 atom block **for the portable rung** (1.2-2.5x faster than PyTorch, where it exists).

**The invariant fails in two ways, and they are different problems.**

1. **Portable slower than PyTorch (48 rows, 8 ops), all of them H100 (sm_90) numbers with untuned Triton**: bias-only DiT 14 / 14 rows (1.03-1.10x, bf16 and fp32, inference and training),
   Transition 11 rows (1.04-1.35x, growing with width: D384-D768, inference D768 1.24x and training D768 1.35x), ConditionedTransition 7 (1.04-1.15x), LayerNorm (pair) 5 (1.01-1.07x, forward
   ties), RMSNorm 4 (1.17-1.42x), DiT `extra_pair_bias` 3 (1.07-1.09x), AdaLN fp32 2 (1.09-1.18x), OPM RF3 divide-after 2 (ties). **These can only be judged as real violations of the user's rule on
   Ampere, where the portable kernels are tuned**; on H100 they are the "untuned Hopper build" the earlier records already name.
2. **CUDA slower than portable (30 rows, 8 ops)**: `pair_embed_geom` 10 / 10 rows (1.08-1.38x: forward 1.08-1.18x, training 1.16-1.38x), `pair_embed_ca` training 5 / 5 (1.31-1.73x),
   `token_pair_init` inference 5 / 5 (1.00-1.13x), `LayerNorm` (pair) training 2 (1.18 / 1.22x: the H100 CUDA backward routes D128), `AttentionPairBias` at pair width 256 2 (1.03-1.07x),
   AdaLN bf16 inference L384 (1.03, noise), `smooth_lddt` A48 N1024 inference (1.002, a tie at the 0.16 ms launch floor), RMSNorm inference L384 (1.09x), plus three exact ties (OPM RF3 divide-after,
   DiT `extra_pair_bias` fp32, `token_pair_init` B48 L256). Other agents are already fixing `pair_embed_geom` / `ca` / `adaln` (worktree `cuda-pair`).

**Routing gaps** (separate from violations; no CUDA rung exists, so the ladder is not testable, but the user gets PyTorch although a portable kernel is faster): on H100 `miniworld` runs the
PyTorch reference for **training** of `AugmentedAttention` (1.67-2.47x slower than the portable kernels at the production shapes), of the Boltz-2 / RoseTTAFold3 / Chai-1 token (1.5-1.6x) and atom
(2.1-2.3x) stacks and of ESMFold2's atom block (1.9-2.0x), for `AugmentedAttention` inference (1.3-1.6x), for bidirectional TriMul with `d_hidden != d_pair` (D64 / hidden 128: 3.9x inference, 2.6x
training), and for the OPM model variants with 8 groups (Chai-1, 1.07-1.27x). Weighted by production use (weight 3 = the MiniWorld / FoldForge / La-Proteina shape), the ranked list is in the
second violations table below: the La-Proteina pair builders (`pair_embed_ca` / `geom` training, `token_pair_init` inference) and the pair LayerNorm backward come first, then the bias-only DiT (all rows) and
the Transition widths of the token track.

## What the invariant implies for ops with a structurally missing rung

* **Sampler glue (`augment_noise`, `align_euler`, `pool_atoms`) and SWA global attention have no portable path by design** (CUDA kernels + PyTorch reference; SWA attention is FlashAttention-4).
  The ladder collapses to CUDA < PyTorch, and it holds everywhere: `augment_noise` 0.50-0.58 of compiled PyTorch (0.010 vs 0.018 ms at 4096 atoms, B5), `align_euler` 0.12-0.19
  (0.017 vs 0.089 ms ... 0.026 vs 0.213 ms at 24,576 atoms), `pool_atoms` 0.16-0.28; SWA global attention 0.032-0.069 (inference 0.076 vs 1.12 ms and 0.133 vs 4.14 ms at 4096 / 8192 atoms, training
  2.0 vs 29.6 ms and 4.0 vs 116.7 ms), the fused SWA DiT block 0.035-0.090 (bf16) and 0.052-0.066 (fp32). **The PyTorch rung of SWA is not FA2**: it is the module's `implementation=PYTORCH`
  path, which on this install runs PyTorch's memory-efficient SDPA with a band mask (`fmha_cutlassF` 0.99 ms of the 1.12 ms) and the module's rows; the env has FlashAttention-4 only, FA2 is
  not installed, so a CUDA-vs-FA2 rung could not be measured. FA4 is itself the attention core of the engine path (`kernel_cutlass_kernel_flash_at...`), so against FA4 the ladder would only
  compare the fused block around it (RMSNorm-adaLN, SwiGLU, gates), which is the 0.035-0.090 above measured against the SDPA-based reference.
  Consequence of the invariant for these ops: the gate can only say CUDA < PyTorch; a future portable rung (an Ampere card needs one for the sampler and for SWA) would have to land between
  the two numbers above, which leaves it room (the SDPA reference is 12-30x slower than the engine's block), i.e. the invariant is satisfiable but not yet testable.
  `SWADiTBlock` additionally **cannot be compiled** with the engine path in place (the compiled rows fail: `torch._check` / data-dependent guard in `kernels/_capture.py:151` for inference, a
  Dynamo `autograd.Function` skip for training); the CUDA rung of the SWA DiT rows is therefore the not-compiled, CUDA-graph-replayed block, which is how it is deployed.
  `force_portable_kernels` on SWA raises (`native H100 SWA rejects force_portable_kernels`), so the portable rung of `SWADiTBlock` is `n.a.` on H100 by construction.
* **ConditionedTransition and AdaLN have no CUDA path on H100 yet** (`docs/dispatch.md`: "PyTorch (no CUDA path yet)"): their `cuda` column equals the PyTorch statements, the
  cuda_vs_portable pair is a missing rung, and the invariant reduces to portable < PyTorch, which fails (below).
* **LayerNorm (pair, forward)**: on H100 `layernorm_kernel` forward IS the portable Triton kernel (no distinct CUDA rung); the pair is `n.a.` and the invariant reduces to portable < PyTorch, which
  fails by 1-4 % (HBM-bound at 82-90 % of peak; a tie in practice, flagged as a violation because a tie is not "faster").

## Violations: root-cause hypotheses (cheap, from the kernel profile of the row; the fixes are not part of this task)

Source of every number: the `torch.profiler` kernel list of one call of the timed step, stored with each row (`top_kernels`, the "Violation detail" section below has them all). A hypothesis
says what the profile shows and what to look at first; none was verified by a change.

| # | violation | what the profile shows | hypothesis (first thing to look at) |
|---|---|---|---|
| 1 | `pair_embed_ca` / `pair_embed_geom` training, CUDA 1.16-1.73x slower than portable (weight 3 at B16) | B16 L384 ca: CUDA `pg_bwd_mma` 1812 us, `pg_ln_bwd` 1260, `pg_fwd_ca` 594 against portable `_geom_dw` 772, `_geom_ln_bwd` 830, `_geom_fwd_gather` 427 (+ `_geom_dw_reduce` 75, `_ca_bins` 22). geom L384: `pg_bwd_mma` 2964 / `pg_fwd` 1756 / `pg_ln_bwd` 1526 against 2015 / 1042 + 151 / 1322 | all three CUDA kernels are slower than their Triton counterparts, the table-gradient MMA kernel by 2.3x (ca) / 1.5x (geom): its tiling / occupancy, not the structure (the portable kernel is the same three-stage design). Being fixed in worktree `cuda-pair` |
| 2 | `pair_embed_geom` inference 1.08-1.18x | `pg_fwd` 1128 us (L384 B16) against `_geom_fwd_gather` 821 + `_geom_bins` 150 | same kernel family as 1: the CUDA fused forward does the per-pair geometry replicated over the channel tile (the Triton note in `portable_pair_ops_20261025.md`: "one kernel doing both was 5-6x slower"; the CUDA fused kernel pays a smaller version of that) |
| 3 | `token_pair_init` inference fp32, 1.00-1.13x (weight 3) | `tpi_fwd` 442 us against `_tpi_fwd` 459 at L384 (3 % apart, equal within noise at L384 / 256 B48; 13 % at L128 B16: 67 vs 56 us) | memory-bound output write (`z` [B, L, L, 128]): both kernels sit at the HBM write roofline, the small-L difference is launch / tail; no CUDA headroom to find except at L128 |
| 4 | LayerNorm (pair) training, CUDA 1.18-1.22x (weight 3) | portable: `layer_norm_bwd_dx_fused` 52 us (L384) / 186 us (L768) + fwd; the CUDA row's profile shows only `layer_norm_bwd_reduce_kernel` 6 us (its main kernels are not visible to the profiler), total 0.104 / 0.360 ms against 0.088 / 0.296 | `layernorm_kernel` routes bf16 128 <= N <= 512 to the hand CUDA backward (tuned at D512: 1.20x there). At **D128** with M = L*L = 147k / 590k rows the persistent Triton backward wins by 18-22 %: look at `_static_bwd_path` / the WAVES constant for N = 128 |
| 5 | RMSNorm, portable 1.17-1.42x slower than compiled PyTorch; CUDA 1.09x slower than portable at L384 inference (weight 2) | inference L1024 x[48,1024,768]: PyTorch `triton_per_fused__fused_rms_norm` 56 us (79 % of HBM peak), portable `rmsnorm_fwd_kernel` 80 us (56 %). Training L1024: portable `rmsnorm_bwd` 128 + `rmsnorm_fwd` 68 us against inductor 98 + 12 | the Triton RMSNorm is a row-per-program kernel with an untuned (Ampere) `num_warps` / rows-per-program for sm_90 at N = 768; inductor's persistent reduction (one row block per program, rstd recomputed in the backward) does one pass less. Ties within 3 % at L384 (launch-bound) |
| 6 | bias-only DiT portable vs PyTorch, 14 / 14 rows 1.03-1.10x (weight 3) | inference A4 L256: portable `_squeeze_gate` 18 us + `_expand_swiglu` 16 + `_adaln_gemm_gate` 15 x 2 + LN 14 + cuBLAS 3 GEMMs; PyTorch: 6 + 4 cuBLAS GEMMs + 14 small fused Triton kernels (`triton_poi_fused__to_copy*`, 12-14 us). Training L256: portable `_dh_swiglu_bwd` 130 us, `_bwd_x_kernel` 119 x 2, `_adaln_fwd_gate` 105 x 2, cuBLAS 203 + 180 us against PyTorch cuBLAS 293 + 237 | the same untuned Hopper build of the AdaLN + ConditionedTransition kernel families that `portable_parity_modules_20261025.md` profiled (71 us against 47 us): expand / squeeze GEMM kernels cost 34 us where two cuBLAS calls cost 13; the attention core and the pair-bias path are at parity or better. The loss is the Triton GEMM-in-epilogue kernels on sm_90, not the composition |
| 7 | ConditionedTransition portable 1.04-1.15x slower (7 / 8 rows), AdaLN fp32 1.09-1.18x (2 rows) | CT inference d768: portable `_expand_swiglu` 40 us + `_squeeze_gate` 37 + `_adaln_gemm_gate` 16 against PyTorch cuBLAS 34 + 23 + 16 (+ fused pointwise 13) | same as 6 (these are the parts of the bias-only block); on H100 the `cuda` column of these two ops IS the PyTorch statements (no CUDA path yet), so their CUDA rung is missing |
| 8 | Transition portable vs PyTorch, 11 rows 1.04-1.35x, growing with width (D384-D768) (weight 2) | D768 L384 inference: portable `_squeeze_residual_kernel` 1990 us, total 5.43 ms against PyTorch 4.36 (3 cuBLAS GEMMs 2846 us in 192x192 nvjet tiles, + 4 fused pointwise 0.9 ms) and CUDA 3.47 (`tr90::gemm_kernel`); training D768: `_transition_expand_gatebwd_kernel` 4393 us + `transition_fwd_kernel` 3016 + `_squeeze_residual` 1980 | the Triton dual-GEMM kernels use `BLOCK_K = next_pow2(K)` / default tiles that scale badly with K and N and are not wgmma-tuned on sm_90 (the Ampere-tuned configs); the cuBLAS 192x192 2x1-cluster kernels win from D384 on. At D128 (1.5-1.8x faster than PyTorch) and D256 inference the portable path wins |
| 9 | DiT `extra_pair_bias` bf16 training 1.07-1.09x, fp32 inference 1.09x | block of the same parts as 6 (the attention core of the portable block is 1.5-2x faster than PyTorch's: DiT variant stacks) | the [L, L] bias term is not the cost; the loss is the AdaLN / ConditionedTransition kernels of 6 / 7, and the term-free token DiT (fused CUDA runner) is OK in all 8 rows |
| 10 | `AttentionPairBias` at pair width 256, CUDA 1.03-1.07x slower than portable (weight 1) | CUDA 0.158 ms against portable 0.147 and PyTorch 0.152 (inference, L384) | at pair 128 the CUDA path wins 2.4x; at 256 it is level with PyTorch, so the H100 fused bias producer is probably specialised for 128-wide pairs and the 256 case takes the generic leg |
| 11 | OPM RF3 "divide after" (projection bias, ESMFold2 order): ties in every direction | CUDA 0.867 / portable 0.887 / PyTorch 0.873 ms (training L256) | `opm_train` refuses `normalize_before_proj=False` (documented in `docs/dispatch.md`); the portable path's gain (1.04-1.05x for "divide before") disappears: the backward is cuBLAS-bound |
| 12 | Routing gaps (no CUDA rung) | `miniworld` rows run `triton_*` Inductor kernels (e.g. the atom stacks spend 3.4 ms of 9.3 ms in `triton_poi_fused_index_put_new_zeros`, the reference's gather backward, which the portable `_window_bwd` + `_window_fold` pair replaces by 1.1 ms) | the H100 CUDA paths are inference-only for these blocks (DiT variants, ESMFold2 atom block: bf16 inference; `AugmentedAttention` has no module-level CUDA route), so a training call falls back to the reference even though the engine already holds a faster portable path; routing the call to the portable kernels under `miniworld` on H100 would turn every gap row into a 1.2-3.9x win until a CUDA training path exists (`docs/gpus/h100/dispatch.md` says "no Triton on H100, on purpose": the maintainer's call) |
| 13 | `smooth_lddt` A48 N1024 inference tie (1.002) and `LayerNorm` / `RMSNorm` L384 ties | 0.161 vs 0.161 ms (`slddt_fwd_full` 128 us + prep 9 us vs `smooth_lddt_fwd_kernel` 133 us) | launch / latency floor of a 130 us kernel; both are the same algorithm on CUDA cores. Not a performance bug; a tie fails a strict "faster" gate, so `performance_ladder_known.txt` lists it |
| 14 | single-direction TriMul portable fails with fp32 master parameters | `CompilationError: Both operands must be same dtype. Got bf16 and fp32` (`trimul_inproj` front, `tl.dot(a, w)` with `a` bf16 and `w` fp32) and, in the backward, `torch.bmm(d_tri, rf)` with mixed dtypes | the bidirectional module and TriAttn cast the parameters to the activation dtype inside; the single-direction front / back do not (`portable_parity_modules_20261025.md` item 4 fixed TriAttn variants only). MiniWorld's trainer (fp32 master under bf16 autocast) would hit this on any recorded Ampere card |
| 15 | `pair_embed_adaln` portable training raises under the default `compile_wrap=custom_op` | custom-op aliasing check: an output of `pair_embed_adaln_bwd` aliases another output / an input | `portable_backward` returns `dl[0, :C].contiguous()`, `dl[0, C:].contiguous()` (both views of one tensor, already contiguous) and `dtbl.view(NT, C)`: return clones |
| 16 | `MSAPairWeightedAveraging` CUDA training fails after a portable row in the same process | `CUDA error: invalid argument` in `pwa_train.py:720` (`backward`), at L384 / 768, never at L256, never in its own process (1.41 / 3.40 ms) | state of the portable rung (`force_portable_kernels` row before it) left in a cache of `pwa_train` (launch configuration / workspace keyed without the switch); harmless in production, breaks A/B harnesses that switch in one process |

## Not measured / limits

* No Ampere card: no A100 / A5000 / A6000 numbers, so the portable-vs-PyTorch rung is unchecked on the cards the portable kernels are tuned for. Everything portable here ran **untuned on Hopper**
  (the committed autotune caches are Ampere's; the autotuner searched the default config lists on first use of a shape bucket).
* The Anthropic and cuEquivariance rows are not part of the ladder (a different question). `B > 1` TriMul / TriAttn, TriAttn model variants (RF3 / Chai-1 biases, dual output), TriMul at L1536+,
  atom counts above 8,192 for the DiT blocks (the pair builders and the sampler go to 24,576 atoms), Pairformer-level composition and multi-layer stacks were not swept: the grid is the production
  shapes of the brief (pair 64-512, token 384 / 768, L 128-1024, atoms 4k-8k for the blocks, B / samples 1-48 where the op has them), not every shape the engine serves.
* `smooth_lddt` was swept at A48 x N 1024 / 2048 and A16 x N 4096: the PyTorch reference runs out of its int32 index range at A48 x N4096 (illegal memory access), which is a limit of the reference
  (45-1000x slower than either kernel) and not of the engine.
* Single-direction `TriangleMultiplication` with the benchmark default (fp32 master parameters, bf16 activations) has **no portable rung** at all: `CompilationError: Both operands must be same
  dtype. Got bf16 and fp32` in the front kernel (inference) and `expected scalar type Float but found BFloat16` in the backward (training). Its numbers are the `bf16 params` rows. The bidirectional module,
  TriAttn and the DiT blocks take fp32 master parameters.
* `pair_embed_adaln` **training** has no portable measurement: under the default `compile_wrap=custom_op` the portable backward raises `The output of this custom operator ... may not alias any inputs
  to this custom operator or other returns` (hypothesis: `portable_backward` returns `dl[0, :C].contiguous()` / `dl[0, C:].contiguous()`, which are views of one buffer; `dtbl.view(...)` likewise). The
  earlier record measured it with `MINIWORLD_COMPILE_WRAP=disable`, where the registration does not apply. Forward rows are unaffected.
* `MSAPairWeightedAveraging` training (CUDA rung) at L384 / 768: see the table; the first runs hit `CUDA error: invalid argument` in `integrations/pwa_train.py:720` when the rung shared its process with
  the portable rung (State left behind by the portable run, or by the profiler of the kernel probe); the re-run in its own process is what the table holds.
* Timings are CUDA-graph or compiled replays; nothing here is eager, except the `pt_e` reference column.

## Generated tables (benchmarks/runners/ladder.py report)

Provenance: commit `95670aa9` (branch `perf/ladder-audit`, dirty=False), card **NVIDIA H100 80GB HBM3**, torch 2.13.0+cu129, run 2026-10-25T18:35:49 on node02, Slurm job 35248.

### Summary by op

| rows | OK | VIOLATION(cuda_vs_portable) | VIOLATION(portable_vs_pytorch) | VIOLATION(cuda_vs_pytorch) | n.a.(cuda rung missing: no CUDA path, `miniworld` runs PyTorch) | n.a.(other rung missing) |
|---|---|---|---|---|---|---|
| 305 | 159 | 30 | 48 | 0 | 38 | 43 |

| op | rows | OK | cuda>=portable | portable>=pytorch | cuda>=pytorch (no portable) | no CUDA path (routing gap) | other n.a. |
|---|---|---|---|---|---|---|---|
| TriMul outgoing | 40 | 20 | 0 | 0 | 0 | 0 | 20 |
| TriMul incoming | 12 | 6 | 0 | 0 | 0 | 0 | 6 |
| TriMul outgoing hidden!=D | 12 | 6 | 0 | 0 | 0 | 0 | 6 |
| TriMul bidirectional | 14 | 14 | 0 | 0 | 0 | 0 | 0 |
| TriMul bidirectional hidden!=D | 2 | 0 | 0 | 0 | 0 | 2 | 0 |
| TriAttn | 10 | 10 | 0 | 0 | 0 | 0 | 0 |
| Transition | 26 | 15 | 0 | 11 | 0 | 0 | 0 |
| OuterProductMean | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| MSAPairWeightedAveraging | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| OuterProductMean RF3 (bias, divide before) | 4 | 2 | 0 | 0 | 0 | 2 | 0 |
| OuterProductMean RF3 (bias, divide after) | 4 | 1 | 1 | 2 | 0 | 2 | 0 |
| OuterProductMean Chai-1 (8 groups) | 4 | 0 | 0 | 0 | 0 | 4 | 0 |
| MSAPairWeightedAveraging value_dim 8 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| AttentionPairBias | 8 | 6 | 2 | 0 | 0 | 0 | 0 |
| AugmentedAttention token | 6 | 0 | 0 | 0 | 0 | 6 | 0 |
| AugmentedAttention atom | 4 | 0 | 0 | 0 | 0 | 4 | 0 |
| token DiT (fused+composed) | 8 | 8 | 0 | 0 | 0 | 0 | 0 |
| token DiT + extra_pair_bias | 6 | 2 | 1 | 3 | 0 | 2 | 0 |
| DiT variant stack | 6 | 3 | 0 | 0 | 0 | 3 | 0 |
| LocalDiT atom block | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| LocalDiT ESMFold2 atom block | 4 | 2 | 0 | 0 | 0 | 2 | 0 |
| DiT variant stack (atom) | 6 | 3 | 0 | 0 | 0 | 3 | 0 |
| bias-only DiT | 14 | 0 | 0 | 14 | 0 | 0 | 0 |
| SWA DiT (atom) | 6 | 0 | 0 | 0 | 0 | 0 | 6 |
| SWA atom attention (global) | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| AdaLN | 8 | 5 | 1 | 2 | 0 | 2 | 0 |
| ConditionedTransition | 8 | 0 | 0 | 7 | 0 | 6 | 0 |
| LayerNorm (pair) | 8 | 1 | 3 | 5 | 0 | 0 | 0 |
| RMSNorm | 4 | 0 | 1 | 4 | 0 | 0 | 0 |
| pair_embed_geom | 10 | 0 | 10 | 0 | 0 | 0 | 0 |
| pair_embed_ca | 10 | 5 | 5 | 0 | 0 | 0 | 0 |
| pair_embed_adaln | 10 | 5 | 0 | 0 | 0 | 0 | 5 |
| token_pair_init | 10 | 5 | 5 | 0 | 0 | 0 | 0 |
| smooth_lddt | 6 | 5 | 1 | 0 | 0 | 0 | 0 |
| sampler augment_noise | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| sampler align_euler | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| sampler pool_atoms | 3 | 3 | 0 | 0 | 0 | 0 | 0 |

### Routing gaps: the CUDA rung does not exist, `miniworld` on this card runs the PyTorch reference

Rows where the hand CUDA path does not serve the call (a dtype, a shape, a mode, an option it does not take) and the engine's portable kernels would be faster than what `miniworld` runs. The ladder invariant is not testable for them (no CUDA rung); the routing is what a user sees. `speed-up` = PyTorch / portable.

| op | mode | dtype | shape | L | miniworld (= PyTorch) ms | portable ms | speed-up lost | weight |
|---|---|---|---|---|---|---|---|---|
| TriMul bidirectional hidden!=D | inference | bf16 | D64 hidden128 | 384 | 1.231 | 0.318 | 3.86x | 2 |
| TriMul bidirectional hidden!=D | training | bf16 | D64 hidden128 | 384 | 3.643 | 1.379 | 2.64x | 2 |
| AugmentedAttention atom | training | bf16 | d128 A5/A48 | 384 | 37.574 | 15.200 | 2.47x | 1 |
| DiT variant stack (atom) | training | bf16 | RoseTTAFold3 atom | 512 | 9.353 | 4.147 | 2.26x | 2 |
| DiT variant stack (atom) | training | bf16 | Chai-1 atom | 512 | 8.005 | 3.604 | 2.22x | 2 |
| DiT variant stack (atom) | training | bf16 | Boltz-2 atom | 512 | 9.340 | 4.428 | 2.11x | 2 |
| LocalDiT ESMFold2 atom block | training | bf16 | d128 192 keys rotary (atoms=8L) | 1024 | 25.772 | 12.806 | 2.01x | 2 |
| AugmentedAttention token | training | bf16 | d768 A5/A48 | 768 | 14.424 | 7.283 | 1.98x | 2 |
| LocalDiT ESMFold2 atom block | training | bf16 | d128 192 keys rotary (atoms=8L) | 512 | 12.664 | 6.568 | 1.93x | 2 |
| AugmentedAttention atom | training | bf16 | d128 A5/A48 | 128 | 4.028 | 2.225 | 1.81x | 1 |
| AugmentedAttention token | training | bf16 | d768 A5/A48 | 384 | 4.328 | 2.587 | 1.67x | 2 |
| AugmentedAttention atom | inference | bf16 | d128 A5/A48 | 384 | 1.843 | 1.136 | 1.62x | 1 |
| DiT variant stack | training | bf16 | Chai-1 token | 384 | 5.124 | 3.214 | 1.59x | 2 |
| DiT variant stack | training | bf16 | RoseTTAFold3 token | 384 | 5.471 | 3.584 | 1.53x | 2 |
| DiT variant stack | training | bf16 | Boltz-2 token | 384 | 5.603 | 3.736 | 1.50x | 2 |
| AugmentedAttention token | inference | bf16 | d768 A5/A48 | 768 | 0.553 | 0.387 | 1.43x | 2 |
| AugmentedAttention atom | inference | bf16 | d128 A5/A48 | 128 | 0.242 | 0.182 | 1.34x | 1 |
| AugmentedAttention token | inference | bf16 | d768 A5/A48 | 384 | 0.212 | 0.162 | 1.31x | 2 |
| OuterProductMean Chai-1 (8 groups) | inference | bf16 | S512 pair256 8x64 | 256 | 0.277 | 0.218 | 1.27x | 2 |
| OuterProductMean Chai-1 (8 groups) | inference | bf16 | S512 pair256 8x64 | 384 | 0.552 | 0.446 | 1.24x | 2 |
| AugmentedAttention token | inference | fp32 | d768 A5/A48 | 384 | 0.343 | 0.286 | 1.20x | 1 |
| AugmentedAttention token | training | fp32 | d768 A5/A48 | 384 | 6.161 | 5.226 | 1.18x | 1 |
| OuterProductMean Chai-1 (8 groups) | training | bf16 | S512 pair256 8x64 | 384 | 1.594 | 1.451 | 1.10x | 2 |
| ConditionedTransition | inference | fp32 | d768 A5/A48 | 384 | 0.136 | 0.125 | 1.09x | 2 |
| OuterProductMean Chai-1 (8 groups) | training | bf16 | S512 pair256 8x64 | 256 | 0.811 | 0.758 | 1.07x | 2 |
| OuterProductMean RF3 (bias, divide before) | training | bf16 | S512 msa64 pair128 | 384 | 1.767 | 1.677 | 1.05x | 2 |
| OuterProductMean RF3 (bias, divide before) | training | bf16 | S512 msa64 pair128 | 256 | 0.864 | 0.829 | 1.04x | 2 |

### Violations ranked by size (ratio = lower rung / upper rung; must be < 1)

| # | op | mode | dtype | shape | L | pair | ratio | CUDA ms | portable ms | PyTorch best ms (form) | weight | cause / note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | pair_embed_ca | training | bf16 out, fp32 geometry | B48 | 256 | cuda_vs_portable | 1.726 | 4.694 | 2.720 | 9.614 (pt_compiled) | 2 | CUDA kernel slower than the portable kernel |
| 2 | pair_embed_ca | training | bf16 out, fp32 geometry | B16 | 384 | cuda_vs_portable | 1.714 | 3.628 | 2.116 | 7.261 (pt_compiled) | 3 | CUDA kernel slower than the portable kernel |
| 3 | pair_embed_ca | training | bf16 out, fp32 geometry | B48 | 128 | cuda_vs_portable | 1.620 | 1.229 | 0.758 | 2.442 (pt_compiled) | 2 | CUDA kernel slower than the portable kernel |
| 4 | pair_embed_ca | training | bf16 out, fp32 geometry | B16 | 256 | cuda_vs_portable | 1.590 | 1.622 | 1.020 | 3.242 (pt_compiled) | 3 | CUDA kernel slower than the portable kernel |
| 5 | RMSNorm | inference | bf16 | x[48,L,768] | 1024 | portable_vs_pytorch | 1.421 | 0.068 | 0.080 | 0.056 (pt_compiled) | 2 | portable path slower than PyTorch |
| 6 | pair_embed_geom | training | bf16 out, fp32 geometry | B16 | 384 | cuda_vs_portable | 1.384 | 6.208 | 4.484 | 15.244 (pt_compiled) | 3 | CUDA kernel slower than the portable kernel |
| 7 | Transition | training | bf16 | D768 | 384 | portable_vs_pytorch | 1.351 | 11.025 | 16.568 | 12.268 (pt_compiled) | 2 | portable path slower than PyTorch |
| 8 | pair_embed_geom | training | bf16 out, fp32 geometry | B48 | 256 | cuda_vs_portable | 1.336 | 7.929 | 5.935 | 20.146 (pt_compiled) | 2 | CUDA kernel slower than the portable kernel |
| 9 | pair_embed_geom | training | bf16 out, fp32 geometry | B16 | 256 | cuda_vs_portable | 1.318 | 2.763 | 2.097 | 6.850 (pt_compiled) | 3 | CUDA kernel slower than the portable kernel |
| 10 | pair_embed_ca | training | bf16 out, fp32 geometry | B16 | 128 | cuda_vs_portable | 1.308 | 0.438 | 0.335 | 0.856 (pt_compiled) | 3 | CUDA kernel slower than the portable kernel |
| 11 | pair_embed_geom | training | bf16 out, fp32 geometry | B48 | 128 | cuda_vs_portable | 1.283 | 2.071 | 1.614 | 5.167 (pt_compiled) | 2 | CUDA kernel slower than the portable kernel |
| 12 | RMSNorm | training | bf16 | x[48,L,768] | 1024 | portable_vs_pytorch | 1.281 | 0.155 | 0.212 | 0.165 (pt_compiled) | 2 | portable path slower than PyTorch |
| 13 | Transition | training | bf16 | D512 | 384 | portable_vs_pytorch | 1.270 | 6.028 | 7.871 | 6.199 (pt_compiled) | 1 | portable path slower than PyTorch |
| 14 | Transition | training | bf16 | D384 | 768 | portable_vs_pytorch | 1.243 | 13.659 | 19.940 | 16.043 (pt_compiled) | 2 | portable path slower than PyTorch |
| 15 | Transition | inference | bf16 | D768 | 384 | portable_vs_pytorch | 1.240 | 3.377 | 5.402 | 4.356 (pt_compiled) | 2 | portable path slower than PyTorch |
| 16 | RMSNorm | training | bf16 | x[48,L,768] | 384 | portable_vs_pytorch | 1.230 | 0.072 | 0.084 | 0.069 (pt_compiled) | 2 | portable path slower than PyTorch |
| 17 | LayerNorm (pair) | training | bf16 | x[1,L,L,128] | 768 | cuda_vs_portable | 1.217 | 0.360 | 0.296 | 0.300 (pt_compiled) | 3 | CUDA kernel slower than the portable kernel; portable_vs_pytorch within noise (ratio 0.984, spread 0.002) |
| 18 | Transition | training | bf16 | D384 | 384 | portable_vs_pytorch | 1.213 | 3.638 | 4.958 | 4.087 (pt_compiled) | 2 | portable path slower than PyTorch |
| 19 | LayerNorm (pair) | training | bf16 | x[1,L,L,128] | 384 | cuda_vs_portable | 1.184 | 0.104 | 0.088 | 0.094 (pt_compiled) | 3 | CUDA kernel slower than the portable kernel |
| 20 | pair_embed_geom | inference | bf16 out, fp32 geometry | B16 | 128 | cuda_vs_portable | 1.179 | 0.161 | 0.137 | 1.280 (pt_compiled) | 3 | CUDA kernel slower than the portable kernel |
| 21 | AdaLN | inference | fp32 | d768 cond384 A5/A48 | 384 | portable_vs_pytorch | 1.176 | 0.035 | 0.040 | 0.034 (pt_compiled) | 2 | portable path slower than PyTorch; n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference |
| 22 | pair_embed_geom | inference | bf16 out, fp32 geometry | B16 | 384 | cuda_vs_portable | 1.173 | 1.152 | 0.982 | 10.563 (pt_compiled) | 3 | CUDA kernel slower than the portable kernel |
| 23 | pair_embed_geom | inference | bf16 out, fp32 geometry | B16 | 256 | cuda_vs_portable | 1.168 | 0.528 | 0.452 | 4.693 (pt_compiled) | 3 | CUDA kernel slower than the portable kernel |
| 24 | RMSNorm | inference | bf16 | x[48,L,768] | 384 | portable_vs_pytorch | 1.165 | 0.031 | 0.028 | 0.024 (pt_compiled) | 2 | portable path slower than PyTorch |
| 25 | pair_embed_geom | training | bf16 out, fp32 geometry | B16 | 128 | cuda_vs_portable | 1.158 | 0.752 | 0.650 | 1.827 (pt_compiled) | 3 | CUDA kernel slower than the portable kernel |
| 26 | ConditionedTransition | inference | bf16 | d768 A5/A48 | 1024 | portable_vs_pytorch | 1.153 | 0.144 | 0.171 | 0.148 (pt_compiled) | 2 | portable path slower than PyTorch |
| 27 | token_pair_init | inference | fp32 in, fp32 out | B16 | 128 | cuda_vs_portable | 1.133 | 0.089 | 0.078 | 0.544 (pt_compiled) | 3 | CUDA kernel slower than the portable kernel |
| 28 | Transition | training | bf16 | D256 | 768 | portable_vs_pytorch | 1.127 | 7.166 | 10.165 | 9.019 (pt_compiled) | 2 | portable path slower than PyTorch |
| 29 | Transition | inference | bf16 | D512 | 384 | portable_vs_pytorch | 1.119 | 1.587 | 2.618 | 2.340 (pt_compiled) | 1 | portable path slower than PyTorch |
| 30 | pair_embed_geom | inference | bf16 out, fp32 geometry | B48 | 256 | cuda_vs_portable | 1.114 | 1.434 | 1.287 | 13.683 (pt_compiled) | 2 | CUDA kernel slower than the portable kernel |
| 31 | token_pair_init | inference | fp32 in, fp32 out | B48 | 128 | cuda_vs_portable | 1.105 | 0.205 | 0.185 | 1.563 (pt_compiled) | 2 | CUDA kernel slower than the portable kernel |
| 32 | Transition | training | bf16 | D256 | 384 | portable_vs_pytorch | 1.104 | 1.844 | 2.580 | 2.337 (pt_compiled) | 2 | portable path slower than PyTorch |
| 33 | bias-only DiT | inference | fp32 | d768 16x48 A5/A48 | 256 | portable_vs_pytorch | 1.100 | 0.136 | 0.237 | 0.216 (pt_compiled) | 2 | portable path slower than PyTorch |
| 34 | RMSNorm | inference | bf16 | x[48,L,768] | 384 | cuda_vs_portable | 1.090 | 0.031 | 0.028 | 0.024 (pt_compiled) | 2 | CUDA kernel slower than the portable kernel |
| 35 | AdaLN | training | fp32 | d768 cond384 A5/A48 | 384 | portable_vs_pytorch | 1.088 | 0.641 | 0.699 | 0.643 (pt_compiled) | 2 | portable path slower than PyTorch; n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference |
| 36 | token DiT + extra_pair_bias | inference | fp32 | d768 A2 shared-cond +[L,L] bias | 256 | portable_vs_pytorch | 1.088 | 0.205 | 0.224 | 0.206 (pt_compiled) | 1 | portable path slower than PyTorch; n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference |
| 37 | token DiT + extra_pair_bias | training | bf16 | d768 A2 shared-cond +[L,L] bias | 256 | portable_vs_pytorch | 1.084 | 0.419 | 0.463 | 0.426 (pt_compiled) | 2 | portable path slower than PyTorch; n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference |
| 38 | pair_embed_geom | inference | bf16 out, fp32 geometry | B48 | 128 | cuda_vs_portable | 1.083 | 0.374 | 0.345 | 3.560 (pt_compiled) | 2 | CUDA kernel slower than the portable kernel |
| 39 | bias-only DiT | training | bf16 | d768 16x48 A5/A48 | 256 | portable_vs_pytorch | 1.077 | 1.463 | 1.798 | 1.670 (pt_compiled) | 3 | portable path slower than PyTorch |
| 40 | ConditionedTransition | training | bf16 | d768 A5/A48 | 768 | portable_vs_pytorch | 1.075 | 2.300 | 2.473 | 2.301 (pt_compiled) | 2 | portable path slower than PyTorch; n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference |
| 41 | ConditionedTransition | training | bf16 | d768 A5/A48 | 1024 | portable_vs_pytorch | 1.074 | 2.979 | 3.206 | 2.986 (pt_compiled) | 2 | portable path slower than PyTorch; n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference |
| 42 | AttentionPairBias | inference | bf16 | 8x48 pair256 | 384 | cuda_vs_portable | 1.073 | 0.158 | 0.147 | 0.152 (pt_compiled) | 1 | CUDA kernel slower than the portable kernel |
| 43 | bias-only DiT | training | bf16 | 24x32 | 512 | portable_vs_pytorch | 1.073 | 3.054 | 3.578 | 3.336 (pt_compiled) | 1 | portable path slower than PyTorch |
| 44 | bias-only DiT | training | bf16 | d768 16x48 A5/A48 | 512 | portable_vs_pytorch | 1.072 | 2.850 | 3.552 | 3.313 (pt_compiled) | 3 | portable path slower than PyTorch |
| 45 | ConditionedTransition | training | bf16 | d768 A5/A48 | 384 | portable_vs_pytorch | 1.072 | 1.238 | 1.335 | 1.245 (pt_compiled) | 2 | portable path slower than PyTorch; n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference |
| 46 | LayerNorm (pair) | training | bf16 | x[1,L,L,256] | 384 | portable_vs_pytorch | 1.072 | 0.165 | 0.175 | 0.163 (pt_compiled) | 2 | portable path slower than PyTorch |
| 47 | token DiT + extra_pair_bias | training | bf16 | d768 A2 shared-cond +[L,L] bias | 512 | portable_vs_pytorch | 1.068 | 0.699 | 0.748 | 0.701 (pt_compiled) | 2 | portable path slower than PyTorch |
| 48 | Transition | training | fp32 | D256 TF32 | 384 | portable_vs_pytorch | 1.065 | 4.049 | 5.255 | 4.935 (pt_compiled) | 1 | portable path slower than PyTorch |
| 49 | bias-only DiT | inference | bf16 | 12x64 | 512 | portable_vs_pytorch | 1.062 | 0.124 | 0.301 | 0.284 (pt_compiled) | 1 | portable path slower than PyTorch |
| 50 | bias-only DiT | training | bf16 | d768 16x48 A5/A48 | 768 | portable_vs_pytorch | 1.062 | 4.351 | 5.356 | 5.044 (pt_compiled) | 3 | portable path slower than PyTorch |
| 51 | bias-only DiT | training | bf16 | 12x64 | 512 | portable_vs_pytorch | 1.061 | 2.889 | 3.515 | 3.314 (pt_compiled) | 1 | portable path slower than PyTorch |
| 52 | bias-only DiT | inference | bf16 | d768 16x48 A5/A48 | 256 | portable_vs_pytorch | 1.060 | 0.075 | 0.165 | 0.156 (pt_compiled) | 3 | portable path slower than PyTorch |
| 53 | bias-only DiT | inference | bf16 | d768 16x48 A5/A48 | 512 | portable_vs_pytorch | 1.058 | 0.122 | 0.296 | 0.280 (pt_compiled) | 3 | portable path slower than PyTorch |
| 54 | ConditionedTransition | inference | bf16 | d768 A5/A48 | 768 | portable_vs_pytorch | 1.058 | 0.120 | 0.131 | 0.124 (pt_compiled) | 2 | portable path slower than PyTorch |
| 55 | ConditionedTransition | inference | bf16 | d768 A5/A48 | 384 | portable_vs_pytorch | 1.051 | 0.082 | 0.087 | 0.083 (pt_compiled) | 2 | portable path slower than PyTorch; n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference |
| 56 | Transition | inference | bf16 | D384 | 768 | portable_vs_pytorch | 1.050 | 3.783 | 6.263 | 5.963 (pt_compiled) | 2 | portable path slower than PyTorch |
| 57 | bias-only DiT | training | fp32 | d768 16x48 A5/A48 | 512 | portable_vs_pytorch | 1.050 | 6.688 | 7.578 | 7.218 (pt_compiled) | 2 | portable path slower than PyTorch |
| 58 | token_pair_init | inference | fp32 in, fp32 out | B16 | 256 | cuda_vs_portable | 1.047 | 0.233 | 0.222 | 2.101 (pt_compiled) | 3 | CUDA kernel slower than the portable kernel |
| 59 | bias-only DiT | inference | bf16 | 24x32 | 512 | portable_vs_pytorch | 1.043 | 0.126 | 0.293 | 0.281 (pt_compiled) | 1 | portable path slower than PyTorch |
| 60 | bias-only DiT | inference | fp32 | d768 16x48 A5/A48 | 512 | portable_vs_pytorch | 1.042 | 0.231 | 0.453 | 0.435 (pt_compiled) | 2 | portable path slower than PyTorch |
| 61 | Transition | inference | bf16 | D384 | 384 | portable_vs_pytorch | 1.041 | 0.965 | 1.593 | 1.531 (pt_compiled) | 2 | portable path slower than PyTorch; portable_vs_pytorch within noise (ratio 1.041, spread 0.054) |
| 62 | LayerNorm (pair) | inference | bf16 | x[1,L,L,256] | 384 | portable_vs_pytorch | 1.040 | 0.059 | 0.059 | 0.056 (pt_compiled) | 2 | portable path slower than PyTorch; cuda_vs_portable within noise (ratio 1.006, spread 0.002) |
| 63 | bias-only DiT | training | fp32 | d768 16x48 A5/A48 | 256 | portable_vs_pytorch | 1.037 | 3.533 | 4.021 | 3.877 (pt_compiled) | 2 | portable path slower than PyTorch |
| 64 | LayerNorm (pair) | inference | bf16 | x[1,L,L,128] | 384 | portable_vs_pytorch | 1.036 | 0.032 | 0.032 | 0.031 (pt_compiled) | 3 | portable path slower than PyTorch; cuda_vs_portable within noise (ratio 0.998, spread 0.005) |
| 65 | ConditionedTransition | training | fp32 | d768 A5/A48 | 384 | portable_vs_pytorch | 1.035 | 2.894 | 3.004 | 2.902 (pt_compiled) | 2 | portable path slower than PyTorch; n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference |
| 66 | LayerNorm (pair) | inference | fp32 | x[1,L,L,128] | 384 | portable_vs_pytorch | 1.034 | 0.057 | 0.057 | 0.055 (pt_compiled) | 1 | portable path slower than PyTorch; n.a.: cuda_vs_portable: cuda rung launched the portable kernels (no distinct CUDA path) |
| 67 | AttentionPairBias | training | bf16 | 8x48 pair256 | 384 | cuda_vs_portable | 1.033 | 0.419 | 0.406 | 0.423 (pt_compiled) | 1 | CUDA kernel slower than the portable kernel |
| 68 | token_pair_init | inference | fp32 in, fp32 out | B16 | 384 | cuda_vs_portable | 1.031 | 0.479 | 0.464 | 4.751 (pt_compiled) | 3 | CUDA kernel slower than the portable kernel |
| 69 | AdaLN | inference | bf16 | d768 cond384 A5/A48 | 384 | cuda_vs_portable | 1.029 | 0.026 | 0.025 | 0.027 (pt_compiled) | 2 | CUDA kernel slower than the portable kernel; cuda_vs_portable within noise (ratio 1.029, spread 0.050) |
| 70 | bias-only DiT | inference | bf16 | d768 16x48 A5/A48 | 768 | portable_vs_pytorch | 1.028 | 0.173 | 0.441 | 0.429 (pt_compiled) | 3 | portable path slower than PyTorch |
| 71 | OuterProductMean RF3 (bias, divide after) | training | bf16 | S512 msa64 pair128 | 256 | portable_vs_pytorch | 1.016 | 0.867 | 0.887 | 0.873 (pt_compiled) | 1 | portable path slower than PyTorch; portable_vs_pytorch within noise (ratio 1.016, spread 0.010); n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference |
| 72 | LayerNorm (pair) | inference | bf16 | x[1,L,L,128] | 768 | portable_vs_pytorch | 1.010 | 0.107 | 0.107 | 0.106 (pt_compiled) | 3 | portable path slower than PyTorch; portable_vs_pytorch within noise (ratio 1.010, spread 0.002); n.a.: cuda_vs_portable: cuda rung launched the portable kernels (no distinct CUDA path) |
| 73 | LayerNorm (pair) | inference | bf16 | x[1,L,L,256] | 384 | cuda_vs_portable | 1.006 | 0.059 | 0.059 | 0.056 (pt_compiled) | 2 | CUDA kernel slower than the portable kernel; cuda_vs_portable within noise (ratio 1.006, spread 0.002) |
| 74 | OuterProductMean RF3 (bias, divide after) | training | bf16 | S512 msa64 pair128 | 384 | portable_vs_pytorch | 1.004 | 1.801 | 1.816 | 1.809 (pt_compiled) | 1 | portable path slower than PyTorch; portable_vs_pytorch within noise (ratio 1.004, spread 0.004); n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference |
| 75 | token_pair_init | inference | fp32 in, fp32 out | B48 | 256 | cuda_vs_portable | 1.003 | 0.625 | 0.623 | 6.205 (pt_compiled) | 2 | CUDA kernel slower than the portable kernel; cuda_vs_portable within noise (ratio 1.003, spread 0.066) |
| 76 | smooth_lddt | inference | fp32 | A48 B1 | 1024 | cuda_vs_portable | 1.002 | 0.161 | 0.161 | 63.153 (pt_compiled) | 3 | CUDA kernel slower than the portable kernel; cuda_vs_portable within noise (ratio 1.002, spread 0.002) |
| 77 | OuterProductMean RF3 (bias, divide after) | inference | bf16 | S512 msa64 pair128 | 256 | cuda_vs_portable | 1.001 | 0.221 | 0.221 | 0.333 (pt_compiled) | 1 | CUDA kernel slower than the portable kernel; cuda_vs_portable within noise (ratio 1.001, spread 0.008) |
| 78 | token DiT + extra_pair_bias | training | fp32 | d768 A2 shared-cond +[L,L] bias | 256 | cuda_vs_portable | 1.000 | 0.684 | 0.683 | 0.684 (pt_compiled) | 1 | CUDA kernel slower than the portable kernel; cuda_vs_portable within noise (ratio 1.000, spread 0.010); portable_vs_pytorch within noise (ratio 0.999, spread 0.010) |

### Violations ranked by importance (production weight, then size)

| # | weight | op | mode | dtype | shape | L | pair | ratio |
|---|---|---|---|---|---|---|---|---|
| 1 | 3 | pair_embed_ca | training | bf16 out, fp32 geometry | B16 | 384 | cuda_vs_portable | 1.714 |
| 2 | 3 | pair_embed_ca | training | bf16 out, fp32 geometry | B16 | 256 | cuda_vs_portable | 1.590 |
| 3 | 3 | pair_embed_geom | training | bf16 out, fp32 geometry | B16 | 384 | cuda_vs_portable | 1.384 |
| 4 | 3 | pair_embed_geom | training | bf16 out, fp32 geometry | B16 | 256 | cuda_vs_portable | 1.318 |
| 5 | 3 | pair_embed_ca | training | bf16 out, fp32 geometry | B16 | 128 | cuda_vs_portable | 1.308 |
| 6 | 3 | LayerNorm (pair) | training | bf16 | x[1,L,L,128] | 768 | cuda_vs_portable | 1.217 |
| 7 | 3 | LayerNorm (pair) | training | bf16 | x[1,L,L,128] | 384 | cuda_vs_portable | 1.184 |
| 8 | 3 | pair_embed_geom | inference | bf16 out, fp32 geometry | B16 | 128 | cuda_vs_portable | 1.179 |
| 9 | 3 | pair_embed_geom | inference | bf16 out, fp32 geometry | B16 | 384 | cuda_vs_portable | 1.173 |
| 10 | 3 | pair_embed_geom | inference | bf16 out, fp32 geometry | B16 | 256 | cuda_vs_portable | 1.168 |
| 11 | 3 | pair_embed_geom | training | bf16 out, fp32 geometry | B16 | 128 | cuda_vs_portable | 1.158 |
| 12 | 3 | token_pair_init | inference | fp32 in, fp32 out | B16 | 128 | cuda_vs_portable | 1.133 |
| 13 | 3 | bias-only DiT | training | bf16 | d768 16x48 A5/A48 | 256 | portable_vs_pytorch | 1.077 |
| 14 | 3 | bias-only DiT | training | bf16 | d768 16x48 A5/A48 | 512 | portable_vs_pytorch | 1.072 |
| 15 | 3 | bias-only DiT | training | bf16 | d768 16x48 A5/A48 | 768 | portable_vs_pytorch | 1.062 |
| 16 | 3 | bias-only DiT | inference | bf16 | d768 16x48 A5/A48 | 256 | portable_vs_pytorch | 1.060 |
| 17 | 3 | bias-only DiT | inference | bf16 | d768 16x48 A5/A48 | 512 | portable_vs_pytorch | 1.058 |
| 18 | 3 | token_pair_init | inference | fp32 in, fp32 out | B16 | 256 | cuda_vs_portable | 1.047 |
| 19 | 3 | LayerNorm (pair) | inference | bf16 | x[1,L,L,128] | 384 | portable_vs_pytorch | 1.036 |
| 20 | 3 | token_pair_init | inference | fp32 in, fp32 out | B16 | 384 | cuda_vs_portable | 1.031 |
| 21 | 3 | bias-only DiT | inference | bf16 | d768 16x48 A5/A48 | 768 | portable_vs_pytorch | 1.028 |
| 22 | 3 | LayerNorm (pair) | inference | bf16 | x[1,L,L,128] | 768 | portable_vs_pytorch | 1.010 |
| 23 | 3 | smooth_lddt | inference | fp32 | A48 B1 | 1024 | cuda_vs_portable | 1.002 |
| 24 | 2 | pair_embed_ca | training | bf16 out, fp32 geometry | B48 | 256 | cuda_vs_portable | 1.726 |
| 25 | 2 | pair_embed_ca | training | bf16 out, fp32 geometry | B48 | 128 | cuda_vs_portable | 1.620 |
| 26 | 2 | RMSNorm | inference | bf16 | x[48,L,768] | 1024 | portable_vs_pytorch | 1.421 |
| 27 | 2 | Transition | training | bf16 | D768 | 384 | portable_vs_pytorch | 1.351 |
| 28 | 2 | pair_embed_geom | training | bf16 out, fp32 geometry | B48 | 256 | cuda_vs_portable | 1.336 |
| 29 | 2 | pair_embed_geom | training | bf16 out, fp32 geometry | B48 | 128 | cuda_vs_portable | 1.283 |
| 30 | 2 | RMSNorm | training | bf16 | x[48,L,768] | 1024 | portable_vs_pytorch | 1.281 |
| 31 | 2 | Transition | training | bf16 | D384 | 768 | portable_vs_pytorch | 1.243 |
| 32 | 2 | Transition | inference | bf16 | D768 | 384 | portable_vs_pytorch | 1.240 |
| 33 | 2 | RMSNorm | training | bf16 | x[48,L,768] | 384 | portable_vs_pytorch | 1.230 |
| 34 | 2 | Transition | training | bf16 | D384 | 384 | portable_vs_pytorch | 1.213 |
| 35 | 2 | AdaLN | inference | fp32 | d768 cond384 A5/A48 | 384 | portable_vs_pytorch | 1.176 |
| 36 | 2 | RMSNorm | inference | bf16 | x[48,L,768] | 384 | portable_vs_pytorch | 1.165 |
| 37 | 2 | ConditionedTransition | inference | bf16 | d768 A5/A48 | 1024 | portable_vs_pytorch | 1.153 |
| 38 | 2 | Transition | training | bf16 | D256 | 768 | portable_vs_pytorch | 1.127 |
| 39 | 2 | pair_embed_geom | inference | bf16 out, fp32 geometry | B48 | 256 | cuda_vs_portable | 1.114 |
| 40 | 2 | token_pair_init | inference | fp32 in, fp32 out | B48 | 128 | cuda_vs_portable | 1.105 |
| 41 | 2 | Transition | training | bf16 | D256 | 384 | portable_vs_pytorch | 1.104 |
| 42 | 2 | bias-only DiT | inference | fp32 | d768 16x48 A5/A48 | 256 | portable_vs_pytorch | 1.100 |
| 43 | 2 | RMSNorm | inference | bf16 | x[48,L,768] | 384 | cuda_vs_portable | 1.090 |
| 44 | 2 | AdaLN | training | fp32 | d768 cond384 A5/A48 | 384 | portable_vs_pytorch | 1.088 |
| 45 | 2 | token DiT + extra_pair_bias | training | bf16 | d768 A2 shared-cond +[L,L] bias | 256 | portable_vs_pytorch | 1.084 |
| 46 | 2 | pair_embed_geom | inference | bf16 out, fp32 geometry | B48 | 128 | cuda_vs_portable | 1.083 |
| 47 | 2 | ConditionedTransition | training | bf16 | d768 A5/A48 | 768 | portable_vs_pytorch | 1.075 |
| 48 | 2 | ConditionedTransition | training | bf16 | d768 A5/A48 | 1024 | portable_vs_pytorch | 1.074 |
| 49 | 2 | ConditionedTransition | training | bf16 | d768 A5/A48 | 384 | portable_vs_pytorch | 1.072 |
| 50 | 2 | LayerNorm (pair) | training | bf16 | x[1,L,L,256] | 384 | portable_vs_pytorch | 1.072 |
| 51 | 2 | token DiT + extra_pair_bias | training | bf16 | d768 A2 shared-cond +[L,L] bias | 512 | portable_vs_pytorch | 1.068 |
| 52 | 2 | ConditionedTransition | inference | bf16 | d768 A5/A48 | 768 | portable_vs_pytorch | 1.058 |
| 53 | 2 | ConditionedTransition | inference | bf16 | d768 A5/A48 | 384 | portable_vs_pytorch | 1.051 |
| 54 | 2 | Transition | inference | bf16 | D384 | 768 | portable_vs_pytorch | 1.050 |
| 55 | 2 | bias-only DiT | training | fp32 | d768 16x48 A5/A48 | 512 | portable_vs_pytorch | 1.050 |
| 56 | 2 | bias-only DiT | inference | fp32 | d768 16x48 A5/A48 | 512 | portable_vs_pytorch | 1.042 |
| 57 | 2 | Transition | inference | bf16 | D384 | 384 | portable_vs_pytorch | 1.041 |
| 58 | 2 | LayerNorm (pair) | inference | bf16 | x[1,L,L,256] | 384 | portable_vs_pytorch | 1.040 |
| 59 | 2 | bias-only DiT | training | fp32 | d768 16x48 A5/A48 | 256 | portable_vs_pytorch | 1.037 |
| 60 | 2 | ConditionedTransition | training | fp32 | d768 A5/A48 | 384 | portable_vs_pytorch | 1.035 |
| 61 | 2 | AdaLN | inference | bf16 | d768 cond384 A5/A48 | 384 | cuda_vs_portable | 1.029 |
| 62 | 2 | LayerNorm (pair) | inference | bf16 | x[1,L,L,256] | 384 | cuda_vs_portable | 1.006 |
| 63 | 2 | token_pair_init | inference | fp32 in, fp32 out | B48 | 256 | cuda_vs_portable | 1.003 |
| 64 | 1 | Transition | training | bf16 | D512 | 384 | portable_vs_pytorch | 1.270 |
| 65 | 1 | Transition | inference | bf16 | D512 | 384 | portable_vs_pytorch | 1.119 |
| 66 | 1 | token DiT + extra_pair_bias | inference | fp32 | d768 A2 shared-cond +[L,L] bias | 256 | portable_vs_pytorch | 1.088 |
| 67 | 1 | AttentionPairBias | inference | bf16 | 8x48 pair256 | 384 | cuda_vs_portable | 1.073 |
| 68 | 1 | bias-only DiT | training | bf16 | 24x32 | 512 | portable_vs_pytorch | 1.073 |
| 69 | 1 | Transition | training | fp32 | D256 TF32 | 384 | portable_vs_pytorch | 1.065 |
| 70 | 1 | bias-only DiT | inference | bf16 | 12x64 | 512 | portable_vs_pytorch | 1.062 |
| 71 | 1 | bias-only DiT | training | bf16 | 12x64 | 512 | portable_vs_pytorch | 1.061 |
| 72 | 1 | bias-only DiT | inference | bf16 | 24x32 | 512 | portable_vs_pytorch | 1.043 |
| 73 | 1 | LayerNorm (pair) | inference | fp32 | x[1,L,L,128] | 384 | portable_vs_pytorch | 1.034 |
| 74 | 1 | AttentionPairBias | training | bf16 | 8x48 pair256 | 384 | cuda_vs_portable | 1.033 |
| 75 | 1 | OuterProductMean RF3 (bias, divide after) | training | bf16 | S512 msa64 pair128 | 256 | portable_vs_pytorch | 1.016 |
| 76 | 1 | OuterProductMean RF3 (bias, divide after) | training | bf16 | S512 msa64 pair128 | 384 | portable_vs_pytorch | 1.004 |
| 77 | 1 | OuterProductMean RF3 (bias, divide after) | inference | bf16 | S512 msa64 pair128 | 256 | cuda_vs_portable | 1.001 |
| 78 | 1 | token DiT + extra_pair_bias | training | fp32 | d768 A2 shared-cond +[L,L] bias | 256 | cuda_vs_portable | 1.000 |

### Violation detail: where the time goes (top GPU kernels of one call, microseconds x calls)

`torch.profiler` over one call of the timed step, in the regime of the row. A rung that launches many small kernels, or spends its time in a cuBLAS GEMM the other rung fuses, shows it here.

* **pair_embed_ca training bf16 out, fp32 geometry B16 L=384**, cuda_vs_portable ratio 1.714
  * cuda: `void pg_bwd_mma<1>(__nv_bfloat16 const*,` 1812us x1, `void pg_ln_bwd<1>(__nv_bfloat16 const*, ` 1260us x1, `pg_fwd_ca(__nv_bfloat16 const*, float co` 594us x1, `void at::native::unrolled_elementwise_ke` 4us x1, `void at::native::unrolled_elementwise_ke` 4us x1
  * portable: `_geom_ln_bwd` 830us x1, `_geom_dw` 772us x1, `_geom_fwd_gather` 427us x1, `_geom_dw_reduce` 75us x1, `_ca_bins` 22us x1
* **pair_embed_ca training bf16 out, fp32 geometry B16 L=256**, cuda_vs_portable ratio 1.590
  * cuda: `void pg_bwd_mma<1>(__nv_bfloat16 const*,` 809us x1, `void pg_ln_bwd<1>(__nv_bfloat16 const*, ` 556us x1, `pg_fwd_ca(__nv_bfloat16 const*, float co` 268us x1, `void at::native::unrolled_elementwise_ke` 4us x1, `void at::native::unrolled_elementwise_ke` 4us x1
  * portable: `_geom_dw` 374us x1, `_geom_ln_bwd` 364us x1, `_geom_fwd_gather` 192us x1, `_geom_dw_reduce` 73us x1, `_ca_bins` 12us x1
* **pair_embed_geom training bf16 out, fp32 geometry B16 L=384**, cuda_vs_portable ratio 1.384
  * cuda: `void pg_bwd_mma<0>(__nv_bfloat16 const*,` 2964us x1, `pg_fwd(__nv_bfloat16 const*, float const` 1756us x1, `void pg_ln_bwd<0>(__nv_bfloat16 const*, ` 1526us x1, `void at::native::unrolled_elementwise_ke` 11us x3, `void at::native::unrolled_elementwise_ke` 5us x2
  * portable: `_geom_dw` 2015us x1, `_geom_ln_bwd` 1322us x1, `_geom_fwd_gather` 1042us x1, `_geom_bins` 151us x1, `_geom_dw_reduce` 81us x1
* **pair_embed_geom training bf16 out, fp32 geometry B16 L=256**, cuda_vs_portable ratio 1.318
  * cuda: `void pg_bwd_mma<0>(__nv_bfloat16 const*,` 1330us x1, `pg_fwd(__nv_bfloat16 const*, float const` 790us x1, `void pg_ln_bwd<0>(__nv_bfloat16 const*, ` 658us x1, `void at::native::unrolled_elementwise_ke` 11us x3, `void at::native::unrolled_elementwise_ke` 4us x2
  * portable: `_geom_dw` 918us x1, `_geom_ln_bwd` 592us x1, `_geom_fwd_gather` 462us x1, `_geom_dw_reduce` 83us x1, `_geom_bins` 68us x1
* **pair_embed_ca training bf16 out, fp32 geometry B16 L=128**, cuda_vs_portable ratio 1.308
  * cuda: `void pg_bwd_mma<1>(__nv_bfloat16 const*,` 208us x1, `void pg_ln_bwd<1>(__nv_bfloat16 const*, ` 146us x1, `pg_fwd_ca(__nv_bfloat16 const*, float co` 72us x1, `void at::native::unrolled_elementwise_ke` 4us x1, `void at::native::unrolled_elementwise_ke` 3us x1
  * portable: `_geom_dw` 109us x1, `_geom_ln_bwd` 91us x1, `_geom_dw_reduce` 67us x1, `_geom_fwd_gather` 48us x1, `_ca_bins` 5us x1
* **LayerNorm (pair) training bf16 x[1,L,L,128] L=768**, cuda_vs_portable ratio 1.217
  * cuda: `void layer_norm_bwd_reduce_kernel<c10::B` 6us x1
  * portable: `layer_norm_bwd_dx_fused` 186us x1, `layer_norm_fwd_fused` 98us x1, `void at::native::vectorized_elementwise_` 4us x2, `void at::native::vectorized_elementwise_` 3us x2
* **LayerNorm (pair) training bf16 x[1,L,L,128] L=384**, cuda_vs_portable ratio 1.184
  * cuda: `void layer_norm_bwd_reduce_kernel<c10::B` 6us x1
  * portable: `layer_norm_bwd_dx_fused` 52us x1, `void at::native::vectorized_elementwise_` 3us x2, `void at::native::vectorized_elementwise_` 2us x1
* **pair_embed_geom inference bf16 out, fp32 geometry B16 L=128**, cuda_vs_portable ratio 1.179
  * cuda: `pg_fwd(__nv_bfloat16 const*, float const` 139us x1, `void at::native::unrolled_elementwise_ke` 11us x3, `void at::native::unrolled_elementwise_ke` 5us x2, `void at::native::elementwise_kernel<128,` 3us x1, `void (anonymous namespace)::elementwise_` 3us x2
  * portable: `_geom_fwd_gather` 95us x1, `_geom_bins` 22us x1, `void at::native::unrolled_elementwise_ke` 11us x3, `void at::native::unrolled_elementwise_ke` 5us x2, `void at::native::elementwise_kernel<128,` 3us x1
* **pair_embed_geom inference bf16 out, fp32 geometry B16 L=384**, cuda_vs_portable ratio 1.173
  * cuda: `pg_fwd(__nv_bfloat16 const*, float const` 1128us x1, `void at::native::unrolled_elementwise_ke` 11us x3, `void at::native::unrolled_elementwise_ke` 5us x2, `void at::native::elementwise_kernel<128,` 3us x1, `void (anonymous namespace)::elementwise_` 3us x2
  * portable: `_geom_fwd_gather` 821us x1, `_geom_bins` 150us x1, `void at::native::unrolled_elementwise_ke` 11us x3, `void at::native::unrolled_elementwise_ke` 5us x2, `void at::native::elementwise_kernel<128,` 3us x1
* **pair_embed_geom inference bf16 out, fp32 geometry B16 L=256**, cuda_vs_portable ratio 1.168
  * cuda: `pg_fwd(__nv_bfloat16 const*, float const` 506us x1, `void at::native::unrolled_elementwise_ke` 11us x3, `void at::native::unrolled_elementwise_ke` 4us x2, `void at::native::elementwise_kernel<128,` 3us x1, `void (anonymous namespace)::elementwise_` 3us x2
  * portable: `_geom_fwd_gather` 369us x1, `_geom_bins` 68us x1, `void at::native::unrolled_elementwise_ke` 11us x3, `void at::native::unrolled_elementwise_ke` 5us x2, `void at::native::elementwise_kernel<128,` 3us x1
* **pair_embed_geom training bf16 out, fp32 geometry B16 L=128**, cuda_vs_portable ratio 1.158
  * cuda: `void pg_bwd_mma<0>(__nv_bfloat16 const*,` 339us x1, `pg_fwd(__nv_bfloat16 const*, float const` 211us x1, `void pg_ln_bwd<0>(__nv_bfloat16 const*, ` 175us x1, `void at::native::unrolled_elementwise_ke` 10us x3, `void at::native::unrolled_elementwise_ke` 5us x2
  * portable: `_geom_dw` 252us x1, `_geom_ln_bwd` 154us x1, `_geom_fwd_gather` 119us x1, `_geom_dw_reduce` 88us x1, `_geom_bins` 23us x1
* **token_pair_init inference fp32 in, fp32 out B16 L=128**, cuda_vs_portable ratio 1.133
  * cuda: `tpi_fwd(float const*, float const*, floa` 67us x1, `Memcpy DtoD (Device -> Device)` 7us x4, `void at::native::unrolled_elementwise_ke` 5us x1, `void at::native::unrolled_elementwise_ke` 4us x1, `void at::native::unrolled_elementwise_ke` 3us x1
  * portable: `_tpi_fwd` 56us x1, `Memcpy DtoD (Device -> Device)` 7us x4, `void at::native::unrolled_elementwise_ke` 4us x1, `void at::native::unrolled_elementwise_ke` 4us x1, `void at::native::unrolled_elementwise_ke` 3us x1
* **bias-only DiT training bf16 d768 16x48 A5/A48 L=256**, portable_vs_pytorch ratio 1.077
  * portable: `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TN` 203us x5, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NN` 180us x6, `_dh_swiglu_bwd_kernel` 130us x1, `_bwd_x_kernel` 119us x2, `_adaln_fwd_gate_kernel` 105us x2
  * pt_compiled: `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NN` 293us x12, `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TN` 237us x8, `nvjet_tss_64x64_64x13_2x4_h_bz_NTT` 116us x5, `nvjet_tst_160x64_64x6_1x4_h_bz_NTN` 95us x2, `nvjet_tst_96x64_64x8_2x4_h_bz_NTN` 95us x3
* **bias-only DiT training bf16 d768 16x48 A5/A48 L=512**, portable_vs_pytorch ratio 1.072
  * portable: `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TN` 387us x5, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NN` 350us x6, `_dh_swiglu_bwd_kernel` 254us x1, `_bwd_x_kernel` 246us x2, `_adaln_fwd_gate_kernel` 204us x2
  * pt_compiled: `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NN` 558us x12, `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TN` 441us x8, `nvjet_tst_128x160_64x5_2x2_h_bz_splitK_N` 162us x2, `triton_per_fused__to_copy__unsafe_view_a` 155us x1, `nvjet_tss_96x128_64x6_4x2_h_bz_splitK_NT` 152us x5
* **bias-only DiT training bf16 d768 16x48 A5/A48 L=768**, portable_vs_pytorch ratio 1.062
  * portable: `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TN` 623us x6, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NN` 585us x7, `_dh_swiglu_bwd_kernel` 376us x1, `_bwd_x_kernel` 367us x2, `nvjet_tss_192x192_64x3_2x1_v_bz_coopB_sp` 354us x3
  * pt_compiled: `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NN` 898us x13, `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TN` 723us x9, `triton_per_fused__to_copy__unsafe_view_a` 233us x1, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_sp` 231us x2, `nvjet_tss_96x128_64x6_4x2_h_bz_splitK_NT` 202us x5
* **bias-only DiT inference bf16 d768 16x48 A5/A48 L=256**, portable_vs_pytorch ratio 1.060
  * portable: `_squeeze_gate_kernel` 18us x1, `nvjet_tst_128x64_64x8_2x4_h_bz_TNT` 16us x3, `_expand_swiglu_kernel` 16us x1, `_adaln_gemm_gate_kernel` 15us x2, `layer_norm_fwd_fused` 14us x1
  * pt_compiled: `nvjet_tst_128x64_64x8_2x4_h_bz_TNT` 33us x6, `nvjet_tst_128x64_64x8_2x4_h_bz_bias_TNT` 20us x4, `nvjet_tst_128x128_64x6_4x2_h_bz_TNT` 15us x2, `triton_poi_fused__to_copy_1` 14us x6, `triton_poi_fused__to_copy_13` 12us x3
* **bias-only DiT inference bf16 d768 16x48 A5/A48 L=512**, portable_vs_pytorch ratio 1.058
  * portable: `_expand_swiglu_kernel` 34us x1, `nvjet_tst_16x512_64x3_1x2_h_bz_TNN` 32us x1, `_squeeze_gate_kernel` 32us x1, `nvjet_tst_128x128_64x6_2x4_h_bz_TNT` 24us x3, `_adaln_gemm_gate_kernel` 23us x2
  * pt_compiled: `triton_per_fused__to_copy_native_layer_n` 47us x1, `nvjet_tst_128x128_64x6_2x4_h_bz_TNT` 47us x6, `nvjet_tst_16x512_64x3_1x2_h_bz_TNN` 30us x1, `nvjet_tst_128x128_64x6_2x4_h_bz_bias_TNT` 26us x4, `nvjet_tst_256x128_64x4_2x4_h_bz_coopA_TN` 25us x2
* **token_pair_init inference fp32 in, fp32 out B16 L=256**, cuda_vs_portable ratio 1.047
  * cuda: `tpi_fwd(float const*, float const*, floa` 205us x1, `Memcpy DtoD (Device -> Device)` 7us x4, `void at::native::unrolled_elementwise_ke` 5us x1, `void at::native::unrolled_elementwise_ke` 4us x1, `void at::native::unrolled_elementwise_ke` 4us x1
  * portable: `_tpi_fwd` 208us x1, `Memcpy DtoD (Device -> Device)` 7us x4, `void at::native::unrolled_elementwise_ke` 5us x1, `void at::native::unrolled_elementwise_ke` 4us x1, `void at::native::unrolled_elementwise_ke` 4us x1
* **LayerNorm (pair) inference bf16 x[1,L,L,128] L=384**, portable_vs_pytorch ratio 1.036
  * portable: (no profile)
  * pt_compiled: `triton_per_fused_native_layer_norm_0` 23us x1
* **token_pair_init inference fp32 in, fp32 out B16 L=384**, cuda_vs_portable ratio 1.031
  * cuda: `tpi_fwd(float const*, float const*, floa` 442us x1, `void at::native::unrolled_elementwise_ke` 10us x1, `Memcpy DtoD (Device -> Device)` 7us x4, `void at::native::unrolled_elementwise_ke` 7us x1, `void at::native::unrolled_elementwise_ke` 4us x1
  * portable: `_tpi_fwd` 459us x1, `void at::native::unrolled_elementwise_ke` 10us x1, `void at::native::unrolled_elementwise_ke` 7us x1, `Memcpy DtoD (Device -> Device)` 5us x3, `void at::native::unrolled_elementwise_ke` 4us x1
* **bias-only DiT inference bf16 d768 16x48 A5/A48 L=768**, portable_vs_pytorch ratio 1.028
  * portable: `nvjet_tst_16x384_64x4_1x2_h_bz_TNN` 62us x1, `_expand_swiglu_kernel` 41us x1, `_squeeze_gate_kernel` 37us x1, `_adaln_gemm_gate_kernel` 34us x2, `triton_red_fused__softmax__to_copy__unsa` 33us x1
  * pt_compiled: `triton_per_fused__to_copy_native_layer_n` 101us x1, `nvjet_tst_192x128_64x5_4x2_h_bz_coopB_TN` 63us x6, `nvjet_tst_16x384_64x4_1x2_h_bz_TNN` 62us x1, `nvjet_tst_128x192_64x5_4x2_h_bz_coopB_TN` 35us x2, `nvjet_tst_192x128_64x5_4x2_h_bz_coopB_bi` 33us x4
* **LayerNorm (pair) inference bf16 x[1,L,L,128] L=768**, portable_vs_pytorch ratio 1.010
  * portable: `layer_norm_fwd_fused` 99us x1
  * pt_compiled: (no profile)
* **smooth_lddt inference fp32 A48 B1 L=1024**, cuda_vs_portable ratio 1.002
  * cuda: `slddt_fwd_full` 128us x1, `slddt_prep` 9us x1, `void at::native::unrolled_elementwise_ke` 5us x2, `void at::native::unrolled_elementwise_ke` 3us x1, `void at::native::reduce_kernel<512, 1, a` 3us x1
  * portable: `smooth_lddt_fwd_kernel` 133us x1, `void at::native::unrolled_elementwise_ke` 5us x2, `void at::native::reduce_kernel<512, 1, a` 5us x1, `smooth_lddt_prep_kernel` 4us x1, `void at::native::unrolled_elementwise_ke` 3us x1
* **pair_embed_ca training bf16 out, fp32 geometry B48 L=256**, cuda_vs_portable ratio 1.726
  * cuda: `void pg_bwd_mma<1>(__nv_bfloat16 const*,` 2337us x1, `void pg_ln_bwd<1>(__nv_bfloat16 const*, ` 1602us x1, `pg_fwd_ca(__nv_bfloat16 const*, float co` 776us x1, `void at::native::unrolled_elementwise_ke` 4us x1, `pg_reduce_tbl(float const*, __nv_bfloat1` 4us x1
  * portable: `_geom_ln_bwd` 1049us x1, `_geom_dw` 1018us x1, `_geom_fwd_gather` 572us x1, `_geom_dw_reduce` 74us x1, `_ca_bins` 28us x1
* **pair_embed_ca training bf16 out, fp32 geometry B48 L=128**, cuda_vs_portable ratio 1.620
  * cuda: `void pg_bwd_mma<1>(__nv_bfloat16 const*,` 604us x1, `void pg_ln_bwd<1>(__nv_bfloat16 const*, ` 414us x1, `pg_fwd_ca(__nv_bfloat16 const*, float co` 196us x1, `void at::native::unrolled_elementwise_ke` 4us x1, `void at::native::unrolled_elementwise_ke` 4us x1
  * portable: `_geom_dw` 270us x1, `_geom_ln_bwd` 259us x1, `_geom_fwd_gather` 142us x1, `_geom_dw_reduce` 73us x1, `_ca_bins` 10us x1
* **RMSNorm inference bf16 x[48,L,768] L=1024**, portable_vs_pytorch ratio 1.421
  * portable: (no profile)
  * pt_compiled: (no profile)
* **Transition training bf16 D768 L=384**, portable_vs_pytorch ratio 1.351
  * portable: `_transition_expand_gatebwd_kernel` 4241us x1, `transition_fwd_kernel` 2858us x1, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NN` 2828us x2, `_squeeze_residual_kernel` 2017us x1, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NT` 1762us x1
  * pt_compiled: `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NN` 2850us x3, `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TN` 2845us x3, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_sp` 2634us x3, `triton_poi_fused__unsafe_view_add_mul_si` 1474us x1, `triton_poi_fused__unsafe_view_mul_sigmoi` 877us x1
* **pair_embed_geom training bf16 out, fp32 geometry B48 L=256**, cuda_vs_portable ratio 1.336
  * cuda: `void pg_bwd_mma<0>(__nv_bfloat16 const*,` 3893us x1, `pg_fwd(__nv_bfloat16 const*, float const` 2246us x1, `void pg_ln_bwd<0>(__nv_bfloat16 const*, ` 1879us x1, `void at::native::unrolled_elementwise_ke` 11us x3, `void at::native::unrolled_elementwise_ke` 5us x2
  * portable: `_geom_dw` 2650us x1, `_geom_ln_bwd` 1758us x1, `_geom_fwd_gather` 1384us x1, `_geom_bins` 180us x1, `_geom_dw_reduce` 82us x1
* **pair_embed_geom training bf16 out, fp32 geometry B48 L=128**, cuda_vs_portable ratio 1.283
  * cuda: `void pg_bwd_mma<0>(__nv_bfloat16 const*,` 1000us x1, `pg_fwd(__nv_bfloat16 const*, float const` 569us x1, `void pg_ln_bwd<0>(__nv_bfloat16 const*, ` 494us x1, `void at::native::unrolled_elementwise_ke` 11us x3, `void at::native::unrolled_elementwise_ke` 5us x2
  * portable: `_geom_dw` 684us x1, `_geom_ln_bwd` 446us x1, `_geom_fwd_gather` 348us x1, `_geom_dw_reduce` 87us x1, `_geom_bins` 52us x1
* **RMSNorm training bf16 x[48,L,768] L=1024**, portable_vs_pytorch ratio 1.281
  * portable: `rmsnorm_bwd_kernel` 128us x1, `rmsnorm_fwd_kernel` 68us x1, `void at::native::vectorized_elementwise_` 2us x1, `void at::native::vectorized_elementwise_` 2us x1
  * pt_compiled: `triton_per_fused__fused_rms_norm__fused_` 98us x1, `void at::native::reduce_kernel<128, 4, a` 12us x1, `triton_poi_fused__fused_rms_norm_backwar` 2us x1, `Memset (Device)` 1us x1
* **Transition training bf16 D384 L=768**, portable_vs_pytorch ratio 1.243
  * portable: `_transition_expand_gatebwd_kernel` 5748us x1, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NN` 3195us x2, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_sp` 2793us x2, `_ln_bwd_residual_kernel` 850us x1, `void cublasLt::splitKreduce_kernel<32, 1` 16us x2
  * pt_compiled: `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NN` 3253us x3, `triton_poi_fused__unsafe_view_add_mul_si` 2942us x1, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_sp` 2797us x3, `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TN` 2105us x2, `triton_poi_fused__unsafe_view_mul_sigmoi` 1752us x1
* **Transition inference bf16 D768 L=384**, portable_vs_pytorch ratio 1.240
  * portable: `transition_fwd_kernel` 2993us x1, `_squeeze_residual_kernel` 1967us x1, `layer_norm_fwd_fused` 165us x1, `void at::native::vectorized_elementwise_` 17us x3
  * pt_compiled: `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TN` 2846us x3, `triton_poi_fused__unsafe_view_mul_sigmoi` 875us x1, `triton_per_fused__to_copy_native_layer_n` 228us x1, `triton_poi_fused__unsafe_view_add_3` 217us x1, `triton_poi_fused__to_copy_1` 25us x3
* **RMSNorm training bf16 x[48,L,768] L=384**, portable_vs_pytorch ratio 1.230
  * portable: `rmsnorm_bwd_kernel` 53us x1, `rmsnorm_fwd_kernel` 21us x1, `void at::native::vectorized_elementwise_` 2us x1, `void at::native::vectorized_elementwise_` 2us x1
  * pt_compiled: `triton_per_fused__fused_rms_norm__fused_` 34us x1, `triton_per_fused__fused_rms_norm_0` 16us x1, `void at::native::reduce_kernel<128, 4, a` 10us x1, `triton_poi_fused__fused_rms_norm_backwar` 2us x1, `Memset (Device)` 1us x1
* **Transition training bf16 D384 L=384**, portable_vs_pytorch ratio 1.213
  * portable: `_transition_expand_gatebwd_kernel` 1381us x1, `transition_fwd_kernel` 842us x1, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NN` 812us x2, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_sp` 702us x2, `_squeeze_residual_kernel` 557us x1
  * pt_compiled: `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NN` 848us x3, `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TN` 816us x3, `triton_poi_fused__unsafe_view_add_mul_si` 737us x1, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_sp` 702us x3, `triton_poi_fused__unsafe_view_mul_sigmoi` 440us x1
* **AdaLN inference fp32 d768 cond384 A5/A48 L=384**, portable_vs_pytorch ratio 1.176
  * portable: `_adaln_gemm_gate_kernel` 25us x1, `_ln_kernel` 5us x1, `triton_poi_fused_adaln_gemm_gate_clone_t` 5us x2, `_stats_kernel` 4us x1
  * pt_compiled: `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_t` 20us x2, `triton_per_fused__unsafe_view_add_addmm_` 7us x1, `triton_per_fused_native_layer_norm_0` 3us x1
* **RMSNorm inference bf16 x[48,L,768] L=384**, portable_vs_pytorch ratio 1.165
  * portable: (no profile)
  * pt_compiled: `triton_per_fused__fused_rms_norm_0` 17us x1
* **ConditionedTransition inference bf16 d768 A5/A48 L=1024**, portable_vs_pytorch ratio 1.153
  * portable: `_expand_swiglu_kernel` 58us x1, `_squeeze_gate_kernel` 53us x1, `_adaln_gemm_gate_kernel` 20us x1, `triton_poi_fused__to_copy_conditioned_tr` 12us x3, `triton_poi_fused_add_view_4` 7us x1
  * pt_compiled: `nvjet_tst_128x160_64x5_2x1_v_bz_TNT` 37us x2, `nvjet_tst_256x128_64x4_1x4_h_bz_coopA_TN` 28us x2, `nvjet_tst_256x128_64x4_1x4_h_bz_coopA_bi` 18us x2, `triton_poi_fused__to_copy_4` 15us x3, `triton_poi_fused__unsafe_view_mul_silu_5` 12us x1
* **Transition training bf16 D256 L=768**, portable_vs_pytorch ratio 1.127
  * portable: `_transition_expand_gatebwd_kernel` 3149us x1, `transition_fwd_kernel` 1977us x1, `_squeeze_residual_kernel` 1061us x1, `nvjet_tst_128x256_64x4_2x1_v_bz_coopA_NN` 948us x1, `nvjet_tst_128x256_64x4_2x1_v_bz_coopA_sp` 860us x1
  * pt_compiled: `triton_poi_fused__unsafe_view_add_mul_si` 1964us x1, `triton_poi_fused__unsafe_view_mul_sigmoi` 1169us x1, `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TN` 1131us x2, `nvjet_tst_128x256_64x4_2x1_v_bz_coopA_sp` 1033us x2, `nvjet_tst_128x256_64x4_2x1_v_bz_coopA_NN` 1005us x2
* **pair_embed_geom inference bf16 out, fp32 geometry B48 L=256**, cuda_vs_portable ratio 1.114
  * cuda: `pg_fwd(__nv_bfloat16 const*, float const` 1415us x1, `void at::native::unrolled_elementwise_ke` 11us x3, `void at::native::unrolled_elementwise_ke` 5us x2, `void at::native::elementwise_kernel<128,` 3us x1, `void (anonymous namespace)::elementwise_` 3us x2
  * portable: `_geom_fwd_gather` 1100us x1, `_geom_bins` 180us x1, `void at::native::unrolled_elementwise_ke` 11us x3, `void at::native::unrolled_elementwise_ke` 5us x2, `void at::native::elementwise_kernel<128,` 3us x1
* **token_pair_init inference fp32 in, fp32 out B48 L=128**, cuda_vs_portable ratio 1.105
  * cuda: `tpi_fwd(float const*, float const*, floa` 184us x1, `Memcpy DtoD (Device -> Device)` 7us x4, `void at::native::unrolled_elementwise_ke` 4us x1, `void at::native::unrolled_elementwise_ke` 4us x1, `void at::native::unrolled_elementwise_ke` 4us x1
  * portable: `_tpi_fwd` 166us x1, `Memcpy DtoD (Device -> Device)` 7us x4, `void at::native::unrolled_elementwise_ke` 4us x1, `void at::native::unrolled_elementwise_ke` 4us x1, `void at::native::unrolled_elementwise_ke` 4us x1
* **Transition training bf16 D256 L=384**, portable_vs_pytorch ratio 1.104
  * portable: `_transition_expand_gatebwd_kernel` 730us x1, `transition_fwd_kernel` 480us x1, `nvjet_tst_128x256_64x4_2x1_v_bz_coopA_NN` 249us x1, `_squeeze_residual_kernel` 246us x1, `nvjet_tst_128x192_64x5_2x1_v_bz_coopB_sp` 223us x1
  * pt_compiled: `triton_poi_fused__unsafe_view_add_mul_si` 493us x1, `nvjet_tst_128x128_64x6_2x1_v_bz_splitK_N` 397us x3, `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TN` 296us x2, `triton_poi_fused__unsafe_view_mul_sigmoi` 296us x1, `nvjet_tst_128x256_64x4_2x1_v_bz_coopA_NN` 261us x2
* **bias-only DiT inference fp32 d768 16x48 A5/A48 L=256**, portable_vs_pytorch ratio 1.100
  * portable: `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_t` 43us x4, `_adaln_gemm_gate_kernel` 40us x2, `_expand_swiglu_kernel` 28us x1, `_squeeze_gate_kernel` 26us x1, `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_t` 24us x1
  * pt_compiled: `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_t` 95us x10, `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_t` 48us x3, `triton_per_fused_native_layer_norm_0` 22us x1, `triton_poi_fused__unsafe_view_add_addmm_` 9us x1, `triton_per_fused__unsafe_view_add_addmm_` 9us x1
* **RMSNorm inference bf16 x[48,L,768] L=384**, cuda_vs_portable ratio 1.090
  * cuda: `void mwln::ln_fwd_kernel<__nv_bfloat16, ` 20us x1, `void at::native::unrolled_elementwise_ke` 4us x1
  * portable: (no profile)
* **AdaLN training fp32 d768 cond384 A5/A48 L=384**, portable_vs_pytorch ratio 1.088
  * portable: `_adaln_fwd_gate_kernel` 192us x1, `_bwd_x_kernel` 179us x1, `sm90_xmma_gemm_f32f32_tf32f32_f32_nt_n_t` 114us x1, `sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_t` 98us x1, `_ln_bwd_kernel` 35us x1
  * pt_compiled: `sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_t` 140us x2, `void cutlass::Kernel2<cutlass_80_tensoro` 134us x2, `triton_per_fused_mul_native_layer_norm_n` 94us x1, `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_t` 85us x2, `triton_per_fused__unsafe_view_add_mul_na` 79us x1
* **token DiT + extra_pair_bias training bf16 d768 A2 shared-cond +[L,L] bias L=256**, portable_vs_pytorch ratio 1.084
  * portable: `void at::native::(anonymous namespace)::` 53us x3, `nvjet_tst_128x64_64x8_2x4_h_bz_NTT` 24us x5, `nvjet_tst_64x48_64x15_2x1_v_bz_NNT` 24us x5, `layer_norm_bwd_dx_fused` 24us x1, `_bwd_x_kernel` 24us x2
  * pt_compiled: `nvjet_tst_64x48_64x15_2x1_v_bz_NNT` 36us x7, `nvjet_tst_48x64_64x15_2x1_v_bz_TNN` 32us x8, `nvjet_tst_64x40_64x16_2x4_h_bz_NNT` 28us x6, `nvjet_tst_128x64_64x8_2x4_h_bz_NTT` 24us x5, `nvjet_tss_64x64_64x13_2x4_h_bz_NTT` 21us x5
* **pair_embed_geom inference bf16 out, fp32 geometry B48 L=128**, cuda_vs_portable ratio 1.083
  * cuda: `pg_fwd(__nv_bfloat16 const*, float const` 353us x1, `void at::native::unrolled_elementwise_ke` 11us x3, `void at::native::unrolled_elementwise_ke` 5us x2, `void at::native::elementwise_kernel<128,` 3us x1, `void (anonymous namespace)::elementwise_` 3us x2
  * portable: `_geom_fwd_gather` 276us x1, `_geom_bins` 51us x1, `void at::native::unrolled_elementwise_ke` 11us x3, `void at::native::unrolled_elementwise_ke` 4us x2, `void at::native::elementwise_kernel<128,` 3us x1
* **ConditionedTransition training bf16 d768 A5/A48 L=768**, portable_vs_pytorch ratio 1.075
  * portable: `_dh_swiglu_bwd_kernel` 373us x1, `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TN` 356us x2, `nvjet_tss_192x192_64x3_2x1_v_bz_coopB_sp` 325us x3, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NN` 255us x2, `_bwd_x_kernel` 184us x1
  * pt_compiled: `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NN` 471us x6, `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TN` 400us x4, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_sp` 209us x2, `triton_poi_fused__unsafe_view_mul_silu_s` 186us x1, `triton_per_fused__to_copy_add_mul_native` 154us x1
* **ConditionedTransition training bf16 d768 A5/A48 L=1024**, portable_vs_pytorch ratio 1.074
  * portable: `_dh_swiglu_bwd_kernel` 492us x1, `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TN` 466us x2, `nvjet_tss_192x192_64x3_2x1_v_bz_coopB_sp` 420us x3, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NN` 334us x2, `_bwd_x_kernel` 243us x1
  * pt_compiled: `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NN` 609us x6, `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TN` 528us x4, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_sp` 268us x2, `triton_poi_fused__unsafe_view_mul_silu_s` 248us x1, `triton_per_fused__to_copy_add_mul_native` 203us x1
* **ConditionedTransition training bf16 d768 A5/A48 L=384**, portable_vs_pytorch ratio 1.072
  * portable: `_dh_swiglu_bwd_kernel` 191us x1, `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TN` 186us x2, `nvjet_tss_128x160_64x5_2x2_h_bz_splitK_N` 117us x2, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NN` 108us x1, `_bwd_x_kernel` 91us x1
  * pt_compiled: `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TN` 209us x4, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NN` 177us x3, `nvjet_tst_128x160_64x5_2x2_h_bz_splitK_N` 119us x2, `triton_poi_fused__unsafe_view_mul_silu_s` 92us x1, `triton_per_fused__to_copy_add_mul_native` 76us x1
* **LayerNorm (pair) training bf16 x[1,L,L,256] L=384**, portable_vs_pytorch ratio 1.072
  * portable: `layer_norm_bwd_dx_fused` 113us x1, `void at::native::vectorized_elementwise_` 4us x2, `void at::native::vectorized_elementwise_` 1us x1
  * pt_compiled: `triton_per_fused_native_layer_norm_nativ` 85us x1, `triton_per_fused_native_layer_norm_0` 50us x1, `void at::native::reduce_kernel<128, 4, a` 17us x2, `triton_poi_fused_native_layer_norm_backw` 3us x2, `Memset (Device)` 2us x2
* **token DiT + extra_pair_bias training bf16 d768 A2 shared-cond +[L,L] bias L=512**, portable_vs_pytorch ratio 1.068
  * portable: `layer_norm_bwd_dx_fused` 85us x1, `_attn_bwd` 47us x1, `nvjet_tst_96x64_64x8_1x2_h_bz_NNN` 39us x6, `void at::native::(anonymous namespace)::` 38us x2, `nvjet_tst_64x32_64x16_2x1_v_bz_splitK_NT` 36us x1
  * pt_compiled: `triton_per_fused__to_copy_native_layer_n` 76us x1, `nvjet_tst_96x64_64x8_1x2_h_bz_NNN` 41us x7, `nvjet_tst_96x64_64x10_1x2_h_bz_TNN` 39us x8, `nvjet_tst_64x16_64x16_2x1_v_bz_splitK_NN` 34us x1, `nvjet_tst_16x512_64x3_1x2_h_bz_TNN` 32us x1
* **ConditionedTransition inference bf16 d768 A5/A48 L=768**, portable_vs_pytorch ratio 1.058
  * portable: `_expand_swiglu_kernel` 40us x1, `_squeeze_gate_kernel` 37us x1, `_adaln_gemm_gate_kernel` 16us x1, `triton_poi_fused__to_copy_conditioned_tr` 12us x3, `triton_poi_fused_add_view_4` 7us x1
  * pt_compiled: `nvjet_tst_128x192_64x5_4x2_h_bz_coopB_TN` 34us x2, `nvjet_tst_192x128_64x5_4x2_h_bz_coopB_TN` 23us x2, `nvjet_tst_192x128_64x5_4x2_h_bz_coopB_bi` 16us x2, `triton_poi_fused__to_copy_4` 13us x3, `triton_poi_fused__unsafe_view_mul_silu_5` 9us x1
* **ConditionedTransition inference bf16 d768 A5/A48 L=384**, portable_vs_pytorch ratio 1.051
  * portable: `_squeeze_gate_kernel` 26us x1, `_expand_swiglu_kernel` 22us x1, `triton_poi_fused__to_copy_conditioned_tr` 13us x3, `_adaln_gemm_gate_kernel` 8us x1, `triton_poi_fused__to_copy_adaln_gemm_gat` 5us x2
  * pt_compiled: `nvjet_tst_128x192_64x5_4x2_h_bz_coopB_TN` 20us x2, `nvjet_tst_128x96_64x7_2x4_h_bz_TNT` 14us x2, `triton_poi_fused__to_copy_4` 12us x3, `nvjet_tst_128x96_64x7_2x4_h_bz_bias_TNT` 11us x2, `triton_poi_fused__to_copy_0` 7us x3
* **Transition inference bf16 D384 L=768**, portable_vs_pytorch ratio 1.050
  * portable: `_squeeze_residual_kernel` 2191us x1
  * pt_compiled: `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TN` 2103us x2, `triton_poi_fused__unsafe_view_mul_sigmoi` 1753us x1, `triton_poi_fused__unsafe_view_add_3` 436us x1, `triton_poi_fused__to_copy_1` 7us x2, `Memset (Device)` 2us x2
* **bias-only DiT training fp32 d768 16x48 A5/A48 L=512**, portable_vs_pytorch ratio 1.050
  * portable: `sm90_xmma_gemm_f32f32_tf32f32_f32_nt_n_t` 1308us x6, `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_t` 988us x9, `sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_t` 868us x6, `_bwd_x_kernel` 467us x2, `sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_t` 454us x3
  * pt_compiled: `sm90_xmma_gemm_f32f32_tf32f32_f32_nt_n_t` 1334us x6, `sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_t` 1200us x11, `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_t` 1152us x13, `void cutlass::Kernel2<cutlass_80_tensoro` 566us x7, `triton_per_fused__unsafe_view_add_mul_na` 302us x1
* **bias-only DiT inference fp32 d768 16x48 A5/A48 L=512**, portable_vs_pytorch ratio 1.042
  * portable: `layer_norm_fwd_fused` 93us x1, `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_t` 68us x1, `_adaln_gemm_gate_kernel` 56us x2, `_expand_swiglu_kernel` 53us x1, `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_t` 46us x3
  * pt_compiled: `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_t` 172us x9, `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_t` 68us x4, `void cutlass::Kernel2<cutlass_80_tensoro` 17us x1, `triton_poi_fused__unsafe_view_add_addmm_` 16us x1, `triton_per_fused__unsafe_view_add_addmm_` 15us x1
* **Transition inference bf16 D384 L=384**, portable_vs_pytorch ratio 1.041
  * portable: `transition_fwd_kernel` 890us x1, `_squeeze_residual_kernel` 546us x1, `triton_poi_fused__to_copy_transition_exp` 4us x1
  * pt_compiled: `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TN` 819us x3, `triton_poi_fused__unsafe_view_mul_sigmoi` 439us x1, `triton_per_fused__to_copy_native_layer_n` 113us x1, `triton_poi_fused__unsafe_view_add_3` 108us x1, `triton_poi_fused__to_copy_1` 11us x3
* **LayerNorm (pair) inference bf16 x[1,L,L,256] L=384**, portable_vs_pytorch ratio 1.040
  * portable: (no profile)
  * pt_compiled: `triton_per_fused_native_layer_norm_0` 48us x1
* **bias-only DiT training fp32 d768 16x48 A5/A48 L=256**, portable_vs_pytorch ratio 1.037
  * portable: `sm90_xmma_gemm_f32f32_tf32f32_f32_nt_n_t` 673us x5, `sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_t` 490us x7, `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_t` 458us x8, `sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_t` 342us x3, `_adaln_fwd_gate_kernel` 279us x2
  * pt_compiled: `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_t` 594us x13, `sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_t` 549us x9, `sm90_xmma_gemm_f32f32_tf32f32_f32_nt_n_t` 479us x3, `sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_t` 359us x4, `void cutlass::Kernel2<cutlass_80_tensoro` 303us x7
* **ConditionedTransition training fp32 d768 A5/A48 L=384**, portable_vs_pytorch ratio 1.035
  * portable: `sm90_xmma_gemm_f32f32_tf32f32_f32_nt_n_t` 675us x3, `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_t` 414us x3, `sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_t` 398us x2, `sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_t` 204us x2, `_adaln_fwd_gate_kernel` 192us x1
  * pt_compiled: `sm90_xmma_gemm_f32f32_tf32f32_f32_nt_n_t` 682us x3, `sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_t` 639us x6, `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_t` 510us x6, `void cutlass::Kernel2<cutlass_80_tensoro` 198us x3, `triton_poi_fused__unsafe_view_mul_silu_s` 185us x1
* **AdaLN inference bf16 d768 cond384 A5/A48 L=384**, cuda_vs_portable ratio 1.029
  * cuda: `nvjet_tst_128x96_64x7_2x4_h_bz_bias_TNT` 6us x1, `nvjet_tst_128x96_64x7_2x4_h_bz_TNT` 5us x1, `triton_per_fused__to_copy__unsafe_view_a` 4us x1, `triton_per_fused__to_copy_native_layer_n` 3us x1, `triton_poi_fused__to_copy_0` 2us x1
  * portable: `_adaln_gemm_gate_kernel` 9us x1, `triton_poi_fused__to_copy_adaln_gemm_gat` 5us x2, `_ln_kernel` 4us x1, `triton_poi_fused__to_copy_adaln_gemm_gat` 1us x1
* **LayerNorm (pair) inference bf16 x[1,L,L,256] L=384**, cuda_vs_portable ratio 1.006
  * cuda: `layer_norm_fwd_fused` 50us x1
  * portable: (no profile)
* **token_pair_init inference fp32 in, fp32 out B48 L=256**, cuda_vs_portable ratio 1.003
  * cuda: `tpi_fwd(float const*, float const*, floa` 585us x1, `void at::native::unrolled_elementwise_ke` 11us x1, `void at::native::unrolled_elementwise_ke` 8us x1, `Memcpy DtoD (Device -> Device)` 7us x4, `void at::native::unrolled_elementwise_ke` 4us x1
  * portable: `_tpi_fwd` 616us x1, `void at::native::unrolled_elementwise_ke` 11us x1, `void at::native::unrolled_elementwise_ke` 8us x1, `Memcpy DtoD (Device -> Device)` 8us x4, `void at::native::unrolled_elementwise_ke` 4us x1
* **Transition training bf16 D512 L=384**, portable_vs_pytorch ratio 1.270
  * portable: `_transition_expand_gatebwd_kernel` 2058us x1, `transition_fwd_kernel` 1358us x1, `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NN` 1285us x2, `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_sp` 1180us x2, `_squeeze_residual_kernel` 956us x1
  * pt_compiled: `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TN` 1354us x3, `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NN` 1312us x3, `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_sp` 1176us x3, `triton_poi_fused__unsafe_view_add_mul_si` 986us x1, `triton_poi_fused__unsafe_view_mul_sigmoi` 584us x1
* **Transition inference bf16 D512 L=384**, portable_vs_pytorch ratio 1.119
  * portable: `transition_fwd_kernel` 1360us x1, `_squeeze_residual_kernel` 953us x1, `triton_poi_fused__to_copy_transition_exp` 5us x1
  * pt_compiled: `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TN` 1357us x3, `triton_poi_fused__unsafe_view_mul_sigmoi` 586us x1, `triton_per_fused__to_copy_native_layer_n` 152us x1, `triton_poi_fused__unsafe_view_add_3` 145us x1, `triton_poi_fused__to_copy_1` 14us x3
* **token DiT + extra_pair_bias inference fp32 d768 A2 shared-cond +[L,L] bias L=256**, portable_vs_pytorch ratio 1.088
  * portable: `_adaln_gemm_gate_kernel` 43us x2, `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_t` 40us x5, `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_t` 23us x1, `layer_norm_fwd_fused` 22us x1, `_squeeze_gate_kernel` 22us x1
  * pt_compiled: `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_t` 52us x6, `void cutlass::Kernel2<cutlass_80_tensoro` 43us x6, `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_t` 25us x1, `triton_per_fused_native_layer_norm_1` 21us x1, `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_t` 20us x2
* **AttentionPairBias inference bf16 8x48 pair256 L=384**, cuda_vs_portable ratio 1.073
  * cuda: `void mwln::ln_fwd_kernel<__nv_bfloat16, ` 52us x1, `nvjet_tst_8x384_64x4_1x2_h_bz_TNN` 31us x1, `fmha_cutlassF_bf16_aligned_64x64_rf_sm80` 28us x1, `void at::native::(anonymous namespace)::` 16us x1, `nvjet_tst_64x32_64x16_2x4_h_bz_TNT` 14us x4
  * portable: `layer_norm_fwd_fused` 56us x2, `nvjet_tst_8x384_64x4_1x2_h_bz_TNN` 34us x1, `void at::native::(anonymous namespace)::` 16us x1, `nvjet_tst_64x32_64x16_2x4_h_bz_TNT` 14us x4, `_attn_fwd` 10us x1
* **bias-only DiT training bf16 24x32 L=512**, portable_vs_pytorch ratio 1.073
  * portable: `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TN` 384us x5, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NN` 349us x6, `_dh_swiglu_bwd_kernel` 258us x1, `_bwd_x_kernel` 246us x2, `_adaln_fwd_gate_kernel` 203us x2
  * pt_compiled: `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NN` 557us x12, `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TN` 441us x8, `nvjet_tst_128x160_64x5_2x2_h_bz_splitK_N` 163us x2, `triton_per_fused__to_copy__unsafe_view_a` 155us x1, `nvjet_tss_96x128_64x6_4x2_h_bz_splitK_NT` 150us x5
* **Transition training fp32 D256 TF32 L=384**, portable_vs_pytorch ratio 1.065
  * portable: `_transition_expand_gatebwd_kernel` 1266us x1, `sm90_xmma_gemm_f32f32_tf32f32_f32_nt_n_t` 1001us x2, `sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_t` 867us x1, `_squeeze_residual_kernel` 427us x1, `sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_t` 306us x1
  * pt_compiled: `sm90_xmma_gemm_f32f32_tf32f32_f32_nt_n_t` 993us x3, `triton_poi_fused__unsafe_view_add_mul_si` 983us x1, `sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_t` 608us x2, `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_t` 586us x2, `triton_poi_fused__unsafe_view_mul_sigmoi` 584us x1
* **bias-only DiT inference bf16 12x64 L=512**, portable_vs_pytorch ratio 1.062
  * portable: `layer_norm_fwd_fused` 47us x1, `void cutlass::Kernel2<cutlass_80_wmma_te` 43us x1, `_expand_swiglu_kernel` 34us x1, `_squeeze_gate_kernel` 33us x1, `nvjet_tst_128x128_64x6_2x4_h_bz_TNT` 24us x3
  * pt_compiled: `nvjet_tst_128x128_64x6_2x4_h_bz_TNT` 47us x6, `triton_per_fused__to_copy_native_layer_n` 46us x1, `void cutlass::Kernel2<cutlass_80_wmma_te` 44us x1, `nvjet_tst_128x128_64x6_2x4_h_bz_bias_TNT` 26us x4, `nvjet_tst_256x128_64x4_2x4_h_bz_coopA_TN` 25us x2
* **bias-only DiT training bf16 12x64 L=512**, portable_vs_pytorch ratio 1.061
  * portable: `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TN` 382us x5, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NN` 345us x6, `_dh_swiglu_bwd_kernel` 253us x1, `_bwd_x_kernel` 245us x2, `_adaln_fwd_gate_kernel` 201us x2
  * pt_compiled: `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NN` 561us x12, `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TN` 443us x8, `nvjet_tst_128x160_64x5_2x2_h_bz_splitK_N` 164us x2, `triton_per_fused__to_copy__unsafe_view_a` 155us x1, `nvjet_tss_96x128_64x6_4x2_h_bz_splitK_NT` 149us x5
* **bias-only DiT inference bf16 24x32 L=512**, portable_vs_pytorch ratio 1.043
  * portable: `layer_norm_fwd_fused` 46us x1, `_expand_swiglu_kernel` 33us x1, `nvjet_tst_24x512_64x3_1x2_h_bz_TNN` 32us x1, `_squeeze_gate_kernel` 27us x1, `nvjet_tst_128x128_64x6_2x4_h_bz_TNT` 23us x3
  * pt_compiled: `nvjet_tst_128x128_64x6_2x4_h_bz_TNT` 47us x6, `triton_per_fused__to_copy_native_layer_n` 47us x1, `nvjet_tst_24x512_64x3_1x2_h_bz_TNN` 33us x1, `nvjet_tst_128x128_64x6_2x4_h_bz_bias_TNT` 26us x4, `nvjet_tst_256x128_64x4_2x4_h_bz_coopA_TN` 24us x2
* **LayerNorm (pair) inference fp32 x[1,L,L,128] L=384**, portable_vs_pytorch ratio 1.034
  * portable: `layer_norm_fwd_fused` 47us x1
  * pt_compiled: `triton_per_fused_native_layer_norm_0` 48us x1
* **AttentionPairBias training bf16 8x48 pair256 L=384**, cuda_vs_portable ratio 1.033
  * cuda: `void mwln::ln_bwd_kernel<__nv_bfloat16, ` 82us x1, `fmha_cutlassB_bf16_aligned_64x64_k64_seq` 54us x1, `void mwln::ln_fwd_kernel<__nv_bfloat16, ` 52us x1, `nvjet_tst_128x160_64x5_2x1_v_bz_NTT` 34us x1, `nvjet_tst_8x384_64x4_1x2_h_bz_TNN` 32us x1
  * portable: `layer_norm_bwd_dx_fused` 122us x2, `layer_norm_fwd_fused` 50us x1, `nvjet_tst_8x384_64x4_1x2_h_bz_TNN` 35us x1, `nvjet_tss_64x8_64x16_4x1_v_bz_splitK_NNT` 33us x1, `nvjet_tst_128x160_64x5_2x1_v_bz_NTT` 30us x1
* **OuterProductMean RF3 (bias, divide after) training bf16 S512 msa64 pair128 L=256**, portable_vs_pytorch ratio 1.016
  * portable: `_opm_bwd_epilogue_dx_kernel` 146us x1, `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NT` 108us x1, `_opm_bwd_layernorm_gemm_kernel` 94us x1, `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TN` 87us x1, `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NN` 87us x1
  * pt_compiled: `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NN` 146us x2, `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NT` 110us x1, `triton_poi_fused__unsafe_view_bmm_clone_` 94us x1, `triton_poi_fused_bmm_clone_permute_view_` 93us x1, `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TN` 87us x1
* **OuterProductMean RF3 (bias, divide after) training bf16 S512 msa64 pair128 L=384**, portable_vs_pytorch ratio 1.004
  * portable: `_opm_bwd_epilogue_dx_kernel` 325us x1, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NT` 248us x1, `nvjet_tst_192x128_64x5_1x2_h_bz_coopB_TN` 198us x1, `nvjet_tst_192x128_64x5_1x2_h_bz_coopB_NN` 197us x1, `void at::native::reduce_kernel<128, 4, a` 163us x1
  * pt_compiled: `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NT` 252us x1, `triton_poi_fused__unsafe_view_bmm_clone_` 207us x1, `triton_poi_fused_bmm_clone_permute_view_` 203us x1, `nvjet_tst_128x192_64x5_2x1_v_bz_coopB_TN` 202us x1, `nvjet_tst_192x128_64x5_1x2_h_bz_coopB_NN` 201us x1
* **OuterProductMean RF3 (bias, divide after) inference bf16 S512 msa64 pair128 L=256**, cuda_vs_portable ratio 1.001
  * cuda: `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TN` 110us x1, `(anonymous namespace)::opm_epilogue_kern` 70us x1, `triton_poi_fused__to_copy_0` 3us x1, `(anonymous namespace)::opm_count_tiled(u` 2us x1, `triton_poi_fused__to_copy_1` 2us x1
  * portable: `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NT` 110us x1, `_opm_epilogue_kernel` 71us x1, `_opm_layernorm_gemm_kernel` 16us x1, `void at::native::vectorized_elementwise_` 9us x5, `void at::native::unrolled_elementwise_ke` 4us x1
* **token DiT + extra_pair_bias training fp32 d768 A2 shared-cond +[L,L] bias L=256**, cuda_vs_portable ratio 1.000
  * cuda: `void cutlass::Kernel2<cutlass_80_tensoro` 60us x8, `void cutlass::Kernel2<cutlass_80_tensoro` 57us x6, `void cutlass::Kernel2<cutlass_80_tensoro` 52us x5, `void cutlass::Kernel2<cutlass_80_tensoro` 46us x3, `void cutlass::Kernel2<cutlass_80_tensoro` 38us x6
  * portable: `void cutlass::Kernel2<cutlass_80_tensoro` 88us x8, `void cutlass::Kernel2<cutlass_80_tensoro` 51us x5, `void cutlass::Kernel2<cutlass_80_tensoro` 46us x3, `layer_norm_bwd_dx_fused` 39us x1, `sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_t` 34us x3

### Full ladder table (ms; CUDA-graph replay of the compiled step, median of 3)

`pt_c` = PyTorch compiled + graph, `pt_eg` = eager statements in a graph, `pt_e` = plain eager; the PyTorch rung is the fastest of the three. `c/p` = CUDA / portable, `p/t` = portable / PyTorch, `c/t` = CUDA / PyTorch.

| op | mode | dtype | shape | L | pt_c | pt_eg | pt_e | portable | CUDA | c/p | p/t | c/t | verdict | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TriMul outgoing | inference | bf16 | D128 | 128 | 0.098 | 0.209 | 0.336 | - | 0.033 | - | - | 0.335 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing | inference | bf16 | D128 | 384 | 0.771 | 1.697 | 1.754 | - | 0.146 | - | - | 0.190 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing | inference | bf16 | D128 | 768 | 6.212 | 9.969 | 10.023 | - | 0.566 | - | - | 0.091 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing | inference | bf16 | D128 | 1024 | 11.721 | 19.522 | 19.583 | - | 1.086 | - | - | 0.093 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing | training | bf16 | D128 | 128 | 0.321 | 0.754 | 1.393 | - | 0.164 | - | - | 0.511 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing | training | bf16 | D128 | 384 | 2.416 | 5.431 | 5.595 | - | 0.762 | - | - | 0.315 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing | training | bf16 | D128 | 768 | 18.847 | 41.140 | 41.335 | - | 2.773 | - | - | 0.147 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing | training | bf16 | D128 | 1024 | 35.436 | 80.967 | 81.128 | - | 5.117 | - | - | 0.144 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing | inference | bf16 | D128, bf16 params | 128 | 0.090 | 0.201 | - | 0.042 | 0.031 | 0.733 | 0.467 | 0.342 | OK |  |
| TriMul outgoing | inference | bf16 | D128, bf16 params | 384 | 0.765 | 1.689 | - | 0.260 | 0.144 | 0.554 | 0.340 | 0.188 | OK |  |
| TriMul outgoing | inference | bf16 | D128, bf16 params | 768 | 6.208 | 9.996 | - | 0.964 | 0.568 | 0.589 | 0.155 | 0.091 | OK |  |
| TriMul outgoing | inference | bf16 | D128, bf16 params | 1024 | 11.722 | 19.517 | - | 1.804 | 1.093 | 0.606 | 0.154 | 0.093 | OK |  |
| TriMul outgoing | training | bf16 | D128, bf16 params | 128 | 0.308 | 0.730 | - | 0.164 | 0.162 | 0.988 | 0.533 | 0.526 | OK | cuda_vs_portable within noise (ratio 0.988, spread 0.007) |
| TriMul outgoing | training | bf16 | D128, bf16 params | 384 | 2.407 | 5.395 | - | 1.017 | 0.760 | 0.747 | 0.423 | 0.316 | OK |  |
| TriMul outgoing | training | bf16 | D128, bf16 params | 768 | 18.838 | 41.151 | - | 3.769 | 2.802 | 0.743 | 0.200 | 0.149 | OK |  |
| TriMul outgoing | training | bf16 | D128, bf16 params | 1024 | 35.409 | 80.903 | - | 6.772 | 5.109 | 0.754 | 0.191 | 0.144 | OK |  |
| TriMul outgoing | inference | bf16 | D64 | 128 | 0.068 | 0.151 | 0.328 | - | 0.021 | - | - | 0.315 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing | inference | bf16 | D64 | 384 | 0.368 | 1.053 | 1.109 | - | 0.081 | - | - | 0.221 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing | training | bf16 | D64 | 128 | 0.221 | 0.538 | 1.384 | - | 0.069 | - | - | 0.314 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing | training | bf16 | D64 | 384 | 1.253 | 3.071 | 3.227 | - | 0.256 | - | - | 0.204 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing | inference | bf16 | D64, bf16 params | 128 | 0.061 | 0.145 | - | 0.028 | 0.020 | 0.711 | 0.452 | 0.321 | OK |  |
| TriMul outgoing | inference | bf16 | D64, bf16 params | 384 | 0.364 | 1.045 | - | 0.105 | 0.079 | 0.755 | 0.289 | 0.218 | OK |  |
| TriMul outgoing | training | bf16 | D64, bf16 params | 128 | 0.208 | 0.513 | - | 0.116 | 0.069 | 0.594 | 0.558 | 0.331 | OK |  |
| TriMul outgoing | training | bf16 | D64, bf16 params | 384 | 1.239 | 3.045 | - | 0.534 | 0.256 | 0.479 | 0.431 | 0.206 | OK |  |
| TriMul outgoing | inference | bf16 | D256 | 384 | 1.457 | 3.038 | 3.063 | - | 0.356 | - | - | 0.244 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing | inference | bf16 | D256 | 768 | 12.350 | 19.035 | 19.075 | - | 1.493 | - | - | 0.121 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing | training | bf16 | D256 | 384 | 4.693 | 10.306 | 10.444 | - | 1.737 | - | - | 0.370 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing | training | bf16 | D256 | 768 | 37.935 | 82.451 | 82.672 | - | 6.758 | - | - | 0.178 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing | inference | bf16 | D256, bf16 params | 384 | 1.448 | 3.033 | - | 0.649 | 0.356 | 0.549 | 0.448 | 0.246 | OK |  |
| TriMul outgoing | inference | bf16 | D256, bf16 params | 768 | 12.371 | 19.056 | - | 2.574 | 1.487 | 0.578 | 0.208 | 0.120 | OK |  |
| TriMul outgoing | training | bf16 | D256, bf16 params | 384 | 4.666 | 10.287 | - | 2.205 | 1.731 | 0.785 | 0.473 | 0.371 | OK |  |
| TriMul outgoing | training | bf16 | D256, bf16 params | 768 | 37.865 | 82.439 | - | 8.996 | 6.746 | 0.750 | 0.238 | 0.178 | OK |  |
| TriMul outgoing | inference | bf16 | D512 | 384 | 2.942 | 5.716 | 5.781 | - | 1.155 | - | - | 0.393 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing | inference | bf16 | D512 | 768 | 25.733 | 37.941 | 37.969 | - | 4.638 | - | - | 0.180 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing | training | bf16 | D512 | 384 | 9.352 | 20.222 | 20.337 | - | 4.291 | - | - | 0.459 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing | training | bf16 | D512 | 768 | 78.918 | 176.7 | 177.1 | - | 17.678 | - | - | 0.224 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing | inference | bf16 | D512, bf16 params | 384 | 2.939 | 5.681 | - | 2.215 | 1.148 | 0.518 | 0.754 | 0.391 | OK |  |
| TriMul outgoing | inference | bf16 | D512, bf16 params | 768 | 25.725 | 37.925 | - | 8.859 | 4.608 | 0.520 | 0.344 | 0.179 | OK |  |
| TriMul outgoing | training | bf16 | D512, bf16 params | 384 | 9.338 | 20.205 | - | 5.538 | 4.284 | 0.774 | 0.593 | 0.459 | OK |  |
| TriMul outgoing | training | bf16 | D512, bf16 params | 768 | 78.800 | 176.9 | - | 24.853 | 17.635 | 0.710 | 0.315 | 0.224 | OK |  |
| TriMul incoming | inference | bf16 | D128 | 384 | 0.773 | 1.698 | 1.759 | - | 0.148 | - | - | 0.191 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul incoming | inference | bf16 | D128 | 768 | 6.211 | 10.002 | 10.028 | - | 0.587 | - | - | 0.095 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul incoming | training | bf16 | D128 | 384 | 2.418 | 5.438 | 5.586 | - | 0.764 | - | - | 0.316 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul incoming | training | bf16 | D128 | 768 | 18.853 | 41.163 | 41.315 | - | 2.798 | - | - | 0.148 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul incoming | inference | bf16 | D128, bf16 params | 384 | 0.768 | 1.690 | - | 0.261 | 0.146 | 0.559 | 0.339 | 0.190 | OK |  |
| TriMul incoming | inference | bf16 | D128, bf16 params | 768 | 6.207 | 9.998 | - | 0.988 | 0.587 | 0.595 | 0.159 | 0.095 | OK |  |
| TriMul incoming | training | bf16 | D128, bf16 params | 384 | 2.400 | 5.399 | - | 1.017 | 0.763 | 0.750 | 0.424 | 0.318 | OK |  |
| TriMul incoming | training | bf16 | D128, bf16 params | 768 | 18.842 | 41.125 | - | 3.795 | 2.821 | 0.743 | 0.201 | 0.150 | OK |  |
| TriMul incoming | inference | bf16 | D512 | 384 | 2.948 | 5.719 | 5.792 | - | 1.157 | - | - | 0.392 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul incoming | training | bf16 | D512 | 384 | 9.350 | 20.228 | 20.283 | - | 4.287 | - | - | 0.459 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul incoming | inference | bf16 | D512, bf16 params | 384 | 2.938 | 5.686 | - | 2.214 | 1.143 | 0.516 | 0.753 | 0.389 | OK |  |
| TriMul incoming | training | bf16 | D512, bf16 params | 384 | 9.347 | 20.221 | - | 5.584 | 4.302 | 0.770 | 0.597 | 0.460 | OK |  |
| TriMul outgoing hidden!=D | inference | bf16 | D64 hidden128 | 128 | 0.092 | 0.198 | 0.369 | - | 0.027 | - | - | 0.292 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing hidden!=D | inference | bf16 | D64 hidden128 | 384 | 0.700 | 1.552 | 1.601 | - | 0.117 | - | - | 0.167 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing hidden!=D | training | bf16 | D64 hidden128 | 128 | 0.293 | 0.712 | 1.407 | - | 0.092 | - | - | 0.315 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing hidden!=D | training | bf16 | D64 hidden128 | 384 | 2.129 | 4.870 | 5.014 | - | 0.380 | - | - | 0.179 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing hidden!=D | inference | bf16 | D64 hidden128, bf16 params | 128 | 0.084 | 0.191 | - | 0.036 | 0.025 | 0.685 | 0.431 | 0.295 | OK |  |
| TriMul outgoing hidden!=D | inference | bf16 | D64 hidden128, bf16 params | 384 | 0.682 | 1.543 | - | 0.182 | 0.115 | 0.633 | 0.266 | 0.169 | OK |  |
| TriMul outgoing hidden!=D | training | bf16 | D64 hidden128, bf16 params | 128 | 0.284 | 0.689 | - | 0.140 | 0.092 | 0.659 | 0.493 | 0.325 | OK |  |
| TriMul outgoing hidden!=D | training | bf16 | D64 hidden128, bf16 params | 384 | 2.119 | 4.843 | - | 0.815 | 0.379 | 0.466 | 0.385 | 0.179 | OK |  |
| TriMul outgoing hidden!=D | inference | bf16 | D128 hidden256 | 384 | 1.293 | 2.710 | 2.767 | - | 0.241 | - | - | 0.187 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing hidden!=D | training | bf16 | D128 hidden256 | 384 | 4.082 | 9.144 | 9.366 | - | 0.962 | - | - | 0.236 | n.a. | n.a.: cuda_vs_portable: portable: Both operands must be same dtype | portable_vs_pytorch: portable: Both operands must be same dtype |
| TriMul outgoing hidden!=D | inference | bf16 | D128 hidden256, bf16 params | 384 | 1.277 | 2.705 | - | 0.431 | 0.239 | 0.556 | 0.338 | 0.188 | OK |  |
| TriMul outgoing hidden!=D | training | bf16 | D128 hidden256, bf16 params | 384 | 4.049 | 9.084 | - | 1.609 | 0.962 | 0.598 | 0.397 | 0.238 | OK |  |
| TriMul bidirectional | inference | bf16 | D128 | 128 | 0.156 | 0.298 | 0.391 | 0.063 | 0.042 | 0.661 | 0.403 | 0.266 | OK |  |
| TriMul bidirectional | inference | bf16 | D128 | 384 | 1.315 | 2.555 | 2.622 | 0.436 | 0.241 | 0.554 | 0.331 | 0.184 | OK |  |
| TriMul bidirectional | inference | bf16 | D128 | 768 | 11.674 | 16.700 | 16.729 | 1.690 | 1.023 | 0.605 | 0.145 | 0.088 | OK |  |
| TriMul bidirectional | training | bf16 | D128 | 128 | 0.518 | 0.970 | 1.653 | 0.238 | 0.198 | 0.830 | 0.460 | 0.382 | OK |  |
| TriMul bidirectional | training | bf16 | D128 | 384 | 4.000 | 7.266 | 7.454 | 1.622 | 0.966 | 0.596 | 0.405 | 0.241 | OK |  |
| TriMul bidirectional | training | bf16 | D128 | 768 | 35.170 | 48.007 | 48.186 | 6.577 | 3.922 | 0.596 | 0.187 | 0.112 | OK |  |
| TriMul bidirectional | inference | bf16 | D64 | 384 | 0.696 | 1.506 | 1.565 | 0.182 | 0.120 | 0.658 | 0.262 | 0.172 | OK |  |
| TriMul bidirectional | training | bf16 | D64 | 384 | 2.038 | 4.002 | 4.211 | 0.819 | 0.390 | 0.477 | 0.402 | 0.191 | OK |  |
| TriMul bidirectional | inference | bf16 | D256 | 384 | 2.571 | 4.693 | 4.747 | 1.093 | 0.601 | 0.550 | 0.425 | 0.234 | OK |  |
| TriMul bidirectional | inference | bf16 | D256 | 768 | 24.056 | 33.097 | 33.139 | 4.444 | 2.587 | 0.582 | 0.185 | 0.108 | OK |  |
| TriMul bidirectional | training | bf16 | D256 | 384 | 7.916 | 13.865 | 14.077 | 3.720 | 2.685 | 0.722 | 0.470 | 0.339 | OK |  |
| TriMul bidirectional | training | bf16 | D256 | 768 | 73.150 | 97.807 | 98.173 | 16.356 | 11.106 | 0.679 | 0.224 | 0.152 | OK |  |
| TriMul bidirectional | inference | bf16 | D512 | 384 | 5.444 | 9.839 | 9.903 | 3.710 | 1.999 | 0.539 | 0.681 | 0.367 | OK |  |
| TriMul bidirectional | training | bf16 | D512 | 384 | 17.030 | 28.740 | 29.028 | 9.929 | 7.111 | 0.716 | 0.583 | 0.418 | OK |  |
| TriMul bidirectional hidden!=D | inference | bf16 | D64 hidden128 | 384 | 1.219 | 2.409 | 2.462 | 0.318 | 1.231 | 3.864 | 0.261 | 1.010 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 3.86x faster than what H100 runs) |
| TriMul bidirectional hidden!=D | training | bf16 | D64 hidden128 | 384 | 3.695 | 6.692 | 6.929 | 1.379 | 3.643 | 2.642 | 0.373 | 0.986 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 2.64x faster than what H100 runs) |
| TriAttn | inference | bf16 | D128 4x32 | 128 | 0.107 | 0.234 | 0.456 | 0.066 | 0.034 | 0.515 | 0.613 | 0.315 | OK |  |
| TriAttn | inference | bf16 | D128 4x32 | 384 | 1.624 | 3.644 | 3.697 | 0.507 | 0.261 | 0.516 | 0.312 | 0.161 | OK |  |
| TriAttn | inference | bf16 | D128 4x32 | 768 | 11.385 | 24.989 | 25.099 | 2.611 | 1.322 | 0.506 | 0.229 | 0.116 | OK |  |
| TriAttn | training | bf16 | D128 4x32 | 128 | 0.351 | 0.659 | 1.648 | 0.230 | 0.152 | 0.660 | 0.657 | 0.433 | OK |  |
| TriAttn | training | bf16 | D128 4x32 | 384 | 4.656 | 8.943 | 9.104 | 1.700 | 1.082 | 0.636 | 0.365 | 0.232 | OK |  |
| TriAttn | training | bf16 | D128 4x32 | 768 | 31.907 | 60.150 | 60.341 | 8.421 | 6.096 | 0.724 | 0.264 | 0.191 | OK |  |
| TriAttn | inference | bf16 | D256 8x32 | 384 | 3.622 | 7.049 | 7.101 | 1.009 | 0.612 | 0.607 | 0.278 | 0.169 | OK |  |
| TriAttn | training | bf16 | D256 8x32 | 384 | 9.456 | 17.420 | 17.561 | 3.956 | 2.511 | 0.635 | 0.418 | 0.266 | OK |  |
| TriAttn | inference | bf16 | D64 2x32 | 384 | 0.827 | 1.942 | 1.989 | 0.251 | 0.159 | 0.635 | 0.303 | 0.192 | OK |  |
| TriAttn | training | bf16 | D64 2x32 | 384 | 2.313 | 4.682 | 4.822 | 1.040 | 0.747 | 0.718 | 0.449 | 0.323 | OK |  |
| Transition | inference | bf16 | D128 | 128 | 0.057 | 0.125 | 0.154 | 0.040 | 0.023 | 0.562 | 0.702 | 0.395 | OK |  |
| Transition | inference | bf16 | D128 | 384 | 0.452 | 1.021 | 1.049 | 0.284 | 0.113 | 0.397 | 0.629 | 0.250 | OK |  |
| Transition | inference | bf16 | D128 | 768 | 1.711 | 3.944 | 3.970 | 1.081 | 0.409 | 0.378 | 0.632 | 0.239 | OK |  |
| Transition | inference | bf16 | D128 | 1024 | 3.033 | 6.964 | 6.998 | 1.913 | 0.741 | 0.387 | 0.631 | 0.244 | OK |  |
| Transition | training | bf16 | D128 | 128 | 0.161 | 0.382 | 0.549 | 0.152 | 0.086 | 0.563 | 0.946 | 0.532 | OK |  |
| Transition | training | bf16 | D128 | 384 | 1.213 | 2.846 | 2.924 | 1.051 | 0.546 | 0.519 | 0.867 | 0.450 | OK |  |
| Transition | training | bf16 | D128 | 768 | 4.514 | 10.884 | 10.971 | 3.991 | 2.149 | 0.539 | 0.884 | 0.476 | OK |  |
| Transition | training | bf16 | D128 | 1024 | 7.957 | 19.190 | 19.299 | 7.046 | 3.798 | 0.539 | 0.886 | 0.477 | OK |  |
| Transition | inference | bf16 | D256 | 384 | 0.882 | 1.827 | 1.850 | 0.817 | 0.407 | 0.498 | 0.927 | 0.462 | OK |  |
| Transition | inference | bf16 | D256 | 768 | 3.414 | 7.124 | 7.144 | 3.335 | 1.593 | 0.478 | 0.977 | 0.467 | OK | portable_vs_pytorch within noise (ratio 0.977, spread 0.098) |
| Transition | training | bf16 | D256 | 384 | 2.337 | 5.263 | 5.334 | 2.580 | 1.844 | 0.715 | 1.104 | 0.789 | VIOLATION(portable_vs_pytorch) |  |
| Transition | training | bf16 | D256 | 768 | 9.019 | 20.543 | 20.600 | 10.165 | 7.166 | 0.705 | 1.127 | 0.794 | VIOLATION(portable_vs_pytorch) |  |
| Transition | inference | bf16 | D384 | 384 | 1.531 | 2.809 | 2.832 | 1.593 | 0.965 | 0.606 | 1.041 | 0.630 | VIOLATION(portable_vs_pytorch) | portable_vs_pytorch within noise (ratio 1.041, spread 0.054) |
| Transition | inference | bf16 | D384 | 768 | 5.963 | 11.085 | 11.104 | 6.263 | 3.783 | 0.604 | 1.050 | 0.634 | VIOLATION(portable_vs_pytorch) |  |
| Transition | training | bf16 | D384 | 384 | 4.087 | 8.240 | 8.318 | 4.958 | 3.638 | 0.734 | 1.213 | 0.890 | VIOLATION(portable_vs_pytorch) |  |
| Transition | training | bf16 | D384 | 768 | 16.043 | 32.385 | 32.445 | 19.940 | 13.659 | 0.685 | 1.243 | 0.851 | VIOLATION(portable_vs_pytorch) |  |
| Transition | inference | bf16 | D512 | 384 | 2.340 | 3.944 | 3.964 | 2.618 | 1.587 | 0.606 | 1.119 | 0.678 | VIOLATION(portable_vs_pytorch) |  |
| Transition | training | bf16 | D512 | 384 | 6.199 | 11.579 | 11.638 | 7.871 | 6.028 | 0.766 | 1.270 | 0.972 | VIOLATION(portable_vs_pytorch) |  |
| Transition | inference | bf16 | D768 | 384 | 4.356 | 6.662 | 6.684 | 5.402 | 3.377 | 0.625 | 1.240 | 0.775 | VIOLATION(portable_vs_pytorch) |  |
| Transition | training | bf16 | D768 | 384 | 12.268 | 19.948 | 20.009 | 16.568 | 11.025 | 0.665 | 1.351 | 0.899 | VIOLATION(portable_vs_pytorch) |  |
| Transition | inference | fp32 | D128 TF32 | 384 | 0.846 | 1.510 | 1.518 | 0.488 | 0.407 | 0.834 | 0.577 | 0.481 | OK |  |
| Transition | inference | fp32 | D128 TF32 | 768 | 3.288 | 5.911 | 5.924 | 1.867 | 1.588 | 0.851 | 0.568 | 0.483 | OK |  |
| Transition | training | fp32 | D128 TF32 | 384 | 2.372 | 4.513 | 4.564 | 1.986 | 1.638 | 0.825 | 0.837 | 0.690 | OK |  |
| Transition | training | fp32 | D128 TF32 | 768 | 9.145 | 17.692 | 17.738 | 7.742 | 6.414 | 0.829 | 0.847 | 0.701 | OK |  |
| Transition | inference | fp32 | D256 TF32 | 384 | 1.695 | 2.839 | 2.845 | 1.275 | 0.990 | 0.777 | 0.752 | 0.584 | OK |  |
| Transition | training | fp32 | D256 TF32 | 384 | 4.935 | 8.910 | 8.941 | 5.255 | 4.049 | 0.770 | 1.065 | 0.820 | VIOLATION(portable_vs_pytorch) |  |
| OuterProductMean | inference | bf16 | S1024 msa64 pair128 | 256 | 0.448 | 1.248 | 1.285 | 0.319 | 0.309 | 0.968 | 0.712 | 0.689 | OK |  |
| OuterProductMean | inference | bf16 | S1024 msa64 pair128 | 384 | 0.930 | 2.316 | 2.345 | 0.681 | 0.652 | 0.958 | 0.732 | 0.701 | OK |  |
| OuterProductMean | inference | bf16 | S1024 msa64 pair128 | 768 | 3.416 | 7.369 | 7.401 | 2.600 | 2.463 | 0.947 | 0.761 | 0.721 | OK |  |
| OuterProductMean | training | bf16 | S1024 msa64 pair128 | 256 | 1.225 | 2.839 | 2.913 | 1.122 | 0.946 | 0.844 | 0.916 | 0.772 | OK |  |
| OuterProductMean | training | bf16 | S1024 msa64 pair128 | 384 | 2.514 | 5.316 | 5.389 | 2.365 | 1.969 | 0.833 | 0.941 | 0.783 | OK |  |
| OuterProductMean | training | bf16 | S1024 msa64 pair128 | 768 | 9.116 | 16.990 | 17.009 | 8.510 | 7.266 | 0.854 | 0.933 | 0.797 | OK |  |
| MSAPairWeightedAveraging | inference | bf16 | S1024 8x32 pair128 | 256 | 0.635 | 1.571 | 1.616 | 0.327 | 0.240 | 0.734 | 0.515 | 0.378 | OK |  |
| MSAPairWeightedAveraging | inference | bf16 | S1024 8x32 pair128 | 384 | 0.973 | 2.462 | 2.503 | 0.500 | 0.405 | 0.810 | 0.514 | 0.416 | OK |  |
| MSAPairWeightedAveraging | inference | bf16 | S1024 8x32 pair128 | 768 | 2.138 | 5.707 | 5.752 | 1.190 | 1.072 | 0.901 | 0.557 | 0.502 | OK |  |
| MSAPairWeightedAveraging | training | bf16 | S1024 8x32 pair128 | 256 | 1.739 | 3.997 | 4.119 | 1.254 | 0.956 | 0.762 | 0.721 | 0.550 | OK |  |
| MSAPairWeightedAveraging | training | bf16 | S1024 8x32 pair128 | 384 | 2.623 | 5.828 | 5.943 | 1.862 | 1.409 | 0.757 | 0.710 | 0.537 | OK |  |
| MSAPairWeightedAveraging | training | bf16 | S1024 8x32 pair128 | 768 | 5.708 | 13.145 | 13.259 | 4.319 | 3.397 | 0.787 | 0.757 | 0.595 | OK |  |
| OuterProductMean RF3 (bias, divide before) | inference | bf16 | S512 msa64 pair128 | 256 | 0.333 | 0.875 | 0.913 | 0.223 | 0.221 | 0.992 | 0.670 | 0.665 | OK | cuda_vs_portable within noise (ratio 0.992, spread 0.008) |
| OuterProductMean RF3 (bias, divide before) | inference | bf16 | S512 msa64 pair128 | 384 | 0.682 | 1.702 | 1.743 | 0.460 | 0.443 | 0.963 | 0.674 | 0.649 | OK |  |
| OuterProductMean RF3 (bias, divide before) | training | bf16 | S512 msa64 pair128 | 256 | 0.865 | 2.029 | 2.125 | 0.829 | 0.864 | 1.042 | 0.959 | 0.999 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 1.04x faster than what H100 runs) |
| OuterProductMean RF3 (bias, divide before) | training | bf16 | S512 msa64 pair128 | 384 | 1.769 | 3.929 | 4.022 | 1.677 | 1.767 | 1.053 | 0.948 | 0.999 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 1.05x faster than what H100 runs) |
| OuterProductMean RF3 (bias, divide after) | inference | bf16 | S512 msa64 pair128 | 256 | 0.333 | 0.718 | 0.756 | 0.221 | 0.221 | 1.001 | 0.663 | 0.664 | VIOLATION(cuda_vs_portable) | cuda_vs_portable within noise (ratio 1.001, spread 0.008) |
| OuterProductMean RF3 (bias, divide after) | inference | bf16 | S512 msa64 pair128 | 384 | 0.686 | 1.351 | 1.387 | 0.457 | 0.447 | 0.978 | 0.666 | 0.651 | OK |  |
| OuterProductMean RF3 (bias, divide after) | training | bf16 | S512 msa64 pair128 | 256 | 0.873 | 1.714 | 1.801 | 0.887 | 0.867 | 0.978 | 1.016 | 0.993 | VIOLATION(portable_vs_pytorch) | portable_vs_pytorch within noise (ratio 1.016, spread 0.010); n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference |
| OuterProductMean RF3 (bias, divide after) | training | bf16 | S512 msa64 pair128 | 384 | 1.809 | 3.233 | 3.312 | 1.816 | 1.801 | 0.992 | 1.004 | 0.996 | VIOLATION(portable_vs_pytorch) | portable_vs_pytorch within noise (ratio 1.004, spread 0.004); n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference |
| OuterProductMean Chai-1 (8 groups) | inference | bf16 | S512 pair256 8x64 | 256 | 0.288 | 0.980 | 1.008 | 0.218 | 0.277 | 1.272 | 0.757 | 0.963 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 1.27x faster than what H100 runs) |
| OuterProductMean Chai-1 (8 groups) | inference | bf16 | S512 pair256 8x64 | 384 | 0.577 | 1.865 | 1.893 | 0.446 | 0.552 | 1.238 | 0.772 | 0.957 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 1.24x faster than what H100 runs) |
| OuterProductMean Chai-1 (8 groups) | training | bf16 | S512 pair256 8x64 | 256 | 0.811 | 2.410 | 2.492 | 0.758 | 0.811 | 1.069 | 0.935 | 1.000 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 1.07x faster than what H100 runs) |
| OuterProductMean Chai-1 (8 groups) | training | bf16 | S512 pair256 8x64 | 384 | 1.601 | 4.227 | 4.286 | 1.451 | 1.594 | 1.099 | 0.906 | 0.996 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 1.10x faster than what H100 runs) |
| MSAPairWeightedAveraging value_dim 8 | inference | bf16 | S512 8x8 pair128 | 256 | 0.130 | 0.552 | 0.595 | 0.104 | 0.074 | 0.712 | 0.802 | 0.571 | OK |  |
| MSAPairWeightedAveraging value_dim 8 | inference | bf16 | S512 8x8 pair128 | 384 | 0.229 | 0.949 | 0.986 | 0.161 | 0.115 | 0.714 | 0.703 | 0.502 | OK |  |
| MSAPairWeightedAveraging value_dim 8 | training | bf16 | S512 8x8 pair128 | 256 | 0.468 | 1.504 | 1.621 | 0.426 | 0.367 | 0.862 | 0.910 | 0.784 | OK |  |
| MSAPairWeightedAveraging value_dim 8 | training | bf16 | S512 8x8 pair128 | 384 | 0.744 | 2.153 | 2.254 | 0.655 | 0.551 | 0.841 | 0.880 | 0.740 | OK |  |
| AttentionPairBias | inference | bf16 | 8x48 pair128 | 384 | 0.106 | 0.416 | 0.454 | 0.101 | 0.041 | 0.408 | 0.958 | 0.391 | OK |  |
| AttentionPairBias | inference | bf16 | 8x48 pair128 | 768 | 0.259 | 1.433 | 1.469 | 0.236 | 0.090 | 0.380 | 0.914 | 0.347 | OK |  |
| AttentionPairBias | training | bf16 | 8x48 pair128 | 384 | 0.308 | 0.897 | 1.166 | 0.269 | 0.122 | 0.454 | 0.873 | 0.397 | OK |  |
| AttentionPairBias | training | bf16 | 8x48 pair128 | 768 | 0.741 | 2.934 | 3.039 | 0.655 | 0.271 | 0.413 | 0.884 | 0.365 | OK |  |
| AttentionPairBias | inference | bf16 | 16x24 pair128 | 384 | 0.107 | 0.425 | 0.466 | 0.105 | 0.042 | 0.396 | 0.986 | 0.390 | OK | portable_vs_pytorch within noise (ratio 0.986, spread 0.005) |
| AttentionPairBias | training | bf16 | 16x24 pair128 | 384 | 0.309 | 0.910 | 1.168 | 0.279 | 0.129 | 0.464 | 0.904 | 0.419 | OK |  |
| AttentionPairBias | inference | bf16 | 8x48 pair256 | 384 | 0.152 | 0.564 | 0.602 | 0.147 | 0.158 | 1.073 | 0.971 | 1.042 | VIOLATION(cuda_vs_portable) |  |
| AttentionPairBias | training | bf16 | 8x48 pair256 | 384 | 0.423 | 1.254 | 1.356 | 0.406 | 0.419 | 1.033 | 0.959 | 0.991 | VIOLATION(cuda_vs_portable) |  |
| AugmentedAttention token | inference | bf16 | d768 A5/A48 | 384 | 0.211 | 0.722 | 0.820 | 0.162 | 0.212 | 1.309 | 0.767 | 1.004 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 1.31x faster than what H100 runs) |
| AugmentedAttention token | inference | bf16 | d768 A5/A48 | 768 | 0.550 | 2.425 | 2.527 | 0.387 | 0.553 | 1.429 | 0.704 | 1.006 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 1.43x faster than what H100 runs) |
| AugmentedAttention token | training | bf16 | d768 A5/A48 | 384 | 4.326 | 7.999 | 8.268 | 2.587 | 4.328 | 1.673 | 0.598 | 1.000 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 1.67x faster than what H100 runs) |
| AugmentedAttention token | training | bf16 | d768 A5/A48 | 768 | 14.451 | 25.217 | 25.460 | 7.283 | 14.424 | 1.981 | 0.504 | 0.998 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 1.98x faster than what H100 runs) |
| AugmentedAttention token | inference | fp32 | d768 A5/A48 | 384 | 0.338 | 0.658 | 0.722 | 0.286 | 0.343 | 1.196 | 0.847 | 1.013 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 1.20x faster than what H100 runs) |
| AugmentedAttention token | training | fp32 | d768 A5/A48 | 384 | 6.169 | 9.110 | 9.256 | 5.226 | 6.161 | 1.179 | 0.847 | 0.999 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 1.18x faster than what H100 runs) |
| AugmentedAttention atom | inference | bf16 | d128 A5/A48 | 128 | 0.238 | 2.229 | 2.318 | 0.182 | 0.242 | 1.336 | 0.764 | 1.021 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 1.34x faster than what H100 runs) |
| AugmentedAttention atom | inference | bf16 | d128 A5/A48 | 384 | 1.797 | 19.160 | 19.273 | 1.136 | 1.843 | 1.622 | 0.632 | 1.025 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 1.62x faster than what H100 runs) |
| AugmentedAttention atom | training | bf16 | d128 A5/A48 | 128 | 4.038 | 12.033 | 12.276 | 2.225 | 4.028 | 1.810 | 0.551 | 0.997 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 1.81x faster than what H100 runs) |
| AugmentedAttention atom | training | bf16 | d128 A5/A48 | 384 | 37.595 | 99.244 | 98.168 | 15.200 | 37.574 | 2.472 | 0.404 | 0.999 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 2.47x faster than what H100 runs) |
| token DiT (fused+composed) | inference | bf16 | d768 16x48 A5/A48 shared-cond | 384 | 0.281 | 0.858 | 1.309 | 0.243 | 0.104 | 0.427 | 0.863 | 0.368 | OK |  |
| token DiT (fused+composed) | inference | bf16 | d768 16x48 A5/A48 shared-cond | 768 | 0.666 | 2.646 | 2.790 | 0.504 | 0.181 | 0.359 | 0.757 | 0.272 | OK |  |
| token DiT (fused+composed) | training | bf16 | d768 16x48 A5/A48 shared-cond | 384 | 5.638 | 10.526 | 10.937 | 3.959 | 3.395 | 0.858 | 0.702 | 0.602 | OK |  |
| token DiT (fused+composed) | training | bf16 | d768 16x48 A5/A48 shared-cond | 768 | 17.017 | 30.057 | 30.429 | 9.851 | 7.803 | 0.792 | 0.579 | 0.459 | OK |  |
| token DiT (fused+composed) | inference | fp32 | d768 A5/A48 shared-cond | 384 | 0.471 | 0.834 | 0.944 | 0.410 | 0.197 | 0.479 | 0.872 | 0.418 | OK |  |
| token DiT (fused+composed) | training | fp32 | d768 A5/A48 shared-cond | 384 | 9.156 | 13.148 | 13.404 | 8.260 | 7.200 | 0.872 | 0.902 | 0.786 | OK |  |
| token DiT (fused+composed) | inference | bf16 | d768 qk-norm shared-cond | 384 | 0.285 | 0.924 | 1.331 | 0.250 | 0.108 | 0.431 | 0.876 | 0.378 | OK |  |
| token DiT (fused+composed) | training | bf16 | d768 qk-norm shared-cond | 384 | 5.804 | 11.895 | 12.341 | 4.064 | 3.470 | 0.854 | 0.700 | 0.598 | OK |  |
| token DiT + extra_pair_bias | inference | bf16 | d768 A2 shared-cond +[L,L] bias | 256 | 0.152 | 0.400 | 1.330 | 0.152 | 0.062 | 0.409 | 1.000 | 0.409 | OK | portable_vs_pytorch within noise (ratio 1.000, spread 0.010) |
| token DiT + extra_pair_bias | inference | bf16 | d768 A2 shared-cond +[L,L] bias | 512 | 0.266 | 0.984 | 1.321 | 0.243 | 0.076 | 0.311 | 0.916 | 0.285 | OK |  |
| token DiT + extra_pair_bias | training | bf16 | d768 A2 shared-cond +[L,L] bias | 256 | 0.426 | 1.190 | 3.681 | 0.463 | 0.419 | 0.905 | 1.084 | 0.982 | VIOLATION(portable_vs_pytorch) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference |
| token DiT + extra_pair_bias | training | bf16 | d768 A2 shared-cond +[L,L] bias | 512 | 0.701 | 2.188 | 3.804 | 0.748 | 0.699 | 0.934 | 1.068 | 0.997 | VIOLATION(portable_vs_pytorch) |  |
| token DiT + extra_pair_bias | inference | fp32 | d768 A2 shared-cond +[L,L] bias | 256 | 0.206 | 0.359 | 0.818 | 0.224 | 0.205 | 0.917 | 1.088 | 0.998 | VIOLATION(portable_vs_pytorch) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference |
| token DiT + extra_pair_bias | training | fp32 | d768 A2 shared-cond +[L,L] bias | 256 | 0.684 | 1.216 | 2.639 | 0.683 | 0.684 | 1.000 | 0.999 | 0.999 | VIOLATION(cuda_vs_portable) | cuda_vs_portable within noise (ratio 1.000, spread 0.010); portable_vs_pytorch within noise (ratio 0.999, spread 0.010) |
| DiT variant stack | inference | bf16 | Boltz-2 token | 384 | 0.292 | 0.901 | 1.039 | 0.215 | 0.117 | 0.545 | 0.737 | 0.402 | OK |  |
| DiT variant stack | training | bf16 | Boltz-2 token | 384 | 5.604 | 10.331 | 10.741 | 3.736 | 5.603 | 1.500 | 0.667 | 1.000 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 1.50x faster than what H100 runs) |
| DiT variant stack | inference | bf16 | RoseTTAFold3 token | 384 | 0.274 | 0.905 | 1.118 | 0.196 | 0.109 | 0.553 | 0.717 | 0.397 | OK |  |
| DiT variant stack | training | bf16 | RoseTTAFold3 token | 384 | 5.448 | 10.607 | 11.004 | 3.584 | 5.471 | 1.527 | 0.658 | 1.004 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 1.53x faster than what H100 runs) |
| DiT variant stack | inference | bf16 | Chai-1 token | 384 | 0.294 | 0.983 | 1.106 | 0.208 | 0.109 | 0.526 | 0.707 | 0.372 | OK | cuda_vs_portable within noise (ratio 0.526, spread 0.684); portable_vs_pytorch within noise (ratio 0.707, spread 0.684) |
| DiT variant stack | training | bf16 | Chai-1 token | 384 | 5.120 | 9.921 | 10.190 | 3.214 | 5.124 | 1.594 | 0.628 | 1.001 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 1.59x faster than what H100 runs) |
| LocalDiT atom block | inference | bf16 | d128 32x128 (atoms=8L) | 512 | 0.379 | 1.722 | 1.906 | 0.217 | 0.097 | 0.448 | 0.573 | 0.257 | OK |  |
| LocalDiT atom block | inference | bf16 | d128 32x128 (atoms=8L) | 1024 | 0.687 | 3.298 | 3.497 | 0.342 | 0.164 | 0.480 | 0.498 | 0.239 | OK |  |
| LocalDiT atom block | training | bf16 | d128 32x128 (atoms=8L) | 512 | 8.554 | 25.238 | 25.631 | 5.570 | 3.496 | 0.628 | 0.651 | 0.409 | OK |  |
| LocalDiT atom block | training | bf16 | d128 32x128 (atoms=8L) | 1024 | 16.813 | 49.182 | 49.669 | 11.012 | 6.642 | 0.603 | 0.655 | 0.395 | OK |  |
| LocalDiT atom block | inference | fp32 | d128 32x128 (atoms=8L) | 512 | 0.534 | 1.864 | 1.983 | 0.389 | 0.153 | 0.393 | 0.729 | 0.286 | OK |  |
| LocalDiT atom block | training | fp32 | d128 32x128 (atoms=8L) | 512 | 12.455 | 26.134 | 26.439 | 10.568 | 5.796 | 0.548 | 0.848 | 0.465 | OK |  |
| LocalDiT ESMFold2 atom block | inference | bf16 | d128 192 keys rotary (atoms=8L) | 512 | 0.515 | 2.615 | 2.914 | 0.238 | 0.094 | 0.394 | 0.463 | 0.182 | OK |  |
| LocalDiT ESMFold2 atom block | inference | bf16 | d128 192 keys rotary (atoms=8L) | 1024 | 0.979 | 5.108 | 5.418 | 0.423 | 0.167 | 0.395 | 0.432 | 0.171 | OK |  |
| LocalDiT ESMFold2 atom block | training | bf16 | d128 192 keys rotary (atoms=8L) | 512 | 12.665 | 36.725 | 37.426 | 6.568 | 12.664 | 1.928 | 0.519 | 1.000 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 1.93x faster than what H100 runs) |
| LocalDiT ESMFold2 atom block | training | bf16 | d128 192 keys rotary (atoms=8L) | 1024 | 25.737 | 73.364 | 73.700 | 12.806 | 25.772 | 2.013 | 0.498 | 1.001 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 2.01x faster than what H100 runs) |
| DiT variant stack (atom) | inference | bf16 | Boltz-2 atom | 512 | 0.329 | 1.890 | 2.073 | 0.195 | 0.159 | 0.816 | 0.593 | 0.484 | OK |  |
| DiT variant stack (atom) | training | bf16 | Boltz-2 atom | 512 | 9.353 | 20.419 | 20.983 | 4.428 | 9.340 | 2.109 | 0.473 | 0.999 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 2.11x faster than what H100 runs) |
| DiT variant stack (atom) | inference | bf16 | RoseTTAFold3 atom | 512 | 0.331 | 1.967 | 2.156 | 0.192 | 0.155 | 0.811 | 0.578 | 0.469 | OK |  |
| DiT variant stack (atom) | training | bf16 | RoseTTAFold3 atom | 512 | 9.357 | 21.555 | 22.090 | 4.147 | 9.353 | 2.255 | 0.443 | 1.000 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 2.26x faster than what H100 runs) |
| DiT variant stack (atom) | inference | bf16 | Chai-1 atom | 512 | 0.296 | 1.820 | 1.983 | 0.164 | 0.129 | 0.788 | 0.554 | 0.437 | OK |  |
| DiT variant stack (atom) | training | bf16 | Chai-1 atom | 512 | 8.002 | 19.221 | 19.654 | 3.604 | 8.005 | 2.221 | 0.450 | 1.000 | n.a.(cuda rung missing) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 2.22x faster than what H100 runs) |
| bias-only DiT | inference | bf16 | d768 16x48 A5/A48 | 256 | 0.156 | 0.374 | 0.914 | 0.165 | 0.075 | 0.453 | 1.060 | 0.481 | VIOLATION(portable_vs_pytorch) |  |
| bias-only DiT | inference | bf16 | d768 16x48 A5/A48 | 512 | 0.280 | 0.941 | 1.074 | 0.296 | 0.122 | 0.410 | 1.058 | 0.434 | VIOLATION(portable_vs_pytorch) |  |
| bias-only DiT | inference | bf16 | d768 16x48 A5/A48 | 768 | 0.429 | 1.855 | 1.988 | 0.441 | 0.173 | 0.392 | 1.028 | 0.403 | VIOLATION(portable_vs_pytorch) |  |
| bias-only DiT | training | bf16 | d768 16x48 A5/A48 | 256 | 1.670 | 3.700 | 4.011 | 1.798 | 1.463 | 0.814 | 1.077 | 0.876 | VIOLATION(portable_vs_pytorch) |  |
| bias-only DiT | training | bf16 | d768 16x48 A5/A48 | 512 | 3.313 | 7.346 | 7.674 | 3.552 | 2.850 | 0.802 | 1.072 | 0.860 | VIOLATION(portable_vs_pytorch) |  |
| bias-only DiT | training | bf16 | d768 16x48 A5/A48 | 768 | 5.044 | 11.691 | 12.014 | 5.356 | 4.351 | 0.812 | 1.062 | 0.863 | VIOLATION(portable_vs_pytorch) |  |
| bias-only DiT | inference | fp32 | d768 16x48 A5/A48 | 256 | 0.216 | 0.345 | 0.568 | 0.237 | 0.136 | 0.574 | 1.100 | 0.632 | VIOLATION(portable_vs_pytorch) |  |
| bias-only DiT | inference | fp32 | d768 16x48 A5/A48 | 512 | 0.435 | 0.834 | 0.912 | 0.453 | 0.231 | 0.509 | 1.042 | 0.531 | VIOLATION(portable_vs_pytorch) |  |
| bias-only DiT | training | fp32 | d768 16x48 A5/A48 | 256 | 3.877 | 5.519 | 5.685 | 4.021 | 3.533 | 0.879 | 1.037 | 0.911 | VIOLATION(portable_vs_pytorch) |  |
| bias-only DiT | training | fp32 | d768 16x48 A5/A48 | 512 | 7.218 | 10.166 | 10.338 | 7.578 | 6.688 | 0.883 | 1.050 | 0.927 | VIOLATION(portable_vs_pytorch) |  |
| bias-only DiT | inference | bf16 | 24x32 | 512 | 0.281 | 0.976 | 1.090 | 0.293 | 0.126 | 0.431 | 1.043 | 0.449 | VIOLATION(portable_vs_pytorch) |  |
| bias-only DiT | training | bf16 | 24x32 | 512 | 3.336 | 7.430 | 7.759 | 3.578 | 3.054 | 0.854 | 1.073 | 0.916 | VIOLATION(portable_vs_pytorch) |  |
| bias-only DiT | inference | bf16 | 12x64 | 512 | 0.284 | 0.939 | 1.068 | 0.301 | 0.124 | 0.412 | 1.062 | 0.438 | VIOLATION(portable_vs_pytorch) |  |
| bias-only DiT | training | bf16 | 12x64 | 512 | 3.314 | 7.327 | 7.629 | 3.515 | 2.889 | 0.822 | 1.061 | 0.872 | VIOLATION(portable_vs_pytorch) |  |
| SWA DiT (atom) | inference | bf16 | d128 atoms=8L shared-cond | 512 | 1.189 | 3.257 | 3.407 | - | 0.107 | - | - | 0.090 | n.a. | n.a.: cuda_vs_portable: portable: RuntimeError: native H100 SWA rejects force_portable_kernels | portable_vs_pytorch: portable: RuntimeError: native H100 SWA rejects force_portable_kernels |
| SWA DiT (atom) | inference | bf16 | d128 atoms=8L shared-cond | 1024 | 4.282 | 11.462 | 11.618 | - | 0.196 | - | - | 0.046 | n.a. | n.a.: cuda_vs_portable: portable: RuntimeError: native H100 SWA rejects force_portable_kernels | portable_vs_pytorch: portable: RuntimeError: native H100 SWA rejects force_portable_kernels |
| SWA DiT (atom) | training | bf16 | d128 atoms=8L shared-cond | 512 | 31.619 | 55.076 | 55.477 | - | 2.223 | - | - | 0.070 | n.a. | n.a.: cuda_vs_portable: portable: RuntimeError: native H100 SWA rejects force_portable_kernels | portable_vs_pytorch: portable: RuntimeError: native H100 SWA rejects force_portable_kernels |
| SWA DiT (atom) | training | bf16 | d128 atoms=8L shared-cond | 1024 | 120.6 | 197.9 | 198.0 | - | 4.260 | - | - | 0.035 | n.a. | n.a.: cuda_vs_portable: portable: RuntimeError: native H100 SWA rejects force_portable_kernels | portable_vs_pytorch: portable: RuntimeError: native H100 SWA rejects force_portable_kernels |
| SWA DiT (atom) | inference | fp32 | atoms=8L shared-cond | 512 | 3.345 | 5.139 | 5.364 | - | 0.221 | - | - | 0.066 | n.a. | n.a.: cuda_vs_portable: portable: RuntimeError: native H100 SWA rejects force_portable_kernels | portable_vs_pytorch: portable: RuntimeError: native H100 SWA rejects force_portable_kernels |
| SWA DiT (atom) | training | fp32 | atoms=8L shared-cond | 512 | 88.121 | 110.7 | 110.9 | - | 4.615 | - | - | 0.052 | n.a. | n.a.: cuda_vs_portable: portable: RuntimeError: native H100 SWA rejects force_portable_kernels | portable_vs_pytorch: portable: RuntimeError: native H100 SWA rejects force_portable_kernels |
| SWA atom attention (global) | inference | bf16 | atoms=8L | 512 | 1.118 | 3.009 | 3.132 | - | 0.076 | - | - | 0.068 | OK | no portable rung by design |
| SWA atom attention (global) | inference | bf16 | atoms=8L | 1024 | 4.139 | 11.037 | 11.151 | - | 0.133 | - | - | 0.032 | OK | no portable rung by design |
| SWA atom attention (global) | training | bf16 | atoms=8L | 512 | 29.552 | 49.994 | 50.207 | - | 2.043 | - | - | 0.069 | OK | no portable rung by design |
| SWA atom attention (global) | training | bf16 | atoms=8L | 1024 | 116.7 | 187.6 | 188.3 | - | 3.973 | - | - | 0.034 | OK | no portable rung by design |
| AdaLN | inference | bf16 | d768 cond384 A5/A48 | 384 | 0.027 | 0.061 | 0.110 | 0.025 | 0.026 | 1.029 | 0.942 | 0.969 | VIOLATION(cuda_vs_portable) | cuda_vs_portable within noise (ratio 1.029, spread 0.050) |
| AdaLN | inference | bf16 | d768 cond384 A5/A48 | 768 | 0.036 | 0.093 | 0.116 | 0.035 | 0.034 | 0.971 | 0.986 | 0.958 | OK | portable_vs_pytorch within noise (ratio 0.986, spread 0.006) |
| AdaLN | inference | bf16 | d768 cond384 A5/A48 | 1024 | 0.041 | 0.115 | 0.139 | 0.039 | 0.039 | 0.993 | 0.958 | 0.951 | OK | cuda_vs_portable within noise (ratio 0.993, spread 0.003) |
| AdaLN | training | bf16 | d768 cond384 A5/A48 | 384 | 0.372 | 1.005 | 1.062 | 0.337 | 0.304 | 0.902 | 0.905 | 0.816 | OK |  |
| AdaLN | training | bf16 | d768 cond384 A5/A48 | 768 | 0.690 | 1.905 | 1.955 | 0.618 | 0.557 | 0.902 | 0.895 | 0.808 | OK |  |
| AdaLN | training | bf16 | d768 cond384 A5/A48 | 1024 | 0.885 | 2.498 | 2.546 | 0.793 | 0.711 | 0.896 | 0.896 | 0.804 | OK |  |
| AdaLN | inference | fp32 | d768 cond384 A5/A48 | 384 | 0.034 | 0.049 | 0.059 | 0.040 | 0.035 | 0.858 | 1.176 | 1.008 | VIOLATION(portable_vs_pytorch) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference |
| AdaLN | training | fp32 | d768 cond384 A5/A48 | 384 | 0.643 | 1.038 | 1.073 | 0.699 | 0.641 | 0.918 | 1.088 | 0.998 | VIOLATION(portable_vs_pytorch) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference |
| ConditionedTransition | inference | bf16 | d768 A5/A48 | 384 | 0.083 | 0.131 | 0.323 | 0.087 | 0.082 | 0.938 | 1.051 | 0.986 | VIOLATION(portable_vs_pytorch) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference |
| ConditionedTransition | inference | bf16 | d768 A5/A48 | 768 | 0.124 | 0.204 | 0.320 | 0.131 | 0.120 | 0.916 | 1.058 | 0.969 | VIOLATION(portable_vs_pytorch) |  |
| ConditionedTransition | inference | bf16 | d768 A5/A48 | 1024 | 0.148 | 0.260 | 0.328 | 0.171 | 0.144 | 0.841 | 1.153 | 0.970 | VIOLATION(portable_vs_pytorch) |  |
| ConditionedTransition | training | bf16 | d768 A5/A48 | 384 | 1.245 | 2.355 | 2.471 | 1.335 | 1.238 | 0.928 | 1.072 | 0.994 | VIOLATION(portable_vs_pytorch) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference |
| ConditionedTransition | training | bf16 | d768 A5/A48 | 768 | 2.301 | 4.429 | 4.542 | 2.473 | 2.300 | 0.930 | 1.075 | 1.000 | VIOLATION(portable_vs_pytorch) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference |
| ConditionedTransition | training | bf16 | d768 A5/A48 | 1024 | 2.986 | 5.789 | 5.906 | 3.206 | 2.979 | 0.929 | 1.074 | 0.998 | VIOLATION(portable_vs_pytorch) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference |
| ConditionedTransition | inference | fp32 | d768 A5/A48 | 384 | 0.137 | 0.163 | 0.190 | 0.125 | 0.136 | 1.092 | 0.913 | 0.998 | n.a.(cuda rung missing) | portable_vs_pytorch within noise (ratio 0.913, spread 0.333); n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference (the portable kernels are 1.09x faster than what H100 runs) |
| ConditionedTransition | training | fp32 | d768 A5/A48 | 384 | 2.902 | 3.771 | 3.834 | 3.004 | 2.894 | 0.963 | 1.035 | 0.997 | VIOLATION(portable_vs_pytorch) | n.a.: cuda_vs_portable: no CUDA path serves this call: `miniworld` ran the PyTorch reference |
| LayerNorm (pair) | inference | bf16 | x[1,L,L,128] | 384 | 0.031 | 0.165 | 0.165 | 0.032 | 0.032 | 0.998 | 1.036 | 1.033 | VIOLATION(portable_vs_pytorch) | cuda_vs_portable within noise (ratio 0.998, spread 0.005) |
| LayerNorm (pair) | inference | bf16 | x[1,L,L,128] | 768 | 0.106 | 0.642 | 0.642 | 0.107 | 0.107 | 1.001 | 1.010 | 1.011 | VIOLATION(portable_vs_pytorch) | portable_vs_pytorch within noise (ratio 1.010, spread 0.002); n.a.: cuda_vs_portable: cuda rung launched the portable kernels (no distinct CUDA path) |
| LayerNorm (pair) | training | bf16 | x[1,L,L,128] | 384 | 0.094 | 0.360 | 0.365 | 0.088 | 0.104 | 1.184 | 0.935 | 1.107 | VIOLATION(cuda_vs_portable) |  |
| LayerNorm (pair) | training | bf16 | x[1,L,L,128] | 768 | 0.300 | 1.334 | 1.339 | 0.296 | 0.360 | 1.217 | 0.984 | 1.198 | VIOLATION(cuda_vs_portable) | portable_vs_pytorch within noise (ratio 0.984, spread 0.002) |
| LayerNorm (pair) | inference | fp32 | x[1,L,L,128] | 384 | 0.055 | 0.225 | 0.226 | 0.057 | 0.057 | 1.001 | 1.034 | 1.035 | VIOLATION(portable_vs_pytorch) | n.a.: cuda_vs_portable: cuda rung launched the portable kernels (no distinct CUDA path) |
| LayerNorm (pair) | training | fp32 | x[1,L,L,128] | 384 | 0.157 | 0.450 | 0.456 | 0.138 | 0.137 | 0.999 | 0.878 | 0.878 | OK | n.a.: cuda_vs_portable: cuda rung launched the portable kernels (no distinct CUDA path) |
| LayerNorm (pair) | inference | bf16 | x[1,L,L,256] | 384 | 0.056 | 0.193 | 0.193 | 0.059 | 0.059 | 1.006 | 1.040 | 1.046 | VIOLATION(cuda_vs_portable)+VIOLATION(portable_vs_pytorch) | cuda_vs_portable within noise (ratio 1.006, spread 0.002) |
| LayerNorm (pair) | training | bf16 | x[1,L,L,256] | 384 | 0.163 | 0.433 | 0.439 | 0.175 | 0.165 | 0.944 | 1.072 | 1.012 | VIOLATION(portable_vs_pytorch) |  |
| RMSNorm | inference | bf16 | x[48,L,768] | 384 | 0.024 | 0.030 | 0.030 | 0.028 | 0.031 | 1.090 | 1.165 | 1.270 | VIOLATION(cuda_vs_portable)+VIOLATION(portable_vs_pytorch) |  |
| RMSNorm | inference | bf16 | x[48,L,768] | 1024 | 0.056 | 0.067 | 0.067 | 0.080 | 0.068 | 0.849 | 1.421 | 1.207 | VIOLATION(portable_vs_pytorch) |  |
| RMSNorm | training | bf16 | x[48,L,768] | 384 | 0.069 | 0.144 | 0.147 | 0.084 | 0.072 | 0.853 | 1.230 | 1.050 | VIOLATION(portable_vs_pytorch) |  |
| RMSNorm | training | bf16 | x[48,L,768] | 1024 | 0.165 | 0.346 | 0.348 | 0.212 | 0.155 | 0.734 | 1.281 | 0.939 | VIOLATION(portable_vs_pytorch) |  |
| pair_embed_geom | inference | bf16 out, fp32 geometry | B16 | 128 | 1.280 | 3.151 | 3.419 | 0.137 | 0.161 | 1.179 | 0.107 | 0.126 | VIOLATION(cuda_vs_portable) |  |
| pair_embed_geom | inference | bf16 out, fp32 geometry | B16 | 256 | 4.693 | 12.201 | 12.500 | 0.452 | 0.528 | 1.168 | 0.096 | 0.113 | VIOLATION(cuda_vs_portable) |  |
| pair_embed_geom | inference | bf16 out, fp32 geometry | B16 | 384 | 10.563 | 27.633 | 27.943 | 0.982 | 1.152 | 1.173 | 0.093 | 0.109 | VIOLATION(cuda_vs_portable) |  |
| pair_embed_geom | training | bf16 out, fp32 geometry | B16 | 128 | 1.827 | 4.618 | 4.885 | 0.650 | 0.752 | 1.158 | 0.356 | 0.412 | VIOLATION(cuda_vs_portable) |  |
| pair_embed_geom | training | bf16 out, fp32 geometry | B16 | 256 | 6.850 | 17.953 | 18.252 | 2.097 | 2.763 | 1.318 | 0.306 | 0.403 | VIOLATION(cuda_vs_portable) |  |
| pair_embed_geom | training | bf16 out, fp32 geometry | B16 | 384 | 15.244 | 40.557 | 40.919 | 4.484 | 6.208 | 1.384 | 0.294 | 0.407 | VIOLATION(cuda_vs_portable) |  |
| pair_embed_geom | inference | bf16 out, fp32 geometry | B48 | 128 | 3.560 | 9.167 | 9.454 | 0.345 | 0.374 | 1.083 | 0.097 | 0.105 | VIOLATION(cuda_vs_portable) |  |
| pair_embed_geom | inference | bf16 out, fp32 geometry | B48 | 256 | 13.683 | 37.017 | 37.357 | 1.287 | 1.434 | 1.114 | 0.094 | 0.105 | VIOLATION(cuda_vs_portable) |  |
| pair_embed_geom | training | bf16 out, fp32 geometry | B48 | 128 | 5.167 | 13.517 | 13.805 | 1.614 | 2.071 | 1.283 | 0.312 | 0.401 | VIOLATION(cuda_vs_portable) |  |
| pair_embed_geom | training | bf16 out, fp32 geometry | B48 | 256 | 20.146 | 54.295 | 54.672 | 5.935 | 7.929 | 1.336 | 0.295 | 0.394 | VIOLATION(cuda_vs_portable) |  |
| pair_embed_ca | inference | bf16 out, fp32 geometry | B16 | 128 | 0.497 | 1.386 | 1.439 | 0.066 | 0.065 | 0.983 | 0.133 | 0.130 | OK | cuda_vs_portable within noise (ratio 0.983, spread 0.004) |
| pair_embed_ca | inference | bf16 out, fp32 geometry | B16 | 256 | 1.841 | 5.543 | 5.594 | 0.215 | 0.206 | 0.959 | 0.117 | 0.112 | OK |  |
| pair_embed_ca | inference | bf16 out, fp32 geometry | B16 | 384 | 4.131 | 12.484 | 12.555 | 0.462 | 0.440 | 0.952 | 0.112 | 0.106 | OK |  |
| pair_embed_ca | training | bf16 out, fp32 geometry | B16 | 128 | 0.856 | 2.727 | 2.795 | 0.335 | 0.438 | 1.308 | 0.392 | 0.512 | VIOLATION(cuda_vs_portable) |  |
| pair_embed_ca | training | bf16 out, fp32 geometry | B16 | 256 | 3.242 | 10.839 | 10.907 | 1.020 | 1.622 | 1.590 | 0.315 | 0.500 | VIOLATION(cuda_vs_portable) |  |
| pair_embed_ca | training | bf16 out, fp32 geometry | B16 | 384 | 7.261 | 24.367 | 24.468 | 2.116 | 3.628 | 1.714 | 0.291 | 0.500 | VIOLATION(cuda_vs_portable) |  |
| pair_embed_ca | inference | bf16 out, fp32 geometry | B48 | 128 | 1.398 | 4.164 | 4.213 | 0.165 | 0.159 | 0.960 | 0.118 | 0.113 | OK |  |
| pair_embed_ca | inference | bf16 out, fp32 geometry | B48 | 256 | 5.446 | 16.694 | 16.757 | 0.610 | 0.579 | 0.949 | 0.112 | 0.106 | OK |  |
| pair_embed_ca | training | bf16 out, fp32 geometry | B48 | 128 | 2.442 | 8.132 | 8.217 | 0.758 | 1.229 | 1.620 | 0.311 | 0.503 | VIOLATION(cuda_vs_portable) |  |
| pair_embed_ca | training | bf16 out, fp32 geometry | B48 | 256 | 9.614 | 32.550 | 32.644 | 2.720 | 4.694 | 1.726 | 0.283 | 0.488 | VIOLATION(cuda_vs_portable) |  |
| pair_embed_adaln | inference | bf16 out, fp32 geometry | B16 | 128 | 0.530 | 3.215 | 3.277 | 0.209 | 0.139 | 0.662 | 0.395 | 0.262 | OK |  |
| pair_embed_adaln | inference | bf16 out, fp32 geometry | B16 | 256 | 1.925 | 12.723 | 12.805 | 0.788 | 0.410 | 0.520 | 0.409 | 0.213 | OK |  |
| pair_embed_adaln | inference | bf16 out, fp32 geometry | B16 | 384 | 4.373 | 28.648 | 28.756 | 1.737 | 0.987 | 0.568 | 0.397 | 0.226 | OK |  |
| pair_embed_adaln | training | bf16 out, fp32 geometry | B16 | 128 | 1.538 | 5.696 | 5.796 | - | 0.975 | - | - | 0.634 | n.a. | n.a.: cuda_vs_portable: portable: RuntimeError: pair_embed_adaln_bwd (with implementation in <module 'torch._library.custom_ | portable_vs_pytorch: portable: RuntimeError: pair_embed_adaln_bwd (with implementation in <module 'torch._library.custom_ |
| pair_embed_adaln | training | bf16 out, fp32 geometry | B16 | 256 | 5.774 | 22.426 | 22.518 | - | 3.702 | - | - | 0.641 | n.a. | n.a.: cuda_vs_portable: portable: RuntimeError: pair_embed_adaln_bwd (with implementation in <module 'torch._library.custom_ | portable_vs_pytorch: portable: RuntimeError: pair_embed_adaln_bwd (with implementation in <module 'torch._library.custom_ |
| pair_embed_adaln | training | bf16 out, fp32 geometry | B16 | 384 | 12.854 | 50.397 | 50.526 | - | 8.725 | - | - | 0.679 | n.a. | n.a.: cuda_vs_portable: portable: RuntimeError: pair_embed_adaln_bwd (with implementation in <module 'torch._library.custom_ | portable_vs_pytorch: portable: RuntimeError: pair_embed_adaln_bwd (with implementation in <module 'torch._library.custom_ |
| pair_embed_adaln | inference | bf16 out, fp32 geometry | B48 | 128 | 1.464 | 9.606 | 9.683 | 0.589 | 0.320 | 0.543 | 0.402 | 0.219 | OK |  |
| pair_embed_adaln | inference | bf16 out, fp32 geometry | B48 | 256 | 5.634 | 38.210 | 38.329 | 2.308 | 0.983 | 0.426 | 0.410 | 0.174 | OK |  |
| pair_embed_adaln | training | bf16 out, fp32 geometry | B48 | 128 | 4.400 | 16.903 | 17.008 | - | 2.649 | - | - | 0.602 | n.a. | n.a.: cuda_vs_portable: portable: RuntimeError: pair_embed_adaln_bwd (with implementation in <module 'torch._library.custom_ | portable_vs_pytorch: portable: RuntimeError: pair_embed_adaln_bwd (with implementation in <module 'torch._library.custom_ |
| pair_embed_adaln | training | bf16 out, fp32 geometry | B48 | 256 | 17.365 | 67.197 | 67.362 | - | 10.673 | - | - | 0.615 | n.a. | n.a.: cuda_vs_portable: portable: RuntimeError: pair_embed_adaln_bwd (with implementation in <module 'torch._library.custom_ | portable_vs_pytorch: portable: RuntimeError: pair_embed_adaln_bwd (with implementation in <module 'torch._library.custom_ |
| token_pair_init | inference | fp32 in, fp32 out | B16 | 128 | 0.544 | 1.285 | 1.362 | 0.078 | 0.089 | 1.133 | 0.144 | 0.163 | VIOLATION(cuda_vs_portable) |  |
| token_pair_init | inference | fp32 in, fp32 out | B16 | 256 | 2.101 | 4.856 | 4.944 | 0.222 | 0.233 | 1.047 | 0.106 | 0.111 | VIOLATION(cuda_vs_portable) |  |
| token_pair_init | inference | fp32 in, fp32 out | B16 | 384 | 4.751 | 10.897 | 11.005 | 0.464 | 0.479 | 1.031 | 0.098 | 0.101 | VIOLATION(cuda_vs_portable) |  |
| token_pair_init | training | fp32 in, fp32 out | B16 | 128 | 0.913 | 1.596 | 1.688 | 0.497 | 0.305 | 0.613 | 0.545 | 0.334 | OK |  |
| token_pair_init | training | fp32 in, fp32 out | B16 | 256 | 3.140 | 5.968 | 6.071 | 1.461 | 1.012 | 0.692 | 0.465 | 0.322 | OK |  |
| token_pair_init | training | fp32 in, fp32 out | B16 | 384 | 7.105 | 13.274 | 13.403 | 3.042 | 2.170 | 0.713 | 0.428 | 0.305 | OK |  |
| token_pair_init | inference | fp32 in, fp32 out | B48 | 128 | 1.563 | 3.653 | 3.737 | 0.185 | 0.205 | 1.105 | 0.119 | 0.131 | VIOLATION(cuda_vs_portable) |  |
| token_pair_init | inference | fp32 in, fp32 out | B48 | 256 | 6.205 | 14.506 | 14.626 | 0.623 | 0.625 | 1.003 | 0.100 | 0.101 | VIOLATION(cuda_vs_portable) | cuda_vs_portable within noise (ratio 1.003, spread 0.066) |
| token_pair_init | training | fp32 in, fp32 out | B48 | 128 | 2.391 | 4.492 | 4.590 | 1.123 | 0.846 | 0.753 | 0.470 | 0.354 | OK |  |
| token_pair_init | training | fp32 in, fp32 out | B48 | 256 | 9.328 | 17.653 | 17.791 | 3.970 | 2.943 | 0.741 | 0.426 | 0.315 | OK |  |
| smooth_lddt | inference | fp32 | A48 B1 | 1024 | 63.153 | 65.893 | 65.953 | 0.161 | 0.161 | 1.002 | 0.003 | 0.003 | VIOLATION(cuda_vs_portable) | cuda_vs_portable within noise (ratio 1.002, spread 0.002) |
| smooth_lddt | inference | fp32 | A48 B1 | 2048 | 254.3 | 263.5 | 263.6 | 0.550 | 0.489 | 0.890 | 0.002 | 0.002 | OK |  |
| smooth_lddt | training | fp32 | A48 B1 | 1024 | 99.371 | 103.9 | 103.3 | 0.406 | 0.319 | 0.786 | 0.004 | 0.003 | OK |  |
| smooth_lddt | training | fp32 | A48 B1 | 2048 | 402.8 | 421.3 | 418.6 | 1.497 | 1.059 | 0.708 | 0.004 | 0.003 | OK |  |
| smooth_lddt | inference | fp32 | A16 B1 | 4096 | 350.3 | 362.6 | 362.7 | 0.815 | 0.594 | 0.728 | 0.002 | 0.002 | OK |  |
| smooth_lddt | training | fp32 | A16 B1 | 4096 | 548.0 | 572.0 | 571.4 | 2.089 | 1.279 | 0.612 | 0.004 | 0.002 | OK |  |
| sampler augment_noise | inference | fp32 | B5 | 4096 | 0.018 | 0.059 | 0.240 | - | 0.010 | - | - | 0.576 | OK | no portable rung by design |
| sampler augment_noise | inference | fp32 | B5 | 12288 | 0.023 | 0.071 | 0.244 | - | 0.012 | - | - | 0.508 | OK | no portable rung by design |
| sampler augment_noise | inference | fp32 | B5 | 24576 | 0.027 | 0.072 | 0.245 | - | 0.014 | - | - | 0.503 | OK | no portable rung by design |
| sampler align_euler | inference | fp32 | B5 | 4096 | 0.089 | 0.283 | 1.413 | - | 0.017 | - | - | 0.193 | OK | no portable rung by design |
| sampler align_euler | inference | fp32 | B5 | 12288 | 0.140 | 0.385 | 1.416 | - | 0.021 | - | - | 0.152 | OK | no portable rung by design |
| sampler align_euler | inference | fp32 | B5 | 24576 | 0.213 | 0.500 | 1.416 | - | 0.026 | - | - | 0.122 | OK | no portable rung by design |
| sampler pool_atoms | inference | bf16 | B5 c768 | 4096 | 0.079 | 0.259 | 0.282 | - | 0.022 | - | - | 0.276 | OK | no portable rung by design |
| sampler pool_atoms | inference | bf16 | B5 c768 | 12288 | 0.224 | 0.720 | 0.741 | - | 0.041 | - | - | 0.182 | OK | no portable rung by design |
| sampler pool_atoms | inference | bf16 | B5 c768 | 24576 | 0.442 | 1.411 | 1.434 | - | 0.070 | - | - | 0.158 | OK | no portable rung by design |
