# A100 80GB PCIe: Triton record (`run_all` at `9cf93c27`, 2026-11-01)

Base: `origin/main` 9cf93c27. Branch: `a100-triton-main` (worktree, staged, not committed). Card: NVIDIA A100 80GB PCIe (sm80), node gpu06,
torch 2.13 + cu129, triton 3.7 (the repo pixi env, `PYTHONPATH` = the worktree `src`). No Triton kernel was written or edited;
the registry, bands and code are unchanged. Only the A100 manifest and docs changed.

## Procedure

`run_all` records through `devices.record`, which merges and rewrites the `#provenance` row to the running commit, so (as in
`triton_gate_rows_20261031.md`) it was run against a scratch copy of the manifests: `devices._DEVICES` was pointed at
`~/miniworld-engine-scratch/triton_main/manifests_scratch/` (a copy of `autotune/manifests/`) by a three-line wrapper
(`wrap_run_all.py`), then `python wrap_run_all.py --verbose --json counts_<dtype>.json` with `MINIWORLD_DRIVER_DTYPE=bf16`, then `fp32`,
one Slurm job (83401, `-p A100 --gres=gpu:A100:1 -c 8 --mem=64G --no-requeue`, bf16 01:22-02:37, fp32 02:37-03:17), no `--only` filter.
Unlike the H100 partial run, this was a FULL run of every declared row for both precisions at one commit, and the A100 manifest on
`main` is the v1.0.0 one (provenance 1.0.0, 2026-09-01, 78 lines), so the scratch manifest REPLACES it whole: its provenance row
(2.2.0, 9cf93c278466, clean, v2.4.0-15-g9cf93c27) describes the run that produced every row in it, and no row is merged by hand.
The `failed` rows are recorded as failed; none was hidden or edited. The JIT / Triton caches of the run lived under
`~/miniworld-engine-scratch/triton_main/{ext,triton,jit}` (note: `MINIWORLD_ENGINE_JIT_ROOT` is no longer read from the environment;
the kernels run here are Triton, whose cache is `TRITON_CACHE_DIR`). Every autotuned kernel ran with no tuned A100 cache
("no tuned autotune cache" warning), i.e. its default config set, which is what an untuned card launches.

Logs: `~/miniworld-engine-scratch/triton_main/logs/{run_all_bf16.log,run_all_fp32.log,counts_bf16.json,counts_fp32.json,slurm_83401.out}`.

## Counts (200 declared rows)

| dtype | driven | ok | failed | skipped | checked against a reference |
|---|---|---|---|---|---|
| bf16 | 159 | 151 | 8 | 41 (30 `needs sm90`, 11 declare fp32 only) | 151 |
| fp32 | 71 | 71 | 0 | 129 (30 `needs sm90`, 99 declare bf16 only) | 71 |

Manifest after: 222 ok, 8 failed, 30 skipped rows (the skip rows are the sm90 CUDA rows). 0 launched-unverified, 0 mislabelled arch.

## Failures (bf16; fp32 has none)

| kernel | measured | declared band | root cause | kind |
|---|---|---|---|---|
| `cond_transition_fwd_b2b_saveact_triton` | Y 2.06e-3, **AB 3.66e-3**, H 2.07e-3, Out 3.50e-3, Scale 2.22e-3 | 3.2e-3 (`bf16=3.2e-03` in registry.csv) | the check compares bf16 stores (`ab`, `out`) against an fp32 reference, so its error is the bf16 rounding floor of that data. The driver's row count became 8192 (`_ROWS_SATURATE`) after the band was calibrated at 512 rows (the committed A100 row, AB = 2.35e-3, is the 512-row number), and at 8192 rows the seed-0 data's floor is 3.66e-3. The kernel is bit-for-bit what a correctly rounded bf16(fp32 reference) gives: see the probe below | stale band (one shape), kernel exact |
| `mpnn_edge_tail_bwd_layernorm_saveact_triton`, `..._bwd_dx_saveact_triton`, `..._bwd_dx_gather_saveact_triton` | edge_states 5.76e-2, query_projection 2.77e-2, ... | tighter than 1e-2 | MPNN backward numbers wrong by 1e-2 .. 1e-1: out of scope, the same kernels are `failed` on H100 / A5000 / A6000 (A5000 / A6000: also smem `Required: 104448` for the recompute pair) | MPNN (out of scope) |
| `mpnn_edge_tail_bwd_recompute_triton`, `..._bwd_dx_recompute_triton` | edge_states 1.12e-1, query_projection 3.84e-2 | | same | MPNN (out of scope) |
| `mpnn_node_message_bwd_recompute_triton`, `..._bwd_dx_triton` | edge_states 1.12e-1, query_projection 8.89e-3 | | same | MPNN (out of scope) |

The `adaln` checker failure of the 2.4.1 run does not occur on this commit (fixed on `main`; `adaln_*` rows are ok in both precisions).
No failure is A100-specific smem / limits.

### Probe for `cond_transition_fwd_b2b_saveact_triton` (A100, job 83418, scratch `probe_ab.py`, outside the checkout)

AB error (max |a - e| / max |e|, the check's metric), the kernel, its sibling `fwd_saveact._fwd_expand_swiglu` (same inputs, declared 1.2e-2),
and the oracle `bf16(fp32 reference)` (the best any bf16-storing kernel can do), eight seeds per row count, D = 128, ND = 256, default configs:

| rows | kernel AB range | sibling AB range | oracle AB range | kernel == oracle |
|---|---|---|---|---|
| 512 | 2.18e-3 .. 2.37e-3 | identical | identical | 8 of 8 |
| 1024 | 2.18e-3 .. 2.33e-3 | identical | identical | 8 of 8 |
| 2048 | 2.06e-3 .. 2.30e-3 | identical | identical | 8 of 8 |
| 4096 | 1.99e-3 .. 2.23e-3 | identical | identical | 8 of 8 |
| 8192 (the driver's) | 1.97e-3 .. 2.16e-3, **seed 0 (the check's seed): 3.66e-3** | identical | identical | 8 of 8 |

The H dimension (SwiGLU output) behaves the same (1.75e-3 .. 2.95e-3 in every row, equal to the oracle). So only the check's seed-0, 8192-row
draw lands above 3.2e-3 and the kernel loses nothing against the best possible bf16 result. The H100 row (AB 2.17e-3) is also a 512-row number;
the kernel computes the same values on H100, so the same check at the driver's 8192 rows would be expected to read 3.66e-3 there too (not run).

**Proposal (not applied; registry unchanged):** `bf16=1.5e-02` for `cond_transition_fwd_b2b_saveact_triton` (3.66e-3 x `RTOL_MARGIN` 4 = 1.46e-2,
the siblings are 1.2e-2 / 1.4e-2). It states the measured bf16 data floor of this check (no accuracy-losing direction: the kernel equals the oracle); it keeps the
fp32 band 9.0e-3. It also changes `registry.csv` (a `build_rev` / restamp question), so it is left to the maintainer.

## What the gate says now (A100, `records.missing`, CPU job 83420)

BEFORE = the manifest on `main`, AFTER = the staged manifest. "blocked" = kernels without an `ok` row at the call's precision.

| mode / dtype | BEFORE (kernels of main's v1.0.0 manifest) | AFTER |
|---|---|---|
| bf16 inference | portable: triangle_multiplication, triangle_attention, rmsnorm, layernorm, gated_linear, augmented_attention_pair_bias, swiglu_ffn, rms_norm_modulation, layer_norm_linear, bias_only_core, dit_variant; 23 ops blocked | every op portable (33) except `mpnn_relative_position` (no bf16 forward kernel, not a gap) |
| bf16 training | portable: triangle_attention, rmsnorm, layernorm, gated_linear, rms_norm_modulation, layer_norm_linear, bias_only_core | portable for all but `conditioned_transition` (1 of 21: `cond_transition_fwd_b2b_saveact_triton`), `mpnn_edge_tail` (5 of 10), `mpnn_node_message` (2 of 3) |
| fp32 inference | all blocked (no fp32 rows) | portable for every op with fp32 kernels (22); OPM, PWA, pair_embed_*, token_pair_init, MPNN declare no fp32 kernel |
| fp32 training | all blocked | the same 22 portable |

Per op, AFTER (bf16 / fp32 columns: inference, training):

| op | bf16 inf | bf16 train | fp32 inf | fp32 train |
|---|---|---|---|---|
| transition, triangle_multiplication, triangle_attention, adaptive_layernorm, augmented_attention, augmented_attention_pair_bias, rmsnorm, layernorm, gated_linear, gated_residual, swiglu_ffn, rms_norm_modulation, layer_norm_linear, swa_atom_attention, swa_dit, bias_only_core, bias_only_pair_bias, local_dit, dit_variant, dit_variant_atom, sampler_ops, smooth_lddt | portable | portable | portable | portable |
| conditioned_transition | portable | **blocked 1/21** | portable | portable |
| outer_product_mean, msa_pair_weighted_averaging, pair_embed_geom / _ca / _adaln, token_pair_init | portable | portable | n/a (no fp32 kernel) | n/a |
| mpnn_message, mpnn_edge_mlp, mpnn_edge_dropout | portable | portable | n/a | n/a |
| mpnn_edge_tail | portable | blocked 5/10 (failed backward rows) | n/a | n/a |
| mpnn_node_message | portable | blocked 2/3 (failed backward rows) | n/a | n/a |
| mpnn_relative_position | n/a (no forward kernel) | portable | n/a | n/a |

`UNREGISTERED_OPS` is empty. This is only the record side of the gate: Transition on A100 still takes the fused sm_80 CUDA kernels first at d = 128, n = 4.
Not run here: the module-level `dispatch.route_for_card` end to end and the CPU test suite (a `ruff` / `pytest` pass belongs to the maintainer's CI run).

## Staged

* `src/miniworld_engine/autotune/manifests/NVIDIA A100 80GB PCIe (sm80).csv` (the scratch manifest above, whole file).
* `docs/dispatch.md`: the A100 column of the table, rewritten from the helper numbers above; the cells of A5000 / A6000 that said "same" now carry the old A100 text
  they referred to; one note under the table.
* this record.

Not touched: `registry.csv`, bands, any kernel, `docs/gpus/` pages (the maintainer's 성능 확인 rows), the autotune caches (an A100 tuned cache for the
Triton kernels is still not built, so they launch their default configs).

## Open decisions for the maintainer

1. `cond_transition_fwd_b2b_saveact_triton` bf16 band (3.2e-3 -> proposed 1.5e-2), and whether to re-record the H100 row (its AB is the 512-row number).
   Until decided, A100 `conditioned_transition` bf16 training warns and runs PyTorch.
2. MPNN backward kernels (7, out of scope): wrong numbers on A100 (1e-2 .. 1e-1), smem failures on A5000 / A6000, `failed` everywhere including H100.
3. Whether the whole-file manifest replacement (A100 manifest was the v1.0.0 one) is acceptable, as against a by-hand merge of selected rows.
4. A tuned A100 Triton cache (`dev build`) is not part of this record; the default config sets ran.
