# Triton record gate, part 2: the call's precision, and registry rows for the unregistered kernels (2026-10-31)

Base: `main` 82d533af (part 1: `triton_gate_audit_20261031.md`). Branch: `fix/triton-gate-2`.

## (A) fp32 calls asked bf16 rows

`dispatch.portable_leg_allowed` had no dtype argument, so the gates of `LocalDiTBlock`, the DiT variant stacks (token and atom),
`SWA3DRoPEAttention` (its pointwise pieces and `_qk_norm`) and `SWADiTBlock` asked an fp32 call for bf16 rows. It now takes
`dtype=` and every caller passes `dispatch.record_dtype` of the activation the kernels run on (`x` / `single` / `q`: the window core, the
DiT attention and pair-bias kernels and the SWA kernels all run in the activation's dtype). Default `"bf16"`, and the H100 / B200 branch
(`restricted_card`) is taken before the dtype is read: those cards are unchanged.

## (B) registry rows

| kernel(s) | before | now |
|---|---|---|
| `gated_projection/triton/residual.py` `_forward`, `_backward` (`ops.gated_residual`) | no row; `records.UNREGISTERED_OPS`: never recorded, always PyTorch off H100 / B200 | `gated_projection_fwd_residual_triton`, `gated_projection_bwd_residual_triton` (elem, level both, sides atom, bf16 \| fp32); op `gated_residual` |
| `bias_only_attention/triton/pair_bias.py` `_ln_bias_fwd_kernel`, `_ln_bias_bwd_kernel` | no row; `UNREGISTERED_OPS` | `bias_only_attention_layernorm_gemm_triton`, `bias_only_attention_bwd_layernorm_gemm_triton` (gemm, level both, sides pair, bf16 \| fp32, stack diffusion); op `bias_only_pair_bias` |
| `mpnn_message/triton/inference.py` `_message_inference_kernel` (the no-save fusion `auto` takes with autograd off) | no row; `mpnn_message` counted the grad-path kernels for a no-grad call | `mpnn_message_fwd_gemm_gelu_reduce_triton` (gemm, bf16); in `mpnn_message`, inference calls only (`records._INFERENCE_ONLY`) |
| `augmented_attention_bwd_reduce_triton` | declared fp32 (its own buffers: the fp32 dq partials), so a bf16 training call -- which launches it -- never counted it | the declaration stays fp32 (it is what the kernel runs at); the ROUTING now counts it for every call and asks its fp32 row (`records._RUNS_AT`, `records.row_dtype`) |
| `augmented_attention_bwd_atomic_triton` (`kernel_type="memory_efficient"`'s backward) | `developed=no`, so never counted, yet reachable through a public option (`modules.AugmentedAttentionPairBias`, `architecture.layers.AugmentedAttentionPairBias`, `ops.augmented_attention_pair_bias`) | `developed=yes`: it already had a driver, a check, bands and `ok` rows on A100 / A5000 / A6000 / H100 |

How the three formerly fixed launches became autotuned without changing what runs: each kernel takes `@triton.autotune` with a `shape_key`;
its **default** config set is exactly the launch it had (gated residual: 256 elements, 4 warps, 3 stages; pair bias: 64 rows per program,
forward 4 warps, backward 4 or 8 warps by channel count; MPNN: the two A5000 policies), and an early prune selects, outside a build, the one
the old rule picked (`_bwd_prune`, `inference._prune`). A card with no tuned cache therefore launches what it launched before (the
cache reader warns once that no tuned entry exists, as for every untuned op); the **grid** set is what `build` searches. The pair-bias
backward now reads its program count in-kernel (`tl.num_programs`) so the rows per program can be a config axis; the MPNN prune also drops,
always, any config whose row tile cannot hold its residue groups.

Bands (H100, `check_one` worst relative error x 4, `RTOL_MARGIN`):

| kernel | bf16 measured | fp32 measured | band |
|---|---|---|---|
| gated_projection_fwd_residual_triton | 0 | 0 | 0.0 (the eager statement, bit for bit) |
| gated_projection_bwd_residual_triton | 2.71e-03 | 0 | bf16 1.1e-2, fp32 0.0 |
| bias_only_attention_layernorm_gemm_triton | 3.50e-03 | 3.03e-04 | bf16 1.4e-2, fp32 1.2e-3 |
| bias_only_attention_bwd_layernorm_gemm_triton | 4.57e-03 | 4.06e-04 | bf16 1.9e-2, fp32 1.7e-3 |
| mpnn_message_fwd_gemm_gelu_reduce_triton | 1.07e-03 | (bf16 only) | 4.3e-3 |
| augmented_attention_bwd_atomic_triton | 4.70e-03 | 9.63e-04 | unchanged (bf16 1.9e-2, fp32 8.9e-3) |

### Helper kernels left without a row

* `transition/triton/residual.py:_split_k_residual_kernel` -- the ordered sum of the fp32 split-K partials plus the residual. Its numbers
  are now checked under `transition_squeeze_residual_triton`'s row: that check gained a 128 x 768 x 3072 case, which `_split_k` splits 8
  ways on every card from an A100 up (H100: 1.91e-03 bf16, 2.72e-04 fp32, inside the existing band). A row of its own would add a build op
  for a reduction with no free schedule axis worth tuning.
* `kernels/_cast.py:_cast_kernel` -- bit-identical to `tensor.to(dtype)` by construction (its own docstring); nothing to check numerically
  and nothing to tune beyond a memcpy.
* `kernels/_tf32.py:_grid_many_kernel` -- rounds fp32 to the TF32 grid, exact; the same.
* `mpnn_message/triton/main.py:_zero_bias_grad_kernel` -- fills a `hidden`-element buffer with zeros in one program.

All four run inside an op whose gate counts that op's real kernels.

### Routing changes on the packaged manifests (bf16)

| card | change | why |
|---|---|---|
| A100, A5000 | `augmented_attention_pair_bias` and `dit_variant` training: portable -> warns | the split backward's reduce runs at fp32 and these manifests hold no fp32 row for it (A6000 does: unchanged) |
| A5000 | MPNN `auto` no-grad message reduction: portable -> warns | the no-save fusion has a row now and no card but H100 has recorded it; grad calls stay portable |
| all | `gated_residual`, `bias_only_pair_bias`: still warn | rows exist now; no Ampere manifest has them until `run_all` runs there |

H100 / B200: unchanged routing (the rule passes them through). On H100 the five new kernels launch their default configs -- the launches they had -- with one "no tuned autotune cache" warning per op until a cache is built for them.

## run_all on H100

`run_all` records through `devices.record`, which MERGES but also rewrites the `#provenance` row to the running commit. So it was run
(commit ddb66667, 2026-10-31, `--only gated_projection,bias_only_attention,mpnn_message,transition,augmented_attention`, bf16 and fp32)
into a scratch copy of the manifests (`devices._DEVICES` pointed at a scratch directory), and only the nine rows of the new kernels were
copied into `autotune/manifests/NVIDIA H100 80GB HBM3 (sm90).csv` by hand: bf16 for all five, fp32 for the four that declare it. Each
carries `[run_all at ddb66667 2026-10-31 merged by hand; the provenance row is the 2.4.1 run]` in its detail; the `#provenance` row is
untouched (2.4.1, a6ac9fda), and so is every other row.

All 42 bf16 and 21 fp32 kernels of those families passed on H100, among them (not merged: their committed 2.4.1 rows stand)
`transition_squeeze_residual_triton` with its new split-K case (bf16 out 1.07e-03, split 1.91e-03; fp32 2.68e-04 / 2.72e-04),
`augmented_attention_bwd_atomic_triton` (bf16 4.70e-03, fp32 9.63e-04) and `augmented_attention_bwd_reduce_triton` at fp32 (0).

Still to record: the Ampere cards (A100, A5000, A6000) for the new rows and the fp32 row of `augmented_attention_bwd_reduce_triton`
(A100, A5000), on those cards, with the same partial procedure.
