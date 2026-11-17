# Triangle multiplicative update on H100 (sm90)

Scope: `TriangleMultiplication` (backends `pytorch`, `cuda`, `triton`, `cueq`, `anthropic`), `BidirectionalTriangleMultiplication`
(`pytorch`, `cuda`, `triton`) and `TriangleMultiplicationRF3` (`pytorch`, `cuda`, `triton`, `anthropic`: the contraction divided by the unmasked length is
folded into the left projection `W_a / n` for `cuda` / `triton` (one launch per sample when the masks differ), into `W_b / n` per
sample for `anthropic`) of `modules/triangle.py`. The requirement rows (`requirements/requirements_module.csv`) ask for L 128-768
(multiples of 128), dtype `master` (bf16 activations, fp32 master parameters), modes `eval` and `train`, `p_drop=0.25`:
single direction (outgoing and incoming) at d_pair 64, 64 with d_hidden 128, 128, 256, 384, 512; bidirectional at d_pair 64, 128,
256, 384, 512; RF3 at 64 and 128. Backend selection, `_supports_<backend>` and the measurer: [../modules.md](../modules.md),
[../kernels.md](../kernels.md), [../measure.md](../measure.md).

```
x = LN_in(pair)
a = sigmoid(x W_ag) o (x W_a),  b = sigmoid(x W_bg) o (x W_b)        (masked by m_i & m_j)
t = sum_k a_ik b_jk  (outgoing) | sum_k a_ki b_kj  (incoming)        (bidirectional: first d_hidden channels outgoing, second incoming)
y = pair + drop_row(sigmoid(x W_g) o (LN_out(t) W_o))
```

## Backends

| backend | what runs | range (`_supports_*`) |
|---|---|---|
| `pytorch` | the module's statements | reference |
| `cuda` | `integrations/trimul_h100.py` over `kernels/trimul_inproj/cuda/`: `update_inference` (`serves_inference`) without grad or live dropout, else `update` (bidirectional: `serves`; single direction: `serves_single`) | bf16 pair on capability (9, 0) (A100 (8, 0): `integrations/trimul_sm80.py`, `trimul_inproj/cuda/sm80*.py`), square, bool `[B, L]` mask, plus the integration's contract (below) |
| `triton` | `trimul_inproj/triton/unidirectional.trimul_triton` / `bidirectional.bidirectional_trimul_triton`: the whole update incl. residual and row dropout, one launch sequence per sample | CUDA, bf16 pair, square, bool mask; d_hidden != d_pair only with both multiples of 16, d_pair <= 256, d_hidden <= 512 (`contract.hidden_refusal`) |
| `cueq` | `cuequivariance_torch.triangle_multiplicative_update` (single direction), residual and dropout by the module | fp32 masters cast to the pair's dtype in autograd, one LN epsilon, widths % 32, bf16 / fp32, the library's kernel band |
| `anthropic` | `opt_core.kernels.trimul` (tier `fast`), single direction, residual in its epilogue | inference only, no live dropout, the face casts fp32 masters to the pair's dtype, rows listed in `_ANTHROPIC_EXCLUDED_ROWS` excluded |

## Kernels

Names from `requirements/requirements_kernel.csv` (family `trimul_inproj`, 97 names, 6 of them the A100 `*_sm80` kernels); sources under
`kernels/trimul_inproj/cuda/h100_sources/` unless noted.

| kernels | role | what they do |
|---|---|---|
| `trimul_fwd_front_sm90`, `trimul_fwd_output_sm90`, `trimul_fwd_output_wide_sm90` | forward / forward_infer | packaged K1 / K3 unit (`inference/tmn_kernels.cuh`, `tmn_k3_wide.cuh`; `h100_inference.py`): K1 = LN_in + four gated projections into channel-major planes, K3 = LN_out + output projection + gate + residual |
| `trimul_output_saveact_sm90`, `trimul_b1_bwd_sm90`, `trimul_b1_reduce_bwd_sm90`, `trimul_b7_bwd_l384_sm90`, `trimul_b7_bwd_sm90` | forward_train / backward | D128 bidirectional training (`h100_training.py`): K3 with saves (`output/save_k3.cu`), fused B1 (`b1/b1_fused.cu`) and its reduce, B7 (`b7_384/joint.cu`, `b7_768/joint.cu`) |
| `trimul_single_front_sm90`, `trimul_single_b1_bwd_sm90`, `trimul_single_reduce_bwd_sm90`, `trimul_single_b7_bwd_sm90` | forward_train / backward | D128 single direction (`h100_single.py`): K1 (`h100_native`), on-chip B1 (`single_output/kernel.cu`), streamed producer-consumer B7 (`single_b7/joint.cu`) |
| `trimul_d64_output_saveact_sm90`, `trimul_d64_b1_bwd_sm90`, `trimul_d64_b2_bwd_sm90`, `trimul_d64_fin_bwd_sm90`, `trimul_d64_uni_*` (3) | forward_train / backward | D64 training (`h100_d64_b200gate.py`, `h100_uni_d64_b200gate.py`; `d64_train/`): K3 with saves, B1 / B2 / finish, bidirectional and single direction |
| `trimul_b200style_*` (16) | forward / backward | the Hopper port of the B200 wide algorithm (`h100_b200style*.py`, `b200style/`, `b200_sources/wide_bwd.cu`): fronts (RS / direct / D512 shared-resident `k1ss*`, saving and backward variants), K3 outputs (standard, static, reorder, `tmasave` saving variants per width), LN-affine fold, gate / LN_out / column-sum / LN_in backward |
| `trimul_wide_front_sm90`, `trimul_wide_front_shared_sm90`, `trimul_wide_output_stream_sm90` | forward_infer | bidirectional D256 / 384 / 512 inference (`h100_wide_inference.py`, `wide_forward/`) |
| `trimul_wide_*` (other 30), `trimul_uni_wide_front_d512_saveact_sm90`, `trimul_uni_wide_ln_in_d512_sm90` | forward_train / backward | wide training, bidirectional (`h100_wide_training.py`) and single direction (`h100_uni_wide_training.py`), `wide_train/*.cu`: fronts with saved preactivations, output LN, output epilogue, gate / gate-affine backward, output-LN backward, `contract_gp`, D512 compact / delta / gather / scatter backward, input-LN backward and reductions |
| `trimul_aux_*` (14) | forward / backward | `mw_aux_*` glue (`_h100_pack.py`, `h100_b200style_aux.py`): weight packs, pair / training masks, unit dropout scale, LN-affine fold, norms, copies, gradient layout, parameter-gradient finishes |
| `trimul_gemm_gate_mmajor`, `trimul_outproj_layernorm_gemm_gate`, `trimul_output_f567_train`, `trimul_bwd_gate_packed`, `trimul_input_ln_residual_bwd`, `trimul_input_dual_bwd`, `gated_projection_bwd_gate_dropres` | forward / backward | the Triton path (`trimul_inproj/triton/`), see [Triton path](#triton-path) |

`kernels/tm1/` (`trimul_gemm_gate_triton`, `trimul_bwd_gate_recompute_triton`) and `kernels/tm2/` (`trimul_outproj_gemm_gate_triton`,
`trimul_outproj_bwd_gate_recompute_triton`) are Triton gated-GEMM kernels that no module imports and no requirement row names.

## CUDA path selection (`integrations/trimul_h100.py`)

| width | inference | training |
|---|---|---|
| D64 / D128, d_hidden = D | packaged K1 -> cuBLAS contraction -> K3, any L >= 16; single direction from L >= 1000 (D64) / 768 (D128) the port's row-padded form (`ROWPAD_FROM`) | bidirectional: `h100_d64_b200gate` (D64), `h100_training` (D128); single: `h100_uni_d64_b200gate` (D64), `h100_single` (D128); L % 128 == 0 up to 10240 |
| D64 / D128, d_hidden = 2D, one direction | the bidirectional K1 / K3 unit with a one-direction contraction (`_hidden_twice`) | the bidirectional D64 / D128 kernels with `direction` 1 / 2 |
| D256 / D384 / D512 | the B200-style port at L % 128 == 0 (and any L for single direction D256 / D384, B = 1); B = 1 bidirectional (D384, L768) / (D512, L384): `h100_wide_inference`; B = 1 single D256 below L256: packaged K1 / K3 (`h100_uni_wide_inference` and the D384 L512 native-residual K1 / K3 are checked only after the port, which serves those shapes first) | the port, except B = 1 at L384 / 768: bidirectional `h100_wide_training`, single `h100_uni_wide_training` (`_legacy_native_faster`); B > 1 at those cells: per-sample preserved kernels up to `PER_SAMPLE_LEGACY_TRAINING[(D, L, bidir)]`, else the port |

Common contract: LN epsilon 1e-5 for both LayerNorms, fp32 LN affine (the `master` rows), contiguous pair, `B L^2 < 2^31`. D128
training needs a full 132-SM H100 (its B1 / B7 grids synchronise through global counters; not served on MIG slices).

### Packaged K1 / K3 (inference, D64 / D128, single D256 at short L)

`h100_inference.py`, `_h100_infer_kernel.py`, `inference/tmn_kernels.cuh`: one kernel family parameterised over C_Z x C_H in
{64, 128, 256, 384}^2, derived from Anthropic's `trimul_tx` 1.2. Persistent CTAs of three warpgroups: WG0 producer (TMA bulk loads
into 128B-swizzled shared memory, mbarrier rings), WG1 / WG2 consumers (ldmatrix -> LN in registers -> wgmma m64n64k16 with A from
registers and the weight chunk from shared memory -> fused epilogue, one weight block ahead).

- K1: `ab[ch, i, j] = bf16(sigmoid(LN_in(z) Wg[ch]) (LN_in(z) Wp[ch]) mask)`, written channel-major into planes `[2 C_H][Np][Np]`
  (zero pad to a multiple of 16) so the contraction is one strided-batched GEMM. The pair mask is formed in-kernel from the
  token mask (`m[i] & m[j]`); no `[L, L]` mask tensor.
- Contraction: cuBLAS bmm per channel with no workspace (`_h100_bmm.py`: avoids the stream-K semaphore memset node).
- K3: `out = z + cvt(sigmoid(LN_in(z) Wog^T) o (LN_out(X) Wo^T))`, X = the contraction planes.
- Launch plans (K1Params / K3Params blocks with tensor maps) are cached per weight set (`PLAN_SETS = 256`); kernels are launched
  through the CUDA driver (`_h100_driver.py`, `_h100_launch.py`).

### D128 bidirectional training (`h100_training.py`)

Forward: K1 front, contraction, K3 with saves. Saved: x_n, left / right planes, t, LN_out statistics, packed weights. Backward:
fused B1 (built per length, `b1/configs.json`; a `B1_BATCHED` build for B > 1, `MASTER_FP32` for fp32 master gradients), four
cuBLAS contractions for the plane gradients, and B7 (`b7_384` source at L384, `b7_768` elsewhere), which recomputes the input LN
and the projections and produces the input gradient and weight partials. The bidirectional D128 path takes the token mask; the
other training paths read a bf16 pair mask built once per call.

### D128 single direction (`h100_single.py`)

Saved: x_n, left / right, t, packed weights. The projections / gates and the output LN are recomputed on chip in the backward:
K3 + B1 derivatives (`single_output`), then a producer-consumer B7 port (`single_b7`). Per-length tiles in `single_output/selection.json`.

B1 (`single_output/kernel.cu`, kernel `single_b1`): a CTA owns 64-token tiles (132 CTAs, 1 per SM) and runs six phases per tile one after the other (input LN recompute,
gate / projection `wgmma`, gate epilogue, dW and dn `wgmma`, output-LN backward, TMA stores). Removal ablation (L768, B1; all other phases kept; times
are the whole kernel, baseline 0.862 ms): no loads after the first tile 0.884x, no LN recompute 0.860x, no gate epilogue 0.730x, no `wgmma` 0.769x, no
LN backward 0.810x, no stores 0.934x: the phases add up (sum of the removals 104 %), so the kernel is serialisation, not one resource. Changes that keep
every per-element operation and every accumulation order (bit-identical on y, dx and all parameter gradients, `tests/gpu/test_trimul_single_train_b1.py`):
the next tile's xn / tri / dy are TMA-prefetched into a second shared-memory set (smem 181248 -> 230400 B; 0.93x), the gate epilogue and the dn stores
move two adjacent columns per 32-bit shared access and per dropout-scale load (`B1_VEC`; 0.79x with the prefetch), and each warp takes two LN rows
(r, r + 8) per pass with the shuffle reductions interleaved (`B1_LNI` = 2, accumulation order of dgamma / dbeta unchanged; 0.71x). Rejected: dropout scales
held in registers (the kernel already spills at 255 registers: 1.16x slower), LN gamma / beta in registers (no change), four interleaved rows (0.725x vs 0.714x
for two). `single_b7` builds, loads and checks the cooperative-launch occupancy once per process (`_build`): the repeated `T.compile` source / header hashing
and occupancy query cost ~0.7 ms of host time per backward in eager training.
Also bit-identical (2.5.10): `B1_PK` (default on) rounds the gate / projection pairs with one `cvt.rn.bf16x2` and multiplies dy by the dropout scale as
bf16x2 (the product of two bf16 is exact in fp32, so one rounding as before; `single_b1` 0.95x at L768); `single_b7` frees the plane staging buffers once
the bulk stores have read them (`B7_EARLY_FREE`, as the bidirectional B7) and stages the dx tile in the dg buffer, releasing it before the dx store.
Module forward + backward vs 2.5.9 (eager, interleaved in one process, master weights, B1): L256 / 384 1.00x (host-bound), L512 0.98 / 0.99x, L768 0.91x
(outgoing / incoming).

### D64 (`h100_d64_b200gate.py`, `h100_uni_d64_b200gate.py`)

B200-style gate-gradient placement: B1 stores bf16 dGout, B7 accumulates `dGout Wg` with the input-projection gradient in fp32.
K3 with saves (`save_k3_d64.cu`), B1 / B2 / finish (`d64_bwd_wg.cu`, `uni_bwd_wg.cu`). A single-direction module with hidden 128
runs the bidirectional kernels with the one-direction contraction (`_triangle`, `_contraction_grads`, `direction` 1 / 2).

### B200-style port (D256 / D384 / D512, `h100_b200style.py`)

K1 writes the gated planes and LN_in statistics; the folded K3 writes y, projection, gate and LN_out statistics. Neither the
normalised rows nor the front preactivations survive the forward (D512 additionally keeps the normalised input). The backward uses
the B200 folding identities and recomputes the front preactivations in a wgmma K1 epilogue; the auxiliary backward kernels
(`b200_sources/wide_bwd.cu`: gate, LN_out, column sums, LN_in) are plain CUDA shared with B200. Contraction: B200's for both
directions, `_h100_bmm.contract` (measured cuBLASLt configuration) for one direction. Saved: planes `[2h, rows, n]`, t, LN
statistics, LN-affine folds, and per-token tensors (plus one more at D512).

Row-padded K3 (inference, one direction, any L; D64 / D128 from `ROWPAD_FROM`, D256 / D384 where L % 128 != 0): built with the k3.cu pipeline macros
`K3_XSPF` (weight / operand prefetch across tiles), `K3_XEARLY` (residual read early), `K3_DESC` (TMA descriptor prefetch) and `K3_EPI2` (two-stage
epilogue) at D64 / D128 (`h100_b200style.K3_EXTRA`; D256 / D384 build the plain K3, slower with them). Each macro keeps the instruction order of every
accumulation: the output is bit-identical to the plain K3 (`tests/gpu/test_trimul_h100_k3_pipe.py`, real Protenix / Boltz-2 / AF3 dumps). Measured on the
module (warm, CUDA events, 2026-11-15): D64 L1024-2048 +2.4..3.3 %, D128 L768-1024 +2.5..2.9 %, D128 L1536-2048 within noise (K3 alone 1.04-1.13x). `backend_versions["cuda:h100_sm90"]` +1 on the three TriMul classes.

### Wide inference (`h100_wide_inference.py`, `h100_uni_wide_inference.py`)

Bidirectional: K1 (the native K1 extended to H = 2D) writes the packed planes and per-token LN_in statistics (at D512 a separate
input LN writes them), two cuBLAS contractions, and a streamed-operand K3 (`output_stream.cu`) that reads t channel-major with both
LayerNorms folded into its weights. Single D512: LN_in, K1, one contraction, the gate GEMM on cuBLAS (`g = bf16(xn Wg^T)`), and K3
(`uni_wide/output.cu`) with LN_out folded into the weight (`Wp' = bf16(gamma o Wp)`, `P = rs (t Wp'^T - mu u) + v`) and the
epilogue `x + bf16(P) sigmoid(g)`. Schedules: `wide_forward/selection.json`, `uni_wide/selection.json`.

### Wide training (`h100_wide_training.py`, `h100_uni_wide_training.py`)

Frozen sources under `wide_train/`; dense GEMMs on cuBLASLt with explicit algorithms (`lt_selection.json`). Single direction:
weight pack -> front (`front_pre_d384.cu`, or at D512 `ln_in_d512.cu` + `front_pre_ss.cu` with wgmma A / B from shared memory;
LN_in + gated projections, saves x_n and channel-major preactivations) -> contraction -> `forward_norm_d384.cu` (LN_out) ->
projection and gate GEMMs -> `output_dense.cu` (gate, dropout, residual). Backward: `gate.cu` -> dn GEMM -> `output_ln_*.cu` ->
`contract_gp_*.cu` (`GP_DIR`; dl / dr never materialised, the epilogue writes the preactivation gradients) -> weight GEMMs -> dX
GEMM -> `input_ln_*.cu` (input LN + residual). Each call owns its buffers (no cached activations or weight packs).
Single D512 (2.5.10): the output-LN backward runs `output_ln_uni_pf.cu` (`_LN_PF`): the same grid and per-CTA row order as `output_ln_d512_l384.cu`, with
the next row tile's TMA loads in flight in a second shared-memory set; bit-identical y, dx and all gradients
(`tests/gpu/test_trimul_uni_wide_ln_prefetch.py`). Module forward + backward vs 2.5.9 (eager, interleaved, B1): L384 0.99 / 1.00x, L768 0.98x.

#### CUDA-graph device time of the bidirectional route (2026-11-16, branch `bidir-k3gap`)

One floor method: per stage max(byte floor at 3.05 TB/s over every HBM read and write of the stage, tensor floor at 748 TF). Per pair token (M = L^2), hidden channels h (bidirectional D128: h = 256, hidden-twice single direction: 256, single direction hidden 128: 128): K1 reads 2 D B and writes 4 h B (planes), flops 8 M D h; the contractions read 4 h B and write 2 h B per token (two bmm of h / 2 channels) at 2 L M h flops; K3 reads 2 h B (tri) + 2 D B (z) and writes 2 D B, flops 2 M h D + 2 M D D. Single-shot graph replay, D128, h = 256 (same bytes in all three rows):

| L | route | K1 / floor | contractions | K3 | module / floor-sum |
|---|---|---|---|---|---|
| 1024 | bidirectional packaged (default) | 0.556 / 0.440 = 79 % | 0.779 / 0.735 = 94 % (two 256x128 launches) | 0.409 / 0.352 = 86 % | 1.756 -> 87 % |
| 1024 | single direction, hidden 256, packaged K1 / K3 | 0.556 = 79 % | 0.743 = 99 % (one 128x128 launch) | 0.411 = 86 % | 1.727 -> 88 % |
| 1024 | port K1 + port K3 at h = 256 | 0.588 = 75 % | (single) | 0.375 = 94 % | |
| 2048 | bidirectional packaged | 2.350 = 75 % | 5.755 = 102 % | 1.763 = 80 % | 87 % |
| 2048 | port K1 + port K3 at h = 256 | 2.448 = 72 % | | 1.530 = 92 % | |

The gap to the single-direction 88 % K3 is real and is the K3 itself: the packaged `tmn_k3_z128_h256` (LN_out and LN_in as register passes, 32-channel output blocks) runs 80-86 % of its floor at the bidirectional byte volume, the B200-style port K3 at the same volume (LayerNorms folded into the weights, LN_in statistics from K1) 92-94 %. K1 is the same ~75-79 % in both routes (packaged and port). Kernel-to-kernel gaps are zero or negative (PDL overlap -0.1 .. -0.5 us per boundary) at every length measured: no launch-tail gap exists to remove. Two streams for the two contractions inside the graph: 0.98-1.005 of serial (control +-0.5 %), 1.5-2 % only at D64 / D128 L <= 512 and lost again by the K3 PDL overlap: not shipped.

Shipped, bitwise equal to ec11768fc (212 module cases D64..D512, L17..2048, masks, B = 1 / 2, real activation dumps): the bidirectional contractions go through `_h100_bmm.contract` and the measured cuBLASLt table `_h100_bmm_lt_bidir.LT_BIDIR` (algorithm 66, no split-K, output equal to the default pick on every entry; 178 entries, lengths multiples of 16 from 144 to 2048 where the in-graph module replay showed <= 0.995 of the default in two processes). Module graph replay, D128 / D64, mean over L = 128..2048 step 32: 0.983 / 0.987 of the default (control 1.002), best lengths 0.88-0.95 (D128 L832 / 1408 / 1600). The table is passed by the bidirectional call only (`contract(..., table=LT_BIDIR)`): the single-direction D64 / D128 contractions (same plane counts) keep `LT_CONFIGS` and the default pick, unchanged (not measured with these entries; a follow-up).

Port K3, the default since 2.5.12 (`h100_inference.PORT_K3`, `MINIWORLD_TRIMUL_BIDIR_PORT_K3=0` switches it off; D128 inference, L >= 160): the packaged K1 in its save-statistics l2 form (planes unchanged), the contractions, the port K3 (`h100_b200style.rowpad_k3`). Not bit-identical (bf16 ulps). Error against an fp64 reference (same bf16 weights): rms 3.82e-3 against 4.12e-3 of the packaged K3 (-7.4 %) on random inputs at every length / mask / B = 2 case measured (36 cases), max error lower in 32 of 36 and higher in 4 (L160 tail, L300 tail, L384 none and ones: +3 .. +14 %), real activation inputs equal (output rounding bound: 6.83e-2 / 6.84e-2, 7.14e-2 / 7.14e-2, 1.289e-1 / 1.289e-1). Module graph replay on top of the table: K3 stage -8 % (L1024) .. -15 % (L2048), module 0.963 of the default (mean over L = 128..2048 step 32; 0.977 / 0.985 / 0.976 / 0.977 at L256 / 512 / 1024 / 2048 against the table-less route in a separate run). D64 measured no gain (its K3 is already the port's) and is not served; L < 160 loses the fold launch (2 us).

Engine accuracy standard (measure/accuracy.py, registry row 95 = D128, every length, eval + train, port K3 on vs off; 2026-11-16): no light changes (eval L128-768 GREEN with it on and off; training is not on the route: its records are the same, 10 GREEN and the 2 YELLOW gradients of before). Eval max error / tol 8e-3 off -> on: L256 4.65e-3 -> 4.28e-3, L384 4.83e-3 -> 4.94e-3, L512 4.65e-3 -> 4.94e-3, L640 4.94e-3 -> 4.58e-3, L768 5.03e-3 -> 4.76e-3 (L128: below the minimum length, same route). Rows of the other widths do not reach it (D128 only). `BidirectionalTriangleMultiplication.backend_versions["cuda:h100_sm90"]` 6.

### Bidirectional inference: route per width, what applies, %SOL (2026-11-16, branch `fp8/bidir-tm`; measured, default bf16 route)

Which kernels run (one sample, `BidirectionalTriangleMultiplication`, d_hidden = d_pair, eval, no dropout; kernel names from the profiler):

| d_pair | inference route | which of the single-direction improvements apply |
|---|---|---|
| 64 | packaged K1 `tmn_k1_z64_h128` -> two cuBLAS bmm (`nvjet_tst`, TNT / NTT on transposed views) -> packaged K3 `tmn_k3_z64_h128`, any L >= 16 | `K3_EXTRA` (2.5.9 K3 pipe flags): no, they exist only in `b200style/k3.cu`, which the one-direction row-padded form uses; the packaged K3 is a different kernel. Host transpose / stats / min-L / FP8 prebuild entries: no (FP8-only items; the bf16 bidirectional route has no host transpose: the second contraction is a transposed view) |
| 128 | packaged K1 `tmn_k1_z128_h256` -> two bmm -> packaged K3 `tmn_k3_z128_h256`, any L >= 16 | same; FP8 planes (new, opt-in): L >= 256 |
| 256 | B200-style port, direction 0: `mw_b200style_k1_direct` -> two bmm -> `mw_b200style_fold` -> `mw_b200style_k3_static` (token-flat), L % 128 == 0 (B = 1 at other L: the `cuda` backend refuses; B > 1: padded to 128) | `K3_EXTRA`: not applied by design (D256 / D384 build the plain K3, slower with the flags). FP8 planes (new, opt-in): L >= 1024 |
| 384 | the port at every L % 128 == 0 except (D384, L768): `h100_wide_inference` | none (no FP8 route) |
| 512 | the port except (D512, L384): `h100_wide_inference` (D512: separate input LN, `k1ss`, `k3reorder`) | none |

Training: D64 `h100_d64_b200gate`, D128 `h100_training`, D256 / D384 / D512 the port (B = 1 at L384 / 768: `h100_wide_training`); none of the FP8 / K3-pipe items applies (inference only).

Candidate "bidirectional D64 / D128 through the port's row-padded form (direction 0: direct K1 at h = 2 D, `K3_EXTRA` K3 over 2 D channels)": measured against the packaged route and REJECTED: 25-40 % of the output elements differ (max 3-6e-2, a few bf16 ulp: the folded LN_out of k3.cu is a different rounding order than the packaged K3's), so it is not bit-identical, and the module is slower except at D128 L1024 (graph replay, candidate / default): D64 L256 / 512 / 1024 / 2048 1.10 / 1.08 / 1.06 / 1.03, D128 1.04 / 1.01 / 0.97 / 1.00. The pipe flags are bit-identical to the plain K3 inside that candidate (0 differing elements at D64 / D128, 4 lengths) and give it 0.3-3 %. The packaged unit's own K3 experiment macro `TMN_K3_EARLYP=1` (identical bytes, 0 differing elements) measured (graph replay, EARLYP / default) D64 L256..2048 1.000 / 1.001 / 0.997 / 1.029, D128 0.996 / 0.998 / 1.000 / 1.012: no gain, not enabled.

%SOL of the default bf16 route (CUDA-graph replay of the module, profiler device time per kernel, same-callable control within 0.5 %; SOL = max(bytes / 3.05 TB/s, FLOPs / 748 TF cuBLAS bf16): K1 reads z, writes 8 D n^2 B of planes and 8 n^2 B of stats, 16 n^2 D^2 FLOPs; the two contractions 4 D n^3 FLOPs; K3 reads the tri planes 4 D n^2 B and z, stats, writes y, 6 n^2 D^2 FLOPs):

| D | L | K1 ms (%SOL) | contraction ms (%SOL) | K3 ms (%SOL) | graph ms | module %SOL |
|---|---|---|---|---|---|---|
| 64 | 512 / 1024 / 2048 | 0.069 (81) / 0.269 (83) / 1.224 (73) | 0.077 (85) / 0.409 (90) / 3.086 (95) | 0.056 (79) / 0.205 (87) / 0.809 (88) | 0.198 / 0.889 / 5.031 | 84 / 86 / 90 |
| 128 | 512 / 1024 / 2048 | 0.162 (69) / 0.624 (71) / 2.463 (72) | 0.150 (88) / 0.833 (88) / 6.043 (97) | 0.120 (74) / 0.460 (77) / 1.832 (77) | 0.413 / 2.011 / 11.02 | 80 / 76 / 82 |
| 256 | 512 / 1024 / 2048 | 0.411 (89) / 1.775 (83) / 6.779 (87) | 0.293 (90) / 1.779 (83) / 12.64 (93) | 0.252 (70) / 1.090 (65) / 4.202 (67) | 0.963 / 4.674 / 24.11 | 84 / 78 / 85 |
| 384 | 512 / 1024 / 2048 | 1.056 (78) / 4.227 (78) / 13.21 (100) | 0.429 (92) / 2.393 (92) / 18.22 (97) | 0.512 (61) / 2.027 (61) / 8.284 (60) | 2.041 / 8.991 / 44.75 | 75 / 75 / 80 |
| 512 | 512 / 1024 / 2048 | 1.989 (74) / 8.139 (72) / 33.80 (70) | 0.565 (94) / 3.271 (90) / 24.44 (96) | 0.897 (61) / 3.596 (61) / 14.52 (61) | 3.572 / 15.34 / 73.07 | 71 / 72 / 76 |

(L256 and L768, 1536 in `~/miniworld-engine-scratch/fp8/bidir/res/brk_base_*.jsonl`. The D384 L2048 K1 row at 100 % is 13.2 ms against a 13.2 ms tensor floor at 748 TF: the K1 of that cell runs at the cuBLAS-measured peak; the L1024 K1 78 %.) Removal ablations (graph replay with one stage launch removed, outputs wrong, `res/abld_*.jsonl`, marginal = base - ablated): D128 L1024 K1 0.55 / contraction 0.76 / K3 0.46 ms (kernel times 0.62 / 0.83 / 0.46); D256 L1024 1.75 / 1.90 / 1.31 (1.78 / 1.78 / 1.09); D64 L1024 0.31 / 0.39 / 0.23 (0.27 / 0.41 / 0.21). The stages are serial (marginals add up to the module time within 10 %); at L2048 the removal of K1 or of the contraction lets the other stage run on empty / stale planes and the power-bound GEMM runs faster (D128 L2048 marginal of the contraction 7.4 ms vs 6.0 ms kernel time): the L1024 marginals are the ones to read. These ablations establish the stage times, not what bounds a stage: no removal ablation of K1's or K3's internals (loads / stores / LN / gate) was run for the bidirectional kernels, so no bound is claimed for the sub-90 % rows.

### Batching, lengths, chunking

Every width takes B samples per launch (`MAX_BATCH_BY_WIDTH` = 48 at D64-D512): tokens b-major, K1 stacks the samples along its
slow token axis (`rows = B L`, each row reads its own sample's mask), the contraction batches over channels x B, everything after
it is token-flat and reads the row-dropout scale `[B L, D]` per sample. A call with more samples, or a working set above
`CHUNK_MEMORY_FRACTION = 0.85` of total device memory (bytes per pair token per channel: `_WORKING_SET_PER_CHANNEL`,
`_HELD_PER_CHANNEL`), runs in balanced chunks (`_plan`, a pure function of shapes, mode and total memory); training chunks are one
autograd node (`_Chunked`), plain or checkpointed, writing y / dx in place. Training: L % 128 == 0, 128 <= L <= 10240. Inference:
any L >= 16; B > 1 at L % 16 != 0 is padded on the host to 16 (D256+: 128) with masked pad tokens.

## Triton path

`trimul_inproj/triton/`, B = 1 per launch sequence (the module loops over samples, `_per_sample`):

- `input_ln_residual` (LN_in and the identity residual as one autograd function) -> `bidir_front_triton`
  (`trimul_gemm_gate_mmajor`: masked gated projections written channel-major; in training also the preactivations) ->
  cuBLAS bmm per channel (`contract.packed_forward` for both directions).
- Inference: `trimul_back_triton` (`trimul_outproj_layernorm_gemm_gate`: LN_out + projection GEMM + gate GEMM + product +
  residual in one kernel).
- Training (bf16): LN_out materialised (`layernorm_linear` te-style) -> `output_f567_train` (projection / gate GEMMs, dropout,
  residual; saves y, proj, gate). Backward: `gate_elem_bwd_ew` (`gated_projection_bwd_gate_dropres`), LN_out backward, the plane
  gradients on cuBLAS, `front_bwd_dW` (`trimul_bwd_gate_packed`: the gated-GEMM elementwise backward fused into the weight-gradient
  GEMM inputs), `input_dual_bwd` (the two input dgrad GEMMs fused), `trimul_input_ln_residual_bwd` (LN backward + residual gradient).
- The training path always carries a dropout scale (ones when p_drop is 0).

## Numerics

- Inference K1 / K3: fp32 LN statistics (two-pass, centred); LN output bf16; `gate * proj * mask` on fp32 accumulators -> bf16
  planes; contraction bf16 operands, fp32 accumulation, bf16 t; K3 LN_out in fp32 statistics, projection / gate on fp32
  accumulators.
- Parameters: projections fp32 (master) or bf16, cast to bf16 for the kernels; LN affine must be fp32. Training builds with
  `MASTER_FP32=1` return fp32 gradients for fp32 masters.
- No fp32 TriMul kernel exists (policy; `_trimul_cuda_route` refuses non-bf16 pairs).
- Determinism: the port's backward sums LN / gate partials with `atomicAdd` (`wide_bwd.cu`), so its gradients agree run to run only
  up to fp32 summation order.

## Launch configs

`kernels/trimul_inproj/configs/h100_sm90.json`:

- `trimul_fwd_sm90_cuda` (the packaged K1 / K3 tiles, read by `h100_inference._select_config` through `choose`): keys
  `bfloat16|B=1,CH,CZ,DIR,L,MASK=1` for (CZ, CH) = (64, 64), (64, 128), (128, 128), (128, 256), (256, 256), (384, 384), DIR 0 / 1 / 2,
  L 128-768; a missing key takes the nearest entry over (B, L).
- Triton: `trimul_outproj_layernorm_gemm_gate_triton`, `trimul_output_f567_train_triton`, `trimul_bwd_gate_packed_triton`,
  `trimul_input_ln_residual_bwd_triton`, `gated_projection_gate_dropres_triton`, `gated_projection_bwd_gate_dropres_triton`,
  `trimul_gemm_gate_mmajor_triton`, `trimul_input_dual_bwd_triton` (exact key lookup). `trimul_bwd_gate_packed_recompute_triton`
  has no entry and runs its default config.
- `kernels/tm1/configs`, `kernels/tm2/configs` hold tables for the unused tm1 / tm2 kernels.
- The other CUDA paths use family JSONs under `h100_sources/` (`b1/configs.json`, `single_output/selection.json`,
  `wide_forward/selection.json`, `uni_wide/selection.json`, `wide_train/lt_selection.json`) and hard-wired per-width tiles.

## Limits / unsupported calls

`cuda`: bf16 pair only; capability (9, 0) (capability (8, 0) goes to `_trimul_cuda_route_sm80`); square pair and bool `[B, L]` mask; LN eps 1e-5 and fp32 LN affine; d_hidden = d_pair,
or 2 d_pair in one direction at D64 / D128 (other hidden widths, and every width outside 64 / 128 / 256 / 384 / 512, are refused);
training lengths L % 128 == 0 up to 10240; D128 training only on a 132-SM device. `triton`: bf16 pair, `contract.hidden_refusal`.
`cueq` / `anthropic`: single direction only. `TriangleMultiplicationRF3` has no `cueq` backend
(the library op has no 1 / L); its `cuda` takes one sample's contract (`pair[:1]`) when the samples' masks may differ. Any refusal is a reason string (`BackendUnsupported`), never a fallback.

## Historical numbers

Historical (from the 2026-10-15 note, H100 80GB, compiled + manual CUDA graph, B = 1, fp32 masters, mask probability 0.1, L
128-1024, D 64-512; not re-measured under the current measurer): the CUDA path was the fastest of PyTorch, cuEquivariance and
NVIDIA's dtv1 Triton stack in all 240 cells, smallest ratio 1.10x (bidirectional D512 L128 training). From the 2026-10-21 note
(outgoing, hidden = 2 D, token mask 15 %, CUDA graph): 2.1-3.4x cuEquivariance in inference and 2.0-3.7x in training at D64 / D128,
B 1 / 4, L 128-768.

## History

The previous, longer notes (H100 TriMul, trimul_inproj, tm1 / tm2, the module, the Anthropic payload) were deleted; see git
history.

## Experimental: e4m3 planes between K1 and the contraction (opt-in, off by default)

`kernels/trimul_inproj/cuda/h100_b200style_fp8.py`, `_h100_fp8.py` (2026-11-14, branch `fp8-trimul-fwd`). Single direction, one sample, D128 / D256
(hidden = pair), bf16 inference, capability (9, 0). Nothing runs unless `h100_b200style_fp8.POLICY` (env `MINIWORLD_TRIMUL_FP8`) is set, so every
default route (and every token / judge result) is unchanged.

- K1 (`h100_b200style_direct.front(qinv=...)`, `tmn_kernels.cuh` `TMN_K1_F8`): the n128 blocks (D128 takes the `(2, 64, 6, 1, 1, 6, 128)` tile in this mode),
  the projection weights prescaled per plane channel by a power of two, planes written as e4m3 bytes (saturating `cvt.rn.satfinite.e4m3x2`, byte
  exchange between neighbouring lanes, 64-byte rows, 64B-swizzled TMA store). Incoming: K1 reads the pair through a transposed tensor map, so the planes come out
  transposed (K-major) and the LN statistics are transposed afterwards.
- Contraction: cuBLASLt batched e4m3 GEMM (TN, bf16 out, no FAST_ACCUM: the tensor-core accumulator is promoted to fp32 periodically). Per-channel scales
  factor out of the contraction exactly (`t = sa sb sum_k qa qb`); scales are powers of two, so the rescale is exact in bf16.
- K3 (`k3.cu` `K3_CSCALE`): the scale `sa sb` is folded into the projection operand `Wpf` (host, exact) and weights the LN_out statistics.
- Scales: `dynamic` (exact amax of the bf16 planes, a bf16 K1 pre-pass: numerics only) or `calib` (first call of a module, 4x headroom, static afterwards).

### Selecting the FP8 routes (opt-in; every default route is bit-identical, 24 configs D64/D128/D256 checked against origin/main ac7b5f5d7)

| env (read at import) | effect |
|---|---|
| `MINIWORLD_TRIMUL_FP8=calib` + `MINIWORLD_TRIMUL_FP8_GUARD=lazy` | e4m3 planes (bf16-compute K1), cuBLASLt e4m3 contraction, static calibrated scales, lazy overflow guard (graph-capturable) |
| `MINIWORLD_TRIMUL_FP8_FASTACC=1` | cuBLASLt fast accumulation of the e4m3 contraction: 1.07-1.08x of the contraction at L>=1024, module output moves by < 0.1 % of the update (the error of the route is unchanged to 4 digits on real Protenix dumps) |
| `MINIWORLD_TRIMUL_FP8_K1=1` | the K1 projections themselves in e4m3 (EXPERIMENTAL: update error 6.9 % vs 3.8 % for planes only; FAILED the Protenix n=12 trunk-seed pLDDT criterion: never a default) |

D128 FP8 planes now use the three-warpgroup K1 tile `(3, 64, 6, 1, 1, 6, 128)` (24 KB ring slots; bf16 weights in z-chunk-sized slots, one `[64 k][128 rows]` TMA load per slot):
K1 0.84-0.90x of the two-warpgroup tile, planes bit-identical. D256 cannot take it (the ring would need 8 slots x 24 KB). D64 is not covered (the n128 ring K1 does not run at D64).
cuBLASLt heuristic candidates (8) and a neighbourhood sweep of tile / stages / cluster / swizzle around the top candidate (168 valid configs at D128 L2048) found nothing faster than the
default heuristic pick (1.29 PFLOPs = 65 % of the dense e4m3 peak), so the contraction algorithm is not a lever.

### FP8 planes: what changed on 2026-11-15 (`fp8/tm2`; numbers and the per-stage table in `docs/fp8.md`)

- Incoming: K1 reads the pair through the transposed map and now stores the LN_in statistics at the swapped pair token itself (`K1Params.stats_t`, a field in the padding after `rows`, read only in `TMN_K1_F8` builds), so K3 reads them directly: the host transpose of `xs` (4 host ops + one elementwise kernel, 3-14 us) is gone.
  The amax buffer is allocated once per module (it was a `torch.zeros(1)` per call through `setdefault`). Output bytes unchanged.
- Min-L gates (`MIN_L`, `MINIWORLD_TRIMUL_FP8_MINL`): D128 out 144, D128 in 192, D256 out 224, D256 in 144, bidirectional D128 256, bidirectional D256 1024; below them the bf16 route runs unchanged.
- Bidirectional block (2026-11-16, `fp8/bidir-tm`): `h100_b200style_fp8.run(direction=0)` runs the two directions as two single-direction FP8 K1 + e4m3 contractions into one `tri` and one K3 over 2 d channels (`K3_CSCALE=1`); numbers and the audit of the default bidirectional routes in `docs/fp8.md` and "Bidirectional inference" above.
- Prebuild: the FP8 K1 tile(s) and the `K3_CSCALE` K3 (row form; D256 also the token-flat form) are in the sm90 plan (`kernels/trimul_inproj/cuda/__init__.py`); an FP8 forward against a prebuilt bundle and an empty JIT cache compiles nothing.
- Measured (graph replay, module, vs the 2.5.9 default): D128 -9 .. -25 % (L256 .. L2048), D256 -4 .. -16 %. The contraction is a serial stage (removing the cuBLAS call removes exactly its time); the K1 overflow guard costs < 1 % (K1 with the amax pointer removed);
  D256 is capped by Amdahl (contraction 30 % of the module, e4m3 1.47x of it) and its FP8 K1 is 2-6 % slower than the bf16 K1.

### FP8 planes at D64 / D384, bidirectional D64 / D384, B > 1 (3.0.0 exploration, branch `fp8x/trimul`, 2026-11-17; numbers in `docs/fp8.md`)

- K1 (`tmn_kernels.cuh`, `TMN_K1_F8` builds only): an e4m3 plane store for the n64 weight blocks (32 plane channels per block, accumulator groups
  q < 4 gate / q + 4 projection): the n128 form's byte exchange on four column groups, staged `[32 ch][64 tok]` (64-byte rows, 64B swizzle), one TMA
  store per block. `h100_b200style_direct.F8_CFG`: D64 takes the long D64 direct tile `(3, 64, 4, 1, 1)` (BJ 64: TMA stores), D384 the D384 direct
  tile of the bf16 route `(2, 64, 2, 6, 1)` (an n128 D384 ring does not fit shared memory). The n64 blocks take unhalved prescaled weights
  (`prescale(..., half=False)`: no `TMN_K1_FOLD`).
- K3: the route's existing `K3_CSCALE` K3; D384 at L % 128 == 0 takes the token-flat K3 (the D384 default tile), else the row form.
- `serves`: widths `WIDTHS = (64, 128, 256, 384)`, 1-16 samples (`MAX_SAMPLES`; B > 1 runs `run_samples`, the one-sample route per sample into one
  output); gates in `MIN_L` (D64 512, D384 128, bidirectional D64 2048, bidirectional D384 1024).
- D512 is not served: its inference K1 is the packed `_front` and its K3 `k3reorder.cu` has no `K3_CSCALE`.
- Default route: unchanged (the header code is under `TMN_K1_F8`; sha1 of the outputs against the base tree in `docs/fp8.md`).
