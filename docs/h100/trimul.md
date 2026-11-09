# Triangle multiplicative update on H100 (sm90)

Scope: `TriangleMultiplication` (backends `pytorch`, `cuda`, `triton`, `cueq`, `anthropic`), `BidirectionalTriangleMultiplication`
(`pytorch`, `cuda`, `triton`) and `TriangleMultiplicationRF3` (`pytorch` only: the contraction divided by the sequence length is
in no kernel) of `modules/triangle.py`. The requirement rows (`requirements/requirements_module.csv`) ask for L 128-768
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
| `cuda` | `integrations/trimul_h100.py` over `kernels/trimul_inproj/cuda/`: `update_inference` without grad or live dropout, else `update` (bidirectional: `serves`; single direction: `serves_single`) | bf16 pair on capability (9, 0), square, bool `[B, L]` mask, plus the integration's contract (below) |
| `triton` | `trimul_inproj/triton/unidirectional.trimul_triton` / `bidirectional.bidirectional_trimul_triton`: the whole update incl. residual and row dropout, one launch sequence per sample | CUDA, bf16 pair, square, bool mask; d_hidden != d_pair only with both multiples of 16, d_pair <= 256, d_hidden <= 512 (`contract.hidden_refusal`) |
| `cueq` | `cuequivariance_torch.triangle_multiplicative_update` (single direction), residual and dropout by the module | projections in the pair's dtype (the `master` rows are unsupported), one LN epsilon, widths % 32, bf16 / fp32, the library's kernel band |
| `anthropic` | `opt_core.kernels.trimul` (tier `fast`), single direction, residual in its epilogue | inference only, no live dropout, projections in the pair's dtype, rows listed in `_ANTHROPIC_EXCLUDED_ROWS` excluded |

## Kernels

Names from `requirements/requirements_kernel.csv` (family `trimul_inproj`, 98 names); sources under
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
  `trimul_input_ln_residual_bwd_triton`, `gated_projection_gate_dropres_triton`, `gated_projection_bwd_gate_dropres_triton`
  (exact key lookup). `trimul_gemm_gate_mmajor_triton`, `trimul_input_dual_bwd_triton` and
  `trimul_bwd_gate_packed_recompute_triton` have no entries and run their default config.
- `kernels/tm1/configs`, `kernels/tm2/configs` hold tables for the unused tm1 / tm2 kernels.
- The other CUDA paths use family JSONs under `h100_sources/` (`b1/configs.json`, `single_output/selection.json`,
  `wide_forward/selection.json`, `uni_wide/selection.json`, `wide_train/lt_selection.json`) and hard-wired per-width tiles.

## Limits / unsupported calls

`cuda`: bf16 pair only; capability (9, 0); square pair and bool `[B, L]` mask; LN eps 1e-5 and fp32 LN affine; d_hidden = d_pair,
or 2 d_pair in one direction at D64 / D128 (other hidden widths, and every width outside 64 / 128 / 256 / 384 / 512, are refused);
training lengths L % 128 == 0 up to 10240; D128 training only on a 132-SM device. `triton`: bf16 pair, `contract.hidden_refusal`.
`cueq` / `anthropic`: single direction only, projections in the pair's dtype. `TriangleMultiplicationRF3` has no backend but
`pytorch`. Any refusal is a reason string (`BackendUnsupported`), never a fallback.

## Historical numbers

Historical (from the 2026-10-15 note, H100 80GB, compiled + manual CUDA graph, B = 1, fp32 masters, mask probability 0.1, L
128-1024, D 64-512; not re-measured under the current measurer): the CUDA path was the fastest of PyTorch, cuEquivariance and
NVIDIA's dtv1 Triton stack in all 240 cells, smallest ratio 1.10x (bidirectional D512 L128 training). From the 2026-10-21 note
(outgoing, hidden = 2 D, token mask 15 %, CUDA graph): 2.1-3.4x cuEquivariance in inference and 2.0-3.7x in training at D64 / D128,
B 1 / 4, L 128-768.

## History

The previous notes are in git HEAD: `docs/gpus/h100/trimul/trimul.md`, `docs/kernels/trimul-inproj.md`, `docs/kernels/tm1.md`,
`docs/kernels/tm2.md`, `docs/kernels/triangle-multiplication-module.md`, `docs/kernels/anthropic-trimul-payload.md`
(`git show HEAD:<path>`).
