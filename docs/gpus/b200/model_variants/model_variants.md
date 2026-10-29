# Boltz-2 / RoseTTAFold3 / Chai-1 DiT stacks on B200 (sm100), inference and training

`modules.dit_variant.DiTVariantStack(n, **BOLTZ2 / ROSETTAFOLD3 / CHAI1)` (token, 768 wide, 16 x 48 heads) and
`AtomDiTVariantStack(n, **BOLTZ2_ATOM / ROSETTAFOLD3_ATOM / CHAI1_ATOM)` (atoms, 128 wide, 4 x 32 heads, 32 x 128 windows) on B200,
2026-10-29. Before this they ran the PyTorch statements (`kernels/dit_variants/reference.py`) on B200; the H100 page is
[../../h100/model_variants/model_variants.md](../../h100/model_variants/model_variants.md).

The variants' attention is the AF3 math the engine's B200 cores already compute, so B200 runs those cores -- tcgen05 / TMEM / TMA,
no new attention kernel -- around the variants' CUDA row kernels (`kernels/dit_variants/cuda/rows.cu`, `train_rows.cu`, now built for
sm_100a: plain CUDA, no architecture-specific instruction) and cuBLAS for the plain GEMMs. The glue is
`/NHNHOME/WORKSPACE/26mohw002_A/psk6950/eng-k4-dv/src/miniworld_engine/kernels/dit_variants/cuda/sm100.py`.

| path | B200 kernels | what differs from AF3 and how it is served |
|---|---|---|
| token, bf16 inference (`interface.py`) | `augmented_attention/cuda/sm100/attn_inf.cu` (`GatedInferenceCore`) | q \| k \| v \| g in exp2 units, the bias of every block one head-major [nb 16, L, L] log2 tensor (masked keys -inf), no gating query (Chai-1): gate column +1e4; RF3's kq LayerNorm: `qk_ln_rows` with the logit scale in its affine. Any L that is a multiple of 8 (H100: of 128) ; the GEMM-epilogue kernels `kernels/dit_variants/cuda/dit_gemm_sm100.cu` (below) |
| token, fp32 inference (`cuda/infer32.py`) | `attn_inf_tf32.cu` (TF32 tensor cores) | q \| k \| g + v^T, the bias in natural key order (H100's TF32 core takes it key-permuted) |
| token, training (`cuda/train.py`) | `attn_fwd2` / `attn_dkv` / `attn_dqb` (bf16), `attn_fwd_tf32` / `attn_dkv_tf32` / `attn_dqb_tf32` (fp32) | q, k, v unscaled (column views of the q \| k \| v \| g GEMM without kq-norm), bias in natural units with the masked keys -inf, the fp32 O gated by the new row kernel `ogate_fwd`; dbias comes back in natural units. Deterministic `-DDET=1 -DDQPART=1` builds (below). Odd A: one zero sample padded in. RF3 / Chai-1 bf16 at M >= 3840: the expand GEMM with the SwiGLU in its epilogue (`gemm_swiglu2_sm100 -DSAVE_AB`) |
| atom, all paths | `sm100_atom_local`: `lattn_fwd` / `lattn_dq` / `lattn_dkv` and the `*_tf32` twins with the slid key window inside (`slide.cuh` edge CTAs, `lslide` adds their dK / dV partials), `lbias` (pair bias) | the pair bias in natural units from `lbias` per block (per-block or shared pair norm, output channel 4 b + h); Chai-1's same-conformer mask folded in as -inf at the keys each window really reads (the slid ones for "slide": the edge CTAs read the same bias slots over their slid keys); N padded to a multiple of 128 (zero rows, the padded keys masked) |

## Deterministic backward

The token training path takes the AugmentedAttention core's deterministic backward (`kernels/augmented_attention/cuda/sm100/fused_pair.py`
`_bwd_bf16` / `_bwd_tf32`: attn_dkv / attn_dqb and their TF32 twins built `-DDET=1 -DDQPART=1`, the two MMA issuers in a fixed order,
one dQ partial per key chunk) and sums the dQ partials in chunk order; the atom path's local DiT kernels and every row kernel sum in a
fixed order. The default builds of the token cores (what the AF3 token DiT training runs) are not bitwise repeatable: their dQ is
reduce-added per key chunk and the TF32 dK / dV issuers are unordered (measured: dQ, and in fp32 dK / dV, differ run to run).

## Tests

`tests/gpu/sm100/test_b200_dit_variant_gpu.py` (89 tests, all passed 2026-10-29), bars of the H100 tests: inference within max(1.5 x the bf16 reference's
error against fp64, 1e-2) (fp32: 1.5 x the fp32 reference's); training the output and every gradient (single, cond, pair, every parameter)
within 1.5 x the same-dtype PyTorch reference's error against fp64, a tensor whose true gradient is zero held to 1e-4 of the largest
gradient norm. Cases: the three models x token / atom x bf16 / fp32; token L 128-768 incl. L 200 (inference), masked / unmasked, odd A;
atom N 256-1600 incl. N not a multiple of 128, slid / padded key window, Chai-1 `uid`; bitwise repeatability (token L 384 and atom, bf16
and fp32); a CUDA-graph replay equal to an eager call; a torch.compile'd stack equal to the eager one (the B200 stacks run outside
Dynamo, a graph break: traced, token training failed torch's custom-op aliasing check and atom training hit a Dynamo recursion error).

## Measurements (2026-10-29)

B200 node 2, one GPU (another user's idle-time filler on the card, which yields), `benchmarks/runners/bench.py level=module target=dit |
local_dit +dit_variant=<model>`: compile on, CUDA graph (inference graphed; training graph off and on), the harness defaults (bf16 =
`bf16-mixed`, fp32 master parameters under autocast; fp32 = `precision=32`, TF32), one run per cell. "before" is the `pytorch` rung:
the module's PyTorch statements compiled -- what B200 ran for these stacks until now; "after" the `miniworld` rung. ms. The rows whose
kernels changed with the rebase on the port (token training: the merged `-DDET=1 -DDQPART=1` backward; every atom row: the slid window
inside the local DiT kernels) were re-measured on it (miniworld rung only; the PyTorch rung's code is unchanged).

### Inference, A = 5 (24 blocks Boltz-2 / RF3, 16 Chai-1; atom stacks 3 blocks)

| model | dtype | token L384 before / after | token L768 before / after | atom 8192 before / after |
|---|---|---|---|---|
| Boltz-2 | bf16 | 4.69 / 1.57 (2.98x) | 9.28 / 2.78 (3.34x) | 1.140 / 0.518 (2.20x) |
| RoseTTAFold3 | bf16 | 4.46 / 1.59 (2.81x) | 9.03 / 2.63 (3.43x) | 1.180 / 0.512 (2.30x) |
| Chai-1 | bf16 | 3.03 / 1.06 (2.87x) | 7.09 / 1.77 (4.00x) | 1.021 / 0.393 (2.60x) |
| Boltz-2 | fp32 | 6.30 / 3.01 (2.09x) | 13.74 / 5.43 (2.53x) | 1.580 / 0.811 (1.95x) |
| RoseTTAFold3 | fp32 | 5.99 / 2.88 (2.08x) | 12.99 / 5.07 (2.56x) | 1.584 / 0.755 (2.10x) |
| Chai-1 | fp32 | 5.26 / 1.99 (2.65x) | 14.57 / 3.54 (4.11x) | 1.437 / 0.622 (2.31x) |

### Training step (forward + backward), A = 48, one block: token L384, atom 4096 (512 tokens)

CUDA graph on; graph off in brackets. The compiled PyTorch atom rung failed the harness's own graph-replay check in four of six
cells (its gradients differ between the timed replay and the reference call by 3e-3 to 5e-3) -- those "before" cells are its graph-off time.

| model | dtype | token before / after | atom before / after |
|---|---|---|---|
| Boltz-2 | bf16 | 3.65 / 1.74 (2.10x) [3.96 / 2.45] | (6.88) / 2.07 [6.88 / 2.41] |
| RoseTTAFold3 | bf16 | 3.65 / 1.65 (2.21x) [3.94 / 2.16] | (6.82) / 2.09 [6.82 / 2.43] |
| Chai-1 | bf16 | 3.38 / 1.48 (2.29x) [3.64 / 1.96] | (5.95) / 1.60 [5.95 / 2.04] |
| Boltz-2 | fp32 | 4.36 / 2.92 (1.49x) [4.50 / 3.09] | 6.31 / 2.95 (2.14x) [6.58 / 3.19] |
| RoseTTAFold3 | fp32 | 4.06 / 2.68 (1.51x) [4.23 / 2.85] | (6.35) / 2.85 [6.35 / 3.08] |
| Chai-1 | fp32 | 3.85 / 2.52 (1.53x) [4.01 / 2.69] | 5.33 / 2.34 (2.28x) [5.55 / 2.58] |

### Against H100 (the ladder's shapes and harness: compile + CUDA graph, one block; H100 values from
[../../../records/dit_variants_cuda_ladder_20261026.md](../../../records/dit_variants_cuda_ladder_20261026.md), different date and card)

Inference A = 5, training A = 48; token L384, atom 4096. B200 "before" is the compiled PyTorch rung of the same run.

| row | dtype | H100 CUDA | B200 before | B200 after | H100 / B200 |
|---|---|---|---|---|---|
| Boltz-2 token, inference | bf16 | 0.118 | 0.207 | 0.090 | 1.31 |
| RoseTTAFold3 token, inference | bf16 | 0.108 | 0.192 | 0.086 | 1.26 |
| Chai-1 token, inference | bf16 | 0.108 | 0.192 | 0.080 | 1.35 |
| Boltz-2 atom, inference | bf16 | 0.159 | 0.258 | 0.119 | 1.34 |
| RoseTTAFold3 atom, inference | bf16 | 0.154 | 0.264 | 0.117 | 1.32 |
| Chai-1 atom, inference | bf16 | 0.129 | (failed its check) | 0.086 | 1.50 |
| Boltz-2 token, inference | fp32 | 0.236 | 0.266 | 0.141 | 1.67 |
| RoseTTAFold3 token, inference | fp32 | 0.206 | 0.255 | 0.133 | 1.55 |
| Chai-1 token, inference | fp32 | 0.213 | 0.270 | 0.137 | 1.55 |
| Boltz-2 atom, inference | fp32 | 0.290 | 0.314 | 0.153 | 1.89 |
| RoseTTAFold3 atom, inference | fp32 | 0.273 | 0.307 | 0.147 | 1.85 |
| Chai-1 atom, inference | fp32 | 0.228 | 0.282 | 0.118 | 1.93 |
| Boltz-2 token, training | bf16 | 3.054 | 3.651 | 1.738 | 1.76 |
| RoseTTAFold3 token, training | bf16 | 2.786 | 3.649 | 1.649 | 1.69 |
| Chai-1 token, training | bf16 | 2.637 | 3.380 | 1.477 | 1.79 |
| Boltz-2 atom, training | bf16 | 3.481 | (6.884 graph off) | 2.067 | 1.68 |
| RoseTTAFold3 atom, training | bf16 | 3.392 | (6.822 graph off) | 2.092 | 1.62 |
| Chai-1 atom, training | bf16 | 2.853 | (5.948 graph off) | 1.600 | 1.78 |
| Boltz-2 token, training | fp32 | 6.630 | 4.361 | 2.922 | 2.27 |
| RoseTTAFold3 token, training | fp32 | 6.071 | 4.058 | 2.682 | 2.26 |
| Chai-1 token, training | fp32 | 5.828 | 3.852 | 2.516 | 2.32 |
| Boltz-2 atom, training | fp32 | 5.835 | 6.309 | 2.953 | 1.98 |
| RoseTTAFold3 atom, training | fp32 | 5.541 | (6.354 graph off) | 2.848 | 1.95 |
| Chai-1 atom, training | fp32 | 4.770 | 5.327 | 2.338 | 2.04 |

## GEMM-epilogue fusion of the bf16 token step (2026-10-29)

`kernels/dit_variants/cuda/dit_gemm_sm100.cu` (host `cuda/dit_gemm.py`), hand CUDA sm_100a, two kernels:

* `dit_glu_gemm_sm100` -- the expand GEMM with the GLU in its epilogue: h = silu(a) b, and Boltz-2's up-gated silu(a) b u (three
  operands, the up weight concatenated under [Wg; Wv]) -- the up-gate no longer runs cuBLAS + the GLU row pass. 2-CTA
  (`cta_group::2`) persistent over (256-row tile pair, hidden chunk) items: 128-unit chunks (N 256) for two operands, 64-unit (N 192)
  for three; [a | b | u] double-buffered in tensor memory, h staged in shared memory, TMA stores.
* `dit_res_gemm_sm100` -- the to_out / squeeze GEMMs with the residual and the next AdaLN in their epilogue: x += gA (o Wo^T)
  [+ gT (h Wsq^T)] (two accumulators for a parallel block), then x written back and LN(x) f(ms) + mb for one or two AdaLNs, or the
  stack's bf16 output. One 8-CTA cluster per 128-row tile (a CTA per 96 columns, 1-CTA MMA, the A tile multicast over the cluster);
  the x and gate tiles come in by TMA during the main loop (cp.async where a tile's table rows wrap around L), the AdaLN tables into
  the ring once it is done; the row's LayerNorm partials meet over distributed shared memory (`st.async`) and are combined (Chan).
* Per step the gate and AdaLN-scale logits of the conditioning tables become sigmoid values once (`rows.cu` `sigmoid_tables`, L rows),
  and the conditioning LayerNorm is one CUDA pass (`layernorm_rows` at 384, `ln_mod_rows` at 768) instead of cast + LN + cast.
* Both kernels use programmatic dependent launch (`griddepcontrol`; the prologue -- barriers, tensor-memory allocation, the cluster
  barrier, ~1 us -- overlaps the previous kernel). No spills (ptxas: GLU 158-166, RES 127-136 registers of 168).

Routing (`interface._inference_b200`, measured): sequential blocks (Boltz-2) take `glu_gemm` at every M (to_out / squeeze stay cuBLAS
+ the residual row pass: the 1-CTA res_gemm with a single accumulator measured slower than them at L 384, A 5); parallel blocks (RF3,
Chai-1) take glu_gemm + the two-accumulator res_gemm up to M = 2048 rows (one wave of 8-CTA clusters), with one block only for the
sigmoid AdaLN (RF3), i.e. where it measured faster. Kernels a block: Boltz-2 8 -> 7, RF3 / Chai-1 9 / 8 -> 6 / 5.

Measured (B200 node 2, `bench.py level=module target=dit +dit_variant=<model>`, compile, CUDA graph, bf16-mixed, A = 5, one run per
cell; "before" = the row-pass step of the same build, `interface.FUSED_B200 = False`), ms:

| model | blocks | L384 before / after | L768 before / after |
|---|---|---|---|
| Boltz-2 | 1 | 0.0880 / **0.0818** | 0.1393 / **0.1269** |
| RoseTTAFold3 | 1 | 0.0860 / **0.0838** | 0.1309 (row-pass step kept) |
| Chai-1 | 1 | 0.0799 (row-pass step kept) | 0.1269 (row-pass step kept) |
| Boltz-2 | 24 | 1.576 / **1.508** | 2.778 / **2.597** |
| RoseTTAFold3 | 24 | 1.594 / **1.549** | 2.631 (kept; fused measured 2.955) |
| Chai-1 | 16 | 1.057 / **1.030** | 1.766 (kept; fused measured 2.002) |

One block against H100 (0.118 / 0.108 / 0.108 ms, the table above): Boltz-2 1.44x (was 1.34x on the same day's row-pass step),
RF3 1.29x, Chai-1 1.35x.

Why not more (kernel-level, M = 1920, CUDA-graph replays of the kernel alone): a tcgen05 kernel carries ~2 us of fixed cost here
(launch 0.8, tensor-memory alloc / dealloc 0.6, cluster barriers, the first TMA), the fp32 residual's epilogue traffic is the row
pass's own (x read and written, the per-token tables re-read per sample), and the GEMMs are TMA-ingress bound (~55 B/clk/SM) like
cuBLAS. res_gemm one accumulator + LN 10.8 us against cuBLAS + `res_rows` 10.6; two accumulators + two AdaLNs 18.5 against 21.4;
glu_gemm up-gate 15.0 against 16.9, plain SwiGLU 14.4 against 14.0 (cuBLAS + glu_rows). Things measured on the way: a 2-CTA, 6-CTA
cluster res_gemm (48 SMs at M 1920) was epilogue-bound at 30 us; one row per thread with direct global accesses serialised in L1
(32 lines a warp instruction); a release-scope DSMEM arrive waited for the thread's in-flight cp.async (3 us; `st.async` fixed it);
volatile shared-memory asm serialised the epilogue math; the epilogue sigmoid was MUFU-bound (hence the per-step sigmoid of the
tables).

Tests: `tests/gpu/sm100/test_b200_dit_gemm_gpu.py` (both kernels against fp32 math of the same operands, within 1.5x the unfused
path's error; raw and pre-sigmoided tables; M 200-3840 incl. M not a multiple of 128 and tables wrapping L; a captured chain replayed
3000 times bitwise equal) and `test_b200_dit_variant_gpu.py` (unchanged bars).

Not done / open: a 2-CTA res_gemm (16-CTA cluster, 256 x 96 pairs) would cut its main loop by ~25 % and is what the sequential
blocks' to_out / squeeze would need to pay; the RES path past one wave (L 768); fusing RF3's kq LayerNorm into the projection GEMM.

Before this (2026-10-29 morning): one-block bf16 token inference at A = 5 is latency-bound (1.26-1.35x over H100, under the 1.4x line): per block it is still the
H100 path's ~9 launches (rows, cuBLAS, the attention core); the levers are the gate and the residual rows in the attention / GEMM epilogues
and the AF3 runner's per-step conditioning hoist across sampler steps. Boltz-2's up-gate transition (silu(g) v u) keeps cuBLAS + the GLU
row pass (the SwiGLU GEMM has no third operand). The fp32 token training rows are TF32 GEMM-bound (1.5x over compiled PyTorch).
