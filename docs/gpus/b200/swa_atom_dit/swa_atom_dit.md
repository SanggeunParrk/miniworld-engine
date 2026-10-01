# SWA atom DiT on B200 (sm100)

Kernel-level status of the ESMFold2-style SWA atom DiT block (MiniWorld `block_style: esmfold2`, engine `modules/swa_dit`) on
B200: adaLN (RMSNorm, shift / scale / gate from the conditioning) -> q | k | v | gate projection with per-head RMS norm and RoPE ->
sliding-window attention (|i - j| <= 64, 4 heads x 32) -> gated out-projection + residual -> adaLN + SwiGLU (hidden 256) + residual;
d_single = d_cond = 128. Columns are (Length, Dimension) from the shape registry (`swa_atom_attention`, atom axis, bf16 only);
Length is the atom count S. The module-level summary is in [b200.md](../b200.md).

**Where the code is.** The hand-written sm_100a kernels below live in the B200 research capsule
`experiments/swaatom_sm100` (ignored by git, not in this repository) and are measured there on the block of the H100 fused
Triton algorithm with its launches swapped for the sm_100a kernels (`b200_block.install()`). **The engine does not dispatch them
yet**: on B200 the engine's `SWADiTBlock` runs FA4 window attention + Triton row kernels + cuBLAS. "CUDA" in the kernel tables
therefore means "a hand-written kernel exists and was run at that shape in the capsule"; the module-level table in b200.md
says 미구현 until the engine dispatches it.

- Everything is hand CUDA (tcgen05 / TMEM / TMA) except the four weight-gradient GEMMs (cuBLAS). No Triton, no quack.
- Shapes: A = augments per sample (rows = A S). Inference A = 5 and A = 1 at S = 1024 / 2048 / 4096; training A = 48 at
  S = 4096 / 8192. **S must be a multiple of 128**: callers pad the atoms (seqused masks the padding); anything else raises
  `ValueError` -- there is no fallback path.
- Dtypes: activations, weights and their gradients bf16; the modulation (shift / scale / gate, [B S, 768]) and its gradient
  fp32; RoPE tables and the attention LSE fp32; every MMA accumulates in fp32 (TMEM).
- cache build ✓ everywhere: nothing autotunes (cubins with fixed launch shapes; dispatch by fixed size rules).
- 성능 확인 in this page follows the rule set on 2026-09-30: **✓** = the block is the fastest of every measured
  implementation at that shape **and** the kernel reaches >= 70 % of its measured floor (SoL section); **△** = the block is the
  fastest, the kernel below 70 %; **✗** = not measured at that shape (CUDA 미검증: the kernel accepts it, nothing ran there).

## Inference

Per block: I1 -> I2 -> I3 -> I4, with PDL between them (the next kernel's launch and weight prologue run under the previous one).
With the modulation precomputed (hoisted per fold, as Anthropic's driver takes it) I1 drops out.

### I1 · adaLN modulation `mod_fwd` (silu(c) Wmod^T, fp32 out; bf16 MMA with exact products)

| (Length, Dimension) | (1024, 128) | (2048, 128) | (3072, 128) | (4096, 128) | (5120, 128) | (6144, 128) | (7168, 128) | (8192, 128) |
|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA 미검증 | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA 미검증 | CUDA 미검증 |
| 성능 확인 | △ | △ | ✗ | △ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### I2 · RMS-adaLN + q | k | v | gate projections + head RMS + RoPE `qkvg_fwd2` / `qkvg_fwd`

`qkvg_fwd2` (transposed: 32-row tiles, the 4 x 128 x 128 weights resident in TMEM as the A operand) when A < 5 or A S <= 9472
(A = 1: its ATM = 32 build, one tile's 32 atoms of modulation); else `qkvg_fwd` (128-row tiles).

| (Length, Dimension) | (1024, 128) | (2048, 128) | (3072, 128) | (4096, 128) | (5120, 128) | (6144, 128) | (7168, 128) | (8192, 128) |
|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA 미검증 | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA 미검증 | CUDA 미검증 |
| 성능 확인 | △ | △ | ✗ | △ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### I3 · window attention forward `attn_fwd3` / `attn_fwd1h`

`attn_fwd3`: items (sample, 128-query tile, head pair), both 128-key blocks' scores in TMEM, the row maximum over both before any
exponential; the two heads' P phases take turns. `attn_fwd1h` (one head per item, the two warpgroups splitting its keys) when its
items fit in one wave (N (S / 128) 4 <= 148: A = 1).

| (Length, Dimension) | (1024, 128) | (2048, 128) | (3072, 128) | (4096, 128) | (5120, 128) | (6144, 128) | (7168, 128) | (8192, 128) |
|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA 미검증 | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA 미검증 | CUDA 미검증 |
| 성능 확인 | △ | △ | ✗ | △ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### I4 · gated out-projection + residual + RMS-adaLN + SwiGLU + residual `ffn_fwd2`

Transposed, 32-row tiles; Wu and Wd resident in TMEM, Wo in shared memory; double-buffered tile stages (A >= 4) or its ATM = 32,
single-stage build (A = 1 - 3).

| (Length, Dimension) | (1024, 128) | (2048, 128) | (3072, 128) | (4096, 128) | (5120, 128) | (6144, 128) | (7168, 128) | (8192, 128) |
|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA 미검증 | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA 미검증 | CUDA 미검증 |
| 성능 확인 | △ | △ | ✗ | △ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Training

Forward T1 -> T4 (the inference kernels, saving q1 / att / y / ffn and x / rn(p_q) / rn(p_k) for the backward), backward T5 -> T10
plus four cuBLAS weight-gradient GEMMs (dWqkv | dWg as one GEMM over dP, dWo, dWu, dWd). The modulation gradient accumulates in
fp32 across T6, T7, T9 (atomics per atom and channel) before T10.

### T1 · `mod_fwd` · T2 · `qkvg_fwd` (saves) · T3 · `attn_fwd3` · T4 · `ffn_fwd2` (saves)

| (Length, Dimension) | (1024, 128) | (2048, 128) | (3072, 128) | (4096, 128) | (5120, 128) | (6144, 128) | (7168, 128) | (8192, 128) |
|---|---|---|---|---|---|---|---|---|
| implementation | CUDA 미검증 | CUDA 미검증 | CUDA 미검증 | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA 미검증 | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | △ | ✗ | ✗ | ✗ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### T5 · FFN backward, gate `ffn_bwd_gate` (dffn, h, dA | dB with Wu and Wd^T resident in TMEM)

| (Length, Dimension) | (1024, 128) | (2048, 128) | (3072, 128) | (4096, 128) | (5120, 128) | (6144, 128) | (7168, 128) | (8192, 128) |
|---|---|---|---|---|---|---|---|---|
| implementation | CUDA 미검증 | CUDA 미검증 | CUDA 미검증 | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA 미검증 | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### T6 · FFN backward, input side `ffn_bwd_dy` (dy = dAB Wu, the adaLN backward, dq1, dmod)

| (Length, Dimension) | (1024, 128) | (2048, 128) | (3072, 128) | (4096, 128) | (5120, 128) | (6144, 128) | (7168, 128) | (8192, 128) |
|---|---|---|---|---|---|---|---|---|
| implementation | CUDA 미검증 | CUDA 미검증 | CUDA 미검증 | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA 미검증 | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### T7 · out-projection backward `oproj_bwd` (dO, dG, the attention D = rowsum(dO o), dmod)

| (Length, Dimension) | (1024, 128) | (2048, 128) | (3072, 128) | (4096, 128) | (5120, 128) | (6144, 128) | (7168, 128) | (8192, 128) |
|---|---|---|---|---|---|---|---|---|
| implementation | CUDA 미검증 | CUDA 미검증 | CUDA 미검증 | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA 미검증 | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### T8 · window attention backward, dQ + dK + dV in one pass `attn_dkvq`

dS^T written once to shared memory is the K-major A of dK and the MN-major A of dQ (M = 64). A 64-query block's dQ comes from
exactly two key tiles; CTAs walk contiguous runs of key tiles, so the pair meets inside a CTA (fp32 carry in shared memory) or,
at a run's ends, through a global slot and a counter (the second to arrive sums: deterministic). No fp32 dQ buffer. Used when
every CTA gets >= 2 items; else `attn_dq` + `attn_dkv`.

| (Length, Dimension) | (1024, 128) | (2048, 128) | (3072, 128) | (4096, 128) | (5120, 128) | (6144, 128) | (7168, 128) | (8192, 128) |
|---|---|---|---|---|---|---|---|---|
| implementation | CUDA 미검증 | CUDA 미검증 | CUDA 미검증 | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA 미검증 | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | △ | ✗ | ✗ | ✗ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### T9 · q | k | v | gate backward `qkvg_bwd` (head RMS + RoPE backward, dx = W^T dP^T with W^T resident in TMEM, dmod)

| (Length, Dimension) | (1024, 128) | (2048, 128) | (3072, 128) | (4096, 128) | (5120, 128) | (6144, 128) | (7168, 128) | (8192, 128) |
|---|---|---|---|---|---|---|---|---|
| implementation | CUDA 미검증 | CUDA 미검증 | CUDA 미검증 | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA 미검증 | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### T10 · modulation backward `mod_bwd` (dc, dWmod; dmod fp32 split into three bf16 terms)

Two kernels, each reducing over a cluster (dc: 4 CTAs over the 768 channels; dW: up to 16 CTAs over the rows) through
bulk shared::cluster copies, summed in rank order.

| (Length, Dimension) | (1024, 128) | (2048, 128) | (3072, 128) | (4096, 128) | (5120, 128) | (6144, 128) | (7168, 128) | (8192, 128) |
|---|---|---|---|---|---|---|---|---|
| implementation | CUDA 미검증 | CUDA 미검증 | CUDA 미검증 | CUDA | CUDA 미검증 | CUDA 미검증 | CUDA 미검증 | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | △ | ✗ | ✗ | ✗ | △ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Measurements (2026-09-30)

- Setup: B200 (148 SMs, 1000 W power cap), GPU 2 through `gpuq` with no other process on it, old uv venv of the capsule
  (torch 2.10 cu130) for the capsule, H100 fused, PyTorch and Anthropic rows; the v2.2.0 pixi env (torch 2.13.0+cu129) for the
  engine row; the capsule's cubins built by nvcc 13.1 (`/usr/local/cuda-13.1`). B = 1, bf16. Run-to-run spread +-3-5 % (power cap).
- Harness (capsule, one run: `run_doc.sh`): inference in a CUDA graph (median of 5 x 10 replays); training = forward +
  `torch.autograd.grad` of every input (q, conditioning, all weights) without a graph, CUDA events. Tables in ms; × = ours
  against the fastest of the other columns.
- Columns: **PyTorch compiled** = `torch.compile` of the block's reference math in bf16 (dense banded mask SDPA; not the engine
  module); **Anthropic** = its fused `SWAAtomBlock` (esmfold2 `ef2_atom.py` fast tier, bf16 GEMMs; fp32 residual stream, modulation
  / RoPE per row: inference only, modulation precomputed); **engine v2.2** = the engine's `SWADiTBlock` (FA4 window attention +
  Triton + cuBLAS, the modulation computed per row inside); **H100 fused** = the H100 fused Triton block (team-gm `swa_fused_triton`
  algorithm, its Triton path: the sm_90a kernels do not build for sm_100); **ours** = the same block with the sm_100a kernels.
  cuEquivariance has no such block (—).
- Accuracy against the fp64 block (A8 S1024): ours output 2.58e-3, dq 2.78e-3, dc 4.60-4.62e-3, worst weight gradient
  7.4-7.6e-3; the H100 fused Triton path 2.58e-3 / 2.78e-3 / 4.91e-3 / 7.4-7.6e-3 (bf16 rounding points of the same algorithm).
  Anthropic 7.4e-4 (A4 S1024: its residual stream stays fp32).
- Charts: under each table, a length sweep at D128 (`python -m miniworld_engine.viz.measure_bars docs/gpus/b200/swa_atom_dit/swa_atom_dit.md`).

### Inference A5 · modulation in the block

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Anthropic | engine v2.2 | H100 fused | ours | × |
|---|---|---|---|---|---|---|---|
| (1024, 128) | 0.1037 | — | — | 0.0773 | 0.0529 | 0.0342 | 1.55 |
| (2048, 128) | 0.2617 | — | — | 0.1270 | 0.0695 | 0.0450 | 1.54 |
| (4096, 128) | 0.7093 | — | — | 0.2204 | 0.1158 | 0.0707 | 1.64 |

![Inference A5 · modulation in the block, length sweep at D128](figures/swa_atom_dit_inference_a5_modulation_in_the_block_length.png) <!-- measure_bars -->

### Inference A5 · modulation precomputed

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Anthropic | engine v2.2 | H100 fused | ours | × |
|---|---|---|---|---|---|---|---|
| (1024, 128) | — | — | 0.0442 | — | 0.0357 | 0.0320 | 1.12 |
| (2048, 128) | — | — | 0.0559 | — | 0.0483 | 0.0403 | 1.20 |
| (4096, 128) | — | — | 0.1068 | — | 0.0803 | 0.0606 | 1.33 |

![Inference A5 · modulation precomputed, length sweep at D128](figures/swa_atom_dit_inference_a5_modulation_precomputed_length.png) <!-- measure_bars -->

### Inference A1 · modulation in the block

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Anthropic | engine v2.2 | H100 fused | ours | × |
|---|---|---|---|---|---|---|---|
| (1024, 128) | 0.0959 | — | — | 0.0455 | 0.0430 | 0.0247 | 1.74 |
| (2048, 128) | 0.1618 | — | — | 0.0506 | 0.0480 | 0.0287 | 1.67 |
| (4096, 128) | 0.3302 | — | — | 0.0632 | 0.0625 | 0.0369 | 1.69 |

![Inference A1 · modulation in the block, length sweep at D128](figures/swa_atom_dit_inference_a1_modulation_in_the_block_length.png) <!-- measure_bars -->

### Inference A1 · modulation precomputed

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Anthropic | engine v2.2 | H100 fused | ours | × |
|---|---|---|---|---|---|---|---|
| (1024, 128) | — | — | 0.0231 | — | 0.0252 | 0.0204 | 1.13 |
| (2048, 128) | — | — | 0.0247 | — | 0.0258 | 0.0225 | 1.10 |
| (4096, 128) | — | — | 0.0335 | — | 0.0289 | 0.0275 | 1.05 |

![Inference A1 · modulation precomputed, length sweep at D128](figures/swa_atom_dit_inference_a1_modulation_precomputed_length.png) <!-- measure_bars -->

### Training A48 · forward + backward

| (Length, Dimension) | PyTorch compiled | cuEquivariance | Anthropic | engine v2.2 | H100 fused | ours | × |
|---|---|---|---|---|---|---|---|
| (4096, 128) | 19.028 | — | — | 5.009 | 2.565 | 1.248 | 2.06 |
| (8192, 128) | 83.761 | — | — | 9.662 | 4.815 | 2.270 | 2.12 |

![Training A48 · forward + backward, length sweep at D128](figures/swa_atom_dit_training_a48_forward_backward_length.png) <!-- measure_bars -->

(Anthropic's block has no backward.)

## Speed of light (measured, 2026-09-30)

`sol_measure.py` in the capsule, one run. Every kernel of a block step is recorded at the driver and replayed alone in a CUDA
graph for 2 s after a 1-s settle; time from the wall clock, energy from the NVML counter. Ceilings measured in the same run and
regime: idle 248 W, sustained maximum 985 W, HBM 6.78 TB/s, read 123 pJ/B, write 78 pJ/B, bf16 MMA 0.53 pJ/FLOP (cuBLAS 8192^3,
1.36 PFLOP/s sustained). A kernel's floor = max(bytes / HBM, FLOPs / 2.38 PF, exps / 4.65 T/s, (R e_r + W e_w + F e_f) / (985 - 248 W))
over its compulsory bytes read / written (R, W), FLOPs (F) and exponentials; **SoL = floor / measured time**; energy SoL = floor
dynamic energy / measured dynamic energy (energy minus idle x time).

### Training A48 (µs; SoL time / energy)

| kernel | S4096 µs | SoL | energy SoL | S8192 µs | SoL | energy SoL | 성능 확인 |
|---|---:|---:|---:|---:|---:|---:|---|
| T1 `mod_fwd` | 10.7 | 19.7 % | 85.1 % | 16.5 | 25.5 % | 84.4 % | △ |
| T2 `qkvg_fwd` (saves) | 144.8 | 44.8 % | 53.1 % | 280.1 | 46.3 % | 51.5 % | △ |
| T3 `attn_fwd3` | 84.5 | 47.4 % | 51.8 % | 161.3 | 49.7 % | 51.5 % | △ |
| T4 `ffn_fwd2` (saves) | 168.9 | 50.6 % | 62.9 % | 334.2 | 51.1 % | 60.5 % | △ |
| T5 `ffn_bwd_gate` | 112.6 | 72.8 % | 70.4 % | 218.7 | 75.0 % | 75.3 % | ✓ |
| T6 `ffn_bwd_dy` | 100.6 | 83.6 % | 84.5 % | 199.2 | 84.4 % | 85.7 % | ✓ |
| T7 `oproj_bwd` | 78.8 | 76.8 % | 78.0 % | 155.0 | 78.0 % | 78.2 % | ✓ |
| T8 `attn_dkvq` | 165.3 | 44.6 % | 46.5 % | 315.0 | 46.8 % | 48.4 % | △ |
| T9 `qkvg_bwd` | 137.3 | 82.4 % | 83.3 % | 267.1 | 84.7 % | 80.8 % | ✓ |
| T10 `mod_bwd` dc | 11.7 | 25.4 % | 73.2 % | 23.0 | 25.8 % | 79.6 % | △ |
| T10 `mod_bwd` dW | 15.8 | 18.1 % | 67.4 % | 22.5 | 25.4 % | 74.9 % | △ |
| weight GEMMs (cuBLAS, 4) | 162.5 | 109 % | 103 % | 272.8 | 129 % | 129 % | — |
| sum of the kernels | 1193.4 | 66.0 % | | 2265.3 | 69.6 % | | |

The block takes 1247.8 / 2269.9 µs (kernels back to back plus one fill). The model counts every byte from HBM, so the cuBLAS
GEMMs, which get part of their operands from L2, land above 100 %. Against one fully fused block (every activation read and
written once, weights, the essential FLOPs; `sol_swa.py`) the floor is 199.6 / 399.1 µs (energy): the block is at 16 / 18 % of
it -- the rest is the kernel boundaries' HBM traffic, not the kernels.

### Inference (µs; SoL time / energy)

| kernel | A5 S1024 | A5 S2048 | A5 S4096 | A1 S1024 | A1 S2048 | A1 S4096 |
|---|---|---|---|---|---|---|
| I1 `mod_fwd` | 5.3 · 11 % / 82 % | 5.5 · 20 % / 86 % | 10.7 · 20 % / 85 % | 5.2 · 11 % / 80 % | 5.5 · 20 % / 85 % | 10.7 · 20 % / 85 % |
| I2 `qkvg_fwd2` / `qkvg_fwd` | 9.0 · 16 % / 48 % | 11.0 · 27 % / 76 % | 19.1 · 31 % / 78 % | 6.0 · 8 % / 65 % | 6.2 · 15 % / 70 % | 8.3 · 22 % / 71 % |
| I3 `attn_fwd3` / `attn_fwd1h` | 6.8 · 15 % / 71 % | 11.1 · 19 % / 72 % | 14.7 · 28 % / 71 % | 4.7 · 4 % / 47 % | 4.7 · 9 % / 65 % | 5.0 · 17 % / 66 % |
| I4 `ffn_fwd2` | 10.5 · 19 % / 55 % | 14.9 · 27 % / 67 % | 23.7 · 34 % / 78 % | 6.1 · 12 % / 85 % | 6.5 · 22 % / 87 % | 9.4 · 29 % / 79 % |
| sum of the kernels | 31.6 · 16 % | 42.5 · 24 % | 68.2 · 29 % | 22.1 · 9 % | 22.9 · 17 % | 33.4 · 22 % |

At these sizes a kernel's compulsory work is 0.2-8 µs while one launch of a persistent tcgen05 kernel costs ~4-5 µs end to end
(TMEM allocation, weight prologue, one tile's latency chain), so time SoL stays low even where the energy is near its floor. The
remaining lever is fewer kernels (fusing I1-I4), not faster ones.

## What was tried and not kept

| attempt | result |
|---|---|
| FFN forward with one tile per warpgroup, two in flight (`ffn_fwd3`) | 170 -> 212 µs (A48 S4096): the per-tile thread phases double when a warpgroup owns a tile |
| attention forward with two threads per query row (16 softmax warps, TMEM `.16x32bx2`) | 85 -> 104 µs: more masked chunk pairs and warps; the softmax phases are issue-bound per SM sub-partition |
| sm_100 three-input max and paired fp32 FMA in the softmax | same values, 85 -> 92 µs |
| mask-free 8-column chunks in the separate dq / dkv kernels | dq 100 -> 108 µs |
| dQ into an fp32 buffer by TMA reduce-add | kernel 176.7 µs, but zeroing + converting the buffer cost 83 µs; replaced by the two-tile pairing |
| issuing the dQ TMEM load before the P / dS writes | 56 B of spills, 165 -> 175 µs |
| modulation backward: TMA reduce-add into a work buffer + last-CTA conversion; DSMEM pulls | 59 / 33 µs vs 28.7 with bulk pushes |
| cluster multicast of the weights (ffn_fwd2 CL = 2 / 4, qkvg_fwd2 CL = 2) | no gain (per-SM intake), CL = 4 broke wave fitting |
| fewer CTAs, two tiles each, for A1 qkvg | 9.1 -> 9.4 / 13.5 µs |

## Limits and next

- **Not in the engine.** Next: move the kernels and their dispatch into `kernels/swa_dit` + `modules/swa_dit`, build them in the pixi
  env with nvcc 13.1 (as measured here), add GPU tests; then b200.md's row says CUDA for the engine.
- Measured only at B = 1 and at the lengths above; A = 2 - 4 inference ran the kernels' accuracy check only.
- The modulation gradient accumulates with atomics (as the Triton path): not bitwise deterministic.
- Forward kernels and `attn_dkvq` sit near 45-50 % SoL (one tile at a time through serial thread phases); the block is at 16-18 %
  of the fully fused floor, so the large remaining gain is fewer kernel boundaries.
