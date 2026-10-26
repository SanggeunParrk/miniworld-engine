# ESMFold2 local atom block training and SWA atom DiT: performance-ladder fixes on H100 (2026-10-27)

Branch `perf/final-esm-swa` (from `integ/ladder-all2` 1361277f). The ladder rule is hand CUDA < portable (Triton) < PyTorch (best of CUDA-graph eager and
`torch.compile`), ratio lower rung / upper rung < 1, clear wins <= 0.95. In [the integration run](performance_ladder_20261025.md) (commit e35029a1) two
families had open rows on H100:

* **LocalDiT ESMFold2 atom block, bf16 TRAINING** (`LocalDiTBlock(**ESMFOLD2)`, rotary q / k, 192-key rank windows): `miniworld` ran the PyTorch reference
  (CUDA 12.663 / 25.774 ms = PyTorch 12.656 / 25.746), 1.93x / 2.01x slower than the portable kernels. Inference rows were already OK.
* **SWA DiT (atom), all six rows**: the portable rung could not be measured (`RuntimeError: native H100 SWA rejects force_portable_kernels`).

## What changed

* `kernels/swa_dit/dispatch.py` `native_h100_enabled`: `settings.force_portable_kernels=True` (the A/B switch of every other op) now turns the native H100 stages off,
  so the fused block runs `kernels/swa_dit/triton/*` (modulation, qkvg, window attention, out-projection + FFN and their backward) on H100. `swa_dit_h100_native=True`
  still forces the native stages. Global attention (no portable rung by design) is unchanged.
* SWA CUDA INFERENCE (the CUDA rung only; the portable rung is untouched): the hoisted modulation `silu(c) Wmod^T` of an inference call is made once per
  (conditioning tensor, version, adaLN weight) and read from `_capture.hoisted_table` afterwards (a sampler's steps eager; a captured step under `static_inputs()`),
  and the fp32 block's weights are prepared (tf32-rounded / split / K-permuted, `swa_fp32.cu::prep`) once per weight version (`_capture.param_forms`, new optional
  `wpre` argument of `swa_fp32.cu::fwd`). Both left the H100 CUDA step before this: silu + cast + cuBLAS GEMM (16 us fp32, 22 us bf16 at 4096 atoms, A = 5) and
  `prep_kernel` (3.6 us) were 8 - 19 % of the step. Under `torch.compile` the hoist is skipped (it reads storage addresses) and the opaque op stays in the graph.
  `_capture.hoisted_table` is the former `integrations.local_dit._hoisted` moved so both users share it.
* `integrations/local_dit_h100.py` (`_esm_train*`) + `kernels/augmented_attention/cuda/sm100_atom_local/lattn90_esm.cu` + `qknorm_local.cu` +
  `kernels/dit_variants/cuda/train_rows.cu`: the hand-CUDA bf16 TRAINING step of the ESMFold2 block, forward and backward, as two opaque ops (`local_dit_h100_esm_train_fwd` /
  `_bwd`) in one `autograd.Function`, plus one for the windowed pair bias (`local_dit_h100_esm_pair_bias_fwd` / `_bwd`). It reuses the DiT-variant atom training block:
  the row kernels of `train_rows.cu` (LN + modulation, residual + AdaLN, SwiGLU, gate backward, AdaLN backward with the table gradient as fixed-order partial rows) got an ESMFold2
  switch (`mode` bit 1: RMSNorm statistics, raw output gates; the other callers pass the old flag unchanged), cuBLAS for every GEMM, and the
  attention of the inference path extended with a backward: `h_attn_esm` also writes the row LSE, `h_attn_esm_dq` (dQ + the pair-bias gradient as per-chunk partials), `h_attn_esm_dkv`
  (dK, dV), `h_pair_bias_esm_bwd` (dbias -> dz, dgamma, dWb), `qknorm_rope_bwd` (norm backward behind the transposed rotary). The conditioning tables
  (`silu(c) [Ws1; Wb1; Wskv; Wbkv; Wso; Ws2; Wb2; Wst]^T`, scales + 1, gates raw) and the pair bias are made outside the block by autograd ops, so their gradients chain through
  the module's own layers. Contract: bf16 single / cond / pair (fp32 or bf16 parameters, autocast), the conditioning shared by the samples (expanded view, or one sample), any A, any N
  (padded to 128), real-first key mask; `settings.local_dit_esm_train_cuda` switches it off. fp32 ESMFold2 training and a per-sample conditioning keep the composed PyTorch path (open).

## Ladder (H100 80GB HBM3, torch 2.13 cu129, compile + CUDA graph, median of 3 repeats; `ladder.py run --group '^(local_esm|swa_)'`, Slurm 36290)

Ratio c/p = CUDA / portable, p/t = portable / PyTorch (best of compiled and eager graph). "before" = commit e35029a1 ladder run (`out2_J2`).

| row | PyTorch ms | portable ms | CUDA ms before | CUDA ms after | c/p before | c/p after | p/t before | p/t after | verdict |
|---|---|---|---|---|---|---|---|---|---|
| ESMFold2 atom, training bf16 L512 | 12.664 | 6.558 | 12.663 | 3.266 | 1.93 | **0.498** | 0.519 | 0.518 | OK |
| ESMFold2 atom, training bf16 L1024 | 25.738 | 12.798 | 25.774 | 6.213 | 2.01 | **0.485** | 0.497 | 0.497 | OK |
| ESMFold2 atom, inference bf16 L512 | 0.514 | 0.242 | 0.094 | 0.094 | 0.391 | 0.389 | 0.469 | 0.470 | OK (unchanged) |
| ESMFold2 atom, inference bf16 L1024 | 0.978 | 0.423 | 0.167 | 0.167 | 0.394 | 0.395 | 0.432 | 0.433 | OK (unchanged) |
| SWA DiT, inference bf16 L512 | 1.212 | 0.114 | 0.106 | 0.091 | n.a. | **0.805** | n.a. | 0.094 | OK |
| SWA DiT, inference bf16 L1024 | 4.216 | 0.181 | 0.197 | 0.170 | n.a. | **0.943** | n.a. | 0.043 | OK (tight) |
| SWA DiT, training bf16 L512 | 31.161 | 2.900 | 2.205 | 2.195 | n.a. | **0.757** | n.a. | 0.093 | OK |
| SWA DiT, training bf16 L1024 | 119.0 | 5.719 | 4.206 | 4.218 | n.a. | **0.737** | n.a. | 0.048 | OK |
| SWA DiT, inference fp32 L512 | 3.324 | 0.223 | 0.222 | 0.203 | n.a. | **0.912** | n.a. | 0.067 | OK |
| SWA DiT, training fp32 L512 | 86.677 | 7.182 | 4.610 | 4.591 | n.a. | **0.639** | n.a. | 0.083 | OK |

Measured with the switch fix alone (first run of this branch, before the two hoists; its other inference rows were discarded by the harness because the tree was edited during the run): fp32 L512
CUDA 0.221 against portable 0.225 ms (ratio 0.981, a tie within noise) and bf16 L512 portable 0.114 against the base run's CUDA 0.106 ms (0.93). Profile of the fp32 step then (A = 5, 4096 atoms, graph
replay, us): `oproj_ffn_fwd` 139.5, `qkvg_fwd` 44.0, `attn_fwd` 18.2, modulation GEMM 10.4 + silu / casts 5.7, `prep_kernel` 3.6 -- against the Triton path's 141.7 / 46.5 / 22.7 / 10.4 + 2.8. The hand-CUDA
fp32 kernels are ~5 % ahead of Triton's; the gain came from the per-call prologue. After the hoists: 138.3 / 44.7 / 17.9, step 0.2025 ms against the portable 0.2206 ms (bf16 L512: 0.0893 against 0.1109).

## Parity (ESMFold2 training, vs the fp64 module)

Output, dsingle, dcond (the shared conditioning), dpair and every parameter gradient, relative Frobenius error against the fp64 PyTorch module on the valid atoms, for the engine and for the
module's own bf16-autocast run (same weights and inputs); the bound is engine <= 1.5x module. Probe `~/miniworld-engine-scratch/fin-esmswa/probe/probe_esm.py` (outside the repo), A / N / real atoms:

| case | out | dsingle | dcond | dpair | worst parameter gradient (engine / module) |
|---|---|---|---|---|---|
| 5 / 512 / 512 | 4.83e-3 / 5.18e-3 | 5.63e-3 / 6.28e-3 | 8.36e-3 / 9.36e-3 | 1.06e-2 / 1.19e-2 | `norm_query.weight` 1.41e-2 / 1.29e-2 (1.10x) |
| 3 / 1024 / 1000 | 4.82e-3 / 5.19e-3 | 5.67e-3 / 6.31e-3 | 8.33e-3 / 9.43e-3 | 1.07e-2 / 1.19e-2 | `to_query.bias` 1.26e-2 / 1.21e-2 (1.04x) |
| 2 / 640 / 193 | 4.82e-3 / 5.14e-3 | 5.79e-3 / 6.37e-3 | 8.50e-3 / 9.56e-3 | 1.07e-2 / 1.16e-2 | `norm_query.weight` 1.68e-2 / 1.30e-2 (1.29x) |
| 1 / 300 / 300 | 4.86e-3 / 5.24e-3 | 5.67e-3 / 6.34e-3 | 8.35e-3 / 9.41e-3 | 1.07e-2 / 1.19e-2 | `norm_key.weight` 1.29e-2 / 1.22e-2 (1.05x) |

The run is bitwise repeatable (test). Tests: `tests/gpu/sm90/test_h100_local_dit_esmfold2_gpu.py` (`test_training_*`), the existing SWA, local-DiT and DiT-variant files unchanged.

## Open

* ESMFold2 fp32 (TF32) training and a conditioning of its own per sample still run the composed PyTorch path (not in the ladder; the TF32 attention kernel has no backward).
* SWA bf16 inference at L1024 is 0.943, inside the bar but not a wide margin: the Triton row kernels run within a few percent of the hand-CUDA ones at these sizes (fp32 L512 table above), so a
  further gain needs a faster row-stage kernel, not more glue.
* The `swa_dit_mod_cuda` kernels (`swa_mod_wg.cu`) are shadowed on H100 by `native_h100_enabled` (the native modulation takes the call first); the hoist above removes the modulation from
  the inference step instead. Training still runs the native cuBLAS modulation.
