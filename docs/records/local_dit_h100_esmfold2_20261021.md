# H100 local atom DiT, ESMFold2 variant, 2026-10-21

Measurement record (not edited after the fact). Branch `feat/local-dit-esmfold2` (worktree `~/miniworld-engine-scratch/wt-esm-atom`, on top of
`09e42b01` plus the rotary-table copy skip in `local_dit_h100._run_esm`), H100 80GB HBM3, driver 575.51.03, torch 2.13.0+cu129, CUDA 12.9.
`LocalDiTBlock(128, 128, 16, 4, n=2, **ESMFOLD2)`, random weights (`p / sqrt(fan_in)`, vectors 0.1 randn, q / k norm weights 1), A = 5, one
conditioning `[1, 1, N, 128]` expanded over the samples, all N atoms real, rotary angles `3 randn`. Script `~/miniworld-engine-scratch/esm-atom/bench.py`
(outside the repo): one block call captured in a CUDA graph and replayed 30 times (PyTorch 10), mean ms. "static" captures under
`kernels._capture.static_inputs()` after an eager call (the conditioning tables and the pair bias come from the hoist). PyTorch = the module
on `ImplementationType.PYTORCH`, fp32 matmuls without TF32 (torch default); compiled = `torch.compile(module, dynamic=False)`. bf16 = autocast
bf16 with fp32 parameters. rel err = |kernels - PyTorch| / |PyTorch| (Frobenius) in the dtype of the row.

## fp32 (TF32 kernels)

| N | A | kernels ms | kernels, static cond/pair ms | PyTorch ms | compiled PyTorch ms | vs PyTorch | vs compiled | rel err vs PyTorch |
|---|---|---|---|---|---|---|---|---|
| 1024 | 5 | 0.078 | 0.060 | 0.935 | 0.319 | 11.98x | 4.08x | 3.76e-04 |
| 2048 | 5 | 0.121 | 0.093 | 1.673 | 0.524 | 13.82x | 4.33x | 3.75e-04 |
| 4096 | 5 | 0.209 | 0.166 | 3.204 | 0.971 | 15.30x | 4.64x | 3.74e-04 |
| 8192 | 5 | 0.362 | 0.293 | 6.268 | 1.850 | 17.32x | 5.11x | 3.73e-04 |
| 16384 | 5 | 0.647 | 0.528 | 12.303 | 3.415 | 19.03x | 5.28x | 3.73e-04 |
| 24576 | 5 | 0.918 | 0.760 | 18.242 | 5.040 | 19.88x | 5.49x | 3.75e-04 |

## bf16

| N | A | kernels ms | kernels, static cond/pair ms | PyTorch ms | compiled PyTorch ms | vs PyTorch | vs compiled | rel err vs PyTorch |
|---|---|---|---|---|---|---|---|---|
| 1024 | 5 | 0.058 | 0.046 | 0.812 | 0.195 | 13.95x | 3.36x | 4.09e-03 |
| 2048 | 5 | 0.082 | 0.061 | 1.424 | 0.308 | 17.45x | 3.77x | 4.09e-03 |
| 4096 | 5 | 0.130 | 0.099 | 2.623 | 0.519 | 20.16x | 3.99x | 4.08e-03 |
| 8192 | 5 | 0.218 | 0.172 | 5.124 | 0.989 | 23.49x | 4.53x | 4.09e-03 |
| 16384 | 5 | 0.397 | 0.319 | 10.356 | 1.969 | 26.06x | 4.95x | 4.08e-03 |
| 24576 | 5 | 0.563 | 0.457 | 15.485 | 2.917 | 27.52x | 5.18x | 4.09e-03 |

## Accuracy against the fp64 module (valid rows; `accuracy.py`, commit `09e42b01`)

| A | N | real | fp32 kernels | bf16 kernels | bf16 PyTorch module | fp32 PyTorch module |
|---|---|---|---|---|---|---|
| 1 | 256 | 256 | 3.72e-04 | 5.43e-03 | 5.35e-03 | 2.67e-07 |
| 2 | 512 | 400 | 3.68e-04 | 5.38e-03 | 5.29e-03 | 2.63e-07 |
| 5 | 1024 | 1000 | 3.73e-04 | 5.39e-03 | 5.28e-03 | 2.95e-07 |
| 1 | 384 | 150 | 3.78e-04 | 5.42e-03 | 5.35e-03 | 3.03e-07 |
| 3 | 640 | 193 | 3.78e-04 | 5.38e-03 | 5.32e-03 | 2.99e-07 |
| 2 | 300 | 300 | 3.71e-04 | 5.39e-03 | 5.27e-03 | 2.96e-07 |
| 1 | 1536 | 1500 | 3.76e-04 | 5.39e-03 | 5.28e-03 | 2.96e-07 |
| 5 | 4096 | 4000 | 3.72e-04 | 5.35e-03 | 5.24e-03 | 2.95e-07 |
| 5 | 24576 | 24500 | 3.72e-04 | 5.35e-03 | 5.25e-03 | 2.96e-07 |

## Per kernel (torch profiler, eager after warm-up, `local_dit_hoist=False`, A = 5, us per call)

| kernel | bf16 N 8192 | fp32 N 8192 | bf16 N 24576 | fp32 N 24576 |
|---|---|---|---|---|
| post90_fwd (RMS) | 66.3 | 91.6 | 185.4 | 233.1 |
| pre90_fwd (RMS) | 43.8 | 97.5 | 121.7 | 264.7 |
| h_attn_esm / _tf32 | 40.0 | 67.5 | 102.1 | 161.9 |
| h_pair_bias_esm | 24.8 | 38.9 | 66.0 | 100.2 |
| qknorm_fwd_k (rotary) | 17.6 | 31.0 | 48.5 | 91.6 |
| cond90 (RMS) | 14.9 | 24.6 | 28.2 | 48.2 |
| torch: real-atom count, rotary table copies | 12.1 | 12.3 | 14.8 | 15.0 |
| total | 219.6 | 363.6 | 566.7 | 914.7 |

(The rotary-table copies were removed for unpadded N after this profile; the timing tables above are with that change.)

## Parity with FoldForge's dense block (`parity.py`, fp32, CPU)

FoldForge `DiffusionCrossAttTransformer(spec=ESMFOLD2)` against the engine's PyTorch path with its weights copied; the 192-key gathers from
`alphafold3.model.features.AtomCrossAtt.compute_features` on synthetic ragged layouts (tokens of 1-14 atoms), the rotary and the rank-window
mask from FoldForge's own `build_atom_rope` / `sliding_window_mask`. rel = Frobenius relative error over the real atoms.

| weights | real atoms (padded), A | rel(FoldForge, engine) | rel(FoldForge fp32, engine fp64) | rel(engine fp32, engine fp64) |
|---|---|---|---|---|
| random | 150 (224) .. 24 500 (24 576), A 1 / 5, 9 cases | 2.9e-7 - 3.1e-7 | 4.8e-7 - 4.9e-7 | 4.8e-7 - 5.0e-7 |
| trained, `evoformer_conditioning.atom_transformer_encoder` | same 9 cases | 6.4e-7 - 7.4e-7 | 1.2e-6 - 1.3e-6 | 1.2e-6 - 1.3e-6 |
| trained, `diffusion_head.atom_cross_att_encoder.atom_transformer_encoder` | same 9 cases | 2.1e-7 - 2.6e-7 | 4.1e-7 - 5.9e-7 | 3.9e-7 - 6.0e-7 |
| trained, `diffusion_head.atom_cross_att_decoder.atom_transformer_decoder` | same 9 cases | 2.8e-7 - 3.5e-7 | 4.9e-7 - 5.8e-7 | 5.0e-7 - 5.8e-7 |

Trained weights: the converted biohub/ESMFold2 checkpoint (`esmfold2.bin.zst`, FoldForge `import_jax_weights_`).
