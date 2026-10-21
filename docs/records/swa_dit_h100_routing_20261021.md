# SWA atom DiT on H100: the stage routing regression and its fix (2026-10-21)

H100 (sm_90), torch 2.13 + cu129, one `SWADiTBlock` (d_atom 128, 4 heads, window 128, SwiGLU hidden 256). Bench rows:
`benchmarks/runners/bench.py target=swa_dit level=module n_layers=1 +shared_cond=true compile=false` (atoms = `seq_len` x 8,
`allow_tf32: true`, so fp32 runs at matmul precision "high"); inference in a CUDA graph, A = 5 (`n_augment`); training A = 48,
`cudagraph=manual`. Anthropic = the ESMFold2 kit's atom driver `ef2_atom` (`gemm="bf16"`: bf16 GEMM operands, fp32 accumulation and
residual stream, whatever the input dtype), inference only.

## What regressed

The 2026-10-14 page numbers (bf16 block ahead of Anthropic 2.8 - 3.6x) were the composed hand-CUDA stages. Two later changes moved
H100 calls off them:

1. 2c745caf (10-13) chose the native stages' window attention as `tiled_attention if fusion else native_attention`, where `fusion` is
   the B200-style row fusion (`_native_b200_fusion`), off for every call whose samples share their conditioning (every diffusion
   sampling step) and every fp32 call. The untiled `native_attention`: 1520 us against the tiled kernel's 41 us at A 5, 8192 atoms.
2. 08e5e4bc (10-18) made `native_h100_enabled` true for every H100 call (so no cell mixes Triton in), which sent every call to the
   native stages; the composed stages had become all hand-CUDA in 02073e40, so this only cost speed.

Inside the native stages the hoisted modulation ran as an fp32 FFMA GEMM (49 us shared, 228 us per-sample at A 5, 8192 atoms) and the
generic `swa_native::rows` stages run at 30 - 50 % of HBM bandwidth.

## Fix

- `_tiled_serves`: the tiled attention serves every bf16-head call with S % 64 == 0 (and a bf16 dO in the backward).
- `native_h100_stages(q)` picks the stages; `native_h100_enabled` again says only whether H100 serves the call. bf16 -> composed;
  fp32 under TF32 -> composed (`swa_fp32.cu`); fp32 under "highest", global attention, `swa_dit_h100_native=True`, or a composed
  extension that does not load -> native.
- The native modulation runs bf16 operands on the tensor cores for a bf16 block.
- A bf16 modulation reaching the native stages (global attention under a bf16 call) is widened to fp32 there; the composed bf16
  backward returns fp32 weight gradients (the op contract; the autograd node casts them).

No Triton kernel launches on any of these calls (launch hook, A 1 - 48 x 256 - 16 384 atoms, ragged, shared and per-sample
conditioning, inference and training, bf16 / fp32 / fp32 master weights under bf16 autocast).

## Inference, A = 5, shared conditioning (ms)

| precision | atoms | PyTorch | before (2.3.6) | after | Anthropic `ef2_atom` | after vs Anthropic |
|---|---|---|---|---|---|---|
| bf16 | 1024 | 0.419 | 0.264 | 0.051 | 0.078 | 1.54x |
| bf16 | 4096 | 3.28 | - | 0.108 | 0.250 | 2.32x |
| bf16 | 8192 | 11.48 | 1.663 | 0.196 | 0.442 | 2.26x |
| bf16 | 24 576 | 95.6 | 4.835 | 0.539 | 1.269 | 2.35x |
| fp32 (TF32) | 1024 | 0.535 | 0.102 | 0.074 | 0.068 | 0.92x |
| fp32 (TF32) | 4096 | 5.16 | 0.320 | 0.222 | 0.202 | 0.91x |
| fp32 (TF32) | 8192 | 19.48 | 0.595 | 0.428 | 0.356 | 0.83x |
| fp32 (TF32) | 24 576 | 168.2 | 1.793 | 1.199 | 1.025 | 0.86x |

"before (2.3.6)" bf16 is the main-branch module at 93a64414; its fp32 column is the tiled-attention fix alone on the native stages.
The fp32 rows are not like for like: Anthropic's kernels round the GEMM operands to bf16 for an fp32 call; ours run tf32 wgmma with
the FFN weights split hi + lo. At equal GEMM precision (our bf16 rows) the engine is 1.5 - 2.4x ahead.

## Training fwd + bwd, A = 48 (ms)

| precision | atoms | PyTorch | native stages | after (composed) |
|---|---|---|---|---|
| bf16 | 1024 | 5.82 | - | 0.655 |
| bf16 | 2048 | 16.87 | 2.82 | 1.176 |
| bf16 | 4096 | 54.8 | - | 2.190 |
| fp32 (TF32) | 1024 | 9.51 | 2.054 | 1.254 |
| fp32 (TF32) | 2048 | 30.2 | 3.830 | 2.401 |
| fp32 (TF32) | 4096 | 109.1 | 7.362 | 4.605 |

Anthropic's kit has no backward. fp32 native against composed in one process (eager, A 48, `forward_hoisted(...).sum().backward()`):
2.36 / 3.86 / 7.39 against 1.37 / 2.56 / 4.82 ms at 1024 / 2048 / 4096 atoms; output relative error against the "highest" fp32
reference 5.4e-4 (native) and 4.9e-4 (composed).

## Shared against per-sample conditioning, bf16 inference (ms)

A 5, 1024 atoms 0.047 / 0.055; A 5, 8192 atoms 0.191 / 0.276; A 1, 8192 atoms 0.069 / 0.068.

## Mixed precision

fp32 parameters and fp32 x under bf16 autocast run the fp32 stages (no refusal, no Triton): relative error against the fp32 reference
3.3e-4, against 4.9e-3 for PyTorch under the same autocast. An earlier probe that read "engine == PyTorch, rel 0" had the zero-initialised
adaLN projection, under which the block is the identity.

## Tests (H100)

`test_swa_dit_fused_gpu.py` 97 passed / 1 skipped, `test_input_feature_embedder_gpu.py` 17, `test_op_contracts_gpu.py` 5,
`test_architecture_no_triton_gpu.py` 28, `test_swa_fp32_highest_gpu.py` 6, `test_swa_dit_kernels.py` 14, `test_swa_fa4_attention_gpu.py` 9,
`test_swa_gate_out_gpu.py` 8.

Not kept: splitting the native `rows` reduce stages over sample chunks (no gain on the native path; stash on
`fix/swa-h100-regression`).
