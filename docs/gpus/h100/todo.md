# H100 (sm90) — module status and TODO

Snapshot 2026-09-30, engine `main` 2719096e, torch 2.13 / triton 3.7. The required shapes are the
module rows of `kernels/registry/registry_module.csv` (MiniWorld, AF3, Protenix, OpenDDE, ESMFold2
configurations); lengths are L128–768 for token/pair/MSA streams and 1024–8192 atoms. A module is
**complete** when every required shape runs a native (CUDA) path on H100 (training at L384/768,
inference at every L). ✓ = CUDA on H100, ✗ = Triton or PyTorch. [dispatch.md](dispatch.md) predates
some of this (it still calls single-direction training D128-only).

## Complete

| Module | Required shapes | H100 |
|---|---|---|
| Bidirectional TriMul | d_pair = hidden = 128 (MiniWorld), bf16 | ✓ inference L128–768, training L384/768 (also D64 / 256 / 384 / 512 training); checked by a kernel profile of MiniWorld v1.3 blocks |

## Partial

| Module | Required shape | H100 | Note |
|---|---|---|---|
| **Transition** | n=4: pair D128 / 256 / 384, MSA D64 / 128, single D384 | ✓ | bf16, rows a multiple of 128 (`fused_sm90a` D128, `fused_wide_sm90a` D64 / 256 / 384 / 512) |
| | **n=2**: pair D64 / 128 / 256 (template, conditioning), single D384 / 768 (conditioning) | ✗ | the CUDA kernels hard-code hidden = 4·D |
| | fp32 (any D) | ✗ | `guard_dtype` sends it to the PyTorch reference |
| | rows not a multiple of 128 | ✗ | no ragged-tail path |
| **Single-direction TriMul** | D64 / 128 / 256 / 384, bf16 | ✓ training L384/768, ✓ inference (native tile table, uni D512 wide) | single-direction rows are not in h100.md yet |
| | same widths, fp32 | ✗ | Triton |
| | D512 training | ✗ | |
| **Bidirectional TriMul inference, wide** | D256 / 384 / 512 at L128 / 256 / 512 / 640 | ✗ | CUDA only at L384 / 768 |
| **OuterProductMean** | MSA 64 / pair 128 / hidden 32 (MiniWorld) | ✓ | contract: L % 64 == 0, MSA depth % 256 == 0, no interchain mask |
| | MSA 128 / pair 256 / hidden 32 (Protenix-v2, ESMFold2), MSA 128 / pair 384 (OpenDDE) | ✗ | |
| **MSAPairWeightedAveraging** | MSA 64 / pair 128, 8 heads × 32 (MiniWorld) | ✓ | contract: L % 128 == 0, even MSA depth |
| | MSA 128 / pair 256, 8 heads (ESMFold2, Protenix-v2 8 × 8), MSA 128 / pair 384, 8 × 8 (OpenDDE) | ✗ | |
| **SWA atom DiT** (`kernels/swa_dit`) | d_atom 128, 4 heads, window 128, SwiGLU 256; bf16 and fp32 | bf16 ✓ partly (3 CUDA stages + Triton), fp32 ✗ (Triton) | GPU suite 33 passed / 1 skipped, 17/17 registry checkers (2026-09-30); no H100 cache; opt-in in team-gm, so MiniWorld v1.3 does not use it yet |
| **TriangleAttention** | MiniWorld n_head 4 / hidden 128 | ✗ core (Triton) + ✓ pieces (bias / LN backward CUDA) | template / Protenix / OpenDDE head counts: Triton; B200 has a whole-module CUDA path, H100 does not |
| **Token DiT** | single 768 / cond 384 / pair 128, 16 heads; bf16 and fp32; QK-norm on / off | ✓ inference (shared or per-sample cond) and training, CUDA + cuBLAS, branch `h100/token-dit` ([token_dit/token_dit.md](token_dit/token_dit.md)) | upstream capture-scoped pack / bias caches retained |
| **AugmentedAttention** | atom 128 / pair 16 / 4 heads; token 768 / 384 / 128 / 16 heads; token 768 / 768 / 256 | bf16 sm_90 core (opt-in) | fp32 default path is Triton |

## Not implemented on H100

| Module | Required shapes | Current path |
|---|---|---|
| ConditionedTransition | atom 128 / cond 128; token 768 / cond 384; token 768 / cond 768 (bf16, fp32) | Triton |
| AdaptiveLayerNorm | atom 128 / 128; token 768 / 384 (bf16, fp32) | Triton (+ CUDA LayerNorm backward) |
| AttentionPairBias | single 384 / pair 128 / 8 heads | Triton |
| SWA atom attention (per-op) | d 128 / 4 heads | FlashAttention-4 (external) + Triton `qk_norm_rope` |
| Primitives: LayerNorm, LayerNormLinear, gated linear, SwiGLU FFN, projected attention, RMSNorm modulation | many widths (see the csv) | Triton; LayerNorm backward is CUDA for bf16 128 ≤ N ≤ 512 |
| MPNN families | — | native paths target A6000 bf16; nothing H100-specific |

## TODO

1. **MiniWorld v1.3 atom path (largest measured gap).** The input feature embedder runs the atom
   transformer in fp32. `rmsnorm_adamod_fwd_triton` has no H100 cache; the config served without
   runtime tuning (BLOCK 64/64/64) takes 1322 µs where the best in its grid takes 59 µs
   (M=4096, N=K=128), about 7.6 ms per microbatch. Build H100 caches for the fp32 atom shapes of
   `rmsnorm_adamod_{fwd,bwd}`, `qk_norm_rope_{fwd,bwd}`, `rmsnorm_{fwd,bwd}`, `layernorm_fwd_saveact`,
   `layernorm_bwd_atomic`, `transition_expand_swiglu`, `transition_bwd_swiglu_recompute` (the ten
   Triton misses of the v1.3 run), or give them a first config that is fast at these shapes.
2. **H100 autotune caches in general.** The committed H100 Triton caches were built under
   torch 2.10 / triton 3.6 and are rejected by the environment identity under torch 2.13; nine more are
   stale by kernel source (`dev cache-status --gpu H100`). Rebuild on this toolchain.
3. **SWA atom DiT.**
   - The speed depends on autotuning, and there is no H100 cache. fwd+bwd of 3 blocks at N=1, S=4096,
     CUDA graph, cluster H100 (2026-09-30):

     | | runtime tuning off (miss cap 1) | on (cap 24) |
     |---|---|---|
     | fp32 per-op path (v1.3 today) | 9.57 ms | 1.84 ms |
     | fp32 fused | 7.58 ms | 0.83 ms |
     | bf16 fused | 0.86 ms | 0.78 ms |

     With tuning off, `_swa_ffn_bwd_fp32_kernel` alone takes 6.6 ms (2.2 ms per block).
   - **Every stage in CUDA on H100** (in progress 2026-09-30): the fp32 block is Triton only, and
     six of the nine bf16 stages are Triton (window attention forward and backward, out-projection
     backward, qkvg backward). Native kernels with fixed tiles also remove the autotune dependence
     above; the Triton kernels stay as the non-H100 fallback and the accuracy reference.
   - Decide with the maintainer whether v1.3 turns the fused path on (fp32 keeps the numerics of the
     per-op path; bf16 moves the atom outputs by about 1%).
   - fp32 CUDA variant.
4. **Transition.** n=2 CUDA path (pair D64 / 128 / 256, single D384 / 768); a ragged-tail path for
   rows that are not a multiple of 128; decide whether fp32 needs a native path.
5. **OPM / PWA** at the wider MSA / pair widths (MSA 128 with pair 256 / 384).
6. **TriMul.**
   - Bidirectional wide inference at L128 / 256 / 512 / 640.
   - Single-direction training D512 and fp32; add the single-direction rows to h100.md and the
     figures to `trimul/trimul.md`.
   - Remove the remaining D128 weight-pack kernel (P1).
   - Wide single-direction training towards 90% of speed of light: at D384 L768 the step is at 86.5%
     of the design SoL; worst kernels are the front (69.6%), the output-LN backward (65.3%) and
     contract_gp (88%).
   - Measurements tables in `trimul/trimul.md` (PyTorch compiled / cuEquivariance / Anthropic / ours / ×).
7. **TriangleAttention** whole-module CUDA on H100, all required head counts.
8. **Token DiT**, fully CUDA (cuBLAS allowed for the GEMMs, as on B200), inference and training, bf16 and fp32,
   QK-norm on and off: done on branch `h100/token-dit` (2026-09-30, [token_dit/token_dit.md](token_dit/token_dit.md)).
   Optimised 2026-10-01 (SwiGLU-epilogue expand GEMM, og from the attention forward, fused conditioning rows): inference
   bf16 0.178 ms at L768, training bf16 7.28 ms. Left: bf16 `attn_dqb` dbias (L2 reduce traffic), the fp32 inference v^T
   GEMM, the fp32 training bias transposes; re-run the token DiT tests on B200 (shared code changed).
   FP32 backward layout candidate (2026-10-05): 1.050–1.070x whole-block F+B on L384/768, A4/6; numerical, compile and
   replay checks passed. New-kernel sanitizers and whole-block racecheck/synccheck passed; a baseline-common memcheck
   API error remains. Explicit experiment only, default dispatch unchanged; see the token DiT experiment section.
   - Inference retains upstream's shared and per-sample conditioning. Shared cond `[1, 1, L, 384]` or stride 0 runs
     the AdaLN / gate projections on L rows; per-sample cond `[S, 1, L, 384]` runs them on S L rows.
     Both use the H100 CUDA rows and attention core. Weight packs / pair bias retain upstream's capture scopes.
     Training always sees per-sample cond.
9. **ConditionedTransition, AdaptiveLayerNorm, AttentionPairBias, AugmentedAttention fp32**: native
   H100 paths (diffusion-side modules; not in the v1.3 trunk).
10. `test_op_contracts_gpu`: `rmsnorm_adamod_bwd` fails opcheck (`test_aot_dispatch_static`, eager vs
    AOT outputs differ); pre-existing on main before swa_dit.
11. Refresh [dispatch.md](dispatch.md) to the current single-direction and D64 training paths.
