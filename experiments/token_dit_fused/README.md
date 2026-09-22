# Fused token DiT, inference

One sampling step of the token DiT -- 24 x `modules.dit.DiTBlock` (AF3 Alg. 23: d 768, cond 384, pair 128, 16 heads x 48,
transition n = 2), S = 5 samples, bf16 -- rebuilt around what is invariant across samples and steps. Forward only,
no QK-norm, no key mask (both are listed under "Not done").

    runner = FusedTokenDiT(blocks)            # packs every weight once
    bias = runner.hoist(pair)                 # once per sample(): every block's pair bias, head-major
    single = runner.step(single, cond, bias)  # once per solver step

## Result

node01 H100 80 GB, CUDA-graph replay of one step, per block (`bench.py`, `results/{bf16,fp32}-L{384,768}.json`). The
reference is the engine's PYTORCH block in IEEE fp32 on the same weights.

**bf16** (MiniWorld v2's diffusion dtype):

| | L384 | L768 | rel_rms vs fp32 |
|---|---:|---:|---:|
| engine `MINIWORLD` today | 273.2 us | 558.3 us | 1.1e-2 |
| Anthropic's parts, pair bias per step | 157.7 us | 302.0 us | 4.2e-3 |
| Anthropic's parts, pair bias hoisted | 132.4 us | 233.0 us | 4.2e-3 |
| this package v4 | 95.8 us | 200.4 us | 4.6e-3 |
| **this package v6** (v5 core + SwiGLU in the expand GEMM) | **82.6 us** | **176.3 us** | 4.4e-3 |
| vs engine / vs Anthropic hoisted | 2.85x / 1.38x | 2.79x / 1.16x | |

**fp32** (MiniWorld v1's recipe: fp32 activations, TF32 GEMMs; every fast path runs TF32, the reference does not):

| | L384 | L768 | rel_rms vs fp32 |
|---|---:|---:|---:|
| engine `MINIWORLD` today | 410.2 us | 1055.2 us | 1.9e-3 |
| Anthropic's parts, pair bias per step | 279.4 us | 596.5 us | 3.8e-4 |
| Anthropic's parts, pair bias hoisted | 222.9 us | 412.4 us | 3.8e-4 |
| **this package** | **180.7 us** | **383.9 us** | 3.6e-4 |
| vs engine / vs Anthropic hoisted | 2.27x / 1.23x | 2.75x / 1.07x | |

Once per sample, every block's pair bias: bf16 74.5 / 262.1 us and fp32 186.8 / 718.0 us here, against 515.0 / 1524.6
and 1285.4 / 4393.5 us for 24 of Anthropic's `ln_proj.pair_bias` calls. Both fast paths keep the residual stream in
fp32, which is why they sit 2.4x (bf16) and 5x (fp32) closer to the reference than the engine.

`FusedTokenDiT(blocks, dtype=torch.bfloat16 | torch.float32)`. In fp32 the GEMMs follow
`torch.backends.cuda.matmul.allow_tf32` (the caller's policy, as for any matmul) and the attention core's MMA precision
is `core_precision=` `"tf32"` (default, matching the TF32 GEMMs) / `"tf32x3"` / `"ieee"`.

## What each call hoists, and why it may

**Once per sample.** The pair representation carries no noise level (`DiffusionConditioning`: the time embedding is added
to the single track only), so every block's pair bias is the same at every step. All 24 blocks' `ln_pair` share the
LayerNorm statistics of a pair row, so with each block's LayerNorm weight folded into its 128 -> 16 projection the 24
biases are ONE GEMM, [L^2, 128] x [128, 384], over one pass of the pair, written head-major so the attention core reads
it with no permute.

**Once per step.** The single conditioning carries the noise level and changes every step, but all S samples share it,
so it is computed for L token rows, not S * L. Its LayerNorm statistics are shared by every block's AdaLN too, so with
each cond-LayerNorm weight folded, all 24 blocks' AdaLN scale / shift and output gates are two GEMMs per step.

**Per block.** Four cuBLAS GEMMs, one attention core, three row kernels:

| launch | work |
|---|---|
| `addmm` | q\|k\|v\|g = xa @ [Wq;Wk;Wv;Wg]^T + [bq;0;0;0], one [M, 4D] output that the core reads through strided views |
| `tdit.attn` | softmax(q k^T + bias) v, gated by sigmoid(g) in the epilogue, written over q |
| `mm` | out projection |
| `resgate_adaln_rows` | x += sigmoid(gl) * y (fp32, in place), then AdaLN for the transition -- one pass |
| `mm` | [Wa;Wb] expand |
| `swiglu_rows` | silu(a) * b |
| `mm` | squeeze |
| `resgate_adaln_rows` | x += sigmoid(gl) * y, then the NEXT block's AdaLN -- one pass |

## How it got here (L768 per block)

| step | us | what changed |
|---|---:|---|
| engine today | 557.9 | 29 launches; 41 % is the pair-bias path (LayerNorm 102 + skinny GEMM 64 + a permute copy 43) |
| v1 | 393.7 | hoisting + five launches with AdaLN fused into Triton GEMM prologues -- see "Negative results" |
| v2 | 213.6 | cuBLAS GEMMs + row kernels across module boundaries; engine attention core |
| v3 | 203.0 | own core: head dim 48 as 32 + 16 instead of padding to 64; gate in the epilogue; sigmoids moved into the consumers |
| v4 | 198.2 | sm_scale*log2(e) folded into Wq / bq and log2(e) into the pair-bias weights: no per-logit scale or division |
| v5 | 186.4 | attention core with the bias through TMA, no masks in the hot loop, key mask folded into the hoisted bias (bc8db071) |
| v6 | 176.3 | SwiGLU in the expand GEMM's epilogue (quack `gemm_act`, sm90): `ab` never reaches HBM; bf16 only |

v4 bf16 at L768, per block: attention core ~64 us, the four GEMMs 87 us (qkvg at 588 TFLOPS, cuBLAS-level), the two
`resgate_adaln_rows` 27 us and `swiglu_rows` 9.5 us (both at their HBM-byte floor), conditioning 5.5 us. In fp32 the
core (135 us, TF32) and the TF32 GEMMs (174 us) are 82 % of the block; the row passes matter even less there.

## Is this the best fusion? No -- it is the best under "cuBLAS GEMMs + separate kernels"

Per block at L768 bf16 the schedule moves 263 MB through HBM against 62.5 MB that is essential (the fp32 residual in and
out, the pair bias, the weights, the conditioning), and does 49.8 GFLOP of GEMM plus 9.1 GFLOP of attention MMA. Within
the current structure every non-GEMM pass is already at its byte floor, so what is left is either a different kernel
boundary or a better kernel:

| lever | kind | est. gain at L768 bf16 | what it needs |
|---|---|---:|---|
| attention core that overlaps softmax with wgmma (FA3-style, head dim 48) | kernel | ~30 us | CUDA / CuTe; core is 64 us against a ~25 us ALU floor |
| SwiGLU in the expand GEMM's epilogue | fusion | ~10 us | a Hopper GEMM with a custom epilogue at cuBLAS speed (a Triton one only ties cuBLAS + a row pass) |
| residual update in the Wo / squeeze GEMM epilogue | fusion | ~5-8 us | same; the AdaLN that follows needs whole-row statistics, so either 768-wide tiles (60 CTAs at M = 3840: half the SMs idle) or partial statistics plus an AdaLN prologue in the next GEMM, which only pays with a CUDA register-operand prologue |
| one persistent kernel per block (dataflow, RF3-`mk2` style) | fusion | up to ~90 us | overlaps the memory-bound passes with GEMM / core compute; the block's floor is then ~100-110 us (GEMM at 700 TFLOPS + core ALU) |

Not worth it or not possible: the out projection inside the attention epilogue (it needs every head of a row: a
cross-program reduction); keeping h on chip between expand and squeeze (the 64 x 768 fp32 output accumulator does not
fit, and splitting it recomputes the expand -- the engine reaches the same verdict for d >= 256).

## Negative results, measured

- **AdaLN in the GEMM prologue (v1) loses in Triton.** The transformed A has to go registers -> shared memory inside the
  k-loop, which breaks the software pipeline, and every N tile repeats the transform (24 times for q|k|v|g): K1 was
  139.5 us for 18.1 GFLOP (130 TFLOPS) against 39 us for cuBLAS + a separate pass. `step_v1` keeps it for the record.
- **Sample-fastest grid order in the core** (so the S readers of a bias tile share it through L2) measured 77.4 us against
  the engine's 75.5: the core is not bias-bandwidth-bound at these shapes.
- The engine's attention launcher (`_aa_fwd`) cannot take q / k / v as strided views: `_attn_fwd` computes one base
  offset from q's strides and applies it to k, v AND the output, and the launcher allocates a contiguous output --
  which is written out of bounds. `tdit.attn` writes over q instead, so all four share strides.

## v6 notes (SwiGLU-epilogue GEMM)

- `quack.gemm_act.gemm_act` is called directly: `quack.gemm_interface` does not import under this torch (its custom-op
  schema has a `RoundingMode` enum default the schema parser rejects).
- The expand weight is packed with gate/up rows interleaved (a0, b0, a1, b1, ...). The tile config is timed per M on the
  first call (`GATED_CFGS`); best measured 128x192, pingpong, cluster 2 at M = 3840 and cluster 1 at M = 1920.
- Alone, expand + SwiGLU: 42.5 -> 28.9 us at L768, 22.5 -> 17.8 at L384 (`bench_gated.py`), and closer to fp32
  (h is no longer rounded through a bf16 `ab`).
- fp32 keeps cuBLAS + `swiglu_rows`: quack rejects fp32 A ("a_dtype should be float16 or float8"), so fp32 v6 == v5.
- Measured and not taken: quack plain GEMMs for q|k|v|g, Wo and squeeze (`bench_gemms.py`) only beat cuBLAS for
  q|k|v|g at L768 (29.1 vs 32.8 us); Wo and squeeze tie or lose. A residual-gate GEMM epilogue moves the same HBM
  bytes as today unless the following AdaLN moves too, which needs a whole 768-column row per tile (quack caps N at 208).
- Triton 3.6 `tl.range(..., warp_specialize=True)` on the v2 core does not compile on sm90 (NVGPUWarpSpecialization pass).

## Traps

- Every kernel that updates a tensor in place (`resgate_adaln_rows`, `gate_resgate`, the core writing over q) must
  autotune with `restore_value=[...]`: the autotuner runs the kernel once per config, and without it the residual was
  updated once per candidate (rel_rms 2e3).
- `Transition` / `DiTBlock` zero-initialise `to_out`, `squeeze` and `to_bias`; a benchmark on fresh weights exercises
  none of the attention path. `bench.py` randomises every weight and scales the two output projections by 0.25 so a
  24-deep stream stays bounded.

## Not done

- QK-norm (MiniWorld's v1 token DiT sets `use_qk_norm: true`) and a key mask: the mask is sample-invariant and folds into
  the hoisted bias at no per-step cost; QK-norm would go into the core's q / k load.
- The two remaining levers, both larger: an attention core that overlaps softmax with wgmma for head dim 48 (the core is
  64 us against a ~25 us ALU floor), and GEMMs with the residual / SwiGLU in their epilogue at cuBLAS speed (would remove
  the 36.5 us of row passes).
- Backward: this is an inference path.

## Files

`tdit/runner.py` packing, hoisting, the per-step schedule (`step`, and `step_v1` for the record); `tdit/kernels.py` the
row kernels, the pair-bias GEMM, and the v1 prologue kernels; `tdit/attn.py` the core; `bench.py`; `prof.py` per-kernel
profile; `run.sh` / `run_any.sh` compute-node wrappers.
