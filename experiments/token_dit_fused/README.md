# Fused token DiT, inference

One sampling step of the token DiT -- 24 x `modules.dit.DiTBlock` (AF3 Alg. 23: d 768, cond 384, pair 128, 16 heads x 48,
transition n = 2), S = 5 samples, bf16 -- rebuilt around what is invariant across samples and steps. Forward only,
no QK-norm, no key mask (both are listed under "Not done").

    runner = FusedTokenDiT(blocks)            # packs every weight once
    bias = runner.hoist(pair)                 # once per sample(): every block's pair bias, head-major
    single = runner.step(single, cond, bias)  # once per solver step

## Result

node01 H100 80 GB, CUDA-graph replay of one step, per block (`bench.py`, `results/L{384,768}.json`):

| | L384 | L768 | rel_rms vs fp32 |
|---|---:|---:|---:|
| engine `MINIWORLD` today | 273.6 us | 557.9 us | 1.1e-2 |
| Anthropic's parts, pair bias per step | 157.8 us | 302.8 us | 4.2e-3 |
| Anthropic's parts, pair bias hoisted | 132.5 us | 233.0 us | 4.2e-3 |
| **this package (v4)** | **95.3 us** | **198.2 us** | 4.6e-3 |
| vs engine | 2.87x | 2.81x | |
| vs Anthropic hoisted | 1.39x | 1.18x | |

Once per sample, every block's pair bias: 75.0 / 264.6 us here, against 513.5 / 1524.4 us for 24 of Anthropic's
`ln_proj.pair_bias` calls. The reference is the engine's PYTORCH block in fp32 on the same weights; the fp32 residual
stream is why both fast paths are 2.4x closer to it than the engine's bf16 stream.

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

v4 at L768, per block: attention core ~64 us, the four GEMMs 87 us (qkvg at 588 TFLOPS, cuBLAS-level), the two
`resgate_adaln_rows` 27 us and `swiglu_rows` 9.5 us (both at their HBM-byte floor), conditioning 5.5 us.

## Negative results, measured

- **AdaLN in the GEMM prologue (v1) loses in Triton.** The transformed A has to go registers -> shared memory inside the
  k-loop, which breaks the software pipeline, and every N tile repeats the transform (24 times for q|k|v|g): K1 was
  139.5 us for 18.1 GFLOP (130 TFLOPS) against 39 us for cuBLAS + a separate pass. `step_v1` keeps it for the record.
- **Sample-fastest grid order in the core** (so the S readers of a bias tile share it through L2) measured 77.4 us against
  the engine's 75.5: the core is not bias-bandwidth-bound at these shapes.
- The engine's attention launcher (`_aa_fwd`) cannot take q / k / v as strided views: `_attn_fwd` computes one base
  offset from q's strides and applies it to k, v AND the output, and the launcher allocates a contiguous output --
  which is written out of bounds. `tdit.attn` writes over q instead, so all four share strides.

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
