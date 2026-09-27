# A100 TriMul in Triton — the CUDA kernels' fusion algorithm

`trimul_triton.py` is the fusion of the A100 CUDA kernels (`../a100_trimul_fwd`, itself the sm_90 algorithm) in Triton 3.6. It covers
single and bidirectional TriMul, inference and training. It is faster than the engine's existing Triton path at every measured shape.

## Pipeline

| stage | kernel | what it fuses |
|---|---|---|
| forward | `_k1` | LN_in (two-pass fp32) → x_n; (g, p) = x_n · [Wg \| Wp] in one dot per output tile (gate / projection columns interleaved); `sigmoid(g) · p · m_i m_j` → channel-major planes a \| b; saves the LN_in (mean, rstd) |
| | cuBLAS `bmm` | contraction: outgoing `A Bᵀ`, incoming `Aᵀ B` (bidirectional: one half each) |
| | `_k3` | LN_out(X) · W_oᵀ × `sigmoid(x_n · W_ogᵀ)` (× ds[j]) + z, output width tiled by BN; x_n from K1's statistics. Training also saves (μ_o, r_o, μ_i, r_i) and x_n |
| backward | `_b1e` | o, g recomputed per output tile from the saved statistics and x_n; d_o, d_g, A_o = d_o r_o, fold vectors Σ A_o μ_o and Σ d_o |
| | `_b1d` | dx̂ = (d_o · W_o) diag g_o as a GEMM, the LN_out backward in its epilogue → dX planes |
| | cuBLAS, side stream | W_o gradient (split-K `G = A_oᵀXᵀ` + rank-1 LayerNorm folds) and `dW_og = d_gᵀ x_n`, overlapped with the contraction backward |
| | cuBLAS `bmm` | contraction backward: dA, dB of every channel |
| | `_src` | input-side source as a GEMM grid (token tile × SW plane channels): (g, p) recomputed, dg / dp → the operand buffer |
| | cuBLAS, side stream | `dW_in = dgpᵀ x_n` (split-K), overlapped with the consumer |
| | `_con` | consumer: dx_n = [dg \| dp \| d_g] · [W_in ; W_og], LN_in backward + residual → dz, dγ / dβ |
| optional | `_b7j` (`TT_JOINT=1`) | joint input side: source + consumer programs meeting in an L2 ring (release / acquire flags) |

The joint kernel needs every program resident at once. Triton has no cooperative launch, so co-residency rests on an occupancy estimate
(`_joint_capacity`). It measured slower than the split path, so the split path is the default.

## Configuration

- **Autotune.** Every kernel ships the repo-width ladder (`W_*`: BM 16–128, tile widths 16–128, `num_warps 1 2 4 8`,
  `num_stages` up to 6–10). Configs impossible for the live shape (a tile wider than, or not dividing, the dimension it tiles) are pruned
  per call, and the autotuner drops configs the compiler rejects. `TT_AUTOTUNE=dev` selects the small development sets (`D_*`).
- **Other knobs.** `TT_OVERLAP` (side streams), `TT_JOINT`, `JOINT` (the joint kernel's static configuration), `GEMM_TK` (split-K chunking).
- **Pair width D** comes from the module and must be a power of two. Plane offsets are 64-bit (channel × T passes 2³¹ beyond L = 2048).
  Every kernel masks the token tail.
- **Slurm.** Jobs use one Triton cache directory each (`TRITON_CACHE_DIR`); concurrent jobs sharing the cache produced intermittent
  `CompilationError`s.

## Results

Same job (gpu08), CUDA-graph replay. The engine is measured in the same job; the Anthropic, PyTorch and cuEq columns are the baseline's
numbers (`../a100_anthropic_baseline`). Ratio = baseline time ÷ our time (> 1: ours faster).

Inference (ms):

| module | L | implementation | time | SoL | Anthropic | engine | PyTorch | cuEq |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| bidirectional | 384 | CUDA | 0.624 | 64% | 1.27× | 1.39× | 4.10× | 2.11× |
| bidirectional | 384 | Triton | 0.726 | 55% | 1.09× | 1.19× | 3.53× | 1.81× |
| bidirectional | 768 | CUDA | 2.970 | 67% | 1.15× | 1.33× | 6.85× | 1.84× |
| bidirectional | 768 | Triton | 3.363 | 59% | 1.02× | 1.17× | 6.05× | 1.62× |
| single | 384 | CUDA | 0.372 | 60% | 1.09× | 1.46× | 3.68× | 1.67× |
| single | 384 | Triton | 0.433 | 51% | 0.94× | 1.26× | 3.17× | 1.43× |
| single | 768 | CUDA | 1.701 | 64% | 1.04× | 1.35× | 6.33× | 1.55× |
| single | 768 | Triton | 1.998 | 54% | 0.89× | 1.15× | 5.38× | 1.32× |

Training, fwd + bwd (ms):

| module | L | implementation | time | SoL | Anthropic | engine | PyTorch | cuEq |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| bidirectional | 384 | CUDA | 2.596 | 63% | — (no backward) | 1.26× | 2.84× | 1.71× |
| bidirectional | 384 | Triton | 3.031 | 54% | — (no backward) | 1.08× | 2.43× | 1.46× |
| bidirectional | 768 | CUDA | 11.713 | 66% | — (no backward) | 1.22× | 5.08× | 1.56× |
| bidirectional | 768 | Triton | 13.447 | 58% | — (no backward) | 1.06× | 4.42× | 1.36× |
| single | 384 | CUDA | 1.640 | 61% | — (no backward) | 1.23× | 2.59× | 1.59× |
| single | 384 | Triton | 1.851 | 54% | — (no backward) | 1.09× | 2.29× | 1.41× |
| single | 768 | CUDA | 6.866 | 67% | — (no backward) | 1.23× | 4.65× | 1.55× |
| single | 768 | Triton | 7.799 | 59% | — (no backward) | 1.08× | 4.09× | 1.36× |

Accuracy vs the fp32 module, every shape including tails (L72 / 100) and L2064:
- inference rel 2.4–2.5e-3 (the bf16 module: 3.1–3.2e-3);
- dz 3.3e-3;
- worst parameter gradient 5.1–5.2e-3 (the bf16 module: 7.5–7.8e-3).

## How it got here (what mattered in Triton)

1. **The first port (joint + v1 kernels) was correct but slower than the engine.** Register pressure dominated: [64 × CH] fp32 tiles and
   whole weight operands made the first B1 compile to 32 registers with 3.5 KB of spills (67 ms).
2. **Audit fixes.** The token tail was dropped (T % 128 ≠ 0 gave NaN dz at L72) and 32-bit plane offsets overflowed past L = 2048. Both
   are fixed everywhere.
3. **The engine's Triton tiling.**
   - Weights pre-transposed to [K, N], so a dot's B operand needs no transpose.
   - The output width tiled by BN.
   - The LN output cast to bf16 once.
   - Pipelined N loops; autotune.
4. **B1 split in two** (elementwise + GEMM-epilogue). The one-kernel version held a [BM, CH] accumulator beside the y tile and ran at 6 ms.
5. **Source as a GEMM grid, with dW as a split-K GEMM on a side stream.** Two separate dots, no `tl.join` / `split` interleave: those went
   through shared memory (`mio_throttle` 2.9, 11.6 M bank conflicts).
6. **sigmoid as `tanh.approx.f32`** (inline PTX): `tl.sigmoid` compiles to an IEEE division.

## Files

| file | purpose |
|---|---|
| `trimul_triton.py` | kernels + autograd function |
| `bench_triton.py` | accuracy vs the fp32 module; CUDA-graph timing of Triton / engine / CUDA in one process |
| `check_edges.py` | token-tail and large-L (64-bit offset) checks |
| `prof_triton.py`, `prof_engine.py` | per-kernel CUPTI times of this path and of the engine's Triton path |
| `run_triton_op.py`, `ncu_stalls.sh`, `ncu_lines.sh` | Nsight Compute drivers |
| `job.sbatch`, `job_ab.sbatch`, `job_wide.sbatch` | Slurm wrappers (A/B over env settings; the shipped wide autotune spaces) |
| `logs/final_bench_v3_same_job.log` | the run behind the result tables (Triton / engine / CUDA in one job) |

**Wide spaces verified (2026-09-26, `logs/wide-53416.log`).** One cold run of the shipped spaces takes 1 h 48 min for single L384
(inference + training), since every kernel is tuned over its full ladder. It finds configs the development sets miss. At single L384:
- inference **394.5 µs** vs 432 µs with the dev sets (engine 524.2 µs, Anthropic best 407 µs);
- training **1.804 ms** vs 1.848 ms (engine 1.997 ms).

The tables above are the dev-set numbers.
