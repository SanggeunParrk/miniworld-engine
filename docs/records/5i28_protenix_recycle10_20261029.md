# The 5I28 benchmark at 10 trunk passes: Protenix v1 / v2, FoldForge against the Anthropic kits (2026-10-29)

This is the protocol of `benchmarks/5i28.md` (real MSA, 4 distinct real templates on both sides, warm forward), with one change:
the trunk runs 10 times instead of 4.
- Kits: `-c 10`; every kit log shows `cycle=10`, 4 real templates and `N_msa 6790`.
- FoldForge: `recycles: 10`; the run JSON shows `trunk_passes: 10`, `msa: 6790`, 0 compiles.

The engine, FoldForge tree and kit modes are as in `5i28_benchmark_all_models_20261029.md` (Protenix v1 / v2: kit fast, timer
`PHASE total_s` / `fwd_s`). Each model was run at n = 1 on both sides first, then n = 2 - 16 (Slurm jobs 39248, 39249).
Harness: `~/miniworld-engine-scratch/bench5i28/r10/`.

## Result (seconds, warm medians; kit / FoldForge, kit/FF in bold)

| model | n=1 | 2 | 4 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| Protenix v1 | 2.162 / 0.493 | 2.266 / 0.797 | 3.326 / 1.986 | 11.123 / 7.344 | 26.206 / 18.681 | 60.314 / 37.494 |
| kit/FF | **4.39** | **2.84** | **1.67** | **1.51** | **1.40** | **1.61** |
| Protenix v2 | 0.967 / 0.586 | 1.519 / 1.156 | 3.906 / 3.459 | 15.188 / 14.303 | 37.761 / 37.683 | 84.223 / 73.191 |
| kit/FF | **1.65** | **1.31** | **1.13** | **1.06** | **1.00** | **1.15** |

At 4 trunk passes the Protenix-v2 ratios were 1.43 / 1.33 / 1.17 / 1.10 / 1.03 / 1.26. Ten passes give the trunk more weight, and
the Protenix-v2 trunk is at parity with the kit from 1024 tokens on. Its triangle attention (d_pair 256, 8 heads) runs the engine's
generic H100 path, because the fused kernels serve only d_pair 128 / 4 heads. At 1536 tokens (n=12) the two are equal.
