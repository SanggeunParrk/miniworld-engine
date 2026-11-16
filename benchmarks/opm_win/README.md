# OuterProductMean at large MSA depth (H100): engine `cuda` vs the Anthropic kits

Rows 55 / 56 / 57 / 58 / 63 of `requirements_module.csv` (d_msa 64, d_pair 128, d_hidden 32; 55 AF3 order, 56 + projection biases,
57 Boltz order, 58 Boltz order + biases, 63 AF3 order + `count_offset` 1e-3), inference, bf16 autocast, L 256 / 384 / 512 / 768,
MSA depth S 1024 / 2048 / 8192 / 16384. Row 56 has no Anthropic kernel (`_anthropic_route`: projection biases with the division after
the projection).


## Verdict

Under the fair protocol (below), after this branch, at S 8192 / 16384:

- **Rows 57 / 58 (Boltz order): clearly ahead** of the boltz2 kit: lead 1.08-1.22 cool, 1.09-1.26 sustained (before: 1.01-1.18).
- **Rows 55 / 63 (AF3 order): ahead but not by 3 % everywhere.** Cool lead 1.00-1.05 (row 55: 1.02-1.05; row 63: 1.00-1.04;
  before: 0.95-1.00); sustained 1.01-1.11. Cells at or above 1.03 vs Anthropic: 22 / 32 cool, 29 / 32 sustained. The weakest cells are
  S 16384 cool (1.02 at row 55, 0.997-1.02 at row 63) and row 63 at L 256.
- **Row 56 (no Anthropic kernel) vs the engine's Triton backend: even** (cool 0.996-1.025, sustained 0.975-1.16).
- S <= 2048 (the fused kernel, unchanged): after / before speed-up over PyTorch mean 1.005 cool (min 0.981), 1.003 sustained (min 0.973):
  no regression beyond the protocol's spread.

Why not more: the af3_torch kit, the engine's Triton backend and this route all spend 90-93 % of the call in the same cuBLAS product
(`nvjet` 192x192 / 320x128, bitwise the same in every cuBLASLt heuristic algorithm we tried). That product runs at the tensor peak per
cycle and is power-bound; what is left to win is the prologue + epilogue + count (now 1.33 + 0.47 + 0.04 ms of 26.3 at L 768 / S 16384;
the DRAM floor for those bytes is ~1.4 ms). A lead of 3 % over a cuBLAS-based competitor at S 16384 needs a product faster than cuBLAS;
the fused kernel (one pass, no O round trip) is 8-10 % slower per call than cuBLAS + epilogue (82-88 % of the tensor peak per cycle).

## Protocol (`fair.py`)

The H100 is power-capped (700 W): after ~50-80 ms of load the SM clock falls from 1.98 GHz to 1.3-1.5 GHz; job-to-job drift is +-1-2 %.
Every number is a ratio to the PyTorch backend measured in the same process:

- each backend is captured once in a CUDA graph (the requirements construction of `measure.config`, `unpadded_masks`);
- per repetition (3, the order rotated): **cool** = 2 s idle, then a burst of `k` calls (`k * t <= 40 ms`, 256 MB L2 flush before each);
  **sustained** = 2 s idle, 1 s of back-to-back replays, then `k2` timed calls; NVML clock / power / energy around the sustained burst;
- cool = median of the reps, sustained = mean of the reps;
- Anthropic runs in the kits' CUDA-13 stack (`STACK=kit job.sh`: torch 2.13 + cu130), the engine backends in the engine env (torch 2.13
  + cu129); **lead = (PyTorch_kit / Anthropic) / (PyTorch_engine / CUDA)** speed-up ratio (> 1: the engine is faster).

The first version (`raw/base_*.jsonl`, `raw/after2_engine.jsonl`) had no idle before the sustained phase: in a cyclic rotation the backend
after PyTorch (the hottest, 60 ms calls) always inherited its heat, which was the `cuda` backend in the engine stack and Anthropic in half
of the kit reps; it showed `cuda` 3-5 % behind Triton sustained with the same kernels. All tables here are the fixed protocol (`raw/v3_*`):
kit `v3_kit_r*`, before `v3_before_r*` (opm-bigs `0ac874471` from a `git archive` snapshot, `jobb.sh`), after `v3_after_r*` with the
N 256 / 384 re-runs `v3_after_lt_r*` and the S <= 2048 re-runs `v3_after_small_r*` (after the fused kernel was restored). `reportjob.sh`
merges them and runs `report.py` (table + graphs).

## Where the time goes (L 768, S 16384, ms cool, `kprof.py` / `kbench.py`)

| piece | before | after | Triton backend | change |
|---|---|---|---|---|
| prologue (LN + projections + mask) | 1.53 | 1.33 | 1.08 | setmaxnreg rebalance (112 B stack frame gone) + tiles in groups of 64 tokens |
| product | fused kernel 26.2-26.9 (incl. projection) | cuBLAS 24.3 | cuBLAS 24.5 | route: cuBLAS + epilogue past depth 4096 |
| pair counts | (in the fused kernel) | 0.044 (fp32-output GEMM of the bf16 mask) | 0.05 | count kernel took 0.14 |
| epilogue | (fused) | 0.47 on the count table | 0.53 | the bit-row popcount took 0.85 |

At N 256 / 384 the product takes the cuBLASLt heuristic algorithm 1 instead of 0 (`opm_train._LT_ALGO`; `ltab.py` in-graph A/B:
2 % faster cool and 1-4 % sustained at N 256; bitwise the same product).

## Findings (counter / ablation evidence)

- **Fused kernel vs cuBLAS**: in the sustained regime the fused kernel takes 42.3 M cycles per call at S 16384 / L 768 at every clock
  (`wt-opm-bound` stamps); removing its TMA loads keeps the cycles (41.7 M) and only raises the clock: the main loop is issue-bound
  (82-88 % of the tensor peak per cycle), cuBLAS is at ~99 % (766 TF-equiv at 1.43 GHz).
- **Route sweep** (`sweep.sh`, `raw/sweep1.jsonl`, L 256-768): cuBLAS + epilogue is 2-6 % faster cool and 3-11 % sustained at S 8192 /
  16384, even at S 4096, slower at S <= 2048 -> `_product_route_deep` (S > 4096), also in `composed.biased_wide`.
- **Rejected for the fused kernel** (kept out of the code): 2-CTA / 2x2 clusters with TMA multicast of the shared A / B tile (bitwise the
  same; 1-2 % faster cool at S 16384, 6-10 % at S 2048, slower sustained; the first version's `mbarrier.arrive.release.cluster` on the
  peer was 4.4x slower, CUTLASS's plain `shared::cluster` arrive fixed it), two-stage commit groups (15 % slower); BK 64 does not fit.
  The template carrying them cost 2-5 % cool at S <= 2048 even with clusters off (old-vs-new A/B), so the opm-bigs kernel is back.
- **Prologue** (ncu, S 8192): DRAM 63 -> 76 % of peak, 775 -> 640 us under ncu clocks; groups of 128 / 256 / 768 tokens or 4 / 6 stages:
  no further gain; an s-major output + NT product (the Triton / Anthropic layout; `rejected_smajor_prologue.patch`): bitwise the same,
  no faster (whole-chain A/B). Bitwise the same A2 / BT / mask bits as before (cross-build check in `kbench.py`).
- **Epilogue**: the per-CTA popcount of S / 8-byte mask rows grew with S (0.63 ms at S 8192, 0.85 at 16384 for the same O); on a
  divisor table 0.47. Rejected: popcount spread over the K loop (0.87), 256-pair tiles x 3 / 4 stages (0.87 / 0.83), 128 x 4 (1.22),
  the gathered-TMA variants (0.86 / 0.97).
- **cuBLASLt algorithms** (`ltbench.py`): the 8-12 heuristic algorithms are bitwise the same here; at N 512 / 768 index 0 is the fastest.

## Accuracy

`tests/gpu/test_opm_deep_product_route.py` (+ `test_opm_count_offset.py`): the five row forms at (L, S) = (256, 8192), (272, 6000:
ragged depth, N % 32 != 0), (384, 16384), (256, 8192: row-chunked product), (256, 2048: the fused kernel), padded tokens, 97 masked MSA
rows, the pair residual; `max|x - gold| / max|gold|` vs the float64 module <= max(err_pytorch, 8e-3), run twice bitwise: 37 / 37 pass.
Worst err / tol 0.81 (e.g. AF3 order L 384 S 16384: 3.92e-3 vs PyTorch 6.10e-3). The product route changes outputs past depth 4096 (O
rounded to bf16 as one product instead of per c pass): `backend_versions["cuda:h100_sm90"]` 3.

## Results

Speed-up over PyTorch per stack and leads: [`results/table.md`](results/table.md) (all rows, L, S; cool and sustained), graphs
`results/speedup_cool.png`, `results/speedup_sus.png` (engine gold, Anthropic orange hatched, PyTorch grey, Triton blue dotted).

Large depth (lead vs Anthropic before -> after | after vs Triton), cool and sustained:

| row | L | S | cool lead | cool vs Triton | sustained lead | sustained vs Triton |
|---|---|---|---|---|---|---|
| 55 | 256 | 8192 | 0.999 → 1.036 | 1.010 | 1.044 → 1.022 | 1.004 |
| 55 | 256 | 16384 | 0.978 → 1.021 | 1.001 | 0.997 → 1.091 | 1.009 |
| 55 | 384 | 8192 | 0.984 → 1.046 | 1.014 | 1.021 → 1.110 | 1.027 |
| 55 | 384 | 16384 | 0.958 → 1.021 | 1.000 | 0.958 → 1.071 | 1.029 |
| 55 | 512 | 8192 | 0.995 → 1.048 | 1.001 | 1.018 → 1.083 | 0.998 |
| 55 | 512 | 16384 | 0.958 → 1.027 | 0.995 | 0.957 → 1.032 | 1.005 |
| 55 | 768 | 8192 | 0.977 → 1.044 | 1.001 | 0.971 → 1.046 | 1.027 |
| 55 | 768 | 16384 | 0.953 → 1.024 | 0.999 | 1.031 → 1.110 | 1.131 |
| 56 | 256 | 8192 | - → - | 1.016 | - → - | 0.994 |
| 56 | 256 | 16384 | - → - | 1.011 | - → - | 0.975 |
| 56 | 384 | 8192 | - → - | 1.025 | - → - | 1.011 |
| 56 | 384 | 16384 | - → - | 1.010 | - → - | 1.030 |
| 56 | 512 | 8192 | - → - | 1.008 | - → - | 1.012 |
| 56 | 512 | 16384 | - → - | 1.000 | - → - | 1.046 |
| 56 | 768 | 8192 | - → - | 1.007 | - → - | 1.054 |
| 56 | 768 | 16384 | - → - | 0.996 | - → - | 1.158 |
| 57 | 256 | 8192 | 1.165 → 1.222 | 1.011 | 1.127 → 1.221 | 1.006 |
| 57 | 256 | 16384 | 1.072 → 1.139 | 1.006 | 1.120 → 1.155 | 0.989 |
| 57 | 384 | 8192 | 1.114 → 1.198 | 1.017 | 1.190 → 1.258 | 1.031 |
| 57 | 384 | 16384 | 1.028 → 1.108 | 1.001 | 1.037 → 1.158 | 0.999 |
| 57 | 512 | 8192 | 1.113 → 1.170 | 1.005 | 1.128 → 1.223 | 1.011 |
| 57 | 512 | 16384 | 1.010 → 1.094 | 1.002 | 1.028 → 1.135 | 1.015 |
| 57 | 768 | 8192 | 1.120 → 1.170 | 1.005 | 1.128 → 1.236 | 1.048 |
| 57 | 768 | 16384 | 1.013 → 1.087 | 0.997 | 1.041 → 1.150 | 1.191 |
| 58 | 256 | 8192 | 1.181 → 1.179 | 0.998 | 1.108 → 1.216 | 1.050 |
| 58 | 256 | 16384 | 1.097 → 1.113 | 0.989 | 1.105 → 1.090 | 0.966 |
| 58 | 384 | 8192 | 1.114 → 1.171 | 1.011 | 1.125 → 1.224 | 1.041 |
| 58 | 384 | 16384 | 1.038 → 1.102 | 1.005 | 1.035 → 1.133 | 1.001 |
| 58 | 512 | 8192 | 1.110 → 1.151 | 1.008 | 1.141 → 1.192 | 0.993 |
| 58 | 512 | 16384 | 1.019 → 1.081 | 1.002 | 1.044 → 1.153 | 1.052 |
| 58 | 768 | 8192 | 1.128 → 1.172 | 1.010 | 1.111 → 1.221 | 1.051 |
| 58 | 768 | 16384 | 1.009 → 1.083 | 0.999 | 1.036 → 1.139 | 1.088 |
| 63 | 256 | 8192 | 0.992 → 1.003 | 0.988 | 1.015 → 1.012 | 0.981 |
| 63 | 256 | 16384 | 0.973 → 0.997 | 0.983 | 0.988 → 1.034 | 0.975 |
| 63 | 384 | 8192 | 0.979 → 1.027 | 1.004 | 0.994 → 1.054 | 0.998 |
| 63 | 384 | 16384 | 0.949 → 1.007 | 0.994 | 0.955 → 1.058 | 1.014 |
| 63 | 512 | 8192 | 0.991 → 1.042 | 1.000 | 1.044 → 1.103 | 0.987 |
| 63 | 512 | 16384 | 0.956 → 1.019 | 0.993 | 0.976 → 1.024 | 0.977 |
| 63 | 768 | 8192 | 0.970 → 1.031 | 1.001 | 0.990 → 1.052 | 1.105 |
| 63 | 768 | 16384 | 0.945 → 1.016 | 1.000 | 1.012 → 1.047 | 1.021 |
