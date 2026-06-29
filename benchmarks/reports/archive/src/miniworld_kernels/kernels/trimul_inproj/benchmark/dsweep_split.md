# trimul back — single fused (v4) vs SPLIT (v6) — D sweep

The single fused back (`triton/back.py`) does LN_out + @Wp + gate + mul in ONE kernel
= TWO GEMMs (proj + gate) in one program → blows registers/shared at D≥256. The
**split** back runs the two GEMMs as two kernels, each lighter:

- **v4 (single)**: `trimul_back_triton` — one fused kernel.
- **v6 (split)**: `trimul_back_split` — ① cute `layernorm_linear_cute_fused`
  (LN_out + @Wp) + ② triton `gate_elem` (gate + mul).

Forward, B=1, bf16, no mask, d_pair=d_hidden=D (square). pytorch = `torch.compile`,
others = manual CUDA-graph (BENCHMARKING.md HARD RULE; no eager). Each D in its own
process. H100 80GB. `dsweep_bench.py` (with the ours_v6 column).

## Result (ms/layer; lower is better)

| D | L | pytorch | nvidia(dtv1) | cuequiv | ours_v4 (single) | ours_v6 (split) |
|----:|----:|--------:|------:|--------:|--------:|--------:|
| 64 | 512 | 3.363 | 0.276 | 0.286 | **0.169** | ✗ LNL tile_n=64 |
| 64 | 1024 | 6.514 | 1.090 | 1.131 | **0.658** | ✗ |
| 128 | 512 | 4.191 | 0.571 | 0.470 | **0.351** | 0.361 |
| 128 | 1024 | 13.132 | 2.384 | 1.933 | 1.475 | **1.449** |
| 256 | 512 | 4.337 | 1.290 | 2.252 | ✗ regs/shared | **0.880** |
| 256 | 1024 | 26.700 | 5.390 | 9.160 | ✗ | **3.643** |
| 512 | 512 | 7.588 | **3.382** | 9.323 | ✗ | ✗ regs/shared |
| 512 | 1024 | 51.669 | **14.654** | 37.675 | ✗ | ✗ |

cos(v6, pytorch) = 0.99998 where it runs (D∈{128,256}).

![v4 vs v6](dsweep_split_latency.png)

## Findings

- **The split unlocks D=256** — the single fused back fails there (ptxas
  `Register allocation failed (255)` + `shared mem Required 278528 > limit 232448`),
  but the split runs correctly (cos 0.99998) and is the **fastest** option at D=256:
  **0.880 ms vs dtv1 1.290 (1.47×), cuequiv 2.252 (2.56×)** @ L=512.
- **At D=128 split ≈ single** (0.351 vs 0.361 @ L=512; v6 slightly ahead at L=1024,
  1.449 vs 1.475). The extra `proj` HBM round-trip of the split is offset by the
  tuned cute LayerNormLinear. So no regression from splitting.
- **D=512 still fails for both** — v6's GateElem and the cute LNL both load full N at
  256/512; D=512 needs N-tiling (the same tiling work tracked for the single kernel).
- **D=64 v6 fails** on the cute LayerNormLinear `tile_n=64` config (config bug, not
  fundamental); v4 single is fine at D=64.

This split back is exactly what makes **bidirectional trimul** viable (its back runs
over 2·d_hidden = 256 channels) — see `benchmark/bidir.md`.

## Next

- N-tile GateElem + the cute LNL config so D=512 (and bidirectional d_pair=512) work.
- Fix the cute LNL `tile_n=64` path for D=64.

## Reproduce

```bash
srun --partition=h100 --account=cssb --qos=cssb_h100 --gres=gpu:h100:1 \
     --cpus-per-task=8 --mem=80G --time=00:50:00 \
     bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && export QUACK_CACHE_DIR=<fresh> && \
       for D in 64 128 256 512; do PYTHONPATH=src pixi run --frozen python \
         src/miniworld_kernels/kernels/trimul_inproj/cute/dsweep_bench.py $D; done' \
     > .../benchmark/dsweep_split.out 2>&1
srun ... --cpus-per-task=4 --mem=16G --time=00:08:00 \
     bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && \
       PYTHONPATH=src pixi run --frozen python \
       src/miniworld_kernels/kernels/trimul_inproj/cute/dsweep_plot.py'
```
