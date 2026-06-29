# trimul forward — D (channel) sweep

Forward, B=1, bf16, NO mask, `d_pair == d_hidden == D` (square). Sweeps the channel
dim D at two representative L; the L-axis sweep lives in `compare_bench` (D=128).

**Regime (per `benchmark/BENCHMARKING.md` HARD RULE — all benchmarks run compiled,
no eager):** `pytorch` = `torch.compile(module, reduce-overhead)`, warmed,
steady-state. `nvidia(dtv1)` / `cuequivariance` / `ours_v4` = manual CUDA-graph (the
launch-overhead-free regime; torch.compile graph-breaks the cute/triton path and
dtv1's autograd Fns are `@torch.compiler.disable`'d, so CUDA-graph is the fair method
for them — memory `compile-vs-cudagraph-for-cute`). Each D runs in its OWN process
(a CUDA illegal-access poisons the whole context). H100 80GB, `dsweep_bench.py`.

`ours_v4` = triton LN_in → `trimul_inproj` (left+right, one gated GEMM, bdll) →
`torch.bmm` → triton fused back (LN_out + @Wp + gate + mul in one kernel).

## Result (ms/layer; lower is better)

| D | L | pytorch (compiled) | nvidia(dtv1) | cuequivariance | ours_v4 |
|----:|----:|-----------------:|-------------:|---------------:|--------:|
| 64 | 512 | 3.368 | 0.275 | 0.286 | **0.169** ✅ |
| 64 | 1024 | 6.515 | 1.088 | 1.121 | **0.654** ✅ |
| 128 | 512 | 2.895 | 0.576 | 0.470 | **0.352** ✅ |
| 128 | 1024 | 14.251 | 2.330 | 1.915 | **1.466** ✅ |
| 256 | 512 | 5.539 | **1.290** | 2.253 | ✗ regs/shared |
| 256 | 1024 | 26.772 | **5.643** | 9.260 | ✗ regs/shared |
| 512 | 512 | 7.506 | **3.392** | 9.316 | ✗ regs/shared |
| 512 | 1024 | 52.058 | **15.355** | 37.847 | ✗ regs/shared |

`✗ regs/shared` = the fused back kernel fails to **compile** at this D (see below).
cos vs pytorch (where ours runs): v4 = 0.99998 at every D∈{64,128} × L.

ours_v4 speedup (vs the best baseline at that point):
- D=64: **1.63×** vs dtv1 (L=512), 1.66× (L=1024).
- D=128: **1.34×** vs cuequiv (L=512), 1.31× (L=1024); 1.64× vs dtv1.

## Findings

1. **ours_v4 runs + wins at D∈{64,128}, but does not yet compile at D≥256.** This is
   a kernel-generality bug, NOT a perf ceiling:
   - `triton/back.py::_back_kernel` puts the WHOLE `N=K=D` in one block
     (`tl.arange(0,K)`, `tl.arange(0,N)`, full `(D,D)` weight tile,
     `tl.dot((BM,D)@(D,D))`). At D=256/512 ptxas reports
     **`Register allocation failed (count 255)`** and Triton reports
     **`out of resource: shared memory, Required 278528 > limit 232448`**.
   - Fix = tile the kernel over N and K (BN/BK blocks, accumulate) + add D to the
     autotune key. (D=64 earlier showed a transient IMA from a stale quack cache;
     with a fresh `QUACK_CACHE_DIR` it is correct and fastest.)
2. **Baseline D-scaling** (where ours can't yet compete):
   - **nvidia(dtv1) scales best** — wins decisively at D=256/512.
   - **cuequivariance degrades** with D: at D=512 it is SLOWER than *compiled*
     pytorch (9.3 vs 7.5 ms @ L=512). It only leads at D≤128.
   - compiled pytorch is the slowest small-D baseline but overtakes cuequiv by D=512.

## Next (to make ours competitive across D — a fix, not a limit)

- BN/BK-tile the triton back kernel so it compiles+runs at D∈{256,512}; re-key
  autotune on D. Then re-sweep ours_v4 vs **dtv1** at large D — dtv1 is the bar.
- v5 (cute layernorm_linear back) was omitted: its tile configs are N=128-specialized
  — same generality work applies.

## Reproduce

```bash
# bench (GPU): each D in its own process, fresh quack cache
srun --partition=h100 --account=cssb --qos=cssb_h100 --gres=gpu:h100:1 \
     --cpus-per-task=8 --mem=80G --time=00:50:00 \
     bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && \
       export QUACK_CACHE_DIR=<fresh> && \
       for D in 64 128 256 512; do PYTHONPATH=src pixi run --frozen python \
         src/miniworld_kernels/kernels/trimul_inproj/cute/dsweep_bench.py $D; done' \
     > .../benchmark/dsweep.out 2>&1
# render (CPU only, NO --gres): shared viz style
srun --partition=h100 --account=cssb --qos=cssb_h100 --cpus-per-task=4 --mem=16G --time=00:08:00 \
     bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && \
       PYTHONPATH=src pixi run --frozen python \
       src/miniworld_kernels/kernels/trimul_inproj/cute/dsweep_plot.py'
```

![latency](dsweep_latency.png)
