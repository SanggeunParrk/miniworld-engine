# trimul forward — L sweep (v6 split back focus), D=128 & D=256

How does speed scale with sequence length L? Forward, B=1, bf16, no mask, square
(d_pair=d_hidden=D). pytorch = `torch.compile`, others = manual CUDA-graph (HARD RULE;
no eager). `dsweep_bench.py` with `DSWEEP_LS=128,256,384,512,768,1024`, D∈{128,256}.
ours_v6 = SPLIT back (cute LayerNormLinear + triton GateElem); v4 single fails at
D=256. cos(v6, pytorch)=0.99998 everywhere.

## ms/layer

**D=128**

| L | ours_v6 | ours_v4 | dtv1 | cuequiv | pytorch | v6 vs dtv1 |
|----:|--------:|--------:|-----:|--------:|--------:|:--:|
| 128 | 0.059 | 0.037 | 0.049 | 0.053 | 1.606 | 0.83× |
| 256 | 0.118 | 0.102 | 0.162 | 0.137 | 1.734 | 1.37× |
| 384 | 0.219 | 0.206 | 0.334 | 0.266 | 2.161 | 1.53× |
| 512 | 0.361 | 0.354 | 0.574 | 0.470 | 4.188 | 1.59× |
| 768 | 0.780 | 0.791 | 1.276 | 1.040 | 7.786 | 1.64× |
| 1024 | 1.454 | 1.478 | 2.381 | 1.930 | 14.450 | 1.64× |

**D=256** (v4 single won't compile — split only)

| L | ours_v6 | dtv1 | cuequiv | pytorch | v6 vs dtv1 |
|----:|--------:|-----:|--------:|--------:|:--:|
| 128 | 0.090 | 0.093 | 0.110 | 3.124 | 1.03× |
| 256 | 0.256 | 0.350 | 0.543 | 3.443 | 1.37× |
| 384 | 0.516 | 0.741 | 1.287 | 2.977 | 1.44× |
| 512 | 0.895 | 1.286 | 2.266 | 4.496 | 1.44× |
| 768 | 2.009 | 2.941 | 5.103 | 14.225 | 1.46× |
| 1024 | 3.713 | 5.420 | 9.243 | 26.676 | 1.46× |

![L-sweep](lsweep_latency.png)

## Findings — v6 (split back) L-scaling

- **Scaling shape: sub-quadratic at small L → ~L² at large L.** v6 per-2×-L factor
  (D=256): 128→256 = 2.8×, 256→512 = 3.5×, 512→1024 = **4.15×**. Small L is launch/
  setup-bound (fixed overhead dominates); at large L it approaches the O(L²·D) HBM-
  traffic limit (~4× per 2×). The O(D·L³) bmm is not yet dominant (would be 8×).
- **v6's lead over dtv1 GROWS with L, then plateaus** — D=128: 0.83× @L=128 (v6
  *loses*, our fixed overhead) → 1.64× @L≥768. D=256: ~1.0× @L=128 → ~1.46× @L≥768.
  The win is biggest where it matters (large L) and saturates, not unbounded.
- **Small-L caveat**: at L=128 v6 carries fixed launch/setup overhead (3 kernels +
  cute compile-config) and is ~tied or slightly behind dtv1. v4 single is cheaper at
  tiny L (D=128, L=128: 0.037 vs v6 0.059) — fewer launches.
- **cuequiv scales worst**, especially at D=256 (2.0× slower than v6 @L=256 → 2.5×
  @L=1024). dtv1 is the real competitor; v6 stays ~1.4–1.6× ahead across L.

So: v6 is the fastest option from L≈256 up, and its advantage is largest at the long
sequences that dominate wall-clock — exactly the useful regime.
