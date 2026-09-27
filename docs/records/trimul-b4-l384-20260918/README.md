# B4: length-dependent dispatch and stronger baseline correction

## Correction to the earlier B4 report

The earlier 1.187x L768 result compared native TMA against the **currently
selected Triton persistent path**, not the fastest measured Triton implementation.
Forcing the updated Triton atomic path is faster than that native TMA path at
L768. The earlier number remains a valid comparison of those two implementations,
but does **not** establish a 15% win over the strongest measured Triton baseline.
No production dispatch or native cache has been changed by this investigation.

## Why L384 and L768 differ

For B1, D=hidden=128, the output LayerNorm backward has M=L^2 rows and N=256.
`layernorm_linear/triton/mmajor_bwd.py` selects atomic below M=300,000, and the
canonical persistent branch for these larger N256 inputs. Thus L384 (147,456
rows) uses atomic and L768 (589,824 rows) uses persistent. `out_ln_bwd` currently
replaces only the latter. The two baselines are different implementations.

Atomic accumulates parameter gradients directly. Persistent saves FP32 partials
and runs two final reductions. At L384 the NCU profile measured ~8.7 us for each
ATen reduction; those fixed costs matter more for the smaller workload. However,
the old TMA main itself was also slower than atomic (114.27 vs 101.76 us under
NCU). It is not enough to say that the smaller input cannot fill the GPU.
Triton atomic reached 2.261 TB/s versus 2.006 TB/s for the old TMA main in that
profile. These are profiler measurements, not the graph timings below.

## Experimental improvement, unchanged persistent fusion boundaries

The new candidate maps adjacent lanes to adjacent m-major rows (four row lanes),
changes the feature reduction mapping, and replaces each final ATen reduction
with a small TMA reduction. **The parameter reductions remain two separate
launches**. No fusion of B4 with another TriMul stage, no added GEMM or copy.
Selected experimental main: BM32/BK256, four warps, two stages. Final reduction:
four feature columns per CTA, four warps. This mapping/reduction search is still
experimental, not a complete production configuration-space qualification.

H100 80GB, BF16 N256, two independent captures with rotated paired measurements.
All B4 timings include parameter-gradient initialization or final reductions.

| L | Capture | Triton atomic, us | Triton persistent, us | Existing TMA, us | Experimental TMA, us |
|---:|---:|---:|---:|---:|---:|
|384|0|102.784|146.080|118.272|100.384|
|384|1|102.720|145.888|118.144|100.416|
|768|0|384.272|475.264|399.648|355.552|
|768|1|383.584|475.200|399.808|355.712|

Versus the fastest measured Triton (atomic), the candidate achieves ~1.023x at
L384 and ~1.080x at L768. Time reductions are ~2.3% and ~7.4%, respectively.
Neither meets the 15% speedup target. Atomic TMA variants were also tested and
lost to Triton atomic; TMA by itself does not guarantee a gain.

## Whole bidirectional TriMul, dropout enabled

Official benchmark fixture, B1/D=hidden=128, BF16 mixed precision, dropout=0.25,
static compile plus manual stochastic CUDA graph, forward+backward, no optimizer.
All arms retain the same H100 F2/F567/B9+B10. Only B4 changes; this is **not** a
whole-module comparison against pure Triton. Same data, parameters and RNG are
used for numerical checks, and every timed replay advances production RNG.

| L | Capture | Triton B4 default, ms | Existing TMA opt-in, ms | Forced atomic B4, ms | Experimental B4, ms |
|---:|---:|---:|---:|---:|---:|
|384|0|1.532440|1.532304|1.532752|1.527152|
|384|1|1.533448|1.531640|1.531216|1.531648|
|768|0|6.016784|5.939664|5.929008|5.899936|
|768|1|6.030752|5.939104|5.930208|5.904200|

L384 has **no consistent whole-module improvement** across independent captures.
L768 improves ~0.44–0.49% versus forced atomic. Some L768 samples show jitter;
these small module gains should not be extrapolated to full-model training.
All fixture checks passed, including finite/present gradients, RNG advancement,
reset repeatability and gradient overwrite. Maximum paired gradient relative-L2
versus the default B4 arm was 1.90e-4 at L384 and 3.07e-4 at L768; the experimental
reduction changes floating-point summation order. This is not bitwise equality.

## Status and evidence

Production routing remains unchanged. L384 retains Triton atomic. The existing
L768 persistent/TMA dispatch is not the fastest measured available choice and
needs a separately validated selection policy. The new mapped candidate is
experimental: broad shape/config validation and sanitizer coverage remain before
production promotion; the previous native kernel's checks do not cover it.

JSON evidence is stored alongside this report. Reproduction scripts, candidate
sources and NCU report are retained under
`/home/psk6950/MiniWorld/runs/trimul_b4_l384_20260918/`:
`qualify.py`, `module_mapped.py`, `persistent_mapped.py`, `reduce_tiled.py`,
`compare.py` and `l384-paths.ncu-rep`. The `triton` arm label in module JSON means
H100 mixed module with default Triton B4, not a pure Triton module.
