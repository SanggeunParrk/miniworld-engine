# Single-direction v6 trimul — fwd+bwd (training): ours BEATS dtv1 at all L

v6 = SPLIT-back trimul training (autograd-composed kernels). Structure (deliberate):
LN_in kept SEPARATE (x_n is reused by the gate → fusing LN_in would force a 2nd LN) →
`_FrontFn` (cute fwd + triton `front_bwd_fused` bwd) → contraction (`_TriContract`,
contiguous-grad bwd) → LN_out+@Wp via `layernorm_linear_te_fn` (stride-transparent,
copy-free on the m-major tri view) → `GateElem`.

**Regime:** ALL methods `torch.compile`, params require grad (EXACT training), manual
CUDA-event timed. B=1, bf16, outgoing, d_pair=d_hidden=128. H100. cos(ours,fp32)=0.99997.

## Result (fwd+bwd, ms/layer; lower is better)

LN_out uses the stride-transparent TE-style `layernorm_linear_te_fn` (fused LN+@Wp,
reads the m-major tri view copy-free, returns dx m-major → free tri-grad reshape):

| L | pytorch | **ours_v6** | nvidia(dtv1) | ours vs dtv1 |
|----:|--------:|------------:|-------------:|:------------:|
| 256 | 1.137 | **0.894** | 1.413 | **1.58× faster** |
| 384 | 2.803 | **1.107** | 1.589 | **1.44× faster** |
| 512 | 5.417 | **1.879** | 2.444 | **1.30× faster** |
| 768 | 20.603 | **4.139** | 5.324 | **1.29× faster** |
| 1024 | 38.426 | **7.406** | 9.551 | **1.29× faster** |

v6 wins at every L (1.58× small → ~1.29× large). cuequiv ≈ 1.6/2.6/10.3 also beaten.
**Accuracy holds across all L: fwd cos 0.99998, grad_x cos 0.99997 (vs fp32 ref) at every
L ∈ {256,384,512,768,1024}** — the optimizations (transpose-fused LN → front_bwd cuBLAS
rewrite → te-style LN_out) changed speed only, not correctness.

![fwd+bwd](v6_fwdbwd_compile.png)

ours v6 is the **fastest training fwd+bwd at every L** — beats NVIDIA dtv1,
cuEquivariance, and compiled PyTorch.

## How we got here — the gap was KERNEL optimization, NOT structure

The first cut was 2.6× SLOWER than dtv1. Two fixes (structure unchanged):

1. **Transpose-fused LN_out** (`layer_norm_transpose`, dbn→bnd): the contraction emits
   tri channel-major; the old `.t()` to feed LayerNormLinear forced 2× strided
   (D,L,L) transpose-clones = **40% of the step** (profiled). Reading tri channel-major
   directly kills them. → 1.4× slower.
2. **front_bwd_fused rewrite** (the decisive fix): the hand-written split-K `_dw_kernel`
   (4×(D,D) fp32 accumulators, detuned to BK=64 to fit shared) ran at **~30 TFLOPS**.
   Replaced by a light channel-major elementwise (build `d_concat (4D,M)` =
   `[d_gLlog;d_pL;d_gRlog;d_pR]`) + **TWO cuBLAS GEMMs** (`dW = d_concat@x_n`;
   `dxn = d_concatᵀ@W_stack`). front bwd **4.0 → 1.46 ms** (2.75×, now < dtv1's fused 1.57).
   → v6 fwd+bwd 13.6 → 7.63 ms, **wins dtv1**.

**Lesson:** a hand-written split-K wgrad kernel at ~30 TFLOPS is the smell — cuBLAS does
huge-K reduction GEMMs far better. The contraction bmm (~1.2 ms, identical to dtv1) was
never the bottleneck; misreading profiler CUDA-total vs self-time sent us chasing it twice.

## Reproduce
`cute/v6_bench.py <method>` (per-method process), `cute/v6_compile_plot.py`. srun h100.
