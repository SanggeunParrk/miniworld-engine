# trimul — full TriangleMultiplication inference

## Math

```
x_normed   = LN_in(x)                                  # (B, L, L, D)
left/right = tm1(x_normed)                             # 2 × (B, L, L, D)
            [apply 2D pair mask]
tri_out    = einsum("bikd,bjkd->bijd", left, right)    # outgoing direction
out_normed = LN_out(tri_out)
y          = σ(x_normed @ W_g) ⊙ (out_normed @ W_p)    # tm2
```

The bench compares four end-to-end implementations.

## Implementations

| name              | description                                                                       |
|-------------------|-----------------------------------------------------------------------------------|
| `pt` (pytorch)    | `F.layer_norm` + plain matmul + `torch.einsum` + manual `σ` ⊙ — entirely in `[B,L,L,D]`. |
| `nv` (nvidia-triton) | `perf/trimul` PR #84's `fused_triangle_multiplicative_update_dtv1` — a stack of three Triton kernels (LN, dual-acc input GEMM, dual-acc output GEMM) + a `bmm`-based contraction in `[D,B,L,L]`. |
| `cq` (cuequivariance) | `cuequivariance_torch.triangle_multiplicative_update` — same architecture as dtv1 but uses cuequiv's AOT-tuned versions of each Triton kernel. |
| `cu` (cute, ours) | tm1 = patched `quack.GemmGatedSm90` writing directly to `[B,D,L,L]`. Contraction = `torch.einsum("bdik,bdjk->bijd")` (cuBLAS bmm). LN_in + mask fused in a custom Triton kernel. LN_out fused with the post-contraction transpose via cuequiv `dbn->bnd`. tm2 = cuequiv `fused_sigmoid_gated_dual_gemm_dual_x`. |

The cute path's lever is the `[B,d,L,L]` layout: it turns the 4-D
einsum-with-permute into a flat batched matmul (`bdll` → `bmm`), and
lets LN_out fuse the layout flip back to `[B,L,L,D]`. Combined with
LN_in + mask fusion, the cute path lands between dtv1 and cuequivariance
at moderate L and ahead of both at L ≥ 512.

## Results (H100, bf16, B=1, D=128)

Wall time (ms):

| L    | pytorch | nv-triton (dtv1) | cuequivariance | **cute** |
|------|--------:|-----------------:|---------------:|---------:|
| 384  |   1.29  |             0.36 |           0.36 |    0.46  |
| 512  |   2.45  |             0.61 |           0.47 |  **0.47**|
| 768  |   8.29  |             1.31 |           1.05 |  **0.93**|
| 1024 |  15.74  |             2.39 |           2.02 |  **1.71**|

TFLOPS:

| L    | pytorch | nv-triton | cuequivariance | **cute** |
|------|--------:|----------:|---------------:|---------:|
| 384  |    33.7 |     121.8 |          121.2 |     95.5 |
| 512  |    35.0 |     142.0 |          181.3 | **184.1**|
| 768  |    28.0 |     177.7 |          221.2 | **250.4**|
| 1024 |    30.6 |     201.5 |          237.8 | **282.2**|

Speedup vs pytorch / vs others:

| L    | nv/pt | cq/pt | cu/pt | **cu/nv** | **cu/cq** |
|------|------:|------:|------:|----------:|----------:|
| 384  |  3.62 |  3.60 |  2.83 |    0.78   |    0.79   |
| 512  |  4.06 |  5.18 |  5.26 |  **1.30** |    1.02   |
| 768  |  6.35 |  7.90 |  8.95 |  **1.41** |  **1.13** |
| 1024 |  6.59 |  7.78 |  9.23 |  **1.40** |  **1.19** |

## Per-stage breakdown of the cute path (L = 1024)

| stage                                            | ms     |
|--------------------------------------------------|-------:|
| 1. `fused_ln_mask` (LN_in + mask in one Triton kernel) | 0.23  |
| 2. tm1 cute direct-BDLL launch (two `GemmGatedSm90` launches -> `[B,D,L,L]`) | 0.43  |
| 3. `einsum("bdik,bdjk->bdij", left, right)` (cuBLAS bmm) | 0.45  |
| 4. cuequiv `layer_norm_transpose(dbn->bnd)` (LN_out + permute back) | 0.34  |
| 5. tm2 cute gated dual-GEMM launch (`fused_sigmoid_gated_dual_gemm_dual_x`) | 0.29  |
| **total**                                        | **~1.74** |

(Measured aggregate via `do_bench` lands at 1.71 ms — the per-stage sum
includes small per-call overhead that disappears under do_bench timing.)

## Optimizations log

| change                                               | L=1024 ms | Δ |
|------------------------------------------------------|----------:|--:|
| start: `bmm + permute(blld).contig + F.layer_norm`   |     5.25  | — |
| swap permute for `einsum("bdik,bdjk->bijd")` (no net win — LN_out then sees non-contig) | 5.32 | (no change) |
| switch LN_in to cuequiv `layer_norm_transpose` (`nd->nd`) | 3.45 | −1.80 |
| fold LN_out + post-permute into cuequiv `dbn->bnd`   |     2.00  | −1.45 |
| fuse LN_in + per-row mask into one Triton kernel     |     1.70  | −0.30 |

## Benchmark Files

Runtime code:

- `src/miniworld_engine/modules/triangle_multiplication/`
- `src/miniworld_engine/kernels/tm1/`
- `src/miniworld_engine/kernels/tm2/`
- `src/miniworld_engine/kernels/fused_ln_mask/`

Benchmark code and generated results:

- `benchmarks/modules/triangle_multiplication/configs/bench.yaml`
- `benchmarks/modules/triangle_multiplication/artifacts/`
- `benchmarks/runners/bench.py`

Run the unified compiled benchmark with:

```bash
python benchmarks/runners/bench.py target=triangle_multiplication level=module
```

The trimul CSV and SVG figures are generated under
`benchmarks/modules/triangle_multiplication/artifacts/<GPU name>/`.

The unified runner writes benchmark CSVs only; `benchmarks/runners/plot_csv.py`
renders SVGs from those CSVs as a separate step. Current trimul runs generate
both inference and training CSVs:

- `triangle_multiplication_n_layers=1_inference_time_bf16-mixed_compile_seq_len_L_sweep.csv`
- `triangle_multiplication_n_layers=1_inference_time_bf16-mixed_compile_d_pair_d_sweep.csv`
- `triangle_multiplication_n_layers=1_training_time_bf16-mixed_compile_seq_len_L_sweep.csv`
- `triangle_multiplication_n_layers=1_training_time_bf16-mixed_compile_d_pair_d_sweep.csv`

Those CSVs contain method, dimensions, dtype/precision, mode, metric, device,
compile flag, sweep axis, status/error, and value. Render grouped bar plots from
them with short output names. If `--name` is omitted, the renderer includes the
mode automatically:

The trimul d sweep uses the canonical channel widths `d_pair=128,256,512` at
fixed `L=384`; do not use an arithmetic range that inserts `d_pair=384`.

```bash
python benchmarks/runners/plot_csv.py \
  benchmarks/modules/triangle_multiplication/artifacts/<GPU name>/triangle_multiplication_n_layers=1_inference_time_bf16-mixed_compile_seq_len_L_sweep.csv \
  benchmarks/modules/triangle_multiplication/artifacts/<GPU name>

python benchmarks/runners/plot_csv.py \
  benchmarks/modules/triangle_multiplication/artifacts/<GPU name>/triangle_multiplication_n_layers=1_training_time_bf16-mixed_compile_d_pair_d_sweep.csv \
  benchmarks/modules/triangle_multiplication/artifacts/<GPU name>
```
