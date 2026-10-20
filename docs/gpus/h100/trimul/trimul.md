# TriangleMultiplication on H100 (sm90)

Kernel-level status of the TriMul modules on H100; the module-level summary is in
[h100.md](../h100.md). bf16 only; columns are (Length, Dimension). H100 = CUDA; a shape
without a CUDA path is 미구현 (it runs the PyTorch reference: no portable-kernel fallback on H100, see [dispatch.md](../dispatch.md)). Figures: one box per kernel, left to right, HBM reads (blue,
left) and writes (red, right); generated from `figures/trimul_bidir.json` by
`python -m miniworld_engine.viz.kernel_flow`, then `rsvg-convert -z 2 -b white` to PNG.
Dispatch: `integrations/trimul_h100.py`.

## Batched samples (B <= 48 per launch, any B per call, every width)

Every width takes B samples in one call, inference and training, bidirectional and either single direction: tokens b-major, one
launch per stage instead of B calls. MiniWorld folds samples into the batch of one pair stack (the template embedder's templates at
D64; structures / augments at D128 and wider). The MiniWorld-facing limit is `integrations.trimul_h100.max_batch(width)`
(`MAX_BATCH_BY_WIDTH`, 48 at D64 / 128 / 256 / 384 / 512); `MAX_BATCH` keeps its D64 meaning (= `max_batch(64)`, as on B200). `max_batch` is what ONE LAUNCH takes; a call with more samples, or with more than fit in memory, runs in
several launches (`trimul_h100._plan`, below), so any B is served and nothing falls back to the PyTorch statements because of size.
A length / width no batched path takes runs the module's PyTorch statements, as before.

**Chunking** (`_plan`, `_Chunked`). The chunk is a deterministic function of the call's shapes, width, direction and mode and of the
device's *total* memory (never of live free memory), so torch.compile and CUDA-graph capture see the same split every time. One
launch's working set is a measured coefficient x D bytes per pair token (one batched call's peak plus 5 %: inference 18.1
bidirectional / 12.1 single direction; training D64 / D128 40.3 / 32.0, D256+ 62.2 / 42.2) and must fit `CHUNK_MEMORY_FRACTION` = 0.85
of the device. Chunked inference writes each chunk into the output in place (`inference_into`). Chunked training is one autograd
node (`_Chunked`): y and dx are allocated once and each chunk's launch writes its rows of them in place (the path's own y / dx
allocation takes the rows, `_h100_runtime.output_buffers`), so nothing is copied or concatenated and the batch holds only the
caller's x / y / dy / dx (8 D bytes per pair token) next to the chunk being run; parameter gradients are the chunks' sums in chunk
order. Plain chunks keep their graph from forward to backward when every chunk's saved state (measured 10-20 D bytes per pair token)
fits next to one chunk's backward workspace; otherwise each chunk is checkpointed (forward without saves, again with grad in the
backward). D256 L768 B48, D384 L768 B26 and D512 L768 B20 training now run (peak 67.5 / 63.3 / 70.0 GB of 85 GB, checkpointed
chunks of 1; x / y / dy / dx alone are 58.0 / 47.1 / 48.3 GB), y and dx bitwise equal to B one-sample calls. Where x / y / dy / dx
alone exceed the device, no split helps: that is 8 B L^2 D bytes, e.g. B > 70 at D256 L768 or B > 35 at D512 L768 on 80 GB.

| width | inference (B > 1) | training (B > 1) | path |
|---|---|---|---|
| D64 | any L >= 16 (L % 16 != 0: host pad to 16) | L % 128 == 0 | native K1 / K3; `save_k3_d64`, B1 / B2 |
| D128 | any L >= 16 (L % 16 != 0: host pad to 16) | L % 128 == 0 | native K1 / K3; training front K1, `save_k3`, B1, B7 (bidirectional) / `h100_native` K1, `save_k3`, `single_b1`, `single_b7` (single direction) |
| D256 / D384 / D512 | L % 128 == 0; other L >= 16: each sample padded on the host to the next multiple of 128 (masked pad tokens, pad rows dropped) | L % 128 == 0 | the B200-style port, at every L (the preserved single-sample cells, `_legacy_native_faster`, keep B = 1 only) |

The design is the D64 one at every width. K1 takes the samples stacked along its slow token axis (`K1Params.rows` = B L, 0 = one
sample, in the struct's alignment padding: the packaged inference header, the B200-style port's forward / backward K1 and now the
upstream header behind `h100_native.Front`), so a tile may straddle two samples: every row reads its own sample's token mask
(`m[i] & m[s L + j]`). The planes are `[c, B L, L]`; the contraction folds the samples into cuBLAS's batch dim (c x B, one call per
product, forward and backward). Everything after the contraction is token-flat (M = B L^2) and only needs the row-dropout scale
`[B L, D]` read at row `s L + j`:

| kernel | change |
|---|---|
| inference K1 / K3 (D128) | none beyond D64's (`rows`); the D128 tile table has no wide K3 |
| `save_k3.cu` (D128 training K3, both modules) | `RecomputeParams.rows`; dropout-scale base hoisted once per tile (`ds_rows`), as in `save_k3_d64` |
| `b1_fused.cu` (D128 bidirectional B1) | a separate build `B1_BATCHED=1` (selected when M > L^2): the per-period dropout fragment cache is off and the scale row is `s L + j` (base per 64-token tile). One sample keeps the original cubin: the first version computed the base in the shared build and cost B1 +5 % (180 -> 189 us at L384) |
| `b7_384` / `b7_768` `joint.cu` (D128 bidirectional B7) | token-mask mode reads `m[t / L] & m[t / L^2 L + t % L]` (compile-time L); the pair-mask mode is flat |
| upstream `tmn_kernels.cuh` K1 (`h100_native.Front`, D128 single direction) | `K1Params.rows`, row bound and per-sample token mask, as in the inference header |
| `single_output/kernel.cu` (`single_b1`) | dropout-scale base per 64-row tile |
| `single_b7` | none (flat pair mask `[B L, L]`) |
| B200-style port: K1 (`_front_rs`, `h100_b200style_direct`) | `K1Params.rows`, maps over B L rows; the D512 shared-resident fronts (`k1ss*`) and every backward kernel are token-flat already |
| B200-style K3 (`k3`, `k3reorder`, `k3tmasave`, `k3d384_tmasave`, `k3scalar_tmasave`) | dropout-scale base per 64-token warpgroup (`dsb`) |
| `b200_sources/wide_bwd.cu` `gate_bwd_kernel` (shared with B200) | scale row `row % L + row / L^2 L`, only when M > L^2 (one sample keeps the single division; B200's arithmetic unchanged) |
| `mw_aux_owned_mask` | `samples`: the bf16 pair mask `[B L, L]` from a token mask `[B L]` or a pair mask `[B L L]` |
| integration | `max_batch(width)`; `[B L]` masks and `[B L, D]` scales; B > 1 at D128 always takes the preserved (B1 / B7) kernels; B > 1 at D256+ always the port; padded wide inference (`_padded_inference(..., 128)`) |

Correctness, H100 80GB HBM3, torch 2.13.0+cu129, bf16 pair, FP32 parameters, a token mask and a row-dropout scale drawn per sample:

- **B 16 / 32 / 48 x L128 / 384 x every width x three directions x inference / training (144 / 144 cells)**: one call against B
  calls of one sample on the same path AND against the FP32 PyTorch statement (per sample; the weight gradients summed over the
  samples, which is exact for the batch). y and dx bitwise equal to the B one-sample calls in every cell; against FP32 y <= 3.0e-3,
  dx <= 4.7e-3, weight gradients <= 7.8e-3 relative, and the batch is as close to FP32 as the sum of B one-sample calls (weight
  gradients batch-vs-samples <= 8.6e-4, the FP32 summation order). The L384 training cells at D256+ with B 32 / 48 ran chunked
  (`_plan`), some checkpointed: the same results. So 48 is backed at every width (one launch at L128 for every mode; L384 inference
  one launch, L384 training chunked where 48 do not fit).
- the same at L768 for B 4 / 8 / 16 / 32 / 48 (bidirectional and outgoing, training; inference B 16 / 48): every cell that fits
  passes, including the plain-chunk and checkpointed-chunk tiers.
- B 2 ... 48 x L 128 / 384 / 768 / 1536 against B one-sample calls (bitwise y / dx, weight gradients <= 1.2e-3): every cell that fits
  in one launch passes (the sweep that set the one-launch memory coefficients).
- D128 is bitwise run to run; the port's backward sums its LN-out row statistics and gate-affine partials with `atomicAdd`
  (`wide_bwd.cu`, unchanged), so at D256+ only y is bitwise run to run (gradients 7e-8 relative), as for one sample.

Where chunking reaches its end: the caller's own x / y / dy / dx (training, 8 D bytes per pair token per sample) or x / y
(inference, 4 D) have to fit; everything else is per chunk. The kernels' own index bound is the int32 token index per launch
(B L^2 < 2^31).

Largest B one launch took in the first sweep (80 GB; bidirectional / single direction) and peak memory per pair token:

| width | inference L128 / 384 / 768 / 1536 | training L128 / 384 / 768 / 1536 |
|---|---|---|
| D128 | 48 / 48 / 48 / 8 (both) | 48 / 48 / 16 / 4 (bidirectional), 48 / 48 / 32 / 8 (single) |
| D256 | 48 / 48 / 16 / 4 | 48 / 32 / 8 / 2 (bidirectional), 48 / 48 / 8 / 2 (single) |
| D384 | 48 / 48 / 16 / 4 | 48 / 16 / 4 / - (bidirectional), 48 / 32 / 8 / 2 (single) |
| D512 | 48 / 48 / 8 / 2 | 48 / 16 / 4 / - (both) |

Training D128 bidirectional 4.6 kB, D256 bidirectional 15 kB, D512 bidirectional 30 kB (the port's backward workspace; it holds
only 18-20 D bytes per pair token from the forward to the backward); inference D128 bidirectional 2.2 kB, D512 bidirectional 8.6 kB.

B = 1 is unchanged: y, dx and every weight gradient bitwise equal to main at D128 (L384 / 768, every direction), y and dx bitwise at
D256 / 384 / 512 (their weight gradients differ only where main itself is not bitwise run to run), and the time within main's
run-to-run spread (0.984-1.012, two interleaved rounds, every width and direction at L384 / 768; the record has every cell).

Measured 2026-10-19 on one H100 80GB HBM3, CUDA-graph timing (best of 5 runs of up to 30 replays), the module (`BidirectionalTriangle
Multiplication(D)` / `TriangleMultiplication(D, outgoing=True)`, hidden = D), FP32 parameters, bf16 pair, a token mask (last L/8
tokens masked); training = forward + backward of x and every parameter, row dropout on. "B calls" = B one-sample calls of the same
module in one graph (at the preserved single-sample cells, `_legacy_native_faster`, the preserved kernels). cuEquivariance = `implementation="cuequivariance"`,
PyTorch = the module's statements, both under bf16 autocast (what H100 ran for B > 1 before). OOM: did not fit in 80 GB (the
graph's memory pool included). Record: [records/trimul_h100_batched_wide_20261019.md](../../../records/trimul_h100_batched_wide_20261019.md).

| D | module | L | B | mode | one call (ms) | B calls (ms) | cuEquivariance (ms) | PyTorch (ms) | × vs B calls | × vs cuEquivariance | × vs PyTorch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | bidirectional | 128 | 4 | inference | 0.121 | 0.160 | 0.326 | 1.125 | 1.33 | 2.70 | 9.31 |
| 128 | bidirectional | 128 | 4 | training | 0.499 | 0.794 | 1.467 | 3.250 | 1.59 | 2.94 | 6.52 |
| 128 | bidirectional | 128 | 48 | inference | 1.245 | 1.901 | 4.440 | 12.847 | 1.53 | 3.57 | 10.32 |
| 128 | bidirectional | 128 | 48 | training | 4.819 | 9.462 | 17.024 | 31.454 | 1.96 | 3.53 | 6.53 |
| 128 | bidirectional | 384 | 4 | inference | 0.877 | 0.948 | 3.042 | 9.920 | 1.08 | 3.47 | 11.31 |
| 128 | bidirectional | 384 | 4 | training | 3.539 | 3.806 | 12.731 | 24.238 | 1.08 | 3.60 | 6.85 |
| 128 | bidirectional | 384 | 48 | inference | 10.591 | 11.301 | 40.099 | 118.813 | 1.07 | 3.79 | 11.22 |
| 128 | bidirectional | 384 | 48 | training | 42.308 | 45.271 | 156.420 | OOM | 1.07 | 3.70 | - |
| 128 | bidirectional | 768 | 4 | inference | 3.688 | 3.738 | 13.523 | 67.234 | 1.01 | 3.67 | 18.23 |
| 128 | bidirectional | 768 | 4 | training | 14.125 | 14.448 | 49.139 | 137.069 | 1.02 | 3.48 | 9.70 |
| 128 | bidirectional | 768 | 48 | inference | 43.028 | 45.213 | OOM | OOM | 1.05 | - | - |
| 128 | bidirectional | 768 | 48 | training | 214.840 | 171.330 | OOM | OOM | 0.80 | - | - |
| 128 | outgoing | 128 | 4 | inference | 0.072 | 0.117 | 0.145 | 0.736 | 1.61 | 2.00 | 10.17 |
| 128 | outgoing | 128 | 4 | training | 0.399 | 0.672 | 0.864 | 2.707 | 1.68 | 2.17 | 6.78 |
| 128 | outgoing | 128 | 48 | inference | 0.673 | 1.384 | 1.417 | 8.513 | 2.06 | 2.11 | 12.66 |
| 128 | outgoing | 128 | 48 | training | 3.533 | 7.946 | 7.796 | 25.623 | 2.25 | 2.21 | 7.25 |
| 128 | outgoing | 384 | 4 | inference | 0.530 | 0.566 | 1.078 | 6.540 | 1.07 | 2.03 | 12.33 |
| 128 | outgoing | 384 | 4 | training | 2.680 | 3.040 | 5.973 | 19.926 | 1.13 | 2.23 | 7.44 |
| 128 | outgoing | 384 | 48 | inference | 6.261 | 6.838 | 12.572 | 77.976 | 1.09 | 2.01 | 12.45 |
| 128 | outgoing | 384 | 48 | training | 31.297 | 36.435 | 68.606 | 235.521 | 1.16 | 2.19 | 7.53 |
| 128 | outgoing | 768 | 4 | inference | 2.044 | 2.179 | 4.268 | 40.081 | 1.07 | 2.09 | 19.61 |
| 128 | outgoing | 768 | 4 | training | 10.549 | 10.939 | 23.504 | 141.556 | 1.04 | 2.23 | 13.42 |
| 128 | outgoing | 768 | 48 | inference | 25.602 | 26.198 | 51.108 | OOM | 1.02 | 2.00 | - |
| 128 | outgoing | 768 | 48 | training | 126.730 | 131.450 | OOM | OOM | 1.04 | - | - |
| 256 | bidirectional | 128 | 4 | inference | 0.272 | 0.315 | 0.791 | 2.051 | 1.16 | 2.91 | 7.54 |
| 256 | bidirectional | 128 | 4 | training | 1.370 | 1.755 | 3.151 | 5.717 | 1.28 | 2.30 | 4.17 |
| 256 | bidirectional | 128 | 48 | inference | 2.838 | 3.761 | 9.279 | 23.522 | 1.33 | 3.27 | 8.29 |
| 256 | bidirectional | 128 | 48 | training | 13.955 | 20.920 | 35.675 | 60.424 | 1.50 | 2.56 | 4.33 |
| 256 | bidirectional | 384 | 4 | inference | 2.144 | 2.289 | 6.976 | 18.415 | 1.07 | 3.25 | 8.59 |
| 256 | bidirectional | 384 | 4 | training | 10.434 | 10.414 | 28.203 | 46.786 | 1.00 | 2.70 | 4.48 |
| 256 | bidirectional | 384 | 48 | inference | 25.307 | 27.218 | 84.739 | OOM | 1.08 | 3.35 | - |
| 256 | bidirectional | 384 | 48 | training | 125.454 | 125.003 | OOM | OOM | 1.00 | - | - |
| 256 | bidirectional | 768 | 4 | inference | 9.158 | 9.352 | 28.276 | 132.712 | 1.02 | 3.09 | 14.49 |
| 256 | bidirectional | 768 | 4 | training | 41.501 | 41.456 | 111.018 | 276.376 | 1.00 | 2.68 | 6.66 |
| 256 | bidirectional | 768 | 48 | inference | 108.135 | 110.995 | OOM | OOM | 1.03 | - | - |
| 256 | bidirectional | 768 | 48 | training | 659.791 | 499.952 | OOM | OOM | 0.76 | - | - |
| 256 | outgoing | 128 | 4 | inference | 0.166 | 0.233 | 0.581 | 1.346 | 1.40 | 3.49 | 8.09 |
| 256 | outgoing | 128 | 4 | training | 0.911 | 1.201 | 2.173 | 4.748 | 1.32 | 2.39 | 5.21 |
| 256 | outgoing | 128 | 48 | inference | 1.705 | 2.767 | 7.348 | 15.183 | 1.62 | 4.31 | 8.91 |
| 256 | outgoing | 128 | 48 | training | 8.899 | 14.300 | 23.482 | 48.752 | 1.61 | 2.64 | 5.48 |
| 256 | outgoing | 384 | 4 | inference | 1.300 | 1.389 | 5.240 | 11.750 | 1.07 | 4.03 | 9.04 |
| 256 | outgoing | 384 | 4 | training | 6.717 | 6.926 | 17.557 | 38.394 | 1.03 | 2.61 | 5.72 |
| 256 | outgoing | 384 | 48 | inference | 15.238 | 16.497 | 62.233 | 141.080 | 1.08 | 4.08 | 9.26 |
| 256 | outgoing | 384 | 48 | training | 79.112 | 83.059 | 209.411 | OOM | 1.05 | 2.65 | - |
| 256 | outgoing | 768 | 4 | inference | 5.359 | 5.462 | 20.854 | 76.236 | 1.02 | 3.89 | 14.22 |
| 256 | outgoing | 768 | 4 | training | 26.924 | 26.932 | 70.697 | 284.983 | 1.00 | 2.63 | 10.58 |
| 256 | outgoing | 768 | 48 | inference | 63.228 | 64.164 | OOM | OOM | 1.01 | - | - |
| 256 | outgoing | 768 | 48 | training | 404.797 | 322.384 | OOM | OOM | 0.80 | - | - |
| 384 | bidirectional | 128 | 4 | inference | 0.497 | 0.582 | 1.629 | 3.072 | 1.17 | 3.28 | 6.18 |
| 384 | bidirectional | 128 | 4 | training | 2.278 | 2.816 | 5.569 | 8.294 | 1.24 | 2.44 | 3.64 |
| 384 | bidirectional | 128 | 48 | inference | 5.484 | 6.963 | 19.203 | 35.411 | 1.27 | 3.50 | 6.46 |
| 384 | bidirectional | 128 | 48 | training | 25.490 | 33.793 | 64.091 | 91.374 | 1.33 | 2.51 | 3.58 |
| 384 | bidirectional | 384 | 4 | inference | 4.118 | 4.306 | 14.418 | 27.799 | 1.05 | 3.50 | 6.75 |
| 384 | bidirectional | 384 | 4 | training | 18.132 | 18.209 | 49.103 | 70.991 | 1.00 | 2.71 | 3.92 |
| 384 | bidirectional | 384 | 48 | inference | 48.893 | 51.730 | OOM | OOM | 1.06 | - | - |
| 384 | bidirectional | 384 | 48 | training | 286.925 | 217.878 | OOM | OOM | 0.76 | - | - |
| 384 | bidirectional | 768 | 4 | inference | 17.439 | 17.311 | 58.946 | 206.705 | 0.99 | 3.38 | 11.85 |
| 384 | bidirectional | 768 | 4 | training | 71.131 | 71.153 | 200.038 | OOM | 1.00 | 2.81 | - |
| 384 | bidirectional | 768 | 48 | inference | 207.065 | 210.314 | OOM | OOM | 1.02 | - | - |
| 384 | bidirectional | 768 | 48 | training | OOM | 863.148 | OOM | OOM | - | - | - |
| 384 | outgoing | 128 | 4 | inference | 0.294 | 0.332 | 1.304 | 1.922 | 1.13 | 4.44 | 6.55 |
| 384 | outgoing | 128 | 4 | training | 1.417 | 1.825 | 3.981 | 6.724 | 1.29 | 2.81 | 4.75 |
| 384 | outgoing | 128 | 48 | inference | 3.140 | 3.925 | 15.434 | 21.933 | 1.25 | 4.92 | 6.99 |
| 384 | outgoing | 128 | 48 | training | 14.809 | 21.903 | 45.074 | 73.017 | 1.48 | 3.04 | 4.93 |
| 384 | outgoing | 384 | 4 | inference | 2.368 | 2.507 | 11.615 | 16.980 | 1.06 | 4.90 | 7.17 |
| 384 | outgoing | 384 | 4 | training | 11.121 | 11.050 | 33.897 | 57.546 | 0.99 | 3.05 | 5.17 |
| 384 | outgoing | 384 | 48 | inference | 27.983 | 29.978 | 139.530 | OOM | 1.07 | 4.99 | - |
| 384 | outgoing | 384 | 48 | training | 132.708 | 132.875 | OOM | OOM | 1.00 | - | - |
| 384 | outgoing | 768 | 4 | inference | 9.772 | 9.917 | 46.583 | 113.581 | 1.01 | 4.77 | 11.62 |
| 384 | outgoing | 768 | 4 | training | 43.661 | 43.658 | 136.007 | 442.309 | 1.00 | 3.12 | 10.13 |
| 384 | outgoing | 768 | 48 | inference | 117.218 | 117.494 | OOM | OOM | 1.00 | - | - |
| 384 | outgoing | 768 | 48 | training | OOM | 525.124 | OOM | OOM | - | - | - |
| 512 | bidirectional | 128 | 4 | inference | 0.885 | 1.167 | 2.723 | 4.220 | 1.32 | 3.08 | 4.77 |
| 512 | bidirectional | 128 | 4 | training | 3.529 | 4.371 | 8.544 | 11.006 | 1.24 | 2.42 | 3.12 |
| 512 | bidirectional | 128 | 48 | inference | 9.350 | 14.103 | 32.231 | 50.637 | 1.51 | 3.45 | 5.42 |
| 512 | bidirectional | 128 | 48 | training | 38.770 | 52.381 | 98.744 | 125.761 | 1.35 | 2.55 | 3.24 |
| 512 | bidirectional | 384 | 4 | inference | 7.075 | 7.390 | 24.191 | 39.047 | 1.04 | 3.42 | 5.52 |
| 512 | bidirectional | 384 | 4 | training | 27.092 | 27.058 | 75.418 | 97.787 | 1.00 | 2.78 | 3.61 |
| 512 | bidirectional | 384 | 48 | inference | 83.573 | 88.792 | OOM | OOM | 1.06 | - | - |
| 512 | bidirectional | 384 | 48 | training | 443.453 | 323.598 | OOM | OOM | 0.73 | - | - |
| 512 | bidirectional | 768 | 4 | inference | 29.400 | 29.536 | 98.713 | 273.413 | 1.00 | 3.36 | 9.30 |
| 512 | bidirectional | 768 | 4 | training | 104.748 | 104.221 | OOM | OOM | 0.99 | - | - |
| 512 | outgoing | 128 | 4 | inference | 0.515 | 0.649 | 2.317 | 2.487 | 1.26 | 4.50 | 4.83 |
| 512 | outgoing | 128 | 4 | training | 2.149 | 2.704 | 6.379 | 8.586 | 1.26 | 2.97 | 4.00 |
| 512 | outgoing | 128 | 48 | inference | 5.331 | 7.826 | 27.556 | 28.965 | 1.47 | 5.17 | 5.43 |
| 512 | outgoing | 128 | 48 | training | 22.736 | 31.751 | 73.332 | 95.408 | 1.40 | 3.23 | 4.20 |
| 512 | outgoing | 384 | 4 | inference | 4.038 | 4.336 | 20.681 | 22.422 | 1.07 | 5.12 | 5.55 |
| 512 | outgoing | 384 | 4 | training | 16.209 | 16.142 | 55.183 | 76.512 | 1.00 | 3.40 | 4.72 |
| 512 | outgoing | 384 | 48 | inference | 47.526 | 52.075 | OOM | OOM | 1.10 | - | - |
| 512 | outgoing | 384 | 48 | training | 258.167 | 193.311 | OOM | OOM | 0.75 | - | - |
| 512 | outgoing | 768 | 4 | inference | 16.423 | 16.708 | 82.860 | 150.889 | 1.02 | 5.05 | 9.19 |
| 512 | outgoing | 768 | 4 | training | 64.828 | 64.706 | 220.907 | OOM | 1.00 | 3.41 | - |

One call against B calls: 1.13-2.25x at L128 (B = 4 and 48); inference at L384 / 768 0.99-1.10x (chunked calls included); D128
training at L384 / 768 1.02-1.17x in one launch. Against cuEquivariance 2.0-5.2x, against the PyTorch statements 3.1-19.6x, where
they fit. The "B calls" column differentiates each call with `torch.autograd.grad` right after its forward and does not sum the
parameter gradients; B calls whose gradients accumulate (`backward()`, as a model runs them) are 2-18 % slower than that column.

- **D256+ training at L384 / 768 (the preserved single-sample cells, `_legacy_native_faster`)**: per cell and batch size the call
  runs whichever measured faster, one launch / chunks of the port or one launch per sample of the preserved kernels writing into
  the batch's y / dx (`PER_SAMPLE_LEGACY_TRAINING`: the largest B routed per (D, L, direction)). B = 4: 0.994-1.031x of the B calls
  above and 1.02-1.16x of accumulating B calls (the port alone was 0.88-1.03x: the routed call does the B calls' work and sums the
  gradients). Routing first lost 3-6 % to concatenating y and dx; writing in place removed that.
- **Training that does not fit one launch (B L^2 large): checkpointed chunks 0.73-0.80x** of the B calls above (0.75-0.89x of
  accumulating B calls). Those B calls hold
  one sample's saved tensors at a time because each is differentiated right after its forward; a model that runs the B calls and
  then one backward holds all of them, which at these sizes is beyond 80 GB. Plain (uncheckpointed) chunks and the routed
  per-sample calls are at 1.00-1.05x. The OOM rows at D384 / D512 L768 B48 are the caller's own x / y / dy / dx (87 / 116 GB > 85 GB).

### D64 (the first batched width)

The D64 path takes B samples in one call, inference and training, bidirectional and either single direction: tokens b-major, one
launch per stage instead of B calls (MiniWorld's template embedder runs its templates as the B samples of one pair stack). K1 / K3
(packaged inference K1 / K3, the training front K1 and `save_k3_d64`) take the samples stacked along the slow token axis
(`K1Params.rows` / `K3Params.rows` / `RecomputeParams.rows` = B L, 0 = one sample), so a tile may straddle two samples: every row
reads its own sample's token mask (K1: `m[i] & m[s L + j]`), and K3 / B1 read the row-dropout scale `[B L, 64]` at row `s L + j`.
The training preparation builds the FP32 pair mask `[B L, L]` per sample (`mw_aux_d64_training_prepare` / `mw_aux_train_mask`,
`samples`), B2 reads it per row, the planes are `[c, B L, L]` and the contraction batches over c x B (one cuBLAS call per product),
and the B1 / B2 weight partials sum over every sample in a fixed order (deterministic).
B > `MAX_BATCH` runs in several launches (chunking, above).

`integrations.trimul_h100.MAX_BATCH = 48` is the largest batch validated, not a kernel limit: B in {1, 2, 4, 8, 16, 32, 48} x
L in {128, 256, 384, 768} x the three directions, inference and training, all pass with y and dx **bitwise equal** to B calls of
one sample (the weight gradients differ only by the FP32 order of the per-sample sum, <= 2.1e-4 relative at L768 B48) and bitwise
run to run. The only index bound in the kernels is the int32 token index (B L^2 < 2^31); memory is the practical one (training
L768 B48 peaks at 68 GB). Tests: `tests/gpu/sm90/test_h100_trimul_batch_gpu.py` (one call vs B calls with a mask and a dropout
scale per sample, so an index mix-up between samples is a large error; the FP32 PyTorch reference, every gradient; bitwise
determinism; CUDA-graph capture with live inputs, masks, dropout scales and weights).

Measured 2026-10-19 on one H100 80GB HBM3, torch 2.13.0+cu129, bf16 pair, FP32 parameters, a token mask per sample, CUDA-graph
timing (median of 5 x 20 replays), B = 4: the MiniPairformer block (`BidirectionalTriangleMultiplication(64)` + `Transition(64)`,
the template stack's block; training = forward + backward of every gradient, row dropout on). "PyTorch" = one call of 4 samples
as H100 ran it before (TriMul's PyTorch statements for B > 1, Transition already CUDA on the flattened rows). Record:
[records/trimul_h100_batched_samples_20261019.md](../../../records/trimul_h100_batched_samples_20261019.md).

| L | mode | one call (ms) | 4 calls (ms) | PyTorch (ms) | × vs 4 calls | × vs PyTorch |
|---|---|---|---|---|---|---|
| 128 | inference | 0.079 | 0.135 | 0.436 | 1.72 | 5.54 |
| 128 | training | 0.309 | 0.681 | 1.255 | 2.21 | 4.07 |
| 384 | inference | 0.601 | 0.695 | 3.832 | 1.16 | 6.37 |
| 384 | training | 2.130 | 2.922 | 10.228 | 1.37 | 4.80 |
| 768 | inference | 2.467 | 2.776 | 28.452 | 1.13 | 11.53 |
| 768 | training | 9.045 | 10.712 | 60.315 | 1.18 | 6.67 |

## Long and ragged lengths (MiniWorld's L1536 / 2048 / 2560 rungs)

Contract (`integrations.trimul_h100`), both modules, every direction:

| width | inference | training |
|---|---|---|
| D64 / D128 (B <= 48) | any L >= 16 (`MIN_INFERENCE_LENGTH`) | L a multiple of 128 up to 10240 (`TRAINING_GRANULE`, `MAX_TRAINING_LENGTH`, B200's bound) |
| D256 / D384 / D512 (B > 1: the B200-style port) | as before up to L1024; above: L a multiple of 128 (B200-style port; single-direction D256 / D384 keep the packaged K1 / K3 at any L) | as before up to L1024; above: L a multiple of 128 (B200-style port) |

Bound common to both: the int32 token index, B L^2 < 2^31 (L rounded up to 16). Memory binds first on an 80 GB card: training at
L2560 peaks at 28 GB (D128 bidirectional), 13.5 GB (D64, one sample), 54 GB (D64, B = 4); at L4096 D64 bidirectional fits, D128
bidirectional training ends near L4300. Before this change every H100 TriMul call outside L128 ... 1024 (multiples of 128) ran the
module's PyTorch statements.

What it took, per kernel:

| stage | change |
|---|---|
| inference K1 / K3 (packaged, D64 / D128) | none: the tile table already has length buckets to N 4096 (`by_n`), plane offsets are 64-bit and one sample pads the planes to `Np` = L rounded up to 16 (zero pad rows written by K1). One sample with L % 16 != 0 failed since the batched-samples change (it passed `rows = Np` to K1, which the batched check rejects); B = 1 now leaves `rows` unset |
| B samples at L % 16 != 0 (inference) | the K1 / K3 sample stacking (`rows = B L`) has no pad, so `update_inference` pads each sample to the next multiple of 16 on the host: zero pair entries on masked pad tokens (K1's `m[i] & m[j]` zeroes their left / right rows and columns, so a real pair contracts the same terms), output sliced back |
| training front / `save_k3` / B1 / B7 (D128), `save_k3_d64` / B1 / B2 (D64), `single_b1` / `single_b7` | none: B1 is built per L (`TRAIN_L`), B7 runs the `b7_768` source (two-part weight partial) at every L but 384, the B1 / B7 / single-direction configurations take their 768 entries above L768; the length guards (`n in (128, ..., 1024)`) became `n % 128 == 0` |
| B200-style port (D256 / D384 / D512) | the length guard only (`n % 128 == 0`); its backward sums LN / gate partials with global atomics (`b200_sources/wide_bwd.cu`), so its gradients agree run to run only up to the FP32 summation order (y is bitwise), as at L <= 1024 |

Per-kernel time per pair token (torch profiler, `prof_longl.py`) against L768: the contraction (cuBLAS) runs at 730 TFLOP/s at
L2048 / 2560; inference K1 / K3 and B7 cost 12-14 % more per token, the D64 `b2w` 21 %, the D128 B1 38 % and `save_k3` 50 % (both
read and write the channel-strided `[c, L, L]` planes, `L^2` apart). The B1 builds with the extended dropout-fragment cache
(`B1_EXT_MASK`), another dTri store tile or gate demotion move the step by +-1 % (L1536-2560): not adopted.

Measured 2026-10-19, one H100 80GB HBM3, torch 2.13.0+cu129, bf16 pair, FP32 parameters, a token mask, CUDA-graph timing (median of
5 x 20 replays); training = forward + backward of x and every parameter, row dropout on. "PyTorch" = the module's PyTorch statements,
which H100 ran at these lengths before (under bf16 autocast, CUDA graph); cuEquivariance = `implementation="cuequivariance"` under bf16
autocast. Record: [records/trimul_h100_long_l_20261019.md](../../../records/trimul_h100_long_l_20261019.md).

| module | L | mode | ours (ms) | PyTorch (ms) | cuEquivariance (ms) | × vs PyTorch | × vs cuEquivariance |
|---|---|---|---|---|---|---|---|
| bidirectional D64, B = 4 | 1536 | inference | 9.50 | 119.04 | 20.13 | 12.5 | 2.12 |
| bidirectional D64, B = 4 | 1536 | training | 30.47 | 248.36 | 97.56 | 8.15 | 3.20 |
| bidirectional D64, B = 4 | 2048 | inference | 20.09 | 214.76 | 39.03 | 10.7 | 1.94 |
| bidirectional D64, B = 4 | 2048 | training | 62.27 | 468.93 | 181.60 | 7.53 | 2.92 |
| bidirectional D64, B = 4 | 2560 | inference | 36.71 | 351.62 | 65.95 | 9.58 | 1.80 |
| bidirectional D64, B = 4 | 2560 | training | 112.54 | OOM | OOM | - | - |
| bidirectional D128 | 1536 | inference | 5.19 | 60.96 | 14.51 | 11.7 | 2.79 |
| bidirectional D128 | 1536 | training | 19.72 | 191.54 | 55.33 | 9.71 | 2.81 |
| bidirectional D128 | 2048 | inference | 10.59 | 111.18 | 26.95 | 10.5 | 2.54 |
| bidirectional D128 | 2048 | training | 40.06 | 355.27 | 102.82 | 8.87 | 2.57 |
| bidirectional D128 | 2560 | inference | 19.26 | 175.94 | 44.57 | 9.14 | 2.31 |
| bidirectional D128 | 2560 | training | 69.47 | 563.60 | 159.07 | 8.11 | 2.29 |
| bidirectional D64, B = 1 | 1536 / 2048 / 2560 | inference | 2.39 / 4.98 / 9.14 | 29.89 / 53.69 / 87.81 | | 12.5 / 10.8 / 9.61 | |
| bidirectional D64, B = 1 | 1536 / 2048 / 2560 | training | 7.63 / 15.74 / 28.19 | 91.50 / 169.05 / 273.29 | | 12.0 / 10.7 / 9.69 | |
| outgoing D128 | 1536 / 2048 / 2560 | inference | 2.96 / 6.06 / 10.61 | 44.59 / 80.38 / 119.37 | | 15.0 / 13.3 / 11.3 | |
| outgoing D128 | 1536 / 2048 / 2560 | training | 12.89 / 25.30 / 43.77 | 183.86 / 337.61 / 524.41 | | 14.3 / 13.3 / 12.0 | |

OOM: the PyTorch statements and the cuEquivariance composition need more than 80 GB for D64 B = 4 training at L2560; ours peaks at 54 GB.
The served lengths are unchanged: L384 / L768, D64 (B = 1 and 4, every direction) and D128 (every direction), inference and training,
worktree / main 0.994-1.009 (two interleaved rounds on two GPUs, the record has every cell).

Tests: `tests/gpu/sm90/test_trimul_h100_long_l_gpu.py` (inference L 48 / 200 / 520 / 1040 / 1536 / 2048 / 2560, D64 B = 1 and 4,
D128, every direction, with and without a mask, against the FP32 PyTorch statement computed in row chunks; training L 1152 / 1536 /
2048 / 2560 (D64 and D128, every direction), 2816 / 3584 (D128) and 4096 (D64), every gradient, and D64 B = 4 at L1152 against the
reference and at L2560 against four one-sample calls; the B200-style port at D256 / D384 / D512 L1152 / 1536; bitwise run-to-run
determinism; CUDA-graph capture with live inputs, masks, dropout scales and weights).

## Bidirectional (`BidirectionalTriangleMultiplication`)

Outgoing + incoming in one module (h = D per direction), one output LayerNorm over 2h.

### Inference

#### K1/K3 path · D64 / D128, every L

![TriMul bidirectional inference, K1/K3 path](figures/trimul_bidir_inference_k1k3.png)

##### F1 · input LN + gated proj

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

##### F3 · output LN + proj + gate

| (Length, Dimension) | (128, 64) | (256, 64) | (384, 64) | (512, 64) | (640, 64) | (768, 64) | (128, 128) | (256, 128) | (384, 128) | (512, 128) | (640, 128) | (768, 128) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

#### Wide path · D256 / D384 / D512, L384 / L768

![TriMul bidirectional inference, wide path](figures/trimul_bidir_inference_wide.png)

##### P1 · weight pack

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F1 · input LN + gated proj

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F3 · LN-affine fold

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F4 · output LN + proj + gate

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Training

#### D64 path · L384 / L768

![TriMul bidirectional training, D64](figures/trimul_bidir_training_d64.png)

##### P1 · weight pack

| (Length, Dimension) | (384, 64) | (768, 64) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ |
| cache build | ✓ | ✓ |

##### F1 · input LN + gated proj

| (Length, Dimension) | (384, 64) | (768, 64) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ |
| cache build | ✓ | ✓ |

##### F3 · output LN + proj + gate

| (Length, Dimension) | (384, 64) | (768, 64) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ |
| cache build | ✓ | ✓ |

##### B1 · output-side backward

| (Length, Dimension) | (384, 64) | (768, 64) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ |
| cache build | ✓ | ✓ |

##### B3 · input-side backward

| (Length, Dimension) | (384, 64) | (768, 64) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ |
| cache build | ✓ | ✓ |

##### B4 · weight-grad reduce

| (Length, Dimension) | (384, 64) | (768, 64) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ |
| cache build | ✓ | ✓ |

#### D128 path · L384 / L768

![TriMul bidirectional training, D128](figures/trimul_bidir_training_d128.png)

##### P1 · weight pack

| (Length, Dimension) | (384, 128) | (768, 128) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ |
| cache build | ✓ | ✓ |

##### F1 · input LN + gated proj

| (Length, Dimension) | (384, 128) | (768, 128) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ |
| cache build | ✓ | ✓ |

##### F3 · output LN + proj + gate

| (Length, Dimension) | (384, 128) | (768, 128) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ |
| cache build | ✓ | ✓ |

##### B1 · output-side backward

| (Length, Dimension) | (384, 128) | (768, 128) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ |
| cache build | ✓ | ✓ |

##### B3 · input-side backward

| (Length, Dimension) | (384, 128) | (768, 128) |
|---|---|---|
| implementation | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ |
| cache build | ✓ | ✓ |

#### Wide path · D256 / D384 / D512, L384 / L768

![TriMul bidirectional training, wide path](figures/trimul_bidir_training_wide.png)

##### P1 · weight pack

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F1 · input LN + gated proj

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F3 · output LN

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### F5 · gate + residual

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B1 · gate backward

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B3 · output-LN backward

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B4 · contraction grads

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | cuBLAS + CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

##### B7 · input-LN backward + reduce

| (Length, Dimension) | (384, 256) | (768, 256) | (384, 384) | (768, 384) | (384, 512) | (768, 512) |
|---|---|---|---|---|---|---|
| implementation | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |
| 성능 확인 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| cache build | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |


## Single direction (`TriangleMultiplication`, outgoing / incoming)

One contraction, hidden = D (hidden = 2D at D64 / D128: [below](#hidden-twice-the-width-one-direction-d64--d128)), bf16 only. Both directions run the same kernels (`GP_DIR` / the transposed contraction operand selects the
direction). Kernel level, per width:

| Width | Inference | Training |
|---|---|---|
| D64 | K1 → cuBLAS contraction → K3 (`h100_inference`, packaged tile table) | `h100_uni_d64_b200gate` |
| D128 | K1 → cuBLAS → K3 (tile table) | fused B1, two cuBLAS gradient contractions, streamed producer-consumer B7 (`h100_single*`) |
| D256 | tmn K1 / K3 tiles at every L except 384; B200-style port at L384 | `h100_uni_wide_training` (L384 / L768), B200-style port at the other L |
| D384 | B200-style port (`mw_b200style_k1_direct`, cuBLAS, `mw_b200style_k3_static`) | `h100_uni_wide_training` (L384 / L768), B200-style port at the other L |
| D512 | B200-style port; D512 L384 / L768 may use `h100_uni_wide_inference` | `h100_uni_wide_training`: `ln_in_d512` → `front_pre_ss` (wgmma A / B from shared memory) → cuBLAS contraction → output LN → projection / gate GEMMs → output epilogue; backward gate → dn GEMM → output-LN backward → `contract_gp` → weight GEMMs → input-LN backward |

D512 training at L768 (outgoing, ms per kernel): front 2.74 + LN 0.43, `contract_gp` 2.50 (78 % of DRAM), two weight / dgrad GEMMs
1.74 + 1.76 (cuBLAS, about 70 % of the tensor peak), output-LN backward 1.24, gate 1.09, input-LN backward 0.84, output epilogue 0.82,
forward LN-out 0.47: 17.1 ms in total. Kernel flow figures for the single direction are not generated yet.

Weight pack, LN-affine fold and input LN use the CUDA aux kernels whenever the call runs eagerly, and every caller sits inside an
opaque custom op, so the P1 / F3-fold rows above are CUDA; their Triton versions only serve a `torch.compile` region that traces the
caller. No Triton kernel remains in any D256 / D384 / D512 cell at L384 / L768.

## Hidden twice the width (one direction, D64 / D128)

`TriangleMultiplication(D, d_hidden=2D)` -- the AF3 / Protenix template block (pair 64, hidden 128), and D128 hidden 256 -- runs the
bidirectional D64 / D128 kernels (it took the PyTorch reference before: the H100 dispatch required `d_hidden == D`). Per token it is
the bidirectional block: K1 writes 4D planes (`a` then `b`, 2D each), K3 normalises a 2D-channel contraction output and projects it back
to D, and the backward is the same chain (D64: B1 / B2 / fin of `h100_d64_b200gate`; D128: B1 / B7 of `h100_training`). The only
difference is the cuBLAS contraction between the kernels: all 2D channels of `a` and `b` in the module's one direction (`t = a b^T`
outgoing, `a^T b` incoming, one call each way instead of D channels per direction), and its two gradient contractions likewise
(`direction` 1 / 2 of `h100_d64_b200gate._triangle` / `_contraction_grads`, `h100_training._contract` / `_contract_grads`). Inference
is the packaged K1 / K3 unit at hidden 2D (the bidirectional one) with the one-direction contraction. Dispatch:
`trimul_h100._hidden_twice`; the chunking of B samples uses the bidirectional working set. Batched samples, ragged L in inference,
the token mask, row dropout and CUDA graphs work as for the bidirectional module; other hidden widths keep the PyTorch reference.
The hidden = D paths are bitwise unchanged (y and every gradient of the bidirectional and the one-direction modules, D64 / D128,
B 1 / 2 / 4, L 128 - 768 and ragged L200 inference, compared with `origin/main` 3d41dc3d: 54 of 54 equal). Tests:
`tests/gpu/sm90/test_trimul_h100_hidden2x_gpu.py` (y and every gradient against the FP32 PyTorch reference, both directions, D64 and
D128, B 1 / 4, L 128 / 256 / 384 and ragged L200 inference, mask, dropout, graph capture, the native entry spied on).

Measured 2026-10-21 on one H100 80GB HBM3 (132 SMs, 700 W), torch 2.13.0+cu129, bf16 pair, FP32 parameters, outgoing, a token mask
(15 % masked), row dropout 0.25 in training (fwd + bwd w.r.t. x and every parameter), CUDA-graph timing (30 replays after warm-up),
ms. cuEquivariance 0.12 is the same module with `implementation=cuequivariance` (bf16 parameters); PyTorch is the path this replaces
(the same module with the H100 contract off: the PyTorch statements under the bf16 fallback autocast). Incoming: ours within
-3 / +4 % of outgoing in every cell, the same ratios to cuEquivariance (inference 2.0 - 3.5x, training 2.0 - 3.6x).

| D | hidden | B | L | inference: ours / cuEq / PyTorch | × cuEq | × PyTorch | training fwd + bwd: ours / cuEq / PyTorch | × cuEq | × PyTorch |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 128 | 1 | 128 | 0.022 / 0.047 / 0.205 | 2.13 | 9.3 | 0.092 / 0.245 / 0.741 | 2.65 | 8.0 |
| 64 | 128 | 1 | 256 | 0.053 / 0.123 / 0.678 | 2.33 | 12.8 | 0.195 / 0.661 / 2.623 | 3.38 | 13.4 |
| 64 | 128 | 1 | 384 | 0.113 / 0.265 / 1.557 | 2.35 | 13.8 | 0.376 / 1.363 / 4.889 | 3.63 | 13.0 |
| 64 | 128 | 1 | 512 | 0.198 / 0.485 / 2.936 | 2.45 | 14.8 | 0.632 / 2.344 / 9.830 | 3.71 | 15.5 |
| 64 | 128 | 1 | 768 | 0.443 / 1.058 / 9.452 | 2.39 | 21.4 | 1.420 / 5.055 / 39.178 | 3.56 | 27.6 |
| 64 | 128 | 4 | 128 | 0.052 / 0.120 / 0.670 | 2.31 | 12.8 | 0.193 / 0.665 / 2.499 | 3.45 | 12.9 |
| 64 | 128 | 4 | 256 | 0.200 / 0.480 / 2.696 | 2.41 | 13.5 | 0.637 / 2.331 / 8.012 | 3.66 | 12.6 |
| 64 | 128 | 4 | 384 | 0.440 / 1.037 / 6.029 | 2.36 | 13.7 | 1.340 / 4.992 / 17.914 | 3.72 | 13.4 |
| 64 | 128 | 4 | 512 | 0.785 / 1.828 / 11.504 | 2.33 | 14.7 | 2.395 / 8.761 / 35.598 | 3.66 | 14.9 |
| 64 | 128 | 4 | 768 | 1.748 / 4.112 / 37.886 | 2.35 | 21.7 | 5.658 / 19.842 / 133.395 | 3.51 | 23.6 |
| 128 | 256 | 1 | 128 | 0.037 / 0.079 / 0.311 | 2.11 | 8.3 | 0.199 / 0.393 / 1.170 | 1.98 | 5.9 |
| 128 | 256 | 1 | 256 | 0.117 / 0.293 / 1.212 | 2.51 | 10.4 | 0.487 / 1.302 / 4.418 | 2.68 | 9.1 |
| 128 | 256 | 1 | 384 | 0.236 / 0.673 / 2.721 | 2.85 | 11.5 | 0.947 / 2.996 / 9.196 | 3.16 | 9.7 |
| 128 | 256 | 1 | 512 | 0.407 / 1.400 / 5.179 | 3.44 | 12.7 | 1.647 / 4.927 / 18.743 | 2.99 | 11.4 |
| 128 | 256 | 1 | 768 | 0.899 / 2.868 / 17.821 | 3.19 | 19.8 | 3.782 / 11.724 / 78.233 | 3.10 | 20.7 |
| 128 | 256 | 4 | 128 | 0.115 / 0.292 / 1.201 | 2.54 | 10.5 | 0.489 / 1.299 / 4.205 | 2.65 | 8.6 |
| 128 | 256 | 4 | 256 | 0.416 / 1.390 / 4.650 | 3.34 | 11.2 | 1.659 / 4.891 / 14.887 | 2.95 | 9.0 |
| 128 | 256 | 4 | 384 | 0.899 / 2.824 / 10.609 | 3.14 | 11.8 | 3.646 / 11.508 / 34.343 | 3.16 | 9.4 |
| 128 | 256 | 4 | 512 | 1.632 / 5.547 / 20.458 | 3.40 | 12.5 | 6.620 / 20.437 / 68.647 | 3.09 | 10.4 |
| 128 | 256 | 4 | 768 | 3.744 / 12.618 / 71.272 | 3.37 | 19.0 | 15.368 / 46.589 / 267.144 | 3.03 | 17.4 |

## Open

- Single direction in fp32 (bf16 only today); kernel flow figures for the single direction.
- D512 single-direction training at L768 (17.1 ms): front 2.7 ms (latency bound; multicast, N-split and BI=2 variants measured and
  not faster), `contract_gp` 2.5 ms (at its DRAM traffic), output-LN backward 1.2 ms (reads 0.7 GB more than needed). Each remaining
  item is 1 - 3 % of the step.
- Wide single-direction training towards 90 % of speed of light: at D384 L768 the step is at 86.5 % of the design SoL; the worst
  kernels are the front (69.6 %), the output-LN backward (65.3 %) and `contract_gp` (88 %).
- Bidirectional D512 L128 training is the thinnest cell against the other implementations (1.10x).

(Carried over from the 2026-09-30 status snapshot, [docs/records/h100_status_20260930.md](../../../records/h100_status_20260930.md).)

## Measurements (2026-10-15)

H100 80GB HBM3 (132 SMs), torch 2.13.0+cu129, `benchmarks/runners/bench.py` (compiled, manual CUDA graph), default dispatch; B=1, bf16
activations, FP32 master parameters, training = forward + backward with dropout 0.25, inference dropout 0, mask probability 0.1;
L = 128 ... 1024 in steps of 128, D = 64 ... 512. Cells show
× = fastest other implementation / ours, with that implementation in parentheses; the others are PyTorch, cuEquivariance and DTv1. Absolute
latencies, the Anthropic payload (same-process alternating A/B, D64 ... D384 single direction and D64 / D128 bidirectional, ours faster in
all 48 cells, thinnest 1.006x at D256 L768) and the Triton rows are in
[the record](../../../records/trimul_h100_comparison_20261015.md). **Ours is the fastest in all 240 cells** (smallest 1.10x).
The D256 single-direction inference cells were measured before the 2026-10-15 selection change (0.7 - 2.5 % faster now).
The 2026-10-13 tables (absolute latencies of the bidirectional module against Triton, DTv1 and cuEquivariance):
[docs/records/h100_trimul_20261013.md](../../../records/h100_trimul_20261013.md).

### Single direction, outgoing · Training

| L \ D | D64 | D128 | D256 | D384 | D512 |
|---|---|---|---|---|---|
| 128 | 1.89 (DTv1) | 1.21 (DTv1) | 1.18 (DTv1) | 1.23 (DTv1) | 1.13 (DTv1) |
| 256 | 2.26 (DTv1) | 1.66 (DTv1) | 1.43 (DTv1) | 1.38 (DTv1) | 1.27 (DTv1) |
| 384 | 2.60 (DTv1) | 1.81 (DTv1) | 1.55 (DTv1) | 1.52 (DTv1) | 1.41 (DTv1) |
| 512 | 2.69 (DTv1) | 1.84 (DTv1) | 1.47 (DTv1) | 1.41 (DTv1) | 1.27 (DTv1) |
| 640 | 2.68 (DTv1) | 1.85 (DTv1) | 1.49 (DTv1) | 1.41 (DTv1) | 1.27 (DTv1) |
| 768 | 2.65 (DTv1) | 1.85 (DTv1) | 1.54 (DTv1) | 1.48 (DTv1) | 1.36 (DTv1) |
| 896 | 2.64 (DTv1) | 1.78 (DTv1) | 1.49 (DTv1) | 1.39 (DTv1) | 1.28 (DTv1) |
| 1024 | 2.47 (DTv1) | 1.80 (DTv1) | 1.49 (DTv1) | 1.40 (DTv1) | 1.27 (DTv1) |

### Single direction, outgoing · Inference

| L \ D | D64 | D128 | D256 | D384 | D512 |
|---|---|---|---|---|---|
| 128 | 1.61 (cuEquivariance) | 1.64 (DTv1) | 1.77 (cuEquivariance) | 2.02 (DTv1) | 1.49 (DTv1) |
| 256 | 1.66 (cuEquivariance) | 1.85 (cuEquivariance) | 1.96 (cuEquivariance) | 2.11 (DTv1) | 1.75 (DTv1) |
| 384 | 1.94 (cuEquivariance) | 1.92 (cuEquivariance) | 2.06 (cuEquivariance) | 2.16 (DTv1) | 1.80 (DTv1) |
| 512 | 2.05 (cuEquivariance) | 1.98 (cuEquivariance) | 2.01 (cuEquivariance) | 2.07 (DTv1) | 1.84 (DTv1) |
| 640 | 2.03 (cuEquivariance) | 1.99 (cuEquivariance) | 2.05 (cuEquivariance) | 2.19 (DTv1) | 1.78 (DTv1) |
| 768 | 2.06 (cuEquivariance) | 1.97 (cuEquivariance) | 2.05 (cuEquivariance) | 2.25 (DTv1) | 1.81 (DTv1) |
| 896 | 2.09 (cuEquivariance) | 1.96 (cuEquivariance) | 1.97 (cuEquivariance) | 2.01 (DTv1) | 1.75 (DTv1) |
| 1024 | 1.98 (cuEquivariance) | 1.92 (cuEquivariance) | 1.96 (cuEquivariance) | 1.97 (DTv1) | 1.74 (DTv1) |

### Single direction, incoming · Training

| L \ D | D64 | D128 | D256 | D384 | D512 |
|---|---|---|---|---|---|
| 128 | 1.88 (DTv1) | 1.21 (DTv1) | 1.18 (DTv1) | 1.23 (DTv1) | 1.13 (DTv1) |
| 256 | 2.28 (DTv1) | 1.66 (DTv1) | 1.42 (DTv1) | 1.38 (DTv1) | 1.26 (DTv1) |
| 384 | 2.59 (DTv1) | 1.80 (DTv1) | 1.55 (DTv1) | 1.51 (DTv1) | 1.42 (DTv1) |
| 512 | 2.69 (DTv1) | 1.85 (DTv1) | 1.47 (DTv1) | 1.39 (DTv1) | 1.28 (DTv1) |
| 640 | 2.68 (DTv1) | 1.84 (DTv1) | 1.50 (DTv1) | 1.41 (DTv1) | 1.28 (DTv1) |
| 768 | 2.65 (DTv1) | 1.85 (DTv1) | 1.53 (DTv1) | 1.46 (DTv1) | 1.35 (DTv1) |
| 896 | 2.61 (DTv1) | 1.82 (DTv1) | 1.48 (DTv1) | 1.40 (DTv1) | 1.26 (DTv1) |
| 1024 | 2.51 (DTv1) | 1.80 (DTv1) | 1.50 (DTv1) | 1.40 (DTv1) | 1.27 (DTv1) |

### Single direction, incoming · Inference

| L \ D | D64 | D128 | D256 | D384 | D512 |
|---|---|---|---|---|---|
| 128 | 1.63 (cuEquivariance) | 1.61 (DTv1) | 1.79 (cuEquivariance) | 2.00 (DTv1) | 1.49 (DTv1) |
| 256 | 1.67 (cuEquivariance) | 1.83 (cuEquivariance) | 1.97 (cuEquivariance) | 2.09 (DTv1) | 1.76 (DTv1) |
| 384 | 1.90 (cuEquivariance) | 1.92 (cuEquivariance) | 2.05 (cuEquivariance) | 2.16 (DTv1) | 1.78 (DTv1) |
| 512 | 2.04 (cuEquivariance) | 1.99 (cuEquivariance) | 2.03 (cuEquivariance) | 2.08 (DTv1) | 1.78 (DTv1) |
| 640 | 2.03 (cuEquivariance) | 2.00 (cuEquivariance) | 2.01 (cuEquivariance) | 2.11 (DTv1) | 1.83 (DTv1) |
| 768 | 2.03 (cuEquivariance) | 1.95 (cuEquivariance) | 1.98 (cuEquivariance) | 2.04 (DTv1) | 1.83 (DTv1) |
| 896 | 2.07 (cuEquivariance) | 1.92 (cuEquivariance) | 1.96 (cuEquivariance) | 2.01 (DTv1) | 1.75 (DTv1) |
| 1024 | 2.00 (cuEquivariance) | 1.91 (cuEquivariance) | 2.04 (cuEquivariance) | 2.01 (DTv1) | 1.76 (DTv1) |

### Bidirectional · Training

| L \ D | D64 | D128 | D256 | D384 | D512 |
|---|---|---|---|---|---|
| 128 | 1.72 (DTv1) | 1.45 (DTv1) | 1.24 (DTv1) | 1.23 (DTv1) | 1.10 (DTv1) |
| 256 | 2.46 (DTv1) | 2.05 (DTv1) | 1.43 (DTv1) | 1.34 (DTv1) | 1.22 (DTv1) |
| 384 | 2.73 (DTv1) | 2.24 (DTv1) | 1.59 (DTv1) | 1.47 (DTv1) | 1.39 (DTv1) |
| 512 | 2.85 (DTv1) | 2.19 (DTv1) | 1.49 (DTv1) | 1.37 (DTv1) | 1.23 (DTv1) |
| 640 | 2.82 (DTv1) | 2.17 (DTv1) | 1.49 (DTv1) | 1.36 (DTv1) | 1.22 (DTv1) |
| 768 | 2.73 (DTv1) | 2.10 (DTv1) | 1.52 (DTv1) | 1.44 (DTv1) | 1.38 (DTv1) |
| 896 | 2.77 (DTv1) | 1.95 (DTv1) | 1.48 (DTv1) | 1.36 (DTv1) | 1.21 (DTv1) |
| 1024 | 2.69 (DTv1) | 1.94 (DTv1) | 1.49 (DTv1) | 1.23 (DTv1) | 1.24 (DTv1) |

### Bidirectional · Inference

| L \ D | D64 | D128 | D256 | D384 | D512 |
|---|---|---|---|---|---|
| 128 | 1.68 (cuEquivariance) | 1.90 (cuEquivariance) | 2.01 (DTv1) | 1.87 (DTv1) | 1.34 (DTv1) |
| 256 | 1.94 (cuEquivariance) | 2.38 (DTv1) | 2.16 (DTv1) | 1.95 (DTv1) | 1.67 (DTv1) |
| 384 | 2.23 (cuEquivariance) | 2.19 (cuEquivariance) | 2.21 (DTv1) | 1.98 (DTv1) | 1.72 (DTv1) |
| 512 | 2.37 (cuEquivariance) | 2.16 (cuEquivariance) | 2.21 (DTv1) | 1.88 (DTv1) | 1.66 (DTv1) |
| 640 | 2.32 (cuEquivariance) | 2.13 (cuEquivariance) | 2.31 (DTv1) | 1.96 (DTv1) | 1.67 (DTv1) |
| 768 | 2.31 (cuEquivariance) | 2.00 (cuEquivariance) | 2.20 (DTv1) | 1.96 (DTv1) | 1.61 (DTv1) |
| 896 | 2.33 (cuEquivariance) | 1.96 (cuEquivariance) | 2.02 (DTv1) | 1.81 (DTv1) | 1.58 (DTv1) |
| 1024 | 2.19 (cuEquivariance) | 1.93 (cuEquivariance) | 2.06 (DTv1) | 1.86 (DTv1) | 1.64 (DTv1) |
