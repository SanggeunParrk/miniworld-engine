# trimul_inproj — results

H100 80GB HBM3, B=1, D=128, bf16, **forward-only**, median ms (`triton.do_bench`).
Diagnostics under `cute/` (perf.py, e2e_perf.py, compile_bench.py, stack_test.py).

## 1. Front kernel (left/right projection) — layout & fusion

`trimul_inproj`'s fused left+right (one gated GEMM, `[B,D,L,L]` direct write) vs
the permute fallback vs tm1's two launches. All include the same torch gate.

| L    | new bdll | new fallback | tm1 2-launch | bdll/fallback | bdll/tm1 |
|-----:|---------:|-------------:|-------------:|--------------:|---------:|
| 384  |   0.136  |       0.448  |       0.145  |     3.30×     |   1.07×  |
| 512  |   0.215  |       0.985  |       0.228  |     4.58×     |   1.06×  |
| 768  |   0.433  |       4.950  |       0.462  |    11.44×     |   1.07×  |
| 1024 |   0.738  |       9.589  |       0.802  |    13.00×     |   1.09×  |

- **bdll direct write is the dominant lever (3–13× over permute).** Needs the
  in-repo `_bdll_patch` shim (stock quack rejects the M-major gated postact).
- Fusing left+right into one launch beats tm1's two launches by 6–9%.

## 2. End-to-end module (forward, eager)

Three back-half choices on the same weights:
- **current** = `_forward_cute` (tm1 2-launch → bmm → LN_out → tm2)
- **v2** = trimul_inproj (left+right fused) → bmm → LN_out → tm2  (gate stays in tm2)
- **v3** = trimul_inproj (+gate) → bmm → **layernorm_linear** + **torch** gate mul

| L    | current | v2     | v3     | v2/cur | v3/cur |
|-----:|--------:|-------:|-------:|-------:|-------:|
| 384  |  0.542  | 0.479  | 0.463  | 1.13×  | 1.17×  |
| 512  |  0.541  | 0.475  | 0.700  | 1.14×  | 0.77×  |
| 768  |  0.936  | 0.908  | 1.465  | 1.03×  | 0.64×  |
| 1024 |  1.712  | 1.676  | 3.111  | 1.02×  | 0.55×  |

- **v2 wins everywhere and is bit-exact vs current** (diff 0). Clean win = swap
  tm1's 2-launch front for the fused left+right; keep tm2 (it already fuses
  gate+proj+mul in-register, no gate materialization).
- **v3 loses at large L**: layernorm_linear needs `[M,D]` (blld), but tri is bdll
  → an explicit `bdll→blld` transpose is needed (current fuses this away via
  `layer_norm_transpose(dbn->bnd)`). The transpose blows up with L. (diff 3.1e-2
  is bf16 op-order rounding.)

## 3. Execution mode — eager vs torch.compile vs manual CUDA graph (forward)

| L    | variant | eager | compile-RO | manual graph | e/compile | e/graph |
|-----:|---------|------:|-----------:|-------------:|----------:|--------:|
| 384  | current | 0.512 |     0.283  |      0.256   |   1.81×   |  2.00×  |
| 384  | v2      | 0.462 |     0.273  |      0.245   |   1.69×   |  1.89×  |
| 512  | current | 0.536 |     0.484  |      0.423   |   1.11×   |  1.27×  |
| 512  | v2      | 0.487 |     0.465  |      0.410   |   1.05×   |  1.19×  |
| 768  | current | 0.931 |     1.047  |      0.924   |   0.89×   |  1.01×  |
| 768  | v2      | 0.909 |     1.005  |      0.897   |   0.90×   |  1.01×  |
| 1024 | current | 1.739 |     1.933  |      1.721   |   0.90×   |  1.01×  |
| 1024 | v2      | 1.687 |     1.853  |      1.675   |   0.91×   |  1.01×  |

- Small L is **launch/latency-bound** (eager 384≈512); graph removes the floor
  (2× at L=384). Large L is **compute-bound** (bmm, L³) — graph does nothing.
- torch.compile fully captures the pipeline: **graph_breaks=0, cudagraph
  recorded, no skips** (layer_norm_transpose/tm2/gemm_act are all custom_op
  registered). In **isolation** compile-RO carries a per-call cudagraph_trees
  overhead → ~eager at L=512, slower at L=1024. (`mark_static_address` did NOT
  change this — the gap is per-call bookkeeping, not input copy.)

## 4. Stack amortization — does compile match manual in a real (deep) model?

Per-layer ms for a K-deep stack of v2 layers (AlphaFold stacks many blocks).

| L    | K | eager/lyr | compile/lyr | manual/lyr | comp/man |
|-----:|--:|----------:|------------:|-----------:|---------:|
| 512  | 1 |   0.470   |    0.465    |    0.404   |  1.15×   |
| 512  | 4 |   0.492   |    0.409    |    0.400   |  1.02×   |
| 512  | 8 |   0.508   |  **0.399**  |    0.406   |  0.98×   |
| 1024 | 1 |   1.679   |    1.852    |    1.654   |  1.12×   |
| 1024 | 4 |   1.688   |    1.696    |    1.683   |  1.01×   |
| 1024 | 8 |   1.761   |  **1.671**  |    1.708   |  0.98×   |

- **compile-RO's per-layer overhead amortizes with depth → at K=8 it matches
  (even slightly beats) manual CUDA graph.** The isolated single-op regression
  (§3) is a benchmarking artifact; in a real stacked model `torch.compile(
  mode="reduce-overhead")` over the whole stack is sufficient — **no manual
  capture needed.**
- At depth, compile also beats eager (512 K=8: 1.27×; 1024 K=8: 1.05×) — the
  ~5-launches/layer floor collapses to one replay per step.

## 5. Front gated-GEMM efficiency (left+right)

Pure left+right GLU GEMM only (`x(M,D) @ b_lr(D,4D) -> glu -> (M,2D)`), kernel
time. Memory-bound: bytes ≈ M·(D + 2D)·2; floor at 3.35 TB/s.

**bdll (strided M-major write) vs blld (contiguous):**

| L | blld(ms) | bdll(ms) | bdll/blld | bdll GB/s | %roofline |
|--:|--:|--:|--:|--:|--:|
| 384 | 0.054 | 0.055 | 1.00× | 2077 | 62% |
| 512 | 0.094 | 0.095 | 1.01× | 2120 | 63% |
| 768 | 0.201 | 0.202 | 1.00× | 2245 | 67% |
| 1024| 0.374 | 0.360 | 0.96× | 2236 | 67% |

→ **bdll strided write is FREE** (TMA handles it). The GEMM is already ~67% of
HBM roofline — near-saturated for a memory-bound kernel; little headroom inside
the GEMM itself. (The 0.738ms quoted earlier for "new bdll" was the FULL
trimul_inproj incl. torch gate + cat, not the GEMM.)

**Tile sweep (18 sm90 gated configs):** quack's autotuner already picks the
optimum — autotuned ≥ any single swept config (L=1024: autotuned 0.341ms / 71%
floor vs best swept 0.363ms). Config spread up to ~2×, but no hand-tuning win.

**Simple Triton gated GEMM vs quack:**

| L | quack(ms) | triton(ms) | q/t | quack GB/s | triton GB/s |
|--:|--:|--:|--:|--:|--:|
| 384 | 0.054 | 0.114 | 0.47× | 2095 | 992 |
| 512 | 0.092 | 0.197 | 0.47× | 2185 | 1022 |
| 768 | 0.204 | 0.416 | 0.49× | 2225 | 1089 |
| 1024| 0.384 | 0.723 | 0.53× | 2096 | 1114 |

→ A naive Triton kernel (blocked matmul, K=128 single block, no TMA) hits only
~33% of peak BW; **quack is ~2× faster** via TMA + WGMMA + warp-spec + pipelining.
See `front_gemm_compare.png`.

**Techniques already in quack** (so the front GEMM is hard to beat):
warp specialization (producer TMA / consumer WGMMA, mbarrier pipeline,
`gemm_sm90.py`); sigmoid·mul fused in the epilogue (registers), overlapping the
MMA of the next tile (`gemm_act.py:222`); fast `tanh.approx.f32`-based sigmoid
(`activation.py`). It's memory-bound, so the sigmoid cost is hidden anyway.

**Verdict:** the front GEMM is already efficient (~67% roofline, autotuned, 2×
over naive Triton). The only real front lever left is **fusing LN_in into it**
(remove the separate `layer_norm_transpose` pass; ~0.15ms at L=1024) — a custom
LN-stats-in-GEMM + GLU kernel, not GEMM-internal tuning.

## Caveats / scope

- **Forward-only.** Training needs backward: the cute kernels need
  `register_autograd` (graph_breaks=0 ≠ backward exists) before a compiled
  `.backward()` works. This is the open prerequisite for training.
- `bdll_direct` requires the in-repo `_bdll_patch` shim (stock quack). Same shim
  also un-breaks tm1's existing cute path on this env.
- B=1 only.
