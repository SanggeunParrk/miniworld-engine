# Transition module benchmark (H100, bf16)

End-to-end benchmark of the user-facing `Transition` nn.Module (LayerNorm + SwiGLU MLP,
n=4) — not the bare kernel. Pair input `(1, L, L, d)` so `M = L²`. Median ms via
`triton.testing.do_bench`, TF32 matmul on. Compared implementations:

- **PyTorch** — eager reference (`ImplementationType.PYTORCH`).
- **Triton (prev)** — the LEGACY triton path: module LN + `kernels.triton_transition` (separate
  expand kernel, `h` round-trips HBM, cuBLAS squeeze). The pre-b2b baseline we improved on.
- **Triton b2b (ours)** — the SHIPPED `ImplementationType.TRITON` path: d-aware dispatch,
  fused triton b2b (LN+expand+SwiGLU+squeeze, `h` never in HBM) at `d ≤ 128`, cute composed
  (fused LN+dual-GEMM expand + cuBLAS squeeze) at `d ≥ 256`.
- **cute (ours)** — forced `ImplementationType.CUTE` (quack SM90 WGMMA) at every d.

Regenerate: `_trans_tmp/bench_module.py` (srun, writes the CSVs) → `scripts/plot_sweep.py`
on each CSV (renders `*_speedup.png` / `*_latency.png` + the per-sweep md).

## Forward — d=128, n=4
The model's real `d_pair=128`: triton-b2b is the pick, **4.2–4.6× over PyTorch** and **~2.0× over
the previous triton** (which round-trips `h` through HBM); forced cute trails b2b here (b2b keeps
the whole SwiGLU off HBM). See [transition_module_forward.md](transition_module_forward.md).

![forward speedup](transition_module_forward_speedup.png)

## Forward + backward — d=128, n=4
Training fwd+bwd: triton b2b **~1.9–2.0× over PyTorch** and **~1.17× over the previous triton**
(smaller margin than forward — the backward is more cuBLAS-GEMM-bound). See
[transition_module_fwd_bwd.md](transition_module_fwd_bwd.md).

![fwd+bwd speedup](transition_module_fwd_bwd_speedup.png)

## Why the d-aware dispatch — crossover (forward, L=512, M=262144)

| d | n·d | PyTorch | Triton (prev) | Triton b2b (ours) | cute (ours) | dispatch picks |
|--:|--:|--:|--:|--:|--:|:--|
| 128 |  512 | 1.3460 | 0.6624 | **0.3143** | 0.4670 | triton b2b |
| 256 | 1024 | 2.4668 | 1.3080 | **1.0038** | 1.0203 | cute (`d≥256`) |
| 512 | 2048 | 5.5491 | 3.6653 | **2.8706** | 2.8816 | cute (`d≥256`) |

The previous triton is slower at every d. At `d=128` triton-b2b wins (cute slower); at `d≥256`
the shipped Triton path routes to cute (the b2b and cute columns coincide, modulo noise) because
the GEMMs turn compute-bound and quack's WGMMA pulls ahead — the win grows with d. The dispatch
always lands on the faster backend. (Kernel-level detail:
`src/miniworld_kernels/kernels/transition/benchmark/transition_fwd.md`.)
