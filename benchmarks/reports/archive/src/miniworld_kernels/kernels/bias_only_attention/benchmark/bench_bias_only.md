# Bias-only triangle attention — optimization

`TriangleAttention(use_self_attention=False)`: attention weights come only from a
learned bias projection (no Q/K).

    out[b,h,i,j,d] = sum_k softmax_k(bias[b,h,j,k]) * value[b,h,i,k,d]

## What we found (H100, bf16, B=1, d_pair=128, H=4, D=32)

**The attention op itself is already optimal and should not be hand-written.**
`softmax(bias)` does not depend on `i`, so `torch.einsum("bhjk,bhikd->bhijd", A, V)`
lowers (via opt_einsum) to a single big GEMM per `(b,h)` — `A[Lj,Lk] @ Vperm[Lk, Li·D]`.

Op-level micro-bench (forward median ms), candidates:

| L | torch_einsum | triton_flash (vendored) | triton_fused (gather) | A.unsqueeze@V |
|---|---|---|---|---|
| 256 | **0.051** | 0.061 | 0.175 | 0.201 |
| 512 | **0.168** | 0.402 | 0.915 | 1.44 |
| 1024 | **0.773** | 2.97 | 6.23 | 9.59 |

- The vendored `triton_bias_only_attention` flash kernel is a **dead end**: it tiles
  the grid over `i` and so recomputes `softmax(bias)` L times. Since `bias` is
  already a materialized `[B,H,L,L]` input, flash fusion saves no memory — only the
  redundant softmax remains. It was also never wired into the module.
- `A.unsqueeze(2) @ value` (broadcast matmul over `i`) explodes into B·H·L tiny
  `[L,L]@[L,32]` GEMMs — **6–12× slower**. Do not use this formulation.
- A custom triton GEMM reading `value` strided as `V'[k,(i,d)]` to avoid the permute
  is **8× slower** (`bench_bias_only_module` / `fused.py`): the `i`-stride is `L·D`,
  so the load is non-coalesced. torch's permute→contiguous→cuBLAS wins.

**The real cost is module-level, not the op.** Forward stage breakdown at L=1024:

| stage | ms | share |
|---|---|---|
| LayerNorm (torch) | 1.12 | 29% |
| attention (softmax+einsum) | 0.77 | 20% |
| gate (to_gate + sigmoid) | 0.64 | 17% |
| value rearrange `.contiguous()` | 0.35 | 9% |
| out rearrange `.contiguous()` | 0.39 | 10% |
| to_value / to_bias / to_out | 0.56 | 14% |

## The optimization

Changes in `TriangleAttention.forward` (TRITON impl), all gated on `L >= 384`
(`_KERNEL_MIN_L`) where the kernels' launch/dispatch cost is amortized.

**Shared (inference + training):**

1. **Drop `.contiguous()`** on the value/bias/out rearranges — the einsum consumes
   the strided views directly (folds the permute into GEMM prep), and `sigmoid_gate`
   already materializes a contiguous result. Helps/neutral everywhere; applies to all
   implementations (unconditional).
2. **`kernels.layernorm_kernel`** for the LayerNorm — this repo's own developed
   standalone LayerNorm (autograd-aware), NOT the legacy vendored `triton_layernorm`
   (which the layernorm README lists only as a baseline). Wins at large L; its
   dispatch overhead regresses at small L, hence the `L >= 384` gate.
3. **`kernels.fused_gate_out`** — fuses `sigmoid(to_gate(pair)) * out` and the
   `to_out` GEMM into one triton kernel: the gated tensor never hits HBM and the
   standalone elementwise kernel disappears. `to_gate` stays on cuBLAS (the bigger
   GEMM). Forward is the fused kernel; backward does the two GEMMs on cuBLAS plus a
   single fused elementwise pass (`_gate_bwd_elem`) — without that fused bwd pass the
   naive torch backward was a net loss.

**Inference-only** (`_bias_only_inference`, gated on `not torch.is_grad_enabled()`):

4. **`layernorm_linear` (LN + concat[value|bias|gate] projection fused)** — the
   normalized pair never materializes; one fused kernel replaces LN + 3 GEMMs. This
   wins ~1.6× on the LN+proj region for inference but its fused backward LOSES to
   (layernorm_kernel + cuBLAS) for training (the repo's separate-op backward is
   already strong), so it is restricted to the no-grad path. Uses the portable
   `layernorm_linear_triton` (no quack dependency); the cute dispatch could be faster
   on SM90 in the unified repo env.

Rejected: layernorm_linear / fused-projection in the **training** path (slower bwd);
a custom strided-gather attention kernel (8× slower); fusing the whole back into one
back-to-back GEMM (forward 1.45×, but end-to-end neutral — bwd is GEMM-bound and the
inference gate is already folded into the LN concat).

## Dispatch + cache

The three crossovers live in `dispatch.py` (not hardcoded in the module):
`use_kernels(L)` (KERNEL_MIN_L=384), `use_infer_concat(d_hidden)` (≤256), and
`gate_use_fused(d_hidden, …)` — the gate backend (fused vs split) flips with
`d_hidden` and is the real perf cliff, so it is the **static H100 default on sm90**
but **calibrated once on the real tensors and cached per-GPU** on other arches
(layernorm `dispatch_cache.py` convention; `MINIWORLD_BIASONLY_AUTOTUNE=auto|off|force`,
only ever picks among correct backends). The inference concat weight `[Wv|Wb|Wg]` is
cached on the module (`_inproj_weight`, keyed on the params' version counters) instead
of a per-forward `torch.cat`.

The backward of `fused_gate_out` fuses the dgrad GEMM (`do@wo`) with the gate-backward
epilogue (`_dgrad_epi`): d_a never materializes, gate/out read once; only the wgrad
(`do^T @ gated`) stays on cuBLAS. This lifted the gate-region fwd+bwd from 1.5× to
1.71× over the earlier "cuBLAS d_a + separate elementwise" backward.

## Result — triton path vs pytorch baseline (both with the `.contiguous()` fix)

Correctness: output / d_pair-grad / weight-grad / inference cosine = **0.99999–1.00000**
at all L (the fused kernels compute in fp32 then cast, hence 0.99999 in bf16).
self-attention path (`use_self_attention=True`) also verified: cos 0.99997, no regression.

| L | inference | forward+backward | path |
|---|---|---|---|
| 128 | 1.03× | 1.00× | torch fallback |
| 192 | 1.00× | 1.00× | torch fallback |
| 256 | 1.01× | 1.00× | torch fallback |
| 384 | 1.41× | 1.05× | kernels |
| 512 | **1.50×** | **1.35×** | kernels |
| 768 | **1.48×** | **1.35×** | kernels |
| 1024 | **1.46×** | **1.35×** | kernels |

Inference gets the extra LN+projection fusion (`layernorm_linear`); training shares
everything except that. vs the **original** team-gm code (`.contiguous()` + torch LN),
the gap is larger still.

Region micro-benches at L≥384: fused gate+out — forward 1.7–2.4×, fwd+bwd 1.5×
(`bench_gate_out.py`); LN+projection — inference 1.6× (`bench_ln_proj.py`, where it
also confirms the training-bwd regression).

![speedup](bench_bias_only_module.png)

## Reproduce (compute node only — never the login node)

    srun --partition=h100 --account=cssb --qos=cssb_h100 --gres=gpu:h100:1 \
         --cpus-per-task=8 --mem=64G --time=00:20:00 \
         bash -c 'ENV=/home/psk6950/team-gm/.pixi/envs/default; \
           export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH PYTHONPATH=src; \
           cd /home/psk6950/miniworld-kernels; \
           "$ENV/bin/python" src/miniworld_kernels/kernels/bias_only_attention/benchmark/bench_bias_only_module.py'

- `bench_bias_only_module.py` — canonical: triton vs pytorch, inference + fwd+bwd correctness + timing.
- `bench_op_breakdown.py` — per-operation inference vs fwd+bwd timing (optimized path).
- `bench_gate_out.py` — fused sigmoid-gate + to_out micro-bench (fwd+bwd correctness + timing).
- `bench_ln_proj.py` — LN+projection fusion (layernorm_linear) inference vs fwd+bwd.
- `bench_bias_only.py` — op-level candidate comparison (einsum / flash / fused).
- `bench_module_breakdown.py` — per-stage forward breakdown.
- `bench_module_variants.py` — what-if probes (triton_ln / no_contig / fused_proj).
