# TriangleAttention module-level opts (beyond the attention core) — explored, all NO-SHIP

**GPU:** H100 (sm90). **Shape:** d_pair=512, L=384, n_head=4, bf16. **Base:** v7 (6.165 ms/layer, 1.78× vs the 10.98 ms baseline).

After the attention-backward campaign (v2→v7), explored the NON-attention-core parts of the
training step (LN + the 5 input projections q/k/v/gate/bias + gate + to_out ≈ 58% of the step,
of which GEMMs ~34% and pair-grad accumulation ~10%). Everything below LOST or washed; **v7
stands unchanged.**

## 1. torch.compile is a no-op on this module

Deployed regime is `model.compile()` + manual cudagraph. Compiled vs eager profile: identical
kernels, 6.172 vs 6.165 ms; compile even added a 0.5 ms DtoD memcpy. Reasons: (a) inductor has
no horizontal-matmul-fusion pass, so the 5 Linears sharing the LN'd pair stay 5 separate cuBLAS
GEMMs; (b) the module is full of `@torch.compiler.disable()` Triton kernels (attention Function,
`layernorm_kernel`, `sigmoid_gate_fused`, `fused_gate_out`) → graph breaks fragment it so
inductor can't even epilogue-fuse the pair-grad accumulation. **Eager profiling IS
representative here.** cudagraph removes only launch overhead, not kernel composition.

## 2. `layernorm_linear` (fused LN + concat projection) — 1.56× SLOWER

Wired `layernorm_linear_te_fn` over a concat `[q|k|v|gate|bias]` weight (single fused
LN+projection) into training: **9.60 vs 6.165 ms (1.56× slower)**, grad cos 0.99998→0.99393.
te_style materializes LN then a **ragged N=2052 cuBLAS GEMM** (the +4 `to_bias` tail — a clean
N=2048 is ~2.3× faster than N=2052) + a T-decomposition backward. layernorm_linear's LN-fusion
is not the lever for TA: the fold path loses in training and TE-style only ~ties TE on
*contiguous* inputs (it wins on *strided* inputs like trimul's `[B,D,L,L]`; TA's pair is
contiguous).

## 3. Plain GEMM fusion (concat q|k|v|gate, no LN fusion) — 1.01× (wash)

Concatenate the four `[512,512]` weights → one GEMM, split the output; `to_bias` (N=4) stays
separate. Honest in-proj-block microbench = 1.10× (the earlier 1.28× was optimistic — it handed
the backward a single concat-shaped grad instead of the 4 separate grads the attention+gate
actually emit, which `split`-backward must cat-scatter). **Measured end-to-end: 6.165 → 6.082 ms
= 1.014× (~1.4%, a wash)**, grad cos 0.99998→0.99447. The block's 0.23 ms GEMM-merge is offset
by (a) the `gate` split-view forcing a `.contiguous()` copy inside `_gate_out`
(`sigmoid_gate_fused`/`fused_gate_out` do `gate.reshape(-1,DH).contiguous()`), (b) the
dq/dk/dv/dgate scatter into the concat grad, (c) q/k/v read at stride 2048.

Note: any q/k/v concat needs the attention backward to write dq/dk/dv in their own grad layout
(the kernel had reused `q.stride()` for the dq write → wrong-stride write → NaN when q is a
stride-2052 split-view). A `gs_*` grad-stride group decouples it (regression-verified no-op for
separate projections); correct and reusable, but not committed — no shipping consumer.

## Verdict

All module-level, attention-core-preserving levers are exhausted with no meaningful win. The
only clean win (fold LN+5proj+attn+gate into one custom autograd Function that writes every grad
into a shared concat buffer, eliminating the scatter + the gate copy) is a module-wide rewrite
for ~4%. Not worth it. **Ship v7.** Remaining real levers: the attention algorithm itself
(FA2–4) or that full-module fusion.
