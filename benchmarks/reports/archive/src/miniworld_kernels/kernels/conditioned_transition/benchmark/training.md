# ConditionedTransition training (production) vs torch.compile — H100, fp32 / TF32

_Primary baseline = `torch.compile(reference, mode="reduce-overhead")` of the pure-pytorch
reference (graph-break check on fwd: graph_count=1, break_count=0). OURS measured under our own
CUDA graph; COMPILE uses its own cudagraphs — apples-to-apples graph-captured. Eager is a
context column. Sources: `training_vs_compile_cudagraph.out`, `training_forward_ab_cudagraph.out`._

Production path `cond_transition_train` / `ConditionedTransitionTailFunction`:
- **Forward:** d/M-aware `auto` pick of {fused-triton b2b/composed forward, cuBLAS-fwd} (both
  emit the saved tensors ab,h,out,scale for backward).
- **Backward:** cuBLAS GEMMs (dgrad dh/dcond/dx + wgrad dWa/dWb/dWs/dWsc, cat-merged expand) +
  fused-triton elementwise (gate-bwd → dout,dscale in one pass; swiglu-bwd → packed dab).

## Correctness

All 7 grads (dx, dcond, dWa, dWb, dWs, dWsc, db_sc) + cos_y = **1.00000** vs autograd-through-
torch reference (TF32), every shape. PASS.

## Training fwd+bwd vs torch.compile (both CUDA graph)

| stream | M | d | cos_min | ours us | compile us | eager us | **vs compile** | vs eager |
|---|---|---|---|---|---|---|---|---|
| atom | 2048 | 128 | 1.00000 | 91.5 | 252.3 | 114.7 | **2.76×** | 1.25× |
| atom | 4096 | 128 | 1.00000 | 115.7 | 269.0 | 142.6 | **2.33×** | 1.23× |
| atom | 8192 | 128 | 1.00000 | 158.3 | 266.6 | 198.4 | **1.68×** | 1.25× |
| token | 384 | 768 | 1.00000 | 146.7 | 278.6 | 175.2 | **1.90×** | 1.19× |
| token | 512 | 768 | 1.00000 | 165.1 | 278.1 | 195.9 | **1.68×** | 1.19× |
| token | 768 | 768 | 1.00000 | 213.2 | 276.3 | 236.3 | **1.30×** | 1.11× |
| token | 1024 | 768 | 1.00000 | 227.9 | 279.2 | 261.7 | **1.23×** | 1.15× |

**Ours beats torch.compile at every shape (1.23–2.76×)** and eager (1.11–1.25×). Lowest margin is
token M=1024 (1.23× compile). Note: `compile(reduce-overhead)` on this fwd+bwd is ~250–290 us
flat — slower than even eager — i.e. its backward codegen/cudagraph overhead is high for this op;
we beat it comfortably regardless.

## Forward backend A/B (CUDA graph, identical backward)

The fused-triton forward (inference b2b/composed) was wired into training but must additionally
write the saved-for-backward tensors (ab,h,out,scale), which erodes the inference-proven fusion:
forward-only it wins at large-atom (1.14×) / small-token (1.31–1.33× vs cuBLAS-fwd) but regresses
at token≥768. The shipped `_FWD_MODE="auto"` routes to the per-regime winner (fused for atom
M≥8192 + token M≤512; cuBLAS otherwise), capturing the gain with no regression. Override via
`set_forward_mode("auto"|"cublas"|"fused")`.

## Negative result (recorded)

Folding `db_sc = dscale.sum(0)` into the gate-bwd kernel (a full-D tile + in-kernel `tl.sum`)
**regressed token 1.25→0.33× compile** — torch's `dscale.sum(0)` is a fast cuBLAS-adjacent
reduction and a wide-tile triton reduction is far worse. Reverted; kept as a comment in the code.

## Verdict

Production training (fused/cuBLAS auto-forward + cuBLAS-GEMM backward + fused-triton elementwise)
**beats the primary baseline torch.compile at every measured shape (1.23–2.76×)**, correct to
cos = 1.00000. wgrad stays cuBLAS; the all-triton-dgrad experiment (lost) is recorded separately
in `training_fused_no_cublas.md`.
