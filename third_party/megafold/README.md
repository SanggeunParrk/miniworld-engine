# MegaFold (UIUC SSAIL, MIT)

* `megafold/model/FusedEvoAttention/evoattention.py` (`TritonEvoformer`): memory-efficient EvoAttention with online softmax, forward + backward, Q/K/V `[B, N, Q, H, d]`,
  additive key mask `[B, N, 1, 1, K]` (the upstream `__main__` example passes a bool: wrong), pair bias `[B, 1, H, Q, K]`. It matches TWO of our classes: the triangle-attention classes (N = rows) and AttentionPairBias (N = 1). Backend `megafold`.
  Fixed tile sizes (tuned for H200, L 384). Loaded as one source file (the package imports DeepSpeed, not needed).
* `FusedTransition` / `FusedLayernormLinear` (Triton LayerNorm + expand GEMM, SwiGLU): our `Transition` class (backend `megafold_tr`, bf16, forward + backward). Needs liger-kernel
  (fetch.sh installs the pinned wheel next to the tree). MegaFold falls back to torch for shapes outside its tuned configs: the adapter reports those `unsupported`.
