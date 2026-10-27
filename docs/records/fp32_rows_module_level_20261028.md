# AugmentedAttention fp32 (H100, TF32) at module level: packed projection + core + gate (2026-10-28)

Branch `perf/final-fp32-rows`, on top of c259e7de (`fp32_rows_followup_20261027.md`). Same harness (`ladder.py`, compile + CUDA graph, median of 3, H100). Rule: CUDA / portable <= 0.95.

## Why the kernel-level glue work could not close L128

The portable rung is measured under `torch.compile`, which fuses the module's elementwise passes (sigmoid, product, adds, adaLN glue): its L128 training step is 1.50 ms against 1.65 ms eager. The CUDA rung's
glue was already one pass each, so compile did not help it (1.55 either way), and the four separate projection GEMMs, three autograd gradient adds and four weight-gradient GEMMs were common-looking costs that only the CUDA
rung's structure could remove.

## Changes (module level, `modules/augmented_attention`, `kernels/augmented_attention/cuda`)

* One autograd node `_ProjCoreGate` for `sigmoid(x Wg^T) * attention(x Wq^T + bq, x Wk^T, x Wv^T, pair)` (fp32, TF32, H100, no QK norm; `AugmentedAttentionPairBias._proj_core_gate`):
  * q | k | v | gate as ONE GEMM over the packed weight (`addmm`, one read of x);
  * the TF32 prep pass reads q / k / v in place from the packed result (`prep_k` takes row strides) and writes q^T / k^T in the same pass in a training forward only;
  * the output gate `sigmoid(g) * o` is one kernel each way (`gate_fwd_k`, `gate_bwd_k`) instead of sigmoid + product + three backward passes; the second gate (`to_scale`) uses the same op;
  * custom ops receive the packed contiguous matrix and slice columns inside (no strided view for `torch.compile` to copy);
  * backward: `gate_bwd_k` writes d gate, and the attention backward kernels (`attn_tf32_dqb` / `dkv`, which gained a row-stride argument) write dQ / dK / dV, straight into the column blocks of ONE [A L, 4 n]
    gradient matrix; `dx = dpack Wcat` and `dWcat = dpack^T x` are one GEMM each (four of each and three adds before); dO^T is written by the dO rounding pass; the pair backward reads the key-permuted dbias in place.
* Inference: nothing for a backward is made or kept (no q^T / k^T / saved tensors), and the packed weight / bias is built once per parameter version (`_packed_weights`: weak references to the five tensors, so a freed
  address reused by another module cannot hit; a new version replaces the entry; never filled under `torch.compile` tracing or a CUDA-graph capture).
* Fallbacks unchanged: QK norm, bf16 / autocast, `compute_dtype`, `extra_pair_bias`, shapes outside the core's contract take the previous routes (a packed projection alone when only the core is unavailable).

## Results (ms; L, A = 5 inference / 48 training)

| group | mode | L | PyTorch ms | portable ms | CUDA ms | CUDA/portable | portable/PyTorch | verdict |
|---|---|---|---|---|---|---|---|---|
| aat_fp32 | infer | 128 | 0.10461 | 0.09162 | 0.07462 | 0.815 | 0.876 | OK |
| aat_fp32 | infer | 384 | 0.34202 | 0.27232 | 0.17859 | 0.656 | 0.796 | OK |
| aat_fp32 | infer | 768 | 1.04021 | 0.73229 | 0.40707 | 0.556 | 0.704 | OK |
| aat_fp32 | train | 128 | 1.65709 | 1.50611 | 1.39392 | 0.926 | 0.909 | OK |
| aat_fp32 | train | 384 | 6.17664 | 5.16525 | 4.35005 | 0.842 | 0.836 | OK |
| aat_fp32 | train | 768 | 18.59997 | 13.6487 | 10.71254 | 0.785 | 0.734 | OK |
| aat_fp32_c768 | infer | 128 | 0.12067 | 0.10742 | 0.09446 | 0.879 | 0.890 | OK |
| aat_fp32_c768 | infer | 384 | 0.42419 | 0.36307 | 0.272 | 0.749 | 0.856 | OK |
| aat_fp32_c768 | infer | 768 | 1.35952 | 1.05318 | 0.71875 | 0.682 | 0.775 | OK |
| aat_fp32_c768 | train | 128 | 1.82976 | 1.67648 | 1.5637 | 0.933 | 0.916 | OK |
| aat_fp32_c768 | train | 384 | 6.80486 | 5.78813 | 4.95123 | 0.855 | 0.851 | OK |
| aat_fp32_c768 | train | 768 | 20.0604 | 15.09827 | 12.14504 | 0.804 | 0.753 | OK |
| aaa_fp32 | infer | 128 | 0.37773 | 0.26125 | 0.12317 | 0.471 | 0.692 | OK |
| aaa_fp32 | infer | 384 | 2.9063 | 1.76963 | 0.61749 | 0.349 | 0.609 | OK |
| aaa_fp32 | infer | 1024 | 19.75104 | 11.93824 | 3.70648 | 0.310 | 0.604 | OK |
| aaa_fp32 | train | 128 | 6.00568 | 3.6168 | 2.73206 | 0.755 | 0.602 | OK |
| aaa_fp32 | train | 384 | 49.67923 | 24.88843 | 18.91914 | 0.760 | 0.501 | OK |
| aaa_fp32 | train | 1024 |  | 171.76932 | 123.90304 | 0.721 | 0.000 | OK |

Before (c259e7de): `aat_fp32` training L128 c/p 1.017, `aat_fp32_c768` training L128 1.010, `aat_fp32_c768` inference L128 1.012 (0.98 for d_pair 128). Targets (<= 0.95) met on all three, with the L384 / L768
rows improving (training 0.91 / 0.80 -> 0.84 / 0.78 and 0.92 / 0.82 -> 0.86 / 0.80; inference 0.72 / 0.59 -> 0.66 / 0.56). Kernel-time totals of one L128 training step (profiler, compile): CUDA 1654 -> 1499 us, portable 1597 us.

## Tests (`tests/gpu/sm90/test_attention_pair_fused_gpu.py`)

* fp32 token / token256 / atom: the new node is asserted to be taken; module error vs fp64 <= 1.5x the PyTorch TF32 error (forward and every gradient);
* `test_fp32_training_step_is_bitwise_repeatable` (token, token256, atom);
* no-grad inference takes the packed route and matches the training forward to 1e-3 (the adaLN kernels of the two modes differ);
* packed weights: reused per version, rebuilt after an in-place update, never served to another module on a reused address, not filled under a recorded gradient;
* the generic pair-bias pass against the tensor-core one; the torch.compile route test.

Other suites run on this tree: the attention / DiT GPU suites (`test_augmented_attention_*`, `test_dit_extra_pair_bias_h100_gpu`, `test_h100_attention_master_gpu`, small-head, compute-memory): 122 passed, 1 failed -
`test_attention_master_compile_and_live_graph[swa]` (activation error 0.0218), which fails identically on the unmodified base `b93a655e` (the SWA window block, not touched here). CPU `tests/cpu/compile`: all pass
(after adding docstrings to the new op fakes, which `test_fake_matches_body` requires).
