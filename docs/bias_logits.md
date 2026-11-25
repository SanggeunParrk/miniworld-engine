# Pair-bias logit precision of the bias-only token DiT

`TokenDiTMiniWorldBiasOnly` computes its attention weights as `P = softmax(to_bias(LN(pair)))`. The logits are 1e3 .. 2e5 with a key spread of about
500, so their precision decides which keys the attention picks. Two modes, one switch (`foldjet_engine.bias_logits`).

| mode | logits | softmax | P | where |
|---|---|---|---|---|
| `"fp32"` (default) | LN0 fp32, weights `W diag(gamma)` unrounded fp32, fp32 products, fp32 store | fp32 | rounded to the activation dtype | training and inference, every backend and card (H100 / A100 / RTX 5090 `cuda`, `triton`, `pytorch`) |
| `"bf16"` (opt-in) | LN0 and `W diag(gamma)` rounded to bf16, bf16 GEMM with fp32 accumulation, bf16 store | fp32 (over the bf16 values) | bf16 | inference only, bf16 activations |

`"fp32"` is the behaviour of engine 3.0.6 (user rule 2026-11-19: bf16 logits changed which keys the attention picked and broke peptide C-N bonds
when the model was trained and run that way). `"bf16"` is the engine's earlier inference path, kept as a faster / smaller option for a model trained
with fp32 logits, whose inference showed no problem with bf16 logits (user experiment 2026-11-21; no model-level validation is part of the engine).

## The switch

```python
block.set_pair_bias_logits("bf16")                 # one TokenDiTMiniWorldBiasOnly (or BiasOnlyAttention); None = follow the process default
foldjet_engine.bias_logits.set_default("bf16")   # every module that has not set its own; None = back to the environment
MINIWORLD_BIAS_LOGITS=bf16 python run.py            # environment default, read at every call (fp32 | bf16; empty = unset)
with foldjet_engine.bias_logits.scope("bf16"): ...
```

Precedence: the module's own setting, else `set_default`, else `MINIWORLD_BIAS_LOGITS`, else `"fp32"`.

Rules (`bias_logits.effective`):

* **Training is always fp32.** A call with autograd on (grad enabled and an input or a parameter requires grad) runs the `fp32` mode silently whatever
  the switch says; so does a `float32` block (fp32 activations keep fp32 logits). Only bf16 activations (bf16 modules, or fp32 master weights under
  bf16 autocast) without a gradient take `"bf16"`.
* `pair_bias_all(blocks, pair)` follows the first block's mode: without autograd and a bf16 pair in the bf16 mode it returns bf16 biases (the
  consumers' softmax is fp32 on them); otherwise fp32 biases as before.
* Every cache that holds a P or a bias is keyed by the mode (the H100 inference P cache, the hoisted bias), so switching at run time never serves
  the other mode's tensors.

## Implementation per backend

| backend / card | `"bf16"` mode |
|---|---|
| H100 `cuda` | `FusedBiasOnlyDiT.hoist(..., logits_bf16=True)` -> `hoist_bf16`: the training rows' one-pass `pair_bias` (LN on chip, bf16 mma, bf16 store; 12 / 16 / 24 heads, pair 128 / 256) else LayerNorm rows + one cuBLAS bf16 GEMM; then `softmax_hoist` in place (fp32 softmax over the bf16 logits). `pair_bias_all` -> `bias_only_dit_hoist.pair_bias_all_bf16` |
| A100, RTX 5090 `cuda` | `pair_bias_generic(..., out_dtype=None)` stores bf16 (the same kernel as the fp32 mode, bf16 instantiation), then the row softmax (bf16 in, fp32 inside, bf16 P) |
| `triton` | `_ln_bias_fwd_kernel` with `BF16_LOGITS=True` (bf16 operands, fp32 accumulate, bf16 store; the default branch is the unchanged fp32 `ieee` dot) |
| `pytorch` | `attention.bf16_pair_bias_logits`: LN0 rounded to bf16, folded weights rounded to bf16, one bf16 matmul stored bf16; softmax fp32 |

No kernel source changed: the mode reuses kernels that already ship (the prebuilt plan is unchanged), and the `fp32` mode runs the same code as 3.0.6
(outputs bit-identical, so no module `version` and no `backend_versions` change).

Tests: `tests/gpu/test_bias_only_logit_modes.py` (per card). Speed and memory per mode and card: `benchmarks/bias_logits_modes.py`.

## Measured (2026-11-21, `benchmarks/bias_logits_modes.py bench`)

Same process, interleaved A-B-A-B, CUDA-graph replay with a 256 MB L2 flush, median; bf16 activations, one block, A = 1. `pbuild` = the logits + softmax
that make P of one block (the part the mode changes), time in microseconds and peak memory above what is held before the call (MiB), `fp32 -> bf16`.

| card | cond/pair | L 384 | L 768 | L 2048 | L 3072 | L 4096 |
|---|---|---|---|---|---|---|
| H100 | 384 / 128 | 45.8 -> 35.4 (x1.29), 14 -> 6 | 123.6 -> 87.5 (x1.41), 54 -> 22 | 770 -> 538 (x1.43), 384 -> 160 | 1765 -> 1243 (x1.42), 864 -> 360 | 3383 -> 2186 (x1.55), 1536 -> 640 |
| H100 | 768 / 256 | 72.8 -> 61.6 (x1.18), 14 -> 6 | 214.6 -> 165.4 (x1.30), 54 -> 22 | 1400 -> 1094 (x1.28), 384 -> 160 | 3193 -> 2509 (x1.27), 864 -> 360 | 5788 -> 4466 (x1.30), 1536 -> 640 |
| A100 | 384 / 128 | 76.8 -> 76.8 (x1.00), 14 -> 9 | 238.6 -> 225.3 (x1.06), 54 -> 36 | 1563 -> 1502 (x1.04), 384 -> 256 | 3536 -> 3453 (x1.02), 864 -> 576 | n/a |
| A100 | 768 / 256 | 130.0 -> 129.0 (x1.01), 14 -> 9 | 410.6 -> 400.4 (x1.03), 54 -> 36 | 2868 -> 2805 (x1.02), 384 -> 256 | 6410 -> 6408 (x1.00), 864 -> 576 | n/a |
| RTX 5090 | 384 / 128 | 43.9 -> 43.0 (x1.02), 14 -> 9 | 147.5 -> 135.1 (x1.09), 54 -> 36 | 1100 -> 923 (x1.19), 384 -> 256 | 2475 -> 2065 (x1.20), 864 -> 576 | n/a |
| RTX 5090 | 768 / 256 | 73.7 -> 71.5 (x1.03), 14 -> 9 | 249.9 -> 231.4 (x1.08), 54 -> 36 | 1748 -> 1561 (x1.12), 384 -> 256 | 3911 -> 3488 (x1.12), 864 -> 576 | n/a |

The mode is clearly faster and 2.4x smaller on the H100 (the fp32 mode runs four TF32 mma products per logit on fp32 operands and writes fp32 logits; the bf16 mode one bf16 mma and a bf16 store), 1.1 - 1.2x
faster at long L on the RTX 5090, and **not faster on the A100** (x1.00 - 1.06: the sm_80 pair-bias kernel does the same FFMA arithmetic in both modes, only the store width
narrows); on the A100 and RTX 5090 it saves one third of the peak memory of the build (the bf16 instead of the fp32 logits, 576 against 864 MiB at L 3072).
The whole block (one forward with its P build, A = 1) gains the same absolute time (H100 x1.05 .. x1.43, RTX 5090 x1.01 .. x1.15, A100 x1.00 .. x1.04).
