# MPNN node-message compute policy — 2026-09-12

Follow-up to [the seven-family evaluation](mpnn-a6000-optimization.md), based on
`e7030495` on local `mpnn`. The previous node-message policy traded training speed
for saved memory. A new explicit `triton_compute` policy keeps two BF16 projection
results and removes their backward replay. The existing `triton` memory policy
and the model's `off` default remain available and unchanged.

## Implementation and use

The fused forward optionally writes its preactivation and second projection to
saved buffers. Backward reuses the message family's GELU/reduction and chunked
weight-gradient kernels, with native matrix products and embedding reduction
visible to AOTAutograd. The new path computes the first weight gradient in one
matrix product. Inference uses the same no-save fused forward as the memory
policy. This adds no new kernel registry rows or tuning-grid axes.

Select `ProteinMPNNConfig(node_message_backend="triton_compute", ...)` for the
model, or `node_message_reduce(..., backend="triton_compute")` for the operation.
The model's existing `dense_training` route bypasses fused node-message policies;
the full-model regression explicitly selects the block route and witnesses the
compute-forward call. `encoder_node_w1_recompute` must be `"off"` with either
fused policy, since both own that projection.

The forward autotune key now includes `SAVE_PREACT`. Its ordinary registry driver
executes both flag values even under an outer `no_grad()` context. A GPU regression
witnesses both actual launches. Compute backward reuses existing registered message
kernels. The main dispatch plan was re-derived: 5,258 invocations, 1,008 required
keys; parsed evidence changes only its source identity. This is wiring evidence,
not a completed exhaustive cache build. MPNN still has 88 driver build units for
22 kernels at the four canonical node counts. There are no shipped MPNN cache
JSONs on this branch; measurements used runtime heuristic candidates on cache miss.

Reusing the message reduction for the node interface's general neighbor count
exposed its assumption that neighbor counts were multiples of 16. Added a tail
mask for loads/stores in the shared backward. The new K=1 regression caught this
during development; the previously supported K=48 message workload did not have
that out-of-bounds tail. K=1/48/128, repeated neighbor IDs, fully masked queries,
strided packed weights, and single/multiple gradient chunks now have regressions.

## Measurement contract

One RTX A6000 on `gpu01`; PyTorch 2.10.0+cu128, CUDA 12.8,
Triton 3.6.0. Batch 1, width 128, 48 neighbors, BF16 activations, FP32 parameters
under BF16 autocast, TF32 enabled, and 20% masked edges. Node message has no dropout
operation. Half of each neighbor list is local and half random.

The existing shared benchmark harness executes and witnesses fullgraph compilation.
Inference uses manual CUDA Graph replay; training disables CUDA Graphs and measures
forward plus backward for all differentiable inputs/parameters, with fresh gradients
and no optimizer step. Compilation/autotuning warmup is excluded. Each cell below
is the median of five samples. The final sweep ran after validation and derivation
had released their Slurm steps, without overlapping those jobs. Source and runner
hashes, all samples, observed compile/graph flags, and the earlier baseline are in
the [machine-readable record](mpnn-node-compute-a6000.json).

## Training

| Nodes | Compiled PyTorch ms | Triton memory ms | Triton compute ms | PyTorch / compute | Memory / compute |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.318464 | 0.545792 | 0.690176 | 0.46x | 0.79x |
| 1024 | 0.589824 | 0.703488 | 0.713728 | 0.83x | 0.99x |
| 2048 | 1.127424 | 1.293312 | 0.962560 | 1.17x | 1.34x |
| 8192 | 4.510720 | 5.160960 | 3.773440 | 1.20x | 1.37x |

Ratios above 1 favor compute. Small-node host execution costs remain significant;
this patch does not establish a small-input speedup or an automatic policy rule.
The table measures the node-message operation, not whole-model throughput.

## Inference

| Nodes | Compiled PyTorch ms | Triton memory ms | Triton compute ms | PyTorch / compute | Memory / compute |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.047104 | 0.038912 | 0.038912 | 1.21x | 1.00x |
| 1024 | 0.182272 | 0.112640 | 0.112640 | 1.62x | 1.00x |
| 2048 | 0.353280 | 0.203776 | 0.203776 | 1.73x | 1.00x |
| 8192 | 1.436160 | 0.821248 | 0.819200 | 1.75x | 1.00x |

Both custom policies execute the identical no-save inference kernel. Differences
between their timings are measurement variation, not a compute-policy benefit.

## Saved memory

| Backend | Unique saved backing storage, MiB |
|---|---:|
| pytorch | 97.188 |
| triton | 26.250 |
| triton_compute | 73.250 |

At N=2048, eager `saved_tensors_hooks` counts unique backing storages, including
saved inputs and weights. These numbers describe retained backward state; they
are not compiled peak VRAM or whole-model memory. The compute policy explicitly
spends memory to reduce replay work, so users prioritizing memory can retain
`triton`. FP32 native neighbor scatter and separate edge-gradient GEMM variants
were also tried; they did not produce a consistent improvement at large sizes
and are absent from the final implementation.

## Validation

- Entire MPNN GPU suite: **250 passed, 1 skipped**; subsequently added fullgraph
  forward/backward parity regression: **1 passed** (251 GPU checks in total).
- Entire CPU suite: **2,839 passed, 63 skipped, 401 deselected**.
- Targeted boundary, FP64 accuracy, driver, inference-save, and full-model gradient
  checks: **21 passed**. The full-model check verifies actual compute dispatch.
- CUDA compute-sanitizer memcheck on the K=1 multi-chunk regression: **0 errors**.
- Refreshed plan and generated HTML checks: **18 passed, 3 skipped**.
- Ruff (`src tests benchmarks`) and ty (`src benchmarks/runners/mpnn_compare.py`)
  passed. Inherited full-repository test typing diagnostics remain outside that claim.
- All 24 final benchmark rows passed numerical checks and witnessed compilation.
  Maximum forward relative L2: **0.000006**; maximum gradient relative L2:
  **0.004180**, against the BF16 PyTorch operation chain.

Other GPUs, exhaustive tile-cache qualification, and end-to-end model performance
were not measured in this follow-up. Reproduce on a GPU compute node:

```bash
python -m benchmarks.runners.mpnn_compare \
  --families node_message --nodes 256 1024 2048 8192 --repeats 5 \
  --out /path/to/new-node-message-results.json
python -m pytest -q tests/mpnn
```
