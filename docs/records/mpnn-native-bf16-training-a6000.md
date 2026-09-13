# A6000 native BF16 MPNN training measurement

Measured on 2026-09-12 in Slurm job **1682255**, gpu01, one allocated RTX A6000
(47.41 GiB usable), eight CPU cores, 24 GiB host memory. Job completed with exit
code 0. Each policy ran sequentially in a fresh process within the same GPU
allocation. No GPU work ran on the login node.

## Results: B8, L8192

| Implementation / policy | Median step (ms) | Peak allocated (GiB) | Peak reserved (GiB) | Speedup vs optimized PyTorch |
| --- | ---: | ---: | ---: | ---: |
| Original PyTorch naive, precision-adapted | OOM in forward | 46.88 at failure | 46.94 at failure | unavailable |
| Optimized pure PyTorch (`pytorch`) | 452.02 | 34.74 | 34.77 | 1.000x |
| MiniWorld current (`current`) | 431.48 | 21.67 | 22.35 | 1.048x |
| MiniWorld all compute, regenerate RBF (`compute`) | 427.36 | 23.14 | 23.73 | 1.058x |
| MiniWorld all compute, retain RBF (`compute-save-rbf`) | 424.62 | 25.19 | 25.84 | 1.065x |

The current policy saves 37.6% of peak allocated memory relative to optimized pure
PyTorch. Retaining RBF with all compute kernels gives 1.016x the current policy's
throughput, for an additional 3.52 GiB peak allocation. These are seven repeated
samples from one allocation, not confidence intervals across independent GPUs.

The original naive implementation failed while allocating another FP32
`(8, 8192, 8192)` tensor (2.00 GiB), after 46.88 GiB was already allocated.
Its original dense distance construction and message concatenations are retained.
There is **no B8 naive timing or naive speedup**. The optimized PyTorch control
retains production KNN-first feature construction and blockwise/reduced message
algebra, but disables MiniWorld custom kernels. It must not be labeled naive.

## Exact contract

- B8, L8192, K48, node/edge/hidden width 128, three encoder and three decoder layers.
- Ordinary parameters: 1,656,389 BF16 elements. LayerNorm affine: 4,096 FP32 elements.
  No autocast, FP32 master model, or hidden parameter copy. Gradients and AdamW
  moments use their parameter's dtype. Norm statistics, geometry/distance/RBF
  construction, and cross-entropy math use FP32; indices remain integer.
- Dropout 0.25 in training. Coordinate noise zero; synthetic, all-valid inputs,
  patch size eight, random decoding order; backbone coordinates do not require grad.
- Real `torch.compile(fullgraph=True)` forward and loss with AOT backward;
  `triton.cudagraphs=False`, no manual CUDA Graph. Every successful timing row has
  observed compiled execution, graphs disabled, and finite outputs/gradients.
- Each measured step includes zero_grad, forward, loss, backward, and eager
  AdamW (lr 1e-4). No checkpoint API calls or gradient accumulation. The current
  policy does internal kernel/RBF recomputation; it does not checkpoint layers.
- Two real optimizer warmups exclude initial compilation/tuning from timing.
  Seven shared-harness timing samples, followed by a dispatch witness and 20
  checked optimizer steps. Shared timing uses Triton CUDA events; its generic
  `measurement_scope=forward_backward` field is supplemented by the explicit
  `timing_includes_optimizer=true` and full scope in the outer record.
- Peak memory is the maximum of warmup and validation training steps, including
  optimizer state. The separate timing peak includes the harness's 256 MiB cache
  flush buffer and is stored separately. Allocated and reserved are distinct;
  neither is total device use including the CUDA context.
- PyTorch 2.10.0+cu128, CUDA 12.8, TF32 allowed, allocator
  `expandable_segments:True`, no allocator cap; OMP/MKL threads 1 and Inductor
  compile threads 4. Existing tuning cache plus the default 24-candidate fallback.

`current` overrides the first encoder node-message backend to `triton`
(recompute preactivation), with the other two at `triton_compute` (save it).
All three edge tails and message projections use compute kernels. `compute`
saves all three node preactivations. Both regenerate RBF in backward;
`compute-save-rbf` retains RBF through the PyTorch feature backend instead.
Actual dispatch was checked, including node SAVE_PREACT counts and radial backward.
The base `model_config` field alone omits the current policy's first-layer override;
use the policy plus actual dispatch evidence to reconstruct that configuration.

All policies load corresponding weights from the same seeded naive reference.
Input seed is 20260914 and initialization seed is 20260913. Per-model parameter
iteration hashes differ between reference and production layouts; numerical parity
uses the explicit parameter-layout mapping. Historical autocast measurements used
a different initialization/input experiment, so this run is not an isolated
measurement of the speed effect of switching precision.

## Numerical checks

At B2/L128, dropout zero, the precision adapter's FP32 forward is bitwise equal to
the frozen original naive model. Against native BF16 naive forward/backward, the
four compiled production policies have logit relative L2 error 0.852–0.855%,
gradient relative L2 error 0.393–0.427%, and gradient cosine at least 0.9999915.
Full-size successful policies passed all 20 additional optimizer steps with finite
loss and every parameter gradient present, finite, and of the expected dtype.
This validates execution and numerical agreement, not training convergence.

The original naive file is untouched. `mpnn_native_reference.py` adapts only norm,
projection and mask dtype boundaries around its original forward. No MiniWorld
custom kernel executes in either PyTorch control, as checked by dispatch probes.

## Records and reproduction

[Machine-readable results](mpnn-native-bf16-training-a6000.json) contain all seven
samples, individual memory checks, model configuration, dtype/compile evidence,
errors, and parity results. Raw logs and separate records:
`.scratch/mpnn-native-bf16/measure-1682255/`.

Source identity: `64903f6d1ad3189393f7e9ca4e97a2cdc4f5353fdbacd618a59578e5714c2fed`.
Base mpnn commit: `f045b823`, with the uncommitted native BF16 changes described in
[the precision record](mpnn-native-bf16.md). Main was not modified by this experiment.
Successful processes checked source identity again on completion. The runner hash
is stored in each record. Scripts are preserved under
`.scratch/mpnn-native-bf16/measurement-scripts/`.

Run only on an allocated compute node, using the repo's configured environment:

```bash
export PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
python -m benchmarks.runners.mpnn_training --verify --out NEW_PARITY.json
python -m benchmarks.runners.mpnn_training --policy current \
  --batch 8 --length 8192 --repeats 7 --validation-steps 20 --out NEW_CURRENT.json
# Other policies: naive, pytorch, compute, compute-save-rbf. Use a fresh process
# and a new output path for each; existing records are never overwritten.
```

A supplemental matched **B4/L8192** naive/current/compute-save-rbf comparison was
submitted as job **1682257**. At publication it is **PENDING (Priority)** after the
B8 allocation completed, with no B4 result yet. Its records will be under
`.scratch/mpnn-native-bf16/measure-1682257/`. B4 results, when available, must not be
reported as a B8 speedup. No A5000 native BF16 full-size fit is claimed here.

Runner SHA256: `9c5f0d85ac0f45669d5451e5177848e512521f22b5df78cd2b1acd174ff0a6c6`.
Native reference adapter SHA256: `3edb952b68aa84611634d3867ec2f34a7ea32fc94d725316a14bd8f0c4930a09`.
