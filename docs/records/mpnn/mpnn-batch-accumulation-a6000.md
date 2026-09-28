# Actual batch versus gradient accumulation — A6000, 2026-09-12

Measured on `gpu01`, RTX A6000, at `e8b8a3ce` after the
[node-message compute optimization](mpnn-node-compute-a6000.md). The effective
batch is always **8**. This compares GPU training throughput and memory, not
convergence or validation accuracy.

## Experiment contract

- PyTorch 2.10.0+cu128, CUDA 12.8, Triton 3.6.0; FP32 parameters,
  BF16 autocast, TF32 enabled. Actual fullgraph compilation is executed and
  witnessed; CUDA Graphs are disabled for training.
- Actual batch sizes 1/2/4/8 use accumulation counts 8/4/2/1. Parameter gradients
  are cleared once at the start of the effective batch and accumulate across
  its microbatches. Every row performs forward and backward on eight examples.
- Whole model: 3 encoder and 3 decoder layers, widths 128, 48 neighbors,
  **dropout 0.25**, cross-entropy, no checkpointing, coordinate noise 0.
  Loss is the microbatch mean divided by the accumulation count. Backbone
  coordinates are fixed inputs, so only model parameters need gradients.
  Graph construction and feature extraction are included in each forward.
- Only the node-message policy changes between the model backend columns.
  Other configurable operation backends are explicitly PyTorch. The block
  threshold is fixed at 0 to keep the execution route constant across batches.
  This is not a comparison of fully optimized all-MiniWorld versus all-PyTorch
  model presets. Actual custom-forward launches and their save flags are witnessed.
- The isolated node-message experiment matches the preceding kernel evaluation:
  all three differentiable inputs and all three parameters receive gradients,
  width 128, K=48, 20% masked edges, half-local/half-random neighbors. This
  operation has no dropout. Its upstream gradient is divided by 8; parameter
  gradients accumulate, input gradients are released after each microbatch.
- Each effective batch consists of **eight copies of one synthetic example**.
  Only the current microbatch is resident as GPU input. This keeps total work
  and deterministic loss weighting identical across splits, and avoids padding
  and data-loader effects. Real heterogeneous protein batches were not tested.
- Timing includes gradient clearing and accumulation but excludes compilation,
  tuning warmup, data transfer, and the optimizer step. There would be one
  optimizer step in every configuration. Each result is the median of five
  shared-harness measurements. All GPU jobs used the same allocation serially.
- Memory is the median of three **PyTorch peak allocated** measurements after
  warmup, with gradients cleared before resetting the peak. It includes live
  microbatch inputs, parameters, gradients, saved activations, and temporaries.
  CUDA context memory and reserved-but-unused allocator blocks are excluded.
  This differs from the previous report's eager saved-storage-only metric.

## Whole-model compute-policy batch curve

| Nodes | Batch × accumulation | ms / effective batch | Examples/s | Speedup over B1 | Peak allocated MiB |
|---:|---:|---:|---:|---:|---:|
| 256 | 1 × 8 | 83.957 | 95.3 | 1.00x | 177.0 |
| 256 | 2 × 4 | 40.830 | 195.9 | 2.06x | 318.2 |
| 256 | 4 × 2 | 20.173 | 396.6 | 4.16x | 603.4 |
| 256 | 8 × 1 | 17.443 | 458.6 | 4.81x | 1169.0 |
| 2048 | 1 × 8 | 143.768 | 55.6 | 1.00x | 1175.4 |
| 2048 | 2 × 4 | 135.122 | 59.2 | 1.06x | 2313.9 |
| 2048 | 4 × 2 | 130.875 | 61.1 | 1.10x | 4598.8 |
| 2048 | 8 × 1 | 128.664 | 62.2 | 1.12x | 9153.9 |

Times are milliseconds to process all eight examples, not per-microbatch latency.
The speedup column compares each row with B1/A8 at the same node count and policy.
The results support increasing the physical batch for short inputs; the larger
input's measured curve determines how much extra memory is useful before
throughput gains become small. These sizes all fit in the A6000; the experiment
does not locate the maximum feasible batch size.

At N=256, B8/A1 gives **4.81x**
the throughput of B1/A8. At N=2048, B4/A2 takes only
**1.7%** longer
than B8/A1 while using **50.2%**
of its peak allocated memory. B2/A4 takes
**5.0%** longer
while using **25.3%**
of the memory. These are practical measured choices when memory headroom matters;
they are not changes to the model's default policy.

## Whole-model backend controls

| Nodes | Batch × accumulation | PyTorch ms | Memory ms | Compute ms | PyTorch peak MiB | Memory peak MiB | Compute peak MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1 × 8 | 77.367 | 83.315 | 83.957 | 179.8 | 170.5 | 177.0 |
| 256 | 8 × 1 | 17.342 | 17.664 | 17.443 | 1191.1 | 1121.0 | 1169.0 |
| 2048 | 1 × 8 | 144.301 | 145.845 | 143.768 | 1197.4 | 1127.3 | 1175.4 |
| 2048 | 8 × 1 | 128.617 | 130.267 | 128.664 | 9330.0 | 8769.9 | 9153.9 |

`Memory` selects `node_message_backend="triton"`; `Compute` selects
`"triton_compute"`; `PyTorch` selects `"off"`. All other model settings are held
constant. Small differences between backend columns should not be interpreted
as a general whole-model kernel speedup: most operations use the same PyTorch
path in these controls.

## Isolated node-message batch curve

| Nodes | Batch × accumulation | PyTorch ms | Memory ms | Compute ms | PyTorch peak MiB | Memory peak MiB | Compute peak MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1 × 8 | 5.503 | 7.380 | 8.094 | 33.1 | 36.5 | 33.1 |
| 256 | 2 × 4 | 2.385 | 3.533 | 3.858 | 48.6 | 55.3 | 48.7 |
| 256 | 4 × 2 | 1.187 | 1.582 | 1.719 | 79.6 | 92.8 | 79.7 |
| 256 | 8 × 1 | 1.108 | 1.249 | 0.952 | 141.6 | 167.7 | 141.6 |
| 2048 | 1 × 8 | 9.109 | 10.535 | 10.238 | 141.7 | 167.8 | 141.8 |
| 2048 | 2 × 4 | 8.861 | 10.278 | 7.551 | 266.7 | 318.8 | 266.8 |
| 2048 | 4 × 2 | 8.669 | 10.239 | 7.438 | 516.7 | 493.0 | 484.8 |
| 2048 | 8 × 1 | 8.613 | 10.254 | 7.371 | 1011.4 | 707.6 | 883.6 |

The node-message operation can show a much larger batching benefit than the
whole model. Its short-input Python/launch costs are paid repeatedly by small
microbatches. It is not valid to apply that operation-only ratio to total model
training time. At large input sizes the legacy memory policy's bounded backward
buffers can reduce peak allocation, but its latency remains a separate tradeoff.
At smaller sizes it can even peak above compute despite saving fewer tensors.

## Repeat and correctness checks

The endpoint operation cases were rerun in a fresh process with the batch and
backend orders reversed. Every raw sample is retained; host-sensitive small
inputs should be interpreted using both runs rather than a single exact ratio.

| Nodes | Backend | First B1/B8 speedup | Reversed-order repeat B1/B8 speedup |
|---:|---|---:|---:|
| 256 | pytorch | 4.97x | 4.77x |
| 256 | triton | 5.91x | 5.94x |
| 256 | triton_compute | 8.50x | 8.36x |
| 2048 | pytorch | 1.06x | 1.06x |
| 2048 | triton | 1.03x | 1.04x |
| 2048 | triton_compute | 1.39x | 1.12x |

All **36 operation cases** (24 initial, 12 repeat) passed output, parameter-gradient,
and input-gradient checks against the same eager BF16 reference. Comparing the
accumulated parameter gradient with the single-example reference verifies the
effective-batch normalization. Maximum relative L2 across these checks:
**0.005948**. Every timed case witnessed compiled execution with graphs off.

All **16 whole-model cases** produced finite losses and gradients for every model
parameter. Two additional compiled whole-model checks compare B1/A8 and B8/A1
with an eager PyTorch reference at N=256, with dropout disabled only for this
numerical comparison. Their gradient relative L2 errors are
**0.002934** and
**0.002869**; both meet the existing full-model
threshold of relative L2 < 0.02 and cosine > 0.999. Comparison reductions use FP64
to avoid FP32 summation error over the 1.66-million-element gradient vector.
Timed model dropout remains 0.25.

## Reproduction and scope

No production kernel, model default, or cache artifact changed in this experiment.
It used the current runtime cache-miss heuristic candidates; this is not an
exhaustive MPNN tile-cache qualification. The
[JSON record](mpnn-batch-accumulation-a6000.json) stores configurations, complete
experiment scripts, engine/harness and script hashes, measured compile/graph
flags, all latency and memory samples, and numerical checks. Scripts were kept
under `.scratch/mpnn-batch/`, and can be recovered from each record's
`experiment_script` for reproduction on a GPU compute node.

The practical choice is a physical batch near the measured throughput plateau,
then accumulation to reach the desired effective batch. Do not infer a universal
largest-batch rule from this GPU, uniform shapes, or synthetic repeated samples.
