# MPNN kernel evaluation and fixes — 2026-09-12

Measured on one RTX A6000 on `gpu01`, on local `mpnn` after rebase commit
`a829b1d9`. This closes the two inherited GPU failures recorded in
[the rebase report](mpnn-main-rebase.md). It measures seven kernel families and
composite operations, not end-to-end ProteinMPNN throughput.

## Changes

1. **First-autotune bias gradient corruption.** The message backward uses an atomic
   bias accumulator. Clearing it once before an autotuner call allowed all timing
   trials to accumulate into the first real result. Added `reset_to_zero` to the
   tuner, and retained the explicit zero needed by warm-cache/single-config calls.
   The initial family probe had a maximum gradient relative L2 error of about
   5,639; after the fix it is about 0.00167. In the failing whole-model comparison,
   the separate-operation arm's affected bias gradient norm was 347.26 versus
   0.05262 in the fused arm. This was a real first-call bug, not a tolerance issue.
2. **Inference compilation.** Replaced the recursive `triton_op`/`wrap_triton`
   fake-tensor path with the repository's opaque custom-op boundary and an explicit
   FP32 output fake implementation. The existing fullgraph inference test passes.
3. **Message backward overhead.** The first weight-gradient GEMM now initializes
   the FP32 accumulator directly, removing a zero and an addition. Later chunks
   keep the same FP32 accumulation order. Three message/reused dX kernels now use
   32-bit tile indexing when addresses fit, retaining the 64-bit path otherwise.
   Single/multiple chunks and both offset widths have numerical regression tests.

No backend default, numerical tolerance, tile search space, or registry row changed.
The compile fix also exposed an existing registry-audit blind spot: this inference
kernel has two fixed A5000-derived configurations, not an autotune ladder. Its old
wrapped launch escaped the static scan. The test now explicitly records that
exception; the operation/fake names and documentation follow the common contract.
The atomic bias reduction was also compared with the existing partial-buffer plus
sum path at N=256 through 8,192. Atomic was faster at every tested size, so it stays.

## Measurement contract

The new `benchmarks.runners.mpnn_compare` runner uses the existing
`measured_result`/`compile_for_benchmark` implementation. Compilation is actually
executed and witnessed in each result, rather than inferred from a requested flag.
Source and runner hashes are recorded and checked during the sweep.

- PyTorch 2.10.0+cu128, Triton 3.6.0; batch 1, width 128, 48 neighbors.
- BF16 activations and FP32 parameters with BF16 autocast. LayerNorm input is FP32;
  relative-position indices are INT64 and its table is FP32. TF32 is enabled.
- Inference: fullgraph-compiled forward with manual CUDA Graph replay.
- Training: fullgraph-compiled forward and autograd backward, fresh gradients,
  CUDA Graphs disabled. No optimizer step. Every differentiable input/parameter
  of the measured operation participates in backward.
- Training dropout is 0.25 for edge-tail and edge-dropout. Edge-tail receives its
  seed through the kernel API; seed generation is outside this kernel timing.
- Message edge masks discard 20%; half of the 48 neighbor slots are local and half
  random. These are explicit synthetic inputs, not a protein dataset.
- Timing excludes compile/autotune warmup. Each table cell is the median of three
  shared-harness timing results. Numerical comparison uses dropout 0 so unrelated
  random masks do not masquerade as arithmetic error; dropout behavior is also
  covered by the existing GPU suite.
- Inference dropout is the identity and compressed-save LayerNorm delegates to
  PyTorch in inference; these are marked not applicable, not acceleration wins.

## Final N=2048 results

| Operation / backend | Inference ms | PyTorch / backend | Training ms | PyTorch / backend |
|---|---:|---:|---:|---:|
| message / pytorch | 0.240640 | 1.00x | 0.701440 | 1.00x |
| message / triton_compute | 0.115712 | 2.08x | 0.553984 | 1.27x |
| message / triton_memory | 0.115712 | 2.08x | 0.697344 | 1.01x |
| edge_mlp / pytorch | 0.361472 | 1.00x | 1.092608 | 1.00x |
| edge_mlp / triton_compute | 0.269312 | 1.34x | 0.904192 | 1.21x |
| edge_mlp / triton_memory | 0.201728 | 1.79x | 0.989184 | 1.10x |
| edge_tail / pytorch | 0.596992 | 1.00x | 2.262016 | 1.00x |
| edge_tail / triton_compute | 0.443392 | 1.35x | 1.689600 | 1.34x |
| edge_tail / triton | 0.375808 | 1.59x | 3.374080 | 0.67x |
| edge_layernorm / pytorch | 0.173056 | 1.00x | 0.481280 | 1.00x |
| edge_layernorm / memory | n/a | n/a | 0.436224 | 1.10x |
| node_message / pytorch | 0.355328 | 1.00x | 1.131520 | 1.00x |
| node_message / triton | 0.210944 | 1.68x | 1.314816 | 0.86x |
| relative_position / pytorch | 0.017408 | 1.00x | 0.159744 | 1.00x |
| relative_position / triton | 0.017408 | 1.00x | 0.090112† | 1.77x† |
| relative_position / index_add | 0.017408 | 1.00x | 0.151552 | 1.05x |
| edge_dropout / pytorch | n/a | n/a | 0.219136 | 1.00x |
| edge_dropout / bitpack | n/a | n/a | 0.224256 | 0.98x |

The compute and memory names are backend policies. Message inference uses the same
no-grad kernel for both policies; differences between those inference measurements
are run-to-run variation. `edge_tail/triton` is its recomputation/memory policy.

† Relative-position is variable without CUDA Graphs. Its correctness-fixed
baseline measured 0.163840 ms, while the final sweep measured 0.090112 ms despite
no source change to this family. Two additional fresh processes, five samples each,
gave the following medians:

| Independent process | PyTorch ms | Triton ms | PyTorch / Triton |
|---|---:|---:|---:|
| 1 | 0.155648 | 0.151552 | 1.03x |
| 2 | 0.156672 | 0.089088 | 1.76x |

The second process itself ranged from 0.089088 to 0.154624 ms. Thus the isolated
1.77x result is not a stable speedup claim or an effect of this patch. Host-side
execution variability is a possible contributor; this experiment did not isolate
its cause. Both repeats retain correct gradients and observed compilation with
graphs disabled. Their full samples are preserved in the JSON record.

## Effect of the performance patch

These before/after measurements both include the correctness and compile fixes;
only the indexing/accumulation changes differ. At N=2048 and N=8192, message
training improves by about 2% on the compute policy and 3% on the memory policy.
The PyTorch control at those sizes changes by less than 0.2%.

| N | Backend | Before ms | After ms | Before / after |
|---:|---|---:|---:|---:|
| 256 | pytorch | 0.228352 | 0.301568 | 0.757x |
| 256 | triton_compute | 0.494592 | 0.504832 | 0.980x |
| 256 | triton_memory | 0.625664 | 0.560128 | 1.117x |
| 1024 | pytorch | 0.367616 | 0.366592 | 1.003x |
| 1024 | triton_compute | 0.492544 | 0.481280 | 1.023x |
| 1024 | triton_memory | 0.560128 | 0.606208 | 0.924x |
| 2048 | pytorch | 0.704512 | 0.703488 | 1.001x |
| 2048 | triton_compute | 0.568320 | 0.556032 | 1.022x |
| 2048 | triton_memory | 0.720896 | 0.699392 | 1.031x |
| 8192 | pytorch | 2.693120 | 2.686976 | 1.002x |
| 8192 | triton_compute | 2.226176 | 2.182144 | 1.020x |
| 8192 | triton_memory | 2.855936 | 2.771968 | 1.030x |

At N=256/1024, custom message training is still slower than compiled PyTorch.
Short-input samples also vary materially; this patch does not establish a reliable
small-input speedup. No shape-dependent default was inferred from those samples.

## Memory-policy assessment

| Operation | Backend | Saved backing storage, MiB |
|---|---|---:|
| message | pytorch | 72.406 |
| message | triton_compute | 48.438 |
| message | triton_memory | 24.438 |
| edge_mlp | pytorch | 96.062 |
| edge_mlp | triton_compute | 48.125 |
| edge_mlp | triton_memory | 24.125 |
| edge_tail | pytorch | 181.595 |
| edge_tail | triton_compute | 108.938 |
| edge_tail | triton | 25.939 |
| edge_layernorm | pytorch | 48.751 |
| edge_layernorm | memory | 24.750 |
| node_message | pytorch | 97.188 |
| node_message | triton | 26.250 |
| relative_position | pytorch | 0.750 |
| relative_position | triton | 0.750 |
| relative_position | index_add | 0.750 |
| edge_dropout | pytorch | 12.000 |
| edge_dropout | bitpack | 1.500 |

These are unique backing storages observed with eager autograd
`saved_tensors_hooks`, including saved inputs and weights. They measure retained
backward state, not compiled peak VRAM or total model memory. In particular, the
bitpacked dropout benefit is mask storage; it is not a throughput claim.

Node-message backward recomputes its projections/GELUs, computes weight gradients,
and scatters neighbor gradients; its memory saving has a runtime cost here.
Edge-tail's compute policy saves projection results; its memory policy replays the
chain during backward. Fewer saved tensors do not by themselves make training
faster. Packed dropout offers no latency gain here. Relative-position training
has substantial timing variability (see the repeated measurements above), so a
reliable large speedup is not established.

## Validation and scope

- Entire MPNN suite on A6000: **233 passed, 1 skipped**, including the two formerly failing tests and five new regressions.
- Entire non-GPU suite: **2,839 passed** (GPU-only checks skipped/deselected).
- Ruff (`src tests benchmarks`) and ty (`src benchmarks/runners/mpnn_compare.py`) passed.
- Full-repository ty still has inherited test typing diagnostics from the rebase; this is not a claim of clean full CI.
- All 33 applicable final benchmark rows passed the declared numerical checks and recorded executed compilation; 3 rows are not applicable.
- Dispatch derivation: 5,258 invocations, zero errors, 1,008 required keys; only the source identity changed in the parsed evidence. Plan/HTML checks: 18 passed, 3 skipped.
- Final sweep maximum relative L2 error: forward 0.004134 (0.413%), gradient 0.006313 (0.632%) against the BF16 PyTorch operation chains.

The MPNN driver plan still contains 88 build units: 22 kernels at
1,024/2,048/4,096/8,192 nodes. This task used runtime tuning and did not build or
publish an exhaustive MPNN cache grid. There are no shipped MPNN cache JSONs on
this branch. The refreshed 1,008-key main-module plan is separate evidence; it is
not an MPNN cache-completion claim. In particular, the fixed-policy message
inference kernel is outside those 22 registered MPNN kernels. Other GPUs and
whole-model performance remain outside this measurement.

Reproduce the latency sweep on a GPU compute node with the repository environment:

```bash
python -m benchmarks.runners.mpnn_compare \
  --nodes 2048 --repeats 3 --out /path/to/new-results.json
python -m benchmarks.runners.mpnn_compare \
  --families message --modes training --nodes 256 1024 2048 8192 \
  --repeats 3 --out /path/to/new-message-results.json
python -m pytest -q tests/mpnn
```

The [machine-readable record](mpnn-a6000-optimization.json) contains the final
samples and observed compile/graph flags, the correctness-fixed baseline,
message before/after samples, saved-storage measurements, and the bias-reduction
experiment. The interrupted overlapping diagnostic run and the invalid initial
pilot are excluded from the performance record.
