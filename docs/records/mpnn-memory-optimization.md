# MPNN fixed-geometry training memory — 2026-09-12

The compute edge tail now stores its dropout decisions at one bit per element.
The opt-in `feature_backend="memory"` stores fixed atom-pair distances instead of
retaining the expanded RBF tensor for the feature projection's weight gradient.
Both changes preserve BF16 activations, FP32 parameters and gradients, dropout,
and the full 3 encoder + 3 decoder model. They do not invoke a checkpoint API.

## Results

| Policy | ms/step, including AdamW | Training max allocated GiB | Training max reserved GiB |
|---|---:|---:|---:|
| Previous mixed, same GPU recheck | 520.34 | 21.940 | 22.557 |
| Packed masks only, three memory nodes | 520.52 | 20.956 | 21.170 |
| Packed masks + RBF memory, three memory nodes | 527.79 | 18.909 | 19.119 |
| Packed masks + RBF memory, one memory node (confirmed) | 505.85 | 21.846 | 21.973 |
| Packed masks + RBF memory, all compute (uncapped A6000) | 493.88 | 23.315 | 23.943 |

The confirmed one-node-memory candidate reduces step time by **2.8%** against the previous mixed implementation remeasured on the same allocation. Its seven timing samples are followed by twenty checked optimizer steps. The previous mixed recheck uses the same counts.

Training maxima include cold warmup and additional real training steps. Timer-only peaks include its separate 256 MiB flush buffer and are recorded in JSON. The all-compute candidate still has insufficient practical headroom for an A5000; its uncapped A6000 success is not an A5000 fit result.

The early three-memory-node and all-compute screens used the identical packed-mask arithmetic before the opaque ABI names were bumped to v2. Their source patch and hashes are preserved. The mask-only run and selected-policy confirmation use the final v2 source.

## Implementation and compiler contract

At B8/L8192/K48/D128, the three compute-tail masks shrink from **1.125 GiB**
to **0.140625 GiB**. The forward's Philox offsets, seven rounds, threshold and
update scaling are unchanged. Packing happens in the output projection; backward
loads and decodes the saved words. Dropout-off retains only a placeholder.

The RBF branch previously retained `[8,8192,48,400]` BF16 (**2.34375 GiB**).
The memory backend retains `[8,8192,48,25]` FP32 distances (**0.29296875 GiB**)
and regenerates RBF values immediately before the feature weight-gradient GEMM.
The compiled distance view has a padded residue stride (1,216 floats versus
1,200 payload floats), so its actual backing storage is approximately **304 MiB**,
rather than the 300 MiB payload. This is observed storage, not a dtype estimate.
The RBF is still a full-sized transient GEMM operand; it is not a streamed or
fused RBF/GEMM kernel. Its important lifetime change is that it no longer remains
live through all six model layers. Accounting for that padding, observed forward-resident storage falls by
**3.03125 GiB** with both changes: **21.0723 → 18.0411 GiB**. The warmed step peak
similarly falls from **21.7025 → 18.6713 GiB** for the unchanged mixed policy.

A plain custom `autograd.Function` was insufficient under AOTAutograd: its common
forward/backward expression was merged and the expanded RBF was saved again.
A regression test checks actual saved shapes under fullgraph compilation. The
backward-only `mpnn_features_radial_dw_v1` boundary prevents this hoisting; its
body compiles independently with CUDA Graphs disabled, preserving pointwise
fusion and native matrix multiplication. It does not introduce a separately
autotuned Triton family or a missing `build all` key. Standard Inductor compilation
still occurs on first use of the new backward shape.

`memory` is opt-in. `auto`, `pytorch` and `recompute` keep their existing selection.
If input geometry requires gradients, `memory` uses the ordinary differentiable
feature path. The older `recompute` backend still uses the checkpoint API and was
not used in these full-size measurements. No model checkpoint, accumulation or
offload was enabled. The previously profiled edge-tail buffer-lifetime prototype
remains an experiment and is not part of this change.

## Measurement contract

These are A6000 screening measurements, not A5000 throughput claims. The workload
is actual B8/L8192, K48, all widths128, 3+3 layers, BF16 CUDA autocast, FP32
parameters/gradients/AdamW, dropout .25 and fixed input coordinates. Fullgraph
forward, AOT backward, training CUDA Graphs off. Timings include real AdamW.
The checkpoint function is replaced with a function that raises; all completed
runs record zero calls and finite losses/gradients. Each candidate runs in a fresh
process using the existing shared `measured_result` harness, followed by ten
additional optimizer steps.

The shared Triton timer allocates an additional **256 MiB L2-flush buffer**.
One-node-memory policy passes ordinary warmup under 22 GiB but fails timing at
that same allocator cap. A separate `--timing-cap-gib=22.5` allows the unchanged
timer to run; the original **22 GiB training cap is restored** before dispatch
witnessing and the additional ten training steps. Timer peaks and real-training
peaks are recorded separately. The 22 GiB timing OOM is preserved as an experimental
result, not silently reclassified as successful. An allocator cap does not account
for all driver/context memory and does not prove A5000 fit.

## Validation

- Packed decisions match original Philox draws exactly, including incomplete row
  tiles and signed high bits; backward decoding matches those same decisions.
- FP32/BF16 radial output and dW checks cover eager and fullgraph paths, strided
  weight slices, and actual saved tensor shapes.
- Fixed-coordinate and coordinate-gradient model comparisons preserve parameter
  gradients; coordinate gradients take the ordinary path.
- Full 3+3 model comparison with pure PyTorch, B2/L256, masked residues and dropout
  disabled only for numerical comparison: aggregate gradient relative L2 about
  **0.00477**, cosine **0.99999534**. Actual full-size training uses dropout .25.
- CPU suite: **2,839 passed, 63 skipped, 417 deselected**; final ABI registration recheck **14 passed**. GPU suite **60 passed**. Memcheck **10 passed, zero errors**, with the standard allocator for instrumentation; real training retains expandable segments. Lint and type checks pass.

The build plan was re-derived: **5,258 invocations, 1,008 keys, zero errors**.
Parsed keys and unit evidence are unchanged; only the source identity is refreshed.
The plan/page checks pass (**15 tests**).

The accompanying JSON preserves executable measurement scripts and source hashes.
MPNN tile choices use the existing heuristic shortlist where tuned caches are
absent; these experiments are not an exhaustive tuning-cache build.

## Compiled-cache compatibility

The packed mask changes the opaque output's shape and dtype. Reusing the original
`mpnn_edge_tail_compute_fwd_v1` name allowed a cached AOT graph to retain the old
INT8 shape: the mask-only full-model test failed with expected 402,653,184 elements
versus 12,582,912 packed words. Forward and backward now use **v2** names, so a
source upgrade cannot select those v1 cached contracts. The regression is checked
with the existing compiler cache left in place. The initial failed record is kept
alongside the corrected run. Kernel arithmetic and autotune bucket axes are
unchanged by this ABI separation.

## Apply the screened policy before compilation

```python
config = ProteinMPNNConfig(
    encoder_depth=3, decoder_depth=3,
    node_width=128, edge_width=128, hidden_width=128,
    k_neighbors=48, coordinate_noise=0, dropout=0.25,
    block_linear_min_edges=0,
    feature_backend="memory", knn_backend="cdist",
    edge_w1_recompute="off", encoder_node_w1_recompute="off",
    decoder_node_w1_recompute="off", transition_recompute="off",
    node_message_backend="triton_compute",
    edge_tail_backend="triton_compute",
    message_backend="triton_compute", edge_mlp_backend="triton_compute",
    edge_norm_backend="pytorch", edge_dropout_backend="pytorch",
    relative_position_backend="triton",
)
model = ProteinMPNN(config)
model.encoder.layers[0].node_message_backend = "triton"
# Compile only after setting the policy. Forward uses checkpoint_layers=False.
```

This is the one-node-memory candidate; setting all three encoder node backends to
`"triton"` gives the lower-memory alternative. Selection for A5000 is subject to
actual-device verification, not inferred from the A6000 allocator cap.

## A5000 follow-up

Pending job **1680997** was updated in place to `mw-mpnn-memory-opt` (90-minute limit), preserving its queue position. It now executes the verified frozen snapshot in `.scratch/mpnn-memory-opt/a5000-source`. It compares the previous full-memory/mixed baselines with six new node policies (7, 3, 1, 2, 4, 0), confirms the three fastest passing new policies in reverse order, and checks the winner against PyTorch. OOM cases are retained and skipped; other errors stop the search. Results will be written to `.scratch/mpnn-memory-opt/a5000-1680997/summary.json`. The A5000 cache builder **1680274** was not changed. **Actual A5000 qualification is pending GPU allocation.**
