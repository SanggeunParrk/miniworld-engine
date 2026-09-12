# MPNN B8/L8192 without gradient checkpointing — 2026-09-12

The full memory-policy combination completed two real training steps with a
**22 GiB PyTorch allocator cap** on an RTX A6000. This supports the expectation
that this configuration fits an A5000, whose listed capacity is
[24 GB](https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/nvidia-rtx-a5000-datasheet.pdf).
**This is not direct A5000 validation.** All A5000 devices were allocated; our
pending probe was estimated to start about four hours later and was cancelled.
Existing A5000 cache builds were left running.

## Measured results

| Configuration | Allocator cap | Outcome | Max allocated GiB | Max reserved GiB |
|---|---:|---|---:|---:|
| Full memory policy | 22 GiB | Two forward/backward/AdamW steps passed | 12.660 | 15.650 |
| Full compute policy | 22 GiB | OOM during first forward | — | — |
| Previous batch-benchmark policy | 22 GiB | OOM during first forward | — | — |
| Full compute policy | Uncapped A6000 | Two forward/backward/AdamW steps passed | 26.347 | 26.650 |

The uncapped compute requirement exceeds A5000 capacity under this measured
configuration. The full memory policy has substantial headroom in the capped
probe. The previous batch experiment changed only node-message policy while
leaving the other operation backends on PyTorch; extrapolating that configuration
does not describe the full memory-policy combination tested here.

## Meaning of checkpoint OFF

- `checkpoint_layers=False` on every model call.
- Encoder/decoder node W1 checkpoints and edge W1 checkpoints are all `"off"`.
- Transition recomputation is `"off"`; feature backend is `"pytorch"`, because
  feature `"recompute"` would call the checkpoint API.
- The experiment replaces `torch.utils.checkpoint.checkpoint` with a function
  that raises if called. **Zero calls** occurred.
- The memory kernels still recompute selected intermediates internally during
  backward. Thus the successful result means **no model/projection gradient
  checkpoint API**, not “no recomputation anywhere.” The compute preset saves
  projections instead and has the larger measured requirement.

## Exact successful configuration

Batch 8, length 8192, 3 encoder + 3 decoder layers, all widths 128, 48 neighbors,
BF16 autocast with FP32 parameters, dropout 0.25, coordinate noise 0,
cross-entropy loss, AdamW with learning rate 1e-4. Eight independent synthetic
backbones are used. Input coordinates do not require gradients. There is no
gradient accumulation, optimizer offload, or feature precomputation.
The standard `cdist` neighbor backend is retained.

```python
config = ProteinMPNNConfig(
    encoder_depth=3, decoder_depth=3,
    node_width=128, edge_width=128, hidden_width=128,
    k_neighbors=48, dropout=0.25, coordinate_noise=0.0,
    block_linear_min_edges=0,
    node_message_backend="triton",
    message_backend="triton_memory",
    edge_tail_backend="triton",
    edge_mlp_backend="triton_memory",
    edge_norm_backend="memory",
    edge_dropout_backend="bitpack",
    relative_position_backend="triton",
    feature_backend="pytorch",
    knn_backend="cdist",
    edge_w1_recompute="off",
    encoder_node_w1_recompute="off",
    decoder_node_w1_recompute="off",
    transition_recompute="off",
)
# model(*inputs, checkpoint_layers=False), inside BF16 CUDA autocast
```

The fused edge-tail route owns its edge update, including its normalization and
dropout; requesting edge-MLP/norm/dropout settings does not imply their standalone
kernels also execute in that route. Actual launches of node-message, message,
edge-tail forward/backward, and relative-position backward were witnessed in the
successful memory run. All model parameters received finite gradients and all
118 parameter entries had AdamW state after the optimizer step.

## Validation and limits

PyTorch 2.10.0+cu128, CUDA 12.8; engine/harness hash and exact
probe script are in the [JSON record](mpnn-b8-l8192-memory-fit.json). The forward
is actually fullgraph compiled, with executed-graph evidence. AOTAutograd runs
backward; AdamW is an ordinary eager optimizer step. CUDA Graphs are disabled.
The second step includes already-created optimizer state. Both steps' allocation
and reservation peaks are preserved, including cold compilation/autotuning in
the first step. These elapsed times are diagnostic and are not benchmark results.

PyTorch allocated memory measures live tensor allocations; reserved memory includes
the allocator's reusable blocks. Neither includes all CUDA-context/driver memory.
The 22 GiB cap is an allocator limit on a larger GPU, not an emulated A5000.
GPU-specific launch configurations and workspace usage still require direct A5000
qualification. This two-step check establishes finite training execution under the
tested memory budget, not a long training run or a new convergence/accuracy study.
No kernel source or default policy changed, and no exhaustive MPNN cache was built.
