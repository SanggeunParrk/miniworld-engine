# MPNN mixed compute/memory policy — 2026-09-12

Keeping encoder edge-tails and decoder messages on compute, while switching only
the three encoder node messages to memory, measured **523.06 ms per full
training step** and **21.940 GiB peak allocated** on the A6000 screen.
This is **1.37× throughput** versus full memory
(27.2% less step time), while retaining the requested workload.
It is the fastest passing mixture measured in this bounded screen, not a proof of
global optimality or direct A5000 performance.

**A5000 qualification is pending.** All devices were allocated. Independent job
`1680997` will compare ten candidates, recheck the top three in reverse order and
run whole-model numerical parity for its winner. The job uses a source snapshot
with the same engine/harness hash, so later worktree edits do not change it.
Existing A5000 cache builds remain running. Results will appear at
`.scratch/mpnn-mixed/a5000-1680997/summary.json` in this worktree. No A5000 latency
or final A5000 policy is claimed by this record.

## Measurements on A6000

| Policy | Allocator cap GiB (0 = uncapped) | Fwd+bwd+AdamW ms | Max allocated GiB | Max reserved GiB |
|---|---:|---:|---:|---:|
| Full compute (uncapped) | 0 | 487.34 | 26.346 | 26.658 |
| Node memory; tails + decoder compute | 23 | 523.06 | 21.940 | 22.557 |
| First two nodes + first tail memory | 22 | 568.04 | 20.815 | 21.014 |
| Full memory | 22 | 718.27 | 12.659 | 13.299 |

The three capped rows above use five fresh timing samples and ten additional
checked training steps, after two warmup steps; full compute uses three timing
samples. The shared `measured_result` harness supplies timing and executed compile
evidence. Its generic module scope label is `forward_backward`; this runner's
callable explicitly **also includes eager AdamW**, recorded as
`timing_includes_optimizer=true`. Compare these rows with each other, not directly
with earlier optimizer-free module timings.

The GPU is an RTX A6000 with 44.430 GiB CUDA-visible memory. Allocator limits screen
capacity only; they do not emulate A5000 speed, its launch selections, or all
non-PyTorch memory. A read-only A5000 inventory reported **24,564 MiB per device**;
a CUDA driver query without creating a context reported **23.5578 GiB CUDA-visible
capacity** on all six checked A5000s. The 23 GiB candidate has limited headroom,
which is why the actual-device search also includes safer alternatives.
Allocated memory measures live tensor allocations; reserved memory includes
reusable allocator blocks. Both exclude some driver/context allocations.

## Workload and executable policy

- Actual batch **8**, length **8192**, width **128**, **48 neighbors**, **3 encoder + 3 decoder** layers.
- BF16 CUDA autocast, FP32 parameters, parameter gradients and AdamW state.
- Dropout **0.25**, coordinate noise **0**, eight independent synthetic backbones;
  input coordinates do not require gradients. No accumulation or offload.
- Fullgraph `torch.compile` forward and AOTAutograd backward; training CUDA Graphs
  disabled. AdamW learning rate `1e-4`, all 118 parameter entries acquire state.
- `checkpoint_layers=False`; all W1 checkpoint policies and transition recompute
  off; feature backend PyTorch. The checkpoint API is replaced with a function
  that raises. **Zero calls**, finite gradients and finite losses in validation.
- `PYTORCH_ALLOC_CONF=expandable_segments:True`, PyTorch 2.10.0+cu128, CUDA 12.8.

```python
from miniworld_engine.modules.mpnn import ProteinMPNN, ProteinMPNNConfig

config = ProteinMPNNConfig(
    encoder_depth=3, decoder_depth=3,
    node_width=128, edge_width=128, hidden_width=128,
    k_neighbors=48, dropout=0.25, coordinate_noise=0.0,
    block_linear_min_edges=0,
    node_message_backend="triton",          # memory: all three encoder nodes
    edge_tail_backend="triton_compute",    # retain compute
    message_backend="triton_compute",      # retain compute in decoder
    edge_mlp_backend="triton_compute",
    edge_norm_backend="pytorch", edge_dropout_backend="pytorch",
    relative_position_backend="triton", feature_backend="pytorch",
    knn_backend="cdist",
    edge_w1_recompute="off", encoder_node_w1_recompute="off",
    decoder_node_w1_recompute="off", transition_recompute="off",
)
model = ProteinMPNN(config)
# Inside BF16 autocast: model(*inputs, checkpoint_layers=False)
```

The memory node kernel internally recomputes intermediates during backward.
This is the allowed kernel memory policy; no model/projection gradient checkpoint
API executes. The fused edge-tail owns its MLP, normalization and dropout;
standalone edge norm/dropout settings do not change its saved tensors.

For the conservative 22 GiB alternative, apply these overrides to the model above
before compiling. Its decoder remains compute:

```python
for index, layer in enumerate(model.encoder.layers):
    layer.node_message_backend = "triton" if index < 2 else "triton_compute"
    layer.edge_tail_backend = "triton" if index == 0 else "triton_compute"
```

Requested policies and actual kernel launches are stored separately.
For the fast candidate, the witnessed step launched three memory node forwards,
three compute tail forwards/backwards and three decoder projection forwards.
The conservative candidate witnessed two memory nodes, one compute node, one
memory tail and two compute tails. Thus policy flags are not the only evidence.

The policy-search runner uses the shared measurement harness and remains an
experimental script, following the repository's rule against adding target-specific
Python to `benchmarks/`. Its exact validated source is embedded in the JSON record.
To reconstruct it after cloning, on a compute node in the repository environment:

```python
import json
from pathlib import Path

record = json.loads(Path("docs/records/mpnn-mixed-policy-a6000.json").read_text())
script = Path(".scratch/mpnn-mixed/measure.py")
script.parent.mkdir(parents=True, exist_ok=True)
script.write_text(record["scripts_by_sha256"][record["validated_runner_sha256"]])
```

Then, on an allocated GPU with the repository's `src` and root on `PYTHONPATH`:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
python .scratch/mpnn-mixed/measure.py \
  --node-mask 7 --tail-mask 0 --decoder-mask 0 \
  --cap-gib 23 --expected-device A5000 --out mixed-a5000.json
```

Each set bit selects a memory layer: bit 0 is the first layer, `3` is layers 1–2,
and `7` is all three. Unset bits retain compute. The runner checks actual GPU,
kernel dispatch, source identity, finite gradients and the no-checkpoint contract.
It refuses to overwrite a result, returns 2 for OOM and 1 for other failures.
GPU-independent lint/type checks passed. No engine default or kernel code changed.

## Why the first 22 GiB probe rejected the fast policy

The 22 GiB limit was a conservative screening budget, not the A5000's physical
capacity. Node-memory-only reached 21.940 GiB allocated and
22.557 GiB reserved in the successful 23 GiB run. Cold autotuning and
the shared timing harness also allocate temporary cache-flush storage. A candidate
can therefore execute warm steps and still fail a tighter timed probe. Such an
OOM is preserved with its phase and is not labeled a kernel correctness failure.
The 23 GiB candidate still needs direct A5000 validation, including context and
GPU-specific workspace requirements. The 22 GiB alternative provides more margin.

No exhaustive MPNN tuning cache was built in this experiment. Runtime warnings
reported heuristic shortlists on missing cache entries; latency reflects the
selected runtime configurations, not a claim that every tile combination was tuned.

## Whole-model numerical checks

B2/L256, the same 3+3 layers and K48, including masked residues. Compiled mixtures
are compared with eager pure PyTorch under BF16 autocast and identical weights.
Dropout is zero **only** for this arithmetic comparison; timed and full-size
validation runs retain 0.25. Host FP64 error metrics avoid overflow in reductions.
Acceptance limits: loss difference <0.02, aggregate parameter-gradient relative
L2 <0.02, gradient cosine >0.999. Per-parameter relative errors are retained in JSON.

| Node/tail/decoder memory masks | Logit relative L2 | Gradient relative L2 | Gradient cosine |
|---|---:|---:|---:|
| 3,1,0 | 0.004342 | 0.004745 | 0.999995356 |
| 7,0,7 | 0.004376 | 0.004761 | 0.999995342 |
| 3,0,7 | 0.004376 | 0.004749 | 0.999995333 |
| 7,0,0 | 0.004376 | 0.004774 | 0.999995339 |

## Complete policy screen

These are screening samples, separate from the five-sample validation rows above.
Each configuration uses a fresh process and the GPU measurements are serialized.

| Node/tail/decoder memory masks | Cap GiB | Outcome | Step ms | Peak allocated GiB |
|---|---:|---|---:|---:|
| 3/0/7 | 23 | OOM: warmup/backward | — | 22.743 |
| 7/0/0 | 23 | OK | 522.02 | 21.940 |
| 7/0/3 | 23 | OK | 532.61 | 21.558 |
| 7/0/7 | 23 | OK | 537.07 | 21.558 |
| 1/0/7 | 23 | OOM: warmup/backward | — | 22.362 |
| 3/0/3 | 23 | OOM: warmup/backward | — | 22.743 |
| 5/0/3 | 23 | OOM: warmup/backward | — | 22.743 |
| 0/0/0 | 0 | OK | 487.34 | 26.346 |
| 0/0/7 | 22 | OOM: warmup/forward | — | 21.364 |
| 0/7/0 | 22 | OK | 661.55 | 18.565 |
| 0/7/7 | 22 | OK | 679.82 | 17.065 |
| 7/0/0 | 22 | OOM: warmup/backward | — | 21.659 |
| 7/0/7 | 22 | OOM: timing/backward | — | 21.808 |
| 7/7/0 | 22 | OK | 700.92 | 14.159 |
| 7/7/7 | 22 | OK | 713.27 | 12.659 |
| 0/1/7 | 22 | OOM: warmup/backward | — | 21.237 |
| 0/3/0 | 22 | OK | 610.21 | 21.159 |
| 0/5/0 | 22 | OK | 608.50 | 21.159 |
| 0/6/0 | 22 | OK | 609.67 | 21.159 |
| 1/1/7 | 22 | OOM: warmup/backward | — | 21.618 |
| 3/1/0 | 22 | OK | 568.73 | 20.815 |
| 3/2/0 | 22 | OK | 574.49 | 20.815 |
| 3/4/0 | 22 | OK | 569.50 | 20.815 |
| 7/1/0 | 22 | OK | 580.33 | 19.346 |

The [JSON record](mpnn-mixed-policy-a6000.json) preserves every sample, OOM phase,
resolved policy, memory repetition, numerical result and exact script. Scripts
are deduplicated by SHA256. The queued A5000 driver, batch script, source snapshot
manifest and output path are included so the outstanding qualification is reviewable.
