# ProteinMPNN optimization

## Numerical reference

`miniworld_kernels.modules.mpnn.NaiveProteinMPNN` is the frozen numerical oracle
for the MPNN work in this repository. It follows:

- repository: `ProteinMPNN_CSSB`
- revision: `origin/dev@4870bcaf4f55c45b5d7ee5ff8097a3ce3d020ac0`
- source files:
  - `fullmoon_initial_package/model/model.py`
  - `fullmoon_initial_package/model/modules.py`
  - `fullmoon_initial_package/model/graph.py`
  - `fullmoon_initial_package/model/initializer.py`
  - `fullmoon_initial_package/common/geometry.py`

The reference intentionally preserves the upstream parameter names, operation
order, intermediate tensor materialization, initialization, and masking rules.
Optimized implementations must not share compute helpers with it. Matching the
reference therefore remains an independent correctness signal.

The first frozen scope is the parallel `ProteinMPNN.forward` used by training.
Patch-selection, symmetry policy, and autoregressive sampling will be isolated
from the numerical model core and frozen independently.

The benchmark harness calls eval-mode parallel forward `inference` to match the
repository-wide mode vocabulary. It is a teacher-forced model-core benchmark,
not autoregressive sequence-generation latency.

## Production training workload

The frozen `origin/dev` training configuration is substantially larger than a
conventional `B=1, L<=256` model-core benchmark:

- `CROP=2048` is the maximum length of each logical training example.
- The checked-in training launcher passes logical batch size 8. Its
  `train.py::collate` concatenates those examples along length, so the model
  receives physical shape
  `[1, sum(L_i), ...]` plus `len_tensor=[L_1, ..., L_8]`, not `[8, L, ...]`.
- `K=48`, hidden/node/edge width 128, three encoder layers, three decoder
  layers, and patch size 8 are the source defaults.
- The checked-in training script sets `USE_AMP=False`, so FP32 forward/backward
  is the source-faithful primary workload. BF16 mixed precision remains a
  separate optimized-deployment track.

Consequently, a single `[1, 2048, ...]` crop is necessary but not sufficient.
The benchmark matrix keeps these cases separate:

| case | physical shape | purpose |
|---|---:|---|
| single crop | `[1, 2048, ...]` | per-example crop ceiling |
| packed batch 8 | `[1, 8*S, ...]`, `S=256/384/512` | representative source training layout |
| packed stress | `[1, 16384, ...]` | worst-case `8*2048`; memory/OOM characterization only |
| true padded batch | `[B, L_bucket, ...]` | independent graphs, item-balanced training, and the production batch path |

For an equal-length packed benchmark, `seq_len` denotes the per-example length,
`batch_size` the logical example count, and `tokens=batch_size*seq_len` the
physical length. CSV rows also record the input layout, physical batch size,
segment count/lengths, patch size, and whether numerical correctness ran in
that process. Decoding-order and patch-index offsets follow the source collate
function exactly.

## Production tensor contract

| Production input | CSSB name | Shape | Meaning |
|---|---|---:|---|
| `backbone` | `xyz` | `[B, L, 4, 3]` | N, CA, C, O backbone coordinates |
| `sequence` | `seq` | `[B, L]` | integer residue types in `[0, 20]` |
| `residue_mask` | `mask` | `[B, L]` | valid residue mask |
| `residue_index` | `residue_idx` | `[B, L]` | residue indices for relative position encoding |
| `chain_index` | `chain_idx` | `[B, L]` | chain labels |
| `decoding_order` | `decoding_order` | `[B, L]` | decoding step to residue permutation |
| `patch_index` | `patch_index` | `[B, L]` | patch id indexed by decoding step |
| `segment_lengths` | `len_tensor` | `[N_segments]` or `None` | packed segment lengths summing to `L`; `None` means one segment |

The default output is `[B, 21, L]`. With `return_log_prob=True`, it is
`[B, L, 21]`.

The production core does not accept the source's unused `loss_mask`. Code that
must retain the old positional call can wrap the model in
`CSSBForwardAdapter`; the adapter discards only that argument and delegates to
the clean core. Execution options after `segment_lengths` are keyword-only so a
legacy positional tuple cannot silently bind to the wrong option.

`segment_lengths` is a physical-`B=1` compatibility input for the source-style
packed layout. A true batch uses `B>1`, zero padding, `residue_mask`, and
`segment_lengths=None`; combining `B>1` with `segment_lengths` is rejected so a
single segment map cannot accidentally be broadcast across different items.

## True-batch training

The production training boundary is one protein or complex per batch row. A
multi-chain complex remains one graph and uses `chain_index` to distinguish its
chains; chains are not treated as separate loss items. Variable-length items
are length-bucketed and zero-padded to `[B, L_bucket, ...]`. This evaluates KNN
as `B` independent `L_bucket x L_bucket` problems instead of one global
`(sum L_i) x (sum L_i)` packed problem, and prevents edges from crossing item
boundaries by construction.

The public data and loss helpers are:

```python
from miniworld_kernels.modules.mpnn import (
    LengthBucketBatchSampler,
    MPNNTrainingSample,
    collate_mpnn_samples,
    item_balanced_cross_entropy,
)

batch = collate_mpnn_samples(samples, pad_to_length=bucket_length).to("cuda")
logits = model(*batch.model_inputs(), **batch.model_keyword_arguments())
result = item_balanced_cross_entropy(
    logits,
    batch.sequence,
    batch.loss_mask,
    residue_mask=batch.residue_mask,
    label_smoothing=0.1,
)
result.loss.backward()
```

`collate_mpnn_samples` preserves each item's decoding permutation, appends a
valid permutation for padding positions, carries a per-item fixed/motif prefix
length, and rejects items with no supervised residue. Padding coordinates and
tokens are finite zeros. The model propagates an explicit edge-valid mask
through encoder and decoder message reductions, so a short item with `L_i<K`
matches running that graph independently. The reduction divisor remains the
configured `K`, preserving the mathematical algorithm.

The objective first computes the masked mean CE within each item and then the
mean over active items. Thus a long graph does not outweigh a short graph merely
because it has more supervised residues. `loss_mask` remains separate from the
model's structural `residue_mask`; it belongs to this training utility rather
than the model forward. Logits at padded output positions are unspecified and
must never be supervised; `batch.supervision_mask` and the loss helper enforce
that boundary. For source-faithful training, callers explicitly select
`label_smoothing=0.1`. `ItemBalancedLoss.for_ddp_backward(...)` supplies the
scale needed when different ranks have different active-item counts before
standard DDP gradient averaging, while its detached statistics can be summed
for globally correct logging.

Length buckets affect padding efficiency only. They must not change sampling
weights if the dataset-level goal is equal treatment of training items. The
`LengthBucketBatchSampler` visits every index exactly once by default, groups
nearby lengths, and reshuffles deterministically after `set_epoch(epoch)`.
Pair its `bucket_width` with `collate_mpnn_samples(..., pad_to_multiple=...)` so
the DataLoader emits a small, stable set of CUDA-Graph shapes. `drop_last=False`
is the equal-coverage default; distributed callers should shard the epoch's
indices before constructing a sampler for each rank. The
current core uses fixed-K ELL-shaped neighbor tensors `[B,L,K]`, which is also
the intended interface for later custom KNN/message kernels.
Very narrow, sparsely populated buckets can produce underfilled batches; choose
the width from the observed length distribution.

### Token-budget batching

A fixed item count is the wrong knob for this model. Memory and compute are
linear in *padded* tokens -- the graph is fixed-K, so a padded residue still costs
a full 48-edge row whose result is only masked away afterwards -- and
`batch_size` has to be chosen so that `batch_size * crop_limit` fits in memory.
Every batch of short items then leaves most of the device idle.

`TokenBudgetBatchSampler` budgets padded tokens instead: items are grouped into
explicit `length_buckets` and each batch carries `token_budget // bucket` items,
so a batch of 256-residue chains holds 256 items where a batch of 2048-residue
chains holds 32. Pair it with `make_bucketed_collate_fn(length_buckets)` so the
padded width the sampler budgeted for is the width the batch receives.

```python
lengths = [sample.length for sample in dataset]
sampler = TokenBudgetBatchSampler(
    lengths,
    token_budget=32 * 2048,              # padded tokens, ~0.2 MiB each
    length_buckets=(256, 512, 1024, 2048),
)
loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    collate_fn=make_bucketed_collate_fn((256, 512, 1024, 2048)),
)
for batch_shape in sampler.shape_plan():        # pre-warm compile / CUDA Graphs
    ...
```

Static shapes are what make this free. A dataset's per-bucket item count is
fixed, so the emitted `(batch, padded_length)` set is identical in every epoch and
is reported by `shape_plan()`; shuffling changes which items share a batch, never
the shapes. `drop_last=True` leaves exactly one shape per occupied bucket. This
is the same bargain LLM pretraining strikes with packing -- fill the budget, keep
one static shape -- reached the way this model's cost function requires, since
here the padded batch *is* the block-diagonal neighbor construction. Packing into
a single `[1, sum L]` row instead makes KNN `O((sum L)^2)`: at 65,536 tokens the
distance matrix alone is 17.2 GiB against 0.54 GiB for `[32, 2048]`.

Two diagnostics are exposed rather than left to guesswork:
`budget_occupancy()` is the mean fraction of the budget each batch uses (what a
fixed count gives up) and `token_utilization()` is the fraction of budgeted
padded tokens that carry real residues (what the bucket choice costs). Wider
buckets mean fewer shapes and lower utilization.

#### Step time is linear in padded tokens, so bucket width is the only lever

Measured on an A6000 with the explicit memory preset, across ten shapes spanning
`B=12..256` and `L=256..2048`, the step time per padded token is **7.33 us with a
spread of +-1.2%**:

| Shape | Padded tokens | Step | us / token |
|---|---:|---:|---:|
| `B=32, L=512` | 16,384 | 119.88 ms | 7.32 |
| `B=32, L=1024` | 32,768 | 238.57 ms | 7.28 |
| `B=256, L=256` | 65,536 | 478.08 ms | 7.29 |
| `B=128, L=512` | 65,536 | 481.13 ms | 7.34 |
| `B=64, L=1024` | 65,536 | 482.52 ms | 7.36 |
| `B=32, L=2048` | 65,536 | 488.64 ms | 7.46 |

Two consequences, both counter-intuitive:

- **A bigger batch buys nothing by itself.** `B=128,L=512` costs 4.01x
  `B=32,L=512` for 4x the padded tokens. The device is already saturated at
  16k tokens, so filling the budget only amortizes launch overhead, which is
  negligible here. Spare device memory is therefore *not* spare throughput; a
  larger budget buys fewer optimizer steps, not a faster epoch.
- **Padding is the whole cost.** Time tracks padded tokens so exactly that the
  epoch can be predicted from the sampler alone. On a log-normal length proxy
  (n=20,000, median 248, mean 317, p90 606, 6.35M real residues):

| Bucket ladder | Shapes | Token utilization | Steps | Predicted | Measured |
|---|---:|---:|---:|---:|---:|
| `(2048,)` -- single crop | 1 | 0.155 | 625 | 300.2 s | -- |
| `(512, 1024, 1536, 2048)` | 8 | 0.526 | 186 | 88.5 s | 88.3 s |
| `(256, 512, 1024, 2048)` | 8 | 0.662 | 148 | 70.3 s | 70.5 s |
| `(128, 256, 384, 512, 768, 1024, 1536, 2048)` | 16 | 0.802 | 124 | 58.0 s | -- |
| every multiple of 64 | 51 | 0.911 | 123 | 51.1 s | -- |

Peak memory was an identical 13,094.5 MiB in every row, because peak is set by
the largest shape and every ladder ends at the same budget. The prediction column
is `7.33 us x padded tokens`; where both are available they agree to 0.3%.

So choose the ladder by how many compiled variants and CUDA Graph pools are
affordable, not by memory. Sixteen shapes buy 1.52x over a 512-multiple ladder;
fifty-one buy 1.73x.

#### Spend spare memory on the compute policy, not on a larger batch

Because step time is flat per padded token, doubling the batch buys nothing.
Measured at L=2048 on an A6000, compiled, with the explicit memory preset:

| B | Step | us / token | Peak | Same 65,536 tokens |
|---:|---:|---:|---:|---:|
| 4 | 63.20 ms | 7.715 | 1,658.6 MiB | -- |
| 8 | 123.64 ms | 7.546 | 3,279.7 MiB | -- |
| 16 | 244.86 ms | 7.473 | 6,525.3 MiB | 489.7 ms |
| 32 | 486.92 ms | 7.430 | 13,039.4 MiB | 486.9 ms |

`B=32` costs 1.989x `B=16` for exactly 2x the tokens, so it is 0.58% faster for
the same data at twice the peak; separate-process and eager runs put the same
figure at 0.8% and 0.4%. Saturation is effectively reached by `B=8`.

The comparison that does matter is against the *policy* stack. At a comparable
peak, the compute-oriented default is much faster per token:

| Configuration | Peak | Step | us / token | Same 65,536 tokens |
|---|---:|---:|---:|---:|
| memory preset, `B=32` | 13,073.7 MiB | 490.7 ms | 7.49 | 490.7 ms |
| compute preset (`auto`), `B=16` | 14,586.8 MiB | 206.8 ms | 6.31 | **413.7 ms** |
| memory preset, `B=16` | 6,541.9 MiB | 246.3 ms | 7.51 | 492.6 ms |
| compute preset (`auto`), `B=8` | 7,310.6 MiB | 104.6 ms | 6.38 | 418.3 ms |

For 12% more memory the compute policy is **16% faster on the same data**, at
both budget points. The memory preset's purpose is therefore narrower than "train
at a bigger batch": it raises the longest single item that fits at all, and it
lets a fixed large batch fit in one optimizer step when the optimizer -- not the
hardware -- requires one. If the goal is throughput per device, halve the batch
and turn the policies off before spending memory on `B`.

Unlike upstream ProteinMPNN's `StructureLoader`, no item is ever dropped.
Upstream flushes a batch when `size * (len(batch) + 1)` exceeds the budget and
then starts the next batch *without* the item that triggered the flush, silently
losing one protein per batch boundary per epoch, and any protein longer than the
budget entirely; its `batch_max` variable is assigned but never read. A test pins
that difference. Upstream also normalizes its training loss by a hardcoded
`2000.0` rather than by the residue count, which makes gradient magnitude scale
with batch composition -- the item-balanced objective above is a deliberate
departure, and `ItemBalancedLoss.for_ddp_backward(...)` keeps it exact when ranks
receive different batch sizes.

Neighbor gathers take a different path at `B=1` than at `B>1`, which matters for
memory accounting above the shapes the policies were tuned on. At `B=1` the node
tensor is indexed directly; at `B>1` the local indices must be shifted into a
flattened `[B*L]` node axis, and `F.embedding` retains that shifted index tensor
for backward. Every projection in every layer gathers with the same index
tensor, so it is now built once per distinct index tensor instead of once per
call: about fifteen gathers per training step would otherwise each retain a
distinct 25 MiB `int64` copy at `B=32,L=2048,K=48`, roughly 380 MiB of identical
indices. Inductor already eliminates the duplicate addition inside a compiled
graph, so the memo is skipped while tracing and only changes eager execution.
None of the published `B=1`/`B=4` numbers are affected: `B=1` never took this
path and the `B=4` absolute cost was small.

## Preserved upstream behavior

- The CA k-nearest-neighbor graph includes each residue itself.
- Geometric features contain 25 ordered atom-pair RBF blocks over
  `(N, CA, C, O, virtual-CB)`.
- `k_neighbors` is also the message-reduction divisor, even when `L < K`.
- The frozen oracle accepts `loss_mask`, but the source never uses it. The
  production core removes it and provides an explicit compatibility adapter.
- Training augmentation is additive coordinate noise. The production config
  omits `augment_rot` because that source option is unreachable/no-op code.
- The frozen oracle requires `len_tensor` on the input device. The production
  core normalizes `segment_lengths` to the backbone device before constructing
  packed-segment ids.
- The source training filter enforces logical lengths of at least 150, above
  `K=48`. For a synthetic packed segment shorter than `K`, the frozen source's
  fixed-size global `topk` can select masked residues from another segment.
  Reference parity preserves that quirk; production benchmarks reject `S<K`.
  The true padded-batch path instead masks filler edges and is tested against
  independent execution for items shorter than `K`.

## Verification

The repository-local CPU suite includes a committed deterministic output and
coordinate-gradient anchor for the frozen reference. When the upstream checkout
is available, the integration test archives the exact revision above (not the
moving branch tip) and checks forward plus every gradient directly. It can be
run on an A5000 with:

```bash
srun --partition=gpu --gres=gpu:A5000:1 --time=00:10:00 \
  bash -lc 'cd /home/psk6950/practice/miniworld-kernels && \
  PYTHONPATH=src .pixi/envs/default/bin/python -m pytest -q \
  tests/test_mpnn_reference.py tests/test_mpnn_upstream_parity.py \
  tests/test_mpnn_conversion.py tests/test_mpnn_numerical.py \
  tests/test_mpnn_batch.py'
```

### What counts as a passing comparison

Bitwise equality is asserted only where it is a real property: a policy against
its own recompute, the two message policies against each other, native dropout
output and RNG state, and the LayerNorm forward. Comparisons between a Triton
policy and PyTorch are compared against an FP32 run of the same block instead.

Recompute boundaries are also exact only in eager execution. A checkpoint changes
the graph, so under compilation the partitioner replays the boundary inside the
backward graph and Inductor may fuse that replay differently than the forward it
reproduces. The compiled transition-recompute check therefore requires the
forward output and both RNG states to be bitwise identical -- dropout is outside
the boundary, so the policy must not perturb the mask or the generator -- while
gradients are checked to one FP32 ULP (`4.8e-7` absolute observed on an A6000).
Bitwise gradient equality for the same boundary is asserted in eager on CPU,
where it does hold. The same applies to seeded dropout masks: comparing two
independently compiled graphs requires `fallback_random`, because Inductor's
functionalized Philox derives offsets from each generated kernel's own tiling.

This distinction was learned the hard way. The whole-layer integration check
originally demanded that a compiled BF16 Triton layer stay within `1e-3` of a
compiled BF16 PyTorch layer. Eager Triton and eager PyTorch are in fact bitwise
identical at that shape, but Inductor's own fusion of the reference moves it by
about `1.8e-3`, so the assertion was measuring Inductor rather than the kernel.
The check now requires the Triton policy's error against FP32 to be no worse
than the PyTorch policy's error against FP32, with a loose absolute ceiling that
only catches both degrading together, and it reports every comparison on failure.
Observed errors at `L=64` are `1.0e-3` for the node output, `2.8e-3` for the edge
output, and `4e-3` to `6e-3` for bias gradients, which are sums of 3072 rows at
the BF16 accumulation floor. The ratio bound cannot be tighter than the spread
between two independently compiled BF16 graphs: Inductor fuses the reference's
own reductions differently, so even quantities the edge MLP never touches -- the
node message's output bias, for one -- sit 20-30% apart relative to FP32.

## Production PyTorch architecture

`miniworld_kernels.modules.mpnn.ProteinMPNN` preserves the frozen reference's
mathematics but has an independent, semantic module hierarchy. Its production
checkpoint is intentionally not load-compatible with the CSSB schema.

The package keeps ownership boundaries explicit:

- `model.py`, `features.py`, `layers.py`, and `masking.py` contain model math.
- `data.py` owns sample/batch contracts, collation, and length bucketing.
- `loss.py` owns item-balanced CE and distributed reduction state.
- `batch.py` is only a compatibility re-export for the earlier combined API.
- `naive.py` remains the frozen numerical oracle and shares no compute helpers
  with production.
- `benchmarks/modules/mpnn/workload.py` owns benchmark inputs and objectives;
  the shared runner only handles model setup, correctness, capture, and timing.

`BackboneFeatures.build_graph()` returns a `NeighborGraph` containing edge
features, indices, and validity together. The historical
`BackboneFeatures.forward()` 2-tuple remains available for compatibility, but
production no longer uses a flag-dependent tuple shape.

1. CA-to-CA `cdist` is evaluated once to select neighbors. The remaining 24
   atom-pair distances are evaluated only at those neighbors, reducing that
   work from `O(B*L^2)` to `O(B*L*K)`.
2. Relative residue/chain features are gathered only at selected neighbors,
   and `Linear(one_hot(bucket))` is evaluated as a weight-column embedding
   lookup plus bias. This removes the source's dense positional `L*L` tensors.
3. Floating-point neighbor gathers use `embedding` semantics. The forward is
   the same indexed lookup, while Inductor can fuse the repeated-index backward
   reductions into their surrounding producer kernels instead of launching 11
   generic `scatter_reduce` kernels at the production shape.
4. The final node-message projection is moved after the neighbor reduction via
   linearity, changing a `B*L*K` GEMM into a `B*L` GEMM. The affine bias is
   scaled by the number of contributing neighbors, including masked encoder
   neighborhoods, so the source mathematics is preserved.
5. Training message projections are shape-routed by total directed-edge count.
   Below 49,152 edges, one dense concatenated projection is faster. At or above
   that threshold, `[self, edge, neighbor]` block projections avoid repeated
   node work and large edgewise concatenations. For the production physical
   batch `B=1, K=48`, the block path begins at `L=1024`.
6. The first encoder layer uses the fact that its input node state is exactly
   zero and projects only the edge block. Decoder sequence/current-node/encoder
   context is projected directly, so `h_ES`, `h_EX`, `h_EXV`, and `h_ESV` are
   not materialized on the crop-scale path.
7. In parameter-only mixed-precision training, the 16-wide positional feature
   block and the constant 400-wide RBF block are projected separately. This
   avoids retaining coordinate-feature autograd state without changing the
   coordinate-gradient path.

The model exposes `encode_backbone(...) -> EncodedMPNN` and
`score_sequence(encoded, ...)`. `forward(...)` delegates to these two methods,
so a backbone encoding can be reused across multiple teacher-forced sequence
scores. Incremental autoregressive sampling will consume the same encoded state
in a later layer; it is not implemented by repeatedly calling parallel forward.

An encoded state is intentionally ephemeral: use it with the same unchanged
model instance and mode. Under autograd, multiple score losses sharing one
encoding must be combined into a single backward call (or backward must retain
the graph); eval/no-grad reuse has no such restriction. Owner and mode mismatches
raise an error.

### Checkpoint boundary

The frozen oracle keeps keys such as `W_e`, `encoder_layers.0.W1`, and
`decoder_layers.0.dense.W_in`. The production schema names ownership and intent:

| Frozen CSSB key | Production key |
|---|---|
| `W_e.weight` | `edge_input_projection.weight` |
| `W_s.weight` | `sequence_embedding.weight` |
| `encoder_layers.0.W1.weight` | `encoder.layers.0.node_message.input_projection.weight` |
| `encoder_layers.0.W11.weight` | `encoder.layers.0.edge_message.input_projection.weight` |
| `decoder_layers.0.dense.W_in.weight` | `decoder.layers.0.node_transition.expand_projection.weight` |
| `W_out.weight` | `output_projection.weight` |

Packed message projections remain one physical parameter so the small-shape
dense GEMM and crop-scale block views share exactly the same weights. Relative
position weights are the one deliberate layout change: the source linear
`[channel, bucket]` matrix becomes a contiguous embedding
`[bucket, channel]`, while its shared bias remains a separate parameter.

Legacy checkpoints must be loaded explicitly:

```python
from miniworld_kernels.modules.mpnn import load_cssb_weights

# A frozen module exposes its graph width as features.top_k.
load_cssb_weights(production_model, frozen_reference_module)

# A raw tensor dictionary does not encode K, so it must be supplied.
load_cssb_weights(production_model, raw_state_dict, source_k_neighbors=48)
```

`ProteinMPNN.load_state_dict()` accepts only the semantic production schema;
there is no implicit load hook or legacy alias in the core model. Conversion
validates every key and shape, clones source tensors, and transposes only the
relative-position table. The loader additionally rejects a source/target
`k_neighbors` mismatch; raw state dictionaries require an explicit source K
because that graph hyperparameter is absent from tensor shapes. Optimizer,
scheduler, and scaler states are not migrated because their parameter
identities belong to the old schema.

Most of the model remains PyTorch/Inductor. The fixed production hidden-message
shape (`K=48`, `D=128`) additionally has a custom Triton reduction described
below. Training dispatch is calibrated on an A5000: small edge sets keep the
dense path because extra GEMM launches dominate, while crop-scale edge sets use
the block-linear path because activation traffic and repeated node work
dominate. Eval/no-grad always uses the concat-free reduced-message path.

### Triton hidden-message reduction

The custom boundary is deliberately only the first four lines below. The
output projection, dropout, residual, and layer normalization remain ordinary
model operations:

```python
hidden = F.gelu(preactivation)
hidden = F.gelu(F.linear(hidden, W2, b2))
hidden = hidden * edge_mask[..., None]
reduced = hidden.sum(dim=-2) / K

update = F.linear(reduced, W3) + scaled_b3  # outside the custom op
states = layer_norm(states + dropout(update))
```

Training forward uses two physical kernels. Kernel 1 fuses the first GELU with
the `128x128` `W2` projection and bias. Kernel 2 fuses the second GELU, FP32
structural mask, and exact `K=48` FP32 reduction. The projected tensor is saved
for backward but marked non-differentiable, so autograd does not materialize a
same-size zero gradient. Inference has a separate one-kernel no-save path.

Backward keeps `dW2 = dP.T @ GELU(preactivation)` as a PyTorch BF16 GEMM. The
Triton tail computes BF16 `dP` in three exact 16-neighbor chunks and reduces the
48 values for `db2` while they are live. The default path atomically adds those
FP32 CTA partials into the 128-element bias gradient. When
`torch.use_deterministic_algorithms(True)` is enabled, it instead writes one
FP32 partial per residue and finishes with a deterministic PyTorch reduction.
The two paths emit bitwise-identical `dP`; 64-run atomic `db2` variation was at
most `2.2e-6` relative on the A5000.

The general projection-input-gradient kernel uses `BM64/BN128/BK32`, eight
warps, and three stages. It emits both `dX` and BF16 `GELU(preactivation)`,
avoiding a separate activation pass before `dW2`. It is now used at every shape.

An earlier revision special-cased exactly `B*L=8192`, leaving PyTorch `F.linear`
and GELU visible to Inductor there because that was 0.67% faster end to end at
`B=4,L=2048` while B=1/B=2 tied and B=8 regressed. That exception has been
removed for three measured reasons:

- isolated backward at its own shape is **22.25% slower** through the PyTorch
  path (2.2673 ms versus 1.8547 ms on the A5000), so the 0.67% whole-graph gain
  was not attributable to the kernel choice;
- gradients are bitwise-equal between the two paths (`dpre`/`dW2`/`db2` all
  `3.469e-3`/`2.878e-3`/`2.499e-3` relative to an FP32 reference), so the
  exception bought no accuracy either;
- the lookup hashed a shape-derived value (`groups in {8192}`). Under
  dynamic-shape compilation `groups` is a `SymInt`, and AOTAutograd tracing of
  backward failed with `TypeError: unhashable type: non-nested SymInt`. This
  appeared only on a frame's second compilation, which is why it surfaced as an
  order-dependent `BackendCompilerFailed` in the test suite rather than in
  isolated runs.

Backward is therefore shape-independent: the same launch sequence runs at every
batch size, and gradients no longer change with the batch dimension. Restoring
a shape-keyed rule would require a paired whole-model A/B plus a
`SymInt`-safe predicate.

The fixed configurations came from targeted A5000 sweeps over forward
partitioning, tail tile/warp choices, dX tiles, fused `dP+dX`, dW layouts,
partial/atomic bias reductions, and compiler-visible custom-op variants at
B=1/2/4/8. Production does not run `triton.autotune`; fixed choices avoid
autotune work and CUDA Graph capture variability. The principal production
tiles are projection `BM128/BN128/BK16` with four warps/three stages, forward
tail `BN64` with one warp, and backward tail `BN64` with two warps.

Final isolated compiled-operator measurements were:

| B, L=2048 | PyTorch forward | Triton forward | PyTorch fwd+bwd | Triton fwd+bwd | Training change |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.231 ms | 0.163 ms | 0.622 ms | 0.478 ms | 23.2% faster |
| 2 | 0.445 ms | 0.336 ms | 1.214 ms | 0.970 ms | 20.1% faster |
| 4 | 0.882 ms | 0.673 ms | 2.367 ms | 2.147 ms | 9.3% faster |
| 8 | 1.763 ms | 1.391 ms | 4.686 ms | 3.859 ms | 17.7% faster |

B=4 deliberately gives up some isolated performance to expose dX to the whole
compiled graph. The endpoint, rather than the isolated number, selects that
bucket.

The Triton path requires contiguous CUDA BF16 preactivation `[...,48,128]`,
BF16 projection math (normally FP32 parameters under CUDA BF16 autocast), and a
contiguous non-differentiable FP32 mask `[...,48]`. Unsupported shapes, dtypes,
devices, empty tensors, and differentiable masks use the independent PyTorch
reference. At supported shapes, `auto` now selects Triton for both training and
inference; explicit `pytorch` remains the oracle/fallback. Double backward is
not supported. The backend owns no parameters or buffers, so backend selection
does not alter checkpoint keys, shapes, or strict loading.

The whole-model message-backend A/B at this historical optimization stage uses
three encoder and three decoder layers, item-balanced CE with label smoothing
0.1, `torch.compile`, and manual CUDA Graph replay. B=1/B=4 are medians of three
alternating fresh-process runs; B=2/B=8 are fresh-process confirmation pairs.
Memory is absolute peak allocated in a separate compiled run without graph
capture. Later sections add the edge and selective-memory policies.

| B | Message PyTorch | Optimized message | Latency change | PyTorch peak | Optimized peak |
|---:|---:|---:|---:|---:|---:|
| 1 | 16.466 ms | 16.241 ms | 1.37% faster | 1198.562 MiB | 1087.724 MiB |
| 2 | 31.605 ms | 30.864 ms | 2.34% faster | — | 2140.739 MiB |
| 4 | 60.061 ms | 59.248 ms | 1.35% faster | 4691.265 MiB | 4340.349 MiB |
| 8 | 119.919 ms | 118.432 ms | 1.24% faster | — | 8468.880 MiB |

The separate no-save inference path measured 5.827 ms at B=1 and 21.891 ms at
B=4. The corresponding compiled-PyTorch medians were 6.174 ms and 23.062 ms.
Peak inference allocation remains unchanged because the projected tensor is not
retained in either endpoint measurement.

At B=1,L=2048, the checked full model measured output relative error 0.00555
with cosine 0.999985 and parameter-gradient relative error 0.00222 with cosine
0.999998 against the frozen naive oracle. The implementation preserves the
real-valued mathematical model but changes BF16 reduction order, so mixed-
precision validation uses cosine and relative-error checks rather than bitwise
comparison.

#### Node-message compute and memory policies

`triton` remains a compatibility alias for `triton_compute`. The compute path
saves both the input preactivation and the projected activation for backward.
The explicit `triton_memory` path runs the same two forward kernels and returns
the same reduced value, but saves only the preactivation, weight, bias, and
mask. It recomputes the projected activation once at the start of backward,
then reuses the compute path's reduction derivative, dX kernel, and PyTorch dW
GEMM. `auto` continues to select the compute path during training.

There are six node-message calls in the three-layer model: three encoder and
three decoder calls. At B=1,L=2048 each projected BF16 tensor is 24 MiB, so the
memory policy removes 144 MiB from the saved forward graph (576 MiB at B=4).
The measured peak falls by five tensor equivalents because one full-edge
temporary is live at the backward peak:

| B | Message policy | CUDA Graph training | Peak allocated |
|---:|---|---:|---:|
| 1 | `triton_compute` | 16.396 ms | 865.201 MiB |
| 1 | `triton_memory` | 17.177 ms | 745.201 MiB |
| 4 | `triton_compute` | 60.443 ms | 3469.826 MiB |
| 4 | `triton_memory` | 63.679 ms | 2989.826 MiB |

The table holds the encoder edge MLP on `triton_memory`. Direct operator checks
found bitwise-identical output, dX, and dW; only the nondeterministic atomic
bias-reduction order changed, with relative error below `1.8e-7`. Deterministic
mode retains the existing partial-reduction path.

A 22-configuration sweep also fused projection recomputation into the
reduction backward. Register pressure made the best fused kernel 37--38%
slower than the two-kernel backward, and neither this fusion nor reusing the
inference one-kernel forward lowered the full-model peak. The simpler separate
recompute path is therefore retained.

### Encoder edge-MLP execution policies

The encoder edge update has a separate policy from the node/decoder
neighbor-reduction kernel. This separation is intentional: the fastest
training path and the smallest saved-activation path make different choices.
Neither policy changes module parameters, state-dict keys, model inputs, or
model outputs.

The independent PyTorch reference is:

```python
hidden_1 = F.gelu(edge_preactivation)
projected = F.linear(hidden_1, W2, b2)
hidden_2 = F.gelu(projected)
update = F.linear(hidden_2, W3, b3)
edge_states = layer_norm(edge_states + dropout(update))
```

`triton_compute` uses two physical forward kernels. Each fuses exact GELU with
one `128x128` projection; it saves `edge_preactivation` and `projected` for
backward. `triton_memory` instead evaluates both dependent projections in one
kernel and writes only `update`; autograd saves only `edge_preactivation` and
recomputes `projected` during backward. Dropout, residual addition, and layer
normalization remain ordinary compiled PyTorch operations in both policies.

The memory policy's recompute is close but not bitwise, and that is a measured
choice rather than an oversight. Its fused forward contracts all 128 channels in
one `tl.dot`, while the recompute reuses the message kernel's split-K projection
and accumulates eight 16-wide chunks: `projected` is reproduced to `1.2e-5` and
the composed update to `3.8e-5`, so backward differentiates a very slightly
different function than forward evaluated.

Reproducing the fused order exactly requires a standalone single full-width
`tl.dot`, which is pathologically slow on sm_86. Measured at crop 2048, per
encoder layer, on an A5000:

| Launch | GEMMs | Time |
|---|---:|---:|
| shipped fused forward (`triton_memory`) | 2 | 0.1796 ms |
| two-kernel pair (`triton_compute`) | 2 | 0.2348 ms |
| split-K recompute (`BK=16`, shipped) | 1 | 0.1090 ms |
| full-width recompute (`K=128`) | 1 | 1.2533 ms |

A lone `tl.dot` whose result goes straight to global memory pays a full layout
conversion; the fused forward escapes it only because its first dot feeds a
second one. Three encoder layers would therefore pay about 3.4 ms per step at
B=1 -- roughly 18% of the step -- to remove a `1.2e-5` inconsistency that already
sits inside this policy's own `4.04e-5` forward error against the PyTorch
reference, and far inside the `2.2e-4` gradient error accepted for the
edge-LayerNorm memory policy. The drift is bounded by a test instead
(`< 1e-4`), so a future change cannot widen it unnoticed.

The two policies also differ from each other by design, which is why each
policy's backward is judged against its own forward rather than the other's.

Both backwards reuse the message projection dX kernel. That kernel also emits
the corresponding BF16 GELU result while it is live. The two global weight
gradients therefore remain PyTorch BF16 GEMMs, without separate GELU materialization.
Bias gradients use a BF16 reduction before conversion to the FP32 parameter
dtype, matching CUDA-autocast `nn.Linear` backward rather than silently changing
the naive mixed-precision algorithm. Double backward and `vmap` are not
supported. Forward hooks attached directly to the two inner `nn.Linear`
modules are not invoked by a Triton policy because the policy consumes their
weights and biases at the enclosing message-update boundary.

The A5000 forward sweep selected the following fixed configurations:

- memory forward: one CTA owns 128 rows and all 128 outputs,
  `BM128/BN128`, eight warps, two stages; 172 registers/thread, no spill,
  64 KiB shared memory;
- compute forward: two `GELU -> Linear` kernels, each
  `BM128/BN128/BK16`, four warps, three stages.

The production contract is contiguous CUDA BF16 `[...,128]`, contiguous
`[128,128]`/`[128]` projection parameters using native BF16 or FP32 parameters
under CUDA BF16 autocast, nonempty signed-int32-addressable storage, and width
128. Explicit unsupported Triton requests raise; `auto` falls back to PyTorch.
Because the fixed choices were calibrated on A5000 crop shapes, automatic
Triton dispatch is restricted to sm_86 and at least `2048*48` flattened edge
rows. In grad mode it selects `triton_compute`; under `no_grad` it selects the
one-kernel `triton_memory` forward. Smaller problems and other architectures
remain on PyTorch; either Triton policy can still be requested explicitly.
Correctness is architecture-portable, but automatic-policy performance numbers
in this section are validated only on the A5000.

Isolated outer-CUDA-Graph measurements at crop 2048 were:

| B | Policy | Forward | Forward + backward | Edge-sized tensors saved by autograd |
|---:|---|---:|---:|---:|
| 1 | PyTorch | 0.3033 ms | 1.0229 ms | 96 MiB |
| 1 | `triton_compute` | 0.2435 ms | 0.9014 ms | 48 MiB |
| 1 | `triton_memory` | 0.2219 ms | 1.0075 ms | 24 MiB |
| 4 | PyTorch | 1.1415 ms | 3.8705 ms | 384 MiB |
| 4 | `triton_compute` | 1.0559 ms | 3.4613 ms | 192 MiB |
| 4 | `triton_memory` | 0.8692 ms | 3.8472 ms | 96 MiB |

The saved-tensor counts are four, two, and one full edge tensors per encoder
layer respectively. Consequently the three-layer memory policy removes exactly
216 MiB at B=1 and 864 MiB at B=4 from the full forward graph. Fresh compiled
whole-model peaks were 1081.388 -> 865.201 MiB at B=1 and
4334.013 -> 3469.826 MiB at B=4. The recomputation tradeoff was visible in
whole-model training: encoder forward improved 2.40% at B=4, while encoder
backward regressed 5.37%; end-to-end memory-policy latency was 1.15% slower at
B=1 and 1.64% slower at B=4. It is therefore retained as an explicit memory
policy rather than being mislabeled as the compute winner.

The compute policy was then measured in alternating fresh processes with the
node/decoder message backend held on Triton. Each process reports the median of
20 samples with five CUDA Graph replays per sample:

| B, L=2048 | PyTorch edge | `triton_compute` | Time change | PyTorch peak | Compute peak | Peak saved |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16.2887 ms | 16.0929 ms | 1.20% faster | 1081.388 MiB | 937.201 MiB | 144.188 MiB (13.33%) |
| 4 | 59.5695 ms | 59.5939 ms | tie (+0.04%) | 4334.013 MiB | 3757.826 MiB | 576.188 MiB (13.29%) |

Thus `auto` uses the compute policy for A5000 crop-scale training: it improves
B=1 and is neutral at B=4 while still removing two of the four saved edge
tensors per layer. A fresh `auto` B=1 process resolved to `triton_compute` and
successfully completed full-model compile plus manual CUDA Graph replay.
Inference uses the memory one-kernel forward; whole-model B=1/B=4 measurements
were within 0.33% of PyTorch and therefore treated as ties.

At B=1,L=2048, the memory kernel measured forward relative error `4.04e-5`,
dX `2.01e-5`, dW2 `6.95e-5`, and dW3 `4.11e-5` against the BF16 PyTorch
reference; both bias gradients were bitwise exact. Rank-one `[128]`, production
crop shape, D=128/K=48 encoder integration, fullgraph compile, and first-order
backward have dedicated tests.

### Geometric-feature recompute policy

The RBF feature tensor has shape `[B,L,K,25*num_rbf]`, or
`[B,L,K,400]` at the production settings. With parameter gradients but no
coordinate gradients, its BF16 autocast copy is retained only for the radial
projection's weight gradient. This costs 75 MiB at B=1,L=2048 and 300 MiB at
B=4.

`feature_backend="recompute"` uses a non-reentrant activation-checkpoint
boundary around only the radial RBF projection. It retains FP32 pair distances
`[B,L,K,25]` (9.375/37.5 MiB) and recreates the RBF values during backward.
The positional and radial projections remain two GEMMs in the same order as
the ordinary parameter-gradient path. When coordinates require gradients, the
boundary expands to the existing single concatenated feature projection so its
reduction order and coordinate derivative are preserved. There is no random
operation inside either checkpoint boundary.

With the node-message backend held on compute and the edge MLP on its memory
policy, the feature policy measured:

| B | Ordinary features | Recomputed features | Peak change |
|---:|---:|---:|---:|
| 1 | 16.478 ms / 865.201 MiB | 16.847 ms / 799.201 MiB | -66 MiB |
| 4 | 60.923 ms / 3469.826 MiB | 61.953 ms / 3205.826 MiB | -264 MiB |

For B=1 with coordinate gradients, peak changed from 895.447 to 799.345 MiB
and latency from 17.325 to 17.641 ms. Small-shape checks produced bitwise-equal
outputs and losses; aggregate parameter/coordinate-gradient relative error was
below `5.9e-9`. Both coordinate modes pass `torch.compile(fullgraph=True)`, AOT
backward, and outer CUDA Graph capture with no graph breaks.

### Unmet recompute contracts are reported

The two checkpoint policies differ from the kernel backends in how they refuse
work. An explicit unsupported kernel backend raises; a checkpoint policy whose
runtime contract is unmet -- wrong K or width, a non-contiguous or wrongly typed
operand, a non-`long` index tensor -- falls back to the ordinary path and still
produces correct results. That fallback used to be silent, which allowed a
configured memory policy to be benchmarked as if it were active. Each layer now
emits a single `RuntimeWarning` the first time it falls back. Deliberate bypasses
are not warnings: `no_grad` inference and outer whole-layer checkpointing are
documented behavior. Tracing is exempt as well, because the warned-once flag is a
module mutation that would break a `fullgraph` capture.

### Encoder edge-W1 recompute policy

The memory edge MLP still needs its input preactivation
`[B,L,K,128]` for backward. `edge_w1_recompute="checkpoint"` moves the
checkpoint boundary one projection earlier: it retains compressed BF16 node
and edge operands, recreates the packed encoder W1 projection and edge MLP in
backward, and leaves dropout, residual addition, and LayerNorm outside the
boundary. Consequently there is no random operation to replay and
`preserve_rng_state=False` is safe.

This is an explicit grad-enabled block-path policy and requires
`edge_mlp_backend="triton_memory"`. Its runtime guard shares the edge kernel's
width, layout, dtype, device, and signed-int indexing contract. Standard
`no_grad` inference bypasses it; grad-enabled `eval()` intentionally retains it
for attribution or fine-tuning workloads. It supports native BF16 and CUDA
BF16 autocast, fullgraph compile, outer CUDA Graph replay, and nesting inside
whole-layer checkpointing. The forward result is bitwise unchanged. Compiled
zero-dropout gradient checks stay below `1e-4` relative error. A deliberately
small dropout-0.1 case preserved the forward/backward RNG states exactly and
kept input gradients below `3e-3` relative error and parameter gradients below
`6e-3`; the crop-scale path has a much better-conditioned reduction.

Composed with message, edge-MLP, and feature memory policies, the W1 boundary
measured:

| B | W1 saved normally | W1 recomputed | Peak change | Time change |
|---:|---:|---:|---:|---:|
| 1 | 17.612 ms / 677.826 MiB | 18.181 ms / 605.544 MiB | -72.281 MiB | +3.23% |
| 4 | 65.968 ms / 2726.326 MiB | 66.642 ms / 2477.525 MiB | -248.801 MiB | +1.02% |

The forward graph drops three 24-MiB preactivations at B=1 (288 MiB total at
B=4). The B=4 peak reduction is smaller than that logical tape reduction
because a recomputed full-edge temporary is live at the backward peak.

### Encoder edge-LayerNorm memory policy

`edge_norm_backend="memory"` keeps the encoder edge LayerNorm's native PyTorch
forward unchanged, but stores its backward input in BF16. In the composed
policy, native autograd retains one 24-MiB BF16 edge input and two 48-MiB FP32
edge inputs, or 120 MiB at B=1,L=2048. The custom boundary retains three
24-MiB tensors instead, reducing the logical tape by 48 MiB. The existing
Triton atomic LayerNorm backward reads those tensors directly; it never
restores a full FP32 copy. `auto` remains compute-oriented and calls the
original `nn.LayerNorm` module, as does explicit `pytorch`. Inference,
unsupported shapes/dtypes/devices, and deterministic-algorithm mode fall back
to native PyTorch. The backend is loaded lazily and owns no parameters or
buffers, so state-dict keys and strict loading are unchanged.

The forward and loss are bitwise identical to native LayerNorm. Compression
changes first-order gradients slightly: at L=512 the whole-model aggregate
gradient relative error was 0.00149 with cosine 0.999999. Per-layer LayerNorm
weight-gradient relative errors were at most 0.000428 and bias-gradient errors
at most 0.000458. The path supports `torch.compile(fullgraph=True)`, AOT
backward, and outer CUDA Graph replay.

With message, edge-MLP, feature, and edge-W1 recompute policies already
enabled, fresh A5000 measurements were:

| B | Native edge LN | BF16-save edge LN | Peak change | Time change |
|---:|---:|---:|---:|---:|
| 1 | 18.210 ms / 605.544 MiB | 18.662 ms / 558.294 MiB | -47.250 MiB | +2.48% |
| 4 | 67.037 ms / 2477.525 MiB | 69.020 ms / 2291.025 MiB | -186.500 MiB | +2.96% |

These are the direct edge-LayerNorm A/B timings. The cumulative policy CSV
uses the later node-W1 experiment's native node-W1 baseline at the same peak
(18.673/70.197 ms); latency deltas are therefore computed only within each
recorded comparison group.

Expanding the edge-W1 checkpoint through dropout, residual addition, and
native LayerNorm was also tested as an exact-recompute alternative with RNG
state preservation. It reached 575.131 MiB / 19.576 ms at B=1, worse in both
peak and latency than the dedicated BF16-save boundary.

### Encoder node-W1 recompute policy

The encoder node messages have the same full-edge W1 activation shape as the
edge messages. `encoder_node_w1_recompute="checkpoint"` checkpoints only the
packed encoder W1 projection and the fused hidden-message reduction. The W3
projection, dropout, residual addition, node LayerNorm, transition, residue
mask, and edge update remain outside the boundary. The first encoder layer
uses its exact zero-node specialization, so it replays only the edge block of
W1; later layers replay the packed query/edge/neighbor projection.

This policy is training-only and requires `message_backend="triton_memory"`,
the block-linear encoder path, K=48, width 128, and CUDA BF16 projection math.
Evaluation, `no_grad`, dense or unsupported shapes, and outer whole-layer
checkpointing use the ordinary path. The checkpoint contains no random
operation, so it does not save or replay RNG state. It retains the BF16 edge
operand instead of the distinct W1 activation and changes neither parameters
nor state-dict keys.

The forward output is bitwise unchanged. A compiled full-model gradient check
had aggregate relative error below `2e-3`; a dropout-0.1 check preserved the
forward and backward RNG states exactly. The dedicated storage-identity test
also verifies that the W1 output itself is absent from the saved-tensor tape.

Composed with the message, edge-MLP, feature, edge-W1, and edge-LayerNorm
memory policies, production A5000 measurements were:

| B | Node W1 saved normally | Node W1 recomputed | Peak change | Time change |
|---:|---:|---:|---:|---:|
| 1 | 18.673 ms / 558.294 MiB | 19.072 ms / 487.076 MiB | -71.219 MiB (-12.76%) | +2.14% |
| 4 | 70.197 ms / 2291.025 MiB | 71.181 ms / 2002.307 MiB | -288.719 MiB (-12.60%) | +1.40% |

### Encoder edge-dropout bitpack policy

The encoder's three edge updates each retain a boolean dropout mask with shape
`[B,L,K,128]`. At B=1,L=2048,K=48, those masks occupy 36 MiB even though each
element contains only one bit of information. `edge_dropout_backend="bitpack"`
calls the same ATen native-dropout forward, returns its output unchanged, and
immediately packs only the saved boolean mask to one bit per element. Backward
uses a fused Triton unpack-and-scale kernel; it never materializes a restored
boolean mask.

The policy is implemented by an `EdgeDropout(nn.Dropout)` subclass at the
existing `edge_message.dropout` module path, so train/eval behavior, module
hooks, and state-dict structure are preserved. It is applied only to encoder
edge messages; every node-message, transition, and decoder dropout remains the
ordinary `nn.Dropout`. `auto` and `pytorch` select native PyTorch. Explicit
`bitpack` falls back to PyTorch for evaluation, no-grad, deterministic mode,
CPU, FP16, non-contiguous or empty inputs, p=0/1, inplace operation, and shapes
outside the signed-int32 indexing contract. Triton is imported only after a
supported explicit dispatch.

FP32 and BF16 tests, including the production 12,582,912-element mask and a
nontrivial p=0.37 scale, preserve the output, forward RNG state, backward RNG
state, and input gradient bitwise. `torch.compile(fullgraph=True)`, AOT
backward, and outer CUDA Graph replay are supported. The logical B1 tape drops
from 36 to 4.5 MiB; at B=4 it drops from 144 to 18 MiB. The explicit bitpack
backend supports first-order training gradients only. Differentiating through its
backward raises autograd's `@once_differentiable` error, which names the cause;
requesting `create_graph=True` and then differentiating a gradient that is not
itself part of the graph fails earlier with autograd's generic
"does not require grad" message instead.

With encoder node-W1 recompute enabled and transition recompute held off, the
cumulative endpoints before and after enabling bitpack were:

| B | Native dropout masks | Bit-packed masks | Peak change |
|---:|---:|---:|---:|
| 1 | 19.072 ms / 487.076 MiB | 19.447 ms / 456.951 MiB | -30.125 MiB |
| 4 | 71.181 ms / 2002.307 MiB | 72.105 ms / 1876.432 MiB | -125.875 MiB |

This endpoint table is not a same-process paired latency A/B; it records the
successive production-policy measurements. Dedicated paired bitpack runs are
used by the implementation tests and raw benchmark artifacts.

### Fused encoder edge tail

`edge_tail_backend="triton"` replaces the whole encoder edge update with one
forward launch. The chain it absorbs is

```
preactivation = query + edge @ W1e^T + neighbor[index]
update        = W3(gelu(W2(gelu(preactivation))))
edge_out      = layer_norm(edge + dropout(update))
```

and its autograd boundary saves **no** edge-sized tensor other than the output
the next layer needs anyway. Backward replays the chain from that same input.
The two node-side blocks of the packed W1 projection stay in PyTorch: they are
`[B, T, 128]`, cost nothing, and keep their weight gradients on the ordinary
autograd path. `edge_w1_recompute` is rejected alongside this policy because the
fused replay subsumes it.

Two deviations are deliberate and each has a test in
`tests/test_mpnn_edge_tail.py`:

* The residual stream becomes BF16 rather than the FP32 that autocast's
  `layer_norm` promotion returns. The value is a LayerNorm output feeding an
  autocast `F.linear` that casts it straight back to BF16, and
  `edge_norm_backend="memory"` already stored a BF16 copy of the same quantity.
* Dropout draws from Triton's Philox stream keyed by an explicit seed tensor
  rather than `aten::native_dropout`'s stream, so backward can redraw the mask
  instead of storing it. The keep probability and `1 / (1 - p)` scale are
  identical; only the particular draw differs.

Accuracy is measured against an FP64 evaluation of the same mathematics rather
than against the BF16 PyTorch chain, so the comparison attributes a difference
to whichever path is further from the truth. Across four shapes with and without
dropout, every one of the eleven quantities stayed within 2.4x of the PyTorch
chain's own error, and six of the ten gradients were **more** accurate:

| Quantity | Fused vs FP64 | PyTorch vs FP64 |
|---|---:|---:|
| output | 2.54e-03 | 1.44e-03 |
| d(edge states) | 2.20e-03 | 9.28e-04 |
| d(hidden weight) | 5.77e-03 | 6.52e-03 |
| d(output weight) | 5.85e-03 | 6.33e-03 |
| d(output bias) | 2.18e-03 | 2.87e-03 |

The `output` and `d(edge states)` gaps are exactly the BF16 output: 2^-9 is
2.0e-03.

#### The register budget is what decides how far this can fuse

A 128-wide chain cannot be fused arbitrarily on sm_86. One `[128, 128]` BF16
weight tile held across a kernel costs 32 registers per thread at eight warps;
one `[128, 128]` FP32 gradient accumulator costs 64; the ceiling is 255. A first
version put the whole backward in one kernel -- three weights in contraction
orientation, three `tl.trans` of them for the reverse GEMMs, and one accumulator
-- and measured **255 registers with 1784 spill bytes at 3.1 TFLOP/s**, against
40-42 TFLOP/s for the repository's existing two-weight kernels. Whole-model time
went to 2x the separate-operation baseline.

`tl.trans` on a loop-invariant weight is a shared-memory cost, not a register
one: three `tl.dot` calls alone need about 96 KiB of operand staging against a
100 KiB limit, so adding the transposes made every `TILES > 1` configuration
fail to compile outright at 160-278 KiB. Loading the second orientation directly
(`ptr + n*stride + k` instead of transposing `ptr + k + n*stride`) is also a
plain coalesced load and costs no shared memory.

The tail is therefore split by what has to stay resident, not by what is
conceptually separate. **No pass carries both a weight set and a gradient
accumulator, and none transposes a weight.** Forward is two launches; backward is
two over one row chunk at a time, plus three cuBLAS reductions.

| Pass | Weights | Accumulators | GEMMs | Role |
|---|---:|---:|---:|---|
| `project` | 2 | 0 | 2 | W1 edge block, first GELU, hidden projection |
| `norm` | 1 | 0 | 1 | second GELU, output projection, dropout, residual, LayerNorm |
| `replay` | 3 | 0 | 3 | replay the chain, then LayerNorm/dropout/residual backward |
| `dx` | 3 | 0 | 3 | both MLP projections and the W1 edge block backwards |
| `torch.mm` x3 | -- | -- | 3 | the three weight-gradient reductions |

Measured per 262144-row chunk on an A6000, as the monolithic backward was
progressively taken apart:

| Form | ms/chunk |
|---|---:|
| one kernel: 3 weights + 3 transposes + 1 accumulator | 19.21 |
| same, `BLOCK_M` 64 -> 32 | 9.73 |
| accumulator moved out to its own pass | 7.45 |
| transposes removed, split into `replay` (2.09) + `dx` (1.79) | 3.88 |

**Splitting further loses.** Halving each of those two passes again -- into a
two-weight and a one-weight kernel, so that nothing held three -- was measured
*slower* end to end: 665.06 against 604.50 ms at `B=8`, and 1349.54 against
1221.29 ms at `B=16`. It also pushed reserved memory from 20736 to 28608 MiB,
past the 24166 MiB this policy exists to fit. Three weight tiles with no
accumulator does not spill badly enough to be worth two more edge-sized round
trips per layer. The rule that actually holds is narrower than "keep weight tiles
below three": it is **do not pair a full weight set with a 128x128 accumulator**,
and **do not transpose a loop-invariant weight**.

Three sweep findings were each worth more than any tiling intuition:

* **Row tile dominates once spilling starts.** The same kernel ran 9.73 ms at
  `BLOCK_M=32` against 19.21 ms at `BLOCK_M=64`. A `BLOCK_M * TILES >= 64` floor
  in the first autotune config list hid that configuration entirely, which is
  why no such floor is imposed now.
* **`num_stages` must be swept, not defaulted.** `triton.Config` defaults to
  three. Three stages on top of three transposes and three accumulators measured
  7.15 ms per chunk against 1.13 ms at two -- a 6x regression that was the single
  largest cost in the fused tail until it was pinned.
* **cuBLAS is *not* the wrong tool for the weight gradients**, contrary to what a
  profile bucket suggested. They are `[128, rows] x [rows, 128]` reductions, and an
  aggregated `ampere_bf16_s1688gemm..._tn` row at 437 us per call looked slow enough
  to replace. A direct sweep overturned that: the same `torch.mm` measures **0.197 ms
  at 43.6 TFLOP/s and 681 GB/s**, and the best Triton replacement reachable was
  0.230 ms. The Triton version's time was almost exactly linear in program count --
  16384 CTAs 3.49 ms, 4096 CTAs 1.04 ms, 1024 CTAs 0.385 ms, 128 CTAs 0.230 ms --
  because each program ends in a `tl.atomic_add` of a 128x128 FP32 block, so the
  output atomics, not the GEMM, set the floor. The reductions stayed on cuBLAS.

The eight chunk buffers are a fixed 576 MiB at 262144 rows regardless of batch
size, and they replace what would otherwise be eight full edge tensors -- 12 GB at
`B=16`.

#### Measured effect

The allocator trace at `B=16, T=8192` attributed fifteen live 1536 MiB blocks to
the compiled backward peak -- eight forward tape, seven backward transients --
and the edge tail owned most of them. On an A6000, compiled, against the
memory preset with `decoder_node_w1_recompute="checkpoint"`:

| B | Baseline | Fused edge tail | Peak change | Reserved change |
|---:|---|---|---:|---:|
| 8 | 556.97 ms / 12134.3 MiB / 16266 reserved | 607.26 ms / 9582.5 MiB / 10890 reserved | -21.0% | -33.0% |
| 16 | 1122.38 ms / 24177.2 MiB / 31488 reserved | 1238.87 ms / 19073.3 MiB / 20736 reserved | -21.1% | -34.1% |

The time cost came down from 1.36x to 1.10x over one optimization pass, and the
two changes that did it were both corrections of an earlier wrong call: moving the
weight gradients back to cuBLAS, and splitting forward's three-GEMM kernel in two.
Three predicted improvements measured *worse* and were reverted -- one accumulator
instead of three, and halving the two backward passes. Every one of those was
caught by measurement rather than by reasoning, which is why the sweep harness
matters more here than the design does.

Per token the peak falls from 188.9 to 149.0 KiB. Reserved falls further than
peak because the fused path allocates far fewer distinct edge-sized blocks, so
the caching allocator fragments less -- and reserved, not peak, is what has to
fit a card.

That is the point of the policy, and it was confirmed on the card rather than
inferred from an A6000. On an **RTX A5000** (24123.2 MiB, 64 SMs):

| B | Baseline | Fused edge tail |
|---:|---|---|
| 16 | **out of memory** | 1421.52 ms / 19073.3 MiB / 20736 reserved |
| 8 | 609.35 ms / 12134.3 MiB / 16264 reserved | 711.31 ms / 9582.5 MiB / 10888 reserved |

The target shape does not run at all without this policy. The time ratio is
1.17x on the A5000 against 1.09x on the A6000: the fused backward replays its
chain rather than reading it back, so it is the more compute-bound of the two
paths and the A5000's 64 SMs against 84 cost it more than they cost the baseline.

This policy is off by default. It buys memory at a 1.10x time cost, so it is the
right choice when reserved memory is the binding constraint -- which at
`B=16, T=8192` on a 24 GB card it is, by 7 GB.

What remains is not in the edge tail. The encoder and decoder *node* messages are
still on the separate-operation machinery and are worth roughly 224 ms of the
557 ms `B=8` step between them: `_projection_fwd_kernel`, both reduction kernels,
the node share of `_projection_dx_kernel`, the node share of the weight-gradient
GEMMs, and the node share of the packed-W1 block plus `embedding_dense_backward`.
`kernels/mpnn_node_message/` already implements and verifies that fusion; wiring it
needs its backward reshaped to the two-pass form established here first.

#### Where the dX pass actually spends its time

Profiling the fused path alone answers the wrong question. It says
`_edge_tail_dx_kernel` is the largest kernel in the step; it does not say whether
the chain is cheaper than what it replaced. Profiling *both* on the same card
does, and the answer is that it is not:

| A5000, `B=8` | total | fused edge-tail kernels | everything else |
|---|---:|---:|---:|
| baseline | 602.63 ms | -- | 602.63 ms |
| fused | 667.60 ms | 249.19 ms | 418.41 ms |

The four fused kernels cost 249 ms to replace roughly 184 ms of separate-operation
work -- about 1.35x -- and that difference is the whole time gap. `_edge_tail_dx`
alone, at 116.45 ms, is two thirds of the entire baseline chain, while running at
7% of tensor-core peak and 27% of bandwidth: neither compute nor bandwidth bound.

Ablating the pass one component at a time, sweeping the full configuration grid
per variant so a lighter variant is not judged at a configuration chosen for the
full kernel (A5000, one 262144-row chunk, ms):

| variant | clustered index | uniform index |
|---|---:|---:|
| full kernel | 2.987 | 3.113 |
| minus the `grad_neighbor` scatter atomic | 2.233 | 2.240 |
| minus the `grad_query` grouped atomic | 2.876 | 3.018 |
| minus the three dW-operand stores | 2.119 | 2.231 |
| minus both atomics | 2.059 | 2.108 |
| only the three dots and the `grad_edge` store | 1.268 | 1.288 |

The components are additive: 1.27 real work + 0.87 dW operands + 0.87 atomics =
2.99, and 36 chunks of that is 108 ms against the 116.45 measured in the step.

Three things in that table were not what was expected.

The scatter is almost **insensitive to locality** -- a uniform random index costs
4% more than a clustered one, not several times more. A row tile at `BLOCK_M=32`
against `NEIGHBORS=48` holds the neighbours of a single query, and a query's k
nearest are distinct by construction, so there are no duplicates to coalesce
inside a tile and nothing for locality to help. The cost is raw atomic throughput:
33.5M FP32 atomics per chunk.

**The three dW-operand stores cost more than the scatter does** (0.87 against 0.75
ms per chunk, 31.2 against 27.1 ms at `B=8`). This is the structural reason the
fused path loses. Fusion exists so the activations are never materialised -- but
the weight gradients need them, so the pass writes them back out for cuBLAS alone.
The separate-operation baseline materialises them anyway and pays no such extra
store. Weight gradients are out of scope by instruction here; the number is
recorded because it is half the gap, not to reopen the decision.

`grad_query`'s in-tile group reduction already earns its place, at 4%. It is the
one piece of bookkeeping in this kernel that is not worth attacking.

`bench_tail_kernel.py` and `bench_dweight.py` in the review scratch directory are
the loop for any of this: they force one configuration at a time against the
underlying JIT function and print register counts, spill counts and CTA counts,
rather than trusting `triton.autotune` to find a good point on its own. Every
finding in this section came from those harnesses *after* `triton.autotune` had
already reported its own best configuration as acceptable -- including the two that
overturned a change already believed to be an improvement.

### Relative-position embedding backward

The sequence-offset bucket is one index per *edge* into a 66-row, 16-channel table, so
its backward reduces 6,291,456 rows into 66 at `B=16, T=8192, K=48`. The clamp puts
every long-range contact in the two end buckets -- measured at 33% of all edges -- so
the reduction is as unbalanced as it can be.

`F.embedding`'s backward handles that badly, and the reason is structural rather than a
matter of the bucket count. It launches sixteen kernels: a radix sort of the index, a
`vectorized_gather_kernel` that materialises a reordered copy of the whole gradient
(which is where its 768 MiB of temporaries go), a partial-segment reduction, and a
`sum_and_scatter`. `index_add_` launches one. That is the right algorithm for a
50,000-token vocabulary and the wrong one for a 16-channel row -- and the ratio holds at
about 6x from 66 buckets to 262,144, so it is row *width* that decides, not vocabulary.

| 6,291,456 rows -> [66, 16], real index | ms | rel-err vs FP64 | reproduces | peak MiB |
|---|---:|---:|:---:|---:|
| `F.embedding` backward | 17.6 | 6.7e-06 | yes | 768.0 |
| `one_hot @ W^T` (upstream's own form) | 14.5 | -- | yes | 3960.0 |
| `index_add_` | **2.6** | 2.2e-05 | **no** | 0.8 |
| privatised table, atomic combine | 3.6 | 2.8e-06 | **no** | 0.0 |
| privatised table, fixed-order combine | 3.8 | **2.7e-06** | yes | **0.0** |

Two of those rows are worth keeping in mind. Upstream's `one_hot(bucket) @ W^T` is the
same function and its weight gradient is a dense GEMM with no atomics at all, which
sounds ideal until the `[N, buckets]` one-hot is materialised -- 3960 MiB, ten times the
gradient it consumes, because `F.one_hot` yields INT64. Converting that to an
`nn.Embedding` was right on both axes. And the fixed-order combine is 0.15 ms slower
than the atomic one and reproducible, which is the trade this ships on: the two end
buckets accumulate about a million values each, and a flat FP32 chain over a million
terms is both irreproducible and worse conditioned than P partial sums in a tree.

**It does not help end to end, and that is the headline.** At `B=16` against a 1343.41 ms
step: `index_add` 1340.04, `triton` 1357.41. An autograd boundary is a fusion boundary,
so the reduction's 16 ms saving is paid back in elementwise work that used to fuse --
+17 ms here, and +24 ms when this was a `torch.library.custom_op`, which is opaque to
Inductor rather than merely a boundary. The backend therefore ships `off`.

It is kept because three of its properties are not visible in that comparison: under
eager execution there is no fusion to lose and the op is a straight fivefold win, the
kernel is 2.5x more accurate than what it replaces, and it allocates nothing against
768 MiB.

The diagnosis that started this was wrong and the correction is the useful part. The
`embedding backward` profile category is 214 ms of a 1358 ms step, and this pass is
about 19 ms of it. The rest is `gather_neighbors`, which gathers node values at the
neighbour index and scatters `[B*T*K, 128]` back into `[B*T, 128]` -- 1.5 GiB into
131,072 destinations. At that width the ranking inverts: `F.embedding`'s sort-based
backward runs at 205 GB/s compiled while `index_add_` manages 83, and a BF16
`index_add_` collapses to 9 GB/s because the hardware has no BF16 atomic add. The
privatised table cannot help there either, since a `[131072, 128]` accumulator is
67 MiB. That pass is close to its floor and is left alone.

### Fused encoder node message

`kernels/mpnn_node_message/` applies the same treatment to the node half of an
encoder layer -- packed W1 edge block, both GELUs, W2, and the mask-weighted
neighbor reduction in one launch. Because each program owns whole neighbor
groups, the reduction happens in registers and the query-block gradient is an
exact per-group row sum with no atomics.

Its accuracy is verified the same way, and it is the stronger result: across
three shapes the worst quantity was 1.015x the PyTorch chain's own error against
FP64, and six of the seven were more accurate.

Its backward was then reshaped to the envelope this section establishes -- four
weight tiles in both orientations, no accumulator, no `tl.trans`, both weight
gradients on cuBLAS -- and wired behind `node_message_backend="triton"`.

**It is off by default because measurement rejected it.** On top of the fused
edge tail:

| B | Edge tail only | Edge tail + node message |
|---:|---|---|
| 8 | 606.84 ms / 9582.5 MiB | 734.56 ms / 14102.2 MiB |
| 16 | 1233.80 ms / 19073.3 MiB | 1489.67 ms / 28113.8 MiB |

At `B=16` that peak no longer fits the card the policy exists for.

The allocator trace says the node message is not what grew. Its own live
footprint at the peak is three 16 MiB node-sized blocks; nothing edge-sized. What
appears instead is a single **2400 MiB** block in `features.py:_project_radial`,
3.1x an edge tensor, alongside the decoder's own edge projections. Making the
encoder backward cheaper moved the peak to a different moment -- the geometric
feature recompute -- and exposed a consumer that had been sitting underneath the
encoder's peak all along.

That is the next target, and it is larger than anything left in the encoder: one
2400 MiB transient plus a 768 MiB companion in the same function, against the
1536 MiB the whole three-layer encoder edge path now holds. The node-message
kernel stays in the tree, tested at 1.015-1.041x the PyTorch chain's own FP64
error across four shapes including a multi-chunk one, and can be revisited once
the feature path stops setting the peak.

### Transition update recompute policy

`transition_recompute="update"` checkpoints only
`W_out(GELU(W_expand(states)))` in every encoder and decoder transition.
Dropout, residual addition, and post-LayerNorm stay outside the boundary, so
the replay is deterministic and `preserve_rng_state=False` is safe. An outer
whole-layer checkpoint explicitly disables this inner boundary. The existing
`ResidualTransition`, its submodules, parameters, hooks, and state-dict paths
remain intact; internal projection hooks may run again during the standard
checkpoint replay.

Compared with checkpointing the complete transition, this partial boundary was
the better default selective policy in the pre-bitpack experiment: at B=1 it
was both faster and slightly lower-peak, while at B=4 the complete boundary
bought only another 14.843 MiB for additional replay work. The production
policy therefore keeps the smaller update-only boundary. With encoder node-W1
recompute and edge-dropout bitpack already enabled, it measured:

| B | Transition saved normally | Update recomputed | Peak change | Time change |
|---:|---:|---:|---:|---:|
| 1 | 19.447 ms / 456.951 MiB | 19.554 ms / 433.576 MiB | -23.375 MiB (-5.12%) | +0.56% |
| 4 | 72.105 ms / 1876.432 MiB | 72.267 ms / 1823.557 MiB | -52.875 MiB (-2.82%) | +0.22% |

The explicit crop-2048 memory preset is the composition below; it does not
change model inputs, outputs, parameters, or state-dict keys:

```python
ProteinMPNNConfig(
    message_backend="triton_memory",
    edge_mlp_backend="triton_memory",
    edge_norm_backend="memory",
    edge_dropout_backend="bitpack",
    feature_backend="recompute",
    edge_w1_recompute="checkpoint",
    encoder_node_w1_recompute="checkpoint",
    transition_recompute="update",
)
```

| B | Compute-oriented policy | Explicit memory policy | Peak reduction | Time cost |
|---:|---:|---:|---:|---:|
| 1 | 16.093 ms / 937.201 MiB | 19.554 ms / 433.576 MiB | 53.74% | 21.51% |
| 4 | 59.594 ms / 3757.826 MiB | 72.267 ms / 1823.557 MiB | 51.47% | 21.27% |

Whole-layer checkpointing remains available, but it is no longer the
lower-memory point. With feature recompute retained and compute-oriented inner
operators, the current outer-checkpoint peak was 482.733/1946.069 MiB at
B=1/B=4. The selective policy is another 49.157/122.512 MiB lower while
avoiding complete encoder/decoder replay. Keeping the inner memory boundaries
inside the outer checkpoint is redundant and worse: it measured
617.409/2528.495 MiB. The old whole-layer latency comparison was measured
against an earlier policy stack and is intentionally not mixed into the final
table.

The remaining packed-stress bottleneck is explicit: neighbor selection still
uses a global `O((sum L_i)^2)` CA distance problem. Inductor already fuses the
pair-mask and adjusted-distance elementwise work, so true-batch KNN is not the
training peak: isolated B=1/B=4 measurements were 0.506/1.419 ms and
20.181/80.720 MiB, and its transient does not overlap the later activation
peak. Query chunking therefore saved only 4.041/14.629 MiB at B=1/B=4 while
making KNN 100%/43% slower.

Packed B=4 represented as `[1,8192]` is different: global KNN measured
4.291 ms / 272.783 MiB, while 256-row query chunks reached
7.498 ms / 60.625 MiB. A static packed-to-padded-batch adapter would reduce
work toward `sum(L_i^2)`, but changing the candidate axis changes ATen
`topk`'s unspecified tie ordering, especially for padded or invalid rows.
Direct Triton distances would also change the `cdist` reduction and rounding
order. Neither candidate satisfies the strict reference-index contract, so no
KNN fast path is enabled in production.

Packed segment IDs use `repeat_interleave(..., output_size=physical_length)`.
The explicit output size preserves the source values while avoiding a
data-dependent output-shape synchronization during compile/CUDA Graph capture.
Each captured graph is valid only for its fixed segment-count/length profile.

## A5000 `L=2048` benchmarks

All production-shape measurements use `L=2048`, `K=48`, width 128, three
encoder and three decoder layers, and patch size 8 on an NVIDIA RTX A5000.
Timing uses `torch.compile` plus manual CUDA Graph replay; compilation and graph
capture are excluded. Memory uses a separate compiled run with CUDA Graph
allocation disabled and reports absolute peak allocated memory.

The older sequence-length sweeps are retained as historical B=1 model-core
`output_grad` measurements: their timed step is teacher-forced forward plus a
supplied output-gradient backward. The true-batch and memory-policy results use
`mpnn_training_objective=item_ce`: their step includes model forward, FP32
item-balanced cross entropy with label smoothing, and backward. Both exclude
optimizer work, gradient zeroing, DDP, coordinate gradients, and
coordinate-noise augmentation. Current runner outputs and curated policy tables
record the objective and step scope. Retained historical CSVs predate those
fields and are explicitly scoped here so the two workloads are not compared as
if they were identical.

The checked-in batch benchmark config targets crop 2048 with compile plus a
manual CUDA Graph:

```bash
srun --partition=gpu --gres=gpu:A5000:1 \
  .pixi/envs/default/bin/python benchmarks/runners/bench.py \
  --config-dir benchmarks/modules/mpnn/configs --config-name bench_batch \
  batch_size=2
```

### Historical message-kernel-stage true-batch item-CE results

On 2026-07-25, fresh A5000 processes measured equal-length crop-2048 batches at
the retained hidden-message-kernel stage. These rows predate the later edge-MLP
and selective-memory policies and are not the current automatic-policy endpoint.
Timing is the manual-CUDA-Graph replay median of forward + item-balanced CE
(`label_smoothing=0.1`) + parameter backward. Memory is a separate compiled run
with CUDA Graph disabled and reports absolute peak allocated memory.

| B | Naive time | Production time | Speedup | Production items/s | Naive peak | Production peak |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 27.749 ms | 16.241 ms | 1.709x | 61.57 | 2111.392 MiB | 1087.724 MiB |
| 2 | 54.377 ms | 30.864 ms | 1.762x | 64.80 | 4190.946 MiB | 2140.739 MiB |
| 4 | 106.362 ms | 59.248 ms | 1.795x | 67.51 | 8351.181 MiB | 4340.349 MiB |
| 8 | 210.489 ms | 118.432 ms | 1.777x | 67.55 | 16668.524 MiB | 8468.880 MiB |

All four production batches fit on the 24-GiB A5000. At B=8, production saves
49.19% of absolute allocated peak memory versus the naive reference. A separate
compiled BF16 correctness run at `B=1,L=2048` measured output relative error
0.00555/cosine 0.999985 and parameter-gradient relative error 0.00222/cosine
0.999998 against the frozen oracle.

### Pre-kernel true-batch training profile

The production profile uses the same A5000, crop 2048, `K=48`, width 128,
three-layer encoder/decoder, patch size 8, BF16 autocast, FP32 parameters,
dropout 0.1, and item-balanced CE with label smoothing 0.1. It includes all
parameter gradients but excludes optimizer work, gradient zeroing, DDP
communication, coordinate gradients, and coordinate-noise augmentation. Fresh
manual-CUDA-Graph replay measurements were:

| B | Forward + item CE + backward |
|---:|---:|
| 1 | 16.779 ms |
| 4 | 61.458 ms |

CUDA timing events cannot be queried from inside a captured graph on this
software stack. Phase boundaries were therefore measured separately on the
same compiled model before CUDA Graph capture. The loss remains outside the
compiled model, matching the benchmark workload. These medians explain the
CUDA-Graph endpoint but are not themselves CUDA-Graph replay latency:

| B | Forward | Item CE | Backward | Compiled-eager total |
|---:|---:|---:|---:|---:|
| 1 | 6.523 ms (36.7%) | 0.099 ms (0.6%) | 11.154 ms (62.8%) | 17.768 ms |
| 4 | 22.825 ms (36.6%) | 0.113 ms (0.2%) | 39.482 ms (63.2%) | 62.439 ms |

For finer attribution, features, encoder, decoder, and output/loss were
compiled as separate static-shape stages, with CUDA events at both forward and
backward boundaries. Those extra boundaries make the staged pipeline slower
than the whole compiled model, so its absolute latency is diagnostic; the
shares identify where the work resides. Each row below combines the stage's
forward and backward time:

| B | Features | Encoder | Decoder | Output head + CE | Staged total |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.578 ms (13.9%) | 10.193 ms (54.9%) | 5.518 ms (29.7%) | 0.295 ms (1.6%) | 18.583 ms |
| 4 | 8.505 ms (13.3%) | 36.993 ms (57.8%) | 18.265 ms (28.5%) | 0.246 ms (0.4%) | 64.058 ms |

Isolated compiled KNN takes 0.498 ms at B=1 and 1.457 ms at B=4. This is only
about 2--3% of the staged step and 31--34% of the feature stage. A ten-step raw
CUDA trace gives the following non-exhaustive kernel-family shares; these are
shares of device kernel duration, not additional phase timings:

| Kernel family | B=1 | B=4 |
|---|---:|---:|
| BF16 GEMM | 39.64% | 37.66% |
| fused Triton kernels explicitly labeled backward | 37.36% | 41.06% |
| top-k selection | 1.63% | 1.64% |
| RBF/atom-pair vector norms | 1.29% | 1.50% |

This profile motivated targeting the shared encoder/decoder node-message MLP
and its backward before KNN or the output head: encoder work is 55--58% of the
step, decoder work is about 29%, and backward as a whole is about 63%. The
selected first boundary is the repeated hidden projection/reduction described
above. An isolated KNN kernel or a fused CE/output head has a much smaller
crop-2048 ceiling; KNN may become more important for longer lengths or a
different feature/message-width balance.

### Historical model-core results

| Workload | Precision | Naive | Production | Speedup | Latency reduction |
|---|---|---:|---:|---:|---:|
| Training fwd+bwd | BF16 mixed | 27.518 ms | 16.356 ms | 1.682x | 40.56% |
| Training fwd+bwd | FP32, TF32 off | 71.582 ms | 38.933 ms | 1.839x | 45.61% |
| Eval forward, teacher-forced | BF16 mixed | 11.828 ms | 5.976 ms | 1.979x | 49.48% |

Memory is measured in a fresh process with the compiled model but CUDA Graph
disabled, so graph-pool reservation is not mixed into model memory. The table
reports absolute `torch.cuda.max_memory_allocated`, not the benchmark CSV's
baseline-subtracted `value` field.

| Workload | Precision | Naive peak | Production peak | Saved | Reduction |
|---|---|---:|---:|---:|---:|
| Training fwd+bwd | BF16 mixed | 2111.474 MiB | 1197.101 MiB | 914.373 MiB | 43.30% |
| Training fwd+bwd | FP32, TF32 off | 3812.312 MiB | 2193.034 MiB | 1619.278 MiB | 42.47% |
| Eval forward, teacher-forced | BF16 mixed | 554.366 MiB | 244.473 MiB | 309.893 MiB | 55.90% |

The historical BF16 training endpoint is the median of three isolated production
runs after the semantic checkpoint refactor. A separate fresh naive run measured
27.518 ms. The full historical sweep shows the intended routing transition: speedup
is 1.20--1.29x through `L=768`, then
rises to 1.62x at `L=1024` when the crop-scale block path activates, and reaches
1.68x at `L=2048` in the sweep run.

Numerical checks cover the paths excluded from the timed replay:

| Check | Output relative / cosine | Gradient relative / cosine |
|---|---:|---:|
| BF16 coordinate-grad, `L=1024` | 0.005407 / 0.999985 | 0.011953 / 0.999929 |
| FP32 parameter-grad, `L=1024` | 5.354e-7 / 1.000000 | 1.259e-6 / 1.000000 |
| BF16 packed `4x256`, parameter-grad | 0.005499 / 0.999985 | 0.006111 / 0.999981 |

Profile-guided ablations identified edgewise message activations and repeated
neighbor-gather backward reductions as the original training bottlenecks. At an
intermediate 19.49 ms stage, 11 generic `scatter_reduce` launches consumed
2.87 ms per step. Expressing floating gathers as embeddings let Inductor fuse
those reductions and removed about 1.70 ms. In that message-backend trace,
GEMM/GEMV kernels account for roughly 6.57 ms per step; the remaining neighbor
reductions are already fused with GELU/mask producers. A standalone custom
gather would break that fusion.

Tracked results live under `benchmarks/modules/mpnn/results/a5000/tables/`:
`training_seq_len.csv` contains the historical model-core BF16 training sweep,
`inference_seq_len.csv` contains the current small-length inference sweep, and
`l2048_summary.csv` contains the historical output-gradient time and
absolute-memory endpoint summary. `true_batch_l2048.csv` preserves the
historical message-kernel-stage B=1/2/4/8 item-CE timing, throughput, and
absolute-memory measurements reported above. `message_backend_l2048.csv`
contains all fresh-process PyTorch-versus-Triton B=1/B=4 latency repeats and
the matching absolute-memory runs for the custom message operator.
`edge_mlp_backend_l2048.csv` records the separate compute/memory edge-policy
A/B runs, including all process medians and absolute allocated peaks.
`memory_policy_l2048.csv` composes the message, edge-MLP, geometric-feature,
edge LayerNorm, edge-dropout, edge-W1, encoder node-W1, and transition policies,
and records whole-layer checkpointing as a separate comparison point. The
sequence-length latency/speedup SVGs live in the sibling `plots/` directory. Raw local
ablations and profiler traces remain under the ignored `artifacts/` and
`profiles/` directories.
