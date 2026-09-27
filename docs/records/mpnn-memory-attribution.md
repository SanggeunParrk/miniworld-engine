# MPNN tensor memory attribution and fusion candidates — 2026-09-12

The mixed B8/L8192 model already holds **21.072 GiB after forward**.
Its warmed training peak is **21.703 GiB**, during the last decoder layer's message
backward, while all encoder saves are still live. Most memory is saved edge-sized
activation storage. Model parameters are only **6.334 MiB**; parameters, their
gradients and AdamW state together are about **25.34 MiB**.

This is a direct A6000 allocation profile, not an A5000 timing result. The earlier
mixed-policy probe's **21.940 GiB** includes cold warmup. This record isolates a
warmed step after two real optimizer updates. The existing A5000 policy-search
job `1680997` continues from its separate source snapshot; no production kernel,
configuration default or queued experiment was changed by this analysis.

## Where the memory goes

At B8, L8192, K48 and D128, one edge-shaped tensor has
`8 * 8192 * 48 * 128 = 402,653,184` elements. Even in BF16 that is
**805,306,368 bytes = 768 MiB = 0.750 GiB**. A node-shaped BF16 tensor is only
16 MiB. The major edge tensors below were actually observed as BF16; this is not
an inference from an autocast flag.

Mixed policy, actual resident allocations at the end of forward:

| Allocation group | GiB |
|---|---:|
| Encoder edge outputs and saved activations | 9.000 |
| Encoder edge dropout masks | 1.125 |
| Decoder preactivations and saved projections | 4.500 |
| Expanded RBF features | 2.344 |
| Initial feature and edge embedding states | 2.250 |
| Node/transition states, indices, parameters and other allocations | 1.854 |
| **Total** | **21.072** |

The encoder edge row contains three retained edge outputs (2.250 GiB) and three
saved matrices per layer—first projection, second projection, pre-normalization
values—totaling 6.750 GiB. Each layer also retains one byte per dropout element:
384 MiB per layer, 1.125 GiB across three layers. See the
[compute-tail forward](../../src/miniworld_engine/kernels/mpnn_edge_tail/triton/compute.py).

Each decoder retains one full preactivation matrix and one projected matrix:
six times 768 MiB = 4.500 GiB. Its preactivation contains the edge projection and
conditional query/sequence/current-neighbor/encoder-neighbor contributions.

The RBF buffer has shape `[8, 8192, 48, 400]`, BF16: **2.34375 GiB**.
These are 25 atom-pair distances expanded into 16 radial basis values each. It
survives until feature-projection weight backward even though input coordinates
do not require gradients. The initial feature/edge state row contains three
additional width-128 edge matrices: feature normalization input, normalized
features and the initial edge embedding.

| Policy | Forward-end allocated GiB | Warmed step peak GiB | End-of-step reserved GiB |
|---|---:|---:|---:|
| Full compute | 25.479 | 26.109 | 26.404 |
| Mixed: node memory, tail/decoder compute | 21.072 | 21.703 | 22.557 |
| Full memory | 11.041 | 12.421 | 13.299 |

Full compute retains six additional encoder node projections, 4.500 GiB of raw
edge-shaped storage; the observed net forward difference is 4.406 GiB because
auxiliary graph allocations also differ. The full memory policy still retains
decoder preactivations, the RBF expansion and initial edge features.

The N×N `cdist`/neighbor-selection matrices are early temporaries and are gone
at the overall peak. They are not the principal explanation for the 21–26 GiB
peak in this workload. Likewise, the compiler already stores the large feature
normalization tensors here in BF16; assuming a full FP32 save from a source
comment would incorrectly double that contribution.

## The peak and a tested counterexample

The allocator event that reaches the global maximum is inside
[`_projection_dx_weight_op`](../../src/miniworld_engine/kernels/mpnn_message/triton/main.py),
in the last decoder's backward. At that moment the model still holds the encoder
states and saves, RBF input, all decoder preactivations and two earlier decoder
projected matrices, plus the active message gradient buffers. The activation
scratch for dW is already chunked to 64 MiB; it is not another full 768 MiB save.

The compute edge-tail backward does allocate seven width-128 edge buffers up
front: grad-values, grad-update, grad-hidden, grad-preactivation, two activated
matrices and grad-edge. This is 5.250 GiB of local buffers. We tested a process-local
override that delays allocation, computes each weight gradient immediately after
its producer and releases the two consumed buffers. It uses the same kernels and
GEMMs, with no extra projection recomputation.

| Mixed-policy experiment | Peak during edge-tail backward GiB | Overall peak GiB | End-of-step reserved GiB |
|---|---:|---:|---:|
| Current implementation | 21.558 | 21.703 | 22.557 |
| Shortened temporary lifetimes | 19.037 | 21.703 | 21.893 |

The local improvement is real, but **the global tensor peak does not move**.
The decoder reaches it before the edge-tail backward begins. Reservation improves
because the allocation pattern changes. This prototype completed warmup and a
profiled training step with finite parameter gradients; it has not undergone a
dedicated numerical/speed qualification and was not applied to production.

## Fusion and storage priorities

1. **Pack the compute-tail dropout mask into bits.** The mask stores Boolean data
   in INT8. Three masks would shrink from 1.125 to 0.140625 GiB, a **0.984375 GiB
   storage reduction**. Preserve the existing RNG decisions and decode the bits
   in backward. This requires validating packing/unpacking and timing overhead,
   but does not require repeating the edge GEMMs. The separate `bitpack` dropout
   backend does not change the mask inside the fused compute-tail kernel.

2. **Fuse/stream RBF generation with feature projection and its weight gradient.**
   Retain the 25 FP32 distances per edge (0.292969 GiB) instead of all 400 BF16
   RBF values (2.343750 GiB), then generate tiles when consumed. The structural
   saving is **2.050781 GiB**, before any new workspace. This needs a kernel-level
   backward for the fixed-coordinate contract. The existing
   `feature_backend="recompute"` calls the checkpoint API, so simply enabling it
   would violate the requested checkpoint-OFF condition. The extra exponentials,
   GEMM tile reuse and accumulation accuracy must be benchmarked.

3. **Extend fusion across the decoder projection/message boundary.** A dedicated
   decoder node-message path could combine W1's edge GEMM, conditional gathers,
   past/future masking, W2/GELU and K reduction, with a backward that retains fewer
   full edge matrices. The current 4.500 GiB of decoder saves is the target, not
   a guaranteed saving. A forward fusion that continues saving both full matrices
   will not remove that footprint. Preserve all three distinct neighbor/sequence
   contributions and the decoding masks; the encoder node kernel alone does not
   implement those semantics.

The first two candidates target about **3.035 GiB of persistent storage** together.
This is a storage budget, not a measured speedup or guaranteed additive reduction
in the whole-model peak: new workspace and a shifted peak may change the outcome.
It could allow more compute policy to be retained, subject to actual measurements.

The generated Inductor code already fuses decoder gather/mask/add into an in-place
pointwise pass over the W1 edge output. It also fuses RBF distance/exp/cast work,
and feature add/normalization/mask operations. Therefore the remaining opportunities
are the GEMM/custom-op boundaries and backward storage contracts, rather than
reimplementing those already-fused pointwise chains. Edge-tail output GEMM,
residual, dropout and normalization are already combined as well.

## Evidence and reproduction

Same B8/L8192, 3 encoder + 3 decoder, D128/K48, BF16 autocast, FP32 parameters,
dropout 0.25, coordinate noise 0, no accumulation/offload/checkpoint API, compiled
forward/AOT backward, training CUDA Graphs OFF and real AdamW as the previous
policy study. Input coordinates do not require gradients. PyTorch 2.10.0+cu128,
CUDA 12.8, `PYTORCH_ALLOC_CONF=expandable_segments:True`.

After two warm steps, CUDA allocator history records allocation/free events and
stacks through forward, backward and AdamW. Saved-tensor hooks retain only metadata
outside the normal autograd saves. Shapes, dtypes and storage addresses are recorded;
views are deduplicated by storage. Replaying the history reproduces the allocator
peak within 4 KiB. Allocation origins and the actual generated Inductor code supply
the semantic mapping. These are simultaneous live allocations, not cumulative
operator allocation totals. No timing claim is made from instrumented execution.

The [JSON record](mpnn-memory-attribution.json) includes all four profile metadata
records, saved-storage groups, peak origins, exact probe/override sources and
generated-code excerpts. Large raw snapshots and generated sources remain under
`.scratch/mpnn-memory-analysis/`. No production source changed.
