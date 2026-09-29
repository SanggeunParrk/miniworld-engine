# A6000 AdaLN workload-aware cache repair and SWA fusion — 2026-09-11

AdaLN training was using register-spilling one-warp candidates. Refreshing the candidate set,
without changing the fused GEMM or switching to cuBLAS, makes the complete training step faster
than compiled PyTorch at L384. ConditionedTransition benefits from the same repaired AdaLN.

## Controlled module benchmark

A6000 UUID `be7e377a-4994-4692-b5dc-f915732cc395`, L384, augmentation 48, token/condition widths 768/384,
depth 1, BF16 inputs and parameters, TF32 OFF, actual torch.compile ON, CUDA Graph OFF.
The native module constructors and timing function are unchanged. Forward+backward, no optimizer.
These two modules have no dropout. Each result is the median of three timing samples in one
process. All six processes ran sequentially in allocation 1677787 on the same physical GPU.
The before case reads a saved copy of only the original AdaLN forward-gate cache; after reads
production cache normally. Cache misses raise, and no benchmark process changed tile caches.

| Module | PyTorch ms | MiniWorld before ms | MiniWorld after ms | Reduction |
|---|---:|---:|---:|---:|
| AdaLN | 1.224704 | 1.622016 | 1.123840 | 30.7% |
| ConditionedTransition | 6.065152 | 6.223872 | 5.803008 | 6.8% |

## Repair and scope

`cache.py` and `capture.py` now separate AdaLN forward-gate timing records by physical tensor
shapes, strides, dtypes and scalar arguments. The shared round cache, committed-time reuse,
incremental searched-config subtraction, capture flush and shard merge carry the same workload
identity. A changed workload also clears the builder's in-process Triton winner, so a logical-key
hit cannot bypass measurement for the new physical input. Each workload keeps its own top five;
runtime candidates are their union, without selecting by incomparable raw milliseconds.
Unattributed legacy shards cannot overwrite a workload-aware repaired entry.

This compatibility policy is currently enabled for `adaln_fwd_gate_triton`, the demonstrated
regression. It does not assert that every historical kernel cache has been audited or repaired.
The kernel body, config grid and fused/cuBLAS dispatch were not changed.

Refreshed six active BF16 token-training keys at L128/256/384/512/640/768, each at M=48*L,
NX768/NC384. Compared 55 existing-grid candidates per key, including every historical cached
winner signature and the nearby 2/4-warp options. 53 ran and passed numerical checks per key;
two require 122880 bytes shared memory, above A6000's 101376-byte limit, and were recorded as
observed failures. The five fastest valid candidates were published only after all six lengths
had completed. Unrelated cache entries were unchanged.

This is a bounded measured repair, not a claim to have re-swept all 2160 grid configurations or
found the global optimum. The stored searched set records exactly the 55 candidates; later
incremental builds can measure the remainder for their own workload. Other GPU caches were not
refreshed as part of this repair.

## Numerical validation

Three seeds (13/41/79), full L384/A48 compiled AdaLN, nonzero random projection weights and
biases, compared with the FP32 PyTorch module loaded from the same rounded parameters and inputs.
Checked output, both input gradients, and every parameter gradient: 21 checks passed.
Output relative L2 error is 0.2232–0.2236%; maximum gradient relative L2 error is 0.3781%.
The acceptance threshold was 2%, all values finite. Candidate-level output and saved-gate checks
also passed the existing kernel's 1.4% relative-L2 band (318 valid candidate/shape combinations).

Cache reader/policy, capture/shard merge, prediction provenance and shape-key regression suite:
86 passed, followed by 8 AdaLN shipped-cache structural checks. The added in-process workload
invalidation is covered by the subsequent targeted run recorded in `workload_checks.out`.

## What Inductor fuses in SWA inference

Inspected the actual generated executable from the controlled A6000 SWA run (L384 = 3072 atoms,
A5, four heads of width32). Its main preprocessing kernel reads the original interleaved QKV
buffer and jointly emits rotated Q and K. It contains:

- Q/K selection and view/permute index calculations, without materializing separate Q/K clones.
- FP32 RMSNorm arithmetic for each head: square, sum/mean, epsilon addition, rsqrt, scaling.
- 3D RoPE application: cosine/sine addressing, half-vector rotation and sign change, multiplication
  by the precomputed cos/sin, and addition; final BF16 Q/K stores.

The cos/sin tables are inputs, not computed by this fused kernel. Dtype conversions are included
in the generated arithmetic; an explicit source-level cast does not imply a materialized buffer
or an intermediate BF16 rounding instruction in the optimized kernel.

A second pointwise kernel fuses `sigmoid(gate_projection) * attention_output` and writes into
an available output buffer. QKV projection, gate projection and output projection remain separate
GEMMs. FlashAttention-2 is a separate shared custom op. The generated preprocessing kernel's
long name includes `flash_window` because of source-node attribution, not because attention's
QK/softmax/AV computation was fused into it.

Previously measured preprocessing GPU durations: compiled PyTorch 26.13 us versus MiniWorld's
Q/K copies + RMSNorm + RoPE totaling 71.8 us. See the prior SWA attribution record for the profiler
scope; these are not a fresh SWA timing result from this repair.

[Raw benchmark samples, numerical checks and refresh provenance](a6000-adaln-workload-cache-fix.json).
Reproduction scripts, logs, original cache backup and generated SWA source:
`/home/psk6950/practice/miniworld-engine/.scratch/a6000-2026-09/mw-adaln-fix/`.
