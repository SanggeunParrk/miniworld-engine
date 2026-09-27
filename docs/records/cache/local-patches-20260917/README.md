# MiniWorld consumer patches merged with concurrent engine main

## Scope and provenance

This integration brings the 24 active MiniWorld engine patches onto engine main.
`patches.json` records each input patch's SHA-256 and its application commit.
The starting engine revision is `1bc0803e3b2fef3b963fdc383e090c0adcccdb43`
(`cache/h100`). Before upstream integration edits, all 113 patch-affected files
matched the consumer's installed engine byte for byte.

The independent main updates through
`58667df26d8d06e1677ba6bf854ec9e34bb3bee5` are retained as a merge parent.
They include the ESMFold2 SWA DiT equations, model-derived shape declarations,
FA2 dispatch/packing changes, compact cache search spaces, and A5000/A6000 data.

The inactive experimental `trimul-f567-cute` patch is not part of the consumer's
24-patch installation and is not promoted by this integration.

## Resulting behavior

- TriMul uses packed contraction buffers without the three concatenation copies,
  the configured F567 Triton forward fusion, and the dual-dgrad and
  LayerNorm/residual backward fusions. Tiling and tile traversal use the shared
  helpers; native and Triton remain independently selectable.
- Large compute-efficient attention backward bounds its scratch workspace by
  processing chunks. It does not silently substitute the atomic algorithm.
- AdaLN alignment, mixed-affine LayerNorm, backend propagation, strict Triton
  selection, CUDA graph AMP, and FA4 compiled-backward fixes are included.
- Native tuning preserves per-candidate measurements and retryable failures,
  supports expanded configuration spaces, and validates implementation and
  measurement provenance before reusing a winner.
- Wide bidirectional TriMul (hidden=512, packed reduction KP=4096) no longer
  overflows the shape-key radix. Existing small-width keys are unchanged;
  `KP_DIV8` stores the exact wide reduction in a distinct axis namespace.

## Merge decisions

The five textual conflicts were resolved by behavior:

1. `registry.csv`: merge each kernel's fields, retaining new rows from both sides.
2. `shape_key.py`: retain main's compiler-constant axis-name checksum helper.
3. `cache.py`: retain native provenance/lazy candidate handling and main's compact
   search-space compatibility. Normalize dictionary candidates before comparing
   signatures so unused Triton warp/stage fields remain optional for CuTe.
4. Attention interface: preserve the chunked compute-efficient algorithm and
   main's explicit compute/memory dispatch option.
5. SWA attention: retain main's FA2 packing/traceability and the consumer's opaque
   native FA4 backward. Both keep their existing recomputation semantics.

The upstream harness rules are also preserved: TriMul driver/check code lives
in its owning family, fake implementations use the established naming/schema
conventions, and GEMM metadata recognizes external CuTe GEMM calls.
The explicit H100 squeeze-residual D=512 driver stays reachable even though
current model-derived widths no longer include that optional configuration.

## Cache evidence

37 cache files no longer match the combined implementation: H100 28, A5000 5,
A6000 4. They are moved byte-for-byte to `incompatible-caches/`, with original
paths, reasons and SHA-256 in its manifest. Their measurements are preserved
for inspection, but are not shipped as runtime winners for a different kernel.
Compatible incoming caches remain in `autotune/data`.

The pre-existing A100 stale-cache exception remains; this integration does not
claim to rebuild A100, A5000, A6000 or all H100 configurations. Runtime cache
coverage and source integration are separate results. The ongoing consumer
cache job uses its own immutable pre-merge snapshot; its output must pass the
normal source/measurement identity checks before publication to this revision.

## Validation

Validation uses the remaining H100 allocation, with the four-GPU training and
three-GPU cache job unchanged. Commands and final results are recorded in
`validation.md`. H100 tests exercise FA4; the upstream FA2 graph suite requires
an sm80-family GPU and is not a claim of H100 validation.

The cross-architecture derivation recorder also declares its fake external
FlashAttention contracts available before entering SWA. Otherwise an H100
machine with only FA4 could not derive sm86: the real FA2 installation probe
rejected the module before its fake external kernel was reached. Only the
isolated derivation process changes these probes, and its fake core rejects
real tensors.

The ladder completeness guard now checks actual required model keys against a
verified architecture plan. An equal count of old, different shape keys cannot
justify narrowing the new workload's configuration space. It also excludes
winners from superseded kernel source, as its documented policy requires.
