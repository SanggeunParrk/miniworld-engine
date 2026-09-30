# Running TriMul from the Anthropic payload

> **2.2.0:** the default H100 dispatch already runs the same K1/K3 sources natively
> (`kernels/trimul_inproj/cuda/h100_inference.py`), so this payload is only needed to benchmark
> or reproduce the external route. Its builder was archived with `experiments/`.

The TriMul kernels of Anthropic's `uplifting-biomolecular-modeling` release can be run by the module layer
instead of its own. The kernels are not vendored here: a *payload* is assembled from the pinned upstream package plus this
repo's overlay by `archive/experiments-20260928:experiments/trimul_k1k3_inference/build_payload.py`, and named at runtime.

```bash
mkdir -p /tmp/k1k3 && git archive archive/experiments-20260928 experiments/trimul_k1k3_inference | tar -x -C /tmp/k1k3| tar -x -C /tmp/k1k3
e=/tmp/k1k3/experiments/trimul_k1k3_inference
python $e/build_payload.py --upstream <pkg/v5> --jobs 8 --out $e/payload \
  --unit tmn90_z128_h128,tmn90_z128_h256,tmn90_z64_h64,tmn90_z64_h128
export TRIMUL_NATIVE_BUILD_DIR=$e/payload/build     # this is the opt-in
```

One payload can carry several units and a process loads exactly one payload, so build every width the model uses into it. A unit is
named by `(c_z, c_hidden)`, and the bidirectional module is ONE unit at twice the hidden width — its two directions share one input
LayerNorm and one 2·`d_hidden` output LayerNorm, so it is not two unidirectional calls, which would normalise each half separately and
compute a different function. A model usually needs more than one width: MiniWorld's trunk, MSA module and confidence head are pair
width 128, while its AF3 template embedder runs a per-template pair trunk at `num_channels = 64`.

| module | pair width | unit |
|---|---|---|
| `TriangleMultiplication` | 128 | `tmn90_z128_h128` |
| `BidirectionalTriangleMultiplication` | 128 | `tmn90_z128_h256` |
| `TriangleMultiplication` | 64 | `tmn90_z64_h64` |
| `BidirectionalTriangleMultiplication` | 64 | `tmn90_z64_h128` |

A width with no unit in the payload is an ordinary fallback, so a missing one costs speed, not correctness.

## What each option does

| `implementation=` | with a payload that can serve the call | otherwise |
|---|---|---|
| `"miniworld"` (auto) | uses it | falls back to this engine's own backends, silently and correctly |
| `"anthropic"` | uses it | **raises**, with the reason — it never reroutes |
| anything else | untouched | untouched |

`integrations.anthropic_trimul.refusal()` is the single list of reasons, and all of them are ordinary fallbacks for the auto option:
no payload named, autograd enabled, a live row-dropout scale, a non-bf16 pair, a non-CUDA device or one that is neither sm_90 nor
sm_100, more than one square pair plane, no unit for the module's `(c_z, c_hidden)`, or (sm_100) no sm_100a build of the sm_80
member in the payload. **Training is therefore never affected**: a forward under grad, or with
dropout active, always takes the engine's own path.

One direction goes through `trimul_native.face.serve`, which applies the release's manifest and test-vector gate (so a payload must
keep its `python/`, `csrc/` and `testvectors/` beside `build/`, and a rebuild must regenerate vectors). The bidirectional composition
has no face entry and is composed in the adapter from the package's own primitives after the same `face.check()`.

## B200 (sm_100)

No binary for sm_100 ships with the release: its `sm_90a` units are TMA + WGMMA and cannot run there. Its **sm_80 member**
(`csrc/sm80/trimul_k1_sm80.cu`, `trimul_k3_sm80.cu`: cp.async + mma.sync) can, once compiled for `sm_100a`. The command below does
that with the release's own builder (`trimul_native.build.build_unit`: the release's fixed flags, source hashes and ptxas report,
recorded in `build/manifest.json`); the sources stay unmodified, and their `// build: archs=sm_80` line only filters the builder's
default job list.

```bash
cp -a <upstream>/common/opt_core/opt_core/kernels/trimul/native/pkg/v5 $PAYLOAD      # a copy: the checkout stays as shipped
miniworld-engine dev build-anthropic-sm100a $PAYLOAD/build --nvcc /usr/local/cuda-13.1/bin/nvcc   # ~1 min
export TRIMUL_NATIVE_BUILD_DIR=$PAYLOAD/build            # the module layer (implementation="anthropic" / "miniworld")
export ANTHROPIC_TRIMUL_BUILD_DIR=$PAYLOAD/build         # the harness's `anthropic` rows (a rebuild of unmodified sources)
```

What runs: `integrations.anthropic_trimul` registers cc 10.0 as `sm_100a` in the release's loader, which then loads
`build/sm_100a/trimul_k{1,3}_sm80.cubin` manifest-verified, and assembles the member as `sm80_ops.serve_sm80` does (that function
admits cc 8.x only): K1 -> `torch.bmm` -> K3 with the residual fused. The bidirectional module is ONE unit at twice the hidden
width, composed as on sm_90 (natural planes, outgoing half NT, incoming half TN). Served: the member's tile rows,
`sm80_ops.TILES` -- one direction at D64 / D128 / D256 / D384, bidirectional at D64 / D128 (units h128 / h256); every other width
refuses with the reason (bidirectional D256+ and D512 have no unit). The release's test-vector gate has no sm_100 class, so
`tests/integrations/test_anthropic_trimul_b200_gpu.py` holds the path to the fp32 reference (and to CUDA-graph capture) instead.

## Measured, node02 H100, bf16, C128, module level, one-call CUDA graph

Every path built once and timed in the same session (3 x 100 replays after 20 warm), so the columns are comparable to each other:

| case | L | payload | this engine's Triton | this engine's CuTe (the sm_90 default) | vs Triton | vs CuTe |
|---|---:|---:|---:|---:|---:|---:|
| outgoing | 384 | **143.3 µs** | 265.8 | 283.9 | 1.85x | 1.98x |
| outgoing | 768 | **552.3** | 970.5 | 1036.6 | 1.76x | 1.88x |
| incoming | 384 | **145.5** | 271.3 | 289.4 | 1.86x | 1.99x |
| incoming | 768 | **572.9** | 978.1 | 1051.1 | 1.71x | 1.83x |
| bidirectional | 384 | **244.0** | 441.9 | 586.6 | 1.81x | 2.40x |
| bidirectional | 768 | **1015.2** | 1664.2 | 2110.6 | 1.64x | 2.08x |

**Read the CuTe column with care.** On this checkout that path has no tuned autotune cache for its ops on an H100 — it warns
(`no tuned autotune cache for op 'trimul_inproj_masked_sm90_cute' ... falling back to the full autotune grid`) and picks a config off
the full grid. It is what `implementation="miniworld"` runs today without a payload, so the ratio against it is the honest "what does
this change for a run right now" number, but it is NOT a tuned-kernel comparison. Triton is the engine's best measured path here, and
against it the payload is 1.64–1.86x. That the gap between the two engine backends is much wider for the bidirectional module
(CuTe +33 %) than for one direction (+7 %) is a dispatch/cache question this page does not settle.

### Against the release itself

The same modules on the same clock, served by a payload built from the pinned upstream with no overlay and no switches, and by ours
(three interleaved rounds of separate processes per row, `records/bidirectional/vs-upstream/`; both payloads reproduce the fp32
reference at the same rel-RMS, so this is a like-for-like timing):

| case | L | upstream v5 | ours | |
|---|---:|---:|---:|---:|
| outgoing | 384 | 166.8 µs | **143.6** | −13.9 % (1.16x) |
| outgoing | 768 | 625.3 | **554.9** | −11.3 % (1.13x) |
| incoming | 384 | 165.0 | **145.0** | −12.1 % (1.14x) |
| incoming | 768 | 634.3 | **569.4** | −10.2 % (1.11x) |
| bidirectional | 384 | 280.2 | **245.6** | −12.3 % (1.14x) |
| bidirectional | 768 | 1118.3 | **1003.2** | −10.3 % (1.12x) |
| outgoing, width 64 | 384 | 92.2 | **80.5** | −12.7 % |
| outgoing, width 64 | 768 | 323.6 | **300.8** | −7.1 % |
| incoming, width 64 | 384 | 89.9 | **80.5** | −10.5 % |
| incoming, width 64 | 768 | 327.9 | **301.5** | −8.0 % |
| bidirectional, width 64 | 384 | 135.0 | **124.2** | −8.0 % |
| bidirectional, width 64 | 768 | 511.7 | **480.8** | −6.0 % |

Round spread 0.1–4.0 µs. The ceiling is the contraction: it is a third of the op, already at its DRAM/tensor balance point, and
untouched by any of this — K1 is 19 % faster and K3 9 %, and that is what 10–14 % of the whole op looks like.

`archive/experiments-20260928:experiments/trimul_k1k3_inference/README.md` has the per-kernel breakdown, the tile choices and the rejected candidates;
`records/bidirectional/` has the raw rows.
