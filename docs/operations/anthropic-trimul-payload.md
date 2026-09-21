# Running TriMul from the Anthropic payload

The TriMul kernels of Anthropic's `uplifting-biomolecular-modeling` release are faster than ours on H100, so the module layer can
run them instead of its own. The kernels are not vendored here: a *payload* is assembled from the pinned upstream package plus this
repo's overlay by `experiments/trimul_k1k3_inference/build_payload.py`, and named at runtime.

```bash
e=experiments/trimul_k1k3_inference
python $e/build_payload.py --upstream <pkg/v5> --unit tmn90_z128_h128,tmn90_z128_h256 --out $e/payload --jobs 8
export TRIMUL_NATIVE_BUILD_DIR=$PWD/$e/payload/build     # this is the opt-in
```

One payload can carry several units and a process loads exactly one payload, so build every width the model uses into it:
`tmn90_z128_h128` serves `TriangleMultiplication` (one direction) and `tmn90_z128_h256` serves
`BidirectionalTriangleMultiplication`, whose two directions share one input LayerNorm and one 2·`d_hidden` output LayerNorm and are
therefore ONE unit at twice the hidden width — not two unidirectional calls, which would normalise each half separately and compute a
different function.

## What each option does

| `implementation=` | with a payload that can serve the call | otherwise |
|---|---|---|
| `"miniworld"` (auto) | uses it | falls back to this engine's own backends, silently and correctly |
| `"anthropic"` | uses it | **raises**, with the reason — it never reroutes |
| anything else | untouched | untouched |

`integrations.anthropic_trimul.refusal()` is the single list of reasons, and all of them are ordinary fallbacks for the auto option:
no payload named, autograd enabled, a live row-dropout scale, a non-bf16 pair, a non-CUDA or non-sm_90 device, more than one square
pair plane, or no unit for the module's `(c_z, c_hidden)`. **Training is therefore never affected**: a forward under grad, or with
dropout active, always takes the engine's own path.

One direction goes through `trimul_native.face.serve`, which applies the release's manifest and test-vector gate (so a payload must
keep its `python/`, `csrc/` and `testvectors/` beside `build/`, and a rebuild must regenerate vectors). The bidirectional composition
has no face entry and is composed in the adapter from the package's own primitives after the same `face.check()`.

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

Round spread 0.1–4.0 µs. The ceiling is the contraction: it is a third of the op, already at its DRAM/tensor balance point, and
untouched by any of this — K1 is 19 % faster and K3 9 %, and that is what 10–14 % of the whole op looks like.

`experiments/trimul_k1k3_inference/README.md` has the per-kernel breakdown, the tile choices and the rejected candidates;
`records/bidirectional/` has the raw rows.
