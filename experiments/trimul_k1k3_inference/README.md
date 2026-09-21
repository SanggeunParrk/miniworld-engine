# H100 TriMul K1/K3 inference optimisation

Anthropic's published native v5 TriMul (`trimul_native`, three launches: K1 input-LayerNorm + gated projections → cuBLAS
contraction → K3 output-LayerNorm + gate/projection + residual) was already faster than our own inference kernels. This
experiment keeps that implementation and its contracts (bf16 tolerance class, one serve face, manifest and test-vector gates) and
rebuilds the C128 unit with a small overlay of compile switches and host-package extensions. It is an explicit experiment
served through `TRIMUL_NATIVE_BUILD_DIR`; installing the engine does not select it, and production dispatch is unchanged.

![K1 / K3 pipelines, changed operations, measurements](wiring.svg)

## Result

Engine-path TriMul op (the adoption benchmark `bench.py --family trimul --row native_rebuilt`: warm CUDA graph, 5 × 50 replays,
outgoing, residual, C128, bf16 pair mask), node02 H100 80 GB, CUDA 12.9, PyTorch 2.10 cu128, same session, mean of two runs
(`records/engine-bench/`):

| L | Anthropic v5 rebuild | round 3 (final2) | **this payload** | change |
|---:|---:|---:|---:|---:|
| 384 | 169.7 µs | 152.9 µs | **148.0 µs** | **−12.8 %** |
| 768 | 613.8 µs | 566.9 µs | **552.6 µs** | **−10.0 %** |

Per kernel (CUPTI device time, 60 eager calls, µs; `records/kernel-times/`):

| kernel | L | v5 rebuild | this payload, bf16 mask | this payload, bool mask |
|---|---:|---:|---:|---:|
| K1 `tmn_k1_z128_h128_b_t6x32_s8k2_m{1,2,3}_l2_v0` | 384 / 768 | 56.4 / 205.9 | 49.9 / 183.9 | 48.2 / 179.9 |
| cuBLAS `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | 384 / 768 | 39.3 / 178.0 | 39.4 / 178.4 | 39.3 / 178.5 |
| K3 `tmn_k3_z128_h128_b_t3x64_s8a1_l1` | 384 / 768 | 61.5 / 210.8 | 52.2 / 175.4 | 52.2 / 175.5 |
| other (mask cast + cuBLAS memset) | 384 / 768 | 5.6 / 6.0 | 1.0 / 1.0 | 1.0 / 1.0 |

Numerics: rel-RMS against the fp32 module 2.587e-3 at both lengths, unchanged from the v5 rebuild; the round-2 changes (tanh sigmoid,
K1 LayerNorm class) move about 3 % of bf16 output elements by one ulp, every later change (bf16x2 residual, wait skip, K3 3 WG, mask
template, PDL) is bit-identical to its predecessor (`bench_op.py --save` outputs compared with `torch.equal`).

## What the overlay changes

`overlay/` holds the eight files that differ from upstream (`k1k3-inference.patch` is the same as a unified diff; `OVERLAY.json`
carries the base and overlay SHA-256 and the recorded build). Every kernel change sits behind a `TMN_*` switch whose default
reproduces the upstream bytes; `build_payload.py` turns them on:

| switch | kernel change | evidence |
|---|---|---|
| `TMN_SIGMOID_TANH` | gate sigmoid `rcp(1+ex2(-g·log2e))` → `fma(tanh.approx(g/2), .5, .5)`: one MUFU op, three dependent steps | K1 −4.2 %, K3 −6.3 % (XU pipe 42 → 25 %, 21 → 12 %) |
| `TMN_K1_FORCE_LNM` | K1 LayerNorm class 1 (`ln_fragment`) instead of the reference-order class 2 the tile table named for bf16 z | K1 −5.4 %, instructions −9 % |
| `TMN_RESIDUAL_BF16X2_NATIVE` | K3 residual `bf16(z + o)` as one `add.rn.bf16x2` (5 instructions before), bit-identical | K3 −3.1 % |
| `TMN_WSKIP` | resident weight ring: skip `mbar_wait(barW_full)` after tile 0 in K1 and K3 | K1 −0.7…2.5 %, K3 −1…2 % |
| K3 three consumer warpgroups | `K3Cfg` generalised to `NCWG`; `t3x64` (192 tokens per CTA, 512 threads, 160 registers) instantiated and tabled as the K3 default for N ≤ 1536 | K3 L768 190.1 → 175.0 µs, eligible warps 0.71 → 1.00 |
| `TMN_MASK_TEMPLATE` | K1 mask element type is a template parameter (`m2` = bf16, `m3` = bool/uint8 instantiations); the host package hands bf16 / bool masks straight to K1 (`MASK_CODE`, fp32-cast fallback where the unit lacks the instantiation) | removes the per-call fp32 mask cast kernel, 4.4 / 4.9 µs; a bool mask is a further K1 −4 µs at L768 |
| `TMN_PDL` | programmatic dependent launch: `griddepcontrol.wait` before the first dependent global access (producer and consumers), `griddepcontrol.launch_dependents` at the last tile; host `cuLaunchKernelEx` with the programmatic-stream-serialisation attribute (`TRIMUL_NATIVE_PDL`, default on) | K1 prologue overlaps the previous op's K3 tail: 2-op chain −1.9 µs at L384 |

Rejected on the same harness (details in the report): the `exact` variant, tile re-tuning, K1 start offsets, K1 and K3 bulk-store
switches, the K3 24/240 register split, K3 LayerNorm serial-affine off, K1 schedule 1, a 32-token TMA store box for K1, K1 → K3
LayerNorm statistics handoff (bit-identical but K3 +3 %, latency-bound), warp-local plane stores (2.3× slower), paired stores, gate
FMA fusion, a runtime mask-type branch (K1 +3 µs), a one-tile-ahead mask prefetch, and every cuBLAS form (NN layout, split batches on
two streams, all eight cuBLASLt heuristic algorithms).

## Where the remaining time is

* **The cuBLAS contraction is at its ceiling.** NCU at L768: DRAM read 304 MB + write 139 MB, exactly the essential bytes, DRAM 72 %
  of peak and tensor pipe 78 % active at the same time, with the SM clock power-throttled to 1.58 GHz; at L384 it is at the memory floor
  (74 %). The heuristic's first algorithm is the fastest of the eight (`probes/cublaslt-algos-L*.json`, `records/ncu-summary.md`).
* **K1 and K3 tiles stream at 93–95 % of the pattern floor** (2.85 TB/s, measured for this access pattern in the B1–B4 work): 12.5–12.7 K
  cycles per 192-token tile of 144 KB. What remains is structural: CTA startup (K1 4.2, K3 3.3–4.8 µs, 132 CTAs pulling their resident
  weights through L2 at once), the ragged last wave (3072 tiles / 132 CTAs = 23.27 at L768, ≈5.4 µs per kernel), and SM-to-SM speed
  variance (`probes/cta-timeline-L*.json`).
* Against a composite floor that keeps the three-kernel decomposition (K1 158 + contraction ≈159 + K3 158 µs at L768; ≈120 µs at L384)
  the payload sits at **86 % (L768) and 81 % (L384)**. Candidates not implemented, each worth ≤ 2 %: splitting the last wave into
  64-token warpgroup units (L768 −5 µs per kernel), 2-CTA cluster TMA multicast of the weight fill (K3 startup −2 µs), and handing the
  face a bool pair mask from the engine (K1 −4 µs at L768, one line at the call site).

## The bidirectional shape (c_hidden = 256)

A bidirectional TriMul (AF3's outgoing + incoming in one block, one shared input LayerNorm, one shared output LayerNorm over both
halves) is the same three kernels at twice the hidden width: K1 with `c_hidden = 2 x 128` in natural layout, the contraction split by
channel half (outgoing NT on the first 128 planes, incoming TN on the second), K3 normalising all 256 channels and projecting back to
128. Upstream ships the unit for it, `tmn90_z128_h256`. Measured with `bench_bidir.py` (the composition through `trimul_native.ops`)
and `bench_engine_bidir.py` (the engine's own path on the same inputs), node02 H100, bf16, B1, C128, pair mask, residual fused in K3,
one-call CUDA graph, three interleaved rounds of separate processes per row:

| L | engine CuTe (the engine's default today) | v5 `z128_h256`, pristine | **this payload's defaults** | per kernel, pristine → ours |
|---:|---:|---:|---:|---|
| 384 | 600.9 µs | 270.2 | **239.1 (−11.5 %)** — 2.51x the CuTe path | K1 107.6 → 87.3, K3 75.4 → 68.6, cuBLAS 84.0 → 84.3 |
| 768 | 2125.0 µs | 1112.3 | **1000.1 (−10.1 %)** — 2.12x the CuTe path | K1 392.6 → 319.3, K3 253.8 → 231.9, cuBLAS 367.6 → 368.8 |

The defaults are K1 `(3,64,8,2)` and K3 `(2,64,8,1)`, both selected by the tile table with no config override (the measured rows are in
`records/bidirectional/`). Round-to-round spread was 0.6–1.3 µs; every row matches the fp32 module reference at rel-RMS 2.585e-3 (the CuTe
path at 2.539e-3, its own rounding). The engine figures agree with the engine's own recorded bidirectional inference benchmark
(0.586 / 2.106 ms) to 2.5 %.

**What transfers from the 128/128 result and what does not.** The arithmetic switches (tanh gate, K1 LayerNorm class, bf16x2 residual)
and the host-side ones (mask element type, PDL) apply unchanged — K1's input LayerNorm is over `c_z`, which is still 128. K1's weight
stream is 256 KB, so no ring makes it resident and `TMN_WSKIP` is inert here. The unit is instantiated with the tile candidates the
128/128 unit carries, and the K1 default becomes the measured winner `(3,64,8,2)`, a 192-token tile with three consumer warpgroups:
**K1 −18.7 % at both lengths**.

**The K3 ring was wasting a third of its shared memory.** A K3 ring slot holds one output block's gate *or* projection rows, and
`SLOT_BYTES` sized every slot at the larger of the two. At `c_z = c_hidden` they are equal and nothing is lost, which is why this never
showed at 128/128; at `c_z 128 / c_hidden 256` a gate slot needs 8 KB and occupies 16 KB. The producer walks `seq = 2b` (projection)
then `2b + 1` (gate), so for an even `NSLOT` a slot's parity *is* its kind and the ring can pair one of each: `TMN_K3_PACKED_RING`,
which changes one address expression (`K3Cfg::slot_off`) and its host mirror, and is compiled out where `c_z == c_hidden`. It frees
`(NSLOT/2) x 8 KB` and with it the two K3 tiles that were over the 227 KB limit — the 8-slot ring (249 → 216 KB) and the 192-token
three-warpgroup tile (241 → 224 KB). Measured, both now built (interleaved, K1 fixed at its default, `records/bidirectional/k3-ring/`):

| K3 tile | L384 op | L768 op | K3 kernel |
|---|---:|---:|---|
| `(2,64,4,1)` the tabled 4-slot ring | 243.8 µs | 1014.7 | 73.0 / 242.6 |
| **`(2,64,8,1)` 8-slot ring** | **239.4 (−1.8 %)** | **1001.1 (−1.3 %)** | **68.5 / 233.0 (−6.2 / −4.0 %)** |
| `(2,64,6,1)` 6-slot ring | 240.3 | 1012.7 | 70.2 / 238.2 |
| `(3,64,4,1)` 192-token, three warpgroups | 271.0 (+11.1 %) | 1123.8 (+10.8 %) | 100.8 / 379.5 |

So the 8-slot ring becomes the K3 default, and the tile that carried the 128/128 K3 result loses here by 11 % — for a reason the
shared-memory limit had been hiding: at 512 threads the launch bound caps it at **128 registers** and its 256-channel epilogue spills
444 loads / 408 stores against the served tile's 64 / 44 at 168 registers. Shared memory was never the real wall for that tile at this
width; registers are.

Of the other K3 tiles, `(2,64,4,2)`, `(1,128,4,1)` and `(1,64,4,1)` all lose. The bf16 (m2) and bool (m3) mask instantiations cost the
same as fp32 here, so their value is only that the caller no longer pays the per-call fp32 cast (4.7 µs).

**Which of the 128/128 changes actually act at this width.** Not all of them, and the build flags do not say which:

| change | at c_hidden 256 | measured / reason |
|---|---|---|
| tanh gate sigmoid | acts | part of the K1 −18.7 % / K3 −3.3…−4.2 % |
| K1 LayerNorm class 1 | acts | K1's input LayerNorm is over `c_z`, unchanged at 128 |
| bf16x2 residual | acts | K3's residual is over `c_z` |
| K1 `(3,64,8,2)` tile | acts (new here) | K1 107.4 → 87.3 µs (L384), 392.8 → 319.3 (L768) |
| mask element type (m2/m3) | compiled, ties | bf16 and bool cost the same as fp32 here; the gain is only the caller's dropped 4.7 µs cast |
| `TMN_K3_PACKED_RING` | acts (new here) | frees 16 KB, which is what makes the K3 8-slot ring exist: K3 −6.2 / −4.0 % |
| `TMN_WSKIP` | K1 **inert**, K3 acts (new) | K1's `W_RESIDENT` needs `NSLOT >= NBLK·SPB = 16` and has 8. K3's needs `NSLOT >= 2·NB = 8`: the 4-slot ring missed it, the packed ring's 8-slot one reaches it, so the K3 weights are resident and the skip applies — part of why 8 slots beat 6 |
| PDL | compiled, **below the noise** | see below |
| K3 three consumer warpgroups | instantiable now, **loses by 11 %** | 128 registers at 512 threads; epilogue spills 444/408 |

PDL was measured with `bench_bidir_pdl.py` (on/off alternating in one process, one op and a two-op chain, outputs bitwise equal in every
case). At L384 it is zero to within 1.4 µs in both 8-round and 20-round runs. At L768 the two runs disagree in sign — 8 rounds gave
−10.0 µs single / −61.0 chain, 20 rounds gave +10.9 / −2.6 — with round spreads of 79–249 µs, so there is no effect this measurement can
resolve. That is consistent with what PDL does: it overlaps a fixed prologue with the previous kernel's tail, and here each kernel is
about three times longer than at 128/128, where the chain gain was 1.9 µs.

**Register spills.** Every K3 instantiation at this width spills (the served `t2x64_s8a1_l1`: 168 registers, 64 spill loads, 44 stores,
48 B stack; `t2x64_s4a2_l1` at 204/184 and the 192-token tile at 444/408), while the served K1 `t3x64` is at 128 registers with none.
The K3 launch bound (384 threads, 1 CTA/SM) caps the budget at 168 and the 256-channel epilogue does not fit it. The spill count does
not order the tiles by speed — `t1x64` spills 24/4 and is the slowest of them — and after the ring change K3 runs at 91 % of its
streaming floor, above K1's 83 %, so what is left there is bounded by 9 %.

**Where the remaining time is.** Against the same 2.85 TB/s pattern floor, at L768 the essential bytes are K1 756 MB, contraction
906 MB, K3 604 MB: K1 is at 83 %, the contraction at 86 % (629 TFLOP/s bf16 at the same time — the balance point the 128/128 audit
found), K3 at 91 %, the whole op at 79 % of the sum. The contraction is now 37 % of the op. Fusing its two GEMMs into one — which a
per-half transposed K1 store would allow, the way upstream's `INCOMING_MODE="kt"` transposes the unidirectional incoming direction —
was measured as an upper bound on the same buffers (`bench_contract_forms.py`): **5.8 µs at L384 and nothing outside the noise at
L768**, so it is not worth the kernel change.

## The template shape (c_z 64)

A model does not run TriMul at one width. MiniWorld's trunk, MSA module and confidence head are all pair width 128, but its AF3
template embedder runs a per-template pair trunk at `num_channels = 64`, so its bidirectional TriMul is `c_z 64 / c_hidden 128` — unit
`tmn90_z64_h128`, a different cell with its own tile table row. Upstream tunes K1 well there (six variants, 192-token tiles already the
default) but had instantiated only three bf16 K3 tiles, none of them the ones that won at the other widths.

Shared memory is not the constraint at this width — the tabled K3 uses 100 KB of 227 — so the missing tiles were simply instantiated
and measured (`records/width64/sweep/`, three interleaved rounds per row, K1 at its table default):

| K3 tile | L384 op | L768 op | K3 kernel |
|---|---:|---:|---|
| `(2,64,4,1)` the tabled tile | 122.6 µs | 479.4 | 36.3 / 122.7 |
| `(2,64,8,1)` 8-slot ring | 122.5 | 480.4 | 36.3 / 122.7 |
| `(2,64,6,1)` | 122.7 | 478.8 | 36.3 / 122.7 |
| **`(3,64,4,1)` 192-token, three warpgroups** | **120.8 (−1.5 %)** | **473.0 (−1.3 %)** | **34.4 / 115.4 (−5.5 / −6.6 %)** |
| `(3,64,8,1)` both | 120.6 | 472.8 | 34.3 / 114.6 |

Ring size does nothing here, and for a structural reason: K3's `W_RESIDENT` needs `NSLOT >= 2·NB = 4` at `c_z 64`, which the tabled
4-slot ring already satisfies — where at `c_z 128` it needs 8, which is why the 8-slot ring mattered there. The 192-token tile is the
win, and `(3,64,8,1)` ties it at 166 KB against 133, so the cheaper one becomes the default. Split-N cannot exist at this width at all:
`NB = c_z/32 = 2`, so a split warpgroup would get one output block and the epilogue finishes them in pairs (`K3Cfg`'s static_assert).

Against the release, module level, three interleaved rounds (`records/width64/final/`):

| case | L | upstream v5 | ours | |
|---|---:|---:|---:|---:|
| outgoing | 384 | 92.2 µs | **80.5** | −12.7 % |
| outgoing | 768 | 323.6 | **300.8** | −7.1 % |
| incoming | 384 | 89.9 | **80.5** | −10.5 % |
| incoming | 768 | 327.9 | **301.5** | −8.0 % |
| bidirectional | 384 | 135.0 | **124.2** | −8.0 % |
| bidirectional | 768 | 511.7 | **480.8** | −6.0 % |

Less than the 10–14 % at width 128, and the byte model says why: at L768 K1 is 154 µs and K3 115, but the op is 473 — the contraction
is **40 %** of it here against 37 % at `c_z 128`. The narrower the pair, the larger the share of the one kernel none of this touches.

Against the 2.85 TB/s pattern floor this width is close to done: at L768 K1 is at 86 %, the contraction at 84 %, **K3 at 92 %**, the op
at 84 % of their sum (L384: 85 / 88 / 77 %). One lever is left and it is measured rather than assumed — unlike at width 128, fusing the
two contraction GEMMs into one IS worth something here (`records/width64/c64-L*.json`, same buffers, rounds within 0.3 µs):

| | split NT+TN (today) | one NT, batch 128 | two NT |
|---|---:|---:|---:|
| L384 | 46.2 µs | **44.0 (−2.2)** | 46.4 |
| L768 | 191.6 | **180.9 (−10.7)** | 208.1 |

−1.8 % and −2.3 % of the op. Note `two_nt` is *slower*, so the gain is not the NT form, it is issuing ONE batched call instead of two
of half the batch — which at width 128, where each half is already 128 planes, measured as nothing. Collecting it needs either a K1 that
stores the incoming half transposed (upstream has that mechanism for the unidirectional incoming direction, `INCOMING_MODE="kt"`; this
would make it per-half) or a grouped-batched cuBLAS call with a different transpose per group, outside `torch.bmm`. Neither is done
here: 2 % of the one width that only the two-block template trunk uses did not justify a new kernel path.

## Reproduce

Needs an allocated H100 (sm_90a), nvcc 12.x, PyTorch CUDA with `cuda.bindings` or `libcuda.so.1` for the driver binding, and the
upstream package `common/opt_core/opt_core/kernels/trimul/native/pkg/v5` of
[anthropics/uplifting-biomolecular-modeling](https://github.com/anthropics/uplifting-biomolecular-modeling) at revision
`f4f62fa6592ae4938d49b1757bea0cfeff9f468e` (the engine's `scripts/import_anthropic.py` on the `perf/trimul-sm90-parity` line fetches it
and sets `MINIWORLD_ANTHROPIC_ROOT`). Build and test-vector generation compile and run kernels: do them on a compute node.

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
e=experiments/trimul_k1k3_inference
python $e/verify_package.py                                   # CPU: hashes, patch coverage, records
python $e/build_payload.py --upstream <pkg/v5> --out $e/payload --jobs 4   # nvcc → payload/build, then `vectors make --grid r2` (H100)
export TRIMUL_NATIVE_BUILD_DIR=$PWD/$e/payload/build
python $e/bench_op.py --length 384 --iters 60 --output op-L384.json                  # per-kernel CUPTI + one-call graph; --mask-dtype bool|fp32
python $e/bench_chain.py --length 384 --iters 60 --output chain-L384.json            # outgoing → incoming chain in one graph
python $e/bench_pdl.py --length 384 --output pdl-L384.json                           # PDL on/off, alternating replays, bitwise output check
python $e/bench_contraction.py --length 768 --output contraction-L768.json           # cuBLAS forms of the contraction (no payload needed)
python $e/build_payload.py --upstream <pkg/v5> --out $e/payload-probe --probe --grid smoke
TRIMUL_NATIVE_BUILD_DIR=$PWD/$e/payload-probe/build python $e/probe_cta_timeline.py --length 384 --output cta-L384.json

# the bidirectional shape: the same overlay, the c_hidden = 256 unit
python $e/build_payload.py --upstream <pkg/v5> --unit tmn90_z128_h256 --out $e/payload256 --jobs 4 --no-vectors
TRIMUL_NATIVE_BUILD_DIR=$PWD/$e/payload256/build \
  python $e/bench_bidir.py --length 768 --iters 40 --mask-dtype bool --output bidir-L768.json   # --configs sweep for the tiles
python $e/bench_engine_bidir.py --length 768 --output engine-L768.json                          # the engine's own path, same inputs
python $e/bench_contract_forms.py --length 768 --output contract-L768.json                      # one-GEMM contraction bound (no payload)
```

Serving from the engine: the `native_rebuilt` row of `miniworld_engine.integrations.anthropic.triangle_multiplication` (parity line) imports
`trimul_native.face` from `$TRIMUL_NATIVE_BUILD_DIR/../python`, runs `face.check()` and calls `face.serve(z, mask, direction=..., weights=...,
residual=..., cache=...)`; the same call works without the integration. The face refuses a payload whose test vectors or source digests
do not match its manifest, so a rebuilt payload always re-runs `vectors make`. `TRIMUL_NATIVE_PDL=0` disables the programmatic launch
attribute without rebuilding.

## Validation performed for this package

On node02 H100 (2026-09-20): `build_payload.py` from the pinned upstream reproduced the recorded unit (62 kernels, K1 `t6x32` m1/m2/m3
at 128 registers with zero spills), `vectors make --grid r2` produced 216 cases, and `bench_op.py` (bf16 and bool masks) /
`bench_pdl.py` / `bench_chain.py` at L384 reproduced the recorded kernel times (op 142.3 µs, other 0.9 µs, rel-RMS 2.587e-3 before and
after replay), the chain gain (+2.5 µs) and bitwise-equal PDL on/off outputs; the `--probe` build reproduced the per-CTA timeline and
`bench_contraction.py` the cuBLAS forms.
On node02 H100 (2026-09-21), for the bidirectional round: the `tmn90_z128_h256` unit built from the pinned upstream through the same
script (60 kernels), the tile table's default for the shape selected without a config override (`tmn_k1_..._t3x64`, op 241.3–244.3 µs at
L384 and 1012.4–1015.3 at L768, reproducing the explicit row), and the unchanged `tmn90_z128_h128` payload rebuilt from the edited
overlay still reproduces its recorded times through the face (L384 op 141.8 µs vs 142.3 recorded; L768 K1 187.8 / cuBLAS 178.6 /
K3 174.7 vs 183.9 / 178.4 / 175.4), so the unit-header and tile-table edits did not disturb the 128/128 result.
The engine-path numbers in `records/engine-bench/` were measured with the adoption benchmark of the parity line, not re-run from this
directory. `verify_package.py` is CPU-only. No CI claim is made for `experiments/`.

## Measurement caveats

* With PDL on, CUPTI's K1 duration includes the time its CTAs spend in `griddepcontrol.wait` behind the previous kernel's tail (L384
  +3.7 µs, L768 +7 µs as "kernel time"); judge PDL by graph wall time (`bench_pdl.py` alternates the two graphs in one process).
* L768 graph wall times on node02 varied by ±5–50 µs between rounds on the measurement day (dedicated GPUs; clock and power state); the
  per-kernel CUPTI times were stable to ±0.6 µs. Compare payloads only within one session, interleaved.
* The pattern floor (2.85 TB/s) is a measured streaming floor for this access pattern, not the 3.35 TB/s HBM peak; against the peak the
  kernels are at 73–77 %.

## Attribution and source integrity

Anthropic `uplifting-biomolecular-modeling` revision `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`, native v5 (`pkg/v5`, version
`v5 1.2.2`, Apache-2.0), supplies the kernels, the host package, the build and vector tooling and the driver binding. `OVERLAY.json`
records the SHA-256 of each upstream file this overlay replaces (they agree with the subset vendored in
`../trimul_b7b12/vendor/anthropic_v5/UPSTREAM.json`) and of each overlay file; `build_payload.py` refuses an upstream tree whose base
files differ. The `tmn90_z128_h128` (unidirectional) and `tmn90_z128_h256` (bidirectional) units are built; other widths take the upstream defaults
and were not measured.

## Files

`overlay/` (3 kernel headers, 5 host modules), `k1k3-inference.patch`, `OVERLAY.json`, `build_payload.py`, `fixture.py`, `bench_op.py`,
`bench_chain.py`, `bench_pdl.py`, `bench_contraction.py`, `lt_search.cu` (cuBLASLt algorithm sweep), `probe_cta_timeline.py`,
`bench_bidir.py`, `bench_engine_bidir.py`, `bench_contract_forms.py`, `bench_bidir_pdl.py` (the bidirectional round),
`verify_package.py`, `wiring.svg` (`wiring.ko.svg`: Korean original), `records/` (engine and kernel timings, probes, cuBLASLt sweep,
payload manifest, NCU summary, the full Korean report `report-2026-09-20.ko.md` with the per-round rejection tables, and
`records/bidirectional/` with the interleaved rows, the tile sweep, the mask-dtype rows, the confirmation/regression runs and the
c_hidden 256 payload manifest).
