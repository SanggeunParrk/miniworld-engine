# miniworld-engine 2.5.14

Base: 2.5.13 (b65376704), release candidate `rc/2.5.14` (= 2.5.13 + `cand/a100-port` + `cand/5090-port`, both on 6bb7d21fb). A100 and
RTX 5090 only; no H100 (sm_90) source changed:

1. **A100** (sm_80 build): the generalised TriangleAttention inference core (`attn2_ops.cu`) runs the register-resident core
   (`attn_rf_fwd_sm80.cuh`) from L 128 (64-channel heads) / L 256 (32- and 16-channel heads), was L 512 / 1024. **Outputs change at those
   L, inside the engine accuracy standard** (the rf core's error, which the L >= 512 cases already carried).
2. **RTX 5090** (sm_120a build): the register-resident core keeps 2 ring slots (was 4) for 32- / 64-channel heads, so an SM holds two CTAs.
   **Bit-identical.** The sm_120a build keeps the earlier rf floors (L 512 / 1024, under `TRIATTN_SM120`).

No module `version` bump. `backend_versions["cuda:a100_sm80"]`: `_TriangleAttentionBase` (TriangleAttentionAF3, RF3, Chai1,
Chai1Confidence) 5 -> 6, `TriangleAttention` 6 -> 7. H100 / RTX 5090: no version or token change.

## Changes

| # | Card | Change | Outputs | Versions | Measured (new / old time) |
|---|---|---|---|---|---|
| 1 | A100 | `attn2_ops.cu`: rf floors `RF_MIN64 / RF_MIN32 / RF_MIN16 = 128 / 256 / 256` and the 16-channel fixed-shift automatic floor 1024 -> 256 (sm_80 build); `#ifdef TRIATTN_SM120` keeps 512 / 512 / 512 / 1024 for the sm_120a build of the same source | changed at L 128-384 (64-ch) / 256-384 (32-ch) / 256-768 (16-ch), within the standard | `cuda:a100_sm80` base 5 -> 6, TriangleAttention 6 -> 7 | core 64-ch 1.39-1.62x (L 128-384), 32-ch 1.27-1.44x (L 256 / 384), 16-ch 1.04-1.12x (L 256-768); module 1.02-1.19x |
| 2 | RTX 5090 | `attn_rf_fwd_sm80.cuh`: `RF_ST16 / RF_ST32 / RF_ST64` ring slots of the dispatched rf geometries, used by `ops.cu` and `attn2_ops.cu`; sm_120a: 4 / 2 / 2 (64 / 80 KiB of shared memory left the 5090, 99 KB a block / 100 KB an SM, one CTA an SM), sm_80: 4 / 4 / 4 (A100 sm_80 SASS identical) | bit-identical | - | core 64-ch 1.15-1.17x, 32-ch 1.12-1.15x; module 1.04-1.09x (32 / 64-channel heads), 16-ch 1.000 |
| 3 | both | `tests/gpu/test_triattn_sm80_rf_core.py` runs on the A100 and the RTX 5090; the floors are card-aware (`_rf_floor`), new `test_rf_floor_default_rule`, `test_rf_core_attn2` also at the floor L | - | - | - |

### A100 port check of the 2.5.13 H100 items

* **TA_RC (peeled last key tile)**: the cause does not exist on sm_80. No A100 core has a last-tile branch in its key loop or a
  loop-carried gate prefetch (the gated cores read g after the loop); the rf cores have no spills (`cuobjdump -res-usage`: 199 / 224 /
  166 registers at 64 / 32 / 16-channel heads, stack 0). Not ported.
* **64-channel fp16 core**: the A100 rf core already has the main ideas (bias as the score accumulator's C operand, no softmax shift, gate
  after the loop) and runs at parity with SDPA flash *without* bias (L 512-2048, 8 heads: 152-169 TFLOP/s, 0.95-1.01x flash, 3x SDPA
  efficient / cuDNN with the bias). sm_80 has no bf16x2 exponential and its fp16-accumulate mma is not faster. Not ported. What paid
  instead is change 1: below L 512 the attn2 core was 1.4-1.6x slower at 64-channel heads.
* **Transition D 768**: the A100 route has no such entry (D 768 inference runs `fused_wide_sm80` without saves, 1.13-1.25x the bare cuBLAS
  GEMM sum). Not ported.

## Bit-identity (sha1 of the TriangleAttention module outputs, cuda backend, eval, bf16, master weights, last L/8 keys masked)

Compact set: 11 configs (TriangleAttention 64:4:16 / 64:2:32 / 128:4 / 256:8, AF3 64 / 512, RF3 64 x 4 x 256 / 128, Chai1 64 x 4 x 128 /
256, Chai1Confidence 256) x L 128 / 256 / 512 x starting / ending = 66 outputs; each tree in its own process.

| card | compared | identical |
|---|---|---|
| H100 | rc/2.5.14 vs 2.5.13 (b65376704) | 42 / 42 measured (TriangleAttention 64:4:16 / 64:2:32 / 128:4 / 256:8, AF3 64 / 512 (L 128 / 256), Chai1 64 x 4 x 128); the rest not run: no sm_90 source changed and the sm90a prebuild keys equal 2.5.13's |
| A100 | rc/2.5.14 vs `cand/a100-port` (3142ac436) | 66 / 66 |
| RTX 5090 | rc/2.5.14 vs 2.5.13 (b65376704) | 66 / 66 |

* `cand/5090-port` (5090, one process, 2 slots vs the same tree at 4 slots): 512 / 512 module cases identical (13 classes x starting /
  ending x L 512 / 640 / 768 / 1024 / 1536 x B 1 / 2 x key mask none / last L/8 padded); sm_120a SASS at 4 slots equals the base's.
* `cand/a100-port` (A100): outputs change only on the rows of change 1; d_pair 64 x 2, 64 x 4 x 32 (starting), 128 x 4, 256 x 8, 384 x
  12, AF3 128 / 256 bitwise unchanged.

## Accuracy (engine standard, `measure/accuracy.py`: per tensor max|x - gold| / max|gold| vs fp64 PyTorch, tol = max(err_pytorch, 8e-3))

* Change 1 (`cand/a100-port`, A100, rows 71-85, eval, L 128-768, both trees): 180 / 180 cases same light; the 15 whose error moved are
  GREEN (new / old 0.90-1.18, worst err / tol 0.53). Core alone vs fp64 at L 512: rms 6.5e-4 / 7.8e-4, max 8.5e-3 (the rf core's own).
* Judge re-measure of every owed A100 cell: see Judge.

## Speed

A100 80GB PCIe (`cand/a100-port`; CUDA graph, interleaved, new rule vs the rc rule in one process, same-graph control 0.99-1.01x), module
eval, bf16 pair, master weights, 1/8 masked keys: AF3 512 x 8 1.05-1.12x, RF3 64 x 4 x 64 1.06-1.19x, RF3 128 x 4 1.03-1.10x (L >= 256),
Chai-1 256 x 4 1.05-1.14x, Chai-1 confidence 1.04-1.12x, Chai-1 64 x 4 x 32 ending 1.04-1.10x (L >= 256), d_pair 64 x 4 x 16
(TriangleAttention / AF3) 1.02-1.06x (L 256-768). Unchanged (bitwise): d_pair 64 x 2, 64 x 4 x 32 (starting), 128 x 4, 256 x 8, 384 x 12,
AF3 128 / 256. Error of the update vs the fp64 module unchanged (rms within 1 %).

RTX 5090 (`cand/5090-port`; one process, CUDA graph, interleaved; old = the tree built with 4 slots, ctl = the tree rebuilt with an unused
define, ctl / new 0.998-1.002), module eval, master weights, bf16 autocast, starting L 512-1536 + ending L 768: AF3 512:8 1.04-1.06x,
RF3 64 x 4 x 256 1.08-1.09x, Chai1 256:4 1.05-1.07x, Chai1Confidence 1.04-1.06x, TriAttn 128:4 / AF3 128:4 1.06-1.08x, 256:8 1.04-1.06x,
384:12 1.05-1.06x, RF3 128:4 1.06-1.07x, Chai1 64 x 4 x 128 / 64:2 1.07-1.09x; 16-channel heads (64:4) 1.000. Core alone: 64-channel
1.15-1.17x (now 198-214 TF/s, 1.01-1.03x of bias-free SDPA flash), 32-channel 1.12-1.15x. Cool / sustained are not separated
on these two cards (interleaved rounds, ratios only).

## Judge (version-based tokens)

* `judge owed` on the merged tree before the re-measure: A100 460 accuracy + 460 speed (`adapter:cuda 5 -> 6` 258: AF3, RF3, Chai1,
  Chai1Confidence; `6 -> 7` 202: TriangleAttention); H100 and RTX 5090 0.
* Re-measured once on the A100 (A100 80GB PCIe, cssb3) with `tools/slurm/cycle.sh` on the rc tree (40 groups, 1-GPU 10-minute jobs):
  1840 tokens issued.
* Accuracy lights old -> new: GREEN -> GREEN 660, YELLOW -> YELLOW 74, skipped -> skipped 166, error -> error 18 (TriangleAttention
  128 x 4 master training on the A100, as before), GREEN -> skipped 2 (TriangleAttention 256 x 8 starting L640 master **train**, cuda and
  pytorch: the fp64 reference ran out of memory this time; the training path did not change). **No light got worse, RED 0.**
* Speed (speed-up vs PyTorch, new / old, every re-issued cell, median / min / max): TriangleAttention 1.000 / 0.953 / 1.062,
  TriangleAttentionAF3 1.000 / 0.919 / 1.089, TriangleAttentionChai1 1.000 / 0.953 / 1.115, TriangleAttentionChai1Confidence 1.000 /
  0.976 / 1.131, TriangleAttentionRF3 1.000 / 0.963 / 1.206 (most re-issued cells run unchanged code; the spread is run-to-run).
* RTX 5090: bit-identical, no version bump, its speed tokens are left as they are (conservative: the 5090 is 1.04-1.09x faster than they
  record).
* `judge owed` after the cycle: H100, A100, RTX 5090 accuracy and speed owed 0, pending 0 (17719 carried each).

## Tests and checks

* A100 (rc tree): `test_triattn_sm80_rf_core.py` 47 passed; `test_determinism.py -k riangle` 8 skipped (H100-only cases), 0 failed.
* RTX 5090 (rc tree): `test_triattn_sm80_rf_core.py` 47 passed (the card-aware floors).
* H100: no sm_90 source changed; TriangleAttention sha1 above.
* CPU on the rc tree: ruff check / format clean, pyright 0 errors (touched files); `test_requirements.py` 1167 passed;
  `test_judge.py`, `test_requests_api.py`, `test_init.py`, `test_anthropic_kit_rows.py`, `test_capture_api.py`, `test_prebuild_plan.py`
  318 passed, 2 skipped.

## Packaging: prebuilt artifacts whose keys change

`_prebuild --list` / `--check` against the 2.5.13 tree: same artifact counts (sm90 513, sm80 72, sm120 64), the same names (none dropped,
none added), `--check` findings identical on every arch.

* **sm80**: `triattn_sm80` (d424530f -> ffed6700) and `triattn_sm80_core2` (df48da52 -> d5183bbd) re-keyed, same names.
* **sm120**: `triattn_sm120` (54438543 -> c905fcb8) and `triattn_sm120_core2` (170a3638 -> 10f7e222) re-keyed, same names.
* **sm90**: no change.

## Release build (2.5.14)

* Incremental prebuild from the 2.5.13 release cache (GCC 12.4 host compiler): sm90 0 built, sm80 2 (`triattn_sm80` [ffed6700],
  `triattn_sm80_core2` [d5183bbd]), sm120 2 (`triattn_sm120` [c905fcb8], `triattn_sm120_core2` [10f7e222], `sm_120a`); 0 failed, 0
  absent. Artifacts: sm90 513, sm80 72, sm120 64 (as 2.5.13).
* Manifest names vs the 2.5.13 upload: identical on sm90 / sm80 / sm120 (none dropped or added). Kernels wheel file lists: sm90 identical
  to 2.5.13; sm80 / sm120 differ only by the two re-keyed directories each. Carried byte-for-byte as before: `triattn_h100_core_h16_v2`
  (sm90, from 2.5.5). Engine wheel file list identical to 2.5.13's (modulo the version).
* Installed-wheel checks (wheels unpacked, no repo / CUTLASS / nvcc, empty JIT cache): A100: 72 / 72 sm80 artifacts resolved by the
  runtime loaders, 0 compiled, `test_triattn_sm80_rf_core.py` 47 passed, 0 builds; RTX 5090: `test_triattn_sm80_rf_core.py` 47
  passed, 0 builds; H100: import, capture API and init presets from the installed wheels OK; the sm90 kernels wheel payload is byte-identical to 2.5.13's (every file CRC; its 513 / 513 load check carries), the full sm90 load pass did not fit a 10-minute job on the slow shared filesystem today
* PyPI sha256 (= `dist/SHA256SUMS`): engine wheel `e74b0ceb...`, sdist `fdc66f12...`, sm90 `666476fc...`, sm80 `dc4992b9...`, sm120
  `35de25cd...`.
