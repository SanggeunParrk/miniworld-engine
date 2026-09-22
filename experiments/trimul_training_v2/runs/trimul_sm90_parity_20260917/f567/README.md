# H100 F567 parity implementation

`parity_f567.py` is a new implementation of the current Triton F567 boundary:

- Consumes already materialized/rounded F4 `te_xn`, raw input `x_n`, `Wp`, and `Wg`.
- Independent projection and gate K extents; no LN affine folding or extra bias.
- FP32 accumulators, BF16 projection and BF16 gate logits, FP32 sigmoid for the output, BF16 saved gate.
- Dropout scale and residual stay in this epilogue, with the original row modulo L broadcast.
- Returns separate contiguous `(y, projection, gate)` buffers.
- TMA input/residual loads and output stores; explicit WGMMA GEMMs; no transpose-copy.
- Strided/transposed aligned weights supported using their actual row/column-major TMA layouts.
- Explicit `num_stages` ring buffers. A slot is recycled only after WGMMA completion and CTA synchronization. Ring slots saturate at the actual K trip count.
- `GROUP_M` uses the same grouped schedule and partial last-group handling as Triton.
- Both 4 and 8 warp schedules are mapped to physical warp groups across M/N. One/two warp schedules and M16/M32 do not have whole WGMMA instruction groups and are reported infeasible, without rounding their logical tiles.

The declared candidate domain comes directly from `trimul_output_f567_train_triton.csv`; no separate reduced CSV is maintained. The kernel reports rejected configurations and shared-memory limits to the native cache resolver. This is opt-in experimental code, not a claim of a performance win.

## Reuse decision

The old standalone F567 implementation provided useful TMA/WGMMA plumbing, grouped scheduling, saved-output stores, and BF16 epilogue structure. Its whole-K staging, folded affine weights/bias interface, and transpose-copy wrapper were not reusable as the required production algorithm. Those parts were replaced.

## Evidence files

- `validation.json`: initial 24 layout/tile/tail checks against the actual Triton kernel. Saved projection/gate matched exactly; maximum output relative L2 1.43e-5.
- `memcheck.log`: initial 16 selected tests, 0 sanitizer errors (before expanded warp mapping and final performance edits).
- `benchmark_final.log`: the first 72-candidate/shape sweep. This implementation was slower than the production Triton selection.
- `compare_epi.json`: output staging comparison when available.

## Final selected source validation

- `memcheck_final.log`: **24 tests passed, Compute Sanitizer 0 errors**. This includes both 4/8-warp mappings, partial M/N/K tiles, independent K extents, pipeline stages 2/3/4, grouped scheduling, and both weight layouts.
- Strict output relative-L2 tolerance is 1e-4 (0.01%).
- `final_benchmark.json`: final source measured after the expanded warp mapping and WGMMA chain overlap edits. This retests four candidates containing the earlier 72-candidate sweep winners and newly supported warp mappings. It is not a full config-space tune.
- Production Triton comparison uses the runtime's heuristic 24-candidate cache-miss selection; neither side is advertised as exhaustively tuned.
- Benchmarks are F567 only, BF16, M=L*L, KP256, KG128, N128, CUDA graph GPU timing. They are not complete TriMul module timings.

| L | Triton us | CuTe us | Triton/CuTe |
|---|---:|---:|---:|
| 128 | 11.343 | 15.817 | 0.717x |
| 384 | 106.290 | 151.682 | 0.701x |
| 768 | 414.277 | 542.944 | 0.763x |

The new parity kernel is currently **slower** in these three cases. Leave it opt-in. The old TMA/WGMMA plumbing is reusable, but merely replacing the instructions does not establish a speedup. Direct global output stores were rejected because they were substantially slower. Three independent output shared tiles were also tested and did not give a consistent win; the final source keeps one reusable TMA output tile.

Final source SHA-256: `71f6f5de85210da25d88821bedfd0d6cd3ce31c4252eb367b88b082783f6a9ba`.

`f567.ptx` and `ptx-evidence.json` confirm compiled `.target sm_90a`, 24 WGMMA instruction lines and 28 TMA instruction lines for the selected probe. PTX was retained using `cute.compile(..., options="--keep-ptx")`; the environment-only dump attempt did not retain artifacts.

Native tuning callback preparation was subsequently fixed: buffers, tensor validation, transpose views, DLPack adapters, and cache-key prefix are prepared once; candidate replay only dispatches to those prepared buffers. GPU kernel math and generated class code were unchanged. `preallocated-tests.log`: 24 selected GPU tests passed after this wrapper change.
