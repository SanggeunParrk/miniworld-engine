# Complete bidirectional TriMul training: previous Miniworld versus current

2026-09-20. H100 node02; batch1/C128/H128 per direction; BF16 weights/activations, FP32 LN parameters; supplied dropout25% and pair mask.
All variants use explicit single-call CUDA Graph; 3 alternating blocks of 200 samples, pooled median.

| Scope | L | Previous Triton ms | Previous H100 ms | Current all CUDA ms | vs Triton | vs H100 | cuEq compiled ms | vs cuEq |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 학습 Forward | 384 | 0.536 | 0.458 | 0.461 | 1.161x | 0.993x | 0.554 | 1.200x |
| 학습 Forward | 768 | 2.146 | 1.803 | 1.765 | 1.216x | 1.021x | 2.157 | 1.222x |
| Forward + backward | 384 | 1.613 | 1.515 | 1.168 | 1.381x | 1.298x | 2.040 | 1.747x |
| Forward + backward | 768 | 6.455 | 6.008 | 4.607 | 1.401x | 1.304x | 7.965 | 1.729x |

## What is connected

- Forward: selected Anthropic-derived fused input LN + gate/projection, packed bidirectional cuBLAS contractions, fused output, all training saves.
- Backward B1-B4: dual_ln_prefetch, selected and retained by the Claude SoL report.
- B5-B6: unchanged cuBLAS contraction backward.
- B7-B12: L384 front_prefetch_lnpair_storepipe; L768 front_ring96_cache3_glu_ahead_u4_early_writer32.
- Fresh forward saves are rebound into both backward plans each call. All 11 gradients and per-step weight layout conversions, including B1 Wp transpose, are timed.
- Excludes optimizer, stochastic-mask generation, CPU/autograd dispatch and compilation. This is one TriMul operation, not full-model training.
- The inference-only K1/K3 payload does not provide these training saves and is not included. Forward rows are training-mode forward with saves and dropout, not inference.

## Baselines

- Previous Miniworld algorithms are rerun from the retained routes in the current checkout, with output_backend=triton. This is not a recreation of an old environment or whole historical commit.
- Previous Triton/H100 and cuEq use static fullgraph compile, dynamic=False, Inductor CUDA graphs disabled; explicit CUDA Graph wraps every timed path.
- Old H100 enables front/f567/dual_bwd with the previously measured configs in trimul_sm90_round2_20260917/module/measured-configs-L*.json. Actual resolver calls and configs are in the JSON.
- Triton uses the available cache and default 24 heuristic candidates on a miss. The input-dual backward misses its tuned cache. This is not a full-grid retuning claim.
- cuEq uses the same shared-256-channel-LN bidirectional primitive composition, ONDEMAND tuning, torch0.9.1 / ops-cu12 0.10.0.

## Validation and numerical scope

- B1 and B7 regional contracts are separately checked with identical inputs. B1 dg is bit-exact. Existing tight per-output limits remain unchanged.
- Applying the B7-only limits to a whole chain after changing B1 initially failed for dx/input-LN gradients. B1 changes propagate downstream. The combined chain is assessed with the existing B1 full-backward contract (relative L2 <=5e-4), while preserving the stricter regional checks.
- Combined eager and captured outputs: forward exact vs identical saved-forward reference; all 11 gradients <=5e-4. Maximum gradient relative L2: L384 2.9234e-4; L768 4.9876e-4.
- The exact measured combined graph also passed after modifying input x, an input weight and upstream dy in place. Restoring the inputs reproduces the forward exactly; both cooperative counters reset to zero.
- Cross-backend comparisons use their own BF16 rounding/saves and are not bitwise. Old Miniworld gradient differences versus the saved-forward reference peak at0.0727%/0.0600%; cuEq peaks at0.8054%/0.5119%. No convergence claim.

## Results interpretation

- Complete training is 1.381x/1.401x faster than old Triton and 1.298x/1.304x faster than old H100.
- Training forward versus old H100: L384 is0.7% slower, L768 is2.1% faster; the main gain is backward.
- Adding B1-B4 to the previous B7-only experiment reduces total times from1.296 to1.168ms and5.158 to4.607ms (same run), a further1.110x/1.119x.
- With both backward regions included, cuEq total comparison is1.747x/1.729x. This supersedes the earlier B7-only current row for complete-integration claims.

## Reproduction and artifacts

- compare_all_training.py --length 384 or --length 768 (space before the number in actual CLI). Use the existing anthropic_adoption_20260919/env.sh environment.
- validate_all_replay.py --length 384 or --length 768 validates the exact captured callable with changed inputs; it does not overwrite timing records.
- all-training-L*.json:600 raw samples/path, all block medians, precision checks, source hashes and old SM90 configs.
- all-training-replay-L*.json:changed/restored-input checks.
