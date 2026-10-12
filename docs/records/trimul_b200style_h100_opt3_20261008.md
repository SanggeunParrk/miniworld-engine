# TriMul B200-style H100: register reuse for input LN — 2026-10-08

## Result and selected path

D512 selected front retains half of the input-LN values in registers. Against the previously optimized B200-style opt-in, inference geomean is **1.02077x**, and complete training F+B geomean is **1.00511x** over B1/D512, outgoing and bidirectional, L384/L768. The training improvement is small. These are not speedups against the project default or external vendor kernels.

- Actual D512 forward dispatch is `mw_b200style_k1ssreg_n128t128s4p0w1c4`; backward recomputation remains the original SS kernel. D256/384 retain the original RS path.
- Input normalization previously reread all values for mean, variance and affine. The candidate caches the first 256 channels in registers, reading that half once and the remaining half three times. Per 128-token D512 tile, nominal input shared reads fall from 384 KiB to 256 KiB; writes remain 128 KiB. This is shared traffic, not an HBM reduction.
- The B200 fusion/recomputation contract is preserved: no normalized input or front preactivation is saved by forward. BF16 rounding, affine, mask, contraction, folded K3, dropout and residual formulas are preserved.
- Both selected forward variants (H512/H1024) have zero stack/local allocation and zero local spill instructions, max SASS register index 229, and 229888 bytes dynamic shared memory. Original backward max index 148, zero spills. This does not establish SOL90.
- Selection is only under existing `MINIWORLD_TRIMUL_H100_B200STYLE=1`. The project default flag value remains zero.

## Paired full-module timing

Unmodified `benchmarks/runners/bench.py`: BF16 activation/matrices, FP32 norm affines, B1, mask probability 0.1, dropout=0 in inference / 0.25 in training, torch.compile, manual CUDA Graph, 90 alternating AB/BA pairs after ten paired warmups. Same input/parameter states are checked. Training includes all input/parameter gradients, excludes optimizer. Job 27776 uses frozen snapshot-final, and baseline overrides only the D512 front back to original SS. Candidate uses the actual final selector. Eager profiles establish dispatch, not final timing.

| Mode | Direction | L | Previous ms | New ms | Speedup |
|---|---|---:|---:|---:|---:|
| inference | Outgoing | 384 | 1.333440 | 1.304544 | 1.02215x |
| inference | Outgoing | 768 | 5.297824 | 5.174848 | 1.02376x |
| inference | Bidi | 384 | 2.181088 | 2.143936 | 1.01733x |
| inference | Bidi | 768 | 8.867744 | 8.695008 | 1.01987x |
| training | Outgoing | 384 | 5.003616 | 4.971424 | 1.00648x |
| training | Outgoing | 768 | 19.733473 | 19.645632 | 1.00447x |
| training | Bidi | 384 | 8.254336 | 8.219040 | 1.00429x |
| training | Bidi | 768 | 33.327423 | 33.154911 | 1.00520x |

Within-run paired bootstrap (10000 resamples) gives positive 95% mean-time-delta intervals for all eight final cells; see summary.json. These intervals do not model all temporal correlation or between-job variation. An earlier pilot job 27768 showed 1.01987x inference and 1.00489x training at Bidi/L768, but its training interval crossed zero; it is not treated as independently conclusive.

## Project default comparison

Separate same-GPU paired job 27780 uses default selector=0 versus new opt-in=1. Do not combine absolute times across the two jobs into a three-way matched result.

| Mode | D | L | Default ms | New B200-style ms | Speedup |
|---|---:|---:|---:|---:|---:|
| inference | 512 | 768 | 7.849216 | 8.923328 | 0.87963x |
| training | 512 | 768 | 28.056513 | 34.419106 | 0.81514x |

The B200-style path still does not beat the project default at this measured cell. External kernels were not remeasured.

## Validation

- Frozen final job 27775: 49/49 unchanged integration tests pass, including independent FP32 reference output/all gradients, FP32 masters, BF16 matrices, all three directions, L128/384/768 coverage, actual module integration, front recomputation, zero masks, changed input/weight/mask/dropout/upstream-gradient CUDA Graph replay.
- Final memcheck: 25 tests, errors 0. Racecheck: five full native F+B/changed-graph safety scenarios, hazards/errors/warnings 0. Synccheck: 25 tests, errors 0. No sanitizer kernel filters or suppression. Sanitizer sizes are L128; wide numerical/performance evidence is separate.
- Live selected-source hashes match frozen final active sources. Live selected-path smoke evidence is recorded separately after promotion. Auxiliary experimental source files also present in live checkout affect the family-wide build digest; they are not dispatched, and active inputs/headers match the validated snapshot.
- GPU work uses normal_h100, one GPU per job, at most two of this task concurrently. Other sessions are left untouched. No commit or push.

## Rejected and limited experiments

- Atomic-free row gate / bounded output-LN partials: initial configuration passes 49 tests (27743); six D512/L768 training configurations measure 0.985..1.003x. These are not selected.
- Fused dpr@Wp + output-LN backward avoids materializing rounded dor in HBM, nominally removing 2.416 GB of write+read at Bidi/D512/L768. Initial compile error (27747) was corrected with a templated WGMMA width dispatcher. Base N256/MT128 then passes 18 gradient tests (27749), full49 and all three sanitizers (27751), but whole training is slower (27750).
- ldmatrix epilogue variant passes 18 gradient cases (27755), N256 full-workload candidates remain below 1x (27759). N128/MT64 with a runtime two-resident-CTA check passes 18 cases (27760) and measures 0.99548x; deeper stages fail the occupancy guard (27761). No fused-output variant is selected.
- N128/MT128 DT candidate and CACHEK=2 register candidate fail during multi-candidate graph replay in jobs 27750/27770. Root cause is unconfirmed; they are rejected in those runs. A standalone fresh-process N128/MT128 wide stage passes memcheck plus eager/graph replays (27766). Fresh-process CACHEK=2/3 official pairs (27772/27773) complete, but give smaller gains than CACHEK=4. This does not prove the multi-candidate failures are harmless or identify their cause.
- Register attempt v1 spills a 256-byte array (112 local instructions) and loses inference speed (27763). Fully unrolling the whole input still spills (v2). Caching half the input gives zero spill forward/backward (v3), passes49 (27769), and improves paired timing (27768). Final selection delegates backward to the original SS front and retains only the qualified forward change.
- No exhaustive configuration coverage, incoming-wide timing, FP32-master timing, B>1 qualification or SOL90 claim.

## Evidence

- [Results, source manifests, original pair samples, logs and official artifacts](../../runs/benchmarks-h100/benchmarks/modules/triangle_multiplication/results/h100/b200style_opt3_20261008/).
- Frozen snapshots/reproduction scripts: `/home/psk6950/miniworld-engine-scratch/trimul-b200style-opt3-20261008/`.
- [Previous optimized port](trimul_b200style_h100_opt_20261007.md), [previous unsuccessful fusion search](trimul_b200style_h100_opt2_20261007.md).
