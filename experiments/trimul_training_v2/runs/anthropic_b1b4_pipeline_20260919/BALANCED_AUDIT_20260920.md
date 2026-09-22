# B1–B4 training CUDA: asymmetric CTA roles, 2026-09-20

**Selected: dual_balanced.cu + dual_balanced.py.** Exact same-process
core.baseline() Triton/cuBLAS, dropout0.25: **L384 1.7018x; L768 1.9051x**.
L384 has little margin and misses the absolute173us target (176.00us).
This is an experimental plan validated through the full backward harness;
production defaults and autotune dispatch are unchanged.

Anthropic native v5 TriMul CUDA primitives are the foundation. Our previous
inference development was behind Anthropic's results; this work extends
their implementation and ideas to training. These speedups compare training
B1–B4 against Triton/cuBLAS, not against Anthropic inference.

## 1. Operation, shared memory, ownership and assumptions

```
y = dy * ds[row % L]
dp = bf16(y * gate)
dg = bf16(((y * proj) * gate) * (1 - gate))
dWg = bf16(sum_fp32(xn.T @ dg))
dWp = bf16(sum_fp32(dp.T @ norm))
dnorm = bf16(dp @ Wp)
xhat = (tri - mean) * rstd; h = dnorm * gamma
dtri = bf16(rstd * ((h - mean(h)) - xhat * mean(h*xhat)))
dgamma = sum_fp32(dnorm*xhat); dbeta = sum_fp32(dnorm)
```

Forward saves, six outputs and BF16 rounding points are unchanged. B5+
consumes dg/dtri on Triton. Internal scheduling changes: one cooperative
launch has40 weight CTAs and92 input/LN CTAs at UCOUNT132. Each role computes
dp independently; no global dp/dnorm. dy/gate are loaded by both roles.
No clusters/multicast and no separate dW kernel in the selected version.

Compile-time ratio10:23 gives20/46 at count66. Each role processes cyclic
64-row tiles: split+round*role_count. Empty CTAs publish zero partials.
L384: DW57/58 tiles, DX25/26. L768: DW230/231, DX100/101.
Each CTA has256 threads, two128-thread WGs. DW: three64x128 weight-gradient
tiles per WG,192 FP32 accumulators/thread. WG0 owns two dWg and one dWp
tiles; WG1 owns three dWp tiles. DX: each WG handles128 LN channels.
Raw tri and BF16 dnorm stay in registers through LN; no32KiB dnorm
shared-memory round trip.

Full dynamic SMEM map, half-open byte ranges; columns are alternative CTA roles:

| Bytes | DW role | DX role |
| --- | --- | --- |
| 0–16384 | dy → dp | dy → dp (slot0) |
| 16384–32768 | gate → dg | gate → row stats / mean / rstd / gamma |
| 32768–49152 | proj | tri → dtri (first half) |
| 49152–65536 | xn | tri → dtri (second half) |
| 65536–73728 | norm (part) | 8KiB warp dgamma/dbeta partials |
| 73728–98304 | norm (remainder) | unused |
| 98304–114688 | dy → dp (slot1) | dy → dp (slot1) |
| 114688–131072 | gate → dg (slot1) | gate → statistics (slot1) |
| 131072–147456 | proj (slot1) | tri → dtri (first half of slot1) |
| 147456–163840 | xn (slot1) | tri → dtri (second half of slot1) |
| 163840–196608 | norm (slot1) | resident Wp (first half) |
| 196608–229376 | unused | resident Wp (second half) |
| 229376–231424 | initialized, unused | running FP32 dgamma/dbeta sums |

Dynamic231424B (226KiB) +static1024B; oneCTA/SM. DW has two96KiB stages.
DX has two64KiB stages,64KiB resident Wp,8KiB warp sums and2KiB running sums.
Selected count66/132 × PART1/2:252 registers, zero stack/spills (hardware
allocation rounds to256). No setmaxnreg rebalance in this selected layout.

Initialize/proxy-fence two mbarriers before issuing TMA. First DW96KiB;
first DX128KiB including Wp. Later DW96KiB/DX64KiB. Slot=round&1;
phase=(round/2)&1. Refill a slot for tile+2*role_count after every consumer
finishes. Generic B1 stores are proxy-fenced before WGMMA; register fences,
commit/wait protect accumulators and operands. CTA barriers publish both
channel halves' LN statistics and all warp partials. Each WG fences dtri,
commits/waits TMA stores before slot reuse. All partial writers threadfence
before ticket publication; volatile final reads follow the grid barrier.
Last completion ticket resets both counters. Same-stream replay begins
after completion. PART1 uses a separate194-block reducer.

Assumptions: contiguous TMA-aligned BF16, C128/H256, L>=64 and L²%64=0,
int32 index range, one stream/workspace per non-reentrant plan. ds is zero
or a common positive BF16 dropout scale. Mask cache applies only when
L/gcd(64*role_count,L)<=3; longer periods use normal ds loads. No L384/L768
or p0.25 special case. Wp/saves are fixed for a plan. Packing/allocation/
compilation/descriptors precede graph capture and event timing.

## 2. Correctness issues in the original and audit findings

- Original host wrapper rejected count>row_tiles. Experiment wrapper allows
  empty CTAs needed for L64 tests; original dual.py remains unchanged.
- No new arithmetic/race defect was established in dual_dspref for required
  shapes. Prior descriptions of52MB as unavoidable HBM traffic and dedicated
  simultaneous dW/dX WGs were inaccurate.
- New slot/fragment lifetimes have explicit synchronization; numerical
  graph replay and requested sanitizer tools pass.
- Extra L72 seed20261151 repeats dtri relativeL2=2.1726077e-5, above the
  large-shape2e-5 bound. Original dual and selected dtri are bit-identical
  across three diagnostic seeds. No universal all-L/all-seed accuracy claim.
- SHA-256 of original dual.cu, dual_dspref.cu, dual_primitives.cuh, dual.py
  and core.py still matches the pre-experiment audit. Previous selected
  kernel is preserved for A/B.

## 3. Time split, bottleneck and roofline

Instrumented completion-frontier medians, us,20 diagnostic launches:

| L | Version | Body | Dump+publish | Grid wait | Reduce+publish | Reset |
| --- | --- | --- | --- | --- | --- | --- |
| 384 | original dspref | 223.056 | 18.416 | 0.624 | 6.240 | 0.288 |
| 768 | original dspref | 841.376 | 23.888 | 0.640 | 6.208 | 0.192 |
| 384 | previous selected | 194.080 | 5.568 | 0.576 | 5.920 | 0.256 |
| 768 | previous selected | 715.168 | 8.256 | 0.608 | 6.352 | 0.288 |
| 384 | balanced selected | 163.456 | 0.704 | 0.576 | 2.416 | 0.224 |
| 768 | balanced selected | 615.136 | 0.704 | 0.608 | 2.592 | 0.288 |

A frontier uses the latest CTA completion at each timestamp. Earlier DW
partial dumps overlap DX body;0.704us is exposed tail, not all DW store work.
Per-CTA waits can be much larger. These phases are not headline event timing.
32-bit timer probe rejects wrap-crossing launches, compiles without spills;
uninstrumented/instrumented174.032/173.872us and621.056/620.944us in its
process: no measurable probe penalty. Historical phase deltas are approximate.

Useful row bytes=(2056 read+768 write)*L²=416.416/1665.663MB. Supplied
3.35TB/s gives bandwidth-only minima124.30/497.21us;3.0TB/s gives
138.81/555.22us. GEMM work24.16/96.64GFLOP. This does not rule out1.7x,
but is not an achievable runtime promise. New roles request512B/row extra
dy/gate before L2 reuse: nominal3336B/row,491.913/1967.653MB. Measured HBM
is lower because requests may hit L2. Do not infer partial HBM by subtraction.

Partial bytes one direction=40*49152*4+92*512*4=8,052,736.
Read+write16.105MB versus52.445MB:69.3% less global traffic. Host capacity
remains the oversized26.223MB allocation; only touched regions shrink.
This is not a52MB HBM saving. Body dominates; final reduction about2.5us.

## 4. Ranked experiments and measured decisions

Conditional estimates, not measured savings or additive guarantees:

| Priority | Hypothesis; estimated saving L384/L768 vs previous selected | Result |
| --- | --- | --- |
| 1 | Asymmetric roles and fewer partials:20–40 /80–150us | Adopt40/92; final paired savings35.17 /121.68us. |
| 2 | Two32-row stages in one CTA with register LN:0–25 /0–100us | Built zero-spill variants; best244 /878us, slower. More instructions. |
| 3 | Three WGs, wider WGMMA or earlier TMA:0–10 /0–40us | Three-WG spills rejected before execution. N128, store widths, early prefetch/reduction gave no reliable gain. |

Controlled ratio sweep, local tuning (not final headline):66/66 gave
243.024/893.568us;44/88 gave186.032/694.080;42/90 gave175.456/668.736;
40/92 gave174.688/624.928. Rename40/92 to dual_balanced, then independently
validate and repeat. Longer mask periods can lose register-cache benefits;
shared mask-cache variants were also tested but not selected. The32-row
schedule reduced long-scoreboard stalls but raised executed instructions
59.0M→84.3M at L384, explaining its loss.

## 5. Complete code and host integration

- dual_balanced.cu: complete selected device source; existing
  dual_primitives.cuh and Anthropic v5 headers are reused.
- dual_balanced.py: Plan wrapper, count132/PART2 default; PART1 supported.
- dual_experiment.py: source/dependency/flag hashed cubins, zero-spill hard
  gate, workspace/maps and experiment launch support.
- integrate.backward_cuda: B1–B4 replaced in the full backward harness.
- Production dispatch and original sources unchanged. No cutlass-prefixed
  files or toolkit upgrade. Source archive includes local harness files;
  Anthropic headers and MiniWorld engine remain repository dependencies.

## 6. Correctness tests and results

L64/384/768 × p0/.25 × count66/132 × PART1/2 =24 cases. Each normal launch
plus two graph replays with changed dy/ds:72 six-output comparisons, counters
zero throughout. Maximum relativeL2:

| Output | Maximum error | Limit |
| --- | --- | --- |
| dg | 0 | bit-exact incl signed zero |
| dWg | 0.000364898442 | 5e-4 |
| dtri | 1.46946841e-05 | 2e-5 |
| dgamma | 1.97076497e-06 | 5e-6 |
| dbeta | 6.65265588e-07 | 5e-6 |
| dWp | 0.00035537759 | 5e-4 |

Full backward11 gradients pass relativeL2<=5e-4 at both target shapes.
PART2 memcheck/racecheck/synccheck at L64/384 pass. Unfiltered PART1
memcheck also covers standalone reducer, both shapes pass.
Extra L72/80/136 × p.1/.25 × count66/132 × PART1/2:22/24 pass these strict
limits. Two L72 cases repeat the original bit-identical issue in section2.
Do not label this extra matrix all-pass.

## 7. Reproduction commands and measured results

All GPU commands run in an assigned node02 H100 allocation, CUDA12.9/sm90a.
Verification GPU was released; existing training job13228 remained running.
No node01 use. Exact compiler command for selected count132/PART2:

```bash
nvcc -std=c++17 -O3 -arch=sm_90a --cubin -lineinfo -Xptxas=-v -I/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine/third_party/anthropic/upstream/common/opt_core/opt_core/kernels/trimul/native/pkg/v5/csrc -DROLE=0 -DUCOUNT=132 -DPART_ONLY=2 /home/psk6950/MiniWorld/runs/anthropic_b1b4_pipeline_20260919/dual_balanced.cu -o /home/psk6950/MiniWorld/runs/anthropic_b1b4_pipeline_20260919/build/e2c4d8f5995e375cd941108a879265200487c62ed5a0ec0f63ea7eb608b98018.cubin
```

Other validated variants change UCOUNT66/132 and PART_ONLY1/2 only.
The complete timed/sanitizer/NCU sequence is balanced_validation.sh:

```bash
P=runs/anthropic_b1b4_pipeline_20260919
E=runs/anthropic_adoption_20260919/env.sh
bash "$P/balanced_validation.sh"
bash "$E" python -u -B "$P/measure_balanced_final.py"
bash "$E" python -u -B "$P/check_balanced_extra.py"
bash "$E" python -u -B "$P/measure_experiment.py" --sources dual_balanced dual_balanced_timing --parts 2 --output balanced-phase-results.json
```

Sanitizer/profiler commands used for length64/384 (sanitizer) and384/768 (NCU):

```bash
for tool in memcheck racecheck synccheck; do
  for length in 64 384; do
    bash "$E" compute-sanitizer --tool "$tool" --error-exitcode 3 --kernel-name kns=dual_b1b4 python -u -B "$P/sanitize_experiment.py" --source dual_balanced --length "$length" --count 132 --part 2
  done
done
for length in 64 384; do
  bash "$E" compute-sanitizer --tool memcheck --error-exitcode 3 python -u -B "$P/sanitize_experiment.py" --source dual_balanced --length "$length" --count 132 --part 1
done
for length in 384 768; do
  bash "$E" ncu --set full --import-source yes -k regex:dual_b1b4 --profile-from-start off --force-overwrite -o "$P/balanced-L$length" python -u -B "$P/profile_experiment.py" --source dual_balanced --length "$length"
done
```

Primary protocol: same-process CUDA graphs, separate core/full timing
domains,3 blocks each20 warmups+200 events, alternating order. Report pooled
600-sample median/p90, not fastest block. Dropout0.25; no attached profiler.

| L | Triton/cuBLAS median/p90 us | Previous CUDA median/p90 us | Balanced median/p90 us | Speedup | Time saved vs previous |
| --- | --- | --- | --- | --- | --- |
| 384 | 299.52 / 301.98 | 211.17 / 212.51 | 176.00 / 178.88 | 1.7018x | 16.7% |
| 768 | 1182.08 / 1186.14 | 742.16 / 744.22 | 620.48 / 633.28 | 1.9051x | 16.4% |

Full backward, B5+ unchanged:

| L | Baseline median/p90 us | Balanced median/p90 us | Speedup | Time reduction |
| --- | --- | --- | --- | --- |
| 384 | 1084.77 / 1086.69 | 967.15 / 970.94 | 1.1216x | 10.8% |
| 768 | 4342.58 / 4369.06 | 3799.81 / 3845.38 | 1.1428x | 12.5% |

Every primary core block:

| L | Block | Baseline median us | Balanced median us | Speedup |
| --- | --- | --- | --- | --- |
| 384 | 1 | 299.520 | 175.904 | 1.702747x |
| 384 | 2 | 299.824 | 176.288 | 1.700762x |
| 384 | 3 | 299.424 | 175.968 | 1.701582x |
| 768 | 1 | 1182.256 | 621.072 | 1.903573x |
| 768 | 2 | 1181.968 | 620.016 | 1.906351x |
| 768 | 3 | 1182.032 | 620.288 | 1.905618x |

Earlier independent20/200 validation mixed core/full timing domains:
L384298.240→175.008us (1.704x), L7681189.136→645.728us (1.842x).
PART1 gave175.024/646.064us, similar: keep simple split-reducer fallback.
Those results remain in balanced-final-results.json. Separate-domain
repeated results above are primary; warm cache state and system/clock
variation affect timing. No locked-clock claim.

## 8. Profiler signals: observed versus expected

NCU full-replay durations differ from warm event medians:

| L | Version | NCU us | DRAM % | HBM read MB | HBM write MB | L2 MB derived | Tensor active % | Bank conflicts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 384 | previous | 232.42 | 54.49% | 304.63 | 119.88 | 711.50 | 13.11% | 2.325M |
| 384 | balanced | 213.18 | 60.51% | 324.67 | 107.75 | 764.01 | 14.67% | 0.689M |
| 768 | previous | 827.33 | 60.38% | 1214.95 | 459.52 | 2595.89 | 14.55% | 9.366M |
| 768 | balanced | 749.95 | 70.62% | 1325.96 | 449.44 | 2974.30 | 16.70% | 3.369M |

DRAM>70% goal: L38460.51% misses, L76870.62% reaches the profiler threshold.
This does not prove roofline saturation. Instruction work, shared stores,
barriers and cache traffic remain.252 registers and local load/store
sectors0 on both shapes agree with ptxas. L2 bytes=sectors*32; direct
lts__t_bytes.sum unavailable. Independently replayed aggregate/submetric
bank counters need not add exactly; avoid inferring exact overlap from them.

Stall metrics are per-issue-active ratios, not elapsed-time percentages:

| L | Version | Barrier | Long scoreboard | Short scoreboard | Membar | Wait | Sleeping |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 384 | previous | 1.1505 | 0.9994 | 0.9915 | 0.0354 | 0.9891 | 0.0033 |
| 384 | balanced | 1.5313 | 0.4674 | 0.4250 | 0.0303 | 0.7857 | 0.0046 |
| 768 | previous | 0.9363 | 0.8632 | 1.0090 | 0.0111 | 0.9813 | 0.0013 |
| 768 | balanced | 1.2710 | 0.3489 | 0.4211 | 0.0077 | 0.7673 | 0.0022 |

Long-scoreboard and wait ratios improve. Exposed spin-barrier tail<1us;
removing only it cannot yield another large gain. Shared-store conflicts
and CTA barriers remain. Static module SASS, including standalone reducer:

| Instruction | Static count |
| --- | --- |
| HGMMA | 28 |
| UTMALDG | 59 |
| UTMASTG | 4 |
| LDG | 275 |
| STG | 106 |
| LDS | 60 |
| STS | 47 |
| LDL | 0 |
| STL | 0 |

Explicit TMA/WGMMA and zero local spills confirmed. Static counts are not
dynamic traffic. Selected executed instructions59.46M/232.10M; rejected
32-row schedule84.34M at L384.

## 9. Gap, risks and the single next experiment

Relative1.7x passes all3 target-shape core blocks. L384 pooled1.7018x has
only0.19us margin against299.52/1.7=176.19us; this is not a robust guarantee
across machines/runs. Absolute173us is still missed by3.00us. L768620.48us
beats its693us target. Full backward1.122x/1.143x because B5+ is unchanged;
no end-to-end training-step speedup has been established.

Limits: fixed C128/H256, single-stream non-reentrant workspace, no production
dispatch/cache promotion, extra-L72 original numerical caveat, shape-dependent
mask periods/tails, cache/clock variability. Prior sources retained.

**Single next experiment:** change only DX warp-partial shared-memory layout
to reduce store-bank conflicts. Keep40/92 roles, fragment arithmetic and TMA
maps fixed. Conditional estimate2–5us at L384 /5–15us at L768, not a promise.
Require lower NCU bank/store stalls, zero spills, strict checks and repeated
same-process gains. Proposed only; not implemented or claimed faster.
