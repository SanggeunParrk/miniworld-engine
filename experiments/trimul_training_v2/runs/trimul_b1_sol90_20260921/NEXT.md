# Continuing the active SoL90 goal

Current validated development entry: `../trimul_training_current.py` -> `../trimul_b1_sol90_selected_20260921/sol_policy.py`.
Publication v49 succeeded; receipt in selected directory. Owner-private audience preserved. A source push was initially rejected for authorization ambiguity; same push was approved after verifying the user's standing HTML publication instruction, same existing artifact categories, and owner-only access. No pending publication approval.

Node01 now explicitly authorized. Node02 full with own and other-user training; leave those jobs alone. Latest own GPU jobs (13569 full,13571 entry) completed; allocate free node01 GPUs through Slurm anew. No subagents authorized.

Selected: count132, part2, GATE_PHASE1, LOWREG1, STREAM_LN1, XHAT_FP32=-1, DIRECT_DP1, NO_REDUNDANT_SYNC1, PAIR_WP1, EARLY_RAW1, PREFETCH_DY0, SPLIT_DN1, AFFINE_BOTH1, DTRI_STORE_C64, DN_PAIR1, DN_REG1, affine unroll4 (verify selected JSON). No spills; 254/255 registers. bf16 tri + fp32 mean/rstd, saved input xn remains; no saved output activation.

Matched node01 baseline was previous `trimul_b1_tri_opt_20260921`:
L384 B1 221.664 ->204.992us, full1221.824->1204.432us.
L768 B1 748.416->695.520us, full4957.952->4907.584us.
11 gradients bit-exact including mutation/gamma0, memcheckbothlengths/racecheck384/entry pass.
NCU fixed measured traffic roofline efficiency56.0%/60.6%; optimistic unique-payload simplified model38.4%/45.6%. SoL90 NOT achieved. Do not mark active goal complete. Existing B7 independent-reference error remains.

Stage instrumentation on the earlier split-DN candidate showed output-LN derivative+dNorm+dTri store ~43% of CTA total cycles; prepareLN/gate/proj ~26%; gate phase~15%; others wait/reduction. Current stage profile should be refreshed only if needed since registerDN/affineboth changed instruction mix.

Follow-up ideas:
- Current bottleneck isn't DRAM saturation; dNorm/pointwise/shared/synchronization serial stages remain. Use NCU source/SASS sampling to focus changes.
- RegDN now at255 registers: holding more activations risks spills. Avoid assuming more fusion is automatically faster.
- Separate dWgate phase rereads xn+dGate (512*L² bytes); this prevents claiming algorithmic90 based on fixed-scheduletraffic utilization. Eliminating that phase requires reducing persistent dWp accumulator register burden or a different work distribution. Prior GATE_PHASE0 caused pressure, but current scheduling may merit a carefully isolated retest.
- One-pass GATE_PHASE0 requires care: x_n used for dWgate before split-DN buffer reuse is safe; output dGate TMA store synchronization already exists. GATE_PHASE0 does not execute gate phase barrier; kernel still finalreduces. Tune flags and inspect spills. Do not mutate selected snapshot; derive new directory.
- Larger dNorm WGMMA N128 could reduce instructions further but B operand layout differs across N64 chunks; validate descriptor/layout rather than substituting blindly.
- CTA128/120/96 already tested and slower; do not repeat without new implementation evidence. Count128 lengthens cached dropout mask period and uses254 registers vs243 in prior no-regDN variant.

Scripts `derive*.py` are reproducible history. Initial dnreg compile error fixed via constexpr lambda index; failed evidence preserved. Goal text still saysnode02 but latest user permission supersedes that constraint.

Follow-up completed: `../trimul_b1_onepass_20260921`, jobs13575(initialguardrejection)/13577(validtuning).12 configs/length, all6outputs bit-exact, all slower. Plain one-pass best+3.1%/+5.4%; explicit64KiBshared Wgaccparkingbest+6.5%/+7.8%. This route is rejected for the current implementation; no dispatch change. More detailed architecture work is needed to reduce persistent192 accumulator registers/thread if eliminatinggatephase. Potential three-warp-group distribution must account for totalSM65536 registers and per-role128 persistent accumulators; simply adding a third role cannot fit current peaktemporaryregister demand. Current v49 remains valid. No own GPU jobs should remain after13577; confirm before new allocation.
