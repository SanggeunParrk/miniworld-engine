# transition_b2b — v0 baseline (before CUTLASS-collective campaign)

- GPU: NVIDIA H100 80GB HBM3; CUDA sm_90a; bf16 in/out, fp32 accum.
- Shape: M=524288, K=128, ND=512, D=128 (AF3 transition d=128, n=4).
- Reference: Triton `_transition_b2b_kernel` = 543 us (cos 1.0). AOT of Triton's cubin = 1.008x (ceiling proof).
- Baseline kernel (this commit): hand CuTe fused b2b, 2-consumer-warpgroup cooperative + TMA weight ring,
  h kept in registers (RS wgmma), LN prologue folded. **658 us = 0.825x of Triton, cos 1.0.**
- ncu (M=262144): SM 31%, occ 12.5%, stalls: wait 1.36, lg_throttle 1.40, long_scoreboard 0.79,
  barrier 0.53, gmma 0.26. Compute roofline ~200 us (2.7x headroom) — Triton is the FLOOR.
- Journey to here (via codex): cuBLASDx 0.11x -> hand-CuTe fused 0.34x -> +LN-prologue fix 0.60x ->
  +TMA 2-CTA 0.76x -> +barrier-cut 0.77x -> +wgmma-overlap 0.82x -> +2WG-coop 0.825x.
  Hand-rolled ping-pong (3WG, NamedBarrier) both REGRESSED (sync overhead / occupancy).
- Next campaign: use CUTLASS's REAL collective warp-specialized scheduler (not hand-rolled) for the
  fused b2b, to break the coupled-stall plateau (wait<->lg<->gmma) and beat Triton.
