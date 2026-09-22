# Rejected before activation

Normal full-module checks and deliberate between-CTA delay checks passed, but compute-sanitizer memcheck changed scheduling and revealed incorrect dTri/output-LN gradients despite0 reported address errors. B1_REG_EARLY_RAW lacked a CTA barrier after per-warp-group WGMMA waits; group0 could start the opposite-slot TMA while group1 still read dProj. This candidate is not selected or published. Correction and fresh verification are in `../trimul_b1_epilogue_fixed_20260921`. Original code/results are retained for diagnosis.
