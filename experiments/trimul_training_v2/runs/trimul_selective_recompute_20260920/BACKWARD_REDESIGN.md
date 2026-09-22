# Agreed direction: inference forward and fused backward recomputation

User clarification, 2026-09-20: do not checkpoint the entire module, and do
not treat separate activation-restoration kernels as the target design.

## Forward

- Use Anthropic inference K1/K3 pipelines, with the established training
  dropout, residual and BF16 rounding contract preserved.
- Retain only contraction operands `left/right` and contraction result `tri`
  among intermediate activations. These already pass through HBM in inference.
- Retain original input, parameters, pair mask and the same dropout mask/scale.
- Do not add forward HBM saves for input/output normalized activations,
  projection/gate intermediates or LN statistics.

## Backward

| Region | Retained inputs | Recompute inside the backward kernel |
|---|---|---|
| B1-B4 | x, tri, dy, weights, dropout | Input LN for output-gate/dW, output LN/stats, output projection and sigmoid gate; immediately consume for dtri, dWp, dWg, LN parameter gradients and gate-input gradient |
| B5-B6 | left/right and dtri | Keep existing cuBLAS contraction backward; no repeated forward contraction |
| B7-B12 | x, dleft/dright, weights, mask, gate-input gradient | Input LN/stats and PL/GL/PR/GR, immediately consume for GLU derivatives, dx and input weight/LN gradients |

Reconstructed forward activations must remain in registers/shared memory
within their consuming backward tiles. Cross-kernel gradients and reduction
workspace still exist; this does not mean zero HBM traffic in backward.

## Status and acceptance

- This redesign is **not implemented yet**. `adapter_selective.py` is only
  a control that rematerializes activations into HBM with two separate kernels.
- Whole-module checkpoint results from the prior experiment answer a different
  question and must not determine the fused-recomputation design's performance.
- Initial selective-control results pass bit-exact intermediate/output/all-11-
  gradient checks, including changed-input/weight/mask/dropout graph replay.
- Further tuning of that control was stopped after the user's clarification.
  Some attempted configs hit a CUDA 12.9 ptxas signal-11 compiler failure;
  do not advertise an exhaustive tune or sanitizer completion for the control.
- Validate each fused region against fixed-input backward references; then
  validate all gradients in the complete training graph, preserve dropout,
  and compare L384/L768 forward, backward, total and actual HBM traffic.
- The performance target remains the user's earlier >=1.7x / near-SoL goal;
  no current result establishes that target for this new design.
