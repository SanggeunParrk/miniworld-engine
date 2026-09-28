"""trimul_inproj — fused input projections for the triangle multiplicative update.

One fusion unit that produces all three input projections from the normalized
pair representation in a single read of x:

    left  = sigmoid(x @ WLg) * (x @ WL)     -> [B, D, L, L]   (for the bmm)
    right = sigmoid(x @ WRg) * (x @ WR)     -> [B, D, L, L]   (for the bmm)
    gate  = sigmoid(x @ Wg)                 -> [B, L, L, D]   (final elementwise mul)

This is distinct from ``tm1`` (left+right only) and ``tm2`` (the output
gate+projection+mul). Pulling ``gate`` to the front lets the back half fold the
final mul into the layernorm-linear epilogue — see
``docs/kernels/trimul-inproj.md``.

Execution paths (v2.2.0):

  - ``cuda/`` — the hand-written H100 kernels (``integrations.trimul_h100`` states the
    qualified training / inference contract: widths, lengths, B=1, sm_90).
  - ``triton/`` — the portable fused pipeline (``unidirectional.trimul_triton`` /
    ``bidirectional.bidirectional_trimul_triton``), autograd-capable, every arch; the
    fallback for any shape the CUDA kernels do not serve.
  - ``reference.py`` / ``autograd.py`` — the PyTorch oracle and the manual-backward scaffold.

``whole_op.py`` exposes the cuequivariance-signature facade over the Triton pipeline.
"""
