"""trimul_inproj — fused input projections for the triangle multiplicative update.

One fusion unit that produces all three input projections from the normalized
pair representation in a single read of x:

    left  = sigmoid(x @ WLg) * (x @ WL)     -> [B, D, L, L]   (for the bmm)
    right = sigmoid(x @ WRg) * (x @ WR)     -> [B, D, L, L]   (for the bmm)
    gate  = sigmoid(x @ Wg)                 -> [B, L, L, D]   (final elementwise mul)

This is distinct from ``tm1`` (left+right only) and ``tm2`` (the output
gate+projection+mul). Pulling ``gate`` to the front lets the back half fold the
final mul into the layernorm-linear epilogue — see ``README.md``.
"""
