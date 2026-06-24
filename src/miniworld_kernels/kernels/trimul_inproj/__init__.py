"""trimul_inproj — fused input projections for the triangle multiplicative update.

One fusion unit that produces all three input projections from the normalized
pair representation in a single read of x:

    left  = sigmoid(x @ WLg) * (x @ WL)     -> [B, D, L, L]   (for the bmm)
    right = sigmoid(x @ WRg) * (x @ WR)     -> [B, D, L, L]   (for the bmm)
    gate  = sigmoid(x @ Wg)                 -> [B, L, L, D]   (final elementwise mul)

This is distinct from ``tm1`` (left+right only) and ``tm2`` (the output
gate+projection+mul). Pulling ``gate`` to the front lets the back half fold the
final mul into the layernorm-linear epilogue — see ``README.md``.

Three execution paths (import directly to avoid forcing the quack import):

  - ``cute.inference.trimul_inproj_inference`` — forward-only, saves NOTHING,
    maximally fused (single fused back kernel). Lowest latency; NOT for training.
  - ``cute.training.TriMulInproj`` — trainable module. ``.forward`` is the
    training path (autograd.Function: forward SAVES the tensors its backward
    needs; what it saves is co-designed with the backward) and ``.inference``
    dispatches to the forward-only path above.

The split matters because *what the forward saves* changes the backward's
algorithm and speed (save-vs-recompute); the inference forward, saving nothing,
is free to fuse/discard everything the training forward cannot.
"""
