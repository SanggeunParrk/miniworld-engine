import sys
root = sys.argv[1]
p = root + "/src/foldjet_engine/kernels/integrations/bidir_triattn_h100.py"; s = open(p).read()
old = "    y, mean, rstd = g.ln_fwd(x, lnw.float().contiguous(), lnb.float().contiguous(), eps, L, False)"
assert s.count(old) == 2, s.count(old)
s = s.replace(old, "    y, mean, rstd = _ln_fwd(g, x, lnw, lnb, eps, L)")
anchor = "def _ln_bwd(g, dyln, x, dy, lnw, mean, rstd, L):"
helper = '''def _ln_fwd(g, x, lnw, lnb, eps, L):
    """-> (y, mean, rstd) of the LayerNorm forward in the starting layout (the shared glue's ``ln_fwd``; C = 128 on the B200 kernel)."""
    if B200_LNBWD and _b200() and x.shape[1] == 128:
        from foldjet_engine.kernels.triangle_attention.cuda import b200_bidir

        return b200_bidir.lnb().ln_fwd128(x, lnw.float().contiguous(), lnb.float().contiguous(), eps, 4)
    return g.ln_fwd(x, lnw.float().contiguous(), lnb.float().contiguous(), eps, L, False)


'''
s = s.replace(anchor, helper + anchor, 1)
s = s.replace("#: (N1, cuda:b200_sm100 71 / Bidirectional 5) the bias-only paths' LayerNorm backward (C = 128) on the B200 kernel of",
              "#: (N1, cuda:b200_sm100 71 / Bidirectional 5) the bias-only paths' LayerNorm forward and backward (C = 128) on the B200 kernels of")
s = s.replace("dx bitwise equal, d gamma / d beta fp32 sums regrouped. FOLDJET_ENGINE_BIDIR_B200_LNBWD=0 turns it off.", "forward y / mean / rstd and dx bitwise equal, d gamma / d beta fp32 sums regrouped. FOLDJET_ENGINE_BIDIR_B200_LNBWD=0 turns it off.")
open(p, "w").write(s)
print("ok")
