import sys
p = sys.argv[1] + "/src/foldjet_engine/kernels/integrations/bidir_triattn_h100.py"; s = open(p).read()
old = '''def _ln_bwd(g, dyln, x, dy, lnw, mean, rstd, L):
    """-> (dx, partials [n, 2, C]) of the LayerNorm backward + the residual's dy (the shared glue's ``ln_bwd`` in the starting layout)."""
    if B200_LNBWD and _b200() and x.shape[1] == 128:
        from foldjet_engine.kernels.triangle_attention.cuda import b200_bidir

        return b200_bidir.lnb().ln_bwd128(dyln, x, dy, lnw.float().contiguous(), mean, rstd, L, False, 2)
    return g.ln_bwd(dyln, x, dy, lnw.float().contiguous(), mean, rstd, L, False)
'''
new = '''def _ln_bwd(g, bo, dyln, x, dy, lnw, mean, rstd, L):
    """-> (dx, [d gamma; d beta] fp32 [2, C]) of the LayerNorm backward + the residual's dy (the shared glue's ``ln_bwd`` in the starting layout,
    its partial rows summed by ``bo_colsum``; C = 128 on the B200 kernel, whose last CTA sums them in the same launch)."""
    if B200_LNBWD and _b200() and x.shape[1] == 128:
        from foldjet_engine.kernels.triangle_attention.cuda import b200_bidir

        return b200_bidir.lnb().ln_bwd128(dyln, x, dy, lnw.float().contiguous(), mean, rstd, L, False, 2)
    dx, part = g.ln_bwd(dyln, x, dy, lnw.float().contiguous(), mean, rstd, L, False)
    return dx, (bo.bo_colsum(part) if bo is not None else part.sum(0))  # B200: one deterministic launch (torch: a slow tall reduce)
'''
assert old in s
s = s.replace(old, new)
old2 = "    dx, part = _ln_bwd(g, dyln, x, dy, lnw, mean, rstd, L)  # + the residual's dy\n    dgb = bo.bo_colsum(part) if bo is not None else part.sum(0)  # B200: one deterministic launch (torch: a slow tall reduce)\n"
assert s.count(old2) == 2
s = s.replace(old2, "    dx, dgb = _ln_bwd(g, bo, dyln, x, dy, lnw, mean, rstd, L)  # + the residual's dy\n")
open(p, "w").write(s); print("ok2")
