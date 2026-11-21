import sys
p = sys.argv[1] + "/src/miniworld_engine/kernels/integrations/bidir_triattn_h100.py"; s = open(p).read()
old = '''        return b200_bidir.lnb().ln_bwd128(dyln, x, dy, lnw.float().contiguous(), mean, rstd, L, False, 2)
    dx, part = g.ln_bwd(dyln, x, dy, lnw.float().contiguous(), mean, rstd, L, False)
    return dx, (bo.bo_colsum(part) if bo is not None else part.sum(0))  # B200: one deterministic launch (torch: a slow tall reduce)'''
new = '''        dx, part = b200_bidir.lnb().ln_bwd128(dyln, x, dy, lnw.float().contiguous(), mean, rstd, L, False, 2, False)
    else:
        dx, part = g.ln_bwd(dyln, x, dy, lnw.float().contiguous(), mean, rstd, L, False)
    return dx, (bo.bo_colsum(part) if bo is not None else part.sum(0))  # B200: one deterministic launch (torch: a slow tall reduce)'''
assert old in s
s = s.replace(old, new)
s = s.replace("its partial rows summed by ``bo_colsum``; C = 128 on the B200 kernel, whose last CTA sums them in the same launch).", "its partial rows summed by ``bo_colsum``; C = 128 on the B200 kernel).")
open(p, "w").write(s); print("ok2")
