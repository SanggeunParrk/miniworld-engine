import sys
p = sys.argv[1] + "/src/foldjet_engine/kernels/integrations/bidir_triattn_h100.py"; s = open(p).read()
old = '''        dx, part = b200_bidir.lnb().ln_bwd128(dyln, x, dy, lnw.float().contiguous(), mean, rstd, L, False, 2, False)
    else:'''
new = '''        dx, part = b200_bidir.lnb().ln_bwd128(dyln, x, dy, lnw.float().contiguous(), mean, rstd, L, False, 2, False)
        return dx, b200_bidir.lnb().colsum256(part)  # one launch (the shared colsum: two)
    else:'''
assert old in s
s = s.replace(old, new); open(p, "w").write(s)
p2 = sys.argv[1] + "/src/foldjet_engine/modules/triangle.py"; t = open(p2).read()
t = t.replace("the bias-only paths' LayerNorm backward (C = 128) on a B200 kernel (loads issued together and a row step ahead, 2 CTAs an SM): the single-direction row 188 step -2.2 .. -4 %, dx bitwise equal;", "the bias-only paths' LayerNorm forward / backward and gate backward (C = 128, K = 128) on B200 kernels (loads issued together and a row step ahead, 2-4 CTAs an SM, d gamma / d beta column sum in one launch): the single-direction row 188 step -3.5 .. -5.8 %, dx / dO / dG / y bitwise equal;")
t = t.replace("the LayerNorm backward (C = 128) on a B200 kernel (shared with the single-direction bias-only path of TriangleAttention), dx bitwise equal;", "the LayerNorm forward / backward and gate backward (C = 128) on B200 kernels (shared with the single-direction bias-only path of TriangleAttention), outputs and gradients bitwise equal;")
open(p2, "w").write(t); print("ok2")
