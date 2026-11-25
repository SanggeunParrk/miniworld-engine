import sys
root = sys.argv[1]
p = root + "/src/foldjet_engine/kernels/integrations/bidir_triattn_h100.py"; s = open(p).read()
old = """    if FUSED_GBWD and wo.shape[0] == 128 and K in (128, 256) and dy.shape[0] % 64 == 0 and _planes_ext("gbwd") is not None:
        return _ext().gate_bwd_fused(dy.contiguous(), wo, g, o, dg)"""
new = """    if FUSED_GBWD and wo.shape[0] == 128 and K in (128, 256) and dy.shape[0] % 64 == 0 and _planes_ext("gbwd") is not None:
        if B200_LNBWD and _b200() and K == 128:  # (N1) the same kernel with the gate / o loads of a tile issued before its GEMM: bitwise equal
            from foldjet_engine.kernels.triangle_attention.cuda import b200_bidir

            return b200_bidir.lnb().gate_bwd128(dy.contiguous(), wo, g, o, dg, 2)
        return _ext().gate_bwd_fused(dy.contiguous(), wo, g, o, dg)"""
assert old in s
s = s.replace(old, new)
s = s.replace("the bias-only paths' LayerNorm forward and backward (C = 128) on the B200 kernels of", "the bias-only paths' LayerNorm forward / backward and fused gate backward (C = 128, K = 128) on the B200 kernels of")
open(p, "w").write(s); print("ok")
