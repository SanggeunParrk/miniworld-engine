import sys
p = sys.argv[1] + "/src/foldjet_engine/kernels/integrations/bidir_triattn_h100.py"; s = open(p).read()
s = s.replace("(B200_SIDE & 2) and B == 1 and L <= B200_SIDE_MAX_L and dW is None and _b200():", "(B200_SIDE & 2) and B == 1 and dW is None and _b200():")
s = s.replace('B200_SIDE_MAX_L = int(os.environ.get("FOLDJET_ENGINE_BIDIR_B200_SIDE_MAX_L", "256"))   # longer pairs: the chain\'s kernels fill the SMs, the branches only slow it',
              'B200_SIDE_MAX_L = int(os.environ.get("FOLDJET_ENGINE_BIDIR_B200_SIDE_MAX_L", "256"))   # the dWo branch only up to this length (beyond it the core GEMMs fill the SMs: it slows the chain); the dW branch at every length')
# the bidirectional backward: same two branches
i0 = s.index("def _backward(dy, pair, lnw, w, wo, saved, H, Dh):")
i1 = s.index("def _weights(wv, wb, wg, wo):")
body = s[i0:i1]
old1 = "        dWo = _acc(dWo, _mm_f32(dy[rows].t(), u.t()))\n"
new1 = '''        s_wo = None
        if (B200_SIDE & 1) and B == 1 and L <= B200_SIDE_MAX_L and dWo is None and _b200():
            from foldjet_engine.kernels.bias_only_dit.cuda import side

            s_wo = side.fork(x.device)
        if s_wo is not None:  # branch: dWo beside the gate backward and the core GEMMs
            with side.on(s_wo):
                dWo = _mm_f32(dy[rows].t(), u.t())
        else:
            dWo = _acc(dWo, _mm_f32(dy[rows].t(), u.t()))
'''
assert body.count(old1) == 1
body = body.replace(old1, new1)
old2 = "        dW = _acc(dW, _mm_f32(dc, y[rows]))\n        _mm_f32_into(dyln[rows], dc.t(), w)\n"
new2 = '''        s_w = None
        if (B200_SIDE & 2) and B == 1 and dW is None and _b200():
            from foldjet_engine.kernels.bias_only_dit.cuda import side

            s_w = side.fork(x.device)
        if s_w is not None:  # branch: dW beside the dyln GEMM and the LayerNorm backward
            with side.on(s_w):
                dW = _mm_f32(dc, y[rows])
        else:
            dW = _acc(dW, _mm_f32(dc, y[rows]))
        _mm_f32_into(dyln[rows], dc.t(), w)
'''
assert body.count(old2) == 1
body = body.replace(old2, new2)
old3 = "    return dx.view(B, L, L, C), dgb[0], dgb[1], dW, dWo\n"
new3 = '''    if B == 1 and (s_wo is not None or s_w is not None):
        from foldjet_engine.kernels.bias_only_dit.cuda import side

        side.join(x.device, s_wo)
        side.join(x.device, s_w)
    return dx.view(B, L, L, C), dgb[0], dgb[1], dW, dWo
'''
assert body.count(old3) == 1
body = body.replace(old3, new3)
s = s[:i0] + body + s[i1:]
open(p, "w").write(s); print("ok")
