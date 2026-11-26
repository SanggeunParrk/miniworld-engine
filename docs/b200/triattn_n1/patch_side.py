import sys
p = sys.argv[1] + "/src/foldjet_engine/kernels/integrations/bidir_triattn_h100.py"; s = open(p).read()
# flag + helper
anchor = "def _ln_fwd(g, x, lnw, lnb, eps, L):"
flag = '''#: (N1) the single-direction training backward's two weight-gradient GEMMs (dWo = dt^T u, dW = dc^T y: tall-K split-K GEMMs of 9-12 us at L128, 36 us at L512
#: with their reduce) off the main chain as branches of the captured graph (they read tensors the chain already has and nothing reads their results before the
#: end): the dWo GEMM beside the gate backward / core GEMMs, the dW GEMM beside the dyln GEMM and the LayerNorm backward. One sample (B = 1), B200, graph capture only.
#: FOLDJET_ENGINE_BIDIR_B200_SIDE=0 turns it off.
B200_SIDE = os.environ.get("FOLDJET_ENGINE_BIDIR_B200_SIDE", "1") != "0"


'''
assert anchor in s
s = s.replace(anchor, flag + anchor, 1)
# _backward1 edits
i0 = s.index("def _backward1(")
i1 = s.index("def _weights1(")
body = s[i0:i1]
old1 = "        dWo = _acc(dWo, _mm_f32(dt.t(), u.t()))\n"
new1 = '''        s_wo = None
        if B200_SIDE and B == 1 and dWo is None and _b200():
            from foldjet_engine.kernels.bias_only_dit.cuda import side

            s_wo = side.fork(x.device)
        if s_wo is not None:  # branch: dWo beside the gate backward and the core GEMMs
            with side.on(s_wo):
                dWo = _mm_f32(dt.t(), u.t())
        else:
            dWo = _acc(dWo, _mm_f32(dt.t(), u.t()))
'''
assert body.count(old1) == 1
body = body.replace(old1, new1)
old2 = "        dW = _acc(dW, _mm_f32(dc, y[rows]))\n"
new2 = '''        s_w = None
        if B200_SIDE and B == 1 and dW is None and _b200():
            from foldjet_engine.kernels.bias_only_dit.cuda import side

            s_w = side.fork(x.device)
        if s_w is not None:  # branch: dW beside the dyln GEMM and the LayerNorm backward
            with side.on(s_w):
                dW = _mm_f32(dc, y[rows])
        else:
            dW = _acc(dW, _mm_f32(dc, y[rows]))
'''
assert body.count(old2) == 1
body = body.replace(old2, new2)
old3 = "    return dx.view(B, L, L, C), dgb[0], dgb[1], dW, dWo\n"
new3 = '''    if B == 1 and s_wo is not None:
        from foldjet_engine.kernels.bias_only_dit.cuda import side

        side.join(x.device, s_wo)
        side.join(x.device, s_w)
    return dx.view(B, L, L, C), dgb[0], dgb[1], dW, dWo
'''
assert body.count(old3) == 1
body = body.replace(old3, new3)
s = s[:i0] + body + s[i1:]
open(p, "w").write(s); print("ok")
