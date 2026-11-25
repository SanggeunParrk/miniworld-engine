import sys
p = sys.argv[1] + "/src/foldjet_engine/kernels/integrations/bidir_triattn_h100.py"; s = open(p).read()
# _forward1: optional wait() after the LayerNorm
i0 = s.index("def _forward1(")
i1 = s.index("def _backward1(")
body = s[i0:i1]
hdr_old = "def _forward1(pair, mask, drop, lnw, lnb, eps, w, wo, H, Dh, starting, save):"
assert hdr_old in body
body = body.replace(hdr_old, "def _forward1(pair, mask, drop, lnw, lnb, eps, w, wo, H, Dh, starting, save, wait=None):")
ln_old = "    y, mean, rstd = _ln_fwd(g, x, lnw, lnb, eps, L)\n"
assert body.count(ln_old) == 1
body = body.replace(ln_old, ln_old + "    if wait is not None:  # (N1) the weight cast ran as a branch beside the LayerNorm: joined before the planes GEMM reads it\n        wait()\n")
s = s[:i0] + body + s[i1:]
old = '''        w, wo16 = _weights1(wv, wb, wg, wo)
        out, saved = _forward1(pair, mask, drop, lnw, lnb, eps, w, wo16, H, Dh, starting, True)
'''
new = '''        s_c = None
        if (B200_SIDE & 8) and pair.shape[0] == 1 and pair.shape[1] <= B200_SIDE_MAX_L and _b200():
            from foldjet_engine.kernels.bias_only_dit.cuda import side

            s_c = side.fork(pair.device)
        if s_c is not None:  # branch: the fp32 -> bf16 weight cast beside the LayerNorm forward
            with side.on(s_c):
                w, wo16 = _weights1(wv, wb, wg, wo)
            out, saved = _forward1(pair, mask, drop, lnw, lnb, eps, w, wo16, H, Dh, starting, True, lambda: side.join(pair.device, s_c))
        else:
            w, wo16 = _weights1(wv, wb, wg, wo)
            out, saved = _forward1(pair, mask, drop, lnw, lnb, eps, w, wo16, H, Dh, starting, True)
'''
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace('int(os.environ.get("MINIWORLD_BIDIR_B200_SIDE", "3"))   # bit 0: the dWo branch, bit 1: the dW branch', 'int(os.environ.get("MINIWORLD_BIDIR_B200_SIDE", "3"))   # bit 0: the dWo branch, bit 1: the dW branch, bit 3 (8): the weight cast beside the LayerNorm forward')
open(p, "w").write(s); print("ok")
