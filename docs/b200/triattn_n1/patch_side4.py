import sys
p = sys.argv[1] + "/src/miniworld_engine/kernels/integrations/bidir_triattn_h100.py"; s = open(p).read()
i0 = s.index("def _backward1(")
i1 = s.index("def _weights1(")
body = s[i0:i1]
old = '''        if starting:  # O[h] = V[h] P[h]^T
            dos = do.view(H, dh * L, L)
            torch.bmm(dos, p, out=dc[:Dh].view(H, dh * L, L))
            dp = torch.bmm(dos.transpose(1, 2), c[:Dh].view(H, dh * L, L), out_dtype=torch.float32)  # [H, j, k]
        else:  # O[c] = P[h] V[c]
            dp = torch.empty(H, L, L, device=x.device, dtype=torch.float32)
            if bo is not None and L <= BO_GEMM_MAX_L:  # B200: every head in one launch each
                bo.bo_pv(p, do, dh, True, dc[:Dh])
                bo.bo_dp(do, c[:Dh], dh, dp)
'''
new = '''        s_dv = None
        if (B200_SIDE & 4) and B == 1 and L <= B200_SIDE_MAX_L and (bo is not None or starting) and _b200():
            from miniworld_engine.kernels.bias_only_dit.cuda import side

            s_dv = side.fork(x.device)   # branch: the dV contraction beside the dP contraction, the row dot and the softmax backward
        if starting:  # O[h] = V[h] P[h]^T
            dos = do.view(H, dh * L, L)
            if s_dv is not None:
                with side.on(s_dv):
                    torch.bmm(dos, p, out=dc[:Dh].view(H, dh * L, L))
            else:
                torch.bmm(dos, p, out=dc[:Dh].view(H, dh * L, L))
            dp = torch.bmm(dos.transpose(1, 2), c[:Dh].view(H, dh * L, L), out_dtype=torch.float32)  # [H, j, k]
        else:  # O[c] = P[h] V[c]
            dp = torch.empty(H, L, L, device=x.device, dtype=torch.float32)
            if bo is not None and L <= BO_GEMM_MAX_L:  # B200: every head in one launch each
                if s_dv is not None:
                    with side.on(s_dv):
                        bo.bo_pv(p, do, dh, True, dc[:Dh])
                else:
                    bo.bo_pv(p, do, dh, True, dc[:Dh])
                bo.bo_dp(do, c[:Dh], dh, dp)
'''
assert body.count(old) == 1
body = body.replace(old, new)
# join before the dW GEMM
old2 = "        s_w = None\n        if (B200_SIDE & 2)"
new2 = "        if s_dv is not None:\n            side.join(x.device, s_dv)   # dc[:Dh] (dV) is read by the dW and dyln GEMMs\n        s_w = None\n        if (B200_SIDE & 2)"
assert body.count(old2) == 1
body = body.replace(old2, new2)
s = s[:i0] + body + s[i1:]
s = s.replace('int(os.environ.get("MINIWORLD_BIDIR_B200_SIDE", "3"))   # bit 0: the dWo branch, bit 1: the dW branch', 'int(os.environ.get("MINIWORLD_BIDIR_B200_SIDE", "3"))   # bit 0: the dWo branch, bit 1: the dW branch, bit 2: the dV branch')
open(p, "w").write(s); print("ok")
