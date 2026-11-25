import sys
p = sys.argv[1] + "/src/foldjet_engine/kernels/integrations/bidir_triattn_h100.py"; s = open(p).read()
s = s.replace('B200_SIDE = os.environ.get("MINIWORLD_BIDIR_B200_SIDE", "1") != "0"', 'B200_SIDE = int(os.environ.get("MINIWORLD_BIDIR_B200_SIDE", "3"))   # bit 0: the dWo branch, bit 1: the dW branch\nB200_SIDE_MAX_L = int(os.environ.get("MINIWORLD_BIDIR_B200_SIDE_MAX_L", "256"))   # longer pairs: the chain\'s kernels fill the SMs, the branches only slow it')
s = s.replace("        if B200_SIDE and B == 1 and dWo is None and _b200():", "        if (B200_SIDE & 1) and B == 1 and L <= B200_SIDE_MAX_L and dWo is None and _b200():")
s = s.replace("        if B200_SIDE and B == 1 and dW is None and _b200():", "        if (B200_SIDE & 2) and B == 1 and L <= B200_SIDE_MAX_L and dW is None and _b200():")
s = s.replace("    if B == 1 and s_wo is not None:\n        from foldjet_engine.kernels.bias_only_dit.cuda import side\n\n        side.join(x.device, s_wo)\n        side.join(x.device, s_w)\n",
              "    if B == 1 and (s_wo is not None or s_w is not None):\n        from foldjet_engine.kernels.bias_only_dit.cuda import side\n\n        side.join(x.device, s_wo)\n        side.join(x.device, s_w)\n")
open(p, "w").write(s); print("ok")
