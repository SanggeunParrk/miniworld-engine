import os, sys
sys.argv = [sys.argv[0], "none"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t_pwa.py")).read())
m = torch.randn(S, N, D, device="cuda", dtype=bf); y = torch.randn(S, N, D, device="cuda", dtype=bf)
w = torch.softmax(torch.randn(H, N, N, device="cuda") * 2, -1).to(bf); v = torch.randn(H, N, S * C, device="cuda", dtype=bf)
wg = (torch.randn(H * C, D, device="cuda") * 0.2).to(bf); wo = (torch.randn(D, H * C, device="cuda") * 0.05).to(bf)
for _ in range(3): ext.pwa_fwd(w, v, y, wg, wo, m, False, None, 1.0)
torch.cuda.synchronize()
