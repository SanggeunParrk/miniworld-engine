import os, sys
sys.argv = [sys.argv[0], "none"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t_pwa.py")).read())
w = torch.softmax(torch.randn(H, N, N, device="cuda") * 2, -1).to(bf); dO = torch.randn(H, N, S * C, device="cuda", dtype=bf)
dgv = torch.zeros(S, N, 2 * H * C, device="cuda", dtype=bf)
for _ in range(3): ext.pwa_plain(w, dO, dgv)
torch.cuda.synchronize()
