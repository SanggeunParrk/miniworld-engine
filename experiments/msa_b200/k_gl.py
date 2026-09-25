import os, sys
sys.argv = [sys.argv[0], "none"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t_pwa.py")).read())
o = torch.randn(S, N, H * C, device="cuda", dtype=bf); y = torch.randn(S, N, D, device="cuda", dtype=bf); dres = torch.randn(S, N, D, device="cuda", dtype=bf)
wg = (torch.randn(H * C, D, device="cuda") * 0.2).to(bf); wot = (torch.randn(D, H * C, device="cuda") * 0.05).to(bf).t().contiguous()
dgv = torch.zeros(S, N, 2 * H * C, device="cuda", dtype=bf)
for _ in range(3): ext.pwa_glue(o, y, dres, wg, wot, dgv, None, 1.0)
torch.cuda.synchronize()
