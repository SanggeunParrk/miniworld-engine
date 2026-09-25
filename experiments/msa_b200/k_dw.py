import os, sys
sys.argv = [sys.argv[0], "none"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t_opm.py")).read())
dzp = (torch.randn(N, N, CZ, device="cuda") * 0.01).to(bf); O = torch.randn(N * CH, N * CH, device="cuda", dtype=bf)
for _ in range(3): ext.opm_dwo(dzp, O, N, N, 0)
torch.cuda.synchronize()
