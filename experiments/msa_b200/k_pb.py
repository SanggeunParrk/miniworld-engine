import os, sys
sys.argv = [sys.argv[0], "none"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t_opm.py")).read())
S = 1024; CM = 64
m = torch.randn(S, N, CM, device="cuda", dtype=bf); mask = torch.rand(S, N, device="cuda") > 0.1
lnw = 1 + 0.1 * torch.randn(CM, device="cuda"); lnb = 0.1 * torch.randn(CM, device="cuda")
wa = (torch.randn(CH, CM, device="cuda") * 0.1).to(bf); wb = (torch.randn(CH, CM, device="cuda") * 0.1).to(bf)
_, _, _, stats, _ = ext.opm_prologue(m, mask, lnw, lnb, 1e-5, wa, wb, True, False)
dA = torch.randn(S, N * CH, device="cuda", dtype=bf); dB = torch.randn(S, N * CH, device="cuda", dtype=bf)
for _ in range(3): ext.opm_prologue_bwd(dA, dB, m, stats, mask, lnw, lnb, wa, wb)
torch.cuda.synchronize()
