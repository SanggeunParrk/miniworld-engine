import os, sys
sys.argv = [sys.argv[0], "none"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t_opm.py")).read())
S = 1024
mask = torch.rand(S, N, device="cuda") > 0.1
bits = torch.zeros(N, S // 32, dtype=torch.int64, device="cuda")
for w in range(32): bits |= (mask.t().reshape(N, S // 32, 32)[..., w].long() << w)
bits = (bits - ((bits >> 31) & 1) * (1 << 32)).to(torch.int32)
dz = torch.randn(N, N, CZ, device="cuda", dtype=bf); wo = (torch.randn(CZ, CH * CH, device="cuda") * 0.03).to(bf)
for _ in range(3): ext.opm_dgrad(dz, bits, wo, N, N, 0)
torch.cuda.synchronize()
