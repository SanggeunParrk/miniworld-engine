import torch, pair_bias as pb
from common import graph_time
N = 3072
z = torch.randn(N, N, 16, device="cuda").to(torch.bfloat16); gamma = torch.ones(16, device="cuda"); wb = torch.randn(4, 16, device="cuda")
db = torch.randn(4, N, N, device="cuda")
for t in [(8, 64, 4), (16, 32, 4), (16, 64, 4), (16, 64, 8), (32, 32, 4), (32, 64, 8), (8, 128, 4), (16, 128, 8), (4, 128, 4)]:
    pb.TILE_F = pb.TILE_B = t
    try:
        tf = graph_time(lambda: pb.pair_bias_fwd(z, gamma, wb)); tb = graph_time(lambda: pb.pair_bias_bwd(z, gamma, wb, db))
        print(f"{t}: fwd {tf:7.1f}  bwd {tb:8.1f} us", flush=True)
    except Exception as e:
        print(t, "fail", str(e)[:80])
