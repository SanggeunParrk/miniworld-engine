import torch
from common import H, D, make, graph_time
from attn_op import prep_do
A = 48
for L in (384, 768):
    q, k, v, bias = make(A, L)
    do = torch.randn(A, 1, L, H, D, device="cuda"); O = torch.randn(A * L, H * D, device="cuda")
    bt = torch.empty_like(bias); DQ = torch.empty(A * L, H * D, device="cuda")
    print(f"L{L}: prep_do {graph_time(lambda: prep_do(do, O, A, L)):6.1f} us   bias transpose {graph_time(lambda: bt.copy_(bias.transpose(1, 2))):6.1f} us"
          f"   dQ zero {graph_time(lambda: DQ.zero_()):6.1f} us   (dO+O read {2 * do.numel() * 4 / 1e6:.0f} MB)", flush=True)
