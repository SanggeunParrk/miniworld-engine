"""Checks kind::f8f6f4 e4m3 operand layouts against torch (fp8 values decoded exactly, fp32 matmul)."""
import torch, drv
k = drv.Kernel("build/f8_test.cubin", "f8_test", 32768 + 1024)
def swz(t):   # t: [128 rows][128 bytes] uint8 -> SW128 image (16-byte chunk q of row r at chunk q ^ (r & 7))
    t = t.view(128, 8, 16)
    idx = torch.arange(8, device=t.device)[None, :] ^ (torch.arange(128, device=t.device)[:, None] & 7)
    o = torch.empty_like(t); o[torch.arange(128)[:, None], idx] = t
    return o.reshape(-1).contiguous()
torch.manual_seed(0)
A = (torch.randn(128, 128, device="cuda") * 2).to(torch.float8_e4m3fn)   # A [m][k]
B = (torch.randn(128, 128, device="cuda") * 2).to(torch.float8_e4m3fn)   # B [n][k]
ref = A.float() @ B.float().t()
u8 = lambda x: x.view(torch.uint8)
imgs = {0: (swz(u8(A)), swz(u8(B))), 1: (swz(u8(A.t().contiguous())), swz(u8(B.t().contiguous()))),
        2: (swz(u8(A)), swz(u8(B))), 3: (swz(u8(A)), swz(u8(B.t().contiguous())))}
def swz64(t):  # t: [rows][64 bytes] -> SW64 image (chunk q of row r at q ^ ((r >> 1) & 3))
    R = t.shape[0]; t = t.reshape(R, 4, 16)
    idx = torch.arange(4, device=t.device)[None, :] ^ ((torch.arange(R, device=t.device)[:, None] >> 1) & 3)
    o = torch.empty_like(t); o[torch.arange(R)[:, None], idx] = t
    return o.reshape(-1).contiguous()
Bh = B[:64]                                           # N = 64
img4b = torch.zeros(16384, dtype=torch.uint8, device="cuda"); img4b[:8192] = swz64(u8(Bh.t().contiguous()))
imgs[4] = (swz(u8(A)), img4b)
for mode, (ia, ib) in imgs.items():
    out = torch.zeros(128, 128, device="cuda")
    k((1, 1, 1), (128, 1, 1), ia, ib, u8(A).contiguous().view(torch.int32), out, mode); torch.cuda.synchronize()
    o, r = (out[:, :64], ref[:, :64]) if mode == 4 else (out, ref)
    print(f"mode {mode}: max |err| {(o - r).abs().max().item():.3e}  (ref max {r.abs().max().item():.1f})")
