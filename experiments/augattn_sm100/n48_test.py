import torch, drv
k = drv.Kernel("build/n48_test.cubin", "n48_test", 32768 + 1024)
def swz(t):   # t: [rows][64] bf16 -> SW128 image
    R = t.shape[0]; u = t.view(torch.int16).view(R, 8, 8)
    idx = torch.arange(8, device=t.device)[None, :] ^ (torch.arange(R, device=t.device)[:, None] & 7)
    o = torch.empty_like(u); o[torch.arange(R)[:, None], idx] = u
    return o.reshape(-1).view(torch.int32).contiguous()
torch.manual_seed(0)
for kdim in (48, 64):
    A = torch.randn(128, kdim, device="cuda").bfloat16(); B = torch.randn(kdim, 48, device="cuda").bfloat16()   # D = A B
    ref = A.float() @ B.float()
    Ap = torch.zeros(128, 64, device="cuda", dtype=torch.bfloat16); Ap[:, :kdim] = A
    # mode 0: B MN-major: rows = K, 64 columns (48 used)
    Bm = torch.full((64, 64), float("nan"), device="cuda", dtype=torch.bfloat16); Bm[:kdim, :48] = B; Bm[kdim:, :] = 0
    Bm[:, 48:] = float("nan")
    img_b0 = torch.zeros(4096, dtype=torch.int32, device="cuda"); img_b0[:2048] = swz(Bm)
    # mode 1: B K-major: rows = N (48), K columns
    Bk = torch.zeros(128, 64, device="cuda", dtype=torch.bfloat16); Bk[:48, :kdim] = B.t()
    for mode, ib in ((0, img_b0), (1, swz(Bk))):
        out = torch.zeros(128, 48, device="cuda")
        k((1, 1, 1), (128, 1, 1), swz(Ap), ib, out, mode, kdim); torch.cuda.synchronize()
        print(f"K={kdim} mode {mode}: max |err| {(out - ref).abs().max().item():.3e}  nan {torch.isnan(out).any().item()}")
