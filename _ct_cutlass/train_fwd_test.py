"""Verify the two fused-epilogue forward GEMMs (fused_h, fused_y) vs torch TF32 reference."""
import torch
import ct_train_ext as ext

torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = "cuda"


def cos(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def check(tag, M, K, ND, d, dc):
    x = torch.randn(M, K, device=dev)
    Wa = torch.randn(ND, K, device=dev) * 0.05
    Wb = torch.randn(ND, K, device=dev) * 0.05
    cond = torch.randn(M, dc, device=dev)
    Wsc = torch.randn(d, dc, device=dev) * 0.05
    b_sc = torch.randn(d, device=dev)
    out = torch.randn(M, d, device=dev)

    # ref
    a = x @ Wa.t()
    b = x @ Wb.t()
    h_ref = torch.nn.functional.silu(a) * b
    scale_ref = cond @ Wsc.t() + b_sc
    y_ref = torch.sigmoid(scale_ref) * out

    # cutlass fused
    h = ext.fused_h(x, Wa, b)                       # h = silu(x@Wa^T) * b
    y, scale = ext.fused_y(cond, Wsc, b_sc, out)    # y = sigmoid(cond@Wsc^T + b_sc)*out ; scale

    print(f"[{tag}] M={M} K={K} ND={ND} d={d}")
    print(f"  cos(h)     = {cos(h, h_ref):.6f}")
    print(f"  cos(scale) = {cos(scale, scale_ref):.6f}")
    print(f"  cos(y)     = {cos(y, y_ref):.6f}")


check("atom", 2048, 128, 256, 128, 384)
check("token", 1024, 768, 1536, 768, 384)
print("FWD-FUSED DONE")
