"""Verify fused_h alone (h = silu(x@Wa^T)*b) vs torch TF32."""
import torch
import ct_train_ext as ext

torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"


def cos(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


for tag, (M, K, ND) in {"atom": (2048, 128, 256), "token": (1024, 768, 1536)}.items():
    x = torch.randn(M, K, device=dev)
    Wa = torch.randn(ND, K, device=dev) * 0.05
    Wb = torch.randn(ND, K, device=dev) * 0.05
    b = x @ Wb.t()
    a = x @ Wa.t()
    h_ref = torch.nn.functional.silu(a) * b
    h = ext.fused_h(x, Wa, b)
    print(f"[{tag}] cos(h) = {cos(h, h_ref):.6f}  (M={M} K={K} ND={ND})")
print("FUSED_H DONE")
