"""Verify fused_y (y=sigmoid(cond@Wsc^T+b_sc)*out, scale stored) vs torch TF32."""
import torch
import ct_train_ext as ext
torch.manual_seed(0); torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"


def cos(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return (a @ b / (a.norm()*b.norm()+1e-20)).item()


for tag, (M, dc, d) in {"atom": (2048, 384, 128), "token": (1024, 384, 768)}.items():
    cond = torch.randn(M, dc, device=dev)
    Wsc = torch.randn(d, dc, device=dev) * 0.05
    b_sc = torch.randn(d, device=dev)
    out = torch.randn(M, d, device=dev)
    scale_ref = cond @ Wsc.t() + b_sc
    y_ref = torch.sigmoid(scale_ref) * out
    y, scale = ext.fused_y(cond, Wsc, b_sc, out)
    print(f"[{tag}] cos(y)={cos(y,y_ref):.6f} cos(scale)={cos(scale,scale_ref):.6f} (M={M} d={d})")
print("FUSED_Y DONE")
