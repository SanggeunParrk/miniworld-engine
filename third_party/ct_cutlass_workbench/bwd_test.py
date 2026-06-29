"""Verify input-fused da/db (swiglu-bwd folded into dh-GEMM epilogue) vs torch TF32 ref."""
import torch
import ct_bwd_ext as ext

torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"


def cos(p, q):
    p = p.flatten().double(); q = q.flatten().double()
    return (p @ q / (p.norm() * q.norm() + 1e-20)).item()


for tag, (M, d, ND) in {"atom": (2048, 128, 256), "token": (1024, 768, 1536)}.items():
    dout = torch.randn(M, d, device=dev)
    ws = torch.randn(d, ND, device=dev) * 0.05      # Ws:(D,ND)
    a = torch.randn(M, ND, device=dev)
    b = torch.randn(M, ND, device=dev)

    dh = dout @ ws                                  # (M, ND)
    sa = torch.sigmoid(a)
    silu = a * sa
    silu_p = sa * (1 + a * (1 - sa))
    da_ref = dh * b * silu_p
    db_ref = dh * silu

    wsT = ws.t().contiguous()                       # (ND, D) so B^T = Ws
    da = ext.fused_da(dout, wsT, a, b)
    db = ext.fused_db(dout, wsT, a)
    # dual-output: one dh-GEMM emits da (return) + db (AuxStore into provided tensor)
    db2 = torch.empty(M, ND, device=dev)
    da2 = ext.fused_dab(dout, wsT, a, b, db2)
    print(f"[{tag}] cos(da)={cos(da, da_ref):.6f}  cos(db)={cos(db, db_ref):.6f}  "
          f"| dual cos(da)={cos(da2, da_ref):.6f} cos(db)={cos(db2, db_ref):.6f}  (M={M} d={d} ND={ND})")
print("BWD-FUSED DONE")
