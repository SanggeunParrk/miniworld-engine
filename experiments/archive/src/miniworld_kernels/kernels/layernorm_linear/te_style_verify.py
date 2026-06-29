"""Verify TE-style trainable LayerNormLinear across dtypes (fp32/bf16/fp16) and layouts
(contiguous + m-major trimul view), incl. dx returned in the input layout and the fp32 TF32 knob."""
import torch, torch.nn.functional as F
from miniworld_kernels.kernels.layernorm_linear.te_style import (
    layernorm_linear_te_fn, set_fp32_matmul_precision,
)
D = torch.device("cuda")
def cos(a, b): return F.cosine_similarity(a.float().flatten(), b.float().flatten(), 0).item()


def run(M, K, N, dtype, mmajor):
    eps = 1e-5; torch.manual_seed(0)
    if mmajor:                                   # trimul BDLL view: (M=L*L, K=D), strides (1, M)
        x = torch.randn(K, M, device=D, dtype=dtype).t()
    else:
        x = torch.randn(M, K, device=D, dtype=dtype)
    g = torch.randn(K, device=D, dtype=dtype); b = torch.randn(K, device=D, dtype=dtype)
    W = (torch.randn(N, K, device=D, dtype=dtype) * K**-0.5); bias = torch.randn(N, device=D, dtype=dtype)
    dY = torch.randn(M, N, device=D, dtype=dtype)

    # true-fp32 autograd oracle (always full fp32, regardless of the path's dtype/precision)
    xo = x.float().clone().requires_grad_(True); go = g.float().clone().requires_grad_(True)
    bo = b.float().clone().requires_grad_(True); Wo = W.float().clone().requires_grad_(True)
    bio = bias.float().clone().requires_grad_(True)
    Y = F.layer_norm(xo, (K,), go, bo, eps) @ Wo.t() + bio
    Y.backward(dY.float())

    xs0 = x.clone().requires_grad_(True)
    others = [t.clone().requires_grad_(True) for t in (g, b, W, bias)]
    Yk = layernorm_linear_te_fn(xs0, *others, eps); Yk.backward(dY)
    layout_ok = (xs0.grad.stride() == x.stride()) and (xs0.grad.dtype == dtype)
    worst = min(cos(Yk, Y), cos(xs0.grad, xo.grad), cos(others[0].grad, go.grad),
                cos(others[1].grad, bo.grad), cos(others[2].grad, Wo.grad), cos(others[3].grad, bio.grad))
    tag = "m-major" if mmajor else "contig "
    print(f"  {str(dtype).replace('torch.',''):>8} [{tag}] M={M} K={K} N={N}  "
          f"worst-cos={worst:.6f}  dx.stride={xs0.grad.stride()} dtype-ok={layout_ok}")


for dtype in (torch.float32, torch.bfloat16, torch.float16):
    if dtype is torch.float32:
        set_fp32_matmul_precision("highest")     # match the true-fp32 oracle for the cos check
    for mm in (False, True):
        run(65536, 256, 256, dtype, mm)

# fp32 TF32 ('high') still correct (looser vs true-fp32 oracle, but cos must stay ~1)
set_fp32_matmul_precision("high")
print("  -- fp32 TF32('high') --")
run(65536, 256, 256, torch.float32, False)
