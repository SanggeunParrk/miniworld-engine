"""Verify TE-style trainable LayerNormLinear: correctness on contiguous AND m-major (trimul)
input, + that dx comes out in the SAME layout as x (stride coverage, no copy)."""
import torch, torch.nn.functional as F
from miniworld_kernels.kernels.layernorm_linear.te_style import layernorm_linear_te_fn
D = torch.device("cuda"); dt = torch.bfloat16
def cos(a, b): return F.cosine_similarity(a.float().flatten(), b.float().flatten(), 0).item()


def run(M, K, N, mmajor):
    eps = 1e-5; torch.manual_seed(0)
    if mmajor:   # trimul BDLL view: (M=L*L, K=D), strides (1, M)  [reshape(D, L*L).t()]
        base = torch.randn(K, M, device=D, dtype=dt)
        x = base.t()                       # (M,K) strides (1, M)  m-major
    else:
        x = torch.randn(M, K, device=D, dtype=dt)
    g = torch.randn(K, device=D, dtype=dt); b = torch.randn(K, device=D, dtype=dt)
    W = (torch.randn(N, K, device=D, dtype=dt) * K**-0.5); bias = torch.randn(N, device=D, dtype=dt)
    dY = torch.randn(M, N, device=D, dtype=dt)

    # oracle (fp32 autograd on a contiguous copy of x)
    xo = x.float().clone().requires_grad_(True); go = g.float().clone().requires_grad_(True)
    bo = b.float().clone().requires_grad_(True); Wo = W.float().clone().requires_grad_(True)
    bio = bias.float().clone().requires_grad_(True)
    Y = F.layer_norm(xo, (K,), go, bo, eps) @ Wo.t() + bio
    Y.backward(dY.float())

    xs0 = x.clone().requires_grad_(True)          # keeps x's strides
    others = [t.clone().requires_grad_(True) for t in (g, b, W, bias)]
    Yk = layernorm_linear_te_fn(xs0, *others, eps); Yk.backward(dY)
    dx_strides_match = (xs0.grad.stride() == x.stride())
    tag = "m-major" if mmajor else "contig "
    print(f"  [{tag}] M={M} K={K} N={N}  Y={cos(Yk,Y):.5f} dx={cos(xs0.grad,xo.grad):.5f} "
          f"dg={cos(others[0].grad,go.grad):.5f} db={cos(others[1].grad,bo.grad):.5f} "
          f"dW={cos(others[2].grad,Wo.grad):.5f} dbias={cos(others[3].grad,bio.grad):.5f}  "
          f"dx.stride={xs0.grad.stride()} (matches x: {dx_strides_match})")


for mm in (False, True):
    for (M, K, N) in [(65536, 128, 128), (65536, 256, 256), (16384, 384, 384)]:
        run(M, K, N, mm)
