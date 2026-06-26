"""Verify the fused-1+4 backward decomposition.

dx  : dgrad_lnbwd_cute (fused dY@W GEMM + LN-norm-backward epilogue), no dx_normed round-trip.
dW/dγ/dβ/db : derived from T = dYᵀ@x̂ (one wgrad GEMM) — no dx_normed, no x_normed recompute:
    db = Σ_m dY ;  dW = γ⊙T + outer(db,β) ;  dγ = (W⊙T).sum(0) ;  dβ = db @ W
All compared cos vs a torch autograd oracle on the real LayerNormLinear forward.
"""
import torch, torch.nn.functional as F
from miniworld_kernels.kernels.layernorm_linear.cute.dgrad_lnbwd import dgrad_lnbwd_cute
D = torch.device("cuda"); dt = torch.bfloat16
def cos(a, b): return F.cosine_similarity(a.float().flatten(), b.float().flatten(), 0).item()

for (M, K, N) in [(8192, 128, 128), (8192, 256, 256), (16384, 128, 512), (4096, 256, 384)]:
    eps = 1e-5; torch.manual_seed(0)
    x = torch.randn(M, K, device=D, dtype=dt)
    g = torch.randn(K, device=D, dtype=dt); b_ln = torch.randn(K, device=D, dtype=dt)
    w = (torch.randn(N, K, device=D, dtype=dt) * K**-0.5); bias = torch.randn(N, device=D, dtype=dt)
    dY = torch.randn(M, N, device=D, dtype=dt)

    # ---- autograd oracle on the true forward Y = LN(x)@Wᵀ + bias ----
    xo = x.clone().float().requires_grad_(True)
    go = g.clone().float().requires_grad_(True); bo = b_ln.clone().float().requires_grad_(True)
    wo = w.clone().float().requires_grad_(True); biaso = bias.clone().float().requires_grad_(True)
    xn = F.layer_norm(xo, (K,), go, bo, eps)
    Y = xn @ wo.t() + biaso
    Y.backward(dY.float())
    dx_ref, dg_ref, db_ref, dW_ref, dbias_ref = xo.grad, go.grad, bo.grad, wo.grad, biaso.grad

    # ---- our decomposition ----
    mean = x.float().mean(-1); rstd = torch.rsqrt(x.float().var(-1, unbiased=False) + eps)
    xhat = ((x.float() - mean[:, None]) * rstd[:, None]).to(dt)        # saved x̂ (bf16)
    dx = dgrad_lnbwd_cute(dY, w, xhat, g, rstd)                        # fused 1+4 → dx
    T = (dY.float().t() @ xhat.float())                               # (N,K) wgrad on x̂
    dbias = dY.float().sum(0)                                          # (N,)  linear bias grad
    dW = g.float()[None, :] * T + dbias[:, None] * b_ln.float()[None, :]
    dgamma = (w.float() * T).sum(0)
    dbeta = dbias @ w.float()

    print(f"M={M} K={K} N={N}  "
          f"dx={cos(dx, dx_ref):.5f} dW={cos(dW, dW_ref):.5f} "
          f"dγ={cos(dgamma, dg_ref):.5f} dβ={cos(dbeta, db_ref):.5f} "
          f"dbias={cos(dbias, dbias_ref):.5f}")
