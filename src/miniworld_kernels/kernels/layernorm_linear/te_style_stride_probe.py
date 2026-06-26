"""Probe: does TE accept a strided (trimul BDLL m-major) input directly?
And does our te_fn take the real bmm BDLL output and do the right thing?

Builds the ACTUAL trimul-shaped tensor: left/right [B,D,L,L], bmm-style einsum → tri [B,D,L,L],
then the (M=L*L, K=D) m-major view (strides (1, L*L)) — exactly what feeds the next LayerNorm.
Feeds that view RAW (no .contiguous()) to (a) TE, (b) ours; checks run/correctness/layout/timing.
"""
import torch, triton
import torch.nn.functional as F
from miniworld_kernels.kernels.layernorm_linear.te_style import layernorm_linear_te_fn
D = torch.device("cuda"); dt = torch.bfloat16
def cos(a, b): return F.cosine_similarity(a.float().flatten(), b.float().flatten(), 0).item()
def bench(fn, rg):
    try: return triton.testing.do_bench(fn, warmup=20, rep=80, grad_to_none=rg)
    except Exception as e: return f"ERR {type(e).__name__}"

try:
    import transformer_engine.pytorch as te
    HAVE_TE = True
except Exception as e:
    print(f"[warn] no TE: {e}"); HAVE_TE = False

B, Dd, L = 1, 128, 256
M, K, N = L * L, Dd, Dd
eps = 1e-5; torch.manual_seed(0)

# real trimul bmm output, BDLL
left = torch.randn(B, Dd, L, L, device=D, dtype=dt)
right = torch.randn(B, Dd, L, L, device=D, dtype=dt)
tri = torch.einsum("bdik,bdjk->bdij", left, right)          # [B,D,L,L] BDLL
xv = tri.reshape(B, Dd, M)[0].t()                          # (M=L*L, K=D), strides (1, L*L)  m-major
print(f"trimul view: shape={tuple(xv.shape)} strides={xv.stride()} is_contiguous={xv.is_contiguous()}")

g = torch.randn(K, device=D, dtype=dt); b = torch.randn(K, device=D, dtype=dt)
W = (torch.randn(N, K, device=D, dtype=dt) * K**-0.5); bias = torch.randn(N, device=D, dtype=dt)
gout = torch.randn(M, N, device=D, dtype=dt)

# fp32 oracle on a contiguous copy of the view
xo = xv.float().contiguous().requires_grad_(True)
go = g.float().clone().requires_grad_(True); bo = b.float().clone().requires_grad_(True)
Wo = W.float().clone().requires_grad_(True); bio = bias.float().clone().requires_grad_(True)
Yo = F.layer_norm(xo, (K,), go, bo, eps) @ Wo.t() + bio
Yo.backward(gout.float())

print("\n--- (1) TE fed the RAW strided view (no .contiguous()) ---")
if HAVE_TE:
    te_mod = te.LayerNormLinear(Dd, Dd, bias=True, params_dtype=dt).to(D)
    with torch.no_grad():
        te_mod.layer_norm_weight.copy_(g); te_mod.layer_norm_bias.copy_(b)
        te_mod.weight.copy_(W); te_mod.bias.copy_(bias)
    xt = xv.clone().detach().requires_grad_(True)            # keeps strided layout
    try:
        Yte = te_mod(xt); Yte.backward(gout)
        print(f"  RAN. Y cos={cos(Yte,Yo):.5f}  dx cos={cos(xt.grad,xo.grad):.5f}  "
              f"dx.stride={xt.grad.stride()} (x.stride={xv.stride()})")
    except Exception as e:
        print(f"  RAISED {type(e).__name__}: {e}")

print("\n--- (2) ours fed the RAW strided view ---")
xs = [xv.clone().detach().requires_grad_(True),
      *[t.clone().requires_grad_(True) for t in (g, b, W, bias)]]
Yk = layernorm_linear_te_fn(*xs, eps); Yk.backward(gout)
print(f"  Y cos={cos(Yk,Yo):.5f}  dx cos={cos(xs[0].grad,xo.grad):.5f}  "
      f"dx.stride={xs[0].grad.stride()} matches_x={xs[0].grad.stride()==xv.stride()}")

print("\n--- (3) timing (full fwd+bwd, ms) on the raw view ---")
if HAVE_TE:
    xt2 = xv.clone().detach().requires_grad_(True); teg = [xt2, *te_mod.parameters()]
    print(f"  TE   raw-strided : {bench(lambda: te_mod(xt2).backward(gout), teg)}")
    xt3 = xv.clone().detach().requires_grad_(True); teg3 = [xt3, *te_mod.parameters()]
    print(f"  TE   .contiguous(): {bench(lambda: te_mod(xt3.contiguous()).backward(gout), teg3)}")
print(f"  ours raw-strided : {bench(lambda: layernorm_linear_te_fn(*xs, eps).backward(gout), xs)}")
