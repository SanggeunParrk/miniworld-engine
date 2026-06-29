"""Full fwd+bwd: cute (fused-1+4 at K=128) vs Transformer Engine vs torch.compile, d=128.

Answers "how does the wired fused backward compare to TE?" — TE/torch run a full module
fwd+bwd; ours runs layernorm_linear_fn (forward = stats-saving M1, backward = fused 1+4).
Median ms via triton.testing.do_bench, grads zeroed each iter.
"""
import torch, triton
import torch.nn.functional as F
from miniworld_kernels.kernels.layernorm_linear.autograd import layernorm_linear_fn
from miniworld_kernels.kernels.layernorm_linear.reference import LayerNormLinearRef

D = torch.device("cuda"); dt = torch.bfloat16
def bench(fn, rg): return triton.testing.do_bench(fn, warmup=25, rep=100,
                                                  quantiles=[0.5, 0.2, 0.8], grad_to_none=rg)[0]
try:
    import transformer_engine.pytorch as te
    HAVE_TE = True
except Exception as e:
    print(f"[warn] no TE: {e}"); HAVE_TE = False

d = 128; eps = 1e-5
print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}  d={d}")
print(f"{'M':>8} | {'cute(fused) ms':>14} {'TE ms':>9} {'pytorch ms':>11} | {'vs TE':>6} {'vs pt':>6}")
for M in (16384, 65536, 262144):
    torch.manual_seed(0)
    ref = LayerNormLinearRef(d, d).to(D, dt); refc = torch.compile(ref)
    g = torch.randn(M, d, device=D, dtype=dt)
    x = torch.randn(M, d, device=D, dtype=dt, requires_grad=True)
    gamma, beta, W, bias = ref.layer_norm_weight, ref.layer_norm_bias, ref.weight, ref.bias

    # cute fused (layernorm_linear_fn) — full, forward-only, and backward-only
    xs = [t.detach().clone().requires_grad_(True) for t in (x, gamma, beta, W, bias)]
    t_cute = bench(lambda: layernorm_linear_fn(*xs, eps).backward(g), xs)
    t_cfwd = bench(lambda: layernorm_linear_fn(*xs, eps), xs)
    yk = layernorm_linear_fn(*xs, eps)
    t_cbwd = bench(lambda: layernorm_linear_fn(*xs, eps).backward(g), xs) - t_cfwd
    print(f"         cute fwd-only={t_cfwd:.4f}  bwd≈{t_cbwd:.4f}")

    # torch.compile
    rg = [x, *ref.parameters()]
    t_pt = bench(lambda: refc(x).backward(g), rg)

    # TE
    if HAVE_TE:
        te_mod = te.LayerNormLinear(d, d, bias=True, params_dtype=dt).to(D)
        with torch.no_grad():
            te_mod.layer_norm_weight.copy_(gamma); te_mod.layer_norm_bias.copy_(beta)
            te_mod.weight.copy_(W); te_mod.bias.copy_(bias)
        xte = x.detach().clone().requires_grad_(True)
        teg = [xte, *te_mod.parameters()]
        t_te = bench(lambda: te_mod(xte).backward(g), teg)
    else:
        t_te = float('nan')
    print(f"{M:>8} | {t_cute:>14.4f} {t_te:>9.4f} {t_pt:>11.4f} | "
          f"{t_te/t_cute:>5.2f}x {t_pt/t_cute:>5.2f}x")
