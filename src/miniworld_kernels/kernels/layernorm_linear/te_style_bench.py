"""Full fwd+bwd: TE-style custom LayerNormLinear vs Transformer Engine vs torch.compile.

Two regimes:
  contig  : x is contiguous (M,K) — fair head-to-head (no copy advantage).
  m-major : x is a trimul BDLL view (strides (1,M)) — TE must .contiguous() (BDLL→BLLD copy),
            ours consumes it directly. Shows the stride-coverage win.
Median ms via triton.testing.do_bench, grads zeroed each iter.
"""
import torch, triton
import torch.nn.functional as F
from miniworld_kernels.kernels.layernorm_linear.te_style import layernorm_linear_te_fn
from miniworld_kernels.kernels.layernorm_linear.reference import LayerNormLinearRef

D = torch.device("cuda"); dt = torch.bfloat16
def bench(fn, rg): return triton.testing.do_bench(fn, warmup=25, rep=100,
                                                  quantiles=[0.5, 0.2, 0.8], grad_to_none=rg)[0]
try:
    import transformer_engine.pytorch as te
    HAVE_TE = True
except Exception as e:
    print(f"[warn] no TE: {e}"); HAVE_TE = False

torch.matmul(torch.zeros(8, 8, device=D, dtype=dt), torch.zeros(8, 8, device=D, dtype=dt))
torch.cuda.synchronize()   # establish CUDA context before timed regions (avoid cold-context noise)
print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
for mmajor in (False, True):
    print(f"\n=== {'m-major (trimul view; TE needs .contiguous())' if mmajor else 'contiguous'} ===")
    print(f"{'M':>8} {'d':>5} | {'ours ms':>8} {'TE ms':>8} {'pytorch ms':>11} | {'vs TE':>6} {'vs pt':>6}")
    for d in (128, 256, 384, 512):
        for M in (16384, 65536, 262144):
            torch.manual_seed(0)
            ref = LayerNormLinearRef(d, d).to(D, dt)
            refc = torch.compile(ref)
            g = torch.randn(M, d, device=D, dtype=dt)
            gamma, beta, W, bias = ref.layer_norm_weight, ref.layer_norm_bias, ref.weight, ref.bias
            if mmajor:
                xbase = torch.randn(d, M, device=D, dtype=dt); xv = xbase.t()   # (M,d) strides (1,M)
            else:
                xv = torch.randn(M, d, device=D, dtype=dt)

            # ours — consumes xv directly (any layout)
            os_ = [xv.clone().detach().requires_grad_(True),
                   *[t.detach().clone().requires_grad_(True) for t in (gamma, beta, W, bias)]]
            t_ours = bench(lambda: layernorm_linear_te_fn(*os_, 1e-5).backward(g), os_)

            # torch.compile (ref handles strides via its own ops)
            xr = xv.clone().detach().requires_grad_(True)
            rg = [xr, *ref.parameters()]
            t_pt = bench(lambda: refc(xr).backward(g), rg)

            # TE — m-major must be made contiguous first (the copy we avoid)
            if HAVE_TE:
                te_mod = te.LayerNormLinear(d, d, bias=True, params_dtype=dt).to(D)
                with torch.no_grad():
                    te_mod.layer_norm_weight.copy_(gamma); te_mod.layer_norm_bias.copy_(beta)
                    te_mod.weight.copy_(W); te_mod.bias.copy_(bias)
                xt = xv.clone().detach().requires_grad_(True)
                teg = [xt, *te_mod.parameters()]
                if mmajor:
                    t_te = bench(lambda: te_mod(xt.contiguous()).backward(g), teg)
                else:
                    t_te = bench(lambda: te_mod(xt).backward(g), teg)
            else:
                t_te = float('nan')
            print(f"{M:>8} {d:>5} | {t_ours:>8.4f} {t_te:>8.4f} {t_pt:>11.4f} | "
                  f"{t_te/t_ours:>5.2f}x {t_pt/t_ours:>5.2f}x")
