"""Where does the ~5-9% contiguous gap vs TE come from? Decompose ours into per-op times and
split fwd/bwd for both ours and TE."""
import torch, triton
import torch.nn.functional as F
from miniworld_kernels.kernels.layernorm_linear import te_style as tes
from miniworld_kernels.kernels.layernorm_linear.reference import LayerNormLinearRef

D = torch.device("cuda"); dt = torch.bfloat16
def b(fn, rg=None): return triton.testing.do_bench(fn, warmup=25, rep=100, grad_to_none=rg or [])
torch.matmul(torch.zeros(8, 8, device=D, dtype=dt), torch.zeros(8, 8, device=D, dtype=dt)); torch.cuda.synchronize()
try:
    import transformer_engine.pytorch as te; HAVE_TE = True
except Exception as e:
    print("no TE", e); HAVE_TE = False

for (M, d) in [(262144, 256), (262144, 384)]:
    torch.manual_seed(0); eps = 1e-5
    x = torch.randn(M, d, device=D, dtype=dt)
    g = torch.randn(d, device=D, dtype=dt); be = torch.randn(d, device=D, dtype=dt)
    W = (torch.randn(d, d, device=D, dtype=dt) * d**-0.5); bias = torch.randn(d, device=D, dtype=dt)
    gout = torch.randn(M, d, device=D, dtype=dt)
    print(f"\n=== M={M} d={d} ===")

    # ---- ours, per-op ----
    xn, mean, rstd = tes._ln_materialize(x, g, be, eps)
    dxn = torch.matmul(gout, W)
    t_mat   = b(lambda: tes._ln_materialize(x, g, be, eps))
    t_fgemm = b(lambda: F.linear(xn, W, bias))
    t_dgrad = b(lambda: torch.matmul(gout, W))
    t_lnbwd = b(lambda: tes._ln_bwd(dxn, x, g, mean, rstd, x.stride()))
    t_wgrad = b(lambda: torch.matmul(gout.t(), xn))
    t_db    = b(lambda: gout.sum(0))
    ours_fwd = t_mat + t_fgemm
    ours_bwd = t_dgrad + t_lnbwd + t_wgrad + t_db
    print(f"  ours FWD  {ours_fwd:.4f} = materialize {t_mat:.4f} + fwd_gemm {t_fgemm:.4f}")
    print(f"  ours BWD  {ours_bwd:.4f} = dgrad {t_dgrad:.4f} + ln_bwd {t_lnbwd:.4f} "
          f"+ wgrad {t_wgrad:.4f} + db {t_db:.4f}")
    # reference plain GEMMs (cuBLAS ceiling for fwd/dgrad/wgrad)
    print(f"  (cuBLAS-only: fwd_gemm {t_fgemm:.4f}  dgrad {t_dgrad:.4f}  wgrad {t_wgrad:.4f} "
          f"→ 3 GEMMs = {t_fgemm+t_dgrad+t_wgrad:.4f}; LN overhead = mat+lnbwd+db = "
          f"{t_mat+t_lnbwd+t_db:.4f})")

    # ---- TE, fwd vs bwd ----
    if HAVE_TE:
        m = te.LayerNormLinear(d, d, bias=True, params_dtype=dt).to(D)
        with torch.no_grad():
            m.layer_norm_weight.copy_(g); m.layer_norm_bias.copy_(be); m.weight.copy_(W); m.bias.copy_(bias)
        xt = x.clone().requires_grad_(True); teg = [xt, *m.parameters()]
        t_te_fwd = b(lambda: m(xt))
        t_te_full = b(lambda: m(xt).backward(gout), teg)
        print(f"  TE   FWD  {t_te_fwd:.4f}   TE BWD  {t_te_full - t_te_fwd:.4f}   TE full {t_te_full:.4f}")
        print(f"  Δ FWD (ours-TE) {ours_fwd - t_te_fwd:+.4f}   Δ BWD {ours_bwd - (t_te_full-t_te_fwd):+.4f}")
