"""Speed across dtypes: TE-style ours vs TE vs torch.compile, full fwd+bwd, H100.
fp32 uses the default TF32 ('high') policy. Reports contiguous (fair) + m-major (trimul) speedups."""
import torch, triton
import torch.nn.functional as F
from miniworld_kernels.kernels.layernorm_linear.te_style import (
    layernorm_linear_te_fn, set_fp32_matmul_precision,
)
from miniworld_kernels.kernels.layernorm_linear.reference import LayerNormLinearRef

D = torch.device("cuda")
def bench(fn, rg): return triton.testing.do_bench(fn, warmup=25, rep=100,
                                                  quantiles=[0.5, 0.2, 0.8], grad_to_none=rg)[0]
torch.matmul(torch.zeros(8, 8, device=D), torch.zeros(8, 8, device=D)); torch.cuda.synchronize()
try:
    import transformer_engine.pytorch as te; HAVE_TE = True
except Exception as e:
    print("no TE", e); HAVE_TE = False

SHAPES = [(65536, 256), (262144, 256), (65536, 512), (262144, 512)]

for dtype in (torch.float32, torch.bfloat16, torch.float16):
    if dtype is torch.float32:
        set_fp32_matmul_precision("high")   # TF32 (the fast default)
    print(f"\n===== dtype={str(dtype).replace('torch.','')} "
          f"{'(fp32 GEMM = TF32/high)' if dtype is torch.float32 else ''} =====")
    print(f"{'M':>8} {'d':>5} | {'contig: ours':>12} {'TE':>8} {'vsTE':>5} | "
          f"{'mmaj: ours':>11} {'TE':>8} {'vsTE':>5}")
    for (M, d) in SHAPES:
        torch.manual_seed(0)
        ref = LayerNormLinearRef(d, d).to(D, dtype)
        g = torch.randn(M, d, device=D, dtype=dtype)
        gamma, beta, W, bias = ref.layer_norm_weight, ref.layer_norm_bias, ref.weight, ref.bias

        def te_mod():
            if not HAVE_TE: return None
            m = te.LayerNormLinear(d, d, bias=True, params_dtype=dtype).to(D)
            with torch.no_grad():
                m.layer_norm_weight.copy_(gamma); m.layer_norm_bias.copy_(beta)
                m.weight.copy_(W); m.bias.copy_(bias)
            return m

        row = [f"{M:>8} {d:>5} |"]
        for mmajor in (False, True):
            if mmajor:
                xv = torch.randn(d, M, device=D, dtype=dtype).t()
            else:
                xv = torch.randn(M, d, device=D, dtype=dtype)
            os_ = [xv.clone().detach().requires_grad_(True),
                   *[t.detach().clone().requires_grad_(True) for t in (gamma, beta, W, bias)]]
            t_ours = bench(lambda: layernorm_linear_te_fn(*os_, 1e-5).backward(g), os_)
            try:
                m = te_mod(); xt = xv.clone().detach().requires_grad_(True); teg = [xt, *m.parameters()]
                call = (lambda: m(xt.contiguous()).backward(g)) if mmajor else (lambda: m(xt).backward(g))
                t_te = bench(call, teg)
                sp = f"{t_te/t_ours:.2f}"
            except Exception as e:
                t_te, sp = float('nan'), f"ERR:{type(e).__name__}"
            row.append(f" {t_ours:>10.4f} {t_te:>8.4f} {sp:>5} |")
        print("".join(row))
