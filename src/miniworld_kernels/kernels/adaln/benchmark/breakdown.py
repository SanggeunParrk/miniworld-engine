"""Per-component breakdown of adaln_train fwd+bwd at token d=768 to find the bottleneck vs compile."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from miniworld_kernels.kernels.adaln.triton.training import (
    _bwd_x, _epilogue_train,
)
from miniworld_kernels.kernels.layernorm_linear.te_style import (
    _ln_materialize, _ln_bwd, _bias_grad, _fp32_matmul_ctx,
)

DEVICE = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def t(fn, gtn=None):
    m, _, _ = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8],
                                      grad_to_none=gtn or [])
    return m * 1000  # us


def run(d, seq, n_aug=32, dtype=torch.float32, eps=1e-5):
    M = n_aug * seq
    NX = NC = d
    x = torch.randn(M, NX, device=DEVICE, dtype=dtype)
    cond = torch.randn(M, NC, device=DEVICE, dtype=dtype)
    lnw = torch.randn(NC, device=DEVICE, dtype=dtype)
    Ws = torch.randn(NX, NC, device=DEVICE, dtype=dtype) * NC ** -0.5
    Wb = torch.randn(NX, NC, device=DEVICE, dtype=dtype) * NC ** -0.5
    sb_b = torch.randn(NX, device=DEVICE, dtype=dtype) * 0.1
    beta0 = torch.zeros(NC, device=DEVICE, dtype=dtype)
    w_cat = torch.cat([Ws, Wb], 0).contiguous()
    b_cat = torch.cat([sb_b, torch.zeros(NX, device=DEVICE, dtype=dtype)], 0).contiguous()
    dy = torch.randn(M, NX, device=DEVICE, dtype=dtype)

    print(f"\n## d={d} seq={seq} M={M} {dtype}")

    # ---- forward components ----
    cond_aff, mean_c, rstd_c = _ln_materialize(cond, lnw, beta0, eps)
    with _fp32_matmul_ctx(dtype):
        sb = F.linear(cond_aff, w_cat, b_cat)
    y, mean_x, rstd_x, gate = _epilogue_train(x, sb, eps)

    t_condaff = t(lambda: _ln_materialize(cond, lnw, beta0, eps))
    t_fgemm = t(lambda: F.linear(cond_aff, w_cat, b_cat))
    t_epi = t(lambda: _epilogue_train(x, sb, eps))
    fwd = t_condaff + t_fgemm + t_epi
    print(f"  FWD: cond_aff={t_condaff:.1f}  gemm={t_fgemm:.1f}  epilogue={t_epi:.1f}  | sum={fwd:.1f}")

    # ---- backward components ----
    D, dx = _bwd_x(dy, x, mean_x, rstd_x, gate)
    t_bwdx = t(lambda: _bwd_x(dy, x, mean_x, rstd_x, gate))
    with _fp32_matmul_ctx(dtype):
        t_dcond = t(lambda: torch.matmul(D, w_cat))
        t_dwcat = t(lambda: torch.matmul(D.t(), cond_aff))
        t_dsb = t(lambda: _bias_grad(D[:, :NX].contiguous()))
    dcond_aff = torch.matmul(D, w_cat)
    t_condbwd = t(lambda: _ln_bwd(dcond_aff, cond, lnw, mean_c, rstd_c, cond.stride()))
    bwd = t_bwdx + t_dcond + t_dwcat + t_dsb + t_condbwd
    print(f"  BWD: bwd_x={t_bwdx:.1f}  dcond_aff(D@W)={t_dcond:.1f}  dW(Dt@aff)={t_dwcat:.1f}  "
          f"dsb={t_dsb:.1f}  condLNbwd={t_condbwd:.1f}  | sum={bwd:.1f}")
    print(f"  TOTAL fwd+bwd ~= {fwd+bwd:.1f} us")


def main():
    print("torch", torch.__version__, torch.cuda.get_device_name(0))
    for seq in (384, 1024):
        run(768, seq)
    run(128, 8192)


if __name__ == "__main__":
    main()
