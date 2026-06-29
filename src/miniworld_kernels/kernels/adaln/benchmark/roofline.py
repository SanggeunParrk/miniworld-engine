"""Roofline + autotune-config audit for adaln_train components (token d=768, atom d=128).

Shows, per kernel: measured us, useful HBM bytes → HBM-roofline us → % of roofline (mem-bound parts),
and for GEMMs: GFLOP → TF32-roofline us → % achieved. Plus the autotuner's chosen config (to detect
edge-of-search-space picks). H100 SXM: HBM ~3.35 TB/s, TF32 dense ~494 TFLOP/s.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import triton

from miniworld_kernels.kernels.adaln.triton import training as T
from miniworld_kernels.kernels.layernorm_linear.te_style import (
    _ln_materialize, _ln_bwd, _fp32_matmul_ctx,
)

DEVICE = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
HBM = 3.35e12      # bytes/s
TF32 = 494e12      # flop/s


def t(fn):
    m, _, _ = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8])
    return m * 1000  # us


def mem(us, gbytes):
    roof = gbytes / HBM * 1e6
    return f"{us:7.1f}us  bytes={gbytes/1e6:6.0f}MB  roof={roof:6.1f}us  {roof/us*100:4.0f}%ofpeak"


def flop(us, gflop):
    roof = gflop / TF32 * 1e6
    return f"{us:7.1f}us  {gflop/1e9:6.1f}GF  roof={roof:6.1f}us  {roof/us*100:4.0f}%ofpeak"


def cfg(autotuner):
    try:
        bc = autotuner.best_config
        return str(bc)
    except Exception:
        return "n/a"


def run(d, seq, n_aug=32, dt=torch.float32, eps=1e-5):
    M = n_aug * seq
    NX = NC = d
    el = 4 if dt == torch.float32 else 2
    x = torch.randn(M, NX, device=DEVICE, dtype=dt)
    cond = torch.randn(M, NC, device=DEVICE, dtype=dt)
    lnw = torch.randn(NC, device=DEVICE, dtype=dt)
    Ws = torch.randn(NX, NC, device=DEVICE, dtype=dt) * NC ** -0.5
    Wb = torch.randn(NX, NC, device=DEVICE, dtype=dt) * NC ** -0.5
    sb_b = torch.randn(NX, device=DEVICE, dtype=dt) * 0.1
    beta0 = torch.zeros(NC, device=DEVICE, dtype=dt)
    w_cat = torch.cat([Ws, Wb], 0).contiguous()
    b_cat = torch.cat([sb_b, torch.zeros(NX, device=DEVICE, dtype=dt)], 0).contiguous()
    dy = torch.randn(M, NX, device=DEVICE, dtype=dt)
    print(f"\n#### d={d} seq={seq} M={M} {dt}  (el={el}B)")

    # FWD
    cond_aff, mean_c, rstd_c = _ln_materialize(cond, lnw, beta0, eps)
    sb = F.linear(cond_aff, w_cat, b_cat)
    y, mean_x, rstd_x, gate = T._epilogue_train(x, sb, eps)

    print("FWD")
    print("  cond_aff LN  :", mem(t(lambda: _ln_materialize(cond, lnw, beta0, eps)),
                                  (M*NC + M*NC)*el + M*4*2))  # read cond, write aff, +stats
    print("  GEMM s|b     :", flop(t(lambda: F.linear(cond_aff, w_cat, b_cat)), 2*M*NC*2*NX))
    print("  epilogue     :", mem(t(lambda: T._epilogue_train(x, sb, eps)),
                                  (M*NX + M*2*NX)*el + (M*NX + M*NX)*el + M*4*2))  # rd x,sb wr y,gate
    print("  cfg cond_aff :", cfg(_ln_materialize.__wrapped__ if hasattr(_ln_materialize,'__wrapped__') else None) if False else cfg(__import__('miniworld_kernels.kernels.layernorm_linear.te_style', fromlist=['_ln_mat_kernel'])._ln_mat_kernel))
    print("  cfg epilogue :", cfg(T._epilogue_train_kernel))

    # BWD
    D, dx = T._bwd_x(dy, x, mean_x, rstd_x, gate)
    print("BWD")
    print("  bwd_x        :", mem(t(lambda: T._bwd_x(dy, x, mean_x, rstd_x, gate)),
                                  (M*NX*3)*el + (2*NX*M + M*NX)*el + M*4*2))  # rd dy,x,gate wr D,dx
    with _fp32_matmul_ctx(dt):
        print("  wgrad D@aff  :", flop(t(lambda: torch.matmul(D, cond_aff)), 2*2*NX*M*NC))
        print("  dgrad Dt@W   :", flop(t(lambda: torch.matmul(D.t(), w_cat)), 2*M*2*NX*NC))
        print("  dsb sum      :", mem(t(lambda: D[:NX].sum(1)), (NX*M)*el))
    dcond_aff = torch.matmul(D.t(), w_cat)
    print("  cond LN bwd  :", mem(t(lambda: _ln_bwd(dcond_aff, cond, lnw, mean_c, rstd_c, cond.stride())),
                                  (M*NC + M*NC)*el + (M*NC)*el))
    print("  cfg bwd_x    :", cfg(T._bwd_x_kernel))


def main():
    print("torch", torch.__version__, torch.cuda.get_device_name(0))
    print(f"HBM={HBM/1e12}TB/s  TF32={TF32/1e12}TFLOP/s")
    run(768, 1024)
    run(128, 8192)


if __name__ == "__main__":
    main()
