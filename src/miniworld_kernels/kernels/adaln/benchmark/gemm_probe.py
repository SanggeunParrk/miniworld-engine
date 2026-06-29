"""Probe wgrad/dgrad GEMM layout variants for adaln backward at token d=768, M=32768."""
from __future__ import annotations
import torch
import triton

DEVICE = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def t(fn):
    m, _, _ = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8])
    return m * 1000


def run(M, NX, NC, dtype=torch.float32):
    print(f"\n## M={M} NX={NX} NC={NC} {dtype}")
    dscale = torch.randn(M, NX, device=DEVICE, dtype=dtype)
    dy = torch.randn(M, NX, device=DEVICE, dtype=dtype)
    D = torch.randn(M, 2 * NX, device=DEVICE, dtype=dtype)
    cond_aff = torch.randn(M, NC, device=DEVICE, dtype=dtype)
    Ws = torch.randn(NX, NC, device=DEVICE, dtype=dtype)
    Wb = torch.randn(NX, NC, device=DEVICE, dtype=dtype)
    w_cat = torch.cat([Ws, Wb], 0).contiguous()      # (2NX, NC)

    # ---- wgrad: want [dWs;dWb] = D^T @ cond_aff  (2NX, NC) ----
    print("  -- wgrad --")
    print(f"   D.t()@cond_aff            : {t(lambda: torch.matmul(D.t(), cond_aff)):.1f}")
    print(f"   (cond_aff.t()@D).t()      : {t(lambda: torch.matmul(cond_aff.t(), D).t()):.1f}")
    print(f"   separate dscale.t,dy.t    : {t(lambda: (torch.matmul(dscale.t(), cond_aff), torch.matmul(dy.t(), cond_aff))):.1f}")
    Dc = D.t().contiguous()  # (2NX, M) contiguous
    print(f"   Dc(contig)@cond_aff       : {t(lambda: torch.matmul(Dc, cond_aff)):.1f}  (+ contiguify cost excl.)")

    # ---- dgrad: want dcond_aff = D @ w_cat  (M, NC) ----
    print("  -- dgrad --")
    print(f"   D@w_cat                   : {t(lambda: torch.matmul(D, w_cat)):.1f}")
    print(f"   addmm dscale@Ws + dy@Wb   : {t(lambda: torch.addmm(torch.matmul(dscale, Ws), dy, Wb)):.1f}")

    # ---- fwd gemm: [scale|bias] = cond_aff @ w_cat.t() ----
    print("  -- fwd gemm --")
    wt = w_cat.t().contiguous()  # (NC, 2NX)
    print(f"   cond_aff@w_cat.t()        : {t(lambda: torch.matmul(cond_aff, w_cat.t())):.1f}")
    print(f"   cond_aff@wt(contig)       : {t(lambda: torch.matmul(cond_aff, wt)):.1f}")


def main():
    print("torch", torch.__version__, torch.cuda.get_device_name(0))
    run(32768, 768, 768)
    run(12288, 768, 768)
    run(262144, 128, 128)


if __name__ == "__main__":
    main()
