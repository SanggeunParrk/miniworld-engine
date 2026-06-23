"""Simple Triton gated GEMM vs quack (autotuned), for the left+right front.

left  = sigmoid(x@WLg) * (x@WL),  right = sigmoid(x@WRg) * (x@WR)
Packed: Bg = [WLg|WRg] (K,2D), Bp = [WL|WR] (K,2D); out = sigmoid(x@Bg)*(x@Bp)
out (M, 2D): [:, :D]=left, [:, D:]=right (blld contiguous).

K=128 (single block, no K-loop), N=2D=256. Basic blocked matmul + glu epilogue
(two dots so gates/projs split cleanly). Compares to quack's autotuned gemm_act.
B=1, D=128, bf16. COMPUTE NODE only.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_src_root = _Path(__file__).resolve()
while _src_root.name != "src" and _src_root.parent != _src_root:
    _src_root = _src_root.parent
if str(_src_root) not in _sys.path:
    _sys.path.insert(0, str(_src_root))

import torch
import triton
import triton.language as tl
from quack.gemm_interface import gemm_act

from miniworld_kernels.kernels.trimul_inproj.cute.launch import prepack_lr_operand

PEAK_TBPS = 3.35


@triton.autotune(
    configs=[
        triton.Config({"BM": bm}, num_warps=nw, num_stages=ns)
        for bm in (64, 128, 256)
        for nw in (4, 8)
        for ns in (2, 3, 4)
    ],
    key=["M"],
)
@triton.jit
def _gated_gemm(x_ptr, bg_ptr, bp_ptr, out_ptr, M,
                K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_k = tl.arange(0, K)
    offs_n = tl.arange(0, N)
    m_mask = offs_m[:, None] < M
    x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=m_mask, other=0.0)
    bg = tl.load(bg_ptr + offs_k[:, None] * N + offs_n[None, :])
    bp = tl.load(bp_ptr + offs_k[:, None] * N + offs_n[None, :])
    g = tl.dot(x, bg)  # (BM, N) fp32
    p = tl.dot(x, bp)
    out = (tl.sigmoid(g) * p).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], out, mask=m_mask)


def triton_gated(x, Bg, Bp):
    M, K = x.shape
    N = Bg.shape[1]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]),)  # noqa: E731
    _gated_gemm[grid](x, Bg, Bp, out, M, K=K, N=N)
    return out


def _bench(fn, *, warmup=20, rep=60):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def main():
    assert torch.cuda.is_available()
    print(f"triton vs quack gated GEMM on {torch.cuda.get_device_name(0)}", flush=True)
    dtype, D = torch.bfloat16, 128
    scale = D**-0.5

    def w():
        return (torch.randn(D, D, device="cuda", dtype=dtype) * scale).contiguous()

    WL, WLg, WR, WRg = w(), w(), w(), w()
    Bg = torch.cat([WLg, WRg], dim=1).contiguous()   # (D, 2D) gates
    Bp = torch.cat([WL, WR], dim=1).contiguous()     # (D, 2D) projs
    b_lr = prepack_lr_operand(WL, WLg, WR, WRg)       # (D, 4D) for quack glu

    # correctness (L=128)
    xchk = torch.randn(128 * 128, D, device="cuda", dtype=dtype)
    out_t = triton_gated(xchk, Bg, Bp)
    xf = xchk.float()
    left = torch.sigmoid(xf @ WLg.float()) * (xf @ WL.float())
    right = torch.sigmoid(xf @ WRg.float()) * (xf @ WR.float())
    ref = torch.cat([left, right], dim=1)
    err = (out_t.float() - ref).abs().max().item()
    print(f"triton correctness max_abs={err:.3e}  {'OK' if err < 1e-1 else 'FAIL'}", flush=True)

    print(f"\n{'L':>5} | {'quack(ms)':>9} | {'triton(ms)':>10} | {'q/t':>5} | "
          f"{'quack GB/s':>10} | {'triton GB/s':>11}")
    print("-" * 64)
    rows = []
    for L in (384, 512, 768, 1024):
        M = L * L
        x = torch.randn(M, D, device="cuda", dtype=dtype)
        post = torch.empty(M, 2 * D, device="cuda", dtype=dtype)
        bytes_ = M * D * 2 + M * (2 * D) * 2
        t_q = _bench(lambda: gemm_act(A=x, B=b_lr, activation="glu",
                                      store_preact=False, postact_out=post))
        t_t = _bench(lambda: triton_gated(x, Bg, Bp))
        gq, gt = bytes_ / (t_q * 1e-3) / 1e9, bytes_ / (t_t * 1e-3) / 1e9
        rows.append((L, t_q, t_t))
        print(f"{L:>5} | {t_q:>9.3f} | {t_t:>10.3f} | {t_q/t_t:>4.2f}x | "
              f"{gq:>10.0f} | {gt:>11.0f}", flush=True)
    print("\nDATA " + ";".join(f"{L},{q:.4f},{t:.4f}" for L, q, t in rows), flush=True)


if __name__ == "__main__":
    main()
