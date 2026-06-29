"""Validate + time the fused front-backward triton kernels vs torch reference."""

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

from miniworld_kernels.kernels.trimul_inproj.cute.launch import prepack_lr_operand
from miniworld_kernels.kernels.trimul_inproj.triton.back_fused import front_bwd_fused


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def ref(d_lr, preact, x_n, WL, WLg, WR, WRg):
    B, D2, L, _ = d_lr.shape
    D = D2 // 2
    M = B * L * L
    dL = d_lr.reshape(D2, M)[:D].t().float()
    dR = d_lr.reshape(D2, M)[D:].t().float()
    pre = preact.reshape(4 * D, M).t().float()  # (M,4D) interleaved
    gLlog, pL = pre[:, 0:2 * D:2], pre[:, 1:2 * D:2]
    gRlog, pR = pre[:, 2 * D:4 * D:2], pre[:, 2 * D + 1:4 * D:2]
    gL, gR = torch.sigmoid(gLlog), torch.sigmoid(gRlog)
    d_pL = dL * gL
    d_gLlog = dL * pL * gL * (1 - gL)
    d_pR = dR * gR
    d_gRlog = dR * pR * gR * (1 - gR)
    xf = x_n.reshape(M, D).float()
    dx = (d_gLlog @ WLg.float().t() + d_pL @ WL.float().t()
          + d_gRlog @ WRg.float().t() + d_pR @ WR.float().t()).reshape(B, L, L, D)
    return dx, xf.t() @ d_pL, xf.t() @ d_gLlog, xf.t() @ d_pR, xf.t() @ d_gRlog


def main():
    assert torch.cuda.is_available()
    print(f"fused front-bwd on {torch.cuda.get_device_name(0)}", flush=True)
    torch.manual_seed(0)
    D = 128
    for L in (256, 512, 1024):
        M = L * L
        dt = torch.bfloat16
        x_n = torch.randn(1, L, L, D, device="cuda", dtype=dt) * 0.5
        WL, WLg, WR, WRg = (torch.randn(D, D, device="cuda", dtype=dt) * D**-0.5 for _ in range(4))
        b_lr = prepack_lr_operand(WL, WLg, WR, WRg)
        preact = (x_n.reshape(M, D) @ b_lr).t().reshape(1, 4 * D, L, L).contiguous()
        d_lr = torch.randn(1, 2 * D, L, L, device="cuda", dtype=dt)
        d_left, d_right = d_lr[:, :D].contiguous(), d_lr[:, D:].contiguous()

        dx, dWL, dWLg, dWR, dWRg = front_bwd_fused(d_left, d_right, preact, x_n, WL, WLg, WR, WRg)
        rdx, rdWL, rdWLg, rdWR, rdWRg = ref(d_lr, preact, x_n, WL, WLg, WR, WRg)
        if L == 256:
            print(f"  cos dx={cos(dx,rdx):.5f} dWL={cos(dWL,rdWL):.5f} dWLg={cos(dWLg,rdWLg):.5f} "
                  f"dWR={cos(dWR,rdWR):.5f} dWRg={cos(dWRg,rdWRg):.5f}", flush=True)

        def fused():
            front_bwd_fused(d_left, d_right, preact, x_n, WL, WLg, WR, WRg)

        def torch_path():
            ref(d_lr, preact, x_n, WL, WLg, WR, WRg)
        tf = triton.testing.do_bench(fused, warmup=10, rep=50, return_mode="median")
        tt = triton.testing.do_bench(torch_path, warmup=10, rep=50, return_mode="median")
        print(f"  L={L:>4}: fused {tf:.3f} ms | torch-ref {tt:.3f} ms", flush=True)


if __name__ == "__main__":
    main()
