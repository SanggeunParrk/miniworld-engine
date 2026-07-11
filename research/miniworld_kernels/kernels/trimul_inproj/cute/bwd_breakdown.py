"""Diagnose where ours' backward time goes @L=1024: time the major bwd components
standalone (compiled), bf16, M=L*L, D=128. COMPUTE NODE.
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
import torch.nn.functional as F
import triton


def bench(fn, comp=True):
    f = torch.compile(fn) if comp else fn
    try:
        for _ in range(5):
            f()
        return triton.testing.do_bench(f, warmup=10, rep=50, return_mode="median")
    except Exception as e:  # noqa: BLE001
        return float("nan")


def main():
    D = 128
    for L in (512, 1024):
        M = L * L
        dt = torch.bfloat16
        x = torch.randn(M, D, device="cuda", dtype=dt)
        W = torch.randn(D, D, device="cuda", dtype=dt)
        Wcat = torch.randn(D, 4 * D, device="cuda", dtype=dt)
        DL = torch.randn(M, 4 * D, device="cuda", dtype=dt)
        lb = torch.randn(1, D, L, L, device="cuda", dtype=dt)
        rb = torch.randn(1, D, L, L, device="cuda", dtype=dt)
        dtri = torch.randn(1, D, L, L, device="cuda", dtype=dt)
        x4 = torch.randn(1, L, L, D, device="cuda", dtype=dt)
        g4 = torch.randn(1, L, L, D, device="cuda", dtype=dt)

        print(f"\n=== L={L} (M={M}) ms ===", flush=True)
        print(f"  recompute pg  = x@Wcat (M,4D)        : {bench(lambda: x @ Wcat):.3f}", flush=True)
        print(f"  input-grad    = DL@Wcat.t (M,D)      : {bench(lambda: DL @ Wcat.t()):.3f}", flush=True)
        print(f"  wgrad x4      = x.t@DL (D,4D)        : {bench(lambda: x.t() @ DL):.3f}", flush=True)
        print(f"  bmm bwd d_left= dtri@rb              : {bench(lambda: torch.einsum('bdij,bdjk->bdik', dtri, rb)):.3f}", flush=True)
        print(f"  bmm bwd d_right=dtriT@lb             : {bench(lambda: torch.einsum('bdij,bdik->bdjk', dtri, lb)):.3f}", flush=True)
        print(f"  einsum fwd    = lb@rbT (tri)         : {bench(lambda: torch.einsum('bdik,bdjk->bdij', lb, rb)):.3f}", flush=True)

        def ln_bwd():
            xr = x4.float().requires_grad_(True)
            y = F.layer_norm(xr, (D,), None, None, 1e-5)
            return torch.autograd.grad(y, xr, g4.float())[0]
        print(f"  LN bwd (autograd, fp32 over D)       : {bench(ln_bwd, comp=False):.3f}", flush=True)

        def ln_bwd_bf():
            xr = x4.clone().requires_grad_(True)
            y = F.layer_norm(xr, (D,), None, None, 1e-5)
            return torch.autograd.grad(y, xr, g4)[0]
        print(f"  LN bwd (autograd, bf16 over D)       : {bench(ln_bwd_bf):.3f}", flush=True)

        # gated EW prologue (build DL from d_left,d_right + x@Wcat sigmoids)
        def gated_ew():
            pg = x @ Wcat
            pL, gLl = pg[:, :D], pg[:, D:2 * D]
            gL = torch.sigmoid(gLl)
            dL = dtri[:, :D].permute(0, 2, 3, 1).reshape(M, D)
            return (dL * gL), ((dL * pL) * gL * (1 - gL))
        print(f"  gated EW prologue (fused-able)       : {bench(gated_ew):.3f}", flush=True)


if __name__ == "__main__":
    main()
