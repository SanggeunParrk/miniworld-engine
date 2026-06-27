"""Verify GateElem forward + backward (triton) vs fp32 torch autograd. COMPUTE NODE only.

GateElem: y = proj ⊙ sigmoid(x_n @ Wg). Checks forward y and grads dx_n, d_proj, dWg
against autograd of the same math (bf16 tolerance, cos > 0.99).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_here = _Path(__file__).resolve().parent
_sys.path[:] = [p for p in _sys.path if _Path(p).resolve() != _here]
_src_root = _here
while _src_root.name != "src" and _src_root.parent != _src_root:
    _src_root = _src_root.parent
if str(_src_root) not in _sys.path:
    _sys.path.insert(0, str(_src_root))

import torch

from miniworld_kernels.kernels.trimul_inproj.triton.gate_elem import GateElem, gate_elem_triton


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def main():
    import triton
    assert torch.cuda.is_available()
    print(f"GateElem fwd+bwd on {torch.cuda.get_device_name(0)}", flush=True)
    torch.manual_seed(0)
    L = 512
    M = L * L
    ok_all = True
    for D in (128, 256, 512):       # the D=512 case used to OutOfResources (full-N tl.dot)
        K = N = D
        x_n = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
        proj = torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 0.5
        Wg = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * K**-0.5
        dy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

        xr = x_n.float().clone().requires_grad_(True)
        pr = proj.float().clone().requires_grad_(True)
        wr = Wg.float().clone().requires_grad_(True)
        yr = pr * torch.sigmoid(xr @ wr)
        gx, gp, gw = torch.autograd.grad(yr, (xr, pr, wr), dy.float())

        xo = x_n.clone().requires_grad_(True)
        po = proj.clone().requires_grad_(True)
        wo = Wg.clone().requires_grad_(True)
        try:
            yo = GateElem.apply(xo, po, wo)
            yo.backward(dy)
        except Exception as e:  # noqa: BLE001
            print(f"  D={D}: FAIL {type(e).__name__}: {str(e)[:70]}", flush=True)
            ok_all = False
            continue
        fwd = cos(yr, yo)
        cdx, cdp, cdw = cos(gx, xo.grad), cos(gp, po.grad), cos(gw, wo.grad)
        ok = min(fwd, cdx, cdp, cdw) > 0.99
        ok_all &= ok

        def fb():
            xo.grad = po.grad = wo.grad = None
            GateElem.apply(xo, po, wo).backward(dy)
        t = triton.testing.do_bench(fb, warmup=10, rep=50, return_mode="median")
        print(f"  D={D} L={L}: fwd={fwd:.5f} dx={cdx:.5f} d_proj={cdp:.5f} dWg={cdw:.5f} "
              f"| fwd+bwd={t:.3f}ms -> {'PASS' if ok else 'FAIL'}", flush=True)

    print(f"-> {'ALL PASS' if ok_all else 'FAIL'} (bf16, cos>0.99)", flush=True)


if __name__ == "__main__":
    main()
