"""Verify B>1 for the remaining TRITON trimul pieces: the single-direction module
(unidirectional path) and the standalone front / back kernels.

  1. Single-direction module (TriangleMultiplication, TRITON vs PYTORCH, shared weights):
     forward + input-grad + param-grad cosine, outgoing & incoming, B in {1,2,4}.
  2. front kernel (trimul_front_triton) vs trimul_inproj_pytorch: left/right/gate for B>1.
  3. back  kernel (trimul_back_triton) vs a pytorch LN_D+@Wp+gate reference for B>1.

Run under `pixi run` on an A100. Exits non-zero on any failure.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from miniworld_kernels.kernels.trimul_inproj.reference import trimul_inproj_pytorch
from miniworld_kernels.kernels.trimul_inproj.triton.back import trimul_back_triton
from miniworld_kernels.kernels.trimul_inproj.triton.front import trimul_front_triton
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.module import (
    TriangleMultiplication,
)

DEVICE, BF16 = "cuda", torch.bfloat16


def _cos(a, b) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return F.cosine_similarity(a, b, dim=0, eps=1e-8).item()


def _run_grad(mod, pair, mask):
    pair = pair.detach().clone().requires_grad_(True)
    y = mod(pair, mask)
    y.float().sum().backward()
    pg = {n: p.grad.detach().clone() for n, p in mod.named_parameters()}
    return y.detach(), pair.grad.detach(), pg


def check_module(outgoing: bool) -> bool:
    d = 128
    ref = TriangleMultiplication(d, outgoing=outgoing, implementation=ImplementationType.PYTORCH)
    tri = TriangleMultiplication(d, outgoing=outgoing, implementation=ImplementationType.TRITON)
    torch.manual_seed(0)
    for p in ref.parameters():
        p.data.normal_(0, 0.5)
    tri.load_state_dict(ref.state_dict())
    ref, tri = ref.to(DEVICE, BF16), tri.to(DEVICE, BF16)
    ok = True
    for L in (128, 256):
        for B in (1, 2, 4):
            torch.manual_seed(10 + B + L)
            pair = torch.randn(B, L, L, d, device=DEVICE, dtype=BF16)
            mask = torch.ones(B, L, dtype=torch.bool, device=DEVICE)
            mask[:, L - 1] = False                       # a dropped residue
            yr, gr, pgr = _run_grad(ref, pair, mask)
            yt, gt, pgt = _run_grad(tri, pair, mask)
            cf, cg = _cos(yt, yr), _cos(gt, gr)
            cp = min(_cos(pgt[n], pgr[n]) for n in pgr if pgr[n].abs().sum() > 0)
            good = cf >= 0.99 and cg >= 0.99 and cp >= 0.99
            ok &= good
            dirn = "out" if outgoing else "in"
            print(f"  [{'OK' if good else 'FAIL'}] uni dir={dirn} L={L} B={B}"
                  f"  fwd={cf:.5f} dx={cg:.5f} dW={cp:.5f}")
    return ok


def check_front() -> bool:
    d, ok = 128, True
    torch.manual_seed(1)
    W = {k: torch.randn(d, d, device=DEVICE, dtype=BF16) * 0.1 for k in "abcde"}
    WL, WLg, WR, WRg, Wg = W.values()
    for L in (128, 256):
        for B in (1, 3):
            x = torch.randn(B, L, L, d, device=DEVICE, dtype=BF16)
            lt, rt, gt = trimul_front_triton(x, WL, WLg, WR, WRg, Wg)
            lr, rr, gr = trimul_inproj_pytorch(x, WL, WLg, WR, WRg, Wg)
            cl, cr, cg = _cos(lt, lr), _cos(rt, rr), _cos(gt, gr)
            good = min(cl, cr, cg) >= 0.99
            ok &= good
            print(f"  [{'OK' if good else 'FAIL'}] front L={L} B={B}"
                  f"  left={cl:.5f} right={cr:.5f} gate={cg:.5f}")
    return ok


def check_back() -> bool:
    d, ok, eps = 128, True, 1e-5
    torch.manual_seed(2)
    Wp = torch.randn(d, d, device=DEVICE, dtype=BF16) * 0.1
    Wg = torch.randn(d, d, device=DEVICE, dtype=BF16) * 0.1
    ln_w = torch.randn(d, device=DEVICE, dtype=BF16)
    ln_b = torch.randn(d, device=DEVICE, dtype=BF16)
    for L in (128, 256):
        for B in (1, 3):
            tri = torch.randn(B, d, L, L, device=DEVICE, dtype=BF16)   # bdll
            x_n = torch.randn(B, L, L, d, device=DEVICE, dtype=BF16)
            y = trimul_back_triton(tri, x_n, Wp, Wg, ln_w, ln_b, eps)
            # reference: proj = LN_D(tri_blld) @ Wp ; gate = sigmoid(x_n @ Wg) ; y = gate*proj
            tri_blld = tri.permute(0, 2, 3, 1).float()                 # (B,L,L,d)
            ln = F.layer_norm(tri_blld, (d,), ln_w.float(), ln_b.float(), eps)
            proj = ln @ Wp.float()
            gate = torch.sigmoid(x_n.float() @ Wg.float())
            yr = (gate * proj)
            c = _cos(y, yr)
            good = c >= 0.99
            ok &= good
            print(f"  [{'OK' if good else 'FAIL'}] back  L={L} B={B}  y={c:.5f}")
    return ok


def main() -> int:
    assert torch.cuda.is_available()
    print(f"device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
    ok = True
    print("== single-direction module (unidirectional triton) ==")
    ok &= check_module(outgoing=True)
    ok &= check_module(outgoing=False)
    print("== front kernel ==")
    ok &= check_front()
    print("== back kernel ==")
    ok &= check_back()
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
