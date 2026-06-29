"""B0 correctness: manual backward vs torch autograd of the SAME torch forward.

Builds a plain-torch trimul forward (autograd differentiable), and the
TriMulManualBwd Function (same forward, hand-written backward). Compares grads
for every input. COMPUTE NODE.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_src_root = _Path(__file__).resolve()
while _src_root.name != "src" and _src_root.parent != _src_root:
    _src_root = _src_root.parent
if str(_src_root) not in _sys.path:
    _sys.path.insert(0, str(_src_root))

import importlib.util as _ilu

import torch

# Load autograd.py directly (avoid kernels/__init__'s triton import chain).
_spec = _ilu.spec_from_file_location(
    "_trimul_autograd", str(_Path(__file__).resolve().parent / "autograd.py"))
_ag = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ag)
TriMulManualBwd, _ln_fwd = _ag.TriMulManualBwd, _ag._ln_fwd


def torch_forward(x, WL, WLg, WR, WRg, Wg, Wp, ln_in_w, ln_in_b, ln_out_w, ln_out_b, eps):
    """Autograd-differentiable torch reference (same math as the Function)."""
    x_n, *_ = _ln_fwd(x, ln_in_w, ln_in_b, eps)
    left = (x_n @ WL) * torch.sigmoid(x_n @ WLg)
    right = (x_n @ WR) * torch.sigmoid(x_n @ WRg)
    left_b = left.permute(0, 3, 1, 2)
    right_b = right.permute(0, 3, 1, 2)
    tri = torch.einsum("bdik,bdjk->bdij", left_b, right_b)
    tri_lld = tri.permute(0, 2, 3, 1)
    out_n, *_ = _ln_fwd(tri_lld, ln_out_w, ln_out_b, eps)
    gate = torch.sigmoid(x_n @ Wg)
    proj = out_n @ Wp
    return proj * gate


def main():
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    dev = "cuda"
    B, L, D, eps = 1, 64, 128, 1e-5
    dt = torch.float64  # tight tolerance oracle

    def mk(*shape):
        return torch.randn(*shape, device=dev, dtype=dt, requires_grad=True)

    x = mk(B, L, L, D)
    WL, WLg, WR, WRg, Wg, Wp = (mk(D, D) * D**-0.5 for _ in range(6))
    for t in (WL, WLg, WR, WRg, Wg, Wp):
        t.retain_grad()
    ln_in_w = torch.ones(D, device=dev, dtype=dt, requires_grad=True)
    ln_in_b = torch.zeros(D, device=dev, dtype=dt, requires_grad=True)
    ln_out_w = (torch.ones(D, device=dev, dtype=dt) + 0.1 * torch.randn(D, device=dev, dtype=dt)).requires_grad_()
    ln_out_b = (0.1 * torch.randn(D, device=dev, dtype=dt)).requires_grad_()

    args = (x, WL, WLg, WR, WRg, Wg, Wp, ln_in_w, ln_in_b, ln_out_w, ln_out_b, eps)
    names = ["x", "WL", "WLg", "WR", "WRg", "Wg", "Wp",
             "ln_in_w", "ln_in_b", "ln_out_w", "ln_out_b"]

    dy = torch.randn(B, L, L, D, device=dev, dtype=dt)

    # forward equivalence
    y_ref = torch_forward(*args)
    y_man = TriMulManualBwd.apply(*args)
    fwd_err = (y_ref - y_man).abs().max().item()
    print(f"forward max_abs(ref vs Function) = {fwd_err:.3e}", flush=True)

    # autograd grads
    g_ref = torch.autograd.grad(y_ref, args[:-1], grad_outputs=dy, retain_graph=False)
    # manual grads
    leaves = [a.detach().clone().requires_grad_(True) for a in args[:-1]] + [eps]
    y_man2 = TriMulManualBwd.apply(*leaves)
    g_man = torch.autograd.grad(y_man2, leaves[:-1], grad_outputs=dy)

    print(f"\n{'param':>10} | {'max_abs':>12} | {'rel':>10}", flush=True)
    print("-" * 40)
    worst = 0.0
    for nm, gr, gm in zip(names, g_ref, g_man):
        ma = (gr - gm).abs().max().item()
        rel = ma / (gr.abs().max().item() + 1e-30)
        worst = max(worst, rel)
        flag = "" if rel < 1e-8 else "  <-- MISMATCH"
        print(f"{nm:>10} | {ma:>12.3e} | {rel:>10.2e}{flag}", flush=True)
    print(f"\nworst rel = {worst:.2e}  ->  {'PASS' if worst < 1e-8 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
