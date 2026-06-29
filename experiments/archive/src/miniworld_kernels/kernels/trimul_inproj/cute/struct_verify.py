"""Verify the inference/training split: inference vs pytorch ref (mask), and the
training autograd.Function fwd + param grads vs a torch autograd reference."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_src_root = _Path(__file__).resolve()
while _src_root.name != "src" and _src_root.parent != _src_root:
    _src_root = _src_root.parent
if str(_src_root) not in _sys.path:
    _sys.path.insert(0, str(_src_root))

import torch
import torch.nn as nn

from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.training import TriMulInproj
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.module import TriangleMultiplication

D = 128
EPS = 1e-5


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def main():
    assert torch.cuda.is_available()
    print(f"struct verify on {torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    torch.manual_seed(0)
    dt = torch.bfloat16
    L = 256

    base = TriangleMultiplication(d_pair=D, implementation=ImplementationType.PYTORCH).cuda()
    for lin in (base.to_left, base.to_left_gate, base.to_right, base.to_right_gate,
                base.to_gate, base.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)
    ref = base.cuda()                       # fp32 pytorch reference
    ours = TriMulInproj(base).to(dt).cuda()

    pair = torch.randn(1, L, L, D, device="cuda")
    mask = torch.rand(1, L, device="cuda") > 0.2

    # forward parity (bf16 ours vs fp32 ref), with mask
    y_ref = ref(pair, mask)
    y_inf = ours.inference(pair.to(dt), mask)
    y_tr = ours(pair.to(dt), mask)
    print(f"  cos inference vs ref(mask) = {cos(y_inf, y_ref):.5f}", flush=True)
    print(f"  cos training  vs ref(mask) = {cos(y_tr, y_ref):.5f}", flush=True)

    # backward parity: grads of params, ours (bf16 manual bwd) vs ref (fp32 autograd)
    ref2 = TriangleMultiplication(d_pair=D, implementation=ImplementationType.PYTORCH).cuda()
    ref2.load_state_dict(base.state_dict())
    p = pair.clone().requires_grad_(True)
    y = ref2(p, mask)
    g = torch.randn_like(y)
    y.backward(g)

    pours = pair.to(dt).clone().requires_grad_(True)
    yo = ours(pours, mask)
    yo.backward(g.to(dt))

    pairs = [
        ("dWL", ours.WL.grad, ref2.to_left.weight.grad.t()),
        ("dWLg", ours.WLg.grad, ref2.to_left_gate.weight.grad.t()),
        ("dWp", ours.Wp.grad, ref2.to_out.weight.grad.t()),
        ("dWg", ours.Wg.grad, ref2.to_gate.weight.grad.t()),
        ("dln_in_w", ours.ln_in_w.grad, ref2.ln_pair.weight.grad),
        ("dln_out_w", ours.ln_out_w.grad, ref2.ln_out.weight.grad),
        ("dx", pours.grad, p.grad),
    ]
    for name, go, gr in pairs:
        c = cos(go, gr) if go is not None else float("nan")
        print(f"  cos {name:>10} = {c:.5f}", flush=True)


if __name__ == "__main__":
    main()
