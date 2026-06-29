"""Verify ALL gradients of single-dir v6 (bf16) vs fp32 torch reference — not just dx,
but every weight grad (dWL/dWLg/dWR/dWRg/dWg/dWp/dLN_in/dLN_out). COMPUTE NODE only."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_here = _Path(__file__).resolve().parent
_sys.path[:] = [p for p in _sys.path if _Path(p).resolve() != _here]
_src = _here
while _src.name != "src" and _src.parent != _src:
    _src = _src.parent
if str(_src) not in _sys.path:
    _sys.path.insert(0, str(_src))

import copy

import torch
import torch.nn as nn

from miniworld_kernels.kernels.trimul_inproj.cute.v6_training import V6TriMul
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.module import TriangleMultiplication

D = 128


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def main():
    assert torch.cuda.is_available()
    print(f"v6 ALL-grad verify on {torch.cuda.get_device_name(0)}", flush=True)
    torch.manual_seed(0)
    base = TriangleMultiplication(d_pair=D, outgoing=True,
                                  implementation=ImplementationType.PYTORCH).cuda()
    for lin in (base.to_left, base.to_left_gate, base.to_right, base.to_right_gate,
                base.to_gate, base.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)
    base_fp = copy.deepcopy(base).float()
    v6 = V6TriMul(base.to(torch.bfloat16), direction="out")

    ok_all = True
    for L in (384, 768):
        pair = torch.randn(1, L, L, D, device="cuda")
        dy = torch.randn_like(pair)

        # fp32 reference: full backward, populate param grads
        base_fp.zero_grad(set_to_none=True)
        xr = pair.float().clone().requires_grad_(True)
        base_fp(xr).backward(dy)

        # v6 (bf16): full backward
        v6.zero_grad(set_to_none=True)
        xo = pair.to(torch.bfloat16).clone().requires_grad_(True)
        yo = v6(xo)
        yo.backward(dy.to(torch.bfloat16))

        # grad map: v6 param  ->  reference grad (x@W-form weights need .t(); Wp_nn is (N,K))
        checks = {
            "dx (input)": (xo.grad, xr.grad),
            "dWL": (v6.WL.grad, base_fp.to_left.weight.grad.t()),
            "dWLg": (v6.WLg.grad, base_fp.to_left_gate.weight.grad.t()),
            "dWR": (v6.WR.grad, base_fp.to_right.weight.grad.t()),
            "dWRg": (v6.WRg.grad, base_fp.to_right_gate.weight.grad.t()),
            "dWg(gate)": (v6.Wg.grad, base_fp.to_gate.weight.grad.t()),
            "dWp(out)": (v6.Wp_nn.grad, base_fp.to_out.weight.grad),
            "dLN_in_w": (v6.ln_in_w.grad, base_fp.ln_pair.weight.grad),
            "dLN_in_b": (v6.ln_in_b.grad, base_fp.ln_pair.bias.grad),
            "dLN_out_w": (v6.ln_out_w.grad, base_fp.ln_out.weight.grad),
            "dLN_out_b": (v6.ln_out_b.grad, base_fp.ln_out.bias.grad),
        }
        print(f"\n--- L={L} (cos vs fp32 ref) ---", flush=True)
        for name, (g_ours, g_ref) in checks.items():
            if g_ours is None or g_ref is None:
                print(f"  {name:12}: MISSING (ours={g_ours is not None} ref={g_ref is not None})",
                      flush=True)
                ok_all = False
                continue
            c = cos(g_ours, g_ref)
            ok = c > 0.99
            ok_all &= ok
            print(f"  {name:12}: cos={c:.5f}  {'PASS' if ok else 'FAIL'}", flush=True)

    print(f"\n-> ALL grads {'PASS' if ok_all else 'FAIL'} (bf16, cos>0.99)", flush=True)


if __name__ == "__main__":
    main()
