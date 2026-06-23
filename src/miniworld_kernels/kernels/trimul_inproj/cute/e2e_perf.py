"""End-to-end trimul: current cute path vs the new (trimul_inproj) pipeline.

Kernel-development diagnostic. Compares, on identical weights:

  current  = TriangleMultiplication._forward_cute
             LN_in -> tm1 (2 launches) -> bmm -> LN_out(dbn->bnd) -> tm2(gate+proj+mul)

  new      = LN_in -> trimul_inproj (left+right 1 launch + gate precomputed, bdll)
             -> bmm -> LN_out(dbn->bnd) -> (out_normed @ W_out) * gate

Both need the bdll patch (current's tm1 + new's trimul both use M-major postact);
we apply our in-repo shim once up front.

The new path's back half is NOT yet a fused kernel (torch proj + torch mul) — this
measures whether the front fusion + precomputed gate already helps, and isolates
how much the back-half fusion (fold the mul into a layernorm_linear epilogue) is
still worth. Forward-only, B=1, D=128, bf16. Run on a COMPUTE NODE (srun).
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
import torch.nn as nn
import triton

from miniworld_kernels.kernels.layernorm_linear.cute.gemm_layernorm_linear_fused import (
    layernorm_linear_cute_fused,
)
from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.launch import trimul_inproj_cute_forward
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.module import (
    TriangleMultiplication,
    _load_cute_fns,
)


def _bench(fn, *, warmup=25, rep=100):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def main():
    assert torch.cuda.is_available()
    print(f"trimul e2e: current cute vs new pipeline on {torch.cuda.get_device_name(0)}")
    _bdll_patch.apply()  # both paths use M-major (bdll) gated postact

    dtype = torch.bfloat16
    D = 128
    _, tm2_cute_forward, _fused_ln_mask, layer_norm_transpose = _load_cute_fns()

    mod = TriangleMultiplication(d_pair=D, implementation=ImplementationType.CUTE).cuda()
    # default init zeroes several gates/projections -> vacuous compare; randomize.
    torch.manual_seed(0)
    for lin in (mod.to_left, mod.to_left_gate, mod.to_right, mod.to_right_gate,
                mod.to_gate, mod.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)
    mod = mod.to(dtype)

    def new_path(pair):
        b, l1, l2, d = pair.shape
        o = layer_norm_transpose(
            pair.reshape(b * l1 * l2, d), mod.ln_pair.weight, mod.ln_pair.bias,
            eps=mod.ln_pair.eps, layout="nd->nd",
        )
        x_normed = (o[0] if isinstance(o, tuple) else o).view(b, l1, l2, d)
        left, right, gate = trimul_inproj_cute_forward(
            x_normed, mod.to_left.weight.T, mod.to_left_gate.weight.T,
            mod.to_right.weight.T, mod.to_right_gate.weight.T, mod.to_gate.weight.T,
            bdll_direct=True,
        )
        tri = torch.einsum("bdik,bdjk->bdij", left, right)  # [B,D,L,L]
        tri_dbn = tri.permute(1, 0, 2, 3).reshape(d, b, l1 * l2)
        o2 = layer_norm_transpose(
            tri_dbn, mod.ln_out.weight, mod.ln_out.bias,
            eps=mod.ln_out.eps, layout="dbn->bnd",
        )
        out_normed = (o2[0] if isinstance(o2, tuple) else o2).view(b, l1, l2, d)
        return (out_normed @ mod.to_out.weight.T) * gate

    def _ln_in(pair):
        b, l1, l2, d = pair.shape
        o = layer_norm_transpose(
            pair.reshape(b * l1 * l2, d), mod.ln_pair.weight, mod.ln_pair.bias,
            eps=mod.ln_pair.eps, layout="nd->nd",
        )
        return (o[0] if isinstance(o, tuple) else o).view(b, l1, l2, d)

    def _ln_out(tri, b, l1, l2, d):
        tri_dbn = tri.permute(1, 0, 2, 3).reshape(d, b, l1 * l2)
        o2 = layer_norm_transpose(
            tri_dbn, mod.ln_out.weight, mod.ln_out.bias,
            eps=mod.ln_out.eps, layout="dbn->bnd",
        )
        return (o2[0] if isinstance(o2, tuple) else o2).view(b, l1, l2, d)

    def new_v2_path(pair):
        # Only the FRONT changes vs current: left+right fused into one launch
        # (gate stays inside tm2, NOT precomputed -> no extra HBM round trip).
        b, l1, l2, d = pair.shape
        x_normed = _ln_in(pair)
        left, right, _ = trimul_inproj_cute_forward(
            x_normed, mod.to_left.weight.T, mod.to_left_gate.weight.T,
            mod.to_right.weight.T, mod.to_right_gate.weight.T, None,
            bdll_direct=True, compute_gate=False,
        )
        tri = torch.einsum("bdik,bdjk->bdij", left, right)
        out_normed = _ln_out(tri, b, l1, l2, d)
        # tm2 wants nn.Linear-form weights (N, K) -> pass .weight directly.
        return tm2_cute_forward(x_normed, out_normed, mod.to_gate.weight, mod.to_out.weight)

    def new_v3_path(pair):
        # User's plan: front lr+gate fused; back = EXISTING cute layernorm_linear
        # (LN_out + proj) ; final gate mul left as torch (NOT fused -> ~14T, same
        # as current in theory; measure actual). tri must be blld for the LN over D.
        b, l1, l2, d = pair.shape
        x_normed = _ln_in(pair)
        left, right, gate = trimul_inproj_cute_forward(
            x_normed, mod.to_left.weight.T, mod.to_left_gate.weight.T,
            mod.to_right.weight.T, mod.to_right_gate.weight.T, mod.to_gate.weight.T,
            bdll_direct=True, compute_gate=True,
        )
        # einsum->bijd returns a permuted *view* (d not contiguous); reshape would
        # give an m-major (1, L*L) view, but layernorm_linear needs k-major
        # contiguous (M,K). .contiguous() forces the bdll->blld transpose here
        # (the cost current's layer_norm_transpose(dbn->bnd) fuses away).
        tri = torch.einsum("bdik,bdjk->bijd", left, right).contiguous()  # [B,L,L,D]
        proj = layernorm_linear_cute_fused(
            tri.reshape(b * l1 * l2, d), mod.ln_out.weight, mod.ln_out.bias,
            mod.to_out.weight, None, eps=mod.ln_out.eps,
        )
        return proj.view(b, l1, l2, d) * gate  # final gate mul: torch

    print(f"\n{'L':>5} | {'current(ms)':>11} | {'v2 tm2(ms)':>10} | {'v3 lnl(ms)':>10} | "
          f"{'v2/cur':>6} | {'v3/cur':>6} | {'v2 diff':>9} | {'v3 diff':>9}")
    print("-" * 92)
    for L in (384, 512, 768, 1024):
        torch.manual_seed(L)
        pair = torch.randn(1, L, L, D, device="cuda", dtype=dtype)
        with torch.no_grad():
            y_cur = mod(pair)
            y_v2 = new_v2_path(pair)
            y_v3 = new_v3_path(pair)
            diff_v2 = (y_cur.float() - y_v2.float()).abs().max().item()
            diff_v3 = (y_cur.float() - y_v3.float()).abs().max().item()
            t_cur = _bench(lambda: mod(pair))
            t_v2 = _bench(lambda: new_v2_path(pair))
            t_v3 = _bench(lambda: new_v3_path(pair))
        print(f"{L:>5} | {t_cur:>11.3f} | {t_v2:>10.3f} | {t_v3:>10.3f} | "
              f"{t_cur / t_v2:>5.2f}x | {t_cur / t_v3:>5.2f}x | "
              f"{diff_v2:>9.3e} | {diff_v3:>9.3e}")


if __name__ == "__main__":
    main()
