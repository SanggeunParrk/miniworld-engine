"""Does compile-RO's per-call overhead amortize across a STACK of layers?

Isolated single-op compile-RO carries a per-call cudagraph_trees overhead that
manual replay avoids. But real models (AlphaFold-style) stack many trimul blocks
and compile the whole stack -> one cudagraph tree, replayed once per step. This
measures per-layer time for a K-deep stack: eager vs compile-RO vs manual graph.
If compile-RO's per-layer time falls toward manual as K grows, torch.compile
alone is sufficient for the real (stacked) case. Fwd-only, B=1, D=128, bf16.
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

from miniworld_kernels.kernels.trimul_inproj.cute import _bdll_patch
from miniworld_kernels.kernels.trimul_inproj.cute.launch import trimul_inproj_cute_forward
from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication.module import (
    TriangleMultiplication,
    _load_cute_fns,
)


def _bench(fn, *, warmup=20, rep=60):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def _make_graphed(fn, pair, warmup=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn(pair)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(pair)
    return g


def main():
    assert torch.cuda.is_available()
    print(f"stack amortization test on {torch.cuda.get_device_name(0)}", flush=True)
    _bdll_patch.apply()
    dtype, D = torch.bfloat16, 128
    _t1, tm2_cute_forward, _f, layer_norm_transpose = _load_cute_fns()

    mod = TriangleMultiplication(d_pair=D, implementation=ImplementationType.CUTE).cuda()
    torch.manual_seed(0)
    for lin in (mod.to_left, mod.to_left_gate, mod.to_right, mod.to_right_gate,
                mod.to_gate, mod.to_out):
        nn.init.normal_(lin.weight, std=D**-0.5)
    mod = mod.to(dtype)

    def _ln_in(pair):
        b, l1, l2, d = pair.shape
        o = layer_norm_transpose(pair.reshape(b * l1 * l2, d), mod.ln_pair.weight,
                                 mod.ln_pair.bias, eps=mod.ln_pair.eps, layout="nd->nd")
        return (o[0] if isinstance(o, tuple) else o).view(b, l1, l2, d)

    def _ln_out(tri, b, l1, l2, d):
        tri_dbn = tri.permute(1, 0, 2, 3).reshape(d, b, l1 * l2)
        o2 = layer_norm_transpose(tri_dbn, mod.ln_out.weight, mod.ln_out.bias,
                                  eps=mod.ln_out.eps, layout="dbn->bnd")
        return (o2[0] if isinstance(o2, tuple) else o2).view(b, l1, l2, d)

    def v2_one(pair):
        b, l1, l2, d = pair.shape
        x_normed = _ln_in(pair)
        left, right, _ = trimul_inproj_cute_forward(
            x_normed, mod.to_left.weight.T, mod.to_left_gate.weight.T,
            mod.to_right.weight.T, mod.to_right_gate.weight.T, None,
            bdll_direct=True, compute_gate=False)
        tri = torch.einsum("bdik,bdjk->bdij", left, right)
        out_normed = _ln_out(tri, b, l1, l2, d)
        return tm2_cute_forward(x_normed, out_normed, mod.to_gate.weight, mod.to_out.weight)

    def make_stack(K):
        def stack(pair):
            for _ in range(K):
                pair = v2_one(pair)
            return pair
        return stack

    print(f"\n{'L':>5} {'K':>3} | {'eager/lyr':>9} | {'compile/lyr':>11} | "
          f"{'manual/lyr':>10} | {'comp/man':>8}")
    print("-" * 60)
    for L in (512, 1024):
        torch.manual_seed(L)
        pair = torch.randn(1, L, L, D, device="cuda", dtype=dtype)
        for K in (1, 4, 8):
            stack = make_stack(K)
            with torch.no_grad():
                t_e = _bench(lambda: stack(pair)) / K
                torch._dynamo.reset()
                cfn = torch.compile(stack, mode="reduce-overhead")
                for _ in range(6):
                    cfn(pair)
                t_c = _bench(lambda: cfn(pair)) / K
                g = _make_graphed(stack, pair)
                t_g = _bench(lambda: g.replay()) / K
            print(f"{L:>5} {K:>3} | {t_e:>9.3f} | {t_c:>11.3f} | {t_g:>10.3f} | "
                  f"{t_c / t_g:>7.2f}x", flush=True)


if __name__ == "__main__":
    main()
