"""CUDA-graph vs eager for the trimul forward: current cute path vs v2.

Tests whether capturing the whole forward into a CUDA graph collapses the
launch/latency floor that dominates at small L (where 384 and 512 timed ~equal).
Forward-only. B=1, D=128, bf16. Run on a COMPUTE NODE (srun).
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


def _bench(fn, *, warmup=25, rep=100):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def _make_graphed(fn, pair, warmup=5):
    """Capture fn(pair) into a CUDA graph; return a replay thunk (or None)."""
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
    print(f"trimul CUDA-graph vs eager on {torch.cuda.get_device_name(0)}")
    _bdll_patch.apply()

    dtype = torch.bfloat16
    D = 128
    _, tm2_cute_forward, _f, layer_norm_transpose = _load_cute_fns()

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

    def current_fn(pair):
        return mod(pair)

    def v2_fn(pair):
        b, l1, l2, d = pair.shape
        x_normed = _ln_in(pair)
        left, right, _ = trimul_inproj_cute_forward(
            x_normed, mod.to_left.weight.T, mod.to_left_gate.weight.T,
            mod.to_right.weight.T, mod.to_right_gate.weight.T, None,
            bdll_direct=True, compute_gate=False)
        tri = torch.einsum("bdik,bdjk->bdij", left, right)
        tri_dbn = tri.permute(1, 0, 2, 3).reshape(d, b, l1 * l2)
        o2 = layer_norm_transpose(tri_dbn, mod.ln_out.weight, mod.ln_out.bias,
                                  eps=mod.ln_out.eps, layout="dbn->bnd")
        out_normed = (o2[0] if isinstance(o2, tuple) else o2).view(b, l1, l2, d)
        return tm2_cute_forward(x_normed, out_normed, mod.to_gate.weight, mod.to_out.weight)

    variants = (("current", current_fn), ("v2", v2_fn))
    print(f"\n{'L':>5} | {'variant':>8} | {'eager(ms)':>10} | {'graph(ms)':>10} | "
          f"{'eager/graph':>11}")
    print("-" * 58)
    for L in (384, 512, 768, 1024):
        torch.manual_seed(L)
        pair = torch.randn(1, L, L, D, device="cuda", dtype=dtype)
        for name, fn in variants:
            with torch.no_grad():
                t_eager = _bench(lambda: fn(pair))
                try:
                    g = _make_graphed(fn, pair)
                    t_graph = _bench(lambda: g.replay())
                    speedup = f"{t_eager / t_graph:.2f}x"
                except Exception as e:  # noqa: BLE001
                    t_graph, speedup = float("nan"), f"capture FAIL: {type(e).__name__}"
            print(f"{L:>5} | {name:>8} | {t_eager:>10.3f} | {t_graph:>10.3f} | {speedup:>11}")


if __name__ == "__main__":
    main()
