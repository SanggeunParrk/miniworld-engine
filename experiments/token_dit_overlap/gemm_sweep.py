"""The block's four GEMMs: cuBLAS against quack tile / cluster configs, including cluster_N (A multicast) which the
packaged v7 never tries -- its _quack_mm passes cluster_N = 1. These shapes read A once per column tile and W once per
row tile, so at BN = 192 a CTA pulls ~491 KB per K sweep and the one-wave grid asks ~9 TB/s of L2. Multicasting A over
the cluster's N direction (and W over its M direction) is what cuts that.
"""
import argparse
import itertools
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "token_dit_fused"))
from quack.gemm_act import gemm_act  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--samples", type=int, default=5)
p.add_argument("--wide", action="store_true", help="also sweep tile_M")
a = p.parse_args()
dev, bf = "cuda", torch.bfloat16
M = a.samples * a.length
D = 768


def time_us(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(reps):
            fn()
    out = []
    for _ in range(7):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record(); g.replay(); en.record(); torch.cuda.synchronize()
        out.append(st.elapsed_time(en) * 1e3 / reps)
    return statistics.median(out)


# name, N, K, bias, activation ("swiglu" halves the output columns)
SHAPES = [("qkvg", 4 * D, D, True, None), ("Wo", D, D, False, None),
          ("expand+swiglu", 4 * D, D, False, "swiglu"), ("squeeze", D, 2 * D, False, None)]
TILES = [128, 192, 256]
TILES_M = [128]
CLUSTERS = [(1, 1), (2, 1), (1, 2), (2, 2), (1, 4), (2, 4), (4, 1), (4, 2)]
if a.wide:
    TILES_M = [64, 128, 256]
    CLUSTERS = [(1, 1), (2, 1), (1, 2), (2, 2), (1, 4), (2, 4)]

print(f"L={a.length} S={a.samples} M={M} bf16; us and TFLOP/s, best first", flush=True)
for name, N, K, has_bias, act in SHAPES:
    A = torch.randn(M, K, device=dev, dtype=bf) * K ** -0.5
    W = (torch.randn(N, K, device=dev) * K ** -0.5).to(bf)
    bias = torch.randn(N, device=dev, dtype=bf) if has_bias else None
    No = N // 2 if act else N
    out = torch.empty(M, No, device=dev, dtype=bf)
    flops = 2 * M * N * K
    rows = []
    if act is None:
        ref = torch.addmm(bias, A, W.t()) if has_bias else torch.mm(A, W.t())
        t = time_us(lambda: torch.addmm(bias, A, W.t(), out=out)) if has_bias else time_us(lambda: torch.mm(A, W.t(), out=out))
        rows.append((t, "cuBLAS"))
    else:
        h = torch.mm(A, W.t())
        ref = torch.nn.functional.silu(h[:, 0::2]) * h[:, 1::2]        # quack's gate/up interleave
    for (tm, tn, cm, cn) in [(m, t, c[0], c[1]) for m in TILES_M for t in TILES for c in CLUSTERS]:
        if (N // tn) % cn or (M // tm) % cm:
            continue
        for pp in ((True, False) if tn <= 208 else (False,)):
            try:
                fn = lambda: gemm_act(A[None], W[None], None, None, out[None], None, act, tm, tn, cm, cn,
                                      pingpong=pp, rowvec_bias=None if bias is None else bias[None])
                fn(); torch.cuda.synchronize()
                err = float((out.float() - ref.float()).norm() / ref.float().norm())
                if err > 2e-2:
                    continue
                rows.append((time_us(fn), f"quack tM{tm} tN{tn} cl{cm}x{cn} pp{int(pp)}"))
            except Exception as e:  # noqa: BLE001
                pass
    rows.sort()
    print(f"\n{name}: M{M} N{N} K{K}{' +bias' if has_bias else ''}{' swiglu' if act else ''}", flush=True)
    for t, nm in rows[:6]:
        print(f"  {nm:<28s} {t:7.2f} us  {flops / t / 1e6:6.0f} TFLOP/s", flush=True)
