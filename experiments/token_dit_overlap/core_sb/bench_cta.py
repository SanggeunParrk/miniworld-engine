"""Cluster (num_ctas) over the sample dimension: do the same-bias programs share the bias tile?"""
import os
import statistics
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tdit.attn import bias_descriptor  # noqa: E402
import attn_ab  # noqa: E402

S = int(sys.argv[1]) if len(sys.argv) > 1 else 4
H, DS, NB, dev, bf = 16, 768, 24, "cuda", torch.bfloat16
D = DS // H


def t_us(fn, reps=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(reps):
            fn()
    out = []
    for _ in range(7):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1e3 / reps)
    return statistics.median(out)


for L in (384, 768):
    qkvg = torch.randn(S * L, 4 * DS, device=dev, dtype=bf) * DS ** -0.5
    q4, k4, v4, g4 = (qkvg.view(S, L, 4 * DS)[..., i * DS:(i + 1) * DS].unflatten(-1, (H, D)) for i in range(4))
    bias = torch.randn(NB * H, L, L, device=dev, dtype=bf) * 0.3
    bdesc = bias_descriptor(bias)
    base = t_us(lambda: attn_ab.attention_ab(q4, k4, v4, g4, bdesc, 0, "tf32"))
    nb = t_us(lambda: attn_ab.attention_ab(q4, k4, v4, g4, bdesc, 0, "tf32", no_bias=True))
    print(f"S={S} L{L} num_ctas={attn_ab.CTAS}: core {base:5.1f} us   no bias {nb:5.1f}   bias {base - nb:5.1f} us", flush=True)
