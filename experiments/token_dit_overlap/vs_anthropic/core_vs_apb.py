"""Inference attention core head-to-head: our CUDA core (attn_core.cu) vs Anthropic apb_views, same inputs, two timings:
graph = CUDA-graph replay median, inputs hot in L2 (how the 2026-09-22 comparison timed Anthropic);
cold  = triton do_bench, L2 flushed between calls (how our core was timed)."""
import statistics
import sys
from pathlib import Path
import torch
import triton
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "token_dit_fused"))
from tdit.cuda_core import attn_core  # noqa: E402
from opt_core.kernels.apb.fpf_apb.apb_triton import apb_views  # noqa: E402

dev, bf, H, D, DS = "cuda", torch.bfloat16, 16, 48, 768


def graph_us(fn, reps=50, rounds=15):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(rounds):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1000 / reps)
    return statistics.median(out)


def cold_us(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median") * 1e3


for S in (5, 48):
    for L in (384, 768):
        torch.manual_seed(0)
        qkvg = (torch.randn(S * L, 4 * DS, device=dev) * 0.3).to(bf)
        bias = (torch.randn(H, L, L, device=dev) * 0.5).to(bf)            # one block's bias, head-major
        work = qkvg.clone()
        ours = lambda: attn_core(work, bias, 0, S, H)                     # noqa: E731  (gated, written over q)
        v4 = qkvg.view(S, L, 4, H, D)
        q, k, v = v4[:, :, 0], v4[:, :, 1], v4[:, :, 2]                   # strided views, as the block holds them
        anth = lambda: apb_views(q, k, v, bias, None, scale=D ** -0.5)    # noqa: E731  (no gate)
        with torch.no_grad():
            r = {n: (graph_us(f), cold_us(f)) for n, f in (("ours", ours), ("anthropic", anth))}
        print(f"cmp: S{S:2d} L{L}  ours graph {r['ours'][0]:7.1f} cold {r['ours'][1]:7.1f} | "
              f"anthropic graph {r['anthropic'][0]:7.1f} cold {r['anthropic'][1]:7.1f} us", flush=True)
