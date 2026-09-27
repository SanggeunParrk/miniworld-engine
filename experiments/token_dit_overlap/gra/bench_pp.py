"""Kernel-level: gemm_resgate_adaln_pp vs torch.mm + resgate_adaln_rows, do_bench (bench.us) and a captured replay."""
import statistics, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import us as t_us  # noqa
from tdit import kernels as K  # noqa
from gra import fused  # noqa
gemm_resgate_adaln_pp = fused()[0]


def graph_us(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=st):
        for _ in range(n): fn()
    best = 1e9
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) * 1e3 / n)
    return best


dev, bf, D = "cuda", torch.bfloat16, 768
for L in (384, 768):
    M = 5 * L
    for Kd, sa, nm in ((768, 3072, "Wo"), (1536, 1536, "squeeze")):
        abuf = torch.randn(M, sa, device=dev, dtype=bf); a = abuf[:, :Kd]
        w = (torch.randn(D, Kd, device=dev) * Kd ** -0.5).to(bf)
        x = torch.randn(M, D, device=dev); y = torch.empty(M, D, device=dev, dtype=bf); xa = torch.empty_like(y)
        g = torch.randn(L, 4, D, device=dev, dtype=bf); gl, ms, mb = g[:, 0], g[:, 1], g[:, 2]
        base = lambda: (torch.mm(a, w.t(), out=y), K.resgate_adaln_rows(x, y, gl, ms, mb, xa, L))
        fused = lambda: gemm_resgate_adaln_pp(a, w, x, gl, ms, mb, xa, L)
        print(f"L{L} {nm:<7s}: mm+rows do_bench {t_us(base):6.1f}  graph {graph_us(base):6.1f}   |   "
              f"pp do_bench {t_us(fused):6.1f}  graph {graph_us(fused):6.1f}  us", flush=True)
