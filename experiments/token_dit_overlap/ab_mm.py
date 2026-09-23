"""A/B the GEMM picker against cuBLAS in one process, interleaved, so node contention cancels out."""
import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "token_dit_fused"))
from tdit import FusedTokenDiT                                    # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--rounds", type=int, default=4)
a = p.parse_args()
L, S, NB, dev, bf = a.length, 5, 24, "cuda", torch.bfloat16
DS, DC, DP, H = 768, 384, 128, 16
from miniworld_engine.modules.dit import DiTBlock                     # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType   # noqa: E402

torch.manual_seed(0)
blocks = torch.nn.ModuleList(DiTBlock(DS, DC, DP, H, n=2, implementation=ImplementationType.PYTORCH)
                             for _ in range(NB)).to(dev)
with torch.no_grad():
    for prm in blocks.parameters():
        if prm.ndim == 2:
            prm.normal_(std=prm.shape[1] ** -0.5)
        elif prm.numel() > 1:
            prm.add_(torch.randn_like(prm) * 0.1)
blocks = blocks.to(bf).eval()
f = FusedTokenDiT(blocks, dtype=bf)
single = torch.randn(S, 1, L, DS, device=dev, dtype=bf)
cond = torch.randn(1, 1, L, DC, device=dev, dtype=bf).expand(S, 1, L, DC).contiguous()
bias = f.hoist(torch.randn(1, L, L, DP, device=dev, dtype=bf))


def graph_time(reps=5):
    fn = lambda: f.step(single, cond, bias)                        # noqa: E731
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(7):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(reps):
            g.replay()
        en.record(); torch.cuda.synchronize()
        out.append(st.elapsed_time(en) * 1e3 / reps)
    return statistics.median(out) / NB


with torch.no_grad():
    f.step(single, cond, bias)                                     # let the picker measure once
    picked = dict(f._mm_cfg)
    cublas = {k: "cublas" for k in picked}
    print(f"L={L}: picked configs " + ", ".join(f"{k[1]}x{k[2]}:{v}" for k, v in picked.items()), flush=True)
    runs = {"cuBLAS": [], "picked": []}
    for r in range(a.rounds):
        for name, cfg in (("cuBLAS", cublas), ("picked", picked)):
            f._mm_cfg = dict(cfg)
            runs[name].append(graph_time())
    for name, v in runs.items():
        print(f"  {name:<8s} {statistics.median(v):6.2f} us/block   (runs {' '.join(f'{x:.1f}' for x in v)})", flush=True)
    d = statistics.median(runs["cuBLAS"]) - statistics.median(runs["picked"])
    print(f"  picker saves {d:+.2f} us/block ({100 * d / statistics.median(runs['cuBLAS']):+.1f} %)", flush=True)
