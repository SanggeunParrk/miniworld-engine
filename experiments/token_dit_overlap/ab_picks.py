"""Step A/B: the GEMM picks as raced (cuBLAS included) against quack-only picks, in one process, interleaved.

cuBLAS is launched without PDL, so wherever it wins the per-GEMM race it breaks the block's programmatic-launch chain;
the race times each GEMM alone and cannot see that. This measures it in the step.
"""
import argparse, statistics, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tdit import FusedTokenDiT                                        # noqa: E402
from miniworld_engine.modules.dit import DiTBlock                     # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType    # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--rounds", type=int, default=6)
a = p.parse_args()
L, S, NB, dev, bf = a.length, 5, 24, "cuda", torch.bfloat16
DS, DC, DP, H = 768, 384, 128, 16
torch.manual_seed(0)
blocks = torch.nn.ModuleList(DiTBlock(DS, DC, DP, H, n=2, implementation=ImplementationType.PYTORCH)
                             for _ in range(NB)).to(dev)
with torch.no_grad():
    for prm in blocks.parameters():
        if prm.ndim == 2: prm.normal_(std=prm.shape[1] ** -0.5)
        elif prm.numel() > 1: prm.add_(torch.randn_like(prm) * 0.1)
    for blk in blocks:
        blk.attention.to_out.weight.mul_(0.25); blk.transition.squeeze.weight.mul_(0.25)
blocks = blocks.to(bf).eval()
f = FusedTokenDiT(blocks, dtype=bf)
single = torch.randn(S, 1, L, DS, device=dev, dtype=bf)
cond = torch.randn(1, 1, L, DC, device=dev, dtype=bf).expand(S, 1, L, DC).contiguous()
pair = torch.randn(1, L, L, DP, device=dev, dtype=bf)


def time_us(fn, reps=5):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=st): fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(7):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps): g.replay()
        e1.record(); torch.cuda.synchronize()
        out.append(e0.elapsed_time(e1) * 1e3 / reps)
    return statistics.median(out) / NB


NAMES = {(S * L, 4 * DS, DS, True): "q|k|v|g", (S * L, DS, DS, False): "Wo", (S * L, DS, 2 * DS, False): "squeeze"}
with torch.no_grad():
    bias = f.hoist(pair)
    picks = {}
    for mode, allow in (("raced", True), ("quack-only", False)):
        f.mm_cublas, f._mm_cfg = allow, {}
        f.step(single, cond, bias)                                    # picks on first call, before any capture
        picks[mode] = dict(f._mm_cfg)
    print(f"L={L} S={S} {NB} blocks", flush=True)
    for k in picks["raced"]:
        n = NAMES.get(k, f"cond {k[0]}x{k[1]}x{k[2]}")
        print(f"  {n:<22} raced {str(picks['raced'][k]):<28} quack-only {picks['quack-only'][k]}", flush=True)
    runs, outs = {m: [] for m in picks}, {}
    for _ in range(a.rounds):
        for m in picks:
            f._mm_cfg = picks[m]
            runs[m].append(time_us(lambda: f.step(single, cond, bias)))
            outs[m] = f.step(single, cond, bias).float().clone()
    base = statistics.median(runs["raced"])
    for m, v in runs.items():
        md = statistics.median(v)
        print(f"  {m:<11} {md:7.2f} us/block ({base - md:+.2f} saved)   runs {' '.join(f'{x:.2f}' for x in v)}", flush=True)
    d = float((outs["quack-only"] - outs["raced"]).norm() / outs["raced"].norm())
    print(f"  output rel diff quack-only vs raced {d:.2e} (different GEMM kernels round differently)", flush=True)
