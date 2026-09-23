"""A/B the step with the Triton core against the CUDA one, in one process, interleaved."""
import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tdit import FusedTokenDiT                                    # noqa: E402
from tdit import runner as R                                      # noqa: E402
from core_cu import attn_core                                     # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--rounds", type=int, default=4)
a = p.parse_args()
L, S, NB, dev, bf = a.length, 5, 24, "cuda", torch.bfloat16
DS, DC, DP, H = 768, 384, 128, 16
from miniworld_engine.modules.dit import DiTBlock                     # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType   # noqa: E402

torch.manual_seed(0)
ref_blocks = torch.nn.ModuleList(DiTBlock(DS, DC, DP, H, n=2, implementation=ImplementationType.PYTORCH)
                                 for _ in range(NB)).to(dev)
with torch.no_grad():
    for prm in ref_blocks.parameters():
        if prm.ndim == 2:
            prm.normal_(std=prm.shape[1] ** -0.5)
        elif prm.numel() > 1:
            prm.add_(torch.randn_like(prm) * 0.1)
    for blk in ref_blocks:
        blk.attention.to_out.weight.mul_(0.25)
        blk.transition.squeeze.weight.mul_(0.25)
bf_blocks = torch.nn.ModuleList(DiTBlock(DS, DC, DP, H, n=2, implementation=ImplementationType.PYTORCH)
                                for _ in range(NB)).to(dev)
bf_blocks.load_state_dict(ref_blocks.state_dict())
bf_blocks = bf_blocks.to(bf).eval()
single = torch.randn(S, 1, L, DS, device=dev)
cond = torch.randn(1, 1, L, DC, device=dev).expand(S, 1, L, DC).contiguous()
pair = torch.randn(1, L, L, DP, device=dev)
s_bf, c_bf, z_bf = single.to(bf), cond.to(bf), pair.to(bf)
with torch.no_grad():
    x = single
    for blk in ref_blocks:
        x = blk(x, cond, pair)
    ref = x.reshape(S * L, DS)

f = FusedTokenDiT(bf_blocks, dtype=bf)
bias = f.hoist(z_bf)
triton_core = R.attention_gated_in_place2


def cuda_core(q4, k4, v4, g4, bdesc, block, prec):
    """q4 is a view of the packed q|k|v|g buffer; rebuild it and hand the whole thing to the kernel."""
    qkvg = torch.as_strided(q4, (S * L, 4 * DS), (4 * DS, 1))
    attn_core(qkvg, bias, block, S, H)
    return q4


def time_us(fn, reps=5):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=st):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(7):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps):
            g.replay()
        e1.record(); torch.cuda.synchronize()
        out.append(e0.elapsed_time(e1) * 1e3 / reps)
    return statistics.median(out) / NB


def rel(got):
    return float((got.float().reshape(S * L, DS) - ref).norm() / ref.norm())


with torch.no_grad():
    runs = {"triton core": [], "cuda core": []}
    errs = {}
    for r in range(a.rounds):
        for name, fn in (("triton core", triton_core), ("cuda core", cuda_core)):
            R.attention_gated_in_place2 = fn
            try:
                runs[name].append(time_us(lambda: f.step(s_bf, c_bf, bias)))
                errs[name] = rel(f.step(s_bf, c_bf, bias))
            finally:
                R.attention_gated_in_place2 = triton_core
    print(f"L={L} S={S} {NB} blocks, per block", flush=True)
    for name, v in runs.items():
        print(f"  {name:<12s} {statistics.median(v):6.2f} us   rel_rms {errs[name]:.2e}   "
              f"(runs {' '.join(f'{x:.1f}' for x in v)})", flush=True)
    d = statistics.median(runs["triton core"]) - statistics.median(runs["cuda core"])
    print(f"  cuda core saves {d:+.2f} us/block ({100 * d / statistics.median(runs['triton core']):+.1f} %)", flush=True)
