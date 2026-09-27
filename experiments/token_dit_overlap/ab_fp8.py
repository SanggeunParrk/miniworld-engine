"""fp8 q|k|v|g (TDIT_FP8_QKVG) for real: accuracy against the IEEE fp32 reference for a few static xa bounds, and the
step A/B against bf16, in one process, interleaved. Two runners share the weights' source blocks and the hoisted bias."""
import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tdit import FusedTokenDiT                                    # noqa: E402
from tdit import runner as R                                      # noqa: E402
from tdit.cuda_core import attn_core                                     # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--rounds", type=int, default=6)
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


import os, statistics                                              # noqa: E402
f16 = FusedTokenDiT(bf_blocks, dtype=bf)
os.environ["TDIT_FP8_QKVG"] = "1"
f8 = FusedTokenDiT(bf_blocks, dtype=bf)
os.environ["TDIT_FP8_QKVG"] = "0"
assert f8.fp8_qkvg and not f16.fp8_qkvg
bias = f16.hoist(z_bf)


def set_bound(bound):
    new = bound / 448.0
    for p in f8.per:
        p["alpha8"] = p["alpha8"] / f8.xa_scale * new
    f8.xa_scale = new


def err(f):
    out = f.step(s_bf, c_bf, bias).float().reshape(S * L, DS)
    return float((out - ref).norm() / ref.norm())


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


with torch.no_grad():
    f16.step(s_bf, c_bf, bias); f8.step(s_bf, c_bf, bias)          # picks and builds before anything is timed
    base = err(f16)
    print(f"L={L}: bf16 rel_rms {base:.2e}   fp8 picks {f8._mm8_cfg}", flush=True)
    for bound in (32, 64, 128, 256):
        set_bound(bound)
        e = err(f8)
        print(f"  fp8 q|k|v|g, xa bound {bound:>3}: rel_rms {e:.2e} (x{e / base:.2f})", flush=True)
    set_bound(128)
    runs = {"bf16": [], "fp8 qkvg": []}
    for _ in range(a.rounds):
        for n, f in (("bf16", f16), ("fp8 qkvg", f8)):
            runs[n].append(time_us(lambda f=f: f.step(s_bf, c_bf, bias)))
    b0 = statistics.median(runs["bf16"])
    for n, v in runs.items():
        md = statistics.median(v)
        print(f"  {n:<9} {md:7.2f} us/block ({b0 - md:+.2f} saved)   runs {' '.join(f'{x:.2f}' for x in v)}", flush=True)
    outs = [f8.step(s_bf, c_bf, bias).float().clone() for _ in range(4)]
    print(f"  fp8 deterministic: {all(torch.equal(o, outs[0]) for o in outs)}", flush=True)
