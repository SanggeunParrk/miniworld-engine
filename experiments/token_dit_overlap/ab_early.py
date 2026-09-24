"""A/B, with PDL on everywhere: loads that do not depend on the previous kernel moved before the PDL wait.

  base   the row pass reads x, y and the conditioning after gdc_wait; the core waits before touching anything
  early  the row pass reads x and the conditioning rows before the wait, only y after (TDIT_EARLY)
  bpre   the core pulls its bias rows into L2 before the wait (BPRE)
  both
"""
import argparse, os, statistics, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tdit import FusedTokenDiT                                        # noqa: E402
from tdit import kernels as K                                         # noqa: E402
from tdit import cuda_core as CC                                      # noqa: E402
from miniworld_engine.modules.dit import DiTBlock                     # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType    # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--rounds", type=int, default=4)
a = p.parse_args()
L, S, NB, dev, bf = a.length, 5, 24, "cuda", torch.bfloat16
DS, DC, DP, H = 768, 384, 128, 16

def build(defs):
    """One extension per define set; _ext is lru_cache(1), so clear it between builds and keep the module."""
    old = os.environ.get("ATTN_DEFS", "")
    os.environ["ATTN_DEFS"] = defs
    CC._ext.cache_clear()
    ext = CC._ext()
    os.environ["ATTN_DEFS"] = old
    CC._ext.cache_clear()
    return ext

EXT = {"base": build(""), "bpre": build("BPRE=1")}
MODES = {"base": ("base", False), "early": ("base", True), "bpre": ("bpre", False), "both": ("bpre", True)}

torch.manual_seed(0)
blocks = torch.nn.ModuleList(DiTBlock(DS, DC, DP, H, n=2, implementation=ImplementationType.PYTORCH)
                             for _ in range(NB)).to(dev)
with torch.no_grad():
    for prm in blocks.parameters():
        if prm.ndim == 2: prm.normal_(std=prm.shape[1] ** -0.5)
        elif prm.numel() > 1: prm.add_(torch.randn_like(prm) * 0.1)
    for blk in blocks:
        blk.attention.to_out.weight.mul_(0.25); blk.transition.squeeze.weight.mul_(0.25)
ref_out = None
blocks = blocks.to(bf).eval()
f = FusedTokenDiT(blocks, dtype=bf)
single = torch.randn(S, 1, L, DS, device=dev, dtype=bf)
cond = torch.randn(1, 1, L, DC, device=dev, dtype=bf).expand(S, 1, L, DC).contiguous()
pair = torch.randn(1, L, L, DP, device=dev, dtype=bf)

def use(mode):
    ext = EXT[MODES[mode][0]]
    f._cuda_core = lambda q, b, blk, s, h, dbg=None, _e=ext: _e.attn_core(q, b, blk, s, h, dbg)
    K.PDL, K.EARLY = True, MODES[mode][1]

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
    bias = f.hoist(pair)
    runs, outs = {m: [] for m in MODES}, {m: [] for m in MODES}
    for r in range(a.rounds):
        for mode in MODES:
            use(mode)
            runs[mode].append(time_us(lambda: f.step(single, cond, bias)))
            outs[mode].append(f.step(single, cond, bias).float().clone())
    base = statistics.median(runs["base"])
    print(f"L={L} S={S} {NB} blocks, per block", flush=True)
    for m, v in runs.items():
        md = statistics.median(v)
        print(f"  {m:<6} {md:7.2f} us  ({base - md:+.2f} saved)   runs {' '.join(f'{x:.2f}' for x in v)}", flush=True)
    rd = lambda u, v: float((u - v).norm() / v.norm())
    print("  bit-identical to base: " + "  ".join(f"{m} {rd(outs[m][-1], outs['base'][0]):.1e}" for m in MODES), flush=True)
