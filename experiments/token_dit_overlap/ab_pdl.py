"""A/B the PDL chain, in one process, interleaved: the CUDA core and the Triton row passes with and without
programmatic dependent launch. quack's sm90 GEMMs already wait/trigger (gemm_sm90.py, use_pdl=True), so this
closes the chain where OUR kernels break it.
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

EXT = {"off": build(""), "on": build("PDL=1")}

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
    ext = EXT[mode]
    f._cuda_core = lambda q, b, blk, s, h, dbg=None, _e=ext: _e.attn_core(q, b, blk, s, h, dbg)
    K.PDL = (mode == "on")

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
    runs, outs = {"off": [], "on": []}, {"off": [], "on": []}
    for r in range(a.rounds):
        for mode in ("off", "on"):
            use(mode)
            runs[mode].append(time_us(lambda: f.step(single, cond, bias)))
            outs[mode].append(f.step(single, cond, bias).float().clone())
    d = statistics.median(runs["off"]) - statistics.median(runs["on"])
    print(f"L={L} S={S} {NB} blocks, per block", flush=True)
    for m, v in runs.items():
        print(f"  PDL {m:<3s} {statistics.median(v):7.2f} us   (runs {' '.join(f'{x:.2f}' for x in v)})", flush=True)
    print(f"  PDL saves {d:+.2f} us/block ({100 * d / statistics.median(runs['off']):+.2f} %)", flush=True)
    def rd(u, v): return float((u - v).norm() / v.norm())
    # off-vs-off and on-vs-on separate a race from a systematic difference: PDL cannot change the arithmetic.
    print(f"  off vs off {rd(outs['off'][-1], outs['off'][0]):.2e}   on vs on {rd(outs['on'][-1], outs['on'][0]):.2e}"
          f"   on vs off {rd(outs['on'][0], outs['off'][0]):.2e}", flush=True)
