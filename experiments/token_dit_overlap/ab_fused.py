"""Step A/B of the fused residual GEMM (gra/gemm_resgate_adaln_ws.cu) against mm + resgate_adaln_rows, in one
process, interleaved: base (as shipped), wo (the attention's Wo + row pass fused), both (Wo and the transition's
squeeze). The runner is left untouched: its _mm is intercepted for the Wo / squeeze weights and the following
resgate_adaln_rows call runs the fused kernel on the intercepted operands."""
import argparse, os, statistics, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "token_dit_fused"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tdit import FusedTokenDiT                                        # noqa: E402
from tdit import kernels as K                                         # noqa: E402
from tdit import runner as R                                        # noqa: E402
from gra import fused                                                # noqa: E402
from miniworld_engine.modules.dit import DiTBlock                     # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType    # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--rounds", type=int, default=6)
a = p.parse_args()
L, S, NB, dev, bf = a.length, 5, 24, "cuda", torch.bfloat16
DS, DC, DP, H = 768, 384, 128, 16

torch.manual_seed(0)
ref_blocks = torch.nn.ModuleList(DiTBlock(DS, DC, DP, H, n=2, implementation=ImplementationType.PYTORCH)
                                 for _ in range(NB)).to(dev)
with torch.no_grad():
    for prm in ref_blocks.parameters():
        if prm.ndim == 2: prm.normal_(std=prm.shape[1] ** -0.5)
        elif prm.numel() > 1: prm.add_(torch.randn_like(prm) * 0.1)
    for blk in ref_blocks:
        blk.attention.to_out.weight.mul_(0.25); blk.transition.squeeze.weight.mul_(0.25)
bf_blocks = torch.nn.ModuleList(DiTBlock(DS, DC, DP, H, n=2, implementation=ImplementationType.PYTORCH)
                                for _ in range(NB)).to(dev)
bf_blocks.load_state_dict(ref_blocks.state_dict())
bf_blocks = bf_blocks.to(bf).eval()
f = FusedTokenDiT(bf_blocks, dtype=bf)
single = torch.randn(S, 1, L, DS, device=dev)
cond = torch.randn(1, 1, L, DC, device=dev).expand(S, 1, L, DC).contiguous()
pair = torch.randn(1, L, L, DP, device=dev)
s_bf, c_bf, z_bf = single.to(bf), cond.to(bf), pair.to(bf)


ws = fused("ws")[0]
MODE = {"m": "base"}
WO = {p["wo"].data_ptr() for p in f.per}
WS = {p["ws"].data_ptr() for p in f.per}
pending = {}
orig_mm, orig_rows = f._mm, R.K.resgate_adaln_rows


def mm(A, W, out, bias=None):
    m = MODE["m"]
    take = bias is None and ((m in ("wo", "both") and W.data_ptr() in WO) or (m == "both" and W.data_ptr() in WS))
    if take:
        pending["aw"] = (A, W)
        return
    return orig_mm(A, W, out, bias)


def rows(x, y, gl, ms, mb, out, L, eps=1e-5, inv_s=1.0):
    if "aw" in pending:
        A, W = pending.pop("aw")
        return ws(A, W, x, gl, ms, mb, out if ms is not None else None, L, eps)
    return orig_rows(x, y, gl, ms, mb, out, L, eps, inv_s)


f._mm = mm
R.K.resgate_adaln_rows = rows


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
    x32 = single
    for blk in ref_blocks:
        x32 = blk(x32, cond, pair)
    ref = x32.reshape(S * L, DS)
    bias = f.hoist(z_bf)
    modes = ("base", "wo", "both")
    runs, errs = {m: [] for m in modes}, {}
    for m in modes:                                                 # builds + first-call picks before any capture
        MODE["m"] = m; f.step(s_bf, c_bf, bias)
    for _ in range(a.rounds):
        for m in modes:
            MODE["m"] = m
            runs[m].append(time_us(lambda: f.step(s_bf, c_bf, bias)))
            out = f.step(s_bf, c_bf, bias).float().reshape(S * L, DS)
            errs[m] = float((out - ref).norm() / ref.norm())
    b0 = statistics.median(runs["base"])
    print(f"L={L} S={S} {NB} blocks, per block", flush=True)
    for m in modes:
        md = statistics.median(runs[m])
        print(f"  {m:<5} {md:7.2f} us ({b0 - md:+.2f} saved)  rel_rms {errs[m]:.2e}   runs {' '.join(f'{x:.1f}' for x in runs[m])}",
              flush=True)
