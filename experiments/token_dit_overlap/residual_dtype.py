"""The v6/v7 step with the residual stream x kept in bf16 instead of fp32: x is the dominant traffic of the two row
passes (read + write per half-block, 4 x M x 768 x 4 B per block in fp32), and the engine's own bf16 path keeps it in bf16.
Accuracy is against the IEEE fp32 PyTorch reference, set up exactly as token_dit_fused/bench.py does (same seed and init),
so the rel_rms numbers line up with results/bf16-L*.json (engine 1.1e-2, v7 4.4e-3)."""
import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "token_dit_fused"))
from tdit import FusedTokenDiT                                   # noqa: E402
from tdit import kernels as K                                    # noqa: E402
from tdit.attn import attention_gated_in_place2, bias_descriptor  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
a = p.parse_args()
L, S, NB, dev, bf = a.length, 5, 24, "cuda", torch.bfloat16
DS, DC, DP, H = 768, 384, 128, 16
torch.backends.cuda.matmul.allow_tf32 = False
from miniworld_engine.modules.dit import DiTBlock                    # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType  # noqa: E402

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
eng_blocks = torch.nn.ModuleList(DiTBlock(DS, DC, DP, H, n=2, implementation=ImplementationType.MINIWORLD)
                                 for _ in range(NB)).to(dev)
eng_blocks.load_state_dict(ref_blocks.state_dict())
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
bdesc = bias_descriptor(bias)
M = S * L


def buffers(xdt):
    return dict(x=torch.empty(M, DS, device=dev, dtype=xdt), xa=torch.empty(M, DS, device=dev, dtype=bf),
                qkvg=torch.empty(M, 4 * DS, device=dev, dtype=bf), y=torch.empty(M, DS, device=dev, dtype=bf),
                h=torch.empty(M, 2 * DS, device=dev, dtype=bf))


def step(buf):
    g1, g2 = f._cond(c_bf, L, DS)
    x, xa, qkvg, y, h = (buf[k] for k in ("x", "xa", "qkvg", "y", "h"))
    x.copy_(s_bf.reshape(M, DS))
    K.adaln_rows(x, g1[:, 0, 0], g1[:, 0, 1], xa, L, f.eps)
    q4, k4, v4, g4 = (qkvg.view(S, L, 4 * DS)[..., i * DS:(i + 1) * DS].unflatten(-1, (H, DS // H)) for i in range(4))
    for b, pk in enumerate(f.per):
        torch.addmm(pk["bqkvg"], xa, pk["wqkvg"].t(), out=qkvg)
        attention_gated_in_place2(q4, k4, v4, g4, bdesc, b, f.core_precision)
        torch.mm(qkvg[:, :DS], pk["wo"].t(), out=y)
        K.resgate_adaln_rows(x, y, g2[:, b, 0], g1[:, b, 2], g1[:, b, 3], xa, L, f.eps)
        f._expand_swiglu(xa, pk["wab_i"], h)
        torch.mm(h, pk["ws"].t(), out=y)
        last = b + 1 == NB
        K.resgate_adaln_rows(x, y, g2[:, b, 1], None if last else g1[:, b + 1, 0],
                             None if last else g1[:, b + 1, 1], xa, L, f.eps)
    return x


def time_us(fn, reps=5):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
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
        en.record()
        torch.cuda.synchronize()
        out.append(st.elapsed_time(en) * 1000.0 / reps)
    return statistics.median(out)


def rel(got, want):
    return float((got.float() - want.float()).norm() / want.float().norm())


with torch.no_grad():
    print(f"L={L} S={S} {NB} blocks bf16, per block; rel_rms vs IEEE fp32 PyTorch", flush=True)
    t = time_us(lambda: f.step(s_bf, c_bf, bias))
    print(f"  FusedTokenDiT.step (v7 as packaged)   {t / NB:7.1f} us   {rel(f.step(s_bf, c_bf, bias).reshape(M, DS), ref):.2e}")
    for name, xdt in (("residual fp32", torch.float32), ("residual bf16", bf)):
        buf = buffers(xdt)
        out = step(buf).clone()
        print(f"  {name:<36s} {time_us(lambda: step(buf)) / NB:7.1f} us   {rel(out, ref):.2e}", flush=True)
