"""Where can the fused token DiT step (token_dit_fused v6, frozen at cf70909b) still go faster without new kernels?

  base        the v6 schedule, re-assembled here from the package's kernels (should match FusedTokenDiT.step)
  -rows       both resgate_adaln_rows passes skipped: the most that moving AdaLN into a GEMM could ever save (wrong numbers)
  -core       attention skipped: what the core costs in the step (wrong numbers)
  streams G   the S samples split into groups G, each group's 24 blocks on its own CUDA stream. Samples never interact in
              the token DiT (pair bias and conditioning are shared read-only), so the groups are fully independent; the
              point is to let one group's attention core (ALU-bound softmax) run beside another group's GEMMs (tensor-bound)
              and row passes (memory-bound). Conditioning is computed once per step, before the fork.
"""
import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "token_dit_fused"))
from tdit import FusedTokenDiT                                  # noqa: E402
from tdit import kernels as K                                   # noqa: E402
from tdit.attn import attention_gated_in_place2, bias_descriptor  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, default=768)
p.add_argument("--blocks", type=int, default=24)
p.add_argument("--samples", type=int, default=5)
a = p.parse_args()
L, S, NB, dev, bf = a.length, a.samples, a.blocks, "cuda", torch.bfloat16
DS, DC, DP, H = 768, 384, 128, 16

from miniworld_engine.modules.dit import DiTBlock                    # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType  # noqa: E402

torch.manual_seed(0)
blocks = torch.nn.ModuleList(DiTBlock(DS, DC, DP, H, n=2, implementation=ImplementationType.PYTORCH)
                             for _ in range(NB)).to(dev)
with torch.no_grad():
    for prm in blocks.parameters():
        if prm.ndim == 2:
            prm.normal_(std=prm.shape[1] ** -0.5)
        elif prm.numel() > 1:
            prm.add_(torch.randn_like(prm) * 0.1)
    for blk in blocks:
        blk.attention.to_out.weight.mul_(0.25)
        blk.transition.squeeze.weight.mul_(0.25)
blocks = blocks.to(bf).eval()
f = FusedTokenDiT(blocks, dtype=bf)
single = torch.randn(S, 1, L, DS, device=dev, dtype=bf)
cond = torch.randn(1, 1, L, DC, device=dev, dtype=bf).expand(S, 1, L, DC).contiguous()
pair = torch.randn(1, L, L, DP, device=dev, dtype=bf)
bias = f.hoist(pair)
bdesc = bias_descriptor(bias)


def buffers(Sg):
    M = Sg * L
    return dict(x=torch.empty(M, DS, device=dev), xa=torch.empty(M, DS, device=dev, dtype=bf),
                qkvg=torch.empty(M, 4 * DS, device=dev, dtype=bf), y=torch.empty(M, DS, device=dev, dtype=bf),
                h=torch.empty(M, 2 * DS, device=dev, dtype=bf))


def run_group(sg, s0, buf, g1, g2, rows=True, core=True):
    """The v6 per-block schedule for samples [s0, s0 + sg)."""
    M = sg * L
    x, xa, qkvg, y, h = (buf[k] for k in ("x", "xa", "qkvg", "y", "h"))
    x.copy_(single[s0:s0 + sg].reshape(M, DS))
    if rows:
        K.adaln_rows(x, g1[:, 0, 0], g1[:, 0, 1], xa, L, f.eps)
    q4, k4, v4, g4 = (qkvg.view(sg, L, 4 * DS)[..., i * DS:(i + 1) * DS].unflatten(-1, (H, DS // H)) for i in range(4))
    for b, pk in enumerate(f.per):
        torch.addmm(pk["bqkvg"], xa, pk["wqkvg"].t(), out=qkvg)
        if core:
            attention_gated_in_place2(q4, k4, v4, g4, bdesc, b, f.core_precision)
        torch.mm(qkvg[:, :DS], pk["wo"].t(), out=y)
        if rows:
            K.resgate_adaln_rows(x, y, g2[:, b, 0], g1[:, b, 2], g1[:, b, 3], xa, L, f.eps)
        f._expand_swiglu(xa, pk["wab_i"], h)
        torch.mm(h, pk["ws"].t(), out=y)
        if rows:
            last = b + 1 == NB
            K.resgate_adaln_rows(x, y, g2[:, b, 1], None if last else g1[:, b + 1, 0],
                                 None if last else g1[:, b + 1, 1], xa, L, f.eps)
    return x


GROUPS = {"base": (S,), "streams 3+2": (3, 2), "streams 2+2+1": (2, 2, 1), "streams 1x5": (1,) * S}
BUFS = {name: [buffers(sg) for sg in gs] for name, gs in GROUPS.items()}
STREAMS = [torch.cuda.Stream() for _ in range(S)]


def step(name, rows=True, core=True):
    gs, bufs = GROUPS[name], BUFS[name]
    g1, g2 = f._cond(cond, L, DS)
    if len(gs) == 1:
        return run_group(gs[0], 0, bufs[0], g1, g2, rows, core)
    cur = torch.cuda.current_stream()
    s0 = 0
    for sg, buf, st in zip(gs, bufs, STREAMS):
        st.wait_stream(cur)
        with torch.cuda.stream(st):
            run_group(sg, s0, buf, g1, g2, rows, core)
        s0 += sg
    for st in STREAMS[:len(gs)]:
        cur.wait_stream(st)
    return torch.cat([b["x"] for b in bufs])


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
    for _ in range(5):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(reps):
            g.replay()
        en.record()
        torch.cuda.synchronize()
        out.append(st.elapsed_time(en) * 1000.0 / reps)
    return statistics.median(out)


with torch.no_grad():
    ref = f.step(single, cond, bias).float().reshape(S * L, DS)
    t_pkg = time_us(lambda: f.step(single, cond, bias))
    print(f"L={L} S={S} {NB} blocks bf16, per block")
    print(f"  FusedTokenDiT.step (v6)        {t_pkg / NB:7.1f} us")
    for name, kw in (("base", {}), ("base", dict(rows=False)), ("base", dict(core=False)),
                     ("streams 3+2", {}), ("streams 2+2+1", {}), ("streams 1x5", {})):
        out = step(name, **kw).clone().float()
        tag = name + ("  -rows (upper bound)" if kw.get("rows") is False else "") + ("  -core" if kw.get("core") is False else "")
        d = float((out - ref).norm() / ref.norm())
        print(f"  {tag:<32s} {time_us(lambda: step(name, **kw)) / NB:7.1f} us   vs step {d:.1e}", flush=True)
