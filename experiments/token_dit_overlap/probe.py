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
from gra import gemm_resgate_adaln               # noqa: E402
import l2p                                          # noqa: E402
from quack.gemm_act import gemm_act                  # noqa: E402

# best (tile_N, cluster_M, cluster_N, pingpong) per shape from gemm_sweep.py at M = 3840 / 1920; cuBLAS loses to all of
# them (qkvg 32.6 vs 26.4, Wo 9.4 vs 9.0, squeeze 15.1 vs 14.4 us at L768), and the packaged v7 sends only qkvg to quack.
# per M, from gemm_sweep.py; None = cuBLAS wins there. cluster_N > 1 (A multicast) is what v7's candidate list misses.
QCFG = {3840: {"qkvg": (192, 1, 1, True), "wo": (192, 1, 4, False), "sq": (192, 1, 4, False)},
        1920: {"qkvg": (192, 1, 1, True), "wo": None, "sq": None}}


def qmm(A, W, out, key, bias=None):
    cfg = QCFG.get(A.shape[0], {}).get(key)
    if cfg is None:
        if bias is None:
            torch.mm(A, W.t(), out=out)
        else:
            torch.addmm(bias, A, W.t(), out=out)
        return
    tn, cm, cn, pp = cfg
    gemm_act(A[None], W[None], None, None, out[None], None, None, 128, tn, cm, cn, pingpong=pp,
             rowvec_bias=None if bias is None else bias[None])

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
    """x, xa and y in one contiguous pool: an L2 persisting window is a single byte range per stream."""
    M = Sg * L
    pool = torch.empty(M * DS * 8, device=dev, dtype=torch.uint8)
    x = pool[: M * DS * 4].view(torch.float32).view(M, DS)
    xa = pool[M * DS * 4: M * DS * 6].view(bf).view(M, DS)
    y = pool[M * DS * 6:].view(bf).view(M, DS)
    return dict(pool=pool, x=x, xa=xa, y=y,
                qkvg=torch.empty(M, 4 * DS, device=dev, dtype=bf), h=torch.empty(M, 2 * DS, device=dev, dtype=bf))


def run_group(sg, s0, buf, g1, g2, rows=True, core=True, gra=False, one_w=False, qmm_on=False):
    """The v6 per-block schedule for samples [s0, s0 + sg)."""
    M = sg * L
    x, xa, qkvg, y, h = (buf[k] for k in ("x", "xa", "qkvg", "y", "h"))
    x.copy_(single[s0:s0 + sg].reshape(M, DS))
    if rows:
        K.adaln_rows(x, g1[:, 0, 0], g1[:, 0, 1], xa, L, f.eps)
    q4, k4, v4, g4 = (qkvg.view(sg, L, 4 * DS)[..., i * DS:(i + 1) * DS].unflatten(-1, (H, DS // H)) for i in range(4))
    for b, pk in enumerate([f.per[0]] * NB if one_w else f.per):
        if qmm_on:
            qmm(xa, pk["wqkvg"], qkvg, "qkvg", pk["bqkvg"])
        else:
            torch.addmm(pk["bqkvg"], xa, pk["wqkvg"].t(), out=qkvg)
        if core:
            attention_gated_in_place2(q4, k4, v4, g4, bdesc, b, f.core_precision)
        last = b + 1 == NB
        if gra:
            gemm_resgate_adaln(qkvg[:, :DS], pk["wo"], x, g2[:, b, 0], g1[:, b, 2], g1[:, b, 3], xa, L, f.eps)
            f._expand_swiglu(xa, pk["wab_i"], h)
            gemm_resgate_adaln(h, pk["ws"], x, g2[:, b, 1], None if last else g1[:, b + 1, 0],
                               None if last else g1[:, b + 1, 1], None if last else xa, L, f.eps)
            continue
        if qmm_on:
            qmm(qkvg[:, :DS], pk["wo"], y, "wo")
        else:
            torch.mm(qkvg[:, :DS], pk["wo"].t(), out=y)
        if rows:
            K.resgate_adaln_rows(x, y, g2[:, b, 0], g1[:, b, 2], g1[:, b, 3], xa, L, f.eps)
        f._expand_swiglu(xa, pk["wab_i"], h)
        if qmm_on:
            qmm(h, pk["ws"], y, "sq")
        else:
            torch.mm(h, pk["ws"].t(), out=y)
        if rows:
            K.resgate_adaln_rows(x, y, g2[:, b, 1], None if last else g1[:, b + 1, 0],
                                 None if last else g1[:, b + 1, 1], xa, L, f.eps)
    return x


GROUPS = {"base": (S,), "streams 3+2": (3, 2), "streams 2+2+1": (2, 2, 1), "streams 1x5": (1,) * S}
BUFS = {name: [buffers(sg) for sg in gs] for name, gs in GROUPS.items()}
STREAMS = [torch.cuda.Stream() for _ in range(S)]


def step(name, rows=True, core=True, gra=False, one_w=False, l2=False, qmm_on=False):
    gs, bufs = GROUPS[name], BUFS[name]
    g1, g2 = f._cond(cond, L, DS)
    if len(gs) == 1:
        return run_group(gs[0], 0, bufs[0], g1, g2, rows, core, gra, one_w, qmm_on)
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
                     ("base", dict(rows=False, core=False)), ("base", dict(rows=False, core=False, one_w=True)),
                     ("base", dict(qmm_on=True)), ("base", dict(qmm_on=True, rows=False, core=False)),
                     ):
        # the window is a stream attribute: set it before capture (cudaStreamSetAttribute is illegal while capturing),
        # and the kernel nodes inherit it
        l2_on = kw.pop("l2", False)
        if l2_on:
            l2p.set_window(BUFS[name][0][l2_on], float(kw.pop("hit", 1.0)))
        else:
            l2p.clear_window()
        out = step(name, **kw).clone().float()
        l2tag = f"  +L2 persist({l2_on})" if l2_on else ""
        tag = name + ("  -rows (upper bound)" if kw.get("rows") is False else "") + ("  -core" if kw.get("core") is False else "") + ("  +gemm_resgate_adaln" if kw.get("gra") else "") + ("  one weight set (L2-resident)" if kw.get("one_w") else "") + ("  all GEMMs via quack" if kw.get("qmm_on") else "") + l2tag
        d = float((out - ref).norm() / ref.norm())
        print(f"  {tag:<32s} {time_us(lambda: step(name, **kw)) / NB:7.1f} us   vs step {d:.1e}", flush=True)
