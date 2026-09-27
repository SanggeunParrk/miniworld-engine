"""Training (fwd + bwd) correctness vs the fp32 module (same dropout scale) and CUDA-graph timing.
   python train_bench.py --variant bidir single --length 64 384 768 [--no-time]"""
import argparse, copy, statistics, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trimul_a100 as TA  # noqa: E402
import trimul_train as TT  # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType as I  # noqa: E402
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication  # noqa: E402
from miniworld_engine.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variant", nargs="+", default=["bidir", "single"])
ap.add_argument("--length", nargs="+", type=int, default=[64, 384])
ap.add_argument("--no-time", action="store_true")
ap.add_argument("--p", type=float, default=0.25)
ap.add_argument("--extra", nargs="*", default=[])
a = ap.parse_args()
a.extra = [x for e in a.extra for x in e.split()]
ext = TA.build(extra=a.extra)


def init_(m):
    torch.manual_seed(1234)
    with torch.no_grad():
        for n, t in m.named_parameters():
            if t.ndim >= 2: t.normal_(std=t.shape[-1] ** -.5)
            elif "weight" in n: t.copy_(1 + .1 * torch.randn_like(t))
            else: t.normal_(std=.05)
    return m


def gtime(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
        torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s): fn()
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(30): g.replay()
    r = []
    for _ in range(7):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(20): g.replay()
        en.record(); en.synchronize(); r.append(st.elapsed_time(en) / 20)
    return statistics.median(r)


for var in a.variant:
    for L in a.length:
        mod = (BidirectionalTriangleMultiplication(128, implementation=I.PYTORCH, p_drop=a.p) if var == "bidir"
               else TriangleMultiplication(128, implementation=I.PYTORCH, p_drop=a.p))
        mod = init_(mod.cuda().bfloat16()).train()
        torch.manual_seed(90323)
        z = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        mask = torch.rand(1, L, device="cuda") > .1
        ds = ((torch.rand(1, 1, L, 128, device="cuda") > a.p).to(torch.bfloat16) / (1 - a.p)).to(torch.bfloat16)
        dy = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16)
        # reference: fp32 module, the same dropout scale
        ref = copy.deepcopy(mod).float().train()
        ref._make_drop_row_scale = lambda pair, p, _ds=ds: _ds.float()
        zr = z.detach().float().requires_grad_()
        yr = ref(zr, mask)
        gr = torch.autograd.grad(yr, [zr] + [dict(ref.named_parameters())[n] for n in TT.PARAMS], dy.float())
        params = [dict(mod.named_parameters())[n] for n in TT.PARAMS]
        y = TT.forward_train(ext, mod, z, mask, ds)
        gs = torch.autograd.grad(y, [z] + params, dy)
        names = ["dz"] + list(TT.PARAMS)
        errs = {n: float((g_.float() - r_).norm() / (r_.norm() + 1e-30)) for n, g_, r_ in zip(names, gs, gr)}
        ey = float((y.float() - yr).norm() / yr.norm())
        # yardstick: the bf16 PyTorch module's own gradients against the same fp32 reference
        mb = copy.deepcopy(mod); mb._make_drop_row_scale = lambda pair, p, _ds=ds: _ds
        zb = z.detach().clone().requires_grad_()
        gb = torch.autograd.grad(mb(zb, mask), [zb] + [dict(mb.named_parameters())[n] for n in TT.PARAMS], dy)
        eb = max(float((g_.float() - r_).norm() / (r_.norm() + 1e-30)) for g_, r_ in zip(gb, gr))
        worst = max(errs.items(), key=lambda kv: kv[1])
        line = f"{var:6s} L{L}: out {ey:.2e} | " + " ".join(f"{k.split('.')[0]}:{v:.1e}" for k, v in errs.items()) + f" | worst {worst[0]} {worst[1]:.2e} (bf16 module worst {eb:.2e})"
        # the correctness graph pins z's / the weights' AccumulateGrad nodes with default-stream metadata: release it before capturing
        del y, gs, gb
        if not a.no_time and L >= 384:
            def step():
                yy = TT.forward_train(ext, mod, z, mask, ds)
                return torch.autograd.grad(yy, [z] + params, dy)
            ms = gtime(step)
            line += f" | fwd+bwd {ms:.3f} ms"
        print(line, flush=True)
