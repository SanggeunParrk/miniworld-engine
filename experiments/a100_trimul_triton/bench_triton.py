"""Triton TriMul vs the fp32 module (inference + training gradients) and vs the A100 CUDA kernels, CUDA-graph timing.
   python bench_triton.py --variant bidir single --length 64 384 768 [--no-time] [--no-cuda]"""
import argparse, copy, statistics, sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "a100_trimul_fwd"))
import trimul_triton as TT  # noqa: E402
from miniworld_engine.modules.exceptions import ImplementationType as I  # noqa: E402
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication  # noqa: E402
from miniworld_engine.modules.triangle_multiplication.bidirectional import BidirectionalTriangleMultiplication  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variant", nargs="+", default=["bidir", "single"])
ap.add_argument("--length", nargs="+", type=int, default=[64, 384, 768])
ap.add_argument("--no-time", action="store_true")
ap.add_argument("--no-cuda", action="store_true")
ap.add_argument("--p", type=float, default=0.25)
a = ap.parse_args()
if not a.no_cuda:
    import trimul_a100 as TA  # noqa: E402
    import trimul_train as TC  # noqa: E402
    ext = TA.build()


def init_(m):
    torch.manual_seed(1234)
    with torch.no_grad():
        for n, t in m.named_parameters():
            if t.ndim >= 2: t.normal_(std=t.shape[-1] ** -.5)
            elif "weight" in n: t.copy_(1 + .1 * torch.randn_like(t))
            else: t.normal_(std=.05)
    return m


def gtime(fn, reps=20):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
        torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s): fn()
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(20): g.replay()
    r = []
    for _ in range(7):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(reps): g.replay()
        en.record(); en.synchronize(); r.append(st.elapsed_time(en) / reps)
    return statistics.median(r)


rel = lambda u, v: float((u.float() - v).norm() / (v.norm() + 1e-30))  # noqa: E731
for var in a.variant:
    for L in a.length:
        mk = lambda: (BidirectionalTriangleMultiplication(128, implementation=I.PYTORCH, p_drop=a.p) if var == "bidir"  # noqa: E731
                      else TriangleMultiplication(128, implementation=I.PYTORCH, p_drop=a.p))
        mod = init_(mk().cuda().bfloat16())
        torch.manual_seed(90323)
        z = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        mask = torch.rand(1, L, device="cuda") > .1
        ds = ((torch.rand(1, 1, L, 128, device="cuda") > a.p).to(torch.bfloat16) / (1 - a.p)).to(torch.bfloat16)
        dy = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16)
        # ---- inference vs the fp32 module
        ref = copy.deepcopy(mod).float().eval()
        with torch.no_grad():
            yi_ref = ref(z.detach().float(), mask)
            yi = TT.forward(mod, z.detach(), mask)
            mb = copy.deepcopy(mod).eval()
            yi_b = mb(z.detach(), mask)
        line = f"{var:6s} L{L}: infer rel {rel(yi, yi_ref):.2e} (bf16 module {rel(yi_b, yi_ref):.2e})"
        # ---- training vs the fp32 module (same dropout scale)
        reft = copy.deepcopy(mod).float().train()
        reft._make_drop_row_scale = lambda pair, p, _ds=ds: _ds.float()
        zr = z.detach().float().requires_grad_()
        gr = torch.autograd.grad(reft(zr, mask), [zr] + [dict(reft.named_parameters())[n] for n in TT.PARAMS], dy.float())
        params = [dict(mod.named_parameters())[n] for n in TT.PARAMS]
        mod.train()
        gs = torch.autograd.grad(TT.forward_train(mod, z, mask, ds), [z] + params, dy)
        errs = {n: rel(g_, r_) for n, g_, r_ in zip(["dz"] + list(TT.PARAMS), gs, gr)}
        worst = max(errs.items(), key=lambda kv: kv[1])
        line += f" | train dz {errs['dz']:.2e} worst {worst[0]} {worst[1]:.2e}"
        del gs
        if not a.no_time and L >= 384:
            with torch.no_grad():
                ti = gtime(lambda: TT.forward(mod, z.detach(), mask))
            tt = gtime(lambda: torch.autograd.grad(TT.forward_train(mod, z, mask, ds), [z] + params, dy))
            line += f" | TRITON infer {ti * 1000:.1f} us train {tt:.3f} ms"
            # the engine's Triton path, same job, same inputs (its own dropout RNG in training)
            eng = mk().cuda().bfloat16()
            eng.load_state_dict(mod.state_dict())
            eng = eng.__class__(128, implementation=I.TRITON, p_drop=a.p).cuda().bfloat16() if False else eng
            eng.implementation = I.TRITON
            try:
                from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication as _TM  # noqa: F401
                engm = (BidirectionalTriangleMultiplication(128, implementation=I.TRITON, p_drop=a.p) if var == "bidir"
                        else TriangleMultiplication(128, implementation=I.TRITON, p_drop=a.p)).cuda().bfloat16()
                engm.load_state_dict(mod.state_dict())
                engm.eval()
                with torch.no_grad():
                    ei = gtime(lambda: engm(z.detach(), mask))
                engm.train()
                eparams = list(engm.parameters())
                et = gtime(lambda: torch.autograd.grad(engm(z, mask), [z] + eparams, dy, allow_unused=True))
                line += f" | ENGINE infer {ei * 1000:.1f} us train {et:.3f} ms"
            except Exception as e:
                line += f" | ENGINE failed: {type(e).__name__} {str(e)[:80]}"
            if not a.no_cuda:
                pkc = TA.pack(mod)
                bufs = {}
                with torch.no_grad():
                    ci = gtime(lambda: TA.forward(ext, z.detach(), mask, pkc, bufs))
                ct = gtime(lambda: torch.autograd.grad(TC.forward_train(ext, mod, z, mask, ds), [z] + params, dy))
                line += f" | CUDA infer {ci * 1000:.1f} us train {ct:.3f} ms"
        print(line, flush=True)
