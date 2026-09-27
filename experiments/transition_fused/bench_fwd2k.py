"""The two-kernel Transition forward for wide channels: LN -> swiglu_gemm (h) -> h Ws^T + x (cuBLAS addmm), against the
engine module (Triton), torch.compile, and the fp32 reference.   python bench_fwd2k.py --width 512 --length 384
"""
import argparse, statistics, sys
from pathlib import Path
import torch, torch.nn.functional as F
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drv  # noqa: E402
p = argparse.ArgumentParser(); p.add_argument("--width", type=int, default=512); p.add_argument("--length", type=int, default=384)
p.add_argument("--cubin", default=str(HERE / "build/sg_tanh.cubin")); p.add_argument("--sq-bn", type=int, default=0); a = p.parse_args()
D, H, L = a.width, 4 * a.width, a.length; M = L * L
PEAK = 989e12
from miniworld_engine import settings                      # noqa: E402
from miniworld_engine.kernels.transition.reference import transition_pytorch  # noqa: E402
from miniworld_engine.modules import Transition           # noqa: E402
torch.manual_seed(2319)
mod = Transition(D, n=4, implementation="triton").cuda().bfloat16()
with torch.no_grad():
    for prm in mod.parameters():
        if prm.ndim == 2: prm.normal_(std=prm.shape[-1] ** -0.5)
x = torch.randn(1, L, L, D, device="cuda", dtype=torch.bfloat16)
lw, lb, eps = mod.ln_in.weight, mod.ln_in.bias, mod.ln_in.eps
wa, wb, ws = mod.expand_a.weight.detach(), mod.expand_b.weight.detach(), mod.squeeze.weight.detach()
w1p = torch.stack([wa.view(H // 128, 128, D), wb.view(H // 128, 128, D)], 1).reshape(2 * H, D).contiguous()
wst = ws.t().contiguous()
xf = x.view(M, D)
xn = torch.empty_like(xf); h = torch.empty(M, H, device="cuda", dtype=torch.bfloat16); out = torch.empty_like(xf)
k = drv.Kernel(a.cubin, "swiglu_gemm", 229376 + 128)
tm = lambda t, dims, stride, box: drv.TensorMap(t, dims=dims, stride_bytes=stride, box=box)
maps = (tm(xn, [D, M], D * 2, [64, 64]), tm(w1p, [D, 2 * H], D * 2, [64, 256]), tm(h, [H, M], H * 2, [64, 64]))
lw16, lb16 = lw.to(torch.bfloat16), lb.to(torch.bfloat16)
from miniworld_engine.autotune.shape_key import both_key          # noqa: E402
from miniworld_engine.kernels.layernorm.triton.main import _ln_fwd  # noqa: E402
key = both_key(M)
gf, bfp = lw.float().contiguous(), lb.float().contiguous()
bn = a.sq_bn or (256 if D % 256 == 0 else 192 if D % 192 == 0 else 128)
ksq = drv.Kernel(str(HERE / f"build/sq_bn{bn}.cubin"), "squeeze_gemm", 231424)
wsd = ws.contiguous()
out2 = torch.empty_like(xf)
sqmaps = (tm(h, [H, M], H * 2, [64, 64]), tm(wsd, [H, D], H * 2, [64, min(bn, 256)]), tm(xf, [D, M], D * 2, [64, 64]), tm(out2, [D, M], D * 2, [64, 64]))
def ln():
    xn_, _, _ = _ln_fwd(xf, gf, bfp, None, eps, False, key)
    xn.copy_(xn_) if xn_.data_ptr() != xn.data_ptr() else None
def ln_only():
    return _ln_fwd(xf, gf, bfp, None, eps, False, key)
def sq():
    ksq((132, 1, 1), (384, 1, 1), *sqmaps, int(M), int(H), int(D))
    return out2


def two_kernel():
    xn.copy_(F.layer_norm(xf, (D,), lw16, lb16, eps))
    k((132, 1, 1), (384, 1, 1), *maps, int(M), int(D), int(H))
    torch.addmm(xf, h, wst, out=out)
    return out


def ours():
    xn_, _, _ = _ln_fwd(xf, gf, bfp, None, eps, False, key)
    xn.copy_(xn_)
    k((132, 1, 1), (384, 1, 1), *maps, int(M), int(D), int(H))
    return sq()


with torch.no_grad():
    ref = transition_pytorch(x.float(), lw.float(), lb.float(), wa.float(), wb.float(), ws.float(), 4, eps).view(M, D)
    y = two_kernel().clone(); torch.cuda.synchronize()
    y2 = ours().clone(); torch.cuda.synchronize()
    settings.configure(engine_backend="triton", transition_residual_fusion=True, transition_fused_sm90a=True)
    ye = mod(x).view(M, D)
    rr = lambda t: float((t.float() - ref).norm() / ref.norm())
    print(f"D{D} L{L}: rel_rms vs fp32  two-kernel(addmm) {rr(y):.3e}  ours(Triton LN + swiglu_gemm + squeeze_gemm BN {bn}) {rr(y2):.3e}  engine {rr(ye):.3e}")


def t(fn, reps=10, graph=True):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    if graph:
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s): fn()
        run = g.replay
    else:
        run = fn
    torch.cuda.synchronize(); o = []
    for _ in range(5):
        st, en = torch.cuda.Event(True), torch.cuda.Event(True); st.record()
        for _ in range(reps): run()
        en.record(); torch.cuda.synchronize(); o.append(st.elapsed_time(en) * 1e3 / reps)
    return statistics.median(o)


floor = 24 * M * D * D / PEAK * 1e6
def eng():
    with torch.no_grad(): return mod(x)
tp = torch.compile(lambda t_: transition_pytorch(t_, lw, lb, wa, wb, ws, 4, eps))
def comp():
    with torch.no_grad(): return tp(x)
with torch.no_grad():
    rows = [("ours (LN + swiglu_gemm + squeeze_gemm)", t(ours)), ("two-kernel (F.layer_norm + swiglu + addmm)", t(two_kernel)), ("engine module (Triton)", t(eng)), ("torch.compile", t(comp, graph=False))]
    parts = [("  LN (engine Triton _ln_fwd)", t(ln_only)), ("  LN (F.layer_norm + copy)", t(lambda: xn.copy_(F.layer_norm(xf, (D,), lw16, lb16, eps)))),
             ("  squeeze_gemm (+x)", t(sq)),
             ("  swiglu_gemm", t(lambda: k((132, 1, 1), (384, 1, 1), *maps, int(M), int(D), int(H)))),
             ("  addmm (h Ws^T + x)", t(lambda: torch.addmm(xf, h, wst, out=out)))]
for n, us in rows + parts:
    print(f"  {n:40s} {us:8.1f} us   {100 * floor / us:5.1f} % of the forward floor ({floor:.0f} us)" if not n.startswith("  ") else f"  {n:40s} {us:8.1f} us")
