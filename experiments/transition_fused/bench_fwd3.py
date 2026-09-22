"""Wide-channel Transition forward, every candidate in one process:
  (A) ln_swiglu_gemm (LN inside, resident xn) + squeeze_gemm          two kernels
  (B) engine Triton LN + swiglu_gemm + squeeze_gemm                    three kernels
  (C) the fused single kernel (D = 256 only)
  (D) the engine module (Triton)
Inference forward (no saved activations), CUDA-graph replay medians, rel_rms against fp32.
  python bench_fwd3.py --width 256 --length 384
"""
import argparse, statistics, sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drv  # noqa: E402
p = argparse.ArgumentParser(); p.add_argument("--width", type=int, default=512); p.add_argument("--length", type=int, default=384)
p.add_argument("--sq", default=""); p.add_argument("--fused", default=""); p.add_argument("--lnsg", default=""); p.add_argument("--only-a", action="store_true")
a = p.parse_args()
D, H, L = a.width, 4 * a.width, a.length; M = L * L
PEAK = 989e12
from miniworld_engine import settings                                   # noqa: E402
from miniworld_engine.kernels.transition.reference import transition_pytorch  # noqa: E402
from miniworld_engine.modules import Transition                        # noqa: E402
from miniworld_engine.autotune.shape_key import both_key               # noqa: E402
from miniworld_engine.kernels.layernorm.triton.main import _ln_fwd     # noqa: E402
torch.manual_seed(2319)
mod = Transition(D, n=4, implementation="triton").cuda().bfloat16()
with torch.no_grad():
    for prm in mod.parameters():
        if prm.ndim == 2: prm.normal_(std=prm.shape[-1] ** -0.5)
    mod.ln_in.weight.normal_(1, 0.1); mod.ln_in.bias.normal_(0, 0.1)
x = torch.randn(1, L, L, D, device="cuda", dtype=torch.bfloat16)
xf = x.view(M, D)
eps = mod.ln_in.eps
g32, b32 = mod.ln_in.weight.detach().float().contiguous(), mod.ln_in.bias.detach().float().contiguous()
wa, wb, ws = mod.expand_a.weight.detach(), mod.expand_b.weight.detach(), mod.squeeze.weight.detach().contiguous()
pack = lambda blk: torch.stack([wa.view(H // blk, blk, D), wb.view(H // blk, blk, D)], 1).reshape(2 * H, D).contiguous()
w1p64, w1p128 = pack(64), pack(128)
tm = lambda t, dims, stride, box: drv.TensorMap(t, dims=dims, stride_bytes=stride, box=box)
h = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
out = torch.empty_like(xf)
bn = 256 if D % 256 == 0 else 192
ksq = drv.Kernel(a.sq or str(HERE / f"build/sq_bn{bn}.cubin"), "squeeze_gemm", 231424)
sqmaps = (tm(h, [H, M], H * 2, [64, 64]), tm(ws, [H, D], H * 2, [64, min(bn, 256)]), tm(xf, [D, M], D * 2, [64, 64]), tm(out, [D, M], D * 2, [64, 64]))
sq = lambda: ksq((132, 1, 1), (384, 1, 1), *sqmaps, int(M), int(H), int(D))
kln = drv.Kernel(a.lnsg or str(HERE / f"build/lnsg_d{D}.cubin"), "ln_swiglu_gemm", 231424)
lnmaps = (tm(xf, [D, M], D * 2, [64, 64]), tm(w1p64, [D, 2 * H], D * 2, [64, 128]), tm(h, [H, M], H * 2, [64, 64]))
dummy = torch.empty(1, device="cuda", dtype=torch.float32); dummyb = torch.empty(1, device="cuda", dtype=torch.bfloat16)
def path_a():
    kln((132, 1, 1), (384, 1, 1), *lnmaps, g32, b32, dummyb, dummy, dummy, int(M), int(H), float(eps), 0)
    sq(); return out
ksg = drv.Kernel(str(HERE / "build/sg_tanh.cubin"), "swiglu_gemm", 229376 + 128)
key = both_key(M)
def path_b():
    xn, _, _ = _ln_fwd(xf, g32, b32, None, eps, False, key)
    ksg((132, 1, 1), (384, 1, 1), tm(xn, [D, M], D * 2, [64, 64]), tm(w1p128, [D, 2 * H], D * 2, [64, 256]), tm(h, [H, M], H * 2, [64, 64]), int(M), int(D), int(H))
    sq(); return out
paths = {"A ln_swiglu_gemm + squeeze (2 kernels)": path_a, "B Triton LN + swiglu_gemm + squeeze (3 kernels)": path_b}
if a.only_a: paths = {} if a.fused else {"A ln_swiglu_gemm + squeeze (2 kernels)": path_a}
if D == 256 and (a.fused or not a.only_a):
    kf = drv.Kernel(a.fused or str(HERE / "build/transition_fwd_d256.cubin"), "transition_fwd_fused", 213248)
    outf = torch.empty_like(xf); wst = ws.t().contiguous()
    fmaps = (tm(xf, [D, M], D * 2, [64, 64]), tm(wa, [D, H], D * 2, [64, 32]), tm(wb, [D, H], D * 2, [64, 32]), tm(wst, [D, H], D * 2, [64, 32]), tm(outf, [D, M], D * 2, [64, 64]))
    def path_c():
        kf((132, 1, 1), (384, 1, 1), *fmaps, g32, b32, dummyb, outf, dummy, dummy, int(M), int(M // 128), float(eps), 0); return outf
    paths["C fused single kernel"] = path_c
settings.configure(engine_backend="triton", transition_residual_fusion=True, transition_fused_sm90a=True)
def path_d():
    with torch.no_grad(): return mod(x).view(M, D)
if not a.only_a: paths["D engine module (Triton)"] = path_d
with torch.no_grad():
    ref = transition_pytorch(x.float(), mod.ln_in.weight.float(), mod.ln_in.bias.float(), wa.float(), wb.float(), mod.squeeze.weight.float(), 4, eps).view(M, D)
def t(fn, reps=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s): fn()
    torch.cuda.synchronize(); o = []
    for _ in range(5):
        st, en = torch.cuda.Event(True), torch.cuda.Event(True); st.record()
        for _ in range(reps): g.replay()
        en.record(); torch.cuda.synchronize(); o.append(st.elapsed_time(en) * 1e3 / reps)
    return statistics.median(o)
floor = 24 * M * D * D / PEAK * 1e6
res = {}
for name, fn in paths.items():
    with torch.no_grad():
        y = fn().clone(); torch.cuda.synchronize()
        err = float((y.float() - ref).norm() / ref.norm())
        res[name] = (t(fn), err)
eng = res["D engine module (Triton)"][0] if "D engine module (Triton)" in res else 1.0
print(f"D{D} L{L} (forward floor {floor:.0f} us)")
for name, (us, err) in res.items():
    print(f"  {name:48s} {us:8.1f} us  {100 * floor / us:5.1f} % floor  x{eng / us:4.2f} vs engine  rel {err:.3e}")
with torch.no_grad():
    print(f"    parts: ln_swiglu_gemm {t(lambda: kln((132, 1, 1), (384, 1, 1), *lnmaps, g32, b32, dummyb, dummy, dummy, int(M), int(H), float(eps), 0)):.1f} us"
          f" | squeeze_gemm {t(sq):.1f} us | Triton LN {t(lambda: _ln_fwd(xf, g32, b32, None, eps, False, key)):.1f} us")
