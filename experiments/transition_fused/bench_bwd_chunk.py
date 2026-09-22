"""Wide-D backward run in ROW CHUNKS so one chunk's intermediates (h: C x H, dAB: C x 2H, bf16, reused buffers) stay in L2:
gate_gemm2 -> addmm(fp32) dWs += dy_c^T h, dWab += dAB^T xn_c -> d_xn + LN bwd per chunk.  Everything captured in one CUDA graph.
   python bench_bwd_chunk.py --width 256 --length 384 --chunks 0,32768,16384,8192   (0 = unchunked)"""
import argparse, copy, statistics, sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drv  # noqa: E402
p = argparse.ArgumentParser(); p.add_argument("--width", type=int, default=256); p.add_argument("--length", type=int, default=384)
p.add_argument("--chunks", default="0,32768,16384,8192"); p.add_argument("--gate", default=str(HERE / "build/gg2_k32s4h_w0.cubin"))
p.add_argument("--dxln", default=""); a = p.parse_args()
from miniworld_engine import settings
from miniworld_engine.modules import Transition
from miniworld_engine.kernels.transition.triton.fused import _transition_ln_bwd
D, L = a.width, a.length; H, M = 4 * D, L * L; bf = torch.bfloat16
settings.configure(engine_backend="triton", transition_residual_fusion=True, transition_fused_sm90a=True)
torch.manual_seed(0)
m = Transition(D, n=4, implementation="triton").cuda().bfloat16()
with torch.no_grad():
    for prm in m.parameters():
        if prm.ndim == 2: prm.normal_(std=prm.shape[-1] ** -0.5)
        elif prm is m.ln_in.weight: prm.copy_(1 + 0.2 * torch.randn_like(prm))
        else: prm.normal_(std=0.2)
x = torch.randn(1, L, L, D, device="cuda", dtype=bf, requires_grad=True); dy = torch.randn_like(x)
gamma, beta = m.ln_in.weight, m.ln_in.bias
wa, wb, ws = m.expand_a.weight, m.expand_b.weight, m.squeeze.weight
ref = copy.deepcopy(m).float(); xr = x.detach().float().requires_grad_(True); ref(xr).backward(dy.float())
R = [xr.grad.reshape(M, D), ref.ln_in.weight.grad, ref.ln_in.bias.grad, ref.expand_a.weight.grad, ref.expand_b.weight.grad, ref.squeeze.weight.grad]
del ref, xr
x2 = x.detach().reshape(M, D); go = dy.reshape(M, D)
with torch.no_grad():
    xf = x2.float(); mu = xf.mean(-1); rstd = torch.rsqrt(xf.var(-1, unbiased=False) + 1e-5); c1 = (mu * rstd).contiguous()
    xn = ((xf * rstd[:, None] - c1[:, None]) * gamma.float() + beta.float()).to(bf).contiguous()
    del xf
w1p = torch.stack([wa.detach().view(H // 128, 128, D), wb.detach().view(H // 128, 128, D)], 1).reshape(2 * H, D).contiguous()
wst = ws.detach().t().contiguous(); w_ab = torch.cat((wa.detach(), wb.detach()), 0); wabT = w_ab.t().contiguous()
g32 = gamma.detach().float().contiguous()
kg = drv.Kernel(a.gate, "gate_gemm", 231424)
kd = drv.Kernel(a.dxln, "dxn_lnbwd", 231424) if a.dxln else None
t32 = lambda t, dims, stride, box: drv.TensorMap(t, dims=dims, stride_bytes=stride, box=box, swizzle=64)
tm = lambda t, dims, stride, box: drv.TensorMap(t, dims=dims, stride_bytes=stride, box=box)
rel = lambda u, r: float((u.float().reshape(-1) - r.reshape(-1)).norm() / r.norm())

def build(C):
    """Everything for chunk size C (C = M: unchunked). Returns the step function; all buffers and descriptors are bound here."""
    n = M // C
    h = torch.empty(C, H, device="cuda", dtype=bf); dab = torch.empty(C, 2 * H, device="cuda", dtype=bf)
    dWs32 = torch.empty(D, H, device="cuda"); dWab32 = torch.empty(2 * H, D, device="cuda")
    dx = torch.empty(M, D, device="cuda", dtype=bf); dg = torch.empty(D, device="cuda"); db = torch.empty(D, device="cuda")
    pdg = torch.empty(n, 132, D, device="cuda"); pdb = torch.empty(n, 132, D, device="cuda")
    mH, mDAB = tm(h, [H, C], H * 2, [64, 64]), tm(dab, [2 * H, C], 2 * H * 2, [64, 64])
    mW1, mWS = t32(w1p, [D, 2 * H], D * 2, [32, 128]), t32(wst, [D, H], D * 2, [32, 128])
    per = []
    for i in range(n):
        r = slice(i * C, (i + 1) * C)
        per.append(dict(r=r, mXN=t32(xn[r], [D, C], D * 2, [32, 64]), mDY=t32(go[r], [D, C], D * 2, [32, 64]),
                        mA2=tm(dab, [2 * H, C], 2 * H * 2, [64, 64]) if kd else None))
    mB2 = tm(wabT, [2 * H, D], 2 * H * 2, [64, D // 2]) if kd else None
    def step():
        for i, q in enumerate(per):
            r = q["r"]
            kg((132, 1, 1), (384, 1, 1), q["mXN"], q["mDY"], mW1, mWS, mH, mDAB, int(C), int(D), int(H))
            if i == 0:
                torch.mm(go[r].t(), h, out_dtype=torch.float32, out=dWs32); torch.mm(dab.t(), xn[r], out_dtype=torch.float32, out=dWab32)
            else:
                torch.addmm(dWs32, go[r].t(), h, out_dtype=torch.float32, out=dWs32); torch.addmm(dWab32, dab.t(), xn[r], out_dtype=torch.float32, out=dWab32)
            if kd:
                kd((132, 1, 1), (384, 1, 1), q["mA2"], mB2, x2[r], go[r], dx[r], g32, rstd[r], c1[r], pdg[i], pdb[i], int(C))
            else:
                dxc, dgc, dbc = _transition_ln_bwd(dab @ w_ab, x2[r], rstd[r], c1[r], gamma.detach())
                dx[r].copy_(dxc.add_(go[r])); pdg[i, 0].copy_(dgc); pdb[i, 0].copy_(dbc); pdg[i, 1:].zero_(); pdb[i, 1:].zero_()
        torch.sum(pdg, (0, 1), out=dg); torch.sum(pdb, (0, 1), out=db)
        return dx, dg, db, dWab32[:H], dWab32[H:], dWs32
    return step
def graphed(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return g.replay
def t(fn, reps=5):
    for _ in range(2): fn()
    torch.cuda.synchronize(); o = []
    for _ in range(5):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
        for _ in range(reps): fn()
        e.record(); torch.cuda.synchronize(); o.append(s.elapsed_time(e) * 1e3 / reps)
    return statistics.median(o)
P = (gamma, beta, wa, wb, ws)
def eng_step():
    for q in (x,) + P: q.grad = None
    m(x).backward(dy)
ef, efb = t(graphed(lambda: m(x))), t(graphed(eng_step))
print(f"D{D} L{L}: engine module (CUDA graph) fwd {ef:.1f}  fwd+bwd {efb:.1f}  -> bwd {efb - ef:.1f} us")
for C in [int(c) or M for c in a.chunks.split(",")]:
    if M % C or C % 256: print(f"  chunk {C}: skipped (M % C)"); continue
    step = build(C); O = step(); torch.cuda.synchronize()
    errs = "  ".join(f"{n} {rel(o, r):.2e}" for n, o, r in zip(("dx", "dg", "db", "dWa", "dWb", "dWs"), O, R))
    us = t(graphed(step))
    print(f"  chunk {C:6d} ({M // C:3d} chunks, h+dAB {3 * C * H * 2 / 1e6:6.1f} MB): bwd {us:8.1f} us  x{(efb - ef) / us:.2f}  | {errs}")
    del step; torch.cuda.empty_cache()
