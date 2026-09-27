"""sm100 TriangleAttention module kernels vs torch: correctness + sustained timing.  python t_trimod.py [L]"""
import os, sys, pathlib, torch
from torch.utils.cpp_extension import load
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from energy_sol import sustained
src = pathlib.Path(__file__).parent.parent / "src/miniworld_engine/integrations/csrc/sm100"
d = pathlib.Path(os.environ["MINIWORLD_ENGINE_JIT_ROOT"]) / "trimod"; d.mkdir(parents=True, exist_ok=True)
ext = load("trimod", [str(src / "triattn_mod_sm100.cu")], extra_include_paths=[str(src)], build_directory=str(d),
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "--use_fast_math"])
L = int(sys.argv[1]) if len(sys.argv) > 1 else 384
B, C, H = 1, 128, 4
g = torch.Generator(device="cuda").manual_seed(0)
x = torch.randn(B, L, L, C, device="cuda", generator=g).to(torch.bfloat16)
lnw = 1 + 0.1 * torch.randn(C, device="cuda", generator=g); lnb = 0.1 * torch.randn(C, device="cuda", generator=g)
w4 = (torch.randn(4 * C, C, device="cuda", generator=g) * C ** -0.5).to(torch.bfloat16)
wb = (torch.randn(H, C, device="cuda", generator=g) * C ** -0.5).to(torch.bfloat16)
mask = torch.rand(B, L, device="cuda", generator=g) > 0.1
q, k, v, gg, bias = ext.tri_front(x.view(-1, C), lnw, lnb, 1e-5, w4, wb, mask, B, L)
y = torch.nn.functional.layer_norm(x.float(), (C,), lnw, lnb, 1e-5).to(torch.bfloat16)
ref = (y.float() @ w4.float().t()).view(-1, 4, C)
rb = (y.float() @ wb.float().t()).permute(0, 3, 1, 2)
rb = rb.masked_fill(~mask[:, None, None, :], float("-inf"))
rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
for i, (n, t) in enumerate(zip("qkvg", (q, k, v, gg))): print(f"front {n}: rel err {rel(t, ref[:, i]):.2e}")
fin = torch.isfinite(rb)
print(f"front bias: rel err {rel(bias.float()[fin], rb[fin]):.2e}  masked ok {bool((bias.float()[~fin] < -1e38).all())}")
def torch_front():
    yy = torch.nn.functional.layer_norm(x, (C,), lnw.to(torch.bfloat16), lnb.to(torch.bfloat16), 1e-5)
    o = yy @ w4.t(); b_ = (yy @ wb.t()).permute(0, 3, 1, 2).masked_fill(~mask[:, None, None, :], float("-inf")).contiguous()
    return o, b_
for name, fn in (("torch", torch_front), ("ours", lambda: ext.tri_front(x.view(-1, C), lnw, lnb, 1e-5, w4, wb, mask, B, L))):
    r = sustained(fn, secs=2.0)
    print(f"front {name:6s} {r['ms']*1e3:8.1f} us  {r['J']*1e3:7.2f} mJ  {r['W']:5.0f} W", flush=True)
# ---- tail
o_ = torch.randn(B * L * L, C, device="cuda", generator=g).to(torch.bfloat16)
g_ = torch.randn(B * L * L, C, device="cuda", generator=g).to(torch.bfloat16)
wo = (torch.randn(C, C, device="cuda", generator=g) * C ** -0.5).to(torch.bfloat16)
xf = x.view(-1, C)
yt = ext.tri_tail(g_, o_, xf, wo)
u = (torch.sigmoid(g_.float()) * o_.float()).to(torch.bfloat16)
yr = xf.float() + u.float() @ wo.float().t()
print(f"tail: rel err {rel(yt - xf, yr - xf.float()):.2e}")
for name, fn in (("torch", lambda: xf + (torch.sigmoid(g_) * o_) @ wo.t()), ("ours", lambda: ext.tri_tail(g_, o_, xf, wo))):
    r = sustained(fn, secs=2.0)
    print(f"tail {name:6s} {r['ms']*1e3:8.1f} us  {r['J']*1e3:7.2f} mJ  {r['W']:5.0f} W", flush=True)
# ---- gate backward
dy = torch.randn(B * L * L, C, device="cuda", generator=g).to(torch.bfloat16)
do_, dg_, delta, dwo, seed = ext.tri_gate_bwd(dy, g_, o_, wo, B, L)
du = dy.float() @ wo.float()
sg = torch.sigmoid(g_.float())
do_r = du * sg; dg_r = du * o_.float() * sg * (1 - sg)
de_r = (do_r * o_.float()).view(B, L, L, 4, 32).sum(-1).permute(0, 1, 3, 2)
dwo_r = dy.float().t() @ (sg * o_.float()).to(torch.bfloat16).float()
print(f"gate_bwd: do {rel(do_, do_r):.2e}  dg {rel(dg_, dg_r):.2e}  delta {rel(delta, de_r):.2e}  dWo {rel(dwo, dwo_r):.2e}  seed exact {bool(torch.equal(seed, dy))}")
def torch_gate_bwd():
    du_ = dy @ wo; s_ = torch.sigmoid(g_); d_ = du_ * s_
    return d_, du_ * o_ * s_ * (1 - s_), (d_.float() * o_.float()).view(B, L, L, 4, 32).sum(-1), dy.t() @ (s_ * o_)
for name, fn in (("torch", torch_gate_bwd), ("ours", lambda: ext.tri_gate_bwd(dy, g_, o_, wo, B, L))):
    r = sustained(fn, secs=2.0)
    print(f"gate_bwd {name:6s} {r['ms']*1e3:8.1f} us  {r['J']*1e3:7.2f} mJ  {r['W']:5.0f} W", flush=True)
# ---- head backward
d4 = [torch.randn(B * L * L, C, device="cuda", generator=g).to(torch.bfloat16) * 0.1 for _ in range(4)]
dbias = torch.randn(B, 4, L, L, device="cuda", generator=g) * 0.1
dres = torch.randn(B * L * L, C, device="cuda", generator=g).to(torch.bfloat16)
xr_ = x.view(-1, C).float().requires_grad_(True)
yy = torch.nn.functional.layer_norm(xr_, (C,), lnw, lnb, 1e-5)
proj = torch.cat([yy @ w4[i * C:(i + 1) * C].float().t() for i in range(4)], -1)
bp = yy @ wb.float().t()
loss = (proj * torch.cat([t.float() for t in d4], -1)).sum() + (bp * dbias.permute(0, 2, 3, 1).reshape(-1, 4)).sum()
gx, = torch.autograd.grad(loss, [xr_])
ref_dp = dres.float() + gx
lw_ = lnw.clone().requires_grad_(True); lb_ = lnb.clone().requires_grad_(True)
yy2 = torch.nn.functional.layer_norm(x.view(-1, C).float(), (C,), lw_, lb_, 1e-5)
l2 = (torch.cat([yy2 @ w4[i * C:(i + 1) * C].float().t() for i in range(4)], -1) * torch.cat([t.float() for t in d4], -1)).sum() + ((yy2 @ wb.float().t()) * dbias.permute(0, 2, 3, 1).reshape(-1, 4)).sum()
dgam_r, dbet_r = torch.autograd.grad(l2, [lw_, lb_])
dp = dres.clone()
ext.tri_head_bwd(*d4, dbias, x.view(-1, C), w4, wb, lnw, lnb, 1e-5, dp, B, L)
mt, cs, bh, cdb = ext.tri_wgrad(*d4, dbias, x.view(-1, C), 1e-5, B, L, w4, wb, lnw, lnb, False, torch.empty(0, device="cuda"))
dwf, dgbf, _ = ext.tri_wgrad(*d4, dbias, x.view(-1, C), 1e-5, B, L, w4, wb, lnw, lnb, True, torch.empty(0, device="cuda"))
dW = [(lnw[:, None] * mt[t] + lnb[:, None] * cs[t][None, :]).t() for t in range(4)]          # dW_t [o, c]
dgam = sum((w4[t * C:(t + 1) * C].float().t() * mt[t]).sum(1) for t in range(4)) + (wb.float() * bh).sum(0)
dbet = sum(cs[t] @ w4[t * C:(t + 1) * C].float() for t in range(4)) + cdb @ wb.float()
dwb = lnw[None, :] * bh + cdb[:, None] * lnb[None, :]
xh = torch.nn.functional.layer_norm(x.view(-1, C).float(), (C,), None, None, 1e-5)
yf32 = xh * lnw + lnb
dw_r = [d4[i].float().t() @ yf32 for i in range(4)]
dwb_r = dbias.permute(1, 0, 2, 3).reshape(4, -1) @ yf32
print(f"head_bwd: dpair {rel(dp, ref_dp):.2e}  dgamma {rel(dgam, dgam_r):.2e}  dbeta {rel(dbet, dbet_r):.2e}  dWb {rel(dwb, dwb_r):.2e}  "
      + "  ".join(f"dW{n} {rel(dW[i], dw_r[i]):.2e}" for i, n in enumerate("qkvg")))
print(f"wgrad finish: dW {max(rel(dwf[i], dw_r[i]) for i in range(4)):.2e}  dgamma {rel(dgbf[0], dgam_r):.2e}  dbeta {rel(dgbf[1], dbet_r):.2e}  dWb {rel(dgbf[2:], dwb_r):.2e}")
def torch_head():
    xx = x.view(-1, C).requires_grad_(True)
    y3 = torch.nn.functional.layer_norm(xx, (C,), lnw.to(torch.bfloat16), lnb.to(torch.bfloat16), 1e-5)
    gy = sum(d4[i] @ w4[i * C:(i + 1) * C] for i in range(4)) + (dbias.permute(0, 2, 3, 1).reshape(-1, 4).to(torch.bfloat16) @ wb)
    return torch.autograd.grad(y3, [xx], gy)[0] + dres, y3
for name, fn in (("torch", torch_head), ("ours head", lambda: ext.tri_head_bwd(*d4, dbias, x.view(-1, C), w4, wb, lnw, lnb, 1e-5, dp, B, L)),
                 ("ours wgrad", lambda: ext.tri_wgrad(*d4, dbias, x.view(-1, C), 1e-5, B, L, w4, wb, lnw, lnb, True, torch.empty(0, device="cuda")))):
    r = sustained(fn, secs=2.0)
    print(f"head_bwd {name:10s} {r['ms']*1e3:8.1f} us  {r['J']*1e3:7.2f} mJ  {r['W']:5.0f} W", flush=True)
