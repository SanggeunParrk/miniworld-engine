"""Which rounding point makes the D64 fused backward's grads ~20 % less accurate than the engine module's? fp32 torch emulation."""
import copy, torch
from miniworld_engine import settings
from miniworld_engine.modules import Transition
torch.manual_seed(0); D, H, L = 64, 256, 384; M = L * L; bf = torch.bfloat16
settings.configure(engine_backend="triton", transition_residual_fusion=True, transition_fused_sm90a=True)
m = Transition(D, n=4, implementation="triton").cuda().bfloat16()
with torch.no_grad():
    for prm in m.parameters():
        if prm.ndim == 2: prm.normal_(std=prm.shape[-1] ** -0.5)
        elif prm is m.ln_in.weight: prm.copy_(1 + 0.2 * torch.randn_like(prm))
        else: prm.normal_(std=0.2)
x = torch.randn(1, L, L, D, device="cuda", dtype=bf, requires_grad=True); dy = torch.randn_like(x)
ref = copy.deepcopy(m).float(); xr = x.detach().float().requires_grad_(True); ref(xr).backward(dy.float())
R = dict(dWa=ref.expand_a.weight.grad, dWb=ref.expand_b.weight.grad, dWs=ref.squeeze.weight.grad)
m(x).backward(dy); E = dict(dWa=m.expand_a.weight.grad, dWb=m.expand_b.weight.grad, dWs=m.squeeze.weight.grad)
rel = lambda u, r: float((u.float() - r).norm() / r.norm())
print("engine module:", "  ".join(f"{k} {rel(E[k], R[k]):.2e}" for k in R))
g, b = m.ln_in.weight.float(), m.ln_in.bias.float(); wa, wb, ws = (w.float() for w in (m.expand_a.weight, m.expand_b.weight, m.squeeze.weight))
xf = x.detach().reshape(M, D).float(); go = dy.reshape(M, D).float()
xn32 = torch.nn.functional.layer_norm(xf, (D,), g, b, 1e-5); xn16 = xn32.to(bf).float()
r16 = lambda t: t.to(bf).float()
def run(tag, xn_gemm, xn_dw, round_dh, round_hab, round_ab=False):
    A = xn_gemm @ wa.t(); B = xn_gemm @ wb.t()
    if round_ab: A, B = r16(A), r16(B)
    dh = go @ ws; dh = r16(dh) if round_dh else dh
    s = torch.sigmoid(A); l = A * s
    h = l * B; dA = dh * B * (s + l * (1 - s)); dB = dh * l
    if round_hab: h, dA, dB = r16(h), r16(dA), r16(dB)
    O = dict(dWa=r16(dA.t() @ xn_dw), dWb=r16(dB.t() @ xn_dw), dWs=r16(go.t() @ h))
    print(f"{tag:44s}", "  ".join(f"{k} {rel(O[k], R[k]):.2e}" for k in R))
run("ours: xn bf16, dh bf16, h/dA/dB bf16", xn16, xn16, True, True)
run("xn fp32 in the a/b GEMM only", xn32, xn16, True, True)
run("xn fp32 in the dW GEMM only", xn16, xn32, True, True)
run("dh fp32", xn16, xn16, False, True)
run("h/dA/dB fp32", xn16, xn16, True, False)
run("a, b rounded to bf16", xn16, xn16, True, True, True)
run("everything fp32 except weights", xn32, xn32, False, False)
# ---- the kernel itself against the same contract, per hidden slice of 64 (DW-role slices)
import sys; sys.argv = [sys.argv[0], "--length", str(L)]
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("bm", str(Path(__file__).resolve().parent / "bench_mod_d64_lib.py"))
bm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bm)
P = (m.ln_in.weight, m.ln_in.bias, m.expand_a.weight, m.expand_b.weight, m.squeeze.weight)
for q in (x,) + P: q.grad = None
bm.Ours.apply(x, *P).backward(dy)
K = dict(dWa=m.expand_a.weight.grad, dWb=m.expand_b.weight.grad, dWs=m.squeeze.weight.grad, dx=x.grad)
A = xn16 @ wa.t(); B = xn16 @ wb.t(); dh = r16(go @ ws); s_ = torch.sigmoid(A); l = A * s_
h, dA, dB = r16(l * B), r16(dh * B * (s_ + l * (1 - s_))), r16(dh * l)
C = dict(dWa=r16(dA.t() @ xn16), dWb=r16(dB.t() @ xn16), dWs=r16(go.t() @ h))
print("kernel vs fp32:    ", "  ".join(f"{k} {rel(K[k], R[k]):.2e}" for k in R))
print("kernel vs contract:", "  ".join(f"{k} {rel(K[k], C[k]):.2e}" for k in C), "(engine vs contract:", "  ".join(f"{k} {rel(E[k], C[k]):.2e}" for k in C) + ")")
for k in ("dWa", "dWs"):
    kk, cc = (K[k], C[k]) if k != "dWs" else (K[k].t(), C[k].t())
    print(f"  {k} per 64-hidden slice kernel-vs-contract:", " ".join(f"{rel(kk[i*64:(i+1)*64], cc[i*64:(i+1)*64]):.1e}" for i in range(H // 64)))
    print(f"  {k} per 16-col group of D:", " ".join(f"{rel(kk[:, j*16:(j+1)*16], cc[:, j*16:(j+1)*16]):.1e}" for j in range(D // 16)))

def contract(xa=xn16, xw=xn16, rab=False, rdh=True, rl=False, xdy=go):
    A = xa @ wa.t(); B = xa @ wb.t()
    if rab: A, B = r16(A), r16(B)
    dh = xdy @ ws; dh = r16(dh) if rdh else dh
    s_ = torch.sigmoid(A); l = A * s_
    if rl: l = r16(l)
    h, dA, dB = r16(l * B), r16(dh * B * (s_ + l * (1 - s_))), r16(dh * l)
    return dict(dWa=r16(dA.t() @ xw), dWb=r16(dB.t() @ xw), dWs=r16(go.t() @ h))
for tag, kw in [("a,b bf16", dict(rab=True)), ("l bf16", dict(rl=True)), ("dh fp32", dict(rdh=False)),
                ("xn fp32 both", dict(xa=xn32, xw=xn32)), ("xn fp32 gemm", dict(xa=xn32)), ("xn fp32 dW", dict(xw=xn32))]:
    Cv = contract(**kw)
    print(f"kernel vs {tag:14s}", "  ".join(f"{k} {rel(K[k], Cv[k]):.2e}" for k in Cv))
Cc = contract()
for k in ("dWa", "dWb", "dWs"):
    kv, rv, cv = K[k].float().reshape(-1), R[k].reshape(-1), Cc[k].reshape(-1)
    al = float(kv @ rv / (rv @ rv)); ac = float(cv @ rv / (rv @ rv))
    print(f"{k}: kernel scale vs fp32 {al:.6f} (resid after scale {float((kv - al * rv).norm() / rv.norm()):.2e}) | contract scale {ac:.6f}")
# per-row-tile probe: dWs from one 128-row tile at a time is not observable; instead zero dy outside a row window and rerun
for lo, hi in ((0, 128), (0, 1024), (M - 1024, M)):
    dyw = torch.zeros_like(dy).reshape(M, D); dyw[lo:hi] = dy.reshape(M, D)[lo:hi]
    for q in (x,) + P: q.grad = None
    bm.Ours.apply(x, *P).backward(dyw.reshape(dy.shape))
    kw = m.squeeze.weight.grad.float()
    gw = dyw.float(); hh = r16(torch.nn.functional.silu(xn16 @ wa.t()) * (xn16 @ wb.t()))
    cw = r16(gw.t() @ hh)
    print(f"rows [{lo},{hi}): dWs kernel vs contract {rel(kw, cw):.2e}")
r, d = 5, 3
dy1 = torch.zeros_like(dy).reshape(M, D); dy1[r, d] = 1
for q in (x,) + P: q.grad = None
bm.Ours.apply(x, *P).backward(dy1.reshape(dy.shape))
hk = m.squeeze.weight.grad[d].float()                    # = kernel h[r, :]
Ar = xn16[r] @ wa.t(); Br = xn16[r] @ wb.t(); hc = r16(torch.nn.functional.silu(Ar) * Br)
print("one-hot probe: h kernel vs contract rel", rel(hk, hc), " exact matches", int((hk == hc).sum()), "/", H)
diff = (hk - hc).abs(); i = int(diff.argmax())
print("  worst unit", i, "kernel", float(hk[i]), "contract", float(hc[i]), "a", float(Ar[i]), "b", float(Br[i]))
print("  first 8 kernel  ", [round(float(v), 5) for v in hk[:8]])
print("  first 8 contract", [round(float(v), 5) for v in hc[:8]])
# is it h of a DIFFERENT row, or a different hidden permutation?
Hall = r16(torch.nn.functional.silu(xn16[:256] @ wa.t()) * (xn16[:256] @ wb.t()))
best = ((Hall - hk).norm(dim=1)).argmin(); print("  closest contract row among first 256:", int(best), "rel", rel(hk, Hall[best]))
xk = bm.Ours.last[0].float()
print("fwd-kernel xn vs torch xn16: mismatching elements", int((xk != xn16).sum()), "/", xk.numel(), " rel", rel(xk, xn16))
hk2 = r16(torch.nn.functional.silu(xk[r] @ wa.t()) * (xk[r] @ wb.t()))
print("h from the KERNEL's xn vs kernel h: rel", rel(hk, hk2), " exact", int((hk == hk2).sum()), "/", H)
print("gamma/beta dtypes", m.ln_in.weight.dtype, m.ln_in.bias.dtype)
