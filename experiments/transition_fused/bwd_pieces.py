"""Where the wide-channel Transition backward's time goes when every piece is a plain cuBLAS / torch call (bf16 operands,
fp32 accumulation), per width.  The pieces a fused design would replace:
  gate    : ab = xn [Wa;Wb]^T, dh = dy Ws, h / dA / dB elementwise        (2 GEMMs + elementwise)
  dW      : dWa|dWb = [dA|dB]^T xn, dWs = dy^T h                            (2 GEMMs, K = M)
  dxn     : [dA|dB] [Wa;Wb]                                                 (1 GEMM, K = 2H)
  lnbwd   : LayerNorm backward + residual -> dx, dgamma, dbeta             (elementwise + row reductions)
  python bwd_pieces.py --width 256 --length 384
"""
import argparse, statistics, torch, torch.nn.functional as F
p = argparse.ArgumentParser(); p.add_argument("--width", type=int, default=256); p.add_argument("--length", type=int, default=384); a = p.parse_args()
D, H, M = a.width, 4 * a.width, a.length ** 2
PEAK = 989e12
torch.manual_seed(1)
bf = torch.bfloat16
x = torch.randn(M, D, device="cuda", dtype=bf); dy = torch.randn(M, D, device="cuda", dtype=bf)
g = (torch.rand(D, device="cuda") + 0.5); b = torch.randn(D, device="cuda") * 0.1
wab = (torch.randn(2 * H, D, device="cuda") * D ** -0.5).to(bf); ws = (torch.randn(D, H, device="cuda") * H ** -0.5).to(bf)
mean = x.float().mean(-1); rstd = torch.rsqrt(x.float().var(-1, unbiased=False) + 1e-5)
xn = ((x.float() - mean[:, None]) * rstd[:, None] * g + b).to(bf)
def t(fn, reps=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); o = []
    for _ in range(5):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
        for _ in range(reps): fn()
        e.record(); torch.cuda.synchronize(); o.append(s.elapsed_time(e) * 1e3 / reps)
    return statistics.median(o)
def gate():
    ab = xn @ wab.t(); dh = dy @ ws
    A, B = ab[:, :H].float(), ab[:, H:].float(); s = torch.sigmoid(A); l = A * s
    h = (l * B).to(bf); dA = (dh.float() * B * (s + l * (1 - s))).to(bf); dB = (dh.float() * l).to(bf)
    return h, dA, dB
h, dA, dB = gate(); dAB = torch.cat([dA, dB], 1)
def dW():
    return dAB.t() @ xn, dy.t() @ h
def dxn():
    return dAB @ wab
dn = dxn()
def lnbwd():
    xh = (x.float() - mean[:, None]) * rstd[:, None]; wdy = g * dn.float()
    ca = (xh * wdy).mean(-1, keepdim=True); cb = wdy.mean(-1, keepdim=True)
    dx = ((wdy - xh * ca - cb) * rstd[:, None]).to(bf) + dy
    return dx, (dn.float() * xh).sum(0), dn.float().sum(0)
fl = {"gate": 2 * M * D * 3 * H, "dW": 2 * M * D * 3 * H, "dxn": 2 * M * D * 2 * H, "lnbwd": 0}
tot = 0
print(f"D{D} L{a.length}: backward tensor floor {16 * M * D * H / PEAK * 1e6:.0f} us")
for n, fn in (("gate", gate), ("dW", dW), ("dxn", dxn), ("lnbwd", lnbwd)):
    us = t(fn); tot += us
    print(f"  {n:6s} {us:8.1f} us" + (f"   {100 * fl[n] / PEAK / (us * 1e-6):5.1f} % of tensor peak" if fl[n] else ""))
print(f"  total  {tot:8.1f} us")
