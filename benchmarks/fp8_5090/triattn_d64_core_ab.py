"""Core only, 64-channel heads (Chai-1 trunk: d_pair 256 = 4 x 64): the default (sm80_core2 rf core + gate_rows) vs the e4m3 cores (PV; PV + QK) with their quant pass;
same process, alternating, CUDA graph, 128 MB L2 flush; error of the gated output against float64 (L <= 256) or against the default.
python triattn_d64_core_ab.py [--rounds N] [--heads H] L ..."""
import math
import sys

import torch

from miniworld_engine.kernels.triangle_attention.cuda import sm80_core2 as c2
from miniworld_engine.kernels.triangle_attention.cuda import sm80_wide as W
from miniworld_engine.kernels.triangle_attention.cuda import sm120_fp8 as F

args = sys.argv[1:]
rounds, H = 6, 4
for flag in ("--rounds", "--heads"):
    if flag in args:
        i = args.index(flag)
        val = int(args[i + 1])
        del args[i : i + 2]
        rounds, H = (val, H) if flag == "--rounds" else (rounds, val)
HD = 64
flush = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")


def bench(f, n=2):
    f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            flush.zero_()
            f()
    gf = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gf):
        for _ in range(n):
            flush.zero_()

    def t(gr):
        gr.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(7):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            gr.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        return sorted(ts)[3]

    return (t(g) - t(gf)) / n * 1e3


def ref64(q, k, v, g, bias, L, scale):
    """float64 gated output [T, H HD] (pair row by pair row)."""
    out = torch.empty(L * L, H * HD, dtype=torch.float64, device="cuda")
    bb = bias[0].double()  # [H, L(q), L(k)]
    for i in range(L):
        sl = slice(i * L, (i + 1) * L)
        qi = q[sl].double().view(L, H, HD).transpose(0, 1)
        ki = k[sl].double().view(L, H, HD).transpose(0, 1)
        vi = v[sl].double().view(L, H, HD).transpose(0, 1)
        s = scale * qi @ ki.transpose(1, 2) + bb
        o = torch.softmax(s, -1) @ vi  # [H, L, HD]
        out[sl] = torch.sigmoid(g[sl].double()) * o.transpose(0, 1).reshape(L, H * HD)
    return out


print("heads", H, "x", HD, flush=True)
for L in map(int, args):
    torch.manual_seed(0)
    T, dh = L * L, H * HD
    n = 4 * dh + 8
    y = torch.randn(T, n, device="cuda", dtype=torch.bfloat16)
    q, k, v, g = (y[:, i * dh : (i + 1) * dh] for i in range(4))
    bias = (2.0 * torch.randn(1, H, L, L, device="cuda")).to(torch.bfloat16)
    scale = 1.0 / math.sqrt(HD)
    scl = 1.4426950408889634 * scale
    rows = W._rows()
    o = torch.empty(T, dh, device="cuda", dtype=torch.bfloat16)
    a0, a1, a2 = (torch.empty(T, dh, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    lay_y, lay_o = c2.token_major(L, n, HD), c2.token_major(L, dh, HD)
    u8 = torch.empty(0, dtype=torch.uint8, device="cuda")
    lse = torch.empty(0, dtype=torch.float32, device="cuda")

    def d():
        c2.forward(q, k, v, bias, u8, o, lse, (lay_y, lay_y, lay_y, lay_o), L, H, 1, HD, scale)
        rows.gate_rows(o, g, a0)

    f8 = F._ext()
    vinv = torch.full((H,), 64.0, device="cuda")
    vsc = 1 / vinv
    ks = torch.full((H,), 1.0, device="cuda")
    qs = 1 / ks
    v8 = torch.empty(T * dh, dtype=torch.uint8, device="cuda")
    k8 = torch.empty_like(v8)
    e8 = v8.new_empty((0,))
    nul = torch.empty(0, dtype=torch.float32, device="cuda")
    peak = torch.zeros(4, dtype=torch.int32, device="cuda")

    def pv():
        f8.quant_hd(v, vinv, v8, v, nul, e8, peak, L, H, L, n, n, HD)
        f8.attn_f8_gate_hd(q, k, v8, vsc, e8, nul, peak, bias, g, a1, L, H, 1, n, n, n, dh, scl, u8, HD)

    def qk():
        f8.quant_hd(v, vinv, v8, k, ks, k8, peak, L, H, L, n, n, HD)
        f8.attn_f8_gate_hd(q, k, v8, vsc, k8, qs, peak, bias, g, a2, L, H, 1, n, n, n, dh, scl, u8, HD)

    def core_pv():
        f8.attn_f8_gate_hd(q, k, v8, vsc, e8, nul, peak, bias, g, a1, L, H, 1, n, n, n, dh, scl, u8, HD)

    def core_qk():
        f8.attn_f8_gate_hd(q, k, v8, vsc, k8, qs, peak, bias, g, a2, L, H, 1, n, n, n, dh, scl, u8, HD)

    d(), pv(), qk()
    torch.cuda.synchronize()
    if L <= 256:
        r = ref64(q, k, v, g, bias, L, scale)
    else:
        r = a0.double()
    nr = r.pow(2).mean().sqrt()

    def err(a):
        e = a.double() - r
        return (e.pow(2).mean().sqrt() / nr).item(), (e.abs().max() / r.abs().max()).item()

    errs = {nm: err(a) for nm, a in (("default", a0), ("pv", a1), ("pvqk", a2))}
    ts = {nm: [] for nm in ("default", "pv", "pvqk", "core_pv", "core_qk")}
    fns = {"default": d, "pv": pv, "pvqk": qk, "core_pv": core_pv, "core_qk": core_qk}
    for _ in range(rounds):
        for nm, f in fns.items():
            ts[nm].append(bench(f))
    med = {nm: sorted(t)[len(t) // 2] for nm, t in ts.items()}
    ref_nm = "fp64" if L <= 256 else "default"
    print(
        f"L{L} err vs {ref_nm} (rms / max): "
        + ", ".join(f"{nm} {e[0]:.3e} / {e[1]:.3e}" for nm, e in errs.items())
        + f" | peak {peak.tolist()} | us: "
        + ", ".join(f"{nm} {t:.0f}" for nm, t in med.items())
        + f" | default / pv {med['default'] / med['pv']:.3f}, default / pvqk {med['default'] / med['pvqk']:.3f}",
        flush=True,
    )
