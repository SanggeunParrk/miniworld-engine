"""PW + X with xn / stats given (as if the forward saved them): accuracy + time.  python pw_xn.py L..."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import transition_bwd_a100 as TB  # noqa: E402
from bench_common import graph_ms, rel_rms  # noqa: E402
import copy
ext = TB.build(extra=["-DPW_XN"])
for L in map(int, sys.argv[1:]):
    mod, x = TB.TA.fixture(L)
    torch.manual_seed(7); dy = torch.randn_like(x)
    pk = TB.pack(mod)
    xf = x.float(); mean = xf.mean(1); rstd = torch.rsqrt(xf.var(1, unbiased=False) + pk["eps"])
    stats = torch.stack([mean, rstd], 1).contiguous()
    xn = ((xf - mean[:, None]) * rstd[:, None] * pk["gamma"] + pk["beta"]).to(torch.bfloat16).contiguous()
    m32 = copy.deepcopy(mod).float(); x32 = x.float().requires_grad_(True)
    ref = torch.autograd.grad(m32(x32), [x32, m32.ln_in.weight, m32.ln_in.bias, m32.expand_a.weight, m32.expand_b.weight, m32.squeeze.weight], dy.float())
    bufs = {}
    fn = lambda: TB.backward_pwx(ext, x, dy, pk, bufs, xn=xn, stats=stats)  # noqa: E731
    got = fn(); torch.cuda.synchronize()
    print(f"L{L}: pwx(xn given) {graph_ms(fn)[0]*1e3:.1f} us  rel " + " ".join(f"{rel_rms(g, r):.2e}" for g, r in zip(got, ref)), flush=True)
    fn3 = lambda: TB.backward(ext, x, dy, pk, bufs)  # noqa: E731
    fn3(); torch.cuda.synchronize()
    for k in range(2):
        print(f"L{L}: round {k}: 3k {graph_ms(fn3)[0]*1e3:.1f} us  pwx(xn given) {graph_ms(fn)[0]*1e3:.1f} us", flush=True)
