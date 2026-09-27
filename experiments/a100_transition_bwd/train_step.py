"""Training step (fwd + bwd) on one node: 3-kernel bwd vs PW + X with the forward saving xn / stats.  python train_step.py L..."""
import copy, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import transition_bwd_a100 as TB  # noqa: E402
from bench_common import graph_ms, rel_rms  # noqa: E402
TA = TB.TA
fext = TA.build(extra=sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
bext = TB.build(extra=["-DPW_XN"])
for L in map(int, sys.argv[1:sys.argv.index("--")] if "--" in sys.argv else sys.argv[1:]):
    mod, x = TA.fixture(L)
    torch.manual_seed(7); dy = torch.randn_like(x)
    fpk, pk = TA.pack(mod), TB.pack(mod)
    T = x.shape[0]
    out = torch.empty_like(x); st = torch.empty(T, 2, device=x.device); xn = torch.empty_like(x)
    m32 = copy.deepcopy(mod).float(); x32 = x.float().requires_grad_(True)
    ref = torch.autograd.grad(m32(x32), [x32, m32.ln_in.weight, m32.ln_in.bias, m32.expand_a.weight, m32.expand_b.weight, m32.squeeze.weight], dy.float())
    b3, bq = {}, {}
    f0 = lambda: TA.forward(fext, x, fpk, out)  # noqa: E731
    f1 = lambda: TA.forward(fext, x, fpk, out, stats=st, xn=xn)  # noqa: E731
    s3 = lambda: (f0(), TB.backward(bext, x, dy, pk, b3))  # noqa: E731
    sq = lambda: (f1(), TB.backward_pwx(bext, x, dy, pk, bq, xn=xn, stats=st))  # noqa: E731
    o0 = f0().clone(); f1(); torch.cuda.synchronize()
    xf = x.float(); ref_xn = ((xf - xf.mean(1, keepdim=True)) * torch.rsqrt(xf.var(1, unbiased=False, keepdim=True) + pk["eps"]) * pk["gamma"] + pk["beta"])
    got = sq()[1]; torch.cuda.synchronize()
    print(f"L{L}: fwd out same with saving: {torch.equal(o0, out)}  xn rel {rel_rms(xn, ref_xn):.2e}  grads rel " + " ".join(f"{rel_rms(g, r):.2e}" for g, r in zip(got, ref)), flush=True)
    for k in range(2):
        t = {n: graph_ms(f)[0] * 1e3 for n, f in [("fwd", f0), ("fwd+save", f1), ("step 3k", s3), ("step pwx", sq)]}
        print(f"L{L} round {k}: " + "  ".join(f"{n} {v:.1f}" for n, v in t.items()) + " us", flush=True)
