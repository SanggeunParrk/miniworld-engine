"""Input-projection dW accumulation error: B7 joint / B7src dW against an fp64 product of the same bf16 dgp and x_n (B7src's dgp;
the joint kernel computes the identical dgp tiles in its ring).   python dw_acc.py --length 384 768 [--extra -DB7J_DWSEG=2]"""
import argparse, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trimul_a100 as TA  # noqa: E402
import trimul_train as TT  # noqa: E402
from train_bench_fixture import make  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variant", nargs="+", default=["bidir", "single"])
ap.add_argument("--length", nargs="+", type=int, default=[384, 768])
ap.add_argument("--extra", nargs="*", default=[])
a = ap.parse_args()
ext = TA.build(extra=[x for e in a.extra for x in e.split()])
print("B7J_DWSEG", ext.b7j_dwseg())
for var in a.variant:
    for L in a.length:
        mod, z, mask, ds, dy = make(var, L)
        params = [dict(mod.named_parameters())[n] for n in TT.PARAMS]
        res = {}
        for joint in (False, True):
            TT._B7J = joint
            TT.DEBUG = {}
            y = TT.forward_train(ext, mod, z, mask, ds)
            torch.autograd.grad(y, [z] + params, dy)
            res[joint] = TT.DEBUG
        TT.DEBUG = None
        ref = res[False]["dgp"].double().t() @ res[False]["xn"].double()
        f32 = torch.mm(res[False]["dgp"].float().t(), res[False]["xn"].float())
        rel = lambda d: float((d.double() - ref).norm() / ref.norm())  # noqa: E731
        mx = lambda d: float((d.double() - ref).abs().max() / ref.abs().max())  # noqa: E731
        print(f"{var:6s} L{L}: tokens {L * L}  joint rel {rel(res[True]['dw_rows']):.3e} (max {mx(res[True]['dw_rows']):.2e})  "
              f"b7src rel {rel(res[False]['dw_rows']):.3e}  torch fp32 mm rel {rel(f32):.3e}", flush=True)
