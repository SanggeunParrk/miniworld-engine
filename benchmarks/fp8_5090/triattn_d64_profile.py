"""Kernel breakdown of the default Chai-1 TriAttn (64-channel heads) inference on the 5090: python triattn_d64_profile.py [--model chai1|chai1_confidence] [--ending] [--mask] D,H,L ...
(D = d_pair, H heads of D / H channels; bf16 module as FoldJet runs it, key mask 1/8 masked with --mask)."""
import sys

import torch
from torch.profiler import ProfilerActivity, profile

import foldjet_engine  # noqa: F401
from foldjet_engine.modules.triangle import TriangleAttention

args = sys.argv[1:]
model = "chai1"
if "--model" in args:
    i = args.index("--model")
    model = args[i + 1]
    del args[i : i + 2]
starting = "--ending" not in args
use_mask = "--mask" in args
args = [a for a in args if a not in ("--ending", "--mask")]
for c in args:
    d, h, L = map(int, c.split(","))
    torch.manual_seed(0)
    m = TriangleAttention(d, h, model=model, starting=starting, p_drop=0.0).cuda().eval().to(torch.bfloat16)
    m.set_backend("cuda")
    x = torch.randn(1, L, L, d, device="cuda", dtype=torch.bfloat16)
    mk = torch.ones(1, L, device="cuda", dtype=torch.bool)
    if use_mask:
        mk[:, L - L // 8 :] = False

    def f():
        with torch.no_grad():
            return m(x, mk)

    for _ in range(3):
        f()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(3):
            f()
        torch.cuda.synchronize()
    tot = sum(e.device_time_total for e in p.key_averages()) / 3
    print(f"== {model} {'start' if starting else 'end'} d{d} h{h} L{L} mask={use_mask}: total {tot:.0f} us / call")
    for e in sorted(p.key_averages(), key=lambda e: -e.device_time_total)[:10]:
        print(f"  {e.device_time_total / 3:9.0f} us {100 * e.device_time_total / 3 / tot:5.1f} %  {e.count // 3}x  {e.key[:110]}")
