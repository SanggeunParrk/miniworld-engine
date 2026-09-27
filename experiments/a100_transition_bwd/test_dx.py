"""The recomputing DX role alone: dx, dgamma, dbeta against fp32 autograd, and its time.  python test_dx.py 384 768"""
import copy
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transition_bwd_a100 as TB  # noqa: E402
from bench_common import graph_ms, rel_rms  # noqa: E402

ext = TB.build(extra=os.environ.get("DX_EXTRA", "").split())
for L in map(int, sys.argv[1:]):
    mod, x = TB.TA.fixture(L)
    torch.manual_seed(7)
    dy = torch.randn_like(x)
    m32 = copy.deepcopy(mod).float()
    x32 = x.float().requires_grad_(True)
    ref = torch.autograd.grad(m32(x32), [x32, m32.ln_in.weight, m32.ln_in.bias], dy.float())
    pk = TB.pack(mod)
    xf = x.float()
    mean, var = xf.mean(1), xf.var(1, unbiased=False)
    rstd = torch.rsqrt(var + mod.ln_in.eps)
    stats = torch.stack([mean, rstd], 1).contiguous()
    xn = ((xf - mean[:, None]) * rstd[:, None] * mod.ln_in.weight.float() + mod.ln_in.bias.float()).bfloat16()
    dx = torch.empty_like(x)
    dgb = torch.empty(108, 256, device="cuda")
    fn = lambda: ext.bwd_dx(x, xn, dy, stats, pk["wdx"], pk["gamma"], dx, dgb)  # noqa: E731
    fn()
    g = dgb.view(-1, 2, 128).sum(0)
    ms = graph_ms(fn)[0]
    flop = 10 * x.shape[0] * 128 * 512
    print(f"L{L}: DX {ms*1e3:.1f} us  ({flop / ms / 1e9:.1f} TFLOP/s of its 10 MDH)  rel dx {rel_rms(dx, ref[0]):.2e} "
          f"dgamma {rel_rms(g[0], ref[1]):.2e} dbeta {rel_rms(g[1], ref[2]):.2e}", flush=True)
