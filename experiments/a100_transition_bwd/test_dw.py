"""The recomputing DW role alone: dWa, dWb, dWs against fp32 autograd, and its time.  python test_dw.py 384 768"""
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transition_bwd_a100 as TB  # noqa: E402
from bench_common import graph_ms, rel_rms  # noqa: E402

import os
extra = os.environ.get("DW_EXTRA", "").split()
ext = TB.build(extra=extra)
print("build", extra)
for L in map(int, sys.argv[1:]):
    mod, x = TB.TA.fixture(L)
    torch.manual_seed(7)
    dy = torch.randn_like(x)
    m32 = copy.deepcopy(mod).float()
    x32 = x.float().requires_grad_(True)
    ref = torch.autograd.grad(m32(x32), [m32.expand_a.weight, m32.expand_b.weight, m32.squeeze.weight], dy.float())
    pk = TB.pack(mod)
    xn = torch.nn.functional.layer_norm(x.float(), (128,), mod.ln_in.weight.float(), mod.ln_in.bias.float(), mod.ln_in.eps).bfloat16()
    nrep = 13
    part = torch.empty(nrep, 3, 512, 128, device="cuda")
    fn = lambda: ext.bwd_dw(xn, dy, pk["wdw"], pk["gamma"], pk["beta"], part, pk["eps"])  # noqa: E731
    fn()
    s = part.sum(0)
    half = 0.5 if os.environ.get("DW_HALF_DA") else 1.0
    got = (half * s[0], s[1], s[2].t())            # the kernel accumulates 2 dA
    ms = graph_ms(fn)[0]
    flop = 12 * x.shape[0] * 128 * 512
    print(f"L{L}: DW {ms*1e3:.1f} us  ({flop / ms / 1e9:.1f} TFLOP/s of its 12 MDH)  rel dWa {rel_rms(got[0], ref[0]):.2e} "
          f"dWb {rel_rms(got[1], ref[1]):.2e} dWs {rel_rms(got[2], ref[2]):.2e}", flush=True)
