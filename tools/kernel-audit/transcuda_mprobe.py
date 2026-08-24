"""Are transition_bwd_cuda's weight gradients worse specifically at a non-multiple M?

The checker gave dwa/dwb ~4.5e-03 at M=4096 and ~1.4e-02 at M=4093 -- 3x, while dws and dx did
not move. Under the 5e-02 threshold either way, so the sweep passes it; that is exactly the kind
of "in band, therefore fine" reasoning that hid three bugs in this repo.

Two hypotheses, and one sweep separates them:
  (a) benign: the wgrad reduces over M rows in bf16, so the error tracks M and the reduction
      order, not divisibility. Then 4090 (also non-multiple) sits near 4096, and 2048 is lower.
  (b) a tail-handling defect: then every non-multiple M is elevated and every multiple is not.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "src")

import torch


def one(m: int, k: int, seed: int = 1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    from miniworld_engine.kernels.drivers.transition import _CUDA_N, _transition_cuda_ext
    ext = _transition_cuda_ext()
    nk = _CUDA_N * k
    dev = torch.device("cuda")
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16).contiguous()
    wa = torch.randn(nk, k, device=dev, dtype=torch.bfloat16).contiguous()
    wb = torch.randn(nk, k, device=dev, dtype=torch.bfloat16).contiguous()
    ws = torch.randn(k, nk, device=dev, dtype=torch.bfloat16).contiguous()
    g = torch.randn_like(x).contiguous()
    got = ext.backward(g, x, wa, wb, ws, _CUDA_N)

    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        xf = x.float().requires_grad_(True)
        af = wa.float().requires_grad_(True)
        bf = wb.float().requires_grad_(True)
        sf = ws.float().requires_grad_(True)
        (torch.nn.functional.silu(xf @ af.t()) * (xf @ bf.t()) @ sf.t()).backward(g.float())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev

    # rel = max|a-e| / max|e|, so an elevated rel can come from a bigger numerator OR a smaller
    # denominator. K=125 and K=127 were 3x while K=253 -- the same residue mod 8 -- was clean, so
    # no divisibility rule explains it. Report both halves and settle which one moves.
    out = {}
    for name, a, e in (("dx", got[0], xf.grad), ("dwa", got[1], af.grad),
                       ("dwb", got[2], bf.grad), ("dws", got[3], sf.grad)):
        num = (a.float() - e).abs().max().item()
        den = e.abs().max().item()
        out[name] = (num / (den or 1.0), num, den)
    return out


def main() -> int:
    # The first version of this probe swept M at a fixed K=128 and found nothing, because the
    # elevated run had K=125 as well -- K is the feature width AND the GEMM contraction extent,
    # and it was the variable not being varied. Sweep both.
    print(f"device={torch.cuda.get_device_name()}")
    print(f"{'K':>5} {'seed':>5} | {'dwa rel':>10} {'dwa |err|':>10} {'dwa max|e|':>11} "
          f"| {'dwb rel':>10} {'dwb |err|':>10} {'dwb max|e|':>11}")
    for k in (128, 125, 127, 253):
        for seed in (1234, 4321):
            r = one(4093, k, seed)
            a, b = r["dwa"], r["dwb"]
            print(f"{k:>5} {seed:>5} | {a[0]:>10.3e} {a[1]:>10.3e} {a[2]:>11.4f} "
                  f"| {b[0]:>10.3e} {b[1]:>10.3e} {b[2]:>11.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
