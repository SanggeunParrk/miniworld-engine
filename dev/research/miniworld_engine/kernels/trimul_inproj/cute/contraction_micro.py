"""Micro-bench the contraction bmm (fwd+bwd) under different left/right LAYOUTS to find
why ours (~3.3ms/bmm) is slower than dtv1's. (128, L, L) bf16. COMPUTE NODE."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_src = _Path(__file__).resolve()
while _src.name != "src" and _src.parent != _src:
    _src = _src.parent
if str(_src) not in _sys.path:
    _sys.path.insert(0, str(_src))

import torch
import triton

D = 128


def bench(fn):
    try:
        for _ in range(3):
            fn()
        return triton.testing.do_bench(fn, warmup=10, rep=50, return_mode="median")
    except Exception as e:  # noqa: BLE001
        return float("nan")


def run(tag, make_lr, L):
    left, right = make_lr(L)
    g = torch.randn(D, L, L, device="cuda", dtype=torch.bfloat16)

    def fwd():
        torch.bmm(left, right.transpose(1, 2))

    def fwdbwd():
        l = left.detach().requires_grad_(True)
        r = right.detach().requires_grad_(True)
        o = torch.bmm(l, r.transpose(1, 2))
        o.backward(g)
    tf = bench(fwd)
    tfb = bench(fwdbwd)
    print(f"  [{tag:28}] L={L}: bmm fwd={tf:.3f}  fwd+bwd={tfb:.3f}  "
          f"left.contig={left.is_contiguous()} off={left.storage_offset()}/{right.storage_offset()}",
          flush=True)


def main():
    assert torch.cuda.is_available()
    print(f"contraction micro on {torch.cuda.get_device_name(0)}", flush=True)
    for L in (512, 1024):
        # (a) fresh contiguous left/right (dtv1-like)
        def fresh(L):
            return (torch.randn(D, L, L, device="cuda", dtype=torch.bfloat16),
                    torch.randn(D, L, L, device="cuda", dtype=torch.bfloat16))
        # (b) sliced from a shared (1,2D,L,L) buffer (OURS: lr[:, :D] / lr[:, D:])
        def sliced(L):
            lr = torch.randn(1, 2 * D, L, L, device="cuda", dtype=torch.bfloat16)
            return lr[:, :D].reshape(D, L, L), lr[:, D:].reshape(D, L, L)
        # (c) sliced then .contiguous() (kill the storage offset)
        def sliced_contig(L):
            lr = torch.randn(1, 2 * D, L, L, device="cuda", dtype=torch.bfloat16)
            return lr[:, :D].reshape(D, L, L).contiguous(), lr[:, D:].reshape(D, L, L).contiguous()
        run("fresh-contiguous", fresh, L)
        run("ours-sliced", sliced, L)
        run("sliced+contiguous", sliced_contig, L)


if __name__ == "__main__":
    main()
