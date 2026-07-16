"""Verify the TRITON bidirectional trimul generalised to B>1.

Three checks, all on the module's TRITON backend (== the live A100 path, since cute is
sm90+ only):
  1. Correctness vs the PYTORCH reference (same module class, shared weights): forward +
     input-grad + param-grad cosine, for B in {1,2,4}, full and partial masks.
  2. Batch independence: tri(pair)[b] must equal tri(pair[b:b+1]) — proves no cross-batch
     bleed in the batched grid / channel-outer preact.
  3. B==1 regression + scaling: fwd/bwd latency at B=1 vs B=2/4 (per-batch cost).

Run under `pixi run` on an A100. Exits non-zero if any correctness check fails.
"""

from __future__ import annotations

import sys
import time

import torch

from miniworld_kernels.modules.exceptions import ImplementationType
from miniworld_kernels.modules.triangle_multiplication import (
    BidirectionalTriangleMultiplication,
)

DEVICE = "cuda"
BF16 = torch.bfloat16


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-8).item()


def _build(d_pair: int) -> tuple[torch.nn.Module, torch.nn.Module]:
    ref = BidirectionalTriangleMultiplication(d_pair, implementation=ImplementationType.PYTORCH)
    tri = BidirectionalTriangleMultiplication(d_pair, implementation=ImplementationType.TRITON)
    # Randomize EVERY parameter (defaults zero-init the gates / to_out -> trivial 0 output).
    torch.manual_seed(0)
    for p in ref.parameters():
        p.data.normal_(0, 0.5)
    tri.load_state_dict(ref.state_dict())      # share weights exactly
    return ref.to(DEVICE, BF16).eval(), tri.to(DEVICE, BF16).eval()


def _mask(B: int, L: int, partial: bool) -> torch.Tensor:
    if not partial:
        return torch.ones(B, L, dtype=torch.bool, device=DEVICE)
    m = torch.ones(B, L, dtype=torch.bool, device=DEVICE)
    for b in range(B):                          # drop a different tail per batch row
        m[b, L - 1 - b:] = False
    return m


def _run_grad(mod, pair, mask):
    pair = pair.detach().clone().requires_grad_(True)
    y = mod(pair, mask)
    y.float().sum().backward()
    pgrads = {n: p.grad.detach().clone() for n, p in mod.named_parameters()}
    return y.detach(), pair.grad.detach(), pgrads


def check_correctness(d_pair: int) -> bool:
    ok = True
    ref, tri = _build(d_pair)
    for L in (128, 256):
        for B in (1, 2, 4):
            for partial in (False, True):
                torch.manual_seed(100 + B + L)
                pair = torch.randn(B, L, L, d_pair, device=DEVICE, dtype=BF16)
                mask = _mask(B, L, partial)
                yr, gr, pgr = _run_grad(ref, pair, mask)
                yt, gt, pgt = _run_grad(tri, pair, mask)
                cf = _cos(yt, yr)
                cg = _cos(gt, gr)
                cp = min(_cos(pgt[n], pgr[n]) for n in pgr if pgr[n].abs().sum() > 0)
                tag = "partial" if partial else "full"
                good = cf >= 0.99 and cg >= 0.99 and cp >= 0.99
                ok &= good
                print(f"  [{'OK' if good else 'FAIL'}] d={d_pair} L={L} B={B} mask={tag:7s}"
                      f"  fwd={cf:.5f} dx={cg:.5f} dW={cp:.5f}")
    return ok


def check_batch_independence(d_pair: int) -> bool:
    ref, tri = _build(d_pair)
    L, B = 192, 4
    torch.manual_seed(7)
    pair = torch.randn(B, L, L, d_pair, device=DEVICE, dtype=BF16)
    mask = _mask(B, L, partial=True)
    with torch.no_grad():
        y_full = tri(pair, mask)
        y_each = torch.cat([tri(pair[b:b + 1], mask[b:b + 1]) for b in range(B)], dim=0)
    c = _cos(y_full, y_each)
    maxabs = (y_full.float() - y_each.float()).abs().max().item()
    scale = y_full.float().abs().max().item() + 1e-6
    relmax = maxabs / scale                     # bf16 1-ULP is ~2^-8 of the output scale
    ok = c >= 0.9999 and relmax < 2e-2
    print(f"  [{'OK' if ok else 'FAIL'}] batch-independence  cos={c:.6f} "
          f"maxabs={maxabs:.4e} relmax={relmax:.4e}")
    return ok


def bench(d_pair: int) -> None:
    ref, tri = _build(d_pair)
    L = 256
    print(f"  timing (d={d_pair}, L={L}, fwd+bwd, ms/iter and ms/batch):")
    for B in (1, 2, 4):
        pair = torch.randn(B, L, L, d_pair, device=DEVICE, dtype=BF16, requires_grad=True)
        mask = _mask(B, L, partial=False)
        for _ in range(5):                       # warmup + autotune
            tri(pair, mask).float().sum().backward()
            pair.grad = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 20
        for _ in range(n):
            tri(pair, mask).float().sum().backward()
            pair.grad = None
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n * 1e3
        print(f"    B={B}: {ms:7.3f} ms/iter   {ms / B:7.3f} ms/batch")


def main() -> int:
    assert torch.cuda.is_available(), "need a GPU"
    print(f"device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
    ok = True
    print("== correctness vs pytorch reference ==")
    ok &= check_correctness(128)
    print("== batch independence ==")
    ok &= check_batch_independence(128)
    print("== latency / scaling ==")
    bench(128)
    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
