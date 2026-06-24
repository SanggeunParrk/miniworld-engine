"""Backward correctness gate for the trainable LayerNormLinear (autograd Fn).

Compares all five grads (dx, dW, db, dgamma, dbeta) from ``layernorm_linear_fn`` against an
fp32 PyTorch autograd oracle (F.linear(F.layer_norm(...))). cos is the metric (bf16 GEMMs
round at ~1e-2, so element-wise abs error is meaningless on near-zero entries). COMPUTE NODE
only (srun).
"""

from __future__ import annotations

import sys
from pathlib import Path

_src = Path(__file__).resolve()
while _src.name != "src" and _src.parent != _src:
    _src = _src.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import torch
import torch.nn.functional as F

from miniworld_kernels.kernels.layernorm_linear.autograd import layernorm_linear_fn


def cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def check(M, K, N, dtype, has_bias):
    torch.manual_seed(0)
    eps = 1e-5
    xb = torch.randn(M, K, device="cuda", dtype=dtype)
    gb = torch.randn(K, device="cuda", dtype=dtype)
    bb = torch.randn(K, device="cuda", dtype=dtype)
    wb = (torch.randn(N, K, device="cuda", dtype=dtype) * K**-0.5)
    biasb = torch.randn(N, device="cuda", dtype=dtype) if has_bias else None
    dY = torch.randn(M, N, device="cuda", dtype=dtype)

    # ours
    x = xb.clone().requires_grad_(True)
    g = gb.clone().requires_grad_(True)
    b = bb.clone().requires_grad_(True)
    w = wb.clone().requires_grad_(True)
    bias = biasb.clone().requires_grad_(True) if has_bias else None
    Y = layernorm_linear_fn(x, g, b, w, bias, eps)
    Y.backward(dY)

    # fp32 autograd oracle
    xo = xb.float().clone().requires_grad_(True)
    go = gb.float().clone().requires_grad_(True)
    bo = bb.float().clone().requires_grad_(True)
    wo = wb.float().clone().requires_grad_(True)
    biaso = biasb.float().clone().requires_grad_(True) if has_bias else None
    Yo = F.linear(F.layer_norm(xo, (K,), go, bo, eps), wo, biaso)
    Yo.backward(dY.float())

    fcos = cos(Y, Yo)
    rows = [("dx", x.grad, xo.grad), ("dW", w.grad, wo.grad),
            ("dgamma", g.grad, go.grad), ("dbeta", b.grad, bo.grad)]
    if has_bias:
        rows.append(("db", bias.grad, biaso.grad))
    parts = " ".join(f"{n}={cos(a, o):.5f}" for n, a, o in rows)
    worst = min([fcos] + [cos(a, o) for _, a, o in rows])
    flag = "OK " if worst >= 0.999 else "FAIL"
    print(f"[{flag}] M={M:>7} K={K:>4} N={N:>4} {str(dtype).split('.')[-1]:>8} bias={int(has_bias)} | "
          f"Y={fcos:.5f} {parts}  worst={worst:.5f}", flush=True)
    return worst >= 0.999


def main():
    print(f"backward verify on {torch.cuda.get_device_name(0)}")
    ok = True
    for dtype in (torch.bfloat16, torch.float16):
        for (M, K, N) in [(8192, 128, 128), (16384, 256, 256), (8192, 384, 384),
                          (16384, 512, 512), (8192, 768, 768), (4096, 256, 512)]:
            for hb in (True, False):
                ok &= check(M, K, N, dtype, hb)
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
