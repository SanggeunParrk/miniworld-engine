# pyright: reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportOptionalIterable=false, reportArgumentType=false, reportGeneralTypeIssues=false, reportPrivateImportUsage=false
"""S90: Transition TRAINING on the B200 reads no uninitialised memory.

One training step (y, dx, every parameter gradient) is compared bitwise
 (a) between two fresh CUDA-graph captures,
 (b) eager on clean memory against eager after the allocator's free blocks were filled with 0xFF,
 (c) a capture against a capture whose pool is poisoned (0xFF blocks allocated, filled and freed inside the capture before the step).
Covers every training route of ``cuda:b200_sm100``: the two-role D 64 / D 128 backwards, the D 256 builds (the X2 training forward from the
second tile of a CTA), D 384 / 512 / 768 (small-M tsx_* chains, ``tg_bwd_small``, the wide kernels) and the fp32 TF32 generic path.
"""

from __future__ import annotations

import pytest
import torch

from foldjet_engine.modules import Transition

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="needs a B200 (sm_100)"
)

# (d_hidden, n, rows, activation dtype)
CASES = [
    (64, 2, 16384, "bf16"),
    (64, 4, 16384, "bf16"),
    (64, 2, 65536, "bf16"),
    (128, 2, 16384, "bf16"),
    (128, 4, 16384, "bf16"),
    (256, 2, 16384, "bf16"),
    (256, 4, 65536, "bf16"),  # the X2 training forward (tiles > SMs)
    (256, 2, 65536, "bf16"),
    (384, 2, 16384, "bf16"),
    (384, 4, 16384, "bf16"),
    (512, 2, 16384, "bf16"),
    (384, 2, 128, "bf16"),  # tsx_* chain (rows 217 / 218)
    (384, 4, 128, "bf16"),
    (768, 2, 128, "bf16"),  # tg_bwd_small (row 219)
    (128, 2, 16384, "fp32"),
    (256, 2, 16384, "fp32"),
    (512, 2, 16384, "fp32"),
    (384, 2, 128, "fp32"),  # tg_bwd_small, fp32 single stream
    (768, 2, 128, "fp32"),
]


def _poison(gb: int = 8) -> None:
    ts = []
    for mb in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        for _ in range(max(1, gb * 1024 // 9 // mb)):
            t = torch.empty(mb << 20, dtype=torch.uint8, device="cuda")
            t.fill_(0xFF)
            ts.append(t)
    del ts


@pytest.mark.parametrize(("d", "n", "rows", "dt"), CASES)
def test_transition_train_no_uninitialised_reads(d, n, rows, dt) -> None:
    dtype = torch.bfloat16 if dt == "bf16" else torch.float32
    torch.manual_seed(0)
    m = Transition(d, n=n, backend="cuda").cuda().float().train()
    if dtype is torch.bfloat16:
        m.set_master_weights()
    for p in m.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, 0, d**-0.5)
    x = torch.randn(1, rows, d, device="cuda", dtype=dtype)
    xg = x.clone().requires_grad_(True)
    dy = torch.randn(x.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(5)).to(dtype)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True

    def step():
        for p in m.parameters():
            p.grad = None
        xg.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype is torch.bfloat16):
            y = m(xg)
        y.backward(dy)
        return y.detach()

    def grab(y):
        return [y.float().cpu(), xg.grad.float().cpu(), *[p.grad.float().cpu().clone() for p in m.parameters()]]

    def eager(poisoned):
        if poisoned:
            _poison(24)
        y = step()
        torch.cuda.synchronize()
        return grab(y)

    def capture(poisoned):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                step()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            if poisoned:
                _poison(8)
            y = step()
            outs = (y, xg.grad, [p.grad for p in m.parameters()])
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        return [outs[0].float().cpu(), outs[1].float().cpu(), *[q.float().cpu().clone() for q in outs[2]]]

    try:
        for _ in range(3):
            eager(False)
        e0, e1 = eager(False), eager(True)
        a, b, c = capture(False), capture(False), capture(True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    names = ["y", "dx", *[nm for nm, _ in m.named_parameters()]]
    for label, (u, v) in {"eager clean vs eager poisoned": (e0, e1), "capture vs fresh capture": (a, b), "capture vs poisoned-pool capture": (a, c)}.items():
        bad = [nm for nm, p, q in zip(names, u, v, strict=True) if not torch.equal(p, q)]
        assert not bad, f"{label}: {bad}"
