"""H100 bidirectional TriMul in fp32 on TF32 tensor cores (D = hidden = 128): ``kernels/trimul_inproj/cuda/h100_f32.py``.

Accuracy is judged the way the project judges every fp32 / TF32 kernel: the module's error against an fp64 run of the same maths is at
most 1.5x the error of the module's own PyTorch statements under TF32 (matmul precision "high"), for the output and for every gradient.
The path takes no atomics, so y and every gradient are bitwise repeatable.
"""

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine import settings
from miniworld_engine.integrations import trimul_h100_f32
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

D = 128
NAMES = ["x", "to_left_gate", "to_left", "to_right_gate", "to_right", "to_out", "to_gate", "ln_pair.w", "ln_pair.b", "ln_out.w", "ln_out.b"]


@pytest.fixture(autouse=True)
def hopper_tf32():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("H100 (sm_90) required")
    torch.compiler.reset()
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    yield
    torch.set_float32_matmul_precision(prev)


def make(seed=0, p_drop=0.0, impl=ImplementationType.MINIWORLD):
    torch.manual_seed(seed)
    m = BidirectionalTriangleMultiplication(D, implementation=impl, p_drop=p_drop).cuda()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return m


def reference_like(m):
    ref = make(impl=ImplementationType.PYTORCH)
    ref.load_state_dict(m.state_dict())
    return ref


def rel(a, b):
    return float((a.double() - b).norm() / b.norm().clamp_min(1e-12))


def fp64(m, x, mask, ds):
    """The module's maths in fp64 (no module code: an independent statement)."""
    p = {k: v.detach().double() for k, v in m.named_parameters()}
    x = x.double()
    xn = F.layer_norm(x, (D,), p["ln_pair.weight"], p["ln_pair.bias"], 1e-5)
    left = torch.sigmoid(xn @ p["to_left_gate.weight"].T) * (xn @ p["to_left.weight"].T)
    right = torch.sigmoid(xn @ p["to_right_gate.weight"].T) * (xn @ p["to_right.weight"].T)
    if mask is not None:
        m2 = (mask.unsqueeze(-1) & mask.unsqueeze(-2))[..., None]
        left, right = left * m2, right * m2
    out = torch.cat([torch.einsum("bikd,bjkd->bijd", left[..., :D], right[..., :D]),
                     torch.einsum("bkid,bkjd->bijd", left[..., D:], right[..., D:])], -1)
    on = F.layer_norm(out, (2 * D,), p["ln_out.weight"], p["ln_out.bias"], 1e-5)
    y = torch.sigmoid(xn @ p["to_gate.weight"].T) * (on @ p["to_out.weight"].T)
    return x + (y if ds is None else y * ds.double())


def inputs(L, B=1, seed=1, masked=True, dropout=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(B, L, L, D, device="cuda", generator=g) * 2 + 0.3
    mask = None
    if masked:
        mask = torch.ones(B, L, dtype=torch.bool, device="cuda")
        mask[:, -max(L // 8, 1):] = False
    ds = None
    if dropout:
        ds = (torch.rand(B, 1, L, D, device="cuda", generator=g) > 0.25).float() / 0.75
    return x, mask, ds


@pytest.mark.parametrize("L", [16, 48, 100, 130, 384, 1024])
@pytest.mark.parametrize("masked,dropout", [(False, False), (True, True)])
def test_inference_accuracy_and_serving(L, masked, dropout):
    m = make().eval()
    ref = reference_like(m).eval()
    x, mask, ds = inputs(L, masked=masked, dropout=dropout)
    with torch.no_grad():
        assert trimul_h100_f32.serves(m, x, mask, ds)
        y = trimul_h100_f32.update(m, x, mask, ds)
        y_ref = ref(x, mask) if ds is None else x + (ref(x, mask) - x) * ds   # the reference module applies the residual itself
        y64 = fp64(m, x, mask, None if ds is None else ds.reshape(1, L, D))
        y2 = trimul_h100_f32.update(m, x, mask, ds)
    assert torch.equal(y, y2), "bitwise repeatable"
    assert rel(y, y64) <= 1.5 * rel(y_ref, y64) + 1e-7, (rel(y, y64), rel(y_ref, y64))


@pytest.mark.parametrize("L", [16, 64, 128, 196, 512])
@pytest.mark.parametrize("masked", [False, True])
def test_training_gradients(L, masked):
    m = make(p_drop=0.25).train()
    ref = reference_like(m)
    x0, mask, _ = inputs(L, masked=masked)
    ds = (torch.rand(1, 1, L, D, device="cuda") > 0.25).float() / 0.75
    gy = torch.randn_like(x0)

    def grads(fn, mod):
        x = x0.clone().requires_grad_(True)
        for p in mod.parameters():
            p.grad = None
        y = fn(x)
        y.backward(gy)
        return y.detach(), [x.grad] + [dict(mod.named_parameters())[n].grad for n in
                                       ["to_left_gate.weight", "to_left.weight", "to_right_gate.weight", "to_right.weight",
                                        "to_out.weight", "to_gate.weight", "ln_pair.weight", "ln_pair.bias", "ln_out.weight", "ln_out.bias"]]

    assert trimul_h100_f32.serves(m, x0.clone().requires_grad_(True), mask, ds)
    y, g = grads(lambda x: trimul_h100_f32.update(m, x, mask, ds), m)
    y_t, g_t = grads(lambda x: x + (ref(x, mask) - x) * ds, ref)
    x64 = x0.double().clone().requires_grad_(True)
    p64 = {k: v.detach().double().clone().requires_grad_(True) for k, v in m.named_parameters()}

    def fp64_graph(x):
        xn = F.layer_norm(x, (D,), p64["ln_pair.weight"], p64["ln_pair.bias"], 1e-5)
        left = torch.sigmoid(xn @ p64["to_left_gate.weight"].T) * (xn @ p64["to_left.weight"].T)
        right = torch.sigmoid(xn @ p64["to_right_gate.weight"].T) * (xn @ p64["to_right.weight"].T)
        if mask is not None:
            m2 = (mask.unsqueeze(-1) & mask.unsqueeze(-2))[..., None]
            left, right = left * m2, right * m2
        out = torch.cat([torch.einsum("bikd,bjkd->bijd", left[..., :D], right[..., :D]),
                         torch.einsum("bkid,bkjd->bijd", left[..., D:], right[..., D:])], -1)
        on = F.layer_norm(out, (2 * D,), p64["ln_out.weight"], p64["ln_out.bias"], 1e-5)
        return x + ds.double() * (torch.sigmoid(xn @ p64["to_gate.weight"].T) * (on @ p64["to_out.weight"].T))

    y64 = fp64_graph(x64)
    y64.backward(gy.double())
    g64 = [x64.grad] + [p64[n].grad for n in ["to_left_gate.weight", "to_left.weight", "to_right_gate.weight", "to_right.weight",
                                                "to_out.weight", "to_gate.weight", "ln_pair.weight", "ln_pair.bias", "ln_out.weight",
                                                "ln_out.bias"]]
    assert rel(y, y64.detach()) <= 1.5 * rel(y_t, y64.detach()) + 1e-7
    for name, a, b, c in zip(NAMES, g, g_t, g64):
        assert rel(a, c) <= 1.5 * rel(b, c) + 1e-7, (name, rel(a, c), rel(b, c))
    y2, g2 = grads(lambda x: trimul_h100_f32.update(m, x, mask, ds), m)
    assert torch.equal(y, y2) and all(torch.equal(a, b) for a, b in zip(g, g2)), "bitwise repeatable"


def test_batch_equals_samples():
    m = make().eval()
    x, mask, ds = inputs(64, B=3, dropout=True)
    with torch.no_grad():
        y = trimul_h100_f32.update(m, x, mask, ds)
        for b in range(3):
            yb = trimul_h100_f32.update(m, x[b:b + 1], mask[b:b + 1], ds[b:b + 1])
            assert torch.equal(y[b:b + 1], yb)


def test_module_dispatch_and_force_portable():
    m = make().eval()
    x, mask, _ = inputs(64)
    with torch.no_grad():
        y = m(x, mask)
        assert torch.equal(y, trimul_h100_f32.update(m, x, mask, None))
        prev = settings.configure(force_portable_kernels=True)
        try:
            assert not trimul_h100_f32.serves(m, x, mask, None)
        finally:
            settings.configure(force_portable_kernels=prev.force_portable_kernels)
        # "highest" precision: not TF32, not served
        torch.set_float32_matmul_precision("highest")
        assert not trimul_h100_f32.serves(m, x, mask, None)


def test_cuda_graph_capture_inference_and_training():
    m = make(p_drop=0.25).train()
    x0, mask, _ = inputs(64)
    x = x0.clone().requires_grad_(True)
    ds = (torch.rand(1, 1, 64, D, device="cuda") > 0.25).float() / 0.75
    gy = torch.randn_like(x)

    def step():
        m.zero_grad(set_to_none=True)
        x.grad = None
        y = trimul_h100_f32.update(m, x, mask, ds)
        y.backward(gy)
        return y

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):                      # warm-up on the capture stream (extension build, cuBLAS handles)
        for _ in range(3):
            step()
        ref_y = step().detach().clone()
        ref_g = x.grad.clone()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y_cap = step()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(y_cap, ref_y) and torch.equal(x.grad, ref_g)

    with torch.no_grad():                           # inference under capture too
        xi = x0.clone()
        for _ in range(2):
            trimul_h100_f32.update(m, xi, mask, ds)
        torch.cuda.synchronize()
        ref = trimul_h100_f32.update(m, xi, mask, ds)
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            out = trimul_h100_f32.update(m, xi, mask, ds)
        g2.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref)


def test_checkpointed_batch_equals_plain(monkeypatch):
    """A batch whose saved state would not fit runs every sample checkpointed: y and every gradient bitwise equal to the plain call."""
    m = make(p_drop=0.25).train()
    x0, mask, _ = inputs(64, B=3)
    ds = (torch.rand(3, 1, 64, D, device="cuda") > 0.25).float() / 0.75
    gy = torch.randn_like(x0)

    def run():
        x = x0.clone().requires_grad_(True)
        m.zero_grad(set_to_none=True)
        y = trimul_h100_f32.update(m, x, mask, ds)
        y.backward(gy)
        return y.detach(), [x.grad] + [p.grad.clone() for p in m.parameters()]

    assert not trimul_h100_f32._checkpointed(x0)
    y, g = run()
    monkeypatch.setattr(trimul_h100_f32, "CHECKPOINT_FRACTION", 0.0)
    assert trimul_h100_f32._checkpointed(x0)
    y2, g2 = run()
    assert torch.equal(y, y2) and all(torch.equal(a, b) for a, b in zip(g, g2))
