"""Hopper B200-style folding: complete gradients, checkpointing and graph replay."""

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.kernels.trimul_inproj.cuda import h100_b200style as K

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(autouse=True)
def hopper():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("H100 required")


def fixture(d, n, direction, master=False):
    torch.manual_seed(519 + d + n + direction)
    h = d * (2 if direction == 0 else 1)
    x = torch.randn(1, n, n, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dtype = torch.float32 if master else torch.bfloat16
    matrices = [
        (torch.randn(*shape, device="cuda") * d**-0.5).to(dtype).requires_grad_()
        for shape in [(h, d)] * 4 + [(d, d), (d, h)]
    ]
    affine = [
        (
            (1 + 0.1 * torch.randn(c, device="cuda"))
            if weight
            else 0.05 * torch.randn(c, device="cuda")
        ).requires_grad_()
        for c, weight in ((d, True), (d, False), (h, True), (h, False))
    ]
    mask = torch.rand(1, n, device="cuda") > 0.2
    ds = ((torch.rand(n, d, device="cuda") > 0.25) / 0.75).to(torch.bfloat16)
    dy = torch.randn_like(x) * 0.1
    return [x, *matrices, *affine], mask, ds, dy


def reference(vals, mask, ds, direction):
    x, wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = vals
    d = x.shape[-1]
    xn = F.layer_norm(x, (d,), gi, bi, 1e-5)
    a = (xn @ wl.t()) * torch.sigmoid(xn @ wlg.t())
    b = (xn @ wr.t()) * torch.sigmoid(xn @ wrg.t())
    pm = (mask[:, :, None] & mask[:, None, :])[:, :, :, None]
    a = (a * pm)[0].permute(2, 0, 1).contiguous()
    b = (b * pm)[0].permute(2, 0, 1).contiguous()
    if direction == 0:
        t = torch.cat(
            (a[:d] @ b[:d].transpose(-1, -2), a[d:].transpose(-1, -2) @ b[d:])
        )
    elif direction == 1:
        t = a @ b.transpose(-1, -2)
    else:
        t = a.transpose(-1, -2) @ b
    norm = F.layer_norm(t.permute(1, 2, 0), (wp.shape[1],), go, bo, 1e-5)
    return x + ds[None, None] * (norm @ wp.t())[None] * torch.sigmoid(xn @ wg.t())


def relative(a, b):
    return float(
        (a.detach().float() - b.detach().float()).norm()
        / b.detach().float().norm().clamp_min(1e-12)
    )


def assert_reference(leaves, mask, ds, dy, direction, y, grads):
    refs = [v.detach().float().requires_grad_() for v in leaves]
    want = reference(refs, mask, ds.float(), direction)
    want.backward(dy.float())
    assert relative(y, want) < 0.006
    for index, (got, ref) in enumerate(zip(grads, refs, strict=True)):
        assert torch.isfinite(got).all(), index
        assert relative(got, ref.grad) < 0.01, (index, relative(got, ref.grad))


@pytest.mark.parametrize("d", K.WIDTHS)
@pytest.mark.parametrize("direction", [0, 1, 2])
@pytest.mark.parametrize("master", [False, True])
def test_all_gradients(d, direction, master):
    leaves, mask, ds, dy = fixture(d, 128, direction, master)
    y = K.trimul(leaves, mask, ds, direction)
    y.backward(dy)
    assert_reference(leaves, mask, ds, dy, direction, y, [v.grad for v in leaves])


@pytest.mark.parametrize("d", K.WIDTHS)
@pytest.mark.parametrize("n", [384, 768])
@pytest.mark.parametrize("direction", [0, 1])
def test_full_shapes(d, n, direction):
    leaves, mask, ds, dy = fixture(d, n, direction)
    y = K.trimul(leaves, mask, ds, direction)
    y.backward(dy)
    assert_reference(leaves, mask, ds, dy, direction, y, [v.grad for v in leaves])


@pytest.mark.parametrize("direction", [0, 1, 2])
def test_zero_mask_and_inference(direction):
    leaves, mask, ds, dy = fixture(256, 128, direction)
    mask.zero_()
    y = K.trimul(leaves, mask, ds, direction)
    y.backward(dy)
    assert_reference(leaves, mask, ds, dy, direction, y, [v.grad for v in leaves])
    with torch.no_grad():
        yi = K.trimul(leaves, mask, ds, direction)
    assert torch.equal(y, yi)


@pytest.mark.parametrize("d", [256, 512])
@pytest.mark.parametrize("direction", [0, 1])
def test_changed_graph_inputs_weights_mask_dropout_dy(d, direction):
    leaves, mask, ds, dy = fixture(d, 128, direction, master=True)

    def step():
        for v in leaves:
            v.grad = None
        y = K.trimul(leaves, mask, ds, direction)
        y.backward(dy)
        return y

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        y = step()
    grads = [v.grad for v in leaves]
    for trial in range(3):
        with torch.no_grad():
            leaves[0].normal_()
            leaves[1].add_(0.01 * torch.randn_like(leaves[1]))
            mask.copy_(torch.rand_like(mask, dtype=torch.float32) > 0.3)
            ds.copy_((torch.rand_like(ds) > 0.25) / 0.75)
            dy.normal_(std=0.1)
            if trial == 2:
                mask.zero_()
        graph.replay()
        assert_reference(leaves, mask, ds, dy, direction, y, grads)


@pytest.mark.parametrize("d", K.WIDTHS)
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("master", [False, True])
def test_module_dispatch_and_all_gradients(d, bidirectional, master, monkeypatch):
    from miniworld_engine import settings
    from miniworld_engine.integrations import trimul_h100
    from miniworld_engine.modules.exceptions import ImplementationType
    from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
    from miniworld_engine.modules.triangle_multiplication.bidirectional import (
        BidirectionalTriangleMultiplication,
    )

    cls = (
        BidirectionalTriangleMultiplication if bidirectional else TriangleMultiplication
    )
    old = settings.configure(force_triton_kernels=False)
    try:
        leaves, mask, ds, dy = fixture(d, 128, 0 if bidirectional else 1, master)
        module = (
            cls(d, p_drop=0.25, implementation=ImplementationType.MINIWORLD)
            .cuda()
            .train()
        )
        actual = [
            module.to_left.weight,
            module.to_left_gate.weight,
            module.to_right.weight,
            module.to_right_gate.weight,
            module.to_gate.weight,
            module.to_out.weight,
            module.ln_pair.weight,
            module.ln_pair.bias,
            module.ln_out.weight,
            module.ln_out.bias,
        ]
        with torch.no_grad():
            for parameter, value in zip(actual, leaves[1:], strict=True):
                parameter.data = parameter.data.to(value.dtype)
                parameter.copy_(value)
        monkeypatch.setattr(
            module, "_make_drop_row_scale", lambda pair, p: ds[None, None]
        )
        assert trimul_h100._b200style_serves(module, leaves[0])
        y = module(leaves[0], mask)
        y.backward(dy)
        assert_reference(
            [leaves[0], *actual],
            mask,
            ds,
            dy,
            0 if bidirectional else 1,
            y,
            [leaves[0].grad, *(w.grad for w in actual)],
        )
        with torch.no_grad():
            yi = module(leaves[0], mask)
        assert torch.equal(y, yi)
    finally:
        settings.configure(**vars(old))


@pytest.mark.parametrize("d", K.WIDTHS)
@pytest.mark.parametrize("direction", [0, 1, 2])
def test_inference_without_scale_is_bitwise_unit_scale(d, direction):
    leaves, mask, ds, _dy = fixture(d, 128, direction, master=True)
    ds.fill_(1)
    with torch.no_grad():
        y = K.trimul(leaves, mask, None, direction)
        expected = K.trimul(leaves, mask, ds, direction)
    assert torch.equal(y, expected)


@pytest.mark.parametrize("d", K.WIDTHS)
@pytest.mark.parametrize("direction", [0, 1, 2])
def test_changed_inference_graph_without_scale(d, direction):
    leaves, mask, ds, _ = fixture(d, 128, direction, master=True)
    ds.fill_(1)

    def step():
        with torch.no_grad():
            return K.trimul(leaves, mask, None, direction)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        y = step()
    for trial in range(3):
        with torch.no_grad():
            leaves[0].normal_()
            for weight in leaves[1:]:
                weight.add_(0.005 * torch.randn_like(weight))
            mask.copy_(torch.rand_like(mask, dtype=torch.float32) > 0.3)
            if trial == 2:
                mask.zero_()
        graph.replay()
        with torch.no_grad():
            want = reference([v.float() for v in leaves], mask, ds.float(), direction)
        assert relative(y, want) < 0.006


@pytest.mark.parametrize("d", K.WIDTHS)
@pytest.mark.parametrize("training", [False, True])
def test_selected_output_storage_dispatch(d, training):
    leaves, mask, ds, dy = fixture(d, 256, 0, master=True)

    def step():
        if training:
            for value in leaves:
                value.grad = None
            result = K.trimul(leaves, mask, ds, 0)
            result.backward(dy)
            return result
        with torch.no_grad():
            return K.trimul(leaves, mask, None, 0)

    step()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        for _ in range(3):
            step()
            torch.cuda.synchronize()
    names = [
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    if not names:
        pytest.skip(
            "the profiler returned no CUDA events (CUPTI is not available in this session)"
        )
    if training:
        expected = {
            256: "mw_b200style_k3_scalar_tmasave",
            384: "mw_b200style_k3_d384_tmasave",
            512: "mw_b200style_k3tmasave",
        }[d]
    else:
        expected = "mw_b200style_k3_static"
    assert any(name == expected or name.startswith(expected + "(") for name in names), (
        names
    )
    if training and d in (256, 384):
        expected_k1 = (
            "mw_b200style_k1wb_d256_dplpipe"
            if d == 256
            else "mw_b200style_k1wb_emitxn_tma"
        )
        assert any(name.split("(")[0] == expected_k1 for name in names), names
        assert not any(
            name.split("(")[0] == f"mw_b200style_apply_{d}" for name in names
        ), names
    assert all("triton" not in name.lower() for name in names), names
