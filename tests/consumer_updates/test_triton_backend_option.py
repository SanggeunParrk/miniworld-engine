"""An application Triton policy must not escape into native H100 engine kernels."""

import pytest
import torch

from miniworld_engine import kernels, settings
from miniworld_engine.modules import dispatch
from miniworld_engine.modules.exceptions import ImplementationType as EngineImpl


def configure_engine_backend(backend):
    settings.configure(engine_backend=backend)


@pytest.fixture(autouse=True)
def restore_policy():
    previous = settings.current()
    yield
    settings.configure(**vars(previous))


def test_h100_auto_and_strict_resolution(monkeypatch):
    monkeypatch.setattr(dispatch, "is_sm90plus", lambda *_: True)
    configure_engine_backend("auto")
    assert dispatch.resolve("triangle_multiplication", "miniworld").value == "cute"
    configure_engine_backend("triton")
    # Even the old per-op pin cannot override the process's strict policy.
    settings.configure(trimul_impl="cute")
    for op in dispatch._MINIWORLD_KNOWN_BEST:
        assert dispatch.resolve(op, "miniworld").value == "triton"
        assert dispatch.resolve(op, "pytorch").value == "pytorch"
    with pytest.raises(ValueError, match="conflicts"):
        dispatch.resolve("transition", "cute")


def test_standalone_layernorm_cannot_select_cuda():
    from miniworld_engine.kernels.layernorm.compile_native import _resolve_bwd_path

    configure_engine_backend("triton")
    settings.configure(layernorm_bwd_path="cuda")
    # No CUDA tensors needed: policy precedes hardware lookup, cache and native override.
    x = torch.zeros(2, 128, dtype=torch.bfloat16)
    w = torch.ones(128, dtype=x.dtype)
    stat = torch.zeros(2)
    assert _resolve_bwd_path(2, 128, x, x, w, stat, stat) == "atomic"
    assert _resolve_bwd_path(2, 512, x, x, w, stat, stat) == "persistent"


@pytest.mark.parametrize("training", [True, False])
def test_transition_policy_precedes_internal_h100_dispatch(monkeypatch, training):
    from miniworld_engine.modules import Transition

    configure_engine_backend("triton")
    module = Transition(128, implementation=EngineImpl.MINIWORLD).train(training)
    monkeypatch.setattr(module, "_old_triton_forward", lambda x: x * 3)

    def forbidden(*args, **kwargs):
        raise AssertionError("Native/fused auto path was entered")

    for name in [
        "cute_transition_fused",
        "cuda_transition_b2b",
        "triton_transition_fused",
    ]:
        monkeypatch.setattr(kernels, name, forbidden)
    x = torch.ones(2, 128, dtype=torch.bfloat16)
    assert torch.equal(module(x), x * 4)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("kind", "width"),
    [("transition", 128), ("transition", 512), ("trimul", 128)],
    ids=["transition128", "transition512", "trimul128"],
)
def test_cuda_training_and_graph_no_native_engine_calls(monkeypatch, kind, width):
    from miniworld_engine.autotune import native
    from miniworld_engine.kernels.layernorm import cuda as ln_cuda
    from miniworld_engine.modules import BidirectionalTriangleMultiplication, Transition

    configure_engine_backend("triton")
    settings.configure(
        layernorm_cuda_bwd=True
    )  # strict policy must beat this native opt-in
    native_calls = []

    def forbidden(*args, **kwargs):
        native_calls.append(True)
        raise AssertionError("Native H100 engine kernel entered under Triton policy")

    monkeypatch.setattr(native, "choose_config", forbidden)
    monkeypatch.setattr(ln_cuda, "layer_norm_bwd_cuda", forbidden)
    for name in [
        "cute_transition_fused",
        "cuda_transition_b2b",
        "triton_transition_fused",
    ]:
        monkeypatch.setattr(kernels, name, forbidden)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.manual_seed(9)
        # Includes wide Transition's unconditional CuTe branch in the old auto route.
        layer = (
            BidirectionalTriangleMultiplication(
                width, implementation=EngineImpl.MINIWORLD, p_drop=0.0
            )
            if kind == "trimul"
            else Transition(width, implementation=EngineImpl.MINIWORLD)
        )
        shape = (1, 128, 128, width) if kind == "trimul" else (1, 128, width)
        layer = layer.to(device="cuda", dtype=torch.bfloat16).train()
        with torch.no_grad():
            for _n, p in layer.named_parameters():
                if p.ndim == 2:
                    p.normal_(std=0.02)
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        ref = (
            type(layer)(
                128 if len(shape) == 4 else shape[-1],
                implementation=EngineImpl.PYTORCH,
                **({"p_drop": 0.0} if len(shape) == 4 else {}),
            )
            .to(device="cuda", dtype=torch.bfloat16)
            .train()
        )
        ref.load_state_dict(layer.state_dict())
        xr = x.detach().clone().requires_grad_()
        dy = torch.randn_like(x)
        expected = ref(xr)
        expected.backward(dy)
        compiled = torch.compile(layer, dynamic=False)
        actual = compiled(x)
        actual.backward(dy)

        def error(a, b):
            return float(
                (a.detach().float() - b.detach().float()).norm()
                / b.detach().float().norm().clamp_min(1e-8)
            )

        assert error(actual, expected) < 0.02
        assert error(x.grad, xr.grad) < 0.03
        for (name, p), (_, rp) in zip(layer.named_parameters(), ref.named_parameters(), strict=False):
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name
            assert error(p.grad, rp.grad) < 0.04, (name, error(p.grad, rp.grad))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            graph_y = compiled(x)
            graph_y.backward(dy)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(graph_y).all()
        layer.eval()
        with torch.no_grad():
            assert torch.isfinite(layer(x)).all()
        assert not native_calls
        del compiled, layer, ref, x, xr, actual, expected, graph, graph_y
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_standalone_layernorm_blocks_native_override(monkeypatch):
    from miniworld_engine.kernels.layernorm import cuda as ln_cuda

    configure_engine_backend("triton")
    settings.configure(layernorm_bwd_path="cuda")

    def forbidden(*args, **kwargs):
        raise AssertionError("Native LayerNorm entered under Triton policy")

    monkeypatch.setattr(ln_cuda, "layer_norm_bwd_cuda", forbidden)
    for width in (128, 512):
        x = torch.randn(
            1, 128, width, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        w = torch.randn(width, device="cuda", dtype=x.dtype, requires_grad=True)
        b = torch.randn_like(w, requires_grad=True)
        dy = torch.randn_like(x)
        actual = kernels.layernorm_kernel(x, w, b)
        expected = torch.nn.functional.layer_norm(x, (width,), w, b)
        actual_grads = torch.autograd.grad(actual, (x, w, b), dy)
        expected_grads = torch.autograd.grad(expected, (x, w, b), dy)
        for a, e in zip((actual, *actual_grads), (expected, *expected_grads), strict=False):
            assert torch.isfinite(a).all()
            assert (a.float() - e.float()).norm() / e.float().norm().clamp_min(
                1e-8
            ) < 0.02
