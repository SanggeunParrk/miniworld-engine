"""Hopper residual fusion: rounding, affine-gradient isolation, dispatch and graphs."""
import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def hopper_settings(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper required")
    from miniworld_engine import settings
    monkeypatch.setattr(settings, "_ACTIVE", settings.current())
    settings.configure(engine_backend="auto", transition_residual_fusion=True,
                       transition_h100_residual=True, autotune_miss_cap=3)
    torch.manual_seed(371)


def make_case(width, rows=128):
    from miniworld_engine.modules import Transition
    model = Transition(width, implementation="miniworld").cuda().bfloat16()
    with torch.no_grad():
        for param in model.parameters():
            if param.ndim == 2:
                param.normal_(std=width**-.5)
        model.ln_in.weight.uniform_(.3, 1.3)
        model.ln_in.weight[0] = 0  # no gamma division allowed in this path
        model.ln_in.bias.normal_(std=.1)
    x = torch.randn(1, rows, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    return model, x, torch.randn_like(x)


@pytest.mark.parametrize("width", [128, 256, 384, 512, 768])
def test_all_gradients_and_functional_dispatch(width):
    from miniworld_engine import ops
    from miniworld_engine.kernels.transition.hopper import transition_residual_hopper
    model, x, dy = make_case(width)
    leaves = (x, *model.parameters())
    reference = model._residual_triton_forward(x)
    expected = torch.autograd.grad(reference, leaves, dy)
    actual = transition_residual_hopper(
        x, model.ln_in.weight, model.ln_in.bias, model.expand_a.weight,
        model.expand_b.weight, model.squeeze.weight, model.ln_in.eps,
    )
    gradients = torch.autograd.grad(actual, leaves, dy)
    for a, e in zip((actual, *gradients), (reference, *expected)):
        assert torch.isfinite(a).all()
        # Existing LN-folded/dual-GEMM kernels differ at BF16 rounding boundaries.
        rel = (a.float() - e.float()).norm() / e.float().norm().clamp_min(1e-12)
        assert rel < .02
        assert torch.nn.functional.cosine_similarity(a.float().flatten(), e.float().flatten(), dim=0) > .9995
    functional = ops.transition(
        x, ln_in_weight=model.ln_in.weight, ln_in_bias=model.ln_in.bias,
        expand_a_weight=model.expand_a.weight, expand_b_weight=model.expand_b.weight,
        squeeze_weight=model.squeeze.weight, n=4, eps=model.ln_in.eps,
    )
    torch.testing.assert_close(model(x), functional, rtol=0, atol=0)


def test_explicit_cute_residual_backend():
    from miniworld_engine.modules.dispatch import KernelBackend
    from miniworld_engine.kernels.transition.hopper import transition_residual_hopper
    model, x, dy = make_case(128)
    model._backend = KernelBackend.CUTE
    expected = transition_residual_hopper(
        x, model.ln_in.weight, model.ln_in.bias, model.expand_a.weight,
        model.expand_b.weight, model.squeeze.weight, model.ln_in.eps, use_b2b=False,
    )
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    assert all(torch.isfinite(g).all() for g in torch.autograd.grad(model(x), (x, *model.parameters()), dy))


def test_large_pair_keeps_fp32_affine_gradients():
    model, x, dy = make_case(128, rows=16384)
    assert model.ln_in.weight.dtype == torch.float32
    # Native forward uses BF16 affine values, but the residual LN reducer emits
    # FP32 dgamma/dbeta. Rounding those to BF16 amplifies tiny atomic-order noise
    # into a whole BF16 ULP before autograd casts them back to FP32 parameters.
    grads = torch.autograd.grad(model(x), (model.ln_in.weight, model.ln_in.bias), dy)
    for grad in grads:
        assert grad.dtype == torch.float32
        assert torch.isfinite(grad).all()
        assert torch.any(grad != grad.bfloat16().float())


@pytest.mark.parametrize("width", [128, 256, 512, 1024])
def test_cuda_ln_residual_exact_rounding_and_affine_isolation(width):
    from miniworld_engine.kernels.layernorm.cuda import layer_norm_bwd_cuda
    x = torch.randn(257, width, device="cuda", dtype=torch.bfloat16)
    dy, dr = torch.randn_like(x), torch.randn_like(x)
    gamma = torch.randn(width, device="cuda", dtype=x.dtype)
    gamma[0] = 0
    mean = x.float().mean(-1)
    rstd = torch.rsqrt(x.float().var(-1, unbiased=False) + 1e-5)
    row_scale = torch.rand(257, device="cuda")
    plain = layer_norm_bwd_cuda(dy, x, gamma, mean, rstd, row_scale)
    fused = layer_norm_bwd_cuda(dy, x, gamma, mean, rstd, row_scale, residual=dr)
    for a, e in zip(fused, (plain[0] + dr, plain[1], plain[2])):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(35, 96, 384), (128, 512, 2048)])
def test_cute_squeeze_residual_rounding_and_tails(shape):
    from miniworld_engine.kernels.transition.cute.squeeze_residual import squeeze_residual
    m, n, k = shape
    h = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / k**.5
    w = torch.randn(n, k, device="cuda", dtype=h.dtype)
    r = torch.randn(m, n, device="cuda", dtype=h.dtype)
    # BF16 GEMM reference retains the accumulator->BF16 boundary before addition.
    expected = h @ w.T + r
    actual = squeeze_residual(h, w, r)
    rel = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert rel < 1e-4


def test_forced_triton_and_unsupported_shapes(monkeypatch):
    from miniworld_engine import settings
    from miniworld_engine.kernels.transition import hopper
    model, x, _ = make_case(128)
    assert hopper.enabled(x, 4)
    settings.configure(engine_backend="triton")
    assert not hopper.enabled(x, 4)
    monkeypatch.setattr(hopper, "transition_residual_hopper", lambda *a: pytest.fail("native path selected"))
    torch.testing.assert_close(model(x), model._residual_triton_forward(x), rtol=0, atol=0)
    settings.configure(engine_backend="auto")
    assert not hopper.enabled(x[:, :127], 4)
    assert not hopper.enabled(x, 2)
    assert not hopper.enabled(x.float(), 4)


@pytest.mark.parametrize("width", [128, 512])
def test_static_compile_cuda_graph_backward(width):
    # Create/evaluate leaves on the same stream as capture. An earlier default-stream
    # AccumulateGrad otherwise adds a forbidden legacy-stream dependency at capture.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        model, x, dy = make_case(width)
        if width > 256:
            from miniworld_engine.modules.dispatch import KernelBackend
            model._backend = KernelBackend.CUTE
        leaves = (x, *model.parameters())
        expected_y = model(x)
        expected_g = torch.autograd.grad(expected_y, leaves, dy)
        compiled = torch.compile(model, fullgraph=True, dynamic=False)
        held = {}

        def step():
            for t in leaves:
                t.grad = None
            held["y"] = compiled(x)
            held["y"].backward(dy)
            held["g"] = tuple(t.grad for t in leaves)

        for _ in range(3):
            step()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            step()
        for _ in range(3):
            graph.replay()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    for a, e in zip((held["y"], *held["g"]), (expected_y, *expected_g)):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.parametrize("width", [128, 256])
def test_b2b_saved_operand_and_residual(width):
    from miniworld_engine.kernels.transition.cuda import transition_b2b_fwd, transition_b2b_fwd_saved
    from miniworld_engine.kernels.layernorm_linear.triton.stats import stats_triton
    model, x, dy = make_case(width, rows=256)
    x2 = x.detach().reshape(-1, width)
    rstd, c1 = stats_triton(x2, 1e-5)
    g, b = model.ln_in.weight.bfloat16(), model.ln_in.bias.bfloat16()
    args = (x2, rstd, c1, g, b, model.expand_a.weight, model.expand_b.weight)
    y, xn = transition_b2b_fwd_saved(*args, model.squeeze.weight)
    plain = transition_b2b_fwd(*args, model.squeeze.weight)
    torch.testing.assert_close(y, plain, rtol=0, atol=0)
    expected = ((x2.float() * rstd[:, None] - c1[:, None]) * g + b).bfloat16()
    assert (xn.float()-expected.float()).norm() / expected.float().norm() < .005
    assert xn.is_contiguous() and xn.dtype == x.dtype and xn.data_ptr() != x.data_ptr()
    zero_y, _ = transition_b2b_fwd_saved(*args, torch.zeros_like(model.squeeze.weight))
    torch.testing.assert_close(zero_y, x2, rtol=0, atol=0)


@pytest.mark.parametrize("width", [128, 256])
def test_b2b_saved_all_gradients_and_identity(width):
    from miniworld_engine.kernels.transition.hopper import transition_residual_hopper
    model, x, dy = make_case(width, rows=256)
    leaves = (x, *model.parameters())
    args = (x, model.ln_in.weight, model.ln_in.bias, model.expand_a.weight,
            model.expand_b.weight, model.squeeze.weight)
    ref = model._residual_triton_forward(x)
    grads_ref = torch.autograd.grad(ref, leaves, dy)
    y = transition_residual_hopper(*args, save_xn=True)
    grads = torch.autograd.grad(y, leaves, dy)
    for a, e in zip((y, *grads), (ref, *grads_ref)):
        assert torch.isfinite(a).all()
        assert (a.float()-e.float()).norm() / e.float().norm().clamp_min(1e-12) < .02
    with torch.no_grad():
        model.squeeze.weight.zero_()
    y = transition_residual_hopper(*args, save_xn=True)
    dx, dg, db = torch.autograd.grad(y, leaves[:3], dy)
    torch.testing.assert_close(y, x, rtol=0, atol=0)
    torch.testing.assert_close(dx, dy, rtol=0, atol=0)
    assert torch.count_nonzero(dg) == torch.count_nonzero(db) == 0


@pytest.mark.parametrize("width", [128, 256])
def test_b2b_saved_static_compile_and_graph(width):
    from miniworld_engine import settings
    settings.configure(transition_h100_save_xn=True)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        model, x, dy = make_case(width, rows=16384)
        leaves = (x, *model.parameters())
        compiled = torch.compile(model, fullgraph=True, dynamic=False)
        held = {}
        def step():
            for leaf in leaves:
                leaf.grad = None
            held['y'] = compiled(x)
            held['y'].backward(dy)
        for _ in range(3):
            step()
        expected = (held['y'].clone(), *(t.grad.clone() for t in leaves))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            step()
        for _ in range(3):
            graph.replay()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    for actual, ref in zip((held['y'], *(t.grad for t in leaves)), expected):
        error = (actual.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-12)
        assert error < (1e-4 if actual.dtype == torch.float32 else .02)
