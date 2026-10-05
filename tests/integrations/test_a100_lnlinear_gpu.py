"""``ops.layer_norm_linear`` (``Linear(LayerNorm(x))`` to a few outputs, LayerNorm scale only, no bias) on A100: the hand-CUDA kernels of ``kernels/layernorm_linear/cuda/sm80.py`` against the
fp32 reference, forward and every gradient, over the registry rows (token_pair d_norm 64 .. 384 -> 2 .. 16 outputs, atom_pair d_norm 16 -> 4 / 12, atom_single 128 -> 3), with the errors held
to the bf16 PyTorch module's own error; dispatch, the env switch, ``torch.compile`` and CUDA-graph capture."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

EPS = 1e-5
#: (d_norm, n_head) of the registry rows
TOKEN = [(64, 2), (64, 4), (128, 4), (128, 8), (128, 16), (256, 8), (256, 16), (384, 8), (384, 12), (384, 16)]
ATOM_PAIR = [(16, 4), (16, 12)]
ATOM_OUTPUT = [(128, 3)]


@pytest.fixture(autouse=True)
def ampere():
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Ampere (sm_80) required")


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


def _case(shape, d, nh, ln_dtype=torch.float32, seed=1, offset=0.0):
    torch.manual_seed(seed)
    x = (torch.randn(*shape, d, device="cuda") + offset).to(torch.bfloat16)
    lnw = (1.0 + 0.2 * torch.randn(d, device="cuda")).to(ln_dtype)
    pw = (torch.randn(nh, d, device="cuda") * d ** -0.5).to(torch.bfloat16)
    dout = torch.randn(*shape, nh, device="cuda", dtype=torch.bfloat16)
    return x, lnw, pw, dout


def _run(fn, x, lnw, pw, dout, dtype=None):
    xx, ww, pp = (t.detach().clone().to(dtype or t.dtype).requires_grad_() for t in (x, lnw, pw))
    out = fn(xx, ww, pp)
    gx, gw, gp = torch.autograd.grad(out, [xx, ww, pp], dout.to(out.dtype))
    return out.detach(), gx, gw, gp


def _ref(x, w, pw):
    return F.linear(F.layer_norm(x, (x.shape[-1],), w, None, EPS), pw)


def _bf16_module(x, w, pw):
    return F.linear(F.layer_norm(x, (x.shape[-1],), w.to(torch.bfloat16), None, EPS), pw)


def _cuda(x, w, pw):
    from miniworld_engine import ops

    return ops.layer_norm_linear(x, w, pw, EPS)


def _check(shape, d, nh, ln_dtype=torch.float32, offset=0.0, floors=(3e-3, 5e-3, 5e-3, 5e-3)):
    from miniworld_engine.kernels.layernorm_linear.cuda import sm80

    x, lnw, pw, dout = _case(shape, d, nh, ln_dtype, offset=offset)
    assert sm80.serves(x, lnw, pw)
    want = _run(_ref, x, lnw, pw, dout, torch.float32)
    base = _run(_bf16_module, x, lnw, pw, dout)
    got = _run(_cuda, x, lnw, pw, dout)
    assert got[0].dtype is torch.bfloat16
    assert got[0].shape == (*shape, nh)
    assert got[1].dtype is torch.bfloat16
    assert got[1].shape == x.shape
    assert got[2].dtype is lnw.dtype
    assert got[3].dtype is torch.bfloat16
    for name, g, b, w, floor in zip(("out", "dx", "dscale", "dproj"), got, base, want, floors, strict=True):
        mine, ref = _rel(g, w), _rel(b, w)
        assert mine <= max(1.1 * ref, floor), f"{name}: cuda {mine:.3e} vs the bf16 module {ref:.3e}"


# ------------------------------------------------------------------------------------------------------------------------------------------- the registry rows
@pytest.mark.parametrize(("d", "nh"), TOKEN)
def test_token_pair_rows_match_the_fp32_reference(d, nh):
    _check((1, 128, 128), d, nh)


@pytest.mark.parametrize(("d", "nh"), ATOM_PAIR)
def test_atom_pair_windows_match_the_fp32_reference(d, nh):
    _check((1, 32, 32, 128), d, nh)                       # atoms 1024: windows of 32 queries x 128 keys


@pytest.mark.parametrize(("d", "nh"), ATOM_OUTPUT)
def test_the_atom_output_projection_matches_the_fp32_reference(d, nh):
    _check((1, 1024), d, nh)


@pytest.mark.parametrize(("d", "nh"), [(64, 4), (128, 16), (384, 12), (256, 8), (512, 4), (512, 16)])
def test_a_larger_pair_stack(d, nh):
    _check((1, 384, 384), d, nh)


@pytest.mark.parametrize(("d", "nh"), [(64, 2), (384, 16), (16, 12)])
def test_the_layernorm_scale_may_be_bf16(d, nh):
    _check((1, 64, 64), d, nh, ln_dtype=torch.bfloat16)


@pytest.mark.parametrize(("d", "nh"), [(128, 4), (384, 8), (16, 4)])
def test_inputs_with_a_nonzero_mean(d, nh):
    _check((1, 64, 64), d, nh, offset=2.0, floors=(4e-3, 8e-3, 8e-3, 8e-3))


@pytest.mark.parametrize("nh", [1, 2, 3, 5, 7, 8, 9, 13, 16])
@pytest.mark.parametrize("d", [64, 16])
def test_any_output_count_up_to_16_including_odd_ones(d, nh):
    _check((1, 48, 48), d, nh)


@pytest.mark.parametrize("rows", [1, 5, 15, 17, 33, 1000])
@pytest.mark.parametrize(("d", "nh"), [(64, 3), (128, 4), (384, 12), (16, 4)])
def test_row_counts_that_are_not_a_multiple_of_the_tile(rows, d, nh):
    _check((rows,), d, nh)


@pytest.mark.parametrize("shape", [(3, 37), (2, 5, 7), (1, 1, 1)])
def test_leading_dimensions_of_any_rank(shape):
    _check(shape, 128, 4)


# --------------------------------------------------------------------------------------------------------------------------------------------- determinism
@pytest.mark.parametrize(("d", "nh"), [(64, 4), (384, 16), (16, 12)])
def test_a_replay_is_bit_identical(d, nh):
    x, lnw, pw, dout = _case((1, 96, 96), d, nh)
    a = _run(_cuda, x, lnw, pw, dout)
    b = _run(_cuda, x, lnw, pw, dout)
    for u, v in zip(a, b, strict=True):
        assert torch.equal(u, v)


def test_inputs_are_left_alone_and_the_non_contiguous_view_is_read_correctly():
    x, lnw, pw, _ = _case((1, 64, 64), 128, 4)
    xt = x.transpose(1, 2)                                       # a non-contiguous view of the same values
    keep = x.clone()
    a = _cuda(xt, lnw, pw)
    want = _cuda(xt.contiguous(), lnw, pw)
    assert torch.equal(a, want)
    assert torch.equal(x, keep)


def test_only_some_of_the_inputs_want_gradients():
    x, lnw, pw, dout = _case((1, 64, 64), 128, 4)
    xx = x.clone().requires_grad_()
    out = _cuda(xx, lnw, pw)                                     # only x
    g, = torch.autograd.grad(out, [xx], dout)
    w2 = lnw.clone().requires_grad_()
    p2 = pw.clone().requires_grad_()
    out = _cuda(x, w2, p2)                                       # only the parameters
    gw, gp = torch.autograd.grad(out, [w2, p2], dout)
    assert torch.isfinite(g.float()).all()
    assert torch.isfinite(gw).all()
    assert torch.isfinite(gp.float()).all()
    with torch.no_grad():
        assert _cuda(x, lnw, pw).requires_grad is False


# ----------------------------------------------------------------------------------------------------------------------------------------------- dispatch
def test_the_op_runs_the_cuda_kernels_and_falls_back_to_triton_where_they_do_not_serve(monkeypatch):
    from miniworld_engine import ops
    from miniworld_engine.kernels.layernorm_linear import dispatch
    from miniworld_engine.kernels.layernorm_linear.cuda import sm80

    calls = []
    real_cuda, real_triton = sm80.layer_norm_linear, dispatch.triton_layer_norm_linear
    monkeypatch.setattr(sm80, "layer_norm_linear", lambda *a, **k: (calls.append("cuda"), real_cuda(*a, **k))[1])
    monkeypatch.setattr(dispatch, "triton_layer_norm_linear", lambda *a, **k: (calls.append("triton"), real_triton(*a, **k))[1])
    x, lnw, pw, _ = _case((1, 64, 64), 128, 4)
    ops.layer_norm_linear(x, lnw, pw, EPS)
    ops.layer_norm_linear(x.float(), lnw, pw.float(), EPS)                     # fp32 activations: the Triton op
    x2, lnw2, pw2, _ = _case((1, 64, 64), 96, 4)                               # a width without a kernel
    ops.layer_norm_linear(x2, lnw2, pw2, EPS)
    monkeypatch.setenv("MINIWORLD_LNLINEAR_SM80", "0")
    ops.layer_norm_linear(x, lnw, pw, EPS)                                     # the switch
    assert calls == ["cuda", "triton", "triton", "triton"], calls


def test_the_gate():
    from miniworld_engine.kernels.layernorm_linear.cuda import sm80

    x, lnw, pw, _ = _case((1, 8), 128, 4)
    assert sm80.serves(x, lnw, pw)
    assert not sm80.serves(x.float(), lnw, pw)                                  # fp32 activations
    assert not sm80.serves(x, lnw, pw.float())                                  # fp32 projection
    assert not sm80.serves(x, lnw[:64], pw)                                     # scale of the wrong width
    assert not sm80.serves(x.cpu(), lnw, pw)
    assert not sm80.serves(x, lnw, torch.randn(17, 128, device="cuda", dtype=torch.bfloat16))   # 17 outputs
    assert not sm80.serves(x, lnw, torch.randn(0, 128, device="cuda", dtype=torch.bfloat16))
    assert not sm80.serves(torch.zeros(0, 128, device="cuda", dtype=torch.bfloat16), lnw, pw)
    for d in (16, 64, 128, 256, 384, 512):
        assert sm80.serves(torch.zeros(4, d, device="cuda", dtype=torch.bfloat16), torch.ones(d, device="cuda"), torch.zeros(4, d, device="cuda", dtype=torch.bfloat16))
    for d in (8, 32, 96, 192, 640):
        assert not sm80.serves(torch.zeros(4, d, device="cuda", dtype=torch.bfloat16), torch.ones(d, device="cuda"), torch.zeros(4, d, device="cuda", dtype=torch.bfloat16))


def test_the_engine_backend_can_force_triton(monkeypatch):
    from miniworld_engine import settings
    from miniworld_engine.kernels.layernorm_linear.cuda import sm80

    x, lnw, pw, _ = _case((1, 8), 128, 4)
    assert sm80.serves(x, lnw, pw)
    previous = settings.current().engine_backend
    settings.configure(engine_backend="triton")
    try:
        assert not sm80.serves(x, lnw, pw)
    finally:
        settings.configure(engine_backend=previous)
    assert sm80.serves(x, lnw, pw)


# ------------------------------------------------------------------------------------------------------------------------------- compile and CUDA graphs
@pytest.mark.parametrize(("d", "nh"), [(64, 4), (384, 12), (16, 4)])
def test_the_compiled_op_matches_eager(d, nh):
    x, lnw, pw, dout = _case((1, 64, 64), d, nh)

    def f(a, b, c):
        return _cuda(a, b, c)

    compiled = torch.compile(f, fullgraph=True)
    eager = _run(f, x, lnw, pw, dout)
    got = _run(compiled, x, lnw, pw, dout)
    for u, v in zip(eager, got, strict=True):
        assert torch.equal(u, v)
    with torch.no_grad():
        assert torch.equal(compiled(x, lnw, pw), f(x, lnw, pw))


@pytest.mark.parametrize(("d", "nh"), [(64, 4), (384, 12), (16, 4)])
def test_a_cuda_graph_replays_the_forward_and_the_training_step(d, nh):
    x, lnw, pw, dout = _case((1, 64, 64), d, nh)
    xx, ww, pp = (t.clone().requires_grad_() for t in (x, lnw, pw))

    def step():
        out = _cuda(xx, ww, pp)
        return (out, *torch.autograd.grad(out, [xx, ww, pp], dout))

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static = step()
    xx.data.copy_(torch.randn_like(x))
    want = step()
    graph.replay()
    for got, ref in zip(static, want, strict=True):
        assert torch.equal(got, ref)
