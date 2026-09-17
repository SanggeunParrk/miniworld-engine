"""Qualify packed contractions and the compiled bidirectional training path."""

import pytest
import torch

from miniworld_engine.kernels.trimul_inproj.cute import contract as cute_contract
from miniworld_engine.kernels.trimul_inproj.triton import contract as triton_contract
from miniworld_engine.modules import BidirectionalTriangleMultiplication
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def models(kind, implementation="miniworld"):
    cls = BidirectionalTriangleMultiplication
    options = {}
    ref = (
        cls(128, p_drop=0, implementation=ImplementationType.PYTORCH, **options)
        .cuda()
        .float()
    )
    actual = (
        cls(128, p_drop=0, implementation=ImplementationType(implementation), **options)
        .cuda()
        .to(torch.bfloat16)
    )
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if "ln_" not in name:
                p.normal_(std=128**-0.5)
    actual.load_state_dict(ref.state_dict())
    # Compare the same representable weights, retaining an FP32 reference calculation.
    ref.load_state_dict(actual.state_dict())
    return actual, ref


def check(name, a, b):
    assert a is not None, name
    assert b is not None, name
    assert torch.isfinite(a).all(), name
    assert torch.isfinite(b).all(), name
    err = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)
    assert err < 0.025, (name, float(err))


@pytest.mark.parametrize(
    ("length", "hidden", "strided"),
    [
        (31, 7, False),
        (128, 128, False),
        (384, 128, False),
        (768, 128, False),
        (384, 128, True),
    ],
)
@pytest.mark.parametrize("contract", [cute_contract, triton_contract])
def test_packed_contractions(length, hidden, strided, contract):
    torch.manual_seed(479)
    torch.backends.cuda.matmul.allow_tf32 = False
    tensors = [
        torch.randn(2 * hidden, length, length, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    if strided:
        tensors = [t.transpose(1, 2) for t in tensors]
    left, right, grad = tensors
    use_quack = (
        length in (384, 768)
        and hidden == 128
        and not strided
        and torch.cuda.get_device_name() == "NVIDIA H100 80GB HBM3"
    )
    if contract is cute_contract:
        assert contract._use_quack(*tensors) == use_quack
    saved = [t.clone() for t in tensors]
    tri = contract.packed_forward(left, right, hidden)
    dl, dr = contract.packed_backward(grad, left, right, hidden)
    # Independent autograd reference verifies all transpose/gradient formulas.
    lr, rr = left.float().requires_grad_(), right.float().requires_grad_()
    reference = torch.cat(
        (
            lr[:hidden] @ rr[:hidden].transpose(1, 2),
            lr[hidden:].transpose(1, 2) @ rr[hidden:],
        )
    )
    reference.backward(grad.float())
    for name, value, expected in (
        ("tri", tri, reference),
        ("dl", dl, lr.grad),
        ("dr", dr, rr.grad),
    ):
        assert expected is not None
        check(name, value, expected)
        assert value.is_contiguous()
        error = (value.float() - expected).norm() / expected.norm()
        assert error < 0.004, (name, float(error))
    for before, after in zip(saved, tensors, strict=False):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert (
        len({t.untyped_storage().data_ptr() for t in [left, right, grad, tri, dl, dr]})
        == 6
    )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("implementation", ["miniworld", "triton"])
def test_cold_compiled_training_reference(dynamic, implementation):
    torch.manual_seed(921)
    torch.compiler.reset()
    actual, ref = models("bidir", implementation)
    x = torch.randn(
        1, 384, 384, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    xr = x.detach().float().requires_grad_()
    mask = torch.ones(1, 384, device="cuda", dtype=torch.bool)
    mask[:, ::3] = False
    compiled = torch.compile(
        actual, dynamic=dynamic, fullgraph=True, options={"triton.cudagraphs": False}
    )
    y, yr = compiled(x, mask), ref(xr, mask)
    dy = torch.randn_like(y)
    y.backward(dy)
    yr.backward(dy.float())
    check("output", y, yr)
    check("input", x.grad, xr.grad)
    for (name, p), (rn, rp) in zip(actual.named_parameters(), ref.named_parameters(), strict=False):
        assert name == rn
        check(name, p.grad, rp.grad)


@pytest.mark.parametrize("implementation", ["miniworld", "triton"])
def test_compiled_training_graph_rng_residual(implementation):
    torch.manual_seed(531)
    torch.compiler.reset()
    actual, _ = models("bidir", implementation)
    actual.p_drop = 0.25
    x = torch.randn(
        1, 384, 384, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    dy = torch.randn_like(x)
    mask = torch.ones(1, 384, device="cuda", dtype=torch.bool)
    mask[:, ::3] = False
    compiled = torch.compile(
        actual, dynamic=False, fullgraph=True, options={"triton.cudagraphs": False}
    )

    def step():
        actual.zero_grad(set_to_none=False)
        if x.grad is not None:
            x.grad.zero_()
        y = compiled(x, mask)
        y.backward(dy)
        return y

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = step()
    graph.replay()
    previous = output.clone()
    graph.replay()
    assert not torch.equal(previous, output), "dropout RNG froze"
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    mask.zero_()
    graph.replay()
    torch.testing.assert_close(output, x, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, dy, rtol=0, atol=0)
    for name, param in actual.named_parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        # Masking the front does not mask the output LayerNorm bias: its
        # derivative remains live even at beta=0 and zero contraction output.
        if name != "ln_out.bias":
            assert torch.count_nonzero(param.grad) == 0, name


# Engine CI selects GPU checks explicitly.
pytestmark = [pytest.mark.gpu, *([pytestmark] if "pytestmark" in globals() else [])]
