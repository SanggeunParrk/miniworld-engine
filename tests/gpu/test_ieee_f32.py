"""IEEE fp32 CUDA chain (A100 / RTX 5090): ``kernels/ieee_f32`` GEMM, the Transition built on it (fp32 with TF32 off), and the fp32 PoolAtoms.
Accuracy standard: max|y - y64| / max|y64| against float64, tolerance max(err of PyTorch's own IEEE fp32, 8e-3); TF32 is never enabled here."""

import pytest
import torch

from miniworld_engine.modules import PoolAtoms, Transition
from miniworld_engine.modules.functional import pool_atoms

pytestmark = [pytest.mark.gpu]


def _card() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((8, 0), (12, 0))


skip = pytest.mark.skipif(not _card(), reason="A100 / RTX 5090 (the sm_80 sources)")
skip_h100 = pytest.mark.skipif(not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)), reason="H100 (the sources built for sm_90a)")


@pytest.fixture(autouse=True)
def _ieee():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


def _err(y: torch.Tensor, y64: torch.Tensor) -> float:
    return ((y.double() - y64).abs().max() / y64.abs().max()).item()


@skip
@pytest.mark.parametrize("ta", [False, True])
@pytest.mark.parametrize("tb", [False, True])
@pytest.mark.parametrize(("m", "n", "k"), [(384, 512, 384), (36, 20, 124), (256, 128, 9000), (384, 768, 384), (3072, 384, 768), (4096, 128, 128), (128, 128, 4096), (512, 128, 32768), (1536, 768, 3072), (3072, 3072, 768)])
def test_mm_layouts(ta: bool, tb: bool, m: int, n: int, k: int) -> None:
    from miniworld_engine.kernels import ieee_f32

    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn((k, m) if ta else (m, k), device="cuda", generator=g)
    b = torch.randn((n, k) if tb else (k, n), device="cuda", generator=g)
    a_, b_ = (a.t() if ta else a), (b.t() if tb else b)
    bias = torch.randn(n, device="cuda", generator=g)
    y64 = a_.double() @ b_.double()
    assert _err(ieee_f32.mm(a_, b_), y64) < max(_err(a_ @ b_, y64) * 2, 1e-5)
    assert _err(ieee_f32.mm(a_, b_, bias=bias), y64 + bias.double()) < max(_err(a_ @ b_ + bias, y64 + bias.double()) * 2, 1e-5)
    assert torch.equal(ieee_f32.mm(a_, b_), ieee_f32.mm(a_, b_)), "fixed accumulation order: bitwise repeatable"


def _module(d: int, h: int) -> Transition:
    m = Transition(d_hidden=d, n=h // d, master_weights=True, backend="cuda").cuda()
    torch.manual_seed(0)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.5 / p.shape[-1] ** 0.5) if p.dim() == 2 else None
    return m


@skip
@pytest.mark.parametrize(("shape", "d", "h"), [((384, 384), 384, 1536), ((3, 130, 128), 128, 512), ((2, 50, 64), 64, 256), ((17, 256), 256, 1024)])
def test_transition_ieee_forward_backward(shape: tuple[int, ...], d: int, h: int) -> None:
    m = _module(d, h)
    ref = Transition(d_hidden=d, n=h // d, master_weights=True, backend="pytorch").cuda()
    ref.load_state_dict(m.state_dict())
    ref64 = Transition(d_hidden=d, n=h // d, master_weights=True, backend="pytorch").cuda().double()
    ref64.load_state_dict({k: v.double() for k, v in m.state_dict().items()})
    g = torch.Generator(device="cuda").manual_seed(1)
    x = (torch.randn(shape, device="cuda", generator=g) * 2 + 0.3).requires_grad_(True)
    dy = torch.randn(shape, device="cuda", generator=g)
    assert m.unsupported(x) is None, m.unsupported(x)
    y = m(x)
    y.backward(dy)
    got = {"y": y, "dx": x.grad, **{n_: p.grad for n_, p in m.named_parameters()}}
    x2 = x.detach().clone().requires_grad_(True)
    yr = ref(x2)
    yr.backward(dy)
    r32 = {"y": yr, "dx": x2.grad, **{n_: p.grad for n_, p in ref.named_parameters()}}
    x3 = x.detach().double().requires_grad_(True)
    y64 = ref64(x3)
    y64.backward(dy.double())
    r64 = {"y": y64, "dx": x3.grad, **{n_: p.grad for n_, p in ref64.named_parameters()}}
    for name, v in got.items():
        assert v is not None
        e, e32 = _err(v, r64[name]), _err(r32[name], r64[name])
        assert e <= max(e32, 8e-3), (name, e, e32)
        assert e < 1e-4, (name, e)  # IEEE fp32: nowhere near TF32's 1e-3


@skip
@pytest.mark.parametrize("length", [1024, 3000])
def test_pool_atoms_fp32(length: int) -> None:
    m = PoolAtoms(backend="cuda")
    g = torch.Generator(device="cuda").manual_seed(0)
    inp = m.make_inputs(length=length, samples=3, depth=None, dtype=torch.float32, device="cuda", generator=g)
    x = inp["x"].detach().clone().requires_grad_(True)
    assert m.unsupported(x, inp["idx"], inp["gather_mask"], inp["atom_mask"]) is None
    out = m(x, inp["idx"], inp["gather_mask"], inp["atom_mask"])
    gout = torch.randn_like(out)
    out.backward(gout)
    x64 = inp["x"].detach().double().requires_grad_(True)
    o64 = pool_atoms(x64, inp["idx"], inp["gather_mask"], inp["atom_mask"])
    o64.backward(gout.double())
    assert out.dtype is torch.float32
    assert _err(out, o64.detach()) < 1e-6
    assert _err(x.grad, x64.grad) < 1e-6
    again = inp["x"].detach().clone().requires_grad_(True)
    m(again, inp["idx"], inp["gather_mask"], inp["atom_mask"]).backward(gout)
    assert torch.equal(x.grad, again.grad), "the backward is bitwise reproducible"


@skip
def test_torch_mm_drop_in() -> None:
    """``ieee_f32.torch_mm`` / ``torch_addmm`` (the engine's own fp32 ``torch.mm`` / ``torch.addmm`` calls): the IEEE chain with TF32 off, ``out=`` into a column slice, the bias in the epilogue."""
    from miniworld_engine.kernels import ieee_f32

    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn((4096, 128), device="cuda", generator=g)
    w = torch.randn((256, 128), device="cuda", generator=g)
    bias = torch.randn(256, device="cuda", generator=g)
    y64 = a.double() @ w.double().t()
    buf = torch.zeros((4096, 512), device="cuda")
    out = buf[:, 128:384]
    ieee_f32.torch_mm(a, w.t(), out=out)
    assert _err(out, y64) < 1e-5
    assert not buf[:, :128].any() and not buf[:, 384:].any(), "out= writes its own columns only"
    assert _err(ieee_f32.torch_addmm(bias, a, w.t()), y64 + bias.double()) < 1e-5
    g_ = torch.randn((4096, 256), device="cuda", generator=g)
    assert _err(ieee_f32.torch_mm(g_.t(), a), g_.double().t() @ a.double()) < 1e-5  # a weight gradient: transposed A, long K
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 allowed: the call stays torch's (cuBLAS on tensor cores)
    assert not ieee_f32._owned(a, w.t())


@skip
@pytest.mark.parametrize("bias", [True, False])
def test_linear_autograd(bias: bool) -> None:
    """``ieee_f32.linear``: forward and the three gradients against float64, IEEE fp32 (TF32 stays off)."""
    from miniworld_engine.kernels import ieee_f32

    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn((4096, 128), device="cuda", generator=g, requires_grad=True)
    w = (torch.randn((384, 128), device="cuda", generator=g) / 128**0.5).requires_grad_(True)
    b = torch.randn(384, device="cuda", generator=g, requires_grad=True) if bias else None
    dy = torch.randn((4096, 384), device="cuda", generator=g)
    y = ieee_f32.linear(x, w, b)
    y.backward(dy)
    x64, w64 = x.detach().double().requires_grad_(True), w.detach().double().requires_grad_(True)
    b64 = None if b is None else b.detach().double().requires_grad_(True)
    y64 = torch.nn.functional.linear(x64, w64, b64)
    y64.backward(dy.double())
    for got, ref in ((y, y64), (x.grad, x64.grad), (w.grad, w64.grad), *(((b.grad, b64.grad),) if bias else ())):  # pyright: ignore[reportOptionalMemberAccess]
        assert _err(got, ref.detach()) < 1e-5  # pyright: ignore[reportArgumentType, reportOptionalMemberAccess]


@skip_h100
@pytest.mark.parametrize(("shape", "d", "h"), [((16384, 128), 128, 256), ((384, 384), 384, 768), ((3, 130, 128), 128, 512), ((17, 256), 256, 512), ((300, 768), 768, 1536)])
def test_transition_ieee_h100_train(shape: tuple[int, ...], d: int, h: int) -> None:
    """H100, fp32 with TF32 off: the Transition's cuda backend serves training (forward + the gradients) on the IEEE fp32 chain (it was refused to Triton)."""
    m = _module(d, h)
    ref64 = Transition(d_hidden=d, n=h // d, master_weights=True, backend="pytorch").cuda().double()
    ref64.load_state_dict({k: v.double() for k, v in m.state_dict().items()})
    g = torch.Generator(device="cuda").manual_seed(1)
    x = (torch.randn(shape, device="cuda", generator=g) * 2 + 0.3).requires_grad_(True)
    dy = torch.randn(shape, device="cuda", generator=g)
    assert m.unsupported(x) is None, m.unsupported(x)
    y = m(x)
    y.backward(dy)
    x3 = x.detach().double().requires_grad_(True)
    y64 = ref64(x3)
    y64.backward(dy.double())
    got = {"y": y, "dx": x.grad, **{n_: p.grad for n_, p in m.named_parameters()}}
    ref = {"y": y64, "dx": x3.grad, **{n_: p.grad for n_, p in ref64.named_parameters()}}
    for name, v in got.items():
        assert v is not None and _err(v, ref[name]) < 1e-5, (name, _err(v, ref[name]))
    with torch.no_grad():  # inference
        assert _err(m(x.detach()), y64.detach()) < 1e-5
