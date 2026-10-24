"""The DiT block variants (Boltz-2, RoseTTAFold3, Chai-1; token and atom stacks) on the portable Triton path
(``kernels/dit_variants/triton/portable.py``) against the fp64 PyTorch reference: forward, the gradients of every input and every
parameter, masked tokens / atoms, the slid and the clipped atom window, Chai-1's same-conformer mask."""

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.kernels.dit_variants.triton import portable
from miniworld_engine.modules.dit_variant import (
    BOLTZ2,
    BOLTZ2_ATOM,
    CHAI1,
    CHAI1_ATOM,
    ROSETTAFOLD3,
    ROSETTAFOLD3_ATOM,
    AtomDiTVariantStack,
    DiTVariantStack,
)
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def _triton_everywhere():
    previous = settings.configure(force_triton_kernels=True)
    yield
    settings.configure(force_triton_kernels=previous.force_triton_kernels)


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _randomise(module, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            r = torch.randn(p.shape, generator=g)
            p.copy_((r / p.shape[-1] ** 0.5 if p.ndim > 1 else 0.1 * r + (1.0 if "weight" in name and "norm" in name else 0.0)).to(p.dtype))
    return module


def _compare(make, args, call, ok, dtype, live):
    ref = make(ImplementationType.PYTORCH).cuda().double()
    served = make(ImplementationType.MINIWORLD).cuda()
    served.load_state_dict(ref.state_dict())
    own = make(ImplementationType.PYTORCH).cuda()
    own.load_state_dict(ref.state_dict())
    if dtype == torch.bfloat16:
        served, own = served.bfloat16(), own.bfloat16()
    go = torch.randn_like(args[0])

    def run(module, dt):
        xs = [t.to(dt).clone().requires_grad_() for t in args]
        for p in module.parameters():
            p.grad = None
        out = call(module, xs)
        out.backward(go.to(out.dtype))
        return out.detach(), [x.grad for x in xs], {k: p.grad.double() for k, p in module.named_parameters() if p.grad is not None}

    assert ok(served, [t.to(dtype) for t in args])
    want, base, got = run(ref, torch.float64), run(own, dtype), run(served, dtype)
    floor = 6e-3 if dtype == torch.float32 else 0.0
    assert _rel(got[0][:, live], want[0][:, live]) < 2 * _rel(base[0][:, live], want[0][:, live]) + floor + 1e-6
    for g, b, w in zip(got[1], base[1], want[1], strict=True):
        assert _rel(g, w) < 2 * _rel(b, w) + floor + 1e-6
    # a parameter whose true gradient is (nearly) zero -- a LayerNorm bias on the keys or on the pair bias shifts every logit of a
    # query by one constant, which the softmax cancels -- is judged against the largest parameter gradient: its error is noise
    largest = max(w.norm() for w in want[2].values())
    def err(t, w):
        return float((t.double() - w).norm() / max(w.norm(), 1e-12))

    for key, w in want[2].items():
        if w.norm() < 0.02 * largest:
            assert (got[2][key].double() - w).norm() < 0.02 * largest, key
        else:
            assert err(got[2][key], w) < 2 * err(base[2][key], w) + floor + 1e-6, key


TOKEN = [("boltz2", BOLTZ2), ("rosettafold3", ROSETTAFOLD3), ("chai1", CHAI1)]
ATOM = [("boltz2", BOLTZ2_ATOM), ("rosettafold3", ROSETTAFOLD3_ATOM), ("chai1", CHAI1_ATOM)]


@pytest.mark.parametrize(("name", "options"), TOKEN, ids=[t[0] for t in TOKEN])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("real", [300, None])
def test_the_token_stack(name, options, dtype, real):
    length, samples = 384, 2
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(samples, length, 768, device="cuda", generator=g)
    s = torch.randn(length, options["d_cond"], device="cuda", generator=g)
    z = torch.randn(length, length, options["d_pair"], device="cuda", generator=g)
    mask = None
    if real is not None:
        mask = torch.zeros(length, dtype=torch.bool, device="cuda")
        mask[:real] = True

    def make(impl):
        return _randomise(DiTVariantStack(3, 768, options["d_cond"], options["d_pair"], 16, 2, implementation=impl,
                                          **{k: v for k, v in options.items() if k not in ("d_cond", "d_pair")}))

    _compare(make, (x, s, z), lambda m, xs: m(*xs, mask), lambda m, xs: portable.serves(m, *xs, mask), dtype, slice(0, real or length))


@pytest.mark.parametrize(("name", "options"), ATOM, ids=[t[0] for t in ATOM])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(("key_window", "real"), [("slide", 450), ("pad", 512), ("pad", 300)])
def test_the_atom_stack(name, options, dtype, key_window, real):
    atoms, samples = 512, 2
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(samples, atoms, 128, device="cuda", generator=g)
    c = torch.randn(atoms, 128, device="cuda", generator=g)
    z = torch.randn(atoms // 32, 32, 128, 16, device="cuda", generator=g)
    mask = torch.zeros(atoms, dtype=torch.bool, device="cuda")
    mask[:real] = True
    uid = (torch.arange(atoms, device="cuda") // 45).int() if name == "chai1" else None

    def make(impl):
        return _randomise(AtomDiTVariantStack(
            3, 128, 128, 16, 2, parallel=options["parallel"], per_block_pair_norm=options["per_block_pair_norm"],
            pair_norm_bias=options["pair_norm_bias"], key_window=key_window, implementation=impl,
            **{k: v for k, v in options.items() if k in ("identity_scale", "gating", "project_output", "kq_norm", "up_gate")}))

    _compare(make, (x, c, z), lambda m, xs: m(*xs, mask, uid), lambda m, xs: portable.atom_serves(m, *xs, mask, uid), dtype, slice(0, real))


def test_the_pytorch_implementation_and_unsupported_calls_stay_on_the_reference():
    options = BOLTZ2
    stack = _randomise(DiTVariantStack(1, 768, options["d_cond"], options["d_pair"], 16, 2, implementation=ImplementationType.PYTORCH,
                                       **{k: v for k, v in options.items() if k not in ("d_cond", "d_pair")})).cuda()
    x = torch.randn(1, 128, 768, device="cuda")
    s = torch.randn(128, options["d_cond"], device="cuda")
    z = torch.randn(128, 128, options["d_pair"], device="cuda")
    assert not portable.serves(stack, x, s, z, None)
    stack.implementation = ImplementationType.MINIWORLD
    assert portable.serves(stack, x, s, z, None)
    assert not portable.serves(stack, x.double(), s, z, None)
    assert not portable.serves(stack, x.cpu(), s, z, None)
