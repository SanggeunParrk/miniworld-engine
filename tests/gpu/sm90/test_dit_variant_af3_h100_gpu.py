"""AlphaFold 3's own token and atom DiT stacks (``dit_variant.AF3`` / ``AF3_ATOM``: Boltz-2's rows with the up-gate off, sequential, one
shared pair LayerNorm) on the H100 CUDA paths, at the registry's declared shapes (``augmented_attention`` token_single 768 / d_cond 384 /
d_pair 128 / 16 heads, lengths 128..768; atom_single 128 / 128 / 16 / 4 heads, lengths 1024..8192).

Inference (bf16, no gradient) and training (bf16 under autocast with fp32 master parameters, forward and backward) against the fp64 PyTorch
reference: the engine's error is held to 1.5 x the error of the same-dtype PyTorch composition against fp64 (the output and every gradient;
a tensor whose true gradient is zero, the pair LayerNorm offset, is held to a floor of the largest gradient norm)."""

import copy

import pytest
import torch

from miniworld_engine.kernels.dit_variants import interface
from miniworld_engine.kernels.dit_variants.cuda import train
from miniworld_engine.modules.dit_variant import (
    AF3,
    AF3_ATOM,
    AtomDiTVariantStack,
    DiTVariantStack,
)
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="Hopper (sm_90) required",
    ),
]

ERR_RATIO = 1.5
FLOOR = 1e-4


def _init(stack):
    torch.manual_seed(0)
    with torch.no_grad():
        for p in stack.parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[1] ** -0.5)
            elif p.ndim == 1 and p.numel() > 1:
                p.add_(torch.randn_like(p) * 0.1)
    return stack.cuda()


def _case(kind, length, a, *, masked=True):
    """(stack, inputs, mask, dy, live). ``masked``: a tenth of the atoms / the last 40 tokens are padding (else no mask)."""
    g = torch.Generator(device="cuda").manual_seed(1)
    if kind == "atom":
        n, live = length, int(length * 0.9) if masked else length
        stack = _init(AtomDiTVariantStack(2, 128, 128, 16, 2, key_window="slide", **AF3_ATOM))
        shapes = {"single": (a, n, 128), "cond": (n, 128), "pair": (n // 32, 32, 128, 16)}
    else:
        n, live = length, length - 40 if masked else length
        stack = _init(DiTVariantStack(2, 768, AF3["d_cond"], AF3["d_pair"], 16, 2, **{k: v for k, v in AF3.items() if k not in ("d_cond", "d_pair")}))
        shapes = {"single": (a, n, 768), "cond": (n, AF3["d_cond"]), "pair": (n, n, AF3["d_pair"])}
    mask = torch.arange(n, device="cuda") < live if masked else None
    inputs = {k: torch.randn(*s, device="cuda", generator=g) for k, s in shapes.items()}
    dy = torch.randn(*shapes["single"], device="cuda", generator=g)
    dy[:, live:] = 0       # a padding row's output is not read (the slid edge windows differ from the reference's there)
    return stack, inputs, mask, dy, live


def _rel(a, b, rows=slice(None)):
    a, b = a[:, rows].double(), b[:, rows].double()
    return ((a - b).norm() / b.norm()).item()


# ----------------------------------------------------------------------------------------------------------------- inference
@pytest.mark.parametrize(("length", "a", "masked"), [(128, 5, True), (256, 2, False), (384, 1, True)])
def test_token_inference(length, a, masked):
    stack, inputs, mask, _, live = _case("token", length, a, masked=masked)
    ref64 = copy.deepcopy(stack).double()
    ref64.implementation = ImplementationType.PYTORCH
    ref16 = copy.deepcopy(stack).to(torch.bfloat16)
    ref16.implementation = ImplementationType.PYTORCH
    eng = stack.to(torch.bfloat16)
    bf = [inputs[k].to(torch.bfloat16) for k in ("single", "cond", "pair")]
    with torch.no_grad():
        assert interface.serves(eng, *bf, mask), "the CUDA inference route must serve the AF3 token stack"
        assert eng.serving(*bf, mask) == "cuda"
        gold = ref64(*[inputs[k].double() for k in ("single", "cond", "pair")], mask)
        base = ref16(*bf, mask)
        got = eng(*bf, mask)
    assert torch.isfinite(got).all()
    rows = slice(0, live)
    err, err16 = _rel(got, gold, rows), _rel(base, gold, rows)
    assert err <= max(ERR_RATIO * err16, 1e-2), f"engine {err:.2e} vs bf16 reference {err16:.2e}"


@pytest.mark.parametrize(("n", "a", "masked"), [(1024, 5, True), (2048, 2, False)])
def test_atom_inference(n, a, masked):
    stack, inputs, mask, _, live = _case("atom", n, a, masked=masked)
    ref64 = copy.deepcopy(stack).double()
    ref64.implementation = ImplementationType.PYTORCH
    ref16 = copy.deepcopy(stack).to(torch.bfloat16)
    ref16.implementation = ImplementationType.PYTORCH
    eng = stack.to(torch.bfloat16)
    bf = [inputs[k].to(torch.bfloat16) for k in ("single", "cond", "pair")]
    with torch.no_grad():
        assert interface.atom_serves(eng, *bf, mask, None), "the CUDA inference route must serve the AF3 atom stack"
        assert eng.serving(*bf, mask, None) == "cuda"
        gold = ref64(*[inputs[k].double() for k in ("single", "cond", "pair")], mask, None)
        base = ref16(*bf, mask, None)
        got = eng(*bf, mask, None)
    assert torch.isfinite(got[:, :live]).all()
    rows = slice(0, live)
    err, err16 = _rel(got, gold, rows), _rel(base, gold, rows)
    assert err <= max(ERR_RATIO * err16, 1e-2), f"engine {err:.2e} vs bf16 reference {err16:.2e}"


# -------------------------------------------------------------------------------------------------------------------- training
def _run(kind, stack, inputs, mask, dy, live, dtype, bf16, impl):
    m = copy.deepcopy(stack)
    m.implementation = impl
    ins = {k: v.clone().to(dtype).requires_grad_() for k, v in inputs.items()}
    args = (ins["single"], ins["cond"], ins["pair"], mask) + ((None,) if kind == "atom" else ())
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        if impl == ImplementationType.MINIWORLD and dtype != torch.float64:
            assert (train.atom_serves if kind == "atom" else train.serves)(m, *args), "the CUDA training route must serve this call"
            assert m.serving(*args) == "cuda"
        y = m(*args)
    y.backward(dy.to(y.dtype))
    grads = {k: (v.grad if v.grad is not None else torch.zeros_like(v)).double() for k, v in ins.items()}
    grads["single"] = grads["single"][:, :live]
    for name, p in m.named_parameters():
        grads["P." + name] = (p.grad if p.grad is not None else torch.zeros_like(p)).double()
    return y.double()[:, :live], grads


@pytest.mark.parametrize(("kind", "length", "masked"), [("token", 256, True), ("token", 128, False), ("atom", 1024, True), ("atom", 1024, False)])
def test_training_matches_fp64(kind, length, masked):
    stack, inputs, mask, dy, live = _case(kind, length, 3, masked=masked)
    saved = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        y64, g64 = _run(kind, stack.double(), {k: v.double() for k, v in inputs.items()}, mask, dy.double(), live, torch.float64, False,
                        ImplementationType.PYTORCH)
        stack = stack.float()
        yr, gr = _run(kind, stack, inputs, mask, dy, live, torch.bfloat16, True, ImplementationType.PYTORCH)
        ye, ge = _run(kind, stack, inputs, mask, dy, live, torch.bfloat16, True, ImplementationType.MINIWORLD)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved
    assert torch.isfinite(ye).all()
    assert (ye - y64).norm() / y64.norm() <= ERR_RATIO * (yr - y64).norm() / y64.norm(), "output"
    floor = FLOOR * max(g.norm().item() for g in g64.values())
    bad = []
    for name, ref in g64.items():
        err_e, err_r = (ge[name] - ref).norm().item(), (gr[name] - ref).norm().item()
        if err_e > ERR_RATIO * err_r + floor:
            bad.append((name, err_e / max(ref.norm().item(), 1e-30), err_r / max(ref.norm().item(), 1e-30)))
    assert not bad, bad
