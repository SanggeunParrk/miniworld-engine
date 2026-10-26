"""The Boltz-2 / RoseTTAFold3 / Chai-1 DiT variant stacks (token and atom) on the H100 TRAINING path (``kernels/dit_variants/cuda/train.py``)
against the fp64 PyTorch reference: the output and every gradient (inputs, conditioning, pair, each parameter) of the engine are held to
1.5 x the error of the same-dtype PyTorch reference against fp64 (bf16 under autocast with fp32 master parameters; fp32 with TF32 matmuls).
A tensor whose fp64 gradient is mathematically zero (the LayerNorm bias of k and the pair LayerNorm bias only shift every logit of a row by
one constant, which the softmax cancels) is held to an absolute floor instead of a relative error."""

import copy

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.kernels.dit_variants.cuda import train
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

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="Hopper (sm_90) required",
    ),
]

TOKEN = {"boltz2": BOLTZ2, "rf3": ROSETTAFOLD3, "chai1": CHAI1}
ATOM = {"boltz2": BOLTZ2_ATOM, "rf3": ROSETTAFOLD3_ATOM, "chai1": CHAI1_ATOM}
ERR_RATIO = 1.5
FLOOR = 1e-4          # of the largest gradient norm, for the tensors whose true gradient is zero


def _init(stack):
    torch.manual_seed(0)
    with torch.no_grad():
        for p in stack.parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[1] ** -0.5)
            elif p.ndim == 1 and p.numel() > 1:
                p.add_(torch.randn_like(p) * 0.1)
    return stack.cuda()


def _case(kind, model, a, length):
    """(stack, inputs, mask, uid, dy, live rows)."""
    g = torch.Generator(device="cuda").manual_seed(1)
    if kind == "atom":
        n = length * 8
        stack = _init(AtomDiTVariantStack(2, 128, 128, 16, 2, **ATOM[model]))
        shapes = {"single": (a, n, 128), "cond": (n, 128), "pair": (n // 32, 32, 128, 16)}
        live = int(n * 0.9)
        mask = torch.arange(n, device="cuda") < live
        uid = (torch.arange(n, device="cuda") // 45).int() if model == "chai1" else None
    else:
        o = TOKEN[model]
        n, live = length, length - 40
        stack = _init(DiTVariantStack(2, 768, o["d_cond"], o["d_pair"], 16, 2, **{k: v for k, v in o.items() if k not in ("d_cond", "d_pair")}))
        shapes = {"single": (a, n, 768), "cond": (n, o["d_cond"]), "pair": (n, n, o["d_pair"])}
        mask = torch.arange(n, device="cuda") < live
        uid = None
    inputs = {k: torch.randn(*s, device="cuda", generator=g) for k, s in shapes.items()}
    dy = torch.randn(*shapes["single"], device="cuda", generator=g)
    if kind == "atom":
        dy[:, live:] = 0      # a padding atom's output is not read (the slid edge windows differ from the reference's there)
    return stack, inputs, mask, uid, dy, live


def _run(kind, stack, inputs, mask, uid, dy, live, dtype, bf16, impl):
    m = copy.deepcopy(stack)
    m.implementation = impl
    ins = {k: v.clone().to(dtype).requires_grad_() for k, v in inputs.items()}
    args = (ins["single"], ins["cond"], ins["pair"], mask) + ((uid,) if kind == "atom" else ())
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        if impl == ImplementationType.MINIWORLD and dtype != torch.float64:
            assert (train.atom_serves if kind == "atom" else train.serves)(m, *args), "the CUDA training route must serve this call"
        y = m(*args)
    y.backward(dy.to(y.dtype))
    grads = {k: (v.grad if v.grad is not None else torch.zeros_like(v)).double() for k, v in ins.items()}
    grads["single"] = grads["single"][:, :live]
    for name, p in m.named_parameters():
        grads["P." + name] = (p.grad if p.grad is not None else torch.zeros_like(p)).double()
    return y.double()[:, :live], grads


def _rel(a, b, floor=0.0):
    return ((a - b).norm() / b.norm().clamp_min(floor)).item()


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
@pytest.mark.parametrize("model", list(TOKEN))
@pytest.mark.parametrize("kind", ["token", "atom"])
def test_training_matches_fp64(kind, model, precision):
    a, length = 6, 256
    stack, inputs, mask, uid, dy, live = _case(kind, model, a, length)
    bf16 = precision == "bf16"
    dtype = torch.bfloat16 if bf16 else torch.float32
    saved = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        y64, g64 = _run(kind, stack.double(), {k: v.double() for k, v in inputs.items()}, mask, uid, dy.double(), live, torch.float64, False,
                        ImplementationType.PYTORCH)
        torch.backends.cuda.matmul.allow_tf32 = not bf16
        stack = stack.float()
        yr, gr = _run(kind, stack, inputs, mask, uid, dy, live, dtype, bf16, ImplementationType.PYTORCH)
        ye, ge = _run(kind, stack, inputs, mask, uid, dy, live, dtype, bf16, ImplementationType.MINIWORLD)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved
    assert torch.isfinite(ye).all()
    assert _rel(ye, y64) <= ERR_RATIO * _rel(yr, y64), "output"
    floor = FLOOR * max(g.norm().item() for g in g64.values())
    bad = []
    for name, ref in g64.items():
        err_e, err_r = (ge[name] - ref).norm().item(), (gr[name] - ref).norm().item()
        if err_e > ERR_RATIO * err_r + floor:
            bad.append((name, err_e / max(ref.norm().item(), 1e-30), err_r / max(ref.norm().item(), 1e-30)))
    assert not bad, bad


@pytest.mark.parametrize("kind", ["token", "atom"])
def test_training_is_bitwise_repeatable(kind):
    """Every row kernel sums in a fixed order (partial rows, no atomics) and so do the attention backwards: two runs agree exactly."""
    stack, inputs, mask, uid, dy, live = _case(kind, "rf3", 3, 256)
    runs = [_run(kind, stack, inputs, mask, uid, dy, live, torch.bfloat16, True, ImplementationType.MINIWORLD) for _ in range(2)]
    assert torch.equal(runs[0][0], runs[1][0])
    for name in runs[0][1]:
        assert torch.equal(runs[0][1][name], runs[1][1][name]), name


def test_the_training_switch_sends_the_call_to_the_portable_route():
    stack, inputs, mask, uid, dy, live = _case("token", "chai1", 3, 256)
    ins = [inputs[k].bfloat16().requires_grad_() for k in ("single", "cond", "pair")]
    previous = settings.current().dit_variant_train_cuda
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert train.serves(stack, *ins, mask)
            settings.configure(dit_variant_train_cuda=False)
            assert not train.serves(stack, *ins, mask)
    finally:
        settings.configure(dit_variant_train_cuda=previous)


def test_a_call_without_a_gradient_or_off_the_layout_is_not_served():
    stack, inputs, mask, uid, dy, live = _case("token", "boltz2", 3, 256)
    single, cond, pair = (inputs[k].bfloat16() for k in ("single", "cond", "pair"))
    with torch.no_grad():
        assert not train.serves(stack, single, cond, pair, mask)
    single.requires_grad_()
    assert train.serves(stack, single, cond, pair, mask)
    assert not train.serves(stack, single[:, :200], cond[:200], pair[:200, :200], mask[:200])      # L not a multiple of 128
    assert not train.serves(stack, single, cond[None].expand(3, -1, -1), pair, mask)               # a conditioning row set per sample
