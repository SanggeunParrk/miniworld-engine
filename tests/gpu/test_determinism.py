"""The H100 CUDA kernels are deterministic: two identical calls (forward and, in train, backward with the same upstream gradient)
give bitwise-equal outputs and gradients."""

from __future__ import annotations

import pytest
import torch

from miniworld_engine.measure import accuracy
from miniworld_engine.measure import config as cfgmod

pytestmark = [pytest.mark.gpu, pytest.mark.h100]

MODULES = ("TriangleMultiplication", "TriangleAttention", "Transition", "AttentionPairBias", "LayerNorm", "TokenDiTProtenix")


def _case(module: str, mode: str) -> cfgmod.Case:
    cfg = cfgmod.from_dict(
        {"modules": [module], "lengths": "min", "depths": "min", "samples": "min", "dtypes": ["master"], "modes": [mode], "device": "cuda"}
    )
    return next(iter(cfgmod.cases(cfg)))


def _call(module: torch.nn.Module, inputs: dict, upstream: list[torch.Tensor] | None, train: bool) -> list[torch.Tensor]:
    inputs = {
        k: v.detach().clone().requires_grad_(train) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
        for k, v in inputs.items()
    }
    for p in module.parameters():
        p.grad = None
    with torch.set_grad_enabled(train), torch.autocast("cuda", dtype=torch.bfloat16):
        out = module(**inputs)
    outs = [out] if isinstance(out, torch.Tensor) else [t for t in out if isinstance(t, torch.Tensor) and t.is_floating_point()]
    values = [t.detach().clone() for t in outs]
    if train:
        assert upstream is not None
        torch.autograd.backward(
            [t for t in outs if t.requires_grad], [g.to(t.dtype) for t, g in zip(outs, upstream, strict=True) if t.requires_grad]
        )
        values += [v.grad.clone() for v in inputs.values() if isinstance(v, torch.Tensor) and v.grad is not None]
        values += [p.grad.clone() for p in module.parameters() if p.grad is not None]
    return values


@pytest.mark.parametrize("mode", ["eval", "train"])
@pytest.mark.parametrize("module", MODULES)
def test_cuda_is_bitwise_deterministic(module: str, mode: str):
    case = _case(module, mode)
    if "cuda" not in case.cls.backends:
        pytest.skip(f"{module} has no cuda backend")
    m = cfgmod.build(case, device="cuda").set_backend("cuda")
    accuracy._quiet(m)  # dropout off: its masks are random draws, not numerics (as in the measurer)
    inputs = cfgmod.inputs(case, m, device="cuda")
    train = mode == "train"
    with torch.set_grad_enabled(train), torch.autocast("cuda", dtype=torch.bfloat16):
        why = m.unsupported(**inputs)
    if why is not None:
        pytest.skip(f"cuda does not serve {case.label}: {why}")
    gen = torch.Generator(device="cuda").manual_seed(1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        shapes = m(**inputs)
    shapes = [shapes] if isinstance(shapes, torch.Tensor) else [t for t in shapes if isinstance(t, torch.Tensor) and t.is_floating_point()]
    upstream = [torch.randn(t.shape, device="cuda", generator=gen) for t in shapes] if train else None
    first = _call(m, inputs, upstream, train)
    second = _call(m, inputs, upstream, train)
    assert len(first) == len(second) and first
    for i, (a, b) in enumerate(zip(first, second, strict=True)):
        assert torch.equal(a, b), (
            f"{case.label}: tensor {i} differs between two identical calls (max |diff| {(a - b).abs().max().item():.3e})"
        )


#: TriMul training paths past the minimum case above: (class, d_pair, length, outgoing): the single-direction L384 legacy path
#: (incoming D256), the bidirectional L384 / L768 legacy paths (D384 / D512: their LayerNorm dgamma / dbeta) and the B200-style port at
#: D384 (the W_o column sums).
TRIMUL_TRAIN_CELLS = (
    ("TriangleMultiplication", 256, 384, False),
    ("BidirectionalTriangleMultiplication", 384, 384, None),
    ("BidirectionalTriangleMultiplication", 512, 768, None),
    ("TriangleMultiplication", 384, 640, True),
)


@pytest.mark.parametrize(("module", "d_pair", "length", "outgoing"), TRIMUL_TRAIN_CELLS)
def test_trimul_training_paths_are_bitwise_deterministic(module: str, d_pair: int, length: int, outgoing: bool | None):
    cfg = cfgmod.from_dict(
        {
            "modules": [module],
            "lengths": [length],
            "depths": "min",
            "samples": "min",
            "dtypes": ["master"],
            "modes": ["train"],
            "device": "cuda",
        }
    )
    case = next(c for c in cfgmod.cases(cfg) if c.kwargs.get("d_pair") == d_pair and c.kwargs.get("outgoing") == outgoing)
    m = cfgmod.build(case, device="cuda").set_backend("cuda")
    accuracy._quiet(m)
    inputs = cfgmod.inputs(case, m, device="cuda")
    with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        why = m.unsupported(**inputs)
    if why is not None:
        pytest.skip(f"cuda does not serve {case.label}: {why}")
    gen = torch.Generator(device="cuda").manual_seed(1)
    upstream = [torch.randn(inputs["pair"].shape, device="cuda", generator=gen)]
    first = _call(m, inputs, upstream, True)
    second = _call(m, inputs, upstream, True)
    for i, (a, b) in enumerate(zip(first, second, strict=True)):
        assert torch.equal(a, b), f"{case.label}: tensor {i} differs between two identical calls"
