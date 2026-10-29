"""``torch.compile`` of B200 engine modules through a no-grad call (a recycle) and then a grad call (the step), in the training dtype
(fp32 master parameters under bf16 autocast) -- what the MiniWorld trainer compiles. The no-grad calls take the modules' inference paths,
which read their weight forms through ``kernels._capture`` (``param_forms`` / ``lookup`` / ``hoisted_table``); Dynamo raised there before
(``InternalTorchDynamoError: NameError: name 'torch' is not defined`` from the Transition's wide path at the first no-grad recycle).

Every case compiles with ``fullgraph=True`` (a graph break is an error), cold (no eager call first), and must match eager: the no-grad
output, the grad output and the parameter gradients. ``test_compiled_recycle_graph_replay_sees_weight_updates``: the compiled no-grad
recycle + grad step captured in a CUDA graph (as the random-recycle trainer does) replays with the weights an in-place update wrote."""
from __future__ import annotations

import gc

import pytest
import torch
import torch._dynamo as dynamo
from portable_helpers import relative_float as relative
from settings_helpers import policy_b200_portable_off as policy  # noqa: F401

from miniworld_engine.modules import (
    AdaptiveLayerNorm,
    AttentionPairBias,
    BidirectionalTriangleMultiplication,
    ConditionedTransition,
    MSAPairWeightedAveraging,
    OuterProductMean,
    Transition,
    TriangleAttention,
)
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]
MW = ImplementationType.MINIWORLD
L = 128


def _rand(*shape, grad=False, dtype=torch.bfloat16):
    return (torch.randn(*shape, device="cuda") * 0.5).to(dtype).requires_grad_(grad)


def _randomize(m):
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim >= 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return m.cuda()


#: name -> (fp32-master module, inputs(grad)); the widths MiniWorld and FoldForge build on B200
CASES = {
    "Transition D128 n4 (fused)": lambda: (Transition(128, 4, implementation=MW), lambda g: (_rand(1, L, L, 128, grad=g),)),
    "Transition D64 n4 (wide)": lambda: (Transition(64, 4, implementation=MW), lambda g: (_rand(1, L, L, 64, grad=g),)),
    "Transition D256 n4 (wide)": lambda: (Transition(256, 4, implementation=MW), lambda g: (_rand(1, 4 * L, 256, grad=g),)),
    "Transition D384 n4 (single)": lambda: (Transition(384, 4, implementation=MW), lambda g: (_rand(1, 4 * L, 384, grad=g),)),
    "BidirectionalTriMul D128": lambda: (BidirectionalTriangleMultiplication(128, implementation=MW, p_drop=0.0),
                                         lambda g: (_rand(1, L, L, 128, grad=g), torch.ones(1, L, dtype=torch.bool, device="cuda"))),
    "BidirectionalTriMul D64": lambda: (BidirectionalTriangleMultiplication(64, implementation=MW, p_drop=0.0),
                                        lambda g: (_rand(1, L, L, 64, grad=g), torch.ones(1, L, dtype=torch.bool, device="cuda"))),
    "OuterProductMean 64->128": lambda: (OuterProductMean(64, 128, 32, implementation=MW),
                                         lambda g: (_rand(1, 256, L, 64, grad=g), torch.ones(1, 256, L, dtype=torch.bool, device="cuda"))),
    "MSAPairWeightedAveraging 64/128": lambda: (MSAPairWeightedAveraging(64, 128, 8, implementation=MW),
                                                lambda g: (_rand(1, 256, L, 64, grad=g), _rand(1, L, L, 128),
                                                           torch.ones(1, L, dtype=torch.bool, device="cuda"))),
    "TriangleAttention D128 4h": lambda: (TriangleAttention(128, 4, implementation=MW, p_drop=0.0),
                                          lambda g: (_rand(1, L, L, 128, grad=g), None)),
    "AttentionPairBias 384/128 16h": lambda: (AttentionPairBias(384, 128, 16, implementation=MW),
                                              lambda g: (_rand(1, 256, 384, grad=g), _rand(1, 256, 256, 128), None)),
    "AdaptiveLayerNorm 128/128": lambda: (AdaptiveLayerNorm(128, 128, implementation=MW),
                                          lambda g: (_rand(4, 1024, 128, grad=g), _rand(4, 1024, 128))),
    "ConditionedTransition 128/128": lambda: (ConditionedTransition(128, 128, 2, implementation=MW),
                                              lambda g: (_rand(4, 1024, 128, grad=g), _rand(4, 1024, 128))),
}


@pytest.fixture(autouse=True)
def _reset():
    dynamo.reset()
    yield
    dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()


def _run(m, inputs, probe):
    """A no-grad call (recycle) on detached inputs, then a grad call; -> (no-grad out, grad out, grads of the inputs and params)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            y0 = m(*inputs(False))
        args = inputs(True)
        y1 = m(*args)
    wrt = [a for a in args if torch.is_tensor(a) and a.requires_grad] + list(m.parameters())
    gs = torch.autograd.grad((y1.float() * probe(y1)).sum(), wrt, allow_unused=True)
    return y0.detach(), y1.detach(), [g for g in gs if g is not None]


@pytest.mark.parametrize("name", list(CASES))
def test_compile_no_grad_then_grad_fullgraph(name):
    torch.manual_seed(3)
    make, inputs_fn = CASES[name]()
    m = _randomize(make).train()
    torch.manual_seed(4)
    fixed = {}

    def inputs(grad):                       # the same values for the eager and the compiled run
        if "x" not in fixed:
            fixed["x"] = inputs_fn(False)
        return tuple(t.detach().requires_grad_(grad and i == 0) if torch.is_tensor(t) and t.is_floating_point() else t
                     for i, t in enumerate(fixed["x"]))

    probes = {}

    def probe(y):
        if "p" not in probes:
            probes["p"] = torch.randn(y.shape, device=y.device)
        return probes["p"]

    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        from torch._dynamo.backends.registry import lookup_backend

        return lookup_backend("aot_eager")(gm, example_inputs)

    cm = torch.compile(m, backend=backend, fullgraph=True, dynamic=False)
    got = _run(cm, inputs, probe)                                  # cold: the compiled run goes first
    want = _run(m, inputs, probe)
    assert len(graphs) == 2, f"{name}: {len(graphs)} graphs (one per grad mode expected)"
    assert len(got[2]) == len(want[2]) and len(want[2]) > 1
    for i, (a, b) in enumerate(zip([got[0], got[1], *got[2]], [want[0], want[1], *want[2]], strict=True)):
        err = relative(a, b)
        assert err < 1e-2, f"{name}: output {i} compiled vs eager relative error {err:.2e}"


@pytest.mark.parametrize("width", [64, 128])
def test_compiled_recycle_graph_replay_sees_weight_updates(width):
    """A compiled (inductor) no-grad recycle then a grad call, captured in one CUDA graph after warm-up (the random-recycle trainer's
    shape); an in-place update of the fp32 master weights, then a replay: the replay matches the eager step with the new weights."""
    torch.manual_seed(5)
    m = _randomize(Transition(width, 4, implementation=MW)).train()
    x = _rand(1, L, L, width)
    probe = torch.randn(1, L, L, width, device="cuda")
    cm = torch.compile(m, fullgraph=True, dynamic=False)

    def step(mod):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                y = mod(x)
            y = mod(y.detach())
        gs = torch.autograd.grad((y.float() * probe).sum(), list(m.parameters()))
        return (y.detach(), *gs)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step(cm)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = step(cm)
    g.replay()
    torch.cuda.synchronize()
    before = [t.clone() for t in out]
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * p.abs().mean().clamp_min(1e-3) * torch.randn_like(p))
    g.replay()
    torch.cuda.synchronize()
    replayed = [t.clone() for t in out]
    eager = step(m)
    for i, (r, e, b) in enumerate(zip(replayed, eager, before, strict=True)):
        moved = relative(e, b)
        assert moved > 1e-2, f"output {i}: the update does not change the result ({moved:.1e})"
        assert relative(r, e) < 0.05 * moved, f"output {i}: replay {relative(r, e):.2e} off eager (update moved it {moved:.2e})"
