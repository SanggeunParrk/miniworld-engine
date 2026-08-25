"""Per-kernel correctness does not add up to a correct model. This asserts the sum.

47 test files, all unit or contract: each kernel against its own reference, on inputs the
checker built for it. None of them answers the question a consumer actually has -- if I swap
these kernels in, does my model produce the same numbers? A tolerance that is comfortable on one
kernel's own inputs can compound across depth; a kernel can be right on synthetic input and wrong
on what the previous layer emits; a dtype can promote silently in one path and not the other.

So: one Pairformer stack, built twice from the SAME weights, run on the SAME input, once with
`ImplementationType.PYTORCH` and once with `MINIWORLD`. That is the substitution a consumer makes,
and this is the whole-stack error it produces.

Three things keep it from being vacuous:

* the weights are copied, not re-seeded, so any difference is the kernels and not initialisation;
* dispatch is asserted to have chosen a non-PyTorch backend, or "agreement" would just be the
  reference agreeing with itself;
* a deliberately corrupted weight must break the comparison, which is checked here rather than
  taken on faith.
"""
from __future__ import annotations

import copy

import pytest

pytestmark = pytest.mark.gpu

#: Small enough to run in seconds, deep enough that per-layer error has to compound to show up.
#: n_block=4 is the AF3 default for this stack.
#:
#: d_pair stays at the config default. Shrinking it to 64 while `d_hidden_tri_multi` kept its
#: default 128 gave `mat1 and mat2 shapes cannot be multiplied (16384x64 and 128x64)` -- the
#: hidden widths are not independent of d_pair, so the cheap axis to shrink is L.
N_BLOCK, D_PAIR, L, B = 4, 128, 64, 1

#: How large the randomised weights are, and it is the parameter this whole test turns on. Swept on
#: an A6000, 4 blocks, L=64, comparing the honest kernel-vs-reference error against the error from
#: replacing one projection with noise:
#:
#:     scale   branch/input   honest      corrupted    separation
#:      0.05          0.04    7.53e-03     1.66e-02        2.2x
#:      0.15          0.75    1.30e-02     1.09e-01        8.4x
#:      0.30         14.60    2.00e-02     4.80e-01         24x
#:      0.60        477.72    6.86e-02     8.33e-01         12x
#:
#: 0.05 leaves the residual dominating the output, so a completely wrong branch barely moves it --
#: the test would have no teeth. 0.30 and above have activations growing 14x and 478x through four
#: blocks, which is not a regime any trained model is in. 0.15 puts the branch on the same order as
#: what it is added to, which is the realistic one, and separates honest from broken by 8.4x.
WEIGHT_SCALE = 0.15

#: Budget: 4x the 1.30e-02 measured at that scale, the same margin and reasoning as
#: `run_all.RTOL_MARGIN`. NOT a per-kernel tolerance -- it is what the composition costs after each
#: kernel's error has been through four blocks. The corrupted case sits at 1.09e-01, well clear.
MAX_REL = 6e-2


def _randomise(model) -> None:
    """Replace the default init with noise.

    `primitives.Init.ZERO`/`GATING` zero-initialise the output projections, so a freshly built
    Pairformer's residual branches contribute EXACTLY zero and the whole stack is the identity --
    `out == in`, bitwise, for both implementations. A substitution test on that model compares
    nothing and passes: measured, before this existed, at L=64, 128 and 256, and a weight
    perturbed by 0.05 moved the output by 0.000e+00 because no weight reached the output at all.
    """
    import torch

    with torch.no_grad():
        for i, p in enumerate(model.parameters()):
            if p.dtype.is_floating_point:
                torch.manual_seed(1000 + i)
                p.copy_((torch.randn_like(p, dtype=torch.float32) * WEIGHT_SCALE).to(p.dtype))


def _stack(impl, config, weights=None):
    import torch

    from miniworld_engine.modules.pairformer import Pairformer
    torch.manual_seed(0)
    # bf16 weights, not fp32 weights under autocast. autocast casts activations, not parameters,
    # so a kernel that reads both ends up with `tl.dot(bf16, fp32)` and refuses to compile --
    # "Both operands must be same dtype". The bench harness builds its modules the same way
    # (bench.py:539, `base.to(device=DEVICE, dtype=torch.bfloat16)`), so this is the supported
    # arrangement rather than a workaround for the test.
    model = Pairformer(config, implementation=impl).cuda().to(torch.bfloat16).eval()
    _randomise(model)
    if weights is not None:
        model.load_state_dict(weights)
    return model


def _config():
    from miniworld_engine.modules.pairformer import PairformerConfig
    # p_drop=0: dropout is the one source of divergence that is not the kernels.
    return PairformerConfig(d_pair=D_PAIR, n_block=N_BLOCK, p_drop=0.0)


def _rel(a, b) -> float:
    d = (a.float() - b.float()).abs().max().item()
    scale = b.float().abs().max().item()
    return d / scale if scale else d


def test_the_stack_is_not_the_identity() -> None:
    """Everything below compares two stacks. If the stack does nothing, they agree perfectly and
    the comparison is vacuous -- which is what a default-initialised Pairformer does."""
    import torch

    from miniworld_engine.modules.exceptions import ImplementationType
    model = _stack(ImplementationType.PYTORCH, _config())
    torch.manual_seed(1)
    pair = torch.randn(B, L, L, D_PAIR, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(B, L, device="cuda", dtype=torch.bool)
    with torch.no_grad():
        out = model(pair.clone(), mask)
    assert not torch.equal(out, pair), (
        "the reference stack returned its input unchanged, so every comparison below is between "
        "two identity functions. The output projections are zero-initialised; `_randomise` exists "
        "to defeat that and is evidently not working.")


def test_miniworld_actually_dispatches_to_a_kernel() -> None:
    """Without this the comparison below could be the reference agreeing with itself."""
    import torch

    from miniworld_engine.modules.dispatch import (
        KernelBackend,
        resolve_transition,
        resolve_triangle_attention,
    )
    from miniworld_engine.modules.exceptions import ImplementationType
    dev = torch.device("cuda")
    chosen = {
        "transition": resolve_transition(ImplementationType.MINIWORLD, dev),
        "triangle_attention": resolve_triangle_attention(ImplementationType.MINIWORLD, dev),
    }
    assert all(b is not KernelBackend.PYTORCH for b in chosen.values()), (
        f"MINIWORLD resolved to the PyTorch reference for {chosen}; the substitution test below "
        f"would then be comparing the reference against itself.")


def test_the_stack_agrees_with_its_reference() -> None:
    import torch

    from miniworld_engine.modules.exceptions import ImplementationType
    config = _config()
    ref = _stack(ImplementationType.PYTORCH, config)
    # Same weights, not the same seed: construction order differs between backends, and a seed
    # would only make them equal by luck.
    mw = _stack(ImplementationType.MINIWORLD, config, weights=copy.deepcopy(ref.state_dict()))

    torch.manual_seed(1)
    pair = torch.randn(B, L, L, D_PAIR, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(B, L, device="cuda", dtype=torch.bool)

    with torch.no_grad():
        a = ref(pair.clone(), mask)
        b = mw(pair.clone(), mask)

    assert a.shape == b.shape, (a.shape, b.shape)
    rel = _rel(b, a)
    print(f"\nwhole-stack relative error, {N_BLOCK} blocks, bf16: {rel:.3e} (budget {MAX_REL:.0e})")
    assert not torch.equal(a, b), (
        "the two stacks produced bitwise-identical output, which means the MINIWORLD path ran the "
        "same code as the reference and this test proves nothing")
    assert rel <= MAX_REL, (
        f"whole-stack relative error {rel:.3e} exceeds the budget {MAX_REL:.0e} over {N_BLOCK} "
        f"blocks. Per-kernel tolerances can each be met while their composition is not.")


def test_a_corrupted_weight_breaks_the_comparison() -> None:
    """The teeth. If this passes, the test above cannot be trusted to fail when it should."""
    import torch

    from miniworld_engine.modules.exceptions import ImplementationType
    config = _config()
    ref = _stack(ImplementationType.PYTORCH, config)
    weights = copy.deepcopy(ref.state_dict())
    # Replace a weight outright rather than nudge it. Adding 0.05 to an LN scale moved the stack
    # by 1.51e-02 -- only 1.7x the honest 9.04e-03, so it says nothing about whether the budget
    # can see a WRONG KERNEL, which is what this test is for. A tensor of the same magnitude and
    # entirely different content is that failure's shape.
    key = next(k for k, v in weights.items()
               if v.dtype.is_floating_point and v.numel() > 1024)
    torch.manual_seed(7)
    weights[key] = (torch.randn_like(weights[key], dtype=torch.float32)
                    * WEIGHT_SCALE * 4).to(weights[key].dtype)
    mw = _stack(ImplementationType.MINIWORLD, config, weights=weights)

    torch.manual_seed(1)
    pair = torch.randn(B, L, L, D_PAIR, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(B, L, device="cuda", dtype=torch.bool)
    with torch.no_grad():
        a = ref(pair.clone(), mask)
        b = mw(pair.clone(), mask)
    rel = _rel(b, a)
    print(f"\ncorrupted-weight relative error: {rel:.3e} (must exceed {MAX_REL:.0e})")
    assert rel > MAX_REL, (
        f"perturbing {key} by 0.05 moved the stack output by only {rel:.3e}, which is inside the "
        f"budget. The budget is too loose to detect a real regression.")
