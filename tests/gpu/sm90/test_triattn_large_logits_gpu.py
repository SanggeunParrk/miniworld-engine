"""H100 bf16 triangle attention on large logits: no overflow, whatever the spread of a row's logits.

The core used to fix each row's softmax offset at the maximum of its first 32 keys, so a later logit more than ~126 log2 units above it
became inf and the output NaN. Trained weights do that (projection std ~0.5 gives logits of 100 to 200). The offset is now the running
maximum, raised lazily. The bar is the repository's: the output is no worse than 1.5x the error of the PyTorch module run in bf16
autocast, against the fp32 PyTorch module (same bf16-rounded weights and inputs), plus an absolute floor. A fixed tolerance would
not do: at logits of several hundred the bf16 rounding of q and k alone moves a logit by a nat or more.
"""

import pytest
import torch

from miniworld_engine.modules import TriangleAttention
from miniworld_engine.modules.exceptions import ImplementationType as Impl

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="Hopper (sm_90) required",
    ),
]


def _modules(weight_std, starting, transposed_bias):
    torch.manual_seed(0)
    kwargs = dict(starting=starting, transposed_bias=transposed_bias, p_drop=0.0)
    engine = TriangleAttention(128, 4, implementation=Impl.MINIWORLD, **kwargs)
    with torch.no_grad():
        for name, p in engine.named_parameters():
            if p.ndim > 1:
                p.copy_(torch.randn_like(p) * weight_std)
            else:
                p.copy_(torch.randn_like(p) * 0.05 + (1.0 if name == "ln_pair.weight" else 0.0))
    reference = TriangleAttention(128, 4, implementation=Impl.PYTORCH, **kwargs)
    reference.load_state_dict(engine.state_dict())
    return _bf16(engine), reference.cuda().float().eval()


def _bf16(module):
    module = module.cuda().bfloat16()
    for name, p in module.named_parameters():
        if "ln_" in name:
            p.data = p.data.float()
    return module.eval()


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


@pytest.mark.parametrize("weight_std", [0.35, 0.5, 0.8, 1.2])
@pytest.mark.parametrize("length", [128, 256])
@pytest.mark.parametrize(("starting", "transposed_bias"), [(True, True), (False, True), (False, False)])
@pytest.mark.parametrize("masked", [False, True])
def test_large_logits_stay_finite_and_accurate(weight_std, length, starting, transposed_bias, masked):
    engine, reference = _modules(weight_std, starting, transposed_bias)
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(2, length, length, 128, device="cuda", generator=g).bfloat16()
    mask = None
    if masked:
        mask = torch.ones(2, length, dtype=torch.bool, device="cuda")
        mask[0, -5:] = False
        mask[1, : length // 2] = False
    with torch.no_grad():
        actual = engine(x, mask)
        expected = reference(x.float(), mask)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            baseline = reference(x, mask)
    assert torch.isfinite(actual).all(), float((~torch.isfinite(actual)).float().mean())
    # the rows of masked queries are not part of the answer, but their keys are
    keep = slice(None) if mask is None else (mask[:, :, None] & mask[:, None, :])
    pick = (lambda t: t) if mask is None else (lambda t: t[keep])
    err, base = _rel(pick(actual), pick(expected)), _rel(pick(baseline.float()), pick(expected))
    assert err <= 1.5 * base + 5e-3, (err, base)


def test_first_tile_fully_masked_then_a_far_larger_logit():
    """The offset starts at -inf: a row whose first 128 keys are masked takes it from a later tile."""
    engine, reference = _modules(0.8, True, True)
    g = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(1, 256, 256, 128, device="cuda", generator=g).bfloat16()
    mask = torch.ones(1, 256, dtype=torch.bool, device="cuda")
    mask[0, :128] = False
    with torch.no_grad():
        actual, expected = engine(x, mask), reference(x.float(), mask)
    assert torch.isfinite(actual[:, :, :]).all()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        baseline = reference(x, mask)
    err, base = _rel(actual[:, 128:, :], expected[:, 128:, :]), _rel(baseline.float()[:, 128:, :], expected[:, 128:, :])
    assert err <= 1.5 * base + 5e-3, (err, base)
