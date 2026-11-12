"""A100 (sm_80) Transition past one call's rows (FoldForge 5I28: the template pair stack at 4 x 2048^2 rows, the protenix confidence block's
pair transition at 5 x 1536^2). The one-kernel forward's per-CTA work list holds 253 tiles per SM and the wide chain's GEMM grid 65 535
tiles of 128 rows; the cuda backend raised ``fwd: too many work items per CTA`` / ``invalid configuration argument`` there. Inference now
runs one call per contiguous row block (``kernels.transition.cuda.transition_sm80``); a training call past the limit is refused with a
reason (the backward's work lists would drop rows)."""

from __future__ import annotations

import typing

import pytest
import torch

from miniworld_engine.kernels.transition.cuda import fused_fwd_sm80
from miniworld_engine.measure import config as C
from miniworld_engine.modules.primitives import BackendUnsupported
from miniworld_engine.modules.transition import Transition

pytestmark = [pytest.mark.gpu, pytest.mark.a100, pytest.mark.slow]


def _a100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)


def _master(module: torch.nn.Module) -> typing.Any:
    module = module.cuda().float()
    C.randomize(module, torch.Generator(device="cuda").manual_seed(0))
    return C.cast(module, "master").eval()


@pytest.mark.skipif(not _a100(), reason="A100 (sm_80) Transition")
@pytest.mark.parametrize(("d_hidden", "n"), [(128, 4), (256, 2)])  # fused_sm80 / the one-kernel forward; the wide chain
def test_transition_sm80_inference_past_one_call(d_hidden: int, n: int) -> None:
    m = _master(Transition(d_hidden=d_hidden, n=n))
    x = torch.randn(2, 2048, 2048, d_hidden, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1)).bfloat16()
    assert x.numel() // d_hidden > fused_fwd_sm80.max_rows(x.device, x.dtype)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        m.set_backend("cuda")
        assert m.unsupported(x) is None
        y = m(x)
        m.set_backend("pytorch")
        ref = m(x)
    assert y.shape == ref.shape and bool(torch.isfinite(y).all())
    assert float((y.float() - ref.float()).abs().max() / ref.float().abs().max()) < 2e-2


@pytest.mark.skipif(not _a100(), reason="A100 (sm_80) Transition")
def test_transition_sm80_training_past_one_call_refused() -> None:
    m = _master(Transition(d_hidden=64, n=2)).train()
    x = torch.randn(4, 2048, 2048, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    m.set_backend("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        why = m.unsupported(x)
        assert isinstance(why, str) and "rows with a gradient" in why
        with pytest.raises(BackendUnsupported):
            m(x)
