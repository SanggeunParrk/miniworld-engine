"""RTX 5090 (sm_120a) Transition, bf16 inference at D >= 256: the LayerNorm folded into the dual GEMM (statistics kernel + gamma-folded weights + the epilogue's
rstd (acc - mean cs) + bias) stays inside the accuracy standard (error against an fp64 evaluation no larger than the unfused ln_fwd + dual chain's, up to rounding),
also for an input with a large mean, and is the path that serves from ``SM120_FUSED_LN_MIN_ROWS`` rows on."""

import copy

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules import Transition

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0), reason="needs an RTX 5090 (sm_120)"),
]


def _module(d: int, n: int) -> Transition:
    module = Transition(d_hidden=d, n=n, master_weights=True).cuda().float().train()
    C.randomize(module, torch.Generator(device="cuda").manual_seed(0))
    module.set_master_weights()
    return module


@pytest.mark.parametrize(
    ("d", "n", "rows", "offset"), [(256, 2, 40000, 0.0), (384, 2, 33000, 0.0), (512, 4, 40001, 0.0), (256, 4, 65536, 12.0)]
)
def test_fused_layernorm_within_the_unfused_error(monkeypatch: pytest.MonkeyPatch, d: int, n: int, rows: int, offset: float) -> None:
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm80 as fw

    module = _module(d, n)
    kern = copy.deepcopy(module).set_backend("cuda").eval()
    ref = copy.deepcopy(module).double().eval()
    x = (torch.randn(1, rows, d, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1)) * 2 + offset).to(torch.bfloat16)
    with torch.no_grad():
        want = ref(x.double())
        assert fw._fused_ln(rows, d, 2, x)
        fused = kern(x)
        monkeypatch.setattr(fw, "SM120_FUSED_LN", False)
        unfused = kern(x)
    err_f = (fused.double() - want).abs()
    err_u = (unfused.double() - want).abs()
    assert err_f.pow(2).mean().sqrt() <= err_u.pow(2).mean().sqrt() * 1.05 + 1e-6
    assert err_f.max() <= err_u.max() * 1.25 + 1e-3


def test_few_rows_and_training_keep_the_unfused_chain() -> None:
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm80 as fw

    x = torch.zeros(1, 16384, 256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        assert not fw._fused_ln(16384, 256, 2, x)
        assert fw._fused_ln(65536, 256, 2, x)
        assert not fw._fused_ln(65536, 128, 2, x)
    assert not fw._fused_ln(65536, 256, 2, x)  # grad enabled (training): xn is saved for the backward
