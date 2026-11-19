"""Chai-1's token DiT (``IdentityAdaptiveLayerNorm``: no ``ln_cond``, a ``(1 + W_s c)`` scale) on the sm_80-family ``cuda`` path (A100, RTX 5090).

rc/3.0.4's fp32 IEEE change put the AdaLN of the AugmentedAttention block on ``adaln_sm80`` (``augattn_sm80._adaln``); ``adaln_sm80.serves`` and
``conditioned_transition_sm80.serves`` read ``module.ln_cond`` and raised ``AttributeError: 'IdentityAdaptiveLayerNorm' object has no attribute
'ln_cond'`` for every fp32 call that reached them (the 5090 judge re-measure: 44 ``error`` tokens, rows 110 / 111 / 268). The identity AdaLN is
not the sigmoid form of those kernels: both gates now refuse it (the module serves it) instead of reading a leaf it does not have.
"""

from __future__ import annotations

import typing

import pytest
import torch

from miniworld_engine.kernels.integrations import adaln_sm80
from miniworld_engine.kernels.integrations import conditioned_transition_sm80 as ct_sm80
from miniworld_engine.measure import config as C
from miniworld_engine.modules.token_dit import TokenDiTChai1

pytestmark = [pytest.mark.gpu]


def _sm80_family() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((8, 0), (12, 0))


def _fp32(seed: int = 0) -> typing.Any:
    m = TokenDiTChai1(d_single=768, d_cond=384, d_pair=256, n_head=16, n=2, adaln_eps=1e-5).cuda().float()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(seed))
    return m.eval()


def _inputs(m: typing.Any, length: int, samples: int = 2) -> dict:
    g = torch.Generator(device="cuda").manual_seed(1)
    return m.make_inputs(length=length, samples=samples, depth=None, dtype=torch.float32, device="cuda", generator=g, stream="token_single")


@pytest.mark.skipif(not _sm80_family(), reason="sm_80-family (A100 / RTX 5090) cuda path")
def test_identity_adaln_gates_refuse_without_reading_ln_cond() -> None:
    m = _fp32()
    x = torch.randn(2, 1, 128, 768, device="cuda")
    c = torch.randn(2, 1, 128, 384, device="cuda")
    assert not hasattr(m.attention.ada_ln_in, "ln_cond")
    assert adaln_sm80.serves(m.attention.ada_ln_in, x, c, torch.float32, False) is False
    assert ct_sm80.serves(m.transition, x, c, torch.float32, False) is False


@pytest.mark.skipif(not _sm80_family(), reason="sm_80-family (A100 / RTX 5090) cuda path")
@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize("length", [128, 256])
def test_chai1_token_dit_fp32_ieee_cuda_vs_fp64(length: int, grad: bool) -> None:
    """fp32, TF32 off (IEEE): the cuda route serves the call (no AttributeError, no refusal) and stays within the pytorch error of the fp64 reference."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    m = _fp32()
    inp = _inputs(m, length)
    ref64 = _fp32()
    ref64.load_state_dict(m.state_dict())
    ref64 = ref64.double()
    inp64 = {k: (v.double() if isinstance(v, torch.Tensor) and v.is_floating_point() else v) for k, v in inp.items()}
    with torch.set_grad_enabled(grad):
        ref = ref64(**inp64)
        m.set_backend("cuda")
        assert m._supports_cuda(inp["single"], inp["cond"], inp["pair"], inp["mask"]) is None
        y = m(**inp)
        m.set_backend("pytorch")
        yp = m(**inp)
    err = float((y.detach().double() - ref.detach()).abs().max())
    err_pt = float((yp.detach().double() - ref.detach()).abs().max())
    assert err <= max(err_pt, 8e-3)
