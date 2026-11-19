"""RTX 5090 (sm_120a) Transition: (1) bf16 inference past ``SM120_INFER_MAX_ROWS`` runs as row blocks written into one output (the 64-row tile configs
stay usable: the L 2048 pair call ran the 128-row ones) with the output bits of the single call; (2) the training forward keeps ``xn`` (no rebuild) above
``SM120_XN_RECOMPUTE_MAX_ROWS`` rows and gives the same gradients as the rebuilding path."""

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


def test_inference_row_blocks_equal_the_single_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm80 as fw

    kern = copy.deepcopy(_module(256, 2)).set_backend("cuda").eval()
    x = torch.randn(1, 20480, 256, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1)).to(torch.bfloat16)
    with torch.no_grad():
        whole = kern(x)
        monkeypatch.setattr(fw, "SM120_INFER_MAX_ROWS", 8192)
        blocks = kern(x)
    assert torch.equal(whole, blocks)


def test_xn_is_saved_above_the_recompute_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from miniworld_engine.kernels.transition.cuda import fused_wide_sm80 as fw

    module = _module(128, 2)
    kern = copy.deepcopy(module).set_backend("cuda")
    x = torch.randn(1, 128, 128, 128, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1)).to(torch.bfloat16)
    up = torch.randn(x.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(9)).to(torch.bfloat16)
    assert fw._recompute_xn(x.reshape(-1, 128), True, 256)
    grads = []
    for limit in (fw.SM120_XN_RECOMPUTE_MAX_ROWS, 1000):
        monkeypatch.setattr(fw, "SM120_XN_RECOMPUTE_MAX_ROWS", limit)
        assert fw._recompute_xn(x.reshape(-1, 128), True, 256) == (limit > 1000)
        z = x.detach().clone().requires_grad_(True)
        kern.zero_grad(set_to_none=True)
        kern(z).backward(up)
        grads.append([z.grad] + [p.grad for p in kern.parameters() if p.grad is not None])
    for a, b in zip(*grads, strict=True):
        assert torch.equal(a, b)
