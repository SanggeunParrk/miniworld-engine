"""``miniworld_engine.capture.weights_changed`` after a fused optimizer step (the documented contract).

``Adam(fused=True)`` writes the parameters without bumping their ``_version``; the engine's weight packs are keyed on
``(data_ptr, _version)``, so the next eager forward of an H100 token DiT block would read the packs of the old weights. After
``weights_changed(optimizer)`` the block's output is the one of the new weights (== after ``clear_weight_caches``, bitwise).
"""

from __future__ import annotations

import typing

import pytest
import torch

from miniworld_engine import capture
from miniworld_engine.measure import config as C
from miniworld_engine.modules.token_dit import TokenDiTProtenix

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) token DiT")
@pytest.mark.parametrize("target", ["optimizer", "module", "parameters"])
def test_fused_adam_step_is_seen_after_weights_changed(target) -> None:
    module = TokenDiTProtenix(d_single=768, d_cond=384, d_pair=128, n_head=16, n=2).cuda().float()
    C.randomize(module, torch.Generator(device="cuda").manual_seed(0))
    m: typing.Any = C.cast(module, "master")
    m.set_backend("cuda")
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-2, fused=True)
    inp = m.make_inputs(length=128, samples=5, depth=None, dtype=torch.bfloat16, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))

    def infer():
        m.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return m(**inp).clone()

    before = infer()
    m.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m(**inp).float().square().mean().backward()
    versions = [p._version for p in params]
    opt.step()
    # the premise of the contract; if a torch release starts bumping here, the call below is merely redundant
    silent = [p._version for p in params] == versions
    n = capture.weights_changed({"optimizer": opt, "module": m, "parameters": params}[target])
    assert n == len(params)
    assert all(p._version > v for p, v in zip(params, versions, strict=True))
    after = infer()
    assert not torch.equal(after, before)  # the step changed the weights and the block read them
    capture.clear_weight_caches(m)
    fresh = infer()
    assert torch.equal(after, fresh), float((after.float() - fresh.float()).abs().max())
    if silent:
        # without the call the eager forward is served the old packs: what the contract is about
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m.train()
            opt.zero_grad()
            m(**inp).float().square().mean().backward()
        opt.step()
        stale = infer()
        assert torch.equal(stale, after)  # the second step is not seen ...
        capture.weights_changed(opt)
        assert not torch.equal(infer(), after)  # ... until the engine is told
    capture.clear_weight_caches(m)
