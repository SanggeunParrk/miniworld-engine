"""A100: the token DiT's pair bias is hoisted per (pair, weights, key mask) at inference (``augattn_sm80._TOKEN_BIAS``).

A sampler calls every block with the SAME pair at every step; the sm_80 attention used to make the pair bias (pair LayerNorm + head projection, 47 us per
block call at 128 tokens, 13 % of an ESMFold2 forward) again at each call. The hoist is the same kernel's output, so every call equals the call with the hoist
off (``FOLDJET_ENGINE_TOKEN_BIAS_HOIST=0``) BITWISE; a new pair tensor, an in-place change of the pair or of a bias weight, and another key mask are not served the old table.
"""

from __future__ import annotations

import typing

import pytest
import torch

from foldjet_engine.kernels.integrations import augattn_sm80
from foldjet_engine.measure import config as C

pytestmark = [pytest.mark.gpu]


def _a100() -> bool:  # the sm_80 route: A100 and RTX 5090 (the same sources built for sm_120a)
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((8, 0), (12, 0))


def _token(dtype: str) -> typing.Any:
    from foldjet_engine.modules.token_dit import TOKEN_DIT_MODELS

    m = TOKEN_DIT_MODELS["protenix"](d_single=768, d_cond=384, d_pair=128, n_head=16, n=2).cuda().float()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    built: typing.Any = C.cast(m, dtype)
    return built.set_backend("cuda").eval()


def _run(m, kw, hoist: bool, monkeypatch: pytest.MonkeyPatch) -> torch.Tensor:
    monkeypatch.setenv("FOLDJET_ENGINE_TOKEN_BIAS_HOIST", "1" if hoist else "0")
    with torch.no_grad():
        return m(**kw).clone()


@pytest.mark.skipif(not _a100(), reason="A100 (sm_80)")
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("L", [128, 300])
def test_hoist_is_bitwise_and_keyed_on_inputs(dtype: str, L: int, monkeypatch: pytest.MonkeyPatch) -> None:
    m = _token(dtype)
    act = torch.float32 if dtype == "float32" else torch.bfloat16
    inp = m.make_inputs(length=L, samples=5, depth=None, dtype=act, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
    mask = torch.ones(1, L, dtype=torch.bool, device="cuda")
    mask[0, L - L // 8 :] = False
    kw = dict(inp, mask=mask)
    if m.unsupported(**kw) is not None:
        pytest.skip(f"not served: {m.unsupported(**kw)}")
    augattn_sm80._TOKEN_BIAS.clear()
    ref = _run(m, dict(kw), False, monkeypatch)
    assert not augattn_sm80._TOKEN_BIAS
    for _ in range(3):  # the first call makes the table, the next ones read it
        torch.testing.assert_close(_run(m, dict(kw), True, monkeypatch), ref, rtol=0, atol=0)
    assert augattn_sm80._TOKEN_BIAS  # the table was made (and read by the later calls)
    # an in-place change of the pair bumps its version: the table is remade
    kw2 = dict(kw, pair=inp["pair"].clone())
    kw2["pair"].mul_(1.5)
    ref2 = _run(m, dict(kw2), False, monkeypatch)
    out1 = _run(m, dict(kw2), True, monkeypatch)
    kw2["pair"].mul_(0.5)  # in place again, back to a different pair
    ref3 = _run(m, dict(kw2), False, monkeypatch)
    out3 = _run(m, dict(kw2), True, monkeypatch)
    torch.testing.assert_close(out1, ref2, rtol=0, atol=0)
    torch.testing.assert_close(out3, ref3, rtol=0, atol=0)
    assert not torch.equal(out1, out3)
    # another key mask on the same pair
    mask2 = mask.clone()
    mask2[0, : L // 4] = False
    ref4 = _run(m, dict(kw, mask=mask2), False, monkeypatch)
    torch.testing.assert_close(_run(m, dict(kw, mask=mask2), True, monkeypatch), ref4, rtol=0, atol=0)
