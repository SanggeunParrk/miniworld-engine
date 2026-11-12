"""FoldForge R3 and R7 on the ``cuda`` backend (H100 and A100).

R3: ``AttentionPairBias`` takes an additive ``extra_pair_bias`` at inference (OpenDDE's structural-token refiner: one [L, L] bias on every
head), in each accepted form ([L, L], [H, L, L], [B, H, L, L], expanded views), and refuses it with a gradient (the training Functions return
no gradient for it) instead of dropping it.

R7: the Chai-1 DiT blocks serve FoldForge's mixed layout under the master weights: bf16 ``single`` with fp32 ``cond`` / ``pair`` (token and
atom block) and the fp32 sampler glue under bf16 autocast (atom block), the activations rounded to bf16 at the entry.
"""

from __future__ import annotations

import typing

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules.atom_dit import AtomDiTChai1
from miniworld_engine.modules.attention import AttentionPairBias
from miniworld_engine.modules.token_dit import TokenDiTChai1

pytestmark = [pytest.mark.gpu]


def _card() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((9, 0), (8, 0))


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max() / b.float().abs().max())


def _built(module: torch.nn.Module, dtype: str) -> typing.Any:
    module = module.cuda().float()
    C.randomize(module, torch.Generator(device="cuda").manual_seed(0))
    return C.cast(module, dtype).eval()


def _both(m, inp: dict, autocast: bool):
    """(cuda, pytorch) outputs of one call."""
    out = []
    for backend in ("cuda", "pytorch"):
        m.set_backend(backend)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out.append(m(**inp))
    return out


@pytest.mark.skipif(not _card(), reason="H100 (sm_90) or A100 (sm_80)")
@pytest.mark.parametrize("dtype", ["master", "float32"])
@pytest.mark.parametrize("form", ["LL", "HLL", "BHLL_expanded"])
def test_attention_pair_bias_extra_bias(dtype: str, form: str) -> None:
    m = _built(AttentionPairBias(d_single=384, d_pair=384, n_head=8), dtype)
    g = torch.Generator(device="cuda").manual_seed(1)
    act = torch.bfloat16 if dtype == "master" else torch.float32
    inp = m.make_inputs(length=256, samples=1, depth=None, dtype=act, device="cuda", generator=g, stream="token_single_extra_bias")
    plane = inp["extra_pair_bias"][0, 0].contiguous()
    inp["extra_pair_bias"] = {
        "LL": plane,
        "HLL": plane[None].repeat(8, 1, 1) + torch.arange(8, device="cuda", dtype=act)[:, None, None] * 0.1,  # a different plane per head
        "BHLL_expanded": plane[None, None].expand(1, 8, 256, 256),
    }[form]
    with torch.no_grad():
        assert m._supports_cuda(inp["single"], inp["pair"], inp["mask"], inp["extra_pair_bias"]) is None
    y, ref = _both(m, inp, autocast=dtype == "master")
    assert _rel(y, ref) < (2e-2 if dtype == "master" else 5e-3)
    # the bias is not dropped: without it the output moves
    inp_none = {**inp, "extra_pair_bias": None}
    y0, _ = _both(m, inp_none, autocast=dtype == "master")
    assert _rel(y0, y) > 1e-2


@pytest.mark.skipif(not _card(), reason="H100 (sm_90) or A100 (sm_80)")
def test_attention_pair_bias_extra_bias_refused_with_gradient() -> None:
    m = _built(AttentionPairBias(d_single=384, d_pair=384, n_head=8), "master").train(True)
    g = torch.Generator(device="cuda").manual_seed(1)
    inp = m.make_inputs(
        length=128, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=g, stream="token_single_extra_bias"
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        why = m._supports_cuda(inp["single"], inp["pair"], inp["mask"], inp["extra_pair_bias"])
    assert isinstance(why, str) and "extra_pair_bias" in why


@pytest.mark.skipif(not _card(), reason="H100 (sm_90) or A100 (sm_80)")
@pytest.mark.parametrize("stream", ["token_single_fp32_cond"])
def test_chai1_token_dit_mixed_layout(stream: str) -> None:
    m = _built(TokenDiTChai1(d_single=768, d_cond=384, d_pair=256, n_head=16, n=2), "master")
    g = torch.Generator(device="cuda").manual_seed(1)
    inp = m.make_inputs(length=256, samples=5, depth=None, dtype=torch.bfloat16, device="cuda", generator=g, stream=stream)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert m._supports_cuda(inp["single"], inp["cond"], inp["pair"], inp["mask"]) is None
    y, ref = _both(m, inp, autocast=True)
    assert y.dtype == inp["single"].dtype
    assert _rel(y, ref) < 2e-2


@pytest.mark.skipif(not _card(), reason="H100 (sm_90) or A100 (sm_80)")
@pytest.mark.parametrize("stream", ["atom_single_fp32_cond", "atom_single_fp32"])
@pytest.mark.parametrize("key_window", ["pad", "slide"])
def test_chai1_atom_dit_mixed_layout(stream: str, key_window: str) -> None:
    m = _built(AtomDiTChai1(d_single=128, d_cond=128, d_pair=16, n_head=4, n=2, key_window=key_window), "master")
    g = torch.Generator(device="cuda").manual_seed(1)
    inp = m.make_inputs(length=1024, samples=5, depth=None, dtype=torch.bfloat16, device="cuda", generator=g, stream=stream)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        kw = {k: v for k, v in inp.items() if k not in ("single", "cond", "pair", "mask")}
        assert m._supports_cuda(inp["single"], inp["cond"], inp["pair"], inp["mask"], **kw) is None
    y, ref = _both(m, inp, autocast=True)
    assert y.dtype == inp["single"].dtype
    assert _rel(y, ref) < 3e-2
