"""3.0.1 inference glue (H100): ``AttentionPairBias`` at L % 128 != 0 reads the pair as it is, and the Chai-1 DiT's bf16 entry form of the
fp32 conditioning / pair is made once per tensor.

APB: the call runs the core at the next multiple of 128 (pad keys masked, pad query rows dropped); the pair pass writes the real [L, L]
block of the padded bias planes (``apb.pair_bias_padded``) instead of ``F.pad``-ing the whole pair. Same arithmetic for every real element:
the output is bitwise the padded-pair flow's (the 3.0.0 flow, run here through ``_reads_ragged`` switched off), for B = 1 and the batched
H100 path, with and without a key mask / extra_pair_bias, every d_pair the pair pass takes.

Entry cache: ``MasterEntryCast`` rounds fp32 ``cond`` / ``pair`` to bf16 at the entry; a sampler's steps pass the same tensors, so the bf16 form
is cached per tensor (``kernels._capture.hoisted_table``): a hit while the tensor is unchanged, a new form when it is written in place, no cache
for a tensor the autograd records.
"""

from __future__ import annotations

import typing

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules import dit_variant_cuda as dvc
from miniworld_engine.modules.attention import AttentionPairBias
from miniworld_engine.modules.token_dit import TokenDiTChai1

pytestmark = [pytest.mark.gpu]


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


needs_h100 = pytest.mark.skipif(not _h100(), reason="H100 (sm_90)")


def _built(module: torch.nn.Module, dtype: str = "master") -> typing.Any:
    module = module.cuda().float()
    C.randomize(module, torch.Generator(device="cuda").manual_seed(0))
    built: typing.Any = C.cast(module, dtype)
    return built.eval().set_backend("cuda")


def _run(m, inp: dict, old: bool) -> torch.Tensor:
    """One call; ``old``: the 3.0.0 flow (the pair padded to a multiple of 128 by the module)."""
    keep = AttentionPairBias.__dict__["_reads_ragged"]
    if old:
        AttentionPairBias._reads_ragged = staticmethod(lambda s: False)  # pyright: ignore[reportAttributeAccessIssue]
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            assert m.unsupported(**inp) is None
            return m._forward_cuda(**inp)
    finally:
        AttentionPairBias._reads_ragged = keep  # pyright: ignore[reportAttributeAccessIssue]


@needs_h100
@pytest.mark.parametrize("length", [130, 200, 1000, 1030])
@pytest.mark.parametrize("d_pair,heads", [(128, 16), (256, 16), (384, 8), (384, 16)])
@pytest.mark.parametrize("masked", [False, True])
def test_apb_ragged_is_bitwise_the_padded_flow(length: int, d_pair: int, heads: int, masked: bool) -> None:
    m = _built(AttentionPairBias(d_single=384, d_pair=d_pair, n_head=heads))
    g = torch.Generator(device="cuda").manual_seed(1)
    inp = m.make_inputs(length=length, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=g)
    inp["mask"] = (torch.arange(length, device="cuda") < length - 7)[None] if masked else None
    new, old = _run(m, inp, False), _run(m, inp, True)
    assert new.shape == inp["single"].shape and torch.isfinite(new.float()).all()
    assert torch.equal(new, old)


@needs_h100
@pytest.mark.parametrize("length", [200, 1000])
@pytest.mark.parametrize("form", ["LL", "HLL"])
def test_apb_ragged_extra_bias(length: int, form: str) -> None:
    m = _built(AttentionPairBias(d_single=384, d_pair=384, n_head=8))
    g = torch.Generator(device="cuda").manual_seed(1)
    inp = m.make_inputs(length=length, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=g, stream="token_single_extra_bias")
    plane = inp["extra_pair_bias"][0, 0].contiguous()
    inp["extra_pair_bias"] = plane if form == "LL" else plane[None].repeat(8, 1, 1) * torch.linspace(0.5, 2, 8, device="cuda")[:, None, None]
    assert torch.equal(_run(m, inp, False), _run(m, inp, True))


@needs_h100
@pytest.mark.parametrize("length", [130, 1000])
def test_apb_ragged_batched(length: int) -> None:
    """B > 1: the batch rows in one launch per stage (a different key mask per row)."""
    m = _built(AttentionPairBias(d_single=384, d_pair=128, n_head=16))
    g = torch.Generator(device="cuda").manual_seed(1)
    inp = m.make_inputs(length=length, samples=3, depth=None, dtype=torch.bfloat16, device="cuda", generator=g)
    inp["mask"] = torch.arange(length, device="cuda")[None] < (length - 3 - 5 * torch.arange(3, device="cuda"))[:, None]
    new, old = _run(m, inp, False), _run(m, inp, True)
    assert torch.equal(new, old)
    rows = torch.cat([_run(m, {**inp, "single": inp["single"][b : b + 1], "pair": inp["pair"][b : b + 1], "mask": inp["mask"][b : b + 1]}, False) for b in range(3)])
    assert torch.equal(new, rows)


@needs_h100
def test_apb_ragged_reads_the_pair_unpadded() -> None:
    """No [Lp, Lp, d_pair] copy: the peak allocation of the call stays far below one padded pair."""
    length, d_pair = 1000, 384
    m = _built(AttentionPairBias(d_single=384, d_pair=d_pair, n_head=8))
    g = torch.Generator(device="cuda").manual_seed(1)
    inp = m.make_inputs(length=length, samples=1, depth=None, dtype=torch.bfloat16, device="cuda", generator=g)
    _run(m, inp, False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    _run(m, inp, False)
    torch.cuda.synchronize()
    padded_pair = 1024 * 1024 * d_pair * 2
    assert torch.cuda.max_memory_allocated() - base < padded_pair // 2


# ------------------------------------------------------------------------------------------------------------ the Chai-1 entry form
def _chai(length: int = 256):
    m = _built(TokenDiTChai1(d_single=768, d_cond=384, d_pair=256, n_head=16, n=2, adaln_eps=1e-5))
    g = torch.Generator(device="cuda").manual_seed(1)
    inp = m.make_inputs(length=length, samples=2, depth=None, dtype=torch.bfloat16, device="cuda", generator=g, stream="token_single_fp32_cond")
    assert inp["pair"].dtype == torch.float32 and inp["cond"].dtype == torch.float32
    return m, inp


def _call(m, inp: dict) -> torch.Tensor:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return m._forward_cuda(**inp)


@needs_h100
def test_entry_form_is_made_once_per_tensor() -> None:
    m, inp = _chai()
    dvc._ENTRY_BF16.clear()
    first = _call(m, inp)
    n = len(dvc._ENTRY_BF16)
    assert n >= 1
    forms = {id(e[0]) for e in dvc._ENTRY_BF16.values()}
    second = _call(m, inp)
    assert len(dvc._ENTRY_BF16) == n and {id(e[0]) for e in dvc._ENTRY_BF16.values()} == forms  # served, not remade
    assert torch.equal(first, second)
    # the cached form is what the entry cast makes: the same output as the cast at every call
    keep = dvc._round_bf16
    dvc._round_bf16 = lambda t: t.to(torch.bfloat16)  # pyright: ignore[reportAttributeAccessIssue]
    try:
        assert torch.equal(_call(m, inp), first)
    finally:
        dvc._round_bf16 = keep  # pyright: ignore[reportAttributeAccessIssue]


@needs_h100
def test_entry_form_follows_an_in_place_write() -> None:
    m, inp = _chai()
    dvc._ENTRY_BF16.clear()
    before = _call(m, inp)
    inp["pair"].mul_(1.5).add_(0.25)  # a new version of the same storage: the cached form is stale
    after = _call(m, inp)
    fresh = {**inp, "pair": inp["pair"].clone(), "cond": inp["cond"].clone()}
    assert torch.equal(after, _call(m, fresh))
    assert not torch.equal(before, after)


@needs_h100
def test_entry_form_not_cached_under_autograd() -> None:
    t = torch.randn(4, 4, device="cuda", requires_grad=True)
    dvc._ENTRY_BF16.clear()
    out = dvc._round_bf16(t)
    assert out.requires_grad and out.dtype is torch.bfloat16 and not dvc._ENTRY_BF16
    with torch.no_grad():
        a, b = dvc._round_bf16(t.detach()), dvc._round_bf16(t.detach())
    assert a.dtype is torch.bfloat16 and torch.equal(a, t.detach().to(torch.bfloat16)) and b is a


def test_entry_cache_is_a_registered_weight_cache() -> None:
    from miniworld_engine.kernels import _capture

    assert _capture.weight_caches()["dit_variant_cuda.entry_bf16"] is dvc._ENTRY_BF16
