"""Attention cores skip the key blocks past each sample's last attended key (``kernels._key_len``).

A padded tail (FoldForge buckets token counts: mask True for the first n) is masked in every query row; after a row's live max its
exponentials are exact zeros, so a core that stops its key loop at the block holding the last attended key writes the same bits. Checked
here against the same call with the skip off (``key_len`` patched to None in the module that computes it), bitwise, at every position
(the padded query rows included), for paddings that drop 0, some and most key blocks, a fully masked row (kept whole: the references'
finite fill makes it a uniform average) and a mask with holes.

- AttentionPairBias (H100 attn_inf bf16 one row / the B > 1 batch, TF32 fp32; A100 the fused step and the composed path);
- the token DiT block (H100 attn_inf / attn_tf32_inf with the klen held next to the hoisted pair bias; A100 the sm_80 core);
- the bias-only block (H100 pvx / pvx32, A100 pv_gate);
- R26: a captured token stack replayed after ``refresh_static_inputs`` on a mask with another live count equals eager.
"""

from __future__ import annotations

import importlib
import typing

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules.attention import AttentionPairBias

pytestmark = [pytest.mark.gpu]

#: the modules that compute klen for a core (patched to None for the reference call)
_SITES = (
    "miniworld_engine.kernels.integrations.attention_pair_bias_b200",
    "miniworld_engine.kernels.integrations.attention_pair_bias_sm80",
    "miniworld_engine.kernels.conditioned_transition.cuda",
    "miniworld_engine.kernels.integrations.augattn_sm80",
    "miniworld_engine.kernels.integrations.bias_only_dit",
    "miniworld_engine.kernels.integrations.bias_only_dit_sm80",
)


def _card() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((9, 0), (8, 0))


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _no_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _SITES:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(mod, "key_len"):
            monkeypatch.setattr(mod, "key_len", lambda mask, length: None)


def _masks(B: int, L: int) -> list[torch.Tensor]:
    """Per-row masks: a long tail (most blocks dead), a short tail (no block dead), all attended, none attended, holes + a tail."""
    pats = []
    m = torch.ones(B, L, dtype=torch.bool, device="cuda")
    m[0, L - L // 8 :] = False  # the requirement inputs' L // 8
    if B > 1:
        m[1, 100:] = False  # most of the key blocks dead
    if B > 2:
        m[2, L - 5 :] = False  # inside the last block: none dead
    pats.append(m)
    g = torch.Generator(device="cuda").manual_seed(7)
    h = torch.rand(B, L, device="cuda", generator=g) > 0.3
    h[:, 3 * L // 4 :] = False
    pats.append(h)
    if B > 1:
        z = torch.ones(B, L, dtype=torch.bool, device="cuda")
        z[0] = False  # no attended key: every block kept
        z[1, L // 2 :] = False
        pats.append(z)
    return pats


def _apb(d_pair: int, dtype: str) -> typing.Any:
    m = AttentionPairBias(d_single=384, d_pair=d_pair, n_head=16).cuda().float()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    built: typing.Any = C.cast(m, dtype)
    return built.set_backend("cuda").eval()


def _autocast(dtype: str):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == "master")


@pytest.mark.skipif(not _card(), reason="H100 (sm_90) or A100 (sm_80)")
@pytest.mark.parametrize("dtype", ["master", "float32"])
@pytest.mark.parametrize(("B", "L"), [(1, 512), (3, 384), (2, 300)])
def test_apb_skip_bitwise(dtype: str, B: int, L: int, monkeypatch: pytest.MonkeyPatch) -> None:
    m = _apb(128, dtype)
    act = torch.float32 if dtype == "float32" else torch.bfloat16
    g = torch.Generator(device="cuda").manual_seed(1)
    inp = m.make_inputs(length=L, samples=B, depth=None, dtype=act, device="cuda", generator=g)
    for mask in _masks(B, L):
        with torch.no_grad(), _autocast(dtype):
            y = m(inp["single"], inp["pair"], mask).clone()
        with monkeypatch.context() as mp:
            _no_skip(mp)
            with torch.no_grad(), _autocast(dtype):
                ref = m(inp["single"], inp["pair"].clone(), mask.clone())
        torch.testing.assert_close(y, ref, rtol=0, atol=0, equal_nan=True)  # (the same NaN positions too)


def _token(key: str, dtype: str, d_cond: int = 384, d_pair: int = 128) -> typing.Any:
    from miniworld_engine.modules.token_dit import TOKEN_DIT_MODELS

    m = TOKEN_DIT_MODELS[key](d_single=768, d_cond=d_cond, d_pair=d_pair, n_head=16, n=2).cuda().float()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    built: typing.Any = C.cast(m, dtype)
    return built.set_backend("cuda").eval()


@pytest.mark.skipif(not _card(), reason="H100 (sm_90) or A100 (sm_80)")
@pytest.mark.parametrize("dtype", ["master", "float32"])
@pytest.mark.parametrize("key", ["protenix", "boltz2", "chai1", "miniworld_bias_only"])
@pytest.mark.parametrize("L", [512, 384])
def test_token_dit_skip_bitwise(key: str, dtype: str, L: int, monkeypatch: pytest.MonkeyPatch) -> None:
    m = _token(key, dtype)
    act = torch.float32 if dtype == "float32" else torch.bfloat16
    g = torch.Generator(device="cuda").manual_seed(1)
    inp = m.make_inputs(length=L, samples=5, depth=None, dtype=act, device="cuda", generator=g)
    for mask in _masks(1, L):
        kw = dict(inp, mask=mask)
        if m.unsupported(**kw) is not None:
            pytest.skip(f"not served: {m.unsupported(**kw)}")
        with torch.no_grad(), _autocast(dtype):
            y = m(**kw).clone()
        with monkeypatch.context() as mp:
            _no_skip(mp)
            # fresh pair / mask storage: the hoisted bias (and its klen) is keyed on them
            with torch.no_grad(), _autocast(dtype):
                ref = m(**dict(kw, pair=inp["pair"].clone(), mask=mask.clone()))
        torch.testing.assert_close(y, ref, rtol=0, atol=0, equal_nan=True)  # (the same NaN positions too)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) static input refresh")
def test_refresh_with_another_live_count() -> None:
    """FoldForge R26 with the key skip: one captured token stack step, folds whose masks drop different numbers of key blocks."""
    from miniworld_engine.capture import refresh_static_inputs, release_static_inputs, static_inputs, static_weights
    from miniworld_engine.modules import TokenDiTStack

    S, L = 5, 512
    torch.manual_seed(0)
    stack = TokenDiTStack("protenix", 2, 768, 384, 128, 16, 2, backend="cuda").float().cuda().eval().requires_grad_(False)
    with torch.no_grad():
        for p in stack.parameters():
            torch.nn.init.normal_(p, std=0.05) if p.dim() > 1 else torch.nn.init.normal_(p, mean=0.3, std=0.2)
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(S, 1, L, 768, device="cuda", generator=g).bfloat16()
    c = torch.empty(1, 1, L, 384, device="cuda", dtype=torch.bfloat16)
    z = torch.empty(1, L, L, 128, device="cuda", dtype=torch.bfloat16)
    mk = torch.ones(1, L, device="cuda", dtype=torch.bool)

    def fold(seed: int, n: int) -> None:
        gg = torch.Generator(device="cuda").manual_seed(seed)
        c.copy_(torch.randn(c.shape, device="cuda", generator=gg).bfloat16())
        z.copy_(torch.randn(z.shape, device="cuda", generator=gg).bfloat16())
        mk.fill_(True)
        mk[0, n:] = False

    def step() -> torch.Tensor:
        return stack(x, c.expand(S, 1, L, 384), z, mk)

    def eager() -> torch.Tensor:
        with torch.no_grad():
            return step().clone()

    fold(1, 480)
    graph, side = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), static_weights(), static_inputs() as tables:
        with torch.cuda.stream(side):
            for _ in range(2):
                step()
        torch.cuda.current_stream().wait_stream(side)
        with torch.cuda.graph(graph):
            out = step()
    try:
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager())
        for seed, n in ((2, 130), (3, 512), (4, 300)):  # fewer key blocks, every block, some again
            fold(seed, n)
            ref = eager()
            refresh_static_inputs(stack)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, ref), f"fold {seed} (n {n}): refresh + replay != eager"
    finally:
        release_static_inputs(tables=tables)
