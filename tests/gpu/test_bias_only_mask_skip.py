"""The bias-only token DiT's P v core skips the key blocks past the mask's last attended key (``kernels._key_len``; H100 pvx / pvx32,
A100 pv_gate / pv_gate_tf32): P = softmax over the keys of the masked pair bias is exactly 0 there in every row that attends a key, so
the blocks add nothing and the output is the same bits. A row attending no key keeps every block (a uniform P).

- H100: bitwise against the same call with the core's mask off (``kernel_mask`` patched to None in the integration), every position,
  bf16 and fp32, paddings that drop 0 / some / most key blocks, a mask with holes, the per-call P path (a hoisted bias, ``bias=``);
- both cards: against the PyTorch reference at every position;
- R26 (H100): a captured bias-only stack replayed after ``refresh_static_inputs`` on masks with other live counts equals eager.
"""

from __future__ import annotations

import typing

import pytest
import torch

from miniworld_engine.measure import config as C

pytestmark = [pytest.mark.gpu]


def _card() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((9, 0), (8, 0))


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _block(dtype: str) -> typing.Any:
    from miniworld_engine.modules.token_dit import TokenDiTMiniWorldBiasOnly

    m = TokenDiTMiniWorldBiasOnly(d_single=768, d_cond=384, d_pair=128, n_head=16, n=2).cuda().float()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    out: typing.Any = C.cast(m, dtype)
    return out.set_backend("cuda").eval()


def _autocast(dtype: str):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == "master")


def _masks(L: int) -> list[torch.Tensor]:
    out = []
    for n in (L, L - L // 8, 130, L - 3):
        m = torch.zeros(1, L, dtype=torch.bool, device="cuda")
        m[0, :n] = True
        out.append(m)
    g = torch.Generator(device="cuda").manual_seed(5)
    h = torch.rand(1, L, device="cuda", generator=g) > 0.3
    h[0, 3 * L // 4 :] = False
    out.append(h)
    return out


def _inputs(m, L: int, dtype: str):
    act = torch.float32 if dtype == "float32" else torch.bfloat16
    g = torch.Generator(device="cuda").manual_seed(1)
    return m.make_inputs(length=L, samples=5, depth=None, dtype=act, device="cuda", generator=g)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90): the skip toggled in the integration")
@pytest.mark.parametrize("dtype", ["master", "float32"])
@pytest.mark.parametrize("L", [384, 512])
def test_skip_bitwise(dtype: str, L: int, monkeypatch: pytest.MonkeyPatch) -> None:
    from miniworld_engine.kernels.integrations import bias_only_dit as I

    m = _block(dtype)
    inp = _inputs(m, L, dtype)
    for mask in _masks(L):
        kw = dict(inp, mask=mask)
        assert m.unsupported(**kw) is None, m.unsupported(**kw)
        with torch.no_grad(), _autocast(dtype):
            y = m(**kw).clone()
        with monkeypatch.context() as mp:
            mp.setattr(I, "kernel_mask", lambda mask, length: None)
            with torch.no_grad(), _autocast(dtype):  # fresh pair / mask storage: the P cache is keyed on them
                ref = m(**dict(kw, pair=inp["pair"].clone(), mask=mask.clone()))
        torch.testing.assert_close(y, ref, rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90): the hoisted-bias path")
@pytest.mark.parametrize("dtype", ["master", "float32"])
def test_skip_bitwise_hoisted_bias(dtype: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from miniworld_engine.kernels.integrations import bias_only_dit as I
    from miniworld_engine.modules.token_dit import pair_bias_all

    L = 512
    m = _block(dtype)
    inp = _inputs(m, L, dtype)
    with torch.no_grad(), _autocast(dtype):
        (bias,) = pair_bias_all([m], inp["pair"])
    for mask in _masks(L):
        kw = dict(inp, mask=mask)
        with torch.no_grad(), _autocast(dtype):
            y = m(**kw, bias=bias).clone()
        with monkeypatch.context() as mp:
            mp.setattr(I, "kernel_mask", lambda mask, length: None)
            with torch.no_grad(), _autocast(dtype):
                ref = m(**kw, bias=bias)
        torch.testing.assert_close(y, ref, rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not _card(), reason="H100 (sm_90) or A100 (sm_80)")
@pytest.mark.parametrize("dtype", ["master", "float32"])
def test_padded_against_reference(dtype: str) -> None:
    L = 512
    m = _block(dtype)
    inp = _inputs(m, L, dtype)
    for n in (150, 400):
        kw = dict(inp, mask=(torch.arange(L, device="cuda") < n)[None])
        with torch.no_grad(), _autocast(dtype):
            out = m(**kw).double()
            m.set_backend("pytorch")
            ref = m(**kw).double()
            m.set_backend("cuda")
        err = ((out - ref).abs().max() / ref.abs().max()).item()
        assert err < (3e-2 if dtype == "master" else 3e-3), (n, err)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) static input refresh")
def test_refresh_with_another_live_count() -> None:
    from miniworld_engine.capture import refresh_static_inputs, release_static_inputs, static_inputs, static_weights
    from miniworld_engine.modules import TokenDiTStack

    S, L = 5, 512
    torch.manual_seed(0)
    stack = TokenDiTStack("miniworld_bias_only", 2, 768, 384, 128, 16, 2, backend="cuda").float().cuda().eval().requires_grad_(False)
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
        for seed, n in ((2, 130), (3, 512), (4, 300)):
            fold(seed, n)
            ref = eager()
            refresh_static_inputs(stack)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, ref), f"fold {seed} (n {n}): refresh + replay != eager"
    finally:
        release_static_inputs(tables=tables)
