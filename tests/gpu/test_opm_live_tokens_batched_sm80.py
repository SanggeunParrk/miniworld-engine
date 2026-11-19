"""A100 (sm_80) OuterProductMean's masked outer product helpers: the dead-token liveness kernel (``ogemm_sm80.live_tokens``: one launch instead of
torch's reduction over the MSA rows) and the batched skip product of the Chai-1 groups (``ogemm_sm80.outer_batched``: one launch for the 8 groups,
bit for bit the eight per-group launches)."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.a100]


def _a100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)


@pytest.mark.skipif(not _a100(), reason="an A100 (sm_80)")
@pytest.mark.parametrize("rows", [1, 7, 384, 1024, 2048, 3000])
@pytest.mark.parametrize("length", [4, 36, 100, 128, 384, 640])
def test_live_tokens_matches_the_row_max(rows: int, length: int) -> None:
    from miniworld_engine.kernels.outer_product_mean.cuda import ogemm_sm80

    gen = torch.Generator(device="cuda").manual_seed(rows * 1000 + length)
    mask = (torch.rand(rows, length, device="cuda", generator=gen) > 0.7).to(torch.uint8)
    mask[:, length - length // 8 :] = 0  # padding positions: dead in every row
    assert torch.equal(ogemm_sm80.live_tokens(mask), mask.amax(0).contiguous())


@pytest.mark.skipif(not _a100(), reason="an A100 (sm_80)")
@pytest.mark.parametrize("tokens,depth", [(384, 128), (512, 256), (768, 64)])
def test_batched_skip_product_is_the_per_group_launches(tokens: int, depth: int) -> None:
    from miniworld_engine.kernels.outer_product_mean.cuda import ogemm_sm80

    gen = torch.Generator(device="cuda").manual_seed(tokens)
    a = torch.randn(8, depth, 8 * tokens, device="cuda", dtype=torch.bfloat16, generator=gen)
    b = torch.randn(8, depth, 8 * tokens, device="cuda", dtype=torch.bfloat16, generator=gen)
    live = (torch.rand(tokens, device="cuda", generator=gen) > 0.25).to(torch.uint8)
    live[:8] = 1
    live[-tokens // 8 :] = 0
    dead = (live == 0).repeat_interleave(8)  # a dead token has a = b = 0 (the prologue masks them)
    a[:, :, dead] = 0
    b[:, :, dead] = 0
    batched = ogemm_sm80.outer_batched(a, b, live, 8)
    each = torch.empty_like(batched)
    for g in range(8):
        ogemm_sm80.outer(a[g], b[g], live, 8, each[g])
    assert torch.equal(batched, each)
    dense = torch.bmm(a.transpose(1, 2), b)
    assert (batched.float() - dense.float()).abs().max().item() <= 1e-2 * dense.float().abs().max().item()


def _opm_outputs(length: int, depth: int, padded: bool, probe: str, monkeypatch) -> tuple[torch.Tensor, torch.Tensor]:
    import contextlib

    from miniworld_engine.kernels.outer_product_mean.cuda import ogemm_sm80
    from miniworld_engine.modules.msa import OuterProductMean
    from miniworld_engine.modules.primitives import unpadded_masks

    monkeypatch.setenv("MINIWORLD_OPM_PROBE", probe)
    ogemm_sm80._PROBES.clear()
    torch.manual_seed(0)
    m = OuterProductMean(d_msa=64, d_pair=128, d_hidden=32, normalize_before_proj=True).cuda().to(torch.bfloat16)
    gen = torch.Generator(device="cuda").manual_seed(1)
    with contextlib.nullcontext() if padded else unpadded_masks():
        inp = m.make_inputs(length=length, samples=1, depth=depth, dtype=torch.bfloat16, device="cuda", generator=gen)
    outs = []
    for backend in ("cuda", "pytorch"):
        m.set_backend(backend)
        with torch.no_grad():
            outs.append(m(**inp).float())
    return outs[0], outs[1]


@pytest.mark.skipif(not _a100(), reason="an A100 (sm_80)")
def test_all_live_mask_takes_the_dense_product_and_padding_the_skip_product(monkeypatch) -> None:
    for padded in (False, True):
        on, ref = _opm_outputs(256, 384, padded, "1", monkeypatch)
        off, _ = _opm_outputs(256, 384, padded, "0", monkeypatch)
        assert (on - ref).abs().max().item() <= 2e-2 * ref.abs().max().item()  # both products against the PyTorch reference
        if padded:
            assert torch.equal(on, off)  # padded: the same skip kernel either way
        else:
            assert (on - off).abs().max().item() <= 2e-2 * off.abs().max().item()


@pytest.mark.skipif(not _a100(), reason="an A100 (sm_80)")
def test_chai_products_per_group_route_table_and_values() -> None:
    from miniworld_engine.kernels.outer_product_mean.cuda import chai_sm80, ogemm_sm80

    per_group = chai_sm80._per_group
    assert not per_group(1024, 1280) and per_group(1024, 1536) and per_group(1024, 2048)
    assert not per_group(512, 2048) and not per_group(2048, 640) and per_group(2048, 768) and per_group(4096, 640) and not per_group(4096, 512)
    s, n = 2048, 768  # the per-group route; all-live and padded masks
    gen = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(8, s, 8 * n, device="cuda", dtype=torch.bfloat16, generator=gen)
    b = torch.randn(8, s, 8 * n, device="cuda", dtype=torch.bfloat16, generator=gen)
    for pad in (False, True):
        mask = torch.ones(s, n, device="cuda", dtype=torch.uint8)
        if pad:
            mask[:, n - n // 8 :] = 0
            dead = (mask.amax(0) == 0).repeat_interleave(8)
            a[:, :, dead] = 0
            b[:, :, dead] = 0
        ogemm_sm80._PROBES.clear()
        got = chai_sm80.chai_products(a.reshape(8, s * n, 8), b.reshape(8, s * n, 8), mask, s, n)
        each = torch.empty_like(got)
        live = ogemm_sm80.live_tokens(mask)
        for g in range(8):
            ogemm_sm80.outer(a[g], b[g], live, 8, each[g])
        assert torch.equal(got, each)
        dense = torch.bmm(a.transpose(1, 2), b)
        assert (got.float() - dense.float()).abs().max().item() <= 1e-2 * dense.float().abs().max().item()
