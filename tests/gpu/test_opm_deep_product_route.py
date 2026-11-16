"""OuterProductMean inference past MSA depth 4096 on the H100 ``cuda`` backend: the cuBLAS product + CUDA epilogue route
(``opm_train._product_route_deep`` / ``_product_epilogue``; per chunk of i rows when the O workspace would pass ``_O_CHUNK_BYTES``), for the five
module forms of the requirements rows (AF3 order, Boltz order, both with RoseTTAFold3's projection biases, AF3's ``count_offset``): inside the
engine accuracy standard against the float64 reference (``max|x - gold| / max|gold| <= max(err_pytorch, 8e-3)``), padded tokens and masked MSA
rows included, ragged depth, the pair residual, and bitwise the same on a second call. Depth <= 4096 keeps the fused kernel (checked too)."""

import copy

import pytest
import torch

from miniworld_engine.kernels.integrations import opm_train
from miniworld_engine.measure import config as C
from miniworld_engine.modules import OuterProductMean

pytestmark = [pytest.mark.gpu]

FORMS = {  # requirements rows 55 / 57 / 56 / 58 / 63
    "af3": {"normalize_before_proj": False},
    "boltz": {"normalize_before_proj": True},
    "rf3_after": {"normalize_before_proj": False, "projection_bias": True},
    "rf3_before": {"normalize_before_proj": True, "projection_bias": True},
    "af3_offset": {"normalize_before_proj": False, "count_offset": 1e-3},
}


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _call(module, msa, mask, residual):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=msa.dtype != torch.float64):
        return module(msa, mask, residual=residual).double()


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90a)")
@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize(
    ("length", "depth", "residual", "chunk_rows"),
    [
        (256, 8192, False, 0),
        (272, 6000, True, 0),  # ragged depth, N % 32 != 0
        (384, 16384, False, 0),
        (256, 8192, True, 64),  # the row-chunked product (workspace of 64 rows)
        (256, 2048, False, 0),  # the fused kernel (unchanged route)
    ],
)
def test_deep_product_route(form, length, depth, residual, chunk_rows, monkeypatch) -> None:
    gen = torch.Generator(device="cuda").manual_seed(depth + length)
    base = OuterProductMean(64, 128, 32, **FORMS[form]).cuda()
    C.randomize(base, gen)
    inp = base.make_inputs(length=length, samples=1, depth=depth, dtype=torch.bfloat16, device="cuda", generator=gen)
    msa, mask = inp["msa"], inp["mask"]
    mask[:, depth // 3 : depth // 3 + 97] = False  # whole masked MSA rows too
    res = torch.randn((1, length, length, 128), device="cuda", generator=gen).to(torch.bfloat16) if residual else None
    if chunk_rows:
        monkeypatch.setattr(opm_train, "_O_CHUNK_BYTES", chunk_rows * length * 32 * 32 * 2)
    calls = []
    real = opm_train._product_epilogue
    monkeypatch.setattr(opm_train, "_product_epilogue", lambda *a, **k: calls.append(1) or real(*a, **k))

    gold = _call(copy.deepcopy(base).double(), msa.double(), mask, None if res is None else res.double())
    ref = _call(copy.deepcopy(base), msa, mask, res)
    kern = copy.deepcopy(base)
    kern.set_backend("cuda")
    assert kern.unsupported(msa, mask) is None
    got = _call(kern, msa, mask, res)
    again = _call(kern, msa, mask, res)
    # the biased forms run composed.biased_wide (S % 256, N % 64), else composed.inference (cuBLAS + gather; no fused kernel to replace)
    served = not FORMS[form].get("projection_bias") or (depth % 256 == 0 and length % 64 == 0)
    assert bool(calls) == (depth > 4096 and served), f"product route taken {len(calls)} times at depth {depth}"

    scale = gold.abs().max().clamp_min(1e-30)
    err_ref = float((ref - gold).abs().max() / scale)
    err = float((got - gold).abs().max() / scale)
    print(f"{form} L{length} S{depth} res {residual} chunk {chunk_rows}: err {err:.3e} pytorch {err_ref:.3e}")
    assert err <= max(err_ref, 8e-3), f"cuda err {err:.3e}, pytorch {err_ref:.3e}"
    assert torch.equal(got, again), "run twice: not bitwise the same"
