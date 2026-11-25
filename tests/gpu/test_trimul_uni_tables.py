"""H100 single-direction TriMul inference (bf16) at D64 / D128, the two tables of 2.5.13:

* the contraction (hidden = D: the packaged route and, from ``ROWPAD_FROM``, the port's row-padded form) runs through ``_h100_bmm.contract`` and
  the measured single-direction cuBLASLt table (``_h100_bmm_lt_uni.LT_UNI``): the module output is BIT-IDENTICAL to the same call with the
  rc/2.5.13 table (``LT_CONFIGS``), over listed and unlisted lengths, both directions, masks, B = 1 / 2;
* the launch-table cells without a mask (``trimul_fwd_sm90_cuda`` MASK=0, DIR 1 / 2, hidden D and 2 D): the selected K1 / K3 tiles give an
  output BIT-IDENTICAL to the kernel default's (what a no-mask call ran before the cells existed), at grid and off-grid lengths.
"""

import pytest
import torch

from foldjet_engine.kernels.trimul_inproj.cuda import _h100_bmm as H
from foldjet_engine.kernels.trimul_inproj.cuda import _h100_bmm_lt_uni as U
from foldjet_engine.kernels.trimul_inproj.cuda import h100_b200style as PORT
from foldjet_engine.kernels.trimul_inproj.cuda import h100_inference as HI
from foldjet_engine.modules.triangle import TriangleMultiplication as TM

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def _module(d: int, hidden: int, outgoing: bool) -> TM:
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")
    torch.manual_seed(0)
    m = TM(d, d_hidden=hidden, outgoing=outgoing, master_weights=True, backend="cuda").cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() > 1:
                p.normal_(0, d**-0.5)
    return m


def test_table_keys_are_single_direction() -> None:
    assert U.LT_UNI
    assert {k[2] for k in U.LT_UNI} <= {64, 128, 256}  # planes per launch = D B
    assert HI.LT_UNI is U.LT_UNI and PORT.LT_UNI is U.LT_UNI


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize(("batch", "length"), [(1, 144), (1, 300), (1, 512), (1, 800), (1, 1024), (1, 1536), (2, 256), (2, 640), (2, 1024)])
def test_contraction_table_is_bit_identical(d: int, outgoing: bool, batch: int, length: int, monkeypatch: pytest.MonkeyPatch) -> None:
    m = _module(d, d, outgoing)
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(batch, length, length, d, device="cuda", generator=g).bfloat16()
    mask = torch.rand(batch, length, device="cuda", generator=g) > 0.1
    with torch.no_grad():
        tabled = (m(x, None), m(x, mask))
        before = dict(H.LT_CONFIGS)  # the rc/2.5.13 route: contract(table=None)
        monkeypatch.setattr(HI, "LT_UNI", before)
        monkeypatch.setattr(PORT, "LT_UNI", before)
        plain = (m(x, None), m(x, mask))
    assert torch.equal(tabled[0], plain[0]) and torch.equal(tabled[1], plain[1])


@pytest.mark.parametrize(("d", "hidden"), [(64, 64), (128, 256)])
@pytest.mark.parametrize(("batch", "length"), [(1, 144), (1, 300), (2, 176), (3, 256)])
def test_no_mask_off_grid_takes_the_kernel_default(d: int, hidden: int, batch: int, length: int) -> None:
    """Between the grid lengths (and for B > 2) a no-mask single-direction call runs the length's own kernel default, not the nearest cell."""
    m = _module(d, hidden, True)
    x = torch.empty(batch, length, length, d, device="cuda", dtype=torch.bfloat16)
    w = [m.to_left.weight.bfloat16()]
    assert HI._select_config(x, w, None, 1, 1e-5) == HI.trimul_candidates(d, hidden, length, 1)[0]


@pytest.mark.parametrize(("d", "hidden"), [(64, 64), (128, 128), (64, 128), (128, 256)])
@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize(("batch", "length"), [(1, 128), (1, 200), (1, 384), (1, 640), (2, 256), (2, 512)])
def test_no_mask_cells_are_bit_identical_to_default(
    d: int, hidden: int, outgoing: bool, batch: int, length: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = _module(d, hidden, outgoing)
    x = torch.randn(batch, length, length, d, device="cuda").bfloat16()
    picked = []
    select = HI._select_config
    monkeypatch.setattr(HI, "_select_config", lambda *a: picked.append(select(*a)) or picked[-1])
    with torch.no_grad():
        tabled = m(x, None)
        assert picked, "the packaged route was not taken"
        default = HI.trimul_candidates(d, hidden, length, 1 if outgoing else 2)[0]
        monkeypatch.setattr(HI, "_select_config", lambda *a: default)
        plain = m(x, None)
    assert torch.equal(tabled, plain)


@pytest.mark.parametrize(("d", "hidden"), [(64, 128), (128, 256)])
@pytest.mark.parametrize(("batch", "length"), [(1, 144), (1, 176), (1, 912), (1, 1104), (3, 256)])
def test_bidirectional_no_mask_off_grid_takes_the_kernel_default(d: int, hidden: int, batch: int, length: int) -> None:
    """The bidirectional DIR 0 MASK=0 cells are read at their exact (B, L) too: between the grid lengths (and B > 2) the kernel default."""
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")
    x = torch.empty(batch, length, length, d, device="cuda", dtype=torch.bfloat16)
    w = [torch.empty(hidden, d, device="cuda", dtype=torch.bfloat16)]
    assert HI._select_config(x, w, None, 0, 1e-5) == HI.trimul_candidates(d, hidden, length, 0)[0]


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize(("batch", "length"), [(1, 144), (1, 176), (1, 912), (1, 1104), (2, 300)])
def test_bidirectional_no_mask_off_grid_is_bit_identical_to_nearest_cell(
    d: int, batch: int, length: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The output at an off-grid length is the same with the kernel default (now) as with the nearest cell's tiles (2.5.12)."""
    from foldjet_engine.kernels import _launch_config as LC
    from foldjet_engine.modules.triangle import BidirectionalTriangleMultiplication as BTM

    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")
    torch.manual_seed(0)
    m = BTM(d, d_hidden=d, master_weights=True, backend="cuda").cuda().eval()
    x = torch.randn(batch, length, length, d, device="cuda").bfloat16()
    picked = []
    select = HI._select_config
    monkeypatch.setattr(HI, "_select_config", lambda *a: picked.append(select(*a)) or picked[-1])
    with torch.no_grad():
        now = m(x, None)
        if not picked:
            pytest.skip("not the packaged route")
        choose = LC.choose
        monkeypatch.setattr(HI, "choose", lambda *a, **k: choose(*a, **{**k, "near": ("B", "L")}))
        picked.clear()
        before = m(x, None)
    assert torch.equal(now, before)
