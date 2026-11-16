"""H100 bidirectional TriMul inference (bf16) at D64 / D128:

* the contractions run through ``_h100_bmm.contract`` and the measured cuBLASLt table (``_h100_bmm_lt_bidir.LT_BIDIR``): the module output is BIT-IDENTICAL
  to the same call with the table removed (the default heuristic pick), over listed and unlisted lengths, masks, B = 1 / 2;
* ``h100_inference.PORT_K3`` (default off, MINIWORLD_TRIMUL_BIDIR_PORT_K3): off is byte-identical to the packaged route; on (D128, L >= 160) the output
  differs in bf16 ulps but its error against an fp64 reference is not above the packaged K3's (random inputs, ragged L, masks, B = 2).
"""

import pytest
import torch

from miniworld_engine.kernels.trimul_inproj.cuda import _h100_bmm as H
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_bmm_lt_bidir as T
from miniworld_engine.kernels.trimul_inproj.cuda import h100_inference as HI
from miniworld_engine.modules.triangle import BidirectionalTriangleMultiplication as BT

pytestmark = [pytest.mark.gpu, pytest.mark.h100]


def _module(d: int) -> BT:
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")
    torch.manual_seed(0)
    m = BT(d, master_weights=True, backend="cuda").cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() > 1:
                p.normal_(0, d**-0.5)
    return m


def test_table_entries_are_bidirectional_keys() -> None:
    assert T.LT_BIDIR
    assert {k[2] for k in T.LT_BIDIR} <= {64, 128}
    assert all(H.LT_CONFIGS[k] == v for k, v in T.LT_BIDIR.items())


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize(("batch", "length"), [(1, 144), (1, 416), (1, 832), (1, 1024), (1, 1000), (1, 300), (2, 256)])
def test_table_is_bit_identical_to_default_pick(d: int, batch: int, length: int, monkeypatch: pytest.MonkeyPatch) -> None:
    m = _module(d)
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(batch, length, length, d, device="cuda", generator=g).bfloat16()
    mask = torch.rand(batch, length, device="cuda", generator=g) > 0.1
    with torch.no_grad():
        tabled = m(x, mask)
        for key in [k for k in H.LT_CONFIGS if k[2] == d]:
            monkeypatch.delitem(H.LT_CONFIGS, key)
        plain = m(x, mask)
    assert torch.equal(tabled, plain)


@pytest.mark.parametrize("length", [160, 300, 1000])
def test_port_k3_off_is_byte_identical(length: int, monkeypatch: pytest.MonkeyPatch) -> None:
    m = _module(128)
    x = torch.randn(1, length, length, 128, device="cuda").bfloat16()
    with torch.no_grad():
        monkeypatch.setattr(HI, "PORT_K3", False)
        a = m(x, None)
        b = m(x, None)
    assert torch.equal(a, b)


@pytest.mark.parametrize(("batch", "length", "masked"), [(1, 160, False), (1, 300, True), (1, 1000, True), (2, 256, True)])
def test_port_k3_error_not_above_packaged(batch: int, length: int, masked: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    d = 128
    m = _module(d)
    ref = BT(d, master_weights=False, backend="pytorch").cuda().eval().double()
    with torch.no_grad():
        ref.load_state_dict({k: v.detach().bfloat16().double() if v.dim() > 1 else v.detach().double() for k, v in m.state_dict().items()})
        g = torch.Generator(device="cuda").manual_seed(2)
        x = torch.randn(batch, length, length, d, device="cuda", generator=g).bfloat16()
        mask = (torch.rand(batch, length, device="cuda", generator=g) > 0.2) if masked else None
        y64 = ref(x.double(), mask)
        monkeypatch.setattr(HI, "PORT_K3", False)
        packaged = m(x, mask)
        monkeypatch.setattr(HI, "PORT_K3", True)
        port = m(x, mask)
    rms = lambda y: (y.double() - y64).pow(2).mean().sqrt().item()
    assert not torch.equal(packaged, port)
    assert rms(port) <= rms(packaged)
