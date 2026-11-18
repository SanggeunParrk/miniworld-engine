"""OuterProductMean inference (d_msa 64, d_pair 128) at lengths past 768 and at depths outside the TMA table on the H100 ``cuda`` backend: the
persistent TMA (wide) prologue replaces the non-TMA native one (825 us at L 1024 against 105 us). Checks the routing table, the engine accuracy
standard against the float64 reference (``max|x - gold| / max|gold| <= max(err_pytorch, 8e-3)``), and that the route is bitwise the explicit TMA
prologue route (``opm_prologue_tma_hopper``) on the same inputs. Training routing is unchanged."""

import copy

import pytest
import torch

from miniworld_engine.kernels.integrations import opm_train
from miniworld_engine.measure import config as C
from miniworld_engine.modules import OuterProductMean

pytestmark = [pytest.mark.gpu]


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def test_routing_table() -> None:
    f = opm_train._h100_path
    for n in (1024, 1536, 2048):  # past 768: wide TMA prologue at every depth
        for s in (256, 512, 768, 1024, 1536, 2048, 4096):
            assert f(s, n, training=False) == ("tma" if s == 512 else "legacy"), (s, n)
    assert f(1024, 128, training=False) == "tma" and f(1024, 768, training=False) == "legacy"
    assert f(768, 256, training=False) == "legacy"  # depth outside the table: was the native prologue
    assert f(1024, 1024, training=True) == "tma" and f(768, 1024, training=True) == "native"  # training untouched


def _call(module, msa, mask):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=msa.dtype != torch.float64):
        return module(msa, mask).double()


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90a)")
@pytest.mark.parametrize("before", [True, False])
@pytest.mark.parametrize(("length", "depth"), [(1024, 1024), (1088, 768), (192, 1536)])
def test_wide_prologue_route(before, length, depth, monkeypatch) -> None:
    gen = torch.Generator(device="cuda").manual_seed(length + depth)
    base = OuterProductMean(64, 128, 32, normalize_before_proj=before).cuda()
    C.randomize(base, gen)
    inp = base.make_inputs(length=length, samples=1, depth=depth, dtype=torch.bfloat16, device="cuda", generator=gen)
    msa, mask = inp["msa"], inp["mask"]
    mask[:, depth // 3 : depth // 3 + 33] = False
    gold = _call(copy.deepcopy(base).double(), msa.double(), mask)
    ref = _call(copy.deepcopy(base), msa, mask)
    kern = copy.deepcopy(base)
    kern.set_backend("cuda")
    assert kern.unsupported(msa, mask) is None
    got = _call(kern, msa, mask)
    scale = gold.abs().max().clamp_min(1e-30)
    err_ref, err = float((ref - gold).abs().max() / scale), float((got - gold).abs().max() / scale)
    print(f"before={before} L{length} S{depth}: err {err:.3e} pytorch {err_ref:.3e}")
    assert err <= max(err_ref, 8e-3), f"cuda err {err:.3e}, pytorch {err_ref:.3e}"
    monkeypatch.setattr(opm_train, "_h100_path", lambda s, n, *, training, wide=False: "tma")
    kern2 = copy.deepcopy(base)
    kern2.set_backend("cuda")
    assert torch.equal(got, _call(kern2, msa, mask)), "wide prologue route differs from the TMA prologue route"
