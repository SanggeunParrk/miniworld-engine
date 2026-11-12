"""FoldForge R13 on the kernel backends (H100 and A100): ``AtomDiTOpenFold3(mask_form="af3")`` under 128 real atoms matches the reference
backend's form, and the real rows read the padding keys (they differ from the default form's)."""

from __future__ import annotations

import typing

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules.atom_dit import AtomDiTOpenFold3

pytestmark = [pytest.mark.gpu]


def _card() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((9, 0), (8, 0))


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max() / b.float().abs().max())


@pytest.mark.skipif(not _card(), reason="H100 (sm_90) or A100 (sm_80)")
@pytest.mark.parametrize("backend", ["cuda", "triton"])
@pytest.mark.parametrize("dtype", ["master", "float32"])
def test_af3_mask_form_on_kernels(backend: str, dtype: str) -> None:
    m = AtomDiTOpenFold3(d_single=128, d_cond=128, d_pair=16, n_head=4, n=2, mask_form="af3").cuda().float()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    m = typing.cast("AtomDiTOpenFold3", C.cast(m, dtype).eval())
    act = torch.bfloat16 if dtype == "master" else torch.float32
    g = torch.Generator(device="cuda").manual_seed(1)
    inp = m.make_inputs(length=128, samples=5, depth=None, dtype=act, device="cuda", generator=g)
    real = inp["mask"][0]
    assert int(real.sum()) < 128
    out = {}
    for form in ("af3", "keys"):
        for b in (backend, "pytorch"):
            m.mask_form = form
            m.set_backend(b)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == "master"):
                out[form, b] = m(**inp)
    tol = 2e-2 if dtype == "master" else 2e-3
    assert _rel(out["af3", backend], out["af3", "pytorch"]) < tol
    # the forms differ there, far above the backends' rounding
    assert _rel(out["af3", backend][:, :, real], out["keys", backend][:, :, real]) > 2 * tol
