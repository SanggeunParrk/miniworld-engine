"""FoldForge R13: ``AtomDiTOpenFold3(mask_form="af3")``, AlphaFold 3's mask form (a masked logit only where both the query and the key are
padding), on the reference backend.

- Below 128 real atoms a real query reads the padding keys of its window [0, 128): its rows equal a call in which every one of the 128 atoms
  is a real key (the key mask None at 128 atoms, a separate path), and they move when a padding atom moves (the padding rows are not
  AlphaFold 3's, which never reads them: not compared).
- From 128 real atoms on the two forms agree exactly.
- The form is defined on the slid window only.
"""

from __future__ import annotations

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules.atom_dit import AtomDiTOpenFold3


def _pair(mask_form: str) -> tuple[AtomDiTOpenFold3, AtomDiTOpenFold3]:
    keys = AtomDiTOpenFold3(d_single=128, d_cond=128, d_pair=16, n_head=4, n=2)
    C.randomize(keys, torch.Generator().manual_seed(0))
    af3 = AtomDiTOpenFold3(d_single=128, d_cond=128, d_pair=16, n_head=4, n=2, mask_form=mask_form)
    af3.load_state_dict(keys.state_dict())
    return keys.eval(), af3.eval()


def _inputs(length: int, real: int) -> dict:
    m = AtomDiTOpenFold3()
    inp = m.make_inputs(length=length, samples=2, depth=None, dtype=torch.float32, device="cpu", generator=torch.Generator().manual_seed(1))
    inp["mask"] = (torch.arange(length) < real)[None]
    return inp


@torch.no_grad()
def test_af3_form_under_128_real_atoms() -> None:
    keys, af3 = _pair("af3")
    inp = _inputs(128, 100)
    real = inp["mask"][0]
    y = af3(**inp)
    y_keys = keys(**inp)
    y_all = keys(**{**inp, "mask": None})  # every atom a key: the window [0, 128) with no key off
    torch.testing.assert_close(y[:, :, real], y_all[:, :, real], rtol=0, atol=0)
    assert not torch.allclose(y[:, :, real], y_keys[:, :, real])  # the forms differ here
    moved = dict(inp)
    moved["single"] = inp["single"].clone()
    moved["single"][:, :, ~real] += 1.0  # the padding atoms are keys of the real queries
    assert not torch.allclose(af3(**moved)[:, :, real], y[:, :, real])
    torch.testing.assert_close(keys(**moved)[:, :, real], y_keys[:, :, real], rtol=0, atol=0)


@torch.no_grad()
@pytest.mark.parametrize("real", [128, 200])
def test_af3_form_from_128_real_atoms_is_the_default(real: int) -> None:
    keys, af3 = _pair("af3")
    inp = _inputs(256, real)
    torch.testing.assert_close(af3(**inp), keys(**inp), rtol=0, atol=0)


def test_af3_form_needs_the_slid_window() -> None:
    with pytest.raises(ValueError, match="slide"):
        AtomDiTOpenFold3(key_window="pad", mask_form="af3")
    with pytest.raises(ValueError, match="mask_form"):
        AtomDiTOpenFold3(mask_form="both")
