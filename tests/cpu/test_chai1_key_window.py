"""FoldForge R9: ``AtomDiTChai1(key_window="pad")`` + ``uid`` is Chai-1's release window exactly; there is no circular window to add.

Chai-1 0.6.1 (``chai_lab/model/utils.py``): ``get_qkv_indices_for_blocks`` takes keys ``[32 w - 48, 32 w + 80)`` and returns
``kv_mask = (kv_indices < N) & (kv_indices >= 0)`` before ``kv_indices % N`` (the ``%`` only indexes: "mask out positions where kv_indices
gets wrapped"); ``get_block_atom_pair_mask`` ANDs that mask with the query and key atom masks, and the atom-pair feature generator
(``blocked_atom_pair_distances.get_blocked_atom_pair_dists``) ANDs the same-conformer test ``atom_ref_space_uid`` into it in place. So a
wrapped key is never attended, for any system size and any conformer layout (also a conformer covering both ends of the atom list, and
under 128 atoms).

The test reads which keys every real query attends from the reference block's Jacobian (a query's output depends on another atom only
through the attention) and compares it with that transcription of the release's mask.
"""

from __future__ import annotations

import pytest
import torch

from miniworld_engine.measure import config as C
from miniworld_engine.modules.atom_dit import AtomDiTChai1


def _release_mask(atom_mask: torch.Tensor, uid: torch.Tensor) -> torch.Tensor:
    """[N, N] bool: query i attends key j in Chai-1's release (a transcription of the three functions named above)."""
    n = atom_mask.shape[0]
    q_idx = torch.arange(n).view(n // 32, 32)
    kv_idx = q_idx[:, :1] + (32 - 128) // 2 + torch.arange(128)
    kv_mask = (kv_idx < n) & (kv_idx >= 0)
    kv_idx = kv_idx % n
    pair = atom_mask[q_idx][..., None] & atom_mask[kv_idx][:, None, :] & kv_mask[:, None, :]
    pair &= uid[q_idx][..., None] == uid[kv_idx][:, None, :]
    out = torch.zeros(n, n, dtype=torch.bool)
    for w in range(n // 32):
        for qi in range(32):
            out[32 * w + qi, kv_idx[w][pair[w, qi]]] = True
    return out


@pytest.mark.parametrize(
    ("n", "real", "conformers"),
    [
        (160, 160, "one"),  # one conformer spanning both ends: the wrapped keys of window 0 are its own atoms
        (160, 150, "two"),  # two conformers, padding at the end
        (96, 80, "one"),  # under 128 atoms
    ],
)
def test_pad_plus_uid_is_the_release_window(n: int, real: int, conformers: str) -> None:
    torch.manual_seed(0)
    m = AtomDiTChai1(d_single=32, d_cond=32, d_pair=4, n_head=1, n=2, key_window="pad").double()
    C.randomize(m, torch.Generator().manual_seed(0))
    atom_mask = torch.arange(n) < real
    uid = torch.zeros(n, dtype=torch.long) if conformers == "one" else (torch.arange(n) >= n // 2).long()
    inp = m.make_inputs(length=n, samples=1, depth=None, dtype=torch.float64, device="cpu", generator=torch.Generator().manual_seed(1))
    inp["mask"], inp["uid"] = atom_mask[None], uid[None]
    single = inp["single"].clone().requires_grad_(True)
    out = m(**{**inp, "single": single})
    seen = torch.zeros(n, n, dtype=torch.bool)
    for i in range(real):
        (g,) = torch.autograd.grad(out[0, 0, i].sum(), single, retain_graph=True)
        seen[i] = g[0, 0].abs().sum(-1) > 0
    expected = _release_mask(atom_mask, uid)
    eye = torch.eye(n, dtype=torch.bool)
    torch.testing.assert_close(seen[:real] & ~eye[:real], expected[:real] & ~eye[:real], rtol=0, atol=0)
