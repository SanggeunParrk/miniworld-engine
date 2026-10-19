"""``LocalDiTBlock(kv_norm=..., key_window=...)``: AlphaFold 3's atom attention against its own definition.

AF3 normalises the keys and values with a second AdaLN of the *raw* atoms next to the first one (``kv_norm="parallel"``; Protenix chains
it after the first, the default), and gives every query window the 128 keys ``[32 w - 48, 32 w + 80)`` slid back inside the real atoms
(``key_window="slide"``; the default clips the window and masks what is outside). The reference here is the window rule of
``alphafold3.model.features.AtomCrossAtt.compute_features`` copied, and an attention written per window from it.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.local_dit.module import KEYS, QUERIES, LocalDiTBlock, windows

N_PAD = 7 * QUERIES            # 224 atoms, 7 query windows


def af3_key_windows(num_atoms: int, padded_atoms: int) -> np.ndarray:
    """The AF3 rule, as in ``AtomCrossAtt.compute_features`` (queries_subset_size 32, keys_subset_size 128)."""
    centers = np.arange(QUERIES / 2, padded_atoms, QUERIES)
    gathers = (centers[:, None] + np.arange(-KEYS / 2, KEYS / 2)[None, :]).astype(int)
    for row in range(gathers.shape[0]):
        if gathers[row, 0] < 0:
            gathers[row, :] -= gathers[row, 0]
        elif gathers[row, -1] > num_atoms - 1:
            gathers[row, :] -= gathers[row, -1] - (num_atoms - 1)
    return gathers


def build(**options):
    torch.manual_seed(0)
    module = LocalDiTBlock(cross_attention=True, implementation=ImplementationType.PYTORCH, **options)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.1)
    return module.eval()


def inputs(num_atoms):
    torch.manual_seed(1)
    single = torch.randn(2, 1, N_PAD, 128)
    cond = torch.randn(2, 1, N_PAD, 128)
    pair = torch.randn(1, windows(N_PAD), QUERIES, KEYS, 16)
    mask = torch.zeros(1, N_PAD, dtype=torch.bool)
    mask[:, :num_atoms] = True
    return single, cond, pair, mask


def af3_attention_delta(module, single, cond, pair, num_atoms):
    """Atom attention per query window, from the gathers of ``af3_key_windows``: q from the first AdaLN, k / v from the second AdaLN
    of the raw atoms (AF3), softmax over the 128 keys (all inside the real atoms), gate and output projection."""
    att = module.attention
    x = att.ada_ln_in(single, cond)
    xkv = att.ada_ln_kv(single, cond)
    q, k, v, gate = att.to_query(x), att.to_key(xkv), att.to_value(xkv), att.to_gate(x)
    h, d = att.n_head, x.shape[-1] // att.n_head
    gathers = af3_key_windows(num_atoms, N_PAD)
    z = F.layer_norm(pair, (pair.shape[-1],), att.ln_pair.weight, None, att.ln_pair.eps)
    bias = F.linear(z, att.to_bias.weight)                                  # [1, w, 32, 128, h]
    out = torch.zeros(q.shape)
    for w in range(windows(N_PAD)):
        keys = torch.as_tensor(gathers[w])
        for a in range(q.shape[0]):
            qw = q[a, 0, w * QUERIES : (w + 1) * QUERIES].view(QUERIES, h, d)
            kw, vw = k[a, 0, keys].view(KEYS, h, d), v[a, 0, keys].view(KEYS, h, d)
            logits = torch.einsum("qhd,khd->hqk", qw, kw) * d**-0.5 + bias[0, w].permute(2, 0, 1)
            attended = torch.einsum("hqk,khd->qhd", logits.softmax(-1), vw).reshape(QUERIES, h * d)
            out[a, 0, w * QUERIES : (w + 1) * QUERIES] = attended
    out = att.to_out(torch.sigmoid(gate) * out)
    return torch.sigmoid(att.to_scale(cond)) * out


@pytest.mark.parametrize("num_atoms", [N_PAD, 200, 150, 130])
def test_af3_block_matches_the_af3_definition(num_atoms):
    module = build(kv_norm="parallel", key_window="slide")
    single, cond, pair, mask = inputs(num_atoms)
    with torch.no_grad():
        actual = module.attention_delta(single, cond, pair, mask)
        expected = af3_attention_delta(module, single, cond, pair, num_atoms)
    torch.testing.assert_close(actual[:, :, :num_atoms], expected[:, :, :num_atoms], rtol=1e-4, atol=1e-4)


def test_the_window_rule_is_the_af3_one():
    from miniworld_engine.modules.local_dit.module import slide_starts

    for num_atoms in (130, 150, 200, N_PAD):
        gathers = af3_key_windows(num_atoms, N_PAD)
        starts = slide_starts(torch.tensor([num_atoms]), windows(N_PAD))[0]
        assert starts.tolist() == gathers[:, 0].tolist()


def test_defaults_are_untouched_and_the_options_change_the_edges_only():
    single, cond, pair, mask = inputs(N_PAD)
    default = build()
    slide = build(key_window="slide")
    parallel = build(kv_norm="parallel")
    with torch.no_grad():
        a = default.attention_delta(single, cond, pair, mask)
        b = slide.attention_delta(single, cond, pair, mask)
        c = parallel.attention_delta(single, cond, pair, mask)
    # the slide differs from the clipped window on the first two and the last windows only
    interior = slice(2 * QUERIES, (windows(N_PAD) - 2) * QUERIES)
    torch.testing.assert_close(a[:, :, interior], b[:, :, interior], rtol=1e-5, atol=1e-5)
    assert (a[:, :, : 2 * QUERIES] - b[:, :, : 2 * QUERIES]).abs().max() > 1e-3
    # the K/V norm changes every atom
    assert (a - c).abs().mean() > 1e-3


def test_option_validation():
    with pytest.raises(ValueError, match="kv_norm"):
        LocalDiTBlock(cross_attention=True, kv_norm="serial")
    with pytest.raises(ValueError, match="key_window"):
        LocalDiTBlock(cross_attention=True, key_window="wrap")
    with pytest.raises(ValueError, match="cross_attention"):
        LocalDiTBlock(kv_norm="parallel")
