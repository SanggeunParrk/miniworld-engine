"""``AtomDiTESMFold2Pair`` (FoldForge R6): ESMFold2's atom block with the atom pair against its own definition, written per query window.

ESMFold2 (FoldForge ``spec.ESMFOLD2``) is AlphaFold 3's atom block -- parallel K / V AdaLN, slid key window -- with a 192-key subset
(``AtomCrossAtt.compute_features(keys_subset_size=192)``), a +/- 64 window by rank among the real atoms, AdaLNs of the form
``(1 + W_s silu(c)) * rms(x) + W_b silu(c)``, raw output gates ``W silu(c)``, and q / k through an affine-free RMSNorm per head and a
half-split rotary from per-atom tables. The reference below spells each of those out from the definition (``esm_block``), with the window
rule of the AF3 featuriser copied. (Restored from the 2.4.x ``tests/cpu/compile/test_local_dit_esmfold2_variant.py``.)
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.modules import AtomDiT, AtomDiTESMFold2, AtomDiTESMFold2Pair
from miniworld_engine.modules.atom_dit import QUERIES, windows
from miniworld_engine.modules.functional import slide_starts
from miniworld_engine.modules.primitives import BackendUnsupported

KEYS, HALF = 192, 64
EPS = torch.finfo(torch.float32).eps


def af3_key_windows(num_atoms: int, padded_atoms: int) -> np.ndarray:
    """The AF3 rule, as in ``AtomCrossAtt.compute_features`` (queries_subset_size 32, keys_subset_size 192)."""
    centers = np.arange(QUERIES / 2, padded_atoms, QUERIES)
    gathers = (centers[:, None] + np.arange(-KEYS / 2, KEYS / 2)[None, :]).astype(int)
    for row in range(gathers.shape[0]):
        if gathers[row, 0] < 0:
            gathers[row, :] -= gathers[row, 0]
        elif gathers[row, -1] > num_atoms - 1:
            gathers[row, :] -= gathers[row, -1] - (num_atoms - 1)
    return gathers


def build(seed=0):
    torch.manual_seed(seed)
    module = AtomDiTESMFold2Pair()
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.1)
    return module.eval().double()


def inputs(padded, num_atoms, a=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    single = torch.randn(a, 1, padded, 128, generator=g, dtype=torch.float64)
    cond = torch.randn(a, 1, padded, 128, generator=g, dtype=torch.float64)
    pair = torch.randn(1, windows(padded), QUERIES, KEYS, 16, generator=g, dtype=torch.float64)
    mask = torch.zeros(1, padded, dtype=torch.bool)
    mask[:, :num_atoms] = True
    angles = torch.randn(1, padded, 16, generator=g, dtype=torch.float64) * 3
    return single, cond, pair, mask, (angles.cos(), angles.sin())


def rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


def adaln(x, c, w_scale, w_bias):
    c = F.silu(c)
    return (c @ w_scale.T + 1) * rms(x) + c @ w_bias.T


def rotate(x, cos, sin):  # x [..., 32], cos / sin [..., 16]: channel i pairs with i + 16 (FoldForge's tiled tables)
    a, b = x[..., :16], x[..., 16:]
    return torch.cat([a * cos - b * sin, b * cos + a * sin], -1)


def esm_block(module, single, cond, pair, num_atoms, rope):
    """The block from the definition: one window at a time, one sample at a time."""
    p = dict(module.named_parameters())
    w = lambda name: p[name].detach().double()
    cos, sin = rope
    padded = single.shape[2]
    gathers = af3_key_windows(num_atoms, padded)
    out = single.clone()
    x = adaln(single, cond, w("attention.ada_ln_in.to_scale.weight"), w("attention.ada_ln_in.to_bias.weight"))
    xkv = adaln(single, cond, w("attention.ada_ln_kv.to_scale.weight"), w("attention.ada_ln_kv.to_bias.weight"))
    q = x @ w("attention.to_query.weight").T + w("attention.to_query.bias")
    k, v = xkv @ w("attention.to_key.weight").T, xkv @ w("attention.to_value.weight").T
    gate = torch.sigmoid(x @ w("attention.to_gate.weight").T)
    q = rotate(rms(q.reshape(*q.shape[:-1], 4, 32)), cos[0, :, None], sin[0, :, None])
    k = rotate(rms(k.reshape(*k.shape[:-1], 4, 32)), cos[0, :, None], sin[0, :, None])
    v = v.reshape(*v.shape[:-1], 4, 32)
    z = F.layer_norm(pair, (16,), w("attention.ln_pair.weight"), None, 1e-5)
    bias = (z @ w("attention.to_bias.weight").T)[0]  # [nwin, 32, 192, h]
    att = torch.zeros(q.shape, dtype=q.dtype)
    for win in range(windows(padded)):
        slots = torch.as_tensor(gathers[win])
        keys = slots.clamp(0, padded - 1)
        real = (slots >= 0) & (slots < num_atoms)  # AF3's negative indices are its padding
        queries = torch.arange(win * QUERIES, (win + 1) * QUERIES)
        visible = real[None, :] & ((queries[:, None] - slots[None, :]).abs() <= HALF)
        for s in range(single.shape[0]):
            logits = torch.einsum("qhd,khd->hqk", q[s, 0, queries], k[s, 0, keys]) / math.sqrt(32) + bias[win].permute(2, 0, 1)
            logits = logits.masked_fill(~visible[None], float("-inf"))
            att[s, 0, queries] = torch.einsum("hqk,khd->qhd", torch.nan_to_num(logits.softmax(-1)), v[s, 0, keys])
    att = (gate * att.reshape(*att.shape[:-2], 128)) @ w("attention.to_out.weight").T
    out = out + (F.silu(cond) @ w("attention.to_scale.weight").T) * att
    x2 = adaln(out, cond, w("transition.ada_ln_in.to_scale.weight"), w("transition.ada_ln_in.to_bias.weight"))
    hidden = F.silu(x2 @ w("transition.expand_a.weight").T) * (x2 @ w("transition.expand_b.weight").T)
    return out + (F.silu(cond) @ w("transition.to_scale.weight").T) * (hidden @ w("transition.squeeze.weight").T)


@pytest.mark.parametrize(("padded", "num_atoms"), [(256, 256), (288, 250), (320, 193), (224, 150), (256, 100)])
def test_block_matches_the_definition(padded, num_atoms):
    module = build()
    single, cond, pair, mask, rope = inputs(padded, num_atoms)
    with torch.no_grad():
        actual = module(single, cond, pair, mask, rope=rope)
        expected = esm_block(module, single, cond, pair, num_atoms, rope)
    torch.testing.assert_close(actual[:, :, :num_atoms], expected[:, :, :num_atoms], rtol=1e-10, atol=1e-10)


def test_the_window_rule_is_the_af3_one_at_192_keys():
    for padded, num_atoms in ((256, 256), (288, 250), (320, 193), (224, 150), (256, 100), (4096, 4000)):
        starts = slide_starts(torch.tensor([num_atoms]), windows(padded), KEYS)[0]
        assert starts.tolist() == af3_key_windows(num_atoms, padded)[:, 0].tolist()


def test_the_parameter_layout():
    """FoldForge's ESMFold2 atom block: no conditioning LayerNorm, no scale / gate biases, an affine-free q / k norm; the query bias kept."""
    names = {name for name, _ in AtomDiTESMFold2Pair().named_parameters()}
    assert names == {
        "attention.ada_ln_in.to_scale.weight",
        "attention.ada_ln_in.to_bias.weight",
        "attention.ada_ln_kv.to_scale.weight",
        "attention.ada_ln_kv.to_bias.weight",
        "attention.to_query.weight",
        "attention.to_query.bias",
        "attention.to_key.weight",
        "attention.to_value.weight",
        "attention.ln_pair.weight",
        "attention.to_bias.weight",
        "attention.to_gate.weight",
        "attention.to_out.weight",
        "attention.to_scale.weight",
        "transition.ada_ln_in.to_scale.weight",
        "transition.ada_ln_in.to_bias.weight",
        "transition.expand_a.weight",
        "transition.expand_b.weight",
        "transition.squeeze.weight",
        "transition.to_scale.weight",
    }


def test_selector_keeps_the_pair_free_block():
    assert type(AtomDiT("esmfold2_pair")) is AtomDiTESMFold2Pair
    assert type(AtomDiT("esmfold2")) is AtomDiTESMFold2  # ESMFold2's released SWA block is unchanged


def test_validation_and_cpu_refusals():
    module = build()
    single, cond, pair, mask, rope = inputs(256, 256)
    with pytest.raises(ValueError, match="rope"):
        module(single, cond, pair, mask)
    with pytest.raises(ValueError, match="pair"):
        module(single, cond, pair[:, :, :, :128], mask, rope=rope)
    for backend in ("triton", "cuda"):
        module.set_backend(backend)
        assert module.unsupported(single, cond, pair, mask, rope=rope)
        with pytest.raises(BackendUnsupported):
            module(single, cond, pair, mask, rope=rope)


def test_make_inputs_run():
    module = AtomDiTESMFold2Pair()
    ins = module.make_inputs(
        length=256, samples=2, depth=None, dtype=torch.float32, device="cpu", generator=torch.Generator().manual_seed(0)
    )
    assert ins["pair"].shape == (1, 8, 32, 192, 16) and ins["rope"][0].shape == (1, 256, 16)
    with torch.no_grad():
        assert module(**ins).shape == (2, 1, 256, 128)
