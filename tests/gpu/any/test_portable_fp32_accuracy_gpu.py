"""The portable (Triton) fp32 kernels on TF32 tensor cores are as accurate as the PyTorch TF32 statements.

Triton's ``tl.dot(..., input_precision="tf32")`` TRUNCATES fp32 operands to the 10-bit TF32 mantissa where cuBLAS rounds to nearest, which
makes a kernel that feeds raw fp32 tiles to the MMA 3-4x less accurate than ``torch.matmul`` under ``allow_tf32``. The kernels round the
operand tiles to the TF32 grid first (``kernels/_tf32.py``). This file holds the module-level yardstick of the repo: the engine's error
against an fp64 run of the same inputs and cotangent is at most 1.5x the PyTorch statements' own error, for the output and for every
gradient (``tests/tf32_accuracy.py`` measures; ``PARITY_REPORT=1 TF32ACC_VERBOSE=1`` prints every ratio).
"""
from __future__ import annotations

import pytest
import torch
from portable_helpers import portable_on
from tf32_accuracy import YARDSTICK, measure, randomise, report, worst

from miniworld_engine.modules import (
    AdaptiveLayerNorm,
    AttentionPairBias,
    AugmentedAttentionPairBias,
    ConditionedTransition,
    MSAPairWeightedAveraging,
    OuterProductMean,
    Transition,
)
from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.dit_variant import (
    BOLTZ2,
    BOLTZ2_ATOM,
    AtomDiTVariantStack,
    DiTVariantStack,
)
from miniworld_engine.modules.local_dit import LocalDiTBlock
from miniworld_engine.modules.triangle_attention import TriangleAttention
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]
CUDA = "cuda"


def _g(seed=1):
    return torch.Generator(device=CUDA).manual_seed(seed)


def _r(*shape, g):
    return torch.randn(*shape, device=CUDA, generator=g)


def _mask(b, n, real):
    m = torch.zeros(b, n, dtype=torch.bool, device=CUDA)
    m[:, :real] = True
    return m


def case_transition(d, n, lead):
    g = _g()
    return (lambda impl: randomise(Transition(d, n=n, implementation=impl)), (_r(*lead, d, g=g),), None, None)


def case_ct(d, dc, length, a):
    g = _g()
    return (lambda impl: randomise(ConditionedTransition(d_hidden=d, d_cond=dc, n=2, implementation=impl)),
            (_r(a, 1, length, d, g=g), _r(a, 1, length, dc, g=g)), None, None)


def case_adaln(d, dc, length, a):
    g = _g()
    return (lambda impl: randomise(AdaptiveLayerNorm(d, dc, implementation=impl)),
            (_r(a, 1, length, d, g=g), _r(a, 1, length, dc, g=g)), None, None)


def case_aat(length, a, real):
    g = _g()
    mask = _mask(1, length, real)
    return (lambda impl: randomise(AugmentedAttentionPairBias(768, 384, 128, 16, implementation=impl)),
            (_r(a, 1, length, 768, g=g), _r(a, 1, length, 384, g=g), _r(1, length, length, 128, g=g), mask),
            None, (slice(None), slice(None), slice(0, real)))


def case_dit(length, a, real):
    g = _g()
    mask = _mask(1, length, real)
    return (lambda impl: randomise(DiTBlock(768, 384, 128, 16, 2, implementation=impl)),
            (_r(a, 1, length, 768, g=g), _r(a, 1, length, 384, g=g), _r(1, length, length, 128, g=g), mask),
            None, (slice(None), slice(None), slice(0, real)))


def case_bodit(length, a, real):
    g = _g()
    mask = _mask(1, length, real)
    return (lambda impl: randomise(BiasOnlyDiTBlock(768, 384, 128, 16, 2, implementation=impl)),
            (_r(a, 1, length, 768, g=g), _r(a, 1, length, 384, g=g), _r(1, length, length, 128, g=g), mask),
            None, (slice(None), slice(None), slice(0, real)))


def case_local(length, a):
    g = _g()
    opts = {"cross_attention": True, "kv_norm": "parallel", "key_window": "slide"}
    mask = _mask(1, length, length - 30)
    return (lambda impl: randomise(LocalDiTBlock(128, 128, 16, 4, n=2, implementation=impl, **opts)),
            (_r(a, 1, length, 128, g=g), _r(a, 1, length, 128, g=g), _r(1, (length + 31) // 32, 32, 128, 16, g=g), mask),
            None, (slice(None), slice(None), slice(0, length - 30)))


def case_tokenstack(length, a):
    g = _g()
    o = BOLTZ2
    return (lambda impl: randomise(DiTVariantStack(3, 768, o["d_cond"], o["d_pair"], 16, 2, implementation=impl,
                                                   **{k: v for k, v in o.items() if k not in ("d_cond", "d_pair")})),
            (_r(a, length, 768, g=g), _r(length, o["d_cond"], g=g), _r(length, length, o["d_pair"], g=g)), None, None)


def case_atomstack(atoms, a):
    g = _g()
    o = BOLTZ2_ATOM
    return (lambda impl: randomise(AtomDiTVariantStack(
        3, 128, 128, 16, 2, parallel=o["parallel"], per_block_pair_norm=o["per_block_pair_norm"], pair_norm_bias=o["pair_norm_bias"],
        key_window="slide", implementation=impl, **{k: v for k, v in o.items() if k in ("identity_scale", "gating", "project_output", "kq_norm", "up_gate")})),
            (_r(a, atoms, 128, g=g), _r(atoms, 128, g=g), _r(atoms // 32, 32, 128, 16, g=g)), None, None)


def case_trimul(length, outgoing):
    g = _g()
    mask = _mask(1, length, length - 7)
    return (lambda impl: randomise(TriangleMultiplication(128, outgoing=outgoing, implementation=impl, p_drop=0.0)),
            (_r(1, length, length, 128, g=g), mask), None, None)


def case_triattn(length, starting):
    g = _g()
    mask = _mask(1, length, length - 7)
    return (lambda impl: randomise(TriangleAttention(128, 4, starting=starting, implementation=impl, p_drop=0.0)),
            (_r(1, length, length, 128, g=g), mask), None, None)


def case_opm(length, msa):
    g = _g()
    return (lambda impl: randomise(OuterProductMean(64, 128, implementation=impl)),
            (_r(1, msa, length, 64, g=g), None, None, _r(1, length, length, 128, g=g)), None, None)


def case_pwa(length, msa):
    g = _g()
    return (lambda impl: randomise(MSAPairWeightedAveraging(64, 128, 8, implementation=impl)),
            (_r(1, msa, length, 64, g=g), _r(1, length, length, 128, g=g), _mask(1, length, length - 5)), None, None)


def case_apb(length):
    g = _g()
    return (lambda impl: randomise(AttentionPairBias(384, 128, 8, implementation=impl)),
            (_r(1, length, 384, g=g), _r(1, length, length, 128, g=g), _mask(1, length, length - 5)), None, None)


CASES = {
    "transition_d128": lambda: case_transition(128, 4, (1, 96, 96)),
    "transition_d256": lambda: case_transition(256, 4, (1, 64, 64)),
    "transition_d64n2": lambda: case_transition(64, 2, (1, 80, 80)),
    "ct_768": lambda: case_ct(768, 384, 128, 3),
    "ct_atom128": lambda: case_ct(128, 128, 256, 2),
    "adaln_768": lambda: case_adaln(768, 384, 128, 3),
    "adaln_128": lambda: case_adaln(128, 128, 256, 2),
    "aat_768": lambda: case_aat(128, 3, 110),
    "dit_768": lambda: case_dit(128, 3, 110),
    "bodit_768": lambda: case_bodit(128, 3, 110),
    "localdit": lambda: case_local(512, 2),
    "tokenstack_boltz2": lambda: case_tokenstack(128, 2),
    "atomstack_boltz2": lambda: case_atomstack(512, 2),
    "trimul_out": lambda: case_trimul(96, True),
    "trimul_in": lambda: case_trimul(96, False),
    "triattn_start": lambda: case_triattn(96, True),
    "triattn_end": lambda: case_triattn(96, False),
    "opm": lambda: case_opm(96, 32),
    "pwa": lambda: case_pwa(96, 32),
    "apb": lambda: case_apb(128),
}
INFER = {"transition_d128", "ct_768", "adaln_768", "aat_768", "dit_768", "bodit_768", "trimul_out", "triattn_start", "opm", "pwa", "apb"}


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("train", [True, False], ids=["train", "infer"])
def test_portable_fp32_is_within_1p5x_of_the_pytorch_tf32_error(name, train):
    if not train and name not in INFER:
        pytest.skip("inference measured on the families with a separate inference kernel")
    make, args, call, live = CASES[name]()
    with portable_on():
        rows = measure(make, args, call, train=train, live=live)
    report(f"{name} {'train' if train else 'infer'}", rows)
    bad = [f"{r.name}: {r.engine:.2e} vs {r.pytorch:.2e} ({r.ratio:.2f}x)" for r in rows if r.ratio > YARDSTICK]
    assert not bad, f"worst {worst(rows).ratio:.2f}x\n  " + "\n  ".join(bad)


RAGGED = {
    "transition_d192n2_ragged": lambda: case_transition(192, 2, (1, 70, 70)),
    "ct_d192": lambda: case_ct(192, 96, 100, 3),
    "adaln_d192": lambda: case_adaln(192, 96, 100, 3),
}


@pytest.mark.parametrize("name", sorted(RAGGED))
@pytest.mark.parametrize("train", [True, False], ids=["train", "infer"])
def test_ragged_widths_stay_within_1p5x(name, train):
    """Widths that are not a multiple of the K tiles (192, 96): a partial last K tile must not change the rounding."""
    make, args, call, live = RAGGED[name]()
    with portable_on():
        rows = measure(make, args, call, train=train, live=live)
    report(f"{name} {'train' if train else 'infer'}", rows)
    bad = [f"{r.name}: {r.engine:.2e} vs {r.pytorch:.2e} ({r.ratio:.2f}x)" for r in rows if r.ratio > YARDSTICK]
    assert not bad, f"worst {worst(rows).ratio:.2f}x\n  " + "\n  ".join(bad)
