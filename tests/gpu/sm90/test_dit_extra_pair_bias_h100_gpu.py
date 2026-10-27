"""``DiTBlock(extra_pair_bias=)`` on the H100 hand-CUDA paths: inference (bf16, fp32/TF32; the [L, L] term folded into the hoisted base-2 pair bias)
and training (bf16, fp32/TF32; folded into the fused block's bias GEMM output, gradient to the term itself from the attention's dbias).
Output and every gradient are held to 1.5 x the same-dtype PyTorch error against fp64 (fp32: the engine runs TF32 tensor cores, the PyTorch block
full fp32, so the bound has an absolute floor as in tests/gpu/any/test_dit_extra_pair_bias_portable_gpu.py). The routing is asserted: the CUDA
block must be the one that ran, not a silent fallback to the composed statements."""

from __future__ import annotations

import copy

import pytest
import torch
from portable_helpers import portable_on, rel, report
from portable_helpers import randomize_normalised as _randomize
from portable_helpers import run_extra_pair_bias as _run

from miniworld_engine.integrations import token_dit, token_dit_train
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType as I

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="Hopper (sm_90) required"),
]
DEV, BF, F32 = "cuda", torch.bfloat16, torch.float32
WIDTH = (768, 384, 128, 16)


def _block(implementation, seed=5):
    torch.manual_seed(seed)
    return DiTBlock(*WIDTH, implementation=implementation).to(DEV)


def _data(A, L, masked):
    g = torch.Generator(device=DEV).manual_seed(2)
    r = lambda *s: torch.randn(*s, device=DEV, generator=g)   # noqa: E731
    mask = (torch.rand(1, L, device=DEV, generator=g) > 0.1) if masked else None
    return r(A, 1, L, WIDTH[0]), r(A, 1, L, WIDTH[1]), r(1, L, L, WIDTH[2]), mask, 2 * r(L, L), r(A, 1, L, WIDTH[0])


@pytest.fixture
def routes(monkeypatch):
    """Counts the calls of the CUDA blocks (inference runner, training block)."""
    n = {"infer": 0, "train": 0}
    infer, train = token_dit.update, token_dit_train.block

    def update(*a, **k):
        n["infer"] += 1
        return infer(*a, **k)

    def block(*a, **k):
        n["train"] += 1
        return train(*a, **k)

    monkeypatch.setattr(token_dit, "update", update)
    monkeypatch.setattr(token_dit_train, "block", block)
    return n


@pytest.mark.parametrize("training", [False, True], ids=["infer", "train"])
@pytest.mark.parametrize("dtype", [BF, F32], ids=["bf16", "fp32"])
@pytest.mark.parametrize(("A", "L", "masked"), [(2, 256, False), (2, 256, True), (3, 130, True)], ids=["A2L256", "A2L256-mask", "A3L130-ragged"])
def test_extra_pair_bias_cuda_matches_pytorch(dtype, training, A, L, masked, routes):
    base = _randomize(_block(I.PYTORCH))
    gold = copy.deepcopy(base).double()
    ref = _block(I.PYTORCH)
    ref.load_state_dict(base.state_dict())
    ref = ref.to(dtype)
    eng = _block(I.MINIWORLD)
    eng.load_state_dict(base.state_dict())
    eng = eng.to(dtype)
    data = _data(A, L, masked)
    out_gold, g_gold = _run(gold, tuple(t.double() if t is not None and t.dtype.is_floating_point else t for t in data), torch.float64, training)
    out_ref, g_ref = _run(ref, data, dtype, training)
    with portable_on(False):
        out, g_eng = _run(eng, data, dtype, training)
    # inference: the native runner needs L a multiple of 128; training pads any L
    expect = "train" if training else ("infer" if L % 128 == 0 else None)
    if expect is not None:
        assert routes[expect] == 1, f"the CUDA block did not serve this call: {routes}"
    assert torch.isfinite(out).all()
    report(f"dit_xbias_cuda {str(dtype)[6:]} {'train' if training else 'infer'} A{A}L{L}m{int(masked)}", out, out_ref, out_gold, g_eng, g_ref, g_gold)
    floor = 3e-3 if dtype is BF else 6e-3      # fp32 engine = TF32 tensor cores, the fp32 PyTorch block is full fp32: the bound has a floor
    factor = 1.5 if dtype is BF else 1.0
    assert rel(out, out_gold) <= max(factor * rel(out_ref, out_gold), floor), (rel(out, out_gold), rel(out_ref, out_gold))
    for i, (a, r, c) in enumerate(zip(g_eng, g_gold, g_ref, strict=True)):
        if r is None:
            continue
        assert rel(a, r) <= max(factor * rel(c, r), 2 * floor), (i, rel(a, r), rel(c, r))
    if training:
        assert g_eng[3] is not None and g_eng[3].abs().max() > 0, "the term itself must receive a gradient"


def test_training_is_bitwise_repeatable():
    eng = _randomize(_block(I.MINIWORLD)).to(BF)
    data = _data(2, 256, True)
    with portable_on(False):
        a = _run(eng, data, BF, True)
        b = _run(eng, data, BF, True)
    assert torch.equal(a[0], b[0])
    for x, y in zip(a[1], b[1], strict=True):
        assert (x is None and y is None) or torch.equal(x, y)
