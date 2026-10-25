"""The bias-only token DiT block on the portable kernels (every card but H100 / B200): its parts (AdaptiveLayerNorm, ConditionedTransition,
the pair LayerNorm) route per card like any module, so under ``portable_on`` the block runs the portable AdaLN / transition kernels and the
batched einsum core, forward AND backward, bf16 and fp32 (TF32), one sample or several augmentations, a pair of its own per sample (psb),
the hoisted bias (``pair_bias_all``), key masks. Against the block's fp64 PyTorch statements; the bf16 engine error is held to the bf16
PyTorch block's own (x 1.5, floor 3e-3). On an H100 the portable output is also compared with the CUDA block's.
"""
from __future__ import annotations

import copy
import warnings

import pytest
import torch
from portable_helpers import portable_on, rel, report

from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock, pair_bias_all
from miniworld_engine.modules.dispatch import KernelBackend, KernelRefusedWarning, UnrecordedCardWarning
from miniworld_engine.modules.exceptions import ImplementationType as I

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]
DEV, BF, F32 = "cuda", torch.bfloat16, torch.float32

# (d_single, d_cond, d_pair, n_head, d_head): the AF3 token widths and layouts the CUDA path serves, and narrower / wider ones it does not
WIDTHS = [
    (768, 384, 128, 16, None),
    (768, 384, 128, 24, None),
    (768, 384, 128, 12, None),
    (768, 384, 128, 16, 64),
    (256, 128, 64, 8, None),
    (384, 192, 96, 6, None),
    (1024, 384, 128, 16, None),
]


@pytest.fixture(autouse=True)
def portable():
    with portable_on():
        yield


def _randomize(m):
    with torch.no_grad():
        for name, p in m.named_parameters():
            p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else
                    1 + 0.1 * torch.randn_like(p) if name.endswith("weight") else 0.1 * torch.randn_like(p))
    return m


def _block(width, implementation, seed=3):
    d_single, d_cond, d_pair, n_head, d_head = width
    torch.manual_seed(seed)
    return BiasOnlyDiTBlock(d_single, d_cond, d_pair, n_head, d_head=d_head, implementation=implementation).to(DEV)


def _inputs(width, A, B, L, dtype, seed=1):
    d_single, d_cond, d_pair, *_ = width
    g = torch.Generator(device=DEV).manual_seed(seed)
    single = torch.randn(A, B, L, d_single, device=DEV, generator=g, dtype=F32)
    cond = torch.randn(A, B, L, d_cond, device=DEV, generator=g, dtype=F32)
    pair = torch.randn(B, L, L, d_pair, device=DEV, generator=g, dtype=F32)
    mask = torch.rand(B, L, device=DEV, generator=g) > 0.1
    dy = torch.randn(A, B, L, d_single, device=DEV, generator=g, dtype=F32)
    return single, cond, pair, mask, dy


def _run(block, args, dy, dtype, training, use_hoist=False):
    single, cond, pair, mask = (a.to(dtype) if a is not None and a.dtype.is_floating_point else a for a in args)
    leaves = [t.requires_grad_(training) for t in (single, cond, pair)]
    single, cond, pair = leaves
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        if use_hoist:
            bias = pair_bias_all([block], pair)[0]
            out = block(single, cond, None, mask, bias=bias)
        else:
            out = block(single, cond, pair, mask)
        if not training:
            return out, ()
        params = [p for p in block.parameters()]
        grads = torch.autograd.grad(out, [*leaves, *params], dy.to(dtype), allow_unused=True)
    return out, tuple(g for g in grads)


def _compare(width, A, B, L, dtype, training, use_hoist=False, bound_floor=3e-3):
    base = _randomize(_block(width, I.PYTORCH))               # the fp32 weights every candidate is loaded from
    gold = copy.deepcopy(base).double()
    ref = _block(width, I.PYTORCH)
    ref.load_state_dict(base.state_dict())
    ref = ref.to(dtype)
    eng = _block(width, I.MINIWORLD)
    eng.load_state_dict(base.state_dict())
    eng = eng.to(dtype)
    for part in (eng.attention.ada_ln_in, eng.transition, eng.transition.ada_ln_in):
        assert part._backend == KernelBackend.PORTABLE, "the part did not resolve to the portable kernels under portable_on"
    single, cond, pair, mask, dy = _inputs(width, A, B, L, dtype)
    args = (single, cond, pair, mask)
    out_gold, g_gold = _run(gold, tuple(a.double() if a is not None and a.dtype.is_floating_point else a for a in args), dy, torch.float64, training, use_hoist)
    out_ref, g_ref = _run(ref, args, dy, dtype, training, use_hoist)
    with warnings.catch_warnings():
        for category in (UnrecordedCardWarning, KernelRefusedWarning):   # neither: the portable kernels ran
            warnings.simplefilter("error", category)
        out, g_eng = _run(eng, args, dy, dtype, training, use_hoist)
    assert torch.isfinite(out).all()
    report(f"bias_only_dit w{width[0]}h{width[3]} {str(dtype)[6:]} {'train' if training else 'infer'} A{A}B{B}L{L}{' hoist' if use_hoist else ''}",
           out, out_ref, out_gold, g_eng, g_ref, g_gold)
    # fp32 engine = TF32 tensor cores (ConditionedTransition's contract); the fp32 PyTorch block is full fp32
    floor = bound_floor if dtype is BF else 6e-3
    factor = 1.5 if dtype is BF else 1.0
    assert rel(out, out_gold) <= max(factor * rel(out_ref, out_gold), floor), (rel(out, out_gold), rel(out_ref, out_gold))
    for i, (a, r, c) in enumerate(zip(g_eng, g_gold, g_ref, strict=True)):
        if r is None:
            continue
        assert rel(a, r) <= max(factor * rel(c, r), 2 * floor), (i, rel(a, r), rel(c, r))
        assert a.dtype == ([dtype] * 3 + [p.dtype for p in eng.parameters()])[i], "a gradient comes back in its leaf's dtype"


@pytest.mark.parametrize("training", [False, True], ids=["infer", "train"])
@pytest.mark.parametrize("dtype", [BF, F32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("width", WIDTHS, ids=[f"{w[0]}h{w[3]}d{w[4]}" for w in WIDTHS])
def test_block_matches_pytorch(width, dtype, training):
    _compare(width, A=3, B=1, L=192, dtype=dtype, training=training)


@pytest.mark.parametrize("training", [False, True], ids=["infer", "train"])
@pytest.mark.parametrize("dtype", [BF, F32], ids=["bf16", "fp32"])
def test_one_sample_and_ragged_length(dtype, training):
    _compare(WIDTHS[0], A=1, B=1, L=200, dtype=dtype, training=training)


@pytest.mark.parametrize("training", [False, True], ids=["infer", "train"])
@pytest.mark.parametrize("dtype", [BF, F32], ids=["bf16", "fp32"])
def test_a_pair_of_its_own_per_sample(dtype, training):
    """psb: the samples folded into the batch axis, ``single`` [1, B, L, d] with ``pair`` [B, L, L, d_pair], every sample its own bias."""
    _compare(WIDTHS[0], A=1, B=3, L=128, dtype=dtype, training=training)


@pytest.mark.parametrize("training", [False, True], ids=["infer", "train"])
@pytest.mark.parametrize("dtype", [BF, F32], ids=["bf16", "fp32"])
def test_hoisted_pair_bias(dtype, training):
    _compare(WIDTHS[4], A=2, B=1, L=160, dtype=dtype, training=training, use_hoist=True)
    _compare(WIDTHS[0], A=1, B=2, L=128, dtype=dtype, training=training, use_hoist=True)


def test_matches_the_cuda_block_on_h100():
    """On an H100 the CUDA block and the portable one answer the same call: both within the bf16 PyTorch statements' own distance of fp32."""
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("H100 only")
    width = WIDTHS[0]
    gold = _randomize(_block(width, I.PYTORCH))
    single, cond, pair, mask, _ = _inputs(width, 3, 1, 256, BF)
    ref16 = _block(width, I.PYTORCH)
    ref16.load_state_dict(gold.state_dict())
    ref16 = ref16.to(BF)
    with torch.no_grad():
        want = gold(single, cond, pair, mask)
        r16 = ref16(single.to(BF), cond.to(BF), pair.to(BF), mask)
    cuda_block = _block(width, I.MINIWORLD)
    with portable_on(False):
        cuda_block = _block(width, I.MINIWORLD)
        cuda_block.load_state_dict(gold.state_dict())
        cuda_block = cuda_block.to(BF)
        with torch.no_grad():
            c = cuda_block(single.to(BF), cond.to(BF), pair.to(BF), mask)
    portable_block = _block(width, I.MINIWORLD)
    portable_block.load_state_dict(gold.state_dict())
    portable_block = portable_block.to(BF)
    with torch.no_grad():
        p = portable_block(single.to(BF), cond.to(BF), pair.to(BF), mask)
    bound = max(1.5 * rel(r16, want), 3e-3)
    assert rel(c, want) <= bound
    assert rel(p, want) <= bound
    assert rel(p, c) <= 2 * bound
