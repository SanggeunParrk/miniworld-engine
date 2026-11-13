"""The ``"triton"`` backends restored from engine 2.4 (any card): the parts of the bias-only token DiT (AdaptiveLayerNorm,
ConditionedTransition, BiasOnlyAttention), AdaptiveRMSNorm (no requirement row: tested here only), the fused SWA atom block
(AtomDiTMiniWorld) and the bias-only one-directional TriangleAttention, against the module's own PyTorch statements in float64 --
forward and every gradient, bf16 and fp32 (TF32 allowed, as the measurers run). The block classes with requirement rows are also covered by
``test_accuracy_smoke.py`` through the measurer.
"""

from __future__ import annotations

import copy
from typing import TypeGuard, TypeVar

import pytest
import torch

from miniworld_engine.modules.atom_dit import AtomDiTMiniWorld
from miniworld_engine.modules.attention import BiasOnlyAttention
from miniworld_engine.modules.embedder import InputFeatureEmbedder
from miniworld_engine.modules.norm import AdaptiveLayerNorm, AdaptiveRMSNorm
from miniworld_engine.modules.primitives import BackendModule
from miniworld_engine.modules.token_dit import TokenDiTMiniWorldBiasOnly
from miniworld_engine.modules.transition import ConditionedTransition
from miniworld_engine.modules.triangle import TriangleAttention

pytestmark = [pytest.mark.gpu]

#: relative max error to the float64 statements: bf16 inputs / fp32 with TF32 products
TOL = {torch.bfloat16: 4e-2, torch.float32: 5e-3}


M = TypeVar("M", bound=BackendModule)


def _randomize(module: M, seed: int = 0) -> M:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            r = torch.randn(p.shape, generator=g)
            # norms' weights / biases near 1, matrices ~ 1 / sqrt(fan in)
            p.copy_(1.0 + 0.1 * r if p.ndim == 1 else r * p.shape[-1] ** -0.5)
    return module


def _float_input(v) -> TypeGuard[torch.Tensor]:
    return isinstance(v, torch.Tensor) and v.is_floating_point()


def _rel(x: torch.Tensor, gold: torch.Tensor) -> float:
    return float((x.double() - gold).abs().max() / gold.abs().max().clamp_min(1e-30))


def _check(module: BackendModule, inputs: dict, dtype: torch.dtype, *, grad: bool = True, tol: float | None = None) -> None:
    """``module`` on its "triton" backend in ``dtype`` (fp32 parameters, bf16 activations under bf16 autocast for bf16) vs its
    "pytorch" statements in float64: the output and, with ``grad``, every floating input's and parameter's gradient."""
    tol = TOL[dtype] if tol is None else tol
    ref = copy.deepcopy(module).double().set_backend("pytorch")

    def wide(v):  # structural inputs (a tuple of RoPE tables and masks) in float64 for the reference
        return tuple(t.double() if _float_input(t) else t for t in v) if isinstance(v, tuple) else v

    ref_in = {k: (v.detach().double().requires_grad_(grad) if _float_input(v) else wide(v)) for k, v in inputs.items()}
    out_ref = ref(**ref_in)
    tri = module.cuda().set_backend("triton")
    tri_in = {k: (v.detach().to(dtype).requires_grad_(grad) if _float_input(v) else v) for k, v in inputs.items()}
    torch.backends.cuda.matmul.allow_tf32 = True
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        out = tri(**tri_in)
    assert out.dtype == dtype
    assert _rel(out, out_ref.detach()) < tol, f"output {_rel(out, out_ref.detach()):.3g}"
    if not grad:
        return
    g = torch.randn(out.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64).cuda()
    out_ref.backward(g)
    out.backward(g.to(out.dtype))
    for k, v in tri_in.items():
        r = ref_in[k]
        if _float_input(v) and _float_input(r):
            assert v.grad is not None and r.grad is not None, k
            assert _rel(v.grad, r.grad) < tol, f"d{k} {_rel(v.grad, r.grad):.3g}"
    for (name, p), (_, pr) in zip(tri.named_parameters(), ref.named_parameters(), strict=True):
        if pr.grad is not None and pr.grad.abs().max() > 0:
            assert p.grad is not None, name
            assert _rel(p.grad, pr.grad) < tol, f"d{name} {_rel(p.grad, pr.grad):.3g}"


def _rand(*shape, dtype=torch.float32, seed=2) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=dtype).cuda()


DTYPES = [torch.bfloat16, torch.float32]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize("d", [128, 768])
def test_adaptive_layernorm(dtype, grad, d):
    m = _randomize(AdaptiveLayerNorm(d, 384 if d == 768 else 128).cuda())
    _check(m, {"x": _rand(2, 1, 96, d), "cond": _rand(2, 1, 96, 384 if d == 768 else 128, seed=3)}, dtype, grad=grad)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("grad", [False, True])
def test_adaptive_rmsnorm(dtype, grad):
    m = _randomize(AdaptiveRMSNorm(128, 128).cuda())
    _check(m, {"x": _rand(3, 200, 128), "cond": _rand(3, 200, 128, seed=3)}, dtype, grad=grad)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize("d", [128, 768])  # the fused b2b inference kernel / the composed pair
def test_conditioned_transition(dtype, grad, d):
    dc = 384 if d == 768 else 128
    m = _randomize(ConditionedTransition(d, dc, n=2).cuda())
    _check(m, {"x": _rand(2, 1, 128, d), "cond": _rand(2, 1, 128, dc, seed=3)}, dtype, grad=grad)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize("per_sample", [False, True])  # one pair for every sample / a pair per sample (the training core's two layouts)
def test_bias_only_attention(dtype, grad, per_sample):
    m = _randomize(BiasOnlyAttention(768, 384, 128, 16).cuda())
    a, b, n = (1, 3, 64) if per_sample else (3, 1, 64)
    mask = torch.ones(b, n, dtype=torch.bool, device="cuda")
    mask[:, -5:] = False
    _check(
        m,
        {"single": _rand(a, b, n, 768), "cond": _rand(a, b, n, 384, seed=3), "pair": _rand(b, n, n, 128, seed=4), "mask": mask},
        dtype,
        grad=grad,
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_bias_only_token_dit_block(dtype):
    m = _randomize(TokenDiTMiniWorldBiasOnly().cuda())
    inputs = m.make_inputs(
        length=128, samples=2, depth=None, dtype=torch.float32, device="cuda", generator=torch.Generator("cuda").manual_seed(0)
    )
    _check(m, inputs, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize("half_window", [64, 500000, -1])  # the window, and global attention (the window kernels, every key in reach)
def test_swa_atom_block(dtype, grad, half_window):
    m = _randomize(AtomDiTMiniWorld(half_window=half_window).cuda())
    inputs = m.make_inputs(
        length=320 if half_window != 64 else 256,
        samples=2,
        depth=None,
        dtype=torch.float32,
        device="cuda",
        generator=torch.Generator("cuda").manual_seed(0),
    )
    _check(m, inputs, dtype, grad=grad, tol=6e-2 if dtype == torch.bfloat16 else 2e-2)  # the window attention runs on bf16 operands


@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize("grad", [False, True])
def test_bias_only_triangle_attention(starting, grad):
    m = _randomize(TriangleAttention(128, 4, starting=starting, use_self_attention=False, p_drop=0.0, master_weights=True).cuda())
    mask = torch.ones(1, 96, dtype=torch.bool, device="cuda")
    mask[:, -7:] = False
    _check(m, {"pair": _rand(1, 96, 96, 128), "mask": mask}, torch.bfloat16, grad=grad)


@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize("grad", [False, True])
def test_bias_only_triangle_attention_cuda(starting, grad):
    """The ``"cuda"`` backend of the one-directional bias-only form (BidirectionalTriangleAttention's H100 / A100 kernels, one direction)."""
    cap = torch.cuda.get_device_capability()
    if cap not in ((9, 0), (8, 0)):
        pytest.skip("the bias-only cuda path is the H100 / A100 one")
    m = _randomize(TriangleAttention(128, 4, starting=starting, use_self_attention=False, p_drop=0.0, master_weights=True).cuda())
    mask = torch.ones(1, 96, dtype=torch.bool, device="cuda")
    mask[:, -7:] = False
    pair = _rand(1, 96, 96, 128)
    ref = copy.deepcopy(m).set_backend("triton")
    m.set_backend("cuda")
    p1, p2 = pair.bfloat16().requires_grad_(grad), pair.bfloat16().requires_grad_(grad)
    with torch.set_grad_enabled(grad):
        out, out_t = m(p1, mask), ref(p2, mask)
    assert _rel(out, out_t.double()) < 2e-2
    if grad:
        g = _rand(1, 96, 96, 128, seed=5).bfloat16()
        out.backward(g)
        out_t.backward(g)
        assert p1.grad is not None and p2.grad is not None
        assert _rel(p1.grad, p2.grad.double()) < 4e-2
        for (name, a), (_, b) in zip(m.named_parameters(), ref.named_parameters(), strict=True):
            assert a.grad is not None and b.grad is not None, name
            assert _rel(a.grad, b.grad.double()) < 4e-2, name


def test_refusals():
    t = ConditionedTransition(128, 128).cuda().set_backend("triton")
    assert t.unsupported(
        torch.zeros(1, 4, 128, device="cuda", dtype=torch.float16), torch.zeros(1, 4, 128, device="cuda", dtype=torch.float16)
    )
    swa = AtomDiTMiniWorld(half_window=32).cuda().set_backend("triton")
    inputs = swa.make_inputs(
        length=128, samples=1, depth=None, dtype=torch.float32, device="cuda", generator=torch.Generator("cuda").manual_seed(0)
    )
    assert "half_window 32" in (swa.unsupported(**inputs) or "")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize(
    ("features", "window", "token_pool"), [("esmfold2", 1000000, "linear"), ("esmfold2", 128, "relu"), ("legacy", 1000000, "relu")]
)
def test_input_feature_embedder(dtype, grad, features, window, token_pool):
    """The composition on "triton" (its SWAAtomTransformer's Triton block stages, window or global; its norms' Triton kernels) vs its
    PyTorch statements in float64. The atom positions stay fp32 (the RoPE tables); bf16 runs under bf16 autocast (the "master" dtype)."""
    m = _randomize(InputFeatureEmbedder(128, 384, 4, 3, swa_window_size=window, features=features, token_pool=token_pool).cuda(), seed=3)
    with torch.no_grad():  # adaLN-Zero scale: keep the residual stream bounded through three blocks
        for name, p in m.named_parameters():
            if "adaln_modulation" in name:
                p.mul_(0.1)
    inputs = m.make_inputs(
        length=320, samples=2, depth=None, dtype=torch.float32, device="cuda", generator=torch.Generator("cuda").manual_seed(0)
    )
    ref = copy.deepcopy(m).double().set_backend("pytorch")
    pos_ref = inputs["pos"].double()
    out_ref = ref(**{**inputs, "pos": pos_ref})
    tri = m.set_backend("triton")
    assert tri.unsupported(**inputs) is None, tri.unsupported(**inputs)
    torch.backends.cuda.matmul.allow_tf32 = True
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        out = tri(**inputs)
    tol = 8e-2 if dtype == torch.bfloat16 else 2e-2  # three blocks of window attention on bf16 operands
    assert _rel(out, out_ref.detach()) < tol, f"output {_rel(out, out_ref.detach()):.3g}"
    if not grad:
        return
    g = torch.randn(out.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64).cuda()
    out_ref.backward(g)
    out.backward(g.to(out.dtype))
    for (name, p), (_, pr) in zip(tri.named_parameters(), ref.named_parameters(), strict=True):
        if pr.grad is not None and pr.grad.abs().max() > 0:
            assert p.grad is not None, name
            assert _rel(p.grad, pr.grad) < 2 * tol, f"d{name} {_rel(p.grad, pr.grad):.3g}"
