"""A100 (sm_80) / RTX 5090 (sm_120a) triangle-attention TRAINING path (``kernels/triangle_attention/cuda/sm80.py`` ``forward_train`` /
``backward_train``) through the modules: ``TriangleAttention`` (default and ``model="af3"``, starting and ending node) and a
``PairformerBlock`` with the single track, ``cuda`` against ``pytorch`` on the same fp32 master weights under bf16 autocast, forward output
and every parameter gradient, with and without masked keys.

Regression: b04dc9caf (2026-11-13) added the ``key_mask`` argument to the sm_80 attention op for the inference key streaming and left the
training forward calling it without one, so every training call raised ``missing value for argument 'key_mask'`` (judge tokens of
TriangleAttention / TriangleAttentionAF3 / PairformerBlock train rows on a100_sm80 recorded that error).
"""

import copy

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _sm80_card() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((8, 0), (12, 0))


skip = pytest.mark.skipif(not _sm80_card(), reason="the sm_80 sources serve the A100 and the RTX 5090 only")


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def _pair(L: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(1, L, L, d, device="cuda", generator=g).to(torch.bfloat16)


def _mask(L: int, masked: bool) -> torch.Tensor:
    m = torch.ones(1, L, dtype=torch.bool, device="cuda")
    if masked:
        m[:, L - L // 5 :] = False  # trailing padded keys
    return m


def _train(mod, args):
    mod.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = mod(*args)
    outs = out if isinstance(out, tuple) else (out,)
    loss = sum((o.float() * torch.linspace(0.5, 1.5, o.shape[-1], device="cuda")).square().mean() for o in outs)
    loss.backward()  # pyright: ignore[reportAttributeAccessIssue]
    return [o.detach() for o in outs], {n: p.grad.detach().clone() for n, p in mod.named_parameters() if p.grad is not None}


def _compare(ref_mod, cuda_mod, args, tol: float):
    ro, rg = _train(ref_mod, args)
    co, cg = _train(cuda_mod, args)
    for a, b in zip(co, ro, strict=True):
        assert torch.isfinite(a).all()
        assert _rel(a, b) < tol, _rel(a, b)
    assert rg.keys() == cg.keys() and rg
    worst = max((_rel(cg[n], rg[n]), n) for n in rg)
    assert worst[0] < tol, worst


@skip
@pytest.mark.parametrize("model", ["miniworld", "af3"])
@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("L", [128, 256])
def test_triangle_attention_train_cuda_matches_pytorch(model, starting, masked, L):
    from foldjet_engine.modules import TriangleAttention

    torch.manual_seed(0)
    ref = TriangleAttention(d_pair=128, n_head=4, d_hidden=128, starting=starting, p_drop=0.0, model=model).cuda().train()
    ref.set_master_weights()
    ref.set_backend("pytorch")
    cud = copy.deepcopy(ref)
    cud.set_backend("cuda")
    _compare(ref, cud, (_pair(L, 128, 1), _mask(L, masked)), tol=2e-2)


@skip
@pytest.mark.parametrize("masked", [False, True])
def test_pairformer_block_train_cuda_matches_pytorch(masked):
    from foldjet_engine.modules import PairformerBlock, PairformerConfig

    torch.manual_seed(0)
    cfg = PairformerConfig(
        d_pair=128, d_hidden_tri_multi=128, d_hidden_tri_attention=32, n_head_tri_attention=4, p_drop=0.0, model="protenix", d_single=384
    )
    ref = PairformerBlock(cfg, master_weights=True, backend="pytorch").cuda().train()
    cud = copy.deepcopy(ref)
    cud.set_backend("cuda")
    L = 256
    g = torch.Generator(device="cuda").manual_seed(2)
    single = torch.randn(1, L, 384, device="cuda", generator=g).to(torch.bfloat16)
    _compare(ref, cud, (_pair(L, 128, 3), _mask(L, masked), single), tol=3e-2)


@skip
def test_pairformer_block_train_compiles_and_matches_eager():
    """torch.compile of a Protenix trunk block: the Transition refusal reads the SM count (``fused_fwd_sm80._num_sms``, a lazy module-dict cache); written
    inside the trace it raised ``AssertionError: Guard failed on the same frame it was created`` (``___dict_contains(0, ..._SMS)``)."""
    from foldjet_engine.modules import PairformerBlock, PairformerConfig

    torch.manual_seed(0)
    cfg = PairformerConfig(
        d_pair=128, d_hidden_tri_multi=128, d_hidden_tri_attention=32, n_head_tri_attention=4, p_drop=0.0, model="protenix", d_single=384
    )
    eager = PairformerBlock(cfg, master_weights=True, backend="cuda").cuda().train()
    comp = copy.deepcopy(eager)
    L = 128
    g = torch.Generator(device="cuda").manual_seed(2)
    single = torch.randn(1, L, 384, device="cuda", generator=g).to(torch.bfloat16)
    args = (_pair(L, 128, 3), _mask(L, False), single)
    eo, eg = _train(eager, args)
    from foldjet_engine.kernels.transition.cuda import fused_fwd_sm80

    fused_fwd_sm80._SMS.clear()  # the cache cold, as in a fresh process whose first call is the compiled one (the eager call above filled it)
    comp.compile()
    co, cg = _train(comp, args)
    for a, b in zip(co, eo, strict=True):
        assert _rel(a, b) < 1e-2, _rel(a, b)
    assert eg.keys() == {n.removeprefix("_orig_mod.") for n in cg} and eg
    cg = {n.removeprefix("_orig_mod."): v for n, v in cg.items()}
    worst = max((_rel(cg[n], eg[n]), n) for n in eg)
    assert worst[0] < 2e-2, worst
