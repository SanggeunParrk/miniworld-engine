"""B200 atom DiT block (``integrations.atom_dit``): DiTBlock at atom widths runs the sm_100a kernels for inference and
training -- with or without a [B, N] key mask, at any N (padded to a multiple of 128 inside) --, its output and every gradient
no worse against an fp64 PyTorch block than the module path's, and every call it does not serve keeps the module path."""

import pytest
import torch

from miniworld_engine.integrations import atom_dit
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100)"),
]

DS, DC, DP, NH = 128, 128, 16, 4


def _block(seed=0):
    torch.manual_seed(seed)
    ref = DiTBlock(DS, DC, DP, NH, implementation=ImplementationType.PYTORCH)
    with torch.no_grad():  # the zero inits (to_out, squeeze, to_bias) would make the block an identity
        for name, p in ref.named_parameters():
            if p.ndim > 1:
                p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5)
            else:
                p.copy_(torch.randn_like(p) * 0.1 + (1.0 if name.endswith("weight") else 0.0))
    eng = DiTBlock(DS, DC, DP, NH, implementation=ImplementationType.MINIWORLD)
    eng.load_state_dict(ref.state_dict())
    return ref.cuda().double(), eng.cuda().to(torch.bfloat16)


def _inputs(A, N, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(A, 1, N, DS, device="cuda", generator=g), torch.randn(A, 1, N, DC, device="cuda", generator=g),
            torch.randn(1, N, N, DP, device="cuda", generator=g))


def _mask(N, seed):
    """A [1, N] key mask with scattered masked atoms and a masked tail (as a padded structure has)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    m = torch.rand(1, N, device="cuda", generator=g) > 0.15
    m[:, N - N // 8:] = False
    return m


def _rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _train(m, ins, dtype, w, mask=None):
    leaves = [t.detach().clone().to(dtype).requires_grad_() for t in ins]
    out = m(*leaves, mask)
    (out.double() * w).sum().backward()
    res = {"out": out.detach(), "dsingle": leaves[0].grad, "dcond": leaves[1].grad, "dpair": leaves[2].grad}
    res.update({n: p.grad.clone() for n, p in m.named_parameters()})
    m.zero_grad(set_to_none=True)
    return res


def _spy(monkeypatch):
    calls = []
    orig_apply, orig_infer = atom_dit._AtomBlock.apply, atom_dit._infer
    monkeypatch.setattr(atom_dit._AtomBlock, "apply", lambda *a: calls.append("train") or orig_apply(*a))
    monkeypatch.setattr(atom_dit, "_infer", lambda *a: calls.append("infer") or orig_infer(*a))
    return calls


@pytest.mark.parametrize(("A", "N", "masked"), [(4, 256, False), (3, 384, False), (3, 384, True), (2, 200, True), (3, 300, False)])
def test_training_matches_the_module_path(A, N, masked, monkeypatch):
    ref, eng = _block()
    ins = _inputs(A, N)
    mask = _mask(N, 11) if masked else None
    w = torch.randn(A, 1, N, DS, device="cuda", dtype=torch.float64)
    truth = _train(ref, ins, torch.float64, w, mask)
    calls = _spy(monkeypatch)
    fused = _train(eng, ins, torch.bfloat16, w, mask)
    assert calls == ["train"], calls
    monkeypatch.setenv("MINIWORLD_ATOM_DIT_SM100", "0")
    module = _train(eng, ins, torch.bfloat16, w, mask)
    assert calls == ["train"], "the switch did not keep the module path"
    assert set(fused) == set(truth)
    worst = []
    for k in truth:
        ef, em = _rel(fused[k], truth[k]), _rel(module[k], truth[k])
        worst.append((ef / max(em, 1e-6), k, ef, em))
        assert torch.isfinite(fused[k]).all(), k
        assert fused[k].dtype == module[k].dtype, k
        assert ef < 1.5 * em + 3e-3, f"{k}: sm_100a {ef:.2e} vs module path {em:.2e}"
    print("worst ratios:", sorted(worst, reverse=True)[:3])


@pytest.mark.parametrize(("A", "N", "masked"), [(5, 128, False), (1, 384, False), (2, 1024, False), (5, 1024, True), (5, 300, True),
                                                (1, 1000, True), (2, 100, False)])
def test_inference_matches_the_module_path(A, N, masked, monkeypatch):
    ref, eng = _block(2)
    s, c, z = _inputs(A, N, 3)
    mask = _mask(N, 13) if masked else None
    with torch.no_grad():
        truth = ref(s.double(), c.double(), z.double(), mask)
        calls = _spy(monkeypatch)
        fused = eng(s.bfloat16(), c.bfloat16(), z.bfloat16(), mask)
        assert calls == ["infer"], calls
        monkeypatch.setenv("MINIWORLD_ATOM_DIT_SM100", "0")
        module = eng(s.bfloat16(), c.bfloat16(), z.bfloat16(), mask)
    assert fused.dtype == torch.bfloat16
    assert fused.shape == s.shape
    ef, em = _rel(fused, truth), _rel(module, truth)
    print(f"inference A{A} N{N} mask {masked}: sm_100a {ef:.2e} module path {em:.2e}")
    assert ef < 1.5 * em + 3e-3, f"sm_100a {ef:.2e} vs module path {em:.2e}"


@pytest.mark.parametrize(("N", "masked"), [(256, False), (300, True)])
def test_inference_is_cuda_graph_capturable(N, masked, monkeypatch):
    _, eng = _block(4)
    s, c, z = (t.bfloat16() for t in _inputs(2, N, 5))
    mask = _mask(N, 15) if masked else None
    with torch.no_grad():
        eager = eng(s, c, z, mask)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            eng(s, c, z, mask)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = eng(s, c, z, mask)
        graph.replay()
        torch.cuda.synchronize()
    torch.testing.assert_close(out, eager, atol=0, rtol=0)


def test_calls_it_does_not_serve_keep_the_module_path(monkeypatch):
    _, eng = _block(6)
    s, c, z = (t.bfloat16() for t in _inputs(2, 256, 7))
    calls = _spy(monkeypatch)
    bmask = torch.ones(1, 256, dtype=torch.bool, device="cuda")
    assert atom_dit.serves(eng, s, c, z, bmask)                                # a [1, N] key mask is served
    assert not atom_dit.serves(eng, s, c, z, torch.ones(2, 256, dtype=torch.bool, device="cuda"))   # B != 1
    assert not atom_dit.serves(eng, s, c, z, torch.ones(1, 255, dtype=torch.bool, device="cuda"))   # not N keys
    assert not atom_dit.serves(eng, s, c, z, bmask.float())                   # not a bool mask
    assert not atom_dit.serves(eng, s.float(), c.float(), z.float(), None)   # fp32 inputs
    tok = DiTBlock(implementation=ImplementationType.MINIWORLD).cuda().to(torch.bfloat16)   # token widths
    assert not atom_dit.serves(tok, torch.zeros(2, 1, 128, 768, device="cuda", dtype=torch.bfloat16),
                               torch.zeros(2, 1, 128, 384, device="cuda", dtype=torch.bfloat16),
                               torch.zeros(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16), None)
    assert calls == [], calls
