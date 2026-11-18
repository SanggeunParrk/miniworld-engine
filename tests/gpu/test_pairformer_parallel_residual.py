"""``PairformerBlock(parallel=True)`` on ``"cuda"``: the updates' deltas are summed by the fused parallel-residual kernel (one pass, fp32
sum, one rounding) instead of the reference's bf16 subtraction / addition statements (Chai-1's trunk: 19-22 % of its forward at 512 to 2048
tokens went into those adds, audit 2026-11-18).

The fused result is judged as the engine judges a backend: its error against an fp64 run of the reference statements is at most
``max(err of the unfused cuda block, 8e-3)`` (``measure/accuracy.py``'s tolerance), forward and gradients; the kernel itself is bitwise the
reference's arithmetic (``test_requests_api_gpu.py``)."""

import pytest
import torch

import miniworld_engine.modules.pairformer as pf
from miniworld_engine.measure import config as C
from miniworld_engine.modules.pairformer import PairformerBlock, PairformerConfig
from miniworld_engine.modules.primitives import key_mask

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device")]

L = 128


def _config(single: bool) -> PairformerConfig:
    """Chai-1's trunk block (d_pair 256, 4 heads x 64, transition n = 2, the single track with its gate bias) or, without ``single``, its
    pair track alone."""
    return PairformerConfig(
        d_pair=256,
        d_hidden_tri_multi=256,
        d_hidden_tri_attention=64,
        n_head_tri_attention=4,
        p_drop=0.0,
        model="chai1",
        d_single=384 if single else 0,
        n_head_single=16,
        single_gate_bias=1.0,
        pair_transition_n=2,
        single_transition_n=2,
        parallel=True,
    )


def _block(single: bool) -> PairformerBlock:
    torch.manual_seed(0)
    m = PairformerBlock(_config(single)).cuda().float()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(0))
    return C.cast(m, "master")  # pyright: ignore[reportReturnType]


def _inputs(single: bool, requires_grad: bool = False):
    g = torch.Generator(device="cuda").manual_seed(1)
    pair = torch.randn(1, L, L, 256, device="cuda", generator=g).to(torch.bfloat16).requires_grad_(requires_grad)
    mask = key_mask(1, L, device="cuda")
    s = torch.randn(1, L, 384, device="cuda", generator=g).to(torch.bfloat16).requires_grad_(requires_grad) if single else None
    return pair, mask, s


def _err(x: torch.Tensor, gold: torch.Tensor) -> float:
    d = (x.double() - gold.double()).abs().max().item()
    s = gold.double().abs().max().item()
    return d / s if s > 0 else d


def _run(m: PairformerBlock, pair, mask, s):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(pair, mask, s) if s is not None else m(pair, mask)
    return out if isinstance(out, tuple) else (out,)


def _gold(m: PairformerBlock, pair, mask, s):
    ref = PairformerBlock(m.config).cuda().double()
    ref.load_state_dict({k: v.double() for k, v in m.state_dict().items()})
    ref.set_backend("pytorch").eval()
    with torch.no_grad():
        out = ref(pair.double(), mask, s.double() if s is not None else None) if s is not None else ref(pair.double(), mask)
    return out if isinstance(out, tuple) else (out,)


@pytest.mark.parametrize("single", [True, False])
def test_fused_eval_in_the_accuracy_standard(single, monkeypatch):
    m = _block(single).set_backend("cuda").eval()
    pair, mask, s = _inputs(single)
    calls = []
    real = pf.fused_parallel_residual
    monkeypatch.setattr(pf, "fused_parallel_residual", lambda x, b: calls.append(len(b)) or real(x, b))
    with torch.no_grad():
        fused = _run(m, pair, mask, s)
    assert calls == ([5, 2] if single else [5]), calls  # five pair updates (two trimuls, two triattns, transition), two single ones
    monkeypatch.setattr(pf, "fused_parallel_residual", lambda x, b: None)
    with torch.no_grad():
        plain = _run(m, pair, mask, s)
    gold = _gold(m, pair, mask, s)
    for f, p, g in zip(fused, plain, gold, strict=True):
        assert f.dtype == p.dtype and f.shape == p.shape
        ef, ep = _err(f, g), _err(p, g)
        print(f"fused err {ef:.3e}  unfused err {ep:.3e}  fused-vs-unfused max rel {_err(f, p):.3e}")
        assert ef <= max(ep, 8e-3)


def test_fused_train_gradients(monkeypatch):
    m = _block(True).set_backend("cuda").train()
    grads = []
    real = pf.fused_parallel_residual
    for fuse in (True, False):
        monkeypatch.setattr(pf, "fused_parallel_residual", real if fuse else (lambda x, b: None))
        m.zero_grad(set_to_none=True)
        pair, mask, s = _inputs(True, requires_grad=True)
        out = _run(m, pair, mask, s)
        out_p, out_s = out[0], out[1]
        g = torch.Generator(device="cuda").manual_seed(2)
        (out_p.float() * torch.randn(out_p.shape, device="cuda", generator=g)).sum().add_(
            (out_s.float() * torch.randn(out_s.shape, device="cuda", generator=g)).sum()
        ).backward()
        named = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
        grads.append((out_p.detach(), out_s.detach(), pair.grad, s.grad, named))  # pyright: ignore[reportOptionalMemberAccess]
    (fp, fs, fgp, fgs, fn), (pp, ps, pgp, pgs, pn) = grads
    assert fn.keys() == pn.keys() and len(fn) > 20
    assert _err(fp, pp) < 1e-2 and _err(fs, ps) < 1e-2
    assert _err(fgp, pgp) < 2e-2 and _err(fgs, pgs) < 2e-2
    worst = max(_err(fn[k], pn[k]) for k in fn)
    print(f"worst parameter-gradient rel diff fused vs unfused: {worst:.3e}")
    assert worst < 5e-2


def test_other_backends_keep_the_reference_statements(monkeypatch):
    """``pytorch`` never calls the kernel hook (the reference statements)."""
    monkeypatch.setattr(pf, "fused_parallel_residual", lambda x, b: pytest.fail("the fused kernel must not run here"))
    m = _block(False).eval()
    pair, mask, _ = _inputs(False)
    with torch.no_grad():
        _run(m.set_backend("pytorch"), pair, mask, None)


def test_per_sub_module_backends_fuse_too(monkeypatch):
    """FoldForge picks the backend per op: the block stays on ``"pytorch"``, its sub-modules run ``cuda``; the fused sum runs as for a block set
    to ``cuda`` (bitwise the same output)."""
    m = _block(True).set_backend("cuda").eval()
    pair, mask, s = _inputs(True)
    with torch.no_grad():
        want = _run(m, pair, mask, s)
    mixed = _block(True).set_backend("pytorch").eval()
    for _, child in mixed._backend_children():
        child.set_backend("cuda")
    calls = []
    real = pf.fused_parallel_residual
    monkeypatch.setattr(pf, "fused_parallel_residual", lambda x, b: calls.append(len(b)) or real(x, b))
    with torch.no_grad():
        got = _run(mixed, pair, mask, s)
    assert calls == [5, 2], calls
    for a, b in zip(got, want, strict=True):
        assert torch.equal(a, b)
