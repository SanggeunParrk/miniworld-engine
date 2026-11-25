"""B200 (sm_100a) TriangleAttention family on the ``"cuda"`` backend (``kernels/integrations/triattn_b200``; the bias-only forms on
``bidir_triattn_h100`` with the sm_100a build of its glue, ``b200_bidir``).

- Every variant class routes to the B200 kernels at its requirement geometry (heads of 16 / 32 / 64 channels, d_pair 64 .. 512) and
  matches the fp32 PyTorch statements, inference and training (output, dx and every parameter gradient), starting and ending node.
- A ragged length (not a multiple of 128) and a batch of samples, each with its own key mask.
- Deterministic forward (two calls, the same bits) and a CUDA-graph replay equal to the eager call.
"""

import copy

import pytest
import torch

from foldjet_engine.measure.config import randomize
from foldjet_engine.modules.triangle import (
    BidirectionalTriangleAttention,
    TriangleAttention,
    TriangleAttentionAF3,
    TriangleAttentionChai1,
    TriangleAttentionChai1Confidence,
    TriangleAttentionRF3,
)

pytestmark = [pytest.mark.gpu]


def _needs_b200() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("needs a B200 (sm_100)")


def _module(cls, d: int, h: int, kw: dict, starting: bool, train: bool = False):
    torch.manual_seed(0)
    m = cls(d, h, starting=starting, master_weights=True, backend="cuda", **kw).cuda()
    m.train(train)
    if train:
        m.p_drop = 0.0  # numerics, not the random mask
    with torch.no_grad():
        randomize(m, torch.Generator(device="cuda").manual_seed(1))
    return m


def _mask(B: int, L: int, g: torch.Generator) -> torch.Tensor:
    mask = torch.rand(B, L, device="cuda", generator=g) > 0.15
    mask[:, 0] = True
    if B > 1:
        mask[1, L - 9 :] = False
    return mask


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30))


#: (class, d_pair, n_head, extra kwargs): the requirement geometries of every class (heads of 16 / 32 / 64 channels)
CASES = [
    (TriangleAttention, 128, 4, {}),
    (TriangleAttention, 64, 2, {}),
    (TriangleAttention, 64, 4, {}),
    (TriangleAttention, 64, 4, {"d_hidden": 128}),
    (TriangleAttention, 256, 8, {}),
    (TriangleAttention, 384, 12, {}),
    (TriangleAttentionAF3, 64, 4, {}),
    (TriangleAttentionAF3, 128, 4, {}),
    (TriangleAttentionAF3, 512, 8, {}),
    (TriangleAttentionRF3, 64, 4, {"d_hidden": 256}),
    (TriangleAttentionRF3, 128, 4, {}),
    (TriangleAttentionChai1, 64, 4, {"d_hidden": 128}),
    (TriangleAttentionChai1, 256, 4, {}),
    (TriangleAttentionChai1Confidence, 256, 4, {}),
]


def _check(cls, d, h, kw, starting, train, B, L):
    m = _module(cls, d, h, kw, starting, train)
    ref = copy.deepcopy(m).set_backend("pytorch")
    ref.float()
    bf = copy.deepcopy(m).set_backend("pytorch")  # the bf16 statements: the error bar
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(B, L, L, d, device="cuda", generator=g).bfloat16()
    mask = _mask(B, L, g)
    assert m._cuda_route(x, mask)[0] == "sm100", m._cuda_route(x, mask)[1]
    dy = torch.randn(B, L, L, d, device="cuda", generator=g).bfloat16()

    def run(mod, inp):
        inp = inp.detach().requires_grad_(train)
        with torch.set_grad_enabled(train), torch.autocast("cuda", torch.bfloat16, enabled=inp.dtype == torch.bfloat16):
            y = mod(inp, mask)
        if not train:
            return [y.detach()]
        y.backward(dy.to(y.dtype))
        return [y.detach(), inp.grad, *(p.grad for p in mod.parameters())]

    got, gold, base = run(m, x), run(ref, x.float()), run(bf, x)
    for a, r, b in zip(got, gold, base, strict=True):
        err, bar = _rel(a, r), _rel(b, r)
        assert torch.isfinite(a).all()
        assert err <= 1.5 * bar + 2e-3, (err, bar)


@pytest.mark.parametrize("cls,d,h,kw", CASES, ids=lambda v: getattr(v, "__name__", str(v)))
@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize("train", [False, True])
def test_variants_match_fp32(cls, d, h, kw, starting, train) -> None:
    _needs_b200()
    _check(cls, d, h, kw, starting, train, 1, 128)


@pytest.mark.parametrize("cls,d,h,kw", [CASES[0], CASES[2], CASES[9], CASES[13]], ids=lambda v: getattr(v, "__name__", str(v)))
@pytest.mark.parametrize("train", [False, True])
def test_ragged_batched(cls, d, h, kw, train) -> None:
    _needs_b200()
    _check(cls, d, h, kw, False, train, 2, 200)


@pytest.mark.parametrize("cls,d,h,kw", [CASES[0], CASES[2], CASES[4]], ids=lambda v: getattr(v, "__name__", str(v)))
@pytest.mark.parametrize("train", [False, True])
def test_three_row_core(cls, d, h, kw, train) -> None:
    """L >= 384: the core's three-softmax-group build (one pair row a group); 400 rows = a padding row in the last 3-row task."""
    _needs_b200()
    _check(cls, d, h, kw, True, train, 1, 400)


@pytest.mark.parametrize("p", [0.25, 0.15])
@pytest.mark.parametrize("dim", [1, 2])
def test_drop_scale_is_the_torch_draw(p, dim) -> None:
    """The B200 build of the H100 Philox kernel draws the module's dropout scale bit for bit as the torch statements, eager and captured."""
    _needs_b200()
    from foldjet_engine.kernels.triangle_attention.cuda import b200_triattn
    from foldjet_engine.modules.functional import dropout_scale

    pair = torch.zeros(2, 200, 200, 64, device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(11)
    ref = dropout_scale(pair, p, dim).reshape(2, 200, 64)
    torch.manual_seed(11)
    got = b200_triattn.drop_scale(pair, p)
    assert got is not None and torch.equal(got, ref)
    assert b200_triattn.drop_scale(pair.float(), p) is None


@pytest.mark.parametrize("cls,d,h,kw", [CASES[0], CASES[4], CASES[13]], ids=lambda v: getattr(v, "__name__", str(v)))
@pytest.mark.parametrize("L", [256, 384])
def test_deterministic_and_graph(cls, d, h, kw, L) -> None:
    _needs_b200()
    m = _module(cls, d, h, kw, False).eval()
    g = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(2, L, L, d, device="cuda", generator=g).bfloat16()
    mask = _mask(2, L, g)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        a, b = m(x, mask), m(x, mask)
        assert torch.equal(a, b)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            m(x, mask)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = m(x, mask)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(y, a)


@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("starting", [True, False])
def test_bias_only_forms(train, starting) -> None:
    _needs_b200()
    m = _module(TriangleAttention, 128, 4, {"use_self_attention": False}, starting, train)
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(1, 128, 128, 128, device="cuda", generator=g).bfloat16()
    assert m._cuda_route(x, None)[0] == "bias_only"
    torch.manual_seed(0)
    bi = BidirectionalTriangleAttention(128, master_weights=True, backend="cuda").cuda().train(train)
    with torch.no_grad():
        randomize(bi, torch.Generator(device="cuda").manual_seed(1))
    assert bi._supports_cuda(x, None) is None
    for mod in (m, bi):
        ref = copy.deepcopy(mod).set_backend("pytorch").float()
        bf = copy.deepcopy(mod).set_backend("pytorch")
        with torch.set_grad_enabled(train), torch.autocast("cuda", torch.bfloat16):
            y, y_bf = mod(x, None).detach(), bf(x, None).detach()
        y_ref = ref(x.float(), None).detach()
        err, bar = _rel(y.float() - x.float(), y_ref - x.float()), _rel(y_bf.float() - x.float(), y_ref - x.float())
        assert err <= 1.5 * bar + 2e-3, (err, bar)


@pytest.mark.parametrize("L", [128, 200, 384])
def test_bias_only_native_ending_kernels(L) -> None:
    """The B200-native ending contractions (``b200_bidir.bo``): O = P V and dV = P^T dO equal the per-head batched GEMMs, dP = sum_c dO
    V^T matches fp64 and is the same bits at every call (cluster reduction in rank order), the row dot matches torch."""
    _needs_b200()
    from foldjet_engine.kernels.triangle_attention.cuda import b200_bidir

    bo = b200_bidir.bo()
    g = torch.Generator(device="cuda").manual_seed(7)
    H, dh = 4, 32
    p = torch.softmax(torch.randn(H, L, L, device="cuda", generator=g), -1).bfloat16()
    v = torch.randn(H * dh, L * L, device="cuda", generator=g).bfloat16()
    do = torch.randn(H * dh, L * L, device="cuda", generator=g).bfloat16()
    for trans, b in ((False, v), (True, do)):
        out = torch.empty_like(b)
        bo.bo_pv(p, b, dh, trans, out)
        ref = torch.cat(
            [torch.bmm((p[h].t() if trans else p[h]).expand(dh, L, L), b[h * dh : (h + 1) * dh].view(dh, L, L)) for h in range(H)]
        )
        assert torch.equal(out.view(-1), ref.view(-1))
    dp = torch.empty(H, L, L, device="cuda")
    bo.bo_dp(do, v, dh, dp)
    ref = torch.stack(
        [
            torch.einsum("dij,dkj->ik", do[h * dh : (h + 1) * dh].view(dh, L, L).double(), v[h * dh : (h + 1) * dh].view(dh, L, L).double())
            for h in range(H)
        ]
    )
    assert (dp.double() - ref).abs().max().item() <= 1e-5 * ref.abs().max().item()
    dp2 = torch.empty_like(dp)
    bo.bo_dp(do, v, dh, dp2)
    assert torch.equal(dp, dp2)
    s = bo.bo_rowdot(p, dp)
    torch.testing.assert_close(s, (p.float() * dp).sum(-1, keepdim=True), rtol=1e-5, atol=1e-4)
