"""The fused bias-only token DiT inference path on B200 (integrations/bias_only_dit.py, kernels/bias_only_dit/cuda): against
the PyTorch block in the same bf16 regime and against fp32, the attention core and the softmax against their references, and
the caches (weight pack, hoisted attention weights) against live changes."""

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import bias_only_dit
from miniworld_engine.kernels.bias_only_dit import cuda as C
from miniworld_engine.kernels.bias_only_dit import reference as R
from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def relative(a, b):
    return float((a.detach().float() - b.detach().float()).norm() / b.detach().float().norm().clamp_min(1e-12))


def randomize(module):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return module


@pytest.fixture(autouse=True)
def policy():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Blackwell (sm_100) required")
    old = settings.configure(engine_backend="auto")
    try:
        yield
    finally:
        settings.configure(**vars(old))


LAYOUTS = [(16, 48), (24, 32), (12, 64), (16, 64)]


def blocks(seed=811, n_head=16, d_head=None):
    torch.manual_seed(seed)
    ref = randomize(BiasOnlyDiTBlock(n_head=n_head, d_head=d_head, implementation=ImplementationType.PYTORCH)).cuda().eval()
    fast = BiasOnlyDiTBlock(n_head=n_head, d_head=d_head, implementation=ImplementationType.MINIWORLD).cuda()
    fast.load_state_dict(ref.state_dict())
    ref_bf = BiasOnlyDiTBlock(n_head=n_head, d_head=d_head, implementation=ImplementationType.PYTORCH).cuda()
    ref_bf.load_state_dict(ref.state_dict())
    return ref, fast.to(torch.bfloat16).eval(), ref_bf.to(torch.bfloat16).eval()


# The served bf16 step is the three-kernel one (test_inf3_bf16_* below); this test pins MINIWORLD_BIAS_ONLY_DIT_INF3_BF16=0, the
# 12-launch step that stays selectable (and that a failed three-kernel build falls back to).
@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize("L", [128, 384, 768])
@pytest.mark.parametrize("S", [1, 3, 5])
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("masked", [False, True])
def test_block_matches_the_pytorch_block(monkeypatch, L, S, shared, masked, n_head, d_head):
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_INF3_BF16", "0")
    ref, fast, ref_bf = blocks(n_head=n_head, d_head=d_head)
    x = torch.randn(S, 1, L, 768, device="cuda")
    c = torch.randn(1, 1, L, 384, device="cuda").expand(S, 1, L, 384) if shared else torch.randn(S, 1, L, 384, device="cuda")
    p = torch.randn(1, L, L, 128, device="cuda")
    mask = (torch.rand(1, L, device="cuda") > 0.2) if masked else None
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        want = ref(x, c, p, mask)
        args = (x.bfloat16(), c.bfloat16(), p.bfloat16(), mask)
        assert bias_only_dit.serves(fast, *args)
        got = fast(*args)
        base = ref_bf(*args)
    # against fp32: within the PyTorch bf16 block's own error (measured 4.0e-3 vs its 4.5e-3)
    assert relative(got, want) <= 1.05 * relative(base, want)
    assert relative(got, base) < 0.01


@pytest.mark.parametrize("inf3_env", ["0", "1"])
def test_caches_follow_live_weights_and_pair(monkeypatch, inf3_env):
    """Eager / torch.compile and the caches (weight pack, hoisted attention weights, tables), for the 12-launch step (INF3_BF16=0) and
    the served three-kernel step."""
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_INF3_BF16", inf3_env)
    _, fast, ref_bf = blocks(seed=5)
    L = 256
    x = torch.randn(5, 1, L, 768, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(5, 1, L, 384, device="cuda", dtype=torch.bfloat16)
    p = torch.randn(1, L, L, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        got = fast(x, c, p)
        assert relative(got, ref_bf(x, c, p)) < 0.01
        compiled = torch.compile(fast, fullgraph=True, options={"triton.cudagraphs": False})
        assert relative(compiled(x, c, p), got) < 1e-5
        # a weight updated in place bumps its version: the pack misses
        fast.transition.squeeze.weight.mul_(0.8)
        ref_bf.transition.squeeze.weight.mul_(0.8)
        new = fast(x, c, p)
        assert not torch.equal(got, new)
        assert relative(new, ref_bf(x, c, p)) < 0.01
        # a pair changed in place bumps its version: the hoisted attention weights miss
        p.add_(0.5 * torch.randn_like(p))
        assert relative(fast(x, c, p), ref_bf(x, c, p)) < 0.01


def test_declines_what_it_does_not_serve():
    _, fast, _ = blocks()
    x = torch.randn(5, 1, 384, 768, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(5, 1, 384, 384, device="cuda", dtype=torch.bfloat16)
    p = torch.randn(1, 384, 384, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        assert bias_only_dit.serves(fast, x, c, p)
        assert not bias_only_dit.serves(fast, x.float(), c.float(), p.float())             # bf16 only
        assert not bias_only_dit.serves(fast, x[:, :, :200], c[:, :, :200], p[:, :200, :200])   # L % 128
        assert not bias_only_dit.serves(fast, x, c, p, torch.ones(5, 384, device="cuda", dtype=torch.bool))  # a mask per sample
    assert not bias_only_dit.serves(fast, x, c, p)                                          # autograd on


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize("L", [128, 256, 384, 512, 640, 768])
@pytest.mark.parametrize("S", [1, 2, 5, 8])
def test_core_and_softmax_match_their_references(L, S, n_head, d_head):
    torch.manual_seed(L + S)
    M, DA = S * L, n_head * d_head
    vg = torch.randn(M, 2 * DA, device="cuda", dtype=torch.bfloat16)
    bias = 2 * torch.randn(n_head * L, L, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(L, device="cuda") > 0.3
    P = torch.empty_like(bias)
    C.softmax_rows(bias, P, mask)
    assert relative(P, R.attention_weights(bias, mask)) < 1e-3
    a = torch.empty(M, DA, device="cuda", dtype=torch.bfloat16)
    C.PvGateCore(torch.cuda.current_device(), nh=n_head, dh=d_head)(vg[:, :DA], P, a, S, g=vg[:, DA:])
    assert relative(a, R.pv_gate(vg, P, S)) < 1e-3


def test_fully_masked_rows_are_uniform():
    bias = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16)
    P = torch.empty_like(bias)
    C.softmax_rows(bias, P, torch.zeros(256, device="cuda", dtype=torch.bool))
    assert torch.allclose(P.float(), torch.full_like(P.float(), 1 / 256), rtol=1e-2)


# ------------------------------------------------------------------------------------------------------------ bf16 three-kernel step
# the served bf16 step (MINIWORLD_BIAS_ONLY_DIT_INF3_BF16=0 turns it off): per block bo_front_bf16 -> pv_gate_inf -DPDL_INF ->
# bo_tail_bf16 (or the pair tail bo_tail2_bf16 where it takes fewer rounds), behind the hoisted pre-sigmoided bf16 tables
TAIL_ROWS = ("adaln_in_rows", "resgate_adaln_rows", "resgate_out_rows", "swiglu_rows", "gemm_swiglu")   # the 12-launch step's
INF3B = ("bo_front_bf16_sm100", "bo_pv_gate_inf_sm100", "bo_tail")                                     # front, core, tail


@pytest.fixture
def inf3b(monkeypatch):
    """The switch on (the default, set explicitly), and a spy that the runner really took the bf16 three-kernel step (a failed build
    would fall back silently)."""
    from miniworld_engine.kernels.bias_only_dit.cuda import runner as RN
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_INF3_BF16", "1")
    calls = []
    orig = RN.FusedBiasOnlyDiT._step3b

    def spy(self, *a, **k):
        calls.append(1)
        return orig(self, *a, **k)

    monkeypatch.setattr(RN.FusedBiasOnlyDiT, "_step3b", spy)
    return calls


@pytest.mark.parametrize(("env", "want"), [(None, True), ("1", True), ("0", False)])
def test_inf3_bf16_is_the_default(monkeypatch, env, want):
    """No switch set: the runner takes the three-kernel step; MINIWORLD_BIAS_ONLY_DIT_INF3_BF16=0: the 12-launch step."""
    from miniworld_engine.kernels.bias_only_dit.cuda import runner as RN
    if env is None:
        monkeypatch.delenv("MINIWORLD_BIAS_ONLY_DIT_INF3_BF16", raising=False)
    else:
        monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_INF3_BF16", env)
    calls = []
    orig = RN.FusedBiasOnlyDiT._step3b

    def spy(self, *a, **k):
        calls.append(1)
        return orig(self, *a, **k)

    monkeypatch.setattr(RN.FusedBiasOnlyDiT, "_step3b", spy)
    _, fast, ref_bf = blocks(seed=9)
    x = torch.randn(5, 1, 384, 768, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(5, 1, 384, 384, device="cuda", dtype=torch.bfloat16)
    p = torch.randn(1, 384, 384, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        got = fast(x, c, p)
        assert relative(got, ref_bf(x, c, p)) < 0.01
    assert bool(calls) is want, calls


def _ln64(x):
    x = x.double()
    return (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + 1e-5)


@pytest.mark.parametrize(("xbf", "obf"), [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("cl", [8, 6])
@pytest.mark.parametrize("da", [768, 1024])
@pytest.mark.parametrize(("M", "T"), [(128, 128), (640, 128), (1920, 1920), (3840, 768)])
def test_tail_bf16_kernel_matches_fp64(xbf, obf, cl, da, M, T):
    """bo_tail_bf16.cu alone: out = x1 + gate2 (silu(xt Wa^T) (xt Wb^T)) Wsq^T, x1 = x + gate1 a Wo^T, xt = LN(x1) s2 + sh2 against
    fp64 on the same bf16 inputs -- the block update out - x within 1.5e-2 (bf16 xt and h, unit roundoff 2^-9), xt / h within their
    one bf16 rounding -- for x fp32 / bf16 and out fp32 / bf16; the scratch fully written; bit-identical on a rerun."""
    from miniworld_engine.kernels.bias_only_dit.cuda import inf3_bf16 as B16
    bf = torch.bfloat16
    torch.manual_seed(M + da + cl)
    dev = torch.cuda.current_device()
    a = torch.randn(M, da, device="cuda").to(bf)
    x = (1.5 * torch.randn(M, 768, device="cuda") + 0.3).to(bf if xbf else torch.float32)
    tab = torch.randn(T, 6, 768, device="cuda")
    tab[:, :4] = torch.sigmoid(tab[:, :4] + 1.0)
    tab = tab.to(bf)
    wo = (torch.randn(768, da, device="cuda") / da ** 0.5).to(bf)
    wab = (torch.randn(3072, 768, device="cuda") / 768 ** 0.5).to(bf)
    wsq = (torch.randn(768, 1536, device="cuda") / 1536 ** 0.5).to(bf)
    tail = B16.TailBF16(dev, da)
    k = tail.kernel(cl, xbf, obf)
    assert k.lmem == 0, (k.regs, k.lmem)
    assert k.regs <= 168, (k.regs, k.lmem)
    wop, wsqp = B16.pack_pairs_bf16(wo, cl), B16.pack_pairs_bf16(wsq, cl, z_order=True)
    nc = 768 // cl
    k0 = torch.arange(64, device="cuda")
    assert torch.equal(wop.view(cl, -1, 2, nc, 64)[3, 2, 1, 7], wo[nc * 3 + 7, 64 * 5 + k0])   # P[((c p) 2 + h) NC + n, k]
    order = B16.tail_order(cl, 1)
    assert torch.equal(wsqp.view(cl, -1, 2, nc, 64)[1, 0, 1, 5], wsq[nc + 5, 64 * order[1] + k0])
    out = torch.full((M, 768), float("nan"), device="cuda").to(bf if obf else torch.float32)
    xt = torch.full((M, 832), float("nan"), device="cuda", dtype=bf)[:, :768]
    hs = torch.full((M * 24, 64), float("nan"), device="cuda", dtype=bf)
    tail(a, x, tab, wop, wab, wsq=wsqp, out=out, T=T, xt=xt, h=hs, cl=cl)
    tok = torch.arange(M, device="cuda") % T
    t64 = tab.double()[tok]
    x1 = x.double() + t64[:, 0] * (a.double() @ wo.double().t())
    xt64 = _ln64(x1) * t64[:, 3] + t64[:, 5]
    ab = xt64 @ wab.double().t()
    h = torch.nn.functional.silu(ab[:, :1536]) * ab[:, 1536:]
    want = x1 + t64[:, 1] * (h @ wsq.double().t())
    assert torch.isfinite(out.float()).all()
    assert relative(out.double() - x.double(), want - x.double()) < 1.5e-2, relative(out.double() - x.double(), want - x.double())
    assert torch.isfinite(xt.float()).all()                    # the exchange scratch: every block written
    assert torch.isfinite(hs.float()).all()
    h_rows = hs.view(M // 128, 24, 128, 64).permute(0, 2, 1, 3).reshape(M, 1536)
    assert relative(xt.double(), xt64) < 8e-3
    assert relative(h_rows.double(), h) < 1.5e-2
    again = torch.empty_like(out)
    tail(a, x, tab, wop, wab, wsq=wsqp, out=again, T=T, xt=xt, h=hs, cl=cl)
    assert torch.equal(again, out)


@pytest.mark.parametrize(("xbf", "obf"), [(False, False), (True, True), (True, False)])
@pytest.mark.parametrize("da", [768, 1024])
@pytest.mark.parametrize(("M", "T"), [(128, 128), (640, 128), (3200, 3200), (3840, 768)])
def test_tail2_bf16_kernel_matches_fp64(xbf, obf, da, M, T):
    """bo_tail2_bf16.cu (the pair tail: two row tiles x 4 column groups per cluster of 8, tcgen05 cta_group::2) alone against fp64 at
    bo_tail_bf16.cu's bounds; its padded scratch fully written (an odd tile count -- M 128, 640, 3200 -- leaves the last pair one
    tile short: that tile writes its own padding rows and no output); bit-identical on a rerun."""
    from miniworld_engine.kernels.bias_only_dit.cuda import inf3_bf16 as B16
    bf = torch.bfloat16
    torch.manual_seed(M + da + 3)
    dev = torch.cuda.current_device()
    a = torch.randn(M, da, device="cuda").to(bf)
    x = (1.5 * torch.randn(M, 768, device="cuda") + 0.3).to(bf if xbf else torch.float32)
    tab = torch.randn(T, 6, 768, device="cuda")
    tab[:, :4] = torch.sigmoid(tab[:, :4] + 1.0)
    tab = tab.to(bf)
    wo = (torch.randn(768, da, device="cuda") / da ** 0.5).to(bf)
    wab = (torch.randn(3072, 768, device="cuda") / 768 ** 0.5).to(bf)
    wsq = (torch.randn(768, 1536, device="cuda") / 1536 ** 0.5).to(bf)
    tail = B16.TailBF16(dev, da)
    k = tail.kernel(B16.TAIL_PAIR, xbf, obf)
    assert k.lmem == 0, (k.regs, k.lmem)
    assert k.regs <= 168, (k.regs, k.lmem)
    Mp = B16.tail_rows(M, B16.TAIL_PAIR)
    wop, wsqp = B16.pack_pairs_bf16(wo, 8), B16.pack_pairs_bf16(wsq, 8)
    out = torch.full((M, 768), float("nan"), device="cuda").to(bf if obf else torch.float32)
    xt = torch.full((Mp, 832), float("nan"), device="cuda", dtype=bf)[:, :768]
    hs = torch.full((Mp * 24, 64), float("nan"), device="cuda", dtype=bf)
    tail(a, x, tab, wop, wab, wsqp, out, T, xt=xt, h=hs, cl=B16.TAIL_PAIR)
    tok = torch.arange(M, device="cuda") % T
    t64 = tab.double()[tok]
    x1 = x.double() + t64[:, 0] * (a.double() @ wo.double().t())
    xt64 = _ln64(x1) * t64[:, 3] + t64[:, 5]
    ab = xt64 @ wab.double().t()
    h = torch.nn.functional.silu(ab[:, :1536]) * ab[:, 1536:]
    want = x1 + t64[:, 1] * (h @ wsq.double().t())
    assert torch.isfinite(out.float()).all()
    assert relative(out.double() - x.double(), want - x.double()) < 1.5e-2, relative(out.double() - x.double(), want - x.double())
    assert torch.isfinite(xt.float()).all()                    # every block of every tile, padding included
    assert torch.isfinite(hs.float()).all()
    h_rows = hs.view(Mp // 128, 24, 128, 64).permute(0, 2, 1, 3).reshape(Mp, 1536)[:M]
    assert relative(xt[:M].double(), xt64) < 8e-3
    assert relative(h_rows.double(), h) < 1.5e-2
    again = torch.empty_like(out)
    tail(a, x, tab, wop, wab, wsqp, again, T, xt=xt, h=hs, cl=B16.TAIL_PAIR)
    assert torch.equal(again, out)


@pytest.mark.parametrize("xbf", [True, False])
@pytest.mark.parametrize("da", [768, 1024])
@pytest.mark.parametrize("cl", [4, 6, 8])
@pytest.mark.parametrize(("M", "T"), [(128, 128), (640, 128), (1920, 1920), (3840, 768)])
def test_front_bf16_kernel_matches_fp64(xbf, da, cl, M, T):
    """bo_front_bf16.cu alone: v | g = (LN(x) s1 + sh1) [Wv; Wg]^T against fp64 on the same bf16 inputs (within 1e-2: xa and v | g
    rounded to bf16), its xa scratch fully written and within its one bf16 rounding, bit-identical on a rerun; every cluster size,
    x fp32 / bf16."""
    from miniworld_engine.kernels.bias_only_dit.cuda import inf3_bf16 as B16
    if cl == 6 and da == 1024:
        pytest.skip("front CL 6 splits 1536 v|g columns: 768 attention channels only")
    bf = torch.bfloat16
    torch.manual_seed(M + da + cl)
    dev = torch.cuda.current_device()
    x = (1.5 * torch.randn(M, 768, device="cuda") + 0.3).to(bf if xbf else torch.float32)
    tab = torch.randn(T, 6, 768, device="cuda")
    tab[:, :4] = torch.sigmoid(tab[:, :4] + 1.0)
    tab = tab.to(bf)
    w = (torch.randn(2 * da, 768, device="cuda") / 768 ** 0.5).to(bf)
    front = B16.FrontBF16(dev, da)
    k = front.kernel(cl, xbf)
    assert k.lmem == 0, (k.regs, k.lmem)
    assert k.regs <= 168, (k.regs, k.lmem)
    vg = torch.full((M, 2 * da), float("nan"), device="cuda", dtype=bf)
    xa = torch.full((M, 832), float("nan"), device="cuda", dtype=bf)[:, :768]
    front(x, tab, w, vg, T, xa=xa, cl=cl)
    tok = torch.arange(M, device="cuda") % T
    t64 = tab.double()[tok]
    xa64 = _ln64(x) * t64[:, 2] + t64[:, 4]
    want = xa64 @ w.double().t()
    assert torch.isfinite(vg.float()).all()
    assert torch.isfinite(xa.float()).all()
    assert relative(xa.double(), xa64) < 8e-3, relative(xa.double(), xa64)
    assert relative(vg.double(), want) < 1e-2, relative(vg.double(), want)
    again = torch.empty_like(vg)
    front(x, tab, w, again, T, xa=xa, cl=cl)
    assert torch.equal(again, vg)


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize("L", [128, 384, 768])
@pytest.mark.parametrize("S", [1, 5])
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("masked", [False, True])
def test_inf3_bf16_block_matches_the_pytorch_block(inf3b, L, S, shared, masked, n_head, d_head):
    """The bf16 three-kernel step at the 12-launch step's bounds: against the fp32 module within 1.05 x the PyTorch bf16 block's own
    error, and within 1e-2 of the PyTorch bf16 block (L768, S 5: the pair tail)."""
    ref, fast, ref_bf = blocks(n_head=n_head, d_head=d_head)
    x = torch.randn(S, 1, L, 768, device="cuda")
    c = torch.randn(1, 1, L, 384, device="cuda").expand(S, 1, L, 384) if shared else torch.randn(S, 1, L, 384, device="cuda")
    p = torch.randn(1, L, L, 128, device="cuda")
    mask = (torch.rand(1, L, device="cuda") > 0.2) if masked else None
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        want = ref(x, c, p, mask)
        args = (x.bfloat16(), c.bfloat16(), p.bfloat16(), mask)
        assert bias_only_dit.serves(fast, *args)
        got = fast(*args)
        base = ref_bf(*args)
    assert inf3b, "the bf16 three-kernel step did not run"
    assert got.dtype is torch.bfloat16
    assert relative(got, want) <= 1.05 * relative(base, want), (relative(got, want), relative(base, want))
    assert relative(got, base) < 0.01


def _poison(mb=512):
    """Fill the caching allocator's next blocks with NaN: a buffer the step reads before writing would show it."""
    junk = torch.full((mb << 18,), float("nan"), device="cuda")
    torch.cuda.synchronize()
    del junk


@pytest.mark.parametrize(("n_head", "d_head"), [(16, 48), (16, 64)])
@pytest.mark.parametrize("L", [128, 384, 512, 640, 768])
@pytest.mark.parametrize("shared", [True, False])
def test_inf3_bf16_bit_identical_reruns_with_a_poisoned_allocator(inf3b, n_head, d_head, L, shared):
    """Twenty steps from scratch (runner, buffers, tables, bound launches and P dropped; free memory NaN-filled before each): finite
    and bit-identical -- fixed-order reductions, no atomics, nothing read before it is written."""
    _, fast, _ = blocks(seed=9, n_head=n_head, d_head=d_head)
    S = 5
    bf = torch.bfloat16
    x = torch.randn(S, 1, L, 768, device="cuda", dtype=bf)
    c = (torch.randn(1, 1, L, 384, device="cuda", dtype=bf).expand(S, 1, L, 384) if shared else
         torch.randn(S, 1, L, 384, device="cuda", dtype=bf))
    p, mask = torch.randn(1, L, L, 128, device="cuda", dtype=bf), torch.rand(1, L, device="cuda") > 0.2
    outs = []
    for _ in range(20):
        bias_only_dit._RUNNERS.clear()
        _poison()
        with torch.no_grad():
            outs.append(fast(x, c, p, mask).clone())
        torch.cuda.synchronize()
    assert len(inf3b) == 20
    assert torch.isfinite(outs[0].float()).all()
    for o in outs[1:]:
        assert torch.equal(o, outs[0])


@pytest.mark.parametrize("static", [True, False])
@pytest.mark.parametrize(("n_head", "d_head"), [(16, 48), (16, 64)])
@pytest.mark.parametrize("L", [384, 768])
def test_inf3_bf16_graph_is_three_kernels_and_replays(inf3b, static, n_head, d_head, L):
    """The captured bf16 step: with the weights and the conditioning static exactly the three kernels (front, core, tail -- the pair
    tail at L768), replaying bit-identically to the eager call and following a new single copied in; without, the hoists are recorded
    too and the three kernels close the graph. None of the 12-launch step's rows."""
    import contextlib

    from tests.cuda_graph_nodes import graph_kernels

    from miniworld_engine.kernels import _capture
    _, fast, _ = blocks(seed=5, n_head=n_head, d_head=d_head)
    S, bf = 5, torch.bfloat16
    x, c = torch.randn(S, 1, L, 768, device="cuda", dtype=bf), torch.randn(S, 1, L, 384, device="cuda", dtype=bf)
    p = torch.randn(1, L, L, 128, device="cuda", dtype=bf)
    xs, cs, ps = x.clone(), c.clone(), p.clone()
    ctx = contextlib.ExitStack()
    if static:
        ctx.enter_context(_capture.static_weights())
        ctx.enter_context(_capture.static_inputs())
    with ctx, torch.no_grad():
        eager = fast(x, c, p)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                fast(xs, cs, ps)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            out = fast(xs, cs, ps)
        names = graph_kernels(graph)
        graph.replay()
        torch.cuda.synchronize()
        tail = "bo_tail2_bf16_sm100" if L == 768 else "bo_tail_bf16_sm100"
        assert not any(r in n for n in names for r in TAIL_ROWS), names
        if static:
            assert len(names) == 3, names
            assert all(k in n for k, n in zip(INF3B, names, strict=True)), names
            assert tail in names[2], names
            assert torch.equal(out, eager)
        else:
            assert all(k in n for k, n in zip(INF3B, names[-3:], strict=True)), names
            assert tail in names[-1], names
            assert sum(any(k in n for k in INF3B) for n in names) == 3, names
            assert relative(out, eager) < 1e-5
        x2 = torch.randn_like(x)
        xs.copy_(x2)
        graph.replay()
        torch.cuda.synchronize()
        new = fast(x2, c, p)
        if static:
            assert torch.equal(out, new)
        else:
            assert relative(out, new) < 1e-5


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
def test_inf3_bf16_kernels_do_not_spill(n_head, d_head):
    """No local memory in the bf16 three-kernel step's cubins (the served, non-TRACE builds): every front (CL 8 / 6 / 4, x fp32 /
    bf16), every tail (CL 8 / 6 and the pair tail, x fp32 / bf16, out fp32 / bf16) within __launch_bounds__(384, 1)'s 168 registers,
    and the PDL core for every sample group."""
    from miniworld_engine.kernels.bias_only_dit.cuda import inf3_bf16 as B16
    dev, da = torch.cuda.current_device(), n_head * d_head
    tail, front = B16.TailBF16(dev, da), B16.FrontBF16(dev, da)
    for cl in front.clusters():
        for xbf in (False, True):
            k = front.kernel(cl, xbf)
            assert k.lmem == 0, (cl, xbf, k.regs, k.lmem)
            assert k.regs <= 168, (cl, xbf, k.regs, k.lmem)
    for cl in (*B16.TAIL_CLUSTERS, B16.TAIL_PAIR):
        for xbf, obf in ((False, False), (False, True), (True, True), (True, False)):
            k = tail.kernel(cl, xbf, obf)
            assert k.lmem == 0, (cl, xbf, obf, k.regs, k.lmem)
            assert k.regs <= 168, (cl, xbf, obf, k.regs, k.lmem)
    core = C.PvGateCore(dev, nh=n_head, dh=d_head, pdl=True)
    for sg in (1, 2, 3, 4, 5):
        k = core.kernel(sg, True)
        assert k.lmem == 0, (sg, k.regs, k.lmem)


@pytest.mark.parametrize(("L", "want", "want_front"), [(384, 8, 8), (512, 6, 6), (640, 4, 4), (768, 4, 4)])
def test_inf3_bf16_cluster_choice(inf3b, monkeypatch, L, want, want_front):
    """A = 5, read off the runner's choices (spies, not the profiler). Tail: CL 8 at L384 (15 tiles, one round), CL 6 at L512 (20
    tiles: one round of 22 against two of 15), the pair tail (4) at L640 / L768 (25 / 30 tiles: one round of clusters of two tiles
    against two). Front: 8 / 6 / 4 / 4 (rounds of ~15 / 22 / 33 resident clusters x 8 / CL, ties to fewer rounds)."""
    from miniworld_engine.kernels.bias_only_dit.cuda import inf3_bf16 as B16
    picked, picked_front = [], []
    orig, orig_front = B16.TailBF16.cluster, B16.FrontBF16.cluster

    def spy(self, n_tiles):
        cl = orig(self, n_tiles)
        picked.append(cl)
        return cl

    def spy_front(self, n_tiles):
        cl = orig_front(self, n_tiles)
        picked_front.append(cl)
        return cl

    monkeypatch.setattr(B16.TailBF16, "cluster", spy)
    monkeypatch.setattr(B16.FrontBF16, "cluster", spy_front)
    _, fast, _ = blocks(seed=3)
    S, bf = 5, torch.bfloat16
    x, c, p = (torch.randn(S, 1, L, 768, device="cuda", dtype=bf), torch.randn(S, 1, L, 384, device="cuda", dtype=bf),
               torch.randn(1, L, L, 128, device="cuda", dtype=bf))
    bias_only_dit._RUNNERS.clear()
    with torch.no_grad():
        out = fast(x, c, p)
    torch.cuda.synchronize()
    assert inf3b, "the bf16 three-kernel step did not run"
    assert picked, picked
    assert picked == [want] * len(picked), (len(inf3b), picked)
    front = B16.FrontBF16(torch.cuda.current_device(), 768)
    resident = {cl: B16._max_clusters(front.kernel(cl), cl) for cl in front.clusters()}     # in the message: the model's inputs
    assert picked_front, resident
    assert picked_front == [want_front] * len(picked_front), (picked_front, resident)
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize("gate", [-80.0, -86.5, -87.5, -88.0, -88.5, -89.0, -100.0, -120.0, 88.0, 100.0])
def test_core_stays_finite_for_saturated_gates(gate, n_head, d_head):
    # (from 5a46c2ef) sigmoid(g) near and below the smallest normal f32: the epilogue's reciprocal seed is only valid up to d = 1 + 2^125, past
    # g = -86.6 it gave NaN (-88 .. -88.7) and inf (below) for the whole gated product
    torch.manual_seed(7)
    L, S = 256, 3
    M, DA = S * L, n_head * d_head
    vg = 10 * torch.randn(M, 2 * DA, device="cuda", dtype=torch.bfloat16)
    vg[:, DA:] = gate
    P = torch.softmax(3 * torch.randn(n_head * L, L, device="cuda"), dim=-1).to(torch.bfloat16)
    a = torch.empty(M, DA, device="cuda", dtype=torch.bfloat16)
    C.PvGateCore(torch.cuda.current_device(), nh=n_head, dh=d_head)(vg[:, :DA], P, a, S, g=vg[:, DA:])
    assert torch.isfinite(a.float()).all()
    torch.testing.assert_close(a.float(), R.pv_gate(vg, P, S).float(), rtol=2e-2, atol=1e-30 if gate < 0 else 1e-2)


def test_core_one_saturated_gate_does_not_touch_its_neighbours():
    # a single g = -88 among ordinary gates: only that product may change (the incident: one NaN element spread over its token row)
    torch.manual_seed(8)
    L, S, n_head, d_head = 128, 2, 16, 48
    M, DA = S * L, n_head * d_head
    vg = torch.randn(M, 2 * DA, device="cuda", dtype=torch.bfloat16)
    vg[5, DA + 98] = -88.0
    P = torch.softmax(torch.randn(n_head * L, L, device="cuda"), dim=-1).to(torch.bfloat16)
    a = torch.empty(M, DA, device="cuda", dtype=torch.bfloat16)
    C.PvGateCore(torch.cuda.current_device(), nh=n_head, dh=d_head)(vg[:, :DA], P, a, S, g=vg[:, DA:])
    assert torch.isfinite(a.float()).all()
    assert relative(a, R.pv_gate(vg, P, S)) < 1e-3
