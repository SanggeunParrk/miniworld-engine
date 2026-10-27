"""The bias-only token DiT training block on a pair bias of its own per sample (``bias_only_dit_train.block_psb``, reached through
``BiasOnlyDiTBlock(single [1, A, L, d], cond, pair [A, L, L, 128])``) against the PyTorch module fed the same per-sample pair in
fp32 IEEE: loss, output, d single / d cond / d pair and every parameter gradient, bf16 and fp32 (TF32), with a key mask per sample,
shared or none. Perturbed (non-zero) weights: zero-initialised ones make most gradients vanish. Every step is also run twice for
bitwise determinism. The shared-pair path ([1, H, L, L] bias, one P for all samples) keeps its own tests
(``test_h100_bias_only_dit_gpu.py``)."""

import math

import pytest
import torch
from portable_helpers import randomize_half_std as randomize
from settings_helpers import policy_portable_off as policy  # noqa: F401

from miniworld_engine import settings
from miniworld_engine.integrations import bias_only_dit_train as bt
from miniworld_engine.kernels.bias_only_dit import cuda as C
from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="Hopper (sm_90) required"
    ),
]
LN2 = math.log(2.0)


def rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def cosine(a, b):
    a, b = a.detach().double().flatten(), b.detach().double().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def make_pair(nh=16, dh=48, d_cond=384):
    torch.manual_seed(7)
    ref = randomize(BiasOnlyDiTBlock(d_cond=d_cond, n_head=nh, d_head=dh, implementation=ImplementationType.PYTORCH).cuda().float())
    eng = BiasOnlyDiTBlock(d_cond=d_cond, n_head=nh, d_head=dh, implementation=ImplementationType.MINIWORLD).cuda().float()
    eng.load_state_dict(ref.state_dict())
    return ref, eng


def make_inputs(A, L, mask_kind, seed=8, d_cond=384):
    g = torch.Generator(device="cuda").manual_seed(seed)
    single = torch.randn(A, L, 768, device="cuda", generator=g)
    cond = torch.randn(A, L, d_cond, device="cuda", generator=g)
    pair = torch.randn(A, L, L, 128, device="cuda", generator=g)
    if mask_kind == "none":
        mask = None
    elif mask_kind == "shared":
        mask = (torch.rand(1, L, device="cuda", generator=g) > 0.15).expand(A, L).contiguous()
    else:
        mask = torch.rand(A, L, device="cuda", generator=g) > 0.15
    w = torch.randn(A, L, 768, device="cuda", generator=g)
    return single, cond, pair, mask, w


def run(m, single, cond, pair, mask, w, dtype, tf32, autocast=False):
    """Loss = <out, w>; the module on [1, A, L, d] (the samples in the batch axis: each reads its own pair row)."""
    A, L = single.shape[:2]
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    try:
        ins = [t.detach().clone().to(dtype).requires_grad_() for t in (single, cond, pair)]
        m.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out = m(ins[0].view(1, A, L, -1), ins[1].view(1, A, L, -1), ins[2], mask)
        loss = (out.float() * w.view(1, A, L, -1)).sum()
        loss.backward()
        res = {"loss": loss.detach().double(), "out": out.detach().float(), "dsingle": ins[0].grad.float(),
               "dcond": ins[1].grad.float(), "dpair": ins[2].grad.float()}
        res.update({"p." + n: p.grad.float().clone() for n, p in m.named_parameters() if p.grad is not None})
        return res
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


class Spy:
    """Counts the calls that reach the fused per-sample block."""

    def __init__(self, monkeypatch):
        self.n = 0
        real = bt.block_psb

        def counted(*a, **k):
            self.n += 1
            return real(*a, **k)

        monkeypatch.setattr(bt, "block_psb", counted)


def metrics(got, truth):
    grads = [k for k in truth if k not in ("loss", "out")]
    whole_got = torch.cat([got[k].flatten() for k in grads])
    whole_truth = torch.cat([truth[k].flatten() for k in grads])
    worst = min(grads, key=lambda k: cosine(got[k], truth[k]))
    return {
        "loss_rel": abs(float(got["loss"] - truth["loss"])) / abs(float(truth["loss"])),
        "out_rel": rel(got["out"], truth["out"]),
        "whole_cos": cosine(whole_got, whole_truth),
        "worst_cos": cosine(got[worst], truth[worst]),
        "worst_name": worst,
    }


CASES = [  # (A, L, mask)
    (1, 128, "none"),
    (2, 256, "per_sample"),
    (5, 384, "shared"),
    (5, 640, "per_sample"),
    (48, 384, "per_sample"),
    (3, 768, "none"),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("A", "L", "mask_kind"), CASES)
def test_per_sample_bias_block_matches_the_fp32_module_and_is_bitwise_repeatable(dtype, A, L, mask_kind, monkeypatch):
    ref, eng = make_pair()
    single, cond, pair, mask, w = make_inputs(A, L, mask_kind)
    bf = dtype is torch.bfloat16
    truth = run(ref, single, cond, pair, mask, w, torch.float32, tf32=False)
    spy = Spy(monkeypatch)
    got = run(eng, single, cond, pair, mask, w, dtype, tf32=True, autocast=bf)
    # one sample is the shared-pair call ([1, 1, L, d], pair [1, L, L, 128]): the shared path serves it, block_psb is reached directly below
    assert spy.n == (1 if A > 1 else 0), "the per-sample fused block did not serve the call"
    again = run(eng, single, cond, pair, mask, w, dtype, tf32=True, autocast=bf)
    for k in got:
        assert torch.equal(got[k], again[k]), f"{k}: the step is not bitwise repeatable"
    assert set(got) == set(truth), sorted(set(truth) ^ set(got))
    m = metrics(got, truth)
    # the PyTorch bf16 module measures ~4e-3 on this recipe (test_h100_bias_only_dit_gpu); TF32 ~3e-4
    tol = 8e-3 if bf else 2.5e-3
    assert m["loss_rel"] < tol, m
    assert m["out_rel"] < tol, m
    assert m["whole_cos"] > 1 - 2e-4 * (4 if bf else 1), m
    assert m["worst_cos"] > 1 - (4 * tol) ** 2 / 2 * 4, m
    for k in ("dsingle", "dcond", "dpair"):
        assert torch.isfinite(got[k]).all(), k
        assert rel(got[k], truth[k]) < 3 * tol, (k, rel(got[k], truth[k]))


@pytest.mark.parametrize("nh", [(24, 32), (12, 64), (16, 64)])
def test_other_head_layouts(nh, monkeypatch):
    ref, eng = make_pair(*nh)
    single, cond, pair, mask, w = make_inputs(3, 256, "per_sample")
    truth = run(ref, single, cond, pair, mask, w, torch.float32, tf32=False)
    spy = Spy(monkeypatch)
    got = run(eng, single, cond, pair, mask, w, torch.float32, tf32=True)
    assert spy.n == 1
    m = metrics(got, truth)
    assert m["out_rel"] < 2.5e-3 and m["whole_cos"] > 1 - 2e-4, m


def test_a_length_off_the_tile_keeps_the_module_composition(monkeypatch):
    """L not a multiple of 128: nothing fused serves; the PyTorch composition runs (no Triton fallback)."""
    ref, eng = make_pair()
    single, cond, pair, mask, w = make_inputs(2, 200, "per_sample")
    spy = Spy(monkeypatch)
    got = run(eng, single, cond, pair, mask, w, torch.float32, tf32=False)
    truth = run(ref, single, cond, pair, mask, w, torch.float32, tf32=False)
    assert spy.n == 0
    assert rel(got["out"], truth["out"]) < 1e-5


@pytest.mark.parametrize("f32", [False, True])
@pytest.mark.parametrize(("A", "L", "mask_kind"), [(1, 128, "none"), (4, 384, "per_sample"), (5, 640, "shared")])
def test_softmax_t_per_sample(f32, A, L, mask_kind):
    """The per-sample softmax: P = softmax(ln 2 bias + key mask of the sample), P^T optional."""
    H = 16
    bias = torch.randn(H, A, L, L, device="cuda") * 2
    _, _, _, mask, _ = make_inputs(A, L, mask_kind)
    km = None if mask is None else (mask[:1] if mask_kind == "shared" else mask)
    dt = torch.float32 if f32 else torch.bfloat16
    b = bias.to(dt)
    logits = b.float() * LN2
    if mask is not None:
        logits = logits.masked_fill(~mask.view(1, A, 1, L), -3.0e38)
    want = torch.softmax(logits, -1)
    p = torch.empty(H, A, L, L, device="cuda", dtype=torch.float32 if f32 else torch.bfloat16)
    if f32:
        C_ = __import__("miniworld_engine.kernels.bias_only_dit.cuda.tf32", fromlist=["rows32"]).rows32()
        C_.softmax_t_cuda(b.view(-1, L), p.view(-1, L), None, km, A, LN2)
        pt = torch.empty_like(p)
        C_.softmax_t_cuda(b.view(-1, L), p.view(-1, L), pt.view(-1, L), km, A, LN2)
    else:
        C.softmax_t(b.view(-1, L), p.view(-1, L), None, km, A, LN2)
        pt = torch.empty_like(p)
        C.softmax_t(b.view(-1, L), p.view(-1, L), pt.view(-1, L), km, A, LN2)
    assert rel(p, want) < (1e-6 if f32 else 5e-3)
    assert torch.equal(pt, p.transpose(-1, -2)), "P^T"


def _stack_run(blocks, single, cond, pair, mask, w, dtype, autocast, hoisted):
    """Blocks chained on [1, A, L, d]; the pair's biases from ONE ``pair_bias_all`` (hoisted) or each block's own pair."""
    from miniworld_engine.modules.bias_only_dit import pair_bias_all

    A, L = single.shape[:2]
    ins = [t.detach().clone().to(dtype).requires_grad_() for t in (single, cond, pair)]
    for b in blocks:
        b.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        x, c = ins[0].view(1, A, L, -1), ins[1].view(1, A, L, -1)
        biases = pair_bias_all(blocks, ins[2]) if hoisted else [None] * len(blocks)
        for b, bias in zip(blocks, biases, strict=True):
            x = b(x, c, None if hoisted else ins[2], mask, bias=bias)
    loss = (x.float().view(A, L, -1) * w).sum()
    loss.backward()
    got = {"loss": loss.detach().double(), "out": x.detach().float().view(A, L, -1), "dsingle": ins[0].grad.float(),
           "dcond": ins[1].grad.float(), "dpair": ins[2].grad.float()}
    for i, b in enumerate(blocks):
        got.update({f"p{i}." + n: p.grad.float().clone() for n, p in b.named_parameters() if p.grad is not None})
    return got


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("A", "L", "mask_kind"), [(2, 256, "per_sample"), (5, 384, "none")])
def test_hoisted_per_sample_bias_over_a_stack(dtype, A, L, mask_kind, monkeypatch):
    """Three blocks on biases from one ``pair_bias_all`` (the hoist leaves the head-major buffers the block reads without a copy):
    against the fp32 PyTorch stack, each block on its own pair; bitwise repeatable; the gradients reach pair / ln_pair / to_bias."""
    from miniworld_engine.modules.bias_only_dit import pair_bias_all

    refs, engs = zip(*[make_pair() for _ in range(3)], strict=True)
    for i, (r, e) in enumerate(zip(refs, engs, strict=True)):  # three different blocks
        torch.manual_seed(100 + i)
        randomize(r)
        e.load_state_dict(r.state_dict())
    single, cond, pair, mask, w = make_inputs(A, L, mask_kind)
    truth = _stack_run(list(refs), single, cond, pair, mask, w, torch.float32, False, hoisted=False)
    spy = Spy(monkeypatch)
    bf = dtype is torch.bfloat16
    got = _stack_run(list(engs), single, cond, pair, mask, w, dtype, bf, hoisted=True)
    assert spy.n == 3
    again = _stack_run(list(engs), single, cond, pair, mask, w, dtype, bf, hoisted=True)
    assert all(torch.equal(got[k], again[k]) for k in got), "not bitwise repeatable"
    m = metrics(got, truth)
    assert m["out_rel"] < (1.2e-2 if bf else 3e-3) and m["whole_cos"] > 1 - (1e-3 if bf else 2e-4), m
    if bf:  # the fused hoist hands out the head-major buffer's permuted view: no copy in front of the block
        with torch.no_grad():
            b0 = pair_bias_all(list(engs), pair.to(dtype))[0]
        assert b0.permute(1, 0, 2, 3).is_contiguous()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("A", [1, 2])
def test_block_psb_called_directly(dtype, A):
    """``block_psb`` on a head-major bias, A = 1 included (the module routes one sample to the shared-pair path)."""
    from miniworld_engine.modules.bias_only_dit import pair_bias_all

    ref, eng = make_pair()
    L = 256
    single, cond, pair, mask, w = make_inputs(A, L, "per_sample")
    truth = run(ref, single, cond, pair, mask, w, torch.float32, tf32=False)
    bf = dtype is torch.bfloat16
    ins = [t.detach().clone().to(dtype).requires_grad_() for t in (single, cond, pair)]
    eng.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf):
        if A > 1:
            bias = pair_bias_all([eng], ins[2])[0].permute(1, 0, 2, 3)
        else:  # the shared hoist makes [1, H, L, L]: the same thing
            bias = pair_bias_all([eng], ins[2])[0].permute(1, 0, 2, 3)
        bias = bias if bias.is_contiguous() else bias.contiguous()
        s, c = ins[0].view(A, 1, L, -1), ins[1].view(A, 1, L, -1)
        assert bt.serves_psb(eng, s, c, bias, mask)
        out = bt.block_psb(eng, s, c, bias, mask)
    loss = (out.float().view(A, L, -1) * w).sum()
    loss.backward()
    got = {"loss": loss.detach().double(), "out": out.detach().float().view(A, L, -1), "dsingle": ins[0].grad.float(),
           "dcond": ins[1].grad.float(), "dpair": ins[2].grad.float()}
    got.update({"p." + n: p.grad.float().clone() for n, p in eng.named_parameters() if p.grad is not None})
    m = metrics(got, truth)
    assert m["out_rel"] < (8e-3 if bf else 2.5e-3) and m["whole_cos"] > 1 - 8e-4, m


# ------------------------------------------------------------------------------------------------ the single-pass kernels (bo_psb.cu)
def _psb_inputs(A, L, H, dh, mask_kind, seed=3):
    g = torch.Generator(device="cuda").manual_seed(seed)
    bf = torch.bfloat16
    DA = H * dh
    bias = (torch.randn(H, A, L, L, device="cuda", generator=g) * 2).to(bf)
    vg = torch.randn(A * L, 2 * DA, device="cuda", generator=g).to(bf)
    do = torch.randn(A * L, DA, device="cuda", generator=g).to(bf)
    dd = torch.randn(A, H, L, device="cuda", generator=g)
    mask = None
    if mask_kind == "shared":
        mask = torch.rand(1, L, device="cuda", generator=g) > 0.2
    elif mask_kind == "per_sample":
        mask = torch.rand(A, L, device="cuda", generator=g) > 0.2
    return bias, vg, do, dd, mask


@pytest.mark.parametrize(("H", "dh"), [(16, 48), (24, 32), (12, 64)])
@pytest.mark.parametrize(("A", "L", "mask_kind"), [(1, 128, "none"), (3, 256, "per_sample"), (5, 384, "shared"), (2, 640, "per_sample"), (2, 768, "none")])
def test_psb_kernels_match_fp64_and_are_bitwise_repeatable(H, dh, A, L, mask_kind):
    """psb_fwd (softmax + gated P v) and psb_bwd (dV = P^T dO, dbias = P o (dO v^T - D)) against an fp64 oracle on the same bf16 operands."""
    bias, vg, do, dd, mask = _psb_inputs(A, L, H, dh, mask_kind)
    DA = H * dh
    p = torch.empty(H, A, L, L, device="cuda", dtype=torch.bfloat16)
    og = torch.empty(A * L, DA, device="cuda", dtype=torch.bfloat16)
    km = None if mask is None else (mask.reshape(-1) if mask_kind == "shared" else mask)
    C.psb_fwd(bias.view(-1, L), p.view(-1, L), km, vg, og, A, H, dh)
    logits = bias.double()
    if mask is not None:
        logits = logits.masked_fill(~mask.view(-1, 1, 1, L) if mask_kind == "shared" else ~mask.view(1, A, 1, L), -1e30)
    P = torch.softmax(logits, -1)                                                  # [H, A, L, L]
    v = vg[:, :DA].double().view(A, L, H, dh).permute(2, 0, 1, 3)                  # [H, A, L, dh]
    g = torch.sigmoid(vg[:, DA:].double()).view(A, L, H, dh).permute(2, 0, 1, 3)
    want_og = (g * (P @ v)).permute(1, 2, 0, 3).reshape(A * L, DA)
    assert rel(p, P) < 6e-3
    assert rel(og, want_og) < 6e-3
    dvg = torch.zeros(A * L, 2 * DA, device="cuda", dtype=torch.bfloat16)
    dbias = torch.empty_like(p)
    C.psb_bwd(p.view(-1, L), do, vg, dd, dbias.view(-1, L), dvg, A, H, dh)
    Pb = p.double()                                                                # the bf16 P the backward reads
    dO = do.double().view(A, L, H, dh).permute(2, 0, 1, 3)
    want_dv = (Pb.transpose(-1, -2) @ dO).permute(1, 2, 0, 3).reshape(A * L, DA)
    want_db = Pb * (dO @ v.transpose(-1, -2) - dd.double().permute(1, 0, 2).unsqueeze(-1))
    assert rel(dvg[:, :DA], want_dv) < 6e-3
    assert rel(dbias, want_db) < 6e-3
    assert not dvg[:, DA:].any(), "the g columns of dvg are not written"
    p2, og2, dvg2, dbias2 = torch.empty_like(p), torch.empty_like(og), torch.zeros_like(dvg), torch.empty_like(dbias)
    C.psb_fwd(bias.view(-1, L), p2.view(-1, L), km, vg, og2, A, H, dh)
    C.psb_bwd(p2.view(-1, L), do, vg, dd, dbias2.view(-1, L), dvg2, A, H, dh)
    assert torch.equal(p, p2) and torch.equal(og, og2) and torch.equal(dvg, dvg2) and torch.equal(dbias, dbias2)
    # the recompute form: no P in HBM, the forward's two row statistics, P rebuilt from the bias -- the same bits as the stored-P form
    stats = torch.empty(H * A * L, 2, device="cuda")
    og3, dvg3, dbias3 = torch.empty_like(og), torch.zeros_like(dvg), torch.empty_like(dbias)
    C.psb_fwd(bias.view(-1, L), None, km, vg, og3, A, H, dh, stats=stats)
    C.psb_bwd(bias.view(-1, L), do, vg, dd, dbias3.view(-1, L), dvg3, A, H, dh, stats=stats, mask=km)
    assert torch.equal(og, og3) and torch.equal(dvg, dvg3) and torch.equal(dbias, dbias3), "recompute differs from stored P"


@pytest.mark.parametrize("core", ["stored", "cublas"])
@pytest.mark.parametrize(("A", "L", "mask_kind"), [(3, 256, "per_sample"), (5, 384, "none"), (4, 640, "shared")])
def test_recompute_core_agrees_with_the_other_cores(core, A, L, mask_kind):
    """The default core (P rebuilt from the bias in the backward) against the stored-P kernels and the v0 cuBLAS core."""
    ref, eng = make_pair()
    single, cond, pair, mask, w = make_inputs(A, L, mask_kind)
    old = settings.configure(bias_only_dit_psb_core=core)
    try:
        other = run(eng, single, cond, pair, mask, w, torch.bfloat16, tf32=True, autocast=True)
    finally:
        settings.configure(**vars(old))
    assert settings.current().bias_only_dit_psb_core == "recompute"
    v1 = run(eng, single, cond, pair, mask, w, torch.bfloat16, tf32=True, autocast=True)
    m = metrics(v1, other)
    assert m["whole_cos"] > 1 - 1e-5 and m["out_rel"] < 2e-3, m
    if core == "stored":   # same arithmetic, P just takes a trip through HBM: bit for bit
        assert all(torch.equal(v1[k], other[k]) for k in v1), "recompute and stored-P differ"


@pytest.mark.parametrize(("H", "dh"), [(16, 48), (24, 32), (12, 64)])
@pytest.mark.parametrize(("A", "L", "mask_kind"), [(1, 128, "none"), (3, 256, "per_sample"), (5, 384, "shared"), (2, 512, "per_sample")])
def test_psb32_kernels_match_fp64_and_are_bitwise_repeatable(H, dh, A, L, mask_kind):
    """The fp32 (TF32) forward / backward against an fp64 oracle (operands in fp32; TF32 rounds them at 2^-11)."""
    DA = H * dh
    g = torch.Generator(device="cuda").manual_seed(5)
    bias = torch.randn(H, A, L, L, device="cuda", generator=g) * 2
    vg = torch.randn(A * L, 2 * DA, device="cuda", generator=g)
    do = torch.randn(A * L, DA, device="cuda", generator=g)
    dd = torch.randn(A, H, L, device="cuda", generator=g)
    mask = None
    if mask_kind == "shared":
        mask = torch.rand(1, L, device="cuda", generator=g) > 0.2
    elif mask_kind == "per_sample":
        mask = torch.rand(A, L, device="cuda", generator=g) > 0.2
    km = None if mask is None else (mask.reshape(-1) if mask_kind == "shared" else mask)
    stats = torch.empty(H * A * L, 2, device="cuda")
    og = torch.empty(A * L, DA, device="cuda")
    C.psb32_fwd(bias.view(-1, L), stats, km, vg, og, A, H, dh)
    logits = bias.double()
    if mask is not None:
        logits = logits.masked_fill(~mask.view(-1, 1, 1, L) if mask_kind == "shared" else ~mask.view(1, A, 1, L), -1e30)
    P = torch.softmax(logits, -1)
    v = vg[:, :DA].double().view(A, L, H, dh).permute(2, 0, 1, 3)
    gt = torch.sigmoid(vg[:, DA:].double()).view(A, L, H, dh).permute(2, 0, 1, 3)
    want_og = (gt * (P @ v)).permute(1, 2, 0, 3).reshape(A * L, DA)
    assert rel(og, want_og) < 1.5e-3
    dvg = torch.zeros(A * L, 2 * DA, device="cuda")
    dbias = torch.empty_like(bias)
    C.psb32_bwd(bias.view(-1, L), stats, km, do, vg, dd, dbias.view(-1, L), dvg, A, H, dh)
    dO = do.double().view(A, L, H, dh).permute(2, 0, 1, 3)
    want_dv = (P.transpose(-1, -2) @ dO).permute(1, 2, 0, 3).reshape(A * L, DA)
    want_db = P * (dO @ v.transpose(-1, -2) - dd.double().permute(1, 0, 2).unsqueeze(-1))
    assert rel(dvg[:, :DA], want_dv) < 1.5e-3
    assert rel(dbias, want_db) < 1.5e-3
    assert not dvg[:, DA:].any()
    og2, dvg2, dbias2, stats2 = torch.empty_like(og), torch.zeros_like(dvg), torch.empty_like(dbias), torch.empty_like(stats)
    C.psb32_fwd(bias.view(-1, L), stats2, km, vg, og2, A, H, dh)
    C.psb32_bwd(bias.view(-1, L), stats2, km, do, vg, dd, dbias2.view(-1, L), dvg2, A, H, dh)
    assert torch.equal(og, og2) and torch.equal(dvg, dvg2) and torch.equal(dbias, dbias2) and torch.equal(stats, stats2)


# ------------------------------------------------------------------------------------------------ the fused hoist backward (bo_hoist_bwd.cu)
@pytest.mark.parametrize("n_rows", [16, 48, 100, 192])
@pytest.mark.parametrize("rows", [4096, 64 * 1000])
def test_fused_hoist_backward_matches_fp64_and_the_cublas_path(n_rows, rows):
    """d pair and dW' of ``bias_only_dit_hoist._hoist_bwd``: the one-pass kernel against an fp64 oracle and against the cuBLAS + ln0_bwd_rows
    path (switched by settings.bias_only_dit_hoist_bwd_fused), bitwise repeatable."""
    from miniworld_engine.integrations import bias_only_dit_hoist as hoist

    g = torch.Generator(device="cuda").manual_seed(11)
    bf = torch.bfloat16
    pair = (torch.randn(rows, 128, device="cuda", generator=g) * 1.5 + 0.3).to(bf)
    wf = (torch.randn(n_rows, 128, device="cuda", generator=g) * 0.2).to(bf)
    dball = torch.randn(n_rows, rows, device="cuda", generator=g).to(bf)
    dpair, dwf = hoist._hoist_bwd(pair, wf, dball, 1e-5, hoist._split(rows))
    dpair2, dwf2 = hoist._hoist_bwd(pair, wf, dball, 1e-5, hoist._split(rows))
    assert torch.equal(dpair, dpair2) and torch.equal(dwf, dwf2), "not bitwise repeatable"
    old = settings.configure(bias_only_dit_hoist_bwd_fused=False)
    try:
        dpair_c, dwf_c = hoist._hoist_bwd(pair, wf, dball, 1e-5, hoist._split(rows))
    finally:
        settings.configure(**vars(old))
    x = pair.double()
    mean, var = x.mean(-1, keepdim=True), x.var(-1, unbiased=False, keepdim=True)
    rstd = (var + 1e-5).rsqrt()
    y = (x - mean) * rstd
    dln = dball.double().t() @ wf.double()
    want_dp = rstd * (dln - dln.mean(-1, keepdim=True) - y * (dln * y).mean(-1, keepdim=True))
    want_dw = dball.double() @ y
    assert rel(dpair, want_dp) < 6e-3                       # bf16 output
    assert rel(dwf, want_dw) < 2e-4                         # hi + lo split: the sum against LN0 to fp32 accumulation
    assert rel(dpair, dpair_c) < 6e-3 and rel(dwf, dwf_c) < 2e-4


# ------------------------------------------------------------------------------------------------ the one-pass hoist (bo_hoist_pass.cu)
def _ln_oracle(pair):
    x = pair.double()
    mean, var = x.mean(-1, keepdim=True), x.var(-1, unbiased=False, keepdim=True)
    rstd = (var + 1e-5).rsqrt()
    return (x - mean) * rstd, rstd


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_rows", [16, 32, 64])
@pytest.mark.parametrize("rows", [16, 4112, 64 * 1000])
def test_one_pass_hoist_forward(dtype, n_rows, rows):
    """``pass_fwd``: bias^T = W' LN0(pair)^T in one pass over the pair (any R multiple of 16, up to 64 bias rows) against an fp64 oracle, against the
    LayerNorm kernel + GEMM it replaces, bitwise repeatable; the routing (``_hoist_fwd``) takes it."""
    from miniworld_engine.integrations import bias_only_dit_hoist as hoist
    from miniworld_engine.kernels.bias_only_dit.cuda import hoist as HK

    g = torch.Generator(device="cuda").manual_seed(5)
    pair = (torch.randn(rows, 128, device="cuda", generator=g) * 1.5 + 0.3).to(dtype)
    wf = (torch.randn(n_rows, 128, device="cuda", generator=g) * 0.2).to(dtype)
    assert HK.pass_fwd_served(pair, wf)
    out = HK.pass_fwd(pair, wf)
    assert out.dtype is dtype and out.shape == (n_rows, rows)
    assert torch.equal(out, HK.pass_fwd(pair, wf)), "not bitwise repeatable"
    y, _ = _ln_oracle(pair)
    want = wf.double() @ y.t()
    assert rel(out, want) < (6e-3 if dtype is torch.bfloat16 else 1.5e-3)
    old = settings.configure(bias_only_dit_hoist_fwd_fused=False)
    try:
        ref = hoist._hoist_fwd(pair, wf, 1e-5)
    finally:
        settings.configure(**vars(old))
    assert rel(out, ref) < (6e-3 if dtype is torch.bfloat16 else 1.5e-3)
    assert torch.equal(hoist._hoist_fwd(pair, wf, 1e-5), out)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [16, 4112, 64 * 1000, 4 * 256 * 256])
def test_one_pass_hoist_backward(dtype, rows):
    """``pass_bwd`` (one block of 16 heads, bf16 and fp32 / TF32): d pair and dW' against an fp64 oracle and against the cuBLAS + ln0_bwd_rows path
    (``bias_only_dit_hoist_bwd_fused=False``), bitwise repeatable, any R multiple of 16."""
    from miniworld_engine.integrations import bias_only_dit_hoist as hoist
    from miniworld_engine.kernels.bias_only_dit.cuda import hoist as HK

    g = torch.Generator(device="cuda").manual_seed(13)
    pair = (torch.randn(rows, 128, device="cuda", generator=g) * 1.5 + 0.3).to(dtype)
    wf = (torch.randn(16, 128, device="cuda", generator=g) * 0.2).to(dtype)
    dball = torch.randn(16, rows, device="cuda", generator=g).to(dtype)
    assert HK.pass_bwd_served(pair, wf)
    dpair, dwf = hoist._hoist_bwd(pair, wf, dball, 1e-5, hoist._split(rows))
    dpair2, dwf2 = hoist._hoist_bwd(pair, wf, dball, 1e-5, hoist._split(rows))
    assert torch.equal(dpair, dpair2) and torch.equal(dwf, dwf2), "not bitwise repeatable"
    assert dpair.dtype is dtype and dwf.dtype is torch.float32 and dwf.shape == (16, 128)
    old = settings.configure(bias_only_dit_hoist_bwd_fused=False)
    try:
        dpair_c, dwf_c = hoist._hoist_bwd(pair, wf, dball, 1e-5, hoist._split(rows))
    finally:
        settings.configure(**vars(old))
    y, rstd = _ln_oracle(pair)
    dln = dball.double().t() @ wf.double()
    want_dp = rstd * (dln - dln.mean(-1, keepdim=True) - y * (dln * y).mean(-1, keepdim=True))
    want_dw = dball.double() @ y
    bf = dtype is torch.bfloat16
    assert rel(dpair, want_dp) < (6e-3 if bf else 2e-3)
    assert rel(dwf, want_dw) < (2e-4 if bf else 2e-3)          # bf16: hi + lo split; fp32: TF32 operands
    assert rel(dpair, dpair_c) < (6e-3 if bf else 2e-3) and rel(dwf, dwf_c) < (2e-4 if bf else 2e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("A", "L"), [(1, 256), (3, 256)])
def test_conditioning_width_768(dtype, A, L, monkeypatch):
    """d_cond 768 (the row kernels are templated on it): the block, its gradients (cond and the AdaLN / cond-LN weights included) against fp32;
    A = 1 is the shared-pair call, A > 1 the per-sample bias."""
    ref, eng = make_pair(d_cond=768)
    single, cond, pair, mask, w = make_inputs(A, L, "per_sample", d_cond=768)
    truth = run(ref, single, cond, pair, mask, w, torch.float32, tf32=False)
    spy = Spy(monkeypatch)
    bf = dtype is torch.bfloat16
    got = run(eng, single, cond, pair, mask, w, dtype, tf32=True, autocast=bf)
    again = run(eng, single, cond, pair, mask, w, dtype, tf32=True, autocast=bf)
    assert spy.n == (2 if A > 1 else 0)
    assert all(torch.equal(got[k], again[k]) for k in got), "not bitwise repeatable"
    m = metrics(got, truth)
    assert m["out_rel"] < (8e-3 if bf else 2.5e-3) and m["whole_cos"] > 1 - (8e-4 if bf else 2e-4), m
    assert got["dcond"].shape[-1] == 768
