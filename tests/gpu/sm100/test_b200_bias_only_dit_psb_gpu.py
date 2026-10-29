"""The bias-only token DiT on B200 with a pair bias of its own per sample (``BiasOnlyDiTBlock(single [1, A, L, d], cond, pair
[A, L, L, 128])`` or a hoisted [A, H, L, L] bias), bf16 and fp32 (TF32): the ``-DPSB`` builds of the shared-bias cores
(``pv_gate_inf`` / ``pv_gate_tf32``: a P plane per (head, sample); ``dpb_sm100`` / ``dpb32_sm100``: dbias per sample, nothing summed
over the samples, masked keys zeroed) against fp64, and the training block
(``bias_only_dit_train.block_psb``) / the inference step (``bias_only_dit.update_psb``) against the PyTorch module fed the same
per-sample pair in fp32 IEEE: loss, output, d single / d cond / d pair and every parameter gradient, a key mask per sample, shared,
none, or one sample with every key masked (its rows uniform, as the module's finfo.min fill makes them -- never NaN). Every step runs
twice for bitwise determinism; a captured training step replays the eager bits. Perturbed (non-zero) weights: zero-initialised ones
make most gradients vanish. The bars are the Hopper per-sample tests' (``tests/gpu/sm90/test_bias_only_dit_psb_gpu.py``)."""

import math

import pytest
import torch
from portable_helpers import randomize_half_std as randomize
from settings_helpers import policy_portable_off as policy  # noqa: F401

from miniworld_engine.integrations import bias_only_dit as bi
from miniworld_engine.integrations import bias_only_dit_train as bt
from miniworld_engine.kernels.bias_only_dit import cuda as C
from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="Blackwell (sm_100) required"
    ),
]
LAYOUTS = [(16, 48), (24, 32), (12, 64), (16, 64)]


def rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def cosine(a, b):
    a, b = a.detach().double().flatten(), b.detach().double().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


# ------------------------------------------------------------------------------------------------------------------- the cores
def _planes(H, A, L, g):
    """Per-sample attention weights P [H, A, L, L] bf16 (a softmax per row, a key mask per sample; with A > 1 sample 1 has every key
    masked: uniform rows, as the module's finfo.min fill makes them) and that mask [A, L]."""
    logits = torch.randn(H, A, L, L, device="cuda", generator=g) * 2
    keep = torch.rand(A, L, device="cuda", generator=g) > 0.2
    keep[:, 0] = True
    if A > 1:
        keep[1] = False
    logits = logits.masked_fill(~keep.view(1, A, 1, L), torch.finfo(torch.float32).min)
    return torch.softmax(logits, -1).bfloat16(), keep


def _tf32():
    from miniworld_engine.kernels.bias_only_dit.cuda import tf32

    return tf32


@pytest.mark.parametrize(("nh", "dh"), LAYOUTS)
@pytest.mark.parametrize(("A", "L"), [(1, 128), (3, 384), (5, 768), (48, 384)])
@pytest.mark.parametrize("gate", [True, False], ids=["gate", "nogate"])
@pytest.mark.parametrize("f32", [False, True], ids=["bf16", "tf32"])
def test_pv_gate_inf_per_sample(nh, dh, A, L, gate, f32):
    """a_s = sigmoid(g_s) (P_{h,s} v_s) per head and sample (gate), or P_{h,s} v_s (the training dV form), against fp64."""
    g = torch.Generator(device="cuda").manual_seed(A * 1000 + L + nh)
    DA = nh * dh
    dt = torch.float32 if f32 else torch.bfloat16
    P, _ = _planes(nh, A, L, g)
    P = P.to(dt)
    vg = torch.randn(A * L, 2 * DA, device="cuda", generator=g).to(dt)
    out = torch.empty(A * L, DA, device="cuda", dtype=dt)
    if f32:
        core = _tf32().PvGateCoreTF32(torch.cuda.current_device(), nh=nh, dh=dh, psb=True)
    else:
        core = C.PvGateCore(torch.cuda.current_device(), nh=nh, dh=dh, psb=True)
    core(vg[:, :DA], P.view(nh * A * L, L), out, A, g=vg[:, DA:] if gate else None)
    v = vg[:, :DA].double().view(A, L, nh, dh).permute(2, 0, 1, 3)                       # [H, A, L, dh]
    o = torch.einsum("haij,hajd->haid", P.double(), v).permute(1, 2, 0, 3).reshape(A * L, DA)
    if gate:
        o = o * torch.sigmoid(vg[:, DA:].double())
    assert rel(out, o) < (2e-3 if f32 else 1e-2), rel(out, o)
    again = torch.empty_like(out)
    core(vg[:, :DA], P.view(nh * A * L, L), again, A, g=vg[:, DA:] if gate else None)
    assert torch.equal(out, again)


@pytest.mark.parametrize(("nh", "dh"), LAYOUTS)
@pytest.mark.parametrize(("A", "L"), [(1, 128), (3, 384), (5, 768), (48, 256)])
@pytest.mark.parametrize("mask_kind", ["none", "shared", "per_sample"])
@pytest.mark.parametrize("f32", [False, True], ids=["bf16", "tf32"])
def test_dpb_per_sample(nh, dh, A, L, mask_kind, f32):
    """dbias_{h,a} = P_{h,a} o (do_a v_a^T - D_{h,a}) per sample (nothing summed over the samples), 0 on the masked keys (the
    module's masked_fill: also where a fully masked sample's P is uniform), against fp64."""
    g = torch.Generator(device="cuda").manual_seed(A * 1000 + L + nh + 1)
    DA = nh * dh
    dt = torch.float32 if f32 else torch.bfloat16
    P, keep = _planes(nh, A, L, g)
    P = P.to(dt)
    km = None if mask_kind == "none" else keep[0].contiguous() if mask_kind == "shared" else keep
    do = torch.randn(A * L, DA, device="cuda", generator=g).to(dt)
    vg = torch.randn(A * L, 2 * DA, device="cuda", generator=g).to(dt)
    dd = torch.randn(A, nh, L, device="cuda", generator=g)
    out = torch.empty(nh * A * L, L, device="cuda", dtype=dt)
    if f32:
        k = _tf32().Dpb32(torch.cuda.current_device(), nh=nh, dh=dh, psb=True)
    else:
        k = C.DpbKernel(torch.cuda.current_device(), nh=nh, dh=dh, psb=True)
    k(do, vg[:, :DA], P.view(nh * A * L, L), dd, out, A, mask=km)
    dO = do.double().view(A, L, nh, dh).permute(2, 0, 1, 3)
    v = vg[:, :DA].double().view(A, L, nh, dh).permute(2, 0, 1, 3)
    dP = torch.einsum("haid,hajd->haij", dO, v) - dd.double().permute(1, 0, 2)[..., None]
    want = P.double() * dP
    if km is not None:
        want = want * km.reshape(1, -1, 1, L).double()
        assert (out.view(nh, A, L, L) * ~km.reshape(1, -1, 1, L) == 0).all(), "masked keys carry no bias gradient"
    tol = 2e-3 if f32 else 1e-2
    assert rel(out.view(nh, A, L, L), want) < tol, rel(out.view(nh, A, L, L), want)
    for a in range(min(A, 3)):
        assert rel(out.view(nh, A, L, L)[:, a], want[:, a]) < tol
    again = torch.empty_like(out)
    k(do, vg[:, :DA], P.view(nh * A * L, L), dd, again, A, mask=km)
    assert torch.equal(out, again)


# ------------------------------------------------------------------------------------------------------------------- the block
def make_pair(nh=16, dh=48, seed=7):
    torch.manual_seed(seed)
    ref = randomize(BiasOnlyDiTBlock(n_head=nh, d_head=dh, implementation=ImplementationType.PYTORCH).cuda().float())
    eng = BiasOnlyDiTBlock(n_head=nh, d_head=dh, implementation=ImplementationType.MINIWORLD).cuda().float()
    eng.load_state_dict(ref.state_dict())
    return ref, eng


def make_inputs(A, L, mask_kind, seed=8):
    g = torch.Generator(device="cuda").manual_seed(seed)
    single = torch.randn(A, L, 768, device="cuda", generator=g)
    cond = torch.randn(A, L, 384, device="cuda", generator=g)
    pair = torch.randn(A, L, L, 128, device="cuda", generator=g)
    if mask_kind == "none":
        mask = None
    elif mask_kind == "shared":
        mask = (torch.rand(1, L, device="cuda", generator=g) > 0.15).expand(A, L).contiguous()
    else:
        mask = torch.rand(A, L, device="cuda", generator=g) > 0.15
        if mask_kind == "one_all_masked":  # sample 1 has no valid key: uniform rows, as the module's finfo.min fill makes them
            mask[1] = False
    w = torch.randn(A, L, 768, device="cuda", generator=g)
    return single, cond, pair, mask, w


def run(m, single, cond, pair, mask, w, dtype, autocast=False):
    """Loss = <out, w>; the module on [1, A, L, d] (the samples in the batch axis: each reads its own pair row). PyTorch matmuls in IEEE
    fp32 (the engine's fp32 path forces TF32 itself)."""
    A, L = single.shape[:2]
    ins = [t.detach().clone().to(dtype).requires_grad_() for t in (single, cond, pair)]
    m.zero_grad(set_to_none=True)
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out = m(ins[0].view(1, A, L, -1), ins[1].view(1, A, L, -1), ins[2], mask)
        loss = (out.float() * w.view(1, A, L, -1)).sum()
        loss.backward()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    res = {"loss": loss.detach().double(), "out": out.detach().float(), "dsingle": ins[0].grad.float(),
           "dcond": ins[1].grad.float(), "dpair": ins[2].grad.float()}
    res.update({"p." + n: p.grad.float().clone() for n, p in m.named_parameters() if p.grad is not None})
    return res


class Spy:
    """Counts the calls that reach ``target`` of ``module``."""

    def __init__(self, monkeypatch, module=bt, target="block_psb"):
        self.n = 0
        real = getattr(module, target)

        def counted(*a, **k):
            self.n += 1
            return real(*a, **k)

        monkeypatch.setattr(module, target, counted)


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
    (2, 256, "per_sample"),
    (5, 384, "shared"),
    (4, 384, "one_all_masked"),
    (48, 384, "per_sample"),
    (3, 768, "none"),
    (5, 768, "per_sample"),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize(("A", "L", "mask_kind"), CASES)
def test_per_sample_bias_training_matches_the_fp32_module_and_is_bitwise_repeatable(A, L, mask_kind, dtype, monkeypatch):
    ref, eng = make_pair()
    single, cond, pair, mask, w = make_inputs(A, L, mask_kind)
    truth = run(ref, single, cond, pair, mask, w, torch.float32)
    spy = Spy(monkeypatch)
    bf = dtype is torch.bfloat16
    got = run(eng, single, cond, pair, mask, w, dtype, autocast=bf)
    assert spy.n == 1, "the per-sample fused block did not serve the call"
    again = run(eng, single, cond, pair, mask, w, dtype, autocast=bf)
    for k in got:
        assert torch.isfinite(got[k]).all(), k
        assert torch.equal(got[k], again[k]), f"{k}: the step is not bitwise repeatable"
    assert set(got) == set(truth), sorted(set(truth) ^ set(got))
    m = metrics(got, truth)
    tol = 8e-3 if bf else 2.5e-3  # the PyTorch bf16 module measures ~4e-3 on this recipe; TF32 ~3e-4 (the Hopper bars)
    assert m["loss_rel"] < tol, m
    assert m["out_rel"] < tol, m
    assert m["whole_cos"] > 1 - (8e-4 if bf else 2e-4), m
    assert m["worst_cos"] > 1 - (4 * tol) ** 2 / 2 * 4, m
    for k in ("dsingle", "dcond", "dpair"):
        assert rel(got[k], truth[k]) < 3 * tol, (k, rel(got[k], truth[k]))
    if mask_kind == "one_all_masked":  # the fully masked sample alone: its output and gradients follow the module's uniform rows
        for k in ("out", "dsingle", "dpair"):
            g1, t1 = got[k].reshape(A, -1)[1], truth[k].reshape(A, -1)[1]
            assert rel(g1, t1) < 3 * tol, (k, rel(g1, t1))


@pytest.mark.parametrize(("nh", "dh"), LAYOUTS[1:])
def test_other_head_layouts(nh, dh, monkeypatch):
    ref, eng = make_pair(nh, dh)
    single, cond, pair, mask, w = make_inputs(3, 256, "per_sample")
    truth = run(ref, single, cond, pair, mask, w, torch.float32)
    spy = Spy(monkeypatch)
    got = run(eng, single, cond, pair, mask, w, torch.bfloat16, autocast=True)
    assert spy.n == 1
    m = metrics(got, truth)
    assert m["out_rel"] < 8e-3 and m["whole_cos"] > 1 - 8e-4, m


@pytest.mark.parametrize(("nh", "dh"), LAYOUTS[1:])
def test_other_head_layouts_fp32(nh, dh, monkeypatch):
    ref, eng = make_pair(nh, dh)
    single, cond, pair, mask, w = make_inputs(3, 256, "per_sample")
    truth = run(ref, single, cond, pair, mask, w, torch.float32)
    spy = Spy(monkeypatch)
    got = run(eng, single, cond, pair, mask, w, torch.float32)
    assert spy.n == 1
    m = metrics(got, truth)
    assert m["out_rel"] < 2.5e-3 and m["whole_cos"] > 1 - 2e-4, m


def _stack_run(blocks, single, cond, pair, mask, w, hoisted):
    """Blocks chained on [1, A, L, d] (bf16 activations, fp32 master under autocast); the biases from ONE ``pair_bias_all`` (hoisted) or
    each block's own pair."""
    from miniworld_engine.modules.bias_only_dit import pair_bias_all

    A, L = single.shape[:2]
    dt = torch.bfloat16 if hoisted else torch.float32
    ins = [t.detach().clone().to(dt).requires_grad_() for t in (single, cond, pair)]
    for b in blocks:
        b.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=hoisted):
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


@pytest.mark.parametrize(("A", "L", "mask_kind"), [(2, 256, "per_sample"), (5, 384, "none")])
def test_hoisted_per_sample_bias_over_a_stack(A, L, mask_kind, monkeypatch):
    """Three blocks on biases from one ``pair_bias_all`` (the hoist's head-major buffers, the bias gradients written into its slots):
    against the fp32 PyTorch stack, each block on its own pair; bitwise repeatable; the gradients reach pair / ln_pair / to_bias."""
    refs, engs = zip(*[make_pair(seed=100 + i) for i in range(3)], strict=True)
    single, cond, pair, mask, w = make_inputs(A, L, mask_kind)
    truth = _stack_run(list(refs), single, cond, pair, mask, w, hoisted=False)
    spy = Spy(monkeypatch)
    got = _stack_run(list(engs), single, cond, pair, mask, w, hoisted=True)
    assert spy.n == 3
    again = _stack_run(list(engs), single, cond, pair, mask, w, hoisted=True)
    assert all(torch.equal(got[k], again[k]) for k in got), "not bitwise repeatable"
    assert set(got) == set(truth), sorted(set(truth) ^ set(got))
    m = metrics(got, truth)
    assert m["out_rel"] < 1.2e-2 and m["whole_cos"] > 1 - 1e-3, m


def test_training_step_cuda_graph_replays_the_eager_bits():
    ref, eng = make_pair()
    A, L = 4, 256
    single, cond, pair, mask, w = make_inputs(A, L, "per_sample")
    want = run(eng, single, cond, pair, mask, w, torch.bfloat16, autocast=True)
    ins = [t.detach().clone().bfloat16().requires_grad_() for t in (single, cond, pair)]

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = eng(ins[0].view(1, A, L, -1), ins[1].view(1, A, L, -1), ins[2], mask)
        (out.float() * w.view(1, A, L, -1)).sum().backward()
        return out

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            for t in (*ins, *eng.parameters()):
                t.grad = None
            step()
    torch.cuda.current_stream().wait_stream(stream)
    for t in (*ins, *eng.parameters()):
        t.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.detach().float(), want["out"])
    assert torch.equal(ins[0].grad.float(), want["dsingle"]) and torch.equal(ins[2].grad.float(), want["dpair"])
    for n, p in eng.named_parameters():
        if p.grad is not None:
            assert torch.equal(p.grad.float(), want["p." + n]), n


@pytest.mark.parametrize(("A", "L", "mask_kind"), [(2, 256, "per_sample"), (5, 384, "shared"), (4, 384, "one_all_masked"),
                                                   (5, 768, "per_sample"), (3, 768, "none")])
@pytest.mark.parametrize(("nh", "dh"), [(16, 48), (24, 32)])
def test_per_sample_bias_inference(A, L, mask_kind, nh, dh, monkeypatch):
    """Inference (no autograd) on a pair of its own per sample: the per-sample softmax + the -DPSB core inside the runner's step,
    against the fp32 module; within the bf16 PyTorch module's error x 1.3 (+ a floor); bitwise repeatable."""
    ref, eng = make_pair(nh, dh)
    single, cond, pair, mask, _ = make_inputs(A, L, mask_kind)
    s, c = single.view(1, A, L, -1), cond.view(1, A, L, -1)
    spy = Spy(monkeypatch, bi, "update_psb")
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            truth = ref(s, c, pair, mask)
            got32 = eng(s, c, pair, mask)  # fp32: the TF32 -DPSB core
            with torch.autocast("cuda", dtype=torch.bfloat16):
                got = eng(s.bfloat16(), c.bfloat16(), pair.bfloat16(), mask)
                again = eng(s.bfloat16(), c.bfloat16(), pair.bfloat16(), mask)
            tb = ref.bfloat16()(s.bfloat16(), c.bfloat16(), pair.bfloat16(), mask)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    assert spy.n == 3, "the per-sample inference step did not serve the call"
    assert torch.isfinite(got32).all() and rel(got32, truth) < 2.5e-3, rel(got32, truth)
    assert torch.isfinite(got).all()
    assert torch.equal(got, again)
    e, e0 = rel(got, truth), rel(tb, truth)
    assert e < 1.3 * e0 + 2e-3, (e, e0)
    if mask_kind == "one_all_masked":
        assert rel(got[0, 1], truth[0, 1]) < 1.3 * rel(tb[0, 1], truth[0, 1]) + 2e-3


def test_softmax_t_all_masked_sample_is_uniform_and_finite():
    """The per-sample softmax with one sample's keys all masked (and -1e30 already in its logits): uniform rows, no NaN."""
    H, A, L = 16, 3, 256
    g = torch.Generator(device="cuda").manual_seed(3)
    bias = torch.randn(H, A, L, L, device="cuda", generator=g)
    mask = torch.rand(A, L, device="cuda", generator=g) > 0.2
    mask[1] = False
    bias[:, 1] = -1e30
    b = bias.bfloat16()
    p = torch.empty(H, A, L, L, device="cuda", dtype=torch.bfloat16)
    pt = torch.empty_like(p)
    C.softmax_t(b.view(-1, L), p.view(-1, L), pt.view(-1, L), mask, A)
    assert torch.isfinite(p).all()
    assert torch.equal(p[:, 1], torch.full_like(p[:, 1], 1.0 / L))
    want = torch.softmax(b.float().masked_fill(~mask.view(1, A, 1, L), torch.finfo(torch.float32).min), -1)
    assert rel(p, want) < 5e-3
    assert torch.equal(pt, p.transpose(-1, -2))
    assert math.isclose(float(p[:, 0].float().sum(-1).mean()), 1.0, rel_tol=1e-2)
