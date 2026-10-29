"""The Boltz-2 / RoseTTAFold3 / Chai-1 DiT variant stacks (``modules.dit_variant``, token and atom) on B200 (sm_100a): inference (bf16
``interface.py``, fp32 ``cuda/infer32.py``) and training (``cuda/train.py``) on the B200 attention cores (``cuda/sm100.py``: the token DiT's
attn_inf / attn_fwd2 / attn_dkv / attn_dqb and their TF32 twins, the local DiT's lattn / lslide / lbias), against the fp64 PyTorch reference
with the H100 tests' bars: inference within max(1.5 x the bf16 reference's error, 1e-2) (fp32: 1.5 x the fp32 reference's); training the
output and every gradient within 1.5 x the same-dtype PyTorch reference's error against fp64, a tensor whose true gradient is zero (the k
LayerNorm bias, the pair LayerNorm bias) held to an absolute floor."""

import copy

import pytest
import torch

from miniworld_engine.kernels.dit_variants import interface
from miniworld_engine.kernels.dit_variants.cuda import infer32, train
from miniworld_engine.modules.dit_variant import (
    BOLTZ2,
    BOLTZ2_ATOM,
    CHAI1,
    CHAI1_ATOM,
    ROSETTAFOLD3,
    ROSETTAFOLD3_ATOM,
    AtomDiTVariantStack,
    DiTVariantStack,
)
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
        reason="B200 (sm_100) required",
    ),
]

TOKEN = {"boltz2": BOLTZ2, "rf3": ROSETTAFOLD3, "chai1": CHAI1}
ATOM = {"boltz2": BOLTZ2_ATOM, "rf3": ROSETTAFOLD3_ATOM, "chai1": CHAI1_ATOM}
ERR_RATIO = 1.5
FLOOR = 1e-4


def _randomize(stack):
    torch.manual_seed(0)
    with torch.no_grad():
        for name, p in stack.named_parameters():
            if "norm" in name and p.ndim == 1:
                p.copy_(1 + 0.1 * torch.randn_like(p) if name.endswith("weight") else 0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else torch.randn_like(p) * 0.3)
    return stack


def _rel(a, b, rows=None):
    if rows is not None:
        a, b = a[:, rows], b[:, rows]
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item()


# ------------------------------------------------------------------------------------------------------------- bf16 inference
@pytest.mark.parametrize("model", list(TOKEN))
@pytest.mark.parametrize(("a", "n", "real"), [(5, 256, 256), (2, 384, 300), (1, 128, 100), (3, 200, 190), (5, 768, 700)])
def test_token_inference_matches_fp64(model, a, n, real):
    options = TOKEN[model]
    stack = _randomize(DiTVariantStack(3, **options).cuda())
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(a, n, 768, device="cuda", generator=g)
    s = torch.randn(n, options["d_cond"], device="cuda", generator=g)
    z = torch.randn(n, n, options["d_pair"], device="cuda", generator=g)
    mask = torch.arange(n, device="cuda") < real
    ref64 = copy.deepcopy(stack).double()
    ref64.implementation = ImplementationType.PYTORCH
    ref16 = copy.deepcopy(stack).to(torch.bfloat16)
    ref16.implementation = ImplementationType.PYTORCH
    eng = stack.to(torch.bfloat16)
    bf = [t.to(torch.bfloat16) for t in (x, s, z)]
    with torch.no_grad():
        assert interface.serves(eng, *bf, mask)
        gold = ref64(x.double(), s.double(), z.double(), mask)
        base = ref16(*bf, mask)
        got = eng(*bf, mask)
    assert got.shape == gold.shape and torch.isfinite(got).all()
    rows = slice(0, real)
    err, err16 = _rel(got, gold, rows), _rel(base, gold, rows)
    assert err <= max(ERR_RATIO * err16, 1e-2), f"{model}: engine {err:.2e} vs bf16 reference {err16:.2e}"


def test_token_inference_reuses_pair_bias_and_pack():
    stack = _randomize(DiTVariantStack(3, **CHAI1).cuda()).to(torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(2, 256, 768, device="cuda", generator=g, dtype=torch.bfloat16)
    s = torch.randn(256, 384, device="cuda", generator=g, dtype=torch.bfloat16)
    z = torch.randn(256, 256, 256, device="cuda", generator=g, dtype=torch.bfloat16)
    with torch.no_grad():
        first = stack(x, s, z, None)
        second = stack(x, s, z, None)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


# (A, N, real atoms): full; ragged; under 128 real atoms; N not a multiple of 128 (padded inside)
@pytest.mark.parametrize("model", list(ATOM))
@pytest.mark.parametrize("key_window", ["slide", "pad"])
@pytest.mark.parametrize(("a", "n", "real"), [(5, 1024, 1024), (2, 768, 700), (1, 256, 100), (3, 544, 544), (2, 1600, 1500)])
def test_atom_inference_matches_fp64(model, key_window, a, n, real):
    stack = _randomize(AtomDiTVariantStack(3, key_window=key_window, **ATOM[model]).cuda())
    g = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(a, n, 128, device="cuda", generator=g)
    c = torch.randn(n, 128, device="cuda", generator=g)
    z = torch.randn(n // 32, 32, 128, 16, device="cuda", generator=g)
    mask = torch.arange(n, device="cuda") < real
    uid = (torch.arange(n, device="cuda") // 37) if model == "chai1" else None
    ref64 = copy.deepcopy(stack).double()
    ref64.implementation = ImplementationType.PYTORCH
    ref16 = copy.deepcopy(stack).to(torch.bfloat16)
    ref16.implementation = ImplementationType.PYTORCH
    eng = stack.to(torch.bfloat16)
    bf = [t.to(torch.bfloat16) for t in (x, c, z)]
    with torch.no_grad():
        assert interface.atom_serves(eng, *bf, mask, uid)
        gold = ref64(x.double(), c.double(), z.double(), mask, uid)
        base = ref16(*bf, mask, uid)
        got = eng(*bf, mask, uid)
    assert got.shape == gold.shape
    assert torch.isfinite(got[:, :real]).all()
    rows = slice(0, real)
    err, err16 = _rel(got, gold, rows), _rel(base, gold, rows)
    assert err <= max(ERR_RATIO * err16, 1e-2), f"{model} {key_window}: engine {err:.2e} vs bf16 reference {err16:.2e}"


def test_atom_inference_without_mask():
    """No key mask, every atom real: the slid edge windows at both sequence ends."""
    stack = _randomize(AtomDiTVariantStack(2, key_window="slide", **BOLTZ2_ATOM).cuda())
    g = torch.Generator(device="cuda").manual_seed(3)
    x, c, z = (torch.randn(*s, device="cuda", generator=g) for s in ((3, 512, 128), (512, 128), (16, 32, 128, 16)))
    ref64 = copy.deepcopy(stack).double()
    ref64.implementation = ImplementationType.PYTORCH
    ref16 = copy.deepcopy(stack).to(torch.bfloat16)
    ref16.implementation = ImplementationType.PYTORCH
    eng = stack.to(torch.bfloat16)
    bf = [t.to(torch.bfloat16) for t in (x, c, z)]
    with torch.no_grad():
        gold = ref64(x.double(), c.double(), z.double())
        err, err16 = _rel(eng(*bf), gold), _rel(ref16(*bf), gold)
    assert err <= max(ERR_RATIO * err16, 1e-2), (err, err16)


# ------------------------------------------------------------------------------------------------------------------- training
def _init(stack):
    torch.manual_seed(0)
    with torch.no_grad():
        for p in stack.parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[1] ** -0.5)
            elif p.ndim == 1 and p.numel() > 1:
                p.add_(torch.randn_like(p) * 0.1)
    return stack.cuda()


def _case(kind, model, a, length, *, masked=True, window="slide", atoms=None):
    """(stack, inputs, mask, uid, dy, live rows). ``masked``: a tenth of the tokens / atoms are padding (else no mask at all)."""
    g = torch.Generator(device="cuda").manual_seed(1)
    if kind == "atom":
        n = atoms or length * 8
        stack = _init(AtomDiTVariantStack(2, 128, 128, 16, 2, key_window=window, **ATOM[model]))
        shapes = {"single": (a, n, 128), "cond": (n, 128), "pair": (n // 32, 32, 128, 16)}
        live = int(n * 0.9) if masked else n
        mask = torch.arange(n, device="cuda") < live if masked else None
        uid = (torch.arange(n, device="cuda") // 45).int() if model == "chai1" else None
    else:
        o = TOKEN[model]
        n, live = length, length - 40 if masked else length
        stack = _init(DiTVariantStack(2, 768, o["d_cond"], o["d_pair"], 16, 2, **{k: v for k, v in o.items() if k not in ("d_cond", "d_pair")}))
        shapes = {"single": (a, n, 768), "cond": (n, o["d_cond"]), "pair": (n, n, o["d_pair"])}
        mask = torch.arange(n, device="cuda") < live if masked else None
        uid = None
    inputs = {k: torch.randn(*s, device="cuda", generator=g) for k, s in shapes.items()}
    dy = torch.randn(*shapes["single"], device="cuda", generator=g)
    if kind == "atom":
        dy[:, live:] = 0      # a padding atom's output is not read (the slid windows past the real atoms differ from the reference's there)
    return stack, inputs, mask, uid, dy, live


def _run(kind, stack, inputs, mask, uid, dy, live, dtype, bf16, impl):
    m = copy.deepcopy(stack)
    m.implementation = impl
    ins = {k: v.clone().to(dtype).requires_grad_() for k, v in inputs.items()}
    args = (ins["single"], ins["cond"], ins["pair"], mask) + ((uid,) if kind == "atom" else ())
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        if impl == ImplementationType.MINIWORLD and dtype != torch.float64:
            assert (train.atom_serves if kind == "atom" else train.serves)(m, *args), "the CUDA training route must serve this call"
        y = m(*args)
    y.backward(dy.to(y.dtype))
    grads = {k: (v.grad if v.grad is not None else torch.zeros_like(v)).double() for k, v in ins.items()}
    grads["single"] = grads["single"][:, :live]
    for name, p in m.named_parameters():
        grads["P." + name] = (p.grad if p.grad is not None else torch.zeros_like(p)).double()
    return y.double()[:, :live], grads


def _check(kind, model, precision, a=6, length=256, **case):
    stack, inputs, mask, uid, dy, live = _case(kind, model, a, length, **case)
    bf16 = precision == "bf16"
    dtype = torch.bfloat16 if bf16 else torch.float32
    saved = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        y64, g64 = _run(kind, stack.double(), {k: v.double() for k, v in inputs.items()}, mask, uid, dy.double(), live, torch.float64, False,
                        ImplementationType.PYTORCH)
        torch.backends.cuda.matmul.allow_tf32 = not bf16
        stack = stack.float()
        yr, gr = _run(kind, stack, inputs, mask, uid, dy, live, dtype, bf16, ImplementationType.PYTORCH)
        ye, ge = _run(kind, stack, inputs, mask, uid, dy, live, dtype, bf16, ImplementationType.MINIWORLD)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved
    assert torch.isfinite(ye).all()
    assert _rel(ye, y64) <= ERR_RATIO * _rel(yr, y64), ("output", _rel(ye, y64), _rel(yr, y64))
    floor = FLOOR * max(g.norm().item() for g in g64.values())
    bad = []
    for name, ref in g64.items():
        err_e, err_r = (ge[name] - ref).norm().item(), (gr[name] - ref).norm().item()
        if not (err_e <= ERR_RATIO * err_r + floor):
            bad.append((name, err_e / max(ref.norm().item(), 1e-30), err_r / max(ref.norm().item(), 1e-30)))
    assert not bad, bad


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
@pytest.mark.parametrize("model", list(TOKEN))
@pytest.mark.parametrize("kind", ["token", "atom"])
def test_training_matches_fp64(kind, model, precision):
    _check(kind, model, precision)


@pytest.mark.parametrize(
    ("kind", "model", "case"),
    [
        ("token", "rf3", {"masked": False}),                        # no key mask
        ("token", "boltz2", {"a": 3}),                              # odd A: one zero sample padded in (the forward pairs the samples)
        ("token", "chai1", {"length": 384}),
        ("token", "rf3", {"a": 16}),                                # M = A L >= 3840: the SwiGLU in the expand GEMM (-DSAVE_AB)
        ("token", "chai1", {"a": 16, "length": 384}),
        ("atom", "boltz2", {"masked": False}),                      # every atom real: the slid edge windows at the sequence ends
        ("atom", "chai1", {"window": "pad"}),                       # the padded key window, same-conformer mask, padding atoms
        ("atom", "rf3", {"masked": False, "window": "pad"}),
        ("atom", "chai1", {"atoms": 1600}),                         # N not a multiple of 128: padded inside
        ("atom", "rf3", {"atoms": 1600, "masked": False}),
    ],
)
def test_training_matches_fp64_other_options(kind, model, case):
    _check(kind, model, "bf16", **case)


@pytest.mark.parametrize("kind", ["token", "atom"])
@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_training_is_bitwise_repeatable(kind, precision):
    """The row kernels sum in a fixed order and so do the sm_100a attention backwards (the token dQ as per-chunk partials summed in order:
    L 384 has three bf16 / six TF32 chunks): two runs agree exactly."""
    stack, inputs, mask, uid, dy, live = _case(kind, "rf3", 4, 384 if kind == "token" else 256)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    runs = [_run(kind, stack, inputs, mask, uid, dy, live, dtype, precision == "bf16", ImplementationType.MINIWORLD) for _ in range(2)]
    assert torch.equal(runs[0][0], runs[1][0])
    for name in runs[0][1]:
        assert torch.equal(runs[0][1][name], runs[1][1][name]), name


def test_routing_on_b200():
    stack, inputs, mask, *_ = _case("token", "boltz2", 3, 256)
    single, cond, pair = (inputs[k].bfloat16() for k in ("single", "cond", "pair"))
    with torch.no_grad():
        assert not train.serves(stack, single, cond, pair, mask)
        assert interface.serves(stack.bfloat16(), single, cond, pair, mask)
    single.requires_grad_()
    assert train.serves(stack, single, cond, pair, mask)
    assert not train.serves(stack, single[:, :200], cond[:200], pair[:200, :200], mask[:200])      # L not a multiple of 128
    assert not train.serves(stack, single, cond[None].expand(3, -1, -1), pair, mask)               # a conditioning row set per sample


# ---------------------------------------------------------------------------------------------------------------- fp32 inference
@pytest.mark.parametrize("model", list(TOKEN))
@pytest.mark.parametrize("kind", ["token", "atom"])
@pytest.mark.parametrize("masked", [True, False])
def test_fp32_inference_matches_fp64(kind, model, masked):
    stack, inputs, mask, uid, _, live = _case(kind, model, 5, 256, masked=masked)
    args32 = [inputs[k].float() for k in ("single", "cond", "pair")]
    tail = (mask,) + ((uid,) if kind == "atom" else ())
    saved = torch.backends.cuda.matmul.allow_tf32
    try:
        with torch.no_grad():
            assert (infer32.atom_serves if kind == "atom" else infer32.serves)(stack, *args32, *tail)
            torch.backends.cuda.matmul.allow_tf32 = False
            ref64 = copy.deepcopy(stack).double()
            ref64.implementation = ImplementationType.PYTORCH
            y64 = ref64(*[t.double() for t in args32], *tail)
            torch.backends.cuda.matmul.allow_tf32 = True
            ref32 = copy.deepcopy(stack)
            ref32.implementation = ImplementationType.PYTORCH
            yr = ref32(*args32, *tail)
            ye = stack(*args32, *tail)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved
    rows = slice(0, live)
    assert torch.isfinite(ye).all()
    assert _rel(ye, y64, rows) <= ERR_RATIO * _rel(yr, y64, rows), (_rel(ye, y64, rows), _rel(yr, y64, rows))


def test_inference_in_a_cuda_graph():
    """bf16 token and atom inference captured once and replayed with new single / cond values: the replay equals an eager call."""
    tok = _randomize(DiTVariantStack(2, **ROSETTAFOLD3).cuda()).to(torch.bfloat16)
    atom = _randomize(AtomDiTVariantStack(2, key_window="slide", **CHAI1_ATOM).cuda()).to(torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(4)
    bf = dict(device="cuda", dtype=torch.bfloat16)
    x, s, z = torch.randn(3, 256, 768, **bf), torch.randn(256, 384, **bf), torch.randn(256, 256, 128, **bf)
    xa, ca, za = torch.randn(3, 1024, 128, **bf), torch.randn(1024, 128, **bf), torch.randn(32, 32, 128, 16, **bf)
    mask, amask = torch.arange(256, device="cuda") < 230, torch.arange(1024, device="cuda") < 1000
    uid = torch.arange(1024, device="cuda") // 37
    with torch.no_grad():
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                tok(x, s, z, mask)
                atom(xa, ca, za, amask, uid)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            yt = tok(x, s, z, mask)
            ya = atom(xa, ca, za, amask, uid)
        for t in (x, s, xa, ca):
            t.copy_(torch.randn(t.shape, generator=g, device="cuda").to(t.dtype))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(yt, tok(x, s, z, mask), rtol=0, atol=0)
        torch.testing.assert_close(ya, atom(xa, ca, za, amask, uid), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["token", "atom"])
def test_under_torch_compile(kind):
    """A torch.compile'd stack (FoldForge compiles its denoiser; the benchmark harness compiles every rung) takes the same B200 path: the
    B200 stacks run outside Dynamo (a graph break), so the compiled call equals the eager one bitwise, training and bf16 inference."""
    stack, inputs, mask, uid, dy, live = _case(kind, "chai1", 4, 256)
    tail = (mask,) + ((uid,) if kind == "atom" else ())
    results = []
    for compiled in (False, True):
        m = copy.deepcopy(stack)
        f = torch.compile(m) if compiled else m
        ins = {k: v.clone().requires_grad_() for k, v in inputs.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = f(ins["single"], ins["cond"], ins["pair"], *tail)
        y.backward(dy.to(y.dtype))
        mb = copy.deepcopy(stack).bfloat16()
        fb = torch.compile(mb) if compiled else mb
        with torch.no_grad():
            yi = fb(*(inputs[k].bfloat16() for k in ("single", "cond", "pair")), *tail)
        results.append((y.detach(), ins["single"].grad, ins["pair"].grad, yi))
    for a, b in zip(*results, strict=True):
        assert torch.equal(a, b)
