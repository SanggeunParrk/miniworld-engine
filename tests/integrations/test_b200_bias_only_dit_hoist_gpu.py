"""The hoisted pair bias of the bias-only token DiT on B200: ``pair_bias_all`` (modules/bias_only_dit/hoist.py,
integrations/bias_only_dit_hoist.py) and the blocks' ``bias=`` paths (integrations/bias_only_dit_train.py, bias_only_dit.py).

  * ``pair_bias_all`` forward and backward (d pair, every dW_b, every d gamma_b) against fp64 for 1 / 3 / 24 blocks, every head
    layout, bf16 and fp32 (TF32); the PyTorch fold the same; unused biases; the LN0 backward rows against fp64, no spills;
  * a stack of blocks hoisted (fused) against the stack in fp64, every gradient within the training files' bounds (bf16: 1.1x the
    PyTorch bf16 stack's error + 1e-4; fp32: 1.5x the PyTorch TF32 stack's + 1e-3), with and without a mask, L384 / L768;
  * inference on hoisted biases against the per-pair path; checkpointing (non-reentrant) bit-identical; 20 poisoned-allocator steps
    bit-identical; the hoisted block's launches (no pair_bias / pair_bias_bwd, one finalize); steady memory and a captured step.

Tolerances of the direct ``pair_bias_all`` checks (fp64 reference on the same rounded inputs): bf16 rounds LN0 and W' = W diag(g)
once each (unit roundoff 2^-9) and the bias once more on output -- ~2-3e-3 relative, bound 6e-3 (d pair 8e-3: d LN0 carries W''s
rounding, d pair its own output rounding; dW_b 6e-3: its bf16 output rounding); d gamma_b (fp32) 1e-4: dW' is the bf16 d bias against
LN0 split in two bf16 terms (~2^-17) with fp32 accumulation and no rounding after it (hoist1 rounded LN0 to bf16 there). fp32 runs the
GEMMs on TF32 (2^-11 per operand), bound 3e-3 (the TF32 file's kernel bound)."""

import contextlib
import copy

import pytest
import torch
import torch.nn.functional as Fn

from miniworld_engine import settings
from miniworld_engine.integrations import bias_only_dit as INF
from miniworld_engine.integrations import bias_only_dit_hoist as HO
from miniworld_engine.integrations import bias_only_dit_train as TR
from miniworld_engine.modules.bias_only_dit import BiasOnlyAttention, BiasOnlyDiTBlock, pair_bias_all
from miniworld_engine.modules.bias_only_dit import hoist as hoist_mod
from miniworld_engine.modules.bias_only_dit import module as bo_module
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

LAYOUTS = [(16, 48), (24, 32), (12, 64), (16, 64)]
BF, F32, F64 = torch.bfloat16, torch.float32, torch.float64
MW, PT = ImplementationType.MINIWORLD, ImplementationType.PYTORCH


def relative(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


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


@contextlib.contextmanager
def tf32(on):
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = on
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


@pytest.fixture(autouse=True)
def policy():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Blackwell (sm_100) required")
    old = settings.configure(engine_backend="auto")
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        settings.configure(**vars(old))


@contextlib.contextmanager
def fused_only():
    """Neither the block's PyTorch composition nor the PyTorch fold of the hoist may run."""
    def boom(*a, **k):
        raise AssertionError("a PyTorch path ran")
    old_fwd, old_ref = bo_module.BiasOnlyAttention.forward, hoist_mod._reference
    bo_module.BiasOnlyAttention.forward = boom
    hoist_mod._reference = boom
    try:
        yield
    finally:
        bo_module.BiasOnlyAttention.forward = old_fwd
        hoist_mod._reference = old_ref


def _poison(mb=512):
    """Fill the caching allocator's next blocks with NaN: a buffer read before it is written would show it."""
    junk = torch.full((mb << 18,), float("nan"), device="cuda")
    torch.cuda.synchronize()
    del junk


# ------------------------------------------------------------------------------------------------ pair_bias_all alone
def attentions(nb, n_head, d_head, dtype, seed=5):
    torch.manual_seed(seed)
    mods = [randomize(BiasOnlyAttention(768, 384, 128, n_head, d_head)).cuda() for _ in range(nb)]
    return [m.to(dtype) for m in mods]           # bf16: to_bias bf16, ln_pair stays fp32 (the engine pins norm weights)


def hoist_ref64(attns, pair, dbs):
    """Each block's to_bias(ln_pair(pair)) [1, H, L, L] in fp64, and the gradients of sum_b <bias_b, db_b>."""
    p = pair.detach().double().requires_grad_(True)
    gs = [a.ln_pair.weight.detach().double().requires_grad_(True) for a in attns]
    ws = [a.to_bias.weight.detach().double().requires_grad_(True) for a in attns]
    ln = Fn.layer_norm(p, (128,), eps=1e-5)
    outs = [((ln * g) @ w.t()).permute(0, 3, 1, 2) for g, w in zip(gs, ws, strict=True)]
    torch.autograd.backward(outs, [d.double() for d in dbs])
    return [o.detach() for o in outs], p.grad, [g.grad for g in gs], [w.grad for w in ws]


def hoist_run(attns, pair, dbs, implementation=MW, use=None):
    """pair_bias_all and the gradients of sum_b <bias_b, db_b> (only the blocks in ``use``)."""
    p = pair.detach().clone().requires_grad_(True)
    for a in attns:
        a.zero_grad(set_to_none=True)
    outs = pair_bias_all(attns, p, implementation=implementation)
    idx = range(len(attns)) if use is None else use
    torch.autograd.backward([outs[i] for i in idx], [dbs[i] for i in idx])
    return [o.detach() for o in outs], p.grad, [a.ln_pair.weight.grad for a in attns], [a.to_bias.weight.grad for a in attns]


TOL = {BF: {"bias": 6e-3, "dpair": 8e-3, "dg": 1e-4, "dW": 6e-3}, F32: {"bias": 3e-3, "dpair": 3e-3, "dg": 3e-3, "dW": 3e-3}}


@pytest.mark.parametrize("dtype", [BF, F32])
@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize("nb", [1, 3, 24])
def test_pair_bias_all_against_fp64(nb, n_head, d_head, dtype):
    """The fused hoist: every block's bias (a contiguous head-major slice of one buffer), d pair, d gamma_b and dW_b against fp64."""
    L = 384
    attns = attentions(nb, n_head, d_head, dtype)
    g = torch.Generator(device="cuda").manual_seed(nb + n_head)
    pair = torch.randn(1, L, L, 128, device="cuda", generator=g).to(dtype)
    dbs = [torch.randn(1, n_head, L, L, device="cuda", generator=g).to(dtype) for _ in range(nb)]
    assert HO.serves(MW, attns, pair)
    with fused_only():
        outs, dp, dgs, dws = hoist_run(attns, pair, dbs)
    w_outs, w_dp, w_dgs, w_dws = hoist_ref64(attns, pair, dbs)
    tol = TOL[dtype]
    base = outs[0].untyped_storage().data_ptr()
    for i, (o, w) in enumerate(zip(outs, w_outs, strict=True)):
        assert o.dtype is dtype and o.shape == (1, n_head, L, L) and o.is_contiguous(), i
        assert o.data_ptr() == base + i * n_head * L * L * o.element_size(), i          # slice i of the one [nb H, L L] buffer
        assert relative(o, w) < tol["bias"], (i, relative(o, w))
    assert dp.dtype is dtype and relative(dp, w_dp) < tol["dpair"], relative(dp, w_dp)
    for i in range(nb):
        assert dgs[i].dtype is attns[i].ln_pair.weight.dtype and dws[i].dtype is attns[i].to_bias.weight.dtype, i
        assert relative(dgs[i], w_dgs[i]) < tol["dg"], (i, relative(dgs[i], w_dgs[i]))
        assert relative(dws[i], w_dws[i]) < tol["dW"], (i, relative(dws[i], w_dws[i]))


@pytest.mark.parametrize("dtype", [BF, F32])
def test_pair_bias_all_pytorch_fold_and_unused_biases(dtype):
    """The PyTorch fold (implementation PYTORCH, and the engine path with MINIWORLD_BIAS_ONLY_DIT_HOIST=0) against fp64, the same
    API; a backward through two of three biases: the third block's gradients are zero, the rest as fp64 with a zero d bias."""
    L, nb, H, DH = 256, 3, 16, 48
    attns = attentions(nb, H, DH, dtype, seed=9)
    g = torch.Generator(device="cuda").manual_seed(17)
    pair = torch.randn(1, L, L, 128, device="cuda", generator=g).to(dtype)
    dbs = [torch.randn(1, H, L, L, device="cuda", generator=g).to(dtype) for _ in range(nb)]
    w_outs, w_dp, w_dgs, w_dws = hoist_ref64(attns, pair, dbs)
    outs, dp, dgs, dws = hoist_run(attns, pair, dbs, implementation=PT)
    bound = 2e-2 if dtype is BF else 3e-3
    for o, w in zip(outs, w_outs, strict=True):
        assert o.shape == (1, H, L, L) and relative(o, w) < bound
    assert relative(dp, w_dp) < bound
    for a, b, c, d in zip(dgs, w_dgs, dws, w_dws, strict=True):
        assert relative(a, b) < bound and relative(c, d) < bound
    # unused bias 1, fused
    zero = [dbs[0], torch.zeros_like(dbs[1]), dbs[2]]
    w_outs, w_dp, w_dgs, w_dws = hoist_ref64(attns, pair, zero)
    with fused_only():
        outs, dp, dgs, dws = hoist_run(attns, pair, dbs, use=[0, 2])
    tol = TOL[dtype]
    assert relative(dp, w_dp) < tol["dpair"]
    assert torch.count_nonzero(dgs[1]) == 0 and torch.count_nonzero(dws[1]) == 0
    for i in (0, 2):
        assert relative(dgs[i], w_dgs[i]) < tol["dg"] and relative(dws[i], w_dws[i]) < tol["dW"], i


def test_serves_and_declines(monkeypatch):
    attns = attentions(2, 16, 48, BF)
    p = torch.randn(1, 256, 256, 128, device="cuda", dtype=BF)
    assert HO.serves(MW, attns, p) and HO.serves(ImplementationType.TRITON, attns, p)
    assert not HO.serves(PT, attns, p)                                                  # the PyTorch implementation: the fold
    assert not HO.serves(MW, attns, p.expand(2, -1, -1, -1))                             # B == 1 only
    assert not HO.serves(MW, attns, p.half())                                            # bf16 / fp32 only
    a32 = attentions(2, 16, 48, F32)
    assert HO.serves(MW, a32, p.float())
    assert not HO.serves(MW, attns, p.float())                                           # fp32 needs fp32 weights
    with torch.autocast("cuda", dtype=BF):
        assert not HO.serves(MW, a32, p.float())                                         # fp32 under autocast: the fold
    monkeypatch.setenv("MINIWORLD_BIAS_ONLY_DIT_HOIST", "0")
    assert not HO.serves(MW, attns, p)
    with pytest.raises(ValueError):
        pair_bias_all([*attns, *attentions(1, 24, 32, BF)], p, implementation=MW)       # head counts differ


@pytest.mark.parametrize("R", [16, 1000, 147456])
@pytest.mark.parametrize(("xdt", "odt"), [(BF, BF), (F32, F32), (BF, F32), (F32, BF)])
def test_ln0_backward_rows_against_fp64(R, xdt, odt):
    """ln0_bwd_rows: d x and the dW' operand y (bf16: [hi | lo], hi + lo = LN0 to ~2^-17; fp32: LN0) -- every row written
    (NaN-filled outputs; R off the 16-row block grain), d x within its output rounding of fp64, hi + lo within 3e-5, bit-identical on
    reruns; no local memory. y takes the path's dtype (that of x)."""
    from miniworld_engine.kernels.bias_only_dit.cuda import hoist as HK
    for name, regs, lmem in HK.kernel_attrs():
        assert lmem == 0 and regs <= 128, (name, regs, lmem)
    g = torch.Generator(device="cuda").manual_seed(R)
    x = (3 * torch.randn(R, 128, device="cuda", generator=g) + 0.5).to(xdt)
    dy = torch.randn(R, 128, device="cuda", generator=g)
    xd = x.double().requires_grad_(True)
    ln64 = Fn.layer_norm(xd, (128,), eps=1e-5)
    ln64.backward(dy.double())
    K = HK.operand_cols(xdt)
    outs = []
    for _ in range(3):
        dx = torch.full((R, 128), float("nan"), device="cuda", dtype=odt)
        y = torch.full((R, K), float("nan"), device="cuda", dtype=xdt)
        HK.ln0_bwd_rows(x, dy, dx, y, 1e-5)
        torch.cuda.synchronize()
        outs.append((dx, y))
    dx, y = outs[0]
    assert torch.isfinite(dx.float()).all() and torch.isfinite(y.float()).all()
    assert relative(dx, xd.grad) < (6e-3 if odt is BF else 1e-5), relative(dx, xd.grad)
    ln = y.double() if K == 128 else y[:, :128].double() + y[:, 128:].double()
    assert relative(ln, ln64) < 3e-5, relative(ln, ln64)
    assert all(torch.equal(a, dx) and torch.equal(b, y) for a, b in outs[1:])


# ------------------------------------------------------------------------------------------------ stacks of blocks
def stacks(nb, n_head, d_head, seed=3):
    """(the PyTorch fp32 blocks, the engine's blocks), same weights."""
    torch.manual_seed(seed)
    ref = [randomize(BiasOnlyDiTBlock(n_head=n_head, d_head=d_head, implementation=PT)).cuda() for _ in range(nb)]
    fast = [BiasOnlyDiTBlock(n_head=n_head, d_head=d_head, implementation=MW).cuda() for _ in range(nb)]
    for f, r in zip(fast, ref, strict=True):
        f.load_state_dict(r.state_dict())
    return ref, fast


def inputs(L, A, masked, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(A, 1, L, 768, device="cuda")
    c = torch.randn(A, 1, L, 384, device="cuda")
    p = torch.randn(1, L, L, 128, device="cuda")
    dy = torch.randn(A, 1, L, 768, device="cuda")
    mask = (torch.rand(1, L, device="cuda") > 0.2) if masked else None
    return x, c, p, dy, mask


def forward(blocks, s, c, p, mask, hoist, wrap=None):
    if hoist:
        biases = pair_bias_all(blocks, p)
        fns = [(lambda t, b=b, bb=bb: b(t, c, None, mask, bias=bb)) for b, bb in zip(blocks, biases, strict=True)]
    else:
        fns = [(lambda t, b=b: b(t, c, p, mask)) for b in blocks]
    for fn in fns:
        s = fn(s) if wrap is None else wrap(fn, s)
    return s


def stack_step(blocks, x, c, p, mask, dy, dtype, hoist, wrap=None):
    leaves = [t.detach().to(dtype).requires_grad_(True) for t in (x, c, p)]
    for b in blocks:
        b.zero_grad(set_to_none=True)
    y = forward(blocks, *leaves, mask, hoist, wrap)
    y.backward(dy.to(dtype))
    return [y.detach()] + [t.grad for t in leaves] + [q.grad for b in blocks for q in b.parameters()]


def names(blocks):
    return ["out", "d single", "d cond", "d pair"] + [f"{i}.{n}" for i, b in enumerate(blocks) for n, _ in b.named_parameters()]


STACK = [(3, 384, 4, False), (3, 384, 4, True), (3, 768, 2, True), (3, 768, 2, False), (24, 384, 2, True)]


@pytest.mark.parametrize("dtype", [BF, F32])
@pytest.mark.parametrize(("n_head", "d_head"), [(16, 48), (24, 32)])
@pytest.mark.parametrize(("nb", "L", "A", "masked"), STACK)
def test_hoisted_stack_every_gradient_against_fp64(nb, L, A, masked, n_head, d_head, dtype):
    """nb blocks on the hoisted biases (fused hoist + fused blocks; no PyTorch path runs) against the stack in fp64 (per-block bias).

    3 blocks: the output and every gradient within the training files' bounds against the PyTorch stack in the same dtype (bf16 1.1x
    + 1e-4, fp32 1.5x + 1e-3).

    24 blocks (bf16): every gradient within max(1.1x the PyTorch bf16 stack's error, 1.05x the fused per-block stack's) + 1e-4, AND the
    hoist's own arithmetic exact on what it receives: each block's ln_pair / to_bias gradient against the same gradient computed in
    fp64 from the very d bias the hoist's backward was handed (captured at the op) and the pair, within the rounding to the
    parameter's dtype + 1e-4. Why the second reference: at 24 blocks of bf16 the d bias that reaches a deep block carries the stack's
    accumulated bf16 error, and that error alone can put one block's d gamma past 1.1x the PyTorch stack's -- for the fused per-block
    path as well. Measured on B200 (hoist2, ``bench_scripts/bo_hoist_diag.py``; this case, 16 x 48, L384 A2 masked): block 22's
    ln_pair gradient, hoisted / fused per-block / PyTorch bf16 error 1.172e-2 / 1.154e-2 / 1.008e-2 (1.16x / 1.14x; with
    MINIWORLD_BIAS_ONLY_DIT_BWD_FUSED=0 1.152e-2 / 1.187e-2: 1.14x / 1.18x); the hoist's arithmetic on the received d bias 4.5e-6 (every
    ln_pair gradient 3.8-4.9e-6; to_bias 1.6e-3 = its bf16 output rounding); the error is the d bias's (upstream 1.172e-2); over the
    24 blocks the hoisted error is at or below the per-block path's at most blocks. 3 blocks hold the PyTorch bound."""
    if nb == 24 and (dtype is F32 or n_head == 24):
        pytest.skip("24 blocks: bf16 16 x 48 only (time)")
    ref, fast = stacks(nb, n_head, d_head)
    ref64 = [copy.deepcopy(r).double() for r in ref]
    x, c, p, dy, mask = inputs(L, A, masked)
    want = stack_step(ref64, x, c, p, mask, dy, F64, hoist=False)
    if dtype is BF:
        fast = [f.to(BF) for f in fast]
        base_m, k, floor = [copy.deepcopy(r).to(BF) for r in ref], 1.1, 1e-4
    else:
        base_m, k, floor = ref, 1.5, 1e-3
    xb, cb, pb = (t.to(dtype).requires_grad_(True) for t in (x, c, p))
    bias0 = pair_bias_all(fast, pb)[0]
    assert HO.serves(MW, [b.attention for b in fast], pb) and TR.serves(fast[0], xb, cb, None, mask, bias=bias0)
    del bias0
    cap, orig = {}, HO._hoist_bwd

    def spy(pair2d, wf, dball, eps, split):                    # what the hoist's backward is handed
        cap["dball"], cap["pair"] = dball.clone(), pair2d.clone()
        return orig(pair2d, wf, dball, eps, split)
    HO._hoist_bwd = spy
    try:
        with fused_only():
            got = stack_step(fast, x, c, p, mask, dy, dtype, hoist=True)
    finally:
        HO._hoist_bwd = orig
    with fused_only():
        per_block = stack_step(fast, x, c, p, mask, dy, dtype, hoist=False)
    with tf32(dtype is F32):
        base = stack_step(base_m, x, c, p, mask, dy, dtype, hoist=False)
    nm = names(fast)
    deep = nb == 24
    for n, g, b, w, f in zip(nm, got, base, want, per_block, strict=True):
        assert g.dtype is b.dtype, n
        bound = max(k * relative(b, w), 1.05 * relative(f, w)) if deep else k * relative(b, w)
        assert relative(g, w) <= bound + floor, (n, "hoisted", relative(g, w), "pytorch", relative(b, w), "fused per-block",
                                                 relative(f, w))
    if deep:                                                    # the hoist's own arithmetic, in fp64 on its captured inputs
        dwf = cap["dball"].double() @ Fn.layer_norm(cap["pair"].double(), (128,), eps=1e-5)        # [nb H, 128]
        for i, blk in enumerate(fast):
            gam, wb = blk.attention.ln_pair.weight, blk.attention.to_bias.weight
            d = dwf[i * n_head:(i + 1) * n_head]
            for prm, val in ((gam, (d * wb.double()).sum(0)), (wb, d * gam.double()[None, :])):
                n = f"{i}.attention." + ("ln_pair.weight" if prm is gam else "to_bias.weight")
                g = got[nm.index(n)]
                rounding = relative(val.to(prm.dtype), val)                                       # 0 for an fp32 parameter
                assert relative(g, val) <= rounding + 1e-4, (n, "hoist arithmetic", relative(g, val), "output rounding", rounding)


@pytest.mark.parametrize("dtype", [BF, F32])
@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize("masked", [False, True])
def test_inference_on_hoisted_biases_matches_the_per_pair_path(masked, n_head, d_head, dtype):
    """No autograd: blocks on hoisted biases (P = softmax(bias) per call) against the same blocks on the pair (P cached per pair):
    within the inference bounds of the per-pair path against fp64 (bf16 1.1x + 1e-4, fp32 1.5x + 1e-3)."""
    nb, L, S = 3, 384, 5
    ref, fast = stacks(nb, n_head, d_head, seed=21)
    ref64 = [copy.deepcopy(r).double() for r in ref]
    if dtype is BF:
        fast = [f.to(BF) for f in fast]
    x, c, p, _, mask = inputs(L, S, masked, seed=4)
    k, floor = (1.1, 1e-4) if dtype is BF else (1.5, 1e-3)
    with torch.no_grad():
        want = forward(ref64, x.double(), c.double(), p.double(), mask, hoist=False)
        xs, cs, ps = (t.to(dtype) for t in (x, c, p))
        with fused_only():
            plain = forward(fast, xs, cs, ps, mask, hoist=False)
            biases = pair_bias_all(fast, ps)
            assert INF.serves(fast[0], xs, cs, None, mask, bias=biases[0])
            hoisted = forward(fast, xs, cs, ps, mask, hoist=True)
    assert relative(hoisted, want) <= k * relative(plain, want) + floor, (relative(hoisted, want), relative(plain, want))
    assert relative(hoisted, plain) < (1e-2 if dtype is BF else 3e-3)


@pytest.mark.parametrize("dtype", [BF, F32])
def test_hoisted_stack_under_checkpointing_is_bit_identical(dtype):
    """The hoist outside non-reentrant checkpointed blocks (MiniWorld's place): the recompute runs the same kernels on the same
    inputs, so the output and every gradient equal the plain hoisted step bit for bit."""
    from torch.utils.checkpoint import checkpoint
    _, fast = stacks(3, 16, 48, seed=13)
    if dtype is BF:
        fast = [f.to(BF) for f in fast]
    x, c, p, dy, mask = inputs(384, 3, True, seed=6)
    plain = [t.clone() for t in stack_step(fast, x, c, p, mask, dy, dtype, hoist=True)]
    ck = stack_step(fast, x, c, p, mask, dy, dtype, hoist=True, wrap=lambda fn, s: checkpoint(fn, s, use_reentrant=False))
    for n, a, b in zip(names(fast), ck, plain, strict=True):
        assert torch.equal(a, b), n


@pytest.mark.parametrize("dtype", [BF, F32])
@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
def test_hoisted_steps_bit_identical_with_a_poisoned_allocator(n_head, d_head, dtype):
    """Twenty training steps of a hoisted 3-block stack (forward + backward, the shared bias-gradient buffer included), the
    allocator's free memory filled with NaN before each: every gradient finite and bit-identical across the steps."""
    _, fast = stacks(3, n_head, d_head, seed=11)
    if dtype is BF:
        fast = [f.to(BF) for f in fast]
    x, c, p, dy, mask = inputs(384, 5, True, seed=2)
    runs = []
    for _ in range(20):
        _poison()
        runs.append([t.clone() for t in stack_step(fast, x, c, p, mask, dy, dtype, hoist=True)])
        torch.cuda.synchronize()
    for i, g in enumerate(runs[0]):
        assert torch.isfinite(g.float()).all(), i
        for r in runs[1:]:
            assert torch.equal(r[i], g), i


@pytest.mark.parametrize("dtype", [BF, F32])
@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
def test_hoisted_block_launches_no_pair_kernels(dtype, n_head, d_head):
    """The captured forward / backward of one block on a hoisted bias (under static weights: no repack) against the same block on the
    pair: the forward launches one kernel fewer (no pair_bias), the backward has no pair_bias_bwd (nor its pair_bias_bwd_fin merge)
    and exactly one finalize: as many launches as the pair's backward minus its pair kernels, plus finalize where the pair's was
    merged into pair_bias_bwd_fin."""
    from miniworld_engine.kernels import _capture
    from tests.cuda_graph_nodes import graph_kernels
    _, fast = stacks(1, n_head, d_head, seed=8)
    blk = fast[0].to(dtype)
    L, A, H = 384, 4, n_head
    x, c, p, dy, mask = (t if t is None else t.to(dtype) if t.is_floating_point() else t for t in inputs(L, A, True, seed=8))
    bias = torch.randn(1, H, L, L, device="cuda", dtype=dtype)
    params = [blk.get_parameter(n) for n in TR.NAMES]
    f32 = dtype is F32
    fwd, fwdh = (TR._fwd32, TR._fwd32h) if f32 else (TR._fwd, TR._fwdh)
    bwd, bwdh = (TR._bwd32, TR._bwd32h) if f32 else (TR._bwd, TR._bwdh)
    db = torch.empty(H, L * L, device="cuda", dtype=dtype)

    def captured(fn):
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            fn()
        names_ = graph_kernels(graph)
        graph.replay()
        torch.cuda.synchronize()
        return names_

    with _capture.static_weights(), torch.no_grad():
        _, *saved = fwd(x, c, p, mask, params)
        _, *saved_h = fwdh(x, c, bias, mask, params)
        nf = captured(lambda: fwd(x, c, p, mask, params))
        nfh = captured(lambda: fwdh(x, c, bias, mask, params))
        nb_ = captured(lambda: bwd(x, c, p, mask, params, saved, dy))
        nbh = captured(lambda: bwdh(x, c, mask, params, saved_h, dy, db))
    assert not any("pair_bias" in n for n in nfh), nfh
    assert sum("softmax_t" in n for n in nfh) == 1 and len(nfh) == len(nf) - 1, (nf, nfh)
    assert not any("pair_bias" in n for n in nbh), nbh
    assert sum("finalize" in n for n in nbh) == 1, nbh
    merged = any("pair_bias_bwd_fin" in n for n in nb_)
    assert len(nbh) == len(nb_) - sum("pair_bias_bwd" in n or "finalize" in n for n in nb_) + 1, (nb_, nbh, merged)


@pytest.mark.parametrize("dtype", [BF, F32])
@pytest.mark.parametrize(("n_head", "d_head"), [(16, 48), (24, 32)])
def test_hoisted_stack_graph_capture_and_steady_memory(dtype, n_head, d_head):
    """A hoisted 3-block training step: after two warm-up steps every further step leaves memory_reserved unchanged (the shared
    bias-gradient buffer is made in the first block backward and released by the hoist's); a captured step replays to the eager
    gradients (bf16 2e-3, fp32 1e-4: cuBLAS may pick other algorithms under capture)."""
    _, fast = stacks(3, n_head, d_head, seed=7)
    if dtype is BF:
        fast = [f.to(BF) for f in fast]
    L, A = 384, 8
    x, c, p, dy, _ = inputs(L, A, False, seed=1)
    x, c, p, dy = (t.to(dtype) for t in (x, c, p, dy))
    xs, cs, ps = (t.clone().requires_grad_(True) for t in (x, c, p))
    params = [q for b in fast for q in b.parameters()]

    def zero():
        for t in (xs, cs, ps):
            t.grad = None
        for q in params:
            if q.grad is not None:
                q.grad.zero_()

    def run():
        zero()
        forward(fast, xs, cs, ps, None, hoist=True).backward(dy)
        return [t.grad.clone() for t in (xs, cs, ps)] + [q.grad.clone() for q in params]

    torch.cuda.synchronize()
    eager = run()
    run()
    torch.cuda.synchronize()
    r2 = torch.cuda.memory_reserved()
    steps = []
    for _ in range(4):
        run()
        torch.cuda.synchronize()
        steps.append(torch.cuda.memory_reserved() - r2)
    assert steps == [0, 0, 0, 0], steps
    for q in params:
        q.grad = torch.zeros_like(q)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            run()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        zero()
        forward(fast, xs, cs, ps, None, hoist=True).backward(dy)
    graph.replay()
    torch.cuda.synchronize()
    replayed = [t.grad for t in (xs, cs, ps)] + [q.grad for q in params]
    tol = 2e-3 if dtype is BF else 1e-4
    for n, g, e in zip(names(fast)[1:], replayed, eager, strict=True):
        assert relative(g, e) < tol, (n, relative(g, e))
