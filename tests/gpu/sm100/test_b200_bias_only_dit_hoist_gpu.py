"""The hoisted pair bias of the bias-only token DiT on B200: ``pair_bias_all`` (modules/bias_only_dit/hoist.py,
integrations/bias_only_dit_hoist.py) and the blocks' ``bias=`` paths (integrations/bias_only_dit_train.py, bias_only_dit.py).

  * ``pair_bias_all`` forward and backward (d pair, every dW_b, every d gamma_b) against fp64 for 1 / 3 / 24 blocks, every head
    layout, bf16 and fp32 (TF32); the PyTorch fold the same; unused biases; the LN0 backward rows against fp64, no spills;
  * the sm_100a one-pass hoist (``bo_hoist_sm100.cu``, bf16 and fp32 / TF32, any number of weight rows: chunks and row ranges per
    launch) against fp64 alone (rows far from zero mean, a constant row, padded weight rows, bitwise reruns, no local memory) and
    under ``pair_bias_all`` on a pair per sample;
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
from portable_helpers import poison_allocator as _poison
from portable_helpers import randomize
from portable_helpers import relative_fp64 as relative
from portable_helpers import tf32_matmul_flag as tf32

from miniworld_engine import settings
from miniworld_engine.integrations import bias_only_dit as INF
from miniworld_engine.integrations import bias_only_dit_hoist as HO
from miniworld_engine.integrations import bias_only_dit_train as TR
from miniworld_engine.modules.bias_only_dit import (
    BiasOnlyAttention,
    BiasOnlyDiTBlock,
    pair_bias_all,
)
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


def hopper_serves_layout(n_head, d_head, dtype=None):
    """Hopper's fp32 (TF32 wgmma) block kernels serve 16 x 48 only; the other fp32 layouts take the module path (the hoist itself
    serves every layout, and bf16 serves them all)."""
    if torch.cuda.get_device_capability() == (9, 0) and dtype is F32 and (n_head, d_head) != (16, 48):
        pytest.skip("Hopper fp32 block kernels: 16 x 48 only")


@pytest.fixture(autouse=True)
def policy():
    if torch.cuda.get_device_capability() not in ((10, 0), (9, 0)):
        pytest.skip("Blackwell (sm_100) or Hopper (sm_90) required")
    old = settings.configure(force_portable_kernels=False)
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
    hoist_mod._reference = boom  # ty: ignore[invalid-assignment] -- deliberate stub
    try:
        yield
    finally:
        bo_module.BiasOnlyAttention.forward = old_fwd
        hoist_mod._reference = old_ref


# ------------------------------------------------------------------------------------------------ pair_bias_all alone
def attentions(nb, n_head, d_head, dtype, seed=5):
    torch.manual_seed(seed)
    mods = [
        randomize(BiasOnlyAttention(768, 384, 128, n_head, d_head)).cuda()
        for _ in range(nb)
    ]
    return [
        m.to(dtype) for m in mods
    ]  # bf16: to_bias bf16, ln_pair stays fp32 (the engine pins norm weights)


def hoist_ref64(attns, pair, dbs):
    """Each block's to_bias(ln_pair(pair)) [1, H, L, L] in fp64, and the gradients of sum_b <bias_b, db_b>."""
    p = pair.detach().double().requires_grad_(True)
    gs = [a.ln_pair.weight.detach().double().requires_grad_(True) for a in attns]
    ws = [a.to_bias.weight.detach().double().requires_grad_(True) for a in attns]
    ln = Fn.layer_norm(p, (128,), eps=1e-5)
    outs = [((ln * g) @ w.t()).permute(0, 3, 1, 2) for g, w in zip(gs, ws, strict=True)]
    torch.autograd.backward(outs, [d.double() for d in dbs])
    return (
        [o.detach() for o in outs],
        p.grad,
        [g.grad for g in gs],
        [w.grad for w in ws],
    )


def hoist_run(attns, pair, dbs, implementation=MW, use=None):
    """pair_bias_all and the gradients of sum_b <bias_b, db_b> (only the blocks in ``use``)."""
    p = pair.detach().clone().requires_grad_(True)
    for a in attns:
        a.zero_grad(set_to_none=True)
    outs = pair_bias_all(attns, p, implementation=implementation)
    idx = range(len(attns)) if use is None else use
    torch.autograd.backward([outs[i] for i in idx], [dbs[i] for i in idx])
    return (
        [o.detach() for o in outs],
        p.grad,
        [a.ln_pair.weight.grad for a in attns],
        [a.to_bias.weight.grad for a in attns],
    )


TOL = {
    BF: {"bias": 6e-3, "dpair": 8e-3, "dg": 1e-4, "dW": 6e-3},
    F32: {"bias": 3e-3, "dpair": 3e-3, "dg": 3e-3, "dW": 3e-3},
}


@pytest.mark.parametrize("dtype", [BF, F32])
@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize("nb", [1, 3, 24])
def test_pair_bias_all_against_fp64(nb, n_head, d_head, dtype):
    """The fused hoist: every block's bias (a contiguous head-major slice of one buffer), d pair, d gamma_b and dW_b against fp64."""
    L = 384
    attns = attentions(nb, n_head, d_head, dtype)
    g = torch.Generator(device="cuda").manual_seed(nb + n_head)
    pair = torch.randn(1, L, L, 128, device="cuda", generator=g).to(dtype)
    dbs = [
        torch.randn(1, n_head, L, L, device="cuda", generator=g).to(dtype)
        for _ in range(nb)
    ]
    assert HO.serves(MW, attns, pair)
    with fused_only():
        outs, dp, dgs, dws = hoist_run(attns, pair, dbs)
    w_outs, w_dp, w_dgs, w_dws = hoist_ref64(attns, pair, dbs)
    tol = TOL[dtype]
    base = outs[0].untyped_storage().data_ptr()
    for i, (o, w) in enumerate(zip(outs, w_outs, strict=True)):
        assert o.dtype is dtype, i
        assert o.shape == (1, n_head, L, L), i
        assert o.is_contiguous(), i
        assert o.data_ptr() == base + i * n_head * L * L * o.element_size(), (
            i
        )  # slice i of the one [nb H, L L] buffer
        assert relative(o, w) < tol["bias"], (i, relative(o, w))
    assert dp.dtype is dtype, relative(dp, w_dp)
    assert relative(dp, w_dp) < tol["dpair"], relative(dp, w_dp)
    for i in range(nb):
        assert dgs[i].dtype is attns[i].ln_pair.weight.dtype, i
        assert dws[i].dtype is attns[i].to_bias.weight.dtype, i
        assert relative(dgs[i], w_dgs[i]) < tol["dg"], (i, relative(dgs[i], w_dgs[i]))
        assert relative(dws[i], w_dws[i]) < tol["dW"], (i, relative(dws[i], w_dws[i]))


@pytest.mark.parametrize("dtype", [BF, F32])
def test_pair_bias_all_pytorch_fold_and_unused_biases(dtype):
    """The PyTorch fold (implementation PYTORCH) against fp64, the same
    API; a backward through two of three biases: the third block's gradients are zero, the rest as fp64 with a zero d bias."""
    L, nb, H, DH = 256, 3, 16, 48
    attns = attentions(nb, H, DH, dtype, seed=9)
    g = torch.Generator(device="cuda").manual_seed(17)
    pair = torch.randn(1, L, L, 128, device="cuda", generator=g).to(dtype)
    dbs = [
        torch.randn(1, H, L, L, device="cuda", generator=g).to(dtype) for _ in range(nb)
    ]
    w_outs, w_dp, w_dgs, w_dws = hoist_ref64(attns, pair, dbs)
    outs, dp, dgs, dws = hoist_run(attns, pair, dbs, implementation=PT)
    bound = 2e-2 if dtype is BF else 3e-3
    for o, w in zip(outs, w_outs, strict=True):
        assert o.shape == (1, H, L, L)
        assert relative(o, w) < bound
    assert relative(dp, w_dp) < bound
    for a, b, c, d in zip(dgs, w_dgs, dws, w_dws, strict=True):
        assert relative(a, b) < bound
        assert relative(c, d) < bound
    # unused bias 1, fused
    zero = [dbs[0], torch.zeros_like(dbs[1]), dbs[2]]
    w_outs, w_dp, w_dgs, w_dws = hoist_ref64(attns, pair, zero)
    with fused_only():
        outs, dp, dgs, dws = hoist_run(attns, pair, dbs, use=[0, 2])
    tol = TOL[dtype]
    assert relative(dp, w_dp) < tol["dpair"]
    assert torch.count_nonzero(dgs[1]) == 0
    assert torch.count_nonzero(dws[1]) == 0
    for i in (0, 2):
        assert relative(dgs[i], w_dgs[i]) < tol["dg"], i
        assert relative(dws[i], w_dws[i]) < tol["dW"], i


def test_serves_and_declines():
    attns = attentions(2, 16, 48, BF)
    p = torch.randn(1, 256, 256, 128, device="cuda", dtype=BF)
    assert HO.serves(MW, attns, p)
    assert HO.serves(ImplementationType.MINIWORLD, attns, p)
    assert not HO.serves(PT, attns, p)  # the PyTorch implementation: the fold
    assert not HO.serves(MW, attns, p.expand(2, -1, -1, -1))  # B == 1 only
    assert not HO.serves(MW, attns, p.half())  # bf16 / fp32 only
    a32 = attentions(2, 16, 48, F32)
    assert HO.serves(MW, a32, p.float())
    assert not HO.serves(MW, attns, p.float())  # fp32 needs fp32 weights
    with torch.autocast("cuda", dtype=BF):
        assert not HO.serves(MW, a32, p.float())  # fp32 under autocast: the fold
    with pytest.raises(ValueError, match="must share eps and head count"):
        pair_bias_all(
            [*attns, *attentions(1, 24, 32, BF)], p, implementation=MW
        )  # head counts differ


@pytest.mark.parametrize("R", [16, 1000, 147456])
@pytest.mark.parametrize(("xdt", "odt"), [(BF, BF), (F32, F32), (BF, F32), (F32, BF)])
def test_ln0_backward_rows_against_fp64(R, xdt, odt):
    """ln0_bwd_rows: d x and the dW' operand y (bf16: [hi | lo], hi + lo = LN0 to ~2^-17; fp32: LN0) -- every row written
    (NaN-filled outputs; R off the 16-row block grain), d x within its output rounding of fp64, hi + lo within 3e-5, bit-identical on
    reruns; no local memory. y takes the path's dtype (that of x)."""
    from miniworld_engine.kernels.bias_only_dit.cuda import hoist as HK

    for name, regs, lmem in HK.kernel_attrs():
        assert lmem == 0, (name, regs, lmem)
        assert regs <= 128, (name, regs, lmem)
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
    assert torch.isfinite(dx.float()).all()
    assert torch.isfinite(y.float()).all()
    assert relative(dx, xd.grad) < (6e-3 if odt is BF else 1e-5), relative(dx, xd.grad)
    ln = y.double() if K == 128 else y[:, :128].double() + y[:, 128:].double()
    assert relative(ln, ln64) < 3e-5, relative(ln, ln64)
    assert all(torch.equal(a, dx) and torch.equal(b, y) for a, b in outs[1:])


# ------------------------------------------------------------------------------------------------ the sm_100a one-pass hoist
def _sm100_only():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("the sm_100a one-pass hoist (bo_hoist_sm100.cu) is B200-only")


def _rows_ref64(pair2d, wf, dball, eps=1e-5):
    """bias_all^T = W' LN0(x)^T, d x and dW' = dbias_all LN0 in fp64 from the same bf16 inputs."""
    x = pair2d.double().requires_grad_(True)
    ln = Fn.layer_norm(x, (128,), eps=eps)
    bias = wf.double() @ ln.t()
    torch.autograd.backward(bias, dball.double())
    return bias.detach(), x.grad, dball.double() @ ln.detach()


# (rows, weight rows, row mean): per-sample shapes (A L^2 rows), 12 / 24 / 48 weight rows (padded to 16 / 32 / 48), the backward's
# largest (64), a grid whose CTAs end a tile apart (149 tiles), and rows whose mean is 20 standard deviations from zero
SM100_ROWS = [
    (128, 16, 0.0),
    (3 * 256 * 256, 16, 0.0),
    (2 * 384 * 384, 48, 0.5),
    (4 * 128 * 128, 12, 0.0),
    (5 * 128 * 128, 24, -1.0),
    (3 * 128 * 128, 64, 0.0),
    (149 * 128, 16, 0.0),
    (2 * 256 * 256, 16, 20.0),
]


@pytest.mark.parametrize(("R", "N", "shift"), SM100_ROWS)
def test_sm100_one_pass_hoist_against_fp64(R, N, shift):
    """``bo_hoist_sm100.cu`` alone: bias_all^T, d pair and dW' against fp64 -- bias and d pair within their bf16 output rounding (the
    bounds of ``TOL``), dW' within 3e-5 (the scaled d bias as hi + lo against the exact pair, fp32 accumulation; measured 2.5-6e-6 at
    zero mean); rows with a mean 20 standard deviations off zero (the forward's projection on raw rows and the backward's S - mean_k S
    both cancel a mean term) and a constant row (variance 0) included; bit-identical reruns; no local memory; at most 64 weight rows
    in the one-chunk backward (more: the chunked backward, ``test_sm100_hoist_chunks_and_tf32``)."""
    _sm100_only()
    from miniworld_engine.kernels.bias_only_dit.cuda import hoist as HK

    g = torch.Generator(device="cuda").manual_seed(R + N)
    pair = (torch.randn(R, 128, device="cuda", generator=g) + shift).to(BF)
    pair[5] = 0.75  # a constant row: LN0 = 0, rstd = 1 / sqrt(eps)
    wf = (0.1 * torch.randn(N, 128, device="cuda", generator=g)).to(BF)
    db = torch.randn(N, R, device="cuda", generator=g).to(BF)
    assert HK.sm100_served(pair, wf) and HK.sm100_served(pair, wf, bwd=True)
    np_ = (N + 15) // 16 * 16
    h = HK._hoist100(torch.cuda.current_device())
    for bwd in (False, True):
        k = h._kernel(np_, bwd)
        assert k.lmem == 0, (np_, bwd, k.regs, k.lmem)
    runs = []
    for _ in range(3):
        runs.append((HK.sm100_fwd(pair, wf), *HK.sm100_bwd(pair, wf, db)))
    torch.cuda.synchronize()
    bias, dp, dw = runs[0]
    assert all(torch.equal(a, b) for r in runs[1:] for a, b in zip(r, runs[0], strict=True))
    w_bias, w_dp, w_dw = _rows_ref64(pair, wf, db)
    assert bias.shape == (N, R) and dp.shape == (R, 128) and dw.shape == (N, 128) and dw.dtype is F32
    assert relative(bias, w_bias) < TOL[BF]["bias"], relative(bias, w_bias)
    assert relative(dp, w_dp) < TOL[BF]["dpair"], relative(dp, w_dp)
    assert relative(dw, w_dw) < 3e-5, relative(dw, w_dw)
    assert float(bias[:, 5].float().abs().max()) < 1e-2  # the constant row's bias is ~0
    wide = (0.1 * torch.randn(80, 128, device="cuda", generator=g)).to(BF)
    assert HK.sm100_served(pair, wide) and HK.sm100_served(pair, wide, bwd=True)  # more rows: the chunked backward
    assert not HK.sm100_served(pair[:100], wf)  # rows: a multiple of 128
    assert HK.sm100_served(pair.float(), wf.float()) and HK.sm100_served(pair.float(), wf.float(), bwd=True)  # fp32: TF32
    assert not HK.sm100_served(pair.float(), wf)  # W' in the pair's dtype


# (rows, weight rows, row mean, dtype): bf16 past the one-chunk kernels (forward: weight rows on M, one launch up to 384 rows, 384 + 192
# at 576; backward: the normalised-tile kernel, dbias chunks of 128 -- two d LN0 accumulators up to 256 rows, one past that, an S-only
# launch past 384) and fp32 / TF32 (forward: one launch up to 64 rows, past that CTA pairs -- an odd tile count, whose last tile pairs
# with zero rows, a partial last chunk, 576 = 320 + 256 in two launches; backward chunks of 32, an S-only launch past 384, and up to 16
# rows chunks of 16 with W' resident and three tile slots, -DSMALLN: 3, 12, 16 rows, a tile count not a multiple of 3), a grid whose
# CTAs end a tile apart, rows 20 standard deviations off zero, five tiles with a partial last chunk
SM100_CHUNKS = [
    (5 * 128 * 128, 80, 0.5, BF),
    (2 * 128 * 128, 144, 0.0, BF),
    (2 * 128 * 128, 384, -1.0, BF),
    (149 * 128, 576, 0.0, BF),
    (2 * 256 * 256, 288, 20.0, BF),
    (5 * 128, 200, 0.0, BF),
    (3 * 128 * 128, 16, 0.0, F32),
    (149 * 128, 48, 0.5, F32),
    (3 * 128 * 128, 100, 0.0, F32),
    (2 * 128 * 128, 384, 20.0, F32),
    (2 * 128 * 128, 576, 0.0, F32),
    (149 * 128, 200, 0.0, F32),
    (5 * 128, 136, -1.0, F32),
    (3 * 128, 72, 0.0, F32),
    (149 * 128, 12, 0.5, F32),
    (5 * 128, 16, 20.0, F32),
]


@pytest.mark.parametrize(("R", "N", "shift", "dt"), SM100_CHUNKS)
def test_sm100_hoist_chunks_and_tf32(R, N, shift, dt):
    """``bo_hoist_sm100.cu`` over weight-row chunks (``fwdt_plan`` / ``fwd_plan`` / ``bwd3_plan``) and in fp32 (kind::tf32: the forward normalises the
    tile in shared memory first, the backward takes S = xhat^T dbias) alone: bias_all^T, d pair and dW' against fp64 (bf16: the bars
    of ``test_sm100_one_pass_hoist_against_fp64``; fp32: the TF32 bound of ``TOL``, LN0 and dbias rounded to tf32 once each), rows far
    from zero mean and a constant row, bit-identical reruns, no local memory in any cubin used; served forward and backward at every N."""
    _sm100_only()
    from miniworld_engine.kernels.bias_only_dit.cuda import hoist as HK

    g = torch.Generator(device="cuda").manual_seed(R + N + 1)
    pair = (torch.randn(R, 128, device="cuda", generator=g) + shift).to(dt)
    pair[5] = 0.75
    wf = (0.1 * torch.randn(N, 128, device="cuda", generator=g)).to(dt)
    db = torch.randn(N, R, device="cuda", generator=g).to(dt)
    # served at every N both ways (the fp32 forward past 128 rows was T7's until the CTA-pair kernel, 2026-10-30)
    assert HK.sm100_served(pair, wf) and HK.sm100_served(pair, wf, bwd=True)
    runs = []
    for _ in range(3):
        runs.append((HK.sm100_fwd(pair, wf), *HK.sm100_bwd(pair, wf, db)))
    torch.cuda.synchronize()
    bias, dp, dw = runs[0]
    assert all(torch.equal(a, b) for r in runs[1:] for a, b in zip(r, runs[0], strict=True))
    h = HK._hoist100(torch.cuda.current_device())
    for run in h.runs.values():
        assert run.k.lmem == 0, (run.k.regs, run.k.lmem)
    w_bias, w_dp, w_dw = _rows_ref64(pair, wf, db)
    assert bias.shape == (N, R) and bias.dtype is dt and dp.shape == (R, 128) and dp.dtype is dt and dw.shape == (N, 128)
    tol = TOL[dt]
    assert relative(bias, w_bias) < tol["bias"], relative(bias, w_bias)
    assert relative(dp, w_dp) < tol["dpair"], relative(dp, w_dp)
    assert relative(dw, w_dw) < (3e-5 if dt is BF else tol["dW"]), relative(dw, w_dw)
    for n0 in range(0, N, 64):  # every chunk / launch range on its own
        assert relative(bias[n0 : n0 + 64], w_bias[n0 : n0 + 64]) < tol["bias"], n0
        assert relative(dw[n0 : n0 + 64], w_dw[n0 : n0 + 64]) < (3e-5 if dt is BF else tol["dW"]), n0
    assert float(bias[:, 5].float().abs().max()) < 1e-2


@pytest.mark.parametrize(
    ("A", "L", "nb", "n_head", "d_head"),
    [(3, 256, 1, 16, 48), (2, 384, 3, 16, 48), (4, 128, 2, 24, 32), (2, 256, 1, 12, 64), (2, 128, 5, 16, 48)],
)
@pytest.mark.parametrize("dtype", [BF, F32])
def test_per_sample_hoist_runs_the_sm100_kernels(A, L, nb, n_head, d_head, dtype, monkeypatch):
    """``pair_bias_all`` on a pair per sample [A, L, L, 128] (bf16 and fp32) takes ``bo_hoist_sm100.cu`` forward and backward at every
    weight-row count (5 blocks of 16 heads: the chunked backward) and meets the hoist's fp64 bars (``TOL``) for every output and
    gradient; ``settings.bias_only_dit_hoist_sm100=False`` runs the previous path to the same bars."""
    _sm100_only()
    from miniworld_engine.kernels.bias_only_dit.cuda import hoist as HK

    calls = {"fwd": 0, "bwd": 0}
    f0, b0 = HK.sm100_fwd, HK.sm100_bwd

    def fwd(*a, **k):
        calls["fwd"] += 1
        return f0(*a, **k)

    def bwd(*a, **k):
        calls["bwd"] += 1
        return b0(*a, **k)

    monkeypatch.setattr(HK, "sm100_fwd", fwd)
    monkeypatch.setattr(HK, "sm100_bwd", bwd)
    attns = attentions(nb, n_head, d_head, dtype)
    g = torch.Generator(device="cuda").manual_seed(A * L + nb)
    pair = torch.randn(A, L, L, 128, device="cuda", generator=g).to(dtype)
    dbs = [torch.randn(A, n_head, L, L, device="cuda", generator=g).to(dtype) for _ in range(nb)]
    assert HO.serves(MW, attns, pair, per_sample=True)
    w_outs, w_dp, w_dgs, w_dws = hoist_ref64(attns, pair, dbs)
    for on in (True, False):
        old = settings.configure(bias_only_dit_hoist_sm100=on)
        try:
            before = dict(calls)
            with fused_only():
                outs, dp, dgs, dws = hoist_run(attns, pair, dbs)
            torch.cuda.synchronize()
        finally:
            settings.configure(**vars(old))
        ran = {k: calls[k] - before[k] for k in calls}
        assert ran == ({"fwd": 1, "bwd": 1} if on else {"fwd": 0, "bwd": 0}), (on, ran)
        tol = TOL[dtype]
        for i, (o, w) in enumerate(zip(outs, w_outs, strict=True)):
            assert o.shape == (A, n_head, L, L), i
            assert relative(o, w) < tol["bias"], (on, i, relative(o, w))
        assert relative(dp, w_dp) < tol["dpair"], (on, relative(dp, w_dp))
        for i in range(nb):
            assert relative(dgs[i], w_dgs[i]) < tol["dg"], (on, i, relative(dgs[i], w_dgs[i]))
            assert relative(dws[i], w_dws[i]) < tol["dW"], (on, i, relative(dws[i], w_dws[i]))


# ------------------------------------------------------------------------------------------------ stacks of blocks
def stacks(nb, n_head, d_head, seed=3):
    """(the PyTorch fp32 blocks, the engine's blocks), same weights."""
    torch.manual_seed(seed)
    ref = [
        randomize(
            BiasOnlyDiTBlock(n_head=n_head, d_head=d_head, implementation=PT)
        ).cuda()
        for _ in range(nb)
    ]
    fast = [
        BiasOnlyDiTBlock(n_head=n_head, d_head=d_head, implementation=MW).cuda()
        for _ in range(nb)
    ]
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
        fns = [
            (lambda t, b=b, bb=bb: b(t, c, None, mask, bias=bb))
            for b, bb in zip(blocks, biases, strict=True)
        ]
    else:
        fns = [(lambda t, b=b: b(t, c, p, mask)) for b in blocks]
    for fn in fns:
        s = fn(s) if wrap is None else wrap(fn, s)
    return s


def stack_step(blocks, x, c, p, mask, dy, dtype, hoist, wrap=None):
    leaves = [t.detach().to(dtype).requires_grad_(True) for t in (x, c, p)]
    for b in blocks:
        b.zero_grad(set_to_none=True)
    y = forward(blocks, leaves[0], leaves[1], leaves[2], mask, hoist, wrap)
    y.backward(dy.to(dtype))
    return (
        [y.detach()]
        + [t.grad for t in leaves]
        + [q.grad for b in blocks for q in b.parameters()]
    )


def names(blocks):
    return ["out", "d single", "d cond", "d pair"] + [
        f"{i}.{n}" for i, b in enumerate(blocks) for n, _ in b.named_parameters()
    ]


STACK = [
    (3, 384, 4, False),
    (3, 384, 4, True),
    (3, 768, 2, True),
    (3, 768, 2, False),
    (24, 384, 2, True),
]


@pytest.mark.parametrize("dtype", [BF, F32])
@pytest.mark.parametrize(("n_head", "d_head"), [(16, 48), (24, 32)])
@pytest.mark.parametrize(("nb", "L", "A", "masked"), STACK)
def test_hoisted_stack_every_gradient_against_fp64(
    nb, L, A, masked, n_head, d_head, dtype
):
    hopper_serves_layout(n_head, d_head, dtype)
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
    settings.bias_only_dit_bwd_fused=0 1.152e-2 / 1.187e-2: 1.14x / 1.18x); the hoist's arithmetic on the received d bias 4.5e-6 (every
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
    assert HO.serves(MW, [b.attention for b in fast], pb)
    assert TR.serves(fast[0], xb, cb, None, mask, bias=bias0)
    del bias0
    cap, orig = {}, HO._hoist_bwd

    def spy(pair2d, wf, dball, eps, split):  # what the hoist's backward is handed
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
        bound = (
            max(k * relative(b, w), 1.05 * relative(f, w))
            if deep
            else k * relative(b, w)
        )
        assert relative(g, w) <= bound + floor, (
            n,
            "hoisted",
            relative(g, w),
            "pytorch",
            relative(b, w),
            "fused per-block",
            relative(f, w),
        )
    if deep:  # the hoist's own arithmetic, in fp64 on its captured inputs
        dwf = cap["dball"].double() @ Fn.layer_norm(
            cap["pair"].double(), (128,), eps=1e-5
        )  # [nb H, 128]
        for i, blk in enumerate(fast):
            gam, wb = blk.attention.ln_pair.weight, blk.attention.to_bias.weight
            d = dwf[i * n_head : (i + 1) * n_head]
            for prm, val in (
                (gam, (d * wb.double()).sum(0)),
                (wb, d * gam.double()[None, :]),
            ):
                n = f"{i}.attention." + (
                    "ln_pair.weight" if prm is gam else "to_bias.weight"
                )
                g = got[nm.index(n)]
                rounding = relative(val.to(prm.dtype), val)  # 0 for an fp32 parameter
                assert relative(g, val) <= rounding + 1e-4, (
                    n,
                    "hoist arithmetic",
                    relative(g, val),
                    "output rounding",
                    rounding,
                )


@pytest.mark.parametrize("dtype", [BF, F32])
@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
@pytest.mark.parametrize("masked", [False, True])
def test_inference_on_hoisted_biases_matches_the_per_pair_path(
    masked, n_head, d_head, dtype
):
    hopper_serves_layout(n_head, d_head, dtype)
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
    assert relative(hoisted, want) <= k * relative(plain, want) + floor, (
        relative(hoisted, want),
        relative(plain, want),
    )
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
    ck = stack_step(
        fast,
        x,
        c,
        p,
        mask,
        dy,
        dtype,
        hoist=True,
        wrap=lambda fn, s: checkpoint(fn, s, use_reentrant=False),
    )
    for n, a, b in zip(names(fast), ck, plain, strict=True):
        assert torch.equal(a, b), n


@pytest.mark.parametrize("dtype", [BF, F32])
@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
def test_hoisted_steps_bit_identical_with_a_poisoned_allocator(n_head, d_head, dtype):
    hopper_serves_layout(n_head, d_head, dtype)
    """Twenty training steps of a hoisted 3-block stack (forward + backward, the shared bias-gradient buffer included), the
    allocator's free memory filled with NaN before each: every gradient finite and bit-identical across the steps."""
    _, fast = stacks(3, n_head, d_head, seed=11)
    if dtype is BF:
        fast = [f.to(BF) for f in fast]
    x, c, p, dy, mask = inputs(384, 5, True, seed=2)
    runs = []
    for _ in range(20):
        _poison()
        runs.append(
            [t.clone() for t in stack_step(fast, x, c, p, mask, dy, dtype, hoist=True)]
        )
        torch.cuda.synchronize()
    for i, g in enumerate(runs[0]):
        assert torch.isfinite(g.float()).all(), i
        for r in runs[1:]:
            assert torch.equal(r[i], g), i


@pytest.mark.parametrize("dtype", [BF, F32])
@pytest.mark.parametrize(("n_head", "d_head"), LAYOUTS)
def test_hoisted_block_launches_no_pair_kernels(dtype, n_head, d_head):
    hopper_serves_layout(n_head, d_head, dtype)
    """The captured forward / backward of one block on a hoisted bias (under static weights: no repack) against the same block on the
    pair: the forward launches one kernel fewer (no pair_bias), the backward has no pair_bias_bwd (nor its pair_bias_bwd_fin merge)
    and exactly one finalize: as many launches as the pair's backward minus its pair kernels, plus finalize where the pair's was
    merged into pair_bias_bwd_fin."""
    from tests.cuda_graph_nodes import graph_kernels

    from miniworld_engine.kernels import _capture

    _, fast = stacks(1, n_head, d_head, seed=8)
    blk = fast[0].to(dtype)
    L, A, H = 384, 4, n_head
    x, c, p, dy, mask = (
        t if t is None else t.to(dtype) if t.is_floating_point() else t
        for t in inputs(L, A, True, seed=8)
    )
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
    assert sum("softmax_t" in n for n in nfh) == 1, (nf, nfh)
    assert len(nfh) == len(nf) - 1, (nf, nfh)
    assert not any("pair_bias" in n for n in nbh), nbh
    assert sum("finalize" in n for n in nbh) == 1, nbh
    merged = any("pair_bias_bwd_fin" in n for n in nb_)
    assert (
        len(nbh)
        == len(nb_) - sum("pair_bias_bwd" in n or "finalize" in n for n in nb_) + 1
    ), (nb_, nbh, merged)


@pytest.mark.parametrize("dtype", [BF, F32])
@pytest.mark.parametrize(("n_head", "d_head"), [(16, 48), (24, 32)])
def test_hoisted_stack_graph_capture_and_steady_memory(dtype, n_head, d_head):
    hopper_serves_layout(n_head, d_head, dtype)
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
