"""The fp32 SWA atom DiT block's backward on B200 TF32 tensor cores (``kernels/swa_dit/cuda/sm100/tf32_bwd.py``): every gradient
(dq, dmod, dWqkv, dWg, dWo, dWu, dWd) against an fp64 reference block, and no worse than the Triton fp32 backward
(``dispatch._swa_dit_bwd_fp32_launch``) fed the same saved tensors; the modulation backward likewise; no Triton kernel on the way;
the deterministic outputs bitwise stable; every kernel within 128 registers and spill-free.

The saved tensors come from an fp64 recomputation of the forward (windowed attention, so A = 48 at S = 4096 fits), cast to fp32 --
the fp32 forward's contract -- so the test does not depend on the fp32 forward kernels. Each backward gets the attention saves its own
forward writes: for the TF32 kernels Q / K / V rounded to TF32 (cvt.rna; the recommended forward contract: the backward's MMAs read
them as stored, so its scores are the forward's) with the LSE of those rounded operands; for the Triton backward Q / K / V and O in
bf16 with the LSE of the bf16 operands. Everything else is the same fp32 tensors. Both backwards run their weight-gradient GEMMs on
cuBLAS TF32 (allow_tf32 on: MiniWorld trains fp32 under "medium"; tf32_bwd forces it anyway).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0), reason="B200 (sm_100)"),
]
DEV, F64 = "cuda", torch.float64
C, H, D, HIDDEN, HW = 128, 4, 32, 256, 64
EPS = float(torch.finfo(torch.float32).eps)
NAMES = ("dq", "dmod", "dWqkv", "dWg", "dWo", "dWu", "dWd")
#: relative Frobenius error against fp64. Activations / dmod: ours <= max(RATIO x Triton's, FLOOR_ACT). Weight gradients (cuBLAS
#: TF32 GEMMs: unit roundoff 2^-11 ~ 4.9e-4, where the Triton reference's dWu / dWd may be IEEE-fp32-accurate): ours <= W_ABS and
#: <= max(RATIO x Triton's, W_ABS) -- fp32 here means TF32. CEIL bounds every gradient of the unreferenced variant.
RATIO, FLOOR_ACT, W_ABS, CEIL = 1.5, 5e-4, 1e-3, 5e-3
WEIGHTS = {"dWqkv", "dWg", "dWo", "dWu", "dWd", "dWmod"}


def _band_ok(name, ex, et):
    if name in WEIGHTS:
        return ex <= W_ABS and ex <= max(RATIO * et, W_ABS)
    return ex <= max(RATIO * et, FLOOR_ACT)


def _check(names, ours, tri, ref, tag):
    """Every gradient's error (ours, Triton's) printed, then each against its band; the failure message carries them all."""
    rows = []
    for name, x, t, e in zip(names, ours, tri, ref, strict=True):
        assert x.dtype == torch.float32 and x.shape == e.shape, (name, x.dtype, tuple(x.shape))
        assert torch.isfinite(x).all(), name
        rows.append((name, _rel(x, e), _rel(t, e)))
    report = "; ".join(f"{n}: tf32 {ex:.2e} triton {et:.2e}" for n, ex, et in rows)
    print(f"[{tag}] {report}")
    bad = [n for n, ex, et in rows if not _band_ok(n, ex, et)]
    assert not bad, f"out of band: {bad} -- {report}"


def _cuda_kernel_names(fn, want=()):
    """Names of the CUDA kernels one call of ``fn`` launches, from a CUDA-graph capture of the call (``tests.cuda_graph_nodes``),
    not the profiler: in long pytest processes it drops the records of driver-API launches (these kernels) while keeping
    cuBLAS / ATen ones. ``want`` is kept for the callers; it is not used."""
    from tests.cuda_graph_nodes import launched_kernels

    return set(launched_kernels(fn))


def _rel(a, e):
    a, e = a.double(), e.double()
    return float((a - e).norm() / e.norm().clamp_min(1e-30))


def _rms(x, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _rope(x, cs, sn):
    """x [N, S, H, D]; cs / sn [N, S, D/2]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cs.unsqueeze(2), sn.unsqueeze(2)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def _window_attention(qh, kh, vh, seqused, tile=128):
    """Banded attention of reference.py (|i - j| <= HW, keys < seqused, padding queries -> 0) one 128-query tile at a time:
    (o [N, S, C], lse [N, H, S]) with lse = log sum_j exp(q k_j / sqrt D) over the allowed keys."""
    N, Hh, S, Dd = qh.shape
    pos = torch.arange(S, device=qh.device)
    valid = pos.view(1, S) < seqused.long().view(N, 1)
    outs, lses = [], []
    for t0 in range(0, S, tile):
        k0, k1 = max(t0 - HW, 0), min(t0 + tile + HW, S)
        i, j = pos[t0:t0 + tile], pos[k0:k1]
        allowed = ((i.view(-1, 1) - j.view(1, -1)).abs() <= HW).view(1, 1, len(i), len(j)) & valid[:, k0:k1].view(N, 1, 1, -1)
        allowed = allowed | (i.view(-1, 1) == j.view(1, -1)).view(1, 1, len(i), len(j))   # padding rows: their own key (output zeroed)
        sc = torch.einsum("nhid,nhjd->nhij", qh[:, :, t0:t0 + tile], kh[:, :, k0:k1]) * Dd ** -0.5
        sc = sc.masked_fill(~allowed, float("-inf"))
        lse = torch.logsumexp(sc, -1)
        o = torch.einsum("nhij,nhjd->nhid", torch.exp(sc - lse.unsqueeze(-1)), vh[:, :, k0:k1])
        outs.append(torch.where(valid[:, t0:t0 + tile].view(N, 1, -1, 1), o, torch.zeros_like(o)))
        lses.append(lse)
    return torch.cat(outs, 2).transpose(1, 2).reshape(N, S, Hh * Dd), torch.cat(lses, 2)


def _forward(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, b):
    """reference.swa_dit_block_reference with its intermediates: out and the fp32 forward's saves (in q's dtype)."""
    N, S, _ = q.shape
    A = N // b

    def per_row(t):
        return t.reshape(b, S, t.shape[-1]).repeat(A, 1, 1)

    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = per_row(mod).chunk(6, dim=-1)
    cs, sn = per_row(cos), per_row(sin)
    x = _rms(q, EPS) * (1 + scale_a) + shift_a
    pq, pk, pv = (x @ wqkv.t()).chunk(3, dim=-1)
    g = x @ wg.t()
    qh = _rope(_rms(pq.reshape(N, S, H, D), EPS), cs, sn).transpose(1, 2)
    kh = _rope(_rms(pk.reshape(N, S, H, D), EPS), cs, sn).transpose(1, 2)
    vh = pv.reshape(N, S, H, D).transpose(1, 2)
    o, lse = _window_attention(qh, kh, vh, seqused)
    att = (torch.sigmoid(g) * o) @ wo.t()
    q1 = q + gate_a * att
    y = _rms(q1, EPS) * (1 + scale_f) + shift_f
    ffn = (F.silu(y @ wu[:HIDDEN].t()) * (y @ wu[HIDDEN:].t())) @ wd.t()
    out = q1 + gate_f * ffn
    saves = {"qh": qh, "kh": kh, "vh": vh, "g": g, "o": o, "lse": lse, "q1": q1, "x": x, "pq": pq, "pk": pk, "att": att, "y": y,
             "ffn": ffn}
    return out, saves


def _tf32(t):
    """fp32 -> TF32 rounded to nearest, ties away (cvt.rna.tf32.f32), kept in fp32."""
    bits = t.float().contiguous().view(torch.int32)
    return ((bits + 0x1000) & -8192).view(torch.float32)


def _case(a, b, s, seed=0):
    """fp64 leaves on the GPU: q, mod, cos, sin, seqused, the five weights; ragged lengths (front-packed, one row short of the window)."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g, dtype=F64) * scale).to(DEV)

    n = a * b
    q, mod = r(n, s, C), r(b * s, 6 * C, scale=0.3)
    ws = [r(3 * C, C, scale=C ** -0.5), r(C, C, scale=C ** -0.5), r(C, C, scale=C ** -0.5), r(2 * HIDDEN, C, scale=C ** -0.5),
          r(C, HIDDEN, scale=HIDDEN ** -0.5)]
    angles = torch.randn(b * s, D // 2, generator=g, dtype=F64) * 3.0
    lengths = ([s, max(1, s - 17), 5, s - 1] * n)[:n]
    return q, mod, angles.cos().to(DEV), angles.sin().to(DEV), torch.tensor(lengths, dtype=torch.int32, device=DEV), ws


def _grads(a, b, s, seed=0, round_qkv=True):
    """(fp64 reference gradients, fp32 arguments of block_bwd_tf32, bf16 attention saves for the Triton backward). round_qkv: the
    TF32 path's Q / K / V saves rounded to TF32 and its LSE taken from them (else the plain fp32 casts and the fp64 LSE)."""
    q, mod, cos, sin, seqused, ws = _case(a, b, s, seed)
    leaves = [t.clone().requires_grad_() for t in (q, mod, *ws)]
    out, saves = _forward(leaves[0], leaves[1], cos, sin, seqused, *leaves[2:], b)
    dy = torch.randn(out.shape, generator=torch.Generator().manual_seed(seed + 1), dtype=F64).to(DEV)
    out.backward(dy)
    ref = [leaves[0].grad, leaves[1].grad, *(t.grad for t in leaves[2:])]
    f32 = lambda t: t.detach().float().contiguous()  # noqa: E731
    sv = {k: f32(v) for k, v in saves.items()}
    M = a * b * s
    for k in ("g", "o", "q1", "x", "pq", "pk", "att", "y", "ffn"):
        sv[k] = sv[k].reshape(M, C)
    args = dict(dy=f32(dy), q=f32(q), mod=f32(mod), cos=f32(cos), sin=f32(sin), seqused=seqused, wqkv=f32(ws[0]), wg=f32(ws[1]),
                wo=f32(ws[2]), wu=f32(ws[3]), wd=f32(ws[4]), **sv)
    def lse_of(qkv):                                       # the LSE a forward on these attention operands writes
        return _window_attention(*(t.double() for t in qkv), seqused)[1].float().contiguous()

    with torch.no_grad():
        if round_qkv:
            for k in ("qh", "kh", "vh"):
                args[k] = _tf32(args[k])
            args["lse"] = lse_of((args["qh"], args["kh"], args["vh"]))
        bf = {k: saves[k].detach().to(torch.bfloat16).contiguous() for k in ("qh", "kh", "vh")}
        bf["o"] = saves["o"].detach().reshape(M, C).to(torch.bfloat16).contiguous()
        bf["lse"] = lse_of((bf["qh"], bf["kh"], bf["vh"]))
    return ref, args, bf


def _ours(args, b):
    from miniworld_engine.kernels.swa_dit.cuda.sm100 import tf32_bwd

    a = args
    return tf32_bwd.block_bwd_tf32(a["dy"], a["q"], a["mod"], a["cos"], a["sin"], a["seqused"], a["wqkv"], a["wg"], a["wo"], a["wu"],
                                   a["wd"], a["qh"], a["kh"], a["vh"], a["g"], a["o"], a["lse"], a["q1"], a["x"], a["pq"], a["pk"],
                                   a["att"], a["y"], a["ffn"], b)


def _triton(args, bf, b):
    from miniworld_engine.kernels.swa_dit import dispatch

    a = args
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        return dispatch._swa_dit_bwd_fp32_launch(a["dy"], a["q"], a["mod"], a["cos"], a["sin"], a["seqused"], a["wqkv"], a["wg"],
                                                 a["wo"], a["wu"], a["wd"], bf["qh"], bf["kh"], bf["vh"], a["g"], bf["o"], bf["lse"],
                                                 a["q1"], a["x"], a["pq"], a["pk"], a["att"], a["y"], a["ffn"], b, HW, EPS)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def test_window_reference_matches_dense_reference():
    """The tiled fp64 forward used below is reference.swa_dit_block_reference (dense attention) at a size where both fit."""
    from miniworld_engine.kernels.swa_dit.reference import swa_dit_block_reference

    q, mod, cos, sin, seqused, ws = _case(3, 2, 384)
    out, _ = _forward(q, mod, cos, sin, seqused, *ws, 2)
    exp = swa_dit_block_reference(q, mod, cos, sin, seqused, *ws, 2, half_window=HW, eps=EPS)
    assert _rel(out, exp) < 1e-12


# (A, B, S): A = 1 (one augment, SP = 1 / AT = 128), A = 5 (SP = 5 / AT = 25: partial tiles), A = 48 (the training shape, AT = 8),
# one B = 2 case; S = 1024 and 4096
@pytest.mark.parametrize(("a", "b", "s"), [(1, 1, 1024), (5, 1, 1024), (48, 1, 1024), (3, 2, 1024),
                                           (1, 1, 4096), (5, 1, 4096), (48, 1, 4096)])
def test_backward_matches_fp64_and_is_no_worse_than_triton(a, b, s):
    torch.manual_seed(0)
    ref, args, bf = _grads(a, b, s)
    ours = _ours(args, b)
    tri = _triton(args, bf, b)
    torch.cuda.synchronize()
    _check(NAMES, ours, tri, ref, f"A{a} B{b} S{s}")


def test_backward_with_unrounded_qkv_saves():
    """A forward that stores Q / K / V as plain fp32 (its MMAs truncating them as ours do): still within the error ceiling."""
    ref, args, _ = _grads(48, 1, 1024, round_qkv=False)
    ours = _ours(args, 1)
    errs = {name: _rel(x, e) for name, x, e in zip(NAMES, ours, ref, strict=True)}
    print("[unrounded qkv] " + "; ".join(f"{n}: tf32 {v:.2e}" for n, v in errs.items()))
    assert all(v <= CEIL for v in errs.values()), errs


@pytest.mark.parametrize(("a", "b", "s"), [(48, 1, 1024), (5, 1, 1024)])
def test_backward_launches_no_triton_kernel(a, b, s):
    _, args, _ = _grads(a, b, s)
    _ours(args, b)                                         # build / load / weight forms outside the trace
    want = {"swa_ffn_bwd_tf32_sm100", "swa_oproj_bwd_tf32_sm100", "swa_attn_dkv_tf32_sm100", "swa_attn_dq_tf32_sm100",
            "swa_qkvg_bwd_tf32_sm100"}
    names = _cuda_kernel_names(lambda: _ours(args, b), want)
    assert want <= names, sorted(names)
    assert not [n for n in names if "triton" in n.lower() or n.startswith("_swa_")], sorted(names)


def test_backward_deterministic():
    """dq and the weight gradients are bitwise stable (no atomics on their way: the attention dQ / dK / dV sum in a fixed order);
    dmod gathers fp32 atomics across CTAs, so it is only close."""
    _, args, _ = _grads(48, 1, 1024)
    first = _ours(args, 1)
    second = _ours(args, 1)
    for name, x, y in zip(NAMES, first, second, strict=True):
        if name == "dmod":
            torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)
        else:
            assert torch.equal(x, y), name


@pytest.mark.parametrize(("b", "s"), [(1, 1024), (2, 4096)])
def test_modulation_backward(b, s):
    from miniworld_engine.kernels.swa_dit.cuda.sm100 import tf32_bwd

    gen = torch.Generator().manual_seed(3)
    c = torch.randn(b * s, C, generator=gen, dtype=F64).to(DEV)
    w = (torch.randn(6 * C, C, generator=gen, dtype=F64) * 0.05).to(DEV)
    g = torch.randn(b * s, 6 * C, generator=gen, dtype=F64).to(DEV)
    leaves = [c.clone().requires_grad_(), w.clone().requires_grad_()]
    (F.silu(leaves[0]) @ leaves[1].t()).backward(g)
    ours = tf32_bwd.mod_bwd_tf32(g.float(), c.float(), w.float())
    l32 = [c.float().requires_grad_(), w.float().requires_grad_()]
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        (F.silu(l32[0]) @ l32[1].t()).backward(g.float())   # the per-op fp32 path under TF32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    _check(("dc", "dWmod"), ours, (l32[0].grad, l32[1].grad), (leaves[0].grad, leaves[1].grad), f"mod B{b} S{s}")


def test_kernels_fit_128_registers_without_spills():
    from miniworld_engine.kernels.swa_dit.cuda.sm100 import tf32_bwd

    K = tf32_bwd.kernels(torch.cuda.current_device())
    for name in tf32_bwd.KERNELS:
        k = getattr(K, name)
        assert k.lmem == 0, (name, k.lmem)
        assert k.regs <= 128, (name, k.regs)


def test_supported():
    from miniworld_engine.kernels.swa_dit.cuda.sm100 import tf32_bwd

    q = torch.zeros(2, 256, C, device=DEV)
    assert tf32_bwd.supported_tf32(q, HIDDEN, HW, EPS)
    assert not tf32_bwd.supported_tf32(q.bfloat16(), HIDDEN, HW, EPS)
    assert not tf32_bwd.supported_tf32(torch.zeros(2, 200, C, device=DEV), HIDDEN, HW, EPS)
    assert not tf32_bwd.supported_tf32(q, HIDDEN, 32, EPS)
