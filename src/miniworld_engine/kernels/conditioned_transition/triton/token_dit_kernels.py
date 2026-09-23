"""Selected token DiT row-fusion kernels and fused pair-bias projection.

Older AdaLN-in-GEMM experiments remain in the research archive.
"""
import torch
import triton
import triton.language as tl

STAT_W = 128

@triton.autotune(configs=[triton.Config({"BJ": bj, "BN": bn}, num_warps=w, num_stages=s)
                          for bj in (64, 128) for bn in (64, 128) for w in (4, 8) for s in (2, 3)], key=["L", "NB"])
@triton.jit
def _pair_bias_all_kernel(Z, WT, OUT, L, sz, eps, C: tl.constexpr, NB: tl.constexpr,
                          BJ: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    nj = tl.cdiv(L, BJ)
    i = pid // nj
    js = (pid % nj) * BJ + tl.arange(0, BJ)
    jm = js < L
    cs = tl.arange(0, C)
    rowz = (i * L + js).to(tl.int64)
    z = tl.load(Z + rowz[:, None] * sz + cs[None, :], mask=jm[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(z, 1) / C
    d = z - mean[:, None]
    zh = (d * (1.0 / tl.sqrt(tl.sum(d * d, 1) / C + eps))[:, None]).to(WT.dtype.element_ty)
    LL = L.to(tl.int64) * L
    for n0 in tl.static_range(0, NB, BN):
        ns = n0 + tl.arange(0, BN)
        nm = ns < NB                                   # a short stack (NB < BN) or NB not a multiple of BN
        acc = tl.dot(zh, tl.load(WT + cs[:, None] * NB + ns[None, :], mask=nm[None, :], other=0.0))
        tl.store(OUT + ns[None, :].to(tl.int64) * LL + i * L + js[:, None], acc.to(OUT.dtype.element_ty),
                 mask=jm[:, None] & nm[None, :])


def pair_bias_all(z2d, wt, out, L, eps=1e-5):
    C = z2d.shape[1]
    NB = wt.shape[1]
    grid = lambda c: (L * triton.cdiv(L, c["BJ"]),)
    _pair_bias_all_kernel[grid](z2d, wt, out, L, z2d.stride(0), eps, C=C, NB=NB)


# ================================================================================================ v2: row kernels
# v1 above fuses AdaLN into the GEMM A-operand load. In Triton that costs more than it saves: the transformed A has
# to go registers -> shared memory inside the k-loop, which breaks the software pipeline, and every N tile repeats
# the transform (24 times for q|k|v|g). Measured at L=768, S=5: K1 139.5 us for 18.1 GFLOP (130 TFLOPS) against
# 39 us for cuBLAS + a separate AdaLN pass. v2 leaves the four GEMMs to cuBLAS and fuses everything between them
# into row kernels that cross the module boundaries: the residual update of one half-block and the AdaLN of the
# next are ONE pass, and the gate of the attention output is applied on its way into the output projection.
def _rows_cfgs():
    return [triton.Config({"BR": br}, num_warps=w) for br in (1, 2, 4, 8) for w in (4, 8)]


@triton.autotune(configs=_rows_cfgs(), key=["M", "L"])
@triton.jit
def _adaln_rows_kernel(X, MS, MB, OUT, M, L, sx, sms, smb, eps, D: tl.constexpr, DP: tl.constexpr, BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = rows < M
    tok = rows % L
    cs = tl.arange(0, DP)
    cm = cs < D
    mk = rm[:, None] & cm[None, :]
    r64 = rows[:, None].to(tl.int64)
    x = tl.load(X + r64 * sx + cs[None, :], mask=mk, other=0.0)
    mean = tl.sum(x, 1) / D
    d = tl.where(cm[None, :], x - mean[:, None], 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(d * d, 1) / D + eps)
    ms = tl.sigmoid(tl.load(MS + tok[:, None] * sms + cs[None, :], mask=cm[None, :], other=0.0).to(tl.float32))
    mb = tl.load(MB + tok[:, None] * smb + cs[None, :], mask=cm[None, :], other=0.0).to(tl.float32)
    tl.store(OUT + r64 * D + cs[None, :], (d * rstd[:, None] * ms + mb).to(OUT.dtype.element_ty), mask=mk)


def adaln_rows(x, ms, mb, out, L, eps=1e-5):
    M, D = x.shape
    _adaln_rows_kernel[lambda c: (triton.cdiv(M, c["BR"]),)](x, ms, mb, out, M, L, x.stride(0), ms.stride(0), mb.stride(0),
                                                             eps, D=D, DP=triton.next_power_of_2(D))


# restore_value: X is the residual, updated in place; without it every config the autotuner benches would add again.
@triton.autotune(configs=_rows_cfgs(), key=["M", "L", "HAS_ADALN"], restore_value=["X"])
@triton.jit
def _resgate_adaln_rows_kernel(X, Y, GL, MS, MB, OUT, M, L, sx, sy, sgl, sms, smb, eps,
                               D: tl.constexpr, DP: tl.constexpr, HAS_ADALN: tl.constexpr, BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = rows < M
    tok = rows % L
    cs = tl.arange(0, DP)
    cm = cs < D
    mk = rm[:, None] & cm[None, :]
    r64 = rows[:, None].to(tl.int64)
    gl = tl.sigmoid(tl.load(GL + tok[:, None] * sgl + cs[None, :], mask=cm[None, :], other=0.0).to(tl.float32))
    y = tl.load(Y + r64 * sy + cs[None, :], mask=mk, other=0.0).to(tl.float32)
    xp = X + r64 * sx + cs[None, :]
    x = tl.load(xp, mask=mk, other=0.0) + gl * y
    tl.store(xp, x, mask=mk)
    if HAS_ADALN:
        mean = tl.sum(x, 1) / D
        d = tl.where(cm[None, :], x - mean[:, None], 0.0)
        rstd = 1.0 / tl.sqrt(tl.sum(d * d, 1) / D + eps)
        ms = tl.sigmoid(tl.load(MS + tok[:, None] * sms + cs[None, :], mask=cm[None, :], other=0.0).to(tl.float32))
        mb = tl.load(MB + tok[:, None] * smb + cs[None, :], mask=cm[None, :], other=0.0).to(tl.float32)
        tl.store(OUT + r64 * D + cs[None, :], (d * rstd[:, None] * ms + mb).to(OUT.dtype.element_ty), mask=mk)


def resgate_adaln_rows(x, y, gl, ms, mb, out, L, eps=1e-5):
    """x += gl * y in place (fp32); then, when ms is given, out = AdaLN(x) for the next half-block."""
    M, D = x.shape
    has = ms is not None
    _resgate_adaln_rows_kernel[lambda c: (triton.cdiv(M, c["BR"]),)](
        x, y, gl, ms if has else gl, mb if has else gl, out if has else y, M, L, x.stride(0), y.stride(0), gl.stride(0),
        (ms if has else gl).stride(0), (mb if has else gl).stride(0), eps, D=D, DP=triton.next_power_of_2(D), HAS_ADALN=has)


@triton.autotune(configs=_rows_cfgs(), key=["M"])
@triton.jit
def _gate_rows_kernel(O, G, OUT, M, so, sg, D: tl.constexpr, DP: tl.constexpr, BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cs = tl.arange(0, DP)
    mk = (rows < M)[:, None] & (cs < D)[None, :]
    r64 = rows[:, None].to(tl.int64)
    o = tl.load(O + r64 * so + cs[None, :], mask=mk, other=0.0).to(tl.float32)
    g = tl.load(G + r64 * sg + cs[None, :], mask=mk, other=0.0).to(tl.float32)
    tl.store(OUT + r64 * D + cs[None, :], (o * tl.sigmoid(g)).to(OUT.dtype.element_ty), mask=mk)


def gate_rows(o, g, out):
    M, D = o.shape
    _gate_rows_kernel[lambda c: (triton.cdiv(M, c["BR"]),)](o, g, out, M, o.stride(0), g.stride(0),
                                                           D=D, DP=triton.next_power_of_2(D))


@triton.autotune(configs=_rows_cfgs(), key=["M"])
@triton.jit
def _swiglu_rows_kernel(AB, OUT, M, sab, N: tl.constexpr, NP: tl.constexpr, BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cs = tl.arange(0, NP)
    mk = (rows < M)[:, None] & (cs < N)[None, :]
    r64 = rows[:, None].to(tl.int64)
    a = tl.load(AB + r64 * sab + cs[None, :], mask=mk, other=0.0).to(tl.float32)
    b = tl.load(AB + r64 * sab + N + cs[None, :], mask=mk, other=0.0).to(tl.float32)
    tl.store(OUT + r64 * N + cs[None, :], (a * tl.sigmoid(a) * b).to(OUT.dtype.element_ty), mask=mk)


def swiglu_rows(ab, out):
    M, N2 = ab.shape
    N = N2 // 2
    _swiglu_rows_kernel[lambda c: (triton.cdiv(M, c["BR"]),)](ab, out, M, ab.stride(0), N=N, NP=triton.next_power_of_2(N))
