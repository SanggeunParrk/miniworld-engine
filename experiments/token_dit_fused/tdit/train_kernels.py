"""Row kernels of the fused token DiT training block (tdit/train_b200.py). One program per ROWS rows; every kernel reads and
writes each operand once. fp32 math, bf16 GEMM operands, fp32 residual stream. Widths: D = 768, DC = 384 (padded to 1024 / 512
lanes), heads 16 x 48 (padded to 64)."""
import torch
import triton
import triton.language as tl

D, DC, H, DH = 768, 384, 16, 48
EPS = 1e-5


@triton.jit
def _sig(x):
    return 1.0 / (1.0 + tl.exp2(-1.4426950408889634 * x))


# ------------------------------------------------------------------------------------------------------------ forward
@triton.jit
def _cond_prep(C, CHAT, CBF, CSTAT, M, N: tl.constexpr, BN: tl.constexpr, ROWS: tl.constexpr, eps):
    """c [M, N] fp32 -> c_hat = LN(c) bf16, c bf16, (mean, rstd)."""
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    n = tl.arange(0, BN)
    msk = (r[:, None] < M) & (n[None, :] < N)
    c = tl.load(C + r[:, None] * N + n[None, :], mask=msk, other=0.0)
    mean = tl.sum(c, 1) / N
    xc = tl.where(msk, c - mean[:, None], 0.0)
    rstd = tl.rsqrt(tl.sum(xc * xc, 1) / N + eps)
    tl.store(CHAT + r[:, None] * N + n[None, :], (xc * rstd[:, None]).to(tl.bfloat16), mask=msk)
    tl.store(CBF + r[:, None] * N + n[None, :], c.to(tl.bfloat16), mask=msk)
    tl.store(CSTAT + r * 2, mean, mask=r < M)
    tl.store(CSTAT + r * 2 + 1, rstd, mask=r < M)


@triton.jit
def _adaln_a(X, G, sg, BS, XA, XSTAT, M, N: tl.constexpr, BN: tl.constexpr, ROWS: tl.constexpr, eps):
    """xa = sigmoid(G[:, :N] + bs) LN(x) + G[:, N:2N]  (bf16); saves x's (mean, rstd)."""
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    n = tl.arange(0, BN)
    rm, nm = r < M, n < N
    msk = rm[:, None] & nm[None, :]
    x = tl.load(X + r[:, None] * N + n[None, :], mask=msk, other=0.0)
    mean = tl.sum(x, 1) / N
    xc = tl.where(msk, x - mean[:, None], 0.0)
    rstd = tl.rsqrt(tl.sum(xc * xc, 1) / N + eps)
    s = tl.load(G + r[:, None] * sg + n[None, :], mask=msk, other=0.0).to(tl.float32) + tl.load(BS + n, mask=nm, other=0.0)[None, :]
    sh = tl.load(G + r[:, None] * sg + N + n[None, :], mask=msk, other=0.0).to(tl.float32)
    tl.store(XA + r[:, None] * N + n[None, :], (_sig(s) * xc * rstd[:, None] + sh).to(tl.bfloat16), mask=msk)
    tl.store(XSTAT + r * 2, mean, mask=rm)
    tl.store(XSTAT + r * 2 + 1, rstd, mask=rm)


@triton.jit
def _qknorm(QKVG, WQ, WK, QN, KN, VC, RQK, M, eq, ek, ROWS: tl.constexpr):
    """Per head RMSNorm of q and k (weights [48]); q, k, v out contiguous bf16 [M, 768]; saves rq, rk [M, 16] each."""
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    hh = tl.arange(0, 16)
    dd = tl.arange(0, 64)
    rm = r < M
    dm = dd < 48
    msk = rm[:, None, None] & dm[None, None, :]
    col = hh[None, :, None] * 48 + dd[None, None, :]
    q = tl.load(QKVG + r[:, None, None] * 3072 + col, mask=msk, other=0.0).to(tl.float32)
    k = tl.load(QKVG + r[:, None, None] * 3072 + 768 + col, mask=msk, other=0.0).to(tl.float32)
    v = tl.load(QKVG + r[:, None, None] * 3072 + 1536 + col, mask=msk, other=0.0)
    wq = tl.load(WQ + dd, mask=dm, other=0.0)[None, None, :]
    wk = tl.load(WK + dd, mask=dm, other=0.0)[None, None, :]
    rq = tl.rsqrt(tl.sum(q * q, 2) / 48 + eq)
    rk = tl.rsqrt(tl.sum(k * k, 2) / 48 + ek)
    tl.store(QN + r[:, None, None] * 768 + col, (q * rq[:, :, None] * wq).to(tl.bfloat16), mask=msk)
    tl.store(KN + r[:, None, None] * 768 + col, (k * rk[:, :, None] * wk).to(tl.bfloat16), mask=msk)
    tl.store(VC + r[:, None, None] * 768 + col, v, mask=msk)
    tl.store(RQK + r[:, None] * 32 + hh[None, :], rq, mask=rm[:, None])
    tl.store(RQK + r[:, None] * 32 + 16 + hh[None, :], rk, mask=rm[:, None])


@triton.jit
def _gate_o(O, QKVG, OG, M, ROWS: tl.constexpr):
    """og = sigmoid(g) o  (bf16), g = qkvg[:, 2304:3072]."""
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    n = tl.arange(0, 1024)
    msk = (r[:, None] < M) & (n[None, :] < 768)
    o = tl.load(O + r[:, None] * 768 + n[None, :], mask=msk, other=0.0)
    g = tl.load(QKVG + r[:, None] * 3072 + 2304 + n[None, :], mask=msk, other=0.0).to(tl.float32)
    tl.store(OG + r[:, None] * 768 + n[None, :], (_sig(g) * o).to(tl.bfloat16), mask=msk)


@triton.jit
def _res_adaln_b(X, Y, GG, sgg, BG, G, sg, BS, X1, XT, XSTAT, M, N: tl.constexpr, BN: tl.constexpr, ROWS: tl.constexpr, eps):
    """x1 = x + sigmoid(GG[:, :N] + bg) y (fp32 out), then xt = sigmoid(G[:, 2N:3N] + bs) LN(x1) + G[:, 3N:4N] (bf16)."""
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    n = tl.arange(0, BN)
    rm, nm = r < M, n < N
    msk = rm[:, None] & nm[None, :]
    x = tl.load(X + r[:, None] * N + n[None, :], mask=msk, other=0.0)
    y = tl.load(Y + r[:, None] * N + n[None, :], mask=msk, other=0.0).to(tl.float32)
    gg = tl.load(GG + r[:, None] * sgg + n[None, :], mask=msk, other=0.0).to(tl.float32) + tl.load(BG + n, mask=nm, other=0.0)[None, :]
    x1 = x + _sig(gg) * y
    tl.store(X1 + r[:, None] * N + n[None, :], x1, mask=msk)
    x1 = tl.where(msk, x1, 0.0)
    mean = tl.sum(x1, 1) / N
    xc = tl.where(msk, x1 - mean[:, None], 0.0)
    rstd = tl.rsqrt(tl.sum(xc * xc, 1) / N + eps)
    s = tl.load(G + r[:, None] * sg + 2 * N + n[None, :], mask=msk, other=0.0).to(tl.float32) + tl.load(BS + n, mask=nm, other=0.0)[None, :]
    sh = tl.load(G + r[:, None] * sg + 3 * N + n[None, :], mask=msk, other=0.0).to(tl.float32)
    tl.store(XT + r[:, None] * N + n[None, :], (_sig(s) * xc * rstd[:, None] + sh).to(tl.bfloat16), mask=msk)
    tl.store(XSTAT + r * 2, mean, mask=rm)
    tl.store(XSTAT + r * 2 + 1, rstd, mask=rm)


@triton.jit
def _swiglu(AB, Hh, M, ROWS: tl.constexpr):
    """h = silu(a) b (bf16), ab [M, 3072] = a | b."""
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    n = tl.arange(0, 2048)
    msk = (r[:, None] < M) & (n[None, :] < 1536)
    a = tl.load(AB + r[:, None] * 3072 + n[None, :], mask=msk, other=0.0).to(tl.float32)
    b = tl.load(AB + r[:, None] * 3072 + 1536 + n[None, :], mask=msk, other=0.0).to(tl.float32)
    tl.store(Hh + r[:, None] * 1536 + n[None, :], (a * _sig(a) * b).to(tl.bfloat16), mask=msk)


@triton.jit
def _res_c(X1, Z, GG, sgg, BG, OUT, M, N: tl.constexpr, BN: tl.constexpr, ROWS: tl.constexpr):
    """out = x1 + sigmoid(GG[:, N:2N] + bg) z  (fp32)."""
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    n = tl.arange(0, BN)
    nm = n < N
    msk = (r[:, None] < M) & nm[None, :]
    x1 = tl.load(X1 + r[:, None] * N + n[None, :], mask=msk, other=0.0)
    z = tl.load(Z + r[:, None] * N + n[None, :], mask=msk, other=0.0).to(tl.float32)
    gg = tl.load(GG + r[:, None] * sgg + N + n[None, :], mask=msk, other=0.0).to(tl.float32) + tl.load(BG + n, mask=nm, other=0.0)[None, :]
    tl.store(OUT + r[:, None] * N + n[None, :], x1 + _sig(gg) * z, mask=msk)


# ----------------------------------------------------------------------------------------------------------- backward
@triton.jit
def _res_c_bwd(DOUT, Z, GG, sgg, BG, DZ, DGG, sdg, PG2, M, N: tl.constexpr, BN: tl.constexpr, ROWS: tl.constexpr):
    """out = x1 + s z, s = sigmoid(gg2): dz = dout s (bf16), dgg2 = dout z s (1 - s) (bf16, DGG[:, N:2N])."""
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    n = tl.arange(0, BN)
    nm = n < N
    msk = (r[:, None] < M) & nm[None, :]
    do = tl.load(DOUT + r[:, None] * N + n[None, :], mask=msk, other=0.0)
    z = tl.load(Z + r[:, None] * N + n[None, :], mask=msk, other=0.0).to(tl.float32)
    s = _sig(tl.load(GG + r[:, None] * sgg + N + n[None, :], mask=msk, other=0.0).to(tl.float32) + tl.load(BG + n, mask=nm, other=0.0)[None, :])
    tl.store(DZ + r[:, None] * N + n[None, :], (do * s).to(tl.bfloat16), mask=msk)
    dg2 = (do * z * s * (1 - s)).to(tl.bfloat16)
    tl.store(DGG + r[:, None] * sdg + N + n[None, :], dg2, mask=msk)
    tl.atomic_add(PG2 + n, tl.sum(dg2.to(tl.float32), 0), mask=nm, sem="relaxed")


@triton.jit
def _swiglu_bwd(DH_, AB, DAB, M, ROWS: tl.constexpr):
    """h = silu(a) b: da = dh b silu'(a), db = dh silu(a) -> dab [M, 3072] bf16."""
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    n = tl.arange(0, 2048)
    msk = (r[:, None] < M) & (n[None, :] < 1536)
    dh = tl.load(DH_ + r[:, None] * 1536 + n[None, :], mask=msk, other=0.0).to(tl.float32)
    a = tl.load(AB + r[:, None] * 3072 + n[None, :], mask=msk, other=0.0).to(tl.float32)
    b = tl.load(AB + r[:, None] * 3072 + 1536 + n[None, :], mask=msk, other=0.0).to(tl.float32)
    sa = _sig(a)
    tl.store(DAB + r[:, None] * 3072 + n[None, :], (dh * b * sa * (1 + a * (1 - sa))).to(tl.bfloat16), mask=msk)
    tl.store(DAB + r[:, None] * 3072 + 1536 + n[None, :], (dh * a * sa).to(tl.bfloat16), mask=msk)


@triton.jit
def _res_adaln_b_bwd(DOUT, DXT, X1, XSTAT, G, sg, BS, GG, sgg, BG, Y, DX1, DY, DG, sdG, DGG, sdg, PS2, PG1, M,
                     N: tl.constexpr, BN: tl.constexpr, ROWS: tl.constexpr):
    """Backward of x1 = x + sigmoid(gg1) y ; xt = sigmoid(s2) LN(x1) + sh2 ; out = x1 + ... (dout carried by the residual).
    Out: dx1 = dout + LN'(dxt sigmoid(s2)) (fp32), dy = dx1 sigmoid(gg1) (bf16), dgg1 (DGG[:, :N]), ds2 | dsh2 (DG[:, 2N:4N])."""
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    n = tl.arange(0, BN)
    rm, nm = r < M, n < N
    msk = rm[:, None] & nm[None, :]
    dxt = tl.load(DXT + r[:, None] * N + n[None, :], mask=msk, other=0.0).to(tl.float32)
    x1 = tl.load(X1 + r[:, None] * N + n[None, :], mask=msk, other=0.0)
    mean = tl.load(XSTAT + r * 2, mask=rm, other=0.0)
    rstd = tl.load(XSTAT + r * 2 + 1, mask=rm, other=0.0)
    xh = tl.where(msk, (x1 - mean[:, None]) * rstd[:, None], 0.0)
    s2 = _sig(tl.load(G + r[:, None] * sg + 2 * N + n[None, :], mask=msk, other=0.0).to(tl.float32) + tl.load(BS + n, mask=nm, other=0.0)[None, :])
    tl.store(DG + r[:, None] * sdG + 3 * N + n[None, :], dxt.to(tl.bfloat16), mask=msk)
    ds2 = (dxt * xh * s2 * (1 - s2)).to(tl.bfloat16)
    tl.store(DG + r[:, None] * sdG + 2 * N + n[None, :], ds2, mask=msk)
    tl.atomic_add(PS2 + n, tl.sum(ds2.to(tl.float32), 0), mask=nm, sem="relaxed")
    dxh = tl.where(msk, dxt * s2, 0.0)
    m1 = tl.sum(dxh, 1) / N
    m2 = tl.sum(dxh * xh, 1) / N
    dx1 = tl.load(DOUT + r[:, None] * N + n[None, :], mask=msk, other=0.0) + rstd[:, None] * (dxh - m1[:, None] - xh * m2[:, None])
    tl.store(DX1 + r[:, None] * N + n[None, :], dx1, mask=msk)
    y = tl.load(Y + r[:, None] * N + n[None, :], mask=msk, other=0.0).to(tl.float32)
    g1 = _sig(tl.load(GG + r[:, None] * sgg + n[None, :], mask=msk, other=0.0).to(tl.float32) + tl.load(BG + n, mask=nm, other=0.0)[None, :])
    tl.store(DY + r[:, None] * N + n[None, :], (dx1 * g1).to(tl.bfloat16), mask=msk)
    dg1 = (dx1 * y * g1 * (1 - g1)).to(tl.bfloat16)
    tl.store(DGG + r[:, None] * sdg + n[None, :], dg1, mask=msk)
    tl.atomic_add(PG1 + n, tl.sum(dg1.to(tl.float32), 0), mask=nm, sem="relaxed")


@triton.jit
def _gate_o_bwd(DOG, O, QKVG, DOB, DD, DQKVG, L, M, ROWS: tl.constexpr):
    """og = sigmoid(g) o: dO = dog s -> bf16 dO and D = rowsum_head(dO o) [A, 16, L] (the core backward's prep), and
    dg = dog o s (1 - s) into dqkvg[:, 2304:3072] (bf16)."""
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    hh = tl.arange(0, 16)
    dd = tl.arange(0, 64)
    rm = r < M
    dm = dd < 48
    msk = rm[:, None, None] & dm[None, None, :]
    col = hh[None, :, None] * 48 + dd[None, None, :]
    dog = tl.load(DOG + r[:, None, None] * 768 + col, mask=msk, other=0.0).to(tl.float32)
    o = tl.load(O + r[:, None, None] * 768 + col, mask=msk, other=0.0)
    g = tl.load(QKVG + r[:, None, None] * 3072 + 2304 + col, mask=msk, other=0.0).to(tl.float32)
    s = _sig(g)
    do = dog * s
    dob = do.to(tl.bfloat16)
    tl.store(DOB + r[:, None, None] * 768 + col, dob, mask=msk)
    dsum = tl.sum(do * o, 2)
    a = r // L
    tl.store(DD + (a[:, None] * 16 + hh[None, :]) * L + (r % L)[:, None], dsum, mask=rm[:, None])
    tl.store(DQKVG + r[:, None, None] * 3072 + 2304 + col, (dog * o * s * (1 - s)).to(tl.bfloat16), mask=msk)


@triton.jit
def _qknorm_bwd(DQ, DK, DV, QKVG, RQK, WQ, WK, DQKVG, DWQK, PBQ, M, ROWS: tl.constexpr):
    """qn = q rq wq (rq = rms(q)^-1): dq = rq (dqn wq - q rq mean(dqn wq q rq)); same for k; dv copied. bf16 into
    dqkvg[:, :2304]. Per-program partial sums of dwq, dwk (dqn * q rq summed over rows and heads) -> DWQK [programs, 2, 64]."""
    pid = tl.program_id(0)
    r = pid * ROWS + tl.arange(0, ROWS)
    hh = tl.arange(0, 16)
    dd = tl.arange(0, 64)
    rm = r < M
    dm = dd < 48
    msk = rm[:, None, None] & dm[None, None, :]
    col = hh[None, :, None] * 48 + dd[None, None, :]
    for t in tl.static_range(2):
        dn = tl.load((DQ if t == 0 else DK) + r[:, None, None] * 768 + col, mask=msk, other=0.0)
        x = tl.load(QKVG + r[:, None, None] * 3072 + t * 768 + col, mask=msk, other=0.0).to(tl.float32)
        rr = tl.load(RQK + r[:, None] * 32 + t * 16 + hh[None, :], mask=rm[:, None], other=0.0)
        w = tl.load((WQ if t == 0 else WK) + dd, mask=dm, other=0.0)[None, None, :]
        xh = x * rr[:, :, None]
        dxh = dn * w
        dx = rr[:, :, None] * (dxh - xh * (tl.sum(dxh * xh, 2) / 48)[:, :, None])
        dxb = dx.to(tl.bfloat16)
        tl.store(DQKVG + r[:, None, None] * 3072 + t * 768 + col, dxb, mask=msk)
        if t == 0:
            tl.atomic_add(PBQ + hh[:, None] * 48 + dd[None, :], tl.sum(dxb.to(tl.float32), 0), mask=dm[None, :], sem="relaxed")
        tl.atomic_add(DWQK + t * 64 + dd, tl.sum(tl.sum(dn * xh, 0), 0), mask=dm, sem="relaxed")
    dv = tl.load(DV + r[:, None, None] * 768 + col, mask=msk, other=0.0)
    tl.store(DQKVG + r[:, None, None] * 3072 + 1536 + col, dv.to(tl.bfloat16), mask=msk)


@triton.jit
def _adaln_a_bwd(DXA, X, XSTAT, G, sg, BS, DX1, DX, DG, sdG, PS1, M, N: tl.constexpr, BN: tl.constexpr, ROWS: tl.constexpr):
    """xa = sigmoid(s1) LN(x) + sh1: dsh1 = dxa, ds1 = dxa xh s (1 - s) (DG[:, :2N], bf16), dx = dx1 + LN'(dxa s) (fp32)."""
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    n = tl.arange(0, BN)
    rm, nm = r < M, n < N
    msk = rm[:, None] & nm[None, :]
    dxa = tl.load(DXA + r[:, None] * N + n[None, :], mask=msk, other=0.0).to(tl.float32)
    x = tl.load(X + r[:, None] * N + n[None, :], mask=msk, other=0.0)
    mean = tl.load(XSTAT + r * 2, mask=rm, other=0.0)
    rstd = tl.load(XSTAT + r * 2 + 1, mask=rm, other=0.0)
    xh = tl.where(msk, (x - mean[:, None]) * rstd[:, None], 0.0)
    s = _sig(tl.load(G + r[:, None] * sg + n[None, :], mask=msk, other=0.0).to(tl.float32) + tl.load(BS + n, mask=nm, other=0.0)[None, :])
    tl.store(DG + r[:, None] * sdG + N + n[None, :], dxa.to(tl.bfloat16), mask=msk)
    ds1 = (dxa * xh * s * (1 - s)).to(tl.bfloat16)
    tl.store(DG + r[:, None] * sdG + n[None, :], ds1, mask=msk)
    tl.atomic_add(PS1 + n, tl.sum(ds1.to(tl.float32), 0), mask=nm, sem="relaxed")
    dxh = tl.where(msk, dxa * s, 0.0)
    m1 = tl.sum(dxh, 1) / N
    m2 = tl.sum(dxh * xh, 1) / N
    dx = tl.load(DX1 + r[:, None] * N + n[None, :], mask=msk, other=0.0) + rstd[:, None] * (dxh - m1[:, None] - xh * m2[:, None])
    tl.store(DX + r[:, None] * N + n[None, :], dx, mask=msk)


@triton.jit
def _cond_bwd(DCHAT, DCG, C, CSTAT, DC, M, N: tl.constexpr, BN: tl.constexpr, ROWS: tl.constexpr):
    """dc = LN'(dc_hat) + dc_gate (fp32)."""
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    n = tl.arange(0, BN)
    rm = r < M
    msk = rm[:, None] & (n[None, :] < N)
    dch = tl.load(DCHAT + r[:, None] * N + n[None, :], mask=msk, other=0.0).to(tl.float32)
    c = tl.load(C + r[:, None] * N + n[None, :], mask=msk, other=0.0)
    mean = tl.load(CSTAT + r * 2, mask=rm, other=0.0)
    rstd = tl.load(CSTAT + r * 2 + 1, mask=rm, other=0.0)
    ch = tl.where(msk, (c - mean[:, None]) * rstd[:, None], 0.0)
    m1 = tl.sum(dch, 1) / N
    m2 = tl.sum(dch * ch, 1) / N
    dc = rstd[:, None] * (dch - m1[:, None] - ch * m2[:, None]) + tl.load(DCG + r[:, None] * N + n[None, :], mask=msk, other=0.0).to(tl.float32)
    tl.store(DC + r[:, None] * N + n[None, :], dc, mask=msk)


def grid(M, rows):
    return (triton.cdiv(M, rows),)


# ------------------------------------------------------------------------------------------------------- pair bias
@triton.jit
def _pair_bias_fwd(PAIR, WP, WB, BIAS, PST, R2, eps, ROWS: tl.constexpr):
    """bias[h, r] = sum_c LN(pair[r])[c] wp[c] Wbias[h, c], written head-major bf16 [16, L L]; saves the row stats."""
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    c = tl.arange(0, 128)
    hh = tl.arange(0, 16)
    rm = r < R2
    p = tl.load(PAIR + r[:, None] * 128 + c[None, :], mask=rm[:, None], other=0.0)
    mean = tl.sum(p, 1) / 128
    xc = p - mean[:, None]
    rstd = tl.rsqrt(tl.sum(xc * xc, 1) / 128 + eps)
    pn = xc * rstd[:, None] * tl.load(WP + c)[None, :]
    wbt = tl.load(WB + hh[None, :] * 128 + c[:, None])                      # [128, 16] = Wbias^T
    b = tl.dot(pn, wbt, input_precision="tf32")                             # [ROWS, 16]
    tl.store(BIAS + hh[None, :] * R2 + r[:, None], b.to(tl.bfloat16), mask=rm[:, None])
    tl.store(PST + r * 2, mean, mask=rm)
    tl.store(PST + r * 2 + 1, rstd, mask=rm)


@triton.jit
def _pair_bias_bwd(DB, PAIR, PST, WP, WB, DPAIR, PART, R2, ROWS: tl.constexpr):
    """dbias [16, L L] fp32 -> dpair (fp32) and per-program partials of dWbias [16, 128] and dwp [128] -> PART[pid, 17, 128]."""
    pid = tl.program_id(0)
    r = pid * ROWS + tl.arange(0, ROWS)
    c = tl.arange(0, 128)
    hh = tl.arange(0, 16)
    rm = r < R2
    db = tl.load(DB + hh[None, :] * R2 + r[:, None], mask=rm[:, None], other=0.0)          # [ROWS, 16]
    wb = tl.load(WB + hh[:, None] * 128 + c[None, :])                                      # [16, 128]
    dpn = tl.dot(db, wb, input_precision="tf32")                                           # [ROWS, 128]
    p = tl.load(PAIR + r[:, None] * 128 + c[None, :], mask=rm[:, None], other=0.0)
    mean = tl.load(PST + r * 2, mask=rm, other=0.0)
    rstd = tl.load(PST + r * 2 + 1, mask=rm, other=0.0)
    ph = tl.where(rm[:, None], (p - mean[:, None]) * rstd[:, None], 0.0)
    wp = tl.load(WP + c)
    tl.atomic_add(PART + 16 * 128 + c, tl.sum(dpn * ph, 0), sem="relaxed")
    dwb = tl.dot(tl.trans(db), ph * wp[None, :], input_precision="tf32")                   # [16, 128]
    tl.atomic_add(PART + hh[:, None] * 128 + c[None, :], dwb, sem="relaxed")
    dxh = dpn * wp[None, :]
    m1 = tl.sum(dxh, 1) / 128
    m2 = tl.sum(dxh * ph, 1) / 128
    tl.store(DPAIR + r[:, None] * 128 + c[None, :], rstd[:, None] * (dxh - m1[:, None] - ph * m2[:, None]), mask=rm[:, None])


@triton.jit
def _colsum(X, sx, OUT, M, N: tl.constexpr, ROWS: tl.constexpr, BN: tl.constexpr):
    """Per-program column partial sums of a bf16 / fp32 [M, N] view (row stride sx) -> OUT[pid, N] fp32."""
    pid = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros([BN], tl.float32)
    for r0 in range(pid * ROWS, pid * ROWS + ROWS, 32):
        r = r0 + tl.arange(0, 32)
        acc += tl.sum(tl.load(X + r[:, None] * sx + n[None, :], mask=(r[:, None] < M) & (n[None, :] < N), other=0.0).to(tl.float32), 0)
    tl.store(OUT + pid * N + n, acc, mask=n < N)


def colsum(x, rows=64):
    """Column sums of a 2-D (possibly strided-row) view in fp32: one two-level pass instead of torch's strided reduce."""
    M, N = x.shape
    np_ = triton.cdiv(M, rows)
    part = torch.empty(np_, N, device=x.device, dtype=torch.float32)
    _colsum[(np_, triton.cdiv(N, 256))](x, x.stride(0), part, M, N=N, ROWS=rows, BN=256)
    return part.sum(0)


@triton.jit
def _unfold_lnw(DWN, WS1, WB1, WS2, WB2, W1, W2, DW, DW12, K: tl.constexpr, BK: tl.constexpr):
    """The cond-LN weights are folded into the conditioning GEMM (Wn = W diag(w)). Per output row i of Wn (4 x 768 rows):
    dW[i] = dWn[i] w, and dw += dWn[i] * W[i] (column sums over each group of 768 rows -> DW12 [2, K] via atomics)."""
    i = tl.program_id(0)                                         # 0 .. 3071
    k = tl.arange(0, BK)
    km = k < K
    grp = i // 768
    row = i % 768
    lw = tl.where(grp < 2, tl.load(W1 + k, mask=km, other=0.0), tl.load(W2 + k, mask=km, other=0.0))
    d = tl.load(DWN + i * K + k, mask=km, other=0.0)
    w = tl.where(grp == 0, tl.load(WS1 + row * K + k, mask=km, other=0.0),
                 tl.where(grp == 1, tl.load(WB1 + row * K + k, mask=km, other=0.0),
                          tl.where(grp == 2, tl.load(WS2 + row * K + k, mask=km, other=0.0), tl.load(WB2 + row * K + k, mask=km, other=0.0))))
    tl.store(DW + i * K + k, d * lw, mask=km)
    tl.atomic_add(DW12 + (grp // 2) * K + k, d * w, mask=km, sem="relaxed")
