"""Triton kernels of the fused token DiT inference path (AF3 Alg. 23 block: d 768, cond 384, pair 128, 16 x 48, n = 2).

Per block and sampling step the whole block is five launches: two GEMMs that apply AdaLN to their A operand
on load, the attention core, and two GEMMs that fold the output gate and the residual update into their
epilogue. The residual stream stays fp32 and is updated in place. LayerNorm statistics travel between them
as per-128-column (mean, M2) partials that the next AdaLN prologue merges (Chan et al.), so no kernel ever
re-reads the residual just to normalise it.

  row_stats         x -> (mean, M2) partials; only for the stack's input, every later block gets them from K3 / K5
  adaln_qkvg   K1   q|k|v|g = AdaLN(x) @ [Wq;Wk;Wv;Wg]^T + [bq;0;0;0]    -> four contiguous [M, 768] planes
  gate_resgate K3   x += sigmoid(gl) * ((o * sigmoid(g)) @ Wo^T)          + partials
  adaln_swiglu K4   h = silu(AdaLN(x) @ Wa^T) * (AdaLN(x) @ Wb^T)
  gate_resgate K5   x += sigmoid(gl) * (h @ Ws^T)                          + partials

Conditioning rows are indexed ``row % L``: at a sampling step every sample shares the conditioning, so the
modulation tensors are [L, *] and are read by all S samples.

  pair_bias_all     one pass over the pair: z -> LayerNorm -> [L*L, 128] @ [128, 24 * 16] -> head-major bias of
                    every block, [24 * 16, L, L]. Each block's LayerNorm weight is folded into its projection.
"""
import torch
import triton
import triton.language as tl

STAT_W = 128  # column width of one statistics partial; every producer and consumer agrees on it


def _cfgs(bms=(64, 128), bns=(64, 128), bks=(32, 64), warps=(4, 8), stages=(3, 4)):
    return [triton.Config({"BM": m, "BN": n, "BK": k}, num_warps=w, num_stages=s)
            for m in bms for n in bns for k in bks for w in warps for s in stages]


# ----------------------------------------------------------------------------------------------- statistics
@triton.jit
def _row_stats_kernel(X, MEAN, M2, M, sx, K: tl.constexpr, T: tl.constexpr, BM: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    rm = rows < M
    W: tl.constexpr = K // T
    for t in tl.static_range(T):
        cols = t * W + tl.arange(0, W)
        x = tl.load(X + rows[:, None].to(tl.int64) * sx + cols[None, :], mask=rm[:, None], other=0.0)
        mean = tl.sum(x, 1) / W
        d = x - mean[:, None]
        tl.store(MEAN + rows * T + t, mean, mask=rm)
        tl.store(M2 + rows * T + t, tl.sum(d * d, 1), mask=rm)


def row_stats(x, mean, m2):
    M, K = x.shape
    T = K // STAT_W
    _row_stats_kernel[(triton.cdiv(M, 32),)](x, mean, m2, M, x.stride(0), K=K, T=T, BM=32, num_warps=4)


@triton.jit
def _merged_stats(MEAN, M2, rows, rm, K: tl.constexpr, T: tl.constexpr, TP: tl.constexpr, eps):
    """Chan's parallel merge of T equal-count (mean, M2) partials -> (mean, rstd) per row."""
    ts = tl.arange(0, TP)
    tm = ts < T
    mk = rm[:, None] & tm[None, :]
    mp = tl.load(MEAN + rows[:, None] * T + ts[None, :], mask=mk, other=0.0)
    m2p = tl.load(M2 + rows[:, None] * T + ts[None, :], mask=mk, other=0.0)
    mean = tl.sum(mp, 1) / T
    d = tl.where(tm[None, :], mp - mean[:, None], 0.0)
    m2 = tl.sum(m2p, 1) + (K // T) * tl.sum(d * d, 1)
    return mean, 1.0 / tl.sqrt(m2 / K + eps)


# ----------------------------------------------------------------------------------------------- K1
@triton.autotune(configs=_cfgs(), key=["M", "L"])
@triton.jit
def _adaln_qkvg_kernel(X, MEAN, M2, MS, MB, WT, BIAS, OUT, M, L, sx, sms, smb, eps,
                       K: tl.constexpr, N: tl.constexpr, T: tl.constexpr, TP: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    rm = rows < M
    tok = rows % L
    mean, rstd = _merged_stats(MEAN, M2, rows, rm, K, T, TP, eps)
    cols = pid_n * BN + tl.arange(0, BN)
    r64 = rows[:, None].to(tl.int64)
    acc = tl.zeros([BM, BN], tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        x = tl.load(X + r64 * sx + ks[None, :], mask=rm[:, None], other=0.0)
        ms = tl.load(MS + tok[:, None] * sms + ks[None, :]).to(tl.float32)
        mb = tl.load(MB + tok[:, None] * smb + ks[None, :]).to(tl.float32)
        a = ((x - mean[:, None]) * rstd[:, None] * ms + mb).to(tl.bfloat16)
        w = tl.load(WT + ks[:, None] * N + cols[None, :])
        acc = tl.dot(a, w, acc)
    acc += tl.load(BIAS + cols).to(tl.float32)[None, :]
    D: tl.constexpr = N // 4
    seg = (pid_n * BN) // D                                   # BN divides D: a tile never straddles q|k|v|g
    out = OUT + seg * M * D + r64 * D + (cols - seg * D)[None, :]
    tl.store(out, acc.to(tl.bfloat16), mask=rm[:, None])


def adaln_qkvg(x, mean, m2, ms, mb, wt, bias, out, L, eps=1e-5):
    M, K = x.shape
    N = wt.shape[1]
    T = K // STAT_W
    grid = lambda c: (triton.cdiv(M, c["BM"]), N // c["BN"])
    _adaln_qkvg_kernel[grid](x, mean, m2, ms, mb, wt, bias, out, M, L, x.stride(0), ms.stride(0), mb.stride(0), eps,
                             K=K, N=N, T=T, TP=triton.next_power_of_2(T))


# ----------------------------------------------------------------------------------------------- K4
@triton.autotune(configs=_cfgs(), key=["M", "L"])
@triton.jit
def _adaln_swiglu_kernel(X, MEAN, M2, MS, MB, WAT, WBT, H, M, L, sx, sms, smb, eps,
                         K: tl.constexpr, N: tl.constexpr, T: tl.constexpr, TP: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    rm = rows < M
    tok = rows % L
    mean, rstd = _merged_stats(MEAN, M2, rows, rm, K, T, TP, eps)
    cols = pid_n * BN + tl.arange(0, BN)
    r64 = rows[:, None].to(tl.int64)
    acc_a = tl.zeros([BM, BN], tl.float32)
    acc_b = tl.zeros([BM, BN], tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        x = tl.load(X + r64 * sx + ks[None, :], mask=rm[:, None], other=0.0)
        ms = tl.load(MS + tok[:, None] * sms + ks[None, :]).to(tl.float32)
        mb = tl.load(MB + tok[:, None] * smb + ks[None, :]).to(tl.float32)
        a = ((x - mean[:, None]) * rstd[:, None] * ms + mb).to(tl.bfloat16)
        acc_a = tl.dot(a, tl.load(WAT + ks[:, None] * N + cols[None, :]), acc_a)
        acc_b = tl.dot(a, tl.load(WBT + ks[:, None] * N + cols[None, :]), acc_b)
    h = acc_a * tl.sigmoid(acc_a) * acc_b
    tl.store(H + r64 * N + cols[None, :], h.to(tl.bfloat16), mask=rm[:, None])


def adaln_swiglu(x, mean, m2, ms, mb, wat, wbt, out, L, eps=1e-5):
    M, K = x.shape
    N = wat.shape[1]
    T = K // STAT_W
    grid = lambda c: (triton.cdiv(M, c["BM"]), N // c["BN"])
    _adaln_swiglu_kernel[grid](x, mean, m2, ms, mb, wat, wbt, out, M, L, x.stride(0), ms.stride(0), mb.stride(0), eps,
                               K=K, N=N, T=T, TP=triton.next_power_of_2(T))


# ----------------------------------------------------------------------------------------------- K3 / K5
# restore_value: the residual is updated in place, and autotuning runs the kernel once per config -- without it
# every benchmarked config would add its update to x again.
@triton.autotune(configs=_cfgs(bns=(STAT_W,)), key=["M", "L", "K"], restore_value=["X"])
@triton.jit
def _gate_resgate_kernel(A, G, WT, GL, X, MEAN, M2, M, L, K, sa, sg, sgl, sx,
                         N: tl.constexpr, HAS_GATE: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    rm = rows < M
    tok = rows % L
    cols = pid_n * BN + tl.arange(0, BN)
    r64 = rows[:, None].to(tl.int64)
    acc = tl.zeros([BM, BN], tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        a = tl.load(A + r64 * sa + ks[None, :], mask=rm[:, None], other=0.0)
        if HAS_GATE:
            g = tl.load(G + r64 * sg + ks[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
            a = (a.to(tl.float32) * tl.sigmoid(g)).to(tl.bfloat16)
        acc = tl.dot(a, tl.load(WT + ks[:, None] * N + cols[None, :]), acc)
    gl = tl.load(GL + tok[:, None] * sgl + cols[None, :]).to(tl.float32)        # sigmoid already applied
    xp = X + r64 * sx + cols[None, :]
    x = tl.load(xp, mask=rm[:, None], other=0.0) + gl * acc
    tl.store(xp, x, mask=rm[:, None])
    mean = tl.sum(x, 1) / BN
    d = x - mean[:, None]
    T: tl.constexpr = N // BN
    tl.store(MEAN + rows * T + pid_n, mean, mask=rm)
    tl.store(M2 + rows * T + pid_n, tl.sum(d * d, 1), mask=rm)


def gate_resgate(a, g, wt, gl, x, mean, m2, L):
    M, K = a.shape
    N = wt.shape[1]
    assert N % STAT_W == 0
    grid = lambda c: (triton.cdiv(M, c["BM"]), N // c["BN"])
    _gate_resgate_kernel[grid](a, g if g is not None else a, wt, gl, x, mean, m2, M, L, K,
                               a.stride(0), (g if g is not None else a).stride(0), gl.stride(0), x.stride(0),
                               N=N, HAS_GATE=g is not None)


# ----------------------------------------------------------------------------------------------- pair bias
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
    zh = (d * (1.0 / tl.sqrt(tl.sum(d * d, 1) / C + eps))[:, None]).to(tl.bfloat16)
    LL = L.to(tl.int64) * L
    for n0 in tl.static_range(0, NB, BN):
        ns = n0 + tl.arange(0, BN)
        nm = ns < NB                                   # a short stack (NB < BN) or NB not a multiple of BN
        acc = tl.dot(zh, tl.load(WT + cs[:, None] * NB + ns[None, :], mask=nm[None, :], other=0.0))
        tl.store(OUT + ns[None, :].to(tl.int64) * LL + i * L + js[:, None], acc.to(tl.bfloat16),
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
    tl.store(OUT + r64 * D + cs[None, :], (d * rstd[:, None] * ms + mb).to(tl.bfloat16), mask=mk)


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
        tl.store(OUT + r64 * D + cs[None, :], (d * rstd[:, None] * ms + mb).to(tl.bfloat16), mask=mk)


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
    tl.store(OUT + r64 * D + cs[None, :], (o * tl.sigmoid(g)).to(tl.bfloat16), mask=mk)


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
    tl.store(OUT + r64 * N + cs[None, :], (a * tl.sigmoid(a) * b).to(tl.bfloat16), mask=mk)


def swiglu_rows(ab, out):
    M, N2 = ab.shape
    N = N2 // 2
    _swiglu_rows_kernel[lambda c: (triton.cdiv(M, c["BR"]),)](ab, out, M, ab.stride(0), N=N, NP=triton.next_power_of_2(N))
