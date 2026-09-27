"""OuterProductMean in Triton (portable: any GPU Triton supports), forward + backward.

The decomposition keeps the reference's statements and fuses only where the PyTorch module pays extra memory passes:

  forward   prologue  (Triton)  LN(msa) -> left / right projections -> mask            a, b [S, L, CH] bf16
            GEMM1     (cuBLAS)  O = a^T b in the grouped layout [(i, d), (j, e)]        (torch.mm, as the module's einsum)
            counts    (cuBLAS)  n = mask^T mask (fp32, clamp 1)
            epilogue  (Triton)  out[i, j] = W_out . vec(O_ij) / n_ij + bias (+ residual)   the [i,j,d,e] permute folded into the loads
  backward  dgrad     (Triton)  dzn = dz / n, dO = dzn . W_out in the grouped layout
            dA, dB    (cuBLAS)  dA = b . dO^T, dB = a . dO
            dW_out    (Triton)  sum_ij dzn_ij (x) O_ij, read from the saved grouped O
            prologue backward (Triton)  mask, dy = [dA | dB] W, dW_left / dW_right, LayerNorm backward -> dmsa, dgamma, dbeta

normalize_before_proj=True (AF3 order, the default) divides before the projection; False (ESMFold2) divides after it, bias included.
mask_interchain is not fused (the module falls back).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------------------------------- kernels
@triton.jit
def _prologue_kernel(M, MASK, LNW, LNB, WL, WR, A, B, T, eps,
                     CM: tl.constexpr, CH: tl.constexpr, BT: tl.constexpr, HAS_MASK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BT + tl.arange(0, BT)
    rm = rows < T
    cols = tl.arange(0, CM)
    x = tl.load(M + rows[:, None] * CM + cols[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    mu = tl.sum(x, axis=1) / CM
    xc = x - mu[:, None]
    rs = tl.rsqrt(tl.sum(xc * xc, axis=1) / CM + eps)
    y = (xc * rs[:, None] * tl.load(LNW + cols)[None, :] + tl.load(LNB + cols)[None, :]).to(tl.bfloat16)
    ch = tl.arange(0, CH)
    wl = tl.load(WL + ch[None, :] * CM + cols[:, None])            # [CM, CH] = W_left^T
    wr = tl.load(WR + ch[None, :] * CM + cols[:, None])
    a = tl.dot(y, wl).to(tl.bfloat16)
    b = tl.dot(y, wr).to(tl.bfloat16)
    if HAS_MASK:
        mk = tl.load(MASK + rows, mask=rm, other=0) != 0
        a = tl.where(mk[:, None], a, 0.0)
        b = tl.where(mk[:, None], b, 0.0)
    tl.store(A + rows[:, None] * CH + ch[None, :], a, mask=rm[:, None])
    tl.store(B + rows[:, None] * CH + ch[None, :], b, mask=rm[:, None])


@triton.jit
def _epilogue_kernel(O, WOT, BIAS, NORM, RES, OUT, L,
                     CH: tl.constexpr, CZ: tl.constexpr, BJ: tl.constexpr, HAS_RES: tl.constexpr, NORM_FIRST: tl.constexpr):
    # program = (pair row i, block of BJ j): out[i, j, :] = sum_(d, e) O[i CH + d, j CH + e] W_out[:, d CH + e], K = CH^2 in CH steps
    i = tl.program_id(0)
    jb = tl.program_id(1)
    j = jb * BJ + tl.arange(0, BJ)
    jm = j < L
    e = tl.arange(0, CH)
    c = tl.arange(0, CZ)
    ldo = L * CH
    acc = tl.zeros((BJ, CZ), dtype=tl.float32)
    for d in range(CH):
        a = tl.load(O + (i * CH + d) * ldo + j[:, None] * CH + e[None, :], mask=jm[:, None], other=0.0)
        w = tl.load(WOT + (d * CH + e)[:, None] * CZ + c[None, :])
        acc += tl.dot(a, w)
    n = tl.load(NORM + i * L + j, mask=jm, other=1.0)
    bias = tl.load(BIAS + c)
    if NORM_FIRST:
        z = acc / n[:, None] + bias[None, :]
    else:
        z = (acc + bias[None, :]) / n[:, None]
    ptr = (i * L + j)[:, None] * CZ + c[None, :]
    if HAS_RES:
        z += tl.load(RES + ptr, mask=jm[:, None], other=0.0).to(tl.float32)
    tl.store(OUT + ptr, z.to(tl.bfloat16), mask=jm[:, None])


@triton.jit
def _dgrad_kernel(DZ, NORM, WO, DZN, DO, L,
                  CH: tl.constexpr, CZ: tl.constexpr, BJ: tl.constexpr, NORM_FIRST: tl.constexpr):
    # dzn = dz / n (both orders: the projection is linear), dO[(i, d), (j, e)] = sum_c dzn[i, j, c] W_out[c, d CH + e]
    i = tl.program_id(0)
    jb = tl.program_id(1)
    j = jb * BJ + tl.arange(0, BJ)
    jm = j < L
    c = tl.arange(0, CZ)
    e = tl.arange(0, CH)
    ptr = (i * L + j)[:, None] * CZ + c[None, :]
    dz = tl.load(DZ + ptr, mask=jm[:, None], other=0.0).to(tl.float32)
    n = tl.load(NORM + i * L + j, mask=jm, other=1.0)
    dzn = (dz / n[:, None]).to(tl.bfloat16)
    tl.store(DZN + ptr, dzn, mask=jm[:, None])
    ldo = L * CH
    for d in range(CH):
        w = tl.load(WO + c[:, None] * (CH * CH) + d * CH + e[None, :])     # [CZ, CH]
        o = tl.dot(dzn, w)
        tl.store(DO + (i * CH + d) * ldo + j[:, None] * CH + e[None, :], o.to(tl.bfloat16), mask=jm[:, None])


@triton.jit
def _dwo_kernel(DZN, O, DWO, L, i_per,
                CH: tl.constexpr, CZ: tl.constexpr, BK: tl.constexpr):
    # program = (d, i-range): dWo[:, d CH + e] += sum_(i in range, j) dzn[i, j, :]^T O[i CH + d, j CH + e]; fp32 atomics once per program
    d = tl.program_id(0)
    sp = tl.program_id(1)
    c = tl.arange(0, CZ)
    e = tl.arange(0, CH)
    kk = tl.arange(0, BK)
    ldo = L * CH
    acc = tl.zeros((CZ, CH), dtype=tl.float32)
    i0 = sp * i_per
    for ii in range(i_per):
        i = i0 + ii
        if i < L:
            for j0 in range(0, L, BK):
                j = j0 + kk
                jm = j < L
                a = tl.load(DZN + (i * L + j)[:, None] * CZ + c[None, :], mask=jm[:, None], other=0.0)     # [BK, CZ]
                o = tl.load(O + (i * CH + d) * ldo + j[:, None] * CH + e[None, :], mask=jm[:, None], other=0.0)   # [BK, CH]
                acc += tl.dot(tl.trans(a), o)
    tl.atomic_add(DWO + c[:, None] * (CH * CH) + d * CH + e[None, :], acc)


@triton.jit
def _prologue_bwd_kernel(M, MASK, LNW, LNB, WL, WR, DA, DB, DM, DWL, DWR, DG, DBETA, T, eps,
                         CM: tl.constexpr, CH: tl.constexpr, BT: tl.constexpr, HAS_MASK: tl.constexpr):
    # persistent: the program walks token blocks pid, pid + nprog, ..., keeping dW_left / dW_right / dgamma / dbeta in registers
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, CM)
    ch = tl.arange(0, CH)
    g = tl.load(LNW + cols)
    bta = tl.load(LNB + cols)
    wl = tl.load(WL + ch[:, None] * CM + cols[None, :])            # [CH, CM]
    wr = tl.load(WR + ch[:, None] * CM + cols[None, :])
    dwl = tl.zeros((CH, CM), dtype=tl.float32)
    dwr = tl.zeros((CH, CM), dtype=tl.float32)
    dgs = tl.zeros((CM,), dtype=tl.float32)
    dbs = tl.zeros((CM,), dtype=tl.float32)
    nblk = tl.cdiv(T, BT)
    for blk in range(pid, nblk, nprog):
        rows = blk * BT + tl.arange(0, BT)
        rm = rows < T
        x = tl.load(M + rows[:, None] * CM + cols[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        mu = tl.sum(x, axis=1) / CM
        xc = x - mu[:, None]
        rs = tl.rsqrt(tl.sum(xc * xc, axis=1) / CM + eps)
        xh = xc * rs[:, None]
        y = (xh * g[None, :] + bta[None, :]).to(tl.bfloat16)
        da = tl.load(DA + rows[:, None] * CH + ch[None, :], mask=rm[:, None], other=0.0)
        db = tl.load(DB + rows[:, None] * CH + ch[None, :], mask=rm[:, None], other=0.0)
        if HAS_MASK:
            mk = tl.load(MASK + rows, mask=rm, other=0) != 0
            da = tl.where(mk[:, None], da, 0.0).to(tl.bfloat16)
            db = tl.where(mk[:, None], db, 0.0).to(tl.bfloat16)
        dwl += tl.dot(tl.trans(da), y)
        dwr += tl.dot(tl.trans(db), y)
        dy = tl.dot(da, wl) + tl.dot(db, wr)                           # [BT, CM] fp32
        dgs += tl.sum(dy * xh, axis=0)
        dbs += tl.sum(dy, axis=0)
        dxh = dy * g[None, :]
        m1 = tl.sum(dxh, axis=1) / CM
        m2 = tl.sum(dxh * xh, axis=1) / CM
        dx = rs[:, None] * (dxh - m1[:, None] - xh * m2[:, None])
        tl.store(DM + rows[:, None] * CM + cols[None, :], dx.to(tl.bfloat16), mask=rm[:, None])
    tl.atomic_add(DWL + ch[:, None] * CM + cols[None, :], dwl)
    tl.atomic_add(DWR + ch[:, None] * CM + cols[None, :], dwr)
    tl.atomic_add(DG + cols, dgs)
    tl.atomic_add(DBETA + cols, dbs)


# ---------------------------------------------------------------------------------------------------- autograd
def _nsm():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def _forward(msa, mask, lnw, lnb, wl, wr, wo, bo, eps, norm_first, residual):
    _, S, L, CM = msa.shape
    CH, CZ = wl.shape[0], wo.shape[0]
    T = S * L
    dev = msa.device
    bf = torch.bfloat16
    x = msa.reshape(T, CM).contiguous()
    m8 = mask.reshape(T).contiguous().view(torch.uint8) if mask is not None else x
    a = torch.empty(T, CH, device=dev, dtype=bf)
    b = torch.empty(T, CH, device=dev, dtype=bf)
    BT = 64
    _prologue_kernel[(triton.cdiv(T, BT),)](x, m8, lnw, lnb, wl, wr, a, b, T, eps, CM=CM, CH=CH, BT=BT,
                                           HAS_MASK=mask is not None, num_warps=4)
    o = torch.mm(a.view(S, L * CH).t(), b.view(S, L * CH))                     # [(i, d), (j, e)]
    if mask is not None:
        mf = mask.reshape(S, L).float()
        norm = (mf.t() @ mf).clamp_(min=1).contiguous()
    else:
        norm = torch.full((L, L), float(S), device=dev, dtype=torch.float32)
    out = torch.empty(L * L, CZ, device=dev, dtype=bf)
    woT = wo.t().contiguous()
    res = residual.reshape(L * L, CZ).contiguous() if residual is not None else out
    BJ = 64
    _epilogue_kernel[(L, triton.cdiv(L, BJ))](o, woT, bo, norm, res, out, L, CH=CH, CZ=CZ, BJ=BJ,
                                             HAS_RES=residual is not None, NORM_FIRST=norm_first, num_warps=4, num_stages=3)
    return out.view(1, L, L, CZ), (x, m8 if mask is not None else None, a, b, o, norm)


class OPMTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, msa, mask, lnw, lnb, wl, wr, wo, bo, residual, eps, norm_first):
        bf = torch.bfloat16
        prm = [t.detach().to(bf).contiguous() for t in (wl, wr, wo)]
        lnf = [t.detach().float().contiguous() for t in (lnw, lnb)]
        out, saved = _forward(msa, mask, *lnf, *prm, bo.detach().float().contiguous(), eps, norm_first, residual)
        x, m8, a, b, o, norm = saved
        ctx.save_for_backward(x, a, b, o, norm, *lnf, *prm, *( [m8] if m8 is not None else []))
        ctx.has_mask = m8 is not None
        ctx.shape = msa.shape
        ctx.eps, ctx.norm_first = eps, norm_first
        ctx.dtypes = (lnw.dtype, lnb.dtype, wl.dtype, wr.dtype, wo.dtype, bo.dtype)
        ctx.has_res = residual is not None
        return out

    @staticmethod
    def backward(ctx, dz):
        x, a, b, o, norm, lnw, lnb, wl, wr, wo = ctx.saved_tensors[:10]
        m8 = ctx.saved_tensors[10] if ctx.has_mask else None
        _, S, L, CM = ctx.shape
        CH, CZ = wl.shape[0], wo.shape[0]
        T = S * L
        dev = x.device
        bf = torch.bfloat16
        dzc = dz.reshape(L * L, CZ).contiguous()
        dzn = torch.empty(L * L, CZ, device=dev, dtype=bf)
        dO = torch.empty(L * CH, L * CH, device=dev, dtype=bf)
        BJ = 64
        _dgrad_kernel[(L, triton.cdiv(L, BJ))](dzc, norm, wo, dzn, dO, L, CH=CH, CZ=CZ, BJ=BJ, NORM_FIRST=ctx.norm_first,
                                               num_warps=4, num_stages=2)
        if ctx.norm_first:
            dbo = dzc.float().sum(0)
        else:                                                          # bias divided by n too
            dbo = (dzc.float() / norm.reshape(L * L, 1)).sum(0)
        dA = torch.mm(b.view(S, L * CH), dO.t())                       # [S, (i, d)]
        dB = torch.mm(a.view(S, L * CH), dO)                           # [S, (j, e)]
        dwo = torch.zeros(CZ, CH * CH, device=dev, dtype=torch.float32)
        nsplit = max(1, min(L, 4 * _nsm() // CH))
        i_per = triton.cdiv(L, nsplit)
        _dwo_kernel[(CH, triton.cdiv(L, i_per))](dzn, o, dwo, L, i_per, CH=CH, CZ=CZ, BK=64, num_warps=4, num_stages=3)
        dm = torch.empty(T, CM, device=dev, dtype=bf)
        red = torch.zeros(2 * CH * CM + 2 * CM, device=dev, dtype=torch.float32)
        dwl, dwr = red[:CH * CM].view(CH, CM), red[CH * CM:2 * CH * CM].view(CH, CM)
        dg, dbeta = red[2 * CH * CM:2 * CH * CM + CM], red[2 * CH * CM + CM:]
        BT = 64
        _prologue_bwd_kernel[(min(triton.cdiv(T, BT), 4 * _nsm()),)](x, m8 if m8 is not None else x, lnw, lnb, wl, wr, dA.view(T, CH),
                                                                    dB.view(T, CH), dm, dwl, dwr, dg, dbeta, T, ctx.eps, CM=CM, CH=CH,
                                                                    BT=BT, HAS_MASK=m8 is not None, num_warps=4)
        t = ctx.dtypes
        return (dm.view(ctx.shape), None, dg.to(t[0]), dbeta.to(t[1]), dwl.to(t[2]), dwr.to(t[3]), dwo.to(t[4]), dbo.to(t[5]),
                dz if ctx.has_res else None, None, None)


def outer_product_mean(module, msa, mask=None, residual=None):
    """The module's forward (OuterProductMean, no interchain mask) through the Triton path; differentiable."""
    return OPMTriton.apply(msa, mask, module.ln_msa.weight, module.ln_msa.bias, module.to_left.weight, module.to_right.weight,
                           module.to_out.weight, module.to_out.bias, residual, float(module.ln_msa.eps),
                           bool(module.normalize_before_proj))
