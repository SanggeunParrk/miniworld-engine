"""OuterProductMean in Triton (portable: any GPU Triton supports), forward + backward.

The decomposition keeps the reference's statements and fuses only where the PyTorch module pays extra memory passes:

  forward   prologue  (Triton)  LN(msa) -> left / right projections -> mask            a, b [S, L, CH] bf16
            GEMM1     (cuBLAS)  O = a^T b in the grouped layout [(i, d), (j, e)]        (torch.mm, as the module's einsum)
            counts    (cuBLAS)  n = mask^T mask (fp32, clamp 1)
            epilogue  (Triton)  out[i, j] = W_out . vec(O_ij) / n_ij + bias (+ residual)   the [i,j,d,e] permute folded into the loads
  backward  dgrad     (Triton)  dzn = dz / n, dO = dzn . W_out in the grouped layout
            dA, dB    (cuBLAS)  dA = b . dO^T, dB = a . dO
            dW_out    (Triton)  sum_ij dzn_ij (x) O_ij, read from the saved grouped O (M = c, N = (d, e), split-K over pairs)
            prologue backward (Triton)  mask, dy = [dA | dB] W, dW_left / dW_right, LayerNorm backward -> dmsa, dgamma, dbeta

Tiles, warps, stages and persistent program counts are autotune axes (``_tuning.SPECS``, the engine's GRID SPEC form); the only
constants in the kernels are the model dimensions (constexpr). Offsets that scale with L^2 CH^2 or S L are 64-bit.
normalize_before_proj=True (AF3 order, the default) divides before the projection; False (ESMFold2) divides after it, bias included.
Unsupported calls (mask_interchain, non-bf16 input, dimensions that are not powers of two >= 16) are refused with the reason
(``refusal``) so the caller can take the module's own statements.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    from . import _tuning as T
except ImportError:                       # run as a script from this directory
    import _tuning as T


def _nsm():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


# ---------------------------------------------------------------------------------------------------- kernels
@triton.autotune(configs=T.configs("opm_prologue_triton"), key=["shape_key", "CM", "CH", "HAS_MASK"])
@triton.jit
def _prologue_kernel(M, MASK, LNW, LNB, WL, WR, A, B, T_, eps, shape_key,
                     CM: tl.constexpr, CH: tl.constexpr, HAS_MASK: tl.constexpr, BT: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid.to(tl.int64) * BT + tl.arange(0, BT)
    rm = rows < T_
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


@triton.autotune(configs=T.configs("opm_epilogue_triton"), key=["shape_key", "CH", "CZ", "HAS_RES", "NORM_FIRST"],
                 prune_configs_by={"early_config_prune": T.prune(lambda c, a: c["BN"] <= a["CZ"] and c["BKD"] <= a["CH"])})
@triton.jit
def _epilogue_kernel(O, WOT, BIAS, NORM, RES, OUT, L, shape_key,
                     CH: tl.constexpr, CZ: tl.constexpr, HAS_RES: tl.constexpr, NORM_FIRST: tl.constexpr,
                     TI: tl.constexpr, BJ: tl.constexpr, BN: tl.constexpr, BKD: tl.constexpr):
    # tile = TI pair rows i x BJ j (M = TI BJ pairs, so W_out is re-read once per TI BJ pairs) x BN outputs; K = CH^2 in steps of
    # BKD d values (K chunk BKD CH): A[(i, j), (d, e)] = O[i CH + d, j CH + e]
    r = tl.arange(0, TI * BJ)
    i = tl.program_id(0) * TI + r // BJ
    j = tl.program_id(1) * BJ + r % BJ
    rm = (i < L) & (j < L)
    n = tl.program_id(2) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BKD * CH)
    dd = kk // CH
    e = kk % CH
    ldo = L.to(tl.int64) * CH
    arow = i.to(tl.int64) * CH * ldo + j.to(tl.int64) * CH
    acc = tl.zeros((TI * BJ, BN), dtype=tl.float32)
    for d0 in range(0, CH, BKD):
        a = tl.load(O + arow[:, None] + ((d0 + dd).to(tl.int64) * ldo + e)[None, :], mask=rm[:, None], other=0.0)
        w = tl.load(WOT + ((d0 + dd) * CH + e)[:, None] * CZ + n[None, :])
        acc += tl.dot(a, w)
    nrm = tl.load(NORM + i.to(tl.int64) * L + j, mask=rm, other=1.0)
    bias = tl.load(BIAS + n)
    if NORM_FIRST:
        z = acc / nrm[:, None] + bias[None, :]
    else:
        z = (acc + bias[None, :]) / nrm[:, None]
    ptr = (i.to(tl.int64) * L + j)[:, None] * CZ + n[None, :]
    if HAS_RES:
        z += tl.load(RES + ptr, mask=rm[:, None], other=0.0).to(tl.float32)
    tl.store(OUT + ptr, z.to(tl.bfloat16), mask=rm[:, None])


@triton.autotune(configs=T.configs("opm_dgrad_triton"), key=["shape_key", "CH", "CZ"],
                 prune_configs_by={"early_config_prune": T.prune(lambda c, a: c["BKC"] <= a["CZ"] and c["BN"] <= a["CH"] ** 2)})
@triton.jit
def _dgrad_kernel(DZ, NORM, WO, DZN, DO, L, shape_key,
                  CH: tl.constexpr, CZ: tl.constexpr, TI: tl.constexpr, BJ: tl.constexpr, BN: tl.constexpr, BKC: tl.constexpr):
    # dzn = dz / n (both orders: the projection is linear), dO[(i, d), (j, e)] = sum_c dzn[i, j, c] W_out[c, d CH + e].
    # tile = TI i x BJ j pairs x BN of the CH^2 (d, e) outputs, K = CZ in BKC steps; the n-block-0 programs also store dzn.
    r = tl.arange(0, TI * BJ)
    i = tl.program_id(0) * TI + r // BJ
    j = tl.program_id(1) * BJ + r % BJ
    rm = (i < L) & (j < L)
    pn = tl.program_id(2)
    n = pn * BN + tl.arange(0, BN)
    d = n // CH
    e = n % CH
    nrm = tl.load(NORM + i.to(tl.int64) * L + j, mask=rm, other=1.0)
    prow = (i.to(tl.int64) * L + j) * CZ
    acc = tl.zeros((TI * BJ, BN), dtype=tl.float32)
    for c0 in range(0, CZ, BKC):
        c = c0 + tl.arange(0, BKC)
        dz = tl.load(DZ + prow[:, None] + c[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        dzn = (dz / nrm[:, None]).to(tl.bfloat16)
        if pn == 0:
            tl.store(DZN + prow[:, None] + c[None, :], dzn, mask=rm[:, None])
        w = tl.load(WO + c[:, None] * (CH * CH) + n[None, :])                                    # [BKC, BN]
        acc += tl.dot(dzn, w)
    ldo = L.to(tl.int64) * CH
    orow = i.to(tl.int64) * CH * ldo + j.to(tl.int64) * CH
    tl.store(DO + orow[:, None] + (d.to(tl.int64) * ldo + e)[None, :], acc.to(tl.bfloat16), mask=rm[:, None])


@triton.autotune(configs=T.configs("opm_dwo_triton"), key=["shape_key", "CH", "CZ"], restore_value=["DWO"],
                 prune_configs_by={"early_config_prune": T.prune(lambda c, a: c["BM"] <= a["CZ"] and c["BN"] <= a["CH"] ** 2)})
@triton.jit
def _dwo_kernel(DZN, O, DWO, L, shape_key,
                CH: tl.constexpr, CZ: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    # dWo[c, (d, e)] = sum_(i, j) dzn[i, j, c] O[i CH + d, j CH + e]: an M = CZ, N = CH^2, K = L^2 GEMM, tile BM x BN, K split over
    # SPLIT ranges of i (fp32 atomics once per program). dzn is re-read CH^2 / BN times (v1: once per d, CH times).
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    d = n // CH
    e = n % CH
    sp = tl.program_id(2)
    i_per = tl.cdiv(L, SPLIT)
    ldo = L.to(tl.int64) * CH
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for ii in range(i_per):
        i = sp * i_per + ii
        if i < L:
            for j0 in range(0, L, BK):
                j = j0 + kk
                jm = j < L
                a = tl.load(DZN + (i.to(tl.int64) * L + j)[:, None] * CZ + m[None, :], mask=jm[:, None], other=0.0)      # [BK, BM]
                b = tl.load(O + i.to(tl.int64) * CH * ldo + j[:, None].to(tl.int64) * CH + (d.to(tl.int64) * ldo + e)[None, :],
                            mask=jm[:, None], other=0.0)                                                              # [BK, BN]
                acc += tl.dot(tl.trans(a), b)
    tl.atomic_add(DWO + m[:, None] * (CH * CH) + n[None, :], acc)


@triton.autotune(configs=T.configs("opm_prologue_bwd_triton"), key=["shape_key", "CM", "CH", "HAS_MASK"],
                 restore_value=["DWL", "DWR", "DG", "DBETA"])
@triton.jit
def _prologue_bwd_kernel(M, MASK, LNW, LNB, WL, WR, DA, DB, DM, DWL, DWR, DG, DBETA, T_, eps, shape_key,
                         CM: tl.constexpr, CH: tl.constexpr, HAS_MASK: tl.constexpr, BT: tl.constexpr, PPS: tl.constexpr):
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
    nblk = tl.cdiv(T_, BT)
    for blk in range(pid, nblk, nprog):
        rows = blk.to(tl.int64) * BT + tl.arange(0, BT)
        rm = rows < T_
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
def refusal(module, msa, mask=None, residual=None, token_asym_id=None) -> str | None:
    """None if the Triton path serves this OuterProductMean call, else why not."""
    if module.mask_interchain and token_asym_id is not None:
        return "mask_interchain is applied after the projection and is not fused"
    if not msa.is_cuda:
        return "the input is not on a CUDA device"
    if msa.dtype != torch.bfloat16 or (residual is not None and residual.dtype != torch.bfloat16):
        return f"the Triton path is bf16, got {msa.dtype}"
    return T.pow2_at_least_16(d_msa=module.to_left.weight.shape[1], d_hidden=module.to_left.weight.shape[0],
                              d_pair=module.to_out.weight.shape[0])


def _forward(msa, mask, lnw, lnb, wl, wr, wo, bo, eps, norm_first, residual):
    S, L, CM = msa.shape
    CH, CZ = wl.shape[0], wo.shape[0]
    T_ = S * L
    dev = msa.device
    bf = torch.bfloat16
    x = msa.reshape(T_, CM).contiguous()
    m8 = mask.reshape(T_).contiguous().view(torch.uint8) if mask is not None else x
    a = torch.empty(T_, CH, device=dev, dtype=bf)
    b = torch.empty(T_, CH, device=dev, dtype=bf)
    key = T.shape_key(S, L)
    _prologue_kernel[lambda M_: (triton.cdiv(T_, M_["BT"]),)](x, m8, lnw, lnb, wl, wr, a, b, T_, eps, key, CM=CM, CH=CH,
                                                             HAS_MASK=mask is not None)
    o = torch.mm(a.view(S, L * CH).t(), b.view(S, L * CH))                     # [(i, d), (j, e)]
    if mask is not None:
        mf = mask.reshape(S, L).float()
        norm = (mf.t() @ mf).clamp_(min=1).contiguous()
    else:
        norm = torch.full((L, L), float(S), device=dev, dtype=torch.float32)
    out = torch.empty(L * L, CZ, device=dev, dtype=bf)
    woT = wo.t().contiguous()
    res = residual.reshape(L * L, CZ).contiguous() if residual is not None else out
    grid = lambda M_: (triton.cdiv(L, M_["TI"]), triton.cdiv(L, M_["BJ"]), triton.cdiv(CZ, M_["BN"]))  # noqa: E731
    _epilogue_kernel[grid](o, woT, bo, norm, res, out, L, key, CH=CH, CZ=CZ, HAS_RES=residual is not None, NORM_FIRST=norm_first)
    return out.view(L, L, CZ), (x, m8 if mask is not None else None, a, b, o, norm)


class OPMTriton(torch.autograd.Function):
    """One MSA stack (no batch dim): msa [S, L, CM], mask [S, L] or None, residual [L, L, CZ] or None."""

    @staticmethod
    def forward(ctx, msa, mask, lnw, lnb, wl, wr, wo, bo, residual, eps, norm_first):
        bf = torch.bfloat16
        prm = [t.detach().to(bf).contiguous() for t in (wl, wr, wo)]
        lnf = [t.detach().float().contiguous() for t in (lnw, lnb)]
        out, saved = _forward(msa, mask, *lnf, *prm, bo.detach().float().contiguous(), eps, norm_first, residual)
        x, m8, a, b, o, norm = saved
        ctx.save_for_backward(x, a, b, o, norm, *lnf, *prm, *([m8] if m8 is not None else []))
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
        S, L, CM = ctx.shape
        CH, CZ = wl.shape[0], wo.shape[0]
        T_ = S * L
        dev = x.device
        bf = torch.bfloat16
        key = T.shape_key(S, L)
        dzc = dz.reshape(L * L, CZ).contiguous()
        dzn = torch.empty(L * L, CZ, device=dev, dtype=bf)
        dO = torch.empty(L * CH, L * CH, device=dev, dtype=bf)
        grid = lambda M_: (triton.cdiv(L, M_["TI"]), triton.cdiv(L, M_["BJ"]), triton.cdiv(CH * CH, M_["BN"]))  # noqa: E731
        _dgrad_kernel[grid](dzc, norm, wo, dzn, dO, L, key, CH=CH, CZ=CZ)
        if ctx.norm_first:
            dbo = dzc.float().sum(0)
        else:                                                          # bias divided by n too
            dbo = (dzc.float() / norm.reshape(L * L, 1)).sum(0)
        dA = torch.mm(b.view(S, L * CH), dO.t())                       # [S, (i, d)]
        dB = torch.mm(a.view(S, L * CH), dO)                           # [S, (j, e)]
        dwo = torch.zeros(CZ, CH * CH, device=dev, dtype=torch.float32)
        grid = lambda M_: (triton.cdiv(CZ, M_["BM"]), triton.cdiv(CH * CH, M_["BN"]), M_["SPLIT"])  # noqa: E731
        _dwo_kernel[grid](dzn, o, dwo, L, key, CH=CH, CZ=CZ)
        dm = torch.empty(T_, CM, device=dev, dtype=bf)
        red = torch.zeros(2 * CH * CM + 2 * CM, device=dev, dtype=torch.float32)
        dwl, dwr = red[:CH * CM].view(CH, CM), red[CH * CM:2 * CH * CM].view(CH, CM)
        dg, dbeta = red[2 * CH * CM:2 * CH * CM + CM], red[2 * CH * CM + CM:]
        nsm = _nsm()
        grid = lambda M_: (min(triton.cdiv(T_, M_["BT"]), M_["PPS"] * nsm),)  # noqa: E731
        _prologue_bwd_kernel[grid](x, m8 if m8 is not None else x, lnw, lnb, wl, wr, dA.view(T_, CH), dB.view(T_, CH), dm, dwl, dwr, dg,
                                   dbeta, T_, ctx.eps, key, CM=CM, CH=CH, HAS_MASK=m8 is not None)
        t = ctx.dtypes
        return (dm.view(ctx.shape), None, dg.to(t[0]), dbeta.to(t[1]), dwl.to(t[2]), dwr.to(t[3]), dwo.to(t[4]), dbo.to(t[5]),
                dz if ctx.has_res else None, None, None)


def outer_product_mean(module, msa, mask=None, residual=None):
    """``residual + OPM(msa)`` (or ``OPM(msa)``) of an OuterProductMean through the Triton path; differentiable. Batches are run per
    stack (the counts and the grouped GEMM are per stack). Raises NotImplementedError with the reason for an unsupported call."""
    why = refusal(module, msa, mask, residual)
    if why is not None:
        raise NotImplementedError(why)
    args = (module.ln_msa.weight, module.ln_msa.bias, module.to_left.weight, module.to_right.weight, module.to_out.weight,
            module.to_out.bias)
    outs = [OPMTriton.apply(msa[bi], None if mask is None else mask[bi], *args, None if residual is None else residual[bi],
                            float(module.ln_msa.eps), bool(module.normalize_before_proj)) for bi in range(msa.shape[0])]
    return torch.stack(outs, 0)
