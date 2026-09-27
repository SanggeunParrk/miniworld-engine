"""Token DiT block, TRAINING (forward + backward), fused for B200 -- v0: the fused algorithm in torch ops with an explicit
backward, bf16 GEMM operands (fp32 accumulation), fp32 parameters and residual stream, the sm_100a attention core.

Same math as modules.dit.DiTBlock with qk_norm (AF3 Alg. 23 + 25), rearranged the way the inference step is:
  * the six conditioning projections become two GEMMs: every block-internal LayerNorm of cond shares its statistics, so its
    weight folds into the projection (c_hat @ [Ws1 w1; Wb1 w1; Ws2 w2; Wb2 w2]^T), and the two output gates read raw cond
    (c @ [Wsc1; Wsc2]^T);
  * q|k|v|g is one GEMM (bias on q only);
  * the attention core is augattn_sm100 (attn_fwd2 + attn_dqb + attn_dkv) on bf16 q, k, v and a head-major bf16 bias.
Tensors are [M, *] with M = A L (B == 1). Later versions replace the torch glue with row kernels; the interface stays.
"""
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

H, DH, D = 16, 48, 768
EPS_LN = 1e-5
_AUG = Path(__file__).resolve().parents[2] / "augattn_sm100"


def _core():
    """augattn_sm100 host classes (cubins built by augattn_sm100/build.sh)."""
    if str(_AUG) not in sys.path:
        sys.path.insert(0, str(_AUG))
    import attn_op
    cwd = os.getcwd()
    os.chdir(_AUG)
    try:
        k = attn_op.kernels()
    finally:
        os.chdir(cwd)
    return attn_op, k


def pack(block):
    """fp32 master parameters of a modules.dit.DiTBlock -> the fused layout (views and small concatenations; recomputed
    each call so the optimizer's in-place updates are seen)."""
    at, tr = block.attention, block.transition
    a1, a2 = at.ada_ln_in, tr.ada_ln_in
    P = dict(
        w1=a1.ln_cond.weight, w2=a2.ln_cond.weight,
        Ws1=a1.to_scale.weight, bs1=a1.to_scale.bias, Wb1=a1.to_bias.weight,
        Ws2=a2.to_scale.weight, bs2=a2.to_scale.bias, Wb2=a2.to_bias.weight,
        Wg1=at.to_scale.weight, bg1=at.to_scale.bias, Wg2=tr.to_scale.weight, bg2=tr.to_scale.bias,
        Wq=at.to_query.weight, bq=at.to_query.bias, Wk=at.to_key.weight, Wv=at.to_value.weight, Wgt=at.to_gate.weight,
        nq=at.norm_query.weight, nk=at.norm_key.weight, eq=at.norm_query.effective_eps(torch.float32)
        if hasattr(at.norm_query, "effective_eps") else torch.finfo(torch.float32).eps,
        ek=at.norm_key.effective_eps(torch.float32) if hasattr(at.norm_key, "effective_eps") else torch.finfo(torch.float32).eps,
        Wo=at.to_out.weight,
        Wa=tr.expand_a.weight, Wb=tr.expand_b.weight, Wsq=tr.squeeze.weight,
    )
    return P


def _ln(x):
    mean = x.mean(-1, keepdim=True)
    var = (x - mean).square().mean(-1, keepdim=True)
    rstd = torch.rsqrt(var + EPS_LN)
    return (x - mean) * rstd, rstd


def _ln_bwd(dxhat, xhat, rstd):
    """d/dx of xhat = (x - mean) rstd, given dL/dxhat."""
    n = xhat.shape[-1]
    return rstd * (dxhat - dxhat.mean(-1, keepdim=True) - xhat * (dxhat * xhat).mean(-1, keepdim=True))


def _mm(a, w):
    """a [M, K] @ w[N, K]^T in bf16 operands, fp32 result."""
    return torch.mm(a.to(torch.bfloat16), w.to(torch.bfloat16).t(), out_dtype=torch.float32)


def _mmT(dy, x):
    """dW [N, K] = dy[M, N]^T @ x[M, K], bf16 operands, fp32."""
    return torch.mm(dy.to(torch.bfloat16).t(), x.to(torch.bfloat16), out_dtype=torch.float32)


class _Block(torch.autograd.Function):
    """v1: every elementwise / row step is one Triton row kernel (tdit/train_kernels.py); bf16 GEMMs through torch.mm."""
    @staticmethod
    def forward(ctx, x, c, tok, P, names, *params):
        ctx.P_dict, ctx.names = P, names
        from . import train_kernels as T
        from . import qgemm as Q
        store, bix = P["_store"], P["_b"]                                 # the hoisted pair bias of every block (_PairBiasAll)
        M, L = x.shape[0], store.L
        A, dev, bf = M // L, x.device, torch.bfloat16
        at, K = _core()
        W = _weights_cached(P, names)
        R = 8
        gr = T.grid(M, R)
        chat = torch.empty(M, 384, device=dev, dtype=bf); cbf = torch.empty_like(chat); cst = torch.empty(M, 2, device=dev)
        T._cond_prep[gr](c, chat, cbf, cst, M, N=384, BN=512, ROWS=R, eps=EPS_LN)
        G = torch.mm(chat, W["Wn"].t())                                   # [M, 3072] s1 | sh1 | s2 | sh2 (bf16)
        Gg = torch.mm(cbf, W["Wg"].t())                                   # [M, 1536] gate1 | gate2
        xa = torch.empty(M, D, device=dev, dtype=bf); xst = torch.empty(M, 2, device=dev)
        T._adaln_a[T.grid(M, 2)](x, G, G.stride(0), P["bs1"], xa, xst, M, N=D, BN=1024, ROWS=2, eps=EPS_LN, num_warps=4)
        qkvg = torch.addmm(W["bqkvg"], xa, W["Wqkvg"].t())                # [M, 3072] bf16
        qn = torch.empty(M, D, device=dev, dtype=bf); kn = torch.empty_like(qn); vc = torch.empty_like(qn)
        rqk = torch.empty(M, 32, device=dev)
        T._qknorm[T.grid(M, 4)](qkvg, P["nq"], P["nk"], qn, kn, vc, rqk, M, float(P["eq"]), float(P["ek"]), ROWS=4, num_warps=4)
        bias_hm = store.bias[bix]                                          # [H, L, L] bf16
        shp = (A, 1, L, H, DH)
        run, O, LSE = K.fwd.bind(qn.view(shp), kn.view(shp), vc.view(shp), bias_hm)
        run()
        og = torch.empty(M, D, device=dev, dtype=bf)
        T._gate_o[T.grid(M, 2)](O, qkvg, og, M, ROWS=2, num_warps=8)
        y = torch.mm(og, W["Wo"].t())
        x1 = torch.empty_like(x); xt = torch.empty(M, D, device=dev, dtype=bf); x1st = torch.empty(M, 2, device=dev)
        T._res_adaln_b[T.grid(M, 2)](x, y, Gg, Gg.stride(0), P["bg1"], G, G.stride(0), P["bs2"], x1, xt, x1st, M, N=D, BN=1024, ROWS=2, eps=EPS_LN, num_warps=4)
        ab, h = Q.swiglu_fwd(xt, W["Wab_i"])                              # one GEMM: pre-activation (saved) and h
        z = torch.mm(h, W["Wsq"].t())
        out = torch.empty_like(x)
        T._res_c[T.grid(M, 2)](x1, z, Gg, Gg.stride(0), P["bg2"], out, M, N=D, BN=1024, ROWS=2, num_warps=8)
        ctx.save_for_backward(x, c, chat, cbf, cst, G, Gg, xst, xa, qkvg, rqk, qn, kn, vc, bias_hm,
                              O, LSE, og, y, x1, x1st, xt, ab, z)
        ctx.W, ctx.shape = W, (A, L)
        return out

    @staticmethod
    def backward(ctx, dout):
        (x, c, chat, cbf, cst, G, Gg, xst, xa, qkvg, rqk, qn, kn, vc, bias_hm,
         O, LSE, og, y, x1, x1st, xt, ab, z) = ctx.saved_tensors
        store, bix = ctx.P_dict["_store"], ctx.P_dict["_b"]
        from . import train_kernels as T
        from . import qgemm as Q
        P, W = ctx.P_dict, ctx.W
        A, L = ctx.shape
        M, dev, bf = x.shape[0], x.device, torch.bfloat16
        at, K = _core()
        R = 8
        gr = T.grid(M, R)
        dout = dout.float().contiguous()
        dG = torch.empty(M, 4 * D, device=dev, dtype=bf); dGg = torch.empty(M, 2 * D, device=dev, dtype=bf)
        dz = torch.empty(M, D, device=dev, dtype=bf)
        acc = torch.zeros(5 * D + 17 * 128 + 128, device=dev)             # atomic accumulators: bg2 bs2 bg1 bs1 bq | (unused) | dnq dnk
        pbias = acc[:5 * D].view(5, D)
        T._res_c_bwd[gr](dout, z, Gg, Gg.stride(0), P["bg2"], dz, dGg, dGg.stride(0), pbias[0], M, N=D, BN=1024, ROWS=R)
        dab, h = Q.swiglu_bwd(dz, W["WsqT"], ab)                         # dh GEMM with the SwiGLU backward in its epilogue
        dWsq = _mmT(dz, h)
        dxt = torch.mm(dab, W["Wab_i"])
        dWab = _mmT(dab, xt)                                              # interleaved rows a0, b0, a1, b1, ...
        dx1 = torch.empty_like(x); dy = torch.empty(M, D, device=dev, dtype=bf)
        T._res_adaln_b_bwd[T.grid(M, 4)](dout, dxt, x1, x1st, G, G.stride(0), P["bs2"], Gg, Gg.stride(0), P["bg1"], y, dx1, dy,
                               dG, dG.stride(0), dGg, dGg.stride(0), pbias[1], pbias[2], M, N=D, BN=1024, ROWS=4, num_warps=4)
        dog = torch.mm(dy, W["Wo"])
        dWo = _mmT(dy, og)
        dqkvg = torch.empty(M, 4 * D, device=dev, dtype=bf)
        shp = (A, 1, L, H, DH)
        dob = torch.empty(M, D, device=dev, dtype=bf); dd = torch.empty(A, H, L, device=dev)
        T._gate_o_bwd[T.grid(M, 1)](dog, O, qkvg, dob, dd, dqkvg, L, M, ROWS=1, num_warps=4)
        bias_t = at.bias_transpose(bias_hm)
        rq_, DQ, DB = K.dqb.bind(qn.view(shp), kn.view(shp), vc.view(shp), dob.view(shp), bias_hm, LSE, dd, zeroed=True)
        rk_, DK, DV = K.dkv.bind(qn.view(shp), kn.view(shp), vc.view(shp), dob.view(shp), bias_t, LSE, dd, dq_zero=DQ)
        rk_(); rq_()
        nprog = triton.cdiv(M, 4)
        dwqk = acc[5 * D + 17 * 128:].view(2, 64)
        T._qknorm_bwd[(nprog,)](DQ, DK, DV, qkvg, rqk, P["nq"], P["nk"], dqkvg, dwqk, pbias[4], M, ROWS=4, num_warps=2)
        dnq, dnk = dwqk[0, :48], dwqk[1, :48]
        dxa = torch.mm(dqkvg, W["Wqkvg"])
        dWqkvg = _mmT(dqkvg, xa)
        dbq = pbias[4]
        dx = torch.empty_like(x)
        T._adaln_a_bwd[T.grid(M, 4)](dxa, x, xst, G, G.stride(0), P["bs1"], dx1, dx, dG, dG.stride(0), pbias[3], M, N=D, BN=1024, ROWS=4, num_warps=4)
        dchat = torch.mm(dG, W["Wn"])
        dWn = _mmT(dG, chat)
        dcg = torch.mm(dGg, W["Wg"])
        dWg = _mmT(dGg, cbf)
        dc = torch.empty_like(c)
        T._cond_bwd[gr](dchat, dcg, c, cst, dc, M, N=384, BN=512, ROWS=R)
        store.dbias[bix].copy_(DB)                                         # the hoisted pair bias backward runs once for all blocks
        bsum = pbias                                                      # bg2, bs2, bg1, bs1
        dWu = torch.empty_like(dWn); dw12 = torch.zeros(2, 384, device=dev)
        T._unfold_lnw[(4 * D,)](dWn, P["Ws1"], P["Wb1"], P["Ws2"], P["Wb2"], P["w1"], P["w2"], dWu, dw12, K=384, BK=512)
        dWs1, dWb1, dWs2, dWb2 = dWu.split(D)
        grads = dict(
            Ws1=dWs1, Wb1=dWb1, Ws2=dWs2, Wb2=dWb2, w1=dw12[0], w2=dw12[1],
            bs1=bsum[3], bs2=bsum[1], Wg1=dWg[:D], Wg2=dWg[D:], bg1=bsum[2], bg2=bsum[0],
            Wq=dWqkvg[:D], bq=dbq, Wk=dWqkvg[D:2 * D], Wv=dWqkvg[2 * D:3 * D], Wgt=dWqkvg[3 * D:], nq=dnq, nk=dnk,
            Wo=dWo, Wa=dWab[0::2], Wb=dWab[1::2], Wsq=dWsq,
        )
        pg = [grads.get(n) if ctx.P_dict[n].requires_grad else None for n in ctx.names]
        return (dx, dc, torch.zeros_like(store.tok), None, None, *pg)


_WCACHE = {}


def _weights_cached(P, names):
    """The bf16 pack is rebuilt only when a parameter changed (optimizer steps bump ._version)."""
    key = tuple((id(P[n]), P[n]._version) for n in names)
    ent = _WCACHE.get(id(P["Wq"]))
    if ent is None or ent[0] != key:
        ent = _WCACHE[id(P["Wq"])] = (key, _weights_bf16(P))
    return ent[1]


def _weights_bf16(P):
    bf = torch.bfloat16
    return dict(
        Wn=torch.cat([P["Ws1"] * P["w1"], P["Wb1"] * P["w1"], P["Ws2"] * P["w2"], P["Wb2"] * P["w2"]]).to(bf),
        Wg=torch.cat([P["Wg1"], P["Wg2"]]).to(bf),
        Wqkvg=torch.cat([P["Wq"], P["Wk"], P["Wv"], P["Wgt"]]).to(bf),
        bqkvg=torch.cat([P["bq"], torch.zeros(3 * D, device=P["bq"].device)]).to(bf),
        Wo=P["Wo"].to(bf), Wab_i=torch.stack([P["Wa"], P["Wb"]], 1).reshape(-1, D).to(bf), Wsq=P["Wsq"].to(bf),
        WsqT=P["Wsq"].t().contiguous().to(bf),
    )


class _Store:
    """Shared by one _PairBiasAll and the blocks it feeds: every block's bias (bf16, head-major) and their dbias (fp32)."""
    pass


class _PairBiasAll(torch.autograd.Function):
    """Every block's pair bias from ONE LayerNorm of the pair (the blocks' ln_pair share its statistics; each block's
    weight folds into its projection) and ONE GEMM: bias [NB H, L L] = W' LN(pair)^T with W' = [Wbias_b diag(wp_b)]_b.
    Backward, once for all blocks: dW' = dbias LN(pair), dLN(pair) = dbias^T W', then the LayerNorm backward. The blocks
    reach it through a 1-element token (their dbias travels in the shared store)."""
    @staticmethod
    def forward(ctx, pair, store, *wts):
        from . import train_kernels as T
        L, dev, bf = pair.shape[1], pair.device, torch.bfloat16
        R2, NB = L * L, len(wts) // 2
        p2 = pair.reshape(R2, -1)
        phat = torch.empty(R2, 128, device=dev, dtype=bf); pst = torch.empty(R2, 2, device=dev)
        T._ln_rows[(triton.cdiv(R2, 32),)](p2, phat, pst, R2, EPS_LN, ROWS=32, N=128, num_warps=4)
        Wf = torch.cat([wts[2 * b + 1] * wts[2 * b] for b in range(NB)])   # [NB H, 128]
        store.bias = torch.mm(Wf.to(bf), phat.t()).view(NB, H, L, L)       # head-major bf16
        store.dbias = torch.empty(NB, H, L, L, device=dev)
        store.L, store.tok = L, torch.zeros(1, device=dev)
        ctx.store = store
        ctx.save_for_backward(pair, phat, pst, Wf, *wts)
        return store.tok.clone()

    @staticmethod
    def backward(ctx, dtok):
        from . import train_kernels as T
        pair, phat, pst, Wf, *wts = ctx.saved_tensors
        store = ctx.store
        L, NB = store.L, len(wts) // 2
        R2, bf = L * L, torch.bfloat16
        db = store.dbias.view(NB * H, R2).to(bf)
        dWf = torch.mm(db, phat, out_dtype=torch.float32)                  # [NB H, 128]
        dph = torch.mm(db.t(), Wf.to(bf), out_dtype=torch.float32)         # [R2, 128]
        dpair = torch.empty_like(pair)
        T._ln_rows_bwd[(triton.cdiv(R2, 32),)](dph, pair.reshape(R2, -1), pst, dpair.view(R2, -1), R2, ROWS=32, N=128, num_warps=4)
        g = []
        for b in range(NB):
            wp, Wb, dW = wts[2 * b], wts[2 * b + 1], dWf[b * H:(b + 1) * H]
            g += [(dW * Wb).sum(0), dW * wp]                               # d wp_b, d Wbias_b
        return (dpair, None, *g)


def stack_forward(blocks, single, cond, pair):
    """A stack of DiT blocks (the token DiT's 24): the pair bias hoisted for all of them, each block one fused Function."""
    A, B, L, _ = single.shape
    assert B == 1
    store = _Store()
    wts = []
    for blk in blocks:
        wts += [blk.attention.ln_pair.weight, blk.attention.to_bias.weight]
    tok = _PairBiasAll.apply(pair, store, *wts)
    x, c = single.reshape(A * L, -1), cond.reshape(A * L, -1)
    for b, blk in enumerate(blocks):
        P = pack(blk)
        P["_store"], P["_b"] = store, b
        names = tuple(n for n, v in P.items() if isinstance(v, torch.Tensor))
        x = _Block.apply(x, c, tok, P, names, *[P[n] for n in names])
    return x.view(A, 1, L, -1)


def block_forward(block, single, cond, pair):
    """One block (stack of one)."""
    return stack_forward([block], single, cond, pair)


