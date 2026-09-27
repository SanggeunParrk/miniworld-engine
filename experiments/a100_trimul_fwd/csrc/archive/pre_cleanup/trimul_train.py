"""A100 TriMul training (forward + backward), v1.

forward : K1 -> contraction -> K3 (with the row-dropout scale); saves z, planes (a | b), X, ds
backward: B1 (output side: dX, d_o r_o, d_g r_i, d_g, LN statistics)
          -> weight gradients: W_o as a long-K GEMM against the raw X + rank-1 folds of LN_out; W_og = d_g^T x_n
          -> contraction backward (one custom launch: dA, dB of every channel; A100_CBWD=cublas for torch.bmm)
          -> B7src (input side, channel-stationary: dg / dp + dW of the four input projections)
          -> dx_n = [dgp | d_g] . [W_in ; W_og] (cuBLAS) -> LayerNorm_in backward + residual (torch, v1)
"""
import os

import torch

import trimul_a100 as TA

_CBWD = os.environ.get("A100_CBWD", "cublas")
_B7J = os.environ.get("A100_B7J", "1") == "1"             # the sm_90 joint algorithm (L2 ring) instead of B7src + B8
_B7J_C = {256: int(os.environ.get("A100_B7J_C256", "8")), 128: int(os.environ.get("A100_B7J_C128", "6"))}
# the sm_90 B1: G = A_o^T X^T accumulated on chip (no A_o store, no G GEMM).  Measured: a gain at CH = 128; at CH = 256 its 128 G registers
# spill and the latency-bound B1 absorbs the extra tensor work, so B1 + the split-K GEMM stays faster.  A100_B1G = list of CH, or "all" / "none".
_B1G_ENV = os.environ.get("A100_B1G", "128")
_B1G = {"all": (128, 256), "none": ()}[_B1G_ENV] if _B1G_ENV in ("all", "none") else tuple(int(c) for c in _B1G_ENV.split(","))
_OVERLAP = os.environ.get("A100_OVERLAP", "1") == "1"      # weight-gradient GEMMs on a side stream under the contraction backward
_SIDE = {}


def _side_stream(dev):
    if dev not in _SIDE:
        _SIDE[dev] = torch.cuda.Stream(device=dev)
    return _SIDE[dev]


_B7J_RINGS = int(os.environ.get("A100_B7J_RINGS", "8"))
PARAMS = ("ln_pair.weight", "ln_pair.bias", "to_left_gate.weight", "to_left.weight", "to_right_gate.weight", "to_right.weight",
          "ln_out.weight", "ln_out.bias", "to_gate.weight", "to_out.weight")


def _mm32(a, b):
    """bf16 x bf16 -> fp32 matmul (fp32 accumulate, fp32 result)."""
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    except TypeError:
        return torch.mm(a.float(), b.float())


def _gemm_tk(a, b):
    """a [T, M] (row stride lda), b [T, N] or its transposed view -> a^T b in fp32; K = T is split into chunks (a batched GEMM + a sum), since
    cuBLAS picks a non-split-K kernel for these 128 x 256 x (L^2) shapes (0.99 ms instead of ~0.2 at L = 768)."""
    T = a.shape[0]
    S = next((s for s in (32, 16, 8, 4, 2) if T % s == 0 and T // s >= 256), 1)            # ~ the DRAM floor at S = 32 (gemm_tk_bench.py)
    if S == 1:
        return _mm32(a.t(), b)
    ac = a.unflatten(0, (S, T // S)).transpose(1, 2)            # [S, M, T/S]
    bc = b.unflatten(0, (S, T // S))                              # [S, T/S, N]
    try:
        part = torch.bmm(ac, bc, out_dtype=torch.float32)
    except (TypeError, RuntimeError):
        part = torch.bmm(ac.float(), bc.float())
    return part.sum(0)


@torch.no_grad()
def pack_train(m):
    pk = TA.pack(m)                                      # inference packing (0.5-scaled K1 / K3 weights)
    f = lambda t: t.detach().float()  # noqa: E731
    ch = pk["ch"]
    nstep = ch // 16
    # unscaled K1 weights: bf16(0.5 W) * 2 is exact
    w1u = (pk["w1"].float() * 2).to(torch.bfloat16).contiguous()                 # granule-major blocks (B7src)
    wcat = w1u.view(nstep, 16, 64, 8).transpose(1, 2).reshape(4 * ch, 128).contiguous()   # row-major, K1 row order (dx GEMM)
    wo = (f(m.to_out.weight) * f(m.ln_out.weight)[None, :]).to(torch.bfloat16).contiguous()
    wg = (f(m.to_gate.weight) * f(m.ln_pair.weight)[None, :]).to(torch.bfloat16).contiguous()
    # the K1 row order: row -> (plane channel oc, gate?) for unpacking the input-weight gradients
    dev = wcat.device
    step = torch.arange(nstep, device=dev).view(-1, 1, 1)
    nw = torch.arange(4, device=dev).view(1, -1, 1)
    n = torch.arange(16, device=dev).view(1, 1, -1)
    oc = (32 * step + 8 * nw + (n % 8)).reshape(-1)
    is_gate = (n < 8).expand(nstep, 4, 16).reshape(-1)
    pk.update(w1u=w1u, wdx=torch.cat([wcat, m.to_gate.weight.detach().to(torch.bfloat16)], 0).contiguous(),
              wo_b1=wo, wg_b1=m.to_gate.weight.detach().to(torch.bfloat16).contiguous(),   # B1's gate runs on the saved x_n: raw W_og
               so_b1=wo.float().sum(1).contiguous(), bo_b1=(f(m.to_out.weight) @ f(m.ln_out.bias)).contiguous(),
              sg_b1=wg.float().sum(1).contiguous(), bg_b1=(f(m.to_gate.weight) @ f(m.ln_pair.bias)).contiguous(),
              row_oc=oc, row_gate=is_gate)
    # packed row of (weight, output channel c): gather indices for unpacking the input-projection gradients (graph-capturable)
    rows = torch.empty(4, ch, dtype=torch.long, device=dev)
    for k, (is_g, lo) in enumerate(((True, 0), (False, 0), (True, ch), (False, ch))):
        sel = torch.nonzero((is_gate == is_g) & (oc >= lo) & (oc < lo + ch)).flatten()
        rows[k, oc[sel] - lo] = sel
    pk["unpack_rows"] = rows
    return pk


class TriMulA100(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, mask, ds, ext, m, *params):
        key = tuple((t.data_ptr(), t._version) for t in params)   # repack only when a weight changed (packing syncs: never inside a capture)
        if getattr(m, "_a100_train_key", None) != key:
            m._a100_train_pk, m._a100_train_key = pack_train(m), key
        pk = m._a100_train_pk
        B, L, _, C = z.shape
        ch, T = pk["ch"], L * L
        zf = z.reshape(T, C).contiguous()
        ab = torch.empty(2 * ch, T, device=z.device, dtype=torch.bfloat16)
        x = torch.empty(ch, L, L, device=z.device, dtype=torch.bfloat16)
        mk = mask.reshape(L).to(torch.uint8).contiguous() if mask is not None else torch.empty(0, dtype=torch.uint8, device=z.device)
        none = torch.empty(0, device=z.device)
        zst = torch.empty(T, 2, device=z.device, dtype=torch.float32) if TA.ZST else none
        if TA.ZST:
            ext.k1z(zf, mk, pk["w1"], pk["g_in"], pk["b_in"], ab, L, pk["eps_in"], zst)
        else:
            ext.k1(zf, mk, pk["w1"], pk["g_in"], pk["b_in"], ab, L, pk["eps_in"], 0, none)
        TA.contract(ab[:ch].view(ch, L, L), ab[ch:].view(ch, L, L), x, pk, ext)
        out = torch.empty_like(zf)
        dsv = ds.reshape(L, C).contiguous() if ds is not None else none.to(torch.bfloat16)
        # training K3 also saves the LayerNorm statistics (mu_o, r_o, mu_i, r_i) and x_n for the backward (the sm_90 forward's saved tensors)
        stats = torch.empty(T, 4, device=z.device, dtype=torch.float32)
        xst = torch.empty(T, 2, device=z.device, dtype=torch.float32) if TA.XST else none
        if TA.XST:
            ext.xstat(x.view(ch, T), xst, pk["eps_out"])
        xn = torch.empty_like(zf)
        ext.k3_train(x.view(ch, T), zf, pk["wo"], pk["wg"], pk["so"], pk["bo"], pk["sg"], pk["bg"], out, pk["eps_out"], dsv, L, stats, xn,
                     pk["g_in"], pk["b_in"], zst, xst)
        ctx.save_for_backward(zf, ab, x, dsv, mk, stats, xn)
        ctx.ext, ctx.pk, ctx.L, ctx.m = ext, pk, L, m
        return out.view(B, L, L, C)

    @staticmethod
    def backward(ctx, dy):
        zf, ab, x, dsv, mk, stats, xn = ctx.saved_tensors
        ext, pk, L, m = ctx.ext, ctx.pk, ctx.L, ctx.m
        ch, T, C = pk["ch"], L * L, 128
        dev = zf.device
        dyf = dy.reshape(T, C).contiguous().to(torch.bfloat16)
        # ---- B1
        dX = torch.empty(ch, L, L, device=dev, dtype=torch.bfloat16)
        ao = torch.empty(T, C, device=dev, dtype=torch.bfloat16)
        joint = _B7J and T % 128 == 0
        if joint:                                   # d_g alone; dg / dp of the planes live only in the joint kernel's L2 ring
            ldd = C
            dgp = torch.empty(T, C, device=dev, dtype=torch.bfloat16)
            dgv = dgp
        else:
            ldd = 4 * ch + C
            dgp = torch.empty(T, ldd, device=dev, dtype=torch.bfloat16)    # [dg | dp of the planes (K1 order) | d_g of the output gate]
            dgv = dgp[:, 4 * ch:]
        if ch in _B1G:
            nb = ext.b1g_grid(T)
            red = torch.empty(nb, 2, C, device=dev, dtype=torch.float32)
            gpart = torch.empty(nb, C, ch, device=dev, dtype=torch.float32)
            ext.b1g(x.view(ch, T), xn, dyf, dsv, pk["wo_b1"], pk["wg_b1"], pk["so_b1"], pk["bo_b1"], stats, dX.view(ch, T), dgv, ldd, red, gpart, L)
        else:
            red = torch.empty(ext.b1_red_rows(T, ch), 2, C, device=dev, dtype=torch.float32)
            gpart = None
            ext.b1(x.view(ch, T), zf, dyf, dsv, pk["wo_b1"], pk["wg_b1"], pk["so_b1"], pk["bo_b1"], pk["sg_b1"], pk["bg_b1"],
                   dX.view(ch, T), ao, dgv, ldd, stats, red, xn, pk["g_in"], pk["b_in"], L, pk["eps_out"])

        def weight_grads():
            v_o, S_o = red.sum(0).unbind(0)
            # W_o, LN_out affine:  H = sum_t d_o r (X - mu) = A_o^T X^T - (A_o^T mu) 1^T ;  S_o = sum_t d_o
            G = gpart.sum(0) if gpart is not None else _gemm_tk(ao, x.view(ch, T).t())      # [128, CH]
            H = G - v_o[:, None]
            g_out, b_out = m.ln_out.weight.detach().float(), m.ln_out.bias.detach().float()
            Wo = m.to_out.weight.detach().float()
            d_wo = H * g_out[None, :] + S_o[:, None] * b_out[None, :]
            d_gout = (Wo * H).sum(0)
            d_bout = (Wo * S_o[:, None]).sum(0)
            # W_og (gate on the saved x_n, affine included): dW_og = d_g^T x_n, no LayerNorm fold (cuBLAS splits K: at the DRAM floor)
            d_wog = _mm32(dgv.t(), xn)
            return d_wo, d_gout, d_bout, d_wog

        # the weight-gradient GEMMs are DRAM-bound, the contraction backward is tensor-bound: run them side by side
        main = torch.cuda.current_stream()
        if _OVERLAP:
            side = _side_stream(dev)
            side.wait_stream(main)
            for t in (red, gpart, ao, x, dgv, xn):
                if t is not None:
                    t.record_stream(side)
            with torch.cuda.stream(side):
                d_wo, d_gout, d_bout, d_wog = weight_grads()
        else:
            d_wo, d_gout, d_bout, d_wog = weight_grads()
        # ---- contraction backward -> dA | dB planes
        dab = torch.empty(2 * ch, L, L, device=dev, dtype=torch.bfloat16)
        A, Bp, dA, dB, dXv = ab[:ch].view(ch, L, L), ab[ch:].view(ch, L, L), dab[:ch], dab[ch:], dX
        h = ch // 2 if pk["bidir"] else (ch if pk["outgoing"] else 0)
        if _CBWD == "cublas" or L % 128:
            if h:                                  # X = A B^T: dA = dX B, dB = dX^T A
                torch.bmm(dXv[:h], Bp[:h], out=dA[:h])
                torch.bmm(dXv[:h].transpose(1, 2), A[:h], out=dB[:h])
            if h < ch:                             # X = A^T B: dA = B dX^T, dB = A dX
                torch.bmm(Bp[h:], dXv[h:].transpose(1, 2), out=dA[h:])
                torch.bmm(A[h:], dXv[h:], out=dB[h:])
        else:                                      # one launch; modes 0 NT (P Q^T), 1 TN (P^T Q), 2 NN (P Q)
            seg = []
            if h:
                seg += [(dXv[:h], Bp[:h], dA[:h], 2), (dXv[:h], A[:h], dB[:h], 1)]
            if h < ch:
                seg += [(Bp[h:], dXv[h:], dA[h:], 0), (A[h:], dXv[h:], dB[h:], 2)]
            ext.contract_multi(*[list(t) for t in zip(*seg)], 0)
        nstep = ch // 16
        dz = torch.empty_like(zf)
        if joint:
            # ---- B7 joint: sources (dg / dp + dW per weight block) -> L2 ring -> consumers (dx_n GEMM + LN_in backward + residual)
            S, Cc = nstep, _B7J_C[ch]
            Gn = ext.b7j_capacity() // (S + Cc)
            ring = torch.empty(Gn * _B7J_RINGS * 128 * 64 * S, device=dev, dtype=torch.bfloat16)
            flags = torch.zeros(Gn * _B7J_RINGS * (S + 1), device=dev, dtype=torch.int32)
            dwp = torch.empty(Gn, 4 * ch, C, device=dev, dtype=torch.float32)
            part = torch.empty(Gn * Cc, 2, C, device=dev, dtype=torch.float32)
            ext.b7j(xn, pk["w1"], dab.view(2 * ch, T), mk, pk["wdx"], dgv, zf, dyf, stats, pk["g_in"], ring,
                    flags[:Gn * _B7J_RINGS * S], flags[Gn * _B7J_RINGS * S:], dz, dwp, part, L, Cc, Gn, _B7J_RINGS)
        else:
            # ---- B7src: dg / dp (columns 0 .. 4CH) + input-weight gradient partials
            splits = max(1, TA.num_sms(dev) // ext.b7_ctas_per_split(nstep) * (1 if ext.b7_ctas_per_split(nstep) < nstep else 2))
            dwp = torch.empty(splits, 4 * ch, C, device=dev, dtype=torch.float32)
            ext.b7src(xn, pk["w1"], dab.view(2 * ch, T), mk, dgp, ldd, dwp, L)
            # ---- B8: dx_n = [dgp | d_g] . [W_in ; W_og] fused with the LayerNorm_in backward + residual
            part = torch.empty(min((T + 127) // 128, 2 * TA.num_sms(dev)), 2, C, device=dev, dtype=torch.float32)
            ext.b8(dgp, ldd, pk["wdx"], zf, dyf, stats, pk["g_in"], dz, part)
        dw_rows = dwp.sum(0)                                                  # [4CH, 128] in K1 row order
        if _OVERLAP:
            main.wait_stream(side)
            for t in (d_wo, d_gout, d_bout, d_wog):
                t.record_stream(main)
        d_gin, d_bin = part.sum(0).unbind(0)
        # ---- unpack the input-projection gradients
        rows = pk["unpack_rows"]
        grads_in = {name: dw_rows[rows[k]] for k, name in enumerate(("to_left_gate", "to_left", "to_right_gate", "to_right"))}
        out = {"ln_pair.weight": d_gin, "ln_pair.bias": d_bin, "to_left_gate.weight": grads_in["to_left_gate"],
               "to_left.weight": grads_in["to_left"], "to_right_gate.weight": grads_in["to_right_gate"], "to_right.weight": grads_in["to_right"],
               "ln_out.weight": d_gout, "ln_out.bias": d_bout, "to_gate.weight": d_wog, "to_out.weight": d_wo}
        pgrads = [out[n].to(dict(m.named_parameters())[n].dtype) for n in PARAMS]
        return (dz.view(1, L, L, C), None, None, None, None, *pgrads)


def forward_train(ext, m, z, mask, ds):
    params = [dict(m.named_parameters())[n] for n in PARAMS]
    return TriMulA100.apply(z, mask, ds, ext, m, *params)
