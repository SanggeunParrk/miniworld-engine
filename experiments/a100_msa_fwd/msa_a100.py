"""A100 (sm_80) MSA forward: OuterProductMean and MSAPairWeightedAveraging (inference).

OPM:  K_P (CUDA: LN + left/right projections + mask -> a, b [S, L, 32])  ->  GEMM1 O = a^T b (cuBLAS, [(i,d), (j,e)])
      -> K_E (CUDA: per-pair W_out . vec(O_ij) / n_ij + bias + residual; the [i,j,d,e] permute folded into its loads).

    ext = build()
    pk = pack_opm(module)                         # OuterProductMean (d_msa 64, d_hidden 32, d_pair 128, normalize_before_proj)
    out = opm_forward(ext, msa, mask, pk, residual)   # msa [1,S,L,64] bf16, mask [1,S,L] bool, residual [1,L,L,128] or None

PWA:  K_Z (CUDA: LN_z + proj_z + key mask + softmax -> w [8, L, L])  ->  K_Y (CUDA: LN_m + value projection -> v [8][L][S*32])
      -> K_M (CUDA: contraction over j per head, gate from LN(msa) recomputed in-kernel, sigmoid x o, out-projection summed over
         heads, residual).

    pk = pack_pwa(module)                         # MSAPairWeightedAveraging (64, 128, 8 heads x 32)
    out = pwa_forward(ext, msa, pair, mask, pk)   # msa [1,S,L,64], pair [1,L,L,128], mask [1,L] bool or None -> msa + pwa
"""
import hashlib
import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_EXT = {}


def build(verbose=False, extra=()):
    src_hash = hashlib.sha1(b"".join(f.read_bytes() for f in sorted((HERE / "csrc").glob("*.cu*")))).hexdigest()[:12]
    key = tuple(extra)
    if key in _EXT:
        return _EXT[key]
    from torch.utils.cpp_extension import load
    name = "a100_msa_fwd" + ("_" + hashlib.sha1(repr(key).encode()).hexdigest()[:8] if key else "")
    bdir = Path(os.environ.get("A100_MSA_BUILD", Path.home() / ".cache/miniworld-a100/ext")) / name
    bdir.mkdir(parents=True, exist_ok=True)
    _EXT[key] = load(name=name, sources=[str(HERE / "csrc/ops.cu")], build_directory=str(bdir), verbose=verbose,
                     extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80", "-lineinfo", "-Xptxas=-v",
                                        f"-DA100_SRC_HASH=0x{src_hash}", *extra],
                     extra_cflags=["-O3"])
    return _EXT[key]


# ---------------------------------------------------------------------------------------------------- OPM
@torch.no_grad()
def pack_opm(m):
    f = lambda t: t.detach().float()  # noqa: E731
    assert m.normalize_before_proj and not m.mask_interchain
    wl, wr = f(m.to_left.weight), f(m.to_right.weight)                  # [32, 64]
    assert wl.shape == (32, 64) and m.to_out.weight.shape == (128, 1024)
    g, b = f(m.ln_msa.weight), f(m.ln_msa.bias)
    w = torch.cat([wl, wr], 0)                                           # [64 out, 64 in]
    wo = m.to_out.weight.detach().to(torch.bfloat16)
    return dict(w_in=(w * g[None, :]).to(torch.bfloat16).contiguous(), b_in=(w @ b).contiguous(),
                w_out=wo.contiguous(), b_out=f(m.to_out.bias).contiguous(),
                w_out_t=wo.view(128, 32, 32).transpose(1, 2).reshape(128, 1024).contiguous(),     # K order (e, d): the transposed O
                eps=float(m.ln_msa.eps))


def opm_forward(ext, msa, mask, pk, residual=None, bufs=None, transposed=None):
    """transposed (default: OPM_T=1 when grad is off): GEMM1 as O^T = b^T a (cuBLAS picks a 1-3% faster kernel); the training
    path keeps O = a^T b, which opm_dwo reads."""
    if bufs is None:
        bufs = {}
    if transposed is None:
        transposed = os.environ.get("OPM_T", "1") != "0" and not bufs.get("train", False)
    B, S, L, C = msa.shape
    assert B == 1 and C == 64 and msa.dtype == torch.bfloat16 and S % 32 == 0 and L % 32 == 0
    T = S * L
    if bufs is None:
        bufs = {}
    if bufs.get("key") != (S, L):
        dev = msa.device
        bufs.update(key=(S, L), a=torch.empty(S, 32 * L, device=dev, dtype=torch.bfloat16),
                    b=torch.empty(S, 32 * L, device=dev, dtype=torch.bfloat16),
                    o=torch.empty(32 * L, 32 * L, device=dev, dtype=torch.bfloat16),
                    bits=torch.empty(L, S // 32, device=dev, dtype=torch.int32))
    a, b, o, bits = bufs["a"], bufs["b"], bufs["o"], bufs["bits"]
    m8 = mask.reshape(T).view(torch.uint8) if mask is not None else torch.empty(0, dtype=torch.uint8, device=msa.device)
    ext.opm_prologue(msa.reshape(T, C), m8, pk["w_in"], pk["b_in"], a, b, pk["eps"], 0)
    ext.opm_maskbits(m8, bits, S, L)
    if transposed:
        torch.mm(b.t(), a, out=o)                                             # O^T[(j, e), (i, d)]
    else:
        torch.mm(a.t(), b, out=o)
    out = torch.empty(L * L, 128, device=msa.device, dtype=torch.bfloat16)
    res = residual.reshape(L * L, 128) if residual is not None else torch.empty(0, device=msa.device, dtype=torch.bfloat16)
    ext.opm_epilogue(o, pk["w_out_t"] if transposed else pk["w_out"], pk["b_out"], bits, res, out, L, 0, int(transposed))
    return out.view(1, L, L, 128)


# ---------------------------------------------------------------------------------------------------- OPM training
@torch.no_grad()
def pack_opm_train(m):
    pk = pack_opm(m)
    f = lambda t: t.detach().float()  # noqa: E731
    pk.update(w_raw=torch.cat([f(m.to_left.weight), f(m.to_right.weight)], 0).to(torch.bfloat16).contiguous(),
              gamma=f(m.ln_msa.weight).contiguous(), beta=f(m.ln_msa.bias).contiguous(),
              woT=m.to_out.weight.detach().t().to(torch.bfloat16).contiguous())
    return pk


def opm_train_forward(ext, msa, mask, pk, residual=None, bufs=None):
    """Training forward: opm_forward, keeping a, b, O and the mask bits for opm_backward (O: 1.2 GB at L768, S1024)."""
    if bufs is None:
        bufs = {}
    bufs["train"] = True
    out = opm_forward(ext, msa, mask, pk, residual, bufs, transposed=False)
    return out


def opm_backward(ext, dz, msa, mask, pk, bufs):
    """-> dict of gradients: msa, residual, ln_w, ln_b, w_left, w_right, w_out, b_out (fp32 for the weights)."""
    B, S, L, C = msa.shape
    T = S * L
    dev = msa.device
    a, b, o, bits = bufs["a"], bufs["b"], bufs["o"], bufs["bits"]
    if bufs.get("bkey") != (S, L):
        bufs.update(bkey=(S, L), dzn=torch.empty(L * L, 128, device=dev, dtype=torch.bfloat16),
                    dO=torch.empty(32 * L, 32 * L, device=dev, dtype=torch.bfloat16),
                    dA=torch.empty(S, 32 * L, device=dev, dtype=torch.bfloat16), dB=torch.empty(S, 32 * L, device=dev, dtype=torch.bfloat16),
                    red=torch.empty(128 + 128 * 1024 + 64 * 64 + 128, device=dev, dtype=torch.float32))
    red = bufs["red"]
    red.zero_()
    dbo, dwo = red[:128], red[128:128 + 128 * 1024].view(128, 1024)
    dw, dgam, dbet = red[128 + 128 * 1024:128 + 128 * 1024 + 4096].view(64, 64), red[-128:-64], red[-64:]
    dzf = dz.reshape(L * L, 128).contiguous()
    ext.opm_dgrad(dzf, pk["woT"], bits, bufs["dzn"], bufs["dO"], dbo, L)
    torch.mm(b, bufs["dO"].t(), out=bufs["dA"])                   # dA[s, (i,d)] = sum_(j,e) dO[(i,d),(j,e)] b[s,(j,e)]
    torch.mm(a, bufs["dO"], out=bufs["dB"])                       # dB[s, (j,e)] = sum_(i,d) a[s,(i,d)] dO[(i,d),(j,e)]
    ext.opm_dwo(bufs["dzn"], o, dwo, L, 0)
    m8 = mask.reshape(T).view(torch.uint8) if mask is not None else torch.empty(0, dtype=torch.uint8, device=dev)
    dmsa = torch.empty(T, 64, device=dev, dtype=torch.bfloat16)
    ext.opm_pbwd(msa.reshape(T, 64), m8, bufs["dA"].view(T, 32), bufs["dB"].view(T, 32), pk["w_raw"], pk["gamma"], pk["beta"],
                 dmsa, dw, dgam, dbet, pk["eps"], 0)
    return dict(msa=dmsa.view(B, S, L, C), residual=dz, ln_w=dgam, ln_b=dbet, w_left=dw[:32], w_right=dw[32:], w_out=dwo, b_out=dbo)


def opm_forward_pipelined(ext, msa, mask, pk, residual=None, bufs=None, ci=64, ring=2):
    """GEMM1 split into i-chunks of ci rows (O_c = a[:, i-chunk]^T b, [32 ci, 32 L]) on one stream and each chunk's epilogue on a
    second, over `ring` O slots: the epilogue of chunk c overlaps GEMM1 of chunk c + 1, and for small ci the O chunk is read back
    from L2 instead of DRAM (v1 wrote and re-read all of O: 1.2 GB each way at L768)."""
    B, S, L, C = msa.shape
    assert B == 1 and C == 64 and L % ci == 0 and ci % 4 == 0
    T = S * L
    if bufs is None:
        bufs = {}
    dev = msa.device
    key = (S, L, ci, ring)
    if bufs.get("pkey") != key:
        bufs.update(pkey=key, a=torch.empty(S, 32 * L, device=dev, dtype=torch.bfloat16),
                    b=torch.empty(S, 32 * L, device=dev, dtype=torch.bfloat16),
                    orr=[torch.empty(32 * ci, 32 * L, device=dev, dtype=torch.bfloat16) for _ in range(ring)],
                    bits=torch.empty(L, S // 32, device=dev, dtype=torch.int32),
                    st=[torch.cuda.Stream(device=dev) for _ in range(2)])
    a, b, orr, bits = bufs["a"], bufs["b"], bufs["orr"], bufs["bits"]
    s_mm, s_ep = bufs["st"]
    cur = torch.cuda.current_stream()
    m8 = mask.reshape(T).view(torch.uint8) if mask is not None else torch.empty(0, dtype=torch.uint8, device=dev)
    ext.opm_prologue(msa.reshape(T, C), m8, pk["w_in"], pk["b_in"], a, b, pk["eps"], 0)
    ext.opm_maskbits(m8, bits, S, L)
    out = torch.empty(L * L, 128, device=dev, dtype=torch.bfloat16)
    res = residual.reshape(L * L, 128) if residual is not None else torch.empty(0, device=dev, dtype=torch.bfloat16)
    s_mm.wait_stream(cur)
    s_ep.wait_stream(cur)
    n = L // ci
    ep_done = [None] * n
    for c in range(n):
        sl = c % ring
        with torch.cuda.stream(s_mm):
            if c >= ring:
                s_mm.wait_event(ep_done[c - ring])
            torch.mm(a[:, 32 * ci * c:32 * ci * (c + 1)].t(), b, out=orr[sl])
            ev = torch.cuda.Event()
            ev.record(s_mm)
        with torch.cuda.stream(s_ep):
            s_ep.wait_event(ev)
            ext.opm_epilogue(orr[sl], pk["w_out"], pk["b_out"], bits, res, out, L, ci * c, 0)
            ep_done[c] = torch.cuda.Event()
            ep_done[c].record(s_ep)
    cur.wait_stream(s_mm)
    cur.wait_stream(s_ep)
    return out.view(1, L, L, 128)


def opm_sol_us(S, L, bw=1.602e12, tc=240e12):
    """Floor of the whole op: GEMM1 + GEMM2 FLOP at the measured tensor ceiling vs essential bytes (msa in, pair in / out)."""
    flop = 2 * (32 * L) ** 2 * S + 2 * L * L * 1024 * 128
    byt = S * L * 64 * 2 + 2 * L * L * 128 * 2
    return dict(flop_us=flop / tc * 1e6, byte_us=byt / bw * 1e6, total=max(flop / tc, byt / bw) * 1e6)


# ---------------------------------------------------------------------------------------------------- PWA
@torch.no_grad()
def pack_pwa(m):
    f = lambda t: t.detach().float()  # noqa: E731
    assert m.n_head == 8 and m.to_value.weight.shape == (256, 64) and m.to_bias.weight.shape == (8, 128)
    gm, bm = f(m.ln_msa.weight), f(m.ln_msa.bias)
    gz, bz = f(m.ln_pair.weight), f(m.ln_pair.bias)
    wv, wg, wb = f(m.to_value.weight), f(m.to_gate.weight), f(m.to_bias.weight)
    bf = lambda t: t.to(torch.bfloat16).contiguous()  # noqa: E731
    return dict(wv=bf(wv * gm[None]), bv=(wv @ bm).contiguous(), wg=bf(wg * gm[None]), bg=(wg @ bm).contiguous(),
                wb=bf(wb * gz[None]), bb=(wb @ bz).contiguous(), wof=_wo_fragments(bf(f(m.to_out.weight))),
                wgf=_wg_fragments(bf(wg * gm[None])), wo=bf(f(m.to_out.weight)),
                eps_m=float(m.ln_msa.eps), eps_z=float(m.ln_pair.eps))


def _wo_fragments(wo):
    """Wo [64, 256] bf16 -> [8 h][32 lanes][32] int32: lane (g, q) element e = (nt 2 + kc) 2 + half is the m16n8k16 B register
    {Wo[8 nt + g][32 h + 16 kc + 8 half + 2 q], +1} of head h's out-projection (K = d, N = output channel)."""
    wi = wo.contiguous().view(torch.int32)                                   # [64, 128] bf16 pairs
    dev = wo.device
    h = torch.arange(8, device=dev).view(8, 1, 1)
    lane = torch.arange(32, device=dev).view(1, 32, 1)
    e = torch.arange(32, device=dev).view(1, 1, 32)
    nt, kc, half = e // 4, (e // 2) % 2, e % 2
    row = 8 * nt + lane // 4
    col = (32 * h + 16 * kc + 8 * half + 2 * (lane % 4)) // 2
    return wi[row.expand(8, 32, 32), col.expand(8, 32, 32)].contiguous()


def _wg_fragments(wg):
    """Wg' [256, 64] bf16 -> [8 h][32 lanes][32] int32: lane (g, q) element e = (dt 4 + kc) 2 + half is the m16n8k16 B register
    {Wg'[32 h + 8 dt + g][16 kc + 8 half + 2 q], +1} of head h's gate GEMM (K = the 64 y channels, N = d)."""
    wi = wg.contiguous().view(torch.int32)                                   # [256, 32] bf16 pairs
    dev = wg.device
    h = torch.arange(8, device=dev).view(8, 1, 1)
    lane = torch.arange(32, device=dev).view(1, 32, 1)
    e = torch.arange(32, device=dev).view(1, 1, 32)
    dt, kc, half = e // 8, (e // 2) % 4, e % 2
    row = 32 * h + 8 * dt + lane // 4
    col = (16 * kc + 8 * half + 2 * (lane % 4)) // 2
    return wi[row.expand(8, 32, 32), col.expand(8, 32, 32)].contiguous()


def pwa_forward(ext, msa, pair, mask, pk, bufs=None, split=False, compact=None):
    B, S, L, C = msa.shape
    assert B == 1 and C == 64 and msa.dtype == torch.bfloat16 and pair.shape == (1, L, L, 128)
    if bufs is None:
        bufs = {}
    if bufs.get("key") != (S, L):
        dev = msa.device
        bufs.update(key=(S, L), w=torch.empty(8, L, L, device=dev, dtype=torch.bfloat16),
                    v=torch.empty(8, L, S * 32, device=dev, dtype=torch.bfloat16))
    w, v = bufs["w"], bufs["v"]
    m8 = mask.reshape(L).view(torch.uint8) if mask is not None else torch.empty(0, dtype=torch.uint8, device=msa.device)
    x = msa.reshape(S * L, 64)
    gate_out = split and ext.ctr_nogate()     # default build: the gate (and its LN) run in pwa_out2, no y buffer
    # key compaction (gate-in-out path only): w / v / the contraction's K cover the n valid keys; w, v keep their [.., L, ..] shapes
    if compact is None:
        compact = os.environ.get("PWA_COMPACT", "1") != "0"
    compact = compact and gate_out
    none = torch.empty(0, dtype=torch.int32, device=msa.device)
    if compact:
        if "kidx" not in bufs:
            bufs.update(kidx=torch.empty(L, dtype=torch.int32, device=msa.device), kcnt=torch.empty(2, dtype=torch.int32, device=msa.device),
                        kpos=torch.empty(L, dtype=torch.int32, device=msa.device))
        kidx, kcnt = bufs["kidx"], bufs["kcnt"]
        ext.pwa_compact(m8, kidx, kcnt, L, bufs["kpos"])
    else:
        kidx, kcnt = none, none
    bufs["compacted"] = compact
    cur = torch.cuda.current_stream()
    side = bufs.get("side") if os.environ.get("PWA_SERIAL", "0") == "0" else None
    if side is None and os.environ.get("PWA_SERIAL", "0") == "0":
        side = bufs["side"] = torch.cuda.Stream(device=msa.device)
    if side is not None:     # pair side (w) and value side (v, y) are independent: pair runs on a side stream, joined before K_C
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            ext.pwa_pair(pair.reshape(L * L, 128), m8, pk["wb"], pk["bb"], w, kidx, kcnt, L, pk["eps_z"])
    else:
        ext.pwa_pair(pair.reshape(L * L, 128), m8, pk["wb"], pk["bb"], w, kidx, kcnt, L, pk["eps_z"])
    if split and bufs.get("ukey") != (S, L):
        bufs.update(ukey=(S, L), u=torch.empty(S * L, 256, device=msa.device, dtype=torch.bfloat16),
                    y=torch.empty(0 if gate_out else S * L * 64, device=msa.device, dtype=torch.bfloat16).view(-1, 64))
    y = bufs["y"] if split else torch.empty(0, device=msa.device, dtype=torch.bfloat16)
    ext.pwa_value(x, pk["wv"], pk["bv"], v, y, kidx, kcnt, S, L, pk["eps_m"], 0)
    if side is not None:
        cur.wait_stream(side)
    if split:       # K_C (contraction + gate from y -> u) then K_O (out = msa + u Wo^T, one rounding)
        ext.pwa_ctr(y, w, v, pk["wgf"], pk["bg"], bufs["u"], kcnt, S, L, pk["eps_m"])
        out = torch.empty_like(x)
        if gate_out:              # bufs["u"] holds o (fragment order); the gate and LN(msa) run in pwa_out2
            keep = bufs.get("keep")
            ext.pwa_out2(x, bufs["u"], pk["wg"], pk["bg"], pk["wo"], out, pk["eps_m"], 0,
                         keep if keep is not None else torch.empty(0, device=msa.device, dtype=torch.bfloat16), bufs.get("scale", 1.0), L)
        else:
            ext.pwa_out(x, bufs["u"], pk["wo"], out, 0)
        return out.view(1, S, L, 64)
    out = torch.empty_like(x)
    ext.pwa_main(x, w, v, pk["wg"], pk["bg"], pk["wof"], out, S, L, pk["eps_m"])
    return out.view(1, S, L, 64)


# ---------------------------------------------------------------------------------------------------- PWA training
@torch.no_grad()
def pack_pwa_train(m):
    pk = pack_pwa(m)
    f = lambda t: t.detach().float()  # noqa: E731
    pk.update(gm=f(m.ln_msa.weight).contiguous(), bm=f(m.ln_msa.bias).contiguous(), gz=f(m.ln_pair.weight).contiguous(),
              bz=f(m.ln_pair.bias).contiguous(), wv_raw=f(m.to_value.weight), wg_raw=f(m.to_gate.weight), wb_raw=f(m.to_bias.weight),
              wo_raw=f(m.to_out.weight), p_drop=float(m.drop_msa.p_drop))
    return pk


def pwa_train_forward(ext, msa, pair, mask, pk, bufs, keep=None):
    """Training forward of the split path (dense keys): keeps w, v, o in bufs. keep: [L, 64] bf16 0/1 dropout mask (None: no dropout)."""
    L = msa.shape[2]
    if keep is not None:
        bufs.update(keep=keep.contiguous(), scale=1.0 / (1.0 - pk["p_drop"]))
    else:
        bufs.pop("keep", None)
        bufs["scale"] = 1.0
    return pwa_forward(ext, msa, pair, mask, pk, bufs, split=True, compact=None)      # compaction per PWA_COMPACT (default on)


def _o_natural(o, T):
    """o in pwa_ctr's fragment order (per head: position 8 q + 2 dt + e holds d = 8 dt + 2 q + e) -> natural [T, 256]."""
    return o.view(T, 8, 4, 4, 2).permute(0, 1, 3, 2, 4).reshape(T, 256)


def _ln_bwd(xh, rstd, dxh):
    return rstd * (dxh - dxh.mean(-1, keepdim=True) - xh * (dxh * xh).mean(-1, keepdim=True))


def pwa_backward_ref(dres, msa, pair, mask, pk, bufs):
    """The backward decomposition the kernels implement, in plain torch (fp32) on the forward's saved w, v, o. -> gradient dict."""
    B, S, L, C = msa.shape
    T = S * L
    f32 = torch.float32
    x = msa.reshape(T, 64).float()
    mu, var = x.mean(-1, keepdim=True), x.var(-1, unbiased=False, keepdim=True)
    rs = torch.rsqrt(var + pk["eps_m"])
    xh = (x - mu) * rs
    y = xh * pk["gm"] + pk["bm"]
    g = torch.sigmoid(y @ pk["wg_raw"].t())                                         # [T, 256]
    o = _o_natural(bufs["u"], T).float()
    u = g * o
    dr = dres.reshape(T, 64).float()
    keep = bufs.get("keep")
    dout = dr * (keep.float().repeat(S, 1) * bufs["scale"]) if keep is not None else dr
    dWo = dout.t() @ u                                                               # [64, 256]
    du = dout @ pk["wo_raw"]                                                         # [T, 256]
    do = du * g
    dgp = du * o * g * (1 - g)
    w = bufs["w"].float()                                                            # [8, L, L]
    do_h = do.view(S, L, 8, 32).permute(2, 1, 0, 3).reshape(8, L, S * 32)           # head-major [h][i][(s, d)]
    v = bufs["v"].float()                                                            # [8][L][(s, d)]
    dv_h = torch.bmm(w.transpose(1, 2), do_h)                                        # [h][j][(s, d)]
    dw = torch.bmm(do_h, v.transpose(1, 2))                                          # [h][i][j]
    dv = dv_h.view(8, L, S, 32).permute(2, 1, 0, 3).reshape(T, 256)
    dWg, dWv = dgp.t() @ y, dv.t() @ y
    dy = dgp @ pk["wg_raw"] + dv @ pk["wv_raw"]
    dgm, dbm = (dy * xh).sum(0), dy.sum(0)
    dmsa = _ln_bwd(xh, rs, dy * pk["gm"]) + dr
    # pair side: dlogit = w (dw - sum_j w dw); logits = LN_z(z) Wb^T (masked keys: constant -> no gradient, and w = 0 there)
    dl = w * (dw - (w * dw).sum(-1, keepdim=True))                                   # [8, L(i), L(j)]
    z = pair.reshape(L * L, 128).float()
    zm, zv = z.mean(-1, keepdim=True), z.var(-1, unbiased=False, keepdim=True)
    zr = torch.rsqrt(zv + pk["eps_z"])
    zh = (z - zm) * zr
    zy = zh * pk["gz"] + pk["bz"]
    dlp = dl.permute(1, 2, 0).reshape(L * L, 8)                                      # [(i, j), h]
    if mask is not None:
        dlp = dlp * mask.reshape(1, L).expand(L, L).reshape(L * L, 1).float()
    dWb = dlp.t() @ zy
    dzy = dlp @ pk["wb_raw"]
    dgz, dbz = (dzy * zh).sum(0), dzy.sum(0)
    dpair = _ln_bwd(zh, zr, dzy * pk["gz"])
    return dict(msa=dmsa.view(B, S, L, C), pair=dpair.view(1, L, L, 128), ln_msa_w=dgm, ln_msa_b=dbm, w_value=dWv, w_gate=dWg,
                ln_pair_w=dgz, ln_pair_b=dbz, w_bias=dWb, w_out=dWo)


def pwa_backward(ext, dres, msa, pair, mask, pk, bufs):
    """PWA backward (split-path forward; dense or compacted keys, as the forward ran) with the CUDA kernels:
    pwa_bglue -> pwa_ctr_dv (dv = w^T do) and pwa_dw (dw = do v^T) -> pwa_bproj (msa side) and pwa_bpair (pair side)."""
    B, S, L, C = msa.shape
    T = S * L
    dev = msa.device
    bf16 = torch.bfloat16
    if bufs.get("bwkey") != (S, L):
        bufs.update(bwkey=(S, L), do_h=torch.empty(8, L, S * 32, device=dev, dtype=bf16), dgp=torch.empty(T, 256, device=dev, dtype=bf16),
                    dv=torch.empty(T, 256, device=dev, dtype=bf16),
                    dwp=torch.empty(int(os.environ.get("PWA_DW_KSPLIT", "0")) or max(1, min(8, 1152 // ((L // 128) ** 2 * 8))), 8, L, L,
                                    device=dev, dtype=torch.float32),
                    red=torch.empty(64 * 256 + 512 * 64 + 128 + 8 * 128 + 256, device=dev, dtype=torch.float32),
                    wgv=torch.cat([pk["wg_raw"], pk["wv_raw"]], 0).to(bf16).contiguous())
    red = bufs["red"]
    red.zero_()
    o_ = 0
    dwo = red[o_:o_ + 64 * 256].view(64, 256); o_ += 64 * 256
    dwgv = red[o_:o_ + 512 * 64].view(512, 64); o_ += 512 * 64
    dgm, dbm = red[o_:o_ + 64], red[o_ + 64:o_ + 128]; o_ += 128
    dwb = red[o_:o_ + 8 * 128].view(8, 128); o_ += 8 * 128
    dgz, dbz = red[o_:o_ + 128], red[o_ + 128:o_ + 256]
    keep = bufs.get("keep")
    dr = dres.reshape(T, 64).contiguous()
    x = msa.reshape(T, 64)
    none = torch.empty(0, dtype=torch.int32, device=dev)
    cp = bufs.get("compacted", False)
    kidx, kcnt, kpos = (bufs["kidx"], bufs["kcnt"], bufs["kpos"]) if cp else (none, none, none)
    if os.environ.get("PWA_BWD_V", "1") == "2":
        # v2: the gate's gradient is consumed inside pwa_bglue2 (dWg, dy_g); pwa_bproj2 handles the value side only
        if "dyg" not in bufs or bufs["dyg"].shape[0] != T:
            bufs.update(dyg=torch.empty(T, 64, device=dev, dtype=bf16), wg16=pk["wg_raw"].to(bf16).contiguous(),
                        wv16=pk["wv_raw"].to(bf16).contiguous())
        ext.pwa_bglue2(x, dr, bufs["u"], keep if keep is not None else torch.empty(0, device=dev, dtype=bf16), bufs.get("scale", 1.0),
                       bufs["wg16"], pk["gm"], pk["bm"], pk["wo"], bufs["do_h"], bufs["dyg"], dwo, dwgv[:256], S, L, pk["eps_m"], 0)
        ext.pwa_ctr_dv(bufs["w"], bufs["do_h"], bufs["dv"], S, L, kidx, kcnt)
        ext.pwa_dw(bufs["do_h"], bufs["v"], bufs["dwp"], kcnt, L)
        dmsa = torch.empty(T, 64, device=dev, dtype=bf16)
        ext.pwa_bproj2(x, dr, bufs["dyg"], bufs["dv"], bufs["wv16"], pk["gm"], pk["bm"], kpos, dmsa, dwgv[256:], dgm, dbm, L, pk["eps_m"], 0)
        dpair = torch.empty(L * L, 128, device=dev, dtype=bf16)
        ext.pwa_bpair(pair.reshape(L * L, 128), bufs["w"], bufs["dwp"], pk["wb_raw"].contiguous(), pk["gz"], pk["bz"], dpair, dwb, dgz, dbz,
                      L, pk["eps_z"], kpos, kcnt,
                      mask.reshape(L).view(torch.uint8) if mask is not None else torch.empty(0, dtype=torch.uint8, device=dev))
        return dict(msa=dmsa.view(B, S, L, C), pair=dpair.view(1, L, L, 128), ln_msa_w=dgm, ln_msa_b=dbm, w_value=dwgv[256:],
                    w_gate=dwgv[:256], ln_pair_w=dgz, ln_pair_b=dbz, w_bias=dwb, w_out=dwo)
    ext.pwa_bglue(x, dr, bufs["u"], keep if keep is not None else torch.empty(0, device=dev, dtype=bf16), bufs.get("scale", 1.0),
                  pk["wg"], pk["bg"], pk["wo"], bufs["do_h"], bufs["dgp"], dwo, S, L, pk["eps_m"], 0)
    ext.pwa_ctr_dv(bufs["w"], bufs["do_h"], bufs["dv"], S, L, kidx, kcnt)          # dv = w^T do (valid keys only when compacted)
    ext.pwa_dw(bufs["do_h"], bufs["v"], bufs["dwp"], kcnt, L)                        # dw = do v^T, fp32 split-K partials
    dmsa = torch.empty(T, 64, device=dev, dtype=bf16)
    ext.pwa_bproj(x, dr, bufs["dgp"], bufs["dv"], bufs["wgv"], pk["gm"], pk["bm"], dmsa, dwgv, dgm, dbm, pk["eps_m"], 0, kpos, L)
    dpair = torch.empty(L * L, 128, device=dev, dtype=bf16)
    ext.pwa_bpair(pair.reshape(L * L, 128), bufs["w"], bufs["dwp"], pk["wb_raw"].contiguous(), pk["gz"], pk["bz"], dpair, dwb, dgz, dbz,
                  L, pk["eps_z"], kpos, kcnt,
                  mask.reshape(L).view(torch.uint8) if mask is not None else torch.empty(0, dtype=torch.uint8, device=dev))
    return dict(msa=dmsa.view(B, S, L, C), pair=dpair.view(1, L, L, 128), ln_msa_w=dgm, ln_msa_b=dbm, w_value=dwgv[256:], w_gate=dwgv[:256],
                ln_pair_w=dgz, ln_pair_b=dbz, w_bias=dwb, w_out=dwo)


def pwa_forward_pipelined(ext, msa, pair, mask, pk, bufs=None, cs=128, ring=2):
    """Split path pipelined over s-chunks of cs sequences: value(c) -> ctr(c) -> out(c) on three streams, so the memory-bound value /
    out kernels of neighbouring chunks overlap the compute-bound contraction. v, y, u live in `ring` chunk-sized slots (small enough,
    for small cs, to stay L2-resident between producer and consumer). Every chunk is an ordinary call with S = cs: v_c [8][L][cs*32]."""
    B, S, L, C = msa.shape
    assert B == 1 and C == 64 and S % cs == 0 and cs % 16 == 0
    if bufs is None:
        bufs = {}
    key = (S, L, cs, ring)
    dev = msa.device
    if bufs.get("pkey") != key:
        bufs.update(pkey=key, w=torch.empty(8, L, L, device=dev, dtype=torch.bfloat16),
                    vr=[torch.empty(8, L, cs * 32, device=dev, dtype=torch.bfloat16) for _ in range(ring)],
                    yr=[torch.empty(cs * L, 64, device=dev, dtype=torch.bfloat16) for _ in range(ring)],
                    ur=[torch.empty(cs * L, 256, device=dev, dtype=torch.bfloat16) for _ in range(ring)],
                    st=[torch.cuda.Stream(device=dev) for _ in range(4)])
    w, vr, yr, ur = bufs["w"], bufs["vr"], bufs["yr"], bufs["ur"]
    s_pair, s_val, s_ctr, s_out = bufs["st"]
    cur = torch.cuda.current_stream()
    m8 = mask.reshape(L).view(torch.uint8) if mask is not None else torch.empty(0, dtype=torch.uint8, device=dev)
    none = torch.empty(0, dtype=torch.int32, device=dev)
    x = msa.reshape(S * L, 64)
    out = torch.empty_like(x)
    for st_ in bufs["st"]:
        st_.wait_stream(cur)
    with torch.cuda.stream(s_pair):
        ext.pwa_pair(pair.reshape(L * L, 128), m8, pk["wb"], pk["bb"], w, none, none, L, pk["eps_z"])
    s_ctr.wait_stream(s_pair)
    n = S // cs
    ctr_done, out_done = [None] * n, [None] * n
    for c in range(n):
        sl = c % ring
        xc, oc = x[c * cs * L:(c + 1) * cs * L], out[c * cs * L:(c + 1) * cs * L]
        with torch.cuda.stream(s_val):
            if c >= ring:
                s_val.wait_event(ctr_done[c - ring])          # slot's v / y consumed by ctr(c - ring)
            ext.pwa_value(xc, pk["wv"], pk["bv"], vr[sl], yr[sl], none, none, cs, L, pk["eps_m"], 0)
            ev = torch.cuda.Event()
            ev.record(s_val)
        with torch.cuda.stream(s_ctr):
            s_ctr.wait_event(ev)
            if c >= ring:
                s_ctr.wait_event(out_done[c - ring])          # slot's u consumed by out(c - ring)
            ext.pwa_ctr(yr[sl], w, vr[sl], pk["wgf"], pk["bg"], ur[sl], none, cs, L, pk["eps_m"])
            ctr_done[c] = torch.cuda.Event()
            ctr_done[c].record(s_ctr)
        with torch.cuda.stream(s_out):
            s_out.wait_event(ctr_done[c])
            ext.pwa_out(xc, ur[sl], pk["wo"], oc, 0)
            out_done[c] = torch.cuda.Event()
            out_done[c].record(s_out)
    for st_ in bufs["st"]:
        cur.wait_stream(st_)
    return out.view(1, S, L, 64)


def pwa_sol_us(S, L, bw=1.602e12, tc=240e12):
    """Floor: contraction + projection FLOP at the measured tensor ceiling vs essential bytes (msa in / out, pair in)."""
    flop = 2 * 8 * L * L * S * 32 + S * L * 2 * (64 * 256 * 2 + 256 * 64) + L * L * 2 * 128 * 8
    byt = 2 * S * L * 64 * 2 + L * L * 128 * 2
    return dict(flop_us=flop / tc * 1e6, byte_us=byt / bw * 1e6, total=max(flop / tc, byt / bw) * 1e6)
