"""A100 (sm_80) pair Transition backward: roles P (dA, dB, h), X (dx, dgamma, dbeta), W (dWa, dWb, dWs) -- see csrc/tr_bwd_sm80.cuh.

    ext = build()
    pk = pack(module)
    dx, dgamma, dbeta, dwa, dwb, dws = backward(ext, x, dy, pk)      # x, dy [T, 128] bf16 (T % 256 == 0)
"""
import hashlib
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
FWD = HERE.parent / "a100_transition_fwd"
sys.path.insert(0, str(FWD))
import transition_a100 as TA  # noqa: E402

COMMON = HERE.parent / "a100_trimul_fwd" / "csrc"
D, H, CH = 128, 512, 32
_EXT = {}


def build(verbose=False, extra=()):
    key = tuple(extra)
    if key in _EXT:
        return _EXT[key]
    from torch.utils.cpp_extension import load
    srcs = sorted((HERE / "csrc").glob("*.cu*")) + sorted((FWD / "csrc").glob("*.cuh")) + [COMMON / "sm80_common.cuh"]
    h = hashlib.sha1(b"".join(f.read_bytes() for f in srcs) + repr(key).encode()).hexdigest()[:12]
    name = f"a100_transition_bwd_{h}"
    bdir = Path(os.environ.get("A100_TRANSITION_BUILD", Path.home() / ".cache/miniworld-a100/ext")) / name
    bdir.mkdir(parents=True, exist_ok=True)
    _EXT[key] = load(name=name, sources=[str(HERE / "csrc/ops.cu")], build_directory=str(bdir),
                     verbose=verbose or bool(os.environ.get("A100_VERBOSE")),
                     extra_include_paths=[str(COMMON), str(FWD / "csrc"), str(HERE / "csrc")],
                     extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80", "-lineinfo", "-Xptxas=-v", *extra],
                     extra_cflags=["-O3"])
    return _EXT[key]


@torch.no_grad()
def pack(m):
    f = lambda t: t.detach().float()  # noqa: E731
    wa, wb, ws = f(m.expand_a.weight), f(m.expand_b.weight), f(m.squeeze.weight)
    dev = wa.device
    kp, op = TA.k_perm(dev), TA.o_perm(dev)
    nchunk = H // CH
    fw = TA.pack(m)                                         # the forward's W1 (0.5 Wa | Wb, k-permuted) and LN table
    w1 = fw["w"].view(nchunk, -1)[:, : 2 * CH * D]          # [chunk][16 granules][64 rows][8]
    w3 = ws.t()[:, kp].reshape(nchunk, CH, 16, 8).transpose(1, 2).reshape(nchunk, -1).to(torch.bfloat16)   # Ws^T, k-permuted
    wp = torch.cat([w1, w3], 1).contiguous().view(-1)
    xa = wa[:, op].reshape(nchunk, CH, 16, 8).transpose(1, 2).reshape(nchunk, -1)        # [chunk][16 n-granules][32 hidden][8]
    xb = wb[:, op].reshape(nchunk, CH, 16, 8).transpose(1, 2).reshape(nchunk, -1)
    wx = torch.cat([xa, xb], 1).to(torch.bfloat16).contiguous()
    wpx = torch.cat([w1, w3, wx], 1).contiguous().view(-1)          # PX kernel: [chunk][W1 | W3 | WX]
    wx16 = torch.cat([0.5 * xa, xb], 1).to(torch.float16).view(torch.bfloat16)   # DX forms 2 dA
    wdx = torch.cat([w1, w3, wx16], 1).contiguous().view(-1)          # DX role: [chunk][W1 | W3 | WX in f16]
    wx = wx.view(-1)
    # DW role (recomputing, H100 style): per 64-unit slice, W1s rows r = 32 p + 16 ab + hh (hidden 64 s + 16 p + hh), plain k order,
    # [16 k-granules][128 rows][8] then W3s = Ws^T slice [16 k-granules][64 rows][8]
    w1s = torch.stack([(0.5 * wa).view(8, 4, 16, D), wb.view(8, 4, 16, D)], 2).reshape(8, 128, D).view(8, 128, 16, 8).transpose(1, 2)
    w3s = ws.t().reshape(8, 64, 16, 8).transpose(1, 2)
    wdw = torch.cat([w1s.reshape(8, -1), w3s.reshape(8, -1)], 1).to(torch.bfloat16).contiguous().view(-1)
    return dict(wdx=wdx, wdw=wdw, wp=wp, wx=wx, wpx=wpx, gb=fw["gb"], gamma=f(m.ln_in.weight).contiguous(), beta=f(m.ln_in.bias).contiguous(), eps=fw["eps"])


def backward(ext, x, dy, pk, bufs=None, nrep=27, px=False):
    T = x.shape[0]
    dev = x.device
    if bufs is None:
        bufs = {}
    if bufs.get("T") != T:
        e = lambda *s, dt=torch.bfloat16: torch.empty(*s, device=dev, dtype=dt)  # noqa: E731
        bufs.update(T=T, xn=e(T, D), stats=e(T, 2, dt=torch.float32), ab=e(3 * T * H), dx=e(T, D),
                    dgb=e(torch.cuda.get_device_properties(dev).multi_processor_count, 256, dt=torch.float32),
                    part=e(nrep, 3, H, D, dt=torch.float32))
    b = bufs
    args = (x, dy, pk["wp"], pk["wx"], pk["gb"], pk["gamma"], b["xn"], b["stats"], b["ab"], b["dx"], b["dgb"], pk["eps"])
    if px:
        ext.bwd_px(x, dy, pk["wpx"], *args[3:])
    else:
        ext.bwd_p(*args)
        ext.bwd_x(*args)
    ext.bwd_w(x, dy, b["stats"], pk["gamma"], pk["beta"], b["ab"], b["part"])
    gsum = b["dgb"].view(-1, 2, D).sum(0)
    wsum = b["part"].sum(0)
    return b["dx"], gsum[0], gsum[1], wsum[0], wsum[1], wsum[2].t()


def backward_fused(ext, x, dy, pk, bufs=None, ndx=52):
    """The one-launch (H100-layout) backward: prologue LN, then roles DX and DW (see csrc/tr_bwd_fused_sm80.cuh)."""
    T, dev = x.shape[0], x.device
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    nrep = (nsm - ndx) // 8
    if bufs is None:
        bufs = {}
    if bufs.get("Tf") != (T, ndx):
        e = lambda *s, dt=torch.bfloat16: torch.empty(*s, device=dev, dtype=dt)  # noqa: E731
        bufs.update(Tf=(T, ndx), fxn=e(T, D), fst=e(T, 2, dt=torch.float32), bar=torch.zeros(1, dtype=torch.int32, device=dev),
                    fdx=e(T, D), fdgb=e(ndx, 256, dt=torch.float32), fpart=e(nrep, 3, H, D, dt=torch.float32))
    b = bufs
    ext.bwd_fused(x, dy, pk["wdx"], pk["wdw"], pk["gamma"], pk["beta"], b["fxn"], b["fst"], b["bar"], b["fdx"], b["fdgb"], b["fpart"],
                  ndx, pk["eps"])
    g = b["fdgb"].view(-1, 2, D).sum(0)
    w = b["fpart"].sum(0)
    return b["fdx"], g[0], g[1], w[0], w[1], w[2].t()


RING_K = 8
_EMPTY_I64 = {}


def backward_ring(ext, x, dy, pk, bufs=None, ndxp=64, prof=None):
    """One launch, every product once: DXP CTAs -> L2 ring -> W CTAs (csrc/tr_bwd_ring_sm80.cuh)."""
    T, dev = x.shape[0], x.device
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    nrep = (nsm - ndxp) // 4
    if bufs is None:
        bufs = {}
    if bufs.get("Tr") != (T, ndxp):
        e = lambda *s, dt=torch.bfloat16: torch.empty(*s, device=dev, dtype=dt)  # noqa: E731
        bufs.update(Tr=(T, ndxp), rst=e(T, 2, dt=torch.float32), rsc=e(T, dt=torch.float32), rdx=e(T, D), rdgb=e(ndxp, 256, dt=torch.float32),
                    rpart=e(nrep, 3, H, D, dt=torch.float32), ring=e(ndxp * 16 * 3072 * 8), flags=torch.zeros(2 * ndxp * 16, dtype=torch.int32, device=dev))
    b = bufs
    ext.bwd_ring(x, dy, pk["wdx"], pk["gb"], pk["gamma"], pk["beta"], b["rst"], b["rsc"], b["rdx"], b["rdgb"], b["rpart"], b["ring"], b["flags"], ndxp, pk["eps"],
                  prof if prof is not None else _EMPTY_I64.setdefault(dev, torch.empty(0, dtype=torch.int64, device=dev)))
    g = b["rdgb"].view(-1, 2, D).sum(0)
    w = b["rpart"].sum(0)
    return b["rdx"], g[0], g[1], 0.5 * w[0], w[1], w[2].t()


def backward_2k(ext, x, dy, pk, bufs=None, nrep=27, gpx=0):
    """Two launches, every product once: PX (the ring kernel's DXP role, blocks to global memory) then W."""
    T, dev = x.shape[0], x.device
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    if bufs is None:
        bufs = {}
    if bufs.get("T2") != (T, nrep):
        e = lambda *s, dt=torch.bfloat16: torch.empty(*s, device=dev, dtype=dt)  # noqa: E731
        bufs.update(T2=(T, nrep), kst=e(T, 2, dt=torch.float32), krsc=e(T, dt=torch.float32), kab=e(3 * T * H), kdx=e(T, D),
                    kdgb=e(nsm, 256, dt=torch.float32), kpart=e(nrep, 3, H, D, dt=torch.float32))
    b = bufs
    ext.bwd_2k(x, dy, pk["wdx"], pk["gb"], pk["gamma"], pk["beta"], b["kst"], b["krsc"], b["kab"], b["kdx"], b["kdgb"], b["kpart"], gpx, pk["eps"])
    g = b["kdgb"][: min(T // 256, gpx or nsm)].view(-1, 2, D).sum(0)
    w = b["kpart"].sum(0)
    return b["kdx"], g[0], g[1], 0.5 * w[0], w[1], w[2].t()


def backward_seq(ext, x, dy, pk, bufs=None, wr=32, prof=None):
    """One launch, PX then W per CTA with balanced W row ranges (csrc/tr_bwd_ring_sm80.cuh, tr_bwd_seq_kernel)."""
    T, dev = x.shape[0], x.device
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    if bufs is None:
        bufs = {}
    if bufs.get("Ts") != T:
        e = lambda *s, dt=torch.bfloat16: torch.empty(*s, device=dev, dtype=dt)  # noqa: E731
        bufs.update(Ts=T, sst=e(T, 2, dt=torch.float32), srsc=e(T, dt=torch.float32), sab=e(3 * T * H), sdx=e(T, D),
                    sdgb=e(nsm, 256, dt=torch.float32), spart=e(nsm // 4, 3, H, D, dt=torch.float32),
                    sdone=torch.zeros(T // 256, dtype=torch.int32, device=dev))
    b = bufs
    ext.bwd_seq(x, dy, pk["wdx"], pk["gb"], pk["gamma"], pk["beta"], b["sst"], b["srsc"], b["sab"], b["sdx"], b["sdgb"], b["spart"], b["sdone"],
                wr, pk["eps"], prof if prof is not None else _EMPTY_I64.setdefault(dev, torch.empty(0, dtype=torch.int64, device=dev)))
    g = b["sdgb"][: min(T // 256, nsm)].view(-1, 2, D).sum(0)
    w = b["spart"].sum(0)
    return b["sdx"], g[0], g[1], 0.5 * w[0], w[1], w[2].t()


def backward_pwx(ext, x, dy, pk, bufs=None, nrep=13, xn=None, stats=None):
    """Two launches, every product once: PW (a, b, dh per 64-unit slice -> dW; dA | dB blocks out) then X (d_xn, LN backward)."""
    T, dev = x.shape[0], x.device
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    if bufs is None:
        bufs = {}
    if bufs.get("Tq") != (T, nrep):
        e = lambda *s, dt=torch.bfloat16: torch.empty(*s, device=dev, dtype=dt)  # noqa: E731
        bufs.update(Tq=(T, nrep), qst=e(T, 2, dt=torch.float32), qab=e(3 * T * H), qdx=e(T, D), qxn=e(1, D),
                    qdgb=e(nsm, 256, dt=torch.float32), qpart=e(nrep, 3, H, D, dt=torch.float32))
    b = bufs
    # xn given (build -DPW_XN): PW reads it in place of x and skips its LayerNorm; X then needs the given stats
    ext.bwd_pw(x if xn is None else xn, dy, pk["wdw"], pk["gamma"], pk["beta"], b["qst"], b["qab"], b["qpart"], pk["eps"])
    ext.bwd_x(x, dy, pk["wp"], pk["wx"], pk["gb"], pk["gamma"], b["qxn"], b["qst"] if stats is None else stats, b["qab"], b["qdx"], b["qdgb"], pk["eps"])
    g = b["qdgb"][: min(T // 256, nsm)].view(-1, 2, D).sum(0)
    w = b["qpart"].sum(0)
    return b["qdx"], g[0], g[1], w[0], w[1], w[2].t()


def sol_us(M, tc=240e12):
    """16 M D H FLOP (every product of the backward once) over the card's measured tensor ceiling."""
    return 16 * M * D * H / tc * 1e6
