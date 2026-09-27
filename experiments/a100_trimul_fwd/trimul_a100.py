"""A100 (sm_80) TriMul forward: K1 (CUDA) -> strided-batched contraction (cuBLAS) -> K3 (CUDA).

    ext = build()
    pk = pack(module)                       # TriangleMultiplication (one direction) or BidirectionalTriangleMultiplication
    out = forward(ext, z, mask, pk)         # z [1, L, L, 128] bf16, mask [1, L] bool -> z + trimul(z)
"""
import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_EXT = {}
_NOPROF = None


def build(verbose=False, extra=()):
    import hashlib
    src_hash = hashlib.sha1(b"".join(f.read_bytes() for f in sorted((HERE / "csrc").glob("*.cu*")))).hexdigest()[:12]
    key = tuple(extra)
    if key in _EXT:
        return _EXT[key]
    from torch.utils.cpp_extension import load
    name = "a100_trimul_fwd" + ("_" + str(abs(hash(key)) % 10 ** 8) if key else "")
    bdir = Path(os.environ.get("A100_TRIMUL_BUILD", Path.home() / ".cache/miniworld-a100/ext")) / name
    bdir.mkdir(parents=True, exist_ok=True)
    import time
    lk = bdir / "lock"                         # a build killed mid-way leaves torch's baton; any live build refreshes it within minutes
    if lk.exists() and time.time() - lk.stat().st_mtime > 1200:
        lk.unlink(missing_ok=True)
    _EXT[key] = load(name=name, sources=[str(HERE / "csrc/ops.cu")], build_directory=str(bdir), verbose=verbose,
                     extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80", "-lineinfo", "-Xptxas=-v", f"-DA100_SRC_HASH=0x{src_hash}", *extra],
                     extra_cflags=["-O3"])
    return _EXT[key]


ZST = os.environ.get("TRIMUL_ZST", "1") == "1"


def num_sms(dev):
    return torch.cuda.get_device_properties(dev).multi_processor_count


@torch.no_grad()
def pack(m):
    """Weights of a TriMul module in the kernels' layouts (inference: packed once)."""
    bidir = hasattr(m, "d_hidden") and m.to_left.weight.shape[0] == 2 * m.d_hidden and m.__class__.__name__.startswith("Bidirectional")
    ch = m.to_left.weight.shape[0]
    f = lambda t: t.detach().float()  # noqa: E731
    # K1: blocks of 64 rows = 4 warps x (8 gate rows | 8 proj rows) of output channels oc = 32 step + 8 nw + c8
    wg_all = torch.cat([f(m.to_left_gate.weight), f(m.to_right_gate.weight)], 0)     # [2 CH, 128]
    wp_all = torch.cat([f(m.to_left.weight), f(m.to_right.weight)], 0)
    nstep = ch // 16
    dev = wg_all.device
    step = torch.arange(nstep, device=dev).view(-1, 1, 1)
    nw = torch.arange(4, device=dev).view(1, -1, 1)
    n = torch.arange(16, device=dev).view(1, 1, -1)
    oc = (32 * step + 8 * nw + (n % 8)).reshape(-1)
    is_gate = (n < 8).expand(nstep, 4, 16).reshape(-1)
    w1 = (0.5 * torch.where(is_gate[:, None], wg_all[oc], wp_all[oc])).to(torch.bfloat16)   # sigmoid(g) p = p' (1 + tanh g'), x' = x / 2
    w1 = w1.view(nstep, 64, 16, 8).transpose(1, 2).contiguous()                              # [block][16 B k-granule][64 rows][8]: smem layout
    # K3: LayerNorm affine folded into the weights (W' = bf16(W diag gamma), s = sum_k W', b = W beta)
    wo = 0.5 * f(m.to_out.weight) * f(m.ln_out.weight)[None, :]           # 0.5: sigmoid(g) p = p' (1 + tanh g'), exact in bf16
    wg = 0.5 * f(m.to_gate.weight) * f(m.ln_pair.weight)[None, :]
    wo16, wg16 = wo.to(torch.bfloat16).contiguous(), wg.to(torch.bfloat16).contiguous()
    return dict(bidir=bidir, outgoing=getattr(m, "outgoing", True), ch=ch, w1=w1,
                g_in=f(m.ln_pair.weight).contiguous(), b_in=f(m.ln_pair.bias).contiguous(),
                wo=wo16, wg=wg16, so=wo16.float().sum(1).contiguous(), bo=(0.5 * f(m.to_out.weight) @ f(m.ln_out.bias)).contiguous(),
                sg=wg16.float().sum(1).contiguous(), bg=(0.5 * f(m.to_gate.weight) @ f(m.ln_pair.bias)).contiguous(),
                eps_in=float(m.ln_pair.eps), eps_out=float(m.ln_out.eps))


def contract(a, b, x, pk, ext=None):
    ch = pk["ch"]
    mode = os.environ.get("TRIMUL_CONTRACT", "auto")    # auto: the custom kernel where it measured ahead (bidirectional, L <= 512: one launch)
    use = mode == "custom" or (mode == "auto" and pk["bidir"] and a.shape[1] <= 512)
    if ext is not None and use and a.shape[1] % 128 == 0:
        h = ch // 2 if pk["bidir"] else (ch if pk["outgoing"] else 0)      # channels [0, h) outgoing (NT), the rest incoming (TN)
        ext.contract(a, b, x, h, 0)
        return
    if pk["bidir"]:
        h = ch // 2
        torch.bmm(a[:h], b[:h].transpose(1, 2), out=x[:h])      # outgoing: sum_k a[i,k] b[j,k]
        torch.bmm(a[h:].transpose(1, 2), b[h:], out=x[h:])      # incoming: sum_k a[k,i] b[k,j]
    elif pk["outgoing"]:
        torch.bmm(a, b.transpose(1, 2), out=x)
    else:
        torch.bmm(a.transpose(1, 2), b, out=x)


def forward(ext, z, mask, pk, bufs=None):
    global _NOPROF
    if _NOPROF is None:
        _NOPROF = torch.empty(0, device=z.device)
    B, L, L2, C = z.shape
    assert B == 1 and L == L2 and C == 128 and z.dtype == torch.bfloat16 and L % 8 == 0
    ch, T = pk["ch"], L * L
    zf = z.reshape(T, C)
    if bufs is None:
        bufs = {}
    key = (L, ch)
    if bufs.get("key") != key:
        bufs.update(key=key, ab=torch.empty(2 * ch, T, device=z.device, dtype=torch.bfloat16),
                    x=torch.empty(ch, L, L, device=z.device, dtype=torch.bfloat16),
                    zst=torch.empty(T, 2, device=z.device, dtype=torch.float32))
    ab, x = bufs["ab"], bufs["x"]
    m = mask.reshape(L).to(torch.uint8) if mask is not None else torch.empty(0, dtype=torch.uint8, device=z.device)
    if ZST:                                      # K1 saves the LN_in statistics; K3 skips its z statistics and their exchange
        ext.k1z(zf, m, pk["w1"], pk["g_in"], pk["b_in"], ab, L, pk["eps_in"], bufs["zst"])
    else:
        ext.k1(zf, m, pk["w1"], pk["g_in"], pk["b_in"], ab, L, pk["eps_in"], 0, _NOPROF)
    contract(ab[:ch].view(ch, L, L), ab[ch:].view(ch, L, L), x, pk, ext)
    out = torch.empty_like(zf)
    if ZST:
        ext.k3z(x.view(ch, T), zf, pk["wo"], pk["wg"], pk["so"], pk["bo"], pk["sg"], pk["bg"], out, pk["eps_out"], bufs["zst"])
    else:
        ext.k3(x.view(ch, T), zf, pk["wo"], pk["wg"], pk["so"], pk["bo"], pk["sg"], pk["bg"], out, pk["eps_out"], 0, _NOPROF, _NOPROF, L)
    return out.view(1, L, L, C)


def sol_us(L, ch, bw=1.602e12, tc=240e12):
    """Composite floor of the three-kernel decomposition (per kernel max(essential bytes / BW, FLOP / tensor peak)), microseconds."""
    T = L * L
    k1 = max(T * (256 + 4 * ch) / bw, T * 2 * 128 * 4 * ch / tc)
    ct = max(T * 6 * ch / bw, T * 2 * ch * L / tc)
    k3 = max(T * (2 * ch + 512) / bw, T * 2 * (ch * 128 + 128 * 128) / tc)
    return dict(k1=k1 * 1e6, contraction=ct * 1e6, k3=k3 * 1e6, total=(k1 + ct + k3) * 1e6)


def sol_train_us(L, ch, bw=1.602e12, tc=240e12):
    """Training (fwd + bwd) composite floor, same rule as sol_us per kernel of the decomposition (ch = plane channels: 256 bidir, 128 single):
    fwd K1 / contraction / K3; bwd B1 (reads X, z, dy; writes dX, A_o, A_g, d_g, x_n, stats), weight-gradient GEMMs (A_o^T X^T, A_g^T z),
    contraction backward (dA, dB of every channel), B7src (reads x_n, dA|dB; writes dgp; g/p recompute + dW), B8 (reads dgp, z, dy; writes dz)."""
    T = L * L
    f = sol_us(L, ch, bw, tc)
    k = lambda by, fl: max(T * by / bw, T * fl / tc) * 1e6  # noqa: E731
    b1 = k(2 * ch + 256 + 256 + 2 * ch + 4 * 256 + 16, 2 * ch * 128 + 2 * 128 * 128 + 2 * 128 * ch)
    wg = k(256 + 2 * ch, 2 * 128 * ch) + k(256 + 256, 2 * 128 * 128)
    cb = k(2 * 6 * ch, 2 * 2 * ch * L)
    b7 = k(256 + 4 * ch + 8 * ch, 2 * 2 * 4 * ch * 128)
    b8 = k(2 * (4 * ch + 128) + 256 + 256 + 16 + 256, 2 * (4 * ch + 128) * 128)
    d = dict(fwd=f["total"], b1=b1, wgrad=wg, contraction_bwd=cb, b7src=b7, b8=b8)
    d["total"] = sum(d.values())
    return d
