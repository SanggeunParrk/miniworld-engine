"""The atom DiT's pair-bias producer: bias[h, i, j] = LayerNorm(z[i, j, :]) . Wb[h, :]   (no LN offset, eps 1e-5)
as one memory-bound pass that writes the bias in both layouts the attention kernels read: head-major [H, N(query), N(key)]
and transposed [H, N(key), N(query)]; and its backward, dz / dgamma / dWb from dbias [H, N, N] (fp32, summed over samples).

The engine runs this as LayerNorm over N^2 rows of 16 channels plus a 16 -> 4 GEMM (2.0 ms at N = 3072, inference) and 7.9 ms in
training. Here gamma is folded into the projection, W' = Wb * gamma, and bias = rstd * (z . W' - mean * sum_c W')."""
import torch
import triton
import triton.language as tl

C, H = 16, 4


@triton.jit
def _pb_fwd(z, w, bo, bt, N, eps, C: tl.constexpr, H: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr, TRANS: tl.constexpr):
    i0, j0 = tl.program_id(0) * BI, tl.program_id(1) * BJ
    ri, rj, rc = i0 + tl.arange(0, BI), j0 + tl.arange(0, BJ), tl.arange(0, C)
    x = tl.load(z + (ri[:, None, None] * N + rj[None, :, None]) * C + rc[None, None, :]).to(tl.float32)   # [BI, BJ, C]
    mean = tl.sum(x, 2) / C
    xc = x - mean[:, :, None]
    rstd = tl.rsqrt(tl.sum(xc * xc, 2) / C + eps)
    for h in tl.static_range(H):
        wh = tl.load(w + h * C + rc).to(tl.float32)                                                       # gamma * Wb[h]
        y = tl.sum(xc * wh[None, None, :], 2) * rstd                                                      # [BI, BJ]
        yb = y.to(tl.bfloat16)
        tl.store(bo + (h * N + ri[:, None]) * N + rj[None, :], yb)
        if TRANS:
            tl.store(bt + (h * N + rj[:, None]) * N + ri[None, :], tl.trans(yb))


@triton.jit
def _pb_bwd(z, w, db, dz, pw, N, eps, C: tl.constexpr, H: tl.constexpr, BI: tl.constexpr, BJ: tl.constexpr):
    i0, j0 = tl.program_id(0) * BI, tl.program_id(1) * BJ
    ri, rj, rc = i0 + tl.arange(0, BI), j0 + tl.arange(0, BJ), tl.arange(0, C)
    x = tl.load(z + (ri[:, None, None] * N + rj[None, :, None]) * C + rc[None, None, :]).to(tl.float32)
    mean = tl.sum(x, 2) / C
    xc = x - mean[:, :, None]
    rstd = tl.rsqrt(tl.sum(xc * xc, 2) / C + eps)
    xh = xc * rstd[:, :, None]                                                                            # x-hat
    dxh = tl.zeros((BI, BJ, C), dtype=tl.float32)
    pid = tl.program_id(0) * tl.num_programs(1) + tl.program_id(1)
    for h in tl.static_range(H):
        g = tl.load(db + (h * N + ri[:, None]) * N + rj[None, :])                                         # [BI, BJ] fp32
        wh = tl.load(w + h * C + rc).to(tl.float32)
        dxh += g[:, :, None] * wh[None, None, :]
        tl.store(pw + (pid * H + h) * C + rc, tl.sum(tl.sum(g[:, :, None] * xh, 0), 0))                   # partial d(W') [H, C]
    m1 = tl.sum(dxh, 2) / C
    m2 = tl.sum(dxh * xh, 2) / C
    d = (dxh - m1[:, :, None] - xh * m2[:, :, None]) * rstd[:, :, None]
    tl.store(dz + (ri[:, None, None] * N + rj[None, :, None]) * C + rc[None, None, :], d.to(dz.dtype.element_ty))


TILE_F = (16, 64, 4)          # (BI, BJ, num_warps), from sweep_pair_bias.py
TILE_B = (16, 64, 4)


def _tiles(N, t):
    assert N % t[0] == 0 and N % t[1] == 0
    return t


def pair_bias_fwd(z, gamma, wb, eps=1e-5, trans=True):
    """z [N, N, 16] (or [1, N, N, 16]) bf16; gamma [16]; wb [4, 16] -> bias [4, N, N] bf16 (and bias^T [4, N, N])."""
    z = z.reshape(z.shape[-3], z.shape[-2], C)
    N = z.shape[0]
    w = (wb.float() * gamma.float()[None]).contiguous()
    bo = torch.empty(H, N, N, device=z.device, dtype=torch.bfloat16)
    bt = torch.empty_like(bo) if trans else bo
    BI, BJ, nw = _tiles(N, TILE_F)
    _pb_fwd[(N // BI, N // BJ)](z, w, bo, bt, N, eps, C=C, H=H, BI=BI, BJ=BJ, TRANS=trans, num_warps=nw)
    return bo, (bt if trans else None)


def pair_bias_bwd(z, gamma, wb, dbias, eps=1e-5):
    """dbias [4, N, N] fp32 -> dz [N, N, 16] (z's dtype), dgamma [16], dwb [4, 16] (fp32)."""
    z = z.reshape(z.shape[-3], z.shape[-2], C)
    N = z.shape[0]
    w = (wb.float() * gamma.float()[None]).contiguous()
    dz = torch.empty_like(z)
    BI, BJ, nw = _tiles(N, TILE_B)
    nprog = (N // BI) * (N // BJ)
    pw = torch.empty(nprog, H, C, device=z.device, dtype=torch.float32)
    _pb_bwd[(N // BI, N // BJ)](z, w, dbias.contiguous(), dz, pw, N, eps, C=C, H=H, BI=BI, BJ=BJ, num_warps=nw)
    dw = pw.sum(0)                                                                                        # d(W') = d(gamma * Wb)
    return dz, (dw * wb.float()).sum(0), dw * gamma.float()[None]


_CU = {}


def _cu():
    if not _CU:
        import drv
        from pathlib import Path
        cub = str(Path(__file__).resolve().parent / "build" / "pair_bias.cubin")
        _CU["f"] = drv.Kernel(cub, "pair_bias_fwd", 0)
        _CU["b"] = drv.Kernel(cub, "pair_bias_bwd", 0)
    return _CU


def pair_bias_fwd_cu(z, gamma, wb, eps=1e-5, trans=True):
    """CUDA version of pair_bias_fwd (build/pair_bias.cubin)."""
    z = z.reshape(z.shape[-3], z.shape[-2], C)
    N = z.shape[0]
    assert N % 64 == 0
    w = (wb.float() * gamma.float()[None]).contiguous()
    bo = torch.empty(H, N, N, device=z.device, dtype=torch.bfloat16)
    bt = torch.empty_like(bo) if trans else None
    _cu()["f"]((N // 64, N // 32, 1), (256, 1, 1), z, w, bo, bt, int(N), float(eps))
    return bo, bt


def pair_bias_bwd_cu(z, gamma, wb, dbias, eps=1e-5):
    """CUDA version of pair_bias_bwd."""
    z = z.reshape(z.shape[-3], z.shape[-2], C)
    N = z.shape[0]
    w = (wb.float() * gamma.float()[None]).contiguous()
    dz = torch.empty_like(z)
    nb = (N // 64) * (N // 32)
    pw = torch.empty(nb, H, C, device=z.device, dtype=torch.float32)
    _cu()["b"]((N // 64, N // 32, 1), (256, 1, 1), z, w, dbias.contiguous(), dz, pw, int(N), float(eps))
    dw = pw.sum(0)
    return dz, (dw * wb.float()).sum(0), dw * gamma.float()[None]


def reference(z, gamma, wb, eps=1e-5):
    zf = z.reshape(z.shape[-3], z.shape[-2], C).double()
    y = torch.nn.functional.layer_norm(zf, (C,), gamma.double(), None, eps) @ wb.double().t()             # [N, N, H]
    return y.permute(2, 0, 1)
