"""sm_100a (B200) kernels of the AF3 windowed atom attention (32 queries x 128 keys per window, per-window pair bias).

    S = q k^T / sqrt(32) + bias[h][w]      query window w = i // 32 sees the keys [32 w - 48, 32 w + 80) (clipped to [0, N))

The attention stage of ``modules/atom_local``'s block, forward and backward; mma.sync (the 32-row windows are too small for tcgen05's
128-row tiles), CUDA only. The cubins are built on first use like ``sm100_atom``'s and launch through the CUDA driver on torch's current stream.

  lbias.cu       bias[h, w, i, j] = LN(z[w, i, j]) . Wb[h] over the trunked pair z [nwin, 32, 128, 16], and its backward.
  lattn_fwd.cu   O and the row LSE.
  lattn_dq.cu    dQ (window-local).
  lattn_dkv.cu   dK, dV and dbias (key-centric: 16 keys see exactly 4 query windows = 128 queries; dbias is summed over the samples).
  lcross.cu      the row kernels of the cross-attention mode (keys / values from a second AdaLN): LN of the conditioning, the K / V AdaLN
                 and its backward, the extra gradient through AdaLN 1.

The fp32 path (fp32 activations; ``KERNELS_TF32``, loaded by ``_load32`` and only when an fp32 call is served) has its own twins:
  lattn_fwd_tf32.cu, lattn_dq_tf32.cu, lattn_dkv_tf32.cu   the three attention kernels on tcgen05 kind::tf32 (fp32 operands, P / dS fp32
                 in place over S in TMEM, MN-major operands in the 32-B-atom swizzle), fp32 O / dQ / dK / dV into column blocks of the
                 caller's [A N, 512] tensors.
  lbias_tf32.cu  the pair bias and its backward on an fp32 pair.
"""

from __future__ import annotations

import functools
import hashlib
import os
import subprocess
from pathlib import Path

import torch

NH, DH, DM, DP = 4, 32, 128, 16
WQ, WK, KOFF = 32, 128, 48
_dir = Path(__file__).parent
SOURCES = ("lattn_fwd", "lattn_dq", "lattn_dkv", "lbias", "lcross")


@functools.lru_cache(maxsize=None)
def cubin(stem: str) -> str:
    """Path of ``<stem>.cu`` built for sm_100a; rebuilt only when a source or a flag changes."""
    from miniworld_engine.kernels.transition.cuda.fused_sm100a import kernel_toolchain

    nvcc, rel, host = kernel_toolchain()
    flags = (*host, "-std=c++17", "-O3", "-arch=sm_100a", "-cubin", "-lineinfo", f"-I{_dir}")
    h = hashlib.sha256(" ".join((nvcc, str(rel), *flags)).encode())
    for f in (_dir / "local.cuh", _dir.parent / "sm100" / "sm100.cuh", _dir / f"{stem}.cu"):
        h.update(f.read_bytes())
    root = Path(os.environ.get("MINIWORLD_ENGINE_JIT_ROOT", Path.home() / ".cache" / "miniworld_engine_jit"))
    out = root / "atom_local_sm100" / f"{stem}_{h.hexdigest()[:16]}.cubin"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        res = subprocess.run([nvcc, *flags, str(_dir / f"{stem}.cu"), "-o", str(tmp)], capture_output=True, text=True,
                             timeout=900, check=False)
        if res.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"nvcc {rel[0]}.{rel[1]} failed on {stem}.cu:\n{res.stderr[-4000:]}")
        os.replace(tmp, out)
    return str(out)


#: (stem, function, dynamic smem bytes) of every kernel
KERNELS = {
    "fwd": ("lattn_fwd", "local_attn_fwd", 203264),
    "dq": ("lattn_dq", "local_attn_dq", 222720),
    "dkv": ("lattn_dkv", "local_attn_dkv", 223744),
    "pb_f": ("lbias", "local_bias_fwd", 0),
    "pb_b": ("lbias", "local_bias_bwd", 0),
    "pb_fin": ("lbias", "local_bias_fin", 0),
    "cond_ln": ("lcross", "local_cond_ln", 0),
    "kv_f": ("lcross", "local_kv_fwd", 0),
    "kv_b": ("lcross", "local_kv_bwd", 0),
    "a1x": ("lcross", "local_adaln1_extra", 0),
    "cln_b": ("lcross", "local_cond_ln_bwd", 0),
}


@functools.lru_cache(maxsize=None)
def _load(name: str, device_index: int):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    stem, func, smem = KERNELS[name]
    with torch.cuda.device(device_index):
        return driver.Kernel(cubin(stem), func, smem)


def _dev(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


@functools.lru_cache(maxsize=8)
def nsm(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def nwindows(n: int) -> int:
    return (n + WQ - 1) // WQ


_TMAPS: dict = {}


def _tmap(t, dims, strides, box, swizzle, dtype="bf16"):
    """A TMA descriptor, cached by everything it encodes (address, dims, strides, box, swizzle, dtype): building one costs ~5-10 us of
    host time per call. The cache does not hold the tensor (a reused address with the same geometry encodes the same descriptor)."""
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    key = (t.data_ptr(), tuple(dims), tuple(strides), tuple(box), swizzle, dtype)
    m = _TMAPS.get(key)
    if m is None:
        if len(_TMAPS) >= 512:
            _TMAPS.clear()
        m = driver.TensorMap(t, dims, strides, box, swizzle=swizzle, dtype=dtype)
        m.keep = None
        _TMAPS[key] = m
    return m


def _rows(t, n):
    """TMA map of a contiguous [A, N, 128] bf16 tensor: boxes of n rows x one head (32 columns), 64-B swizzle."""
    A, N, _ = t.shape
    return _tmap(t, (DM, N, A), (DM * 2, N * DM * 2), (DH, n, 1), 64)


def _vec(t, n):
    """TMA map of a contiguous [A, 4, N] fp32 tensor (LSE, D): boxes of n rows of one head."""
    A, _, N = t.shape
    return _tmap(t, (N, NH, A), (N * 4, NH * N * 4), (n, 1, 1), 0, "f32")


def attn_fwd(q, k, v, bias, kmask=None):
    """q, k, v [A, N, 128] bf16 (contiguous); bias [4, nwin, 32, 128] fp32; kmask [N] bool/uint8 or None -> O [A, N, 128] bf16,
    LSE [A, 4, N] fp32 (log2). tcgen05: a CTA per (128-query chunk, head) and sample range."""
    A, N, _ = q.shape
    nwin = nwindows(N)
    d = _dev(q)
    o = torch.empty_like(q)
    lse = torch.empty(A, NH, N, device=q.device, dtype=torch.float32)
    km = None if kmask is None else kmask.to(torch.uint8).contiguous()
    nq = (N + 127) // 128
    sp = max(1, min(A, nsm(d) // (nq * NH)))
    out = _tmap(o, (DM, N, A), (DM * 2, N * DM * 2), (16, 32, 1), 0)
    _load("fwd", d)((nq * NH * sp, 1, 1), (544, 1, 1), _rows(q, 128), _rows(k, 224), _rows(v, 224), out, bias, km, lse,
                    int(N), int(A), int(nwin), int(sp))
    return o, lse


def attn_bwd(q, k, v, do, bias, lse, dd, dkv, kmask=None):
    """dQ, dK, dV (bf16) into dkv [A N, 512] (blocks 0 / 1 / 2) and dbias [4, nwin, 32, 128] fp32 (summed over the samples).

    ``dd`` is D = rowsum(dO * O) per head [A, 4, N] fp32, ``lse`` the forward's [A, 4, N] (log2 units). q, k, v, do are contiguous
    [A, N, 128]. dQ (query-centric) and dK / dV / dbias (key-centric) run on tcgen05, a CTA per (128-row chunk, head) and sample range,
    through TMA maps over the [A, N, *] operands (zero fill outside [0, N), stores clipped at N)."""
    A, N, _ = q.shape
    nwin = nwindows(N)
    d = _dev(q)
    ldd = dkv.stride(0)
    km = None if kmask is None else kmask.to(torch.uint8).contiguous()
    dbias = torch.zeros(NH, nwin, WQ, WK, device=q.device, dtype=torch.float32)
    out = _tmap(dkv, (dkv.shape[1], N, A), (ldd * 2, N * ldd * 2), (16, 32, 1), 0)
    nq = (N + 127) // 128
    sp = max(1, min(A, nsm(d) // (nq * NH)))                         # split the samples when there are few chunks
    _load("dq", d)((nq * NH * sp, 1, 1), (544, 1, 1), _rows(q, 128), _rows(k, 224), _rows(v, 224), _rows(do, 128), _vec(lse, 128), _vec(dd, 128),
                   out, bias, km, int(N), int(A), int(nwin), int(sp))
    nk = (N + 112 + 127) // 128
    sp = 2 if 2 * nk * NH <= nsm(d) and A > 1 else 1                  # two sample halves add into dbias exactly (0 + a + b)
    _load("dkv", d)((nk * NH * sp, 1, 1), (512, 1, 1), _rows(q, 224), _rows(k, 128), _rows(v, 128), _rows(do, 224), _vec(lse, 224), _vec(dd, 224),
                    out, bias, km, dkv, dbias, int(ldd), int(N), int(A), int(nwin), int(sp))
    return dbias


def _params(wb, gamma):
    """Wb [4, 16] and gamma [16] as the kernels read them: contiguous, bf16 or fp32 (anything else is converted to fp32)."""
    if wb.dtype != gamma.dtype or wb.dtype not in (torch.bfloat16, torch.float32):
        wb, gamma = wb.float(), gamma.float()
    return wb.contiguous(), gamma.contiguous(), int(wb.dtype == torch.bfloat16)


def pair_bias_fwd(z, gamma, wb, eps=1e-5):
    """z [nwin, 32, 128, 16] bf16 -> bias [4, nwin, 32, 128] fp32."""
    nwin = z.shape[0]
    rows = z.numel() // DP
    wb, gamma, bf = _params(wb, gamma)
    out = torch.empty(NH, nwin, WQ, WK, device=z.device, dtype=torch.float32)
    d = _dev(z)
    _load("pb_f", d)((min(4 * nsm(d), (rows + 255) // 256), 1, 1), (256, 1, 1), z.contiguous(), wb, gamma, out, int(rows), float(eps), bf)
    return out


def pair_bias_bwd(z, gamma, wb, dbias, eps=1e-5):
    """dbias [4, nwin, 32, 128] fp32 -> dz (z's layout and dtype), dgamma [16], dWb [4, 16] (fp32)."""
    rows = z.numel() // DP
    wb, gamma, bf = _params(wb, gamma)
    dz = torch.empty_like(z)
    d = _dev(z)
    nb = min(nsm(d), (rows + 255) // 256)
    part = torch.empty(nb, NH * DP, device=z.device, dtype=torch.float32)
    dgamma = torch.empty(DP, device=z.device, dtype=torch.float32)
    dwb = torch.empty(NH, DP, device=z.device, dtype=torch.float32)
    _load("pb_b", d)((nb, 1, 1), (256, 1, 1), z.contiguous(), wb, gamma, dbias.contiguous(), dz, part, int(rows), float(eps), bf)
    _load("pb_fin", d)((1, 1, 1), (64, 1, 1), part, int(nb), wb, gamma, dgamma, dwb, bf)
    return dz, dgamma, dwb


# ------------------------------------------------------------------------------------------------ cross-attention mode (second AdaLN for K / V)
def cond_ln(c, gamma, eps=1e-5):
    """c [M, 128] bf16 -> LN(c) * gamma [M, 128] bf16."""
    M = c.shape[0]
    out = torch.empty_like(c)
    _load("cond_ln", _dev(c))(((M + 7) // 8, 1, 1), (256, 1, 1), c, gamma.float().contiguous(), out, int(M), float(eps))
    return out


def kv_fwd(x1, mkv, bs, eps=1e-5):
    """xkv = LN(x1) * sigmoid(mkv[:, :128] + bs) + mkv[:, 128:]  ([M, 128] bf16)."""
    M = x1.shape[0]
    out = torch.empty_like(x1)
    _load("kv_f", _dev(x1))(((M + 7) // 8, 1, 1), (256, 1, 1), x1, mkv, bs.float().contiguous(), out, int(M), float(eps))
    return out


def kv_bwd(x1, mkv, bs, dxkv, eps=1e-5):
    """dxkv -> dx1 [M, 128] and dmkv [M, 256] = [d pre-sigmoid scale | d shift] (bf16)."""
    M = x1.shape[0]
    dx1 = torch.empty_like(x1)
    dmkv = torch.empty(M, 2 * DM, device=x1.device, dtype=torch.bfloat16)
    _load("kv_b", _dev(x1))(((M + 7) // 8, 1, 1), (256, 1, 1), x1, mkv, bs.float().contiguous(), dxkv, dx1, dmkv, int(M), float(eps))
    return dx1, dmkv


def adaln1_extra(a, mod, dx1, ds, dmod, dbias, eps=1e-5):
    """ds [M, 128] and dmod [M, 768] (blocks 0 and 1), dbias [128] fp32 += the gradient dx1 through AdaLN 1 (in place)."""
    M = a.shape[0]
    d = _dev(a)
    _load("a1x", d)((min(nsm(d) * 4, (M + 7) // 8), 1, 1), (256, 1, 1), a, mod, dx1, ds, dmod, dbias, int(M), float(eps))


def cond_ln_bwd(c, dcn, gamma, dgamma, eps=1e-5):
    """-> dc [M, 128] bf16; dgamma [128] fp32 += sum_rows dcn * LN(c)."""
    M = c.shape[0]
    dc = torch.empty_like(c)
    d = _dev(c)
    _load("cln_b", d)((min(nsm(d) * 4, (M + 7) // 8), 1, 1), (256, 1, 1), c, dcn, gamma.float().contiguous(), dc, dgamma, int(M), float(eps))
    return dc


# ------------------------------------------------------------------------------------------------ fp32 path (TF32 tensor cores)
SOURCES_TF32 = ("lattn_fwd_tf32", "lattn_dq_tf32", "lattn_dkv_tf32", "lbias_tf32")

#: (stem, function, dynamic smem bytes) of every fp32-path kernel (the smem sizes are static_asserted in the sources)
KERNELS_TF32 = {
    "fwd32": ("lattn_fwd_tf32", "local_attn_fwd_tf32", 172288),
    "dq32": ("lattn_dq_tf32", "local_attn_dq_tf32", 227584),
    "dkv32": ("lattn_dkv_tf32", "local_attn_dkv_tf32", 209152),
    "pb_f32": ("lbias_tf32", "local_bias_fwd_f32", 0),
    "pb_b32": ("lbias_tf32", "local_bias_bwd_f32", 0),
    "pb_fin32": ("lbias_tf32", "local_bias_fin_f32", 0),
}


@functools.lru_cache(maxsize=None)
def _load32(name: str, device_index: int):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    stem, func, smem = KERNELS_TF32[name]
    with torch.cuda.device(device_index):
        return driver.Kernel(cubin(stem), func, smem)


def _rows32(t, n, swizzle=128):
    """TMA map of an fp32 [A, N, 128] view (rows ``t.stride(1)`` elements apart, e.g. a column block of [A, N, 512]): boxes of n rows x one
    head (32 fp32 = 128 B). swizzle 128: the K-major operand layout; "128a32": the 128-B swizzle with 32-B atoms, the MN-major one."""
    A, N, _ = t.shape
    assert t.stride(2) == 1 and t.dtype == torch.float32
    return _tmap(t, (DM, N, A), (t.stride(1) * 4, t.stride(0) * 4), (DH, n, 1), swizzle, "f32")


def _out32(t):
    """TMA store map of a contiguous fp32 [A N, C] tensor viewed [A, N, C]: boxes of 32 rows x 16 columns (64 B, 64-B swizzle)."""
    A, N, C = t.shape
    return _tmap(t, (C, N, A), (C * 4, N * C * 4), (16, 32, 1), 64, "f32")


def attn_fwd32(q, k, v, bias, kmask=None):
    """The fp32 path: q, k, v fp32 [A, N, 128] views (unit column stride, any row stride); bias [4, nwin, 32, 128] fp32; kmask [N] or None
    -> O [A, N, 128] fp32, LSE [A, 4, N] fp32 (log2). tcgen05 kind::tf32: a CTA per (128-query chunk, head) and sample range."""
    A, N, _ = q.shape
    nwin = nwindows(N)
    d = _dev(q)
    o = torch.empty(A, N, DM, device=q.device, dtype=torch.float32)
    lse = torch.empty(A, NH, N, device=q.device, dtype=torch.float32)
    km = None if kmask is None else kmask.to(torch.uint8).contiguous()
    nq = (N + 127) // 128
    sp = max(1, min(A, nsm(d) // (nq * NH)))
    _load32("fwd32", d)((nq * NH * sp, 1, 1), (544, 1, 1), _rows32(q, 128), _rows32(k, 224), _rows32(v, 224, "128a32"), _out32(o),
                        bias, km, lse, int(N), int(A), int(nwin), int(sp))
    return o, lse


def attn_bwd32(q, k, v, do, bias, lse, dd, dp, cols, kmask=None):
    """The fp32 path's backward: dQ, dK, dV (fp32) into the column blocks ``cols`` = (q, k, v) of dp [A N, C] and dbias [4, nwin, 32, 128]
    fp32 (summed over the samples). q, k, v fp32 [A, N, 128] views, do [A, N, 128] fp32, lse / dd [A, 4, N] fp32 (log2 units / D)."""
    A, N, _ = q.shape
    nwin = nwindows(N)
    d = _dev(q)
    ldd = dp.stride(0)
    km = None if kmask is None else kmask.to(torch.uint8).contiguous()
    dbias = torch.zeros(NH, nwin, WQ, WK, device=q.device, dtype=torch.float32)
    out = _out32(dp.view(A, N, dp.shape[1]))
    qcol, kcol, vcol = cols
    nq = (N + 127) // 128
    sp = max(1, min(A, nsm(d) // (nq * NH)))                         # split the samples when there are few chunks
    _load32("dq32", d)((nq * NH * sp, 1, 1), (544, 1, 1), _rows32(q, 128), _rows32(k, 224), _rows32(v, 224), _rows32(k, 224, "128a32"),
                       _rows32(do, 128), _vec(lse, 128), _vec(dd, 128), out, bias, km, int(qcol), int(N), int(A), int(nwin), int(sp))
    nk = (N + 112 + 127) // 128
    sp = 2 if 2 * nk * NH <= nsm(d) and A > 1 else 1                  # two sample halves add into dbias exactly (0 + a + b)
    _load32("dkv32", d)((nk * NH * sp, 1, 1), (544, 1, 1), _rows32(q, 224), _rows32(q, 224, "128a32"), _rows32(k, 128), _rows32(v, 128),
                        _rows32(do, 224), _rows32(do, 224, "128a32"), _vec(lse, 224), _vec(dd, 224), out, bias, km, dp, dbias, int(ldd),
                        int(kcol), int(vcol), int(N), int(A), int(nwin), int(sp))
    return dbias


def pair_bias_fwd32(z, gamma, wb, eps=1e-5):
    """z [nwin, 32, 128, 16] fp32 -> bias [4, nwin, 32, 128] fp32."""
    z = z.contiguous()
    nwin = z.shape[0]
    rows = z.numel() // DP
    wb, gamma, bf = _params(wb, gamma)
    out = torch.empty(NH, nwin, WQ, WK, device=z.device, dtype=torch.float32)
    d = _dev(z)
    _load32("pb_f32", d)((min(4 * nsm(d), (rows + 255) // 256), 1, 1), (256, 1, 1), z, wb, gamma, out, int(rows), float(eps), bf)
    return out


def pair_bias_bwd32(z, gamma, wb, dbias, eps=1e-5):
    """dbias [4, nwin, 32, 128] fp32 -> dz [nwin, 32, 128, 16] fp32, dgamma [16], dWb [4, 16] (fp32)."""
    z = z.contiguous()
    rows = z.numel() // DP
    wb, gamma, bf = _params(wb, gamma)
    dz = torch.empty_like(z)
    d = _dev(z)
    nb = min(nsm(d), (rows + 255) // 256)
    part = torch.empty(nb, NH * DP, device=z.device, dtype=torch.float32)
    dgamma = torch.empty(DP, device=z.device, dtype=torch.float32)
    dwb = torch.empty(NH, DP, device=z.device, dtype=torch.float32)
    _load32("pb_b32", d)((nb, 1, 1), (256, 1, 1), z, wb, gamma, dbias.contiguous(), dz, part, int(rows), float(eps), bf)
    _load32("pb_fin32", d)((1, 1, 1), (64, 1, 1), part, int(nb), wb, gamma, dgamma, dwb, bf)
    return dz, dgamma, dwb
