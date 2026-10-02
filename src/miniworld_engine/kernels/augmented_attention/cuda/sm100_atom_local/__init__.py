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
    for f in (_dir / "local.cuh", _dir / f"{stem}.cu"):
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
    "fwd": ("lattn_fwd", "local_attn_fwd", 91648),
    "dq": ("lattn_dq", "local_attn_dq", 99840),
    "dkv": ("lattn_dkv", "local_attn_dkv", 101376),
    "pb_f": ("lbias", "local_bias_fwd", 0),
    "pb_b": ("lbias", "local_bias_bwd", 0),
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


def attn_fwd(q, k, v, bias, kmask=None):
    """q, k, v [A, N, 128] bf16; bias [4, nwin, 32, 128] fp32; kmask [N] bool/uint8 or None -> O [A, N, 128] bf16, LSE [A, 4, N] fp32 (log2)."""
    A, N, _ = q.shape
    nwin = nwindows(N)
    d = _dev(q)
    o = torch.empty_like(q)
    lse = torch.empty(A, NH, N, device=q.device, dtype=torch.float32)
    npair = (A + 1) // 2
    ctas = nwin * NH
    psplit = max(1, min(npair, (2 * nsm(d) + ctas - 1) // ctas))      # about two CTAs per SM
    km = None if kmask is None else kmask.to(torch.uint8).contiguous()
    _load("fwd", d)((nwin, NH, psplit), (128, 1, 1), q, k, v, bias, km, o, lse, int(N), int(A), int(nwin), int(psplit))
    return o, lse


def attn_bwd(q, k, v, do, bias, lse, dd, dkv, kmask=None):
    """dQ, dK, dV (bf16) into dkv [A N, 512] (blocks 0 / 1 / 2) and dbias [4, nwin, 32, 128] fp32 (summed over the samples).

    ``dd`` is D = rowsum(dO * O) per head [A, 4, N] fp32, ``lse`` the forward's [A, 4, N] (log2 units)."""
    A, N, _ = q.shape
    nwin = nwindows(N)
    d = _dev(q)
    ldd = dkv.stride(0)
    km = None if kmask is None else kmask.to(torch.uint8).contiguous()
    dbias = torch.zeros(NH, nwin, WQ, WK, device=q.device, dtype=torch.float32)
    npair = (A + 1) // 2
    ctas = nwin * NH
    psplit = max(1, min(npair, (2 * nsm(d) + ctas - 1) // ctas))
    _load("dq", d)((nwin, NH, psplit), (128, 1, 1), q, k, v, do, bias, km, lse, dd, dkv, int(ldd), int(N), int(A), int(nwin), int(psplit))
    units = (N + 15) // 16
    _load("dkv", d)(((units + 3) // 4, NH, 1), (128, 1, 1), q, k, v, do, bias, km, lse, dd, dkv, dbias, int(ldd), int(N), int(A), int(nwin))
    return dbias


def pair_bias_fwd(z, gamma, wb, eps=1e-5):
    """z [nwin, 32, 128, 16] bf16 -> bias [4, nwin, 32, 128] fp32."""
    nwin = z.shape[0]
    rows = z.numel() // DP
    w = (wb.float() * gamma.float()[None]).contiguous()
    out = torch.empty(NH, nwin, WQ, WK, device=z.device, dtype=torch.float32)
    d = _dev(z)
    _load("pb_f", d)((min(4 * nsm(d), (rows + 255) // 256), 1, 1), (256, 1, 1), z.contiguous(), w, out, int(rows), float(eps))
    return out


def pair_bias_bwd(z, gamma, wb, dbias, eps=1e-5):
    """dbias [4, nwin, 32, 128] fp32 -> dz (z's layout and dtype), dgamma [16], dWb [4, 16] (fp32)."""
    rows = z.numel() // DP
    w = (wb.float() * gamma.float()[None]).contiguous()
    dz = torch.empty_like(z)
    d = _dev(z)
    nb = min(4 * nsm(d), (rows + 255) // 256)
    part = torch.empty(nb, NH * DP, device=z.device, dtype=torch.float32)
    _load("pb_b", d)((nb, 1, 1), (256, 1, 1), z.contiguous(), w, dbias.contiguous(), dz, part, int(rows), float(eps))
    G = part.sum(0).reshape(NH, DP)
    return dz, (G * wb.float()).sum(0), G * gamma.float()[None]


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
