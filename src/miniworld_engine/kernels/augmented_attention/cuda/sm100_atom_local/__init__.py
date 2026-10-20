"""sm_100a (B200) kernels of the AF3 windowed atom attention (32 queries x 128 keys per window, per-window pair bias).

    S = q k^T / sqrt(32) + bias[h][w]      query window w = i // 32 sees the keys [32 w - 48, 32 w + 80) (clipped to [0, N))

The attention stage of ``modules/atom_local``'s block, forward and backward; mma.sync (the 32-row windows are too small for tcgen05's
128-row tiles), CUDA only. The cubins are built on first use like ``sm100_atom``'s and launch through the CUDA driver on torch's current stream.

  lbias.cu       bias[h, w, i, j] = LN(z[w, i, j]) . Wb[h] over the trunked pair z [nwin, 32, 128, 16], and its backward.
  lattn_fwd.cu   O and the row LSE.
  lattn_dq.cu    dQ (window-local).
  lattn_dkv.cu   dK, dV and dbias (key-centric: 16 keys see exactly 4 query windows = 128 queries; dbias is summed over the samples).
  lcross.cu      the row kernels of the cross-attention mode (keys / values from a second AdaLN): LN of the conditioning, the K / V AdaLN
                 (the K / V AdaLN and its backward are inside ``sm100_atom``'s pre_fwd / pre_bwd, ``-DCROSS=1``).

sm_90a (H100, ``-DATOM_HOPPER=1``, ``sm100_atom._hopper``): the same sources carry a Hopper half (wgmma attention kernels ``*_sm90``, and in
lcross.cu the K / V AdaLN row kernels ``kv_fwd`` / ``kv_bwd`` / ``adaln1_extra``, deterministic ``cond_ln_bwd`` and small gradient-pack
kernels); the wrappers pick it per device.

The fp32 path (fp32 activations; ``KERNELS_TF32``, loaded by ``_load32`` and only when an fp32 call is served) has its own twins:
  lattn_fwd_tf32.cu, lattn_dq_tf32.cu, lattn_dkv_tf32.cu   the three attention kernels on tcgen05 kind::tf32 (fp32 operands, P / dS fp32
                 in place over S in TMEM, MN-major operands in the 32-B-atom swizzle), fp32 O / dQ / dK / dV into column blocks of the
                 caller's [A N, 512] tensors.
  lbias_tf32.cu  the pair bias and its backward on an fp32 pair.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from miniworld_engine import settings

NH, DH, DM, DP = 4, 32, 128, 16
WQ, WK, KOFF = 32, 128, 48
_dir = Path(__file__).parent
SOURCES = ("lattn_fwd", "lattn_dq", "lattn_dkv", "lbias", "lcross")


@functools.lru_cache(maxsize=None)
def cubin(stem: str, hopper: bool = False) -> str:
    """Path of ``<stem>.cu`` built for sm_100a; rebuilt only when a source or a flag changes."""
    from miniworld_engine.kernels import _nvcc

    flags = (
        "-std=c++17",
        "-O3",
        "-arch=sm_90a" if hopper else "-arch=sm_100a",
        *(("-DATOM_HOPPER=1",) if hopper else ()),
        "-cubin",
        "-lineinfo",
        f"-I{_dir}",
    )
    family = "atom_local_sm90" if hopper else "atom_local_sm100"
    return str(_nvcc.cubin(f"{family}/{stem}", flags, source=_dir / f"{stem}.cu",
                           toolchain="torch" if hopper else "sm100"))


#: (stem, function, dynamic smem bytes) of every kernel
KERNELS = {
    "fwd": ("lattn_fwd", "local_attn_fwd", 203264),
    "dq": ("lattn_dq", "local_attn_dq", 222720),
    "dkv": ("lattn_dkv", "local_attn_dkv", 223744),
    "pb_f": ("lbias", "local_bias_fwd", 0),
    "pb_b": ("lbias", "local_bias_bwd", 0),
    "pb_fin": ("lbias", "local_bias_fin", 0),
    "cond_ln": ("lcross", "local_cond_ln", 0),
    "cln_b": ("lcross", "local_cond_ln_bwd", 0),
}


#: sm_90a-only kernels of ``lcross.cu`` (``ATOM_HOPPER``), by their ``_load`` name: the K / V AdaLN as row kernels ("kv_f", "kv_b", "a1x") and
#: the sm_90a variant of the conditioning LayerNorm backward ("cln_b": dgamma as per-block partials, no atomics). Not in ``KERNELS``, which
#: lists what sm_100a loads.
_HOPPER_CROSS = {
    "kv_f": "local_kv_fwd_sm90",
    "kv_b": "local_kv_bwd_sm90",
    "a1x": "local_adaln1_extra_sm90",
    "cln_b": "local_cond_ln_bwd_sm90",
}


#: mma.sync kernels of ``lattn90.cu`` (sm_90a): (function, dynamic shared memory)
_MMA_KERNELS = {
    "h_prep": ("h_bias_prep", 0),
    "h_fwd": ("h_attn_fwd", 64 * 136 * 4 + 2 * (64 + 160 + 160) * 40 * 2),
    "h_dq": ("h_attn_dq", 64 * 136 * 4 + (64 * 2 + 160 * 2) * 40 * 2 + 2 * 64 * 4),
    "h_dkv": (
        "h_attn_dkv",
        4 * 16 * 132 * 4 + (192 * 2 + 64 * 2) * 40 * 2 + 2 * 192 * 4,
    ),
    "h_pb": ("h_pair_bias", 0),
    "h_pbb": ("h_pair_bias_bwd", 0),
}
#: TF32 twins of the mma.sync kernels (``lattn90_tf32.cu``)
_MMA_KERNELS32 = {
    "h_fwd32": ("h_attn_fwd_tf32", 90112),
    "h_dq32": (
        "h_attn_dq_tf32",
        64 * 136 * 4 + (64 * 2 + 160 * 2) * 36 * 4 + 2 * 64 * 4,
    ),
    "h_dkv32": (
        "h_attn_dkv_tf32",
        4 * 16 * 132 * 4 + (192 * 2 + 64 * 2) * 36 * 4 + 2 * 192 * 4,
    ),
}
#: Hopper attention through the mma.sync kernels (False: the wgmma ``*_sm90`` kernels of lattn_{fwd,dq,dkv}.cu).
HOPPER_MMA_ATTN = True
_SAMPLE_CHUNK = 16  # samples per CTA (the bias tile and its prologue amortised; measured A48 bwd 551 -> 476 us against 8)


def _chunk(A: int, tiles: int = 0) -> int:
    """Samples per CTA. 16 amortises the bias tile (measured A48 backward 551 -> 476 us against 8); with few (key block, head) tiles the
    chunk shrinks until the grid has about one CTA per SM (the per-chunk dbias partials stay small: few tiles means few windows)."""
    want = _SAMPLE_CHUNK
    if tiles:
        while want > 1 and tiles * -(-A // want) < 132:
            want = (want + 1) // 2
            if want == 1:
                break
    return -(-A // -(-A // want))  # evenly split into ceil(A / want) chunks


def _mma_bias(bias, km, n, d):
    out = torch.empty_like(bias)
    _load("h_prep", d)(
        (min(8 * nsm(d), (bias.numel() + 255) // 256), 1, 1),
        (256, 1, 1),
        bias,
        km,
        out,
        int(n),
    )
    return out


@functools.lru_cache(maxsize=None)
def _load(name: str, device_index: int):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    from miniworld_engine.kernels.augmented_attention.cuda.sm100_atom import _hopper

    hopper = _hopper(device_index)
    if hopper and name in _MMA_KERNELS32:
        stem, func, smem = "lattn90_tf32", *_MMA_KERNELS32[name]
    elif hopper and name in _MMA_KERNELS:
        stem, func, smem = "lattn90", *_MMA_KERNELS[name]
    elif hopper and name == "fwd":
        stem, func, smem = "lattn_fwd", "local_attn_fwd_sm90", 57376
    elif hopper and name in ("dq", "dkv"):
        stem, func, smem = f"lattn_{name}", f"local_attn_{name}_sm90", 65568
    elif hopper and name == "db_samples":
        stem, func, smem = "lattn_dkv", "local_bias_samples_sm90", 0
    elif hopper and name in (
        "kv_bias_part",
        "reduce_columns",
        "zero_kv",
        "add_cond",
        "cross_grad_pack",
        "kv_pack",
    ):
        stem, func, smem = "lcross", f"local_{name}_sm90", 0
    elif (
        hopper and name in _HOPPER_CROSS
    ):  # the K / V AdaLN row kernels (sm_100a has them inside sm100_atom's pre_fwd_kv / pre_bwd_kv)
        stem, func, smem = "lcross", _HOPPER_CROSS[name], 0
    elif name not in KERNELS:
        raise NotImplementedError(f"local atom kernel {name!r} exists on sm_90a only")
    else:
        stem, func, smem = KERNELS[name]
    pdl = (
        hopper
        and (name in _MMA_KERNELS or name in _MMA_KERNELS32)
        and settings.current().local_dit_h100_pdl
    )
    with torch.cuda.device(device_index):
        return driver.Kernel(cubin(stem, hopper), func, smem, pdl=pdl)


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


def _u8(kmask):
    """The key mask as the kernels read it: uint8 [N]. A contiguous bool mask is reinterpreted (no kernel launch), anything else converted."""
    if kmask is None:
        return None
    if kmask.dtype == torch.bool and kmask.is_contiguous():
        return kmask.view(torch.uint8)
    return kmask.to(torch.uint8).contiguous()


def attn_fwd(q, k, v, bias, kmask=None):
    """q, k, v [A, N, 128] bf16 (contiguous); bias [4, nwin, 32, 128] fp32; kmask [N] bool/uint8 or None -> O [A, N, 128] bf16,
    LSE [A, 4, N] fp32 (log2). tcgen05: a CTA per (128-query chunk, head) and sample range."""
    A, N, _ = q.shape
    nwin = nwindows(N)
    d = _dev(q)
    o = torch.empty_like(q)
    lse = torch.empty(A, NH, N, device=q.device, dtype=torch.float32)
    km = _u8(kmask)
    from miniworld_engine.kernels.augmented_attention.cuda.sm100_atom import _hopper

    if _hopper(
        d
    ):  # sm_90a: a CTA (4 warps of mma + a TMA warp) per (64 queries, head, sample); register softmax, wgmma P V
        if N % 64:
            raise ValueError("Local Hopper attention requires padded N divisible by 64")
        if HOPPER_MMA_ATTN:
            b2 = _mma_bias(bias, km, N, d)
            spc = _chunk(A, (N // 64) * NH)
            _load("h_fwd", d)(
                (N // 64, NH, (A + spc - 1) // spc),
                (128, 1, 1),
                q,
                k,
                v,
                b2,
                o,
                lse,
                int(N),
                int(A),
                int(spc),
            )
            return o, lse
        maps = tuple(
            _tmap(t, (DM, N, A), (DM * 2, N * DM * 2), (64, 64, 1), 128)
            for t in (q, k, v)
        )
        _load("fwd", d)(
            (N // 64, A * NH, 1), (160, 1, 1), *maps, bias, km, o, lse, int(N), int(A)
        )
        return o, lse
    nq = (N + 127) // 128
    sp = max(1, min(A, nsm(d) // (nq * NH)))
    out = _tmap(o, (DM, N, A), (DM * 2, N * DM * 2), (16, 32, 1), 0)
    _load("fwd", d)(
        (nq * NH * sp, 1, 1),
        (544, 1, 1),
        _rows(q, 128),
        _rows(k, 224),
        _rows(v, 224),
        out,
        bias,
        km,
        lse,
        int(N),
        int(A),
        int(nwin),
        int(sp),
    )
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
    km = _u8(kmask)
    from miniworld_engine.kernels.augmented_attention.cuda.sm100_atom import _hopper

    if _hopper(
        d
    ):  # sm_90a: dQ, dK / dV (+ per-sample dbias partials) as two wgmma kernels, then the fixed-order sample sum
        if N % 64 or any(
            t.dtype != torch.bfloat16 or not t.is_contiguous() for t in (q, k, v, do)
        ):
            raise ValueError(
                "Local Hopper backward requires contiguous bf16 operands and padded N divisible by 64"
            )
        dbias = torch.empty(NH, nwin, WQ, WK, device=q.device, dtype=torch.float32)
        if HOPPER_MMA_ATTN:
            b2 = _mma_bias(bias, km, N, d)
            spc = _chunk(A, (N // 64) * NH)
            sp = (A + spc - 1) // spc
            partial = torch.empty(
                sp, NH, nwin, WQ, WK, device=q.device, dtype=torch.float32
            )
            grid = (N // 64, NH, sp)
            _load("h_dq", d)(
                grid,
                (128, 1, 1),
                q,
                k,
                v,
                do,
                b2,
                lse,
                dd,
                dkv,
                int(ldd),
                int(N),
                int(A),
                int(spc),
            )
            _load("h_dkv", d)(
                grid,
                (128, 1, 1),
                q,
                k,
                v,
                do,
                b2,
                lse,
                dd,
                dkv,
                partial,
                int(ldd),
                int(N),
                int(A),
                int(spc),
            )
            _load("db_samples", d)(
                (min(4 * nsm(d), (dbias.numel() + 255) // 256), 1, 1),
                (256, 1, 1),
                partial,
                dbias,
                int(N),
                int(sp),
            )
            return dbias
        partial = torch.empty(A, NH, nwin, WQ, WK, device=q.device, dtype=torch.float32)
        maps = tuple(
            _tmap(t, (DM, N, A), (DM * 2, N * DM * 2), (64, 64, 1), 128)
            for t in (q, k, v, do)
        )
        _load("dq", d)(
            (N // 64, A * NH, 1),
            (160, 1, 1),
            *maps,
            bias,
            km,
            lse,
            dd,
            dkv,
            int(ldd),
            int(N),
            int(A),
        )
        _load("dkv", d)(
            (N // 64, A * NH, 1),
            (160, 1, 1),
            *maps,
            bias,
            km,
            lse,
            dd,
            dkv,
            partial,
            int(ldd),
            int(N),
            int(A),
        )
        _load("db_samples", d)(
            (min(4 * nsm(d), (dbias.numel() + 255) // 256), 1, 1),
            (256, 1, 1),
            partial,
            dbias,
            int(N),
            int(A),
        )
        return dbias
    dbias = torch.zeros(NH, nwin, WQ, WK, device=q.device, dtype=torch.float32)
    out = _tmap(dkv, (dkv.shape[1], N, A), (ldd * 2, N * ldd * 2), (16, 32, 1), 0)
    nq = (N + 127) // 128
    sp = max(
        1, min(A, nsm(d) // (nq * NH))
    )  # split the samples when there are few chunks
    _load("dq", d)(
        (nq * NH * sp, 1, 1),
        (544, 1, 1),
        _rows(q, 128),
        _rows(k, 224),
        _rows(v, 224),
        _rows(do, 128),
        _vec(lse, 128),
        _vec(dd, 128),
        out,
        bias,
        km,
        int(N),
        int(A),
        int(nwin),
        int(sp),
    )
    nk = (N + 112 + 127) // 128
    sp = (
        2 if 2 * nk * NH <= nsm(d) and A > 1 else 1
    )  # two sample halves add into dbias exactly (0 + a + b)
    _load("dkv", d)(
        (nk * NH * sp, 1, 1),
        (512, 1, 1),
        _rows(q, 224),
        _rows(k, 128),
        _rows(v, 128),
        _rows(do, 224),
        _vec(lse, 224),
        _vec(dd, 224),
        out,
        bias,
        km,
        dkv,
        dbias,
        int(ldd),
        int(N),
        int(A),
        int(nwin),
        int(sp),
    )
    return dbias


def _params(wb, gamma):
    """Wb [4, 16] and gamma [16] as the kernels read them: contiguous, bf16 or fp32 (anything else is converted to fp32)."""
    from miniworld_engine.kernels.augmented_attention.cuda.sm100_atom import _hopper

    if _hopper(_dev(wb)):
        if wb.dtype not in (torch.bfloat16, torch.float32) or gamma.dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError("Local Hopper pair parameters require BF16 or FP32")
        return (
            wb.contiguous(),
            gamma.contiguous(),
            int(wb.dtype == torch.bfloat16) | (int(gamma.dtype == torch.bfloat16) << 1),
        )
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
    _load("pb_f", d)(
        (min(4 * nsm(d), (rows + 255) // 256), 1, 1),
        (256, 1, 1),
        z.contiguous(),
        wb,
        gamma,
        out,
        int(rows),
        float(eps),
        bf,
    )
    return out


def pair_bias_bwd(z, gamma, wb, dbias, eps=1e-5):
    """dbias [4, nwin, 32, 128] fp32 -> dz (z's shape and dtype, contiguous), dgamma [16], dWb [4, 16] (fp32)."""
    rows = z.numel() // DP
    wb, gamma, bf = _params(wb, gamma)
    dz = torch.empty(
        z.shape, device=z.device, dtype=z.dtype
    )  # contiguous, like the z.contiguous() the kernel reads (a view-able dpair)
    d = _dev(z)
    nb = min(nsm(d), (rows + 255) // 256)
    part = torch.empty(nb, NH * DP, device=z.device, dtype=torch.float32)
    dgamma = torch.empty(DP, device=z.device, dtype=torch.float32)
    dwb = torch.empty(NH, DP, device=z.device, dtype=torch.float32)
    _load("pb_b", d)(
        (nb, 1, 1),
        (256, 1, 1),
        z.contiguous(),
        wb,
        gamma,
        dbias.contiguous(),
        dz,
        part,
        int(rows),
        float(eps),
        bf,
    )
    _load("pb_fin", d)((1, 1, 1), (64, 1, 1), part, int(nb), wb, gamma, dgamma, dwb, bf)
    return dz, dgamma, dwb


# ------------------------------------------------------------------------------------------------ cross-attention mode (second AdaLN for K / V)
def _row_grid(M, d):
    """Blocks of the row kernels of ``lcross.cu``: a row per half-warp (16 rows per block), persistent at 8 blocks per SM."""
    return max(1, min(8 * nsm(d), (M + 15) // 16))


def cond_ln(c, gamma, eps=1e-5):
    """c [M, 128] bf16 -> LN(c) * gamma [M, 128] bf16."""
    M = c.shape[0]
    out = torch.empty_like(c)
    d = _dev(c)
    _load("cond_ln", d)(
        (_row_grid(M, d), 1, 1),
        (256, 1, 1),
        c,
        gamma.float().contiguous(),
        out,
        int(M),
        float(eps),
    )
    return out


def cond_ln_bwd(c, dcn, gamma, dgamma, eps=1e-5, acc=None):
    """-> dc [M, 128] bf16; dgamma [128] fp32 += sum_rows dcn * LN(c). ``acc`` (bf16 [M, 128], contiguous): the result is added to it in
    bf16, in place (the gradient of c's other consumer), and returned.

    sm_90a: ``dgamma`` is *written* (not accumulated into): the blocks store their column sums into a [grid, 128] table and
    ``reduce_columns`` adds it in a fixed order (no atomics, deterministic), so ``dgamma`` may be uninitialised."""
    M = c.shape[0]
    dc = torch.empty_like(c) if acc is None else acc
    d = _dev(c)
    from miniworld_engine.kernels.augmented_attention.cuda.sm100_atom import _hopper

    if _hopper(d):
        nb = _row_grid(M, d)
        part = torch.empty(nb, DM, device=c.device, dtype=torch.float32)
        _load("cln_b", d)(
            (nb, 1, 1),
            (256, 1, 1),
            c,
            dcn,
            gamma.float().contiguous(),
            dc,
            acc,
            part,
            int(M),
            float(eps),
        )
        _load("reduce_columns", d)((1, 1, 1), (128, 1, 1), part, dgamma, int(nb))
        return dc
    _load("cln_b", d)(
        (_row_grid(M, d), 1, 1),
        (256, 1, 1),
        c,
        dcn,
        gamma.float().contiguous(),
        dc,
        acc,
        dgamma,
        int(M),
        float(eps),
    )
    return dc


# ------------------------------------------------------------------------------------------------ sm_90a cross-attention helpers
# On Hopper pre_fwd / pre_bwd carry no K / V AdaLN (sm_100a's pre_fwd_kv / pre_bwd_kv do): it is these row kernels around cuBLAS GEMMs.
def kv_fwd(x1, mkv, bs, eps=1e-5):
    """sm_90a. xkv = LN(x1) * sigmoid(mkv[:, :128] + bs) + mkv[:, 128:]  ([M, 128] bf16)."""
    M = x1.shape[0]
    out = torch.empty_like(x1)
    _load("kv_f", _dev(x1))(
        ((M + 7) // 8, 1, 1),
        (256, 1, 1),
        x1,
        mkv,
        bs.float().contiguous(),
        out,
        int(M),
        float(eps),
    )
    return out


def kv_bwd(x1, mkv, bs, dxkv, eps=1e-5):
    """sm_90a. dxkv -> dx1 [M, 128] and dmkv [M, 256] = [d pre-sigmoid scale | d shift] (bf16)."""
    M = x1.shape[0]
    dx1 = torch.empty_like(x1)
    dmkv = torch.empty(M, 2 * DM, device=x1.device, dtype=torch.bfloat16)
    _load("kv_b", _dev(x1))(
        ((M + 7) // 8, 1, 1),
        (256, 1, 1),
        x1,
        mkv,
        bs.float().contiguous(),
        dxkv,
        dx1,
        dmkv,
        int(M),
        float(eps),
    )
    return dx1, dmkv


def adaln1_extra(a, mod, dx1, ds, dmod, dbias, eps=1e-5):
    """sm_90a. ds [M, 128] and dmod [M, 768] (blocks 0 and 1) += the gradient dx1 through AdaLN 1 (in place). ``dbias`` is not touched: the
    scale-bias column sums come from ``sm100_atom.finish_columns_hopper`` (it reduces dmod after this call)."""
    M = a.shape[0]
    d = _dev(a)
    _load("a1x", d)(
        (min(nsm(d) * 4, (M + 7) // 8), 1, 1),
        (256, 1, 1),
        a,
        mod,
        dx1,
        ds,
        dmod,
        dbias,
        int(M),
        float(eps),
    )


def kv_bias_hopper(dmkv):
    """sm_90a. d (scale bias) [128] fp32 = column sums of dmkv[:, :128]'s pre-sigmoid gradient, in a fixed order."""
    d = _dev(dmkv)
    nb = 4 * nsm(d)
    partial = torch.empty(nb, DM, device=dmkv.device, dtype=torch.float32)
    out = torch.empty(DM, device=dmkv.device, dtype=torch.float32)
    _load("kv_bias_part", d)((nb, 1, 1), (256, 1, 1), dmkv, partial, int(dmkv.shape[0]))
    _load("reduce_columns", d)((1, 1, 1), (128, 1, 1), partial, out, nb)
    return out


def zero_kv_hopper(dp):
    """sm_90a. Zero the key / value gradient columns [128, 384) of dp [M, 512] (bf16)."""
    _load("zero_kv", _dev(dp))((256, 1, 1), (256, 1, 1), dp, int(dp.shape[0]))


def add_cond_hopper(dc, extra):
    """sm_90a. dc [M, 128] bf16 += extra (rounded to bf16 per element), in place."""
    _load("add_cond", _dev(dc))((256, 1, 1), (256, 1, 1), dc, extra, int(dc.shape[0]))


def cross_grad_pack_hopper(grads, weights):
    """sm_90a. The four cross-mode parameter gradients (fp32, contiguous) -> tensors of the weights' shapes and dtypes."""
    if len(grads) != 4 or any(
        g.dtype != torch.float32 or not g.is_contiguous() for g in grads
    ):
        raise ValueError(
            "Local Hopper cross gradient pack requires four contiguous FP32 gradients"
        )
    out = [torch.empty_like(w) for w in weights]
    bfmask = sum((w.dtype == torch.bfloat16) << i for i, w in enumerate(weights))
    _load("cross_grad_pack", _dev(weights[0]))(
        (128, 1, 1), (256, 1, 1), *grads, *out, int(bfmask)
    )
    return out


def kv_pack_hopper(weights, packed_pre):
    """sm_90a. (gamma, to_scale.weight, to_scale.bias, to_bias.weight) -> Wm [256, 128] bf16, scale bias, LN weight (fp32 [128]), and the
    Wk / Wv views of ``packed_pre`` (the pack's ``wp``)."""
    gamma, w_scale, b_scale, w_bias = weights
    dev = gamma.device
    wm = torch.empty(256, 128, device=dev, dtype=torch.bfloat16)
    bs, gg = (torch.empty(128, device=dev, dtype=torch.float32) for _ in range(2))
    if any(
        w.dtype not in (torch.bfloat16, torch.float32) or not w.is_contiguous()
        for w in weights
    ):
        raise ValueError(
            "Local Hopper KV pack requires contiguous BF16/FP32 parameters"
        )
    bfmask = sum((w.dtype == torch.bfloat16) << i for i, w in enumerate(weights))
    _load("kv_pack", _dev(gamma))(
        (128, 1, 1), (256, 1, 1), *weights, wm, bs, gg, int(bfmask)
    )
    return wm, bs, gg, packed_pre[0][128:256], packed_pre[0][256:384]


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


#: the fp32-path kernels that execute griddepcontrol.wait before any global access (programmatic dependent launch, ``sm100_atom.PDL32``)
_PDL_TF32 = ("fwd32",)


@functools.lru_cache(maxsize=None)
def _load32(name: str, device_index: int):
    from miniworld_engine.kernels.augmented_attention.cuda import sm100_atom
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    stem, func, smem = KERNELS_TF32[name]
    pdl = name in _PDL_TF32
    with torch.cuda.device(device_index):
        k = driver.Kernel(cubin(stem), func, smem, pdl=pdl and sm100_atom.PDL32[0])
    if pdl:
        sm100_atom._PDL32_LOADED.append(k)
    return k


def _rows32(t, n, swizzle=128):
    """TMA map of an fp32 [A, N, 128] view (rows ``t.stride(1)`` elements apart, e.g. a column block of [A, N, 512]): boxes of n rows x one
    head (32 fp32 = 128 B). swizzle 128: the K-major operand layout; "128a32": the 128-B swizzle with 32-B atoms, the MN-major one."""
    A, N, _ = t.shape
    assert t.stride(2) == 1 and t.dtype == torch.float32
    return _tmap(
        t, (DM, N, A), (t.stride(1) * 4, t.stride(0) * 4), (DH, n, 1), swizzle, "f32"
    )


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
    km = _u8(kmask)
    nq = (N + 127) // 128
    sp = max(1, min(A, nsm(d) // (nq * NH)))
    _load32("fwd32", d)(
        (nq * NH * sp, 1, 1),
        (544, 1, 1),
        _rows32(q, 128),
        _rows32(k, 224),
        _rows32(v, 224, "128a32"),
        _out32(o),
        bias,
        km,
        lse,
        int(N),
        int(A),
        int(nwin),
        int(sp),
    )
    return o, lse


def attn_bwd32(q, k, v, do, bias, lse, dd, dp, cols, kmask=None):
    """The fp32 path's backward: dQ, dK, dV (fp32) into the column blocks ``cols`` = (q, k, v) of dp [A N, C] and dbias [4, nwin, 32, 128]
    fp32 (summed over the samples). q, k, v fp32 [A, N, 128] views, do [A, N, 128] fp32, lse / dd [A, 4, N] fp32 (log2 units / D)."""
    A, N, _ = q.shape
    nwin = nwindows(N)
    d = _dev(q)
    ldd = dp.stride(0)
    km = _u8(kmask)
    dbias = torch.zeros(NH, nwin, WQ, WK, device=q.device, dtype=torch.float32)
    out = _out32(dp.view(A, N, dp.shape[1]))
    qcol, kcol, vcol = cols
    nq = (N + 127) // 128
    sp = max(
        1, min(A, nsm(d) // (nq * NH))
    )  # split the samples when there are few chunks
    _load32("dq32", d)(
        (nq * NH * sp, 1, 1),
        (544, 1, 1),
        _rows32(q, 128),
        _rows32(k, 224),
        _rows32(v, 224),
        _rows32(k, 224, "128a32"),
        _rows32(do, 128),
        _vec(lse, 128),
        _vec(dd, 128),
        out,
        bias,
        km,
        int(qcol),
        int(N),
        int(A),
        int(nwin),
        int(sp),
    )
    nk = (N + 112 + 127) // 128
    sp = (
        2 if 2 * nk * NH <= nsm(d) and A > 1 else 1
    )  # two sample halves add into dbias exactly (0 + a + b)
    _load32("dkv32", d)(
        (nk * NH * sp, 1, 1),
        (544, 1, 1),
        _rows32(q, 224),
        _rows32(q, 224, "128a32"),
        _rows32(k, 128),
        _rows32(v, 128),
        _rows32(do, 224),
        _rows32(do, 224, "128a32"),
        _vec(lse, 224),
        _vec(dd, 224),
        out,
        bias,
        km,
        dp,
        dbias,
        int(ldd),
        int(kcol),
        int(vcol),
        int(N),
        int(A),
        int(nwin),
        int(sp),
    )
    return dbias


def pair_bias_fwd32(z, gamma, wb, eps=1e-5):
    """z [nwin, 32, 128, 16] fp32 -> bias [4, nwin, 32, 128] fp32."""
    z = z.contiguous()
    nwin = z.shape[0]
    rows = z.numel() // DP
    wb, gamma, bf = _params(wb, gamma)
    out = torch.empty(NH, nwin, WQ, WK, device=z.device, dtype=torch.float32)
    d = _dev(z)
    _load32("pb_f32", d)(
        (min(4 * nsm(d), (rows + 255) // 256), 1, 1),
        (256, 1, 1),
        z,
        wb,
        gamma,
        out,
        int(rows),
        float(eps),
        bf,
    )
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
    _load32("pb_b32", d)(
        (nb, 1, 1),
        (256, 1, 1),
        z,
        wb,
        gamma,
        dbias.contiguous(),
        dz,
        part,
        int(rows),
        float(eps),
        bf,
    )
    _load32("pb_fin32", d)(
        (1, 1, 1), (64, 1, 1), part, int(nb), wb, gamma, dgamma, dwb, bf
    )
    return dz, dgamma, dwb


# ------------------------------------------------------------------------------------------------ sm_90a mma.sync attention (lattn90*.cu)
def pair_bias_mma(z, gamma, wb, kmask, n, eps=1e-5):
    """z [nwin, 32, 128, 16] bf16 / fp32 -> bias2 [4, nwin, 32, 128] fp32: LN(z) gamma . Wb log2(e), -inf at invalid / masked keys."""
    d = _dev(z)
    rows = z.numel() // DP
    out = torch.empty(NH, z.shape[0], WQ, WK, device=z.device, dtype=torch.float32)
    _load("h_pb", d)(
        (min(8 * nsm(d), (rows + 255) // 256), 1, 1),
        (256, 1, 1),
        z.contiguous(),
        gamma.float().contiguous(),
        wb.float().contiguous(),
        _u8(kmask),
        out,
        int(n),
        int(rows),
        float(eps),
        int(z.dtype == torch.bfloat16),
    )
    return out


def attn_fwd_mma32(q, k, v, bias2):
    """TF32 window attention forward. q, k, v [A, N, 128] fp32 (tf32-rounded), bias2 from ``pair_bias_mma`` -> O [A, N, 128] fp32, LSE."""
    A, N, _ = q.shape
    d = _dev(q)
    o = torch.empty_like(q)
    lse = torch.empty(A, NH, N, device=q.device, dtype=torch.float32)
    spc = _chunk(A, (N // 64) * NH)
    _load("h_fwd32", d)(
        (N // 64, NH, (A + spc - 1) // spc),
        (128, 1, 1),
        q,
        k,
        v,
        bias2,
        o,
        lse,
        int(N),
        int(A),
        int(spc),
    )
    return o, lse


def attn_fwd_mma(q, k, v, bias2):
    """bf16 window attention forward on the mma.sync kernel. q, k, v [A, N, 128] bf16, bias2 from ``pair_bias_mma`` -> O bf16, LSE."""
    A, N, _ = q.shape
    d = _dev(q)
    o = torch.empty_like(q)
    lse = torch.empty(A, NH, N, device=q.device, dtype=torch.float32)
    spc = _chunk(A, (N // 64) * NH)
    _load("h_fwd", d)(
        (N // 64, NH, (A + spc - 1) // spc),
        (128, 1, 1),
        q,
        k,
        v,
        bias2,
        o,
        lse,
        int(N),
        int(A),
        int(spc),
    )
    return o, lse


def attn_bwd_mma(q, k, v, do, bias2, lse, dd, dkv, tf32: bool):
    """dQ, dK, dV into dkv [A N, >= 384] (row stride ``dkv.stride(0)``, columns 0 / 128 / 256 relative to its base) and dbias
    [4, nwin, 32, 128] fp32 (summed over the samples). bias2 from ``pair_bias_mma``; q, k, v, do contiguous [A, N, 128]."""
    A, N, _ = q.shape
    d = _dev(q)
    nwin = nwindows(N)
    ldd = dkv.stride(0)
    spc = _chunk(A, (N // 64) * NH)
    sp = (A + spc - 1) // spc
    dbias = torch.empty(NH, nwin, WQ, WK, device=q.device, dtype=torch.float32)
    partial = torch.empty(sp, NH, nwin, WQ, WK, device=q.device, dtype=torch.float32)
    grid = (N // 64, NH, sp)
    sfx = "32" if tf32 else ""
    _load("h_dq" + sfx, d)(
        grid,
        (128, 1, 1),
        q,
        k,
        v,
        do,
        bias2,
        lse,
        dd,
        dkv,
        int(ldd),
        int(N),
        int(A),
        int(spc),
    )
    _load("h_dkv" + sfx, d)(
        grid,
        (128, 1, 1),
        q,
        k,
        v,
        do,
        bias2,
        lse,
        dd,
        dkv,
        partial,
        int(ldd),
        int(N),
        int(A),
        int(spc),
    )
    _load("db_samples", d)(
        (min(4 * nsm(d), (dbias.numel() + 255) // 256), 1, 1),
        (256, 1, 1),
        partial,
        dbias,
        int(N),
        int(sp),
    )
    return dbias


def pair_bias_bwd_mma(z, gamma, wb, dbias, eps=1e-5):
    """dbias [4, nwin, 32, 128] fp32 -> dz (z's shape / dtype), dgamma [16], dWb [4, 16] (fp32); z bf16 or fp32 [nwin, 32, 128, 16]."""
    d = _dev(z)
    rows = z.numel() // DP
    z = z.contiguous()
    dz = torch.empty_like(z)
    nb = min(nsm(d), (rows + 255) // 256)
    part = torch.empty(nb, NH * DP, device=z.device, dtype=torch.float32)
    dgamma = torch.empty(DP, device=z.device, dtype=torch.float32)
    dwb = torch.empty(NH, DP, device=z.device, dtype=torch.float32)
    wbf, gf = wb.float().contiguous(), gamma.float().contiguous()
    _load("h_pbb", d)(
        (nb, 1, 1),
        (256, 1, 1),
        z,
        wbf,
        gf,
        dbias.contiguous(),
        dz,
        part,
        int(rows),
        float(eps),
        int(z.dtype == torch.bfloat16),
    )
    _load("pb_fin", d)((1, 1, 1), (64, 1, 1), part, int(nb), wbf, gf, dgamma, dwb, 0)
    return dz, dgamma, dwb
