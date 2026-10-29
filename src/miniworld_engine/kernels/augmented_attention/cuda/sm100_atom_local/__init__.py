"""sm_100a (B200) kernels of the AF3 windowed atom attention (32 queries x 128 keys per window, per-window pair bias).

    S = q k^T / sqrt(32) + bias[h][w]      query window w = i // 32 sees the keys [32 w - 48, 32 w + 80) (clipped to [0, N))

The attention stage of ``modules/atom_local``'s block, forward and backward; mma.sync (the 32-row windows are too small for tcgen05's
128-row tiles), CUDA only. The cubins are built on first use like ``sm100_atom``'s and launch through the CUDA driver on torch's current stream.

  lbias.cu       bias[h, w, i, j] = LN(z[w, i, j]) . Wb[h] over the trunked pair z [nwin, 32, 128, 16], and its backward.
  lattn_fwd.cu   O and the row LSE.
  lattn_dq.cu    dQ (window-local).
  lattn_dkv.cu   dK, dV and dbias (key-centric: 16 keys see exactly 4 query windows = 128 queries; dbias is summed over the samples).
  lslide.cu      AlphaFold 3's slid key window: the (at most five) windows whose slid keys differ from the clipped ones, recomputed
                 over their slid keys after the window kernel (inference).
  lattn_esm.cu, lattn_esm_dq.cu, lattn_esm_dkv.cu, lesm.cu   ESMFold2's 192-key rank-window attention (forward; dQ + bias gradient;
                 dK / dV) and its q / k RMSNorm + rotary, position-indexed pair bias and conditioning SiLU (inference and training).
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

from miniworld_engine.kernels._device import num_sms

from miniworld_engine import settings

NH, DH, DM, DP = 4, 32, 128, 16
WQ, WK, KOFF = 32, 128, 48
_dir = Path(__file__).parent
SOURCES = ("lattn_fwd", "lattn_dq", "lattn_dkv", "lbias", "lcross", "lslide")


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
    "slide": ("lslide", "local_slide_edges", 0),
    "slide_p": ("lslide", "local_slide_bwd_prep", 0),
    "slide_b": ("lslide", "local_slide_bwd", 75520),
    "slide_f": ("lslide", "local_slide_bwd_fin", 0),
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
#: ESMFold2's attention (``lattn90_esm.cu``, sm_90a, inference): (function, dynamic shared memory)
_ESM_KERNELS = {
    "h_pb_esm": ("h_pair_bias_esm", 0),
    "h_esm": ("h_attn_esm", 64 * 168 * 4 + 2 * (64 + 192 + 192) * 40 * 2),
    "h_esm32": ("h_attn_esm_tf32", 64 * 168 * 4 + (64 + 192 + 192) * 36 * 4),
    "h_esm_dq": ("h_attn_esm_dq", 64 * 168 * 4 + (64 * 2 + 192 * 2) * 40 * 2 + 2 * 64 * 4),
    "h_esm_dkv": ("h_attn_esm_dkv", 4 * 16 * 164 * 4 + (192 * 2 + 64 * 2) * 40 * 2 + 2 * 192 * 4),
    "h_pbb_esm": ("h_pair_bias_esm_bwd", 0),
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
        (min(8 * num_sms(d), (bias.numel() + 255) // 256), 1, 1),
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
    if hopper and name in _ESM_KERNELS:
        stem, func, smem = "lattn90_esm", *_ESM_KERNELS[name]
    elif hopper and name in _MMA_KERNELS32:
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
        and (name in _MMA_KERNELS or name in _MMA_KERNELS32 or name in _ESM_KERNELS)
    )
    with torch.cuda.device(device_index):
        return driver.Kernel(cubin(stem, hopper), func, smem, pdl=pdl)


def _dev(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()




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
    sp = max(1, min(A, num_sms(d) // (nq * NH)))
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


def slide_edges(q, k, v, bias, kmask, o, n0, lse=None):
    """AlphaFold 3's slid key window (``lslide.cu``): after ``attn_fwd`` / ``attn_fwd32`` ran the clipped windows, recompute the windows
    whose slid keys differ (0, 1 and the last three real ones, chosen on the device from the key mask's real atom count, or ``n0``
    without a mask) over their slid keys, in place in ``o`` [A, N, 128] (and ``lse`` [A, 4, N] when given). q, k, v [A, N, 128] views with
    one row stride (bf16 projections or fp32 column blocks of P32); bias [4, nwin, 32, 128] fp32."""
    A, N, _ = o.shape
    ld = q.stride(1)
    assert k.stride(1) == v.stride(1) == ld and q.stride(0) == k.stride(0) == v.stride(0) == N * ld and q.stride(2) == 1
    assert o.is_contiguous() and q.dtype == o.dtype
    d = _dev(q)
    kern = _load32("slide32", d) if q.dtype == torch.float32 else _load("slide", d)
    km = _u8(kmask)
    if km is not None and km.data_ptr() % 4:  # the kernel counts the mask a word at a time
        km = km.clone()
    kern((5, NH, A), (512, 1, 1), q, k, v, int(ld), bias, km, o, lse, int(N), int(n0))
    return o


#: ESMFold2's atom attention on sm_100a (``LocalDiTBlock(**ESMFOLD2)``, bf16 inference), loaded apart from KERNELS (``load_esm``)
KERNELS_ESM = {
    "esm": ("lattn_esm", "local_attn_esm", 191232),
    "esm_bias": ("lesm", "local_esm_bias", 0),
    "esm_qk": ("lesm", "local_esm_qkrope", 0),
}


@functools.lru_cache(maxsize=None)
def _load_esm(name: str, device_index: int):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    stem, func, smem = KERNELS_ESM[name]
    with torch.cuda.device(device_index):
        return driver.Kernel(cubin(stem), func, smem)


def load_esm(device_index: int) -> None:
    """Build and load ESMFold2's attention kernels on the device (raises on any failure)."""
    for name in KERNELS_ESM:
        _load_esm(name, device_index)


def esm_bias(z, gamma, wb, kmask, n0, eps=1e-5):
    """ESMFold2's pair bias by position (``lesm.cu`` ``local_esm_bias``): z the trunked pair [nwin, 32, 192, 16] (bf16 or fp32,
    contiguous), gamma [16], Wb [4, 16], kmask [N] or None (then all n0 = N atoms are real) -> bias2 [4, nwin, 32, 160] fp32 (log2 units,
    -inf where the position is not a key of the query)."""
    nwin = z.shape[0]
    N = 32 * nwin
    assert tuple(z.shape) == (nwin, 32, ESM_KEYS, DP) and z.is_contiguous() and N % 128 == 0
    d = _dev(z)
    out = torch.empty(NH, nwin, 32, ESM_SPAN, device=z.device, dtype=torch.float32)
    km = _u8(kmask)
    if km is not None and km.data_ptr() % 16:
        km = km.clone()
    wb_, g_ = wb.contiguous(), gamma.contiguous()
    flags = int(wb_.dtype == torch.bfloat16) | (int(g_.dtype == torch.bfloat16) << 1) | (int(z.dtype == torch.bfloat16) << 2)
    assert {wb_.dtype, g_.dtype, z.dtype} <= {torch.bfloat16, torch.float32}
    _load_esm("esm_bias", d)((min(4 * num_sms(d), (nwin * 32 * ESM_SPAN + 255) // 256), 1, 1), (256, 1, 1), z, g_, wb_, km, out,
                             int(N), int(n0), float(eps), flags)
    return out


def esm_qkrope(q, k, wq, wk, cos, sin, eps):
    """ESMFold2's q / k RMSNorm (per head, weights fp32 [32], contiguous) and rotary (tables cos / sin fp32 [n, 16], row r is atom r % n): q, k [M, 128]
    bf16 views with one row stride -> qo, ko [M, 128] bf16 (``lesm.cu`` ``local_esm_qkrope``)."""
    M = q.shape[0]
    assert q.stride(0) == k.stride(0) and q.stride(1) == k.stride(1) == 1 and q.dtype == k.dtype == torch.bfloat16
    assert cos.dtype == sin.dtype == torch.float32 and cos.is_contiguous() and sin.is_contiguous() and cos.shape[1] == 16
    qo, ko = torch.empty(M, DM, device=q.device, dtype=q.dtype), torch.empty(M, DM, device=q.device, dtype=q.dtype)
    assert wq.dtype == wk.dtype == torch.float32 and wq.is_contiguous() and wk.is_contiguous()
    _load_esm("esm_qk", _dev(q))(((M + 7) // 8, 1, 1), (256, 1, 1), q, k, int(q.stride(0)), wq, wk,
                                 cos, sin, qo, ko, int(M), int(cos.shape[0]), float(eps))
    return qo, ko


def attn_esm_sm100(q, k, v, bias2, lse=None):
    """ESMFold2's atom attention (``lattn_esm.cu``, tcgen05): q, k, v [A, N, 128] bf16 (contiguous; q / k normalised and rotated), bias2
    [4, nwin, 32, 160] (``esm_bias``) -> O [A, N, 128] bf16 (and the row LSE, log2 units, into ``lse`` [A, 4, N] when given)."""
    A, N, _ = q.shape
    nwin = nwindows(N)
    d = _dev(q)
    o = torch.empty_like(q)
    nq = (N + 127) // 128
    sp = max(1, min(A, num_sms(d) // (nq * NH)))
    out = _tmap(o, (DM, N, A), (DM * 2, N * DM * 2), (16, 32, 1), 0)
    _load_esm("esm", d)(
        (nq * NH * sp, 1, 1),
        (544, 1, 1),
        _rows(q, 128),
        _rows(k, 256),
        _rows(v, 256),
        out,
        bias2,
        lse,
        int(N),
        int(A),
        int(nwin),
        int(sp),
    )
    return o


#: ESMFold2's training kernels on sm_100a (bf16): the attention backward (``lattn_esm_dq.cu``: dQ + the bias gradient; ``lattn_esm_dkv.cu``: dK,
#: dV), the q / k RMSNorm + rotary backward and the conditioning's SiLU backward (``lesm.cu``), the position-indexed pair bias backward
#: (``lbias.cu``; its block partials reduced by ``local_bias_fin``). Loaded apart from the inference kernels (``load_esm_train``).
KERNELS_ESM_TRAIN = {
    "esm_dq": ("lattn_esm_dq", "local_attn_esm_dq", 193280),
    "esm_dkv": ("lattn_esm_dkv", "local_attn_esm_dkv", 200960),
    "esm_qk_b": ("lesm", "local_esm_qkrope_bwd", 0),
    "esm_qk_fin": ("lesm", "local_esm_qk_fin", 0),
    "esm_silu_b": ("lesm", "local_esm_silu_bwd", 0),
    "esm_bias_b": ("lbias", "local_esm_bias_bwd", 0),
    "pb_fin": ("lbias", "local_bias_fin", 0),
}


@functools.lru_cache(maxsize=None)
def _load_esm_train(name: str, device_index: int):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    stem, func, smem = KERNELS_ESM_TRAIN[name]
    with torch.cuda.device(device_index):
        return driver.Kernel(cubin(stem), func, smem)


def load_esm_train(device_index: int) -> None:
    """Build and load ESMFold2's training kernels on the device (raises on any failure)."""
    for name in KERNELS_ESM_TRAIN:
        _load_esm_train(name, device_index)


def attn_esm_bwd_sm100(q, k, v, do, bias2, lse, dd, dkv):
    """The backward of :func:`attn_esm_sm100` (tcgen05): q, k, v, do [A, N, 128] bf16 contiguous (q / k the normalised, rotated ones), bias2
    [4, nwin, 32, 160] (log2), lse [A, 4, N] (the forward's, log2), dd [A, 4, N] = rowsum(dO O) per head -> dQ, dK, dV (bf16) into the column
    blocks 0 / 1 / 2 of dkv [A N, 512] (row stride dkv.stride(0)); returns the bias gradient [4, nwin, 32, 160] fp32 (natural units, by
    position, summed over the samples in a fixed order: deterministic)."""
    A, N, _ = q.shape
    nwin = nwindows(N)
    d = _dev(q)
    ldd = dkv.stride(0)
    nq = N // 128
    out = _tmap(dkv, (dkv.shape[1], N, A), (ldd * 2, N * ldd * 2), (16, 32, 1), 0)
    sp = max(1, min(A, num_sms(d) // (nq * NH)))
    dbias = torch.empty((sp, NH, nwin, WQ, ESM_SPAN) if sp > 1 else (NH, nwin, WQ, ESM_SPAN), device=q.device, dtype=torch.float32)
    _load_esm_train("esm_dq", d)(
        (nq * NH * sp, 1, 1), (512, 1, 1),
        _rows(q, 128), _rows(k, 256), _rows(v, 256), _rows(do, 128), _vec(lse, 128), _vec(dd, 128), out,
        bias2, dbias, int(N), int(A), int(nwin), int(sp),
    )
    sp2 = max(1, min(A, num_sms(d) // (nq * NH)))
    _load_esm_train("esm_dkv", d)(
        (nq * NH * sp2, 1, 1), (512, 1, 1),
        _rows(q, 256), _rows(k, 128), _rows(v, 128), _rows(do, 256), _vec(lse, 256), _vec(dd, 256), out,
        bias2, int(N), int(A), int(nwin), int(sp2),
    )
    return dbias.sum(0) if sp > 1 else dbias


def esm_qkrope_bwd(dq, dk, q, k, wq, wk, cos, sin, eps):
    """The backward of :func:`esm_qkrope`, in place: dq, dk [M, 128] bf16 views with one row stride (the gradients of the rotated q / k) become
    the gradients of q / k [M, 128] (the projections' bf16 outputs, one row stride). -> the norm weights' gradients [64] fp32 (q 0..31, k 32..63,
    summed over the rows and heads in a fixed order)."""
    M = q.shape[0]
    assert dq.stride(0) == dk.stride(0) and dq.stride(1) == dk.stride(1) == 1 and q.stride(0) == k.stride(0) and q.stride(1) == k.stride(1) == 1
    assert cos.dtype == sin.dtype == torch.float32 and cos.is_contiguous() and sin.is_contiguous() and cos.shape[1] == 16
    d = _dev(q)
    nb = max(1, min(4 * num_sms(d), (M + 7) // 8))
    part = torch.empty(nb, 64, device=q.device, dtype=torch.float32)
    out = torch.empty(64, device=q.device, dtype=torch.float32)
    _load_esm_train("esm_qk_b", d)((nb, 1, 1), (256, 1, 1), dq, dk, int(dq.stride(0)), q, k, int(q.stride(0)), wq, wk, cos, sin, part,
                                   int(M), int(cos.shape[0]), float(eps))
    _load_esm_train("esm_qk_fin", d)((1, 1, 1), (512, 1, 1), part, int(nb), out)
    return out


def esm_silu_bwd(c, g):
    """c [M, 128] bf16 (the conditioning), g [M, 128] fp32 (the tables' gradient times their weights) -> dc = rn(g silu'(c)), sc = rn(silu(c))."""
    assert c.is_contiguous() and g.is_contiguous() and g.dtype == torch.float32 and c.shape == g.shape
    dc, sc = torch.empty_like(c), torch.empty_like(c)
    n4 = c.numel() // 4
    d = _dev(c)
    _load_esm_train("esm_silu_b", d)((max(1, min(8 * num_sms(d), (n4 + 255) // 256)), 1, 1), (256, 1, 1), c, g, dc, sc, int(n4))
    return dc, sc


def esm_bias_bwd(z, gamma, wb, dbias2, kmask, n0, eps=1e-5):
    """The backward of :func:`esm_bias`: z the trunked pair [nwin, 32, 192, 16] bf16 (contiguous), dbias2 [4, nwin, 32, 160] fp32 (natural
    units) -> dz (z's shape, bf16), dgamma [16], dWb [4, 16] (fp32; the block partials summed in order)."""
    nwin = z.shape[0]
    N = 32 * nwin
    assert tuple(z.shape) == (nwin, 32, ESM_KEYS, DP) and z.is_contiguous() and z.dtype == torch.bfloat16 and N % 128 == 0
    d = _dev(z)
    wb, gamma, bf = _params(wb, gamma)
    km = _u8(kmask)
    if km is not None and km.data_ptr() % 16:
        km = km.clone()
    rows = z.numel() // DP
    dz = torch.empty_like(z)
    nb = min(num_sms(d), (rows + 255) // 256)
    part = torch.empty(nb, NH * DP, device=z.device, dtype=torch.float32)
    dgamma = torch.empty(DP, device=z.device, dtype=torch.float32)
    dwb = torch.empty(NH, DP, device=z.device, dtype=torch.float32)
    _load_esm_train("esm_bias_b", d)((nb, 1, 1), (256, 1, 1), z, wb, gamma, dbias2.contiguous(), km, dz, part, int(N), int(n0), float(eps), bf)
    _load_esm_train("pb_fin", d)((1, 1, 1), (64, 1, 1), part, int(nb), wb, gamma, dgamma, dwb, bf)
    return dz, dgamma, dwb


def slide_bwd_prep(do, dd, kmask, n0, A):
    """Before the clipped-window backward of a slid-window block: the moved windows' rows of dO [A N, 128] and D [A, 4, N] are copied out and
    zeroed in place (``lslide.cu``) -> the saved (sdo, sdd) for ``slide_bwd``."""
    N = dd.shape[2]
    assert do.is_contiguous() and dd.is_contiguous() and do.shape[0] == A * N
    f32 = do.dtype == torch.float32
    d = _dev(do)
    sdo = torch.empty(5, A, WQ, DM, device=do.device, dtype=do.dtype)
    sdd = torch.empty(5, A, NH, WQ, device=do.device, dtype=torch.float32)
    km = _u8(kmask)
    if km is not None and km.data_ptr() % 16:
        km = km.clone()
    (_load32("slide_p32", d) if f32 else _load("slide_p", d))((5, A, 1), (256, 1, 1), do, dd, km, sdo, sdd, int(N), int(n0), int(A))
    return sdo, sdd


def slide_bwd(q, k, v, bias, kmask, n0, lse, saved, dq, dk, dv, dbias):
    """After the clipped-window backward of a slid-window block (its dO / D rows of the moved windows zeroed by ``slide_bwd_prep``, whose
    ``saved`` they are): the moved windows' gradients over their slid keys -- dQ written into ``dq``, dK / dV added into ``dk`` / ``dv``, their
    dbias windows of ``dbias`` [4, nwin, 32, 128] written (summed over the samples in order; deterministic). q, k, v, dq, dk, dv [A, N, 128]
    views with one row stride each (q / k / v one, the gradients one); lse [A, 4, N] (the slid forward's)."""
    A, N, _ = q.shape
    ld, ldd = q.stride(1), dq.stride(1)
    assert k.stride(1) == v.stride(1) == ld and dk.stride(1) == dv.stride(1) == ldd
    f32 = q.dtype == torch.float32
    d = _dev(q)
    sdo, sdd = saved
    km = _u8(kmask)
    if km is not None and km.data_ptr() % 16:
        km = km.clone()
    skv = torch.empty(5, A, WK, NH, 64, device=q.device, dtype=torch.float32)
    sdb = torch.empty(5, A, NH, WQ, WK, device=q.device, dtype=torch.float32)
    (_load32("slide_b32", d) if f32 else _load("slide_b", d))(
        (5, NH, A), (256, 1, 1), q, k, v, int(ld), bias, km, lse, sdo, sdd, dq, int(ldd), skv, sdb, int(N), int(n0), int(A)
    )
    total = skv.numel() + 5 * NH * WQ * WK
    (_load32("slide_f32", d) if f32 else _load("slide_f", d))(
        (min(8 * num_sms(d), (total + 255) // 256), 1, 1), (256, 1, 1), km, skv, sdb, dk, dv, int(ldd), dbias, int(N), int(n0), int(A)
    )


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
                (min(4 * num_sms(d), (dbias.numel() + 255) // 256), 1, 1),
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
            (min(4 * num_sms(d), (dbias.numel() + 255) // 256), 1, 1),
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
        1, min(A, num_sms(d) // (nq * NH))
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
        2 if 2 * nk * NH <= num_sms(d) and A > 1 else 1
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
        (min(4 * num_sms(d), (rows + 255) // 256), 1, 1),
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
    nb = min(num_sms(d), (rows + 255) // 256)
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
    return max(1, min(8 * num_sms(d), (M + 15) // 16))


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
        (min(num_sms(d) * 4, (M + 7) // 8), 1, 1),
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
    nb = 4 * num_sms(d)
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
    "slide32": ("lslide", "local_slide_edges_f32", 0),
    "slide_p32": ("lslide", "local_slide_bwd_prep_f32", 0),
    "slide_b32": ("lslide", "local_slide_bwd_f32", 75520),
    "slide_f32": ("lslide", "local_slide_bwd_fin_f32", 0),
}


#: the fp32-path kernels that execute griddepcontrol.wait before any global access (programmatic dependent launch)
_PDL_TF32 = ("fwd32",)


@functools.lru_cache(maxsize=None)
def _load32(name: str, device_index: int):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    stem, func, smem = KERNELS_TF32[name]
    with torch.cuda.device(device_index):
        return driver.Kernel(cubin(stem), func, smem, pdl=name in _PDL_TF32)


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
    sp = max(1, min(A, num_sms(d) // (nq * NH)))
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
        1, min(A, num_sms(d) // (nq * NH))
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
        2 if 2 * nk * NH <= num_sms(d) and A > 1 else 1
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
        (min(4 * num_sms(d), (rows + 255) // 256), 1, 1),
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
    nb = min(num_sms(d), (rows + 255) // 256)
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
        (min(8 * num_sms(d), (rows + 255) // 256), 1, 1),
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
        (min(4 * num_sms(d), (dbias.numel() + 255) // 256), 1, 1),
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
    nb = min(num_sms(d), (rows + 255) // 256)
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


# ------------------------------------------------------------------------------------------------ sm_90a ESMFold2 attention (lattn90_esm.cu)
ESM_KEYS, ESM_SPAN = 192, 160


def pair_bias_esm(z, gamma, wb, kmask, nreal, n, eps=1e-5):
    """ESMFold2's trunked pair z [nwin, 32, 192, 16] bf16 / fp32 -> bias2 [4, nwin, 32, 160] fp32 by POSITION: entry (w, i, c) is the bias of
    the key at 32 w - 64 + c, read from the slot of AlphaFold 3's 192-key slid subset that holds it (log2 domain), -inf outside the +/- 64
    rank window, outside the real keys (``kmask`` [n] or None) or outside the subset. ``nreal``: the real atom count, a device int32 scalar."""
    d = _dev(z)
    out = torch.empty(NH, z.shape[0], WQ, ESM_SPAN, device=z.device, dtype=torch.float32)
    _load("h_pb_esm", d)(
        (min(8 * num_sms(d), (out[0].numel() + 255) // 256), 1, 1),
        (256, 1, 1),
        z.contiguous(),
        gamma.float().contiguous(),
        wb.float().contiguous(),
        _u8(kmask),
        nreal,
        out,
        int(n),
        float(eps),
        int(z.dtype == torch.bfloat16),
    )
    return out


def attn_esm(q, k, v, bias2, tf32: bool):
    """ESMFold2's window attention forward (inference). q, k, v [A, N, 128] (bf16, or fp32 rounded to tf32; q / k normalised and rotated),
    bias2 from ``pair_bias_esm`` -> O [A, N, 128] in q's dtype. N a multiple of 64."""
    A, N, _ = q.shape
    d = _dev(q)
    o = torch.empty_like(q)
    spc = _chunk(A, (N // 64) * NH)
    grid = (N // 64, NH, (A + spc - 1) // spc)
    if tf32:
        _load("h_esm32", d)(grid, (128, 1, 1), q, k, v, bias2, o, int(N), int(A), int(spc))
    else:
        _load("h_esm", d)(grid, (128, 1, 1), q, k, v, bias2, o, None, int(N), int(A), int(spc))
    return o


def attn_esm_fwd_lse(q, k, v, bias2):
    """The bf16 forward of :func:`attn_esm` for training: O [A, N, 128] and the row LSE [A, 4, N] fp32 (log2 units; 1e30 for a row with no key)."""
    A, N, _ = q.shape
    d = _dev(q)
    o = torch.empty_like(q)
    lse = torch.empty(A, NH, N, device=q.device, dtype=torch.float32)
    spc = _chunk(A, (N // 64) * NH)
    _load("h_esm", d)((N // 64, NH, (A + spc - 1) // spc), (128, 1, 1), q, k, v, bias2, o, lse, int(N), int(A), int(spc))
    return o, lse


def attn_esm_bwd(q, k, v, do, bias2, lse, dd, dkv):
    """bf16 backward of the ESMFold2 window attention: dQ, dK, dV into ``dkv`` [A N, >= 384] (row stride ``dkv.stride(0)``, columns 0 / 128 / 256
    relative to its base) and the bias gradient [4, nwin, 32, 160] fp32 (natural units, by position, summed over the samples). ``lse`` from
    :func:`attn_esm_fwd_lse`, ``dd`` [A, 4, N] = rowsum(dO * O) per head."""
    A, N, _ = q.shape
    d = _dev(q)
    nwin = nwindows(N)
    ldd = dkv.stride(0)
    spc = _chunk(A, (N // 64) * NH)
    sp = (A + spc - 1) // spc
    partial = torch.empty(sp, NH, nwin, WQ, ESM_SPAN, device=q.device, dtype=torch.float32)
    grid = (N // 64, NH, sp)
    _load("h_esm_dq", d)(grid, (128, 1, 1), q, k, v, do, bias2, lse, dd, dkv, partial, int(ldd), int(N), int(A), int(spc))
    _load("h_esm_dkv", d)(grid, (128, 1, 1), q, k, v, do, bias2, lse, dd, dkv, int(ldd), int(N), int(A), int(spc))
    return partial.sum(0)


def pair_bias_esm_bwd(z, gamma, wb, dbias, nreal, eps=1e-5):
    """dbias [4, nwin, 32, 160] fp32 (natural units) -> dz (z's shape / dtype), dgamma [16], dWb [4, 16] (fp32); ``nreal`` as in :func:`pair_bias_esm`."""
    d = _dev(z)
    n = dbias.shape[1] * WQ
    z = z.contiguous()
    rows = z.numel() // DP
    dz = torch.empty_like(z)
    nb = min(num_sms(d), (rows + 255) // 256)
    part = torch.empty(nb, NH * DP, device=z.device, dtype=torch.float32)
    dgamma = torch.empty(DP, device=z.device, dtype=torch.float32)
    dwb = torch.empty(NH, DP, device=z.device, dtype=torch.float32)
    wbf, gf = wb.float().contiguous(), gamma.float().contiguous()
    _load("h_pbb_esm", d)(
        (nb, 1, 1), (256, 1, 1), z, wbf, gf, dbias.contiguous(), nreal, dz, part, int(n), float(eps), int(z.dtype == torch.bfloat16)
    )
    _load("pb_fin", d)((1, 1, 1), (64, 1, 1), part, int(nb), wbf, gf, dgamma, dwb, 0)
    return dz, dgamma, dwb
