"""B200-only (sm_100a, card ``b200_sm100``) kernels of the AF3 windowed atom attention (32 queries x 128 keys per window, per-window
pair bias): ``origin/b200/ladder-final``'s ``sm100_atom_local`` (dc425caa5) in its own package (``sm100_atom_local`` keeps the H100
build's sources). ESMFold2's rank-window kernels and the H100 mma.sync files were not taken (only ``kernels/integrations/local_dit_b200``
calls this package). Nothing here is built or loaded on another card.

    S = q k^T / sqrt(32) + bias[h][w]      query window w = i // 32 sees the keys [32 w - 48, 32 w + 80) (clipped to [0, N))

The attention stage of ``modules/atom_local``'s block, forward and backward; mma.sync (the 32-row windows are too small for tcgen05's
128-row tiles), CUDA only. The cubins are built on first use like ``sm100_atom``'s and launch through the CUDA driver on torch's current stream.

  lbias.cu       bias[h, w, i, j] = LN(z[w, i, j]) . Wb[h] over the trunked pair z [nwin, 32, 128, 16], and its backward.
  lattn_fwd.cu   O and the row LSE.
  lattn_dq.cu    dQ (window-local).
  lattn_dkv.cu   dK, dV and dbias (key-centric: 16 keys see exactly 4 query windows = 128 queries; dbias is summed over the samples).
  slide.cuh      AlphaFold 3's slid key window inside lattn_fwd / dq / dkv (and their tf32 twins): the windows whose slid keys differ
                 from the clipped ones run as edge CTAs of the same launches, on the tensor cores -- the TAIL too (the query windows of
                 padding atoms, which read the last 128 real keys: groups of 4 windows, slide_ngroups; 2026-11-23, before: CUDA cores in
                 the one group-1 CTA per head, ~114 ms dQ + dK / dV at N = 4096, 2,048 real atoms, 48 samples).
  lslide.cu      the slid keys' dK / dV partials of those edge CTAs (one per group), added into dk / dv in group order (deterministic).
  lattn_esm.cu, lattn_esm_dq.cu, lattn_esm_dkv.cu, lesm.cu   ESMFold2's 192-key rank-window attention (forward; dQ + bias gradient;
                 dK / dV) and its q / k RMSNorm + rotary, position-indexed pair bias and conditioning SiLU (inference and training).
                 Their fp32 twins: lattn_esm_tf32.cu, lattn_esm_dq_tf32.cu, lattn_esm_dkv_tf32.cu (tcgen05 kind::tf32).
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


NH, DH, DM, DP = 4, 32, 128, 16
WQ, WK, KOFF = 32, 128, 48
_dir = Path(__file__).parent
SOURCES = ("lattn_fwd", "lattn_dq", "lattn_dkv", "lbias", "lcross", "lslide")


@functools.lru_cache(maxsize=None)
def cubin(stem: str, hopper: bool = False, defs: tuple = ()) -> str:
    """Path of ``<stem>.cu`` built for sm_100a (+ ``-D`` ``defs``); rebuilt only when a source or a flag changes."""
    from miniworld_engine.kernels import _nvcc

    flags = (
        "-std=c++17",
        "-O3",
        "-arch=sm_90a" if hopper else "-arch=sm_100a",
        *(("-DATOM_HOPPER=1",) if hopper else ()),
        "-cubin",
        "-lineinfo",
        f"-I{_dir}",
        *("-D" + d for d in defs),
    )
    family = "atom_local_sm90" if hopper else "atom_local_b200"
    return str(_nvcc.cubin(f"{family}/{stem}", flags, source=_dir / f"{stem}.cu", toolchain="torch" if hopper else "sm100"))


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
    "ssum": ("lcross", "local_sample_sum", 0),
    "tsum": ("lcross", "local_table_sum", 0),
    "slide_add": ("lslide", "local_slide_add", 0),
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
def _load_pdl_fwd(device_index: int):
    """``local_attn_fwd`` built ``-DLATTN_PDL`` and launched as a programmatic dependent (the B200 variant path, 2026-11-17): it waits for
    the previous kernel before any global access and lets the next one (``post_fwd``) stage its weights while it runs."""
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    stem, func, smem = KERNELS["fwd"]
    with torch.cuda.device(device_index):
        return driver.Kernel(cubin(stem, False, ("LATTN_PDL=1",)), func, smem, pdl=True)


@functools.lru_cache(maxsize=None)
def _load(name: str, device_index: int):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    from miniworld_engine.kernels.augmented_attention.cuda.b200_atom import _hopper

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
    elif hopper and name in _HOPPER_CROSS:  # the K / V AdaLN row kernels (sm_100a has them inside sm100_atom's pre_fwd_kv / pre_bwd_kv)
        stem, func, smem = "lcross", _HOPPER_CROSS[name], 0
    elif name not in KERNELS:
        raise NotImplementedError(f"local atom kernel {name!r} exists on sm_90a only")
    else:
        stem, func, smem = KERNELS[name]
    pdl = hopper and (name in _MMA_KERNELS or name in _MMA_KERNELS32 or name in _ESM_KERNELS)
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


def _slide_mask(km):
    """The key mask as the slid-window CTAs count it (16 bytes at a time: 16-B aligned)."""
    return km.clone() if km is not None and km.data_ptr() % 16 else km


def slide_ngroups(nwin: int) -> int:
    """Edge groups of the fp32 path's launches (``slide.cuh`` slide_ngroups: group 0, group 1 and the tail as groups of 4 windows on the
    tensor cores), on the bf16 and the fp32 path alike."""
    return 1 + (nwin + 1) // 4


def _edge_split(main_ctas, sp, A, d, cap=None, groups=2):
    """Sample splits of the slid window's edge CTAs (``groups`` groups x 4 heads each, ``slide.cuh``; they come after the clipped-window CTAs
    in the grid). In sample-times: the clipped-window CTAs take tm = ceil(A / sp); edge CTAs that fit the SMs the last wave leaves free run
    beside them (makespan max(tm, ceil(A / f))), more of them run partly as a tail (~ tm + ceil(A / s)). The better of the two, at most
    ``cap`` (the dK / dV kernel: 2, its dbias adds exactly)."""
    hi = min(A, cap or A)
    tm = -(-A // sp)
    f = min(hi, (-main_ctas % num_sms(d)) // (groups * NH))
    s = min(hi, 2 * sp)
    if f >= 1 and max(tm, -(-A // f)) <= tm + -(-A // s):
        return f
    return s


def attn_fwd(q, k, v, bias, kmask=None, slide=False, qblock=False, folds=1, pdl=False):
    """q, k, v [A, N, 128] bf16 (contiguous); bias [4, nwin, 32, 128] fp32; kmask [N] bool/uint8 or None -> O [A, N, 128] bf16,
    LSE [A, 4, N] fp32 (log2). tcgen05: a CTA per (128-query chunk, head) and sample range. ``slide``: AlphaFold 3's slid key window
    (``slide.cuh``: the windows it moves run in edge CTAs of the same launch; the real atom count is the mask's population count).
    ``qblock``: IntelliFold2's ``slide_qblock`` rule (the slide against the real atom count rounded up to 32). ``folds`` F > 1 (inference):
    q, k, v [S F, N, 128] with rows [S][F][N], bias [4, F nwin, 32, 128] (the F folds' trunked pair biases end to end: one
    ``pair_bias_fwd`` of the F trunks)."""
    A, N, _ = q.shape
    nf = int(folds)
    assert A % nf == 0 and (nf == 1 or bias.shape[1] == nf * nwindows(N)), (A, nf, tuple(bias.shape))
    A_all, A = A, A // nf
    nwin = nwindows(N)
    d = _dev(q)
    o = torch.empty_like(q)
    lse = torch.empty(A_all, NH, N, device=q.device, dtype=torch.float32)
    km = _u8(kmask)
    from miniworld_engine.kernels.augmented_attention.cuda.b200_atom import _hopper

    if _hopper(d):  # sm_90a: a CTA (4 warps of mma + a TMA warp) per (64 queries, head, sample); register softmax, wgmma P V
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
                1,
            )
            return o, lse
        maps = tuple(_tmap(t, (DM, N, A), (DM * 2, N * DM * 2), (64, 64, 1), 128) for t in (q, k, v))
        _load("fwd", d)((N // 64, A * NH, 1), (160, 1, 1), *maps, bias, km, o, lse, int(N), int(A))
        return o, lse
    nq = (N + 127) // 128
    sp = max(1, min(A, num_sms(d) // (nq * NH * nf)))
    spe = _edge_split(nq * NH * sp * nf, sp, A, d, groups=slide_ngroups(nwin)) if slide else 0
    if slide:
        km = _slide_mask(km)
    out = _tmap(o, (DM, N, A_all), (DM * 2, N * DM * 2), (16, 32, 1), 0)
    (_load_pdl_fwd(d) if pdl else _load("fwd", d))(
        ((nq * sp + slide_ngroups(nwin) * spe) * NH, nf, 1),
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
        int(spe),
        int(bool(qblock)),
        nf,
        q,  # the raw operands (unused since the tail runs as edge groups on the tensor cores, slide.cuh)
        k,
        v,
        o,
    )
    return o, lse


def _tail_strides(t, *same):
    """(row, sample) strides in elements of an [A, N, 128] operand view (unit column stride; ``same``: views with the same strides)."""
    assert t.stride(2) == 1 and all(s.stride() == t.stride() for s in same), (t.stride(), [s.stride() for s in same])
    return int(t.stride(1)), int(t.stride(0))


def _slide_add(kern, spart, dkv, ldd, kcol, vcol, N, A):
    """dK / dV of the slid keys += the edge CTAs' fp32 partials (``lslide.cu``; deterministic)."""
    kern((A * 8, 1, 1), (256, 1, 1), spart, dkv, int(ldd), int(kcol), int(vcol), int(N), int(A))


def attn_bwd(q, k, v, do, bias, lse, dd, dkv, kmask=None, slide=False, qblock=False):
    """dQ, dK, dV (bf16) into dkv [A N, 512] (blocks 0 / 1 / 2) and dbias [4, nwin, 32, 128] fp32 (summed over the samples). ``slide``:
    AlphaFold 3's slid key window (edge CTAs in both launches, ``slide.cuh``; their dK / dV partials added by ``lslide.cu``); ``qblock``:
    IntelliFold2's slide_qblock rule.

    ``dd`` is D = rowsum(dO * O) per head [A, 4, N] fp32, ``lse`` the forward's [A, 4, N] (log2 units). q, k, v, do are contiguous
    [A, N, 128]. dQ (query-centric) and dK / dV / dbias (key-centric) run on tcgen05, a CTA per (128-row chunk, head) and sample range,
    through TMA maps over the [A, N, *] operands (zero fill outside [0, N), stores clipped at N)."""
    A, N, _ = q.shape
    nwin = nwindows(N)
    d = _dev(q)
    ldd = dkv.stride(0)
    km = _u8(kmask)
    from miniworld_engine.kernels.augmented_attention.cuda.b200_atom import _hopper

    if _hopper(d):  # sm_90a: dQ, dK / dV (+ per-sample dbias partials) as two wgmma kernels, then the fixed-order sample sum
        if N % 64 or any(t.dtype != torch.bfloat16 or not t.is_contiguous() for t in (q, k, v, do)):
            raise ValueError("Local Hopper backward requires contiguous bf16 operands and padded N divisible by 64")
        dbias = torch.empty(NH, nwin, WQ, WK, device=q.device, dtype=torch.float32)
        if HOPPER_MMA_ATTN:
            b2 = _mma_bias(bias, km, N, d)
            spc = _chunk(A, (N // 64) * NH)
            sp = (A + spc - 1) // spc
            partial = torch.empty(sp, NH, nwin, WQ, WK, device=q.device, dtype=torch.float32)
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
        maps = tuple(_tmap(t, (DM, N, A), (DM * 2, N * DM * 2), (64, 64, 1), 128) for t in (q, k, v, do))
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
    sp = max(1, min(A, num_sms(d) // (nq * NH)))  # split the samples when there are few chunks
    spe = _edge_split(nq * NH * sp, sp, A, d, groups=slide_ngroups(nwin)) if slide else 0
    if slide:
        km = _slide_mask(km)
    _load("dq", d)(
        ((nq * sp + slide_ngroups(nwin) * spe) * NH, 1, 1),
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
        int(spe),
        int(bool(qblock)),
        q,  # the raw operands and dQ (unused since the tail runs as edge groups on the tensor cores, slide.cuh)
        k,
        v,
        do,
        lse,
        dd,
        dkv,
        int(ldd),
    )
    nk = (N + 112 + 127) // 128
    sp = 2 if 2 * nk * NH <= num_sms(d) and A > 1 else 1  # two sample halves add into dbias exactly (0 + a + b)
    spe = _edge_split(nk * NH * sp, sp, A, d, cap=2, groups=slide_ngroups(nwin)) if slide else 0  # (<= 2: the moved windows' dbias adds exactly too)
    spart = torch.empty(slide_ngroups(nwin) * A * NH * 64 * 128 + 4, device=q.device, dtype=torch.float32) if slide else None  # (+ n_real)
    _load("dkv", d)(
        ((nk * sp + slide_ngroups(nwin) * spe) * NH, 1, 1),
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
        spart,
        int(ldd),
        int(N),
        int(A),
        int(nwin),
        int(sp),
        int(spe),
        int(bool(qblock)),
        q,  # the raw operands (unused since the tail runs as edge groups on the tensor cores, slide.cuh)
        k,
        v,
        do,
        lse,
        dd,
    )
    if slide:
        _slide_add(_load("slide_add", d), spart, dkv, ldd, DM, 2 * DM, N, A)
    return dbias


def _params(wb, gamma):
    """Wb [4, 16] and gamma [16] as the kernels read them: contiguous, bf16 or fp32 (anything else is converted to fp32)."""
    from miniworld_engine.kernels.augmented_attention.cuda.b200_atom import _hopper

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
        (min(8 * num_sms(d), (rows + 255) // 256), 1, 1),  # (8 CTAs an SM: lbias.cu)
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


def pair_bias_in_pre(z, gamma, wb, eps=1e-5):
    """``pair_bias_fwd`` made by the input-projection launch (round 3 part 12: ``b200_atom`` ``pre_*_eb`` builds, ``-DEB=1``): -> (bias,
    the launch's trailing arguments). Call it BEFORE the chain's launches (z, the parameters and the fresh bias buffer made here: the
    projection kernel's compute warps write the bias before their first tile, without a grid-dependency wait). Bit for bit local_bias_fwd."""
    z = z.contiguous()
    assert z.dtype == torch.bfloat16
    nwin = z.shape[0]
    rows = z.numel() // DP
    wb, gamma, bf = _params(wb, gamma)
    out = torch.empty(NH, nwin, WQ, WK, device=z.device, dtype=torch.float32)
    return out, (z, wb, gamma, out, int(rows), float(eps), bf)


def pair_bias_bwd(z, gamma, wb, dbias, eps=1e-5):
    """dbias [4, nwin, 32, 128] fp32 -> dz (z's shape and dtype, contiguous), dgamma [16], dWb [4, 16] (fp32)."""
    rows = z.numel() // DP
    wb, gamma, bf = _params(wb, gamma)
    dz = torch.empty(z.shape, device=z.device, dtype=z.dtype)  # contiguous, like the z.contiguous() the kernel reads (a view-able dpair)
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


def sample_sum(x, A, out=None):
    """x [A n, cols] bf16 (unit column stride, any row stride; cols a multiple of 8) -> [n, cols] bf16: the sum over the A samples of each
    row, in the samples' order (fp32 sums; deterministic)."""
    M, cols = x.shape
    n = M // A
    assert x.dtype == torch.bfloat16 and x.stride(1) == 1 and cols % 8 == 0 and n * A == M
    out = torch.empty(n, cols, device=x.device, dtype=x.dtype) if out is None else out
    d = _dev(x)
    total = n * cols // 8
    _load("ssum", d)(
        (max(1, min(8 * num_sms(d), (total + 255) // 256)), 1, 1),
        (256, 1, 1),
        x,
        int(x.stride(0)),
        out,
        int(out.stride(0)),
        int(A),
        int(n),
        int(cols),
    )
    return out


class TableSums:
    """The shared conditioning's table gradient summed over the samples inside the backward row kernels (``sm100_atom/tsum.cuh``): one
    fp32 partial buffer per kernel (``tr`` [n / 16, J, 16, 128], ``post`` [.., 384], ``pre`` [.., 256, or 512 with the K / V modulation]),
    then :meth:`finish` adds the partials of each table tile in slot order into the bf16 gradient [n, 1024] = [dmod | dmkv]."""

    def __init__(self, A, n, cross, device):
        ntile = A * n // 16
        d = _dev_index(device)
        self.A, self.n, self.ntile = A, n, ntile
        self.U = 2 * min(num_sms(d), ntile)  # the row kernels' grid: min(SMs, tiles) CTAs, two warpgroups each
        self.J = self.U // (n // 16) + 2  # slots per table tile (an upper bound)
        mk = lambda c: torch.empty(n // 16, self.J, 16, c, device=device, dtype=torch.float32)
        self.tr, self.post, self.pre = mk(128), mk(384), mk(512 if cross else 256)
        self.cross = cross

    def finish(self, out=None):
        n = self.n
        out = torch.empty(n, 1024, device=self.tr.device, dtype=torch.bfloat16) if out is None else out  # (no K / V: 768.. unwritten)
        d = _dev(self.tr)
        total = n * self.pre.shape[-1] // 8
        _load("tsum", d)(
            (max(1, min(4 * num_sms(d), (total + 255) // 256)), 3, 1),
            (256, 1, 1),
            self.tr,
            128,
            128,
            640,
            640,
            self.post,
            384,
            384,
            256,
            256,
            self.pre,
            self.pre.shape[-1],
            256,
            0,
            768,
            out,
            int(out.stride(0)),
            int(n),
            int(self.A),
            int(self.U),
            int(self.ntile),
            int(self.J),
        )
        return out


def _dev_index(device):
    return device.index if device.index is not None else torch.cuda.current_device()


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
    from miniworld_engine.kernels.augmented_attention.cuda.b200_atom import _hopper

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
    nb = _row_grid(M, d)
    part = torch.empty(nb, DM, device=c.device, dtype=torch.float32)  # one partial row per block (lcross.cu block_reduce8)
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
    dgamma.add_(part.sum(0))  # the rows in a fixed order: deterministic
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


SOURCES_TF32 = ("lattn_fwd_tf32", "lattn_dq_tf32", "lattn_dkv_tf32", "lbias_tf32")

#: (stem, function, dynamic smem bytes) of every fp32-path kernel (the smem sizes are static_asserted in the sources)
KERNELS_TF32 = {
    "fwd32": ("lattn_fwd_tf32", "local_attn_fwd_tf32", 172288),
    "dq32": ("lattn_dq_tf32", "local_attn_dq_tf32", 227584),
    "dkv32": ("lattn_dkv_tf32", "local_attn_dkv_tf32", 209152),
    "pb_f32": ("lbias_tf32", "local_bias_fwd_f32", 0),
    "pb_f32e": ("lbias_tf32", "local_bias_fwd_f32_early", 0),  # (programmatic dependent of atom_pre_tf32: pair_bias_early)
    "pb_b32": ("lbias_tf32", "local_bias_bwd_f32", 0),
    "pb_fin32": ("lbias_tf32", "local_bias_fin_f32", 0),
    "slide_add32": ("lslide", "local_slide_add_f32", 0),
}


#: the fp32-path kernels that execute griddepcontrol.wait before any global access (programmatic dependent launch)
_PDL_TF32 = ("fwd32", "pb_f32e")


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
    return _tmap(t, (DM, N, A), (t.stride(1) * 4, t.stride(0) * 4), (DH, n, 1), swizzle, "f32")


def _out32(t):
    """TMA store map of a contiguous fp32 [A N, C] tensor viewed [A, N, C]: boxes of 32 rows x 16 columns (64 B, 64-B swizzle)."""
    A, N, C = t.shape
    return _tmap(t, (C, N, A), (C * 4, N * C * 4), (16, 32, 1), 64, "f32")


def attn_fwd32(q, k, v, bias, kmask=None, slide=False, qblock=False):
    """The fp32 path: q, k, v fp32 [A, N, 128] views (unit column stride, any row stride); bias [4, nwin, 32, 128] fp32; kmask [N] or None
    -> O [A, N, 128] fp32, LSE [A, 4, N] fp32 (log2). tcgen05 kind::tf32: a CTA per (128-query chunk, head) and sample range. ``qblock``: IntelliFold2's slide_qblock rule."""
    A, N, _ = q.shape
    nwin = nwindows(N)
    d = _dev(q)
    o = torch.empty(A, N, DM, device=q.device, dtype=torch.float32)
    lse = torch.empty(A, NH, N, device=q.device, dtype=torch.float32)
    km = _u8(kmask)
    nq = (N + 127) // 128
    ng = slide_ngroups(nwin)
    sp = max(1, min(A, num_sms(d) // (nq * NH)))
    spe = _edge_split(nq * NH * sp, sp, A, d, groups=ng) if slide else 0
    if slide:
        km = _slide_mask(km)
    _load32("fwd32", d)(
        ((nq * sp + ng * spe) * NH, 1, 1),
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
        int(spe),
        int(bool(qblock)),
        q,  # the raw operands (unused since the tail runs as edge groups on the tensor cores, slide.cuh)
        k,
        v,
        o,
        *_tail_strides(q, k, v),
    )
    return o, lse


def attn_bwd32(q, k, v, do, bias, lse, dd, dp, cols, kmask=None, slide=False, qblock=False):
    """The fp32 path's backward: dQ, dK, dV (fp32) into the column blocks ``cols`` = (q, k, v) of dp [A N, C] and dbias [4, nwin, 32, 128]
    fp32 (summed over the samples). ``slide``: AlphaFold 3's slid key window (as ``attn_bwd``). q, k, v fp32 [A, N, 128] views, do [A, N, 128] fp32, lse / dd [A, 4, N] fp32 (log2 units / D). ``qblock``: IntelliFold2's slide_qblock rule."""
    A, N, _ = q.shape
    nwin = nwindows(N)
    d = _dev(q)
    ldd = dp.stride(0)
    km = _u8(kmask)
    dbias = torch.zeros(NH, nwin, WQ, WK, device=q.device, dtype=torch.float32)
    out = _out32(dp.view(A, N, dp.shape[1]))
    qcol, kcol, vcol = cols
    ng = slide_ngroups(nwin)
    nq = (N + 127) // 128
    sp = max(1, min(A, num_sms(d) // (nq * NH)))  # split the samples when there are few chunks
    spe = _edge_split(nq * NH * sp, sp, A, d, groups=ng) if slide else 0
    if slide:
        km = _slide_mask(km)
    _load32("dq32", d)(
        ((nq * sp + ng * spe) * NH, 1, 1),
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
        int(spe),
        int(bool(qblock)),
        q,  # the raw operands and dQ (unused since the tail runs as edge groups on the tensor cores, slide.cuh)
        k,
        v,
        do,
        lse,
        dd,
        dp[:, qcol:],
        int(ldd),
        *_tail_strides(q, k, v),
        *_tail_strides(do),
    )
    nk = (N + 112 + 127) // 128
    sp = 2 if 2 * nk * NH <= num_sms(d) and A > 1 else 1  # two sample halves add into dbias exactly (0 + a + b)
    spe = _edge_split(nk * NH * sp, sp, A, d, cap=2, groups=ng) if slide else 0
    spart = torch.empty(ng * A * NH * 64 * 128 + 4, device=q.device, dtype=torch.float32) if slide else None  # (+ n_real)
    _load32("dkv32", d)(
        ((nk * sp + ng * spe) * NH, 1, 1),
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
        spart,
        int(ldd),
        int(kcol),
        int(vcol),
        int(N),
        int(A),
        int(nwin),
        int(sp),
        int(spe),
        int(bool(qblock)),
        q,  # the raw operands (unused since the tail runs as edge groups on the tensor cores, slide.cuh)
        k,
        v,
        do,
        lse,
        dd,
        *_tail_strides(q, k, v),
        *_tail_strides(do),
    )
    if slide:
        _slide_add(_load32("slide_add32", d), spart, dp, ldd, kcol, vcol, N, A)
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


def pair_bias_early32(z, gamma, wb, eps=1e-5):
    """The fp32 inference pair bias as a programmatic dependent of ``atom_pre_tf32`` (``local_bias_fwd_f32_early``, round 3 part 8): ->
    (bias, launch). Call it BEFORE the launches of the chain (z, the parameters and the bias buffer are made here, so the kernel's writes
    before its grid-dependency wait touch nothing a running kernel reads), and ``launch()`` right after ``pre32`` (whose dependents start
    after its own wait: everything before it is complete then); the attention follows as the bias kernel's dependent."""
    z = z.contiguous()
    nwin = z.shape[0]
    rows = z.numel() // DP
    wb, gamma, bf = _params(wb, gamma)
    out = torch.empty(NH, nwin, WQ, WK, device=z.device, dtype=torch.float32)
    d = _dev(z)
    k = _load32("pb_f32e", d)
    grid = (min(4 * num_sms(d), (rows + 255) // 256), 1, 1)

    def launch():
        k(grid, (256, 1, 1), z, wb, gamma, out, int(rows), float(eps), bf)

    return out, launch


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
    _load32("pb_fin32", d)((1, 1, 1), (64, 1, 1), part, int(nb), wb, gamma, dgamma, dwb, bf)
    return dz, dgamma, dwb


# ------------------------------------------------------------------------------------------------ sm_90a mma.sync attention (lattn90*.cu)
