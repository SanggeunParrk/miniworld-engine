"""A100 (sm_80) builds of the atom DiT-variant training path (``train.py``: ``atom_fwd`` / ``atom_bwd`` and the pair-bias / edge
Functions): plain CUDA rows (``train_rows.cu``, ``edge.cu``) and the mma.sync window attention of
``augmented_attention/cuda/sm100_atom_local/lattn90*.cu``. None of those use an sm_90 feature but programmatic dependent launch, so the
A100 builds are the same sources:

* ``train_rows.cu`` / ``edge.cu``: torch extensions for sm_80 (their own names; ``edge.cu``'s sm_90a build serves local_dit_h100);
* ``lattn90.cu`` / ``lattn90_tf32.cu``: cubins for sm_80 of the source text with the two ``griddepcontrol`` statements (``pdl_wait`` /
  ``pdl_trigger``) emptied -- without a dependent launch they have nothing to wait for -- so there is one source of these kernels;
* the two small reductions the H100 wrappers take from sm_90a-only files: the per-sample dbias sum (``local_bias_samples_sm90``) as a
  fixed-order torch sum, and the pair-bias weight finish (``local_bias_fin``) from ``sm80_atom_local/lbias.cu``.

:data:`LOCAL` has the five wrapper functions ``train.py`` calls on ``sm100_atom_local`` (same arguments, same results).
"""

from __future__ import annotations

import functools
from pathlib import Path
from types import SimpleNamespace

import torch

_dir = Path(__file__).parent
_LOCAL_DIR = _dir.parents[1] / "augmented_attention" / "cuda" / "sm100_atom_local"
NH, DP, WQ, WK = 4, 16, 32, 128
_PDL = ('asm volatile("griddepcontrol.wait;" ::: "memory");', 'asm volatile("griddepcontrol.launch_dependents;" ::: "memory");')


def _extension(name: str, source: str):
    from miniworld_engine.kernels._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name=name,
        sources=[str(_dir / source)],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            *gencodes("80"),
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


@functools.lru_cache(maxsize=None)
def train_ext():
    """``train_rows.cu`` for sm_80."""
    return _extension("dit_variants_train_rows_sm80", "train_rows.cu")


@functools.lru_cache(maxsize=None)
def edge_ext():
    """``edge.cu`` for sm_80."""
    return _extension("dit_variants_edge_sm80", "edge.cu")


@functools.lru_cache(maxsize=None)
def cubin(stem: str) -> str:
    """``sm100_atom_local/<stem>.cu`` (lattn90 / lattn90_tf32) for sm_80, its dependent-launch statements emptied."""
    from miniworld_engine.kernels import _nvcc

    text = (_LOCAL_DIR / f"{stem}.cu").read_text()
    for stmt in _PDL:
        if stmt not in text:
            raise RuntimeError(f"{stem}.cu: {stmt!r} not found (the sm_80 build empties it)")
        text = text.replace(stmt, "")
    flags = ("-std=c++17", "-O3", "-arch=sm_80", "-cubin", "-lineinfo", f"-I{_LOCAL_DIR}")
    return str(_nvcc.cubin(f"atom_local_mma_sm80/{stem}", flags, text=text))


#: name -> (stem, function, dynamic shared memory): the lattn90*.cu kernels of the H100 wrappers (``sm100_atom_local._MMA_KERNELS*``)
_KERNELS = {
    "h_fwd": ("lattn90", "h_attn_fwd", 64 * 136 * 4 + 2 * (64 + 160 + 160) * 40 * 2),
    "h_dq": ("lattn90", "h_attn_dq", 64 * 136 * 4 + (64 * 2 + 160 * 2) * 40 * 2 + 2 * 64 * 4),
    "h_dkv": ("lattn90", "h_attn_dkv", 4 * 16 * 132 * 4 + (192 * 2 + 64 * 2) * 40 * 2 + 2 * 192 * 4),
    "h_pb": ("lattn90", "h_pair_bias", 0),
    "h_pbb": ("lattn90", "h_pair_bias_bwd", 0),
    "h_fwd32": ("lattn90_tf32", "h_attn_fwd_tf32", 90112),
    "h_dq32": ("lattn90_tf32", "h_attn_dq_tf32", 64 * 136 * 4 + (64 * 2 + 160 * 2) * 36 * 4 + 2 * 64 * 4),
    "h_dkv32": ("lattn90_tf32", "h_attn_dkv_tf32", 4 * 16 * 132 * 4 + (192 * 2 + 64 * 2) * 36 * 4 + 2 * 192 * 4),
}


@functools.lru_cache(maxsize=None)
def _load(name: str, device_index: int):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    stem, func, smem = _KERNELS[name]
    with torch.cuda.device(device_index):
        return driver.Kernel(cubin(stem), func, smem)


def ready(device_index: int) -> None:
    """Build / load every sm_80 kernel of the path on ``device_index`` (the prebuild plans the same calls)."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm80_atom_local

    train_ext()
    edge_ext()
    for name in _KERNELS:
        _load(name, device_index)
    sm80_atom_local._load("pb_fin", device_index)


def _dev(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


def _sms(d: int) -> int:
    return torch.cuda.get_device_properties(d).multi_processor_count


def _chunk(A: int, tiles: int, sms: int) -> int:
    """``sm100_atom_local._chunk`` with the card's SM count: samples per CTA (16, shrunk until the grid has about one CTA per SM)."""
    want = 16
    while want > 1 and tiles * -(-A // want) < sms:
        want = (want + 1) // 2
    return -(-A // -(-A // want))


def _u8(kmask):
    if kmask is None:
        return None
    if kmask.dtype == torch.bool and kmask.is_contiguous():
        return kmask.view(torch.uint8)
    return kmask.to(torch.uint8).contiguous()


def pair_bias_mma(z, gamma, wb, kmask, n, eps=1e-5):
    """z [nwin, 32, 128, 16] bf16 / fp32 -> bias2 [4, nwin, 32, 128] fp32: LN(z) gamma . Wb log2(e), -inf at invalid / masked keys."""
    d = _dev(z)
    rows = z.numel() // DP
    out = torch.empty(NH, z.shape[0], WQ, WK, device=z.device, dtype=torch.float32)
    _load("h_pb", d)(
        (min(8 * _sms(d), (rows + 255) // 256), 1, 1),
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
    """TF32 window attention forward: q, k, v [A, N, 128] fp32 (tf32-rounded) -> O [A, N, 128] fp32, LSE [A, 4, N] (log2)."""
    A, N, _ = q.shape
    d = _dev(q)
    o = torch.empty_like(q)
    lse = torch.empty(A, NH, N, device=q.device, dtype=torch.float32)
    spc = _chunk(A, (N // 64) * NH, _sms(d))
    _load("h_fwd32", d)((N // 64, NH, (A + spc - 1) // spc), (128, 1, 1), q, k, v, bias2, o, lse, int(N), int(A), int(spc))
    return o, lse


def attn_fwd_mma(q, k, v, bias2):
    """bf16 window attention forward: q, k, v [A, N, 128] bf16 -> O bf16, LSE [A, 4, N] (log2)."""
    A, N, _ = q.shape
    d = _dev(q)
    o = torch.empty_like(q)
    lse = torch.empty(A, NH, N, device=q.device, dtype=torch.float32)
    spc = _chunk(A, (N // 64) * NH, _sms(d))
    _load("h_fwd", d)((N // 64, NH, (A + spc - 1) // spc), (128, 1, 1), q, k, v, bias2, o, lse, int(N), int(A), int(spc), 1)
    return o, lse


@functools.lru_cache(maxsize=16)
def _domain(n: int, device: torch.device) -> torch.Tensor:
    """[1, nwin, 1, 128] bool: the key slot 32 w - 48 + j of window w is inside [0, N) (the slots the key-centric kernel writes)."""
    nwin = n // WQ
    key = WQ * torch.arange(nwin, device=device)[:, None] - 48 + torch.arange(WK, device=device)
    return ((key >= 0) & (key < n)).view(1, nwin, 1, WK)


def attn_bwd_mma(q, k, v, do, bias2, lse, dd, dkv, tf32: bool):
    """dQ, dK, dV into dkv [A N, >= 384] (columns 0 / 128 / 256 of its base) and dbias [4, nwin, 32, 128] fp32 summed over the samples
    (the per-chunk partials summed in order; slots outside [0, N) are 0)."""
    A, N, _ = q.shape
    d = _dev(q)
    ldd = dkv.stride(0)
    spc = _chunk(A, (N // 64) * NH, _sms(d))
    sp = (A + spc - 1) // spc
    partial = torch.empty(sp, NH, N // WQ, WQ, WK, device=q.device, dtype=torch.float32)
    grid = (N // 64, NH, sp)
    sfx = "32" if tf32 else ""
    _load("h_dq" + sfx, d)(grid, (128, 1, 1), q, k, v, do, bias2, lse, dd, dkv, int(ldd), int(N), int(A), int(spc))
    _load("h_dkv" + sfx, d)(grid, (128, 1, 1), q, k, v, do, bias2, lse, dd, dkv, partial, int(ldd), int(N), int(A), int(spc))
    dbias = partial[0] if sp == 1 else partial.sum(0)
    return torch.where(_domain(N, q.device), dbias, 0.0)


def pair_bias_bwd_mma(z, gamma, wb, dbias, eps=1e-5):
    """dbias [4, nwin, 32, 128] fp32 -> dz (z's shape / dtype), dgamma [16], dWb [4, 16] (fp32)."""
    from miniworld_engine.kernels.augmented_attention.cuda import sm80_atom_local

    d = _dev(z)
    rows = z.numel() // DP
    z = z.contiguous()
    dz = torch.empty_like(z)
    nb = min(_sms(d), (rows + 255) // 256)
    part = torch.empty(nb, NH * DP, device=z.device, dtype=torch.float32)
    dgamma = torch.empty(DP, device=z.device, dtype=torch.float32)
    dwb = torch.empty(NH, DP, device=z.device, dtype=torch.float32)
    wbf, gf = wb.float().contiguous(), gamma.float().contiguous()
    _load("h_pbb", d)(
        (nb, 1, 1), (256, 1, 1), z, wbf, gf, dbias.contiguous(), dz, part, int(rows), float(eps), int(z.dtype == torch.bfloat16)
    )
    sm80_atom_local._load("pb_fin", d)((1, 1, 1), (64, 1, 1), part, int(nb), wbf, gf, dgamma, dwb, 0)
    return dz, dgamma, dwb


#: what ``train.py``'s atom path calls on ``sm100_atom_local``
LOCAL = SimpleNamespace(
    pair_bias_mma=pair_bias_mma,
    attn_fwd_mma=attn_fwd_mma,
    attn_fwd_mma32=attn_fwd_mma32,
    attn_bwd_mma=attn_bwd_mma,
    pair_bias_bwd_mma=pair_bias_bwd_mma,
)
