"""A100 AF3 32-query/128-key CUDA, ported from B200's mma.sync implementation.

Uses direct token loads, shared pair bias, and key-centric backward without window gathers.
The block dispatches here after full-module accuracy and paired performance validation.
"""

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels import _nvcc

NH, DH, DM, DP, WQ, WK = 4, 32, 128, 16, 32, 128
PREFETCH = False  # L2 prefetch of the next sample pair's rows in the attention kernels: measured slower (A = 48, N = 4096: fwd 208 -> 294 us, dq +11 %, dkv +17 %), so off; True and clear the caches of ``cubin`` / ``_load`` to repeat
_dir = Path(__file__).parent
KERNELS = {
    "fwd": ("lattn_fwd", "local_attn_fwd", 54272),
    "dq": ("lattn_dq", "local_attn_dq", 17408 + 2 * (2 * 2048 + 2 * 8192)),
    "dkv": ("lattn_dkv", "local_attn_dkv", 67072),
    "pb_f": ("lbias", "local_bias_fwd", 0),
    "pb_b": ("lbias", "local_bias_bwd", 0),
    "pb_fin": ("lbias", "local_bias_fin", 0),
    "delta": ("glue", "local_attn_delta", 0),
    "dbsum": ("glue", "local_dbias_sum", 0),
    "gate_f": ("glue", "local_gate_fwd", 0),
    "gate_b": ("glue", "local_gate_bwd", 0),
}


@functools.lru_cache(maxsize=None)
def cubin(stem):
    """The cubin of ``<stem>.cu`` (sm_80), through ``_nvcc.cubin``: the packaged prebuilt kernels, the content-keyed cache, else built
    (``kernels/_prebuild.py`` plans it)."""
    flags = ["-std=c++17", "-O3", "-arch=sm_80", "-cubin", "-lineinfo", f"-DATT_PF={int(PREFETCH)}", f"-I{_dir}"]
    return str(_nvcc.cubin(f"atom_local_sm80/{stem}", flags, source=_dir / f"{stem}.cu"))


@functools.lru_cache(maxsize=None)
def _load(name, device):
    # Kernel is a general CUDA driver launcher; this path never creates a B200 TensorMap.
    from miniworld_engine.kernels.augmented_attention.cuda.sm100.driver import Kernel

    stem, function, smem = KERNELS[name]
    with torch.cuda.device(device):
        return Kernel(cubin(stem), function, smem)


def _dev(t):
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


def _nsm(t):
    return torch.cuda.get_device_properties(_dev(t)).multi_processor_count


def ready(device):
    for name in KERNELS:
        _load(name, device)


def _split(samples, windows, slots, bias_cost=0.5):
    """Sample-pair chunks per (window, head): the grid of windows x 4 x split CTAs runs in whole waves of ``slots`` resident CTAs.
    Cost of a split = waves x (pairs per CTA + the bias load, in units of one pair)."""
    pairs = (samples + 1) // 2
    best, best_cost = 1, None
    for split in range(1, pairs + 1):
        waves = -(-windows * NH * split // slots)
        cost = waves * (-(-pairs // split) + bias_cost)
        if best_cost is None or cost < best_cost:
            best, best_cost = split, cost
    return best


def _split_samples(samples, ctas, slots, overhead=2.0):
    """Sample chunks of a grid of ``ctas`` CTAs per chunk, ``slots`` resident at a time: waves x (samples per CTA + setup, in samples)."""
    best, best_cost = 1, None
    for split in range(1, samples + 1):
        cost = -(-ctas * split // slots) * (-(-samples // split) + overhead)
        if best_cost is None or cost < best_cost:
            best, best_cost = split, cost
    return best


def attn_fwd(q, k, v, bias, mask=None, split=None):
    samples, n, _ = q.shape
    windows = (n + WQ - 1) // WQ
    out = torch.empty_like(q)
    lse = torch.empty(samples, NH, n, device=q.device, dtype=torch.float32)
    if split is None:
        split = 1  # measured (A = 1 .. 48, N = 2048 .. 8192): one chunk of samples per (window, head) is the fastest; more only repeat the bias load
    _load("fwd", _dev(q))(
        (windows, NH, split),
        (128, 1, 1),
        q,
        k,
        v,
        bias,
        mask,
        out,
        lse,
        n,
        samples,
        windows,
        split,
    )
    return out, lse


def attn_bwd(
    q,
    k,
    v,
    dout,
    bias,
    out,
    lse,
    mask=None,
    *,
    packed_output=False,
    delta=None,
    packed=None,
    split_dq=None,
    split_kv=None,
):
    """dQ / dK / dV and dbias of the window attention. ``delta`` [A, 4, N] fp32 = the row term D (made here from O and dO when None). ``packed`` is the caller's [rows, 4 DM] buffer:
    dq goes to its columns 0 .., dk to 2 DM .., dv to 3 DM .. (the columns DM .. 2 DM are the caller's) and only dbias is returned; otherwise the buffer is made here (dq | dk | dv in its
    columns 0 .., DM .., 2 DM ..) and returned (``packed_output``) or split into views."""
    samples, n, _ = q.shape
    windows = (n + WQ - 1) // WQ
    external = packed is not None
    if packed is None:
        packed = torch.empty(samples * n, 4 * DM, device=q.device, dtype=q.dtype)
    dkv_base = packed[:, DM:] if external else packed
    nsm = _nsm(q)
    units = (n + 15) // 16
    ctas_kv = (units + 3) // 4
    if split_dq is None:
        split_dq = _split(samples, windows, 2 * nsm)
    if split_kv is None:
        split_kv = _split_samples(samples, ctas_kv * NH, 2 * nsm)
    dbp = torch.empty(
        split_dq, NH, windows, WQ, WK, device=q.device, dtype=torch.float32
    )
    device = _dev(q)
    if delta is None:
        delta = torch.empty_like(lse)
        _load("delta", device)(
            ((samples * n * 16 + 255) // 256, 1, 1),
            (256, 1, 1),
            out,
            dout,
            delta,
            n,
            samples,
        )
    _load("dq", device)(
        (windows, NH, split_dq),
        (128, 1, 1),
        q,
        k,
        v,
        dout,
        bias,
        mask,
        lse,
        delta,
        packed,
        dbp,
        4 * DM,
        n,
        samples,
        windows,
        split_dq,
    )
    _load("dkv", device)(
        (ctas_kv, NH, split_kv),
        (128, 1, 1),
        q,
        k,
        v,
        dout,
        bias,
        mask,
        lse,
        delta,
        dkv_base,
        4 * DM,
        n,
        samples,
        windows,
        split_kv,
    )
    if split_dq == 1:
        dbias = dbp[0]
    else:
        dbias = torch.empty_like(bias)
        total = dbias.numel()
        _load("dbsum", device)(
            ((total // 4 + 255) // 256, 1, 1), (256, 1, 1), dbp, dbias, split_dq, total
        )
    if external:
        return dbias
    if packed_output:
        return packed, dbias
    return (
        packed[:, 0:DM].view(samples, n, DM),
        packed[:, DM : 2 * DM].view(samples, n, DM),
        packed[:, 2 * DM : 3 * DM].view(samples, n, DM),
        dbias,
    )


def gate_fwd(gate, x):
    out = torch.empty_like(x)
    sigmoid = torch.empty_like(x)
    _load("gate_f", _dev(x))(
        ((x.numel() + 255) // 256, 1, 1), (256, 1, 1), gate, x, out, sigmoid, x.numel()
    )
    return out, sigmoid


def gate_bwd(dy, sigmoid, x):
    dgate = torch.empty_like(x)
    dx = torch.empty_like(x)
    _load("gate_b", _dev(x))(
        ((x.numel() + 255) // 256, 1, 1),
        (256, 1, 1),
        dy,
        sigmoid,
        x,
        dgate,
        dx,
        x.numel(),
    )
    return dgate, dx


def _params(wb, gamma):
    if wb.dtype not in (torch.bfloat16, torch.float32) or gamma.dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("BF16 or FP32 parameters required")
    if not wb.is_contiguous() or not gamma.is_contiguous():
        raise ValueError("contiguous parameters required")
    return int(wb.dtype == torch.bfloat16) + 2 * int(gamma.dtype == torch.bfloat16)


def pair_bias_fwd(z, gamma, wb, eps=1e-5):
    rows = z.numel() // DP
    bias = torch.empty(NH, z.shape[0], WQ, WK, device=z.device, dtype=torch.float32)
    _load("pb_f", _dev(z))(
        (min(4 * _nsm(z), (rows + 255) // 256), 1, 1),
        (256, 1, 1),
        z,
        wb,
        gamma,
        bias,
        rows,
        float(eps),
        _params(wb, gamma),
    )
    return bias


def pair_bias_bwd(z, gamma, wb, dbias, eps=1e-5):
    rows = z.numel() // DP
    blocks = min(_nsm(z), (rows + 255) // 256)
    part = torch.empty(blocks, NH * DP, device=z.device, dtype=torch.float32)
    dz = torch.empty_like(z)
    dgamma = torch.empty(DP, device=z.device, dtype=torch.float32)
    dwb = torch.empty(NH, DP, device=z.device, dtype=torch.float32)
    flags = _params(wb, gamma)
    device = _dev(z)
    _load("pb_b", device)(
        (blocks, 1, 1),
        (256, 1, 1),
        z,
        wb,
        gamma,
        dbias,
        dz,
        part,
        rows,
        float(eps),
        flags,
    )
    _load("pb_fin", device)(
        (1, 1, 1), (64, 1, 1), part, blocks, wb, gamma, dgamma, dwb, flags
    )
    return dz, dgamma, dwb
