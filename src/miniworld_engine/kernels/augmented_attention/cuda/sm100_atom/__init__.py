"""sm_100a (B200) kernels of the AF3-style atom DiT block (``modules/dit.DiTBlock`` at atom widths), forward and backward.

    a = a + sigmoid(to_scale(s)) * to_out(sigmoid(g) * softmax(q k^T / sqrt 32 + bias) v)     bias = LN(z) Wb^T [4, N, N]
    a = a + ConditionedTransition(a, s)                                                          SwiGLU 128 -> 256 -> 128

d_single = d_cond = 128, d_pair = 16, 4 heads x 32, B = 1, N a multiple of 128; the A samples share one pair tensor and one
key mask, which the pair-bias producer folds into the bias (masked keys -1e4), so the attention kernels need no mask of
their own; the caller pads the atoms to a multiple of 128 and masks the padding. Developed in ``experiments/atomdit_sm100`` (rounds v1-v2); page: ``docs/gpus/b200/atom_dit/atom_dit.md``.

The kernels (``sm100.cuh`` holds the tcgen05 / TMA / mbarrier helpers):
  pair_bias.cu    bias[h, i, j] = LN(z[i, j]) . Wb[h] in both layouts the attention reads (head-major and transposed), with
                  the key mask and the padding folded in, and its backward (dz, partial d(gamma * Wb)) from dbias.
  cond_fwd.cu     the six conditioning projections -> mod [M, 768] = s1 | bi1 | so | s2 | bi2 | st (gates as rn(sigmoid)).
  pre_fwd.cu      AdaLN 1 + the q / k / v / gate projections.
  attn_fwd.cu     O = softmax(q k^T / sqrt 32 + bias) v and the row LSE (log2 units).
  post_fwd.cu     gate, out projection, residual, AdaLN 2, SwiGLU, squeeze, residual (+ the activations the backward reads).
  tr_bwd_gate.cu  the transition's gate backward and SwiGLU backward.
  post_bwd.cu     AdaLN 2 + LN backward, residual, output gate, the out projection's dgrad, the attention gate; dO and D.
  attn_dd.cu      D = rowsum(dO O) per head (the inference-free backward's helper).
  attn_dkv.cu, attn_dq.cu, attn_dbias.cu
                  dK / dV, dQ (bf16, straight into the dP buffer the input projections' dgrad reads), dbias (fp32, summed
                  over the samples).
  pre_bwd.cu      the input projections' dgrad, AdaLN 1 + LN backward -> d single.
  cond_bwd.cu     the conditioning projections' dgrad and the two cond LayerNorm backwards -> d cond.
The weight gradients are cuBLAS GEMMs on the activations these kernels write.

The cubins are built on first use by the newest nvcc here that knows sm_100a (``transition.cuda.fused_sm100a.kernel_toolchain``:
the experiment's measurements are 13.1 builds; 12.9's ptxas gives slightly longer code) and cached under
``MINIWORLD_ENGINE_JIT_ROOT`` keyed by the sources and the flags. They launch through the CUDA driver
(``augmented_attention.cuda.sm100.driver``) on torch's current stream, CUDA-graph capturable.
"""

from __future__ import annotations

import functools
import hashlib
import os
import subprocess
from pathlib import Path

import torch

NH, DH, DM, DP = 4, 32, 128, 16
_dir = Path(__file__).parent
_SMEM = 232448
SOURCES = ("attn_fwd", "attn_dkv", "attn_dq", "attn_dbias", "attn_dd", "pair_bias", "cond_fwd", "pre_fwd", "post_fwd",
           "tr_bwd_gate", "post_bwd", "pre_bwd", "cond_bwd")


# --------------------------------------------------------------------------------------------------- build
@functools.lru_cache(maxsize=None)
def cubin(stem: str, hopper: bool = False) -> str:
    """Path of ``<stem>.cu`` built for sm_100a with the sources' default options; rebuilt only when a source or a flag changes."""
    from miniworld_engine.kernels.transition.cuda.fused_sm100a import kernel_toolchain

    if hopper:
        from miniworld_engine.kernels._nvcc import ensure_cuda_home, host_flags, nvcc_path
        ensure_cuda_home()
        nvcc, rel, host = nvcc_path(), (12, 9), host_flags()
        if not nvcc or tuple(map(int, torch.version.cuda.split(".")[:2])) > (12, 9):
            raise RuntimeError("Atom Hopper kernels require the matching CUDA <= 12.9 toolchain")
    else:
        nvcc, rel, host = kernel_toolchain()
    flags = (*host, "-std=c++17", "-O3", "-arch=sm_90a" if hopper else "-arch=sm_100a",
             *(("-DATOM_HOPPER=1",) if hopper else ()),
             *(("-DATOM_HOPPER_TRUE_MASK=1",) if hopper and stem == "pair_bias"
               and (os.environ.get("MINIWORLD_ATOM_DIT_H100_TRUE_MASK", "0") == "1"
                    or os.environ.get("MINIWORLD_ATOM_DIT_H100_EMPTY_UNIFORM", "0") == "1") else ()),
             "-cubin", "-lineinfo", f"-I{_dir}")
    h = hashlib.sha256(" ".join((nvcc, str(rel), *flags)).encode())
    for f in (_dir / "sm100.cuh", _dir / f"{stem}.cu"):
        h.update(f.read_bytes())
    if hopper:
        h.update((_dir.parents[2] / "swa_dit" / "cuda" / "gen" / "wgmma_bf16.inc").read_bytes())
    root = Path(os.environ.get("MINIWORLD_ENGINE_JIT_ROOT", Path.home() / ".cache" / "miniworld_engine_jit"))
    out = root / ("atom_dit_sm90" if hopper else "atom_dit_sm100") / f"{stem}_{h.hexdigest()[:16]}.cubin"
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


@functools.lru_cache(maxsize=None)
def _load_atom_kernel(stem: str, func: str, device_index: int, smem: int = _SMEM, pdl: bool = False):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    with torch.cuda.device(device_index):
        return driver.Kernel(cubin(stem, _hopper(device_index)), func, smem, pdl=pdl)


#: (stem, function, dynamic smem, PDL) of every kernel the block launches
KERNELS = {
    "fwd": ("attn_fwd", "augattn_fwd2_sm100", _SMEM, False),
    "dkv": ("attn_dkv", "augattn_dkv_sm100", _SMEM, False),
    "dq": ("attn_dq", "atom_dq_sm100", _SMEM, False),
    "dbias": ("attn_dbias", "atom_dbias_sm100", _SMEM, False),
    "dd": ("attn_dd", "atom_dd", 0, False),
    "pb_f": ("pair_bias", "pair_bias_fwd", 0, False),
    "pb_fp": ("pair_bias", "pair_bias_fwd_pad", 0, False),
    "pb_b": ("pair_bias", "pair_bias_bwd", 0, False),
    "cond": ("cond_fwd", "atom_cond_fwd_sm100", _SMEM, True),
    "pre": ("pre_fwd", "atom_pre_fwd_sm100", _SMEM, True),
    "post": ("post_fwd", "atom_post_fwd_sm100", _SMEM, True),
    "trg": ("tr_bwd_gate", "atom_tr_bwd_gate_sm100", _SMEM, True),
    "postb": ("post_bwd", "atom_post_bwd_sm100", _SMEM, True),
    "preb": ("pre_bwd", "atom_pre_bwd_sm100", _SMEM, True),
    "condb": ("cond_bwd", "atom_cond_bwd_sm100", _SMEM, True),
}


@functools.lru_cache(maxsize=8)
def _hopper(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) == (9, 0)


_HOPPER_KERNELS = {
    "cond": ("cond_fwd", "atom_cond_fwd_sm90", 131136, False),
    "cond1": ("cond_fwd", "atom_cond_fwd_single_sm90", 114752, False),
    "pre": ("pre_fwd", "atom_pre_fwd_sm90", 180288, False),
    "pre_cross": ("pre_fwd", "atom_pre_cross_fwd_sm90", 114752, False),
    "post": ("post_fwd", "atom_post_fwd_sm90", 147520, False),
    "post2": ("post_fwd", "atom_post_fwd_2stage_sm90", 114752, False),
    "pack": ("cond_fwd", "atom_pack_sm90", 0, False),
    "pack_t": ("tr_bwd_gate", "atom_pack_transpose_sm90", 0, False),
    "trg": ("tr_bwd_gate", "atom_tr_bwd_sm90", 131136, False),
    "postb": ("post_bwd", "atom_post_bwd_sm90", 98368, False),
    "preb": ("pre_bwd", "atom_pre_bwd_sm90", 98368, False),
    "condb": ("cond_bwd", "atom_cond_bwd_sm90", 168000, False),
    "grad_pack": ("pre_bwd", "atom_grad_pack_sm90", 0, False),
    "col_part": ("cond_bwd", "atom_column_partials_sm90", 0, False),
    "col_fin": ("cond_bwd", "atom_column_finish_sm90", 0, False),
    "pb_w": ("pair_bias", "atom_pair_weight_sm90", 0, False),
    "pb_fin": ("pair_bias", "atom_pair_finish_sm90", 0, False),
    "pb_part": ("pair_bias", "atom_pair_partials_sm90", 0, False),
    "db_native": ("attn_dbias", "atom_dbias_sm90", 82208, False),
    "db_tma": ("attn_dbias", "atom_dbias_tma_sm90", 82240, False),
    "db_sum": ("pair_bias", "atom_dbias_transpose_sum_sm90", 0, False),
    "dq_pack": ("pair_bias", "atom_pack_dqkv_sm90", 0, False),
    "empty_uniform": ("pair_bias", "atom_empty_uniform_sm90", 0, False),
    **{k: KERNELS[k] for k in ("pb_f", "pb_fp", "pb_b")},
}


def atom_kernel(name: str, device_index: int):
    if _hopper(device_index) and name not in _HOPPER_KERNELS:
        raise NotImplementedError(f"Atom Hopper kernel {name} is not implemented")
    stem, func, smem, pdl = (_HOPPER_KERNELS if _hopper(device_index) else KERNELS)[name]
    return _load_atom_kernel(stem, func, device_index, smem, pdl)


def load_all(device_index: int) -> None:
    """Build and load every kernel on the device (raises on any failure)."""
    for name in (_HOPPER_KERNELS if _hopper(device_index) else KERNELS):
        atom_kernel(name, device_index)


@functools.lru_cache(maxsize=8)
def nsm(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _dev(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


def empty_uniform_enabled(device_index: int) -> bool:
    return _hopper(device_index) and os.environ.get("MINIWORLD_ATOM_DIT_H100_EMPTY_UNIFORM", "0") == "1"


def empty_uniform_hopper(keys, src, dst, n, *, value_gradient=False):
    """Live-mask uniform-mean epilogue; src is dense [A*N,128] BF16."""
    if keys is None or not empty_uniform_enabled(_dev(src)):
        return
    if (src.dtype != torch.bfloat16 or dst.dtype != torch.bfloat16
            or not src.is_contiguous() or not dst.is_contiguous()):
        raise ValueError("Atom uniform edge requires contiguous BF16 tensors")
    atom_kernel("empty_uniform", _dev(src))((4, src.shape[0] // n, 1), (256, 1, 1),
        keys, src, dst, int(n), int(512 if value_gradient else 128), int(256 if value_gradient else 0))


# --------------------------------------------------------------------------------------------------- tensor maps
def _tm(t, rows, box_rows, cols=DM, box_cols=DH, swizzle=128):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    return driver.TensorMap(t, [cols, rows], cols * 2, [box_cols, box_rows], swizzle=swizzle)


def _map(t, dims, stride, box, swizzle=128, dtype="bf16"):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    return driver.TensorMap(t, dims, stride, box, swizzle=swizzle, dtype=dtype)


def _rows(t, cols=DM):
    """Row-major bf16 [M, cols] as 32-row x 64-column boxes (128-B swizzle)."""
    return _map(t, [cols, t.shape[0]], cols * 2, [64, 32])


def _rows16(t, cols=DM):
    return _map(t, [cols, t.shape[0]], cols * 2, [64, 16])


# --------------------------------------------------------------------------------------------------- weights
def cond_params(blk):
    """The six conditioning projections stacked for cond_fwd: Wmod [768, 128] bf16 = [Wsc1; Wbi1; Wos; Wsc2; Wbi2; Wts],
    bias [768] fp32 (zero for the shifts), the two cond LayerNorm weights g1, g2 [128] fp32."""
    att, tr = blk.attention, blk.transition
    a1, a2 = att.ada_ln_in, tr.ada_ln_in
    W = torch.cat([a1.to_scale.weight, a1.to_bias.weight, att.to_scale.weight, a2.to_scale.weight, a2.to_bias.weight,
                   tr.to_scale.weight])
    z = torch.zeros(DM, device=W.device)
    b = torch.cat([a1.to_scale.bias.float(), z, att.to_scale.bias.float(), a2.to_scale.bias.float(), z, tr.to_scale.bias.float()])
    return (W.to(torch.bfloat16).contiguous(), b.contiguous(), a1.ln_cond.weight.float().contiguous(),
            a2.ln_cond.weight.float().contiguous())


def pre_params(blk):
    """The attention's four input projections stacked for pre_fwd: W [512, 128] bf16 = [Wq; Wk; Wv; Wg], bq [128] fp32."""
    att = blk.attention
    W = torch.cat([att.to_query.weight, att.to_key.weight, att.to_value.weight, att.to_gate.weight])
    return W.to(torch.bfloat16).contiguous(), att.to_query.bias.float().contiguous()


def post_params(blk):
    """post_fwd's weights: Wo [128, 128], Wu = [Wa; Wb] [512, 128], Ws [128, 256] (bf16)."""
    att, tr = blk.attention, blk.transition
    def bf(t):
        return t.to(torch.bfloat16).contiguous()
    return bf(att.to_out.weight), bf(torch.cat([tr.expand_a.weight, tr.expand_b.weight])), bf(tr.squeeze.weight)


# --------------------------------------------------------------------------------------------------- launches
def pair_bias_fwd(z, gamma, wb, eps=1e-5, trans=True, n=None, kv=None):
    """z [1, NZ, NZ, 16] bf16 -> bias [4, n, n] bf16 (head-major) and, for the backward, bias^T [4, n(key), n(query)].
    n (default NZ) is the attention's length, a multiple of 128 >= NZ; kv [NZ] bool marks the valid keys (None: all of them).
    Masked and padded keys get -1e4 (bf16 -9984), padded query rows 0."""
    z = z.reshape(z.shape[-3], z.shape[-2], DP)
    NZ = z.shape[0]
    N = NZ if n is None else n
    w = pair_weight_hopper(wb, gamma) if _hopper(_dev(z)) else (wb.float() * gamma.float()[None]).contiguous()
    bo = torch.empty(NH, N, N, device=z.device, dtype=torch.bfloat16)
    bt = torch.empty_like(bo) if trans else None
    atom_kernel("pb_f" if N == NZ else "pb_fp", _dev(z))((N // 64, N // 32, 1), (256, 1, 1), z, w, bo, bt, int(N), int(NZ), kv,
                                                        float(eps))
    return bo, bt


def pair_bias_bwd(z, gamma, wb, dbias, eps=1e-5, kv=None, *, native_hopper=False):
    """dbias [4, n, n] fp32 (summed over the samples) -> dz (z's dtype and layout), dgamma [16], dWb [4, 16] (fp32); dbias on
    keys kv marks invalid is dropped."""
    z = z.reshape(z.shape[-3], z.shape[-2], DP)
    NZ, N = z.shape[0], dbias.shape[-1]
    w = pair_weight_hopper(wb, gamma) if _hopper(_dev(z)) else (wb.float() * gamma.float()[None]).contiguous()
    dz = torch.empty_like(z)
    nb = (N // 64) * (N // 32)
    pw = torch.empty(nb, NH, DP, device=z.device, dtype=torch.float32)
    atom_kernel("pb_b", _dev(z))((N // 64, N // 32, 1), (256, 1, 1), z, w, dbias.contiguous(), dz, pw, int(N), int(NZ), kv,
                                 float(eps))
    if _hopper(_dev(z)):
        dg = torch.empty(DP, device=z.device, dtype=torch.float32)
        dw = torch.empty(NH, DP, device=z.device, dtype=torch.float32)
        if native_hopper or os.environ.get("MINIWORLD_ATOM_DIT_H100_BWD_V2", "0") == "1":
            part = torch.empty(16, NH, DP, device=z.device, dtype=torch.float32)
            atom_kernel("pb_part", _dev(z))((4, 16, 1), (256, 1, 1), pw, part, int(nb))
            pw, nb = part, 16
        bfmask = int(wb.dtype == torch.bfloat16) | (int(gamma.dtype == torch.bfloat16) << 1)
        atom_kernel("pb_fin", _dev(z))((1, 1, 1), (64, 1, 1), pw, wb, gamma, dg, dw, int(nb), bfmask)
        return dz, dg, dw
    dw = pw.sum(0)
    return dz, (dw * wb.float()).sum(0), dw * gamma.float()[None]


def attn_fwd(q, k, v, bias):
    """q, k, v [A, N, 128] bf16 (4 heads x 32), bias [4, N, N] bf16 -> O [A N, 128] bf16, LSE [A, 4, N] fp32 (log2)."""
    A, N, _ = q.shape
    q2, k2, v2 = (t.reshape(A * N, DM) for t in (q, k, v))
    if _hopper(_dev(q)):
        from miniworld_engine.kernels.augmented_attention import cuda as core
        return core.natural_forward(q2, k2, v2, bias, heads=NH, head_dim=DH, output_bf16=True)
    O = torch.empty(A * N, DM, device=q.device, dtype=torch.bfloat16)
    LSE = torch.empty(A, NH, N, device=q.device, dtype=torch.float32)
    bsrc = bias.contiguous()
    maps = (_tm(q2, A * N, 128, swizzle=64), _tm(k2, A * N, 64, swizzle=64), _tm(v2, A * N, 64, swizzle=64),
            _map(bsrc, [N, NH * N], N * 2, [64, 128]), _map(O, [DM, A * N], DM * 2, [32, 128], swizzle=64))
    items = ((A + 1) // 2) * NH * (N // 128)
    d = _dev(q)
    atom_kernel("fwd", d)((min(nsm(d), items), 1, 1), (384, 1, 1), *maps, LSE, int(N), int(A))
    return O, LSE


def attn_dd(do, o):
    """D[a, h, i] = sum over head h's 32 columns of dO O: dO, O [A, N, 128] bf16 -> fp32 [A, 4, N]."""
    A, N, _ = do.shape
    D = torch.empty(A, NH, N, device=do.device)
    rows = A * N
    atom_kernel("dd", _dev(do))(((4 * rows + 255) // 256, 1, 1), (256, 1, 1), do.contiguous(), o.contiguous(), D, int(N), int(rows))
    return D


def attn_bwd(q, k, v, do, bias, bias_t, LSE, Dd, dP, *, native_hopper=False):
    """dK, dV, dQ (bf16, into dP's blocks 1 / 2 / 0; dP [A N, 512]) and dbias [4, N, N] fp32 (summed over the samples)."""
    A, N, _ = q.shape
    d = _dev(q)
    q2, k2, v2, do2 = (t.reshape(A * N, DM) for t in (q, k, v, do))
    if _hopper(d):
        from miniworld_engine.kernels.augmented_attention import cuda as core
        if native_hopper or os.environ.get("MINIWORLD_ATOM_DIT_H100_BWD_V2", "0") == "1":
            out = tuple(dP[:, DM * i:DM * (i + 1)] for i in range(3))
            _, _, _, db = core.natural_backward(q2, k2, v2, do2, bias, LSE, Dd, heads=NH, head_dim=DH,
                                                output_bf16=True, out=out, return_bias=A == 1, bias_transposed=False)
            if A != 1:
                db = torch.empty(NH, N, N, device=q.device, dtype=torch.float32)
                maps = tuple(_map(t, [DM, A * N], DM * 2, [64, 64]) for t in (q2, k2, v2, do2))
                if native_hopper or os.environ.get("MINIWORLD_ATOM_DIT_H100_DB_TMA", "0") == "1":
                    mb = _map(bias, [N, NH * N], N * 2, [64, 64])
                    atom_kernel("db_tma", d)((N // 64, N // 64, NH), (160, 1, 1),
                                             *maps, mb, LSE, Dd, db, int(N), int(A))
                else:
                    atom_kernel("db_native", d)((N // 64, N // 64, NH), (160, 1, 1),
                                                *maps, bias, LSE, Dd, db, int(N), int(A))
            return db
        dq, dk, dv, dbt = core.natural_backward(q2, k2, v2, do2, bias, LSE, Dd,
                                             heads=NH, head_dim=DH, output_bf16=True)
        db = torch.empty(NH, N, N, device=q.device, dtype=torch.float32)
        atom_kernel("db_sum", d)((N // 32, N // 32, NH), (32, 8, 1), dbt, db, int(N), int(A))
        atom_kernel("dq_pack", d)((min(4 * nsm(d), (A * N * 64 + 255) // 256), 1, 1),
                                  (256, 1, 1), dq, dk, dv, dP, int(A * N))
        return db
    DQ, DK, DV = (dP[:, DM * i:DM * (i + 1)] for i in range(3))
    DB = torch.empty(NH, N, N, device=q.device, dtype=torch.float32)
    mb = _map(bias, [N, NH * N], N * 2, [64, 128])
    mbt = _map(bias_t, [N, NH * N], N * 2, [64, 128])
    m128 = tuple(_tm(t, A * N, 128, swizzle=64) for t in (q2, k2, v2, do2))              # every kernel: dense 64-B rows (SW64)
    def bfo(t, bc, sw):                                                                  # a 128-column block of dP
        return _map(t, [DM, A * N], 8 * DM, [bc, 128], swizzle=sw)
    mkv = (_tm(q2, A * N, 64, swizzle=64), m128[1], m128[2], _tm(do2, A * N, 64, swizzle=64), mbt, bfo(DK, 32, 64), bfo(DV, 32, 64))
    mq = (*m128, mb, bfo(DQ, 16, 32))
    mbias = (*m128, mb)
    g_kv = (min(nsm(d), A * NH * (N // 128)), 1, 1)
    g_b = (min(nsm(d), NH * (N // 128) ** 2), 1, 1)
    atom_kernel("dkv", d)(g_kv, (384, 1, 1), *mkv, LSE, Dd, DK, DV, None, int(N), int(A))
    atom_kernel("dq", d)(g_kv, (384, 1, 1), *mq, LSE, Dd, int(N), int(A))
    atom_kernel("dbias", d)(g_b, (384, 1, 1), *mbias, LSE, Dd, DB, int(N), int(A))
    return DB


def cond_fwd(c, W, b, g1, g2, eps=1e-5):
    """c [M, 128] bf16 -> mod [M, 768] bf16 = [s1 | bi1 | so | s2 | bi2 | st] (the gates as rn(sigmoid))."""
    M = c.shape[0]
    out = torch.empty(M, 6 * DM, device=c.device, dtype=torch.bfloat16)
    if _hopper(_dev(c)):
        d = _dev(c)
        maps = (_map(c, [DM, M], DM * 2, [64, 64]),
                _map(W, [DM, 6 * DM], DM * 2, [64, 128]))
        wide = M > 64 * nsm(d)
        atom_kernel("cond1" if wide else "cond", d)((min((2 if wide else 1) * nsm(d), (M + 63) // 64), 2, 1), (160, 1, 1),
                              *maps, g1, g2, b, out, int(M), float(eps))
        return out
    maps = (_rows(c), _map(W, [DM, 6 * DM], DM * 2, [32, 128], swizzle=64), _rows(out, 6 * DM))
    ntile = M // 32
    d = _dev(c)
    atom_kernel("cond", d)((min(nsm(d) // 2, ntile), 2, 1), (384, 1, 1), *maps, g1, g2, b, int(ntile), float(eps))
    return out


def pre_fwd(a, mod, W, bq, save=False, eps=1e-5, cross=False):
    """a [M, 128], mod [M, 768] -> Q, K, V, G [M, 128] bf16 (+ x1 = AdaLN 1 output when ``save``)."""
    M = a.shape[0]
    Q, K, V, G = (torch.empty(M, DM, device=a.device, dtype=torch.bfloat16) for _ in range(4))
    X = torch.empty(M, DM, device=a.device, dtype=torch.bfloat16) if save else G
    if _hopper(_dev(a)):
        d = _dev(a)
        maps = (_map(a, [DM, M], DM * 2, [64, 64]),
                _map(mod, [6 * DM, M], 12 * DM, [64, 64]),
                _map(W, [DM, 4 * DM], DM * 2, [64, 128]))
        atom_kernel("pre_cross" if cross else "pre", d)((min((2 if cross else 1) * nsm(d), (M + 63) // 64), 1, 1), (160, 1, 1),
                             *maps, bq, Q, K, V, G, X, int(M), float(eps), int(save))
        return Q, K, V, G, X
    maps = (_rows(a), _map(W, [DM, 4 * DM], DM * 2, [32, 128], swizzle=64), _rows(mod, 6 * DM),
            _rows(Q), _rows(K), _rows(V), _rows(G), _rows(X))
    ntile = M // 32
    d = _dev(a)
    atom_kernel("pre", d)((min(nsm(d), ntile), 1, 1), (384, 1, 1), *maps, bq, int(ntile), float(eps), int(save))
    return Q, K, V, G, X


def post_fwd(a, o, g, mod, Wo, Wu, Wd, save=False, eps=1e-5):
    """a, o, g [M, 128], mod [M, 768] -> the block output [M, 128] bf16 (+ u, a2, x2, t when ``save``)."""
    M = a.shape[0]
    def new():
        return torch.empty(M, DM, device=a.device, dtype=torch.bfloat16)
    out = new()
    sv = tuple(new() for _ in range(4)) if save else (out,) * 4
    if _hopper(_dev(a)):
        d = _dev(a)
        maps = (_map(Wo, [DM, DM], DM * 2, [64, 128]),
                _map(Wu, [DM, 4 * DM], DM * 2, [64, 128]),
                _map(Wd, [2 * DM, DM], DM * 4, [64, 128]))
        wide = M > 64 * nsm(d)
        atom_kernel("post2" if wide else "post", d)((min((2 if wide else 1) * nsm(d), (M + 63) // 64), 1, 1), (160, 1, 1),
                              a, o, g, mod, *maps, out, *sv, int(M), float(eps), int(save))
        return (out, *sv)
    maps = (_rows16(a), _rows16(o), _rows16(g), _rows16(mod, 6 * DM), _map(Wo, [DM, DM], DM * 2, [64, 128]),
            _map(Wu, [DM, 4 * DM], DM * 2, [32, 128], swizzle=64), _map(Wd, [2 * DM, DM], 4 * DM, [32, 128], swizzle=64),
            _rows16(out), *(_rows16(t) for t in sv))
    ntile = M // 16
    d = _dev(a)
    atom_kernel("post", d)((min(nsm(d), ntile), 1, 1), (384, 1, 1), *maps, int(ntile), float(eps), int(save))
    return (out, *sv)


def tr_bwd_gate(dy, t, mod, x2, Wu, WsT, dmod, dbts):
    """dy, t, mod (st), x2 -> DT [M, 128], HH [M, 256], DAB [M, 512]; dts into dmod[:, 640:768]; d bts += the column sums."""
    M = dy.shape[0]
    dev = dy.device
    DT = torch.empty(M, DM, device=dev, dtype=torch.bfloat16)
    HH = torch.empty(M, 2 * DM, device=dev, dtype=torch.bfloat16)
    DAB = torch.empty(M, 4 * DM, device=dev, dtype=torch.bfloat16)
    if _hopper(_dev(dy)):
        d = _dev(dy)
        maps = (_map(Wu, [DM, 4 * DM], DM * 2, [64, 128]),
                _map(WsT, [DM, 2 * DM], DM * 2, [64, 128]))
        atom_kernel("trg", d)((min(nsm(d), (M + 63) // 64), 1, 1), (160, 1, 1),
                             dy, t, mod, x2, *maps, DT, HH, DAB, dmod, int(M))
        return DT, HH, DAB
    maps = (_rows16(dy), _rows16(t), _rows16(mod, 6 * DM), _rows16(x2), _map(Wu, [DM, 4 * DM], DM * 2, [32, 128], swizzle=64),
            _map(WsT, [DM, 2 * DM], DM * 2, [32, 128], swizzle=64), _rows16(DT), _rows16(dmod, 6 * DM), _rows16(HH, 2 * DM),
            _rows16(DAB, 4 * DM))
    ntile = M // 16
    d = _dev(dy)
    atom_kernel("trg", d)((min(nsm(d), ntile), 1, 1), (384, 1, 1), *maps, dbts, int(ntile))
    return DT, HH, DAB


def post_bwd(DAB, dy, a2, mod, u, sg, o, WuT, WoT, dmod, dP, Dd, DBIAS, N, eps=1e-5):
    """-> DA, DU, GATED, DO [M, 128]; dmod blocks 2-4, dP block 3 (dg), D [A, 4, N]; d bsc2 / d bos into DBIAS."""
    M = dy.shape[0]
    def new():
        return torch.empty(M, DM, device=dy.device, dtype=torch.bfloat16)
    DA, DU, GATED, DO = new(), new(), new(), new()
    if _hopper(_dev(dy)):
        d = _dev(dy)
        maps = (_map(WuT, [4 * DM, DM], 8 * DM, [64, 128]),
                _map(WoT, [DM, DM], DM * 2, [64, 128]))
        atom_kernel("postb", d)((min(nsm(d), (M + 63) // 64), 1, 1), (160, 1, 1),
                               DAB, dy, a2, mod, u, sg, o, *maps, DA, DU, GATED, DO, dmod, dP, Dd,
                               int(M), int(N), float(eps))
        return DA, DU, GATED, DO
    maps = (_rows16(DAB, 4 * DM), _rows16(dy), _rows16(a2), _rows16(mod, 6 * DM), _rows16(u), _rows16(sg), _rows16(o),
            _map(WuT, [4 * DM, DM], 8 * DM, [32, 128], swizzle=64), _map(WoT, [DM, DM], DM * 2, [32, 128], swizzle=64),
            _rows16(DA), _rows16(dmod, 6 * DM), _rows16(DU), _rows16(GATED), _rows16(DO), _rows16(dP, 4 * DM))
    ntile = M // 16
    d = _dev(dy)
    atom_kernel("postb", d)((min(nsm(d), ntile), 1, 1), (384, 1, 1), *maps, Dd, DBIAS, int(N), int(ntile), float(eps))
    return DA, DU, GATED, DO


def pre_bwd(dP, a, mod, DA, WT, dmod, DBIAS, DBQ, eps=1e-5):
    """dP [M, 512], a, mod (s1), DA -> d single [M, 128]; dmod blocks 0-1; d bsc1 into DBIAS[0:128], d bq into DBQ."""
    M = a.shape[0]
    DS = torch.empty(M, DM, device=a.device, dtype=torch.bfloat16)
    if _hopper(_dev(a)):
        d = _dev(a)
        atom_kernel("preb", d)((min(nsm(d), (M + 63) // 64), 1, 1), (160, 1, 1),
                              dP, a, mod, DA, _map(WT, [4 * DM, DM], 8 * DM, [64, 128]),
                              DS, dmod, int(M), float(eps))
        return DS
    maps = (_rows16(dP, 4 * DM), _rows16(a), _rows16(mod, 6 * DM), _rows16(DA), _map(WT, [4 * DM, DM], 8 * DM, [32, 128], swizzle=64),
            _rows16(DS), _rows16(dmod, 6 * DM))
    ntile = M // 16
    d = _dev(a)
    atom_kernel("preb", d)((min(nsm(d), ntile), 1, 1), (384, 1, 1), *maps, DBIAS, DBQ, int(ntile), float(eps))
    return DS


def cond_bwd(dmod, c, WmodT, g1, g2, DG, eps=1e-5):
    """dmod [M, 768] (backward layout), c [M, 128] -> dc, cn1, cn2 [M, 128]; d g1 / d g2 into DG [256]."""
    M = c.shape[0]
    def new():
        return torch.empty(M, DM, device=c.device, dtype=torch.bfloat16)
    dc, cn1, cn2 = new(), new(), new()
    if _hopper(_dev(c)):
        d = _dev(c)
        atom_kernel("condb", d)((min(nsm(d), (M + 63) // 64), 1, 1), (160, 1, 1),
                               dmod, c, _map(WmodT, [6 * DM, DM], 12 * DM, [64, 128]),
                               g1, g2, dc, cn1, cn2, DG, int(M), float(eps))
        return dc, cn1, cn2
    maps = (_rows16(dmod, 6 * DM), _rows16(c), _map(WmodT, [6 * DM, DM], 12 * DM, [32, 128], swizzle=64),
            _rows16(dc), _rows16(cn1), _rows16(cn2))
    ntile = M // 16
    d = _dev(c)
    atom_kernel("condb", d)((min(nsm(d), ntile), 1, 1), (384, 1, 1), *maps, g1, g2, DG, int(ntile), float(eps))
    return dc, cn1, cn2


# PyTorch may erase its cached cuBLASLt workspace when a different graph is reset.
# A capture-local tensor keeps each retained graph's scratch storage alive.
# This is an opt-in lifetime repair; all projection operands and CUDA kernels stay unchanged.
_OWNED_WGRAD_CPP = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
torch::Tensor owned_wgrad(torch::Tensor g, torch::Tensor x) {
  TORCH_CHECK(g.is_cuda() && x.is_cuda() && g.device()==x.device()
    && g.scalar_type()==at::kBFloat16 && x.scalar_type()==at::kBFloat16
    && g.dim()==2 && x.dim()==2 && g.size(0)==x.size(0)
    && g.stride(1)==1 && x.stride(1)==1 && g.size(0)>0
    && g.stride(0)>=g.size(1) && x.stride(0)>=x.size(1),
    "owned cuBLASLt wgrad requires BF16 row-major or row-padded operands");
  c10::cuda::CUDAGuard guard(g.device());
  const int64_t reduction=g.size(0), outputs=g.size(1), inputs=x.size(1);
  auto out=torch::empty({outputs,inputs},g.options().dtype(at::kFloat));
  const size_t workspace_bytes=at::cuda::getCUDABlasLtWorkspaceSize();
  auto workspace=torch::empty({int64_t(workspace_bytes)},g.options().dtype(at::kByte));
  auto checked=[](cublasStatus_t s){TORCH_CHECK(s==CUBLAS_STATUS_SUCCESS,"owned cuBLASLt wgrad status ",int(s));};
  struct Descriptors {
    cublasLtMatmulDesc_t operation=nullptr;
    cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
    cublasLtMatmulPreference_t preference=nullptr;
    ~Descriptors(){
      if(preference)cublasLtMatmulPreferenceDestroy(preference);
      if(c)cublasLtMatrixLayoutDestroy(c);
      if(b)cublasLtMatrixLayoutDestroy(b);
      if(a)cublasLtMatrixLayoutDestroy(a);
      if(operation)cublasLtMatmulDescDestroy(operation);
    }
  } d;
  checked(cublasLtMatmulDescCreate(&d.operation,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t op_a=CUBLAS_OP_N,op_b=CUBLAS_OP_T;
  checked(cublasLtMatmulDescSetAttribute(d.operation,CUBLASLT_MATMUL_DESC_TRANSA,&op_a,sizeof(op_a)));
  checked(cublasLtMatmulDescSetAttribute(d.operation,CUBLASLT_MATMUL_DESC_TRANSB,&op_b,sizeof(op_b)));
  checked(cublasLtMatrixLayoutCreate(&d.a,CUDA_R_16BF,inputs,reduction,x.stride(0)));
  checked(cublasLtMatrixLayoutCreate(&d.b,CUDA_R_16BF,outputs,reduction,g.stride(0)));
  checked(cublasLtMatrixLayoutCreate(&d.c,CUDA_R_32F,inputs,outputs,inputs));
  checked(cublasLtMatmulPreferenceCreate(&d.preference));
  checked(cublasLtMatmulPreferenceSetAttribute(d.preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes)));
  const void* ptrs[]={x.data_ptr(),g.data_ptr(),out.data_ptr(),out.data_ptr()};
  cublasLtMatmulPreferenceAttributes_t attrs[]={CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES};
  for(int i=0;i<4;++i){
    uint32_t alignment=256;
    while(reinterpret_cast<uintptr_t>(ptrs[i])%alignment)alignment>>=1;
    checked(cublasLtMatmulPreferenceSetAttribute(d.preference,attrs[i],&alignment,sizeof(alignment)));
  }
  cublasLtMatmulHeuristicResult_t algo{};int found=0;
  auto handle=at::cuda::getCurrentCUDABlasLtHandle();
  checked(cublasLtMatmulAlgoGetHeuristic(handle,d.operation,d.a,d.b,d.c,d.c,d.preference,1,&algo,&found));
  TORCH_CHECK(found==1,"owned cuBLASLt wgrad has no valid algorithm");
  float alpha=1.f,beta=0.f;
  checked(cublasLtMatmul(handle,d.operation,&alpha,x.data_ptr(),d.a,g.data_ptr(),d.b,
    &beta,out.data_ptr(),d.c,out.data_ptr(),d.c,&algo.algo,workspace.data_ptr(),workspace_bytes,at::cuda::getCurrentCUDAStream()));
  return out;
}

"""


@functools.lru_cache(maxsize=1)
def _owned_wgrad_extension():
    from torch.utils.cpp_extension import load_inline
    from miniworld_engine.kernels._nvcc import ensure_cuda_home

    cuda_home = ensure_cuda_home()
    if cuda_home is None or tuple(map(int, torch.version.cuda.split(".")[:2])) > (12, 9):
        raise RuntimeError("Atom Hopper cuBLASLt gradients require CUDA <= 12.9")
    root = Path(os.environ.get("MINIWORLD_ENGINE_JIT_ROOT", Path.home() / ".cache" / "miniworld_engine_jit"))
    build = root / "atom_owned_wgrad" / hashlib.sha256(_OWNED_WGRAD_CPP.encode()).hexdigest()[:16]
    build.mkdir(parents=True, exist_ok=True)
    return load_inline(
        name="atom_hopper_owned_wgrad", cpp_sources=_OWNED_WGRAD_CPP, functions=["owned_wgrad"],
        with_cuda=True, build_directory=str(build),
        extra_include_paths=[str(Path(cuda_home) / "targets" / "x86_64-linux" / "include")],
        extra_cflags=["-O3"], extra_ldflags=["-lcublasLt"], verbose=False,
    )


def owned_wgrad_hopper(gradient, activation):
    """BF16 GEMM to FP32 with cuBLASLt scratch owned by the captured graph."""
    return _owned_wgrad_extension().owned_wgrad(gradient, activation)


def pack_hopper(weights, backward=False):
    """Live CUDA weight packing; identical launch count for BF16 and FP32 masters."""
    if len(weights) != 23:
        raise ValueError("Atom weight pack requires the 23 canonical weights")
    d, dev = _dev(weights[0]), weights[0].device
    shapes = ((768, 128), (512, 128), (128, 128), (512, 128), (128, 256))
    wc, wp, wo, wu, wd = (torch.empty(s, device=dev, dtype=torch.bfloat16) for s in shapes)
    b = torch.empty(768, device=dev, dtype=torch.float32)
    g1, g2, bq = (torch.empty(128, device=dev, dtype=torch.float32) for _ in range(3))
    if any(w.dtype not in (torch.bfloat16, torch.float32) or not w.is_contiguous() for w in weights):
        raise ValueError("Atom Hopper pack requires contiguous BF16/FP32 parameters")
    bfmask = sum((w.dtype == torch.bfloat16) << i for i, w in enumerate(weights))
    atom_kernel("pack", d)((384, 1, 1), (256, 1, 1), *weights, wc, wp, wo, wu, wd, b, g1, g2, bq, bfmask)
    wb = {}
    if backward:
        wt = [torch.empty(tuple(reversed(s)), device=dev, dtype=torch.bfloat16) for s in shapes]
        atom_kernel("pack_t", d)((272, 1, 1), (32, 8, 1), wc, wp, wo, wu, wd, *wt)
        wb = dict(zip(("WmodT", "WpT", "WoT", "WuT", "WsT"), wt, strict=True))
    return (wc, b, g1, g2), (wp, bq), (wo, wu, wd), wb


def pair_weight_hopper(wb, gamma):
    out = torch.empty(NH, DP, device=wb.device, dtype=torch.float32)
    bfmask = int(wb.dtype == torch.bfloat16) | (int(gamma.dtype == torch.bfloat16) << 1)
    atom_kernel("pb_w", _dev(wb))((1, 1, 1), (64, 1, 1), wb, gamma, out, bfmask)
    return out


def finish_columns_hopper(dmod, dp, dgparts, dbias, dbq, dg):
    part = torch.empty(4, 896, device=dmod.device, dtype=torch.float32)
    d, m = _dev(dmod), dmod.shape[0]
    atom_kernel("col_part", d)((56, 4, 1), (256, 1, 1), dmod, dp, part, int(m))
    atom_kernel("col_fin", d)((4, 1, 1), (256, 1, 1), part, dgparts, dbias, dbq, dg, int((m + 63) // 64))


def grad_pack_hopper(grads, weights):
    if any(g.dtype != torch.float32 or not g.is_contiguous() for g in grads):
        raise ValueError("Hopper gradient pack requires contiguous FP32 gradient views")
    out = [torch.empty_like(w) for w in weights]
    bfmask = sum((w.dtype == torch.bfloat16) << i for i, w in enumerate(weights))
    atom_kernel("grad_pack", _dev(weights[0]))((128, 1, 1), (256, 1, 1), *grads, *out, bfmask)
    return out
