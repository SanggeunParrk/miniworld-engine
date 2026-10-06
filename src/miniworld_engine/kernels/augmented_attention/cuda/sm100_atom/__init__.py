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

The fp32 path (fp32 activations, ``KERNELS_TF32``, loaded by ``atom_kernel32`` only when an fp32 call is served; used by the local
atom block, ``integrations/local_dit``):
  gemm_tf32.cu    every projection and activation-gradient product as one tcgen05 kind::tf32 GEMM launch with the block's epilogues
                  (bias + sigmoid blocks, gated residual, SwiGLU).
  rows_tf32.cu    the row-wise stages around them (LayerNorms, AdaLN and their backward, gates, SwiGLU backward, D), fp32 CUDA cores.

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
def cubin(stem: str) -> str:
    """Path of ``<stem>.cu`` built for sm_100a with the sources' default options; rebuilt only when a source or a flag changes."""
    from miniworld_engine.kernels.transition.cuda.fused_sm100a import kernel_toolchain

    nvcc, rel, host = kernel_toolchain()
    flags = (*host, "-std=c++17", "-O3", "-arch=sm_100a", "-cubin", "-lineinfo", f"-I{_dir}")
    h = hashlib.sha256(" ".join((nvcc, str(rel), *flags)).encode())
    for f in (_dir / "sm100.cuh", _dir / f"{stem}.cu"):
        h.update(f.read_bytes())
    root = Path(os.environ.get("MINIWORLD_ENGINE_JIT_ROOT", Path.home() / ".cache" / "miniworld_engine_jit"))
    out = root / "atom_dit_sm100" / f"{stem}_{h.hexdigest()[:16]}.cubin"
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
        return driver.Kernel(cubin(stem), func, smem, pdl=pdl)


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


def atom_kernel(name: str, device_index: int):
    stem, func, smem, pdl = KERNELS[name]
    return _load_atom_kernel(stem, func, device_index, smem, pdl)


def load_all(device_index: int) -> None:
    """Build and load every kernel on the device (raises on any failure)."""
    for name in KERNELS:
        atom_kernel(name, device_index)


@functools.lru_cache(maxsize=8)
def nsm(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _dev(t: torch.Tensor) -> int:
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


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
    w = (wb.float() * gamma.float()[None]).contiguous()
    bo = torch.empty(NH, N, N, device=z.device, dtype=torch.bfloat16)
    bt = torch.empty_like(bo) if trans else None
    atom_kernel("pb_f" if N == NZ else "pb_fp", _dev(z))((N // 64, N // 32, 1), (256, 1, 1), z, w, bo, bt, int(N), int(NZ), kv,
                                                        float(eps))
    return bo, bt


def pair_bias_bwd(z, gamma, wb, dbias, eps=1e-5, kv=None):
    """dbias [4, n, n] fp32 (summed over the samples) -> dz (z's dtype and layout), dgamma [16], dWb [4, 16] (fp32); dbias on
    keys kv marks invalid is dropped."""
    z = z.reshape(z.shape[-3], z.shape[-2], DP)
    NZ, N = z.shape[0], dbias.shape[-1]
    w = (wb.float() * gamma.float()[None]).contiguous()
    dz = torch.empty_like(z)
    nb = (N // 64) * (N // 32)
    pw = torch.empty(nb, NH, DP, device=z.device, dtype=torch.float32)
    atom_kernel("pb_b", _dev(z))((N // 64, N // 32, 1), (256, 1, 1), z, w, dbias.contiguous(), dz, pw, int(N), int(NZ), kv,
                                 float(eps))
    dw = pw.sum(0)
    return dz, (dw * wb.float()).sum(0), dw * gamma.float()[None]


def attn_fwd(q, k, v, bias):
    """q, k, v [A, N, 128] bf16 (4 heads x 32), bias [4, N, N] bf16 -> O [A N, 128] bf16, LSE [A, 4, N] fp32 (log2)."""
    A, N, _ = q.shape
    q2, k2, v2 = (t.reshape(A * N, DM) for t in (q, k, v))
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


def attn_bwd(q, k, v, do, bias, bias_t, LSE, Dd, dP):
    """dK, dV, dQ (bf16, into dP's blocks 1 / 2 / 0; dP [A N, 512]) and dbias [4, N, N] fp32 (summed over the samples)."""
    A, N, _ = q.shape
    d = _dev(q)
    q2, k2, v2, do2 = (t.reshape(A * N, DM) for t in (q, k, v, do))
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
    maps = (_rows(c), _map(W, [DM, 6 * DM], DM * 2, [32, 128], swizzle=64), _rows(out, 6 * DM))
    ntile = M // 32
    d = _dev(c)
    atom_kernel("cond", d)((min(nsm(d) // 2, ntile), 2, 1), (384, 1, 1), *maps, g1, g2, b, int(ntile), float(eps))
    return out


def pre_fwd(a, mod, W, bq, save=False, eps=1e-5):
    """a [M, 128], mod [M, 768] -> Q, K, V, G [M, 128] bf16 (+ x1 = AdaLN 1 output when ``save``)."""
    M = a.shape[0]
    Q, K, V, G = (torch.empty(M, DM, device=a.device, dtype=torch.bfloat16) for _ in range(4))
    X = torch.empty(M, DM, device=a.device, dtype=torch.bfloat16) if save else G
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
    maps = (_rows16(dmod, 6 * DM), _rows16(c), _map(WmodT, [6 * DM, DM], 12 * DM, [32, 128], swizzle=64),
            _rows16(dc), _rows16(cn1), _rows16(cn2))
    ntile = M // 16
    d = _dev(c)
    atom_kernel("condb", d)((min(nsm(d), ntile), 1, 1), (384, 1, 1), *maps, g1, g2, DG, int(ntile), float(eps))
    return dc, cn1, cn2


# --------------------------------------------------------------------------------------------------- fp32 path (TF32 tensor cores)
SOURCES_TF32 = ("gemm_tf32", "rows_tf32")
F32 = torch.float32


@functools.lru_cache(maxsize=None)
def cubin_tf32(stem: str) -> str:
    """``cubin`` for the fp32-path sources, which include ``../sm100/sm100.cuh`` (the kind::tf32 helpers): that header is hashed."""
    from miniworld_engine.kernels.transition.cuda.fused_sm100a import kernel_toolchain

    nvcc, rel, host = kernel_toolchain()
    flags = (*host, "-std=c++17", "-O3", "-arch=sm_100a", "-cubin", "-lineinfo", f"-I{_dir}")
    h = hashlib.sha256(" ".join((nvcc, str(rel), *flags)).encode())
    for f in (_dir.parent / "sm100" / "sm100.cuh", _dir / f"{stem}.cu"):
        h.update(f.read_bytes())
    root = Path(os.environ.get("MINIWORLD_ENGINE_JIT_ROOT", Path.home() / ".cache" / "miniworld_engine_jit"))
    out = root / "atom_dit_sm100_tf32" / f"{stem}_{h.hexdigest()[:16]}.cubin"
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


#: (stem, function, dynamic smem) of every fp32-path kernel (gemm_tf32's size is static_asserted in the source)
KERNELS_TF32 = {
    "gemm": ("gemm_tf32", "atom_gemm_tf32", 213248),
    "ln": ("rows_tf32", "f32_ln_aff", 0),
    "adaln": ("rows_tf32", "f32_adaln", 0),
    "gate": ("rows_tf32", "f32_gate", 0),
    "tail_b": ("rows_tf32", "f32_tail_bwd", 0),
    "swiglu_b": ("rows_tf32", "f32_swiglu_bwd", 0),
    "adaln_b": ("rows_tf32", "f32_adaln_bwd", 0),
    "gate_b": ("rows_tf32", "f32_gate_bwd", 0),
    "ln_b": ("rows_tf32", "f32_ln_bwd", 0),
}


@functools.lru_cache(maxsize=None)
def atom_kernel32(name: str, device_index: int):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    stem, func, smem = KERNELS_TF32[name]
    with torch.cuda.device(device_index):
        return driver.Kernel(cubin_tf32(stem), func, smem)


def load_all32(device_index: int) -> None:
    """Build and load every fp32-path kernel on the device (raises on any failure)."""
    for name in KERNELS_TF32:
        atom_kernel32(name, device_index)


_TMAPS32: dict = {}


def _tm32(t, box, swizzle=128):
    """TMA map of a row-major fp32 [rows, cols] tensor (unit column stride, any row stride), cached by everything it encodes (address,
    geometry, box, swizzle; encoding one costs ~5-10 us of host time). The cache does not hold the tensor."""
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    assert t.dtype == F32 and t.dim() == 2 and t.stride(1) == 1, (t.dtype, t.shape, t.stride())
    rows, cols = t.shape
    key = (t.data_ptr(), cols, rows, t.stride(0), tuple(box), swizzle)
    m = _TMAPS32.get(key)
    if m is None:
        if len(_TMAPS32) >= 512:
            _TMAPS32.clear()
        m = driver.TensorMap(t, [cols, rows], t.stride(0) * 4, list(box), swizzle=swizzle, dtype="f32")
        m.keep = None
        _TMAPS32[key] = m
    return m


#: gemm32 epilogues
STORE, RESID_GATE, SWIGLU = 0, 1, 2


def gemm32(a, w, out=None, *, k=None, n=None, a_col0=0, b_row0=0, b_col0=0, ocol0=0, bias=None, sigmask=0, rndmask=0, mode=STORE,
           res=None, gate=None, out2=None):
    """The fp32 path's TF32 product (``gemm_tf32.cu``): acc = a[:, a_col0 : a_col0 + k] @ w[b_row0 : b_row0 + n, b_col0 : b_col0 + k]^T.

    STORE       out[:, ocol0 + j] = acc[:, j] (+ bias[b_row0 + j]), sigmoid on the 128-column blocks (b_row0 + j) // 128 set in sigmask,
                rounded to tf32 (RNA) on the blocks set in rndmask (columns that only feed later MMAs)
    RESID_GATE  out[:, j] = res[:, j] + gate[:, j] * acc[:, j]; out2 (optional) = acc
    SWIGLU      w is the interleaved [Wa_0; Wb_0; Wa_1; Wb_1] pack (n = 512): out [M, 256] = silu(a) * b; out2 (optional) = acc [M, 512]
    a, w, out, out2 are fp32 [rows, cols] with unit column stride (out allocated [M, n] (SWIGLU: [M, n / 2]) when None); res and gate
    fp32 [M, >= n] views. M is a multiple of 128, k of 32, n of 128 (SWIGLU: 512). The kernel rounds the A slices to tf32 (RNA) in shared
    memory before the MMAs; w must already be tf32-rounded (``round_tf32``): kind::tf32 truncates whatever it is given."""
    M = a.shape[0]
    k = a.shape[1] - a_col0 if k is None else k
    n = w.shape[0] - b_row0 if n is None else n
    nt = 256 if (mode == SWIGLU or n % 256 == 0) else 128
    assert M % 128 == 0 and k % 32 == 0 and n % nt == 0, (M, k, n, nt)
    assert mode != SWIGLU or n == 512
    if out is None:
        out = torch.empty(M, n // 2 if mode == SWIGLU else n, device=a.device, dtype=F32)
    ma, mb = _tm32(a, (32, 128)), _tm32(w, (32, nt))
    mo = _tm32(out, (32, 32))
    mo2 = mo if out2 is None else _tm32(out2, (32, 32))
    tiles = (M // 128) * (n // nt)
    d = _dev(a)
    atom_kernel32("gemm", d)((min(nsm(d), tiles), 1, 1), (512, 1, 1), ma, mb, mo, mo2, bias, res, gate, int(M), int(n), int(k), int(nt),
                             int(a_col0), int(b_row0), int(b_col0), int(ocol0), int(sigmask), int(mode), int(out2 is not None),
                             int(res.stride(0)) if res is not None else 0, int(gate.stride(0)) if gate is not None else 0, int(rndmask))
    return out


def round_tf32(t):
    """t (fp32) rounded to tf32, to nearest with ties away from zero (cvt.rna.tf32.f32): add half a tf32 ulp to the magnitude bits, drop
    the low 13. For the weight operands of gemm32, once per pack (kind::tf32 MMAs truncate)."""
    i = t.detach().float().contiguous().view(torch.int32)
    return ((i + 0x1000) & -0x2000).view(torch.float32)


def _rowgrid(M):
    return ((M + 7) // 8, 1, 1)


def _new(t, cols=DM):
    return torch.empty(t.shape[0], cols, device=t.device, dtype=F32)


def ln32(c, g1, g2, g3=None, eps=1e-5):
    """cn_k = LN(c) * g_k  (c [M, 128] fp32; g3 optional) -> (cn1, cn2, cn3 or None)."""
    o1, o2 = _new(c), _new(c)
    o3 = _new(c) if g3 is not None else None
    atom_kernel32("ln", _dev(c))(_rowgrid(c.shape[0]), (256, 1, 1), c, g1, o1, g2, o2, g3, o3, int(c.shape[0]), float(eps))
    return o1, o2, o3


def adaln32(a, scale, shift, eps=1e-5):
    """LN(a) * scale + shift: a [M, 128] fp32, scale / shift fp32 [M, 128] views (the stored sigmoid scale)."""
    out = _new(a)
    atom_kernel32("adaln", _dev(a))(_rowgrid(a.shape[0]), (256, 1, 1), a, scale, int(scale.stride(0)), shift, int(shift.stride(0)), out,
                                    int(a.shape[0]), float(eps))
    return out


def gate32(g, o):
    """sigmoid(g) * o: g a fp32 [M, 128] view, o [M, 128]."""
    out = _new(o)
    atom_kernel32("gate", _dev(o))(_rowgrid(o.shape[0]), (256, 1, 1), g, int(g.stride(0)), o, out, int(o.shape[0]))
    return out


def tail_bwd32(dy, t, st, dst):
    """-> dt = dy * st; dst (a view, written) = dy * t * st (1 - st)."""
    dt = _new(dy)
    atom_kernel32("tail_b", _dev(dy))(_rowgrid(dy.shape[0]), (256, 1, 1), dy, t, st, int(st.stride(0)), dt, dst, int(dst.stride(0)),
                                      int(dy.shape[0]))
    return dt


def swiglu_bwd32(dh, u):
    """dh [M, 256], u [M, 512] (interleaved [a_0 | b_0 | a_1 | b_1]) -> d[a | b] [M, 512] (same layout), h = silu(a) b [M, 256]."""
    dab, hh = _new(u, 4 * DM), _new(u, 2 * DM)
    atom_kernel32("swiglu_b", _dev(u))(_rowgrid(u.shape[0]), (256, 1, 1), dh, u, dab, hh, int(u.shape[0]))
    return dab, hh


def adaln_bwd32(a, scale, dx, dres, dscale, dshift, y=None, so=None, dso=None, eps=1e-5):
    """AdaLN backward: -> da = dres + LN backward of dx * scale; dscale (view) = dx LN(a) scale (1 - scale), dshift (view) = dx. With y:
    also d so (view) = da y so (1 - so) and the returned dyo = da so (the attention output gate upstream of a1 = a + so y)."""
    M = a.shape[0]
    da = _new(a)
    dyo = _new(a) if y is not None else None
    atom_kernel32("adaln_b", _dev(a))(_rowgrid(M), (256, 1, 1), a, scale, int(scale.stride(0)), dx, dres, da, dscale, int(dscale.stride(0)),
                                      dshift, int(dshift.stride(0)), y, so, int(so.stride(0)) if so is not None else 0, dso,
                                      int(dso.stride(0)) if dso is not None else 0, dyo, int(M), float(eps))
    return da, dyo


def gate_bwd32(dgated, g, o, dg, A, n):
    """dgated [M, 128], g a view, o [M, 128] -> dO [M, 128], D [A, 4, n]; dg (view, written) = dgated o s (1 - s)."""
    M = o.shape[0]
    do = _new(o)
    D = torch.empty(A, NH, n, device=o.device, dtype=F32)
    atom_kernel32("gate_b", _dev(o))(_rowgrid(M), (256, 1, 1), dgated, g, int(g.stride(0)), o, do, dg, int(dg.stride(0)), D, int(M), int(n))
    return do, D


def ln_bwd32(c, dc, dcn1, g1, dcn2, g2, dcn3=None, g3=None, eps=1e-5):
    """dc [M, 128] += the LayerNorm backward of dcn_k * g_k (k = 1, 2, 3); -> dg [384] fp32 = [dg1 | dg2 | dg3] (sum_rows dcn_k LN(c))."""
    M = c.shape[0]
    d = _dev(c)
    dg = torch.zeros(3 * DM, device=c.device, dtype=F32)
    atom_kernel32("ln_b", d)((max(1, min(4 * nsm(d), (M + 7) // 8)), 1, 1), (256, 1, 1), c, dc, dcn1, g1, dcn2, g2, dcn3, g3, dg, int(M),
                             float(eps))
    return dg
