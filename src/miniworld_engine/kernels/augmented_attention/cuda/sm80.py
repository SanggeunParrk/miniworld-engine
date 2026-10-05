"""A100 (sm_80) attention with a pair bias shared by the samples, hand-written CUDA (``mma.sync`` / ``ldmatrix`` / ``cp.async``): the token DiT's attention
core (head dim 48, 16 heads at d_single 768) and the ``AugmentedAttentionPairBias`` core (head dim 32 -- the atom DiT's 4 heads -- and 48, bf16 outputs,
with the atom width's pair bias fused: ``pair_bias_forward`` / ``pair_bias_backward``).

    out[a, i, h] = softmax_k(scl q[a, i, h] . k[a, k, h] + bscl bias[h, i, k]) v[a, k, h]       (base 2: scl = sm_scale log2 e, bscl = log2 e)

The structure is the TriangleAttention kernels' (``kernels/triangle_attention/cuda/sm80``): one CTA = (head, 128 queries) x R samples sharing the
bias tile, K | V through a cp.async ring, the online softmax in registers; the samples play the role of the pair rows.  Operands are token-major
views ``[A * L, ld]`` of bf16 (the q | k | v | g column blocks of the projection GEMM's output), the bias ``[H, L, L]`` bf16 with ``-inf`` on masked
keys, ``L`` a multiple of 128.

* ``GatedInferenceCore``: the fused step runner's core on sm_80 (the contract of ``kernels/augmented_attention/cuda/sm100.GatedInferenceCore``): q
  pre-scaled by sm_scale log2 e and the bias by log2 e, ``sigmoid(g) o`` written over q.
* ``forward`` / ``backward``: the token DiT training step's attention (the contract of ``sm100.forward`` / ``sm100.backward``): natural units, ``o`` as fp32
  ``[A L, H hd]`` and the log-sum-exp, fp32 gradients.
* ``plain_forward`` / ``plain_backward``: the module-level core (``integrations/augattn_sm80.py``): bf16 ``o``, bf16 dq / dk / dv, the bias gradient fp32,
  the masked keys folded into the bias.

Built on first use (``load_extension``), never at import.
"""

import functools
import hashlib
import math
import os
from pathlib import Path

import torch

from ..._nvcc import ensure_cuda_home, host_flags, load_extension

_dir = Path(__file__).parent / "sm80"

HD = 48
_L2E = 1.4426950408889634
_MODE_GATE, _MODE_TRAIN = 0, 1
#: no per-sample key penalties (the kernels read ``kpen.numel() > 0`` as "a per-sample key mask")
_NOPEN = torch.empty(0)


def _fwd_schedule(samples: int) -> tuple[int, int, int]:
    """(samples per CTA, K | V ring stages, CTAs per SM) of the forward (every epilogue).  A CTA of two samples shares one bias tile, which pays once the grid
    has several waves (A = 48: 10-15 % over one sample, A = 8: 6 %); at a handful of samples a CTA of one sample balances the waves better, with a four-stage
    ring (head dim 32, A = 5: 7 % over three stages; head dim 48: 1-2 %)."""
    return (2, 3, 2) if samples >= 32 and samples % 2 == 0 else (1, 4, 2)


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    # MINIWORLD_AUGATTN_SM80_FLAGS: extra -D / nvcc flags for A/B experiments (their own build)
    extra = os.environ.get("MINIWORLD_AUGATTN_SM80_FLAGS", "").split()
    tag = "" if not extra else "_" + hashlib.sha1(" ".join(extra).encode()).hexdigest()[:8]
    return load_extension(
        name=f"augmented_attention_sm80{tag}",
        sources=[str(_dir / "ops.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "-gencode=arch=compute_80,code=sm_80", f"-I{_dir}", *extra],
        extra_cflags=["-std=c++17", "-O3"], verbose=False,
    )


@functools.lru_cache(maxsize=1)
def _glue_ext():
    """The gates of the module path (``glue_sm80.cuh``): a second, small extension of the family."""
    ensure_cuda_home()
    return load_extension(
        name="augmented_attention_glue_sm80",
        sources=[str(_dir / "glue_ops.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "-gencode=arch=compute_80,code=sm_80", f"-I{_dir}"],
        extra_cflags=["-std=c++17", "-O3"], verbose=False,
    )


def gate_rows(o: torch.Tensor, g: torch.Tensor, og: torch.Tensor) -> None:
    """``og = sigmoid(g) o`` over [M, d] bf16 / fp32 operands (row strides multiples of 16 bytes; ``g`` may be a column block of a wider tensor)."""
    _glue_ext().gate_rows_cuda(o, g, og)


def gate_bwd(dog: torch.Tensor, o: torch.Tensor, g: torch.Tensor, dob: torch.Tensor, dg: torch.Tensor) -> None:
    """The backward of :func:`gate_rows`: ``dob = dog sigmoid(g)``, ``dg = dog o sigmoid(g) (1 - sigmoid(g))`` (``dg`` may be a column block of a wider tensor)."""
    _glue_ext().gate_bwd_cuda(dog, o, g, dob, dg)


def res_gate(y: torch.Tensor, g2: torch.Tensor, res: torch.Tensor | None, out: torch.Tensor) -> None:
    """``out = res + sigmoid(g2) y`` (``res`` None: ``out = sigmoid(g2) y``)."""
    _glue_ext().res_gate_cuda(y, g2, _NOPEN if res is None else res, out)


def res_gate_bwd(dout: torch.Tensor, y: torch.Tensor, g2: torch.Tensor, dy: torch.Tensor, dg2: torch.Tensor) -> None:
    """The backward of :func:`res_gate` (its residual passes the gradient through): ``dy = dout sigmoid(g2)``, ``dg2 = dout y sigmoid(g2) (1 - sigmoid(g2))``."""
    _glue_ext().res_gate_bwd_cuda(dout, y, g2, dy, dg2)


def bias_pack(bias: torch.Tensor, kmask: torch.Tensor | None, length: int, padded: int, heads: int, scale: float, fill: float) -> torch.Tensor:
    """The attention core's bias from a natural-unit bias ``[B, H, L, L]`` (bf16 / fp32, contiguous; L a multiple of 8): bf16 ``[B, H, Lp, Lp]`` = bias x ``scale`` (the raw units:
    sqrt(head dim)), ``fill`` on the keys a bool ``[B, L]`` ``kmask`` masks and on the keys past L, 0 on the query rows past L, in one pass."""
    batch = bias.shape[0]
    out = torch.empty(batch, heads, padded, padded, device=bias.device, dtype=torch.bfloat16)
    _glue_ext().bias_pack_cuda(bias.view(batch * heads, length, length), _NOPEN if kmask is None else kmask, out.view(batch * heads, padded, padded), length, padded, heads, scale, fill)
    return out


@functools.lru_cache(maxsize=8)
def _is_ampere(index: int) -> bool:
    return torch.cuda.get_device_capability(index) == (8, 0)


def supported(dtype: torch.dtype, length: int, d: int, heads: int, index: int) -> bool:
    """The gated inference core's requirements: sm_80, bf16 (the mma.sync core) or fp32 (the TF32 core), head dim 48 (bf16) or 32 / 48 (fp32), L a multiple of 128."""
    if os.environ.get("MINIWORLD_AUGATTN_SM80", "1") == "0":
        return False
    if dtype is torch.float32:
        return d % heads == 0 and d // heads in TF32_HEAD_DIMS and length % 128 == 0 and _is_ampere(index)
    return dtype is torch.bfloat16 and d // heads == HD and length % 128 == 0 and _is_ampere(index)


def _views(qkvg: torch.Tensor):
    """q, k, v, g: the four column blocks of the projection output ``[M, 4 d]`` (token-major views, row stride 4 d)."""
    d = qkvg.shape[1] // 4
    return tuple(qkvg[:, i * d:(i + 1) * d] for i in range(4))


class GatedInferenceCore:
    """``core(qkvg, bias, b, S)``: the gated attention of block ``b`` over ``S`` samples, written over q.  ``qkvg`` ``[S L, 4 d]`` bf16 (q | k | v | g),
    ``bias`` the hoisted ``[n_blocks H, L, L]`` (block ``b`` = rows ``[b H, b H + H)``), pre-scaled by log2 e with ``-inf`` on masked keys; q is
    pre-scaled by sm_scale log2 e (the runner folds both into the weights), so the kernel's scales are 1."""

    #: the runner reads this to tell the A100 cores (q | k | v | g column blocks, the product written over q) from the sm_100a ones (fp32: q | k | g and v^T)
    ampere = True

    def __init__(self, index: int, dtype: torch.dtype, heads: int, head_dim: int):
        """bf16: the mma.sync core (head dim 32 / 48); fp32: the TF32 core of ``ops_tf32.cu`` (fp32 operands on the tensor cores, head dim 32 / 48), same contract."""
        assert dtype in (torch.bfloat16, torch.float32) and head_dim in GATE_HEAD_DIMS, (dtype, head_dim)
        self.heads, self.head_dim, self.dtype = heads, head_dim, dtype
        self.ext = _ext() if dtype is torch.bfloat16 else _tf32_ext()

    def __call__(self, qkvg: torch.Tensor, bias: torch.Tensor, b: int, samples: int) -> None:
        q, k, v, g = _views(qkvg)
        h = self.heads
        length = bias.shape[-1]
        ld = qkvg.stride(0)
        if self.dtype is torch.float32:
            self.ext.attn_fwd_tf32(q, k, v, bias[b * h:(b + 1) * h], g, q, _NOPEN, samples, length, h, self.head_dim, ld, ld, ld, ld, ld, 1.0, 1.0,
                                   *_fwd32_schedule(self.head_dim), 0, _NOPEN, 0)
            return
        self.ext.attn_fwd(q, k, v, bias[b * h:(b + 1) * h], g, q, torch.empty(0, device=q.device), samples, length, h, self.head_dim, ld, ld, ld, ld, ld,
                          1.0, 1.0, _MODE_GATE, *_fwd_schedule(samples), 0, _NOPEN, 0)


#: head dims of the gated inference core: 48 (the token DiT's 16 heads) and 32 (AttentionPairBias at 12 x 32 / 16 x 24 padded / 16 x 32)
GATE_HEAD_DIMS = (32, 48)


def supported_gate(dtype: torch.dtype, length: int, d: int, heads: int, index: int) -> bool:
    """The gated core's requirements: sm_80, bf16, head dim 32 or 48 (``d`` = ``heads`` x head dim), L a multiple of 128."""
    if os.environ.get("MINIWORLD_AUGATTN_SM80", "1") == "0":
        return False
    return dtype is torch.bfloat16 and d % heads == 0 and d // heads in GATE_HEAD_DIMS and length % 128 == 0 and _is_ampere(index)


def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, samples: int, length: int, heads: int, head_dim: int):
    """The training forward: ``q`` / ``k`` / ``v`` token-major bf16 views ``[A L, H hd]`` (row strides >= H hd), ``bias`` ``[H, L, L]`` bf16 in natural
    units (-inf on masked keys).  Returns ``(o, lse)``: ``o`` fp32 ``[A L, H hd]``, ``lse`` fp32 ``[A, H, L]`` (the log-sum-exp in the scaled base-2
    domain, what the backward recomputes the probabilities from)."""
    assert head_dim == HD, head_dim
    o = torch.empty(samples * length, heads * HD, device=q.device, dtype=torch.float32)
    lse = torch.empty(samples, heads, length, device=q.device, dtype=torch.float32)
    _ext().attn_fwd(q, k, v, bias, q, o, lse, samples, length, heads, HD, q.stride(0), k.stride(0), v.stride(0), q.stride(0), heads * HD,
                    math.sqrt(1.0 / HD) * _L2E, _L2E, _MODE_TRAIN, *_fwd_schedule(samples), 0, _NOPEN, 0)
    return o, lse


#: key side: query-tile ring stages, CTAs per SM
DKV_NST, DKV_MINB = 3, 2


def _dkv_schedule(head_dim: int) -> tuple[int, int]:
    """(query-tile ring stages, CTAs per SM) of the key side: three stages at head dim 48 (a fourth no longer leaves room for two CTAs per SM), four at head dim 32
    (0.6-1.6 % over three)."""
    return (4, 2) if head_dim == 32 else (DKV_NST, DKV_MINB)


def _dq_schedule(samples: int) -> tuple[int, int, int]:
    """(samples per CTA, K | V ring stages, CTAs per SM) of the query side.  Four samples per CTA halve the bias-gradient partials that two write and
    read back (A = 48: dq + reduction 12-20 % faster), at one CTA per SM; two samples when A is not a multiple of four, one when it is odd."""
    return (4, 3, 1) if samples % 4 == 0 else (2, 4, 1) if samples % 2 == 0 else (1, 3, 2)


def backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dob: torch.Tensor, bias: torch.Tensor, lse: torch.Tensor, delta: torch.Tensor,
             samples: int, length: int, heads: int, head_dim: int):
    """The training backward (the contract of ``sm100.backward``): ``q`` / ``k`` / ``v`` as for ``forward``, ``dob`` the output's gradient bf16 ``[A L, H hd]``,
    ``bias`` natural units, ``lse`` the forward's, ``delta`` ``[A, H, L]`` fp32 = sum_d o do per query and head.  Returns ``(dq, dk, dv, db)``: fp32
    ``[A L, H hd]`` each and the bias gradient fp32 ``[H, L, L]`` (the sum over the samples of dS; masked keys are 0).  The bias gradient is summed from
    bf16 partials of 1-4 samples in a fixed order: a replay is bit-identical."""
    assert head_dim == HD, head_dim
    ext = _ext()
    dev, m, d = q.device, samples * length, heads * HD
    sm_scale = 1.0 / math.sqrt(HD)
    scl = sm_scale * _L2E
    dq, dk, dv = (torch.empty(m, d, device=dev, dtype=torch.float32) for _ in range(3))
    rows, nkv, minb = _dq_schedule(samples)
    groups = samples // rows
    dbp = torch.empty(groups, heads, length, length, device=dev, dtype=torch.bfloat16)
    ext.attn_bwd_dq(q, k, v, dob, bias, lse, delta, dq, dbp, samples, length, heads, HD, q.stride(0), k.stride(0), v.stride(0), dob.stride(0), d,
                    scl, _L2E, sm_scale, rows, nkv, minb, 0, _NOPEN, 0)
    db = torch.empty(heads, length, length, device=dev, dtype=torch.float32)
    ext.db_reduce(dbp, db, groups, 1.0)
    bias_t = torch.empty_like(bias)
    ext.bias_transpose(bias, bias_t, length)
    ext.attn_bwd_dkv(q, k, v, dob, bias_t, lse, delta, dk, dv, samples, length, heads, HD, q.stride(0), k.stride(0), v.stride(0), dob.stride(0), d, d,
                     scl, _L2E, sm_scale, DKV_NST, DKV_MINB, 0, _NOPEN, 0)
    return dq, dk, dv, db


# ------------------------------------------------------------------------------------------------- the attention module's core (any head dim of HEAD_DIMS)
#: head dims the plain-epilogue kernels are instantiated for: 32 (the atom DiT: 4 heads) and 48 (the token DiT: 16 heads)
HEAD_DIMS = (32, 48)
_MODE_PLAIN, _MODE_PGATE = 2, 3
#: the pair bias of the atom width: 16 pair channels, 4 heads of 32 (``aux_sm80.cuh``); a masked or padded key's bias is this in natural units (bf16 -9984)
PAIR_C, PAIR_H, PAIR_HD, MASKED_BIAS = 16, 4, 32, -1e4


def supported_plain(dtype: torch.dtype, length: int, d: int, heads: int, index: int) -> bool:
    """The module-level core's requirements: sm_80, bf16, head dim 32 or 48, L a multiple of 128."""
    if os.environ.get("MINIWORLD_AUGATTN_SM80", "1") == "0":
        return False
    return dtype is torch.bfloat16 and d // heads in HEAD_DIMS and d % heads == 0 and length % 128 == 0 and _is_ampere(index)


def key_penalty(mask: torch.Tensor) -> torch.Tensor:
    """The per-sample key penalties of the kernels (``kpen``): fp32 with the shape of the bool key ``mask`` -- 0 where a key is valid, -inf where it is masked."""
    return torch.zeros(mask.shape, device=mask.device, dtype=torch.float32).masked_fill_(~mask, float("-inf"))


def _penalty(kpen: torch.Tensor | None, kps: int, length: int) -> tuple[torch.Tensor, int]:
    return (_NOPEN, 0) if kpen is None else (kpen, kps or length)


def plain_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, samples: int, length: int, heads: int, head_dim: int, *,
                  save_lse: bool, sm_scale: float | None = None, kpen: torch.Tensor | None = None, kps: int = 0, out: torch.Tensor | None = None,
                  lse: torch.Tensor | None = None):
    """``softmax(q k^T / sqrt(hd) + bias) v`` for ``samples`` problems that share the bias: ``q`` / ``k`` / ``v`` token-major bf16 views ``[A L, H hd]`` (row
    strides >= H hd and a multiple of 8), ``bias`` head-major bf16 ``[H, L, L]`` **in raw units**, ``bias_natural / sm_scale`` (the units of ``q . k``: the kernel adds it
    into S through the tensor core), masked keys very negative.  Returns ``(o, lse)``: ``o`` bf16 ``[A L, H hd]``, ``lse`` fp32 ``[A, H, L]`` (the scaled base-2
    log-sum-exp) or an empty tensor.  ``sm_scale`` (default 1 / sqrt(head_dim)) is the softmax scale: a head padded with zero columns (24 -> 32) keeps its real one.
    ``kpen`` (``key_penalty`` of a bool ``[A, L]`` mask, sample stride ``kps`` floats, default L) is a key mask that differs per sample; ``out``: write ``o`` into this
    bf16 ``[A L, >= H hd]`` view instead of a new tensor, ``lse`` the fp32 ``[A, H, L]`` tensor the log-sum-exp goes into (with ``save_lse``)."""
    assert head_dim in HEAD_DIMS, head_dim
    dev = q.device
    sm_scale = 1.0 / math.sqrt(head_dim) if sm_scale is None else sm_scale
    o = torch.empty(samples * length, heads * head_dim, device=dev, dtype=torch.bfloat16) if out is None else out
    if not save_lse:
        lse = torch.empty(0, device=dev, dtype=torch.float32)
    elif lse is None:
        lse = torch.empty(samples, heads, length, device=dev, dtype=torch.float32)
    pen, kps = _penalty(kpen, kps, length)
    _ext().attn_fwd(q, k, v, bias, q, o, lse, samples, length, heads, head_dim, q.stride(0), k.stride(0), v.stride(0), q.stride(0), o.stride(0),
                    _L2E * sm_scale, _L2E, _MODE_PLAIN, *_fwd_schedule(samples), 0, pen, kps)
    return o, lse


def pgate_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, bias: torch.Tensor, samples: int, length: int, heads: int, head_dim: int, *,
                  sm_scale: float | None = None, kpen: torch.Tensor | None = None, kps: int = 0, out: torch.Tensor | None = None) -> torch.Tensor:
    """The module-level inference core: ``sigmoid(g) softmax(q k^T / sqrt(hd) + bias) v`` bf16 ``[A L, H hd]`` (the operands, the raw-unit bias, ``kpen`` and ``out`` as for
    :func:`plain_forward`; ``g`` token-major like q).  ``out`` may be ``q`` itself (the result is written over it: q is dead once its tile is loaded)."""
    assert head_dim in HEAD_DIMS, head_dim
    sm_scale = 1.0 / math.sqrt(head_dim) if sm_scale is None else sm_scale
    o = torch.empty(samples * length, heads * head_dim, device=q.device, dtype=torch.bfloat16) if out is None else out
    pen, kps = _penalty(kpen, kps, length)
    _ext().attn_fwd(q, k, v, bias, g, o, _NOPEN, samples, length, heads, head_dim, q.stride(0), k.stride(0), v.stride(0), g.stride(0), o.stride(0),
                    _L2E * sm_scale, _L2E, _MODE_PGATE, *_fwd_schedule(samples), 0, pen, kps)
    return o


def plain_backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dob: torch.Tensor, bias: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, samples: int,
                   length: int, heads: int, head_dim: int, *, db_natural: bool = True, sm_scale: float | None = None, kpen: torch.Tensor | None = None, kps: int = 0,
                   dq: torch.Tensor | None = None, dk: torch.Tensor | None = None, dv: torch.Tensor | None = None):
    """The backward of ``plain_forward``: ``dob`` the output's gradient bf16 ``[A L, H hd]``, ``o`` / ``lse`` the forward's, ``bias`` in raw units as there.  Returns
    ``(dq, dk, dv, db)``: bf16 ``[A L, H hd]`` each and the bias gradient fp32 ``[H, L, L]`` (the sum over the samples; masked keys 0) -- with respect to the
    NATURAL-unit bias (``db_natural``), or with respect to the raw-unit ``bias`` passed in (``db_natural=False``: times sm_scale).  A replay is bit-identical.
    ``kpen`` / ``kps``: as for :func:`plain_forward`.  ``dq`` / ``dk`` / ``dv``: write the gradients into these bf16 ``[A L, >= H hd]`` views (e.g. the first three
    column blocks of the projection's gradient) instead of new tensors."""
    assert head_dim in HEAD_DIMS, head_dim
    ext = _ext()
    dev, m, d = q.device, samples * length, heads * head_dim
    sm_scale = 1.0 / math.sqrt(head_dim) if sm_scale is None else sm_scale
    scl = sm_scale * _L2E
    delta = torch.empty(samples, heads, length, device=dev, dtype=torch.float32)
    ext.attn_delta(dob, o, delta, samples, length, heads, head_dim)
    dq, dk, dv = (torch.empty(m, d, device=dev, dtype=torch.bfloat16) if t is None else t for t in (dq, dk, dv))
    pen, kps = _penalty(kpen, kps, length)
    rows, nkv, minb = _dq_schedule(samples)
    groups = samples // rows
    dbp = torch.empty(groups, heads, length, length, device=dev, dtype=torch.bfloat16)
    ext.attn_bwd_dq(q, k, v, dob, bias, lse, delta, dq, dbp, samples, length, heads, head_dim, q.stride(0), k.stride(0), v.stride(0), dob.stride(0), dq.stride(0),
                    scl, _L2E, sm_scale, rows, nkv, minb, 0, pen, kps)
    db = torch.empty(heads, length, length, device=dev, dtype=torch.float32)
    ext.db_reduce(dbp, db, groups, 1.0 if db_natural else sm_scale)
    bias_t = torch.empty_like(bias)
    ext.bias_transpose(bias, bias_t, length)
    ext.attn_bwd_dkv(q, k, v, dob, bias_t, lse, delta, dk, dv, samples, length, heads, head_dim, q.stride(0), k.stride(0), v.stride(0), dob.stride(0), dk.stride(0),
                     dv.stride(0), scl, _L2E, sm_scale, *_dkv_schedule(head_dim), 0, pen, kps)
    return dq, dk, dv, db


def _kv(mask: torch.Tensor | None, device) -> torch.Tensor:
    return torch.empty(0, device=device, dtype=torch.bool) if mask is None else mask.reshape(-1).contiguous()


def pair_bias_forward(z: torch.Tensor, ln_weight: torch.Tensor, bias_weight: torch.Tensor, mask: torch.Tensor | None, n: int, eps: float = 1e-5, *,
                      out: torch.Tensor | None = None) -> torch.Tensor:
    """The atom width's pair bias ``LayerNorm(z) Wb^T`` head-major: ``z`` ``[1, NZ, NZ, 16]`` bf16 or fp32, ``ln_weight`` ``[16]`` (no offset), ``bias_weight``
    ``[4, 16]``, ``mask`` a bool ``[1, NZ]`` key mask or None, ``n`` the attention's length (a multiple of 128 >= NZ).  bf16 ``z``: returns bf16 ``[4, n, n]`` in the
    attention core's raw units (``bias_natural sqrt(32)``): masked and padded keys are ``MASKED_BIAS`` (in those units), padded query rows 0.  fp32 ``z`` (the TF32
    kernels): fp32 ``[4, n, n]`` in natural units, masked and padded keys ``-1e4`` (finite: an all-masked row stays a uniform softmax).  ``out``: write into this
    contiguous ``[4, n, n]`` tensor (the one returned) instead of a new one."""
    nz = z.shape[-2]
    w = (bias_weight.float() * ln_weight.float()[None]).contiguous()
    bo = out if out is not None else torch.empty(PAIR_H, n, n, device=z.device, dtype=z.dtype)
    _ext().pair_bias_fwd(z.reshape(nz, nz, PAIR_C), w, bo, _kv(mask, z.device), n, nz, eps, 1.0 if z.dtype is torch.float32 else math.sqrt(PAIR_HD))
    return bo


#: head counts of the generic pair-bias kernel (``pair_bias_gen``)
GEN_HEADS = (4, 8, 12, 16, 24, 32)
#: rows a thread of the generic kernel takes (1, 2 or 4; MINIWORLD_AUGATTN_SM80_PBGEN_ROWS overrides it for A/B runs)
_GEN_ROWS = int(os.environ.get("MINIWORLD_AUGATTN_SM80_PBGEN_ROWS", "2"))


def pair_bias_generic_supported(dp: int, heads: int, dtype: torch.dtype) -> bool:
    """The generic pair-bias kernel's requirements: bf16 / fp32 pair, a width that is a multiple of 8, the heads of :data:`GEN_HEADS`, W' (dp x heads fp32) in shared memory."""
    return dtype in (torch.bfloat16, torch.float32) and dp % 8 == 0 and dp >= 8 and heads in GEN_HEADS and (dp * heads + heads) * 4 <= 160 * 1024


def pair_bias_generic(z: torch.Tensor, wcm: torch.Tensor, mask: torch.Tensor | None, n: int, eps: float = 1e-5, *, oscale: float = 1.0, out: torch.Tensor | None = None,
                      rows: int | None = None) -> torch.Tensor:
    """The pair bias ``LayerNorm(z) Wb^T`` at any width, head-major ``[H, n, n]`` in z's dtype (bf16 or fp32): ``z`` ``[NZ, NZ, C]`` (contiguous), ``wcm`` = the folded weight
    ``bias_weight x ln_weight`` fp32 channel-major ``[C, H]``, ``mask`` a bool ``[NZ]`` key mask or None, ``n`` the attention's length (a multiple of 128 >= NZ), ``oscale`` the scale
    of the output (sqrt(head dim) for the bf16 core's raw units, 1 for the TF32 core's natural units).  A masked or padded key carries ``oscale * MASKED_BIAS``, a padded query row 0."""
    nz, c, heads = z.shape[0], z.shape[-1], wcm.shape[1]
    bo = out if out is not None else torch.empty(heads, n, n, device=z.device, dtype=z.dtype)
    _ext().pair_bias_gen(z.reshape(nz, nz, c), wcm, bo, _kv(mask, z.device), n, nz, eps, oscale, _GEN_ROWS if rows is None else rows)
    return bo


#: pair-bias backward at any width (``pair_bias_gen_bwd``): the widths (-> elements per tile) and head counts it is instantiated for
_GEN_BWD_TE = {64: 128, 128: 64, 256: 32, 512: 16}
GEN_BWD_HEADS = (8, 12, 16, 24)


def pair_bias_generic_bwd_supported(dp: int, heads: int, dtype: torch.dtype) -> bool:
    """The generic pair-bias backward's requirements: bf16 / fp32 pair, a width of 64 / 128 / 256 / 512, 8 / 12 / 16 / 24 heads, the tile in shared memory."""
    te = _GEN_BWD_TE.get(dp)
    return te is not None and dtype in (torch.bfloat16, torch.float32) and heads in GEN_BWD_HEADS and (dp * heads + 2 * te * (dp + 4) + te * heads) * 4 <= 160 * 1024


def pair_bias_generic_backward(z: torch.Tensor, wcm: torch.Tensor, db: torch.Tensor, mask: torch.Tensor | None, n: int, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """The backward of :func:`pair_bias_generic` in natural units: ``z`` ``[NZ, NZ, C]``, ``wcm`` the folded weight fp32 ``[C, H]``, ``db`` fp32 ``[H, n, n]`` (the bias gradient summed over the
    samples), ``mask`` the bool ``[NZ]`` key mask the forward folded in (a masked key's gradient is dropped) or None.  Returns ``(dz, dw)``: ``dz`` in z's dtype and shape, ``dw`` the fp32
    ``[H, C]`` gradient of the folded weight (summed over the kernel's blocks in a fixed order: a replay is bit-identical)."""
    nz, c, heads = z.shape[0], z.shape[-1], wcm.shape[1]
    te = _GEN_BWD_TE[c]
    tiles = -(-nz // te)
    rows = max(1, min(8, nz * tiles // 864))                         # about four waves of two blocks per SM
    pw = torch.empty(tiles * -(-nz // rows), heads, c, device=z.device, dtype=torch.float32)
    dz = torch.empty_like(z)
    _ext().pair_bias_gen_bwd(z.reshape(nz, nz, c), wcm, db, dz.view(nz, nz, c), pw, _kv(mask, z.device), n, nz, eps, rows)
    return dz, pw.sum(0)


def pair_bias_backward_w(z: torch.Tensor, w: torch.Tensor, db: torch.Tensor, mask: torch.Tensor | None, eps: float = 1e-5):
    """:func:`pair_bias_backward` on the folded weight ``w = bias_weight x ln_weight`` fp32 ``[4, 16]``: returns ``(dz, dw)`` with ``dw`` the fp32 gradient of ``w``
    (summed over the kernel's blocks in a fixed order)."""
    nz = z.shape[-2]
    n = db.shape[-1]
    dz = torch.empty_like(z)
    nb = (n // 64) * (n // 32)
    pw = torch.empty(nb, PAIR_H, PAIR_C, device=z.device, dtype=torch.float32)
    _ext().pair_bias_bwd(z.reshape(nz, nz, PAIR_C), w, db, dz.view(nz, nz, PAIR_C), pw, _kv(mask, z.device), n, nz, eps)
    return dz, pw.sum(0)


def pair_bias_backward(z: torch.Tensor, ln_weight: torch.Tensor, bias_weight: torch.Tensor, db: torch.Tensor, mask: torch.Tensor | None, eps: float = 1e-5):
    """The backward of ``pair_bias_forward``: ``db`` fp32 ``[4, n, n]`` (the bias gradient, summed over the samples).  Returns ``(dz, dln_weight, dbias_weight)``:
    ``dz`` in ``z``'s dtype and shape, the parameters' gradients fp32."""
    nz = z.shape[-2]
    n = db.shape[-1]
    w = (bias_weight.float() * ln_weight.float()[None]).contiguous()
    dz = torch.empty_like(z)
    nb = (n // 64) * (n // 32)
    pw = torch.empty(nb, PAIR_H, PAIR_C, device=z.device, dtype=torch.float32)
    _ext().pair_bias_bwd(z.reshape(nz, nz, PAIR_C), w, db, dz.view(nz, nz, PAIR_C), pw, _kv(mask, z.device), n, nz, eps)
    dw = pw.sum(0)                                          # d(W') [4, 16], summed over the blocks in a fixed order
    return dz, (dw * bias_weight.float()).sum(0), dw * ln_weight.float()[None]


# ------------------------------------------------------------------------------------------------- fp32 operands on the TF32 tensor cores (``ops_tf32.cu``)
#: head dims the TF32 kernels are instantiated for
TF32_HEAD_DIMS = (32, 48)
#: the fp32 bias-gradient partials of one backward launch are at most this many bytes (the query tiles are run in chunks beyond it)
_DBP_BUDGET = 2 << 30


@functools.lru_cache(maxsize=1)
def _tf32_ext():
    ensure_cuda_home()
    return load_extension(
        name="augmented_attention_tf32_sm80",
        sources=[str(_dir / "ops_tf32.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", "-gencode=arch=compute_80,code=sm_80", f"-I{_dir}"],
        extra_cflags=["-std=c++17", "-O3"], verbose=False,
    )


def supported_tf32(dtype: torch.dtype, length: int, d: int, heads: int, index: int) -> bool:
    """The TF32 kernels' requirements: sm_80, fp32 operands, head dim 32 or 48, L a multiple of 128."""
    if os.environ.get("MINIWORLD_AUGATTN_SM80", "1") == "0":
        return False
    return dtype is torch.float32 and d % heads == 0 and d // heads in TF32_HEAD_DIMS and length % 128 == 0 and _is_ampere(index)


def _fwd32_schedule(head_dim: int) -> tuple[int, int]:
    """(K | V ring stages, CTAs per SM) of the TF32 forward."""
    return (2, 2) if head_dim == 32 else (2, 1)


def tf32_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, samples: int, length: int, heads: int, head_dim: int, *, sm_scale: float | None = None,
                 prescaled: bool = False, gate: torch.Tensor | None = None, out: torch.Tensor | None = None, lse: torch.Tensor | None = None, kpen: torch.Tensor | None = None,
                 kps: int = 0):
    """``softmax(q k^T / sqrt(hd) + bias) v`` for ``samples`` problems that share the bias, fp32 operands on the TF32 tensor cores: ``q`` / ``k`` / ``v`` token-major fp32 views
    ``[A L, H hd]`` (row strides >= H hd and a multiple of 4), ``bias`` head-major fp32 ``[H, L, L]`` in natural units, masked keys ``-inf``.  Returns ``(o, lse)``: ``o`` fp32
    ``[A L, H hd]``, ``lse`` fp32 ``[A, H, L]`` (the scaled base-2 log-sum-exp); with ``gate`` (a token-major fp32 view like q) ``o = sigmoid(gate) softmax(...) v`` is written
    to ``out`` (default: over ``q``'s view) and ``lse`` is None.  ``prescaled``: q is already scaled by sm_scale log2 e and the bias by log2 e (the fused token DiT's pack).
    ``kpen`` / ``kps`` / ``out`` / ``lse``: as for :func:`plain_forward`."""
    assert head_dim in TF32_HEAD_DIMS, head_dim
    sm_scale = 1.0 / math.sqrt(head_dim) if sm_scale is None else sm_scale
    dev = q.device
    o = torch.empty(samples * length, heads * head_dim, device=dev, dtype=torch.float32) if out is None and gate is None else (q if out is None else out)
    if gate is None:
        lse = torch.empty(samples, heads, length, device=dev, dtype=torch.float32) if lse is None else lse
    pen, kps = _penalty(kpen, kps, length)
    scl, bscl = (1.0, 1.0) if prescaled else (_L2E * sm_scale, _L2E)
    _tf32_ext().attn_fwd_tf32(q, k, v, bias, _NOPEN if gate is None else gate, o, _NOPEN if gate is not None else lse, samples, length, heads, head_dim, q.stride(0), k.stride(0),
                              v.stride(0), 0 if gate is None else gate.stride(0), o.stride(0), scl, bscl, *_fwd32_schedule(head_dim), 0, pen, kps)
    return o, (None if gate is not None else lse)


def _dq32_schedule(samples: int, head_dim: int) -> tuple[int, int, int]:
    """(samples per CTA, K | V ring stages, CTAs per SM) of the TF32 query side: two samples share a bias tile at head dim 32 (smem), one at 48."""
    return (2 if head_dim == 32 and samples % 2 == 0 else 1, 2, 1)


def tf32_backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dob: torch.Tensor, bias: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, samples: int, length: int,
                  heads: int, head_dim: int, *, sm_scale: float | None = None, kpen: torch.Tensor | None = None, kps: int = 0, dq: torch.Tensor | None = None,
                  dk: torch.Tensor | None = None, dv: torch.Tensor | None = None):
    """The backward of :func:`tf32_forward` (without the gate): ``dob`` the output's gradient fp32 ``[A L, H hd]``, ``o`` / ``lse`` the forward's, ``bias`` fp32 natural units.
    Returns ``(dq, dk, dv, db)``: fp32 ``[A L, H hd]`` each (or the views passed) and the bias gradient fp32 ``[H, L, L]`` (the sum over the samples; masked keys 0), summed from fp32
    partials of 1-2 samples in a fixed order: a replay is bit-identical."""
    assert head_dim in TF32_HEAD_DIMS, head_dim
    ext = _tf32_ext()
    dev, m, d = q.device, samples * length, heads * head_dim
    sm_scale = 1.0 / math.sqrt(head_dim) if sm_scale is None else sm_scale
    scl = sm_scale * _L2E
    delta = torch.empty(samples, heads, length, device=dev, dtype=torch.float32)
    ext.attn_delta32(dob, o, delta, samples, length, heads, head_dim)
    dq, dk, dv = (torch.empty(m, d, device=dev, dtype=torch.float32) if t is None else t for t in (dq, dk, dv))
    pen, kps = _penalty(kpen, kps, length)
    rows, nkv, minb = _dq32_schedule(samples, head_dim)
    groups = samples // rows
    tiles = length // 128
    per_tile = groups * heads * 128 * length * 4
    chunk = max(1, min(tiles, _DBP_BUDGET // per_tile))
    dbp = torch.empty(groups, heads, chunk * 128, length, device=dev, dtype=torch.float32)
    db = torch.empty(heads, length, length, device=dev, dtype=torch.float32)
    for qt0 in range(0, tiles, chunk):
        nqt = min(chunk, tiles - qt0)
        buf = dbp if nqt == chunk else torch.empty(groups, heads, nqt * 128, length, device=dev, dtype=torch.float32)
        ext.attn_bwd_dq_tf32(q, k, v, dob, bias, lse, delta, dq, buf, samples, length, heads, head_dim, q.stride(0), k.stride(0), v.stride(0), dob.stride(0), dq.stride(0), scl,
                             _L2E, sm_scale, rows, nkv, minb, 0, pen, kps, qt0, nqt)
        ext.db_reduce32(buf, db, groups, heads, nqt * 128, length, qt0 * 128, 1.0)
    bias_t = torch.empty_like(bias)
    ext.bias_transpose32(bias, bias_t, length)
    ext.attn_bwd_dkv_tf32(q, k, v, dob, bias_t, lse, delta, dk, dv, samples, length, heads, head_dim, q.stride(0), k.stride(0), v.stride(0), dob.stride(0), dk.stride(0), dv.stride(0),
                          scl, _L2E, sm_scale, 2, 1, 0, pen, kps)
    return dq, dk, dv, db
