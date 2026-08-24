
from miniworld_engine.kernels._compile import opaque
# vendored from team-gm origin/exp/miniworld@32e3897 : src/team_gm/modules/kernels/augmented_attention_pair_bias_compute_efficient.py

from miniworld_engine.autotune.configs import configs_for
import torch
import triton
import triton.language as tl

from jaxtyping import Bool, Float

from miniworld_engine.autotune.shape_key import atom_key
from miniworld_engine._typecheck import typecheck


# NOT called from this file any more -- every launch below keys on `atom_key(L)` (see
# autotune/shape_key.py: the key is L, the atom count, and this family is level=atom in
# kernels/registry.csv). Kept only because ``checks/augmented_attention.py`` still imports it.
def get_seq_group(length) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    from miniworld_engine.autotune.buckets import bucket_linear
    return bucket_linear(length)


# HEAD_DIM_PAD is a LAUNCHER value, next_power_of_2(HEAD_DIM), not a CSV axis: head dims here are
# 24 and 48, and a block_ptr block_shape has to be a power of two. Delta = sum_d(o*do) reduces over
# d inside that pad.

# The config list is read once and reused -- by the decorator below and by the split-count
# arithmetic, which has to agree with the BLOCK_M2 the backward actually runs.

def _bwd_min_block_n() -> int:
    """Smallest BLOCK_N the backward can be launched with, read from its own config set.

    `dq_expand` is allocated with `cdiv(L, this)` split slots and `_attn_bwd` writes slot `pid`
    from a `cdiv(L, BLOCK_N)` grid, so this must be the minimum over the configs the autotuner can
    actually pick. Deriving it from anything else under-allocates: a run whose configs reach
    BLOCK_N=16 while this said 64 gets 16 blocks storing into a 4-slot buffer, which walks off the
    end of `dq_expand` into the next allocation -- observed as NaN in `bias`, an argument this
    kernel only reads, and as an illegal memory access at tile (32, 16).
    """
    blocks = [c.kwargs["BLOCK_M2"] for c in configs_for("augmented_attention_bwd_split_triton")
              if "BLOCK_M2" in c.kwargs]
    if not blocks:
        raise RuntimeError(
            "augmented_attention_bwd has no configs: dq_expand's split count is undefined. "
            "Select a config directory that covers this op.")
    return min(blocks)


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    k_ptr,
    v_ptr,
    start_m,
    start_d,
    qk_scale,
    b_ptr,
    mask_ptr,
    stride_kn,
    stride_vn,
    stride_bm,
    stride_bn,
    BLOCK_M1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    N_CTX,
):
    tl.static_assert(HEAD_DIM_PAD >= HEAD_DIM)
    offset_m = start_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    offset_k = start_d * HEAD_DIM_PAD + tl.arange(0, HEAD_DIM_PAD)
    for start_n in range(0, N_CTX, BLOCK_M2):
        offset_n = start_n + tl.arange(0, BLOCK_M2)
        bias_mask = (offset_m[:, None] < N_CTX) & (offset_n[None, :] < N_CTX)
        bias_val = tl.load(b_ptr, mask=bias_mask, other=float("-inf"))

        key_mask = tl.load(mask_ptr + offset_n, mask=offset_n < N_CTX, other=False)
        bias_val = tl.where(key_mask[None, :], bias_val, float("-inf"))

        k_mask = (offset_n[None, :] < N_CTX) & (offset_k[:, None] < HEAD_DIM)
        v_mask = (offset_n[:, None] < N_CTX) & (offset_k[None, :] < HEAD_DIM)
        k = tl.load(k_ptr, mask=k_mask, other=0.0)
        v = tl.load(v_ptr, mask=v_mask, other=0.0)

        qk = tl.dot(q, k) + bias_val / (qk_scale / 1.44269504)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        m_ij = tl.maximum(m_ij, -1e38)
        qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)

        m_i = m_ij
        b_ptr += BLOCK_M2 * stride_bn
        k_ptr += BLOCK_M2 * stride_kn
        v_ptr += BLOCK_M2 * stride_vn
    return acc, l_i, m_i




# AUTOTUNE KEY: `A` and `B` are constexpr but stay OUT of the key. `A` (the augmentation count) is
# never read by this body -- the grid extent A*B*H is built at the launcher from python ints -- so
# the generated code is the same for every A and a key entry could only split the cache. `B` appears
# only in the scalar index decomposition (`off_z // B`, `off_z % B`): no branch, no change to the
# tile's work shape, and it varies per run, so keying it would fragment every bucket by batch size
# for two integer ops. What does set the work shape is already keyed -- HEAD_DIM (and with it
# HEAD_DIM_PAD = next_power_of_2(HEAD_DIM)), H (also the q/k/v row-stride multiplier H*D), and
# shape_key (the L bucket).
@triton.autotune(configs=configs_for("augmented_attention_fwd_triton"),
                 key=['shape_key', 'H', 'HEAD_DIM'])
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    Bias,
    Mask,
    sm_scale,
    M,
    Out,
    stride_qz,
    stride_qm,
    stride_qh,
    stride_qk,
    stride_kz,
    stride_kn,
    stride_kh,
    stride_kk,
    stride_vz,
    stride_vn,
    stride_vh,
    stride_vk,
    stride_oz,
    stride_om,
    stride_oh,
    stride_on,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_maska,
    stride_maskb,
    A: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
    shape_key,
):
    tl.static_assert(HEAD_DIM_PAD >= HEAD_DIM)
    start_m = tl.program_id(0).to(tl.int64)
    off_hz = tl.program_id(1).to(tl.int64)
    start_d = tl.program_id(2).to(tl.int64)
    off_z = off_hz // H
    off_h = off_hz % H
    off_a = off_z // B
    off_b = off_z % B
    qvk_offset = off_z * stride_qz + off_h * stride_qh
    bias_offset = off_b * stride_bz + off_h * stride_bh
    mask_offset = off_a * stride_maska + off_b * stride_maskb

    offset_m = start_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    offset_n = tl.arange(0, BLOCK_M2)
    offset_k = start_d * HEAD_DIM_PAD + tl.arange(0, HEAD_DIM_PAD)

    q_ptr = (
        Q + qvk_offset + offset_m[:, None] * stride_qm + offset_k[None, :] * stride_qk
    )
    k_ptr = (
        K + qvk_offset + offset_n[None, :] * stride_kn + offset_k[:, None] * stride_kk
    )
    v_ptr = (
        V + qvk_offset + offset_n[:, None] * stride_vn + offset_k[None, :] * stride_vk
    )
    o_ptr = (
        Out + qvk_offset + offset_m[:, None] * stride_om + offset_k[None, :] * stride_on
    )
    bias_ptr = (
        Bias
        + bias_offset
        + offset_m[:, None] * stride_bm
        + offset_n[None, :] * stride_bn
    )

    m_i = tl.zeros([BLOCK_M1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M1], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M1, HEAD_DIM_PAD], dtype=tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.44269504

    q_mask = (offset_m[:, None] < N_CTX) & (offset_k[None, :] < HEAD_DIM)
    q = tl.load(q_ptr, mask=q_mask, other=0.0)

    mask_ptr = Mask + mask_offset

    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        k_ptr,
        v_ptr,
        start_m,
        start_d,
        qk_scale,
        bias_ptr,
        mask_ptr,
        stride_kn,
        stride_vn,
        stride_bm,
        stride_bn,
        BLOCK_M1,
        BLOCK_M2,
        HEAD_DIM_PAD,
        HEAD_DIM,
        N_CTX,
    )

    l_i = tl.maximum(l_i, 1e-30)  # guard fully-masked rows: finite 0 output, no 0/0 NaN
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + start_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)

    tl.store(m_ptrs, m_i, mask=offset_m < N_CTX)
    out_mask = (offset_m[:, None] < N_CTX) & (offset_k[None, :] < HEAD_DIM)
    tl.store(o_ptr, acc.to(Out.type.element_ty), mask=out_mask)




# AUTOTUNE KEY: `A`/`B` are not read by this body at all (see _attn_fwd above), and the four
# constexpr strides stay out too. Both launchers pass the strides of a contiguous (A, B, L, H, D)
# q, so stride_k == 1 always; stride_h == HEAD_DIM and stride_m == H*HEAD_DIM are products of two
# entries already in the key; and stride_z == L*H*HEAD_DIM would key on the exact L instead of its
# bucket -- one cache entry per sequence length, which is what shape_key exists to prevent. None of
# them changes the tile or the loop trip count: they only shift each program's base pointer.
@triton.autotune(configs=configs_for("augmented_attention_bwd_pre_triton"),
                 key=['shape_key', 'H', 'HEAD_DIM'])
@triton.jit
def _attn_bwd_preprocess(
    O,
    DO,
    Delta,
    A: tl.constexpr,
    B: tl.constexpr,
    N_CTX,
    stride_z: tl.constexpr,
    stride_m: tl.constexpr,
    stride_h: tl.constexpr,
    stride_k: tl.constexpr,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
    shape_key,
):
    tl.static_assert(HEAD_DIM_PAD >= HEAD_DIM)
    off_m = tl.program_id(0).to(tl.int64) * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    off_z = tl.program_id(1).to(tl.int64)
    off_h = tl.program_id(2).to(tl.int64)

    offset = off_z * stride_z + off_h * stride_h
    mask_delta = off_m < N_CTX

    # delta[m] = sum_d o[m,d]*do[m,d] -- the d axis is a reduction, so HEAD_DIM_PAD tiles it and the
    # partials accumulate in fp32 (summation is associative: exact for any tile).
    delta = tl.zeros([BLOCK_M1], dtype=tl.float32)
    for d0 in range(0, HEAD_DIM, HEAD_DIM_PAD):
        off_k = d0 + tl.arange(0, HEAD_DIM_PAD)
        o_ptr = O + offset + off_m[:, None] * stride_m + off_k[None, :] * stride_k
        do_ptr = DO + offset + off_m[:, None] * stride_m + off_k[None, :] * stride_k
        mask_m = (off_m[:, None] < N_CTX) & (off_k[None, :] < HEAD_DIM)
        o = tl.load(o_ptr, mask=mask_m, other=0.0).to(tl.float32)
        do = tl.load(do_ptr, mask=mask_m, other=0.0).to(tl.float32)
        delta += tl.sum(o * do, axis=1)
    delta_ptr = Delta + off_z * H * N_CTX + off_h * N_CTX + off_m
    tl.store(delta_ptr, delta, mask=mask_delta)


@triton.jit
def _attn_bwd_dqdkdv(
    dk,
    dv,
    DBias,
    Q,
    k,
    v,
    Bias,
    mask_ptr,
    qk_scale,
    DO,
    DQ,
    M,
    D,
    stride_tok,
    stride_d,
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_n,
    start_m,
    stride_bm,
    stride_bn,
):
    tl.static_assert(HEAD_DIM_PAD >= HEAD_DIM)
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_M2)
    offs_k = tl.arange(0, HEAD_DIM_PAD)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dqT_ptrs = DQ + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    biasT_ptrs = Bias + offs_m[None, :] * stride_bm + offs_n[:, None] * stride_bn
    dbiasT_ptrs = DBias + offs_m[None, :] * stride_bm + offs_n[:, None] * stride_bn
    m_ptrs = M + offs_m
    d_ptrs = D + offs_m

    key_mask = tl.load(mask_ptr + offs_n, mask=offs_n < N_CTX, other=False)

    for _start_m in range(0, N_CTX, BLOCK_M1):
        offs_m = _start_m + tl.arange(0, BLOCK_M1)
        qT_mask = (offs_m[None, :] < N_CTX) & (offs_k[:, None] < HEAD_DIM)
        do_mask = (offs_m[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM)
        qT = tl.load(qT_ptrs, mask=qT_mask, other=0.0)

        bias_mask = (offs_m[None, :] < N_CTX) & (offs_n[:, None] < N_CTX)
        biasT = tl.load(biasT_ptrs, mask=bias_mask, other=float("-inf"))
        biasT = tl.where(key_mask[:, None], biasT, float("-inf"))
        do = tl.load(do_ptrs, mask=do_mask, other=0.0)
        m = tl.load(m_ptrs, mask=offs_m < N_CTX, other=0.0)
        Di = tl.load(d_ptrs, mask=offs_m < N_CTX, other=0.0)

        qkT = tl.dot(k, qT) + biasT / (qk_scale / 1.44269504)
        m_safe = tl.maximum(m, -1e38)
        qkT = qkT * qk_scale - m_safe[None, :]

        pT = tl.math.exp2(qkT)
        dv = tl.dot(pT.to(do.dtype), do, dv)

        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - Di[None, :])

        tl.store(dbiasT_ptrs, dsT, mask=bias_mask)

        dsT = dsT.to(qT.dtype)
        dk = tl.dot(dsT, tl.trans(qT), dk)

        # store, not atomic_add: each split writes its own slot.
        dqT_to_add = tl.dot(tl.trans(k), dsT) * (qk_scale / 1.44269504)
        tl.store(dqT_ptrs, dqT_to_add, mask=qT_mask)

        qT_ptrs += BLOCK_M1 * stride_tok
        do_ptrs += BLOCK_M1 * stride_tok
        dqT_ptrs += BLOCK_M1 * stride_tok
        biasT_ptrs += BLOCK_M1 * stride_bm
        dbiasT_ptrs += BLOCK_M1 * stride_bm
        m_ptrs += BLOCK_M1
        d_ptrs += BLOCK_M1
    return dk, dv




# AUTOTUNE KEY: `A`/`B` out for the same reason as the forward -- `A` is unread here, and `B` only
# folds into the constant M_offset stride (B*H*N_CTX).
@triton.autotune(configs=configs_for("augmented_attention_bwd_split_triton"),
                 key=['shape_key', 'H', 'HEAD_DIM'],
                 reset_to_zero=['DQ', 'DBias'])
@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    Bias,
    Mask,
    sm_scale,
    DO,
    DQ,
    DK,
    DV,
    DBias,
    M,
    D,
    stride_a,
    stride_z,
    stride_n,
    stride_h,
    stride_d,
    stride_dq_split,  # stride of dq_expand's split dimension
    bias_stride_a,
    bias_stride_z,
    bias_stride_h,
    bias_stride_m,
    bias_stride_n,
    stride_maska,
    stride_maskb,
    A: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
    shape_key,
):
    tl.static_assert(HEAD_DIM_PAD >= HEAD_DIM)
    pid = tl.program_id(0).to(tl.int64)
    aid = tl.program_id(1).to(tl.int64)
    bhid = tl.program_id(2).to(tl.int64)
    bid = bhid // H
    hid = bhid % H
    qkv_offset = aid * stride_a + bid * stride_z + hid * stride_h
    M_offset = aid * (B * H * N_CTX) + bhid * N_CTX

    Q += qkv_offset
    K += qkv_offset
    V += qkv_offset
    DO += qkv_offset
    # DQ carries a split offset so each pid owns an independent slot.
    DQ += pid * stride_dq_split + qkv_offset
    DK += qkv_offset
    DV += qkv_offset
    M += M_offset
    D += M_offset

    offset_bias = bid * bias_stride_z + hid * bias_stride_h
    Bias += offset_bias
    DBias += aid * bias_stride_a + offset_bias

    mask_offset = aid * stride_maska + bid * stride_maskb
    mask_ptr = Mask + mask_offset

    offs_k = tl.arange(0, HEAD_DIM_PAD)
    start_n = pid * BLOCK_M2
    offs_n = start_n + tl.arange(0, BLOCK_M2)

    dv = tl.zeros([BLOCK_M2, HEAD_DIM_PAD], dtype=tl.float32)
    dk = tl.zeros([BLOCK_M2, HEAD_DIM_PAD], dtype=tl.float32)

    k_ptr = K + offs_n[:, None] * stride_n + offs_k[None, :] * stride_d
    v_ptr = V + offs_n[:, None] * stride_n + offs_k[None, :] * stride_d

    k = tl.load(
        k_ptr,
        mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM),
        other=0.0,
    )
    v = tl.load(
        v_ptr,
        mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM),
        other=0.0,
    )

    qk_scale = sm_scale * 1.44269504
    dk, dv = _attn_bwd_dqdkdv(
        dk,
        dv,
        DBias,
        Q,
        k,
        v,
        Bias,
        mask_ptr,
        qk_scale,
        DO,
        DQ,
        M,
        D,
        stride_n,
        stride_d,
        H,
        N_CTX,
        BLOCK_M1,
        BLOCK_M2,
        HEAD_DIM_PAD,
        HEAD_DIM,
        start_n,
        start_m=0,
        stride_bm=bias_stride_m,
        stride_bn=bias_stride_n,
    )

    dv_ptrs = DV + offs_n[:, None] * stride_n + offs_k[None, :] * stride_d
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_n + offs_k[None, :] * stride_d

    tl.store(
        dv_ptrs,
        dv.to(dv_ptrs.dtype.element_ty),
        mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM),
    )
    tl.store(
        dk_ptrs,
        dk.to(dk_ptrs.dtype.element_ty),
        mask=(offs_n[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM),
    )


# ── dq_expand reduction ──────────────────────────────────────────────────
# The grid is a meta-lambda over the same BLOCK_E the kernel compiles with: `dq` is a
# `torch.empty` and a grid that does not cover N_ELEM leaves part of it undefined.
#
# This kernel has produced a NaN input gradient before (forward clean, backward NaN) and the cause
# was never established -- two explanations written here in turn did not survive inspection. The
# form below verifies clean at BLOCK_E 64 / 1024 / 4096 through a real autograd backward. If NaNs
# reappear, reproduce the failure before trusting any explanation.


# ``_dq_reduce`` used to bucket its own flat element count A*B*L*H*D through the helper below,
# because that count is above 1e5 for every real shape and saturated the old LINEAR edge set. It now
# keys on `atom_key(L)` like every other kernel in this family: L is what the config was tuned
# against, and the element count is a function of it (times fixed A/H/D), so nothing is lost by
# keying the cause instead of the product -- and the family stops holding two bucket spaces at once.
#
# NOT called from this file any more. Kept only because ``checks/augmented_attention.py`` still imports it.
def get_elem_group(n_elem) -> int:
    """Bucket a flat ELEMENT count (canonical autotune.buckets). Superseded by `atom_key(L)`."""
    from miniworld_engine.autotune.buckets import bucket_mixed
    return bucket_mixed(n_elem)


# AUTOTUNE KEY: ['shape_key'] -- was ['N_ELEM'], the raw element count A*B*L*H*D, so this
# backward-path kernel re-swept its whole config space at every new sequence length seen in a
# training run. `N_ELEM` is still the body's bound; `shape_key` is its bucket and is never read
# by the kernel, so the generated code -- and the gradient -- are unchanged.
@triton.autotune(configs=configs_for("augmented_attention_bwd_reduce_triton"), key=['shape_key'])
@triton.jit
def _dq_reduce(
    DQ_Expand,  # (num_splits, A, B, L, H, D)
    DQ_Out,  # (A, B, L, H, D)
    num_splits,
    stride_split,
    stride_inner,  # element stride, not the A*B*L*H*D span
    N_ELEM,  # elements in one split (= A*B*L*H*D)
    BLOCK_E: tl.constexpr,
    shape_key,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offs < N_ELEM

    stride_split = stride_split.to(tl.int64)
    acc = tl.zeros([BLOCK_E], dtype=tl.float32)
    for s in range(num_splits):
        ptr = DQ_Expand + s * stride_split + offs
        val = tl.load(ptr, mask=mask, other=0.0)
        acc += val

    out_ptr = DQ_Out + offs
    tl.store(out_ptr, acc, mask=mask)


def _aa_fwd_fake(q, k, v, bias, mask, shape_key):
    """``(out, m)``: ``out`` like ``q``; ``m`` is the per-row logsumexp, ``(A, B, H, L)`` and
    fp32 while the activations are bf16 -- the backward recomputes ``p = exp2(qk*scale - m)``
    from it, so its digits land in an exponent.
    """
    A, B, L, H, D = q.shape
    return torch.empty_like(q), q.new_empty((A, B, H, L), dtype=torch.float32)


@opaque(fake=_aa_fwd_fake, name="augmented_attention_fwd")
def _aa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    shape_key: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The attention launch -> ``(out, m)``; ``m`` is the per-row logsumexp the backward reuses.

    Split out of ``TritonAugmentedAttentionFunction.forward`` so the bias permute, the default
    mask, the contiguous() calls and ``save_for_backward`` stay traceable -- see
    ``kernels._compile``. Every tensor arrives already permuted and contiguous, so this only
    flattens (A, B) into the kernel's batch axis and launches.
    """
    A, B, L, H, D = q.shape
    sm_scale = D**-0.5
    out = torch.empty_like(q)
    m = torch.empty(A, B, H, L, device=q.device, dtype=torch.float32)
    # The kernel wants (A*B) as one batch axis; the caller keeps (A, B) split.
    qf, kf, vf, outf = (x.view(A * B, L, H, D) for x in (q, k, v, out))

    grid = lambda META: (
        triton.cdiv(L, META["BLOCK_M1"]),
        A * B * H,
        triton.cdiv(D, META["HEAD_DIM_PAD"]),
    )
    _attn_fwd[grid](
        qf,
        kf,
        vf,
        bias,
        mask,
        sm_scale,
        m,
        outf,
        *qf.stride(),
        *kf.stride(),
        *vf.stride(),
        *outf.stride(),
        *bias.stride(),
        *mask.stride()[:2],
        A,
        B,
        H,
        L,
        D,
        HEAD_DIM_PAD=triton.next_power_of_2(D),
        shape_key=shape_key,
    )
    return out, m


def _aa_bwd_fake(dy, q, k, v, bias, mask, o, m, shape_key):
    """``(dq, dk, dv, dbias_raw)``: ``dq`` comes back fp32 -- it is summed out of the fp32
    per-split ``dq_expand`` buffer -- while ``dk``/``dv`` keep their inputs' dtype.
    ``dbias_raw`` is the UNREDUCED ``(A, B, H, L, L)`` fp32 accumulator: the sum over A and the
    permute to the caller's ``(B, L, L, H)`` layout stay outside the op, where they fuse.
    """
    A, B, L, H, D = q.shape
    return (
        q.new_empty((A, B, L, H, D), dtype=torch.float32),
        torch.empty_like(k),
        torch.empty_like(v),
        q.new_empty((A, B, H, L, L), dtype=torch.float32),
    )


@opaque(fake=_aa_bwd_fake, name="augmented_attention_bwd")
def _aa_bwd(
    dy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    o: torch.Tensor,
    m: torch.Tensor,
    shape_key: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three backward launches -> ``(dq, dk, dv, dbias_raw)``.

    ``dbias_raw`` is the UNREDUCED ``(A, B, H, L, L)`` accumulator: the sum over A and the permute
    back to the caller's ``(B, L, L, H)`` layout are plain torch, so they are left to the caller
    where the compiler can fuse them instead of being buried in an opaque node.
    """
    A, B, L, H, D = q.shape
    sm_scale = D**-0.5
    delta = torch.empty_like(m)

    grid = lambda META: (triton.cdiv(L, META["BLOCK_M1"]), A * B, H)
    _attn_bwd_preprocess[grid](
        o,
        dy,
        delta,
        A,
        B,
        L,
        q.stride(1),
        q.stride(2),
        q.stride(3),
        q.stride(4),
        H,
        D,
        shape_key=atom_key(L),
        HEAD_DIM_PAD=triton.next_power_of_2(D),
    )

    # dq_expand is (num_splits, A, B, L, H, D): one slot per BLOCK_M2 block.
    num_splits = triton.cdiv(L, _bwd_min_block_n())
    dq_expand = torch.zeros(
        int(num_splits), A, B, L, H, D, device=q.device, dtype=torch.float32
    )

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dbias = torch.zeros(A, B, H, L, L, device=q.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(L, META["BLOCK_M2"]),
        A,
        B * H,
    )
    _attn_bwd[grid](
        q,
        k,
        v,
        bias,
        mask,
        sm_scale,
        dy,
        dq_expand,  # DQ slot buffer
        dk,
        dv,
        dbias,
        m,
        delta,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        q.stride(4),
        dq_expand.stride(0),  # stride_dq_split
        dbias.stride(0),
        dbias.stride(1),
        dbias.stride(2),
        dbias.stride(3),
        dbias.stride(4),
        *mask.stride()[:2],
        A,
        B,
        H,
        L,
        D,
        HEAD_DIM_PAD=triton.next_power_of_2(D),
        shape_key=atom_key(L),
    )

    # sum the splits into the final dq
    dq = torch.empty(A, B, L, H, D, device=q.device, dtype=torch.float32)
    n_elem = A * B * L * H * D
    # META-lambda grid: BLOCK_SIZE is a tuned constexpr, so the launch geometry MUST be
    # derived from the config the kernel is compiled with. A grid pinned to a different
    # block size leaves part of the (torch.empty) dq unwritten -- see the note on _dq_reduce.
    grid_reduce = lambda META: (triton.cdiv(n_elem, META["BLOCK_E"]),)  # noqa: E731
    _dq_reduce[grid_reduce](
        dq_expand,
        dq,
        int(num_splits),
        dq_expand.stride(0),
        1,  # element stride (contiguous)
        n_elem,
        shape_key=atom_key(L),
    )

    return dq, dk, dv, dbias


class TritonAugmentedAttentionFunction(torch.autograd.Function):
    @typecheck
    @staticmethod
    def forward(
        ctx,
        q: Float[torch.Tensor, "A B L H D"],
        k: Float[torch.Tensor, "A B L H D"],
        v: Float[torch.Tensor, "A B L H D"],
        bias: Float[torch.Tensor, "B L L H"],
        mask: Bool[torch.Tensor, "A B L"] | None = None,
    ) -> Float[torch.Tensor, "A B L H D"]:
        A, B, L, H, D = q.shape
        if D > 64:
            msg = f"Only support HEAD_DIM <= 64, but got {D}."
            raise ValueError(msg)

        bias = bias.permute(0, 3, 1, 2)
        q, k, v, bias = [x.contiguous() for x in (q, k, v, bias)]

        if mask is None:
            mask = torch.ones(A, B, L, dtype=torch.bool, device=q.device)
        mask = mask.contiguous()

        out, m = _aa_fwd(q, k, v, bias, mask, atom_key(L))

        ctx.save_for_backward(q, k, v, bias, mask, out, m)
        return out

    @staticmethod
    def backward(ctx, *grad_output: torch.Tensor):
        (dy,) = grad_output
        q, k, v, bias, mask, o, m = ctx.saved_tensors
        if dy.dtype != q.dtype:
            dy = dy.to(q.dtype)

        dq, dk, dv, dbias = _aa_bwd(
            dy.contiguous(), q, k, v, bias, mask, o, m, atom_key(q.shape[2]),
        )
        # Reduce over A and restore the caller's (B, L, L, H) bias layout OUTSIDE the op, so these
        # stay in the graph and fuse with whatever consumes dbias.
        dbias = dbias.sum(dim=0).permute(0, 2, 3, 1).contiguous()
        return dq, dk, dv, dbias, None   # mask takes no gradient


triton_augmented_attention_pair_bias = TritonAugmentedAttentionFunction.apply
