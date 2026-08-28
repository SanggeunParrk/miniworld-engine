
from miniworld_engine.kernels._compile import opaque
# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/augmented_attention_pair_bias.py

from miniworld_engine.autotune.configs import configs_for

# The forward and the delta preprocess here were byte-for-byte copies of main.py's (bitwise
# equal on M/Out and on Delta -- see .bench/eq_*.out). Only the backward differs: this file
# accumulates dq/dbias atomically instead of into a per-program expansion buffer.
from .main import _attn_bwd_preprocess, _attn_fwd
import torch
import triton
import triton.language as tl

from jaxtyping import Bool, Float

from miniworld_engine._typecheck import typecheck
from miniworld_engine.autotune.shape_key import atom_key


# HEAD_DIM_PAD was a launch constant (next_power_of_2(HEAD_DIM)). delta = sum_d(o*do) is a plain
# reduction over d, so it tiles with an accumulating loop and HEAD_DIM_PAD joins the sweep.


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

    # Load key mask once (for rows n in transposed view)
    key_mask = tl.load(mask_ptr + offs_n, mask=offs_n < N_CTX, other=False)

    for _start_m in range(0, N_CTX, BLOCK_M1):
        offs_m = _start_m + tl.arange(0, BLOCK_M1)
        qT_mask = (offs_m[None, :] < N_CTX) & (offs_k[:, None] < HEAD_DIM)
        do_mask = (offs_m[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM)
        qT = tl.load(qT_ptrs, mask=qT_mask, other=0.0)

        bias_mask = (offs_m[None, :] < N_CTX) & (offs_n[:, None] < N_CTX)
        biasT = tl.load(biasT_ptrs, mask=bias_mask, other=float("-inf"))
        # Apply attention mask (key_mask is for n dimension = rows in transposed view)
        biasT = tl.where(key_mask[:, None], biasT, float("-inf"))
        do = tl.load(do_ptrs, mask=do_mask, other=0.0)
        m = tl.load(m_ptrs, mask=offs_m < N_CTX, other=0.0)
        Di = tl.load(d_ptrs, mask=offs_m < N_CTX, other=0.0)

        qkT = tl.dot(k, qT) + biasT / (qk_scale / 1.44269504)
        m_safe = tl.maximum(m, -1e38)  # Prevent -inf to avoid NaN in subtraction
        qkT = qkT * qk_scale - m_safe[None, :]

        pT = tl.math.exp2(qkT)
        dv = tl.dot(pT.to(do.dtype), do, dv)

        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - Di[None, :])

        tl.atomic_add(dbiasT_ptrs, dsT, mask=bias_mask)

        dsT = dsT.to(qT.dtype)
        dk = tl.dot(dsT, tl.trans(qT), dk)
        dqT_to_add = tl.dot(tl.trans(k), dsT) * (qk_scale / 1.44269504)
        tl.atomic_add(dqT_ptrs, dqT_to_add, qT_mask)

        qT_ptrs += BLOCK_M1 * stride_tok
        do_ptrs += BLOCK_M1 * stride_tok
        dqT_ptrs += BLOCK_M1 * stride_tok
        biasT_ptrs += BLOCK_M1 * stride_bm
        dbiasT_ptrs += BLOCK_M1 * stride_bm
        m_ptrs += BLOCK_M1
        d_ptrs += BLOCK_M1
    return dk, dv




# AUTOTUNE KEY: same as main.py's split backward -- `A` is never read by this body and `B` only
# folds into the constant M_offset stride (B*H*N_CTX), so neither can change which config wins;
# keying them would partition the cache per augmentation count and per batch size.
@triton.autotune(configs=configs_for("augmented_attention_bwd_atomic_triton"),
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
    bias_stride_z,
    bias_stride_m,
    bias_stride_h,
    bias_stride_n,
    stride_maska,
    stride_maskb,
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
    DQ += qkv_offset
    DK += qkv_offset
    DV += qkv_offset
    M += M_offset
    D += M_offset

    offset_bias = bid * bias_stride_z + hid * bias_stride_h
    Bias += offset_bias
    DBias += offset_bias

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

    qk_scale = sm_scale * 1.44269504  # 1/log(2)
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


def _memeff_fwd_fake(q, k, v, bias, mask, shape_key):
    """``(out, m)``: ``out`` like ``q``; ``m`` is the ``(A, B, H, L)`` per-row logsumexp, fp32
    while the activations are bf16 -- the backward rebuilds ``p`` from it.
    """
    A, B, L, H, D = q.shape
    return torch.empty_like(q), q.new_empty((A, B, H, L), dtype=torch.float32)


@opaque(fake=_memeff_fwd_fake, name="augmented_attention_memeff_fwd")
def _memeff_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    shape_key: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The attention launch -> ``(out, m)``. ``bias`` arrives already permuted to (B, L, H, L).

    Split out of ``TritonAugmentedAttentionFunction.forward`` (the memory-efficient variant) so
    the autocast casts, the permute, the default mask and ``save_for_backward`` stay traceable --
    see ``kernels._compile``.
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
    # bias is (B, L, H, L) here but main.py's kernel reads stride_bz/bh/bm/bn in that order
    _sbz, _sbm, _sbh, _sbn = bias.stride()
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
        _sbz, _sbh, _sbm, _sbn,
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


def _memeff_bwd_fake(grad_output, q, k, v, bias, mask, o, m, shape_key):
    """``(dq, dk, dv, dbias)``: ``dq`` is fp32 because the backward accumulates into it with
    atomic adds; ``dk``/``dv`` match their inputs. ``dbias`` is the fp32 ``(B, L, H, L)`` atomic
    accumulator -- already summed over A, but still in the kernel's bias frame; the permute back
    to ``(B, L, L, H)`` is left to the caller.
    """
    A, B, L, H, D = q.shape
    return (
        q.new_empty((A, B, L, H, D), dtype=torch.float32),
        torch.empty_like(k),
        torch.empty_like(v),
        q.new_empty((B, L, H, L), dtype=torch.float32),
    )


@opaque(fake=_memeff_bwd_fake, name="augmented_attention_memeff_bwd")
def _memeff_bwd(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    o: torch.Tensor,
    m: torch.Tensor,
    shape_key: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The two backward launches -> ``(dq, dk, dv, dbias)``, ``dbias`` still in (B, L, H, L).

    The permute back to the caller's (B, L, L, H) layout is plain torch and stays outside, where
    the compiler can fuse it.
    """
    A, B, L, H, D = q.shape
    sm_scale = D**-0.5
    delta = torch.empty_like(m)  # (A, B, H, L)

    grid = lambda META: (triton.cdiv(L, META["BLOCK_M1"]), A * B, H)
    _attn_bwd_preprocess[grid](
        o,
        grad_output,
        delta,
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

    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dbias = torch.zeros(B, L, H, L, device=q.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(L, META["BLOCK_M2"]),
        A,
        B * H,
    )  # Suppose that D <= HEAD_DIM_PAD
    _attn_bwd[grid](
        q,
        k,
        v,
        bias,
        mask,
        sm_scale,
        grad_output,
        dq,
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
        bias.stride(0),
        bias.stride(1),
        bias.stride(2),
        bias.stride(3),
        *mask.stride()[:2],
        B,
        H,
        L,
        D,
        HEAD_DIM_PAD=triton.next_power_of_2(D),
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
        if D > 64:  # noqa: PLR2004
            msg = f"Only support HEAD_DIM <= 64, but got {D}."
            raise ValueError(msg)

        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
            q = q.to(dtype)
            k = k.to(dtype)
            v = v.to(dtype)
            bias = bias.to(dtype)
        bias = bias.permute(0, 1, 3, 2)  # (B, L, L, H) -> (B, L, H, L)
        q, k, v, bias = [x.contiguous() for x in (q, k, v, bias)]

        if mask is None:
            mask = torch.ones(A, B, L, dtype=torch.bool, device=q.device)
        mask = mask.contiguous()

        out, m = _memeff_fwd(q, k, v, bias, mask, atom_key(L))

        ctx.save_for_backward(q, k, v, bias, mask, out, m)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, bias, mask, o, m = ctx.saved_tensors
        if grad_output.dtype != q.dtype:
            grad_output = grad_output.to(q.dtype)

        dq, dk, dv, dbias = _memeff_bwd(
            grad_output.contiguous(), q, k, v, bias, mask, o, m, atom_key(q.shape[2]),
        )
        dbias = dbias.permute(0, 1, 3, 2).contiguous()  # (B, L, H, L) -> (B, L, L, H)
        return dq, dk, dv, dbias, None   # mask takes no gradient


triton_augmented_attention_pair_bias = TritonAugmentedAttentionFunction.apply
