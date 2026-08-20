# vendored from team-gm origin/perf/trimul@3fbb02b : src/team_gm/modules/kernels/fused_triangle_mul_dtv1.py
"""dt-v1: Fused Triton triangle multiplicative update — fully optimized kernel.

Architecture overview
---------------------
The triangle multiplicative update computes:

    x_normed = LayerNorm(x)
    a        = sigmoid(x_normed @ g_in.T) * (x_normed @ p_in.T)   [input path]
    out      = einsum("bikd,bjkd->bijd", a[:D], a[D:])              [outgoing]
    out_n    = LayerNorm(out)
    y        = sigmoid(x_normed @ g_out.T) * (out_n @ p_out.T)     [output path]

Forward optimizations
---------------------
1. **Fused input gated GEMM** — a single Triton kernel computes
   ``sigmoid(xn @ wg.T) * (xn @ wp.T)`` fusing:
   - Layer-norm (via cuequivariance ``layer_norm_transpose`` op)
   - Dual GEMM (gate + projection paths share one K-loop)
   - Sigmoid + elementwise multiply
   - Optional mask broadcast
   This avoids materialising the LN output and the two separate GEMM results.

2. **Transpose-out layout** — the input GEMM writes ``(2D, M)`` directly,
   which maps to ``(D, B, I, J)`` per side with a free ``.view()``.  The
   triangle contraction therefore never requires an explicit transpose.

3. **Single-pass dual-accumulator output GEMM** — gate and projection
   K-loops run together inside one Triton CTA with two accumulators
   ``(acc_g, acc_p)``.  On Hopper, TMA pipelines all four tensor loads
   (x_normed, x_out, w_gate, w_proj) concurrently, fully hiding HBM
   latency in one pass.

4. **Occupancy-tuned tile configs** — extended autotuning grids for both
   input and output kernels including:
   - ``maxnreg=128/96`` for Hopper register-file occupancy.
   - ``num_stages=3/4/5`` for Hopper async-copy TMA pipeline overlap.
   - ``BLOCK_M1=64, BLOCK_K=32, num_warps=4`` small-tile configs that
     reach up to 6 CTAs/SM for the input kernel and 4 CTAs/SM for the
     output kernel at short and medium sequence lengths.
   - **Note**: ``maxnreg ≤ 80`` is forbidden for the dual-accumulator
     kernels (need ≈84 registers for the two BLOCK_M1×BLOCK_N tiles);
     use ``maxnreg=128`` or none.

Backward optimizations
----------------------
5. **Saved sigmoid** — forward saves ``sig_m = sigmoid(xn @ wg.T) * mask``
   alongside ``ab``; backward reuses it directly, eliminating the
   recomputation of sigmoid that PyTorch autograd would otherwise trigger.

6. **Combined 4D×2D input-path backward GEMMs** — instead of two separate
   ``(D, M) @ (M, K)`` weight-gradient GEMMs, a single ``(4D, M) @ (M, K)``
   GEMM is issued and then split.  Likewise the input gradient uses one
   ``(M, 4D) @ (4D, K)`` GEMM instead of two ``(M, D) @ (D, K)`` GEMMs.
   This halves GEMM kernel launches and lets cuBLAS pick a larger tile.

7. **Pre-saved w_combined** — ``[w_gate; w_proj]`` (shape 4D×K, ≈1 MB at
   D=128) is computed once in forward and stored in the autograd context,
   avoiding a ``torch.cat`` allocation on every backward step.

8. **Custom Triton elementwise backward** — a fused Triton kernel computes
   ``d_gate = grad * ab * (1 − sig_m)`` and ``d_proj = grad * sig_m`` in one
   pass over HBM, with BLOCK=2048 (16 elems/thread) for reduced launch
   overhead and better vectorised access patterns.

9. **maxnreg tuning** — ``maxnreg=128`` caps register usage to 32 768
   regs/CTA on H100, enabling 2 CTAs/SM (vs typically 1 without the cap).
   ``maxnreg=96`` with smaller blocks allows up to 5 CTAs/SM.

Target: Hopper (H100/H200), fp32 or bf16, B=1, L >= 256.
"""

from __future__ import annotations

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.autotune.configs import configs_for

import torch
import triton
import triton.language as tl
from cuequivariance_ops.triton import Layout

from miniworld_engine.autotune import tensor_dtype_of
from miniworld_engine.autotune.buckets import bucket_mixed as _bucket
# Shared helpers
# ---------------------------------------------------------------------------


def _output_layer_norm_transpose(x, weight, bias, b, n, d, eps):
    """Layer-norm over the D dimension of a (D, B, N) tensor.

    Accepts x shaped (D, B, I*J) from the triangle contraction and returns
    (B, I*J, D) ready for the output GEMM.
    """
    from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose

    return layer_norm_transpose(
        x.reshape(d, b, n),
        weight,
        bias,
        eps=eps,
        layout="dbn->bnd",
    ).reshape(b, n, d)


class _TriangleContractBMM(torch.autograd.Function):
    """Triangle contraction via batched GEMM with contiguous-safe backward.

    PyTorch autograd computes grad_right = grad_right_T.transpose(1,2) which is
    non-contiguous, then the downstream reshape calls .contiguous() — a 120μs
    copy at seq_len=384.  This custom backward avoids that by choosing equivalent
    gradient formulas whose outputs are already contiguous:

      outgoing (O = L @ R^T):
        grad_L = G @ R          — standard, already contiguous
        grad_R = G^T @ L        — avoids (L^T @ G)^T; result is contiguous

      incoming (O = L^T @ R):
        grad_L = R @ G^T        — avoids (G @ R^T)^T; result is contiguous
        grad_R = L @ G          — standard, already contiguous
    """

    @staticmethod
    def forward(ctx, left, right, direction_flag):
        # direction_flag: 0 = outgoing, 1 = incoming
        if direction_flag == 0:
            out = torch.bmm(left, right.transpose(1, 2))
        else:
            out = torch.bmm(left.transpose(1, 2), right)
        ctx.save_for_backward(left, right)
        ctx.direction_flag = direction_flag
        return out

    @staticmethod
    def backward(ctx, grad_out):
        left, right = ctx.saved_tensors
        if ctx.direction_flag == 0:
            # O = L @ R^T  →  grad_L = G @ R,  grad_R = G^T @ L
            grad_left = torch.bmm(grad_out, right)
            grad_right = torch.bmm(grad_out.transpose(1, 2), left)
        else:
            # O = L^T @ R  →  grad_L = R @ G^T,  grad_R = L @ G
            grad_left = torch.bmm(right, grad_out.transpose(1, 2))
            grad_right = torch.bmm(left, grad_out)
        return grad_left, grad_right, None


def _triangle_contract_bmm_dbij(
    a_dbij: torch.Tensor,
    b_dbij: torch.Tensor,
    direction: str,
) -> torch.Tensor:
    """Triangle contraction from (D, B, I, J) tensors.

    Returns a (D, B, I, J) tensor.
    """
    d, b, i, j = a_dbij.shape
    if i != j:
        msg = f"Triangle multiplicative update expects square pair map, got {i}x{j}."
        raise ValueError(msg)
    if b_dbij.shape != a_dbij.shape:
        msg = f"Mismatched contraction shapes: {a_dbij.shape} vs {b_dbij.shape}"
        raise ValueError(msg)

    left = a_dbij.reshape(d * b, i, j)
    right = b_dbij.reshape(d * b, i, j)

    direction_flag = 0 if direction == "outgoing" else 1
    if direction not in ("outgoing", "incoming"):
        msg = f"Invalid direction: {direction}"
        raise ValueError(msg)

    out = _TriangleContractBMM.apply(left, right, direction_flag)
    return out.view(d, b, i, j)


# ---------------------------------------------------------------------------
# Autotuning configs
# ---------------------------------------------------------------------------
# V2 baselines + maxnreg=128 (2 CTAs/SM) and maxnreg=96 (up to 5 CTAs/SM).
# maxnreg=128: 128 × 256 threads = 32 768 regs/CTA → 2 CTAs/SM on H100.
# maxnreg=96 : 96  × 128 threads = 12 288 regs/CTA → ~5 CTAs/SM.


# Output kernel: single-pass dual-accumulator (4 tensors per stage).
# BK=32 / BM=64 small-tile configs reach up to 4 CTAs/SM.
# SMEM: 4×(64×32)×2 per stage × 3 stages = 48 KB → floor(227/48) = 4 CTAs/SM
# NOTE: maxnreg ≤ 80 is FORBIDDEN (need ≈84 regs → 10-15× spill slowdown)


# ---------------------------------------------------------------------------
# Forward Triton kernels
# ---------------------------------------------------------------------------


def get_seq_group(rows) -> int:
    """Bucket a raw row/element count for the autotune key (canonical autotune.buckets).

    Every count in this file is either the pair row count M = b*i*j (the GEMMs) or a flat
    element count (the elementwise backward), so all three ops share ONE bucketer.
    bucket_mixed is used rather than bucket_squared even for the pair-only GEMMs because
    COMBINED_EDGES is the UNION of the linear and squared edge sets -- it is a strict
    refinement of the squared edges, never coarser, and it also covers the elementwise
    op whose count is ~2*D*M and lands far above the squared edges.
    """
    return _bucket(rows)


# AUTOTUNE KEY: ['seq_group', 'K', 'ALLOW_TF32'] -- was ['M', 'N', 'K', 'ALLOW_TF32']. Keying
# raw M (the pair row count b*i*j) minted one full config sweep per sequence length; `seq_group` is
# that count bucketed. `N` is DROPPED as redundant, not ignored -- see the proof above each launcher.
# ALLOW_TF32 STAYS: it is a real codegen switch (it selects the tl.dot input precision for fp32
# operands). It is read from the PROCESS-GLOBAL torch.backends.cuda.matmul.allow_tf32 at every
# launch, so flipping torch.set_float32_matmul_precision mid-run legitimately costs a re-tune.
@triton.autotune(configs=configs_for("trimul_gemm_gate_saveact_triton"),
                 key=['seq_group', 'K', 'ALLOW_TF32'])
@triton.jit
def _input_gated_gemm_kernel(
    xn_ptr,
    wg_ptr,
    wp_ptr,
    mask_ptr,
    out_ptr,
    sig_ptr,   # saved sigmoid*mask — reused in backward
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_wn,
    APPLY_MASK: tl.constexpr,
    TRANSPOSE_OUT: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    seq_group,
):
    """Fused gated GEMM: out = sigmoid(xn @ wg.T) * (xn @ wp.T) [* mask].

    Also writes sig_ptr = sigmoid(xn @ wg.T) * mask (saved for backward).

    Gate and projection K-loops run simultaneously; a single pass over the
    K dimension produces both accumulators, halving HBM reads of ``xn``.

    When TRANSPOSE_OUT=True outputs are written as (N, M) instead of (M, N),
    which lets the caller reshape directly to (D, B, I, J) without an extra copy.
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    xn_bp = tl.make_block_ptr(
        base=xn_ptr,
        shape=(M, K),
        strides=(K, 1),
        offsets=(pid_m * BLOCK_M1, 0),
        block_shape=(BLOCK_M1, BLOCK_K),
        order=(1, 0),
    )
    wg_bp = tl.make_block_ptr(
        base=wg_ptr,
        shape=(N, K),
        strides=(stride_wn, 1),
        offsets=(pid_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    wp_bp = tl.make_block_ptr(
        base=wp_ptr,
        shape=(N, K),
        strides=(stride_wn, 1),
        offsets=(pid_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )

    # The K loop below walks the contraction axis in BLOCK_K steps. `boundary_check=(0,)` bounds
    # only axis 0 (M/N), so a K that is not a multiple of the tuned BLOCK_K contracted out-of-range
    # columns into every accumulator, and the weight tiles were loaded with no boundary_check at
    # all -- unbounded on BOTH axes. Both K and N are constexpr here, so these fold at compile time
    # and the fully aligned path keeps exactly the loads it had; the same EVEN_* dispatch as
    # triangle_attention/triton/atomic.py:156.
    EVEN_K = K % BLOCK_K == 0
    EVEN_N = N % BLOCK_N == 0

    acc_g = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
    acc_p = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K, BLOCK_K):
        if EVEN_K:
            xn = tl.load(xn_bp, boundary_check=(0,))
        else:
            xn = tl.load(xn_bp, boundary_check=(0, 1))
        if EVEN_K and EVEN_N:
            wg = tl.load(wg_bp)
            wp = tl.load(wp_bp)
        else:
            wg = tl.load(wg_bp, boundary_check=(0, 1))
            wp = tl.load(wp_bp, boundary_check=(0, 1))
        acc_g += tl.dot(xn, tl.trans(wg), allow_tf32=ALLOW_TF32)
        acc_p += tl.dot(xn, tl.trans(wp), allow_tf32=ALLOW_TF32)
        xn_bp = tl.advance(xn_bp, (0, BLOCK_K))
        wg_bp = tl.advance(wg_bp, (0, BLOCK_K))
        wp_bp = tl.advance(wp_bp, (0, BLOCK_K))

    sig = tl.sigmoid(acc_g)
    ab = sig * acc_p

    # (the old `row_scale = tl.full((BLOCK_M1, 1), 1.0)` initialiser is gone: it was dead outside
    # the APPLY_MASK branch, and its literal `1` trailing dim was a tile extent that came from
    # nowhere in the config space. The masked path builds row_scale from the loaded mask instead.)
    if APPLY_MASK:
        offs_m = pid_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
        mask_m = offs_m < M
        row_scale = tl.load(mask_ptr + offs_m, mask=mask_m, other=0.0)[:, None]
        ab = ab * row_scale
        sig = sig * row_scale   # sig_m = sigmoid * mask

    ab_store = ab.to(xn_ptr.dtype.element_ty)
    # ab is cast to the input dtype so that:
    #   • dtype consistency is maintained throughout the full pipeline
    #     (triangle contraction + backward GEMMs all use the same dtype)
    #   • if ab were fp32, grad_ab in backward would also be fp32, causing a
    #     dtype mismatch against bf16 w_combined in the weight-grad GEMM.
    # sig is kept as fp32 (sig_ptr always points to a float32 buffer).
    # The critical backward path is d_gate = grad * ab * (1 - sig_m).
    # When sig_m ≈ 1 is stored in bf16, (1 - sig_m) collapses to 0;
    # keeping sig_m in fp32 preserves the correct small-gradient signal.
    # Note: ab still limits d_gate to the precision of the input dtype, but
    # the dominant catastrophic-cancellation hazard lives in (1 - sig_m).

    if TRANSPOSE_OUT:
        out_bp = tl.make_block_ptr(
            base=out_ptr,
            shape=(N, M),
            strides=(M, 1),
            offsets=(pid_n * BLOCK_N, pid_m * BLOCK_M1),
            block_shape=(BLOCK_N, BLOCK_M1),
            order=(1, 0),
        )
        sig_bp = tl.make_block_ptr(
            base=sig_ptr,
            shape=(N, M),
            strides=(M, 1),
            offsets=(pid_n * BLOCK_N, pid_m * BLOCK_M1),
            block_shape=(BLOCK_N, BLOCK_M1),
            order=(1, 0),
        )
        tl.store(out_bp, tl.trans(ab_store), boundary_check=(0, 1))
        tl.store(sig_bp, tl.trans(sig), boundary_check=(0, 1))
    else:
        out_bp = tl.make_block_ptr(
            base=out_ptr,
            shape=(M, N),
            strides=(N, 1),
            offsets=(pid_m * BLOCK_M1, pid_n * BLOCK_N),
            block_shape=(BLOCK_M1, BLOCK_N),
            order=(1, 0),
        )
        sig_bp = tl.make_block_ptr(
            base=sig_ptr,
            shape=(M, N),
            strides=(N, 1),
            offsets=(pid_m * BLOCK_M1, pid_n * BLOCK_N),
            block_shape=(BLOCK_M1, BLOCK_N),
            order=(1, 0),
        )
        # (0, 1), not (0,): BLOCK_N columns written into an N-wide row spilled into the next
        # row. The transposed branch above already checks both axes; this one did not.
        tl.store(out_bp, ab_store, boundary_check=(0, 1))
        tl.store(sig_bp, sig, boundary_check=(0, 1))


# AUTOTUNE KEY: ['seq_group', 'K', 'ALLOW_TF32'] -- was ['M', 'N', 'K', 'ALLOW_TF32']. Keying
# raw M (the pair row count b*i*j) minted one full config sweep per sequence length; `seq_group` is
# that count bucketed. `N` is DROPPED as redundant, not ignored -- see the proof above each launcher.
# ALLOW_TF32 STAYS: it is a real codegen switch (it selects the tl.dot input precision for fp32
# operands). It is read from the PROCESS-GLOBAL torch.backends.cuda.matmul.allow_tf32 at every
# launch, so flipping torch.set_float32_matmul_precision mid-run legitimately costs a re-tune.
@triton.autotune(configs=configs_for("trimul_outproj_gemm_gate_saveact_triton"),
                 key=['seq_group', 'K', 'ALLOW_TF32'])
@triton.jit
def _output_gated_gemm_kernel(
    x1n_ptr,
    x2_ptr,
    wg_ptr,
    wp_ptr,
    out_ptr,
    sig_ptr,   # saved sigmoid — reused in backward
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_wn,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    seq_group,
):
    """Single-pass dual-accumulator output GEMM. Also writes sig = sigmoid(x1n @ wg.T).

    Gate and projection K-loops run together with dual accumulators (acc_g, acc_p).
    On Hopper, TMA pipelines all four tensor loads (x1n, x2, wg, wp) concurrently,
    fully hiding HBM latency in one pass vs. the two-pass approach that reads
    the weight tiles twice and serialises the two K-loops.
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    x1_bp = tl.make_block_ptr(x1n_ptr, (M, K), (K, 1), (pid_m * BLOCK_M1, 0), (BLOCK_M1, BLOCK_K), (1, 0))
    x2_bp = tl.make_block_ptr(x2_ptr,  (M, K), (K, 1), (pid_m * BLOCK_M1, 0), (BLOCK_M1, BLOCK_K), (1, 0))
    wg_bp = tl.make_block_ptr(wg_ptr,  (N, K), (stride_wn, 1), (pid_n * BLOCK_N, 0), (BLOCK_N, BLOCK_K), (1, 0))
    wp_bp = tl.make_block_ptr(wp_ptr,  (N, K), (stride_wn, 1), (pid_n * BLOCK_N, 0), (BLOCK_N, BLOCK_K), (1, 0))

    # The K loop below walks the contraction axis in BLOCK_K steps. `boundary_check=(0,)` bounds
    # only axis 0 (M/N), so a K that is not a multiple of the tuned BLOCK_K contracted out-of-range
    # columns into every accumulator, and the weight tiles were loaded with no boundary_check at
    # all -- unbounded on BOTH axes. Both K and N are constexpr here, so these fold at compile time
    # and the fully aligned path keeps exactly the loads it had; the same EVEN_* dispatch as
    # triangle_attention/triton/atomic.py:156.
    EVEN_K = K % BLOCK_K == 0
    EVEN_N = N % BLOCK_N == 0

    acc_g = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
    acc_p = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K, BLOCK_K):
        if EVEN_K:
            x1 = tl.load(x1_bp, boundary_check=(0,))
            x2 = tl.load(x2_bp, boundary_check=(0,))
        else:
            x1 = tl.load(x1_bp, boundary_check=(0, 1))
            x2 = tl.load(x2_bp, boundary_check=(0, 1))
        if EVEN_K and EVEN_N:
            wg = tl.load(wg_bp)
            wp = tl.load(wp_bp)
        else:
            wg = tl.load(wg_bp, boundary_check=(0, 1))
            wp = tl.load(wp_bp, boundary_check=(0, 1))
        acc_g += tl.dot(x1, tl.trans(wg), allow_tf32=ALLOW_TF32)
        acc_p += tl.dot(x2, tl.trans(wp), allow_tf32=ALLOW_TF32)
        x1_bp = tl.advance(x1_bp, (0, BLOCK_K))
        x2_bp = tl.advance(x2_bp, (0, BLOCK_K))
        wg_bp = tl.advance(wg_bp, (0, BLOCK_K))
        wp_bp = tl.advance(wp_bp, (0, BLOCK_K))

    gate = tl.sigmoid(acc_g)
    # out_tile is cast to the input dtype: it becomes the final function output
    # (returned from _OutputGEMM.apply) so the caller's dtype contract must be
    # preserved.  gate (sigmoid) is stored as fp32 into sig_ptr so the backward
    # d_gate = grad * ab * (1 - gate) avoids bf16 cancellation when gate ≈ 1.
    out_tile = (gate * acc_p).to(x1n_ptr.dtype.element_ty)

    out_bp = tl.make_block_ptr(
        base=out_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M1, pid_n * BLOCK_N),
        block_shape=(BLOCK_M1, BLOCK_N),
        order=(1, 0),
    )
    sig_bp = tl.make_block_ptr(
        base=sig_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M1, pid_n * BLOCK_N),
        block_shape=(BLOCK_M1, BLOCK_N),
        order=(1, 0),
    )
    # (0, 1), not (0,): BLOCK_N columns written into an N-wide row spilled into the next row.
    tl.store(out_bp, out_tile, boundary_check=(0, 1))
    tl.store(sig_bp, gate, boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# Elementwise backward kernel
# ---------------------------------------------------------------------------
# BLOCK=2048 → 16 elems/thread at num_warps=4.
# Reduces kernel-launch overhead and improves vectorised HBM access patterns
# for the 3-input / 2-output elementwise op over large M.

# BLOCK_E is a CSV tile rather than the module-level literal
# 2048 the two launch sites passed. A CSV row may still spell 2048, so that value is
# still reachable; the grid below is a meta-lambda so the launch geometry always matches the
# BLOCK the kernel is compiled with.


# AUTOTUNE KEY: ['seq_group'] -- was ['N_total'], the raw flat element count, i.e. a fresh config
# sweep for every distinct shape this backward sees. `N_total` is still the loop bound in the
# body; only the KEY is bucketed. seq_group is never read by the kernel, so the generated code is
# unchanged and the result stays bit-identical.
@triton.autotune(configs=configs_for("gated_projection_bwd_gate_recompute_flat_triton"), key=['seq_group'])
@triton.jit
def _gated_gemm_bwd_elemwise_kernel(
    grad_ptr,
    ab_ptr,
    sig_ptr,
    d_gate_ptr,
    d_proj_ptr,
    N_total,
    BLOCK_E: tl.constexpr,
    seq_group,
):
    """Fused elementwise backward for gated GEMM.

    Computes (in fp32):
        d_gate = grad * ab * (1 - sig_m)
        d_proj = grad * sig_m
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offs < N_total

    grad = tl.load(grad_ptr + offs, mask=mask).to(tl.float32)
    ab = tl.load(ab_ptr + offs, mask=mask).to(tl.float32)
    sig_m = tl.load(sig_ptr + offs, mask=mask).to(tl.float32)

    d_gate = grad * ab * (1.0 - sig_m)
    d_proj = grad * sig_m

    out_dtype = grad_ptr.dtype.element_ty
    tl.store(d_gate_ptr + offs, d_gate.to(out_dtype), mask=mask)
    tl.store(d_proj_ptr + offs, d_proj.to(out_dtype), mask=mask)


def _elemwise_bwd_combined(grad, ab, sig_m):
    """Backward for input gated GEMM — writes (d_gate, d_proj) into a single (2N, M) buffer.

    Returns d_combined[shape=(2N, M)] where:
      d_combined[:N] = d_gate   (= grad * ab * (1 - sig_m))
      d_combined[N:] = d_proj   (= grad * sig_m)

    Both halves are contiguous, so the subsequent single (4D, M) @ (M, K)
    weight-grad GEMM reads them as one tensor with no extra copy.
    """
    n_total = grad.numel()
    n, m = grad.shape  # (N, M) for TRANSPOSE_OUT=True input path

    d_combined = torch.empty((2 * n, m), dtype=grad.dtype, device=grad.device)
    grid = lambda META: (triton.cdiv(n_total, META["BLOCK_E"]),)  # noqa: E731
    _gated_gemm_bwd_elemwise_kernel[grid](
        grad.contiguous(), ab.contiguous(), sig_m.contiguous(),
        d_combined[:n], d_combined[n:],
        n_total,
        seq_group=get_seq_group(n_total),
    )
    return d_combined  # (2N, M) fully contiguous


def _elemwise_bwd_separate(grad, ab, sig_m):
    """Backward for output gated GEMM — returns (d_gate, d_proj) as separate tensors."""
    n_total = grad.numel()
    d_gate = torch.empty_like(grad)
    d_proj = torch.empty_like(grad)
    grid = lambda META: (triton.cdiv(n_total, META["BLOCK_E"]),)  # noqa: E731
    _gated_gemm_bwd_elemwise_kernel[grid](
        grad.contiguous(), ab.contiguous(), sig_m.contiguous(),
        d_gate, d_proj,
        n_total,
        seq_group=get_seq_group(n_total),
    )
    return d_gate, d_proj


# ---------------------------------------------------------------------------
# PyTorch helpers
# ---------------------------------------------------------------------------


def _ln_fwd(x, norm_w, norm_b, eps):
    m, k = x.shape
    x_normed_3d, mean_2d, rstd_2d = torch.ops.cuequivariance.layer_norm_transpose(
        x.view(1, m, k),
        norm_w,
        norm_b,
        eps,
        True,
        Layout.BND_BND,
    )
    return x_normed_3d.view(m, k), mean_2d.view(m), rstd_2d.view(m)


def _layernorm_backward_fused(grad_x_normed, x, mean, rstd, norm_w):
    m, k = x.shape
    grad_x_op, grad_w_tiles, grad_b_tiles = torch.ops.cuequivariance.layer_norm_transpose_bwd(
        grad_x_normed.to(x.dtype).view(1, m, k),
        x.view(1, m, k),
        norm_w,
        mean.view(1, m),
        rstd.view(1, m),
        True,
        Layout.BND_BND,
    )
    grad_w = grad_w_tiles.sum(dim=(0, 1)).to(norm_w.dtype)
    grad_b = grad_b_tiles.sum(dim=(0, 1)).to(norm_w.dtype)
    return grad_x_op.view(m, k), grad_w, grad_b


def _input_gemm_fwd(x_normed, w_gate, w_proj, mask, transpose_out):
    """Run _input_gated_gemm_kernel; return (ab, sig_m).

    ab (= sigmoid(xn@wg.T) * (xn@wp.T) * mask) is stored in the input dtype
    (e.g. bf16) to maintain dtype consistency throughout the full pipeline:
      • the triangle contraction and its backward use the same dtype as the
        input projection weights (w_gate, w_proj)
      • the backward weight-grad GEMM torch.mm(d_combined.T, w_combined) needs
        d_combined (which inherits grad_ab's dtype) to match w_combined; if ab
        were fp32, grad_ab would be fp32, causing a dtype mismatch against bf16
        w_combined

    sig_m (= sigmoid * mask) is stored in fp32 to avoid bf16 catastrophic
    cancellation in (1 - sig_m) when the gate is saturated (sig_m ≈ 1).
    """
    # N == 2*K, so N is NOT in this kernel's autotune key (K is).
    # PROOF (public entry point fused_triangle_multiplicative_update_dtv1): x is (b, i, j, d) and
    # x_flat = x.reshape(m, d), so K = x_normed.shape[1] = d. The kernel writes ab_t of shape
    # (n, m); the caller then does `a_t, b_t = torch.chunk(ab_t, 2, dim=0)` followed by
    # `a_t.view(d, b, i, j)` with m = b*i*j -- a view that only succeeds if n/2 == d. Hence
    # n == 2*d == 2*K (g_in_weight is the [left; right] combined gate weight, 2D rows).
    m, k = x_normed.shape
    n = w_gate.shape[0]
    shape = (n, m) if transpose_out else (m, n)
    ab = torch.empty(shape, device=x_normed.device, dtype=x_normed.dtype)
    sig_m = torch.empty(shape, device=x_normed.device, dtype=torch.float32)
    apply_mask = mask is not None
    mask_t = mask if apply_mask else torch.empty(0, device=x_normed.device, dtype=x_normed.dtype)
    _input_gated_gemm_kernel[(
        lambda meta: (triton.cdiv(m, meta["BLOCK_M1"]) * triton.cdiv(n, meta["BLOCK_N"]),)
    )](
        x_normed, w_gate, w_proj, mask_t, ab, sig_m,
        m, n, k,
        w_gate.stride(0),
        APPLY_MASK=apply_mask,
        TRANSPOSE_OUT=transpose_out,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        seq_group=get_seq_group(m),
    )
    return ab, sig_m


def _output_gemm_fwd(x_normed, x_out, w_gate, w_proj):
    """Run _output_gated_gemm_kernel; return (ab, sig).

    ab (the gated output) is in the input dtype — it becomes the final function
    return value, so the caller's dtype contract must be preserved.
    sig (= sigmoid(x_normed @ w_gate.T)) is float32 so that the backward
    d_gate = grad * ab * (1 - sig) avoids bf16 catastrophic cancellation.
    """
    # N == K, so N is NOT in this kernel's autotune key (K is).
    # PROOF (public entry point): x_normed is the same (m, d) tensor as on the input path, so
    # K = d. The kernel writes ab of shape (m, n) and the entry point returns
    # `result.reshape(b, i, j, d)` with m = b*i*j -- a reshape that only succeeds if n == d.
    # Hence n == d == K (g_out_weight has D rows, unlike the 2D-row input gate weight).
    m, k = x_normed.shape
    n = w_gate.shape[0]
    ab = torch.empty((m, n), device=x_normed.device, dtype=x_normed.dtype)
    sig = torch.empty((m, n), device=x_normed.device, dtype=torch.float32)
    _output_gated_gemm_kernel[(
        lambda meta: (triton.cdiv(m, meta["BLOCK_M1"]) * triton.cdiv(n, meta["BLOCK_N"]),)
    )](
        x_normed, x_out, w_gate, w_proj, ab, sig,
        m, n, k,
        w_gate.stride(0),
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        seq_group=get_seq_group(m),
    )
    return ab, sig


# ---------------------------------------------------------------------------
# Autograd functions
# ---------------------------------------------------------------------------


class _InputLNAndGEMM(torch.autograd.Function):
    """Input LN + fused gated GEMM.

    Forward: single Triton kernel computes sigmoid(xn @ wg.T) * (xn @ wp.T)
             and saves sig_m = sigmoid * mask for the backward.
    Backward: custom Triton elementwise kernel + combined 4D×2D GEMMs +
              cuequivariance fused LN backward.

    Saved in context: x, x_normed, norm_w, w_gate, w_proj, mask, mean, rstd,
                      ab, sig_m, w_combined  (w_combined avoids torch.cat in bwd).
    """

    @staticmethod
    @opaque()
    def forward(ctx, x, norm_w, norm_b, w_gate, w_proj, mask, eps, transpose_out):
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
            x, w_gate, w_proj = x.to(dtype), w_gate.to(dtype), w_proj.to(dtype)

        x = x.contiguous()
        x_normed, mean, rstd = _ln_fwd(x, norm_w, norm_b, eps)

        apply_mask = mask is not None
        if apply_mask:
            mask = mask.to(x.dtype).contiguous().view(-1)

        ab, sig_m = _input_gemm_fwd(
            x_normed, w_gate, w_proj,
            mask if apply_mask else None,
            transpose_out,
        )

        # Pre-compute [w_gate; w_proj] (4D×K) once in forward; reuse in backward
        # to avoid a torch.cat allocation on every gradient step.
        w_combined = torch.cat([w_gate, w_proj], dim=0)

        ctx.save_for_backward(
            x, x_normed, norm_w, w_gate, w_proj,
            mask if apply_mask else None,
            mean, rstd, ab, sig_m,
            w_combined,
        )
        ctx.transpose_out = transpose_out
        return ab, mean, rstd, x_normed

    @staticmethod
    @opaque()
    def backward(ctx, grad_ab, _grad_mean, _grad_rstd, grad_xn_ext):
        (
            x, x_normed, norm_w, w_gate, w_proj, mask,
            mean, rstd, ab, sig_m,
            w_combined,
        ) = ctx.saved_tensors
        n = w_gate.shape[0]   # 2D (full combined gate weight rows: left+right)

        # Elementwise backward → d_combined (2N, M) contiguous buffer.
        # d_combined[:N] = d_gate,  d_combined[N:] = d_proj
        d_combined = _elemwise_bwd_combined(grad_ab, ab, sig_m)

        # Weight grad: single (4D, M) @ (M, K) GEMM then split — reads x_normed once.
        grad_w_combined = d_combined @ x_normed   # (4D, K)
        grad_w_gate = grad_w_combined[:n].contiguous()
        grad_w_proj = grad_w_combined[n:].contiguous()

        # Input grad: single (M, 4D) @ (4D, K) GEMM — w_combined was pre-saved.
        # torch.addmm fuses the GEMM and the external-gradient addition in one
        # cuBLAS call, eliminating a separate 37 MB read+write for the add.
        if grad_xn_ext is not None:
            grad_xn = torch.addmm(grad_xn_ext, d_combined.T, w_combined)
        else:
            grad_xn = torch.mm(d_combined.T, w_combined)   # (M, K)

        grad_x, grad_nw, grad_nb = _layernorm_backward_fused(grad_xn, x, mean, rstd, norm_w)
        return grad_x, grad_nw, grad_nb, grad_w_gate, grad_w_proj, None, None, None


class _OutputGEMM(torch.autograd.Function):
    """Output gated GEMM with saved sigmoid.

    Forward: two-pass Triton kernel; saves sig = sigmoid(x1n @ wg.T).
    Backward: custom Triton elementwise kernel + four separate GEMMs.
    """

    @staticmethod
    @opaque()
    def forward(ctx, x_normed, x_out, w_gate, w_proj):
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
            x_normed, x_out = x_normed.to(dtype), x_out.to(dtype)
            w_gate, w_proj = w_gate.to(dtype), w_proj.to(dtype)

        x_normed, x_out = x_normed.contiguous(), x_out.contiguous()
        ab, sig = _output_gemm_fwd(x_normed, x_out, w_gate, w_proj)
        ctx.save_for_backward(x_normed, x_out, w_gate, w_proj, ab, sig)
        return ab

    @staticmethod
    @opaque()
    def backward(ctx, grad_out):
        x_normed, x_out, w_gate, w_proj, ab, sig = ctx.saved_tensors

        d_gate, d_proj = _elemwise_bwd_separate(grad_out.contiguous(), ab, sig)

        grad_w_gate = d_gate.T @ x_normed    # (D, M) @ (M, K) = (D, K)
        grad_w_proj = d_proj.T @ x_out       # (D, M) @ (M, K) = (D, K)
        grad_x_normed = d_gate @ w_gate      # (M, D) @ (D, K) = (M, K)
        grad_x_out = d_proj @ w_proj         # (M, D) @ (D, K) = (M, K)
        return grad_x_normed, grad_x_out, grad_w_gate, grad_w_proj


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fused_triangle_multiplicative_update_dtv1(
    x,
    direction,
    mask,
    norm_in_weight,
    norm_in_bias,
    p_in_weight,
    g_in_weight,
    norm_out_weight,
    norm_out_bias,
    p_out_weight,
    g_out_weight,
    eps=1e-5,
):
    """dt-v1: fully optimized fused Triton triangle multiplicative update.

    See module docstring for the complete list of forward and backward
    optimizations.
    """
    b, i, j, d = x.shape
    m = b * i * j
    x_flat = x.reshape(m, d)
    mask_flat = mask.reshape(-1) if mask is not None else None

    ab_t, _, _, x_normed = _InputLNAndGEMM.apply(
        x_flat,
        norm_in_weight,
        norm_in_bias,
        g_in_weight,
        p_in_weight,
        mask_flat,
        eps,
        True,  # TRANSPOSE_OUT — writes (2D, M) for free reshape to (D, B, I, J)
    )
    a_t, b_t = torch.chunk(ab_t, 2, dim=0)
    a_dbij = a_t.view(d, b, i, j)
    b_dbij = b_t.view(d, b, i, j)
    tri_out = _triangle_contract_bmm_dbij(a_dbij, b_dbij, direction)

    x_out_flat = _output_layer_norm_transpose(
        tri_out,
        norm_out_weight,
        norm_out_bias,
        b,
        i * j,
        d,
        eps,
    )
    result = _OutputGEMM.apply(
        x_normed,
        x_out_flat.reshape(m, d),
        g_out_weight,
        p_out_weight,
    )
    return result.reshape(b, i, j, d)
