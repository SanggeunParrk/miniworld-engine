"""Fused sigmoid-gate + output projection.

Baseline does three kernels:
    gate = to_gate(pln)                 # cuBLAS GEMM   [M, dh]
    gated = sigmoid(gate) * out_r       # elementwise   [M, dh]   (~0.4ms @ L1024)
    out = gated @ Wo^T                   # cuBLAS GEMM   [M, d_pair]

This fuses the elementwise + the to_out GEMM into one triton kernel: the GEMM's
A-tile is computed in the prologue as sigmoid(gate)*out_r, so `gated` never
touches HBM and the standalone elementwise kernel disappears. `to_gate` stays on
cuBLAS (it's the bigger GEMM -- don't risk it).

Forward is the fused triton kernel; backward is plain torch (cuBLAS) using the
saved gate/out_r, so it matches the baseline bwd exactly (no regression) while the
forward gets the fusion win. Wrapped as an autograd.Function.

M = B*L*L (rows), dh = d_hidden (contraction), N = d_pair (output width).
"""

from __future__ import annotations

from miniworld_engine.kernels._compile import opaque
from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.kernels.gated_projection.triton.main import _sigmul_bwd, _sigmul_fwd

import torch
import triton
import triton.language as tl


from miniworld_engine.autotune.shape_key import both_key, length_of, rows_of, token_key





from miniworld_engine.autotune.buckets import bucket_mixed as _bucket


def get_seq_group(rows) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    return _bucket(rows)


def _key_of(shape) -> int:
    """token_key(L) from an activation's PRE-flatten shape ``[..., DH]``.

    ``length_of`` reads L at ``shape[-2]``: right for both the pair ``(B, L, L, DH)`` and token
    ``(B, L, DH)`` activations the module hands in, with no branch on which one it is. Only the
    autograd Functions call this -- they still hold the un-flattened shape. The inner launchers
    below take the finished key as a parameter, because once the activation is ``(M, DH)`` the
    L is gone and M alone cannot say what produced it.
    """
    return token_key(length_of(shape))


@triton.autotune(configs=configs_for("gated_projection_gate_gemm_triton"), key=['shape_key', 'N', 'DH'])
@triton.jit
def _gate_out_fwd(
    gate_ptr,   # [M, DH]
    outr_ptr,   # [M, DH]
    wo_ptr,     # [N, DH]   (to_out.weight: out_features=N, in_features=DH)
    o_ptr,      # [M, N]
    M, N,
    DH: tl.constexpr,
    stride_gm, stride_gd,
    stride_om, stride_od,
    stride_wn, stride_wd,
    stride_cm, stride_cn,
    BLOCK_M1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    shape_key,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    offs_m = pid_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_ok = offs_m < M
    n_ok = offs_n < N

    acc = tl.zeros([BLOCK_M1, BLOCK_N], dtype=tl.float32)
    for k0 in range(0, DH, BLOCK_K):
        kk = k0 + offs_k
        k_ok = kk < DH
        # A-tile = sigmoid(gate) * out_r  -> [BLOCK_M1, BLOCK_K]
        g = tl.load(
            gate_ptr + offs_m[:, None] * stride_gm + kk[None, :] * stride_gd,
            mask=m_ok[:, None] & k_ok[None, :], other=0.0,
        ).to(tl.float32)
        r = tl.load(
            outr_ptr + offs_m[:, None] * stride_om + kk[None, :] * stride_od,
            mask=m_ok[:, None] & k_ok[None, :], other=0.0,
        ).to(tl.float32)
        a = (tl.sigmoid(g) * r).to(wo_ptr.dtype.element_ty)
        # Wo-tile [BLOCK_K, BLOCK_N]: wo[n, k] -> transpose for the dot
        wo = tl.load(
            wo_ptr + offs_n[None, :] * stride_wn + kk[:, None] * stride_wd,
            mask=n_ok[None, :] & k_ok[:, None], other=0.0,
        )
        acc = tl.dot(a, wo, acc)

    tl.store(
        o_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(o_ptr.dtype.element_ty),
        mask=m_ok[:, None] & n_ok[None, :],
    )


# BLOCK_N tiles the DH output axis, which used to be the raw shape constant DH (`tl.arange(0, DH)` and
# a `[BLOCK_M1, DH]` accumulator): the live accumulator width was d_hidden, not a config choice. DH is a
# free (non-reduced) axis here -- the contraction is N -- so it moves onto the grid. ``BLOCK_N`` is not
# constrained, so its candidate values live in the CSV like every other axis; they are the
# canonical 2-D set.
@triton.autotune(configs=configs_for("gated_projection_bwd_dx_triton"), key=['shape_key', 'N', 'DH'])
@triton.jit
def _dgrad_epi(
    do_ptr,     # [M, N]   = grad_out
    wo_ptr,     # [N, DH]
    g_ptr,      # [M, DH]  = gate (pre-sigmoid)
    r_ptr,      # [M, DH]  = out_r
    dr_ptr,     # out: d_out_r  [M, DH]
    dg_ptr,     # out: d_gate   [M, DH]
    a_ptr,      # out: gated = sigmoid(gate)*r  [M, DH]  (for the d_wo GEMM)
    M, N: tl.constexpr, DH: tl.constexpr,
    s_dom, s_don, s_won, s_woh, s_gm, s_gh, s_rm, s_rh, s_om, s_oh,
    BLOCK_M1: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
    shape_key,
):
    """Fuses the dgrad GEMM d_a = grad_out @ wo with the gate-backward epilogue:
    d_a is never materialized, gate/out_r are read once. One kernel replaces the
    cuBLAS dgrad + a separate elementwise pass.

    The contraction dim N is TILED (BLOCK_K) and accumulated, so shared memory is bounded by
    the [BLOCK_M1,BLOCK_K]+[BLOCK_K,DH] tiles instead of the full [N,DH] weight. The old single-shot
    ``wo[N, DH]`` load needed ~N*DH*2 bytes of smem (e.g. 128 KB at N=DH=256), which fits
    A100/H100 but exceeds the ~100 KB/SM of sm_86 (RTX A5000/A6000); tiling makes it
    launchable on any GPU. Math is unchanged (same GEMM + epilogue)."""
    pid = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    rm = pid * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    rh = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
    mm = rm[:, None] < M
    hm = rh[None, :] < DH
    em = mm & hm
    da = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)                              # [BLOCK_M1, BLOCK_N] acc
    for n0 in range(0, N, BLOCK_K):
        rn = n0 + tl.arange(0, BLOCK_K)
        nmask = rn < N
        do = tl.load(do_ptr + rm[:, None] * s_dom + rn[None, :] * s_don,
                     mask=mm & nmask[None, :], other=0.0)                  # [BLOCK_M1, BLOCK_K]
        wo = tl.load(wo_ptr + rn[:, None] * s_won + rh[None, :] * s_woh,
                     mask=nmask[:, None] & hm, other=0.0)                  # [BLOCK_K, BLOCK_N]
        da = tl.dot(do, wo, da)                                           # accumulate [BLOCK_M1, BLOCK_N]
    s = tl.sigmoid(tl.load(g_ptr + rm[:, None] * s_gm + rh[None, :] * s_gh,
                           mask=em, other=0.0).to(tl.float32))
    r = tl.load(r_ptr + rm[:, None] * s_rm + rh[None, :] * s_rh, mask=em, other=0.0).to(tl.float32)
    off = rm[:, None] * s_om + rh[None, :] * s_oh
    tl.store(dr_ptr + off, (s * da).to(dr_ptr.dtype.element_ty), mask=em)
    tl.store(dg_ptr + off, (da * r * s * (1.0 - s)).to(dg_ptr.dtype.element_ty), mask=em)
    tl.store(a_ptr + off, (s * r).to(a_ptr.dtype.element_ty), mask=em)


@opaque(fake=lambda do2, wo, g2, r2, shape_key=None: (
            torch.empty_like(g2), torch.empty_like(g2), torch.empty_like(g2)),
        name="bias_only_attention_gate_out_dgrad_epilogue")
def _dgrad_epilogue(do2: torch.Tensor, wo: torch.Tensor, g2: torch.Tensor, r2: torch.Tensor,
                    shape_key: int | None = None,
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One kernel: d_a=do2@wo (GEMM) + gate-bwd epilogue -> (d_out_r, d_gate, gated).

    ``shape_key`` is ``token_key(L)``, computed by the caller: everything here is already the
    flattened ``(M, DH)`` matrix, and M alone cannot say what L produced it. None -> smallest
    bucket (bench/driver entry only).
    """
    M, DH = g2.shape
    N = wo.shape[0]
    dr = torch.empty_like(g2)
    dg = torch.empty_like(g2)
    a = torch.empty_like(g2)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M1"]), triton.cdiv(DH, META["BLOCK_N"]))
    _dgrad_epi[grid](
        do2, wo, g2, r2, dr, dg, a, M, N, DH,
        do2.stride(0), do2.stride(1), wo.stride(0), wo.stride(1),
        g2.stride(0), g2.stride(1), r2.stride(0), r2.stride(1),
        dr.stride(0), dr.stride(1),
        shape_key=token_key(0) if shape_key is None else shape_key,
    )
    return dr, dg, a


@opaque(fake=lambda gate2d, outr2d, wo, shape_key=None: gate2d.new_empty(
            (gate2d.shape[0], wo.shape[0])),
        name="bias_only_attention_gate_out_fwd")
def _fwd(gate2d: torch.Tensor, outr2d: torch.Tensor, wo: torch.Tensor,
         shape_key: int | None = None) -> torch.Tensor:
    """``shape_key`` is ``token_key(L)`` from the caller (see ``_dgrad_epilogue``)."""
    M, DH = gate2d.shape
    N = wo.shape[0]
    out = torch.empty((M, N), device=gate2d.device, dtype=gate2d.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M1"]), triton.cdiv(N, META["BLOCK_N"]))
    _gate_out_fwd[grid](
        gate2d, outr2d, wo, out,
        M, N, DH,
        gate2d.stride(0), gate2d.stride(1),
        outr2d.stride(0), outr2d.stride(1),
        wo.stride(0), wo.stride(1),
        out.stride(0), out.stride(1),
        shape_key=token_key(0) if shape_key is None else shape_key,
    )
    return out


class _FusedGateOut(torch.autograd.Function):
    """``_fwd`` and ``_dgrad_epilogue`` are each wrapped by ``opaque`` at their definition, so
    these methods need no wrapper -- the reshapes and the wgrad GEMM stay in the graph. See
    ``kernels._compile``."""

    @staticmethod
    def forward(ctx, gate, outr, wo):
        # gate, outr: [..., DH]; wo: [N, DH]
        shape = gate.shape
        DH = shape[-1]
        g2 = gate.reshape(-1, DH).contiguous()
        r2 = outr.reshape(-1, DH).contiguous()
        out2 = _fwd(g2, r2, wo.contiguous(), shape_key=_key_of(shape))
        ctx.save_for_backward(g2, r2, wo)
        ctx.shape = shape
        ctx.N = wo.shape[0]
        return out2.reshape(*shape[:-1], wo.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        g2, r2, wo = ctx.saved_tensors
        N = ctx.N
        do2 = grad_out.reshape(-1, N).contiguous()
        # out = a @ wo^T, a = sigmoid(gate)*out_r. Fuse the dgrad GEMM (d_a=do@wo)
        # with the gate-backward epilogue so d_a never materializes; only the wgrad
        # (d_wo = do^T @ a, needs the materialized gated `a`) stays on cuBLAS.
        d_r, d_g, a = _dgrad_epilogue(do2, wo, g2, r2, shape_key=_key_of(ctx.shape))
        d_wo = do2.transpose(0, 1) @ a              # GEMM  [N, DH]
        return (
            d_g.reshape(ctx.shape),
            d_r.reshape(ctx.shape),
            d_wo,
        )


def fused_gate_out(gate: torch.Tensor, out_r: torch.Tensor, wo: torch.Tensor) -> torch.Tensor:
    """sigmoid(gate) * out_r, then @ wo^T. gate/out_r [...,DH], wo [N,DH] -> [...,N].

    Folds the gate-mul into the to_out GEMM prologue (gated tensor never hits HBM).
    WINS at small DH (<=128); at DH>=256 the wide tl.dot tile degrades (SM90 shared
    pressure) and `sigmoid_gate_fused` + a cuBLAS to_out is faster -- see the d-aware
    dispatch in the module and bench_back_designs.py.
    """
    return _FusedGateOut.apply(gate, out_r, wo)


# ─────────────── split path: one-pass sigmoid*mul (for DH>=256, gate-out via cuBLAS) ──────────


@opaque(fake=lambda gate, out, shape_key: torch.empty_like(gate), name="bias_only_attention_sigmul_fwd")
def _sigmul(gate: torch.Tensor, out: torch.Tensor, shape_key: int) -> torch.Tensor:
    """``sigmoid(gate) * out`` in one pass."""
    a = torch.empty_like(gate)
    n = gate.numel()
    grid = lambda M: (triton.cdiv(n, M["BLOCK_E"]),)
    _sigmul_fwd[grid](gate.contiguous(), out.contiguous(), a, n, shape_key=shape_key)
    return a


@opaque(fake=lambda da, gate, out, shape_key: (torch.empty_like(gate), torch.empty_like(out)),
        name="bias_only_attention_sigmul_bwd")
def _sigmul_grad(da: torch.Tensor, gate: torch.Tensor, out: torch.Tensor,
                 shape_key: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradients of ``sigmoid(gate) * out`` -> ``(dgate, dout)``."""
    dg = torch.empty_like(gate)
    do = torch.empty_like(out)
    n = gate.numel()
    grid = lambda M: (triton.cdiv(n, M["BLOCK_E"]),)
    _sigmul_bwd[grid](da.contiguous(), gate, out, dg, do, n, shape_key=shape_key)
    return dg, do


class _SigmoidGate(torch.autograd.Function):
    # both_key, NOT _key_of. `_sigmul_fwd` belongs to gated_projection and is declared
    # level=both, while `_key_of` is token_key -- whose top bucket is 512. Keying a
    # level=both kernel through it makes every L >= 512 record 512, so its 1024..8192
    # buckets are unreachable AND the same kernel ends up in two bucket spaces, since
    # conditioned_transition/{training,train_fused}.py launch it with both_key. Measured over
    # every bucket: L=512,1024,2048,4096 all recorded shape_key=512 from this path.
    @staticmethod
    def forward(ctx, gate, out):
        a = _sigmul(gate, out, both_key(rows_of(gate.shape)))
        ctx.save_for_backward(gate, out)
        return a

    @staticmethod
    def backward(ctx, da):
        gate, out = ctx.saved_tensors
        return _sigmul_grad(da, gate, out, both_key(rows_of(gate.shape)))


def sigmoid_gate_fused(gate: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """sigmoid(gate) * out in ONE triton pass (vs torch's sigmoid then mul = 2 passes).

    For the DH>=256 back path: this fused elementwise + a cuBLAS to_out beats the
    wide fused tl.dot of `fused_gate_out`. gate/out same shape -> same shape."""
    return _SigmoidGate.apply(gate, out)
