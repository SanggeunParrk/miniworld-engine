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

import torch
import triton
from miniworld_engine.autotune.grids import brute, BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_1D
import triton.language as tl

from miniworld_engine.autotune import (
    key_bucket_of,
    make_cache_prune,
    make_device_smem_prune,
    tensor_dtype_of,
)

_configs = brute({"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K})


_bias_only_gate_out_fwd_prune = make_cache_prune(
    "bias_only_gate_out_fwd", dtype_of=tensor_dtype_of("gate_ptr"),
    bucket_of=key_bucket_of("N", "DH"),
)


@triton.autotune(
    configs=_configs, key=["M", "N", "DH"],
    prune_configs_by={"early_config_prune": _bias_only_gate_out_fwd_prune},
)
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_ok = offs_m < M
    n_ok = offs_n < N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k0 in range(0, DH, BLOCK_K):
        kk = k0 + offs_k
        k_ok = kk < DH
        # A-tile = sigmoid(gate) * out_r  -> [BLOCK_M, BLOCK_K]
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


def _dgrad_smem_bytes(config, named_args, kwargs):
    """Conservative static-smem estimate for _dgrad_epi (device-smem prune). The pipelined
    loop loads do[BM, BN] and wo[BN, DH] in bf16; the 2.8x factor calibrates raw tile bytes
    to Triton's real allocation. Only needs to rank configs so the over-limit ones are dropped
    before the autotuner compiles (and dies on) them; the prune always keeps the smallest."""
    dh = None
    if hasattr(named_args, "__contains__") and "DH" in named_args:
        dh = named_args["DH"]
    elif kwargs is not None:
        dh = kwargs.get("DH")
    if dh is None:
        return None
    bm = int(config.kwargs["BM"])
    bn = int(config.kwargs["BN"])
    ns = int(config.num_stages)
    raw = ns * 2 * (bm * bn + bn * int(dh))  # bf16 do[BM,BN] + wo[BN,DH]
    return int(raw * 2.8)


_bias_only_gate_out_bwd_prune = make_cache_prune(
    "bias_only_gate_out_bwd", dtype_of=tensor_dtype_of("do_ptr"),
    bucket_of=key_bucket_of("N", "DH"),
    base_prune=make_device_smem_prune(_dgrad_smem_bytes),
)


@triton.autotune(
    configs=[triton.Config({"BM": bm, "BN": bn}, num_warps=w, num_stages=s)
             for bm in (32, 64, 128) for bn in (64, 128) for w in (4, 8) for s in (2, 3)],
    key=["M", "N", "DH"],
    prune_configs_by={"early_config_prune": _bias_only_gate_out_bwd_prune},
)
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
    BM: tl.constexpr, BN: tl.constexpr,
):
    """Fuses the dgrad GEMM d_a = grad_out @ wo with the gate-backward epilogue:
    d_a is never materialized, gate/out_r are read once. One kernel replaces the
    cuBLAS dgrad + a separate elementwise pass.

    The contraction dim N is TILED (BN) and accumulated, so shared memory is bounded by
    the [BM,BN]+[BN,DH] tiles instead of the full [N,DH] weight. The old single-shot
    ``wo[N, DH]`` load needed ~N*DH*2 bytes of smem (e.g. 128 KB at N=DH=256), which fits
    A100/H100 but exceeds the ~100 KB/SM of sm_86 (RTX A5000/A6000); tiling makes it
    launchable on any GPU. Math is unchanged (same GEMM + epilogue)."""
    pid = tl.program_id(0).to(tl.int64)
    rm = pid * BM + tl.arange(0, BM)
    rh = tl.arange(0, DH)
    mm = rm[:, None] < M
    da = tl.zeros((BM, DH), dtype=tl.float32)                              # [BM, DH] acc
    for n0 in range(0, N, BN):
        rn = n0 + tl.arange(0, BN)
        nmask = rn < N
        do = tl.load(do_ptr + rm[:, None] * s_dom + rn[None, :] * s_don,
                     mask=mm & nmask[None, :], other=0.0)                  # [BM, BN]
        wo = tl.load(wo_ptr + rn[:, None] * s_won + rh[None, :] * s_woh,
                     mask=nmask[:, None], other=0.0)                       # [BN, DH]
        da = tl.dot(do, wo, da)                                           # accumulate [BM, DH]
    s = tl.sigmoid(tl.load(g_ptr + rm[:, None] * s_gm + rh[None, :] * s_gh,
                           mask=mm, other=0.0).to(tl.float32))
    r = tl.load(r_ptr + rm[:, None] * s_rm + rh[None, :] * s_rh, mask=mm, other=0.0).to(tl.float32)
    off = rm[:, None] * s_om + rh[None, :] * s_oh
    tl.store(dr_ptr + off, (s * da).to(dr_ptr.dtype.element_ty), mask=mm)
    tl.store(dg_ptr + off, (da * r * s * (1.0 - s)).to(dg_ptr.dtype.element_ty), mask=mm)
    tl.store(a_ptr + off, (s * r).to(a_ptr.dtype.element_ty), mask=mm)


def _dgrad_epilogue(do2, wo, g2, r2):
    """One kernel: d_a=do2@wo (GEMM) + gate-bwd epilogue -> (d_out_r, d_gate, gated)."""
    M, DH = g2.shape
    N = wo.shape[0]
    dr = torch.empty_like(g2)
    dg = torch.empty_like(g2)
    a = torch.empty_like(g2)
    grid = lambda META: (triton.cdiv(M, META["BM"]),)
    _dgrad_epi[grid](
        do2, wo, g2, r2, dr, dg, a, M, N, DH,
        do2.stride(0), do2.stride(1), wo.stride(0), wo.stride(1),
        g2.stride(0), g2.stride(1), r2.stride(0), r2.stride(1),
        dr.stride(0), dr.stride(1),
    )
    return dr, dg, a


def _fwd(gate2d, outr2d, wo):
    M, DH = gate2d.shape
    N = wo.shape[0]
    out = torch.empty((M, N), device=gate2d.device, dtype=gate2d.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    _gate_out_fwd[grid](
        gate2d, outr2d, wo, out,
        M, N, DH,
        gate2d.stride(0), gate2d.stride(1),
        outr2d.stride(0), outr2d.stride(1),
        wo.stride(0), wo.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


class _FusedGateOut(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable()
    def forward(ctx, gate, outr, wo):
        # gate, outr: [..., DH]; wo: [N, DH]
        shape = gate.shape
        DH = shape[-1]
        g2 = gate.reshape(-1, DH).contiguous()
        r2 = outr.reshape(-1, DH).contiguous()
        out2 = _fwd(g2, r2, wo.contiguous())
        ctx.save_for_backward(g2, r2, wo)
        ctx.shape = shape
        ctx.N = wo.shape[0]
        return out2.reshape(*shape[:-1], wo.shape[0])

    @staticmethod
    @torch.compiler.disable()
    def backward(ctx, grad_out):
        g2, r2, wo = ctx.saved_tensors
        N = ctx.N
        do2 = grad_out.reshape(-1, N).contiguous()
        # out = a @ wo^T, a = sigmoid(gate)*out_r. Fuse the dgrad GEMM (d_a=do@wo)
        # with the gate-backward epilogue so d_a never materializes; only the wgrad
        # (d_wo = do^T @ a, needs the materialized gated `a`) stays on cuBLAS.
        d_r, d_g, a = _dgrad_epilogue(do2, wo, g2, r2)
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
_bias_only_sigmul_fwd_prune = make_cache_prune(
    "bias_only_sigmul_fwd", dtype_of=tensor_dtype_of("g_ptr"),
    bucket_of=key_bucket_of(),
)


@triton.autotune(
    configs=[triton.Config({"BLK": b}, num_warps=w) for b in (1024, 2048, 4096) for w in (4, 8)],
    key=["n"],
    prune_configs_by={"early_config_prune": _bias_only_sigmul_fwd_prune},
)
@triton.jit
def _sigmul_fwd(g_ptr, o_ptr, a_ptr, n, BLK: tl.constexpr):
    off = tl.program_id(0).to(tl.int64) * BLK + tl.arange(0, BLK)
    m = off < n
    g = tl.sigmoid(tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32))
    o = tl.load(o_ptr + off, mask=m, other=0.0).to(tl.float32)
    tl.store(a_ptr + off, (g * o).to(a_ptr.dtype.element_ty), mask=m)


_bias_only_sigmul_bwd_prune = make_cache_prune(
    "bias_only_sigmul_bwd", dtype_of=tensor_dtype_of("g_ptr"),
    bucket_of=key_bucket_of(),
)


@triton.autotune(
    configs=[triton.Config({"BLK": b}, num_warps=w) for b in (1024, 2048, 4096) for w in (4, 8)],
    key=["n"],
    prune_configs_by={"early_config_prune": _bias_only_sigmul_bwd_prune},
)
@triton.jit
def _sigmul_bwd(da_ptr, g_ptr, o_ptr, dg_ptr, do_ptr, n, BLK: tl.constexpr):
    off = tl.program_id(0).to(tl.int64) * BLK + tl.arange(0, BLK)
    m = off < n
    da = tl.load(da_ptr + off, mask=m, other=0.0).to(tl.float32)
    s = tl.sigmoid(tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32))
    o = tl.load(o_ptr + off, mask=m, other=0.0).to(tl.float32)
    tl.store(do_ptr + off, (da * s).to(do_ptr.dtype.element_ty), mask=m)
    tl.store(dg_ptr + off, (da * o * s * (1.0 - s)).to(dg_ptr.dtype.element_ty), mask=m)


class _SigmoidGate(torch.autograd.Function):
    @staticmethod
    @torch.compiler.disable()
    def forward(ctx, gate, out):
        a = torch.empty_like(gate)
        n = gate.numel()
        grid = lambda M: (triton.cdiv(n, M["BLK"]),)
        _sigmul_fwd[grid](gate.contiguous(), out.contiguous(), a, n)
        ctx.save_for_backward(gate, out)
        return a

    @staticmethod
    @torch.compiler.disable()
    def backward(ctx, da):
        gate, out = ctx.saved_tensors
        dg = torch.empty_like(gate)
        do = torch.empty_like(out)
        n = gate.numel()
        grid = lambda M: (triton.cdiv(n, M["BLK"]),)
        _sigmul_bwd[grid](da.contiguous(), gate, out, dg, do, n)
        return dg, do


def sigmoid_gate_fused(gate: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """sigmoid(gate) * out in ONE triton pass (vs torch's sigmoid then mul = 2 passes).

    For the DH>=256 back path: this fused elementwise + a cuBLAS to_out beats the
    wide fused tl.dot of `fused_gate_out`. gate/out same shape -> same shape."""
    return _SigmoidGate.apply(gate, out)
