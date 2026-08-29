"""Composed (two-kernel) INFERENCE path for the token stream (d_hidden >= 256).

The fully-fused b2b (``inference.py``) keeps the whole ``(M, ND)`` SwiGLU tile and
``out_acc[BM, D]`` live in registers, which only fits when K = d_hidden is small (the
atom stream, d=128). For the token stream (d=768) ``BLOCK_K = next_pow2(768) = 1024``
won't compile and ``out_acc[BM, 768]`` spills. So we *compose* two autotuned kernels and
let ``h:(M, ND)`` round-trip HBM:

    kernel A  (expand + SwiGLU):   h = silu(x @ Wa^T) * (x @ Wb^T)        -> HBM
    kernel B  (squeeze + gate):    out = h @ Ws^T
                                   scale = cond @ Wsc^T + b_sc            (DC-tiled, fused)
                                   y     = sigmoid(scale) * out

Both matmuls are K-tiled (loop BLOCK_K over the contraction dim), so any K compiles.
fp32 io with TF32 tensor cores (``input_precision="tf32"``).
"""

from miniworld_engine.autotune.configs import configs_for
import torch

from miniworld_engine.kernels._compile import opaque
import triton
import triton.language as tl




# --- kernel A: expand + SwiGLU -> h:(M, ND) ----------------------------------




# fmt: off
# shape_key's value is L -- the ATOM count (this family is level=atom in kernels/registry.csv) --
# never the row count a kernel receives.
#
# WHERE THAT L COMES FROM, and it is the one thing to know about this family: every entry point
# here is handed an ALREADY-FLATTENED (M, K) activation -- modules/conditioned_transition/module.py
# does `x.reshape(-1, d)` before it calls -- so `length_of` of that 2-D matrix is M = B*A, which is
# the atom count A only when B == 1. The module therefore reads A off the un-flattened activation
# and passes it down as the `length` argument of every entry point; the entry point buckets it once
# and hands the result to the inner launchers as `shape_key`. `length=None` does NOT fall back to
# `length_of(x.shape)` == M for the direct callers that have no un-flattened tensor to read (the
# anything: `length_of` refuses a rank-2 shape, so that path raises with a message saying to compute
# the key at the caller. Every live caller passes `length`; the drivers and checkers pass
# `shape_key=` directly.
from miniworld_engine.autotune.shape_key import atom_key, length_of, pack


@triton.autotune(configs=configs_for("cond_transition_expand_swiglu_triton"), key=['shape_key'])
@triton.jit
def _expand_swiglu_kernel(
    x_ptr, wa_ptr, wb_ptr, h_ptr,
    M, ND, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,    # Wa, Wb: (ND, K) row-major
    stride_hm, stride_hn,
    BLOCK_M1: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    shape_key,
):
    # GROUP_M is the tile VISIT ORDER, and the order is what decides whether x stays in L2.
    # One program computes one (row-tile, col-tile) of h; it reads a row strip of x and a column
    # strip of W. Programs run roughly in launch order, so consecutive ids decide what is reused:
    #
    #   GROUP_M = 1          walk the columns first -- one x strip serves all n_n programs
    #   GROUP_M >= n_m       walk the rows first    -- one W strip serves all n_m programs
    #   in between           blocks of GROUP_M rows, balancing the two
    #
    # This kernel used a 2-D grid, which fixes the order: CUDA varies axis 0 fastest, so
    # `pid_m = program_id(0)` walked the ROWS first and protected W. W is the small operand -- 4.5
    # MB at d=768, inside an A6000's 6 MB L2 whatever the order -- while x is 96 MB and was
    # re-read once per column tile. Measured on an A6000 at M=32768, d=768, tile 64x64 fp32:
    #
    #   2-D grid  4.675 ms      GROUP_M=512 (= n_m)  4.675 ms      <- the same point on this axis
    #   GROUP_M=1 3.806 ms      GROUP_M=8            3.569 ms      <- 1.31x
    #
    # It is ONE axis and the 2-D grid sat at the end of it, so the ladder in the config CSV carries
    # a value larger than any n_m: the tuner can always choose the old behaviour, which it should
    # on a card whose L2 holds x (an H100's 50 MB, a B200's 126 MB) -- there the re-read never
    # happens and this ordering buys nothing. The benefit appeared between x/L2 = 2 and 4 and
    # plateaued by 4; below 2 it was within noise.
    #
    # TUNED AGAINST TUNED, which is the number that decides whether the axis earns its cost: over
    # 455 configs (tile 32..256 square-ish, warps 4/8, stages 2/3/4, both arms free to pick any of
    # them), the best this kernel can do WITHOUT the axis is 64x128 w4 s2 at 3.431 ms and WITH it
    # 256x64 w8 s2 GROUP_M=4 at 3.116 ms -- 1.10x, not the 1.33x a single fixed tile suggests. A
    # fixed tile flatters the axis because the warp/stage pair that suits row-first order is not
    # the one that suits column-first, and holding those fixed hands the old order a bad config.
    #
    # 1.10x is why the ladder is three values and not seven: the axis multiplies this kernel's grid,
    # and 4/16 straddle the measured optimum while 65536 keeps today's behaviour reachable. The
    # other 13 GEMMs on the same 2-D pattern are NOT converted on this evidence -- 10% on one card,
    # from a search that could not run a third of its configs, does not pay for 3x the tuning of
    # each of them.
    pid = tl.program_id(0).to(tl.int64)
    n_m = tl.cdiv(M, BLOCK_M1)
    n_n = tl.cdiv(ND, BLOCK_N)
    per_group = GROUP_M * n_n
    first_m = (pid // per_group) * GROUP_M
    size_m = min(n_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % per_group) % size_m)
    pid_n = (pid % per_group) // size_m
    rows = pid_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < ND
    a = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
    b = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < K
        x = tl.load(
            x_ptr + rows[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=row_mask[:, None] & k_mask[None, :], other=0.0,
        )
        wa = tl.load(
            wa_ptr + cols[None, :] * stride_wn + k[:, None] * stride_wk,
            mask=k_mask[:, None] & col_mask[None, :], other=0.0,
        )
        wb = tl.load(
            wb_ptr + cols[None, :] * stride_wn + k[:, None] * stride_wk,
            mask=k_mask[:, None] & col_mask[None, :], other=0.0,
        )
        a += tl.dot(x, wa, out_dtype=tl.float32, input_precision="tf32")
        b += tl.dot(x, wb, out_dtype=tl.float32, input_precision="tf32")
    h = a * tl.sigmoid(a) * b
    tl.store(
        h_ptr + rows[:, None] * stride_hm + cols[None, :] * stride_hn,
        h, mask=row_mask[:, None] & col_mask[None, :],
    )
# fmt: on


# --- kernel B: squeeze + conditioning gate -> y:(M, D) -----------------------
# D is tiled too (BLOCK_D, power-of-2) so D need not be a power of 2 (D=768) and the
# (BLOCK_M1, BLOCK_D) out/scale tiles stay register-sized. Grid = (M tiles, D tiles).
# Every tile, BLOCK_K_DC included, comes from the CSV. Nothing filters for the running card's
# shared-memory limit: a row that does not fit fails at launch.


# fmt: off
@triton.autotune(configs=configs_for("cond_transition_squeeze_gate_triton"), key=['shape_key'])
@triton.jit
def _squeeze_gate_kernel(
    h_ptr, cond_ptr, ws_ptr, wsc_ptr, bsc_ptr, out_ptr,
    M, ND, D,
    DC: tl.constexpr,
    stride_hm, stride_hn,
    stride_cm, stride_cc,
    stride_sd, stride_sn,     # Ws: (D, ND) row-major
    stride_scd, stride_scc,   # Wsc: (D, DC) row-major
    stride_om, stride_od,
    BLOCK_M1: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K_ND: tl.constexpr, BLOCK_K_DC: tl.constexpr,
    shape_key,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)
    rows = pid_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    dcols = pid_d * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    d_mask = dcols < D

    # out[:, d_tile] = h @ Ws[d_tile, :]^T   (K = ND, tiled)
    out_acc = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
    for n0 in range(0, ND, BLOCK_K_ND):
        n = n0 + tl.arange(0, BLOCK_K_ND)
        n_mask = n < ND
        h = tl.load(
            h_ptr + rows[:, None] * stride_hm + n[None, :] * stride_hn,
            mask=row_mask[:, None] & n_mask[None, :], other=0.0,
        )
        ws_t = tl.load(  # (BLOCK_K_ND, BLOCK_N): Ws[d_tile, n]^T
            ws_ptr + n[:, None] * stride_sn + dcols[None, :] * stride_sd,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0,
        )
        out_acc += tl.dot(h, ws_t, out_dtype=tl.float32, input_precision="tf32")

    # scale[:, d_tile] = cond @ Wsc[d_tile, :]^T + b_sc  (DC tiled), y = sigmoid(scale) * out
    scale = tl.zeros((BLOCK_M1, BLOCK_N), dtype=tl.float32)
    for c0 in range(0, DC, BLOCK_K_DC):
        dc = c0 + tl.arange(0, BLOCK_K_DC)
        dc_mask = dc < DC
        cond = tl.load(
            cond_ptr + rows[:, None] * stride_cm + dc[None, :] * stride_cc,
            mask=row_mask[:, None] & dc_mask[None, :], other=0.0,
        )
        wsc_t = tl.load(  # (BLOCK_K_DC, BLOCK_N): Wsc[d_tile, dc]^T
            wsc_ptr + dcols[None, :] * stride_scd + dc[:, None] * stride_scc,
            mask=dc_mask[:, None] & d_mask[None, :], other=0.0,
        )
        scale += tl.dot(cond, wsc_t, out_dtype=tl.float32, input_precision="tf32")
    bsc = tl.load(bsc_ptr + dcols, mask=d_mask, other=0.0)
    scale += bsc[None, :]
    y = tl.sigmoid(scale) * out_acc
    tl.store(
        out_ptr + rows[:, None] * stride_om + dcols[None, :] * stride_od,
        y, mask=row_mask[:, None] & d_mask[None, :],
    )
# fmt: on


def _expand_swiglu_fake(x, wa, wb, shape_key=None):
    """(M, ND) h -- ND = wa.shape[0], the expanded width."""
    return x.new_empty((x.shape[0], wa.shape[0]))


@opaque(fake=_expand_swiglu_fake, name="conditioned_transition_composed_expand_swiglu")
def _expand_swiglu(x: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor,
                   shape_key: int | None = None) -> torch.Tensor:
    """h = silu(x @ Wa^T) * (x @ Wb^T) -> (M, ND)."""
    M, K = x.shape
    if shape_key is None:
        raise ValueError(
            "shape_key is required here: this launcher receives an already-flattened "
            "(M, D) matrix, and M alone cannot say whether it is L or L*L. Compute the key "
            "at the caller that still holds the pre-flatten shape -- atom_key(length_of(x.shape)) "
            "-- and pass it down. The `None` default is the signature the @opaque fakes share, "
            "not a working fallback: length_of refuses a rank-2 shape."
        )
    ND = wa.shape[0]
    h = torch.empty(M, ND, device=x.device, dtype=x.dtype)
    # 1-D: the kernel derives both tile indices, so GROUP_M can order them (see the kernel).
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M1"]) * triton.cdiv(ND, meta["BLOCK_N"]),)  # noqa: E731
    _expand_swiglu_kernel[grid](
        x, wa, wb, h,
        M, ND, K,
        x.stride(0), x.stride(1),
        wa.stride(0), wa.stride(1),
        h.stride(0), h.stride(1),
        shape_key=pack(shape_key, ND=ND, K=K),
    )
    return h


def _squeeze_gate_fake(h, cond, ws, wsc, bsc, shape_key=None):
    """(M, D) y -- D = ws.shape[0], the squeezed width, not h's ND."""
    return h.new_empty((h.shape[0], ws.shape[0]))


@opaque(fake=_squeeze_gate_fake, name="conditioned_transition_composed_squeeze_gate")
def _squeeze_gate(h: torch.Tensor, cond: torch.Tensor, ws: torch.Tensor, wsc: torch.Tensor,
                  bsc: torch.Tensor, shape_key: int | None = None) -> torch.Tensor:
    """y = sigmoid(cond @ Wsc^T + b_sc) * (h @ Ws^T) -> (M, D)."""
    M, ND = h.shape
    if shape_key is None:
        raise ValueError(
            "shape_key is required here: this launcher receives an already-flattened "
            "(M, D) matrix, and M alone cannot say whether it is L or L*L. Compute the key "
            "at the caller that still holds the pre-flatten shape -- atom_key(length_of(x.shape)) "
            "-- and pass it down. The `None` default is the signature the @opaque fakes share, "
            "not a working fallback: length_of refuses a rank-2 shape."
        )
    D = ws.shape[0]
    DC = cond.shape[1]
    out = torch.empty(M, D, device=h.device, dtype=h.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M1"]), triton.cdiv(D, meta["BLOCK_N"]))  # noqa: E731
    _squeeze_gate_kernel[grid](
        h, cond, ws, wsc, bsc, out,
        M, ND, D, DC,
        h.stride(0), h.stride(1),
        cond.stride(0), cond.stride(1),
        ws.stride(0), ws.stride(1),
        wsc.stride(0), wsc.stride(1),
        out.stride(0), out.stride(1),
        shape_key=pack(shape_key, ND=ND, DC=DC),
    )
    return out


def cond_transition_inference_composed(x, cond, wa, wb, ws, wsc, bsc, length=None):
    """Two-kernel composed inference for the token stream (large d_hidden).

    Same math/signature as ``cond_transition_inference`` but K-tiled, so d>=256 compiles.

    ``length`` is L -- the ATOM count A of the un-flattened activation, supplied by
    modules/conditioned_transition/module.py. None raises: M alone cannot say whether it is L or L*L.
    """
    wa = wa.contiguous(); wb = wb.contiguous(); ws = ws.contiguous()
    wsc = wsc.contiguous(); bsc = bsc.contiguous()
    shape_key = atom_key(length if length is not None else length_of(x.shape))
    h = _expand_swiglu(x, wa, wb, shape_key=shape_key)
    return _squeeze_gate(h, cond, ws, wsc, bsc, shape_key=shape_key)


