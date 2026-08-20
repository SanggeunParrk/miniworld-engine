"""SM100 (B200) trimul FRONT: quack gated GEMM (N-major blld) + fast transpose to bdll.

WHY this shape (the SM100 finding):
  quack's SM100 gated-GLU epilogue stores the postact through a TMA atom whose box
  geometry assumes the channel (N) dim is contiguous. With an N-major postact
  ``[M, 2D]`` the kernel is bit-correct (cos 0.999999). With the M-major *bdll*
  postact view ``[2D, L, L]`` (channel strided by L*L) the TMA store silently writes
  only HALF of each tile_M block — output alternates 64-good / 64-bad rows
  (cos ~0.5). So the M-major bdll TMA store the SM90 path relies on is NOT a valid
  layout for quack's SM100 gated epilogue. (No ValueError is raised — it's a silent
  half-tile write; verified at L=64..1024.)

  Pivot (this file): run the SM100 gated GEMM into a fast N-contiguous blld
  ``[M, 2D]`` postact, then a tuned Triton transpose to bdll ``[2D, M]`` (= the
  ``[B, 2D, L, L]`` buffer the downstream bmm needs). The transpose runs at ~6.4
  TB/s (B200 HBM peak), so gemm + transpose still beats the Triton front.

  Output contract matches ``trimul_inproj_cute_forward``: ``(left_bdll, right_bdll)``
  each ``[B, D, L, L]`` contiguous (B=1). Gate handled elsewhere (compute_gate path
  unchanged — not this file's concern; the front target is left/right).
"""

from __future__ import annotations
from miniworld_engine.autotune.configs import configs_for

import torch
import triton
import triton.language as tl


from miniworld_engine.autotune import key_bucket_of, tensor_dtype_of
from miniworld_engine.autotune.shape_key import token_key




@triton.autotune(configs=configs_for("trimul_transpose_triton"), key=['shape_key', 'N'])
@triton.jit
def _transpose_kernel(src_ptr, dst_ptr, M, N, BLOCK_M1: tl.constexpr, BLOCK_N: tl.constexpr,
                      shape_key):
    """src (M,N) row-major -> dst (N,M) row-major. dst[n,m] = src[m,n]."""
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    src = tl.load(
        src_ptr + rm[:, None] * N + rn[None, :],
        mask=(rm[:, None] < M) & (rn[None, :] < N),
        other=0.0,
    )
    tl.store(
        dst_ptr + rn[:, None] * M + rm[None, :],
        tl.trans(src),
        mask=(rn[:, None] < N) & (rm[None, :] < M),
    )


def _transpose_blld_to_bdll(blld: torch.Tensor, out_2d_m: torch.Tensor, *,
                            seq_len: int | None = None) -> None:
    """blld (M, 2D) row-major -> out (2D, M) row-major, in place into out_2d_m.

    ``seq_len`` is L (tokens); both arguments are already flattened (M = L*L rows), so the
    caller has to supply it. None -> smallest bucket (bench/driver entry only).
    """
    M, N = blld.shape
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M1"]), triton.cdiv(N, meta["BLOCK_N"]))  # noqa: E731
    _transpose_kernel[grid](blld, out_2d_m, M, N,
                            shape_key=token_key(seq_len if seq_len is not None else 0))


def _interleave(Wg: torch.Tensor, Wp: torch.Tensor) -> torch.Tensor:
    """(D,D),(D,D) -> (D,2D): glu epilogue wants (gate, proj) adjacent (sigmoid(c0)*c1)."""
    D = Wg.shape[1]
    out = torch.empty(Wg.shape[0], 2 * D, device=Wg.device, dtype=Wg.dtype)
    out[:, 0::2] = Wg
    out[:, 1::2] = Wp
    return out


def prepack_lr_operand_sm100(WL, WLg, WR, WRg) -> torch.Tensor:
    """Build the fused (D, 4D) GLU B-operand ONCE. Cols: [il(WLg,WL) | il(WRg,WR)]."""
    return torch.cat([_interleave(WLg, WL), _interleave(WRg, WR)], dim=1).contiguous()


def trimul_front_sm100(
    x: torch.Tensor,            # (B, L, L, D) bf16, contiguous
    WL=None, WLg=None, WR=None, WRg=None,
    *,
    b_lr: torch.Tensor | None = None,   # pre-packed (D, 4D); skips per-call interleave
):
    """SM100 front. Returns (left_bdll, right_bdll), each (B, D, L, L) contiguous (B=1).

    left  = sigmoid(x@WLg) * (x@WL)   ;  right = sigmoid(x@WRg) * (x@WR).
    """
    from miniworld_engine.kernels._quack_compat import gemm_act
    try:
        from . import _bdll_patch
    except ImportError:
        import _bdll_patch
    _bdll_patch.ensure_sigmoid_act()  # harmless; registers sigmoid for the gate path

    assert x.dim() == 4 and x.is_cuda and x.is_contiguous()
    B, L, L2, D = x.shape
    assert B == 1 and L == L2
    M = L * L
    x_flat = x.reshape(M, D)

    if b_lr is None:
        b_lr = prepack_lr_operand_sm100(WL, WLg, WR, WRg)

    # SM100 gated GEMM -> N-contiguous blld postact (M, 2D): cols [:D]=left, [D:]=right.
    blld = torch.empty(M, 2 * D, device=x.device, dtype=x.dtype)
    gemm_act(A=x_flat, B=b_lr, activation="glu", postact_out=blld, store_preact=False)

    # blld (M, 2D) -> bdll (2D, M) == [B, 2D, L, L]; left/right are the D-plane slices.
    lr = torch.empty(B, 2 * D, L, L, device=x.device, dtype=x.dtype)
    _transpose_blld_to_bdll(blld, lr.view(2 * D, M), seq_len=L)
    return lr[:, :D], lr[:, D:]
