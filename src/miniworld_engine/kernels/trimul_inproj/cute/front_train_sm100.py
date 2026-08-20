"""SM100 (B200) trimul FRONT for the v6-faithful TRAINING path — bdll-direct, 0 transposes.

Reproduces the H100 v6 front contract EXACTLY: one call returns
    (left_bdll, right_bdll, preact_bdll)
with ``left``/``right`` each ``[B, D, L, L]`` contiguous and ``preact`` ``[B, 4D, L, L]``
(the pre-GLU logits, channel-major, gate/proj interleaved per side, left then right — the
layout ``triton.back_fused.front_bwd_dW`` consumes). No ``.permute()``/transpose kernel.

WHY a separate sm100 front (vs the sm90 ``trimul_inproj_cute_forward(bdll_direct=True)``):
  The sm90 path hands quack's GATED-GLU epilogue an M-major (bdll) postact view so the GEMM
  writes ``[2D,L,L]`` directly. On quack's sm_100 gated epilogue that M-major store SILENTLY
  HALF-WRITES each tile (verified: left/right cos ~0.05) — the StMatrix m-major store atom
  delivers a fixed 32-channel/thread tile that the GLU N-halving cannot compose with (see
  ``front_sm100.py`` header + project memory ``trimul-sm100-front-fused-win``).

  BUT this is only the GATED epilogue. The finding does NOT apply to a NON-gated store: a plain
  GEMM store has a free layout (no reduction/no N-halving), so an M-major (bdll) postact is
  bit-correct on sm_100 (verified: preact cos 0.999999, left/right cos 0.999997 at L=128..1024).
  So the sm100 front = ONE non-gated m-major GEMM ``x @ b_lr -> preact[4D,L,L]`` + a Triton GLU
  pass ``left/right[d] = sigmoid(preact[gate_d]) * preact[proj_d]`` -> ``[2D,L,L]``. Same DATA as
  the sm90 gated launch (preact + postact), same layout contract, produced with 0 transposes.

  (This is exactly the store-layout freedom the round plan called out: a reduction-free GEMM
  store may pick col/row-major freely — unlike the dgrad LN-bwd epilogue whose reduction axis is
  coupled to the store. The GLU is done as a separate elementwise pass instead of fused into the
  GEMM epilogue; that is an implementation detail, not an algorithm/precision change.)

B=1, square L, bf16 in / fp32 acc / bf16 out.
"""
from __future__ import annotations
from miniworld_engine.autotune.configs import configs_for

import os

import torch
import triton
import triton.language as tl
from miniworld_engine.autotune import tensor_dtype_of
from miniworld_engine.kernels._quack_compat import gemm_act, gemm_act_tuned
from quack.gemm_config import GemmConfig
from miniworld_engine import settings

# v14: fuse the GLU into the GEMM epilogue (kills the preact re-read + the _glu launch).
# Opt-out via MINIWORLD_TRAIN_FRONT_FUSED=0 (falls back to the v13 gemm_act + _glu_bdll path).
_FRONT_FUSED = settings.current().trimul_train_front_fused


def _fused_available(device) -> bool:
    return _FRONT_FUSED and torch.cuda.get_device_capability(device)[0] in (10, 11)


# --- Shape-specialized front-GEMM config (v13) ---------------------------------
# The front is a single non-gated GEMM x_flat(M=L*L, K=D=128) @ b_lr(128, 4D=512)
# -> preact, bf16 in / fp32 acc, with an m-major postact store (postact strides
# (1, M)), activation=None. quack's gemm_act autotuner IS shape-aware (its cache
# key auto-appends each tensor's shape/stride/dtype), but it selects with a short
# noisy do_bench(warmup=5, rep=25) and persists the pick to ~/.quack/cache. An
# exhaustive per-shape sweep of the full valid sm100 config space (148 configs)
# with CUDA-graph do_bench(warmup=25, rep=100) showed:
#   L=384 (M=147456):  best == autotuner pick (tile256x512, cl(2,1), swap_ab, dyn)  -> ceiling
#   L=768 (M=589824):  best == autotuner pick (same)                                -> ceiling
#   L=1024(M=1048576): best = tile256x512, cl(2,1), NO swap_ab, dyn_persistent=False
#                      = 260.0us vs autotuner's dyn_persistent=True 269.2us (~3.5%).
# Pinning the swept-best config here makes the selection deterministic (independent
# of the fragile disk autotune cache), removes the cold-start 148-config autotune,
# and captures the small L=1024 gain. Precision is unchanged (bf16 in / fp32 acc);
# this is config selection only. Non-sm100 devices / unknown M fall back to the
# autotuned gemm_act path.
_FRONT_GEMM_BASE = dict(
    tile_m=256, tile_n=512, cluster_m=2, cluster_n=1, pingpong=False,
    max_swizzle_size=8, device_capacity=10, use_tma_gather=False,
)
_FRONT_GEMM_SPEC = {
    384 * 384:   dict(swap_ab=True,  is_dynamic_persistent=True),
    768 * 768:   dict(swap_ab=True,  is_dynamic_persistent=True),
    1024 * 1024: dict(swap_ab=False, is_dynamic_persistent=False),
}


def _front_gemm_config(M: int, device) -> "GemmConfig | None":
    if torch.cuda.get_device_capability(device)[0] not in (10, 11):
        return None
    spec = _FRONT_GEMM_SPEC.get(M)
    if spec is None:
        return None
    return GemmConfig(**_FRONT_GEMM_BASE, **spec)


def _interleave(Wg: torch.Tensor, Wp: torch.Tensor) -> torch.Tensor:
    """(D,H),(D,H) -> (D,2H): GLU wants (gate, proj) adjacent so preact plane order is
    [g0,p0,g1,p1,...] per side (matches front_bwd_dW's 2d / 2d+1 indexing)."""
    H = Wg.shape[1]
    out = torch.empty(Wg.shape[0], 2 * H, device=Wg.device, dtype=Wg.dtype)
    out[:, 0::2] = Wg
    out[:, 1::2] = Wp
    return out


def prepack_lr_operand_sm100(WL, WLg, WR, WRg) -> torch.Tensor:
    """Build the fused (D, 4H) GLU B-operand ONCE. Cols: [il(WLg,WL) | il(WRg,WR)].
    Contract-identical to trimul_inproj.cute.launch.prepack_lr_operand (drop-in b_lr)."""
    return torch.cat([_interleave(WLg, WL), _interleave(WRg, WR)], dim=1).contiguous()




from miniworld_engine.autotune.buckets import bucket_squared as _bucket


def get_seq_group(rows) -> int:
    """Delegates to canonical size-bucketing (autotune.buckets)."""
    return _bucket(rows)


@triton.autotune(configs=configs_for("gated_projection_gate_packed_mmajor_triton"), key=['H', 'seq_group'])
@triton.jit
def _glu_bdll_kernel(preact, lr, H: tl.constexpr, M, BLOCK_E: tl.constexpr, seq_group):
    """preact (4H,M) channel-major -> lr (2H,M). Per side: even plane=gate, odd=proj.
    Grid over the (H,M) per-side positions; each program emits left plane d AND right plane d.
    int64 offsets (4*H*M can exceed int32 at L=1024, H=128)."""
    Mi = M.to(tl.int64)
    idx = tl.program_id(0).to(tl.int64) * BLOCK_E + tl.arange(0, BLOCK_E).to(tl.int64)
    HM = H * Mi
    mask = idx < HM
    d = idx // Mi
    m = idx - d * Mi
    D2 = 2 * H
    gL = tl.load(preact + (2 * d) * Mi + m, mask=mask, other=0.0).to(tl.float32)
    pL = tl.load(preact + (2 * d + 1) * Mi + m, mask=mask, other=0.0).to(tl.float32)
    gR = tl.load(preact + (D2 + 2 * d) * Mi + m, mask=mask, other=0.0).to(tl.float32)
    pR = tl.load(preact + (D2 + 2 * d + 1) * Mi + m, mask=mask, other=0.0).to(tl.float32)
    et = lr.dtype.element_ty
    tl.store(lr + idx, (tl.sigmoid(gL) * pL).to(et), mask=mask)        # left plane d
    tl.store(lr + HM + idx, (tl.sigmoid(gR) * pR).to(et), mask=mask)   # right plane d


def trimul_front_sm100_train_sig(x_n: torch.Tensor, b_lr: torch.Tensor, H: int):
    """σ(gate) front: returns (left_bdll, right_bdll, sg_bdll) where sg = σ(gate) [B,2H,L,L].
    Same left/right as trimul_front_sm100_train, but the backward-only tensor is sg[2H] (σ(gate))
    instead of preact[4H] (raw gate+proj logits) — ~1/3 fewer front store bytes. The backward
    (`front_bwd_dW_sig`) reconstructs the GLU grads from (left, right, sg). Requires the fused
    sm100 kernel (no v13 fallback)."""
    assert x_n.dim() == 4 and x_n.is_cuda and x_n.is_contiguous()
    B, L, L2, D = x_n.shape
    assert B == 1 and L == L2
    assert _fused_available(x_n.device), "sig front requires the fused sm100 kernel (cap 10/11)"
    M = L * L
    x_flat = x_n.reshape(M, D)
    from miniworld_engine.kernels.trimul_inproj.cute.front_fused_gemm_sm100 import (
        fused_front_gemm_sig,
    )
    Bg = b_lr[:, 0::2].t().contiguous()   # (2H, D) = [WLg | WRg]
    Bp = b_lr[:, 1::2].t().contiguous()   # (2H, D) = [WL  | WR ]
    lr = torch.empty(B, 2 * H, L, L, device=x_n.device, dtype=x_n.dtype)
    sg = torch.empty(B, 2 * H, L, L, device=x_n.device, dtype=x_n.dtype)
    fused_front_gemm_sig(x_flat, Bp, Bg, lr.view(2 * H, M), sg.view(2 * H, M))
    return lr[:, :H], lr[:, H:], sg


def trimul_front_sm100_train(x_n: torch.Tensor, b_lr: torch.Tensor, H: int):
    """SM100 v6-faithful front. Returns (left_bdll, right_bdll, preact_bdll):
      left/right : (B, H, L, L) contiguous  (== left.reshape(H,L,L) is a free view)
      preact     : (B, 4H, L, L)            ([g,p] interleaved per side, left|right)
    x_n : (B, L, L, D) contiguous bf16 ; b_lr : (D, 4H) from prepack_lr_operand_sm100.
    """
    assert x_n.dim() == 4 and x_n.is_cuda and x_n.is_contiguous()
    B, L, L2, D = x_n.shape
    assert B == 1 and L == L2
    M = L * L
    x_flat = x_n.reshape(M, D)
    preact = torch.empty(B, 4 * H, L, L, device=x_n.device, dtype=x_n.dtype)

    if _fused_available(x_n.device):
        # (v14) ONE fused persistent GEMM: its epilogue emits BOTH left/right[2H,L,L] (GLU)
        # AND the raw preact[4H,L,L] (interleaved [g,p] planes) straight from the in-TMEM
        # proj/gate accumulators — no preact HBM re-read, no separate GLU launch. Bp/Bg are
        # the proj / gate columns of the interleaved b_lr operand.
        from miniworld_engine.kernels.trimul_inproj.cute.front_fused_gemm_sm100 import (
            fused_front_gemm,
        )
        Bg = b_lr[:, 0::2].t().contiguous()   # (2H, D) = [WLg | WRg]  (gate weights)
        Bp = b_lr[:, 1::2].t().contiguous()   # (2H, D) = [WL  | WR ]  (proj weights)
        lr = torch.empty(B, 2 * H, L, L, device=x_n.device, dtype=x_n.dtype)
        fused_front_gemm(x_flat, Bp, Bg, lr.view(2 * H, M), preact.view(4 * H, M))
        return lr[:, :H], lr[:, H:], preact

    # (v13 fallback) non-gated m-major GEMM into the bdll preact buffer + a Triton GLU pass.
    preact_view = preact.view(4 * H, M).T  # (M,4H) strides (1,M) -> m-major store
    _cfg = _front_gemm_config(M, x_n.device)
    if _cfg is None:
        gemm_act(A=x_flat, B=b_lr, activation=None, store_preact=False, postact_out=preact_view)
    else:
        # Explicit config injection = bypass @autotune, same as gemm_act_out's
        # partial(gemm_act_tuned.fn, config=None) idiom. CUDA-graph safe (no bench).
        gemm_act_tuned.fn(x_flat, b_lr, None, preact_view, None, None, None,
                          None, None, False, config=_cfg)
    lr = torch.empty(B, 2 * H, L, L, device=x_n.device, dtype=x_n.dtype)
    grid = lambda meta: (triton.cdiv(H * M, meta["BLOCK_E"]),)  # noqa: E731
    _glu_bdll_kernel[grid](preact.view(4 * H, M), lr.view(2 * H, M), H=H, M=M,
                           seq_group=get_seq_group(M))
    return lr[:, :H], lr[:, H:], preact
