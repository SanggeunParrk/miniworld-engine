"""F567 training: independent projection/gate GEMMs and a saved-output epilogue.

F4 stays in _ln_materialize. All tile/scheduling choices come from the engine's
CSV config sets; the standard autotuner/cache owns selection and cache misses.
Each program owns one (M,N) tile, including y, proj and gate saves. No atomics,
cross-program communication, split-K, or hidden full-width output tile.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from miniworld_engine.kernels._tiles import tile_order, tile_grid

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import token_key
from miniworld_engine.kernels._compile import opaque


def prune_output_configs(configs, named_args, **meta):
    """Remove only oversized K/N tiles and duplicate one-N-tile group schedules.

    Resource feasibility is handled by the engine's compiler/resource tracker;
    there is no architecture-specific register or shared-memory estimate here.
    """
    args = {**named_args, **meta}
    n, kp, kg, m = (int(args[k]) for k in ('N', 'KP', 'KG', 'M'))
    kept = []
    for cfg in configs:
        c = cfg.kwargs
        if (c['BLOCK_N'] > triton.next_power_of_2(n)
                or c['BLOCK_K'] > triton.next_power_of_2(max(kp, kg))):
            continue
        if triton.cdiv(n, c['BLOCK_N']) == 1 and c['GROUP_M'] != 1:
            continue
        if c['GROUP_M'] > triton.cdiv(m, c['BLOCK_M1']):
            continue
        kept.append(cfg)
    # An explicit single-config probe can intentionally exceed an extent to
    # exercise masks. These schedules remain correct; don't make a valid probe
    # set unusable merely because the full search grid has cheaper alternatives.
    return kept or list(configs)


@triton.autotune(
    configs=configs_for('trimul_output_f567_train_triton'),
    key=['shape_key', 'M', 'L', 'wp0', 'wp1', 'wg0', 'wg1'],
    prune_configs_by={'early_config_prune': prune_output_configs},
)
@triton.jit
def _output_f567_kernel(
    XN, X, WP, WG, PROJ, GATE, Y, RES, DS,
    M, L: tl.constexpr, KP: tl.constexpr, KG: tl.constexpr, N: tl.constexpr,
    wp0: tl.constexpr, wp1: tl.constexpr,
    wg0: tl.constexpr, wg1: tl.constexpr,
    BLOCK_M1: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, shape_key,
):
    pid = tl.program_id(0).to(tl.int64)
    nm, nn = tl.cdiv(M, BLOCK_M1), tl.cdiv(N, BLOCK_N)
    pm, pn = tile_order(pid, nm, nn, GROUP_M)
    rm = pm * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)

    # The shared physical K tile keeps both tl.dot operand layouts compatible.
    # KP/KG and loop trip counts remain independent: no padding to the other GEMM
    # extent. BLOCK_K is tunable CSV data, not a fixed kernel constant.
    rkp = tl.arange(0, BLOCK_K)
    ap = tl.zeros((BLOCK_M1, BLOCK_N), tl.float32)
    for k0 in range(tl.cdiv(KP, BLOCK_K)):
        k = k0 * BLOCK_K + rkp
        a = tl.load(XN + rm[:, None] * KP + k[None, :],
                    (rm[:, None] < M) & (k[None, :] < KP), 0)
        w = tl.load(WP + k[:, None] * wp1 + rn[None, :] * wp0,
                    (k[:, None] < KP) & (rn[None, :] < N), 0)
        ap = tl.dot(a, w, ap)
    p = ap.to(PROJ.dtype.element_ty)
    off = rm[:, None] * N + rn[None, :]
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(PROJ + off, p, mask)

    rkg = tl.arange(0, BLOCK_K)
    ag = tl.zeros((BLOCK_M1, BLOCK_N), tl.float32)
    for k0 in range(tl.cdiv(KG, BLOCK_K)):
        k = k0 * BLOCK_K + rkg
        a = tl.load(X + rm[:, None] * KG + k[None, :],
                    (rm[:, None] < M) & (k[None, :] < KG), 0)
        w = tl.load(WG + k[:, None] * wg0 + rn[None, :] * wg1,
                    (k[:, None] < KG) & (rn[None, :] < N), 0)
        ag = tl.dot(a, w, ag)
    # Preserve split cuBLAS BF16 logits and proj rounding, then use the FP32
    # sigmoid for y while storing the BF16 gate expected by the existing bwd.
    g = tl.sigmoid(ag.to(X.dtype.element_ty).to(tl.float32))
    ds = tl.load(DS + (rm % L)[:, None] * N + rn[None, :], mask, 0).to(tl.float32)
    res = tl.load(RES + off, mask, 0).to(tl.float32)
    tl.store(Y + off, p.to(tl.float32) * g * ds + res, mask)
    tl.store(GATE + off, g, mask)


def _output_f567_train_fake(norm, x, wp, wg, residual, dropscale, seq_len):
    """Allocate outputs with the same shape, dtype and strides as output_f567_train."""
    shape = (norm.shape[0], wp.shape[0])
    return tuple((norm.new_empty(shape) for _ in range(3)))


@opaque(fake=_output_f567_train_fake, name='trimul_output_f567_train')
def output_f567_train(
    norm: torch.Tensor, x: torch.Tensor, wp: torch.Tensor, wg: torch.Tensor,
    residual: torch.Tensor, dropscale: torch.Tensor, seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fresh (y, proj, gate), all contiguous; wp=(N,KP), wg=(KG,N).

    Activations/residual/drop scale have the production contiguous layout.
    Weight strides are honored and included in the autotune key. M and actual L
    are also keyed so a partial-row probe cannot reuse a full pair's tune.
    """
    tensors = (norm, x, wp, wg, residual, dropscale)
    if any(t.ndim != 2 for t in tensors):
        raise ValueError('F567 operands must be matrices')
    if any(t.dtype != torch.bfloat16 for t in tensors):
        raise TypeError('F567 currently supports BF16 operands')
    if any(t.device != norm.device for t in tensors) or not norm.is_cuda:
        raise ValueError('F567 operands must share a CUDA device')
    m, kp = norm.shape
    kg, n = wg.shape
    if min(m, kp, kg, n, seq_len) <= 0:
        raise ValueError('F567 dimensions and seq_len must be positive')
    if (x.shape != (m, kg) or wp.shape != (n, kp)
            or residual.shape != (m, n) or dropscale.shape != (seq_len, n)):
        raise ValueError('F567 input/weight/residual/drop-scale shapes disagree')
    if any(not t.is_contiguous() for t in (norm, x, residual, dropscale)):
        raise ValueError('F567 activation/residual/drop-scale operands must be contiguous')
    y, proj, gate = _output_f567_train_fake(*tensors, seq_len)
    grid = lambda meta: tile_grid(m, n, meta['BLOCK_M1'], meta['BLOCK_N'])
    _output_f567_kernel[grid](
        norm, x, wp, wg, proj, gate, y, residual, dropscale,
        m, seq_len, kp, kg, n, *wp.stride(), *wg.stride(),
        shape_key=token_key(seq_len, KP=kp, KG=kg, N=n),
    )
    return y, proj, gate
