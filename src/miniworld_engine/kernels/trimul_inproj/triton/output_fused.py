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
    for pk0 in range(tl.cdiv(KP, BLOCK_K)):
        pk = pk0 * BLOCK_K + rkp
        pa = tl.load(XN + rm[:, None] * KP + pk[None, :],
                    (rm[:, None] < M) & (pk[None, :] < KP), 0)
        pw = tl.load(WP + pk[:, None] * wp1 + rn[None, :] * wp0,
                    (pk[:, None] < KP) & (rn[None, :] < N), 0)
        ap = tl.dot(pa, pw, ap)
    off = rm[:, None] * N + rn[None, :]
    mask = (rm[:, None] < M) & (rn[None, :] < N)

    rkg = tl.arange(0, BLOCK_K)
    # Put the shorter output-tile axis on the gate dot's row axis. On Hopper,
    # tall second-dot layouts can corrupt aligned operands; unconditionally
    # transposing instead breaks wide, one-warp tiles. This constexpr choice
    # keeps every CSV schedule and the same GEMMs, with no architecture or
    # shape-specific tile blacklist. The projection is saved in the epilogue.
    if BLOCK_M1 > BLOCK_N:
        ag = tl.zeros((BLOCK_N, BLOCK_M1), tl.float32)
    else:
        ag = tl.zeros((BLOCK_M1, BLOCK_N), tl.float32)
    for gk0 in range(tl.cdiv(KG, BLOCK_K)):
        gk = gk0 * BLOCK_K + rkg
        ga = tl.load(X + rm[:, None] * KG + gk[None, :],
                    (rm[:, None] < M) & (gk[None, :] < KG), 0)
        gw = tl.load(WG + gk[:, None] * wg0 + rn[None, :] * wg1,
                    (gk[:, None] < KG) & (rn[None, :] < N), 0)
        if BLOCK_M1 > BLOCK_N:
            ag = tl.dot(tl.trans(gw), tl.trans(ga), ag)
        else:
            ag = tl.dot(ga, gw, ag)
    if BLOCK_M1 > BLOCK_N:
        ag = tl.trans(ag)
    p = ap.to(PROJ.dtype.element_ty)
    tl.store(PROJ + off, p, mask)
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



#: Inference back half (LN_out + output projection + gate + residual): the fused ``back._back_kernel`` serves an LN_out width K up to
#: this; wider K runs ``split_back_infer``. ``_back_kernel`` has no persistent tile over K: past one K tile it re-reads and re-normalises
#: the [BLOCK_M1, K] tri rows for every BLOCK_N slice of the output and runs both GEMMs at BLOCK_M1 = 64 (its register ceiling).
#: B200 (2026-10-31, CUDA graph, bf16, one direction D512 L384): fused 1690 us per module call, F567 split 1376 us, this split
#: (cuBLAS projection) 1173 us; compiled PyTorch 1406-1425 us. Bidirectional D256 (K = 512) L384 917 -> 775 us, D512 (K = 1024) 2734 -> 2041 us.
FUSED_BACK_MAX_K = 256


def split_back_infer(tri_km: torch.Tensor, x_n: torch.Tensor, wp: torch.Tensor, wgt: torch.Tensor, ln_w: torch.Tensor,
                     ln_b: torch.Tensor, eps: float, residual: torch.Tensor, seq_len: int) -> torch.Tensor:
    """y = residual + sigmoid(x_n Wg) (LN_out(tri) Wp^T) for a wide LN_out: ``_ln_materialize`` (one pass over the channel-major
    tri [K, M] -> the normalised rows, token-major bf16), the projection as a plain cuBLAS GEMM, and ``gate_elem_train`` (the gate
    logits by cuBLAS, sigmoid / product / residual in one Triton pass; the identity drop scale). F567 (both GEMMs as Triton tiles,
    the training forward's output stage) ran its two K = 512 products at ~0.35 PFLOP/s on the B200 whatever its tile (swept
    BM 64-128 x BN 64-256 x BK 32-128 x warps x stages: 412-500 us against 155 GFLOP). tri_km [K, M]; x_n [M, KG]; wp (N, K)
    (nn.Linear form); wgt (KG, N); residual [M, N]. Returns y [M, N]."""
    from miniworld_engine.autotune.shape_key import both_key
    from miniworld_engine.kernels.layernorm_linear.triton.te_style import _ln_materialize
    from miniworld_engine.kernels.trimul_inproj.triton.gate_elem import gate_elem_train, ones_dropscale

    m = tri_km.shape[1]
    norm, _, _ = _ln_materialize(tri_km.t(), ln_w, ln_b, eps, shape_key=both_key(m))
    proj = torch.mm(norm, wp.t())
    y, _ = gate_elem_train(x_n, proj, wgt, residual, ones_dropscale(seq_len, wp.shape[0], x_n), seq_len)
    return y
