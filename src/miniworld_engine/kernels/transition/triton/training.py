"""The training backward of the split Transition: ONE kernel for the three GEMMs in front of the gate.

For ``h = SwiGLU(a, b)``, ``a = xn @ Wa^T``, ``b = xn @ Wb^T``, ``y = h @ Ws^T`` the backward needs, per tile of the
``(M, ND)`` hidden activation, the two expand halves (recomputed) and the squeeze gradient ``dh = dy @ Ws``::

    dA = dh * b * (sig(a) + silu(a) * (1 - sig(a)))        dB = dh * silu(a)        h = silu(a) * b  (for dWs)

``_squeeze_swiglu_bwd_kernel`` runs the three K = D contractions in one loop (``a``/``b`` as one dot over the row-interleaved
weight, exactly as the forward does, and ``dh``), rounds ``dh`` to the activation dtype (the boundary of the PyTorch statements,
``mm`` then the elementwise gate) and applies the gate to the fp32 ``a`` and ``b`` that never leave registers. It writes ``h`` and
``dA`` / ``dB`` side by side into one ``(M, 2*ND)`` buffer, so ``dx = dAB @ [Wa; Wb]`` and ``[dWa; dWb] = dAB^T @ xn`` are single
stacked GEMMs. The forward saves nothing for this (``a`` and ``b`` as bf16 moved ``dx`` by ~4e-3 against the hand-CUDA kernels, which
keep them in fp32; as fp32 they cost two extra ``(M, ND)`` passes each way). Before this the backward ran ``dh`` as a cuBLAS call and
``_transition_expand_gatebwd_kernel`` (two expand GEMMs, a gate, three stores) after it.
"""
import torch
import triton
import triton.language as tl

from miniworld_engine.kernels._tf32 import tf32_cached, tf32_dot, tf32_grid_many

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import pack
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._dot_precision import dot_precision
from miniworld_engine.kernels._tiles import tile_grid, tile_order

@triton.autotune(configs=configs_for("transition_bwd_squeeze_swiglu_triton"), key=["shape_key", "SAVED", "HALF"])
@triton.jit
def _squeeze_swiglu_bwd_kernel(
    X, Wab, G, W, H, DAB, SA, SB, SCALE, M, ND: tl.constexpr, K: tl.constexpr,
    sxm, swabn, sgm, swk, swn, shm, sdm, ssm,
    shape_key, BLOCK_M1: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
    SAVED: tl.constexpr = False, HALF: tl.constexpr = False,
    PREC: tl.constexpr = "ieee",
    PRE: tl.constexpr = False,
):
    n_m = tl.cdiv(M, BLOCK_M1)
    n_n = tl.cdiv(ND, BLOCK_N)
    et = DAB.dtype.element_ty
    pm, pn = tile_order(tl.program_id(0).to(tl.int64), n_m, n_n, GROUP_M)
    rows = pm * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    cols = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    wrows = pn * (2 * BLOCK_N) + tl.arange(0, 2 * BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M1, 2 * BLOCK_N), tl.float32)
    acc_dh = tl.zeros((BLOCK_M1, BLOCK_N), tl.float32)
    for k0 in range(tl.cdiv(K, BLOCK_K)):
        kk = k0 * BLOCK_K + ks
        g = tl.load(G + rows[:, None] * sgm + kk[None, :],
                    mask=(rows[:, None] < M) & (kk[None, :] < K), other=0)
        s = tl.load(W + kk[:, None] * swk + cols[None, :] * swn,
                    mask=(kk[:, None] < K) & (cols[None, :] < ND), other=0)
        acc_dh = tf32_dot(g, s, acc_dh, input_precision=PREC, pre_b=PRE)
        if not SAVED:
            x = tl.load(X + rows[:, None] * sxm + kk[None, :],
                        mask=(rows[:, None] < M) & (kk[None, :] < K), other=0)
            w = tl.load(Wab + wrows[None, :] * swabn + kk[:, None],
                        mask=(wrows[None, :] < 2 * ND) & (kk[:, None] < K), other=0)
            acc = tf32_dot(x, w, acc, input_precision=PREC, pre_b=PRE)
    mask = (rows[:, None] < M) & (cols[None, :] < ND)
    if SAVED:
        a = tl.load(SA + rows[:, None] * ssm + cols[None, :], mask=mask, other=0).to(tl.float32)
        b = tl.load(SB + rows[:, None] * ssm + cols[None, :], mask=mask, other=0).to(tl.float32)
        if HALF:
            a = a * tl.load(SCALE)
            b = b * tl.load(SCALE)
    else:
        a, b = tl.split(tl.reshape(acc, (BLOCK_M1, BLOCK_N, 2)))
    dh = acc_dh.to(et).to(tl.float32)
    sig = tl.sigmoid(a)
    silu = a * sig
    if not SAVED:
        tl.store(H + rows[:, None] * shm + cols[None, :], (silu * b).to(et), mask=mask)
    off = rows[:, None] * sdm + cols[None, :]
    tl.store(DAB + off, (dh * b * (sig + silu * (1.0 - sig))).to(et), mask=mask)
    tl.store(DAB + off + ND, (dh * silu).to(et), mask=mask)


def _launch_bwd(xn, w_ab, dy, ws, shape_key, saved_a, saved_b, scale=None):
    """Launch the kernel: recompute a, b from ``xn``/``w_ab`` (``saved_a is None``, writes ``h``) or read them (``saved_a``,
    ``saved_b``, the forward's halves; ``xn`` / ``w_ab`` are then unused and ``h`` is not written). ``scale`` (fp32 on the device):
    the halves are fp16 and were stored divided by it."""
    saved = saved_a is not None
    m, k = dy.shape
    nd = ws.shape[1]
    if ws.shape[0] != k:
        raise ValueError("squeeze_swiglu_bwd shapes must be dy[M,K], ws[K,ND]")
    half = saved and scale is not None
    prec = dot_precision(dy.dtype)
    pre = prec == "tf32" and dy.dtype is torch.float32     # TF32 operands rounded to nearest here, once (kernels/_tf32.py)
    if pre:
        ws = tf32_cached((ws,), lambda: tf32_grid_many(ws))[0]
        if not saved:
            w_ab = tf32_grid_many(w_ab)[0]
    if saved:
        if saved_a.shape != (m, nd) or saved_b.stride() != saved_a.stride():
            raise ValueError("squeeze_swiglu_bwd: saved a/b must be [M,ND] with one layout")
        h = xn = w_ab = dy                                   # unused fillers
    else:
        if w_ab.shape != (2 * nd, k) or xn.shape != (m, k):
            raise ValueError("squeeze_swiglu_bwd shapes must be xn[M,K], w_ab[2*ND,K], dy[M,K], ws[K,ND]")
        h = dy.new_empty((m, nd))
        saved_a = saved_b = dy
    if not half:
        scale = dy
    dab = dy.new_empty((m, 2 * nd))
    grid = lambda c: tile_grid(m, nd, c["BLOCK_M1"], c["BLOCK_N"])  # noqa: E731

    _squeeze_swiglu_bwd_kernel[grid](
        xn, w_ab, dy, ws, h, dab, saved_a, saved_b, scale, m, nd, k,
        xn.stride(0), w_ab.stride(0), dy.stride(0), ws.stride(0), ws.stride(1), h.stride(0), dab.stride(0),
        saved_a.stride(0),
        shape_key=pack(shape_key, K=k, ND=nd), SAVED=saved, HALF=half, PREC=prec, PRE=pre,
    )
    return h, dab

def _squeeze_swiglu_bwd_fake(xn, w_ab, dy, ws, shape_key):
    """``(h, dAB)``: ``h`` ``(M, ND)`` and the stacked ``[dA | dB]`` ``(M, 2*ND)``, in the activation dtype."""
    return dy.new_empty((xn.shape[0], ws.shape[1])), dy.new_empty((xn.shape[0], 2 * ws.shape[1]))

@opaque(fake=_squeeze_swiglu_bwd_fake, name="transition_bwd_squeeze_swiglu")
def squeeze_swiglu_bwd(xn: torch.Tensor, w_ab: torch.Tensor, dy: torch.Tensor, ws: torch.Tensor,
                       shape_key: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(h, dAB)`` for ``squeeze(SwiGLU(xn @ Wa^T, xn @ Wb^T))``: ``xn (M, K)``, ``w_ab (2*ND, K)`` the expand weights
    row-interleaved (``torch.stack((wa, wb), 1).reshape(2*ND, K)``), ``dy (M, K)``, ``ws (K, ND)``; one dtype. a, b are
    recomputed in fp32 inside the kernel."""
    return _launch_bwd(xn, w_ab, dy, ws, shape_key, None, None)

def _squeeze_swiglu_bwd_saved_fake(dy, ws, a, b, shape_key, scale=None):
    """``dAB`` ``(M, 2*ND)`` in the activation dtype."""
    return dy.new_empty((dy.shape[0], 2 * ws.shape[1]))

@opaque(fake=_squeeze_swiglu_bwd_saved_fake, name="transition_bwd_squeeze_swiglu_saved")
def squeeze_swiglu_bwd_saved(dy: torch.Tensor, ws: torch.Tensor, a: torch.Tensor, b: torch.Tensor, shape_key: int,
                             scale: torch.Tensor | None = None) -> torch.Tensor:
    """``dAB = [dA | dB]`` from ``dy (M, K)``, ``ws (K, ND)`` and the expand halves ``a``, ``b`` ``(M, ND)`` the forward saved (the
    activation dtype, rounded like the PyTorch statements hold them; under fp32 activations with a ``scale`` they are fp16, the
    TF32 mantissa, stored divided by that fp32 device scalar)."""
    return _launch_bwd(dy, dy, dy, ws, shape_key, a, b, scale)[1]

def swiglu_squeeze_backward_saved(xn, wa, wb, ws, dy, shape_key, saved=None, half_scale=None):
    """``(dx, dWa, dWb, dWs)`` of ``squeeze(SwiGLU(xn @ Wa^T, xn @ Wb^T))``.

    ``xn (M, K)``, ``wa``/``wb`` ``(ND, K)``, ``ws (K, ND)``, ``dy (M, K)``; ``saved = (h, a, b)`` are the training forward's
    ``h = SwiGLU(a, b)`` and halves (``main._expand_swiglu_saveact``; ``half_scale``: they are fp16, divided by it) -- without them a, b are recomputed. One kernel makes
    ``dAB = [dA | dB]``; ``dx = dAB @ [Wa; Wb]`` and ``[dWa; dWb] = dAB^T @ xn`` are then one stacked GEMM each, ``dWs = dy^T @ h``.
    """
    nd = wa.shape[0]
    if saved is None:
        w_ab = torch.stack((wa, wb), dim=1).reshape(2 * nd, xn.shape[1])
        h, dab = squeeze_swiglu_bwd(xn, w_ab, dy, ws, shape_key)
    else:
        h, a, b = saved
        dab = squeeze_swiglu_bwd_saved(dy, ws, a, b, shape_key, half_scale)
    dws = torch.matmul(dy.T, h)                               # (K, ND)
    dwab = torch.matmul(dab.T, xn)                            # [dWa; dWb]  (2*ND, K)
    dx = torch.matmul(dab, torch.cat((wa, wb), dim=0))        # (M, K)
    return dx, dwab[:nd], dwab[nd:], dws
