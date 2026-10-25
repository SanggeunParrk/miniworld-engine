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

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import pack
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._dot_precision import dot_precision
from miniworld_engine.kernels._tiles import tile_grid, tile_order
from miniworld_engine.kernels.transition.triton._tma import num_sms, tma_ok


@triton.autotune(configs=configs_for("transition_bwd_squeeze_swiglu_triton"), key=["shape_key", "USE_TMA"])
@triton.jit
def _squeeze_swiglu_bwd_kernel(
    X, Wab, G, W, H, DAB, M, ND: tl.constexpr, K: tl.constexpr,
    sxm, swabn, sgm, swk, swn, shm, sdm,
    shape_key, BLOCK_M1: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, USE_TMA: tl.constexpr = False,
    NUM_SMS: tl.constexpr = 1, PREC: tl.constexpr = "ieee",
):
    n_m = tl.cdiv(M, BLOCK_M1)
    n_n = tl.cdiv(ND, BLOCK_N)
    et = DAB.dtype.element_ty
    if USE_TMA:
        # Hopper, aligned operands (see _tma.py): tensor-map loads and stores, one persistent program per SM.
        # dA and dB are two maps over the two halves of dAB, each clipped at ND, so a ragged last tile of dA
        # cannot spill into dB.
        x_desc = tl.make_tensor_descriptor(X, [M, K], [sxm, 1], [BLOCK_M1, BLOCK_K])
        w_desc = tl.make_tensor_descriptor(Wab, [2 * ND, K], [swabn, 1], [2 * BLOCK_N, BLOCK_K])
        g_desc = tl.make_tensor_descriptor(G, [M, K], [sgm, 1], [BLOCK_M1, BLOCK_K])
        s_desc = tl.make_tensor_descriptor(W, [K, ND], [swk, 1], [BLOCK_K, BLOCK_N])
        h_desc = tl.make_tensor_descriptor(H, [M, ND], [shm, 1], [BLOCK_M1, BLOCK_N])
        da_desc = tl.make_tensor_descriptor(DAB, [M, ND], [sdm, 1], [BLOCK_M1, BLOCK_N])
        db_desc = tl.make_tensor_descriptor(DAB + ND, [M, ND], [sdm, 1], [BLOCK_M1, BLOCK_N])
        for tile in tl.range(tl.program_id(0), n_m * n_n, NUM_SMS, flatten=True):
            pm, pn = tile_order(tile, n_m, n_n, GROUP_M)
            acc = tl.zeros((BLOCK_M1, 2 * BLOCK_N), tl.float32)
            acc_dh = tl.zeros((BLOCK_M1, BLOCK_N), tl.float32)
            for k0 in range(0, K, BLOCK_K):
                x = x_desc.load([pm * BLOCK_M1, k0])
                w = w_desc.load([pn * (2 * BLOCK_N), k0])
                g = g_desc.load([pm * BLOCK_M1, k0])
                s = s_desc.load([k0, pn * BLOCK_N])
                acc = tl.dot(x, w.T, acc, input_precision=PREC)
                acc_dh = tl.dot(g, s, acc_dh, input_precision=PREC)
            a, b = tl.split(tl.reshape(acc, (BLOCK_M1, BLOCK_N, 2)))
            dh = acc_dh.to(et).to(tl.float32)
            sig = tl.sigmoid(a)
            silu = a * sig
            h_desc.store([pm * BLOCK_M1, pn * BLOCK_N], (silu * b).to(et))
            da_desc.store([pm * BLOCK_M1, pn * BLOCK_N], (dh * b * (sig + silu * (1.0 - sig))).to(et))
            db_desc.store([pm * BLOCK_M1, pn * BLOCK_N], (dh * silu).to(et))
    else:
        pm, pn = tile_order(tl.program_id(0).to(tl.int64), n_m, n_n, GROUP_M)
        rows = pm * BLOCK_M1 + tl.arange(0, BLOCK_M1)
        cols = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        wrows = pn * (2 * BLOCK_N) + tl.arange(0, 2 * BLOCK_N)
        ks = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M1, 2 * BLOCK_N), tl.float32)
        acc_dh = tl.zeros((BLOCK_M1, BLOCK_N), tl.float32)
        for k0 in range(tl.cdiv(K, BLOCK_K)):
            kk = k0 * BLOCK_K + ks
            x = tl.load(X + rows[:, None] * sxm + kk[None, :],
                        mask=(rows[:, None] < M) & (kk[None, :] < K), other=0)
            w = tl.load(Wab + wrows[None, :] * swabn + kk[:, None],
                        mask=(wrows[None, :] < 2 * ND) & (kk[:, None] < K), other=0)
            g = tl.load(G + rows[:, None] * sgm + kk[None, :],
                        mask=(rows[:, None] < M) & (kk[None, :] < K), other=0)
            s = tl.load(W + kk[:, None] * swk + cols[None, :] * swn,
                        mask=(kk[:, None] < K) & (cols[None, :] < ND), other=0)
            acc = tl.dot(x, w, acc, input_precision=PREC)
            acc_dh = tl.dot(g, s, acc_dh, input_precision=PREC)
        a, b = tl.split(tl.reshape(acc, (BLOCK_M1, BLOCK_N, 2)))
        mask = (rows[:, None] < M) & (cols[None, :] < ND)
        dh = acc_dh.to(et).to(tl.float32)
        sig = tl.sigmoid(a)
        silu = a * sig
        tl.store(H + rows[:, None] * shm + cols[None, :], (silu * b).to(et), mask=mask)
        off = rows[:, None] * sdm + cols[None, :]
        tl.store(DAB + off, (dh * b * (sig + silu * (1.0 - sig))).to(et), mask=mask)
        tl.store(DAB + off + ND, (dh * silu).to(et), mask=mask)


def _squeeze_swiglu_bwd_fake(xn, w_ab, dy, ws, shape_key):
    """``(h, dAB)``: ``h`` ``(M, ND)`` and the stacked ``[dA | dB]`` ``(M, 2*ND)``, in the activation dtype."""
    return dy.new_empty((xn.shape[0], ws.shape[1])), dy.new_empty((xn.shape[0], 2 * ws.shape[1]))


@opaque(fake=_squeeze_swiglu_bwd_fake, name="transition_bwd_squeeze_swiglu")
def squeeze_swiglu_bwd(xn: torch.Tensor, w_ab: torch.Tensor, dy: torch.Tensor, ws: torch.Tensor,
                       shape_key: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(h, dAB)`` for ``squeeze(SwiGLU(xn @ Wa^T, xn @ Wb^T))``: ``xn (M, K)``, ``w_ab (2*ND, K)`` the expand weights
    row-interleaved (``torch.stack((wa, wb), 1).reshape(2*ND, K)``), ``dy (M, K)``, ``ws (K, ND)``; one dtype."""
    m, k = xn.shape
    nd = ws.shape[1]
    if ws.shape[0] != k or w_ab.shape != (2 * nd, k) or dy.shape != (m, k):
        raise ValueError("squeeze_swiglu_bwd shapes must be xn[M,K], w_ab[2*ND,K], dy[M,K], ws[K,ND]")
    h = dy.new_empty((m, nd))
    dab = dy.new_empty((m, 2 * nd))
    da, db = dab[:, :nd], dab[:, nd:]
    use_tma = tma_ok(xn, w_ab, dy, ws, h, da, db)
    if use_tma:
        sms = num_sms(dy.device)
        grid = lambda c: (min(sms, tile_grid(m, nd, c["BLOCK_M1"], c["BLOCK_N"])[0]),)  # noqa: E731
    else:
        sms = 1
        grid = lambda c: tile_grid(m, nd, c["BLOCK_M1"], c["BLOCK_N"])  # noqa: E731
    _squeeze_swiglu_bwd_kernel[grid](
        xn, w_ab, dy, ws, h, dab, m, nd, k,
        xn.stride(0), w_ab.stride(0), dy.stride(0), ws.stride(0), ws.stride(1), h.stride(0), dab.stride(0),
        shape_key=pack(shape_key, K=k, ND=nd), USE_TMA=use_tma, NUM_SMS=sms, PREC=dot_precision(dy.dtype),
    )
    return h, dab


def swiglu_squeeze_backward_saved(xn, wa, wb, ws, dy, shape_key):
    """``(dx, dWa, dWb, dWs)`` of ``squeeze(SwiGLU(xn @ Wa^T, xn @ Wb^T))``.

    ``xn (M, K)``, ``wa``/``wb`` ``(ND, K)``, ``ws (K, ND)``, ``dy (M, K)``. One kernel makes ``h`` and ``dAB = [dA | dB]``;
    ``dx = dAB @ [Wa; Wb]`` and ``[dWa; dWb] = dAB^T @ xn`` are then one stacked GEMM each, ``dWs = dy^T @ h``.
    """
    nd = wa.shape[0]
    w_ab = torch.stack((wa, wb), dim=1).reshape(2 * nd, xn.shape[1])
    h, dab = squeeze_swiglu_bwd(xn, w_ab, dy, ws, shape_key)
    dws = torch.matmul(dy.T, h)                               # (K, ND)
    dwab = torch.matmul(dab.T, xn)                            # [dWa; dWb]  (2*ND, K)
    dx = torch.matmul(dab, torch.cat((wa, wb), dim=0))        # (M, K)
    return dx, dwab[:nd], dwab[nd:], dws
