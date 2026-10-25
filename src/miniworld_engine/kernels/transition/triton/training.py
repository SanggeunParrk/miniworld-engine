"""The training backward of the split Transition: one GEMM with the SwiGLU gate-backward epilogue.

The forward (``main._expand_swiglu_saveact``) saves the two expand halves ``a = xn @ Wa^T`` and
``b = xn @ Wb^T``. The squeeze gradient ``dh = dy @ Ws`` (``Ws`` is ``(D, ND)``) is then the only
GEMM left in front of the gate::

    dA = dh * b * (sig(a) + silu(a) * (1 - sig(a)))        dB = dh * silu(a)

``_dh_gate_kernel`` computes a tile of ``dh``, rounds it to the activation dtype (the boundary the
PyTorch statements have, ``mm`` then the elementwise gate), and applies the gate to the saved ``a``
and ``b`` tiles in the epilogue, writing ``dA`` and ``dB`` side by side into one ``(M, 2*ND)``
buffer: ``dx = dAB @ [Wa; Wb]`` and ``[dWa; dWb] = dAB^T @ xn`` are then single stacked GEMMs. Before
this the backward recomputed both expand GEMMs (``_transition_expand_gatebwd_kernel``) and ran
``dh`` as a separate cuBLAS call: two extra GEMM passes over the ``(M, ND)`` operand.
"""
import torch
import triton
import triton.language as tl

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.autotune.shape_key import pack
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._dot_precision import dot_precision
from miniworld_engine.kernels._tiles import tile_grid, tile_order


@triton.autotune(configs=configs_for("transition_bwd_dh_gate_triton"), key=["shape_key"])
@triton.jit
def _dh_gate_kernel(
    G, W, A, B, DAB, M, ND: tl.constexpr, K: tl.constexpr,
    sgm, sgk, swk, swn, sam, san, sdm, sdn,
    shape_key, BLOCK_M1: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, PREC: tl.constexpr = "ieee",
):
    pm, pn = tile_order(tl.program_id(0).to(tl.int64), tl.cdiv(M, BLOCK_M1),
                        tl.cdiv(ND, BLOCK_N), GROUP_M)
    rows = pm * BLOCK_M1 + tl.arange(0, BLOCK_M1)
    cols = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M1, BLOCK_N), tl.float32)
    for k0 in range(tl.cdiv(K, BLOCK_K)):
        kk = k0 * BLOCK_K + ks
        g = tl.load(G + rows[:, None] * sgm + kk[None, :] * sgk,
                    mask=(rows[:, None] < M) & (kk[None, :] < K), other=0)
        w = tl.load(W + kk[:, None] * swk + cols[None, :] * swn,
                    mask=(kk[:, None] < K) & (cols[None, :] < ND), other=0)
        acc = tl.dot(g, w, acc, input_precision=PREC)
    mask = (rows[:, None] < M) & (cols[None, :] < ND)
    et = DAB.dtype.element_ty
    dh = acc.to(et).to(tl.float32)
    a = tl.load(A + rows[:, None] * sam + cols[None, :] * san, mask=mask, other=0).to(tl.float32)
    b = tl.load(B + rows[:, None] * sam + cols[None, :] * san, mask=mask, other=0).to(tl.float32)
    sig = tl.sigmoid(a)
    silu = a * sig
    off = rows[:, None] * sdm + cols[None, :] * sdn
    tl.store(DAB + off, (dh * b * (sig + silu * (1.0 - sig))).to(et), mask=mask)
    tl.store(DAB + off + ND * sdn, (dh * silu).to(et), mask=mask)


def _dh_gate_fake(dy, ws, a, b, shape_key):
    """``dAB`` stacked ``[dA | dB]``, ``(M, 2*ND)`` in the activation dtype."""
    return a.new_empty((a.shape[0], 2 * a.shape[1]))


@opaque(fake=_dh_gate_fake, name="transition_bwd_dh_gate")
def dh_gate(dy: torch.Tensor, ws: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
            shape_key: int) -> torch.Tensor:
    """``dAB = [dA | dB]`` for ``squeeze(SwiGLU(a, b))`` given ``dy (M, D)``, ``ws (D, ND)`` and the saved
    ``a``, ``b`` ``(M, ND)`` (all one dtype; ``a`` and ``b`` share a layout)."""
    m, k = dy.shape
    nd = ws.shape[1]
    if ws.shape[0] != k or a.shape != (m, nd) or b.stride() != a.stride():
        raise ValueError("dh_gate shapes must be dy[M,D], ws[D,ND], a/b[M,ND]")
    dab = a.new_empty((m, 2 * nd))
    _dh_gate_kernel[lambda c: tile_grid(m, nd, c["BLOCK_M1"], c["BLOCK_N"])](
        dy, ws, a, b, dab, m, nd, k,
        *dy.stride(), *ws.stride(), *a.stride(), *dab.stride(),
        shape_key=pack(shape_key, K=k, ND=nd), PREC=dot_precision(dy.dtype),
    )
    return dab
