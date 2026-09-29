"""fp32 forward Triton kernels of the fused SWA atom DiT block (kernel family ``swa_dit``).

The fp32 twin of ``forward.py`` for callers whose atom transformer runs in fp32 (MiniWorld v1.3: fp32 parameters, no
autocast). The residual stream, the output, the hoisted modulation, cos / sin and the weights are fp32, and so is every
elementwise step: the RMSNorms, the adaLN modulate, the per-head q/k RMSNorm, RoPE, the sigmoid gate, the SwiGLU
nonlinearity, the gated residual adds. The GEMMs follow the per-op fp32 path this block replaces (team-gm
``SWAAtomBlock`` MINIWORLD branch = engine ``SWADiTBlock`` per-op path): cuBLAS fp32 ``nn.Linear`` projections, which
MiniWorld's trainer runs as TF32 (``torch.set_float32_matmul_precision("medium")``), around a FlashAttention-4 core on
bf16 operands. Per GEMM (the backward's twins in ``backward_fp32.py`` use the same group constants):

  attention (QK^T, PV)            bf16 operands, fp32 accumulate and softmax, as FA4: Q / K / V are written head-major in
                                  bf16 (one rounding of the fp32 RoPE output, where FA4's cast rounds) and the window
                                  attention is ``forward._swa_attn_fwd_kernel`` unchanged, writing O in bf16.
  x Wq^T, x Wk^T, x Wv^T, x Wg^T  ``QKVG_DOT = "tf32"``
  (sigmoid(g) * o) Wo^T           ``OUT_DOT = "tf32"`` (o is the bf16 attention output widened to fp32, as the per-op
                                  path's ``.to(in_dtype)`` does)
  y Wu^T, h Wd^T                  ``FFN_DOT = "tf32x3"``

Chosen by measurement, not by analogy. Triton's single-pass "tf32" truncates its operands, where cuBLAS TF32 rounds
them, so the same nominal precision is not the same error. Against the fp32 reference (TF32 off), H100, A=3 x B=2,
S=333, relative Frobenius error, per-op path under "medium" in brackets:

  all "tf32"                      out 5.8e-4 [1.3e-4], dx 7.9e-4 [2.4e-4], dcond 2.4e-3 [6.6e-4], dW_ffn 1.7e-3 [4.4e-4]
  FFN "tf32x3", the rest "tf32"   same as all "tf32x3" to within 5%: out 1.4e-4, dx 2.2e-4, dcond 6.3e-4, dW_ffn 4.9e-4
  all "tf32x3" / all "ieee"       out 1.3e-4, dx 2.2e-4, dcond 6.2e-4, dW_ffn 4.9e-4 (identical to each other)

(the Wqkv / gate / out weight gradients sit at 3-4e-3 in every variant and in the per-op path: that is the bf16
attention). So the FFN GEMMs are the only ones whose tf32 truncation shows, and they alone pay for "tf32x3". Cost, fwd
+ bwd of 3 blocks at N=1, S=4096 under a CUDA graph: all "tf32" 0.77 ms, this mix 0.87 ms, all "tf32x3" 1.11 ms, all
"ieee" 5.69 ms, the per-op fp32 path 1.83 ms.

The modulation projection (``rmsnorm_adamod`` computes it with ``input_precision="ieee"``) is not in these kernels: it is
hoisted, ``interface.swa_dit_hoist_modulation``, a torch fp32 matmul that follows the process's matmul precision.
"""
import triton
import triton.language as tl

from miniworld_engine.autotune.configs import configs_for
from miniworld_engine.kernels.swa_dit.triton.forward import _head_rms, _rope

#: ``tl.dot`` input precision per GEMM group -- see the module docstring for the measurement behind each.
QKVG_DOT = tl.constexpr("tf32")
OUT_DOT = tl.constexpr("tf32")
FFN_DOT = tl.constexpr("tf32x3")


# ================================================================ ① qkvg (fp32) =======================================================
@triton.autotune(configs=configs_for("swa_dit_inproj_fwd_fp32_triton"), key=["shape_key", "SAVE"])
@triton.jit(do_not_specialize=["NROWS", "S", "B"])
def _swa_qkvg_fwd_fp32_kernel(Q, MOD, COS, SIN, WQKV, WG, QO, KO, VO, GO, RSTD, XS, PQS, PKS, NROWS, S, B, eps, qk_eps, shape_key,
                              C: tl.constexpr, H: tl.constexpr, D: tl.constexpr, MODW: tl.constexpr, SAVE: tl.constexpr,
                              BR: tl.constexpr):
    """Q [N*S, C] fp32 -> QO / KO / VO head-major [N, H, S, D] bf16 (attention operands), GO [N*S, C] fp32; with SAVE, the
    modulated x and the pre-norm q / k projections (fp32) for the backward."""
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BR + tl.arange(0, BR).to(tl.int64); ok = rows < NROWS
    cc = tl.arange(0, C).to(tl.int64)
    n = rows // S; s = rows - n * S; mrow = (n % B) * S + s                          # (b, atom) row of the hoisted modulation
    m2 = ok[:, None]
    q = tl.load(Q + rows[:, None] * C + cc[None, :], mask=m2, other=0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(q * q, axis=1) / C + eps)
    tl.store(RSTD + rows, rstd, mask=ok)
    shift = tl.load(MOD + mrow[:, None] * MODW + cc[None, :], mask=m2, other=0.0)
    scale = tl.load(MOD + mrow[:, None] * MODW + C + cc[None, :], mask=m2, other=0.0)
    x = q * rstd[:, None] * (1.0 + scale) + shift
    if SAVE:
        tl.store(XS + rows[:, None] * C + cc[None, :], x, mask=m2)
    HALF: tl.constexpr = D // 2
    hc = tl.arange(0, HALF).to(tl.int64)
    cs = tl.load(COS + mrow[:, None] * HALF + hc[None, :], mask=m2, other=1.0)
    sn = tl.load(SIN + mrow[:, None] * HALF + hc[None, :], mask=m2, other=0.0)
    wofs = cc[None, :] * C + cc[:, None]                                              # W^T tile [in][out]
    hm = ((n[:, None] * H + (cc[None, :] // D)) * S + s[:, None]) * D + (cc[None, :] % D)
    pq = tl.dot(x, tl.load(WQKV + wofs), input_precision=QKVG_DOT)
    if SAVE:
        tl.store(PQS + rows[:, None] * C + cc[None, :], pq, mask=m2)
    tl.store(QO + hm, _rope(_head_rms(pq, qk_eps, BR, H, D), cs, sn, BR, H, D).to(tl.bfloat16), mask=m2)
    pk = tl.dot(x, tl.load(WQKV + C * C + wofs), input_precision=QKVG_DOT)
    if SAVE:
        tl.store(PKS + rows[:, None] * C + cc[None, :], pk, mask=m2)
    tl.store(KO + hm, _rope(_head_rms(pk, qk_eps, BR, H, D), cs, sn, BR, H, D).to(tl.bfloat16), mask=m2)
    pv = tl.dot(x, tl.load(WQKV + 2 * C * C + wofs), input_precision=QKVG_DOT)
    tl.store(VO + hm, pv.to(tl.bfloat16), mask=m2)
    pg = tl.dot(x, tl.load(WG + wofs), input_precision=QKVG_DOT)
    tl.store(GO + rows[:, None] * C + cc[None, :], pg, mask=m2)


# ================================================================ ③ out-proj + gated residual + FFN (fp32) ============================
@triton.autotune(configs=configs_for("swa_dit_output_swiglu_fwd_fp32_triton"), key=["shape_key", "SAVE"])
@triton.jit(do_not_specialize=["NROWS", "S", "B"])
def _swa_oproj_ffn_fwd_fp32_kernel(QI, O, G, MOD, WO, WU, WD, Q1, OUT, RSTD, ATS, YS, FFS, NROWS, S, B, eps, shape_key,
                                   C: tl.constexpr, NHID: tl.constexpr, MODW: tl.constexpr, SAVE: tl.constexpr, BR: tl.constexpr,
                                   NH: tl.constexpr):
    """q1 = q + gate_a * ((sigmoid(g) * o) Wo^T);  out = q1 + gate_f * SwiGLU(rmsnorm(q1) * (1 + scale_f) + shift_f).
    QI / G / OUT fp32 [N*S, C], O the bf16 attention output; with SAVE, q1, the attention branch, y and the FFN branch."""
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BR + tl.arange(0, BR).to(tl.int64); ok = rows < NROWS
    cc = tl.arange(0, C).to(tl.int64)
    n = rows // S; s = rows - n * S; mrow = (n % B) * S + s
    m2 = ok[:, None]
    ofs = rows[:, None] * C + cc[None, :]
    g = tl.load(G + ofs, mask=m2, other=0.0)
    gated = tl.sigmoid(g) * tl.load(O + ofs, mask=m2, other=0.0).to(tl.float32)
    att = tl.dot(gated, tl.load(WO + cc[None, :] * C + cc[:, None]), input_precision=OUT_DOT)
    ga = tl.load(MOD + mrow[:, None] * MODW + 2 * C + cc[None, :], mask=m2, other=0.0)
    q1 = tl.load(QI + ofs, mask=m2, other=0.0) + ga * att
    if SAVE:
        tl.store(Q1 + ofs, q1, mask=m2)
        tl.store(ATS + ofs, att, mask=m2)
    rstd = 1.0 / tl.sqrt(tl.sum(q1 * q1, axis=1) / C + eps)
    if SAVE:
        tl.store(RSTD + rows, rstd, mask=ok)
    shift = tl.load(MOD + mrow[:, None] * MODW + 3 * C + cc[None, :], mask=m2, other=0.0)
    scale = tl.load(MOD + mrow[:, None] * MODW + 4 * C + cc[None, :], mask=m2, other=0.0)
    y = q1 * rstd[:, None] * (1.0 + scale) + shift
    if SAVE:
        tl.store(YS + ofs, y, mask=m2)
    nh = tl.arange(0, NH).to(tl.int64)
    acc = tl.zeros((BR, C), dtype=tl.float32)
    for j0 in range(0, NHID, NH):
        hid = j0 + nh
        a = tl.dot(y, tl.load(WU + hid[None, :] * C + cc[:, None]), input_precision=FFN_DOT)
        b = tl.dot(y, tl.load(WU + (NHID + hid)[None, :] * C + cc[:, None]), input_precision=FFN_DOT)
        hh = a * tl.sigmoid(a) * b
        acc += tl.dot(hh, tl.load(WD + cc[None, :] * NHID + hid[:, None]), input_precision=FFN_DOT)
    gf = tl.load(MOD + mrow[:, None] * MODW + 5 * C + cc[None, :], mask=m2, other=0.0)
    if SAVE:
        tl.store(FFS + ofs, acc, mask=m2)
    tl.store(OUT + ofs, q1 + gf * acc, mask=m2)
