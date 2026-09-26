"""The sm_100a augmented pair-bias attention as one autograd op: attn_fwd2 forward, then attn_dqb + attn_dkv backward.

    o = augattn(q, k, v, bias)      q, k, v [A, 1, L, 16, 48] bf16, bias [16, L, L] bf16 (head-major, as the token DiT hoists it)
                                    -> o [A, 1, L, 16, 48] fp32;  grads: dq, dk, dv fp32, dbias [16, L, L] fp32

Backward glue (one Triton kernel, as the sm_90a op's _prep_do): dO -> bf16 and D = rowsum(dO O) [A, H, L]; plus the transposed bias
[H, L(key), L(query)] for attn_dkv. The dQ buffer is zeroed (attn_dqb adds one partial per 128-key chunk)."""
import torch
import triton
import triton.language as tl
from common import H, D
from ops import Fwd2, Dqb, Dkv


@triton.jit
def _prep_do_kernel(do, o, dob, dd, L, NH: tl.constexpr, DH: tl.constexpr, DP: tl.constexpr, ROWS: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    hh = tl.arange(0, NH)
    dcol = tl.arange(0, DP)
    ptr = r[:, None, None] * (NH * DH) + (hh[:, None] * DH + dcol[None, :])[None, :, :]
    msk = (dcol[None, :] < DH)[None, :, :] & (r[:, None, None] >= 0)
    g = tl.load(do + ptr, mask=msk, other=0.0).to(tl.float32)
    x = tl.load(o + ptr, mask=msk, other=0.0)
    tl.store(dob + ptr, g.to(tl.bfloat16), mask=msk)
    s = tl.sum(g * x, axis=2)
    tl.store(dd + ((r // L)[:, None] * NH + hh[None, :]) * L + (r % L)[:, None], s)


@triton.jit
def _transpose_kernel(src, dst, L, BT: tl.constexpr):
    h, i, j = tl.program_id(2), tl.program_id(0) * BT, tl.program_id(1) * BT
    ri, rj = i + tl.arange(0, BT), j + tl.arange(0, BT)
    x = tl.load(src + h * L * L + ri[:, None] * L + rj[None, :])
    tl.store(dst + h * L * L + rj[:, None] * L + ri[None, :], tl.trans(x))


def bias_transpose(bias):
    """[H, L, L] -> [H, L(key), L(query)] (attn_dkv reads its key rows contiguously)."""
    Hh, L, _ = bias.shape
    out = torch.empty_like(bias)
    _transpose_kernel[(L // 64, L // 64, Hh)](bias, out, L, BT=64)
    return out


def prep_do(do, O, A, L):
    do = do.reshape(A * L, H * D).contiguous()
    dob = torch.empty(A * L, H * D, device=do.device, dtype=torch.bfloat16)
    dd = torch.empty(A, H, L, device=do.device, dtype=torch.float32)
    _prep_do_kernel[((A * L) // 8,)](do, O, dob, dd, L, NH=H, DH=D, DP=64, ROWS=8)
    return dob, dd


class Kernels:
    """Loaded cubins (one set per process)."""
    def __init__(self, fwd="build/attn_fwd2.cubin", dqb="build/attn_dqb.cubin", dkv="build/attn_dkv.cubin"):
        self.fwd, self.dqb, self.dkv = Fwd2(fwd), Dqb(dqb), Dkv(dkv)


_K = None


def kernels():
    global _K
    if _K is None:
        _K = Kernels()
    return _K


class AugAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias):
        K = kernels()
        run, O, LSE = K.fwd.bind(q, k, v, bias)
        run()
        ctx.save_for_backward(q, k, v, bias, O, LSE)
        return O.view(q.shape)

    @staticmethod
    def backward(ctx, do):
        q, k, v, bias, O, LSE = ctx.saved_tensors
        A, _, L, _, _ = q.shape
        K = kernels()
        dob, dd = prep_do(do, O, A, L)
        dob = dob.view(q.shape)
        bias_t = bias_transpose(bias)
        rq, DQ, DB = K.dqb.bind(q, k, v, dob, bias, LSE, dd, zeroed=True)
        rk, DK, DV = K.dkv.bind(q, k, v, dob, bias_t, LSE, dd, dq_zero=DQ)
        rk()                                            # dK, dV, and zeros into dQ
        rq()                                            # dQ (reduced over key chunks), dbias
        return DQ.view(q.shape), DK.view(q.shape), DV.view(q.shape), DB


def augattn(q, k, v, bias):
    return AugAttn.apply(q, k, v, bias)
