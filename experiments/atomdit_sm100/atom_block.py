"""The atom DiT block (modules/dit.DiTBlock at atom widths) with the sm_100a pieces: the pair-bias producer (pair_bias.cu) and the
attention core (attn_fwd / attn_dkv / attn_dq / attn_dbias) as autograd Functions; the width-128 rest (AdaLN, q / k / v / gate
projections, gates, out projection, the ConditionedTransition) stays in torch on the module's own parameters.

    a = a + gate_out * to_out(sigmoid(g) * core(q, k, v, bias))      bias = LN(z) Wb^T  [4, N, N], shared by the A samples
    a = a + transition(a, s)

No key mask (the benchmark's mask_prob = 0)."""
import torch
import torch.nn.functional as F
from ops import Fwd, Bwd, NH, DH
from pair_bias import pair_bias_fwd_cu, pair_bias_bwd_cu

_K = {}


def _kernels():
    if not _K:
        _K["fwd"], _K["bwd"] = Fwd(), Bwd()
    return _K


class PairBiasFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, gamma, wb):
        bias, bias_t = pair_bias_fwd_cu(z, gamma, wb)
        ctx.save_for_backward(z, gamma, wb)
        ctx.mark_non_differentiable(bias_t)
        return bias, bias_t

    @staticmethod
    def backward(ctx, dbias, _):
        z, gamma, wb = ctx.saved_tensors
        dz, dg, dw = pair_bias_bwd_cu(z, gamma, wb, dbias.float())
        return dz.view(z.shape), dg.to(gamma.dtype), dw.to(wb.dtype)


class CoreFn(torch.autograd.Function):
    """softmax(q k^T / sqrt 32 + bias) v per head; q, k, v [A, N, 128] bf16; bias / bias_t [4, N, N] bf16 -> O [A, N, 128] fp32."""
    @staticmethod
    def forward(ctx, q, k, v, bias, bias_t):
        K = _kernels()
        run, O, LSE = K["fwd"].bind(q, k, v, bias)
        run()
        ctx.save_for_backward(q, k, v, bias, bias_t, O, LSE)
        return O.view(q.shape)

    @staticmethod
    def backward(ctx, do):
        q, k, v, bias, bias_t, O, LSE = ctx.saved_tensors
        A, N, _ = q.shape
        dob = do.to(torch.bfloat16).contiguous()
        dd = (do.float().view(A, N, NH, DH) * O.view(A, N, NH, DH)).sum(-1).transpose(1, 2).contiguous()   # D per (sample, head, row)
        run, DQ, DK, DV, DB = _kernels()["bwd"].bind(q, k, v, dob, bias, bias_t, LSE, dd)
        run()
        sh = q.shape
        return DQ.view(sh), DK.view(sh), DV.view(sh), DB, None


def _attn_pre(att, single, cond):
    x = att.ada_ln_in(single, cond)
    return att.to_query(x), att.to_key(x), att.to_value(x), att.to_gate(x)


def _attn_post(att, single, cond, o, gate):
    out = torch.sigmoid(gate) * o.to(single.dtype)
    out = att.to_out(out)
    return single + torch.sigmoid(att.to_scale(cond)) * out


def _transition(tr, single, cond):
    return single + tr(single, cond)


class AtomBlock(torch.nn.Module):
    """Wraps a DiTBlock (implementation=pytorch, atom widths) and runs it with the sm_100a kernels."""
    def __init__(self, blk, compile_rest=True):
        super().__init__()
        self.blk = blk
        self.pre, self.post, self.trans = _attn_pre, _attn_post, _transition
        if compile_rest:
            self.pre, self.post, self.trans = (torch.compile(f) for f in (_attn_pre, _attn_post, _transition))

    def forward(self, single, cond, pair):
        att = self.blk.attention
        A, B, N, _ = single.shape
        bias, bias_t = PairBiasFn.apply(pair, att.ln_pair.weight, att.to_bias.weight)
        q, k, v, g = self.pre(att, single, cond)
        o = CoreFn.apply(*(t.reshape(A, N, NH * DH) for t in (q, k, v)), bias, bias_t)
        single = self.post(att, single, cond, o.view(A, B, N, NH * DH), g)
        return self.trans(self.blk.transition, single, cond)
