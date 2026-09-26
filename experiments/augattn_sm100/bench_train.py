"""End-to-end: the autograd op against fp64 (all four gradients), then inference / training latency of the sm_100a kernels
(CUDA-graph replay): inference = forward kernel; training = forward + backward glue + dqb + dkv."""
import math, sys, torch
from common import H, D, make, rel, graph_time
from attn_op import augattn, kernels, prep_do, bias_transpose


def ref_grads(q, k, v, bias, do):
    qd, kd, vd, bd = (t.double().requires_grad_() for t in (q, k, v, bias))
    qh, kh, vh = (t[:, 0].transpose(1, 2) for t in (qd, kd, vd))
    o = (torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(D) + bd[None], -1) @ vh).transpose(1, 2)[:, None]
    o.backward(do.double())
    return o.detach(), qd.grad, kd.grad, vd.grad, bd.grad


A = 48
for L in [int(x) for x in (sys.argv[1:] or ["384", "768"])]:
    q, k, v, bias = make(A, L)
    do = torch.randn(A, 1, L, H, D, device="cuda")
    qs, ks_, vs, bs = (t.clone().requires_grad_() for t in (q, k, v, bias))
    o = augattn(qs, ks_, vs, bs)
    o.backward(do)
    ro, gq, gk, gv, gb = ref_grads(q, k, v, bias, do)
    print(f"L{L}: O {rel(o, ro):.2e}  dq {rel(qs.grad, gq):.2e}  dk {rel(ks_.grad, gk):.2e}  dv {rel(vs.grad, gv):.2e}  dbias {rel(bs.grad, gb):.2e}", flush=True)
    del ro, gq, gk, gv, gb
    K = kernels()
    frun, O, LSE = K.fwd.bind(q, k, v, bias)
    frun(); torch.cuda.synchronize()
    dob, dd = prep_do(do, O, A, L)
    bias_t = bias_transpose(bias)
    rq, DQ, DB = K.dqb.bind(q, k, v, dob.view(q.shape), bias, LSE, dd, zeroed=True)
    rk, DK, DV = K.dkv.bind(q, k, v, dob.view(q.shape), bias_t, LSE, dd, dq_zero=DQ)
    st = {}

    def glue():                                         # the backward's glue, as the op runs it (fresh outputs)
        st["d"] = prep_do(do, O, A, L)
        st["b"] = bias_transpose(bias)

    def train():
        frun(); glue(); rk(); rq()
    t_inf = graph_time(frun); t_tr = graph_time(train)
    t_glue = graph_time(glue); t_q = graph_time(rq); t_k = graph_time(rk)
    print(f"L{L}: inference {t_inf:7.1f} us   training {t_tr:7.1f} us   (glue {t_glue:5.1f}, dkv incl. dQ zero {t_k:6.1f}, dqb {t_q:6.1f})", flush=True)
