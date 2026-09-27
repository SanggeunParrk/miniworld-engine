"""Atom attention core against fp64: forward O and LSE (and later the gradients), plus timing."""
import argparse, math, torch
from common import rel, graph_time
from ops import Fwd, Bwd, NH, DH

p = argparse.ArgumentParser(); p.add_argument("--lengths", type=int, nargs="+", default=[384, 768]); p.add_argument("--A", type=int, nargs="+", default=[5, 48])
p.add_argument("--notime", action="store_true"); p.add_argument("--bwd", action="store_true")
a = p.parse_args()


def make(A, N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (torch.randn(A, N, NH * DH, device="cuda", generator=g).to(torch.bfloat16) for _ in range(3))
    bias = torch.randn(NH, N, N, device="cuda", generator=g).to(torch.bfloat16)
    return q, k, v, bias


def ref_fwd(q, k, v, bias, samples):
    """O [len(samples), N, 128] and LSE (log2) [len(samples), NH, N] in fp64, for a subset of samples."""
    A, N, _ = q.shape
    outs, lses = [], []
    for s in samples:
        qh, kh, vh = (t[s].double().view(N, NH, DH).transpose(0, 1) for t in (q, k, v))
        sc = qh @ kh.transpose(-1, -2) / math.sqrt(DH) + bias.double()
        lses.append(torch.logsumexp(sc, -1) / math.log(2.0))
        outs.append((torch.softmax(sc, -1) @ vh).transpose(0, 1).reshape(N, NH * DH))
    return torch.stack(outs), torch.stack(lses)


for L in a.lengths:
    N = 8 * L
    for A in a.A:
        q, k, v, bias = make(A, N)
        run, O, LSE = Fwd().bind(q, k, v, bias)
        run(); torch.cuda.synchronize()
        samp = sorted({0, A // 2, A - 1})
        ro, rl = ref_fwd(q, k, v, bias, samp)
        o = O.view(A, N, -1)[samp]
        print(f"N{N} A{A}: O rel {rel(o, ro):.2e}  LSE max|err| {(LSE[samp].double() - rl).abs().max().item():.2e}  finite {bool(torch.isfinite(O).all())}", flush=True)
        if not a.notime:
            print(f"N{N} A{A}: fwd {graph_time(run):8.1f} us", flush=True)
        if a.bwd:
            do = torch.randn(A, N, NH * DH, device="cuda").to(torch.bfloat16)
            Dd = (do.float().view(A, N, NH, DH) * O.view(A, N, NH, DH)).sum(-1).permute(0, 2, 1).contiguous()
            bias_t = bias.transpose(1, 2).contiguous()
            brun, DQ, DK, DV, DB = Bwd().bind(q, k, v, do, bias, bias_t, LSE, Dd)
            brun(); torch.cuda.synchronize()
            # fp64 truth: per-sample grads for a subset, dbias needs every sample (accumulated one at a time)
            gb = torch.zeros(NH, N, N, dtype=torch.float64, device="cuda"); gq = {}; gk = {}; gv = {}
            for s in range(A):
                qd, kd, vd = (t[s].double().view(N, NH, DH).transpose(0, 1).requires_grad_() for t in (q, k, v))
                bd = bias.double().requires_grad_()
                o = torch.softmax(qd @ kd.transpose(-1, -2) / math.sqrt(DH) + bd, -1) @ vd
                o.backward(do[s].double().view(N, NH, DH).transpose(0, 1))
                gb += bd.grad
                if s in samp:
                    gq[s], gk[s], gv[s] = (t.grad.transpose(0, 1).reshape(N, NH * DH) for t in (qd, kd, vd))
                del qd, kd, vd, bd, o
            pick = lambda T: T.view(A, N, -1)[samp]
            st = lambda d: torch.stack([d[s] for s in samp])
            print(f"N{N} A{A}: dq {rel(pick(DQ), st(gq)):.2e}  dk {rel(pick(DK), st(gk)):.2e}  dv {rel(pick(DV), st(gv)):.2e}  dbias {rel(DB, gb):.2e}"
                  f"  finite {bool(all(torch.isfinite(t).all() for t in (DQ, DK, DV, DB)))}", flush=True)
            del gb
            if not a.notime:
                tk, tq, tb = (graph_time(f) for f in brun.parts)
                print(f"N{N} A{A}: bwd dkv {tk:8.1f}  dq {tq:8.1f}  dbias {tb:8.1f}  = {tk + tq + tb:8.1f} us", flush=True)
