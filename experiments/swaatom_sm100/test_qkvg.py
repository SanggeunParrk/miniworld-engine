"""qkvg_fwd.cu against the H100 fused block's Triton _qkvg_fwd (same rounding points: expected bitwise or 1-ulp close) and timing."""
import argparse, torch, triton
from common import make, hoist_mod, rel, graph_time, C, H, D, EPS
import h100_fused as SF
from ops import QkvgFwd

p = argparse.ArgumentParser(); p.add_argument("--cases", nargs="+", default=["48:3072", "5:3072", "48:6144", "5:6144"])
a = p.parse_args()
for case in a.cases:
    A, S = map(int, case.split(":"))
    q, cb, cos, sin, su, w = make(A, S)
    mod = hoist_mod(cb, w["wmod"])
    N, M = A, A * S
    for save in (False, True):
        run, (Qh, Kh, Vh, G, X, PQ, PK) = QkvgFwd().bind(q, mod, cos, sin, w["wqkv"], w["wg"], A, 1, save=save)
        run(); torch.cuda.synchronize()
        # Triton reference (the H100 fused module's own kernel)
        rQ, rK, rV = (torch.empty(N, H, S, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        rG, rX, rPQ, rPK = (torch.empty(M, C, device="cuda", dtype=torch.bfloat16) for _ in range(4))
        r1 = torch.empty(M, device="cuda")
        SF._qkvg_fwd[lambda m: (triton.cdiv(M, m["BR"]),)](q.reshape(M, C), mod, cos, sin, w["wqkv"], w["wg"], rQ, rK, rV, rG, r1, rX, rPQ, rPK, M, S, 1,
                                                           EPS, EPS, SF._bucket(M), C=C, H=H, D=D, MODW=6 * C, SAVE=True)
        torch.cuda.synchronize()
        diff = lambda x, y: f"{rel(x, y):.1e}/{(x.float() != y.float()).float().mean().item() * 100:.2f}%"
        msg = f"A{A} S{S} save={int(save)}: Q {diff(Qh, rQ)}  K {diff(Kh, rK)}  V {diff(Vh, rV)}  G {diff(G.view(M, C), rG)}"
        if save:
            msg += f"  X {diff(X, rX)}  PQ {diff(PQ, rPQ)}  PK {diff(PK, rPK)}"
        print(msg + "   (rel / % elements differing)", flush=True)
        t = graph_time(run)
        tt = graph_time(lambda: SF._qkvg_fwd[lambda m: (triton.cdiv(M, m["BR"]),)](q.reshape(M, C), mod, cos, sin, w["wqkv"], w["wg"], rQ, rK, rV, rG, r1,
                                                                                       rX, rPQ, rPK, M, S, 1, EPS, EPS, SF._bucket(M), C=C, H=H, D=D, MODW=6 * C, SAVE=save))
        print(f"A{A} S{S} save={int(save)}: sm100 {t:7.1f} us   triton {tt:7.1f} us", flush=True)
