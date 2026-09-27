"""ffn_fwd.cu against the H100 fused block's Triton _oproj_ffn_fwd (same rounding points), and timing."""
import argparse, torch, triton
from common import make, hoist_mod, rel, graph_time, C, NHID, EPS
import h100_fused as SF
from ops import FfnFwd

p = argparse.ArgumentParser(); p.add_argument("--cases", nargs="+", default=["48:3072", "5:3072", "48:6144", "5:6144"])
a = p.parse_args()
for case in a.cases:
    A, S = map(int, case.split(":"))
    q, cb, cos, sin, su, w = make(A, S)
    mod = hoist_mod(cb, w["wmod"]) * 0.5
    M = A * S
    gen = torch.Generator(device="cuda").manual_seed(1)
    g = torch.randn(M, C, device="cuda", generator=gen).to(torch.bfloat16)
    o = torch.randn(M, C, device="cuda", generator=gen).to(torch.bfloat16)
    qf = q.reshape(M, C)
    for save in (False, True):
        run, (out, Q1, Att, Y, Ff) = FfnFwd().bind(qf, g, o, mod, w["wo"], w["wu"], w["wd"], A, 1, save=save)
        run(); torch.cuda.synchronize()
        rout, rq1, ratt, ry, rff = (torch.empty(M, C, device="cuda", dtype=torch.bfloat16) for _ in range(5))
        r2 = torch.empty(M, device="cuda")
        launch = lambda sv: SF._oproj_ffn_fwd[lambda m: (triton.cdiv(M, m["BR"]),)](qf, o, g, mod, w["wo"], w["wu"], w["wd"], rq1, rout, r2, ratt, ry, ry, rff,
                                                                                    M, S, 1, EPS, SF._bucket(M), C=C, NHID=NHID, MODW=6 * C, SAVE=sv)
        launch(True); torch.cuda.synchronize()
        diff = lambda x, y: f"{rel(x, y):.1e}/{(x.float() != y.float()).float().mean().item() * 100:.2f}%"
        msg = f"A{A} S{S} save={int(save)}: out {diff(out, rout)}"
        if save:
            msg += f"  q1 {diff(Q1, rq1)}  att {diff(Att, ratt)}  y {diff(Y, ry)}  ffn {diff(Ff, rff)}"
        print(msg + "   (rel / % differing)", flush=True)
        print(f"A{A} S{S} save={int(save)}: sm100 {graph_time(run):7.1f} us   triton {graph_time(lambda: launch(save)):7.1f} us", flush=True)
