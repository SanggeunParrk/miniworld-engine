"""ffn_bwd.cu against the H100 fused block's Triton _ffn_bwd (DWOPS=True), and timing."""
import argparse, torch, triton
from common import make, hoist_mod, rel, graph_time, C, NHID, EPS
import h100_fused as SF
from ops import FfnBwd

p = argparse.ArgumentParser(); p.add_argument("--cases", nargs="+", default=["48:3072", "5:3072"])
a = p.parse_args()
for case in a.cases:
    A, S = map(int, case.split(":"))
    q, cb, cos, sin, su, w = make(A, S)
    mod = hoist_mod(cb, w["wmod"]) * 0.5
    M = A * S
    gen = torch.Generator(device="cuda").manual_seed(2)
    rnd = lambda: torch.randn(M, C, device="cuda", generator=gen).to(torch.bfloat16)
    dq2, q1, y, ffn = rnd(), rnd(), rnd(), rnd()
    run, (dq1, dffn, hh, dab, dmod) = FfnBwd().bind(dq2, q1, y, ffn, mod, w["wu"], w["wd"], A, 1)
    run(); torch.cuda.synchronize()
    rdq1, rdffn = (torch.empty(M, C, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    rhh = torch.empty(M, NHID, device="cuda", dtype=torch.bfloat16); rdab = torch.empty(M, 2 * NHID, device="cuda", dtype=torch.bfloat16)
    rdmod = torch.zeros(S, 6 * C, device="cuda")
    grid_t = lambda m: (triton.cdiv(S, m["AT"]), triton.cdiv(A, m["SP"]), 1)
    def tri():
        rdmod.zero_()
        SF._ffn_bwd[grid_t](dq2, q1, mod, w["wu"], w["wd"], y, ffn, rdq1, rdab, rhh, rdffn, rdmod, S, A, 1, EPS, SF._bucket(M),
                            C=C, NHID=NHID, MODW=6 * C, DWOPS=True)
    tri(); torch.cuda.synchronize()
    diff = lambda x, y_: f"{rel(x, y_):.1e}/{(x.float() != y_.float()).float().mean().item() * 100:.2f}%"
    print(f"A{A} S{S}: dq1 {diff(dq1, rdq1)}  dffn {diff(dffn, rdffn)}  h {diff(hh, rhh)}  dab {diff(dab, rdab)}  "
          f"dmod[shift] {rel(dmod[:, 384:512], rdmod[:, 384:512]):.1e} [scale] {rel(dmod[:, 512:640], rdmod[:, 512:640]):.1e} "
          f"[gate] {rel(dmod[:, 640:], rdmod[:, 640:]):.1e}", flush=True)
    print(f"A{A} S{S}: sm100 {graph_time(run):7.1f} us   triton {graph_time(tri):7.1f} us", flush=True)
