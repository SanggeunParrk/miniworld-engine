"""attn_fwd.cu against the H100 fused block's Triton _attn_fwd (O, LSE), incl. a padded sequence, and timing."""
import argparse, torch, triton
from common import rel, graph_time, C, H, D, HW
import h100_fused as SF
from ops import AttnFwd

p = argparse.ArgumentParser(); p.add_argument("--cases", nargs="+", default=["48:3072", "5:3072", "48:6144", "5:6144"])
a = p.parse_args()
for case in a.cases:
    N, S = map(int, case.split(":"))
    g = torch.Generator(device="cuda").manual_seed(0)
    Qh, Kh, Vh = (torch.randn(N, H, S, D, device="cuda", generator=g).to(torch.bfloat16) for _ in range(3))
    for pad in (False, True):
        su = torch.full((N,), S, device="cuda", dtype=torch.int32)
        if pad:
            su[0] = S - 200; su[-1] = S // 3 + 7
        run, O, LSE = AttnFwd().bind(Qh, Kh, Vh, su)
        run(); torch.cuda.synchronize()
        rO = torch.empty(N * S, C, device="cuda", dtype=torch.bfloat16); rL = torch.empty(N, H, S, device="cuda")
        tri = lambda: SF._attn_fwd[lambda m: (triton.cdiv(S, m["BM"]), N * H)](Qh, Kh, Vh, su, rO, rL, S, D ** -0.5, SF._bucket(N * S), C=C, H=H, D=D, HW=HW)
        tri(); torch.cuda.synchronize()
        # fp64 truth on two samples (0 and the last)
        err = []
        for Ot in (O, rO):
            e = []
            for nn in (0, N - 1):
                q64, k64, v64 = (t[nn].double() for t in (Qh, Kh, Vh))
                sc = q64 @ k64.transpose(-1, -2) * D ** -0.5
                i = torch.arange(S, device="cuda")
                ok = ((i[:, None] - i[None, :]).abs() <= HW) & (i[None, :] < su[nn])
                pr = torch.softmax(sc.masked_fill(~ok, float("-inf")), -1).nan_to_num(0.0)
                o64 = (pr @ v64).transpose(0, 1).reshape(S, C) * (i < su[nn])[:, None]
                e.append(rel(Ot.view(N, S, C)[nn], o64))
            err.append(max(e))
        print(f"N{N} S{S} pad={int(pad)}: vs fp64 sm100 {err[0]:.2e} triton {err[1]:.2e};  O rel {rel(O, rO):.2e} ({(O.float() != rO.float()).float().mean().item() * 100:.2f}% differ)  "
              f"LSE max|d| {(LSE - rL).abs().max().item():.2e}  finite {bool(torch.isfinite(O.float()).all())}", flush=True)
        if not pad:
            print(f"N{N} S{S}: sm100 {graph_time(run):7.1f} us   triton {graph_time(tri):7.1f} us", flush=True)
