"""pair_bias.py against fp64 (forward and all three gradients) and its timing vs the engine's LayerNorm + Linear."""
import sys, torch
from common import rel, graph_time
from pair_bias import reference
import pair_bias as pb
CU = "--cu" in sys.argv
pair_bias_fwd = pb.pair_bias_fwd_cu if CU else pb.pair_bias_fwd
pair_bias_bwd = pb.pair_bias_bwd_cu if CU else pb.pair_bias_bwd

for L in [int(x) for x in ([a for a in sys.argv[1:] if not a.startswith("--")] or ["384", "768"])]:
    N = 8 * L
    g = torch.Generator(device="cuda").manual_seed(0)
    z = torch.randn(N, N, 16, device="cuda", generator=g).to(torch.bfloat16)
    gamma = 1 + 0.1 * torch.randn(16, device="cuda", generator=g)
    wb = torch.randn(4, 16, device="cuda", generator=g) / 4
    bo, bt = pair_bias_fwd(z, gamma, wb)
    ref = reference(z, gamma, wb)
    print(f"N{N}: bias rel {rel(bo, ref):.2e}  bias^T rel {rel(bt, ref.transpose(1, 2)):.2e}", flush=True)
    dbias = torch.randn(4, N, N, device="cuda", generator=g)
    dz, dg, dw = pair_bias_bwd(z, gamma, wb, dbias)
    if N <= 3072:
        zd, gd, wd = (t.double().requires_grad_() for t in (z, gamma, wb))
        (torch.nn.functional.layer_norm(zd, (16,), gd, None, 1e-5) @ wd.t()).permute(2, 0, 1).backward(dbias.double())
        print(f"N{N}: dz rel {rel(dz, zd.grad):.2e}  dgamma rel {rel(dg, gd.grad):.2e}  dWb rel {rel(dw, wd.grad):.2e}", flush=True)
        del zd, gd, wd
    t_f = graph_time(lambda: pair_bias_fwd(z, gamma, wb))
    t_b = graph_time(lambda: pair_bias_bwd(z, gamma, wb, dbias))
    ln = torch.nn.LayerNorm(16, bias=False, device="cuda", dtype=torch.bfloat16)
    lin = torch.nn.Linear(16, 4, bias=False, device="cuda", dtype=torch.bfloat16)
    t_t = graph_time(lambda: lin(ln(z)).permute(2, 0, 1).contiguous())
    zb = z.numel() * 2 / 1e6
    print(f"N{N}: fwd {t_f:7.1f} us (z {zb:.0f} MB read, bias x2 {2 * 4 * N * N * 2 / 1e6:.0f} MB written)   bwd {t_b:7.1f} us   "
          f"torch LN+Linear+permute {t_t:7.1f} us", flush=True)
