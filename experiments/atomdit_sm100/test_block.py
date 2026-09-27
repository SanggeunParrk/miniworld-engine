"""The sm_100a atom DiT block against the torch DiTBlock in fp64 (output and every gradient), then its inference / training latency."""
import argparse, copy, torch
from common import make_block, make_inputs, rel, graph_time, event_time
from atom_block import AtomBlock

p = argparse.ArgumentParser(); p.add_argument("--lengths", type=int, nargs="+", default=[384]); p.add_argument("--check-A", type=int, default=4)
p.add_argument("--check-L", type=int, default=128); p.add_argument("--notime", action="store_true"); p.add_argument("--nocompile", action="store_true")
a = p.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False

# ---- correctness at a small size: bf16 block (ours) vs fp64 torch block, same parameters
blk = make_block("pytorch")
ref = copy.deepcopy(blk).double()
ours = AtomBlock(blk, compile_rest=not a.nocompile)
s, c, z = make_inputs(a.check_A, a.check_L, grad=True)
sd, cd, zd = (t.detach().double().requires_grad_() for t in (s, c, z))
dy = torch.randn_like(s)
y = ours(s, c, z); y.backward(dy)
yr = ref(sd, cd, zd); yr.backward(dy.double())
print(f"check A{a.check_A} N{8 * a.check_L}: y {rel(y, yr):.2e}  ds {rel(s.grad, sd.grad):.2e}  dcond {rel(c.grad, cd.grad):.2e}  dpair {rel(z.grad, zd.grad):.2e}", flush=True)
worst = max(((rel(p1.grad, p2.grad), n) for (n, p1), (_, p2) in zip(blk.named_parameters(), ref.named_parameters())), key=lambda t: t[0])
print(f"  worst parameter gradient: {worst[1]} {worst[0]:.2e}", flush=True)
del ref, sd, cd, zd

if not a.notime:
    for L in a.lengths:
        for mode in ("inference", "training"):
            A = 5 if mode == "inference" else 48
            if mode == "inference":
                s, c, z = make_inputs(A, L)
                def step():
                    with torch.no_grad():
                        return ours(s, c, z)
                t = graph_time(step)
            else:
                s, c, z = make_inputs(A, L, grad=True)
                dy = torch.randn_like(s)
                def step():
                    ours(s, c, z).backward(dy)
                t = event_time(step)
            print(f"L{L} (N={8 * L}) sm100 {mode:9s} {t:10.1f} us", flush=True)
            torch.cuda.empty_cache()
