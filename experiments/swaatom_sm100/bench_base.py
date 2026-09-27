"""Baselines for one SWA atom block on B200: torch eager / torch.compile of the block (bf16, dense banded SDPA), and the H100 fused
algorithm (team-gm 14f2c73 swa_fused_triton, its Triton path: the sm_90a CUDA kernels do not build on sm_100). Also checks the H100
fused block against the fp64 reference. Inference: A = 5, CUDA graph. Training: A = 48, fwd + bwd (q, mod and weight grads), no graph."""
import argparse, torch
from common import make, hoist_mod, block_ref, rel, graph_time, event_time, HW
import h100_fused as SF

p = argparse.ArgumentParser(); p.add_argument("--lengths", type=int, nargs="+", default=[384, 768])
p.add_argument("--impls", nargs="+", default=["eager", "compile", "h100fused"]); p.add_argument("--modes", nargs="+", default=["inference", "training"])
a = p.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
W = ("wqkv", "wg", "wo", "wu", "wd")

# ---- the H100 fused block against fp64 (small)
q, cb, cos, sin, su, w = make(4, 1024)
mod = hoist_mod(cb, w["wmod"])
out = SF.swa_block(q, mod, cos, sin, su, *(w[k] for k in W), 1, HW)
ref = block_ref(q, mod, cos, sin, su, *(w[k] for k in W))
print(f"h100 fused vs fp64 (A4, S1024): out rel {rel(out, ref):.2e}", flush=True)

eager_blk = lambda q, mod, cos, sin, su, ws: block_ref(q, mod, cos, sin, su, *ws, dtype=torch.bfloat16)
comp_blk = torch.compile(eager_blk)
for L in a.lengths:
    S = 8 * L
    for impl in a.impls:
        for mode in a.modes:
            A = 5 if mode == "inference" else 48
            try:
                q, cb, cos, sin, su, w = make(A, S)
                train = mode == "training"
                if train:
                    q.requires_grad_(); cb.requires_grad_()
                    for k in w: w[k].requires_grad_()
                ws = [w[k] for k in W]
                def fwd():
                    mod = hoist_mod(cb, w["wmod"])
                    if impl == "h100fused":
                        return SF.swa_block(q, mod, cos, sin, su, *ws, 1, HW)
                    return (comp_blk if impl == "compile" else eager_blk)(q, mod, cos, sin, su, ws)
                if train:
                    dy = torch.randn_like(q)
                    t = event_time(lambda: fwd().backward(dy))
                else:
                    def step():
                        with torch.no_grad():
                            return fwd()
                    t = graph_time(step)
                print(f"L{L} (S={S}) {impl:10s} {mode:9s} {t:10.1f} us", flush=True)
            except Exception as e:                       # noqa: BLE001
                print(f"L{L} (S={S}) {impl:10s} {mode:9s} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
            torch.cuda.empty_cache()
