"""Baselines for the atom DiT block on B200: PyTorch eager, torch.compile, the engine (miniworld: Triton attention etc.).
Inference: A = 5, no grad, CUDA-graph replay. Training: A = 48, forward + backward (input and parameter grads), no graph."""
import argparse, torch
from common import make_block, make_inputs, graph_time, event_time

p = argparse.ArgumentParser(); p.add_argument("--lengths", type=int, nargs="+", default=[384, 768])
p.add_argument("--impls", nargs="+", default=["eager", "compile", "miniworld"]); p.add_argument("--modes", nargs="+", default=["inference", "training"])
a = p.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
for L in a.lengths:
    for impl in a.impls:
        blk = make_block("pytorch" if impl in ("eager", "compile") else "miniworld")
        fwd = torch.compile(blk) if impl == "compile" else blk
        for mode in a.modes:
            A = 5 if mode == "inference" else 48
            try:
                if mode == "inference":
                    blk.eval()
                    s, c, z = make_inputs(A, L)
                    def step():
                        with torch.no_grad():
                            return fwd(s, c, z)
                    t = graph_time(step)
                else:
                    blk.train()
                    s, c, z = make_inputs(A, L, grad=True)
                    dy = torch.randn_like(s)
                    def step():
                        y = fwd(s, c, z)
                        y.backward(dy)
                    t = event_time(step)
                print(f"L{L} (N={8 * L}) {impl:9s} {mode:9s} {t:10.1f} us", flush=True)
            except Exception as e:                       # noqa: BLE001
                print(f"L{L} (N={8 * L}) {impl:9s} {mode:9s} failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
            torch.cuda.empty_cache()
