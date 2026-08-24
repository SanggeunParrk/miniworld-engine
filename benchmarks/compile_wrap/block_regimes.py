"""Is plain torch.compile fast enough COMPARED TO eager + a captured CUDA graph?

That is the question MiniWorld's config forces. `n_recycle_max == 1` gets
scripts/cudagraph_trainer.py: eager, with fwd+loss+bwd captured as ONE torch.cuda.CUDAGraph
(measured 8-GPU: 71% -> ~96-100% util, ~1.8x). `n_recycle_max > 1` -- which is 4 of the 5
top-level configs -- cannot capture a graph at all (the recycle depth varies), so it runs Fabric
+ plain torch.compile. So the comparison that matters is not compile-vs-eager, it is:

    eager + manual CUDA graph   (what the fixed-recycle path gets)
    torch.compile, no graph     (what the MAIN path is stuck with)

and the gap between them is what the main config pays. compile_wrap is the second axis: a graph
break costs nothing under a captured graph, but under compile-only it splits the block into 27
graphs.
"""
from __future__ import annotations

import argparse
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("--wrap", required=True, choices=["disable", "custom_op"])
ap.add_argument("--seq-len", type=int, default=384)
ap.add_argument("--d-pair", type=int, default=128)
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--blocks", type=int, default=4)
args = ap.parse_args()

from miniworld_engine import settings
settings.configure(compile_wrap=args.wrap)          # BEFORE any kernel import

import torch
from miniworld_engine.modules import ImplementationType, PairformerConfig, Pairformer

DEV, DT = "cuda", torch.bfloat16
L, D = args.seq_len, args.d_pair


def build(impl):
    cfg = PairformerConfig(d_pair=D, n_block=args.blocks, p_drop=0.0)
    return Pairformer(cfg, implementation=impl).to(DEV, DT)


def _median(step) -> float:
    torch.cuda.synchronize()
    times = []
    for _ in range(args.iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        step()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def timed(model, train: bool, compile_it: bool, cudagraph: bool) -> float:
    pair = torch.randn(1, L, L, D, device=DEV, dtype=DT, requires_grad=train)
    fn = torch.compile(model) if compile_it else model

    def step():
        if train:
            out = fn(pair)
            out.float().pow(2).mean().backward()
        else:
            with torch.no_grad():
                fn(pair)

    def zero():
        pair.grad = None
        for p in model.parameters():
            p.grad = None

    for _ in range(5):                      # warm + compile
        step()
        zero()

    if not cudagraph:
        return _median(lambda: (step(), zero()))

    # Manual capture of fwd+loss+bwd, exactly what scripts/cudagraph_trainer.py does: one
    # standard torch.cuda.CUDAGraph, replayed. Grads accumulate INTO the captured buffers, so
    # they are not zeroed between replays -- that matches the trainer, which zeroes outside the
    # graph, and keeps the replay measuring the same work every iteration.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    zero()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    return _median(graph.replay)


for impl_name, impl in (("pytorch", ImplementationType.PYTORCH),
                        ("miniworld", ImplementationType.MINIWORLD)):
    for mode, train in (("inference", False), ("training", True)):
        # The two regimes MiniWorld actually runs, plus the two "what if" corners.
        for compile_it, cudagraph in ((False, True), (True, False), (False, False), (True, True)):
            label = (f"wrap={args.wrap} impl={impl_name} mode={mode} "
                     f"compile={str(compile_it).lower()} cudagraph={str(cudagraph).lower()}")
            try:
                model = build(impl)
                ms = timed(model, train, compile_it, cudagraph)
                print(f"{label} blocks={args.blocks} seq_len={L} time={ms:.4f} ms", flush=True)
            except Exception as e:                                  # noqa: BLE001
                print(f"{label} FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
            finally:
                torch.cuda.empty_cache()
