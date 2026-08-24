"""Does torch.compile(mode="reduce-overhead") work now that there are no graph breaks?

MiniWorld's scripts/cudagraph_trainer.py says it does not, and says WHY:

    the trunk's custom `@torch.compiler.disable()` cute/quack kernels are captured fine by the
    standard CUDA-graph API but make inductor's cudagraph-trees (reduce-overhead) silently
    skip / crash.

Inductor's cudagraph-trees bail on a graph break, and until now every kernel entry was one. With
compile_wrap="custom_op" a pairformer block is a single graph, so the stated blocker is gone --
which would give fusion AND launch-overhead removal together, instead of choosing.

"Silently skip" is the trap: reduce-overhead that quietly declines to capture still RUNS and
still returns right answers, just with no graph. So this reads
`torch._dynamo.utils.counters["inductor"]` for the skip counters rather than trusting the timing.
"""
from __future__ import annotations

import argparse
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("--wrap", required=True, choices=["disable", "custom_op"])
ap.add_argument("--seq-len", type=int, default=384)
ap.add_argument("--d-pair", type=int, default=128)
ap.add_argument("--blocks", type=int, default=4)
ap.add_argument("--iters", type=int, default=30)
args = ap.parse_args()

from miniworld_engine import settings
settings.configure(compile_wrap=args.wrap)

import torch
import torch._dynamo as dynamo
from torch._dynamo.utils import counters
from miniworld_engine.modules import ImplementationType, Pairformer, PairformerConfig

DEV, DT = "cuda", torch.bfloat16
L, D = args.seq_len, args.d_pair


def build():
    cfg = PairformerConfig(d_pair=D, n_block=args.blocks, p_drop=0.0)
    return Pairformer(cfg, implementation=ImplementationType.MINIWORLD).to(DEV, DT)


def skip_report() -> str:
    """What inductor says about cudagraphs -- the difference between 'captured' and 'declined'."""
    ind = counters.get("inductor", {})
    keys = {k: v for k, v in ind.items() if "cudagraph" in k}
    return ", ".join(f"{k}={v}" for k, v in sorted(keys.items())) or "no cudagraph counters"


def run(label: str, mode: str | None, manual_graph: bool, train: bool) -> None:
    """Time one configuration and report it with inductor's cudagraph counters."""
    dynamo.reset()
    counters.clear()
    torch.cuda.empty_cache()
    try:
        ms = _measure(mode, manual_graph, train)
        print(f"wrap={args.wrap} mode={label} train={str(train).lower()} "
              f"time={ms:.4f} ms  [{skip_report()}]", flush=True)
    except Exception as e:                                            # noqa: BLE001
        print(f"wrap={args.wrap} mode={label} train={str(train).lower()} "
              f"FAILED {type(e).__name__}: {str(e)[:220]}  [{skip_report()}]", flush=True)
    finally:
        # The model and its activations are locals of _measure, so they are already gone by here
        # and empty_cache can actually hand the memory back before the next configuration builds.
        torch.cuda.empty_cache()


def _measure(mode: str | None, manual_graph: bool, train: bool) -> float:
    model = build()
    pair = torch.randn(1, L, L, D, device=DEV, dtype=DT, requires_grad=train)
    fn = model if mode == "eager" else torch.compile(model, mode=mode)

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

    for _ in range(6):                        # warm + compile + cudagraph-tree record
        step()
        zero()

    if manual_graph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        zero()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
        body = g.replay
    else:
        def body():
            step()
            zero()

    torch.cuda.synchronize()
    times = []
    for _ in range(args.iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        body()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


for train_mode in (False, True):
    run("eager",               "eager",           False, train_mode)
    run("eager+manualgraph",   "eager",           True,  train_mode)
    run("compile",             None,              False, train_mode)
    run("compile+manualgraph", None,              True,  train_mode)
    run("reduce-overhead",     "reduce-overhead", False, train_mode)
    run("max-autotune",        "max-autotune",    False, train_mode)
