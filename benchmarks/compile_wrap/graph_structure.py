"""Does compile_wrap="custom_op" (a) give the same numbers and (b) actually remove EVERY break?

Run once per mode, in a FRESH process: kernels/_compile reads settings.compile_wrap at IMPORT
time, so the two modes can never coexist in one interpreter.

The goal this measures is not "fewer breaks" -- it is ZERO, for every production module. A single
break anywhere in a pairformer block splits the whole block's graph, so the surrounding glue stops
fusing and the compiled baseline keeps the advantage the measurements showed at L=384. Break
REASONS are printed in full: they are the worklist.
"""
from __future__ import annotations

import argparse
import traceback

ap = argparse.ArgumentParser()
ap.add_argument("--wrap", required=True, choices=["disable", "custom_op"])
ap.add_argument("--out", required=True)
ap.add_argument("--seq-len", type=int, default=384)
ap.add_argument("--d-pair", type=int, default=128)
ap.add_argument("--only", default="", help="comma list of targets (default: all)")
args = ap.parse_args()

from miniworld_engine import settings

settings.configure(compile_wrap=args.wrap)            # BEFORE any kernel import

import torch
import torch._dynamo as dynamo

from miniworld_engine.modules import (
    AdaptiveLayerNorm,
    AttentionPairBias,
    AugmentedAttentionPairBias,
    ConditionedTransition,
    ImplementationType,
    MSAPairWeightedAveraging,
    OuterProductMean,
    PairformerBlock,
    PairformerConfig,
    Transition,
    TriangleAttention,
    TriangleMultiplication,
)
from miniworld_engine.modules.triangle_multiplication import (
    BidirectionalTriangleMultiplication,
)

DEV, DT = "cuda", torch.bfloat16
L, D = args.seq_len, args.d_pair
OURS = ImplementationType.MINIWORLD
torch.manual_seed(0)


def _t(*shape, grad=True):
    return torch.randn(*shape, device=DEV, dtype=DT, requires_grad=grad)


def transition():
    return Transition(d_hidden=D, n=4, implementation=OURS).to(DEV, DT), (_t(1, L, L, D),)


def triangle_multiplication():
    m = TriangleMultiplication(d_pair=D, d_hidden=D, outgoing=True, implementation=OURS,
                               p_drop=0.0).to(DEV, DT)
    return m, (_t(1, L, L, D),)


def triangle_multiplication_bidir():
    m = BidirectionalTriangleMultiplication(d_pair=D, d_hidden=D, implementation=OURS,
                                            p_drop=0.0).to(DEV, DT)
    return m, (_t(1, L, L, D),)


def triangle_attention():
    m = TriangleAttention(d_pair=D, d_hidden=32 * 4, n_head=4, starting=True,
                          implementation=OURS).to(DEV, DT)
    return m, (_t(1, L, L, D),)


def augmented_attention_token():
    # AF3's token shape: d_single 384 over 16 heads -> head dim 24 (the kernel needs 16..64).
    ds = 384
    m = AugmentedAttentionPairBias(d_single=ds, d_cond=ds, d_pair=D, n_head=16,
                                   implementation=OURS).to(DEV, DT)
    return m, (_t(1, 1, L, ds), _t(1, 1, L, ds), _t(1, L, L, D))


def attention_pair_bias():
    # No `implementation`: this module dispatches its kernels internally.
    m = AttentionPairBias(d_single=384, d_pair=D, n_head=16).to(DEV, DT)
    return m, (_t(1, L, 384), _t(1, L, L, D))


def conditioned_transition():
    m = ConditionedTransition(d_hidden=D, d_cond=384, n=2, implementation=OURS).to(DEV, DT)
    return m, (_t(1, L, D), _t(1, L, 384))


def adaptive_layernorm():
    m = AdaptiveLayerNorm(d_hidden=D, d_cond=384, implementation=OURS).to(DEV, DT)
    return m, (_t(1, L, D), _t(1, L, 384))


def outer_product_mean():
    m = OuterProductMean(d_msa=D, d_pair=D, d_hidden=32, implementation=OURS).to(DEV, DT)
    return m, (_t(1, 8, L, D),)


def msa_pair_weighted_averaging():
    m = MSAPairWeightedAveraging(d_msa=D, d_pair=D, d_hidden=32, n_head=8,
                                 implementation=OURS).to(DEV, DT)
    return m, (_t(1, 8, L, D), _t(1, L, L, D))


def pairformer_block():
    cfg = PairformerConfig(d_pair=D, n_block=1, p_drop=0.0)
    return PairformerBlock(cfg, implementation=OURS).to(DEV, DT), (_t(1, L, L, D),)


TARGETS = {
    "transition": transition,
    "triangle_multiplication": triangle_multiplication,
    "triangle_multiplication_bidir": triangle_multiplication_bidir,
    "triangle_attention": triangle_attention,
    "augmented_attention_token": augmented_attention_token,
    "attention_pair_bias": attention_pair_bias,
    "conditioned_transition": conditioned_transition,
    "adaptive_layernorm": adaptive_layernorm,
    "outer_product_mean": outer_product_mean,
    "msa_pair_weighted_averaging": msa_pair_weighted_averaging,
    "pairformer_block": pairformer_block,
}
if args.only:
    TARGETS = {k: v for k, v in TARGETS.items() if k in args.only.split(",")}

out = {"wrap": args.wrap}
for name, build in TARGETS.items():
    print(f"\n########## {args.wrap} :: {name}", flush=True)
    try:
        mod, inputs = build()
    except Exception as e:
        print(f"  BUILD FAILED {type(e).__name__}: {e}", flush=True)
        continue

    def step(*xs, _m=mod):
        y = _m(*xs)
        y = y[0] if isinstance(y, tuple) else y
        return y.float().pow(2).mean()

    # --- graph structure FIRST, on a COLD process ------------------------------------------- #
    # Order matters and used to be wrong: the parity call below warms caches whose miss path
    # graph-breaks (trimul_inproj/cute/dispatch.py::pick did exactly that), so explaining after
    # it measured a graph real training never gets -- its first compiled step is a cold trace.
    dynamo.reset()
    try:
        ex = dynamo.explain(step)(*inputs)
        nodes = sum(len(g.graph.nodes) for g in ex.graphs)
        out[f"{name}/breaks"] = ex.graph_break_count
        out[f"{name}/graphs"] = ex.graph_count
        out[f"{name}/nodes"] = nodes
        print(f"  COLD graphs={ex.graph_count} breaks={ex.graph_break_count} nodes={nodes}",
              flush=True)
        for r in ex.break_reasons:
            txt = " | ".join(x.strip() for x in str(getattr(r, "reason", r)).split("\n")
                             if x.strip())
            print(f"    BREAK {txt[:260]}", flush=True)
            for fs in (getattr(r, "user_stack", None) or [])[-3:]:
                print(f"          at {fs.filename.split('miniworld_engine/')[-1]}:{fs.lineno}"
                      f" {fs.name}", flush=True)
    except Exception as e:
        out[f"{name}/breaks"] = -1
        print(f"  EXPLAIN FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)

    # --- eager reference. Identical in both modes: this is the parity check. ---------------- #
    try:
        for t in inputs:
            t.grad = None
        loss = step(*inputs)
        loss.backward()
        out[f"{name}/loss"] = loss.detach().cpu()
        for i, t in enumerate(inputs):
            if t.grad is not None:
                out[f"{name}/grad{i}"] = t.grad.detach().float().cpu()
    except Exception as e:
        print(f"  EAGER FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
        traceback.print_exc()
        continue


torch.save(out, args.out)
print(f"\n[{args.wrap}] wrote {args.out}", flush=True)
