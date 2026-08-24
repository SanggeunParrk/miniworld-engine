"""Per-MODULE wrap A/B, the control for the block-level result.

The block bench says custom_op is worth ~5% on training. The prediction that follows is that a
SINGLE module should show ~nothing: measured alone, its graph break lands on the module boundary,
where there is no neighbouring op to fuse with. The 5% is supposed to come from CHAINING -- the
residual adds, dropouts and reshapes BETWEEN modules that only exist when modules follow one
another. If a single module also showed 5%, the block explanation would be wrong.

Same harness as block_ab.py (torch.compile, no CUDA graph = MiniWorld's main regime).
"""
from __future__ import annotations

import argparse
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("--wrap", required=True, choices=["disable", "custom_op"])
ap.add_argument("--seq-len", type=int, default=384)
ap.add_argument("--d-pair", type=int, default=128)
ap.add_argument("--iters", type=int, default=30)
args = ap.parse_args()

from miniworld_engine import settings
settings.configure(compile_wrap=args.wrap)

import torch
from miniworld_engine.modules import (
    AdaptiveLayerNorm, AugmentedAttentionPairBias, ConditionedTransition, ImplementationType,
    PairformerBlock, PairformerConfig, TriangleAttention, TriangleMultiplication, Transition,
)

DEV, DT = "cuda", torch.bfloat16
L, D = args.seq_len, args.d_pair
OURS = ImplementationType.MINIWORLD


def _t(*shape, grad=True):
    return torch.randn(*shape, device=DEV, dtype=DT, requires_grad=grad)


def transition():
    return Transition(d_hidden=D, n=4, implementation=OURS).to(DEV, DT), (_t(1, L, L, D),)


def triangle_multiplication():
    return (TriangleMultiplication(d_pair=D, d_hidden=D, outgoing=True, implementation=OURS,
                                   p_drop=0.0).to(DEV, DT), (_t(1, L, L, D),))


def triangle_attention():
    return (TriangleAttention(d_pair=D, d_hidden=32 * 4, n_head=4, starting=True,
                              implementation=OURS).to(DEV, DT), (_t(1, L, L, D),))


def augmented_attention_token():
    ds = 384
    return (AugmentedAttentionPairBias(d_single=ds, d_cond=ds, d_pair=D, n_head=16,
                                       implementation=OURS).to(DEV, DT),
            (_t(1, 1, L, ds), _t(1, 1, L, ds), _t(1, L, L, D)))


def conditioned_transition():
    return (ConditionedTransition(d_hidden=D, d_cond=384, n=2, implementation=OURS).to(DEV, DT),
            (_t(1, L, D), _t(1, L, 384)))


def adaptive_layernorm():
    return (AdaptiveLayerNorm(d_hidden=D, d_cond=384, implementation=OURS).to(DEV, DT),
            (_t(1, L, D), _t(1, L, 384)))


def pairformer_block_1():
    cfg = PairformerConfig(d_pair=D, n_block=1, p_drop=0.0)
    return PairformerBlock(cfg, implementation=OURS).to(DEV, DT), (_t(1, L, L, D),)


TARGETS = {
    "transition": transition,
    "triangle_multiplication": triangle_multiplication,
    "triangle_attention": triangle_attention,
    "augmented_attention_token": augmented_attention_token,
    "conditioned_transition": conditioned_transition,
    "adaptive_layernorm": adaptive_layernorm,
    "pairformer_block_1": pairformer_block_1,
}


def timed(model, inputs, train: bool) -> float:
    fn = torch.compile(model)

    def step():
        out = fn(*inputs)
        out = out[0] if isinstance(out, tuple) else out
        if train:
            out.float().pow(2).mean().backward()
            for t in inputs:
                t.grad = None
            for p in model.parameters():
                p.grad = None

    if not train:
        def step(_f=fn):                                   # noqa: F811
            with torch.no_grad():
                _f(*inputs)

    for _ in range(5):
        step()
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


for name, build in TARGETS.items():
    for mode, train in (("inference", False), ("training", True)):
        try:
            model, inputs = build()
            ms = timed(model, inputs, train)
            print(f"wrap={args.wrap} target={name} mode={mode} seq_len={L} time={ms:.4f} ms",
                  flush=True)
        except Exception as e:                                     # noqa: BLE001
            print(f"wrap={args.wrap} target={name} mode={mode} FAILED "
                  f"{type(e).__name__}: {str(e)[:160]}", flush=True)
        finally:
            torch.cuda.empty_cache()
