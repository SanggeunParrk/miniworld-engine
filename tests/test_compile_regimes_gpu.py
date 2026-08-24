"""The graph the compiler actually gets, in the shapes MiniWorld trains in.

``compile_wrap="custom_op"`` is only worth being the default if the single graph it produces
SURVIVES the things real training does to a model. Three of those are not visible in a
single-module bench, and each has its own way of putting the breaks back:

* **a recycle loop** -- the main config is ``n_recycle_max: 4`` and the depth is random per step,
  so the trunk runs a varying number of times. A varying Python loop count is a guard failure,
  and Dynamo answers it with a recompile (fine, bounded) or a break (not fine).
* **DDP** -- ``torch.compile`` under DistributedDataParallel runs ``DDPOptimizer``, which SPLITS
  the graph at gradient-bucket boundaries on purpose, so the "one graph" result can quietly
  become N again on 8 GPUs.
* **the block itself** -- the invariant the rest of this change rests on.

A fourth thing they caught immediately, which no static check could: the trace has to be clean
COLD. Every earlier measurement ran an eager forward first (the same harness did a numerics parity
check), and that eager call warmed a dispatch cache whose MISS path mutates a module global --
so a pairformer block traced to 1 graph warm and 6 cold. Real training's first compiled step is
a cold trace, so the warm number was the one that did not matter. Hence: no test here warms
anything before it measures.

These are marked ``gpu``: they compile and launch real kernels. They assert on graph STRUCTURE,
not on time, because structure is what is reproducible on a shared cluster.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():                      # pragma: no cover - guarded by the marker
    pytest.skip("needs a CUDA device", allow_module_level=True)

import torch._dynamo as dynamo

from miniworld_engine import settings
from miniworld_engine.modules import (
    ImplementationType, Pairformer, PairformerBlock, PairformerConfig,
)

DEV, DT = "cuda", torch.bfloat16
L, D = 384, 128       # the real crop: max_tokens 384, d_pair 128


def _requires_custom_op():
    if settings.current().compile_wrap != "custom_op":
        pytest.skip(f"compile_wrap={settings.current().compile_wrap!r}; these assert the "
                    f"custom_op graph (run with MINIWORLD_COMPILE_WRAP=custom_op)")


def _block(n_block: int = 1):
    cfg = PairformerConfig(d_pair=D, n_block=n_block, p_drop=0.0)
    cls = PairformerBlock if n_block == 1 else Pairformer
    return cls(cfg, implementation=ImplementationType.MINIWORLD).to(DEV, DT)


def _pair(grad: bool = True):
    return torch.randn(1, L, L, D, device=DEV, dtype=DT, requires_grad=grad)


def test_pairformer_block_is_one_graph():
    """The invariant. Under ``disable`` this block is 27 graphs / 26 breaks.

    Deliberately COLD -- no eager call before the trace. A warm-up hides any break whose cause is
    a first-call side effect, which is exactly the bug this test found in the cute dispatcher.
    """
    _requires_custom_op()
    model, pair = _block(), _pair()

    def step(x):
        return model(x).float().pow(2).mean()

    dynamo.reset()
    explanation = dynamo.explain(step)(pair)
    assert explanation.graph_break_count == 0, (
        f"{explanation.graph_break_count} graph break(s):\n  "
        + "\n  ".join(str(getattr(r, "reason", r))[:200] for r in explanation.break_reasons))
    # A block whose forward is entirely disabled would ALSO report zero breaks, with an empty
    # graph. Node count is what separates "nothing broke" from "nothing was captured".
    nodes = sum(len(g.graph.nodes) for g in explanation.graphs)
    assert nodes > 50, f"only {nodes} nodes captured -- the graph is empty, not unbroken"


@pytest.mark.parametrize("n_recycle", [1, 2, 4])
def test_recycle_loop_keeps_one_graph(n_recycle):
    """A varying recycle depth must not reintroduce a break.

    This is the loop the main config runs (``n_recycle_max: 4``) and the reason it cannot capture
    a CUDA graph -- so it is exactly the case where the compiled graph has to hold up.
    """
    _requires_custom_op()
    model, pair = _block(), _pair()

    def trunk(x, n):
        for _ in range(n):
            x = model(x)
        return x.float().pow(2).mean()

    dynamo.reset()
    explanation = dynamo.explain(trunk)(pair, n_recycle)
    assert explanation.graph_break_count == 0, (
        f"recycle={n_recycle}: {explanation.graph_break_count} break(s):\n  "
        + "\n  ".join(str(getattr(r, "reason", r))[:200] for r in explanation.break_reasons))


def test_varying_recycle_depth_recompiles_but_never_breaks():
    """Depth is random per step. Dynamo may recompile per depth; it must not graph-break.

    A recompile is bounded (one per depth, then cached) and is the price of a Python-level loop.
    A break is not bounded: it costs on every step forever.
    """
    _requires_custom_op()
    model, pair = _block(), _pair()

    def trunk(x, n):
        for _ in range(n):
            x = model(x)
        return x.float().pow(2).mean()

    dynamo.reset()
    for n in (1, 2, 3, 4, 2, 1):                 # a depth schedule, repeats included
        explanation = dynamo.explain(trunk)(pair, n)
        assert explanation.graph_break_count == 0, (
            f"depth {n} broke the graph: "
            + "; ".join(str(getattr(r, "reason", r))[:160] for r in explanation.break_reasons))


def test_ddp_breaks_are_only_the_bucket_split(tmp_path):
    """Under DDP the graph splits -- prove the splits are DDP's, not ours.

    ``torch.compile`` + DistributedDataParallel runs ``DDPOptimizer``, which CUTS the graph at
    gradient-bucket boundaries on purpose so each bucket's allreduce can start as soon as its
    grads are ready. Those cuts are reported as graph breaks, so "one graph" does not survive
    verbatim to 8 GPUs and asserting ``breaks == 0`` here would be asserting the wrong thing.

    What must hold is that DDP is the ONLY thing splitting: with ``optimize_ddp`` turned off the
    same model must be back to a single unbroken graph. If it is not, a kernel is breaking under
    distribution and the bucket split is just hiding it.
    """
    _requires_custom_op()
    if torch.cuda.device_count() < 2:
        pytest.skip(f"needs 2 GPUs, have {torch.cuda.device_count()}")

    import json

    import torch.multiprocessing as mp

    out = tmp_path / "ddp.json"
    mp.spawn(_ddp_worker, args=(2, str(out)), nprocs=2, join=True)
    got = json.loads(out.read_text())

    def _fmt(entry):
        return "\n  ".join(f"{r['reason']} @ {' <- '.join(r['where'])}" for r in entry["reasons"])

    bare = got["no_ddp"]
    assert bare["breaks"] == 0, f"the block breaks even without DDP:\n  {_fmt(bare)}"
    assert bare["nodes"] > 50, f"only {bare['nodes']} nodes -- empty graph, not unbroken"

    # Whatever DDP adds must come from torch's own DDP wrapper, never from miniworld code.
    off = got["optimize_ddp_off"]
    ours = [r for r in off["reasons"]
            if any("miniworld_engine" in w for w in r["where"])]
    assert not ours, ("with DDPOptimizer disabled, a break comes from OUR code under DDP:\n  "
                      + "\n  ".join(f"{r['reason']} @ {' <- '.join(r['where'])}" for r in ours))


def _explain(step, pair) -> dict:                                   # pragma: no cover - subproc
    """graphs / breaks / nodes, and for each break BOTH its reason and where it happened.

    ``break_reasons`` can carry an empty reason string, which says nothing about whose code broke.
    The user stack does, and "whose code" is the whole question under DDP.
    """
    ex = dynamo.explain(step)(pair)
    reasons = []
    for r in ex.break_reasons:
        where = [f"{fs.filename}:{fs.lineno} {fs.name}"
                 for fs in (getattr(r, "user_stack", None) or [])[-4:]]
        reasons.append({"reason": str(getattr(r, "reason", r))[:200], "where": where})
    return {
        "graphs": ex.graph_count,
        "breaks": ex.graph_break_count,
        "nodes": sum(len(g.graph.nodes) for g in ex.graphs),
        "reasons": reasons,
    }


def _ddp_worker(rank: int, world: int, out_path: str) -> None:      # pragma: no cover - subproc
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    import json

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        result = {}
        # Baseline in the SAME process: the bare block, no DDP wrapper. Anything DDP adds has to
        # be measured against this, not against a number from a different run.
        bare_model = _block().to(rank)
        bare_pair = torch.randn(1, L, L, D, device=rank, dtype=DT, requires_grad=True)
        dynamo.reset()
        result["no_ddp"] = _explain(
            lambda x, _m=bare_model: _m(x).float().pow(2).mean(), bare_pair)
        del bare_model, bare_pair
        torch.cuda.empty_cache()
        for label, optimize in (("optimize_ddp_on", True), ("optimize_ddp_off", False)):
            model = DDP(_block().to(rank), device_ids=[rank])
            pair = torch.randn(1, L, L, D, device=rank, dtype=DT, requires_grad=True)

            def step(x, _m=model):
                return _m(x).float().pow(2).mean()

            dynamo.reset()
            prev = dynamo.config.optimize_ddp
            dynamo.config.optimize_ddp = optimize
            try:
                result[label] = _explain(step, pair)
            finally:
                dynamo.config.optimize_ddp = prev
            del model, pair
            torch.cuda.empty_cache()
        if rank == 0:
            with open(out_path, "w") as handle:
                json.dump(result, handle)
    finally:
        dist.destroy_process_group()
