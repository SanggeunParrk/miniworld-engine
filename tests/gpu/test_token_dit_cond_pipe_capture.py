"""The H100 bf16 token DiT's pipelined conditioning GEMMs (``kernels/conditioned_transition/cuda._cond_gemm``) under every kind of capture.

The side-stream GEMM of a block reads the block's folded weight ``[W1; W2]`` and the conditioning; a graph replays its kernels in the
order of the captured dependencies only, so each GEMM must depend on whatever made its operands:

- a pack made in the call (the first call after the weights' version changed; every call of a capture outside ``static_weights()``, whose
  packs are the capture's own) is made on the caller's stream AFTER the previous block's start, which is where the GEMM forks;
- a capture outside ``static_weights()`` repacks at every replay, so a grouped chunk weight made eagerly holds weights the replay no
  longer reads.

Checked against the same stack with the pipeline off (one GEMM per block on the caller's stream), bitwise: eager, and a captured graph
replayed with the conditioning and the input refilled in place -- outside ``static_weights()``, with it, with ``static_inputs()`` alone
and with both; packs made eagerly before the capture and inside it; 2 / 12 / 24 blocks; shared and per-sample conditioning.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.h100]

MODELS = {"protenix": (384, 128), "boltz2": (768, 128), "esmfold2": (768, 256)}
_STACKS: dict = {}


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _stack(model: str):
    """24 blocks of ``model``'s token DiT (built once per model and test session)."""
    from miniworld_engine.modules import TokenDiT

    if model not in _STACKS:
        _STACKS.clear()  # one model's stack at a time
        dc, dp = MODELS[model]
        torch.manual_seed(0)
        blocks = []
        for _ in range(24):
            b = TokenDiT(model, 768, dc, dp, 16, 2)
            for p in b.parameters():
                torch.nn.init.normal_(p, std=0.03) if p.dim() > 1 else torch.nn.init.normal_(p, mean=0.3, std=0.2)
            blocks.append(b.float().cuda().eval().requires_grad_(False).set_backend("cuda"))
        _STACKS[model] = blocks
    return _STACKS[model]


@contextlib.contextmanager
def _mode(mode: str):
    from miniworld_engine.capture import static_inputs, static_weights

    with contextlib.ExitStack() as stack:
        if mode in ("weights", "weights+inputs"):
            stack.enter_context(static_weights())
        handle = stack.enter_context(static_inputs()) if mode in ("inputs", "weights+inputs") else None
        yield handle


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) path")
@pytest.mark.parametrize("L", [128, 384])
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("nb", [2, 12, 24])
@pytest.mark.parametrize("mode", ["none", "weights", "inputs", "weights+inputs"])
@pytest.mark.parametrize("model", list(MODELS))
def test_capture_replay_equals_eager(model, mode, nb, shared, L):
    from miniworld_engine.capture import clear_weight_caches
    from miniworld_engine.kernels.conditioned_transition import cuda as C

    dc, dp = MODELS[model]
    blocks = _stack(model)[:nb]
    params = [p for b in blocks for p in b.parameters()]
    S = 5 if shared else 2  # [S L] conditioning rows per sample: within the pipeline's row gate at L 384
    assert (L if shared else S * L) <= C._COND_PIPE_ROWS
    g = torch.Generator(device="cuda").manual_seed(1)
    x_buf = torch.empty(S, 1, L, 768, device="cuda", dtype=torch.bfloat16)
    c_buf = torch.empty(1 if shared else S, 1, L, dc, device="cuda", dtype=torch.bfloat16)
    pair = torch.randn(1, L, L, dp, device="cuda", generator=g).bfloat16()
    mask = torch.ones(1, L, device="cuda", dtype=torch.bool)
    mask[0, -7:] = False

    def fill():
        x_buf.copy_(torch.randn(x_buf.shape, device="cuda", generator=g))
        c_buf.copy_(torch.randn(c_buf.shape, device="cuda", generator=g) * 2 + 0.7)

    def stack(passes=1, n=nb):
        h = x_buf
        cnd = c_buf.expand(S, 1, L, dc) if shared else c_buf
        for _ in range(passes):  # a captured rollout: the same conditioning object, the stack again
            for b in blocks[:n]:
                h = b(h, cnd, pair, mask)
        return h

    def reference(*how):
        rows, C._COND_PIPE_ROWS = C._COND_PIPE_ROWS, 0
        try:
            out = stack(*how).clone()
        finally:
            C._COND_PIPE_ROWS = rows
        torch.cuda.synchronize()
        assert bool(torch.isfinite(out.float()).all())
        return out

    def same(got, what, *how):
        torch.cuda.synchronize()
        ref = reference(*how)
        assert torch.equal(got, ref), (what, float((got.float() - ref.float()).abs().max()))

    def replays(graph, gout, what, *how):
        for i in range(3):
            fill()
            graph.replay()
            same(gout, f"{what}: replay {i}", *how)

    def capture(*how):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gout = stack(*how)
        return graph, gout

    graphs = []
    try:
        with torch.no_grad(), _mode(mode) as handle:
            fill()
            same(stack(), "eager, first call")
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    stack()
            torch.cuda.current_stream().wait_stream(side)
            same(stack(), "eager, warm")
            # packs and group made by the warm-up
            graphs.append(capture())
            replays(*graphs[-1], "capture after warm-up")
            # a pass shorter than the learned group: the look-ahead GEMM of its last block is read by no block of the capture
            graphs.append(capture(1, nb // 2))
            replays(*graphs[-1], "capture of half the stack", 1, nb // 2)
            same(stack(), "eager, after the shorter pass")
            # two passes in one capture (outside static_weights() the second reads chunks made of the capture's own packs)
            graphs.append(capture(2))
            replays(*graphs[-1], "capture of two passes", 2)
            # every pack made inside the capture (new versions, no eager call before it)
            torch.autograd.graph.increment_version(params)
            graphs.append(capture())
            replays(*graphs[-1], "capture with new packs")
            torch.autograd.graph.increment_version(params)
            graphs.append(capture(2))
            replays(*graphs[-1], "capture of two passes with new packs", 2)
            # the first eager call after the weights' version changed makes its packs in the call
            for i in range(3):
                torch.autograd.graph.increment_version(params)
                fill()
                same(stack(), f"eager with new packs {i}")
            if mode == "none" and model == "protenix":
                # outside static_weights() a replay reads the weights as they are then (the plain block's packs are the capture's own)
                graphs.append(capture(2))
                for i in range(2):
                    for p in params:
                        p.mul_(1.0 + 0.01 * (i + 1))
                    fill()
                    graphs[-1][0].replay()
                    same(graphs[-1][1], f"replay after a weight update {i}", 2)
                for p in params:
                    p.div_(1.01 * 1.02)
            del handle
    finally:
        graphs.clear()
        clear_weight_caches()
