"""fp32 master parameters over bf16 activations at inference on B200: the kernel-dtype forms of the parameters (casts, folds) are
cached (``kernels._capture.param_forms``) so a call does not re-cast every weight, and the cache never serves a stale form:
an in-place update of a parameter (its ``_version`` moves) is seen by the next call, eagerly and in a graph captured without
``static_weights``."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
                                reason="B200 (sm_100a) paths")
BF = torch.bfloat16


def _randomize(m, seed=3):
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim >= 2:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
            elif "weight" in n:
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
    return m


def _r(*shape):
    return (torch.randn(*shape, device="cuda") * 0.5).to(BF)


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def _cases():
    from miniworld_engine.modules import (
        AttentionPairBias,
        Transition,
        TriangleMultiplication,
    )
    from miniworld_engine.modules.exceptions import ImplementationType as IT

    return {
        "apb": (lambda: AttentionPairBias(384, 128, 16, implementation=IT.MINIWORLD), lambda: [_r(1, 256, 384), _r(1, 256, 256, 128)],
                lambda m, v: m(v[0], v[1], None)),
        "trimul": (lambda: TriangleMultiplication(128, d_hidden=128, outgoing=True, p_drop=0.0, implementation=IT.MINIWORLD),
                   lambda: [_r(1, 256, 256, 128)], lambda m, v: m(v[0])),
        "transition": (lambda: Transition(128, n=4, implementation=IT.MINIWORLD), lambda: [_r(1, 256, 256, 128)],
                       lambda m, v: m(v[0])),
    }


@pytest.mark.parametrize("name", ["apb", "trimul", "transition"])
def test_inference_with_fp32_master_sees_an_in_place_update(name):
    make, inputs, call = _cases()[name]
    torch.manual_seed(0)
    master = _randomize(make()).cuda().float().eval()
    x = inputs()
    def bf16_copy(m):      # a bf16-parameter copy of the same weights
        c = make().cuda().eval()
        c.load_state_dict(m.state_dict())
        return c.to(BF)

    with torch.no_grad():
        y1 = call(master, x)
        assert _rel(y1, call(bf16_copy(master), x)) < 3e-2
        assert torch.equal(call(master, x), y1), "a second call must serve the cached forms and give the same output"
        for p in master.parameters():                       # an optimizer-style in-place step
            if p.ndim >= 2:
                p.mul_(1.25)
        y2 = call(master, x)
        assert _rel(y2, y1) > 1e-3, "the cached forms are stale: the update was not seen"
        assert _rel(y2, call(bf16_copy(master), x)) < 3e-2


def test_graph_without_static_weights_replays_current_weights():
    make, inputs, call = _cases()["apb"]
    master = _randomize(make()).cuda().float().eval()
    x = inputs()
    with torch.no_grad():
        for _ in range(2):
            call(master, x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = call(master, x)
        graph.replay()
        y1 = out.clone()
        for p in master.parameters():
            if p.ndim >= 2:
                p.mul_(1.25)
        graph.replay()
        assert _rel(out, y1) > 1e-3, "a replay must read the weights as they are, not as they were at capture"


def test_graph_under_static_weights_does_not_recast():
    """Under ``static_weights()`` the capture shares the eager entries: no cast kernel is in the replayed graph."""
    from miniworld_engine.kernels._capture import static_weights

    make, inputs, call = _cases()["apb"]
    master = _randomize(make()).cuda().float().eval()
    x = inputs()
    with torch.no_grad(), static_weights():
        for _ in range(2):
            call(master, x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call(master, x)
        graph.replay()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            graph.replay()
            torch.cuda.synchronize()
        events = [e.key for e in prof.key_averages() if e.device_time_total > 0]
    casts = [k for k in events if "copy" in k.lower() or "cast" in k.lower() or "elementwise" in k.lower()]
    assert not casts, f"cast kernels in the replayed inference graph: {casts}"
