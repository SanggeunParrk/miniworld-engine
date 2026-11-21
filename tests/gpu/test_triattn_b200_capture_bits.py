# pyright: reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportOptionalIterable=false, reportArgumentType=false, reportGeneralTypeIssues=false, reportPrivateImportUsage=false, reportAttributeAccessIssue=false
"""B200 TriangleAttention training: the same step gives the same bits whatever the allocator's free memory holds and across independently
captured CUDA graphs (a kernel reading uninitialised memory would differ).  The free blocks of the caching allocator are pre-filled with
0x00 / 0xFF / 0x7F before each run; output, dx and every parameter gradient must be bitwise equal.

Weights: the module's default init draws from numpy's GLOBAL random state (scipy truncnorm), which ``torch.manual_seed`` does not seed --
two modules built after the same ``torch.manual_seed`` differ, and in the default init (zero out / gate weights) the only non-zero gradient,
dWo, then differs run to run.  This is a test-setup trap, not a kernel property (S90 triattn-train, docs/b200/triattn.md): seed numpy too.
"""

import numpy as np
import pytest
import torch

from miniworld_engine.modules.triangle import TriangleAttention

pytestmark = [pytest.mark.gpu]

CASES = [
    ({"d_pair": 64, "n_head": 4, "p_drop": 0.25, "starting": True}, 128),
    ({"d_pair": 128, "n_head": 4, "p_drop": 0.0, "starting": False}, 128),
    ({"d_pair": 256, "n_head": 8, "p_drop": 0.0, "starting": True}, 128),
    ({"d_pair": 128, "n_head": 4, "p_drop": 0.0, "starting": True, "use_self_attention": False}, 128),
]


def _poison(byte: int) -> None:
    big = torch.empty(8 << 30, dtype=torch.uint8, device="cuda")
    big.fill_(byte)
    del big
    small = [torch.empty(1 << 16, dtype=torch.uint8, device="cuda").fill_(byte) for _ in range(1000)]
    del small


def _run(kw: dict, length: int, byte: int, graph: bool) -> list[torch.Tensor]:
    _poison(byte)
    torch.manual_seed(0)
    np.random.seed(0)  # noqa: NPY002
    mod = TriangleAttention(**kw, backend="cuda").cuda().float().train()
    x = torch.randn(1, length, length, kw["d_pair"], device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)

    def step() -> list[torch.Tensor]:
        for p in mod.parameters():
            p.grad = None
        x.grad = None
        torch.manual_seed(1234)
        with torch.autocast("cuda", torch.bfloat16):
            y = mod(x)
        y.backward(dy)
        return [y.detach().clone(), x.grad.clone(), *[p.grad.clone() for p in mod.parameters()]]

    if not graph:
        step()
        torch.cuda.synchronize()
        _poison(byte)
        return step()
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    _poison(byte)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        res = step()
    g.replay()
    torch.cuda.synchronize()
    return [t.clone() for t in res]


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("kw,length", CASES, ids=lambda v: str(v) if not isinstance(v, dict) else f"d{v['d_pair']}h{v['n_head']}{'e' if not v['starting'] else 's'}{'b' if 'use_self_attention' in v else ''}")
def test_training_step_bits_independent_of_allocator_state(kw, length, graph):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("needs a B200 (sm_100)")
    ref = _run(kw, length, 0x00, graph)
    for byte in (0xFF, 0x7F):
        got = _run(kw, length, byte, graph)
        for i, (a, b) in enumerate(zip(ref, got, strict=True)):
            assert torch.equal(a, b), f"tensor {i} differs under allocator poison {byte:#x}"
