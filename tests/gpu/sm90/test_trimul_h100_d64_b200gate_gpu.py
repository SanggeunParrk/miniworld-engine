"""The H100 D64 TriMul training dispatch: B200-style gate-gradient placement kernels and all gradients against the reference."""
import runpy
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.mark.parametrize("length", [128, 256, 1024])
@pytest.mark.parametrize("direction", [0, 1, 2])
@pytest.mark.parametrize("master", [False, True])
def test_dispatch_and_all_gradients(length, direction, master, monkeypatch):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper required")
    from miniworld_engine import settings
    from miniworld_engine.modules.exceptions import ImplementationType
    from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
    from miniworld_engine.modules.triangle_multiplication.bidirectional import (
        BidirectionalTriangleMultiplication,
    )

    helpers = runpy.run_path(str(Path(__file__).with_name("test_trimul_h100_b200style_gpu.py")))
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

    for constant in ("_TRAIN_CUDA_UNSCOPED", "_TRAIN_FUSED_PREP_UNSCOPED", "_CUDA_AUX_UNSCOPED"):
        monkeypatch.setattr(T, constant, True)
    previous = settings.configure(force_portable_kernels=False)
    try:
        leaves, mask, ds, dy = helpers["fixture"](64, length, direction, master)
        module = (BidirectionalTriangleMultiplication(64, p_drop=0.25, implementation=ImplementationType.MINIWORLD)
                  if direction == 0 else TriangleMultiplication(64, outgoing=direction == 1,
                                                               p_drop=0.25, implementation=ImplementationType.MINIWORLD)).cuda().train()
        actual = [module.to_left.weight, module.to_left_gate.weight, module.to_right.weight,
                  module.to_right_gate.weight, module.to_gate.weight, module.to_out.weight,
                  module.ln_pair.weight, module.ln_pair.bias, module.ln_out.weight, module.ln_out.bias]
        with torch.no_grad():
            for p, value in zip(actual, leaves[1:], strict=True):
                p.data = p.data.to(value.dtype)
                p.copy_(value)
        monkeypatch.setattr(module, "_make_drop_row_scale", lambda pair, p: ds[None, None])

        def step():
            y = module(leaves[0], mask)
            return y, torch.autograd.grad(y, [leaves[0], *actual], dy)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step(); step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            y, grads = step()
        # the graph's own kernel nodes (the torch profiler loses records after a cold-cache build)
        from tests.cuda_graph_nodes import graph_kernels

        names = graph_kernels(graph)
        assert names, "the captured graph has no kernel node"
        graph.replay()   # a capture computes nothing: the outputs y / grads are filled by the replay
        torch.cuda.synchronize()
        assert any("_b2w" in n for n in names), names
        pipelined = not ((direction == 0 and length == 128) or (direction == 1 and length == 256))
        affine_registers = pipelined and direction != 0 and not (
            direction == 1 and T._D64_UNI_AFFINE_FORCED)
        assert any("_b2w_affinereg" in n for n in names) == affine_registers, names
        assert any("_b2w_gatepipe" in n for n in names) == (pipelined and not affine_registers), names
        assert any("_b2w_b7gate" in n for n in names) == (not pipelined), names
        assert not any("triton" in n.lower() for n in names), names
        helpers["assert_reference"]([leaves[0], *actual], mask, ds, dy, direction, y, grads)
        with torch.no_grad():
            leaves[0].normal_()
            actual[0].mul_(0.95)
            mask.logical_not_()
            ds.zero_()
            dy.normal_()
        graph.replay()
        assert torch.equal(y, leaves[0])
        assert torch.equal(grads[0], dy)
        assert all(torch.count_nonzero(g) == 0 for g in grads[1:])
    finally:
        settings.configure(**vars(previous))


@pytest.mark.parametrize("length", [384, 768])
@pytest.mark.parametrize("master", [False, True])
def test_forced_affine_split_keeps_qualified_pipeline(length, master, monkeypatch):
    from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T

    monkeypatch.setattr(T, "_D64_UNI_AFFINE_FORCED", True)
    test_dispatch_and_all_gradients(length, 1, master, monkeypatch)
