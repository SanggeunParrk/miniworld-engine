"""The merged H100 token paths retain upstream's capture-scoped live-weight contract."""

import pytest
import torch
from tests.integrations.test_b200_pack_cache_capture_gpu import _randomize
from tests.integrations.test_b200_token_dit_gpu import (
    test_token_dit_per_sample_conditioning as _check_per_sample_conditioning,
)

from miniworld_engine import settings
from miniworld_engine.integrations import token_dit, token_dit_train
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_h100_preserves_per_sample_conditioning(dtype):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("H100 required")
    old = settings.configure(engine_backend="auto")
    try:
        _check_per_sample_conditioning(dtype)
    finally:
        settings.configure(**vars(old))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("training", [False, True])
def test_token_replay_uses_updated_weights_and_inputs(dtype, training):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("H100 required")
    old = settings.configure(engine_backend="auto")
    try:
        torch.manual_seed(23)
        model = _randomize(DiTBlock(use_qk_norm=True, implementation=ImplementationType.MINIWORLD).cuda()).to(dtype)
        args = [torch.randn(*shape, device="cuda", dtype=dtype).requires_grad_(training)
                for shape in ((2, 1, 256, 768), (2, 1, 256, 384), (1, 256, 256, 128))]
        mask = torch.rand(1, 256, device="cuda") > 0.15
        probe = torch.randn_like(args[0], dtype=torch.float32)
        wrt = args + list(model.parameters())

        def step():
            with torch.set_grad_enabled(training):
                if training:
                    assert token_dit_train.serves(model, args[0], args[1], args[2], mask)
                else:
                    assert token_dit.serves(model, args[0], args[1], args[2])
                y = model(*args, mask)
                if training:
                    grads = torch.autograd.grad((y.float() * probe).sum(), wrt)
                    return (y.detach(), *grads)
                return (y,)

        # Fill eager caches first; capture must record its own packing, on the capture stream.
        step()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                out = step()
        torch.cuda.current_stream().wait_stream(stream)
        graph.replay()
        torch.cuda.synchronize()
        before = [t.clone() for t in out]
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(0.875).add_(0.01)
            for a in args:
                a.mul_(0.75).add_(0.125)
        graph.replay()
        torch.cuda.synchronize()
        replayed = [t.clone() for t in out]
        eager = step()
        assert not torch.allclose(eager[0], before[0]), "mutation must change the output"
        for i, (r, e) in enumerate(zip(replayed, eager, strict=True)):
            assert torch.isfinite(r).all(), i
            error = (r.float() - e.float()).norm() / e.float().norm().clamp_min(1e-12)
            assert float(error) < 1e-5, (i, float(error))
    finally:
        settings.configure(**vars(old))
