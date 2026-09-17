"""Phase 1's trainable atom SWA must compile its FA4 backward, including padding."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="H100 required")


@pytest.mark.parametrize("half_window", [-1, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fa4_compiled_backward_and_graph(half_window, dtype):
    from miniworld_engine.modules.swa_atom_attention.module import (
        _flash_window_core,
        flash_window_seqused,
    )

    n, s, h, d = 2, 128, 4, 32
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.manual_seed(51)
        lengths = torch.tensor([113, 91], device="cuda", dtype=torch.int32)
        cu = torch.arange(0, (n + 1) * s, s, device="cuda", dtype=torch.int32)
        valid = torch.arange(s, device="cuda")[None, :] < lengths[:, None]
        xs = []
        for _ in range(3):
            x = torch.randn(n, s * 2, h, d, device="cuda", dtype=dtype)[:, ::2]
            x[~valid] = float("nan")
            xs.append(x.detach().requires_grad_())
        refs = [x.detach().clone().requires_grad_() for x in xs]
        dy = torch.randn(n, s, h, d, device="cuda", dtype=dtype)

        def call(q, k, v):
            return flash_window_seqused(
                q, k, v, cu, lengths, s, valid, n, s, d**-0.5, half_window
            )

        compiled = torch.compile(
            call, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
        )
        yr = _flash_window_core(
            refs[0], refs[1], refs[2], cu, lengths, s, valid, n, s, d**-0.5, half_window
        )
        yr.backward(dy)
        y = compiled(*xs)
        y.backward(dy)

        def check(a, b):
            assert torch.isfinite(a).all()
            assert torch.equal(a[~valid], torch.zeros_like(a[~valid]))
            assert (a.float() - b.float()).norm() / b.float().norm().clamp_min(
                1e-8
            ) < 0.005

        check(y, yr)
        for a, b in zip(xs, refs, strict=False):
            check(a.grad, b.grad)
        for x in xs:
            assert x.grad is not None
            x.grad.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            gy = compiled(*xs)
            gy.backward(dy)
        graph.replay()
        torch.cuda.synchronize()
        check(gy, yr)
        for a, b in zip(xs, refs, strict=False):
            check(a.grad, b.grad)
        # Opcheck verifies fake shape/stride/dtype and functionalization in addition
        # to the explicit numerical and graph checks above.
        torch.library.opcheck(
            flash_window_seqused,
            (*xs, cu, lengths, s, valid, n, s, d**-0.5, half_window),
            test_utils=(
                "test_schema",
                "test_autograd_registration",
                "test_faketensor",
                "test_aot_dispatch_dynamic",
            ),
        )
    torch.cuda.current_stream().wait_stream(stream)


def test_fa4_partial_requires_grad():
    from miniworld_engine.modules.swa_atom_attention.module import (
        _flash_window_core,
        flash_window_seqused,
    )

    n, s, h, d = 1, 128, 4, 32
    cu = torch.tensor([0, s], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([s], device="cuda", dtype=torch.int32)
    valid = torch.ones(n, s, device="cuda", dtype=torch.bool)
    xs = [
        torch.randn(
            n, s, h, d, device="cuda", dtype=torch.bfloat16, requires_grad=(i == 2)
        )
        for i in range(3)
    ]
    ref = xs[2].detach().clone().requires_grad_()

    def call(q, k, v):
        return flash_window_seqused(q, k, v, cu, lengths, s, valid, n, s, d**-0.5, -1)

    f = torch.compile(
        call, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
    )
    f(*xs).sum().backward()
    _flash_window_core(
        xs[0], xs[1], ref, cu, lengths, s, valid, n, s, d**-0.5, -1
    ).sum().backward()
    torch.testing.assert_close(xs[2].grad, ref.grad, atol=0.01, rtol=0.01)


# Engine CI selects GPU checks explicitly.
pytestmark = [pytest.mark.gpu, *([pytestmark] if "pytestmark" in globals() else [])]
