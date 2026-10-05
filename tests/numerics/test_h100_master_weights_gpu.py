"""Master gradient, compile, live replay, and weight version checks for the changed H100 paths."""

import copy
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch
from tests.integrations.test_token_dit_train_gpu import _randomize

from miniworld_engine import settings
from miniworld_engine.modules.exceptions import ImplementationType as IT
from miniworld_engine.modules.transition import Transition
from miniworld_engine.modules.triangle_multiplication import TriangleMultiplication
from miniworld_engine.modules.triangle_multiplication.bidirectional import (
    BidirectionalTriangleMultiplication,
)


@contextmanager
def _fp32_gemm_reference():
    """Check changed cuBLAS FP32 stores against strict FP32 GEMMs of saved BF16 operands.

    BF16-output cuBLAS can use reduced-precision split-K, so its rounded answer is
    not the oracle for the new FP32 output. Other matrix gradients stay bit exact.
    """
    from miniworld_engine.kernels.trimul_inproj.cuda import h100_wide_training as wide
    original_mm, original_lt, original_sum = torch.mm, wide._lt, torch.Tensor.sum
    class CheckedGEMMs(set):
        def __init__(self):
            super().__init__()
            self.outputs = []

        def matches(self, grad):
            if grad.untyped_storage().data_ptr() in self:
                return True
            # AdaLN copies the two row slices of its independently checked
            # concatenated gradient. Storage identity is lost in contiguous().
            if grad.ndim != 2:
                return False
            for output in self.outputs:
                if output.ndim == 2 and output.shape[1] == grad.shape[1] and output.shape[0] % grad.shape[0] == 0:
                    for part in output.split(grad.shape[0]):
                        if torch.equal(grad, part):
                            return True
            return False

    checked = CheckedGEMMs()

    def check(a, b, output):
        previous = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            reference = original_mm(a.double(), b.double()) if a.ndim == 2 else torch.bmm(a.double(), b.double())
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous
        error = float((output - reference).norm() / reference.norm().clamp_min(1e-12))
        # Long-K FP32 tensor-core accumulation is bounded independently of BF16
        # rounding; 1e-4 is 78 times smaller than a BF16 ULP near one.
        assert error < 1e-4, ("FP64 GEMM reference", tuple(output.shape), error)
        checked.add(output.untyped_storage().data_ptr())
        checked.outputs.append(output)

    def mm(a, b, **kwargs):
        output = original_mm(a, b, **kwargs)
        if kwargs.get("out_dtype") == torch.float32 or (a.dtype == torch.float32 and b.dtype == torch.float32):
            check(a, b, output)
        return output

    def lt(cell, name, a, b, output, workspace, donor=None):
        result = original_lt(cell, name, a, b, output, workspace, donor)
        if output.dtype == torch.float32 and name in ("dwp", "dwg"):
            check(a, b, output)
        return result

    def sum_fp32(tensor, *args, **kwargs):
        output = original_sum(tensor, *args, **kwargs)
        if kwargs.get("dtype") == torch.float32 and tensor.dtype == torch.bfloat16:
            reference_kwargs = {**kwargs, "dtype": torch.float64}
            reference = original_sum(tensor.double(), *args, **reference_kwargs)
            error = float((output - reference).norm() / reference.norm().clamp_min(1e-12))
            assert error < 1e-6, ("FP64 reduction reference", tuple(output.shape), error)
            checked.add(output.untyped_storage().data_ptr())
        return output

    with patch.object(torch, "mm", mm), patch.object(wide, "_lt", lt), patch.object(torch.Tensor, "sum", sum_fp32):
        yield checked


def _check(name, make, shape, graphs=True, compile_check=True, dropout=False):
    torch.manual_seed(88)
    master = _randomize(make().float()).cuda().train()
    with torch.no_grad():
        # Identical BF16 operands in both paths; norm affines retain FP32.
        for p in master.parameters():
            if p.ndim == 2:
                p.copy_(p.bfloat16().float())
    narrow = copy.deepcopy(master).bfloat16()
    narrow.load_state_dict(master.state_dict())
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    mask = None
    scale = None
    if dropout:
        mask = torch.rand(shape[0], shape[-2], device="cuda") > 0.15
        scale = (torch.rand(shape[0], 1, shape[-2], shape[-1], device="cuda") > 0.25).bfloat16() * (4 / 3)
        master._make_drop_row_scale = lambda pair, p: scale
        narrow._make_drop_row_scale = lambda pair, p: scale

    def forward(model):
        return model(x, mask) if dropout else model(x)

    def run(model):
        y = forward(model)
        grad = torch.autograd.grad(y, [x, *model.parameters()], dy)
        return (y.detach(), *grad)

    fp32_gemm_indices = set()

    def compare(actual, expected, phase):
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            a = a.to(b.dtype)
            # Existing wide LayerNorm uses FP32 atomic affine reductions.
            # Different executions can sum in a different order; matrix grads and BF16 outputs remain exact.
            affine = index >= 2 and list(master.parameters())[index - 2].ndim == 1
            if affine and b.dtype == torch.float32:
                assert float((a - b).norm() / b.norm().clamp_min(1e-12)) < 1e-6, (
                    name,
                    phase,
                    index,
                )
            elif index in fp32_gemm_indices and not torch.equal(a, b):
                # FP32 GEMM was independently checked before accepting the BF16
                # reduced-precision split-K difference (bounded by one BF16 ULP in norm).
                tolerance = 0.0078125 if b.dtype == torch.bfloat16 else 2e-5
                error = float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))
                assert error < tolerance, (name, phase, index, error)
            else:
                assert torch.equal(a, b), (
                    name,
                    phase,
                    index,
                    float((a.float() - b.float()).norm()),
                )

    expected = run(narrow)
    with _fp32_gemm_reference() as checked:
        got = run(master)
    fp32_gemm_indices.update(i for i, g in enumerate(got) if checked.matches(g))
    compare(got, expected, "BF16 parity")
    with torch.no_grad():
        assert torch.equal(forward(master), forward(narrow)), (name, "inference parity")
    named = list(master.named_parameters())
    for (key, p), grad in zip(named, got[2:], strict=True):
        assert grad.dtype == torch.float32, (name, key, grad.dtype)
        if p.ndim == 2 and grad.any():
            assert not torch.equal(grad, grad.bfloat16().float()), (
                name,
                key,
                "rounded master gradient",
            )
    if compile_check:
        torch._dynamo.reset()
        compiled = run(torch.compile(master, fullgraph=True))
        compare(compiled, got, "compile")
    if graphs:
        # AOTAutograd may retain AccumulateGrad nodes created on the eager stream.
        # Capture fresh leaves after checking compilation, with warmup on its own stream.
        master = copy.deepcopy(master)
        x = x.detach().clone().requires_grad_()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(master)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                out = run(master)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.no_grad():
            x.mul_(0.75).add_(0.125)
            dy.mul_(0.875).add_(0.01)
            for p in master.parameters():
                p.mul_(0.875).add_(0.01)
        if dropout:
            mask[:, 11::23] = False
            scale.mul_(0.75)
        graph.replay()
        torch.cuda.synchronize()
        fresh = run(master)
        compare(out, fresh, "live graph")
    output = forward(master)
    with torch.no_grad():
        next(p for p in master.parameters() if p.ndim == 2).add_(0.01)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(output, [x, *master.parameters()], dy)
    print(
        name
        + ": BF16 parity, unrounded FP32 gradients, compile, live graph and version checks passed",
        flush=True,
    )


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="H100 required",
    ),
]


@pytest.mark.parametrize("d", [64, 128, 256, 384, 512])
def test_transition_master_gradients_and_live_graph(d):
    settings.configure(engine_backend="auto", autotune_miss_cap=1)
    _check(
        f"transition_D{d}",
        lambda: Transition(d, n=4, implementation=IT.MINIWORLD),
        (1, 32, 32, d),
    )


@pytest.mark.parametrize("d", [64, 128, 256, 384, 512])
@pytest.mark.parametrize("length", [384, 768])
def test_bidirectional_trimul_master_gradients_and_live_graph(d, length):
    settings.configure(engine_backend="auto", autotune_miss_cap=1)
    _check(
        f"trimul_D{d}_L{length}",
        lambda: BidirectionalTriangleMultiplication(
            d, p_drop=0, implementation=IT.MINIWORLD
        ),
        (1, length, length, d),
    )


@pytest.mark.parametrize("d", [64, 128, 256, 384])
@pytest.mark.parametrize("outgoing", [False, True])
def test_single_trimul_master_gradients_and_live_graph(d, outgoing):
    settings.configure(engine_backend="auto", autotune_miss_cap=1)
    _check(
        f"trimul_single_D{d}",
        lambda: TriangleMultiplication(
            d, outgoing=outgoing, p_drop=0, implementation=IT.MINIWORLD
        ),
        (1, 384, 384, d),
    )


@pytest.mark.parametrize("d", [64, 128, 256, 384, 512])
def test_bidirectional_master_mask_dropout_and_changed_graph(d):
    settings.configure(engine_backend="auto", autotune_miss_cap=1)
    _check(
        f"trimul_dropout_D{d}",
        lambda: BidirectionalTriangleMultiplication(d, p_drop=0.25, implementation=IT.MINIWORLD),
        (1, 384, 384, d), dropout=True,
    )
