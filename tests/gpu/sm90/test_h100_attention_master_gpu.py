"""Full H100 master-weight attention correctness and changed-weight graph replay."""

import copy

import pytest
import torch
from tests.gpu.any.test_token_dit_train_gpu import _randomize
from tests.gpu.sm90.test_h100_master_weights_gpu import _fp32_gemm_reference

from miniworld_engine import settings
from miniworld_engine.modules import (
    AttentionPairBias,
    MSAPairWeightedAveraging,
    TriangleAttention,
)
from miniworld_engine.modules.exceptions import ImplementationType as IT
from miniworld_engine.modules.local_dit import LocalDiTBlock
from miniworld_engine.modules.swa_dit import SWADiTBlock

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="H100 required",
)]


@pytest.mark.parametrize("kind", ["apb", "local", "local_cross", "triangle128", "triangle256", "triangle384", "triangle512", "pwa", "swa"])
def test_attention_master_compile_and_live_graph(kind):
    settings.configure(engine_backend="auto", autotune_miss_cap=1)
    torch.manual_seed(89)
    if kind == "apb":
        make = lambda: AttentionPairBias(384, 128, 8, implementation=IT.MINIWORLD)
        shapes = [(1, 384, 384), (1, 384, 384, 128)]
    elif kind.startswith("local"):
        make = lambda: LocalDiTBlock(cross_attention=kind == "local_cross", implementation=IT.MINIWORLD)
        shapes = [(4, 1, 384, 128), (4, 1, 384, 128), (1, 12, 32, 128, 16)]
    elif kind == "pwa":
        make = lambda: MSAPairWeightedAveraging(64, 128, n_head=8, d_hidden=32, p_drop=0, implementation=IT.MINIWORLD)
        shapes = [(1, 256, 384, 64), (1, 384, 384, 128)]
    elif kind == "swa":
        class Block(SWADiTBlock):
            def __init__(self):
                super().__init__(implementation=IT.MINIWORLD)
                angles = torch.zeros(4, 384, 16, device="cuda")
                self.cos, self.sin = angles.cos(), angles.sin()
                self.sequence = torch.full((4,), 376, device="cuda", dtype=torch.int32)
            def forward(self, x, c, mask):
                return self.forward_hoisted(x, c, (self.cos, self.sin, self.sequence, None, 384, mask.expand(4, -1)))
        make = Block
        shapes = [(4, 384, 128), (1, 384, 128)]
    else:
        width = int(kind.removeprefix("triangle"))
        make = lambda: TriangleAttention(width, n_head=4, p_drop=0, implementation=IT.MINIWORLD)
        shapes = [(1, 384, 384, width)]
    model = _randomize(make().float()).cuda().train()
    with torch.no_grad():
        for p in model.parameters():
            # APB's exact FP32 dWb shadow (and its LN-weight-folded Wf) reads the 1-D LayerNorm vectors at the model's
            # parameter dtype. The narrow reference rounds them to bf16; an unrounded fp32 master therefore holds different
            # values and legitimately has a different dWb (norm 0.0143 on to_bias.weight) -- not a rounding defect. Give
            # both models the same bf16-representable values. The cross-attention atom block is served by the native
            # mma kernels, which read the fp32 AdaLN / bias vectors at full precision (norm 0.148 of 442 on the
            # forward), so it needs the same treatment.
            if p.ndim == 2 or kind in ("apb", "local_cross"):
                p.copy_(p.bfloat16().float())
    narrow = copy.deepcopy(model).bfloat16()
    narrow.load_state_dict(model.state_dict())
    inputs = [torch.randn(s, device="cuda", dtype=torch.bfloat16, requires_grad=True) for s in shapes]
    mask = torch.ones((1, 384), device="cuda", dtype=torch.bool)
    mask[:, 7::19] = False
    dy = torch.randn_like(inputs[0])
    parameters = list(model.named_parameters())
    verified = set()

    def run(module, mixed):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed):
            y = module(*inputs, mask)
        return (y.detach(), *torch.autograd.grad(y, [*inputs, *module.parameters()], dy))

    def compare(actual, expected, baseline=False, compiled=False):
        for i, (got, ref) in enumerate(zip(actual, expected, strict=True)):
            rounded = got.to(ref.dtype)
            if torch.equal(rounded, ref):
                continue
            pindex = i - 1 - len(inputs)
            affine = pindex >= 0 and parameters[pindex][1].ndim == 1
            if compiled and affine:
                # Check the compiled FP32 affine gradient against its eager
                # oracle. Require no worse compiler error than the narrow model,
                # using the project's existing 1.5x baseline-error convention.
                denom = eager_narrow[i].float().norm().clamp_min(1e-12)
                candidate_error = float((actual[i].float() - result[i].float()).norm() / denom)
                reference_error = float((eager_narrow[i].float() - compiled_reference[i].float()).norm() / denom)
                assert candidate_error <= 1.5 * reference_error + 1e-4, (kind, i, candidate_error, reference_error)
                continue
            if baseline and i in verified:
                # Independently verified FP32 GEMM/reduction compared with the
                # narrow reference's BF16 rounded gradient (one ULP in norm).
                tolerance = 0.0078125
            elif compiled and i == 0:
                tolerance = 0.001
            elif compiled and i > 0:
                # The compiler fuses BF16 gradient additions differently across
                # native autograd and the FP32-master Functions. Use the existing
                # H100 full-module gradient bound (0.006), preserving eager and
                # live-replay exactness and the independent FP64 GEMM checks.
                tolerance = 0.006
            elif affine and ref.dtype == torch.float32:
                tolerance = 2e-6
            elif not baseline and i in verified:
                tolerance = 2e-5
            else:
                raise AssertionError((kind, i, parameters[pindex][0] if pindex >= 0 else "activation", float((rounded.float() - ref.float()).norm())))
            error = float((rounded.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12))
            assert error < tolerance, (kind, i, error, tolerance)

    expected = eager_narrow = run(narrow, False)
    with _fp32_gemm_reference() as checked:
        result = run(model, True)
    verified.update(i for i, g in enumerate(result) if checked.matches(g))
    compare(result, expected, baseline=True)
    for (name, p), g in zip(parameters, result[1 + len(inputs):], strict=True):
        assert g.dtype == torch.float32, (kind, name)
        if p.ndim == 2 and g.any():
            assert not torch.equal(g, g.bfloat16().float()), (kind, name, "rounded gradient")
    torch._dynamo.reset()
    # Inductor fuses the reference pointwise BF16 operations and changes where
    # they round. Compare identical compiled workloads in the two parameter regimes.
    compiled_result = run(torch.compile(model, fullgraph=True), True)
    compiled_reference = run(torch.compile(narrow, fullgraph=True), False)
    # Compare compiled regimes directly. FP32 norm affine reductions also use
    # the eager master oracle and the measured narrow compiler error, because
    # compilation changes where BF16 gradient additions round.
    assert torch.equal(compiled_result[0], compiled_reference[0]), kind
    compare(compiled_result, compiled_reference, baseline=True, compiled=True)
    for (name, p), g in zip(parameters, compiled_result[1 + len(inputs):], strict=True):
        assert g.dtype == torch.float32, (kind, name, "compiled dtype")
        if p.ndim == 2 and g.any():
            assert not torch.equal(g, g.bfloat16().float()), (kind, name, "compiled rounded gradient")

    model = copy.deepcopy(model)
    inputs = [x.detach().clone().requires_grad_() for x in inputs]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(model, True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = run(model, True)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.no_grad():
        for x in inputs:
            x.mul_(0.875).add_(0.01)
        for p in model.parameters():
            p.mul_(0.875).add_(0.01)
        dy.mul_(0.75)
        mask[:, 11::23] = False
    graph.replay()
    torch.cuda.synchronize()
    compare(captured, run(model, True))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = model(*inputs, mask)
    with torch.no_grad():
        next(p for p in model.parameters() if p.ndim == 2).add_(0.01)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(y, [*inputs, *model.parameters()], dy)
