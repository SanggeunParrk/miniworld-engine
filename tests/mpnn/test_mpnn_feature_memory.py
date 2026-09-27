"""The fixed-geometry feature path saves distances and preserves autocast dW."""

import pytest
import torch
import torch.nn.functional as F

from miniworld_engine.modules.mpnn.features import (
    _fixed_geometry_radial_projection,
    _radial_features,
)

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_radial_projection_autocast_and_saved_storage(compiled, dtype):
    torch.manual_seed(197)
    distances = torch.rand(2, 19, 7, 25, device="cuda") * 25
    original = torch.randn(128, 416, device="cuda") * 0.05
    expected_weight = original.clone().requires_grad_()
    actual_weight = original.clone().requires_grad_()
    upstream = torch.randn(2, 19, 7, 128, device="cuda", dtype=dtype)

    def run(weight):
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            return _fixed_geometry_radial_projection(distances, weight[:, 16:], 16)

    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        expected = F.linear(_radial_features(distances, 16), expected_weight[:, 16:])
    expected.backward(upstream)
    if compiled:
        run = torch.compile(run, fullgraph=True, options={"triton.cudagraphs": False})
    saved = []

    def pack(t):
        saved.append((tuple(t.shape), t.dtype))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        actual = run(actual_weight)
        actual.backward(upstream)
    assert actual.dtype == dtype
    assert not any(shape[-1:] == (400,) for shape, _ in saved), saved
    assert any(shape == tuple(distances.shape) for shape, _ in saved), saved
    tol = 0.012 if dtype == torch.bfloat16 else 2e-4
    relative = lambda a, b: (a.float() - b.float()).norm() / b.float().norm()
    assert relative(actual, expected) < tol
    assert relative(actual_weight.grad, expected_weight.grad) < tol
    assert torch.count_nonzero(actual_weight.grad[:, :16]) == 0
