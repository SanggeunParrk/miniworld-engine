"""Run the real RMSNorm checker, including affine gradients and ragged tiles."""
from unittest.mock import patch

import pytest
import torch
import triton

from miniworld_engine.autotune.run_all import check_one
from miniworld_engine.kernels.checks import rmsnorm as checks
from miniworld_engine.kernels.rmsnorm.triton import main as rms

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


class FixedConfig:
    def __init__(self, kernel, config):
        self.kernel, self.config = kernel, config

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            return self.kernel[grid](*args, **kwargs, **self.config.kwargs,
                                     num_warps=self.config.num_warps, num_stages=self.config.num_stages)
        return launch


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1027, 31), (16384, 32)])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("block_k", [16, 64])
def test_rmsnorm_official_checkers_with_unrounded_reference(monkeypatch, dtype, shape, strided, block_k):
    m, n = shape
    monkeypatch.setattr(checks, "_D", n)
    def inputs(d):
        x = torch.randn(m, d, device="cuda", dtype=dtype)
        if strided:
            x = x.T.contiguous().T
        return x.unsqueeze(0)
    monkeypatch.setattr(checks, "_x", inputs)
    monkeypatch.setattr(checks, "vec", lambda d: torch.randn(d, device="cuda", dtype=dtype))
    config = triton.Config({"BLOCK_M1": 4, "BLOCK_K": block_k}, num_warps=4, num_stages=1)
    # Explicit test tiles exercise both loops without launching any autotune sweep.
    with patch.object(rms, "rmsnorm_fwd_kernel", FixedConfig(rms.rmsnorm_fwd_kernel.fn, config)), \
         patch.object(rms, "rmsnorm_bwd_kernel", FixedConfig(rms.rmsnorm_bwd_kernel.fn, config)):
        for backward in (False, True):
            band = .004 if dtype == torch.bfloat16 else (4e-6 if backward else 8e-7)
            name = "rmsnorm_bwd_triton" if backward else "rmsnorm_fwd_triton"
            passed, detail = check_one(f"miniworld_engine.kernels.checks.rmsnorm:{name}", band)
            assert passed, detail
