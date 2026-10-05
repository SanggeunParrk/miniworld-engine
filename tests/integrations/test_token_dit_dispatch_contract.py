"""H100 additions preserve upstream B200 layouts, ragged lengths and dtype refusals."""

from types import SimpleNamespace

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import token_dit
from miniworld_engine.kernels.augmented_attention.cuda import sm100
from miniworld_engine.modules.dispatch import KernelBackend


@pytest.mark.parametrize(("cap", "heads", "width", "length", "dtype", "expected"), [
    ((9, 0), 16, 768, 384, torch.bfloat16, True),
    ((9, 0), 16, 768, 384, torch.float32, True),
    ((9, 0), 16, 768, 333, torch.bfloat16, False),
    ((9, 0), 24, 768, 384, torch.bfloat16, False),
    ((10, 0), 16, 768, 333, torch.bfloat16, True),
    ((10, 0), 16, 768, 333, torch.float32, True),
    ((10, 0), 24, 768, 384, torch.bfloat16, True),
    ((10, 0), 12, 768, 384, torch.bfloat16, True),
    ((10, 0), 16, 1024, 384, torch.bfloat16, True),
    ((10, 0), 24, 768, 384, torch.float32, False),
])
def test_inference_architecture_contract(monkeypatch, cap, heads, width, length, dtype, expected):
    norm = SimpleNamespace(eps=1e-5)
    adaln = SimpleNamespace(ln_in=norm, ln_cond=norm)
    module = SimpleNamespace(
        attention=SimpleNamespace(_backend=KernelBackend.TRITON, n_head=heads, use_qk_norm=True,
                                  ada_ln_in=adaln, ln_pair=norm),
        transition=SimpleNamespace(ada_ln_in=adaln, expand_a=SimpleNamespace(weight=SimpleNamespace(shape=(2 * width, width)))),
    )
    single = SimpleNamespace(is_cuda=True, device=torch.device("cuda:0"), dtype=dtype, ndim=4, shape=(4, 1, length, width))
    cond = SimpleNamespace(shape=(4, 1, length, 384))
    pair = SimpleNamespace(shape=(1, length, length, 128))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: cap)
    monkeypatch.setattr(token_dit, "_cuda_rows", lambda: True)
    monkeypatch.setattr(sm100, "_is_blackwell", lambda *_: cap == (10, 0))
    old = settings.configure(engine_backend="auto")
    try:
        with torch.no_grad():
            assert token_dit.serves(module, single, cond, pair) is expected
    finally:
        settings.configure(**vars(old))
