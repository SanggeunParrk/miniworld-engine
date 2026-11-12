"""The token DiT's per-block call cache (``modules/token_dit._call_cache``): what invalidates it, and that it is off where it must be."""

from __future__ import annotations

import torch

from miniworld_engine.kernels import _capture
from miniworld_engine.kernels.integrations import token_dit as integ
from miniworld_engine.modules.token_dit import _CALL_CACHE, TokenDiTProtenix, _call_cache, _kernel_view


def _block() -> TokenDiTProtenix:
    return TokenDiTProtenix(d_single=96, d_cond=32, d_pair=16, n_head=4, n=2).eval()


def test_cache_off_with_grad() -> None:
    m = _block()
    with torch.enable_grad():
        assert _call_cache(m) is None
    with torch.no_grad():
        assert _call_cache(m) is not None


def test_cache_kept_between_calls() -> None:
    m = _block()
    with torch.no_grad():
        c = _call_cache(m)
        assert c is not None
        c["x"] = 1
        assert _call_cache(m) is c


def test_parameter_replacement_invalidates() -> None:
    m = _block()
    with torch.no_grad():
        c = _call_cache(m)
        assert c is not None
        c["args", torch.float32] = integ.infer_args(_kernel_view(m), torch.float32)
        old = m.attention.to_out.weight
        m.attention.to_out.weight = torch.nn.Parameter(old.detach().clone())  # a new tensor object for the leaf
        c2 = _call_cache(m)
        assert c2 is not c and not c2
        weights = integ.infer_args(_kernel_view(m), torch.float32)[0]
        assert any(w is m.attention.to_out.weight for w in weights) and not any(w is old for w in weights)


def test_submodule_replacement_invalidates() -> None:
    m = _block()
    with torch.no_grad():
        c = _call_cache(m)
        m.transition = _block().transition
        assert _call_cache(m) is not c


def test_conversion_and_clear_drop_it() -> None:
    m = _block()
    with torch.no_grad():
        c = _call_cache(m)
        m.double()
        assert _CALL_CACHE not in m.__dict__
        c = _call_cache(m)
        _capture.clear_weight_caches(m)
        assert _CALL_CACHE not in m.__dict__
        assert _call_cache(m) is not c


def test_cpu_refusal_not_cached() -> None:
    """Off H100 (here: CPU tensors) the route is answered every call and nothing is kept for it."""
    m = _block().set_backend("cuda")
    x = torch.randn(2, 1, 8, 96)
    c, p = torch.randn(1, 1, 8, 32), torch.randn(1, 8, 8, 16)
    with torch.no_grad():
        why = m.unsupported(x, c, p, None)
        assert isinstance(why, str) and why
        assert not any(k[0] == "route" for k in _call_cache(m) or {})  # pyright: ignore[reportOptionalIterable]
