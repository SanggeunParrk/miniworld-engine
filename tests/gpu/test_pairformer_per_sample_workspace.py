"""R19 (FoldForge): a ``PairformerBlock`` on ``"pytorch"`` called without autograd over B > 1 samples (FoldForge's confidence head:
``[S, L, L, 128]`` fp32 activations under bf16 autocast, one CUDA graph per bucket) runs sample by sample: the same outputs as the batched
composition, and temporaries of one sample instead of B."""

import pytest
import torch

from miniworld_engine.modules.pairformer import PairformerBlock, PairformerConfig

pytestmark = [pytest.mark.gpu]


def _block(single: bool) -> PairformerBlock:
    torch.manual_seed(0)
    cfg = PairformerConfig(d_pair=128, d_single=384 if single else 0, n_head_single=16, model="af3", p_drop=0.0)
    m = PairformerBlock(cfg).cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0.0, 0.05)
    return m


def _inputs(samples: int, length: int, dtype: torch.dtype, single: bool):
    g = torch.Generator(device="cuda").manual_seed(1)
    pair = torch.randn(samples, length, length, 128, device="cuda", generator=g).to(dtype)
    mask = (torch.arange(length, device="cuda") < length - 7)[None].expand(samples, -1).contiguous()
    s = torch.randn(samples, length, 384, device="cuda", generator=g).to(dtype) if single else None
    return pair, mask, s


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device")
@pytest.mark.parametrize("single", [True, False])
@pytest.mark.parametrize("act", ["float32", "float32_autocast", "bfloat16_autocast"])
def test_per_sample_equals_batched(single: bool, act: str) -> None:
    m = _block(single)
    dtype = torch.bfloat16 if act.startswith("bfloat16") else torch.float32
    autocast = act.endswith("autocast")
    pair, mask, s = _inputs(3, 96, dtype, single)
    args = (pair, mask, s) if single else (pair, mask)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        with torch.no_grad():
            per_sample = m(*args)
        with torch.enable_grad():  # the batched composition (a call that records autograd; nothing requires grad here but the weights)
            batched = m(*args)
    per_sample = per_sample if isinstance(per_sample, tuple) else (per_sample,)
    batched = batched if isinstance(batched, tuple) else (batched,)
    for a, b in zip(per_sample, batched, strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype
        b = b.detach()
        # not bitwise: a GEMM over one sample's rows may pick another cuBLAS kernel than over B samples' (fp32: ~2e-7 absolute)
        tol = 1e-5 if dtype == torch.float32 and not autocast else 2e-2
        torch.testing.assert_close(a, b, rtol=tol, atol=tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device")
def test_per_sample_workspace_is_one_samples() -> None:
    m = _block(True)
    samples, length = 4, 256
    pair, mask, s = _inputs(samples, length, torch.float32, True)
    peaks = {}
    for name, grad in (("per_sample", False), ("batched", True)):
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.set_grad_enabled(grad):
            m(pair, mask, s)  # warm
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            out = m(pair, mask, s)
            torch.cuda.synchronize()
            peaks[name] = torch.cuda.max_memory_allocated() - base
            del out
    pair_bytes = pair.numel() * pair.element_size()
    # per sample: the output (one pair) + one sample's temporaries; batched: B samples' temporaries
    assert peaks["per_sample"] < 0.5 * peaks["batched"], peaks
    assert peaks["per_sample"] < 1.0 * pair_bytes + 0.5 * peaks["batched"] / samples * 2, (peaks, pair_bytes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device")
def test_cuda_serves_samples_one_at_a_time_without_autograd() -> None:
    """``cuda``: the single attention takes one sample (``B == 1``); without autograd the block runs the B samples one at a time on the
    kernels (the same calls as B separate ones: bitwise), with autograd the batched call stays refused."""
    from miniworld_engine.modules.primitives import BackendUnsupported

    m = _block(True)  # fp32 masters, bf16 activations under autocast (the "master" case)
    m.set_backend("cuda")
    pair, mask, s = _inputs(3, 128, torch.bfloat16, True)
    assert s is not None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        why = m.unsupported(pair, mask, s)
        if why is not None:
            pytest.skip(f"cuda: {why}")
        p, si = m(pair, mask, s)
        ones = [m(pair[b : b + 1], mask[b : b + 1], s[b : b + 1]) for b in range(3)]
    torch.testing.assert_close(p, torch.cat([o[0] for o in ones]), rtol=0, atol=0)
    torch.testing.assert_close(si, torch.cat([o[1] for o in ones]), rtol=0, atol=0)
    with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert m.unsupported(pair, mask, s) is not None
        with pytest.raises(BackendUnsupported):
            m(pair, mask, s)
