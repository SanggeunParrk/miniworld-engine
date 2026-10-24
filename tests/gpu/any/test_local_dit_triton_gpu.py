"""The local DiT block's attention core on the Triton window kernel (the portable path of the ``miniworld`` backend on a card without a
CUDA path for it), against the fp64 PyTorch module: AlphaFold 3's block (parallel K / V AdaLN, slid window) and ESMFold2's
(RMS / SiLU conditioning, q / k norm + rotary, 192 keys with the rank window), forward and every gradient."""

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.local_dit import ESMFOLD2, LocalDiTBlock

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

AF3 = {"cross_attention": True, "kv_norm": "parallel", "key_window": "slide"}
PROTENIX = {"cross_attention": True, "kv_norm": "chained", "key_window": "pad"}


@pytest.fixture(autouse=True)
def _triton_everywhere():
    """``force_triton_kernels=True``: the portable legs run on every card, H100 / B200 included (their CUDA paths are switched off)."""
    previous = settings.configure(force_triton_kernels=True)
    yield
    settings.configure(force_triton_kernels=previous.force_triton_kernels)


def _block(impl, options):
    torch.manual_seed(0)
    blk = LocalDiTBlock(128, 128, 16, 4, n=2, implementation=impl, **options)
    with torch.no_grad():
        for name, p in blk.named_parameters():
            if "norm_query" in name or "norm_key" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else torch.randn_like(p) * 0.1)
    return blk


def _inputs(a, n, real, keys, rope):
    g = torch.Generator(device="cuda").manual_seed(1)
    s = torch.randn(a, 1, n, 128, device="cuda", generator=g)
    c = torch.randn(1, 1, n, 128, device="cuda", generator=g).expand(a, 1, n, 128).contiguous()
    z = torch.randn(1, (n + 31) // 32, 32, keys, 16, device="cuda", generator=g)
    mask = torch.zeros(1, n, dtype=torch.bool, device="cuda")
    mask[:, :real] = True
    angle = 3 * torch.randn(1, n, 16, device="cuda", generator=g)
    return s, c, z, mask, (angle.cos(), angle.sin()) if rope else None


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


CASES = [("af3", AF3, 128, False, 2, 1024, 900), ("protenix", PROTENIX, 128, False, 2, 700, 700),
         ("esmfold2", ESMFOLD2, 192, True, 2, 1024, 1000), ("esmfold2-short", ESMFOLD2, 192, True, 1, 384, 150)]


@pytest.mark.parametrize(("name", "options", "keys", "rope", "a", "n", "real"), CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_the_triton_core_matches_the_pytorch_module(name, options, keys, rope, a, n, real, dtype):
    ref = _block(ImplementationType.PYTORCH, options).cuda().double()
    blk = _block(ImplementationType.MINIWORLD, options).cuda()
    blk.load_state_dict(ref.state_dict())
    own = _block(ImplementationType.PYTORCH, options).cuda()
    own.load_state_dict(ref.state_dict())
    if dtype == torch.bfloat16:
        blk, own = blk.bfloat16(), own.bfloat16()
    s, c, z, mask, r = _inputs(a, n, real, keys, rope)
    go = torch.randn_like(s)

    def run(module, dt):
        xs = [t.to(dt).clone().requires_grad_() for t in (s, c, z)]
        rr = None if r is None else tuple(t.to(dt) for t in r)
        for p in module.parameters():
            p.grad = None
        out = module(*xs, mask, rope=rr) if rope else module(*xs, mask)
        out.backward(go.to(out.dtype))
        return out.detach(), [x.grad for x in xs], {k: p.grad.double() for k, p in module.named_parameters() if p.grad is not None}

    assert blk._window_triton(s.to(dtype), 32)
    want = run(ref, torch.float64)
    base = run(own, dtype)
    got = run(blk, dtype)
    live = slice(0, real)
    # the Triton core is held to the PyTorch module's own error against the fp64 module (twice it, plus the TF32 floor of fp32)
    floor = 6e-3 if dtype == torch.float32 else 0.0
    assert _rel(got[0][:, :, live], want[0][:, :, live]) < 2 * _rel(base[0][:, :, live], want[0][:, :, live]) + floor + 1e-6
    for g, b, w in zip(got[1], base[1], want[1], strict=True):
        assert _rel(g, w) < 2 * _rel(b, w) + floor + 1e-6
    for key, w in want[2].items():
        assert _rel(got[2][key], w) < 2 * _rel(base[2][key], w) + floor + 1e-6, key


def test_the_pytorch_implementation_keeps_its_own_core():
    blk = _block(ImplementationType.PYTORCH, AF3).cuda()
    assert not blk._window_triton(torch.zeros(1, 1, 64, 128, device="cuda"), 32)
