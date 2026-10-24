"""H100 / B200 have no Triton fallback (docs/gpus/h100/dispatch.md): the architecture blocks -- which call kernels directly, not only through
the module dispatch -- launch no Triton kernel, in either precision, inference and training, windowed and global attention."""
from __future__ import annotations

import collections

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.architecture.blocks.swa_atom_transformer import SWAAtomTransformer
from miniworld_engine.modules.swa_atom_attention import build_attention_params

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10), reason="H100 / B200 only"
    ),
]


@pytest.fixture
def triton_launches(monkeypatch):
    import triton.runtime.jit as jit

    launches = collections.Counter()
    original = jit.JITFunction.run

    def run(self, *args, **kwargs):
        launches[f"{getattr(self.fn, '__module__', '?')}.{getattr(self.fn, '__name__', '?')}"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(jit.JITFunction, "run", run)
    previous = settings.configure(portable_on_hopper_blackwell=False, force_portable_kernels=False)
    yield launches
    settings.configure(**{f: getattr(previous, f) for f in ("portable_on_hopper_blackwell", "force_portable_kernels")})


def _run(config, dtype, grad, s=256, a=2, b=1):
    torch.manual_seed(0)
    model = SWAAtomTransformer(config).cuda()
    for p in model.parameters():
        p.data.normal_(0, 0.05)
    n = a * b
    q = torch.randn(n, s, config.d_atom, device="cuda", dtype=dtype, requires_grad=grad)
    c = torch.randn(n, s, config.d_cond, device="cuda", dtype=dtype)
    ref_pos = torch.randn(b, s, 3, device="cuda")
    uid = torch.arange(s, device="cuda")[None].expand(b, -1)
    cos, sin = model.build_rope(ref_pos, uid)
    valid = torch.ones(n, s, dtype=torch.bool, device="cuda")
    params = build_attention_params(cos, sin, valid, a)
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype is torch.bfloat16):
        out = model(q, c, params)
        if grad:
            out.float().sum().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("grad", [False, True], ids=["inference", "training"])
@pytest.mark.parametrize("variant", ["esmfold2", "af3", "af3_global"])
def test_the_atom_transformer_launches_no_triton(triton_launches, variant, grad, dtype):
    config = SWAAtomTransformer.Config(
        n_block=2,
        block_style="af3" if variant.startswith("af3") else "esmfold2",
        global_attn=variant == "af3_global",
    )
    _run(config, dtype, grad)
    assert not triton_launches, dict(triton_launches)


def _step(model, args, dtype, grad):
    for p in model.parameters():
        p.data.normal_(0, 0.05)
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype is torch.bfloat16):
        out = model(*args)
        out = out[0] if isinstance(out, tuple) else out
        if grad:
            out.float().sum().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("grad", [False, True], ids=["inference", "training"])
@pytest.mark.parametrize("length", [100, 256])
def test_the_pairformer_launches_no_triton(triton_launches, length, grad, dtype):
    from miniworld_engine.architecture.blocks.pairformer import Pairformer
    from miniworld_engine.architecture.exceptions import ImplementationType

    config = Pairformer.Config(
        n_block=2, p_drop=0.0, use_bidirectional_trimul=True, implementation=ImplementationType.MINIWORLD_ENGINE
    )
    model = Pairformer(config).cuda()
    pair = torch.randn(1, length, length, 128, device="cuda", dtype=dtype, requires_grad=grad)
    single = torch.randn(1, length, 384, device="cuda", dtype=dtype, requires_grad=grad)
    _step(model, (pair, single), dtype, grad)
    assert not triton_launches, dict(triton_launches)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("grad", [False, True], ids=["inference", "training"])
@pytest.mark.parametrize("length", [100, 256])
def test_the_diffusion_transformer_launches_no_triton(triton_launches, length, grad, dtype):
    from miniworld_engine.architecture.blocks.diffusion_transformer import DiffusionTransformer
    from miniworld_engine.architecture.exceptions import ImplementationType

    config = DiffusionTransformer.Config(n_block=2, implementation=ImplementationType.MINIWORLD_ENGINE)
    model = DiffusionTransformer(config).cuda()
    single = torch.randn(2, 1, length, 768, device="cuda", dtype=dtype, requires_grad=grad)
    cond = torch.randn(2, 1, length, 384, device="cuda", dtype=dtype)
    pair = torch.randn(1, length, length, 128, device="cuda", dtype=dtype)
    _step(model, (single, cond, pair), dtype, grad)
    assert not triton_launches, dict(triton_launches)
