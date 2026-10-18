"""InputFeatureEmbedder (ESMFold2-style atom encoder, modules/input_feature_embedder): the engine composition agrees with its PyTorch
implementation on the same weights, ragged atom counts included, and on H100 / B200 launches no Triton kernel."""
from __future__ import annotations

import collections
import copy

import pytest
import torch

from miniworld_engine.architecture.exceptions import ImplementationType
from miniworld_engine.modules import InputFeatureEmbedder

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]


def _inputs(b=1, n=300, n_token=40, seed=0):  # n=4096 is a cell whose window default is not the native H100 block
    g = torch.Generator(device="cuda").manual_seed(seed)
    pos = torch.randn(b, n, 3, device="cuda", generator=g) * 5
    atoms = torch.arange(n, device="cuda")[None].expand(b, -1)
    atom_mask = atoms < n - 17                       # front-packed: the last atoms are padding
    ref_mask = atom_mask.float()
    element = torch.randint(1, 30, (b, n), device="cuda", generator=g).float()
    charge = torch.randn(b, n, device="cuda", generator=g) * 0.3
    space_uid = atoms // 8
    atom_to_token = (atoms * n_token // n).clamp(max=n_token - 1)
    return pos, ref_mask, element, charge, space_uid, atom_mask, atom_to_token, n_token


#: window 128 (the SWA default) and global attention (a window past any atom count: MiniWorld's input embedder)
WINDOWS = {"window128": 128, "global": 1 << 18}


def _model(implementation, window=128):
    torch.manual_seed(0)
    model = InputFeatureEmbedder(128, 384, n_head=4, n_block=3, swa_window_size=window, implementation=implementation).cuda()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, p.shape[-1] ** -0.5 if p.ndim > 1 else 0.1)
    return model


def test_parameter_names_match_the_miniworld_encoder():
    names = {n.split(".")[0] for n, _ in _model(ImplementationType.MINIWORLD_ENGINE).named_parameters()}
    assert names == {"to_atom_single_cond", "atom_transformer", "atom_single_rep_to_token_single"}


@pytest.mark.parametrize("window", sorted(WINDOWS))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("grad", [False, True], ids=["inference", "training"])
def test_matches_the_pytorch_implementation(dtype, grad, window, monkeypatch):
    engine = _model(ImplementationType.MINIWORLD_ENGINE, WINDOWS[window])
    fused = []
    original = type(engine.atom_transformer)._forward_fused
    monkeypatch.setattr(type(engine.atom_transformer), "_forward_fused", lambda *a: fused.append(1) or original(*a))
    reference = copy.deepcopy(engine)
    reference.load_state_dict(engine.state_dict())
    for module in reference.modules():
        if hasattr(module, "implementation") and isinstance(module.implementation, ImplementationType):
            module.implementation = ImplementationType.PYTORCH
    args = _inputs()
    outs = []
    for model in (engine, reference):
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype is torch.bfloat16):
            y = model(*args)
            if grad:
                y.float().sum().backward()
        outs.append(y.float())
    rel = float((outs[0] - outs[1]).norm() / outs[1].norm().clamp_min(1e-12))
    assert torch.isfinite(outs[0]).all()
    assert rel < (3e-2 if dtype is torch.bfloat16 else 1e-2), rel
    if torch.cuda.get_device_capability() == (9, 0):
        assert fused, "H100: the atom transformer did not take the fused CUDA block"


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10), reason="H100 / B200 only"
)
@pytest.mark.parametrize("window", sorted(WINDOWS))
@pytest.mark.parametrize("grad", [False, True], ids=["inference", "training"])
def test_no_triton_on_h100_or_b200(monkeypatch, grad, window):
    import triton.runtime.jit as jit

    launches = collections.Counter()
    original = jit.JITFunction.run

    def run(self, *args, **kwargs):
        launches[getattr(self.fn, "__name__", "?")] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(jit.JITFunction, "run", run)
    model = _model(ImplementationType.MINIWORLD_ENGINE, WINDOWS[window])
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
        y = model(*_inputs())
        if grad:
            y.float().sum().backward()
    torch.cuda.synchronize()
    assert not launches, dict(launches)


@pytest.mark.parametrize("window", sorted(WINDOWS))
@pytest.mark.parametrize("grad", [False, True], ids=["inference", "training"])
def test_a_default_window_cell_stays_on_cuda(grad, window, monkeypatch):
    """4096 atoms, one sample: a shape the window block's per-shape defaults route away from the native H100 stages; global
    attention must still run them (it has no other path on H100)."""
    import collections

    import triton.runtime.jit as jit

    launches = collections.Counter()
    original = jit.JITFunction.run
    monkeypatch.setattr(jit.JITFunction, "run", lambda self, *a, **k: launches.update([self.fn.__name__]) or original(self, *a, **k))
    model = _model(ImplementationType.MINIWORLD_ENGINE, WINDOWS[window])
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
        y = model(*_inputs(n=4096, n_token=512))
        if grad:
            y.float().sum().backward()
    assert torch.isfinite(y).all()
    if torch.cuda.get_device_capability()[0] in (9, 10):
        assert not launches, dict(launches)
