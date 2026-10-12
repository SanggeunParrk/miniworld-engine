"""The fused token DiT paths on B200 at the head layouts beside 16 x 48: 24 x 32 and 12 x 64 (d 768), 16 x 64 (d 1024), bf16,
against the PyTorch block (inference: integrations/token_dit.py; training: integrations/token_dit_train.py)."""

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import token_dit
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]
LAYOUTS = [(24, 768), (12, 768), (16, 1024)]          # (heads, d_single)


def relative(a, b):
    return float((a.detach().float() - b.detach().float()).norm() / b.detach().float().norm().clamp_min(1e-12))


def randomize(module):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return module


@pytest.fixture(autouse=True)
def policy():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Blackwell (sm_100) required")
    old = settings.configure(engine_backend="auto")
    try:
        yield
    finally:
        settings.configure(**vars(old))


def _blocks(layout, qk=False, seed=0):
    H, d = layout
    torch.manual_seed(seed)
    m = randomize(DiTBlock(d_single=d, n_head=H, use_qk_norm=qk, implementation=ImplementationType.MINIWORLD).cuda().bfloat16())
    ref = DiTBlock(d_single=d, n_head=H, use_qk_norm=qk, implementation=ImplementationType.PYTORCH).cuda().bfloat16()
    ref.load_state_dict(m.state_dict())
    return m, ref


def _inputs(S, L, d, shared, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    bf = torch.bfloat16
    x = torch.randn(S, 1, L, d, device="cuda", generator=g).to(bf)
    c = (torch.randn(1, 1, L, 384, device="cuda", generator=g).to(bf).expand(S, 1, L, 384) if shared
         else torch.randn(S, 1, L, 384, device="cuda", generator=g).to(bf))
    p = torch.randn(1, L, L, 128, device="cuda", generator=g).to(bf)
    mask = torch.rand(1, L, device="cuda", generator=g) > 0.2
    return x, c, p, mask


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(("S", "L", "shared", "qk"), [(5, 384, True, False), (4, 200, False, False), (3, 333, True, True)])
def test_inference(layout, S, L, shared, qk):
    m, ref = _blocks(layout, qk)
    m.eval(); ref.eval()
    x, c, p, mask = _inputs(S, L, layout[1], shared)
    token_dit._RUNNERS.clear()
    with torch.no_grad():
        assert token_dit.serves(m, x, c, p)
        got = m(x, c, p, mask)
        assert got.shape == x.shape
        assert relative(got, ref(x, c, p, mask)) < 0.025
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = m(x, c, p, mask)
        x.mul_(.9)
        graph.replay()
        assert relative(captured, m(x, c, p, mask)) < 1e-5
    token_dit._RUNNERS.clear()


def _train_step(mod, x, c, p, mask, dy):
    xs, cs, ps = (t.detach().clone().requires_grad_() for t in (x, c, p))
    for q in mod.parameters():
        q.grad = None
    y = mod(xs, cs, ps, mask)
    y.backward(dy)
    return y, {"single": xs.grad, "cond": cs.grad, "pair": ps.grad, **{n: q.grad for n, q in mod.named_parameters()}}


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(("L", "qk"), [(128, False), (384, True), (384, False), (128, True)])
def test_training(layout, L, qk):
    """The fused training block (bf16) against the fp32 PyTorch block, held to the bf16 PyTorch block's own error."""
    from miniworld_engine.integrations import token_dit_train

    H, d = layout
    m, tb = _blocks(layout, qk, seed=2)
    torch.manual_seed(2)
    ref = DiTBlock(d_single=d, n_head=H, use_qk_norm=qk, implementation=ImplementationType.PYTORCH).cuda()
    ref.load_state_dict({k: v.float() for k, v in m.state_dict().items()})
    x, c, p, mask = _inputs(2, L, d, shared=False, seed=3)
    c = c.contiguous()
    dy = torch.randn_like(x)
    with torch.enable_grad():
        assert token_dit_train.serves(m, x.requires_grad_(), c, p, mask)
    x = x.detach()
    yg, gg = _train_step(m, x, c, p, mask, dy)
    yb, gb = _train_step(tb, x, c, p, mask, dy)
    yw, gw = _train_step(ref, x.float(), c.float(), p.float(), mask, dy.float())
    assert relative(yg, yw) < 1.3 * relative(yb, yw) + 1e-3
    worst = []
    for n in gw:
        if gw[n] is None or gw[n].norm() == 0:
            continue
        e, e0 = relative(gg[n], gw[n]), relative(gb[n], gw[n])
        worst.append((e / max(e0, 1e-3), n, e, e0))
    worst.sort(reverse=True)
    print("worst ratios:", worst[:3])
    for _, n, e, e0 in worst:
        assert e < 1.3 * e0 + 3e-3, (n, e, e0)
