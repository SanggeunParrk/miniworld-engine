"""B200 (sm_100a), fp32 with TF32 off: the Transition's cuda backend serves training and inference on the IEEE fp32 chain (``kernels/ieee_f32`` built for sm_100a);
it was refused to Triton (the diffusion-head conditioning transitions of the AF3 / Protenix af3_default recipe).  Accuracy: max|y - y64| / max|y64| vs float64."""

import pytest
import torch

from miniworld_engine.modules import Transition

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)), reason="B200 (sm_100a)")]


@pytest.fixture(autouse=True)
def _ieee():
    old = torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    yield
    torch.backends.cuda.matmul.allow_tf32 = old[0]
    torch.set_float32_matmul_precision(old[1])


def _err(y: torch.Tensor, y64: torch.Tensor) -> float:
    return ((y.double() - y64).abs().max() / y64.abs().max()).item()


# the diffusion-head conditioning shapes (pair d 128 n 2 [384, 384, 128]; single d 384 n 2 [1, 384, 384] / rollout [20, 384, 384]) and a few more
@pytest.mark.parametrize(("shape", "d", "n"), [((384, 384, 128), 128, 2), ((1, 384, 384), 384, 2), ((20, 384, 384), 384, 2), ((17, 256), 256, 2), ((300, 768), 768, 2), ((3, 130, 64), 64, 4)])
def test_transition_ieee_b200(shape: tuple[int, ...], d: int, n: int) -> None:
    m = Transition(d_hidden=d, n=n, master_weights=True, backend="cuda").cuda()
    ref64 = Transition(d_hidden=d, n=n, master_weights=True, backend="pytorch").cuda().double()
    ref64.load_state_dict({k: v.double() for k, v in m.state_dict().items()})
    g = torch.Generator(device="cuda").manual_seed(1)
    x = (torch.randn(shape, device="cuda", generator=g) * 2 + 0.3).requires_grad_(True)
    dy = torch.randn(shape, device="cuda", generator=g)
    assert m.unsupported(x) is None, m.unsupported(x)
    y = m(x)
    y.backward(dy)
    x3 = x.detach().double().requires_grad_(True)
    y64 = ref64(x3)
    y64.backward(dy.double())
    got = {"y": y, "dx": x.grad, **{k: p.grad for k, p in m.named_parameters()}}
    ref = {"y": y64, "dx": x3.grad, **{k: p.grad for k, p in ref64.named_parameters()}}
    for name, v in got.items():
        assert v is not None and _err(v, ref[name]) < 1e-5, (name, _err(v, ref[name]))
    with torch.no_grad():  # inference
        assert _err(m(x.detach()), y64.detach()) < 1e-5
