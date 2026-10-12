"""B200 bias-only DiT: saturated sigmoid gates stay finite and correct on every served path.

A gate pre-activation g <= -87.3 makes the sigmoid denominator 1 + 2^(-g log2 e) pass 2^126 (inf from -88.7). The kernels' Newton
reciprocal starts from a bit-trick seed that is only valid below 2^126, so ``pv_gate_inf`` returned NaN / inf there (phase-2a
training, 2026-10-08: one gate at g = -88.5 in block 20 turned a token row to NaN). These tests drive the attention gate, the
conditioning gates and the AdaLN scales far into saturation (|g| up to several hundred) through the bf16 and fp32 inference and
training paths, each switch setting that serves a different kernel set, and require finite results within the usual bounds.
"""

import contextlib
import copy

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import bias_only_dit_train as T
from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

F64 = torch.float64
GVALS = (0.0, -20.0, -80.0, -87.0, -87.5, -88.0, -88.5, -88.7, -89.0, -90.0, -100.0, -200.0, -1e4, 80.0, 88.5, 89.0, 100.0, 1e4)


@pytest.fixture(autouse=True)
def policy():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Blackwell (sm_100) required")
    old = settings.configure(engine_backend="auto")
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        settings.configure(**vars(old))


@contextlib.contextmanager
def tf32(on):
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = on
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


# bounds against the fp64 block, relative to the PyTorch block in the same dtype (fp32: with TF32 GEMMs, as the kernels). bf16 takes
# 1.25x where the unsaturated tests take 1.1x: with saturated gates the cond-LN weight gradients are cancellation-dominated (the
# PyTorch bf16 block itself is 23-27 % off) and the per-step launches, fused or not, sit at 1.13-1.15x of it.
BOUND = {"bf16": (1.25, 1e-4), "fp32": (1.5, 1e-3)}


def relative64(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@pytest.mark.parametrize("gval", GVALS)
def test_pv_gate_core_with_a_constant_gate(gval):
    """The bf16 attention core alone (the training forward's and the 12-launch inference step's ``pv_gate_inf``), every gate equal
    to ``gval``: finite, and within bf16 rounding of sigmoid(g) (P v)."""
    dev = torch.device("cuda")
    H, dh, A, L = 16, 48, 8, 384
    DA, M = H * dh, A * L
    torch.manual_seed(0)
    P = torch.softmax(torch.randn(H, L, L, device=dev) * 3, -1).to(torch.bfloat16)
    vg = torch.randn(M, 2 * DA, device=dev).mul_(10).to(torch.bfloat16)
    vg[:, DA:] = gval
    out = torch.empty(M, DA, device=dev, dtype=torch.bfloat16)
    T._op("pv", dev, H, dh)(vg[:, :DA], P.view(H * L, L), out, A, g=vg[:, DA:])
    torch.cuda.synchronize()
    assert torch.isfinite(out).all(), (gval, int(torch.isnan(out).sum()), int(torch.isinf(out).sum()))
    v = vg[:, :DA].double().view(A, L, H, dh)
    want = torch.einsum("hmn,anhd->amhd", P.double(), v).reshape(M, DA) * torch.sigmoid(torch.tensor(gval, dtype=F64, device=dev))
    err = float((out.double() - want).abs().max())
    assert err <= 1e-2 * float(want.abs().max()) + 1e-30, (gval, err)


def saturated_blocks(scale, n_head=16, d_head=None, seed=3):
    """A block whose gate pre-activations reach hundreds: random weights, then every gate / scale projection times ``scale``."""
    torch.manual_seed(seed)
    ref = BiasOnlyDiTBlock(n_head=n_head, d_head=d_head, implementation=ImplementationType.PYTORCH).cuda()
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5)
                if "gate" in name or "scale" in name:
                    p.mul_(scale)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return ref


def _inputs(L, A, seed=0):
    torch.manual_seed(seed)
    return (torch.randn(A, 1, L, 768, device="cuda"), torch.randn(A, 1, L, 384, device="cuda"),
            torch.randn(1, L, L, 128, device="cuda"), torch.randn(A, 1, L, 768, device="cuda"))


def _clone(ref, impl, dtype):
    m = BiasOnlyDiTBlock(n_head=ref.attention.n_head, d_head=None, implementation=impl).cuda()
    m.load_state_dict(ref.state_dict())
    return m.to(dtype)


def _gate_range(ref, x, c):
    """The attention gate pre-activations the block sees (to check the test really saturates)."""
    with torch.no_grad():
        att = ref.attention
        g = att.to_gate(att.ada_ln_in(x, c))
    return float(g.min()), float(g.max())


# every switch setting that serves a different kernel set
TRAIN_ENVS = {
    "bf16-default": ("bf16", {}),
    "bf16-unfused": ("bf16", {"MINIWORLD_BIAS_ONLY_DIT_BWD_FUSED": "0"}),
    "bf16-all-fused": ("bf16", {"MINIWORLD_BIAS_ONLY_DIT_BWD_MID": "1"}),
    "fp32-default": ("fp32", {}),
}
INFER_ENVS = {
    "bf16-three-kernel": ("bf16", {}),
    "bf16-12-launch": ("bf16", {"MINIWORLD_BIAS_ONLY_DIT_INF3_BF16": "0"}),
    "bf16-cond-tables": ("bf16", {"MINIWORLD_BIAS_ONLY_DIT_INF3_BF16": "0", "MINIWORLD_BIAS_ONLY_DIT_COND": "1"}),
    "bf16-resln": ("bf16", {"MINIWORLD_BIAS_ONLY_DIT_INF3_BF16": "0", "MINIWORLD_BIAS_ONLY_DIT_RESLN": "1"}),
    "fp32-three-kernel": ("fp32", {}),
    "fp32-12-launch": ("fp32", {"MINIWORLD_BIAS_ONLY_DIT_INF3": "0"}),
}


@pytest.mark.parametrize("scale", [60.0, 400.0])
@pytest.mark.parametrize("L", [384, 768])
@pytest.mark.parametrize("path", list(TRAIN_ENVS))
def test_training_step_with_saturated_gates(monkeypatch, path, L, scale):
    """Forward + backward with gate pre-activations of several hundred: the output and every gradient finite, and within the
    fp64 block within ``BOUND`` of the PyTorch block's error."""
    dt, env = TRAIN_ENVS[path]
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    dtype = torch.bfloat16 if dt == "bf16" else torch.float32
    ref = saturated_blocks(scale)
    A = 16
    x, c, p, dy = _inputs(L, A)
    lo, hi = _gate_range(ref, x, c)
    assert lo < -150 and hi > 150, (lo, hi)                     # the gates really reach saturation (and past -88.7)
    fast, base, ref64 = _clone(ref, ImplementationType.MINIWORLD, dtype), _clone(ref, ImplementationType.PYTORCH, dtype), \
        copy.deepcopy(ref).double()

    def run(m, dtyp):
        leaves = [t.detach().to(dtyp).requires_grad_(True) for t in (x, c, p)]
        m.zero_grad(set_to_none=True)
        y = m(*leaves, None)
        y.backward(dy.to(dtyp))
        return [y.detach()] + [t.grad for t in leaves] + [q.grad for q in m.parameters()]

    got, want = run(fast, dtype), run(ref64, F64)
    with tf32(dt == "fp32"):
        b = run(base, dtype)
    names = ["out", "d single", "d cond", "d pair"] + [n for n, _ in fast.named_parameters()]
    k, add = BOUND[dt]
    for n, g, bb, w in zip(names, got, b, want, strict=True):
        assert torch.isfinite(g).all(), (path, n, int(torch.isnan(g).sum()), int(torch.isinf(g).sum()))
        assert relative64(g, w) <= k * relative64(bb, w) + add, (path, n, relative64(g, w), relative64(bb, w))


@pytest.mark.parametrize("scale", [60.0, 400.0])
@pytest.mark.parametrize("L", [384, 768])
@pytest.mark.parametrize("path", list(INFER_ENVS))
def test_inference_with_saturated_gates(monkeypatch, path, L, scale):
    """Inference (S = 5, per-sample conditioning) with gate pre-activations of several hundred: finite and within the PyTorch
    block's error of the fp64 block (``BOUND``)."""
    dt, env = INFER_ENVS[path]
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    dtype = torch.bfloat16 if dt == "bf16" else torch.float32
    ref = saturated_blocks(scale)
    S = 5
    x, c, p, _ = _inputs(L, S)
    lo, hi = _gate_range(ref, x, c)
    assert lo < -150 and hi > 150, (lo, hi)
    fast, base, ref64 = (_clone(ref, ImplementationType.MINIWORLD, dtype).eval(), _clone(ref, ImplementationType.PYTORCH, dtype).eval(),
                         copy.deepcopy(ref).double().eval())
    with torch.no_grad():
        got = fast(x.to(dtype), c.to(dtype), p.to(dtype))
        want = ref64(x.double(), c.double(), p.double())
        with tf32(dt == "fp32"):
            b = base(x.to(dtype), c.to(dtype), p.to(dtype))
    assert torch.isfinite(got).all(), (path, int(torch.isnan(got).sum()), int(torch.isinf(got).sum()))
    k, add = BOUND[dt]
    assert relative64(got, want) <= k * relative64(b, want) + add, (path, relative64(got, want), relative64(b, want))
