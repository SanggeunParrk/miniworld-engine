"""B200 TriangleAttention with the model options FoldForge's families set, at their shapes (FoldForge ``modules/dense/spec.py``):
AlphaFold 3's ending-node bias orientation (``transposed_bias=False``: AF3, Boltz-2, OpenBind, IntelliFold-2, RoseTTAFold3, Chai-1),
RoseTTAFold3's gate / out-projection biases, heads of 64 channels (Chai-1's trunk and confidence head at d_pair 256, RoseTTAFold3's
template stack at d_pair 64, IntelliFold-2's trunk at d_pair 512) and Chai-1's output variants (the ending node's update left
untransposed; the confidence head's dual output).

Inference (FoldForge's use) against the module's fp64 torch reference (``_biased_reference``), the yardstick the same reference
run in bf16: the kernels may not be meaningfully worse (the bar of the other B200 TriangleAttention tests, on the update).
Training (every option: the bias orientation, the biases, the 64-channel heads and Chai-1's outputs) against fp32 autograd, the
yardstick the bf16 PyTorch module (its statements: ``_attention``, or ``_biased_reference`` for Chai-1's outputs, which ``_attention``
does not compose)."""

import copy

import pytest
import torch
from portable_helpers import relative
from settings_helpers import policy_b200_portable_off as policy  # noqa: F401

from miniworld_engine.integrations import triattn_b200 as tri
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.triangle_attention import TriangleAttention

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

RF3 = {"gate_bias": True, "out_bias": True}
# (d_pair, d_hidden, n_head, starting, transposed_bias, options): the family shapes
VARIANTS = {
    "af3-trunk-ending": (128, 128, 4, False, False, {}),                       # AF3 / Boltz-2 / OpenBind trunk, MSA module, confidence
    "af3-template-ending": (64, 64, 4, False, False, {}),                      # AF3's template stack (4 x 16)
    "boltz2-template-ending": (64, 128, 4, False, False, {}),                  # Boltz-2 / Chai-1 template stack (4 x 32)
    "if2-template-ending": (256, 256, 8, False, False, {}),                    # IntelliFold-2's template stack
    "if2-trunk-starting": (512, 512, 8, True, False, {}),                      # IntelliFold-2's trunk (8 x 64)
    "if2-trunk-ending": (512, 512, 8, False, False, {}),
    "rf3-trunk-starting": (128, 128, 4, True, False, RF3),                     # RoseTTAFold3's trunk / MSA module / confidence
    "rf3-trunk-ending": (128, 128, 4, False, False, RF3),
    "rf3-template-starting": (64, 256, 4, True, False, RF3),                   # RoseTTAFold3's template stack (4 x 64)
    "rf3-template-ending": (64, 256, 4, False, False, RF3),
    "chai1-trunk-starting": (256, 256, 4, True, False, {}),                    # Chai-1's trunk (4 x 64)
    "chai1-trunk-ending": (256, 256, 4, False, False, {"untransposed_output": True}),
    "chai1-confidence-starting": (256, 256, 4, True, False, {"dual_output": True}),
    "chai1-confidence-ending": (256, 256, 4, False, False, {"dual_output": True}),
    "chai1-template-ending": (64, 128, 4, False, False, {"untransposed_output": True}),
}
# (B, padded length, real length): whole tiles, a padded crop, two samples
SHAPES = [(1, 256, 256), (1, 384, 333), (2, 192, 150)]


def _module(d_pair, d_hidden, n_head, starting, transposed_bias, options):
    torch.manual_seed(0)
    m = TriangleAttention(d_pair, n_head, d_hidden=d_hidden, starting=starting, transposed_bias=transposed_bias,
                          implementation=ImplementationType.MINIWORLD, p_drop=0.0, **options).cuda().eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            p.copy_(1 + 0.1 * torch.randn_like(p) if name == "ln_pair.weight" else
                    torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 0.3 * torch.randn_like(p))
    return m


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


@pytest.mark.parametrize(("b", "n", "real"), SHAPES, ids=["256", "333-of-384", "B2-150-of-192"])
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_variant_inference(variant, b, n, real):
    m = _module(*VARIANTS[variant])
    d_pair = m.ln_pair.weight.shape[0]
    g = torch.Generator(device="cuda").manual_seed(1)
    z = torch.randn(b, n, n, d_pair, device="cuda", generator=g)
    mask = (torch.arange(n, device="cuda") < real).expand(b, n).contiguous()
    gold64 = copy.deepcopy(m).double()
    ref16 = copy.deepcopy(m).to(torch.bfloat16)
    eng = m.to(torch.bfloat16)
    zb = z.to(torch.bfloat16)
    with torch.no_grad():
        assert tri.kind_of(eng, zb, mask) is not None
        gold = gold64._biased_reference(zb.double(), mask)
        base = (zb + ref16._biased_reference(zb, mask)).double() - zb.double()
        got = eng(zb, mask).double() - zb.double()
    rows = (slice(None), slice(0, real), slice(0, real))
    assert torch.isfinite(got[rows]).all()
    err, err16 = _rel(got[rows], gold[rows]), _rel(base[rows], gold[rows])
    assert err <= 1.25 * err16 + 1e-3, f"{variant}: engine {err:.3e} vs bf16 reference {err16:.3e}"


def test_paths():
    """The d_pair-128 trunk on AF3's orientation keeps the fused kernels for inference; biases, the 64-channel heads and Chai-1's outputs
    take the wide ones, inference and training."""
    x128 = torch.zeros(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16)
    x256 = torch.zeros(1, 128, 128, 256, device="cuda", dtype=torch.bfloat16)
    af3 = _module(*VARIANTS["af3-trunk-ending"]).bfloat16()
    rf3 = _module(*VARIANTS["rf3-trunk-ending"]).bfloat16()
    chai = _module(*VARIANTS["chai1-trunk-ending"]).bfloat16()
    with torch.no_grad():
        assert tri.kind_of(af3, x128) == tri.FUSED
        assert tri.kind_of(rf3, x128) == tri.GENERIC
        assert tri.kind_of(chai, x256) == tri.GENERIC
    assert tri.kind_of(af3.train(), x128) == tri.GENERIC          # training: the wide path reads its bias gradient in either orientation
    assert tri.kind_of(rf3.train(), x128) == tri.GENERIC
    assert tri.kind_of(chai.train(), x256) == tri.GENERIC
    plain64 = _module(256, 256, 4, True, True, {}).bfloat16()
    with torch.no_grad():
        assert tri.kind_of(plain64, x256) == tri.GENERIC
    assert tri.kind_of(plain64.train(), x256) == tri.GENERIC


# training: every option, at the family shapes
TRAIN = ["af3-trunk-ending", "af3-template-ending", "boltz2-template-ending", "rf3-trunk-starting", "rf3-trunk-ending",
         "rf3-template-starting", "rf3-template-ending", "chai1-trunk-starting", "chai1-trunk-ending", "chai1-confidence-starting",
         "chai1-confidence-ending", "chai1-template-ending", "if2-trunk-starting", "if2-trunk-ending"]


def _trio(variant):
    """(ours, PyTorch in bf16, PyTorch in fp32) on one set of bf16-representable parameters."""
    d_pair, d_hidden, n_head, starting, transposed_bias, options = VARIANTS[variant]
    ours = _module(d_pair, d_hidden, n_head, starting, transposed_bias, options).to(torch.bfloat16)
    refs = []
    for dtype in (torch.bfloat16, torch.float32):
        r = TriangleAttention(d_pair, n_head, d_hidden=d_hidden, starting=starting, transposed_bias=transposed_bias,
                              implementation=ImplementationType.PYTORCH, p_drop=0.0, **options).cuda()
        r.load_state_dict({k: v.to(dtype) for k, v in ours.state_dict().items()})
        refs.append(r.to(dtype))
    return ours, *refs


@pytest.mark.parametrize("p_drop", [0.0, 0.25], ids=["no-dropout", "dropout"])
@pytest.mark.parametrize("variant", TRAIN)
def test_variant_training(variant, p_drop):
    """Output and every gradient (the gate / out biases included) against fp32 autograd, yardstick = the bf16 PyTorch module."""
    ours, ref16, ref32 = _trio(variant)
    d_pair = ours.ln_pair.weight.shape[0]
    L, real = 256, 230
    torch.manual_seed(3)
    x = torch.randn(1, L, L, d_pair, device="cuda", dtype=torch.bfloat16)
    mask = (torch.arange(L, device="cuda") < real)[None].contiguous()
    dy = torch.randn(1, L, L, d_pair, device="cuda") * 0.1
    for m in (ours, ref16, ref32):
        m.p_drop = p_drop
        m.train()
    assert tri.kind_of(ours, x, mask) == tri.GENERIC
    torch.manual_seed(1234)
    xo = x.clone().requires_grad_()
    y = ours(xo, mask)
    y.backward(dy.to(y.dtype))
    got = {"ours": (y, [xo.grad, *(p.grad for p in ours.parameters())])}
    scale = None
    if p_drop:
        torch.manual_seed(1234)
        scale = ours._make_drop_scale(x, p_drop)                  # the same draw, same shape and dtype
    for name, m, xin in (("ref16", ref16, x), ("ref32", ref32, x.float())):
        xi = xin.clone().requires_grad_()
        # the module's PyTorch statements (biases in its Linear layers; Chai-1's output compositions in _biased_reference)
        update = m._biased_reference(xi, mask) if (m.untransposed_output or m.dual_output) else m._attention(xi, mask)
        yr = xi + (update * scale.to(xin.dtype) if scale is not None else update)
        yr.backward(dy.to(yr.dtype))
        got[name] = (yr, [xi.grad, *(p.grad for p in m.parameters())])
    e = relative(got["ours"][0].float() - x.float(), got["ref32"][0] - x.float())
    e16 = relative(got["ref16"][0].float() - x.float(), got["ref32"][0] - x.float())
    assert e <= 1.25 * e16 + 1e-3, (e, e16)
    names = ["x", *(n for n, _ in ours.named_parameters())]
    for i, n in enumerate(names):
        err = relative(got["ours"][1][i], got["ref32"][1][i])
        err16 = relative(got["ref16"][1][i], got["ref32"][1][i])
        assert err <= 1.25 * err16 + 2e-3, (variant, n, err, err16)


@pytest.mark.parametrize("variant", ["af3-trunk-ending", "chai1-confidence-ending", "rf3-template-starting"])
def test_variant_cuda_graph_capture(variant):
    """The variant paths launch nothing host-dependent: inference captures and replays in a CUDA graph."""
    m = _module(*VARIANTS[variant]).bfloat16()
    d_pair = m.ln_pair.weight.shape[0]
    x = torch.randn(1, 256, 256, d_pair, device="cuda", dtype=torch.bfloat16)
    mask = (torch.arange(256, device="cuda") < 200)[None].contiguous()
    with torch.no_grad():
        eager = m(x, mask).clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            m(x, mask)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = m(x, mask)
        graph.replay()
    torch.cuda.synchronize()
    assert relative(out, eager) < 1e-3
