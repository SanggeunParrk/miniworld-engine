"""A100 (sm_80) OuterProductMean / MSAPairWeightedAveraging: the cuBLASLt algorithm tables (opm_lt_table.inc, pwa_lt_table.inc), the pipelined glue tile, the weight-cast cache.

The searched cuBLASLt algorithms must (a) be no less accurate than the cuBLAS heuristic's own choice against the fp32 product, (b) repeat bit for bit, (c) give way to the
previous path for a shape that is not in the table.  The pipelined PWA glue computes the same bits as the first tuned tile; the cast cache of the parameters must see an in-place update.
"""

import pytest
import torch
from settings_helpers import override_setting

from miniworld_engine.kernels.outer_product_mean.cuda import sm80 as opm_sm80
from miniworld_engine.kernels.pair_weighted_averaging.cuda import sm80 as pwa_sm80
from miniworld_engine.modules import MSAPairWeightedAveraging, OuterProductMean
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(autouse=True)
def ampere():
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Ampere (sm_80) required")


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


# ------------------------------------------------------------------------------------------------------------------ OPM
# (M, N, K, transA, transB) of the table: fwd O = A^T B (S 1024, L 128), dA = B dO^T, dB = A dO (S 1024, L 128 and 256)
OPM_HITS = [
    (128, 1024, True, False, "fwd"),
    (128, 1024, False, True, "dA"),
    (256, 1024, False, False, "dB"),
]


@pytest.mark.parametrize(("L", "S", "ta", "tb", "name"), OPM_HITS)
def test_opm_lt_mm_is_as_accurate_and_repeatable(L, S, ta, tb, name):
    n = 32 * L
    g = torch.Generator(device="cuda").manual_seed(0)
    if name == "fwd":
        a, b = (
            torch.randn(S, n, device="cuda", generator=g).bfloat16() for _ in range(2)
        )
    elif name == "dA":
        a = torch.randn(S, n, device="cuda", generator=g).bfloat16()
        b = torch.randn(n, n, device="cuda", generator=g).bfloat16()
    else:
        a = torch.randn(S, n, device="cuda", generator=g).bfloat16()
        b = torch.randn(n, n, device="cuda", generator=g).bfloat16()
    ref32 = (a.float().t() if ta else a.float()) @ (b.float().t() if tb else b.float())
    default = torch.mm(a.t() if ta else a, b.t() if tb else b)
    ours = opm_sm80.lt_mm(a, b, ta, tb)
    again = opm_sm80.lt_mm(a, b, ta, tb)
    assert ours.shape == default.shape
    assert ours.dtype == torch.bfloat16
    assert torch.equal(ours, again), "the explicit algorithm must repeat bit for bit"
    assert _rel(ours.float(), ref32) <= 1.05 * _rel(default.float(), ref32) + 1e-6


def test_opm_lt_mm_falls_back_outside_the_table():
    """A shape that is not in the table (and a non-contiguous operand) runs at::mm: bit-identical to torch.mm."""
    g = torch.Generator(device="cuda").manual_seed(1)
    a = torch.randn(1000, 32 * 40, device="cuda", generator=g).bfloat16()
    b = torch.randn(1000, 32 * 40, device="cuda", generator=g).bfloat16()
    assert torch.equal(opm_sm80.lt_mm(a, b, True, False), torch.mm(a.t(), b))
    a2 = torch.randn(1024, 32 * 128 * 2, device="cuda", generator=g).bfloat16()[:, ::2]
    b2 = torch.randn(1024, 32 * 128, device="cuda", generator=g).bfloat16()
    assert torch.equal(opm_sm80.lt_mm(a2, b2, True, False), torch.mm(a2.t(), b2))


# ------------------------------------------------------------------------------------------------------------------ PWA
# registry shapes with an entry for all three products in pwa_lt_table.inc (a product whose explicit algorithm did not reach 1.015x stays on the default and is absent: see the next test)
@pytest.mark.parametrize(
    ("L", "C"), [(256, 32), (384, 32), (512, 32), (256, 16), (384, 8)]
)
def test_pwa_lt_bcast_and_dw_are_as_accurate_and_repeatable(L, C):
    ns, S = 4, 1024
    N = (S // ns) * C
    B = 8 * ns
    g = torch.Generator(device="cuda").manual_seed(0)
    w = torch.softmax(torch.randn(8, L, L, device="cuda", generator=g), -1).bfloat16()
    v = torch.randn(B, L, N, device="cuda", generator=g).bfloat16()
    d = torch.randn(B, L, N, device="cuda", generator=g).bfloat16()
    wx = w.unsqueeze(1).expand(8, ns, L, L).reshape(B, L, L)
    for transpose, rhs in ((False, v), (True, d)):
        ref32 = torch.bmm(
            (wx.transpose(1, 2) if transpose else wx).float(), rhs.float()
        )
        default = torch.bmm(wx.transpose(1, 2) if transpose else wx, rhs)
        ours = pwa_sm80.lt_bcast(w, rhs, transpose)
        assert ours is not None, "registry shape missing from pwa_lt_table.inc"
        assert torch.equal(ours, pwa_sm80.lt_bcast(w, rhs, transpose))
        assert _rel(ours.float(), ref32) <= 1.05 * _rel(default.float(), ref32) + 1e-6
    ref32 = torch.bmm(d.float(), v.float().transpose(1, 2))
    default = torch.bmm(d, v.transpose(1, 2), out_dtype=torch.float32)
    ours = pwa_sm80.lt_dw(d, v)
    assert ours is not None
    assert ours.dtype == torch.float32
    assert torch.equal(ours, pwa_sm80.lt_dw(d, v))
    assert _rel(ours, ref32) <= 1.05 * _rel(default, ref32) + 1e-9


def test_pwa_lt_declines_a_shape_outside_the_table():
    w = torch.softmax(torch.randn(8, 160, 160, device="cuda"), -1).bfloat16()
    v = torch.randn(32, 160, 96, device="cuda").bfloat16()
    assert pwa_sm80.lt_bcast(w, v, False) is None
    assert pwa_sm80.lt_dw(v, v) is None


# --------------------------------------------------------------------------------------------- the pipelined glue tile
def _pwa_module(seed=0):
    torch.manual_seed(seed)
    m = MSAPairWeightedAveraging(
        64, 128, 8, 32, implementation=ImplementationType.MINIWORLD, p_drop=0.15
    )
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, torch.nn.Linear):
                torch.nn.init.normal_(mod.weight, std=mod.weight.shape[1] ** -0.5)
            elif isinstance(mod, torch.nn.LayerNorm):
                torch.nn.init.normal_(mod.weight, mean=1.0, std=0.1)
                if mod.bias is not None:
                    torch.nn.init.normal_(mod.bias, std=0.1)
    return m.cuda().float()


def _pwa_train_step(module, S=1024, L=384, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    msa = (
        torch.randn(1, S, L, 64, device="cuda", generator=g)
        .bfloat16()
        .requires_grad_(True)
    )
    pair = (
        torch.randn(1, L, L, 128, device="cuda", generator=g)
        .bfloat16()
        .requires_grad_(True)
    )
    mask = torch.rand(1, L, device="cuda", generator=g) > 0.1
    dy = torch.randn(1, S, L, 64, device="cuda", generator=g).bfloat16()
    for p in module.parameters():
        p.grad = None
    torch.manual_seed(123)  # the dropout keep-mask
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = module(msa, pair, mask)
    out.backward(dy)
    return [out.detach(), msa.grad, pair.grad] + [
        p.grad.clone() for p in module.parameters()
    ]


@pytest.mark.parametrize("L", [256, 384])
def test_pwa_glue_pipeline_is_bit_identical_to_the_first_tuned_tile(monkeypatch, L):
    module = _pwa_module()
    override_setting(monkeypatch, "pwa_sm80_glue_pipeline", "1")
    piped = _pwa_train_step(module, L=L)
    override_setting(monkeypatch, "pwa_sm80_glue_pipeline", "0")
    first = _pwa_train_step(module, L=L)
    for k, (x, y) in enumerate(zip(piped, first, strict=True)):
        assert torch.equal(x, y), k


def test_pwa_lt_off_matches_on_within_the_bf16_error(monkeypatch):
    """The cuBLASLt tables change only the order of the K sums: the step stays within the bf16 module's rounding of the heuristic path."""
    module = _pwa_module()
    override_setting(monkeypatch, "pwa_sm80_lt", "1")
    on = _pwa_train_step(module)
    override_setting(monkeypatch, "pwa_sm80_lt", "0")
    off = _pwa_train_step(module)
    for k, (x, y) in enumerate(zip(on, off, strict=True)):
        assert _rel(x.float(), y.float()) < 5e-3, k


# ------------------------------------------------------------------------------------------------------- the cast cache
@pytest.mark.parametrize("which", ["opm", "pwa"])
def test_parameter_cast_cache_sees_an_in_place_update(which):
    """The kernel-dtype copies of the fp32 master parameters are cached per (pointer, version): an in-place optimizer-style update must be seen at the next call."""
    torch.manual_seed(0)
    g = torch.Generator(device="cuda").manual_seed(1)
    if which == "opm":
        module = OuterProductMean(
            64, 128, 32, implementation=ImplementationType.MINIWORLD
        )
        args = (
            torch.randn(1, 256, 128, 64, device="cuda", generator=g).bfloat16(),
            torch.ones(1, 256, 128, device="cuda", dtype=torch.bool),
        )
        kwargs = {
            "residual": torch.randn(
                1, 128, 128, 128, device="cuda", generator=g
            ).bfloat16()
        }
    else:
        module = MSAPairWeightedAveraging(
            64, 128, 8, 32, implementation=ImplementationType.MINIWORLD
        )
        args = (
            torch.randn(1, 256, 128, 64, device="cuda", generator=g).bfloat16(),
            torch.randn(1, 128, 128, 128, device="cuda", generator=g).bfloat16(),
            torch.ones(1, 128, device="cuda", dtype=torch.bool),
        )
        kwargs = {}
    with torch.no_grad():
        for mod in module.modules():
            if isinstance(mod, torch.nn.Linear):
                torch.nn.init.normal_(mod.weight, std=mod.weight.shape[1] ** -0.5)
    module = module.cuda().float().eval()

    def run():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return module(*args, **kwargs)

    before = run().clone()
    with torch.no_grad():
        module.to_out.weight.mul_(2.0)  # in place: bumps _version
    after = run()
    fresh = (
        OuterProductMean(64, 128, 32, implementation=ImplementationType.MINIWORLD)
        if which == "opm"
        else MSAPairWeightedAveraging(
            64, 128, 8, 32, implementation=ImplementationType.MINIWORLD
        )
    )
    fresh.load_state_dict(module.state_dict())
    fresh = fresh.cuda().float().eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = fresh(*args, **kwargs)
    assert not torch.equal(before, after), "the update must change the output"
    assert torch.equal(after, expected), "a stale cast of the old weight was used"


# ------------------------------------------------------------------------- the gate_out ring with few o stages (d_hidden 8)
def test_pwa_gate_out_is_repeatable_with_two_o_stages_per_tile():
    """d_hidden 8 has 2 o stages per tile against a ring of 4: the m rows of the next tile ride in the group of a stage that may still be in flight when the next tile starts (a whole token row came back
    wrong every few runs).  The same call repeated, interleaved with the stages before it, must give the same bits."""
    L, S, D, C, DZ = 256, 1024, 64, 8, 128
    g = torch.Generator(device="cuda").manual_seed(0)
    m = torch.randn(S, L, D, device="cuda", generator=g).bfloat16()
    z = torch.randn(L, L, DZ, device="cuda", generator=g).bfloat16()
    lnw, lnb = (
        1 + 0.1 * torch.randn(D, device="cuda", generator=g),
        0.1 * torch.randn(D, device="cuda", generator=g),
    )
    lnzw, lnzb = (
        1 + 0.1 * torch.randn(DZ, device="cuda", generator=g),
        0.1 * torch.randn(DZ, device="cuda", generator=g),
    )
    wv, wg = (
        (torch.randn(8 * C, D, device="cuda", generator=g) * D**-0.5).bfloat16()
        for _ in range(2)
    )
    wo = (
        torch.randn(D, 8 * C, device="cuda", generator=g) * (8 * C) ** -0.5
    ).bfloat16()
    wb = (torch.randn(8, DZ, device="cuda", generator=g) * DZ**-0.5).bfloat16()
    mask = (torch.rand(L, device="cuda", generator=g) > 0.1).to(torch.uint8)
    ns = pwa_sm80.pick_split(S)

    def step():
        w = pwa_sm80.pair_fwd(z, mask, lnzw, lnzb, wb, 1e-5)
        v, _ = pwa_sm80.ln_v(m, lnw, lnb, wv, 1e-5, False, ns)
        o = pwa_sm80.broadcast_bmm(w, v)
        return pwa_sm80.gate_out(m, o, lnw, lnb, wg, wo, None, 1e-5, 1.0)

    ref = step().clone()
    for _ in range(120):
        assert torch.equal(step(), ref)


# ------------------------------------------------------------------ OPM dgrad / dwo schedules (one barrier per larger chunk)
@pytest.mark.parametrize("cz", [128, 256, 384])
@pytest.mark.parametrize("length", [96, 200])
def test_opm_dgrad_and_dwo_schedules_are_bit_identical_to_the_first_ones(cz, length):
    """The shipped dgrad (Wo chunks of 128 / 64 rows at d_pair 256 / 384) and dwo (64 pairs a barrier) sum in the same order as the first schedules (variant 1): same bits."""
    g = torch.Generator(device="cuda").manual_seed(0)
    dz = torch.randn(length, length, cz, device="cuda", generator=g).bfloat16()
    wo = (torch.randn(cz, 1024, device="cuda", generator=g) * 0.03).bfloat16()
    norm = torch.randint(1, 5, (length, length), device="cuda", generator=g).float()
    new = opm_sm80.dgrad(dz, norm, 1.0, wo, True, variant=0)
    old = opm_sm80.dgrad(dz, norm, 1.0, wo, True, variant=1)
    assert torch.equal(new[0], old[0])
    assert torch.equal(new[1], old[1])
    o = torch.randn(32 * length, 32 * length, device="cuda", generator=g).bfloat16()
    assert torch.equal(
        opm_sm80.dwo(new[1], o, torch.float32, 0),
        opm_sm80.dwo(new[1], o, torch.float32, 1),
    )
