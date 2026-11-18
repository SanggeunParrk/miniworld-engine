"""H100: the experimental e4m3 row kernels of the atom DiT (``local_dit_h100.FP8_ROWS``, opt-in): the bf16 inference forward with pre90_f8 /
post90_f8 stays inside the e4m3 error of the fp32 reference (about 3 % rms of the output, 7x the bf16 route), every value finite; switched off,
the default route is untouched; options outside ``sm90_atom.f8.OPTIONS`` (RoseTTAFold3's KQLN, ESMFold2's RMS), per-sample conditioning and training keep the bf16
kernels. Boltz-2's up-gate (UP: ``post90_f8up``) is on the route."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _needs_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs an H100 (sm_90)")


def _rms_rel(a, b) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _block(cls, n, a, shared: bool = True, kw: dict | None = None):
    torch.manual_seed(0)
    m = cls(128, 128, 16, 4, 2, backend="cuda", **(kw or {})).cuda().float().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * p.shape[-1] ** -0.5 if p.dim() > 1 else 1.0 + 0.1 * torch.randn_like(p))
    nwin = (n + 31) // 32
    single = torch.randn(a, 1, n, 128, device="cuda").bfloat16()
    cond = torch.randn(1, 1, n, 128, device="cuda").bfloat16()
    cond = cond.expand(a, 1, n, 128) if shared else cond.expand(a, 1, n, 128).contiguous()
    pair = torch.randn(1, nwin, 32, 128, 16, device="cuda").bfloat16()
    mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
    mask[0, n - 50 :] = False
    return m, (single, cond, pair, mask)


def _run(m, inputs, fp8):
    from miniworld_engine.kernels.integrations import local_dit_h100 as H

    prev = H.enable_fp8(fp8)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return m(*inputs)
    finally:
        H.enable_fp8(prev)


@pytest.mark.parametrize("n", [200, 1024, 4096])
@pytest.mark.parametrize("model", ["openfold3", "protenix", "chai1", "rf3", "boltz2"])
def test_fp8_rows_error(model: str, n: int, monkeypatch) -> None:
    _needs_h100()
    from miniworld_engine.kernels.integrations import local_dit_h100 as H

    monkeypatch.setattr(H, "FP8_UP_MIN_N", 0)  # Boltz-2: the shipped gate (2048 atoms) would keep the small cases on bf16
    from miniworld_engine.modules import atom_dit as A

    cls = {"openfold3": A.AtomDiTOpenFold3, "protenix": A.AtomDiTProtenix, "chai1": A.AtomDiTChai1, "rf3": A.AtomDiTRoseTTAFold3, "boltz2": A.AtomDiTBoltz2}[model]
    m, inputs = _block(cls, n, 5, kw={"adaln_eps": 1e-5} if model == "chai1" else None)
    ref = m.set_backend("pytorch")
    with torch.no_grad():
        y32 = ref(*(t.float() if t.is_floating_point() else t for t in inputs))
    m.set_backend("cuda")
    ybf = _run(m, inputs, False)
    e_bf = _rms_rel(ybf, y32)
    # Chai-1 (identity AdaLN, raw c in the tables, no out projection): about twice the e4m3 error of the AdaLN models
    # Boltz-2 (up-gate: silu(a) b u, a product of three e4m3 GEMM outputs): post a little above the other AdaLN models
    bounds = {"chai1": ((True, 0.09), ("pre", 0.09), ("post", 0.07)), "boltz2": ((True, 0.07), ("pre", 0.04), ("post", 0.06))}.get(
        model, ((True, 0.06), ("pre", 0.04), ("post", 0.05))
    )
    for parts, bound in bounds:
        y8 = _run(m, inputs, parts)
        assert torch.isfinite(y8.float()).all()
        e8 = _rms_rel(y8, y32)
        print(f"atom DiT {model} n={n} parts={parts}: rms rel vs fp32 {e8:.4f} (bf16 route {e_bf:.4f})")
        assert e8 < bound, (parts, e8, e_bf)
        assert e8 > e_bf  # the e4m3 route really ran
    # off again: the default route, bitwise
    assert torch.equal(_run(m, inputs, False), ybf)


def test_fp8_rows_fallbacks() -> None:
    """A per-sample conditioning is outside the route: FP8 on gives the bf16 output bitwise; a training forward ignores the switch (Boltz-2
    included)."""
    _needs_h100()
    from miniworld_engine.modules import atom_dit as A

    m, inputs = _block(A.AtomDiTBoltz2, 1024, 5, shared=False)
    assert torch.equal(_run(m, inputs, True), _run(m, inputs, False))
    m, inputs = _block(A.AtomDiTOpenFold3, 1024, 5, shared=False)  # per-sample conditioning tables: outside the route
    assert torch.equal(_run(m, inputs, True), _run(m, inputs, False))
    m, inputs = _block(A.AtomDiTOpenFold3, 1024, 4, shared=False)
    m.train()
    from miniworld_engine.kernels.integrations import local_dit_h100 as H

    outs = []
    for on in (False, True):
        prev = H.enable_fp8(on)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = inputs[0].clone().requires_grad_(True)
                y = m(x, *inputs[1:])
                y.float().sum().backward()
            outs.append((y.detach(), x.grad))
        finally:
            H.enable_fp8(prev)
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


def test_fp8_parts_switch() -> None:
    from miniworld_engine.kernels.integrations import local_dit_h100 as H

    prev = H.enable_fp8("pre,post")
    try:
        assert {"pre", "post"} == H.FP8_ROWS
        H.enable_fp8("post")
        assert {"post"} == H.FP8_ROWS
        H.enable_fp8(False)
        assert not H.FP8_ROWS
        with pytest.raises(ValueError):
            H.enable_fp8("attn")
    finally:
        H.enable_fp8(prev)


# ------------------------------------------------------------------------------------------------------------ ESMFold2 (RMS option)
def _esm_block(n: int, a: int, shared: bool = True):
    from miniworld_engine.modules import AtomDiTESMFold2Pair

    torch.manual_seed(0)
    m = AtomDiTESMFold2Pair(backend="cuda").cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5 if p.ndim > 1 else 1 + torch.randn_like(p) * 0.1)
    g = torch.Generator(device="cuda").manual_seed(1)
    single = torch.randn(a, 1, n, 128, device="cuda", generator=g).bfloat16()
    cond = torch.randn(1 if shared else a, 1, n, 128, device="cuda", generator=g).bfloat16().expand(a, 1, n, 128)
    pair = torch.randn(1, (n + 31) // 32, 32, 192, 16, device="cuda", generator=g).bfloat16()
    mask = torch.zeros(1, n, dtype=torch.bool, device="cuda")
    mask[:, : n - 40] = True
    angle = 3 * torch.randn(1, n, 16, device="cuda", generator=g)
    return m, (single, cond, pair, mask), (angle.cos(), angle.sin())


def _run_esm(m, inputs, rope, fp8):
    from miniworld_engine.kernels.integrations import local_dit_h100 as H

    prev = H.enable_fp8(fp8)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return m(*inputs, rope=rope)
    finally:
        H.enable_fp8(prev)


@pytest.mark.parametrize("n", [256, 1024, 2048])
@pytest.mark.parametrize("parts", [True, "pre", "post"])
def test_fp8_rows_esmfold2_pair_error(n: int, parts) -> None:
    """ESMFold2's RMS atom block (``AtomDiTESMFold2Pair``) on the e4m3 rows: finite, inside the e4m3 error of the fp64 module, the e4m3 route
    really ran (differs from bf16); switched off again the default route comes back bitwise."""
    _needs_h100()
    m, inputs, rope = _esm_block(n, 5)
    ref = m.set_backend("pytorch").double()
    with torch.no_grad():
        y64 = ref(*(t.double() if t.is_floating_point() else t for t in inputs), rope=tuple(t.double() for t in rope))
    m.float().set_backend("cuda")
    ybf = _run_esm(m, inputs, rope, False)
    y8 = _run_esm(m, inputs, rope, parts)
    real = n - 40
    e_bf, e8 = _rms_rel(ybf[:, :, :real], y64[:, :, :real]), _rms_rel(y8[:, :, :real], y64[:, :, :real])
    assert torch.isfinite(y8.float()).all()
    assert e8 < 0.06, (parts, e8, e_bf)
    assert e8 > e_bf  # the e4m3 route really ran
    assert torch.equal(_run_esm(m, inputs, rope, False), ybf)


def test_fp8_rows_esmfold2_pair_fallbacks() -> None:
    """A per-sample conditioning and a training forward keep the bf16 kernels (bitwise)."""
    _needs_h100()
    m, inputs, rope = _esm_block(1024, 5, shared=False)
    m.float()
    assert torch.equal(_run_esm(m, inputs, rope, True), _run_esm(m, inputs, rope, False))


def test_fp8_rows_rf3_kqln_served() -> None:
    """RoseTTAFold3's atom block (KQLN, the flat q / k LayerNorm) is served by the e4m3 rows: ``why_opts`` stays 0, ``calls`` / ``pre`` /
    ``post`` count it, and the split grid of small row counts keeps both 64-column units of q and of k in one CTA."""
    _needs_h100()
    from miniworld_engine.kernels.integrations import local_dit_h100 as H
    from miniworld_engine.modules import atom_dit as A

    for a, n in ((5, 200), (2, 64), (5, 1024)):
        m, inputs = _block(A.AtomDiTRoseTTAFold3, n, a)
        before = dict(H.STATS)
        y8 = _run(m, inputs, True)
        assert torch.isfinite(y8.float()).all()
        assert H.STATS["fallbacks"] == before["fallbacks"] and H.STATS["why_opts"] == before["why_opts"]
        assert H.STATS["calls"] == before["calls"] + 1 and H.STATS["pre"] == before["pre"] + 1 and H.STATS["post"] == before["post"] + 1


def test_boltz2_up_gate_on_atoms() -> None:
    """Boltz-2 (UP) takes the e4m3 route from ``FP8_UP_MIN_N`` atoms per sample: below it the output is the bf16 route's, bitwise."""
    _needs_h100()
    from miniworld_engine.kernels.integrations import local_dit_h100 as H
    from miniworld_engine.modules import atom_dit as A

    m, inputs = _block(A.AtomDiTBoltz2, 1024, 5)
    assert H.FP8_UP_MIN_N > 1024
    assert torch.equal(_run(m, inputs, True), _run(m, inputs, False))
    m, inputs = _block(A.AtomDiTBoltz2, 2048, 5)
    assert not torch.equal(_run(m, inputs, True), _run(m, inputs, False))
