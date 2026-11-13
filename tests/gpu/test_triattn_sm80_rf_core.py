"""A100 (sm_80) register-resident triangle-attention inference core (``kernels/triangle_attention/cuda/sm80/attn_rf_fwd_sm80.cuh``): against a
float32 reference on sampled pair rows, the plain and the gated (``bf16(sigmoid(g) bf16(o))``) forms at 32-channel heads, the 16-channel form of
the generalised core, masked keys, a query whose keys are all masked (zero output) and logits large enough that the unshifted pass cannot
finish its tiles (the fix-list recomputation with the running maximum); bitwise equal run to run; the same values as the tuned AttnCfg core
within the bf16 class.
"""

import math

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.a100]

L2E = 1.4426950408889634
BF16_MIN = -3.3895313892515355e38


def _a100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)


def _inputs(L: int, H: int, d: int, kind: str, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    c = H * d
    buf = torch.randn(1, L, L, 4 * c, device="cuda", generator=g).to(torch.bfloat16)  # q | k | v | g columns of one buffer (the module's strides)
    scale = {"overflow": 300.0, "big": 30.0}.get(kind, 1.0)
    bias = (torch.randn(1, H, L, L, device="cuda", generator=g) * scale).to(torch.bfloat16)
    if kind == "masked":
        bias[..., 100:180] = BF16_MIN
    if kind == "allmasked":
        bias[:, :, 5:9, :] = BF16_MIN
    return buf[..., :c], buf[..., c : 2 * c], buf[..., 2 * c : 3 * c], buf[..., 3 * c :], bias


def _reference(q, k, v, g, bias, rows, d: int, gate: bool) -> torch.Tensor:
    L, H = q.shape[1], bias.shape[1]
    out = []
    for i in rows:
        qi, ki, vi = (t[0, i].float().view(L, H, d).transpose(0, 1) for t in (q, k, v))
        s = qi @ ki.transpose(1, 2) / math.sqrt(d) + bias[0].float()
        p = torch.softmax(s, -1)
        p = torch.where((bias[0].float() < -1e38).all(-1, keepdim=True), torch.zeros_like(p), p)
        o = (p @ vi).transpose(0, 1).reshape(L, H * d)
        if gate:
            o = o.to(torch.bfloat16).float() * torch.sigmoid(g[0, i].float())
        out.append(o)
    return torch.stack(out)


def _run32(mode: int, q, k, v, g, bias, gate: bool, key_mask: torch.Tensor | None = None) -> torch.Tensor:
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    ext = sm80._ext()
    L, H = q.shape[1], bias.shape[1]
    c, ld = H * 32, q.stride(2)
    out = torch.empty(1, L, L, c, device="cuda", dtype=torch.bfloat16)
    empty = torch.empty(0, device="cuda")
    km = key_mask if key_mask is not None else torch.empty(0, device="cuda", dtype=torch.uint8)
    ext.attn_rf_set(mode)
    try:
        if gate:
            ext.attn_fwd_gate_fast(q, k, v, bias, torch.empty(0, device="cuda", dtype=torch.bfloat16), empty, g, out, L, H, 1, ld, ld, ld, c, ld, c, L2E / math.sqrt(32), km)
        else:
            ext.attn_fwd_fast(q, k, v, bias, out, empty, L, H, 1, ld, ld, ld, c, L2E / math.sqrt(32), 2, 32, 2, km)
    finally:
        ext.attn_rf_set(-1)
    return out


def _run16(mode: int, q, k, v, bias, key_mask: torch.Tensor | None = None) -> torch.Tensor:
    from miniworld_engine.kernels.triangle_attention.cuda import sm80_core2 as c2

    L, H = q.shape[1], bias.shape[1]
    out = torch.empty(1, L, L, H * 16, device="cuda", dtype=torch.bfloat16)
    lay, lay_o = c2.token_major(L, q.stride(2), 16), c2.token_major(L, H * 16, 16)
    c2.ext().attn2_rf_set(mode)
    try:
        c2.forward(q, k, v, bias, torch.empty(0, dtype=torch.uint8, device="cuda"), out, torch.empty(0, device="cuda"), (lay, lay, lay, lay_o), L, H, 1, 16,
                   c2.default_scale(16), fixed=True, key_mask=key_mask)
    finally:
        c2.ext().attn2_rf_set(-1)
    return out


KINDS = ["plain", "big", "masked", "allmasked", "overflow"]


@pytest.mark.skipif(not _a100(), reason="sm_80 kernels")
@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("kind", KINDS)
def test_rf_core_32(kind: str, gate: bool) -> None:
    L, H = 512, 4
    q, k, v, g, bias = _inputs(L, H, 32, kind)
    rows = [0, 3, L // 2 + 1, L - 1]
    ref = _reference(q, k, v, g, bias, rows, 32, gate)
    rf = _run32(2, q, k, v, g, bias, gate)
    assert torch.equal(rf, _run32(2, q, k, v, g, bias, gate)), "not bitwise deterministic"
    tuned = _run32(0, q, k, v, g, bias, gate)
    err_rf = float((rf[0, rows].float() - ref).abs().max())
    err_tuned = float((tuned[0, rows].float() - ref).abs().max())
    assert not rf.isnan().any()
    assert err_rf <= max(1.5 * err_tuned, 0.02), f"rf {err_rf:.4f} vs the tuned core {err_tuned:.4f}"
    if kind == "allmasked":
        assert rf[0, :, 5:9].abs().max() == 0, "a query with every key masked gives zero"


@pytest.mark.skipif(not _a100(), reason="sm_80 kernels")
@pytest.mark.parametrize("kind", KINDS)
def test_rf_core_16(kind: str) -> None:
    L, H = 512, 4
    q, k, v, g, bias = _inputs(L, H, 16, kind, seed=1)
    rows = [0, 3, L // 2 + 1, L - 1]
    ref = _reference(q, k, v, g, bias, rows, 16, False)
    rf = _run16(1, q, k, v, bias)
    assert torch.equal(rf, _run16(1, q, k, v, bias)), "not bitwise deterministic"
    base = _run16(0, q, k, v, bias)
    err_rf = float((rf[0, rows].float() - ref).abs().max())
    err_base = float((base[0, rows].float() - ref).abs().max())
    assert not rf.isnan().any()
    assert err_rf <= max(1.5 * err_base, 0.02), f"rf {err_rf:.4f} vs the attn2 core {err_base:.4f}"
    if kind == "allmasked":
        assert rf[0, :, 5:9].abs().max() == 0


@pytest.mark.skipif(not _a100(), reason="sm_80 kernels")
@pytest.mark.parametrize("d", [32, 16])
@pytest.mark.parametrize("mode", [2, 0])
def test_dead_tail_keys_skipped_bitwise(d: int, mode: int) -> None:
    """The global key mask (padded samples: the last keys dead in every row, their bias the bf16 minimum) lets the inference cores stop
    streaming at each sample's last live key; the output is the same to the bit (every skipped weight is an exact 0).  Two samples with
    different tails, and the old cores (mode 0) as well as the rf core."""
    L, H = 640, 4
    g = torch.Generator(device="cuda").manual_seed(2)
    c = H * d
    buf = torch.randn(2, L, L, 4 * c, device="cuda", generator=g).to(torch.bfloat16)
    q, k, v, gt = buf[..., :c], buf[..., c : 2 * c], buf[..., 2 * c : 3 * c], buf[..., 3 * c :]
    bias = torch.randn(2, H, L, L, device="cuda", generator=g).to(torch.bfloat16)
    mask = torch.ones(2, L, dtype=torch.bool, device="cuda")
    mask[0, L - L // 8 :] = False
    mask[1, 300:] = False
    mask[1, 17] = False  # an interior hole is kept in the stream (only the tail is skipped)
    bias.masked_fill_(~mask[:, None, None, :], BF16_MIN)
    if d == 32:
        from miniworld_engine.kernels.triangle_attention.cuda import sm80

        ext, L2Ed = sm80._ext(), L2E / math.sqrt(32)
        empty, out = torch.empty(0, device="cuda"), torch.empty(2, L, L, c, device="cuda", dtype=torch.bfloat16)

        def run(km):
            ext.attn_rf_set(mode)
            try:
                ext.attn_fwd_fast(q, k, v, bias, out, empty, L, H, 2, 4 * c, 4 * c, 4 * c, c, L2Ed, 2, 32, 2, km)
            finally:
                ext.attn_rf_set(-1)
            return out.clone()

        full, skipped = run(torch.empty(0, device="cuda", dtype=torch.uint8)), run(mask)
    else:
        from miniworld_engine.kernels.triangle_attention.cuda import sm80_core2 as c2

        lay, lay_o = c2.token_major(L, 4 * c, 16), c2.token_major(L, c, 16)

        def run(km):
            out = torch.empty(2, L, L, c, device="cuda", dtype=torch.bfloat16)
            c2.ext().attn2_rf_set(1 if mode else 0)
            try:
                c2.forward(q, k, v, bias, torch.empty(0, dtype=torch.uint8, device="cuda"), out, torch.empty(0, device="cuda"), (lay, lay, lay, lay_o), L, H, 2,
                           16, c2.default_scale(16), fixed=True, key_mask=km)
            finally:
                c2.ext().attn2_rf_set(-1)
            return out

        full, skipped = run(None), run(mask)
    assert torch.equal(full, skipped)
