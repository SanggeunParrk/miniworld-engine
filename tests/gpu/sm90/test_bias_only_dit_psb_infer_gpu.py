"""The bias-only token DiT INFERENCE step on a pair bias of its own per sample (``integrations.bias_only_dit.serves_psb`` /
``update_psb``, reached through ``BiasOnlyDiTBlock(single [1, B, L, d], cond, pair [B, L, L, 128])`` or ``bias=`` from
``pair_bias_all`` with no gradient) against the PyTorch module fed the same per-sample pair in fp32 IEEE, bf16 and fp32 (TF32), with
a key mask per sample, shared or none; against the same samples run one at a time (the shared-P path); the sample chunking of a
pair-fed call; every step twice for bitwise determinism."""

import pytest
import torch

from miniworld_engine import settings
from miniworld_engine.integrations import bias_only_dit as bi
from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock, pair_bias_all
from miniworld_engine.modules.exceptions import ImplementationType

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="Hopper (sm_90) required"
    ),
]


@pytest.fixture(autouse=True)
def policy():
    old = settings.configure(force_portable_kernels=False)
    try:
        yield
    finally:
        settings.configure(**vars(old))


def rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def randomize(m):
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim == 2:
                p.normal_(std=p.shape[-1] ** -0.5 * 0.5)
            elif "weight" in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(std=0.05)
    return m


def make_pair(nh=16, dh=48):
    torch.manual_seed(7)
    ref = randomize(BiasOnlyDiTBlock(n_head=nh, d_head=dh, implementation=ImplementationType.PYTORCH).cuda().float())
    eng = BiasOnlyDiTBlock(n_head=nh, d_head=dh, implementation=ImplementationType.MINIWORLD).cuda().float()
    eng.load_state_dict(ref.state_dict())
    return ref.eval(), eng.eval()


def make_inputs(B, L, mask_kind, seed=8):
    g = torch.Generator(device="cuda").manual_seed(seed)
    single = torch.randn(1, B, L, 768, device="cuda", generator=g)
    cond = torch.randn(1, B, L, 384, device="cuda", generator=g)
    pair = torch.randn(B, L, L, 128, device="cuda", generator=g)
    if mask_kind == "none":
        mask = None
    elif mask_kind == "shared":
        mask = (torch.rand(1, L, device="cuda", generator=g) > 0.15).expand(B, L).contiguous()
    else:
        mask = torch.rand(B, L, device="cuda", generator=g) > 0.15
    return single, cond, pair, mask


class Spy:
    """Counts the calls that reach the fused per-sample inference step."""

    def __init__(self, monkeypatch):
        self.n = 0
        real = bi._infer_psb

        def counted(*a, **k):
            self.n += 1
            return real(*a, **k)

        monkeypatch.setattr(bi, "_infer_psb", counted)


def call(m, single, cond, pair, mask, dtype, hoisted, autocast=False):
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = dtype is torch.float32 and m.implementation is not ImplementationType.PYTORCH
    try:
        s, c, p = (t.to(dtype) for t in (single, cond, pair))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            if hoisted:
                bias = pair_bias_all([m], p)[0]
                return m(s, c, None, mask, bias=bias).float()
            return m(s, c, p, mask).float()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


CASES = [  # (B, L, mask)
    (2, 128, "none"),
    (4, 256, "per_sample"),
    (4, 512, "shared"),
    (5, 384, "per_sample"),
    (3, 640, "per_sample"),
    (2, 768, "none"),
]


@pytest.mark.parametrize("hoisted", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("B", "L", "mask_kind"), CASES)
def test_per_sample_inference_matches_the_fp32_module_and_is_bitwise_repeatable(dtype, B, L, mask_kind, hoisted, monkeypatch):
    ref, eng = make_pair()
    single, cond, pair, mask = make_inputs(B, L, mask_kind)
    bf = dtype is torch.bfloat16
    truth = call(ref, single, cond, pair, mask, torch.float32, hoisted=False)
    spy = Spy(monkeypatch)
    got = call(eng, single, cond, pair, mask, dtype, hoisted, autocast=bf)
    assert spy.n == 1, "the per-sample inference step did not serve the call"
    again = call(eng, single, cond, pair, mask, dtype, hoisted, autocast=bf)
    assert torch.equal(got, again), "the step is not bitwise repeatable"
    tol = 8e-3 if bf else 2.5e-3
    assert torch.isfinite(got).all()
    assert rel(got, truth) < tol, rel(got, truth)
    # one sample at a time is the shared-pair call (B = 1: the shared path): the same arithmetic up to the GEMM tile choices
    outs = []
    for b in range(B):
        m1 = None if mask is None else mask[b : b + 1]
        outs.append(call(eng, single[:, b : b + 1], cond[:, b : b + 1], pair[b : b + 1], m1, dtype, False, autocast=bf))
    assert rel(got, torch.cat(outs, 1)) < tol / 2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_pair_fed_call_runs_in_deterministic_sample_chunks(dtype, monkeypatch):
    """A pair-fed call whose bias exceeds ``PSB_CHUNK_BYTES`` runs as consecutive chunks of ``chunk_samples`` samples."""
    B, L = 5, 256
    isz = 2 if dtype is torch.bfloat16 else 4
    assert bi.chunk_samples(B, 16, L, isz) == B
    monkeypatch.setattr(bi, "PSB_CHUNK_BYTES", 2 * 16 * L * L * isz)  # two samples per chunk
    assert bi.chunk_samples(B, 16, L, isz) == 2
    ref, eng = make_pair()
    single, cond, pair, mask = make_inputs(B, L, "per_sample")
    bf = dtype is torch.bfloat16
    truth = call(ref, single, cond, pair, mask, torch.float32, hoisted=False)
    spy = Spy(monkeypatch)
    got = call(eng, single, cond, pair, mask, dtype, False, autocast=bf)
    assert spy.n == 3
    again = call(eng, single, cond, pair, mask, dtype, False, autocast=bf)
    assert torch.equal(got, again)
    assert rel(got, truth) < (8e-3 if bf else 2.5e-3)


@pytest.mark.parametrize("nh", [(24, 32), (12, 64), (16, 64)])
def test_other_head_layouts(nh, monkeypatch):
    ref, eng = make_pair(*nh)
    single, cond, pair, mask = make_inputs(3, 256, "per_sample")
    truth = call(ref, single, cond, pair, mask, torch.float32, hoisted=False)
    spy = Spy(monkeypatch)
    got = call(eng, single, cond, pair, mask, torch.float32, hoisted=True)
    assert spy.n == 1
    assert rel(got, truth) < 2.5e-3


def test_off_the_tile_and_switch_keep_the_module_composition(monkeypatch):
    ref, eng = make_pair()
    spy = Spy(monkeypatch)
    single, cond, pair, mask = make_inputs(2, 200, "per_sample")  # L off the 128 tile
    got = call(eng, single, cond, pair, mask, torch.float32, hoisted=False)
    truth = call(ref, single, cond, pair, mask, torch.float32, hoisted=False)
    assert spy.n == 0 and rel(got, truth) < 1e-3  # the module composition on TF32 GEMMs
    single, cond, pair, mask = make_inputs(2, 128, "none")
    settings.configure(force_portable_kernels=False, bias_only_dit_psb_infer=False)
    call(eng, single, cond, pair, mask, torch.float32, hoisted=False)
    assert spy.n == 0


@pytest.mark.parametrize("dh", [32, 48])
@pytest.mark.parametrize(("A", "L", "mask_kind"), [(1, 128, "none"), (3, 256, "per_sample"), (4, 384, "shared"), (2, 512, "per_sample"), (5, 512, "none")])
def test_fp32_inference_core_matches_the_training_core_and_fp64(A, L, mask_kind, dh):
    """``psb32i_fwd`` (inference: no statistics, all warps in P v) against bo_psb32.cu's forward and the fp64 softmax."""
    from miniworld_engine.kernels.bias_only_dit import cuda as C
    from miniworld_engine.kernels.bias_only_dit.cuda import psb_inf

    H = 16
    g = torch.Generator(device="cuda").manual_seed(3)
    bias = (torch.randn(H, A, L, L, device="cuda", generator=g) * 2).contiguous()
    vg = torch.randn(A * L, 2 * H * dh, device="cuda", generator=g)
    mask = None
    if mask_kind != "none":
        mask = torch.rand(A, L, device="cuda", generator=g) > 0.2
        if mask_kind == "shared":
            mask = mask[:1].expand(A, L).contiguous()
        mask[:, 0] = True
    og = torch.empty(A * L, H * dh, device="cuda")
    psb_inf.psb32i_fwd(bias.view(H * A * L, L), mask, vg, og, A, H, dh)
    again = torch.empty_like(og)
    psb_inf.psb32i_fwd(bias.view(H * A * L, L), mask, vg, again, A, H, dh)
    assert torch.equal(og, again), "not bitwise repeatable"
    old = torch.empty_like(og)
    C.psb32_fwd(bias.view(H * A * L, L), torch.empty(H * A * L, 2, device="cuda"), mask, vg, old, A, H, dh)
    assert rel(og, old) < 1e-5, rel(og, old)
    b = bias.double()
    if mask is not None:
        b = b.masked_fill(~mask[None, :, None, :], -1e30)
    p = torch.softmax(b, dim=-1)  # [H, A, L, L]
    v = vg[:, : H * dh].double().view(A, L, H, dh).permute(2, 0, 1, 3)
    gate = torch.sigmoid(vg[:, H * dh :].double().view(A, L, H, dh).permute(2, 0, 1, 3))
    truth = (torch.matmul(p, v) * gate).permute(1, 2, 0, 3).reshape(A * L, H * dh)
    assert rel(og, truth) < 2e-3, rel(og, truth)
