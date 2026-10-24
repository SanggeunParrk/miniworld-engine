"""The fused SWA atom DiT block (kernels/swa_dit) on a GPU: against its fp32 reference, and against the module.

Moved from team-gm (commit 14f2c73), so what is pinned here is the contract team-gm's ``SWAAtomTransformer`` called it
with: A augments x B batch elements on N = A*B rows, the adaLN modulation hoisted once per (b, atom) from the
augment-invariant conditioning, ragged front-packed ``seqused`` (including a row shorter than the half window), bf16,
d_atom 128 / 4 heads / half window 64 / SwiGLU hidden 256. Every differentiable input is checked -- q, the five block
weights, and the conditioning and adaLN weight THROUGH the hoisted modulation (so dmod is covered).

Paths: the Triton kernels everywhere (the hand-CUDA stages pinned off), both FFN weight-gradient modes, the fp32 dq1
option, and the hand-CUDA stages on sm_90 (skipped elsewhere, or when the extensions do not build here): every bf16 stage
CUDA with the slab / wgmma choice left to "auto" or forced either way, team-gm's three stages alone, and each engine CUDA
kernel alone next to Triton (the layout hand-offs); the attention-side kernels also launched directly against the Triton
kernels they replace; CUDA-graph capture + replay and torch.compile(fullgraph). A = 1 (the input feature embedder's
call) is among the shapes.

fp32 (the second half): the fp32 Triton kernels and the fp32 CUDA extension (sm_90, TF32-allowing matmul precision)
against the reference, against the module's own per-op fp32 path, against each other, under a CUDA graph, compiled
fullgraph, the ops' schema / fakes, and team-gm's unfused fp32 SWAAtomTransformer when team-gm is importable.
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import asdict

import pytest
import torch
from settings_helpers import override_setting

from miniworld_engine import settings
from miniworld_engine.kernels.swa_dit.interface import (
    swa_dit_block,
    swa_dit_hoist_modulation,
)
from miniworld_engine.kernels.swa_dit.reference import (
    swa_dit_block_reference,
    swa_dit_hoist_modulation_reference,
)
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.swa_atom_attention import build_attention_params
from miniworld_engine.modules.swa_atom_attention.module import _flash_backend
from miniworld_engine.modules.swa_dit import SWADiTBlock

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]
DEV, BF = "cuda", torch.bfloat16
C, H, HIDDEN, HW = 128, 4, 256, 64

#: The bf16 hand-CUDA stages: the three team-gm ones, then the attention side (window attention fwd + bwd, out-projection
#: backward, qkvg backward).
LEGACY_STAGES = (
    "swa_dit_qkvg_fwd_cuda",
    "swa_dit_ffn_fwd_cuda",
    "swa_dit_ffn_bwd_cuda",
)
NEW_STAGES = ("swa_dit_attn_cuda", "swa_dit_oproj_bwd_cuda", "swa_dit_qkvg_bwd_cuda")
TRITON = dict.fromkeys(LEGACY_STAGES + NEW_STAGES, False)
CUDA = {
    "force_triton_kernels": False,
    **dict.fromkeys(LEGACY_STAGES + NEW_STAGES, True),
    "swa_dit_ffn_dw": "mat",
    "swa_dit_rows_impl": "auto",
}
PATHS = {
    "triton": TRITON,
    "triton_fused_dw": {**TRITON, "swa_dit_ffn_dw": "fused"},
    "triton_dq1_fp32": {**TRITON, "swa_dit_dq1": "fp32"},
    "cuda": CUDA,  # every bf16 stage hand-CUDA (auto: slab / wgmma by shape)
    "cuda_slab": {
        **CUDA,
        "swa_dit_rows_impl": "slab",
    },  # ... with the slab kernels for qkvg fwd, FFN fwd / bwd
    "cuda_wgmma": {
        **CUDA,
        "swa_dit_rows_impl": "wgmma",
    },  # ... with team-gm's wgmma kernels for those
    "cuda_legacy": {
        **CUDA,
        **dict.fromkeys(NEW_STAGES, False),
        "swa_dit_rows_impl": "wgmma",
    },  # team-gm's three stages only
}
#: One hand-CUDA kernel at a time (everything else Triton): the attention-side stages, and the slab kernels of the stages that also
#: have a team-gm wgmma kernel.
ONE_STAGE = {
    "attn": {"swa_dit_attn_cuda": True},
    "oproj_bwd": {"swa_dit_oproj_bwd_cuda": True},
    "qkvg_bwd": {"swa_dit_qkvg_bwd_cuda": True},
    "qkvg_fwd_slab": {"swa_dit_qkvg_fwd_cuda": True, "swa_dit_rows_impl": "slab"},
    "ffn_fwd_slab": {"swa_dit_ffn_fwd_cuda": True, "swa_dit_rows_impl": "slab"},
    "ffn_bwd_slab": {"swa_dit_ffn_bwd_cuda": True, "swa_dit_rows_impl": "slab"},
}


@pytest.fixture(autouse=True)
def triton_not_sm100(monkeypatch):
    """These tests pin the Triton kernels (and the sm_90 CUDA stages) at atom counts off the 128 grid; on B200 the sm_100a
    path would refuse those counts. It has its own tests (tests/gpu/sm100/test_b200_swa_dit_gpu.py)."""
    override_setting(monkeypatch, "swa_dit_sm100", "0")


@pytest.fixture(autouse=True)
def restore_settings():
    previous = settings.current()
    yield
    settings.configure(**asdict(previous))


def _needs_cuda_stages():
    from miniworld_engine.kernels.swa_dit.cuda.loader import ERRORS, extension, is_sm90

    if not is_sm90():
        pytest.skip("the hand-CUDA stages are sm_90a kernels")
    missing = {
        w: ERRORS.get(w)
        for w in ("qkvg", "fwd", "bwd", "attn", "slab")
        if extension(w) is None
    }
    if missing:
        pytest.skip(f"the swa_dit CUDA extensions did not build here: {missing}")


def _use(path: str) -> None:
    if path.startswith("cuda"):
        _needs_cuda_stages()
    settings.configure(**PATHS[path])


def _case(a: int, b: int, s: int, seed: int = 0, dtype: torch.dtype = BF):
    """Leaves (``dtype``) for one block call: q, c_base, wmod, and the five weights; plus cos/sin/seqused (fp32, int32)."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(DEV, dtype)

    n = a * b
    leaves = [
        r(n, s, C),
        r(b, s, C),
        r(6 * C, C, scale=0.05),
        r(3 * C, C, scale=C**-0.5),
        r(C, C, scale=C**-0.5),
        r(C, C, scale=C**-0.5),
        r(2 * HIDDEN, C, scale=C**-0.5),
        r(C, HIDDEN, scale=HIDDEN**-0.5),
    ]
    angles = torch.randn(b, s, C // H // 2, generator=g) * 3.0
    lengths = [s, max(1, s - 17), 5, s - 1] * n
    seqused = torch.tensor(lengths[:n], dtype=torch.int32, device=DEV)
    return leaves, angles.cos().to(DEV), angles.sin().to(DEV), seqused


def _fused(leaves, cos, sin, seqused, b):
    q, c_base, wmod, wqkv, wg, wo, wu, wd = leaves
    mod = swa_dit_hoist_modulation(c_base, wmod)
    return swa_dit_block(
        q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, b, half_window=HW
    )


def _reference(leaves, cos, sin, seqused, b):
    q, c_base, wmod, wqkv, wg, wo, wu, wd = leaves
    mod = swa_dit_hoist_modulation_reference(c_base, wmod)
    return swa_dit_block_reference(
        q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, b, half_window=HW
    )


def _run(fn, leaves, dy, *args):
    live = [t.detach().clone().requires_grad_() for t in leaves]
    out = fn(live, *args)
    out.backward(dy.to(out.dtype))
    return out.detach(), [t.grad for t in live]


def _rel(a, e):
    a, e = a.float(), e.float()
    return ((a - e).norm() / e.norm().clamp_min(1e-30)).item(), (
        (a - e).abs().max() / e.abs().max().clamp_min(1e-30)
    ).item()


NAMES = ("q", "c_base", "w_adaln", "w_qkv", "w_gate", "w_out", "w_up", "w_down")


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize(
    ("a", "b", "s"),
    [(48, 1, 257), (3, 2, 333), (8, 2, 130), (1, 3, 300), (5, 1, 4096)],  # (5, 1, 4096): shared conditioning on a B200-style cell
)
def test_the_block_matches_its_fp32_reference(path, a, b, s):
    """Output and every gradient within bf16's band of the fp32 reference, on the same bf16 values."""
    _use(path)
    torch.manual_seed(0)
    leaves, cos, sin, seqused = _case(a, b, s)
    dy = torch.randn(
        a * b, s, C, device=DEV, dtype=BF
    )  # bf16, so both sides see the same upstream values
    out, grads = _run(_fused, leaves, dy, cos, sin, seqused, b)
    ref, ref_grads = _run(
        _reference, [t.float() for t in leaves], dy, cos, sin, seqused, b
    )
    failures = []
    for name, got, want in [
        ("out", out, ref),
        *zip(NAMES, grads, ref_grads, strict=True),
    ]:
        assert got is not None, name
        assert torch.isfinite(got).all(), name
        frob, peak = _rel(got, want)
        if frob > 3e-2 or peak > 5e-2:
            failures.append(f"{name}: rel_frob={frob:.2e} rel_max={peak:.2e}")
    assert not failures, f"{path} A={a} B={b} S={s}:\n  " + "\n  ".join(failures)


@pytest.mark.parametrize("path", ["cuda", "cuda_slab", "cuda_wgmma"])
def test_the_cuda_stages_agree_with_the_triton_kernels(path):
    """Same rounding points by construction (team-gm wrote the CUDA stages against the Triton ones)."""
    _needs_cuda_stages()
    leaves, cos, sin, seqused = _case(48, 1, 257, seed=3)
    dy = torch.randn(48, 257, C, device=DEV, dtype=BF)
    settings.configure(**TRITON)
    out_t, grads_t = _run(_fused, leaves, dy, cos, sin, seqused, 1)
    settings.configure(**PATHS[path])
    out_c, grads_c = _run(_fused, leaves, dy, cos, sin, seqused, 1)
    for name, got, want in [
        ("out", out_c, out_t),
        *zip(NAMES, grads_c, grads_t, strict=True),
    ]:
        frob, _peak = _rel(got, want)
        assert frob < 1e-2, (name, frob)


@pytest.mark.parametrize("stage", sorted(ONE_STAGE))
@pytest.mark.parametrize(("a", "b", "s"), [(48, 1, 257), (1, 3, 300), (8, 2, 130)])
def test_each_cuda_stage_agrees_with_the_triton_kernels_it_replaces(stage, a, b, s):
    """One hand-CUDA kernel on, everything else Triton: the block's output and every gradient against all-Triton.
    This also runs the layout hand-offs between a CUDA stage and its Triton neighbours (head-major attention gradients vs
    the CUDA stages' shared [M, 4C] buffer, dG in or out of it)."""
    _needs_cuda_stages()
    leaves, cos, sin, seqused = _case(a, b, s, seed=6)
    dy = torch.randn(a * b, s, C, device=DEV, dtype=BF)
    settings.configure(**TRITON)
    out_t, grads_t = _run(_fused, leaves, dy, cos, sin, seqused, b)
    settings.configure(**{**TRITON, "force_triton_kernels": False, **ONE_STAGE[stage]})
    out_c, grads_c = _run(_fused, leaves, dy, cos, sin, seqused, b)
    failures = []
    for name, got, want in [
        ("out", out_c, out_t),
        *zip(NAMES, grads_c, grads_t, strict=True),
    ]:
        frob, _peak = _rel(got, want)
        if frob > 1e-2:
            failures.append(f"{name}: rel_frob={frob:.2e}")
    assert not failures, f"{stage} A={a} B={b} S={s}:\n  " + "\n  ".join(failures)


def test_the_attention_side_cuda_kernels_match_triton_kernel_by_kernel():
    """The three attention-side extensions against the Triton kernels they replace, launched on the same inputs: the
    out-projection backward bit for bit, the window attention and the qkvg backward within bf16 rounding (the CUDA
    forward is single-pass where Triton rescales online, and both round p / ds to bf16 at the same points)."""
    import triton

    from miniworld_engine.autotune.shape_key import atom_key
    from miniworld_engine.kernels.swa_dit.cuda.loader import extension
    from miniworld_engine.kernels.swa_dit.interface import FP32_EPS
    from miniworld_engine.kernels.swa_dit.triton.backward import (
        _swa_attn_bwd_dkv_kernel,
        _swa_attn_bwd_dq_kernel,
        _swa_oproj_bwd_kernel,
        _swa_qkvg_bwd_kernel,
    )
    from miniworld_engine.kernels.swa_dit.triton.forward import _swa_attn_fwd_kernel

    _needs_cuda_stages()
    ea, ep = extension("attn"), extension("slab")
    a, b, s = 8, 2, 333
    n, m, d = a * b, a * b * s, C // H
    g = torch.Generator().manual_seed(9)

    def r(*shape, scale=1.0, dtype=BF):
        return (torch.randn(*shape, generator=g) * scale).to(DEV, dtype)

    qh, kh, vh = (r(n, H, s, d) for _ in range(3))
    seqused = torch.tensor(
        ([s, s - 17, 5, 70, 64, 129, s - 1] * n)[:n], dtype=torch.int32, device=DEV
    )
    key = atom_key(s, A=a, C=C)
    o_t = torch.empty(m, C, device=DEV, dtype=BF)
    lse_t = torch.empty(n, H, s, device=DEV, dtype=torch.float32)
    _swa_attn_fwd_kernel[lambda meta: (triton.cdiv(s, meta["BM"]), n * H)](
        qh, kh, vh, seqused, o_t, lse_t, s, d**-0.5, shape_key=key, C=C, H=H, D=d, HW=HW
    )
    o_c, lse_c = ea.attn_fwd(qh, kh, vh, seqused, d**-0.5)
    valid = (torch.arange(s, device=DEV)[None, None] < seqused[:, None, None]).expand(
        n, H, s
    )
    assert _rel(o_c, o_t)[0] < 5e-3
    assert _rel(lse_c[valid], lse_t[valid])[0] < 1e-5
    rows_valid = (torch.arange(s, device=DEV)[None] < seqused[:, None]).reshape(m)
    assert (o_c[~rows_valid] == 0).all()

    mod = r(b * s, 6 * C, scale=0.3, dtype=torch.float32)
    dq1, gate, att = r(m, C), r(m, C), r(m, C)
    wo = r(C, C, scale=C**-0.5)
    grid_t = lambda meta: (triton.cdiv(s, meta["AT"]), triton.cdiv(a, meta["SP"]), b)
    dmod_t = torch.zeros(b * s, 6 * C, device=DEV)
    do_t, dg_t, datt_t, gated_t = (
        torch.empty(m, C, device=DEV, dtype=BF) for _ in range(4)
    )
    dv_t = torch.empty(n, H, s, device=DEV)
    _swa_oproj_bwd_kernel[grid_t](
        dq1,
        o_t,
        gate,
        mod,
        wo,
        att,
        do_t,
        dg_t,
        dv_t,
        datt_t,
        gated_t,
        dmod_t,
        s,
        a,
        b,
        shape_key=key,
        C=C,
        H=H,
        D=d,
        MODW=6 * C,
    )
    dmod_c = torch.zeros_like(dmod_t)
    dp = torch.empty(m, 4 * C, device=DEV, dtype=BF)
    do_c, datt_c, gated_c, dv_c = ep.oproj_bwd(
        dq1, gate, o_t, att, mod, dmod_c, wo, dp, a, b, s
    )
    for got, want in (
        (do_c, do_t),
        (dp[:, 3 * C :], dg_t),
        (datt_c, datt_t),
        (gated_c, gated_t),
    ):
        assert torch.equal(got, want)
    assert _rel(dv_c, dv_t)[0] < 1e-6
    assert _rel(dmod_c, dmod_t)[0] < 1e-6

    dq_t, dk_t, dvh_t = (
        torch.empty(n, H, s, d, device=DEV, dtype=BF) for _ in range(3)
    )
    _swa_attn_bwd_dq_kernel[lambda meta: (triton.cdiv(s, meta["BM"]), n * H)](
        qh,
        kh,
        vh,
        do_t,
        lse_t,
        dv_t,
        seqused,
        dq_t,
        s,
        d**-0.5,
        shape_key=key,
        C=C,
        H=H,
        D=d,
        HW=HW,
    )
    _swa_attn_bwd_dkv_kernel[lambda meta: (triton.cdiv(s, meta["BN"]), n * H)](
        qh,
        kh,
        vh,
        do_t,
        lse_t,
        dv_t,
        seqused,
        dk_t,
        dvh_t,
        s,
        d**-0.5,
        shape_key=key,
        C=C,
        H=H,
        D=d,
        HW=HW,
    )
    heads = ea.attn_bwd(qh, kh, vh, do_t, lse_t, dv_t, seqused, None, d**-0.5)
    ea.attn_bwd(qh, kh, vh, do_t, lse_t, dv_t, seqused, dp, d**-0.5)
    for i, (got, want) in enumerate(zip(heads, (dq_t, dk_t, dvh_t), strict=True)):
        assert _rel(got, want)[0] < 1e-3, i
        assert torch.equal(
            dp[:, i * C : (i + 1) * C].view(n, s, H, d).transpose(1, 2), got
        ), i

    q, pq, pk = r(m, C), r(m, C), r(m, C)
    angles = torch.randn(b * s, d // 2, generator=g) * 3.0
    cos, sin = angles.cos().to(DEV), angles.sin().to(DEV)
    wqkv, wg = r(3 * C, C, scale=C**-0.5), r(C, C, scale=C**-0.5)
    dmod_t.zero_()
    dq_out_t = torch.empty(m, C, device=DEV, dtype=BF)
    dp_t = torch.empty(m, 4 * C, device=DEV, dtype=BF)
    _swa_qkvg_bwd_kernel[grid_t](
        q,
        mod,
        cos,
        sin,
        wqkv,
        wg,
        pq,
        pk,
        dq_t,
        dk_t,
        dvh_t,
        dg_t,
        dq1,
        dq_out_t,
        dp_t,
        dmod_t,
        s,
        a,
        b,
        1e-5,
        FP32_EPS,
        shape_key=key,
        C=C,
        H=H,
        D=d,
        MODW=6 * C,
    )
    for i, t in enumerate((dq_t, dk_t, dvh_t)):
        dp[:, i * C : (i + 1) * C].view(n, s, H, d).copy_(t.transpose(1, 2))
    dmod_c.zero_()
    dq_out_c = ep.qkvg_bwd(
        q, pq, pk, dq1, mod, dmod_c, cos, sin, wqkv, wg, dp, a, b, s, 1e-5, FP32_EPS
    )
    assert _rel(dq_out_c, dq_out_t)[0] < 1e-4
    assert _rel(dp, dp_t)[0] < 1e-4
    assert _rel(dmod_c, dmod_t)[0] < 1e-4


@pytest.mark.parametrize("path", ["triton", "cuda", "cuda_wgmma"])
def test_a_cuda_graph_replays_the_training_step(path):
    """Forward + backward captured once and replayed on new inputs (copied into the captured buffers) gives what an eager
    step gives on those inputs: every launch is capturable and nothing depends on host-side state between replays."""
    _use(path)
    a, b, s = 3, 2, 333
    leaves, cos, sin, seqused = _case(a, b, s, seed=7)
    live = [t.detach().clone().requires_grad_() for t in leaves]
    dy = torch.randn(a * b, s, C, device=DEV, dtype=BF)

    def step():
        for t in live:
            t.grad = None
        out = _fused(live, cos, sin, seqused, b)
        out.backward(dy)
        return out

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_g = step()
    fresh, _, _, _ = _case(a, b, s, seed=8)
    with torch.no_grad():
        for t, f in zip(live, fresh, strict=True):
            t.copy_(f)
    graph.replay()
    torch.cuda.synchronize()
    replayed = [out_g.detach().clone(), *(t.grad.clone() for t in live)]
    eager_out, eager_grads = _run(_fused, fresh, dy, cos, sin, seqused, b)
    # The modulation gradient (so the conditioning's and the adaLN weight's) is accumulated with fp32 atomics whose order
    # differs run to run, so two EAGER steps differ there too: the replay may differ from eager by what eager differs from
    # itself (x4), and by no more than rounding elsewhere.
    again_out, again_grads = _run(_fused, fresh, dy, cos, sin, seqused, b)
    for name, got, want, again in zip(
        ("out", *NAMES),
        replayed,
        [eager_out, *eager_grads],
        [again_out, *again_grads],
        strict=True,
    ):
        frob, _peak = _rel(got, want)
        assert frob <= max(1e-5, 4 * _rel(again, want)[0]), (
            name,
            frob,
            _rel(again, want)[0],
        )


@pytest.mark.parametrize("path", ["triton", "cuda"])
def test_inference_and_training_forward_agree(path):
    """No gradient recorded -> the inference forward (nothing saved); same numbers as the training forward."""
    _use(path)
    leaves, cos, sin, seqused = _case(3, 2, 333, seed=4)
    with torch.no_grad():
        inference = _fused(leaves, cos, sin, seqused, 2)
    training, _ = _run(
        _fused,
        leaves,
        torch.randn(6, 333, C, device=DEV, dtype=BF),
        cos,
        sin,
        seqused,
        2,
    )
    torch.testing.assert_close(inference, training, rtol=1e-2, atol=1e-2)


def _modulation_weight(block: SWADiTBlock) -> torch.Tensor:
    """The adaLN projection's weight (``adaln_modulation`` is SiLU -> Linear)."""
    linear = block.adaln_modulation[1]
    assert isinstance(linear, torch.nn.Linear)
    return linear.weight


def _block(active=True, dtype=BF):
    torch.manual_seed(11)
    block = SWADiTBlock(C, C, H, implementation=ImplementationType.MINIWORLD).to(
        DEV, dtype
    )
    if active:
        with torch.no_grad():
            _modulation_weight(block).normal_(std=0.05)
    return block


def _module_case(a, b, s, seed=5, dtype=BF):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(a * b, s, C, generator=g).to(DEV, dtype)
    c_base = torch.randn(b, s, C, generator=g).to(DEV, dtype)
    angles = torch.randn(b, s, C // H // 2, generator=g).to(DEV) * 3.0
    lengths = torch.tensor([s, s - 9, 7][: a * b] + [s] * max(0, a * b - 3), device=DEV)
    valid = torch.arange(s, device=DEV)[None] < lengths[:, None]
    return x, c_base, build_attention_params(angles.cos(), angles.sin(), valid, a)


def _module_grads(block, fn, x, c, ap, dy):
    block = copy.deepcopy(block)
    x = x.detach().clone().requires_grad_()
    c = c.detach().clone().requires_grad_()
    out = fn(block, x, c, ap)
    out.backward(dy)
    return out.detach(), [x.grad, c.grad, *(p.grad for p in block.parameters())]


@pytest.mark.parametrize("path", ["triton", "cuda"])
@pytest.mark.parametrize("hoisted", [False, True])
def test_the_module_takes_the_fused_block_and_matches_its_per_op_path(path, hoisted):
    """SWADiTBlock(miniworld): the fused block against the same module with `swa_dit_fused=False` (the per-op path,
    FlashAttention windowed attention). ``hoisted`` goes through `forward_hoisted` with the [B, S] conditioning."""
    if _flash_backend(torch.device(DEV)) is None:
        pytest.skip("the per-op path needs a flash backend")
    _use(path)
    a, b, s = 3, 2, 300
    block = _block()
    x, c_base, ap = _module_case(a, b, s)
    c = c_base if hoisted else c_base.repeat(a, 1, 1)
    assert block.fused_refusal(x, c, ap) is None
    fn = (
        (lambda m, x_, c_, ap_: m.forward_hoisted(x_, c_, ap_))
        if hoisted
        else (lambda m, x_, c_, ap_: m(x_, c_, ap_))
    )
    dy = torch.randn(a * b, s, C, device=DEV, dtype=BF)
    out, grads = _module_grads(block, fn, x, c, ap, dy)
    settings.configure(swa_dit_fused=False)
    ref, ref_grads = _module_grads(block, fn, x, c, ap, dy)
    names = ["x", "cond", *(n for n, _ in block.named_parameters())]
    failures = []
    for name, got, want in [
        ("out", out, ref),
        *zip(names, grads, ref_grads, strict=True),
    ]:
        assert got is not None, name
        assert want is not None, name
        frob, _peak = _rel(got, want)
        if frob > 3e-2:
            failures.append(f"{name}: rel_frob={frob:.2e}")
    assert not failures, "\n  ".join(failures)


def test_zero_initialised_gates_keep_the_block_the_identity():
    """adaLN-Zero: with the modulation weight at its zero init the fused block returns its input exactly."""
    _use("triton")
    block = _block(active=False)
    x, c_base, ap = _module_case(2, 1, 200)
    c = c_base.repeat(2, 1, 1)
    assert block.fused_refusal(x, c, ap) is None
    with torch.no_grad():
        torch.testing.assert_close(block(x, c, ap), x, rtol=0, atol=0)


@pytest.mark.skipif(
    settings.current().compile_wrap != "custom_op",
    reason="fullgraph needs the custom_op launches",
)
@pytest.mark.parametrize("path", ["triton", "cuda"])
def test_the_fused_module_compiles_fullgraph(path):
    """Two opaque ops and an autograd.Function: torch.compile(fullgraph=True) traces the module with no break."""
    _use(path)
    block = _block()
    x, c_base, ap = _module_case(3, 2, 200)
    c = c_base.repeat(3, 1, 1)
    dy = torch.randn(6, 200, C, device=DEV, dtype=BF)
    eager, eager_grads = _module_grads(
        block, lambda m, x_, c_, ap_: m(x_, c_, ap_), x, c, ap, dy
    )
    compiled, compiled_grads = _module_grads(
        block,
        lambda m, x_, c_, ap_: torch.compile(m, fullgraph=True)(x_, c_, ap_),
        x,
        c,
        ap,
        dy,
    )
    # Inductor reorders the per-row modulation (silu + matmul, plain torch ops), so a few bf16 outputs round the other
    # way (1 ULP, 7 of 153600 elements measured on H100); compare the output the way the gradients are compared.
    frob, _peak = _rel(compiled, eager)
    assert frob < 1e-2, frob
    for got, want in zip(compiled_grads, eager_grads, strict=True):
        frob, _peak = _rel(got, want)
        assert frob < 1e-2, frob


# ---------------------------------------------------------------------------------------------------------------------
# fp32: the kernels in triton/forward_fp32.py / backward_fp32.py around the shared bf16-operand window attention, and the
# hand-CUDA extension cuda/swa_fp32.cu (every stage; sm_90, while TF32 is allowed).
# ---------------------------------------------------------------------------------------------------------------------
F32 = torch.float32
FP32_PATHS = ("triton", "cuda")


@pytest.fixture
def matmul_precision():
    """Restore the process's fp32 matmul precision after a test that sets it."""
    previous = torch.get_float32_matmul_precision()
    yield torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision(previous)


def _needs_fp32_cuda():
    from miniworld_engine.kernels.swa_dit.cuda.loader import ERRORS, extension, is_sm90

    if not is_sm90():
        pytest.skip("the fp32 CUDA stages are sm_90a kernels")
    if extension("fp32") is None:
        pytest.skip(
            f"the swa_dit fp32 CUDA extension did not build here: {ERRORS.get('fp32')}"
        )


@contextlib.contextmanager
def _fp32_path(path: str):
    """The fp32 block on ``path`` for the duration: "triton" pins the CUDA extension off; "cuda" pins it on under "high"
    matmul precision (TF32 allowed -- the only precision it serves) and checks that it really is what serves."""
    from miniworld_engine.kernels.swa_dit.dispatch import _fp32_cuda

    previous = torch.get_float32_matmul_precision()
    if path == "cuda":
        _needs_fp32_cuda()
        settings.configure(force_triton_kernels=False, swa_dit_fp32_cuda=True)
        torch.set_float32_matmul_precision("high")
        assert _fp32_cuda(C, HIDDEN, HW, torch.device(DEV)) is not None
    else:
        settings.configure(swa_dit_fp32_cuda=False)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize("path", FP32_PATHS)
@pytest.mark.parametrize(
    ("a", "b", "s"), [(48, 1, 257), (3, 2, 333), (1, 3, 300), (2, 1, 300)]
)
def test_the_fp32_block_matches_its_fp32_reference(a, b, s, path):
    """fp32 in, fp32 out, every gradient in its input's dtype, within the band the bf16 attention operands leave.

    The attention core runs on bf16 q / k / v / P / dO exactly as the per-op path's FlashAttention-4 does, and that is
    the whole error budget: measured on an H100, out ~1.3e-4 and the q / conditioning / FFN gradients <= 7e-4 relative
    Frobenius, the attention-side weight gradients (Wqkv, gate, out) ~3-4e-3 -- the same as the per-op fp32 path. The
    bounds are ~5x those measurements. dmod is covered through the conditioning and the adaLN weight; A > 1 and ragged
    seqused (a row shorter than the half window among them) are in the shapes.
    """
    torch.manual_seed(0)
    leaves, cos, sin, seqused = _case(a, b, s, dtype=F32)
    dy = torch.randn(a * b, s, C, device=DEV)
    with _fp32_path(path):
        out, grads = _run(_fused, leaves, dy, cos, sin, seqused, b)
    assert out.dtype == F32
    assert all(g.dtype == F32 for g in grads)
    ref, ref_grads = _run(_reference, leaves, dy, cos, sin, seqused, b)
    attention_side = {"w_qkv", "w_gate", "w_out"}
    failures = []
    for name, got, want in [
        ("out", out, ref),
        *zip(NAMES, grads, ref_grads, strict=True),
    ]:
        assert torch.isfinite(got).all(), name
        frob, peak = _rel(got, want)
        bound = 2e-2 if name in attention_side else 5e-3
        if frob > bound or peak > 2 * bound:
            failures.append(
                f"{name}: rel_frob={frob:.2e} rel_max={peak:.2e} (bound {bound:.0e})"
            )
    assert not failures, f"fp32 {path} A={a} B={b} S={s}:\n  " + "\n  ".join(failures)


@pytest.mark.parametrize(("a", "b", "s"), [(48, 1, 257), (1, 3, 300)])
def test_the_fp32_cuda_stages_agree_with_the_fp32_triton_kernels(
    a, b, s, matmul_precision
):
    """The CUDA extension against the fp32 Triton kernels on the same values, both under "high" (so the weight-gradient
    GEMMs are the same): the same bf16 attention and TF32 projections, the FFN weights split on the CUDA side and tf32x3
    on the Triton side. Measured on an H100: out ~1.7e-4, the q / conditioning / FFN gradients <= 8e-4, the attention-side
    weight gradients ~4.5e-3."""
    _needs_fp32_cuda()
    leaves, cos, sin, seqused = _case(a, b, s, seed=3, dtype=F32)
    dy = torch.randn(a * b, s, C, device=DEV)
    with _fp32_path("cuda"):
        out_c, grads_c = _run(_fused, leaves, dy, cos, sin, seqused, b)
    matmul_precision("high")
    with _fp32_path("triton"):
        out_t, grads_t = _run(_fused, leaves, dy, cos, sin, seqused, b)
    attention_side = {"w_qkv", "w_gate", "w_out"}
    for name, got, want in [
        ("out", out_c, out_t),
        *zip(NAMES, grads_c, grads_t, strict=True),
    ]:
        frob, _peak = _rel(got, want)
        assert frob < (2e-2 if name in attention_side else 5e-3), (name, frob)


def test_the_fp32_cuda_forward_saves_nothing_and_computes_the_same():
    """No gradient recorded -> the inference forward (no saved tensors); the same numbers as the training forward."""
    _needs_fp32_cuda()
    leaves, cos, sin, seqused = _case(3, 2, 333, seed=4, dtype=F32)
    with _fp32_path("cuda"):
        with torch.no_grad():
            inference = _fused(leaves, cos, sin, seqused, 2)
        training, _ = _run(
            _fused, leaves, torch.randn(6, 333, C, device=DEV), cos, sin, seqused, 2
        )
    torch.testing.assert_close(inference, training, rtol=0, atol=0)


def test_the_fp32_cuda_block_replays_under_a_cuda_graph():
    """Forward + backward captured once and replayed: the same output and gradients as the eager step (the modulation
    gradient's red.global order may differ, so the comparison is not bitwise)."""
    _needs_fp32_cuda()
    leaves, cos, sin, seqused = _case(8, 2, 130, seed=6, dtype=F32)
    dy = torch.randn(16, 130, C, device=DEV)
    with _fp32_path("cuda"):
        live = [t.detach().clone().requires_grad_() for t in leaves]

        def step():
            for t in live:
                t.grad = None
            out = _fused(live, cos, sin, seqused, 2)
            out.backward(dy)
            return out

        eager = step().detach().clone()
        eager_grads = [t.grad.detach().clone() for t in live]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = step()
        for _ in range(2):
            graph.replay()
        torch.cuda.synchronize()
        replayed_grads = [t.grad for t in live]
    frob, _peak = _rel(captured, eager)
    assert frob < 1e-6, frob
    for name, got, want in zip(NAMES, replayed_grads, eager_grads, strict=True):
        frob, _peak = _rel(got, want)
        assert frob < 1e-5, (name, frob)


def _reference_module(block, x, c, ap):
    """reference.py on a module's weights, with one modulation row per row of ``c`` (B = c.shape[0])."""
    b = c.shape[0]
    mod = swa_dit_hoist_modulation_reference(c, block.adaln_modulation[1].weight)
    return swa_dit_block_reference(
        x,
        mod,
        ap[0][:b],
        ap[1][:b],
        ap[2],
        block.attn.Wqkv.weight,
        block.attn.gate_proj.weight,
        block.attn.out_proj.weight,
        block.ffn.w_up.weight,
        block.ffn.w_down.weight,
        b,
        half_window=HW,
    )


@pytest.mark.parametrize(
    ("precision", "path"),
    [("highest", "triton"), ("medium", "triton"), ("medium", "cuda"), ("high", "cuda")],
)
def test_the_fp32_fused_block_is_as_accurate_as_the_per_op_path(
    precision, path, matmul_precision
):
    """The fp32 module on the fused block against the same module on its per-op path (``swa_dit_fused=False``: engine
    rmsnorm_adamod, cuBLAS projections, FlashAttention-4 in bf16, triton_swiglu_ffn), each against the fp32 reference.
    ``medium`` is MiniWorld's trainer setting (cuBLAS TF32). The fused error may not exceed the per-op error by more than
    1.5x (plus 1e-4 for tensors whose error is at rounding level). The CUDA extension serves only while TF32 is allowed,
    so it is held to the per-op path at "medium" and "high"."""
    if _flash_backend(torch.device(DEV)) is None:
        pytest.skip("the per-op path needs a flash backend")
    a, b, s = 3, 2, 300
    block = _block(dtype=F32)
    x, c_base, ap = _module_case(a, b, s, dtype=F32)
    assert block.fused_refusal(x, c_base, ap) is None
    dy = torch.randn(a * b, s, C, device=DEV)
    with _fp32_path(path):
        matmul_precision(precision)
        fused, fused_grads = _module_grads(
            block,
            lambda m, x_, c_, ap_: m.forward_hoisted(x_, c_, ap_),
            x,
            c_base,
            ap,
            dy,
        )
    settings.configure(swa_dit_fused=False)
    matmul_precision(precision)
    perop, perop_grads = _module_grads(
        block, lambda m, x_, c_, ap_: m.forward_hoisted(x_, c_, ap_), x, c_base, ap, dy
    )
    matmul_precision("highest")
    ref, ref_grads = _module_grads(block, _reference_module, x, c_base, ap, dy)
    names = ["out", "x", "cond", *(n for n, _ in block.named_parameters())]
    failures = []
    for name, f, p, r in zip(
        names,
        [fused, *fused_grads],
        [perop, *perop_grads],
        [ref, *ref_grads],
        strict=True,
    ):
        assert f.dtype == F32, name
        err_f, err_p = _rel(f, r)[0], _rel(p, r)[0]
        if err_f > 1.5 * err_p + 1e-4:
            failures.append(f"{name}: fused {err_f:.2e} vs per-op {err_p:.2e}")
    assert not failures, f"precision={precision} {path}:\n  " + "\n  ".join(failures)


def test_the_fp32_module_takes_the_fused_block_and_mixed_dtypes_do_not():
    block = _block(dtype=F32)
    x, c_base, ap = _module_case(2, 1, 150, dtype=F32)
    assert block.fused_refusal(x, c_base, ap) is None
    assert "mixed dtypes" in str(block.fused_refusal(x.to(BF), c_base, ap))


@pytest.mark.skipif(
    settings.current().compile_wrap != "custom_op",
    reason="fullgraph needs the custom_op launches",
)
@pytest.mark.parametrize("path", FP32_PATHS)
def test_the_fp32_fused_module_compiles_fullgraph(path):
    block = _block(dtype=F32)
    x, c_base, ap = _module_case(3, 2, 200, dtype=F32)
    c = c_base.repeat(3, 1, 1)
    dy = torch.randn(6, 200, C, device=DEV)
    with _fp32_path(path):
        eager, eager_grads = _module_grads(
            block, lambda m, x_, c_, ap_: m(x_, c_, ap_), x, c, ap, dy
        )
        compiled, compiled_grads = _module_grads(
            block,
            lambda m, x_, c_, ap_: torch.compile(m, fullgraph=True)(x_, c_, ap_),
            x,
            c,
            ap,
            dy,
        )
    frob, _peak = _rel(compiled, eager)
    assert frob < 1e-4, frob
    for got, want in zip(compiled_grads, eager_grads, strict=True):
        assert got.dtype == F32
        frob, _peak = _rel(got, want)
        assert frob < 1e-3, frob


@pytest.mark.skipif(
    settings.current().compile_wrap != "custom_op",
    reason="opcheck needs the custom_op launches",
)
@pytest.mark.parametrize("path", FP32_PATHS)
def test_the_fp32_ops_satisfy_their_contract(path):
    """Schema and fake against the fp32 forward (both save modes) and backward: the fakes carry the per-output dtypes
    (bf16 attention operands, q's dtype elsewhere), which only an fp32 call can tell apart."""
    from miniworld_engine.kernels.swa_dit.dispatch import swa_dit_block_fwd

    leaves, cos, sin, seqused = _case(3, 2, 200, dtype=F32)
    q, c_base, wmod, *weights = (t.detach() for t in leaves)
    mod = swa_dit_hoist_modulation(c_base, wmod)
    cos, sin = cos.reshape(-1, cos.shape[-1]), sin.reshape(-1, sin.shape[-1])
    fwd_op = torch.ops.miniworld_engine.swa_dit_block_fwd.default
    bwd_op = torch.ops.miniworld_engine.swa_dit_block_bwd.default
    with _fp32_path(path):
        for save in (False, True):
            torch.library.opcheck(
                fwd_op,
                (q, mod, cos, sin, seqused, *weights, 2, HW, 1e-7, save),
                test_utils=("test_schema", "test_faketensor"),
            )
        saved = swa_dit_block_fwd(
            q, mod, cos, sin, seqused, *weights, 2, HW, 1e-7, True
        )
        dy = torch.randn_like(q)
        torch.library.opcheck(
            bwd_op,
            (dy, q, mod, cos, sin, seqused, *weights, *saved[1:], 2, HW, 1e-7),
            test_utils=("test_schema", "test_faketensor"),
        )


def test_the_fp32_block_matches_team_gms_unfused_atom_transformer():
    """team-gm's SWAAtomTransformer (block_style esmfold2, fp32, fused path off) is the fp32 path MiniWorld v1.3 runs.
    The fused block, driven the way team-gm's `_forward_fused` drives it, against it: 3 blocks, A=3 x B=2, ragged; every
    gradient within the per-op-vs-reference band measured above. Skipped without team-gm on the path."""
    tg = pytest.importorskip("team_gm.modules.blocks.swa_atom_transformer")
    if _flash_backend(torch.device(DEV)) is None:
        pytest.skip("team-gm's unfused path needs a flash backend")
    torch.manual_seed(13)
    a, b, s = 3, 2, 300
    model = tg.SWAAtomTransformer(
        tg.SWAAtomTransformer.Config(n_block=3, fused_triton=False)
    ).to(DEV)
    with torch.no_grad():
        for blk in model.blocks:
            blk.adaln_modulation[1].weight.normal_(std=0.05)
    x, c_base, ap = _module_case(a, b, s, dtype=F32)
    dy = torch.randn(a * b, s, C, device=DEV)

    def fused(m, x_, c_, ap_):
        cos, sin, seqused = ap_[0][:b], ap_[1][:b], ap_[2]
        for blk in m.blocks:
            mod = swa_dit_hoist_modulation(c_, blk.adaln_modulation[1].weight)
            x_ = swa_dit_block(
                x_,
                mod,
                cos,
                sin,
                seqused,
                blk.attn.Wqkv.weight,
                blk.attn.gate_proj.weight,
                blk.attn.out_proj.weight,
                blk.ffn.w_up.weight,
                blk.ffn.w_down.weight,
                b,
                half_window=HW,
            )
        return x_

    def unfused(m, x_, c_, ap_):
        return m(x_, c_.repeat(a, 1, 1), ap_, c_base=c_)

    got, got_grads = _module_grads(model, fused, x, c_base, ap, dy)
    want, want_grads = _module_grads(model, unfused, x, c_base, ap, dy)
    names = ["out", "x", "cond", *(n for n, _ in model.named_parameters())]
    failures = []
    for name, g, w in zip(names, [got, *got_grads], [want, *want_grads], strict=True):
        frob, _peak = _rel(g, w)
        if frob > (3e-2 if "attn." in name else 1e-2):
            failures.append(f"{name}: rel_frob={frob:.2e}")
    assert not failures, "\n  ".join(failures)


@pytest.mark.parametrize("dtype", [BF, F32], ids=["bf16", "fp32"])
@pytest.mark.parametrize(("a", "b", "s"), [(1, 1, 300), (5, 1, 257), (8, 2, 130)])
def test_the_cuda_block_is_bitwise_repeatable(dtype, a, b, s):
    """Repeated forward + backward runs of the hand-CUDA block (bf16 stages, or the fp32 extension) agree bit for bit in the
    output, dq and every block-weight gradient. The modulation gradient (``c_base`` and ``w_adaln`` gradients) sums the
    augments' contributions with fp32 ``red.global`` where one atom is served by several slabs (A larger than a slab), so it is
    excluded here -- as on the Triton path."""
    leaves, cos, sin, seqused = _case(a, b, s, dtype=dtype)
    dy = torch.randn(a * b, s, C, device=DEV) * 0.01
    ctx = _fp32_path("cuda") if dtype == F32 else contextlib.nullcontext(_use("cuda"))
    with ctx:
        first = _run(_fused, leaves, dy, cos, sin, seqused, b)
        for _ in range(3):
            again = _run(_fused, leaves, dy, cos, sin, seqused, b)
            assert torch.equal(first[0], again[0])
            for name, x, y in zip(NAMES, first[1], again[1], strict=True):
                if name not in ("c_base", "w_adaln"):
                    assert torch.equal(x, y), name


@pytest.mark.parametrize("path", ["cuda"])
def test_a_bf16_modulation_is_widened_for_the_kernels(path):
    """A conditioning shared by the samples is projected under autocast and arrives as a bf16 modulation: the block widens it
    (the kernels take fp32) and hands its gradient back in bf16."""
    _use(path)
    torch.manual_seed(0)
    leaves, cos, sin, seqused = _case(5, 1, 4096)
    q, c_base, wmod, wqkv, wg, wo, wu, wd = [t.detach().clone().requires_grad_() for t in leaves]
    mod = swa_dit_hoist_modulation(c_base, wmod).to(BF).detach().requires_grad_()
    out = swa_dit_block(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, 1, half_window=HW)
    out.backward(torch.randn_like(out))
    assert torch.isfinite(out).all()
    assert mod.grad is not None and mod.grad.dtype is BF and torch.isfinite(mod.grad).all()
