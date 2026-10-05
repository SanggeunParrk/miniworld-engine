"""``ops.augmented_attention_pair_bias`` at the triangle-attention row geometry on A100 -- the ``projected_attention`` token_pair rows (kernels/triangle_attention/cuda/sm80_projected.py: the
generalised hand-CUDA core, head dim 16 / 32, A == L pair rows sharing one pair bias, a per-row key mask), inference and training, against the fp32 PyTorch reference.  The errors are held to
the Triton path's own error in the same regime (bf16 operands, fp32 accumulation)."""

import pytest
import torch

from miniworld_engine import ops
from miniworld_engine.kernels.augmented_attention.reference import (
    augmented_attention_pair_bias_pytorch,
)

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

#: (n_head, d_hidden) of the registry's projected_attention token_pair rows: head dim 16 / 32 / 32 / 32 / 32
GEOMETRIES = [(4, 64), (4, 128), (2, 64), (8, 256), (12, 384)]
IDS = [f"{h}x{hid // h}" for h, hid in GEOMETRIES]
BF = torch.bfloat16


@pytest.fixture(autouse=True)
def ampere():
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("Ampere (sm_80) required")


def _rel(got, want):
    return float((got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-20))


def _inputs(heads, hidden, length, batch=1, mask_kind="none", seed=1):
    """The registry's leaf inputs (head-major ``[A = L, B, H, L, D]``, bias ``[B, H, L, L]``) and a key mask: ``none`` / ``ones`` (all real) / ``random`` (some padded keys, one fully masked row,
    one row with a single key, one key tile of one row fully masked)."""
    d = hidden // heads
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (torch.randn(length, batch, heads, length, d, device="cuda", dtype=BF, generator=g) for _ in range(3))
    bias = (0.5 * torch.randn(batch, heads, length, length, device="cuda", generator=g)).to(BF)
    mask = None
    if mask_kind == "ones":
        mask = torch.ones(length, batch, length, dtype=torch.bool, device="cuda")
    elif mask_kind == "random":
        mask = torch.rand(length, batch, length, device="cuda", generator=g) > 0.15
        mask[0, 0, :] = False
        mask[1, 0, :] = False
        mask[1, 0, 5] = True
        mask[2, 0, 32:64] = False
    return q, k, v, bias, mask


def _reference(q, k, v, bias, mask):
    """The repo's fp32 reference on (A, B, H, L, D) operands (it takes the token-major layout)."""
    out = augmented_attention_pair_bias_pytorch(q.float().transpose(2, 3), k.float().transpose(2, 3), v.float().transpose(2, 3), bias.float().permute(0, 2, 3, 1), mask)
    return out.transpose(2, 3)


def _run(q, k, v, bias, mask, cot, *, grads=True):
    """Output and gradients (q, k, v, bias) of ``sum(out * cot)`` through the op."""
    leaves = [t.detach().clone().requires_grad_(grads) for t in (q, k, v, bias)]
    out = ops.augmented_attention_pair_bias(*leaves, mask)
    if not grads:
        return out.detach(), None
    return out.detach(), torch.autograd.grad((out.float() * cot).sum(), leaves)


def _triton_run(q, k, v, bias, mask, cot, monkeypatch, *, grads=True):
    with monkeypatch.context() as m:
        m.setenv("MINIWORLD_TRIATTN_SM80", "0")
        return _run(q, k, v, bias, mask, cot, grads=grads)


def _serves(q, k, v, bias, mask):
    from miniworld_engine.kernels.triangle_attention.cuda import sm80_projected

    return sm80_projected.serves(q, k, v, bias, mask)


# ------------------------------------------------------------------------------------------------------------------------------------------------- the gate
def test_the_gate_serves_the_registry_geometries_and_declines_the_rest(monkeypatch):
    for heads, hidden in GEOMETRIES:
        q, k, v, bias, mask = _inputs(heads, hidden, 128, mask_kind="ones")
        assert _serves(q, k, v, bias, mask)
        assert _serves(q, k, v, bias, None)
    heads, hidden = 4, 128
    q, k, v, bias, mask = _inputs(heads, hidden, 128, mask_kind="ones")
    assert not _serves(q.float(), k.float(), v.float(), bias.float(), mask)                       # fp32 operands
    assert not _serves(q, k, v, bias.float(), mask)                                              # a bias in fp32
    assert not _serves(q, k, v, bias, mask.to(torch.uint8))                                      # the mask's dtype
    assert not _serves(q, k, v, bias, mask[:, :, :64])                                           # the mask's shape
    assert not _serves(q[:64], k[:64], v[:64], bias, mask[:64])                                  # A != L (64 pair rows of 128 tokens)
    assert not _serves(q, k, v, bias[:, :, :64], mask)                                           # the bias' shape
    q2, k2, v2, b2, m2 = _inputs(4, 192, 128)                                                    # head dim 48
    assert not _serves(q2, k2, v2, b2, m2)
    q3, k3, v3, b3, m3 = _inputs(4, 128, 192)                                                    # L is not a multiple of 128
    assert not _serves(q3, k3, v3, b3, m3)
    assert not _serves(q.cpu(), k.cpu(), v.cpu(), bias.cpu(), None)
    monkeypatch.setenv("MINIWORLD_TRIATTN_SM80", "0")
    assert not _serves(q, k, v, bias, mask)                                                      # the switch


def test_the_engine_backend_can_force_triton():
    from miniworld_engine import settings

    q, k, v, bias, mask = _inputs(4, 128, 128)
    assert _serves(q, k, v, bias, mask)
    previous = settings.current().engine_backend
    settings.configure(engine_backend="triton")
    try:
        assert not _serves(q, k, v, bias, mask)
    finally:
        settings.configure(engine_backend=previous)
    assert _serves(q, k, v, bias, mask)


def test_the_memory_efficient_backward_keeps_the_triton_kernels(monkeypatch):
    from miniworld_engine.kernels.triangle_attention.cuda import sm80_projected

    q, k, v, bias, mask = _inputs(4, 128, 128)
    calls = []
    real = sm80_projected.attention
    monkeypatch.setattr(sm80_projected, "attention", lambda *a: calls.append(1) or real(*a))
    ops.augmented_attention_pair_bias(q, k, v, bias, mask, kernel_type="memory_efficient")
    assert not calls
    ops.augmented_attention_pair_bias(q, k, v, bias, mask)
    assert calls == [1]


# --------------------------------------------------------------------------------------------------------------------------------------------- inference
@pytest.mark.parametrize(("heads", "hidden"), GEOMETRIES, ids=IDS)
@pytest.mark.parametrize("mask_kind", ["none", "ones", "random"])
@pytest.mark.parametrize("length", [128, 256])
def test_inference_is_no_less_accurate_than_the_triton_path(heads, hidden, mask_kind, length, monkeypatch):
    q, k, v, bias, mask = _inputs(heads, hidden, length, mask_kind=mask_kind)
    assert _serves(q, k, v, bias, mask)
    got, _ = _run(q, k, v, bias, mask, None, grads=False)
    want = _reference(q, k, v, bias, mask)
    assert got.dtype is BF
    assert got.shape == q.shape
    mine = _rel(got, want)
    base = _rel(_triton_run(q, k, v, bias, mask, None, monkeypatch, grads=False)[0], want) if length == 128 else 0.0     # the Triton path is compared at L = 128 (its autotuning is slow)
    assert mine <= max(1.25 * base, 4e-3), f"cuda {mine:.3e} vs triton path {base:.3e}"
    if mask is not None:
        dead = ~mask.any(dim=-1)                                                             # a pair row with no real key: zero output (the reference's contract)
        assert dead.any() == (mask_kind == "random")
        assert (got.float().abs().amax(dim=(2, 3, 4)) == 0)[dead].all()


@pytest.mark.parametrize("heads_hidden", [(4, 64), (4, 128)], ids=["4x16", "4x32"])
def test_a_batch_of_two_does_not_mix_its_elements(heads_hidden):
    heads, hidden = heads_hidden
    q, k, v, bias, mask = _inputs(heads, hidden, 128, batch=2, mask_kind="random")
    assert _serves(q, k, v, bias, mask)
    got, _ = _run(q, k, v, bias, mask, None, grads=False)
    want = _reference(q, k, v, bias, mask)
    assert _rel(got, want) <= 4e-3
    for b in range(2):
        alone, _ = _run(q[:, b:b + 1].contiguous(), k[:, b:b + 1].contiguous(), v[:, b:b + 1].contiguous(), bias[b:b + 1].contiguous(), mask[:, b:b + 1].contiguous(), None, grads=False)
        assert torch.equal(got[:, b:b + 1], alone)


@pytest.mark.parametrize("heads_hidden", [(4, 64), (8, 256)], ids=["4x16", "8x32"])
def test_token_major_views_are_read_in_place_and_give_the_same_bits(heads_hidden):
    """The model's projection outputs are token-major ``[A, B, L, H D]`` tensors viewed head-major: the core reads those strides directly (no transposing copy)."""
    heads, hidden = heads_hidden
    length = 128
    q, k, v, bias, mask = _inputs(heads, hidden, length, mask_kind="random")
    views = [t.permute(0, 1, 3, 2, 4).contiguous().permute(0, 1, 3, 2, 4) for t in (q, k, v)]      # head-major shape, token-major memory
    assert not views[0].is_contiguous()
    got, _ = _run(*views, bias, mask, None, grads=False)
    want, _ = _run(q, k, v, bias, mask, None, grads=False)
    assert torch.equal(got, want)
    assert got.stride() == views[0].stride()                                                      # the output keeps the query's memory layout


def test_an_unaligned_or_strided_operand_is_copied_and_still_right():
    q, k, v, bias, mask = _inputs(4, 128, 128, mask_kind="random")
    wide = torch.randn(128, 1, 4, 128, 40, device="cuda", dtype=BF)
    qs = wide[..., 4:36]                                                                           # channel stride 1, row strides not a multiple of 8 elements: copied
    qs.copy_(q)
    got, _ = _run(qs, k, v, bias, mask, None, grads=False)
    want, _ = _run(q, k, v, bias, mask, None, grads=False)
    assert torch.equal(got, want)
    qt = q.transpose(3, 4).contiguous().transpose(3, 4)                                           # the channel stride is not 1: copied
    got, _ = _run(qt, k, v, bias, mask, None, grads=False)
    assert torch.equal(got, want)


def test_a_bias_given_as_a_view_of_a_token_major_tensor():
    """``ops.layer_norm_linear`` returns the bias token-major ``[B, L, L, H]``: the caller passes its ``[B, H, L, L]`` permuted view."""
    q, k, v, bias, mask = _inputs(4, 128, 128, mask_kind="ones")
    tm = bias.permute(0, 2, 3, 1).contiguous()
    got, _ = _run(q, k, v, tm.permute(0, 3, 1, 2), mask, None, grads=False)
    want, _ = _run(q, k, v, bias, mask, None, grads=False)
    assert torch.equal(got, want)


@pytest.mark.parametrize("heads_hidden", [(4, 64), (4, 128)], ids=["4x16", "4x32"])
def test_all_valid_key_tiles_take_the_unmasked_arithmetic_with_the_same_bits(heads_hidden):
    """The masked kernels skip a key tile's masking when all its keys are real: an all-True mask gives exactly the unmasked result, forward and backward."""
    heads, hidden = heads_hidden
    q, k, v, bias, _ = _inputs(heads, hidden, 128)
    ones = torch.ones(128, 1, 128, dtype=torch.bool, device="cuda")
    cot = torch.randn(q.shape, device="cuda")
    a, ga = _run(q, k, v, bias, None, cot)
    b, gb = _run(q, k, v, bias, ones, cot)
    assert torch.equal(a, b)
    for x, y in zip(ga, gb, strict=True):
        assert torch.equal(x, y)


# ----------------------------------------------------------------------------------------------------------------------------------------------- training
@pytest.mark.parametrize(("heads", "hidden"), GEOMETRIES, ids=IDS)
@pytest.mark.parametrize("mask_kind", ["none", "random"])
@pytest.mark.parametrize("length", [128, 256])
def test_training_matches_the_fp32_reference_as_well_as_the_triton_path(heads, hidden, mask_kind, length, monkeypatch):
    q, k, v, bias, mask = _inputs(heads, hidden, length, mask_kind=mask_kind)
    cot = torch.randn(q.shape, device="cuda")
    out, grads = _run(q, k, v, bias, mask, cot)
    leaves = [t.detach().float().requires_grad_() for t in (q, k, v, bias)]
    out_r = _reference(*leaves, mask)
    grads_r = torch.autograd.grad((out_r * cot).sum(), leaves)
    if length == 128:                                                                                  # the Triton path is compared at L = 128 (its autotuning is slow)
        out_t, grads_t = _triton_run(q, k, v, bias, mask, cot, monkeypatch)
        base_out, base_grads = _rel(out_t, out_r), [_rel(gt, gr) for gt, gr in zip(grads_t, grads_r, strict=True)]
    else:
        base_out, base_grads = 0.0, [0.0] * 4
    assert _rel(out, out_r) <= max(1.25 * base_out, 4e-3)
    for name, g, gr, base in zip("qkvb", grads, grads_r, base_grads, strict=True):
        mine = _rel(g, gr)
        assert mine <= max(1.25 * base, 8e-3), f"d{name}: cuda {mine:.3e} vs triton path {base:.3e}"
        assert g.dtype is BF
        assert g.shape == gr.shape


def test_training_calls_one_forward_and_one_backward_op(monkeypatch):
    from miniworld_engine.kernels.triangle_attention.cuda import sm80_projected as sp

    q, k, v, bias, mask = _inputs(4, 128, 128, mask_kind="random")
    cot = torch.randn(q.shape, device="cuda")
    calls = []
    for name in ("_forward", "_backward"):
        real = getattr(sp, name)
        monkeypatch.setattr(sp, name, lambda *a, _r=real, _n=name: calls.append(_n) or _r(*a))
    _run(q, k, v, bias, mask, cot)
    assert calls == ["_forward", "_backward"], calls


def test_training_replays_bit_identically_and_gradients_follow_the_inputs_layout():
    q, k, v, bias, mask = _inputs(4, 64, 128, mask_kind="random")
    cot = torch.randn(q.shape, device="cuda")
    va = [t.permute(0, 1, 3, 2, 4).contiguous().permute(0, 1, 3, 2, 4) for t in (q, k, v)]
    out_a, ga = _run(*va, bias, mask, cot)
    out_b, gb = _run(*va, bias, mask, cot)
    assert torch.equal(out_a, out_b)
    for a, b in zip(ga, gb, strict=True):
        assert torch.equal(a, b)
    assert ga[0].stride() == va[0].stride()


@pytest.mark.parametrize("wanted", ["q", "bias", "none"])
def test_only_some_operands_want_gradients(wanted):
    q, k, v, bias, mask = _inputs(4, 128, 128, mask_kind="ones")
    cot = torch.randn(q.shape, device="cuda")
    leaves = {name: t.detach().clone().requires_grad_(name == wanted) for name, t in zip(("q", "k", "v", "bias"), (q, k, v, bias), strict=True)}
    out = ops.augmented_attention_pair_bias(*leaves.values(), mask)
    if wanted == "none":
        assert not out.requires_grad
        return
    g, = torch.autograd.grad((out.float() * cot).sum(), [leaves[wanted]])
    assert torch.isfinite(g.float()).all()
    full = _run(q, k, v, bias, mask, cot)[1]
    assert torch.equal(g, full[0 if wanted == "q" else 3])


# --------------------------------------------------------------------------------------------------------------------------- compile and CUDA graphs
@pytest.mark.parametrize("heads_hidden", [(4, 64), (8, 256)], ids=["4x16", "8x32"])
def test_the_compiled_op_matches_eager_in_inference_and_training(heads_hidden):
    heads, hidden = heads_hidden
    q, k, v, bias, mask = _inputs(heads, hidden, 128, mask_kind="random")
    cot = torch.randn(q.shape, device="cuda")
    compiled = torch.compile(lambda a, b, c, d, m: ops.augmented_attention_pair_bias(a, b, c, d, m), fullgraph=True)
    with torch.no_grad():
        eager = ops.augmented_attention_pair_bias(q, k, v, bias, mask)
        got = compiled(q, k, v, bias, mask)
    assert torch.equal(got, eager)
    leaves = [t.detach().clone().requires_grad_() for t in (q, k, v, bias)]
    ge = torch.autograd.grad((ops.augmented_attention_pair_bias(*leaves, mask).float() * cot).sum(), leaves)
    gc = torch.autograd.grad((compiled(*leaves, mask).float() * cot).sum(), leaves)
    for a, b in zip(ge, gc, strict=True):
        assert torch.equal(a, b)


def test_a_cuda_graph_of_the_inference_call_replays_the_eager_result():
    q, k, v, bias, mask = _inputs(4, 128, 128, mask_kind="random")
    with torch.no_grad():
        for _ in range(2):
            ops.augmented_attention_pair_bias(q, k, v, bias, mask)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = ops.augmented_attention_pair_bias(q, k, v, bias, mask)
        q.copy_(torch.randn_like(q))
        want = ops.augmented_attention_pair_bias(q, k, v, bias, mask)
        graph.replay()
    assert torch.equal(out, want)


def test_a_cuda_graph_of_a_training_step_replays_the_eager_gradients():
    q, k, v, bias, mask = _inputs(4, 64, 128, mask_kind="random")
    cot = torch.randn(q.shape, device="cuda")
    leaves = [t.detach().clone().requires_grad_() for t in (q, k, v, bias)]

    def step():
        return torch.autograd.grad((ops.augmented_attention_pair_bias(*leaves, mask).float() * cot).sum(), leaves)

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static = step()
    leaves[0].data.copy_(torch.randn_like(q))
    want = step()
    graph.replay()
    for got, ref in zip(static, want, strict=True):
        assert torch.equal(got, ref)


def test_the_env_switch_leaves_the_triton_numbers_untouched(monkeypatch):
    q, k, v, bias, mask = _inputs(4, 128, 128, mask_kind="random")
    a, _ = _triton_run(q, k, v, bias, mask, None, monkeypatch, grads=False)
    with monkeypatch.context() as m:
        m.setenv("MINIWORLD_TRIATTN_SM80", "0")
        b, _ = _run(q, k, v, bias, mask, None, grads=False)
    c, _ = _run(q, k, v, bias, mask, None, grads=False)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)                                                                   # the CUDA path is a different implementation: not the same bits
