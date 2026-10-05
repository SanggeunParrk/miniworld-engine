"""The A100 (sm_80) hand-CUDA triangle-attention core matches the fp32 reference at least as closely as the bf16 PyTorch statements do (with and
without a key mask), saves the log-sum-exp the backward needs, and the gate only takes what it is built for; the front (input LayerNorm and the
five projections in one pass) does the same."""

import pytest
import torch

pytestmark = pytest.mark.gpu

CUDA = torch.cuda.is_available()
AMPERE = CUDA and torch.cuda.get_device_capability() == (8, 0)
needs_ampere = pytest.mark.skipif(not AMPERE, reason="the sm80 triangle attention is sm_80 only")

H, D = 4, 32
LOG2E = 1.4426950408889634
EPS = 1e-5


def _inputs(length, masked, seed=7):
    """q / k / v as the module hands them to the attention op: [B, H, L, L2, D] views of the token-major projections; the bias with
    the module's masked_fill."""
    torch.manual_seed(seed)
    tm = [torch.randn(1, length, length, H * D, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    q, k, v = (t.view(1, length, length, H, D).permute(0, 3, 1, 2, 4) for t in tm)
    bias = (0.5 * torch.randn(1, length, length, H, device="cuda")).to(torch.bfloat16).permute(0, 3, 1, 2)
    if masked:
        mask = torch.rand(1, length, device="cuda") > 0.1
        bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(torch.bfloat16).min)
    return q, k, v, bias


def _logits(q, k, bias, dtype):
    q, k, bias = (t.to(dtype) for t in (q, k, bias))
    return torch.einsum("bhijd,bhikd->bhijk", q * (D ** -0.5), k) + bias[:, :, None, :, :]


def _reference(q, k, v, bias, dtype):
    attn = torch.softmax(_logits(q, k, bias, dtype), dim=-1)
    return torch.einsum("bhijk,bhikd->bhijd", attn, v.to(dtype))


def _rel(got, want):
    return float((got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-20))


@needs_ampere
@pytest.mark.parametrize("length", [128, 256, 384])
@pytest.mark.parametrize("masked", [False, True])
def test_attention_is_no_less_accurate_than_the_bf16_statements(length, masked):
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    q, k, v, bias = _inputs(length, masked)
    assert sm80.supports(q, k, v, bias)
    out, lse = sm80.attention(q, k, v, bias, save_lse=True)
    assert lse is not None
    assert out.shape == q.shape
    assert out.dtype is torch.bfloat16
    assert lse.shape == (1, H, length, length)
    ref32 = _reference(q, k, v, bias, torch.float32)
    ref16 = _reference(q, k, v, bias, torch.bfloat16)
    mine, base = _rel(out, ref32), _rel(ref16, ref32)
    assert mine <= max(1.25 * base, 3e-3), f"sm80 {mine:.3e} vs bf16 statements {base:.3e}"
    want_lse = torch.logsumexp(_logits(q, k, bias, torch.float32), dim=-1) * LOG2E        # [B, H, i, j] in the base-2 scaled domain
    torch.testing.assert_close(lse, want_lse, rtol=1e-4, atol=1e-3)


@needs_ampere
def test_the_output_is_a_view_the_module_can_use_without_a_copy():
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    q, k, v, bias = _inputs(128, False)
    out, _ = sm80.attention(q, k, v, bias)
    flat = out.permute(0, 2, 3, 1, 4).reshape(1, 128, 128, H * D)      # the module's rearrange "B H L L2 D -> B L L2 (H D)"
    assert flat.data_ptr() == out.data_ptr()
    assert flat.is_contiguous()


@needs_ampere
def test_replay_is_bit_identical():
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    q, k, v, bias = _inputs(256, True)
    a, _ = sm80.attention(q, k, v, bias)
    b, _ = sm80.attention(q, k, v, bias)
    assert torch.equal(a, b)


@needs_ampere
def test_a_fully_masked_key_set_gives_a_finite_zero_output():
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    q, k, v, bias = _inputs(128, False)
    bias = bias.masked_fill(torch.ones(1, 1, 1, 128, dtype=torch.bool, device="cuda"), torch.finfo(torch.bfloat16).min)
    out, lse = sm80.attention(q, k, v, bias, save_lse=True)
    assert lse is not None
    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(lse).all()
    assert float(out.float().abs().max()) == 0.0


@needs_ampere
@pytest.mark.parametrize(("length", "heads"), [(512, 1), (512, 3), (768, 4)])
def test_large_core_preserves_batch_head_and_pair_row_indices(length, heads):
    """One valid key gives an exact oracle for every CTA in the large-grid schedule."""
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    torch.manual_seed(117)
    batch = 2
    tm = [torch.randn(batch, length, length, heads * D, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    q, k, v = (t.view(batch, length, length, heads, D).permute(0, 3, 1, 2, 4) for t in tm)
    bias = torch.full((batch, heads, length, length), torch.finfo(torch.bfloat16).min, device="cuda", dtype=torch.bfloat16)
    key = length - 17
    bias[..., key] = 0
    got, lse = sm80.attention(q, k, v, bias, save_lse=True)
    torch.testing.assert_close(got, v[..., key:key + 1, :].expand_as(got), rtol=0, atol=0)
    assert torch.isfinite(lse).all()


@needs_ampere
@pytest.mark.parametrize("length", [768, 1152])
@pytest.mark.parametrize("valid_keys", [0, 1, 47, 48, 49, 658, "all"])
def test_compact_keys_match_dense_core_for_different_batch_masks(valid_keys, length):
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    torch.manual_seed(119)
    batch, heads = 2, 3
    valid_keys = length if valid_keys == "all" else valid_keys
    tm = [torch.randn(batch, length, length, heads * D, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    q, k, v = (t.view(batch, length, length, heads, D).permute(0, 3, 1, 2, 4) for t in tm)
    mask = torch.zeros(batch, length, device="cuda", dtype=torch.bool)
    for b, n in enumerate((valid_keys, valid_keys // 2)):
        mask[b, torch.randperm(length, device="cuda")[:n]] = True
    bias = torch.randn(batch, heads, length, length, device="cuda", dtype=torch.bfloat16)
    bias.masked_fill_(~mask[:, None, None, :], torch.finfo(torch.bfloat16).min)
    want, _ = sm80.attention(q, k, v, bias)
    got, _ = sm80.attention(q, k, v, bias, compact_key_mask=mask)
    assert torch.isfinite(got).all()
    assert _rel(got, want) < .003
    if valid_keys <= 1:
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@needs_ampere
def test_compact_mask_changes_are_observed_by_graph_replay():
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    q, k, v, unmasked_bias = _inputs(768, False)
    mask = torch.ones(1, 768, device="cuda", dtype=torch.bool)
    bias = unmasked_bias.contiguous().clone()

    def call():
        return sm80.attention(q, k, v, bias, compact_key_mask=mask)[0]

    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call()
    for keep in (1, 658, 0, 768):
        mask.zero_()
        mask[:, :keep] = True
        bias.copy_(unmasked_bias).masked_fill_(~mask[:, None, None, :], torch.finfo(torch.bfloat16).min)
        graph.replay()
        want = call()
        torch.testing.assert_close(out, want, rtol=0, atol=0)


@needs_ampere
@pytest.mark.parametrize("starting", [False, True])
def test_compiled_large_masked_module_matches_eager(starting):
    from miniworld_engine.modules import ImplementationType, TriangleAttention

    torch.manual_seed(123)
    model = TriangleAttention(128, 4, starting=starting, p_drop=0, implementation=ImplementationType.MINIWORLD).cuda().bfloat16().eval()
    with torch.no_grad():
        model.to_out.weight.normal_(std=128 ** -.5)
        pair = torch.randn(1, 768, 768, 128, device="cuda", dtype=torch.bfloat16)
        mask = torch.ones(1, 768, device="cuda", dtype=torch.bool)
        mask[:, ::7] = False
        eager = model(pair, mask)
        compiled = torch.compile(model, fullgraph=True)
        try:
            torch.testing.assert_close(compiled(pair, mask), eager, rtol=0, atol=0)
        finally:
            torch._dynamo.reset()


@needs_ampere
def test_the_core_reads_the_row_strided_views_of_the_fronts_buffer():
    """q / k / v as the slices of one [B, L, L, 512] q | k | v | g buffer (row stride 512), as the front leaves them."""
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    length = 128
    q, k, v, bias = _inputs(length, True)
    buf = torch.randn(1, length, length, 512, device="cuda", dtype=torch.bfloat16)
    for i, t in enumerate((q, k, v)):
        buf[..., 128 * i:128 * (i + 1)] = t.permute(0, 2, 3, 1, 4).reshape(1, length, length, H * D)
    qs, ks, vs, _ = sm80.qkv_views(buf)
    assert sm80.supports(qs, ks, vs, bias)
    got, _ = sm80.attention(qs, ks, vs, bias)
    want, _ = sm80.attention(q, k, v, bias)
    assert torch.equal(got, want)


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
def test_gate_rejects_everything_it_is_not_built_for():
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    q, k, v, bias = _inputs(128, False)
    assert sm80.supports(q, k, v, bias) is AMPERE
    assert not sm80.supports(q.float(), k, v, bias)
    assert not sm80.supports(q, k, v, bias.float())
    q2, k2, v2, bias2 = _inputs(192, False)                                   # L % 128
    assert not sm80.supports(q2, k2, v2, bias2)
    tm = torch.randn(1, 128, 128, 2 * D, device="cuda", dtype=torch.bfloat16)  # a head dim other than 32 / a non-token-major view
    odd = tm.view(1, 128, 128, 2, D).permute(0, 3, 1, 2, 4).transpose(2, 3)
    assert not sm80.supports(odd, odd, odd, torch.zeros(1, 2, 128, 128, device="cuda", dtype=torch.bfloat16))
    assert not sm80.supports(q.cpu(), k.cpu(), v.cpu(), bias.cpu())


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
def test_env_switch_turns_the_gate_off(monkeypatch):
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    monkeypatch.setenv("MINIWORLD_TRIATTN_SM80", "0")
    assert not sm80.supports(*_inputs(128, False))


def _module(starting, seed=5):
    from miniworld_engine.modules.exceptions import ImplementationType
    from miniworld_engine.modules.triangle_attention import TriangleAttention

    torch.manual_seed(seed)
    module = TriangleAttention(128, 4, starting=starting, implementation=ImplementationType.MINIWORLD).cuda().bfloat16()
    with torch.no_grad():                        # the zero-initialised output projection would make the update exactly zero
        for name, t in module.named_parameters():
            if t.ndim >= 2:
                t.normal_(std=t.shape[-1] ** -0.5)
            elif "weight" in name:
                t.copy_(1 + 0.1 * torch.randn_like(t))
            else:
                t.normal_(std=0.05)
    return module.eval()


@needs_ampere
@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize("length", [128, 256])
def test_the_core_alone_serves_the_triton_backend_when_the_whole_module_is_off_the_path(starting, length, monkeypatch):
    from miniworld_engine.integrations import triattn_sm80
    from miniworld_engine.kernels.triangle_attention.cuda import sm80
    from miniworld_engine.modules.exceptions import ImplementationType
    from miniworld_engine.modules.triangle_attention import TriangleAttention

    module = _module(starting)
    reference = TriangleAttention(128, 4, starting=starting, implementation=ImplementationType.PYTORCH).cuda().float().eval()
    reference.load_state_dict({k: v.float() for k, v in module.state_dict().items()})
    torch.manual_seed(3)
    pair = torch.randn(1, length, length, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, length, device="cuda") > 0.1
    monkeypatch.setattr(triattn_sm80, "serves_module", lambda *a, **k: False)     # the whole-module path would take the module first
    from miniworld_engine.integrations import a100_families
    monkeypatch.setattr(a100_families, "serves", lambda *a, **k: False)            # nor the general CUDA composition: this is the core-alone path
    calls = []
    original = sm80.attention
    monkeypatch.setattr(sm80, "attention", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    with torch.no_grad():
        got = module(pair, mask)
        assert calls, "the sm80 attention core was not entered"
        module._sm80_cuda = False
        triton = module(pair, mask)
        want = reference(pair.float(), mask)
    mine, base = _rel(got, want), _rel(triton, want)
    assert mine <= max(1.25 * base, 3e-3), f"cuda core {mine:.3e} vs triton path {base:.3e}"


@needs_ampere
def test_training_keeps_the_triton_path():
    """The backward is not written yet: with autograd on the module must not take the CUDA core."""
    from miniworld_engine.integrations import triattn_sm80

    module = _module(True)
    q, k, v, bias = _inputs(128, False)
    with torch.no_grad():
        assert triattn_sm80.serves(module, q, k, v, bias)
    assert not triattn_sm80.serves(module, q, k, v, bias)          # grad enabled


# ------------------------------------------------------------------------------------------------------------------- the front
def _front_inputs(length, batch=1, seed=11, ln_dtype=torch.float32):
    """A pair stack with a non-zero mean and a spread of variances, the q / k / v / g and bias weights (as one list), the LayerNorm
    parameters and a per-batch key mask."""
    torch.manual_seed(seed)
    pair = (2.0 * torch.randn(batch, length, length, 128, device="cuda") + 0.7).to(torch.bfloat16)
    weights = [(torch.randn(128, 128, device="cuda") * 128 ** -0.5).to(torch.bfloat16) for _ in range(4)]
    weights.append((0.2 * torch.randn(4, 128, device="cuda")).to(torch.bfloat16))
    ln_w = (1 + 0.1 * torch.randn(128, device="cuda")).to(ln_dtype)
    ln_b = (0.05 * torch.randn(128, device="cuda")).to(ln_dtype)
    mask = torch.rand(batch, length, device="cuda") > 0.1
    return pair, weights, ln_w, ln_b, mask


def _front_reference(pair, weights, ln_w, ln_b, mask, dtype, transposed):
    """LayerNorm (statistics in fp32) then the five bias-free projections, in ``dtype`` operands: the module's statements."""
    x = pair.transpose(1, 2) if transposed else pair
    xn = torch.nn.functional.layer_norm(x.to(dtype), (128,), ln_w.to(dtype), ln_b.to(dtype), EPS)
    qkvg = torch.nn.functional.linear(xn, torch.cat(weights[:4]).to(dtype))
    bias = torch.nn.functional.linear(xn, weights[4].to(dtype)).permute(0, 3, 1, 2)
    return qkvg, bias.masked_fill(~mask[:, None, None, :], torch.finfo(torch.bfloat16).min)


@needs_ampere
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize(("batch", "length"), [(1, 128), (1, 256), (2, 128)])
@pytest.mark.parametrize("ln_dtype", [torch.float32, torch.bfloat16])
def test_the_front_is_no_less_accurate_than_the_bf16_statements(transposed, batch, length, ln_dtype):
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    pair, weights, ln_w, ln_b, mask = _front_inputs(length, batch, ln_dtype=ln_dtype)
    assert sm80.supports_front(pair, weights, ln_w, ln_b, mask)
    qkvg, bias, stats = sm80.front(pair, weights, ln_w, ln_b, EPS, mask, transposed=transposed, save_stats=True)
    assert stats is not None
    assert qkvg.shape == (batch, length, length, 512)
    assert bias.shape == (batch, 4, length, length)
    assert qkvg.dtype is torch.bfloat16
    assert bias.dtype is torch.bfloat16
    want32, bias32 = _front_reference(pair, weights, ln_w, ln_b, mask, torch.float32, transposed)
    want16, bias16 = _front_reference(pair, weights, ln_w, ln_b, mask, torch.bfloat16, transposed)
    mine, base = _rel(qkvg, want32), _rel(want16, want32)
    assert mine <= max(1.25 * base, 3e-3), f"qkvg {mine:.3e} vs bf16 statements {base:.3e}"
    keep = mask[:, None, None, :].expand_as(bias)
    mine, base = _rel(bias[keep], bias32[keep]), _rel(bias16[keep], bias32[keep])
    assert mine <= max(1.25 * base, 3e-3), f"bias {mine:.3e} vs bf16 statements {base:.3e}"
    assert (bias[~keep] == torch.finfo(torch.bfloat16).min).all()
    x32 = (pair.transpose(1, 2) if transposed else pair).float().reshape(-1, 128)
    want_stats = torch.stack([x32.mean(-1), (x32.var(-1, unbiased=False) + EPS).rsqrt()], dim=-1)
    torch.testing.assert_close(stats, want_stats, rtol=2e-5, atol=2e-6)


@needs_ampere
def test_the_ending_node_reads_the_transposed_pair_without_a_copy():
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    pair, weights, ln_w, ln_b, mask = _front_inputs(256, 2)
    got = sm80.front(pair, weights, ln_w, ln_b, EPS, mask, transposed=True)
    want = sm80.front(pair.transpose(1, 2).contiguous(), weights, ln_w, ln_b, EPS, mask)
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])


@needs_ampere
def test_the_front_without_a_mask_keeps_every_key_and_replays_bit_identically():
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    pair, weights, ln_w, ln_b, _ = _front_inputs(128)
    qkvg, bias, stats = sm80.front(pair, weights, ln_w, ln_b, EPS)
    again = sm80.front(pair, weights, ln_w, ln_b, EPS)
    assert torch.equal(qkvg, again[0])
    assert torch.equal(bias, again[1])
    assert stats is None
    assert float(bias.float().abs().max()) < 1e4


@needs_ampere
def test_the_packed_weights_fold_the_layernorm_affine():
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    _, weights, ln_w, ln_b, _ = _front_inputs(128)
    wp, bvec = sm80._pack_front(sm80._ext(), weights, ln_w, ln_b)
    w = torch.cat(weights).float()
    # the packed rows of a block of 64 channels are laid out so that the accumulator pair of thread q4 in the 4 n tiles of a group is 8 consecutive
    # channels (``f1_channel``): row s holds channel 32 (s >> 5 & 1) + 8 (s >> 1 & 3) + 2 (s >> 3 & 3) + (s & 1) of its block
    s = torch.arange(512, device="cuda")
    channel = (s & ~63) + 32 * ((s >> 5) & 1) + 8 * ((s >> 1) & 3) + 2 * ((s >> 3) & 3) + (s & 1)
    assert torch.equal(channel.sort().values, s)
    row_of = torch.cat([channel, torch.arange(512, 520, device="cuda")])
    want_w = torch.zeros(520, 128, device="cuda")
    want_w[:516] = (w * ln_w.float())[row_of[:516]]
    want_b = torch.zeros(520, device="cuda")
    want_b[:516] = w @ ln_b.float()                                # the shift is indexed by the channel itself
    assert wp.shape == (520, 128)
    assert torch.equal(wp.float(), want_w.to(torch.bfloat16).float())
    torch.testing.assert_close(bvec, want_b, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
def test_the_front_gate_rejects_everything_it_is_not_built_for():
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    pair, weights, ln_w, ln_b, mask = _front_inputs(128)
    assert sm80.supports_front(pair, weights, ln_w, ln_b, mask) is AMPERE
    assert not sm80.supports_front(pair.float(), weights, ln_w, ln_b)
    assert not sm80.supports_front(pair[:, :, :96], weights, ln_w, ln_b)                                    # not square (and not contiguous)
    odd_length = torch.empty(1, 192, 192, 128, device="cuda", dtype=torch.bfloat16)                         # L % 128
    assert not sm80.supports_front(odd_length, weights, ln_w, ln_b)
    narrow = torch.empty(1, 128, 128, 64, device="cuda", dtype=torch.bfloat16)                              # width
    assert not sm80.supports_front(narrow, weights, ln_w, ln_b)
    assert not sm80.supports_front(pair, [w.float() for w in weights], ln_w, ln_b)
    assert not sm80.supports_front(pair, [*weights[:4], weights[4][:2]], ln_w, ln_b)
    assert not sm80.supports_front(pair, weights[:4], ln_w, ln_b)
    assert not sm80.supports_front(pair, weights, ln_w, ln_b, mask.to(torch.uint8))
    assert not sm80.supports_front(pair, weights, ln_w, ln_b, mask[:, :64])
    assert not sm80.supports_front(pair.transpose(1, 2), weights, ln_w, ln_b)                               # not contiguous


# -------------------------------------------------------------------------------------------------------------------- the back
def _back_inputs(length, batch=1, seed=13):
    """The attention output, the front's buffer (its gate columns), the output projection and the residual."""
    torch.manual_seed(seed)
    o = torch.randn(batch, length, length, 128, device="cuda").to(torch.bfloat16)
    qkvg = (2.0 * torch.randn(batch, length, length, 512, device="cuda")).to(torch.bfloat16)
    wo = (torch.randn(128, 128, device="cuda") * 128 ** -0.5).to(torch.bfloat16)
    res = (2.0 * torch.randn(batch, length, length, 128, device="cuda") + 0.3).to(torch.bfloat16)
    return o, qkvg, wo, res


def _back_reference(o, qkvg, wo, res, dtype, transposed):
    """``res + to_out(sigmoid(g) * o)`` with the intermediates in ``dtype`` (bf16: the module's statements, each rounded)."""
    gated = (torch.sigmoid(qkvg[..., 384:].float()) * o.float()).to(dtype)
    y = torch.nn.functional.linear(gated, wo.to(dtype))
    if transposed:
        y = y.transpose(1, 2)
    return res.to(dtype) + y


@needs_ampere
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize(("batch", "length"), [(1, 128), (1, 256), (2, 128)])
def test_the_back_is_no_less_accurate_than_the_bf16_statements(transposed, batch, length):
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    o, qkvg, wo, res = _back_inputs(length, batch)
    assert sm80.supports_back(res, wo)
    out = sm80.back(o, qkvg, wo, res, transposed=transposed)
    assert out.shape == res.shape
    assert out.dtype is torch.bfloat16
    assert out.data_ptr() != res.data_ptr()
    want32 = _back_reference(o, qkvg, wo, res, torch.float32, transposed)
    want16 = _back_reference(o, qkvg, wo, res, torch.bfloat16, transposed)
    mine, base = _rel(out, want32), _rel(want16, want32)
    assert mine <= max(1.25 * base, 3e-3), f"back {mine:.3e} vs bf16 statements {base:.3e}"


@needs_ampere
def test_the_ending_node_back_reads_and_writes_the_transposed_positions():
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    o, qkvg, wo, res = _back_inputs(256, 2)
    got = sm80.back(o, qkvg, wo, res, transposed=True)
    # token (a, b) of the starting problem is element (b, a) of the module's tensors: the same result as the transposed residual and a transposed result
    want = sm80.back(o, qkvg, wo, res.transpose(1, 2).contiguous())
    assert torch.equal(got, want.transpose(1, 2))


@needs_ampere
def test_the_back_replays_bit_identically_and_leaves_its_inputs_alone():
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    o, qkvg, wo, res = _back_inputs(128)
    before = [t.clone() for t in (o, qkvg, wo, res)]
    first = sm80.back(o, qkvg, wo, res)
    again = sm80.back(o, qkvg, wo, res)
    assert torch.equal(first, again)
    assert all(torch.equal(b, t) for b, t in zip(before, (o, qkvg, wo, res), strict=True))


@pytest.mark.skipif(not CUDA, reason="needs a GPU to build the operands")
def test_the_back_gate_rejects_everything_it_is_not_built_for():
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    _, _, wo, res = _back_inputs(128)
    assert sm80.supports_back(res, wo) is AMPERE
    assert not sm80.supports_back(res.float(), wo)
    assert not sm80.supports_back(res, wo.float())
    assert not sm80.supports_back(res, wo[:64])
    assert not sm80.supports_back(res.transpose(1, 2), wo)                                # not contiguous
    assert not sm80.supports_back(res[:, :96], wo)                                         # not square, L % 128
    assert not sm80.supports_back(res[..., :64], wo)                                       # width


# ------------------------------------------------------------------------------------------------------------------ the whole module
def _spy(monkeypatch, module, name, calls):
    """Record every call of `module.<name>` in `calls` and run it."""
    original = getattr(module, name)

    def spy(*args, **kwargs):
        calls.append(name)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, spy)


@needs_ampere
@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize(("batch", "length"), [(1, 128), (1, 256), (2, 128)])
def test_the_module_runs_the_three_cuda_kernels_and_is_no_less_accurate_than_the_triton_path(starting, batch, length, monkeypatch):
    from miniworld_engine.integrations import triattn_sm80
    from miniworld_engine.kernels.triangle_attention.cuda import sm80
    from miniworld_engine.modules.exceptions import ImplementationType
    from miniworld_engine.modules.triangle_attention import TriangleAttention

    module = _module(starting)
    reference = TriangleAttention(128, 4, starting=starting, implementation=ImplementationType.PYTORCH).cuda().float().eval()
    reference.load_state_dict({k: v.float() for k, v in module.state_dict().items()})
    torch.manual_seed(3)
    pair = torch.randn(batch, length, length, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(batch, length, device="cuda") > 0.1
    calls = []
    for name in ("front", "attention", "back"):
        _spy(monkeypatch, sm80, name, calls)
    with torch.no_grad():
        assert triattn_sm80.serves_module(module, pair, mask)
        got = module(pair, mask)
        assert calls == ["front", "attention", "back"], calls
        module._sm80_cuda = False
        triton = module(pair, mask)
        want = reference(pair.float(), mask)
    assert got.shape == pair.shape
    assert got.dtype is torch.bfloat16
    mine, base = _rel(got, want), _rel(triton, want)
    assert mine <= max(1.25 * base, 3e-3), f"cuda module {mine:.3e} vs triton path {base:.3e}"


@needs_ampere
def test_the_module_gate_keeps_everything_it_is_not_built_for_on_the_triton_path():
    from miniworld_engine.integrations import triattn_sm80

    module = _module(True)
    pair = torch.randn(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, 128, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        assert triattn_sm80.serves_module(module, pair, mask)
        assert not triattn_sm80.serves_module(module, pair.float(), mask)
        assert not triattn_sm80.serves_module(module, pair[:, :, :96], mask)
        assert not triattn_sm80.serves_module(module, pair, mask.to(torch.uint8))
        module.train()
        module.p_drop = 0.25
        assert not triattn_sm80.serves_module(module, pair, mask)                         # dropout is active
        module.p_drop = 0.0
        assert triattn_sm80.serves_module(module, pair, mask)
        module._sm80_cuda = False
        assert not triattn_sm80.serves_module(module, pair, mask)
    assert not triattn_sm80.serves_module(_module(True), pair, mask)                       # grad enabled: the backward is not written yet


@needs_ampere
@pytest.mark.parametrize("starting", [True, False])
def test_the_compiled_module_matches_eager(starting):
    """The three kernels are opaque ops with fake implementations: ``torch.compile`` (``custom_op`` wrapping) traces the module's gate and the front,
    the core and the back as graph nodes and the compiled module equals the eager one."""
    import copy

    module = _module(starting)
    compiled = torch.compile(copy.deepcopy(module))
    torch.manual_seed(3)
    pair = torch.randn(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, 128, device="cuda") > 0.1
    try:
        with torch.no_grad():
            want, got = module(pair, mask), compiled(pair, mask)
        assert _rel(got, want) < 2e-3
    finally:
        torch._dynamo.reset()


# ---------------------------------------------------------------------------------------- score ranges the running maximum must follow
def _check_against_the_references(q, k, v, bias, what):
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    assert sm80.supports(q, k, v, bias)
    out, lse = sm80.attention(q, k, v, bias, save_lse=True)
    assert lse is not None
    ref32 = _reference(q, k, v, bias, torch.float32)
    ref16 = _reference(q, k, v, bias, torch.bfloat16)
    mine, base = _rel(out, ref32), _rel(ref16, ref32)
    assert mine <= max(1.25 * base, 3e-3), f"{what}: sm80 {mine:.3e} vs bf16 statements {base:.3e}"
    want_lse = torch.logsumexp(_logits(q, k, bias, torch.float32), dim=-1) * LOG2E
    finite = torch.isfinite(want_lse)
    torch.testing.assert_close(lse[finite], want_lse[finite], rtol=1e-4, atol=2e-3)


@needs_ampere
@pytest.mark.parametrize("length", [128, 256])
def test_a_late_dominant_key_tile_rescales_the_running_state(length):
    """The last keys carry a bias of +30: every row's maximum jumps by far more than the logit spread so far in the last key tile."""
    q, k, v, bias = _inputs(length, False)
    bias = bias.clone()
    bias[..., -32:] += 30.0
    _check_against_the_references(q, k, v, bias, "late dominant tile")


@needs_ampere
def test_leading_fully_masked_key_tiles_then_a_finite_maximum():
    q, k, v, bias = _inputs(256, False)
    mask = torch.ones(1, 256, dtype=torch.bool, device="cuda")
    mask[:, :96] = False                                      # three whole key tiles of 32 masked: the running maximum stays -inf until the fourth
    bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(torch.bfloat16).min)
    _check_against_the_references(q, k, v, bias, "masked prefix")


@needs_ampere
@pytest.mark.parametrize("scale", [3.0, 8.0])
def test_peaked_softmax_rows(scale):
    """Large q . k: nearly one-hot rows; the running maximum moves in many key tiles."""
    q, k, v, bias = _inputs(256, True)
    tm = lambda t: (t.float() * scale).to(torch.bfloat16)
    _check_against_the_references(tm(q), k, v, bias, f"peaked x{scale}")


@needs_ampere
def test_a_row_with_one_finite_key_among_masked_ones():
    q, k, v, bias = _inputs(128, False)
    bias = bias.masked_fill(torch.ones(1, 1, 1, 128, dtype=torch.bool, device="cuda"), torch.finfo(torch.bfloat16).min)
    bias[..., 77] = 0.25                                       # a single key (inside the third key tile) is the whole softmax
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    out, _ = sm80.attention(q, k, v, bias)
    want = v[:, :, :, 77:78, :].expand_as(out).float()         # weight 1 on key 77 of the same row i
    torch.testing.assert_close(out.float(), want, rtol=0, atol=1e-2)


# ------------------------------------------------------------------------------------------------------------------------------- training saves
@needs_ampere
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("ln_dtype", [torch.float32, torch.bfloat16])
def test_front_train_saves_the_normalised_input_and_leaves_the_other_outputs_alone(transposed, ln_dtype):
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    pair, weights, ln_w, ln_b, mask = _front_inputs(256, 2, ln_dtype=ln_dtype)
    qkvg, bias, stats, xh = sm80.front_train(pair, weights, ln_w, ln_b, EPS, mask, transposed=transposed)
    plain_qkvg, plain_bias, plain_stats = sm80.front(pair, weights, ln_w, ln_b, EPS, mask, transposed=transposed, save_stats=True)
    assert torch.equal(qkvg, plain_qkvg)
    assert torch.equal(bias, plain_bias)
    assert plain_stats is not None
    assert torch.equal(stats, plain_stats)
    x = (pair.transpose(1, 2) if transposed else pair).float()
    mean, rstd = x.mean(-1, keepdim=True), (x.var(-1, unbiased=False, keepdim=True) + EPS).rsqrt()
    want = ((x - mean) * rstd).to(torch.bfloat16)
    assert xh.shape == (2, 256, 256, 136)
    assert xh.dtype is torch.bfloat16
    differing = (xh[..., :128].float() - want.float()).abs() > want.float().abs() * 2 ** -8       # at most one bf16 step (an fp32 rounding tie)
    assert float(differing.float().mean()) < 1e-3
    assert torch.equal(xh[..., 128], torch.ones_like(xh[..., 128]))
    assert not xh[..., 129:].any()


@needs_ampere
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize(("batch", "length"), [(1, 128), (2, 128)])
def test_the_back_applies_the_dropout_scale_before_the_residual(transposed, batch, length):
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    o, qkvg, wo, res = _back_inputs(length, batch)
    keep = torch.rand(batch, length, 128, device="cuda") > 0.25
    ds = (keep.float() / 0.75).to(torch.bfloat16)                    # [B, L(column), 128]
    got = sm80.back(o, qkvg, wo, res, transposed=transposed, ds=ds)

    def reference(dtype):
        gated = (torch.sigmoid(qkvg[..., 384:].float()) * o.float()).to(dtype)
        y = torch.nn.functional.linear(gated, wo.to(dtype))
        if transposed:
            y = y.transpose(1, 2) * ds.to(dtype)[:, :, None, :]      # the module's [B, L, 1, C] scale indexes the first index of the result
        else:
            y = y * ds.to(dtype)[:, None, :, :]                      # [B, 1, L, C]
        return res.to(dtype) + y

    want32, want16 = reference(torch.float32), reference(torch.bfloat16)
    mine, base = _rel(got, want32), _rel(want16, want32)
    assert mine <= max(1.25 * base, 3e-3), f"back with dropout {mine:.3e} vs bf16 statements {base:.3e}"
    dropped = ~keep[:, None, :, :].expand_as(res) if not transposed else ~keep[:, :, None, :].expand_as(res)
    assert torch.equal(got[dropped], res[dropped])                    # a dropped channel leaves the residual untouched, bit for bit


# ---------------------------------------------------------------------------------------------------------------------- the back's backward
@needs_ampere
@pytest.mark.parametrize("dropout", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize(("batch", "length"), [(1, 128), (2, 128), (1, 256)])
def test_the_backs_backward_matches_autograd(dropout, transposed, batch, length):
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    o, qkvg, wo, _ = _back_inputs(length, batch, seed=17)
    torch.manual_seed(19)
    dout = torch.randn(batch, length, length, 128, device="cuda").to(torch.bfloat16)
    ds = ((torch.rand(batch, length, 128, device="cuda") > 0.25).float() / 0.75).to(torch.bfloat16) if dropout else None
    dqkvg = torch.zeros(batch, length, length, 512, device="cuda", dtype=torch.bfloat16)
    dov, dy, a, delta = sm80.back_bwd(dout, o, qkvg, dqkvg, wo, transposed=transposed, ds=ds)
    want_delta = (o.float() * dov.float()).view(batch, length, length, 4, 32).sum(-1).permute(0, 3, 1, 2)         # [B, H, L, L]: the row term
    torch.testing.assert_close(delta, want_delta, rtol=2e-4, atol=1e-3)
    dg = dqkvg[..., 384:]

    def autograd(dtype):
        """The module's statements in the starting frame, differentiated by autograd in ``dtype``."""
        g_, o_, wo_ = (t.detach().to(dtype).clone().requires_grad_() for t in (qkvg[..., 384:], o, wo))
        y = torch.nn.functional.linear(torch.sigmoid(g_) * o_, wo_)
        if ds is not None:
            y = y * ds.to(dtype)[:, None, :, :]
        y.backward((dout.transpose(1, 2) if transposed else dout).to(dtype))
        return g_.grad, o_.grad, wo_.grad

    want_g32, want_o32, want_w32 = autograd(torch.float32)
    want_g16, want_o16, want_w16 = autograd(torch.bfloat16)
    for name, got, want16, want32 in (("dg", dg, want_g16, want_g32), ("do", dov, want_o16, want_o32)):
        mine, base = _rel(got, want32), _rel(want16, want32)
        assert mine <= max(1.25 * base, 3e-3), f"{name} {mine:.3e} vs bf16 autograd {base:.3e}"
    t = batch * length * length
    dwo = dy.view(t, 128).float().T @ a.view(t, 128).float()
    mine, base = _rel(dwo, want_w32), _rel(want_w16, want_w32)
    assert mine <= max(1.25 * base, 3e-3), f"dWo {mine:.3e} vs bf16 autograd {base:.3e}"
    assert torch.equal(dqkvg[..., :384], torch.zeros_like(dqkvg[..., :384]))    # only the gate columns are written


@needs_ampere
def test_the_backs_backward_replays_bit_identically():
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    o, qkvg, wo, _ = _back_inputs(128, 1, seed=23)
    dout = torch.randn(1, 128, 128, 128, device="cuda").to(torch.bfloat16)
    first, second = (torch.zeros(1, 128, 128, 512, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    a = sm80.back_bwd(dout, o, qkvg, first, wo)
    b = sm80.back_bwd(dout, o, qkvg, second, wo)
    assert torch.equal(first, second)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


# --------------------------------------------------------------------------------------------------------------------- the front's backward
def _front_autograd(pair, weights, ln_w, ln_b, transposed, dqkvg, db, dtype):
    """The module's statements in the starting frame in ``dtype``, differentiated by autograd: gradients of the pair tensor, the weights, gamma, beta."""
    x = (pair.transpose(1, 2) if transposed else pair).detach().to(dtype).clone().requires_grad_()
    ws = [w.detach().to(dtype).clone().requires_grad_() for w in weights]
    g_, b_ = ln_w.detach().to(dtype).clone().requires_grad_(), ln_b.detach().to(dtype).clone().requires_grad_()
    xn = torch.nn.functional.layer_norm(x, (128,), g_, b_, EPS)
    qkvg = torch.nn.functional.linear(xn, torch.cat(ws[:4]))
    bias = torch.nn.functional.linear(xn, ws[4]).permute(0, 3, 1, 2)
    torch.autograd.backward([qkvg, bias], [dqkvg.to(dtype), db.to(dtype)])
    dx = x.grad.transpose(1, 2) if transposed else x.grad
    return dx, [w.grad for w in ws], g_.grad, b_.grad


@needs_ampere
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize(("batch", "length"), [(1, 128), (2, 128), (1, 256)])
@pytest.mark.parametrize("ln_dtype", [torch.float32, torch.bfloat16])
def test_the_fronts_backward_matches_autograd(transposed, batch, length, ln_dtype):
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    pair, weights, ln_w, ln_b, mask = _front_inputs(length, batch, seed=29, ln_dtype=ln_dtype)
    torch.manual_seed(31)
    dqkvg = torch.randn(batch, length, length, 512, device="cuda").to(torch.bfloat16)
    db = (0.5 * torch.randn(batch, 4, length, length, device="cuda")).to(torch.bfloat16)
    db = db.masked_fill(~mask[:, None, None, :], 0)                                           # a masked key's bias gradient is zero
    dout = torch.randn(batch, length, length, 128, device="cuda").to(torch.bfloat16)
    _, _, stats, xh = sm80.front_train(pair, weights, ln_w, ln_b, EPS, mask, transposed=transposed)
    dpair = sm80.front_bwd(dqkvg, db, pair, stats, dout, weights, ln_w, transposed=transposed)
    grads, d_gamma, d_beta = sm80.front_weight_grads(dqkvg, db, xh, weights, ln_w, ln_b)

    dx32, dw32, dg32, dbeta32 = _front_autograd(pair, weights, ln_w, ln_b, transposed, dqkvg, db, torch.float32)
    dx16, dw16, dg16, dbeta16 = _front_autograd(pair, weights, ln_w, ln_b, transposed, dqkvg, db, torch.bfloat16)
    want32, want16 = dout.float() + dx32, dout + dx16                                          # the residual's gradient joins
    mine, base = _rel(dpair, want32), _rel(want16, want32)
    assert mine <= max(1.25 * base, 3e-3), f"dpair {mine:.3e} vs bf16 autograd {base:.3e}"
    for name, got, w16, w32 in [*((f"dW{i}", g, a, b) for i, (g, a, b) in enumerate(zip(grads, dw16, dw32, strict=True))),
                                ("dgamma", d_gamma, dg16, dg32), ("dbeta", d_beta, dbeta16, dbeta32)]:
        mine, base = _rel(got, w32), _rel(w16, w32)
        assert mine <= max(1.5 * base, 4e-3), f"{name} {mine:.3e} vs bf16 autograd {base:.3e}"
    assert d_gamma.dtype is ln_dtype
    assert [g.dtype for g in grads] == [w.dtype for w in weights]


@needs_ampere
def test_the_fronts_backward_replays_bit_identically_and_does_not_touch_its_inputs():
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    pair, weights, ln_w, ln_b, mask = _front_inputs(128, 1, seed=37)
    dqkvg = torch.randn(1, 128, 128, 512, device="cuda").to(torch.bfloat16)
    db = torch.randn(1, 4, 128, 128, device="cuda").to(torch.bfloat16)
    dout = torch.randn(1, 128, 128, 128, device="cuda").to(torch.bfloat16)
    _, _, stats, _ = sm80.front_train(pair, weights, ln_w, ln_b, EPS, mask)
    before = [t.clone() for t in (pair, dqkvg, db, dout, stats)]
    first = sm80.front_bwd(dqkvg, db, pair, stats, dout, weights, ln_w)
    again = sm80.front_bwd(dqkvg, db, pair, stats, dout, weights, ln_w)
    assert torch.equal(first, again)
    assert all(torch.equal(b, t) for b, t in zip(before, (pair, dqkvg, db, dout, stats), strict=True))


# ------------------------------------------------------------------------------------------------------------- training through the module
def _loss_and_grads(module, pair, mask, cot, seed):
    """Output and gradients (pair tensor, then every parameter) of ``sum(out * cot)``; the RNG is re-seeded so the dropout draw is the same."""
    pair = pair.detach().clone().requires_grad_()
    torch.manual_seed(seed)
    out = module(pair, mask)
    params = list(module.parameters())
    grads = torch.autograd.grad((out.float() * cot).sum(), [pair, *params])
    return out.detach(), grads


@needs_ampere
@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize(("batch", "length"), [(1, 128), (2, 128), (1, 256)])
@pytest.mark.parametrize("ln_fp32", [False, True])
def test_the_module_trains_through_the_cuda_kernels_and_is_no_less_accurate_than_the_triton_path(starting, batch, length, ln_fp32, monkeypatch):
    from miniworld_engine.integrations import triattn_sm80
    from miniworld_engine.kernels.triangle_attention.cuda import sm80
    from miniworld_engine.modules.exceptions import ImplementationType
    from miniworld_engine.modules.triangle_attention import TriangleAttention

    module = _module(starting)
    if ln_fp32:
        module.ln_pair.float()
    reference = TriangleAttention(128, 4, starting=starting, implementation=ImplementationType.PYTORCH).cuda().float()
    reference.load_state_dict({k: v.float() for k, v in module.state_dict().items()})
    torch.manual_seed(3)
    pair = torch.randn(batch, length, length, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(batch, length, device="cuda") > 0.1
    cot = torch.randn(batch, length, length, 128, device="cuda")
    assert triattn_sm80.serves_train(module, pair.requires_grad_(), mask)
    calls = []
    _spy(monkeypatch, sm80, "forward_train", calls)
    _spy(monkeypatch, sm80, "backward_train", calls)
    out, grads = _loss_and_grads(module, pair, mask, cot, 1)
    assert calls == ["forward_train", "backward_train"], calls
    module._sm80_cuda = False
    out_triton, grads_triton = _loss_and_grads(module, pair, mask, cot, 1)
    out_ref, grads_ref = _loss_and_grads(reference, pair.float(), mask, cot, 1)
    assert _rel(out, out_ref) <= max(1.25 * _rel(out_triton, out_ref), 3e-3)
    names = ["pair", *(n for n, _ in module.named_parameters())]
    for name, mine_g, tri_g, ref_g in zip(names, grads, grads_triton, grads_ref, strict=True):
        mine, base = _rel(mine_g, ref_g), _rel(tri_g, ref_g)
        assert mine <= max(1.25 * base, 6e-3), f"d{name}: cuda {mine:.3e} vs triton path {base:.3e}"
        assert mine_g.dtype == tri_g.dtype, name


@needs_ampere
@pytest.mark.parametrize("starting", [True, False])
def test_the_module_trains_with_the_broadcast_dropout_like_the_triton_path(starting):
    module = _module(starting)
    module.train()
    module.p_drop = 0.25
    torch.manual_seed(3)
    pair = torch.randn(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, 128, device="cuda") > 0.1
    cot = torch.randn(1, 128, 128, 128, device="cuda")
    out, grads = _loss_and_grads(module, pair, mask, cot, 5)
    module._sm80_cuda = False
    out_triton, grads_triton = _loss_and_grads(module, pair, mask, cot, 5)         # the same draw: the same dropped channels
    assert _rel(out, out_triton) < 6e-3
    names = ["pair", *(n for n, _ in module.named_parameters())]
    for name, mine_g, tri_g in zip(names, grads, grads_triton, strict=True):
        assert _rel(mine_g, tri_g) < 2e-2, f"d{name}: {_rel(mine_g, tri_g):.3e}"


@needs_ampere
def test_no_autograd_path_when_nothing_wants_a_gradient():
    from miniworld_engine.integrations import triattn_sm80

    module = _module(True)
    for p in module.parameters():
        p.requires_grad_(False)
    pair = torch.randn(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16)
    assert not triattn_sm80.serves_train(module, pair, None)                       # no gradient wanted anywhere: the inference path serves it
    with torch.no_grad():
        assert triattn_sm80.serves_module(module, pair, None)


# ------------------------------------------------------------------------------------------------------------------ the attention core's backward
def _token_major(sm80, t):
    """The token-major view the kernels take (never None for the views these tests build)."""
    tm = sm80._token_major(t)
    assert tm is not None
    return tm


def _core_backward_case(length, batch, masked, seed=41):
    """q / k / v token-major views of a [B, L, L, 512] buffer, the bias, lse and out of the forward, a cotangent and its row term."""
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    torch.manual_seed(seed)
    buf = torch.randn(batch, length, length, 512, device="cuda").to(torch.bfloat16)
    q5, k5, v5, _ = sm80.qkv_views(buf)
    bias = (0.5 * torch.randn(batch, 4, length, length, device="cuda")).to(torch.bfloat16)
    if masked:
        mask = torch.rand(batch, length, device="cuda") > 0.1
        bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(torch.bfloat16).min)
    out5, lse = sm80.attention(q5, k5, v5, bias, save_lse=True)
    out = out5.permute(0, 2, 3, 1, 4).reshape(batch, length, length, 128)
    dov = torch.randn(batch, length, length, 128, device="cuda").to(torch.bfloat16)
    delta = (out.float() * dov.float()).view(batch, length, length, 4, 32).sum(-1).permute(0, 3, 1, 2).contiguous()
    return buf, (q5, k5, v5), bias, lse, dov, delta


@needs_ampere
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize(("batch", "length"), [(1, 128), (1, 256), (2, 128)])
def test_the_attention_backward_matches_autograd(masked, batch, length):
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    _, (q5, k5, v5), bias, lse, dov, delta = _core_backward_case(length, batch, masked)
    dqkvg = torch.zeros(batch, length, length, 512, device="cuda", dtype=torch.bfloat16)
    qm, km, vm = (_token_major(sm80, t) for t in (q5, k5, v5))
    db = sm80.attention_backward(qm, km, vm, dov, bias, lse, delta, dqkvg)
    assert db.shape == (batch, 4, length, length)
    cot = dov.view(batch, length, length, 4, 32).permute(0, 3, 1, 2, 4)                          # [B, H, L, L2, D]

    def autograd(dtype):
        q_, k_, v_ = (t.detach().to(dtype).clone().requires_grad_() for t in (q5, k5, v5))
        b_ = bias.detach().to(dtype).clone().requires_grad_()
        _reference(q_, k_, v_, b_, dtype).backward(cot.to(dtype))
        return q_.grad, k_.grad, v_.grad, b_.grad

    want32, want16 = autograd(torch.float32), autograd(torch.bfloat16)
    to5 = lambda t: t.view(batch, length, length, 4, 32).permute(0, 3, 1, 2, 4)
    got = [to5(dqkvg[..., 128 * i:128 * (i + 1)]) for i in range(3)] + [db]
    for name, g, w16, w32 in zip(("dq", "dk", "dv", "dbias"), got, want16, want32, strict=True):
        if name == "dbias" and masked:
            keep = (bias[:, :1] > torch.finfo(torch.bfloat16).min).expand_as(db)
            g, w16, w32 = g[keep], w16[keep], w32[keep]
            assert float(db[~keep].abs().max()) == 0.0              # a masked key's bias gradient is zero
        mine, base = _rel(g, w32), _rel(w16, w32)
        assert mine <= max(1.25 * base, 6e-3), f"{name} {mine:.3e} vs bf16 autograd {base:.3e}"
    assert torch.equal(dqkvg[..., 384:], torch.zeros_like(dqkvg[..., 384:]))                     # only the first 384 columns are written


@needs_ampere
def test_the_attention_backward_replays_bit_identically():
    from miniworld_engine.kernels.triangle_attention.cuda import sm80

    _, (q5, k5, v5), bias, lse, dov, delta = _core_backward_case(128, 1, True, seed=43)
    qm, km, vm = (_token_major(sm80, t) for t in (q5, k5, v5))
    first, second = (torch.zeros(1, 128, 128, 512, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    a = sm80.attention_backward(qm, km, vm, dov, bias, lse, delta, first)
    b = sm80.attention_backward(qm, km, vm, dov, bias, lse, delta, second)
    assert torch.equal(a, b)
    assert torch.equal(first, second)


@needs_ampere
@pytest.mark.parametrize("starting", [True, False])
def test_the_compiled_module_trains_like_the_eager_one(starting):
    """The forward and the backward are opaque ops with fake implementations behind one autograd function: ``torch.compile`` traces them as nodes and the
    compiled module's output and gradients equal the eager module's."""
    import copy

    module = _module(starting)
    compiled = torch.compile(copy.deepcopy(module))
    torch.manual_seed(3)
    pair = torch.randn(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, 128, device="cuda") > 0.1
    cot = torch.randn(1, 128, 128, 128, device="cuda")
    try:
        out, grads = _loss_and_grads(module, pair, mask, cot, 1)
        out_c, grads_c = _loss_and_grads(compiled, pair, mask, cot, 1)
        assert _rel(out_c, out) < 2e-3
        for i, (g, g_c) in enumerate(zip(grads, grads_c, strict=True)):
            assert _rel(g_c, g) < 4e-3, f"gradient {i}: {_rel(g_c, g):.3e}"
    finally:
        torch._dynamo.reset()


@needs_ampere
@pytest.mark.parametrize("starting", [True, False])
def test_a_length_that_is_not_a_power_of_two_trains_and_infers_like_the_triton_path(starting):
    """L = 640 = 5 x 128: whole 128-token tiles in every kernel, odd numbers of row groups and key tiles (the loops' tails)."""
    module = _module(starting)
    torch.manual_seed(3)
    pair = torch.randn(1, 640, 640, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(1, 640, device="cuda") > 0.1
    cot = torch.randn(1, 640, 640, 128, device="cuda")
    with torch.no_grad():
        got = module(pair, mask)
        module._sm80_cuda = False
        want = module(pair, mask)
        module._sm80_cuda = True
    assert _rel(got, want) < 3e-3
    out, grads = _loss_and_grads(module, pair, mask, cot, 1)
    module._sm80_cuda = False
    out_t, grads_t = _loss_and_grads(module, pair, mask, cot, 1)
    assert _rel(out, out_t) < 3e-3
    for i, (g, gt) in enumerate(zip(grads, grads_t, strict=True)):
        assert _rel(g, gt) < 2e-2, f"gradient {i}: {_rel(g, gt):.3e}"
