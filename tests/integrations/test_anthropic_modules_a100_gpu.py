"""Public Anthropic module routes: independent reference and live CUDA graphs."""
import os

import pytest
import torch

from miniworld_engine.modules import (
    AdaptiveLayerNorm,
    AttentionPairBias,
    AugmentedAttentionPairBias,
    ConditionedTransition,
    MSAPairWeightedAveraging,
    OuterProductMean,
    Transition,
    TriangleAttention,
)
from miniworld_engine.modules.bias_only_dit import BiasOnlyDiTBlock
from miniworld_engine.modules.dit import DiTBlock
from miniworld_engine.modules.pairformer.module import PairformerBlock, PairformerConfig
from miniworld_engine.modules.primitives import LayerNorm, RMSNorm
from miniworld_engine.modules.swa_atom_attention import (
    SWA3DRoPEAttention,
    build_attention_params,
)
from miniworld_engine.modules.swa_dit.module import SWADiTBlock, SwiGLUFFN
from miniworld_engine.modules.triangle_attention.bidirectional import (
    BidirectionalTriangleAttention,
)

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not os.environ.get("MINIWORLD_ANTHROPIC_ROOT"), reason="needs pinned Anthropic payload")]

FAMILIES = ("layernorm", "rmsnorm", "transition", "triattn", "adaln", "conditioned_transition",
            "attention_pair_bias", "augmented_attention", "dit_token", "dit_atom", "opm", "pwa",
            "swa_attention", "swa_dit", "swiglu_ffn", "bias_only_dit", "opm_post",
            "triattn_bias_start", "triattn_bias_end", "triattn_bidir_bias", "triattn_bidir_sa")


def fixture(family, length, impl):
    kw = {"implementation": impl}
    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    mask = torch.arange(length, device="cuda")[None, :] % 7 != 0
    if family.startswith("pairformer"):
        cfg = PairformerConfig(n_block=1, p_drop=0, use_self_attention="bias" not in family,
                               bidirectional_trimul="bidir" in family)
        return PairformerBlock(cfg, **kw), (rand(1, length, length, 128), mask)
    if family == "layernorm":
        return LayerNorm(128, **kw), (rand(1, length, 128),)
    if family == "rmsnorm":
        return RMSNorm(128, **kw), (rand(1, length, 128),)
    if family == "transition":
        return Transition(128, **kw), (rand(1, length, length, 128),)
    if family == "triattn":
        return TriangleAttention(128, 4, anthropic_row="block:triattn_native", **kw), (rand(1, length, length, 128), mask)
    if family.startswith("triattn_bias"):
        return TriangleAttention(128, 4, use_self_attention=False, starting=family.endswith("start"), **kw), (rand(1, length, length, 128), mask)
    if family.startswith("triattn_bidir"):
        return BidirectionalTriangleAttention(128, 4, use_self_attention=family.endswith("sa"), **kw), (rand(1, length, length, 128), mask)
    if family == "bias_only_dit":
        return BiasOnlyDiTBlock(768, 384, 128, 16, **kw), (rand(2, 1, length, 768), rand(1, 1, length, 384), rand(1, length, length, 128), mask)
    if family in ("adaln", "conditioned_transition"):
        cls = AdaptiveLayerNorm if family == "adaln" else ConditionedTransition
        return cls(128, 128, **kw), (rand(2, 1, length, 128), rand(1, 1, length, 128))
    if family == "attention_pair_bias":
        return AttentionPairBias(384, 128, 8, **kw), (rand(1, length, 384), rand(1, length, length, 128), mask)
    if family in ("augmented_attention", "dit_token", "dit_atom"):
        d, c, p, h = (128, 128, 16, 4) if family == "dit_atom" else (768, 384, 128, 16)
        cls = AugmentedAttentionPairBias if family == "augmented_attention" else DiTBlock
        return cls(d, c, p, h, **kw), (rand(2, 1, length, d), rand(1, 1, length, c), rand(1, length, length, p), mask)
    if family == "opm":
        m = rand(1, 64, length, 64)
        return OuterProductMean(64, 128, **kw), (m, mask[:, None].expand(1, 64, length))
    if family == "opm_post":
        return OuterProductMean(64, 256, normalize_before_proj=False, **kw), (rand(1, 64, length, 64), mask[:, None].expand(1, 64, length))
    if family == "swiglu_ffn":
        return SwiGLUFFN(128, **kw), (rand(2, length, 128),)
    if family.startswith("swa_"):
        angles = torch.randn(1, length, 16, device="cuda")
        valid = torch.arange(length, device="cuda")[None] < torch.tensor([length, length-17, 0], device="cuda")[:, None]
        ap = build_attention_params(angles.cos(), angles.sin(), valid, 3)
        if family == "swa_attention":
            return SWA3DRoPEAttention(128, 4, **kw), (rand(3, length, 128), ap)
        return SWADiTBlock(**kw), (rand(3, length, 128), rand(3, length, 128), ap)
    return MSAPairWeightedAveraging(64, 128, 8, 32, **kw), (rand(1, 64, length, 64), rand(1, length, length, 128), mask)


@pytest.mark.parametrize("length", [128, 384])
@pytest.mark.parametrize("family", [*FAMILIES, "pairformer", "pairformer_bias", "pairformer_bidir", "pairformer_bidir_bias"])
def test_module_reference_and_graph(family, length):
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("requires A100")
    torch.manual_seed(191)
    actual, inputs = fixture(family, length, "anthropic")
    actual = actual.cuda().bfloat16().eval()
    with torch.no_grad():
        for name, param in actual.named_parameters():
            if param.ndim > 1:
                param.normal_(std=param.shape[-1] ** -.5)
            elif name.endswith("weight"):
                param.normal_(mean=1, std=.1)
            else:
                param.normal_(std=.1)
    reference, _ = fixture(family, length, "pytorch")
    reference = reference.cuda().float().eval()
    reference.load_state_dict(actual.state_dict())
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            want = reference(*(x.float() if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x in inputs))
            got = actual(*inputs)
            rel = (got.float() - want).norm() / want.norm().clamp_min(1e-8)
            assert torch.isfinite(got).all()
            assert rel < .04, (family, float(rel))
            if family in {"transition", "triattn", "conditioned_transition", "attention_pair_bias",
                          "augmented_attention", "dit_token", "dit_atom", "pwa", "swa_dit"}:
                delta = want - inputs[0].float()
                assert (got.float() - want).norm() / delta.norm().clamp_min(1e-8) < .06
            assert any(hasattr(m, "anthropic_selection") for m in actual.modules())
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    actual(*inputs)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = actual(*inputs)
            graph.replay()
            torch.testing.assert_close(output, got, rtol=0, atol=0)
            inputs[0].mul_(.5)
            graph.replay()
            torch.testing.assert_close(output, actual(*inputs), rtol=0, atol=0)
        with pytest.raises(RuntimeError, match=r"inference-only|forward-only"):
            actual(*inputs)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


@pytest.mark.parametrize("family", ["attention_pair_bias", "augmented_attention", "opm", "pwa",
                                    "bias_only_dit", "opm_post", "triattn_bias_start", "triattn_bias_end",
                                    "triattn_bidir_bias", "triattn_bidir_sa"])
@pytest.mark.parametrize("mask_kind", ["none", "empty", "per_sample"])
def test_masks_batches_and_changed_weights(family, mask_kind):
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("requires A100")
    if family.startswith("triattn_bidir") and mask_kind == "empty":
        pytest.skip("existing bidirectional reference uses -inf softmax and returns NaN for empty keys")
    torch.manual_seed(153)
    mod, original = fixture(family, 128, "anthropic")
    ref, _ = fixture(family, 128, "pytorch")
    mod, ref = mod.cuda().bfloat16().eval(), ref.cuda().float().eval()
    args = list(original)
    for i, t in enumerate(args[:-1]):
        axis = 1 if family in {"augmented_attention", "bias_only_dit"} and i < 2 else 0
        args[i] = torch.cat((t, t * .7), axis)
    if mask_kind == "none":
        args[-1] = None
    else:
        mask = torch.cat((args[-1], args[-1]), 0)
        if mask_kind == "empty":
            mask.zero_()
        elif family == "augmented_attention":
            mask = torch.stack((mask, ~mask), 0)
        else:
            mask[1] = ~mask[1]
        args[-1] = mask
    with torch.no_grad():
        for _ in range(2):
            for p in mod.parameters():
                if p.ndim > 1:
                    p.normal_(std=p.shape[-1] ** -.5)
            ref.load_state_dict(mod.state_dict())
            got = mod(*args)
            # torch 2.13's fused fp32 SDPA returns zero at an all-finite-min
            # bias. The mathematical module contract is a uniform average;
            # use independent math SDPA rather than that GPU backend as oracle.
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                want = ref(*(t.float() if isinstance(t, torch.Tensor) and t.is_floating_point() else t for t in args))
            assert torch.isfinite(got).all()
            assert (got.float() - want).norm() / want.norm().clamp_min(1e-8) < .04


@pytest.mark.parametrize("cond_shape", [(1, 1, 33, 128), (1, 2, 33, 128), (3, 1, 33, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_conditioning_period_and_graph_updates(cond_shape, dtype):
    torch.manual_seed(41)
    mod = ConditionedTransition(128, 128, implementation="anthropic").cuda().to(dtype).eval()
    ref = ConditionedTransition(128, 128, implementation="pytorch").cuda().float().eval()
    x = torch.randn(3, 2, 33, 128, device="cuda", dtype=dtype)
    cond = torch.randn(cond_shape, device="cuda", dtype=dtype)
    with torch.no_grad():
        for p in mod.parameters():
            if p.ndim > 1:
                p.normal_(std=p.shape[-1] ** -.5)
        ref.load_state_dict(mod.state_dict())
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                mod(x, cond)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = mod(x, cond)
        for _ in range(2):
            cond.mul_(.6)
            graph.replay()
            want = ref(x.float(), cond.float())
            assert (out.float() - want).norm() / want.norm() < .01
            torch.testing.assert_close(out, mod(x, cond), rtol=0, atol=0)
        expected_period = 33 if cond_shape[:2] == (1, 1) else 66 if cond_shape[:2] == (1, 2) else 198
        assert mod.anthropic_selection["gate_period"] == expected_period
