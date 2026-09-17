"""Actual checkpoint dimensions must remain executable build cases."""

from miniworld_engine.autotune.builder import cases


def test_checkpoint_dimensions_reach_build_cases():
    all_cases = {c.name: c for c in cases()}

    def contains(name, **dims):
        assert any(
            all(d.get(k) == v for k, v in dims.items()) for d in all_cases[name].dims
        ), (name, dims)

    contains("triangle_multiplication", d_pair=64, d_hidden=64)
    contains("triangle_multiplication", d_pair=384, d_hidden=384)
    contains("triangle_attention_checkpoint", d_pair=64, d_hidden=128, n_head=4)
    contains("triangle_attention_checkpoint", d_pair=384, d_hidden=384, n_head=12)
    contains("conditioned_transition", d_hidden=768, d_cond=768)
    contains("augmented_attention", d_single=768, d_cond=768, d_pair=256, n_head=16)
    contains("transition", d_hidden=64, n=2)
    contains("swiglu_ffn", d_hidden=128, d_expanded=256)
    contains("gated_linear", d_hidden=64, d_out=128)
    contains("layernorm_linear_native", d_norm=16, n_head=12)
    contains("rms_norm_modulation", d_hidden=128, d_cond=128)
    for width in (16, 267, 451, 831, 833, 2560):
        contains("layernorm_native", d_norm=width)
    # A non-square template is recorded in the census, not advertised as a
    # buildable fused TriMul simply because its constructor accepts the numbers.
    assert not any(
        d["d_pair"] != d["d_hidden"] for d in all_cases["triangle_multiplication"].dims
    )


def test_exact_leaf_axes_do_not_enter_unrelated_gemm_ladders():
    from miniworld_engine.autotune.builder import CASE_NAMES
    from miniworld_engine.autotune.checkpoint_cases import NAMES

    assert set(NAMES) <= set(CASE_NAMES)
    for case in cases():
        if case.name in NAMES:
            assert case.factory
            assert case.inputs
            assert case.rows
            assert len(case.dims) == len(case.streams) == len(case.lengths_by_dim)


def test_compiled_shape_keys_keep_their_values_and_shape_dependence():
    import zlib

    import torch

    from miniworld_engine.autotune.shape_key import _axis_name_tag, pack

    for names in (("K", "N"), ("H", "HEAD_DIM"), ("N",)):
        assert _axis_name_tag(names) == zlib.crc32(",".join(names).encode()) % 4096

    def fn(x):
        return torch.full(
            (1,), pack(128, K=x.shape[-1], N=128), dtype=torch.int64, device=x.device
        )

    compiled = torch.compile(fn, fullgraph=True, backend="eager")
    for width in (128, 256, 833):
        assert compiled(torch.empty(1, 2, width)).item() == pack(128, K=width, N=128)


def test_native_diffusion_norm_keeps_both_augmentation_contracts():
    case = next(c for c in cases() if c.name == "layernorm_native")
    for bias in (0, 1):
        indices = [
            i
            for i, d in enumerate(case.dims)
            if d == {"d_norm": 768, "has_bias": bias}
            and case.stream_for(i) == "token_single"
        ]
        assert {
            (case.augmentation_for(i, False), case.augmentation_for(i, True))
            for i in indices
        } == {(1, 1), (5, 48)}
