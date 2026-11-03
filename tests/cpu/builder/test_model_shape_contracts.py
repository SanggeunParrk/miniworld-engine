"""Checkpoint census is independent of the registry being pruned."""
import json
from pathlib import Path

import pytest
import torch

from miniworld_engine.autotune import builder, checkpoint_cases, module_registry
from miniworld_engine.utils.viz import sweep_page

ROOT = Path(__file__).resolve().parents[3]
CENSUS = json.loads((ROOT / "tests/cpu/data/checkpoint-shapes-20260915.json").read_text())["groups"]


def test_all_affine_checkpoint_norm_dimensions_survive():
    # Model paths decide stream membership; this independent constructor census
    # additionally prevents removing any actual affine width/bias combination.
    want = set()
    for group in CENSUS:
        if group["class"] in {"LayerNorm", "OpenFoldLayerNorm"} and "weight" in group["parameters"]:
            want.add((group["parameters"]["weight"][0], int("bias" in group["parameters"])))
    have = {(r.dims["d_norm"], r.dims["has_bias"]) for r in module_registry.module_rows() if r.module == "layernorm_native"}
    assert want <= have


def test_projected_checkpoint_token_attention_pairs_survive():
    """Every token-level AttentionPairBias of a released checkpoint (no conditioning: the trunk's single attention) is an
    ``attention_pair_bias`` row at its (heads, single width, pair width); with conditioning it is the diffusion's ``augmented_attention``."""
    want, want_diffusion = set(), set()
    for group in CENSUS:
        if group["class"] == "AttentionPairBias" and not group["attrs"]["cross_attention_mode"]:
            shape = (group["attrs"]["n_heads"], group["parameters"]["attention.linear_q.weight"][0], group["parameters"]["layernorm_z.weight"][0])
            (want_diffusion if group["attrs"]["has_s"] else want).add(shape)
    rows = module_registry.module_rows()
    have = {(r.dims["n_head"], r.dims["d_single"], r.dims["d_pair"]) for r in rows if r.module == "attention_pair_bias"}
    have_diffusion = {(r.dims["n_head"], r.dims["d_single"], r.dims["d_pair"]) for r in rows if r.module == "augmented_attention"}
    assert want <= have, sorted(want - have)
    assert want_diffusion <= have_diffusion, sorted(want_diffusion - have_diffusion)


def test_transition_census_pairs_are_preserved():
    want = set()
    for group in CENSUS:
        if group["class"] == "Transition":
            a = group["attrs"]
            want.add((a.get("c_in", a.get("d_hidden")), a.get("n", a.get("num_intermediate_factor"))))
    have = {(r.dims["d_hidden"], r.dims.get("n", 4)) for r in module_registry.module_rows() if r.module == "transition"}
    assert want <= have


def test_semantic_streams_and_lengths():
    for row in module_registry.module_rows():
        assert row.lengths == module_registry.STREAM_LADDERS[row.stream]
        if row.module == "layernorm_native" and row.dims["d_norm"] == 16:
            assert row.stream == "atom_pair"
        if row.module == "layernorm_native" and row.dims["d_norm"] == 256:
            assert row.stream in {"token_pair", "noise"}
    assert module_registry.STREAM_LADDERS["token_single"] == (128, 256, 384, 512, 640, 768)
    assert module_registry.STREAM_LADDERS["atom_single"] == tuple(range(1024, 8193, 1024))


@pytest.mark.parametrize(("stream", "length", "width", "batch", "shape"), [
    ("atom_pair", 4096, 16, 1, (1, 128, 32, 128, 16)),
    ("noise", 1, 256, 48, (48, 1, 256)),
    ("token_pair", 384, 384, 1, (1, 384, 384, 384)),
    ("msa_token", 384, 128, 1, (1, 8, 384, 128)),
])
def test_stream_inputs_match_model_layout(monkeypatch, stream, length, width, batch, shape):
    def meta_randn(*size: int, device=None, dtype=None):
        return torch.empty(size, device="meta", dtype=dtype)

    monkeypatch.setattr(torch, "randn", meta_randn)
    x, = checkpoint_cases._inputs("layernorm_native", batch, length, {"d_norm": width}, torch.bfloat16, stream)
    assert tuple(x.shape) == shape


def test_case_names_and_bench_mappings_remain_valid():
    from miniworld_engine import cli
    names = {case.name for case in builder.cases()}
    assert names == set(builder.CASE_NAMES)
    assert all(set(cases) <= names for cases in cli.KERNEL_TARGETS.values())


def test_stale_html_plan_cannot_reintroduce_cartesian_shapes(monkeypatch):
    from miniworld_engine.autotune import plan
    def stale(*args):
        raise ValueError("stale derivation")
    monkeypatch.setattr(plan, "load", stale)
    with pytest.raises(ValueError, match="stale derivation"):
        sweep_page.collect()


def test_same_dimensions_keep_distinct_augmentation_rows(monkeypatch):
    from miniworld_engine.autotune import derive, plan

    monkeypatch.setattr(builder, "device_sm", lambda: "sm86")
    case = next(c for c in builder.cases() if c.name == "layernorm_native")
    units = builder.units([case])
    labels = [plan.label(u, case) for u in units]
    assert len(labels) == len(set(labels))
    assert set(labels) == {u.label for u in derive.units(list(case.rows), arch="sm86")}
    assert len({u.stem for u in units}) == len(units)


def _rows(module: str) -> list:
    return [r for r in module_registry.module_rows() if r.module == module]


def test_every_released_trunk_module_shape_is_a_row():
    """The trunk modules of every released checkpoint (the census) at the widths the checkpoint holds."""
    tri_attn, opm, pwa, trimul = set(), set(), set(), set()
    for g in CENSUS:
        a = g["attrs"]
        if g["class"] == "TriangleAttention":
            tri_attn.add((a["c_in"], a["no_heads"] * a["c_hidden"], a["no_heads"]))
        elif g["class"] == "GridSelfAttention":
            tri_attn.add((a["c_pair"], a["num_head"] * a["qkv_dim"], a["num_head"]))
        elif g["class"] == "OuterProductMean" and "c_m" in a:
            opm.add((a["c_m"], a["c_z"], a["c_hidden"]))
        elif g["class"] == "OuterProductMean" and "c_msa" in a:
            opm.add((a["c_msa"], a["num_output_channel"], a["num_outer_channel"]))
        elif g["class"] == "MSAPairWeightedAveraging" and "c_m" in a:
            pwa.add((a["c_m"], a["c_z"], a["n_heads"]))
        elif g["class"] in ("TriangleMultiplicationOutgoing", "TriangleMultiplicationIncoming"):
            trimul.add((a["c_z"], a["c_hidden"]))
        elif g["class"] == "TriangleMultiplication" and "c_pair" in a:
            trimul.add((a["c_pair"], a["c_pair"]))
    have_attn = {(r.dims["d_pair"], r.dims["d_hidden"], r.dims["n_head"]) for m in ("triangle_attention_checkpoint", "triangle_attention_heads")
                 for r in _rows(m) if "d_pair" in r.dims} | {(128, r.dims["d_hidden"], r.dims["n_head"]) for r in _rows("triangle_attention_heads")}
    assert tri_attn <= have_attn, sorted(tri_attn - have_attn)
    have_opm = {(r.dims["d_msa"], r.dims["d_pair"], r.dims["d_hidden"]) for r in _rows("outer_product_mean")}
    assert opm <= have_opm, sorted(opm - have_opm)
    have_pwa = {(r.dims["d_msa"], r.dims["d_pair"], r.dims["n_head"]) for r in _rows("msa_pair_weighted_averaging")}
    assert pwa <= have_pwa, sorted(pwa - have_pwa)
    have_trimul = {(r.dims["d_pair"], r.dims["d_hidden"]) for r in _rows("triangle_multiplication")}
    assert trimul <= have_trimul, sorted(trimul - have_trimul)


#: Module shapes of the released checkpoints the census does not hold (Boltz-2, RoseTTAFold3, Chai-1, IntelliFold2, the Protenix-v1 template),
#: read off their widths (pair / MSA / template channels, heads, value and hidden dims, transition factors): (module, stream, dims).
RELEASED_SPEC_ROWS = (
    ("triangle_attention_checkpoint", "token_pair", "d_pair=128;d_hidden=128;n_head=4;gate_bias=1;out_bias=1"),  # RoseTTAFold3 trunk
    ("triangle_attention_checkpoint", "token_pair", "d_pair=64;d_hidden=256;n_head=4;gate_bias=1;out_bias=1"),   # RoseTTAFold3 template
    ("triangle_attention_checkpoint", "token_pair", "d_pair=64;d_hidden=128;n_head=4"),     # Boltz-2 / Chai-1 / Protenix-v1 template
    ("triangle_attention_checkpoint", "token_pair", "d_pair=256;d_hidden=256;n_head=4"),    # Chai-1 trunk
    ("triangle_attention_checkpoint", "token_pair", "d_pair=512;d_hidden=512;n_head=8"),    # IntelliFold2 trunk
    ("triangle_attention_checkpoint", "token_pair", "d_pair=256;d_hidden=256;n_head=8"),    # IntelliFold2 template
    ("triangle_multiplication", "token_pair", "d_pair=512;d_hidden=512"),                   # IntelliFold2 trunk
    ("triangle_multiplication", "token_pair", "d_pair=256;d_hidden=256"),                   # Chai-1 trunk, IntelliFold2 template
    ("triangle_multiplication", "token_pair", "d_pair=64;d_hidden=128"),                    # Protenix-v1 template (template_hidden_dim 128)
    ("transition", "token_pair", "d_hidden=64"),                                            # Boltz-2 / RoseTTAFold3 template, n=4
    ("transition", "token_pair", "d_hidden=512"),                                           # IntelliFold2 trunk
    ("transition", "token_pair", "d_hidden=512;n=2"),                                       # IntelliFold2 diffusion conditioning
    ("transition", "token_pair", "d_hidden=256;n=2"),                                       # Chai-1 trunk (factor 2), IntelliFold2 template
    ("transition", "token_single", "d_hidden=384;n=2"),                                     # Chai-1 single transition
    ("transition", "msa_token", "d_hidden=256;n=4"),                                        # IntelliFold2 MSA
    ("outer_product_mean", "msa_token", "d_msa=256;d_pair=512;d_hidden=32"),                # IntelliFold2
    ("outer_product_mean", "msa_token", "d_msa=64;d_pair=128;d_hidden=32;projection_bias=1"),  # RoseTTAFold3
    ("outer_product_mean", "msa_token", "d_msa=64;d_pair=256;d_hidden=64;groups=8"),        # Chai-1
    ("msa_pair_weighted_averaging", "msa_token", "d_msa=256;d_pair=512;n_head=8"),          # IntelliFold2
    ("msa_pair_weighted_averaging", "msa_token", "d_msa=64;d_pair=256;n_head=8;d_hidden=32"),  # Chai-1
    ("attention_pair_bias", "token_single", "d_single=384;d_pair=512;n_head=16"),           # IntelliFold2 pairformer
    ("attention_pair_bias", "token_single", "d_single=384;d_pair=256;n_head=16;gate_bias=1"),  # Chai-1 pairformer
    ("dit", "token_single", "d_single=768;d_cond=384;d_pair=512;n_head=16"),                # IntelliFold2 diffusion
    # what the 2026-11-03 constructor probe of every FoldForge model and every MiniWorld config built (~/miniworld-engine-scratch/regaudit)
    ("triangle_attention_checkpoint", "token_pair", "d_pair=128;d_hidden=128;n_head=4;transposed_bias=0"),             # AF3 / OpenFold3 trunk
    ("triangle_attention_heads", "token_pair", "n_head=4;d_hidden=128"),                                                  # Boltz-2 / OF3-preview2 / Protenix-v1 trunk
    ("triangle_attention_checkpoint", "token_pair", "d_pair=128;d_hidden=128;n_head=4;gate_bias=1;out_bias=1;transposed_bias=0"),  # RF3 trunk
    ("triangle_attention_checkpoint", "token_pair", "d_pair=256;d_hidden=256;n_head=4;transposed_bias=0;untransposed_output=1"),   # Chai-1 trunk
    ("triangle_attention_checkpoint", "token_pair", "d_pair=256;d_hidden=256;n_head=4;dual_output=1;transposed_bias=0"),          # Chai-1 head
    ("triangle_attention_checkpoint", "token_pair", "d_pair=512;d_hidden=512;n_head=8;transposed_bias=0"),             # IntelliFold2 trunk
    ("triangle_attention_checkpoint", "token_pair", "d_pair=256;d_hidden=256;n_head=8;transposed_bias=0"),             # IntelliFold2 template
    ("triangle_attention_checkpoint", "token_pair", "d_pair=64;d_hidden=128;n_head=4;transposed_bias=0;untransposed_output=1"),    # Chai-1 template
    ("triangle_attention_checkpoint", "token_pair", "d_pair=64;d_hidden=64;n_head=4;transposed_bias=0"),               # AF3 / OF3 template
    ("triangle_attention_checkpoint", "token_pair", "d_pair=64;d_hidden=256;n_head=4;gate_bias=1;out_bias=1;transposed_bias=0"),  # RF3 template
    ("triangle_multiplication_bidirectional", "token_pair", "d_pair=64;d_hidden=64"),                                     # MiniWorld template
    ("triangle_multiplication_bidirectional", "token_pair", "d_pair=128;d_hidden=128"),                                   # MiniWorld trunk
    ("outer_product_mean", "msa_token", "d_msa=64;d_pair=128;d_hidden=32;normalize_before_proj=0"),                       # AF3 / OF3 / Protenix-v1
    ("outer_product_mean", "msa_token", "d_msa=64;d_pair=128;d_hidden=32"),                                               # MiniWorld / Boltz-2
    ("outer_product_mean", "msa_token", "d_msa=64;d_pair=128;d_hidden=32;projection_bias=1;normalize_before_proj=0"),    # RF3
    ("outer_product_mean", "msa_token", "d_msa=128;d_pair=256;d_hidden=32;normalize_before_proj=0"),                      # Protenix-v2 / ESMFold2
    ("outer_product_mean", "msa_token", "d_msa=128;d_pair=384;d_hidden=32;normalize_before_proj=0"),                      # OpenDDE
    ("outer_product_mean", "msa_token", "d_msa=256;d_pair=512;d_hidden=32;normalize_before_proj=0"),                      # IntelliFold2
    ("outer_product_mean", "msa_token", "d_msa=64;d_pair=256;d_hidden=64;groups=8;normalize_before_proj=0"),              # Chai-1
    ("msa_pair_weighted_averaging", "msa_token", "d_msa=64;d_pair=128;n_head=8;d_hidden=32"),                            # MiniWorld / Boltz-2 / RF3
    ("local_dit", "atom_single", "d_single=128;d_cond=128;d_pair=16;n_head=4;cross_attention=1;parallel_kv=1;slide_window=1"),  # AF3 / OF3 atoms
    ("local_dit", "atom_single", "d_single=128;d_cond=128;d_pair=16;n_head=4;cross_attention=1;slide_window=1"),          # Protenix / OpenDDE fast
    ("local_dit", "atom_single", "d_single=128;d_cond=128;d_pair=16;n_head=4;cross_attention=1"),                         # Protenix / OpenDDE exact
    ("atom_dit_variant", "atom_single", "n_blocks=3;boltz2=1;pad_window=1"),                                              # Boltz-2 exact
    ("atom_dit_variant", "atom_single", "n_blocks=3;rosettafold3=1;pad_window=1"),                                        # RF3 exact
    ("swa_atom_attention", "atom_single", "d_model=128;n_heads=4;half_window=500000"),                                    # MiniWorld trunk configs
    ("swa_atom_transformer", "atom_single", "n_block=1;esmfold2=1;window=1000000"),                                        # MiniWorld trunk configs
    ("input_feature_embedder", "atom_single", "d_atom=128;d_token=384;n_head=4;n_block=3;window=1000000"),                # MiniWorld distogram
    ("input_feature_embedder", "atom_single", "d_atom=128;d_token=384;n_head=4;n_block=3;esmfold2=1;linear_pool=0;window=1000000"),  # MiniWorld large v238 / v240
    ("layernorm_native", "token_single", "d_norm=833;has_bias=1"),                                                        # MiniWorld diffusion
)


@pytest.mark.parametrize(("module", "stream", "dims"), RELEASED_SPEC_ROWS)
def test_every_released_spec_module_shape_is_a_row(module, stream, dims):
    want = {k: int(v) for k, v in (kv.split("=") for kv in dims.split(";"))}
    have = [r.dims for r in _rows(module) if r.stream == stream]
    assert want in have, (module, stream, dims)


#: A ``modules/`` directory whose rows carry another name (the directory holds the module the row builds).
ROW_NAME = {"mpnn": "protein_mpnn", "outer_product": "outer_product_mean", "pairformer": "pairformer_block",
            "triangle_attention": "triangle_attention_heads"}


def test_every_engine_module_has_a_row():
    """Every module of ``modules/`` is declared: what the registry does not name is not built, bundled or verified."""
    root = Path(module_registry.__file__).resolve().parents[1] / "modules"
    have = {r.module for r in module_registry.module_rows()}
    dirs = sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "__init__.py").is_file())
    missing = [d for d in dirs if ROW_NAME.get(d, d) not in have]
    assert not missing, missing


def test_the_sm90_plan_keeps_the_native_selectors():
    """The derivation records the selectors the hand-CUDA ops choose a config with: faking ``transition.cuda`` wholesale once dropped the
    Transition rows from the plan (a family's op that records a selector runs; ``derive._records_selector``)."""
    text = (ROOT / "src/miniworld_engine/autotune/plans/sm90/registry_kernel.csv").read_text()
    for kernel in ("transition_fwd_residual_sm90_cuda", "transition_bwd_residual_sm90_cuda", "trimul_fwd_sm90_cuda"):
        assert f"\n{kernel}," in text, kernel
