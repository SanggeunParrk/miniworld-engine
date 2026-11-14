"""H100 / B200 / A100 TriangleAttention CUDA: the loaders live in ``h100_sm90`` / ``h100_sm90_generic`` / ``h100_dh`` / ``h100_triattn`` / ``h100_bidir``
(sm90), ``b200_triattn`` (sm100) and ``sm80`` / ``sm80_wide`` / ``sm80_core2`` (sm80); this module holds the prebuild recipe."""

from __future__ import annotations


def prebuild(target, rows):
    """Every build of the TriangleAttention family on ``target`` (kernels/_prebuild.py): the loaders as the dispatch calls them
    (default schedules: the ``defs`` / ``rows`` arguments are experiment knobs no dispatch passes). From ``rows`` (the
    requirements): which of the family's kernels some module must serve (``kernels``, which carries their roles: inference-only,
    training-only, both) decides which loader is built, and the lengths (sm100: the rows per task of each length). The builds
    themselves are width- and length-generic (the shapes are launch arguments), so no other axis is specialized."""
    import functools as ft

    served = set().union(*(r["kernels"] for r in rows)) if rows else set()

    def needs(*names, like=""):
        """Some requirement kernel named in ``names`` (or containing ``like``) is served."""
        return any(n in served for n in names) or bool(like) and any(like in n for n in served)

    if target.arch == "sm90":
        from miniworld_engine.kernels._nvcc import cutlass_include

        from . import h100_dh, h100_sm90, h100_sm90_generic, h100_triattn

        # the sm90 loaders (h100_sm90 / h100_sm90_generic): attention cores, the module kernels around them, the backward
        if needs("triangle_attention_fwd_sm90"):
            yield h100_sm90._core
        if needs("triangle_attention_fwd_fp16_sm90"):
            yield h100_sm90._core_h16
        if needs(
            "triangle_attention_front_sm90",
            "triangle_attention_tail_sm90",
            "triangle_attention_pack_sm90",
            "triangle_attention_front_wide_sm90",
            "triangle_attention_pack_wide_sm90",
            "triangle_attention_gate_bwd_sm90",
            "triangle_attention_head_bwd_sm90",
            "triangle_attention_wgrad_sm90",
            "triangle_attention_wgrad_reduce_sm90",
            "triangle_attention_param_finish_sm90",
        ):
            yield h100_sm90._mod
        if needs("triangle_attention_bwd_kv_sm90", "triangle_attention_bwd_q_sm90", "triangle_attention_generic_assemble_bwd_sm90"):
            yield h100_sm90._bwd
        if needs(like="triangle_attention_generic_"):
            yield h100_sm90_generic._glue
            yield h100_sm90_generic._lt
        if needs("triangle_attention_bidir_gate_bwd_sm90"):
            from . import h100_bidir

            yield h100_bidir.ext
        if needs("triangle_attention_front_fused_sm90"):
            yield lambda: h100_sm90_generic._front_ext(cutlass_include())
        # the MSA row dropout's Philox keep-mask (pwa_train: ``extension("pack").drop_keep``)
        if needs("triangle_attention_drop_scale_sm90", "triangle_attention_graph_rng_write_sm90"):
            yield ft.partial(h100_triattn.extension, "pack")
        if needs(
            "triangle_attention_dh_fwd_sm90",
            "triangle_attention_dh_fwd_train_sm90",
            "triangle_attention_dh_bwd_kv_sm90",
            "triangle_attention_dh_bwd_q_sm90",
        ):
            yield h100_dh.ext
        if needs("triangle_attention_d64_fwd_sm90"):
            yield h100_dh.ext_d64
    elif target.arch == "sm100":
        from . import b200_triattn

        lengths = {n for r in rows for n in r["lengths"]}
        for rows_per_task in sorted({b200_triattn._rows_per_task(n) for n in lengths}):
            yield ft.partial(b200_triattn._core, rows_per_task)
        yield b200_triattn._mod
        yield b200_triattn._wide
    elif target.arch in ("sm80", "sm120"):  # the 5090 builds the same sources for sm_120a (sm80_arch)
        yield from _sm80(rows)


#: the classes whose A100 calls (some or all) take the variant path (AF3 bias orientation, projection biases, Chai-1 output layouts)
_SM80_VARIANTS = frozenset({"TriangleAttentionAF3", "TriangleAttentionRF3", "TriangleAttentionChai1", "TriangleAttentionChai1Confidence"})


def _sm80(rows):
    """The A100 builds (``kernels/integrations/triattn_sm80``) for the bf16 / master TriangleAttention rows whose geometry the A100 path takes
    (heads of 16 / 32 / 64 channels): the core extension (``sm80``: the fused d_pair-128 path and the core of the wide path), the wide path's
    row kernels (``sm80_wide``), the 16-channel-head core (``sm80_core2``), and the shape-gated helpers the A100 path reaches (the
    training front / backward pieces at their lengths, the d_pair-256 fused output at L128)."""
    from miniworld_engine.kernels._prebuild import _kv

    from . import (
        affine_training_sm80,
        b1_finalize_exact_sm80,
        dq_rows_candidate_sm80,
        native_db_nn_sm80,
        native_out_fused_sm80,
        native_wo_tn_sm80,
        sm80,
        sm80_core2,
        sm80_wide,
    )

    calls = {}
    for r in rows:
        if r["module"] == "BidirectionalTriangleAttention" and r["dtype"] != "float32":  # its glue and gate backward, built for sm_80
            from . import sm80_bidir

            calls[sm80_bidir.glue] = None
            if r["mode"] == "train":
                calls[sm80_bidir.ext] = None
            continue
        if not r["module"].startswith("TriangleAttention") or r["dtype"] == "float32":
            continue
        d = _kv(r["dims"])
        dp, h = int(d["d_pair"]), int(d["n_head"])
        hd = int(d.get("d_hidden") or dp) // h
        if hd not in (16, 32, 64):
            continue
        lengths, train = set(r["lengths"]), r["mode"] == "train"
        calls[sm80._ext] = None
        if hd == 64 or r["module"] in _SM80_VARIANTS:  # the variant path (sm80_wide.variant): the row kernels and the generalised core
            calls[sm80_wide._rows] = None
            calls[sm80_core2.ext] = None
            if hd == 64:
                continue
        if (dp, hd * h, h) != (128, 128, 4):
            calls[sm80_wide._rows] = None
        if hd == 16:
            calls[sm80_core2.ext] = None
        if train:
            calls[affine_training_sm80._ext] = None
            if lengths & {640, 768}:
                calls[dq_rows_candidate_sm80._ext] = None
                calls[native_wo_tn_sm80._ext] = None
            if 768 in lengths:
                calls[native_db_nn_sm80._ext] = None
            if 384 in lengths:
                calls[b1_finalize_exact_sm80._ext] = None
        elif (dp, h, hd) == (256, 8, 32) and 128 in lengths:
            calls[native_out_fused_sm80._ext] = None
    yield from calls
