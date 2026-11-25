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
        from foldjet_engine.kernels._nvcc import cutlass_include

        from . import h100_dh, h100_sm90, h100_sm90_generic, h100_triattn

        # the sm90 loaders (h100_sm90 / h100_sm90_generic): attention cores, the module kernels around them, the backward
        if needs("triangle_attention_fwd_sm90"):
            yield h100_sm90._core
        if needs("triangle_attention_fwd_fp16_sm90"):
            yield h100_sm90._core_h16
            # the opt-in FP8 PV route (docs/fp8.md): the same core with the e4m3 P / V^T hi plane (TA_F8PV) or the hi + lo planes (+ TA_F8HL)
            yield ft.partial(h100_sm90._core_h16, ("TA_F8PV=1",))
            yield ft.partial(h100_sm90._core_h16, ("TA_F8PV=1", "TA_F8HL=1"))
            # its e4m3 q / k form (TA_F8QK, the fused d_pair-128 route; the front's QK8 instances are in ``_mod``)
            yield h100_sm90._core_f8qk
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
        if needs(
            "triangle_attention_bidir_gate_bwd_sm90",
            "triangle_attention_bidir_logits_sm90",
            "triangle_attention_bidir_tail_sm90",
            "triangle_attention_bidir_gate_bwd_fused_sm90",
        ):
            from . import h100_bidir

            yield h100_bidir.ext
        if needs("triangle_attention_front_fused_sm90"):
            yield lambda: h100_sm90_generic._front_ext(cutlass_include())
            yield h100_sm90_generic._front_y16  # the fp16-exact variant (same source, MW_F3_Y16)
            # the opt-in FP8 PV route (docs/fp8.md): the fronts that write the e4m3 V^T planes (hi: MW_F3_VTW=1, hi + lo: 2), bf16 and fp16-exact
            for vtw in (1, 2):
                yield ft.partial(h100_sm90_generic._front, (*h100_sm90_generic.FRONT_DEFS, f"MW_F3_VTW={vtw}"))
                yield ft.partial(h100_sm90_generic._front_y16, (f"MW_F3_VTW={vtw}",))
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
            yield h100_dh.ext_d64_f8  # the opt-in FP8 PV variant (-DTE_F8PV=1, docs/fp8.md; Chai-1 trunk / confidence)
        if needs("triangle_attention_d64_fwd_train_sm90"):
            yield h100_dh.ext_d64t
    elif target.arch == "sm100":
        from . import b200_bidir, b200_triattn  # b200_bidir: the bias-only forms' glue + gate backward

        lengths = {n for r in rows for n in r["lengths"]}
        for rows_per_task in sorted({b200_triattn._rows_per_task(n) for n in lengths}):
            yield ft.partial(b200_triattn._core, rows_per_task)
        # the inference build with the bias pre-scaled by log2 e (b200_triattn.HB_INFER: lengths above the L128 kernel's)
        for rows_per_task in sorted({b200_triattn._rows_per_task(n) for n in lengths if n > b200_triattn.TOK}):
            yield ft.partial(b200_triattn._core, rows_per_task, True)
            if b200_triattn.QS_INFER:
                yield ft.partial(b200_triattn._core, rows_per_task, True, qs=True)   # q pre-scaled by scale * log2 e (b200_triattn.QS_INFER)
            yield ft.partial(b200_triattn._core, rows_per_task, True, True)   # 16-channel heads (b200_triattn.D16_INFER)
            yield ft.partial(b200_triattn._core, rows_per_task, True, True, d16c=True)   # compact (b200_triattn.D16C_INFER)
            if b200_triattn.QS_INFER and rows_per_task >= 3:   # the QS form of the 16-channel heads (above b200_triattn.QS_MIN_L16 = 256: three rows a task)
                yield ft.partial(b200_triattn._core, rows_per_task, True, True, d16c=False, qs=True)
                yield ft.partial(b200_triattn._core, rows_per_task, True, True, d16c=True, qs=True)
        yield b200_triattn._mod
        yield b200_triattn._pack
        yield b200_triattn._wide
        # the wide path's per-head-width cores (b200_triattn.slot: 64 -> _core64, 96 / 128 -> _core128) for the head widths of the
        # TriangleAttention* rows (d_hidden, default d_pair, over n_head); the other classes' triangle attention has 32-channel heads
        from foldjet_engine.kernels._prebuild import _kv

        slots = set()
        for r in rows:
            if r["module"].startswith("TriangleAttention"):
                d = _kv(r["dims"])
                slots.add(b200_triattn.slot(int(d.get("d_hidden", d["d_pair"])) // int(d["n_head"])))
        if 64 in slots:
            yield b200_triattn._core64
            yield ft.partial(b200_triattn._core64, True)   # the inference build (b200_triattn.HB64_INFER)
            if b200_triattn.QS_INFER:
                yield ft.partial(b200_triattn._core64, True, None, True)   # its QS form (q pre-scaled; b200_triattn.QS_MIN_L64)
        if 128 in slots:
            yield b200_triattn._core128
        yield b200_triattn._l128  # d_pair 64 / 128 at L128: the one-kernel path
        yield b200_triattn._tallk  # core backward, key-side tail
        yield b200_bidir.glue
        yield b200_bidir.ext
        yield b200_bidir.lnb   # (N1) the bias-only LayerNorm backward at C = 128
        yield b200_bidir.ltmm  # (N1) tuned cublasLt tall-K GEMMs of the bias-only backward
        from . import b200_fp8

        yield b200_fp8._f8  # the opt-in e4m3 P / V core, head dim 32 (docs/b200/fp8.md "TriAttn"; b200_fp8.route, off unless fp8.enable_triattn)
        yield ft.partial(b200_fp8._f8, ("-DTF_D=64",))  # ... and its 64-channel-head build (the whitelist kind "d64")
        yield ft.partial(b200_fp8._f8, ("-DTF_Q8=1",))  # ... and the e4m3 q / k build (docs/b200/fp8.md "TriAttn e4m3 q / k": the front-written rows; the standalone q / k pass lives in it too)
        yield ft.partial(b200_fp8._f8, ("-DTF_D=64", "-DTF_Q8=1"))  # (the 64-channel-head form: built by the prebuild, gate closed)
    elif target.arch in ("sm80", "sm120"):  # the 5090 builds the same sources for sm_120a (sm80_arch)
        yield from _sm80(rows)
        if target.arch == "sm120":
            from . import sm120_fp8

            yield sm120_fp8._ext  # the opt-in e4m3 core of the RTX 5090 (docs/fp8.md "RTX 5090": sm120_fp8.route, off unless fp8.enable_triattn): no JIT on its first call


#: the classes whose A100 calls (some or all) take the variant path (AF3 bias orientation, projection biases, Chai-1 output layouts)
_SM80_VARIANTS = frozenset({"TriangleAttentionAF3", "TriangleAttentionRF3", "TriangleAttentionChai1", "TriangleAttentionChai1Confidence"})


def _sm80(rows):
    """The A100 builds (``kernels/integrations/triattn_sm80``) for the bf16 / master TriangleAttention rows whose geometry the A100 path takes
    (heads of 16 / 32 / 64 channels): the core extension (``sm80``: the fused d_pair-128 path and the core of the wide path), the wide path's
    row kernels (``sm80_wide``), the 16-channel-head core (``sm80_core2``), and the shape-gated helpers the A100 path reaches (the
    training front / backward pieces at their lengths, the d_pair-256 fused output at L128)."""
    from foldjet_engine.kernels._prebuild import _kv

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
