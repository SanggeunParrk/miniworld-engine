"""H100 / B200 TriangleAttention CUDA: the loaders live in ``h100_sm90`` / ``h100_sm90_generic`` / ``h100_dh`` / ``h100_triattn``
(sm90) and ``b200_triattn`` (sm100); this module holds the prebuild recipe."""
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
        if needs("triangle_attention_front_sm90", "triangle_attention_tail_sm90", "triangle_attention_pack_sm90",
                 "triangle_attention_front_wide_sm90", "triangle_attention_pack_wide_sm90", "triangle_attention_gate_bwd_sm90",
                 "triangle_attention_head_bwd_sm90", "triangle_attention_wgrad_sm90", "triangle_attention_wgrad_reduce_sm90",
                 "triangle_attention_param_finish_sm90"):
            yield h100_sm90._mod
        if needs("triangle_attention_bwd_kv_sm90", "triangle_attention_bwd_q_sm90", "triangle_attention_generic_assemble_bwd_sm90"):
            yield h100_sm90._bwd
        if needs(like="triangle_attention_generic_"):
            yield h100_sm90_generic._glue
            yield h100_sm90_generic._lt
        if needs("triangle_attention_front_fused_sm90"):
            yield lambda: h100_sm90_generic._front_ext(cutlass_include())
        # the MSA row dropout's Philox keep-mask (pwa_train: ``extension("pack").drop_keep``)
        if needs("triangle_attention_drop_scale_sm90", "triangle_attention_graph_rng_write_sm90"):
            yield ft.partial(h100_triattn.extension, "pack")
        if needs("triangle_attention_dh_fwd_sm90", "triangle_attention_dh_fwd_train_sm90", "triangle_attention_dh_bwd_kv_sm90",
                 "triangle_attention_dh_bwd_q_sm90"):
            yield h100_dh.ext
    elif target.arch == "sm100":
        from . import b200_triattn

        lengths = {n for r in rows for n in r["lengths"]}
        for rows_per_task in sorted({b200_triattn._rows_per_task(n) for n in lengths}):
            yield ft.partial(b200_triattn._core, rows_per_task)
        yield b200_triattn._mod
        yield b200_triattn._wide
