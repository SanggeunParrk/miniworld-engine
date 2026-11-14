"""Hand-CUDA sm_90a bf16-wgmma kernels of the fused SWA atom DiT block, JIT-built by ``loader.py``.

``gen/gen_wgmma_bf16.py`` generates ``gen/wgmma_bf16.inc`` (the wgmma wrappers the three ``.cu`` files include); it is
kept beside its output and is not imported or shipped (``gen/`` is not a package and ``*.py`` is not package data)."""

#: stage of ``loader.SOURCES`` -> (activation dtypes it serves, modes it runs in); ``None`` = any. The ``bwd`` / weight-gradient
#: builds only run on the grad path, ``fp32`` / ``wgwg32`` only for float32 activations (a ``master`` row computes in bfloat16
#: activations). Which stage serves which dtype / mode is the dispatch's knowledge (``swa_dit/dispatch.py``); which dtypes and
#: modes are required comes from the requirements rows.
_STAGES = {
    "qkvg": (None, None),
    "qkvg_small": (None, None),
    "fwd": (None, None),
    "fwd_small": (None, None),
    "attn": (None, None),
    "slab": (None, None),
    "mod": (None, None),
    "wgrad": (None, None),
    "bwd": (None, {"train"}),
    "bwd_overwrite": (None, {"train"}),
    "wgwg": ({"bfloat16"}, {"train"}),
    "fp32": ({"float32"}, None),
    "wgwg32": ({"float32"}, {"train"}),
}


def prebuild(target, rows):
    """Every build of the SWA atom DiT family on ``target`` (kernels/_prebuild.py): on sm_90a each stage of ``loader.SOURCES`` that
    some requirements row (dtype, mode) asks for (``_STAGES``), or the sm_100a cubins (``sm100.CUBINS``, ``sm100.tf32_fwd.CUBINS``)."""
    import functools as ft

    if target.arch == "sm90":
        from . import loader

        served = {("bfloat16" if r["dtype"] == "master" else r["dtype"], r["mode"]) for r in rows}
        for which in loader.SOURCES:
            dtypes, modes = _STAGES[which]
            if any((dtypes is None or d in dtypes) and (modes is None or m in modes) for d, m in served):
                yield ft.partial(loader._build, which)
    elif target.arch == "sm80":
        # A100: the sm_80 stages of AtomDiTMiniWorld._forward_a100 (one extension: the bf16 stages and their backwards, the fp32 row
        # stages of sm80/composed.py), for every dtype and mode
        if rows:
            from . import sm80

            yield sm80._ext
    elif target.arch == "sm120":
        # RTX 5090: the sm_120a build of those sm_80 stages (``swa_dit_sm80_sm120``) and its fused bf16 inference stages within 99 KB
        # (``swa_dit_fused_sm120``); the sm_100a cubins below do not run there
        if rows:
            from . import sm80

            yield sm80._ext
            yield sm80._ext_sm120
    else:
        from .sm100 import CUBINS, cubin, tf32_fwd

        for name in CUBINS:
            yield ft.partial(cubin, name)
        for name in tf32_fwd.CUBINS:
            yield ft.partial(tf32_fwd.cubin, name)
