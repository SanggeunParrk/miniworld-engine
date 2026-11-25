"""Per-architecture implementations of the pair-embedding + adaptive-LayerNorm kernels (``PairEmbedAdaLN``)."""


def prebuild(target, rows):
    """The build of the ``pair_embed_adaln`` family (kernels/_prebuild.py): one shape-independent extension (``pair_embed_adaln_sm90``, every size is a
    runtime launch parameter), H100 (sm_90a), and ``pair_embed_adaln_sm80`` on an A100 (sm80/), built once when some requirements row (of ``PairEmbedAdaLN``, the module that launches it) needs
    the family."""
    from .sm90 import ext

    if rows and target.arch == "sm90":
        yield ext
    elif rows and target.arch in ("sm80", "sm120"):  # the A100 build of the same source (sm80/__init__.py; sm_120a on the RTX 5090: _sm120.py)
        from .sm80 import ext as sm80_ext

        yield sm80_ext
    elif rows and target.arch == "sm100":  # the B200 build (sm100/__init__.py: the A100 source + the tcgen05 table gradient)
        from .sm100 import ext as sm100_ext

        yield sm100_ext
