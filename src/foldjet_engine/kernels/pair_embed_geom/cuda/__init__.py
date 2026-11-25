"""Per-architecture implementations of the backbone pair-embedding kernels (``PairEmbedGeom`` / ``PairEmbedCA``)."""


def prebuild(target, rows):
    """The build of the ``pair_embed_geom`` family (kernels/_prebuild.py): one shape-independent extension (``pair_embed_geom_sm90``, every size is a
    runtime launch parameter; the encoder and the decoder pair, the table gradient and the LayerNorm backward are entry points of it), H100 (sm_90a),
    and ``pair_embed_geom_sm80`` on an A100 (sm80/), built once when some requirements row (of ``PairEmbedGeom`` / ``PairEmbedCA``, the modules that launch it) needs the family."""
    from .sm90 import ext

    if rows and target.arch == "sm90":
        yield ext
        from .sm80 import ext as sm80_ext  # the H100 inference forwards (sm80/__init__.py: this source built for sm_90a, pair_embed_geom_sm90gt2)

        yield sm80_ext
    elif rows and target.arch in ("sm80", "sm120"):  # the A100 build of the same source (sm80/__init__.py; sm_120a on the RTX 5090: _sm120.py)
        from .sm80 import ext as sm80_ext

        yield sm80_ext
    elif rows and target.arch == "sm100":  # the B200 build (sm100/__init__.py: the A100 source + the tcgen05 table gradient)
        from .sm100 import ext as sm100_ext

        yield sm100_ext
