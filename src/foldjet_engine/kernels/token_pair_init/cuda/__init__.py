"""Per-architecture implementations of the token-pair initialisation kernels (``TokenPairInit``)."""


def prebuild(target, rows):
    """The build of the ``token_pair_init`` family (kernels/_prebuild.py): one shape-independent extension (``token_pair_init_sm90``, every size is a
    runtime launch parameter), H100 (sm_90a), and ``token_pair_init_sm80`` on an A100 (sm80/), built once when some requirements row (of ``TokenPairInit``, the module that launches it) needs
    the family."""
    from .sm90 import ext

    if rows and target.arch == "sm90":
        yield ext
    elif rows and target.arch in ("sm80", "sm120"):  # the A100 build of the same source (sm80/__init__.py; sm_120a on the RTX 5090: _sm120.py)
        from .sm80 import ext as sm80_ext

        yield sm80_ext
    elif rows and target.arch == "sm100":  # B200 (sm100/__init__.py): one cubin per forward-tile rows of the lengths
        import functools

        from . import sm100

        builds = {2}  # cubin(): tpi_fwd_w (the one-launch forward) and the backward (tpi_bwd / tpi_bwd_w / tpi_bwd_red)
        for r in rows:
            if r["mode"] == "train":  # weights / ids the one-launch forward does not take: sm100.forward -> cubin(fwd_rows(L))
                builds |= {sm100.fwd_rows(length) for length in r["lengths"]}
        for rf in sorted(builds):
            yield functools.partial(sm100.cubin, rf)
