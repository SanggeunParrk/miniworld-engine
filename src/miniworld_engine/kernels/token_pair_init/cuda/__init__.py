"""Per-architecture implementations of the token-pair initialisation kernels (``TokenPairInit``)."""


def prebuild(target, rows):
    """The build of the ``token_pair_init`` family (kernels/_prebuild.py): one shape-independent extension (``token_pair_init_sm90``, every size is a
    runtime launch parameter), H100 (sm_90a), and ``token_pair_init_sm80`` on an A100 (sm80/), built once when some requirements row (of ``TokenPairInit``, the module that launches it) needs
    the family."""
    from .sm90 import ext

    if rows and target.arch == "sm90":
        yield ext
    elif rows and target.arch == "sm80":  # the A100 build of the same source (sm80/__init__.py)
        from .sm80 import ext as sm80_ext

        yield sm80_ext
