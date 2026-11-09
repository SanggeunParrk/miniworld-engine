"""Per-architecture implementations of the token-pair initialisation kernels (``TokenPairInit``)."""


def prebuild(target, rows):
    """The build of the ``token_pair_init`` family (kernels/_prebuild.py): one shape-independent extension (``token_pair_init_sm90``, every size is a
    runtime launch parameter), H100 (sm_90a) only, built once when some requirements row (of ``TokenPairInit``, the module that launches it) needs
    the family."""
    from .sm90 import ext

    if rows and target.arch == "sm90":
        yield ext
