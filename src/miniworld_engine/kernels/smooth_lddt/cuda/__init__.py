"""Per-architecture implementations of the smooth lDDT kernels."""


def prebuild(target, rows):
    """The build of the smooth lDDT family (kernels/_prebuild.py): one cubin per arch (plain CUDA-core code, ``sm100.cubin(arch)``), shape-independent
    (every size is a runtime launch parameter), so it is built once when some requirements row (of the module that launches it) needs the family."""
    import functools as ft

    from .sm100 import cubin

    if rows:
        yield ft.partial(cubin, "sm_90a" if target.arch == "sm90" else "sm_100a")
