# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/cuda/layernorm/__init__.py
"""CUDA implementation of LayerNorm."""

from pathlib import Path


from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

_dir = Path(__file__).parent


def _build():
    """Compile the extension. Called on first attribute access, never at import.

    Importing this package used to compile CUDA for four architectures. Nothing needs that at
    import time -- every consumer (`kernels/drivers/layernorm.py`, `kernels/checks/layernorm.py`)
    already imports the name inside a function body -- and paying it eagerly is what makes an
    unrelated `pkgutil.walk_packages` sweep, like `dev audit`'s import check, build kernels.
    `ensure_cuda_home()` moves in here with it: it mutates os.environ, which an import should not.
    """
    ensure_cuda_home()
    return load_extension(
        name="layer_norm_cuda",
        sources=[str(_dir / "layer_norm_cuda_kernel.cu")],
        # Explicit -gencode so the JIT build never relies on torch's arch autodetect
        # (which can misreport the device, e.g. "Unknown CUDA arch (10.1)" on H100).
        # Arch list filtered against what the local nvcc actually supports -- see kernels/_nvcc.py.
        # A hard-coded compute_90 made this build fail outright when PATH resolved nvcc to CUDA 11.7.
        extra_cuda_cflags=[*host_flags(), "-O3", "--use_fast_math",
                           *gencodes("80", "90", "100", ptx=("100",))],
        verbose=False,
    )


def _ext():
    """The extension, built on first use and cached in globals().

    Functions in THIS module must call this, not the bare name `layer_norm_cuda`. A module-level
    `__getattr__` is consulted for `module.attr` from outside; a bare global lookup inside the
    module is not, so `return layer_norm_cuda.layer_norm_bwd(...)` would raise NameError at call
    time. (`tests/test_no_undefined_names.py` catches exactly that, and did.)
    """
    ext = globals().get("layer_norm_cuda")
    if ext is None:
        ext = globals()["layer_norm_cuda"] = _build()
    return ext


def __getattr__(name: str):
    """PEP 562: build on first use of `layer_norm_cuda`, for importers outside this module."""
    if name != "layer_norm_cuda":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _ext()


def layer_norm_bwd_cuda(dy, x, weight, mean, rstd, row_scale=None):
    """Standalone CUDA LayerNorm backward candidate.

    Signature matches compile_native._bwd_persistent_impl:
    (dy, x, weight, mean, rstd) -> (dx, dw, db).

    Optional ``row_scale`` [M] folds a per-row scale into the backward of
    ``y = LN(x) * row_scale`` (AF triangle pair-mask). The incoming grad is
    scaled by row_scale per row; dx/dw/db all follow (matches the triton path).
    """
    return _ext().layer_norm_bwd(dy, x, weight, mean, rstd, row_scale)
