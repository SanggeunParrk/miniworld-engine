# vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/cuda/layernorm/__init__.py
"""CUDA implementation of LayerNorm."""

from pathlib import Path


from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

_dir = Path(__file__).parent


def _build(config):
    """Compile the extension. Called on first attribute access, never at import.

    Importing this package used to compile CUDA for four architectures. Nothing needs that at
    import time -- every consumer (`kernels/drivers/layernorm.py`, `kernels/checks/layernorm.py`)
    already imports the name inside a function body -- and paying it eagerly is what makes an
    unrelated `pkgutil.walk_packages` sweep, like `dev audit`'s import check, build kernels.
    `ensure_cuda_home()` moves in here with it: it mutates os.environ, which an import should not.
    """
    ensure_cuda_home()
    return load_extension(
        name=f"layer_norm_cuda_w{config['warps']}_b{config['min_blocks']}",
        sources=[str(_dir / "layer_norm_cuda_kernel.cu")],
        # Explicit -gencode so the JIT build never relies on torch's arch autodetect
        # (which can misreport the device, e.g. "Unknown CUDA arch (10.1)" on H100).
        # Arch list filtered against what the local nvcc actually supports -- see kernels/_nvcc.py.
        # A hard-coded compute_90 made this build fail outright when PATH resolved nvcc to CUDA 11.7.
        extra_cuda_cflags=[*host_flags(), "-O3", "--use_fast_math",
                           f"-DMW_LN_WARPS={config['warps']}",
                           f"-DMW_LN_MIN_BLOCKS={config['min_blocks']}",
                           *gencodes("80", "90", "100", ptx=("100",))],
        verbose=False,
    )


_EXTENSIONS = {}


def _ext(config):
    key = (config["warps"], config["min_blocks"])
    if key not in _EXTENSIONS:
        _EXTENSIONS[key] = _build(config)
    return _EXTENSIONS[key]


def __getattr__(name):
    if name != "layer_norm_cuda":
        raise AttributeError(name)
    from types import SimpleNamespace
    return SimpleNamespace(layer_norm_fwd=layer_norm_fwd_cuda, layer_norm_bwd=layer_norm_bwd_cuda)


def layer_norm_fwd_cuda(x, weight, bias, eps=1e-5, *, config=None):
    from miniworld_engine.autotune.hopper_cuda_config import layernorm_candidates
    from miniworld_engine.autotune.native import choose_config, tensor_key
    grid = layernorm_candidates("fwd", x.shape[-1], x.element_size())
    if config is None:
        config = choose_config("layernorm_fwd_cuda", grid, dtype=str(x.dtype),
                               bucket=tensor_key(x, weight, bias, extra=(eps,)),
                               device_index=x.device.index,
                               run=lambda c: layer_norm_fwd_cuda(x, weight, bias, eps, config=c))
    if config not in grid:
        raise ValueError("invalid LayerNorm forward configuration")
    # Forward is scalar, so this extension's backward launch bounds are irrelevant.
    ext_config = layernorm_candidates("compile", x.shape[-1], x.element_size())[0]
    return _ext(ext_config).layer_norm_fwd(x, weight, bias, eps, config["block"])


def layer_norm_bwd_cuda(dy, x, weight, mean, rstd, row_scale=None, *, config=None, residual=None):
    """Configured backward of LN(x)*row_scale, including affine gradients."""
    from miniworld_engine.autotune.hopper_cuda_config import layernorm_candidates
    from miniworld_engine.autotune.native import choose_config, tensor_key
    if residual is not None and (residual.shape != x.shape or residual.dtype != x.dtype or residual.device != x.device):
        raise ValueError("residual must match x shape, dtype and device")
    grid = layernorm_candidates("bwd", x.shape[-1], x.element_size())
    if not grid:
        # The vectorized CUDA kernel has an alignment contract. Preserve arbitrary
        # widths with scalar tensor math rather than launching an out-of-bounds load.
        n = x.shape[-1]
        xf, grad = x.reshape(-1, n).float(), dy.reshape(-1, n).float()
        if row_scale is not None:
            grad = grad * row_scale.reshape(-1, 1).float()
        xhat = (xf - mean.reshape(-1, 1)) * rstd.reshape(-1, 1)
        weighted = grad * weight.float()
        dx = (weighted - weighted.mean(-1, keepdim=True)
              - xhat * (weighted * xhat).mean(-1, keepdim=True)) * rstd.reshape(-1, 1)
        dx = dx.reshape_as(x).to(x.dtype)
        if residual is not None:
            dx = dx + residual
        return dx, (grad * xhat).sum(0).to(weight.dtype), grad.sum(0).to(weight.dtype)
    if config is None:
        config = choose_config("layernorm_bwd_split_cuda", grid, dtype=str(x.dtype),
                               bucket=tensor_key(dy, x, weight, mean, rstd, row_scale, residual),
                               device_index=x.device.index,
                               run=lambda c: layer_norm_bwd_cuda(dy, x, weight, mean, rstd,
                                                                row_scale, config=c, residual=residual))
    if config not in grid:
        raise ValueError("invalid LayerNorm backward configuration")
    return _ext(config).layer_norm_bwd(dy, x, weight, mean, rstd, row_scale,
                                     config["waves"], config["reduce_block"], config["tx_bytes"], residual)
