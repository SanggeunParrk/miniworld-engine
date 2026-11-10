"""CUDA kernels of the DiT block variants, sm_90a, built on first use (``load_extension``), never at import: ``rows.cu`` (inference rows),
``train_rows.cu`` (training rows, forward and backward), ``edge.cu`` (the atom stacks' slid edge windows, training); ``train.py`` is the training
path (autograd Functions over them)."""

from __future__ import annotations

import functools
from pathlib import Path


_dir = Path(__file__).parent


@functools.lru_cache(maxsize=None)
def ext():
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="dit_variants_rows_sm90a",
        sources=[str(_dir / "rows.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            *gencodes("90a"),
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def ln_mod_rows(x, mods, L, eps, sig, *, xin=None):
    """xhat = LN(x) (no affine; x = xin first when given); for each (ms, mb, out) of ``mods`` (one to three): out = xhat * f(ms) + mb,
    f = sigmoid when ``sig`` else identity; ms / mb [L, D] bf16 table views (one row stride) indexed by row % L. D = 768 or 128."""
    ms, mb, out = (list(t) for t in zip(*mods, strict=True))
    ext().ln_mod_rows(xin, x, ms, mb, out, int(L), float(eps), bool(sig))


def res_rows(x, ya, ga, L, *, yt=None, gt=None, ms=None, mb=None, out_ln=None, final=None, eps=1e-5, sig=True):
    """x += sigmoid(ga) ya [+ sigmoid(gt) yt]; into ``final`` (bf16) when given, else back into x (fp32); then
    out_ln = LN(x) * f(ms) + mb when ``out_ln`` is given."""
    ext().res_rows(x, ya, ga, yt, gt, ms, mb, out_ln, final, int(L), float(eps), bool(sig))


def qk_ln_rows(q, k, wq, bq, wk, bk, eps):
    """In place: q <- LN(q) * wq + bq, k <- LN(k) * wk + bk over D (768 or 128) columns (bf16 row views; fp32 [D] affine)."""
    ext().qk_ln_rows(q, k, wq, bq, wk, bk, float(eps))


def gate_rows_inplace(o, g):
    """In place: o <- o * sigmoid(g) (bf16 row views)."""
    ext().gate_rows_inplace(o, g)


def glu_rows(ab, out, up):
    """out = silu(g) * v [* u] for ab = [g | v | u] (bf16)."""
    ext().glu_rows(ab, out, bool(up))


def ln_rows(z, out, eps=1e-5):
    """out = LN(z), no affine, z / out bf16 [rows, 128 | 256]: the row extensions' one LN-rows kernel
    (``conditioned_transition/cuda/token_dit_rows.cu`` ``layernorm_rows``)."""
    from miniworld_engine.kernels.conditioned_transition import cuda as rows

    rows.layernorm_rows(z, out, eps)



@functools.lru_cache(maxsize=None)
def train_ext():
    """The training row kernels (``train_rows.cu``): sm_90a, built on first use."""
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="dit_variants_train_rows_sm90a",
        sources=[str(_dir / "train_rows.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            *gencodes("90a"),
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


@functools.lru_cache(maxsize=None)
def edge_ext():
    """The atom stacks' slid-window edge kernels (``edge.cu``): sm_90a, built on first use."""
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    ensure_cuda_home()
    return load_extension(
        name="dit_variants_edge_sm90a",
        sources=[str(_dir / "edge.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            *gencodes("90a"),
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def prebuild(target, rows):
    """Every build of the DiT-variant family (kernels/_prebuild.py), by the (activation dtype, mode) pairs the requirements rows ask
    for (a ``master`` row computes in bfloat16 activations; the builds take no model dims). ``ext`` (``rows.cu``) is the bf16
    inference rows; ``train_ext`` / ``edge_ext`` serve the training path and the fp32 (TF32) inference (``infer32.py``). Which
    build serves which pair is this family's knowledge."""
    if target.arch == "sm80":   # no sm_80 port of these kernels (the A100 TokenDiTBoltz2 attention is augmented_attention's)
        return
    served = {("bfloat16" if r["dtype"] == "master" else r["dtype"], r["mode"]) for r in rows}
    if ("bfloat16", "eval") in served:
        yield ext
    if ("float32", "eval") in served or any(mode == "train" for _, mode in served):
        yield train_ext
        yield edge_ext
