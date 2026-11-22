"""A100 (sm_80) hand-CUDA OuterProductMean, forward and backward (bf16, d_hidden 32, d_msa 64 / 128 / 256, d_pair 128 / 256 / 384 / 512).

``OuterProductMean`` is LN(msa) -> two projections -> the outer product of the projected rows averaged over the MSA depth -> the 32 x 32 -> d_pair
projection. The outer product is one GEMM in the *grouped* layout, O[(i, c), (j, e)] = sum_s a[s, i, c] b[s, j, e] (M = N = 32 L, K = S): cuBLAS runs
it (and the two gradient GEMMs), and the hand kernels (``sm80/``) are everything around it:

  prologue       LN + both projections + mask -> A, B [S, 32 L] (s-major operands: O = A^T B runs on cuBLAS's fastest NT-class kernel), and the LN statistics
  epilogue       O -> the [i, j, (c, e)] permute, / n_ij, the 1024 -> d_pair projection, bias, pair residual
  dgrad          dz -> dzn = dz / n, dO in the grouped layout (no permuted copy), the bias-gradient partials
  dwo            dWo partials off the kept O (split over rows of i)
  prologue_bwd   dA | dB -> dm, the dWl / dWr / dgamma / dbeta partials (persistent CTAs)
  reduce_rows    fixed-order sums of the partial rows (no atomics: every gradient is bit-reproducible)

``integrations/opm_sm80.py`` drives these from ``OuterProductMean.forward``; the extension is built on first use (``load_extension``), never at import.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch


from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension
from miniworld_engine.kernels import _msa_card as _card

_dir = Path(__file__).parent / "sm80"

CH = 32  # d_hidden
D_MSA = (64, 128, 256)
D_PAIR = (128, 256, 384, 512)
SPLITS = {128: 24, 256: 12, 384: 8, 512: 6}  # dWo: splits over rows of i (CTAs ~ 192 per card)


@functools.lru_cache(maxsize=1)
def ext():
    """The extension (built on first use into ``TORCH_EXTENSIONS_DIR``)."""
    ensure_cuda_home()
    return load_extension(
        name=_card.name("opm_sm80"),
        sources=[str(_dir / "ops.cu")],
        extra_include_paths=[str(_dir)],
        extra_cuda_cflags=_card.cflags([
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-lineinfo",
            *gencodes("80"),
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ]),
        extra_cflags=["-std=c++17", "-O3"],
        extra_ldflags=["-lcublasLt"],  # the searched explicit GEMM algorithms (opm_lt_gemm.cuh)
        verbose=False,
    )


def lt_mm(a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool) -> torch.Tensor:
    """op(a) @ op(b) for the grouped outer-product GEMMs (row-major bf16): cuBLASLt with the searched explicit algorithm of the shape (``sm80/opm_lt_table.inc``), at::mm elsewhere."""
    return ext().lt_mm(a, b, trans_a, trans_b)


def supported(d_msa: int, d_hidden: int, d_pair: int) -> bool:
    return d_hidden == CH and d_msa in D_MSA and d_pair in D_PAIR


def _none(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t.new_empty((0,), dtype=dtype)


def prologue(
    m: torch.Tensor,
    mask: torch.Tensor | None,
    lnw: torch.Tensor,
    lnb: torch.Tensor,
    wl: torch.Tensor,
    wr: torch.Tensor,
    eps: float,
    save_stats: bool,
    variant: int | None = None,
    bl: torch.Tensor | None = None,
    br: torch.Tensor | None = None,
):
    """m [S, L, CM] bf16, mask [S, L] uint8 (or None), lnw / lnb fp32 [CM], wl / wr bf16 [32, CM], bl / br the projection biases fp32 [32] (or None) -> (A, B [S, 32 L] bf16, stats [S, L, 2] fp32 or an empty tensor)."""
    S, L, _ = m.shape
    a = m.new_empty((S, L * CH))
    b = m.new_empty((S, L * CH))
    stats = m.new_empty((S, L, 2), dtype=torch.float32) if save_stats else _none(m, torch.float32)
    if variant is None:
        variant = 0  # schedule experiments (tile buffers); 0 is the shipped one
    ext().prologue(
        m,
        _none(m, torch.uint8) if mask is None else mask,
        lnw,
        lnb,
        wl,
        wr,
        a,
        b,
        stats,
        eps,
        variant,
        _none(m, torch.float32) if bl is None else bl,
        _none(m, torch.float32) if br is None else br,
    )
    return a, b, stats


def epilogue(
    o: torch.Tensor,
    norm: torch.Tensor | None,
    norm_const: float,
    wo: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor | None,
    norm_first: bool,
    variant: int | None = None,
    dmin: float = 1.0,
    doff: float = 0.0,
) -> torch.Tensor:
    """O [32 L, 32 L] bf16, norm fp32 [L, L] (None: the constant ``norm_const``), wo bf16 [CZ, 1024], bias fp32 [CZ], residual bf16 [L, L, CZ] (or None) -> [L, L, CZ] bf16.
    The pair divisor is ``max(n, dmin) + doff`` (``opm_train.divisor_params``)."""
    length = o.shape[0] // CH
    out = o.new_empty((length, length, wo.shape[0]))
    if variant is None:
        # schedule experiments (ring depth / CTAs per SM); 0 is the shipped one.  The RTX 5090 at d_pair 256 from L 512: the wide CTA (16 warps,
        # all channels; its 99 KiB hold one 2-CTA schedule CTA, so the shipped overlap is gone there): 6-9 % faster, the same bits
        variant = 1 if _card.sm120() and wo.shape[0] == 256 and length >= 512 else 0
    ext().epilogue(
        o,
        _none(o, torch.float32) if norm is None else norm,
        norm_const,
        wo,
        bias,
        _none(o, torch.bfloat16) if residual is None else residual,
        out,
        int(norm_first),
        variant,
        float(dmin),
        float(doff),
    )
    return out


def dgrad(
    dz: torch.Tensor,
    norm: torch.Tensor | None,
    norm_const: float,
    wo: torch.Tensor,
    norm_first: bool,
    variant: int | None = None,
    dmin: float = 1.0,
    doff: float = 0.0,
):
    """dz [L, L, CZ] bf16 -> (dO [32 L, 32 L] bf16, dzn [L, L, CZ] bf16, dbo fp32 [CZ]); the pair divisor ``max(n, dmin) + doff``."""
    length, _, cz = dz.shape
    dO = dz.new_empty((length * CH, length * CH))
    dzn = torch.empty_like(dz)
    if variant is None:
        variant = 3 if _card.sm120() and cz in (256, 384) else 0  # RTX 5090 (99 KiB): the 64-pair CTA; else the shipped schedule
    bi = (
        2 if (variant == 3 and cz != 128) or cz == 512 else 4
    )  # variant 3: the 64-pair CTA (two per SM) at d_pair 256 / 384; d_pair 512 always
    parts = dz.new_empty((-(-length // 32) * -(-length // bi), cz), dtype=torch.float32)
    ext().dgrad(
        dz,
        _none(dz, torch.float32) if norm is None else norm,
        norm_const,
        wo,
        dO,
        dzn,
        parts,
        int(norm_first),
        variant,
        float(dmin),
        float(doff),
    )
    return dO, dzn, reduce_rows(parts, torch.float32)


def dwo(dzn: torch.Tensor, o: torch.Tensor, out_dtype: torch.dtype, variant: int = 0) -> torch.Tensor:
    """dWo [CZ, 1024] (``out_dtype``) = sum over the pairs of dzn[i, j, z] O[(i, c), (j, e)]."""
    length, _, cz = dzn.shape
    i_per = -(-length // SPLITS[cz])
    splits = -(-length // i_per)
    part = dzn.new_empty((splits, cz, CH * CH), dtype=torch.float32)
    ext().dwo(dzn, o, part, i_per, variant)
    return reduce_rows(part.view(splits, -1), out_dtype).view(cz, CH * CH)


def prologue_bwd(
    da: torch.Tensor,
    db: torch.Tensor,
    m: torch.Tensor,
    stats: torch.Tensor,
    mask: torch.Tensor | None,
    lnw: torch.Tensor,
    lnb: torch.Tensor,
    wl: torch.Tensor,
    wr: torch.Tensor,
    variant: int | None = None,
):
    """dA, dB [S, 32 L] bf16 (the cuBLAS products dA = B dO^T, dB = A dO), m [S, L, CM], stats [S, L, 2], ... -> (dm bf16, dWl, dWr [32, CM], dgamma, dbeta [CM]) all fp32 but dm."""
    S, L, cm = m.shape
    if variant is None:
        variant = 0  # schedule experiments (buffers / CTAs per SM); 0 is the shipped one
    ntile = -(-S // (32 if cm == 256 else 64)) * -(-L // 2)  # d_msa 256: the wide kernel's tiles of 32 rows x 2 tokens
    nblk = min(
        ntile,
        torch.cuda.get_device_properties(m.device).multi_processor_count * (1 if variant == 1 or cm != 64 else 2),
    )
    dm = torch.empty_like(m)
    part = m.new_empty((nblk, 66 * cm), dtype=torch.float32)
    ext().prologue_bwd(
        da,
        db,
        m,
        stats,
        _none(m, torch.uint8) if mask is None else mask,
        lnw,
        lnb,
        wl,
        wr,
        dm,
        part,
        variant,
    )
    total = reduce_rows(part, torch.float32)
    return (
        dm,
        total[: 32 * cm].view(32, cm),
        total[32 * cm : 64 * cm].view(32, cm),
        total[64 * cm : 65 * cm],
        total[65 * cm :],
    )


def reduce_rows(part: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The rows of an fp32 [R, W] partial buffer summed in order (W a multiple of 4) -> [W] in ``dtype`` (fp32 or bf16)."""
    out = part.new_empty((part.shape[1],), dtype=dtype)
    ext().reduce_rows(part, out)
    return out


# ------------------------------------------------------------------------------------------------------------- RTX 5090: row chunks
#: Past ROWS_TOKENS pairs the 5090's inference forms the grouped outer product per chunk of rows i (``O`` is 2 KiB a pair: 14 GiB at L 2688), a fixed function
#: of the shape; ROWS_BYTES bounds one chunk's O.
ROWS_TOKENS = 2048 * 2048
ROWS_BYTES = 1 << 30


@functools.lru_cache(maxsize=1)
def rows_ext():
    """The sm_120a-only epilogue launcher over a chunk of rows (``sm80/epilogue_rows_sm120.cu``)."""
    ensure_cuda_home()
    return load_extension(
        name="opm_epilogue_rows_sm120",
        sources=[str(_dir / "epilogue_rows_sm120.cu")],
        extra_include_paths=[str(_dir)],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "-gencode=arch=compute_120a,code=sm_120a",
            _card.DEFINE,
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ],
        extra_cflags=["-std=c++17", "-O3"],
        verbose=False,
    )


def rows_chunk(length: int) -> int:
    """Rows i per chunk of the row-chunked inference: the largest multiple of 32 whose O fits ROWS_BYTES."""
    return max(32, (ROWS_BYTES // (2048 * length)) // 32 * 32)


def epilogue_rows(
    a: torch.Tensor,
    b: torch.Tensor,
    norm: torch.Tensor | None,
    norm_const: float,
    wo: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor | None,
    norm_first: bool,
    dmin: float = 1.0,
    doff: float = 0.0,
) -> torch.Tensor:
    """``epilogue(A^T B, ...)`` on chunks of rows i (A, B [S, 32 L] from ``prologue``; norm [L, L] or None, residual [L, L, CZ] or None) -> [L, L, CZ] bf16: each chunk's
    O_c = A[:, 32 i0 : 32 i1]^T B (cuBLAS) and the epilogue kernel over its rows."""
    length = a.shape[1] // CH
    out = a.new_empty((length, length, wo.shape[0]))
    step = rows_chunk(length)
    ext_rows = rows_ext()
    empty_f, empty_b = _none(a, torch.float32), _none(a, torch.bfloat16)
    for i0 in range(0, length, step):
        i1 = min(length, i0 + step)
        i0 = i1 - -(-(i1 - i0) // 4) * 4  # the kernel takes rows in groups of 4: a short last chunk starts earlier (its overlap rewrites the same values)
        o = torch.mm(a[:, i0 * CH : i1 * CH].t(), b)
        ext_rows.epilogue_rows(
            o,
            empty_f if norm is None else norm[i0:i1],
            norm_const,
            wo,
            bias,
            empty_b if residual is None else residual[i0:i1],
            out[i0:i1],
            int(norm_first),
            float(dmin),
            float(doff),
        )
        del o
    return out
