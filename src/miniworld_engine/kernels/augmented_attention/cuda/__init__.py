"""sm_90a pair-bias attention core with bf16 operands, fp32 accumulation, forward AND backward.

The token DiT's attention (16 heads x 48, one pair bias per head shared by all A augmented samples) run as four
hand-written wgmma/TMA kernels, developed in the research branch ``research/token-dit-overlap``
(``archive/experiments-20260928:experiments/token_dit_train``). Measured on an H100 SXM at A = 48 (do_bench, L2 evicted), against this engine's bf16
Triton core and torch SDPA:

    op                    L=384      L=768
    forward               107 us     334 us     (engine bf16 659 us at L768; no-math pattern floor 273 us)
    backward              451 us    1507 us     (engine bf16 ~3977 us at L768)
    forward + backward    558 us    1841 us     (engine bf16 4636, SDPA 4543 us at L768)

Token DiT block fwd+bwd (24 blocks, per-block checkpointing, pair-bias hoist, fp32 model): 22.22 -> 18.77 ms/block at
L768, 10.70 -> 9.52 at L384; fwd+bwd peak memory 8.50 -> 6.59 GB at L768 (the Triton backward writes dbias per sample).

The kernels:
  attn_fwd.cu   O = softmax(q k^T / sqrt 48 + bias) v and the row LSE (log2 units). The bias is seeded into the score
                accumulator and the QK wgmma accumulates onto it; q is pre-scaled by log2 e / sqrt 48 so every
                exponential is one ex2; the running max moves only when a block exceeds it by 8 (log2 units).
  attn_dqb.cu   dQ and dbias. The CTA's three warpgroups are three SAMPLES of one query tile; their dS tiles meet in
                shared memory and one warpgroup issues the L2 reds, so dbias costs a third of the atomics.
  attn_dkv.cu   dK and dV (built with DBIAS=0 next to attn_dqb; DBIAS=1 adds dbias with per-sample L2 atomics, the
                fallback when A is not a multiple of 3, paired with attn_dq.cu).
  attn_dq.cu    dQ alone (the fallback's partner).

Numerics: the operands are rounded to bf16 (q after the scale, the bias after the log2 e scale), P and dS are bf16 for
the tensor core, every sum is fp32. Against an fp64 truth, O / dq / dk / dv / dbias sit at the bf16-input-rounding
floor (O 4.7e-3 against a floor of 4.4e-3 at unit-variance inputs); in the token DiT block the error is that of the
engine's bf16 Triton core (tests/gpu/sm90/test_augmented_attention_bf16_sm90_gpu.py).

The gate is the callers' own (``integrations/token_dit_train.serves``, ``dit_variants``' ``serves``, ``integrations/attention_pair_bias_b200.refusal``):
this module holds the launchers (``forward`` / ``backward`` / ``gated_inference`` and their TF32 forms, ``ApbInferenceCore`` /
``apb_*`` of AttentionPairBias) and the build recipes (``prebuild``).
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch
import triton
import triton.language as tl

from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

_dir = Path(__file__).parent
_tmn = _dir.parent.parent / "transition" / "cuda" / "anthropic_v5"  # tmn_kernels.cuh (TMA / mbarrier helpers)

H, D = 16, 48
# AttentionPairBias (B = 1): key splits per 64 query rows -- APB_KS warpgroups, each walking every APB_KS-th key block, merged in
# shared memory. Measured on H100 (L768, 8 x 48): 2 splits take 1-2 us off each core (about 0.5 % of the step); the default stays 1.
_APB_KS = {"fwd": 1, "inf": 1, "dq": 1, "dkv": 1}


@functools.lru_cache(maxsize=None)
def _ext(name: str, defs: tuple[str, ...] = ()):
    ensure_cuda_home()
    return load_extension(
        name=f"augattn_sm90_{name}" + "".join("_" + d.replace("=", "").replace(".", "p").replace("-", "m") for d in defs),
        sources=[str(_dir / f"{name}.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-std=c++17",
            "-O3",
            *gencodes("90a"),
            f"-I{_tmn}",
            "--expt-relaxed-constexpr",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            *("-D" + d for d in defs),
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


@functools.lru_cache(maxsize=None)
def _rows_ext():
    """``pair_rows.cu``: the fp32 pair -> bias pass the token DiT's TF32 inference builds its 256-wide bias with
    (``conditioned_transition/cuda/__init__.py``, ``pair_bias_rows32``)."""
    ensure_cuda_home()
    return load_extension(
        name="augattn_sm90_pair_rows",
        sources=[str(_dir / "pair_rows.cu")],
        extra_cuda_cflags=[*host_flags(), "-std=c++17", "-O3", *gencodes("90a"), "--expt-relaxed-constexpr"],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


class ApbInferenceCore:
    """Hopper gated APB core: packed projections in, gated attention over Q out.

    Weight packing supplies exp2-scaled Q and bias. This entry point launches only
    hand-written TMA/WGMMA CUDA and keeps the B200 inference tensor lifetimes.
    """

    def __init__(self, device_index: int, heads: int, d: int = 384):
        from .apb import GEOMETRY, width

        _, dh = GEOMETRY[(heads, d)]
        self.heads = heads
        with torch.cuda.device(device_index):
            self.extension = _ext(
                "attn_fwd",
                (
                    f"DHP={dh}",
                    f"ROW_WIDTH={width(heads, d)}",
                    "NWG=1",
                    f"APB_KS={_APB_KS['inf']}",
                ),
            )

    def __call__(self, qkvg: torch.Tensor, bias: torch.Tensor):
        if bias.shape[0] != self.heads or qkvg.shape[0] != bias.shape[1]:
            raise ValueError("Hopper APB inference requires one sample and head-major bias")
        self.extension.attn_inf(qkvg, bias, 0, 1, 1)
        return qkvg


def _apb_training_defs(heads: int, d: int):
    from .apb import GEOMETRY, width

    real, padded = GEOMETRY[(heads, d)]
    return (
        f"DHP={padded}",
        f"ROW_WIDTH={width(heads, d)}",
        "NWG=1",
        "APB_NATURAL=1",
        f"RSQDV={real**-0.5!r}f",
    )


def apb_forward(q, k, v, bias, L, heads: int = 8, d: int = 384):
    """Natural-unit packed column views; B200's saved O/LSE lifetimes on Hopper."""
    O = torch.empty_like(q, memory_format=torch.contiguous_format, dtype=torch.float32)
    LSE = torch.empty(heads, L, device=q.device, dtype=torch.float32)
    _ext("attn_fwd", _apb_training_defs(heads, d)).attn_fwd(q, k, v, bias, None, O, LSE)
    return O, LSE


def apb_forward_gated(qkvg, bias, L, heads: int = 8, d: int = 384):
    """The Pairformer track's training forward with its gate in the epilogue: (O fp32, og = sigmoid(g) O bf16, LSE) from the
    packed q | k | v | g GEMM output [L, 4 W] (natural units) and the head-major bias."""
    from .apb import width

    W = width(heads, d)
    O = torch.empty(L, W, device=qkvg.device, dtype=torch.float32)
    og = torch.empty(L, W, device=qkvg.device, dtype=torch.bfloat16)
    LSE = torch.empty(heads, L, device=qkvg.device, dtype=torch.float32)
    _ext("attn_fwd", _apb_training_defs(heads, d) + (f"APB_KS={_APB_KS['fwd']}",)).attn_fwd_o_og(qkvg, bias, O, og, LSE)
    return O, og, LSE


def apb_backward(q, k, v, dob, bias, LSE, Dd, L, heads: int = 8, d: int = 384):
    """dQB and dKV own their complete rows: no cross-CTA attention atomics."""
    opts = dict(memory_format=torch.contiguous_format, dtype=torch.float32)
    DQ, DK, DV = (torch.empty_like(q, **opts) for _ in range(3))
    DB = torch.empty(heads, L, L, device=q.device, dtype=torch.float32)
    defs = _apb_training_defs(heads, d)
    _ext("attn_dq", defs + (f"APB_KS={_APB_KS['dq']}",)).apb_dqb(q, k, v, dob, bias, LSE, Dd, DQ, DB)
    _ext("attn_dkv", defs + ("DBIAS=0", f"APB_KS={_APB_KS['dkv']}")).attn_dkv(q, k, v, dob, bias, None, LSE, Dd, DK, DV, DB)
    return DQ, DK, DV, DB


# --------------------------------------------------------------------------------------------------- gate
@functools.lru_cache(maxsize=8)
def _is_hopper(index: int) -> bool:
    return torch.cuda.get_device_capability(index) == (9, 0)


# --------------------------------------------------------------------------------------------------- Triton leftovers (kept)
@triton.jit
def _prep_do_kernel(
    do,
    o,
    dob,
    dd,
    L,
    NH: tl.constexpr,
    DH: tl.constexpr,
    DP: tl.constexpr,
    ROWS: tl.constexpr,
    WRITE_DOB: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)  # token rows (a * L + l), int64: r * (NH * DH) is an element offset
    hh = tl.arange(0, NH)
    dcol = tl.arange(0, DP)
    ptr = r[:, None, None] * (NH * DH) + (hh[:, None] * DH + dcol[None, :])[None, :, :]
    msk = (dcol[None, :] < DH)[None, :, :] & (r[:, None, None] >= 0)
    g = tl.load(do + ptr, mask=msk, other=0.0).to(tl.float32)
    x = tl.load(o + ptr, mask=msk, other=0.0)
    if WRITE_DOB:
        tl.store(dob + ptr, g.to(tl.bfloat16), mask=msk)
    s = tl.sum(g * x, axis=2)
    tl.store(dd + ((r // L)[:, None] * NH + hh[None, :]) * L + (r % L)[:, None], s)


@triton.jit
def _scale_bf16_kernel(b, out, n, scale, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(
        out + offs,
        (tl.load(b + offs, mask=m).to(tl.float32) * scale).to(tl.bfloat16),
        mask=m,
    )


def forward(qs, kb, vb, bb, km, A, L, qkvg=None, nh=H, dh=D):
    """The forward kernel on prepared operands, no op wrapper (for callers that are themselves one opaque op, as the fused
    token DiT training block): qs = q log2 e / sqrt 48, kb, vb [A L, 768] bf16 contiguous; bb [16, L, L] bf16 in log2
    units; km None or an additive key mask [A, L] fp32. Returns O fp32 [A L, 768] and the LSE [A, 16, L] (log2); with
    ``qkvg`` (the block's [A L, 3072] q|k|v|g GEMM output) og = sigmoid(g) O in bf16 instead of O -- all the token DiT
    block keeps (its backward takes D = rowsum(dO O) = rowsum(d og * og))."""
    lse = torch.empty(A, nh, L, device=qs.device, dtype=torch.float32)
    ext = _ext("attn_fwd", fwd_defs(nh, dh))
    if qkvg is not None:
        og = torch.empty(A * L, nh * dh, device=qs.device, dtype=torch.bfloat16)
        ext.attn_fwd_og(qs, kb, vb, bb, km, qkvg, og, lse)
        return og, lse
    o = torch.empty(A * L, nh * dh, device=qs.device, dtype=torch.float32)
    ext.attn_fwd(qs, kb, vb, bb, km, o, lse)
    return o, lse


def inference_core_supported(dtype: torch.dtype, L: int, d: int, h: int, device_index: int) -> bool:
    """The fused token DiT step's gated core fits: bf16, 16 heads x 48, L a multiple of 128, H100."""
    return dtype is torch.bfloat16 and (d, h) == (H * D, H) and L % 128 == 0 and _is_hopper(device_index)


def fwd_defs(nh: int = H, dh: int = D) -> tuple[str, ...]:
    """Build defines of ``attn_fwd.cu`` for a token head layout (default 16 x 48 builds without them)."""
    return () if (nh, dh) == (H, D) else (f"DHP={dh}", f"ROW_WIDTH={nh * dh}")


def dq_defs(nh: int = H, dh: int = D) -> tuple[str, ...]:
    """Build defines of ``attn_dq.cu``: ``fwd_defs`` plus the dQ output scale 1/sqrt(head_dim) its epilogue multiplies by (the source's
    default is 1/sqrt(48); without it a 12 x 64 dQ is 15 % too large and a 24 x 32 one 18 % too small)."""
    return fwd_defs(nh, dh) + (() if dh == D else (f"RSQDV={dh**-0.5!r}f",))


def gated_inference(qkvg, bias_all, block, S, nh=H, dh=D, groups=1):
    """The token DiT step's inference core (attn_fwd.cu, GATED): qkvg [S L, 3072] bf16 with q in exp2 units, bias_all
    [nb 16, L, L] bf16 in log2 units (masked keys -inf); writes sigmoid(g) softmax(q k^T + bias) v over q for ``block``'s
    slice of the bias.

    ``groups`` > 1: ``groups`` distinct hoisted biases (one per trunk_seed, each its own pair), bias_all [nb H, groups, L, L] with the
    samples laid out [A][groups][L] (S = A groups): sample s reads group s % groups. The same build serves every ``groups`` (a kernel argument)."""
    L = bias_all.shape[-1]
    _ext("attn_fwd", fwd_defs(nh, dh)).attn_inf(qkvg, bias_all.view(-1, L, L), int(block), int(S), int(groups))


def tf32_defs(nh: int = H, dh: int = D) -> tuple[str, ...]:
    """Build defines of ``attn_tf32.cu`` for a head layout (the default 16 x 48 builds without them)."""
    return () if (nh, dh) == (H, D) else (f"NHP={nh}", f"DHP={dh}")


TF32_LAYOUTS = ((16, 48), (24, 32), (12, 64), (16, 64))


def tf32_fwd_defs(nh: int = H, dh: int = D) -> tuple[str, ...]:
    """The forward's build: ``tf32_defs`` plus the tuning of the layout (the backward keeps ``tf32_defs``: one warpgroup per CTA and a
    128-key block are forward-only). Width 32 (24 x 32): 128-key blocks, one warpgroup per CTA, 3 CTAs / SM, p written back into the
    score registers -- 110 -> 90 us at L768, A = 5 (the other widths measured slower with it)."""
    extra = ()  # (the width-32 candidate "FWD_BN=128", "NWG=1", "BLKSM=3", "INPLACE_P=1" is timed 90 us but not yet correct: off)
    return (*tf32_defs(nh, dh), *extra)


def tf32_inference_core_supported(dtype: torch.dtype, L: int, d: int, h: int, device_index: int) -> bool:
    """The fp32 step's TF32 core (attn_tf32.cu) fits: fp32, h heads of width d / h (``TF32_LAYOUTS``; ``d`` is the attention width), L a multiple of 128, H100."""
    return dtype is torch.float32 and h > 0 and d % h == 0 and (h, d // h) in TF32_LAYOUTS and L % 128 == 0 and _is_hopper(device_index)


def tf32_gated_inference(qkg, vt, bias_all, block, S, goff, nh=H, dh=D):
    """The fp32 step's inference core (attn_tf32.cu, TF32 wgmma): qkg [S L, >= 2304] fp32 with q | k at columns 0 | 768 (q
    in exp2 units) and g at ``goff``; vt = v^T [768, S L]; bias_all [nb 16, L, L] fp32 in log2 units with its key columns in
    ``conditioned_transition.cuda.key_perm`` order (masked keys -inf). Writes sigmoid(g) softmax(q k^T + bias) v over q."""
    _ext("attn_tf32", tf32_fwd_defs(nh, dh)).attn_tf32_inf(qkg, vt, bias_all, int(block), int(S), int(goff))


def tf32_forward(q, k, vt, bias, km, A, L, qkvg=None, nh=H, dh=D):
    """The TF32 training forward: q (exp2 units), k [A L, 768] fp32, vt = v^T [768, A L]; bias [16, L, L] fp32 (log2 units)
    and km [A, L] additive or None, both key-permuted (``key_perm``). Returns O [A L, 768] fp32 and the LSE [A, 16, L]; with
    ``qkvg`` [A L, 3072] og = sigmoid(g) O instead of O (as ``forward``)."""
    o = torch.empty(A * L, nh * dh, device=q.device, dtype=torch.float32)
    lse = torch.empty(A, nh, L, device=q.device, dtype=torch.float32)
    ext = _ext("attn_tf32", tf32_fwd_defs(nh, dh))
    if qkvg is not None:
        ext.attn_tf32_fwd_og(q, k, vt, bias, km, qkvg, o, lse)
    else:
        ext.attn_tf32_fwd(q, k, vt, bias, km, o, lse)
    return o, lse


def tf32_backward(qs, k, v, dob, qt, kt, dot, bias_p, bias_tp, lse, dd, lse_p, dd_p, km, km_p, A, L, nh=H, dh=D, dpack=None):
    """The TF32 attention backward (attn_tf32.cu dqb + dkv). qs (q in exp2 units), k, v, dob = dO [A L, 768] fp32 and their
    transposes qt, kt, dot [768, A L]; bias_p the forward's key-permuted bias [16, L, L]; bias_tp [16, L(key), L] the bias
    transposed with its query columns permuted; lse, dd [A, 16, L] and lse_p, dd_p their query-permuted copies; km / km_p the
    natural / permuted additive key masks (or None). Returns dQ (for the unscaled q), dK, dV [A L, 768] fp32 and dbias
    [16, L, L] fp32 with its key columns still permuted (``key_perm`` order). ``dpack`` [A L, >= 3 nh dh]: dQ, dK, dV are written into its first three column blocks
    (and returned as views of it) instead of three fresh tensors."""
    ext = _ext("attn_tf32", tf32_defs(nh, dh))
    n = nh * dh
    if dpack is None:
        dq, dk, dv = (torch.empty_like(qs) for _ in range(3))
    else:
        dq, dk, dv = (dpack[:, i * n : (i + 1) * n] for i in range(3))
    dbc = torch.zeros(tf32_dqb_chunks(A), nh, L, L, device=qs.device, dtype=torch.float32)
    ext.attn_tf32_dqb(qs, dob, k, v, kt, bias_p, lse, dd, km_p, dq, dbc)
    db_p = dbc.sum(0)  # the chunk buffers in order: bitwise repeatable
    ext.attn_tf32_dkv(qs, dob, qt, dot, k, v, bias_tp, lse_p, dd_p, km, dk, dv)
    return dq, dk, dv, db_p


def tf32_dqb_chunks(A: int) -> int:
    """Partial dbias buffers of the TF32 ``attn_tf32_dqb`` (see ``dqb_chunks``): the largest divisor of A up to 4."""
    return max(c for c in (1, 2, 3, 4) if A % c == 0)


def dqb_chunks(A: int) -> int:
    """Partial dbias buffers of ``attn_dqb``: each CTA adds its A / 3 / C sample groups to ONE buffer in order (the TMA reduce-adds
    of two CTAs into one buffer land in schedule order: not bitwise repeatable); the C buffers are summed in order afterwards.
    The largest divisor of A / 3 up to 4: C x 16 x (L / 64) CTAs, 4 buffers = 4 x 38 MB fp32 at L768."""
    groups = A // 3
    return max(c for c in (1, 2, 3, 4) if groups % c == 0)


def backward(qs, kb, vb, dob, bb, km, lse, dd, A, L, nh=H, dh=D):
    """dQ, dK, dV [A L, 768] fp32 (with respect to the unscaled q, k, v) and dbias [16, L, L] fp32 (natural units;
    a transposed view when A is not a multiple of 3). dob [A L, 768] bf16 and dd = rowsum(dO O) [A, 16, L] fp32; the other operands as ``forward`` took them."""
    dq = torch.empty(A * L, nh * dh, device=qs.device, dtype=torch.float32)
    dk, dv = torch.empty_like(dq), torch.empty_like(dq)
    ld = fwd_defs(nh, dh)
    if A % 3 == 0:  # dbias in the dQ pass, three samples summed on chip before the L2 reds
        chunks = dqb_chunks(A)
        dbc = torch.zeros(chunks, nh, L, L, device=qs.device, dtype=torch.float32)
        _ext("attn_dqb", ld).attn_dqb(qs, kb, vb, dob, bb, km, lse, dd, dq, dbc)
        db = dbc.sum(0)  # the chunks in order: bitwise repeatable
        _ext("attn_dkv", ("DBIAS=0", *ld)).attn_dkv(qs, kb, vb, dob, bb, km, lse, dd, dk, dv, db)
        return dq, dk, dv, db
    # dbias by per-sample L2 atomics in the dK / dV pass (transposed)
    _ext("attn_dq", (*(() if L % 192 == 0 else ("NWG=2",)), *dq_defs(nh, dh))).attn_dq(qs, kb, vb, dob, bb, km, lse, dd, dq)
    dbt = torch.zeros(nh, L, L, device=qs.device, dtype=torch.float32)
    _ext("attn_dkv", ld).attn_dkv(qs, kb, vb, dob, bb, km, lse, dd, dk, dv, dbt)
    return dq, dk, dv, dbt.transpose(1, 2)


__all__ = [
    "backward",
    "dq_defs",
    "forward",
    "gated_inference",
    "inference_core_supported",
    "tf32_backward",
    "tf32_forward",
    "tf32_gated_inference",
    "tf32_inference_core_supported",
]


_P = "augmented_attention_"
#: requirements_kernel.csv kernels (names without ``augmented_attention_`` / ``_sm90``) of the attn_*.cu extensions, by what they build
_ATTN_BF16_FWD = {"fwd"}  # bf16 / master activations: attn_fwd (forward, gated inference)
_ATTN_BF16_BWD = {"bwd_dqb", "bwd_dq", "bwd_dkv"}  # attn_dqb / attn_dq / attn_dkv (backward)
_ATTN_TF32_FWD = {"fwd_tf32"}  # float32 activations: attn_tf32 forward
_ATTN_TF32_BWD = {"bwd_dkv_tf32", "bwd_dqb_tf32"}  # attn_tf32 backward
#: sm90_atom kind -> the atom kernels it launches (``tf32`` builds: float32 rows; ``pack`` / ``finalize`` have no tf32 build)
_ATOM_KINDS = {
    "finalize": "atom_finalize_bwd",
    "pack": "atom_pack",
    "cond": "atom_cond_fwd",
    "cond_split": "atom_cond_fwd_split",
    "pre": "atom_pre_fwd",
    "pre_split": "atom_pre_fwd_split",
    "post": "atom_post_fwd",
    "tr_bwd": "atom_tr_bwd",
    "post_bwd": "atom_post_bwd",
    "post_bwd_s": "atom_post_bwd",
    "pre_bwd": "atom_pre_bwd",
    "cond_bwd": "atom_cond_bwd",
    "cond_bwd_s": "atom_cond_bwd",
    "cond_bwd_sh": "atom_cond_bwd_shared",
    "pre_bwd_one": "atom_pre_bwd",
    "kqln_bwd": "atom_kqln_bwd",
}
#: the model options of the classes on the sm90_atom family (``integrations/local_dit_h100.options``) as its build defines, and the classes
#: with one AdaLN for q and k / v (the ``pre_bwd_one`` entry)
_ATOM_OPTIONS = {"AtomDiTBoltz2": ("UP",), "AtomDiTRoseTTAFold3": ("KQLN", "PARS"), "AtomDiTChai1": ("IDENT", "NOOUT", "PARS")}
_ATOM_ONE_ADALN = frozenset({"AtomDiTBoltz2", "AtomDiTRoseTTAFold3", "AtomDiTChai1"})
#: sm100_atom ``_HOPPER_KERNELS`` source stem -> atom kernels built from it
_ATOM_STEMS = {
    "cond_fwd": ("atom_cond_fwd", "atom_cond_fwd_split", "atom_pack"),
    "pre_fwd": ("atom_pre_fwd", "atom_pre_fwd_split"),
    "post_fwd": ("atom_post_fwd",),
    "tr_bwd_gate": ("atom_tr_bwd",),
    "post_bwd": ("atom_post_bwd",),
    "pre_bwd": ("atom_pre_bwd", "atom_finalize_bwd"),
    "cond_bwd": ("atom_cond_bwd", "atom_cond_bwd_shared"),
}
#: sm_90a local-attention source stem -> local kernels built from it
_LOCAL_STEMS = {
    "lattn90": ("local_pair_bias_fwd", "local_attn_fwd", "local_attn_bwd_dq", "local_attn_bwd_dkv", "local_pair_bias_bwd"),
    "lattn_fwd": ("local_attn_fwd",),
    "lattn_dq": ("local_attn_bwd_dq",),
    "lattn_dkv": ("local_attn_bwd_dkv", "local_bias_bwd_samples"),
    "lbias": ("local_pair_bias_fwd", "local_pair_bias_bwd", "local_pair_bias_bwd_fin"),
    "lcross": ("local_attn_fwd",),
    "lattn90_tf32": ("local_attn_fwd_tf32", "local_attn_bwd_dq_tf32", "local_attn_bwd_dkv_tf32"),
}


def prebuild(target, rows):
    """Every build of the AttentionPairBias / augmented attention family on ``target`` (kernels/_prebuild.py), expanded from the
    requirements ``rows`` (each row names the kernels that need it; the kernel's role fixes its modes):

    * attn_*.cu head layouts: the TokenDiT rows' ``n_head`` x ``d_single / n_head`` (every build below is specialised per layout), the
      AttentionPairBias rows' ``(n_head, d_single)`` for ``apb.GEOMETRY``;
    * dtype: float32 rows -> the TF32 builds, bfloat16 / master rows (bf16 activations) -> the bf16 builds;
    * mode: a build of a backward kernel exists only when a train row needs it, AttentionPairBias inference vs training defines come
      from its eval / train rows.

    What stays here: which extension / source stem / define set implements which kernel (the tables above and the define sets of the
    launchers: attn_dq L % 192, the empty-token variants, KS), and the B200 builds (not enumerated by requirement).
    Each is a call of the loader itself, so the planned key is the runtime's."""
    import functools as ft

    from miniworld_engine.kernels._prebuild import _kv

    if target.arch in ("sm80", "sm120"):  # the A100 builds (sm120: the same ones compiled for sm_120a, kernels/_sm120.py): prebuild_sm80.py (token / atom DiT attention) and _sm80_apb (AttentionPairBias rows)
        from .prebuild_sm80 import prebuild_sm80

        yield from prebuild_sm80(rows)
        yield from _sm80_apb(rows)
        return
    from . import apb, sm90_atom, sm100_atom, sm100_atom_local

    calls = []
    if target.arch == "sm90":

        def act(r):
            return "f32" if r["dtype"] == "float32" else "bf16"  # master computes in bf16 activations

        def need_row(r, names):
            return bool(r["kernels"] & {f"{_P}{n}_sm90" for n in names})

        def need(names, dtype=None):
            """Some row needs one of the kernels ``names`` (without the family prefix / ``_sm90``), for activations ``dtype``."""
            return any(need_row(r, names) and (dtype is None or act(r) == dtype) for r in rows)

        def layouts(names, dtype, token_only=False):
            """(n_heads, head width) of the rows that need ``names`` (every module but AttentionPairBias, whose layouts are
            ``apb.GEOMETRY``; ``token_only``: the TokenDiT rows)."""
            out = set()
            for r in rows:
                if (
                    (r["module"].startswith("TokenDiT") if token_only else r["module"] != "AttentionPairBias")
                    and need_row(r, names)
                    and act(r) == dtype
                ):
                    d = _kv(r["dims"])
                    out.add((int(d["n_head"]), int(d["d_single"]) // int(d["n_head"])))
            return sorted(out)

        ext = lambda name, defs=(): calls.append(ft.partial(_ext, name, tuple(defs)))  # noqa: E731
        ldefs = [(), ("NWG=2",)]  # attn_dq: L % 192 == 0 or not
        empty = [(), ("TOKEN_EMPTY_UNIFORM=1",)]  # token_dit_train._native_attn_bwd90
        # token DiT, bf16 activations (forward / gated_inference; backward; dit_variants/cuda/train.py's FWD_BF16 forward)
        for nh, dh in layouts(_ATTN_BF16_FWD, "bf16"):
            fd = fwd_defs(nh, dh)
            ext("attn_fwd", fd)
            ext("attn_fwd", (*fd, "FWD_BF16=1"))
        for nh, dh in layouts(_ATTN_BF16_BWD, "bf16"):
            fd = fwd_defs(nh, dh)
            for e in empty:
                ext("attn_dqb", (*fd, *e))
                ext("attn_dkv", ("DBIAS=0", *fd, *e))
                ext("attn_dkv", ("DB_PER_SAMPLE=1", *fd, *e))
                for nw in ldefs:
                    ext("attn_dq", (*nw, *dq_defs(nh, dh), *e))
            ext("attn_dkv", fd)  # backward, A % 3 != 0
        # token DiT, float32 activations (TF32 tensor cores)
        for nh, dh in layouts(_ATTN_TF32_FWD, "f32"):
            ext("attn_tf32", tf32_fwd_defs(nh, dh))  # tf32_forward / tf32_gated_inference
        for nh, dh in layouts(_ATTN_TF32_BWD, "f32"):
            td = tf32_defs(nh, dh)
            ext("attn_tf32", td)  # tf32_backward
            for e in empty:
                ext("attn_tf32", ("DB_PER_SAMPLE=1", *td, *e))
        # Pairformer AttentionPairBias (integrations/attention_pair_bias_b200): the (heads, d_single) of its rows, eval -> inference
        # core, train -> training forward / backward
        modes: dict[tuple[int, int], set[str]] = {}
        for r in rows:
            if r["module"] == "AttentionPairBias":
                d = _kv(r["dims"])
                modes.setdefault((int(d["n_head"]), int(d["d_single"])), set()).add(r["mode"])
        for (heads, d), ms in sorted(modes.items()):
            _, dhp = apb.GEOMETRY[(heads, d)]
            td = _apb_training_defs(heads, d)
            if "eval" in ms:
                ext("attn_fwd", (f"DHP={dhp}", f"ROW_WIDTH={apb.width(heads, d)}", "NWG=1", f"APB_KS={_APB_KS['inf']}"))
            if "train" in ms:
                ext("attn_fwd", td)
                ext("attn_fwd", (*td, f"APB_KS={_APB_KS['fwd']}"))
                ext("attn_dq", (*td, f"APB_KS={_APB_KS['dq']}"))
                ext("attn_dkv", (*td, "DBIAS=0", f"APB_KS={_APB_KS['dkv']}"))
        # AttentionPairBias with float32 activations: the TF32 core (attn_tf32.cu) at the padded layout of each head shape (head width 48
        # stays 48, every other width pads to 32), forward / inference and backward builds
        for heads, d in sorted(
            {
                (int(_kv(r["dims"])["n_head"]), int(_kv(r["dims"])["d_single"]))
                for r in rows
                if r["module"] == "AttentionPairBias" and r["dtype"] == "float32"
            }
        ):
            dhp = 48 if d // heads == 48 else 32
            ext("attn_tf32", tf32_fwd_defs(heads, dhp))
            ext("attn_tf32", tf32_defs(heads, dhp))
        if any(k.startswith(f"{_P}apb_") for r in rows for k in r["kernels"]):
            calls.append(ft.partial(apb._ext, "90a", False))
            if any("train" in ms for ms in modes.values()):
                # the training backward's exact fp32 pair-weight gradient on H100 (integrations/attention_pair_bias_b200._pair_weight_shadow)
                calls.append(ft.partial(apb._ext, "90a", True))
        if any(
            k.startswith(
                (
                    f"{_P}pair_rows",
                    f"{_P}pair_bias_fwd_fp32",
                    f"{_P}prep_fwd_fp32",
                    f"{_P}bwd_pre_fp32",
                    f"{_P}bias_transpose",
                    f"{_P}gate_mul",
                )
            )
            for r in rows
            for k in r["kernels"]
        ):
            calls.append(_rows_ext)
        # dense atom block (sm90_atom): the kinds of the atom kernels some row needs; tf32 for float32 rows, bf16 otherwise
        for kind, kernel in _ATOM_KINDS.items():
            for tf32 in (False, True):
                if tf32 and kind in ("pack", "finalize"):
                    continue  # no tf32 build
                optsets = {
                    _ATOM_OPTIONS.get(r["module"], ())
                    for r in rows
                    if need_row(r, [kernel])
                    and (kind in ("pack", "finalize") or act(r) == ("f32" if tf32 else "bf16"))
                    and (kind != "pre_bwd_one" or r["module"] in _ATOM_ONE_ADALN)
                }
                for opts in sorted(optsets):
                    calls.append(ft.partial(sm90_atom.kernel_cubin, kind, tf32, opts))
        # AtomDiTESMFold2Pair (integrations/local_dit_esm): its inference rows on the sm90_atom row kernels built with RMS, the q / k RMSNorm +
        # rotary (qknorm_local.cu) and the 192-key rank-window attention (lattn90_esm.cu; its pair-bias backward finishes on lbias.cu)
        for tf32 in (False, True):
            if need(["atom_rms_rows_fwd"], "f32" if tf32 else "bf16"):
                for kind in ("cond", "cond_split", "pre", "pre_split", "post"):
                    calls.append(ft.partial(sm90_atom.kernel_cubin, kind, tf32, ("RMS",)))
        if need(["atom_rms_rows_fwd"]):
            calls.append(ft.partial(sm90_atom.kernel_cubin, "pack", False, ()))
        if need(["qknorm_rope_fwd", "qknorm_rope_bwd"]):
            ext("qknorm_local")
        if need(["local_esm_attn_fwd", "local_esm_pair_bias_fwd", "local_esm_attn_bwd_dq", "local_esm_attn_bwd_dkv"]):
            calls.append(ft.partial(sm100_atom_local.esm_cubin, True))
        if need(["local_esm_pair_bias_bwd"]):
            calls.append(ft.partial(sm100_atom_local.esm_cubin, True))
            calls.append(ft.partial(sm100_atom_local.cubin, "lbias", True))
        # sm_90a halves of the sm100_atom / sm100_atom_local sources
        for stem, _, _, _, *defs in sm100_atom._HOPPER_KERNELS.values():
            if need(_ATOM_STEMS[stem]):
                calls.append(ft.partial(sm100_atom.cubin, stem, *defs, hopper=True))
        for stem, names in _LOCAL_STEMS.items():
            if need(names, "f32" if stem.endswith("tf32") else None):
                calls.append(ft.partial(sm100_atom_local.cubin, stem, True))
    else:
        calls.append(ft.partial(apb._ext, "100", False))
        atom = sm100_atom.KERNELS.values()
        # (the augattn_sm100/* cubins, sm100.cubin, take stems / defines from many B200 launchers: not enumerated here)
        for stem in sorted({s for s, *_ in sm100_atom.KERNELS_TF32.values()} | {s for s, *_ in sm100_atom.KERNELS_FUSED32.values()}):
            calls.append(ft.partial(sm100_atom.cubin_tf32, stem))
        for f32 in (False, True):  # sm100_atom.AtomDw
            bkr, ns = (32, 2) if f32 else (16, 5)
            calls.append(ft.partial(sm100_atom.cubin_tf32, "dw_atom", (f"DW_BKR={bkr}", f"NS={ns}", "DW_F32") if f32 else (f"NS={ns}",)))
        for stem in sorted({s for s, *_ in sm100_atom_local.KERNELS_TF32.values()}):
            calls.append(ft.partial(sm100_atom_local.cubin, stem))
        for stem, _, _, _, *defs in atom:
            calls.append(ft.partial(sm100_atom.cubin, stem, *defs, hopper=False))
        local = {
            *sm100_atom_local.KERNELS,
            *sm100_atom_local._HOPPER_CROSS,
            *sm100_atom_local._MMA_KERNELS,
            *sm100_atom_local._MMA_KERNELS32,
            "fwd",
            "dq",
            "dkv",
            "db_samples",
            "kv_bias_part",
            "reduce_columns",
            "zero_kv",
            "add_cond",
            "cross_grad_pack",
            "kv_pack",
        }
        stems = set()
        for name in local:
            try:
                stems.add(sm100_atom_local._resolve(name, False)[0])
            except NotImplementedError:
                pass
        for stem in sorted(stems):
            calls.append(ft.partial(sm100_atom_local.cubin, stem, False))
    yield from calls


def _sm80_apb(rows):
    """The A100 builds of AttentionPairBias (``kernels/integrations/attention_pair_bias_sm80``): the row / pair / C++-step extension
    (``apb.ext_sm80``) and the attention core (``sm80._ext``), when a bf16 / master AttentionPairBias row has a layout the A100 path takes
    (``apb.GEOMETRY``, d_pair 128)."""
    from miniworld_engine.kernels._prebuild import _kv

    from . import apb, sm80

    for r in rows:
        if r["module"] != "AttentionPairBias" or r["dtype"] == "float32":
            continue
        d = _kv(r["dims"])
        if (int(d["n_head"]), int(d["d_single"])) in apb.GEOMETRY and int(d["d_pair"]) == apb.DP and "gate_bias" not in r["options"]:
            return [apb.ext_sm80, sm80._ext]
    return []
