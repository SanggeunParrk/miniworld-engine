"""A100 (sm_80) attention core of the bias-only token DiT: ``out = sigmoid(g) * (P v)`` per head and sample, hand CUDA (``mma.sync`` / ``ldmatrix`` / ``cp.async``).

The inference runner of ``cuda/runner.py`` is cuBLAS and the family's CUDA row kernels (``bias_only_dit_rows.cu``, built for sm_80 too) around one attention core; on B200 that core is the
tcgen05 ``pv_gate_inf`` (``PvGateCore``), here it is :class:`PvGate`: the same call, the same layouts, bf16 operands with fp32 accumulation, the gate and one rounding to bf16 in the epilogue.
Built on first use (``load_extension``), never at import.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

from miniworld_engine.kernels._nvcc import ensure_cuda_home, host_flags
from miniworld_engine.kernels._sm120 import building_sm120
from miniworld_engine.kernels._sm120 import sm80_load_extension as load_extension  # sm_120a: same sources (kernels/_sm120.py)

_dir = Path(__file__).parent
#: head widths the kernel is instantiated for
HEAD_WIDTHS = (32, 48, 64)
#: depth of the cp.async ring
STAGES = 2


@functools.lru_cache(maxsize=1)
def _ext():
    ensure_cuda_home()
    return load_extension(
        name="bias_only_dit_sm80",
        sources=[str(_dir / "ops.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-std=c++17",
            "-O3",
            "-gencode=arch=compute_80,code=sm_80",
            f"-I{_dir}",
        ],
        extra_cflags=["-std=c++17", "-O3"],
        **({"extra_ldflags": ["-lcublasLt"]} if building_sm120() else {}),  # sm_120a: the fp32 block's explicit cuBLASLt GEMMs (lt_f32_sm120.cuh)
        verbose=False,
    )


def lt_mm32(a: torch.Tensor, w: torch.Tensor, out: torch.Tensor, cfg) -> torch.Tensor:
    """RTX 5090 only (lt_f32_sm120.cuh): out = a w^T on TF32 through cuBLASLt with the explicit configuration ``cfg`` (7 ints); a / w / out [M, K] / [N, K] /
    [M, N] (out a row-major view) or [B, M, K] / [B, N, K] / [B, M, N] (a strided batch)."""
    _ext().lt_mm32(a, w, out, list(cfg))
    return out


def lt_algos32(a: torch.Tensor, w: torch.Tensor, out: torch.Tensor, count: int = 64) -> list[list[int]]:
    """RTX 5090 only: cuBLASLt's heuristic candidates (7-int configurations) for the shape of ``lt_mm32(a, w, out)``."""
    return _ext().lt_algos32(a, w, out, count)


#: gemm120's epilogues (gemm_sm120.cuh)
EPI_PLAIN, EPI_GATERES, EPI_ADALN, EPI_SWIGLU = 0, 1, 2, 3


def gemm120(epi: int, a: torch.Tensor, w1: torch.Tensor, out: torch.Tensor, *, w2=None, bias=None, r=None, g=None, x=None, xst=None, sb=None, cfg: int = 0):
    """RTX 5090 only (the sm_120a build, gemm_sm120.cuh): ``out`` = the epilogue ``epi`` of a [M, K] [w1; w2]^T, bf16 operands or fp32 (TF32,
    operands rounded to nearest): PLAIN (+ bias), GATERES (r + sigmoid(g) acc), ADALN (sigmoid(S + sb) LN(x) + B with x's row statistics
    ``xst``), SWIGLU (silu(a) b); ``cfg`` the CTA tile (0: 128 x 128, 1: 64 x 128, 2: 64 x 64, 3: 128 x 64, 4: 32 x 128)."""
    _ext().gemm120(epi, a, w1, w2, out, bias, r, g, x, xst, sb, cfg)
    return out


def row_stats120(x: torch.Tensor, eps: float) -> torch.Tensor:
    """RTX 5090 only: the (mean, rstd) [M, 2] fp32 of the rows of x [M, 768] (the AdaLN's LayerNorm, no affine)."""
    st = torch.empty(x.shape[0], 2, device=x.device, dtype=torch.float32)
    _ext().row_stats120(x, st, float(eps))
    return st


def cond3_f32(c: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, eps: float, copy: bool = True) -> torch.Tensor:
    """RTX 5090 only (rows_f32_sm120.cuh): ``[3, M, dc]`` fp32 = (LN(c) w1, LN(c) w2, c) for c [M, dc] (dc 384 / 768): the operands of the
    fp32 block's one batched cond GEMM."""
    out = torch.empty(3 if copy else 2, c.shape[0], c.shape[1], device=c.device, dtype=torch.float32)  # without copy: the two LayerNorms only
    _ext().cond3_f32(c, w1, w2, out, float(eps))
    return out


def resgate_adaln_f32(y, g, bs, x, sbt, sb, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """RTX 5090 only: ``(out, xa)``, out = x + sigmoid(g + bs) y, xa = sigmoid(S + sb) LN(out) + B (sbt = [S | B] [M, 1536]); fp32 [M, 768]."""
    out, xa = torch.empty_like(x), torch.empty_like(x)
    _ext().resgate_adaln_f32(y, g, bs, x, sbt, sb, out, xa, float(eps))
    return out, xa


def gate_res_b_f32(x, z, g, b) -> torch.Tensor:
    """RTX 5090 only: x + sigmoid(g + b) z, fp32 [M, 768] (g a [M, 768] view)."""
    y = torch.empty_like(x)
    _ext().gate_res_b_f32(x, z, g, b, y)
    return y


def pair_softmax_f32(z: torch.Tensor, wcm: torch.Tensor, mask: torch.Tensor | None, length: int, eps: float, cfg: int = 0) -> torch.Tensor:
    """RTX 5090 only (rows_f32_sm120.cuh pair_softmax): P [16, L, L] fp32 = softmax over the keys of the pair bias LN(z) W'^T of z [L, L, C] fp32
    (C 128 / 256, W' = wcm [C, 16] fp32), masked keys (a bool [L] mask) 0 -- the pair bias never leaves shared memory. L a multiple of 128, <= 1024."""
    p = torch.empty(16, length, length, device=z.device, dtype=torch.float32)
    kv = mask.reshape(length).contiguous() if mask is not None else torch.empty(0, dtype=torch.bool, device=z.device)
    _ext().pair_softmax_f32(z, wcm, kv, p, length, float(eps), cfg)
    return p


def pick_group(S: int, L: int, nh: int, nsm: int) -> int:
    """Samples per CTA (1 .. 4): the fewest CTAs that still fill the card -- a group reads P once for all its samples, and a CTA alone on its SM is request-bound -- i.e. the smallest g with
    ``nh (L / 128) ceil(S / g) <= nsm``, else 2 when even that leaves a second wave (A = 48, 16 x 48: 149 / 458 us at L = 384 / 768 against 171 / 502 us for 4: the smaller group's registers
    let two CTAs share an SM; A = 5, L = 768 is within 1 % of 4 either way)."""
    tiles = nh * (L // 128)
    for g in (1, 2, 3, 4):
        if tiles * -(-S // g) <= nsm:
            return g
    return 2


def _kmask(kmask: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    """The cores' ``kmask`` operand: the bool key mask [L] (``kmask`` [1, L] or [L]) P was made with -- the chunks past its last attended key hold
    P = 0 and are skipped (the same bits) -- or an empty tensor."""
    return like.new_empty(0, dtype=torch.bool) if kmask is None else kmask.reshape(-1).to(torch.bool).contiguous()


class PvGate:
    """The A100 twin of ``cuda.PvGateCore``: ``out [S L, W] = sigmoid(g) * (P v)`` -- or, without ``g``, ``P v`` -- per head and sample, W = nh dh."""

    def __init__(self, nh: int, dh: int):
        assert dh in HEAD_WIDTHS, dh
        self.nh, self.dh = nh, dh
        self.nsm = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

    def __call__(
        self,
        v: torch.Tensor,
        P: torch.Tensor,
        out: torch.Tensor,
        S: int,
        g: torch.Tensor | None = None,
        stages: int | None = None,
        sg: int | None = None,
        kmask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """v [S L, W] view, P [nh L, L] contiguous, out [S L, W] view, g [S L, W] view or None; all bf16 (v, g, out: any row stride, unit column stride).
        ``stages`` (2 or 3; default :data:`STAGES`) is the depth of the cp.async ring, ``sg`` (1 .. 4; default :func:`pick_group`) the samples a CTA takes. ``kmask``: the bool key mask [1, L] P
        was made with, or None (``_kmask``)."""
        L = P.shape[1]
        _ext().pv_gate(
            v,
            P,
            out,
            g if g is not None else v.new_empty(0),
            S,
            self.nh,
            self.dh,
            STAGES if stages is None else stages,
            pick_group(S, L, self.nh, self.nsm) if sg is None else sg,
            _kmask(kmask, v),
        )
        return out


#: m16 tiles per warp of the TF32 core
MT32 = 1


class PvGateTF32:
    """The fp32 / TF32 twin of :class:`PvGate`: fp32 ``v`` / ``g`` / ``out`` views [S L, W] and fp32 ``P`` [nh L, L], the products on TF32 tensor cores (operands rounded to nearest tf32, fp32 accumulation)."""

    def __init__(self, nh: int, dh: int):
        assert dh in HEAD_WIDTHS, dh
        self.nh, self.dh = nh, dh
        self.nsm = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

    def __call__(
        self,
        v: torch.Tensor,
        P: torch.Tensor,
        out: torch.Tensor,
        S: int,
        g: torch.Tensor | None = None,
        stages: int | None = None,
        sg: int | None = None,
        mt: int | None = None,
        kmask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        L = P.shape[1]
        _ext().pv_gate32(
            v,
            P,
            out,
            g if g is not None else v.new_empty(0),
            S,
            self.nh,
            self.dh,
            2 if stages is None else stages,
            pick_group(S, L, self.nh, self.nsm) if sg is None else sg,
            MT32 if mt is None else mt,
            _kmask(kmask, v),
        )
        return out


#: schedule of the bias-gradient core (``dpb``): 0 two stages x one sample with two CTAs per SM, 1 three x one, 2 two x two, 3 three x two
DPB_CFG = 0


class Dpb:
    """The A100 twin of ``cuda.DpbKernel``: ``dbias [nh L, L] = P o (sum_a do v^T - D)`` per head, ``D[h, i] = sum_a dd[a, h, i]`` -- ``do`` / ``v`` bf16 ``[A L, W]`` views (any row stride),
    ``P`` / ``dbias`` bf16 ``[nh L, L]`` contiguous, ``dd`` fp32 ``[A, nh, L]``; a masked key (P = 0) gets 0."""

    def __init__(self, nh: int, dh: int):
        assert dh in HEAD_WIDTHS, dh
        self.nh, self.dh = nh, dh

    def __call__(
        self,
        do: torch.Tensor,
        v: torch.Tensor,
        P: torch.Tensor,
        dd: torch.Tensor,
        out: torch.Tensor,
        A: int,
        cfg: int | None = None,
    ) -> torch.Tensor:
        _ext().dpb(
            do,
            v,
            P,
            dd,
            out,
            A,
            self.nh,
            self.dh,
            DPB_CFG if cfg is None else cfg,
        )
        return out


class Dpb32:
    """The TF32 twin of :class:`Dpb`: fp32 ``do`` / ``v`` views [A L, W], fp32 ``P`` / ``dbias`` [nh L, L], ``dd`` fp32 [A, nh, L]; the products on TF32 tensor cores (a fixed order: reruns are bit-identical)."""

    def __init__(self, nh: int, dh: int):
        assert dh in HEAD_WIDTHS, dh
        self.nh, self.dh = nh, dh

    def __call__(self, do, v, P, dd, out, A):
        _ext().dpb32(do, v, P, dd, out, A, self.nh, self.dh)
        return out


# ------------------------------------------------------------------------------------------------- head planes (the bias-only ATTENTION family, ``kernels/bias_only_attention``)
def pv_planes(
    v: torch.Tensor,
    P: torch.Tensor,
    out: torch.Tensor,
    samples: int,
    planes: int,
    dh: int,
    stages: int | None = None,
    sg: int | None = None,
) -> torch.Tensor:
    """``out = P v`` per head plane for ``samples`` problems that share the plane's ``P``: ``v`` / ``out`` bf16 ``[planes samples L, dh]`` contiguous (plane h = ``[samples L][dh]``, one after
    the other: the bias-only attention's ``[B, H, L (t), L (n), D]``), ``P`` bf16 ``[planes L, L]``."""
    L = P.shape[1]
    nsm = torch.cuda.get_device_properties(v.device).multi_processor_count
    _ext().pv_planes(
        v,
        P,
        out,
        samples,
        planes,
        dh,
        STAGES if stages is None else stages,
        pick_group(samples, L, planes, nsm) if sg is None else sg,
    )
    return out


def delta_planes(
    do: torch.Tensor,
    o: torch.Tensor,
    dd: torch.Tensor,
    samples: int,
    length: int,
    planes: int,
    dh: int,
) -> torch.Tensor:
    """``dd[a, h, i] = sum_d do o`` over bf16 ``[planes samples L, dh]`` operands (``dd`` fp32 ``[samples, planes, L]``): the backward's row term."""
    _ext().delta_planes(do, o, dd, samples, length, planes, dh)
    return dd


def dpb_planes(
    do: torch.Tensor,
    v: torch.Tensor,
    P: torch.Tensor,
    dd: torch.Tensor,
    out: torch.Tensor,
    samples: int,
    planes: int,
    dh: int,
    cfg: int | None = None,
) -> torch.Tensor:
    """``out [planes L, L] = P (sum_a do v^T - D)`` per head plane, ``D[h, i] = sum_a dd[a, h, i]``: ``do`` / ``v`` bf16 ``[planes samples L, dh]``, ``P`` / ``out`` bf16 ``[planes L, L]``."""
    _ext().dpb_planes(do, v, P, dd, out, samples, planes, dh, DPB_CFG if cfg is None else cfg)
    return out
