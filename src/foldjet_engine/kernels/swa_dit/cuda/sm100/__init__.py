"""sm_100a (B200) kernels of the fused SWA atom DiT block (``kernels/swa_dit``), forward and backward.

The block is ``interface.swa_dit_block`` -- the ESMFold2 SWA atom block over the flattened [N = A*B, S] atom sequence -- with
every stage on tcgen05 / TMEM / TMA kernels instead of the Triton (and sm_90 wgmma) ones; the weight-gradient GEMMs are one split-K
launch (``dw_bf16.cu``, deterministic reduction). Developed in ``experiments/swaatom_sm100`` (rounds v1-v13); page:
``docs/gpus/b200/swa_atom_dit/swa_atom_dit.md``.

    forward   qkvg (qkvg_fwd: 128-row tiles; qkvg_fwd2: transposed 32-row tiles for small problems / A < 5, the ATM = 32
              build for A < 4) -> window attention (attn_fwd3; attn_fwd1h when its one-head items fit one wave) ->
              out-projection + gated residual + FFN (ffn_fwd2; the ATM = 32, single-stage build for A < 4)
    backward  FFN (ffn_bwd_gate -> ffn_bwd_dy) -> out-projection (oproj_bwd) -> attention (attn_dkvq: dQ inside the dK / dV
              pass when every CTA gets >= 2 items, else attn_dq + attn_dkv) -> qkvg (qkvg_bwd), then dWqkv | dWg as one GEMM
    modulation  silu(c) Wmod^T (mod_fwd) and its backward (mod_bwd)

Global attention (``interface.is_global(half_window)``; B200: forward on gattn_fwd.cu (inference and training), backward gattn_bwd.cu, see
:func:`native_global_train`): the window attention stage (attn_fwd3 / attn_fwd1h forward, attn_dq /
attn_dkv / attn_dkvq backward) is swapped for FlashAttention-4 on the head-major Q / K / V the qkvg stage already writes
(``attn_global_fwd`` / ``attn_global_bwd``); every other stage is unchanged.

Served (:func:`supported`): B200, bf16, C = 128 with 4 heads of 32, SwiGLU hidden 256, half window 64 or global, fp32 eps. The atom
count S must be a multiple of 128 -- callers pad the atoms and ``seqused`` masks the padding; the block refuses with
ValueError otherwise (no other path for it on B200). The cubins are built on first use by the newest nvcc here that knows
sm_100a (``transition.cuda.fused_sm100a.kernel_toolchain``) and cached under ``settings.engine_jit_root``; the launches go
through the CUDA driver (``augmented_attention.cuda.sm100.driver``) on torch's current stream, CUDA-graph capturable.
"""

from __future__ import annotations

import functools
import importlib.util
import os
from pathlib import Path

import torch

from foldjet_engine.kernels._device import num_sms

from foldjet_engine.kernels import _capture

C, H, D, NHID, HW = 128, 4, 32, 256, 64
EPS = float(torch.finfo(torch.float32).eps)
PDL = True  # programmatic dependent launch for the forward kernels
PDL_BWD = True  # B200 (2026-11-17): the backward kernels too (each waits on griddepcontrol.wait before its first global access)
WPRE = True  # round 3 part 8 (development switch): with the forms of master_pack_train, the backward kernels read weights before their wait
DW_MIN_ROWS = 128  # the bf16 weight-gradient split-K on few rows: down to this many rows per CTA (more CTAs than the 512-row rule)
DW_MAX_SPLIT = 16  # ... but at most this many partial sums per problem there (2026-11-17: 148 of them cost a 15 us reduction at 8192 rows)
_dir = Path(__file__).parent
#: the fp32 inference build of the global attention forward (gattn_fwd32, round 3 part 4): 2 = q read fp32 (rounded in the kernel), K | V
#: from the cast launch, O written fp32; 1 = q, K, V all fp32 (K | V rounded once per item: slower from L 2048, kept as a build option)
F32IO_MODE = 2

#: cubin name -> (source stem, extra nvcc flags): the sources' defaults, and the A = 1 - 3 builds (ATM = 32; the FFN single stage)
CUBINS = {
    "qkvg_fwd": ("qkvg_fwd", ()),
    "qkvg_fwd2": ("qkvg_fwd2", ()),
    "qkvg_fwd2_a1": ("qkvg_fwd2", ("-DATM=32", "-DPDL_EARLY=0")),
    "ffn_fwd2": ("ffn_fwd2", ()),
    "ffn_fwd2_a1": ("ffn_fwd2", ("-DATM=32", "-DNSTG=1", "-DPDL_EARLY=0")),
    "attn_fwd3": ("attn_fwd3", ()),
    "attn_fwd1h": ("attn_fwd1h", ()),
    # few tiles (one sample of a few thousand atoms, inference): one 32-row tile per CTA, the modulation inside, W read in place (round 3)
    "qkvg_fwd1t": ("qkvg_fwd1t", ()),
    # round 3 part 8 (master training of one sample): the qkvg stage with its saves, the modulation kernel beside it (LATE_TRIGGER / LATE_WAIT)
    "qkvg_fwd1t_tr": ("qkvg_fwd1t", ("-DSAVE=1", "-DLATE_WAIT=1")),
    "mod_fwd_lt": ("mod_fwd", ("-DLATE_TRIGGER=1",)),
    "ffn_fwd1t": ("ffn_fwd1t", ("-DPDL_EARLY=0",)),
    # many tiles per CTA (inference on more rows than one tile per SM): the 1t design as a loop over the CTA's tiles -- the modulation
    # inside, double-buffered tile inputs (round 3, third pass)
    "qkvg_fwd3": ("qkvg_fwd3", ()),
    "ffn_fwd3": ("ffn_fwd3", ("-DPDL_EARLY=0",)),
    "ffn_bwd_gate": ("ffn_bwd_gate", ()),
    "ffn_bwd_dy": ("ffn_bwd_dy", ()),
    "oproj_bwd": ("oproj_bwd", ()),
    "attn_dq": ("attn_dq", ()),
    "attn_dkv": ("attn_dkv", ()),
    "attn_dkvq": ("attn_dkvq", ()),
    "qkvg_bwd": ("qkvg_bwd", ()),
    "mod_fwd": ("mod_fwd", ()),
    "mod_bwd": ("mod_bwd", ()),
    "dw_bf16": ("dw_bf16", ()),
    "mod_bwd_act": ("mod_bwd_act", ()),
    # A <= 8 (one tile holds every augment of its rows): the modulation gradient by plain stores (no zeroed dmod, no red.add)
    "ffn_bwd_dy_st": ("ffn_bwd_dy", ("-DDMOD_STORE=1",)),
    "oproj_bwd_st": ("oproj_bwd", ("-DDMOD_STORE=1",)),
    "qkvg_bwd_st": ("qkvg_bwd", ("-DDMOD_STORE=1",)),
    # round 3 part 8 (master training of few tiles): the weight forms made at the forward's head in one launch (wpack_bf16.cu), the
    # backward kernels loading them before griddepcontrol.wait (WPRE)
    "wpack_bf16": ("wpack_bf16", ()),
    "ffn_bwd_gate_wp": ("ffn_bwd_gate", ("-DWPRE=1",)),
    "ffn_bwd_dy_stwp": ("ffn_bwd_dy", ("-DDMOD_STORE=1", "-DWPRE=1")),
    "oproj_bwd_stwp": ("oproj_bwd", ("-DDMOD_STORE=1", "-DWPRE=1")),
    "qkvg_bwd_stwp": ("qkvg_bwd", ("-DDMOD_STORE=1", "-DWPRE=1")),
    # the InputFeatureEmbedder's glue (round 3): RoPE tables + seqused, features + projection (+ LayerNorm), the token mean
    "embed_glue": ("embed_glue", ()),
    # global attention on tcgen05 (round 3, swa-global): forward (inference; with the LSE for training)
    "gattn_fwd": ("gattn_fwd", ("-DSPLITB=0",)),
    # its split mode (round 3, part 3: an item's key range cut across CTAs where whole items fill the grid poorly)
    "gattn_fwd_s": ("gattn_fwd", ("-DSPLITB=1",)),
    # its deterministic backward (round 3, part 2: training on the native forward's log2 LSE)
    "gattn_bwd": ("gattn_bwd", ()),
    # its split mode (round 3, part 3: the (item, query step) sequence cut evenly across the CTAs where whole items leave SMs idle)
    "gattn_bwd_s": ("gattn_bwd", ("-DSPLITB=1",)),
    # its cluster split (r3-swa part 18): one item per cluster of CS CTAs, query shares, dK | dV reduce-scattered over DSMEM
    "gattn_bwd_c": ("gattn_bwd", ("-DCSB=1", "-DQR=1")),
    "gattn_bwd_c2": ("gattn_bwd", ("-DCSB=1", "-DQR=1", "-DCSN=2")),
    "gattn_bwd_c3": ("gattn_bwd", ("-DCSB=1", "-DQR=1", "-DCSN=3")),
    "gattn_bwd_c4": ("gattn_bwd", ("-DCSB=1", "-DQR=1", "-DCSN=4")),
    "gattn_bwd_c8": ("gattn_bwd", ("-DCSB=1", "-DQR=1", "-DCSN=8")),
    # the forward on fp32 io (round 3, part 4: the fp32 block's inference -- fp32 heads rounded to bf16 in the kernel, O written fp32):
    # whole items / split mode
    "gattn_fwd32": ("gattn_fwd", ("-DSPLITB=0", f"-DF32IO={F32IO_MODE}")),
    "gattn_fwd32_s": ("gattn_fwd", ("-DSPLITB=1", f"-DF32IO={F32IO_MODE}")),
    # the cluster split (r3-swa part 17): one item per cluster of CS CTAs, key shares, partials reduce-scattered over DSMEM
    "gattn_fwd_c": ("gattn_fwd", ("-DSPLITB=0", "-DCSPLIT=1")),
    "gattn_fwd32_c": ("gattn_fwd", ("-DSPLITB=0", "-DCSPLIT=1", f"-DF32IO={F32IO_MODE}")),
}
#: B200 (2026-11-17, round 3): the modulation on many rows through cuBLAS. Forward ``silu(c) Wmod^T``: bf16 operands, fp32 accumulation
#: and output (``torch.mm(..., out_dtype=float32)``: the same exact products as ``mod_fwd``; its fixed 128-row CTAs at one CTA per SM
#: reach ~1.5 TB/s writing the fp32 [R, 768], cuBLAS ~4.3 TB/s: R 196608 401 -> 139 us, R 40960 82 -> 37 us). Backward from this many
#: rows: ``dc`` and ``dWmod`` as TF32 GEMMs of the fp32 dmod (bf16-valued Wmod / silu(c) exact in TF32; dmod rounded to TF32) around one
#: elementwise kernel (``mod_bwd_act.cu``), in place of ``mod_bwd``'s three-term bf16 split (2048-6144 one-wave cluster CTAs: R 196608
#: 872 -> ~330 us).
MOD_FWD_CUBLAS_ROWS = 4096
MOD_BWD_CUBLAS_ROWS = 16384


# --------------------------------------------------------------------------------------------------- build / load
@functools.lru_cache(maxsize=None)
def cubin(name: str) -> str:
    """Path of the cubin ``name`` (see CUBINS), built on first use; rebuilt only when a source, a flag or the toolchain changes."""
    from foldjet_engine.kernels import _nvcc

    stem, extra = CUBINS[name]
    flags = (
        "-std=c++17",
        "-O3",
        "-arch=sm_100a",
        "-cubin",
        "-lineinfo",
        f"-I{_dir}",
        *extra,
    )
    return str(_nvcc.cubin(f"swa_dit_sm100/{name}", flags, source=_dir / f"{stem}.cu", toolchain="sm100"))


def _load_swa_sm100_kernel(name: str, func: str, smem: int, cluster: int | None = None, pdl: bool = False):
    """The kernel ``func`` of cubin ``name``, loaded in the current device's context."""
    from foldjet_engine.kernels.augmented_attention.cuda.sm100 import driver

    return driver.Kernel(cubin(name), func, smem, cluster=cluster, pdl=pdl)


def _map(t, dims, strides, box, swizzle=128, dtype="bf16"):
    from foldjet_engine.kernels.augmented_attention.cuda.sm100 import driver

    return driver.TensorMap(t, dims, strides, box, swizzle=swizzle, dtype=dtype)


# --------------------------------------------------------------------------------------------------- kernels (host side)
def tiling(A):
    """(SP augments, AT atoms) of a 128-row tile: SP = min(A, 16), AT = 128 // SP (the modulation tile holds AT <= 25 rows)."""
    SP = min(A, 16)
    AT = 128 // SP
    assert AT <= 25, "A >= 6 needed by the resident-weight layout"
    return SP, AT


def map_rows(t, A, B, S, SP, AT):
    """[A, B, S, 128] bf16 rows -> 4-D TMA map, box (64 channels, AT atoms, 1, SP augments), 128-B swizzle."""
    return _map(t, [C, S, B, A], [C * 2, S * C * 2, B * S * C * 2], [64, AT, 1, SP])


class QkvgFwd:
    def __init__(self, cubin="qkvg_fwd"):
        self.k = _load_swa_sm100_kernel(cubin, "swa_qkvg_fwd_sm100", 232448, pdl=PDL)

    def bind(self, q, mod, cos, sin, wqkv, wg, A, B, save=False, W=None):
        """q [N, S, C] bf16 (N = A B); mod [B S, 6C] fp32; cos / sin [B S, D/2] fp32 -> run(), (Qh, Kh, Vh [N, H, S, D], G [N S, C], X, PQ, PK)."""
        N, S, _ = q.shape
        SP, AT = tiling(A)
        dev = q.device
        W = torch.cat([wqkv, wg]).contiguous() if W is None else W
        Qh, Kh, Vh = (torch.empty(N, H, S, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
        G = torch.empty(N * S, C, device=dev, dtype=torch.bfloat16)
        X, PQ, PK = (torch.empty(N * S, C, device=dev, dtype=torch.bfloat16) for _ in range(3)) if save else (G, G, G)
        maps = (
            map_rows(q, A, B, S, SP, AT),
            _map(W, [C, 4 * C], C * 2, [64, 128]),
            _map(
                mod,
                [32, B * S, 24],
                [6 * C * 4, 128],
                [32, AT, 8],
                swizzle=128,
                dtype="f32",
            ),  # (32 ch, rows, 24 blocks)
            _map(cos, [D // 2, B * S], [D // 2 * 4], [D // 2, AT], swizzle=0, dtype="f32"),
            _map(sin, [D // 2, B * S], [D // 2 * 4], [D // 2, AT], swizzle=0, dtype="f32"),
        )
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        grid = (min(num_sms(), ntile), 1, 1)

        def run():
            self.k(
                grid,
                (384, 1, 1),
                *maps,
                int(S),
                int(A),
                int(B),
                int(SP),
                int(AT),
                int(nab),
                int(nag),
                int(ntile),
                float(EPS),
                float(EPS),
                int(save),
                Qh,
                Kh,
                Vh,
                G,
                X,
                PQ,
                PK,
            )

        run.keep = (maps, W)
        return run, (Qh, Kh, Vh, G, X, PQ, PK)


def map_heads(t, A, B, S, SP, AT):
    """head-major [A, B, H, S, D] bf16 -> 5-D TMA map, box (D, AT atoms, 1 head, 1, SP augments), 64-B swizzle."""
    return _map(
        t,
        [D, S, H, B, A],
        [D * 2, S * D * 2, H * S * D * 2, B * H * S * D * 2],
        [D, AT, 1, 1, SP],
        swizzle=64,
    )


class QkvgBwd:
    """dq, dP = [dpq | dpk | dV | dG] and the shift_a / scale_a gradient (the Triton _qkvg_bwd's outputs) from the forward saves (q, pq, pk)
    and the incoming gradients (dq1, dQh / dKh / dVh head-major, dG)."""

    def __init__(self, cubin="qkvg_bwd", trace=None):
        self.k = _load_swa_sm100_kernel(cubin, "swa_qkvg_bwd_sm100", 232448, pdl=PDL_BWD)
        self.trace = trace

    @staticmethod
    def smem_plan(AT):
        aux = (AT * 512 + 2 * ((AT * 64 + 127) // 128 * 128) + 1023) // 1024 * 1024
        ns = min(12, (232448 - 2 * aux - 2048 - 512 - 512) // 16384)
        assert ns >= 8, (AT, ns)
        return ns

    def bind(
        self,
        q,
        dq1,
        pq,
        pk,
        dg,
        dqh,
        dkh,
        dvh,
        mod,
        cos,
        sin,
        wqkv,
        wg,
        A,
        B,
        dmod=None,
        WT=None,
        dq=None,
        dP=None,
    ):
        M = q.numel() // C
        S = M // (A * B)
        SP = min(A, 8)
        AT = 64 // SP  # 64-row tiles
        dev = q.device
        dq = torch.empty(M, C, device=dev, dtype=torch.bfloat16) if dq is None else dq
        dP = torch.empty(M, 4 * C, device=dev, dtype=torch.bfloat16) if dP is None else dP
        own = dmod is None
        if own:
            dmod = torch.zeros(B * S, 6 * C, device=dev)
        WT = torch.cat([wqkv, wg]).t().contiguous() if WT is None else WT
        maps = (
            map_rows(q, A, B, S, SP, AT),
            map_rows(dq1, A, B, S, SP, AT),
            map_rows(pq, A, B, S, SP, AT),
            map_rows(pk, A, B, S, SP, AT),
            map_rows(dg, A, B, S, SP, AT),
            map_heads(dqh, A, B, S, SP, AT),
            map_heads(dkh, A, B, S, SP, AT),
            map_heads(dvh, A, B, S, SP, AT),
            _map(
                mod,
                [32, B * S, 24],
                [6 * C * 4, 128],
                [32, AT, 4],
                swizzle=128,
                dtype="f32",
            ),
            _map(cos, [D // 2, B * S], [D // 2 * 4], [D // 2, AT], swizzle=0, dtype="f32"),
            _map(sin, [D // 2, B * S], [D // 2 * 4], [D // 2, AT], swizzle=0, dtype="f32"),
            _map(dP, [4 * C, S, B, A], [8 * C, S * 8 * C, B * S * 8 * C], [64, AT, 1, SP]),
            _map(
                dP,
                [4 * C, S, B, A],
                [8 * C, S * 8 * C, B * S * 8 * C],
                [32, AT, 1, SP],
                swizzle=64,
            ),
            map_rows(dq, A, B, S, SP, AT),
        )
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        grid = (min(num_sms(), ntile), 1, 1)
        ns = self.smem_plan(AT)

        def run():
            if own:
                dmod.zero_()
            self.k(
                grid,
                (384, 1, 1),
                *maps,
                int(S),
                int(A),
                int(B),
                int(SP),
                int(AT),
                int(nab),
                int(nag),
                int(ntile),
                int(ns),
                float(EPS),
                float(EPS),
                WT,
                dmod,
                self.trace,
            )

        run.keep = (maps, WT)
        return run, (dq, dP, dmod)


class AttnDq:
    """Window-attention backward dQ: Qh, Kh, Vh head-major [N, H, S, D] bf16, dO row-major [N S, C] bf16, LSE / Dv [N, H, S] fp32 ->
    dQh head-major bf16."""

    def __init__(self, cubin="attn_dq", trace=None):
        self.k = _load_swa_sm100_kernel(cubin, "swa_attn_dq_sm100", 232448, pdl=PDL_BWD)
        self.trace = trace

    def bind(self, Qh, Kh, Vh, dO, lse, dv, seqused, dQh=None):
        N, _, S, _ = Qh.shape
        assert S % 128 == 0
        rows = N * H * S
        dQh = torch.empty_like(Qh) if dQh is None else dQh
        maps = (
            _map(Qh.view(rows, D), [D, rows], D * 2, [D, 128], swizzle=64),
            _map(Kh.view(rows, D), [D, rows], D * 2, [D, 64], swizzle=64),
            _map(Vh.view(rows, D), [D, rows], D * 2, [D, 64], swizzle=64),
            _map(dO, [C, N * S], C * 2, [D, 128], swizzle=64),
            _map(dQh.view(rows, D), [D, rows], D * 2, [D, 128], swizzle=64),
        )
        items = N * (S // 128) * 2
        grid = (min(num_sms(), items), 1, 1)

        def run():
            self.k(
                grid,
                (384, 1, 1),
                *maps,
                seqused,
                lse,
                dv,
                int(S),
                int(N),
                float(D**-0.5),
                self.trace,
            )

        run.keep = maps
        return run, dQh


class AttnDkv:
    """Window-attention backward dK / dV: Qh, Kh, Vh head-major [N, H, S, D] bf16, dO row-major [N S, C] bf16, LSE / Dv [N, H, S] fp32 ->
    dKh, dVh head-major bf16."""

    def __init__(self, cubin="attn_dkv"):
        self.k = _load_swa_sm100_kernel(cubin, "swa_attn_dkv_sm100", 232448, pdl=PDL_BWD)

    def bind(self, Qh, Kh, Vh, dO, lse, dv, seqused, dKh=None, dVh=None):
        N, _, S, _ = Qh.shape
        assert S % 128 == 0
        rows = N * H * S
        dKh = torch.empty_like(Kh) if dKh is None else dKh
        dVh = torch.empty_like(Vh) if dVh is None else dVh
        maps = (
            _map(Qh.view(rows, D), [D, rows], D * 2, [D, 64], swizzle=64),
            _map(Kh.view(rows, D), [D, rows], D * 2, [D, 128], swizzle=64),
            _map(Vh.view(rows, D), [D, rows], D * 2, [D, 128], swizzle=64),
            _map(dO, [C, N * S], C * 2, [D, 64], swizzle=64),
            _map(
                lse.view(rows // 64, 64),
                [64, rows // 64],
                64 * 4,
                [64, 1],
                swizzle=0,
                dtype="f32",
            ),
            _map(
                dv.view(rows // 64, 64),
                [64, rows // 64],
                64 * 4,
                [64, 1],
                swizzle=0,
                dtype="f32",
            ),
            _map(dKh.view(rows, D), [D, rows], D * 2, [D, 128], swizzle=64),
            _map(dVh.view(rows, D), [D, rows], D * 2, [D, 128], swizzle=64),
        )
        items = N * (S // 128) * 2
        grid = (min(num_sms(), items), 1, 1)

        def run():
            self.k(grid, (384, 1, 1), *maps, seqused, int(S), int(N), float(D**-0.5))

        run.keep = maps
        return run, (dKh, dVh)


class AttnDkvq(AttnDkv):
    """attn_dkvq.cu: AttnDkv plus dQh (head-major bf16, as AttnDq) from the same pass."""

    def __init__(self, cubin="attn_dkvq"):
        self.k = _load_swa_sm100_kernel(cubin, "swa_attn_dkvq_sm100", 232448, pdl=PDL_BWD)
        self.bufs = {}

    def bind(self, Qh, Kh, Vh, dO, lse, dv, seqused, dKh=None, dVh=None, dQh=None):
        N, _, S, _ = Qh.shape
        rows = N * H * S
        run0, (dKh, dVh) = super().bind(Qh, Kh, Vh, dO, lse, dv, seqused, dKh, dVh)
        dQh = torch.empty_like(Qh) if dQh is None else dQh
        maps = run0.keep[:8]
        grid = (min(num_sms(), N * (S // 128) * 2), 1, 1)
        if grid[0] not in self.bufs:  # kept: the kernel resets the counters it uses
            self.bufs[grid[0]] = (
                torch.empty(grid[0] * 8 * 64 * D, device=Qh.device),  # run-boundary partials [CTA][side][head][slot][64][32]
                torch.zeros((grid[0] + 1) * 4, device=Qh.device, dtype=torch.int32),
            )  # pair counters
        bnd, cnt = self.bufs[grid[0]]

        def run():
            self.k(
                grid,
                (384, 1, 1),
                *maps,
                seqused,
                bnd,
                cnt,
                dQh,
                int(S),
                int(N),
                float(D**-0.5),
            )

        run.keep = (maps, bnd, cnt)
        return run, (dKh, dVh, dQh)


class OprojBwd:
    """Out-projection backward (the Triton _oproj_bwd's outputs): dq1, O, G, att [N S, C] bf16, mod [B S, 6C] fp32, Wo ->
    dO, dG, datt, gated [N S, C] bf16, D [N, H, S] fp32; d gate_a accumulated into dmod."""

    def __init__(self, cubin="oproj_bwd"):
        self.k = _load_swa_sm100_kernel(cubin, "swa_oproj_bwd_sm100", 232448, pdl=PDL_BWD)

    def bind(self, dq1, O, G, mod, wo, att, A, B, dmod=None, WOT=None, outs=None):
        M = dq1.numel() // C
        S = M // (A * B)
        SP = min(A, 8)
        AT = 64 // SP
        dev = dq1.device
        if outs is None:
            dO, dG, datt, gated = (torch.empty(M, C, device=dev, dtype=torch.bfloat16) for _ in range(4))
            Dv = torch.empty(A * B, H, S, device=dev)
        else:
            dO, dG, Dv, datt, gated = outs
        own = dmod is None
        if own:
            dmod = torch.zeros(B * S, 6 * C, device=dev)
        WOT = wo.t().contiguous() if WOT is None else WOT
        maps = (
            map_rows(dq1, A, B, S, SP, AT),
            map_rows(att, A, B, S, SP, AT),
            map_rows(G, A, B, S, SP, AT),
            map_rows(O, A, B, S, SP, AT),
            _map(
                mod,
                [32, B * S, 24],
                [6 * C * 4, 128],
                [32, AT, 4],
                swizzle=128,
                dtype="f32",
            ),
            map_rows(datt, A, B, S, SP, AT),
            map_rows(dG, A, B, S, SP, AT),
            map_rows(gated, A, B, S, SP, AT),
            map_rows(dO, A, B, S, SP, AT),
        )
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        grid = (min(num_sms(), ntile), 1, 1)
        auxst = (AT * 512 + 1023) // 1024 * 1024
        ns = min(12, (232448 - 2 * auxst - 512) // 16384)
        assert ns >= 8, (AT, ns)

        def run():
            if own:
                dmod.zero_()
            self.k(
                grid,
                (384, 1, 1),
                *maps,
                int(S),
                int(A),
                int(B),
                int(SP),
                int(AT),
                int(nab),
                int(nag),
                int(ntile),
                int(ns),
                WOT,
                dmod,
                Dv,
            )

        run.keep = (maps, WOT)
        return run, (dO, dG, Dv, datt, gated, dmod)


class FfnBwdGate:
    """First half of the FFN backward: dq2, y [N S, C] bf16, mod [B S, 6C] fp32, Wu [512, C], Wd [C, 256] -> DFFN [M, C], HH [M, 256],
    DAB [M, 512] (bf16)."""

    def __init__(self, cubin="ffn_bwd_gate"):
        self.k = _load_swa_sm100_kernel(cubin, "swa_ffn_bwd_gate_sm100", 232448, pdl=PDL_BWD)

    def bind(self, dq2, y, mod, wu, wd, A, B, WDT=None, outs=None):
        M = dq2.numel() // C
        S = M // (A * B)
        SP = min(A, 4)
        AT = 32 // SP
        dev = dq2.device
        if outs is None:
            dffn = torch.empty(M, C, device=dev, dtype=torch.bfloat16)
            hh = torch.empty(M, 256, device=dev, dtype=torch.bfloat16)
            dab = torch.empty(M, 512, device=dev, dtype=torch.bfloat16)
        else:
            dffn, hh, dab = outs
        WDT = wd.t().contiguous() if WDT is None else WDT
        st = lambda t, w: _map(t, [w, S, B, A], [w * 2, S * w * 2, B * S * w * 2], [64, AT, 1, SP])
        maps = (
            map_rows(y, A, B, S, SP, AT),
            map_rows(dq2, A, B, S, SP, AT),
            _map(
                mod,
                [32, B * S, 24],
                [6 * C * 4, 128],
                [32, AT, 4],
                swizzle=128,
                dtype="f32",
            ),
            map_rows(dffn, A, B, S, SP, AT),
            st(hh, 256),
            st(dab, 512),
        )
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        grid = (min(num_sms(), ntile), 1, 1)
        auxst = (AT * 512 + 1023) // 1024 * 1024
        ns = min(8, (232448 - 2 * 49152 - 2 * auxst - 512) // 8192)
        assert ns >= 4, (AT, ns)

        def run():
            self.k(
                grid,
                (384, 1, 1),
                *maps,
                int(S),
                int(A),
                int(B),
                int(SP),
                int(AT),
                int(nab),
                int(nag),
                int(ntile),
                int(ns),
                wu,
                WDT,
            )

        run.keep = (maps, WDT)
        return run, (dffn, hh, dab)


class FfnBwdDy:
    """Second half of the FFN backward: DAB [M, 512], dq2, q1, ffn [M, C] bf16, mod, Wu -> dq1 [M, C] bf16; d shift_f / scale_f / gate_f
    accumulated into dmod."""

    def __init__(self, cubin="ffn_bwd_dy", trace=None):
        self.k = _load_swa_sm100_kernel(cubin, "swa_ffn_bwd_dy_sm100", 232448, pdl=PDL_BWD)
        self.trace = trace

    def bind(self, dab, dq2, q1, ffn, mod, wu, A, B, dmod=None, WUT=None, dq1=None):
        M = dq2.numel() // C
        S = M // (A * B)
        SP = min(A, 8)
        AT = 64 // SP
        dev = dq2.device
        dq1 = torch.empty(M, C, device=dev, dtype=torch.bfloat16) if dq1 is None else dq1
        own = dmod is None
        if own:
            dmod = torch.zeros(B * S, 6 * C, device=dev)
        WUT = wu.t().contiguous() if WUT is None else WUT
        maps = (
            _map(dab, [512, S, B, A], [1024, S * 1024, B * S * 1024], [64, AT, 1, SP]),
            map_rows(q1, A, B, S, SP, AT),
            map_rows(ffn, A, B, S, SP, AT),
            map_rows(dq2, A, B, S, SP, AT),
            _map(
                mod,
                [32, B * S, 24],
                [6 * C * 4, 128],
                [32, AT, 4],
                swizzle=128,
                dtype="f32",
            ),
            map_rows(dq1, A, B, S, SP, AT),
        )
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        grid = (min(num_sms(), ntile), 1, 1)
        auxst = (AT * 512 + 1023) // 1024 * 1024
        ns = min(28, (232448 - 2 * auxst - 2048 - 512 - 1024) // 8192)
        assert ns >= 14, (AT, ns)

        def run():
            if own:
                dmod.zero_()
            self.k(
                grid,
                (384, 1, 1),
                *maps,
                int(S),
                int(A),
                int(B),
                int(SP),
                int(AT),
                int(nab),
                int(nag),
                int(ntile),
                int(ns),
                float(EPS),
                WUT,
                dmod,
                self.trace,
            )

        run.keep = (maps, WUT)
        return run, (dq1, dmod)


class FfnFwd2:
    """Transposed out-projection + FFN forward (ffn_fwd2.cu): same inputs / outputs as FfnFwd, 32-row tiles, weights on chip."""

    def __init__(self, cubin="ffn_fwd2", trace=None, cluster=1, max_ctas=None, atm=8):
        self.k = _load_swa_sm100_kernel(
            cubin,
            "swa_ffn_fwd2_sm100",
            232448,
            cluster=cluster if cluster > 1 else None,
            pdl=PDL,
        )
        self.atm = atm  # the build's ATM (ffn_fwd2_a1.cubin: 32, for A = 1 - 3)
        self.cl = cluster
        self.trace = trace
        self.max_ctas = max_ctas  # co-resident CTA cap (clusters must fit in a GPC)
        self.trace = trace

    def bind(self, q, g, o, mod, wo, wu, wd, A, B, save=False):
        M = q.numel() // C
        S = M // (A * B)
        SP = min(A, 8)
        AT = 32 // SP
        assert AT <= self.atm, f"AT = {AT} needs a build with ATM >= {AT}"
        dev = q.device
        out = torch.empty(M, C, device=dev, dtype=torch.bfloat16)
        Q1, Att, Y, Ff = (torch.empty(M, C, device=dev, dtype=torch.bfloat16) for _ in range(4)) if save else (out, out, out, out)
        mr = lambda t: map_rows(t, A, B, S, SP, AT)
        maps = (
            mr(q),
            mr(g),
            mr(o),
            _map(
                mod,
                [32, B * S, 24],
                [6 * C * 4, 128],
                [32, AT, 16],
                swizzle=128,
                dtype="f32",
            ),
            _map(wo, [C, C], C * 2, [64, 128]),
            mr(out),
            mr(Q1),
            mr(Att),
            mr(Y),
            mr(Ff),
            _map(wu, [C, 512], C * 2, [32, 128], swizzle=64),
            _map(wd, [256, C], 256 * 2, [32, 128], swizzle=64),
        )
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        cl = self.cl
        grid = (
            max(cl, min(self.max_ctas or num_sms(), ntile) // cl * cl),
            1,
            1,
        )  # a multiple of the cluster size (idle CTAs still multicast)

        def run():
            self.k(
                grid,
                (384, 1, 1),
                *maps,
                int(S),
                int(A),
                int(B),
                int(SP),
                int(AT),
                int(nab),
                int(nag),
                int(ntile),
                float(EPS),
                int(save),
                wu,
                wd,
                self.trace,
            )

        run.keep = maps
        return run, (out, Q1, Att, Y, Ff)


class AttnFwd2:
    """Window attention forward, two 128-key blocks per 128-query tile (attn_fwd2.cu): same interface as AttnFwd."""

    def __init__(self, cubin="attn_fwd2", trace=None):
        self.k = _load_swa_sm100_kernel(cubin, "swa_attn_fwd2_sm100", 232448, pdl=PDL)
        self.trace = trace

    threads = 384
    heads_per_item = 2

    def bind(self, Qh, Kh, Vh, seqused, O=None, LSE=None):
        N, _, S, _ = Qh.shape
        assert S % 128 == 0
        O = torch.empty(N * S, C, device=Qh.device, dtype=torch.bfloat16) if O is None else O
        LSE = torch.empty(N, H, S, device=Qh.device) if LSE is None else LSE
        rows = N * H * S
        maps = (
            _map(Qh.view(rows, D), [D, rows], D * 2, [D, 128], swizzle=64),
            _map(Kh.view(rows, D), [D, rows], D * 2, [D, 256], swizzle=64),
            _map(Vh.view(rows, D), [D, rows], D * 2, [D, 256], swizzle=64),
            _map(O, [C, N * S], C * 2, [D, 128], swizzle=64),
        )
        items = N * (S // 128) * (H // self.heads_per_item)
        grid = (min(num_sms(), items), 1, 1)

        def run():
            self.k(
                grid,
                (self.threads, 1, 1),
                *maps,
                seqused,
                LSE,
                int(S),
                int(N),
                float(D**-0.5),
                self.trace,
            )

        run.keep = maps
        return run, O, LSE


class AttnFwd3(AttnFwd2):
    """attn_fwd3.cu: attn_fwd2 with the two heads' P phases taking turns and mask-free full chunks (bitwise equal)."""

    def __init__(self, cubin="attn_fwd3", trace=None):
        self.k = _load_swa_sm100_kernel(cubin, "swa_attn_fwd3_sm100", 232448, pdl=PDL)
        self.trace = trace


class AttnFwd1h(AttnFwd2):
    """attn_fwd1h.cu: one head per item, the two warpgroups splitting its keys (few items: small inference batches)."""

    heads_per_item = 1

    def __init__(self, cubin="attn_fwd1h", trace=None):
        self.k = _load_swa_sm100_kernel(cubin, "swa_attn_fwd1h_sm100", 232448, pdl=PDL)
        self.trace = trace


#: :class:`GattnFwd` / :class:`GattnBwd` split mode where the busiest CTA's work of whole items exceeds the even share by this factor
#: (the split build runs ~10 % slower a block: 2026-11-17, N 1 S 7168 1.32 -> 90.0 / 86.0 us, S 8192 1.16 -> 100.4 / 104.4)
SPLIT_GAIN = 1.3
#: ... and at least this many blocks (64 keys / 64 queries) per CTA (shorter pieces cost more than the idle SMs: forward N 1 S 2048
#: 30.6 / 34.7 us, N 5 S 1024 20.4 / 34.8; backward N 1 S 2048 59.4 / 69.5)
SPLIT_MIN_BLOCKS = 12
#: the forward's split gain (r3-swa part 18: N 5 S 6144, 1.235: whole 280.4 / split 245.5 us bf16, 321.5 / 253.7 F32IO)
SPLIT_GAIN_FWD = 1.2
#: :class:`GattnFwd` cluster split (r3-swa part 17, gattn_fwd.cu CSPLIT): one item per cluster of CS CTAs where items x CS fit the SMs
CSPLIT_FWD = os.environ.get("SWA_CSPLIT", "1") != "0"
CSPLIT_SIZES = tuple(int(x) for x in os.environ.get("SWA_CSPLIT_SIZES", "8,4,2").split(","))
CSPLIT_MIN_BLOCKS = int(os.environ.get("SWA_CSPLIT_MIN", "4"))  # key blocks (64) a rank at least
#: :class:`GattnBwd` cluster split (r3-swa part 18, gattn_bwd.cu CSB): one item per cluster of CS CTAs, query shares
CSB_BWD = os.environ.get("SWA_CSB", "1") != "0"
CSB_SIZES = tuple(int(x) for x in os.environ.get("SWA_CSB_SIZES", "8,4,3,2").split(","))
CSB_MIN_BLOCKS = int(os.environ.get("SWA_CSB_MIN", "4"))  # query blocks (64) a rank at least


def _split_pays(items: int, steps: int, grid: int, max_items: int, gain: float = SPLIT_GAIN) -> bool:
    """Split mode for ``items`` work items of ``steps`` blocks each on a whole-item grid of ``grid`` CTAs (``gain``: :data:`SPLIT_GAIN`)."""
    g = num_sms()
    total = items * steps
    if items > max_items or total < SPLIT_MIN_BLOCKS * g:
        return False
    return -(-items // grid) * steps >= gain * total / g


class GattnFwd:
    """Global attention forward (gattn_fwd.cu): Qh, Kh, Vh head-major [N, H, S, D] bf16, seqused [N] int32 -> O [N S, C] bf16 (every row
    written: zeros past seqused) and, with ``LSE`` [N, H, S] fp32, the row sums' log (0 past seqused): natural (FA4's convention) or, with
    ``log2``, the negated log2 -log2(sum) the native backward (:class:`GattnBwd`) reads."""

    threads = 384
    smem = 232448  # the whole shared memory (the kernel uses 99328): one CTA per SM -- two would share an SM's TMEM (512 columns each)

    def __init__(self, cubin="gattn_fwd", trace=None, cl=1, bn=64, split_cubin=None, f32io: int = 0, csplit_cubin=None):
        self.k = _load_swa_sm100_kernel(cubin, "swa_gattn_fwd_sm100", self.smem, cluster=cl if cl > 1 else None, pdl=PDL)
        # the split-mode build (SPLITB=1; None: ``cubin`` itself decides by the runtime flag, a SPLITB=-1 build)
        self.ks = self.k if split_cubin is None else _load_swa_sm100_kernel(split_cubin, "swa_gattn_fwd_sm100", self.smem, pdl=PDL)
        self.trace = trace
        self.cl = cl  # the build's CL: clusters of cl CTAs share (multicast) the K | V stream
        self.bn = bn  # the build's BNK: keys per block
        self.split = None  # None: automatic (:meth:`splits`); True / False: forced (development)
        self.ws = {}  # split mode, per device: WS (partial O / row sums, [grid][2][2][128 x 33] fp32) and CNT ([items][2] int32, zero between launches)
        self.f32io = f32io  # the build's F32IO (0: bf16 io; 1: q, K, V fp32; 2: q fp32): O fp32 [N S, C] (bf16 values) when > 0
        self.csplit_cubin = csplit_cubin  # the cluster-split build (CSPLIT=1), loaded per cluster size on first use
        self.kcs = {}
        self.cs = None  # None: automatic (:meth:`cluster_split`); an int: forced cluster size (0 / 1: off; development)

    def _kc(self, cs: int):
        k = self.kcs.get(cs)
        if k is None:
            k = self.kcs[cs] = _load_swa_sm100_kernel(self.csplit_cubin, "swa_gattn_fwd_sm100", self.smem, cluster=cs, pdl=PDL)
        return k

    def cluster_split(self, N: int, S: int) -> int:
        """Cluster size of the cluster split (0: none): the largest of :data:`CSPLIT_SIZES` whose items x CS CTAs fit the SMs at once with
        at least :data:`CSPLIT_MIN_BLOCKS` key blocks a rank."""
        if self.csplit_cubin is None or self.cl != 1 or not CSPLIT_FWD:
            return 0
        items = N * H * ((S // 128 + 1) // 2)
        for cs in CSPLIT_SIZES:
            if items * cs <= num_sms() and (S // self.bn) >= CSPLIT_MIN_BLOCKS * cs:
                return cs
        return 0

    def bind(self, Qh, Kh, Vh, seqused, O=None, LSE=None, log2=False):
        N, _, S, _ = Qh.shape
        assert S % 128 == 0
        lmul = -1.0 if log2 else 0.6931471805599453  # log2: the NEGATED log2 (gattn_bwd adds it: no negation per element)
        f32 = self.f32io  # 1: q, K, V fp32; 2: q fp32, K, V bf16; O fp32 in both
        dt, dkv = (torch.float32 if f32 else torch.bfloat16), (torch.float32 if f32 == 1 else torch.bfloat16)
        assert Qh.dtype == dt and Kh.dtype == Vh.dtype == dkv
        O = torch.empty(N * S, C, device=Qh.device, dtype=dt) if O is None else O
        assert O.dtype == dt and O.is_contiguous()
        rows = N * H * S
        kb = self.bn // max(self.cl // 2, 1)  # K / V box rows: CL 4 splits each of K, V in two

        def m(t, b):  # fp32: raw rows (128 B, no swizzle) into the kernel's staging; bf16: the 64-B swizzled tiles
            if t.dtype == torch.float32:
                return _map(t.view(rows, D), [D, rows], D * 4, [D, b], swizzle=0, dtype="f32")
            return _map(t.view(rows, D), [D, rows], D * 2, [D, b], swizzle=64)

        # fp32 O: written by plain stores (its map unused)
        mo = m(Qh, 128) if f32 else _map(O, [C, N * S], C * 2, [D, 128], swizzle=64)
        maps = (m(Qh, 128), m(Kh, kb), m(Vh, kb), mo)
        o32 = O if f32 else None
        cl = self.cl
        items = N * H * ((((S // 128 + 1) // 2) + cl - 1) // cl)
        cs = self.cluster_split(N, S) if self.cs is None else (self.cs if self.csplit_cubin is not None and cl == 1 else 0)
        if cs > 1:
            kc = self._kc(cs)
            grid = (items * cs, 1, 1)

            def run_c():  # O by plain stores (its pointer in the O32 slot, bf16 or fp32)
                kc(grid, (self.threads, 1, 1), *maps, seqused, LSE, int(S), int(N), float(D**-0.5), lmul, None, None, 0, self.trace, O)

            run_c.keep = (maps,)
            return run_c, O, LSE
        split = self.splits(N, S) if self.split is None else (self.split and cl == 1 and N <= 256 and items <= 4 * num_sms())
        if split:
            grid = (min(num_sms(), items * (S // self.bn)), 1, 1)
            ws, cnt = self.ws.get(Qh.device, (None, None))
            if ws is None or ws.numel() < grid[0] * 4 * (128 * 33) or cnt.numel() < 2 * items:
                ws = torch.empty(num_sms() * 4 * (128 * 33), device=Qh.device, dtype=torch.float32)
                cnt = torch.zeros(max(2 * items, 4096), device=Qh.device, dtype=torch.int32)
                self.ws[Qh.device] = (ws, cnt)
        else:
            grid = (cl * min(num_sms() // cl, items), 1, 1)
            ws = cnt = None

        def run():
            (self.ks if split else self.k)(
                grid, (self.threads, 1, 1), *maps, seqused, LSE, int(S), int(N), float(D**-0.5), lmul, ws, cnt, int(split), self.trace, o32
            )

        run.keep = (maps, ws, cnt)
        return run, O, LSE

    def splits(self, N: int, S: int) -> bool:
        """Split mode (the key range of an item cut across CTAs, :file:`gattn_fwd.cu`) where whole items fill the persistent grid's
        rounds poorly (:func:`_split_pays`: one sample at L 3072 - 7168, five samples at L 2048 / 4096). At most 4 items per SM (the kernel's piece list) and 256 samples (its prefix table)."""
        if self.cl != 1 or N > 256:
            return False
        items = N * H * ((S // 128 + 1) // 2)
        return _split_pays(items, S // self.bn, min(num_sms(), items), 4 * num_sms(), SPLIT_GAIN_FWD)


class GattnBwd:
    """Global attention backward (gattn_bwd.cu, deterministic): Qh, Kh, Vh head-major [N, H, S, D] bf16, dO token-major [N S, C] bf16,
    LSE2 [N, H, S] fp32 (the negated log2 :class:`GattnFwd` writes with ``log2``), Dv [N, H, S] fp32 (sum_d dO o per head), seqused [N] int32 ->
    dQh, dKh, dVh head-major bf16 (every row written: zeros past seqused). S a multiple of 256."""

    threads = 384
    smem = 232448  # one CTA per SM: the items of a head wait on each other's dQ adds, so every CTA must be resident

    def __init__(self, cubin="gattn_bwd", trace=None, split_cubin=None, csb_cubin=None):
        self.k = _load_swa_sm100_kernel(cubin, "swa_gattn_bwd_sm100", self.smem, pdl=PDL_BWD)
        # the split-mode build (SPLITB=1, :meth:`splits`)
        self.ks = None if split_cubin is None else _load_swa_sm100_kernel(split_cubin, "swa_gattn_bwd_sm100", self.smem, pdl=PDL_BWD)
        self.csb_cubin = csb_cubin  # the cluster-split build (CSB=1), loaded per cluster size on first use (:meth:`cluster_split`)
        self.kcs = {}
        self.maxcl = {}
        self.cs = None  # None: automatic; an int: forced cluster size (0 / 1: off; development)
        self.split = None  # None: automatic; True / False: forced (development)
        self.trace = trace  # a TRACE=1 build: uint64 [64 x 16] clock64 stamps of CTA 0
        self.cnt = {}  # device counters [N H S / 64] int32, zero between launches (the last adder of a block resets its counter)
        self.ws = {}  # split mode, per device: WS (dK | dV partials, [grid][2][2][128 x 64] fp32), CNT2 ([items][2] int32, zero between launches)

    def splits(self, N: int, S: int) -> bool:
        """Split mode where whole items leave SMs idle (:func:`_split_pays`: one sample at L 3072 - 7168,
        S / 64 items of 148 SMs; five samples at L 2048). At most 4 items per SM (the kernel's piece list) and 256 samples (its prefix table)."""
        if self.ks is None or N > 256:
            return False
        return _split_pays(N * H * (S // 256), S // 64, self.grid(N, S), 4 * num_sms())

    def _kc(self, cs: int, ipc: int = 1):
        """The CSB kernel for ``cs`` query shares an item, ``ipc`` items a cluster (cluster size cs x ipc; ipc > 1 needs the per-size
        build, whose CSN fixes the shares)."""
        k = self.kcs.get((cs, ipc))
        if k is None:
            per = f"{self.csb_cubin}{cs}" in CUBINS
            assert per or ipc == 1, "several items a cluster: the per-size (CSN) build"
            name = f"{self.csb_cubin}{cs}" if per else self.csb_cubin
            k = self.kcs[(cs, ipc)] = _load_swa_sm100_kernel(name, "swa_gattn_bwd_sm100", self.smem, cluster=cs * ipc, pdl=PDL_BWD)
        return k

    def _max_clusters(self, cs: int, ipc: int = 1) -> int:
        """Clusters of cs x ipc CTAs of the CSB build co-resident on the device (cuOccupancyMaxActiveClusters; GPC granularity)."""
        m = self.maxcl.get((cs, ipc))
        if m is None:
            from cuda.bindings import driver as cu

            k = self._kc(cs, ipc)
            cfg = cu.CUlaunchConfig()
            cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = cs * ipc, 1, 1
            cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = self.threads, 1, 1
            cfg.sharedMemBytes = self.smem
            at = cu.CUlaunchAttribute()
            at.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            at.value.clusterDim.x, at.value.clusterDim.y, at.value.clusterDim.z = cs * ipc, 1, 1
            cfg.attrs = [at]
            cfg.numAttrs = 1
            err, m = cu.cuOccupancyMaxActiveClusters(k.func, cfg)
            m = int(m) if err == cu.CUresult.CUDA_SUCCESS else 0
            self.maxcl[(cs, ipc)] = m
        return m

    def fits(self, items: int, cs: int, ipc: int = 1) -> bool:
        """items x cs CTAs in clusters of cs x ipc all co-resident (every CTA must be: the dQ adders of a block wait on each other)."""
        # the kernel's receive area holds CS 2 / 3 / 4 / 8 (16 CS ceil32(ceil(256 / CS)) x 16 B <= 72 KB)
        return cs in (2, 3, 4, 8) and items % ipc == 0 and cs * ipc <= 8 and items // ipc <= self._max_clusters(cs, ipc)

    def cluster_split(self, N: int, S: int) -> tuple[int, int]:
        """(query shares CS, items a cluster) of the cluster split ((0, 1): none): the largest of :data:`CSB_SIZES` with at least
        :data:`CSB_MIN_BLOCKS` query blocks a rank whose items x CS CTAs are co-resident in clusters of one item, else of two
        (CS 3 at L 3072: 48 clusters of 3 do not fit, 24 of 6 do)."""
        if self.csb_cubin is None or not CSB_BWD:
            return 0, 1
        items = N * H * (S // 256)
        for cs in CSB_SIZES:
            if S // 64 < CSB_MIN_BLOCKS * cs:
                continue
            for ipc in (1, 2):
                if (ipc == 1 or f"{self.csb_cubin}{cs}" in CUBINS) and self.fits(items, cs, ipc):
                    return cs, ipc
        return 0, 1

    @staticmethod
    def grid(N: int, S: int) -> int:
        """Whole heads per round: a multiple of the S / 256 items of a head, at most one CTA per SM."""
        P = S // 256
        return min(max(num_sms() // P, 1) * P, N * H * P)

    def bind(self, Qh, Kh, Vh, dO, lse2, dv, seqused, dQh=None, dKh=None, dVh=None):
        N, _, S, _ = Qh.shape
        assert S % 256 == 0 and num_sms() >= S // 256
        rows = N * H * S
        dQh = torch.empty_like(Qh) if dQh is None else dQh
        dKh = torch.empty_like(Kh) if dKh is None else dKh
        dVh = torch.empty_like(Vh) if dVh is None else dVh
        maps = (
            _map(Qh.view(rows, D), [D, rows], D * 2, [D, 64], swizzle=64),
            _map(Kh.view(rows, D), [D, rows], D * 2, [D, 128], swizzle=64),
            _map(Vh.view(rows, D), [D, rows], D * 2, [D, 128], swizzle=64),
            _map(dO, [C, N * S], C * 2, [D, 64], swizzle=64),
            _map(lse2.view(rows // 64, 64), [64, rows // 64], 64 * 4, [64, 1], swizzle=0, dtype="f32"),
            _map(dv.view(rows // 64, 64), [64, rows // 64], 64 * 4, [64, 1], swizzle=0, dtype="f32"),
            _map(dKh.view(rows, D), [D, rows], D * 2, [D, 128], swizzle=64),
            _map(dVh.view(rows, D), [D, rows], D * 2, [D, 128], swizzle=64),
        )
        ncnt = rows // 64
        if self.cnt.get(Qh.device) is None or self.cnt[Qh.device].numel() < ncnt:
            self.cnt[Qh.device] = torch.zeros(ncnt, device=Qh.device, dtype=torch.int32)
        cnt = self.cnt[Qh.device]
        # every slot a query block reaches is stored before it is added to
        acc = torch.empty(rows * D, device=Qh.device, dtype=torch.float32)
        items = N * H * (S // 256)
        if self.cs is None:
            cs, ipc = self.cluster_split(N, S)
        else:  # forced (development): an int (one item a cluster) or (cs, ipc)
            cs, ipc = (self.cs, 1) if isinstance(self.cs, int) else self.cs
            cs = cs if self.csb_cubin is not None else 0
        if cs > 1:
            assert self.fits(items, cs, ipc), "CSB: every cluster co-resident"
            kc = self._kc(cs, ipc)
            grid_c = (items * cs, 1, 1)

            def run_c():
                kc(grid_c, (self.threads, 1, 1), *maps, seqused, acc, cnt, dQh, dKh, dVh, int(S), int(N), float(D**-0.5), None, None, self.trace)

            run_c.keep = (maps, acc, cnt)
            return run_c, (dQh, dKh, dVh)
        split = self.splits(N, S) if self.split is None else (self.split and self.ks is not None and N <= 256 and items <= 4 * num_sms())
        if split:
            grid = (num_sms(), 1, 1)  # every CTA resident (the dQ adders of a block wait on each other)
            ws, cnt2 = self.ws.get(Qh.device, (None, None))
            if ws is None or cnt2.numel() < 2 * items:
                ws = torch.empty(num_sms() * 4 * 128 * 64, device=Qh.device, dtype=torch.float32)
                cnt2 = torch.zeros(max(2 * items, 4096), device=Qh.device, dtype=torch.int32)
                self.ws[Qh.device] = (ws, cnt2)
            kern = self.ks
        else:
            grid = (self.grid(N, S), 1, 1)
            ws = cnt2 = None
            kern = self.k

        def run():
            kern(grid, (self.threads, 1, 1), *maps, seqused, acc, cnt, dQh, dKh, dVh, int(S), int(N), float(D**-0.5), ws, cnt2, self.trace)

        run.keep = (maps, acc, cnt, ws, cnt2)
        return run, (dQh, dKh, dVh)


class QkvgFwd2:
    """Transposed qkvg forward (qkvg_fwd2.cu): same interface as QkvgFwd, 32-row tiles, W in TMEM."""

    def __init__(self, cubin="qkvg_fwd2", atm=8, max_ctas=None):
        self.k = _load_swa_sm100_kernel(cubin, "swa_qkvg_fwd2_sm100", 232448, pdl=PDL)
        self.max_ctas = max_ctas
        self.atm = atm  # the build's ATM (qkvg_fwd2_a1.cubin: 32, for A = 1 - 3)

    def bind(self, q, mod, cos, sin, wqkv, wg, A, B, save=False, W=None):
        N, S, _ = q.shape
        SP = min(A, 8)
        AT = 32 // SP
        assert AT <= self.atm, f"AT = {AT} needs a build with ATM >= {AT}"
        dev = q.device
        W = torch.cat([wqkv, wg]).contiguous() if W is None else W
        Qh, Kh, Vh = (torch.empty(N, H, S, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
        G = torch.empty(N * S, C, device=dev, dtype=torch.bfloat16)
        X, PQ, PK = (torch.empty(N * S, C, device=dev, dtype=torch.bfloat16) for _ in range(3)) if save else (G, G, G)
        mr = lambda t: map_rows(t, A, B, S, SP, AT)
        mh = lambda t: map_heads(t, A, B, S, SP, AT)
        maps = (
            mr(q),
            _map(W, [C, 4 * C], C * 2, [32, 128], swizzle=64),
            _map(
                mod,
                [32, B * S, 24],
                [6 * C * 4, 128],
                [32, AT, 8],
                swizzle=128,
                dtype="f32",
            ),
            _map(cos, [D // 2, B * S], [D // 2 * 4], [D // 2, AT], swizzle=0, dtype="f32"),
            _map(sin, [D // 2, B * S], [D // 2 * 4], [D // 2, AT], swizzle=0, dtype="f32"),
            mh(Qh),
            mh(Kh),
            mh(Vh),
            mr(G),
            mr(X),
            mr(PQ),
            mr(PK),
        )
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        grid = (min(self.max_ctas or num_sms(), ntile), 1, 1)

        def run():
            self.k(
                grid,
                (384, 1, 1),
                *maps,
                int(S),
                int(A),
                int(B),
                int(SP),
                int(AT),
                int(nab),
                int(nag),
                int(ntile),
                float(EPS),
                float(EPS),
                int(save),
            )

        run.keep = (maps, W)
        return run, (Qh, Kh, Vh, G, X, PQ, PK)


FORCE_FWD3 = False  # development switch: the many-tile kernels (qkvg_fwd3 / ffn_fwd3) in the few-tile range too
FEW_TILE = True  # round 3: the bf16 inference block of at most one 32-row tile per SM on the few-tile row kernels (modulation inside)


def few_tiles(N: int, S: int) -> bool:
    """One 32-row tile per CTA, at most one CTA per SM: the range of ``block_fwd_c`` (qkvg_fwd1t / ffn_fwd1t)."""
    return FEW_TILE and N * S <= 32 * num_sms()


def _map_w64(w, n):
    """A bf16 weight [n, C] as K-major 64-B-swizzled [128 rows][32] atoms (the MMA A operand / tcgen05.cp source layout)."""
    return _map(w, [C, n], C * 2, [32, 128], swizzle=64)


#: B200 (2026-11-18, round 3 part 12): qkvg_fwd1t on two CTAs per tile where 2 x tiles fit the SMs (rank r: projections r, r + 2)
PAIR1T = __import__("os").environ.get("SWA_PAIR1T", "1") == "1"


class QkvgFwd1t:
    """qkvg_fwd1t.cu: the qkvg stage of few tiles with the modulation inside -- q [N, S, C] bf16, c [N S, C] bf16, Wmod [6C, C] bf16,
    W = [Wqkv; Wg] bf16, cos / sin [N S, D/2] fp32 -> Qh, Kh, Vh [N, H, S, D], G [N S, C]."""

    def __init__(self, cubin="qkvg_fwd1t", trace=None):
        self.trace = trace
        self.k = _load_swa_sm100_kernel(cubin, "swa_qkvg_fwd1t_sm100", 232448, pdl=PDL)

    def bind(self, q, c, wmod, cos, sin, W, save=False, busy=0):
        """``save`` (the SAVE build, round 3 part 8): also X, PQ, PK [N S, C] bf16 (qkvg_fwd2's saves); returns them after G. ``busy``:
        SMs a kernel beside this one holds (the pair build needs 2 x tiles + busy <= SMs)."""
        N, S, _ = q.shape
        dev = q.device
        Qh, Kh, Vh = (torch.empty(N, H, S, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
        G = torch.empty(N * S, C, device=dev, dtype=torch.bfloat16)
        sv = [torch.empty(N * S, C, device=dev, dtype=torch.bfloat16) for _ in range(3)] if save else [G, G, G]
        maps = (
            map_rows(q, 1, N, S, 1, 32),
            _map_w64(W, 4 * C),
            _map_w64(wmod, 6 * C),
            _map(c, [C, N * S], C * 2, [64, 32]),
            _map(cos, [D // 2, N * S], [D // 2 * 4], [D // 2, 32], swizzle=0, dtype="f32"),
            _map(sin, [D // 2, N * S], [D // 2 * 4], [D // 2, 32], swizzle=0, dtype="f32"),
            map_heads(Qh, 1, N, S, 1, 32),
            map_heads(Kh, 1, N, S, 1, 32),
            map_heads(Vh, 1, N, S, 1, 32),
            map_rows(G, 1, N, S, 1, 32),
            *(map_rows(t, 1, N, S, 1, 32) for t in sv),
        )
        nab = S // 32
        pair = int(PAIR1T and 2 * N * nab + busy <= num_sms())
        grid = (N * nab << pair, 1, 1)

        def run():
            self.k(grid, (384, 1, 1), *maps, int(S), int(nab), float(EPS), float(EPS), pair, self.trace)

        run.keep = (maps, W, wmod, c)
        return run, (Qh, Kh, Vh, G, *sv) if save else (Qh, Kh, Vh, G)


class FfnFwd1t:
    """ffn_fwd1t.cu: the out-projection + FFN stage of few tiles with the modulation inside -- q, g, o [N S, C] bf16, c [N S, C] bf16,
    Wmod [6C, C], Wo, Wu, Wd bf16 -> out [N S, C]."""

    def __init__(self, cubin="ffn_fwd1t", trace=None):
        self.trace = trace
        self.k = _load_swa_sm100_kernel(cubin, "swa_ffn_fwd1t_sm100", 232448, pdl=PDL)

    def bind(self, q, g, o, c, wmod, wo, wu, wd, N):
        M = q.numel() // C
        S = M // N
        out = torch.empty(M, C, device=q.device, dtype=torch.bfloat16)
        mr = lambda t: map_rows(t, 1, N, S, 1, 32)
        maps = (
            mr(q),
            mr(g),
            mr(o),
            _map(c, [C, M], C * 2, [64, 32]),
            _map(wo, [C, C], C * 2, [64, 128]),
            mr(out),
            _map_w64(wu, 2 * NHID),
            _map(wd, [NHID, C], NHID * 2, [32, 128], swizzle=64),
            _map_w64(wmod, 6 * C),
        )
        nab = S // 32
        grid = (N * nab, 1, 1)

        def run():
            self.k(grid, (384, 1, 1), *maps, int(S), int(nab), float(EPS), self.trace)

        run.keep = (maps, wmod, c)
        return run, out


class QkvgFwd3(QkvgFwd1t):
    """qkvg_fwd3.cu: QkvgFwd1t's interface for any tile count (grid = min(tiles, SMs), a CTA loops over its tiles)."""

    def __init__(self, cubin="qkvg_fwd3", trace=None):
        self.trace = trace
        self.k = _load_swa_sm100_kernel(cubin, "swa_qkvg_fwd3_sm100", 232448, pdl=PDL)

    def bind(self, q, c, wmod, cos, sin, W):
        N, S, _ = q.shape
        dev = q.device
        Qh, Kh, Vh = (torch.empty(N, H, S, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
        G = torch.empty(N * S, C, device=dev, dtype=torch.bfloat16)
        maps = (
            map_rows(q, 1, N, S, 1, 32),
            _map_w64(W, 4 * C),
            _map_w64(wmod, 6 * C),
            _map(c, [C, N * S], C * 2, [64, 32]),
            _map(cos, [D // 2, N * S], [D // 2 * 4], [D // 2, 32], swizzle=0, dtype="f32"),
            _map(sin, [D // 2, N * S], [D // 2 * 4], [D // 2, 32], swizzle=0, dtype="f32"),
            map_heads(Qh, 1, N, S, 1, 32),
            map_heads(Kh, 1, N, S, 1, 32),
            map_heads(Vh, 1, N, S, 1, 32),
            map_rows(G, 1, N, S, 1, 32),
        )
        nab = S // 32
        ntile = N * nab
        grid = (min(num_sms(), ntile), 1, 1)

        def run():
            self.k(grid, (384, 1, 1), *maps, int(S), int(nab), int(ntile), float(EPS), float(EPS), self.trace)

        run.keep = (maps, W, wmod, c)
        return run, (Qh, Kh, Vh, G)


class FfnFwd3(FfnFwd1t):
    """ffn_fwd3.cu: FfnFwd1t's interface for any tile count (grid = min(tiles, SMs), a CTA loops over its tiles)."""

    def __init__(self, cubin="ffn_fwd3", trace=None):
        self.trace = trace
        self.k = _load_swa_sm100_kernel(cubin, "swa_ffn_fwd3_sm100", 232448, pdl=PDL)

    def bind(self, q, g, o, c, wmod, wo, wu, wd, N):
        M = q.numel() // C
        S = M // N
        out = torch.empty(M, C, device=q.device, dtype=torch.bfloat16)
        mr = lambda t: map_rows(t, 1, N, S, 1, 32)
        maps = (
            mr(q),
            mr(g),
            mr(o),
            _map(c, [C, M], C * 2, [64, 32]),
            _map(wo, [C, C], C * 2, [64, 128]),
            mr(out),
            _map_w64(wu, 2 * NHID),
            _map(wd, [NHID, C], NHID * 2, [32, 128], swizzle=64),
            _map_w64(wmod, 6 * C),
        )
        nab = S // 32
        ntile = N * nab
        grid = (min(num_sms(), ntile), 1, 1)

        def run():
            self.k(grid, (384, 1, 1), *maps, int(S), int(nab), int(ntile), float(EPS), self.trace)

        run.keep = (maps, wmod, c)
        return run, out


class ModFwd:
    """adaLN modulation in one kernel (mod_fwd.cu): c [R, C] bf16, Wmod [6C, C] bf16 -> silu(c) Wmod^T [R, 6C] fp32."""

    def __init__(self, cubin="mod_fwd"):
        self.k = _load_swa_sm100_kernel(cubin, "swa_mod_fwd_sm100", 4 * 32768 + 64, pdl=PDL)

    def bind(self, c, wmod, out=None):
        R = c.shape[0]
        out = torch.empty(R, 6 * C, device=c.device) if out is None else out
        maps = (
            _map(c, [C, R], C * 2, [64, 128]),
            _map(wmod, [C, 6 * C], C * 2, [64, 128]),
            _map(out, [6 * C, R], 6 * C * 4, [32, 128], dtype="f32"),
        )

        def run():
            self.k(((R + 127) // 128, 6, 1), (128, 1, 1), *maps)

        run.keep = maps
        return run, out


class ModBwd:
    """Backward of ModFwd (mod_bwd.cu): g [R, 6C] fp32, c [R, C] bf16, Wmod [6C, C] bf16 -> dc [R, C] bf16, dWmod [6C, C] fp32."""

    def __init__(self, cubin="mod_bwd"):
        self.cubin = cubin
        self.kc = _load_swa_sm100_kernel(self.cubin, "swa_mod_bwd_dc_sm100", 4 * 49152 + 128, cluster=4, pdl=PDL_BWD)
        self.kw = {}  # per cluster size G
        self.trace = None  # (TRACE builds) int64 [16]

    def bind(self, g, c, wmod, dc=None, dw=None):
        R = c.shape[0]
        assert R % 128 == 0 and g.shape == (R, 6 * C) and g.is_contiguous() and c.is_contiguous()
        dev = c.device
        dc = torch.empty(R, C, device=dev, dtype=torch.bfloat16) if dc is None else dc
        dw = torch.empty(6 * C, C, device=dev, dtype=torch.float32) if dw is None else dw
        G = max(x for x in (1, 2, 4, 8, 16) if (R // 64) % x == 0)  # row groups (one cluster) of NS x 64 rows
        NS = R // 64 // G
        if G not in self.kw:
            self.kw[G] = _load_swa_sm100_kernel(self.cubin, "swa_mod_bwd_dw_sm100", 4 * 49152 + 128, cluster=G, pdl=PDL_BWD)
        kw = self.kw[G]
        mc_ = (
            _map(g, [6 * C, R], 6 * C * 4, [32, 128], dtype="f32"),
            _map(wmod, [C, 6 * C], C * 2, [64, 64]),
        )
        mw_ = (
            _map(g, [6 * C, R], 6 * C * 4, [32, 64], dtype="f32"),
            _map(c, [C, R], C * 2, [64, 64]),
        )
        run_dc = lambda: self.kc((4, R // 128, 1), (288, 1, 1), *mc_, c, dc, self.trace)
        run_dw = lambda: kw((G, 6, 1), (288, 1, 1), *mw_, dw, int(NS))

        def run():
            run_dc()
            run_dw()

        run.keep = (mc_, mw_)
        run.dc, run.dw = run_dc, run_dw
        return run, (dc, dw)


# --------------------------------------------------------------------------------------------------- the block
class DwBf16:
    """The five weight gradients in one launch (dw_bf16.cu) + a deterministic reduction of the split-K partial sums (dw_reduce_bf16):
    dP [M, 4C], x, datt, gated, y, dffn [M, C], dab [M, 2 NHID], hh [M, NHID] (bf16, row-major) -> dP^T x [4C, C], datt^T gated [C, C],
    dab^T y [2 NHID, C], dffn^T hh [C, NHID] in fp32. A CTA takes a row slice of one problem; the CTAs go to the problems in proportion to
    their bytes (44 / 28 / 44 / 32 of 148 on B200)."""

    FRAC = (44 / 148, 28 / 148, 44 / 148, 32 / 148)
    SHAPES = ((4 * C, C), (C, C), (2 * NHID, C), (C, NHID))

    def __init__(self):
        self.kd = _load_swa_sm100_kernel("dw_bf16", "swa_dw_bf16_sm100", 5 * 10 * 4096 + 256, pdl=PDL_BWD)
        self.kr = _load_swa_sm100_kernel("dw_bf16", "swa_dw_reduce_bf16_sm100", 0, pdl=PDL_BWD)

    def bind(self, dp, x, datt, gated, dab, y, dffn, hh):
        M = dp.shape[0]
        pairs = ((dp, x), (datt, gated), (dab, y), (dffn, hh))
        for a, b in pairs:
            assert a.is_contiguous() and b.is_contiguous() and a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
            assert a.shape[0] == M and b.shape[0] == M and M % 32 == 0
        cap = max(M // 512, min(M // DW_MIN_ROWS, DW_MAX_SPLIT))  # >= 512 rows per CTA as before, or (few rows) >= 128 and <= 16 splits
        ns = tuple(max(1, min(round(f * num_sms()), cap)) for f in self.FRAC)
        dev = dp.device
        maps = []
        for a, b in pairs:
            for t in (a, b):
                maps.append(_map(t, [t.shape[1], M], t.shape[1] * 2, [64, 32], swizzle=128, dtype="bf16"))
        ws = torch.empty(sum(n * m * k for n, (m, k) in zip(ns, self.SHAPES)), device=dev, dtype=torch.float32)
        outs = tuple(torch.empty(m, k, device=dev, dtype=torch.float32) for m, k in self.SHAPES)
        total4 = sum(m * k // 4 for m, k in self.SHAPES)

        def run():
            self.kd((sum(ns), 1, 1), (256, 1, 1), *maps, ws, int(M), *(int(n) for n in ns))
            self.kr(((total4 + 255) // 256, 1, 1), (256, 1, 1), ws, *outs, *(int(n) for n in ns))

        run.keep = (maps, ws, dp, x, datt, gated, dab, y, dffn, hh)
        return run, outs


class _Kernels:
    """Every kernel the block launches, loaded on one device."""

    def __init__(self) -> None:
        self.qkvg, self.qkvg2, self.qkvg2a = (
            QkvgFwd(),
            QkvgFwd2(),
            QkvgFwd2("qkvg_fwd2_a1", atm=32),
        )
        self.attn, self.attn1h = AttnFwd3(), AttnFwd1h()
        self.ffn, self.ffna = FfnFwd2(), FfnFwd2("ffn_fwd2_a1", atm=32)
        self.gate, self.dy, self.oproj = FfnBwdGate(), FfnBwdDy(), OprojBwd()
        self.dq, self.dkv, self.dkvq, self.qkvgb = (
            AttnDq(),
            AttnDkv(),
            AttnDkvq(),
            QkvgBwd(),
        )
        self.modf, self.modb = ModFwd(), ModBwd()
        self.dw = DwBf16()
        self.act = _load_swa_sm100_kernel("mod_bwd_act", "swa_mod_bwd_act_sm100", 0)
        self.dy_st, self.oproj_st, self.qkvgb_st = FfnBwdDy("ffn_bwd_dy_st"), OprojBwd("oproj_bwd_st"), QkvgBwd("qkvg_bwd_st")
        # round 3 part 8: weights read before griddepcontrol.wait (the forms of wpack_bf16, made at the forward's head)
        self.gate_wp, self.dy_stwp = FfnBwdGate("ffn_bwd_gate_wp"), FfnBwdDy("ffn_bwd_dy_stwp")
        self.oproj_stwp, self.qkvgb_stwp = OprojBwd("oproj_bwd_stwp"), QkvgBwd("qkvg_bwd_stwp")
        self.wpack = _load_swa_sm100_kernel("wpack_bf16", "swa_wpack_bf16_sm100", 0)
        self.save32k = _load_swa_sm100_kernel("wpack_bf16", "swa_save32_sm100", 0, pdl=PDL)
        self.save32lw = _load_swa_sm100_kernel("wpack_bf16", "swa_save32_lw_sm100", 0, pdl=True)
        self.wpackh = _load_swa_sm100_kernel("wpack_bf16", "swa_wpack_hybrid_sm100", 0)
        self.qkvg1t, self.ffn1t = QkvgFwd1t(), FfnFwd1t()
        self.qkvg1t_tr, self.modf_lt = QkvgFwd1t("qkvg_fwd1t_tr"), ModFwd("mod_fwd_lt")
        self.qkvg3, self.ffn3 = QkvgFwd3(), FfnFwd3()
        self.gattn = GattnFwd(split_cubin="gattn_fwd_s", csplit_cubin="gattn_fwd_c")
        self.gattn32 = GattnFwd("gattn_fwd32", split_cubin="gattn_fwd32_s", f32io=F32IO_MODE, csplit_cubin="gattn_fwd32_c")
        self.gattnb = GattnBwd(split_cubin="gattn_bwd_s", csb_cubin="gattn_bwd_c")


@functools.lru_cache(maxsize=16)
def kernels(index: int) -> _Kernels:
    """Build and load every kernel on device ``index`` (raises on a toolchain or driver failure)."""
    with torch.cuda.device(index):
        return _Kernels()


@functools.lru_cache(maxsize=16)
def embed_glue(index: int):
    """The InputFeatureEmbedder's glue kernels (embed_glue.cu) on device ``index``: ``.rope``, ``.feat``, ``.scatter``, ``.tokmean``
    (launched without programmatic dependent launch)."""
    import types

    with torch.cuda.device(index):
        return types.SimpleNamespace(
            **{k: _load_swa_sm100_kernel("embed_glue", f"swa_embed_{k}_sm100", 0) for k in ("rope", "rope2", "feat", "featbwd", "scatter", "tokmean", "outnorm", "outnormbwd", "tokgrad", "colsum", "dwf", "scatter8")}
        )


def supported(q: torch.Tensor, nhid: int, half_window: int, eps: float) -> bool:
    """The shapes and dtypes the kernels serve (``S % 128`` is checked by the block, which refuses instead of falling back)."""
    from foldjet_engine.kernels.swa_dit._common import is_global

    return (
        q.is_cuda
        and q.dtype == torch.bfloat16
        and q.dim() == 3
        and q.shape[-1] == C
        and nhid == NHID
        and (half_window == HW or is_global(half_window))
        and eps == EPS
        and torch.cuda.get_device_capability(q.device) == (10, 0)
    )


def _has_fa4() -> bool:
    """FlashAttention-4 (``flash_attn.cute``) installed -- looked up without importing it."""
    try:
        spec = importlib.util.find_spec("flash_attn")
    except (ImportError, ValueError):
        return False
    return bool(
        spec is not None
        and spec.submodule_search_locations
        and any((Path(r) / "cute" / "__init__.py").exists() for r in spec.submodule_search_locations)
    )


#: Global attention runs on FlashAttention-4; resolved once at import so ``interface.refusal`` can read it while tracing.
FA4_AVAILABLE = _has_fa4()


def need_s128(S: int) -> None:
    if S % 128:
        raise ValueError(
            f"SWA atom DiT block (sm_100a): the atom count S = {S} must be a multiple of 128; pad the atoms (seqused masks them)"
        )


def _cached(fn, *ts):
    """fn(*ts), cached while the very same tensor objects are alive and unmodified (``kernels._capture.param_forms``: the weight forms
    are rebuilt when a weight is updated in place -- its version moves -- or replaced; a new tensor at a freed one's address is not
    served the old one's form). Scoped to the CUDA-graph capture: an entry made outside a capture is not used inside one, and one made
    during a capture serves that capture only. Not cached for inference tensors (they carry no version counter)."""
    return _capture.param_forms(("swa_dit_sm100", fn.__name__), ts, lambda: fn(*ts))


def _w_qkvg(wqkv, wg):
    return torch.cat([wqkv, wg]).contiguous()


def _w_qkvg_t(wqkv, wg):
    return torch.cat([wqkv, wg]).t().contiguous()


def _t(w):
    return w.t().contiguous()


@functools.lru_cache(maxsize=16)
def _side_streams(index: int) -> tuple:
    with torch.cuda.device(index):
        return torch.cuda.Stream(), torch.cuda.Stream()


def _fork_join(side, main) -> None:
    """Run the launches ``side`` (at most two) each on a side stream and ``main`` on the current one, all after the work queued so far, and
    make the current stream wait for them (CUDA-graph capturable: the capture forks and joins). Independent few-CTA kernels of the small
    row counts (one sample of a thousand atoms: 8-16 CTAs each) then share the GPU instead of running back to back."""
    cur = torch.cuda.current_stream()
    streams = _side_streams(cur.device.index)
    fork = torch.cuda.Event()
    fork.record(cur)
    joins = []
    for fn, s in zip(side, streams, strict=False):
        s.wait_event(fork)
        with torch.cuda.stream(s):
            fn()
        ev = torch.cuda.Event()
        ev.record(s)
        joins.append(ev)
    main()
    for ev in joins:
        cur.wait_event(ev)


def master_pack(weights):
    """B200 bf16 training with fp32 master weights: the bf16 forms of (wqkv, wg, wo, wu, wd) as views of ONE flat buffer (a cat and a cast:
    two kernels per step where the per-weight casts and the [Wqkv; Wg] cat were six), wqkv and wg adjacent so that [Wqkv; Wg] is a view
    (``_adjacent``). The values are the plain casts'."""
    flat = torch.empty(sum(w.numel() for w in weights), device=weights[0].device, dtype=torch.bfloat16)
    torch.cat([w.detach().reshape(-1) for w in weights], out=flat)  # the cast inside the cat: one launch
    out, off = [], 0
    for w in weights:
        out.append(flat[off : off + w.numel()].view(w.shape))
        off += w.numel()
    return out


#: element offsets of wpack_bf16.cu's buffer: Wmod, Wqkv, Wg, Wo, Wu, Wd, Wd^T, Wo^T, Wu^T, [Wqkv; Wg]^T, end
_WPACK = (0, 98304, 147456, 163840, 180224, 245760, 278528, 311296, 327680, 393216, 458752)


def pack_ok(weights) -> bool:
    """:func:`master_pack_train` serves these (wmod, wqkv, wg, wo, wu, wd): fp32 contiguous of the block's shapes."""
    shapes = ((6 * C, C), (3 * C, C), (C, C), (C, C), (2 * NHID, C), (C, NHID))
    return all(w.dtype == torch.float32 and w.is_contiguous() and tuple(w.shape) == s for w, s in zip(weights, shapes, strict=True))


def master_pack_train(weights):
    """B200 master training (round 3 part 8): (wmod, wqkv, wg, wo, wu, wd) fp32 -> ([Wmod, Wqkv, Wg, Wo, Wu, Wd] bf16, (WDT, WUT, WOT, WT)
    bf16) -- every form the forward and the backward kernels read, views of one buffer made by ONE launch (wpack_bf16.cu; the values of
    plain casts, as :func:`master_pack` and :func:`_w_bwd`). Made at the head of the forward, so the backward's ``_wpre`` kernels may read
    them before their griddepcontrol.wait."""
    dev = weights[0].device
    K = kernels(dev.index if dev.index is not None else torch.cuda.current_device())
    flat = torch.empty(_WPACK[-1], device=dev, dtype=torch.bfloat16)
    K.wpack(((_WPACK[-1] // 4 + 255) // 256, 1, 1), (256, 1, 1), *weights, flat)
    o = _WPACK
    fwd = [flat[o[i] : o[i + 1]].view(weights[i].shape) for i in range(6)]
    wdt = flat[o[6] : o[7]].view(NHID, C)
    wot = flat[o[7] : o[8]].view(C, C)
    wut = flat[o[8] : o[9]].view(C, 2 * NHID)
    wt = flat[o[9] : o[10]].view(C, 4 * C)
    return fwd, (wdt, wut, wot, wt)


def master_pack_hybrid(weights, stream=None):
    """B200 master training with the bf16 forward and the TF32 backward (round 3 part 8): (wmod, wqkv, wg, wo, wu, wd) fp32 -> ([Wmod, Wqkv,
    Wg, Wo, Wu, Wd] bf16, (W, WO, WAB, WD) TF32-rounded fp32 -- ``tf32_fwd._tf32_block_forms``'s --, (WdT, WoT) TF32 -- ``tf32_bwd._bwd_forms``'s)
    in ONE launch (wpack_bf16.cu swa_wpack_hybrid_sm100; was wprep_tf32 + wprep_bwd_tf32 + swa_wpack_bf16). The same values as those."""
    dev = weights[0].device
    K = kernels(dev.index if dev.index is not None else torch.cuda.current_device())
    n16, n32 = _WPACK[6], 229376
    f16 = torch.empty(n16, device=dev, dtype=torch.bfloat16)
    f32 = torch.empty(n32, device=dev, dtype=torch.float32)
    if stream is None:
        K.wpackh((((n16 + n32) // 4 + 255) // 256, 1, 1), (256, 1, 1), *weights, f16, f32)
    else:  # round 3 part 14: allocated on the current stream, made on ``stream`` (the caller orders the streams: prefetch_hybrid)
        with torch.cuda.stream(stream):
            K.wpackh((((n16 + n32) // 4 + 255) // 256, 1, 1), (256, 1, 1), *weights, f16, f32)
    o = _WPACK
    fwd = [f16[o[i] : o[i + 1]].view(weights[i].shape) for i in range(6)]
    t = (0, 65536, 81920, 147456, 180224, 212992, 229376)
    tf = (f32[t[0] : t[1]].view(4 * C, C), f32[t[1] : t[2]].view(C, C), f32[t[2] : t[3]].view(2 * NHID, C), f32[t[3] : t[4]].view(C, NHID))
    bw = (f32[t[4] : t[5]].view(NHID, C), f32[t[5] : t[6]].view(C, C))
    return fwd, tf, bw


class _Ptr12(__import__("ctypes").Structure):
    _fields_ = [("s", __import__("ctypes").c_uint64 * 12), ("d", __import__("ctypes").c_uint64 * 12)]


def save32(saves):
    """twelve bf16 tensors of one element count -> their fp32 copies, in one launch (wpack_bf16.cu swa_save32_sm100; round 3 part 8: the
    bf16 forward's saves for the TF32 backward). Exact."""
    n = saves[0].numel()
    assert len(saves) == 12 and n % 8 == 0 and all(t.numel() == n and t.dtype == torch.bfloat16 and t.is_contiguous() for t in saves)
    dev = saves[0].device
    K = kernels(dev.index if dev.index is not None else torch.cuda.current_device())
    out = [torch.empty(t.shape, device=dev, dtype=torch.float32) for t in saves]
    p = _Ptr12()
    for i in range(12):
        p.s[i], p.d[i] = saves[i].data_ptr(), out[i].data_ptr()
    n8 = n // 8
    K.save32k(((12 * n8 + 255) // 256, 1, 1), (256, 1, 1), p, int(n8), 12)
    return out


#: r3-swa part 19: the hybrid route widens its saves in the backward, under the FFN gate kernel (:func:`save32_late`), not at the end of
#: the forward (development switch: False = :func:`save32` in the forward, as parts 8 - 18)
SAVE_LATE = __import__("os").environ.get("SWA_SAVE_LATE", "1") == "1"
#: ... up to this many rows (above: the widening in the forward)
SAVE_LATE_MAX_ROWS = int(__import__("os").environ.get("SWA_SAVE_LATE_MAX_ROWS", "8192"))
#: ... its grid: at most this many 256-thread blocks a SM (grid-stride), so the whole grid is resident beside the gate kernel
SAVE_LATE_BPS = int(__import__("os").environ.get("SWA_SAVE_LATE_BPS", "3"))
#: ... where the gate kernel leaves SMs to it: at most this many gate CTAs, or at most SAVE_LATE_SMALL rows (measured, ESMFold2 master
#: training, us before -> after: L 1024 85.8 -> 83.9, 3072 112.6 -> 108.6, 6144 163.8 -> 161.9; L 2048 (128 gate CTAs) 96.2 -> 94.4 / 98.2 in two runs, L 4096 (128 gate CTAs
#: and 38 MB to widen, 132.9 -> 137.0)
SAVE_LATE_GATE_CTAS = 96
SAVE_LATE_SMALL = 1024


def save_late_ok(rows: int) -> bool:
    """:func:`save32_late` for a one-augment hybrid block of ``rows`` rows (the gate kernel's grid as ``FfnBwdGateTf32.bind`` makes it)."""
    if not SAVE_LATE or rows > SAVE_LATE_MAX_ROWS:
        return False
    ntile = (rows + 127) // 128
    ng = 1
    while ng < 8 and ntile * ng * 2 <= num_sms():
        ng *= 2
    return rows <= SAVE_LATE_SMALL or min(num_sms(), ntile * ng) <= SAVE_LATE_GATE_CTAS


def save32_late(saves):
    """:func:`save32` as the launch right after the FFN gate kernel of the backward (``swa_save32_lw_sm100``: triggers its dependents at
    once, widens, waits for the gate kernel at its end; the gate kernel must run with its trigger after its wait -- ``FfnBwdGateTf32``
    ``late_trigger``). The same values (exact)."""
    n = saves[0].numel()
    assert len(saves) == 12 and n % 8 == 0 and all(t.numel() == n and t.dtype == torch.bfloat16 and t.is_contiguous() for t in saves)
    dev = saves[0].device
    K = kernels(dev.index if dev.index is not None else torch.cuda.current_device())
    out = [torch.empty(t.shape, device=dev, dtype=torch.float32) for t in saves]
    p = _Ptr12()
    for i in range(12):
        p.s[i], p.d[i] = saves[i].data_ptr(), out[i].data_ptr()
    n8 = n // 8
    K.save32lw((min((12 * n8 + 255) // 256, SAVE_LATE_BPS * num_sms()), 1, 1), (256, 1, 1), p, int(n8), 12)
    return out


_tls = __import__("threading").local()


def _ctx_current(index: int) -> None:
    """Make device ``index``'s primary context current in this thread (once per thread). The autograd engine runs a backward on a
    worker thread where no CUDA call may have bound a context yet; a backward that encodes TMA maps before any torch kernel launch
    (``block_bwd`` with the forms of :func:`master_pack_train`) needs it bound (CUDA_ERROR_INVALID_CONTEXT otherwise)."""
    done = getattr(_tls, "ctx", None)
    if done is None:
        done = _tls.ctx = set()
    if index in done:
        return
    from cuda.bindings import driver as cu

    from foldjet_engine.kernels.augmented_attention.cuda.sm100 import driver

    driver.init()
    err, ctx = cu.cuDevicePrimaryCtxRetain(index)
    if err == cu.CUresult.CUDA_SUCCESS:
        cu.cuCtxSetCurrent(ctx)
    done.add(index)


def _adjacent(wqkv, wg):
    """[Wqkv; Wg] as a view when ``wg`` directly follows ``wqkv`` in one buffer (``master_pack``), else None."""
    if (
        wqkv.is_contiguous()
        and wg.is_contiguous()
        and wqkv.dtype == wg.dtype
        and wqkv.shape[1] == wg.shape[1]
        and wqkv.untyped_storage().data_ptr() == wg.untyped_storage().data_ptr()
        and wg.storage_offset() == wqkv.storage_offset() + wqkv.numel()
    ):
        return torch.as_strided(wqkv, (wqkv.shape[0] + wg.shape[0], wqkv.shape[1]), (wqkv.shape[1], 1), wqkv.storage_offset())
    return None


def _w_fwd(wqkv, wg):
    """[Wqkv; Wg] for the forward kernels: a view of a ``master_pack``, else the cached cat."""
    w = _adjacent(wqkv, wg)
    return w if w is not None else _cached(_w_qkvg, wqkv, wg)


def _w_bwd(wqkv, wg, wo, wu, wd):
    """(WDT, WUT, WOT, WT) for the backward kernels. From a ``master_pack`` (fresh views every step): two cat launches ([Wd^T; Wo^T] and
    [Wu^T; [Wqkv; Wg]^T], each form a row block, a view), else the cached per-weight transposes."""
    w = _adjacent(wqkv, wg)
    if w is None or wu.shape[0] != w.shape[0] or wd.shape[0] != wo.shape[0]:
        return _cached(_t, wd), _cached(_t, wu), _cached(_t, wo), _cached(_w_qkvg_t, wqkv, wg)
    a = torch.cat([wd.t(), wo.t()])
    b = torch.cat([wu.t(), w.t()])
    nd, nu = wd.shape[1], wu.shape[1]
    return a[:nd], b[:nu], a[nd:], b[nu:]


def _fa2_fallback(t: torch.Tensor) -> bool:
    """A B200 tensor (FlashAttention-4 not installed): global attention on PyTorch's bundled FlashAttention-2. Other cards never reach the
    global-attention launchers without FA4 (their blocks refuse first), so nothing changes there."""
    return t.is_cuda and torch.cuda.get_device_capability(t.device) == (10, 0)


def _fa2_varlen(seqused: torch.Tensor, s: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(cu_seqlens [2N + 1], seqused_k [2N]) int32: sample n is the sequences [n s, n s + u_n) and [n s + u_n, (n + 1) s), the second
    without keys (its rows come out zero), as the A100 block's ``composed._varlen`` (torch ops here: no sm_80 extension on this path)."""
    u = seqused.to(torch.int32)
    lens = torch.stack([u, s - u], 1).reshape(-1)
    cu = torch.nn.functional.pad(torch.cumsum(lens, 0, dtype=torch.int32), (1, 0))
    used_k = torch.stack([u, torch.zeros_like(u)], 1).reshape(-1)
    return cu, used_k


def _fa2_global_fwd(Qh, Kh, Vh, seqused, O, lse):
    """:func:`attn_global_fwd` on FlashAttention-2 (``aten._flash_attention_forward``, varlen): O [N S, C] bf16 (rows past ``seqused``
    zero) and its lse (FA2's layout, numel N H S) into ``lse``, read back only by :func:`_fa2_global_bwd`."""
    N, _, S, _ = Qh.shape
    cu, used_k = _fa2_varlen(seqused, S)
    q, k, v = (t.transpose(1, 2).reshape(N * S, H, D) for t in (Qh, Kh, Vh))
    o, l2, _rng, _unused, _dbg = torch.ops.aten._flash_attention_forward(
        q, k, v, cu, cu, S, S, 0.0, False, False, scale=D**-0.5, seqused_k=used_k
    )
    O.copy_(o.reshape(N * S, C))
    lse.view(-1).copy_(l2.reshape(-1))


def _fa2_global_bwd(Qh, Kh, Vh, O, dO, lse, seqused, dQh, dKh, dVh):
    """:func:`attn_global_bwd` on FlashAttention-2 (``aten._flash_attention_backward``), the lse of :func:`_fa2_global_fwd`."""
    N, _, S, _ = Qh.shape
    cu, _used_k = _fa2_varlen(seqused, S)
    q, k, v = (t.transpose(1, 2).reshape(N * S, H, D) for t in (Qh, Kh, Vh))
    empty = torch.empty(0, device=Qh.device)
    dq, dk, dv = torch.ops.aten._flash_attention_backward(
        dO.reshape(N * S, H, D), q, k, v, O.reshape(N * S, H, D), lse.view(H, N * S), cu, cu, S, S, 0.0, False, empty, empty, scale=D**-0.5
    )
    for dst, src in ((dQh, dq), (dKh, dk), (dVh, dv)):
        dst.copy_(src.view(N, S, H, D).transpose(1, 2))


NATIVE_GLOBAL = True  # B200: the native global attention forward (gattn_fwd.cu) in place of FlashAttention-4's (development switch)


def _native_global(Qh, native: bool = True) -> bool:
    """The B200 global attention forward on gattn_fwd.cu (``GattnFwd``) instead of FlashAttention-4's: on a B200 with FA4 installed, S a multiple of 128,
    inference (``native`` false from the training forwards unless ``NATIVE_TRAIN_FWD``: on the native O the S48 master training gradients
    measured YELLOW 1.10 - 1.13 x tol at L 1024 / 4096 against GREEN 0.96 / 0.95 on FA4's, with FA4's backward or the native one; the
    training forward stays FA4's, the training backward is the native gattn_bwd.cu, :func:`native_global_train`). The kernel keeps no
    running maximum: it relies on the heads' qk RMSNorm without a weight (|q . k| scale <= sqrt(32), exp2 of at most 8.2), which every
    caller's qkvg stage applies. Measured 2026-11-17 (``_tools/gb.py``, graph timing, L2 flushed; FA4 + the zeroed O it needs): N 1 S 1024
    20.4 against 22.5 us, N 1 S 8192 102.4 / 104.5, N 5 S 4096 149 / 160, N 48 S 1024 82 / 111, N 48 S 2048 262 / 319, N 48 S 8192
    3751 / 3955 -- never slower."""
    return native and NATIVE_GLOBAL and FA4_AVAILABLE and _fa2_fallback(Qh) and Qh.shape[2] % 128 == 0


NATIVE_GLOBAL_BWD = True  # B200: global attention training backward on gattn_bwd.cu (development switch)
#: B200 training forward: FlashAttention-4's (False) or the native gattn_fwd.cu (True; round 3 part 3, 2026-11-17). The S48 master
#: gate_proj.weight gradient measures YELLOW 1.13 / 1.09 x tol at L 1024 / 4096 on it (FA4's forward GREEN 0.96 / 0.95): seed noise, not a
#: defect -- FA4 keeps its first block's row maximum (rescale threshold 8), so its P is as rounded as the native one; at seeds 1 / 2 the
#: native forward's worst tensor is GREEN 0.93 / 0.94 where FA4's is 0.94 / YELLOW 1.02 (L 1024) (_tools/accdiag.py). Inside the engine standard (YELLOW <= 5 tol);
#: fp16 P (-DP16, V converted in shared memory) moved L 1024 only to 1.07 and costs 2 % of the kernel: kept as a build option, not routed.
NATIVE_TRAIN_FWD = True


def native_global_train(Qh) -> bool:
    """B200 global attention TRAINING with the native deterministic backward (``GattnBwd``, gattn_bwd.cu; D from the out-projection backward)
    in place of FlashAttention-4's deterministic backward, on FlashAttention-4's forward (its natural-log LSE converted to the negated log2
    the kernel adds) or, with ``NATIVE_TRAIN_FWD``, on the native forward with that LSE written directly (``GattnFwd`` ``log2``). S a multiple of 256 (the backward's key pairs; the requirement lengths
    are multiples of 1024). The forward and the backward take the same predicate (tensor metadata and module flags only)."""
    return NATIVE_GLOBAL_BWD and _native_global(Qh) and Qh.shape[2] % 256 == 0 and num_sms() >= Qh.shape[2] // 256


def attn_global_bwd_native(Qh, Kh, Vh, dO, lse2, Dv, seqused, dQh=None, dKh=None, dVh=None):
    """The native global attention backward (gattn_bwd.cu) on the saves of a :func:`native_global_train` forward: -> (dQh, dKh, dVh)
    head-major bf16, every row written."""
    if not NATIVE_TRAIN_FWD:  # FA4's forward saved its natural-log LSE: P = 2^(s scale log2 e - lse log2 e)
        lse2 = lse2 * -1.4426950408889634
    with torch.cuda.device(Qh.device):
        K = kernels(Qh.device.index if Qh.device.index is not None else torch.cuda.current_device())
        run, outs = K.gattnb.bind(Qh, Kh, Vh, dO, lse2, Dv, seqused, dQh, dKh, dVh)
        run()
    return outs


def global_out(Qh, native: bool = True) -> torch.Tensor:
    """O [N S, C] bf16 for :func:`attn_global_fwd`: uninitialized where the native forward serves (it writes every row, zeros past
    seqused), else zeroed (FlashAttention-4 / -2 skip the rows past seqused)."""
    N, _, S, _ = Qh.shape
    alloc = torch.empty if _native_global(Qh, native) else torch.zeros
    return alloc(N * S, C, device=Qh.device, dtype=torch.bfloat16)


def attn_global_fwd(Qh, Kh, Vh, seqused, O, lse, native: bool = True, log2: bool = False):
    """Global attention forward (on a B200 with FA4 installed: the native gattn_fwd.cu, :func:`_native_global`; else FlashAttention-4): Qh, Kh, Vh head-major [N, H, S, D] bf16 (read through their [N, S, H, D]
    transposed views), seqused [N] int32 -> O [N S, C] bf16 and lse [N, H, S] fp32, written in place. Rows at or past
    ``seqused`` are skipped by FA4 -- ``O`` must arrive zeroed there (and their lse stays unwritten); keys past ``seqused``
    are never attended. FA4 keeps its own lse convention, which only :func:`attn_global_bwd` reads. On a B200 without FA4 installed: PyTorch's bundled
    FlashAttention-2 (:func:`_fa2_global_fwd`, as the A100 block takes it)."""
    if _native_global(Qh, native):  # B200: gattn_fwd.cu (every row of O and lse written: zeros past seqused)
        with torch.cuda.device(Qh.device):
            K = kernels(Qh.device.index if Qh.device.index is not None else torch.cuda.current_device())
            run, _, _ = K.gattn.bind(Qh, Kh, Vh, seqused, O=O, LSE=lse, log2=log2)  # log2: the training save of the native backward
            run()
        return
    if not FA4_AVAILABLE and _fa2_fallback(Qh):
        _fa2_global_fwd(Qh, Kh, Vh, seqused, O, lse)
        return
    from flash_attn.cute.interface import (
        _flash_attn_fwd,
    )  # ty: ignore[unresolved-import]  # optional FlashAttention backend

    N, _, S, _ = Qh.shape
    _flash_attn_fwd(
        Qh.transpose(1, 2),
        Kh.transpose(1, 2),
        Vh.transpose(1, 2),
        seqused_q=seqused,
        seqused_k=seqused,
        softmax_scale=D**-0.5,
        causal=False,
        return_lse=True,
        out=O.view(N, S, H, D),
        lse=lse,
    )


def attn_global_bwd(Qh, Kh, Vh, O, dO, lse, seqused, dQh, dKh, dVh):
    """Global attention backward on FlashAttention-4 (its own softmax-delta pass included): dQh, dKh, dVh head-major
    [N, H, S, D] bf16 written in place -- zero them first, FA4 leaves the rows at or past ``seqused`` alone. On a B200 without FA4: PyTorch's
    bundled FlashAttention-2 (:func:`_fa2_global_bwd`)."""
    if not FA4_AVAILABLE and _fa2_fallback(Qh):
        _fa2_global_bwd(Qh, Kh, Vh, O, dO, lse, seqused, dQh, dKh, dVh)
        return
    from flash_attn.cute.interface import (
        _flash_attn_bwd,
    )  # ty: ignore[unresolved-import]  # optional FlashAttention backend

    N, _, S, _ = Qh.shape
    _flash_attn_bwd(
        Qh.transpose(1, 2),
        Kh.transpose(1, 2),
        Vh.transpose(1, 2),
        O.view(N, S, H, D),
        dO.view(N, S, H, D),
        lse,
        softmax_scale=D**-0.5,
        causal=False,
        seqused_q=seqused,
        seqused_k=seqused,
        dq=dQh.transpose(1, 2),
        dk=dKh.transpose(1, 2),
        dv=dVh.transpose(1, 2),
        # B200 (2026-11-17): FA4's deterministic backward (fixed-order dQ accumulation), run to run the same bits; other cards as before
        **({"deterministic": True} if _fa2_fallback(Qh) else {}),
    )


def block_fwd(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, save, half_window=HW):
    """The bf16 forward: [out] or, with ``save``, [out, Qh, Kh, Vh, G, O, lse, q1, X, PQ, PK, Att, Y, FF] -- the layout of
    ``dispatch.swa_dit_block_fwd``. ``half_window`` is the window (64) or a global-attention value (``is_global``)."""
    from foldjet_engine.kernels.swa_dit._common import is_global

    N, S, _ = q.shape
    need_s128(S)
    A = N // B
    with torch.cuda.device(q.device):
        K = kernels(q.device.index if q.device.index is not None else torch.cuda.current_device())
        W = _w_fwd(wqkv, wg)
        qk = (K.qkvg2a if A < 4 else K.qkvg2) if A < 5 or A * B * S <= 2 * 148 * 32 else K.qkvg
        run, (Qh, Kh, Vh, G, X, PQ, PK) = qk.bind(q, mod, cos, sin, None, None, A, B, save=save, W=W)
        run()
        lse = torch.empty(N, H, S, device=q.device, dtype=torch.float32)
        if is_global(half_window):
            # training (save): FA4's forward (NATIVE_TRAIN_FWD: the native one where native_global_train serves), whose O / lse the backward
            # (native gattn_bwd.cu or FA4's) reads
            nat = not save or (NATIVE_TRAIN_FWD and native_global_train(Qh))
            O = global_out(Qh, nat)
            attn_global_fwd(Qh, Kh, Vh, seqused, O, lse, native=nat, log2=save and nat)
        else:
            O = torch.empty(N * S, C, device=q.device, dtype=torch.bfloat16)
            at = K.attn1h if Qh.shape[0] * (S // 128) * 4 <= num_sms() else K.attn
            run, _, _ = at.bind(Qh, Kh, Vh, seqused, O=O, LSE=lse)
            run()
        run, (out, q1, att, y, ffn) = (K.ffn if A >= 4 else K.ffna).bind(q.reshape(N * S, C), G, O, mod, wo, wu, wd, A, B, save=save)
        run()
    if not save:
        return [out.view(N, S, C)]
    return [out.view(N, S, C), Qh, Kh, Vh, G, O, lse, q1, X, PQ, PK, att, y, ffn]


def train_c_ok(q, B, half_window) -> bool:
    """:func:`block_fwd_train_c` serves this training forward: one augment (N = B), window attention, the few-tile qkvg stage (one 32-row tile
    per CTA) and below mod_fwd's cuBLAS row count (the modulation must be a PDL kernel to run beside the qkvg stage)."""
    N, S, _ = q.shape
    return TRAIN_C and PDL and N == B and half_window == HW and few_tiles(N, S) and N * S < MOD_FWD_CUBLAS_ROWS


#: round 3 part 8 (development switch): :func:`block_fwd_train_c`
TRAIN_C = True


def block_fwd_train_c(q, c, wmod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B):
    """The bf16 training forward of one augment with the qkvg stage's modulation inside (round 3 part 8; :func:`train_c_ok`): the
    modulation kernel (mod_fwd LATE_TRIGGER: it lets the next kernel start after its own wait) and the qkvg stage (qkvg_fwd1t SAVE /
    LATE_WAIT: shift_a / scale_a from c itself, no wait before its loads, one at its stores) run side by side, so the chain is qkvg ->
    attention -> FFN (was modulation -> qkvg -> attention -> FFN); the FFN and the backward read the modulation kernel's [B S, 6C].
    Returns (mod, [out, Qh, Kh, Vh, G, O, lse, q1, X, PQ, PK, Att, Y, FF]) -- :func:`block_fwd`'s values (the same math and rounding)."""
    N, S, _ = q.shape
    need_s128(S)
    with torch.cuda.device(q.device):
        K = kernels(q.device.index if q.device.index is not None else torch.cuda.current_device())
        run_m, mod = K.modf_lt.bind(c, wmod)
        run_m()
        run, (Qh, Kh, Vh, G, X, PQ, PK) = K.qkvg1t_tr.bind(q, c, wmod, cos, sin, _w_fwd(wqkv, wg), save=True, busy=6 * ((N * S + 127) // 128))
        run()
        lse = torch.empty(N, H, S, device=q.device, dtype=torch.float32)
        O = torch.empty(N * S, C, device=q.device, dtype=torch.bfloat16)
        at = K.attn1h if N * (S // 128) * 4 <= num_sms() else K.attn
        run, _, _ = at.bind(Qh, Kh, Vh, seqused, O=O, LSE=lse)
        run()
        run, (out, q1, att, y, ffn) = K.ffna.bind(q.reshape(N * S, C), G, O, mod, wo, wu, wd, 1, B, save=True)
        run()
    return mod, [out.view(N, S, C), Qh, Kh, Vh, G, O, lse, q1, X, PQ, PK, att, y, ffn]


def block_fwd_c(q, c, wmod, cos, sin, seqused, wqkv, wg, wo, wu, wd, half_window=HW):
    """The bf16 inference block of few tiles (``few_tiles``) from the conditioning itself: q [N, S, C], c [N S, C] bf16 (one row per
    sequence row), Wmod [6C, C] bf16 -> out [N, S, C]. The modulation ``silu(c) Wmod^T`` is made inside the two row kernels
    (qkvg_fwd1t / ffn_fwd1t: no modulation launch, no fp32 [N S, 6C] in HBM); the attention stage is ``block_fwd``'s."""
    from foldjet_engine.kernels.swa_dit._common import is_global

    N, S, _ = q.shape
    need_s128(S)
    with torch.cuda.device(q.device):
        K = kernels(q.device.index if q.device.index is not None else torch.cuda.current_device())
        few = few_tiles(N, S) and not FORCE_FWD3
        run, (Qh, Kh, Vh, G) = (K.qkvg1t if few else K.qkvg3).bind(q, c, wmod, cos, sin, _w_fwd(wqkv, wg))
        run()
        lse = torch.empty(N, H, S, device=q.device, dtype=torch.float32)
        if is_global(half_window):
            O = global_out(Qh)
            attn_global_fwd(Qh, Kh, Vh, seqused, O, lse)
        else:
            O = torch.empty(N * S, C, device=q.device, dtype=torch.bfloat16)
            at = K.attn1h if N * (S // 128) * 4 <= num_sms() else K.attn
            run, _, _ = at.bind(Qh, Kh, Vh, seqused, O=O, LSE=lse)
            run()
        run, out = (K.ffn1t if few else K.ffn3).bind(q.reshape(N * S, C), G, O, c, wmod, wo, wu, wd, N)
        run()
    return out.view(N, S, C)


def block_bwd(
    dy,
    q,
    mod,
    cos,
    sin,
    seqused,
    wqkv,
    wg,
    wo,
    wu,
    wd,
    qh,
    kh,
    vh,
    g,
    o,
    lse,
    q1,
    x,
    pq,
    pk,
    att,
    y,
    ffn,
    B,
    half_window=HW,
    modb=None,
    wb=None,
):
    """The bf16 backward: [dq, dmod, dWqkv, dWg, dWo, dWu, dWd] -- the layout of ``dispatch.swa_dit_block_bwd``. ``wb``: the backward's
    weight forms (WDT, WUT, WOT, WT) from :func:`master_pack_train` (made at the forward's head): then the kernels that read them load
    them before their griddepcontrol.wait (the ``_wp`` builds; round 3 part 8)."""
    from foldjet_engine.kernels.swa_dit._common import is_global

    N, S, _ = q.shape
    need_s128(S)
    A = N // B
    M = N * S
    dev = q.device
    with torch.cuda.device(dev):
        K = kernels(dev.index if dev.index is not None else torch.cuda.current_device())
        # A <= 8: every dmod element has one writer (the _st builds store it; B200 2026-11-17: the zero fill was 80 us at 196608 rows)
        st = A <= 8
        kdy, kop, kqb = (K.dy_st, K.oproj_st, K.qkvgb_st) if st else (K.dy, K.oproj, K.qkvgb)
        if wb is not None:  # no torch launch precedes the first TMA map here (the forms come from the forward)
            _ctx_current(dev.index if dev.index is not None else torch.cuda.current_device())
        kgate = K.gate
        if wb is not None and st and WPRE:
            kgate, kdy, kop, kqb = K.gate_wp, K.dy_stwp, K.oproj_stwp, K.qkvgb_stwp
        dmod = (torch.empty if st else torch.zeros)(B * S, 6 * C, device=dev, dtype=torch.float32)
        dy2 = dy.reshape(M, C)
        WDT, WUT, WOT, WT = _w_bwd(wqkv, wg, wo, wu, wd) if wb is None else wb[:4]
        run, (dffn, hh, dab) = kgate.bind(dy2, y, mod, wu, None, A, B, WDT=WDT)
        run()
        run, (dq1, _) = kdy.bind(dab, dy2, q1, ffn, mod, wu, A, B, dmod=dmod, WUT=WUT)
        run()
        dO, dG, datt, gated = (torch.empty(M, C, device=dev, dtype=torch.bfloat16) for _ in range(4))
        Dv = torch.empty(N, H, S, device=dev, dtype=torch.float32)
        run, _ = kop.bind(
            dq1,
            o,
            g,
            mod,
            wo,
            att,
            A,
            B,
            dmod=dmod,
            WOT=WOT,
            outs=(dO, dG, Dv, datt, gated),
        )
        run()
        if is_global(half_window) and native_global_train(qh):  # B200: gattn_bwd.cu on the native forward's saves (every row written)
            dQh, dKh, dVh = attn_global_bwd_native(qh, kh, vh, dO, lse, Dv, seqused)
        elif is_global(half_window):  # FA4: dQ / dK / dV in one call; skipped (padding) rows stay zero
            dQh, dKh, dVh = (torch.zeros(N, H, S, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
            attn_global_bwd(qh, kh, vh, o, dO, lse, seqused, dQh, dKh, dVh)
        else:
            dQh = torch.empty(N, H, S, D, device=dev, dtype=torch.bfloat16)
            dKh, dVh = torch.empty_like(dQh), torch.empty_like(dQh)
            if N * (S // 128) * 2 >= 2 * num_sms():
                run, _ = K.dkvq.bind(qh, kh, vh, dO, lse, Dv, seqused, dKh=dKh, dVh=dVh, dQh=dQh)
                run()
            else:  # dQ on a side stream beside dK / dV (both few-CTA launches here: they overlap)
                run_q, _ = K.dq.bind(qh, kh, vh, dO, lse, Dv, seqused, dQh=dQh)
                run_kv, _ = K.dkv.bind(qh, kh, vh, dO, lse, Dv, seqused, dKh=dKh, dVh=dVh)
                _fork_join([run_q], run_kv)
        dq = torch.empty(M, C, device=dev, dtype=torch.bfloat16)
        dP = torch.empty(M, 4 * C, device=dev, dtype=torch.bfloat16)
        run, _ = kqb.bind(
            q.reshape(M, C),
            dq1,
            pq,
            pk,
            dG,
            dQh,
            dKh,
            dVh,
            mod,
            cos,
            sin,
            wqkv,
            wg,
            A,
            B,
            dmod=dmod,
            WT=WT,
            dq=dq,
            dP=dP,
        )
        run()
        # the five weight gradients (fp32): one split-K launch (dWqkv | dWg is one problem over dP, x read once) + the fixed-order
        # reduction of its partial sums (dw_bf16.cu; was four cuBLAS GEMMs)
        run, (dwqkvg, dwo, dwu, dwd) = K.dw.bind(dP, x.view(M, C), datt, gated, dab, y.view(M, C), dffn, hh)
        if modb is None:
            run()
        else:  # the modulation backward (c, Wmod bf16) from dmod: its dc and dW kernels on side streams beside the weight gradients
            c_mod, w_mod = modb
            if B * S >= MOD_BWD_CUBLAS_ROWS:  # many rows: the cuBLAS form (mod_bwd_cublas), its outputs made on this stream
                dc_mod = torch.empty(B * S, C, device=dev, dtype=torch.bfloat16)
                dw_mod = torch.empty(6 * C, C, device=dev, dtype=torch.float32)
                _fork_join([lambda: mod_bwd_cublas(dmod, c_mod, w_mod, dc_mod, dw_mod)], run)
            else:
                run_m, (dc_mod, dw_mod) = K.modb.bind(dmod, c_mod, w_mod)
                _fork_join([run_m.dc, run_m.dw], run)
    if modb is not None:
        return [dq.view(N, S, C), dmod, dwqkvg[: 3 * C], dwqkvg[3 * C :].clone(), dwo, dwu, dwd, dc_mod, dw_mod]
    # the op's outputs may not alias each other: dWg (128 x 128) leaves as its own tensor
    return [dq.view(N, S, C), dmod, dwqkvg[: 3 * C], dwqkvg[3 * C :].clone(), dwo, dwu, dwd]


class _NoAutocastTF32:
    """cuBLAS calls of the modulation: autocast off (a backward may run under the caller's autocast) and TF32 allowed for the fp32 GEMMs."""

    def __enter__(self):
        self.ac = torch.autocast("cuda", enabled=False)
        self.ac.__enter__()
        self.tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        self.ac.__exit__(*exc)


def mod_bwd_cublas(g: torch.Tensor, c: torch.Tensor, wmod: torch.Tensor, dc=None, dw=None) -> tuple[torch.Tensor, torch.Tensor]:
    """``mod_bwd`` on many rows: pre = g Wmod (TF32), the elementwise kernel (dc bf16 and S = rn_bf16(silu(c)) fp32), dWmod = g^T S (TF32).
    g fp32 [R, 6C] contiguous, c bf16 [R, C] contiguous, Wmod bf16 [6C, C] -> (dc bf16 [R, C], dWmod fp32 [6C, C]); ``dc`` / ``dw`` given:
    written in place (a side-stream caller allocates them on its own stream)."""
    R = c.shape[0]
    with torch.cuda.device(c.device), _NoAutocastTF32():
        K = kernels(c.device.index if c.device.index is not None else torch.cuda.current_device())
        dc = torch.empty(R, C, device=c.device, dtype=torch.bfloat16) if dc is None else dc
        dw = torch.empty(6 * C, C, device=c.device, dtype=torch.float32) if dw is None else dw
        pre = torch.mm(g, wmod.float())
        s = torch.empty(R, C, device=c.device, dtype=torch.float32)
        n8 = R * C // 8
        K.act((min((n8 + 255) // 256, 16 * num_sms()), 1, 1), (256, 1, 1), pre, c, dc, s, int(n8))
        torch.mm(g.t(), s, out=dw)
    return dc, dw


def mod_fwd(c: torch.Tensor, wmod: torch.Tensor) -> torch.Tensor:
    """silu(c) Wmod^T [R, 6C] fp32 in one kernel; c [R, C] bf16 contiguous, Wmod [6C, C] bf16. From ``MOD_FWD_CUBLAS_ROWS`` rows: torch's
    bf16 silu and one cuBLAS GEMM with fp32 output (the same products, fp32 accumulation)."""
    if c.shape[0] >= MOD_FWD_CUBLAS_ROWS:
        with torch.autocast("cuda", enabled=False):
            return torch.mm(torch.nn.functional.silu(c), wmod.t(), out_dtype=torch.float32)
    with torch.cuda.device(c.device):
        K = kernels(c.device.index if c.device.index is not None else torch.cuda.current_device())
        run, out = K.modf.bind(c, wmod)
        run()
    return out


def mod_bwd(g: torch.Tensor, c: torch.Tensor, wmod: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(dc [R, C] bf16, dWmod [6C, C] fp32) from g = d mod [R, 6C] fp32; R a multiple of 128."""
    need_s128(c.shape[0])
    if c.shape[0] >= MOD_BWD_CUBLAS_ROWS:
        return mod_bwd_cublas(g.contiguous(), c, wmod)
    with torch.cuda.device(c.device):
        K = kernels(c.device.index if c.device.index is not None else torch.cuda.current_device())
        run, (dc, dw) = K.modb.bind(g.contiguous(), c, wmod)
        run()
    return dc, dw
