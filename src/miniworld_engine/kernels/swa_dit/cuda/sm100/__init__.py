"""sm_100a (B200) kernels of the fused SWA atom DiT block (``kernels/swa_dit``), forward and backward.

The block is ``interface.swa_dit_block`` -- the ESMFold2 SWA atom block over the flattened [N = A*B, S] atom sequence -- with
every stage on tcgen05 / TMEM / TMA kernels instead of the Triton (and sm_90 wgmma) ones; the weight-gradient GEMMs stay on
cuBLAS. Developed in ``experiments/swaatom_sm100`` (rounds v1-v13, sources unchanged here); page:
``docs/gpus/b200/swa_atom_dit/swa_atom_dit.md``.

    forward   qkvg (qkvg_fwd: 128-row tiles; qkvg_fwd2: transposed 32-row tiles for small problems / A < 5, the ATM = 32
              build for A < 4) -> window attention (attn_fwd3; attn_fwd1h when its one-head items fit one wave) ->
              out-projection + gated residual + FFN (ffn_fwd2; the ATM = 32, single-stage build for A < 4)
    backward  FFN (ffn_bwd_gate -> ffn_bwd_dy) -> out-projection (oproj_bwd) -> attention (attn_dkvq: dQ inside the dK / dV
              pass when every CTA gets >= 2 items, else attn_dq + attn_dkv) -> qkvg (qkvg_bwd), then dWqkv | dWg as one GEMM
    modulation  silu(c) Wmod^T (mod_fwd) and its backward (mod_bwd)

Global attention (``interface.is_global(half_window)``): the window attention stage (attn_fwd3 / attn_fwd1h forward, attn_dq /
attn_dkv / attn_dkvq backward) is swapped for FlashAttention-4 on the head-major Q / K / V the qkvg stage already writes
(``attn_global_fwd`` / ``attn_global_bwd``); every other stage is unchanged.

Served (:func:`supported`): B200, bf16, C = 128 with 4 heads of 32, SwiGLU hidden 256, half window 64 or global, fp32 eps. The atom
count S must be a multiple of 128 -- callers pad the atoms and ``seqused`` masks the padding; the block refuses with
ValueError otherwise (no other path for it on B200). The cubins are built on first use by the newest nvcc here that knows
sm_100a (``transition.cuda.fused_sm100a.kernel_toolchain``) and cached under ``MINIWORLD_ENGINE_JIT_ROOT``; the launches go
through the CUDA driver (``augmented_attention.cuda.sm100.driver``) on torch's current stream, CUDA-graph capturable.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import os
import subprocess
import weakref
from pathlib import Path

import torch

C, H, D, NHID, HW = 128, 4, 32, 256, 64
EPS = float(torch.finfo(torch.float32).eps)
PDL = True                                  # programmatic dependent launch for the forward kernels
_dir = Path(__file__).parent

#: cubin name -> (source stem, extra nvcc flags): the sources' defaults, and the A = 1 - 3 builds (ATM = 32; the FFN single stage)
CUBINS = {
    "qkvg_fwd": ("qkvg_fwd", ()), "qkvg_fwd2": ("qkvg_fwd2", ()), "qkvg_fwd2_a1": ("qkvg_fwd2", ("-DATM=32",)),
    "ffn_fwd2": ("ffn_fwd2", ()), "ffn_fwd2_a1": ("ffn_fwd2", ("-DATM=32", "-DNSTG=1")),
    "attn_fwd3": ("attn_fwd3", ()), "attn_fwd1h": ("attn_fwd1h", ()),
    "ffn_bwd_gate": ("ffn_bwd_gate", ()), "ffn_bwd_dy": ("ffn_bwd_dy", ()), "oproj_bwd": ("oproj_bwd", ()),
    "attn_dq": ("attn_dq", ()), "attn_dkv": ("attn_dkv", ()), "attn_dkvq": ("attn_dkvq", ()), "qkvg_bwd": ("qkvg_bwd", ()),
    "mod_fwd": ("mod_fwd", ()), "mod_bwd": ("mod_bwd", ()),
}


# --------------------------------------------------------------------------------------------------- build / load
@functools.lru_cache(maxsize=None)
def cubin(name: str) -> str:
    """Path of the cubin ``name`` (see CUBINS), built on first use; rebuilt only when a source, a flag or the toolchain changes."""
    from miniworld_engine.kernels.transition.cuda.fused_sm100a import kernel_toolchain

    stem, extra = CUBINS[name]
    nvcc, rel, host = kernel_toolchain()
    flags = (*host, "-std=c++17", "-O3", "-arch=sm_100a", "-cubin", "-lineinfo", f"-I{_dir}", *extra)
    h = hashlib.sha256(" ".join((nvcc, str(rel), *flags)).encode())
    for f in (_dir / "sm100.cuh", _dir / f"{stem}.cu"):
        h.update(f.read_bytes())
    root = Path(os.environ.get("MINIWORLD_ENGINE_JIT_ROOT", Path.home() / ".cache" / "miniworld_engine_jit"))
    out = root / "swa_dit_sm100" / f"{name}_{h.hexdigest()[:16]}.cubin"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        res = subprocess.run([nvcc, *flags, str(_dir / f"{stem}.cu"), "-o", str(tmp)], capture_output=True, text=True,
                             timeout=900, check=False)
        if res.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"nvcc {rel[0]}.{rel[1]} failed on {stem}.cu {' '.join(extra)}:\n{res.stderr[-4000:]}")
        os.replace(tmp, out)
    return str(out)


def _load_swa_sm100_kernel(name: str, func: str, smem: int, cluster: int | None = None, pdl: bool = False):
    """The kernel ``func`` of cubin ``name``, loaded in the current device's context."""
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    return driver.Kernel(cubin(name), func, smem, cluster=cluster, pdl=pdl)


def _map(t, dims, strides, box, swizzle=128, dtype="bf16"):
    from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

    return driver.TensorMap(t, dims, strides, box, swizzle=swizzle, dtype=dtype)


@functools.lru_cache(maxsize=16)
def _nsm(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


# --------------------------------------------------------------------------------------------------- kernels (host side)
def nsm():
    """Streaming multiprocessors of the current device."""
    return _nsm(torch.cuda.current_device())


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
        X, PQ, PK = ((torch.empty(N * S, C, device=dev, dtype=torch.bfloat16) for _ in range(3)) if save else (G, G, G))
        maps = (map_rows(q, A, B, S, SP, AT), _map(W, [C, 4 * C], C * 2, [64, 128]),
                _map(mod, [32, B * S, 24], [6 * C * 4, 128], [32, AT, 8], swizzle=128, dtype="f32"),   # (32 ch, rows, 24 blocks)
                _map(cos, [D // 2, B * S], [D // 2 * 4], [D // 2, AT], swizzle=0, dtype="f32"),
                _map(sin, [D // 2, B * S], [D // 2 * 4], [D // 2, AT], swizzle=0, dtype="f32"))
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        grid = (min(nsm(), ntile), 1, 1)

        def run():
            self.k(grid, (384, 1, 1), *maps, int(S), int(A), int(B), int(SP), int(AT), int(nab), int(nag), int(ntile), float(EPS), float(EPS),
                   int(save), Qh, Kh, Vh, G, X, PQ, PK)
        run.keep = (maps, W)
        return run, (Qh, Kh, Vh, G, X, PQ, PK)


def map_heads(t, A, B, S, SP, AT):
    """head-major [A, B, H, S, D] bf16 -> 5-D TMA map, box (D, AT atoms, 1 head, 1, SP augments), 64-B swizzle."""
    return _map(t, [D, S, H, B, A], [D * 2, S * D * 2, H * S * D * 2, B * H * S * D * 2], [D, AT, 1, 1, SP], swizzle=64)


class QkvgBwd:
    """dq, dP = [dpq | dpk | dV | dG] and the shift_a / scale_a gradient (the Triton _qkvg_bwd's outputs) from the forward saves (q, pq, pk)
    and the incoming gradients (dq1, dQh / dKh / dVh head-major, dG)."""
    def __init__(self, cubin="qkvg_bwd", trace=None):
        self.k = _load_swa_sm100_kernel(cubin, "swa_qkvg_bwd_sm100", 232448)
        self.trace = trace

    @staticmethod
    def smem_plan(AT):
        aux = (AT * 512 + 2 * ((AT * 64 + 127) // 128 * 128) + 1023) // 1024 * 1024
        ns = min(12, (232448 - 2 * aux - 2048 - 512 - 512) // 16384)
        assert ns >= 8, (AT, ns)
        return ns

    def bind(self, q, dq1, pq, pk, dg, dqh, dkh, dvh, mod, cos, sin, wqkv, wg, A, B, dmod=None, WT=None, dq=None, dP=None):
        M = q.numel() // C
        S = M // (A * B)
        SP = min(A, 8); AT = 64 // SP                                         # 64-row tiles
        dev = q.device
        dq = torch.empty(M, C, device=dev, dtype=torch.bfloat16) if dq is None else dq
        dP = torch.empty(M, 4 * C, device=dev, dtype=torch.bfloat16) if dP is None else dP
        own = dmod is None
        if own:
            dmod = torch.zeros(B * S, 6 * C, device=dev)
        WT = torch.cat([wqkv, wg]).t().contiguous() if WT is None else WT
        maps = (map_rows(q, A, B, S, SP, AT), map_rows(dq1, A, B, S, SP, AT), map_rows(pq, A, B, S, SP, AT), map_rows(pk, A, B, S, SP, AT),
                map_rows(dg, A, B, S, SP, AT), map_heads(dqh, A, B, S, SP, AT), map_heads(dkh, A, B, S, SP, AT), map_heads(dvh, A, B, S, SP, AT),
                _map(mod, [32, B * S, 24], [6 * C * 4, 128], [32, AT, 4], swizzle=128, dtype="f32"),
                _map(cos, [D // 2, B * S], [D // 2 * 4], [D // 2, AT], swizzle=0, dtype="f32"),
                _map(sin, [D // 2, B * S], [D // 2 * 4], [D // 2, AT], swizzle=0, dtype="f32"),
                _map(dP, [4 * C, S, B, A], [8 * C, S * 8 * C, B * S * 8 * C], [64, AT, 1, SP]),
                _map(dP, [4 * C, S, B, A], [8 * C, S * 8 * C, B * S * 8 * C], [32, AT, 1, SP], swizzle=64),
                map_rows(dq, A, B, S, SP, AT))
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        grid = (min(nsm(), ntile), 1, 1)
        ns = self.smem_plan(AT)

        def run():
            if own:
                dmod.zero_()
            self.k(grid, (384, 1, 1), *maps, int(S), int(A), int(B), int(SP), int(AT), int(nab), int(nag), int(ntile), int(ns), float(EPS), float(EPS),
                   WT, dmod, self.trace)
        run.keep = (maps, WT)
        return run, (dq, dP, dmod)


class AttnDq:
    """Window-attention backward dQ: Qh, Kh, Vh head-major [N, H, S, D] bf16, dO row-major [N S, C] bf16, LSE / Dv [N, H, S] fp32 ->
    dQh head-major bf16."""
    def __init__(self, cubin="attn_dq", trace=None):
        self.k = _load_swa_sm100_kernel(cubin, "swa_attn_dq_sm100", 232448)
        self.trace = trace

    def bind(self, Qh, Kh, Vh, dO, lse, dv, seqused, dQh=None):
        N, _, S, _ = Qh.shape
        assert S % 128 == 0
        rows = N * H * S
        dQh = torch.empty_like(Qh) if dQh is None else dQh
        maps = (_map(Qh.view(rows, D), [D, rows], D * 2, [D, 128], swizzle=64),
                _map(Kh.view(rows, D), [D, rows], D * 2, [D, 64], swizzle=64),
                _map(Vh.view(rows, D), [D, rows], D * 2, [D, 64], swizzle=64), _map(dO, [C, N * S], C * 2, [D, 128], swizzle=64),
                _map(dQh.view(rows, D), [D, rows], D * 2, [D, 128], swizzle=64))
        items = N * (S // 128) * 2
        grid = (min(nsm(), items), 1, 1)

        def run():
            self.k(grid, (384, 1, 1), *maps, seqused, lse, dv, int(S), int(N), float(D ** -0.5), self.trace)
        run.keep = maps
        return run, dQh


class AttnDkv:
    """Window-attention backward dK / dV: Qh, Kh, Vh head-major [N, H, S, D] bf16, dO row-major [N S, C] bf16, LSE / Dv [N, H, S] fp32 ->
    dKh, dVh head-major bf16."""
    def __init__(self, cubin="attn_dkv"):
        self.k = _load_swa_sm100_kernel(cubin, "swa_attn_dkv_sm100", 232448)

    def bind(self, Qh, Kh, Vh, dO, lse, dv, seqused, dKh=None, dVh=None):
        N, _, S, _ = Qh.shape
        assert S % 128 == 0
        rows = N * H * S
        dKh = torch.empty_like(Kh) if dKh is None else dKh
        dVh = torch.empty_like(Vh) if dVh is None else dVh
        maps = (_map(Qh.view(rows, D), [D, rows], D * 2, [D, 64], swizzle=64),
                _map(Kh.view(rows, D), [D, rows], D * 2, [D, 128], swizzle=64),
                _map(Vh.view(rows, D), [D, rows], D * 2, [D, 128], swizzle=64),
                _map(dO, [C, N * S], C * 2, [D, 64], swizzle=64),
                _map(lse.view(rows // 64, 64), [64, rows // 64], 64 * 4, [64, 1], swizzle=0, dtype="f32"),
                _map(dv.view(rows // 64, 64), [64, rows // 64], 64 * 4, [64, 1], swizzle=0, dtype="f32"),
                _map(dKh.view(rows, D), [D, rows], D * 2, [D, 128], swizzle=64),
                _map(dVh.view(rows, D), [D, rows], D * 2, [D, 128], swizzle=64))
        items = N * (S // 128) * 2
        grid = (min(nsm(), items), 1, 1)

        def run():
            self.k(grid, (384, 1, 1), *maps, seqused, int(S), int(N), float(D ** -0.5))
        run.keep = maps
        return run, (dKh, dVh)


class AttnDkvq(AttnDkv):
    """attn_dkvq.cu: AttnDkv plus dQh (head-major bf16, as AttnDq) from the same pass."""
    def __init__(self, cubin="attn_dkvq"):
        self.k = _load_swa_sm100_kernel(cubin, "swa_attn_dkvq_sm100", 232448)
        self.bufs = {}

    def bind(self, Qh, Kh, Vh, dO, lse, dv, seqused, dKh=None, dVh=None, dQh=None):
        N, _, S, _ = Qh.shape
        rows = N * H * S
        run0, (dKh, dVh) = super().bind(Qh, Kh, Vh, dO, lse, dv, seqused, dKh, dVh)
        dQh = torch.empty_like(Qh) if dQh is None else dQh
        maps = run0.keep[:8]
        grid = (min(nsm(), N * (S // 128) * 2), 1, 1)
        if grid[0] not in self.bufs:                                                    # kept: the kernel resets the counters it uses
            self.bufs[grid[0]] = (torch.empty(grid[0] * 8 * 64 * D, device=Qh.device),    # run-boundary partials [CTA][side][head][slot][64][32]
                                  torch.zeros((grid[0] + 1) * 4, device=Qh.device, dtype=torch.int32))   # pair counters
        bnd, cnt = self.bufs[grid[0]]

        def run():
            self.k(grid, (384, 1, 1), *maps, seqused, bnd, cnt, dQh, int(S), int(N), float(D ** -0.5))
        run.keep = (maps, bnd, cnt)
        return run, (dKh, dVh, dQh)


class OprojBwd:
    """Out-projection backward (the Triton _oproj_bwd's outputs): dq1, O, G, att [N S, C] bf16, mod [B S, 6C] fp32, Wo ->
    dO, dG, datt, gated [N S, C] bf16, D [N, H, S] fp32; d gate_a accumulated into dmod."""
    def __init__(self, cubin="oproj_bwd"):
        self.k = _load_swa_sm100_kernel(cubin, "swa_oproj_bwd_sm100", 232448)

    def bind(self, dq1, O, G, mod, wo, att, A, B, dmod=None, WOT=None, outs=None):
        M = dq1.numel() // C
        S = M // (A * B)
        SP = min(A, 8); AT = 64 // SP
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
        maps = (map_rows(dq1, A, B, S, SP, AT), map_rows(att, A, B, S, SP, AT), map_rows(G, A, B, S, SP, AT), map_rows(O, A, B, S, SP, AT),
                _map(mod, [32, B * S, 24], [6 * C * 4, 128], [32, AT, 4], swizzle=128, dtype="f32"),
                map_rows(datt, A, B, S, SP, AT), map_rows(dG, A, B, S, SP, AT), map_rows(gated, A, B, S, SP, AT), map_rows(dO, A, B, S, SP, AT))
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        grid = (min(nsm(), ntile), 1, 1)
        auxst = (AT * 512 + 1023) // 1024 * 1024
        ns = min(12, (232448 - 2 * auxst - 512) // 16384)
        assert ns >= 8, (AT, ns)

        def run():
            if own:
                dmod.zero_()
            self.k(grid, (384, 1, 1), *maps, int(S), int(A), int(B), int(SP), int(AT), int(nab), int(nag), int(ntile), int(ns), WOT, dmod, Dv)
        run.keep = (maps, WOT)
        return run, (dO, dG, Dv, datt, gated, dmod)


class FfnBwdGate:
    """First half of the FFN backward: dq2, y [N S, C] bf16, mod [B S, 6C] fp32, Wu [512, C], Wd [C, 256] -> DFFN [M, C], HH [M, 256],
    DAB [M, 512] (bf16)."""
    def __init__(self, cubin="ffn_bwd_gate"):
        self.k = _load_swa_sm100_kernel(cubin, "swa_ffn_bwd_gate_sm100", 232448)

    def bind(self, dq2, y, mod, wu, wd, A, B, WDT=None, outs=None):
        M = dq2.numel() // C
        S = M // (A * B)
        SP = min(A, 4); AT = 32 // SP
        dev = dq2.device
        if outs is None:
            dffn = torch.empty(M, C, device=dev, dtype=torch.bfloat16)
            hh = torch.empty(M, 256, device=dev, dtype=torch.bfloat16); dab = torch.empty(M, 512, device=dev, dtype=torch.bfloat16)
        else:
            dffn, hh, dab = outs
        WDT = wd.t().contiguous() if WDT is None else WDT
        st = lambda t, w: _map(t, [w, S, B, A], [w * 2, S * w * 2, B * S * w * 2], [64, AT, 1, SP])
        maps = (map_rows(y, A, B, S, SP, AT), map_rows(dq2, A, B, S, SP, AT),
                _map(mod, [32, B * S, 24], [6 * C * 4, 128], [32, AT, 4], swizzle=128, dtype="f32"),
                map_rows(dffn, A, B, S, SP, AT), st(hh, 256), st(dab, 512))
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        grid = (min(nsm(), ntile), 1, 1)
        auxst = (AT * 512 + 1023) // 1024 * 1024
        ns = min(8, (232448 - 2 * 49152 - 2 * auxst - 512) // 8192)
        assert ns >= 4, (AT, ns)

        def run():
            self.k(grid, (384, 1, 1), *maps, int(S), int(A), int(B), int(SP), int(AT), int(nab), int(nag), int(ntile), int(ns), wu, WDT)
        run.keep = (maps, WDT)
        return run, (dffn, hh, dab)


class FfnBwdDy:
    """Second half of the FFN backward: DAB [M, 512], dq2, q1, ffn [M, C] bf16, mod, Wu -> dq1 [M, C] bf16; d shift_f / scale_f / gate_f
    accumulated into dmod."""
    def __init__(self, cubin="ffn_bwd_dy"):
        self.k = _load_swa_sm100_kernel(cubin, "swa_ffn_bwd_dy_sm100", 232448)

    def bind(self, dab, dq2, q1, ffn, mod, wu, A, B, dmod=None, WUT=None, dq1=None):
        M = dq2.numel() // C
        S = M // (A * B)
        SP = min(A, 8); AT = 64 // SP
        dev = dq2.device
        dq1 = torch.empty(M, C, device=dev, dtype=torch.bfloat16) if dq1 is None else dq1
        own = dmod is None
        if own:
            dmod = torch.zeros(B * S, 6 * C, device=dev)
        WUT = wu.t().contiguous() if WUT is None else WUT
        maps = (_map(dab, [512, S, B, A], [1024, S * 1024, B * S * 1024], [64, AT, 1, SP]),
                map_rows(q1, A, B, S, SP, AT), map_rows(ffn, A, B, S, SP, AT), map_rows(dq2, A, B, S, SP, AT),
                _map(mod, [32, B * S, 24], [6 * C * 4, 128], [32, AT, 4], swizzle=128, dtype="f32"), map_rows(dq1, A, B, S, SP, AT))
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        grid = (min(nsm(), ntile), 1, 1)
        auxst = (AT * 512 + 1023) // 1024 * 1024
        ns = min(28, (232448 - 2 * auxst - 2048 - 512 - 1024) // 8192)
        assert ns >= 14, (AT, ns)

        def run():
            if own:
                dmod.zero_()
            self.k(grid, (384, 1, 1), *maps, int(S), int(A), int(B), int(SP), int(AT), int(nab), int(nag), int(ntile), int(ns), float(EPS), WUT, dmod)
        run.keep = (maps, WUT)
        return run, (dq1, dmod)


class FfnFwd2:
    """Transposed out-projection + FFN forward (ffn_fwd2.cu): same inputs / outputs as FfnFwd, 32-row tiles, weights on chip."""
    def __init__(self, cubin="ffn_fwd2", trace=None, cluster=1, max_ctas=None, atm=8):
        self.k = _load_swa_sm100_kernel(cubin, "swa_ffn_fwd2_sm100", 232448, cluster=cluster if cluster > 1 else None, pdl=PDL)
        self.atm = atm                                                     # the build's ATM (ffn_fwd2_a1.cubin: 32, for A = 1 - 3)
        self.cl = cluster
        self.trace = trace
        self.max_ctas = max_ctas                                           # co-resident CTA cap (clusters must fit in a GPC)
        self.trace = trace

    def bind(self, q, g, o, mod, wo, wu, wd, A, B, save=False):
        M = q.numel() // C
        S = M // (A * B)
        SP = min(A, 8); AT = 32 // SP
        assert AT <= self.atm, f"AT = {AT} needs a build with ATM >= {AT}"
        dev = q.device
        out = torch.empty(M, C, device=dev, dtype=torch.bfloat16)
        Q1, Att, Y, Ff = ((torch.empty(M, C, device=dev, dtype=torch.bfloat16) for _ in range(4)) if save else (out, out, out, out))
        mr = lambda t: map_rows(t, A, B, S, SP, AT)
        maps = (mr(q), mr(g), mr(o), _map(mod, [32, B * S, 24], [6 * C * 4, 128], [32, AT, 16], swizzle=128, dtype="f32"),
                _map(wo, [C, C], C * 2, [64, 128]), mr(out), mr(Q1), mr(Att), mr(Y), mr(Ff),
                _map(wu, [C, 512], C * 2, [32, 128], swizzle=64), _map(wd, [256, C], 256 * 2, [32, 128], swizzle=64))
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        cl = self.cl
        grid = (max(cl, min(self.max_ctas or nsm(), ntile) // cl * cl), 1, 1)   # a multiple of the cluster size (idle CTAs still multicast)

        def run():
            self.k(grid, (384, 1, 1), *maps, int(S), int(A), int(B), int(SP), int(AT), int(nab), int(nag), int(ntile), float(EPS), int(save), wu, wd, self.trace)
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
        maps = (_map(Qh.view(rows, D), [D, rows], D * 2, [D, 128], swizzle=64),
                _map(Kh.view(rows, D), [D, rows], D * 2, [D, 256], swizzle=64),
                _map(Vh.view(rows, D), [D, rows], D * 2, [D, 256], swizzle=64),
                _map(O, [C, N * S], C * 2, [D, 128], swizzle=64))
        items = N * (S // 128) * (H // self.heads_per_item)
        grid = (min(nsm(), items), 1, 1)

        def run():
            self.k(grid, (self.threads, 1, 1), *maps, seqused, LSE, int(S), int(N), float(D ** -0.5), self.trace)
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


class QkvgFwd2:
    """Transposed qkvg forward (qkvg_fwd2.cu): same interface as QkvgFwd, 32-row tiles, W in TMEM."""
    def __init__(self, cubin="qkvg_fwd2", atm=8, max_ctas=None):
        self.k = _load_swa_sm100_kernel(cubin, "swa_qkvg_fwd2_sm100", 232448, pdl=PDL)
        self.max_ctas = max_ctas
        self.atm = atm                                                     # the build's ATM (qkvg_fwd2_a1.cubin: 32, for A = 1 - 3)

    def bind(self, q, mod, cos, sin, wqkv, wg, A, B, save=False, W=None):
        N, S, _ = q.shape
        SP = min(A, 8); AT = 32 // SP
        assert AT <= self.atm, f"AT = {AT} needs a build with ATM >= {AT}"
        dev = q.device
        W = torch.cat([wqkv, wg]).contiguous() if W is None else W
        Qh, Kh, Vh = (torch.empty(N, H, S, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
        G = torch.empty(N * S, C, device=dev, dtype=torch.bfloat16)
        X, PQ, PK = ((torch.empty(N * S, C, device=dev, dtype=torch.bfloat16) for _ in range(3)) if save else (G, G, G))
        mr = lambda t: map_rows(t, A, B, S, SP, AT)
        mh = lambda t: map_heads(t, A, B, S, SP, AT)
        maps = (mr(q), _map(W, [C, 4 * C], C * 2, [32, 128], swizzle=64),
                _map(mod, [32, B * S, 24], [6 * C * 4, 128], [32, AT, 8], swizzle=128, dtype="f32"),
                _map(cos, [D // 2, B * S], [D // 2 * 4], [D // 2, AT], swizzle=0, dtype="f32"),
                _map(sin, [D // 2, B * S], [D // 2 * 4], [D // 2, AT], swizzle=0, dtype="f32"),
                mh(Qh), mh(Kh), mh(Vh), mr(G), mr(X), mr(PQ), mr(PK))
        nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
        ntile = nab * nag * B
        grid = (min(self.max_ctas or nsm(), ntile), 1, 1)

        def run():
            self.k(grid, (384, 1, 1), *maps, int(S), int(A), int(B), int(SP), int(AT), int(nab), int(nag), int(ntile), float(EPS), float(EPS), int(save))
        run.keep = (maps, W)
        return run, (Qh, Kh, Vh, G, X, PQ, PK)


class ModFwd:
    """adaLN modulation in one kernel (mod_fwd.cu): c [R, C] bf16, Wmod [6C, C] bf16 -> silu(c) Wmod^T [R, 6C] fp32."""
    def __init__(self, cubin="mod_fwd"):
        self.k = _load_swa_sm100_kernel(cubin, "swa_mod_fwd_sm100", 4 * 32768 + 64, pdl=PDL)

    def bind(self, c, wmod, out=None):
        R = c.shape[0]
        out = torch.empty(R, 6 * C, device=c.device) if out is None else out
        maps = (_map(c, [C, R], C * 2, [64, 128]), _map(wmod, [C, 6 * C], C * 2, [64, 128]),
                _map(out, [6 * C, R], 6 * C * 4, [32, 128], dtype="f32"))

        def run():
            self.k(((R + 127) // 128, 6, 1), (128, 1, 1), *maps)
        run.keep = maps
        return run, out


class ModBwd:
    """Backward of ModFwd (mod_bwd.cu): g [R, 6C] fp32, c [R, C] bf16, Wmod [6C, C] bf16 -> dc [R, C] bf16, dWmod [6C, C] bf16."""
    def __init__(self, cubin="mod_bwd"):
        self.cubin = cubin
        self.kc = _load_swa_sm100_kernel(self.cubin, "swa_mod_bwd_dc_sm100", 4 * 49152 + 128, cluster=4)
        self.kw = {}                                                        # per cluster size G
        self.trace = None                                                   # (TRACE builds) int64 [16]

    def bind(self, g, c, wmod, dc=None, dw=None):
        R = c.shape[0]
        assert R % 128 == 0 and g.shape == (R, 6 * C) and g.is_contiguous() and c.is_contiguous()
        dev = c.device
        dc = torch.empty(R, C, device=dev, dtype=torch.bfloat16) if dc is None else dc
        dw = torch.empty(6 * C, C, device=dev, dtype=torch.bfloat16) if dw is None else dw
        G = max(x for x in (1, 2, 4, 8, 16) if (R // 64) % x == 0)          # row groups (one cluster) of NS x 64 rows
        NS = R // 64 // G
        if G not in self.kw:
            self.kw[G] = _load_swa_sm100_kernel(self.cubin, "swa_mod_bwd_dw_sm100", 4 * 49152 + 128, cluster=G)
        kw = self.kw[G]
        mc_ = (_map(g, [6 * C, R], 6 * C * 4, [32, 128], dtype="f32"), _map(wmod, [C, 6 * C], C * 2, [64, 64]))
        mw_ = (_map(g, [6 * C, R], 6 * C * 4, [32, 64], dtype="f32"), _map(c, [C, R], C * 2, [64, 64]))
        run_dc = lambda: self.kc((4, R // 128, 1), (288, 1, 1), *mc_, c, dc, self.trace)
        run_dw = lambda: kw((G, 6, 1), (288, 1, 1), *mw_, dw, int(NS))

        def run():
            run_dc(); run_dw()
        run.keep = (mc_, mw_)
        run.dc, run.dw = run_dc, run_dw
        return run, (dc, dw)


# --------------------------------------------------------------------------------------------------- the block
class _Kernels:
    """Every kernel the block launches, loaded on one device."""

    def __init__(self) -> None:
        self.qkvg, self.qkvg2, self.qkvg2a = QkvgFwd(), QkvgFwd2(), QkvgFwd2("qkvg_fwd2_a1", atm=32)
        self.attn, self.attn1h = AttnFwd3(), AttnFwd1h()
        self.ffn, self.ffna = FfnFwd2(), FfnFwd2("ffn_fwd2_a1", atm=32)
        self.gate, self.dy, self.oproj = FfnBwdGate(), FfnBwdDy(), OprojBwd()
        self.dq, self.dkv, self.dkvq, self.qkvgb = AttnDq(), AttnDkv(), AttnDkvq(), QkvgBwd()
        self.modf, self.modb = ModFwd(), ModBwd()


@functools.lru_cache(maxsize=16)
def kernels(index: int) -> _Kernels:
    """Build and load every kernel on device ``index`` (raises on a toolchain or driver failure)."""
    with torch.cuda.device(index):
        return _Kernels()


def supported(q: torch.Tensor, nhid: int, half_window: int, eps: float) -> bool:
    """The shapes and dtypes the kernels serve (``S % 128`` is checked by the block, which refuses instead of falling back)."""
    from miniworld_engine.kernels.swa_dit.interface import is_global

    return (q.is_cuda and q.dtype == torch.bfloat16 and q.dim() == 3 and q.shape[-1] == C and nhid == NHID
            and (half_window == HW or is_global(half_window)) and eps == EPS
            and torch.cuda.get_device_capability(q.device) == (10, 0))


def _has_fa4() -> bool:
    """FlashAttention-4 (``flash_attn.cute``) installed -- looked up without importing it."""
    try:
        spec = importlib.util.find_spec("flash_attn")
    except (ImportError, ValueError):
        return False
    return bool(spec is not None and spec.submodule_search_locations
                and any((Path(r) / "cute" / "__init__.py").exists() for r in spec.submodule_search_locations))


#: Global attention runs on FlashAttention-4; resolved once at import so ``interface.refusal`` can read it while tracing.
FA4_AVAILABLE = _has_fa4()


def need_s128(S: int) -> None:
    if S % 128:
        raise ValueError(f"SWA atom DiT block (sm_100a): the atom count S = {S} must be a multiple of 128; pad the atoms "
                         "(seqused masks them)")


_PACKS: dict = {}


def _cached(fn, *ts):
    """fn(*ts), cached while the very same tensor objects are alive and unmodified (the weight forms are rebuilt when a weight is
    updated in place -- its version moves -- or replaced). The key holds the objects themselves (weakly): a new tensor that the
    allocator happens to place at a freed one's address, with the same shape and version -- a fresh bf16 cast of an fp32 master
    weight, a test's next set of weights -- must not be served the old tensor's packed form.

    Not cached: while a CUDA graph is being captured (a hit would record no pack kernel, and every replay would then read the
    packed copy of the weights as they were at capture, never refreshed after an optimizer step), and for inference tensors
    (they carry no version counter)."""
    if torch.cuda.is_current_stream_capturing():
        return fn(*ts)
    try:
        versions = tuple(t._version for t in ts)
    except RuntimeError:                                  # inference tensors do not track a version counter
        return fn(*ts)
    key = (fn.__name__, *(id(t) for t in ts))
    hit = _PACKS.get(key)
    if hit is not None:
        refs, vers, out = hit
        if vers == versions and all(r() is t for r, t in zip(refs, ts, strict=True)):
            return out
    if len(_PACKS) > 64:
        _PACKS.clear()
    out = fn(*ts)
    _PACKS[key] = (tuple(weakref.ref(t) for t in ts), versions, out)
    return out


def _w_qkvg(wqkv, wg):
    return torch.cat([wqkv, wg]).contiguous()


def _w_qkvg_t(wqkv, wg):
    return torch.cat([wqkv, wg]).t().contiguous()


def _t(w):
    return w.t().contiguous()


def attn_global_fwd(Qh, Kh, Vh, seqused, O, lse):
    """Global attention forward on FlashAttention-4: Qh, Kh, Vh head-major [N, H, S, D] bf16 (read through their [N, S, H, D]
    transposed views), seqused [N] int32 -> O [N S, C] bf16 and lse [N, H, S] fp32, written in place. Rows at or past
    ``seqused`` are skipped by FA4 -- ``O`` must arrive zeroed there (and their lse stays unwritten); keys past ``seqused``
    are never attended. FA4 keeps its own lse convention, which only :func:`attn_global_bwd` reads."""
    from flash_attn.cute.interface import _flash_attn_fwd  # ty: ignore[unresolved-import]  # optional FlashAttention backend

    N, _, S, _ = Qh.shape
    _flash_attn_fwd(Qh.transpose(1, 2), Kh.transpose(1, 2), Vh.transpose(1, 2), seqused_q=seqused, seqused_k=seqused,
                    softmax_scale=D ** -0.5, causal=False, return_lse=True, out=O.view(N, S, H, D), lse=lse)


def attn_global_bwd(Qh, Kh, Vh, O, dO, lse, seqused, dQh, dKh, dVh):
    """Global attention backward on FlashAttention-4 (its own softmax-delta pass included): dQh, dKh, dVh head-major
    [N, H, S, D] bf16 written in place -- zero them first, FA4 leaves the rows at or past ``seqused`` alone."""
    from flash_attn.cute.interface import _flash_attn_bwd  # ty: ignore[unresolved-import]  # optional FlashAttention backend

    N, _, S, _ = Qh.shape
    _flash_attn_bwd(Qh.transpose(1, 2), Kh.transpose(1, 2), Vh.transpose(1, 2), O.view(N, S, H, D), dO.view(N, S, H, D), lse,
                    softmax_scale=D ** -0.5, causal=False, seqused_q=seqused, seqused_k=seqused,
                    dq=dQh.transpose(1, 2), dk=dKh.transpose(1, 2), dv=dVh.transpose(1, 2))


def block_fwd(q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, B, save, half_window=HW):
    """The bf16 forward: [out] or, with ``save``, [out, Qh, Kh, Vh, G, O, lse, q1, X, PQ, PK, Att, Y, FF] -- the layout of
    ``dispatch.swa_dit_block_fwd``. ``half_window`` is the window (64) or a global-attention value (``is_global``)."""
    from miniworld_engine.kernels.swa_dit.interface import is_global

    N, S, _ = q.shape
    need_s128(S)
    A = N // B
    with torch.cuda.device(q.device):
        K = kernels(q.device.index if q.device.index is not None else torch.cuda.current_device())
        W = _cached(_w_qkvg, wqkv, wg)
        qk = (K.qkvg2a if A < 4 else K.qkvg2) if A < 5 or A * B * S <= 2 * 148 * 32 else K.qkvg
        run, (Qh, Kh, Vh, G, X, PQ, PK) = qk.bind(q, mod, cos, sin, None, None, A, B, save=save, W=W)
        run()
        lse = torch.empty(N, H, S, device=q.device, dtype=torch.float32)
        if is_global(half_window):
            O = torch.zeros(N * S, C, device=q.device, dtype=torch.bfloat16)
            attn_global_fwd(Qh, Kh, Vh, seqused, O, lse)
        else:
            O = torch.empty(N * S, C, device=q.device, dtype=torch.bfloat16)
            at = K.attn1h if Qh.shape[0] * (S // 128) * 4 <= nsm() else K.attn
            run, _, _ = at.bind(Qh, Kh, Vh, seqused, O=O, LSE=lse)
            run()
        run, (out, q1, att, y, ffn) = (K.ffn if A >= 4 else K.ffna).bind(q.reshape(N * S, C), G, O, mod, wo, wu, wd, A, B,
                                                                         save=save)
        run()
    if not save:
        return [out.view(N, S, C)]
    return [out.view(N, S, C), Qh, Kh, Vh, G, O, lse, q1, X, PQ, PK, att, y, ffn]


def block_bwd(dy, q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, qh, kh, vh, g, o, lse, q1, x, pq, pk, att, y, ffn, B,
              half_window=HW):
    """The bf16 backward: [dq, dmod, dWqkv, dWg, dWo, dWu, dWd] -- the layout of ``dispatch.swa_dit_block_bwd``."""
    from miniworld_engine.kernels.swa_dit.interface import is_global

    N, S, _ = q.shape
    need_s128(S)
    A = N // B
    M = N * S
    dev = q.device
    with torch.cuda.device(dev):
        K = kernels(dev.index if dev.index is not None else torch.cuda.current_device())
        dmod = torch.zeros(B * S, 6 * C, device=dev, dtype=torch.float32)
        dy2 = dy.reshape(M, C)
        run, (dffn, hh, dab) = K.gate.bind(dy2, y, mod, wu, None, A, B, WDT=_cached(_t, wd))
        run()
        run, (dq1, _) = K.dy.bind(dab, dy2, q1, ffn, mod, wu, A, B, dmod=dmod, WUT=_cached(_t, wu))
        run()
        dO, dG, datt, gated = (torch.empty(M, C, device=dev, dtype=torch.bfloat16) for _ in range(4))
        Dv = torch.empty(N, H, S, device=dev, dtype=torch.float32)
        run, _ = K.oproj.bind(dq1, o, g, mod, wo, att, A, B, dmod=dmod, WOT=_cached(_t, wo), outs=(dO, dG, Dv, datt, gated))
        run()
        if is_global(half_window):                         # FA4: dQ / dK / dV in one call; skipped (padding) rows stay zero
            dQh, dKh, dVh = (torch.zeros(N, H, S, D, device=dev, dtype=torch.bfloat16) for _ in range(3))
            attn_global_bwd(qh, kh, vh, o, dO, lse, seqused, dQh, dKh, dVh)
        else:
            dQh = torch.empty(N, H, S, D, device=dev, dtype=torch.bfloat16)
            dKh, dVh = torch.empty_like(dQh), torch.empty_like(dQh)
            if os.environ.get("MINIWORLD_SWA_DIT_SM100_FUSED_BWD", "1") == "1" and N * (S // 128) * 2 >= 2 * nsm():
                run, _ = K.dkvq.bind(qh, kh, vh, dO, lse, Dv, seqused, dKh=dKh, dVh=dVh, dQh=dQh)
                run()
            else:
                run, _ = K.dq.bind(qh, kh, vh, dO, lse, Dv, seqused, dQh=dQh)
                run()
                run, _ = K.dkv.bind(qh, kh, vh, dO, lse, Dv, seqused, dKh=dKh, dVh=dVh)
                run()
        dq = torch.empty(M, C, device=dev, dtype=torch.bfloat16)
        dP = torch.empty(M, 4 * C, device=dev, dtype=torch.bfloat16)
        run, _ = K.qkvgb.bind(q.reshape(M, C), dq1, pq, pk, dG, dQh, dKh, dVh, mod, cos, sin, wqkv, wg, A, B, dmod=dmod,
                              WT=_cached(_w_qkvg_t, wqkv, wg), dq=dq, dP=dP)
        run()
    dwqkvg = dP.t() @ x                                    # dWqkv | dWg as one GEMM over dP (x read once)
    # the op's outputs may not alias each other: dWg (128 x 128) leaves as its own tensor
    return [dq.view(N, S, C), dmod, dwqkvg[:3 * C], dwqkvg[3 * C:].clone(), datt.t() @ gated, dab.t() @ y, dffn.t() @ hh]


def mod_fwd(c: torch.Tensor, wmod: torch.Tensor) -> torch.Tensor:
    """silu(c) Wmod^T [R, 6C] fp32 in one kernel; c [R, C] bf16 contiguous, Wmod [6C, C] bf16."""
    with torch.cuda.device(c.device):
        K = kernels(c.device.index if c.device.index is not None else torch.cuda.current_device())
        run, out = K.modf.bind(c, wmod)
        run()
    return out


def mod_bwd(g: torch.Tensor, c: torch.Tensor, wmod: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(dc [R, C] bf16, dWmod [6C, C] bf16) from g = d mod [R, 6C] fp32; R a multiple of 128."""
    need_s128(c.shape[0])
    with torch.cuda.device(c.device):
        K = kernels(c.device.index if c.device.index is not None else torch.cuda.current_device())
        run, (dc, dw) = K.modb.bind(g.contiguous(), c, wmod)
        run()
    return dc, dw
