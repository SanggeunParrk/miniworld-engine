"""sm_100a (B200) backward of the fused SWA atom DiT block in fp32: hand-written tcgen05 kernels on TF32 tensor cores
(``kind::tf32``), no Triton and no PyTorch arithmetic on the block's path -- the fp32 counterpart of :func:`sm100.block_bwd`.

Same math and the same outputs as ``dispatch._swa_dit_bwd_fp32_launch`` (the Triton fp32 backward), with two differences of
precision, both towards fp32:

* the window-attention backward runs on the fp32 Q / K / V / dO (TF32 MMAs, fp32 softmax) where the Triton path ran the bf16
  kernels on bf16 operands (dO and dQ / dK / dV in bf16);
* every MMA operand is rounded to TF32 to nearest (``cvt.rna``) before the tensor core sees it -- the weights once per weight version
  (``bwd_prep_tf32.cu``, cached), the operands the kernels' threads produce on their way out -- instead of the hardware's truncation of
  fp32 inputs (a -2^-11 bias per operand). The operands that also feed the cuBLAS TF32 weight gradients (dffn, h, [da | db], datt,
  gated, dP) are stored rounded, which those GEMMs would do anyway. The forward's saves (Q / K / V, y, x, ...) are read as stored.

Stages (``kernels`` names the cubins):

    ffn_bwd_tf32     FFN backward, gate and input side in one kernel: dffn, h, [da | db], dq1, d shift_f / scale_f / gate_f
    oproj_bwd_tf32   out-projection backward: datt, gated, dO, dG, the softmax delta D, d gate_a
    attn_dkv_tf32    window attention dK / dV (key-major)        } separate passes: every dQ is complete in one item, deterministic,
    attn_dq_tf32     window attention dQ (query-major, recomputes S / dP)  } no fp32 dQ buffer, no MN-major A operand
    qkvg_bwd_tf32    q | k | v | gate backward: dP = [dpq | dpk | dV | dG], dq, d shift_a / scale_a
    then dWqkv | dWg = dP^T x, dWo = datt^T gated, dWu = [da | db]^T y, dWd = dffn^T h as cuBLAS TF32 GEMMs (``allow_tf32`` forced on
    around them, restored after) -- the weight gradients are the general GEMMs left to cuBLAS.

    mod_bwd_tf32     the hoisted modulation's backward: dc = silu'(c) (g Wmod) on TF32 tensor cores, dWmod = g^T silu(c) on cuBLAS TF32

Saved tensors (the fp32 forward's contract, all fp32): Qh, Kh, Vh head-major [N, 4, S, 32] (after head RMS + RoPE), G [N S, C] raw gate,
O [N S, C] attention output, lse [N, 4, S] (natural-log sum of exp(scores / sqrt 32), as the bf16 attn_fwd3), q1, X, PQ, PK, Att, Y, FF
[N S, C] with the meanings of the bf16 sm100 path's saves (q1 after the attention residual, X the modulated input, PQ / PK the pre-norm
q / k projections, Att = gated Wo^T, Y the FFN input, FF the FFN output before gate_f).

Served (:func:`supported_tf32`): B200, fp32, C = 128 with 4 heads of 32, SwiGLU hidden 256, half window 64, fp32 eps, S a multiple of
128 (else the caller keeps the Triton path). The cubins build on first use with the newest nvcc that knows sm_100a and are cached under
``MINIWORLD_ENGINE_JIT_ROOT`` (``swa_dit_sm100_tf32_bwd/``); launches go through the CUDA driver on torch's current stream.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import os
import subprocess
from pathlib import Path

import torch

from miniworld_engine.kernels.swa_dit.cuda.sm100 import EPS, HW, NHID, C, D, H, _cached, _map, nsm

_dir = Path(__file__).parent
SMEM_MAX = 232448
#: kernel -> (source stem, function, dynamic shared memory bytes)
KERNELS = {
    "prep": ("bwd_prep_tf32", "swa_tf32_prep", 0),
    "ffn": ("ffn_bwd_tf32", "swa_ffn_bwd_tf32_sm100", SMEM_MAX),
    "oproj": ("oproj_bwd_tf32", "swa_oproj_bwd_tf32_sm100", SMEM_MAX),
    "dkv": ("attn_dkv_tf32", "swa_attn_dkv_tf32_sm100", SMEM_MAX),
    "dq": ("attn_dq_tf32", "swa_attn_dq_tf32_sm100", SMEM_MAX),
    "qkvg": ("qkvg_bwd_tf32", "swa_qkvg_bwd_tf32_sm100", SMEM_MAX),
    "mod": ("mod_bwd_tf32", "swa_mod_bwd_dc_tf32_sm100", 4 * 32768 + 256),
}
THREADS = (384, 1, 1)


# --------------------------------------------------------------------------------------------------- build / load
@functools.lru_cache(maxsize=None)
def cubin(stem: str) -> str:
    """Path of the cubin of ``stem``.cu, built on first use; rebuilt only when the source, sm100.cuh, a flag or the toolchain changes."""
    from miniworld_engine.kernels.transition.cuda.fused_sm100a import kernel_toolchain

    nvcc, rel, host = kernel_toolchain()
    flags = (*host, "-std=c++17", "-O3", "-arch=sm_100a", "-cubin", "-lineinfo", f"-I{_dir}")
    h = hashlib.sha256(" ".join((nvcc, str(rel), *flags)).encode())
    for f in (_dir / "sm100.cuh", _dir / f"{stem}.cu"):
        h.update(f.read_bytes())
    root = Path(os.environ.get("MINIWORLD_ENGINE_JIT_ROOT", Path.home() / ".cache" / "miniworld_engine_jit"))
    out = root / "swa_dit_sm100_tf32_bwd" / f"{stem}_{h.hexdigest()[:16]}.cubin"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        res = subprocess.run([nvcc, *flags, str(_dir / f"{stem}.cu"), "-o", str(tmp)], capture_output=True, text=True,
                             timeout=900, check=False)
        if res.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"nvcc {rel[0]}.{rel[1]} failed on {stem}.cu:\n{res.stderr[-4000:]}")
        os.replace(tmp, out)
    return str(out)


class _Kernels:
    """Every kernel of the fp32 backward, loaded on one device (attribute per ``KERNELS`` key)."""

    def __init__(self) -> None:
        from miniworld_engine.kernels.augmented_attention.cuda.sm100 import driver

        for name, (stem, func, smem) in KERNELS.items():
            setattr(self, name, driver.Kernel(cubin(stem), func, smem))


@functools.lru_cache(maxsize=16)
def kernels(index: int) -> _Kernels:
    """Build and load every kernel on device ``index`` (raises on a toolchain or driver failure)."""
    with torch.cuda.device(index):
        return _Kernels()


kernels_tf32 = kernels  # the name dispatch._tf32_bwd loads up front


def _here() -> _Kernels:
    return kernels(torch.cuda.current_device())


def supported_tf32(q: torch.Tensor, nhid: int, half_window: int, eps: float) -> bool:
    """The calls the fp32 sm_100a kernels serve: B200, q fp32 [N, S, 128] with S a multiple of 128, SwiGLU hidden 256, half window 64
    (not global attention), fp32 eps. Anything else keeps the Triton fp32 path (no refusal: the Triton path serves every S)."""
    return (q.is_cuda and q.dtype == torch.float32 and q.dim() == 3 and q.shape[-1] == C and q.shape[1] % 128 == 0
            and nhid == NHID and half_window == HW and eps == EPS and torch.cuda.get_device_capability(q.device) == (10, 0))


@contextlib.contextmanager
def _tf32():
    """cuBLAS on TF32 tensor cores for the weight gradients whatever the caller's allow_tf32 (restored after): the fp32 block is the
    TF32 recipe end to end (as ``conditioned_transition``'s fp32 runner)."""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


# --------------------------------------------------------------------------------------------------- weight forms (rounded to TF32)
def _prep(src: torch.Tensor, dst: torch.Tensor, transpose: bool) -> None:
    """dst (row stride dst.stride(0)) = rna_tf32(src) or its transpose; src fp32 [rows, cols] contiguous."""
    rows, cols = src.shape
    _here().prep(((cols + 31) // 32, (rows + 31) // 32, 1), (32, 8, 1), src, dst, int(rows), int(cols), int(dst.stride(0)),
                 int(transpose))


def _tf32_ffn(wu: torch.Tensor, wd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(Wu [512, 128], Wd^T [256, 128], Wu^T [128, 512]), rounded."""
    wu, wd = wu.float().contiguous(), wd.float().contiguous()
    wur = torch.empty_like(wu)
    wdt = torch.empty(wd.shape[1], wd.shape[0], device=wd.device)
    wut = torch.empty(wu.shape[1], wu.shape[0], device=wu.device)
    _prep(wu, wur, False)
    _prep(wd, wdt, True)
    _prep(wu, wut, True)
    return wur, wdt, wut


def _tf32_wot(wo: torch.Tensor) -> torch.Tensor:
    """Wo^T [128, 128], rounded."""
    wo = wo.float().contiguous()
    out = torch.empty(wo.shape[1], wo.shape[0], device=wo.device)
    _prep(wo, out, True)
    return out


def _tf32_wt(wqkv: torch.Tensor, wg: torch.Tensor) -> torch.Tensor:
    """[Wqkv; Wg]^T [128, 512], rounded."""
    wqkv, wg = wqkv.float().contiguous(), wg.float().contiguous()
    out = torch.empty(C, 4 * C, device=wqkv.device)
    _prep(wqkv, out, True)
    _prep(wg, out[:, 3 * C:], True)
    return out


def _tf32_wmt(wmod: torch.Tensor) -> torch.Tensor:
    """Wmod^T [128, 768], rounded."""
    wmod = wmod.float().contiguous()
    out = torch.empty(wmod.shape[1], wmod.shape[0], device=wmod.device)
    _prep(wmod, out, True)
    return out


# --------------------------------------------------------------------------------------------------- TMA maps (fp32)
def tiling(A: int) -> tuple[int, int]:
    """(SP augments, AT atoms) of a row tile of <= 128 rows: SP = min(A, 16), AT = 128 // SP (AT = 8 from A = 16 on: the kernels'
    register path for the modulation-gradient augment sums)."""
    SP = min(A, 16)
    return SP, 128 // SP


def _rows(t, W, A, B, S, SP, AT):
    """[A, B, S, W] fp32 rows -> 4-D map, box (32 columns, AT atoms, 1, SP augments): [SP AT rows][32] 128-B swizzled."""
    return _map(t, [W, S, B, A], [W * 4, S * W * 4, B * S * W * 4], [32, AT, 1, SP], swizzle=128, dtype="f32")


def _heads(t, A, B, S, SP, AT):
    """head-major [A, B, H, S, 32] fp32 -> 5-D map, box (32, AT atoms, 1 head, 1, SP augments)."""
    return _map(t, [D, S, H, B, A], [D * 4, S * D * 4, H * S * D * 4, B * H * S * D * 4], [D, AT, 1, 1, SP], swizzle=128, dtype="f32")


def _mat(t, rows, cols, box_rows, swizzle=128):
    """row-major fp32 [rows][cols] -> 2-D map, box (32 columns, box_rows rows)."""
    return _map(t, [cols, rows], cols * 4, [32, box_rows], swizzle=swizzle, dtype="f32")


def _row_grid(A, B, S):
    SP, AT = tiling(A)
    nab, nag = (S + AT - 1) // AT, (A + SP - 1) // SP
    ntile = nab * nag * B
    return SP, AT, nab, nag, ntile, (min(nsm(), ntile), 1, 1)


# --------------------------------------------------------------------------------------------------- stages
def ffn_bwd(dy2, q1, ffn, y, mod, wu, wd, A, B, S, dmod, eps=EPS):
    """FFN backward (ffn_bwd_tf32.cu): dy2 = d out, q1, ffn, y [M, C] fp32 -> (dffn [M, C], hh [M, 256], dab [M, 512], dq1 [M, C]);
    d shift_f / scale_f / gate_f accumulated into dmod [B S, 6C]."""
    M = dy2.shape[0]
    dev = dy2.device
    SP, AT, nab, nag, ntile, grid = _row_grid(A, B, S)
    wur, wdt, wut = _cached(_tf32_ffn, wu, wd)
    dffn = torch.empty(M, C, device=dev)
    hh = torch.empty(M, NHID, device=dev)
    dab = torch.empty(M, 2 * NHID, device=dev)
    dq1 = torch.empty(M, C, device=dev)
    maps = (_rows(y, C, A, B, S, SP, AT), _mat(wur, 2 * NHID, C, 32), _mat(wdt, NHID, C, 32), _mat(wut, C, 2 * NHID, C),
            _rows(hh, NHID, A, B, S, SP, AT), _rows(dab, 2 * NHID, A, B, S, SP, AT))
    _here().ffn(grid, THREADS, *maps, dy2, q1, ffn, mod, dffn, dq1, dmod, int(S), int(A), int(B), int(SP), int(AT), int(nab), int(nag),
                int(ntile), float(eps))
    return dffn, hh, dab, dq1


def oproj_bwd(dq1, att, g, o, mod, wo, A, B, S, dmod):
    """Out-projection backward (oproj_bwd_tf32.cu): dq1, att, g, o [M, C] fp32 -> (datt, gated, dO, dG [M, C], Dv [N, H, S]);
    d gate_a accumulated into dmod."""
    M = dq1.shape[0]
    dev = dq1.device
    SP, AT, nab, nag, ntile, grid = _row_grid(A, B, S)
    wot = _cached(_tf32_wot, wo)
    datt, gated, dO, dG = (torch.empty(M, C, device=dev) for _ in range(4))
    Dv = torch.empty(A * B, H, S, device=dev)
    _here().oproj(grid, THREADS, _mat(wot, C, C, C), dq1, att, g, o, mod, datt, gated, dO, dG, Dv, dmod, int(S), int(A), int(B), int(SP),
                  int(AT), int(nab), int(nag), int(ntile))
    return datt, gated, dO, dG, Dv


def attn_bwd(qh, kh, vh, dO, lse, Dv, seqused):
    """Window-attention backward (attn_dkv_tf32.cu, attn_dq_tf32.cu): Qh, Kh, Vh head-major [N, H, S, D] fp32, dO [N S, C] fp32, LSE / Dv
    [N, H, S] fp32 -> dQh, dKh, dVh head-major fp32."""
    N, _, S, _ = qh.shape
    rows = N * H * S
    dev = qh.device
    dQh, dKh, dVh = (torch.empty(N, H, S, D, device=dev) for _ in range(3))
    q2, k2, v2 = qh.view(rows, D), kh.view(rows, D), vh.view(rows, D)
    items = N * (S // 128) * H
    grid = (min(nsm(), items), 1, 1)
    K = _here()
    K.dkv(grid, THREADS, _mat(q2, rows, D, 32), _mat(q2, rows, D, 32, "128a32"), _mat(k2, rows, D, 128), _mat(v2, rows, D, 128),
          _mat(dO, N * S, C, 32), _mat(dO, N * S, C, 32, "128a32"), lse, Dv, seqused, dKh, dVh, int(S), int(N), float(D ** -0.5))
    K.dq(grid, THREADS, _mat(q2, rows, D, 128), _mat(dO, N * S, C, 128), _mat(k2, rows, D, 32), _mat(v2, rows, D, 32),
         _mat(k2, rows, D, 32, "128a32"), lse, Dv, seqused, dQh, int(S), int(N), float(D ** -0.5))
    return dQh, dKh, dVh


def qkvg_bwd(q2, dq1, pq, pk, dG, dQh, dKh, dVh, mod, cos, sin, wqkv, wg, A, B, S, dmod, eps=EPS):
    """q | k | v | gate backward (qkvg_bwd_tf32.cu) -> (dq [M, C], dP [M, 4C]); d shift_a / scale_a accumulated into dmod."""
    M = q2.shape[0]
    dev = q2.device
    SP, AT, nab, nag, ntile, grid = _row_grid(A, B, S)
    wt = _cached(_tf32_wt, wqkv, wg)
    dq = torch.empty(M, C, device=dev)
    dP = torch.empty(M, 4 * C, device=dev)
    maps = (_rows(pq, C, A, B, S, SP, AT), _rows(pk, C, A, B, S, SP, AT), _rows(dG, C, A, B, S, SP, AT),
            _heads(dQh, A, B, S, SP, AT), _heads(dKh, A, B, S, SP, AT), _heads(dVh, A, B, S, SP, AT),
            _mat(wt, C, 4 * C, C), _rows(dP, 4 * C, A, B, S, SP, AT))
    _here().qkvg(grid, THREADS, *maps, q2, dq1, mod, cos, sin, dq, dmod, int(S), int(A), int(B), int(SP), int(AT), int(nab), int(nag),
                 int(ntile), float(eps), float(EPS))
    return dq, dP


# --------------------------------------------------------------------------------------------------- the block
def block_bwd_tf32(dy, q, mod, cos, sin, seqused, wqkv, wg, wo, wu, wd, qh, kh, vh, g, o, lse, q1, x, pq, pk, att, y, ffn, B,
                   half_window=HW):
    """The fp32 backward: [dq [N, S, C], dmod [B S, 6C], dWqkv, dWg, dWo, dWu, dWd], all fp32 -- the layout of
    ``dispatch.swa_dit_block_bwd``. Every saved tensor fp32 (module docstring)."""
    assert half_window == HW, "the fp32 sm_100a backward serves the 64-atom half window"
    N, S, _ = q.shape
    assert S % 128 == 0, f"the atom count S = {S} must be a multiple of 128"
    A = N // B
    M = N * S
    dev = q.device
    cg = lambda t: t.contiguous()  # noqa: E731  (no-ops for the forward's own tensors)
    with torch.cuda.device(dev):
        dmod = torch.zeros(B * S, 6 * C, device=dev, dtype=torch.float32)
        mod_, q1_, y_ = cg(mod), cg(q1).view(M, C), cg(y).view(M, C)
        dffn, hh, dab, dq1 = ffn_bwd(cg(dy).view(M, C), q1_, cg(ffn).view(M, C), y_, mod_, wu, wd, A, B, S, dmod)
        datt, gated, dO, dG, Dv = oproj_bwd(dq1, cg(att).view(M, C), cg(g).view(M, C), cg(o).view(M, C), mod_, wo, A, B, S, dmod)
        dQh, dKh, dVh = attn_bwd(cg(qh), cg(kh), cg(vh), dO, cg(lse), Dv, cg(seqused))
        dq, dP = qkvg_bwd(cg(q).view(M, C), dq1, cg(pq).view(M, C), cg(pk).view(M, C), dG, dQh, dKh, dVh, mod_,
                          cg(cos).view(B * S, D // 2), cg(sin).view(B * S, D // 2), wqkv, wg, A, B, S, dmod)
        with _tf32():                                       # the weight gradients: cuBLAS TF32 over the rounded operands
            dwqkvg = dP.t() @ cg(x).view(M, C)              # dWqkv | dWg as one GEMM over dP (x read once)
            dwo = datt.t() @ gated
            dwu = dab.t() @ y_
            dwd = dffn.t() @ hh
    # the op's outputs may not alias each other: dWg (128 x 128) leaves as its own tensor
    return [dq.view(N, S, C), dmod, dwqkvg[:3 * C], dwqkvg[3 * C:].clone(), dwo, dwu, dwd]


def mod_bwd_tf32(g: torch.Tensor, c: torch.Tensor, wmod: torch.Tensor) -> list[torch.Tensor]:
    """[dc [R, C] fp32, dWmod [6C, C] fp32] of mod = silu(c) Wmod^T from g = d mod [R, 6C] fp32; c fp32 [R, C], R a multiple of 128."""
    R = c.shape[0]
    assert R % 128 == 0 and g.shape == (R, 6 * C), (R, tuple(g.shape))
    dev = c.device
    with torch.cuda.device(dev):
        g, c = g.float().contiguous(), c.float().contiguous()
        wmt = _cached(_tf32_wmt, wmod)
        dc = torch.empty(R, C, device=dev)
        sc = torch.empty(R, C, device=dev)
        _here().mod((R // 128, 1, 1), THREADS, _mat(g, R, 6 * C, 128), _mat(wmt, C, 6 * C, C), c, dc, sc)
        with _tf32():
            dw = g.t() @ sc
    return [dc, dw]
