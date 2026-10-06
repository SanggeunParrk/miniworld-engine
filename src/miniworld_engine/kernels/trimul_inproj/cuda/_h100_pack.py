"""Live weight packing without a nested torch.compile runtime invocation."""
import os
import triton
import triton.language as tl
import torch
from . import _h100_runtime as T
from miniworld_engine.kernels._compile import opaque


@triton.jit
def _pack(DST, WL, WLG, WR, WRG, R: tl.constexpr, D: tl.constexpr,
          LS0: tl.constexpr, LS1: tl.constexpr,
          LGS0: tl.constexpr, LGS1: tl.constexpr,
          RS0: tl.constexpr, RS1: tl.constexpr,
          RGS0: tl.constexpr, RGS1: tl.constexpr,
          BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // D, i % D
    source_row = (row // 64) * 32 + row % 32
    right = source_row >= R
    gate = row % 64 < 32
    r = source_row % R
    ptr = tl.where(right,
                   tl.where(gate, WRG + r * RGS0 + col * RGS1,
                            WR + r * RS0 + col * RS1),
                   tl.where(gate, WLG + r * LGS0 + col * LGS1,
                            WL + r * LS0 + col * LS1))
    tl.store(DST + i, tl.load(ptr, i < 4 * R * D, other=0), i < 4 * R * D)


def pack_into_triton(dst, wl, wlg, wr, wrg):
    r, d = wl.shape
    if r % 32 or dst.shape != (4 * r, d) or not dst.is_contiguous():
        raise ValueError("weight pack requires 32-row groups and contiguous output")
    if any(w.shape != wl.shape or w.dtype != dst.dtype or w.device != dst.device
           for w in (wl, wlg, wr, wrg)):
        raise ValueError("weight pack inputs must share shape, dtype and device")
    _pack[(triton.cdiv(4 * r * d, 1024),)](
        dst, wl, wlg, wr, wrg, r, d,
        *wl.stride(), *wlg.stride(), *wr.stride(), *wrg.stride(),
        BLOCK=1024, num_warps=4)


@T.device_cache
def _aux_unit():
    path = T.compile(T.SOURCES / "wide_forward/output.cu", [
        "-std=c++17", "-O3", "-arch=sm_90a", "--cubin", "-lineinfo", "-DMW_AUX_ONLY=1"])
    return T.load_unit(str(path), "trimul_live_aux")


def _launch(name, grid, args):
    _aux_unit().kernel(name).launch(grid, (256, 1, 1), args)


def pack_into_cuda(dst, wl, wlg, wr, wrg):
    r, d = wl.shape
    quad = (wl, wlg, wr, wrg)
    if (r % 32 or dst.shape != (4 * r, d) or not dst.is_contiguous()
            or dst.dtype != torch.bfloat16
            or any(w.shape != wl.shape or w.device != dst.device
                   or w.dtype not in (torch.bfloat16, torch.float32) for w in quad)):
        raise ValueError("CUDA weight packing requires BF16 output and BF16/FP32 leaves")
    p = T._launch_module().Struct([
        dst, *quad, r, d, *(w.stride(0) for w in quad), *(w.stride(1) for w in quad),
        *(int(w.dtype == torch.float32) for w in quad)])
    _launch("mw_aux_pack", ((4*r*d+255)//256, 1, 1), [p])


def pack_into(dst, wl, wlg, wr, wrg):
    if T.cuda_aux_enabled():
        return pack_into_cuda(dst, wl, wlg, wr, wrg)
    return pack_into_triton(dst, wl, wlg, wr, wrg)


def cast_weights(weights):
    """One live CUDA launch for the six matrix leaves; BF16 layouts stay aliased."""
    out = [w if w.dtype == torch.bfloat16 and (i < 4 or w.is_contiguous())
           else torch.empty(w.shape, device=w.device, dtype=torch.bfloat16)
           for i, w in enumerate(weights)]
    counts = [0 if a is b else a.numel() for a, b in zip(weights, out)]
    if max(counts):
        p = T._launch_module().Struct([
            *weights, *out, *counts, *(w.shape[1] for w in weights),
            *(w.stride(0) for w in weights), *(w.stride(1) for w in weights),
            *(int(w.dtype == torch.float32) for w in weights)])
        _launch("mw_aux_cast", ((max(counts)+255)//256, 6, 1), [p])
    return out


def normalize_into(dst, x, g, b, stats=None):
    if not x.is_contiguous() or x.dtype != torch.bfloat16 or g.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("CUDA input LN requires contiguous BF16 input and FP32 affine")
    width, rows = x.shape[-1], x.numel()//x.shape[-1]
    if width not in (64, 128, 256, 384, 512):
        raise ValueError("Hopper CUDA normalization requires a supported pair width")
    if os.environ.get("MINIWORLD_TRIMUL_H100_WARP_LN") == "1":
        _launch("mw_aux_norm_warp", ((rows+7)//8, 1, 1), [x, g, b, dst, stats, width, rows])
    else:
        _launch("mw_aux_norm", (rows, 1, 1), [x, g, b, dst, stats, width])


def fold_into(wp, go, bo, wpf, wg, gi, bi, wgf, uv, gate=False):
    d, hidden = wp.shape
    _launch("mw_aux_fold", (d, 2 if gate else 1, 1), [wp, go, bo, wpf, wg, gi, bi, wgf, uv, d, hidden])


def pair_mask(token, length, device):
    out = torch.empty((length, length), device=device, dtype=torch.bfloat16)
    _launch("mw_aux_mask", ((length*length+255)//256, 1, 1), [token, out, length])
    return out


def owned_bf16_mask(mask, length, device):
    """Read live token/pair masks into owned K1/contraction storage without ATen."""
    out = torch.empty((length, length), device=device, dtype=torch.bfloat16)
    absent = mask is None or mask.numel() == 0
    if not absent and (mask.device != torch.device(device) or not mask.is_contiguous() or mask.dtype not in
                       (torch.bool, torch.bfloat16, torch.float32)
                       or mask.numel() not in (length, length * length)):
        raise ValueError("native TriMul mask requires contiguous token or pair data")
    kind = 0 if absent else 1 if mask.dtype == torch.bool else 2 if mask.dtype == torch.bfloat16 else 3
    _launch("mw_aux_owned_mask", ((length*length+255)//256, 1, 1),
            [None if absent else mask, out, length, kind,
             int(not absent and mask.numel() == length)])
    return out


def _gradient_layout_fake(grads, weights):
    return [torch.empty_strided(w.shape, w.stride(), device=w.device, dtype=w.dtype) for w in weights]


@opaque(fake=_gradient_layout_fake, name="trimul_h100_parameter_gradient_layout")
def _gradient_layout(grads: list[torch.Tensor], weights: list[torch.Tensor]) -> list[torch.Tensor]:
    if not 0 < len(grads) == len(weights) <= 6:
        raise ValueError("one to six parameter gradients required")
    if any(not g.is_contiguous() or g.shape != w.shape or g.device != w.device
           or g.dtype != w.dtype or w.dtype not in (torch.bfloat16, torch.float32)
           or not (w.is_contiguous() or w.t().is_contiguous())
           for g, w in zip(grads, weights, strict=True)):
        raise ValueError("native gradient layout requires matching dense matrices")
    out = _gradient_layout_fake(grads, weights)
    pad = lambda values, fill=0: list(values) + [fill] * (6-len(grads))
    with T.native_context(grads[0].device):
        p = T._launch_module().Struct([
            *pad(grads, None), *pad(out, None),
            *pad(w.shape[0] for w in weights), *pad(w.shape[1] for w in weights),
            *pad(w.stride(0) for w in weights), *pad(w.stride(1) for w in weights),
            *pad(int(w.dtype == torch.float32) for w in weights)])
        _launch("mw_aux_grad_layout", ((max(g.numel() for g in grads)+255)//256,len(grads),1), [p])
    return out


def preserve_parameter_layout(grads, weights):
    indices = [i for i,(g,w) in enumerate(zip(grads,weights,strict=True)) if g.stride() != w.stride()]
    if not indices:
        return grads
    copies = _gradient_layout([grads[i] for i in indices], [weights[i] for i in indices])
    result = list(grads)
    for i,g in zip(indices,copies,strict=True):
        result[i] = g
    return result


def stack_backward_weights(weights):
    """Small live [gate,left,left_gate,right,right_gate] matrix stack."""
    width = weights[0].shape[1]
    out = weights[0].new_empty((sum(w.shape[0] for w in weights), width))
    p = T._launch_module().Struct([
        *weights, out, *(w.shape[0] for w in weights),
        *(w.stride(0) for w in weights), *(w.stride(1) for w in weights), width])
    _launch("mw_aux_stack", ((max(w.numel() for w in weights)+255)//256, 5, 1), [p])
    return out


def deterministic_ln_backward(x, dy, gamma, *, planes=False, stats=None, residual=None):
    """Native fixed-order dx/dgamma/dbeta; no atomic reduction or gamma division."""
    width, rows = gamma.numel(), x.numel() // gamma.numel()
    dx = torch.empty_like(x)
    mu, rs = stats if stats is not None else (
        torch.empty(rows, device=x.device, dtype=torch.float32),
        torch.empty(rows, device=x.device, dtype=torch.float32))
    _launch("mw_aux_ln_backward", (rows, 1, 1),
            [x, dy, gamma, residual, dx, mu, rs, rows, width, int(planes), int(stats is not None)])
    blocks = (rows+4095)//4096
    partial = torch.empty((blocks, 2, width), device=x.device, dtype=torch.float32)
    dg, db = torch.empty_like(gamma), torch.empty_like(gamma)
    _launch("mw_aux_ln_affine_partial", (width, blocks, 1),
            [x, dy, mu, rs, partial, rows, width, int(planes)])
    _launch("mw_aux_ln_affine_finish", ((width+255)//256, 1, 1), [partial, dg, db, width, blocks])
    return dx, dg, db


def training_pair_mask(mask, length, device):
    """Owned FP32 pair mask, built directly from token/pair/absent masks."""
    out = torch.empty((length, length), device=device, dtype=torch.float32)
    absent = mask is None or mask.numel() == 0
    if not absent and (not mask.is_contiguous() or mask.dtype not in
                       (torch.bool, torch.bfloat16, torch.float32)
                       or mask.numel() not in (length, length * length)):
        raise ValueError("native TriMul mask requires contiguous token or square pair data")
    kind = 0 if absent else 1 if mask.dtype == torch.bool else 2 if mask.dtype == torch.bfloat16 else 3
    token = int(not absent and mask.numel() == length)
    _launch("mw_aux_train_mask", ((length*length+255)//256, 1, 1),
            [None if absent else mask, out, length, kind, token])
    return out


def _training_unit_scale_fake(pair):
    return torch.empty((pair.shape[1], pair.shape[-1]), device=pair.device, dtype=torch.bfloat16)


@opaque(fake=_training_unit_scale_fake, name="trimul_h100_training_unit_scale")
def _training_unit_scale(pair: torch.Tensor) -> torch.Tensor:
    out = torch.empty((pair.shape[1], pair.shape[-1]), device=pair.device, dtype=torch.bfloat16)
    _launch("mw_aux_unit_scale", ((out.numel()+255)//256, 1, 1), [out, out.numel()])
    return out


def training_unit_scale(pair):
    # The pair supplies shape/device only; the constant scale has no derivative.
    with torch.no_grad():
        return _training_unit_scale(pair)


def _fused_training_prepare_fake(leaves, mask):
    x, weight = leaves[:2]
    length, width, hidden = x.shape[1], x.shape[-1], weight.shape[0]
    return (x.new_empty((4*hidden, width)),
            x.new_empty((length, length), dtype=torch.float32),
            x.new_empty((width, width)), x.new_empty((width, hidden)))


@opaque(fake=_fused_training_prepare_fake, name="trimul_h100_d64_training_prepare")
def fused_training_prepare(leaves: list[torch.Tensor], mask: torch.Tensor | None
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One live CUDA launch for D64 K1 packing, output weights and pair mask."""
    x, *weights = leaves[:7]
    length, width = x.shape[1], x.shape[-1]
    hidden = weights[0].shape[0]
    if width != 64 or hidden not in (64, 128):
        raise ValueError("fused training preparation requires D64/H64 or H128")
    if any(w.dtype not in (torch.bfloat16, torch.float32) for w in weights):
        raise ValueError("fused training preparation requires BF16/FP32 matrices")
    absent = mask is None or mask.numel() == 0
    if not absent and (not mask.is_contiguous() or mask.dtype not in
                       (torch.bool, torch.bfloat16, torch.float32)
                       or mask.numel() not in (length, length * length)):
        raise ValueError("native TriMul mask requires contiguous token or square pair data")
    kind = 0 if absent else 1 if mask.dtype == torch.bool else 2 if mask.dtype == torch.bfloat16 else 3
    token = int(not absent and mask.numel() == length)
    packed = torch.empty((4*hidden, width), device=x.device, dtype=torch.bfloat16)
    gate = torch.empty((width, width), device=x.device, dtype=torch.bfloat16)
    output = torch.empty((width, hidden), device=x.device, dtype=torch.bfloat16)
    pair_mask = torch.empty((length, length), device=x.device, dtype=torch.float32)
    p = T._launch_module().Struct([
        *weights, packed, gate, output, None if absent else mask, pair_mask,
        *(w.stride(0) for w in weights), *(w.stride(1) for w in weights),
        *(int(w.dtype == torch.float32) for w in weights), hidden, length, kind, token])
    _launch("mw_aux_d64_training_prepare", ((max(length*length,4*hidden*width)+255)//256,1,1), [p])
    return packed, pair_mask, gate, output
