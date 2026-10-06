"""Experimental H100 B200-style TriangleAttention, BF16 activations, Dhead32.

Explicit entry point only: no production dispatch changes. Front and tail provide both WGMMA fusion and grouped cuBLAS paths. All attention and backward helpers are native CUDA;
cuBLAS implements the grouped projection products. No Triton is used here.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

from miniworld_engine.kernels._nvcc import (
    ensure_cuda_home,
    gencodes,
    host_flags,
    load_extension,
)

_SRC = Path(__file__).with_name("h100_sources")
_PARENT = _SRC.parent


@functools.lru_cache(maxsize=None)
def extension(part: str, kv_tile: int = 64, row_group: int = 1, pipeline: int = 0):
    if part not in (
        "module",
        "training_forward",
        "dq",
        "bias_fusion",
        "pack",
        "projection_ln",
        "front_grouped",
    ):
        raise ValueError(part)
    ensure_cuda_home()
    cutlass = Path(os.environ.get("CUTLASS_PATH", ""))
    if not (cutlass / "include/cute/tensor.hpp").is_file():
        raise RuntimeError("CUTLASS_PATH must contain CUTLASS 4.2 headers")
    return load_extension(
        name=(f"triattn_h100_b200style_{part}_v1" + (f"_kv{kv_tile}" if kv_tile != 64 else "")
              + (f"_r{row_group}" if row_group != 1 else "")
              + (f"_ws{pipeline}" if pipeline else "")),
        sources=[str(_SRC / f"{part}.cu")],
        extra_include_paths=[str(_PARENT), str(cutlass / "include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            *host_flags(),
            *gencodes("90a"),
            "-O3",
            "-std=c++17",
            f"-DMW_TRIATTN_KV_TILE={kv_tile}",
            *([f"-DMW_TRIATTN_PIPELINE={pipeline}"] if pipeline else []),
            *([f"-DMW_TRIATTN_ROW_GROUP={row_group}"] if row_group != 1 else []),
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-lineinfo",
            *(["--use_fast_math"] if part == "module" else []),
        ],
        verbose=False,
    )


def supports(pair, weights, lnw, lnb, mask=None):
    if not pair.is_cuda or pair.dtype != torch.bfloat16 or pair.ndim != 4:
        return False
    B, L, S, C = pair.shape
    if (
        B < 1
        or L != S
        or L < 128
        or L > 1024
        or L % 128
        or C not in (64, 128, 256, 384, 512)
    ):
        return False
    H = C // 32
    expected = [(C, C)] * 4 + [(H, C), (C, C)]
    if len(weights) != 6 or any(
        w.device != pair.device
        or w.shape != sh
        or w.dtype not in (torch.bfloat16, torch.float32)
        for w, sh in zip(weights, expected, strict=True)
    ):
        return False
    if any(
        t.device != pair.device
        or t.shape != (C,)
        or t.dtype not in (torch.bfloat16, torch.float32)
        for t in (lnw, lnb)
    ):
        return False
    if mask is not None and (
        mask.device != pair.device or mask.dtype != torch.bool or mask.shape != (B, L)
    ):
        return False
    return torch.cuda.get_device_capability(pair.device) == (9, 0)


def pack_weights(wq, wk, wv, wg, wb, wo):
    return tuple(extension("pack").pack([wq, wk, wv, wg, wb, wo]))


def _view(t, B, L, C):
    return t.view(B, L, L, C // 32, 32).permute(0, 3, 1, 2, 4)


def _rows(t, C):
    return t.permute(0, 2, 3, 1, 4).reshape(-1, C)


def _join(parts):
    return parts[0] if len(parts) == 1 else torch.cat(parts)


def _core_forward(q, k, v, bias):
    core = extension("training_forward", int(os.environ.get("MINIWORLD_TRIATTN_H100_KV_TILE", "64")),
                     int(os.environ.get("MINIWORLD_TRIATTN_H100_ROW_GROUP", "2")),
                     int(os.environ.get("MINIWORLD_TRIATTN_H100_PIPELINE", "0")))
    results = [
        core.forward(q[b : b + 1], k[b : b + 1], v[b : b + 1], bias[b : b + 1])
        for b in range(q.shape[0])
    ]
    return (_join([r[0] for r in results]), _join([r[1] for r in results]))


def _weight_gradient(dy, x):
    """Split the long reduction so skinny FP32-output GEMMs fill the H100."""
    if os.environ.get("MINIWORLD_TRIATTN_H100_SPLIT_WGRAD", "1") == "0":
        return torch.mm(dy.T, x, out_dtype=torch.float32)
    tiles = ((dy.shape[1] + 127) // 128) * ((x.shape[1] + 127) // 128)
    splits = 32 if tiles < 32 else 16 if tiles < 64 else 8
    while dy.shape[0] % splits:
        splits //= 2
    rows = dy.shape[0] // splits
    partial = torch.bmm(
        dy.view(splits, rows, -1).transpose(1, 2),
        x.view(splits, rows, -1),
        out_dtype=torch.float32,
    )
    return extension("pack").reduce_partials(partial)


class H100Triangle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop, packed):
        B, L, _, C = pair.shape
        vector_rows = os.environ.get("MINIWORLD_TRIATTN_H100_ROW_VECTOR", "1") == "1"
        m = extension("module")
        x = pair.contiguous().view(-1, C)
        wall, outweight = packed
        lw, lb = lnw.float().contiguous(), lnb.float().contiguous()
        if os.environ.get("MINIWORLD_TRIATTN_H100_FRONTEND", "grouped") == "grouped":
            f = extension("front_grouped")
            z, stats = f.norm(x, lw, lb, eps, vector_rows)
            projection = z @ wall.T
            qr, kr, vr, gr = (projection[:, j * C : (j + 1) * C] for j in range(4))
            bias = f.bias(projection, mask, C, L)
        else:
            p, bias, stats = m.front(x, lw, lb, wall, mask, L, C // 32, eps,
                                     os.environ.get("MINIWORLD_TRIATTN_H100_FRONTEND") == "fused_reuse")
            qr, kr, vr, gr = p.unbind(0)
        q, k, v = (_view(t, B, L, C) for t in (qr, kr, vr))
        o, lse = _core_forward(q, k, v, bias)
        o_rows = _rows(o, C)
        if os.environ.get("MINIWORLD_TRIATTN_H100_TAIL", "grouped") == "grouped":
            f = extension("front_grouped")
            result = f.residual(x, f.gate(gr, o_rows, vector_rows) @ outweight.T, drop, L, vector_rows).view_as(
                pair
            )
        else:
            result = m.tail(x, gr, o_rows, outweight, drop, L).view_as(pair)
        if any(ctx.needs_input_grad[:9]):
            ctx.save_for_backward(
                x,
                lw,
                lb,
                wall,
                outweight,
                qr,
                kr,
                vr,
                gr,
                bias,
                stats,
                o_rows,
                lse,
                mask,
                drop,
            )
            ctx.dims = (B, L, C)
            ctx.dtypes = tuple(w.dtype for w in (lnw, lnb, wq, wk, wv, wg, wb, wo))
        return result

    @staticmethod
    def backward(ctx, dy):
        x, lw, lb, wall, wo, qr, kr, vr, gr, bias, stats, o, lse, mask, drop = (
            ctx.saved_tensors
        )
        B, L, C = ctx.dims
        m, dqe, dve = extension("module"), extension("dq"), extension("bias_fusion")
        dy = dy.contiguous().view(-1, C)
        dys = m.scale(dy, drop, L)
        du = dys @ wo
        do, dg, u, delta = extension("front_grouped").gate_backward(du, gr, o, L)
        q, k, v = (_view(t, B, L, C) for t in (qr, kr, vr))
        do = _view(do, B, L, C)
        dq, dk, dv, db = [], [], [], []
        for b in range(B):
            args = [
                q[b : b + 1],
                k[b : b + 1],
                v[b : b + 1],
                bias[b : b + 1],
                lse[b : b + 1],
                delta[b : b + 1],
                do[b : b + 1],
            ]
            dq.append(dqe.backward(*args))
            kb, vb, bb = dve.backward(*args, 8)
            dk.append(kb)
            dv.append(vb)
            db.append(bb)
        dq, dk, dv = (_rows(_join(ts), C) for ts in (dq, dk, dv))
        dall = extension("front_grouped").pack_backward(
            dq, dk, dv, dg, _join(db), mask, L
        )
        # One grouped projection product instead of five independent dgrad GEMMs.
        if os.environ.get("MINIWORLD_TRIATTN_H100_PROJECT_LN", "0") == "1":
            dx, z, parts = extension("projection_ln").backward(
                x, dall, wall, dy, stats, lw, lb
            )
        else:
            dz = torch.mm(dall, wall, out_dtype=torch.float32)
            dx, z, parts = m.ln_backward(x, dz, dy, stats, lw, lb)
        dnorm = extension("pack").reduce_partials(parts)
        grad = _weight_gradient(dall, z)
        dwo = _weight_gradient(dys, u)
        params = extension("pack").finish(
            grad, dwo, dnorm, [dt == torch.float32 for dt in ctx.dtypes]
        )
        return (dx.view(B, L, L, C), *params, None, None, None, None)


def triangle_attention(
    pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask=None, eps=1e-5, drop=None, packed=None
):
    weights = (wq, wk, wv, wg, wb, wo)
    if not supports(pair, weights, lnw, lnb, mask):
        raise ValueError(
            "H100 native candidate requires square BF16 buckets, C64/128/256/384/512, Dhead32, valid parameters"
        )
    if packed is None:
        packed = pack_weights(*weights)
    mask = (
        mask.contiguous()
        if mask is not None
        else torch.empty(0, device=pair.device, dtype=torch.bool)
    )
    if drop is None:
        drop = pair.new_empty(0)
    return H100Triangle.apply(
        pair, lnw, lnb, *weights, mask, eps, drop.contiguous(), packed
    )


def forward(module, pair, mask=None):
    """Explicit candidate module call with starting/ending and broadcast dropout."""
    if (
        not module.use_self_attention
        or module.use_qk_norm
        or module.to_query.weight.shape[0] != module.n_head * 32
    ):
        raise ValueError(
            "native candidate requires self-attention without QK normalization, Dhead32"
        )
    B, L, _, C = pair.shape
    drop = (
        extension("pack").drop_scale(pair, module.p_drop)
        if module.training and module.p_drop
        else None
    )
    x = pair if module.starting else pair.transpose(1, 2)
    weights = (
        module.to_query.weight,
        module.to_key.weight,
        module.to_value.weight,
        module.to_gate.weight,
        module.to_bias.weight,
        module.to_out.weight,
    )
    from miniworld_engine.kernels import _capture

    key = _capture.scoped(tuple((w.data_ptr(), w._version) for w in weights))
    cached = getattr(module, "_h100_b200style_pack", None)
    if key is None or cached is None or key != cached[0]:
        with torch.no_grad():
            cached = (key, pack_weights(*weights))
        module._h100_b200style_pack = cached
    out = triangle_attention(
        x.contiguous(),
        module.ln_pair.weight,
        module.ln_pair.bias,
        *weights,
        mask,
        module.ln_pair.eps,
        drop,
        cached[1],
    )
    return out if module.starting else out.transpose(1, 2).contiguous()
