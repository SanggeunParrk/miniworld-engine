"""Experimental H100 B200-style TriangleAttention, BF16 activations, Dhead32.

Explicit entry point only: no production dispatch changes. Front and tail provide both WGMMA fusion and grouped cuBLAS paths. All attention and backward helpers are native CUDA;
cuBLAS implements the grouped projection products. No Triton is used here.
"""

from __future__ import annotations

import functools
import os
import threading
from contextlib import contextmanager
from pathlib import Path

import torch

from miniworld_engine.kernels import _capture
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels._nvcc import (
    ensure_cuda_home,
    gencodes,
    host_flags,
    load_extension,
)

_SRC = Path(__file__).with_name("h100_sources")
_PARENT = _SRC.parent

# Retain each forward's resolved workspace policy for its later backward.
_OWNED_BLAS_STATE = threading.local()


def _owned_blas_enabled():
    selected = getattr(_OWNED_BLAS_STATE, "enabled", None)
    if selected is not None:
        return selected
    return os.environ.get("MINIWORLD_TRIATTN_H100_OWNED_BLAS", "0") == "1"


def _owned_blas_for_module(module, pair):
    override = os.environ.get("MINIWORLD_TRIATTN_H100_OWNED_BLAS")
    if override is not None:
        return override == "1"
    B, L, _, C = pair.shape
    return (module.training and torch.is_grad_enabled() and module.starting
            and B == 1 and (C, L) in ((128, 768), (512, 384)))


@contextmanager
def _owned_blas_scope(enabled):
    previous = getattr(_OWNED_BLAS_STATE, "enabled", None)
    _OWNED_BLAS_STATE.enabled = enabled
    try:
        yield
    finally:
        _OWNED_BLAS_STATE.enabled = previous



def _variant_flags(part):
    # Experiment hook: MW_F2_DEFS="-DMW_F2_STAGES_FORCE=3 -DMW_F2_MINB_FORCE=1" and MW_F2_PTXAS=1.
    if part not in ("front_fused2", "front_fused3", "training_forward"):
        return []
    return [*os.environ.get("MW_F2_DEFS", "").split(), *(["-Xptxas=-v"] if os.environ.get("MW_F2_PTXAS") else [])]


def _variant_suffix(part):
    if part not in ("front_fused2", "front_fused3", "training_forward"):
        return ""
    return "".join(c for c in os.environ.get("MW_F2_DEFS", "") + ("p" if os.environ.get("MW_F2_PTXAS") else "") if c.isalnum())


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
        "front_fused2",
        "front_fused3",
        "tail_fused3",
    ):
        raise ValueError(part)
    ensure_cuda_home()
    cutlass = Path(os.environ.get("CUTLASS_PATH", ""))
    if not (cutlass / "include/cute/tensor.hpp").is_file():
        raise RuntimeError("CUTLASS_PATH must contain CUTLASS 4.2 headers")
    return load_extension(
        name=(f"triattn_h100_b200style_{part}_v1" + (f"_kv{kv_tile}" if kv_tile != 64 else "")
              + (f"_r{row_group}" if row_group != 1 else "")
              + (f"_ws{pipeline}" if pipeline else "")
              + _variant_suffix(part)),
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
            *_variant_flags(part),
        ],
        verbose=bool(os.environ.get("MW_F2_PTXAS")) and part in ("front_fused2", "front_fused3", "training_forward"),
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
        return _mm(dy.T, x, out_dtype=torch.float32)
    tiles = ((dy.shape[1] + 127) // 128) * ((x.shape[1] + 127) // 128)
    splits = 32 if tiles < 32 else 16 if tiles < 64 else 8
    while dy.shape[0] % splits:
        splits //= 2
    rows = dy.shape[0] // splits
    partial = _bmm(
        dy.view(splits, rows, -1).transpose(1, 2),
        x.view(splits, rows, -1),
        out_dtype=torch.float32,
    )
    return extension("pack").reduce_partials(partial)


def _frontend(C):
    """Qualified default front end: the one-kernel LN + projection front wins at C128 (64-row tiles) and C256 (128-row
    tiles) (docs/records/triattn_fuse_20261008.md); wider pairs keep the cuBLAS front."""
    return os.environ.get("MINIWORLD_TRIATTN_H100_FRONTEND") or (
        "fused2" if C == 128 else "fused3" if C == 256 else "grouped"
    )


def _resolve_pack(weights, wall, outweight):
    """The (wall, outweight) pack: the caller's, or the weights' own -- cached while they are unmodified, scoped to a capture."""
    if wall is not None and outweight is not None:
        return wall, outweight
    return _capture.param_forms("triattn_h100_pack", weights, lambda: pack_weights(*weights))


def _forward_op_fake(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, drop, wall, outweight, eps, fused_tail, owned_blas):
    """result [B, L, L, C]; proj [rows, 4C] (cuBLAS front end) or [4, rows, C] (fused front ends); bias [B, H, L, L];
    stats [rows, 2] fp32; o_rows [rows, C]; lse [B, H, L, L] fp32. All in pair's dtype unless noted; rows = B L L, H = C / 32."""
    B, L, _, C = pair.shape
    rows, H = B * L * L, C // 32
    proj_shape = (rows, 4 * C) if _frontend(C) == "grouped" else (4, rows, C)
    return (
        pair.new_empty(pair.shape),
        pair.new_empty(proj_shape),
        pair.new_empty((B, H, L, L)),
        pair.new_empty((rows, 2), dtype=torch.float32),
        pair.new_empty((rows, C)),
        pair.new_empty((B, H, L, L), dtype=torch.float32),
    )


@opaque(fake=_forward_op_fake, name="triangle_attention_h100_forward_cuda")
def _forward_op(
    pair: torch.Tensor, lnw: torch.Tensor, lnb: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor,
    wg: torch.Tensor, wb: torch.Tensor, wo: torch.Tensor, mask: torch.Tensor, drop: torch.Tensor,
    wall: torch.Tensor | None, outweight: torch.Tensor | None, eps: float, fused_tail: bool, owned_blas: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Native H100 TriangleAttention forward (front end, core, tail); also returns what the backward needs."""
    with _owned_blas_scope(owned_blas):
        B, L, _, C = pair.shape
        vector_rows = os.environ.get("MINIWORLD_TRIATTN_H100_ROW_VECTOR", "1") == "1"
        m = extension("module")
        x = pair.contiguous().view(-1, C)
        wall, outweight = _resolve_pack((wq, wk, wv, wg, wb, wo), wall, outweight)
        lw, lb = lnw.float().contiguous(), lnb.float().contiguous()
        frontend = _frontend(C)
        if frontend == "grouped":
            f = extension("front_grouped")
            z, stats = f.norm(x, lw, lb, eps, vector_rows)
            proj = _mm(z, wall.T)
            qr, kr, vr, gr = (proj[:, j * C : (j + 1) * C] for j in range(4))
            bias = f.bias(proj, mask, C, L)
        elif frontend == "fused3":
            proj, bias, stats = extension("front_fused3").front(x, lw, lb, wall, mask, L, C // 32, eps)
            qr, kr, vr, gr = proj.unbind(0)
        elif frontend == "fused2":
            proj, bias, stats = extension("front_fused2").front(x, lw, lb, wall, mask, L, C // 32, eps)
            qr, kr, vr, gr = proj.unbind(0)
        else:
            proj, bias, stats = m.front(x, lw, lb, wall, mask, L, C // 32, eps, frontend == "fused_reuse")
            qr, kr, vr, gr = proj.unbind(0)
        q, k, v = (_view(t, B, L, C) for t in (qr, kr, vr))
        o, lse = _core_forward(q, k, v, bias)
        o_rows = _rows(o, C)
        tail = os.environ.get("MINIWORLD_TRIATTN_H100_TAIL", "fused" if fused_tail else "grouped")
        if tail == "grouped":
            f = extension("front_grouped")
            result = f.residual(x, _mm(f.gate(gr, o_rows, vector_rows), outweight.T), drop, L, vector_rows).view_as(pair)
        elif tail == "fused3":
            result = extension("tail_fused3").tail(x, gr, o_rows, outweight, drop, L).view_as(pair)
        else:
            result = m.tail(x, gr, o_rows, outweight, drop, L).view_as(pair)
        return result, proj, bias, stats, o_rows, lse


def _backward_op_fake(dy, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, drop, wall, outweight, proj, bias, stats, o_rows, lse,
                   owned_blas):
    """dx [B, L, L, C] in pair's dtype, then the gradients of the ln weight, ln bias, wq, wk, wv, wg, wb and wo, each shaped
    and typed like its parameter."""
    return (
        pair.new_empty(pair.shape),
        lnw.new_empty(lnw.shape),
        lnb.new_empty(lnb.shape),
        wq.new_empty(wq.shape),
        wk.new_empty(wk.shape),
        wv.new_empty(wv.shape),
        wg.new_empty(wg.shape),
        wb.new_empty(wb.shape),
        wo.new_empty(wo.shape),
    )


@opaque(fake=_backward_op_fake, name="triangle_attention_h100_backward_cuda")
def _backward_op(
    dy: torch.Tensor, pair: torch.Tensor, lnw: torch.Tensor, lnb: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor,
    wv: torch.Tensor, wg: torch.Tensor, wb: torch.Tensor, wo: torch.Tensor, mask: torch.Tensor, drop: torch.Tensor,
    wall: torch.Tensor | None, outweight: torch.Tensor | None, proj: torch.Tensor, bias: torch.Tensor, stats: torch.Tensor,
    o_rows: torch.Tensor, lse: torch.Tensor, owned_blas: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """Native H100 TriangleAttention backward: the input gradient and the eight parameter gradients (parameters' dtypes)."""
    with _owned_blas_scope(owned_blas):
        B, L, _, C = pair.shape
        x = pair.contiguous().view(-1, C)
        wall, wo_packed = _resolve_pack((wq, wk, wv, wg, wb, wo), wall, outweight)
        lw, lb = lnw.float().contiguous(), lnb.float().contiguous()
        if proj.ndim == 2:
            qr, kr, vr, gr = (proj[:, j * C : (j + 1) * C] for j in range(4))
        else:
            qr, kr, vr, gr = proj.unbind(0)
        dtypes = tuple(w.dtype for w in (lnw, lnb, wq, wk, wv, wg, wb, wo))
        m, dqe, dve = extension("module"), extension("dq"), extension("bias_fusion")
        dy = dy.contiguous().view(-1, C)
        dys = m.scale(dy, drop, L)
        du = _mm(dys, wo_packed)
        do, dg, u, delta = extension("front_grouped").gate_backward(du, gr, o_rows, L)
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
        dall = extension("front_grouped").pack_backward(dq, dk, dv, dg, _join(db), mask, L)
        # One grouped projection product instead of five independent dgrad GEMMs.
        if os.environ.get("MINIWORLD_TRIATTN_H100_PROJECT_LN", "0") == "1":
            dx, z, parts = extension("projection_ln").backward(x, dall, wall, dy, stats, lw, lb)
        else:
            dz = _mm(dall, wall, out_dtype=torch.float32)
            dx, z, parts = m.ln_backward(x, dz, dy, stats, lw, lb)
        dnorm = extension("pack").reduce_partials(parts)
        grad = _weight_gradient(dall, z)
        dwo = _weight_gradient(dys, u)
        p = extension("pack").finish(grad, dwo, dnorm, [dt == torch.float32 for dt in dtypes])
        return dx.view(B, L, L, C), p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]


class H100Triangle(torch.autograd.Function):
    """Autograd owner of the native launches: ``ctx`` lives here, the launches are the opaque ops above."""

    @staticmethod
    def forward(ctx, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, eps, drop, wall, outweight, fused_tail, owned_blas):
        result, proj, bias, stats, o_rows, lse = _forward_op(
            pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, drop, wall, outweight, eps, fused_tail, owned_blas
        )
        if any(ctx.needs_input_grad[:9]):
            ctx.save_for_backward(pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, drop, wall, outweight, proj, bias, stats,
                                  o_rows, lse)
            ctx.owned_blas = owned_blas
        return result

    @staticmethod
    def backward(ctx, dy):
        (pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, drop, wall, outweight, proj, bias, stats, o_rows,
         lse) = ctx.saved_tensors
        grads = _backward_op(dy, pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask, drop, wall, outweight, proj, bias, stats,
                             o_rows, lse, ctx.owned_blas)
        return (*grads, None, None, None, None, None, None, None)


def _drop_scale_fake(pair: torch.Tensor, probability: float) -> torch.Tensor:
    """The broadcast dropout scale [B, L, C] in pair's dtype."""
    return pair.new_empty((pair.shape[0], pair.shape[1], pair.shape[3]))


@opaque(fake=_drop_scale_fake, name="triangle_attention_h100_drop_scale_cuda")
def _drop_scale(pair: torch.Tensor, probability: float) -> torch.Tensor:
    """Row-broadcast dropout scale 0 / 1/(1-p), drawn from the default CUDA generator (graph-capturable)."""
    return extension("pack").drop_scale(pair, probability)


def triangle_attention(
    pair, lnw, lnb, wq, wk, wv, wg, wb, wo, mask=None, eps=1e-5, drop=None, packed=None,
    fused_tail=False, owned_blas=None,
):
    """``packed`` (the weights' ``pack_weights``) and ``owned_blas`` (None: the ambient scope / env) are optional; without a
    pack the op packs the weights itself, cached while they are unmodified."""
    weights = (wq, wk, wv, wg, wb, wo)
    if not supports(pair, weights, lnw, lnb, mask):
        raise ValueError(
            "H100 native candidate requires square BF16 buckets, C64/128/256/384/512, Dhead32, valid parameters"
        )
    wall, outweight = packed if packed is not None else (None, None)
    mask = (
        mask.contiguous()
        if mask is not None
        else torch.empty(0, device=pair.device, dtype=torch.bool)
    )
    if drop is None:
        drop = pair.new_empty(0)
    if owned_blas is None:
        owned_blas = _owned_blas_enabled()
    return H100Triangle.apply(
        pair, lnw, lnb, *weights, mask, eps, drop.contiguous(), wall, outweight, fused_tail, owned_blas
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
    # Fused tail: inference, B1, starting node, C128/256 (module sweep 2026-10-08). Evaluate grad
    # mode here: autograd.Function.forward itself always runs with grads disabled.
    fused_tail = (not torch.is_grad_enabled() and not module.training and module.starting
                  and B == 1 and C in (128, 256))
    # detached: the scale is a constant of the step, not a function of the pair (the op has no autograd formula)
    drop = _drop_scale(pair.detach(), module.p_drop) if module.training and module.p_drop else None
    x = pair if module.starting else pair.transpose(1, 2)
    weights = (
        module.to_query.weight,
        module.to_key.weight,
        module.to_value.weight,
        module.to_gate.weight,
        module.to_bias.weight,
        module.to_out.weight,
    )
    # The weight pack is made (and cached, scoped to a CUDA-graph capture) inside the forward / backward ops.
    out = triangle_attention(
        x.contiguous(),
        module.ln_pair.weight,
        module.ln_pair.bias,
        *weights,
        mask,
        module.ln_pair.eps,
        drop,
        None,
        fused_tail=fused_tail,
        owned_blas=_owned_blas_for_module(module, pair),
    )
    return out if module.starting else out.transpose(1, 2).contiguous()


# Opt-in graph-owned scratch; all CUDA arithmetic and cuBLAS layouts are retained.
_OWNED_BLAS_CPP = r"""#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <map>
#include <memory>
struct TokenHandle { cublasHandle_t value; TokenHandle(){TORCH_CHECK(cublasCreate(&value)==CUBLAS_STATUS_SUCCESS,"Token cublasCreate");} ~TokenHandle(){cublasDestroy(value);} };
torch::Tensor owned_mm(torch::Tensor a,torch::Tensor b,bool tf32,bool fp32_output) {
 TORCH_CHECK(a.is_cuda()&&b.is_cuda()&&a.device()==b.device()&&a.dim()==2&&b.dim()==2&&a.size(1)==b.size(0)&&a.scalar_type()==b.scalar_type()&&(a.scalar_type()==at::kFloat||a.scalar_type()==at::kBFloat16),"Token owned GEMM operands");
 c10::cuda::CUDAGuard guard(a.device());
 static thread_local std::map<int,std::unique_ptr<TokenHandle>> handles;
 auto& owned=handles[a.get_device()];if(!owned)owned=std::make_unique<TokenHandle>();
 auto ck=[](cublasStatus_t s){TORCH_CHECK(s==CUBLAS_STATUS_SUCCESS,"Token private cuBLAS status ",int(s));};
 auto handle=owned->value;auto stream=at::cuda::getCurrentCUDAStream();
 ck(cublasSetStream(handle,stream));ck(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));ck(cublasSetAtomicsMode(handle,CUBLAS_ATOMICS_NOT_ALLOWED));
 const bool fp32=a.scalar_type()==at::kFloat;
 ck(cublasSetMathMode(handle,fp32&&tf32?CUBLAS_TF32_TENSOR_OP_MATH:CUBLAS_DEFAULT_MATH));
 const size_t bytes=at::cuda::getChosenWorkspaceSize();auto workspace=torch::empty({int64_t(bytes)},a.options().dtype(at::kByte));
 ck(cublasSetWorkspace(handle,workspace.data_ptr(),bytes));
 const bool output_float=fp32||fp32_output;
 auto out=torch::empty({a.size(0),b.size(1)},a.options().dtype(output_float?at::kFloat:at::kBFloat16));
 auto leading=[](torch::Tensor t){bool row=t.stride(1)==1;TORCH_CHECK(row?t.stride(0)>=t.size(1):(t.stride(0)==1&&t.stride(1)>=t.size(0)),"Token dense matrix view");return int(row?t.stride(0):t.stride(1));};
 int lda=leading(a),ldb=leading(b),m=a.size(0),n=b.size(1),k=a.size(1);float alpha=1,beta=0;
 const auto dtype=fp32?CUDA_R_32F:CUDA_R_16BF;const auto compute=fp32&&tf32?CUBLAS_COMPUTE_32F_FAST_TF32:CUBLAS_COMPUTE_32F;
 ck(cublasGemmEx(handle,b.stride(1)==1?CUBLAS_OP_N:CUBLAS_OP_T,a.stride(1)==1?CUBLAS_OP_N:CUBLAS_OP_T,n,m,k,&alpha,b.data_ptr(),dtype,ldb,a.data_ptr(),dtype,lda,&beta,out.data_ptr(),output_float?CUDA_R_32F:CUDA_R_16BF,n,compute,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
 return out;
}

torch::Tensor owned_mm_out(torch::Tensor a,torch::Tensor b,torch::Tensor out,bool tf32) {
 TORCH_CHECK(a.is_cuda()&&b.is_cuda()&&a.device()==b.device()&&a.dim()==2&&b.dim()==2&&a.size(1)==b.size(0)&&a.scalar_type()==b.scalar_type()&&(a.scalar_type()==at::kFloat||a.scalar_type()==at::kBFloat16),"Token owned GEMM operands");
 c10::cuda::CUDAGuard guard(a.device());
 static thread_local std::map<int,std::unique_ptr<TokenHandle>> handles;
 auto& owned=handles[a.get_device()];if(!owned)owned=std::make_unique<TokenHandle>();
 auto ck=[](cublasStatus_t s){TORCH_CHECK(s==CUBLAS_STATUS_SUCCESS,"Token private cuBLAS status ",int(s));};
 auto handle=owned->value;auto stream=at::cuda::getCurrentCUDAStream();
 ck(cublasSetStream(handle,stream));ck(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));ck(cublasSetAtomicsMode(handle,CUBLAS_ATOMICS_NOT_ALLOWED));
 const bool fp32=a.scalar_type()==at::kFloat;
 ck(cublasSetMathMode(handle,fp32&&tf32?CUBLAS_TF32_TENSOR_OP_MATH:CUBLAS_DEFAULT_MATH));
 const size_t bytes=at::cuda::getChosenWorkspaceSize();auto workspace=torch::empty({int64_t(bytes)},a.options().dtype(at::kByte));
 ck(cublasSetWorkspace(handle,workspace.data_ptr(),bytes));
 const bool output_float=out.scalar_type()==at::kFloat;
 TORCH_CHECK(out.is_cuda()&&out.device()==a.device()&&out.dim()==2&&out.size(0)==a.size(0)&&out.size(1)==b.size(1)&&(out.scalar_type()==at::kFloat||out.scalar_type()==at::kBFloat16),"TriMul owned GEMM output");
 auto leading=[](torch::Tensor t){bool row=t.stride(1)==1;TORCH_CHECK(row?t.stride(0)>=t.size(1):(t.stride(0)==1&&t.stride(1)>=t.size(0)),"Token dense matrix view");return int(row?t.stride(0):t.stride(1));};
 int lda=leading(a),ldb=leading(b),m=a.size(0),n=b.size(1),k=a.size(1);float alpha=1,beta=0;
 const auto dtype=fp32?CUDA_R_32F:CUDA_R_16BF;const auto compute=fp32&&tf32?CUBLAS_COMPUTE_32F_FAST_TF32:CUBLAS_COMPUTE_32F;
 int ldc=leading(out);
 if(out.stride(1)==1)
  ck(cublasGemmEx(handle,b.stride(1)==1?CUBLAS_OP_N:CUBLAS_OP_T,a.stride(1)==1?CUBLAS_OP_N:CUBLAS_OP_T,n,m,k,&alpha,b.data_ptr(),dtype,ldb,a.data_ptr(),dtype,lda,&beta,out.data_ptr(),output_float?CUDA_R_32F:CUDA_R_16BF,ldc,compute,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
 else
  ck(cublasGemmEx(handle,a.stride(1)==1?CUBLAS_OP_T:CUBLAS_OP_N,b.stride(1)==1?CUBLAS_OP_T:CUBLAS_OP_N,m,n,k,&alpha,a.data_ptr(),dtype,lda,b.data_ptr(),dtype,ldb,&beta,out.data_ptr(),output_float?CUDA_R_32F:CUDA_R_16BF,ldc,compute,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
 return out;
}


#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Context.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <map>
#include <memory>
struct OwnedBmmHandle { cublasHandle_t value; OwnedBmmHandle(){TORCH_CHECK(cublasCreate(&value)==CUBLAS_STATUS_SUCCESS,"TriMul cublasCreate");} ~OwnedBmmHandle(){cublasDestroy(value);} };
torch::Tensor owned_bmm(torch::Tensor a,torch::Tensor b,torch::Tensor out,bool tf32) {
 TORCH_CHECK(a.is_cuda()&&b.is_cuda()&&out.is_cuda()&&a.device()==b.device()&&a.device()==out.device()&&a.dim()==3&&b.dim()==3&&out.dim()==3,"TriMul owned BMM operands");
 TORCH_CHECK(a.size(0)==b.size(0)&&a.size(2)==b.size(1)&&out.size(0)==a.size(0)&&out.size(1)==a.size(1)&&out.size(2)==b.size(2),"TriMul owned BMM shapes");
 TORCH_CHECK(a.scalar_type()==b.scalar_type()&&(out.scalar_type()==a.scalar_type()||out.scalar_type()==at::kFloat)&&(a.scalar_type()==at::kBFloat16||a.scalar_type()==at::kFloat),"TriMul owned BMM dtype");
 c10::cuda::CUDAGuard guard(a.device());
 static thread_local std::map<int,std::unique_ptr<OwnedBmmHandle>> handles;
 auto& owned=handles[a.get_device()];if(!owned)owned=std::make_unique<OwnedBmmHandle>();
 auto ck=[](cublasStatus_t s){TORCH_CHECK(s==CUBLAS_STATUS_SUCCESS,"TriMul owned BMM status ",int(s));};
 auto handle=owned->value;auto stream=at::cuda::getCurrentCUDAStream();
 ck(cublasSetStream(handle,stream));ck(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));ck(cublasSetAtomicsMode(handle,CUBLAS_ATOMICS_NOT_ALLOWED));
 const bool fp32=a.scalar_type()==at::kFloat;
 auto math=fp32&&tf32?CUBLAS_TF32_TENSOR_OP_MATH:CUBLAS_DEFAULT_MATH;
 if(!fp32&&at::globalContext().allowBF16ReductionCuBLAS()!=at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK) math=static_cast<cublasMath_t>(math|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
 ck(cublasSetMathMode(handle,math));
 const size_t bytes=at::cuda::getChosenWorkspaceSize();auto workspace=torch::empty({int64_t(bytes)},a.options().dtype(at::kByte));ck(cublasSetWorkspace(handle,workspace.data_ptr(),bytes));
 auto leading=[](torch::Tensor t){bool row=t.stride(2)==1;TORCH_CHECK(row?t.stride(1)>=t.size(2):(t.stride(1)==1&&t.stride(2)>=t.size(1)),"TriMul BMM dense view");return int(row?t.stride(1):t.stride(2));};
 int lda=leading(a),ldb=leading(b),m=a.size(1),n=b.size(2),k=a.size(2);
 TORCH_CHECK(out.stride(2)==1&&out.stride(1)>=n,"TriMul BMM row output");
 float alpha=1,beta=0;auto dtype=fp32?CUDA_R_32F:CUDA_R_16BF;
 auto compute=fp32&&tf32?CUBLAS_COMPUTE_32F_FAST_TF32:CUBLAS_COMPUTE_32F;
 ck(cublasGemmStridedBatchedEx(handle,b.stride(2)==1?CUBLAS_OP_N:CUBLAS_OP_T,a.stride(2)==1?CUBLAS_OP_N:CUBLAS_OP_T,n,m,k,&alpha,b.data_ptr(),dtype,ldb,b.stride(0),a.data_ptr(),dtype,lda,a.stride(0),&beta,out.data_ptr(),out.scalar_type()==at::kFloat?CUDA_R_32F:CUDA_R_16BF,out.stride(1),out.stride(0),a.size(0),compute,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
 return out;
}
"""


@functools.lru_cache(maxsize=1)
def _owned_blas_extension():
    from torch.utils.cpp_extension import load_inline
    import hashlib

    cuda_home = ensure_cuda_home()
    if cuda_home is None or tuple(map(int, torch.version.cuda.split(".")[:2])) > (12, 9):
        raise RuntimeError("Hopper owned BLAS requires CUDA <= 12.9")
    root = Path(os.environ.get("MINIWORLD_ENGINE_JIT_ROOT", Path.home() / ".cache" / "miniworld_engine_jit"))
    build = root / "triattn_owned_blas" / hashlib.sha256(_OWNED_BLAS_CPP.encode()).hexdigest()[:16]
    build.mkdir(parents=True, exist_ok=True)
    return load_inline(
        name="triattn_h100_owned_blas", cpp_sources=_OWNED_BLAS_CPP,
        functions=["owned_mm", "owned_bmm"], with_cuda=True, build_directory=str(build),
        extra_include_paths=[str(Path(cuda_home) / "targets" / "x86_64-linux" / "include")],
        extra_cflags=["-O3"], extra_ldflags=["-lcublas"], verbose=False,
    )


def _mm(a, b, *, out_dtype=None):
    if not _owned_blas_enabled():
        return torch.mm(a, b, **({"out_dtype": out_dtype} if out_dtype is not None else {}))
    return _owned_blas_extension().owned_mm(
        a, b, torch.backends.cuda.matmul.allow_tf32, out_dtype == torch.float32,
    )


def _bmm(a, b, *, out_dtype=None):
    # Held-graph replay requires owned scratch for this reduction geometry only.
    # Other required training cells retain their original cuBLAS BMM calls.
    needs_owned = (a.shape[0] * a.shape[2] == 384 * 384 and b.shape[2] == 256)
    if not _owned_blas_enabled() or not needs_owned:
        return torch.bmm(a, b, **({"out_dtype": out_dtype} if out_dtype is not None else {}))
    out = torch.empty(
        (a.shape[0], a.shape[1], b.shape[2]), device=a.device,
        dtype=a.dtype if out_dtype is None else out_dtype,
    )
    return _owned_blas_extension().owned_bmm(a, b, out, torch.backends.cuda.matmul.allow_tf32)
