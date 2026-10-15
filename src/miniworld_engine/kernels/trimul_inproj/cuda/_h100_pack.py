"""Live weight packing without a nested torch.compile runtime invocation."""
from functools import lru_cache
from pathlib import Path
import triton
import triton.language as tl
import torch
from miniworld_engine import settings
from . import _h100_runtime as T
from miniworld_engine.kernels._compile import opaque

# Captured cuBLAS scratch belongs to each graph's allocator pool, rather than
# the framework's per-stream handle cache. Device arithmetic stays in cuBLAS.
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
 TORCH_CHECK(a.scalar_type()==b.scalar_type()&&out.scalar_type()==a.scalar_type()&&(a.scalar_type()==at::kBFloat16||a.scalar_type()==at::kFloat),"TriMul owned BMM dtype");
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
 ck(cublasGemmStridedBatchedEx(handle,b.stride(2)==1?CUBLAS_OP_N:CUBLAS_OP_T,a.stride(2)==1?CUBLAS_OP_N:CUBLAS_OP_T,n,m,k,&alpha,b.data_ptr(),dtype,ldb,b.stride(0),a.data_ptr(),dtype,lda,a.stride(0),&beta,out.data_ptr(),dtype,out.stride(1),out.stride(0),a.size(0),compute,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
 return out;
}
"""


@lru_cache(maxsize=1)
def _owned_blas_ext():
    import hashlib
    from torch.utils.cpp_extension import load_inline
    from miniworld_engine.kernels._nvcc import ensure_cuda_home

    cuda_home = ensure_cuda_home()
    if cuda_home is None or tuple(map(int, torch.version.cuda.split(".")[:2])) > (12, 9):
        raise RuntimeError("Hopper owned BLAS requires CUDA <= 12.9")
    root = settings.engine_jit_root()
    build = root / "trimul_owned_blas" / hashlib.sha256(_OWNED_BLAS_CPP.encode()).hexdigest()[:16]
    build.mkdir(parents=True, exist_ok=True)
    return load_inline(
        name="trimul_hopper_owned_blas", cpp_sources=_OWNED_BLAS_CPP,
        functions=["owned_mm", "owned_mm_out", "owned_bmm"], with_cuda=True,
        build_directory=str(build),
        extra_include_paths=[str(Path(cuda_home) / "targets" / "x86_64-linux" / "include")],
        extra_cflags=["-O3"], extra_ldflags=["-lcublas"], verbose=False,
    )


def owned_mm(a, b, *, out=None, out_dtype=None):
    """cuBLAS matrix product with graph-owned scratch and optional direct output."""
    if out_dtype not in (None, torch.float32):
        raise ValueError("owned MM supports the input dtype or FP32 output")
    ext = _owned_blas_ext()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    if out is not None:
        if out_dtype is not None and out.dtype != out_dtype:
            raise ValueError("owned MM output dtype does not match out_dtype")
        return ext.owned_mm_out(a, b, out, tf32)
    return ext.owned_mm(a, b, tf32, out_dtype == torch.float32)


def owned_bmm(a, b, *, out):
    """Strided cuBLAS batch product retaining graph-owned scratch."""
    return _owned_blas_ext().owned_bmm(a, b, out, torch.backends.cuda.matmul.allow_tf32)


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


def _pack_into_triton_fake(dst: torch.Tensor, wl: torch.Tensor, wlg: torch.Tensor, wr: torch.Tensor,
                           wrg: torch.Tensor) -> None:
    """Nothing: the op writes ``dst`` in place."""
    return None


@opaque(fake=_pack_into_triton_fake, name="trimul_h100_weight_pack_triton", mutates_args=("dst",))
def pack_into_triton(dst: torch.Tensor, wl: torch.Tensor, wlg: torch.Tensor, wr: torch.Tensor,
                     wrg: torch.Tensor) -> None:
    """Pack the four [R, D] leaves into ``dst`` [4R, D] in 32-row groups (the Triton fallback of ``pack_into_cuda``)."""
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
    vector = (T.preserve_training(d)
              and T.d256_glue_enabled()
              and all(w.dtype == torch.bfloat16 and w.is_contiguous() for w in quad))
    if (not vector and r % 32 == 0 and d % 32 == 0
            and all(w.dtype == torch.bfloat16 and w.stride(0) == 1 and w.stride(1) >= r for w in quad)):
        _aux_unit().kernel("mw_aux_pack_t").launch((4 * r // 32, d // 32, 1), (32, 8, 1), [p])
        return
    _launch("mw_aux_pack_vec" if vector else "mw_aux_pack",
            (((4*r*d//8 if vector else 4*r*d)+255)//256, 1, 1), [p])


def pack_into(dst, wl, wlg, wr, wrg):
    # Eager launches (every caller sits inside an opaque custom op) use the CUDA pack; the Triton op only serves compiled regions.
    if T.cuda_aux_enabled() or (T._GLUE_CUDA and not torch.compiler.is_compiling()):
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


def dense_native(value, dtype=None):
    """Small live matrix layout/cast glue; preserve exact bits when dtype agrees."""
    dtype = value.dtype if dtype is None else dtype
    if value.is_contiguous() and value.dtype == dtype:
        return value
    if value.ndim != 2 or value.dtype not in (torch.bfloat16, torch.float32) or dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("native dense copy requires a BF16/FP32 matrix")
    out = torch.empty(value.shape, device=value.device, dtype=dtype)
    _launch("mw_aux_dense_copy", ((value.numel()+255)//256,1,1),
            [value,out,*value.shape,*value.stride(),int(value.dtype==torch.float32),int(dtype==torch.float32)])
    return out


def dense_native_many(values, dtypes):
    """``dense_native`` for several matrices in one launch (the ones that already are dense keep their storage)."""
    out = list(values)
    todo = [i for i, (v, t) in enumerate(zip(values, dtypes, strict=True)) if not (v.is_contiguous() and v.dtype == t)]
    if not todo:
        return out
    if len(todo) == 1 or len(todo) > 6:
        for i in todo:
            out[i] = dense_native(values[i], dtypes[i])
        return out
    for i in todo:
        v, t = values[i], dtypes[i]
        if v.ndim != 2 or v.dtype not in (torch.bfloat16, torch.float32) or t not in (torch.bfloat16, torch.float32):
            raise ValueError("native dense copy requires a BF16/FP32 matrix")
        out[i] = torch.empty(v.shape, device=v.device, dtype=t)
    pad = lambda xs, fill=0: [xs[i] if i < len(todo) else fill for i in range(6)]
    srcs = [values[i] for i in todo]
    dsts = [out[i] for i in todo]
    p = T._launch_module().Struct([
        *pad(srcs, None), *pad(dsts, None), *pad([s.numel() for s in srcs]), *pad([s.shape[1] for s in srcs], 1),
        *pad([s.stride(0) for s in srcs]), *pad([s.stride(1) for s in srcs]),
        *pad([int(s.dtype == torch.float32) for s in srcs]), *pad([int(d.dtype == torch.float32) for d in dsts])])
    _launch("mw_aux_copy_n", ((max(s.numel() for s in srcs) + 255) // 256, len(todo), 1), [p])
    return out


def zero_native_(out):
    """Initialize an owned numeric workspace without an ATen fill kernel."""
    size = out.numel()*out.element_size()
    if size:
        _launch("mw_aux_zero_bytes", (min(512,(size+255)//256),1,1), [out,size])
    return out


def zeros_native(shape, *, device, dtype=torch.float32):
    return zero_native_(torch.empty(shape,device=device,dtype=dtype))


def dense_copy_into(dst, src):
    if (dst.ndim != 2 or src.shape != dst.shape or not dst.is_contiguous()
            or dst.dtype not in (torch.bfloat16, torch.float32)
            or src.dtype not in (torch.bfloat16, torch.float32)):
        raise ValueError("native dense copy requires matching dense matrices")
    _launch("mw_aux_dense_copy", ((src.numel()+255)//256,1,1),
            [src,dst,*src.shape,*src.stride(),int(src.dtype==torch.float32),int(dst.dtype==torch.float32)])


def finish_affine_partials(pg, pb, dg, db):
    blocks, width = pg.shape
    parallel = T.d256_glue_enabled()
    columns = 4 if parallel else 32
    _aux_unit().kernel("mw_aux_affine_cta_finish_parallel" if parallel
                       else "mw_aux_affine_cta_finish").launch(
        ((width+columns-1)//columns,1,1), (128,1,1), [pg,pb,dg,db,width,blocks])


def fused_wide_glue(width, length, bidirectional=False, selected=False):
    """Per-shape live weight layouts and paired deterministic affine finishing."""
    default = ((width, length, bidirectional) in
               ((384, 384, False), (384, 768, True)))
    return (length in (384, 768) and (width == 384 or (width == 512 and bidirectional))
            and (selected or T.preserve_wide_training(width))
            and default)

def fused_d256_glue(width, length, selected=False, *, bidirectional=False):
    return ((width == 256 and length == 384
             and (selected or T.preserve_d256_training())
             and T.d256_glue_enabled(selected))
            or (width == 256 and length == 768
                and (selected or T.preserve_d256_training())
                and T.d256_register_source_enabled(selected and bidirectional))
            or fused_wide_glue(width, length, bidirectional=bidirectional, selected=selected))

def pack_training_pair(dst, stack, weights):
    """Read live matrices once, retaining both forward and backward layouts."""
    wg, wl, wlg, wr, wrg = weights
    if any(w.dtype != torch.bfloat16 or not w.is_contiguous() for w in weights):
        raise ValueError("fused D256 packing requires dense BF16 working matrices")
    r, d = wl.shape
    _launch("mw_aux_pack_training_pair", (((wg.numel()+4*r*d)//8+255)//256,1,1),
            [dst,stack,wg,wl,wlg,wr,wrg,r,d])


def finish_affine_pair(first, second):
    """Both LayerNorm affine outputs share one fixed-order finishing launch."""
    args = [*first,*second,first[0].shape[1],first[0].shape[0],
            second[0].shape[1],second[0].shape[0]]
    count = (first[0].shape[1]+second[0].shape[1])//4
    _aux_unit().kernel("mw_aux_affine_pair_finish").launch((count,1,1),(128,1,1),args)


def native_training_grads(grads, leaves):
    result = list(grads)
    indices = [i for i in range(1,7) if result[i].dtype!=leaves[i].dtype or result[i].stride()!=leaves[i].stride()]
    if indices:
        copies = _training_gradient_copies([result[i] for i in indices],
                                           [leaves[i] for i in indices])
        for i,g in zip(indices,copies,strict=True):
            result[i]=g
    return result


def _training_cast_indices(weights):
    return [i for i,w in enumerate(weights)
            if w.dtype!=torch.bfloat16 or (i>=4 and not w.is_contiguous())]


def _training_cast_copies_fake(weights: list[torch.Tensor]) -> list[torch.Tensor]:
    """One fresh bf16 tensor per weight that is not already a BF16 alias (see ``_training_cast_indices``)."""
    return [torch.empty(weights[i].shape,device=weights[i].device,dtype=torch.bfloat16)
            for i in _training_cast_indices(weights)]


@opaque(fake=_training_cast_copies_fake,name="trimul_h100_preserved_weight_copies")
def _training_cast_copies(weights: list[torch.Tensor]) -> list[torch.Tensor]:
    """Fresh BF16 copies of the weights that need one; BF16 aliases are not outputs of the op."""
    with T.native_context(weights[0].device):
        out = cast_weights(weights)
        return [out[i] for i in _training_cast_indices(weights)]


def cast_training_weights(weights):
    # A custom op returns only fresh storage; BF16 aliases stay outside its outputs.
    out=list(weights)
    indices=_training_cast_indices(weights)
    if indices:
        for i,value in zip(indices,_training_cast_copies(weights),strict=True):
            out[i]=value
    return out


def _owned_training_mask_fake(pair,mask):
    """[L, L] bf16 mask; ``pair`` supplies only the shape and device."""
    return pair.new_empty((pair.shape[1],pair.shape[1]),dtype=torch.bfloat16)


@opaque(fake=_owned_training_mask_fake,name="trimul_h100_preserved_owned_mask")
def owned_training_mask(pair: torch.Tensor,mask: torch.Tensor | None) -> torch.Tensor:
    """The pair mask as an owned [L, L] BF16 tensor."""
    with T.native_context(pair.device):
        return owned_bf16_mask(mask,pair.shape[1],pair.device)


def normalize_into(dst, x, g, b, stats=None, *, warp=False):
    if not x.is_contiguous() or x.dtype != torch.bfloat16 or g.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("CUDA input LN requires contiguous BF16 input and FP32 affine")
    width, rows = x.shape[-1], x.numel()//x.shape[-1]
    if width not in (64, 128, 256, 384, 512):
        raise ValueError("Hopper CUDA normalization requires a supported pair width")
    if warp or rows >= 4096:    # one warp per row (8 rows per block); the one-block-per-row kernel only serves tiny inputs
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
    """Fresh gradients with each weight's shape, strides and dtype."""
    return [torch.empty_strided(w.shape, w.stride(), device=w.device, dtype=w.dtype) for w in weights]


@opaque(fake=_gradient_layout_fake, name="trimul_h100_parameter_gradient_layout")
def _gradient_layout(grads: list[torch.Tensor], weights: list[torch.Tensor]) -> list[torch.Tensor]:
    """Copy each dense gradient into its weight's (possibly transposed) layout."""
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


def _training_gradient_copies_fake(grads: list[torch.Tensor], weights: list[torch.Tensor]) -> list[torch.Tensor]:
    """Fresh gradients with each weight's shape, strides and dtype."""
    return _gradient_layout_fake(grads, weights)


@opaque(fake=_training_gradient_copies_fake,name="trimul_h100_preserved_gradient_copies")
def _training_gradient_copies(grads: list[torch.Tensor],weights: list[torch.Tensor]) -> list[torch.Tensor]:
    """Gradients densified to the weights' dtype, then laid out like the weights."""
    with T.native_context(grads[0].device):
        dense=dense_native_many(grads,[w.dtype for w in weights])
        return preserve_parameter_layout(dense,weights)


def stack_backward_weights(weights):
    """Small live [gate,left,left_gate,right,right_gate] matrix stack."""
    width = weights[0].shape[1]
    out = weights[0].new_empty((sum(w.shape[0] for w in weights), width))
    p = T._launch_module().Struct([
        *weights, out, *(w.shape[0] for w in weights),
        *(w.stride(0) for w in weights), *(w.stride(1) for w in weights), width])
    vector = (T.preserve_training(width)
              and T.d256_glue_enabled()
              and all(w.dtype == torch.bfloat16 and w.is_contiguous() for w in weights))
    count = max(w.numel() for w in weights) // (8 if vector else 1)
    _launch("mw_aux_stack_vec" if vector else "mw_aux_stack", ((count+255)//256, 5, 1), [p])
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
    """[L, D] bf16 constant scale; ``pair`` supplies only the shape and device."""
    return torch.empty((pair.shape[1], pair.shape[-1]), device=pair.device, dtype=torch.bfloat16)


@opaque(fake=_training_unit_scale_fake, name="trimul_h100_training_unit_scale")
def _training_unit_scale(pair: torch.Tensor) -> torch.Tensor:
    """The constant [L, D] BF16 unit-scale tensor of the training path (no derivative)."""
    out = torch.empty((pair.shape[1], pair.shape[-1]), device=pair.device, dtype=torch.bfloat16)
    _launch("mw_aux_unit_scale", ((out.numel()+255)//256, 1, 1), [out, out.numel()])
    return out


def training_unit_scale(pair):
    # The pair supplies shape/device only; the constant scale has no derivative.
    with torch.no_grad():
        return _training_unit_scale(pair)


def _fused_training_prepare_fake(leaves, mask):
    """(packed [4H, D], pair mask [L, L] fp32, gate [D, D], output [D, H]), shapes read from x and the first weight."""
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
