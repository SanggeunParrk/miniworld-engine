"""CUDA row kernels of the fused token DiT TRAINING block (``token_dit_train_rows.cu``), forward and backward.

Built on first use (``load_extension``), never at import. Entry points take the operands in the layouts
``integrations/token_dit_train.py`` keeps them: residual stream fp32 [M, d], every GEMM operand in the path's dtype (bf16
or fp32), per-column bias / weight gradient sums as per-block partial rows that the host sums.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch
from miniworld_engine import settings

_dir = Path(__file__).parent


@functools.lru_cache(maxsize=None)
def ext(d: int = 768, hd: int = 48, rpb: int = 8, *, empty_uniform: bool = False):
    """The row kernels for model width ``d`` and head dim ``hd`` (one build per layout: 768 x 48 / 32 / 64, 1024 x 64)."""
    from ..._nvcc import ensure_cuda_home, gencodes, host_flags, load_extension

    assert (d, hd) in ((768, 48), (768, 32), (768, 64), (1024, 64)), (d, hd)
    assert rpb in (1, 2, 4, 8), rpb
    ensure_cuda_home()
    hopper = torch.cuda.get_device_capability() == (9, 0)
    name = (
        "token_dit_train_rows_cuda"
        if (d, hd) == (768, 48)
        else f"token_dit_train_rows_cuda_{d}_{hd}"
    )
    if rpb != 8:
        name += f"_rpb{rpb}"
    if empty_uniform:
        name += "_empty_uniform"
    return load_extension(
        name=name,
        sources=[str(_dir / "token_dit_train_rows.cu")],
        extra_cuda_cflags=[
            *host_flags(),
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            f"-DTD_D={d}",
            f"-DTD_HD={hd}",
            f"-DTD_RPB={rpb}",
            *(["-DTOKEN_EMPTY_UNIFORM=1"] if empty_uniform else []),
            *(gencodes("90a") if hopper else gencodes("80", "90", "100", ptx=("100",))),
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


RPB = 8  # rows per block of the D-wide kernels (token_dit_train_rows.cu)


def partials(M: int, width: int, device, rpb: int = RPB) -> torch.Tensor:
    """[blocks, width] buffer for the per-block column sums of a kernel over M rows (every row of it is written)."""
    return torch.empty((M + rpb - 1) // rpb, width, device=device)


__all__ = ["RPB", "ext", "partials"]

# These host entry points reuse cuBLAS/cuBLASLt kernels with scratch allocated
# from PyTorch's active CUDA-graph pool.  Resetting a different graph must not
# invalidate any workspace pointer captured by this block.
_OWNED_GEMM_CPP = r"""#include <torch/extension.h>
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
#include <cublasLt.h>
torch::Tensor owned_addmm(torch::Tensor bias, torch::Tensor a, torch::Tensor b, bool tf32) {
 TORCH_CHECK(a.is_cuda()&&b.is_cuda()&&bias.is_cuda()&&a.device()==b.device()&&a.device()==bias.device()&&a.dim()==2&&b.dim()==2&&a.size(1)==b.size(0)&&a.scalar_type()==b.scalar_type()&&bias.scalar_type()==a.scalar_type()&&bias.dim()==1&&bias.numel()==b.size(1)&&bias.is_contiguous(),"Token owned addmm operands");
 TORCH_CHECK(a.scalar_type()==at::kFloat||a.scalar_type()==at::kBFloat16,"Token owned addmm dtype");
 c10::cuda::CUDAGuard guard(a.device());
 auto ck=[](cublasStatus_t s){TORCH_CHECK(s==CUBLAS_STATUS_SUCCESS,"Token owned Lt status ",int(s));};
 const bool fp32=a.scalar_type()==at::kFloat;
 const auto dtype=fp32?CUDA_R_32F:CUDA_R_16BF;
 const auto compute=fp32&&tf32?CUBLAS_COMPUTE_32F_FAST_TF32:CUBLAS_COMPUTE_32F;
 auto out=torch::empty({a.size(0),b.size(1)},a.options());
 const size_t bytes=at::cuda::getCUDABlasLtWorkspaceSize();
 auto workspace=torch::empty({int64_t(bytes)},a.options().dtype(at::kByte));
 struct Descriptors{
  cublasLtMatmulDesc_t op=nullptr;
  cublasLtMatrixLayout_t x=nullptr,y=nullptr,o=nullptr;
  cublasLtMatmulPreference_t pref=nullptr;
  ~Descriptors(){if(pref)cublasLtMatmulPreferenceDestroy(pref);if(o)cublasLtMatrixLayoutDestroy(o);if(y)cublasLtMatrixLayoutDestroy(y);if(x)cublasLtMatrixLayoutDestroy(x);if(op)cublasLtMatmulDescDestroy(op);}
 }d;
 ck(cublasLtMatmulDescCreate(&d.op,compute,CUDA_R_32F));
 auto layout=[&](torch::Tensor t,cublasLtMatrixLayout_t* p){
  bool row=t.stride(1)==1;
  TORCH_CHECK(row?t.stride(0)>=t.size(1):(t.stride(0)==1&&t.stride(1)>=t.size(0)),"Token owned Lt dense matrix");
  ck(cublasLtMatrixLayoutCreate(p,dtype,row?t.size(1):t.size(0),row?t.size(0):t.size(1),row?t.stride(0):t.stride(1)));
 };
 layout(b,&d.x);layout(a,&d.y);layout(out,&d.o);
 cublasOperation_t tx=b.stride(1)==1?CUBLAS_OP_N:CUBLAS_OP_T,ty=a.stride(1)==1?CUBLAS_OP_N:CUBLAS_OP_T;
 ck(cublasLtMatmulDescSetAttribute(d.op,CUBLASLT_MATMUL_DESC_TRANSA,&tx,sizeof(tx)));
 ck(cublasLtMatmulDescSetAttribute(d.op,CUBLASLT_MATMUL_DESC_TRANSB,&ty,sizeof(ty)));
 cublasLtEpilogue_t epilogue=CUBLASLT_EPILOGUE_BIAS;const void* bp=bias.data_ptr();
 ck(cublasLtMatmulDescSetAttribute(d.op,CUBLASLT_MATMUL_DESC_EPILOGUE,&epilogue,sizeof(epilogue)));
 ck(cublasLtMatmulDescSetAttribute(d.op,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bp,sizeof(bp)));
 ck(cublasLtMatmulDescSetAttribute(d.op,CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,&dtype,sizeof(dtype)));
 ck(cublasLtMatmulPreferenceCreate(&d.pref));
 ck(cublasLtMatmulPreferenceSetAttribute(d.pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes)));
 const void* ptrs[]={b.data_ptr(),a.data_ptr(),out.data_ptr(),out.data_ptr()};
 cublasLtMatmulPreferenceAttributes_t attrs[]={CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES};
 for(int i=0;i<4;++i){uint32_t alignment=256;while(reinterpret_cast<uintptr_t>(ptrs[i])%alignment)alignment>>=1;ck(cublasLtMatmulPreferenceSetAttribute(d.pref,attrs[i],&alignment,sizeof(alignment)));}
 cublasLtMatmulHeuristicResult_t algo{};int found=0;auto handle=at::cuda::getCurrentCUDABlasLtHandle();
 ck(cublasLtMatmulAlgoGetHeuristic(handle,d.op,d.x,d.y,d.o,d.o,d.pref,1,&algo,&found));
 TORCH_CHECK(found==1,"Token owned bias GEMM has no valid algorithm");
 float alpha=1,beta=0;
 ck(cublasLtMatmul(handle,d.op,&alpha,b.data_ptr(),d.x,a.data_ptr(),d.y,&beta,out.data_ptr(),d.o,out.data_ptr(),d.o,&algo.algo,workspace.data_ptr(),bytes,at::cuda::getCurrentCUDAStream()));
 return out;
}
"""


@functools.lru_cache(maxsize=1)
def owned_gemm_ext():
    from miniworld_engine.kernels._nvcc import load_inline_extension as load_inline
    from ..._nvcc import ensure_cuda_home
    import hashlib

    cuda_home = ensure_cuda_home()
    if cuda_home is None or tuple(
        map(int, (torch.version.cuda or "0.0").split(".")[:2])
    ) > (12, 9):
        raise RuntimeError("Token Hopper owned GEMMs require CUDA <= 12.9")
    root = settings.engine_jit_root()
    build = (
        root
        / "token_owned_gemm"
        / hashlib.sha256(_OWNED_GEMM_CPP.encode()).hexdigest()[:16]
    )
    build.mkdir(parents=True, exist_ok=True)
    return load_inline(
        name="token_hopper_owned_gemm",
        cpp_sources=_OWNED_GEMM_CPP,
        functions=["owned_mm", "owned_addmm"],
        with_cuda=True,
        build_directory=str(build),
        extra_include_paths=[
            str(Path(cuda_home) / "targets" / "x86_64-linux" / "include")
        ],
        extra_cflags=["-O3"],
        extra_ldflags=["-lcublas", "-lcublasLt"],
        verbose=False,
    )
