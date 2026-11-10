// cuBLAS batched GEMM with shared pair matrices, without expanding them in HBM.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

__global__ void pwa_batch_pointers(const __nv_bfloat16* w, const __nv_bfloat16* v, __nv_bfloat16* o,
                                  const void** wp, const void** vp, void** op, int batches, int splits, int l, int k) {
  int b=blockIdx.x*blockDim.x+threadIdx.x;
  if(b<batches) {
    wp[b]=w+(int64_t)(b/splits)*l*l;
    vp[b]=v+(int64_t)b*l*k;
    op[b]=o+(int64_t)b*l*k;
  }
}
torch::Tensor broadcast_bmm(torch::Tensor w, torch::Tensor v) {
  TORCH_CHECK(w.is_cuda() && v.is_cuda() && w.device()==v.device() && w.is_contiguous() && v.is_contiguous()
      && w.scalar_type()==at::kBFloat16 && v.scalar_type()==at::kBFloat16, "contiguous BF16 CUDA inputs");
  TORCH_CHECK(w.dim()==3 && v.dim()==3 && w.size(0)>0 && v.size(0)%w.size(0)==0 && w.size(1)==w.size(2) && v.size(1)==w.size(1), "shared batch shapes");
  c10::cuda::CUDAGuard guard(w.device());
  int b=v.size(0), l=v.size(1), k=v.size(2), splits=b/w.size(0);
  auto out=torch::empty_like(v);
  auto ptr=torch::empty({3*b},w.options().dtype(at::kLong));
  const void** wp=reinterpret_cast<const void**>(ptr.data_ptr());
  const void** vp=wp+b;
  void** op=reinterpret_cast<void**>(ptr.data_ptr())+2*b;
  pwa_batch_pointers<<<(b+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(
    reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
    reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),wp,vp,op,b,splits,l,k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  float alpha=1.f,beta=0.f;
  // Row-major O = W V is column-major O^T = V^T W^T.
  auto status=cublasGemmBatchedEx(at::cuda::getCurrentCUDABlasHandle(),CUBLAS_OP_N,CUBLAS_OP_N,
      k,l,l,&alpha,vp,CUDA_R_16BF,k,wp,CUDA_R_16BF,l,&beta,op,CUDA_R_16BF,k,b,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"PWA broadcast cuBLAS GEMM failed: ",int(status));
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("broadcast_bmm",&broadcast_bmm);}
