// Correct shared all-masked rows to masked_fill's uniform softmax, including
// its zero score gradients. Ordinary masks leave the existing core unchanged.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace shared_mask_uniform80 {
template<class T> __device__ __forceinline__ float read(const T* p) { return float(*p); }
template<> __device__ __forceinline__ float read(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template<class T> __device__ __forceinline__ void write(T* p, float x) { *p = T(x); }
template<> __device__ __forceinline__ void write(__nv_bfloat16* p, float x) { *p = __float2bfloat16_rn(x); }
__device__ __forceinline__ float sigmoid(float x) {
  float t; asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(.5f*x));
  return fmaf(t,.5f,.5f);
}
template<class T, bool BWD>
__global__ void __launch_bounds__(256) uniform_kernel(
    const bool* mask, const T* value, const T* gate, T* out,
    float* lse, T* dq, T* dk, T* dv, float* db,
    int N, int LP, int H, int D, long vs, long gs, long os,
    long qs, long ks, long ds) {
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  bool valid=false;
  for (int i=tid;i<N;i+=256) valid |= mask[i];
  if (__syncthreads_or(valid)) return;
  const int a=blockIdx.y, c=blockIdx.x*8+warp;
  if (c<D) {
    float sum=0.f;
    for(int i=lane;i<N;i+=32) sum+=read(value+((long)a*LP+i)*vs+c);
    for(int delta=16;delta;delta>>=1) sum+=__shfl_xor_sync(0xffffffffu,sum,delta);
    const float mean=sum/float(N);
    for(int i=lane;i<LP;i+=32) {
      const long row=(long)a*LP+i;
      if constexpr(BWD) {
        write(dq+row*qs+c,0.f);write(dk+row*ks+c,0.f);
        write(dv+row*ds+c,i<N?mean:0.f);
      } else {
        const float y=gate?mean*sigmoid(read(gate+row*gs+c)):mean;
        write(out+row*os+c,y);
        if(lse && c%(D/H)==0) lse[((long)a*H+c/(D/H))*LP+i]=log2f(float(N));
      }
    }
  }
  if constexpr(BWD) {
    // The shared bias gradient is zero; write each element once, on sample 0.
    if(a==0) for(long i=(long)blockIdx.x*256+tid;i<(long)H*LP*LP;i+=(long)gridDim.x*256) db[i]=0.f;
  }
}
void check(const at::Tensor& mask,const at::Tensor& x,int N,int LP,int A,int H,int D) {
  TORCH_CHECK(mask.is_cuda() && mask.scalar_type()==at::kBool && mask.is_contiguous() && mask.numel()==N,"shared mask: bool [N]");
  TORCH_CHECK(x.is_cuda() && x.device()==mask.device() && (x.scalar_type()==at::kBFloat16 || x.scalar_type()==at::kFloat),"uniform mask: CUDA BF16/FP32");
  TORCH_CHECK(N>0 && LP>=N && A>0 && H>0 && D%H==0 && x.dim()==2 && x.size(0)==A*LP && x.size(1)==D && x.stride(1)==1,"uniform mask shape");
}
void fwd(at::Tensor mask,at::Tensor value,c10::optional<at::Tensor> gate,at::Tensor out,c10::optional<at::Tensor> lse,int N,int LP,int A,int H,int D) {
  check(mask,value,N,LP,A,H,D);check(mask,out,N,LP,A,H,D);
  TORCH_CHECK(out.scalar_type()==value.scalar_type(),"uniform output dtype");
  if(gate) {check(mask,*gate,N,LP,A,H,D);TORCH_CHECK(gate->scalar_type()==value.scalar_type(),"uniform gate dtype");}
  if(lse) TORCH_CHECK(lse->device()==value.device() && lse->scalar_type()==at::kFloat && lse->is_contiguous() && lse->numel()==A*H*LP,"uniform lse");
  const c10::cuda::CUDAGuard guard(value.device());const dim3 grid((D+7)/8,A);
#define LAUNCH(T) uniform_kernel<T,false><<<grid,256,0,at::cuda::getCurrentCUDAStream()>>>(mask.data_ptr<bool>(),(T*)value.data_ptr(),gate?(T*)gate->data_ptr():nullptr,(T*)out.data_ptr(),lse?lse->data_ptr<float>():nullptr,nullptr,nullptr,nullptr,nullptr,N,LP,H,D,value.stride(0),gate?gate->stride(0):0,out.stride(0),0,0,0)
  if(value.scalar_type()==at::kBFloat16) {LAUNCH(__nv_bfloat16);} else {LAUNCH(float);}
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void bwd(at::Tensor mask,at::Tensor dout,at::Tensor dq,at::Tensor dk,at::Tensor dv,at::Tensor db,int N,int LP,int A,int H,int D) {
  for(const auto* p:{&dout,&dq,&dk,&dv}) {check(mask,*p,N,LP,A,H,D);TORCH_CHECK(p->scalar_type()==dout.scalar_type(),"uniform gradient dtype");}
  TORCH_CHECK(db.device()==dout.device() && db.scalar_type()==at::kFloat && db.is_contiguous() && db.numel()==H*LP*LP,"uniform db");
  const c10::cuda::CUDAGuard guard(dout.device());const dim3 grid((D+7)/8,A);
#define LAUNCH(T) uniform_kernel<T,true><<<grid,256,0,at::cuda::getCurrentCUDAStream()>>>(mask.data_ptr<bool>(),(T*)dout.data_ptr(),nullptr,nullptr,nullptr,(T*)dq.data_ptr(),(T*)dk.data_ptr(),(T*)dv.data_ptr(),db.data_ptr<float>(),N,LP,H,D,dout.stride(0),0,0,dq.stride(0),dk.stride(0),dv.stride(0))
  if(dout.scalar_type()==at::kBFloat16) {LAUNCH(__nv_bfloat16);} else {LAUNCH(float);}
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace shared_mask_uniform80
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("fwd",&shared_mask_uniform80::fwd);m.def("bwd",&shared_mask_uniform80::bwd);
}
