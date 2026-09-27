// Four weight gradients share each Z tile through TMA/shared memory.
// Output ownership: one CTA = four projections, 64 output channels, one token split.
// FP32 split partials preserve the reduction precision; no atomic accumulation.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include "fa3_utils.h"
using namespace cute;
using Element=cutlass::bfloat16_t;

struct Config {
  using AL=decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<Element>{},Shape<_64,_64>{}));
  using BL=decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<Element>{},Shape<_128,_64>{}));
  using GS=Shape<_128,int32_t>;
  using ST=Stride<_1,_128>;
  using TA=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),GS{},ST{}),AL{},Shape<_64,_64>{},_1{}));
  using TB=decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),GS{},ST{}),BL{},Shape<_128,_64>{},_1{}));
  using MMA=decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<_64,_128,_64>,GMMA::Major::MN,GMMA::Major::MN>()));
  struct Shared {
    array_aligned<Element,4096,1024> a[2][4];
    array_aligned<Element,8192,1024> z[2];
    cutlass::arch::ClusterTransactionBarrier full[2];
    cutlass::arch::ClusterBarrier empty[2];
  };
  struct Params {TA a[4];TB z;float* partial;int rows,split,nsplit;};
};
template<class TMA,class G,class S>
__device__ __forceinline__ void tcopy(TMA const& tma,G const& g,S const& s,cutlass::arch::ClusterTransactionBarrier& ready) {
  auto c=tma.get_slice(_0{});
  copy(tma.with(reinterpret_cast<uint64_t&>(ready)),c.partition_S(g),c.partition_D(s));
}

__global__ __launch_bounds__(640,1) void grouped_wgrad(CUTE_GRID_CONSTANT Config::Params const p) {
  extern __shared__ char storage[];
  auto& s=*reinterpret_cast<Config::Shared*>(storage);
  int tid=threadIdx.x,wg=tid/128-1,lane=tid%128,mt=blockIdx.x,part=blockIdx.y;
  int first=part*(p.split/64),iters=min(p.split,p.rows-part*p.split)/64;
  if(tid==0) {
    for(int i=0;i<2;++i){s.full[i].init(1);s.empty[i].init(4);}
    for(int i=0;i<4;++i)prefetch_tma_descriptor(p.a[i].get_tma_descriptor());
    prefetch_tma_descriptor(p.z.get_tma_descriptor());
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  if(tid<128) {
    cutlass::arch::warpgroup_reg_dealloc<24>();
    if(tid==0) {
      auto zg=p.z.get_tma_tensor(make_shape(_128{},p.rows));
      for(int it=0;it<iters;++it) {
        int slot=it%2,phase=(it/2)%2;
        s.empty[slot].wait(phase^1);
        s.full[slot].arrive_and_expect_tx((4*4096+8192)*sizeof(Element));
        auto zz=local_tile(zg,Shape<_128,_64>{},make_coord(0,first+it));
        tcopy(p.z,zz,make_tensor(make_smem_ptr(s.z[slot].data()),Config::BL{}),s.full[slot]);
        #pragma unroll
        for(int proj=0;proj<4;++proj) {
          auto ag=p.a[proj].get_tma_tensor(make_shape(_128{},p.rows));
          auto aa=local_tile(ag,Shape<_64,_64>{},make_coord(mt,first+it));
          tcopy(p.a[proj],aa,make_tensor(make_smem_ptr(s.a[slot][proj].data()),Config::AL{}),s.full[slot]);
        }
      }
    }
    return;
  }
  cutlass::arch::warpgroup_reg_alloc<112>();
  Config::MMA mma;auto thr=mma.get_slice(lane);
  auto acc=partition_fragment_C(mma,Shape<_64,_128>{});clear(acc);
  auto coords=thr.partition_C(make_identity_tensor(Shape<_64,_128>{}));
  for(int it=0;it<iters;++it) {
    int slot=it%2,phase=(it/2)%2;s.full[slot].wait(phase);
    auto aa=make_tensor(make_smem_ptr(s.a[slot][wg].data()),Config::AL{});
    auto zz=make_tensor(make_smem_ptr(s.z[slot].data()),Config::BL{});
    auto ra=thr.partition_fragment_A(aa);
    auto rb=thr.partition_fragment_B(zz);
    flash::gemm<false,0>(mma,ra,rb,acc);
    cutlass::arch::NamedBarrier::sync(128,wg+1);
    if(lane==0)s.empty[slot].arrive();
  }
  #pragma unroll
  for(int i=0;i<size(acc);i+=2) {
    int m=mt*64+get<0>(coords(i)),n=get<1>(coords(i));
    int offset=((wg*p.nsplit+part)*128+m)*128+n;
    reinterpret_cast<float2*>(p.partial+offset)[0]=make_float2(acc(i),acc(i+1));
  }
}

__global__ void reduce_wgrad(float const* partial,Element* out,int splits) {
  int proj=blockIdx.y,idx=blockIdx.x*128+threadIdx.x;
  float4 sum=make_float4(0,0,0,0);
  for(int s=0;s<splits;++s) {
    float4 v=reinterpret_cast<float4 const*>(partial)[(proj*splits+s)*4096+idx];
    sum.x+=v.x;sum.y+=v.y;sum.z+=v.z;sum.w+=v.w;
  }
  int off=proj*16384+idx*4;
  uint32_t lo,hi;
  asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(lo):"f"(sum.y),"f"(sum.x));
  asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(hi):"f"(sum.w),"f"(sum.z));
  reinterpret_cast<uint2*>(out+off)[0]=make_uint2(lo,hi);
}

torch::Tensor backward(std::vector<torch::Tensor> dy,torch::Tensor z,int split) {
  TORCH_CHECK(dy.size()==4,"four wide projection gradients required");
  TORCH_CHECK(z.is_cuda() && z.scalar_type()==torch::kBFloat16 && z.is_contiguous() && z.dim()==2 && z.size(1)==128,"contiguous CUDA BF16 [tokens,128] Z required");
  int rows=z.size(0);TORCH_CHECK(rows>0 && rows%64==0 && split>=64 && split%64==0,"token/split multiples of64 required");
  c10::cuda::CUDAGuard guard(z.device());C10_CUDA_CHECK(cudaSetDevice(z.get_device()));
  for(auto const& d:dy)TORCH_CHECK(d.device()==z.device() && d.scalar_type()==z.scalar_type() && d.sizes()==z.sizes() && d.is_contiguous(),"gradient metadata mismatch");
  int splits=(rows+split-1)/split;
  auto partial=torch::empty({4,splits,128,128},z.options().dtype(torch::kFloat32));
  auto out=torch::empty({4,128,128},z.options());
  Config::Params p;p.rows=rows;p.split=split;p.nsplit=splits;p.partial=partial.data_ptr<float>();
  for(int i=0;i<4;++i) {
    auto ag=make_tensor(make_gmem_ptr((Element const*)dy[i].data_ptr()),make_shape(_128{},rows),Config::ST{});
    p.a[i]=make_tma_copy(SM90_TMA_LOAD{},ag,Config::AL{},Shape<_64,_64>{},_1{});
  }
  auto zg=make_tensor(make_gmem_ptr((Element const*)z.data_ptr()),make_shape(_128{},rows),Config::ST{});
  p.z=make_tma_copy(SM90_TMA_LOAD{},zg,Config::BL{},Shape<_128,_64>{},_1{});
  C10_CUDA_CHECK(cudaFuncSetAttribute(grouped_wgrad,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Config::Shared)));
  auto stream=at::cuda::getCurrentCUDAStream();
  grouped_wgrad<<<dim3(2,splits),640,sizeof(Config::Shared),stream>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();
  reduce_wgrad<<<dim3(32,4),128,0,stream>>>(partial.data_ptr<float>(),(Element*)out.data_ptr(),splits);C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("backward",&backward);m.def("smem",[](){return sizeof(Config::Shared);});}
