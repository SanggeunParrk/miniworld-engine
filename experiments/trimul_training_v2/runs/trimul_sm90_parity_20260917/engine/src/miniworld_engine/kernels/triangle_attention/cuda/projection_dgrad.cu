// Five projection input gradients, one FP32 accumulation and one BF16 store.
// One producer warp fills a two-stage TMA ring; one consumer warpgroup runs
// WGMMA. The narrow four-head bias projection is accumulated in the epilogue.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include "fa3_utils.h"
using namespace cute;
using Element = cutlass::bfloat16_t;

template<int M> struct Config {
  using A = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<Element>{}, Shape<Int<M>,_64>{}));
  using B = decltype(tile_to_shape(GMMA::Layout_MN_SW128_Atom<Element>{}, Shape<_128,_64>{}));
  using MMA = decltype(make_tiled_mma(GMMA::ss_op_selector<Element,Element,float,Shape<Int<M>,_128,_64>,GMMA::Major::K,GMMA::Major::MN>()));
  using TA = decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),make_shape(int32_t{},_128{}),make_stride(_128{},_1{})),A{},Shape<Int<M>,_64>{},_1{}));
  using TB = decltype(make_tma_copy(SM90_TMA_LOAD{},make_tensor(make_gmem_ptr((Element const*)nullptr),Shape<_128,_128>{},Stride<_1,_128>{}),B{},Shape<_128,_64>{},_1{}));
  struct Shared {
    array_aligned<Element,cosize_v<A>,1024> a[2];
    array_aligned<Element,cosize_v<B>,1024> b[2];
    cutlass::arch::ClusterTransactionBarrier full[2];
    cutlass::arch::ClusterBarrier empty[2];
  };
  struct Params { TA a[4]; TB b[4]; Element const *db,*wb; Element *dx; int rows; };
};

template<int M> __global__ __launch_bounds__(256)
void projection_dgrad_tma(CUTE_GRID_CONSTANT typename Config<M>::Params const p) {
  using C=Config<M>;
  extern __shared__ char storage[];
  auto &s=*reinterpret_cast<typename C::Shared*>(storage);
  int tid=threadIdx.x;
  if(tid==0) {
    for(int j=0;j<2;++j) { s.full[j].init(1);s.empty[j].init(1); }
    for(int j=0;j<4;++j) { prefetch_tma_descriptor(p.a[j].get_tma_descriptor());prefetch_tma_descriptor(p.b[j].get_tma_descriptor()); }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  if(tid<128) {
    if(tid==0) {
      for(int it=0;it<8;++it) {
        int stage=it%2,phase=(it/2)%2,proj=it/2,kt=it%2;
        s.empty[stage].wait(phase^1);
        s.full[stage].arrive_and_expect_tx((M*64+128*64)*sizeof(Element));
        auto ga=local_tile(p.a[proj].get_tma_tensor(make_shape(p.rows,_128{})),Shape<Int<M>,_64>{},make_coord(blockIdx.x,kt));
        auto gb=local_tile(p.b[proj].get_tma_tensor(Shape<_128,_128>{}),Shape<_128,_64>{},make_coord(0,kt));
        auto sa=make_tensor(make_smem_ptr(s.a[stage].data()),typename C::A{});
        auto sb=make_tensor(make_smem_ptr(s.b[stage].data()),typename C::B{});
        auto ca=p.a[proj].get_slice(_0{});
        auto cb=p.b[proj].get_slice(_0{});
        copy(p.a[proj].with(reinterpret_cast<uint64_t&>(s.full[stage])),ca.partition_S(ga),ca.partition_D(sa));
        copy(p.b[proj].with(reinterpret_cast<uint64_t&>(s.full[stage])),cb.partition_S(gb),cb.partition_D(sb));
      }
    }
    return;
  }
  typename C::MMA mma;
  auto thr=mma.get_slice(tid-128);
  auto acc=partition_fragment_C(mma,Shape<Int<M>,_128>{});
  clear(acc);
  for(int it=0;it<8;++it) {
    int stage=it%2,phase=(it/2)%2;
    s.full[stage].wait(phase);
    auto sa=make_tensor(make_smem_ptr(s.a[stage].data()),typename C::A{});
    auto sb=make_tensor(make_smem_ptr(s.b[stage].data()),typename C::B{});
    auto ra=thr.partition_fragment_A(sa);
    auto rb=thr.partition_fragment_B(sb);
    flash::gemm<false,0>(mma,ra,rb,acc);
    // All 128 consumers have completed their WGMMA before releasing this stage.
    cutlass::arch::NamedBarrier::sync(128,1);
    if(tid==128) s.empty[stage].arrive();
  }
  auto coord=thr.partition_C(make_identity_tensor(Shape<Int<M>,_128>{}));
  #pragma unroll
  for(int i=0;i<size(acc);++i) {
    int row=blockIdx.x*M+get<0>(coord(i)),col=get<1>(coord(i));
    if(row<p.rows) {
      float value=acc(i);
      #pragma unroll
      for(int h=0;h<4;++h) value=fmaf(float(p.db[row*4+h]),float(p.wb[h*128+col]),value);
      p.dx[row*128+col]=Element(value);
    }
  }
}

template<int M> torch::Tensor launch(std::vector<torch::Tensor> const& dy,std::vector<torch::Tensor> const& w) {
  using C=Config<M>;
  int rows=dy[0].numel()/128;
  auto dx=torch::empty({rows,128},dy[0].options());
  typename C::Params p;
  for(int j=0;j<4;++j) {
    auto a=make_tensor(make_gmem_ptr((Element const*)dy[j].data_ptr()),make_shape(rows,_128{}),make_stride(_128{},_1{}));
    auto b=make_tensor(make_gmem_ptr((Element const*)w[j].data_ptr()),Shape<_128,_128>{},Stride<_1,_128>{});
    p.a[j]=make_tma_copy(SM90_TMA_LOAD{},a,typename C::A{},Shape<Int<M>,_64>{},_1{});
    p.b[j]=make_tma_copy(SM90_TMA_LOAD{},b,typename C::B{},Shape<_128,_64>{},_1{});
  }
  p.db=(Element const*)dy[4].data_ptr();p.wb=(Element const*)w[4].data_ptr();p.dx=(Element*)dx.data_ptr();p.rows=rows;
  C10_CUDA_CHECK(cudaFuncSetAttribute(projection_dgrad_tma<M>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename C::Shared)));
  projection_dgrad_tma<M><<<(rows+M-1)/M,256,sizeof(typename C::Shared),at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dx;
}
torch::Tensor dgrad(std::vector<torch::Tensor> dy,std::vector<torch::Tensor> w,int tile) {
  TORCH_CHECK(dy.size()==5 && w.size()==5,"five gradients and weights required");
  c10::cuda::CUDAGuard guard(dy[0].device());
  for(int j=0;j<5;++j) {
    TORCH_CHECK(dy[j].is_cuda() && w[j].device()==dy[0].device() && dy[j].device()==dy[0].device(),"device mismatch");
    TORCH_CHECK(dy[j].scalar_type()==torch::kBFloat16 && w[j].scalar_type()==torch::kBFloat16,"BF16 required");
    TORCH_CHECK(dy[j].is_contiguous() && w[j].is_contiguous(),"contiguous inputs required");
    TORCH_CHECK(w[j].sizes()==torch::IntArrayRef({j<4?128:4,128}),"weight shape");
    TORCH_CHECK(dy[j].numel()==dy[0].numel()/128*(j<4?128:4),"gradient shape");
  }
  TORCH_CHECK(tile==64 || tile==128,"tile must be 64 or 128");
  return tile==64?launch<64>(dy,w):launch<128>(dy,w);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("dgrad",&dgrad);}
