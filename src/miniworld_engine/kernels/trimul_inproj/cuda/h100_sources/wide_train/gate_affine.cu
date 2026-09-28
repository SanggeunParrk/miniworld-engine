// Frozen generated source: early_both_affine_init.EarlyBothAffineGate.
// Research snapshot trimul_d256_bwd_sol90_stage2_20260923 (experiments/trimul_large_d_vast);
// flattened text of the qualified checkpoint chain. Edit only with a new qualification.
// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using namespace tmn::sm90;
constexpr int D=WIDTH,BAR=49152;
struct Params{CUtensorMap proj,gate,dy,ds,dp,dg;int M,N;float *affine_g,*affine_b,*input_g,*input_b;};
extern "C" __global__ __launch_bounds__(128,4)
void mw_early_both_affine_gate(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];auto bar=reinterpret_cast<uint64_t*>(sm+BAR);
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,total=(p.M/64)*(D/64);

 for(int c=blockIdx.x*128+tid;c<2*D;c+=gridDim.x*128){p.affine_g[c]=0;p.affine_b[c]=0;}
 for(int c=blockIdx.x*128+tid;c<D;c+=gridDim.x*128){p.input_g[c]=0;p.input_b[c]=0;}
 if(tid==0){mbar_init(bar,1);fence_barrier_init();}__syncthreads();
 for(int tile=blockIdx.x,it=0;tile<total;tile+=gridDim.x,++it){
  int row=(tile/(D/64))*64,col=(tile%(D/64))*64;
  if(tid==0){
   mbar_arrive_expect_tx(bar,32768);
   tma_load_2d(sm,&p.proj,bar,col,row);tma_load_2d(sm+8192,&p.gate,bar,col,row);
   tma_load_2d(sm+16384,&p.dy,bar,col,row);tma_load_2d(sm+24576,&p.ds,bar,col,row%p.N);
  }
  mbar_wait(bar,it&1);__syncthreads();
  #pragma unroll
  for(int q=0;q<4;++q){uint32_t dg[4];
   #pragma unroll
   for(int j=0;j<4;++j){
    int r=warp*16+lane/4+8*(j&1),c=q*16+2*(lane%4)+8*(j/2);uint32_t off=swz128(r,c*2);
    uint32_t pr=*reinterpret_cast<uint32_t*>(sm+off),ga=*reinterpret_cast<uint32_t*>(sm+8192+off);
    uint32_t dy=*reinterpret_cast<uint32_t*>(sm+16384+off),ds=*reinterpret_cast<uint32_t*>(sm+24576+off);
    float a=math::round_bf16(bf16lo(dy)*bf16lo(ds)),b=math::round_bf16(bf16hi(dy)*bf16hi(ds));
    float g0=math::sigmoid(bf16lo(ga)),g1=math::sigmoid(bf16hi(ga));
    *reinterpret_cast<uint32_t*>(sm+32768+off)=pack_bf16(a*g0,b*g1);
    dg[j]=pack_bf16(((a*bf16lo(pr))*g0)*(1-g0),((b*bf16hi(pr))*g1)*(1-g1));
   }
   int mat=lane/8;uint32_t dst=smem_u32(sm+40960)+swz128(q*16+lane%8+8*(mat>>1),(warp*16+8*(mat&1))*2);
   stsm_x4_t(dst,dg[0],dg[1],dg[2],dg[3]);
  }
  fence_proxy_async();__syncthreads();
  if(tid==0){
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.dp),"r"(smem_u32(sm+32768)),"r"(col),"r"(row):"memory");
   asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%2,%3}],[%1];"::"l"(&p.dg),"r"(smem_u32(sm+40960)),"r"(row),"r"(col):"memory");
   tma_store_commit();tma_store_wait_read<0>();
  }
  __syncthreads();
 }
 if(tid==0)tma_store_wait_all();
}
