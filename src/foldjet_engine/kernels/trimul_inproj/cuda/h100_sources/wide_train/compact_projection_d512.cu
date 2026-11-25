// Frozen generated source: wide_compact_projection.CompactProjection.
// Research snapshot trimul_d256_bwd_sol90_stage2_20260923 (experiments/trimul_large_d_vast);
// flattened text of the qualified checkpoint chain. Edit only with a new qualification.
// SPDX-License-Identifier: Apache-2.0
#include "tmn_kernels.cuh"
using namespace tmn;using bf=__nv_bfloat16;
constexpr unsigned D=WIDTH,H=2*D;
struct Params{const uint64_t* patches;const unsigned* patch_count;unsigned patch_capacity;
 unsigned* seen;unsigned* row_count;unsigned* rows;unsigned capacity;const bf* norm;bf* compact;bf* projected;bf* proj;};
TMN_DEVI bool overflow(const Params& p){return *p.patch_count>p.patch_capacity||*p.row_count>p.capacity;}
extern "C" __global__ __launch_bounds__(256,4)
void mw_wide_compact_changed_rows(__grid_constant__ const Params p){
 unsigned count=*p.patch_count;if(count>p.patch_capacity)return;
 for(unsigned i=blockIdx.x*256+threadIdx.x;i<count;i+=gridDim.x*256){
  unsigned row=uint32_t(p.patches[i])/(H/2);
  if(atomicCAS(p.seen+row,0u,1u)==0){unsigned at=atomicAdd(p.row_count,1u);if(at<p.capacity)p.rows[at]=row;}
 }
}
extern "C" __global__ __launch_bounds__(256,4)
void mw_wide_gather_changed_norm(__grid_constant__ const Params p){
 if(overflow(p))return;unsigned count=*p.row_count;
 for(unsigned i=blockIdx.x*256+threadIdx.x;i<p.capacity*(H/2);i+=gridDim.x*256){
  unsigned row=i/(H/2),c=i%(H/2);
  reinterpret_cast<uint32_t*>(p.compact)[i]=row<count?reinterpret_cast<const uint32_t*>(p.norm)[size_t(p.rows[row])*(H/2)+c]:0;
 }
}
extern "C" __global__ __launch_bounds__(256,4)
void mw_wide_scatter_changed_proj(__grid_constant__ const Params p){
 if(overflow(p))return;unsigned count=*p.row_count;
 for(unsigned i=blockIdx.x*256+threadIdx.x;i<count*(D/2);i+=gridDim.x*256){
  unsigned row=i/(D/2),c=i%(D/2);
  reinterpret_cast<uint32_t*>(p.proj)[size_t(p.rows[row])*(D/2)+c]=reinterpret_cast<const uint32_t*>(p.projected)[i];
 }
}
